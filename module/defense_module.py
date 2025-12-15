from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np  # type: ignore
import torch  # type: ignore
from .training_module import Config, compute_metrics, rolling_forecast


@dataclass
class DefenseResult:
    avg_metrics: Dict[str, float]
    detection: Optional[bool]
    plot_user_id: Optional[int]
    plot_truth: Optional[np.ndarray]
    plot_defense: Optional[np.ndarray]


def run_defense_stage(
    model: torch.nn.Module,
    series_dict: Dict[int, np.ndarray],
    test_ids: Sequence[int],
    cfg: Config,
    log_path: Path,
    attack_log_path: Path,
    plot_user_id: Optional[int] = None,
    detection_threshold: float = 1.2,
    show_metrics: bool = True,
    emit_log: bool = True,
) -> DefenseResult:
    if len(test_ids) == 0:
        raise ValueError("测试用户列表为空，无法执行防御验证流程。")

    log_path = Path(log_path)
    attack_log_path = Path(attack_log_path)

    target_uid = plot_user_id
    if target_uid is None:
        target_uid = int(np.random.default_rng(2025).choice(test_ids))

    defense_metrics = []
    plot_truth = None
    plot_defense = None

    for uid in test_ids:
        series = series_dict[int(uid)]
        y_true, y_pred = rolling_forecast(
            model, series, cfg.seq_len, mask_ratio=0.0, seed=int(uid)
        )
        defense_metrics.append(compute_metrics(y_true, y_pred))

        if plot_truth is None and int(uid) == int(target_uid):
            plot_truth = y_true
            plot_defense = y_pred

    avg_metrics = {k: float(np.mean([m[k] for m in defense_metrics])) for k in defense_metrics[0]}

    if show_metrics:
        print("\n=== 防御验证 ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

    if emit_log:
        _append_log(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "defense",
                "metrics": avg_metrics,
                "threshold": detection_threshold,
            },
        )

    attack_metrics = _load_latest_metrics(attack_log_path)
    detection = None
    if attack_metrics is not None and "MSE" in attack_metrics:
        defense_mse = avg_metrics.get("MSE")
        if defense_mse is not None and defense_mse > 0:
            detection = attack_metrics["MSE"] > detection_threshold * defense_mse

    if show_metrics:
        if detection is True:
            print("\n检测到数据窃取\n")
        elif detection is False:
            print("\n未检测到数据窃取\n")
        else:
            print("\n无法判断是否存在数据窃取（缺少攻击日志）\n")

    return DefenseResult(
        avg_metrics=avg_metrics,
        detection=detection,
        plot_user_id=int(target_uid),
        plot_truth=plot_truth,
        plot_defense=plot_defense,
    )


def _append_log(log_path: Path, payload: Dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def _load_latest_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            return {k: float(v) for k, v in metrics.items()}
    return None

