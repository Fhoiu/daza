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
class AttackResult:
    avg_metrics: Dict[str, float]
    plot_user_id: Optional[int]
    plot_truth: Optional[np.ndarray]
    plot_attack: Optional[np.ndarray]


def run_attack_stage(
    model: torch.nn.Module,
    series_dict: Dict[int, np.ndarray],
    test_ids: Sequence[int],
    cfg: Config,
    log_path: Path,
    plot_user_id: Optional[int] = None,
    show_metrics: bool = True,
    emit_log: bool = True,
) -> AttackResult:
    if len(test_ids) == 0:
        raise ValueError("测试用户列表为空，无法执行疑似数据窃取攻击流程。")

    log_path = Path(log_path)
    target_uid = plot_user_id
    if target_uid is None:
        target_uid = int(np.random.default_rng(2025).choice(test_ids))

    attack_metrics = []
    plot_truth = None
    plot_attack = None

    for uid in test_ids:
        series = series_dict[int(uid)]
        y_true, y_pred = rolling_forecast(
            model, series, cfg.seq_len, mask_ratio=cfg.mask_ratio, seed=int(uid)
        )
        attack_metrics.append(compute_metrics(y_true, y_pred))

        if plot_truth is None and int(uid) == int(target_uid):
            plot_truth = y_true
            plot_attack = y_pred

    avg_metrics = {k: float(np.mean([m[k] for m in attack_metrics])) for k in attack_metrics[0]}

    if show_metrics:
        print("\n=== [疑似]数据窃取攻击 ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

    if emit_log:
        _append_log(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "attack",
                "mask_ratio": cfg.mask_ratio,
                "metrics": avg_metrics,
            },
        )

    return AttackResult(avg_metrics=avg_metrics, plot_user_id=int(target_uid), plot_truth=plot_truth, plot_attack=plot_attack)


def _append_log(log_path: Path, payload: Dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")

