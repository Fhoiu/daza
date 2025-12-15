from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np                
import torch                
from torch import nn                
from torch.utils.data import DataLoader                

from .training_module import (
    Config,
    compute_metrics,
    rolling_forecast,
    PowerDataset,
    LSTMForecaster,
    DEVICE,
)


@dataclass
class AttackResult:
    avg_metrics: Dict[str, float]
    plot_user_id: Optional[int]
    plot_truth: Optional[np.ndarray]
    plot_attack: Optional[np.ndarray]


def _train_togia_lstm(
    victim_model: nn.Module,
    series_dict: Dict[int, np.ndarray],
    user_ids: Sequence[int],
    cfg: Config,
    epochs: Optional[int] = None,
    add_label_noise: bool = False,
) -> nn.Module:
    
    if len(user_ids) == 0:
        raise ValueError("用户列表为空，无法训练攻击者 LSTM。")

                        
    victim_model.eval()

                           
    togia_model = LSTMForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(DEVICE)

                                         
    query_dataset = PowerDataset(series_dict, user_ids, cfg.seq_len)
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(togia_model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

                                         
    total_epochs = max(1, epochs if epochs is not None else 10)

    for epoch in range(1, total_epochs + 1):
        togia_model.train()
        running_loss = 0.0
        total_samples = 0
        for x, _ in query_loader:
            x = x.to(DEVICE)

                                
            with torch.no_grad():
                teacher_pred = victim_model(x)
                if add_label_noise:
                    noise_factor = torch.empty_like(teacher_pred).uniform_(0.5, 1.5)
                    teacher_pred = teacher_pred * noise_factor

            optimizer.zero_grad()
            student_pred = togia_model(x)
            loss = criterion(student_pred, teacher_pred)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = running_loss / max(1, total_samples)
        print(f"[Attack-Steal] Epoch {epoch:02d}/{total_epochs:02d} | distillation_loss={avg_loss:.4f}")

    return togia_model


def run_attack_stage(
    model: torch.nn.Module,
    series_dict: Dict[int, np.ndarray],
    test_ids: Sequence[int],
    cfg: Config,
    log_path: Path,
    plot_user_id: Optional[int] = None,
    show_metrics: bool = True,
    emit_log: bool = True,
    stolen_checkpoint_path: Optional[Path] = None,
    add_label_noise: bool = False,
    attack_epochs: Optional[int] = None,
) -> AttackResult:
    
    if len(test_ids) == 0:
        raise ValueError("测试用户列表为空，无法执行疑似数据窃取攻击流程。")

    log_path = Path(log_path)
    target_uid = plot_user_id
    if target_uid is None:
        target_uid = int(np.random.default_rng(2025).choice(test_ids))

                              
    print("\n=== [疑似]数据窃取攻击：开始训练攻击者自己的 LSTM（togia_model） ===")
    togia_model = _train_togia_lstm(
        victim_model=model,
        series_dict=series_dict,
        user_ids=test_ids,
        cfg=cfg,
        epochs=attack_epochs,
        add_label_noise=add_label_noise,
    )

                                                  
    if stolen_checkpoint_path is not None:
        stolen_checkpoint_path = Path(stolen_checkpoint_path)
        stolen_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": togia_model.state_dict(),
                "config": vars(cfg),
                "source": "attack_togia_lstm",
            },
            stolen_checkpoint_path,
        )
        if show_metrics:
            print(f"攻击者 LSTM checkpoint 已保存至: {stolen_checkpoint_path}")

                                     
    attack_metrics = []
    plot_truth = None
    plot_attack = None

    for uid in test_ids:
        series = series_dict[int(uid)]
                                                           
        y_true, y_pred = rolling_forecast(
            togia_model, series, cfg.seq_len, mask_ratio=0.0, seed=int(uid)
        )
        attack_metrics.append(compute_metrics(y_true, y_pred))

        if plot_truth is None and int(uid) == int(target_uid):
            plot_truth = y_true
            plot_attack = y_pred

    avg_metrics = {k: float(np.mean([m[k] for m in attack_metrics])) for k in attack_metrics[0]}

    if show_metrics:
        print("\n=== [疑似]数据窃取攻击（togia LSTM，完整输入） ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

    if emit_log:
        _append_log(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "attack",
                "attack_type": "togia_lstm_full_input",
                "metrics": avg_metrics,
            },
        )

    return AttackResult(
        avg_metrics=avg_metrics,
        plot_user_id=int(target_uid),
        plot_truth=plot_truth,
        plot_attack=plot_attack,
    )


def _append_log(log_path: Path, payload: Dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")

