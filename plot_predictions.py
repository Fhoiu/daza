from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt                
import numpy as np                

from module.attack_module import run_attack_stage
from module.defense_module import run_defense_stage
from module.training_module import (
    build_series_dict,
    load_dataset,
    load_model_from_checkpoint,
    split_users,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power.pt")
    parser.add_argument("--attack-log-path", type=str, default="logs/attack_metrics.log")
    parser.add_argument("--defense-log-path", type=str, default="logs/defense_metrics.log")
    parser.add_argument("--plot-path", type=str, default="outputs/sample_prediction.png")
    parser.add_argument("--train-users", type=int, default=400)
    parser.add_argument("--val-users", type=int, default=50)
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--plot-user-id", type=int, default=None)
    return parser.parse_args()


def _plot_predictions(
    truth: np.ndarray,
    attack_pred: np.ndarray,
    defense_pred: np.ndarray,
    user_id: int,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(truth))
    plt.figure(figsize=(12, 4))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(steps, truth, label="真实用电量")
    plt.plot(steps, attack_pred, label="[疑似]数据窃取攻击")
    plt.plot(steps, defense_pred, label="防御验证")
    plt.xlabel("时间步 (小时)")
    plt.ylabel("耗电量 (kWh)")
    plt.title(f"用户 {user_id} 预测对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"预测对比图已保存至 {output_path}")


def main(args: argparse.Namespace):
    csv_path = Path(args.data_path)
    df = load_dataset(csv_path)
    series_dict = build_series_dict(df)

    train_ids, val_ids, test_ids = split_users(
        total_users=len(series_dict),
        train=args.train_users,
        val=args.val_users,
        test=args.test_users,
        seed=args.split_seed,
    )

    if len(test_ids) == 0:
        raise RuntimeError("没有可用的测试用户，无法生成对比图。")

    override_cfg = {}
    if args.mask_ratio is not None:
        override_cfg["mask_ratio"] = args.mask_ratio

    model, cfg = load_model_from_checkpoint(Path(args.checkpoint_path), override_cfg=override_cfg or None)

    user_id: Optional[int] = args.plot_user_id
    if user_id is None:
        user_id = int(np.random.default_rng(2025).choice(test_ids))

    attack_result = run_attack_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.attack_log_path),
        plot_user_id=user_id,
        show_metrics=False,
        emit_log=False,
    )

    defense_result = run_defense_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.defense_log_path),
        attack_log_path=Path(args.attack_log_path),
        plot_user_id=user_id,
        detection_threshold=1.2,
        show_metrics=False,
        emit_log=False,
    )

    truth = (
        defense_result.plot_truth
        if defense_result.plot_truth is not None
        else attack_result.plot_truth
    )
    if (
        truth is None
        or attack_result.plot_attack is None
        or defense_result.plot_defense is None
        or user_id is None
    ):
        raise RuntimeError("缺少绘图所需的序列，请确保 attack/defense 阶段生成了相应数据。")

    _plot_predictions(
        truth=truth,
        attack_pred=attack_result.plot_attack,
        defense_pred=defense_result.plot_defense,
        user_id=user_id,
        output_path=Path(args.plot_path),
    )


if __name__ == "__main__":
    main(parse_args())

