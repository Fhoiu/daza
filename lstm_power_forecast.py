from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt                
import numpy as np                

from module.attack_module import AttackResult, run_attack_stage
from module.defense_module import DefenseResult, run_defense_stage
from module.training_module import (
    Config,
    build_series_dict,
    load_dataset,
    run_training_stage,
    split_users,
)


def plot_predictions(
    plot_truth: np.ndarray,
    plot_attack: np.ndarray,
    plot_defense: np.ndarray,
    plot_user_id: int,
    plot_path: Path,
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(plot_truth))
    plt.figure(figsize=(12, 4))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(steps, plot_truth, label="真实用电量")
    plt.plot(steps, plot_attack, label="[疑似]数据窃取攻击")
    plt.plot(steps, plot_defense, label="防御验证")
    plt.xlabel("时间步 (小时)")
    plt.ylabel("耗电量 (kWh)")
    plt.title(f"用户 {plot_user_id} 预测对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"预测对比图已保存至 {plot_path}")


def main(args: argparse.Namespace):
    csv_path = Path(args.data_path)
    df = load_dataset(csv_path)

    series_dict = build_series_dict(df)
    train_ids, val_ids, test_ids = split_users(
        total_users=len(series_dict), train=400, val=50, test=50, seed=2024
    )
    print(f"Train users: {len(train_ids)}, Val users: {len(val_ids)}, Test users: {len(test_ids)}")

    cfg = Config()
    checkpoint_path = Path(args.checkpoint_path)
    model = run_training_stage(series_dict, train_ids, val_ids, cfg, checkpoint_path, args.predict_only)

    if len(test_ids) == 0:
        print("无可用测试用户，跳过攻击与防御流程。")
        return

    plot_user_id: Optional[int]
    if args.plot_user_id is not None:
        plot_user_id = int(args.plot_user_id)
    else:
        plot_user_id = int(np.random.default_rng(2025).choice(test_ids))

    attack_result: AttackResult = run_attack_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.attack_log_path),
        plot_user_id=plot_user_id,
    )

    defense_result: DefenseResult = run_defense_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.defense_log_path),
        attack_log_path=Path(args.attack_log_path),
        plot_user_id=plot_user_id,
        detection_threshold=1.2,
    )

    if (
        attack_result.plot_truth is not None
        and attack_result.plot_attack is not None
        and defense_result.plot_defense is not None
    ):
        truth_for_plot = (
            defense_result.plot_truth
            if defense_result.plot_truth is not None
            else attack_result.plot_truth
        )
        plot_predictions(
            plot_truth=truth_for_plot,
            plot_attack=attack_result.plot_attack,
            plot_defense=defense_result.plot_defense,
            plot_user_id=plot_user_id,
            plot_path=Path(args.plot_path),
        )
    else:
        print("缺少完整的绘图数据，跳过绘图。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM forecast for synthetic power data.")
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv", help="数据 CSV 路径")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power.pt", help="模型 checkpoint 存储路径")
    parser.add_argument("--attack-log-path", type=str, default="logs/attack_metrics.log", help="[疑似]数据窃取攻击日志路径")
    parser.add_argument("--defense-log-path", type=str, default="logs/defense_metrics.log", help="防御验证日志路径")
    parser.add_argument("--predict-only", action="store_true", help="仅加载 checkpoint 直接预测，跳过训练")
    parser.add_argument("--plot-user-id", type=int, default=None, help="绘图用户 ID, 默认随机选择测试用户")
    parser.add_argument("--plot-path", type=str, default="outputs/sample_prediction.png", help="预测对比图保存路径")
    main(parser.parse_args())