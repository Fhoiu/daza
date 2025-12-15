from __future__ import annotations

import argparse
from pathlib import Path

from module.attack_module import run_attack_stage
from module.training_module import build_series_dict, load_dataset, load_model_from_checkpoint, split_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅执行疑似数据窃取攻击阶段（防御：对攻击者训练数据加噪）")
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv", help="输入 CSV 路径")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power_normal.pt", help="训练好的（受害者）正常模型 checkpoint")
    parser.add_argument("--stolen-checkpoint-path", type=str, default="checkpoints/lstm_power_normal_attack_noise.pt", help="攻击者窃取到的 LSTM 模型 checkpoint（加噪场景）")
    parser.add_argument("--attack-log-path", type=str, default="logs/attack_noise.log", help="攻击日志输出路径（加噪场景）")
    parser.add_argument("--train-users", type=int, default=400, help="训练用户数量（需与训练阶段保持一致）")
    parser.add_argument("--val-users", type=int, default=50, help="验证用户数量")
    parser.add_argument("--test-users", type=int, default=50, help="测试用户数量")
    parser.add_argument("--split-seed", type=int, default=2024, help="用户划分随机种子")
    parser.add_argument("--plot-user-id", type=int, default=None, help="指定绘图用户 ID")
    parser.add_argument("--attack-epochs", type=int, default=10, help="攻击者窃取模型的训练轮数（加噪场景）")
    return parser.parse_args()


def main(args: argparse.Namespace):
    csv_path = Path(args.data_path)
    df = load_dataset(csv_path)
    series_dict = build_series_dict(df)
    _, _, test_ids = split_users(
        total_users=len(series_dict),
        train=args.train_users,
        val=args.val_users,
        test=args.test_users,
        seed=args.split_seed,
    )

    model, cfg = load_model_from_checkpoint(Path(args.checkpoint_path), override_cfg=None)

    run_attack_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.attack_log_path),
        plot_user_id=args.plot_user_id,
        stolen_checkpoint_path=Path(args.stolen_checkpoint_path),
        add_label_noise=True,
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main(parse_args())


