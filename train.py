from __future__ import annotations

import argparse
from pathlib import Path

from module.training_module import (
    Config,
    build_series_dict,
    load_dataset,
    run_training_stage,
    split_users,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅执行 LSTM 用电预测训练流程")
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv", help="输入 CSV 路径")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power.pt", help="checkpoint 输出路径")
    parser.add_argument("--train-users", type=int, default=400, help="训练用户数量")
    parser.add_argument("--val-users", type=int, default=50, help="验证用户数量")
    parser.add_argument("--test-users", type=int, default=50, help="测试用户数量（仅用于划分，训练阶段不会用到）")
    parser.add_argument("--split-seed", type=int, default=2024, help="用户划分随机种子")
    parser.add_argument("--seq-len", type=int, default=24, help="序列长度")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden-size", type=int, default=48, help="LSTM 隐层维度")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM 层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="攻击阶段默认 mask 比例（随 checkpoint 保存）")
    parser.add_argument("--predict-only", action="store_true", help="仅加载现有 checkpoint，跳过训练")
    return parser.parse_args()


def main(args: argparse.Namespace):
    csv_path = Path(args.data_path)
    df = load_dataset(csv_path)
    series_dict = build_series_dict(df)
    train_ids, val_ids, _ = split_users(
        total_users=len(series_dict),
        train=args.train_users,
        val=args.val_users,
        test=args.test_users,
        seed=args.split_seed,
    )

    cfg = Config(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
    )

    run_training_stage(
        series_dict=series_dict,
        train_ids=train_ids,
        val_ids=val_ids,
        cfg=cfg,
        checkpoint_path=Path(args.checkpoint_path),
        predict_only=args.predict_only,
    )


if __name__ == "__main__":
    main(parse_args())

