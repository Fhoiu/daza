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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power_normal.pt")
    parser.add_argument("--train-users", type=int, default=400)
    parser.add_argument("--val-users", type=int, default=50)
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--dp-use", action="store_true")
    parser.add_argument("--dp-clip", type=float, default=1.0)
    parser.add_argument("--dp-noise-mult", type=float, default=0.5)
    parser.add_argument("--predict-only", action="store_true")
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
        dp_use=args.dp_use,
        dp_clip=args.dp_clip,
        dp_noise_mult=args.dp_noise_mult,
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

