from __future__ import annotations

import argparse
from pathlib import Path

from module.defense_module import run_defense_stage
from module.training_module import build_series_dict, load_dataset, load_model_from_checkpoint, split_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power_normal.pt")
    parser.add_argument("--attack-log-path", type=str, default="logs/attack_normal.log")
    parser.add_argument("--defense-log-path", type=str, default="logs/defense_normal.log")
    parser.add_argument("--train-users", type=int, default=400)
    parser.add_argument("--val-users", type=int, default=50)
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--detection-threshold", type=float, default=1.05)
    parser.add_argument("--plot-user-id", type=int, default=None)
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

    model, cfg = load_model_from_checkpoint(Path(args.checkpoint_path))

    run_defense_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.defense_log_path),
        attack_log_path=Path(args.attack_log_path),
        plot_user_id=args.plot_user_id,
        detection_threshold=args.detection_threshold,
    )


if __name__ == "__main__":
    main(parse_args())

