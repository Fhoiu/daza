from __future__ import annotations

import argparse
from pathlib import Path

from module.attack_module import run_attack_stage
from module.training_module import build_series_dict, load_dataset, load_model_from_checkpoint, split_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/power_consumption.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/lstm_power_normal.pt")
    parser.add_argument("--stolen-checkpoint-path", type=str, default="checkpoints/lstm_power_normal_attack.pt")
    parser.add_argument("--attack-log-path", type=str, default="logs/attack_normal.log")
    parser.add_argument("--train-users", type=int, default=400)
    parser.add_argument("--val-users", type=int, default=50)
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--plot-user-id", type=int, default=None)
    parser.add_argument("--attack-epochs", type=int, default=10)
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

    override_cfg = {}
    if args.mask_ratio is not None:
        override_cfg["mask_ratio"] = args.mask_ratio

    model, cfg = load_model_from_checkpoint(Path(args.checkpoint_path), override_cfg=override_cfg or None)

    run_attack_stage(
        model=model,
        series_dict=series_dict,
        test_ids=test_ids,
        cfg=cfg,
        log_path=Path(args.attack_log_path),
        plot_user_id=args.plot_user_id,
        stolen_checkpoint_path=Path(args.stolen_checkpoint_path),
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main(parse_args())

