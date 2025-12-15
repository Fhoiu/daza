import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_power_profile(user_id: int, base_seed: int, hours: int) -> pd.DataFrame:
    
    rng = np.random.default_rng(base_seed + user_id)
    start_ts = datetime(2024, 1, 1)
    timestamps = np.array([start_ts + timedelta(hours=i) for i in range(hours)])

    base_load = rng.uniform(0.8, 2.5)                
    daily_phase = rng.uniform(0, np.pi * 2)
    weekly_phase = rng.uniform(0, np.pi * 2)

    hours_array = np.arange(hours)
    daily_pattern = 0.6 * np.sin(2 * np.pi * hours_array / 24 + daily_phase)
    weekly_pattern = 0.3 * np.sin(2 * np.pi * hours_array / (24 * 7) + weekly_phase)
    noise = rng.normal(0, 0.2, size=hours)

    consumption = (
        base_load
        + daily_pattern
        + weekly_pattern
        + noise
        + rng.uniform(0.1, 0.5) * (hours_array / hours)
    )
    consumption = np.clip(consumption, 0.1, None)

    return pd.DataFrame(
        {
            "user_id": user_id,
            "timestamp": timestamps,
            "consumption_kwh": consumption,
        }
    )


def main():
    total_users = 500
    days = 7
    hours_per_day = 24
    total_hours = days * hours_per_day
    base_seed = 2024

    frames = [
        generate_power_profile(user_id, base_seed, total_hours)
        for user_id in range(total_users)
    ]

    data = pd.concat(frames, ignore_index=True)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "power_consumption.csv"

    data.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()

