from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    seq_len: int = 24
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.5
    dp_use: bool = False
    dp_clip: float = 1.0
    dp_noise_mult: float = 0.0


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def split_users(total_users: int, train: int, val: int, test: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    user_ids = np.arange(total_users)
    rng.shuffle(user_ids)
    assert train + val + test <= total_users
    return (
        user_ids[:train],
        user_ids[train : train + val],
        user_ids[train + val : train + val + test],
    )


def build_series_dict(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    return {
        int(uid): group["consumption_kwh"].to_numpy(dtype=np.float32)
        for uid, group in df.groupby("user_id", sort=True)
    }


class PowerDataset(Dataset):
    def __init__(self, series_dict: Dict[int, np.ndarray], user_ids: Sequence[int], seq_len: int):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for uid in user_ids:
            series = series_dict[int(uid)]
            for end_idx in range(seq_len, len(series)):
                x = series[end_idx - seq_len : end_idx]
                y = series[end_idx]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.fc(last).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100.0)
    return {"MSE": mse, "MAE": mae, "MAPE%": mape}


def impute_missing(window: np.ndarray) -> np.ndarray:
    filled = window.copy()
    if not np.isnan(filled).any():
        return filled

    last_val = np.nan
    for idx, val in enumerate(filled):
        if np.isnan(val):
            if not np.isnan(last_val):
                filled[idx] = last_val
        else:
            last_val = val

    if np.isnan(filled).any():
        mean_val = np.nanmean(filled)
        if np.isnan(mean_val):
            mean_val = 0.0
        filled[np.isnan(filled)] = mean_val
    return filled


def rolling_forecast(
    model: nn.Module,
    series: np.ndarray,
    seq_len: int,
    mask_ratio: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    preds = []
    trues = []
    history = series.copy()
    model.eval()

    with torch.no_grad():
        for t in range(seq_len, len(history)):
            window = history[t - seq_len : t].copy()
            if mask_ratio > 0:
                mask = rng.random(size=window.shape) < mask_ratio
                if mask.any():
                    window = window.astype(np.float32)
                    window[mask] = np.nan
                    window = impute_missing(window)

            inp = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred = model(inp).cpu().numpy().item()
            preds.append(pred)
            trues.append(history[t])
            history[t] = history[t]

    return np.array(trues), np.array(preds)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train = total_loss / len(train_loader.dataset)

        val_loss = evaluate_loss(model, val_loader, criterion)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}")

    final_state = best_state if best_state is not None else model.state_dict()
    model.load_state_dict(final_state)
    return final_state


def train_model_dp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    dp_clip: float = 1.0,
    dp_noise_mult: float = 0.0,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dp_clip)
            if dp_noise_mult > 0.0:
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    noise = torch.normal(
                        mean=0.0,
                        std=dp_noise_mult * dp_clip,
                        size=p.grad.shape,
                        device=p.grad.device,
                        dtype=p.grad.dtype,
                    )
                    p.grad.add_(noise)

            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train = total_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}")

    final_state = best_state if best_state is not None else model.state_dict()
    model.load_state_dict(final_state)
    return final_state


def run_training_stage(
    series_dict: Dict[int, np.ndarray],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    cfg: Config,
    checkpoint_path: Path,
    predict_only: bool,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    model = LSTMForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model.to(DEVICE)

    if predict_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict["model_state"])
        print(f"已从 {checkpoint_path} 加载 checkpoint，跳过训练。")
        return model

    train_dataset = PowerDataset(series_dict, train_ids, cfg.seq_len)
    val_dataset = PowerDataset(series_dict, val_ids, cfg.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    if cfg.dp_use:
        best_state = train_model_dp(
            model,
            train_loader,
            val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
            dp_clip=cfg.dp_clip,
            dp_noise_mult=cfg.dp_noise_mult,
        )
    else:
        best_state = train_model(model, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": vars(cfg)}, checkpoint_path)
    print(f"训练完成，已保存 checkpoint 至 {checkpoint_path}")
    model.load_state_dict(best_state)
    return model


def load_model_from_checkpoint(
    checkpoint_path: Path,
    override_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[nn.Module, Config]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    saved_cfg = checkpoint.get("config") or {}
    cfg_kwargs = {**saved_cfg}
    if override_cfg:
        cfg_kwargs.update(override_cfg)
    cfg = Config(**cfg_kwargs)

    model = LSTMForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    return model, cfg

