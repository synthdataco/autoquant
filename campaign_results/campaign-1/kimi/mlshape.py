"""ML shape model: predict per-step standardized-return quantiles q_tau(state_t, h).

Replaces the parametric sigma path + Student-t shape with a learned function of the
prompt-time state and the horizon h (in steps). Trained on the local snapshot only
(no external data). Pooled across assets; the target is standardized by the prompt-time
EWMA vol, so the network learns relative vol term structure, seasonality (via
hour(t+h) features), and horizon-dependent tail shape.

Usage:
  uv run agent/mlshape.py train --train-end 2026-07-13 --out agent/artifacts/mlshape_val.pt
  uv run agent/mlshape.py train --train-end 2026-07-28 --out agent/artifacts/mlshape.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WORKSPACE = Path(__file__).resolve().parent.parent

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "SPYX", "NVDAX", "GOOGLX",
          "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
CRYPTO = ("BTC", "ETH", "SOL", "XRP", "HYPE")
TAUS = np.array([0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995])
H_GRID = np.array([1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288])
EWMA_HLS_STEPS = [12, 48, 288, 864]  # 1h, 4h, 1d, 3d at 5-min steps


def load_prices(asset: str) -> pd.Series:
    parts = sorted(glob.glob(str(WORKSPACE / "market_data/pyth" / asset / "1m" / "date=*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in parts])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["close"].sort_index()


def grid_prices(prices: pd.Series, step_min: int) -> pd.Series:
    p = prices.dropna()
    grid = pd.date_range(p.index[0].floor(f"{step_min}min"), p.index[-1], freq=f"{step_min}min")
    return p.reindex(grid, method="ffill").dropna()


def ewma(x: np.ndarray, lam: float) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 - lam, adjust=False).mean().to_numpy()


def state_features(r: np.ndarray, hours: np.ndarray, i: int) -> np.ndarray | None:
    """Features from returns r[:i] (r[k] = return of step k+1). Mirrors what
    simulate() can compute from the context at prompt time."""
    if i < 900:
        return None
    r2 = r[:i] ** 2
    vols = []
    for hl in EWMA_HLS_STEPS:
        lam = 0.5 ** (1.0 / hl)
        vols.append(ewma(r2, lam))
    sig_1d = math.sqrt(max(vols[2][-1], 1e-18))
    if sig_1d <= 0:
        return None
    lr = [math.log(math.sqrt(max(vols[k][-1], 1e-18)) / sig_1d) for k in (0, 1, 3)]
    z = r[:i] / np.sqrt(np.maximum(vols[2], 1e-18))
    z = z[np.isfinite(z)]
    if z.size < 100:
        return None
    hour = hours[i - 1]
    return np.array([
        math.log(sig_1d * 1e4),                    # log sigma_now in bps
        lr[0], lr[1], lr[2],                       # vol term structure of the past
        float(np.mean(np.abs(z[-12:]))),
        float(np.mean(np.abs(z[-48:]))),
        float(np.max(np.abs(z[-288:]))),
        float(z[-1]),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
    ], dtype=np.float32)


def build_dataset(assets: list[str], train_end: str, step_min: int = 5,
                  t_stride: int = 12, max_h: int = 288):
    """Returns X (n, F+3), Y (n,) of standardized per-step returns at horizon h."""
    Xs, Ys = [], []
    tend = pd.Timestamp(train_end, tz="UTC")
    for asset in assets:
        p = load_prices(asset)
        gp = grid_prices(p, step_min)
        r = np.diff(np.log(gp.to_numpy(dtype=float)))
        hours = gp.index[1:].hour.to_numpy()
        times = gp.index[1:]
        is_crypto = 1.0 if asset in CRYPTO else 0.0
        n = len(r)
        for i in range(900, n - max_h - 1, t_stride):
            if times[i] >= tend:
                break
            f = state_features(r, hours, i)
            if f is None:
                continue
            sig_1d = math.exp(float(f[0])) / 1e4
            for h in H_GRID:
                if i + h >= n:
                    continue
                z_h = r[i + h - 1] / sig_1d  # per-step return h steps ahead, standardized
                if not np.isfinite(z_h):
                    continue
                hh = hours[i + h - 1]
                Xs.append(np.concatenate([f, [
                    math.log(h),
                    math.sin(2 * math.pi * hh / 24), math.cos(2 * math.pi * hh / 24),
                    is_crypto,
                ]]))
                Ys.append(z_h)
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)


class QuantileMLP(nn.Module):
    def __init__(self, n_features: int, n_quantiles: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_quantiles),
        )

    def forward(self, x):
        q = self.net(x)
        return torch.sort(q, dim=-1).values  # enforce monotonicity


def train(train_end: str, out: str, epochs: int = 12, t_stride: int = 12):
    X, Y = build_dataset(ASSETS, train_end, t_stride=t_stride)
    print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features")
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = QuantileMLP(X.shape[1], len(TAUS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xn, device=dev)
    Yt = torch.tensor(Y, device=dev)
    taus = torch.tensor(TAUS, device=dev)
    n = len(Yt)
    bs = 8192
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            q = model(Xt[idx])
            y = Yt[idx].unsqueeze(1)
            err = y - q
            loss = torch.mean(torch.maximum(taus * err, (taus - 1) * err))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"epoch {ep}: pinball {tot/n:.5f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "taus": TAUS.tolist(), "h_grid": H_GRID.tolist()}, out)
    meta = {"train_end": train_end, "n_samples": int(n), "features": X.shape[1]}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    train(args.train_end, args.out, args.epochs)
