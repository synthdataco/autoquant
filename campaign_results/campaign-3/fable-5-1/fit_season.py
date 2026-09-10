"""Fit per-asset variance seasonality profiles -> agent/season.json.

All returns are labelled by the END of their interval and built exactly like the validator's
realized paths (closes sampled at 5-minute marks for 5-min returns, minute closes for 1-min):
  how   : 168 hour-of-week variance multipliers (mean 1), 5-min returns, robust (winsorised).
  mod5  : 288 five-minute-of-day multipliers relative to their own hour (weekdays only; mean 1
          within each hour) - captures the US open (13:30 UTC), close (20:00), half-hour marks.
  moh1  : 60 minute-of-hour multipliers relative to their own 5-minute block (1-min returns),
          for the crypto-1h format.
Profiles are shrunk toward the asset-group profile according to the data volume, and lightly
smoothed. Fitting uses the whole snapshot up to the cutoff. Run: uv run agent/fit_season.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
CUTOFF = "2026-08-29"
GROUPS = {
    "crypto": ["BTC", "ETH", "SOL", "XRP", "HYPE"],
    "equity": ["NVDAX", "GOOGLX", "TSLAX", "AAPLX", "SP500", "SPCX"],
    "commodity": ["XAU", "WTIOIL"],
}
# US equity market holidays inside the snapshot (excluded from profile fitting for
# non-crypto assets; the model handles them separately via modeling.HOLIDAYS).
HOLIDAYS = ["2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
            "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03"]


def load_close(asset: str) -> pd.Series:
    root = WORKSPACE / "market_data" / "prices" / asset / "1m"
    frames = [pd.read_parquet(p, columns=["timestamp", "close"]) for p in sorted(root.glob("date=*.parquet"))]
    df = pd.concat(frames)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = df.set_index("timestamp")["close"].astype(float).sort_index()
    return s[s.index < pd.Timestamp(CUTOFF, tz="UTC")]


def winsor(r: pd.Series, q: float = 0.995, keep_zeros: bool = False) -> pd.Series:
    """Winsorised returns. Exact zeros (no price change) are dropped unless keep_zeros: the model
    deseasonalises squared returns INCLUDING zeros, so a zero-inclusive profile is consistent
    with it (a zero-exclusive one inflates the profile where zeros are frequent, e.g. XAU weekends)."""
    r = r.dropna()
    if not keep_zeros:
        r = r[r != 0.0]
    cap = r[r != 0.0].abs().quantile(q)
    return r.clip(-cap, cap)


def smooth_circ(v: np.ndarray, w=(0.25, 0.5, 0.25)) -> np.ndarray:
    k = len(w) // 2
    out = np.zeros_like(v)
    for i in range(len(v)):
        idx = [(i + j) % len(v) for j in range(-k, k + 1)]
        vals = v[idx]
        ww = np.array(w) * ~np.isnan(vals)
        out[i] = np.nansum(vals * ww) / ww.sum() if ww.sum() > 0 else np.nan
    return out


def shrink(v: np.ndarray, c: np.ndarray, g: np.ndarray, n0: float) -> np.ndarray:
    lam = c / (c + n0)
    v = np.where(np.isnan(v), g, v)
    return lam * v + (1 - lam) * g


ZQ_GRID = np.linspace(0.0, 1.0, 2001)


def std_resid_quantiles(lr: pd.Series, hl_obs: int) -> list[float]:
    """Quantiles of symmetrised standardised residuals: returns divided by the causal EWMA
    (half-life hl_obs observations) of squared returns. Seasonality is left in on purpose: the
    model standardises by the same EWMA-type level x seasonal factor, so this is an approximation
    of the shock shape (fat tails) rather than an exact residual distribution."""
    lr = lr[lr != 0.0]
    x2 = (lr**2).ewm(halflife=hl_obs, min_periods=hl_obs).mean().shift(1)
    z = (lr / np.sqrt(x2)).dropna()
    z = z[np.isfinite(z)]
    z = pd.concat([z, -z]).clip(-10.0, 10.0)
    z = z / z.std()
    q = np.quantile(z.to_numpy(), ZQ_GRID)
    q[0], q[-1] = q[1], q[-2]  # never extrapolate to the sample extremes
    return q.round(4).tolist()


def wmean(x: pd.Series, w: pd.Series, by, index) -> tuple[np.ndarray, np.ndarray]:
    """Time-weighted group mean and effective counts (sum of weights)."""
    num = (x * w).groupby(by).sum().reindex(index)
    den = w.groupby(by).sum().reindex(index)
    return (num / den).to_numpy(), den.fillna(0).to_numpy()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--half-life-days", type=float, default=0.0, help="recency weighting (0 = none)")
    ap.add_argument("--no-smooth", action="store_true", help="no +-1h smoothing of the hour-of-week profile")
    ap.add_argument("--out", default="season.json")
    ap.add_argument("--cutoff", default=CUTOFF, help="fit only on data before this date")
    ap.add_argument("--keep-zeros", action="store_true", help="include exact-zero returns in the profiles")
    args = ap.parse_args()
    globals()["CUTOFF"] = args.cutoff
    cutoff_ts = pd.Timestamp(args.cutoff, tz="UTC")

    def weights(idx: pd.DatetimeIndex) -> pd.Series:
        if args.half_life_days <= 0:
            return pd.Series(1.0, index=idx)
        age = (cutoff_ts - idx).total_seconds() / 86400.0
        return pd.Series(np.power(0.5, age / args.half_life_days), index=idx)

    hol = pd.DatetimeIndex(pd.to_datetime(HOLIDAYS, utc=True)).date
    raw = {}
    for grp, assets in GROUPS.items():
        for a in assets:
            s = load_close(a)
            if grp != "crypto":
                s = s[~np.isin(s.index.date, hol)]
            m5 = s[s.index.minute % 5 == 0]
            r5 = winsor(np.log(m5).diff(), keep_zeros=args.keep_zeros)  # labelled by interval end
            r1 = winsor(np.log(s).diff(), 0.999, keep_zeros=args.keep_zeros)
            # hour-of-week
            how = (r5.index.dayofweek * 24 + r5.index.hour)
            w5 = weights(r5.index)
            v_how, c_how = wmean(r5**2, w5, how, range(168))
            # 5-min-of-day relative to hour (weekdays)
            wk = r5[r5.index.dayofweek < 5]
            mod = wk.index.hour * 60 + wk.index.minute
            v_mod, c_mod = wmean(wk**2, weights(wk.index), mod, range(0, 1440, 5))
            hour_mean = np.nanmean(v_mod.reshape(24, 12), axis=1)
            mod5 = (v_mod.reshape(24, 12) / hour_mean[:, None]).reshape(-1)
            # minute-of-hour relative to own 5-min block (1-min returns, all days)
            moh = r1.index.minute
            v_moh = (r1**2).groupby(moh).mean().reindex(range(60)).to_numpy()
            c_moh = (r1**2).groupby(moh).size().reindex(range(60)).fillna(0).to_numpy()
            # block b = minutes (5b-4 .. 5b] mod 60 : the 1-min returns composing the 5-min return ending at 5b
            block = ((np.arange(60) + 4) // 5) % 12
            block_mean = np.array([np.nanmean(v_moh[block == b]) for b in range(12)])
            moh1 = v_moh / block_mean[block]
            raw[a] = dict(how=(v_how / np.nanmean(v_how), c_how), mod5=(mod5, c_mod), moh1=(moh1, c_moh))
            raw[a]["zq"] = {"24h": std_resid_quantiles(np.log(m5).diff().dropna(), 288), "1h": std_resid_quantiles(np.log(s).diff().dropna(), 360)}
            print(f"{a}: {len(r5)} 5-min returns; open bar 13:30 x{mod5[13*12+6]:.1f} (rel to hour), moh1[0]={moh1[0]:.2f}")
    out: dict[str, dict] = {}
    for grp, assets in GROUPS.items():
        gp = {}
        for key in ("how", "mod5", "moh1"):
            stack = np.vstack([raw[a][key][0] for a in assets])
            gp[key] = np.nanmean(stack, axis=0)
        for a in assets:
            prof = {}
            v, c = raw[a]["how"]
            if args.no_smooth:
                p = shrink(v, c, gp["how"], n0=30 * 12)
            else:
                p = shrink(smooth_circ(v), c, smooth_circ(gp["how"]), n0=30 * 12)
            prof["how"] = (p / p.mean()).round(4).tolist()
            v, c = raw[a]["mod5"]
            p = shrink(v, c, gp["mod5"], n0=40)  # per-bin obs ~ number of weekdays; n0 ~ 40 days
            p = p.reshape(24, 12)
            p = p / p.mean(axis=1, keepdims=True)
            prof["mod5"] = p.reshape(-1).round(4).tolist()
            v, c = raw[a]["moh1"]
            p = shrink(v, c, gp["moh1"], n0=5000)
            block = ((np.arange(60) + 4) // 5) % 12
            for b in range(12):
                p[block == b] /= p[block == b].mean()
            prof["moh1"] = p.round(4).tolist()
            prof["zq"] = raw[a]["zq"]
            # long-run level: mean deseasonalised 5-min variance over the last 60 days before the cutoff
            s_ = load_close(a)
            s_ = s_[s_.index >= cutoff_ts - pd.Timedelta(days=60)]
            m5_ = s_[s_.index.minute % 5 == 0]
            r5_ = np.log(m5_).diff().dropna()
            r5_ = r5_[r5_ != 0.0]
            cap_ = r5_.abs().quantile(0.995)
            r5_ = r5_.clip(-cap_, cap_)
            how_ = np.asarray(prof["how"])[(r5_.index.dayofweek * 24 + r5_.index.hour)]
            prof["v5_longrun"] = float(np.mean(r5_.to_numpy() ** 2 / how_)) if len(r5_) > 500 else None
            out[a] = prof
            print(f"  {a:7s} how {min(prof['how']):.2f}-{max(prof['how']):.2f}  mod5 max {max(prof['mod5']):.2f}  moh1 {min(prof['moh1']):.2f}-{max(prof['moh1']):.2f}")
    (WORKSPACE / "agent" / args.out).write_text(json.dumps(out))
    print(f"wrote agent/{args.out}")


if __name__ == "__main__":
    main()
