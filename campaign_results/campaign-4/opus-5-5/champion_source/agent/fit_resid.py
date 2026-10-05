"""Standardised step returns for filtered historical simulation (FHS).

e = r_step / sqrt(Lhat * S_step), where S_step is the seasonal profile summed over the step's
minutes and Lhat a causal EWMA (half-life --hl minutes) of the deseasonalised 1-min variance,
lagged by one step. Residuals are rescaled to unit mean square. Stored per asset for 1-min
and 5-min steps. Data strictly before --cutoff.

Usage: uv run agent/fit_resid.py --cutoff 2026-08-01 --season agent/artifacts/seasonMF_2026-08-01.npz \
           --out agent/artifacts/resid_2026-08-01.npz
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_asset  # noqa: E402
from fit_season import ASSETS, week_bin  # noqa: E402


def resid(asset, cutoff, days, hl, prof):
    c = load_asset(asset)["close"]
    c = c[(c.index < cutoff) & (c.index >= cutoff - pd.Timedelta(days=days + 7))]
    lp = np.log(c.ffill())
    r = lp.diff()
    s = pd.Series(prof[week_bin(r.index)], index=r.index)
    z = r ** 2 / s
    L = z.ewm(halflife=hl, ignore_na=True, min_periods=hl).mean()
    out = {}
    for k in (1, 5):
        # step ending at minute T (inclusive) covers the k minutes (T-k, T]
        rk = r.rolling(k).sum().iloc[::k]
        sk = s.rolling(k).sum().iloc[::k]
        Lk = L.shift(k).reindex(rk.index)
        e = (rk / np.sqrt(Lk * sk)).dropna()
        e = e[e.index >= cutoff - pd.Timedelta(days=days)]
        e = e[np.isfinite(e)]
        e = e.clip(-30, 30)
        e = e / np.sqrt(np.mean(e ** 2))
        out[k] = e.to_numpy().astype(np.float32)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--hl", type=int, default=720)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    S = np.load(a.season)
    res = {}
    for x in ASSETS:
        o = resid(x, cutoff, a.days, a.hl, S[x])
        res[f"{x}_1"], res[f"{x}_5"] = o[1], o[5]
        e = o[5]
        print(f"{x:7s} n5 {len(e)} kurt5 {np.mean(e**4):.1f} zero5 {np.mean(e == 0):.3f} "
              f"q01 {np.quantile(e, .01):.2f} q50 {np.median(e):.3f} q99 {np.quantile(e, .99):.2f} kurt1 {np.mean(o[1]**4):.1f}")
    np.savez(a.out, **res)
