"""Fit per-asset intraday/weekly volatility profiles (New York local time, 15-min bins of the week).

uv run agent/fit_season.py --cutoff 2026-09-08 --out agent/artifacts/season_pre0908.json
Profile p[bin] multiplies a deseasonalised per-minute variance; mean over the week = 1.
Estimated from 1-min squared log returns (consecutive valid closes only, never interpolated),
each divided by a centred 3-day rolling mean of squared returns to strip the slow level.
Mon-Fri bins are shrunk halfway to the pooled weekday profile of their slot.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np, pandas as pd

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
ROOT = Path(__file__).resolve().parent.parent / "market_data" / "prices"
NB = 96  # 15-min slots per day


def load(a, start, end):
    fs = [f for f in sorted(glob.glob(str(ROOT / a / "1m" / "date=*.parquet"))) if start <= f[-18:-8] < end]
    df = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in fs])
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    return df.set_index("timestamp").close


MACRO = ("CPI", "NFP", "FOMC", "PPI", "GDP", "PCE", "RETAIL_SALES")


def event_mask(idx, asset):
    """Minutes within [-30, +240) of a scheduled macro release: kept out of the profile, the model adds them
    back as explicit event multipliers. EIA (weekly, same slot) is left in: the weekly profile carries it."""
    path = os.environ.get("SYNTH_BENCHMARK_CALENDAR")
    m = np.zeros(len(idx), dtype=bool)
    if not path or os.environ.get("NO_EVENT_MASK"):
        return m
    ev = pd.read_parquet(path)
    ev = ev[ev.event.isin(MACRO)]
    t = pd.DatetimeIndex(pd.to_datetime(ev.time_utc, utc=True)).tz_localize(None).sort_values().to_numpy().astype("datetime64[ns]")
    x = idx.tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[ns]")
    k = np.searchsorted(t, x + np.timedelta64(30, "m"), side="right") - 1
    ok = k >= 0
    dt = (x[ok] - t[k[ok]]) / np.timedelta64(1, "m")
    m[ok] = (dt >= -30) & (dt < 240)
    return m


def week_bin(idx_utc):
    ny = idx_utc.tz_convert("America/New_York")
    return ny.dayofweek * NB + (ny.hour * 60 + ny.minute) // 15


def fit(a, start, cutoff, shrink=0.5, wins=100.0, hl_days=0.0, step=1, l1=False):
    c = load(a, start, cutoff)
    r = np.log(c.ffill()).diff(step)  # no trade = unchanged last price (estimation only, never served)
    r2 = r**2
    lvl = r2.rolling(10080, center=True, min_periods=5000).mean()  # one full week: keeps the weekly shape in the profile
    keep = np.isfinite(r2) & np.isfinite(lvl) & (lvl > 0) & ~event_mask(r2.index, a)
    r2, lvl = r2[keep], lvl[keep]
    z = np.minimum(r2, wins * lvl)
    b = week_bin(z.index - pd.Timedelta(minutes=step // 2))
    age = (pd.Timestamp(cutoff, tz="UTC") - z.index).total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / hl_days) if hl_days > 0 else np.ones(len(z))
    if l1:  # robust: ratio of mean |r| to mean sqrt(level), squared
        num = pd.Series(np.sqrt(z.values) * w).groupby(b).sum()
        den = pd.Series(np.sqrt(lvl.values) * w).groupby(b).sum()
        prof = ((num / den) ** 2).reindex(range(7 * NB))
    else:
        num = pd.Series(z.values * w).groupby(b).sum()
        den = pd.Series(lvl.values * w).groupby(b).sum()
        prof = (num / den).reindex(range(7 * NB))
    cnt = pd.Series(z.values).groupby(b).size().reindex(range(7 * NB)).fillna(0)
    P = prof.values.reshape(7, NB).copy()
    wk = np.nanmean(P[:5], axis=0)
    P[:5] = shrink * wk + (1 - shrink) * np.where(np.isnan(P[:5]), wk, P[:5])
    P = np.where(np.isnan(P), np.nanmean(P), P)
    P = P / P.mean()
    # share of step returns that are exactly zero (both ends traded), same bins and recency weights
    rr = np.log(c).diff(step)
    rr = rr[np.isfinite(rr)]
    ag = (pd.Timestamp(cutoff, tz="UTC") - rr.index).total_seconds().to_numpy() / 86400
    wz = 0.5 ** (ag / hl_days) if hl_days > 0 else np.ones(len(rr))
    bz = week_bin(rr.index - pd.Timedelta(minutes=step // 2))
    zf = (pd.Series((rr.values == 0) * wz).groupby(bz).sum() / pd.Series(wz).groupby(bz).sum()).reindex(range(7 * NB))
    Z = zf.values.reshape(7, NB).copy()
    zwk = np.nanmean(Z[:5], axis=0)
    Z[:5] = shrink * zwk + (1 - shrink) * np.where(np.isnan(Z[:5]), zwk, Z[:5])
    Z = np.where(np.isnan(Z), np.nanmean(Z), Z)
    return P.ravel(), int(cnt.sum()), Z.ravel()


def fit_moh(a, start, cutoff, hl_days=0.0):
    """Minute-of-hour variance multiplier inside each 15-min slot (60 values, mean 1 per slot): activity
    bunches right after :00/:15/:30/:45 and fades before them. L1 ratio of sums against a centred 60-min level."""
    c = load(a, start, cutoff)
    r = np.log(c.ffill()).diff().abs()
    lvl = r.rolling(61, center=True, min_periods=30).mean()
    keep = np.isfinite(r) & np.isfinite(lvl) & (lvl > 0) & ~event_mask(r.index, a)
    r, lvl = r[keep], lvl[keep]
    age = (pd.Timestamp(cutoff, tz="UTC") - r.index).total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / hl_days) if hl_days > 0 else np.ones(len(r))
    mi = r.index.minute
    q = ((pd.Series(r.values * w).groupby(mi).sum() / pd.Series(lvl.values * w).groupby(mi).sum()) ** 2).reindex(range(60))
    q = q.fillna(q.mean()).values.reshape(4, 15)
    return (q / q.mean(axis=1, keepdims=True)).ravel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hl", type=float, default=0.0)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--l1", action="store_true")
    a = ap.parse_args()
    res = {"nb": NB, "tz": "America/New_York", "cutoff": a.cutoff, "hl": a.hl}
    for step in (1, 5):
        out, zout = {}, {}
        for x in ASSETS:
            p, n, zz = fit(x, a.start, a.cutoff, shrink=a.shrink, hl_days=a.hl, step=step, l1=a.l1)
            out[x] = [round(float(v), 4) for v in p]
            zout[x] = [round(float(v), 4) for v in zz]
            print(step, x, n, "wkday/wkend ratio", round(p[: 5 * NB].mean() / p[5 * NB:].mean(), 2), "max", round(p.max(), 2))
        res[f"profiles_{step}"] = out
        res[f"zero_{step}"] = zout
    res["moh"] = {x: [round(float(v), 4) for v in fit_moh(x, a.start, a.cutoff, a.hl)] for x in ASSETS}
    Path(a.out).write_text(json.dumps(res))
