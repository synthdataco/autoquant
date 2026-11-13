"""Seasonal variance profile (minute of the NY week) + scheduled-event multipliers, per asset.

profile[m]: variance multiplier of minute-of-week m (mean 1), from level-normalised squared 1-min
returns outside event windows: pooled weekdays at 1-min resolution (jump-preserving smoothing, with a
minute-of-hour sawtooth factored out), per-weekday slow ratios, own profiles for Sat/Sun.
events[type]: piecewise-constant variance multipliers by minutes since the event, pooled per asset group.

Usage: uv run python research/fit_profiles.py --fit-end 2026-07-24 --out scratch/params_oos
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
import core

MACRO = ["CPI", "NFP", "PPI", "RETAIL_SALES", "GDP", "PCE"]
EXCLUDE_MIN = {**{e: 90 for e in MACRO}, "FOMC": 150, "EIA_PETROLEUM": 60, "EARNINGS": 1440}
EDGES = {**{e: [0, 1, 2, 3, 5, 10, 15, 30, 60, 90] for e in MACRO},
         "FOMC": [0, 1, 2, 5, 10, 20, 30, 35, 45, 60, 90, 150],
         "EIA_PETROLEUM": [0, 1, 2, 5, 10, 20, 40, 60],
         "EARNINGS": [0, 5, 10, 15, 20, 25, 30, 35, 45, 60, 120, 240, 720, 1050, 1440]}
SCALE = "abs"; TWO_CLOCK = True; N_ITER = 12
GROUPS = {"crypto": list(core.CRYPTO), "equity": list(core.EQUITY), "XAU": ["XAU"], "WTIOIL": ["WTIOIL"]}

def wmean(val, w, key, size):
    ok = np.isfinite(val)
    num = np.bincount(key[ok], weights=(val * w)[ok], minlength=size)
    den = np.bincount(key[ok], weights=w[ok], minlength=size)
    return num / np.maximum(den, 1e-12), den

def bilateral(logv, n, hw, tau):
    """Circular jump-preserving smoother of a log profile; n = precision weights."""
    num = np.zeros_like(logv); den = np.zeros_like(logv)
    for k in range(-hw, hw + 1):
        nb, nn = np.roll(logv, -k), np.roll(n, -k)
        w = (1 - abs(k) / (hw + 1)) * np.exp(-0.5 * ((nb - logv) / tau) ** 2) * nn
        num += w * nb; den += w
    return num / np.maximum(den, 1e-12)

def triangular(v, n, hw):
    num = np.zeros_like(v); den = np.zeros_like(v)
    for k in range(-hw, hw + 1):
        w = (1 - abs(k) / (hw + 1)) * np.roll(n, -k)
        num += w * np.roll(v, -k); den += w
    return num / np.maximum(den, 1e-12)

def prepare(asset, fit_end, cal):
    df = load_asset(asset)
    df = df[df.index < fit_end]
    minutes = core.utc_minutes(df.index)
    logc = np.log(df["close"].to_numpy())
    r = np.full(len(df), np.nan); r[1:] = np.diff(logc)
    events = core.asset_events(asset, cal)
    wall = core.ny_wall_minutes(df.index)
    mow = core.minute_of_week(wall, core.closed_days_of(events) if asset in core.EQUITY else None)
    return dict(minutes=minutes, r=r, mow=mow, events=events, n=len(df), wall=wall)

def fit_profile(asset, d, hl_days):
    """Returns (NY minute-of-week profile [10080], UTC daily profile [2, 1440])."""
    r, mow, minutes = d["r"], d["mow"], d["minutes"]
    r2 = pd.Series(r * r)
    lvl = r2.rolling(10080, center=True, min_periods=2880).mean().to_numpy()
    u0 = np.where(lvl > 0, r2.to_numpy() / lvl, np.nan)
    for name, stamps in d["events"].items():      # drop event windows
        if name not in EXCLUDE_MIN: continue
        pos = np.searchsorted(stamps, minutes, side="right") - 1
        since = np.where(pos >= 0, minutes - stamps[np.maximum(pos, 0)], 10**9)
        u0[(since >= 0) & (since < EXCLUDE_MIN[name])] = np.nan
    w = np.exp2(-(minutes[-1] - minutes) / (1440.0 * hl_days))
    weekend_utc = ((minutes // 1440 + 3) % 7 >= 5).astype(np.int64)
    ukey = weekend_utc * 24 + (minutes // 60) % 24
    winter_share = np.mean((d["wall"] - minutes)[np.isfinite(u0)] == -300)
    B = np.ones((2, 1440))
    n_iter = N_ITER if (TWO_CLOCK and winter_share >= 0.15) else 1
    for it in range(n_iter):
        prof = fit_ny(u0 / B[weekend_utc, minutes % 1440], w, mow)
        if it == n_iter - 1: break
        ratio = np.sqrt(np.minimum(u0, 100.0) / prof[mow])        # abs scale
        bh, _ = wmean(ratio, w, ukey, 48)
        bh = bh.reshape(2, 24); bh = (bh / bh.mean(1, keepdims=True)) ** 2
        centers = np.arange(24) * 60 + 30
        for k in range(2):                                         # hourly -> minutes, circular linear interpolation
            B[k] = np.interp(np.arange(1440), np.concatenate([[centers[-1] - 1440], centers, [centers[0] + 1440]]),
                             np.concatenate([[bh[k, -1]], bh[k], [bh[k, 0]]]))
    return prof, B

def fit_ny(u, w, mow):
    dow, mod = mow // 1440, mow % 1440
    daytype = np.where(dow < 5, 0, dow - 4)        # 0 weekday, 1 Sat, 2 Sun
    key5 = daytype * 288 + mod // 5
    p1, _ = wmean(np.minimum(u, 100.0), w, key5, 3 * 288)
    p1 = np.maximum(p1, 1e-3)
    uc = np.minimum(u, 30.0 * p1[key5])
    if SCALE == "abs":                      # profile of (E|r|)^2: the scale CRPS cares about
        ua = np.sqrt(uc)
        pa, _ = wmean(ua, w, key5, 3 * 288); pa = np.maximum(pa, 1e-3)
        F, _ = wmean(ua / pa[key5], w, mod % 60, 60); F = (F / F.mean()) ** 2
        raw, n = wmean(ua, w, mow, 10080)
        raw = np.maximum(raw, 1e-3) ** 2; n = np.maximum(n, 1e-9)
    else:
        F, _ = wmean(uc / p1[key5], w, mod % 60, 60); F = F / F.mean()
        raw, n = wmean(uc, w, mow, 10080)
        raw = np.maximum(raw, 1e-4); n = np.maximum(n, 1e-9)
    FLOOR = 0.01
    rawd, nd = raw.reshape(7, 1440), n.reshape(7, 1440)
    Fm = F[np.arange(1440) % 60]
    wd = np.maximum((rawd[:5] * nd[:5]).sum(0) / nd[:5].sum(0), FLOOR)
    g = bilateral(np.log(wd / Fm), nd[:5].sum(0), 7, 0.35)
    prof_wd = np.exp(g) * Fm
    base = np.empty((7, 1440))
    base[:5] = prof_wd
    for dd in (5, 6):       # weekend: thin and zero-inflated -> 5-min means first, then jump-preserving smoothing
        n5 = nd[dd].reshape(288, 5).sum(1)
        r5 = (rawd[dd] / Fm * nd[dd]).reshape(288, 5).sum(1) / n5
        g5 = bilateral(np.log(np.maximum(r5, FLOOR)), n5, 4, 0.5)
        shape = np.ones((288, 5))
        for bkt in np.flatnonzero(g5 - np.roll(g5, 1) > np.log(3.0)):      # a reopening: keep its minute shape
            sh = np.maximum(rawd[dd, 5 * bkt:5 * bkt + 5] / Fm[5 * bkt:5 * bkt + 5], FLOOR)
            sh = np.maximum(sh / sh.mean(), 0.2)
            shape[bkt] = sh / sh.mean()
        base[dd] = np.repeat(np.exp(g5), 5) * shape.reshape(-1) * Fm
    raw = np.maximum(raw, FLOOR)
    ratio = triangular(np.log(raw / base.reshape(-1)), n, 90)
    ratio[5 * 1440:] = 0.0
    prof = base.reshape(-1) * np.exp(ratio)
    prof = prof / prof.mean()
    return np.maximum(prof, FLOOR)

def event_stats(asset, d, prof):
    """Per event type: [n_events, n_bins] of abs-based variance ratios vs the pre-event level."""
    r, mow, minutes = d["r"], d["mow"], d["minutes"]
    z = np.abs(r) / np.sqrt(prof)          # prof: full seasonal multiplier of every minute
    out = {}
    for name, stamps in d["events"].items():
        if name not in EDGES: continue
        edges = np.array(EDGES[name]); rows = []
        for t in stamps:
            i = np.searchsorted(minutes, t)
            if i >= len(minutes) or minutes[i] != t or i < 400 or i + edges[-1] >= len(minutes): continue
            pre = z[i - 360:i]; pre = pre[np.isfinite(pre)]
            if len(pre) < 100 or pre.mean() <= 0: continue
            row = []
            for a, b in zip(edges[:-1], edges[1:]):
                seg = z[i + a:i + b]; seg = seg[np.isfinite(seg)]
                row.append((seg.mean() / pre.mean()) if len(seg) else np.nan)
            rows.append(row)
        if rows: out[name] = np.array(rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", default="2026-11-09")
    ap.add_argument("--out", default="agent/params")
    ap.add_argument("--hl-days", type=float, default=150.0)
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--scale", default="abs", choices=["abs", "sq"])
    ap.add_argument("--one-clock", action="store_true")
    args = ap.parse_args()
    global SCALE, TWO_CLOCK
    SCALE = args.scale; TWO_CLOCK = not args.one_clock
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    cal = core.load_calendar()
    cal = cal[cal["time_utc"] < fit_end] if cal is not None else None   # no event after the fit window is used to FIT
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data, profs, stats, utcp = {}, {}, {}, {}
    for a in ALL_ASSETS:
        data[a] = prepare(a, fit_end, cal)
        profs[a], utcp[a] = fit_profile(a, data[a], args.hl_days)
        d = data[a]
        full = profs[a][d["mow"]] * utcp[a][((d["minutes"] // 1440 + 3) % 7 >= 5).astype(int), d["minutes"] % 1440]
        stats[a] = event_stats(a, d, full)
    tables = {}
    for gname, members in GROUPS.items():
        tab = {}
        for name in EDGES:
            rows = [stats[a][name] for a in members if name in stats[a]]
            if not rows: continue
            allr = np.concatenate(rows, 0)
            m = np.nanmean(allr, axis=0) ** 2          # (mean |z| ratio)^2: robust variance multiplier
            m = np.where(np.isfinite(m), m, 1.0)
            tab[name] = {"edges": EDGES[name], "mult": np.maximum(m, 1.0).round(3).tolist(), "n": int(len(allr))}
        for a in members: tables[a] = tab
    for a in ALL_ASSETS:
        payload = {"asset": a, "fit_end": args.fit_end, "profile": np.round(profs[a], 5).tolist(), "utc_profile": np.round(utcp[a], 4).tolist(),
                   "events": {} if args.no_events else {k: {"edges": v["edges"], "mult": v["mult"]} for k, v in tables[a].items()}}
        (out / f"profile_{a}.json").write_text(json.dumps(payload))
    np.set_printoptions(linewidth=220, precision=2, suppress=True)
    for gname, members in GROUPS.items():
        print("==", gname)
        for name, v in tables[members[0]].items():
            print(f"  {name:14s} n={v['n']:3d} mult={np.array(v['mult'])}")
    for a in ["BTC", "XAU", "WTIOIL"]:
        p = profs[a].reshape(7, 1440)
        print(a, "UTC weekday hourly sd:", np.sqrt(utcp[a][0].reshape(24, 60).mean(1)))
        print("   Sun 17:58-18:08 sd:", np.sqrt(p[6][1078:1088]))

if __name__ == "__main__":
    main()
