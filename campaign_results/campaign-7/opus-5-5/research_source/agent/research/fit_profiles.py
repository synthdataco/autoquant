"""Fit per-asset weekly volatility profiles keyed on New York local time (minute-of-week).

For each asset, two shapes (mean 1 over the week):
  v5[b]: variance of the 5-min return starting at minute-of-week b (the 24h prompts' step)
  v1[b]: variance of the 1-min return starting at minute-of-week b (the 1h prompts' step)
Each squared return is first divided by a centred 7-day rolling mean of squared returns, so the
profile is a SHAPE and calm/volatile weeks weigh the same. Then circular kernel smoothing.
Usage: fit_profiles.py --cutoff 2026-08-01 --out agent/research/profiles_2026-08-01.npz
"""
import argparse, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS

W = 7 * 1440


def minute_of_week_ny(idx):
    ny = idx.tz_convert('America/New_York')
    return (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()


def smooth_circ(x, half):
    if half <= 0:
        return x
    k = np.concatenate([np.arange(1, half + 2), np.arange(half, 0, -1)]).astype(float)
    k /= k.sum()
    xp = np.concatenate([x[-half:], x, x[:half]])
    return np.convolve(xp, k, mode='valid')


MOH = False


def shape(sq, mow, valid, clip_mult, smooth_half):
    s = pd.Series(np.where(valid, sq, np.nan))
    lvl = s.rolling(7 * 1440, center=True, min_periods=1440).mean().to_numpy()
    z = s.to_numpy() / lvl
    ok = np.isfinite(z)
    z = np.minimum(z[ok], clip_mult)
    b = mow[ok]
    num = np.bincount(b, weights=z, minlength=W)
    cnt = np.bincount(b, minlength=W)
    prof = num / np.maximum(cnt, 1)
    # fallback for empty buckets
    prof[cnt == 0] = np.nanmean(prof[cnt > 0])
    raw = prof.copy()
    prof = smooth_circ(prof, smooth_half)
    if MOH:  # minute-of-hour pattern the smoothing blurs (e.g. the first minute of each hour)
        ratio = raw / prof
        moh = np.array([np.mean(ratio[np.arange(W) % 60 == m]) for m in range(60)])
        prof = prof * (moh / moh.mean())[np.arange(W) % 60]
    return prof / prof.mean(), cnt


def event_mask(asset, index, cal_path):
    """True on minutes (bar labels) inside an event window that concerns the asset."""
    import os
    ev = pd.read_parquet(cal_path or os.environ['SYNTH_BENCHMARK_CALENDAR'])
    ev['t'] = pd.to_datetime(ev['time_utc'], utc=True)
    mask = np.zeros(len(index), dtype=bool)
    for _, e in ev.iterrows():
        if not (e['scope'] == 'all' or asset in str(e['scope']).split(',')):
            continue
        if e['event'] == 'NYSE_CLOSED':
            d0 = e['t'].tz_convert('America/New_York').normalize()
            lo, hi = d0.tz_convert('UTC'), (d0 + pd.Timedelta(days=1)).tz_convert('UTC')
        else:
            span = pd.Timedelta(hours=24) if e['event'] == 'EARNINGS' else pd.Timedelta(minutes=120)
            lo = e['t'] - pd.Timedelta(minutes=1); hi = lo + span
        i0, i1 = index.searchsorted(lo), index.searchsorted(hi)
        mask[i0:i1] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', required=True, help='use data strictly before this day')
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-days', type=int, default=14, help='skip each market first N days (thin)')
    ap.add_argument('--smooth5', type=int, default=4)
    ap.add_argument('--smooth1', type=int, default=4)
    ap.add_argument('--clip', type=float, default=30.0)
    ap.add_argument('--exclude-events', action='store_true')
    ap.add_argument('--moh', action='store_true')
    args = ap.parse_args()
    global MOH
    MOH = args.moh
    out = {}
    for a in ALL_ASSETS:
        df = load_all(a, end=(pd.Timestamp(args.cutoff) - pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M'))
        df = df.iloc[args.skip_days * 1440:]
        lp = np.log(df['close'].to_numpy())
        mow = minute_of_week_ny(df.index)
        r1 = np.full(len(lp), np.nan); r1[:-1] = lp[1:] - lp[:-1]
        r5 = np.full(len(lp), np.nan); r5[:-5] = lp[5:] - lp[:-5]
        if args.exclude_events:
            em = event_mask(a, df.index, None)
            r1[em] = np.nan
            r5[em | np.roll(em, -4)] = np.nan  # a 5-min return touching an event window
        v1, c1 = shape(r1 ** 2, mow, np.isfinite(r1), args.clip, args.smooth1)
        v5, c5 = shape(r5 ** 2, mow, np.isfinite(r5), args.clip, args.smooth5)
        out[f'{a}_v1'] = v1.astype(np.float32)
        out[f'{a}_v5'] = v5.astype(np.float32)
        # robust shapes: mean ABSOLUTE return per bucket (normalised the same way), squared
        a1, _ = shape(np.abs(r1), mow, np.isfinite(r1), np.sqrt(args.clip), args.smooth1)
        a5, _ = shape(np.abs(r5), mow, np.isfinite(r5), np.sqrt(args.clip), args.smooth5)
        out[f'{a}_a1'] = (a1 ** 2 / np.mean(a1 ** 2)).astype(np.float32)
        out[f'{a}_a5'] = (a5 ** 2 / np.mean(a5 ** 2)).astype(np.float32)
        # share of exactly-zero 1-min / 5-min returns per minute of week (smoothed), recent-weighted:
        # only the last 120 days, the density of trading drifts over a market's life
        recent = df.index >= df.index[-1] - pd.Timedelta(days=120)
        for nm, rr, sm in (('z1', r1, args.smooth1), ('z5', r5, args.smooth5)):
            okz = np.isfinite(rr) & recent
            num = np.bincount(mow[okz], weights=(rr[okz] == 0).astype(float), minlength=W)
            cnt = np.bincount(mow[okz], minlength=W)
            share = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
            share[np.isnan(share)] = np.nanmean(share)
            out[f'{a}_{nm}'] = smooth_circ(share, max(sm, 7)).astype(np.float32)
        # variance-ratio diagnostics: Var(k-min) / (k * Var(1-min)), robust-ish (clip extremes)
        vr = {}
        for k in (2, 5, 15, 60):
            rk = lp[k:] - lp[:-k]
            rk = rk[np.isfinite(rk)]
            r1f = r1[np.isfinite(r1)]
            vr[k] = np.mean(np.minimum(rk ** 2, np.quantile(rk ** 2, 0.9995))) / (k * np.mean(np.minimum(r1f ** 2, np.quantile(r1f ** 2, 0.9995))))
        print(f"{a:7s} days={len(df)//1440:4d} minbucket5={c5.min():3d} VR2={vr[2]:.3f} VR5={vr[5]:.3f} VR15={vr[15]:.3f} VR60={vr[60]:.3f}"
              f" v5 max/min={v5.max()/v5.min():.1f}", flush=True)
    np.savez_compressed(args.out, **out)


if __name__ == '__main__':
    main()
