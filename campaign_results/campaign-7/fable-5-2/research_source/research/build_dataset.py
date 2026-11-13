"""Training/validation sets for research/train.py, built with agent/core.py's own feature code.

One npz per (asset, format): features of each prompt's context (cut exactly as the scored evaluation
cuts it), the seasonal/session arrays of its future minutes and its realised log path.
Usage: uv run python research/build_dataset.py --params scratch/params_oos --out scratch/ds_oos [--assets BTC,ETH]
"""
import sys, argparse, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from multiprocessing import Pool
from common import *
import core

CONTEXT_DAYS = 30
CFG = {"24h": dict(tl=86400, inc=5, lead=3, every=60), "1h": dict(tl=3600, inc=1, lead=2, every=12)}

def build(task):
    asset, tag, params_dir, out_dir, seed = task
    cfg = CFG[tag]
    t_start = time.time()
    df = load_asset(asset)
    params = core.load_params(asset, cfg["tl"], Path(params_dir))
    events = core.asset_events(asset, core.load_calendar())
    minutes = core.utc_minutes(df.index)
    close = df["close"].to_numpy(dtype=float)
    good = np.isfinite(close) & (close > 0)
    logc = np.log(np.where(good, close, np.nan))
    r = np.full(len(df), np.nan); r[1:] = np.diff(logc)
    hi, lo = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    okb = np.isfinite(hi) & np.isfinite(lo) & (lo > 0) & (hi >= lo)
    pk = np.where(okb, np.log(np.where(okb, hi, 1.0) / np.where(okb, lo, 1.0)) ** 2 / (4.0 * np.log(2.0)), np.nan)
    seas, sess = core.seasonal_of(minutes, asset, params, events)
    T = cfg["tl"] // 60; inc = cfg["inc"]; H = T // inc
    first = int(np.flatnonzero(good)[0]) + 3 * 1440
    last = len(df) - T - 1
    rng = np.random.default_rng(seed)
    starts = np.arange(first, last, cfg["every"])
    starts = starts + rng.integers(0, cfg["every"], size=len(starts))
    starts = starts[starts < last]
    X, XS, XZ, XSL, CUR, XE, SF, SS, LR, T0, P0 = [], [], [], [], [], [], [], [], [], [], []
    goodidx = np.flatnonzero(good)
    for i0 in starts:
        real = logc[i0 : i0 + T + 1 : inc]
        if not np.isfinite(real).all():
            continue
        end = i0 - cfg["lead"]                      # last label the context holds
        a = max(0, i0 - CONTEXT_DAYS * 1440)
        j = goodidx[np.searchsorted(goodidx, end, side="right") - 1]   # last real close at or before `end`
        if j < a + 60:
            continue
        rr = r[a : j + 1].copy(); rr[0] = np.nan
        x, xs, info = core.features_from_arrays(rr, pk[a : j + 1], seas[a : j + 1], sess[a : j + 1])
        X.append(x); XS.append(xs); XZ.append(info['xz']); XSL.append(info['xsl']); CUR.append(info['cur']); XE.append(info['xe'])
        SF.append(seas[i0 + 1 : i0 + T + 1].astype(np.float32)); SS.append(sess[i0 + 1 : i0 + T + 1].astype(np.int8))
        LR.append(real - logc[j]); T0.append(minutes[i0]); P0.append(close[j])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{asset}_{tag}.npz", x=np.array(X), xs=np.array(XS), xz=np.array(XZ), xsl=np.array(XSL), cur=np.array(CUR), xe=np.array(XE), seas=np.array(SF), sess=np.array(SS),
             lr=np.array(LR), t0=np.array(T0), p0=np.array(P0))
    return f"{asset} {tag}: {len(X)} prompts in {time.time() - t_start:.0f}s"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="all"); ap.add_argument("--tags", default="24h,1h")
    ap.add_argument("--jobs", type=int, default=13); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    assets = ALL_ASSETS if args.assets == "all" else args.assets.split(",")
    tasks = [(a, t, args.params, args.out, args.seed) for t in args.tags.split(",") for a in assets if t == "24h" or a in ASSETS_CRYPTO]
    with Pool(min(args.jobs, len(tasks))) as pool:
        for msg in pool.imap_unordered(build, tasks):
            print(msg, flush=True)
