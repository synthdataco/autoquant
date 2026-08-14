"""Historical-bootstrap oracle: draw recent realized return BLOCKS as the fan.

No-lookahead (uses only context up to start_time). For each prompt, build a fan by
sampling 5-min log-returns from the recent context (block bootstrap to preserve
short-range dependence), then exponentiate from the anchor. Compared to the
Gaussian oracle, this carries the EMPIRICAL distribution shape (kurtosis, asymmetry,
vol clustering) for free — the candidate structural lever that needs no drift lookahead.

Variants:
  - boot_iid   : iid resample of recent 5m log-returns
  - boot_block : block bootstrap (block len = 20 steps) of recent 5m log-returns
  - boot_rets  : use 1m returns then aggregate? (skip; use 5m directly)
Sigma is whatever the empirical history gives (NO oracle vol). Tests whether the
empirical shape alone beats the calibrated Gaussian oracle.
"""
from __future__ import annotations
import argparse, sys, math, warnings
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from eval_harness import (  # noqa: E402
    ensure_cwd, comp_by_name, slug_for, build_close_for_asset,
    context_for_prompt, slice_real, realized_path,
)


def t_shocks(rng, df, shape):
    z = rng.standard_t(df, size=shape); return z / math.sqrt(df/(df-2.0))


def build_5m_logrets(close, t, lookback_days=7):
    """Recent realized 5-min log-returns up to (not incl) t."""
    c = close.loc[:t].iloc[-lookback_days*24*12:]
    c5 = np.log(c).resample("5min").last().dropna()
    return np.diff(c5.values)


def boot_iid(rng, pool, n, nsim):
    idx = rng.integers(0, len(pool), (nsim, n))
    return pool[idx]


def boot_block(rng, pool, n, nsim, block=20):
    out = np.zeros((nsim, n))
    L = len(pool)
    starts = rng.integers(0, L - block, nsim)
    for i in range(nsim):
        pos = 0
        s = starts[i]
        while pos < n:
            take = min(block, n - pos)
            out[i, pos:pos+take] = pool[s % L: s % L + take]
            s += block
            if s + block >= L:
                s = rng.integers(0, L - block)
            pos += take
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--n-prompts", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=3, help="days of history for bootstrap pool")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--df", type=float, default=5.0)
    args = ap.parse_args(); ensure_cwd()
    comp = comp_by_name(args.comp); slug = slug_for(comp)
    tl, ti = comp.time_length, comp.time_increment
    end = pd.Timestamp(args.eval_end, tz="UTC"); start = end - timedelta(days=args.days)
    fld = pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{args.asset}_{slug}.parquet")
    fld["scored_time"] = pd.to_datetime(fld["scored_time"], utc=True)
    fld["start_time"] = pd.to_datetime(fld["start_time"], utc=True)
    fld = fld[(fld["scored_time"]>=start)&(fld["scored_time"]<=end)]
    prompts = fld[["scored_time","start_time"]].drop_duplicates().sort_values("start_time")
    if args.n_prompts: prompts = prompts.head(args.n_prompts)
    if prompts.empty: print(f"[{args.asset}/{slug}] no prompts"); return
    close = build_close_for_asset(args.asset, prompts["start_time"].min() - timedelta(days=args.lookback),
                                  prompts["start_time"].max()+timedelta(seconds=tl))
    modes = ["gauss_oracle", "t_oracle", "boot_iid", "boot_block"]
    res = {m: [] for m in modes}; pct = {m: [] for m in modes}

    for st in prompts["start_time"]:
        rp, _ = slice_real(close, st, tl, ti)
        if len(rp) != tl//ti+1: continue
        rp = rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == tl//ti+1: rp = np.asarray(list(rps), dtype=float)
            else: continue
        last = rp[0]; n = len(rp)
        rets = np.diff(rp)/rp[:-1]
        sig = float(np.std(rets))
        pool = build_5m_logrets(close, st, args.lookback)
        if len(pool) < 50:
            continue
        sub = fld[fld["start_time"]==st]
        for mode in modes:
            rng = np.random.default_rng(0)
            if mode == "gauss_oracle":
                sh = rng.normal(0, sig, (args.num_simulations, n)); sh[:,0]=0
            elif mode == "t_oracle":
                sh = t_shocks(rng, args.df, (args.num_simulations, n)) * sig
            elif mode == "boot_iid":
                sh = boot_iid(rng, pool, n, args.num_simulations)
            elif mode == "boot_block":
                sh = boot_block(rng, pool, n, args.num_simulations, args.block)
            sh[:, 0] = 0.0
            paths = last * np.exp(np.cumsum(sh, axis=1))
            paths = np.where(np.isfinite(paths) & (paths > 0), paths, last)
            c, _ = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
            res[mode].append(c)
            if not sub.empty and len(sub[sub.crps>0])>5:
                pct[mode].append((sub["crps"].values<c).mean()*100)
    print(f"\n[{args.asset}/{slug}] {len(res['gauss_oracle'])} prompts, nsim={args.num_simulations}, lookback={args.lookback}d")
    print(f"{'mode':14s} {'meanCRPS':>10s} {'fieldPct%':>10s}")
    fv = fld[fld["crps"]>0]["crps"]
    for m in modes:
        rr = np.array(res[m]); pp = np.array(pct[m])
        print(f"{m:14s} {rr.mean():10.1f} {pp.mean() if len(pp) else float('nan'):10.1f}")
    print(f"  field p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f} p90={fv.quantile(.9):.1f}")


if __name__ == "__main__":
    main()
