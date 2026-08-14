"""Optimal vol estimator: sweep EWMA half-life (and a realized-range estimator) against
the per-prompt REALIZED 5m vol, measuring both correlation and the CRPS the resulting fan
achieves. The estimator with the best mean-CRPS (not just correlation) is the v2 vol.

Also tests a "regime-shrinkage" estimator: blend the EWMA with the long-run mean vol so
that on calm-after-volatile days the estimate regresses toward the calmer long-run level
(reducing the over-wide penalty). The blend weight is swept.
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


def ewm_vol(logr, hl_min):
    alpha = 1.0 - math.exp(-math.log(2) / hl_min)
    return float(pd.Series(logr).ewm(alpha=alpha, adjust=False).std().iloc[-1])


def fan_crps(last, n, sig, rp, ti, intervals, nsim, rng, df=5.0):
    sh = t_shocks(rng, df, (nsim, n)) * sig
    sh[:, 0] = 0
    p = last * np.exp(np.cumsum(sh, axis=1))
    p = np.where(np.isfinite(p) & (p > 0), p, last)
    return calculate_crps_for_miner(p, rp, ti, intervals)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=600)
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
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max()+timedelta(seconds=tl))
    ctx_minutes = 7*24*60
    step_min = ti//60

    # Precompute per-prompt: context logr, realized sigma
    data = []
    for st in prompts["start_time"]:
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]): continue
        rp, _ = slice_real(close, st, tl, ti)
        if len(rp) != tl//ti+1: continue
        rp = rp.to_numpy(float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == tl//ti+1: rp = np.asarray(list(rps), float)
            else: continue
        px = np.asarray(ctx, float); px = px[np.isfinite(px)]
        logr = np.diff(np.log(px[px>0]))
        # realized 5m return std IS already the per-step (ti=300) sigma — do NOT rescale.
        real_sig = float(np.std(np.diff(rp)/rp[:-1]))
        data.append((st, logr, rp, real_sig))
    print(f"[{args.asset}/{slug}] {len(data)} prompts, nsim={args.num_simulations}")

    halflives = [30, 60, 120, 300, 600, 1200, 2880]
    long_run = float(np.std(np.concatenate([d[1] for d in data]))) if data else 0.0  # global 1m std
    print(f"long-run 1m std = {long_run:.6f}")
    print(f"{'method':24s} {'meanCRPS':>9s} {'corr_est_real':>13s} {'overwide%':>10s}")
    fv = fld[fld["crps"]>0]["crps"]
    # oracle reference
    orc = []
    for st, logr, rp, real_sig in data:
        rng = np.random.default_rng(0)
        orc.append(fan_crps(rp[0], len(rp), real_sig, rp, ti, comp.scoring_intervals, args.num_simulations, rng, args.df))
    print(f"{'ORACLE (real vol)':24s} {np.mean(orc):9.1f} {'':>13s} {'':>10s}   (field p10={fv.quantile(.1):.0f} p50={fv.quantile(.5):.0f})")
    for hl in halflives:
        crps = []; ests = []; reals = []
        for st, logr, rp, real_sig in data:
            s1m = ewm_vol(logr, hl); s_step = s1m * math.sqrt(step_min)
            rng = np.random.default_rng(0)
            crps.append(fan_crps(rp[0], len(rp), s_step, rp, ti, comp.scoring_intervals, args.num_simulations, rng, args.df))
            ests.append(s_step); reals.append(real_sig)
        ests = np.array(ests); reals = np.array(reals)
        corr = np.corrcoef(ests, reals)[0,1] if len(ests) > 2 else float('nan')
        overwide = (ests > reals).mean() * 100
        print(f"{'EWMA hl=%dm'%hl:24s} {np.mean(crps):9.1f} {corr:13.3f} {overwide:10.0f}")
    # shrinkage: ewm(hl) blended with long_run toward long-run 1m std*sqrt(step)
    for hl in [600, 1200]:
        for w in [0.3, 0.5, 0.7]:
            crps = []; ests = []; reals = []
            for st, logr, rp, real_sig in data:
                s1m = ewm_vol(logr, hl)
                shrunk = w * s1m + (1 - w) * long_run
                s_step = shrunk * math.sqrt(step_min)
                rng = np.random.default_rng(0)
                crps.append(fan_crps(rp[0], len(rp), s_step, rp, ti, comp.scoring_intervals, args.num_simulations, rng, args.df))
                ests.append(s_step); reals.append(real_sig)
            ests = np.array(ests); reals = np.array(reals)
            overwide = (ests > reals).mean() * 100
            print(f"{'shrink hl=%d w=%.1f'%(hl,w):24s} {np.mean(crps):9.1f} {np.corrcoef(ests,reals)[0,1]:13.3f} {overwide:10.0f}")


if __name__ == "__main__":
    main()
