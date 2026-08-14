"""Shape-lever oracle: stochastic vol & jump-diffusion fans vs constant Gaussian.

All with NO drift (drift is unpredictable — confirmed). Per-step sigma = realized 5m std
(cheating on vol only, to isolate the SHAPE lever). Tests:
  - gauss  : constant sigma (baseline oracle)
  - t      : Student-t per-step (fatter per-step tails)
  - sv     : log-AR(1) stochastic vol (vol clustering -> fat CUMULATIVE tails)
  - jump   : jump-diffusion (compound Poisson jumps -> fat cumulative tails)
  - sv_jump: SV + jumps
Capped/clipped to avoid overflow. Reports mean CRPS and field-percentile.
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


def sv_shocks(rng, sigma0, volvol, volp, shape, clip=2.0):
    """log-sigma AR(1) around log(sigma0). clip bounds log-sigma deviation to avoid blowup.
    Returns shocks; marginal per-step std ~ sigma0 (we rescale afterwards)."""
    n, steps = shape
    eps = rng.standard_normal(shape)
    eta = rng.standard_normal(shape) * volvol
    ls0 = math.log(sigma0)
    log_sig = np.zeros(shape)
    ls = np.full(n, ls0)
    out = np.zeros(shape)
    for t in range(steps):
        if t > 0:
            ls = volp * (ls - ls0) + ls0 + eta[:, t]
            ls = np.clip(ls, ls0 - clip, ls0 + clip)
        sig = np.exp(ls)
        out[:, t] = eps[:, t] * sig
    # rescale so per-step std == sigma0 exactly (remove level drift from the AR)
    out = out * (sigma0 / np.std(out))
    out[:, 0] = 0.0
    return out


def jump_shocks(rng, sigma0, lam, jsize, shape):
    """Merton jump-diffusion: normal shocks + compound Poisson jumps.
    lam = expected jumps per step; jsize = std of log jump size.
    Calibrated so per-step std ~= sigma0 (subtract mean of jump contribution)."""
    n, steps = shape
    gauss = rng.normal(0, sigma0, shape)
    # jumps: Bernoulli(lam) per step, jump ~ Normal(-0.5*jsize^2, jsize) (mean-corrected)
    n_jumps = rng.poisson(lam, shape)
    jumps = rng.normal(-0.5*jsize**2, jsize, shape) * (n_jumps > 0)
    out = gauss + jumps
    out[:, 0] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--n-prompts", type=int, default=0)
    ap.add_argument("--df", type=float, default=5.0)
    ap.add_argument("--volvol", type=float, default=1.0)
    ap.add_argument("--volp", type=float, default=0.96)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--jsize", type=float, default=2.0)
    ap.add_argument("--modes", default="gauss,t,sv,jump,sv_jump")
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
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max()+timedelta(seconds=tl))
    modes = args.modes.split(",")
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
        sig = float(np.std(np.diff(rp)/rp[:-1]))
        for mode in modes:
            rng = np.random.default_rng(0)
            if mode == "gauss":
                sh = rng.normal(0, sig, (args.num_simulations, n)); sh[:,0]=0
            elif mode == "t":
                sh = t_shocks(rng, args.df, (args.num_simulations, n)) * sig
            elif mode == "sv":
                sh = sv_shocks(rng, sig, args.volvol, args.volp, (args.num_simulations, n))
            elif mode == "jump":
                sh = jump_shocks(rng, sig, args.lam, args.jsize, (args.num_simulations, n))
            elif mode == "sv_jump":
                a = sv_shocks(rng, sig, args.volvol, args.volp, (args.num_simulations, n))
                b = jump_shocks(rng, 0.0, args.lam, args.jsize, (args.num_simulations, n))
                sh = a + b; sh[:,0]=0
            sh[:, 0] = 0.0
            paths = last * np.exp(np.cumsum(sh, axis=1))
            # sanitize
            paths = np.where(np.isfinite(paths) & (paths > 0), paths, last)
            c, _ = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
            res[mode].append(c)
            sub = fld[fld["start_time"]==st]
            if not sub.empty and len(sub[sub.crps>0])>5:
                pct[mode].append((sub["crps"].values<c).mean()*100)
    print(f"\n[{args.asset}/{slug}] {len(res['gauss'])} prompts, nsim={args.num_simulations}")
    print(f"{'mode':10s} {'meanCRPS':>10s} {'fieldPct%':>10s}")
    fv = fld[fld["crps"]>0]["crps"]
    for m in modes:
        rr = np.array(res[m]); pp = np.array(pct[m])
        print(f"{m:10s} {rr.mean():10.1f} {pp.mean() if len(pp) else float('nan'):10.1f}")
    print(f"  field p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f} p90={fv.quantile(.9):.1f}")


if __name__ == "__main__":
    main()
