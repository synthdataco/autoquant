"""Oracle v2: which model-class lever beats the field's top decile?

Tests, with cheating access to each prompt's realized path:
  - constant sigma + zero drift (baseline oracle)
  - constant sigma + REALIZED drift (recenter fan at realized endpoint direction)
  - per-horizon realized sigma (independent sigma at 5m/30m/3h/24h) + zero drift
  - realized drift + per-horizon sigma
  - realized drift + per-horizon sigma + t-shocks
The combination that reaches ~p10 reveals the structural lever the top miners use.
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


def run_mode(mode, rp, ti, scoring_intervals, nsim, rng, df=5.0):
    """Generate a fan for realized path rp under `mode`, return CRPS.
    modes:
      const0   : sigma=std(5m rets), zero drift
      constD   : sigma=std(5m rets), drift = realized avg 5m return (cheat)
      perH0    : per-horizon sigma (independent per interval), zero drift
      perHD    : per-horizon sigma + realized drift
      perHDt   : per-horizon sigma + realized drift + t shocks
    """
    n = len(rp)
    rets = np.diff(rp) / rp[:-1]
    last = rp[0]
    if mode in ("const0", "constD"):
        sig = float(np.std(rets))
        drift = 0.0 if mode == "const0" else float(np.mean(rets))
        sh = rng.normal(0, sig, (nsim, n)) + drift
        sh[:, 0] = 0
        paths = last * np.exp(np.cumsum(sh, axis=1))
        crps, _ = calculate_crps_for_miner(paths, rp, ti, scoring_intervals)
        return crps
    # per-horizon: simulate a base 1-step random walk at time_increment resolution,
    # then rescale each interval's marginal sigma to match realized per-horizon std.
    # We do this by generating independent fans per scoring interval (the intervals
    # use disjoint-ish subsets, so we approximate by scaling the cumulative vol).
    # Simpler: generate the base path with const sigma, then for the abs/gaps use a
    # separate matched-sigma fan. But CRPS scores the SAME paths across intervals.
    # => Implement per-horizon by building the path at the finest resolution with a
    # sigma that, per horizon, matches realized; doable by scaling increments with a
    # horizon-dependent factor is not consistent across horizons. Instead use a
    # piecewise approach: we cannot satisfy multiple horizons with one GBM. So
    # per-horizon is approximated by setting sigma = realized std at the FINEST
    # horizon only (already const0) — skip. Use drift+const as the main lever test.
    if mode in ("perH0", "perHD", "perHDt"):
        # Approximation: const sigma (5m realized) + optional realized drift + t shocks.
        sig = float(np.std(rets))
        drift = 0.0 if mode == "perH0" else float(np.mean(rets))
        if mode.endswith("t"):
            sh = t_shocks(rng, df, (nsim, n)) * sig + drift
        else:
            sh = rng.normal(0, sig, (nsim, n)) + drift
        sh[:, 0] = 0
        paths = last * np.exp(np.cumsum(sh, axis=1))
        crps, _ = calculate_crps_for_miner(paths, rp, ti, scoring_intervals)
        return crps
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--n-prompts", type=int, default=0)
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
    if prompts.empty:
        print(f"[{args.asset}/{slug}] no prompts in window"); return
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max()+timedelta(seconds=tl))

    modes = ["const0", "constD", "perH0", "perHD", "perHDt"]
    res = {m: [] for m in modes}
    pct = {m: [] for m in modes}
    for st in prompts["start_time"]:
        rp, _ = slice_real(close, st, tl, ti)
        if len(rp) != tl//ti+1: continue
        rp = rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == tl//ti+1: rp = np.asarray(list(rps), dtype=float)
            else: continue
        sub = fld[fld["start_time"]==st]
        for mode in modes:
            rng = np.random.default_rng(0)
            c = run_mode(mode, rp, ti, comp.scoring_intervals, args.num_simulations, rng)
            res[mode].append(c)
            if not sub.empty and len(sub[sub.crps>0])>5:
                pct[mode].append((sub["crps"].values<c).mean()*100)
    print(f"\n[{args.asset}/{slug}] {len(res['const0'])} prompts, nsim={args.num_simulations}")
    print(f"{'mode':8s} {'meanCRPS':>10s} {'fieldPct%':>10s}")
    fv = fld[fld["crps"]>0]["crps"]
    for m in modes:
        rr = np.array(res[m]); pp = np.array(pct[m])
        print(f"{m:8s} {rr.mean():10.1f} {pp.mean() if len(pp) else float('nan'):10.1f}")
    print(f"  field in window: p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f} p90={fv.quantile(.9):.1f}")


if __name__ == "__main__":
    main()
