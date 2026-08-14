"""No-lookahead drift test: does recent momentum predict the realized 24h drift well
enough to capture the oracle-drift CRPS gain?

For each scored prompt:
  - realized_drift = (rp[-1]/rp[0] - 1) / steps  (cheating, the oracle lever)
  - momentum_drift = EWMA mean of recent minute returns up to start_time (available,
    NO lookahead) at various horizons / shrink factors
Compares CRPS of: const0 (zero drift), realized-drift (oracle ceiling), momentum-drift
(no lookahead) at several horizons. If momentum-drift captures most of the realized-drift
gain, drift is a USABLE no-lookahead lever.
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


def fan(last, n, sig, drift, nsim, rng, df=5.0, heavy=True):
    if heavy:
        sh = t_shocks(rng, df, (nsim, n)) * sig + drift
    else:
        sh = rng.normal(0, sig, (nsim, n)) + drift
    sh[:, 0] = 0
    return last * np.exp(np.cumsum(sh, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--n-prompts", type=int, default=0)
    ap.add_argument("--mom-windows", default="60,180,720,2880")  # minutes
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
        print(f"[{args.asset}/{slug}] no prompts"); return
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max()+timedelta(seconds=tl))
    ctx_minutes = 7*24*60
    step_min = ti//60
    moms = [int(x) for x in args.mom_windows.split(",")]

    res = {"zero": [], "realized": []}
    for w in moms:
        res[f"mom{w}"] = []
    corr_real = {w: [] for w in moms}

    for st in prompts["start_time"]:
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]): continue
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
        # realized per-step drift (oracle)
        real_dr = float(np.mean(rets))
        # momentum drifts from context (NO lookahead): EWMA mean of recent 1m returns,
        # scaled to per-step (step_min)
        lr = np.diff(np.log(ctx.values[ctx.values>0]))
        for w in moms:
            if len(lr) >= w:
                mom = float(np.mean(lr[-w:])) * step_min  # per-step drift estimate
            else:
                mom = 0.0
            rng = np.random.default_rng(0)
            p = fan(last, n, sig, mom, args.num_simulations, rng)
            c, _ = calculate_crps_for_miner(p, rp, ti, comp.scoring_intervals)
            res[f"mom{w}"].append(c)
            corr_real[w].append((real_dr, mom))
        rng = np.random.default_rng(0)
        p0 = fan(last, n, sig, 0.0, args.num_simulations, rng)
        res["zero"].append(calculate_crps_for_miner(p0, rp, ti, comp.scoring_intervals)[0])
        rng = np.random.default_rng(0)
        pr = fan(last, n, sig, real_dr, args.num_simulations, rng)
        res["realized"].append(calculate_crps_for_miner(pr, rp, ti, comp.scoring_intervals)[0])

    print(f"\n[{args.asset}/{slug}] {len(res['zero'])} prompts, nsim={args.num_simulations}, t(df=5)")
    print(f"{'mode':10s} {'meanCRPS':>10s}")
    fv = fld[fld["crps"]>0]["crps"]
    print(f"{'zero':10s} {np.mean(res['zero']):10.1f}")
    print(f"{'realized':10s} {np.mean(res['realized']):10.1f}  (oracle ceiling)")
    for w in moms:
        print(f"{'mom'+str(w):10s} {np.mean(res[f'mom{w}']):10.1f}")
        # correlation realized vs momentum
        arr = np.array(corr_real[w])
        if len(arr) > 3:
            print(f"           corr(mom{w}, realized_drift)={np.corrcoef(arr[:,0],arr[:,1])[0,1]:.3f}")
    print(f"  field p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f}")


if __name__ == "__main__":
    main()
