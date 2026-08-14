"""Reconcile v1 vs oracle: is my EWMA vol biased high/noisy relative to each prompt's
realized 5m vol? Prints, per prompt, my_sigma / realized_sigma ratio and the CRPS each
would give (oracle uses realized sigma). Locates the vol-estimation gap precisely.
"""
from __future__ import annotations
import sys, math, warnings
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd
AGENT_DIR = Path(__file__).resolve().parent; WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
warnings.simplefilter("ignore")
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.competition_config import ALL_COMPETITIONS
from eval_harness import (ensure_cwd, comp_by_name, slug_for, build_close_for_asset,
                           context_for_prompt, slice_real, realized_path)
import modeling as m


def main():
    comp = comp_by_name("com-equ-24h"); slug = slug_for(comp); tl, ti = comp.time_length, comp.time_increment
    asset = "XAU"
    end = pd.Timestamp("2026-07-22", tz="UTC"); start = end - timedelta(days=3)
    fld = pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{asset}_{slug}.parquet")
    fld["scored_time"] = pd.to_datetime(fld["scored_time"], utc=True)
    fld["start_time"] = pd.to_datetime(fld["start_time"], utc=True)
    fld = fld[(fld["scored_time"]>=start)&(fld["scored_time"]<=end)]
    prompts = fld[["scored_time","start_time"]].drop_duplicates().sort_values("start_time").head(25)
    close = build_close_for_asset(asset, prompts["start_time"].min(), prompts["start_time"].max()+timedelta(seconds=tl))
    ctx_minutes = 7*24*60

    ratios = []; v1_crps = []; orc_crps = []
    for st in prompts["start_time"]:
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]): continue
        rp, _ = slice_real(close, st, tl, ti)
        if len(rp) != tl//ti+1: continue
        rp = rp.to_numpy(float)
        if not np.isfinite(rp).all():
            rps = realized_path(asset, st, tl, ti)
            if rps is not None and len(rps) == tl//ti+1: rp = np.asarray(list(rps), float)
            else: continue
        last = rp[0]; real_sig = float(np.std(np.diff(rp)/rp[:-1]))
        # my sigma
        px = np.asarray(ctx, float); px = px[np.isfinite(px)]
        logr = np.diff(np.log(px[px>0])); my_sig = m._per_minute_sigma(logr) * math.sqrt(ti/60.0)
        ratios.append(my_sig / real_sig)
        if len(ratios) <= 5:
            print(f"  {st}: real_sig={real_sig:.6f} my_sig={my_sig:.6f} ratio={my_sig/real_sig:.3f} "
                  f"rp_finite={np.isfinite(rp).all()} rp_len={len(rp)}")
        # v1 crps
        m.T_DF = 9999.0
        out = m.simulate(asset, st.isoformat(), ti, tl, 1000, ctx)
        paths = np.asarray(out[2:], float)
        c1, _ = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
        v1_crps.append(c1)
        # oracle crps
        rng = np.random.default_rng(0)
        sh = rng.normal(0, real_sig, (1000, len(rp))); sh[:,0]=0
        op = last*np.exp(np.cumsum(sh, axis=1))
        co, _ = calculate_crps_for_miner(op, rp, ti, comp.scoring_intervals)
        orc_crps.append(co)
    ratios = np.array(ratios); v1_crps = np.array(v1_crps); orc_crps = np.array(orc_crps)
    print(f"[{asset}] {len(ratios)} prompts")
    print(f"  my_sigma/realized_sigma: mean={ratios.mean():.3f} median={np.median(ratios):.3f} "
          f"p10={np.percentile(ratios,10):.3f} p90={np.percentile(ratios,90):.3f}")
    print(f"  v1 mean CRPS = {v1_crps.mean():.1f}  oracle mean CRPS = {orc_crps.mean():.1f}  "
          f"ratio={v1_crps.mean()/orc_crps.mean():.3f}")
    print(f"  prompts where my_sigma > real_sigma (over-wide): {(ratios>1).mean()*100:.0f}%")
    # spread of v1 vs oracle crps: noise + bias
    print(f"  v1 CRPS p10={np.percentile(v1_crps,10):.0f} p50={np.percentile(v1_crps,50):.0f}")
    print(f"  oracle CRPS p10={np.percentile(orc_crps,10):.0f} p50={np.percentile(orc_crps,50):.0f}")

if __name__ == "__main__":
    main()
