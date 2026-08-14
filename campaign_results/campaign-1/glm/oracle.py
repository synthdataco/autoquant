"""Oracle ceiling: per-prompt CRPS when sigma = the TRUE realized vol of that prompt.

Decisive diagnostic. For each scored prompt we set the fan's per-step sigma to the
realized 5-min return std of THAT prompt's own realized path (a no-lookahead-cheating
oracle), then score with the validator's CRPS. Compares the oracle's mean CRPS and
implied percentile to:
  - the field's per-prompt distribution (what percentile does perfect-width give?)
  - my model's CRPS (is the gap vol-estimation or model-class?)

If oracle ~= p10  -> vol estimation is the lever (improve the estimate).
If oracle ~= p50  -> constant-sigma Gaussian is the ceiling; need richer structure
                     (intra-path vol dynamics, heavier/matched tails, jump component).
Also tests oracle on the longer intervals (term-structure) and an oracle that knows
the realized vol separately at each horizon.
"""
from __future__ import annotations
import argparse, sys, warnings
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

def slug_for(c): return {"Crypto 1h":"crypto-1h","Crypto 24h":"crypto-24h","Commodities/Equities 24h":"com-equ-24h"}[c.label]

def _t_shocks(rng, df, shape):
    z = rng.standard_t(df, size=shape)
    scale = np.sqrt(df / (df - 2.0))
    return z / scale


def _sv_shocks(rng, sigma0, volvol, vol_persist, shape):
    """Discrete stochastic-vol shocks: log-sigma AR(1) around log(sigma0).
    Each path has its own vol path; vol-of-vol = volvol; persistence = vol_persist.
    Returns shocks scaled so marginal per-step std ~ sigma0 (roughly)."""
    n, steps = shape
    eps = rng.standard_normal(shape)
    eta = rng.standard_normal(shape) * volvol
    log_sig = np.full((n, steps), np.log(sigma0))
    for t in range(1, steps):
        log_sig[:, t] = (vol_persist * (log_sig[:, t - 1] - np.log(sigma0))
                        + np.log(sigma0) + eta[:, t])
    sig = np.exp(log_sig)
    return eps * sig


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comp",default="com-equ-24h")
    ap.add_argument("--asset",default="XAU")
    ap.add_argument("--days",type=int,default=3)
    ap.add_argument("--eval-end",required=True)
    ap.add_argument("--num-simulations",type=int,default=1000)
    ap.add_argument("--n-prompts",type=int,default=0)
    ap.add_argument("--mode",default="gauss",help="gauss|t|sv|sv_t")
    ap.add_argument("--df",type=float,default=5.0)
    ap.add_argument("--volvol",type=float,default=0.8)
    ap.add_argument("--volp",type=float,default=0.97)
    args=ap.parse_args(); ensure_cwd()
    comp=comp_by_name(args.comp); slug=slug_for(comp)
    tl,ti=comp.time_length,comp.time_increment
    end=pd.Timestamp(args.eval_end,tz="UTC"); start=end-timedelta(days=args.days)
    fld=pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{args.asset}_{slug}.parquet")
    fld["scored_time"]=pd.to_datetime(fld["scored_time"],utc=True); fld["start_time"]=pd.to_datetime(fld["start_time"],utc=True)
    fld=fld[(fld["scored_time"]>=start)&(fld["scored_time"]<=end)]
    prompts=fld[["scored_time","start_time"]].drop_duplicates().sort_values("start_time")
    if args.n_prompts: prompts=prompts.head(args.n_prompts)
    if prompts.empty:
        print(f"[{args.asset}/{slug}] no prompts in window"); return
    close=build_close_for_asset(args.asset,prompts["start_time"].min(),prompts["start_time"].max()+timedelta(seconds=tl))
    ctx_minutes=7*24*60

    oracle_crps=[]; pct_at_oracle=[]
    rng=np.random.default_rng(0)
    for st in prompts["start_time"]:
        ctx=context_for_prompt(close,st,ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]): continue
        rp,_=slice_real(close,st,tl,ti)
        if len(rp)!=tl//ti+1: continue
        rp=rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps=realized_path(args.asset,st,tl,ti)
            if rps is not None and len(rps)==tl//ti+1: rp=np.asarray(list(rps),dtype=float)
            else: continue
        last=rp[0]
        rets=np.diff(rp)/rp[:-1]
        oracle_sigma=float(np.std(rets))
        shape=(args.num_simulations,len(rp))
        if args.mode=="gauss":
            sh=rng.normal(0,oracle_sigma,shape)
        elif args.mode=="t":
            sh=_t_shocks(rng,args.df,shape)*oracle_sigma
        elif args.mode=="sv":
            sh=_sv_shocks(rng,oracle_sigma,args.volvol,args.volp,shape)
        elif args.mode=="sv_t":
            sh=_t_shocks(rng,args.df,shape)*_sv_shocks(rng,1.0,args.volvol,args.volp,shape)*oracle_sigma
        sh[:,0]=0
        opaths=last*np.exp(np.cumsum(sh,axis=1))
        ocrps,_=calculate_crps_for_miner(opaths,rp,ti,comp.scoring_intervals)
        oracle_crps.append(ocrps)
        sub=fld[fld["start_time"]==st]
        if not sub.empty and len(sub[sub.crps>0])>5:
            pct=(sub["crps"].values<ocrps).mean()*100
            pct_at_oracle.append(pct)
    oracle_crps=np.array(oracle_crps); pct_at_oracle=np.array(pct_at_oracle)
    print(f"\n[{args.asset}/{slug}] mode={args.mode}, {len(oracle_crps)} prompts, nsim={args.num_simulations}")
    print(f"  mean CRPS = {oracle_crps.mean():.1f}  median={np.median(oracle_crps):.1f}")
    print(f"  mean field-percentile = {pct_at_oracle.mean():.1f}%  (lower=better; ~10% means p10)")
    fv=fld[fld["crps"]>0]["crps"]
    print(f"  field in window: p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f} p90={fv.quantile(.9):.1f}")

if __name__=="__main__":
    main()
