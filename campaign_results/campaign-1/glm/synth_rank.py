"""Rank a SYNTHETIC miner whose per-prompt CRPS is a fixed percentile of the field.
Tells us the CRPS percentile needed for a target rank, decoupling modeling from ranking.
"""
from __future__ import annotations
import argparse, sys, warnings
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth_lib.backtester.scoring import (  # noqa: E402
    calculate_smoothed_scores, _compute_prompt_score_stats_for_group,
)

def slug_for(c): return {"Crypto 1h":"crypto-1h","Crypto 24h":"crypto-24h","Commodities/Equities 24h":"com-equ-24h"}[c.label]
def comp_by_name(n):
    for c in ALL_COMPETITIONS:
        if slug_for(c)==n: return c
    raise ValueError(n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comp",default="com-equ-24h")
    ap.add_argument("--asset",default="XAU")
    ap.add_argument("--days",type=int,default=3)
    ap.add_argument("--eval-end",required=True)
    ap.add_argument("--percentiles",default="1,5,10,25,50")
    args=ap.parse_args()
    comp=comp_by_name(args.comp); slug=slug_for(comp)
    end=pd.Timestamp(args.eval_end,tz="UTC"); start=end-timedelta(days=args.days)
    fld=pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{args.asset}_{slug}.parquet")
    fld["scored_time"]=pd.to_datetime(fld["scored_time"],utc=True); fld["start_time"]=pd.to_datetime(fld["start_time"],utc=True)
    w=fld[(fld["scored_time"]>=start)&(fld["scored_time"]<=end)].copy()
    prompts=w[["scored_time","start_time","asset","time_length","time_increment"]].drop_duplicates()
    needed=["scored_time","miner_uid","asset","crps","time_length","time_increment","start_time"]
    rh=pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"rewards_history_{slug}.parquet")
    rh["updated_at"]=pd.to_datetime(rh["updated_at"],utc=True)
    rh=rh[(rh["updated_at"]>=w["scored_time"].min()-timedelta(days=comp.window_days))&(rh["updated_at"]<=w["scored_time"].max()+timedelta(days=1))]
    rh=rh[["updated_at","prompt_name","miner_uid","smoothed_score","reward_weight"]].copy()
    nr=[{"updated_at":t,"prompt_name":rh["prompt_name"].iloc[0],"miner_uid":999,"smoothed_score":np.nan,"reward_weight":np.nan} for t in rh["updated_at"].sort_values().unique()]
    rh_base=pd.concat([rh,pd.DataFrame(nr)],ignore_index=True).sort_values(["updated_at","miner_uid"])
    print(f"[{args.asset}/{slug}] {len(prompts)} prompts, {len(w)} field rows")
    for pct in [float(x) for x in args.percentiles.split(",")]:
        sr=[]
        for _,p in prompts.iterrows():
            sub=w[w["scored_time"]==p["scored_time"]]
            c=np.percentile(sub["crps"][sub["crps"]>0],pct)
            sr.append({"miner_uid":999,"scored_time":p["scored_time"],"start_time":p["start_time"],"asset":p["asset"],"time_length":p["time_length"],"time_increment":p["time_increment"],"crps":float(c)})
        mine=pd.DataFrame(sr)
        alls=pd.concat([w[needed],mine[needed]],ignore_index=True).sort_values(["scored_time","miner_uid"])
        st=alls.groupby(["scored_time","asset","time_length","time_increment"],group_keys=False)["crps"].apply(_compute_prompt_score_stats_for_group)
        alls["new_prompt_scores"]=st["new_prompt_scores"]; alls["percentile90"]=st["percentile90"]; alls["lowest_score"]=st["lowest_score"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm=calculate_smoothed_scores(alls,rh_base,cutoff_days=comp.window_days,scores_column="new_prompt_scores",competition=comp)
        if sm.empty: print(f"p{pct}: no smoothed scores"); continue
        last=sm[sm["updated_at"]==sm["updated_at"].max()]; me=last[last["miner_uid"]==999]
        if me.empty: print(f"p{pct}: not in final round"); continue
        rank=int((last["reward_weight"]>me["reward_weight"].iloc[0]).sum()+1)
        print(f"p{pct:>4} CRPS miner -> RANK {rank:>3}/{len(last)}  rw={me['reward_weight'].iloc[0]:.6f} smoothed={me['new_smoothed_score'].iloc[0]:.3f}")

if __name__=="__main__":
    main()
