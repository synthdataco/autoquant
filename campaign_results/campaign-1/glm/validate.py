"""Batch official-rank validation across assets & windows.

Generates predictions at the field's EXACT start_times (ALL of them, including prompts
whose start_time precedes the eval window start by up to time_length — those score on
day 1 and must not be left as crps=-1), then runs the official synth_lib backtest and
reports the final + recent rank per asset. This is the GROUND-TRUTH validation used to
pick the champion (validator's own smoothed-score/softmax replay, never reimplemented).

Two windows of opposite regimes are run (within the snapshot's coverage) so the champion
isn't tuned to one vintage. Prints a compact table to stdout and appends to results json.
"""
from __future__ import annotations

import argparse, json, os, sys, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(WORKSPACE)
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth_lib.backtester.orchestration import backtest  # noqa: E402
from synth_lib.preparation.market_data import MinutePriceStore  # noqa: E402
from synth_lib.preparation.config import LEGACY_STORE_SUBDIR, STORE_SUBDIR  # noqa: E402

import modeling as m  # noqa: E402

CTX_MIN = 7 * 24 * 60


def slug_for(c): return {"Crypto 1h":"crypto-1h","Crypto 24h":"crypto-24h","Commodities/Equities 24h":"com-equ-24h"}[c.label]
def comp_by_name(n):
    for c in ALL_COMPETITIONS:
        if slug_for(c)==n: return c
    raise ValueError(n)


def load_minute_series(asset, start, end):
    """NaN-tolerant loader: read daily parquets directly (MinutePriceStore.load_range
    raises on NaN gaps, which SPCX/WTIOIL/SP500 have). Returns a sorted close Series."""
    base = WORKSPACE / "market_data"
    sroot = base / STORE_SUBDIR / asset / "1m"
    sroot = sroot if sroot.exists() else base / LEGACY_STORE_SUBDIR / asset / "1m"
    frames = []
    for d in pd.date_range(start.date(), end.date(), freq="D"):
        p = sroot / f"date={d.date().isoformat()}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no price parquets for {asset} in {sroot}")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    return pd.to_numeric(df["close"], errors="coerce")


def gen_and_rank(asset, comp, end, days, nsim, pred_dir):
    slug = slug_for(comp); tl, ti = comp.time_length, comp.time_increment
    end = pd.Timestamp(end, tz="UTC"); start = end - timedelta(days=days)
    fld = pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{asset}_{slug}.parquet")
    fld["scored_time"] = pd.to_datetime(fld["scored_time"], utc=True)
    fld["start_time"] = pd.to_datetime(fld["start_time"], utc=True)
    fld = fld[(fld["scored_time"] >= start) & (fld["scored_time"] <= end)]
    starts = sorted(fld["start_time"].unique())
    if not starts:
        return None, "no scored prompts in window"
    pmin = min(starts) - timedelta(minutes=CTX_MIN)
    prices = load_minute_series(asset, pmin, end)
    pred_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in starts:
        t = pd.Timestamp(t)
        ctx = prices.loc[:t][-CTX_MIN:]
        if ctx.empty or not np.isfinite(ctx.iloc[-1]):
            continue
        out = m.simulate(asset, t.isoformat(), ti, tl, nsim, ctx)
        paths = [list(map(float, p)) for p in out[2:]]
        payload = {"start_timestamp": t.isoformat(), "asset": asset, "time_increment": ti,
                   "time_length": tl, "num_simulations": len(paths),
                   "num_steps": len(paths[0]), "paths": paths}
        (pred_dir / (t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{tl}.json")).write_text(json.dumps(payload))
        n += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = backtest(miner_name="validate", asset=asset, time_length=tl, time_increment=ti,
                       n_backtest_days=days, predictions_dir=pred_dir, eval_end=end.to_pydatetime(),
                       miner_id=999)
    ss = res.smoothed_scores
    if ss.empty:
        return None, "no smoothed scores"
    last_ts = ss["updated_at"].max(); last = ss[ss["updated_at"] == last_ts]
    me = last[last["miner_uid"] == 999]
    if me.empty:
        return None, "not in final round"
    rank = int((last["reward_weight"] > me["reward_weight"].iloc[0]).sum() + 1)
    recent = ss[ss["updated_at"] >= ss["updated_at"].max() - timedelta(days=1)]
    ranks = []
    for tg, g in recent.groupby("updated_at"):
        mm = g[g["miner_uid"] == 999]
        if not mm.empty:
            ranks.append(int((g["reward_weight"] > mm["reward_weight"].iloc[0]).sum() + 1))
    return {
        "asset": asset, "mean_crps": res.summary["mean_crps"],
        "rank": rank, "n_miners": len(last), "recent_med": int(np.median(ranks)) if ranks else None,
        "recent_min": min(ranks) if ranks else None, "prompts": res.summary["num_prompts"],
        "rw": float(me["reward_weight"].iloc[0]),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--assets", required=True, help="comma-sep")
    ap.add_argument("--eval-ends", required=True, help="comma-sep YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    comp = comp_by_name(args.comp)
    pred_dir = WORKSPACE / "miner_outputs" / "validate" / "predictions"
    assets = args.assets.split(",")
    ends = args.eval_ends.split(",")
    rows = []
    for end in ends:
        print(f"\n##### window ending {end} #####")
        for asset in assets:
            # clear predictions for this asset
            for f in pred_dir.glob(f"*_{asset}_{comp.time_length}.json"):
                f.unlink()
            r, err = gen_and_rank(asset, comp, end, args.days, args.num_simulations, pred_dir)
            if err:
                print(f"  {asset}: {err}")
                continue
            rows.append({"window": end, **r})
            print(f"  {asset:6s} rank={r['rank']:>3}/{r['n_miners']} recent_med={r['recent_med']} mean_crps={r['mean_crps']:.1f} rw={r['rw']:.2e}")
    out = {"tag": args.tag, "comp": args.comp, "days": args.days, "nsim": args.num_simulations, "rows": rows}
    (WORKSPACE / "agent" / f"validate_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print("\n== summary ==")
    print(f"{'window':12s} {'asset':6s} {'rank':>10s} {'recent_med':>10s} {'mean_crps':>10s}")
    for r in rows:
        print(f"{r['window']:12s} {r['asset']:6s} {r['rank']:>3}/{r['n_miners']:<6d} {str(r['recent_med']):>10s} {r['mean_crps']:10.1f}")


if __name__ == "__main__":
    main()
