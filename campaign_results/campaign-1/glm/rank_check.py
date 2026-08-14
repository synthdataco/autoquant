"""Official rank check: run synth_lib backtest() and report the field rank.

Uses the SAME ranking path as agent/predict.py (synth_lib.backtester.orchestration
backtest) but extracts the final + recent rank from BacktestResult.smoothed_scores,
which predict.py does not print. This is the GROUND-TRUTH rank (validator's own
smoothed-score/softmax replay). Cross-check for agent/eval_harness.py.
"""
from __future__ import annotations

import argparse, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(WORKSPACE)

from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth_lib.backtester.orchestration import backtest  # noqa: E402


def slug_for(comp):
    return {"Crypto 1h": "crypto-1h", "Crypto 24h": "crypto-24h",
            "Commodities/Equities 24h": "com-equ-24h"}[comp.label]


def comp_by_name(name):
    for c in ALL_COMPETITIONS:
        if slug_for(c) == name:
            return c
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="crypto-24h")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--cadence", type=int, default=60)
    args = ap.parse_args()
    comp = comp_by_name(args.comp)
    slug = slug_for(comp)
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)
    sys.path.insert(0, str(AGENT_DIR))
    import predict  # noqa: E402
    pred_dir = predict.PREDICTIONS_DIR
    predict.ensure_workspace_cwd()
    predict.check_snapshot_coverage(args.asset, start - timedelta(minutes=predict.CONTEXT_MINUTES), end)

    # Generate at the field's EXACT scored prompt start_times (irregular cadence) so the
    # backtest matches each prediction to a realized path with the SAME anchor price.
    # A fixed grid misanchors the fan (price drifts between grid time and prompt time),
    # inflating CRPS — a cadence artifact absent in production (simulate is called at the
    # exact prompt time).
    fld = pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{args.asset}_{slug}.parquet")
    fld["scored_time"] = pd.to_datetime(fld["scored_time"], utc=True)
    fld["start_time"] = pd.to_datetime(fld["start_time"], utc=True)
    fld = fld[(fld["scored_time"] >= pd.Timestamp(start)) & (fld["scored_time"] <= pd.Timestamp(end))]
    prompt_starts = sorted(fld["start_time"].unique())
    # The backtest also matches prompts whose start_time is ~time_length before the first
    # scored_time in the window (scored on day 1 but started the prior day). Those are real
    # scored prompts; if we don't generate for them they score crps=-1 (filled at p95) and
    # tank the rank. In production simulate() answers every prompt, so generate for all of
    # them. Load context back far enough (7d before the EARLIEST start_time).
    from synth_lib.preparation.market_data import MinutePriceStore
    from synth_lib.preparation.config import LEGACY_STORE_SUBDIR, STORE_SUBDIR
    base = WORKSPACE / "market_data"
    sroot = base / STORE_SUBDIR / args.asset / "1m"
    sroot = sroot if sroot.exists() else base / LEGACY_STORE_SUBDIR / args.asset / "1m"
    store = MinutePriceStore(args.asset, root=sroot)
    pmin = min(prompt_starts) - timedelta(minutes=predict.CONTEXT_MINUTES)
    df = store.load_range(pmin, end).set_index("timestamp")
    prices = pd.to_numeric(df["close"], errors="coerce")
    pred_dir.mkdir(parents=True, exist_ok=True)
    import modeling as m
    n = 0
    for t in prompt_starts:
        t = pd.Timestamp(t)
        context = prices.loc[:t][-predict.CONTEXT_MINUTES:]
        out = m.simulate(asset=args.asset, start_time=t.isoformat(),
                         time_increment=comp.time_increment, time_length=comp.time_length,
                         num_simulations=args.num_simulations, context_prices=context)
        paths = [list(map(float, p)) for p in out[2:]]
        import json
        payload = {"start_timestamp": t.isoformat(), "asset": args.asset,
                   "time_increment": comp.time_increment, "time_length": comp.time_length,
                   "num_simulations": len(paths), "num_steps": len(paths[0]), "paths": paths}
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{args.asset}_{comp.time_length}.json"
        (pred_dir / name).write_text(json.dumps(payload))
        n += 1
    print(f"generated {n} predictions", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = backtest(miner_name="rankcheck", asset=args.asset, time_length=comp.time_length,
                       time_increment=comp.time_increment, n_backtest_days=args.days,
                       predictions_dir=pred_dir, eval_end=end, miner_id=999)
    ss = res.smoothed_scores
    if ss.empty:
        print("no smoothed scores"); return
    print(f"\nsummary: {res.summary}")
    last_ts = ss["updated_at"].max()
    last = ss[ss["updated_at"] == last_ts]
    me = last[last["miner_uid"] == 999]
    if me.empty:
        print(f"miner 999 not in final round ({last_ts})"); return
    rank = int((last["reward_weight"] > me["reward_weight"].iloc[0]).sum() + 1)
    print(f">> RANK {rank}/{len(last)} at {last_ts} (rw={me['reward_weight'].iloc[0]:.6f} smoothed={me['new_smoothed_score'].iloc[0]:.3f})")
    recent = ss[ss["updated_at"] >= ss["updated_at"].max() - timedelta(days=2)]
    ranks = []
    for t, g in recent.groupby("updated_at"):
        m = g[g["miner_uid"] == 999]
        if not m.empty:
            ranks.append(int((g["reward_weight"] > m["reward_weight"].iloc[0]).sum() + 1))
    if ranks:
        print(f">> recent {len(ranks)} rounds: median={int(np.median(ranks))} min={min(ranks)} max={max(ranks)}")


if __name__ == "__main__":
    main()
