"""Compute the achievable RANK ceiling: run a cheating-oracle miner (per-prompt sigma =
that prompt's realized 5m vol, Gaussian) through the official backtester. If even the
oracle ranks poorly, the softmax/concentration makes a high rank structurally hard for a
calibrated model (the top ranks are seed-luck). Tells us the realistic rank target.
"""
from __future__ import annotations
import argparse, os, sys, json, math, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(WORKSPACE)
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth_lib.backtester.orchestration import backtest  # noqa: E402
from synth_lib.preparation.market_data import MinutePriceStore  # noqa: E402
from synth_lib.preparation.config import LEGACY_STORE_SUBDIR, STORE_SUBDIR  # noqa: E402
from synth_lib.backtester.preparation import _slice_real_prices  # noqa: E402
from synth_lib.backtester.loading import download_price_data  # noqa: E402


def slug_for(c): return {"Crypto 1h":"crypto-1h","Crypto 24h":"crypto-24h","Commodities/Equities 24h":"com-equ-24h"}[c.label]
def comp_by_name(n):
    for c in ALL_COMPETITIONS:
        if slug_for(c)==n: return c
    raise ValueError(n)


def oracle_simulate(asset, start_time, ti, tl, nsim, realized_prices, rng_seed=0, mode="gauss", df=5.0, tilt=1.0):
    """CHEATING: sigma = std of realized 5m returns of THIS prompt.
    mode: gauss | t. tilt multiplies sigma (e.g. 0.9 = narrower fan, seed-luck bet)."""
    rp = np.asarray(realized_prices, float)
    sig = float(np.std(np.diff(rp)/rp[:-1])) * tilt if len(rp) > 2 else 1e-4
    last = rp[0] if np.isfinite(rp[0]) else 1.0
    rng = np.random.default_rng(rng_seed)
    if mode == "t":
        z = rng.standard_t(df, (nsim, len(rp))) / math.sqrt(df/(df-2.0))
    else:
        z = rng.standard_normal((nsim, len(rp)))
    sh = z * sig; sh[:, 0] = 0
    paths = last * np.exp(np.cumsum(sh, axis=1))
    paths = np.where(np.isfinite(paths) & (paths > 0), paths, last)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="com-equ-24h")
    ap.add_argument("--asset", default="XAU")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--mode", default="gauss", help="gauss|t")
    ap.add_argument("--df", type=float, default=5.0)
    ap.add_argument("--tilt", type=float, default=1.0)
    ap.add_argument("--seed-mode", default="fixed", help="fixed|random per-prompt rng seed")
    args = ap.parse_args()
    comp = comp_by_name(args.comp); slug = slug_for(comp)
    tl, ti = comp.time_length, comp.time_increment
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)
    fld = pd.read_parquet(WORKSPACE/"market_data"/"offline_data"/f"miner_scores_{args.asset}_{slug}.parquet")
    fld["scored_time"] = pd.to_datetime(fld["scored_time"], utc=True)
    fld["start_time"] = pd.to_datetime(fld["start_time"], utc=True)
    fld = fld[(fld["scored_time"] >= pd.Timestamp(start)) & (fld["scored_time"] <= pd.Timestamp(end))]
    prompt_starts = sorted(fld["start_time"].unique())
    # need minute prices to slice realized for the cheating oracle
    base = WORKSPACE / "market_data"
    sroot = base / STORE_SUBDIR / args.asset / "1m"
    sroot = sroot if sroot.exists() else base / LEGACY_STORE_SUBDIR / args.asset / "1m"
    # generate oracle predictions
    pred_dir = WORKSPACE / "miner_outputs" / "ceiling" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # load minute prices spanning all prompts + their realization
    ps_min = min(prompt_starts); ps_max = max(prompt_starts) + timedelta(seconds=tl)
    prices_bt = download_price_data(ps_min - timedelta(hours=2), ps_max + timedelta(hours=2), args.asset, freq="1min")
    n = 0
    for t in prompt_starts:
        t = pd.Timestamp(t)
        rp = _slice_real_prices(prices_bt, t, tl, ti)
        if len(rp) != tl//ti + 1:
            continue
        if not np.isfinite(pd.to_numeric(pd.Series(rp), errors="coerce")).all():
            from synth_lib.preparation.realized_path_store import RealizedPathStore
            rps = RealizedPathStore(args.asset, tl, ti).get(t)
            if rps is not None and len(rps) == tl//ti+1:
                rp = list(rps)
        seed = (np.random.default_rng().integers(0, 2**31) if args.seed_mode == "random" else 0)
        paths = oracle_simulate(args.asset, t, ti, tl, args.num_simulations, rp,
                                rng_seed=seed, mode=args.mode, df=args.df, tilt=args.tilt)
        payload = {"start_timestamp": t.isoformat(), "asset": args.asset, "time_increment": ti,
                   "time_length": tl, "num_simulations": len(paths), "num_steps": len(paths[0]),
                   "paths": [list(map(float, p)) for p in paths]}
        (pred_dir / (t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{args.asset}_{tl}.json")).write_text(json.dumps(payload))
        n += 1
    print(f"generated {n} oracle predictions", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = backtest(miner_name="ceiling", asset=args.asset, time_length=tl, time_increment=ti,
                       n_backtest_days=args.days, predictions_dir=pred_dir, eval_end=end, miner_id=999)
    print(f"summary: {res.summary}")
    ss = res.smoothed_scores
    if ss.empty: print("no smoothed scores"); return
    last_ts = ss["updated_at"].max(); last = ss[ss["updated_at"] == last_ts]
    me = last[last["miner_uid"] == 999]
    if me.empty: print("not in final round"); return
    rank = int((last["reward_weight"] > me["reward_weight"].iloc[0]).sum() + 1)
    print(f">> ORACLE RANK {rank}/{len(last)} at {last_ts} (rw={me['reward_weight'].iloc[0]:.6f} smoothed={me['new_smoothed_score'].iloc[0]:.3f})")
    recent = ss[ss["updated_at"] >= ss["updated_at"].max() - timedelta(days=2)]
    ranks = []
    for tg, g in recent.groupby("updated_at"):
        m = g[g["miner_uid"] == 999]
        if not m.empty: ranks.append(int((g["reward_weight"] > m["reward_weight"].iloc[0]).sum() + 1))
    if ranks:
        print(f">> recent {len(ranks)} rounds: median={int(np.median(ranks))} min={min(ranks)} max={max(ranks)}")


if __name__ == "__main__":
    main()
