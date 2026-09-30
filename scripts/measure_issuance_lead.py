"""Does the verdict hand the champion a minute the live miner never had?

The validator issues a prompt at `request_time`, which is 60 seconds BEFORE the `start_time` it
scores from, and `real_prices[0]` is the price AT start_time. A live miner therefore answers
without knowing the first point it is scored on. Generation gives the champion
`price_frame.loc[t - 7d : t]` — inclusive of the candle at t — so the benchmark hands it exactly
that point.

This regenerates a sample of prompts both ways against the same realized path:

    full    context up to t         (what the verdict gives the champion today)
    lagged  context up to t - lead  (what the deployed miner could actually have)

and reports the CRPS ratio. If lagged/full matches the champion-vs-live CRPS gap measured from a
verdict's prompt_scores frame, the issuance lead is the explanation.

Both arms draw the same random numbers for a given prompt (the champion seeds from the prompt's
own identity, and PYTHONHASHSEED is pinned by the verdict runner), so the comparison is PAIRED and
the per-prompt ratio is the statistic. A mean of those ratios is not: they are heavy-tailed, and a
mean over a few hundred prompts moved by an order of magnitude between samples. The median and the
sign test are reported with intervals; the mean is kept only for continuity with older runs.

`--stale-uniform` reproduces a live deployment instead of a fixed cutoff: the serving store
refreshes on an interval, so at prompt time its newest bar is uniformly distributed in age across
that interval rather than being exactly one lead old.

    uv run python scripts/measure_issuance_lead.py \
        --leg campaign_results/campaign-3/fable-5-1 \
        --bundle offline_data/verdict_20260926_19d --asset BTC --prompts 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from synth.validator.crps_calculation import calculate_crps_for_miner, calculate_total_score_for_miner

from synth_lib.backtester.config import VOL_CRPS_1H_DATE, competition_for, slug_for
from synth_lib.benchmark.generate_predictions import CONTEXT_MINUTES, load_simulate

SEED_ENV = "SYNTH_BENCHMARK_SEED"
# The store labels a minute bar by its OPEN time (Binance k[OPEN_TIME]), so the row labelled t
# closes at t+60s. Generation's `loc[:t]` therefore ends on the close at t+60s, which is exactly
# real_prices[0]. A miner prompted at request_time = t-60s and answering a few seconds later has
# only the bar labelled t-120s, whose close is the price at t-60s. Two bars apart.
DEFAULT_LEAD_SECONDS = 120


def load_closes(data_root: Path, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Minute frame over [start, end] from daily partitions, whatever columns they carry.

    Deliberately not generate_predictions.load_minute_prices: that one requires the OHLCV columns,
    and a store written before the OHLCV migration is close-only. A champion that reads more than
    `close` will raise on its own.
    """
    root = data_root / "prices" / asset / "1m"
    frames = []
    day = start.date()
    while day <= end.date():
        path = root / f"date={day.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        day += pd.Timedelta(days=1).to_pytimedelta()
    if not frames:
        raise SystemExit(f"no partitions under {root}")
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index().loc[start:end]


def score(paths: np.ndarray, real: np.ndarray, comp, scored_time) -> float:
    """The validator's own CRPS, with the crypto-1h volatility term where it applies."""
    if comp.vol_scoring_blocks and pd.Timestamp(scored_time) >= VOL_CRPS_1H_DATE:
        total, _ = calculate_total_score_for_miner(
            paths, real, comp.time_increment, comp.scoring_intervals, comp.vol_scoring_blocks
        )
    else:
        total, _ = calculate_crps_for_miner(paths, real, comp.time_increment, comp.scoring_intervals)
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", type=Path, required=True, help="published leg holding modeling.py")
    ap.add_argument("--bundle", type=Path, required=True, help="offline bundle with miner_scores_*.parquet")
    ap.add_argument("--data-root", type=Path, default=Path("market_data"))
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--time-length", type=int, default=3600)
    ap.add_argument("--prompts", type=int, default=40, help="sampled evenly across the bundle's range")
    ap.add_argument(
        "--lead-seconds",
        type=int,
        default=DEFAULT_LEAD_SECONDS,
        help="how far back to cut the context, in seconds before start_time (see DEFAULT_LEAD_SECONDS)",
    )
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument(
        "--stale-uniform",
        type=int,
        default=0,
        metavar="SECONDS",
        help="draw each prompt's extra staleness uniformly from [0, SECONDS] on top of --lead-seconds — "
        "what a serving store refreshed on that interval actually looks like, rather than a fixed cut",
    )
    ap.add_argument("--bootstrap", type=int, default=2000, help="resamples for the median's interval")
    args = ap.parse_args()

    comp = competition_for(args.asset, args.time_length)
    scores_file = args.bundle / f"miner_scores_{args.asset}_{slug_for(comp)}.parquet"
    if not scores_file.exists():
        sys.exit(f"no {scores_file}")
    scored = pd.read_parquet(scores_file, columns=["scored_time"]).drop_duplicates()
    scored_times = pd.to_datetime(scored["scored_time"], utc=True).sort_values()
    starts = scored_times - pd.Timedelta(seconds=args.time_length)
    picked = starts.iloc[np.linspace(0, len(starts) - 1, min(args.prompts, len(starts))).astype(int)]

    simulate = load_simulate(args.leg / "modeling.py")
    prices = load_closes(
        args.data_root,
        args.asset,
        picked.min() - pd.Timedelta(minutes=CONTEXT_MINUTES),
        picked.max() + pd.Timedelta(seconds=args.time_length),
    )
    step = args.time_length // comp.time_increment + 1
    lead = pd.Timedelta(seconds=args.lead_seconds)

    rng = np.random.default_rng(0)
    rng = np.random.default_rng(0)
    rows = []
    for t in picked:
        real = (
            prices["close"]
            .reindex(pd.date_range(t, periods=step, freq=pd.Timedelta(seconds=comp.time_increment)))
            .to_numpy(dtype=float)
        )
        if not np.isfinite(real).all():
            continue
        extra = (
            pd.Timedelta(seconds=int(rng.integers(0, args.stale_uniform + 1)))
            if args.stale_uniform
            else pd.Timedelta(0)
        )
        out = {"staleness_s": (lead + extra).total_seconds()}
        for label, cutoff in (("full", t), ("lagged", t - lead - extra)):
            ctx = prices.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : cutoff]
            sim = simulate(
                asset=args.asset,
                start_time=t.isoformat(),
                time_increment=comp.time_increment,
                time_length=args.time_length,
                num_simulations=args.num_simulations,
                context_prices=ctx,
            )
            paths = np.asarray([list(map(float, p)) for p in sim[2:]], dtype=float)
            out[label] = score(paths, real, comp, t + pd.Timedelta(seconds=args.time_length))
            out[f"{label}_anchor"] = float(ctx["close"].iloc[-1])
        out["start_time"], out["truth_at_start"] = t, real[0]
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no prompt had a complete realized path in the local store")
    df["ratio"] = df.lagged / df.full
    worse = int((df.lagged > df.full).sum())
    n = len(df)
    share = worse / n
    se = (share * (1 - share) / n) ** 0.5
    boot = np.median(rng.choice(df.ratio.to_numpy(), size=(args.bootstrap, n), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    stale = f"{args.lead_seconds}s" + (f" + U[0,{args.stale_uniform}s]" if args.stale_uniform else "")
    print(
        f"{args.asset} tl={args.time_length}: {n} prompts, context cut at t-({stale}), "
        f"{args.num_simulations} paths\n"
    )
    print("  PAIRED (both arms use the same draws for a prompt, so the per-prompt ratio is the statistic)")
    print(
        f"    median lagged/full  {np.median(df.ratio):.5f}   95% CI [{lo:.5f}, {hi:.5f}]"
        f"   ({(np.median(df.ratio) - 1) * 100:+.3f}%)"
    )
    print(
        f"    lagged worse on     {share:.1%} of prompts   95% CI "
        f"[{share - 1.96 * se:.1%}, {share + 1.96 * se:.1%}]   (50% = no effect)"
    )
    print(
        f"\n  mean lagged/full    {df.lagged.mean() / df.full.mean():.5f}  "
        f"({(df.lagged.mean() / df.full.mean() - 1) * 100:+.3f}%)  <- heavy-tailed, unstable across samples"
    )
    err = (df.full_anchor - df.truth_at_start).abs()
    print(f"\n  anchor error, full context:   {err.mean():.4f}  (0 = it is handed the first scored point)")
    print(f"  anchor error, lagged context: {(df.lagged_anchor - df.truth_at_start).abs().mean():.4f}")
    if args.stale_uniform:
        print(f"  staleness actually drawn: mean {df.staleness_s.mean():.0f}s  max {df.staleness_s.max():.0f}s")


if __name__ == "__main__":
    main()
