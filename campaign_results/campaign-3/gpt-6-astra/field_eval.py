"""Full-path official backtests and cross-asset validator reward aggregation."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd

from predict import AGENT_DIR, WORKSPACE, backtest, ensure_workspace_cwd, generate, parallel_combined
from research import CRYPTO, EQUITY, WINDOWS, features, generate_variant
from synth_lib.backtester.config import competition_for, slug_for
from synth_lib.backtester.result import BacktestResult
from synth_lib.backtester.scoring import compute_combined_smoothed_scores


def variant_simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices, variant):
    feature = features(context_prices, start_time, time_increment)
    paths = generate_variant(feature, asset, pd.Timestamp(start_time), time_increment, time_length, num_simulations, variant)
    return (start_time, time_increment, *paths.tolist())


def evaluate(task):
    asset, length, window, variant, name, warmup_days = task
    ensure_workspace_cwd()
    begin, end = [pd.Timestamp(value, tz="UTC").to_pydatetime() for value in WINDOWS[window]]
    begin -= pd.Timedelta(days=warmup_days)
    increment = 60 if length == 3600 else 300
    label = f"{asset}_{length}"
    output = AGENT_DIR / "results" / name / window
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"{label}_summary.json").exists():
        return str(output), label
    predictions = WORKSPACE / "miner_outputs" / name / window / label / "predictions"
    kwargs = {} if variant == "model" else {"simulator": partial(variant_simulate, variant=variant)}
    total = generate(asset, begin, end, time_increment=increment, time_length=length,
                     num_simulations=1000, predictions_dir=predictions, **kwargs)
    print(f"Generated {total} {label} {window}", flush=True)
    result = backtest(miner_name=name, asset=asset, time_length=length, time_increment=increment,
                      n_backtest_days=(end-begin).days, predictions_dir=predictions, eval_end=end)
    result.prompt_df.to_parquet(output / f"{label}_prompts.parquet")
    result.smoothed_scores.to_parquet(output / f"{label}_smoothed.parquet")
    (output / f"{label}_summary.json").write_text(json.dumps(result.summary, indent=2))
    shutil.rmtree(predictions)
    return str(output), label


def reward_summary(frame):
    totals = frame.groupby("miner_uid").reward_weight.sum()
    latest = frame.loc[frame.updated_at.eq(frame.updated_at.max())].set_index("miner_uid")
    return {"reward_rank": int(totals.rank(ascending=False, method="min").loc[999]),
            "final_rank": int(latest.new_smoothed_score.rank(method="min").loc[999]),
            "miners": int(len(totals)), "reward_ratio": float(totals.loc[999] / totals.drop(999).max()),
            "reward_rounds": int(frame.updated_at.nunique()),
            "first_round": str(frame.updated_at.min()), "last_round": str(frame.updated_at.max())}


def aggregate(name, window, score_start=None):
    output = AGENT_DIR / "results" / name / window
    cutoff = None if score_start is None else pd.Timestamp(score_start,tz='UTC')
    competition_groups = {}
    for path in sorted(output.glob('*_summary.json')):
        label = path.name.removesuffix('_summary.json')
        asset, length = label.rsplit('_', 1)
        config = competition_for(asset, int(length))
        rounds = pd.read_parquet(output / f'{label}_smoothed.parquet')
        if cutoff is not None:
            rounds = rounds.loc[rounds.updated_at>=cutoff]
        result = BacktestResult(name, pd.read_parquet(output / f'{label}_prompts.parquet'),
                                rounds, json.loads(path.read_text()))
        competition_groups.setdefault(slug_for(config), (config, []))[1].append(result)
    summaries = {}
    for label, (config, results) in competition_groups.items():
        full_path = output / f'{label}_combined.parquet'
        combined_path = full_path if cutoff is None else output / f'{label}_from_{cutoff.value}_combined.parquet'
        if combined_path.exists():
            combined = pd.read_parquet(combined_path)
        elif full_path.exists():
            combined = pd.read_parquet(full_path)
        else:
            combined = parallel_combined(results, config)
        if cutoff is not None:
            combined = combined.loc[combined.updated_at>=cutoff]
        combined.to_parquet(combined_path)
        selected = combined if score_start is None else combined.loc[combined.updated_at >= pd.Timestamp(score_start,tz='UTC')]
        summaries[label] = reward_summary(selected)
        summaries[label]['assets'] = len(results)
        print(label, summaries[label], flush=True)
    summaries['score_estimate'] = 100 * sum(min(1, value['reward_ratio']) for value in summaries.values()) / 3
    (output / 'competition_metrics.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--variant', default='model')
    parser.add_argument('--windows', default='dev1,dev2')
    parser.add_argument('--assets', default=','.join(CRYPTO + EQUITY))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--warmup-days', type=int, default=0)
    parser.add_argument('--score-start')
    args = parser.parse_args()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    if not args.aggregate_only:
        tasks = [(asset, length, window, args.variant, args.name, args.warmup_days)
                 for window in args.windows.split(',') for asset in args.assets.split(',')
                 for length in ([86400, 3600] if asset in CRYPTO else [86400])]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(evaluate, tasks))
    for window in args.windows.split(','):
        print(aggregate(args.name, window, args.score_start), flush=True)


if __name__ == '__main__':
    main()
