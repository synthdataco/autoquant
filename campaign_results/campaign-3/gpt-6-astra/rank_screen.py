"""Official reward replay from scored candidates, with optional round thinning."""

import argparse
import json

import pandas as pd

from predict import AGENT_DIR, replay_prompt_scores, parallel_combined, ensure_workspace_cwd
from research import WINDOWS
from field_eval import reward_summary
from synth_lib.backtester.config import competition_for, slug_for
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.result import BacktestResult
from synth_lib.backtester.scoring import compute_combined_smoothed_scores


def main():
    ensure_workspace_cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('--template')
    parser.add_argument('--template-window')
    parser.add_argument('--score-start')
    parser.add_argument('--round-step',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--variants')
    args = parser.parse_args()
    frame = pd.read_parquet(AGENT_DIR/'results'/f'{args.experiment}.parquet')
    if args.variants:
        frame = frame.loc[frame.variant.isin(args.variants.split(','))]
    summaries = {}
    for window, window_frame in frame.groupby('window'):
        begin,end = [pd.Timestamp(value,tz='UTC') for value in WINDOWS[window]]
        fields = {}
        for (asset,length),group in window_frame.groupby(['asset','length']):
            increment = 60 if length == 3600 else 300
            field = get_miner_scores(begin,end,asset,int(length),increment)
            fields[(asset,length)] = field.loc[field.start_time.isin(group.time.unique())]
        for variant,variant_frame in window_frame.groupby('variant'):
            groups = {}
            for (asset,length),rows in variant_frame.groupby(['asset','length']):
                config = competition_for(asset,int(length))
                candidate = rows[['time','crps']].rename(columns={'time':'start_time'}).copy()
                candidate['scored_time'] = candidate.start_time + pd.Timedelta(seconds=int(length))
                candidate['miner_uid'] = 999
                candidate['asset'] = asset
                candidate['time_length'] = int(length)
                candidate['time_increment'] = config.time_increment
                scored = replay_prompt_scores(fields[(asset,length)],candidate)
                rounds = pd.DataFrame({'updated_at':[end-pd.Timedelta(seconds=1)]})
                if args.template:
                    source = AGENT_DIR/'results'/args.template/(args.template_window or window)/f'{asset}_{length}_smoothed.parquet'
                    rounds = pd.read_parquet(source,columns=['updated_at']).drop_duplicates().sort_values('updated_at').iloc[::args.round_step]
                if args.score_start:
                    rounds = rounds.loc[rounds.updated_at>=pd.Timestamp(args.score_start,tz='UTC')]
                result = BacktestResult(variant,scored,rounds,{})
                groups.setdefault(slug_for(config),(config,[]))[1].append(result)
            for slug,(config,results) in groups.items():
                combined = parallel_combined(results,config,workers=args.workers)
                if args.score_start:
                    combined = combined.loc[combined.updated_at>=pd.Timestamp(args.score_start,tz='UTC')]
                summary = reward_summary(combined)
                key = f'{window}/{variant}/{slug}'
                summaries[key] = summary
                print(key,summary,flush=True)
    suffix = 'sampled_ranks' if not args.template else f'template_step{args.round_step}_ranks'
    if args.score_start:
        suffix += f'_from_{args.score_start}'
    (AGENT_DIR/'results'/f'{args.experiment}_{suffix}.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
