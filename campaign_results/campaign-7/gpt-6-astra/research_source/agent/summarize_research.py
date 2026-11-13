"""Compact registry of completed suites, aggregating official saved CRPS only."""
from __future__ import annotations

import json
import pandas as pd
from run_suite import ROOT, CRYPTO, OTHER


def main():
    rows = []
    for file in sorted((ROOT/'agent/results').glob('*/suite.json')):
        summaries = json.loads(file.read_text())
        if any('error' in r for r in summaries):
            continue
        for name, horizon, assets in [('CRYPTO_24H', 86400, CRYPTO),
                                      ('COM_EQU_24H', 86400, OTHER),
                                      ('CRYPTO_1H', 3600, CRYPTO)]:
            subset = [r for r in summaries if r['time_length'] == horizon and r['asset'] in assets]
            if not subset:
                continue
            n = sum(r['num_prompts'] for r in subset)
            rows.append({'tag': file.parent.name, 'configuration': name,
                         'mean_crps': sum(r['mean_crps']*r['num_prompts'] for r in subset)/n,
                         'prompts': n, 'asset_windows': len(subset),
                         'ends': ','.join(sorted(set(r['eval_end'] for r in subset))),
                         'paths': ','.join(sorted(set(str(r['num_simulations']) for r in subset))),
                         'seed': ','.join(sorted(set(str(r.get('benchmark_seed', '20261108')) for r in subset))),
                         'context_days': ','.join(sorted(set(str(r['context_days']) for r in subset))),
                         'field_times': all(r.get('field_times', False) for r in subset),
                         'generation_seconds_per_prompt': sum(r['generation_seconds'] for r in subset)/n})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT/'agent/results/research_registry.csv', index=False)
    print('Saved', len(out), 'configuration results from', out.tag.nunique(), 'complete suites')


if __name__ == '__main__':
    main()
