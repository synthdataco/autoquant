"""Audit saved official outputs for omissions, sentinels and invalid scores."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from run_suite import ROOT


def main():
    files, prompts, issues = 0, 0, []
    for suite in sorted((ROOT/'agent/results').glob('*/suite.json')):
        rows = json.loads(suite.read_text())
        if any('error' in r for r in rows):
            issues.append({'suite': suite.parent.name, 'error': 'failed subprocess in suite'})
            continue
        for path in sorted(suite.parent.glob('*.parquet')):
            summary = json.loads(path.with_suffix('.json').read_text())
            frame = pd.read_parquet(path)
            id_col = 'miner_id' if 'miner_id' in frame else ('miner_uid' if 'miner_uid' in frame else None)
            if id_col:
                frame = frame[frame[id_col] == 999]
            invalid = int((~np.isfinite(frame.crps.to_numpy()) | (frame.crps.to_numpy() < 0)).sum())
            if invalid or len(frame) != summary['num_prompts']:
                issues.append({'file': str(path.relative_to(ROOT)), 'rows': len(frame),
                               'summary_prompts': summary['num_prompts'], 'invalid_scores': invalid})
            files += 1
            prompts += len(frame)
    result = {'completed_result_files': files, 'official_candidate_scores': prompts, 'issues': issues}
    (ROOT/'agent/results/result_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
