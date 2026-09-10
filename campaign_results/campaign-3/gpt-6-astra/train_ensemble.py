"""Predeclared seed ensemble; never select members on development CRPS."""

import json

import numpy as np

from predict import AGENT_DIR, ensure_workspace_cwd
from train_vol import train


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    members = [dict(np.load(AGENT_DIR / 'volatility_weights.npz'))]
    for seed in [73, 420, 2026]:
        path = AGENT_DIR / f'ensemble_{seed}_weights.npz'
        if not path.exists():
            train(AGENT_DIR / 'results' / 'volatility_training.npz', path, seed=seed)
        members.append(dict(np.load(path)))
    np.savez(AGENT_DIR / 'ensemble_weights.npz', **{key: np.stack([member[key] for member in members]) for key in members[0]})
    (AGENT_DIR / 'ensemble_weights.json').write_text(json.dumps({'seeds': [20260909, 73, 420, 2026],
        'combination': 'equal-weight log multipliers', 'target_cutoff': '2026-07-26', 'validation_start': '2026-07-01'}, indent=2))


if __name__ == '__main__':
    main()
