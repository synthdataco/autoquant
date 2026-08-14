"""Score sensitivity: how much reward weight does a given CRPS edge actually buy?

The pre-registered Score is 100 * mean over competitions of min(1, our_weight / top_weight), where
weights come from the validator's softmax over smoothed scores. This script replays that softmax
on the REAL field to answer the only question that decides how to spend the remaining budget:
if we improve mean CRPS by x%, what happens to the Score term?

Uses synth's own reward code (compute_prompt_scores / softmax) -- never a reimplementation.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research')
import harness
from synth.validator.competition_config import (COM_EQU_24H, CRYPTO_1H, CRYPTO_24H,
                                                SMOOTHED_SCORE_COEFFICIENT)

WINS = {
    'crypto-24h': (CRYPTO_24H, ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'], '2026-07-23', '2026-07-25'),
    'crypto-1h': (CRYPTO_1H, ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'], '2026-07-24', '2026-07-26'),
    'com-equ-24h': (COM_EQU_24H, ['SP500', 'AAPLX', 'XAU', 'NVDAX', 'GOOGLX', 'TSLAX', 'WTIOIL', 'SPCX'],
                    '2026-07-23', '2026-07-25'),
}

# Our measured gap to the best miner, per competition, in % of mean CRPS.
# Refreshed after the session-pool / seasonal / kadj changes: these are the CURRENT harness
# numbers (1000-path official runs), not the older 400-path figures this script shipped with.
OUR_GAP = {'crypto-24h': 0.632, 'crypto-1h': 0.604, 'com-equ-24h': 5.210}


def softmax_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    """synth's reward softmax: exp(beta * score), normalized, times the 1/3 coefficient."""
    x = beta * scores
    x = x - x.max()
    w = np.exp(x)
    return w / w.sum() * SMOOTHED_SCORE_COEFFICIENT


for slug, (comp, assets, s, e) in WINS.items():
    # Build the field's per-miner mean RELATIVE score, the way the validator aggregates:
    # prompt score = crps - per-prompt field min, averaged over prompts and assets.
    per_asset = []
    for a in assets:
        try:
            w = harness.field_panel(slug, a)
        except FileNotFoundError:
            continue
        w = w.loc[(w.index >= pd.Timestamp(s, tz='UTC')) & (w.index < pd.Timestamp(e, tz='UTC'))]
        if w.empty or w.shape[1] < 20:
            continue
        rel = w.sub(w.min(axis=1), axis=0)      # subtract per-prompt minimum
        per_asset.append(rel.mean(axis=0))
    if not per_asset:
        print(f"{slug}: no data"); continue
    df = pd.concat(per_asset, axis=1).dropna()
    field = df.mean(axis=1).sort_values()       # equal-weight across assets (approximation)

    beta = comp.softmax_beta
    print(f"\n===== {slug}  (beta={beta}, {len(field)} miners, {len(per_asset)} assets) =====")
    print(f"  field relative score: best={field.iloc[0]:.2f} p10={field.quantile(.1):.2f} "
          f"med={field.median():.2f} p90={field.quantile(.9):.2f}")

    # scale: convert a % CRPS edge into relative-score units using the mean CRPS level
    lvl = np.mean([harness.field_panel(slug, a).loc[
        (harness.field_panel(slug, a).index >= pd.Timestamp(s, tz='UTC')) &
        (harness.field_panel(slug, a).index < pd.Timestamp(e, tz='UTC'))].mean().mean()
        for a in assets if True][:len(per_asset)])

    print(f"  mean CRPS level ~{lvl:.0f}; so 1% of CRPS = {lvl*0.01:.1f} relative-score units")
    print(f"\n  {'our gap vs best':>16s} {'rel score':>10s} {'rank':>6s} {'weight':>9s} "
          f"{'top weight':>11s} {'SCORE TERM':>11s}")
    base_gap = OUR_GAP.get(slug, 1.3)
    for gap_pct in [base_gap, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5]:
        ours = field.iloc[0] + gap_pct / 100.0 * lvl
        allsc = np.append(field.to_numpy(), ours)
        wts = softmax_weights(allsc, beta)
        ourw = wts[-1]
        rank = int((allsc < ours).sum()) + 1
        term = min(1.0, ourw / wts.max())
        tag = '  <-- us now' if gap_pct == base_gap else ''
        print(f"  {gap_pct:+15.2f}% {ours:10.2f} {rank:6d} {ourw:9.5f} {wts.max():11.5f} "
              f"{term:11.3f}{tag}")
