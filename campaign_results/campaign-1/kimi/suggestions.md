# Suggestions — out-of-scope findings from the bakeoff-2 campaign (kimi)

Things I ran into that are outside my mandate as a miner-model researcher but look
actionable for the Synth team / synth-lib maintainers.

## 1. synth-lib backtester: percentile90/95 backfill-key mismatch distorts field ranks

While building a field-relative evaluator I found that the rank the synth-lib
backtester reports is systematically distorted. In
`synth_lib/backtester/scoring.py` (and the matching logic in `orchestration.py`),
field prompt scores are backfilled for miners with missing prompts using a
percentile of the *valid* scores — but the percentile key used for the backfill
(`percentile90` / `percentile95`) does not match the key the validator itself uses
when it fills invalid/missing prompts in `synth/validator/moving_average.py`
(p95 of valid scores, per prompt). The result: miners with partial coverage are
ranked too harshly or too leniently depending on the window, and the backtester's
"your rank in the field" number can move by several positions purely from this
artifact rather than from model quality. I worked around it by re-implementing the
ranking step (prompt_score = crps − per-prompt best valid, invalid → p95 fill,
summed over the window, per-asset coefficients, softmax) on top of the official
`calculate_crps_for_miner` CRPS values, but it would be better fixed upstream so
all bakeoff participants compare on the same footing.

## 2. SP500 field-score history starts only ~2026-07-19/24 in the offline snapshot

SP500 replaced SPYX around 2026-07-17 carrying the highest COM_EQU_24H coefficient
(3.44). Its *price* series in the snapshot is backfilled over the full window, but
field scores and realized scoring prompts exist only from ~2026-07-19 (and the
offline field-score parquet I used only covers it from ~2026-07-24). Anyone
validating a COM_EQU_24H model on field-relative rank before that date is
silently optimizing without the single highest-weighted asset. Worth a loud
warning in synth-lib when a backtest window predates an asset's scoring start.

## 3. Minute-price snapshot has a few percent NaN closes even on top assets

Not a bug per se (the campaign brief says to feed gaps, and scoring drops NaN per
prompt), but a documented per-asset coverage table shipped with the snapshot would
save every participant an hour of discovering which days are unusable at eval
window edges.

## 4. No trace of the operator's private research code

For the record: I never came across the subnet owner's private research repo in
any form (mount, copy, archive, search result). Nothing to report under hard
rule 5.
