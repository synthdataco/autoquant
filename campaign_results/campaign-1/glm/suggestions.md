# Suggestions (outside this campaign's scope)

## Did not encounter the operator's private research code

I did **not** find or read the operator's private research code `[private path]`
anywhere in my environment (no mount, copy, archive, or search hit). My work is built
entirely from public sources: `synth-lib`, `synth-subnet` (both installed via
`pyproject.toml` and read in the venv site-packages), the bundled offline
`market_data/offline_data` snapshot, and my own modeling. Reporting per the campaign's
hard rule 5: nothing to flag.

## Suggestions for the subnet / harness

1. **The CRPS-ranking is partly an RNG lottery — consider whether that's intended.**
   The decisive finding of this campaign: a perfectly-calibrated constant-σ Gaussian fan
   (verified by running a *cheating* oracle — σ set to each prompt's true realized 5m vol —
   through the official backtester) ranks only ~160-228/256. The podium is won by miners
   whose RNG happened to draw a narrow fan on a calm day. Running 40 seeds of the identical
   oracle reproduces the field's per-prompt p10/p50 almost exactly. If the subnet's goal is
   to reward *calibrated distributional forecasting*, the current softmax(β=-0.15) over
   prompt-scores (crps − min) lets sampling noise dominate modeling quality on the 24h
   formats. A rank/weight derived from CRPS *levels* with less sensitivity to the
   per-prompt minimum, or more simulations per prompt, would make calibration pay more
   reliably. (The 1h format, with its 12 cumulative "gaps" intervals, already rewards
   calibration far better — my model ranks #1 across many consecutive rounds there.)

2. **`response_validation_v2` silently rejects over-precise prices.** The 8-significant-
   digit rule (`float(f"{v:.7e}") == v`) is a scientific-notation rule, not a decimal-place
   rule. A miner that naively rounds to, say, 8 *decimal* places will have every path
   rejected as "too many digits" → CRPS -1 (dead last) with no message that points at the
   real cause. A clearer rejection message (e.g. "exceeds 8 significant figures") would
   save future miners the debugging I had to do.

3. **Scored-prompt `start_time` vs the backtest window.** Prompts scored on day N of a
   backtest window were *started* on day N-1 (start_time = scored_time − time_length). A
   miner that generates predictions only for start_times within [window_start, window_end]
   silently misses ~the first day's prompts → CRPS -1 (filled at p95) → an artificially bad
   rank that does not reflect production (where `simulate` answers every prompt at its exact
   time). The starter `predict.py` and the campaign guidance could make this explicit so
   miners don't mistake the artifact for a model problem (I did, briefly).

4. **SP500 introduction mid-window.** SP500 carries the highest coefficient (3.44) but was
   only scored from ~2026-07-19, and its backfilled price series spans the full snapshot.
   A miner can fit on the whole series but must validate its SP500 *rank* only on the
   post-introduction weeks — and even then the available scored history is short (one day
   in this snapshot). More SP500 scored history would let miners tune the single most
   valuable asset properly.

## Tooling I left in `agent/` (for reuse)

- `eval_harness.py` — faithful local CRPS + rank harness (validator's functions), any
  prompt grid / nsim, for fast iteration.
- `oracle.py`, `oracle2.py`, `shape_oracle.py`, `bootstrap_test.py`, `drift_test.py`,
  `vol_opt.py` — the diagnostic suite that produced the "field is a constant-σ Gaussian
  sea / podium is seed luck" conclusion and the vol-estimator / tilt sweeps.
- `validate.py` — official-rank batch validation across assets & regime windows.
- `ceiling.py` — runs a cheating oracle through the official backtester to measure the
  achievable rank ceiling for a given competition.
