# Suggestions (outside this campaign's scope)

Written for the benchmark operators and for miners reading the published reports.

## Harness / synth-lib

1. **Thread oversubscription in parallel evaluation.** numpy's BLAS spawns one thread per core in every
   worker process. With 13 generation workers on 14 cores my dataset builder went from 3 minutes to
   "never finishes" (load average 207). `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` in the
   sandbox environment (or in `generate_all`) would protect every champion that uses `@`/`dot`.
2. **A fast, parallel `predict.py` should be the starter.** The starter evaluates one asset at a time and
   writes one 5.5 MB JSON per prompt. Going through `generate_predictions.generate()` + `score_crps()` in a
   process pool (what `agent/predict.py --assets all --jobs N` now does) scores 30 000 24h prompts in ~5 min
   and makes paired, per-prompt comparisons with the field routine. The per-prompt field join (best / median
   miner on the same prompts, paired rank of mean CRPS) was far more useful than the moving-average rank.
3. **`pkill -f` in an agent shell kills the agent's own command** when the pattern appears in its command
   line. Worth a line in the constitution's "lessons" list (use `pkill -f "[p]attern"`).
4. **Budget arithmetic.** The credit budget, not the 18 h clock, was binding by a wide margin (53 % of
   credits at 14 % of the time). Telling agents up front that a turn's cost grows with the transcript, and
   roughly by how much, would let them plan long GPU jobs around few, large turns from the start.
5. **Git identity.** The workspace has no `user.name`/`user.email`; the first commit fails. I set a
   repo-local identity (`campaign-7-agent`). `agent/BUDGET.md` is harness-owned and was untracked; a broad
   `git add agent` of mine committed a snapshot of it once, which I then removed from tracking
   (`git rm --cached`, the file itself untouched) and listed in my own `.gitignore` so the work tree reads
   clean. An entry for it in the scaffold's `.gitignore` would avoid this for every agent.
6. **Interface probe.** It calls `simulate()` on a constant-price context only. Adding one probe with a
   NaN-headed context and one with rows missing altogether would catch champions that silently collapse to a
   zero-width fan on sparse feeds (mine did on a context with no two adjacent prints until I added a
   fallback).

## Scoring rule (for the subnet)

7. **Ensemble sampling noise is paid by everyone.** With 1000 i.i.d. paths the expected CRPS excess is
   exactly 1/N = 0.1 % of the score — as large as the gap between the best and the 20th miner. Stratified
   shocks and quantile matching recover most of it; the "fair" (unbiased) ensemble CRPS in the validator would
   remove the incentive to engineer samplers instead of forecasts.
8. **Tick grid.** Realised prices live on the venue's tick grid; fans that are snapped to it score slightly
   better on quiet minutes (SOL/XRP at 1 min: -0.03 … -0.04 %). Harmless, but it is score for free.

## Data

9. **Scheduled weekly re-openings** (CME Sunday 18:00 New York, equity perps Sunday 20:00) behave like
   calendar events: the first minute after the WTI re-open moves 110 bps on average against 5 bps in the
   minutes before. They are deterministic, so they could be listed in the calendar next to NYSE closures
   (with the daily 17:00–18:00 CME break).
10. **Weekly jobless claims (Thursday 08:30 NY), ISM (10:00), JOLTS** are not in the calendar; a profile
    fitted outside the listed events still shows a residual 08:30 bump from them.
