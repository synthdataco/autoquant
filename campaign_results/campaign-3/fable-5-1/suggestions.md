# Suggestions (out of scope for the model itself)

1. **Verdict harness: first-day penalty.** `evaluate_candidate` fetches scored prompts whose
   `scored_time` falls in the window, but predictions are only generated for start times inside
   the window. The first 24 h of scored 24h prompts (start time before the window) therefore have
   no prediction file, are scored as `-1` and filled at the field's 95th percentile. In a 10-day
   verdict that is ~10% of the 24h prompts and a smoothed-score handicap of roughly
   0.1 × (p95 − min) ≈ 10–20 points, i.e. a factor exp(−0.15 × 15) ≈ 0.1 on the reward ratio
   for every candidate. Either generate predictions from `window_start − time_length` or restrict
   the scored prompts to `start_time >= window_start`. (`agent/evaluate.py` does the former by
   default and reproduces the handicap with `--verdict-exact`.)
2. **synth-lib `download_price_data` on a frozen store.** When the backtest window reaches the
   day after the last partition (always the case for the 1h competition with `eval_end` = the
   last snapshot day, because it asks for `start_time + 26 h`), it calls `ingest_range` and fails
   on the read-only mount. A read-only/lenient mode (missing days → NaN) would make the 1h
   backtest usable up to the cutoff. Patched locally in `agent/evaluate.py`.
3. **Speed of the 10-day crypto-1h backtest.** `calculate_smoothed_scores` loops over every
   reward round (~430/day) with a pandas groupby each; for 10 days × 5 assets this dominates the
   run time (≈ 4–5 min per asset, single-threaded under the GIL). Vectorising by round or caching
   the per-round window slices would cut the verdict wall-clock substantially.
4. **SPYX** is still in `COM_EQU_24H.asset_list` and in the offline bundle (empty parquet); the
   harness skips it correctly, but a dead asset in the config costs every newcomer a puzzled hour.
5. **Documentation of the scoring intervals.** The fact that `24hour_abs` reduces to a single
   term (the final price) while `5min` contributes 288 terms is the single most useful fact for a
   new miner and is not stated anywhere; a one-line note in the validator README would help.
6. **Prompt-density mismatch in the verdict grid.** The 1h verdict grid is 10-minute aligned while
   real prompts fall on arbitrary minutes (matched within 30 min). For the 24h format the same
   applies with hourly grid points: the 24h absolute-price term is then evaluated against a
   realized path that starts up to 30 min away from the candidate's start price — a small,
   systematic handicap relative to the live field. Generating on the actual scored start times
   (available in the bundle) would remove it.
7. No private operator code was encountered anywhere in the environment.
