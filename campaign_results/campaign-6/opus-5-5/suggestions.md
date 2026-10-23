# Suggestions (outside this campaign's scope)

## For the benchmark / harness
- **Context ends 2 minutes before the request** (`context_end`: lead + 1 min). Live miners hold prices up to the
  request time, so a backtested miner is handicapped against the field, and the handicap is largest on CRYPTO_1H, where
  the most recent minutes carry most of the information about the next hour. On 09-22..10-11 our c1 model is
  0.6–1.1% behind the best single miner per asset, while 24h equities are within ±0.7% (rank 1–68 of 242). Some of
  the c1 gap is probably this information difference, not the model. A field comparison with the same lead
  for everybody would make the 1h ranks more meaningful.
- The offline field bundle covers 2026-09-08..10-11 only, which leaves one 1h window (after the 09-22 formula change)
  and two 24h windows for validation. A longer bundle of realised validator paths would allow honest
  leave-one-event-out tests of the calendar effects with the official harness. Here the July–August releases had to
  be scored with synth's `calculate_total_score_for_miner` on our own paths (our-vs-our, no field).
- `predict.py` writes into one shared `miner_outputs/campaign` root. Concurrent runs of two *models* on the same
  asset would collide. A `--tag` or model-path argument would make A/B runs through the official tool safer.

## For miners (ideas I did not finish)
- **Event multipliers by asset class.** The calendar effect is much stronger for crypto 1h (−10% CRPS on CPI prompts)
  than for 24h equities. The GDP/PCE multipliers fitted on all assets *hurt* crypto 1h out of sample (+0.7–6%).
  Fitting per class and per release type, with more releases than one year gives, should help.
- **Gap at the cash open for tokenised stocks** (NVDAX, TSLAX, AAPLX, SPCX): at 24h horizons their standardised
  returns are under-dispersed (mean |z| ≈ 1.0 vs ≈ 0.8 expected) during Sep–Oct, but a per-path persistent drift did
  not help. An explicit overnight-gap jump component at 09:30 ET could.
- Holiday calendar (US market holidays) for the equity profiles: none fall in this eval window, so it was not built.

## Integrity notes
- No operator miner models or other participants' outputs were found in the environment.
- No market data or score/reward data after 2026-10-11 was queried. Event dates come only from
  `$SYNTH_BENCHMARK_CALENDAR`.
