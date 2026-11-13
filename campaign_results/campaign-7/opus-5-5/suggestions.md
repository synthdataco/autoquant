# Suggestions (outside this campaign's scope)

Written for the Synth miner community and the benchmark operators.

1. **Benchmark against the best single miner, not the per-prompt p10.** The per-prompt 10th
   percentile mixes different miners on every prompt, so no individual miner reaches it: over
   2026-08-05..08-26 the best single miner with full coverage was +2.1% above it in crypto-1h,
   +1.1% in crypto-24h and +1.2% in com-equ-24h. A dashboard column "CRPS vs the best single
   miner" would tell a miner where they stand far better than "gap to the best miner on each
   prompt" (which `agent/predict.py` prints, and which is +2-4% even for a frontier model).
2. **The top of the field is made of clones.** Dozens of uids score identically (to the 4th
   digit) on every asset. That is worth knowing when reading positions: a "top-10" position can be
   one model registered ten times.
3. **Hyperliquid equity/commodity perps have a near-dead weekend regime.** Saturday 5-min returns
   are exactly zero 42% of the time for XAU and 10-17% for the tokenised stocks, and weekend
   activity varies 10x from one weekend to the next. A mean-square seasonal profile overstates the
   typical weekend by ~2.5x for XAU; a mean-absolute profile is much closer. Miners pricing these
   assets should treat Friday 17:00 -> Sunday 18:00 New York as its own regime.
4. **Binance spot BTC prints the same 1-minute close twice in a row 5.6% of the time** (15% on
   Saturdays) although its tick is 0.001 bp — staleness, not tick size. SOL and XRP sit on a
   coarse grid (1.16 bp and 0.9 bp ticks at the snapshot's prices): rounding simulated prices to
   that grid measurably lowers their 1-minute CRPS.
5. **Event calendar.** The calendar is valuable (CPI-spanning 1h prompts: -6% to -10% CRPS with
   release multipliers). Useful additions: Fed chair speeches / press-conference times (the FOMC
   effect peaks 30-60 min after the statement), ECB decisions for XAU, OPEC meetings for WTIOIL,
   and the earnings session (before/after the open) for every company — several rows say
   "session not given".
6. **The 1h volatility term is ~1% of a 1h score.** With VOL_CRPS_LAMBDA = 5.25 its weight is
   small next to the 1-min/2-min/gap price terms; if its intent is to reward realistic path
   roughness, its weight could be raised.
7. **Constitution rule 6 check.** I found no operator miner model in the workspace or the
   installed packages and read nothing of that kind. synth-lib ships a `synth_default` baseline
   under `synth_lib/benchmark`; I did not open it.
