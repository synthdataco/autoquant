# Suggestions (for the campaign operators / synth-lib)

1. **Offline backtester bundle.** `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is not set in this
   environment. As a result, `predict.py` → `backtest()` queries the live Synth API for field
   scores/rewards.
   - The query extends 24h past `--eval-end`, so the agent has to keep
     `--eval-end ≤ cutoff − 1 day` (≤ cutoff − 2 days for CRYPTO_1H) to avoid touching
     post-cutoff data.
   - `--eval-end 2026-09-14` also fails: it tries to write the missing 09-15 partition into the
     read-only snapshot.
   - An offline bundle would remove both the lookahead risk and the fragility.
2. **`predict.py` cannot score COM_EQU_24H at all, nor HYPE.**
   - `MinutePriceStore.load_range` raises on any NaN minute close.
   - Every equity/commodity has NaN minutes in every 7-day window.
   - HYPE has a ~3-minute gap every Saturday (maintenance).
   - Suggest allowing NaN in the context load (the model contract already says "tolerate NaN,
     drop, never interpolate"), or providing a documented alternative scorer.
3. The docs could state explicitly that the realised paths are forward-filled. XAU has ~40–50%
   exact-zero 5-min changes on Saturdays, and equities have 4–16% overall. Miners otherwise have
   to discover this.
4. No operator miner models were found in the environment. None were read.
5. `SPYX` is still listed in some public docs as a scored asset, while the campaign says it is
   dead. A note in synth-lib's asset list would help.
