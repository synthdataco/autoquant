# Suggestions for the Synth research tooling

- Expose an offline helper that lists the exact historical prompt start times for a scored-date window. Prompt starts have irregular minute offsets; a regular prediction grid is insufficient for a faithful local backtest.
- Offer an exact-match mode in the backtester, or warn when a prediction with a timestamp later than the prompt is matched by the current time tolerance. A later match can leak future context into a historical test.
- Provide a lenient OHLCV context loader that preserves no-trade NaN minutes. `MinutePriceStore.load_range` correctly rejects gaps for strict scoring, but exploratory model generation needs the raw grid. The starter evaluator's prose described a lenient load even though its original implementation called the strict loader.
- Document that score-query date bounds include both calendar dates in the offline historical cache. A query from August 15 to August 16 returns two scored days; this is easy to mistake for one day in research summaries.
- Make the backtester's price-read buffer depend on the prompt horizon, or fail before ingestion when its current `latest start + 26 hours` buffer extends beyond a frozen snapshot. The unconditional 26-hour read also affects 1-hour prompts and caused a September 15 ingest attempt during a September 14 holdout.
