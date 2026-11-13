# Suggestions

- Expose the official per-prompt price-CRPS and volatility-CRPS components in `BacktestResult.prompt_df`. The current total is sufficient for scoring, but a breakdown would make crypto-1h path-calibration experiments much easier to diagnose without modifying the backtester.
- Include a small machine-readable summary and per-prompt output option in the standard evaluation CLI. The `--result-stem` addition in this workspace is one possible interface; making it standard would help miners compare paired experiments without keeping multi-gigabyte prediction files.
- No operator-private miner models or other out-of-scope model artifacts were encountered or used.
