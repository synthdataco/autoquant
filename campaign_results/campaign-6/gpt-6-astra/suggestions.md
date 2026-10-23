# Suggestions for the benchmark and future work

- The campaign says BUDGET.md is at the workspace root; this run's harness writes agent/BUDGET.md. Align the documented location.
- Keep the raw-context issuance cutoff explicit. The supplied evaluator originally took the last 10080 rows before issuance; the official generator uses the seven calendar days ending at issuance. This run aligns with the official generator.
- Daily NumPy predictions substantially reduce serialization overhead compared with one JSON file per prompt. The official loader already supports them; make this the default in the starter evaluator.
- Calendar data sometimes co-locates GDP and PCE. Treat their effects jointly instead of estimating independent additive effects from the same release observations.
- DATA.md omits the measured scored-prompt span described in CAMPAIGN.md. This bundle starts on September 8, 2026 at approximately 23:00 UTC, despite cached realized paths starting in June. Explicitly report both spans.
- Official backtest raises KeyError('miner_uid') when a one-prompt window leaves no smoothed-score rows. It should return CRPS with an unavailable rank, or raise a coverage error before prediction generation.

- Expose a supported CRPS-only mode using cached validator prompt schedules when field scores are unavailable. This campaign verified exact parity through the internal official scoring worker, but a public API would remove reliance on private internals.
- Preserve the official scorer's component totals in backtest output. The validator already computes price-interval and volatility details, but the backtester worker discards them. Keeping them would make distribution-model failures easier to diagnose without changing the metric.

- Report raw configuration CRPS and mean per-asset percentage changes separately. This campaign found opposite signs for those summaries on the same additional-period experiment; a single column labeled improvement can conceal the distinction.
- Encourage matched training dates, feature transforms, and complete-reference origin sets when comparing proxy targets with official-loss supervision. Otherwise a target comparison also changes the data policy.
- Record checkpoint revisions, forecast-cache source hashes, and successful prerequisite gates in durable manifests. A predecessor process exiting is not evidence that its output is valid.
- Expose an official exact-origin CRPS-only training/evaluation API. Finite-difference loss supervision here reused the official worker through predict.py; a supported API would make this workflow easier to reproduce without private imports.
