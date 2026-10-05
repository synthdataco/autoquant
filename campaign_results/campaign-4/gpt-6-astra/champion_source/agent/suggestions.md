# Suggestions for the campaign infrastructure and future work

## Snapshot and evaluator consistency

The supplied environment did not set `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT`, and the advertised historical score/reward bundle was absent. Research initially used the official `synth_lib.backtester.scoring._score_single_prompt` through `predict.py`, which avoids inventing another metric. A local bundle was later recovered from explicitly allowed historical score/reward APIs. Every request ended on or before the data cutoff; no market prices or other miners' served paths were fetched. Full official orchestration was then run against exact validator prompts.

The historical APIs sometimes exceeded requested date boundaries and intermittently timed out. A defensive assertion caught an over-return before its score values were used. Recovery now filters every response to the requested interval immediately and caches individual pages. Consider enforcing bounds in synth-lib itself and shipping a complete immutable comparison bundle as promised.

The starter evaluator used strict `load_range` despite the constitution describing a lenient pattern. Its replacement reads the existing daily partitions, leaves NaNs intact, and checks coverage before any scoring. The exclusive end of a prompt window should not require an unnecessary next-day partition.

## Scoring-version clarity

The pinned library activates the new 1h volatility component on September22, after the snapshot cutoff. Historical orchestration therefore scores 1h price/gap CRPS, while replaying the current published formula adds volatility CRPS. Both are reported separately. A first-class scoring-version argument would be clearer than selecting a version through the internal scorer's scored-time argument. No observations from after September14 were accessed.

## Availability of fitted artifacts

A training cutoff needs an availability timestamp. The final model waits until the last training minute has closed and the prompt's issuance lead has elapsed before using a refreshed artifact. The harness should test this boundary in addition to checking context slices.

An entirely NaN close context cannot define the required initial price. The model raises an explicit error in that case; the serving interface should either guarantee at least one observed close or supply an independent starting price.

## Research left open

Scheduled announcements show elevated volatility in some historical events, but the tested adjustment was not robust across assets and horizons. More event-level samples and a genuinely untouched event holdout would be needed before deployment. Longer evaluation across market regimes would also help quantify uncertainty in the modest crypto improvements and distinguish artifact refresh effects from sampling noise.

The richer tree, neural, OHLCV, drift and residual-bootstrap models tested here did not reliably outperform the simple seasonal HAR mixture. Further work should start with a new independent validation period or a materially different information source, rather than retuning those failed families on these same weeks.
