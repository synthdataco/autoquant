# Reproducing the campaign

Run every command from the workspace root, with the supplied snapshot mounted and the original environment available. Prices end on November 8, 2026. The environment variable `SYNTH_BENCHMARK_CALENDAR` supplies the event file; `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` supplies the historical field bundle. No price download is needed or permitted.

```bash
uv sync
```

The nominated implementation is `agent/modeling.py`. It requests seven days of raw minute OHLCV context and supports both 86400/300-second and 3600/60-second formats. It returns the requested number of paths, including their initial observed price. Randomness uses `SYNTH_BENCHMARK_SEED`, with research default 20261108. Use the same seed for paired comparisons; do not select a favorable seed.

## Historical scores

The 91-day comparison uses actual offline field prompt times, 1000 paths, and the current hourly formula. These commands write new result directories to preserve the campaign's saved evidence:

```bash
uv run agent/run_suite.py --model baseline --tag reproduce_starter --ends 2026-11-08 --days 91 --field --details --workers 6
uv run agent/run_suite.py --model modeling --tag reproduce_champion --ends 2026-11-08 --days 91 --field --details --workers 6
uv run agent/analyze_paired.py --baseline reproduce_starter --candidate reproduce_champion --out reproduce_paired
```

The commands default to all three configurations and 1000 paths. CRPS is computed only by `agent/predict.py` through the official `synth_lib.backtester`. The comparison script aggregates saved official scores; it does not implement CRPS. `--field` chooses actual timestamps but still requests CRPS-only evaluation; no field rank is needed for the objective.

A smaller four-window reproduction uses seven scored days per window, a 120-minute cadence, and a 17-minute offset:

```bash
uv run agent/run_suite.py --model modeling --tag reproduce_general --ends 2026-08-24 2026-10-12 2026-07-20 2026-11-08 --details --workers 6
```

Saved primary evidence is in `results/baseline_wide_field`, `results/candidate_calendar_wide_field`, `results/paired_candidate_starter_wide`, and `results/combined_checkpoint_general`. The serving module reproduced every one of the 6048 four-window candidate scores exactly. `results/research_registry.csv` indexes completed suites. The invalid first temporal-mixture run is explicitly marked by `results/temporal_mixture_general/INVALID.md` and is excluded from comparisons.

The final NumPy export was independently replayed in `results/portable_release_general`. `results/portable_score_parity.json` confirms exact equality of all 6048 official scores against the previous XGBoost-backed implementation.

Week-block intervals resample whole calendar weeks with all their assets and overlapping prompts. They are exploratory and do not adjust for model selection. No interval is reported for a single-week sample. A few weeks also offer weak uncertainty resolution.

## Serving and artifacts

The serving model needs these committed artifacts:

- `artifacts/seasonality.json`: historical RMS normalization and crypto future allocation.
- `artifacts/seasonality_abs5m.json`: separate commodity/equity future allocation.
- `artifacts/ml/manifest.json` and the corresponding `vol` arrays in `artifacts/ml_portable/`: causal one-hour and daily scale corrections.
- `artifacts/events.json` and `artifacts/closures.json`: causal forward calendar coefficients.

Each lookup selects the most recent fit no later than the context endpoint's date. Cutoffs on November 9 contain only data through November 8, with forward training labels purged before the cutoff. Earlier backtests select earlier fits. The production calendar is read from its environment path inside each call and can refresh at the same path. No historical price store, research feature cache, GPU, or network connection is required inside `simulate()`.

`results/release_portable_artifact_audit.json` records required file hashes, tracked status, and successful CPU loading of all retained portable forests. The original XGBoost weights remain in `artifacts/ml/` for provenance and research; XGBoost is required only for training/export, not serving. `export_portable_trees.py` converts the exact numeric forests without fitting, and `results/portable_tree_parity.json` verifies bitwise prediction equality with XGBoost on 49,152 rows. The full artifact folder also preserves rejected experiments for research reproduction; those alternatives are not selected by `modeling.py`. Serving options are explicit and do not depend on `SYNTH_RESEARCH_VARIANT`.

Existing verification scripts can be repeated with:

```bash
AUDIT_BLOCK_XGBOOST=1 AUDIT_OUTPUT=interface_release_check.json uv run agent/audit_interface.py
uv run agent/audit_calendar.py
uv run agent/audit_release_artifacts.py
uv run agent/summarize_research.py
uv run agent/validate_results.py
```

The interface audit blocks networking and checks all eighteen asset/configuration pairs, ordinary and gappy contexts, seeds, counts, unchanged inputs and latency. The calendar audit also covers missing schedules and atomic/in-place refreshes that preserve modification time.

## Training provenance

The retained model was built with `fit_seasonality.py`, `make_ml_data.py`, `train_ml.py`, `fit_events.py`, and `fit_closures.py`. Their source and manifests record the fitting windows and hyperparameters. The original volatility-only trees use depth three, separate hourly/daily targets, a 240-day training lookback, and chronological validation with label purging. Seasonal fits use 120 preceding days; event fits use 365. Training caches are regenerable and intentionally excluded from git; trained artifacts are committed. Do not overwrite the archived artifacts when reproducing a new training experiment.

`journal.md` preserves preregistered hypotheses, thresholds, corrections and verdicts. `report.md` is the public explanation, including failures, sparse-context limitations and the distinction between broad traffic gains and conditional event gains.
