# Suggestions (outside this campaign's scope)

- **A cadence-mode backtest that does not require field scores.** `backtest()` raises
  `NoScoresAvailable` for windows without recorded prompts even when the caller only
  wants CRPS, which caps official validation at the bundle's span (~2026-09-07 onward).
  A CRPS-only path (realised paths from the price store / cached realised arrays) would
  let miners validate on months of history instead of five weeks.
- **Document the vol-CRPS weight in trader units.** λ=5.25 reads as if block-vol
  calibration is a major 1h lever; measured on real fans it is ≈1% of the total score.
  One line in the competition docs would save every new miner a detour.
- **`load_minute_prices` semantics for the first context bar.** The context-end
  convention (bar labelled t closes at t+60s) is correct but only discoverable from
  `predict.py`'s docstring; a note in the benchmark docs would prevent a subtle
  one-step lookahead in home-grown miners.
- **Bundle span metadata.** The offline bundle's prompt span (per asset/competition)
  is only discoverable by querying it; a manifest entry would make backtest-window
  choices less trial-and-error.
- Nothing operator-side or foreign was encountered in the workspace; no unread
  material to report.
