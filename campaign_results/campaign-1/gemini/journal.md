# Journal

## Baseline
- Hypothesis: Basic log-normal martingale baseline.
- Change: None (run provided `modeling.py`).
- Result: 
  - BTC 24H (2026-07-15): mean_crps=2680.56, score=122.96
  - BTC 1H (2026-07-15): mean_crps=553.98, score=105.44
- Verdict: Baseline established.
- Lesson: Need to add fat tails and volatility clustering.

## GARCH-like + Student-t
- Hypothesis: Mean-reverting volatility and Student-t tails will better model the heavy tails and volatility clustering of crypto/equity returns.
- Change: Replaced flat local volatility with EWMA variance decaying to historical variance (half-life 6h). Swapped normal distribution for Student-t (df=5).
- Result: 
  - BTC 24H (2026-07-15): mean_crps=2630.47, smoothed_score=88.87 (Baseline: 122.96)
  - BTC 1H (2026-07-15): mean_crps=527.60, smoothed_score=79.50 (Baseline: 105.44)
  (Note: Lower smoothed_score = better rank)
- Verdict: Substantial improvement over baseline. Keep these features.
- Lesson: Volatility mean-reversion and fat tails are correctly modeling the empirical distributions. Next, need to differentiate per-asset and see if shorter half-life works better for 1H.
