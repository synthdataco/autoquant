
### Iteration 1: Baseline Evaluation
- **Hypothesis**: The starter log-normal model with 7-day constant volatility provides a baseline CRPS.
- **Change**: None (ran starter model on BTC for 1 day).
- **Result**: BTC mean CRPS ~2845.56.
- **Verdict**: Keep as baseline.
- **Lesson**: The evaluation loop works. Next, I need to improve volatility estimation since 7-day constant vol ignores volatility clustering.

### Iteration 2: Volatility Clustering and Distribution Tuning
- **Hypothesis**: The 7-day constant volatility baseline is suboptimal because it ignores short-term volatility clustering (especially relevant for the 1-hour horizon) and the fat-tailed nature of financial returns. Additionally, historical volatility overestimates future 24-hour realized volatility on these venues (due to weekend effects or mean-reversion).
- **Change**: 
  1. For 1-hour horizons (`time_length=3600`), replaced 7-day flat vol with a 2-hour EWMA (`span=120`) and used a Student-t distribution (`df=4`) instead of Normal.
  2. For 24-hour horizons (`time_length=86400`), retained the 7-day flat volatility but applied an asset-class specific scaling factor (0.85 for Crypto, 0.50 for Equities/Commodities) based on empirical backtesting on BTC and SP500, retaining the Normal distribution as Central Limit Theorem makes fat tails less prominent over longer horizons.
- **Result**: 
  - BTC 1H (crypto-1h): mean CRPS improved from ~628 to ~620, smoothed score improved from ~106 to ~98.
  - BTC 24H (crypto-24h): mean CRPS improved from ~2849 to ~2828, smoothed score improved from ~105 to ~94.
  - SP500 24H (com-equ-24h): mean CRPS improved from ~713 (unscaled baseline ~762) to ~697, smoothed score improved from ~62 (unscaled baseline ~103) to ~49.
- **Verdict**: Keep these changes. The combination of short-term EWMA/fat tails for 1H and scaled historical variance for 24H significantly outperforms the baseline across all metrics.
- **Lesson**: The 1-hour competition requires tracking high-frequency volatility regimes (EWMA) and accurately modeling the tails, whereas the 24-hour competition is best served by a heavily discounted long-term variance estimator (likely because continuous perp variance is consistently lower than standard historical close-to-close variance estimators on equities).
