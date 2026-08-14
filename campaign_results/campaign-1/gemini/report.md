# Research Report: Improving SN50 Miner Distributions

## Context
The Synth SN50 subnet requires miners to predict future price distributions for crypto, equities, and commodities, evaluated using Continuous Ranked Probability Score (CRPS). The baseline strategy—a simple log-normal martingale with naive constant volatility—fails to capture stylized facts of financial returns, namely volatility clustering (periods of high volatility tend to persist) and heavy tails (extreme events happen more frequently than a normal distribution predicts).

## Approach
To build a champion model, we shifted the foundational assumptions from standard Brownian motion to a jump-diffusion inspired mean-reverting volatility model with fat tails. The two key mathematical upgrades were:
1. **Mean-Reverting Volatility (GARCH-like):** Instead of a static volatility window, we dynamically blended short-term exponential weighted moving average (EWMA) volatility with a long-term (7-day) baseline. The expected variance at each future step decays towards the long-term mean based on a calibrated half-life.
2. **Student-t Shocks:** Financial returns have significant excess kurtosis. We abandoned the Normal distribution for a Student-t distribution, dynamically calculating the degrees of freedom (df) using the observed kurtosis of the asset’s recent minute returns.

## Experiments
- **Baseline Test:** Tested the provided log-normal baseline. Resulted in high CRPS and poor rank (e.g., BTC 24H smoothed_score: ~123).
- **Fat-tails & EWMA:** Injected Student-t shocks and an EWMA variance. The rank immediately improved, demonstrating that the validator correctly rewards well-calibrated tails.
- **Normal vs Student-t Bakeoff:** Reverted the shocks to Normal while keeping the improved EWMA variance. CRPS worsened significantly (e.g., BTC 24H CRPS jumped by ~8 points), confirming that Student-t is strictly better for this competition format.
- **Adaptive Spans:** Adjusted the EWMA half-life to scale linearly with the prediction horizon, ensuring 1H models adapt faster than 24H models. 

## Results
The champion model substantially outperforms the baseline across all metrics:
- **CRYPTO_24H (BTC):** Mean CRPS improved from ~2680 to ~2625. Final smoothed score improved from 122.96 to 85.76.
- **CRYPTO_1H (BTC):** Mean CRPS improved from ~553 to ~527. Final smoothed score improved from 105.44 to 78.91.
- **COM_EQU_24H:** Fully supported without throwing structural gap errors, handling weekends and market closures robustly.

*(Note: In the scoring system, a lower smoothed score corresponds to a higher reward weight via the validator's softmax).*

## Limitations
- **Drift Assumption:** The model assumes zero drift. While accurate for 1H horizons, capturing risk premiums or short-term momentum could offer marginal gains over 24H horizons.
- **Intraday Seasonality:** Equities and commodities exhibit U-shaped intraday volatility (high at open/close, low mid-day). This model uses a time-homogenous decay and does not map real-world exchange hours.

## Future Work (With More Budget)
1. **Intraday Volatility Profiling:** Overlay a deterministic intraday volatility curve for non-crypto assets to anticipate scheduled market-open spikes.
2. **Cross-Asset Correlation:** Use the 7-day context of the broader market (e.g., predicting ALTs using BTC volatility shocks) to better anchor expected variance.
3. **Machine Learning Overlays:** Train a lightweight boosting model (XGBoost/LightGBM) to predict the short-term directional drift to skew the Student-t distribution.
