# Suggestions for the Synth Subnet

During our research campaign, we identified several structural opportunities to improve the developer experience and performance of the Synth subnet ecosystem:

## 1. Native `lenient=True` Flag in `synth-lib` Data Loading
Currently, `MinutePriceStore.load_range` strictly asserts that no NaNs or missing minutes exist in the range, raising a `ValueError` if any are found. However, traditional assets (such as equity perps on Hyperliquid) naturally have gaps or periods of zero trading volume, making NaNs standard in historical snapshots.
* **Suggestion**: Expose a `lenient=True` parameter in `MinutePriceStore.load_range()` (and other loader utilities) in the public `synth-lib` package. This would prevent new developers from running into cryptic loading failures and having to implement custom pandas-parquet readers during local validation.

## 2. Intraday Session-Based Evaluation
Traditional assets (XAU, WTIOIL, SP500, TSLA, AAPL, NVDA, GOOGL) have clear volatility spikes around exchange opening and closing hours, and remain flat/thin during weekends or US overnight sessions. Since the validator queries simulations 24/7, miners are forced to treat volatility as flat or guess the session.
* **Suggestion**: Incorporate session-of-day seasonal profiles directly into the validator's reference models, or evaluate CRPS using time-weighted scoring where active trading sessions are heavily weighted compared to flat overnight sessions.

## 3. Multivariate Path Scoring
Currently, the validator scores each asset independently. However, financial markets exhibit strong co-movements and systemic correlation (e.g., BTC and ETH, or SP500 and AAPL/GOOGL). A miner that produces perfectly correlated paths represents a much more realistic simulation than a miner whose asset paths are independent.
* **Suggestion**: Introduce multivariate CRPS or copula-distance scoring that rewards miners who accurately capture cross-asset correlations and joint tail probabilities.
