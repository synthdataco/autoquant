# A horizon-aware volatility fan for Synth SN50

## Context

Synth miners forecast a distribution, not a point. The validator scores simulated price paths
with CRPS at several horizons, smooths scores over competition windows, combines assets with
unequal coefficients, and then applies a softmax. This makes calibration errors expensive: a
fan that looks plausible can earn essentially nothing if it is only a few percent behind a
compressed field.

I developed and tested one offline model for all three current competitions. Every reported
CRPS and rank came from the official `synth_lib.backtester`; prices came only from the frozen
minute snapshot through 2026-08-29. SPYX was excluded because it is retired.

## Approach

The final simulator is a zero-drift geometric random walk with several practical refinements:

- It estimates volatility at the requested native step (one minute for 1h, five minutes for
  24h), using only finite endpoint pairs and never interpolating no-trade gaps.
- The 1h profile uses six hours of standard-deviation history, Gaussian increments, and a
  0.85 volatility multiplier.
- The 24h profile uses one day of history by default. A pooled HAR log-variance forecast is
  used selectively for SOL, XAU, SP500, NVDAX, GOOGLX, TSLAX, and AAPLX.
- The 24h innovations are standardized Student-t with five degrees of freedom except for BTC,
  where Gaussian innovations validated better. Widths are clustered at 0.90, 0.70, or 0.50 by
  asset rather than freely optimized.
- Prompt-derived deterministic seeds and antithetic path pairs reduce finite-fan noise. The
  implementation honors arbitrary requested path counts and tolerates NaNs in every context
  column.

The pooled HAR fit normalized each asset by its long-run realized-variance median. On a held-out
period its log-variance RMSE was 0.995, versus 1.160 for last-day variance and 1.117 for a
seven-day mean. Its fitted intercept/day/3-day/7-day coefficients were
-0.152/0.461/-0.430/0.687. Better variance RMSE was not enough on its own, so HAR was retained
only where official CRPS improved on both validation splits.

## Experiments

The campaign ran 32 preregistered iterations. The most useful positive findings were exact
prompt alignment, native-step volatility, antithetic sampling, six-hour memory for 1h,
moderate Student-t tails for 24h, clustered widths, and selective HAR forecasts.

Several reasonable ideas failed honestly:

- Regular prediction grids produced invalid comparisons because synth-lib could match a file
  to a jittered prompt up to 30 minutes away. Generating at archived exact starts fixed this.
- EWMA, MAD, empirical return bootstrap, martingale correction, and longer fixed volatility
  windows did not generalize across regimes.
- Short-term drift and AR(1) increments produced inconsistent or negligible gains.
- Student-t innovations helped at 24h but missed the 0.5% adoption threshold at 1h.
- Narrower and wider adjacent width tests mostly failed on the full ten-day window. Only AAPLX
  justified moving from 0.50 to 0.70.

Three-day ranks were misleadingly optimistic. Expanding evaluation to the full smoothing
window moved crypto-24h from rank 154 to 203 at 200 paths, demonstrating why full-window replay
must supersede short-window selection.

## Results

Final nomination tests used 1000 simulations at every archived exact prompt start. The two
24h profiles cover ten scored days ending 2026-08-29; the 1h profile covers five days ending
2026-08-28 so synth-lib's realization prefetch remains inside the snapshot.

| Competition | Rank | Reward weight | Field leader |
|---|---:|---:|---:|
| CRYPTO_24H | 170/257 | approximately 1e-6 | approximately 0.0161 |
| COM_EQU_24H | 192/211 | 9.744e-10 | 0.0028358 |
| CRYPTO_1H | 237/257 | 1.880e-5 | 0.0041261 |

The model is not competitive with the best mature miners. Its value is a reproducible,
low-latency baseline and an audited map of what did and did not generalize. Moving from 200 to
1000 paths improved crypto-24h rank from 203 to 170, while COM/EQU remained nearly unchanged
(193 to 192) and 1h remained weak (240 to 237).

## Limitations

The selection windows are close to the cutoff, the field is tightly clustered, and numerous
asset-level comparisons increase selection risk despite two-split gates. The model is
conditionally homoskedastic within a path and contains no cross-asset, order-book, volume, or
intraday state. Sparse perpetual listings also make realized-volatility targets noisy. Finally,
historical rank is not a guarantee for the ten unseen evaluation days.

## What I would do with more budget

I would fit a strictly walk-forward distributional volatility model with partial pooling across
assets, optimize it against CRPS rather than variance RMSE, and validate on several nonoverlapping
full smoothing windows. A second priority would be regime-conditioned tail and leverage effects,
followed by a lightweight multivariate factor forecast for the highest-coefficient assets. Any
added complexity should still be tested against this fast deterministic baseline using exact
prompt starts and the official combined replay.
