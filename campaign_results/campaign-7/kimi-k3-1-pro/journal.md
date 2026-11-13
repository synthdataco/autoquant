# Research journal

## Iteration 0 — Starter baseline and error structure

**Hypothesis.** A single full-context standard deviation with Gaussian log-normal increments is miscalibrated because volatility is clustered and strongly seasonal. Replacing it should reduce CRPS, especially on commodity/equity assets and crypto-1h.

**Change.** No model change yet. Installed the pinned environment with `uv sync`; read the validator CRPS and competition configuration. Ran official 1000-path backtests on the latest fully covered scored prompts (24h: 2026-11-05 to 2026-11-08; 1h: 2026-11-06 to 2026-11-08). Measured minute log-return volatility on 2026-08-01 to 2026-11-08 using only the snapshot.

**Result.** Initial mean CRPS: BTC-24h 3431.423 (72 prompts), XAU-24h 2057.948 (30), BTC-1h 519.622 (288, latest formula). The starter's mean gap to the best miner on each prompt was +13.49% on BTC-24h without events, +34.49% on XAU-24h without events, and +50.41% on BTC-1h without events. Remaining asset baselines are being collected under identical settings.

Snapshot diagnostics (minute RMS return, weekday/weekend RMS ratio, max/min New York hour RMS ratio): BTC 4.869 bp, 1.67, 2.22; ETH 6.709 bp, 1.35, 2.24; SOL 8.156 bp, 1.13, 1.94; XRP 9.630 bp, 1.01, 2.17; HYPE 10.170 bp, 1.35, 2.23; XAU 2.777 bp, 2.37, 4.18; NVDAX 4.732 bp, 2.94, 4.55; TSLAX 5.591 bp, 2.97, 6.24; SP500 1.557 bp, 1.78, 3.25; WTIOIL 6.579 bp, 1.65, 2.40.

Scoring facts that constrain the research: 24h scores 288 5-minute changes, 48 30-minute changes, 8 3-hour changes, and one terminal absolute price; short horizons therefore dominate. Crypto-1h adds volatility CRPS on 5-, 15-, and 60-minute blocks, so marginal calibration alone is insufficient there.

**Verdict.** Keep the starter only as the reference. Proceed with a seeded EWMA-volatility candidate, then add New York-time seasonality and non-Gaussian increments one at a time.

**Lesson.** Any candidate must explicitly use `time_length` and `time_increment`; crypto-1h requires realistic within-path volatility as well as calibrated marginal prices. Comparisons will use official CRPS at equal prompts and path counts, never field position as the objective.

**Pre-registered acceptance rule.** A modeling change must improve CRPS on two disjoint windows and in a majority of tested asset-window cells; improvements below 0.5% are treated as noise unless paired per-prompt results are consistent. Final candidates must be rerun at 1000 paths and pass the raw-NaN interface and latency checks.

### Complete starter table

Official mean CRPS at 1000 paths, latest window described above:

| Asset | 24h CRPS | 1h CRPS |
|---|---:|---:|
| BTC | 3431.423 | 519.622 |
| ETH | 4459.142 | 651.352 |
| SOL | 5097.799 | 804.124 |
| XRP | 5133.495 | 943.423 |
| HYPE | 6112.607 | 1009.453 |
| XAU | 2057.948 | — |
| NVDAX | 5884.456 | — |
| GOOGLX | 4457.109 | — |
| TSLAX | 5967.679 | — |
| AAPLX | 3068.570 | — |
| WTIOIL | 4507.370 | — |
| SPCX | 7355.830 | — |
| SP500 | 2122.382 | — |

## Iteration 1 — Fast EWMA variance

**Hypothesis.** A 30-minute EWMA of squared minute returns will track the current volatility regime better than the starter's equally weighted seven-day standard deviation. The variance proxy screen favored short half-lives on every tested asset, including at forecast leads up to 24h; this is a screening result, not evidence of a CRPS gain.

**Change.** Replace only the scale estimator with a 30-minute EWMA. Keep Gaussian log-normal increments, seven-day context, and the same path contract. Seed Python and NumPy generators from `SYNTH_BENCHMARK_SEED`, with a fixed fallback for local runs. Drop invalid returns across NaN gaps instead of differencing across them. Test BTC, XAU, and BTC-1h first on the starter window; retain only if the gain is material, then check a disjoint older window.

**Result.** On the latest window, mean CRPS was 3630.894 for BTC-24h (+5.81% versus starter), 2124.964 for XAU-24h (+3.26%), and 453.097 for BTC-1h (−12.80%). On the disjoint older windows, BTC-24h scored 2345.589 on 2026-08-29 to 2026-09-01, XAU-24h 1413.020 on the same dates, and BTC-1h 598.976 on 2026-09-29 to 2026-10-01 with the latest formula. The older BTC-1h no-event gap to best was +6.21%; the latest was +7.61%, down from +50.41% with the starter.

**Verdict.** Keep the 30-minute EWMA for 1h; discard it for 24h. The 24h candidate failed the latest-window kill threshold on both test assets. The next 24h experiment starts from the full-context estimator instead.

**Lesson.** A forward-variance proxy can favor a fast estimator without implying better multi-horizon CRPS. The 1h volatility term also benefits from a different scale dynamic than 24h price CRPS, so horizon-specific calibration is warranted.

## Iteration 2 — New York-time seasonality for 24h

**Hypothesis.** For 24h forecasts, a long-window variance estimate multiplied by a causal New York-time-of-day and weekday/weekend profile will beat both the starter and the fast EWMA, especially on XAU and tokenised equities.

**Change.** Restore full-context variance for 24h. Estimate mean-absolute-return profiles in 30-minute New York-time bins separately for weekdays and weekends, shrink each type's variance toward the global variance, clip and RMS-normalize the profile, and apply it to future Gaussian increments. Keep the accepted 30-minute EWMA for 1h unchanged. Test BTC, XAU, and TSLAX on latest and September windows.

**Result.** Latest-window mean CRPS: BTC 3315.298 (−3.38% versus starter), XAU 1988.006 (−3.40%), TSLAX 5641.566 (−5.46%). The corresponding no-event gaps to best fell to +6.05%, +15.41%, and +10.79%. September mean CRPS: BTC 2465.831, XAU 1237.493, TSLAX 1674.121. Relative to the fast-EWMA September run, BTC was 5.13% worse and XAU 12.42% better. The 1h regression check reproduced 453.096550 exactly.

**Verdict.** Keep seasonality provisionally for 24h, pending paired starter checks on older windows and broader asset coverage. Do not apply it to 1h yet.

**Lesson.** Local-time profiles can materially improve short-horizon calibration without changing the 1h code path. A latest-window gain is insufficient: BTC's September result shows that the variance level and seasonal decomposition interact by regime.

## Iteration 3 — Empirical 24h residuals

**Hypothesis.** Gaussian increments are too concentrated in the tails and insufficiently concentrated near zero. Snapshot 5-minute standardized returns have median absolute values of 0.25–0.46 versus 0.674 for a normal distribution, 99th-percentile absolute values of 3.24–4.10 versus 2.576, and material zero-change mass. Empirical residuals should therefore lower CRPS.

**Change.** Keep the Iteration 2 seasonal sigma path, but replace Gaussian 24h shocks with median-centered, winsorized overlapping 5-minute log-return residuals from the supplied context, scaled to unit variance. Keep the 1h Gaussian EWMA path unchanged. Use the same latest and September asset-window cells.

**Result.** Latest mean CRPS: BTC 3310.280 (−0.15% versus seasonal Gaussian), XAU 2007.430 (+0.98%), TSLAX 5777.505 (+2.41%). September: BTC 2382.976 (−3.36%), XAU 1199.396 (−3.08%), TSLAX 1591.989 (−4.91%). Latest XAU empirical residuals won only 33.3% of prompts; losses were concentrated in high-CRPS weekday prompts and gains in quiet Friday/weekend prompts. BTC's small mean gain hid a negative median paired difference, so it is not treated as a clean win.

Paired seeded-starter September results also completed the Iteration 2 check: starter BTC 2516.520, XAU 1478.323, TSLAX 2075.544. Seasonal Gaussian therefore improved those cells by 2.01%, 16.29%, and 19.34%, respectively. Seasonality passes both windows against the starter on all three assets.

**Verdict.** Discard the pure empirical distribution: it failed the latest-window kill threshold on two of three assets despite broad September gains. Keep seasonal Gaussian as the provisional 24h model and test a mixture rather than selecting by field-relative gaps.

**Lesson.** Raw empirical residuals match quiet-market cores but can understate volatile-weekday tails when the context mixes regimes. Mean CRPS and per-prompt gap-to-best can rank models differently when a few high-CRPS prompts dominate; the campaign objective is mean CRPS, so the mean regression controls the verdict.

## Iteration 4 — Gaussian/empirical residual mixture

**Hypothesis.** An equal mixture of the seasonal Gaussian and empirical residual distributions will interpolate between quiet-market core calibration and volatile-market tail coverage, improving the weaknesses identified in Iteration 3 without changing the sigma path.

**Change.** For each 24h increment, draw from the existing empirical residual with probability 0.5 and otherwise from a standard normal. Keep all scale estimation, clipping, centering, and 1h behavior fixed. Evaluate the same six asset-window cells.

**Result.** Latest mean CRPS: BTC 3301.220 (−0.43% versus seasonal Gaussian), XAU 1989.245 (+0.06%), TSLAX 5674.022 (+0.58%). September: BTC 2416.956 (−1.98%), XAU 1210.388 (−2.19%), TSLAX 1614.949 (−3.54%). The mixture improved four of six cells, was neutral within noise on latest XAU, and slightly exceeded the 0.5% noise threshold only on latest TSLAX.

**Verdict.** Keep the 50% mixture provisionally, subject to all-asset validation. It is more robust than pure empirical residuals, but not yet a clean all-cell win over seasonal Gaussian.

**Lesson.** Mixing distributions addresses the diagnosed core/tail tradeoff without introducing another scale parameter. Small latest-window differences still require broader validation rather than another mixture-weight sweep on three assets.

The accepted fast-EWMA 1h baseline was also broadened. Latest mean CRPS by asset, with improvement over the starter: BTC 453.097 (−12.80%), ETH 575.340 (−11.67%), SOL 733.410 (−8.79%), XRP 935.174 (−0.87%), HYPE 921.347 (−8.73%).

## Iteration 5 — Empirical residuals for crypto-1h

**Hypothesis.** Zero-inflated heavy-tailed empirical minute residuals will improve both price and realized-volatility CRPS relative to Gaussian increments on the accepted 30-minute EWMA scale.

**Change.** For 1h only, sample median-centered, winsorized one-minute returns from the context, standardized by their full-context standard deviation, and multiply by the current EWMA sigma. Leave the provisional 24h mixture untouched. Run all five crypto assets on the latest window and a disjoint 2026-09-29 to 2026-10-01 window with the latest formula.

**Result.** Latest mean CRPS, versus fast Gaussian: BTC 448.124 (−1.10%), ETH 570.580 (−0.83%), SOL 732.936 (−0.06%), XRP 933.164 (−0.22%), HYPE 917.992 (−0.36%). October: BTC 597.171 (−0.30%), ETH 731.836 (−0.16%), SOL 1054.419 (−0.11%), XRP 1195.947 (−0.12%), HYPE 1285.489 (+0.05%). Nine of ten cells improved, although only latest BTC and ETH exceeded 0.5%.

**Verdict.** Keep empirical 1h residuals. The signed consistency across assets and windows meets the pre-registered rule for small effects; no asset has a material regression.

**Lesson.** Once the current volatility level is estimated well, matching the residual shape is a second-order gain for 1h. It is still worth keeping because it is consistent and adds no serving complexity.

## Iteration 6 — EWMA half-life for 1h

**Hypothesis.** The accepted 30-minute half-life may be too slow for abrupt volatility changes or too noisy for stable periods. A 15-minute or 60-minute half-life could improve the empirical 1h model.

**Change.** Add a research-only environment override for the half-life, leaving the default at 30 minutes. Compare 15 and 60 minutes against the completed 30-minute results on BTC, SOL, and XRP in both latest and October windows. Broaden only a setting that wins consistently.

**Result.** At 15 minutes, latest mean CRPS was BTC 447.760, SOL 732.938, XRP 934.705; October was BTC 595.820, SOL 1054.883, XRP 1196.076. Differences from 30 minutes ranged from −0.23% to +0.17%, with no consistent asset or window winner. At 60 minutes, latest was BTC 450.827, SOL 735.788, XRP 932.841; October was BTC 601.227, SOL 1056.982, XRP 1198.595. The 60-minute setting regressed in five of six cells, by 0.68% on BTC in both windows.

**Verdict.** Keep 30 minutes. Do not broaden 15 minutes; discard 60 minutes.

**Lesson.** The 1h scale is already near a broad optimum. Sub-0.25% half-life changes are noise relative to the campaign's acceptance rule, and slower adaptation is consistently harmful on BTC.

### Broad 24h validation for Iteration 4

Across all 13 assets, the 50% mixture improved every September cell versus seasonal Gaussian. Relative gains ranged from 1.66% (HYPE) to 8.35% (NVDAX). It also improved 7 of 13 latest cells: all five crypto assets, AAPLX, and WTIOIL. Latest crypto gains were 0.36–0.73%; AAPLX gained 2.62%. The six latest commodity/equity regressions were 0.06–0.70%, except that they must still be monitored. This 20-of-26 cell record, with larger older-window gains than latest-window losses, keeps the mixture as the 24h champion candidate without per-asset selection.

## Iteration 7 — Early-fit event multipliers for 24h

**Hypothesis.** Scheduled CPI, NFP, FOMC, and earnings releases cause localized volatility that a generic intraday profile cannot fully capture. Conservative multipliers fitted on events before 2026-08-01 should improve later out-of-sample prompts without changing no-event behavior.

**Change.** Read only `$SYNTH_BENCHMARK_CALENDAR` inside `simulate()` and apply multipliers to 24h steps containing or following an in-scope event. Early-fit choices are CPI [3.0, 1.7, 1.3] at 0/5/10 minutes; NFP [1.8, 2.0, 1.1]; FOMC [2.0, 2.5, 1.75], then 1.4 through 60 minutes; earnings [3, 4, 4], 5 through 30 minutes, then 3 through 60 minutes. Exclude PPI, retail sales, GDP/PCE, and EIA because their early/late evidence was weak, unstable, or collinear. Cap the combined multiplier at 8. Validate on the late NFP window, September CPI/FOMC, October CPI/FOMC and earnings, and require no-event predictions to remain exactly unchanged. The 1h path is unchanged in this iteration.

**Result.** Calendar-on versus paired calendar-off mean CRPS: late NFP BTC 3302.084 versus 3301.220 (+0.03%), XAU 1988.711 versus 1989.245 (−0.03%); September CPI/FOMC BTC 2510.958 versus 2516.813 (−0.23%), XAU 1628.861 versus 1634.817 (−0.36%); October CPI BTC 2946.521 versus 2953.694 (−0.24%), XAU 1552.900 versus 1562.349 (−0.61%); isolated FOMC BTC 2410.903 versus 2413.535 (−0.11%), XAU 1466.365 versus 1469.733 (−0.23%); TSLAX earnings 4208.532 versus 4260.796 (−1.23%); GOOGLX earnings/FOMC 3889.265 versus 3964.111 (−1.89%). The no-event control reproduced 2416.9562832921156 exactly.

**Verdict.** Keep the calendar multipliers. Nine of ten late-validation cells improved; earnings had the largest effect, and the only regression was 0.03% on NFP BTC. No-event behavior is unchanged.

**Lesson.** Conservative, localized event scaling generalizes better than broad event-day inflation. The publishable negative result is equally important: EIA, retail sales, PPI, and collinear GDP/PCE did not have stable enough effects to include.

## Iteration 8 — Context length

**Hypothesis.** Seven days provides only two weekend observations and a noisy residual pool. A 14- or 30-day context may stabilize seasonal profiles and empirical distributions enough to improve CRPS, but may also make the variance level too slow.

**Change.** Compare `CONTEXT_DAYS = 14` and 30 against the completed seven-day no-calendar mixture on BTC, XAU, and TSLAX in latest and September windows. Disable the calendar for every arm so only context length changes. The 1h path is screened separately only if a longer context wins for 24h.

**Result.** Fourteen days, latest versus seven days: BTC 3305.475 (+0.13%), XAU 1991.164 (+0.10%), TSLAX 5670.702 (−0.06%). September: BTC 2356.002 (−2.52%), XAU 1207.757 (−0.22%), TSLAX 1574.620 (−2.50%). Thirty days, latest: BTC 3284.447 (−0.51%), XAU 2000.675 (+0.58%), TSLAX 5717.753 (+0.77%). September: BTC 2262.829 (−6.38%), XAU 1208.131 (−0.19%), TSLAX 1567.733 (−2.93%).

**Broad validation.** Across all 13 assets, 14 days was neutral on average in the latest window and improved 12 of 13 September cells; only SPCX-September regressed (2.64%). September gains were 0.19–8.29%. The 1h check was less favorable: latest differences were −0.16% to +0.32%, while October HYPE regressed 0.61% and the five-asset mean relative change was approximately +0.21%.

**Verdict.** Keep 14 days for 24h, but slice the final seven days inside `simulate()` for 1h so its validated residual pool and scale are unchanged. Discard 30 days: its larger older BTC gain does not offset two material latest-window regressions.

**Lesson.** More history helps 24h most in the older low-volatility regime, where seven days was too wide. The single module-level context length need not force both formats to use the same amount of history; an explicit shorter 1h slice avoids trading a 1h regression for a 24h gain.

## Iteration 9 — Sixty-minute block bootstrap for 1h

**Hypothesis.** Independent empirical increments preserve the marginal residual distribution but destroy volatility clustering and runs of zero changes. Sampling contiguous 60-minute residual paths should improve the 5-, 15-, and 60-minute realized-volatility CRPS without changing the intended marginal calibration.

**Change.** Add a research-only switch that samples each simulated path from one contiguous 60-minute context residual block, rather than drawing 60 independent residuals. Keep the 30-minute EWMA scale and all 24h behavior fixed. Compare against the independent-empirical results on all five crypto assets in latest and October windows.

**Result.** Latest 60-minute block CRPS, versus independent empirical: BTC 443.297 (−1.08%), ETH 562.116 (−1.48%), SOL 728.232 (−0.64%), XRP 935.250 (+0.22%), HYPE 915.849 (−0.23%). October: BTC 598.657 (+0.25%), ETH 735.618 (+0.52%), SOL 1056.507 (+0.20%), XRP 1193.429 (−0.21%), HYPE 1284.966 (−0.04%). The October mean relative change across assets was approximately +0.14%, so the long block failed the two-window rule despite four latest gains.

**Verdict.** Discard the 60-minute block. Test 15-minute concatenated blocks as a bounded follow-up: preserving short-run clustering may help without carrying a stale hour-long regime.

**Lesson.** Marginal residual matching does not guarantee good path-level calibration, but a structure that only helps the latest regime is not acceptable. Long blocks can preserve outdated volatility and trend as well as useful clustering.

## Iteration 10 — Antithetic ensemble sampling

**Hypothesis.** Mirroring each shock path with its negative removes finite-ensemble mean error and balances tails without changing the model's intended distribution. This should reduce empirical-CRPS sampling noise, especially at 1000 paths.

**Change.** Add a research-only antithetic switch after shock generation. Start with the 24h model and the same BTC, XAU, and TSLAX latest/September cells; leave 1h until the block-bootstrap experiment is decided. Compare against identical prompts and seed without mirroring.

**Result.** Antithetic versus ordinary sampling on latest: BTC 3295.243 versus 3301.220 (−0.18%), XAU 1983.055 versus 1989.245 (−0.31%), TSLAX 5668.740 versus 5674.022 (−0.09%). September: BTC 2412.655 versus 2416.956 (−0.18%), XAU 1211.321 versus 1210.388 (+0.08%), TSLAX 1608.422 versus 1614.949 (−0.40%). Five of six cells improved; all effects were small.

**Verdict.** Keep antithetic sampling under the signed-consistency rule, then verify it on the final all-asset model. It is not a substitute for calibration.

**Lesson.** Removing finite-ensemble mean error gives a small, low-risk gain without changing the target distribution or path count.

## Iteration 11 — Short blocks and 1h local-time seasonality

**Hypothesis.** Two independent additions may address remaining 1h errors: 15-minute concatenated residual blocks can preserve short-run volatility clustering without the 60-minute block's stale regime; a New York-time profile normalized to unit daily RMS can capture predictable intraday volatility without changing the current EWMA level.

**Change.** Test the two arms independently against independent-empirical 1h results on all five crypto assets in latest and October windows. Arm A concatenates randomly selected 15-minute residual blocks. Arm B uses independent residuals with a seasonal sigma profile. Do not combine them unless each succeeds; do not change the 24h model.

**Result.** Fifteen-minute blocks, latest versus independent empirical: BTC 444.850 (−0.73%), ETH 564.589 (−1.05%), SOL 730.276 (−0.36%), XRP 932.278 (−0.09%), HYPE 917.273 (−0.08%). October: BTC 597.918 (+0.13%), ETH 732.715 (+0.12%), SOL 1055.514 (+0.10%), XRP 1191.831 (−0.34%), HYPE 1285.051 (−0.03%). October's mean relative change was approximately zero, unlike the 60-minute block's clear regression.

Seasonal 1h, latest: BTC 472.832, ETH 604.557, SOL 760.478, XRP 970.154, HYPE 940.076. October: BTC 611.632, ETH 748.124, SOL 1070.922, XRP 1218.138, HYPE 1287.271. These are large regressions on every asset in both windows, generally 2–6%.

**Verdict.** Keep 15-minute blocks provisionally because all latest assets improved and October was neutral on average; verify antithetic sampling and the final model. Discard 1h seasonality outright.

**Lesson.** Short-run clustering is useful, but the local-time profile that helped 24h double-counted information already captured by the fast 1h EWMA and distorted its short-horizon scale. Features do not transfer automatically between formats.

## Iteration 12 — Macro-event scaling for 1h

**Hypothesis.** The early-fit CPI, NFP, and FOMC volatility profiles should reduce the isolated event-prompt errors seen in crypto-1h. Applying each multiplier over five-minute durations, rather than one output step, matches the measured effect decay.

**Change.** Generalize the validated 24h event helper so each profile entry covers five minutes independently of `time_increment`; 24h behavior remains identical. Enable the helper for 1h only through a research switch. Compare late NFP on all five crypto assets and paired calendar-off/on CPI and FOMC windows on BTC, SOL, and XRP. Earnings do not affect the crypto-1h asset set.

**Result.** Late NFP with event scaling versus independent empirical: BTC 447.701 versus 448.124, ETH 570.047 versus 570.580, SOL 732.905 versus 732.936, XRP 932.595 versus 933.164, HYPE 917.802 versus 917.992; all five improved by 0.004–0.094%. October CPI calendar-off/on: BTC 721.305/719.761, SOL 1251.433/1249.002, XRP 1129.384/1126.677, gains of 0.19–0.24%. October FOMC: BTC 645.634/645.335, SOL 1028.255/1027.333, XRP 967.147/966.531, gains of 0.05–0.09%.

**Verdict.** Keep 1h macro-event scaling. Every late-validation cell improved, although the average effect is small. The 24h event helper's five-minute profile entries remain unchanged in behavior.

**Lesson.** Scheduled events provide a repeatable small gain after core calibration. As with 24h, broad session seasonality is harmful while narrow event timing is useful.

## Iteration 13 — Frozen champion and paired final validation

**Hypothesis.** The independently validated components should compose into a lower-CRPS model on both formats without sacrificing robustness or latency.

**Change.** Freeze a 14-day context for 24h with a seven-day 1h slice; New York-time seasonal variance and a 50/50 empirical/Gaussian residual mixture for 24h; 30-minute EWMA scale, independent empirical residuals in 15-minute blocks for 1h; early-fit event multipliers on both; and antithetic sampling on both. Remove all research-only behavior switches. Validate the exact frozen file against the starter at 1000 paths on identical 10-day 24h and 5-day 1h windows ending 2026-11-08.

The 1h antithetic check on 15-minute blocks improved eight of ten cells. Latest BTC/ETH/SOL were 444.046/564.242/728.659; October BTC/ETH/SOL were 596.457/731.515/1051.984. The only regressions were latest XRP +0.04% and HYPE +0.03%; both improved in October, to 1190.174 and 1282.064, respectively.

**Result.** The final paired official runs covered 1200 CRYPTO_24H prompts, 870 COM_EQU_24H prompts, and 3600 CRYPTO_1H prompts, all at 1000 paths with 100% realized coverage. Every asset improved over the starter.

| Configuration | Asset | Starter CRPS | Champion CRPS | Change |
|---|---|---:|---:|---:|
| CRYPTO_24H | BTC | 3018.443 | 2928.650 | −2.97% |
| CRYPTO_24H | ETH | 3934.768 | 3808.022 | −3.22% |
| CRYPTO_24H | SOL | 4719.838 | 4621.843 | −2.08% |
| CRYPTO_24H | XRP | 4369.443 | 4277.773 | −2.10% |
| CRYPTO_24H | HYPE | 6997.143 | 6837.579 | −2.28% |
| COM_EQU_24H | XAU | 1773.666 | 1705.458 | −3.85% |
| COM_EQU_24H | NVDAX | 3721.360 | 3600.274 | −3.25% |
| COM_EQU_24H | GOOGLX | 4202.380 | 3877.412 | −7.73% |
| COM_EQU_24H | TSLAX | 3926.549 | 3647.733 | −7.10% |
| COM_EQU_24H | AAPLX | 2583.925 | 2412.599 | −6.63% |
| COM_EQU_24H | WTIOIL | 3758.016 | 3587.922 | −4.53% |
| COM_EQU_24H | SPCX | 5096.375 | 4811.138 | −5.60% |
| COM_EQU_24H | SP500 | 1375.821 | 1312.137 | −4.63% |
| CRYPTO_1H | BTC | 680.708 | 643.243 | −5.50% |
| CRYPTO_1H | ETH | 857.292 | 816.447 | −4.76% |
| CRYPTO_1H | SOL | 1016.086 | 983.035 | −3.25% |
| CRYPTO_1H | XRP | 1007.942 | 993.282 | −1.45% |
| CRYPTO_1H | HYPE | 1334.264 | 1277.754 | −4.24% |

Equal-asset configuration means:

| Configuration | Starter CRPS | Champion CRPS | Change |
|---|---:|---:|---:|
| CRYPTO_24H | 4607.926948 | 4494.773379 | −2.4556% |
| COM_EQU_24H | 3304.761512 | 3119.334143 | −5.6109% |
| CRYPTO_1H | 979.258185 | 942.752021 | −3.7279% |

The frozen file also passed compilation, odd and requested path counts, positive finite paths, all-NaN and missing-calendar fallbacks, and deterministic seed checks. Latency at 1000 paths was 0.063s for 24h and 0.021s for 1h.

**Verdict.** Keep and nominate. All 18 final asset cells improved and all delivery checks passed. The half-life, context, seasonal, block-length, and residual-mixture sweeps now show diminishing or inconsistent returns; further tuning on the same windows is not justified by the evidence.

**Lesson.** Explicitly matching the estimator to each scoring format is more valuable than adding model complexity. Robust gains came from conditional scale, residual shape, short-run path structure, and narrowly timed event information, validated one change at a time and then checked as a complete system.
