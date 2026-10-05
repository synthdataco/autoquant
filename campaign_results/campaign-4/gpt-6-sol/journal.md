# Research journal

## Iteration 0 — starter baseline (in progress)

**Hypothesis.** The supplied seven-day, constant-volatility log-normal fan will be a useful reference but will miscalibrate quiet and active periods, especially equity and commodity perps.

**Change.** None. Measure the supplied model at 1000 paths with the official backtester on covered historical windows.

**Result.** Preliminary official backtester, 2026-09-09 to 2026-09-10, 1000 paths, hourly predictions: BTC 24h mean CRPS 2608.9 (25 matched prompts, 100% realized coverage); BTC 1h 598.1 (139 matched prompts, 100%); XAU 24h 1918.0 (25 matched prompts, 100%). The XAU evaluator initially failed because `MinutePriceStore.load_range` rejects valid NaN closes; the evaluator now reads raw partitions and leaves gaps intact. **Caveat discovered:** hourly predictions do not match the prompt cadence (30 minutes at 24h; 5 minutes at 1h), and the backtester may pair a prompt with a *later* nearby prediction. These numbers are not suitable for champion nomination. Native-cadence reruns are required.

**Verdict.** Keep as the fixed comparison baseline.

**Lesson.** The validator scores several return intervals and a final absolute price; CRYPTO_1H additionally scores return gaps and realized volatility blocks. All changes must be evaluated by configuration.

## Iteration 1 — seasonality and recent volatility (in progress)

**Hypothesis.** A recent volatility estimate adjusted for UTC hour and weekend patterns will lower CRPS, especially for XAU and tokenized equities; it may also improve crypto after a regime change.

**Change.** Fit hour-by-weekend relative minute variance from the frozen snapshot through 2026-07-31, with weekly normalization and hour smoothing. Use adjacent valid minute returns only. Estimate the current variance level from the context after removing these historical patterns, then project the pattern over the requested horizon. Compare on 2026-08 and 2026-09 windows against the unchanged baseline.

**Result.** First run was invalid: a timestamp-unit bug excluded every adjacent return, causing a fallback spread roughly 2–4 times too wide. It scored BTC 24h 3381.3, BTC 1h 758.3, XAU 24h 3161.0; these are implementation-error diagnostics, not the seasonality test. After fixing the minute comparison, a BTC sample's 24h terminal SD is 177 bps versus the starter's 173 bps. Preliminary official backtest on 2026-09-09 to 2026-09-10: BTC 24h 2582.8 versus baseline 2608.9; BTC 1h 595.6 versus 598.1; XAU 24h 1868.8 versus 1918.0. On 2026-08-15 to 2026-08-16 (Saturday): BTC 24h 690.3 versus 861.6; BTC 1h 156.3 versus 190.4; XAU 24h 284.9 versus 669.7. All six comparisons have 100% realized coverage and the same matched-prompt counts within each pair, **but** the hourly prediction cadence admits later-prediction matches. Rerun at native cadence before making a final verdict.

**Verdict.** Working candidate only; native-cadence validation pending.

**Lesson.** Weekend variance reduction matters strongly for CRPS, especially XAU. A timestamp precision assumption silently eliminated all returns in the first implementation; use duration comparisons, not raw timestamp integers.

**Exact-prompt correction.** The regular prediction grid also proved invalid: actual historical prompt starts have irregular minute offsets. `predict.py` now fetches start times from the offline score cache before prediction, checks the frozen snapshot through each horizon, and generates one fan per prompt. The first valid paired official backtest uses 1000 paths and full realized coverage for every pair:

| Scored dates (inclusive) | Configuration / asset | Starter CRPS | Seasonal CRPS | Prompts |
|---|---|---:|---:|---:|
| 2026-08-15–16 | CRYPTO_24H / BTC | 1053.2 | 920.8 | 48 |
| 2026-08-15–16 | CRYPTO_1H / BTC | 205.4 | 170.5 | 288 |
| 2026-08-15–16 | COM_EQU_24H / XAU | 841.2 | 542.8 | 48 |
| 2026-09-09–10 | CRYPTO_24H / BTC | 2520.0 | 2499.8 | 48 |
| 2026-09-09–10 | CRYPTO_1H / BTC | 574.8 | 568.9 | 288 |
| 2026-09-09–10 | COM_EQU_24H / XAU | 1848.5 | 1806.1 | 48 |

The prior hourly-grid values above remain in the journal as a discarded evaluator attempt; they are not evidence for selection.

**Expanded exact-prompt result.** Across all eight active COM_EQU assets, the mean of each asset's 48-prompt CRPS was 1613.31 for the starter and 1260.84 for fixed seasonality on 2026-08-15–16; on 2026-09-09–10 it was 2523.42 versus 2411.60. All eight assets improved in both windows. This mean is an unweighted descriptive CRPS average across assets, not a field-relative score.

## Iteration 2 — recent volatility level (planned)

**Hypothesis.** A seven-day variance estimate lags volatility regime changes. Blending it equally with an exponentially weighted estimate (24-hour half-life), after removing the fitted calendar pattern, should improve CRPS on active weekdays without losing the weekend gain.

**Change.** Retain the same calendar profile, fan generator, and seed. Replace only the variance level with 0.5 × seven-day mean standardized squared return + 0.5 × 24-hour-half-life weighted mean.

**Result.** Initial exact-prompt official backtests at 1000 paths: August BTC 24h 925.8 vs fixed seasonal 920.8 (worse 0.5%); August BTC 1h 168.3 vs 170.5 (better 1.3%); August XAU 24h 538.6 vs 542.8 (better 0.8%). September BTC 24h 2487.0 vs 2499.8 (better 0.5%); September BTC 1h 568.2 vs 568.9 (better 0.1%); September XAU 24h 1800.6 vs 1806.1 (better 0.3%). All paired runs have full realized coverage. Across all five crypto assets, mean CRPS per asset for **CRYPTO_24H** on August 15–16 was starter 1697.79, fixed seasonality 1582.61, recent blend 1572.80; on September 9–10 it was 3862.87, 3847.72, 3831.82. For **CRYPTO_1H**, the corresponding means were 349.90, 319.37, 315.47 in August and 873.12, 868.41, 866.21 in September. The recent blend improved fixed seasonality for every crypto asset at 1h in both windows; at 24h it missed on BTC in August by 0.5% and improved the other nine asset-window pairs. Three COM_EQU assets tested with the recent blend showed near-ties, so their working rule remains fixed seasonality.

**Verdict.** Keep recent blend for crypto 24h and 1h; keep fixed seasonality for COM_EQU. Confirm with a ten-day holdout before nomination.

**Lesson.** After removing calendar effects, recent variance still contains useful information for crypto, especially for 1-hour scoring. Its effect is small relative to the calendar correction and cannot be inferred from one asset.

## Iteration 3 — ten-day holdout (in progress)

**Hypothesis.** The combined rule (recent blend for crypto, fixed calendar variance for COM_EQU) will lower mean CRPS across all assets in each of the three configurations on a longer, mixed-weekday window.

**Change.** Consolidate the two selected rules into `agent/modeling.py`; leave the fitted calendar profile unchanged (training ends 2026-07-31). Score exact historical starts dated 2026-09-03 through 2026-09-12 inclusive, at 1000 paths, for all 13 active assets and both crypto horizons. Compare against the original starter on the same prompts. This validation interval spans ten scored days and leaves room for the backtester's price-read buffer inside the snapshot.

**Result.** An initial attempt ending 2026-09-14 failed before scoring: synth-lib's price loader unconditionally reads through the latest prompt start plus 26 hours, including for 1h, and attempted to ingest a missing 2026-09-15 partition. The read-only snapshot blocked the write. No result was used. The evaluator now checks that entire 26-hour range before calling the backtester; the run is moved to 2026-09-03–12. Pending.

**Verdict.** Keep the calendar and recent-volatility rule. The ten-day configuration mean improves against the starter in all three formats; all 18 asset-format pairs improve.

**Lesson.** A fixed calendar surface plus a recent level adjustment for crypto remained useful across the mixed ten-day September interval, not only the initial two-day screens.

**COM_EQU ten-day checkpoint.** All eight active COM_EQU assets completed the September 3–12 exact-prompt run with 240 prompts each, 1000 paths, and full realized coverage. Their unweighted mean CRPS was 2176.07 for the starter, 1981.98 for the calendar Gaussian model, and 1962.43 for Student-t(5). Each of the eight improved against both preceding models. Crypto results are still running.

**CRYPTO_24H ten-day checkpoint.** All five crypto assets completed September 3–12 at 240 prompts each, 1000 paths, and full realized coverage. Their unweighted mean CRPS was 3903.16 for the starter, 3824.47 for calendar Gaussian, and 3797.67 for Student-t(5). Each asset improved at both steps. CRYPTO_1H still awaits HYPE.

**Completed three-format ten-day result.** HYPE 1-hour finished, completing all 18 asset-format pairs. The unweighted mean CRPS across each configuration's active assets was:

| Configuration | Starter | Calendar Gaussian | Student-t(5) |
|---|---:|---:|---:|
| COM_EQU_24H (8 assets) | 2176.07 | 1981.98 | 1962.43 |
| CRYPTO_24H (5 assets) | 3903.16 | 3824.47 | 3797.67 |
| CRYPTO_1H (5 assets) | 831.11 | 810.84 | 806.57 |

Each Student-t(5) asset-format CRPS was below the matched Gaussian and starter CRPS. There were 240 exact prompts per 24-hour asset and 1440 per 1-hour asset, all with full realized coverage and 1000 paths.

## Iteration 4 — heavy-tailed shocks (planned)

**Hypothesis.** Conditional one-minute returns remain much heavier-tailed than Gaussian after calendar and recent-variance scaling. A unit-variance Student-t shock with five degrees of freedom may reduce CRPS for short return intervals and improve the volatility-block distribution in CRYPTO_1H.

**Change.** Preserve all variance forecasts and seeds from the combined model; change only the standardized shock law from normal to Student-t(5) multiplied by `sqrt(3/5)` to keep variance unchanged. Evaluate first on the two exact-prompt windows at 1000 paths. Only test on the ten-day holdout if the small-window mean CRPS improves materially.

**Evidence before test.** On training data from 2026-05-18 to 2026-07-31, after dividing adjacent one-minute returns by fitted calendar factors and a lagged 24-hour-half-life variance estimate: empirical absolute standardized-return 99th/99.9th percentiles are BTC 3.33/6.39, ETH 3.38/6.69, XAU 3.80/7.80; a normal distribution has 2.58/3.29. These are distribution diagnostics, not CRPS results.

**Result.** Initial exact-prompt official backtests at 1000 paths (Student-t versus the selected Gaussian rule): on August 15–16, BTC 24h 905.04 vs 925.76, BTC 1h 164.56 vs 168.27, ETH 1h 217.57 vs 222.19, XAU 24h 525.51 vs 542.83. On September 9–10, BTC 24h 2482.17 vs 2487.05, BTC 1h 567.02 vs 568.20, ETH 1h 703.39 vs 705.91, XAU 24h 1797.44 vs 1806.14. All have full realized coverage and the same exact prompt counts in each pair. The full August sweep favored Student-t for **all 18 asset-format pairs**: mean COM_EQU_24H CRPS 1260.84 → 1214.87, CRYPTO_24H 1572.80 → 1547.42, CRYPTO_1H 315.47 → 311.96. In September, mean COM_EQU_24H improved 2411.60 → 2402.94, CRYPTO_24H improved 3831.82 → 3815.84, and CRYPTO_1H improved 866.21 → 864.43. Three September asset-format pairs **regressed slightly**: WTIOIL 24h 3471.23 → 3472.82 (0.05%), SP500 24h 769.47 → 770.72 (0.16%), and HYPE 1h 1210.55 → 1211.58 (0.09%). Ten-day confirmation pending.

**Verdict.** Keep Student-t(5) as the working 24-hour and 1-hour shock law. It improved all 18 asset-format pairs over the Gaussian rule on September 3–12, including WTIOIL 24-hour, SP500 24-hour, and HYPE 1-hour, which had tiny September 9–10 regressions. For this larger screen, predictions use synth-lib's float32 day-file layout; a BTC 1h 288-prompt parity check changed mean CRPS by only 0.00003 versus the JSON layout.

**Lesson.** Matching variance is not sufficient for short-return CRPS; heavier conditional tails can improve both price and 1h volatility scoring. The benefit is materially larger in the quiet weekend sample than in the active September sample.

## Iteration 5 — uncertainty in the volatility level (planned)

**Hypothesis.** A single volatility estimate for all simulated paths understates the variation in realized block volatility, particularly in CRYPTO_1H. Giving each path an independent, persistent lognormal volatility multiplier with log-standard-deviation 0.3 may lower the official CRPS.

**Change.** Keep the calendar factors, volatility estimator, Student-t(5) shocks, and seeding. Multiply each path's shock scales by `exp(0.3 Z - 0.3²)`, where `Z` is one standard normal draw for that path. This keeps the mean squared multiplier at one. Test on both scored-date windows against Student-t(5) without the multiplier before considering a longer holdout.

**Decision rule.** Reject configuration-wide use if mean CRPS regresses in either window or if any large asset regression outweighs small gains.

**Paired-draw correction.** The initial implementation drew path-scale multipliers before the Student-t shocks, changing the random shock stream relative to the Student-t(5) baseline. Its August gains and mixed September 24-hour result are exploratory because Monte Carlo noise is not controlled within the pair. A follow-up `t5_mix_crn.py` draws the Student-t shocks first, then the multipliers, so the shock array matches the baseline exactly. The first follow-up focuses on BTC and ETH 1-hour, where the initial variant improved both windows.

**Additional holdout.** September 9–10 is inside the September 3–12 ten-day run, so the latter is not fully independent of the short screen. For a decisive 1-hour comparison, reserve August 20–29 (10 scored days) as a separate holdout for all five crypto assets. Its prompt count varies by asset in the offline cache (for example BTC has 1653 and SOL 1463), so comparisons use each asset's exact matched starts. The baseline Student-t(5) is generated in advance; candidate output will be generated only if the paired short screen remains favorable.

**Result.** The paired-draw multiplier improved BTC and ETH 1-hour CRPS in both two-day windows. On the independent August 20–29 holdout, BTC moved 715.72 → 709.54 and ETH 992.48 → 976.73, using exact starts, 1000 paths, and full realized coverage. The unpaired 24-hour variant had September regressions for WTIOIL and SP500 and was not advanced. The selected 1-hour hybrid uses the multiplier for BTC; ETH uses the better supported Student-t(3) variant.

**Verdict.** Keep pathwise volatility mixing for BTC 1-hour only. The initial configuration-wide proposal is not supported across horizons.

**Lesson.** Pathwise uncertainty helped the 1-hour fan, but controlling the shock stream within a paired simulation mattered for interpreting sub-percent CRPS changes.

## Iteration 6 — stronger tail law (planned)

**Hypothesis.** The standardized one-minute training returns have extreme quantiles above a unit-variance Student-t(5), so Student-t(3) shocks may improve short-horizon CRPS further.

**Change.** Retain the selected variance model, independent path shocks, and seeding. Change only the Student-t degrees of freedom from five to three, with variance normalization `sqrt(1/3)`. Compare against Student-t(5) on the two exact-prompt windows before any ten-day confirmation.

**Decision rule.** Keep configuration-wide use only if CRPS improves consistently and outliers do not create material asset regressions.

**Screen extension.** BTC and ETH 1-hour together improve under Student-t(3) in both short windows, though September BTC alone regresses by 0.09%. Test SOL, XRP, and HYPE 1-hour on the same two windows before a configuration-wide verdict. The 24-hour sample shows September regressions for BTC, XAU, and WTIOIL, so this candidate is now primarily an evaluation of 1-hour shocks.

**Full short-screen result.** For CRYPTO_1H, Student-t(3) versus Student-t(5) averaged 307.46 versus 311.96 across five assets on August 15–16 (better 1.44%), but 864.75 versus 864.43 on September 9–10 (worse 0.04%). HYPE's September CRPS rose from 1211.58 to 1217.98, and BTC rose from 567.02 to 567.55. ETH, SOL, and XRP each improved in both windows. Student-t(3) therefore fails the pre-registered configuration-wide consistency rule; test only those three asset-specific possibilities on the independent August 20–29 holdout before deciding whether exceptions are justified. The sampled 24-hour assets also all regressed in September and are rejected for that horizon.

**Independent asset check.** On August 20–29, ETH 1-hour moved 992.48 → 972.99, SOL 1410.55 → 1389.02, and XRP 2056.30 → 2020.91 under Student-t(3), all with exact matched starts, 1000 paths, and full realized coverage. These three had also improved in both short windows, so the selected 1-hour hybrid uses Student-t(3) for them. BTC's August holdout improved but its September short screen regressed slightly; HYPE regressed on the September short screen. Both remain on other shock rules.

**Verdict.** Keep Student-t(3) for ETH, SOL, and XRP 1-hour only; reject it for 24-hour and configuration-wide 1-hour use.

**Lesson.** Heavy tails are useful at one-minute resolution, but a single degrees-of-freedom choice did not improve every asset or the 24-hour format consistently.

## Iteration 7 — multi-horizon variance forecast (planned)

**Hypothesis.** The current half-blend of seven-day and 24-hour realized variance may miss shorter volatility changes relevant to CRYPTO_1H. A regularized regression on past 1-hour, 6-hour, 24-hour, and 7-day calendar-adjusted variance may lower CRPS, especially its realized-volatility component.

**Change.** Fit one small log-variance regression per crypto asset on the frozen January–July 2026 minute data, using only targets whose next hour ends by July 31. Preserve the Student-t(5) shocks, calendar profile, seeding, and 24-hour behavior. Replace only the predicted variance level for 1-hour prompts. Compare at 1000 paths on both exact-prompt short windows before an independent holdout.

**Decision rule.** Advance only if the five-asset mean 1-hour CRPS improves in both short windows without an asset regression above 1%.

**Result.** The fitted hourly regression used 9,797–9,838 January–July training origins per crypto asset. On August 15–16, it lowered the five-asset CRYPTO_1H mean CRPS from Student-t(5)'s 311.96 to 304.49 (2.40%); all five assets improved. On September 9–10, it raised the mean from 864.43 to 865.07 (0.07%); BTC, ETH, SOL, and XRP regressed slightly, while HYPE improved. All runs used exact starts, 1000 paths, and full realized coverage.

**Verdict.** Discard the all-week regression under the pre-registered consistency rule. The observed weekend/weekday split motivates a separate, explicitly conditional test below; it is not evidence that the all-week rule works.

## Iteration 8 — newer calendar fit for COM_EQU (planned)

**Hypothesis.** SPCX and SP500 had relatively little history in the July 31 calendar fit. Adding the complete August month may reduce hourly-profile noise and lower September COM_EQU_24H CRPS.

**Change.** Refit the same per-asset hour-by-weekend profile through August 31, retaining every other model choice from Student-t(5). Screen all eight COM_EQU assets on September 9–10 at exact prompts and 1000 paths, then compare on the nonoverlapping September 1–8 scored dates if the short screen improves the eight-asset mean without an asset regression above 1%.

**Short-screen result.** On September 9–10, the eight-asset COM_EQU_24H mean CRPS moved from 2402.94 with the July profile to 2396.84 with the August 31 profile (0.25% lower). Seven assets improved; NVDAX regressed by 0.04%, below the stated 1% cutoff. SPCX had the largest gain, 4065.71 → 4029.84 (0.88%). This passes the screen for the nonoverlapping September 1–8 check.

**Holdout result.** On the nonoverlapping September 1–8 scored dates, all eight COM_EQU assets improved with the August 31 profile. The unweighted mean CRPS fell from 1890.25 to 1881.08 (0.49%). Each asset had 192 exact prompts, 1000 paths, and full realized coverage. The largest relative gains were NVDAX 0.83%, AAPLX 0.73%, and SPCX 0.70%.

**Full-window confirmation.** On September 3–12, the eight-asset mean CRPS moved from 2176.07 (starter) to 1981.98 (calendar Gaussian), 1962.43 (Student-t(5) with the July profile), and 1953.75 (Student-t(5) with the August 31 profile). The newer profile improved all eight assets again, by 0.12–0.78% each. All pairs had 240 exact prompts per asset, 1000 paths, and full realized coverage.

**Verdict.** Keep the August 31 calendar profile for COM_EQU_24H.

## Iteration 9 — weekend-only hourly regression (planned)

**Hypothesis.** The HAR variance forecast's August benefit may reflect quieter weekends; applying it only to crypto 1-hour prompts starting on Saturday or Sunday may retain that benefit without its September weekday regression.

**Change.** Use the July-trained HAR level only for weekend CRYPTO_1H starts, otherwise use Student-t(5)'s existing volatility estimate. Preserve the same shocks, calendar factors, and seed. Evaluate on the nonoverlapping August 20–29 scored dates, which include weekend and weekday starts, against Student-t(5) for all five assets at 1000 paths. If the five-asset mean improves and no asset regresses above 1%, confirm on September 1–8, which also contains a weekend and does not overlap the September 9–10 screen.

**First holdout result.** On August 20–29, the five-asset mean CRPS fell from 1359.68 for Student-t(5) to 1342.90 for weekend-only HAR (1.23%). Every asset improved, from 0.68% for HYPE to 1.77% for XRP. Exact prompt counts varied from 1463 to 1653 by asset, with 1000 paths and full realized coverage in every pair. This passes the gate for the September 1–8 confirmation.

**Second holdout result.** On September 1–8, the five-asset mean CRPS fell from 806.76 for Student-t(5) to 804.18 for weekend-only HAR (0.32%). All five assets improved. This confirms that using the hourly forecast only on weekends avoids the all-week rule's September weekday regression.

**Verdict.** Keep weekend-only HAR as a variance component; test its interaction with the selected 1-hour shock laws in Iteration 12 before serving it.

## Iteration 10 — selected 1-hour shock laws (planned)

**Hypothesis.** Asset-specific 1-hour shocks supported by two short screens and the separate August 20–29 holdout may lower configuration-wide CRPS on a new September period.

**Change.** Keep Student-t(5) for every 24-hour asset and HYPE 1-hour. For BTC 1-hour, use Student-t(5) plus the paired-draw pathwise lognormal scale with log-standard-deviation 0.3; for ETH, SOL, and XRP 1-hour, use unit-variance Student-t(3). All other variance and calendar calculations remain as in the baseline. Score September 1–8 exact starts at 1000 paths against Student-t(5) for all five crypto assets.

**Decision rule.** Keep the hybrid if its five-asset mean CRPS improves and no asset regresses above 1%; otherwise retain the robust Student-t(5) baseline or only independently supported per-asset exceptions.

**Result.** On the nonoverlapping September 1–8 scored dates, the five-asset mean CRPS fell from 806.76 for Student-t(5) to 803.57 for the hybrid (0.39%). BTC, ETH, SOL, and XRP improved individually by 0.41–0.61%; HYPE was exactly equal because its fan is unchanged. Every asset had matched exact prompt starts, 1000 paths, and full realized coverage.

**Full-window confirmation.** On September 3–12, five-asset CRYPTO_1H mean CRPS was 831.11 for the starter, 810.84 for calendar Gaussian, 806.57 for Student-t(5), and 803.14 for the selected shock hybrid. BTC, ETH, SOL, and XRP each improved on Student-t(5) by 0.55–0.65%; HYPE tied exactly. Every asset had 1440 exact prompts, 1000 paths, and full realized coverage.

**Verdict.** Keep the selected 1-hour shock rules. A separate interaction test will determine whether weekend-only HAR variance adds further benefit.

## Iteration 11 — intermediate 24-hour tail law (planned)

**Hypothesis.** Student-t(4) shocks may retain more of Student-t(3)'s quiet-weekend gain without its active-weekday regressions, improving 24-hour price CRPS beyond Student-t(5).

**Change.** Retain calendar factors, volatility estimates, and seeding. Change only the shock law for both tested 24-hour formats to unit-variance Student-t(4), scaled by `sqrt(1/2)`. Screen BTC 24-hour and XAU 24-hour on August 15–16 and September 9–10 at exact starts and 1000 paths.

**Decision rule.** Expand to all 24-hour assets only if the two-asset mean improves by at least 0.2% in both windows and neither asset regresses more than 0.5% in either window. Otherwise retain Student-t(5).

**Result.** Student-t(4) lowered BTC/XAU 24-hour CRPS on August 15–16 from 905.04/525.51 to 897.01/519.04 (0.89%/1.23%). On September 9–10 it lowered 2482.17/1797.44 to 2480.99/1796.53 (about 0.05% for each). All four official runs used 1000 paths, exact starts, and full realized coverage.

**Verdict.** Discard under the pre-registered 0.2% per-window expansion threshold; the September improvement is too small to justify replacing the well-confirmed Student-t(5) rule.

**Lesson.** An intermediate tail law retains some quiet-weekend gain, but its active-weekday effect is negligible at this screen's resolution.

## Iteration 12 — combine weekend variance and 1-hour shocks (planned)

**Hypothesis.** Weekend-only HAR variance and the selected asset-specific 1-hour shock laws may make complementary corrections and lower CRPS further.

**Change.** Start from `hybrid_1h.py`; replace only its crypto 1-hour variance level on Saturday and Sunday with the July-trained HAR forecast. Keep the hybrid's shock draws, calendar factors, and seed. Compare against the hybrid on September 1–8 at 1000 paths. If the five-asset mean improves without a regression above 1%, confirm on the untouched August 5–14 scored dates against the hybrid and Student-t(5).

**Decision rule.** Keep the combined rule only if its five-asset mean improves over the hybrid on both windows and no asset regresses above 1% on August 5–14.

**September screen result.** On September 1–8, mean CRPS across five crypto 1-hour assets was 806.76 for Student-t(5), 804.18 for weekend-only HAR, 803.57 for the selected shock hybrid, and 801.66 for the combination. The combined rule improved each asset over the shock hybrid by 0.16–0.49%, with exact matched starts, 1000 paths, and full realized coverage. Proceed to the untouched August 5–14 confirmation.

**Untouched holdout result.** On August 5–14, five-asset mean CRPS was 540.23 for Student-t(5), 538.73 for the selected shock hybrid, and 537.29 for the combination. BTC, ETH, XRP, and HYPE improved over the hybrid; SOL rose from 535.98 to 536.18 (0.04%), well below the predeclared 1% asset limit. Every model saw 1440 exact starts per asset, 1000 paths, and full realized coverage.

**Full-window confirmation.** On September 3–12, mean CRYPTO_1H CRPS was 831.11 for the starter, 806.57 for Student-t(5), 803.14 for the shock hybrid, and 799.89 for the combination. All five assets improved over the shock hybrid on this window, with 1440 exact prompts per asset, 1000 paths, and full realized coverage.

**Verdict.** Keep the combined weekend-HAR plus asset-specific 1-hour shock rule. It improves the five-asset mean on both confirmation windows under the decision rule.

**Lesson.** The shock and weekend variance changes remain complementary outside their original two-day screens, though the small SOL regression shows the variance rule is not universally beneficial on every period.

## Serving module verification

The selected serving module at commit `afc9714` completed a direct official September 3–12 backtest at 1000 paths. All 18 asset-format results were present, with 240 exact prompts per 24-hour asset, 1440 exact prompts per 1-hour asset, and full realized coverage. Its unweighted configuration mean CRPS was 1953.75 for COM_EQU_24H, 3797.67 for CRYPTO_24H, and 799.89 for CRYPTO_1H, exactly matching the individually tested components to reported precision. A seven-case path parity check, NaN-context check, benchmark-seed check, and 1000-path serving latency check also passed. This verifies the consolidated code, but does not end the campaign while time remains.

## Iteration 13 — variance scale calibration (planned)

**Hypothesis.** The combined calendar, recent-level, and weekend HAR estimates may overstate path variance after fitting squared returns; a uniform 15% reduction in forecast variance may sharpen the fan and lower CRPS.

**Change.** Multiply only the final predicted minute variance `sigma2` by 0.85, retaining calendar factors, shock laws, pathwise multiplier, seeds, and all horizon-specific logic. Screen BTC 24-hour and ETH 1-hour on both August 15–16 and September 9–10 exact prompts at 1000 paths. Screen XAU 24-hour on September 1–2 and September 9–10; its serving calendar profile was fit through August 31, so an August test would be lookahead.

**Decision rule.** Expand to all assets only if the three-case mean improves on both windows with no case regressing more than 1%; otherwise discard the uniform reduction and investigate selective calibration only if the screen gives a clear, consistent pattern.

**Screen result.** BTC 24-hour moved 905.04 → 891.08 on August 15–16 and 2482.17 → 2481.37 on September 9–10. ETH 1-hour moved 201.37 → 199.74 and 701.46 → 701.81 on those dates. XAU 24-hour moved 2083.06 → 2094.71 on September 1–2 and 1796.77 → 1795.10 on September 9–10. The paired three-case mean improved by about 0.12% using the earlier applicable windows and 0.04% on September 9–10; no individual regression exceeded 1%. This narrowly passes the predeclared expansion rule, but the effect is fragile. Expand to all 18 asset-format cases on September 1–2 and September 9–10, using the same factor and 1000 paths, before any serving change.

**Expansion checkpoint.** On September 1–2, the five-asset CRYPTO_24H mean fell from 4042.47 to 4019.75 (0.56%) and all five assets improved. On September 9–10 it fell from 3815.84 to 3813.28 (0.07%); four assets improved, while HYPE regressed by 0.31%. The September 1–2 CRYPTO_1H mean fell from 886.48 to 885.54 (0.11%), despite BTC, ETH, and XRP regressions. The September 1–2 COM_EQU_24H mean fell from 2329.87 to 2325.38 (0.19%), but four of eight assets regressed; NVDAX had a 2.21% gain. The second-window 1-hour and COM_EQU sweeps remain in progress. For crypto 24-hour only, run all five assets on the untouched August 15–16 weekend before considering a longer holdout. Keep the chosen serving module unchanged throughout this test.

**Crypto 1-hour verdict.** On September 9–10, the five-asset mean rose from 863.22 to 864.58 (0.16% worse); all five assets regressed. Reject the uniform 0.85 variance factor for CRYPTO_1H despite its small September 1–2 mean gain.

**Crypto 24-hour August screen.** On August 15–16, the five-asset mean fell from 1547.42 to 1530.72 (1.08%); all five assets improved. This is directionally consistent with the two September screens, which improved by 0.56% and 0.07%. The single exception across the three screens is HYPE on September 9–10 (0.31% worse).

**COM_EQU verdict.** The September 9–10 eight-asset mean rose from 2396.84 to 2399.95 (0.13% worse), with five assets regressing. Although September 1–2 improved by 0.19%, this fails the cross-window consistency check. Reject the uniform 0.85 factor for COM_EQU_24H.

## Iteration 14 — independent crypto 24-hour variance holdout (planned)

**Hypothesis.** The uniform 0.85 variance factor lowers CRYPTO_24H CRPS beyond the short windows that selected it.

**Change.** Compare the selected model and `variance_scale.py` on all five crypto 24-hour assets over the untouched August 5–14 scored dates, using exact starts, 1000 paths, and the official backtester. This window predates every selection screen for this factor and follows the July crypto calendar fit.

**Decision rule.** Keep the factor for crypto 24-hour only if the five-asset mean improves and no asset regresses more than 1%; then confirm on September 3–12 before changing the serving module. Otherwise retain the selected serving model.

**Independent holdout result.** On August 5–14, the five-asset mean fell from 2510.26 to 2502.36 (0.31%). All five assets improved: BTC 0.42%, ETH 0.71%, SOL 0.27%, XRP 0.19%, HYPE 0.15%. Every pair had 240 exact prompts, 1000 paths, and full realized coverage. This passes the decision rule; run the September 3–12 confirmation, which contains neither the August holdout nor the August weekend screen but overlaps the two September two-day selection screens.

**September confirmation.** On September 3–12, the mean fell from 3797.67 to 3787.20 (0.28%). BTC, ETH, SOL, and XRP improved by 0.36–0.39%; HYPE rose from 5006.46 to 5007.56 (0.02%), below the 1% asset limit. Each candidate run had 240 exact prompts, 1000 paths, and full realized coverage. The configuration mean improved on the independent holdout and this mixed-weekday confirmation.

**Verdict.** Keep the 0.85 variance factor for CRYPTO_24H only. Reject it for CRYPTO_1H and COM_EQU_24H as documented above. Apply only the 24-hour crypto conditional to the serving module, then verify output parity with the scored candidate and direct serving-module CRPS.

**Serving verification.** The conditional factor was applied to `modeling.py`. Path parity passed for all five crypto 24-hour assets against the scored 0.85 candidate; sampled BTC, ETH, and HYPE 1-hour plus XAU and SP500 24-hour outputs were byte-identical to the prior serving commit. A direct official September 3–12 backtest of the updated serving module reproduced all five candidate crypto 24-hour CRPS values exactly, with 240 exact prompts and full coverage per asset. Its five-asset mean was 3787.20. The unchanged formats retain their previously verified COM_EQU_24H 1953.75 and CRYPTO_1H 799.89 means.

**Lesson.** Squared-return calibration left the crypto 24-hour fan modestly too wide on two longer windows. The same adjustment failed for 1-hour crypto and COM_EQU, so the variance correction belongs to the specific horizon and asset group.

## Iteration 15 — stronger crypto 24-hour variance shrinkage (planned)

**Hypothesis.** The 0.85 crypto 24-hour variance factor may be conservative; a factor of 0.70 might further improve CRPS in quiet and active regimes.

**Change.** Keep the selected module's structure, profiles, shocks, and seed; change only its crypto 24-hour variance factor from 0.85 to 0.70 in an isolated candidate. Compare to the already scored 0.85 candidate on all five crypto 24-hour assets over August 15–16 and September 9–10. Both are 1000-path exact-prompt official backtests and share the random draw stream.

**Decision rule.** Advance only if the five-asset mean improves on both windows with no asset regression above 0.5%; then test on the independent August 20–29 window before changing serving code. Otherwise keep 0.85.

**Result.** On August 15–16, five-asset mean CRPS fell from 1530.72 at 0.85 to 1517.88 at 0.70 (0.84% lower); BTC, ETH, SOL, and XRP improved while HYPE rose 0.20%. On September 9–10, it rose from 3813.28 to 3823.61 (0.27% worse); BTC, ETH, XRP, and HYPE regressed, with HYPE up 0.67%. All runs used the same exact 48 prompts per asset, 1000 paths, full realized coverage, and the same shock stream as the 0.85 candidate.

**Verdict.** Discard 0.70 under both parts of the predeclared rule. Keep 0.85 for crypto 24-hour.

**Lesson.** Greater variance shrinkage sharpened the quiet weekend fan but made the active September fan too narrow; the calibrated factor must balance regimes.

## Iteration 16 — weekday crypto 1-hour variance widening (planned)

**Hypothesis.** The 1-hour fan may be too narrow on active weekdays. A uniform 15% variance reduction raised September 9–10 CRPS for all five assets, suggesting a modest increase on weekdays could improve calibration while leaving the selected weekend HAR rule intact.

**Change.** In an isolated module, multiply only crypto 1-hour predicted variance on Monday–Friday starts by 1.10. Keep weekend variance, shock laws, calendar profiles, pathwise mixing, and random seeds unchanged. Score all five assets at 1000 paths on September 1–2 and September 9–10; both windows have weekday starts. Compare to the already scored, byte-identical 1-hour serving baseline on the exact same starts.

**Decision rule.** Advance only if the five-asset mean improves on both windows and no asset regresses more than 0.5% in either. If it passes, test the mixed-weekday August 20–29 independent holdout before any serving change.

**Result.** The first window, September 1–2, failed both conditions. Five-asset mean CRPS rose from 886.48 to 888.28 (0.20% worse), and HYPE rose from 1056.83 to 1063.32 (0.61% worse). BTC, ETH, SOL, and XRP also rose, by 0.003–0.24%. Every asset had 288 exact prompts, 1000 paths, and full realized coverage. The second short-window run was stopped after the first window failed the predeclared decision rule; it could no longer make this candidate eligible.

**Verdict.** Discard weekday 1-hour variance widening. Retain the selected 1-hour variance levels.

**Lesson.** A narrower fan failing on one active weekday window did not imply that the next wider fan would help on another; the CRPS response near the selected scale varies by asset and date.

## Iteration 17 — HYPE 1-hour pathwise volatility uncertainty (planned)

**Hypothesis.** HYPE 1-hour's Student-t(5) fan may understate uncertainty in its future volatility level. BTC 1-hour gained from a pathwise lognormal scale, whereas heavier Student-t(3) shocks failed HYPE on September 9–10. A persistent volatility multiplier may improve HYPE's realized-volatility CRPS without changing its mean variance.

**Change.** For HYPE 1-hour only, after drawing the existing Student-t(5) shocks, multiply each simulated path by `exp(0.3 Z - 0.3²)` with one independently seeded normal draw per path. This preserves the shock stream and mean squared multiplier. Leave all other assets and formats untouched. Compare on August 15–16 and September 9–10 at exact prompts and 1000 paths.

**Decision rule.** Advance only if HYPE 1-hour CRPS improves on both windows, with a gain of at least 0.2% on one. If so, confirm on August 20–29 before editing the serving module.

**Result.** On August 15–16, HYPE 1-hour CRPS fell from 565.96 to 565.07 (0.16%). On September 9–10, it rose from 1211.58 to 1214.31 (0.23% worse). Each pair used 288 exact prompts, 1000 paths, and full realized coverage.

**Verdict.** Discard the HYPE pathwise multiplier under the two-window rule. Retain HYPE's selected Student-t(5) fan without extra pathwise scaling.

**Lesson.** Volatility-level uncertainty helped BTC 1-hour but did not transfer to HYPE on the active September sample.

## Iteration 18 — weekend-only stronger crypto 24-hour shrinkage (planned)

**Hypothesis.** The 0.70 variance factor sharpened the quiet August 15–16 weekend fan but made the active September 9–10 weekday fan too narrow. Applying 0.70 only to Saturday and Sunday crypto 24-hour starts, while retaining the selected 0.85 factor for weekdays, may lower mixed-window CRPS without weekday regressions.

**Change.** In an isolated candidate, change only the crypto 24-hour variance factor: 0.70 for weekend starts and 0.85 otherwise. Calendar profiles, shocks, seeds, and all other formats remain unchanged. Compare against the selected 0.85 rule on September 3–12, then August 5–14 if the first window passes, using all five assets at 1000 paths and exact prompt starts.

**Decision rule.** Keep only if five-asset mean CRPS improves on both mixed windows, no asset regresses more than 0.5% on either, and the independent August 20–29 confirmation also improves the mean without an asset regression above 0.5%. Stop after any failed gate.

**First mixed-window result.** On September 3–12, mean CRPS fell from 3787.20 for the selected 0.85 rule to 3785.15 for weekend-only 0.70 (0.05%). BTC, ETH, and XRP improved by 0.11–0.21%; SOL and HYPE regressed by 0.04% and 0.08%, below the 0.5% limit. Every asset had 240 exact prompts, 1000 paths, and full realized coverage. This narrowly passes the first gate.

**Second mixed-window result.** On August 5–14, mean CRPS fell from 2502.36 to 2501.46 (0.04%). BTC and ETH improved by 0.11% and 0.14%; XRP and HYPE improved by 0.01% and 0.04%; SOL regressed by 0.07%. Every asset had 240 exact prompts, 1000 paths, and full realized coverage. This narrowly passes the second gate. The independent August 20–29 comparison remains required.

**Independent confirmation.** On August 20–29, the five-asset mean fell from 6980.71 to 6964.79 (0.23%). All five assets improved, by 0.10–0.34%, on 240 exact prompts each with 1000 paths and full realized coverage. This passes the final predeclared gate.

**Verdict.** Keep 0.70 for weekend crypto 24-hour starts and 0.85 for weekday starts. Apply it to `modeling.py`, then verify served-path parity with the scored candidate and direct final-module CRPS. The incremental September 3–12 gain is only 0.05%, so the stronger independent August gain is important evidence for retaining the extra rule.

**Serving verification.** The selected conditional rule was applied to `modeling.py`. Path parity passed for all five crypto 24-hour assets on sampled weekend and weekday prompts against the scored candidate. A direct official September 3–12 run of the serving module exactly reproduced all five candidate asset CRPS values, with 240 exact prompts, 1000 paths, and full coverage per asset; mean CRPS was 3785.15. The 1-hour and COM_EQU code branches are unchanged from the prior verified serving commit.

**Lesson.** The volatility scale that best calibrated 24-hour crypto fans differed between quiet weekends and more active weekdays. A conditional rule preserved the broader 0.85 weekday fan while improving two separate August mixed windows.

## Iteration 19 — HYPE weekday HAR variance forecast (planned)

**Hypothesis.** The all-week HAR forecast improved HYPE on the September 9–10 weekday screen even though it regressed the other crypto assets. Extending the already selected weekend HAR rule to HYPE weekdays only may lower HYPE 1-hour CRPS without changing other assets.

**Change.** In an isolated candidate, use the July-trained HAR 1-hour variance forecast for HYPE on every day; retain the selected weekend-only HAR rule for BTC, ETH, SOL, and XRP. Keep shocks, calendar factors, path count, and seed unchanged. Compare paired official HYPE 1-hour CRPS on September 1–8 and August 20–29, which do not overlap the September 9–10 weekday screen.

**Decision rule.** Advance only if HYPE CRPS improves on both longer windows. Then confirm on September 3–12 before changing serving code; reject if either longer window regresses.

**First longer-window result.** On September 1–8, HYPE 1-hour mean CRPS fell from 1006.27 for the selected weekend-only HAR rule to 1000.31 with HAR on all days (0.59%). Both runs used 1152 exact prompts, 1000 paths, and full realized coverage. Proceed to the independent August 20–29 comparison.

**Independent August result.** On August 20–29, HYPE 1-hour CRPS fell from 1612.35 to 1590.96 (1.33%). The paired runs used 1651 exact prompts, 1000 paths, and full realized coverage. This passes the second gate; confirm on September 3–12 before serving.

**September confirmation.** On September 3–12, HYPE 1-hour CRPS fell from 1058.61 to 1054.19 (0.42%) on 1440 exact prompts, 1000 paths, and full realized coverage. The other four crypto 1-hour assets are unchanged, so the five-asset mean falls from 799.89 to 799.00 (0.11%).

**Verdict.** Keep HAR variance forecasting on weekdays for HYPE 1-hour only. Retain weekend-only HAR for the other crypto assets. Verify the consolidated serving output against the scored candidate before relying on the new mean.

**Serving verification.** Sampled HYPE 1-hour weekend and weekday paths from `modeling.py` were byte-identical to the scored candidate. The direct official September 3–12 serving-module backtest reproduced HYPE CRPS 1054.19 exactly, with 1440 exact prompts, 1000 paths, and full realized coverage. The updated five-asset CRYPTO_1H mean is therefore 799.00; unchanged assets retain their prior verified scores.

**Lesson.** The all-week HAR regression failed as a configuration-wide rule but was useful for HYPE on two longer forward windows. Asset-specific variance behavior mattered more than a single 1-hour rule.

## Iteration 20 — recent volatility blend for COM_EQU (planned)

**Hypothesis.** A seven-day calendar-adjusted variance level may lag volatility changes in tokenized equities and commodities. The crypto model's equal blend of seven-day and 24-hour-half-life variance might improve COM_EQU 24-hour CRPS after the calendar profile removes the strong hour and weekend pattern.

**Change.** In an isolated candidate, apply that equal recent blend to the eight COM_EQU assets only. Retain their August 31 calendar profiles, Student-t(5) shocks, seed, and all crypto behavior. Score September 1–2 exact prompts at 1000 paths against the already verified serving baseline. If it passes, score September 5–6 (weekend) with paired baseline and candidate, then September 9–10 as a separate active-weekday confirmation.

**Decision rule.** Continue only if the eight-asset mean improves in each window and no asset regresses more than 1% in any. Stop after the first failed gate; use the selected seven-day level otherwise.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2329.87 to 2315.33 (0.62%). Six assets improved; GOOGLX and TSLAX each regressed by about 0.04%, below the 1% limit. NVDAX supplied most of the gain, improving 3.85%. Every asset had 48 exact prompts, 1000 paths, and full coverage. This passes the first gate; test the September 5–6 weekend next.

**Weekend gate.** The first September 5–6 paired asset, XAU, rose from 734.53 to 745.61 (1.51% worse), exceeding the predeclared 1% asset limit. Both runs had 48 exact prompts, 1000 paths, and full coverage. The remaining weekend asset runs and later weekday confirmation were stopped because this result makes the all-asset candidate ineligible under the decision rule.

**Verdict.** Discard the COM_EQU-wide recent blend. Retain the seven-day calendar-adjusted level. NVDAX's weekday gain is a possible asset-specific research question, but the configuration-wide proposal failed on XAU's weekend regression.

**Lesson.** A recent volatility estimate can sharpen an active equity window while badly miscalibrating a quiet commodity weekend; the fitted calendar surface does not remove every regime difference.

## Iteration 21 — NVDAX-specific recent variance blend (planned)

**Hypothesis.** NVDAX's 3.85% September 1–2 gain from the recent variance blend may reflect a faster volatility regime shift in that market. Restricting the blend to NVDAX could retain its benefit without XAU's weekend regression.

**Change.** Use the same 50/50 seven-day plus 24-hour-half-life adjusted variance blend for NVDAX only, leaving the selected serving model unchanged for every other asset. Compare paired official NVDAX 24-hour CRPS on September 5–6 (weekend) and September 9–10 (weekday), each with 48 exact prompts and 1000 paths. If both pass, check the untouched September 11–12 dates before the full September 3–12 confirmation.

**Decision rule.** Advance only if NVDAX improves on all three two-day windows; reject on any regression. If it passes, keep only if the ten-day September 3–12 score also improves.

**Result.** On the first new window, September 5–6, NVDAX CRPS rose from 1027.71 to 1029.18 (0.14% worse) under the recent blend. Both paired runs had 48 exact prompts, 1000 paths, and full realized coverage. The remaining candidate windows were stopped because any regression failed the predeclared rule.

**Verdict.** Discard the NVDAX-specific recent blend. The September 1–2 gain did not carry to the weekend; keep the selected seven-day variance level.

**Lesson.** A large two-day gain can be regime-specific even for the asset that dominates a configuration-level screen.

## Iteration 22 — half-hour weekday COM_EQU calendar (planned)

**Hypothesis.** The hourly COM_EQU calendar profile blurs sharp volatility changes around US market openings. A 30-minute weekday profile, fitted by the same weekly-normalized squared-return procedure through August 31, may improve 5-minute path CRPS while preserving the established weekend profile and each asset's weekday mean variance.

**Change.** Fit 48 weekday time-of-day bins for each of eight COM_EQU assets, using the same 99.9th-percentile return cap, adjacent valid closes, `[1,2,1]/4` circular smoothing, and 15% shrinkage as the selected hourly fit. Rescale each new weekday profile to the mean of its existing August 31 hourly weekday profile; use the existing weekend factors unchanged. Crypto profiles, variance levels, shocks, and seeds remain unchanged. Screen all eight COM_EQU 24-hour assets on September 1–2, then September 5–6, then September 9–10 at exact prompts and 1000 paths.

**Decision rule.** Advance only if the eight-asset mean improves in each of the three windows and no asset regresses more than 1% on any. Stop at the first failed window. If it passes, confirm on September 3–12 before any serving change.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2329.87 with the selected hourly profile to 2313.99 with weekday half-hour bins (0.68%). All eight assets improved, by 0.07–1.36%, on 48 exact prompts each with 1000 paths and full coverage. The fitted weekday profiles peaked at 13:30 UTC for seven assets and 14:00 for SP500, supporting the market-opening mechanism. This passes the first gate; test the weekend next.

**Weekend result.** On September 5–6, the eight-asset mean fell from 1140.12 to 1136.96 (0.28%). Six assets improved; NVDAX and SP500 regressed by 0.16% and 0.26%, both below the 1% limit. All pairs had 48 exact prompts, 1000 paths, and full coverage. This passes the second gate; test the active September 9–10 weekday window next.

**Second weekday result.** On September 9–10, the eight-asset mean fell from 2396.84 to 2389.61 (0.30%). Six assets improved; GOOGLX and WTIOIL regressed by less than 0.01%. Each pair had 48 exact prompts, 1000 paths, and full coverage. All three short-window gates pass, so confirm on September 3–12 before serving.

**Ten-day confirmation.** On September 3–12, the eight-asset mean fell from 1953.75 to 1947.43 (0.32%). Seven assets improved by 0.06–0.55%; SP500 rose by 0.01%. Each candidate run had 240 exact prompts, 1000 paths, and full realized coverage. The mean improved in all four tested September windows.

**Verdict.** Keep the half-hour weekday profile for COM_EQU 24-hour, retaining the hourly weekend profile and crypto calendars. Apply the profile to `modeling.py`, then verify candidate parity and direct serving-module CRPS.

**Serving verification.** Sampled weekday and weekend output paths from `modeling.py` exactly matched the scored candidate for all eight COM_EQU assets. A direct official September 3–12 backtest of the updated serving module reproduced all eight candidate asset CRPS values exactly, with 240 exact prompts, 1000 paths, and full coverage each. The verified eight-asset mean was 1947.43. Crypto branches were not changed by this iteration.

**Lesson.** Preserving each asset's weekday average variance while resolving the opening period more finely improved most COM_EQU assets. This supports time-of-day shape as a useful calibration dimension beyond the broad weekday/weekend split.

## Iteration 23 — quarter-hour weekday COM_EQU calendar (planned)

**Hypothesis.** The 30-minute weekday profile still averages the first and second quarters after the US market opens. A 15-minute profile may improve path calibration near the opening without changing the weekday average variance or weekend factors.

**Change.** Fit 96 weekday bins through August 31 with the same adjacent-return filter, weekly normalization, 99.9th-percentile cap, circular `[1,2,1]/4` smoothing, and 15% shrinkage used for the selected 30-minute profile. Rescale each asset's weekday mean to the selected profile's mean and leave its weekend profile unchanged. Compare all eight COM_EQU 24-hour assets at exact September 1–2 prompts and 1000 paths. If that passes, check September 5–6, September 9–10, and September 3–12 in sequence.

**Decision rule.** Advance only if eight-asset mean CRPS improves on each window and no asset regresses more than 1%. Stop after the first failed window; otherwise replace only the COM_EQU weekday profile.

**First weekday screen.** On September 1–2, eight-asset mean CRPS fell from 2313.99 with 30-minute bins to 2307.95 with 15-minute bins (0.26%). All eight assets improved, by 0.01–0.57%, with 48 exact prompts, 1000 paths, and full coverage. This passes the first gate; test the weekend next.

**Weekend result.** On September 5–6, the eight-asset mean fell from 1136.96 to 1134.66 (0.20%). All eight assets improved, by 0.01–0.32%, on 48 exact prompts each with 1000 paths and full coverage. This passes the second gate; test September 9–10 next.

**Second weekday result.** On September 9–10, the eight-asset mean fell from 2389.61 to 2385.58 (0.17%). All eight assets improved, by 0.06–0.49%, with 48 exact prompts, 1000 paths, and full coverage. All three short-window gates pass; confirm on September 3–12 before serving.

**Ten-day confirmation.** On September 3–12, the eight-asset mean fell from 1947.43 with 30-minute bins to 1944.22 with 15-minute bins (0.16%). Seven assets improved by 0.10–0.32%; WTIOIL rose by 0.05%, below the 1% limit. Every asset had 240 exact prompts, 1000 paths, and full coverage. The mean improved in all four tested windows.

**Verdict.** Keep the quarter-hour weekday COM_EQU profile, with the same hourly weekend factors and other model components. Verify sampled path parity and direct serving-module scores before considering it consolidated.

**Serving verification.** Sampled weekday and weekend paths from `modeling.py` matched the scored quarter-hour candidate for all eight COM_EQU assets. A direct official September 3–12 backtest of the updated serving module reproduced all eight candidate asset CRPS values exactly, with 240 exact prompts, 1000 paths, and full coverage each. The verified eight-asset mean was 1944.22. Crypto serving logic was unchanged.

At 1000 paths on this hardware, sampled serving calls took 0.094 seconds for XAU 24-hour, 0.097 for NVDAX 24-hour, 0.099 for BTC 24-hour, and 0.069 for HYPE 1-hour; all returned the required path count and finite prices.

## Iteration 24 — five-minute weekday COM_EQU calendar (planned)

**Hypothesis.** Even the selected 15-minute profile may average away the unusually volatile first five minutes after the US cash opening. A 5-minute weekday profile could better calibrate the validator's 5-minute path increments.

**Change.** Fit 288 weekday time-of-day bins through August 31 using the selected profile's return filter, weekly normalization, 99.9th-percentile cap, circular `[1,2,1]/4` smoothing, and 15% shrinkage. Rescale each asset's new weekday factors to the selected 15-minute profile's mean; keep the hourly weekend factors, variance level, shocks, and seeds unchanged. Screen all eight COM_EQU assets on September 1–2 at exact prompts and 1000 paths. If it passes, test September 5–6, September 9–10, and September 3–12 in that order.

**Decision rule.** Advance only if the eight-asset mean CRPS improves on each window and no asset regresses more than 1%. Stop after the first failed window; otherwise replace only the COM_EQU weekday profile.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2307.95 with 15-minute bins to 2300.89 with 5-minute bins (0.31%). All eight assets improved by 0.07–0.85%, on 48 exact prompts each with 1000 paths and full realized coverage. The fitted first-five-minute equity spikes were larger than with coarser bins; XAU's maximum shifted to 01:00 UTC, so weekend and later weekday checks remain important.

**Weekend result.** On September 5–6, the eight-asset mean fell from 1134.66 to 1132.62 (0.18%). All eight assets improved, by 0.05–0.52%, on 48 exact prompts each with 1000 paths and full coverage. This passes the second gate; test September 9–10 next.

**Second weekday result.** On September 9–10, the eight-asset mean fell from 2385.58 to 2381.50 (0.17%). All eight assets improved, by 0.04–0.45%, on 48 exact prompts each with 1000 paths and full coverage. All three short-window gates pass; confirm on September 3–12 before serving.

**Ten-day confirmation.** On September 3–12, the eight-asset mean fell from 1944.22 with 15-minute bins to 1940.77 with 5-minute bins (0.18%). All eight assets improved, from 0.04% for WTIOIL to 0.31% for GOOGLX. Each candidate result used 240 exact prompts, 1000 paths, and full realized coverage. The predeclared mean and per-asset gates pass.

**Verdict.** Keep the five-minute weekday profile, preserving hourly weekend factors and every other component. Apply it to the serving module and verify direct scores.

**Serving check.** Sampled `modeling.py` paths were exactly equal to the scored candidate for all eight COM_EQU assets on both weekday and weekend starts, including a context with NaN closes. Direct official September 3–12 serving-module backtests reproduced all eight candidate CRPS values exactly, with 240 exact prompts, 1000 paths, and full coverage per asset. The verified eight-asset mean is 1940.77.

**Lesson.** The first quarter-hour after the market opening carried enough distinct volatility to improve CRPS beyond half-hour bins, though the incremental gain was small and one commodity regressed slightly on the longer window.

## Iteration 25 — weekday-only recent variance for NVDAX (planned)

**Hypothesis.** The earlier NVDAX recent-variance blend improved its September 1–2 weekday CRPS by 3.85% but regressed on the September 5–6 weekend. Limiting the blend to weekday starts may capture faster weekday volatility changes without the weekend failure.

**Change.** Apply the 50/50 seven-day and exponentially weighted recent calendar-adjusted variance only to NVDAX 24-hour prompts starting Monday through Friday. Keep the currently selected calendar profile, weekend behavior, paths, shocks, and seed unchanged. Use a separate candidate module and isolated predictions.

**Decision rule.** Compare paired official NVDAX 24-hour CRPS at 1000 paths on September 9–10, then September 11–12, then the full September 3–12 window. Advance only if every comparison improves, with full realized coverage. Stop at the first regression.

**Result.** On the independent September 9–10 weekday window, NVDAX CRPS rose from 1948.08 with the five-minute calendar alone to 1948.61 with the weekday recent-variance blend (0.027% worse). Both scores used 48 exact prompts, 1000 paths, and full realized coverage. The September 11–12 and full-window checks were stopped under the prespecified rule.

**Verdict.** Discard the weekday NVDAX blend. The September 1–2 benefit did not repeat on the later active weekday screen.

**Lesson.** Even when the weekend is excluded, a faster volatility estimate can lose its short-window benefit in a nearby equity regime.

## Iteration 26 — stronger 1-hour tails for ETH, SOL, and XRP (planned)

**Hypothesis.** Student-t(3) improved these three crypto 1-hour assets over Student-t(5) on both short windows and a separate August holdout. A modestly heavier unit-variance Student-t(2.5) law may improve CRPS further, especially around intrahour jumps.

**Change.** In an isolated candidate, change only the degrees of freedom for ETH, SOL, and XRP 1-hour shocks from 3 to 2.5, normalizing the variance by `sqrt((2.5-2)/2.5)`. Leave their calendar, variance forecast, seed, and all other branches unchanged.

**Decision rule.** Compare official 1000-path CRPS against the selected serving model on September 1–2, then September 5–6. Advance only if the three-asset mean improves on both windows and no asset regresses more than 0.5% on either; then confirm on September 3–12. Stop at the first failed window.

**Result.** On September 1–2, the three-asset mean rose from 919.99 to 924.62 (0.50% worse). ETH rose 772.28 → 777.12 (0.63%), SOL 946.04 → 947.36 (0.14%), and XRP 1041.65 → 1049.38 (0.74%). Each pair had 288 exact prompts, 1000 paths, and full realized coverage. The mean and two per-asset limits fail; later windows were not run.

**Verdict.** Discard Student-t(2.5) for 1-hour ETH, SOL, and XRP. Keep the selected Student-t(3) law.

**Lesson.** Past evidence for fatter-than-Student-t(5) tails did not imply that still heavier tails improve CRPS; the selected tail weight is near a useful range for these assets.

## Iteration 27 — day-of-week COM_EQU variance factors (planned)

**Hypothesis.** The selected weekday/weekend calendar may leave systematic Monday-through-Friday and Saturday/Sunday variance differences. A regularized day-of-week factor estimated from pre-September residual squared returns could improve 24-hour CRPS without changing the five-minute time-of-day profile.

**Change.** For each COM_EQU asset, fit seven day factors through August 31 from squared adjacent returns divided by the selected five-minute calendar factor, after weekly normalization and 99.9th-percentile return capping. Normalize the five weekday factors and two weekend factors separately to mean one, then shrink them halfway toward one. Multiply both historical and future calendar factors by these day factors. Keep all other components unchanged.

**Decision rule.** Compare all eight COM_EQU assets at 1000 paths on September 1–2, September 5–6, and September 9–10 in that order. Continue only if the eight-asset mean improves in each window and no asset regresses more than 1%. If all short gates pass, confirm on September 3–12 before changing serving code.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2300.89 to 2299.16 (0.075%). Five assets improved and three regressed by at most 0.11%; NVDAX gained 0.64%. Every asset had 48 exact prompts, 1000 paths, and full realized coverage. This passes the first gate, though the effect is small; test September 5–6 next.

**Weekend screen.** On September 5–6, the eight-asset mean fell from 1132.62 to 1131.63 (0.087%). Five assets improved; GOOGLX, SPCX, and SP500 regressed by 0.04%, 0.09%, and 0.19%, respectively, all below the 1% limit. Each result used 48 exact prompts, 1000 paths, and full coverage. This passes the weekend gate; check September 9–10 next.

**Second weekday screen.** On September 9–10, the eight-asset mean rose from 2381.50 to 2381.95 (0.019% worse). Five assets regressed and three improved; NVDAX rose by 0.31%. All had 48 exact prompts, 1000 paths, and full coverage. The mean gate fails, so the full September confirmation was not run.

**Verdict.** Discard the day-of-week factors and retain the selected five-minute weekday/hourly weekend calendar.

**Lesson.** A plausible Wednesday variance peak in the fit sample gave small initial gains but did not generalize across two September weekday windows. Day-specific adjustments need more evidence than the available history provides.

## Iteration 28 — HAR-style crypto 24-hour variance forecast (planned)

**Hypothesis.** The current equal blend of seven-day and 24-hour-half-life variance may miss changes in volatility persistence relevant to a full-day crypto fan. A July-trained forecast using the preceding 1-hour, 6-hour, 24-hour, and 7-day adjusted variance may lower 24-hour CRPS.

**Change.** Fit a regularized per-asset log-variance regression on January–July crypto minute data, with a forward 24-hour adjusted-variance target whose end remains in July. Use it for crypto 24-hour variance in an isolated candidate, preserving the selected calendar, weekend/weekday variance factors, shocks, seeds, and every COM_EQU or 1-hour branch.

**Decision rule.** Compare all five crypto 24-hour assets at 1000 paths on the independent August 5–14 window first. Advance only if mean CRPS improves and no asset regresses more than 1%. Then check September 9–10 under the same rule, followed by full September 3–12 confirmation if both gates pass. Stop at the first failed gate.

**Result.** On the independent August 5–14 window, the five-asset mean rose from 2501.46 to 2512.13 (0.43% worse). BTC, ETH, SOL, XRP, and HYPE regressed by 0.35%, 0.86%, 0.56%, 0.39%, and 0.15%, respectively. Each candidate used 240 exact prompts, 1000 paths, and full realized coverage. The selected serving module was sampled against the previously scored August baseline and produced identical crypto 24-hour paths on weekday and weekend starts. The first mean gate fails; September checks were not run.

**Verdict.** Discard the 24-hour HAR forecast. Retain the selected crypto 24-hour variance blend and weekend/weekday calibration factors.

**Lesson.** A regression that helped some 1-hour regimes did not improve day-ahead variance. The current simple blend generalized better on this independent August window.

## Iteration 29 — robust COM_EQU context variance (planned)

**Hypothesis.** A few extreme minutes in the prior seven days may dominate the calendar-adjusted mean squared return, leaving the next day's COM_EQU fan too wide after a transitory jump. Winsorizing only the highest 0.1% of adjusted squared returns may stabilize the variance estimate while retaining most jump information.

**Change.** For COM_EQU 24-hour only, cap each valid adjusted squared context return at its prompt-specific 99.9th percentile before averaging. Keep the five-minute weekday/hourly weekend calendar, shocks, seed, crypto branches, and all other behavior unchanged. Use an isolated candidate module.

**Decision rule.** Compare all eight COM_EQU assets at 1000 paths on September 1–2, September 5–6, and September 9–10, in order. Advance only if the eight-asset mean improves on every window and no asset regresses above 1%; then confirm on September 3–12. Stop at the first failed gate.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2300.89 to 2293.72 (0.31%). NVDAX improved by 2.61% and SPCX by 0.04%; the other six assets regressed by at most 0.14%. All runs had 48 exact prompts, 1000 paths, and full coverage. The mean and per-asset gates pass; check the weekend next.

**Weekend screen.** On September 5–6, all eight assets improved, with gains from 0.05% for SPCX to 1.08% for XAU. The eight-asset mean fell from 1132.62 to 1128.38 (0.37%). Every candidate used 48 exact prompts, 1000 paths, and full coverage. This passes the weekend gate; test September 9–10 next.

**Second weekday screen.** On September 9–10, the eight-asset mean rose from 2381.50 to 2382.71 (0.051% worse). Six assets regressed by 0.03–0.18%; NVDAX and TSLAX improved slightly. Every candidate had 48 exact prompts, 1000 paths, and full coverage. The mean gate fails, so the full September run was stopped.

**Verdict.** Discard broad 99.9th-percentile context variance winsorization. Keep the selected raw seven-day adjusted mean for COM_EQU.

**Lesson.** Capping rare context jumps helped a quiet weekend but slightly underpredicted an active weekday. The NVDAX gain on the first weekday screen was again not stable across another weekday window.

## Iteration 30 — antithetic COM_EQU 24-hour paths (planned)

**Hypothesis.** At 1000 paths, random draws can leave a fan slightly unbalanced even when the shock law is symmetric. Pairing each Student-t(5) shock path with its negative may reduce finite-fan CRPS error without changing the intended marginal distribution.

**Change.** For COM_EQU 24-hour only, draw half the requested shock paths and append their elementwise negatives, using the existing prompt seed. Handle odd path counts by retaining one unpaired draw. Keep variance estimation, calendar, shock degrees of freedom, all crypto behavior, and output shape unchanged.

**Decision rule.** Compare official 1000-path CRPS on September 1–2, September 5–6, and September 9–10, in order. Advance only if the eight-asset mean improves on every window and no asset regresses above 1%; then confirm on September 3–12. Stop at the first failed gate.

**First weekday screen.** On September 1–2, the eight-asset mean fell from 2300.89 to 2300.05 (0.036%). Six assets improved; GOOGLX and WTIOIL regressed by 0.05% and 0.08%. Every result had 48 exact prompts, 1000 paths, and full realized coverage. This passes the first gate, though the gain is near the Monte Carlo scale; test the weekend next.

**Weekend screen.** On September 5–6, the eight-asset mean fell from 1132.62 to 1131.73 (0.078%). Seven assets improved and SPCX regressed by 0.024%. Each run had 48 exact prompts, 1000 paths, and full coverage. The second gate passes; check September 9–10 next.

**Second weekday screen.** On September 9–10, the eight-asset mean fell from 2381.50 to 2380.85 (0.027%). Six assets improved, while NVDAX and SPCX regressed by 0.014% and 0.046%. Every asset used 48 exact prompts, 1000 paths, and full coverage. All three short-window gates pass; confirm on September 3–12 before changing serving code.

**Additional robustness rule.** If the ten-day confirmation passes, compare baseline and antithetic fans again on September 1–2 and September 9–10 under a different explicit `SYNTH_BENCHMARK_SEED`. Advance to serving only if the eight-asset mean improves on both alternate-seed windows and no asset regresses above 1%. This checks whether the small gains depend on the local fallback random stream.

**Ten-day confirmation.** On September 3–12, the eight-asset mean fell from 1940.77 to 1939.94 (0.043%). Seven assets improved by 0.015–0.085%; SP500 regressed by 0.006%. Each candidate run had 240 exact prompts, 1000 paths, and full realized coverage. The ten-day gate passes; run the alternate-seed check before serving.

**Alternate-seed first weekday check.** With `SYNTH_BENCHMARK_SEED=20261001`, the paired September 1–2 eight-asset mean fell from 2301.34 to 2300.34 (0.043%). Seven assets improved; WTIOIL regressed by 0.010%. Every pair had 48 exact prompts, 1000 paths, and full coverage. The first seed-robustness gate passes; check September 9–10 next under the same seed.

**Alternate-seed second weekday check.** Under the same alternate seed, the September 9–10 eight-asset mean fell from 2381.54 to 2380.80 (0.031%). All eight assets improved, though AAPLX, SPCX, and SP500 were nearly flat. Each pair had 48 exact prompts, 1000 paths, and full coverage. Both seed-robustness gates pass.

**Verdict.** Keep antithetic COM_EQU 24-hour paths. Apply the rule to the serving module, verify path parity, and directly score the serving module before consolidating its results.

**Serving verification.** `modeling.py` and the scored candidate produced exactly equal paths for all eight COM_EQU assets on weekday and weekend starts, with both 31 and 1000 paths and a context containing NaN closes. Direct official September 3–12 serving backtests reproduced all eight candidate CRPS values exactly, with 240 exact prompts, 1000 paths, and full realized coverage per asset. The verified eight-asset mean is 1939.94.

## Iteration 31 — antithetic crypto 24-hour paths (planned)

**Hypothesis.** The same antithetic pairing that improved COM_EQU may reduce finite-fan CRPS error for crypto 24-hour paths under the selected Student-t(5) shock law.

**Change.** Pair each crypto 24-hour shock path with its negative, retaining the selected crypto calendar, variance blend, weekend/weekday factors, seed, and all COM_EQU or 1-hour behavior. Use an isolated candidate and honour any requested path count.

**Decision rule.** Compare all five crypto 24-hour assets at 1000 paths on independent August 5–14 exact prompts first, requiring a lower five-asset mean and no asset regression above 1%. If it passes, check September 9–10 under the same rule and then September 3–12. Stop at the first failed gate.

**Independent August holdout.** The five-asset mean fell from 2501.46 to 2501.02 (0.018%) on August 5–14. All five assets improved, by 0.002–0.039%, with 240 exact prompts, 1000 paths, and full realized coverage each. Sampled selected crypto 24-hour paths matched the previously scored August baseline on weekday and weekend starts. The first gate passes; test September 9–10 next.

**September weekday screen.** On September 9–10, the five-asset mean fell from 3813.28 to 3811.93 (0.035%). BTC, ETH, SOL, and XRP improved by 0.017–0.069%; HYPE regressed by 0.005%. Each pair had 48 exact prompts, 1000 paths, and full coverage. The second gate passes; confirm on September 3–12.

**Ten-day confirmation.** On September 3–12, the five-asset mean fell from 3785.15 to 3783.88 (0.034%). All five assets improved, from 0.007% for HYPE to 0.069% for XRP. Each candidate used 240 exact prompts, 1000 paths, and full coverage. The third gate passes.

**Additional seed check.** Because the gain is small, compare baseline and antithetic crypto fans at 1000 paths on September 9–10 using `SYNTH_BENCHMARK_SEED=20261001`. Advance to serving only if the five-asset mean improves and no asset regresses above 1% under this independent random stream.

**Landing-order stop and verdict.** The harness landing order arrived while the alternate-seed check was running. I stopped that check immediately. No conclusion is drawn from its partial results. The crypto antithetic candidate passed the three stated default-seed windows but did not complete its extra seed gate or direct serving verification. Discard it for this nomination and retain the already verified crypto 24-hour rule in commit `784cff3`.

**Lesson.** Antithetic sampling gave consistent small default-seed gains, including on an independent August holdout. The gains are below 0.04%, so the unfinished seed check leaves too much uncertainty for a last-minute serving change.
