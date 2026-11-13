# Predictable clocks, uncertain volatility

The selected model lowers historical CRPS by **3.80% for daily crypto, 12.16% for daily commodities/equities, and 5.29% for hourly crypto** against the seeded starter over 91 scored days. It combines learned volatility, separate future seasonal allocation, a modest scale mixture, and forward calendar effects. This report records the completed campaign, including failed experiments and the limits of the historical evidence. The nominated implementation is agent/modeling.py; agent/CHAMPION identifies its committed version.

## Context

Synth scores a distribution of future prices, not a single price forecast. Lower CRPS rewards both useful concentration and enough dispersion to cover what actually happens. We study all three requested configurations: crypto over 24 hours, commodities/equities over 24 hours, and crypto over one hour. Every reported model comparison uses the installed official backtester and **1000 paths**. Field rank is not an optimization target.

All prices come from the supplied frozen snapshot ending **November 8, 2026**. Event dates come exclusively from the supplied calendar, including upcoming dates already published before the cutoff. No later prices or scores were queried. Historical forecasts select only artifacts whose training data precede their context endpoint, and forward training labels are purged at each fitting cutoff.

The starter estimates one Gaussian volatility from seven days of minute returns. Its errors have different causes. The August crypto window includes an abrupt volatility expansion: XRP's recent minute volatility was about 9.7 basis points before a next-day realization near 38.4. By contrast, equity and commodity perpetuals have predictable weekday and session variation even though their venues trade continuously. A flat candle is a real observation; a missing price is not. Missing prices remain NaN, and no minute return crosses a missing observation.

## Approach

The core model first removes a seasonal variance profile from historical returns, estimates a changing volatility level, then restores the known future clock profile. Monthly seasonal fits use at most 120 earlier days. Profiles are specific to each asset, with partial pooling across weekdays and a separate weekend pattern. New York time is the default clock; the forecast follows daylight saving rather than freezing the US session to a UTC hour.

The volatility state combines fast and slow exponential averages. The one-hour reference combines 60- and 720-minute variance half-lives; the daily reference combines 720 and 4320 minutes. A small depth-three boosted regression predicts forward absolute-return scale relative to this reference. It sees only the requested asset's context, including several RMS and absolute-return volatility time scales, clock features and an asset indicator. Separate one-hour and daily fits use chronological validation and complete forward-label purging. Training and serving features were checked for exact float32 agreement.

The learned model is a scale model, with zero log-return drift. The retained simulator draws a Gaussian process through ranked, seeded Sobol coordinates and a Brownian bridge, then multiplies each entire log-return path by a lognormal volatility factor with log standard deviation .4 and mean one. This preserves mean absolute log-return scale. Calendar factors change forward variance without modifying historical observations. Distribution, calendar and forecast-curve components were evaluated separately before combination.

The requested context remains **seven days**. Fourteen or thirty days slightly worsen daily scores with the existing weights, while one-hour scores barely change. Three-day contexts serve successfully; matched cold-start training adds only 0.071% for commodities/equities and no crypto benefit, so it is discarded. Every simulator honors the requested path count and seeds its random generators from `SYNTH_BENCHMARK_SEED`.

## Experiments

Development scored windows end August 24 and October 12, 2026; confirmation windows end July 20 and November 8. Each covers seven days, sampled every 120 minutes with a 17-minute offset. Earlier checks end April 13 and June 15. The one-hour score always uses the current formula, including volatility blocks. These historical research samples are not an untouched final holdout. The starter, learned checkpoint, and combined model have also completed a 91-day test at the offline field's actual prompt timestamps.

**Seasonality and clocks.** Seasonal variation is the largest durable improvement. A UTC profile looks competitive in summer, but during the November clock change it raises commodity/equity CRPS by 1.67% against New York, including a 6.06% AAPLX regression. London timing can look good when the two regions share their usual offset, but fails the March US/UK daylight-saving mismatch: every tested asset/configuration cell worsens, with aggregate changes of +0.744% / +2.208% / +0.841% in configuration order. Gold and oil remain more nuanced than the equities.

Robust five-minute seasonal profiles improve aggregate scores but initially worsen November gold by 1.89%. The failure has a mechanical explanation. In a quiet Sunday period, an absolute-return profile gives very small seasonal denominators. Dividing a few genuine returns by those denominators inflates a squared-return volatility state. At the worst affected prompt, forecast scale rises from 6.50 to 11.66 basis points per five-minute step, against a realized absolute-equivalent scale of 5.62. Refitting the regressor under the substituted profile does not reliably fix this interaction.

Separating estimation from allocation is more successful: keep the original historical normalization and learned state, then use the robust profile only to allocate future dispersion, preserving the average step scale. This improves commodities/equities by 0.329% over four windows, in 27/32 asset-window cells. Its benefit remains 0.307% when combined with the Sobol scale mixture. Gold's November regression shrinks to 0.328%, but does not disappear. The candidate uses this separation for commodities/equities; crypto shape gains are smaller.

![A selected gold forecast shows why changing historical normalization can inflate the whole fan](figures/profile_normalization_failure.png)

This diagnostic example was selected because it was the largest regression. It explains the mechanism; the multi-window comparisons establish the aggregate evidence.

**Learned volatility versus simpler references.** The learned correction improves on a fully absolute-return seasonal EWMA, not only on the weaker squared-return starter. Adding range, volume, trade count and signed-return features provides little incremental benefit. Faster EWMA adaptation alone is not universally better; it can worsen the volatile August daily-crypto window.

**Distribution and finite-fan error.** A modest lognormal scale mixture, preserving mean absolute scale, improves the learned Gaussian forecast. Sobol coordinates and a Brownian bridge reduce finite-fan variation. Student-t(5) innovations look useful in development but their one-hour benefit reverses in November. Conditional volatility quantiles and an additional volatility process do not clearly beat the simpler fixed mixture.

A second simulation seed confirms the fixed mixture's daily gains: -0.208% for crypto24 and -0.270% for other24 across the four windows, with 19/20 and 32/32 asset-window cells improving. The one-hour mean improves only 0.055%, and August reverses sign. The ordinary Gaussian reference itself varies by 0.42% across seeds in that volatile week. A subsequent four-seed comparison uses Sobol bridge coordinates for both Gaussian and mixture fans, isolating distributional shape: crypto1 improves from a seed-average 875.098973 to 874.378343 (-0.08235%), with every seed/window pair improving. This is a small repeatable effect, and does not erase the earlier reversal. We do not choose a favorable seed.

**Known events.** Forward release kernels are fitted from earlier events relative to causal volatility forecasts. Coincident GDP/PCE releases are treated together, and correlated assets at one release do not count as independent events. Earnings have broad kernels because their timestamps are session anchors. In a 180-day event-centered test, forward calendar adjustments reduce one-hour crypto CRPS from 1622.017546 to 1581.481152 (-2.50%) and other24 from 3167.322481 to 3145.300386 (-0.70%). On ordinary sparse grids the effects are tiny and mixed. Removing event shocks from historical EWMA does not help; it is discarded.

NYSE closures are a distinct effect. Equity cash-session absolute-return scale is about 0.234 of a neighboring ordinary weekday's after both are normalized by their own causal forecasts. Leaving one closure date out changes that estimate only to 0.216–0.245. A conservative forward attenuation reduces CRPS on 232 closure-centered prompts from 1488.072747 to 1401.496293 (-5.82%), improving all five tested dates. These are conditional holiday results, not an all-prompt improvement of 5.82%. Two historical early-close dates are insufficient for a separate fitted effect.

In the complete 91-day attribution over the original Gaussian model, calendars improve CRPS 0.045% / 0.319% / 0.040%. On top of the separated-shape mixture, they improve 0.052% / 0.309% / 0.048%; seventeen of eighteen asset cells improve. The latter week-block uncertainty intervals exclude zero, but the intervals for the earlier Gaussian crypto attribution include zero. These small effects are sensitive to the reference model and are not selection-adjusted. Calendars slightly worsen the post-cadence and post-DST subsets in every configuration; the combined model still improves the simpler learned checkpoint in each recent subset.

Extra event-amplitude dispersion was screened out: most individual event types have narrower residual log-absolute spread than the existing Gaussian-plus-mixture law. Pooled spread partly reflects differences in type-level calibration. Replacing the fast one-hour event reference with the daily scale gains at most 0.033% on event-centered prompts, below the 0.5% requirement, so it is discarded.

Shortening the event-fitting history from 365 to 180 days worsens all three conditional scores, including hourly crypto by 0.311%. The longer history is retained. Sparse event data benefit from stability even when type-level residuals suggest drift.

![CRPS improvement on each historical closure date](figures/closure_results.png)

**Smaller or unsuccessful ideas.** A hand-designed future-volatility tilt gains only hundredths of a percent. Learned future segments improve other24 in all development cells, but their confirmation mean falls just short of the predeclared 0.2% threshold, and their incremental integration benefit is only about 0.096%. Minute-of-hour refinements improve all twenty one-hour asset-window cells but only by about 0.022%. Empirical innovation distributions preserve literal flat returns but worsen one-hour CRPS in the turbulent window. Directional forecasts worsen crypto and gain at most 0.031% elsewhere. Changing the regression's training horizon, regularization or row weighting does not meet the 0.2% threshold. Uniformly widening the fan helps the volatile August week but hurts October.

Three small recurrent encoders tested whether the order of recent absolute/RMS returns adds information beyond the EWMA summaries. Inputs cover four hours at five-minute resolution, two days at thirty-minute resolution, and seven days at four-hour resolution. Twelve chronological fits were trained on the GPU and exported for CPU inference. Raw neural forecasts and a neural/tree blend both fail the predeclared 0.3% CRPS threshold; the blend's gains are only 0.106% / 0.022% / 0.038%. Successful neural training is not evidence of a better forecast. The original small tree model remains the reference.

Deeper trees improve at most 0.123%, failing the 0.2% screen. A standardized ridge reference and its tree blend worsen all three configurations. Risk-weighted future labels gain at most 0.022%. Changing the future-profile lookback from 120 to 60 or 240 days adds no useful benefit. Antithetic scale sharing removes numerical log-price drift, but worsens hourly crypto by 0.013%; exact symmetry alone is not a better finite fan.

Gamma-deviance learning targets an arithmetic scale rather than a geometric log-scale center. Its best development gain is 0.128%, with an October reversal; it is discarded. Separate daily crypto and commodity/equity regressions gain at most 0.058%, also below threshold. Reallocating half the fixed log-scale variance from the path-wide factor to a transient state gains at most 0.023%. An explicit zero-return atom with a mean-absolute-scale-preserving continuous component gains at most 0.036%. None qualifies for retention.

A daily covariance perturbation preserves each step's Gaussian marginal variance while widening the terminal distribution. It gains only 0.035% / 0.029% on daily crypto and commodities/equities, below the predeclared 0.05% threshold; both exploratory uncertainty intervals include zero. Most of the benefit comes from terminal-price scoring, with negligible changes in the heavily weighted five-minute component. No covariance change is retained.

The transient-state experiment also exposed a sampling pitfall. Pairing two separately scrambled Sobol engines by row produced correlations approaching absolute one. That partial run was stopped and excluded. The corrected comparison used a single joint engine and a static reference with identical coordinates. Separate seeds alone do not make two deterministic point sets a sound joint sampling design.

A one-hour 1.05 width multiplier was transparently reopened after tighter sampling-precision measurements, with a revised threshold. This was a post-selection follow-up, not a pass under its original rule. It fails on three of four newly checked April/June/July/November windows and worsens overall CRPS by 0.052%. Width remains 1.0 and no further width grid follows.

The largest individual losses also fail to justify an automatic shock cap. A HYPE reversal produces a severely over-wide next-hour forecast, but similar low absolute/RMS states in earlier data are already broadly calibrated. The two worst HYPE excess scores together account for only about 0.007% of broad one-hour mean CRPS. A visually striking error is not necessarily a valuable general rule.

The journal records hypotheses before the tests and retains failed variants. `results/research_registry.csv` indexes completed suites; prompt-level parquet files preserve the official scores used in comparisons.

## Results

Core checkpoint results at 1000 paths:

| Window set and model | CRYPTO_24H | COM_EQU_24H | CRYPTO_1H |
|---|---:|---:|---:|
| Development: seeded starter | 4802.283427 | 2004.899638 | 1027.941065 |
| Development: seasonal EWMA | 4716.953273 | 1799.685377 | 998.559771 |
| Development: learned volatility | 4693.825150 | 1772.589566 | 981.761767 |
| Development: combined checkpoint | 4688.329960 | 1760.262716 | 979.570708 |
| Confirmation: seeded starter | 3831.087636 | 3114.391572 | 814.313609 |
| Confirmation: seasonal EWMA | 3730.014936 | 2912.866817 | 776.619142 |
| Confirmation: learned volatility | 3687.472104 | 2867.838497 | 770.135527 |
| Confirmation: combined checkpoint | 3674.787600 | 2854.320576 | 769.293836 |

The learned adjustment improves seasonal EWMA in 71 of 72 asset-window cells across these development and confirmation samples. Its benefit survives comparison with the stronger absolute-return reference.

At actual field prompt timestamps over 91 scored UTC days, August 9 through November 7 inclusive (the November 8 end boundary):

| Model | CRYPTO_24H | COM_EQU_24H | CRYPTO_1H |
|---|---:|---:|---:|
| Seeded starter | 4183.111568 | 2232.231176 | 909.562253 |
| Learned Gaussian checkpoint | 4036.953735 | 1979.165319 | 863.446546 |
| Separated shape and mixture | 4026.095454 | 1966.838705 | 861.899078 |
| Combined serving checkpoint | 4024.014528 | 1960.753172 | 861.489277 |
| Combined reduction vs starter | 3.803% | 12.162% | 5.285% |
| Prompts | 10920 | 16410 | 66012 |

All eighteen asset/configuration cells and all fourteen calendar-week blocks improve against both the starter and the learned Gaussian checkpoint. For the combined model versus the starter, paired calendar-week bootstrap intervals for percentage change are [-5.140, -2.794], [-13.567, -10.823], and [-6.708, -4.292]. These exploratory intervals retain assets and overlapping prompts together within each sampled week; they do not correct for model selection. After the October 30 cadence change, improvements remain 3.701% / 7.834% / 6.231%; after US DST ends, 4.667% / 8.039% / 5.834%.

On the earlier April/June windows, starter scores are 4957.191254 / 3066.525535 / 1062.128246, the learned checkpoint gives 4745.791114 / 2852.769395 / 1003.711716, and the combined model gives 4727.530003 / 2835.011416 / 1001.304924. The combination improves the learned model in 9/10, 14/14 and 10/10 asset-window cells. SPCX is excluded from both because it did not exist in April.

A final winter transfer check covers March 2–6, using the existing chronological fits and excluding SP500 and SPCX before their first trades. Starter scores are 6749.000506 / 3812.217022 / 1414.185462, versus 6627.236414 / 3705.503579 / 1388.618184 for the combined model: improvements of 1.804% / 2.799% / 1.808%. The combination also improves the learned Gaussian reference by 0.252% / 0.548% / 0.060%, in 5/5, 6/6 and 4/5 asset cells. The observed-price slice is complete, while AAPLX still has more than 55% zero-trade minutes. This is one calendar week, insufficient for a week-block uncertainty interval.

![Broad historical CRPS reduction for every supported asset and configuration](figures/broad_asset_improvements.png)

![Core CRPS reductions across development and confirmation](figures/core_improvements.png)

The final code passed 72 serving cases covering every asset/configuration pair with complete context, three real days, a two-day internal gap and a three-hour trailing gap. It also passed seed, odd-count, input-immutability and blocked-network checks with XGBoost imports actively blocked; the slowest measured call was 0.362382 seconds. A separate 24-case audit verifies absent, missing and empty calendar fallback, refresh and event removal at the same path, unknown event types, research-flag independence, and path counts 1, 3, 999, 1000 and 1001. A degenerate one-path stratification case was fixed before these studies. The expanded audit also exposed and fixed a stale calendar cache when a replacement preserved modification time; the reader now checks inode, size and metadata-change time too.

The first harness nomination failed because XGBoost was absent from its runtime, despite being installed in the workspace. Exporting the fitted numeric forests to NumPy arrays removes that serving dependency without changing the learned model. Float32 thresholds, missing-value branches, leaf values and ordered accumulation are preserved; predictions match XGBoost bit for bit on 49,152 checked rows across all 24 archived original forests. A fresh four-window official replay reproduces **all 6048 saved scores exactly**. All twelve retained chronological volatility models load without XGBoost, and required serving artifacts total **19,315,018 bytes**. The original weights remain archived for provenance.

An isolated copy of the nominated commit also passes all eighteen asset/configuration cases with both XGBoost imports and networking blocked, without falling back to working-tree files. At handoff the harness status still displays its initial missing-XGBoost error; no refreshed external pass was observed. The independent commit audit verifies that the reported dependency has been removed, but does not substitute for a new harness status.

The completed research-output audit checks **1,038,608 official candidate scores in 6698 files across 187 suites** with no invalid scores or count discrepancies. Repeated experimental scores are not independent observations.

Successful serving does not remove the cost of missing information. Synthetic trailing three-hour crypto gaps raise CRPS from 874.432272 to 928.563190 (+6.190%), with 71.9% of the penalty in terminal absolute-price scoring. Twelve-hour gaps raise it to 1019.903166 (+16.636%). Six-hour volatility aging improves those stressed scores by 0.158% and 0.494%, below the predeclared requirement of 0.5% under both durations. No aging branch is retained. These are synthetic gaps: the supplied snapshot has no internal missing closes or timestamp gaps after first trade.

The combined model also retains its CRPS gains with three real days and four leading NaN days. It improves the Gaussian reference given the same information by 0.226% / 0.555% / 0.177%, with 19/20, 29/32 and 19/20 asset-window cells improving. Leading NaNs and native three-day frames give exactly identical core forecasts in all eighteen interface combinations.

A more extreme stress test keeps only one fresh price or ten minutes of observations. A causal asset/horizon prior improves one-price crypto confirmation by 3.880% and 5.318%, but commodities/equities improve only 0.799%, with 7/16 cells improving. The ten-minute blend also fails its daily-crypto development screen. Both miss the preregistered all-configuration criteria and are discarded. With no adjacent observed returns, the serving model retains its existing common variance fallback of 1e-7 before the learned correction.

## Limitations

The windows are research samples and have been reused for model selection. Asset-window sign counts are stability diagnostics, not independent statistical trials. Adjacent forecast horizons overlap, and assets share market shocks. Event effects have especially few independent dates; a calendar can predict timing but cannot predict the sign or magnitude of an unscheduled surprise.

Sparse older listings have genuinely lower trading activity. Their flat observations must not be interpolated into a smooth efficient-price series. The model has no access to another asset's live prices and cannot retrieve a missing current price. Serving successfully with a stale context does not prove perfect calibration under that information loss.

The small distribution gains are sensitive to simulation noise unless sampling is controlled. At independent seed 91117 the combined model improves the Gaussian reference 0.219% / 0.577% / 0.042% on the four general windows, but August hourly crypto reverses by +0.164%. The four-seed Sobol-versus-Sobol comparison is the cleaner attribution of distributional shape. No favorable seed is selected. Conditional calendar gains must also be distinguished from the much smaller average-traffic gains.

Coverage diagnostics of the combined checkpoint show a remaining horizon mismatch. Its 90% intervals cover 88.7% / 88.3% / 89.0% of shortest returns, but 81.5% / 86.5% / 90.5% of terminal prices. Daily crypto's terminal coverage is only 68.1% during the August expansion. These are descriptive results on reused, overlapping samples, and are not an independent calibration test. They motivated the separately disclosed daily-covariance experiment, which failed its official-CRPS threshold.

![Coverage differs by horizon and volatility regime](figures/calibration_coverage.png)

## What we would do with more budget

The first priority would be prospective evaluation on untouched dates, logging valid price coverage, missed requests and score components alongside aggregate CRPS. Live measurements of context age and real history length would establish how often the synthetic gap and sparse-context failures matter. The sparse prior's stronger crypto results deserve a separately declared, asset-specific study with new confirmation data; they are not part of this champion.

More independent event dates would help distinguish event-type calibration from noise, especially for earnings and early closes. A longer history across listing-liquidity regimes would also support better seasonal pooling. The many failed width, capacity, distribution and covariance refinements suggest diminishing returns from further tuning on these same windows. Any renewed work should begin with a new source of information or new validation data.

Reproduction commands, serving dependencies and artifact checks are documented in REPRODUCE.md. The campaign used only the supplied prices/calendar and official scoring implementation; no post-cutoff prices or scores were consulted.
