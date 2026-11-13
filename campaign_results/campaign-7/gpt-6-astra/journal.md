# Research journal

The campaign is finalized under the harness landing order. The selected model is the explicit combined implementation in modeling.py. Historical "Pending" text below preserves the preregistration made before each run; subsequent result and verdict entries close those studies. The final report consolidates the retained and rejected components.

## Iteration 0 — establish the measurement and serving contract

**Hypothesis.** The seven-day constant Gaussian starter is miscalibrated during changes in volatility and across equity/commodity trading sessions. The 1h score also rewards a calibrated distribution of future realized volatility, which an independent Gaussian fan may miss.

**Change.** Read the campaign, data documentation, validator configuration and full CRPS implementation. Run `uv sync`. Preserve the starter in `baseline.py`, changing only its unseeded RNG to read `SYNTH_BENCHMARK_SEED` (fallback 20261108) for reproducibility. Extend the evaluator to retain prompt scores and use the library's compact prediction format.

**Result.** Pending baseline measurements. Exact scoring: 24h sums 288 five-minute, 48 thirty-minute, 8 three-hour and 1 absolute-terminal CRPS terms. The 1h score additionally includes initial-to-future gap distributions and volatility blocks with total coefficient 5.25. Context ends three minutes before a 24h prompt and two minutes before a 1h prompt; returns across missing context minutes must not masquerade as minute returns.

**Verdict.** Measurement setup in progress; no model selected.

**Lesson.** Forecasts must model increment distributions throughout the horizon, not just terminal prices. Local clocks, gaps and finite-ensemble uncertainty are substantive calibration issues.

### Iteration 0 measurements

Both development windows, 7 scored days each, every 120 minutes at minute 17, 1000 paths: baseline CRYPTO_24H **4802.283427** (840 prompts), COM_EQU_24H **2004.899638** (1344), CRYPTO_1H **1027.941065** (840). Realized coverage is 100%. Generation including load/write is roughly 0.039 s per 24h prompt and 0.011 s per 1h prompt. Results retained under `results/baseline_dev`.

Anomaly diagnosis before tuning: XRP's worst prompt begins 2026-08-21 10:17 UTC, CRPS 20949.405. Its seven-day context minute sigma is 9.727 bp versus 38.375 bp over the next day. No context price gaps. XRP daily sigma rises from 3.56 bp on August 18 to 10.40, 17.67, 21.34 and 38.09 on August 19–22. BTC and HYPE show the same volatility regime transition, so this is lagged scale estimation, not missing data. XRP's largest minute move is 508 bp. At the same start, HYPE's context/future sigmas are 12.347/24.253 bp. SPCX has large cash-open minute moves (160 bp at 13:30 UTC on August 19), while weekend sigma collapses to 1.75–2.68 bp. SP500 behaves differently and must retain its own profile.

## Iteration 1 — faster volatility adaptation

**Hypothesis.** A causal EWMA will reduce scale lag in the August volatility expansion and in quieter October. Absolute-return estimates may avoid persistent over-widening after isolated jumps, but could underpredict sustained turbulent regimes.

**Change.** Keep Gaussian path generation and its random numbers unchanged. Compare variance EWMA half-lives of 6/24/72 hours for 24h, 30/120/360 minutes for 1h; a 75/25 mixture of 12h/72h (1h: 60min/12h); and absolute-return and blended robust versions of that mixture. Only adjacent real minute returns enter estimation. Retain 7 context days.

**Result.** Pending. Predeclared acceptance: at least 1% mean CRPS improvement on a target configuration, improvement in both windows and a majority of its asset-window cells. If improvements conflict, isolate configuration choices and confirm on an older and a recent window.

**Verdict.** In progress.

**Lesson.** Pending.

### Iteration 1 completed

Development results (CRYPTO_24H / COM_EQU_24H / CRYPTO_1H):

| Estimator | CRPS 24h crypto | CRPS 24h other | CRPS 1h crypto |
|---|---:|---:|---:|
| Starter | 4802.283427 | 2004.899638 | 1027.941065 |
| EWMA mixture | 4766.660089 | 1988.453247 | 1006.719813 |
| 6h / 30min EWMA | 4794.711532 | 1980.164247 | 1007.004892 |
| 24h / 120min EWMA | 4767.358610 | 1997.048394 | 1011.393576 |
| 72h / 360min EWMA | 4775.446689 | 2004.108405 | 1018.466862 |
| Absolute-return EWMA mixture | 4736.703600 | 1912.064334 | 989.903107 |
| 50% absolute / squared-return mixture | 4739.891493 | 1943.489557 | 996.968845 |

**Verdict.** Keep robust scale estimation as a candidate, with confirmation required. Do not select the fastest 24h estimator: it worsens August crypto by 0.18% despite helping October. Raw mixed EWMA gains only 0.74% in crypto 24h, below the predeclared 1% threshold; its role is a fixed reference for the independent seasonality experiments. The absolute-return estimator's lower scale is useful, but will be rechecked after changing the innovation distribution to avoid double-counting tail concentration.

**Lesson.** A faster variance response is not automatically a better day-ahead forecast. Tail robustness and known future session structure are more promising than a single short half-life.

## Iteration 2 — causal local-clock seasonality

**Hypothesis.** Predictable variation over the trading week explains a large part of the commodity/equity CRPS deficit. New York time should transfer through DST better than UTC for the equities; London may fit gold better. The effect on crypto should be smaller and asset-dependent.

**Change.** Fit 15-minute variance bins for each weekday using up to 120 days before monthly fit cutoffs, and select only a fit whose cutoff is before the forecast's context endpoint. Normalize training squared returns by a 15-day local variance level; cap isolated normalized squared returns at 100. Shrink each weekday's profile 65% toward its weekday/weekend pool and apply gentle smoothing. Estimate a de-seasonalized EWMA variance level from the context, then restore the future clock profile. Compare New York, UTC, London, partial-strength and robust-scale variants. Calendar events are not yet used. Artifacts include as-of fit dates so historical tests cannot access later prices.

**Result.** Descriptive measurements: summer weekday New York peak/trough minute-sigma ratios are BTC 2.26, ETH 2.06, SOL 1.93, XRP 1.90, HYPE 1.67; XAU 4.07, NVDAX 5.31, GOOGLX 5.41, TSLAX 7.34, AAPLX 5.62, WTIOIL 2.71, SPCX 4.51, SP500 3.54. Most equities peak in the 09:00 New York hour. Full-history five-minute variance divided by five times minute variance is 0.89 for AAPLX and 0.93 for NVDAX, about 0.98–1.02 for most others. These are descriptions, not CRPS results. Paired scoring pending.

**Verdict.** Require at least 2% COM_EQU_24H CRPS improvement on both development windows and a majority of asset-window cells. Retain crypto seasonality only if its own comparisons improve. Clock selection needs a winter/summer confirmation, not only summer development scores.

**Lesson.** SP500's weekend and intraday behavior differs from the tokenized single stocks; no shared equity profile is imposed.

### Iteration 2 first result

New York seasonality at 1000 paths: CRYPTO_24H **4716.953273**, COM_EQU_24H **1799.685377**, CRYPTO_1H **998.559771** on the development grids. These improve the starter by 1.78%, 10.24%, and 2.86%. This is a provisional candidate; clock, shrinkage and robust-scale comparisons are still running. Exact field-score diagnostics on August 19–23 show that the crypto turbulence affected every miner: XRP 24h starter mean 12096.869, field median 11882.805, best-per-prompt 11253.664; NVDAX starter 1996.406 versus field median 1820.619. The 1h field diagnostic used the historical pre-volatility formula and is not used for current-formula model comparisons.

### Iteration 2 confirmation and clock result

Independent 7-day scored windows ending 2026-07-20 and 2026-11-08, identical 120-minute offset grids and 1000 paths: baseline CRPS is **3831.087636 / 3114.391572 / 814.313609**, New York seasonal EWMA **3730.014936 / 2912.866817 / 776.619142** (crypto24 / other24 / crypto1). Every one of the 36 asset-window cells improves. Seasonality adds to the EWMA itself on 16/16 other24 development cells and 10/10 crypto1 cells.

UTC slightly beats New York in summer but fails the November session shift: other24 CRPS in the November confirmation week is 3327.0802 with UTC versus 3272.3502 with New York, **+1.6725%**. AAPLX's UTC regression is 6.06%, GOOGLX 2.17%, SPCX 2.59%, NVDAX 1.28%, SP500 1.62%, TSLAX 1.16%. Conversely XAU improves 1.84% and WTIOIL 0.67% with UTC. Clock choice must therefore remain asset-specific, and the New York/London alternative will be tested over the March 2026 interval when the US has changed clocks but the UK has not. Predeclare a 7-day scored window ending 2026-03-16, excluding SP500/SPCX because they had not listed.

Development shrinkage/tail-scale ablations: reducing seasonal strength to 0.7 gives **4719.489370 / 1821.521484 / 998.352514**, worse than full strength for other24. A 50% absolute-return scale blend gives **4702.273766 / 1791.503371 / 990.611956**. Retain the full seasonal profile; robust scale remains a candidate alongside distributional changes.

## Iteration 3 — innovation tails and uncertain future volatility

**Hypothesis.** A forecast of future variance is not a known constant. Independent Student innovations or a mixture over future volatility may fit the concentration and tails of returns better, and a volatility process should improve the 1h block-volatility CRPS.

**Change.** Hold the New York seasonal EWMA variance forecast fixed. Compare variance-standardized Student t innovations (df 3, 5, 8), a path-wide lognormal volatility multiplier (log standard deviation .2, .4, .6), and a stationary log-volatility AR(1) (.4, .6, half-life 30 minutes for 1h or 180 minutes for 24h). Normalize each multiplier to preserve its expected squared scale. Seed all NumPy draws from the benchmark seed. No change to the estimator or fitting data.

**Result.** Pending. Require an improvement in both development windows and most asset-window cells, then confirm on disjoint recent and older periods. A 0.2% total-CRPS improvement is the minimum worth carrying to confirmation; smaller changes need sampling-error controls first.

**Verdict.** In progress.

**Lesson.** Pending.

## Iteration 4 — can learned volatility add to the EWMA?

**Hypothesis.** Multiple past volatility timescales may support a better forecast of average future scale. Range, volume, trade counts and signed returns must demonstrate incremental benefit over those volatility features, not merely correlate with future volatility on their own.

**Change.** Construct causal features and forward labels from the snapshot. Fit small gradient-boosted trees to log future absolute-return scale relative to the seasonal EWMA, comparing volatility-only features against an extended feature set. Training targets are forward 1h minute returns or forward 24h five-minute returns, de-seasonalized with the profile available at the forecast date. Use chronological training cutoffs and purge labels that end after each cutoff. The output adjusts scale, never price drift. One GPU training job at a time, limited depth and two CPU threads. First verify vectorized training features exactly match the serving calculation.

**Result.** Feature parity checks on BTC 1h, BTC 24h and AAPLX 24h give maximum float32 feature difference 0 and baseline-scale ratios within 3e-14 of 1. The first check exposed pandas' read-only NumPy views; fixed by making the EWMA work arrays explicit copies. No model CRPS measured yet.

**Verdict.** Keep only if official CRPS improves by at least 0.3% on its target configuration, in both development and confirmation regimes, with most asset-window cells improving. If the extended model cannot beat the volatility-only model, reject its extra features.

**Lesson.** Training/serving feature parity and chronological label purging are necessary before comparing a learned model.

## Iteration 5 — scheduled event kernels and post-event normalization

**Hypothesis.** Known release times can correct predictable variance spikes, while treating a temporary event shock as a permanent volatility regime over-widens subsequent forecasts. Event effects must be estimated on distinct dates, with coincident releases combined.

**Change.** Measure each supplied-calendar release relative to the causal seasonal/learned volatility forecast made 35 minutes beforehand. Fit conservative amplitude kernels by event type, asset group and asset, using only completed past events before each model cutoff. GDP/PCE coincidences are one event. Earnings use broad 30/120/240/480-minute windows because their supplied times are session anchors. Diagnose stability by leaving out every event date in turn; backtested forecasts use strictly earlier event fits, never a fit including the tested release. Compare forward macro effects, all forward effects, half-strength effects, and removal of known event variance from the historical EWMA. No invented dates or downloaded calendars.

**Result.** Anomaly diagnosis: AAPLX's 2026-11-03 00:17 forecast has predicted daily volatility 609.81 bp versus 313.48 bp realized. Its context includes large moves at 22:27–22:32 UTC on November 2, after the calendar's 21:00 earnings anchor. SPCX on July 17 has a similar unscheduled late-day shock, predicted 814.35 bp versus 529.44 bp realized; this cannot be fixed by a calendar alone. A first matched-weekday EIA analysis was uninformative because nearly every Wednesday control is itself an EIA release; use residuals against the seasonal forecast instead. CRPS comparisons pending.

**Verdict.** Require positive forward-only event results across multiple dates, plus no material regression on the four general windows. Report conditional event CRPS separately from all-prompt CRPS. Do not count correlated assets at one release as independent events.

**Lesson.** The calendar helps explain both under-dispersion before a scheduled release and over-dispersion after it; earnings timestamps have materially lower precision than macro timestamps.

## Iteration 6 — distinguish learning from simple calibration

**Hypothesis.** Some of the learned model's gain may be ordinary width correction. A fair reference is a fully absolute-return seasonal EWMA and a few global width adjustments, not only a raw squared-return EWMA. Distribution shape should preserve mean absolute scale when the estimator was trained on mean absolute returns.

**Change.** Compare 75%/100% robust absolute-return scale and global Gaussian scales .8/.9/.95/1.05. Then compare learned-scale fans with mean-absolute-scale-preserving Student innovations and lognormal path-scale mixtures. Test antithetic and stratified normals to separate simulation noise from model improvements. All use 1000 paths and benchmark-seeded randomness.

**Result.** Pending. The volatility-only learned model improves seasonal EWMA on 71/72 development/confirmation asset-window cells; however the stronger simple-scale references have not yet been tested.

**Verdict.** Extra model complexity needs an improvement beyond the stronger reference, on both development and confirmation windows. Distribution candidates must improve at least 0.2% in a target configuration across both regimes; smaller changes need seed replication.

**Lesson.** Pending.
## Iteration 7 — profile resolution and robust seasonal shape

**Hypothesis.** Fifteen-minute variance bins may smear the cash-open burst, and squared returns can let isolated releases dominate ordinary-day shape. More precise five-minute bins or absolute-return profiles may improve calibration. Weekday pooling may also smear Friday/Sunday transitions.

**Change.** Fit causal monthly profiles at 5-minute resolution, at 15-minute resolution using mean absolute returns, and with both changes. Compare weaker weekday shrinkage (70%/100% individual weekday estimates) and pooling only Monday–Thursday, leaving Friday/Saturday/Sunday distinct. Keep the already-fitted ML adjustment and its features fixed to isolate the seasonal forecast component. If a new profile succeeds, revisit matched feature retraining rather than assuming the old adjustment stays optimal.

**Result.** Pending. Descriptive return variance scaling in June–November is close to diffusive for most assets: five-minute variance ratios are about .98–1.07 of five times minute variance. Longer crypto intervals show mild mean reversion, but the likely aggregate gain from that smaller effect is less than the remaining seasonality/calibration questions.

**Verdict.** Require improvements on both development windows, most asset-window cells, and independent confirmation. Prefer the lower-resolution or more strongly pooled model when differences are within sampling noise.

**Lesson.** Pending.

## Iteration 8 — conditional uncertainty in future volatility

**Hypothesis.** A fixed lognormal scale mixture cannot adapt its uncertainty to the current regime. Quantile regressions of future absolute-return scale may improve the distribution of volatility while preserving the causal features and training protocol of the successful point model.

**Change.** Fit seven conditional quantiles of log future scale, using only volatility features, the same purged past validation split and timestamped training cutoffs. Test the raw conditional scale distribution, a version rescaled to the point model's mean, and optional within-path log-volatility fluctuations. The latter separates uncertainty in the horizon average from variation inside it. Training uses a quantile-loss surrogate; model acceptance still uses official CRPS only.

**Result.** Pending.

**Verdict.** Require at least 0.2% improvement beyond the best fixed-shape learned-scale fan on two regimes and a majority of asset-window cells; retain the simpler distribution if gains fail confirmation.

**Lesson.** Pending.

## Iteration 9 — broad actual-prompt baseline

**Hypothesis.** Small gains on sparse grids need a broader test at actual issued/scored start times, especially after the October cadence change.

**Change.** Establish a 91-day baseline ending November 8 on the offline field's actual prompt timestamps, with 1000 paths and the current formula. Use CRPS-only scoring against the snapshot, retaining coverage and per-prompt results; no field rank is optimized. The selected final candidate will be compared on exactly these timestamps. This is a robustness study across the available recent regimes, not a newly claimed untouched holdout.

**Result.** Pending. The current checkpoint has passed 72 serving scenarios across all 18 asset/configuration pairs: complete context, only three real days, a two-day internal gap, and a three-hour trailing gap. It also honors odd path counts, repeats exactly under the same benchmark seed, changes under another seed, leaves inputs unchanged, and serves with network connections blocked. Slowest measured call: 0.676933 s.

**Verdict.** Broad scoring and ordinary-context robustness will decide whether the small conditional improvements survive deployment-like prompt sampling.

**Lesson.** Pending.

## Iteration 10 — reduce finite-fan sampling error

**Hypothesis.** With only 1000 paths, random fan noise creates a small avoidable CRPS cost. Stratified marginals and a seeded Sobol Brownian bridge may improve both increment and gap distributions while leaving the intended stochastic process unchanged.

**Change.** Compare direct Sobol increments, Sobol Brownian bridges on the actual forecast variance clock, rank-stratified coordinates, and a bridge with the same mean-preserving .4 log-scale mixture. Generate exactly the requested number of paths. Cache only deterministic coordinates keyed by the benchmark seed, count and dimension. The bridge is a sampler, not a new scoring formula.

**Result.** Pending. The theoretical order of finite-ensemble CRPS bias is 1/N, so a gain much larger than about a tenth of a percent from sampling alone would need explanation. A smaller consistent improvement is acceptable here after independent-seed replication, unlike an unconfirmed extra forecasting feature.

**Verdict.** Prefer a sampler only if it improves the paired score consistently, passes odd-count/seed/interface tests, and stays well below the serving latency ceiling.

**Lesson.** Pending.

## Iteration 11 — context length and cold-start calibration

**Hypothesis.** Seven days may provide useful state information beyond the three real days available to some new live miners. A 14- or 30-day context might stabilize slow features, but it could also dilute recent regimes.

**Change.** Compare the current checkpoint under literal CONTEXT_DAYS values 3, 14 and 30, holding its already-fitted model fixed. A three-day frame is equivalent, for the volatility-only features, to a seven-day frame whose oldest four days are NaN. These first experiments diagnose sensitivity; if a different length helps, its learned features need matched retraining before attributing the gain to an optimal model at that length.

**Result.** Pending CRPS; the separate interface audit already passes three-real-day contexts without errors.

**Verdict.** Preserve reliable serving even if a short context modestly degrades calibration. Change the chosen length only after a gain across regimes, including the early/thin asset cases, with matched training checked.

**Lesson.** Pending.

## Iteration 12 — a time-varying forecast of volatility

**Hypothesis.** A predicted average scale over the next hour/day should not imply an identical de-seasonalized scale at every future instant. Recent volatility contains stronger information about the near future than the far future.

**Change.** Use the ratio of fast to slower absolute-return EWMAs to tilt the future volatility curve with exponential decay. Preserve the mean de-seasonalized scale forecast exactly. Compare short and long decay times and half-strength tilting, leaving the fitted point model, calendar-free seasonal profile and Gaussian draws unchanged.

**Result.** Pending.

**Verdict.** Require at least 0.2% CRPS improvement in the target configuration on both development windows and most cells, followed by confirmation. Reject merely moving variance toward the first steps if it hurts the rest of the scored horizon.

**Lesson.** Pending.


## Interim review after iterations 6–12

All comparisons below use saved official CRPS at 1000 paths and the same development prompt grid. A fully absolute-return seasonal EWMA scores 4707.9171 / 1794.2152 / 985.5152 (crypto24 / other24 / crypto1), so the learned model still improves a stronger robust reference. Simple width corrections do not explain its gain.

Robust five-minute seasonal profiles score **4693.0219 / 1766.4004 / 980.9638**: changes of -0.017%, -0.349%, -0.081% versus the learned checkpoint, improving 7/10, 13/16 and 10/10 asset-window cells. Weaker weekday pooling hurts. The robust five-minute profile is proceeding to July/November confirmation.

A mean-scale-preserving lognormal mixture with log standard deviation .4 and stratified normal coordinates scores **4688.7811 / 1767.6311 / 980.0071**. Student-t(5) scores **4688.5521 / 1770.7012 / 980.5720**. The changes are small enough to require confirmation and independent seeds. Conditional volatility quantiles plus within-path volatility score **4684.5253 / 1770.2478 / 979.9498**; preserving the point model's average instead gives **4689.3396 / 1768.2599 / 979.9828**. Both proceed to confirmation, without claiming that a fitted uncertainty model has yet beaten a simpler distribution.

Direct Sobol Gaussian sampling scores **4692.9474 / 1771.7886 / 980.5154**. The one-hour gain is -0.127%, consistent across all ten development cells; Brownian bridging adds no benefit. This is approximately the size expected from finite-fan sampling noise, not evidence of a better forecast.

Forward calendar effects on ordinary windows are mixed and tiny: crypto24 changes -0.133% in July and +0.076% in November; other24 -0.124% and +0.087%; crypto1 +0.004% and +0.070%. Development windows largely contain no relevant releases. These general-grid findings qualify the stronger event-centered gains (crypto24 -0.264%, other24 -0.695%, crypto1 -2.499%). Full or partial removal of historical event variance does not help; earnings cleanup raises November other24 CRPS by 0.261%. Discard historical cleanup. Retain forward effects as provisional pending denser evaluation at actual prompt times.

Three days of context, retaining seven-day-trained weights, scores **4683.1532 / 1776.0560 / 981.8023**. Crypto24 improves 0.227% while other24 worsens 0.196%. This is not a universal replacement for seven days, and matched training remains necessary before selecting a context length.

## Iteration 13 — learned volatility through the forecast horizon

**Hypothesis.** The useful lifetime of a recent volatility observation varies with forecast distance. A single learned average scale discards this information. Forecasting separate future segments can improve the calibration of later increments as volatility reverts, beyond a hand-designed current/slow EWMA tilt.

**Change.** Fit low-depth, purged chronological regressions for future absolute-return scale in contiguous segments (one-hour: 0–15, 15–30, 30–60 minutes; daily: 0–1, 1–3, 3–6, 6–12, 12–24 hours), using exactly the checkpoint's causal volatility features and base scale. Use the existing point model as a reference and blend/shrink segment predictions. Maintain Gaussian innovations while comparing this forecast change.

**Result.** Pending.

**Verdict.** Require at least 0.2% improvement in the target configuration across development and confirmation, with most asset-window cells improving. If separate segments only trade errors against each other, retain the scalar model.

**Lesson.** Pending.

## Iteration 14 — supplied NYSE closures

**Hypothesis.** Weekday/time-of-day seasonality overstates variance when the underlying equity market is closed for a supplied-calendar holiday, even though perpetual trading continues.

**Change.** Diagnose realized absolute-return scale against the ordinary seasonal profile on historical closures before fitting anything. Use the supplied NYSE_CLOSED and NYSE_EARLY_CLOSE rows only, respect asset scope, and test chronological/leave-one-date-out stability. Holiday effects must be tested separately from general-day CRPS.

**Result.** Pending.

**Verdict.** Require stable attenuation over multiple distinct dates and no general-day regression. Early-close history has only two dates, so strong bespoke estimates would be unjustified.

**Lesson.** Pending.

### Iterations 6–8: first independent confirmation

On July/November confirmation, robust five-minute profiles score **3681.646732 / 2860.615098 / 769.065924**. Both windows improve in all three configurations. Crypto1 improves all 10 asset-window cells; other24 improves 13/16, but November XAU worsens 1.89% and AAPLX 0.47%. This is a reason to investigate profile/scale interaction, not to hide those assets. A matched-profile retraining will test that interaction.

The .4 log-scale mixture with stratified sampling scores **3675.716722 / 2861.454293 / 769.224387**, improving every one of the 36 confirmation cells. Relative changes are -0.319%, -0.223%, -0.118%. The raw conditional-quantile/SV model scores **3680.810446 / 2860.449509 / 769.609141**. It does not consistently beat the simpler mixture and remains provisional pending the mean-preserving version.

### Iteration 13 label check

Recombining future segments with duration weights reproduces the existing whole-horizon absolute-return target: maximum relative error is 8.3e-15 for BTC 1h and 6.7e-16 for BTC/AAPLX 24h. This checks label alignment and segment definitions without evaluating a new scoring formula.

### Iteration 14 diagnosis before CRPS

On ten distinct closure dates, the equity cash-session absolute-return scale is about **0.234** of the neighboring ordinary weekday scale, after each is normalized by its own causal learned forecast. Leaving out one date changes this mean only to **0.216–0.245**. Premarket attenuation is also stable (mean 0.492), while post-session ratios are much nearer one. Controls that themselves fall on supplied-calendar holidays are excluded. Only control observations completed before an artifact's cutoff enter its fit. Two early-close dates are too few to fit a standalone effect and receive no bespoke coefficient. Forward-only cash-session, full-day and historical-normalization variants are now being scored on 232 fully covered historical prompts, all at 1000 paths.

## Iteration 15 — match training to the robust seasonal profile

**Hypothesis.** Changing the seasonal profile while retaining the old model's feature normalization leaves a small training/serving mismatch. Refitting the same low-depth volatility regression with robust five-minute profiles should distinguish a true shape gain from incidental scale correction.

**Change.** Rebuild features and forward targets using the timestamped robust five-minute seasonal artifact, preserve the original chronological cutoffs/purging and model complexity, and fit only volatility features. Compare the matched model to the profile-only substitution and the original checkpoint.

**Result.** Pending.

**Verdict.** Keep only if it improves on the profile substitution across development and confirmation; do not treat retraining itself as evidence of improvement.

**Lesson.** Pending.

### Iteration 14 CRPS result

On the 232 calendar-centered closure prompts, forward cash-session attenuation reduces other24 CRPS from **1488.072747 to 1424.601870 (-4.265%)**. Extending attenuation to supported premarket/overnight bins gives **1401.496293 (-5.818%)**, with all six assets improving. Normalizing historical holiday returns before EWMA gives **1405.762086 (-5.531%)**, so again historical cleanup is unnecessary. Keep forward attenuation as a candidate; carry it into the broad ordinary-prompt evaluation. These are conditional holiday results, not a claim of a 5.8% all-day gain.

### Iteration 11 final length sensitivity

Fourteen days scores **4697.371082 / 1772.842862 / 981.748864**; thirty days **4697.940810 / 1773.264031 / 981.741436**. Both longer contexts slightly worsen 24h scores and barely affect one-hour scores. Keep literal CONTEXT_DAYS=7. This conclusion concerns these seven-day-trained weights, not every possible model trained on longer history.

### Distribution confirmation decision

Student-t(5) confirmation scores **3682.824967 / 2862.077985 / 770.272570**. Its one-hour result reverses the development gain (+0.018% versus Gaussian), so do not select Student innovations. Mean-preserving conditional quantiles plus within-path volatility score **3679.987888 / 2862.219149 / 769.409900** and still do not consistently beat the fixed .4 mixture. The Sobol bridge/fixed-mixture scores **3675.751870 / 2861.097449 / 769.067054**. It proceeds to a second benchmark seed, alongside the original Gaussian and stratified fixed mixture.

### Iteration 15 parity check

Robust-profile training and serving features agree exactly at float32 on BTC 1h, BTC 24h and AAPLX 24h. Base-scale ratios differ from one by less than 3e-14. Twelve chronological fits are complete; all four general windows are now being evaluated.

## Iteration 16 — context-derived innovation distributions

**Hypothesis.** A Gaussian or fixed Student law cannot represent literal flat returns or the current asset's innovation tail shape. Symmetrized empirical innovations from the supplied context may improve those marginals without fitting a price drift.

**Change.** Build valid 1-minute or 5-minute returns without crossing missing observations, remove the causal seasonal profile, optionally remove a preceding local EWMA volatility, and sample a symmetrized recent empirical distribution. Normalize its mean absolute scale to the checkpoint's Gaussian-equivalent scale. Compare raw residuals, locally standardized residuals, and standardized residuals plus the previously tested .4 path-scale mixture. Gaps remain absent observations; CDF sampling does not fill price gaps.

**Result.** Pending.

**Verdict.** Require at least 0.2% CRPS improvement over the Gaussian checkpoint and improvement beyond the fixed mixture across both development and confirmation. Flat candles are real prices, but their existence alone does not prove this model helps.

**Lesson.** Pending.

## Iteration 17 — cautiously test a conditional price center

**Hypothesis.** Recent signed returns, clock state and venue activity may contain a small directional signal that volatility-only modeling deliberately omits. Such signals can easily overfit overlapping returns; a zero-center forecast is the reference.

**Change.** Build completed forward cumulative-return labels, scaled by the context volatility and known seasonal variance clock. Fit shallow, strongly regularized Huber regressions on the existing causal features, with chronological validation and a full-horizon purge. Start each fit from zero. Bound the served mean to a small fraction of forecast dispersion and compare full/half-strength means; the innovation distribution and volatility estimator stay fixed.

**Result.** Pending.

**Verdict.** Keep only with at least 0.2% official CRPS improvement across both development and confirmation and most asset-window cells. A better training loss is insufficient; discard an unstable mean even if a single period improves.

**Lesson.** Pending.

### Sampling edge case

The first odd-count/seed audit exposed a degenerate one-path case: rank-stratifying a single Sobol coordinate always gives the median, making every shock zero regardless of seed. Counts and shape were valid, but this made the one-path process unrepresentative. Preserve the scrambled raw Sobol coordinate when N=1; rank stratification remains unchanged for N>1. This fix does not alter any 1000-path CRPS result. Re-run the audit at 1, 3, 999, 1000 and 1001 paths.

## Iteration 18 — minute-of-hour variation for the one-hour competition

**Hypothesis.** Five- or fifteen-minute seasonal bins cannot represent the volatility burst at the first minute of an hour or quarter-hour. This matters most for crypto1, whose score uses minute returns.

**Change.** Measure a causal, rolling minute-of-hour profile from past absolute returns, then shrink toward a flat within-bucket pattern. Preserve mean absolute scale within each five-minute group, so the experiment reallocates minute-level dispersion without a global width change. Compare a pooled modulo-five pattern against the sixty-minute pattern. Daily forecasts remain unchanged.

**Result.** Pending.

**Verdict.** Require a stable sub-bucket pattern and at least 0.2% crypto1 CRPS improvement in both development and confirmation, including actual prompt times. Reject a clock detail whose apparent effect is only the small fixed-seed fan error.

**Lesson.** Pending.

### Iteration 18 measured size before scoring

The latest fitted minute-00 amplitudes are only about 1.04–1.10 times the containing five-minute mean; the later minutes are typically 0.93–1.00. These are far smaller deviations than the weekday/session effects. The rough log-width-squared CRPS bound predicts a gain of only hundredths of a percent once the minute-return component's weight is considered. Score the two predeclared patterns, but do not pursue an extensive sub-minute-clock sweep.

## Iteration 19 — scale calibration after log-target learning

**Hypothesis.** The learned point model predicts a log-average future scale. Under uncertainty, its geometric mean need not equal the width appropriate to either a Gaussian fan or a mean-preserving scale mixture. A residual log-scale standard deviation near .3 corresponds to roughly a 2% Gaussian-width or 5% mean-mixture-width correction, not a reason to fit another complex model.

**Change.** Hold features and all artifacts fixed. Compare Gaussian widths .95 and 1.05, then Sobol/fixed-mixture widths .95, 1.05 and 1.10. All changes multiply the learned forecast, unlike the earlier simple-EWMA calibration reference.

**Result.** Pending.

**Verdict.** Require consistent signs across development and confirmation and at least 0.2% gain in a target configuration before retaining a calibration. Use additional seeds for a gain near the finite-fan error.

**Lesson.** Pending.

### Iteration 15 verdict

Matched retraining does not improve the robust-profile substitution. Across all four windows it is +0.151% for crypto24, +0.010% for crypto1, and only -0.019% for other24, with the other24 sign reversing in October. In August crypto24 it is +0.276% versus the profile substitution. Discard the new weights. Retain the original causal feature definitions and original learned correction; the forecast seasonal shape is a separate calibrated component.

### Iteration 10 serving audit after fix

The Sobol mixture now passes requested counts 1, 3, 999, 1000 and 1001, with exact initial price, finite positive paths, repeatability under one seed and changed output under another. The audit evidence is saved in results/sampling_audit.json.

## Iteration 20 — combine validated components and test their interactions

**Hypothesis.** Robust seasonal shape, a modest scale mixture and known-calendar adjustments address different residuals and may compose. The learned future curve may add for other24 only. A within-path volatility process needs an explicit comparison against this simpler combined model.

**Change.** Build a fixed-family integration: original volatility-only features/weights, robust five-minute forecast seasonality, Sobol bridge sampling with a .4 log-scale mixture, optional forward events/closures, and optional mean-preserving learned curves for commodities/equities. Do not apply the failed context-length, historical-cleanup, matched-refit or Student alternatives. A final ablation adds within-path volatility normalized to preserve each path's average scale.

**Result.** Pending.

**Verdict.** Verify gains on all four windows, at actual field timestamps and under an independent seed before promotion. If components overlap or regress, retain the simpler subset. Every serving option will be fixed explicitly in modeling.py at landing, not selected by a research environment variable.

**Lesson.** Pending.

### Iterations 12–13 development decisions

The hand-designed half-strength volatility tilt scores **4692.340367 / 1772.023844 / 981.263690**, improvements of only 0.032% / 0.032% / 0.051%. Slower or stronger tilts do not materially improve this and can reverse the crypto24 sign. Discard the hand-designed tilt under its predeclared threshold.

The learned segment shape scores **4690.870909 / 1767.079846 / 981.523709**. The other24 gain is 0.282% in August and 0.346% in October, with all 16 cells improving; crypto gains are too small and unstable. Raw segment scales give **4688.321426 / 1766.224767 / 982.013194**, improving other24 by 0.359% but worsening crypto1. Proceed with other24 confirmation only. Half-strength variants reduce the other24 benefit, so do not add them to the confirmation search.

### Iteration 18 initial result

The pooled modulo-five clock scores crypto1 **875.758718** versus **875.948647** across four windows (-0.0217%), with all 20 asset-window cells improving. This confirms the predicted small size: the detailed clock effect is real but an order of magnitude below the predeclared 0.2% retention threshold. Await the sixty-minute variant before closing the study.

### Iteration 16 initial failures

Raw empirical residuals score **4693.456546 / 1772.608611 / 987.649180**, including a 0.60% crypto1 regression. Locally standardized empirical residuals score **4685.649488 / 1771.874802 / 982.682547**; their crypto1 sign flips between August (+0.252%) and October (-0.182%). They do not meet the cross-window threshold. The empirical-plus-mixture variant is still running, but literal zero returns alone have not justified replacing the simpler law.

## Iteration 21 — stability of the volatility regression

**Hypothesis.** The matched-profile refit exposed model-estimation variability: changing normalization and refitting can harm a volatile regime despite similar validation loss. More conservative leaves, a different history length, or an ensemble may reduce this variance. Input-based scale weighting may better allocate limited model capacity to high-CRPS states.

**Change.** Keep original features, labels, Gaussian paths, cutoffs and chronological purging. Change one factor per fit: training lookback 120 days, training lookback 365 days, minimum child weight 800 instead of 80, or bounded weights proportional to the context's base volatility. Also test an equal-log-weight ensemble of the original, recent, long and more regularized models. Training remains a surrogate; evaluation remains official CRPS.

**Result.** Pending.

**Verdict.** Require a gain beyond 0.2% in a target configuration on development and confirmation with most cells improving. If the ensemble only smooths the fitted model without improving forecasts, retain the original small regressor.

**Lesson.** Pending.

### Iteration 13 confirmation threshold

The learned shape improves other24 confirmation from **2867.838497 to 2862.755556 (-0.177%)**, and raw segment scales give **2862.475890 (-0.187%)**. Signs are positive on 15/16 and 16/16 cells, respectively, but the aggregate benefit falls just below the predeclared 0.2% confirmation threshold. Neither is promoted yet. The already-registered integration ablation will check interaction with the other components; a further claim requires stronger broad-prompt evidence.

### Iteration 18 verdict

The sixty-minute profile scores **875.755581** versus **875.948647** for crypto1 (-0.0220%), almost identical to the modulo-five profile. Both improve all 20 four-window cells, confirming a real but very small residual. Discard from the main model under the predeclared complexity threshold; retain the measurements as a useful failure of scale, not a failure of causality.

## Iteration 22 — matched training for three real days

**Hypothesis.** The live miner may receive only three real days despite requesting seven. A model calibrated on exactly that information can improve cold-start forecasts while leaving complete-context behavior unchanged.

**Change.** Rebuild the original volatility features/labels using a three-day truncation, retain the same fitting protocol, and compare against the existing seven-day-trained weights on three-day contexts. A production adapter will be considered only for genuinely short real histories; it will not shorten the complete context.

**Result.** Pending.

**Verdict.** Require improvement over the original model under the same three-day information, across both development and confirmation. Serving shape and gap handling remain mandatory irrespective of calibration.

**Lesson.** Pending.

### Independent-seed qualification

At benchmark seed 91117, the Sobol fixed mixture improves four-window means from **4190.669739 / 2320.612053 / 874.871729** to **4181.956139 / 2314.353017 / 874.392231**. Other24 improves all 32 asset-window cells and crypto24 19/20. Crypto1 improves 16/20, but August reverses sign (+0.164%). The Gaussian reference itself changes by 0.42% in that volatile August week between the two seeds; the Sobol mixture is much more stable. Do not hide this reversal or describe every one-hour result as seed invariant. A four-seed, one-hour comparison of Gaussian Sobol and mixture Sobol will isolate distributional shape from the Gaussian reference's accidental finite-fan center.

### Evaluation completeness audit

An audit of 3532 completed result files, containing **308766 official candidate scores**, found no invalid scores, rejection sentinels or row-count discrepancies. The evaluator now records generated/scored prompt counts and refuses invalid or omitted CRPS-only outputs. This is a validity check; it does not change the official scoring formula.

## Iteration 23 — explain the gold profile regression before changing it

**Hypothesis.** Gold's November regression under robust five-minute seasonality may come from the interaction of absolute-return seasonal shape with a squared-return volatility estimator, particularly across a quiet weekend.

**Change.** Inspect the largest paired-CRPS losses, compare predicted width against realized absolute-return scale, and attribute changes in historical EWMA to individual observed returns. Only if this identifies a scale/shape interaction will we test separating historical normalization from future seasonal allocation.

**Result.** The largest regression is the November 2 00:17 prompt: official CRPS rises from 1734.8150 to 1958.1932. Regressions cluster around Sunday evening/Monday, rather than being uniform over the DST week. Further variance attribution is running.

**Verdict.** Do not special-case a ticker based only on its score. A correction must address the diagnosed mechanism and improve multiple windows.

**Lesson.** Pending.

### Iteration 23 diagnosis and follow-up

The mechanism is identified. On November 2 00:17, the original forecast's mean five-minute scale is **6.5032 bp**, the robust-profile substitution predicts **11.6553 bp**, and the next day's realized Gaussian-equivalent mean absolute scale is **5.6248 bp**. Historical base volatility rises by **1.8325x** under the substituted normalization. Several genuine Sunday moves of 4–12 bp are divided by variance-profile values around .010–.013, versus .037–.079 in the original squared-return profile. Absolute-return seasonality makes this quiet-period denominator too small for a squared-return state estimator; a few sparse jumps then inflate Monday's entire fan. This explains the regression without invoking a special gold rule.

Test a general separation: keep the original historical normalization and learned state, use the robust profile only to allocate future dispersion, and preserve either the forecast's average step scale or the learned model's original de-seasonalized scale units. A half-strength future-shape variant is a conservative control. These are Gaussian comparisons first, with all four windows and unchanged random coordinates.

## Iteration 24 — does the order of volatility observations add information?

**Hypothesis.** Six EWMAs summarize recent volatility levels but can discard the order of a buildup, isolated jump or decay. A small multiscale sequence model may extract information beyond those summaries. This is a direct test against the successful EWMA/boosted baseline, not an assumption that a neural model is better.

**Change.** Encode de-seasonalized absolute-return and RMS histories at three scales: 48 five-minute bins, 96 thirty-minute bins, and 42 four-hour bins. Include observed-data fractions; missing prices are never filled. Feed three small GRUs plus the original volatility features, clock and asset identity into a scalar log-scale regressor. Keep the same forward target, chronological cutoffs and purging. Limit one GPU job and its memory, export frozen CPU inference artifacts, and compare the neural point model and an equal-log blend with the boosted model. Verify exact feature alignment before fitting.

**Result.** Pending.

**Verdict.** Require at least 0.3% official CRPS improvement over the boosted baseline in a target configuration on both development and confirmation, plus practical serving latency. If ordered histories add no incremental information, retain the smaller tree model and publish the negative result.

**Lesson.** Pending.

### Iteration 9 full starter reference

The 91-day test at actual field prompt timestamps is complete for the seeded starter. Official CRPS is **4183.111568 / 2232.231176 / 909.562253**, over **10920 / 16410 / 66012** prompts respectively, all at 1000 paths. The learned checkpoint is finishing on the identical timestamps. A separate unchanged-Gaussian/calendar comparison will measure the ordinary-traffic effect of the event and closure kernels across this broader span.

### Iteration 17 verdict

Half-strength directional forecasts score **4695.909257 / 1772.290026 / 982.321558**; full-strength forecasts **4698.098760 / 1772.047383 / 983.027968** on development. Both worsen the crypto configurations and gain at most 0.031% on other24. Discard: no confirmation is needed for a direction model that fails its development criterion.

### Iteration 24 data preparation fix

The first sequence-cache preparation stopped before writing its first model input file because pandas returned a read-only NumPy view. Make the numeric feature array an explicit copy before replacing nonfinite entries. This is the same pandas ownership issue encountered in the earlier EWMA feature builder, not a change to labels or prices. Rebuild, then run sequence parity checks before fitting.

### Iteration 19 verdict

Widening the fixed-mixture fan by 10% improves crypto24 development from **4688.329960 to 4674.905195 (-0.286%)**, but this combines August -0.581% with October +0.298%; only 5/10 cells improve. The 5% widening has the same sign conflict and only -0.177% aggregate gain. Crypto1 gains are at most 0.074%, below the threshold. Other24 worsens under either widening. Gaussian width corrections also fail. Discard global rescaling: the apparent aggregate optimum is driven by one volatile week, not stable miscalibration.

### Iteration 23 separated-profile result

Keeping original historical normalization and reallocating only future dispersion gives **4187.897097 / 2312.570652 / 875.160469**, versus **4190.648627 / 2320.214031 / 875.948647** over four windows. Other24 improves all four windows, by 0.329% overall, in 27/32 cells. The alternative unit normalization is slightly weaker for other24 (-0.307%). The original gold failure is substantially reduced, though gold November still worsens 0.328%, so this is not a universal cellwise improvement. Crypto effects are smaller (-0.066% / -0.090%) and will not motivate a shape change there. The half-strength control is finishing.

## Iteration 25 — a candidate with separated estimation and allocation

**Hypothesis.** Original RMS historical normalization, robust future shape for commodities/equities, a mean-scale-preserving .4 lognormal path mixture and supported forward calendar effects can combine without the quiet-period denominator failure of the earlier integration.

**Change.** Compare two candidates with original volatility features and tree weights: (1) Sobol mixture plus separated future seasonal allocation for other24; (2) the same plus forward macro/earnings/closure factors. Crypto keeps original seasonal profiles. No directional mean, global width correction, minute-of-hour detail or learned curve is added. All scoring remains official, 1000 paths.

**Result.** Pending four-window and broad actual-prompt comparisons.

**Verdict.** Retain a shape change only if it improves other24 development and confirmation, with most asset-window cells improving and broad traffic agreeing. Calendar effects must retain their causal, event-centered benefits and have nonnegative ordinary-traffic benefit; report their conditional and all-prompt effects separately. Audit the assembled candidate before promotion.

**Lesson.** Pending.

### Iteration 20 within-path volatility verdict

Adding normalized within-path volatility to the curve integration changes the four-window means from **4178.270910 / 2304.734732 / 873.642388** to **4178.674484 / 2304.353860 / 873.817359**. Crypto configurations worsen, and other24 gains only 0.017%. Discard this extra process. The simpler path-wide scale mixture remains the distribution candidate.

### Iteration 23 half-strength control

Half-strength future allocation gives **4188.674000 / 2314.654069 / 875.396288**. It is weaker than full separation in every aggregate configuration. Use full separation only for other24 in the candidate attribution; do not select half strength because it softens one gold regression.

## Iteration 26 — price the worst residual before adding a shock rule

**Hypothesis.** A few large, reversing recent returns can dominate the RMS state and make its extrapolation too persistent. A robust scale cap might fix these rare failures.

**Change.** Inspect the largest broad-test losses and screen the proposed condition against completed March–June labels and the March model, before writing a serving correction. No new CRPS implementation or shock-specific fit is used.

**Result.** At HYPE 2026-09-17 13:21, three observed minute returns (-277, -414, +645 bp) account for **94.8%** of fast variance. The learned forecast is **69.87 bp** per minute versus **17.45 bp** realized average absolute-equivalent scale; absolute/RMS state ratio is .378. SOL 2026-08-22 05:19 is also too wide (41.78 versus 26.82 bp), but XRP 2026-08-27 08:01 is too narrow (7.98 versus 16.82 bp). In the earlier March–June sample, low absolute/RMS states are already broadly calibrated: scale-weighted realized/predicted ratios are **1.054 / 1.006 / .960 / 1.011** in the <.5, .5–.6, .6–.65 and .65–.7 buckets. The two largest HYPE excess scores together contribute only about **0.007%** of broad crypto1 mean CRPS.

**Verdict.** Discard the unconditional shock cap at the diagnostic stage. A large individual error does not establish a valuable general correction. A reversal-specific effect would require many more independent episodes and a material aggregate upper bound.

**Lesson.** The worst prompt is useful for understanding a model, but a common rule can damage well-calibrated states to fix a rare, visually striking exception. Evidence: results/shocks/past_residual_buckets.csv and results/paired_ml_wide/extremes.csv.

## Iteration 27 — align the scale label with when the score is paid

**Hypothesis.** The original training label averages future absolute returns after dividing each by its seasonal scale. This gives quiet minutes equal weight in estimating future latent volatility, although their absolute CRPS contribution is much smaller. A label that weights by known future seasonal scale may better predict the common fan width, especially for equities and commodities.

**Change.** Keep features, historical normalization, tree parameters, chronological cutoffs, purging and Gaussian simulation fixed. Replace mean(|return| / seasonal_scale) with sum(|return|) / sum(seasonal_scale), using one-minute returns for crypto1 and the exact five-minute phase for daily forecasts. This is an absolute-return training surrogate, not a replacement scoring formula. Fit the same small regressors; test the new scale and an equal-log blend with the original.

**Result.** Pending.

**Verdict.** Require at least 0.2% official CRPS improvement in a target configuration in both development and confirmation, with most cells improving. Check interaction with the retained mixture and future shape if successful. A better surrogate training loss alone is insufficient.

**Lesson.** Pending.

### Iteration 10 four-seed precision result

Using Sobol bridge coordinates for both laws isolates distributional shape from ordinary Monte Carlo variation. Across four seeds, crypto1 Gaussian Sobol CRPS averages **875.098973** (seed standard deviation .031401), versus **874.378343** (.041224) for the fixed mixture: **-0.08235%** on average. All sixteen seed-by-window comparisons improve; the smallest is July at seed 91117 (-0.0213%), the largest August at the default seed (-0.1592%). This establishes a small repeatable distribution gain. It does not erase the earlier August reversal against a different, ordinary Gaussian Monte Carlo draw. Evidence: results/sampling_precision.csv.

### Iteration 24 raw neural result

The raw sequence model scores **4685.505212 / 1773.046212 / 981.784737** on development, versus the boosted reference **4693.825150 / 1772.589566 / 981.761767**. Crypto24's -0.177% aggregate gain combines August -0.313% with October +0.091%; the other configurations do not improve. This fails the predeclared 0.3% development threshold. The already registered blend is finishing; do not interpret successful GPU training or lower log-loss as an official-score improvement.

### Iteration 9 broad uncertainty and recent periods

The complete 91-day learned checkpoint scores **4036.953735 / 1979.165319 / 863.446546**, versus starter **4183.111568 / 2232.231176 / 909.562253**. Improvements are **3.494% / 11.337% / 5.070%**, across 10920 / 16410 / 66012 prompts. All 18 asset cells and all 14 calendar-week blocks improve. Paired calendar-week bootstrap intervals for percentage change are **[-4.770, -2.539] / [-12.711, -10.066] / [-6.511, -4.060]**. These are exploratory uncertainty intervals, not selection-adjusted guarantees. After the October 30 cadence change, gains remain **3.369% / 7.650% / 6.078%**; after US DST ends, **4.350% / 7.821% / 5.676%**. Evidence: results/paired_ml_wide/.

### Iteration 24 verdict

The equal-log neural/tree blend gives **4688.854531 / 1772.202837 / 981.385435** on development. Gains of **0.106% / 0.022% / 0.038%** remain below the predeclared 0.3% threshold, and the daily crypto sign still reverses in October. Discard both neural variants. The multiscale sequence experiment is a useful negative result: ordered absolute/RMS histories do not add enough information to the multi-timescale boosted baseline here. All twelve causal CPU artifacts and training logs are retained.

### Iteration 21 verdict

Recent (120-day), long (365-day), stronger-leaf and scale-weighted training produce development CRPS, respectively: **4696.796055 / 1772.394964 / 982.493716**; **4699.505879 / 1772.560553 / 981.865569**; **4693.017658 / 1772.906862 / 981.832556**; **4690.993028 / 1773.061382 / 982.299100**. None reaches the 0.2% gain threshold, so retain the original training protocol. The distinction matters: this scale weighting weights training rows; iteration 27 changes how future minutes are combined within a label.

### Iteration 22 verdict

Matched three-day training gives **4188.129957 / 2320.568542 / 876.051539**, versus **4187.317118 / 2322.221120 / 876.049895** for original weights under the identical three-day information. The small other24 benefit (0.071%) improves only 18/32 cells; crypto24 worsens and crypto1 is effectively unchanged. Discard the extra cold-start weights. Continue to serve short real histories with the existing gap-tolerant estimator.

### Starter gap-safety audit

The preserved seeded starter still dropped missing prices before differencing, which could join observations across a hypothetical internal gap. Replace that operation with the same adjacent-minute validity rule used by the research models. An exhaustive snapshot audit finds **no internal missing closes and no timestamp gaps** for any asset; missing closes occur only before the first trade on each listing day. All reported starter contexts from July onward contain no missing prices. Therefore this safety correction changes none of the existing measured starter forecasts or scores. Evidence: results/price_grid_audit.json. Do not reuse the unsafe starter operation in synthetic-gap experiments.

## Iteration 28 — earlier regimes without another tuning sweep

**Hypothesis.** The assembled candidate's gains should survive earlier volatility regimes and younger venue listings, rather than relying entirely on late-summer and autumn conditions.

**Change.** Add seven-day scored windows ending April 13 and June 15, with the same 120-minute, minute-17 grid, current one-hour formula and 1000 paths. Exclude SPCX because it did not exist in the April window; test the other twelve assets and all applicable configurations. Compare the seeded starter, unchanged ML checkpoint, and separated-shape/mixture/calendar candidate using chronological artifacts only. These are CRPS-only tests; there are no comparable field scores for this era.

**Result.** Pending.

**Verdict.** Use this as a generalization check, not as another parameter sweep. Report any reversals and their asset/coverage context. Require the candidate to preserve the large starter improvement; a material regression against the simpler ML model would reopen the implicated component.

**Lesson.** Pending.

## Iteration 29 — calibrate a stale volatility state, not just its interface

**Hypothesis.** The missing-data-safe EWMA renormalizes its surviving weights. When all recent observations are absent, this freezes the last short-term volatility estimate rather than aging it. A volatile state observed hours ago may be less informative than the slower daily forecast.

**Change.** Stress the current candidate with trailing three-hour and twelve-hour NaN gaps, leaving the real scoring paths intact. Never fill the gap or invent a return across it. For crypto1, compare the unchanged stale forecast against log-scale interpolation toward the same model's daily latent scale, with predeclared decay half-lives of one and six hours. Calendar allocation and the innovation law stay fixed; no price-center correction is introduced. This branch is exactly inactive with a complete context.

**Result.** Pending official CRPS over all four general windows.

**Verdict.** Retain a gap-only correction only if it improves both development and confirmation by at least 0.5% under both gap durations, with most asset-window cells improving. Otherwise report the stress loss and keep the original estimator. The serving contract remains mandatory regardless of scoring gains.

**Lesson.** Pending.

### Iteration 27 verdict

The new weighted label scores **4694.042605 / 1772.206218 / 981.718660**; its equal-log blend scores **4693.637739 / 1772.212355 / 981.731963**. The best commodity/equity change is only -0.022%, and crypto changes are negligible. Discard both under the 0.2% threshold. Training-feature parity is exact in all eighteen cached tables; only labels differ. Typical target changes are small for crypto and larger but symmetric for equity/commodity windows (for XAU the ratio's 10th–90th percentiles are .823–1.083). Better alignment of the surrogate's weighting does not materially improve this common-scale model. Evidence: results/risk_label_parity.csv.

## Iteration 30 — remove accidental direction from a symmetric mixture

**Hypothesis.** Stratified Gaussian coordinates have exactly zero sample mean before the path-scale mixture is applied. Multiplying each path by a different volatility factor can reintroduce a small random center. Pairing opposite log-return paths at the same scale should remove this finite-fan error while preserving the intended symmetric predictive law.

**Change.** Generate half as many scrambled Sobol points, pair each price coordinate with its negative, and stratify the complete requested ensemble's normal marginals. Opposite paths share a stratified scale multiplier. Honor odd counts with a final unpaired path, and use raw scrambled coordinates for a single path. Hold the candidate's volatility state, seasonal allocation and calendars fixed. Compare at the default seed on all four windows, then replicate at seed 91117 if the first result is useful.

**Result.** Pending.

**Verdict.** This is a quadrature change, not another forecasting feature. Require at least 0.02% improvement in a target configuration over both development and confirmation, no material aggregate regression elsewhere, and replication with the independent seed before retention. Publish the measured size rather than attributing a sampling gain to predictive information.

**Lesson.** Pending.

## Iteration 31 — seasonal shape as listings mature

**Hypothesis.** The original 120-day seasonality window may be too slow when venue liquidity changes, or too noisy for stable instruments. Keeping historical state normalization fixed lets us test this without repeating the gold denominator failure.

**Change.** Fit robust five-minute future profiles with 60 and 240 days instead of 120; preserve all pooling, clipping and clock choices. Change only the candidate's commodity/equity future allocation. Its historical profile, regression, mean step scale, distribution and calendar factors stay fixed. October-to-November changes in the existing 120-day profile have a weighted log-scale RMS of .070–.111 across these assets, suggesting only about 0.12–0.31% of width-error cost, so a large parameter sweep is not warranted.

**Result.** Pending four-window official CRPS.

**Verdict.** Because this replaces an existing window with no extra serving component, require a modest but repeatable 0.1% gain in both development and confirmation and most cells improving; otherwise retain 120 days. A successful replacement still needs broader traffic agreement. This diagnostic cost estimate is not a CRPS evaluation; all measured scores remain official.

**Lesson.** Pending.

### Iteration 28 result

On the earlier April/June windows, the unchanged ML checkpoint improves the starter from **4957.191254 / 3066.525535 / 1062.128246** to **4745.791114 / 2852.769395 / 1003.711716**, improving every one of 34 asset-window cells. The combined candidate further improves to **4727.530003 / 2835.011416 / 1001.304924**, gains of **0.385% / 0.622% / 0.240%** over ML, with 9/10, 14/14 and 10/10 cells improving. The large starter improvement and smaller candidate increment transfer to these younger-listing regimes. SPCX is excluded from both windows because it did not yet exist in April.

### Iteration 25 independent-seed result

At seed 91117 the combined candidate scores **4181.478738 / 2307.221826 / 874.500903**, versus **4190.669739 / 2320.612053 / 874.871729** for the ordinary Gaussian checkpoint. Changes are **-0.219% / -0.577% / -0.042%**, with 19/20, 29/32 and 16/20 cells improving. The previously documented August crypto1 reversal against this particular ordinary Gaussian draw remains (+0.164%); do not conceal it. The four-seed Sobol-versus-Sobol mixture study remains the cleaner distribution attribution.

### Iteration 30 verdict

Antithetic scale sharing removes the numerical log-price center (at a representative prompt, maximum center changes from .0561 to effectively zero basis points for 1h, and .8643 to zero for 24h), and passes counts 1, 3, 999, 1000 and 1001. But four-window CRPS changes are only **-0.0029% / -0.0034% / +0.0128%**. This fails the 0.02% threshold and worsens crypto1. Discard: exact symmetry alone does not compensate for losing half the independent volatility coordinates. No extra seed search is justified. Evidence: results/antithetic_audit.json and results/antithetic_general/.

## Iteration 32 — test nonlinear capacity before accepting the tree plateau

**Hypothesis.** The successful depth-three trees may miss higher-order interactions among clock, asset identity and volatility state, even though adding raw input features did not help.

**Change.** Increase only tree depth from three to five, retaining the original features, labels, training span, minimum child weight, learning rate, regularization and purged chronological early stopping. Compare the resulting model and an equal-log blend with the original. Inspection shows only one of twelve original fits hits the 600-round cap, and the final November 9 fits stop at 240/304 rounds; simply increasing that cap would not change the final model, so that alternative is not pursued.

**Result.** Pending.

**Verdict.** Require at least 0.2% official CRPS improvement in a target configuration across development and confirmation, with most cells improving and practical inference. Otherwise retain depth three and conclude that capacity does not explain the remaining residual.

**Lesson.** Pending.

### Iteration 29 verdict under the original rule

Three-hour gaps raise crypto1 CRPS from **874.432272 to 928.563190 (+6.190%)**. The terminal absolute-price component accounts for **38.9164 of the 54.1309** extra points (71.9%); the simulator must still start at the last observed, now stale, price. Twelve-hour gaps raise the score to **1019.903166 (+16.636%)**. A six-hour volatility-aging half-life improves the stressed scores to **927.098320 (-0.158%)** and **1014.868285 (-0.494%)**, with 19/20 cells improving in each test. Faster one-hour aging worsens the volatile August three-hour-gap test. Neither correction meets the predeclared 0.5% improvement under both gap lengths, so no aging branch is added to the current candidate. Report the useful but smaller effect and the much larger information-loss penalty. These are synthetic gap tests; the snapshot itself has no internal price holes.

### Iteration 31 verdict

Using 60 days for future profiles gives other24 **2307.952191**, versus **2307.291646** for 120 days; 240 days gives **2307.213961**. The apparent long-window benefit is only **0.0034%**, far below the 0.1% threshold. Retain 120 days. The observed month-to-month profile changes are too small and noisy to justify replacing the existing window on these results.

## Iteration 33 — a disclosed precision-based revisit of one-hour width

**Hypothesis.** Iteration 19's 5% widening improved crypto1 in both development windows by 0.074% overall, but failed its 0.2% threshold. The subsequent four-seed Sobol study measured far smaller numerical uncertainty (about .004% standard deviation for the paired distribution gain). A fixed, cost-free calibration deserves a separately disclosed precision check; this is not a claim that it passed the original rule.

**Change.** Reopen only the already specified 1.05 one-hour width, now over the combined candidate. Do not sweep new constants. First score July/November and the earlier April/June windows, which have not been used to choose this width. If signs hold, replicate on an independent seed and the broad actual prompt timestamps. Daily forecasts are unchanged.

**Result.** Pending.

**Verdict.** This is a post-selection follow-up with a revised criterion, which must remain visible in the report. Require improvement in each of the four newly checked windows, at least 0.02% overall, an independent-seed improvement and a negative broad week-block uncertainty interval. Otherwise retain width 1.0. No further width grid will follow a failure.

**Lesson.** Pending.

### Iteration 33 verdict

The reopened width check fails on the newly checked periods. Width 1.05 gives **885.759750** versus **885.299380** for the candidate: **+0.052%**. April worsens 0.109%, June 0.057%, November 0.052%; only July improves (-0.023%). Only 5/20 asset-window cells improve. Retain width 1.0. The smaller numerical error did not make the original calibration gain generalize, and no additional width grid or seed search follows this failure.

## Iteration 34 — screen extra event-amplitude uncertainty

**Hypothesis.** A scheduled event may need a wider distribution of possible amplitudes than the existing path-wide lognormal mixture provides.

**Change.** Examine causal one-minute event-amplitude residuals before introducing another random factor. Pool summaries by event type and retain distinct-date counts. Compare log-absolute spread descriptively with the Gaussian plus .4 log-scale reference; do not substitute this diagnostic for CRPS.

**Result.** Across 45 March–November dates, log-absolute residual standard deviation is **1.222**, versus **1.181** for the existing mixture. Most individual types are narrower: CPI .951, FOMC .807, NFP 1.096, PPI 1.075 and retail sales .958; GDP/PCE is wider at 1.423. Much of the pooled variation is a difference in type-level calibration, not universal extra uncertainty. In particular, residual mean amplitudes are 2.145 for CPI and 2.772 for NFP, versus .410 for retail sales.

**Verdict.** Do not add uniform event dispersion based on a heterogeneous pooled residual. The more promising question is the reference scale used for the event component. Evidence: results/events/causal_amplitude_spread.csv.

**Lesson.** A mixture across event types can look like uncertainty within each type. Separate those explanations before adding another latent variable.

## Iteration 35 — separate the event's scale from the quiet pre-release state

**Hypothesis.** A fast one-hour volatility estimate can fall during the quiet period before a release and jump after it. Multiplying an event kernel by that state can underpredict the release and double-count its aftermath. A slower reference for the event's extra variance may transfer better.

**Change.** For crypto1 only, retain its original background forecast. Use the already causally fitted daily-reference event kernels and daily latent scale for the event's extra variance: background short variance plus daily variance times the daily event factor's excess over one. Compare full and half replacement of the existing extra component. The slow/fast scale ratio is bounded to [.25, 4]. Non-event prompts and all daily forecasts retain identical behavior. In a diagnostic of historical forecast-scale ratios and completed event returns, type-demeaned residual slopes toward the slower reference are positive at all six cutoffs (unregularized slopes 1.83–3.98), but this is only a hypothesis generator.

**Result.** Pending 180-day event-centered official CRPS over the supplied calendar, with the same six issue leads used previously. Count independent dates, not five correlated coins, in interpreting stability.

**Verdict.** Require at least 0.5% conditional crypto1 improvement over the combined candidate, improvement across most event types/dates, no material general-window regression, and positive broad-traffic benefit before promotion. No directional event forecast is introduced.

**Lesson.** Pending.

## Iteration 36 — a simpler linear volatility reference

**Hypothesis.** A ridge regression can extrapolate smooth volatility relationships beyond tree split boundaries and may diversify the tree estimator, despite using fewer effective degrees of freedom. It is also a stronger simple reference than an unlearned EWMA.

**Change.** Fit linear log-scale corrections on exactly the original volatility, clock and asset features and labels. Standardize within each causal training fold, select ridge strength from [10, 100, 1000, 10000] on the purged past fourteen-day validation, and refit all completed past labels. Keep the 240-day span and Gaussian paths. Compare the linear model and an equal-log blend with the original tree.

**Result.** Pending.

**Verdict.** Require at least 0.2% official CRPS improvement in a target configuration on development and confirmation, with most cells improving. Otherwise retain the nonlinear correction and report the linear result as an additional baseline, not as evidence that more complexity is automatically preferable.

**Lesson.** Pending.

### Iteration 32 verdict

The depth-five model gives development CRPS **4688.028279 / 1771.924303 / 981.710088**; its blend gives **4690.632275 / 1772.105509 / 981.707396**. The largest improvement is 0.123%, below the predeclared 0.2% threshold. Discard both without spending confirmation on a failed development screen. Extra interaction capacity does not explain the remaining error.

### Iteration 35 verdict

On the same 180-day event-centered crypto1 prompts, the current candidate scores **1577.872581**, the full slower-reference replacement **1577.799671**, and the half replacement **1577.352531**. Gains of 0.0046% and 0.0330% fail the 0.5% conditional threshold. Discard both; the positive diagnostic residual slopes did not translate into useful CRPS. None of the 180 causal crypto1 peak event coefficients hits the amplitude cap of 15, so raising that cap is not a supported follow-up.

### Iteration 36 verdict

Raw ridge development scores are **4711.801833 / 1778.032651 / 983.044253**, and the equal-log blend gives **4700.617727 / 1774.326229 / 981.942659**. Every configuration worsens against the original small tree. Discard both. A linear extrapolator is a useful simple reference, but does not diversify this estimator enough to improve the objective.

### Iteration 25 complete broad result

The combined candidate scores **4024.014528 / 1960.753172 / 861.489277** over 91 scored days at actual field prompt timestamps. This improves the original ML checkpoint **0.3205% / 0.9303% / 0.2267%**, all eighteen asset/configuration cells and all fourteen calendar-week blocks. Against the seeded starter, gains are **3.8033% / 12.1617% / 5.2853%**. Evidence: results/paired_candidate_combined_wide/ and results/paired_candidate_starter_wide/.

Adding calendars to the separated-shape mixture improves broad CRPS **0.0517% / 0.3094% / 0.0475%**, with 17/18 asset cells improving. Exploratory week-block intervals for percentage change are **[-0.1035, -0.0061] / [-0.4904, -0.1465] / [-0.0969, -0.0074]**. These are not selection-adjusted. Calendars alone slightly worsen the post-cadence subset (**+0.0281% / +0.0733% / +0.0072%**) and post-US-DST subset (**+0.0541% / +0.0647% / +0.0095%**); the full candidate still improves the original ML checkpoint in each recent subset. The event-centered and closure-centered evidence supports keeping the forward adjustments, with these limits stated explicitly. Evidence: results/paired_candidate_calendar_wide/.

## Iteration 37 — promote and audit the combined serving checkpoint

**Hypothesis.** The validated combination can preserve its measured forecasts while meeting the production context, path-count, seed, latency and refreshed-calendar contracts.

**Change.** Replace modeling.py's Gaussian checkpoint with explicit fixed calls to the original learned state, separated future profile for commodities/equities, forward calendar factors and the .4 Sobol scale mixture. Do not use a research variant flag in serving. Preserve the original checkpoint in ml_models.py and its scored artifacts. Audit missing/empty calendars and replacement at the same path, plus all existing gap, count, seed, input-immutability and offline checks.

**Result.** Pending interface and exact forecast parity checks. No champion nomination yet; the campaign continues.

**Verdict.** Retain only after the serving checks pass. Any failure is a correctness bug to fix before further research.

**Lesson.** A score improvement is useful only when the same model can answer every supported request.

### Iteration 37 interface result

The combined serving checkpoint passes all 72 1000-path interface cases over all eighteen asset/configuration combinations; its maximum measured call takes **0.636499 seconds**. The separate calendar/count audit passes 22 cases, including absent/missing/empty calendar, refresh and removal at the same path, unknown type, five requested path counts, repeatability and research-flag independence. The event is selected from the supplied calendar (**2026-10-14 12:30 UTC**), not entered from an external source. Evidence: results/interface_combined_audit.json and results/calendar_audit.json. A full four-window official rerun of modeling.py is in progress to verify score parity with the candidate module.

The complete saved-score audit covers **5795 result files and 959808 official candidate scores**, with no invalid scores or summary-count discrepancies. Evidence: results/result_audit.json. Retain the combined checkpoint.

## Iteration 38 — event effects over a more recent history

**Hypothesis.** Type-specific event amplitudes can change relative to background volatility. The earlier residual diagnosis found CPI/NFP underprediction and much smaller retail-sales amplitudes, whereas changing the one-hour reference scale did little. A shorter coefficient history may respond to such changes without adding event-specific free parameters.

**Change.** Refit only the event kernels with a 180-day lookback instead of 365 days, using the same cached causal observations, date-level pooling, shrinkage, amplitude caps and cutoffs. Preserve closures, background scale, future profiles and the .4 mixture. Test full replacement on the same 180-day event-centered prompts. This is one prespecified alternative, not a window-length sweep.

**Result.** Pending official CRPS. Examine asset, release-type and distinct-date behavior before interpreting a conditional mean.

**Verdict.** Require at least 0.5% conditional CRPS improvement in a configuration, stable improvement across most distinct event dates, no material regression on ordinary windows, and a positive broad actual-prompt benefit before retention. Otherwise retain the 365-day fit. No dates or amplitudes come from outside the supplied calendar and snapshot.

**Lesson.** Pending.

## Iteration 39 — the scale-learning loss, not another feature set

**Hypothesis.** Squared error on log future absolute-return scale predicts a geometric conditional center. Gamma deviance with a log link targets the arithmetic mean of the positive scale ratio instead. State-dependent differences may matter even though the previous global-width follow-up failed. There is no presumption that arithmetic scale is closer to the CRPS optimum.

**Change.** Train the same depth-three tree, features, original scale labels, 240-day history, purged fourteen-day validation and causal cutoff schedule using gamma deviance. Compare the raw output and an equal-log blend with the existing log-MSE model under the original Gaussian simulator. Preserve all serving options until this isolated point-model test succeeds. No CRPS implementation is introduced; gamma deviance is a training surrogate and all scored comparisons use predict.py.

**Result.** Pending development CRPS. Check scale changes and early-stopping behavior to distinguish useful conditional calibration from indiscriminate widening.

**Verdict.** Require at least 0.2% CRPS improvement in a configuration on both development and confirmation, with most asset-window cells improving. Any successful model must then improve the combined checkpoint. Otherwise retain the original loss.

**Lesson.** Pending.

## Iteration 40 — sharing volatility dynamics across market types

**Hypothesis.** The daily tree shares its limited interaction capacity between continuously traded crypto and instruments whose underlying market has sessions. Asset indicators help, but a depth-three pooled tree may still compromise between different dynamics. Separate daily crypto and commodity/equity fits test this structural assumption; hourly crypto is already a separate model.

**Change.** Split only daily training by competition, preserving original features, log-MSE labels, hyperparameters, causal cutoffs and purged validation. Compare the specialized model and an equal-log blend with the shared model under identical Gaussian sampling. Hourly forecasts remain exactly the existing Gaussian reference.

**Result.** Pending development CRPS. Fewer observations can increase noise, so specialization is not assumed to improve the result.

**Verdict.** Require 0.2% improvement in a daily configuration across development and confirmation with most asset-window cells improving, followed by positive incremental results over the combined checkpoint. Otherwise retain the shared daily model.

**Lesson.** Pending.

### Iteration 37 score parity

The explicit serving implementation reproduces every saved official score in the four-window candidate comparison exactly: **4181.558780 / 2307.291646 / 874.432272**, over 6048 paired prompts and 72 asset-window cells. Evidence: results/combined_checkpoint_general/ and results/paired_serving_parity/. The promotion changes neither the measured forecasting law nor the scoring protocol.

### Iteration 38 verdict

The 180-day event fit gives **4554.522814 / 3133.152144 / 1582.785990** on the conditional event sample, versus **4552.456297 / 3132.435881 / 1577.872581** for the 365-day fit. Changes are **+0.0454% / +0.0229% / +0.3114%**; every crypto asset worsens and only two of eight commodity/equity cells improve. Discard without broader testing. The shorter history sacrifices too much stability to address the observed type-level residuals. Evidence: results/paired_event180/.

## Iteration 41 — reallocate uncertainty across time without widening marginals

**Hypothesis.** The retained lognormal factor is constant for an entire path. The earlier extra-volatility experiment added dispersion and did not establish whether the time dependence of the existing dispersion is correct. Splitting fixed total log-variance into persistent and transient components preserves each step's marginal mixture while changing multi-step returns and block volatility.

**Change.** Keep total log-scale variance at .4 squared. Assign half to the existing path-wide factor and half to a stationary Gaussian AR(1) log-scale state, with a prespecified thirty-minute half-life for one-hour forecasts and three hours for daily forecasts. Its unconditional mean scale stays one. Use the same Sobol price coordinates and a separate deterministically derived seed for the extra state. Do not change forecast variance, shape, calendars or drift. Test one split, not a grid of shares and decay rates.

**Result.** Pending four-window official CRPS, with interval attribution and a second seed if the first comparison is useful.

**Verdict.** This is a different hypothesis from adding variance. Require at least 0.05% improvement in a target configuration across development and confirmation, no aggregate regression of 0.05% elsewhere, most cells improving, and independent-seed plus broad-traffic agreement before promotion. Numerical sampling and temporal dependence must not be conflated.

**Lesson.** Pending.

### Iteration 41 sampling correction before interpretation

The initial implementation paired two separately scrambled Sobol engines by path index. A direct audit found same-coordinate cross-correlations as large as **-0.915**, despite different derived seeds: separate scrambles are not a joint low-discrepancy design. Stop and exclude the partial official run before reading it as evidence about temporal uncertainty. Preserve its source and outputs as explicitly invalid research artifacts.

The corrected experiment uses one joint Sobol engine with separate coordinates for price and state, plus a static-mixture reference using the identical higher-dimensional engine. Changing dimensionality also changes the price scramble, so this paired reference is necessary. The predictive hypothesis and thresholds stay fixed. All generator randomness still comes from the benchmark seed. This corrects the initial plan to retain the old price coordinates; no serving code was affected.

### Iteration 39 verdict

Gamma-deviance raw development scores are **4687.809372 / 1774.227218 / 981.917314**; the blend gives **4690.515467 / 1773.233415 / 981.720069**. The largest gain is 0.128% for daily crypto, but that gain reverses in October (+0.074%). Other24 worsens under both versions. Discard under the 0.2% threshold. The arithmetic scale target does not provide reliable conditional calibration beyond the log-loss model. The gamma training log's legacy text says “past log RMSE”; its recorded values are gamma deviance, correctly identified in the artifact manifest. The logging label has been corrected for later runs.

### Iteration 40 verdict

Separate daily fits give development **4692.611263 / 1772.416789** (crypto24 / other24); their blends give **4691.107170 / 1772.189692**. The largest gain is 0.058%, below the 0.2% requirement. Discard. Sharing daily volatility dynamics does not appear to be the limiting assumption, and specializing by market type adds models without sufficient benefit.

## Iteration 42 — combined-model quality under a three-day cold start

**Hypothesis.** The combined checkpoint's improvement should persist when a miner receives only three real days in its requested seven-day context. Earlier matched-training tests established the small-tree baseline under this constraint, but the final combination also changes future allocation and the path law.

**Change.** Mask all observations earlier than start minus three days with NaNs, preserving the seven-day index and all gaps. Score the combined serving implementation on the four general windows. Compare with the previously measured original-weights three-day Gaussian reference, after verifying that leading NaNs and a physically truncated frame give identical core forecasts. No new gap-specific correction or imputed price is introduced.

**Result.** Pending official CRPS. This is an operational generalization check, not another model-selection sweep.

**Verdict.** Require the combined model to retain its gains against the same-information Gaussian reference without a material configuration regression. Keep context length seven unless contrary evidence arises; the serving contract itself has already passed.

**Lesson.** Pending.

### Iteration 42 information parity

Leading NaN days and the native three-day context yield exactly equal background variance arrays, learned corrections and starting prices in all eighteen asset/configuration checks. Thus the existing three-day Gaussian scores are a valid same-information reference. Evidence: results/cold_context_parity.json.

## Iteration 43 — isolate a flat-return atom from empirical tail estimation

**Hypothesis.** The earlier empirical-innovation experiment changed both literal flat-return mass and the entire tail distribution; its failure does not isolate the value of the atom at zero. The snapshot documents substantial flat-return fractions, especially for thin commodity/equity periods. A simple hurdle distribution can test that feature while retaining the existing continuous law.

**Change.** Fit causal 120-day New York clock profiles of exact zero returns at the served one- or five-minute step, with the existing weekday/weekend pooling and smoothing. Do not infer zero returns from missing prices. Use a conservative half-strength log-odds correction from the current context to accommodate listing-density drift. Compare full and half predicted zero mass, capped at .65. The continuous normal component is rescaled by 1/(1-p) to preserve the same mean absolute step scale, and the existing path-wide .4 mixture remains. Background scale, calendars and seasonality are unchanged.

**Result.** Pending official CRPS. Inspect the fitted/served zero probabilities and per-interval changes, since preserving mean absolute return increases variance and can trade short-interval accuracy against longer horizons.

**Verdict.** Require at least 0.1% improvement in a target configuration on development and confirmation, most cells improving, no material 0.05% regression elsewhere, and independent-seed plus broad-traffic agreement before retention. This is a two-strength feature test, not a new empirical-tail sweep.

**Lesson.** Pending.

## Iteration 44 — calibration diagnostics for the selected distribution

**Hypothesis.** Lower mean CRPS can coexist with recognizable conditional errors. Coverage and probability-integral-transform diagnostics can explain the remaining errors, especially the center mass from exact flat returns, without introducing another fitted correction.

**Change.** Regenerate the fixed combined checkpoint on the same four general-window prompt sets. For the shortest scored return, a longer return block, and the terminal absolute price, record empirical predictive CDF values and 50%, 90% and 98% central coverage. Use the supplied future prices only as retrospective observations, exactly as in a diagnostic plot; they are never passed into simulate. These diagnostics do not calculate CRPS or replace the official score comparisons.

**Result.** Pending. Aggregate by asset and configuration and retain prompt-level counts. Overlapping daily forecasts and shared asset shocks make naive binomial confidence intervals inappropriate.

**Verdict.** Descriptive only. Do not promote a model based on coverage instead of official CRPS, and do not tune a new multiplier to these reused research windows.

**Lesson.** Pending.

### Iteration 41 verdict

With matched joint Sobol coordinates, the static mixture scores **4181.414551 / 2307.240813 / 874.439976**, and the temporal split gives **4181.271648 / 2306.983467 / 874.235233**. Gains of **0.0034% / 0.0112% / 0.0234%** fall below the prespecified 0.05% threshold. Discard without another seed search. The marginal-preserving process is a cleaner test than adding variance, but does not justify the extra state. The corrected sampling audit keeps mean scale within about 0.7% of one; the invalid separate-engine construction had coordinate correlations approaching absolute one. Evidence: results/temporal_sampling_audit.json.

### Iteration 42 verdict

The combined model with three real days gives **4177.840409 / 2309.326921 / 874.496208**, improving the same-information Gaussian reference **0.2263% / 0.5553% / 0.1774%**. Improved cells number **19/20, 29/32 and 19/20**. Retain the combination and the seven-day requested context. Its gains transfer to the expected live cold-start condition. Evidence: results/paired_combined_cold/.

## Iteration 45 — a learned prior when almost no returns are observed

**Hypothesis.** With no adjacent observed returns, the current EWMA uses the same fallback variance, 1e-7, for every asset. That is a finite serving fallback, but the scale-correction tree was trained with real histories. A causal asset-specific prior may calibrate sparse contexts better than this common constant.

**Change.** Fit the median of completed, normalized absolute-return scale labels over the preceding sixty days, separately by asset and horizon at monthly cutoffs. Under fewer than thirty valid adjacent returns, interpolate log scale from this prior toward the existing learned state with data weight n/30. At thirty or more returns the branch is exactly inactive. Stress with only the latest one price or ten minutes retained; the last price is fresh, unlike the earlier trailing-gap test. No missing price is filled and no return crosses a gap.

**Result.** Pending official development and confirmation CRPS.

**Verdict.** Require at least 1% gain in each sparse-context case on both development and confirmation, most cells improving, and exact identity with the existing model on ordinary contexts. If only the no-return case passes, retaining a fallback only when n=0 is a prespecified narrower option. A failed ten-minute shrinkage must not be silently retained with it.

**Lesson.** Pending.

### Iteration 43 verdict

Full zero mass gives development **4686.654358 / 1760.019259 / 979.729974**, versus combined-reference **4688.329960 / 1760.262716 / 979.570708**. Half zero mass gives **4687.393170 / 1759.958857 / 979.516106**. The best configuration gain is only 0.0357%, below 0.1%; full mass worsens hourly crypto. Discard both. The observed atom is real, but adding it with mean-absolute-scale preservation changes variance and the continuous distribution enough to offset its benefit. The smoke audit confirms zero frequencies and mean absolute scales near their intended values, so this is a modeling failure rather than an implementation mismatch.

### Iteration 44 result

Central 90% coverage of the shortest returns is **88.668% / 88.314% / 88.998%** (daily crypto / other24 / hourly crypto). Terminal absolute-price coverage is **81.488% / 86.496% / 90.476%**. Daily crypto is particularly weak in August: terminal coverage 68.095% and mean predictive-CDF rank .762, consistent with the already diagnosed directional volatility expansion. October and November daily-crypto terminal ranks instead average .423 and .465. The hourly terminal distribution is much closer to nominal. Short-return zero fractions are 1.824% / 6.063% / 5.615%. These diagnostics use 6048 forecasts; the observations within them are correlated and the windows have been reused. Evidence: results/calibration/.

## Iteration 46 — daily covariance while preserving short-return marginals

**Hypothesis.** The daily terminal deficit may reflect dependence across future returns that a conditionally independent Gaussian process cannot express. A small symmetric persistent component can widen long-horizon returns while preserving every individual step's marginal distribution. This predicts no direction. Unconditional variance-ratio diagnostics were often below one, so the hypothesis is uncertain and must be judged by CRPS.

**Change.** This follow-up is explicitly motivated by the reused-window diagnostics in iteration 44. Test one fixed covariance perturbation, without fitting a multiplier to nominal coverage: stretch the Brownian endpoint by sqrt(1.25), retain its bridge, then normalize each increment by sqrt(1+.25*v_j/sum(v)). Under the Gaussian process this preserves every step variance exactly and adds weak positive cross-step covariance. Keep the same Sobol coordinates and .4 scale factor, all volatility forecasts and calendars, and all hourly forecasts. Do not search alternative strengths.

**Result.** Pending four-window official daily CRPS and interval attribution. Marginal invariance is a distributional statement, not a claim that the finite 1000-path samples are numerically identical.

**Verdict.** Require at least 0.05% improvement in a daily configuration on both development and confirmation, most cells improving, improvement with an independent seed and on broad actual timestamps, and no material 0.05% regression elsewhere. A coverage improvement by itself does not qualify. Report this additional model-selection step transparently.

**Lesson.** Pending.

### Reproducibility checkpoint

Some early context-length wrappers imported the moving modeling.py entry point. Freeze their original Gaussian law in gaussian_checkpoint.py, and the later combined law in combined_checkpoint.py, so those research modules remain reproducible after serving updates. Redirect the context-length wrappers, combined cold-start wrapper and calibration diagnostic to their appropriate fixed references. These are the same explicit forecast formulas; no saved score is rewritten. The serving module remains independent of all research variant flags. The latest completed-output audit covers **1013216 official scores in 6382 files**, with no invalid scores or count discrepancies.

## Iteration 47 — calendar refresh when modification time is preserved

**Hypothesis.** Caching only the calendar path and modification time can retain a stale schedule when a refresh preserves that time. The earlier audit changed mtime explicitly and therefore did not cover this case.

**Change.** Reproduce an atomic replacement with preserved mtime using an empty file and a supplied historical CPI row. If confirmed, include inode, size and metadata-change time in the version key. Extend the production audit to atomic and in-place refreshes that preserve mtime. Never alter the supplied calendar itself.

**Result.** Pending reproduction and regression audit.

**Verdict.** This is a serving correctness requirement, independent of CRPS optimization. Correct refreshes must take effect; ordinary forecasts with an unchanged calendar must retain exact behavior.

**Lesson.** Pending.

### Iteration 45 verdict

One-price development scores change from **5033.706502 / 1822.050706 / 1106.044612** to **4803.055128 / 1789.054620 / 1040.890016**. Confirmation changes from **3873.686079 / 2984.469983 / 833.278915** to **3723.383950 / 2960.634734 / 788.961468**. The confirmation gains are **3.880% / 0.799% / 5.318%**, with **9/10, 7/16 and 10/10** cells improving. Commodity/equity confirmation fails both the 1% requirement and the majority-cell requirement. Ten-minute development gives **4735.852734 / 1775.085511 / 1012.955646**, versus **4769.125681 / 1827.546244 / 1025.386948**: daily crypto improves only 0.698% and reverses in one window. Discard the sparse blend and the prespecified no-return-only alternative under the stated rule; do not run ten-minute confirmation after its failed screen. Crypto's stronger no-return result remains useful evidence, but is not silently substituted for the all-configuration criterion. Ordinary serving retains its existing fallback. Evidence: results/paired_sparse1_dev/, results/paired_sparse10_dev/, results/paired_sparse1_confirm/.

**Lesson.** Almost-empty context priors can matter much more than ordinary-context refinements, but a common rule does not transfer reliably across all assets. Synthetic cold starts do not establish their frequency in live traffic.

### Iteration 46 verdict

The covariance perturbation gives **4180.078073 / 2306.618397**, against **4181.558780 / 2307.291646** for the daily combined reference. Gains of **0.0354% / 0.0292%** fall below 0.05%; only **14/20 and 18/32** cells improve. The terminal-price term accounts for 1.325 of 1.481 crypto points and .475 of .673 commodity/equity points gained, while five-minute scores worsen slightly. Both exploratory week-bootstrap intervals include zero. Discard without searching other strengths or seeds. Evidence: results/paired_covariance/.

**Lesson.** A conspicuous terminal-coverage deficit need not offer much aggregate CRPS improvement. Interval attribution explains the small gain and the remaining tradeoff.

### Iteration 47 verdict

The stale-cache failure was reproduced: an atomic calendar replacement preserved mtime and changed an empty schedule to one supplied CPI event, but the cached reader still returned zero rows. The cache now keys on path, mtime, ctime, inode and size. All **24** expanded calendar/count cases pass, including atomic and in-place replacements preserving mtime; maximum call time is **0.036703 seconds**. Retain the correctness fix. It changes refresh behavior, not the forecasting law for an unchanged calendar. Evidence: results/calendar_refresh_reproduction.json and results/calendar_audit.json.

**Lesson.** File modification time alone is not a reliable calendar version identifier under common replacement workflows.

## Iteration 48 — winter transfer with thinner early listings

**Hypothesis.** The selected combination's gains should extend to an earlier winter period, when the newer commodity/equity listings were thinner and the US session occupied its winter UTC hours. Earlier April/June checks and the March clock comparison do not directly establish the full combination on this period.

**Change.** Freeze the starter, original learned Gaussian, and combined serving model. Evaluate five scored days ending March 7, 2026, at the existing 120-minute cadence and 17-minute offset, using 1000 paths and the current hourly scoring formula. Daily prompts then begin March 1, after the existing first learned-model cutoff. Exclude SP500 and SPCX, which had not begun trading, from all three models. Use the eleven other assets; no new fit, strength choice or seed search. This is an additional historical validation window, not a final untouched holdout.

**Result.** Pending official CRPS, per-asset counts and interval attribution. Cached realized paths do not cover this era, so verify complete observed-price coverage and keep genuine flat candles.

**Verdict.** Expect the combined model to improve the starter in every configuration and the Gaussian reference without a configuration regression exceeding 0.1%. A failure triggers diagnosis and disclosure, not automatic tuning on this window. Existing broad traffic evidence remains the primary selection evidence.

**Lesson.** Pending.

### Iteration 48 verdict and landing

The five scored days March 2–6 give starter **6749.000506 / 3812.217022 / 1414.185462**, Gaussian **6643.980407 / 3725.910579 / 1389.446968**, and combined **6627.236414 / 3705.503579 / 1388.618184**. Combined gains over the starter are **1.804% / 2.799% / 1.808%**; gains over the Gaussian reference are **0.252% / 0.548% / 0.060%**, with **5/5, 6/6 and 4/5** cells improving against Gaussian. There are **300 / 360 / 300** prompts per model. The full February 22–March 7 coverage slice has 20160 minutes and no missing closes for each included asset; AAPLX, GOOGLX and oil still have 55.36%, 36.32% and 24.69% zero-trade minutes. Keep the combination without another adjustment. The sample occupies only one calendar week, so its bootstrap interval is unavailable, not a spurious zero-width confidence band. The paired reporting script now makes that limitation explicit. Evidence: results/paired_winter_starter/ and results/paired_winter_gaussian/.

**Lesson.** The full model transfers to this winter period, including thin venues, but one week cannot establish long-run uncertainty. The original chronological weights suffice; there was no earlier refit or use of later weights.

The harness issued its LANDING ORDER while this already-started check was finishing, at **85.45 USD** recorded use. No new experiment is started after it. All earlier pending studies have decisions recorded above. Final work is limited to release verification, artifacts, the public report and a committed nomination. Retain seven context days, the original learned volatility state, separate future seasonal allocation for commodities/equities, forward calendars and closures, and the .4 path-scale mixture with ranked Sobol bridge sampling. Reject the sparse prior, covariance modification and other failed refinements. The 91-day official starter/champion CRPS remains **4183.111568 → 4024.014528**, **2232.231176 → 1960.753172**, and **909.562253 → 861.489277**.

### Final release verification

All **12** retained chronological volatility boosters load on CPU. Required serving artifacts total **23,916,102 bytes** and every required source/artifact file is tracked; file hashes are saved in results/release_artifact_audit.json. The final code passes the **72** ordinary/gappy 1000-path interface cases, seed/count/immutability checks and blocked-network check, with maximum call **0.640252 seconds**. The expanded calendar audit remains **24/24** passing. The completed-output audit now covers **6626 result files and 1,032,560 official candidate scores across 186 suites**, with no invalid scores or count discrepancies. These are scores across experiments, not independent observations. Evidence: results/interface_release_audit.json, results/calendar_audit.json and results/result_audit.json. The public report and reproduction guide are complete; the nominated commit and harness interface result are recorded in CHAMPION and the final verification entry.

### Landing dependency correction

The first nomination pointed to b30963c6. The harness interface check failed with **ModuleNotFoundError: No module named 'xgboost'**, despite XGBoost being present in the pinned workspace environment. Local offline checks had not simulated an absent runtime package. This is a release dependency failure, not a reason to change the fitted model or start another experiment.

Export all 24 archived original forests to compressed NumPy arrays, including the twelve volatility-only forests used in serving. Preserve float32 thresholds, missing-value branches, leaf values, base score and sequential tree summation. The portable predictor matches XGBoost **bit for bit on 49,152 feature rows**, including missing values, across all 24 forests. Make XGBoost imports lazy in research utilities and remove the unused segment-model import from the serving import chain. No retraining, calibration or model-selection choice changes. Run the existing interface audit with XGBoost imports actively blocked, and rerun the four-window official comparison to verify the complete scoring behavior before amending the nomination.

### Portable release verification result

With XGBoost imports actively blocked, all **72** interface cases pass, including seeds, counts, gaps, immutable inputs and blocked networking; maximum call **0.362382 seconds**. All **24** calendar cases pass again. Every one of **6048** four-window official scores is bitwise identical to the prior implementation, with maximum score difference **0.0**: **4181.558780 / 2307.291646 / 874.432272**. Required portable serving artifacts total **19,315,018 bytes**; twelve volatility forests load using NumPy alone. The updated audit covers **6698 files, 1,038,608 official candidate scores and 187 suites**, without invalid scores or count discrepancies. Evidence: results/portable_tree_parity.json, results/portable_score_parity.json, results/paired_portable_release/, results/interface_portable_audit.json and results/release_portable_artifact_audit.json. Amend the nomination to this verified portable implementation; retain the same measured CRPS and forecasting choices.

### Final nomination and external-check limitation

The final nomination points to existing commit **5fbf8572b2481903fbb228220fa2b03115012c77** with profiles **low and high**. An isolated audit extracts only the source/artifact files from that commit, verifies their recorded hashes, removes the working agent directory from the import path, and blocks XGBoost and networking. All **18** asset/configuration cases pass at 1000 paths. Evidence: results/nominated_commit_audit.json. This confirms that uncommitted files or the working environment's XGBoost installation are not hiding a serving dependency.

The harness status continued to display its initial **ModuleNotFoundError: No module named 'xgboost'** after the amendment and re-presentation of CHAMPION. No refreshed external pass was observed, so none is claimed. The dependency identified by that error has been removed and independently checked in the nominated commit. Keep this operational limitation visible in the handoff; do not alter the read-only harness budget/status file. All code, artifacts, reports, evidence and the nomination are committed.
