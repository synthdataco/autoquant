# Research journal

## Iteration 0 — protocol and baseline (in progress)
Hypothesis: the seven-day constant Gaussian volatility starter is miscalibrated across sessions and recent volatility regimes. The first measurement establishes its CRPS without changing its distribution.
Change: preserve starter in baseline.py, adding only the required environment-controlled random seed. Repair evaluator's strict NaN loader and resolve snapshot paths using default_store_root. Use only the frozen snapshot, no interpolation or online market requests.
Protocol: assess all 13 active assets and both crypto horizons. Development windows end August 10 and August 24; reserve September 4–13 for confirmation. Compare identical prompts and realized coverage. Prefer changes improving both windows and configuration means; investigate per-asset regressions. Report arithmetic asset-average CRPS by configuration, plus per-asset values; do not optimize rank.
Result: pending. Dependencies installed. Read exact validator score: 24h sums increment CRPS at 5m/30m/3h plus terminal absolute CRPS; 1h also scores origin gaps and block volatility. Thus path dynamics matter, not only terminal distributions.
Verdict: pending.
Lesson: starter documentation claimed a lenient loader but implementation used strict load_range; explicitly corrected before evaluation.

## Iteration 1 — seasonal variance (pre-registered)
Hypothesis: a shrinkage estimate of hour-of-week variance improves increment CRPS, especially for equities and commodities. Fit seasonal shape on June–July only; both August windows and September remain out of fit.
Change: use observed returns at the simulation step, normalize gap duration, calibrate weekly variance after seasonal normalization, forecast a deterministic seasonal variance curve. Antithetic paths reduce simulation noise. This initial comparison bundles grid calibration and seasonality; follow-up constant-grid ablation separates them.
Kill threshold: reject for any configuration if CRPS worsens >1% on both development windows. No per-asset parameter tuning until this broad hypothesis is evaluated.
Measured seasonality: weekday/weekend variance ratios 1.58–2.33 for crypto, 4.28–7.67 for tokenized equities and gold (SP500 1.95, WTIOIL 1.60, SPCX 5.15). Hourly max/min ratios range 2.42–4.40 crypto and 8.00–24.69 other assets.
Result / Verdict: pending.

## Iteration 2 — adaptive variance and tails (pre-registered)
Hypothesis: recent volatility is more informative than a flat seven-day mean, particularly at 1h. Student innovations or a shared random volatility scale may also improve CRPS by avoiding excessive central dispersion while retaining tails.
Change: compare constant-grid, seasonal, adaptive (6h/24h half-life; 75% recent weight), faster adaptive, adaptive scale 0.85, Student t5, and lognormal path-scale mixture 0.4. Same 1000 paths and seed, all 13 assets, 6h prompt grid. Freeze seasonality fit at July 31. Development windows August 1–7 and August 17–23. These windows are chosen ahead of observing their scores, rather than asserting they are opposite regimes without evidence.
Kill threshold: no configuration-wide promotion on a regression in both windows; retain parsimonious alternatives within 0.5% pending denser evaluation. No ranks are produced.
Result / Verdict: pending.

## Iteration 3 — robust scale and learned volatility (pre-registered)
Hypothesis: squared returns overreact to isolated jumps. The success of 0.85 dispersion in the first window suggests estimating typical absolute returns may outperform simply shrinking standard deviation. A regularized HAR regression may learn how quickly recent volatility reverts.
Change: fit log future deseasonalized mean absolute returns from past 15m/1h/6h/24h/72h/168h means, using ridge regression. Fit ends July 31, labels stay before cutoff. Predict 1h for minute paths and 24h for 5m paths. Convert absolute-return scale to Gaussian sigma; separately test absolute-return EWMA, faster EWMA, shared volatility mixture, and Student innovations. Regression's training error is not presented as CRPS.
Kill threshold: learned HAR must beat the simpler absolute estimator on both development windows or show a defensible configuration-specific improvement; otherwise discard complexity.
Result / Verdict: pending.

### Iterations 0–2 results
At 1000 paths and 28 prompts/asset/window, configuration CRPS (COM_EQU_24H / CRYPTO_24H / CRYPTO_1H current-rule replay):
- Starter: Aug1 3307.6917 / 2831.1013 / 629.1511; Aug2 2204.8470 / 6688.7651 / 1215.7233.
- Constant five-minute calibration: Aug1 3294.5858 / 2825.8628 / 630.5776. Most equity benefit is not explained by resampling alone.
- Seasonal: Aug1 3070.0857 / 2794.0311 / 631.9575; Aug2 2021.6821 / 6641.9823 / 1209.4321.
- Adaptive: Aug1 3063.4947 / 2784.7846 / 626.6187; Aug2 2022.8387 / 6598.9684 / 1199.1483.
- Adaptive scale .85: Aug1 3018.3364 / 2751.8141 / 618.1878; Aug2 2017.2154 / 6620.7537 / 1187.7614.
- Adaptive mixture .4: Aug1 3027.3906 / 2763.2964 / 624.1603; Aug2 2012.2900 / 6578.8981 / 1193.8057.
Verdict: keep seasonality, retain robust-scale and mixture candidates; reject universally faster adaptation and universal .85 scale as settled conclusions. Narrowing helps quiet conditions but loses some crypto24h advantage in the busy window. Rank unused.
Lesson: improvement must survive volatility regime changes. Coverage identical across variants (99.762% Aug1, 99.357% Aug2); residual holes limit absolute score interpretation.

## Iteration 4 — session resolution (pre-registered)
Hypothesis: coarse hourly variance profiles blur the US cash open at 13:30 UTC. Robust absolute-return profiles and 30/15-minute bins may better model session transitions.
Change: freeze June–July profiles using squared returns or absolute returns; compare 24, 48, 96 daily bins with the same weekday pooling. Pair with the absolute EWMA estimator; no regression artifact trained under a different normalization is reused.
Kill threshold: retain finer profiles only when both windows improve or the gain is consistent across assets and exceeds 0.5%; otherwise prefer fewer parameters.
Result / Verdict: pending.

## Iteration 5 — shared volatility uncertainty and return dependence (pre-registered)
Hypothesis: HAR with a path-level volatility mixture can calibrate both calm and busy periods, and residual microstructure autocorrelation affects the multistep increment distributions. Fitted MA(1) dependence should help assets whose aggregated variance is lower than independent increments imply.
Change: compare HAR mixtures 0.2/0.6/0.8 against 0.4, HAR t5, and HAR mixture0.4 with lag-one return correlation measured on June–July. Preserve marginal one-step variance in the MA(1) transform. Disallow manual asset-specific correlation choices.
Kill threshold: mixture changes or MA(1) must improve both development windows per configuration to displace the current parsimonious mixture0.4.
Result / Verdict: pending.

### Iteration 3 results
HAR + mixture0.4: Aug1 configuration CRPS 2984.5192 / 2752.6606 / 618.4267; Aug2 2006.9817 / 6549.7428 / 1183.8016 (COM / crypto24 / crypto1).
Absolute fast EWMA: Aug1 2996.2956 / 2752.5880 / 617.2845; Aug2 2012.1435 / 6579.8606 / 1191.3150.
Verdict: keep HAR mixture as a broad candidate. Without mixture, HAR alone did not consistently beat simple robust calibration. Training log-RMSE is not a selection metric.
Lesson: jointly representing uncertainty in future scale matters more than a more elaborate point volatility estimate alone.

### Iteration 4 results
Absolute seasonality at 48 bins/day versus the original variance profile, holding absolute EWMA fixed: COM CRPS 3005.0261 → 2989.0661 (Aug1), 2014.6885 → 1995.6357 (Aug2). 96 bins does not help. Crypto24 gains are inconsistent; crypto1 differences are small. Verdict: retain 48-bin absolute seasonality for equities/commodities; do not universally replace crypto profiles yet. Next fit HAR under the selected normalization rather than reusing incompatible coefficients.

## Iteration 6 — conservative drift (pre-registered)
Hypothesis: a heavily shrunk linear function of past signed returns could improve origin-relative gaps and terminal prices; efficient-market behavior may make this a useful negative result.
Change: fit future standardized returns from past 15m/1h/6h/24h/72h signed returns, June–July only, ridge penalty 1000, no intercept, bounded drift at 0.3 predicted standard deviations. Compare 0.5 and 1.0 fitted drift against HAR mixture0.4.
Kill threshold: discard if either window shows a configuration-level regression >0.2%, unless a consistent independent confirmation supports a restricted use. No unbounded trend extrapolation.
Result / Verdict: pending.

## Iteration 7 — combine validated components (pre-registered)
Hypothesis: refitting HAR under robust 30-minute seasonal normalization will preserve the useful learned mean reversion while improving commodity/equity session calibration.
Change: refit all HAR coefficients with the 48-bin absolute-return seasonal profile (training still ends July 31). Compare Gaussian mixtures0.4, mixture0.4 plus unit-variance t5 innovations, and modest ±10% scale variants. A separate MA(1) branch is retained only for crypto1h, where it improved both initial windows by ~0.04–0.08% and needs seed confirmation.
Kill threshold: combined model must beat the original HAR mixture on both windows for its configuration. Small differences will require denser prompts and multiple seeds rather than choosing the best single draw.
Result / Verdict: pending.

### Iterations 5–6 results
Mixture0.4 beats mixture0.2 and mixture0.8 on both windows across configuration averages. Mixture0.6 does not consistently improve; retain0.4. Fitted MA(1) improves crypto1h by 0.08%/0.035% but regresses COM; retain only as a small effect requiring seed confirmation. Student t5 improves the quiet window but regresses the busy one versus mixture0.4.
Drift0.5/1.0 regresses every configuration in the busy window (crypto24h 6549.7428 → 6559.1766 / 6570.6438). Full drift also regresses COM in the quiet window (2984.5192 → 2990.9158). Verdict: discard drift. Training R² did not translate into CRPS gains.

## Iteration 8 — nonlinear volatility forecasting (pre-registered)
Hypothesis: HAR's log-linear rule may miss leverage effects and transitions between short-lived spikes and persistent regimes. A small pooled gradient-boosted model can exploit these without committing to a large foundation model.
Change: train a shallow histogram gradient boosting regression on past relative absolute/squared return scales, signed returns, calendar features and asset identifiers. Predict log future deseasonalized mean absolute returns relative to the seven-day level. Training ends July 31; evaluate the same two August windows before any September use. Cap CPU threads and model complexity.
Kill threshold: reject added complexity unless it improves both development windows for at least one configuration by >=0.5%, then confirm on an additional August window. Training loss remains a surrogate; official CRPS decides.
Result / Verdict: pending.

## Iteration 9 — empirical innovations (pre-registered)
Hypothesis: a fixed Gaussian family, even with random scale, may miss the sharp center and asymmetry of venue-specific return innovations. Sampling standardized observed returns can learn this shape without fitting tail parameters.
Change: use only seven-day context residuals, divided by the seasonal profile and a causal 6h absolute-return scale, then symmetrize and normalize their mean absolute size. Compare empirical innovations with and without a shared volatility mixture. Do not reproduce any other miner's paths. Preserve 1000 simulation paths and the official scorer.
Kill threshold: require improvements on both windows per configuration; discard if the apparent gain is driven by a few sparse assets or worsens busy-window calibration.
Result / Verdict: pending.

## Iteration 10 — residual volatility clustering (pre-registered)
Hypothesis: independent empirical innovations discard within-hour volatility clustering that affects 1h block-volatility CRPS. Short blocks of past standardized innovations may restore it without a fitted stochastic-volatility process.
Change: compare five-step residual block resampling with zero or 0.25 shared lognormal scale uncertainty, using the same HAR and original seasonal profile. Context-only residuals, no external served paths.
Kill threshold: require improvement on both development windows; check 1h historical-price and current-total scores separately so a volatility-only gain cannot hide a material price regression.
Result / Verdict: pending.

### Iteration 8 preliminary results and verdict
GBM mixture0.4 CRPS: Aug1 3000.4419 / 2753.8286 / 617.1383; Aug2 2004.4537 / 6674.9294 / 1190.3696 (COM / crypto24 / crypto1). It fails the preregistered two-window improvement requirement. XRP contributes much of the busy-window failure, consistent with tree models extrapolating poorly outside training regimes. Verdict: discard as a serving candidate; retain artifacts and evidence. A 50% HAR blend is measured as the prespecified conservative alternative.
Lesson: lower training log-volatility error is not enough. Official out-of-sample CRPS prevents selecting an appealing but fragile nonlinear model.

### Iterations 9–10 results
Empirical innovations and five-step blocks fail to beat the HAR Gaussian mixture across both windows. For example, empirical+mixture crypto1h is 619.0742 / 1190.4479 versus Gaussian mixture 618.4267 / 1183.8016. Five-step blocks reduce the empirical regression but do not reverse it. COM changes are small and inconsistent between windows. Verdict: discard empirical and block resampling as universal replacements. The Gaussian scale mixture is both simpler and better calibrated in these tests.

## Iteration 11 — independent August confirmation and seed stability (pre-registered)
Hypothesis: the gains from seasonal HAR survive a previously unused interval and denser prompts, while tiny dependence/shape gains may be sampling noise.
Change: freeze four finalists: original HAR mixture0.4, robust half-hour HAR mixture0.4, half-hour HAR with scale0.9, and half-hour HAR with fitted MA(1). Evaluate August 8–16 at 3h cadence. Separately repeat half-hour HAR and its MA(1) variant on original August windows with seeds 123 and 987. These runs use 1000 paths.
Kill threshold: discard scale/dependence modifications if gains reverse in the new interval or are not consistent across seeds. Select at most one broadly supported configuration-specific combination before opening September confirmation.
Result / Verdict: pending.

## Iteration 12 — scheduled macro-event volatility (exploratory, preregistered scoring)
Hypothesis: predictable FOMC announcement times carry volatility beyond ordinary hour-of-week seasonality. Unlike price direction, this is knowable before the event. Public Federal Reserve calendar dates are allowed non-market knowledge; no future decisions, statements or price outcomes are used.
Exploration: January–July crypto minute absolute returns in the announcement hour averaged a median 2.01x matched nearby-week controls across 25 asset-events. January was not elevated; later events usually were. July29 was included in this exploration, so subsequent July scoring is an exploratory chronological replay, not an untouched holdout.
Change: refit seasonal and HAR artifacts through July20 only; compare no overlay versus 1.5x and 2.0x sigma in the announcement hour on July28–30. Calendar dates are verified from the Fed's public meeting schedule; the upcoming scheduled September16 date is known in advance, but no post-cutoff outcomes are accessed. Test before deciding whether this extra rule earns inclusion.
Kill threshold: reject if crypto1h and crypto24h do not both improve in the chronological replay, or if reasonable multiplier changes reverse the effect. Do not infer market direction from an announcement.
Result / Verdict: pending.

### Iteration 11 initial results
On the unused August8–16 interval (72 prompts/asset at 1000 paths), robust half-hour HAR versus original hourly HAR gives COM 1839.3241 vs1855.1501, crypto24h 2174.9180 vs2176.4888, crypto1h 477.6199 vs477.8882. A day-cluster bootstrap of coefficient-weighted CRPS differences finds a clear COM improvement (-0.57%, 95% difference interval [-8.83,-6.59]); crypto1h difference is uncertain. These coefficient-weighted CRPS summaries are supplemental sensitivity checks, not field-relative positions.
A universal scale0.9 does not survive the regime check (crypto24h Aug2 6590.5807 vs6552.4661). MA(1) regresses COM and crypto24h on the new interval; its crypto1h advantage is only -0.015%, with an interval spanning zero. Verdict: retain the unscaled, independent-increment half-hour HAR mixture as the parsimonious finalist. Treat seed-level effects as uncertainty, not evidence of alpha.

### Iteration 11 seed result
The tiny fitted-dependence gain is not stable: at seeds123 and987 it worsens crypto1h on both original windows, while other changes flip sign. Verdict: discard MA(1) for serving. Keep the simpler independent Gaussian innovations with common path scale uncertainty.

## Iteration 13 — chronological refitting and September confirmation (pre-registered)
Hypothesis: refreshing the same frozen training recipe on the latest completed month maintains calibration as venue activity changes.
Change: fit seasonal profiles on July1–August31 and HAR through August31, then score the selected half-hour HAR mixture0.4 on September4–13 using only preceding context. Compare the seeded starter and the unchanged July31-fit candidate as diagnostics; do not reopen the discarded drift, tree, tail or dependence families based on September results. Generate 1000 paths on a 3h grid across all active assets/configurations. After confirmation, a final artifact refresh through September14 is allowed for the actual future serving window; clearly distinguish that fit from the chronologically validated checkpoint.
Kill threshold: a >1% configuration regression against the older checkpoint triggers investigation of training-window drift before adopting automatic refresh. If both checkpoints beat starter broadly, choose the more recent recipe only if its CRPS supports that choice.
Result / Verdict: pending.

### Iteration 12 result
On July28–30, no-event / 1.5x / 2.0x sigma gives crypto24h CRPS 3663.5795 / 3658.5177 / 3658.0830 and crypto1h 797.3021 / 797.0045 / 797.7107. The 1h effect reverses at2x and assets disagree. Verdict: do not include the calendar overlay in the core finalist under the preregistered robustness threshold. A scheduled event is a plausible research direction, but this small exploratory sample does not establish a stable adjustment.

## Iteration 14 — OHLCV information beyond close returns (pre-registered)
Hypothesis: within-minute ranges and trading activity contain volatility information that close-to-close HAR misses. A regularized log-linear extension should extrapolate more safely than the rejected tree model.
Change: extend HAR with past 1h/6h/24h range scales, relative volume and trade-count activity, and jump ratios. Fit on pre-August observations and test August windows only, preserving September as confirmation of the already frozen close-only finalist. Train both a pure extension and a 50% geometric blend with the established HAR forecast.
Kill threshold: need >=0.5% CRPS improvement on both original August windows in a configuration, then pass the August8–16 check before any candidate promotion. If the late experiment fails, retain the frozen core and do not retune to September outcomes.
Result / Verdict: pending.

Evaluation safeguard: the first September launch stopped before scoring because an exclusive-end loader requested the September15 midnight partition. Corrected the loader to end one minute before that unnecessary day; no future data were fetched. The intended September13 final prompts remain fully realized within September14.

September robustness failure: the first HAR runs stopped at AAPLX because a short feature window contained no observed return. Fixed feature construction by substituting the longer observed return scale for missing short-window summaries (with a pre-cutoff fitted prior only when no return scale exists). No missing price is filled or interpolated. Re-running the failed candidate evaluations; the starter had completed successfully. This is a contract fix, not score-based parameter tuning.

## Iteration 15 — forecast volatility term structure (pre-registered)
Hypothesis: a single forecast scale for an entire 24h or 1h horizon discards predictable decay of recent volatility. Separate future-block forecasts can allocate uncertainty more accurately across the scored increments.
Change: train the same regularized HAR features against nonoverlapping future blocks: 0–5/5–15/15–30/30–60min for 1h; 0–1/1–3/3–6/6–12/12–24h for 24h. Freeze training at July31 and half-hour seasonal normalization. Compare block forecasts alone and a 50% geometric blend with the established whole-horizon forecast. Sampling remains independent Gaussian increments with shared mixture0.4.
Kill threshold: require >=0.3% improvement in both August development windows per configuration, followed by the unused-August interval. Do not tune against the running September results.
Result / Verdict: pending.

### Iteration 13 results
September4–13, 80 prompts/asset/format, 1000 paths, identical mean realized coverage98.8718%:
- Starter COM / crypto24 / crypto1: 1920.4177 / 3561.5627 / 853.3771.
- July31-fit half-hour HAR mixture: 1651.0233 / 3389.2857 / 803.8759.
- August31-fit same recipe: 1646.3891 / 3399.9605 / 805.9141.
Every asset/format improves versus starter. Refresh improves COM but mildly regresses crypto (~0.25–0.32%), below the preregistered1% investigation threshold. Do not claim retraining universally helps. Coefficient-weighted CRPS reductions versus starter are14.16% COM,5.05% crypto24 and6.04% crypto1, with day-bootstrap difference intervals entirely negative. These intervals describe this period, not uncertainty about future market regimes.
Verdict: core HAR mixture passes confirmation. Preserve both dated checkpoints and examine refresh stability before choosing the final artifact policy; no discarded model family is revived based on these outcomes.

### Iteration 14 result
The OHLCV extension provides only a small crypto1h gain (618.1695→618.0873 in Aug1;1180.7738→1178.0644 in Aug2), below the0.5% two-window threshold. It does not consistently improve either24h group. Verdict: discard additional range/activity features from the champion. The small effect does not justify more dependencies and missing-column handling.

### Iteration 15 result
Blockwise volatility forecasts barely move COM CRPS (2969.8478→2969.0918 and1991.5581→1990.9009) and slightly worsen both crypto groups. The changes are below the0.3% preregistered threshold. Verdict: discard the additional term-structure regressions. Multiple richer feature/path families have now failed to improve the confirmed simple model reliably.

## Iteration 16 — train directly against official CRPS (pre-registered)
Hypothesis: the HAR fitting surrogate (future log mean absolute returns) leaves a small calibration gap that direct CRPS selection on a separate training period can close.
Change: use a July20-fit checkpoint and train calibration adjustments only on July21–27, before all August selection windows. Search global forecast-scale and path-volatility-mixture parameters per configuration, at1000 paths, through the official backtester scorer. Preserve the frozen underlying HAR features and do not optimize ranks. Choose broad minima, then test August1–7,17–23 and8–16 without refitting on their outcomes.
Kill threshold: direct calibration must improve both initial August windows by >=0.3% for a configuration and not reverse in the middle window; otherwise preserve scale1/mixture0.4. This is a final check of whether the training objective, rather than model complexity, is limiting the baseline.
Result / Verdict: pending.

## Iteration 17 — bounded neural correction (pre-registered)
Hypothesis: a small smooth residual model can use nonlinear volatility features while preserving HAR's extrapolation, avoiding the tree model's failure in the volatile window.
Change: train an ensemble of three small networks on log future absolute-return errors relative to HAR, with bounded mean correction (±0.3 log units), strong regularization, and a Gaussian likelihood for conditional scale. All targets end July31. Use one GPU job, capped at35% memory. Serve the learned weights with NumPy, so inference does not need a GPU. Compare bounded mean correction and a conservative half correction at fixed mixture0.4; treat learned uncertainty as exploratory until the mean passes.
Kill threshold: >=0.5% CRPS gain on both August windows per configuration, then the middle-August interval. No promotion based on training likelihood.
Result / Verdict: pending.

### Iteration 16 result
The July CRPS grid has a broad minimum near the existing settings: COM scale1.0–1.1/mixture0.4, crypto24 scale0.9–1.0/mixture0.4, crypto1 scale0.9–1.0/mixture0.2–0.4. The new crypto1 training choice scale0.9/mixture0.2 scores617.2625 vs618.1695 on Aug1, but1181.8137 vs1180.7738 on Aug2 and478.3716 vs477.6199 in mid-August. Verdict: discard the extra calibration change. Small training-week optima did not transfer consistently, supporting the simpler established scale1/mixture0.4.

## Iteration 18 — incomplete final candle (pre-registered)
Hypothesis: prompt issuance cuts the context partway through a five-minute candle, whereas HAR training uses complete candles. Including that short final candle can bias the fastest volatility feature downward.
Change: for 24h forecasting features only, drop an incomplete final five-minute bin while keeping every observed minute for the required first path price. The 1h minute-grid model is unchanged. Compare against established half-hour HAR on both August windows; do not change labels or interpolate prices.
Kill threshold: keep only if both windows improve and the effect survives exact-prompt replay; otherwise treat the mismatch as negligible relative to losing recency.
Result / Verdict: pending.

### Iteration 17 result
The bounded neural ensemble was trained successfully on the L4 within its memory cap. It fails the preregistered CRPS criterion: full correction gives COM2992.1154/2008.6051 versus2984.5192/2006.9817, crypto24h2755.2967/6651.3269 versus2752.6606/6549.7428, and only tiny crypto1h gains617.8559/1183.4360 versus618.4267/1183.8016. The half correction is similarly insufficient. Verdict: discard neural serving dependency. Bounded corrections still fit regime-specific residual patterns that do not transfer to the busy window.

### Iteration 18 result
Dropping incomplete candles changes COM by effectively0% in Aug1 and-0.03% in Aug2, while mildly worsening crypto24h in both. Verdict: discard this additional branch; the loss of recency offsets the cleaner interval interpretation. Keep the last observed price as the path origin in all cases.

## Iteration 19 — longer training history (pre-registered)
Hypothesis: adding the available2025 crypto history improves estimation of volatility persistence and reduces dependence on a single2026 regime. This changes training sample size, not the feature architecture or CRPS calibration.
Change: refit crypto HAR from August1,2025 rather than January1,2026, still ending July31 for August validation. Keep the same seasonal normalization, ridge penalty and mixture. Compare on the three August windows; no new per-asset hyperparameters.
Kill threshold: prefer longer history only if it improves both original August windows and does not regress the middle interval; otherwise retain the established recent-history recipe. Commodity/equity fitting remains unchanged because its early venue history is structurally thinner.
Result / Verdict: pending.

### Iteration 19 initial result
Adding2025 crypto data does not improve either group consistently: crypto1h618.2817/1181.0344 versus618.1695/1180.7738; crypto24h2750.8599/6552.4049 versus2750.0119/6552.4661. Verdict: keep the established2026 training window. The older sample changes predictions very little and does not earn a new serving recipe.

## Iteration 20 — freeze serving implementation and audit causality
Hypothesis: removing research switches and enforcing artifact availability at issuance preserves the confirmed forecast while reducing deployment errors.
Change: archived all experimental branches in research_modeling.py. The final modeling.py loads one local artifact registry, selects the newest checkpoint available before prompt issuance, and uses only HAR + robust half-hour seasonality + mixture0.4. It has no model-selection environment switches. Artifact checkpoints end July20,July31,August31,September14; availability includes the close time of the final minute and the format-specific issuance lead.
Result: bit-for-bit path equality against the confirmed August31 research model on BTC1h/BTC24h/AAPLX24h/SP50024h at1000 paths. Boundary checks prove September14 artifacts are not used at September15 midnight, but become available at00:01 for1h and00:02 for24h. Contract checks cover both grids, arbitrary path counts, missing observations, deterministic same-seed replay and changed output with a different seed.
Verdict: keep the simplified serving implementation. September14 refresh uses the validated recipe but has no post-cutoff CRPS measurement; this limitation will be explicit in the report.
Lesson: the price-data cutoff alone is not enough; a fitted artifact also needs an availability timestamp that respects issuance lead time.

## Final audit and nomination rationale
Full official orchestration is complete for all18 asset-formats:1726 COM,1080 crypto24 and6480 crypto1 prompts per model,1000 paths,100% realized coverage. Configuration-average CRPS starter→champion is2124.404499→1854.090141 (COM),3851.862646→3728.017778 (crypto24),808.128197→768.518439 (crypto1 historical rules). Every asset-format improves. Separate current-rule grid replay including1h volatility also improves853.3771→805.9141. Full per-asset and prompt-level evidence is committed.

The final serving implementation contains only the validated HAR mixture and dated artifacts. It supports all formats, arbitrary positive path counts, both benchmark seeds, sparse contexts and offline execution. The final September14-fit artifact passes all18 asset-format shape/positivity/start-price checks in0.02–0.04seconds per1000-path prompt. Its future CRPS is unknowable within the cutoff, and is not reported as measured.

Early nomination is supported by the validation plateau, not by exhaustion of the envelope: after the core gains, drift, tree models, GPU neural residuals, OHLCV features, volatility term structure, empirical distributions, block resampling, dependence, finer profiles, direct CRPS calibration, older history and incomplete-candle handling all failed their transfer/stability thresholds. Small apparent improvements reversed with another window or seed. Continuing to search these same observations would chiefly increase selection bias. I therefore stop research with the simple, repeatedly confirmed model rather than assert that remaining money proves another experiment will help. This is a judgment about expected generalization, not a proof of global optimality.

Artifacts and reproducibility: research_modeling.py preserves the experimental switches; research runners point to it, while modeling.py is the clean champion. Older experiment outputs remain the historical record; later robustness fixes are recorded in the journal and git history. Final reports distinguish historical scoring rules, current-rule replay, and the unobserved future window.
