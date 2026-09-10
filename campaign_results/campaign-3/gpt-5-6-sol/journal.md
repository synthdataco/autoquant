# Research journal

## Iteration 1 — starter baseline and horizon diagnosis (2026-09-09)

**Hypothesis.** The starter seven-day homoskedastic lognormal model is a useful floor, but its
one-minute volatility estimate (including returns across no-trade gaps) and zero-drift dynamics
will be poorly calibrated, especially for CRYPTO_1H. Establish field-relative and CRPS baselines
on a recent window before changing the model.

**Change.** None: evaluate the supplied starter. First use 200 paths on BTC over 2026-08-27 to
2026-08-28 in both formats as a fast diagnostic, then retain 1000-path results for nomination-grade
tests only.

**Kill threshold.** Replace the starter unless it is already competitive with the top quartile of
the archived field in both formats. A candidate change must improve more than Monte Carlo noise on
at least two disjoint windows and must not materially damage either horizon.

**Result.** BTC, 2026-08-27→28, 200 paths: 24h mean CRPS 3284.98 (only one fully
realized prompt in the one-day boundary diagnostic), 1h mean CRPS 708.93 over 139 prompts;
final per-asset smoothed score 86.67.

**Verdict.** Discard starter unchanged. It is clearly sensitive to stale seven-day volatility.

**Lesson.** Use multi-day 24h windows because the realized horizon removes the right boundary;
the one-day 1h diagnostic is already informative.

## Iteration 2 — volatility recency and width (2026-09-09)

**Hypothesis.** A 24-hour volatility window responds better to the current regime than seven
days, while modestly shrinking Gaussian width improves CRPS because short-horizon realized
returns are concentrated near zero.

**Change.** Make context gaps explicit (only adjacent observed minutes form returns), use a
stable prompt-derived random seed, and test volatility windows of 1, 3, and 7 days with sigma
scales 0.70–1.15. These switches are experiment-only environment variables; default behavior
remains self-contained.

**Kill threshold.** Keep only if the best setting improves both raw CRPS and smoothed score on
the diagnostic, then survives a disjoint earlier window and non-BTC assets.

**Result.** BTC 1h, 2026-08-27→28, 200 paths (139 prompts): 1d/0.70 = 669.78 CRPS,
score 47.77; 1d/0.85 = **665.21**, 43.14; 1d/1.00 = 668.30, 46.05;
3d/0.85 = 669.54, 47.44; 7d/0.85 = 680.68, 58.64; 7d/1.15 = 732.00,
110.04. Best versus starter: -6.2% CRPS and -50.2% smoothed score.

**Verdict.** Provisionally keep 1-day window / 0.85 scale; validation pending.

**Lesson.** Recent volatility matters more than additional samples for liquid BTC in this
regime, and raw seven-day volatility was too wide.

## Iteration 3 — disjoint-window and cross-asset 1h check (2026-09-09)

**Hypothesis.** The 1d/0.85 setting generalizes across the five crypto assets and an earlier,
lower-volatility week rather than merely fitting BTC on August 27.

**Change.** Evaluate all crypto assets on 2026-08-07→08 and 2026-08-27→28. For BTC, also
repeat nearby widths and 3d/7d windows on the earlier split.

**Kill threshold.** Discard 1d/0.85 if another setting improves earlier BTC CRPS by >1% or if
the model fails on legitimate sparse context.

**Result.** Candidate mean CRPS (earlier, recent): BTC (415.95, 665.21), ETH
(543.67, 871.45), SOL (568.56, 1525.46), XRP (722.66, 1319.72), HYPE
(1031.67, 1478.38). Earlier BTC alternatives were 1d/0.70 417.96, 1d/1.00 418.90,
3d/0.85 416.11, and 7d/0.85 416.07. Thus 1d/0.85 remained best, though by only
0.03% versus longer windows in the earlier regime.

**Verdict.** Keep for 1h. Cross-asset field-relative aggregation remains to be run at 1000
paths.

**Lesson.** Recency gives a large gain in volatile conditions without a meaningful quiet-regime
penalty. The supplied evaluator's strict `load_range` contradicted its documented lenient intent
and crashed on 2–3 legitimate HYPE NaNs; replacing it with partition reads that preserve NaNs
fixed evaluation without interpolation.

## Iteration 4 — 24h calibration (2026-09-09)

**Hypothesis.** The 24h horizon benefits from a longer volatility window and potentially a
different width than 1h, so horizon-specific calibration is required.

**Change.** Grid 1d/3d/7d volatility windows and 0.70/0.85/1.00 scales on BTC across recent
and earlier three-day windows using 200 paths.

**Kill threshold.** Select no horizon-specific rule unless it improves the average of both
windows and avoids >3% regression on either.

**Result.** The initial grid suggested a regime-dependent width, but this run used approximate
regular-grid predictions. Iteration 5 demonstrated that such predictions can be reused for
different jittered field starts, so these numbers are not valid selection evidence.

**Verdict.** Invalidate and replace with the exact-start tests in Iterations 5–7.

**Lesson.** Evaluator validity takes precedence over a larger parameter grid.

## Iteration 5 — exact prompt alignment (2026-09-09)

**Hypothesis.** Regular hourly predictions are not a valid evaluation proxy for the jittered
field schedule, particularly CRYPTO_1H, because synth-lib matches the nearest file within 30
minutes without rebasing it to the realized start price.

**Change.** Have `predict.py` read exact archived prompt starts from the offline score bundle and
generate a fresh context/ensemble at every start (about 144/day for 1h and 24/day for 24h).

**Kill threshold.** All subsequent reported selection and nomination tests must use exact starts;
mark approximate-grid diagnostics as exploratory only.

**Result.** On BTC 2026-08-27→28 the corrected 1h evaluator generated/scored 144 distinct
predictions rather than generating 24 and reusing them for 139 scores. The 1d/0.85 candidate
scored 663.42 CRPS. Exact-start 24h three-day tests (72 prompts) gave, for recent scale
0.60/0.75/0.90/1.05: 3219.11/3193.84/3219.74/3283.96; earlier:
1272.47/1288.38/1323.70/1373.31. For exact 1h, 0.85 was best on both windows:
663.42 recent and 415.68 earlier.

**Verdict.** Keep exact-start evaluator. Select 0.85 for 1h and 0.70 as the two-regime compromise
for 24h pending competition-wide tests.

**Lesson.** Prediction cadence/alignment can silently dominate model comparisons. Approximate-grid
results in Iterations 1–4 are retained for auditability but are not nomination evidence.

## Iteration 6 — native-step volatility and variance-reduced ensemble (2026-09-09)

**Hypothesis.** Estimating 5-minute volatility directly from endpoint-valid 5-minute returns
matches the 24h scorer better on sparse assets, and antithetic innovations reduce empirical-CRPS
noise without changing the intended Gaussian law.

**Change.** Estimate at `time_increment` (1m for 1h, 5m for 24h), never filling NaNs. Pair every
normal innovation path with its negative. Use stable prompt-derived seeds.

**Kill threshold.** Keep antithetic sampling only if it improves exact-start BTC on both formats;
served latency must remain far below five seconds at 1000 paths.

**Result.** With 200 paths, antithetic sampling improved exact recent BTC 1h from 663.42 to
662.01 CRPS and 24h to 3190.88 CRPS. Direct timing at 1000 paths was 0.005 s (1h) and
0.021 s (24h), excluding context I/O.

**Verdict.** Keep.

**Lesson.** Deterministic antithetic fans are a cheap serving-quality improvement and make close
research comparisons more reproducible.

## Iteration 7 — nomination-grade field baseline (2026-09-09)

**Hypothesis.** The exact-start, native-step, antithetic candidate is competitive across the
field after asset-coefficient aggregation, not merely on BTC raw CRPS.

**Change.** Generate 1000 paths at every archived prompt start for every active asset, using
three scored days for each 24h competition and two scored days for CRYPTO_1H. Replay official
per-asset and combined rankings with synth-lib. SPYX remains excluded because it is retired.

**Kill threshold.** Do not nominate unless all three combined ranks and reward weights are
available. If a competition is materially weak, optimize its highest-coefficient assets before
expanding the validation window.

**Result.** CRYPTO_1H exact-start 1000-path replay over 2026-08-26→28 scored 288
prompts/asset. Mean CRPS: BTC 659.48, ETH 871.72, SOL 1369.25, XRP 1542.89,
HYPE 1543.68. Combined rank **246/257**, reward weight 8.48e-10 versus top
2.82e-3. Per-asset final ranks were BTC 256, ETH 256, SOL 242, XRP 256, HYPE 225.

**Verdict.** Discard as a champion. The model is far outside the competitive reward region.

**Lesson.** Raw CRPS only modestly worse than the field can translate to essentially zero
emissions after prompt normalization, smoothing, asset weighting, and softmax. Optimize for the
field-relative combined result, not visual plausibility.

## Iteration 8 — width stress test against the field (2026-09-09)

**Hypothesis.** The 0.85 width selected on two isolated days is too wide over the full recent
two-day window; narrower fans may approach the field median, particularly on XRP.

**Change.** Exact-start official backtests at 200 paths for BTC and XRP over
2026-08-26→28, sigma scales 0.25, 0.40, 0.55, 0.70, and 0.85.

**Kill threshold.** A new global width must improve both assets; otherwise introduce a
data-dependent volatility forecast rather than per-asset constants.

**Result.** Recent two-day exact tests: BTC CRPS at scales 0.25/0.40/0.55/0.70/0.85
= 697.13/654.94/**640.42**/644.82/662.34; XRP =
1505.60/1427.45/**1424.79**/1470.53/1548.33. Cross-assets: ETH preferred
0.55 (831.69), while SOL and HYPE preferred 0.70 (1354.04 and 1526.28).
On the earlier 2026-08-06→08 split, however, 0.85 remained modestly better for every
asset: candidate vs 0.85 CRPS was BTC 397.70 vs 394.71, ETH 529.20 vs 527.65,
SOL 563.37 vs 559.88, XRP 722.13 vs 685.83, HYPE 967.80 vs 961.28.

**Verdict.** Discard a permanently narrower global width. Seek a regime-adaptive forecast;
the large XRP reversal makes a recent-only per-asset constant unsafe.

**Lesson.** The effective calibration target contracts following high-volatility contexts.
Median context sigma approximately doubled between splits for XRP (5.14 to 11.67 bp/min),
while its preferred scale fell from 0.85 to 0.55.

## Iteration 9 — shorter volatility memory (2026-09-09)

**Hypothesis.** A 6h or 12h estimator with the stable 0.85 calibration adapts after isolated
volatility bursts and reproduces the recent narrow-width gain without sacrificing the earlier
regime.

**Change.** Exact-start official tests on BTC and XRP using 6h, 12h, and 24h windows with scale
0.85 on both two-day splits.

**Kill threshold.** Keep only a window that beats 24h/0.85 on the recent split and stays within
2% on the earlier split for both assets.

**Result.** Six-hour/0.85 versus 24-hour/0.85 CRPS on earlier and recent splits:
BTC 395.61/640.78 versus 393.90/642.13; XRP 683.65/1425.64 versus
680.41/1419.44. The six-hour rule also generalized cross-asset: ETH 526.86/835.20,
SOL 561.27/1354.16, HYPE 965.95/1527.62. Relative to recent best fixed widths,
regressions were small (ETH +0.4%, SOL +0.01%, HYPE +0.09%), while it avoided the
large earlier XRP regression of fixed 0.55.

**Verdict.** Keep six-hour/0.85 as the 1h model. It satisfies the 2% two-regime threshold.

**Lesson.** Short volatility memory is a robust substitute for regime-specific scale fitting.
Also, Iteration 7 unintentionally exercised the then-served seven-day default (environment
switches were absent), explaining part of its weak result; the served default is now explicitly
horizon-specific.

## Iteration 10 — corrected 1h combined replay (2026-09-09)

**Hypothesis.** Moving the served 1h default from seven days to the validated six-hour estimator
materially improves combined field rank at 1000 paths.

**Change.** Regenerate all five assets at exact starts and 1000 paths over 2026-08-26→28,
then rerun official combined aggregation.

**Kill threshold.** Require a substantial combined-rank improvement over 246/257. If still
outside the reward-bearing cluster, test distributional shape or conditional location next.

**Result.** 1000-path means over 288 prompts: BTC 639.11, ETH 833.88, SOL
1352.17, XRP 1422.79, HYPE 1525.81. Combined rank improved from 246/257 to
**213/257**; reward weight 1.276e-4 versus leader 2.821e-3 (4.5%). Individual ranks:
BTC 234, ETH 233, SOL 212, XRP 229, HYPE 216.

**Verdict.** Keep the six-hour default, but do not nominate: meaningful field-relative headroom
remains.

**Lesson.** A 20–120 CRPS improvement per asset translated into a 33-place combined-rank gain.
The remaining gap is not likely to be solved by width alone.

## Iteration 11 — conditional location (2026-09-09)

**Hypothesis.** Very short crypto returns contain mean reversion from microstructure and/or
short-lived momentum. A small drift forecast derived only from the last 5–60 minutes can improve
the repeatedly scored gap and absolute intervals.

**Change.** Add an experiment-only drift coefficient multiplying the average recent log return,
and test lookbacks 5, 15, 30, 60 minutes with coefficients -0.25, 0, +0.25 on BTC and XRP.

**Kill threshold.** Keep only if a single sign improves both assets and both validation regimes;
otherwise retain zero drift.

**Result.** Every nonzero drift was harmful. Recent BTC baseline was 640.78 CRPS;
5m coefficients -0.25/+0.25 produced 730.97/729.25, 15m 663.41/670.97,
and 60m 640.49/652.93. XRP baseline was 1425.64; corresponding results were
1600.87/1611.46, 1467.06/1486.28, and 1425.17/1450.67. Even the tiny raw gains
for 60m mean reversion worsened field-relative smoothed scores.

**Verdict.** Discard drift; retain martingale location.

**Lesson.** Short-horizon sample means are far noisier than volatility estimates, and the
validator's many interval scores amplify location errors.

## Iteration 12 — robust volatility (2026-09-09)

**Hypothesis.** Standard deviation overreacts to isolated jumps. A MAD volatility estimate over
the same six-hour window may generalize better across calm and turbulent regimes.

**Change.** Compare Gaussian fans whose step sigma is 1.4826 times return MAD, with scales
1.0, 1.25, and 1.5, against the standard-deviation baseline on BTC and XRP in both splits.

**Kill threshold.** Keep MAD only if one scale improves both assets on average without a >2%
regression in either regime.

**Result.** MAD was consistently slightly worse. BTC earlier/recent baseline std CRPS was
395.61/640.78; best MAD (scale 1.0) was 399.18/641.45. XRP baseline
683.65/1425.64; best MAD was 685.06/1426.32. Wider MAD scales deteriorated rapidly.

**Verdict.** Discard MAD.

**Lesson.** Outliers in the short window contain useful volatility information; shortening the
window worked better than robustly suppressing its tails.

## Iteration 13 — serial dependence in simulated increments (2026-09-09)

**Hypothesis.** IID Brownian increments miscalibrate multi-minute horizons when one-minute
returns are mean-reverting. A modest negative AR(1) correlation can preserve one-minute width
while narrowing accumulated gap distributions.

**Change.** Apply stationary AR(1) correlation to Gaussian path increments, testing rho
-0.20, -0.10, 0, +0.10 on BTC and XRP in both splits; retain six-hour std and 0.85 width.

**Kill threshold.** Keep only if one rho improves both assets on average and does not regress
either split by >1%.

**Result.** No tested correlation met the threshold. BTC earlier/recent baseline was
395.61/640.78; rho -0.10 yielded 395.47/640.88 and rho -0.20
396.13/642.29. XRP baseline 683.65/1425.64; rho -0.10 yielded
685.67/1427.39, while +0.10 helped earlier (682.91) but hurt recent (1426.81).

**Verdict.** Discard AR dependence; keep IID increments.

**Lesson.** The gain from modeled linear dependence is smaller than its regime instability.

## Iteration 14 — 24h competition-wide baseline (2026-09-09)

**Hypothesis.** The one-day/native-5m/0.70 model may be more competitive in 24h profiles than
the initial 1h family because its estimator directly matches the dominant 5m scoring interval.

**Change.** Exact-start official 200-path backtests over 2026-08-26→29 for every active crypto,
commodity, and equity asset, followed by combined field replay. Exclude retired SPYX.

**Kill threshold.** If a profile ranks outside the top half, tune width/window on its
highest-coefficient weak assets before running 1000 paths.

**Result.** At 200 paths and 72 prompts/asset, CRYPTO_24H combined rank was
**219/257**, reward 1.73e-7 versus leader 1.52e-2. COM_EQU_24H rank was
**248/257**, reward 4.42e-11 versus leader 4.31e-3. Crypto per-asset CRPS:
BTC 3190.88, ETH 3744.68, SOL 6504.62, XRP 6440.55, HYPE 7230.38.
COM/EQU: XAU 1777.77, NVDAX 4524.86, GOOGLX 1610.22, TSLAX 2729.34,
AAPLX 1759.30, WTIOIL 3335.41, SPCX 3400.12, SP500 782.73.

**Verdict.** Reject baseline for nomination and tune high-coefficient assets.

**Lesson.** The field is extremely compressed: for example SP500 field best/median CRPS was
745/756 versus ours 783, and BTC 3133/3163 versus ours 3191. A few percent raw deficit can
place the model near the bottom.

## Iteration 15 — high-weight 24h width grid (2026-09-09)

**Hypothesis.** The common 0.70 scale is inappropriate across assets; direct scale calibration
on high-coefficient assets can recover most of the field deficit.

**Change.** Exact recent three-day grid at scales 0.30, 0.50, 0.70, 0.90 for BTC, SOL, HYPE,
SP500, XAU, NVDAX, and AAPLX. Validate selected scales on 2026-08-07→10 before keeping.

**Kill threshold.** Require >1% recent improvement and <3% earlier regression. Prefer a common
rule if asset-specific optima are not stable.

**Result.** Recent optimal tested scales/CRPS: BTC 0.70/3190.88; SOL 0.90/6483.65;
HYPE 0.90/7158.22; SP500 0.90/777.19; XAU 0.90/1775.21;
NVDAX 0.70/4524.86; AAPLX 0.50/1742.11. Width alone left large deficits for
NVDAX (field median 4259) and AAPLX (1632), so the planned earlier validation was deferred
until testing distributional shape.

**Verdict.** Keep wider 0.90 as a provisional liquid/high-volatility setting; do not keep
asset-specific scales yet.

**Lesson.** Width cannot fit both the central spike and jump tails of tokenized equities.

## Iteration 16 — empirical innovation shape for sticky assets (2026-09-09)

**Hypothesis.** Resampling standardized historical native-step returns preserves the point mass
near zero and heavy tails that a Gaussian misses, improving AAPLX/NVDAX substantially.

**Change.** Compare IID empirical-bootstrap and Gaussian innovations at the best recent width
for AAPLX, NVDAX, SP500, XAU, SOL, and HYPE on the recent split, then validate improvements on
the earlier split.

**Kill threshold.** Keep bootstrap only where it improves recent CRPS >1% and does not regress
earlier CRPS >2%; prefer a feature-based rule (observed zero-return fraction) over asset names.

**Result.** Bootstrap improved recent NVDAX from 4524.86 (Gaussian 0.70) to
4464.65 at scale 0.90 (1.33%), but its earlier CRPS was 1963.34 versus Gaussian
1861.89 (>5% regression). AAPLX bootstrap best was 1738.82 versus Gaussian
1742.11 (<1% gain) and regressed earlier from 1291.19 to 1306.14. SP500 and HYPE
were flat/worse; XAU and SOL improved only 0.26% and 0.44%.

**Verdict.** Discard empirical innovations under the preregistered threshold.

**Lesson.** Historical marginal shape is highly regime-sensitive; Gaussian shrinkage is more
stable even when it cannot match the field leaders.

## Iteration 17 — short-memory 24h volatility (2026-09-09)

**Hypothesis.** As in CRYPTO_1H, six-hour volatility may adapt faster than a full day and reduce
forecast error for 24h without asset-name overfitting.

**Change.** Compare 6h and 12h windows at scale 0.85 against current best one-day settings for
BTC, SOL, SP500, NVDAX, and AAPLX on both recent and earlier three-day splits.

**Kill threshold.** Keep a common shorter window only if it improves the mean across assets and
splits, with no high-coefficient asset regressing >3%.

**Result.** Short windows did not improve the cross-asset mean. Recent 6h/12h CRPS:
BTC 3203.32/3193.65, SOL 6488.64/6484.93, SP500 784.07/780.03,
NVDAX 4654.48/4623.52, AAPLX 1848.98/1828.00. Earlier results also showed
material NVDAX/AAPLX instability. The one-day tuned baselines remained better.

**Verdict.** Discard short-memory 24h rule.

**Lesson.** The 1h recency result does not transfer: a 24h forecast needs enough context to
stabilize its terminal distribution.

## Iteration 18 — longer-memory thin-asset volatility (2026-09-09)

**Hypothesis.** NVDAX and AAPLX require more than one day to stabilize native-step volatility;
three- or seven-day windows may remove the large field deficit without distributional overfit.

**Change.** Test 3d and 7d windows at scales 0.50, 0.70, 0.90 for NVDAX, AAPLX, and SP500
on recent and earlier splits.

**Kill threshold.** Keep a longer window only if it improves both splits or yields >2% recent
gain with <1% earlier loss.

**Result.** Longer memory was not robust. NVDAX recent improved from 4524.86 to
4437.05 (3d/0.70), but earlier worsened from 1861.89 to 1974.30. AAPLX best
recent was 1735.87 (7d/0.50) versus 1742.11, but earlier worsened to 1336.77
versus 1291.19. SP500 7d/0.70 was essentially flat recent (778.47 versus 777.19).

**Verdict.** Discard a longer common window.

**Lesson.** Longer samples reduce estimation noise but lag volatility regimes; neither alone
solves the 24h deficit.

## Iteration 19 — arithmetic-martingale correction (2026-09-09)

**Hypothesis.** Zero-mean log shocks introduce a positive arithmetic drift. Subtracting half the
step variance should center expected prices on the last observation and improve terminal absolute
CRPS, especially for volatile 24h crypto.

**Change.** Compare identical Gaussian fans with and without `-0.5 sigma^2` per step for BTC,
SOL, HYPE, and SP500 on recent and earlier splits.

**Kill threshold.** Keep if it improves the mean and does not regress any asset/split >0.5%.

**Result.** Effects were negligible and mixed. Recent/earlier deltas (corrected minus
uncorrected CRPS): BTC -0.34/+0.06, SOL +0.31/+0.84, HYPE +1.95/-0.82,
SP500 +0.04/+0.06.

**Verdict.** Discard; retain zero-mean log increments.

**Lesson.** The Jensen drift is too small relative to forecast uncertainty over these windows.

## Iteration 20 — exponentially weighted volatility (2026-09-09)

**Hypothesis.** Hard rolling windows discard information abruptly. An EWMA over seven days can
combine regime responsiveness with stable estimation, improving both recent and earlier 24h
splits.

**Change.** Estimate native-step variance with exponential half-lives of 1h, 6h, and 24h over
the full context; test on BTC, SP500, NVDAX, and AAPLX using their provisional widths.

**Kill threshold.** Keep a common half-life only if average CRPS improves and no asset/split
regresses >2%.

**Result.** A 24h half-life improved recent BTC (3170.61 vs 3190.88), SP500
(774.35 vs 777.19), NVDAX (4450.34 vs 4524.86), and AAPLX (1737.66 vs
1742.11). Earlier changes were BTC 1293.90 vs 1278.36, SP500 725.22 vs
724.20, NVDAX 1939.88 vs 1861.89, AAPLX 1308.94 vs 1291.19. NVDAX breached
the 2% regression threshold; shorter half-lives were generally worse.

**Verdict.** Do not adopt EWMA globally. Retain as a possible recent-regime sensitivity case.

**Lesson.** Smooth weighting modestly helps the latest high-volatility split but cannot avoid
lag/variance tradeoffs across regimes.

## Iteration 21 — moderate heavy tails (2026-09-09)

**Hypothesis.** A standardized Student-t innovation offers more stable tail calibration than
empirical bootstrap while retaining a smooth central density.

**Change.** Compare t innovations with 5 degrees of freedom against Gaussian innovations for
BTC, SOL, SP500, NVDAX, and AAPLX on both splits, holding estimator and width fixed.

**Kill threshold.** Keep only if average CRPS improves with no >1% regression.

**Result.** Student-t improved both splits for SOL (2395.07/6460.85 vs
2403.97/6483.65), SP500 (716.76/776.19 vs 724.20/777.19), NVDAX
(1830.00/4496.08 vs 1861.89/4524.86), and AAPLX (1284.16/1741.68 vs
1291.19/1742.11). BTC improved earlier (1270.02 vs 1278.36) but regressed
recent by 0.29% (3200.22 vs 3190.88). No regression exceeded 1%.

**Verdict.** Provisionally keep t5 for 24h; extend cross-asset validation.

**Lesson.** Moderate parametric tails are more stable than empirical resampling and especially
help the thin listings.

## Iteration 22 — cross-asset t5 width calibration (2026-09-09)

**Hypothesis.** Student-t gains extend to the untested assets, but their optimal widths differ
systematically between liquid crypto/indices and tokenized equities.

**Change.** Recent exact three-day grid at t5 scales 0.50, 0.70, 0.90 for ETH, XRP, HYPE,
XAU, GOOGLX, TSLAX, WTIOIL, and SPCX; validate selected settings on the earlier split.

**Kill threshold.** Keep per-asset widths only if the selected recent optimum is within 3% of
the best earlier alternative; otherwise use the common 0.70 width.

**Result.** Recent t5 optima: ETH 0.90/3741.83, XRP 0.90/6412.04,
HYPE 0.90/7164.88, XAU 0.90/1768.01, GOOGLX 0.70/1599.62,
TSLAX 0.70/2707.19, WTIOIL 0.90/3314.92, SPCX 0.50/3345.14.
Earlier selected t5 versus common Gaussian 0.70: ETH 1702.14 vs 1659.00,
XRP 2478.68 vs 2487.51, HYPE 3609.42 vs 3603.45, XAU 1450.87 vs
1418.49, GOOGLX 1670.50 vs 1714.64, TSLAX 2131.95 vs 2171.51,
WTIOIL 3021.64 vs 2988.77, SPCX 7131.73 vs 7296.77. All selected widths
were within the preregistered 3% earlier tolerance.

**Verdict.** Keep horizon/asset-aware widths and t5 for 24h except BTC, where Gaussian was
slightly stronger on the latest split.

**Lesson.** Optimal width clusters sensibly: 0.90 for liquid/high-volatility assets, 0.70 for
several token equities, and 0.50 for AAPLX/SPCX. This is more defensible than unconstrained
per-asset optimization.

## Iteration 23 — combined 24h candidate replay (2026-09-09)

**Hypothesis.** The validated t5 and clustered-width changes materially improve both combined
24h field ranks versus Iteration 14.

**Change.** Make the clustered settings served defaults, regenerate every active asset at 200
paths over the recent exact three-day window, and run both official combined replays.

**Kill threshold.** Require both ranks to improve. Proceed to 1000 paths only if reward weight
also rises materially.

**Result.** CRYPTO_24H improved from rank 219 to **176/257**, reward 5.30e-6
versus 1.73e-7. COM_EQU_24H improved only from 248 to **246/257**, reward
4.98e-10. Both satisfied rank improvement, but only crypto showed a material reward gain.

**Verdict.** Keep t5/clustered widths provisionally; continue COM/EQU research before 1000-path
nomination testing.

**Lesson.** Stable per-asset raw gains can materially shift crypto rank, but the tokenized-equity
gap remains too uniform to be explained by width alone.

## Iteration 24 — multi-hour trend extrapolation for tokenized equities (2026-09-09)

**Hypothesis.** A weak shared momentum or mean-reversion component over 6–24h explains part of
the common COM/EQU deficit that volatility changes cannot.

**Change.** On GOOGLX, TSLAX, AAPLX, and SPCX, test a 6h return-mean drift with coefficients
-0.10, -0.05, +0.05, +0.10 while holding each t5 width fixed.

**Kill threshold.** Keep only a common sign that improves at least three assets and validates on
the earlier split; otherwise zero drift remains.

**Result.** No common sign emerged. GOOGLX weakly preferred positive drift (1598.45 vs
1599.62), TSLAX negative (2700.54 vs 2707.19), AAPLX zero (1741.68), and
SPCX weakly negative (3343.55 vs 3345.14). Improvements were <0.3%.

**Verdict.** Discard drift.

**Lesson.** The shared deficit is not explained by a shared directional effect.

## Iteration 25 — pooled HAR volatility forecast (2026-09-09)

**Hypothesis.** A fitted combination of daily, three-day, and seven-day realized variance can
reduce the lag/noise tradeoff seen in hard windows and EWMA. Pooling normalized variance across
assets should stabilize coefficients while preserving asset-specific long-run levels.

**Change.** Fit a log-HAR regression using snapshot prices only through 2026-07-26, validate its
next-day variance RMSE on 2026-07-27→08-29, then expose the forecast as an experiment-only
volatility estimator and score it officially.

**Kill threshold.** Do not run an expensive CRPS grid unless out-of-sample log-variance RMSE
beats both last-day variance and the seven-day mean. Keep in the champion only if official CRPS
improves both recent and earlier splits.

**Result.** Pooled log-HAR validation RMSE was 0.995 versus 1.160 for last-day
variance and 1.117 for the seven-day mean. Fitted coefficients (intercept/day/3d/7d)
were -0.152/0.461/-0.430/0.687. Official tests supported HAR for SOL, XAU,
SP500, NVDAX, GOOGLX, TSLAX, and AAPLX. Examples earlier/recent: SOL
2404.00/6392.44, SP500 713.12/773.67, XAU 1403.65/1765.72, GOOGLX
1610.20/1590.94. It failed or materially regressed BTC, ETH, HYPE, WTIOIL; XRP
and SPCX were essentially flat with unfavorable recent/earlier tradeoffs.

**Verdict.** Keep HAR selectively for the seven assets where official two-split evidence agrees;
retain rolling estimates elsewhere.

**Lesson.** Better variance RMSE is helpful but not sufficient for CRPS. Selective use based on
official scoring avoids imposing a pooled statistical result where distribution shape differs.

## Iteration 26 — selective-HAR combined replay (2026-09-09)

**Hypothesis.** Selective HAR improves combined field ranks beyond the clustered t5 candidate
without sacrificing either 24h profile.

**Change.** Make the preregistered HAR asset set the served default, regenerate exact 200-path
predictions for every active 24h asset, and rerun both combined profiles.

**Kill threshold.** Both profiles must maintain or improve rank/reward versus Iteration 23.

**Result.** CRYPTO_24H improved from rank 176 to **154/257**, reward
3.60e-5 versus 5.30e-6. COM_EQU_24H remained **246/257**, but reward increased
from 4.98e-10 to 1.19e-9. Combined normalized scores improved from 120.90 to
108.21 (crypto) and 134.41 to 128.62 (COM/EQU).

**Verdict.** Keep selective HAR; both profiles meet the gate.

**Lesson.** One high-weight SOL forecast improvement moved crypto rank 22 places, while broad
small COM/EQU improvements increased reward without crossing the tightly clustered rank bands.

## Iteration 27 — full smoothing-window validation (2026-09-09)

**Hypothesis.** Three-day rankings are noisy and do not reflect steady-state smoothing. The
candidate's relative performance should be judged on full validator windows before further
model changes.

**Change.** Generate exact 200-path predictions for 10 scored days in both 24h profiles and five
scored days in CRYPTO_1H, then run official combined replay.

**Kill threshold.** If any full-window rank deteriorates by >30 places, revisit its model;
otherwise use these results to prioritize the final research iterations.

**Result.** Full-window 200-path ranks: CRYPTO_24H **203/257**, reward
1.93e-7; COM_EQU_24H **193/211**, reward 4.13e-10; CRYPTO_1H
**240/257**, reward 9.22e-6. The latter two denominators/rank periods differ from the
three-day run, but all are outside the reward-bearing cluster. Crypto-24h deteriorated 49 places
and breached the gate. Full-window per-asset means are recorded in the command transcript;
notable misses versus field median were TSLAX 2806 vs 2599, AAPLX 1753 vs 1647,
NVDAX 2826 vs 2685, and BTC 3917 vs 3867.

**Verdict.** Full windows supersede recent three-day ranks for selection. Retune all profiles
before final 1000-path evaluation.

**Lesson.** The latest three days materially overstated performance. Smoothing-window alignment
is not optional when softmax rewards are the target.

## Iteration 28 — full-window width correction (2026-09-09)

**Hypothesis.** Cluster widths selected on three days are too reactive over the full window;
lower adjacent widths will reduce the systematic token-equity deficit.

**Change.** On the full 10-day window, test one adjacent lower scale for ETH, SOL, HYPE, XAU,
SP500, NVDAX, TSLAX, AAPLX, WTIOIL, and SPCX; compare each to Iteration 27. Also test SOL
without HAR to isolate estimator versus width.

**Kill threshold.** Keep only >0.5% full-window improvements, then recheck the most recent
three-day regression is below 3% using prior results.

**Result.** No lower width met the 0.5% gate. Full-window candidate versus lower-width
CRPS: ETH 5165.74 vs 5152.19 (0.26% gain), SOL 6648.39 vs 6794.95,
HYPE 8747.95 vs 8794.18, XAU 1725.90 vs 1744.68, SP500 743.35 vs
751.49, NVDAX 2825.61 vs 2891.37, TSLAX 2806.13 vs 2839.33,
AAPLX 1753.06 vs 1832.59, WTIOIL 2843.32 vs 2857.99, SPCX
3747.60 vs 3878.95. Replacing HAR with std also worsened SOL to 6705.94.

**Verdict.** Discard lower widths; retain candidate settings.

**Lesson.** The full-window weakness is not simply overdispersion. Several assets strongly
penalize narrower fans.

## Iteration 29 — upper-width full-window check (2026-09-09)

**Hypothesis.** Since every lower-width move failed, several assets may require a wider t5 fan
over the full window than the recent three-day optimum suggested.

**Change.** Test the adjacent upper scale for ETH, HYPE, XAU, SP500, NVDAX, TSLAX, AAPLX,
and SPCX over the full ten-day window.

**Kill threshold.** Keep only >0.5% improvements that remain within 3% of the recent
three-day result.

**Result.** Every upper move except AAPLX failed. Candidate versus upper-width CRPS:
ETH 5165.74 vs 5263.29, HYPE 8747.95 vs 8853.96, XAU 1725.90 vs
1732.63, SP500 743.35 vs 748.40, NVDAX 2825.61 vs 2814.90 (0.38%),
TSLAX 2806.13 vs 2834.13, AAPLX 1753.06 vs 1739.70 (0.76%), SPCX
3747.60 vs 3754.00.

**Verdict.** Keep upper width only for AAPLX: its recent three-day CRPS also improved from
1742.21 to 1739.40. Discard all others.

**Lesson.** Width tuning is exhausted around well-defined local minima; the residual field gap
needs a different source of signal.

## Iteration 30 — full-window 1h memory and width (2026-09-09)

**Hypothesis.** The six-hour estimator selected on two-day windows is slightly too noisy over
the full five-day validator window. Twelve- or 24-hour memory may close the consistent 1–2%
field-median deficit.

**Change.** Full five-day exact tests on BTC and XRP comparing 6h/0.70, 6h/0.85,
12h/0.85, and 24h/0.85.

**Kill threshold.** Adopt a new common setting only if it improves both assets >0.5%; otherwise
retain six-hour/0.85.

**Result.** BTC baseline 6h/0.85 was 735.15 CRPS; 6h/0.70 = 739.22,
12h/0.85 = 732.18, 24h/0.85 = 733.26. XRP baseline 1675.25;
alternatives were 1682.55, 1663.33, and 1673.81. Twelve hours improved XRP
0.71% but BTC only 0.40%, missing the joint gate.

**Verdict.** Retain six-hour/0.85 under the preregistered rule.

**Lesson.** Longer memory helps slightly over five days, but the evidence is too small to
overturn stronger two-regime six-hour validation.

## Iteration 31 — Student-t innovations for 1h (2026-09-09)

**Hypothesis.** Moderate innovation tails that generalized at 24h can close part of the 1h
field-median deficit without changing volatility calibration.

**Change.** Full five-day BTC/XRP comparison of Gaussian versus standardized t5 at the served
six-hour/0.85 setting, followed by all assets only if both improve.

**Kill threshold.** Require >0.5% CRPS improvement on both BTC and XRP.

**Result.** Gaussian versus t5 CRPS was 735.15 versus 734.92 for BTC and 1675.25 versus
1674.89 for XRP. Both gains were below 0.5%.

**Verdict.** Discard t5 for 1h; retain Gaussian innovations.

**Lesson.** The 24h tail-shape gain does not transfer materially to the short horizon, where
the aggregation period itself already limits extreme accumulated moves.

## Iteration 32 — final 1000-path smoothing-window replay (2026-09-09)

**Hypothesis.** The selected deterministic candidate remains stable at the validator's actual
1000-path fan size across complete smoothing windows, and the AAPLX width correction improves
COM/EQU without harming the other profiles.

**Change.** Generate forecasts at every archived exact start time for all active assets: ten
days for both 24h competitions ending 2026-08-29 and five days for CRYPTO_1H ending
2026-08-28. Use 1000 paths and run the official combined backtester replays.

**Kill threshold.** Nominate only after all three interfaces validate and field-relative rank
and reward results are recorded; investigate any material deterioration from the 200-path
full-window replay.

**Result.** At 1000 paths, CRYPTO_24H ranked **170/257** (approximately 1e-6 reward
weight); COM_EQU_24H ranked **192/211**, reward 9.744e-10 versus leader 0.0028358;
CRYPTO_1H ranked **237/257**, reward 1.880e-5 versus leader 0.0041261. Mean CRPS by
asset was: 24h BTC 3911.00, ETH 5154.97, SOL 6637.14, XRP 9505.61, HYPE 8727.63,
XAU 1722.67, NVDAX 2820.96, GOOGLX 1566.23, TSLAX 2800.89, AAPLX 1735.97,
WTIOIL 2836.43, SPCX 3742.21, SP500 741.66; 1h BTC 733.50, ETH 926.29, SOL
1405.03, XRP 1671.57, HYPE 1607.03. All forecasts used archived exact starts and the
official synth-lib scorer.

**Verdict.** Keep and nominate. Results are weak relative to the mature field, but are stable
at the serving fan size, all preregistered local alternatives have been exhausted, and the
model meets both profile contracts with ample latency margin.

**Lesson.** Increasing from 200 to 1000 paths improved crypto-24h rank materially (203 to 170)
through lower ensemble sampling error, but did not erase the structural field gap. Full-window,
field-relative validation remains the honest basis for nomination.
