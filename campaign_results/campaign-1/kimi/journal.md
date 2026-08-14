# Research journal — bakeoff-2 (kimi)

Scoring facts established by reading the validator source (synth-subnet) and synth-lib:

- CRPS is empirical (`properscoring.crps_ensemble`) on **relative changes in bps**, per
  grid point, summed unweighted across points and intervals. `_abs` intervals score the
  absolute price level (normalized by the real final price). NaN realized blocks are skipped.
- 24h comps: intervals 5min (288 terms), 30min (48), 3hour (8), 24hour_abs (1) → the 5-min
  return distribution dominates (~60%+ of total weight under random-walk scaling).
- CRYPTO_1H: 1min/2min/5min/15min/30min point intervals (108 terms) + 12 "gaps" intervals
  (change from t=0 to 5..60 min) + 60min_abs. Short-horizon calibration dominates.
- Prompt score = crps − min(field crps) per prompt (invalid → p95 fill), summed over
  10-day (24h) / 5-day (1h) window, per-asset coefficients (SP500/SPYX 3.44, AAPLX 2.09,
  XAU 1.74, NVDAX 1.60, TSLAX 1.61, BTC 1.0, …, HYPE 0.48, SPCX 0.43), softmax beta
  −0.15 (24h) / −0.3 (1h). Each competition = 1/3 of emissions.
- Response format: exactly `num_simulations` paths × `time_length//time_increment + 1`
  points, ≤8 significant digits, strictly positive prices (a single 0 → prompt invalid).
- All scoring is **cross-sectional per time point**; per-path temporal coherence is not
  scored directly, but per-path increments between grid points ARE (interval changes), so
  paths need realistic increment distributions at every step.
- Offline field scores in `market_data/offline_data/` cover → 2026-07-24; rewards history
  → 2026-07-26. Backtests run fully offline via `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT`.
- Price snapshot: 2026-06-23 → 2026-07-29, ~1.86% NaN minutes on most assets (SPYX 16%,
  SPCX 6.9%, SP500/WTIOIL 0%).

## Iteration 0 — baseline (starter log-normal martingale)

- Hypothesis: establish the pipeline and a reference rank.
- Setup: `uv sync` (uv was missing; installed to /workspace/.bin). Backtest via
  `agent/predict.py`, offline data root set. BTC crypto-24h, 2026-07-20 → 2026-07-23,
  200 sims (quick loop).
- Result: mean_crps 2413.6, final_smoothed_score 94.42. Field reference (rewards history
  final round 2026-07-26): best ≈ 45.9, p10 ≈ 50.1, median ≈ 59.9, p75 ≈ 83.0 → starter
  sits ≈ rank 179/229. Field median per-prompt CRPS on BTC ≈ 2265 (July 10–24).
- Verdict: baseline works; starter is bottom-quartile. Replace it.
- Lesson: pipeline is fast (~4s scoring for 3 days); iteration loop can be dense.

## Iteration 1 — v1: EWMA vol + hour-of-day seasonality + Student-t + stratified sampling

- Hypothesis: a calibrated per-step return distribution (time-varying vol, fat tails,
  seasonality) plus noise-free quantile-grid serving should beat the starter by a wide
  margin.
- Change: replaced the starter with the model described above; added `fieldeval.py`
  (generates at the field's real prompt times, scores with the validator's
  `calculate_crps_for_miner`, ranks by mean prompt_score = crps − per-prompt best valid,
  restricted to full-coverage miners — the backtester's own rank is distorted by a
  percentile90/95 backfill-key mismatch, confirmed: uniform ~205-232/257 ranks across
  assets despite mean CRPS beating the field median on BTC).
- Result (200 sims, 2026-07-20→23, 48 prompts/asset; mean prompt_score, rank):
  - crypto-24h: BTC 27.2 (62/231), ETH 60.4 (151), SOL 37.9 (67), XRP 61.3 (78),
    HYPE 63.7 (1). Coef-weighted ps 47.0.
  - com-equ-24h: XAU 30.5 (13/229), SPYX 27.1 (206!), NVDAX 23.2 (1), GOOGLX 103.3 (1),
    TSLAX 131.3 (2), AAPLX 77.0 (195!), WTIOIL 141.3 (179!), SPCX 301.4 (109).
    Coef-weighted ps 72.9.
  - crypto-1h: BTC 87.5 (160/232), ETH 99.5 (1), SOL 133.2 (45), XRP 115.0 (186),
    HYPE 107.0 (1). Coef-weighted ps 105.9.
- Verdict: keep the architecture; clear weaknesses on SPYX/AAPLX/WTIOIL (24h) and
  BTC/XRP (1h). Data check: com-equ 5-min kurtosis 14-104 (df≈4-4.5, my prior 7 too
  thin), seasonality range 0.15-6× (my 0.5 shrinkage too flat).
- Lesson: the field's top tier is tight (BTC p05 ps ≈ 24); single-asset ranks with 48
  prompts are noisy — use coefficient-weighted prompt_score across assets as the primary
  metric. SPYX has 16% NaN minutes; gap-carrying may distort its vol estimate.

## Iteration 2 — v2: fatter tails, stronger seasonality, vol mean reversion

- Hypothesis: (a) df prior 7→4.5 with 0.6 weight on the kurtosis estimate fixes
  under-tailed innovations; (b) seasonality shrink 0.5→0.8 with median-based per-hour
  stats fixes flat intraday vol on com-equ; (c) variance mean-reverts from the EWMA
  level to the 7-day mean with 2-day half-life over the horizon, improving long-interval
  calibration when the regime is extreme.
- Change: modeling.py hyperparameters + median-based seasonality + MR variance path.
- Result: (pending)
- Result (200 sims, 2026-07-20→23, 48 prompts/asset; mean prompt_score vs v1):
  - BTC 27.2→23.5 (rank 62→4), ETH 60.4→47.6, XRP 61.3→47.1 (78→4),
    XAU 30.5→21.2 (13→1), NVDAX 23.2→13.7 (1→1), AAPLX 77.0→55.7 (195→122),
    WTIOIL 141.3→134.9 (179→163), SPYX 26.1→26.1 (206→206),
    BTC-1h 87.5→85.1 (160→137), XRP-1h 115.0→110.2 (186→66).
- Verdict: KEEP. Broad improvement; remaining gaps concentrated in WTIOIL/AAPLX/SPYX
  (24h) and BTC (1h).
- Lesson: prompt_score is relative to the per-prompt MIN over ~230 miners, so it
  amplifies small CRPS gaps: WTIOIL top-5% is only ~2.4% better in absolute CRPS.
  Calibration probe (2026-07-22 12:00 prompt): predicted 5-min |change| p50 too WIDE
  (SPYX 2.2 vs 1.3 bps realized, AAPLX 5.3 vs 3.6, WTIOIL 10.7 vs 9.3) while p90 can be
  too narrow — a single t with EWMA vol is not peaky enough at the center. SPYX NaNs are
  4 multi-DAY feed outages (not sessions); SPYX is being retired → optimize for SP500,
  treat SPYX only as a noisy proxy.

## Iteration 3 — v3: empirical innovation distribution

- Hypothesis: the per-step innovation shape (peaky center, fat shoulders, skew) is
  asset- and regime-specific; the empirical quantile grid of standardized residuals
  (EWMA × seasonality) captures it better than any single Student-t.
- Change: INNOVATIONS="empirical": z_grid = empirical quantiles of z at (i+0.5)/N
  (requires ≥500 residuals, else Student-t fallback).
- Result: (pending)
- Result: MIXED, mostly worse (200 sims, same window). v3 raw empirical: WTIOIL
  134.9→85.5 (rank 163→19!) but XAU 21.2→61.3, BTC 23.5→38.8, ETH 47.6→62.8,
  XRP 47.1→68.6, BTC-1h 85.1→95.1. v3b (grid renormalized to unit std): BTC 21.1
  (rank 1), WTIOIL 80.0, but XAU 52.3, AAPLX 56.5. v3c (winsorized at t(df) 0.9995):
  no change vs v3b.
- Verdict: DISCARD as default. The empirical grid's scale/shape instability (a few
  extreme context observations) hurts more than the shape realism helps, except on
  WTIOIL.
- Lesson: the served grid's extremes are served at EVERY prompt; one crazy context
  observation becomes a permanent CRPS tax. Parametric tails are safer.

## Iteration 4 — v4: robust scale formulation (EWMA of |r|, no variance)

- Hypothesis: squared-return EWMA is outlier-dominated; an E|r|-based scale with
  empirical z = r/scale_pred should standardize better.
- Result: WORSE everywhere (XAU 71.3, BTC 51.8, WTIOIL 117.2, AAPLX 92.2, ETH 85.8,
  XRP 91.4, BTC-1h 102.2).
- Verdict: DISCARD. |r|-EWMA tracks the center, leaving z with inflated scale and
  unstable shape.
- Lesson: squared-EWMA + parametric t is a stronger baseline than it looked.

## Iteration 5 — v5: two-component normal-mixture innovations (EM)

- Hypothesis: a calm/turbulent normal mixture gives the peaky center + fat shoulders
  with smooth parametric tails.
- Result: WORSE (XAU 43.8, BTC 51.8, AAPLX 83.5, ETH 93.1) and numerically dangerous:
  EM diverged on WTIOIL (s2 exploded → overflow → astronomical CRPS). 
- Verdict: DISCARD. v2 (Student-t, df prior 4.5, shrink 0.6) restored as champion.
- Lesson: on ~2000 fat-tailed residuals, fancier shape fitting is estimation-noise
  dominated. Remaining gap to the top tier is more likely in the VOL FORECAST than in
  the innovation shape → tune EWMA half-life, mean reversion, seasonality next.

## Iteration 6 — hyperparameter sweeps + stochastic vol test (v2 base)

- 1000 vs 200 sims (BTC/XAU/WTIOIL): ps 23.5→21.9 / 21.2→21.3 / 134.9→134.4. Stratified
  quantile serving already removes most sampling noise → 200-sim iteration metric is
  nearly unbiased. KEEP serving 1000 (free ~1 ps on BTC).
- EWMA half-life {0.5, 1.0, 2.0}d: BTC 28.3/23.5/21.8, XAU 19.6/21.2/22.5,
  WTIOIL 143/135/159 → no consistent winner; KEEP 1.0d.
- Calibration probe (8 prompts × 4 horizons, realized |Δ| / served median |Δ|):
  BTC 1.16/0.96/0.80/0.75 (5min→24h; long horizons too WIDE), XAU 1.49/1.46/1.24/2.22,
  WTIOIL 1.41/1.24/1.21/2.27 (too narrow everywhere — but mean-ratio is outlier
  dominated).
- Per-path lognormal vol multiplier (VOL_OF_VOL ∈ {0.2, 0.35}): hurts monotonically
  (XAU 21.2→23.8→31.0, WTIOIL 134.9→138.9→149.3). DISCARD (code kept, default 0).
  Widening is not how the top tier wins on WTIOIL.
- Lesson: knob-level changes move ps by ±2-5; the WTIOIL/AAPLX gap (~2.4% CRPS) is
  shape/level at the 5-min interval, not width.

## Iteration 7 — full-field picture + center-calibration test

- Full window1 eval (v2, 200 sims, coef-weighted ps): crypto-24h ≈ 38 (BTC 25.5,
  ETH 48.1, SOL 27.4, XRP 47.7, HYPE 48.2), com-equ ≈ 68 (XAU 22.0, SPYX 26.3,
  NVDAX 14.1, GOOGLX 85.1, TSLAX 132.6, AAPLX 56.7, WTIOIL 133.1, SPCX 165.5),
  crypto-1h ≈ 103 (BTC 85.6, ETH 99.2, SOL 130.9, XRP 110.9, HYPE 101.7).
  Rank-1 on HYPE/XAU/NVDAX/GOOGLX (24h) and ETH/HYPE (1h); better than field-p05 on
  TSLAX, GOOGLX, SOL-1h. Laggards vs p05: AAPLX (56.7 vs 21.3, coef 2.09 — biggest
  weighted gap), WTIOIL (133 vs 33, coef 0.85), ETH-24h, SPYX (retiring — ignore).
- Diagnostic: robust (q84−q16)/2 std of standardized residuals is 0.65–0.98 across
  assets — the served center is systematically too wide (variance forecast is
  tail-inflated). Tested scaling the served grid by clip(robust_std, 0.75, 1.2).
- Result: helps only AAPLX (56.7→43.4, persistently narrow center 0.65–0.71) and ETH
  marginally; XAU 22→57 and WTIOIL 133→171 (regime shift: in-sample narrow center does
  not generalize); 1h uniformly worse. DISCARD as global default.
- Lesson: in-sample calibration corrections are regime-sensitive; only microstructural
  (persistent) peakiness like AAPLX's justifies them. Candidate: per-asset dead-zone
  only where the narrow center is stable across sub-windows.

## Iteration 8 — second window + regime diagnosis

- Window2 (2026-07-13→16, high-vol regime after a quiet weekend): 24h coef-weighted ps
  74.6, weighted rank ~155 — much worse than W1 (~56/~75). 1h on W2: EXCELLENT
  (weighted rank 7.3: BTC/SOL/HYPE rank 1, ETH 3, XRP 36). SP500 slice (11 prompts,
  07-23→24): ps 36.4, rank 78/138, field p05 8.4 — behind the top tier on the
  highest-coefficient asset.
- W2 24h probes: MR off/0.5/2 → 81.3/75.7/78.7 (MR not the issue); EWMA 0.25/0.5/1.0d
  → 100.8/88.2/78.7 (shorter is WORSE — noise overshoot); SIGMA_MULT 1.1/1.2 → worse in
  BOTH windows (W1 33.2→41.9, W2 78.7→87.9 at 1.1) despite the calibration probe
  showing 20-30% under-dispersion at 5-min: the probe's mean-ratio is dominated by a
  few big moves; widening loses more on typical prompts than it wins on big ones.
- Drift check: corr(24h, next 24h) ∈ [-0.14, +0.14], inconsistent sign across assets →
  no usable drift signal. Interval shares in W2 normal (60/23/10/6).
- Conclusion: the ~2-4% gap to the top tier in high-vol regimes is distributional
  SHAPE, not level. Testing in-simulation GARCH vol clustering (fatten multi-step tails
  without touching single-step cross-sections).

## Iteration 9 — GARCH-in-sim vol clustering (KEEP) + ML shape model (FAIL)

- GARCH-in-sim: per-path variance multiplier h_t follows a GARCH(1,1)-type recursion
  driven by the path's own innovations (alpha=0.15 per 5-min step, persistence 0.94).
  Fattens multi-step tails without touching single-step cross-sections.
  Result (24h probes): W1 33.2→31.2, W2 78.7→75.3; alpha=0.15 ≈ alpha=0.1; neutral on
  1h. KEEP (default CAMPAIGN_GARCH_ALPHA=0.15).
- ML shape model: quantile-MLP (14 features: EWMA vol term structure, |z| stats, hour,
  asset class + log h, hour(t+h)) predicting per-step standardized-return quantiles for
  h ∈ {1..288}, trained on the snapshot before 07-13 (88k samples, pinball 0.183).
  Result: DISASTER both as full distribution (W1 97.8, W2 154.1) and as shape-only over
  the parametric sigma path (W1 63.5, W2 128.7) vs parametric 31.2/75.3.
  Diagnosis: regression dilution — with ~20 days of training data the network's
  conditional quantiles shrink toward the unconditional ones (served q95 ≈ 11 bps vs
  parametric ≈ 19 bps on a BTC prompt); the tail quantiles are the most diluted.
  Verdict: DISCARD. A parametric t with an exact EWMA level beats a learned conditional
  distribution at this data size. (Kept in mlshape.py for the report; not served.)
- Lesson: at ~10^4-10^5 samples of fat-tailed returns, conditional-quantile ML loses
  to a 5-parameter parametric model. The field's top tier is most likely parametric too.

## Iteration 10 — seasonality shrink sweep (KEEP 1.0) + df-prior sweep (keep 4.5)

- Hypothesis: the hour-of-day seasonality multiplier (median-based, shrunk toward 1
  by SEASON_SHRINK) is under-used at 0.8; full strength may calibrate the 5-min
  interval better, especially on equity perps with strong intraday vol patterns.
- Sweep on W1 24h probes (BTC,XAU,AAPLX,NVDAX,XRP, coef-weighted ps):
  0.6 → 38.7, 0.8 → 31.2, **1.0 → 26.97** (weighted rank 20.8; AAPLX 40.0, NVDAX 7.6).
- W2 confirmation (high-vol window): 0.8 → 75.3, **1.0 → 70.5** (BTC 44.2 rank 35,
  XAU 41.4, AAPLX 105.4, NVDAX 78.0, XRP 55.7). Better in BOTH regimes.
- Verdict: KEEP — default CAMPAIGN_SEASON_SHRINK now 1.0 in modeling.py.
- DF_PRIOR sweep (Student-t df = 0.6×kurtosis-est + 0.4×prior): 5.5 → 31.5,
  4.0 → 30.9 vs 4.5 → 31.2 on W1 probes — within noise. Keep 4.5.
- Lesson: unshrunk seasonality wins because the shrinkage was protecting against
  estimation noise that the median + circular smoothing already handle; the equity
  perps (AAPLX, NVDAX) have the strongest intraday vol U-shape and gained the most.

## Iteration 11 — final champion validation (1000 sims, both windows, both formats)

Final champion config (v2 + GARCH-in-sim α=0.15 + SEASON_SHRINK=1.0), evaluated with
the validator's exact CRPS at the field's real prompt times, 48 prompts/asset/window,
1000 sims (validator setting), ranked vs full-coverage field miners:

- **W1 (2026-07-20→23, calm) 24h, 13 scored assets**: coef-weighted ps **49.7**,
  weighted rank **50.9**. Rank 1/231 on BTC (20.7), SOL (16.2), HYPE (47.7),
  XAU (25.0), NVDAX (4.2), GOOGLX (76.0); top-15 on TSLAX, SPCX. Laggards:
  WTIOIL (132.9, rank 163), SPYX (retiring asset, ignore).
- **W2 (2026-07-13→16, high-vol) 24h**: coef-weighted ps **65.0**, weighted rank
  **121.9** (was ~155 before the seasonality fix). Rank 2 on TSLAX, top-10 on ETH,
  HYPE; laggards AAPLX (102.7, rank 206 — coef 2.09, the biggest weighted gap),
  NVDAX, GOOGLX, XAU.
- **W1 crypto-1h**: coef-weighted ps **101.6**, weighted rank **50.8**; ETH and HYPE
  rank 1/232.
- **W2 crypto-1h**: coef-weighted ps **107.4**, weighted rank **2.0** — rank 1/234 on
  BTC, ETH, SOL, HYPE simultaneously; XRP rank 7. The 1h format is the champion's
  strongest competition in stressed regimes.
- **SP500 slice** (11 prompts, 2026-07-23→24 — the only field-scored SP500 window in
  the snapshot): ps **32.5**, rank **77/138** (field p05 = 8.4). Behind the top tier
  on the highest-coefficient asset; flagged as the main limitation.
- Interface smoke test: 1000 paths honored, 289/61 points per path, paths start at
  the last context price, all finite, values round-trip the validator's %.7e rule,
  ~0.3 s per prompt — passes on both formats.
- Official synth-lib backtest (`predict.py`, 1000 sims) run on BTC/XAU/AAPLX/SP500
  for the record; numbers in report.md.

Verdict: SHIP as champion. Lesson: the same parametric core wins both formats; the
remaining gap to the field's top tier is concentrated in com-equ 24h during high-vol
regimes (AAPLX above all) and on SP500 — distributional shape at the 5-min interval,
not width, drift, or sampling.

### Official synth-lib backtest log (iteration 11 addendum)

`agent/predict.py --num-simulations 1000`, W1 window (eval-end 2026-07-23, 3 days)
except SP500 (eval-end 2026-07-25, 2 days — its scored-prompts store starts ~07-24;
eval-end 07-24 raises NoScoresAvailable):

| Asset | scored prompts | mean CRPS (official) | backtester rank (distorted, see suggestions.md) |
|---|---|---|---|
| BTC | 48 | 2373.2 | 200/257 |
| XAU | 49 | 1501.9 | 175/257 |
| AAPLX | 48 | 2020.8 | 150/257 |
| SP500 | 11 | 1106.6 | 107/257 |

Cross-check: official mean CRPS matches fieldeval's independent computation on the
same prompts (SP500: 1106.6 vs 1103.8; BTC 2373 vs 2267 on a slightly different
prompt cadence) — the scoring path is consistent; only the backtester's RANK output
is unreliable (percentile backfill mismatch + inclusion of partial-coverage miners).

## Iteration 12 — AAPLX-only center calibration retest (DISCARD) + SP500 diagnosis

- Hypothesis: with seasonality now at full strength, per-asset center calibration
  (robust-std rescaling, clip [0.75, 1.2]) might still help AAPLX, whose narrow
  center was persistent in earlier sub-windows.
- Result (AAPLX, 1000 sims, 48 prompts): W1 38.1 vs 35.3 without (WORSE);
  W2 126.1 vs 102.7 — WORSE. The season-1.0 fix already captured the calibration
  benefit, and in the high-vol window the in-sample narrow center does not
  generalize. Verdict: DISCARD definitively; no per-asset overrides in the champion.
- SP500 diagnosis: 5-min interval = 63% of total CRPS (normal); on the single
  field-matched prompt in the diagnose window our CRPS is ~3% BELOW the field best,
  but the 11-prompt fieldeval slice says ps 32.5 vs field p05 8.4 — small-sample
  noise dominates an 11-prompt slice; SP500 conclusions stay tentative.
- Lesson: retest rejected ideas after every architecture change — interactions
  between knobs are real, and yesterday's win can become today's regression.

## Iteration 13 — spike-activated fast EWMA (DISCARD) + GARCH alpha 0.22 (keep 0.15)

- Skewness diagnostic (30d, both increments): signed skew is unstable across
  increments and driven by single outliers (SOL −36 at 5-min = one bad tick;
  WTIOIL +7.3 at 5-min vs −1.3 at 1-min). No persistent cross-increment sign
  except weakly XAU (+1.6/+2.8). Per-prompt skew estimates on ~2k fat-tailed
  samples are noise-dominated. Not actionable.
- Hypothesis: in high-vol regime onsets the EWMA(1d) level lags; a max(slow,
  k×fast-EWMA) rule could adapt at spike onsets without permanently widening
  calm fans. Implemented env-guarded (CAMPAIGN_FAST_EWMA / CAMPAIGN_SPIKE_K,
  default off).
- Result (W2 probes, 200 sims): k=1.0 → 84.3, k=0.7 → 73.3 vs champion 70.5.
  WORSE at both margins — the max-bias costs more than spike adaptation wins,
  consistent with the earlier finding that shorter EWMA half-lives lose to noise
  overshoot even in W2. Verdict: DISCARD (knob left in modeling.py, default off).
- GARCH alpha 0.22 (W2 probes): 71.6 vs 70.5 at 0.15 — slightly worse. Keep 0.15.
- Lesson: the champion's knob space is now well explored in both regimes; every
  remaining gap points to distributional shape at the 5-min interval in com-equ
  high-vol, which none of the level/width/tail knobs reach.

## Iteration 14 — WTIOIL empirical-grid retest (DISCARD) + newest-window sanity check

- WTIOIL empirical innovation grid retest (the one asset where the empirical grid
  won back in iteration ~5, retested after the architecture changes):
  W1 91.3 (rank 19 — vs champion 132.9, rank 163) but W2 164.0 vs champion 135.2.
  Regime-fragile: the W1 win does not survive the high-vol window.
  Verdict: DISCARD; the no-per-asset-overrides decision stands.
- Newest-window sanity check (2026-07-26→28, no field scores exist there — CRPS
  levels only, via diagnose.py on generated predictions): BTC 2434.5, AAPLX 1751.2,
  interval shares normal (5-min ≈ 61-64%). In line with W1 (2267/1937) and W2
  (2654/2340) levels — the champion behaves sanely on the most recent regime in
  the snapshot, the best available proxy for the future evaluation window.
- Lesson: with the knob space exhausted, the remaining value is in verification:
  the champion is stable across three regimes and both formats.

## Iteration 15 — robust mean-reversion target (KEEP)

- Hypothesis: the MR target var_long = mean(r^2) over the 7-day context is
  tail-inflated; mean reversion then pulls long-horizon variance levels toward a
  value higher than the typical day's. A median of per-day variances is a robust
  target that should lower long-horizon levels without touching the EWMA level.
- Result (probes, 200 sims): W1 26.4 vs 27.0 (rank 20.0 vs 20.8); W2 70.1 vs 70.5.
  Small but better in BOTH regimes — the pre-registered adoption criterion.
- Verdict: KEEP — CAMPAIGN_MR_TARGET default now "robust".
- Full-field confirmation at 1000 sims (both windows, both formats): W1 24h
  49.4/rank 49.5 (was 49.7/50.9), W2 24h 65.1/121.8, W1 1h 101.7/49.3,
  W2 1h 107.5/rank 2.0 (rank 1 on BTC/ETH/SOL/HYPE), SP500 slice 32.0/77.
  Equal or better everywhere — the robust-MR champion is the shipped one.
- Lesson: robust targets matter wherever a mean of squared fat-tailed variables
  feeds a served level; same reason the seasonality uses medians.

## Iteration 16 — vol-conditional df (no signal) + leverage effect (absent)

- Vol-conditional kurtosis (30d, 5-min, z split by predicted-variance terciles):
  implied df is flat across terciles (4.0-6.3, no consistent direction across
  assets). High-vol periods are NOT systematically more fat-tailed once the level
  is divided out — the pooled per-prompt df estimate (~4.3-5) is the right one.
  Not actionable.
- Leverage effect (corr(r_t, r^2_{t+1}) at 5-min): ≈ 0 for all 13 assets
  (±0.03; GOOGLX +0.09 has the wrong sign). These Hyperliquid perps show no
  equity-style asymmetric vol response, so GJR-GARCH asymmetry in the simulation
  is unjustified. The |r| autocorrelation (0.10-0.34) is plain vol clustering,
  already captured by GARCH-in-sim.
- Lesson: two more structural doors closed cheaply with data checks before any
  backtest. The champion's structure space (level / seasonality / MR / shape /
  tails / serving / drift / skew / asymmetry) is now exhaustively probed on both
  regimes; further gains would need information outside the per-asset 7-day
  context (cross-asset signals, longer history) that the serving contract
  forbids.

## Iteration 17 — serving robustness audit (PASS)

- Edge-case test of simulate() against degraded contexts a future outage could
  produce: full 7d, 100-point, 10-point, single-price, 1-day NaN run in the
  middle, 6h NaN tail, empty context, constant price — at both increments.
- Result: all cases return finite paths with the correct point count
  (289/61); paths start at the last VALID price (NaN-tail case starts at the
  stale-but-valid close, per contract); empty context falls back to the
  documented flat fan; constant price serves a flat fan at that price.
- Lesson: the MIN_RETURNS / fallback-sigma paths work as designed; no serving
  crashes are expected on the future window.

## Iteration 18 — 1h-specific GARCH alpha (keep 0.15)

- Last untested cell: GARCH alpha tuned for the 1h format in the calm window
  (W1 1h is mid-pack at rank ~49 while W2 1h is rank 2).
- Result (W1 1h, 200 sims): alpha 0.3 → 102.8 vs 101.6 at 0.15. Slightly worse.
  Verdict: keep 0.15 globally. The W1/W2 1h difference is field tightness in the
  calm regime, not a tail-shape deficiency the alpha knob can reach.

## Iteration 19 — CRPS-optimal quantization check (no gain available)

- The last untried idea from the report's "with more budget" list: replace the
  served midpoint-quantile grid with atoms optimized directly on the CRPS
  objective (energy-distance optimal quantization).
- Numerical check (agent/artifacts/quant_check.py): exact CRPS of the served
  midpoint-quantile ensemble of a unit-variance Student-t(4.5) at n=1000 vs
  200 steps of gradient descent on the CRPS objective — the optimizer does not
  move (relative gap 0.0000%). The midpoint quantile grid IS the CRPS-optimal
  deterministic ensemble at this n, consistent with the known asymptotic
  optimality of quantile ensembles for CRPS.
- Verdict: nothing to implement; the serving scheme is provably at the optimum.
  Every idea on the research list is now closed with evidence.

## Iteration 20 — anatomy of the AAPLX W2 gap (selection effect identified)

- Per-prompt gap-to-best distribution (AAPLX W2, 55 matched prompts): NOT uniform.
  On 07-13 we BEAT the per-prompt field best on several prompts (gaps to −118);
  all worst prompts sit on the high-vol day 07-14 late → 07-15, spread across all
  UTC hours (no hour-of-day pattern).
- On the worst prompt (07-15 14:00, gap +586): served sigma mean 9.9bps vs realized
  rms 11.2bps — only ~12% under-dispersed (first 3h: 17.9 vs 20.7, ~13% under).
  A 12% width difference cannot explain a ~20-25% CRPS gap.
- The realized 24h moves on the worst prompts: +3.1%, +3.7%, +3.9% — strong
  trends. The prompts we win had moves of ±0.4-0.9%. On trending windows the
  per-prompt field BEST (min over ~240 miners) is captured by fans centered on
  the move — lucky drift or momentum-serving miners. A centered martingale cannot
  match that selection effect, and drift is not predictable (autocorr ≈ 0,
  iteration 8): chasing it would lose on the majority of prompts.
- Verdict: the AAPLX-W2 question is answered. The beatable residual is the ~12%
  early-horizon under-dispersion, but every targeted widening mechanism tested
  (sigma mult, spike-EWMA, VOV) loses more elsewhere than it wins here.
- Lesson (publishable): with min-over-field benchmarks, part of the apparent gap
  to the top tier on trending days is a selection effect in the metric itself,
  not a model deficiency.

## Iteration 21 — day-of-week vol correction (DISCARD)

- Data check: strong, consistent DOW effect — weekend median r^2 is 0.34-0.68x
  (crypto) and 0.03-0.47x (com-equ) of the weekly median. Unlike hour-of-day, the
  24h EWMA window does not average over days-of-week, so the level lags the
  Fri->Sat drop: a real predictable inefficiency for weekend prompts.
- Implemented a forward/backward ratio correction (var_path *= profile[dow(step)]
  / EWMA-weighted backward DOW mix, clip [0.2, 3.0]), profile estimated from the
  7-day context (1 obs per DOW).
- Result (probes, 200 sims): W1 41.2 vs 26.4, W2 87.9 vs 70.1 — MUCH worse in
  both windows. With one observation per DOW the profile is too noisy: mid-week
  ratio adjustments of tens of percent are mostly estimation error, and both
  validation windows are Mon-Thu where the correction should have been ~neutral.
- Verdict: DISCARD (knob CAMPAIGN_DOW left in modeling.py, default off). The
  weekend effect is real but unreachable with a 7-day context: any estimate
  strong enough to help weekends is too noisy to apply mid-week. A committed
  static DOW table from the full snapshot could work around this, but baking a
  static calendar table into a model whose other components are all adaptive —
  for a benefit that only materializes on 2 of 7 days — is a bad trade.
- Lesson: check estimability against context size BEFORE implementing: with
  7 days of context, any weekly-frequency effect has exactly one observation
  per cell.
