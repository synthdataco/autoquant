# Journal — campaign-7 (fable-5-2)

All CRPS numbers come from the official scorer (`synth_lib.backtester` via `agent/predict.py`,
1000 paths unless stated). "field best" = lowest mean CRPS of any single miner on the same prompts
(miners with >= 98 % of the prompts), "field median" = median miner. AGG = asset-coefficient
weighted mean CRPS per competition (the weights the validator's aggregation uses).

## Iteration 0 — setup, scoring anatomy, starter baseline (2026-11-11 13:13 → 13:30 UTC)

**What I read.** `synth.validator.crps_calculation`, `competition_config`, `reward`, `moving_average`,
`response_validation_v2`; `synth_lib` backtester (`orchestration`, `scoring`, `preparation`), the
verdict runner (`benchmark/verdict/run_verdict.py`, `evaluate.py`), `generate_predictions.py`, the
nomination probe and the live serving wrapper (`serving/serve.py`, `champion_miner.py`).

**Scoring anatomy (expected share of a calibrated Gaussian fan's CRPS, by arithmetic).**
- 24h configs: 288 five-minute returns (61 %), 48 thirty-minute (25 %), 8 three-hour (10 %),
  one absolute 24h price (3.6 %). Short-horizon marginals dominate.
- crypto-1h: 1-min returns (26 %), 2-min (18 %), 5-min (12 %), 15-min (7 %), 30-min (5 %),
  60-min abs (3 %), twelve "gap" returns 0→5k min (29 %), plus a volatility-CRPS term on the
  realized std of 5/15/60-min blocks (about 1 % of the total, but miners differ on it by ~0.4 %).
- The context ends 3 bars (24h) / 2 bars (1h) before the first scored price; only the `_abs`
  interval sees the anchor, so the cost is negligible.
- The probe calls simulate() with a constant-price context (zero volatility) for asset "BTC" in
  both formats: the model must not produce NaN there.
- Verdict stores paths as float32; live wrapper rounds to 8 significant digits.

**Field.** 256 miners, extremely tight: on the last 30 days the best miner beats the median miner
by only 0.3–0.5 % of CRPS on each asset. Many uids post bit-identical CRPS (same operator / same model).

**Evaluator.** `agent/predict.py` gained a parallel mode (`--assets ... --jobs N`): generation through
`synth_lib.benchmark.generate_predictions.generate()` (the verdict's own generator, float32 day files),
scoring through `score_crps()` (validator CRPS). It joins the field's per-prompt CRPS from the bundle
and prints, per asset, our mean CRPS vs the best / median miner on the same prompts. 8 410 24h prompts
score in ~1 min on 12 cores. Cross-check: plain single-asset `backtest()` gave BTC 4070.77 on the same
48 prompts where the parallel mode gave 4069.92 (different random draws).

**Starter baseline (harness log-normal martingale, constant 7-day sigma), validator prompts scored
2026-10-08 → 2026-11-07, 1000 paths:**

| config | AGG CRPS | vs field best | vs field median |
|---|---|---|---|
| CRYPTO_24H | 3785.58 | +3.40 % | +3.02 % |
| COM_EQU_24H | 1988.95 | +10.84 % | +9.74 % |
| CRYPTO_1H | 833.37 | +5.67 % | +5.01 % |

Per asset (starter → field best): BTC 2598.35 → 2493.88, ETH 3640.74 → 3504.70, SOL 4341.45 → 4211.43,
XRP 4079.40 → 3948.35, HYPE 5400.21 → 5266.64; XAU 1368.47 → 1256.50, NVDAX 2501.94 → 2318.57,
GOOGLX 2380.79 → 2103.61, TSLAX 3191.51 → 2831.90, AAPLX 1748.24 → 1502.91, WTIOIL 3114.77 → 2880.75,
SPCX 4201.96 → 3760.78, SP500 906.43 → 853.68; 1h: BTC 573.28 → 538.03, ETH 797.30 → 749.21,
SOL 960.41 → 914.62, XRP 902.07 → 854.81, HYPE 1181.44 → 1126.39.

**Lesson.** The whole distance from a naive constant-vol Gaussian to the best miner is 3 % (crypto 24h)
to 11 % (equities, where intraday seasonality is large). Every component I add is worth tenths of a
percent, so experiments must be paired on identical prompts and validated on several windows.

**EDA so far (last 120 days / DST regimes).**
- Vol seasonality follows the NEW YORK clock for every asset checked, crypto included (BTC/ETH hourly
  sigma peaks at 09–10 NY in both DST regimes: 1.5–1.7x; trough 0.7x). Equities: 2.4x in the open hour,
  ~0.5x overnight. A smaller UTC-fixed bump exists at 00–02 UTC for crypto.
- 5-min/1-min variance ratios ~1.0–1.1; 30-min vs 5-min 0.875 (SOL, HYPE) … 1.06 (TSLAX): mild
  mean reversion for some crypto. 5-min zero-return mass: XAU 14.7 %, SPCX 7.4 %, AAPLX 6.8 %.

## Iteration 1 — v1: seasonal profile + multi-scale level + log-normal SV paths, fitted on CRPS (13:30 → 14:15)

**Hypothesis.** Most of the starter's gap is (a) intraday/weekly seasonality, (b) a volatility level that
reacts faster than a 7-day std, (c) fat tails / uncertain future volatility. A small parametric
simulator whose parameters are fitted *directly on the scoring rule* should reach the field.

**Change.**
- `research/fit_profiles.py`: per-asset variance profile by minute of the New York week (10 080 values:
  pooled weekdays at 1-min resolution with jump-preserving smoothing and a minute-of-hour factor, slow
  per-weekday ratios, own Sat/Sun shapes; NYSE holidays read as Saturdays), fitted outside scheduled-event
  windows, plus event multipliers by minutes-since-event (CPI, NFP, PPI, retail, GDP, PCE, FOMC, EIA, earnings),
  pooled per asset group, from the calendar only.
- `agent/core.py`: context features = log of deseasonalised EWMA variance estimators (squared, absolute and
  Parkinson families x 8 half-lives 5 min … 7 d, + context mean); level forecast = softmax-weighted
  geometric pooling with horizon-dependent weights; paths = Gaussian shocks x exp(log-vol), log-vol =
  permanent factor A(h) + OU + white noise; MA(1) on shocks.
- `research/train.py`: differentiable replay of the validator's CRPS (all intervals, gaps, vol blocks) on
  GPU-simulated paths; Adam on ~60 parameters per (asset, format).
- EDA that shaped it: crypto 1-min variance has a strong minute-of-hour sawtooth (x1.4 at :00 and :30,
  x0.62 at :59); the US cash open multiplies 1-min variance x5 (BTC), x11 (SP500), x43 (NVDAX); CPI
  minute x100+.

**Protocol.** Fold A = everything (profile, events table, parameters) fitted on data before 2026-07-24
(parameters: prompts ending before 2026-07-01), scored on the validator's prompts of 2026-07-30 → 2026-11-07
(100 days, fully out of sample, profile up to 3.5 months stale).

**Result (fold A, 1000 paths, official scorer; rank = paired mean CRPS among miners with >=98 % coverage).**

| config | AGG CRPS | vs median miner | per-asset rank |
|---|---|---|---|
| CRYPTO_24H | 3665.11 | -0.55 % | BTC 13/213, ETH 1/213, SOL 1/228, XRP 9/212, HYPE 1/230 |
| COM_EQU_24H | 1686.69 | -0.36 % | NVDAX 1/230, TSLAX 1/228, GOOGLX 42/229, SP500 71/178, AAPLX 93/220, XAU 108/222, SPCX 113/196, WTIOIL 185/227 |
| CRYPTO_1H | 780.30 | -0.46 % | BTC 1/232, ETH 1/218, SOL 1/220, XRP 1/219, HYPE 1/232 |

Per asset CRPS: BTC 2446.74, ETH 3265.86, SOL 4072.89, XRP 4516.45, HYPE 5256.48, XAU 1383.24, NVDAX 2051.37,
GOOGLX 1823.73, TSLAX 2387.82, AAPLX 1531.60, WTIOIL 3258.36, SPCX 3536.18, SP500 751.44;
1h: BTC 516.10, ETH 696.88, SOL 875.76, XRP 962.96, HYPE 1116.61.

**Verdict.** Keep. Already first in the field out of sample on crypto-1h and on 5 of 13 24h assets.

**Explained weakness (before touching anything).** Calibration table of |r|/sqrt(v) by session x horizon on
the validation period: weekend steps forecast from far away are ~40 % too wide (ratio 0.63–0.72 of the
average at 12–24 h) for SP500/AAPLX/XAU/WTI, core-session steps 10–30 % too narrow. Cause: the profile was
fitted on VARIANCE, but weekend returns of the thin perps are zero-inflated and jumpy, so their RMS
overstates the scale CRPS wants (E|r|-like); short horizons hide it because the absolute-return EWMAs adapt.

**Lesson.** A variance profile is the wrong seasonal for CRPS when the return shape changes with the
session. Fit the seasonal on the scale the score rewards, and let the shape (tail weight) depend on the session.

## Iteration 2 — v2: CRPS-scale seasonal, session-dependent tail weight, stratified sampling (approx. 14:15 → 15:05 UTC)

**Hypothesis.** (a) The seasonal profile should be fitted on (E|r|)^2, the scale CRPS rewards, not on E[r^2];
(b) the tail weight (white log-vol noise sd) should depend on the session and on the context's share of
zero-return minutes, and each session needs a level offset with a horizon slope; (c) i.i.d. sampling of the
1000 paths wastes ~0.1 % of CRPS (expected excess of an N-sample ensemble = E|X-X'|/2N = 1/N of CRPS).

**Change.** abs-scale profile (weekend shapes from 5-min means, floor 0.01); per-session `lam_raw`, `dlt`,
`dlt1`, liquidity slopes `gz`, `gl` in the trainer and in `core`; Latin-hypercube shocks (every column of
every Gaussian driver is a permutation of the n stratum midpoints).

**Result (fold A, same 100-day window, official scorer, 1000 paths).**

| config | v1 AGG | v2 AGG | change | paired vs best miner / median miner | ranks |
|---|---|---|---|---|---|
| CRYPTO_24H | 3665.11 | 3661.99 | -0.09 % | -0.16 % / -0.59 % | all five 1st |
| COM_EQU_24H | 1686.69 | 1676.97 | -0.58 % | -0.26 % / -0.92 % | NVDAX, GOOGLX, TSLAX, AAPLX, SP500 1st; SPCX 69/196, WTIOIL 104/227, XAU 109/222 |
| CRYPTO_1H | 780.30 | 779.55 | -0.10 % | -0.26 % / -0.52 % | all five 1st |

Per asset v2: BTC 2443.49, ETH 3263.92, SOL 4068.22, XRP 4512.78, HYPE 5254.51, XAU 1384.69, NVDAX 2044.85,
GOOGLX 1811.58, TSLAX 2380.12, AAPLX 1511.94 (v1 1531.60), WTIOIL 3232.88, SPCX 3524.54, SP500 744.98 (v1 751.44);
1h: BTC 515.54, ETH 696.28, SOL 874.79, XRP 962.02, HYPE 1115.78.

Sampling alone, paired on identical prompts and parameters (v1 params): BTC 24h 40 d: iid 2461.54, antithetic
2460.53, LHS 2459.93 (-0.065 %); BTC 1h 14 d: iid 578.77, antithetic 578.58, LHS 578.48 (-0.05 %).
Hierarchical quantile matching on top of LHS (served ensemble moved rank-by-rank onto the quantiles of an
8x larger reference sample, block sizes fine -> coarse): BTC 24h 2457.10 -> 2456.53, SP500 822.21 -> 822.02,
BTC 1h 578.08 -> 577.89 (-0.02 … -0.03 %), for +0.45 s per 24h prompt. Kept for the champion, off in research runs.

**Verdict.** Keep all three. First on 10 of 13 24h assets and on all 1h assets, out of sample.

**Diagnostics written.** `research/diag_scale.py` prices a width residual per cell: for XAU on the fold-A
window a perfect per-hour-of-week rescale is worth 1.2 % of the step CRPS; the model is 25–50 % too narrow on
weekday steps forecast 4–24 h ahead (short horizons are fine) — a horizon bias learned on the thin Dec–Jun
history. Reopening minutes are huge (first minute after Sunday 18:00 NY: WTI 110 bps vs 5 bps before; XAU
22 vs 1.1; NVDAX Sunday 20:00: 30 vs 2.4) and the 5-min weekend buckets spread that minute over five.

**Caveat.** On the last 14 days alone the fold-A model is only 77/231 on BTC 1h: parameters fitted before
July with a profile 3.5 months stale. Fold B (everything fitted before 2026-10-01, scored on October) is the
realistic proxy for the champion's conditions and is the next run.

## Iteration 3 — v3: session-aware level, power-mean pooling, two clocks, tick grid; fold B (approx. 15:05 → 17:10 UTC)

**Hypotheses.** (a) A quiet night says little about the next cash session: the level of a step should lean
on EWMAs taken over ITS OWN session's minutes, and fast global EWMAs should be discounted when the target
step sits in another session than the context's end. (b) Geometric pooling of variance estimators
under-forecasts from quiet states; a learnable power mean fixes it. (c) Part of the seasonality is UTC-fixed
(Asia), so a New-York-clock profile fitted mostly on summer time is an hour off in winter (the scored
window is winter time). (d) Realised prices live on a tick grid; the fan should too. (e) Signed/absolute
recent-return z-scores and a jump indicator add to the EWMAs.

**Change.** Two session-level features per target step + cross-session discount `gamma`; power-mean
exponent `p`; UTC daily profile backfitted against the NY weekly profile (12 iterations, skipped when an
asset has <15 % winter data); minute shape kept inside weekend re-opening buckets; 2-min half-life; four
"extras" with horizon-linear coefficients; tick snapping; robustness guards (cap of a minute's weight at
400x the context mean; sparse-feed fallback; bare-series contexts).

**Protocol.** Fold B: profile, event tables and parameters fitted on data before 2026-10-01; scored on the
validator's prompts of 2026-10-02 → 2026-11-07 (36 days; profile at most 5 weeks stale — the champion's
situation). Quantile matching off in all research runs.

**Result (fold B, official scorer, 1000 paths).**

| config | v2 AGG | v3 AGG | paired vs best miner / median miner (v3) |
|---|---|---|---|
| CRYPTO_24H | 3535.16 | 3535.55 | -0.03 % / -0.26 % |
| COM_EQU_24H | 1699.70 | 1696.93 | -0.08 % / -0.73 % |
| CRYPTO_1H | 761.26 | 760.73 | -0.06 % / -0.33 % |

v3 per asset (CRPS, paired rank): BTC 2435.84 1/231, ETH 3320.31 1/232, SOL 4038.80 1/232, XRP 3845.87 1/231,
HYPE 5121.09 85/231; XAU 1234.73 1/228, NVDAX 2138.57 1/229, GOOGLX 1946.60 32/232, TSLAX 2603.52 1/231,
AAPLX 1425.00 2/229, WTIOIL 2937.56 191/231, SPCX 3556.53 44/222, SP500 809.68 1/231;
1h: BTC 522.15 1/223, ETH 711.41 1/223, SOL 876.47 1/223, XRP 831.00 1/223, HYPE 1096.42 31/223.

Component evidence:
- Tick snapping, paired: SOL 1h 960.10 → 959.72, XRP 1h 890.09 → 889.85, XAU 24h 1237.10 → 1236.82 (BTC: none).
- Two clocks vs one clock, same parameters, post-DST days only (2026-10-26 → 11-07): crypto-24h 4040.01 vs
  4041.05, crypto-1h 869.89 vs 870.13, com-equ 2279.84 vs 2279.52. Small, kept (-0.03 % on crypto).
- Trainer ablations on fold B (validation loss vs full v3; + = the component helps): no session features /
  discount: SP500 +0.19 %, WTIOIL +0.09 %, AAPLX +0.09 %, XAU +0.03 %, crypto 24h/1h 0.00 ± 0.02 %;
  geometric instead of power mean: +0.00 … +0.05 %. (The "no extras" and "v2-like" runs were cut for budget.)
- Freshness: the same v2 code with parameters fitted before July (fold A) scores, on October, com-equ
  1705.31 against 1699.70 for the September fit (XAU 1246.61 vs 1237.10, SPCX 3643.31 vs 3578.39), crypto
  3536.61 vs 3535.16. Short-history assets want recent data; crypto does not care.
- Events (fold B, gap to the per-prompt median miner): crypto CPI -1.47 %, equity earnings -6.09 %, XAU
  FOMC -2.67 % / NFP -2.15 %, crypto-1h prompts spanning the CPI minute -15 %; PCE/GDP +0.2 … +1.9 %
  (multipliers too high, n is small). WTIOIL loses 0.5 % on ordinary prompts: unexplained.
- Against the top-5 miners of the last 12 days (ex-post selection!) we lose 0.2–0.3 % on crypto, all of it in
  the high-realised-vol deciles and after quiet states; over the 36 days we are level or ahead.

**Verdict.** Keep v3 (com-equ -0.16 %, 1h -0.07 %, crypto-24h flat). Discard nothing; power mean is nearly
free of effect.

**Failures / lessons.**
- BLAS threads x 13 workers = load 207 and a dataset build that never ended; one thread per worker fixed it.
- Editing a shell script while it runs breaks it; `pkill -f` matched my own command line.
- Ten-minute sleeps let the prompt cache expire: the last hour cost three times the first three. The
  budget, not the clock, ended the exploration (MLP level model, volume features, HYPE/WTI digging untested).

## Iteration 4 — nomination under the landing order (approx. 17:10 → 17:30 UTC)

**Change.** None to the model. Champion = commit 6e3fc56 (v3 code, fold-B parameters fitted on data before
2026-10-01), served with Latin-hypercube shocks, quantile matching (8x reference) and tick snapping.

**Result (official scorer, 1000 paths, validator prompts scored 2026-10-08 → 2026-11-07, out of sample for
the served parameters; starter on the same prompts).**

| config | starter | champion | change | paired vs best / median miner |
|---|---|---|---|---|
| CRYPTO_24H | 3785.58 | 3663.19 | -3.23 % | -0.02 % / -0.26 % |
| COM_EQU_24H | 1988.95 | 1799.38 | -9.53 % | -0.01 % / -0.72 % |
| CRYPTO_1H | 833.37 | 791.12 | -5.07 % | -0.06 % / -0.33 % |

Per asset (CRPS, paired rank): XAU 1255.88 (1/228), NVDAX 2313.13 (1/230), GOOGLX 2114.14 (55/236), TSLAX 2812.17 (1/232), AAPLX 1529.61 (5/233), WTIOIL 2919.58 (225/233), SPCX 3772.52 (43/225), SP500 853.07 (1/235), BTC 2495.26 (1/235), ETH 3502.18 (1/233), SOL 4214.10 (1/229), XRP 3946.66 (1/233), HYPE 5280.17 (88/234); 1h: BTC 538.15 (1/226), ETH 752.41 (1/226), SOL 916.59 (1/226), XRP 856.93 (1/226), HYPE 1133.73 (34/226).

Checks: nomination probe ok (0.5 s per 24h call, 0.1 s per 1h call); `research/robust.py` passes (constant
context, 3-day and 5-hour contexts, 15 % NaN + 30 stale minutes, rows missing altogether, a 5 % last-minute
jump, bare series, 1/7/100/250 paths, unknown asset, missing or unreadable calendar; an all-NaN context raises).

**Not done (stopped by the landing order at 90 % of credits).** A refit of profile and parameters through
2026-11-08 was training when the order arrived; I stopped it and serve the validated fold-B parameters
instead. The freshness experiment (iteration 3) suggests the refit would help SPCX, SP500, XAU and WTIOIL;
it is unverified, so it is not in the champion. Also untested: the "no extras" and "v2-like" ablations, a
network for the level, volume features, WTIOIL/HYPE post-mortems.

**Verdict.** Nominate. **Lesson.** Plan the credit budget before the first GPU job: long sleeps expire the
prompt cache and each later turn pays for the whole transcript again.

**Housekeeping after nomination (17:26 UTC).** `agent/BUDGET.md` (harness-owned, rewritten continuously)
is no longer tracked and is listed in `.gitignore`. No model file changed after commit 6e3fc56. The times in
the headers of iterations 2-4 are approximate: I had guessed them, and corrected them from the harness clock
(2.6 h elapsed when the v3 fold-B pipeline was launched, 4.2 h when the landing order arrived).
