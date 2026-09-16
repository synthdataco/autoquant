# Journal — campaign-3 (fable-5-1)

Conventions: CRPS numbers are the validator's summed CRPS per prompt (lower = better). "gap" = my
mean per-prompt score (CRPS minus the field's best CRPS on that prompt), i.e. what the validator
averages into the smoothed score. Field-relative numbers come from `agent/evaluate.py`, which
wraps synth-lib's `backtest()` + `compute_combined_smoothed_scores()` exactly like the verdict
runner. Parameter sweeps (`agent/tune.py`) use the validator's CRPS function on in-memory paths
over pre-field windows (no ranks there).

## 0. Setup & scoring analysis (hour 0-1)
- Field bundle covers 2026-07-27 -> 2026-08-29 (33 days), ~24 prompts/day/asset for 24h
  competitions, ~150/day/asset for crypto-1h, 256 miners.
- Per-prompt score = CRPS - min CRPS over miners; smoothed = coefficient-weighted mean over the
  window (10d for 24h comps, 5d for 1h); reward = softmax(beta * smoothed), beta = -0.15 (24h)
  / -0.3 (1h). Score = mean over comps of min(1, my reward / top miner's reward).
- Top miner smoothed scores over the last 10 days: crypto-24h ~108, com-equ-24h ~25, crypto-1h ~62
  (median field: 169 / 33 / 41). Being 5 points behind the top costs a factor ~0.5 (24h) / ~0.2 (1h).
- Harness artifact (verdict-inherent): predictions are generated only inside the window, so the
  first day of scored prompts (start_time before the window) has no file and is scored at the
  field's p95. For a 10-day verdict that is 10% of the 24h prompts. `evaluate.py` default
  generates the preceding horizon (true model quality); `--verdict-exact` reproduces the handicap.
- Data: strong hour-of-week variance seasonality (crypto peak 13-14 UTC, ratio ~2; tokenized
  equities ratio 4-6 with weekends at ~0.35x). Log hour-of-week profiles are stable out of sample
  (corr 0.6-0.85 except XRP). Realized-vol scaling is ~flat across horizons (near random walk).
  5-min return kurtosis 20-170.

## 1. Starter (log-normal martingale, sd of 1-min returns) — 3d window ending 08-29, 1000 paths
- crypto-24h rank 257/257 (with the first-day p95 artifact), com-equ 248/257, crypto-1h: harness
  error (synth-lib tried to ingest the day after the snapshot; patched with a read-only loader).
- Verdict: reference only.

## 2. v1: seasonal EWMA variance + lognormal vol mixture + Student-t shocks
- Hypothesis: deseasonalised EWMA vol with hour-of-week profile and fat tails beats the starter.
- Result (3d ending 08-29, full coverage): crypto-1h rank 80/257, reward_vs_top 0.42 (SOL gap
  rank 1/257). BUT 24h comps rank 257/257: PIT diagnostic showed 5-min step sd 2.2x too wide in
  the 24h format only.
- Root cause: pandas 3 stores the index at microsecond resolution; `index.asi8 / 60e9` gave
  dt = 1 min for 5-min bars => variance x5. Fixed with `.total_seconds()`.
- Lesson: check PIT/sd ratios at every horizon before tuning; the first tuning sweep (which
  favoured scale 0.8 / very fat mixtures) was compensating for this bug and is discarded.

## 3. v1b (v1 + dt fix) vs the field — 3d ending 08-29, 1000 paths, full prompt coverage
- crypto-24h rank 176/257 (my smoothed 120 vs best 68, median 106); com-equ-24h rank 173/257
  (50 vs 28, median 36); crypto-1h rank 127/257 (51.0 vs 48.0, median 52.2). Score-proxy 18.1.
- Parametric sweeps after the fix are FLAT: vov 0.2-0.5, nu 3-6, scale 0.9-1.1 all within
  +-0.5% CRPS on the Jun-Jul window (both formats). The family is at its optimum; gains must be
  structural.

## 4. Field study (offline bundle, last 10 days; CRPS levels only, no path dissection)
- The field is very tight: the top miner's CRPS is ~0.99x the per-prompt field median in the 24h
  competitions and ~0.997x in crypto-1h; p10-p90 across miners is 0.996-1.018 (BTC 24h). Many
  UIDs share identical scores (same model deployed several times).
- The top's edge is uniform across start hours and a bit larger on high-vol prompts
  (BTC 0.978 vs 0.994). Per-prompt min/median ~0.974 (someone gets lucky each prompt), which is
  why the top's smoothed score is ~68 (crypto-24h) even though it beats the median by 1%.
- Consequence: I sit at the median; reaching the top needs a consistent ~1.5% CRPS gain in the
  24h formats and ~0.6% in the 1h format.

## 5. Structural features, tested one at a time (tune.py, Jun 1 - Jul 25 2026, 162 prompts/asset)
- Fine seasonality (5-min-of-day within hour + minute-of-hour for 1h): com-equ -0.20%, crypto-24h
  -0.05%, crypto-1h -0.13%. KEEP. (US open bar 13:30 UTC = 2-3x its hour for stocks.)
- US holiday calendar (stocks -> Saturday-like profile; SP500 x0.6; commodities x0.5):
  full window -0.13% (com-equ); holiday windows (Juneteenth, Jul 3): stocks -1% to -6%, SP500
  -0.5%, XAU/WTIOIL +1% => commodity factor set to 1.0. KEEP for stocks/SP500. Labor Day
  2026-09-07 is in the evaluation window.
- Exact-zero point mass per step (thin markets): neutral (XAU -0.1%, others +0.05%). DROP.

## 6. Bounds and vol-side experiments (tune.py, Jun 1 - Jul 25 2026)
- ORACLE (perfect knowledge of the horizon's deseasonalised variance): 24h -1.0% to -1.7%
  (BTC 3151 vs 3189), crypto-1h -2.3%. => vol forecasting alone cannot close the ~1.5% gap to
  the 24h top; it is roughly the size of the whole gap. Consistent with the top miners' edge on
  BTC/ETH coming from information I do not have offline (e.g. implied vol).
- Term structure (recent vol decaying into the blend, tau 2-24h, hl_recent 0.5-2h): flat (+-0.1%).
- 1-min vs 5-min returns for the 24h variance estimate: flat. Blend weights / half-lives: flat
  (best hl_mid=12h: -0.07%).
- Empirical standardised-residual bootstrap instead of Student-t: 24h +0.4% (worse), 1h -0.15%.
- 1h: tau 0.5h/hl_recent 0.25h -0.07%; w_short 0.6 -0.07%; nu 5 -0.17%; vov 0.2 -0.1%.
- Lesson: in this family every knob is second-order (CRPS loss ~ 0.5 * (scale error)^2). The
  remaining first-order items are structural (stratified ensembles, holiday/seasonal structure).

## 7. v2 vs the field (3d ending 08-29): crypto-24h 125/257, com-equ 182/257, crypto-1h 82/257
- Per-prompt diagnostics (agent/results/v2_*_prompts.parquet): my CRPS / field-median per prompt:
  BTC 0.999, SOL 0.997, HYPE 0.998, ETH 1.003, XRP 1.011; XAU 1.000, SP500 1.000, but AAPLX 1.012,
  NVDAX 1.011, TSLAX 1.011, WTIOIL 1.015 (worse than the median on 75-93% of prompts). The field's
  p10 miner sits at ~0.985 of the median in the 24h formats, ~0.98 in 1h.

## 8. FIRST-ORDER FINDING: session-level variance miscalibration on equities
- Realized / predicted variance by session bucket (24h format, Jun-Jul and Aug 10-28): US cash
  session (13:30-20:00 UTC) 1.4-2.0x for AAPLX/NVDAX/TSLAX/GOOGLX/SPCX, ~1.0 for off-hours.
  |r|/sd 0.87-0.97 vs ~0.70 expected. The long-run hour-of-week profile understates the current
  US-session share of variance: profile-implied US/off ratio 3.0-3.7 vs 4-7 in recent months
  (drift since the thin early months + my +-1h smoothing blurring the session boundary).
- Fix A (adaptive session calibration from the context: per-bucket variance ratio vs the 7-day
  mean, shrunk with pseudo-count n0=150; weekend ratio frozen with n0=3000):
  com-equ -0.35% (stocks -0.4..-0.8%, SP500 +0.3% -> excluded), crypto-24h -0.2%, 1h -0.1%.
- Fix B (profiles refit with 90-day recency half-life, no hour smoothing = season_v3.json):
  com-equ -0.3%, crypto-24h -0.13%; additive with A (com-equ total -0.6%).
- Also folded in: Latin-hypercube stratified shocks + OU log-vol along the path (s0 0.2 -> vov
  0.35, half-life 12h): -0.10% on 24h; hl_mid 12h: -0.07%. 1h: lhs + nu 5 + vov 0.2 +
  recent-vol adaptation (tau 0.5h) + w_short 0.6: -0.26%.
- v3 = all of the above. Caveat: profiles fit on the whole snapshot are in-sample for every
  test window; the month-by-month drift table independently supports recency weighting.

## 9. v3 vs the field (3d ending 08-29, 1000 paths, full coverage) — Score-proxy 34.1
- crypto-24h: rank 22/257, reward_vs_top 0.44 (smoothed 72.8 vs best 68.3, median 106).
  ETH gap-rank 1/257, SOL 16, XRP 22, BTC 42, HYPE 100.
- com-equ-24h: rank 161/257, reward_vs_top 0.20 (40.9 vs 28.1, median 35.9). SPCX gap-rank 2,
  XAU 96, GOOGLX 97, SP500 101, TSLAX 140, NVDAX 149, AAPLX 150, WTIOIL 230.
- crypto-1h: rank 128/257, reward_vs_top 0.38 (51.7 vs 48.1) — slightly WORSE than v2 (50.5)
  although the 1h changes gained -0.4% on the Jun-Jul tune window. HYPE/XRP degraded.
  Hypothesis: August is a jumpier regime (SOL/XRP weekend variance 4x predicted in the Aug 10-28
  diagnostic); thinner tails (nu 5) / thinner mixture (vov 0.2) lose there. => ablate on the
  Jul 27 - Aug 28 window before keeping any 1h change; adopt a two-window protocol
  (Jun 1 - Jul 25 AND Jul 27 - Aug 28) for every decision from here on.

## 10. Two-window protocol results (Jun 1-Jul 25 "J", Jul 27-Aug 28 "A")
- 1h ablation on A revealed DOUBLE COUNTING: the session ratio multiplied a recent-vol EWMA that
  already reflected the current session (-0.4% without the ratio on A). Fix: deseasonalise the
  context by the adapted profile (profile x ratio) before the EWMAs. After the fix the ratio is
  neutral on A for 1h and the fix itself is worth -0.4% on A (BTC 491.3 -> 489.2).
- 1h on A after fix: tau=0 -0.14% (J: +0.07%) => revert tau to 0 (simpler). nu 5 / vov 0.2 vs
  3.5 / 0.35: flat on A (+-0.05%), keep. LHS: -0.1% on A, keep.
- 24h (J, pre-window season file): 5 session buckets -0.05%, winsor cap 1.5/none flat, long-run
  variance prior lr_w 0.2: equities -0.1%, crypto +0.1% => all dropped (keep 3 buckets, cap 3, no prior).

## 11. Calibration in the most recent month (Jul 27 - Aug 28, v3 model, 24h)
- Realized / simulated variance 1.2-1.5 in US hours AND off-hours for nearly all assets (BTC
  1.44/1.27, NVDAX 1.27/1.63, XAU 1.36/1.46); PIT tail share 0.13-0.17 (ideal 0.10), central
  share 0.37-0.43 (ideal 0.50). In Jun-Jul the same diagnostic was ~1.0 for crypto. => August
  was a rising-vol month that trailing (7-day) estimators lag; not a structural bias.
- Closed-form check: Gaussian CRPS loss is symmetric in log-scale (x0.9 or x1.1 both cost
  ~0.25%), so a scale bias is only a regime bet. Testing scale 1.0-1.15 on both windows.
- Per-asset tails (J window): nu 6 helps single stocks and alt-coins by ~0.2%, hurts
  XAU/SP500/SPCX slightly; vov 0.2 similar but smaller. To be confirmed on the A window.

## 12. Decisions from the two-window protocol -> v4
- 24h: nu 6 (J -0.09%, A -0.14%, every asset >= flat on A), vov 0.3 (0.2-0.35 equivalent under
  nu 6), OU log-vol kept (LHS+OU: J -0.10%, A -0.12%), hl_mid 12 (A: 24h +0.07%), session
  ratio kept (A: -0.08%), scale 1.05 (J -0.09%, A -0.08%; SP500 and WTIOIL prefer 1.0 ->
  SP500 override 1.0). 1h: scale 1.0 (1.05 costs +0.11% on both windows), tau 0, no session
  ratio, lhs on, nu 5, vov 0.2.
- Profile-fit inconsistency found: exact-zero returns were dropped when fitting the seasonal
  profiles but kept in the model's squared returns => the profile is inflated where zeros are
  frequent (XAU weekends 32% zeros; realized/predicted weekend variance 0.58-0.66). Refit with
  zeros kept (season_v4.json): weekend levels -5..-12% for thin assets; testing on J and A.

## 13. v4 vs the field (3d ending 08-29): crypto-1h RANK 1/257 (reward_vs_top 0.97), but crypto-24h 99/257
- 1h: smoothed 46.5 vs best other 48.2 (ETH/SOL/HYPE near the top). The 1h stack works.
- 24h crypto fell from rank 22 (v3) to 99: BTC +0.6%, ETH +0.9%, XRP +1.2% CRPS vs v3 on Aug 26-29.
  Ablation on exactly those prompts: nu 4/vov .35 -0.6%, scale 1.0 -0.5%, no session ratio -0.5%
  (combined -1.4%) — every change that was positive on Jun-Jul AND on Jul 27-Aug 28 is negative
  on the last 3 days. Interpretation: Aug 26-29 is a vol-collapse regime after a jumpy August;
  trailing (7-day) estimates run high, so anything that widens the shoulders loses. hl_mid 24
  is worse there too (more recent weight helps), i.e. the lever is estimator adaptivity, not
  the tail shape per se.
- Zero-inclusive profile fit (season_v4): flat on both windows (+-0.02%) -> not adopted.
- Decision protocol: 3-window sweep (J, A, last-3-days) and pick the setting with the best
  AVERAGE relative CRPS; the evaluation window (Aug 30-Sep 8) follows the collapse regime
  directly, so the last-3-days window gets equal weight.

## 14. Three-window sweep (J: Jun 1-Jul 25, A: Jul 27-Aug 28, L3: Aug 26-29), 24h, 9 assets
- J cannot discriminate (all variants within +-0.1%). A: more adaptive estimators help slightly
  (hl_mid 6-8, -0.06%); nu 4 vs 6 equal; scale 1.0 vs 1.05 equal. L3: nu 5/vov .35/scale 1.0
  -0.38%, nu 4 -0.32%, no session ratio -0.16% (equities mixed: TSLAX/XAU want it, NVDAX not).
- Best average: {nu 5, vov 0.35, scale 1.02, hl_mid 8, w_mid 0.6, w_short 0.2} = J 0.9998,
  A 0.9996, L3 0.9963. Adopted as v5 (24h). Session ratio kept (structural, ~neutral on average).
- Lesson: on this problem the shape/scale knobs are regime bets worth +-0.5% on a 3-day window
  and ~0 over a month; the estimator's adaptivity is the only knob that is >= 0 in all regimes.

## 15. Verdict-exact 10-day estimates (official backtester, first-day penalty included)
- v4: crypto-24h rank 9/257 (reward_vs_top 0.50), com-equ 178/211 (0.000), crypto-1h 1/257
  (3.30, capped) -> Score 50.1.
- v5: crypto-24h rank 1/257 at the final round (reward_vs_top over the last day's rounds 0.44,
  reward-rank 21), com-equ 178/211 (0.000), crypto-1h 1/257 (3.25) -> Score 48.1.
- v5 3-day clean window: crypto-24h 22/257 (0.24), com-equ 165/257 (0.16), crypto-1h 1/257 (1.05).
- com-equ is 0 for ANY candidate in this harness (first-day penalty ~ +40 points vs a top at 29).
- Decision: nominate v5 (equivalent to v4 on the 10-day estimate, better on the 3-day window and
  chosen on three windows). Nomination remains amendable while iterating.
