# Journal — campaign-4 (opus-5-5)

## 0. Setup & reconnaissance (h 0.0–0.5)
- Read CAMPAIGN.md, DATA.md, the validator's `crps_calculation.py` / `competition_config.py`.
  Key scoring facts: 24h = sum of CRPS (bps) over 288×5min + 48×30min + 8×3h changes + the 24h
  absolute price (/price×1e4). 1h = 1/2/5/15/30min changes + 60min abs + 12 "gap" intervals
  (change from t0 to t0+5k min) + since 2026-09-22 13:16 UTC (INSIDE our eval window) a volatility
  CRPS term: 5.25 × mean CRPS of the per-block std of 1-min returns (5/15/60 min blocks).
  → within-path vol-of-vol matters for 1h after 09-22; the marginal per-step vol dominates 24h.
- **Environment finding**: `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is NOT set and no offline score
  bundle exists; the backtester therefore queries the live public Synth API for field scores and
  rewards. `predict.py --eval-end 2026-09-14` also fails (the 24h realized path needs 09-15, absent).
  Decision: always use `--eval-end <= 2026-09-13` (the rewards query extends +24h past the last
  scored prompt, so this keeps every API query <= 09-14). Field scores cached once by
  `agent/fetch_field.py`, hard-capped at 2026-09-14T00:00Z.
- Starter BTC-24h, 3 days to 09-13 via predict.py: mean CRPS 2258.0 (69 prompts), 67 s wall.
- Built `agent/harness.py`: parallel (14 proc) evaluator using the validator's own
  `calculate_crps_for_miner` / `calculate_vol_crps_for_miner` (what backtest() calls), realized
  path via synth-lib's `_slice_real_prices` + the validator realized-path fill, float32 paths.
  ~360 prompts in 10 s.
- Data facts measured (since 2026-06-01): equities/gold vol is strongly seasonal in NY local time
  (09:30 ET bar 2.5–2.8× mean, weekends 0.2–0.5×, XAU weekend ≈ 0.2×, Sunday 18:00 ET reopen).
  Crypto: 8–11 ET hotter (1.4–1.7×), weekends 0.6–0.8×. Variance ratios VR(k)=Var(k-min)/k·Var(1m)
  ≈ 0.9–1.05 for most assets; mean reversion at 3–24h for SP500 (0.74 @1d), SPCX (0.70), WTIOIL,
  SOL (0.84). Training data spans EST and EDT → seasonality modelled in America/New_York time.

## 1. v1: seasonal EWMA-blend vol + Student-t + per-path vol multiplier (h 0.5–1.0)
- Hypothesis: NY-time seasonal variance profile × EWMA level beats the starter's constant
  1-min std across all three configs.
- Change: `modeling.py` v1 (profile from `fit_season.py`, cutoff = validation start;
  EWMA hl 60/360 min + 7-day mean; t(nu=5); log-normal per-path multiplier sv=0.25).
- Result (harness, hourly grid, mean CRPS; V1 = 08-01→08-15, V2 = 08-29→09-13):
  | | starter V1 | v1 V1 | starter V2 | v1 V2 |
  |---|---|---|---|---|
  | BTC-24h | 1776 | 1666 | 2455 | 2299 |
  | NVDAX-24h | 2498 | 2166 | 2048 | 1729 |
  | TSLAX-24h | 2625 | 2227 | 2692 | 2302 |
  | BTC-1h | 387 | 363 | 536 | 501 |
  (all 18 asset-configs improve, −2% … −16%.) Field on crypto-24h (same prompts): v1 is
  −0.04% … +1.07% vs the field MEDIAN; the field p10 is ~1% below the median and the per-prompt
  min ~2% below. The field is tightly packed: the remaining game is in single percents.
- Verdict: keep.

## 2. Ablations/knobs on v1 (paired, common random numbers, stride 2, V1+V2)
- Flat seasonality (profile off): +7.5% CRPS on E24, +2.0% C24, +1.5% C1H. Seasonality is the
  single biggest lever, especially for equities.
- scale 0.9: −0.16…−0.23%; sv 0.4: −0.21…−0.41%; nu 4: −0.02…−0.18%; the other directions
  hurt. → sharper centre + heavier tails wanted; hint that the vol level forecast is noisy.

## 3. Vol-level forecast study without simulation (QLIKE; `volstudy.py`, `harfit.py`)
- Single EWMAs: best horizon-matched half-life is ~1–4h for the next hour, ~12–24h for the
  next day.
- Horizon-bucketed log-HAR (buckets 0-5, 5-15, 15-60, 60-180, 180-480, 480-1440 min; features
  log(EWMA_hl / 7d-mean), hl ∈ {15,60,240,720,1440}; pooled crypto / pooled equities; fitted by
  QLIKE on data < 08-01): out-of-sample (08-01→09-14) QLIKE lower than the v1 blend for EVERY
  asset and bucket (e.g. BTC 480-1440: 0.357 vs 0.476; XRP 0.477 vs 0.975; NVDAX 0.432 vs 0.523).

## 4. HAR (QLIKE) in the simulator + calibration (tune3/tune4)
- Change: `har_file` → per-minute level = l7d·exp(x·β_bucket), interpolated in log-horizon.
- Result vs v1 blend (sv 0.25 → 0.4, mean % CRPS change, V1+V2, stride 2):
  HAR alone: C24 +0.68 (worse!), E24 −0.56, C1H −0.42. HAR+sv0.4: C24 +0.07, E24 −0.95, C1H −0.61.
  On top of HAR+sv0.4: scale 0.8 → C24 −0.40, E24 −0.06, C1H +0.14; scale 0.7 → C24 −0.41,
  E24 +0.09, C1H +0.42; sv 0.55 worse; OU within-path log-vol (sv0 .2→.5, kappa .002) ≈ scale .8.
- Lesson: a much better QLIKE forecast (conditional MEAN variance) is not automatically a better
  CRPS forecast: CRPS for a vol-uncertain return prefers a predictive variance BELOW the mean
  variance (E_σ sqrt(s²+σ²) is concave in σ²). Crypto-24h needs scale ≈ 0.75.
- Verdict: keep HAR, with per-config scale/sv (C24 0.75/0.4, E24 0.9/0.4, C1H 1.0/0.4).

## 5. HAR fitted by closed-form Gaussian CRPS on step returns (`harcrps.py`)
- Hypothesis: fitting the level directly on the scoring rule removes the need for scale tuning
  and may beat QLIKE+scale.
- Pitfall: first fit silently did not converge (L-BFGS stopped on tiny gradients in log-return
  units); fixed by normalising the loss by mean base sd.
- Result (vs HAR-QLIKE sv0.4 scale0.8): sv0 → C24 −0.05, C1H −0.08, E24 +0.31; sv0.4 → +0.3…+0.8.
- Verdict: discard. The CRPS-fitted level is equivalent to QLIKE + one scalar on crypto and
  worse on equities.

## 6. Where we stand vs the field (model m2 = HAR + per-group calibration; full field prompt grid)
- V1: C1H −0.08…−0.38% vs field median; C24 −0.10…+0.24%; E24 −0.50…+0.46%.
- V2: C1H −0.04…+0.06%; C24 −0.17…+0.25%; E24 +0.63…+2.30% (AAPLX +2.3, SPCX +2.1, TSLAX +1.7).
- Field top decile (per-prompt p10) is ~1–2% below the median. The median of this field is
  already a good model; beating it needs a real structural gain.
- Realised (validator) paths are forward-filled; equities/gold have 4–16% exact-zero 5-min
  returns (XAU 16%, weekend closure). Noted for a possible point-mass-at-zero experiment.

## 7. Oracle bound on the vol level (`research_oracle.py`, research only — uses the future)
- Hypothesis: if the realised future deseasonalised level were known, how much CRPS would we gain?
- Result (stride 3, vs m2): flat oracle level C24 −0.12, E24 −0.03, C1H +0.04; per-bucket
  oracle C24 −0.25, E24 +0.01, C1H −0.01. (V1 E24/C1H rows came out exactly 0.000, so part of
  the run may not have applied the oracle; treat as indicative only.)
- Lesson: the headroom in the LEVEL forecast is small at this point; the loss is elsewhere (shape
  of the intraday/weekly profile, distribution). Moved on.

## 8. Weekly profile bias (weekend over-dispersion) — fit_season `--norm-days`, `--hl-days`, `--stat`
- Observation: m2 lost 2–12% vs field median on equity prompts spanning Friday afternoon→weekend.
  Deseasonalised z ratio weekend/weekday on recent weeks was 0.2–0.4 for equities (should be ~1).
- Cause: squared returns were normalised by a CENTERED 3-DAY rolling mean; on a weekend that
  window is weekend-dominated, so weekend returns are divided by a depressed level and weekend
  variance is overstated 2–3× (e.g. TSLAX Sat profile 0.23 vs raw 0.09; XAU 0.10 vs 0.03).
- Change: 7-day (whole-week) normalisation; optional 30-day half-life recency weights; optional
  "abs" statistic (profile from mean |r|, squared — robust to rare jumps in thin bins).
- Result (paired, stride 3, % CRPS vs m2): s7 E24 −0.43 / C24 −0.06 / C1H −0.01;
  s7+hl30 E24 −0.63 / C24 +0.07 / C1H +0.11; abs+hl30 E24 −0.88 / C24 +0.02 / C1H +0.05.
  HAR refit on the new z: no gain (−0.60 vs −0.63). Scale/sv re-tune on E24: flat (±0.2).
- Verdict: equities use the abs+hl30 profile, crypto keeps the 3-day sq profile ("seasonM").
  Full-grid V2 E24 vs field median: from +0.6…+2.3% to −0.65…+0.8%.
- Also: no longer pooling Friday with Mon–Thu (Friday evening = weekend close): ~0 effect.

## 9. Point mass at zero (`fit_zero.py`, `zero_file`)
- Hypothesis: forward-filled realised paths (XAU Sat ~40–50% exact-zero 5-min changes; SOL/XRP
  ~4–5% from tick size) reward a zero atom.
- Result: E24 −0.05%, C24 +0.01…+0.04%. Verdict: discard (kept as an off-by-default option).
- Lesson: CRPS barely rewards atoms at this scale.

## 10. What is left vs the field
- Per consistent single miner (≥95% prompt coverage), the BEST miner beats us by only
  0.3–1.6% on E24 (V2) and 0.2–0.8% on crypto; we beat the median on most asset-configs.
- By absolute CRPS contribution, weekend/closed prompts matter little (Fri+Sun ≈ +0.26% of the
  E24 total even with 10% relative losses on XAU): weekday active-hour prompts dominate the mean.

## 11. Tuning sweeps on the new profile (stride 3, V1+V2)
- scale ±~11% per group: all within ±0.3% (E24 0.8: V1 −0.30 / V2 +0.05); nu 4/8: ±0.2%;
  sv 0.3: worse (+0.05…+0.26). Per-asset: only HYPE consistently prefers a higher scale (tiny).
- Verdict: calibration is at a flat optimum; keep C24 0.75/0.4, E24 0.9/0.4, C1H 1.0/0.4, nu 5.
- Variance-ratio check (5/30/180-min vs 1-min, weekdays, Jun–Sep): VR ≈ 0.9–1.1, no systematic
  microstructure mean reversion to exploit.

## 12. Oracle re-check with re-tuned calibration (research only)
- Per-bucket realised level with scale ∈ {0.85, 1.0}, sv ∈ {0.1, 0.25}: best C24 −0.80%,
  C1H −0.12%, E24 ≈ −0.6…−0.9% (the E24 number is confounded by the old profile in the oracle
  variants). So even a PERFECT vol-level forecast buys < 1%. The level forecast is near
  saturation; the field spread (best consistent miner 0.3–1.6% better) is of the same order.

## 13. Scheduled FOMC releases as part of the seasonal profile (`events`, `event_mult`)
- Hypothesis: the FOMC statement (14:00 NY; 2026-09-16 is inside the eval window; the calendar is
  published a year ahead) creates a predictable vol spike that the weekly profile averages away.
- Evidence (deseasonalised z on 5 FOMC days Jan–Jul 2026, vs prior 3 days): crypto ×4–6 in the
  first 30 min, ×2.5–3.8 in 30–90 min, ×1.4–1.6 in 90–240 min; XAU/SP500 ×6–7, ×6, ×3;
  tokenised stocks ×1.3–2.7; WTIOIL ×1.4–2.
- Change: multiply the profile (context AND future, so the HAR level is not double counted
  after the event) by per-class multipliers, conservatively rounded: crypto [4, 2.5, 1.3],
  XAU/SP500 [4.5, 3.5, 1.8], WTIOIL 1.4, xStocks [2, 1.5, 1].
- Result (`evtest.py`, paired, prompts overlapping each of 4 past releases, 500 paths):
  24h −0.68% of summed CRPS (crypto −0.9…−1.7%, SP500 −1…−3%, XAU up to −3.5%), 1h −1.2%
  (−0.6…−3.9% per asset/event; 2026-03-18 slightly worse +0.1…+0.8%). The multipliers are
  in-sample for these events, but the gain holds across events.
- Verdict: keep (`use_events: True`). Expected eval impact is small (≈10% of 24h prompts
  overlap the 09-16 release) but positive. No other tier-1 US release (CPI/NFP/PPI) falls in
  the 09-14→09-24 window by the published schedule.

## 14. Filtered historical simulation (`fit_resid.py`, `resid_file`)
- Hypothesis: resampling standardised historical step returns (per asset; kurtosis 8–35, real
  zero mass, tick effects) beats Student-t(5) innovations.
- Result (paired, stride 3, % vs t5+sv0.4): FHS sv0.4 C24 −0.01, E24 −0.09, C1H +0.04;
  sv0.25 +0.07…+0.20; sv0.1 +0.26…+0.41; sv0.55 +0.08…+0.30 (+scale 1.1: ≈0).
- Verdict: discard (off by default). The innovation SHAPE is not the lever either; the
  t5 × lognormal(0.4) mixture is already as good as the empirical distribution.

## 15. Final artifacts and official backtests
- Refit with cutoff 2026-09-14 (all data): season.npz = seasonMF (crypto: 3-day sq profile;
  equities: whole-week abs profile, 30-day half-life; Friday unpooled), har.json = har_2026-09-14.
  HAR coefficients are stable vs the 08-29 fit.
- Robustness: 20% NaN context, 2-hour context, trailing NaN, almost-empty context (new
  fallback: ~2 bp/min × profile) → finite paths of the right shape; 0.01–0.06 s per prompt at
  1000 paths; deterministic under SYNTH_BENCHMARK_SEED.
- `predict.py` cannot score COM_EQU_24H: synth-lib's MinutePriceStore.load_range raises on any
  NaN minute close, and every equity has NaN minutes in every 7-day window. E24 is scored with
  harness.py instead (same synth-lib calculate_crps_for_miner, same prompt times as the field).
  CRYPTO_1H needs --eval-end ≤ 2026-09-12 (09-13 makes the backtester try to ingest 09-15).

## 16. Horizon-tilted variance scale (`scale_slope`)
- Hypothesis: level uncertainty grows with horizon, so CRPS might prefer a variance multiplier
  that falls with horizon.
- Result (±0.05 per log-unit of h/60min): +0.05 → C24 +0.12, E24 +0.28; −0.05 → E24 V1 −0.31 but
  V2 +0.04, C24 +0.02, C1H +0.03. Verdict: discard (sign flips between windows, like the E24 scale).

## 17. Out-of-sample summary and champion
- V2 (artifacts < 08-29, 1000 paths, stride 2): champion vs starter −2.8…−19.5% on every asset
  (24h) and −4.9…−9.1% (1h); vs field median −0.67…+0.72%.
- The final commit reproduces predict.py BTC-24h exactly (2121.04 both runs; seeded).
- Nominated as champion: see CHAMPION and report.md.
