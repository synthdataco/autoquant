# Synth SN50 miner research report — campaign-3 (fable-5-1)

Audience: Synth miners. All numbers come from the validator's own CRPS function (via synth-lib)
and, where a rank is quoted, from synth-lib's `backtest()` + `compute_combined_smoothed_scores()`
run against the archived field (offline bundle, 2026-07-27 → 2026-08-29). Data cutoff
2026-08-29. Final numbers are in the "Results" section; the iteration log is `journal.md`.

## 1. Context: what the validator actually rewards

- A prompt's CRPS is the sum over scoring intervals of ensemble CRPS terms. For the 24h format
  (5-min steps) that is 288 five-minute returns (~61% of the total for BTC), 48 thirty-minute
  returns (~25%), 8 three-hour returns (~10%) and ONE absolute price at 24h (~4%). For the 1h
  format (1-min steps): 60 one-minute returns (26%), 30 two-minute, 12 five-minute, 4
  fifteen-minute, 2 thirty-minute returns, the absolute price at 60 min, and twelve "gap"
  returns from t0 to t = 5, 10, ..., 60 min (29%). Returns are in basis points, so the score is
  scale-free and the marginal distribution of SHORT returns dominates.
- Per prompt, a miner's score is `CRPS − min CRPS over miners`; the smoothed score is the
  asset-coefficient-weighted mean over the window (10 days for the 24h competitions, 5 days for
  1h); reward = softmax(β · smoothed) with β = −0.15 (24h) / −0.3 (1h).
- The field is extremely tight. In the last 10 days of the bundle the top miner's CRPS is
  ~0.99× the per-prompt field median in the 24h competitions and ~0.997× in crypto-1h; the
  p10–p90 miner range is 0.996–1.018 of the median (BTC 24h). Whoever is lucky sets the
  per-prompt minimum (min/median ≈ 0.974), so even the top miner carries a smoothed score of
  ~68 (crypto-24h), ~25 (com-equ), ~35–60 (crypto-1h). Consequence: one per-cent of CRPS is
  the difference between the median and the podium.
- Harness detail worth knowing: the verdict runner generates predictions only inside the
  evaluation window, so the first day of scored prompts (whose start time precedes the window)
  has no prediction and is scored at the field's 95th percentile. That is ~10% of the 24h
  prompts in a 10-day window and it hits every candidate equally; `agent/evaluate.py`
  reproduces it with `--verdict-exact` and otherwise generates the preceding horizon so that
  model quality can be measured without it.

## 2. Approach

A closed-form, CPU-only simulator (`agent/modeling.py`, ~100 ms per 1000-path prompt):

1. **Returns built like the validator's realized path** (closes at 5-minute marks for the 24h
   format, minute closes for 1h), NaN dropped, never interpolated.
2. **Seasonality**: per-asset variance profiles fitted offline on the whole snapshot
   (`agent/season.json`, produced by `agent/fit_season.py`): hour-of-week (168 bins),
   5-minute-of-day within the hour (captures the 13:30 UTC US open bar = 2–3× its hour for
   tokenized stocks, 20:00 close, half-hour marks) and, for the 1h format, minute-of-hour
   (the first minute of every hour carries ~1.5× variance for crypto). Profiles use a 90-day
   recency half-life and no smoothing across hours: the tokenized equities' US-session share
   of variance has drifted up since their thin first months, and the smoothing blurred the
   session boundary.
3. **Volatility level**: deseasonalised squared returns from the 7-day context, combined
   from EWMAs (half-lives 2h / 12h for the 24h format, 15 min / 6h for 1h) and the 7-day mean.
4. **Adaptive session calibration** (24h format): the ratio of realized to profile variance
   is re-estimated from the context for three session buckets (US cash session 13:30–20:00
   UTC, weekday off-hours, weekend), shrunk toward 1 with a pseudo-count of 150 observations
   (weekend frozen), and the context is deseasonalised by the adapted profile before the EWMAs
   (otherwise a recent-vol EWMA double counts it — this bug cost 0.4% on the 1h format).
5. **US holiday calendar**: single-stock tokens get a Saturday-like profile on NYSE holidays,
   SP500 (index perp, tracks futures) ×0.6, commodities unchanged (no measurable effect).
   Labor Day 2026-09-07 is inside the evaluation window.
6. **Paths**: log-price martingale; Student-t shocks (ν = 6 for 24h, 5 for 1h) sampled by
   Latin-hypercube stratification per step (removes the Monte-Carlo noise of a 1000-member
   ensemble), multiplied by a per-path lognormal volatility factor that follows an
   Ornstein–Uhlenbeck log-vol process along the path (sd 0.2 at the start relaxing to 0.3 with
   a 12h half-life; constant sd 0.2 for 1h). Global scale 1.05 for the 24h format (1.0 for
   SP500 and for 1h). Prices rounded to 8 significant digits (validator format rule).

Everything was decided by pre-registered A/B comparisons with the validator's CRPS on two
windows of opposite character (Jun 1–Jul 25 2026: calm/normal; Jul 27–Aug 28 2026: rising
vol), one variable at a time, and confirmed by field-relative ranks on the last 3 days of the
bundle. Foundation models were not used: the field's dispersion (±1%) is far below what a
model change of that kind would need to deliver reliably, and the GPU would not have helped a
problem whose losses are second-order calibration terms.

## 3. Experiments (what worked, what did not)

Each row is one pre-registered A/B on the validator's CRPS (mean over ~160 prompts per asset on
the June–July window "J", ~100 on the July 27–August 28 window "A"; negative = better).

| # | Change (one at a time) | J | A | Verdict |
|---|---|---|---|---|
| 1 | Starter (log-normal, sd of 1-min returns) → seasonal EWMA + t-mixture (v1) | large | – | keep; but a pandas-3 resolution bug (`asi8` in µs) inflated 24h variance ×5 until fixed |
| 2 | Fat-tail knobs (ν 3–6, vol-of-vol 0.2–0.5, scale 0.9–1.1) after the fix | ±0.5% | – | flat; family at its optimum |
| 3 | Fine seasonality (5-min-of-day within hour; minute-of-hour for 1h) | −0.2% (equities), −0.05% (crypto), −0.13% (1h) | | keep |
| 4 | US holiday calendar (stocks Saturday-like, SP500 ×0.6) | −0.13% overall; −1…−6% on holiday days | | keep (commodities: no effect → factor 1) |
| 5 | Exact-zero point mass per step (thin markets) | ±0.05% | | drop |
| 6 | Perfect-foresight variance ORACLE (bound) | −1.0…−1.7% (24h), −2.3% (1h) | | vol forecasting alone cannot close the gap to the top |
| 7 | Term structure of vol (recent → blend, τ 2–24h) | ±0.1% | ±0.1% | drop (24h); 1h: −0.07% J / +0.14% A → drop |
| 8 | Empirical standardized-residual bootstrap shocks | +0.4% (24h), −0.15% (1h) | | drop |
| 9 | EWMA half-lives / weights / 1-min vs 5-min estimator | ±0.1% | | flat |
| 10 | Latin-hypercube stratification of shocks | −0.07% | −0.1% | keep (free) |
| 11 | OU log-vol along the path (s0 0.2 → 0.35, 12h) | −0.10% (with 10) | −0.12% | keep |
| 12 | **Adaptive session calibration** (US / off-hours / weekend ratios from the context) | −0.35% equities (−0.4…−0.8% single stocks), −0.2% crypto | −0.08% | keep; SP500 excluded (+0.3%) |
| 13 | **Recency-weighted, unsmoothed profiles** (90-day half-life) | −0.3% equities, −0.13% crypto | | keep |
| 14 | 1h: ν 5, vol-of-vol 0.2, w_short 0.6 | −0.26% (with 10) | flat | keep |
| 15 | 1h: session ratio | −0.11% | +0.05% (after fixing a double-counting bug worth 0.4%) | drop for 1h |
| 16 | 5 session buckets / winsor cap / long-run variance prior | ±0.05% | | drop |
| 17 | ν 6 for 24h | −0.09% | −0.14% | adopted in v4, then revised (see 19) |
| 18 | Scale 1.05 (24h) / 1.05 (1h) | −0.09% / +0.12% | −0.08% / +0.11% | 24h 1.05 in v4, 1h stays 1.0 |
| 19 | Aug 26–29 (vol collapse) ablation: ν 4, scale 1.0, no ratio each −0.5%; 3-window sweep | J flat | A flat | v5: ν 5, vol-of-vol 0.35, scale 1.02, faster mid EWMA (8h, weight 0.6) — best average of J, A and the last 3 days |
| 20 | Zero-inclusive profile fit | ±0.02% | ±0.02% | drop |

Field-relative checkpoints (3 days ending 2026-08-29, 1000 paths, all scored prompts covered):

| model | crypto-24h rank | com-equ-24h rank | crypto-1h rank | Score-proxy |
|---|---|---|---|---|
| starter | 257/257 | 248/257 | n/a | 0 |
| v1b (seasonal EWMA, bug fixed) | 176 | 173 | 127 | 18 |
| v2 (+fine seasonality, holidays) | 125 | 182 | 82 | 14 |
| v3 (+session calibration, recency profiles, LHS/OU) | 22 | 161 | 128 | 34 |
| v4 (ν 6, scale 1.05; 1h: ν 5, vov 0.2, LHS) | 99 | 165 | **1** | 38 |
| v5 (3-window choice for 24h) | 22 | 165 | **1** | 47 |

## 4. Results

Verdict-exact estimate (official backtester, 10 days ending 2026-08-29, hourly / 10-minute
grids, 1000 paths, first-day penalty included exactly as in the verdict runner):

| competition | v5 final rank | reward vs top (last day's rounds) | v4 (for comparison) |
|---|---|---|---|
| CRYPTO_24H | **1 / 257** | 0.44 (reward-rank 21) | rank 9, 0.50 |
| COM_EQU_24H | 178 / 211 | 0.000 | 178, 0.000 |
| CRYPTO_1H | **1 / 257** | 3.25 → capped at 1 | rank 1, 3.30 |
| Score = 100 × mean(min(1, ratio)) | **48.1** | | 50.1 |

Per-asset mean CRPS of v5 over those 10 days (gap = my mean per-prompt score): BTC 3986 (gap
100), ETH 5167 (86), SOL 6928 (165), XRP 10114 (283), HYPE 8999 (159); XAU 1618 (124), NVDAX 2735
(73), GOOGLX 1489 (89), TSLAX 2576 (66), AAPLX 1605 (130), WTIOIL 2735 (69), SPCX 3265 (115),
SP500 727 (14); 1h: BTC 813 (36), ETH 1061 (47), SOL 1409 (61), XRP 1957 (111), HYPE 1849 (76).

Reading the table:
- **crypto-1h**: the model beats the entire field by a wide margin (my smoothed 61.2 vs the best
  other miner's 66.2; on the clean 3-day window 46.3 vs 48.2). The 1h field is extremely
  homogeneous (top ≈ 0.997× median), so a well-calibrated 1-minute model with proper minute-of-
  hour seasonality, stratified sampling and a recent-vol EWMA is enough to lead it.
- **crypto-24h**: rank 1 at the final round, but only ~0.45 of the top miner's summed reward over
  the last day's rounds, because the harness scores my first day of prompts at the field's 95th
  percentile (≈ +20 points of smoothed score). On the valid prompts the model is clearly ahead of
  the field (3-day window: 75.6 vs 68.1 for the best other miner with the field median at 106).
- **com-equ-24h**: 0 for every candidate in this harness — the first-day penalty is ≈ +40 points
  on a field whose top sits at 29. On the clean 3-day window v5 ranks 165/257 (41.8 vs 28.2);
  the single-stock tokens (NVDAX/TSLAX/AAPLX) are still ~1% above the field median, SPCX and
  XAU/GOOGLX/SP500 are around it. This is the part of the model with the most room left.

## 5. Limitations

- Field-relative validation covers only 33 days (the bundle), and rank decisions on a 3-day window
  are noisy: v4's 24h changes were positive on both month-long windows and negative on the last
  three days (vol collapse). I resolved this with a 3-window average, but a regime shift right
  after the cutoff can still move the crypto-24h result by ±0.3 in reward ratio.
- The seasonal profiles are fitted on the whole snapshot (in-sample for every test window); the
  recency weighting is supported by an independent month-by-month drift table, not by an
  out-of-sample test.
- No information beyond the 7-day close series is used. The top 24h miners' remaining edge on
  BTC/ETH (~1% CRPS) is consistent with implied-volatility information I cannot access offline.
- Holidays: the calendar is hard-coded (NYSE full closures through 2027); early-close days are
  not modelled; the SP500 holiday factor (×0.6) rests on three past holidays.
- The 24h absolute-price term is evaluated by the harness against a realized path starting up to
  30 minutes away from the grid start; nothing in the model can correct that.

## 6. What I would do with more budget

1. Single-stock tokens: model the off-hours regime explicitly (pre-/post-market vs overnight vs
   weekend) with an intraday variance *shape* estimated from the context, not only the level;
   and fit a per-asset-group shock distribution for the US session.
2. Replace the constant per-prompt blend with a small regression (log next-24h realized variance
   on the EWMA features by session), fitted across assets — the oracle bound says ≤ 1–1.7% is
   available, and half of it is probably reachable.
3. Weekend behaviour of the commodity perps (XAU too wide, WTIOIL too narrow on weekends).
4. Cross-asset context (BTC vol for the alts) is not available through `simulate()`'s contract;
   a live miner should use it.
5. Run the field-relative validation on several disjoint windows of the bundle with the exact
   verdict setup, and treat the last-days window as a regime indicator rather than a tie-breaker.
