# A seasonal, event-aware volatility model for Synth SN50 — campaign-7 report

*Audience: Synth miners. Every CRPS here comes from synth-lib's own scorer (`score_crps` /
`backtest`, i.e. the validator's CRPS function), on the validator's real prompt times, over the
frozen snapshot (2025-08-01 → 2026-11-08). Lower CRPS is better. Full iteration log:
`agent/journal.md`.*

## 1. Context

A Synth miner answers a prompt `(asset, start_time, horizon)` with 1000 simulated price paths. The
validator scores the fan with CRPS on many interval returns at once. Measured on our own fans:

| configuration | what is summed (each term a CRPS in basis points) | share of the CRPS |
|---|---|---|
| CRYPTO_24H, COM_EQU_24H (5-min steps) | 288 5-min returns / 48 30-min / 8 3-h / final price | 61% / 25% / 11% / 4% |
| CRYPTO_1H (1-min steps) | 60 1-min / 30 2-min / 12 5-min / 4 15-min / 2 30-min / final price / 12 "gap" returns from the start / volatility term | 26 / 19 / 12 / 7 / 5 / 3.5 / 28 / 1% |

Three consequences drove the whole campaign:

1. In the 24h configurations **60% of the score is the 288 individual 5-minute returns**: the
   per-step volatility *profile* across the day matters more than the 24-hour total.
2. In the 1h configuration the very short returns and the cumulative "gap" returns dominate:
   the volatility of the *next tens of minutes* is what counts.
3. CRPS charges a scale error quadratically (≈ ¼·(ln σ/σ*)²): a ±10% volatility error costs
   0.25%. Gains come from fixing the **large** errors — regime boundaries, releases, the
   wrong time of day — not from polishing a global scale.

**The right benchmark.** The per-prompt 10th percentile of the field is not reachable by any
single miner (it mixes different miners on different prompts): over 2026-08-05..08-26 the best
single miner was +2.1% (1h), +1.1% (crypto-24h), +1.2% (com-equ) above it, and the top of the
field is a block of clones with identical scores. I therefore report CRPS against the **best
single miner** (chosen ex post, with full coverage) next to the starter.

## 2. Approach — the champion (v8)

A closed-form, CPU-only model. No weights; two small artifacts fitted offline on the snapshot
(`profiles.npz`, `events.json`). ~0.1 s per 24h prompt and ~0.03 s per 1h prompt.

1. **Weekly volatility profile keyed on New York time** — per asset, one value per minute of the
   week (10,080 buckets), separately for 1-min and 5-min returns. New-York keying keeps the US
   open, the 08:30 releases and the 16:00 close in place across DST switches (the hourly profile
   of every asset, crypto included, is more peaked on the NY clock than on UTC). The shape is
   *robust*: mean absolute return per bucket (squared), normalised by a centred 7-day level so
   calm and wild weeks weigh the same, smoothed ±4 min, the minute-of-hour pattern restored after
   smoothing (the first minute of each hour is 20-50% more active), and fitted **with scheduled
   event windows excluded**.
2. **Level**: three EWMAs (fast / medium / slow) of the deseasonalised absolute k-minute returns
   in the context (k = the prompt's step). The forecast level at each lead is a convex mix of
   them whose weights depend on the lead (HAR-like term structure, weights fitted by QLIKE):
   24h crypto: half-lives 1h/6h/8d, weights (.45,.40,.15) at lead 0-1h → (0,.70,.30) at 12-24h;
   equities: (.65,.25,.10) → (.05,.45,.50); 1h: half-lives 10m/1h/8h, (.55,.20,.25) → (.20,.25,.55).
3. **Scheduled events** from the calendar the miner receives (`$SYNTH_BENCHMARK_CALENDAR`):
   variance multipliers per window after the release (median effect of past releases of the
   type, by asset group). E.g. CPI: ×101 (crypto) / ×58 (equities) in the first minute, ×9 / ×6
   for minutes 1-5, ×3.9 / ×2.0 for 5-15; FOMC ×7 / ×4.6 at 30-60 min (press conference);
   earnings ×60-130 for the first two hours after the release, ×3.3 overnight. Context returns
   inside event windows are deflated by the same multipliers before entering the level EWMAs.
4. **Paths**: per-step variance = scale × level(lead) × profile(minute of week) × event
   multiplier. Innovations are Student-t, multiplied by a per-path log-normal variance factor
   (level uncertainty): t(5) and sd 0.35 for 24h; t(7) and sd 0.6 for 1h. Exactly-zero steps at
   the profiled share of stale minutes; prices rounded to the venue's price grid when it is
   coarse (SOL, XRP: ~1 bp ticks). **Weekend steps get their own extra per-path factor**
   (sd 0.5 crypto, 0.8 equity/commodity perps, 24h only): whether a weekend is dead or active
   cannot be read from a weekday context. **Stratified (Latin-hypercube) draws**: every step's
   draws and the per-path factors cover all 1000 strata.

## 3. Experiments (failures included)

Validation: walk-forward windows (A: profiles/events fitted before 2026-08-01, scored
08-05..08-26; B: fitted before 10-01, scored 10-08..10-29; C: 10-29..11-08, after the NY DST
switch; S: a holdout never used for any decision). Paired comparisons on identical prompts;
a change is kept only if it helps on both A and B in most asset cells.

| # | change | result (CRPS change; negative = better) | verdict |
|---|---|---|---|
| 1 | v1: NY-time weekly profile × EWMA, t + mixture, vs starter | -3..-16% in all 18 asset cells | keep |
| 2 | v2: HAR 3-EWMA lead-dependent level (QLIKE-fitted) | A: crypto-24h -0.6%, com-equ -6.7%, 1h -2.0% (all cells) | keep |
| 3 | exactly-zero steps (staleness) | 0.00 .. -0.15% | keep (cheap) |
| 4 | weekend-only level for weekend steps | +0.01..+0.10% | discard |
| 5 | **robust (mean-|r|) level and profile** | com-equ A -0.71%, B -0.45%; crypto -0.04/-0.22% | keep |
| 6 | fast intra-path stochastic volatility | +0.05..+0.5% (all cells worse) | discard |
| 7 | CONTEXT_DAYS 10/14/30 vs 21 | within ±0.05% | keep 21 |
| 8 | lead-growing level uncertainty (log-variance random walk) | sign flips A→B | discard |
| 9 | **calendar events** (walk-forward multipliers) | CPI-spanning 1h prompts -5.7% (A) / -10.4% (B); earnings-spanning com-equ prompts -3.9 / -7.8 / -9.9% (A/B/C); all prompts -0.02..-0.49% | keep (EIA dropped: mixed) |
| 10 | minute-of-hour pattern in the profile | -0.01..-0.04%, every cell same sign | keep |
| 11 | knobs (scale, mixing sd, t dof) around the optimum | ±0.16%, com-equ winners on A flip on B | no change |
| 12 | high-low range in the level measure | ±0.04%, signs flip | discard |
| 13 | **price-grid rounding** (SOL, XRP) | 1h: SOL -0.21/-0.11%, XRP -0.11/-0.06% (A/B) | keep |
| 14 | profile exponent 0.9/1.1; weekend variance ×0.8/×1.25 | flips / worse | no change |
| 15 | slow-heavy weights for weekend steps (Friday-afternoon problem) | crypto +0.3..+1.4% in A, -0.1% in C | discard |
| 16 | mixing sd growing with fast/slow disagreement | ~0 | discard |
| 17 | **stratified draws** (removes the N=1000 ensemble bias, ≈ E|X-X'|/2N) | -0.06..-0.08% in every cell of A, B and C | keep |
| 18 | other EWMA half-lives | the fitted ones are optimal on both windows | no change |
| 19 | trade count as a vol predictor | +0.000..+0.003 R² over the HAR forecast | discard |
| 20 | 1h: wider per-path dispersion, thinner step tails (realised-vol PIT was U-shaped) | A -0.04%, B -0.05% | keep |
| 21 | **weekend-specific per-path factor** (24h) | com-equ -0.04/-0.02/-0.05% (sd 0.5), -0.08/-0.04/-0.11% (0.8); crypto -0.02/-0.02/-0.10% (0.5) on A/B/C | keep (0.8 equ, 0.5 crypto) |
| 22 | the same for weekday off-session steps (equities) | A -0.06%, B +0.05%, C +0.09% | discard |
| 23 | per-path intensity of release reactions | CPI-spanning 1h prompts +1.3% / +2.6% | discard |
| 24 | HAR weights refitted with an absolute-return QLIKE | +0.05 / +0.01 / +0.03% | discard |

Explained anomalies worth sharing:

* **Weekends of the Hyperliquid equity/commodity perps are a separate regime.** Saturday 5-min
  returns are exactly zero 42% of the time for XAU (10-17% for tokenised stocks) and weekend
  activity varies 10× between weekends. A mean-square profile overstated the typical weekend by
  ~2.5× for XAU; the robust profile fixed most of the Saturday gap (+8.5% → +1-2.5% vs field p10).
* **Hot Friday sessions followed by dead weekends** remain our worst com-equ prompts (SP500,
  AAPLX starting Friday 16:00-23:00 NY: +30-40% vs the field median). Every fix I tried helped in
  late October and hurt in August — whether the weekend follows Friday is itself regime-dependent.
* **The best 1h miner wins when the hour turns out calmer than forecast**, and the PIT of the
  realised hourly volatility inside our paths was U-shaped: our paths' realised volatility was
  too concentrated (fixed partly by #20).

## 4. Results

**Official backtester** (`agent/predict.py`, 1000 paths, validator prompt times, the 10 days of
scored prompts before 2026-11-08; in-sample for the champion's artifacts), mean CRPS over assets:

| configuration | starter | champion v8 | change |
|---|---|---|---|
| CRYPTO_24H (5 assets, 1200 prompts) | 4606.76 | 4432.82 | -3.78% |
| COM_EQU_24H (8 assets, 870 prompts) | 3304.04 | 3035.10 | -8.14% |
| CRYPTO_1H (5 assets, 7200 prompts) | 988.40 | 925.40 | -6.37% |

(v7, before the weekend factor: 4436.95 / 3037.12 / 925.40.)

**Out of sample** (walk-forward artifacts), CRPS relative to the best single field miner:

| window | CRYPTO_24H | COM_EQU_24H | CRYPTO_1H |
|---|---|---|---|
| A (08-05..08-26; 1h to 08-15, field without vol term) | +0.1% (v3) | -0.2% (v3) | +0.6% (v3, biased) |
| C (10-29..11-08) | +0.24% (v6) | +0.54% (v6) | +0.39% (v6) |
| **S holdout (09-05..09-26; 1h 09-23..10-03), v7** | **-0.01%** | **+0.00%** | **-0.16%** |

On the untouched holdout v7 matches the best single miner of the field in all three
configurations, and beats the starter by 3.2% / 12.7% / 4.4%. v8 adds the weekend factor
(-0.02..-0.11% on A/B/C).

**Where it stopped.** The last eight experiments (#17-24) moved CRPS by less than ±0.1% or made
it worse; every global knob (scale, mixing, tails, half-lives, HAR weights, profile exponent,
context length) sits at an optimum that two or three windows agree on. Within this model class
the remaining gains are below what a 10-day evaluation can resolve.

## 5. Limitations

* The 24h comparisons rest on ~20-day windows; asset-level differences below ~0.1% are noise
  (they flip sign between windows), so the model is tuned at the group level only.
* Event multipliers are medians over ~10-15 releases per type (pooled over assets); they are
  fitted leave-window-out, but types with few occurrences (FOMC, earnings) stay noisy. The
  evaluation window holds CPI (11-10), PPI (11-13), retail sales (11-17) and NVDA earnings
  (11-18, 21:00 UTC); a release moved after the cutoff is not in the calendar.
* NYSE holidays are not modelled (none in the evaluation window; Thanksgiving is 11-26).
* The Friday-session → weekend transition and the August-type regimes are not solved.
* No cross-asset information (simulate() sees one asset's context) and no sub-minute data.

## 6. With more budget

1. A regime-switching weekend model for the tokenised equities (the remaining worst prompts).
2. Holiday profiles (NYSE_CLOSED / early close) for live robustness.
3. Per-asset event multipliers once more releases accumulate, and an explicit model of the
   earnings-day open (the overnight multiplier is coarse).
4. A learned (GPU) correction of the level forecast on top of the HAR terms — the QLIKE R² of
   0.66-0.78 for the next hour leaves room, but trade counts and ranges added nothing, so the
   inputs would have to be new (other assets' contexts are not available to simulate()).
