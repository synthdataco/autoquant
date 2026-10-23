# Calibrating a price-simulation fan for Synth: what actually moves CRPS

**Audience**: Synth miners. **Objective**: lowest CRPS (lower is better) across
CRYPTO_24H, COM_EQU_24H and CRYPTO_1H, measured with the official `synth-lib`
backtester at 1000 paths per prompt. Everything below was judged on CRPS only —
never on field-relative position, which re-bases tiny CRPS differences and is far
too noisy to optimise.

## Context: what the score actually is

Reading `synth.validator.crps_calculation` before touching a model paid for itself:

- **24h competitions** sum CRPS over return changes at 5 min (288 points), 30 min (48),
  3 h (8), plus the terminal absolute price. The short-horizon marginals dominate by count.
- **CRYPTO_1H** sums return CRPS at 1/2/5/15/30 min, the terminal price, twelve
  "return from start" gaps, and a volatility CRPS (λ = 5.25 over 60/15/5-min block
  realised vols). Measured on my own fans, **the vol term is only ≈1% of the 1h total** —
  it looks important in the config and is nearly irrelevant in the score. The 1h
  competition is a price-marginal problem like the others.

## Approach: one model, three ingredients

The champion (`agent/modeling.py`) simulates 1-minute log-returns and sub-samples to
the scoring step, so one code path serves both horizons:

1. **Seasonal volatility profile per asset** — a (weekday × hour) multiplier fitted
   offline on the snapshot and shipped in `artifacts.json`. For the eight
   commodity/equity perps the profile is anchored to **America/New_York** time, because
   their vol follows the US session (UTC anchoring smears the daylight-saving shift).
2. **Stochastic vol level from the 7-day context** — returns are deseasonalised,
   outlier-capped, gap-scaled (a return spanning k untraded minutes is divided by √k),
   and fed to fast (6 h) and slow (3 d) EWMA variances. Forward, log-vol follows an
   AR(1) with a **2-day half-life** starting near the fast level and mean-reverting to
   the slow level.
3. **Student-t shocks (ν = 5), zero drift**, plus a calm global width calibration
   (×0.90 for 24h, ×0.95 for 1h) and small volatility multipliers around scheduled
   CPI/FOMC/macro releases read live from the event calendar.

Runtime is ~0.2 s per 1000-path prompt; degenerate contexts (all-NaN-but-one,
constant prices, unknown assets) are unit-tested not to crash.

## Experiments (failures included)

| Change | Result (official CRPS) | Verdict |
|---|---|---|
| Starter (log-normal, constant vol) → seasonal SV-t fan | BTC 24h −6%, XAU −17%, BTC 1h −5.5% | foundation |
| Remove seasonality | +27% BTC, +6% XAU | seasonality is the biggest single lever |
| Remove stochastic vol | +0.9% | small, keep |
| NY-anchored profiles (COM_EQU) | −0.11% (24h, both windows) | keep |
| Vol half-life 6 h → 48 h | −0.27% (24h) | keep — fattens the terminal marginal, which PIT showed under-dispersed (sd(z)≈1.25) |
| Global width ×0.90 (24h) | −0.2…−0.4% per window, 13/13 assets in Oct | keep — the surprise lever, found by a per-horizon width diagnostic, confirmed out-of-window (July, fast proxy) |
| ν, fast/slow blend, vol-of-vol fine grid | ±0.1% | at the noise floor; defaults kept |
| Drift from trailing returns | +2…+7% worse | **rejected** |
| Jump mixture | worse | **rejected** |
| Block-bootstrap shocks (empirical residuals) | worse at every block length | **rejected** — parametric t + SV already spans the shape |
| 168 hour-of-week profiles | mixed sign per asset | **rejected** — fits noise |
| Event multipliers (CPI×1.45, FOMC×1.8, decaying) | 0.00% measured | kept anyway: the pooled event study is unambiguous (post-event vol ×1.3–1.9), the scored windows simply contained few such prompts |

Two methodological notes worth stealing:

- **My fast screening proxy disagreed with the official metric twice** (it missed the
  width gain and overstated tuning gains). It was still useful for killing bad ideas
  (drift, jumps, bootstrap) cheaply — but every promotion was decided by
  `agent/predict.py` on two disjoint windows, counting how many asset×window cells
  improved, not just the mean.
- **Ablations beat intuition**: the vol-CRPS term, the gap-return correction for thin
  assets, and per-asset tail indices all *felt* important and measured ≈0. The pooled
  field data say the best miner on a prompt is only ~2% below the median miner — this
  game is won in the last fractions of a percent of calibration, and most of those
  fractions are vol seasonality and honest width.

## Results

Official backtester, 1000 paths, validator prompts, two 3-day windows
(per-cell numbers in `agent/matrix_v6.log`; champion 24h width 0.90 measured equal
or better than the 0.95 in those cells):

| Config | Mean CRPS (Sep window) | Mean CRPS (Oct window) |
|---|---|---|
| CRYPTO_24H | 3829 | 3648 |
| COM_EQU_24H | 1519 | 1837 |
| CRYPTO_1H | 894 | 726 |

vs the starter on identical prompts: BTC 24h 2329 → 2155 (−7.5%), XAU 1383 → 1160
(−16.2%), BTC 1h 539 → 509 (−5.6%). Gains by ingredient, roughly: seasonality
−6…−17%, vol persistence −0.3%, NY anchoring −0.3%, width −0.3%, tuning −0.1%.

## Limitations

- Only two official validation windows exist (the offline bundle's scored prompts
  start 2026-09-07), each 3 days × 72–432 prompts per asset. Differences below
  ~0.15% are not resolvable; several kept choices sit at that floor and were kept
  for sign-consistency across cells rather than proven effect size.
- Seasonal profiles are fitted on ≤14 months (SPCX: <5). A regime change in session
  patterns would stale them; they are static artifacts, not adaptive.
- The model is a single-asset fan: no cross-asset vol spillover (the context only
  ever contains one asset — this is a serving constraint, not an oversight).
- Event multipliers were never validated on a scored window containing a CPI/FOMC
  prompt; they rest on the pooled event study alone.
- Equity/commodity fans ignore the cash-market close entirely (vol profile ≈0.2×
  overnight, but never zero, matching the perp's own trading).

## With more budget

1. A per-asset, per-horizon width map fitted by direct CRPS minimisation on rolling
   origins (the machinery exists; it needs more windows than the bundle offers, so
   pair it with a cadence-grid evaluator that scores against the price store alone).
2. Term-structure width: the diagnostic wants a narrower body and wider terminal
   than one level shift gives — implementable as a horizon-dependent vol path.
3. Revisit vol-level estimation with a range-based (high-low) estimator on the
   Hyperliquid assets, whose minute ranges are sampled from 1-second closes.
4. Longer event study per asset (earnings for the tokenised names) with strict
   leave-one-event-out validation.
