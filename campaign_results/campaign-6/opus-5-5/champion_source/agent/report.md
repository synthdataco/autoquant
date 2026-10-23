# A seasonal-EWMA simulator for Synth SN50 — what helped, what didn't

*Campaign-6 (opus-5-5), data cutoff 2026-10-11. Audience: Synth miners.*

## 1. Context

Synth (SN50) asks miners for 1000 simulated price paths per prompt. Prompts come in three configurations:
CRYPTO_24H (BTC, ETH, SOL, XRP, HYPE; 24h at 5-min steps), COM_EQU_24H (XAU, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL,
SPCX, SP500; same format) and CRYPTO_1H (the crypto assets, 1h at 1-min steps). Paths are scored by CRPS on the
returns over many intervals. 24h scores 288 5-min returns, 48 30-min, 8 3-hour, and one absolute 24h price. 1h
scores 1/2/5/15/30-min returns, 12 cumulative "gap" intervals, a 60-min absolute price, and a small
volatility-CRPS term. Measured on our own predictions, the shortest returns dominate the total: 5-min is about
60% at 24h, and 1-min plus 2-min is about 45% at 1h. A miner's job is therefore mostly getting **the scale and
shape of the next 5-minute (or 1-minute) return right, minute by minute**.

The goal here was the lowest CRPS per configuration over the 10 days after 2026-10-11. The constraints:
- only a frozen snapshot of minute OHLCV (no live data);
- a 7-day minute context per call;
- an event calendar;
- the official `synth_lib.backtester`, with an offline field bundle of other miners' scores (09-08..10-11).

## 2. Approach

The model is closed-form. It needs no GPU and runs at about 0.05 s per prompt at 1000 paths.

**Variance per minute** = *weekly profile* × *minute-of-hour shape* × *event multiplier* × *level*.

1. **Weekly profile.** 7×96 15-min slots in New York local time (DST-robust), fitted offline per asset and per
   step (1-min returns for the 1h model, 5-min for 24h). Three details mattered a lot:
   - The level normaliser must span a **full week**. A 3-day window divides the weekend effect away.
   - Use **L1 ratio of sums**: (Σ|r| / Σ√level)² per slot. Squared-return profiles get wrecked by single crash bars,
     and a clipped mean of ratios understates the weekday/weekend contrast.
   - **Recency weighting** with a 21-day half-life. The weekday/weekend variance ratio of equities and gold drifted
     strongly upward through 2026 (SPCX about 5× in May → 28× in Sep).
   - Mon–Fri slots are shrunk halfway to the pooled weekday profile.
2. **Minute-of-hour shape.** 60 multipliers per asset, with mean 1 inside each 15-min slot. Crypto 1-min activity is
   about 1.25× at :00, 1.1–1.2× at :15/:30/:45, and about 0.8× in the minutes before them.
3. **Scheduled releases** (CPI, NFP, FOMC, PPI, GDP, PCE from `$SYNTH_BENCHMARK_CALENDAR`). Windows of
   [-30, 0, 2, 5, 15, 30, 60, 120, 240] min around the release. Each multiplier is the squared L1 variance ratio
   versus the same clock time on non-event days, fitted on releases **before 2026-07-01**. It is raised to a power
   of 1.75 at 1-min steps and 1.5 at 5-min, because a median ratio understates the mean variance of a release
   minute. The profile is fitted with those minutes masked out, so nothing is counted twice. Weekly EIA petroleum
   data is *not* an event here: it always lands in the same slot, so the weekly profile carries it.
4. **Level.** Two gap-aware EWMAs of deseasonalised squared returns from the context:
   - The returns use the forward-filled last price, for estimation only; NaNs are dropped, never interpolated.
   - Half-lives are 30 min and 2 days for the 1h model, 2 h and 2 days for 24h. Squared returns are winsorised at
     30× the median.
   - The forecast decays from the short EWMA to the long one with a 60-min (1h) or 240-min (24h) time constant.
5. **Adaptive weekend (non-crypto).** The forecast weekend variance is multiplied by ρ. ρ is the ratio of the last
   weekend's deseasonalised level in the context to the weekdays' level.
6. **Paths.**
   - Increments are Student-t (ν = 5, unit variance).
   - Each path gets a lognormal volatility multiplier (η = 0.2: vol-of-vol, which fattens the longer intervals).
   - There is an atom at zero with the per-slot probability that the price did not change (XAU has 11% unchanged
     5-min steps, SOL 10% of 1-min steps), and the moving steps carry the full variance.
   - The paths are a martingale in price.
   - Seeded from `SYNTH_BENCHMARK_SEED`.

## 3. Experiments (failures included)

All numbers are mean CRPS from the official backtester (latest formula) on prompts the validator actually scored.
Windows: W_A = 09-12..09-26 and W_B = 09-26..10-11 for 24h; 09-22..10-11 or 10-01..10-11 for 1h (after the
09-22 formula change). Profiles are fitted on data before 09-08 unless stated. A change was kept when it helped on
both windows and in most asset-window cells.

| # | Change | Result | Verdict |
|---|---|---|---|
| 1 | Starter (log-normal, 7-day std) → seasonal EWMA v1 | big gain (see §4) | keep |
| 1b | Profile fixes: week-long normaliser, L1 ratio of sums, 21-day recency | e24 W_B 1772.8 → 1720.6 | keep |
| 2 | Atom at zero | e24 W_B 1720.6 → 1718.9 | keep |
| 3 | Global scale ±10%, ν 3/10, η 0/0.4 | all within ±0.3%, signs disagree | defaults |
| 4 | Fixed weekend scale 0.5 for equities | W_B −0.5% but W_A +0.25% | discard |
| 4b | Adaptive weekend ratio ρ^1 (equities) | W_A 1819.2 → 1818.0, W_B 1718.9 → 1710.3, 13/16 cells | keep |
| 5 | Same for crypto | c24 W_B −0.3% but W_A +0.3% on all 5 assets; c1 worse | discard |
| 6 | Event multipliers (pre-July table) | FOMC prompts −1%, CPI −0.3…−0.5% at 24h; c1 CPI −5.6%, FOMC −6.5%, PPI −4.4% (100% wins) | keep |
| 6b | Power on the multipliers | Sep: p=2 best for c1 (CPI −12% vs p=1); Jul–Aug leave-out: p=1.5 ≈ p=2; p=3 overshoots | 1.75 at 1h, 1.5 at 24h |
| 6c | EIA multiplier for WTIOIL | worse on 10/10 out-of-sample releases | removed (profile carries it) |
| 7 | 1h level: HL_S 30, TAU 60 | 690.7 → 690.3 (others: 10/15/60 min, HL_L 1 day, WINS 5/10 all worse) | keep |
| 7b | Parkinson range (high/low) blended into the EWMAs | 690.4 → 690.8 (50%), 692.1 (100%) | discard |
| 8 | 24h level half-lives and decay | all within ±0.1% on both windows | defaults |
| 8b | Per-path persistent drift (extra long-horizon variance) | flat / slightly worse on both windows | discard |
| 9 | Minute-of-hour shape | c1 −0.02%, c24 −0.04%, e24 −0.06% | keep |
| 10 | HAR-style multi-half-life level, fitted | c1 QLIKE −1% (≈0.1% CRPS), 24h worse out of sample | no change |
| 11 | Fresher profile (cutoff at the window start vs 18 days before) | c24 −0.2%, e24 −0.3%, c1 −0.1% | served profile fitted on all data to 10-11 |

Lessons that generalise:
- **Price a residual before chasing it.** CRPS is flat near the right scale: a ±20% σ error costs about 1%. A ±40%
  variance pattern on single minutes (the minute-of-hour effect) was worth under 0.1%. The big wins were the ones
  that fix *large* σ errors on *identifiable* minutes: weekends of equities (σ off by 2–3×) and release minutes
  (variance off by 10–25×).
- **The calendar is the only information the EWMA cannot have.** Range estimators added nothing over a 30-minute
  EWMA of squared returns, and neither did a fitted HAR. Release multipliers gave 5–12% on the 1h prompts that
  contain a CPI/FOMC/PPI release.
- **One window lies.** A fixed weekend scale and the crypto weekend adjustment each looked like −0.3…−0.5% on one
  window and flipped sign on the other.
- **Validate event effects out of sample.** Our table was fitted on releases before July and tested on Jul–Sep.
  The best power differed between the Jul–Aug and Sep releases, so we chose the midpoint.

## 4. Results

Out of sample: profiles fitted on data before each window starts, event table on releases before July, 1000 paths,
official backtester (latest formula), all validator-scored prompts of the window.

| Configuration | Window | Champion | Starter | Change | vs best single miner (per asset) |
|---|---|---|---|---|---|
| CRYPTO_24H | 09-26..10-11 | 3533.1 | 3697.8 | −4.5% | +0.4…+1.3% |
| COM_EQU_24H | 09-26..10-11 | 1704.5 | 2025.6 | −15.8% | −0.03…+0.7% (XAU and SP500 1st, TSLAX and WTIOIL 2nd) |
| CRYPTO_1H | 09-22..10-11 | 807.4 | 853.6 | −5.4% | +0.6…+1.1% |

The served artifact (`agent/artifacts/season.json`) is fitted on all data to 2026-10-11. Backtests with it, through
`agent/predict.py` (1000 paths, latest formula), are in-sample for the profile, so they are lower: CRYPTO_24H
3494.1 vs starter 3698.2, COM_EQU_24H 1689.2 vs 2025.5, CRYPTO_1H 801.8 vs 853.5 (`agent/results/final_*.txt`).

In the evaluation window (10-12..10-21) the calendar has CPI (10-14 12:30 UTC) and PPI with retail sales
(10-15 12:30 UTC), and no FOMC or NFP.

## 5. Limitations

- **Short validation span.** The field bundle starts on 09-08, so there are two 24h windows and one 1h window
  (after the 09-22 formula change). Several kept changes are worth 0.1–0.5%, which is near the noise between
  windows.
- **Few events.** The event table rests on one year of releases (8–12 per type). The power was chosen on 6
  out-of-sample releases. GDP/PCE multipliers hurt crypto 1h out of sample; they are not in the eval window.
- **The profile is static** for the 10 eval days. The weekday/weekend ratio drifts, which is partly handled by the
  adaptive weekend term.
- **No cross-asset information** (`simulate()` sees one asset), and the benchmark context ends 2 minutes before the
  request. Part of the 1h gap to the best miners is likely information we do not have, not the model.
- **US market holidays are not modelled**; none fall in the eval window.

## 6. With more budget

1. Per-class and per-release-type event multipliers fitted on several years of releases, with leave-one-release-out
   tuning of the power.
2. An explicit overnight-gap jump at the 09:30 ET cash open for the tokenised stocks: their 24h returns were
   under-dispersed in Sep–Oct.
3. A holiday-aware equity profile and a proper half-day calendar.
4. An online, faster-adapting profile (e.g. a weekly EWMA of the slot ratios computed from the context itself)
   instead of a frozen artifact.
5. A longer realised-path archive, to backtest the 1h model across more regimes with the official harness.

Code: `agent/modeling.py` (model), `agent/fit_season.py` (profiles), `agent/evalrun.py` (batch backtests through
`synth_lib.backtester`), `agent/analysis/*` (diagnostics), `agent/journal.md` (every iteration with numbers).
