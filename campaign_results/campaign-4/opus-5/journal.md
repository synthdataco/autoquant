# Research journal — campaign-4 (opus-5)

Objective: minimise validator CRPS on CRYPTO_24H, COM_EQU_24H, CRYPTO_1H.

---

## Iter 0 — Reading the metric before writing a model

**What the metric actually is** (`synth.validator.crps_calculation`, `competition_config`):
the score is a **sum of univariate CRPS terms**, each on a *relative price change over an
interval*, expressed in basis points (`×10_000`). There is no multivariate term: the joint law of
a path only matters through the **marginals of the aggregated returns**.

Counting the terms, and using CRPS ≈ c·(scale of the term) for a well-calibrated forecast,
gives the budget each competition really spends (σ = one-step return vol in bp):

*24h competitions* (289 pts, 5-min step):

| interval | terms | scale | ≈ share |
|---|---|---|---|
| 5min  | 288 | σ      | 61% |
| 30min |  48 | σ√6    | 25% |
| 3hour |   8 | σ√36   | 10% |
| 24hour_abs | 1 | σ√288 | 3.6% |

*crypto-1h* (61 pts, 1-min step): 1min 60·σ, 2min 30·σ√2, 5min 12·σ√5, 15min 4·σ√15,
30min 2·σ√30, 60min_abs σ√60, plus **12 `*_gaps` terms** = cumulative return from point 0 to
point 5,10,…,60 min ≈ 67·σ (the single biggest block, ~29%), plus a **volatility CRPS**
(weights sum to 5.25) that is worth <1% of the price CRPS unless the vol forecast is badly wrong.

**Consequence for modelling**: ~86% of a 24h score is decided by the *marginal distribution of
the 5-minute return at each of 288 future steps*. This is a conditional-volatility problem, not a
path-shape problem. Getting the per-step scale (with its intraday seasonality) and the tail shape
right is nearly the whole game.

**Anchoring**: `context_end` = `start − (ISSUANCE_LEAD + 60 s)` = start−180 s for 24h, start−120 s
for 1h, while `real_prices[0]` is the close of the bar labelled `start` (= price at start+60 s).
So our anchor is 3 min (resp. 2 min) stale. Because every non-`_abs` interval scores *relative
changes*, the stale anchor cancels; it only adds √(86580/86400)−1 = 0.1 % of extra dispersion to
the one `24hour_abs` term. Not worth modelling.

---

## Iter 1 — A fast, faithful evaluation loop

**Hypothesis**: `agent/predict.py` (official `backtest()`) is too slow to iterate with — it makes
two Synth-API round trips per asset and needs the live field. A local harness that reuses the
*same* CRPS function and the *same* realized-path assembly will agree with it to within sampling
noise, and run ~100× faster.

**Change**: `agent/harness.py` + `agent/evaluate.py`.
- realized paths: `synth_lib.backtester.preparation._slice_real_prices` on the local minute store
  (reindexed to a gapless 1-min grid, close column only — exactly what `download_price_data`
  hands the slicer), then `_fill_gaps_from_realized_paths` from the offline realized-path cache.
- CRPS: `calculate_total_score_for_miner`, imported through `synth_lib.backtester.scoring`.
- prompts on a fixed grid, cached to pickle; 13 worker processes.

**Result**: 545 prompts × 1000 paths × 5 assets scored in **16 s**.
Cross-check against the official backtester, BTC, `--eval-end 2026-09-10 --days 1`, starter model:

| | CRPS |
|---|---|
| official `backtest()` (25 prompts, starts 2026-09-08 23:32 → 2026-09-09 23:xx) | **2621.9** |
| harness, starts 2026-09-08 00:00→09-09 00:00 | 2425.5 |
| harness, starts 2026-09-09 00:00→09-10 00:00 | 2654.6 |
| same single prediction file, official vs harness, same start | 2743.4 / 2740.9 |

**Verdict**: KEEP. Two bugs found and fixed while building it, both worth recording:
1. `ASSET_COMPETITION` keyed on the asset alone — BTC is in *both* CRYPTO_24H and CRYPTO_1H, so
   dict comprehension order silently scored 24h prompts with the 1h interval set (+80 % CRPS).
   **The competition is keyed on (asset, time_length), never on the asset.**
2. `_slice_real_prices` takes `.iloc[:, 0]`. Hand it a full OHLCV frame and you score against
   **open**, not close (−2.5 % CRPS here — small enough to not look like a bug). Pass
   `frame[["close"]]`.

**Lesson**: the validator's prompt start times are *irregular* (≈1/h at random minutes, e.g.
00:53, 01:18, 02:13), and `backtest()` matches a prediction within ±30 min. A grid-based harness
is therefore not prompt-identical to the official run — it is a faithful *estimator* of the same
quantity, which is what an iteration loop needs. Final numbers still go through `predict.py`.

---

## Iter 2 — Where the CRPS actually is, and what could possibly move it

**Hypothesis**: before tuning anything, find out which part of the score is worth attacking.

**Change**: term-budget analysis of the scoring config (`diagnose.py`), plus *oracle* models that
are handed the answer — `oracle_level` knows the realized average step-volatility of the very
window it predicts, `oracle_path` knows the realized |return| at every future step. Both are
lookahead by construction, never used in `modeling.py`, and exist only to bound the prize.

**Result**

The validator score is a **sum of univariate CRPS terms on relative price changes over
intervals**. There is no multivariate term: the joint law of a path matters only through the
marginals of aggregated returns. For a 24h prompt the budget is

| interval | terms | share |
|---|---|---|
| 5 min | 288 | ≈61 % |
| 30 min | 48 | ≈25 % |
| 3 hour | 8 | ≈10 % |
| 24 hour (abs) | 1 | ≈3.6 % |

So ~86 % of a 24h score is *the marginal distribution of the 5-minute return at each of 288
future steps*. For crypto-1h, the 12 `*_gaps` terms are ≈29 % and the volatility CRPS ~1 %.

Oracle headroom over the then-best model: crypto-24h ≈6 % (level alone 1.2 %), com-equ-24h ≈11 %
(level 3–4.4 %), crypto-1h ≈7.5 % (level 4.9 %).

**Verdict**: attack the per-step volatility, nothing else.

**Lesson**: a scoring function that is a sum of univariate CRPS terms does not reward realistic
*paths*. Cross-sectional correlation, jump-diffusion realism and momentum are all unpriced. Time
spent on them is time not spent on the only thing priced: the width of each step's distribution.

---

## Iter 3 — The scale trap

**Hypothesis**: calibrating the overall width better will pay.

**Change**: swept a global `scale` multiplier, the innovation law (normal / Student-t / empirical
bootstrap), `nu`, and per-path vol-of-vol.

**Result**: Student-t(4) beats normal by ≈1 %, and that is essentially the whole prize from
distribution *shape*. Scale is remarkably flat: ±20 % costs ≈1 %.

The flatness is not an accident. For a Gaussian forecast of scale σ against truth of scale 1,

    E CRPS(σ) = sqrt(2/π)·sqrt(1+σ²) − σ/sqrt(π),  minimised at σ=1 with value 1/sqrt(π)

and expanding gives **E CRPS(σ)/E CRPS(1) ≈ 1 + 0.25·(log σ)²**. A 5 % scale error costs 0.06 %;
a 20 % error costs 1 %. The empirical sweep matches this to two digits.

**Verdict**: stop tuning constants.

**Lesson**: this identity reframes the whole campaign. A *constant* scale error is nearly free;
the entire recoverable CRPS is **Var(log σ̂ − log σ_true)**. Every subsequent experiment was
judged on whether it reduced that variance. It also explains the oracle numbers: sd(log vol) is
≈0.48 for BTC, and 0.25·0.48² ≈ 5.8 % — almost exactly the oracle gap.

Note the expansion is symmetric but the exact loss is not: too *wide* costs linearly without
bound, too *narrow* saturates at +41 %. The CRPS-optimal forecast therefore sits slightly below
`exp(E[log σ])`, which is why every fitted multiplier below comes out under 1.

---

## Iter 4 — Time-of-week seasonality: a failure, then a fix

**Hypothesis**: the venues never close but the underlying markets do, so per-step volatility has
a large deterministic time-of-week component that a constant σ cannot express.

**Change**: `seasonality.py` fits a 336-bucket (half-hour) time-of-week profile per asset and
step size, committed as an artifact; `sv_model` deseasonalises the context returns, forecasts the
deseasonalised level, and re-seasonalises each future step.

**Result, first attempt: WORSE.** XAU +10.05 %, AAPLX +2.96 %, TSLAX +2.48 %, NVDAX +2.28 %.

The estimator was the bucket **median** |return|. In a thin market's quiet hours most 5-minute
returns are *exactly zero* — no trade — so the median collapses to ~0 while the CRPS-relevant
scale does not. The profile made the model absurdly narrow overnight, and CRPS charges heavily
for a forecast that is too narrow when the price does move.

**Change 2**: winsorised **RMS** instead (clip at 6 robust sigmas of the whole series — XRP's raw
return kurtosis on this snapshot is ~7000, so a plain RMS is set by single prints). Dynamic range
tightened from 0.25–3.07 to 0.38–2.26 (NVDAX) and 0.25–2.28 to 0.32–1.74 (XAU). Added a damping
exponent `seas_gamma` so the amplitude is a tunable, not a belief.

**Result, second attempt** (dev window, 13 assets, 1885 prompts, vs no seasonality):

| γ | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| mean CRPS | −0.85 % | −1.49 % | −1.87 % | −1.92 % |

with com-equ gaining most (NVDAX −3.5 %, TSLAX −3.8 %, GOOGLX −3.2 %). γ=0.85 adopted: the mean
is flat between 0.75 and 1.0 but XAU degrades sharply at 1.0.

**Verdict**: KEEP — the single largest improvement of the campaign.

**Lesson**: when a feature makes things worse, the feature is usually right and the *estimator* is
wrong. Also: any statistic of a thin market's returns must survive the fact that "no trade" and
"no move" are the same observation in this data.

---

## Iter 5 — Conditional vol dynamics: mostly a dead end

**Hypothesis**: EWMA half-lives, the blend weights, mean-reversion speed and vol-of-vol are all
worth tuning.

**Change**: swept `revert_halflife` ∈ {24,48,96,192,384}, three half-life triples, three weight
vectors, `vol_of_vol` ∈ {0.2,0.35,0.5}, `winsor` ∈ {2.5,3.5,6}, and an L1 (mean-absolute) variance
estimator as an alternative to the RMS.

**Result**: `revert_halflife=96` steps is the optimum, worth −0.56 % over no reversion. *Every
other knob moved the mean by less than 0.1 %.* The L1 estimator neither helped nor removed the
per-asset spread it was meant to explain.

**Verdict**: keep `revert_halflife=96`; freeze the rest. Consistent with Iter 3 — these knobs move
the *level*, and the level is nearly free.

---

## Iter 6 — Fitting the volatility forecast (rejected out of sample)

**Hypothesis**: Iter 3 says the prize is Var(log σ̂ − log σ). That is a regression target, so
regress it instead of hand-picking EWMA weights.

**Change**: `volfit.py` regresses `log σ_future` on the log EWMA vols of the deseasonalised
context at five half-lives, per asset, for five horizon knots, so the fitted cumulative-variance
curve V(h) can be differentiated into a per-step term structure. Fitted only on data preceding
each validation window.

**Result**: in-sample R² 0.36–0.73 (crypto 0.55–0.73). Measured against realized vol on the dev
prompts, the fit genuinely beats the EWMA blend on crypto (log-error sd 0.281 vs 0.314 for BTC)
and loses on the thin equities. But scored on CRPS it was **worse everywhere**, +1.7 % initially.

Three distinct bugs, each instructive:

1. `np.interp` clamps below its first knot, so the cumulative variance was flat over the first
   steps — i.e. **zero dispersion** at steps 1–2. CRPS punishes that savagely. Fixed by
   log-log extrapolating below the first knot.
2. The "term structure" was almost entirely **small-sample log bias**: the target is
   `log(RMS of h returns)`, and log is concave, so E[log RMS_3] sits ~45 % below log σ for
   fat-tailed returns while E[log RMS_288] is within 0.3 %. Fitting each horizon separately
   manufactures a steep upward slope that is pure estimator artefact. Fixed by measuring the bias
   on non-overlapping blocks of the same series and subtracting it (normalised to 0 at the
   longest horizon, so only the *shape* is corrected).
3. With `revert_halflife=96` over a 288-step path, most of the path has already reverted to the
   long-run EWMA level — so an A/B that "swapped in the regression" was barely using it. The
   honest comparison is against a flat forecast.

After all three fixes, at each model's *individually optimal* multiplier the fitted and EWMA
levels tie (mean excess loss 3.40 % vs 3.37 %). Out of sample on the later window the regression
was **worse** on exactly the assets it was supposed to help (HYPE +1.12 %, XRP +0.74 %, SOL
+0.53 %).

**Verdict**: REJECT. `volfit.py` stays in the repo as a measurement instrument — its residual sd
is what tells us 3.6 % (24h) and 2.5 % (1h) remain on the table — but the champion does not use it.

**Lesson**: a forecast that is better by its own loss function is not necessarily better by CRPS,
and a model fitted on one regime can lose the gain entirely two months later. The only evidence
that counts is a window the fit never saw.

---

## Iter 7 — Per-asset width: the part that did generalise

**Hypothesis**: every sweep kept rediscovering that the CRPS-optimal width differs by asset —
XAU wanted ~0.70× what the vol estimator said, WTIOIL ~1.0×. Fit it rather than hand-code it.

**Change**: `calibrate.py` fits one multiplier per asset on a window that *precedes* both
validation windows (2026-05-20 → 06-20), by parabola through five scale points.

**Result**

| | dev (Jul 5–23) | test (Aug 26 – Sep 13) |
|---|---|---|
| all 13 assets mapped | −0.21 % | **+0.13 %** |
| com-equ only | −0.21 % | **−0.13 %** |

The crypto multipliers were noise and cost real CRPS out of sample (HYPE +0.79 %, XRP +0.78 %).
The com-equ multipliers were structural: refitted four months later on 2026-08-01 → 09-13 they
barely moved — GOOGLX 0.746→0.741, NVDAX 0.761→0.755, SPCX 0.831→0.835, XAU 0.764→0.773 — and
they transferred (TSLAX −1.09 %, XAU −0.62 %, GOOGLX −0.48 % on test).

**Verdict**: KEEP for the eight com-equ assets (shipped value = mean of the two independent
fits); crypto keeps a single flat 0.90.

**Lesson**: "does it help out of sample" and "is the parameter stable when refitted on disjoint
data" are two different questions, and the second one is the one that predicted which half of
this map to keep.

---

## Iter 8 — crypto-1h needs no special model

**Hypothesis**: a 60-minute horizon at 1-minute resolution wants faster vol dynamics than a
24-hour horizon at 5-minute resolution.

**Change**: swept half-lives from (2,10,60) min up to (120,720,5760) min, pool length 2.8 d vs
7 d, empirical vs Student-t innovations, vol-of-vol, and reversion speed — on both windows.

**Result**: *faster* dynamics are clearly worse (the (2,10,60) blend costs +0.41 %). Slower
half-lives, a 7-day pool and empirical innovations all looked good on dev (−0.43 % combined) and
**none of it transferred** — on test every variant was within ±0.08 % of the plain configuration,
with the unchanged one marginally ahead.

**Verdict**: ship one configuration for all three scoring configurations. The 24h settings score
−1.20 % (dev) and −2.98 % (test) against the i.i.d. Student-t reference at 1h without any
1h-specific tuning.

**Lesson**: when two disjoint windows disagree about a knob, the knob is noise. Preferring the
simpler configuration is not a tie-break, it is the conclusion.

---

## Iter 9 — Making the official evaluation actually run

**Hypothesis**: the campaign is done; produce the official numbers.

**Change**: ran `agent/predict.py` at 1000 paths over 7 days ending 2026-09-13, for all 18
asset/configuration combinations, for both the champion and the repo's starting-point model.

**Result**: two infrastructure defects, both of which produced *plausible* output rather than an
error, and both worth recording because either one alone would have invalidated the report.

1. **`generate()` aborts on any feed gap.** It builds the context with
   `MinutePriceStore.load_range`, which raises when a single `close` is NaN. Every com-equ asset
   and HYPE has gaps, so 14 of the 18 runs died with `Local store has 3 missing minute prices`
   — and because the driver script piped stdout through `grep "^summary:"`, the tracebacks were
   swallowed and the script reported `DONE` having scored four assets. Fixed by reading the
   partitions leniently and reindexing onto a gapless minute grid, which is also what makes
   `[-CONTEXT_MINUTES:]` mean seven *calendar* days. The `backtest()` call is untouched — the
   metric is still entirely synth-lib's.

2. **Concurrent runs of different models overwrite each other.** `PREDICTIONS_DIR` is a fixed
   path, never cleared, and filenames are `{start}_{asset}_{time_length}.json` — no model
   identity. I had launched the champion and the baseline in parallel to save wall-clock; they
   wrote to the same files and `backtest()` scored whichever landed last. The numbers looked
   entirely reasonable. Discarded them, gave each model its own directory, reran.

**Verdict**: results below are from the corrected, isolated runs.

**Lesson**: never filter a subprocess's stderr through a grep for the success line. And two
processes that share an output directory are one process.

---

## Iter 10 — The snapshot edge, and the final numbers

**Hypothesis**: with the two defects from Iter 9 fixed, the 18 official runs per model complete.

**Change**: reran both streams in isolated prediction directories, 1000 paths, `--days 7
--eval-end 2026-09-13`.

**Result**: all 13 twenty-four-hour runs completed on both streams. All five crypto-1h runs
failed, on both, with

```
OSError: [Errno 30] Read-only file system: 'market_data/prices/ETH/1m/date=2026-09-15.parquet'
```

`backtest()` step 5 fetches minute prices over `[min(start_time) − 2h, max(start_time) + 26h]`,
and `download_price_data` walks that range a *whole day at a time*: any day partition that is
absent sends it to `MinutePriceStore.ingest_range`, which fetches and then tries to persist.
A 24-hour prompt cannot be scored until 24 h after it starts, so with an anchor of 2026-09-13 the
latest 24h prompt is from 09-12 and `+26h` lands inside the snapshot. A *1-hour* prompt is scored
an hour after it starts, so the 1h prompt set reaches 09-13 22:00, `+26h` crosses midnight into
09-15, and the store reaches for a day that does not exist here.

Moved the crypto-1h leg to `--eval-end 2026-09-12` (`agent/official_eval_1h.sh`), which keeps the
whole requested price range inside the snapshot. Same 7 days of prompts, shifted one day earlier.

**Verdict**: the 1h leg is scored over 2026-09-05→09-12 and the 24h legs over 2026-09-06→09-13.
Both models are scored on identical windows, so the comparison is unaffected.

**Disclosure**: those five failures mean synth-lib itself issued a price request covering
2026-09-15 before the write failed. Nothing came back into the process — every one of the ten runs
raised and produced no output — and no result, parameter or decision in this campaign derives from
it. I record it because the rule is about the request, not just the answer. The champion was
already frozen at commit `a9a8d91` before this happened.

**Lesson**: the edge of a data snapshot is not the last timestamp in it; it is the last timestamp
minus whatever lookahead the *scoring* code adds. Here that is 26 hours, and it bites the short
horizon harder than the long one — the opposite of the intuition.

---

## Iter 11 — Range estimators: the textbook says yes, the data says no

**Hypothesis**: the model estimates volatility from closes only and throws the minute high/low
away. Parkinson and Garman-Klass use the whole bar and are several times more efficient for the
same window; §2 says estimation noise is part of `Var(log sigma_hat - log sigma)`, so a more
efficient estimator should be a direct hit on the residual.

**Change**: `agent/eda_range.py`. Aggregate 1-minute bars to 5-minute high/low/close, build three
per-step variance proxies (close², Parkinson, Garman-Klass), deseasonalise each by the committed
profile, run each through the champion's own EWMA blend, and compare the **centred** log-error sd
against realized 288-step vol. Centring is deliberate: the level is free (§2), so only the spread
of the error is allowed to count.

**Result**: worse, nearly everywhere.

```
d_park_%   -0.161      (CRPS points the swap would be worth; negative = worse)
d_gk_%     -0.272
d_mix_%    -0.073
```

Only XRP, HYPE, WTIOIL and SP500 improve at all, and never by much. Garman-Klass — the more
efficient of the two — is the worse of the two.

**Verdict**: rejected, no CRPS run needed.

**Lesson**: Parkinson's efficiency result assumes the high and low of a continuously observed
Brownian path. What is actually in this store is the extremes of *however many prints happened*,
so the estimator's variance is set by trade intensity, not by the price process — and trade
intensity is precisely what varies most across the week in these assets. The efficiency gain is
real for a liquid continuously-quoted instrument and evaporates here. That GK, which leans hardest
on the range, is worst of all is the signature of the explanation being right.

---

## Iter 12 — The seasonal exponent was confounded with the level

**Hypothesis** (report §8 item 4): the per-asset width multiplier should be a function of how busy
the market is at that step rather than a constant. Its simplest one-parameter form already exists
in the model — `width = scale * v * s^gamma`, so `gamma - 0.85` *is* an activity-dependent
multiplier — so start by asking whether the shipped `gamma = 0.85` is right.

**Change**: fit `gamma` per asset. Then, when that produced nonsense, a 2-D grid over
`(gamma, scale)`.

**Result**, in three stages, and the middle one is the point:

1. Fitting gamma per asset on 2026-05-20..06-20 sent every one of the 13 assets to the top of the
   grid. That window lies *inside* the span its profile was fitted on, so the profile is in-sample
   there. The exponent measures how far to trust the profile; it cannot be fitted where the
   profile already knows the answer. Discarded.

2. Refitting on the two validation windows (causal profile) *still* wanted gamma >= 1 everywhere —
   the opposite of the shipped 0.85. The resolution is that a 1-D sweep of gamma is not a sweep of
   shape: the profile is normalised to mean 1, so by Jensen `E[s^gamma]` grows with gamma, and
   raising gamma silently widens the model. With `scale` re-optimised at each gamma the effect
   mostly disappears — and what is left changes sign:

   | gamma | dev (min over scale) | test (min over scale) |
   |---|---|---|
   | 0.85 | 2857.3 | 2637.7 |
   | 1.15 | 2852.3 | 2624.0 |
   | 1.35 | ~2852 | ~2620 |
   | 2.00 | 2887.5 | 2630.1 |

   An interior optimum near 1.35, not 0.85. The original sweep that chose 0.85 held `scale` fixed
   and was reading the level, not the shape.

3. gamma was now chosen using both validation windows, so the A/B was run on **2026-07-24..08-25**,
   a third window untouched by everything earlier in this campaign. Result at 1000 paths, against
   the shipped champion, with the strictly-causal profile:

   | | dev | holdout | test |
   |---|---|---|---|
   | gamma 1.35 (24h) | −0.20 % | **−0.48 %** | −0.55 % |
   | gamma 1.35 (1h) | +0.74 % | **+0.34 %** | +0.39 % |

   The two horizons want opposite corrections, consistently, on all three windows. So the exponent
   is keyed on the step size — `{5: 1.35, 1: 0.85}` — which leaves crypto-1h byte-identical to
   before (verified: 0.000 % on all three windows).

4. Refitting the eight com-equ multipliers at the new gamma adds a little more on all three
   windows (−0.29 / −0.50 / −0.58 %), and they are again stable under refitting four months apart
   (AAPLX 0.684 → 0.680, NVDAX 0.648 → 0.649, TSLAX 0.609 → 0.612) while the crypto ones again
   wander (HYPE 1.032 → 0.959). Same criterion, same verdict as Iter 7.

**Verdict**: adopted. `seas_gamma = {5: 1.35, 1: 0.85}` and the refit com-equ scale map.

**Lesson**: why would the profile need *amplifying*? Because every step of building it shrinks it
— winsorising at 6 robust sigmas, the width-5 circular smoother, the 0.25–4.0 clip. The estimator
I chose in Iter 4 to stop the profile being set by single prints also flattened it by about a
third, and gamma > 1 is undoing my own regularisation. The general lesson is harder-won: when a
one-dimensional sweep of a *shape* parameter also moves the *level*, it measures the level, and
§2 says the level is the one thing that is nearly free. Sweep shape parameters with the level
re-optimised at every point, or do not believe the answer. This one had been wrong since Iter 4
and survived two validation windows without complaint.

---

## Iter 13 — Re-auditing every other knob with the level free

**Hypothesis**: Iter 12 found that `seas_gamma` had been chosen by a 1-D sweep that was silently
measuring the width *level*. `revert_halflife`, the EWMA half-lives, the winsorisation threshold
and `nu` were all chosen the same way. If the mistake was systematic, more of them are wrong.

**Change**: re-ran each candidate taking the **minimum over `scale` ∈ {0.80, 0.90, 1.00}**, so
every configuration is compared at its own best level, on both validation windows.

**Result** (% vs the champion, each at its own optimal scale):

| candidate | dev | test |
|---|---:|---:|
| revert_hl = 48 | −0.04 | +0.01 |
| revert_hl = 192 | +0.17 | +0.05 |
| revert_hl = 384 | +0.40 | +0.15 |
| revert_hl = ∞ (no reversion) | +0.86 | +0.42 |
| halflives (6,36,288) | +0.09 | +0.07 |
| halflives (24,144,1152) | −0.04 | −0.03 |
| halflives (12,144,1152) | −0.05 | −0.03 |
| winsor 3.5 | −0.01 | +0.17 |
| winsor 10 | +0.11 | −0.01 |
| nu = 3 | +0.06 | +0.03 |
| nu = 6 | +0.03 | −0.07 |

**Verdict**: nothing changes. `revert_halflife = 96` is confirmed (mean reversion is real and
worth +0.86 %/+0.42 % if removed; the optimum is flat between 48 and 96). Everything else is
inside ±0.07 % on both windows. `halflives=(12,144,1152)` is negative on both but by under
0.05 % — below the threshold at which anything in this campaign transferred — so the simpler
incumbent stays.

**Lesson**: the level confound was real but it only mattered for the one parameter that is
*intrinsically* about shape. A half-life or a tail index changes the level as a side effect; an
exponent on a mean-1 profile changes almost nothing else. Worth knowing which of your knobs are
shape knobs before you sweep them.

---

## Iteration 14 — trade count and volume: real regression signal, zero CRPS

**Hypothesis**: every context carries `volume` and `trade_count` and the champion ignores both.
The case for them is strongest exactly where the model is weakest — for a thin asset, trade
count is the most direct measure of whether the next five minutes *can* move at all, and that
is the regime the per-asset width multiplier currently papers over with a constant.

**Change**, in two stages.

*Stage 1, a diagnostic* (`eda_activity.py`). Regress log realized future 24h volatility on the
champion's own volatility predictor alone, then on that predictor plus three activity features
(fast−slow log EWMA of deseasonalised trade count, the same for volume, and the slow count
level). Fit on the first 60 % of each asset's history, measure residual variance on the last
40 %, convert with the §2 identity: gain ≈ 0.25 · (var_before − var_after).

*Stage 2, the CRPS test* (`exp_activity_crps.py`). One bounded coefficient rather than a free
regression: multiply the simulated width by `clip(exp(beta · a), 1/1.3, 1.3)` with `a` the
fast−slow log EWMA of deseasonalised trade count (`models._activity_mult`, and a `{asset}_act`
time-of-week activity profile added to both artifacts so that "unusually quiet" means unusual
*for this hour*). `scale` re-optimised over {0.85, 0.90, 0.95} at every beta, per Iter 12.

**Result**, stage 1 (implied CRPS %, positive = activity helps):

| BTC | ETH | SOL | XRP | HYPE | XAU | NVDAX | GOOGLX | TSLAX | AAPLX | WTIOIL | SPCX | SP500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −0.07 | +0.00 | +0.02 | +0.16 | +0.03 | +9.19 | +0.24 | +0.28 | +0.50 | +0.01 | −2.68 | −0.53 | +0.07 |

Mean +0.56 — but that is XAU alone, whose `sd_base` of 3.08 (vs ~0.3 elsewhere) says its panel
is mostly gap, not signal. **Excluding XAU the mean is −0.16**: the features are net harmful
even in the regression that was built to flatter them.

**Result**, stage 2 (% vs the champion, each row at its own best scale):

| beta | comequ dev | comequ test | crypto dev | crypto test |
|---|---:|---:|---:|---:|
| +0.10 | +0.24 | +0.11 | +0.13 | +0.05 |
| +0.20 | +0.63 | +0.31 | +0.36 | +0.19 |
| +0.35 | +0.88 | +0.49 | +0.73 | +0.45 |
| +0.55 | +1.04 | +0.62 | +0.93 | +0.70 |
| −0.20 | +0.11 | +0.39 | +0.07 | +0.22 |

**Verdict**: rejected, `act_beta = 0`. Every coefficient is worse than zero, in both signs, on
both windows, for both groups, and the penalty rises monotonically in |beta|. This is the
cleanest null in the campaign: a feature carrying any usable width information would show a
small gain at small |beta| of the right sign, and none does.

**Lesson**: an incremental-R² diagnostic on log volatility is not a CRPS test, and the gap
between them is not just noise. Squared log error pays full price for getting a dead market
wrong; CRPS on relative price changes barely charges for it, because when nothing trades the
realized change is ~0 and any reasonably narrow forecast already scores well. That is exactly
why the one asset where the diagnostic screamed (XAU, +9 %) is the one where the CRPS is least
sensitive — and why the two assets it hated most (WTIOIL, SPCX) were enough to sink a linear
fit. Convert a diagnostic into the scored objective before believing it. Second lesson, cheaper:
the diagnostic's own mean was a single-asset artefact; look at the cross-section, not the mean.

The code stays in the tree as a no-op (`act_beta` defaults to 0, `_activity_mult` is never
called, the `{asset}_act` profiles are unused by the champion) so the experiment is rerunnable.
Verified that rebuilding the artifacts left the volatility profiles bit-identical: the champion
scores 2851.324 on the dev window before and after.

---

## Iteration 15 — the seasonal profile's resolution is already at its optimum

**Hypothesis**: the profile uses 30-minute buckets smoothed over 5 of them — a 2.5-hour window,
wide next to the events it is meant to capture, since a cash open is a five-minute affair. Iter
12 found the 1-minute step wants the profile *damped* (γ=0.85) and the explanation I wrote down
was that 30-minute buckets are coarse relative to a 1-minute step. If that explanation is right,
a finer grid should help, and should help the 1-minute step most.

**Change**: made the bucket width a parameter of `fit_profile`, and made `models.py` read the
resolution off the profile's own length (`_bidx`) instead of a module constant, so an artifact
at any resolution drops straight in. Built eight variants and swept resolution × γ × scale
jointly — resolution and γ interact (a sharper profile needs less un-shrinking) and both move
the level, so every cell is reported at its own best `scale` (`exp_resolution.py`).

**Result** (% vs the shipped 30min/s5 at γ=1.35, best scale in each cell):

| profile | span | com-equ dev | com-equ test | crypto dev | crypto test |
|---|---:|---:|---:|---:|---:|
| 30min/s5 *(shipped)* | 150 min | 0.00 | 0.00 | 0.00 | 0.00 |
| 30min/s3 | 90 min | −0.05 | −0.05 | +0.04 | +0.08 |
| 30min/s9 | 270 min | +0.23 | +0.28 | +0.03 | +0.06 |
| 15min/s9 | 135 min | +0.86 | +0.96 | +0.00 | +0.01 |
| 15min/s5 | 75 min | +0.84 | +0.92 | +0.03 | +0.08 |
| 10min/s5 | 50 min | +1.03 | +1.17 | +0.01 | +0.06 |
| 60min/s3 | 180 min | +0.08 | +0.06 | +0.01 | +0.06 |
| 60min/s5 | 300 min | +0.29 | +0.36 | +0.05 | +0.10 |

At the 1-minute step (crypto-1h) resolution moves nothing at all: every variant is within
0.03 % of the shipped profile at matched γ.

**Verdict**: rejected; the shipped 30-minute/5-bucket profile stays. The best candidate,
30min/s3, is −0.05 % on both com-equ windows but positive on both crypto windows, and 0.05 % is
half the threshold anything in this campaign has transferred at.

**Lesson**: two things, and the grid was built to separate them. **Bias-variance here is about
observations per bucket, not about the smoothing window.** 15min/s9 spans 135 minutes, near the
shipped 150, and still loses 0.9 % on com-equ — with the same effective smoothing, halving the
sample per bucket is what costs. The tokenised equities have roughly nine months of prints in
this snapshot; a 30-minute bucket at the 5-minute step gets a few hundred returns and a 10-minute
one gets a third of that, which is where SPCX stopped having any bucket above the 20-observation
threshold at all.

Second, **my stated reason for the 1-minute damping was wrong.** Bucket coarseness is not why
γ=0.85 beats γ=1.0 there — refining the grid 6× leaves the 1-minute step unchanged to three
decimal places. Whatever makes a 1-minute return less seasonal than a 5-minute one is a property
of the return (bid-ask bounce and zero-return minutes contribute width that does not follow the
time-of-week pattern), not of the profile's resolution. The parameter was still fitted correctly
and still transfers; only the story attached to it was wrong, and the report is corrected.

**Also fixed**: `fit_profile` returned an all-NaN array when no bucket cleared the
20-observation threshold (SPCX at 10-minute buckets), which would have silently produced NaN
prices rather than failing. It now returns a flat profile.

---

## Iteration 16 — the leverage effect, and why three feature hunts in a row came back empty

**Hypothesis**: §2 says the remaining prize is the conditional *level* — an oracle handed the
realized volatility of the window it predicts beats the champion by ~6 %, and the EWMA blend
leaves roughly a third of the log-variance of that target unexplained. The leverage effect (down
moves raise future volatility) is the most robust volatility predictor in the literature that
this model has no term for, and it should bite hardest on the tokenised equities.

**Change**: `lev_beta` on the conditional level, in two feature forms, both expressed in the
asset's own sd units so one coefficient can be shared across assets — `"down"`, the share of
recent variance contributed by negative returns (the EWMA analogue of the GJR term), and
`"ret"`, minus the standardised cumulative return (the textbook form). Verified both are centred
at zero under symmetric noise. Scale re-optimised in every cell.

**Result** (% vs champion; both groups, both windows):

| candidate | comequ dev | comequ test | crypto dev | crypto test |
|---|---:|---:|---:|---:|
| ret b=+0.03 | +0.01 | +0.02 | +0.00 | +0.01 |
| ret b=+0.06 | +0.02 | +0.05 | +0.01 | +0.02 |
| ret b=+0.12 | +0.06 | +0.11 | +0.03 | +0.06 |
| ret b=−0.06 | +0.00 | −0.02 | +0.01 | −0.01 |
| down b=+1 | +0.00 | +0.03 | +0.00 | +0.01 |
| down b=+2 | +0.02 | +0.06 | +0.01 | +0.02 |
| down b=+4 | +0.07 | +0.15 | +0.07 | +0.06 |
| down b=−2 | +0.03 | −0.02 | +0.03 | +0.01 |

**Verdict**: rejected, `lev_beta = 0`. Positive coefficients are monotonically worse in both
forms; the negative ones sit inside noise. The whole table spans 0.15 %, which says the feature
is close to orthogonal to anything CRPS can see here.

**Lesson — the one that ties Iters 11, 14 and 16 together.** Range estimators, trade activity
and now leverage all failed, and they failed for the same reason: **the model already carries a
fast EWMA of realized volatility, and every one of those features acts on future volatility
*through* realized volatility.** By the time a sell-off, a volume spike or a wide range has
happened, it is in the squared returns, and a 1-hour-half-life EWMA has already moved. They are
not incremental; they are slower copies of what the model reads directly.

That reframes what is left. A feature can only add if it knows about volatility **that has not
been realized yet** — which means scheduled, forward-looking information: the cash open, the
weekend, a numbers release. The deterministic part of exactly that is the time-of-week profile,
and it is the single largest win in the campaign (−1.9 %). The three failures and the one success
are the same statement seen from opposite sides, and I would have got here faster by asking
"is this incremental to a fast EWMA?" before building any of them.

---

## Iteration 17 — the aggregated intervals were too thin-tailed (champion change)

**Hypothesis**: after three failed attempts to predict the volatility *level* better, look at the
part of the score that is not about the level at all. Roughly 39 % of a 24-hour score sits on
intervals longer than one step (30-min 25 %, 3-hour 10 %, 24-hour 3.6 %), and ~29 % of a 1-hour
score sits on the cumulative `*_gaps` terms. The champion's sigma path is *deterministic* given
the context, so those aggregates are sums of independent innovations and the CLT drags them
toward Gaussian — thinner than the real thing, which keeps clustering volatility inside the
window. The per-step marginal is already fat (Student-t(4)); the aggregates are not.

**Change**: `vov_ar`, a persistent AR(1) factor in log-volatility applied along the path,
renormalised as `exp(h − sd²)` so that **every step's variance is exactly unchanged**. That makes
it a pure shape change aimed at the aggregates — unlike the constant-per-path `vol_of_vol` tried
and rejected earlier, which is the half-life→∞ limit and moves the per-step marginal as much as
the aggregates. Verified directly: at sd=0.5 the simulated 24-hour kurtosis goes 3.25 → 4.57
while the 5-minute sd is unchanged to six digits.

**Result** (% vs champion, scale re-optimised in every cell, three disjoint windows):

| | com-equ dev | com-equ val3 | com-equ test | crypto dev | crypto val3 | crypto test |
|---|---:|---:|---:|---:|---:|---:|
| sd=0.20 | −0.06 | −0.06 | −0.05 | −0.02 | +0.04 | −0.07 |
| sd=0.25 | −0.07 | −0.08 | −0.08 | −0.02 | +0.08 | −0.09 |
| **sd=0.30** | **−0.10** | **−0.10** | **−0.10** | −0.01 | +0.14 | −0.09 |
| sd=0.35 | −0.11 | −0.09 | −0.10 | +0.01 | +0.23 | −0.07 |

Half-life made no difference (36 vs 144 steps agree to three decimals) — both are long relative
to the intervals being aggregated, which is the only thing the factor has to outlive. At the
1-minute step every setting is inside ±0.05 % and val3 flips sign.

**Verdict**: **shipped for com-equ at sd = 0.30**, the only place it transfers. Three disjoint
windows give −0.096 %, −0.096 %, −0.102 % — small, but three independent windows landing within
0.006 % of each other is not what noise looks like, and it is the effect the decomposition
predicted before it was measured. Crypto changes sign between windows at every sd; left at 0.

**Robustness**: per asset, at the shipped setting, against the factor off (each at its own best
level) — the effect is broad, not one or two assets carrying a mean:

| | AAPLX | GOOGLX | NVDAX | SP500 | SPCX | TSLAX | WTIOIL | XAU | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | −0.20 | −0.20 | −0.10 | −0.06 | −0.09 | −0.14 | −0.02 | −0.31 | −0.14 |
| val3 | −0.14 | −0.10 | −0.11 | −0.03 | −0.19 | −0.12 | −0.09 | −0.04 | −0.10 |
| test | −0.12 | −0.14 | −0.11 | −0.08 | −0.19 | −0.20 | −0.04 | −0.11 | −0.12 |

All 24 cells negative. For an effect of this size that matters more than the mean does.

Re-checked `nu` with the factor on, since both control kurtosis at different aggregation scales:
3.5, 5 and 6 are all within ±0.13 % of 4 with no consistent sign. `nu = 4` stands.

**A level correction comes with it.** Switching the factor on moves the com-equ optimum from
0.90 to 0.95 on all three windows, and §2 says why: a fatter mixture at fixed variance is
*narrower in the bulk*, so the CRPS-optimal width moves up. The shift is uniform across the eight
assets, so the fitted per-asset structure is untouched and only the common factor is rescaled
(`_LEV = 0.95/0.90`). Reporting the gain without that correction would have understated it.

**Second change, found while checking the above.** The crypto baseline in these sweeps is itself
scale-optimised, which exposed that flat 0.90 is not the 1-minute optimum. Measured on three
windows at the 1-minute step, 0.95 beats 0.90 by −0.085 %, −0.215 %, −0.049 % — consistent sign,
consistent size. At the 5-minute step crypto flips (dev wants 0.85, val3 wants 1.00, test wants
0.85), so 0.90 remains the compromise there. `scale` is now a dict keyed on step size, exactly
like `seas_gamma`: `{5: 0.90, 1: 0.95}`.

**Lesson**: when a line of attack keeps returning nulls, check whether it is the only axis. Three
iterations went into predicting the volatility level better and all three failed for the same
structural reason; the score's *shape* dimension had never been examined at all, and the first
look at it paid on both configurations that had room. Also: every previous attempt at randomising
volatility (`vol_of_vol`) failed because it perturbed the 61 % of the score that was already
calibrated. Renormalising so the per-step variance is exactly preserved is what turned the same
idea from a loss into a gain — the constraint was the whole trick.

---

## Iteration 18 — two follow-ups on the factor, both negative, champion frozen

**Hypothesis 1**: the factor currently starts at its *stationary* spread, which says the current
volatility level is itself uncertain. But the current level is precisely the thing the context
does pin down; the defensible prior is that the path starts at the implied level and the
uncertainty *grows* with horizon. Added `vov_start="zero"`, with the renormalisation made
per-step (`Var(h_k) = sd²(1 − φ^{2k})`) so that every step's variance is still exactly preserved.

**Hypothesis 2**: crypto was rejected in Iter 17 using a scale grid capped at 0.95, and crypto's
val3 optimum is 1.00 — so the rejection might have been a grid artefact rather than a real sign
flip.

**Result 1** (com-equ, % vs the factor off, each at its own best level):

| variant | dev | val3 | test |
|---|---:|---:|---:|
| stationary sd=0.30 *(shipped)* | −0.19 | −0.13 | −0.14 |
| zero sd=0.30 hl=36 | −0.18 | −0.11 | −0.15 |
| zero sd=0.45 hl=36 | −0.21 | −0.13 | −0.20 |
| zero sd=0.45 hl=144 | −0.17 | −0.14 | −0.18 |
| zero sd=0.80 hl=36 | +0.75 | +1.03 | +0.80 |

**Result 2** (crypto, re-tested at the level each configuration actually ships at, so no grid can
flatter anything):

| variant | 24h dev | 24h val3 | 24h test | 1h dev | 1h val3 | 1h test |
|---|---:|---:|---:|---:|---:|---:|
| stationary sd=0.15 | −0.02 | +0.04 | −0.08 | −0.02 | +0.01 | −0.02 |
| stationary sd=0.30 | +0.01 | +0.23 | −0.24 | +0.07 | +0.18 | +0.05 |
| zero sd=0.30 | −0.02 | +0.13 | −0.21 | −0.00 | −0.00 | −0.03 |
| zero sd=0.45 | +0.10 | +0.40 | −0.30 | +0.01 | +0.02 | −0.04 |

**Verdict**: both rejected; the champion is frozen at stationary, sd=0.30, com-equ only.

Hypothesis 1 was a better story than it was a model. The two parameterisations find the same
optimum from different directions — zero-start needs sd≈0.45 to do what stationary does at 0.30,
and at its best it is 0.02 % ahead on the mean of three windows, which is a tie. The prior I
argued for is more defensible and buys nothing, so the simpler incumbent stays.

Hypothesis 2 was wrong: at the shipped level every crypto-24h setting is *positive* on val3 and
negative on test, which is the same sign flip seen before, not a grid artefact. crypto-1h is
inside ±0.04 % everywhere. The wider grid did change the *size* of the earlier crypto numbers,
which is worth noting as a methodological point — when a sweep takes the minimum over a discrete
level grid, a candidate whose optimum sits outside the grid is penalised, and the fix is to
evaluate at the level the thing would actually ship at.

**Lesson**: a seam that pays once does not keep paying. Two principled extensions of the same
idea, one better-motivated than what shipped, both came back at zero. The check that settled it
in both cases was the same: evaluate at the configuration you would actually serve, not at the
best cell of a grid.

---

## Iteration 19 — Re-audit of the dynamics knobs with the new factor on

**Hypothesis**: the vol-of-vol factor shipped in Iteration 17 adds path-level dispersion to the
volatility, which is the same quantity the *deterministic* dynamics knobs control — the EWMA
half-life set, their weights, the reversion half-life, the winsorising threshold. If the factor
displaced any of them, their previously-fitted values are now stale. Every knob was last fitted
with the factor off, so this needs checking before freezing.

**Change**: no code. One sweep over the com-equ configuration, each candidate re-optimised over
the level grid (0.90, 0.95, 1.00) so nothing is penalised for a level shift, on all three
windows, 500 paths, 3-hour cadence.

**Result** (% vs the champion, each at its own best level):

| variant | dev | val3 | test |
|---|---:|---:|---:|
| `revert_halflife` 48 (vs 96) | −0.05 | −0.07 | +0.04 |
| `revert_halflife` 192 | +0.22 | +0.21 | +0.09 |
| `halflives` (24, 144, 1152) | −0.05 | −0.06 | −0.02 |
| `weights` (0.2, 0.3, 0.5) | −0.05 | −0.05 | −0.03 |
| `winsor` 4.0 (vs 6.0) | −0.07 | +0.02 | +0.01 |

**Verdict**: nothing moves. Every candidate is inside ±0.1 % on every window, and the two that
are negative on all three (`halflives`, `weights`) are negative by 0.02–0.06 % — below the
threshold at which I have been willing to believe anything in this campaign, and both are
*slower* configurations that push in the same direction as the factor without being better than
it. `revert_halflife` 192 is the only candidate with a consistent sign, and it is consistently
worse, which confirms 96 sits on the right side of its optimum. Champion unchanged.

**Lesson**: the factor is genuinely orthogonal to the drift of the deterministic σ path. That is
what the renormalisation was designed to guarantee — each step's variance is preserved exactly,
so the factor can only redistribute dispersion *across* a path, never change its level — and the
sweep is the confirmation that the design held. A re-audit after a structural change is cheap
insurance; this one cost four minutes and bought the right to freeze.

---

## Iteration 20 — The `*_gaps` block in CRYPTO_1H, and a prompt-grid artifact

**Hypothesis**: twelve of the eighteen interval terms in the 1-hour configuration are the
cumulative return from the prompt start to minute 5, 10, … 60 — about 29 % of that budget — and
nothing in the model knows they exist. They all begin at t=0, so they weight the opening minutes
of the path, where the EWMA is freshest, far more heavily than a `30min` term does. If the model's
width has a term structure the score does not share, a width fitted per *term type* rather than
per step size has something to find. This was the largest piece of structure in the scoring rule
I had never touched.

**Change**: no model change. A new diagnostic, `agent/eda_gaps.py`: for every scoring term,
the ratio of the realized term's typical size to the simulated ensemble's, pooled over five assets
and three windows. Above 1 means the model is too narrow on that term.

**Result**, first pass (3-hour prompt cadence, RMS ratio):

| term | ratio | | term | ratio |
|---|---:|---|---|---:|
| `1min` | 1.130 | | `0_5min_gaps` | **1.409** |
| `5min` | 1.123 | | `0_10min_gaps` | **1.334** |
| `30min` | 1.124 | | `0_15min_gaps` | 1.222 |
| `60min_abs` | 1.129 | | `0_20min_gaps` | 1.149 |

A 28 % under-dispersion on the first five minutes, decaying monotonically to the interval level by
minute 20, present in all five assets. That is a large, clean, structural-looking effect.

It is an artifact of my own prompt grid. A 180-minute cadence lands on :00 every single time, so
"the first five minutes of the path" meant "the first five minutes of a round hour" in every
prompt — and round hours are when candles close and volatility spikes. Re-running at 173 and 97
minutes, which walk across the minute-of-hour, the effect disappears completely:

| term | cad 173 | cad 97 | cad 180 |
|---|---:|---:|---:|
| `1min` | 1.106 | 1.086 | 1.115 |
| `5min` | 1.043 | 1.027 | 1.053 |
| `30min` | 1.047 | 0.997 | 1.033 |
| `0_5min_gaps` | 1.059 | 0.981 | **1.225** |
| `0_10min_gaps` | 1.054 | 1.004 | **1.175** |
| **gaps block** | 1.043 | 0.993 | 1.091 |
| **interval block** | 1.056 | 1.029 | 1.060 |

(Also switched the statistic from a mean-square to a mean-absolute ratio: with t(4) tails the
mean-square version was set by two or three prompts, and its pooled level moved ten points between
grids — larger than any effect worth chasing.)

**Verdict**: no seam. Off the round hour, the gaps ladder is flat and sits *at or slightly below*
the interval ladder on both grids — the model is marginally over-dispersed on the gaps, not under,
by 1–3 %. By the identity in the report that is worth 0.25 × 0.02² ≈ 0.01 % of those terms.
Nothing to fit. `0_60min_gaps` and `60min_abs` return identical ratios at every cadence, which is
the check that the machinery is right: they are the same horizon measured from the same origin.

The one pattern that survives all three grids is unrelated to the gaps: `1min` and `2min` are
consistently the most under-dispersed terms (1.09–1.12) and everything at 5 minutes and above sits
near 1.00–1.05. That is the negative return autocorrelation the report already priced — realized
variance aggregates sub-linearly, an i.i.d. model cannot match both ends, and the ~7 % spread
between the ends is worth about 0.04 %. Confirmed by a second, independent route; still below the
threshold at which I will touch the champion.

**Lesson**: sample the prompt grid the way the validator does, or the grid becomes the finding.
Every earlier experiment in this campaign used a 3-hour cadence, which is fine for comparing two
models against each other — both see the same prompts and the bias cancels — but it is not fine
for a *calibration* diagnostic, where the absolute level is the whole result. This one would have
sent me to fit a five-minute width ramp against a clock artifact.

---

## Iteration 21 — A floor on the conditional level (insurance, not a model change)

**Hypothesis**: not a CRPS hypothesis. A stress test of the served interface on degenerate
contexts — 1 row, 99 % NaN, a completely flat price series — found no crashes and correct shapes
everywhere, but one case with an unbounded downside. If every close in the 7-day context is
identical (a halted or delisted venue), the EWMA collapses to `MIN_SIGMA = 1e-6`, which is a
numerical guard rather than a forecast, and the model serves a 24-hour distribution with a
standard deviation of 0.002 % — about 75× narrower than the quietest real prompt in the whole
snapshot. CRPS for too-narrow saturates at +41 % *relative to the correct width*, but the correct
width there could be a hundred times larger, so the absolute loss is unbounded. Thin, recently
listed assets (SP500 from mid-July, SPCX) are the plausible carriers.

**Change**: `vol_floor` in `sv_model`, applied to the conditional level `v0` and to the long-run
level it reverts toward. It is deliberately placed on the **deseasonalised** level, before the
seasonal profile and the width multiplier: a flat floor on the *served* per-step σ would clip the
overnight trough of the time-of-week profile, which is the single most valuable thing the model
does. Default 0 (no-op for every other model in the zoo); the champion carries 1.5e-5.

**Result**: `agent/verify_floor.py` runs the champion with and without the floor over 9802 prompts
— all 13 assets, both horizons, all three validation windows — and every path is **bit-identical**.
So the floor costs exactly zero and the official numbers in report §6 stand unmodified. On the
halted-venue context it raises the served 24-hour width from 0.002 % to 0.031 % (and the 1-hour
from 0.0006 % to 0.012 %), a factor of 15.

**Verdict**: keep. The sizing is the part worth defending. The quietest of those 9802 prompts has
a served per-step σ 3.3× the floor, and since the floor sits upstream of both the seasonal factor
and the 0.72–0.96 width multiplier, its true headroom on the level it actually constrains is
larger than that. Setting it any higher would start to bind on SP500's quietest days; setting it
lower buys less protection for no gain, since binding slightly is cheap — §2's asymmetry says
erring wide at the quietest moment of the quietest asset costs almost nothing.

**Lesson**: a numerical guard and a prior are not the same object, and it is worth checking which
one the model is actually leaning on. `MIN_SIGMA = 1e-6` was written to stop a division by zero
and ended up being the model's answer to "what if this asset stopped trading" — a question nobody
asked it. The test that found it was not a CRPS sweep but a list of eleven deliberately broken
contexts, which took ten minutes and should have been the first thing I ran, not the last.

---

## Iteration 22 — Should the width be tilted across the horizon?

**Hypothesis**: the multiplier in front of the volatility path is a constant, so the same shrink
is applied to the first five minutes of a 24-hour path as to the last. There is a reason from §2
to think it should not be. The CRPS penalty is asymmetric — too wide costs linearly and without
bound, too narrow saturates at +41 % — so the optimal width sits *below* `exp(E[log σ])` by an
amount that grows with the uncertainty in log σ. That uncertainty is near zero at step 1, where
the EWMA has just been measured, and grows toward the unconditional spread (sd ≈ 0.48 in logs for
BTC) by step 288. The shrink should therefore deepen with the horizon, and the fitted ramp should
come out negative. This is the only shape knob left that touches 100 % of the score.

**Change**: `scale_ramp` in `sv_model` tilts the per-step width linearly across the path,
renormalised to mean 1 so it cannot move the level. Swept over {−0.30, −0.15, 0, +0.15, +0.30}
with the level re-optimised over {0.90, 0.95, 1.00} at every point (Iter 12), on all three
windows and all three configurations. `agent/exp_ramp.py`.

**Result** (% vs ramp 0, each cell the minimum over the level grid):

| ramp | com-equ dev / val3 / test | crypto-24h dev / val3 / test | crypto-1h dev / val3 / test |
|---|---|---|---|
| −0.30 | +0.17 / +0.09 / +0.11 | +0.13 / +0.14 / +0.06 | +0.02 / −0.02 / +0.10 |
| −0.15 | +0.04 / −0.00 / +0.01 | +0.01 / +0.02 / −0.02 | −0.04 / −0.06 / −0.00 |
| **0.00** | — | — | — |
| +0.15 | +0.03 / +0.06 / +0.04 | +0.06 / +0.04 / +0.08 | +0.10 / +0.11 / +0.07 |
| +0.30 | +0.09 / +0.15 / +0.12 | +0.16 / +0.12 / +0.21 | +0.23 / +0.25 / +0.17 |

**Verdict**: rejected, and the shape of the rejection is worth keeping. The *sign* is exactly what
the theory predicts — every positive ramp is worse than its negative mirror in all nine cells, so
the surface is tilted toward narrowing the late steps. But the optimum sits between 0 and −0.15,
the best cell anywhere is −0.06 %, and no window agrees with another on where it is. The parameter
stays at its default of 0; the champion is byte-for-byte unchanged (verified against the
pre-change code on a fixed context, 23 200 floats identical).

**Lesson**: a correct mechanism can still be too small to fit. The asymmetry argument is right and
the data can see it — the whole surface leans the way it should — but the effect it produces is
third-order, which is what §2 says it must be: the quadratic term in the CRPS expansion is
symmetric, and shrink-for-uncertainty only appears when you go one order further. I could have
predicted the size before running this, and that is the fourth time in this campaign the identity
would have saved an experiment.

---

## Iteration 23 — Does the time-of-week profile flatten in a hot regime?

**Hypothesis**: the seasonal profile is a fixed shape, scaled up and down by the EWMA. But a macro
shock does not respect trading hours — news arrives overnight and the tokenised equities have to
price it at whatever hour it lands — so in a hot regime the quiet buckets should catch up and the
day/night ratio should compress. This is the one interaction left that is not redundant with the
EWMA under the Iter 16 principle: neither the calendar alone nor the realized volatility alone
contains it, only the product.

**Change**: `seas_gamma_beta` makes the seasonal exponent `γ + β·log(v0 / v_long)` — the ratio of
the current conditional level to the asset's own 7-day level, clipped to ±1.5 in logs. `_bend`
renormalises the profile to mean 1 at every exponent, so only the shape moves and the level is
swept independently. β < 0 is the direction the story predicts. `agent/exp_gamma_regime.py`.

**Result** (% vs β=0, each cell the minimum over the level grid):

| β | com-equ dev / val3 / test | crypto-24h dev / val3 / test | crypto-1h dev / val3 / test |
|---|---|---|---|
| −1.0 | +0.18 / +0.11 / +0.14 | +0.02 / −0.07 / +0.01 | +0.02 / +0.11 / +0.04 |
| −0.5 | +0.03 / −0.02 / +0.00 | −0.00 / −0.05 / −0.01 | +0.00 / +0.04 / +0.01 |
| **0.0** | — | — | — |
| +0.5 | +0.16 / +0.20 / +0.20 | +0.03 / +0.07 / +0.04 | +0.02 / −0.01 / +0.02 |
| +1.0 | +0.50 / +0.60 / +0.61 | +0.07 / +0.17 / +0.12 | +0.06 / +0.01 / +0.07 |

**Verdict**: rejected. The asymmetry is the tell that the story is not nonsense — on com-equ,
β=+1.0 costs 0.5–0.6 % while β=−1.0 costs only 0.14 %, so the surface leans the way the mechanism
says it should. But the optimum sits between 0 and −0.5, the only configuration with a consistent
negative sign is crypto-24h at β=−0.5 (−0.02 % averaged over three windows), com-equ is a tie and
crypto-1h is worse. Nothing clears 0.1 %. Default stays 0; the champion is bit-identical.

**Lesson**: that is now two structurally-motivated shape knobs in a row — the horizon ramp and this
one — where the fitted sign matched the theory and the fitted size was an order of magnitude below
the threshold. Taken together that is the most informative result of the late campaign: the model
is not missing a mechanism, it is sitting at the bottom of a flat basin, and every remaining
mechanism I can name is worth a few hundredths of a percent.

---

## Iteration 24 — Refit the com-equ multipliers against the configuration that actually ships

**Hypothesis**: the eight per-asset multipliers were fitted when the champion had `seas_gamma=0.85`
and no vol-of-vol factor. Both have since changed — the 5-minute exponent is now 1.35 and the AR(1)
factor is on for every com-equ asset — and when the factor landed I corrected the whole map by a
single uniform 0.95/0.90. That is the right *average* correction, but the eight assets have very
different seasonal amplitudes, so the per-asset corrections should differ. Iter 18's rule was to
*evaluate* at the configuration you would serve; this applies the same rule to *fitting*.

**Change**: `agent/exp_recalibrate.py` refits all eight multipliers with the champion's own gamma,
factor, floor and profile, on 2026-05-20→06-20 — before every validation window — and scores the
refit map against the shipped one on all three, level re-optimised for each.

**Result**:

| asset | refit | shipped |
|---|---:|---:|
| XAU | 0.671 | 0.765 |
| NVDAX | 0.680 | 0.761 |
| GOOGLX | 0.664 | 0.753 |
| TSLAX | 0.637 | 0.716 |
| AAPLX | 0.721 | 0.800 |
| WTIOIL | 0.918 | 0.956 |
| SPCX | 0.807 | 0.894 |
| SP500 | 0.865 | 0.938 |

| map | dev | val3 | test |
|---|---:|---:|---:|
| shipped | — | — | — |
| refit | −0.061 % | +0.019 % | −0.011 % |

**Verdict**: rejected; the shipped map stands. The refit moves every asset down by 0.04–0.09, which
is very nearly a constant offset in logs — and a constant offset is a *level*, which the level grid
absorbs. What the fit actually says is that the shape of the map is unchanged and only its overall
position moved, which is exactly what the uniform correction already did. Out of sample the two are
within 0.06 % and disagree in sign across windows.

**Lesson**: the hypothesis was that a uniform correction could not be right for eight heterogeneous
assets. It was right, and it did not matter, because the part it got wrong was the part §2 says is
nearly free. Worth checking rather than assuming — but the identity predicted the answer again.

## Iteration 25 — Shrink the reversion target toward a committed long-run level

**Hypothesis**: `sv_model` reverts its conditional volatility toward `v_long`, an EWMA of the
context with a 2016-step half-life. But the context *is* seven days, so `v_long` is estimated from
one week — and a week that happened to be hot leaves a reversion target that is itself an
overshoot. Volatility mean-reverts over months as well as over hours. A per-asset level fitted
offline over the whole snapshot is exactly the thing to shrink it toward, and it is the one piece
of information about the asset that is genuinely *not* in the context, so it is not redundant with
a fast EWMA the way everything in Iteration 16 was.

**Change**: `agent/vlong.py` fits one long-run level per (asset, step) — the RMS of the winsorised
log return at that step, deseasonalised by the *same* bent time-of-week profile `sv_model` uses, so
the blend is not a units error — and commits it as an artifact. `sv_model` gains `vlong_art` and
`long_blend`/`long_blend_map`: `log v_long ← (1−b)·log v_long + b·log v_committed`. `fit_end` lets a
window be scored against a level fitted strictly before it. `agent/exp_vlong.py` sweeps b.

**Result**. Sweep over b, each cell the minimum over the level grid, 500 paths, artifact fitted
through 2026-06-30 (before all three windows), % vs b=0:

| b | com-equ dev | val3 | test | crypto-24h dev | val3 | test | crypto-1h dev | val3 | test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | −0.171 | −0.041 | −0.037 | +0.026 | +0.021 | −0.145 | −0.040 | +0.002 | −0.024 |
| 0.4 | −0.254 | −0.020 | +0.041 | +0.120 | +0.113 | −0.215 | −0.071 | +0.029 | −0.042 |
| 0.6 | −0.250 | +0.065 | +0.251 | +0.287 | +0.286 | −0.212 | −0.091 | +0.081 | −0.053 |
| 0.8 | −0.161 | +0.210 | +0.614 | +0.533 | +0.632 | −0.122 | +0.161 | −0.057 | — |

Com-equ per asset, b = 0.2: negative in 21 of 24 asset-window cells (only NVDAX/test, SPCX/test and
WTIOIL/val3 positive), and 22/24 at b = 0.1.

Dev is three to four times better than the other two windows. The obvious explanation is staleness —
dev starts five days after the fit, test two months after — so I rebuilt the artifact at 2026-07-23
and 2026-08-25 and rescored each window against a level fitted immediately before it:

| window | stale artifact | fresh artifact |
|---|---:|---:|
| dev | −0.171 | −0.171 |
| val3 | −0.041 | −0.048 |
| test | −0.037 | −0.048 |

Freshness is worth 0.007 pp, not the missing 0.13. Dev is simply a kind window, and the honest
out-of-sample expectation is the val3/test number. At the width that actually ships (`scale`
{5: 0.90} with `SCALE_MAP`, 1000 paths, no level grid): **−0.168 / −0.026 / −0.039 %**, and the
level optimum does not move — b = 0.2 is best at 0.90 on dev and test, and b = 0 already prefers
0.92 on val3, so the blend carries no level correction of its own.

**Verdict**: shipped for the eight com-equ assets at b = 0.2, `vlong_art="vlong_full.npz"`. Crypto
left at 0 — the 24-hour leg shows the same dev/val3-positive, test-negative flip that kept the
vol-of-vol factor off, and the 1-hour leg is flat on val3. Scoping by asset also keeps the two
crypto horizons consistent with each other, which a blend keyed on step size would not. Verified
bit-identical on all twenty crypto prompts tested and changed on all sixteen com-equ ones, so the
crypto-24h and crypto-1h official legs stand unmodified and only the eight com-equ legs were re-run.

This is the first thing I have shipped whose out-of-sample size is *below* the 0.1 % bar the rest of
the model is held to, and I want that on the record. It is shipped on the consistency of the sign,
not the size: negative on three disjoint windows, at every b in 0.1–0.3, at four of five widths, in
21 of 24 asset cells, and under two artifact vintages. Doubling the paths from 500 to 1000 moved the
window means by 0.01–0.03 pp, so the measurement is window-limited, not Monte-Carlo-limited. The
expected gain is about 0.04 % of one configuration — roughly 0.01 % of the score.

**Lesson**: Iteration 16's redundancy principle said only information that is *not* already flowing
through realized volatility can be incremental, and I read that as "only forward-looking
information." That was too narrow. This feature is entirely backward-looking; what makes it
incremental is that it looks back *further than the context window reaches*. The served interface
hands the model seven days, so anything about the asset on a longer timescale is outside it by
construction and has to be carried in as an artifact. That is a second, separate source of
non-redundant information, and it is the only one of the two I could actually act on — and its size,
0.04 %, is a fair measure of how little is left on the table once the conditional level is right.

**Official re-run.** Only the eight com-equ legs changed (verified bit-identical on crypto), and
`predict.py` at 1000 paths moves COM_EQU_24H from 1822.5 to **1820.7** against the starting point's
2076.5 — −12.23 % to **−12.32 %**. Crypto stands at −4.14 % and −4.00 %.

## Iteration 26 — A per-asset tail index, fitted offline

**Hypothesis**: Iteration 25's lesson names a class — properties of the asset on a timescale longer
than the served context — and the long-run level is only one member of it. The fatness of the
innovation is another. `nu` is a single global 4.0, and Iteration 19 re-swept it with the vol-of-vol
factor on and found 3.5, 5 and 6 all within ±0.13 % with no consistent sign. Flat is exactly what a
global sweep looks like when different assets want different values and the average cancels.

**Change**: `agent/nufit.py` estimates one tail index per (asset, step) from *standardised*
residuals — the return deseasonalised by the model's own bent profile, then divided by the model's
own causal EWMA blend of conditional sd, so the estimate is of the innovation rather than of
volatility clustering. Sample kurtosis is useless for nu near 4 (its own variance is infinite), so
the index is read off the quantile ratio q(0.995)/q(0.75) of the absolute residual — a bounded
functional — inverted against the Student-t by bisection. `sv_model` gains `nu_art`/`nu_blend`,
blending the fit into the global value in logs.

**Result**. The fit has real dispersion and a clean story: crypto 4.1–5.8, tokenised equities
3.2–4.0. Continuous markets have thinner standardised tails; a market with an overnight gap and a
session open produces residuals a smooth EWMA cannot track, and that shows up as fat.

| asset | 5-min | | asset | 5-min |
|---|---:|---|---|---:|
| HYPE | 5.84 | | XAU | 3.86 |
| XRP | 4.81 | | TSLAX | 3.70 |
| SOL | 4.68 | | GOOGLX | 3.65 |
| BTC | 4.53 | | SP500 | 3.64 |
| ETH | 3.89 | | NVDAX | 3.60 |
| | | | AAPLX | 3.44 |
| | | | SPCX | 3.38 |
| | | | WTIOIL | 3.29 |

And it is worth nothing. Blend weight 0 → 1, level re-optimised at every point, % vs blend 0:

| blend | com-equ dev | val3 | test | crypto-24h dev | val3 | test | crypto-1h dev | val3 | test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.021 | −0.012 | −0.018 | −0.035 | −0.007 | −0.004 | +0.041 | +0.002 | −0.045 |
| 0.50 | +0.041 | −0.006 | −0.035 | −0.047 | −0.032 | +0.002 | +0.086 | +0.025 | −0.041 |
| 1.00 | +0.048 | +0.002 | −0.007 | −0.044 | −0.019 | +0.088 | +0.097 | −0.049 | −0.011 |

Every cell inside ±0.1 %, and no configuration keeps a sign across its three windows.

**Verdict**: rejected. `nu = 4.0` global stands. Contrast this with Iteration 25, which is the same
size but keeps its sign on every window, every weight and 21 of 24 asset cells — that is what the
difference between a small effect and no effect looks like, and it is why the sign test is the one
worth applying when the magnitude is at the noise floor.

**Lesson**: the class from Iteration 25 is real but it is not a licence. Long-horizon information
only pays where the score is sensitive, and §2 says the score is sensitive to the *width* and almost
nothing else. The long-run level is a width. A tail index is a shape parameter of the per-step
marginal, and between nu = 3 and nu = 6 the bulk of a standardised Student-t barely moves — so a
correctly measured, genuinely asset-specific, genuinely non-redundant quantity produced exactly zero
CRPS, for the third time in this campaign. Measure the sensitivity before fitting the parameter.

## Iteration 27 — The calendar's effect on width is convex, and one exponent cannot say so

**Hypothesis**: suggestions.md item 12, finally tested. The model's only statement about how much of
an asset's width follows the calendar is a single exponent on the time-of-week profile,
`seas_gamma = 1.35` at the 5-minute step. A power is *linear in log-profile*. If the true relation
between "how dead is this bucket" and "how wide should the model be" is not linear, no value of the
exponent can express it, and the sweep that chose 1.35 was choosing the best straight line through a
curve.

**Diagnostic first**. `agent/eda_profile.py` pools every 5-minute step of every path and reports
mean|realized| / mean|simulated| by decile of the bent profile value. Two off-hour cadences, because
of Iteration 20. Reproducible to three decimals between them:

| profile decile | 0.34 | 0.53 | 0.69 | 0.83 | 0.98 | 1.17 | 1.47 | 2.09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| com-equ ratio | 1.213 | 1.069 | 1.027 | 0.999 | 0.945 | 1.018 | 1.050 | 1.153 |
| % exactly zero | 16.1 | 7.4 | 4.6 | 3.3 | 2.8 | 2.0 | 1.4 | 1.0 |

| profile decile | 0.63 | 0.76 | 0.84 | 0.90 | 0.97 | 1.06 | 1.23 | 1.59 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crypto ratio | 1.175 | 1.161 | 1.181 | 1.096 | 1.090 | 1.074 | 1.098 | 1.044 |

Com-equ is **U-shaped**: relative to an average bucket the model is too narrow at *both* ends of the
calendar at once. That is convexity, and a power cannot produce it. Crypto is monotone decreasing —
over-amplified, wanting a smaller exponent, which a power *can* produce.

The com-equ shape has a mechanism. The deadest decile has 16 % of its realized 5-minute returns
exactly zero and still a 21 % *larger* mean absolute size than the ensemble. A closed session is not
a quiet diffusion; it is a point mass at zero plus occasional gap-sized prints. The CRPS-optimal
width for that mixture is wider than the width that matches its bulk, so the buckets where the
estimator sees the least action are exactly the ones where it under-serves.

**Change**: `sv_model` has carried `seas_gamma_lo` — a second exponent applied below the weekly mean
— since the activity experiment, and it had never been swept. `agent/exp_gamma_split.py` sweeps the
pair per asset class, with the width level re-optimised at every cell. Added `seas_gamma_map` and
`seas_gamma_lo_map` so the two classes can differ at the 5-minute step without touching crypto-1h.

**Result**. Com-equ, % vs the shipped single exponent, level re-optimised over 0.85–1.05:

| g_hi \ g_lo | 0.75 | 0.90 | 1.05 | 1.35 |
|---|---:|---:|---:|---:|
| 1.35 | −0.10 / +0.11 / +0.46 | −0.11 / +0.02 / +0.28 | −0.11 / −0.01 / +0.16 | 0 / 0 / 0 |
| 1.65 | −0.20 / −0.09 / +0.10 | −0.19 / −0.12 / −0.02 | −0.14 / −0.13 / −0.11 | +0.03 / −0.04 / −0.19 |
| 1.80 | −0.20 / −0.12 / −0.02 | **−0.16 / −0.14 / −0.12** | −0.10 / −0.11 / −0.19 | +0.12 / +0.01 / −0.23 |
| 1.95 | −0.14 / −0.12 / −0.10 | −0.10 / −0.11 / −0.18 | −0.00 / −0.08 / −0.22 | +0.26 / +0.08 / −0.20 |

A broad diagonal ridge, interior on both axes. Each window's own best cell differs, and test alone
would happily take `g_lo = 1.35` (no hinge, just a bigger exponent) — but that column is positive on
dev at every g_hi, so the hinge is what all three agree on. (1.80, 0.90) is the point where they
agree most evenly. At the shipped level with 1000 paths: **−0.178 / −0.058 / −0.126 %**, and the
level optimum does not move enough to be worth a second knob (0.94 trades 0.04 pp of dev for 0.06 pp
of val3).

**Verdict**: shipped for the eight com-equ assets. Crypto rejected — and instructively. On a level
grid of (0.90, 0.95, 1.00) crypto's hinge measured −0.093 / −0.060 / −0.047, which would have
shipped. Widening the grid to (0.85 … 1.05) let the *baseline* find a better level on test and the
gain became −0.093 / −0.062 / **+0.004**. The entire crypto effect was a level improvement wearing a
shape's clothes — Iteration 12's confound, caught only because the grid was widened for an unrelated
reason.

**Lesson**: two lessons, and the second is the one that nearly cost me. First: when a swept
parameter sits at a clean interior optimum, that is evidence about the best value *within the
functional form*, not evidence that the form is right. `seas_gamma = 1.35` had been swept four times
across this campaign and was correct every time; it was the best straight line through a curve, and
the diagnostic that found the curve took twenty minutes to write. Sweep harder is not the same as
look at the residual. Second: a level grid that does not bracket the optimum for *both* arms turns a
level difference into a shape result. I have written "re-optimise the level at every point" in this
journal since Iteration 12 and still got it wrong, because the grid was wide enough for the champion
and not for the candidate. The rule needs the stronger form: the level grid has to be wide enough
that the optimum is interior *for every cell being compared*, and if any cell's optimum is at an
edge the comparison is not yet a comparison.

**Official re-run.** Again only the eight com-equ legs changed. COM_EQU_24H 1820.7 → **1817.9**
against 2076.5 — **−12.32 % to −12.45 %**. The hinge is the largest single gain since the seasonal
exponent itself.

## Iteration 28 — The same convexity at the 1-minute step

**Hypothesis**: Iteration 27's finding is about the functional form of the calendar dependence, not
about tokenised equities. If it is real it should appear at the 1-minute step too. There the shipped
exponent is 0.85 — a *damping*, justified in Iteration 15 by bid-ask bounce and zero-return minutes
being width that does not follow time-of-week. That argument is about the *average* return; it says
nothing about whether the damping should be the same above and below the weekly mean.

**Change**: sweep `seas_gamma_lo` at the 1-minute step. No asset map is needed — only crypto has a
1-hour competition, so a step-keyed entry reaches exactly those five assets and leaves crypto-24h
untouched by construction.

**Result**, % vs the shipped 0.85, level re-optimised:

| g_hi | g_lo | dev | val3 | test |
|---|---|---:|---:|---:|
| 0.85 | 0.55 | −0.156 | −0.075 | −0.041 |
| 0.85 | 0.40 | −0.216 | −0.094 | −0.041 |
| 0.85 | 0.25 | −0.259 | −0.097 | −0.031 |
| 0.70 | 0.40 | −0.243 | −0.099 | −0.035 |
| 0.70 | 0.25 | −0.276 | −0.096 | −0.024 |

The first run of this sweep had val3's optimal level pinned at the top of the grid for every hinged
cell. By the rule written one iteration earlier those cells were not yet comparisons, so I widened
the grid to 1.15 and re-ran; every argmin is now interior and the numbers are unchanged. The rule
cost twenty minutes and bought the right to believe the table.

At the shipped level (0.95, 1000 paths, no grid), g_lo = 0.40 gives **−0.250 / −0.076 / −0.059 %**,
negative in 13 of 15 asset-window cells — only XRP is flat-to-positive on val3 and test. The shipped
level is already the optimum on two of three windows, so again no second knob.

**Verdict**: shipped, `seas_gamma_lo = {5: 1.35, 1: 0.40}`. The high side keeps 0.85.

**Lesson**: the direction is the same as com-equ's — the quiet end of the calendar is under-served —
and it is the *opposite* of what the 1-minute damping story predicts on its own. "1-minute prints
carry bounce and zero-return minutes, so damp the profile" was right on average and wrong at the
quiet end, where the bounce is a larger share of a smaller number and the true width is closer to
the weekly mean than any power of the profile puts it. Two independent horizons, two asset classes
and two opposite average exponents all agree that a single power is the wrong shape. That is the
result worth carrying out of this campaign, more than either number.

**Official re-run.** Only the five crypto-1h legs changed. CRYPTO_1H 798.7 → **798.2** against
831.9 — **−4.00 % to −4.05 %**.

## Iteration 29 — Re-reading the diagnostic that started the last two ships

**Hypothesis**: the bucket diagnostic (`agent/eda_profile.py`) that exposed the convexity should
now be flat on com-equ, and whatever it still shows is the next thing to fit.

**Change**: none. Re-ran the diagnostic against the shipped champion, both cadences (173 and
97 minutes, so neither can land on a round hour), all three windows pooled.

**Result**. Com-equ went from a clear U to essentially flat at the ends: decile ratios
1.044 / 1.023 / 1.072 / 1.115 / 1.099 / 1.118 / 1.069 / 1.063, hi/lo spread 1.019 at cadence 173
and 1.014 at 97. The hinge did what it was fitted to do. What is left is not a U and not a
monotone trend — it is a hump in the middle four deciles, i.e. the model is ~4 % too narrow
*near the weekly mean* and right at both extremes. That is curvature about the hinge point, which
no two-sided power can reach; it would need a quadratic in log-profile, a third shape knob.

Crypto-24h still shows a monotone decline, 1.175 at the quiet end to 1.044 at the busy end,
spread 0.888, reproduced at both cadences. That is the same shape the com-equ diagnostic had
before Iteration 27, and it is tempting.

**Verdict**: nothing to ship, and — more useful — nothing to try. Both residuals are inside the
noise of the metric, and I can say that with a number rather than a shrug. The identity
`E CRPS(σ)/E CRPS(1) ≈ 1 + 0.25·(log σ)²` converts a spread of bucket width errors into a CRPS
cost directly: for the com-equ hump, `Var(log ratio)` across deciles is about 9e-4, worth
**0.02 %**; for the crypto gradient it is about 2e-3, worth **0.06 %**. Iteration 27 measured the
crypto-24h hinge on CRPS itself and got **+0.004 %** — inside a prediction of at most 0.06 %, and
on the wrong side of zero. The diagnostic and the metric agree; the vein is exhausted.

**Lesson**: a residual diagnostic is not a to-do list. Before fitting anything it suggests, price
it in the units the score is actually in. The CRPS-vs-log-σ identity turns a width-ratio table
into an upper bound on the gain in about a minute of arithmetic, and that minute is what tells
you whether a day of sweeping is worth spending. Two of the three residuals I have chased this
campaign were worth less than 0.1 % before I started; this is the first time I checked first.
