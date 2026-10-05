# Forecasting price distributions for Synth SN50: what moved the CRPS and what didn't

*A campaign report for Synth miners. Everything here was derived from the public snapshot, the
synth-lib backtester and the Synth API; no proprietary model informed it.*

---

## 1. The problem, stated precisely

A Synth validator hands you an asset, a start time, a horizon and a step size, and asks for 1000
simulated price paths. It scores you with CRPS — lower is better. Three configurations are live:

| configuration | assets | horizon / step | points |
|---|---|---|---|
| CRYPTO_24H | BTC ETH SOL XRP HYPE | 86400 s / 300 s | 289 |
| COM_EQU_24H | XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500 | 86400 s / 300 s | 289 |
| CRYPTO_1H | BTC ETH SOL XRP HYPE | 3600 s / 60 s | 61 |

The first thing worth knowing is what the score actually is, because it is not "how realistic are
your paths".

**The validator score is a sum of *univariate* CRPS terms on relative price changes over
intervals.** For a 24-hour prompt:

| interval | number of terms | share of the score |
|---|---|---|
| 5 min | 288 | ≈ 61 % |
| 30 min | 48 | ≈ 25 % |
| 3 hour | 8 | ≈ 10 % |
| 24 hour (absolute) | 1 | ≈ 3.6 % |

There is no multivariate term anywhere. The joint distribution of a path matters *only* through
the marginals of aggregated returns. Roughly 86 % of a 24-hour score is the marginal distribution
of the 5-minute return at each of 288 future steps.

This has a blunt consequence: cross-asset correlation, jump clustering, order-flow realism,
momentum, and every other thing that makes a simulated path look like a real one are **unpriced**.
If you are spending effort there, you are spending it on something the scoring function cannot
see.

CRYPTO_1H is similar in spirit: 1/2/5/15/30/60-minute intervals, plus twelve `*_gaps` terms
(cumulative return from the start to each of minutes 5, 10, …, 60) worth ≈29 %, plus a small
volatility-CRPS block worth ~1 %.

## 2. The identity that decides everything

Because the score is a sum of scale-family CRPS terms, you can write down what a mis-calibrated
width costs. For a Gaussian forecast of scale σ against a truth of scale 1:

```
E CRPS(σ) = sqrt(2/π)·sqrt(1+σ²) − σ/sqrt(π)
```

minimised at σ = 1 with value 1/sqrt(π). Expanding around the optimum:

```
E CRPS(σ) / E CRPS(1)  ≈  1 + 0.25 · (log σ)²
```

I measured this empirically before deriving it, and the two agree to two digits: a **±20 % error
in overall scale costs about 1 % of CRPS**, and a 5 % error costs 0.06 %.

Read that the other way round and it is the whole campaign in one line:

> A *constant* scale error is nearly free. The entire recoverable CRPS is
> **Var(log σ̂ − log σ_true)** — the variance of your volatility forecast error in logs.

It even predicts the size of the prize. For BTC, sd(log realized 24-hour vol) ≈ 0.48, so a model
with a constant width gives up 0.25 × 0.48² ≈ 5.8 %. An oracle handed the realized volatility of
the very window it is predicting scored ≈6 % better than my best causal model on crypto-24h.
The arithmetic and the experiment agree.

One asymmetry matters in practice: the quadratic is symmetric but the exact loss is not. Too
*wide* costs linearly and without bound; too *narrow* saturates at +41 %. So the CRPS-optimal
forecast sits slightly *below* `exp(E[log σ])`, which is why every multiplier I fitted came out
under 1.

## 3. The model

```
r_k = scale(asset, step) · v_k · s(t_k) · m_k · z ,    price_k = last · exp(cumsum(r))
```

* **`s(t_k)` — time-of-week seasonality.** A 336-bucket (half-hourly) profile per asset and step
  size, fitted offline and committed as an artifact. The venues never close but the underlying
  markets do: mean |5-minute return| swings ≈1.8× across the week for crypto and up to ≈6× for
  the tokenised equities. The profile is raised to the power **1.35 at the 5-minute step and 0.85
  at the 1-minute step** — the two horizons want opposite corrections, and three disjoint windows
  agree on both signs (§5).
* **`v_k` — conditional volatility of the *deseasonalised* returns.** A geometric blend of EWMA
  variances at half-lives of 1 h / 6 h / 48 h, reverting log-linearly toward the 7-day level with
  a half-life of 96 steps.
* **`z` — standardised Student-t(4).** Gaussian innovations cost about 1 %.
* **`m_k` — a persistent AR(1) factor in log-volatility**, drawn per path and renormalised so
  that *every step's variance is exactly unchanged*. It does not touch the per-step marginal; it
  fattens the **aggregated** intervals, which a deterministic σ path leaves too close to Gaussian
  by the CLT and which are 39 % of a 24-hour score. On for the com-equ assets (stationary sd
  0.30), off for crypto, where it flips sign between windows.
* **`scale(asset, step)`** — 0.90 at the 5-minute step and 0.95 at the 1-minute step for crypto;
  one fitted multiplier each for the eight com-equ assets (0.72–0.96 of the crypto width).

One configuration serves all three scoring configurations. It runs offline in ~50 ms per prompt
at 1000 paths, against a practical ceiling of about 5 s.

Everything in the model is there because it survived a check on a validation window it was not
tuned on. Most of what I tried did not.

## 4. What worked

### Time-of-week seasonality (−1.9 %, plus −0.5 % from the exponent) — the largest single win

Volatility in these markets is strongly deterministic in time-of-week, and a constant σ is
simultaneously too wide overnight and too narrow at the US cash open. CRPS charges for both.
This is worth −1.9 % across 13 assets, concentrated in com-equ (NVDAX −3.5 %, TSLAX −3.8 %,
GOOGLX −3.2 %). A further −0.5 % came from the *exponent* on the profile, which I had wrong for
most of the campaign — see "The seasonal exponent was confounded with the level" in §5.

### Student-t(4) innovations (−1 %)

Essentially the entire prize available from distribution *shape*. Swapping t(4) for a normal at
identical scale costs ~1 %; `nu=3` and `nu=6` are both worse; an empirical bootstrap of the
context returns is a wash.

### Volatility mean reversion (−0.6 %)

Reverting the current EWMA level toward the 7-day level with a 96-step half-life. This was the
*only* one of the vol-dynamics knobs that mattered — see §5.

### A per-asset width multiplier for com-equ (−0.13 % out of sample)

Fitted on a window preceding both validation windows. Small, but it is the part of the per-asset
calibration that demonstrably generalises, and the diagnosis is the interesting bit (§5).

### Fattening the aggregated intervals (−0.10 % on com-equ, three windows)

The last thing I found, and the only one that came from reading the score's *shape* rather than
its level. About 39 % of a 24-hour score is charged on intervals longer than one step — 30-min,
3-hour, 24-hour. (The 1-hour configuration charges a further ~29 % on the cumulative `*_gaps`
terms, but those turn out to be correctly dispersed already — see §5.) With a σ path
that is deterministic given the context, those aggregates are sums of independent innovations,
so the CLT makes them close to Gaussian; real ones are not, because volatility clusters *inside*
the window.

A persistent AR(1) factor in log-volatility fixes that, but only if it is renormalised so every
step's variance is exactly unchanged — otherwise it also perturbs the 61 % of the score that was
already calibrated, which is why the constant-per-path version of the same idea (`vol_of_vol`,
the half-life→∞ limit) had failed earlier. With the constraint it is worth −0.096 %, −0.096 %,
−0.102 % on three disjoint com-equ windows. On crypto it changes sign between windows and is
left off.

It comes with a level correction that §2 predicts: a fatter mixture at fixed variance is
*narrower in the bulk*, so the CRPS-optimal width moves up — 0.90 → 0.95, on all three windows.
Ship the shape change without the level change and you give most of it back.

### The calendar's effect on width is convex (−0.13 % com-equ, −0.13 % crypto-1h)

The largest win of the campaign was the time-of-week profile, and the second largest was the
exponent applied to it. The exponent had been swept four times, always landing near 1.35 at the
5-minute step with a clean interior optimum. What I had never done was look at the *residual*.

`agent/eda_profile.py` pools every 5-minute step of every simulated path and reports
mean|realized| / mean|simulated| by decile of the bent profile value — the same statistic as the
gaps diagnostic, on off-hour cadences for the same reason. For com-equ it comes out U-shaped:

| profile decile | 0.34 | 0.53 | 0.69 | 0.83 | 0.98 | 1.17 | 1.47 | 2.09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio (>1 = too narrow) | 1.213 | 1.069 | 1.027 | 0.999 | 0.945 | 1.018 | 1.050 | 1.153 |
| % of realized steps exactly zero | 16.1 | 7.4 | 4.6 | 3.3 | 2.8 | 2.0 | 1.4 | 1.0 |

Relative to an average bucket the model is too narrow at *both* ends of the calendar at once. A
power of the profile is linear in log-profile, so no value of the exponent can produce that shape;
the sweeps had been finding the best straight line through a curve.

The bottom row explains the left half. The deadest decile has 16 % of its realized 5-minute
returns *exactly zero* and still a 21 % larger mean absolute size than the ensemble. A closed
session is not a quiet diffusion — it is a point mass at zero plus occasional gap-sized prints —
and the CRPS-optimal width for that mixture is wider than the width that matches its bulk. The
buckets where the volatility estimator sees the least action are precisely the ones it under-serves.

The cheapest way to buy convexity is a hinge: one exponent above the weekly mean and another below.
`sv_model` had carried an unused `seas_gamma_lo` for exactly this shape since an earlier experiment
and it had never been swept. Swept jointly it gives a broad diagonal ridge, interior on both axes,
and (1.80, 0.90) is where three disjoint windows agree most evenly: −0.178 / −0.058 / −0.126 %.

The same thing happens at the 1-minute step, where the shipped exponent is a *damping* (0.85, on
the grounds that bid-ask bounce and zero-return minutes are width that does not follow the
calendar). That argument is about the average minute and says nothing about the two sides
separately. Dropping the quiet-side exponent to 0.40 is worth −0.250 / −0.076 / −0.059 % on
crypto-1h, in 13 of 15 asset-window cells. Two horizons, two asset classes, two opposite average
exponents — and all of them say a single power is the wrong shape and the quiet end of the calendar
is under-served.

Crypto at the 24-hour horizon is the exception and it is worth reading §5 for why: it looked like a
−0.05 % win and was a level difference in disguise.

### Looking back further than the context reaches (−0.04 % on com-equ, three windows)

The model reverts its conditional volatility toward `v_long`, a slow EWMA of the context. But the
context *is* seven days, so that target is an estimate from one week, and a week that happened to be
hot leaves a reversion target which is itself an overshoot. Volatility mean-reverts on a monthly
timescale too, and nothing inside a seven-day window can see it.

So I fitted one long-run level per asset and step offline — the RMS of the winsorised log return,
deseasonalised by the *same* bent profile the model uses, or the blend would be a units error — and
committed it as an artifact, blending it into the target in logs at weight 0.2. On com-equ that is
worth −0.168 / −0.026 / −0.039 % on the three windows at the width that ships, with no level
correction of its own. Crypto changes sign between windows at 24 hours and is flat at one hour, and
is left at 0.

Two honest caveats. First, the size is *below* the 0.1 % bar everything else here had to clear; it
is shipped on the consistency of the sign — negative at every blend from 0.1 to 0.3, at four of five
widths, in 21 of 24 asset-window cells, and under two vintages of the artifact — rather than on the
magnitude. Second, I expected staleness to explain why the first window is four times better than
the others, and it does not: refitting the level immediately before each window buys 0.007 pp. The
first window is just a kind one.

The reason this is worth a paragraph despite being the smallest thing in this section is the
principle underneath it. §5 argues at length that a feature only helps if it is not already flowing
through realized volatility, and I had taken that to mean *forward-looking* information. This
feature is entirely backward-looking. What makes it non-redundant is that it looks back further than
the served context extends — the interface hands the model seven days, so anything on a longer
timescale is outside it by construction and can only get in as an artifact. That is a second source
of non-redundant information, and unlike the forward-looking one, it needs no data I was not
already allowed to have.

## 5. What didn't work — the more useful half

### Median-based seasonality: +10 % on XAU

My first seasonality profile used the bucket **median** |return| and made things dramatically
worse. In a thin market's quiet hours most 5-minute returns are *exactly zero* — no trade — so
the median collapses toward 0 while the width CRPS actually wants does not. The model became
absurdly narrow overnight and was punished for it.

Switching to a **winsorised RMS** (clipped at 6 robust sigmas; XRP's raw return kurtosis on this
snapshot is about 7000, so a plain RMS is set by single prints) turned a +0.35 % regression into
a −1.9 % improvement.

> When a feature makes things worse, the feature is usually right and the *estimator* is wrong.
> And in this data, "no trade" and "no move" are the same observation.

### The seasonal exponent was confounded with the level — the costliest mistake here

For most of this campaign the profile was damped, `s^0.85`, because a sweep said 0.85 beat 1.0 on
both validation windows. That sweep held `scale` fixed, and that is the whole error.

The profile is normalised to mean 1, so by Jensen `E[s^γ]` *grows* with γ: raising the exponent
silently widens the model. A 1-D sweep of γ at fixed scale therefore measures the level — and §2
says the level is the one thing that is nearly free. Re-optimise `scale` at every γ and the
answer moves to an interior optimum near **1.35**, on both windows:

| γ | dev (min over scale) | test (min over scale) |
|---|---:|---:|
| 0.85 | 2857.3 | 2637.7 |
| 1.15 | 2852.3 | 2624.0 |
| 1.35 | ~2852 | ~2620 |
| 2.00 | 2887.5 | 2630.1 |

Since γ had now been chosen using both validation windows, the A/B ran on a third window
(2026-07-24 → 08-25) untouched by anything earlier: **−0.48 %** on the 24-hour configurations,
and **+0.34 %** on crypto-1h. The two horizons want opposite corrections, consistently, on all
three windows — so the exponent is keyed on the step size, `{5: 1.35, 1: 0.85}`, which leaves
crypto-1h exactly unchanged.

Why would a measured profile need *amplifying*? Because every step of building it shrinks it:
winsorising at 6 robust sigmas, a width-5 circular smoother, a 0.25–4.0 clip. The estimator
chosen to stop the profile being set by single prints also flattened it by about a third, and
γ > 1 is undoing that regularisation.

> Sweep a *shape* parameter with the *level* re-optimised at every point, or do not believe the
> answer. This one was wrong from the moment seasonality was introduced and survived two
> validation windows without complaint.

### Range-based volatility estimators (Parkinson, Garman-Klass)

The model reads only `close` and throws the minute high/low away, and the textbook says a range
estimator is several times more efficient for the same window. Measured properly — centred
log-error sd against realized 288-step volatility, since the level is free — Parkinson is worth
**−0.16 %** and Garman-Klass **−0.27 %**. Both are *worse*, and GK, which leans hardest on the
range, is worst.

Parkinson's efficiency assumes the extremes of a continuously observed path. What is in this
store is the extremes of however many prints happened, so the estimator's variance is governed by
trade intensity — which is exactly what varies most across the week in these assets. Rejected
without a CRPS run.

### A finer time-of-week grid

The profile uses 30-minute buckets smoothed over five of them — a 2.5-hour window, which is wide
next to a cash open. Refining it to 15- and 10-minute buckets is worse: +0.9 % and +1.2 % on
com-equ, nothing either way on crypto. Coarsening (60-minute buckets, or wider smoothing) is
also worse. The shipped resolution sits on the optimum from both sides.

The grid was built to separate two explanations, and the answer is unambiguous: **it is
observations per bucket, not the width of the smoothing window.** The 15-minute/9-bucket variant
spans 135 minutes, essentially the shipped 150, and still loses 0.9 % — same effective smoothing,
half the sample per bucket. Nine months of prints is simply not enough to estimate 672 numbers
per asset, and at 10-minute buckets SPCX had no bucket left above the 20-observation threshold at
all.

It also falsified something I had written down as an explanation. I had said the 1-minute step
wants γ<1 because 30-minute buckets are coarse relative to a 1-minute step; refining the grid 6×
leaves the 1-minute results unchanged to three decimals. The damping is a property of the return
— a 1-minute print carries bid-ask bounce and zero-return minutes, width that does not follow
time-of-week — not of the profile. The parameter was fitted correctly and transfers; the story
attached to it was wrong. Worth stating plainly because an unfalsified story about *why* a
parameter works is the thing most likely to send the next experiment in the wrong direction.

### Trade count and volume: real regression signal, zero CRPS

`volume` and `trade_count` sit unused in every context, and §2's logic says to check them: for a
thin asset, trade count is the most direct measure of whether the next five minutes *can* move.
The diagnostic agreed. Regressing log realized future volatility on the model's own volatility
predictor, then on that plus deseasonalised activity features, fitting on the first 60 % of each
history and scoring the last 40 %, the implied CRPS gain was +0.50 % on TSLAX, +0.28 % on
GOOGLX, +0.24 % on NVDAX.

Then the CRPS test: one bounded coefficient, width multiplied by `clip(exp(β·a), 1/1.3, 1.3)`
with `a` the fast-minus-slow log EWMA of deseasonalised trade count, scale re-optimised at every
β. Every β is worse than zero — both signs, both windows, both asset groups, monotonically in
|β|: β=+0.10 costs +0.24 %/+0.11 % on com-equ and +0.13 %/+0.05 % on crypto, rising to about
+1 % at β=+0.55, and β=−0.20 is worse too.

The gap between the two is the lesson, and it is not noise. Squared log error pays full price
for mispredicting a dead market; CRPS on relative price changes barely charges for it, because
when nothing trades the realized change is ≈0 and any reasonably narrow forecast already scores
well. Which is why the asset where the diagnostic screamed loudest (XAU, +9 %, on a panel whose
baseline error sd is ten times everyone else's) is the asset whose CRPS is least sensitive —
and why the diagnostic's positive mean was that one asset. Excluding XAU its mean is −0.16 %.
**Convert a diagnostic into the scored objective before believing it, and read the
cross-section rather than the mean.**

### A fitted volatility regression: better forecast, worse score

§2 says the prize is Var(log σ̂ − log σ). That is a regression target, so I regressed it: log
future volatility on the log EWMA vols of the deseasonalised context at five half-lives, per
asset, at five horizon knots, so the fitted cumulative-variance curve could be differentiated
into a per-step term structure.

In-sample R² was 0.36–0.73. Measured against realized volatility it *was* a better forecast than
the EWMA blend on crypto (log-error sd 0.281 vs 0.314 for BTC). It still lost on CRPS, and it
took three separate fixes to even find out why:

1. **`np.interp` clamps below its first knot.** The cumulative variance curve came out flat over
   the first steps — literally zero dispersion at steps 1 and 2. CRPS punishes a zero-width
   forecast savagely.
2. **The fitted "term structure" was almost entirely small-sample log bias.** The target is
   `log(RMS of h returns)` and log is concave, so `E[log RMS₃]` sits ~45 % below `log σ` for
   fat-tailed returns while `E[log RMS₂₈₈]` is within 0.3 %. Fitting each horizon separately
   manufactures a steep upward slope that is pure estimator artefact. The fix is to measure the
   bias on non-overlapping blocks of the same series and subtract it.
3. **The A/B test was not testing what I thought.** With a 96-step reversion half-life over a
   288-step path, most of the path has already reverted to the long-run level, so "swapping in
   the regression" barely used the regression.

After all three fixes, at each model's individually optimal multiplier the fitted and EWMA levels
**tie** (mean excess loss 3.40 % vs 3.37 %). Out of sample on the later window, the regression was
worse on exactly the assets it was supposed to help: HYPE +1.12 %, XRP +0.74 %, SOL +0.53 %.

Rejected. The code stays in the repo as a *measurement instrument* — its residual sd is what tells
me 3.6 % (24h) and 2.5 % (1h) of CRPS remain on the table.

### The leverage effect

Equity volatility rises after price falls; it is one of the most robust stylised facts there is,
and both standard ways of writing it down are cheap to add to a model that already carries an
EWMA. I tried both: a multiplier driven by the exponentially-weighted *downside* return only, and
one driven by the z-scored signed return, each over half-lives from 30 minutes to 2 days and
coefficients from small to implausible. Every setting was flat or worse on every window and both
asset classes; the best cell of the whole grid was −0.02 %, which is a tie.

### Why those three failures are the same failure

Range estimators, trade count and volume, and the leverage effect are three unrelated features
from three unrelated literatures, and they failed in exactly the same way: real signal in a
regression against future volatility, and zero CRPS once they were added to the model. The reason
is structural. All three act on future volatility *through* realized volatility — a wide bar, a
busy minute and a sharp drop are all ways of saying the market just moved a lot — and a 12-minute
EWMA of realized volatility has already read that. What looks like new information is old
information arriving by a shorter path.

The useful half of this is the corollary. Information that is genuinely **forward-looking** —
known at the prompt time but not derivable from the price history — is not redundant with the
EWMA and is where the gains are. There is exactly one such feature in this model, the time-of-week
profile, and it is the largest single win of the campaign by an order of magnitude. That is not a
coincidence, and it is the first thing I would tell anyone starting here: before testing a
feature, ask whether a fast EWMA of realized volatility can already see it. If it can, test the
*increment* over the EWMA from the start rather than the feature's standalone predictive power,
and you will save the four hours I spent finding this out three times.

### A hinge on the crypto-24h profile, which was a level difference in disguise

The same two-sided exponent that is worth −0.13 % on com-equ measured −0.093 / −0.060 / −0.047 %
on crypto at the 24-hour horizon — negative on three disjoint windows, exactly the pattern this
report has spent nine sections calling evidence. It would have shipped.

It was an artifact of the level grid. Every cell in that sweep re-optimised the width level over
(0.90, 0.95, 1.00), which is the rule §5 introduces two subsections above. Widening the grid to
(0.85 … 1.05) for an unrelated reason let the *baseline* find a better level on the test window,
and the gain collapsed to −0.093 / −0.062 / **+0.004** %. The entire effect was a level improvement
wearing a shape's clothes.

The rule as I had been applying it — "re-optimise the level at every point" — is not strong enough.
The grid has to bracket the optimum for *every* cell being compared, including the baseline, and if
any cell's optimum sits at an edge the comparison is not yet a comparison. One iteration later a
crypto-1h sweep came back with the baseline's optimum interior and every candidate's pinned at the
top of the grid; widening it cost twenty minutes and left the conclusion unchanged, which is the
outcome you want from a check, not a reason to skip it.

### A per-asset tail index

The Student-t index is a single global 4.0, swept twice and flat both times. `agent/nufit.py` fits
one per asset from standardised residuals — deseasonalised by the model's own profile and divided
by the model's own causal EWMA, so the estimate is of the innovation and not of volatility
clustering, and read off a quantile ratio because the sample kurtosis of a t(4) has infinite
variance. The fit has real structure: crypto 4.1–5.8, tokenised equities 3.2–4.0, which is what an
overnight gap and a session open do to residuals a smooth EWMA cannot track.

Blending it in at any weight from 0.25 to 1.0 lands inside ±0.1 % on every configuration with no
consistent sign. A correctly measured, genuinely asset-specific, genuinely non-redundant quantity,
worth exactly nothing — because §2 says the score is sensitive to *width* and almost nothing else,
and between nu = 3 and nu = 6 the bulk of a standardised Student-t barely moves. The lesson is to
measure the score's sensitivity to a parameter before investing in estimating it well.

### Almost every volatility-dynamics knob

Swept `revert_halflife` ∈ {24,48,96,192,384}, three half-life triples, three weight vectors,
vol-of-vol ∈ {0.2,0.35,0.5}, winsorisation ∈ {2.5,3.5,6}, and an L1 variance estimator. Only
`revert_halflife` moved the mean by more than 0.1 %. This is exactly what §2 predicts: these knobs
move the *level*, and the level is nearly free.

Re-swept once more at the very end, after the vol-of-vol factor below had changed the shape of the
σ path, in case the factor had displaced any of them. It had not: every alternative half-life set,
weight vector, reversion half-life and winsorisation threshold came back inside ±0.1 % on all
three windows, and the only candidate with a consistent sign was consistently worse. The factor
is orthogonal to the drift of the deterministic σ path by construction — it preserves each step's
variance exactly — and the re-audit is the confirmation that the construction held.

The vol-of-vol entry in that list deserves a footnote, because the same idea came back later and
worked. What was swept there was a *constant-per-path* log-level shock, which fattens the
aggregated intervals and the per-step marginal alike — and the per-step marginal, 61 % of the
score, was already calibrated, so it lost more than it gained. Constraining the factor to leave
every step's variance exactly unchanged is what turned it into the −0.10 % in §4. The idea was
never wrong; the unconstrained version of it was.

### Per-asset scales for crypto

The same calibration that produced good com-equ multipliers produced crypto multipliers that were
noise. Mapping all 13 assets gave −0.21 % on the first window and **+0.13 % on the second**;
com-equ only gave −0.21 % and −0.13 %.

What separated them was *stability under refitting*. Refitted four months apart on disjoint data,
the com-equ multipliers barely moved — GOOGLX 0.746 → 0.741, NVDAX 0.761 → 0.755, SPCX 0.831 →
0.835, XAU 0.764 → 0.773 — while the crypto ones wandered (HYPE 1.074 → 0.984). Parameter
stability under refitting predicted out-of-sample transfer better than in-sample gain did.

### A bespoke crypto-1h model

A 60-minute horizon at 1-minute resolution *ought* to want faster volatility dynamics. It does
not: the (2,10,60)-minute blend is 0.41 % worse than (12,72,576). Slower half-lives, a 7-day pool
and empirical innovations together looked worth −0.43 % on the first window and delivered nothing
on the second (all variants within ±0.08 %, the unchanged configuration marginally ahead). So
crypto-1h ships with the 24-hour settings untouched.

Its *width level*, on the other hand, did not transfer, and that is the one thing about the
1-hour configuration that ships differently. `scale = 0.90` was fitted on 5-minute returns over
24-hour prompts, and a 1-minute return is a different random variable: about half of them are
exactly zero on the thinner books, so the RMS the estimator reports is depressed relative to the
width CRPS actually wants. The 1-minute optimum is 0.95, worth −0.085 %/−0.215 %/−0.049 % on the
three windows. The per-asset optima around it are all within 0.2 % of the flat value, so the level
is keyed on the *step size* and nothing else: `scale = {5: 0.90, 1: 0.95}`. Applying the same 0.95
to crypto-24h flips the sign, which is the tell that this is a property of the 1-minute return and
not a better global calibration.

### Two shape knobs with the right sign and the wrong size

Late in the campaign I tried the two remaining ideas that were not redundant with the EWMA (§5,
"Why those three failures are the same failure"), both of them about the *shape* of the width
rather than its level.

The first: the multiplier in front of the volatility path is a constant, so the same shrink applies
to the first five minutes of a 24-hour path as to the last. It should not. §2's asymmetry puts the
optimal width *below* `exp(E[log σ])` by an amount that grows with the uncertainty in log σ — and
that uncertainty is near zero at step 1, where the EWMA has just been measured, and grows toward
the unconditional spread by step 288. The shrink should deepen with the horizon.

The second: the seasonal profile is a fixed shape scaled by the EWMA, but a macro shock does not
respect trading hours, so in a hot regime the overnight buckets should catch up and the day/night
ratio should compress. Neither the calendar nor the realized volatility contains that on its own —
only their product — so it escapes the redundancy argument.

Both were swept with the level re-optimised at every point, on three windows and three
configurations. Both came back with **the sign the theory predicts and a magnitude an order of
magnitude below the threshold**: every positive horizon ramp is worse than its negative mirror in
all nine cells, and on com-equ a +1.0 regime coefficient costs 0.5–0.6 % while −1.0 costs 0.14 %,
so both surfaces lean the right way. The best cell anywhere in either sweep is −0.06 %, and no two
windows agree on where it is.

That pair is the most informative negative in this report. The model is not missing a mechanism; it
is sitting at the bottom of a flat basin. §2 even says why the horizon ramp had to be small: the
quadratic term in the CRPS expansion is symmetric, so shrink-for-uncertainty is a third-order
effect and cannot be worth more than a rounding error.

### Refitting the per-asset multipliers against the shipped configuration

The eight com-equ multipliers were fitted when the champion had a different seasonal exponent and
no vol-of-vol factor; when the factor landed I corrected the whole map by one uniform number.
Eight assets with different seasonal amplitudes should not share a correction, so I refit all eight
against the champion's own gamma, factor and profile, on a window preceding every validation
window. Every multiplier moved down by 0.04–0.09 — very nearly a constant offset in logs, i.e. a
*level*, which §2 says is nearly free and which the level sweep absorbs. Out of sample the refit
map is within 0.06 % of the shipped one and disagrees in sign across windows. The shape of the map
is what generalises; its position was already right.

### The `*_gaps` block in CRYPTO_1H — and a clock that nearly fooled me

Twelve of the eighteen interval terms in the 1-hour configuration are the cumulative return from
the prompt start to minute 5, 10, … 60, about 29 % of that budget. They all begin at t=0, so they
weight the opening minutes of the path much more heavily than a `30min` term does, and the model
knows nothing about them. It looked like the largest untouched piece of structure in the scoring
rule.

The diagnostic (`agent/eda_gaps.py`) is the ratio of each term's realized typical size to the
simulated ensemble's, pooled over five assets and three windows; above 1 means too narrow. On my
usual 3-hour prompt grid it showed a 28 % under-dispersion on `0_5min_gaps` decaying monotonically
to the interval level by minute 20, in all five assets — clean, large and structural.

It was my own clock. A 180-minute cadence lands on :00 every time, so "the first five minutes of
the path" meant "the first five minutes of a round hour" in every single prompt, and round hours
are when candles close and volatility spikes. Re-run at 173 and 97 minutes, which walk across the
minute-of-hour the way the validator's irregular prompt times do, the effect vanishes:

| term | cad 173 | cad 97 | cad 180 |
|---|---:|---:|---:|
| `1min` | 1.106 | 1.086 | 1.115 |
| `5min` | 1.043 | 1.027 | 1.053 |
| `30min` | 1.047 | 0.997 | 1.033 |
| `0_5min_gaps` | 1.059 | 0.981 | **1.225** |
| `0_10min_gaps` | 1.054 | 1.004 | **1.175** |
| **gaps block** | 1.043 | 0.993 | 1.091 |
| **interval block** | 1.056 | 1.029 | 1.060 |

Off the round hour the gaps ladder is flat and sits *at or slightly below* the interval ladder:
the model is marginally over-dispersed there, by 1–3 %, which §2 prices at 0.01 % of those terms.
The gaps are already paid for by the terms that share their horizon. (`0_60min_gaps` and
`60min_abs` come back identical at every cadence, which is the check that the measurement is
right — same horizon, same origin.)

The lesson is worth more than the result. A 3-hour cadence is harmless when *comparing two
models*, because both see the same prompts and the bias cancels; every sweep in this report is
safe for that reason. It is not harmless in a calibration diagnostic, where the absolute level is
the entire finding. Sample the prompt grid the way the validator does, or the grid becomes the
finding.

### Return term-structure correction

Empirical variance ratios are below 1 (BTC 0.92 at 5-hour horizons, AAPLX 0.67), so an i.i.d.
model over-disperses the 30-minute and 3-hour blocks. Matching them needs an MA(1) coefficient of
about −0.04, which corrects a ~4 % scale error on 39 % of the budget. By §2 that is worth
0.25 × 0.04² ≈ 0.04 % of those terms. Computed, not implemented — the identity in §2 is very good
at telling you what *not* to build.

## 6. Results

Backtested with `agent/predict.py` (the official `synth_lib.backtester`), 7 days of prompts,
1000 simulated paths — the same count the validator serves. "Starting point" is the log-normal
martingale this repo ships with, run through exactly the same path. The 24-hour legs end
2026-09-13; the 1-hour leg ends 2026-09-12, because the backtester pulls minute prices out to
`max(start_time) + 26 h` and the 1-hour prompt set otherwise crosses the edge of the snapshot
(journal, Iter 10). Both models are scored on identical prompt sets.

**CRYPTO_24H**

| asset | prompts | champion CRPS | starting point CRPS | change |
|---|---:|---:|---:|---:|
| BTC | 159 | 2128.3 | 2274.3 | -6.42% |
| ETH | 155 | 2980.8 | 3144.3 | -5.20% |
| SOL | 168 | 3735.1 | 3873.4 | -3.57% |
| XRP | 169 | 3859.5 | 4070.9 | -5.19% |
| HYPE | 162 | 4934.6 | 5037.5 | -2.04% |
| **mean** | | **3527.7** | **3680.1** | **-4.14%** |

**COM_EQU_24H**

| asset | prompts | champion CRPS | starting point CRPS | change |
|---|---:|---:|---:|---:|
| XAU | 169 | 1362.5 | 1556.7 | -12.47% |
| NVDAX | 167 | 1613.0 | 1790.4 | -9.91% |
| GOOGLX | 167 | 1486.3 | 1642.3 | -9.50% |
| TSLAX | 157 | 1975.3 | 2416.1 | -18.24% |
| AAPLX | 168 | 1569.0 | 1886.3 | -16.82% |
| WTIOIL | 167 | 3172.7 | 3376.6 | -6.04% |
| SPCX | 168 | 2731.8 | 3269.6 | -16.45% |
| SP500 | 161 | 632.9 | 673.7 | -6.05% |
| **mean** | | **1817.9** | **2076.5** | **-12.45%** |

**CRYPTO_1H**

| asset | prompts | champion CRPS | starting point CRPS | change |
|---|---:|---:|---:|---:|
| BTC | 990 | 472.7 | 508.8 | -7.08% |
| ETH | 956 | 666.9 | 703.7 | -5.24% |
| SOL | 977 | 878.1 | 904.9 | -2.96% |
| XRP | 979 | 892.9 | 937.7 | -4.78% |
| HYPE | 992 | 1080.5 | 1104.5 | -2.17% |
| **mean** | | **798.2** | **831.9** | **-4.05%** |

The champion is ahead on **all 18** asset/configuration combinations. The spread across
configurations is itself informative: com-equ gains three times what crypto gains, because the
tokenised equities are where the deterministic time-of-week structure lives. A constant-width
model is grossly miscalibrated across a market that is open six and a half hours a day, and §2
says that miscalibration is priced as a *variance* of log width, not a level.

The gain is smallest on HYPE (−2.0 % / −2.2 %), the asset with the thinnest history and the
noisiest volatility profile — consistent with the picture in §5 that what generalises is the part
of the estimate that is stable under refitting.

For scale: over 2026-07-10→12, the *best miner in the live field* on BTC scored 1841.4, with a
median of 1997 and p75 of 2016 across ~200 miners — many of them identical to within 0.1, which
suggests a widely shared template. The starting-point model in this repo is not competitive with
that; the champion is well inside it.

## 7. Limitations

* **The evaluation window is 10 days of live trading that I cannot see.** Every number here is a
  backtest. The com-equ multipliers are stable across four months of history, but a volatility
  regime change in the evaluation window would hurt com-equ more than crypto.
* **The three committed artifacts are fitted on the whole snapshot** — the time-of-week profiles
  (`tow_full.npz`), the long-run volatility levels (`vlong_full.npz`) and the per-asset exponent
  maps — so the backtested numbers in §6 are mildly optimistic relative to a truly causal run.
  The validation numbers quoted throughout §4–§5 all use artifacts fitted strictly before their
  window, and the effects survive there. For the evaluation window this cuts the other way: the
  artifacts are 10 to 24 days stale by the end of it. The long-run levels were tested for exactly
  this (refit at three cutoffs, §4) and move by less than the effect they carry; the profiles are
  a calendar and do not go stale on that timescale.
* **The two residual mis-calibrations I know about are priced, not fixed.** After the two-sided
  exponent, com-equ is ~4 % too narrow near the weekly mean and crypto-24h runs a monotone width
  gradient across the profile. Both are real and reproduce at two prompt cadences. Neither is
  worth fitting: §2's identity bounds them at 0.02 % and 0.06 % of CRPS respectively, and a direct
  CRPS test of the crypto one returned +0.004 %.
* **Two tokenised equities (SPCX, SP500) have short histories** — SP500 only from the mid-July
  SPYX retirement — so their profiles and multipliers rest on less data than the others.
* **The model has no drift and no cross-asset structure.** Both are defensible given §1. The
  second is the largest structural omission, but it is not one a miner can fix: `simulate()` is
  handed one asset's context per call, so the cross-section is not observable at serve time (§8).
* **The last two parameters had no untouched window left.** The AR(1) dispersion (0.30) and the
  level that goes with it were chosen by looking at all three validation windows at once. They
  agree to within 0.006 % across those three, which is why I believe them, but "consistent on
  three windows I looked at" is a weaker claim than the earlier results, where a window was held
  back and only consulted once. Same for the 1-minute level. Both are small effects, and both are
  the kind §2 says are cheap to get slightly wrong.
* **A halted venue is handled but not modelled.** If all 10 080 context closes were identical
  the volatility estimate would collapse to a numerical guard and the served distribution would
  be degenerate, so the conditional level carries an absolute floor — sized ~7× below anything a
  live asset produces, and verified bit-identical over 9802 real prompts, so it costs nothing.
  It caps that loss rather than removing it: a venue that reopens with a gap would still be
  badly scored, just 15× less badly.
* **`num_simulations` is honoured as passed**, but everything was tuned at 1000. Empirical CRPS is
  biased upward at small sample sizes; at 100 paths the absolute numbers would be worse, though
  the ranking between variants was stable in spot checks.

## 8. What I would do with more budget

1. **Cross-asset volatility pooling** — which would need a change to the interface, and that is
   the interesting part. BTC/ETH/SOL/XRP/HYPE share a common volatility factor and the tokenised
   equities share the US cash session, so a common factor plus per-asset loadings should beat 13
   independent EWMAs, most of all for the assets whose own history is thin. But `simulate()`
   receives exactly one asset's `context_prices`, so a miner cannot do this at serve time without
   smuggling state between calls. If the subnet wants better-calibrated thin assets, handing the
   model the whole cross-section is the cheapest change available to it.
2. **Fit the CRPS objective, not a squared-log-error proxy.** My regression minimised the wrong
   loss and respected neither the asymmetry in §2 nor the seasonal weighting. The CRPS-optimal
   forecast is a different, better-posed estimation problem.
3. **Revisit `volume` and `trade_count` — but against CRPS from the first line.** §5 rejects the
   bounded version: a single coefficient on trade-count activity is worse at every value tried.
   What survives is the per-asset cross-section (TSLAX, GOOGLX, NVDAX all positive in the
   diagnostic) and the fact that I only tested one functional form. A per-asset coefficient
   fitted directly on CRPS, on the three tokenised equities where the signal was consistent,
   is the version I would try next — and I would drop it again if it did not transfer across
   two windows.
4. **Go further on "the multiplier should depend on the seasonal level."** The exponent in §5 is
   the one-parameter version of that idea and it paid −0.5 %. The unconstrained version — a
   fitted monotone map from profile value to multiplier, per asset class — is the obvious next
   step. Note that it is a map from the *deterministic* seasonal level, which is the one thing
   in §5 that kept paying; the two attempts to read the same regime off live data instead,
   price range and trade intensity, both failed.
5. **Push on the score's shape, not its level.** §4's last entry is a single AR(1) factor with
   one fitted number, and it is the only thing that has ever paid on the aggregated intervals.
   The obvious extensions: a term structure for it (the 3-hour and 24-hour terms want more
   clustering than the 30-minute one), and fitting it per asset rather than per group. I stopped
   because the campaign ran out of budget, not because the seam was exhausted.
6. **Find more forward-looking information.** The framing in §5 says the only features that can
   pay are the ones a fast EWMA cannot already see, and the time-of-week profile is the crudest
   possible version of that: it knows the US cash open happens every weekday, but not that this
   particular Wednesday carries a CPI print or an NVDA earnings date. A scheduled-event calendar
   is the obvious next one — and it is the only idea on this list that is *certain* not to be
   redundant with what the model already has.

## 9. Reproducing this

```bash
uv run python agent/seasonality.py                  # rebuild the time-of-week artifacts
uv run python agent/calibrate.py dev                # refit the per-asset multipliers
uv run agent/predict.py --asset BTC --days 7 --eval-end 2026-09-13 --num-simulations 1000
bash agent/official_eval.sh    agent /tmp/off.json /tmp/off.log   # the 13 twenty-four-hour runs
bash agent/official_eval_1h.sh agent /tmp/off.json /tmp/off.log   # the 5 one-hour runs
uv run python agent/results_table.py off_champ.json off_base.json # renders the tables in §6
uv run python agent/eda_range.py                    # §5: range estimators, the diagnostic
uv run python agent/eda_activity.py                 # §5: activity features, the diagnostic
uv run python agent/exp_activity_crps.py            # §5: activity features, the CRPS test
uv run python agent/exp_resolution.py               # §5: time-of-week resolution (24h legs)
uv run python agent/exp_resolution.py 1h            # §5: the same at the 1-minute step
uv run python agent/exp_scale_1h.py                 # §5: the width level at the 1-minute step
uv run python agent/exp_leverage.py                 # §5: the leverage effect, both forms
uv run python agent/exp_vov.py                      # §4: the AR(1) vol-of-vol factor
uv run python agent/eda_gaps.py                     # §5: per-term dispersion, CRYPTO_1H
uv run python agent/verify_floor.py                 # §7: the volatility floor never binds
uv run python agent/exp_ramp.py                     # §5: the horizon width ramp
uv run python agent/exp_gamma_regime.py             # §5: calendar x regime interaction
uv run python agent/exp_recalibrate.py check        # §5: refit the com-equ multipliers
uv run python agent/vlong.py                        # §4: build the long-run level artifacts
uv run python agent/exp_vlong.py                    # §4: sweep the long-run blend weight
uv run python agent/eda_profile.py                  # §4: dispersion by time-of-week decile
uv run python agent/exp_gamma_split.py              # §4: the two-sided profile exponent
uv run python agent/nufit.py                        # §5: per-asset tail indices
uv run python agent/exp_nu.py                       # §5: sweep the tail-index blend
uv run agent/verify_interface.py                    # the contract, with sockets disabled
```

`agent/harness.py` + `agent/evaluate.py` + `agent/sweep.py` are the fast iteration loop — 545
prompts × 1000 paths × 5 assets in 16 s, against ~20 s per asset-day for the official path. It
computes CRPS with the very same `calculate_total_score_for_miner` the validator calls, on
realized paths assembled the way `backtest()` assembles them, and it is cross-checked against
`predict.py` in `journal.md` (2740.9 vs 2743.4 on an identical prediction file). Every number
quoted as a *result* still goes through `predict.py`.
