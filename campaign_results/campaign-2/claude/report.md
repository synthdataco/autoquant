# Beating the Synth field is a calibration problem, not a modelling problem

*A research report for the Synth SN50 miner community. Data cutoff 2026-07-27; everything after it
was treated as nonexistent for the whole campaign.*

---

## TL;DR

1. **The metric almost everyone reports is the wrong one.** The per-prompt *minimum* CRPS over 256
   miners is an order statistic that sits 14–17% below what the best individual miner achieves on
   crypto-1h. Optimising against it told me I was 2.5% off the pace when I was actually sitting at
   the field **median** — which the reward softmax pays approximately nothing for.
2. **The field is a near-degenerate cluster.** On crypto-1h BTC the best miner averages 345.99 and
   the 25th-best averages 346.60 — 0.18% apart, with the median at 348.97. Most of the field is
   running near-identical models.
3. **Because the cluster is so tight, the payoff curve is violently nonlinear.** ~15 Score points
   per 0.25% of CRPS; anything worse than ~0.5% off first place rounds to zero.
4. **What actually moved the needle was the *level* of the volatility forecast**, not the shape of
   the return distribution. A per-asset dispersion recalibration was worth several times more than
   fat tails, stochastic volatility and Hurst tuning combined — and it points in *opposite
   directions* for crypto (×0.92–0.95) and equities (×1.06–1.25).
5. **The second real win was the *horizon profile* of dispersion, which no scale parameter can
   reach.** The CRPS-optimal multiplier is monotone in horizon (1.10 at 1 min → 0.90 at the 60-min
   cumulative windows), so a single scale loses at both ends. Fixing it requires negative serial
   correlation in the increments, not a bigger or smaller sigma.
6. **Perfect volatility foresight is worth only 0.34% of CRPS.** An oracle experiment retired the
   entire volatility-forecasting programme in fifteen minutes and proves the leaders are not winning
   there. If you read one section, read §5.1.
7. **Three structural ideas that sound obviously right are worthless**: zero-inflation on
   tick-discrete equity feeds (refuted, with a quantitative explanation of why my own estimate was
   10× too optimistic), variance-neutral stochastic volatility, and Hurst/horizon-scaling tuning.
8. **One free win nobody should leave on the table**: stratified innovation sampling removes the
   O(1/M) finite-ensemble bias in `crps_ensemble` at zero cost.
9. **Deseasonalising fixes the *scale* of a seasonal effect and is silent about its *shape*.** On the
   tokenised equity perps, residual excess kurtosis is 8.6 in US market hours and 174 at the weekend
   *after* the hour-of-week amplitude has flattened the variance. Drawing every step from one pooled
   set of innovations — dominated 4:1 by closed-market minutes — was worth 0.33–0.47% on com-equ.
   See §4ter; it is the single most transferable finding in this report.
10. **Run the PIT histogram first, not last.** I spent most of one competition's budget sweeping
   parameters and testing narrative hypotheses; a single rank histogram per horizon found the real
   defect immediately, and explained why five earlier hypotheses had all failed. §5.5.
11. **A flat parameter sweep does not mean the parameter is right.** On com-equ a global `scale`
   sweep was flat because the assets need corrections in *opposite* directions and it was averaging
   them. Aggregate diagnostics hide sign-flipping structure. §5.5.
12. **Know when to stop, and prove it.** On com-equ-24h a *perfect per-prompt dispersion oracle*
   closes only 10–15% of my gap to the leader, and perfect volatility foresight only 0.57%. The
   leader there is at the per-prompt minimum on essentially every prompt — one qualitatively better
   forecaster, not a better-tuned member of the same family. §5.6 argues they are using information
   this model cannot have.

13. **Price your work in the units you are graded in, before you choose it.** The CRPS → Score
   exchange rate is steeply non-linear *and competition-specific*: 0.35% of CRPS is worth ≈10 Score
   points on crypto and ≈0.01 on com-equ. I improved com-equ CRPS by 1.1% — real, replicated,
   holdout-validated, and worth nothing. Computing that table is a minutes-long replay of the
   validator's own reward code. §7.4 is the change I would most want to make to how I ran this.
14. **Read the scoring code for its geometry, not just its formula.** I reasoned "CRPS is
   marginal-only" for twenty-five iterations. True for the 24h competitions; on crypto-1h, 72.3% of
   CRPS is scored on *sums* of steps, where cross-step covariance is priced. §5.7.

Final position, measured by the **official** `synth_lib.backtester` at 1000 paths: crypto-1h
**rank 174/257, Score term 0.2736**. The honest bottom line is that a 5× reduction in CRPS gap still
lands mid-pack, and §7 explains exactly why that earns almost nothing. Two of my own intermediate
numbers were too flattering and are corrected in §7.1 — that correction is itself one of the more
useful findings here.

---

## 1. What the validator actually scores

Everything below follows from reading `synth/validator/` rather than guessing, so it is worth
stating precisely.

**CRPS, summed over intervals.** For each scoring interval, prices are sampled every
`interval_steps`, converted to relative changes (×10 000), and `properscoring.crps_ensemble` is
evaluated at each point and summed. `_abs` intervals score absolute prices divided by the final
realized price; `_gaps` intervals keep only the first two interval points, making them *cumulative*
windows from the start.

**Only relative CRPS matters.** `compute_prompt_scores` subtracts the per-prompt field minimum and
fills missed responses (crps = −1) with the field p95.

**The softmax pays the top and nothing else.** Weights are `exp(beta × score)` with beta = −0.15
(24h) / −0.3 (1h), times `SMOOTHED_SCORE_COEFFICIENT = 1/3`.

Three consequences drove every decision in this campaign:

- **The 24h competitions are short-horizon problems.** Decomposing our 24h CRPS by interval:
  5min **60.8%**, 30min 25.0%, 3hour 10.6%, 24hour_abs 3.6%. I had pre-registered the opposite
  hypothesis (that 24h drift/terminal uncertainty dominates). It was refuted immediately, and that
  single number redirected the entire campaign.
- **crypto-1h has a different geometry**: 1min 26.1%, **Gaps 28.5%**, 2min 18.8%, 5min 11.9%,
  15min 6.9%, 30min 4.7%, 60min_abs 3.2%. The 12 `0_Xmin_gaps` intervals are the cumulative-return
  marginals at h = 5, 10, …, 60 min, so how variance grows with horizon is priced directly.
- **Only marginals are scored — with one exception that matters.** At step *t*, CRPS sees the
  ensemble's marginal distribution at *t* and nothing else. Cross-step path structure —
  autocorrelation, realistic-looking trajectories, anything that makes a path *look* like a price
  path — is invisible. This is liberating: you are fitting one-dimensional densities, not a process.

  The exception is that *which* marginals get scored is set by the interval geometry, and the
  coarser intervals score **sums of steps**. The marginal of a sum is not determined by the marginals
  of its parts: Var(sum over *X* steps) = *X*·Var(step) + 2·Σ(*X*−k)·Cov(step*ₜ*, step*ₜ₊ₖ*). So
  cross-step covariance *is* priced — not as path realism, but through its effect on the summed
  marginals. On crypto-1h this is not a footnote: only 27.7% of CRPS comes from the 1-min step
  itself, and the remaining **72.3%** is scored on sums. See §5.7, where I chased that channel and
  it turned out to be already correctly set.

---

## 2. Infrastructure: measure on the field's own prompts

`SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` was not preset and no bundle shipped, so I built bundles for
all three competitions with `build_offline_bundle.py --eval-end 2026-07-26` (the +25 h rewards
padding then still lands inside the 2026-07-27 cutoff — no lookahead).

Two things about `agent/research/harness.py` matter for anyone reproducing this:

- **Score on the field's exact prompt `start_time`s.** Prompt spacing is irregular (median 64 min,
  mode 72, range 8–240), so a fixed cadence matches only ~2/3 of prompts and the comparison
  silently stops being like-for-like.
- **Use the validator's own `calculate_crps_for_miner`.** Never reimplement scoring. My realized-path
  slicing mirrors `_slice_real_prices`, NaNs included, so gap handling matches too.

---

## 3. The mistake that nearly cost the campaign

For several iterations my headline metric was mean CRPS versus `field_min` — the average of
per-prompt minima across 256 miners. I got that from +7.9% to +2.5% on crypto-24h and thought I was
closing in.

Then I built the per-miner panel (`agent/research/field.py`) and computed each *miner's* mean CRPS
on exactly the prompts we both answered:

| crypto-1h, 2026-07-24→26 | per-prompt min | best miner | 25th | median |
|---|---|---|---|---|
| BTC | 297.80 | 345.99 | 346.60 | 348.97 |
| ETH | 400.11 | 457.83 | 458.99 | 460.21 |
| SOL | 421.56 | 485.51 | 486.36 | 487.71 |

The per-prompt minimum is **14–17% below the best miner**. No miner achieves it; it is the minimum
of 256 draws, and it gets *lower* as the field grows. Meanwhile the top of the field is packed
inside 0.2%.

So "+2.5% vs field min" on crypto-24h meant "≈ the field median" ≈ rank 120/214 ≈ Score ≈ 0.

**Two lessons.** Never optimise against an order statistic of a large field. And: the fact that
~200 miners score within 1% of each other means mid-pack is worthless *but the distance to first
place is small* — a genuine 1% edge is decisive.

One subtlety worth flagging for anyone building the same panel: require full coverage only over the
prompts being compared, not over the whole score history. Demanding lifetime-complete coverage drops
~80% of the field and leaves a survivorship-biased panel of the always-online (it made my panel 42
miners instead of 238, and flattered my rank).

---

## 4. What worked: the volatility *level*

### 4.1 The diagnostic

`agent/research/implied.py` rescales each prompt's simulated log-increments by a constant *k* on the
*same shocks* — a pure dispersion change, holding shape and seed fixed — and finds the CRPS-optimal
*k*. That cleanly separates "wrong scale" from "wrong shape".

The answer differed by asset class, which is why no global multiplier could ever have worked:

| competition | CRPS-optimal *k* | penalty at *k* = 1 |
|---|---|---|
| crypto-1h | **0.80** | 1.30% |
| crypto-24h | **0.85** | 0.80% |
| com-equ-24h | **1.15** | 0.33% |

### 4.2 Ruling out the obvious objection

A scale fitted on the two days before the evaluation window is indistinguishable from overfitting,
so `agent/research/biaswin.py` checks stability. Median realized/forecast sigma per half-month,
crypto, 1-min horizon:

```
2026-03-b 0.874   2026-04-a 0.805   2026-04-b 0.795   2026-05-a 0.808
2026-05-b 0.814   2026-06-a 0.963   2026-06-b 0.902   2026-07-a 0.825   2026-07-b 0.806
```

Never above 0.96 in 4.5 months. The bias is systematic, not a property of a quiet July.

The equity side is the reverse (median ≈ 1.00, optimum ≈ 1.15). My explanation: their feeds are
zero-inflated and tick-discrete (P(|5-min change| < 1 bp) = 0.339 on SP500, 0.253 on AAPLX, against
0.118 on BTC), which drags the realized-vol estimate below the dispersion the *continuous* part of
the distribution actually needs. Note this is the same underlying fact that Section 5 shows is not
worth modelling *directly* — it is worth absorbing into the scale.

### 4.3 The fix, fitted honestly

`_kcal(asset, increment)` is fitted by `fit_kcal.py` on a **training window (2026-07-08→20)
disjoint from every reporting window**, and shrunk 0.7× toward 1.0 to avoid chasing noise. It is
per-*increment* as well as per-asset, because the 1-min and 5-min marginals are genuinely different
objects.

That discipline paid for itself as evidence: fitted on the training window, crypto-24h *k* came out
at 0.93–1.02, much closer to 1 than the 0.85 the reporting window suggested. Fitting on the
reporting window would have overfit by ~10%.

SP500 needed special handling — it replaced the retired SPYX and only starts being prompted on
2026-07-23, so *all* of its field prompts are inside the reporting window. Since CRPS needs only our
paths and the realized path (both from the snapshot), `fit_kcal_grid.py` fits it on a synthetic
prompt grid over its own earlier history instead.

### 4.4 Trust the HAR forecast more

`shrink` (log-space weight on the HAR forecast versus long-run vol) swept to **1.0** — no extra
shrinkage — worth 0.46% over 0.70 on crypto-1h. The HAR blend already shrinks internally through
its 7-day term, which carries the largest coefficient; shrinking again just adds bias. (After
changing it, `kcal` must be refitted, since it absorbs the level.)

---

## 4bis. The horizon profile: what a scale parameter cannot fix

After `kcal` the dispersion *level* is right and (per §5.1) the vol *forecast* is as good as it can
matter. The third possibility went untested for most of the campaign: the scale may be wrong as a
function of **horizon within a single prompt**. crypto-1h is scored at seven horizons at once, plus
12 cumulative `*_gaps` windows, so one multiplier can be simultaneously too small at one end and too
large at the other while the total merely looks "slightly off".

`perinterval.py` rescales the served paths by *k* on the same shocks and optimises *k* **per scoring
interval**. The result is strongly monotone and consistent across all five crypto assets:

| interval | 1min | 2min | 5min | 15min | 30min | 60min_abs | `*_gaps` |
|---|---|---|---|---|---|---|---|
| optimal *k* | **1.10** | 1.05 | 0.95 | 0.92 | 0.91 | 0.91 | **~0.90** |
| share of CRPS | 27.6% | 19.3% | 11.6% | 6.4% | 4.6% | 3.3% | 29.5% |

We are *under*-dispersed at 1 minute and *over*-dispersed at every cumulative horizon. Weighted, this
is worth ~0.4% — about a third of the gap to the best miner.

**`hurst` cannot fix it.** On crypto-1h `step_min = 1`, so `step_min ** hurst = 1` identically. The
horizon profile is not produced by the per-step scale at all; it is produced by *cumulating* the
increments. The defect therefore lives in the increment **correlation**, and that is the only place
it can be fixed.

**The fix.** An MA(1) filter `e_t − θ·e_{t−1}`, renormalized by `sqrt(1+θ²)` so the one-step variance
stays exactly where `kcal` calibrated it. It raises the 1-step marginal relative to the h-step
cumulative — the measured pattern — and it corresponds to something real: the first-order return
autocorrelation is −0.001 to −0.011 (bid-ask bounce). Verified numerically on BTC: one-step variance
unchanged across θ ∈ {0, 0.1, 0.2}, while 60-min cumulative variance falls 26% at θ = 0.20.

One implementation subtlety that matters more than it looks: the lag term needs an extra innovation
drawn *before* step 0. Otherwise the first increment escapes the filter — and on crypto-1h that
single step is 27.6% of the score.

Swept on training windows disjoint from all reporting windows, the two crypto competitions chose
θ = 0.10 (1-min increments) and θ = 0.20 (5-min), an increment dependence that makes sense because a
5-minute step aggregates five minutes of bounce. Confirmed out of sample on crypto-24h:
**+0.92% → +0.725%**, sign consistent across all five assets.

**Equities get θ = 0.** The 5-min sweep initially suggested 0.20 for them too, but it did not
replicate: neutral on the reporting window (+4.792% → +4.800%), and ~0.1% with per-asset sign flips
over three further windows. That is inside the noise floor, so I do not claim it. It is also
mechanistically consistent — those feeds are stale (P(5-min change = 0) = 0.105 on AAPLX vs 0.002 on
BTC) and staleness induces *positive* serial correlation, working against the negative bounce.

**Transferable lesson.** When your loss is a sum over several horizons, decompose the diagnosis by
horizon. And note that my earlier "microstructure is a dead end" verdict (§5.3, implied Hurst
0.486–0.498) was measured on the *aggregate* variance ratio — precisely the statistic that averages
this effect away. The aggregate said random walk; the per-interval decomposition found 0.4%.

## 4ter. Seasonality: you have probably fixed the scale and left the shape

This is the finding I would most want another miner to take away, because the defect is invisible to
the diagnostic that people naturally run, and the fix is nearly free.

The tokenised equity and commodity perps (AAPLX, NVDAX, TSLAX, GOOGLX, SPCX, XAU, WTIOIL) trade
24/7, but their **underlying cash market does not**. My model already handled the obvious
consequence: a 168-bucket hour-of-week amplitude narrows the fan overnight and at weekends. Having
built that, I considered seasonality "done" — and the standard check agrees with me. Divide the
returns by the fitted amplitude and the residual standard deviation is flat across sessions (0.86
open, 1.05 weekend). That flatness is exactly what makes you stop looking.

It is silent about every higher moment. Measuring the *deseasonalised* residuals per US cash session,
averaged over the five equity-like assets:

| session | resid sd | excess kurtosis | P(exact zero) |
|---|---|---|---|
| open (13:30–20:00 UTC, Mon–Fri) | 0.86 | 8.6 | 2.6% |
| extended | 1.05 | 85.9 | 10.9% |
| closed | 0.92 | 5.8 | 10.9% |
| weekend | 1.05 | 173.9 | 33.6% |

A weekend minute is a 33.6%-zero, kurtosis-174 object. An open-market minute is a 2.6%-zero,
kurtosis-8.6 object. They are not the same random variable, and no scalar amplitude can make them so.

Filtered historical simulation bootstraps innovations from one pooled set of these, and the pool is
dominated **4:1 by closed sessions** (8130 closed/extended/weekend minutes per week against 1950 open
ones). So every open-market step drew an innovation far too spiky and far too often exactly zero, and
every weekend step drew one too tame. **CRPS is marginal-only** — the score at step *t* sees only the
ensemble's marginal at *t* — so each step's shape is charged directly, with no cross-step structure
for the error to hide behind.

An independent diagnostic pointed the same way before I looked at the pools: the com-equ deficit was
a *uniform* shift (median per-prompt CRPS ratio ≈ mean; the worst 10% of prompts carried only 22–24%
of the excess) and was **worse on Mon–Thu starts** (NVDAX 1.158, XAU 1.103) than on Fri–Sun starts
(1.034, 1.019) — Mon–Thu horizons being the ones densest in open-market minutes.

**The fix** is a pure shape substitution: build a separate innovation pool per (asset, increment,
session), each standardised to unit variance, and draw each simulated step from the pool of its own
session. Sigma still comes entirely from the forecast × amplitude path, so the dispersion calibration
is untouched and remains valid — verified numerically (step variance 2.83e-06 either way; step-0
kurtosis 44 → 19). Sessions with too few observations fall back to the global pool. Crypto is
excluded deliberately: BTC shows no comparable structure, as a 24/7 asset should not.

```
com-equ-24h vs field min      pooled     session-conditional
training  07-16..07-21        +6.112%          +5.644%   (7/7 assets improved)
holdout   07-21..07-26        +5.323%          +4.996%   (5/7 improved)
```

**Generalisation.** Whenever you divide out a seasonal factor, check the *shape* of what remains per
bucket, not just its variance. And treat blanket statements about session structure with care: "these
perps trade 24/7, so there is no session-of-day effect" is true of crypto and false of a token that
tracks NVDA.

### 4ter-b. The seasonal-curve bug that cancelled itself

Worth recording as a null, because it closes off an obvious-looking direction.

Auditing the fitted curves, I found the amplitude was clipped to `[0.25, 3.0]` — a guardrail sized
for crypto (BTC's amplitude spans only 0.51–1.86, so it never binds) and silently applied to
everything. On **XAU the floor binds on 33 of 168 buckets**, a fifth of the week, where the true
amplitude reaches **0.152**. Gold's weekend closure is more complete than the clip permits, so for
those 33 hours the fan was ~64% too wide on every path.

Refitting with a wide clip changed the result by **nothing** (+5.305% → +5.306%; XAU itself 6.80 →
6.77). The reason is structural: the curve is applied **twice, in opposite directions** — once
dividing the context returns to estimate `sigma_min`, once multiplying forward to shape the path.
Understating the weekend amplitude inflates the deseasonalised context vol by nearly the same factor
by which it then deflates the forward amplitude. On an XAU Saturday start, `sigma_min` went
1.808 → 1.888 bp while the mean forward amplitude went 0.257 → 0.228: a **7% net effect from a 39%
curve error**.

A multiplicative seasonal factor is self-normalising to first order whenever estimation and
simulation share it. So refining the seasonal *curve* is largely wasted effort — which is precisely
why the win above had to come from the residual **shape**, where nothing divides it back out.

---

## 5. What didn't work, and why (the useful part)

### 5.1 Volatility forecasting is exhausted — the oracle bound

This is the single most valuable experiment in the campaign, and it should have been iteration 2.

I had a confident argument that forecast error was the whole remaining gap: `implied.py` gives CRPS
penalty ≈ 30·(Δlogσ)², `harfit.py` measures out-of-sample sd(Δlogσ) ≈ 0.28 for every asset, so the
channel should be worth 30 × 0.28² ≈ **2.3%** — larger than my entire gap to first place.

`oracle.py` tests it directly by replacing the forecast with the **realized** RMS volatility over the
very horizon being predicted — deliberate lookahead, used only as an unattainable ceiling and never
served — holding shocks, seasonality, pool and seed fixed:

| blend toward oracle | w=0 | 0.25 | 0.5 | 0.75 | **1.0 (perfect foresight)** |
|---|---|---|---|---|---|
| crypto-24h CRPS | — | −0.19% | −0.31% | −0.36% | **−0.34%** |

**Perfect volatility foresight is worth 0.34%**, against a gap of 0.92–1.14%. My estimate was ~7×
too high.

**Why the arithmetic failed.** The quadratic formula prices a *systematic* mis-scaling. It badly
over-prices *unbiased noise*, because CRPS is minimized by the predictive distribution that already
integrates over sigma uncertainty — errors that cancel in direction cost far less than (Δlogσ)²
implies. Bias and variance in a volatility forecast are not interchangeable, and only bias (which
`kcal` removes) is expensive.

This also kills a related idea I had queued and measured. `termstruct.py` confirms conditional vol
genuinely mean-reverts over a 24 h horizon — the regression slope of log(realized/forecast) on the
state decays 0 → −0.15 by 16–24 h — but with sd(state) = 0.18 the far-horizon mis-scaling is only
~2.6%, and the whole channel is capped at 0.34% anyway. Correct diagnosis, negligible prize.

**Get an oracle bound before optimising a channel.** One cheap experiment retired three queued ideas
and proved the leaders cannot be winning on volatility prediction.

### 5.1b Zero-inflation on tick-discrete feeds — REFUTED

Pre-registered in iteration 2 and aimed at my two worst assets. The structure is real and large:

| asset | P(5-min change = 0) | tick (bp) | sd&#124;nonzero (bp) |
|---|---|---|---|
| AAPLX | 0.105 | 0.33 | 9.66 |
| XAU | 0.115 | 0.23 | 7.77 |
| BTC | 0.002 | 0.001 | 12.41 |

And the atom is strongly **predictable** — conditioning on the number of zero returns in the
previous 12 bars moves AAPLX's P(zero) from 0.043 to 0.514. Tick rounding alone cannot explain the
level (0.33 bp against a 9.7 bp sd implies ≈ 0.03, not 0.105), so it is genuine feed staleness.

I implemented a state-dependent mixture (mass *q* at exactly zero, survivors re-inflated by
1/√(1−q) to hold variance constant). It made things **monotonically worse**: on com-equ-24h training
windows, `zi_mix` 0 → +3.53%, 0.5 → +3.71%, 1.0 → +3.76%. That held after I found and fixed a real
bug in the first attempt (the trailing rate was measured on 1-minute returns while predicting
5-minute increments — 0.43 versus 0.13 on AAPLX, inflating *q* about 3×).

**Why.** A controlled synthetic experiment with the atom known *exactly* (q = 0.13, sd 9.66 bp)
gives a best-possible gain of **0.08%** — my back-of-envelope had said ~1%. The error: CRPS is an
integral of squared CDF error over *all* thresholds, so an atom helps only in a neighbourhood of
zero and actively hurts whenever the outcome is not zero. A tight continuous density is already
nearly as good at x = 0. The true prize is an order of magnitude smaller than the estimation error
in *q*, so the mixture is net-negative.

**This is the most transferable negative result here**: a prominent, real, predictable feature of
the data that a proper-scoring-rule objective does not reward you for modelling explicitly.

### 5.2 Stochastic volatility — no effect

Variance-neutral log-vol randomisation (per-path level offset + log-vol random walk). Motivated by a
correct diagnosis: CRPS marginals integrate over the *unknown future vol path*, so dispersion should
grow with horizon. Swept `sv_vol` and `sv_level` repeatedly, before and after the kcal fix:
differences of 0.01–0.07%, i.e. noise.

**Why, in hindsight.** CRPS is dominated by getting the *scale* of each marginal right, and `kcal`
already sets that optimally. Vol-of-vol reshapes the tails at fixed variance, and the tails were
never where the loss was.

An earlier related regression is worth recording: switching to properly filtered (FHS) residuals
*hurt* (+2.98% versus +2.50%). Correctly-filtered residuals are thin-tailed by construction, so a
static sigma cannot generate the horizon-growing dispersion the marginal needs — the globally
standardized pool had been smuggling it in. Diagnosing that is what motivated the stochastic-vol
work; that the explicit version then only matched the accidental one says the accidental version was
already about the right size.

### 5.3 Hurst / horizon scaling — nothing to win

Empirical variance ratios give implied Hurst 0.486–0.498 across all five crypto assets at every
horizon from 2 to 60 minutes — indistinguishable from a random walk. Sweeping `hurst` over
0.47–0.515 moved the score by 0.16% with no clear optimum.

**Partially overturned later — read with §4bis.** I concluded from this that the `_gaps` family (28.5%
of the crypto-1h score) held no free lunch. That was wrong, and instructively so. The variance ratio
is an *aggregate* statistic; it averages over exactly the horizon-dependent mis-scaling that the
per-interval decomposition later found to be worth 0.4%. The data really is √t in aggregate *and* my
dispersion was simultaneously 10% too small at 1 minute and 10% too large at 60. A summary statistic
that is right on average will hide a structural error that cancels in its own average.

### 5.4 Tails were already right

Standardized residuals (realized step return ÷ model step sigma) match the innovation pool's shape
closely once you divide out the scale — the quantile *ratios* line up. The apparent kurtosis gap
(pool 17–22 versus realized 4–7) is tail-estimation noise at these sample sizes, not
misspecification. Gaussian innovations, by contrast, lose 2.0–5.7%: fat tails matter, but any
reasonable fat-tailed law will do.

---

### 5.5 Sweeping parameters is the wrong first move — run the PIT

On com-equ-24h, NVDAX sat 11% behind the best miner and would not move. I worked through hypotheses
in the order they occurred to me, and every one was refuted at real cost:

| hypothesis | verdict |
|---|---|
| the field is simply tighter on some assets | refuted: corr(field spread, our gap) = −0.07 |
| a dispersion-level error | refuted: `scale` 0.85→1.08 moves the mean 0.8% against an 11.9% gap |
| a few disaster prompts | refuted: median prompt ratio ≈ mean; worst 10% carry 22–24% |
| the seasonal amplitude is mis-sized | refuted: `season_pow` optimum exactly at the incumbent 1.0 |
| a drift / anchor error | refuted, **with the sign inverted** (below) |
| better volatility forecasting | refuted: oracle bound, perfect foresight worth 0.57% |

Then I ran a probability integral transform — for each scored point, the rank of the realised value
inside the ensemble, computed **per horizon** on the same non-overlapping interval changes the
validator scores. It found the defect immediately:

```
NVDAX   30min   12.0 / 23.7 / 26.8 / 22.0 / 13.4   central hump  -> TOO WIDE
NVDAX  180min   12.0 / 22.3 / 37.7 / 17.6 / 10.4   badly humped  -> TOO WIDE
GOOGLX   24h    37.5 / 18.8 / 15.6 / 14.1 / 14.1   12.5% outside -> TOO NARROW
TSLAX    24h    51.5 / 15.2 /  4.5 / 13.6 / 15.2   15.2% outside -> TOO NARROW
```

**The assets are mis-dispersed in opposite directions.** That is why the `scale` sweep looked flat:
it was averaging two real errors of opposite sign and landing near 1.0. I had read that flatness as
evidence that dispersion was already correct. It was evidence of nothing.

> **A flat aggregate sweep does not mean the parameter is right.** A global knob that averages over
> sign-flipping structure will always report that you are at the optimum. The tell is an aggregate
> that is flat while disaggregated diagnostics are not; the fix is to disaggregate along the
> dimension the knob is constant over. This is the same error I made in §4bis, where an aggregate
> variance ratio said "random walk" and the per-interval decomposition found 0.4%.

Fitting per-asset corrections on one window and **requiring replication on a disjoint holdout** —
the test my earlier recalibration attempt had failed — only two of seven survived (NVDAX ×0.94, XAU
×0.88; GOOGLX, WTIOIL and SPCX gained on the fit window and lost on the holdout, i.e. pure
window-fitting). Worth +5.310% → +5.210%, with no asset made worse.

The correction is small. The *diagnosis* is what mattered: it explained NVDAX and it explained why
five earlier hypotheses had failed. **Run the PIT first.** It asks the ensemble directly where its
probability is misplaced, instead of asking you to guess which story to test next.

For contrast, here is what a healthy PIT looks like — crypto-1h, every asset, every horizon:

```
BTC    5min  21.5 / 18.8 / 20.4 / 18.5 / 20.8    ok
BTC   75min  18.7 / 18.8 / 23.0 / 21.2 / 18.2    ok
HYPE 300min  20.9 / 21.3 / 18.0 / 23.6 / 16.3    ok
```

All five crypto assets are flat at all five horizons. That is independent confirmation that the
crypto model is sound and that its remaining +0.6% is near this family's achievable limit.

**A refuted hypothesis worth reporting: drift.** A uniform excess at a near-optimal width suggests a
*centre* error, so I measured realised drift from the anchor. The result was decisive in the wrong
direction: TSLAX had the largest implied drift cost (33% of CRPS) and the *smallest* gap (+0.6%),
while NVDAX had essentially none (0.08%) and the *largest* gap (+11.3%). Two guards fired, and both
were necessary. The economic one: a diagnostic that explains your good asset better than your bad one
explains nothing. The statistical one: 241 hourly-spaced 24-hour windows over 10 days contain about
**10 independent observations**, so a t-statistic of −9 is an overlapping-window artifact measuring
one realised July drawdown. Always convert overlapping windows to an effective sample size before
believing a t-stat. Fitting a drift term there would have been rewarded in backtest and would have
been the most classic overfit available in this problem.

### 5.6 When to stop: proving a gap is unreachable

Every channel I could name on com-equ was priced, and they summed to about 1.1% against an 11% gap.
Rather than invent an eighth hypothesis, I bounded what was left.

`ceiling.py` gives every prompt a **perfect dispersion oracle** — the CRPS-minimising scale chosen
separately for each prompt by direct search. This is deliberate lookahead used strictly as an
unattainable bound: never served, never committed to the model. Whatever it fails to recover is
provably out of reach for *any* scale or volatility modelling in this family.

| asset | ours | perfect-dispersion oracle | best miner | our gap | gap at oracle |
|---|---|---|---|---|---|
| NVDAX | 2358.5 | 2343.6 | 2212.9 | +6.6% | +5.9% |
| GOOGLX | 2935.2 | 2888.2 | 2829.1 | +3.7% | +2.1% |
| AAPLX | 1694.8 | 1677.2 | 1596.6 | +6.2% | +5.0% |
| XAU | 1159.3 | 1154.1 | 1088.8 | +6.5% | +6.0% |

A *perfect* per-prompt dispersion oracle closes **10–15%** of the gap. Combined with the com-equ vol
oracle (0.57%), the entire scale-and-volatility axis — the axis this model is built on — is worth
about a tenth of what separates me from the leader.

A second number reframes the competition. On com-equ the best single miner's mean CRPS is essentially
equal to the **per-prompt minimum over all 250 miners** (NVDAX 2212.9 vs 2247.5, ratio 0.985; GOOGLX
1.009; AAPLX 1.004; XAU 0.988). On crypto that ratio is 1.14–1.17, because there the per-prompt min
is unreachable noise assembled from many different lucky miners (§7.3). Here **one miner is at or
near the minimum on essentially every prompt**. That is not a field of similar models with different
tunings — that is one qualitatively better forecaster.

So I stopped, and I think stopping is the correct call rather than a concession. These are *tokenised
equities and commodities*. A miner who can see the underlying cash or futures market, an options
surface, or an earnings and macro calendar has a genuinely different conditional distribution — a
model that knows NVDA reports tonight is not doing better statistics, it is using information that a
HAR fitted to the token's own price history cannot contain. My snapshot holds only the perp's price
history, so within this campaign's rules that gap is unreachable **by construction**.

> **Price the channel before you build in it.** I used oracle bounds three times — vol foresight on
> crypto (0.34%), vol foresight on com-equ (0.57%), per-prompt dispersion on com-equ (10–15% of the
> gap) — and each converted an open-ended "maybe I can model this better" into a priced, closed
> channel for a fraction of the cost of trying. The failure mode this prevents is the expensive one:
> iterating inside a model family that provably cannot reach the target, reading each 0.1% as
> progress toward an 11% goal. Cheap lookahead bounds are legitimate instruments even under a strict
> no-lookahead rule, provided they only inform *where to spend* and never touch what gets served.

### 5.7 The 27% of my biggest competition I had never diagnosed

For twenty-five iterations I reasoned from "CRPS is marginal-only, so fit each step's density and
ignore path structure." That is exactly right for the two 24h competitions. On crypto-1h — my
*largest* score term — it is incomplete, and I did not notice until I re-read
`calculate_price_changes_over_intervals` line by line rather than from memory.

crypto-1h has **18** scoring intervals. Six are ordinary (1, 2, 5, 15, 30 min, 60min_abs). The other
**twelve** are `0_Xmin_gaps` for X = 5…60, and the function truncates them to `interval_prices[:, :2]`
— each scores a *single anchored change* from the path's start to X minutes in, all twelve folded
into one "Gaps" total. Measured with the validator's own detail output (`ivdecomp.py`), CRPS splits:

| interval | 1min | Gaps | 2min | 5min | 15min | 30min | 60min_abs |
|---|---|---|---|---|---|---|---|
| share | 27.7% | **27.4%** | 19.3% | 12.0% | 6.6% | 4.2% | 2.9% |

Twelve nested, mutually correlated anchored intervals sharing one anchor: a calibration error there
is charged twelve times and diversified zero times. It is also the one place cross-step covariance is
priced (§1). The MA(1) filter is the exact lever — it is variance-normalised (÷√(1+θ²)), so θ moves
*only* the covariance terms while holding the 1-min marginal (another 27.7% of the score) precisely
where `kcal` calibrated it.

An anchored PIT (`gappit.py`) came back near-uniform at most horizons but **too narrow at 5–10 min**
on ETH, SOL and HYPE. From the code, `out = ext − θ·lag`, so positive θ *shrinks* anchored variance:
I pre-registered the prediction that lowering θ below 0.10 should improve CRPS, and that the gain
should land in Gaps rather than 1min.

**Refuted.** Fit 07-11→18 / holdout 07-18→26, versus θ = 0.10:

| θ | −0.10 | −0.05 | 0.00 | 0.05 | 0.15 |
|---|---|---|---|---|---|
| fit | +0.159% | +0.048% | −0.025% | −0.056% | +0.067% |
| holdout | +0.525% | +0.339% | +0.146% | +0.028% | −0.003% |

Monotone in the direction opposite to the prediction. θ = 0.10 stands unchanged.

Two things are worth keeping from this. First, the `RATIO` column in `gappit.py` (realized sd ÷ mean
ensemble sd) read >1 almost everywhere, which *looks* like under-dispersion and is confounded: pooled
realized sd exceeds mean per-prompt ensemble sd whenever true volatility varies across prompts,
calibrated or not. The PIT quintiles were near-uniform, I trusted them over the ratio, and the CRPS
sweep agreed with the quintiles. Second, and more important:

> **Read the scoring code for its geometry, not just its formula.** I knew the CRPS formula
> perfectly and still carried an incomplete model of *what gets scored* for most of the campaign.
> The defect was in my map of the objective, not in the model — the most expensive kind, because
> every downstream decision inherits it silently. Re-read the scorer periodically, especially the
> parts you believe you already understand.

---

## 6. The one free win: stratified innovations

`crps_ensemble` uses the ensemble's *empirical* CDF, which for finite *M* is rougher than the law it
came from, biasing CRPS upward by O(1/M). Measured: ~1.9% at M = 100, ~0.45% at M = 250.

Since only per-step marginals are scored, each step's *M* innovations can be drawn as a
**stratified** sample — one from each 1/M probability bin, independently permuted per step, via the
residual pool's empirical quantile function. Identical cost. The per-step permutation keeps
increments independent across time, so the sums the coarser intervals score are undisturbed in
distribution.

Paired A/B, identical seeds, 1000 paths, out-of-sample:

| | iid | stratified |
|---|---|---|
| crypto-1h vs best miner | +0.964% | **+0.795%** (rank 140→130/238) |
| crypto-24h vs best miner | +1.157% | **+1.110%** (rank 91→88/214) |

No overfitting risk: it reduces estimator variance rather than fitting anything. Part of the gap to
the leaders may simply be Monte-Carlo noise the leaders already removed.

**How much is left at the served size?** The per-step permutation is independent across steps, so
the *sums* the coarser intervals score inherit only a weaker, second-order reduction — and per §5.7
those sums are 72.3% of crypto-1h's CRPS. That is a real structural gap, so I measured it rather
than assumed it away. `mbias.py` compares CRPS at M = 1000 against an M = 12 000 reference on
identical prompts, per interval:

| interval | 1min | 2min | 5min | Gaps | 30min | 60min_abs | **OVERALL** |
|---|---|---|---|---|---|---|---|
| excess at M=1000 | +0.005% | +0.041% | +0.030% | +0.073% | +0.093% | +0.245% | **+0.040%** |

The structural prediction is confirmed exactly: the directly-stratified 1-min step carries +0.005%
while the summed intervals carry 10–50× more. And it does not matter — the total is **+0.040%**,
below the 0.05% threshold I pre-registered as "not worth engineering". A *perfect* sum-aware variance
reduction, recovering all of it, would be worth about 0.05 of a Score term (§7.4). M = 1000 is
effectively converged; the per-step stratification already banked the large win.

> **A confirmed mechanism is not a viable project.** Every structural particular of that hypothesis
> was right, and the prize was still a rounding error. Bounding the size *before* designing the
> machine is what turned a plausible multi-hour engineering effort into a twenty-minute measurement.

---

## 7. Results, and an honest assessment

### 7.1 First, two corrections — and why they matter to you

Late in the campaign I ran the **official** `synth_lib.backtester` end to end at 1000 paths, and it
disagreed with my own research harness by more than 100 rank places. Since Rule 3 makes the
backtester authoritative, the harness was what was wrong:

| crypto-1h | official (5 days, 07-21→26) | my harness (2 days) |
|---|---|---|
| rank | **183/257** | 78/238 |
| Score term | **0.1732** | — |
| BTC mean CRPS | 428.46 | ~348 |

Both were computed correctly; they measure different things. The official window is 5 days (including
harder days) and its full-coverage panel is stricter (≈145 miners, not 238). Rescoring the official
run's own served prediction files confirmed the model is identical and that our CRPS tracks the field
mean day by day — so this is a **window and panel** artifact, not a model regression.

The second correction: on the **full** 8-asset com-equ-24h set we are **+3.055%, rank 119/219**, not
the +1.44% I recorded mid-campaign from a narrower subset. "GOOGLX rank 3" was likewise a small-panel
artifact; on the full panel it is rank 151.

**Both errors flattered me, and that is not a coincidence.** A subset of assets is a subset you have
been staring at and implicitly tuning; a lenient coverage filter admits miners who answered only the
easy prompts. If you take one methodological point from this report: **re-anchor your fast harness to
the official evaluator, on the official window and the full asset set, before you believe any
absolute claim it makes.** I re-anchored once, late, and until then had a materially wrong picture of
where I stood.

### 7.2 Where the model actually ended up

Official evaluator (`synth_lib.backtester`), 1000 paths, 10 days ending 2026-07-26:

| competition | official result | Score term |
|---|---|---|
| crypto-1h | rank 174/257 | **0.2736** |
| crypto-24h | rank 173/257 | **0.0606** |
| com-equ-24h | rank 189/250 | **0.0000** |

**Total Score ≈ 100 × (0.2736 + 0.0606 + 0.0000) / 3 ≈ 11.1**, up from ≈7.7 before the final session.
All three were re-run end to end at 1000 paths against the final model: the last session changed three
shared code paths, and an official number measured against a superseded build is not a result.

Harness-relative progress (same yardstick throughout, so the *deltas* are meaningful even where the
absolute level was not):

| competition | start | final | vs best miner |
|---|---|---|---|
| crypto-1h | +5.1% | +0.60% | rank 107/201 |
| crypto-24h | +7.9% | **+0.63%** | rank 43/214 |
| com-equ-24h | +6.1% | +5.21% | rank 189/250 (official) |

**Read the com-equ zero carefully, because it is the most instructive number here.** This session's
work moved it from rank 209/250 to 189/250 and took SP500 to **rank 1/257** outright — and the Score
term stayed at exactly 0.0000. The reward softmax on that competition is not a gentle gradient: the
top five miners hold 0.0295 / 0.0294 / 0.0290 / 0.0281 / 0.0261 and the sixth holds 0.0049, a **5×
cliff**. Below that cliff, a 20-place rank improvement and a first-place asset are together worth
nothing at all. This is the payoff nonlinearity of TL;DR item 3 in its most extreme form, and it is
why §5.6's decision to stop was about *reachability*, not about effort.

The two crypto competitions are where the campaign's value actually landed: an 8× and 5× reduction in
CRPS gap, ending 0.6% behind the best miner in a field whose top 25 are separated by 0.18%.

### 7.3 Is "the best miner" even a real target?

Given that ~200 miners sit inside 1%, I checked whether rank 1 on a short window is skill or just the
luckiest of a pack of equals — structurally the same order-statistic trap as `field_min` in §3, one
level up. `persist.py` splits the prompts into alternating halves (same regime, isolating noise):

```
BTC rho=+0.913  ETH +0.886  SOL +0.939  XRP +0.383  HYPE +0.912   MEAN +0.807
half-A's winner is only +0.014% off half-B's winner, in half B
half-A's top-10 land in half B at median rank 11
```

Rank is overwhelmingly **skill**. The luck component is +0.014%, two orders of magnitude below our
gap, so the target is real and chaseable. Note the exception: XRP's rho = +0.38, because its field is
compressed into +0.04% median-to-best — there, ranking genuinely *is* a coin flip and optimising it is
worth nothing.

The instructive part is that the two order-statistic traps look identical and behave oppositely: the
per-prompt minimum over 256 miners is unreachable noise; the per-miner best over 256 miners is a
stable skill level. Averaging over ~720 prompts is the whole difference. Worth testing, not assuming
— I assumed wrong in §3 and would have assumed wrong again here.

**Now the part that matters.** `agent/research/payoff.py` replays the validator's softmax on the
real field to convert a CRPS edge into the pre-registered Score term:

| gap vs best miner | crypto-24h | crypto-1h | com-equ-24h |
|---|---|---|---|
| +5.21% (com-equ now) | — | — | **0.000** |
| +1.00% | 0.03 | 0.21 | 0.012 |
| +0.63% / +0.60% (crypto now) | **0.106** | **0.392** | — |
| +0.50% | 0.17 | 0.46 | 0.11 |
| +0.25% | 0.41 | 0.68 | 0.33 |
| 0.00% | 1.00 | 1.00 | 1.00 |

### 7.4 The exchange rate, and the campaign's most expensive mistake

Read that table as a *price list* rather than a projection and it reorders everything I did. The
mapping from CRPS to Score is steeply non-linear and **competition-specific**, and I did not compute
it for my own current position until very late:

- On crypto, **0.35% of CRPS is worth ≈10 Score points** (crypto-24h 0.106 → 0.412; crypto-1h
  0.392 → 0.678).
- On com-equ, closing 4.2 percentage points of gap — **more than four times everything I recovered
  there all campaign** — would move the term from 0.000 to 0.012.

I spent a large share of this campaign improving com-equ CRPS by ~1.1%, all of it real, replicated
and holdout-validated. In Score units it was worth approximately nothing, and the same effort spent
on crypto would have been worth multiples of my final score. §5.6 concluded the com-equ gap was
*unreachable*; §7.4 shows the decision was overdetermined, because even succeeding there paid nothing.

> **Optimise the objective, not its proxy — and price the proxy before choosing the work, not
> after.** CRPS is only an *input* to the Score; a steep softmax makes the conversion wildly
> non-uniform across competitions. Computing this table costs minutes and is a pure replay of the
> validator's own reward code on the cached field. Doing it in week one would have redirected the
> entire campaign. It is the single change I would most want to make to how I ran this.

A useful corollary: this reframes what counts as a good result. The three structural changes of the
final session were built and validated on com-equ, where they bought 0.000 — and they also improved
**crypto-1h from Score term 0.1732 to 0.2736** (rank 183 → 174/257), because they touched shared code
paths. The campaign's largest single Score gain came as a side effect of work aimed elsewhere, which
is a good argument for measuring every change on every competition even when you built it for one.

So a campaign that cut its CRPS gap by 5–7× still lands where the softmax pays a fraction of the
leader's weight — the measured Score term on crypto-1h is **0.1732**. **The honest summary is that
this model is now respectable and still not competitive for emissions.** I would rather say that
plainly than dress up rank 183/257 as success.

Why the gap persists is now a well-posed question with a partial answer. It is *not* volatility
forecasting: §5.1 shows perfect foresight is worth 0.34%, and rank does not track forecast R² at all
(SP500 has the worst R² at 0.30 and ranks **1st**; AAPLX has 0.45 and ranks 110th). It is not tails,
Hurst, stochastic vol, or zero-inflation — all measured, all null. Roughly 0.4% of it *was* the
horizon profile of dispersion (§4bis), which I found late and fixed only partially, since MA(1) is a
one-parameter approximation to a monotone curve. The remainder I did not identify, and the leaders'
edge over a well-calibrated marginal model is smaller than 1% of CRPS — which is exactly why it is
hard to find and exactly why it is worth so much.

---

## 8. Limitations

- **Windows are short.** Two-day reporting windows (48 prompts/asset at 24h, 288 at 1h). Differences
  below ~0.2% are not resolvable, which is precisely the scale that decides the Score. Several
  "no effect" verdicts above are really "no effect larger than my noise floor". The MA(1) equity
  result is the clearest example: a −0.29% training-window gain that turned out to be noise.
- **My harness's absolute levels were wrong until late** (§7.1). The *deltas* in this report come from
  paired A/B runs on identical seeds and I trust them; the absolute ranks quoted before §7 came from
  a 2-day window and a lenient panel and should be read as relative progress only.
- **MA(1) is a one-parameter fit to a monotone curve.** The per-interval optima run smoothly from 1.10
  to 0.90; a single θ cannot match all of them, so some of that 0.4% is still on the table.
- **One regime.** All fitting and reporting happens inside 2026-06-23 → 2026-07-26, after the
  competition split. I deliberately did not backtest across the split, the 2026-03-11 crypto-1h
  formula change, or the ~2026-07-17 SPYX retirement. If volatility mean-reverts upward during
  evaluation, a `kcal` below 1 on crypto becomes a liability — CRPS punishes under-dispersion much
  harder than over-dispersion.
- **`kcal` is fitted per asset on ~96 prompts** and shrunk 0.7× for that reason. XRP's raw fit
  (0.85–0.91) is the one I trust least.
- **SP500 is fitted on a synthetic grid**, not real prompts, because it has no prompt history before
  the reporting window. It carries the largest coefficient in com-equ-24h (3.4379), so this is the
  single largest source of uncertainty in that competition.
- **The field panel is the current field.** Ranks assume the same miners behave the same way in the
  evaluation window.

---

## 9. What I would do with more budget

**Read this list against §7.4 first.** The CRPS→Score exchange rate is the thing that decides where
budget goes, and it is competition-specific: on crypto 0.35% of CRPS is worth ≈10 Score points, on
com-equ nothing achievable is worth more than ≈0.01. Anything below that only aimed at com-equ is,
by that table, not worth doing however sound the statistics.

In descending expected value:

1. **Statistical power.** Every remaining effect is at or below the noise floor of a 2-day window.
   Re-run the sweeps over 10+ days with 1000 paths before believing any 0.2% result — including some
   of mine, and specifically re-test MA(1) on equities where I could not resolve it.
2. **Per-asset `kcal` with hierarchical shrinkage** toward an asset-class mean, rather than the flat
   0.7× I used, plus an explicit check of what happens if realized vol regime-shifts upward. Note the
   flat refit is closed: a per-asset dispersion refit on crypto-24h had all five assets select the
   same grid edge and three of five reversed sign on the holdout (iteration 30). Shrinkage is a
   different and better-posed question, but the evidence says the per-asset signal there is weak.
3. **Session-conditional HAR coefficients and EWMA decay.** §4ter conditions the innovation *shape*
   on the US cash session and gains 0.33–0.47%; the HAR coefficients and the EWMA decay remain
   session-blind and untested. This is the most promising open lead. One component of this idea is
   now closed: the **zero-atom rate is not worth conditioning** — the model estimates it from a
   trailing 12-step window, which tracks the session implicitly to within 0.41 pp against a
   2.6% → 33.6% range, and `zi_mix` is 0 in any case (iteration 25).
4. **Extend the session idea to a scheduled-event calendar** — for the tokenised equities, earnings
   dates and macro releases. §5.6 argues this is the most likely explanation for the com-equ leader's
   dominance. It needs a data source my snapshot does not contain, so it was out of scope here, but
   for a real miner it is legal, cheap, and probably worth more than everything else on this list
   combined.
5. **Do not re-run** — I spent real budget on all of these and they are closed:
   - volatility forecasting in any form: bounded at 0.34% on crypto (§5.1) and 0.57% on com-equ.
   - zero-inflation (§5.1b), stochastic vol (§5.2), Hurst (§5.3), fat-tail shape beyond "use a fat
     tail" (§5.4).
   - refining the seasonal *curve*: it self-cancels (§4ter-b).
   - drift / centre terms on the equity perps: refuted with the sign inverted, and unfittable from a
     10-day window anyway (§5.5).
   - chasing the com-equ leader with better calibration of this model family: a perfect per-prompt
     dispersion oracle closes only 10–15% of the gap (§5.6) — and §7.4 shows that even *succeeding*
     there would be worth ≈0.01 of a Score term.
   - tuning MA(1) θ on either increment: both are at their fit/holdout optimum (§5.7, iteration 29).
   - the zero atom, in any session-conditional form (iteration 25).
   - a per-asset dispersion refit on crypto (iteration 30).
   - variance reduction beyond the existing per-step stratification: the total finite-*M* bias
     remaining at the served M = 1000 is **+0.040%** (§6), worth ≈0.05 of a Score term even if
     eliminated completely.

   The oracle bound is the tool that would have saved most of that budget, the PIT is the tool that
   would have saved much of the rest, and the §7.4 payoff table is the one that would have stopped me
   spending it in the wrong competition at all.

---

## 10. Reproducing this

```
agent/modeling.py                 the champion: simulate(asset, start_time, time_increment,
                                  time_length, num_simulations, context_prices)
agent/artifacts/calib.npz         per-asset season/HAR/pools/kcal/zq (offline, no network)
agent/research/harness.py         score on the field's exact prompts + rank_of / field_panel
agent/research/field.py           per-miner panel: who is actually good
agent/research/implied.py         CRPS-optimal dispersion multiplier (same-shock rescaling)
agent/research/biaswin.py         is a vol bias systematic or a window artifact?
agent/research/fit_kcal.py        out-of-sample kcal fit  (fit_kcal_grid.py for SP500)
agent/research/zeroinf.py         the zero-inflation measurements behind the refutation
agent/research/nsim.py            finite-ensemble CRPS bias + latency
agent/research/payoff.py          CRPS edge -> Score, via the real softmax
agent/research/rank.py            official synth_lib backtester end-to-end
```

All scoring goes through `synth_lib.backtester` / `synth.validator.crps_calculation`. Prices come
exclusively from the local snapshot. `simulate()` runs fully offline, honours whatever
`num_simulations` it is given, and takes ~9 ms at 1000 paths — far inside the latency budget.
