# Suggestions for the subnet owner

Notes from one 18-hour research campaign on the SN50 miner. Ordered by how much I think they matter.
Nothing here is a complaint about difficulty — the hard parts were the interesting parts.

## 1. The competition is nearly degenerate, and the reward curve makes that costly

Measured on the real field (crypto-1h, 5 days to 2026-07-26, full-coverage panel of ~145 miners):

| asset | best miner | field p50 | spread best→p50 |
|---|---|---|---|
| BTC | 420.84 | 422.81 | **0.47%** |
| ETH | 571.64 | 572.96 | 0.23% |
| XRP | 550.76 | 552.21 | **0.26%** |

Roughly 200 miners sit inside 1% of each other. Combined with `beta = -0.3` and relative scoring
against the per-prompt minimum, a **0.25% CRPS improvement moves the Score term by ~0.3** (I measured
the whole curve by replaying the softmax on the real field: at +0.5% off the leader the term is 0.46,
at +0.25% it is 0.68, at +1.3% it is 0.17).

So the payoff is dominated by a difference that is *near or below what a 2-day evaluation window can
resolve*. On XRP the entire field spans +0.04% median-to-best, and split-half reliability of miner
rank there is rho = +0.38 — that competition-asset is close to a coin flip. (On BTC/ETH/SOL/HYPE rho
is +0.89 to +0.94, so rank there is genuinely skill; XRP is the pathological case, not the rule.)

Two consequences worth considering:
- **On the most compressed assets, emissions are being allocated substantially by luck.** Widening the
  softmax (less negative beta) where the field spread is below the noise floor, or scoring over longer
  windows, would make the ranking more reproducible.
- It also means a genuinely better model is nearly indistinguishable from a marginally luckier one,
  which weakens the incentive to do research rather than to run many near-identical miners.

## 2. Consider whether `CRYPTO_1H` should be so concentrated in one horizon

The 18 scoring intervals contribute very unequally. Measured shares of total CRPS:
1min 27.6%, 2min 19.3%, 5min 11.6%, 15min 6.4%, 30min 4.6%, 60min_abs 3.3%, and the 12 `*_gaps`
intervals 29.5% combined.

So ~47% of the score is the 1-minute and 2-minute marginal. At those horizons the dominant signal is
microstructure — bid-ask bounce, feed staleness, tick discreteness — not price dynamics. If the intent
is to reward *price simulation*, the current weighting rewards microstructure modelling about equally.
That may well be deliberate; it surprised me, and it turned out to be the single biggest determinant
of what a rational miner should optimise. (In my case the one clearly productive late-campaign finding
was that the optimal dispersion multiplier runs monotonically from 1.10 at 1 min to 0.90 at 60 min —
i.e. the score is largely measuring the increment autocovariance structure.)

Related: the 12 `*_gaps` intervals are all cumulative windows from the same start, so each is a prefix
of the next and they are strongly mutually correlated. They contribute 29.5% of the score but far less
than 29.5% of the *independent* information. Folding them into a single "Gaps" total in the detail
output already acknowledges this; the scoring weight arguably could too.

## 3. `SMOOTHED_SCORE_COEFFICIENT` interacts awkwardly with new assets

SP500 enters com-equ-24h around 2026-07-23 with a large coefficient (3.4379) but no prompt history.
Any newly introduced asset must be calibrated with no field data at all, pushing miners toward either
(a) not competing on it or (b) fitting on synthetic grids, which is what I had to do. A short grace
period, or a published warm-up window for new assets, would make introductions less arbitrary.

## 3b. The com-equ-24h reward cliff makes most of that competition unincentivized

Suggestion 1 is about a field so tight that luck dominates. com-equ-24h has the opposite pathology,
and it is arguably worse. Measured on the real field via the validator's own softmax replay:

| rank | 1 | 2 | 3 | 4 | 5 | **6** |
|---|---|---|---|---|---|---|
| reward weight | 0.0295 | 0.0294 | 0.0290 | 0.0281 | 0.0261 | **0.0049** |

A **5× cliff** between 5th and 6th. I have a concrete measurement of what that means for incentives.
Over one session I improved com-equ-24h from rank 209/250 to **189/250** and took SP500 to **rank
1/257** outright. The reward weight went from 0.000001 to 0.000001, and the Score term stayed exactly
**0.0000**. Replaying the softmax says that even closing my remaining gap from +5.2% to +1.0% — more
than four times everything I recovered — would be worth 0.012.

For any miner outside the top five, com-equ-24h is effectively a zero-gradient competition: real,
replicated, holdout-validated improvements return nothing measurable. The rational response is to stop
competing there entirely, which is what my own payoff analysis told me to do and what I did. If the
intent is for the commodities/equities competition to attract research effort from more than five
participants, the reward curve currently argues against it. A less negative beta on that competition,
or a floor that pays some weight below the cliff, would restore a usable gradient.

Related: because a top-5 position appears to require information beyond the token's own price history
(underlying cash market, options surface, earnings/macro calendar — see report §5.6), the cliff may be
selecting for *data access* rather than modelling skill. That is a legitimate design choice, but it is
worth making deliberately, and worth stating so miners can decide whether to compete.

## 4. Documentation gaps that cost me real budget

Small, concrete, all fixable:
- **`SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is undocumented and no bundle ships.** I spent early budget
  discovering that `build_offline_bundle.py` exists and that it needs `--no-realized-paths` when
  `market_data/realized` is read-only. A one-paragraph README in `synth_lib/backtester` would help.
- **Prompt spacing is irregular** (median 64 min, mode 72, range 8–240 min). Nothing documents this,
  and a fixed-cadence backtest silently matches only ~2/3 of real prompts — which quietly makes any
  field comparison not like-for-like. Easy trap to fall into, hard to notice.
- **`crps_ensemble` is biased upward by O(1/M)** — I measured ~1.9% at M=100 and ~0.45% at M=250.
  Miners running fewer paths are penalised for the *estimator*, not the model. Worth stating plainly:
  it is a free win for anyone who knows (stratified sampling removes it at zero cost) and a silent tax
  on anyone who doesn't.
- **The per-prompt field minimum is not an achievable target**, but it is the natural thing to compare
  against, and it sits 14–17% below the best individual miner on crypto-1h (it is a minimum over ~256
  draws, and it drifts *lower* as the field grows). I optimised against it for several iterations
  before catching the error. A note in the scoring docs would save new miners the same detour.
- **The backtester's default window and coverage filter differ substantially from a naive harness.**
  My own 2-day/lenient-panel harness reported rank 78/238 where the official 5-day run reported
  183/257 for the identical model. Both were "correct"; only one is authoritative. Documenting the
  official window and coverage semantics prominently would prevent miners from tuning against a
  yardstick that flatters them.

## 5. Rule 5 compliance

I never encountered `[private path]` in any form — no mount, no copy, no archive, no search result.
Nothing to report.

## 6. What I'd most like to see published

The most useful artifact for the miner community would be **an oracle bound per channel**: how much
CRPS is available from perfect volatility foresight, perfect tail shape, perfect drift, and so on.

I computed the volatility one late in the campaign — perfect per-prompt volatility foresight is worth
only **0.34%** of CRPS on crypto-24h, against a 0.9–1.1% gap to the leader — and it immediately
retired three lines of work I had queued and would otherwise have funded, including one I had already
partly built. Publishing those ceilings would push the field's effort toward whatever is genuinely
unexploited, which is presumably what the subnet wants to buy. It would also stop a lot of duplicated
dead-end work across miners.
