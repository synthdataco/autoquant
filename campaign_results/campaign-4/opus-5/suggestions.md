# Suggestions

Notes for whoever runs the next campaign, or maintains the harness.

## On the harness and the data

1. **`SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is documented in CAMPAIGN.md but not set in the
   sandbox**, and there is no offline `miner_scores` bundle. `backtest()` therefore reaches the
   live Synth API for the field and the rewards history on every call (~20 s of the ~60 s a
   `predict.py` run takes, and 700 k rows for one `--days 7` run). Shipping the bundle, or
   setting the variable to `market_data/`, would make the official path several times faster and
   fully offline.

2. **`_fill_gaps_from_realized_paths` silently falls back to one API request per prompt** when it
   does not think it is offline. A local grid of start times never matches the validator's
   irregular prompt times, so the prefetch considers every prompt uncached, issues requests
   0.6 s apart, and then cannot write the cache because `market_data/` is mounted read-only. The
   symptom is a sweep that hangs at 0 % CPU with no error. Setting the offline root around the
   call restricts it to `RealizedPathStore.get`, which snaps within 30 minutes — the same
   tolerance `backtest()` uses. Worth either a warning or a `max_requests` guard.

3. **`_slice_real_prices` takes `.iloc[:, 0]`.** Passing a full OHLCV frame scores against
   `open` instead of `close`, and the resulting CRPS shift (−2.5 % here) is small enough to look
   like a modelling result rather than a bug. A column check would be cheap.

4. **Competition lookup must be keyed on `(asset, time_length)`.** BTC/ETH/SOL/XRP/HYPE belong to
   both `CRYPTO_24H` and `CRYPTO_1H`; the obvious `{a: c for c in comps for a in c.asset_list}`
   comprehension silently scores 24h prompts with the 1h interval set (+80 % CRPS). Perhaps
   `competition_config` could expose a helper.

5. **The parquet index is `datetime64[us]`, not `[ns]`.** `index.view("int64") // 60e9` — the
   idiom most people reach for — yields a minute index that barely advances, which made every
   return look like it spanned a gap and silently emptied my volatility sample. Models looked
   identical and 32 % worse than baseline with no error anywhere. `np.asarray(idx.values,
   dtype="datetime64[m]").astype("int64")` is unit-independent.

6. **"No trade" and "no move" are indistinguishable in this data**, and roughly half of a thin
   asset's overnight minutes are one or the other. Any robust statistic has to be chosen with
   that in mind — a bucket median |return| collapses to exactly 0 and produced the single worst
   regression of this campaign (+10 % on XAU).

## On `agent/predict.py`

A1. **`generate()` cannot produce predictions for most of the asset universe.** It builds the
   context with `MinutePriceStore.load_range`, which raises on a single NaN close. But the model
   contract states that `context_prices` MAY CONTAIN NaN and that the model must drop rather than
   interpolate — exactly the situation every com-equ asset (and HYPE) is in. A `--days 7` run
   aborts with `Local store has 3 missing minute prices in the requested range`, so out of the
   box the official evaluation only works on BTC/ETH/SOL/XRP. I had to swap that one line for a
   lenient partition read (reindexed onto a gapless minute grid) to score 14 of the 18
   asset/configuration combinations. The scoring call is untouched. Worth fixing upstream, since
   the strict loader is right for realized paths and wrong for contexts.

A2. **`PREDICTIONS_DIR` is a fixed shared path that is never cleared**, and the prediction
   filenames depend only on `(start, asset, time_length)`. Two `predict.py` runs of *different
   models* therefore silently overwrite each other's files, and `backtest()` scores whatever is
   on disk. I lost a 25-minute evaluation to this — a champion and a baseline running
   concurrently produced plausible-looking but meaningless numbers, with no warning. Either
   derive the directory from the model identity, or wipe it at the start of a run.

A3. **The backtester reaches 26 hours past the last prompt, and a missing day partition triggers a
   network ingest into a read-only mount.** `backtest()` step 5 asks `download_price_data` for
   `[min(start_time) − 2h, max(start_time) + 26h]`; that helper walks the range one *day* at a
   time and sends any absent partition to `MinutePriceStore.ingest_range`. With a snapshot that
   ends 2026-09-14 this is invisible for 24-hour prompts (the latest scorable one starts 09-12,
   so +26 h stays inside) and fatal for 1-hour prompts (the latest starts 09-13 22:00, so +26 h
   lands on 09-15): all five crypto-1h assets died with
   `OSError: [Errno 30] Read-only file system: .../date=2026-09-15.parquet`. Two problems worth
   fixing: the `+26h` buffer should be `time_length + tolerance`, not a constant sized for the
   24-hour competition; and the ingest should be gated by the offline flag, because in a campaign
   with a data cutoff an implicit fetch for post-cutoff prices is exactly the thing the rules
   forbid. It only failed here because the mount happened to be read-only.

## On the scoring rule

7. **The score is a sum of univariate CRPS terms on interval returns.** Nothing in it is
   multivariate, so the joint law of a simulated path is irrelevant except through the marginals
   of aggregated returns. Cross-asset correlation, realistic microstructure, jump clustering and
   momentum are all unpriced. If the subnet wants path realism it has to price it — a rank
   correlation or a multivariate energy score term would do it. As it stands, 1000 paths is a
   very expensive way to transmit 289 marginal distributions.

8. **`E CRPS(σ)/E CRPS(σ*) ≈ 1 + 0.25·(log σ − log σ*)²`.** This is worth stating in the miner
   documentation. It tells a new miner immediately that global calibration is nearly free
   (±20 % costs 1 %) and that the entire competitive surface is the *conditional* variance of the
   log volatility forecast. I would have saved two hours by knowing it at the start.

9. **The prompt anchor is stale by design.** Context ends at `start − (ISSUANCE_LEAD + 60)` s
   while `real_prices[0]` is the close of the bar labelled `start` — 180 s of unobserved drift for
   24h prompts, 120 s for 1h. The relative-change intervals cancel it; only `24hour_abs` (3.6 %
   of the budget) sees it, and only as ~0.1 % extra dispersion. Correctly not worth modelling,
   but it is not obvious from the code and took a while to convince myself of.

## Things I would try next

10. **Cross-asset volatility pooling — but read the interface first.** BTC/ETH/SOL/XRP/HYPE
    share a common vol factor and the thin equities share the US cash session, so a common
    factor plus an idiosyncratic loading should in principle beat five independent EWMAs. I
    listed this as my top untried idea for most of the campaign and only on re-reading the
    contract realised it **cannot be served**: `simulate()` receives `context_prices` for one
    asset and nothing else, is called once per asset, and must run offline, so a peer's *current*
    returns are simply not available at prediction time. What remains implementable is an
    offline-fitted shrinkage of each asset's volatility toward a pooled long-run level, and that
    is already what `revert_halflife` and `long_halflife` do. Anyone who wants the real version
    has to either ask the subnet for a multi-asset context, or exploit process persistence across
    the validator's near-simultaneous per-asset calls by caching each context as it arrives —
    which works only if the same process serves all five and no ordering guarantee exists. I
    would not build a champion on that. **The lesson generalises: check what the serving
    interface can actually see before investing in a feature.**

11. **A proper GARCH/HAR likelihood fit per asset**, rather than EWMA half-lives chosen by sweep.
    My regression attempt (Iteration 6) failed out of sample, but it was fitted on a
    squared-log-error objective; fitting the *CRPS* objective directly, with the asymmetry in
    item 8 respected, is a different and better-posed problem. Note that Iteration 19 found the
    hand-swept half-lives to be at a flat optimum, so the prize here is small unless the fit is
    conditional on something the sweep cannot express.

12. **A width multiplier that is a function of the seasonal profile value.** The eight com-equ
    multipliers are per-asset constants, and I shipped a second level keyed on the *step size*,
    but the thing both are correcting — the ratio between the RMS the estimator sees and the
    width CRPS wants — is almost certainly a function of how dead the market is at that moment.
    The two effects I did find (a per-asset constant below 1, and a smaller constant at the
    5-minute step than the 1-minute step) are both consistent with a single underlying
    dependence on trading intensity, which suggests one profile-indexed function could replace
    nine hand-fitted numbers and generalise to a new asset with no fitting at all.

    **Partly done, Iterations 27–28.** The crudest profile-indexed version of this — one
    exponent above the weekly mean and another below it — is now shipped, and it was the
    largest remaining win of the campaign (−0.13 % on com-equ, −0.13 % on crypto-1h). It does
    not replace the nine constants; it is a second, orthogonal correction. The full version in
    this item is still open, and the reason to want it is unchanged: nine numbers fitted per
    asset will not transfer to an asset that joins the competition tomorrow, and a function of
    the profile value would.

13. ~~**The `*_gaps` block in CRYPTO_1H.**~~ Tested and closed (Iteration 20). Off the round
    hour the gaps ladder is dispersed at or slightly below the interval ladder, so the twelve
    cumulative-return terms are already paid for by the terms that share their horizon. What is
    worth carrying forward is the trap: **do not run a calibration diagnostic on a prompt grid
    whose cadence divides an hour.** A 180-minute cadence lands on :00 every time, and the
    round-hour volatility spike then shows up as a clean 28 % under-dispersion on the first five
    minutes of every path, in every asset. It is entirely an artifact — 173 or 97 minutes makes
    it vanish. Comparing two models on such a grid is fine, because the bias cancels; measuring
    an absolute calibration level on it is not.

14. **Scheduled events.** The redundancy principle (item 16) says only forward-looking
    information can be incremental, and the time-of-week profile is the crudest possible version
    of it: it knows the US cash open happens every weekday but not that this particular Wednesday
    is a CPI print or an earnings date for NVDA. An event calendar is the obvious next
    forward-looking feature, and unlike everything in item 16 it cannot be read off past returns.
    It would have to come from the snapshot or be hardcoded, since fetching it is out of bounds.

15. **A quadratic bend, and why I am telling you not to bother.** After the two-sided exponent
    shipped, the com-equ residual was no longer a U but a hump about the hinge point: the model
    is ~4 % too narrow near the weekly mean and correct at both extremes. The obvious next knob
    is a quadratic in log-profile instead of two linear pieces. Price it before you fit it. The
    spread of bucket width errors is `Var(log ratio) ≈ 9e-4` across deciles, and the identity in
    item 8 turns that straight into a CRPS ceiling of **0.02 %** — below the noise floor of a
    three-window comparison. The crypto-24h residual, which looks far more inviting (a clean
    monotone 1.175 → 1.044 gradient, reproduced at two cadences), prices at **0.06 %**, and when
    I measured it on CRPS directly it came in at **+0.004 %**. Both are closed by arithmetic.

## What did not work, and the reason they share

16. **The redundancy principle.** Four separate features that the literature says predict
    volatility all produced zero CRPS gain here: Parkinson/Garman-Klass range estimators, trade
    count and volume, the leverage effect (both the "recent down move" and the signed-return
    forms), and an intraday activity multiplier. Each has real in-sample regression signal — the
    activity feature's t-statistic against next-hour log vol is large — and each is worthless
    once a fast EWMA of realized volatility is already in the model. The reason is that all four
    act on future volatility *through* realized volatility, so a fast EWMA has already priced
    them; what looks like new information is the same information arriving by a shorter path.
    Anyone repeating this campaign can skip roughly four hours by testing the *increment* over a
    fast EWMA from the start, rather than the feature's standalone predictive power. The
    corollary is the useful half: information that is genuinely forward-looking, i.e. known at
    the prompt time but not derivable from the price history, is where the remaining gains are —
    which is exactly why the time-of-week profile was the single largest win of the campaign.

    **Correction from Iteration 25.** "Forward-looking" was too narrow a reading of my own
    principle. There is a second source of non-redundant information and it is purely
    backward-looking: anything that looks back *further than the served context reaches*. The
    interface hands `simulate()` seven days, so a monthly-timescale property of the asset — in my
    case its long-run volatility level — is outside the context by construction, cannot be
    recovered from it at any half-life, and has to be carried in as a committed artifact. It was
    worth −0.04 % on com-equ, small but signed the same way on three disjoint windows. The general
    statement of the principle is therefore: a feature is incremental iff it is outside the span of
    what a fast EWMA of the context can already see — which can be because it is in the future, or
    because it is further in the past than the context window extends.

## Process

17. I never encountered any pre-existing miner model in the workspace, so there was nothing to
    leave unread.

18. **Two disjoint validation windows was the single most valuable methodological choice.** Four
    separate things looked like wins on the first window and were noise or worse on the second:
    the fitted vol regression, the crypto half of the scale map, the crypto-1h half-life/pool/
    innovation tuning, and (at the margin) `seas_gamma=1.0`. A single validation window would
    have shipped all four.

19. **Price a residual in CRPS units before fitting it.** A diagnostic that shows structure is
    not a to-do list; it is a claim that needs a magnitude. Item 8's identity converts a table
    of per-bucket width ratios into an upper bound on the achievable gain in about a minute of
    arithmetic — `0.25 · Var(log ratio)` — and that minute decides whether a day of sweeping is
    worth spending. Two of the three residuals I chased this campaign were worth under 0.1 %
    before I started, and I only checked first on the third.

20. **Sign consistency across disjoint windows is separate evidence from magnitude, and is the
    one that survives.** Two changes shipped this campaign on effects near 0.05 % — far too
    small to call individually — because they were signed the same way in 13 of 15, and in 3 of
    3, asset-window cells. Two others with larger headline gains were rejected because the sign
    flipped between windows. Count cells, not basis points.
