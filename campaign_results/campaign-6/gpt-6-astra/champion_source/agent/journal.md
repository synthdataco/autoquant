# Research journal

## Iteration 0 — reproducible baseline (2026-10-21)

Hypothesis: the supplied constant seven-day Gaussian model leaves exploitable temporal volatility structure, especially for equity/commodity perps and the one-hour competition.

Change: preserve the original starter; seed its generator from SYNTH_BENCHMARK_SEED (fallback 20261011). Install pinned dependencies. Read DATA.md and the validator competition/CRPS implementation. Use official backtester only; current formula for all one-hour comparisons. Budget resides in agent/BUDGET.md, not workspace root.

Design: establish 1000-path baseline on recent and earlier disjoint windows, inspect per-asset CRPS. Measure seasonality using history preceding validation. Accept modifications only on CRPS, considering the direction across asset-window cells. Start with short windows for iteration, expand promising changes. No post-cutoff information or online prices.

Result: pending.

Verdict: pending.

Lesson: the score includes non-overlapping return intervals; one-hour additionally scores returns from origin and realized volatility. A terminal marginal alone cannot satisfy it.

## Iteration 1 — measure seasonality; test adaptive variance

Hypothesis: matching base-step return variance and adapting to recent volatility improves the constant minute-return starter. Separately, learned time-of-week variance shape improves that adaptive baseline.

Change: research_model estimates 1-minute variance for 1h and 5-minute variance for 24h, with fast/slow EWMA mixtures and forecast mean reversion. No interpolation. Fit weekday/weekend half-hour profiles on June–August only, using local New York time, normalized rolling variance and a blend of squared/absolute returns with shrinkage. Test flat EWMA and then seasonal EWMA separately at 1000 paths.

Measurement: weekday max/min intraday volatility ratios range 1.88–2.65 for crypto, 2.63–6.84 for commodities/equities. Weekend/weekday ratios range 0.698–0.865 for crypto and 0.473–0.754 otherwise. Training diagnostics in results/seasonality.log. These are volatility diagnostics, not CRPS claims.

Kill criterion: discard a global modification if aggregate normalized CRPS worsens on either window without consistent asset-specific evidence; report heterogeneous results rather than selecting from rank.

Result / Verdict: pending.

## Iteration 2 — innovation tails and volatility uncertainty

Hypothesis: a Gaussian with a point volatility forecast understates both return tails and uncertainty in realized volatility. Compare Student-t innovations (df=5, unit variance) and lognormal path-level volatility uncertainty (log std=0.35, unit second moment), each added separately to seasonal EWMA.

Change: two 1000-path suites on the same pre-registered windows and prompts. Keep fast/slow volatility and seasonal profiles identical.

Kill criterion: require improvement across both windows and most asset cells; do not retain a distribution merely because its field position improves.

Result / Verdict: pending. Diagnostics of calendar effects are recorded separately; no calendar changes in these experiments.

Intermediate result: on the September window, seasonal EWMA improves all 18 asset/format cells versus starter. Mean relative improvements: crypto24h 1.745%, crypto1h 1.757%, com/equ24h 7.521%. Flat EWMA is weaker (0.211%, 0.891%, 0.260% respectively); October crypto24h flat EWMA begins with regressions, confirming that shorter memory alone is not the main contribution. Pilot BTC number differs by 0.0005% from the full suite because the evaluator now slices exactly seven calendar days rather than the last 10080 rows before issuance; comparisons use only the matched suite results.

## Iteration 3 — global width calibration

Hypothesis: a variance-matched Gaussian remains too broad in its central mass for heavy-tailed financial increments. Test a 0.85 volatility multiplier on seasonal EWMA, without any other change. A 15% shrink can plausibly gain around 1% CRPS but may fail during high-volatility regimes.

Change: 1000-path matched suite, same windows. Kill criterion: reject global shrink if either configuration/window regresses; consider horizon differences only with later validation.

Result / Verdict: pending.

## Iteration 4 — genuinely prospective calendar information

Hypothesis: event times improve forecasts beyond an EWMA because they are known before the return shock. Pool normalized event variance by asset group but collapse assets to one observation per event date. Fit through August only. Use 2/3 shrinkage in log multiplier and no suppression for nonsignificant macro bins. Shared GDP/PCE announcements combine by maximum, never multiplication. Calendar is loaded from SYNTH_BENCHMARK_CALENDAR inside simulate; missing file/types are tolerated.

LOEO diagnostic: crypto CPI first-five-minute multiplier 3.46 (leave-one-event-out 2.80–4.28); FOMC 3.83 (2.89–4.15), next25m 2.35 (1.81–3.11). The units are conditional variance multipliers, not CRPS gains. Leave-one-event-out here checks fit stability; final test is entirely later event dates. Raw diagnostics retained in results/event_fit.csv.

Change: enable future event variance only, plus an explicit 0.15 variance factor during calendar NYSE closures for concerned equities. Exact overlap integrates each event kernel over predicted steps. Compare seasonal model without events versus with events on three NEW three-day windows ending Sep09 (holiday/weekend), Sep19 (FOMC), Oct05 (NFP/weekend). These are stress validation windows, not claimed to be untouched after selection.

Kill criterion: require paired CRPS improvement on event-affected prompts across more than one release/window. Reject if any overall configuration regresses materially (>0.2%) without clear support. Holiday and macro effects will be separated if necessary.

Result / Verdict: pending.

Iteration 1 completed: seasonality wins all 36 matched asset/format/window cells. Official mean CRPS (equal asset averages within configuration): September baseline→seasonal crypto24h 4313.4987→4244.4702; com/equ24h 2342.5058→2156.2219; crypto1h 1115.1929→1094.7930. October: crypto24h 4285.8943→4240.6629; com/equ24h 2229.7086→2111.5626; crypto1h 911.4830→891.1912. 1000 paths throughout. Verdict: KEEP seasonal EWMA as provisional serving model; no nomination while further experiments remain worthwhile. Flat EWMA alone: discard for 24h; October all five crypto24h cells regressed, and com/equ mean also regressed. Lesson: deterministic future time-of-week information is more useful here than merely shortening volatility memory.

Iteration 4 correction: the offline scored bundle actually spans 2026-09-08 23:00 through 2026-10-10 23:xx, although DATA.md omits the scored-prompt span promised by CAMPAIGN.md. The Sep09 window contains only one 24h prompt, and official backtest fails in rank aggregation with KeyError('miner_uid'). No CRPS comparison is claimed for that failed run. Replace Sep09 with Sep13 (PPI/CPI and weekend). Keep Sep19 and Oct05. Holiday suppression cannot be validated in the available field window and remains provisional, not selected for serving.

## Iteration 5 — finer recurring clock structure (prepared)

Hypothesis: the weekday/weekend profile misses day-specific transitions and within-hour clustering. Fit a seven-day profile shrunk 50% to the two-group profile, and separately fit minute-of-hour variance profiles for crypto. Training stays June–August.

Measurement: shrunk first-minute variance factors are 1.18–1.29, minute30 factors 1.22–1.41; first-five-minute/rest ratios 1.12–1.19. These are variance diagnostics only. Learned artifacts: seasonality_weekly.json and clock.json. Both optional extensions are implemented in research_model but NOT selected in modeling.py.

Next: test each separately at 1000 paths on the initial matched windows, then stress windows if consistent. Kill criterion: require improvements across both initial windows and most assets in the affected configuration; do not stack unvalidated adjustments.

Iteration 2 result: path volatility mixture improves all 10 crypto1h asset-window cells; equal-asset relative gains are 0.293% in September and 0.535% in October versus seasonal Gaussian. All four daily group means improve, although individual asset-days can regress. Student-t is weaker (0.168% and 0.332%). For crypto24h both tails models slightly worsen the equal-asset means (mixture +0.098%/+0.019%, Student +0.121%/+0.012%). Verdict: KEEP mixture as a candidate for crypto1h, DISCARD blanket heavy tails for crypto24h. Com/equ mixture and Student both look favorable but need broader validation. Full numbers in results/tails_comparison.txt and raw paired artifacts.

Iteration 3 result: scale0.85 worsens every crypto24h asset/window cell; September crypto1h also worsens overall. Verdict: DISCARD global 15% shrink. Lesson: blindly narrowing a Gaussian is not a replacement for conditional volatility and a properly shaped uncertainty distribution.

Iteration 2 follow-up: test the mixture on the new stress windows, retaining Gaussian innovations for crypto24h. The first selection is based on horizon/group behavior, not per-asset cherry-picking. The full candidate still needs an independent-seed comparison later.

## Iteration 6 — finite-ensemble sampling (prepared)

Hypothesis: at the mandatory 1000-path size, scrambled Sobol normals or marginally stratified normals reduce sampling error while preserving the same forecast distribution. This is an ensemble-generation experiment, not a new market signal. Sobol's generator receives a seed derived from the environment-seeded NumPy generator; no unseeded randomness is introduced.

Change: optional samplers implemented in research_model. Sobol includes an extra coordinate for path volatility; stratification preserves each base-step Gaussian marginal. Neither is selected or claimed to improve CRPS yet.

Kill criterion: compare multiple benchmark seeds at identical model parameters; retain only if average official CRPS improves without a material configuration regression.

Checkpoint (campaign still active, no CHAMPION): baseline/ewma/seasonal/mixture/student/scale85 initial suites completed. Provisional serving modeling.py is seasonal Gaussian and independent of MODEL_CONFIG. Research knobs live only in research_model.py. Current running suites: stress_seasonal and stress_events (4 workers each), weekly and clock (2 each), stress_mixture (2). Total cap 14 workers. All start from workspace root, use uv run, exact official scoring, 1000 paths. Failed/partial Sep09 coverage artifacts moved to results/coverage_failed to prevent accidental inclusion in comparisons. First Sep13 event window improves 17/18 cells, but requires later-event confirmation. Read agent/BUDGET.md on every iteration; last status ~$3.79/100 and 17.8h remaining, no landing order.

Next work: finish pending suites; validate horizon-specific mixture + calendar combination; test optional finite-ensemble samplers across seeds; assess shorter/longer volatility memory only after seasonal baseline. Broaden to 10-day evaluation before nomination. Reports are drafts and must be updated. Every deliverable including a real-commit CHAMPION remains required at landing; do not nominate now because promising experiments remain. The active goal tracks campaign continuation.

## Resumption audit

Previous turn: PROGRESS — durable official scores, model artifacts, research code, and commits changed the authoritative state. Re-read campaign, budget, and journal on resumption. Process table confirms no Python evaluation jobs survived the interruption; restarting only incomplete cells is therefore appropriate. Completed summaries and candidate parquet scores are retained; a cell now counts as complete only when both exist. Last budget $4.28/100, 17.8 hours remaining, no landing order. No nomination exists. Continue the full research scope.

## Iteration 7 — residual information beyond EWMA (prepared)

Hypothesis: range, volume/trade count, leverage, or faster realized-volatility ratios may predict residual scale errors after the seasonal fast EWMA. Before spending official backtests, fit an interpretable ridge correction to future mean absolute returns, using January–July for training and August as a diagnostic holdout. All price labels end before September 1. Collapse the target into a one-hour scale; use six features derived only from past data. This proxy is NOT CRPS and cannot justify promotion.

Change: fit_residual.py will report out-of-sample proxy error and an approximate upper bound on potential CRPS gain (quarter of explained log-width error). Only effects with a plausible useful gain will receive official CRPS tests. Range/volume layers must beat the fast/slow EWMA ratios alone, not merely predict raw volatility.

Kill criterion: no official evaluation for feature groups whose diagnostic error does not improve on a constant residual; no serving promotion without actual official paired CRPS improvement across multiple windows. If the residual correction requires feature parity with serving, explicitly verify it before scoring.

## Iteration 8 — announcement timing at the bar boundary (prepared)

Hypothesis: events are concentrated in their first minute, and a bar-label alignment error dilutes current event gains. The loader scores close[t] against close[t+step]; since a bar labelled t closes at t+60s, an increment ending at label e integrates actual time [e-step+60, e+60). The current continuous event-overlap code uses [e-step,e), shifting the effect one minute late. Confirmed the official _slice_real_prices selects candle labels directly.

Change: first test only a +60s shift in the event-overlap interval (same learned five-minute kernels). Separately measure/freeze one-minute then next-four-minute kernels using only training releases, with leave-one-event-out stability checks. Compare those against the correctly aligned five-minute version. Preserve old results/code defaults for matched comparisons.

Kill criterion: verify kernel timing with a simple deterministic boundary example, then evaluate official CRPS on the three prospective event windows. Do not conflate timing correction with larger event multipliers or unrelated volatility changes.

Resource scheduling: on this resumed process tree, bounded queues reserve the cores of existing suites instead of oversubscribing. After clock exits: residual proxy fit, fine-event diagnostics/fit, then Sobol suite (2 cores). After stress_events exits: aligned-event suite (4 cores). After stress_seasonal exits: seeded starter baseline on TWO TEN-DAY windows, scored Sep09–18 and Oct01–10 inclusive (4 cores). Current weekly and stress_mixture retain 2 cores each. Maximum remains 14. New baseline_full establishes the final longer-window comparison; it does not use ranks or post-cutoff data. Queues are scripts, not agents; no subagent delegation.

Iteration 7 preliminary diagnostic: range/volume/all features reduce August proxy MSE, with rough log-width-based potential around 0.17%/0.26%/0.44% CRPS; leverage alone worsens the proxy. Memory ratios alone explain approximately 0.26%, so the incremental all-feature opportunity is smaller, approximately 0.18%. Before official testing, discovered training EWMA used unlimited history while serving only sees seven days. Fix the diagnostic training to exact finite-seven-day weighted moments and shift future targets to reflect the one-hour prompt's issuance lead. The first diagnostic is provisional and superseded on rerun. Serving correction will omit the fitted intercept because a global 15% narrowing already failed; test only conditional residual variation.

Iteration 4 completed: calendar adjustments improve 49/54 stress asset-window cells. All nine configuration/window means improve: Sep13 crypto24h -0.1365%, com/equ -0.0800%, crypto1h -0.1524%; Sep19 -0.2911%, -0.1740%, -0.1867%; Oct05 -0.0021%, -0.0272%, -0.0061%. Verdict: KEEP as a candidate, but effects are small and NFP result nearly neutral. No calendar holiday falls in these final windows, so holiday suppression remains unvalidated and will NOT be selected solely from these results. WTIOIL's small regressions in two windows are explicitly retained in reporting.

Iteration 5 completed: minute-of-hour profile wins 19/20 crypto cells. Equal-asset gains: crypto24h 0.0622%/0.0704%, crypto1h 0.0821%/0.1449% in September/October. Verdict: KEEP for combined validation. Seven-day shape: crypto1h worsens in both windows (0.1564%,0.0367%); October crypto24h worsens all five assets, averaging 0.2606%; com/equ signs mix. Verdict: DISCARD global day-specific refinement. Lesson: more seasonal parameters can add noise even when a coarser temporal shape is strongly useful.

Iteration 7 corrected diagnostic: exact seven-day EWMA plus issuance-aligned target gives August proxy MSE 0.110870 for memory ratios and 0.104188 for all features, against 0.119771 constant residual. Range 0.114133, volume 0.110360, leverage 0.120198. Training/serving feature checks pass for all five crypto assets, max absolute discrepancy <4.7e-13. Verdict: official tests warranted for all-features and memory-only corrections; discard leverage-alone study. Neither proxy number is a CRPS result.

## Iteration 9 — pretrained volatility forecasting

Hypothesis: a small pretrained time-series model can forecast the slowly evolving residual volatility level beyond a fixed EWMA mixture. Use Chronos-T5-tiny on hourly deseasonalized realized volatility, not on raw price levels. Public Hugging Face model metadata identifies revision 29d808298f1a62493e7b9a5e08529d0d930fa189, created 2024-02-28, last modified 2025-11-21 (before all validation dates). No online prices will be fetched. Commit any selected weights; production remains offline.

Plan: acquire weights early, measure latency, and compare a conservative blend of its volatility forecast with the seasonal EWMA using official CRPS. Only one GPU evaluation process at once. If the model cannot beat the fast baseline or exceeds the practical serving ceiling, discard it and report the failure. The model does not get asset cross-series information or any future prices.

Iteration 9 infrastructure: Chronos weights downloaded (33.6 MB) and offline inference validated on the T4. The default PyTorch native-JIT attention override failed because no C compiler is installed; disabling optional native-JIT overrides uses precompiled CUDA operators. Measured load 0.57s, first forecast 1.18s, warmed forecast 0.34s, peak allocated GPU memory 0.095 GB for 32 samples × 25 hours. Set per-process GPU memory cap to 20%. Forecasts cache complete hourly histories keyed by asset, origin, seed, and input values; all market input comes from the supplied seven-day frame. This is latency evidence only, not CRPS evidence.

## Iteration 10 — refreshing the seasonal profile

Hypothesis: newer perpetual listings may change their trading pattern as liquidity matures; a July–September profile may forecast October better than the frozen June–August shape. Fit a second profile ending October 1, then compare only prompts for which that fit was available. This is a deterministic historical update, not a feature using future prices.

Change: train seasonality_oct01.json on July 1–September 30. A prospective refresh option will select it only when the context's latest bar closes on/after its training cutoff; earlier prompts retain the original profile. First measure profile differences to estimate the opportunity, then evaluate the October matched window. If useful, test an additional October window. Do not use the refreshed table for September validation.

Kill criterion: require October CRPS improvement across most asset cells without meaningful configuration regression. A rejected refresh will not be silently used in final serving.

Iteration 2 stress confirmation: mixture0.35 improves all 39 affected group/asset/window cells (24 com/equ and 15 crypto1h); crypto24h is intentionally unchanged. Com/equ gains are 0.650%,0.904%,2.192% on Sep13,Sep19,Oct05; crypto1h gains 0.951%,0.945%,1.432%. Verdict: KEEP horizon/group-specific mixture for combined evaluation. This is stronger evidence than the initial weekday-only gains.

Iteration 7 official result: residual all-features correction loses all five September cells (mean +0.0862%) and wins all five October cells (-0.1796%). Memory-only similarly loses all September cells (+0.0552%) and wins October (-0.0365%). Verdict: DISCARD both as general serving improvements; the proxy gain does not translate consistently into CRPS. Do not choose the favorable window. Lesson: a future-absolute-return proxy can suggest conditional scale predictability without identifying a robust CRPS-optimal correction.

## Iteration 11 — shrinkage in quiet sessions; mixture strength

Hypothesis A: the original seasonal estimator shrinks 10% toward weekday-wide variance, which can overinflate quiet weekend/overnight bins. Reduce this shrinkage to 2% while keeping its learned raw profile and EWMA unchanged. Recover the raw profile algebraically from the recorded 10%-shrunk artifact, preserving training dates. Test 1000-path matched initial windows first, then stress windows if useful.

Hypothesis B: the strong mixture gains during quieter stress windows may favor larger uncertainty. Separately test log-volatility std 0.50 versus the measured 0.35, with unchanged seasonal table. Evaluate the full initial grid to document the crypto24h tradeoff instead of assuming a group-specific improvement.

Kill criteria: reject a blanket modification if it worsens a configuration/window; only promote group-specific settings after broader evidence. Neither experiment may be judged on field rank.

Evaluation durability: write candidate parquet and summary via temporary files and atomic renames, with summary published last. The suite resumes only when a valid JSON summary and readable nonempty parquet footer both exist, preventing an interrupted write from being mistaken for a completed cell. This changes storage only; official scoring is unchanged. Historical experiment arrays are float64 NumPy files (as were the starter JSON floats); the harness's canonical generator writes float32. Before final nomination, repeat the selected model and baseline at canonical float32 precision to bound any effect on small sampling gains.

## Iteration 12 — combined candidate on full windows

Hypothesis: separately validated seasonal, within-hour, event and group-specific mixture effects survive combination. Candidate v1 combines original seasonal EWMA, crypto within-hour shape, five-minute event kernels, and mixture0.35 for crypto1h and com/equ24h only. No unvalidated holiday suppression, residual ML, refresh, stronger mixture, finer kernel, or sampling change is included.

Timing selection: the +60s alignment improves crypto1h on all three stress-window means (~0.006–0.007%) but slightly worsens crypto24h in two windows (~0.009%) and helps it in one (~0.004%). Candidate v1 uses aligned kernels for 1h and retains the empirically validated original overlap for 24h; this is a documented response-timing modeling choice, not a claim that a bookkeeping correction uniformly improved CRPS. The finer event kernel remains under test and initially regresses.

Change: official 1000-path evaluation on two full ten-day windows, scored Sep09–18 and Oct01–10, against the newly completed baseline_full. These windows contain development observations and therefore are broad validation, not a genuinely untouched holdout. The campaign's next-ten-day future window remains unseen. Keep configuration and asset results separate. Kill criterion: candidate must improve all three configuration means on both broad windows and show predominantly favorable per-asset cells.

## Iteration 13 — historical validation without field ranks (infrastructure)

Hypothesis: the many small refinements need validation outside the repeatedly examined September/October cells. The snapshot's cached validator paths include earlier actual prompt times even when the field-score bundle does not. Add a cached-prompts mode to agent/predict.py that calls synth_lib.backtester.scoring._score_single_prompt, the SAME official worker used by orchestration.backtest. Do not reimplement CRPS, retrieve online data, or fabricate field scores. Reference selection remains snapshot closes first and cached validator path only on NaN, matching ordinary backtest behavior.

Before historical use, verify exact per-prompt parity with ordinary backtest on a window already scored. Then fit historical seasonal tables only through July 1 or August 1 before evaluating later July/August prompts. These are additional retrospective robustness checks, not claims of a completely untouched model-selection process. Record coverage and omit rank entirely.

Iteration 10 initial result: July–September seasonal refresh improves 7/8 com/equ assets in the October window (mean -0.2168%) but worsens all five crypto1h cells (+0.1810%) and four of five crypto24h (+0.1080%). Verdict: DISCARD blanket refresh; retain a com/equ-only refresh hypothesis for further October validation. A newer artifact is not automatically better.

Iteration 11 initial result: 2% seasonal shrinkage improves all ten crypto24h cells (means -0.0404%/-0.0482%), but com/equ and crypto1h signs reverse between windows. Keep only as a candidate needing stress tests. Mixture0.50 is weaker and less consistent than 0.35 on initial windows; crypto24h worsens every cell (+0.602%/+0.438%). Retain 0.35 as the preferred mixture. No parameter is selected from a favorable asset alone.

## Iteration 14 — volatility memory sensitivity

Hypothesis: the remaining one-hour error may come from the fixed 60-minute fast EWMA. Test 15-minute and 240-minute fast half-lives separately, keeping the six-hour/three-day mixture weights, forecast reversion, seasonal shape and Gaussian innovations unchanged. This isolates memory from the learned residual layer that failed cross-window validation.

Change: two 1000-path crypto1h suites on the original matched windows. Kill criterion: require mean improvement on both windows and most assets before combining with the validated volatility mixture.

Iteration 13 parity gate passed: cached-prompt mode reproduces ordinary official backtest CRPS exactly (maximum difference 0.0) on BTC24h 48 prompts, BTC1h 288 prompts and sparse AAPLX24h 48 prompts. The gate checks timestamps and every candidate CRPS value, not just the mean. Historical baselines now run using only snapshot/cached paths and no rank.

## Iteration 15 — event multiplier shrinkage

Hypothesis: the event fit's 2/3 log shrinkage is conservative enough to understate reliable CPI/FOMC effects. Raise only positive variance multipliers to power1.5, approximately undoing that shrinkage, while retaining the aligned five-minute kernels. Test the same three held-out release windows against stress_aligned. Do not change event dates, widths, EWMA or innovation law.

Kill criterion: require improvement across more than one event/window; reject stronger multipliers if they materially damage the nearly neutral payroll window. Report low event counts and heterogeneous WTIOIL effects. In parallel, finish the previously registered quiet-session shrinkage hypothesis on stress windows before choosing it.

Iteration 14 initial result: 15-minute fast EWMA improves 9/10 crypto1h cells, mean -0.1803%/-0.2378%; 240-minute EWMA worsens 9/10, +0.0882%/+0.6986%. Verdict: KEEP fast15 for combined validation, DISCARD fast240. This simple causal change is more consistent than the fitted residual layer. Candidate v2 changes only the one-hour fast half-life relative to candidate v1; test its incremental benefit on the same two full ten-day windows.

Iteration 6 multi-seed result: on BTC/SOL/AAPLX/SP500 across three seeds and two windows, Sobol mean change is -0.0700% com/equ24h, -0.0164% crypto24h, but +0.0235% crypto1h. Results vary by seed, and crypto24h reverses at seed37. Verdict: do not promote a universal Sobol sampler. Test marginal stratification separately: it exactly balances each base-step Gaussian marginal at N=1000, unlike truncating a 1024-point Sobol net. Retain the original sampling pre-registered multi-seed requirement for any promotion.

## Iteration 16 — dependence between successive returns (prepared)

Hypothesis: stochastic volatility changes marginal widths but independent increments still impose square-root-of-time aggregation. Bid/ask bounce or short-lived price reversals can make minute volatility overstate longer-horizon uncertainty, especially on perpetual venues. Measure adjacent-return correlations and variance aggregation ratios on June–August data before implementing anything. If effects are material, test a strongly shrunk, context-estimated AR(1) return dependence while preserving each base-step variance. This uses only one asset's observed context and drops missing pairs.

Kill criterion: skip simulation work if correlations and aggregation ratios are negligible; otherwise require official CRPS improvement across multiple windows. Do not infer predictive drift from a variance-ratio diagnostic.

Iteration 12 completed: candidate_v1_full improves all 36 asset/format/window cells at 1000 paths. Equal-asset CRPS means, starter→v1:
- Sep09–18: crypto24h 4296.6726→4217.6681; com/equ24h 2316.0261→2101.4462; crypto1h 960.4265→927.0646.
- Oct01–10: crypto24h 3531.2042→3426.4855; com/equ24h 2065.4112→1818.4707; crypto1h 741.2317→691.7005.
Mean per-asset percentage reductions: September 2.0006%,8.7439%,3.7744%; October 3.0694%,11.8400%,6.9060%. Every configuration's daily mean improves on all 20 scored days, but some asset-days regress (notably HYPE1h Sep17, approximately +14%). Verdict: KEEP and promote to provisional modeling.py; preserve seasonal_checkpoint.py. No CHAMPION yet: faster memory, historical checks, calibration and sampling work remain worthwhile within the envelope.

Serving robustness: calendar cache now keys reads by path, modification time and file size, so a daily refresh at the same environment-provided path is observed. Missing files are not permanently cached. This does not change any existing snapshot forecasts. Both provisional modeling.py and research_model.py receive the fix. Continue com/equ-only profile refresh testing on the independent Oct02–04 stress window; crypto refresh remains rejected.

Iteration 16 measured result: adjacent one-minute correlations range approximately -0.018 to +0.022; five-minute correlations -0.035 to +0.014. HYPE has rho1 +0.0087, contradicting the anticipated large bounce effect. Hourly/minute volatility aggregation ratios mostly lie 0.96–1.03 (SPCX 0.937). A few-percent width adjustment affects only coarser score terms; plausible total gain is small relative to estimation noise. Verdict: DISCARD the AR(1) implementation plan for now. Lesson: measuring dependence first prevents adding unsupported path complexity.

Iteration 9 initial result: Chronos quarter-blend improves 9/10 crypto1h cells, mean -0.1382%/-0.1598%, but crypto24h means worsen (+0.0068%/+0.0395%); the two tested com/equ assets are nearly neutral. Keep only as a one-hour candidate for incremental testing beyond fast15. This is NOT evidence of a universal pretrained-model advantage.

Iteration 10 stress result: com/equ refresh improves all eight Oct02–04 cells, mean -1.5889%, corroborating 7/8 weekday-window gains. Keep for com/equ combined validation; do not refresh crypto.

Resumption audit: budget $17.28/100, 17.2 hours remaining, no landing order. Existing v2 and historical jobs survived; do not duplicate them. Chronos initialization probe confirms neither CPU nor GPU RNG state changes. Nevertheless reseed immediately after loading to make the intended reproducibility boundary explicit; verify cached versus uncached same-input output before launching incremental fast15 comparison.

Iteration 15 stress result: power1.5 improves 49/54 cells and all nine configuration/window means. Crypto24h gains 0.1075%,0.1171%,0.0014%; crypto1h 0.1294%,0.0650%,0.0058%; com/equ 0.0522%,0.0856%,0.0188%. KEEP for incremental combination; WTIOIL remains an exception. Iteration 11 stress result: shrink .02 improves 14/15 crypto24h cells and all three means (0.1582%,0.0314%,0.2282%). KEEP crypto24h-only for combined testing; com/equ remains inconsistent.

## Iteration 17 — isolate petroleum calendar and combine validated changes

Hypothesis: WTIOIL calendar regressions arise from a weak EIA-specific kernel rather than all macro events. Add an event exclusion list and remove only EIA_PETROLEUM on the three existing stress windows, holding power1.5 and aligned overlap fixed. Reject exclusion unless it improves more than one window and produces no material regression. Separately, test stronger events, lower crypto24h shrinkage, and com/equ refresh as individual changes against candidate v1 on full windows before stacking. The one-hour faster-memory candidate remains under test.

Iteration 9 follow-up: test Chronos quarter-blend added to fast15 on all five crypto assets and both initial windows at 1000 paths, one GPU worker. Require improvement in both configuration means and most asset cells. A benefit versus fast60 alone does not establish additional value.

Chronos cold/warm test passed exactly (maximum output difference 0.0 after clearing forecast cache). Added explicit prediction-dtype option to evaluator/suite; default stays float64 for continuity, canonical float32 will be a separately named matched experiment.

## Iteration 18 — filtered empirical innovations (diagnostic)

Hypothesis: standardized observed returns retain tails and central concentration beyond Gaussian increments, potentially improving the one-hour realized-volatility distribution. Before implementing bootstrap, measure June–August returns divided by a strictly lagged seasonal EWMA scale: kurtosis, tail probabilities, and central absolute quantiles. Compare minute and five-minute samples. These diagnostics are not CRPS and cannot promote an innovation model. If excess tails survive filtering, test symmetric empirical resampling with unit variance; preserve zero drift, same volatility forecast and official 1000-path scoring. Reject if gains fail either window; do not assume a large kurtosis implies useful CRPS improvement.

Iteration 13 historical result: July/August seasonal model trained through July1 improves 33/36 asset-window cells; all six configuration means improve. Mean per-asset reductions for windows ending Aug01/Aug15: crypto24h 0.2078%/1.0166%, com/equ 1.8790%/8.0644%, crypto1h 1.6288%/1.4224%. These historical checks corroborate the central seasonal effect with an earlier, causally trained artifact, without field ranks.

Iteration 18 diagnostic reveals strong residual tails (e.g. XAU minute kurtosis32, >3-sigma probability1.49%; Gaussian benchmarks3 and0.27%). Build symmetric per-asset innovation quantiles from June–August standardized returns, cap absolute standardized shocks at8 before quantization to limit unstable extreme samples, then normalize the finite quantile grid to unit variance. Evaluate empirical innovations alone against the matched seasonal Gaussian; do not combine with path mixtures until this isolated test passes. Quantile clipping and fitting choices are fixed before CRPS results.

Iteration 17 petroleum result: exclusion changes WTIOIL CRPS by +0.0033%, -0.0005%, 0.0000% across the three windows. DISCARD exclusion: it neither explains the calendar regressions nor yields consistent gains. The weak EIA kernel is not the main culprit; avoid asset-specific calendar tuning on these tiny effects.

Precision verification started: full two ten-day windows for seeded baseline and candidate v1 at float32, 1000 paths, sequential suites with two workers. Compare each against its matching completed float64 run before attributing sub-basis-point gains to model refinements. Total active evaluation workers capped at14; empirical innovations use two CPU workers and Chronos only one GPU worker.

Iteration 14 full combined result: candidate_v2 improves all ten full-window crypto1h asset cells relative to v1. Equal-asset CRPS September 927.0646→925.4813, October 691.7005→691.3552; mean per-asset gains0.1456%/0.0585%. KEEP the 15-minute fast half-life for the next serving candidate. This gain survives the selected path-level volatility mixture.

## Iteration 19 — volatility memory at the daily horizon

Hypothesis: the successful shorter fast memory in one-hour forecasts may also help daily paths during regime transitions. Test 180-minute fast half-life instead of720 for 24h, keeping 50% slow weight and12h forecast reversion unchanged. This isolates initial volatility adaptation; no change to seasonal shapes, distributions, or calendars. Compare official 1000-path initial windows against seasonal Gaussian, crypto and com/equ separately. Kill criterion: improvement in both window means and most asset cells for a group, followed by stress confirmation before promotion. Do not extrapolate the one-hour result without measurement.

Iteration 17 refresh combined result: all eight full October com/equ cells improve over v1; equal-asset CRPS1818.4707→1806.3759, mean per-asset gain0.6290%. KEEP the com/equ-only refresh for next combined candidate. Earlier prompts use the June–August artifact, selected by the actual context cutoff. The provisional fast15 serving implementation exactly matches candidate_v2 for BTC both horizons and sparse AAPLX at an arbitrary17 paths.

## Iteration 20 — paired antithetic sampling with volatility mixtures

Hypothesis: pairing each complete innovation vector with its negative reduces finite-ensemble location error. For a volatility mixture, each pair must share the same volatility draw; otherwise the intended symmetry is lost. Implement this pairing only behind a research option, preserving Gaussian/mixture distributions and environment-derived seeds. Compare candidate v1 against paired antithetic on BTC/SOL/AAPLX/SP500, both formats where applicable, two initial windows, three seeds (20261011,11,37). This is a sampling change, not extra market information. Require favorable pooled mean in each configuration and consistency across seeds before promotion; report any one-hour volatility-score tradeoff. Do not select based on one seed.

Iteration 11 full combined result: lower crypto24h shrinkage improves9/10 asset-window cells, mean per-asset gains0.0918%/0.0620%. Equal-asset CRPS4217.6681→4214.0169 and3426.4855→3424.4627. KEEP group-specific .02 shrinkage; SOL October is the reported exception.

## Iteration 21 — robust volatility estimation

Hypothesis: rare squared-return shocks make an EWMA scale noisy; an absolute-return estimator can respond more stably. Test an equal blend of current squared-return EWMA and squared absolute-return EWMA, with the latter's correction factor fitted from June–August standardized returns (1/E|z|², where E[z²]=1). Keep the seasonal shape, half-lives, forecast reversion, and Gaussian innovations unchanged. The factor is fitted per asset/base lag, before validation, and never retuned to scored windows. This uses the observed residual tails to change variance estimation, distinct from changing simulated innovations. Require improvements on both initial window means and most cells within a configuration, then stress confirmation. Reject if merely narrowing output reproduces the failed width ablation.

Candidate v3 construction check initially caught a failed text replacement (shrink/refresh keys absent). Fixed before scoring or promotion; exact equality now holds against the measured group-specific ablations for BTC both horizons and sparse AAPLX in both September and October. No score was attributed to the broken draft.

## Iteration 22 — simple control for pretrained volatility gains

Hypothesis: Chronos's small gain could come from incorporating recent hourly variance rather than pretrained dynamics. Add a persistence control using exactly the same complete-hour deseasonalized variance series, same quarter blend and same ratio clipping, but forecast the last complete hour's variance at every future hour. Compare against fast15 and Chronos_fast15 at1000 paths, five crypto assets, both initial windows. This control needs no GPU. Keep the model only if it adds value beyond the simpler control; judge official CRPS, not training narrative.

Iteration19 completed: fast180 wins only6/26 daily-horizon asset-window cells. Crypto24h means -0.0505%/+0.2990%; com/equ +0.0846%/+0.1668%. DISCARD the shorter daily memory. Lesson: volatility persistence and score horizons differ; a one-hour improvement is not a license to reuse the setting for24h.

Uncertainty reporting plan: summarize v1 versus baseline with paired asset-day CRPS differences and circular two-day block resampling inside each ten-day window. Resample days jointly across assets to retain shared shocks. This is a descriptive interval on development-selected observations, not a confirmatory significance test; overlapping24h horizons and model selection remain limitations.

Descriptive block intervals (v1 vs starter): daily-normalized mean reductions are2.37%/3.67% crypto24h,10.20%/14.49% com/equ,4.95%/8.96% crypto1h in September/October. All six circular two-day bootstrap intervals stay below zero; full arrays in results/candidate_v1_full_block_intervals.json. These differ from ratios of window-wide averages because they weight each day's normalized change equally. They do not correct for model selection and are not a guarantee on the unseen period.

Iteration18 initial result: empirical innovations improve crypto1h9/10 cells versus Gaussian (mean gains0.1570%/0.3407%), but lose all10 against the selected path mixture (+0.1369%/+0.1952%). COM improves10/16 versus Gaussian (-0.4895%/-0.2302%) but signs reverse versus mixture (-0.0176%/+0.0549%). Crypto24h worsens both window means (+0.1773%/+0.0826%). DISCARD empirical innovations as a replacement for the selected distributions. One distinct follow-up remains warranted: independent-increment tails could complement path-level uncertainty. Test empirical+mixture0.35 against mixture0.35 for COM and crypto1h only; no crypto24h promotion. Reject if either window regresses for a group.

Iteration22 persistence control completed: only2/10 cells improve; mean changes+0.2380%/+0.0210%. DISCARD hourly persistence. Thus the favorable Chronos result is not reproduced by simply blending the latest complete-hour variance. This does not prove pretrained superiority to every classical model; it rules out this particular simple explanation.

Iteration9 next combined test registered: add Chronos quarter-blend to candidate_v3 for crypto1h on both full ten-day windows. Compare against candidate_v2_full (v3 is exactly identical for1h), holding fast15, clock, event kernels and mixture0.35 fixed. Require favorable means in both windows and predominantly favorable asset cells. Run only after the existing GPU suite exits; one GPU worker throughout.

## Iteration23 — independent seeds of the provisional combined model

Hypothesis: the dominant improvements in candidate_v3 should survive finite-ensemble Monte Carlo variation. Compare seeded starter versus v3 across all assets/configurations, the two initial two-day windows, and two additional seeds11/37, always1000 paths. These complement the default-seed full-window evidence. Keep the underlying model fixed during this audit; no per-seed parameter tuning. Kill criterion: any meaningful configuration regression triggers diagnosis before nomination. These are seed robustness tests on reused price windows, not new independent market regimes.

Iteration9 fast15 follow-up completed: Chronos improves9/10 cells and both means (0.1533%/0.1442%) beyond fast15. KEEP for full combined validation, not yet serving. SOL October is a tiny regression. The stronger simple persistence control failed, as recorded above.

Empirical-mixture scheduling correction: initial command unintentionally included redundant crypto24h cells. Stopped that suite, added an explicit skip-crypto24h option, archived the already generated redundant24h artifacts, and restarted only the26 affected cells. No redundant output enters the planned group comparisons.

Iteration21 initial result: robust variance blend fails both crypto groups (crypto24h+0.0400%/+0.0024%, crypto1h+0.0112%/+0.0116%); DISCARD for crypto. COM improves11/16 cells and both means (-0.1112%/-0.0651%). KEEP as a COM-only hypothesis for the registered three stress windows against seasonal Gaussian before any combined serving test. This effect is modest and heterogeneous, so more favorable windows alone will not justify an asset-specific patch.

## Iteration24 — uncertainty that changes within a path

Hypothesis: the selected mixture gives each path one constant latent volatility for its whole horizon. A stationary mean-reverting log-volatility process can preserve every base-step mixture marginal while allowing uncertainty to change over time. Replace the constant latent normal with an AR(1) Gaussian state, half-life one quarter of the forecast horizon (15min for1h,6h for24h), same stationary std0.35 and unit expected variance multiplier. Preserve the original initial latent draw and innovation draws for a closer comparison. No return drift or cross-asset inputs are introduced.

Test against mixture0.35 at1000 paths on initial windows for COM and crypto1h. Kill criterion: improve both configuration/window means and most asset cells before stress/combination. The empirical innovation experiment already illustrates that more within-path variation can harm one-hour volatility CRPS; report that tradeoff rather than assuming stochastic volatility must win. Queue behind empirical_mix's two workers.

## Iteration25 — causal historical calendar validation

Hypothesis: calendar gains, including the preference for stronger multipliers, should survive releases outside the repeatedly examined September/October period. The supplied calendar shows FOMC onJul29 and CPI onAug12, within the already scored historical windows. Recompute event diagnostics using only January–June prices and the May–June seasonal profile, then fit a separate July1 event artifact. The existing June–August-normalized diagnostics cannot simply be filtered by date: their seasonal normalizer was fitted later. This refit avoids that hidden lookahead.

Compare original-strength and power1.5 historical event models against historical_seasonal on the two existing three-day windows endingAug01/Aug15, all assets/configurations,1000 paths, cached official scoring. Keep the original overlap convention in both event variants to isolate strength. These are retrospective robustness checks, not untouched model-selection data. Run behind robust_stress, retaining the14-worker ceiling.

Calendar audit: upcoming release types/dates were read only from SYNTH_BENCHMARK_CALENDAR, which explicitly includes schedules published by cutoff; no future prices or API queries were used. This audit does not license changing parameters specifically for an unseen release outcome.

Iteration18 combined result: empirical+mixture loses all10 crypto1h cells (+0.2297%/+0.1081%) and both COM means (+0.0448%/+0.2643%). DISCARD empirical innovation layer entirely for selected serving. Lesson: extra tails can double-count uncertainty already captured by a path mixture.

Iteration21 stress result: robust COM variance improves20/24 cells, all three means (-0.2094%/-0.2407%/-0.6176%). KEEP for incremental full-window comparison on candidate_v3, which includes calendar, mixture and refreshed COM profile. Compare September to v1 and October to candidate_refresh (exactly the corresponding v3 COM behavior). Schedule this3-worker suite after candidate_events releases its3 workers. Do not yet promote robust variance: stress gains versus Gaussian can disappear with the selected mixture.

## Iteration26 — smoothed hourly control

Hypothesis: the failed last-hour persistence control is noisy; a six-hour EWMA of the same complete-hour variance series could reproduce some Chronos benefit without learned weights. Test that single control, with the same25% blend, ratio clipping, fast15 base, five crypto assets and both initial windows. No GPU is used. Require both mean improvements and most favorable cells; compare directly with Chronos_fast15 before attributing its gain to model complexity. This is a predeclared simple-control follow-up, not a change to Chronos's inputs.

Iteration20 three-seed result: paired antithetic pooled mean changes are COM-0.0384%, crypto24h+0.0202%, crypto1h+0.0582%. Crypto24h flips from a default-seed gain to losses under11/37; one-hour worsens on every seed's pooled mean. DISCARD universal antithetic sampling. COM-only effect is small and cross-asset inconsistent (AAPLX/SP500); retain random sampling pending broader evidence rather than selecting a tiny pooled improvement from two assets.

Iteration15 full combined result: stronger events improve34/36 asset-window cells and all six configuration/window means. Mean per-asset gains September crypto24h0.0838%,COM0.0451%,crypto1h0.0683%; October0.0013%,0.0077%,0.0017%. KEEP power1.5 for candidate_v4, combined with v3's validated group-specific settings. Queue a full official float32 evaluation behind robust_v3_full, three workers, 1000 paths. No robust-scale or pretrained component is included until its own combined test passes.

Exploratory zero-mass diagnostic: fitted empirical quantiles show only0.4–4.8% exact zero5-minute returns for most assets (XAU11.1%). Minute rates are larger for SOL/XRP/XAU. Given the empirical-tail layer's failure with the selected mixture and small24h zero mass, do not add a separate zero-inflation layer now.

Iteration26 completed: six-hour hourly EWMA improves September4/5 cells (-0.0427%) but loses every October cell (+0.1983%). DISCARD smoothed hourly control. Chronos retains an incremental advantage over both simple controls tested, subject to ongoing full combined validation.

## Iteration27 — attribute one-hour gains using official components

Hypothesis: the path-mixture improvement is concentrated in the one-hour volatility score, whereas empirical or time-varying innovations can worsen it despite more realistic individual returns. Retain the official scorer's already-computed detailed return values through a pass-through recorder inside the cached backtester path; do not reproduce or alter any scoring formula. Record interval totals, especially Vol and Overall, in prompt parquet. Gate the new diagnostics on exact total-score parity with a completed normal run.

Then measure seasonal Gaussian, mixture0.35 and mean-reverting mixture on BTC/SOL across both initial windows. All use1000 paths and the official backtester worker. This is component attribution of known total scores, not an independent experiment or extra evidence of generalization. Abort the diagnostic if any total score changes.

Iteration21 full incremental result: robust COM estimation improves12/16 cells relative to v3. SeptemberCRPS2101.4462→2097.2881, mean per-asset gain0.1391%; October1806.3759→1803.8500, gain0.1462%. October must be compared with candidate_refresh, not v1, to avoid crediting the refresh twice. KEEP robust0.5 for a subsequent combined candidate; no crypto change. Benefits are heterogeneous (WTIOIL September andGOOGLX October contribute strongly), with small regressions explicitly retained.

Iteration24 initial result: reverting latent volatility worsens crypto1h in both windows (+0.0164%/+0.0358%), with only2/10 cells improved. DISCARD for1h. COM improves12/16 cells, means0.0231%/0.0045%; this is very small. Run its preregistered stress check against stress_mixture before considering complexity worthwhile. Preserve the constant mixture otherwise.

Iteration27 recorder gate passed exactly on288 BTC1h prompts: every official total equals the original ordinary-backtest result, and each retained Overall component equals the scored total. Component studies now proceed without altering the metric.

Iteration27 attribution overturns the initial hypothesis: on BTC/SOL's four asset-window cells, mixture0.35 improves BOTH price and volatility components. Approximately76–83% of the absolute total gain comes from price CRPS, despite much larger percentage improvements in the smaller volatility term. September equal-asset price/vol changes -1.4653/-0.3032; October -2.5642/-0.7881. Reverting latent volatility initially improves volatility CRPS while slightly worsening price CRPS, the opposite of the anticipated failure mechanism. Lesson: inspect weighted official components; realistic-sounding explanations from distribution shape alone can be wrong. This attribution is limited to the measured subset.

## Iteration28 — finer intraday seasonal resolution

Hypothesis: the selected half-hour profile smooths rapid changes around underlying-market opens. Fit ten-minute bins from the same June–August normalized five-minute returns, shrinking each fine bin50% toward its existing half-hour parent. Preserve weekday/weekend pooling and the same broad variance estimator. This targets intraday resolution, unlike the rejected seven-day profile that increased weekday parameters.

Compare official1000-path CRPS against seasonal Gaussian across all configurations and both initial windows; require both mean gains and most favorable cells for a group, followed by stress/combined testing. No automatic promotion of fine bins for crypto. Generalize table lookup based on the artifact's column count, and verify that existing48-bin tables return exactly the same values before scoring. Queue fitting and tests after the component-diagnostic workers finish.

Iteration25 completed: historical original-strength events improve32/36 cells and all six configuration/window means. Gains by window (Aug01/Aug15): crypto24h0.1089%/0.0190%,COM0.0790%/0.0199%,crypto1h0.1402%/0.0424%. Power1.5 improves28/36 cells relative to original strength; five of six means improve, but August COM regresses0.0037%. KEEP the event layer; retain1.5 based on the broader stress/full-window evidence while explicitly recording this small historical counterexample. Do not escalate multiplier strength on the repeatedly examined releases.

Iteration27 complete subset: latent-volatility reversion increases price CRPS on all four BTC/SOL cells; its mean volatility component improves in both windows (one BTC October exception), but cannot offset the larger price loss. This confirms that the rejected one-hour extension failed on price forecasts, not mainly on volatility scoring.

Next combined candidate v5 adds only robust0.5 for COM to v4. Test COM at canonical float32 on both full windows versus v4; crypto remains exactly v4. Reserve the two CPU workers released by completed historical calendar studies. Chronos, fine seasonality and latent-volatility reversion remain experimental and excluded.

Iteration24 COM stress result: reverting latent volatility improves22/24 cells and all three means (0.0341%,0.0221%,0.1260%). KEEP as a COM-only candidate for full incremental testing; it remains rejected for1h. Add the same quarter-horizon reversion to candidate_v5, compare at float32 on both full COM windows, and preserve all other parameters. This measures whether the smaller effect survives robust variance and refreshed seasonality rather than assuming additivity.

Iteration23 seed11 completed: v3 improves all36 asset/window cells. Mean per-asset reductions September crypto24h1.8006%,COM7.7518%,crypto1h2.2017%; October1.1731%,5.6253%,3.0204%. Seed37 is still running. These reused-market seed checks corroborate that the main gains are not one random fan's luck.

## Iteration29 — pretrained model capacity

Hypothesis: Chronos-T5-small may improve conditional hourly volatility forecasts beyond tiny without violating practical latency. Public Hugging Face metadata for amazon/chronos-t5-small identifies revision a971ba21945c4f1796b17a91fe69214b5f4ad472, created2024-02-21 and last modified2025-11-21, before the validation periods. Download only those public weights/configuration files now; no market data or post-cutoff scores are fetched.

Keep exactly the existing hourly inputs,32 samples,25 forecast hours,quarter blend and clipping; change only model capacity. Once the current tiny full-window GPU job finishes, measure cold/warm latency and peak memory, then run the initial five-asset/two-window fast15 comparison at1000 paths with one GPU worker. Kill if practical latency exceeds5s per prompt or either window mean fails to improve over tiny without predominantly favorable cells. The final model must remain offline and any selected weights committed. No claim of improvement follows from a larger model alone.

Canonical precision result, baseline: all36 full-window float32 cells completed. Largest absolute cell-mean change versus float64 is0.0000118701%; largest individual-prompt absolute CRPS difference0.007874. The precision effect is far smaller than selected model gains. Candidate v1 float32 verification remains underway; v4/v5 combined tests use float32 directly.

Chronos-small public download completed:184,640,101 bytes of model/configuration/card files, revision recorded in provenance.json. No GPU inference has run concurrently with the current tiny full-window job. Model-size experiments are queued behind that exact live process, with a latency gate before any scoring.

Iteration28 initial result: ten-minute seasonal bins improve14/16 COM cells and both means (0.1080%/0.1026%). Crypto1h worsens both means; crypto24h reverses sign and improves only3/10 cells. DISCARD fine bins for crypto; KEEP COM for stress and combined tests. Fit a July–September fine table using the corresponding monthly parent before combining with v5's refresh policy. Existing coarse-table lookup parity passed exactly for all13 assets over10080 timestamps.

## Iteration30 — additional causal profile refreshes

Hypothesis: the COM benefit from refreshing the seasonal profile may persist with more frequent updates as liquidity changes. Fit coarse rolling three-month profiles endingSep14 andOct05. Add them to v5's existing Sep01/Oct01 artifact schedule, selecting only the latest table whose training end is no later than the supplied context cutoff. Compare against v5 at float32 across the two full10-day windows, all8 COM assets. This tests updates alone, not finer resolution or changed simulation laws. Require both window means and most cells to improve; otherwise retain the simpler monthly policy. No table fitted throughOct11 is selected without this policy evidence.

## Iteration31 — Sobol under the selected COM mixture

Hypothesis: the earlier small Sobol gain on two Gaussian COM assets survives the selected volatility mixture and all8 assets. Compare v3 COM at1000 paths over both initial windows and seeds20261011/11/37; existing seed11/37 v3 runs supply controls, and default-seed COM controls are generated explicitly. Require all three pooled seed means to improve and predominantly favorable asset/window cells before any serving promotion. Crypto sampling stays unchanged. A favorable two-asset result is not sufficient evidence for the full group.

Iteration24 full combined result: latent-volatility reversion improves15/16 COM cells at float32 versus v5, mean per-asset gains0.0270%/0.0341%; CRPS2096.5230→2095.9403 and1803.7327→1803.1622. KEEP as a candidate, but defer serving promotion until independent-seed checks. Its effect is small enough that a shared random fan can create apparently consistent asset/window signs. Test the identical reversion on v3 COM at three seeds using the controls already generated for the Sobol audit; this isolates the stochastic-law change without paying for duplicate controls. Full v5 validation above measures interaction with robust variance.

Serving promotion: v5 is now frozen into modeling.py, independent of MODEL_CONFIG. Exact equality with candidate_v5 passed36 representative asset/date/format calls at arbitrary17 paths, including all13 assets and both crypto horizons in September and October. V5 adds robust COM estimation to v4; crypto behavior is unchanged. Canonical float32 measured CRPS (baseline→current): September crypto24h4296.6726→4210.5414,COM2316.0261→2096.5230,crypto1h960.4266→925.0679; October3531.2042→3424.4340,2065.4112→1803.7327,741.2317→691.3428. Crypto scores come from v4's identical behavior; COM from v5's direct run. All36 asset-window cells improve. No CHAMPION: current envelope still supports useful pending tests.

Iteration23 seed37 completed: v3 improves all36 cells, matching seed11's36/36. Configuration reductions September crypto24h1.8238%,COM7.7640%,crypto1h2.1997%; October1.1988%,5.5981%,2.9784%. Combined72/72 improvements under the two additional seeds strengthen the main-model evidence without making the reused dates independent market holdouts.

Iteration30 causal-boundary check passed on actual sparse AAPLX context. A profile endingSep14 orOct05 is not selected when the latest supplied bar closes one minute before its training cutoff, and is selected exactly when that cutoff becomes available. This tests context availability, not merely prediction date.

## Iteration32 — local width calibration

Hypothesis: rejecting the earlier15% narrowing does not establish that the current combined model is locally calibrated. Test only0.95 and1.05 global volatility multipliers around v5, one at a time, on the two full ten-day windows at1000 paths and float32. The Gaussian local-curvature heuristic predicts only about0.06% loss for a5% width error near optimum, so this is a modest refinement rather than a claim of large upside. Compare crypto with identical v4 controls and COM with v5 controls. Require both window means and most favorable cells within a configuration before selecting a group-specific scale; do not choose per asset or extend a grid merely to find a favorable result.

Comparator safeguard: add an explicit COM control override and reject differing prompt counts, so composed v5 controls remain transparent and windows of unequal length cannot be silently compared just because their end date matches. Allocate the three CPU workers released by the completed v4 suite.

Iteration28 full result: fine COM seasonality improves all16 v5 cells, mean incremental gains0.1012%/0.1569%, with CRPS2096.5230→2094.5295 and1803.7327→1800.9913. Stress confirmation improved22/24 cells and all three means (0.1454%,0.1252%,0.4009%). KEEP COM fine bins.

Iteration30 full result: extra causal coarse-profile refreshes improve15/16 v5 cells, means0.0989%/0.0392%; CRPS2094.1653/1803.0311. KEEP update policy for combined validation. This is evidence for refreshing the estimated seasonal pattern, not for retraining every parameter after each observed return.

Iteration31 three-seed COM result: Sobol improves all six seed/window means, all three pooled seed means (0.0475%,0.0483%,0.0567%), and33/48 cells; pooled gain0.0508%. KEEP COM-only Sobol for full combined-model validation. Crypto sampling remains rejected.

Iteration24 seed audit: reversion improves all three pooled seed means (0.0136%,0.0060%,0.0119%),30/48 cells, pooled0.0105%. One seed/window slightly regresses (+0.0027%). This is weaker than the default-seed full/stress evidence; keep as a candidate, but require a benefit on top of Sobol before adding both stochastic refinements.

## Iteration33 — combine fine profiles with causal refreshes

Fit fine Sep14/Oct05 tables with the same ten-minute/50%-parent rule. Candidate v6 combines those dated fine tables with the validated v5, without sampling or latent-reversion changes. At1000 paths/float32, compare both full COM windows against fine_v5 (incremental refresh) and rolling_v5 (incremental resolution). Require favorable means in both windows and most cells before promoting the combination. Other configurations are unchanged.

## Iteration34 — sampling/reversion interaction

Complete a2×2 COM comparison at v5: random+constant mixture and random+reverting mixture already exist; now score Sobol+constant and Sobol+reverting on both full windows at float32. Require Sobol to improve over random, then require reversion to improve over Sobol alone before selecting both. Small gains from two changes need not add: the earlier seed tests warn against treating common-random-number consistency as independent evidence.

## Iteration35 — a classical learned volatility recursion

Hypothesis: a variance-targeted GARCH(1,1) update may improve one-hour volatility forecasts beyond fixed EWMA weights, providing a useful classical comparison for Chronos. Normalize minute squared returns by the known seasonal profile and a strictly lagged finite-seven-day EWMA level. Fit only the GARCH response and persistence coefficients by Gaussian quasi-likelihood, with persistence constrained to0.90–0.999 and positive weights. Missing returns receive only the model's mean-reversion update; no prices are filled.

First fit June–July using the pre-July seasonal artifact and assess August quasi-likelihood as a causal diagnostic. Then refit June–August using the pre-September seasonal artifact before official September/October scoring. Neither likelihood result is CRPS. Compare1000-path one-hour forecasts against fast15 on all five crypto assets and both initial windows. Reject if either official mean regresses or most assets fail to improve; no24h GARCH inference is selected from this experiment. Check finite-context sensitivity and serving latency before any combined promotion.

Iteration35 diagnostics: GARCH quasi-likelihood improves all five August proxy comparisons; final fitted persistence0.9761–0.9914, alpha0.0475–0.0741, all optimizers converged. This is not CRPS evidence. Finite-seven-day versus14-day warm-history forecast differences are at most0.0181%, below the declared0.1% sensitivity tolerance. Warm forecast cost is0.0037s (first cached-JIT call0.0157s). The finite-window EWMA matches an explicit weighted calculation with missing observations. Proceed to the registered official one-hour test.

Iteration29 initial latency gate FAILED: small-model cold load5.089s plus first prediction0.858s gives5.947s, above the originally imposed5s cold-response gate. Warm predictions are0.280/0.272s, repeatable, peak allocated0.320GiB. No small-model CRPS run was started under that protocol.

Protocol amendment BEFORE small-model scoring: the constitution makes roughly60s the hard response bound and about5s the practical per-prompt throughput ceiling. The original script conflated one-time framework/CUDA startup with recurring inference and did not measure tiny under the same cold-start conditions. Run identical fresh-process tiny/small serving benchmarks, reporting framework startup, model load, first response and warm calls separately. Use cold<60s and warm<5s as the revised gates for a persistent serving process. This is an explicit practical-latency amendment, not a claim that the failed original gate passed. Model capacity still must improve actual CRPS; no score has been inspected for small.

Iteration33 combined result: v6 improves all16 cells versus fine_v5 (incremental refresh gains0.1075%/0.0404%) and all16 versus rolling_v5 (incremental resolution gains0.1097%/0.1581%). KEEP fine profiles plus the causal refresh schedule. COM CRPS2091.9627/1800.2596 before sampling refinement.

Iteration34 interaction result: Sobol alone improves15/16 full v5 cells, means0.0801%/0.0994%. Adding latent-volatility reversion to Sobol worsens BOTH means (+0.0009%/+0.0119%), improving only5/16 cells. DISCARD the reversion layer for the selected Sobol path generator. Its default-random gain was not additive. Candidate v7 now tests COM-only Sobol on v6's combined profiles at float32; crypto is unchanged.

Iteration29 revised fair benchmark: tiny/small both spend about6.97s on framework/CUDA startup, then0.465/0.508s loading weights. Actual1000-path cold responses8.30/8.38s, warm uncached simulation0.232–0.241/0.306–0.309s, peak allocated0.089/0.320GiB. Both pass the explicitly amended cold<60s/warm<5s gates and reproduce every path exactly. Small-model CRPS tests now proceed; the original5s cold gate remains recorded as failed.

Iteration9 full result: tiny Chronos added to v3 improves all10 full-window crypto1h cells. Equal-asset CRPS925.4813→923.8133 and691.3552→690.0451; mean per-asset gains0.1644%/0.1965%. KEEP as the current best fully validated learned one-hour candidate, pending comparison with GARCH/small and combination with later calibration choices. If selected for serving, move chronos-forecasting into default dependencies and remove any reliance on research-only module configuration.

## Iteration36 — additional market-period calibration check

The0.95 width experiment improves33/36 v5 cells and all six full-window means: crypto24h0.2191%/0.5408%,COM0.0820%/0.1908%,crypto1h0.0842%/0.1786%. KEEP provisionally while the registered1.05 comparison finishes. This local result differs from the earlier failed0.85 test and must not be generalized into an unbounded narrowing sweep.

Validate0.95 on an additional eight-day period, scoredSep23–30 (eval-endOct01), entirely after the September22 formula change and outside both original ten-day comparison windows. Use latest scoring,1000 paths,float32,all assets, and compare seeded starter,v5 andv5×0.95. All selected artifacts are available before each prompt; the October1 profile cannot be selected by pre-October context. These are additional retrospective dates, not an untouched formal holdout after the campaign's broad fitting/model selection. Reject a global scale if this window materially reverses a configuration's gain. Allocate four CPU workers within the14-worker total.

Iteration35 initial CRPS result: full GARCH replacement improves7/10 cells over fast15, means0.0885%/0.0522%. It is weaker than the quarter-blended tiny Chronos on both means (+0.0649%/+0.0921%, only4/10 cells better). KEEP as a useful classical candidate, not the current best. The comparison uses different replacement strengths, so a matched conservative blend is warranted before drawing a capacity conclusion.

## Iteration37 — matched GARCH blend control

Blend25% of GARCH variance with75% of fast15 variance, applying the same0.25–4 forecast/base ratio clipping used by Chronos. Preserve the existing100% GARCH implementation for its completed results. Test all five one-hour assets and both initial windows at1000 paths against fast15 and chronos_fast15. Require gains in both windows and most cells; no asset-specific model selection from these two windows.

## Iteration38 — calibration with the refined COM model

Test the already registered0.95 multiplier on v7's fine/refreshed/Sobol COM distribution, both full ten-day windows, at float32 and1000 paths. This isolates interaction with the new distribution; it does not expand the calibration grid. Retain scale1 if narrowing loses either mean or predominantly harms assets. Additional-market-period validation of0.95 on v5 proceeds separately.

Precision commitment completed: both baseline and v1 have36 canonical float32 comparisons. V1's largest cell-mean difference is0.0000119851%, largest prompt absolute difference0.007734; baseline bounds are recorded above. Precision cannot explain the retained improvements.

## Iteration39 — robust hourly-level control and forecast diagnostics

Hypothesis: the similar tiny/small forecasts may derive much of their benefit from moving toward a robust recent hourly variance level. The earlier persistence and EWMA controls do not test that explanation. Add a last144-complete-hours median variance control with the same25% blend and clipping, on all five crypto assets and both initial windows at1000 paths. Require favorable means and most cells, and compare directly with Chronos before attributing gains to pretrained dynamics.

Separately measure learned/base ratios, clipping rates and learned/median ratios on representative actual prompt contexts for both public checkpoints. These are forecast diagnostics, not CRPS evaluations; no future outcomes are needed. Queue this one GPU job behind the current small-model scoring job. The CPU median-control suite waits for the GARCH-blend workers, preserving the shared hardware ceiling.

## Iteration40 — quiet-session floor in the refined COM model

Hypothesis: the earlier low-shrinkage COM experiment mixed large weekend gains with weekday losses. The now-validated fine profiles, robust level estimator and Sobol sampler may change that tradeoff, and a weekend-only floor adjustment may isolate it. On v7, test two predeclared settings: global2% shrinkage, and10% weekday/2% weekend shrinkage. Compare both against the existing10%/10% control on the full September/October windows at1000 paths,float32. Do not change scale or any other parameter.

Require gains in both configuration means and mostly favorable asset cells before considering either; report weekday/weekend daily effects, and validate any retained setting on the added late-September period before final use. No additional shrinkage grid will be inferred from the favorable stress result. Table lookup accepts a two-row shrinkage vector for this experiment, with exact old-scalar parity checked first. Queue behind the existing v7 COM workers.

## Iteration41 — resumed audit and robust-control validation

Budget40.44/100,15.9h remaining; no landing order. Completed v7 improves15/16 COM cells versus v6, incremental mean gains0.0750%/0.0959%; CRPS2090.3574/1798.7229. KEEP and promote after exact serving parity. Scale1.05 worsens35/36 v5 cells and all six means; DISCARD. Scale0.95 on v7 improves10/16 cells, both pooled means0.0364%/0.1507%, but only3/8 September assets. Defer COM narrowing pending additional-period checks; do not select merely from averages.

The matched GARCH25% blend loses9/10 cells against tiny, both means about0.10% worse; DISCARD as selected learned model. Small Chronos wins8/10 against tiny but September mean per-asset regresses0.0007% (raw mean nearly tied); fails the registered both-means criterion. Retain tiny as learned comparator, not small. Hourly median beats tiny9/10 cells, means0.0080%/0.0485%. Diagnostics show learned/base ratios0.75–0.84 and learned/median1.05–1.67: much of learned behavior is level shrinkage. Do not claim pretrained dynamics add value over robust estimation.

Hypothesis: the cheap144-hour median control retains its advantage over tiny on full windows. Score median added to v3, exactly matching the existing chronos_v3_full comparator,1000 paths,float32. Separately measure median on current v7 one-hour behavior (stronger events), then median plus the predeclared0.95 width. Require both full-window means and most asset cells for each incremental choice; validate the selected combination on late September. Reserve four CPU workers, total current maximum10; no GPU needed for median.

Serving promotion: modeling.py now freezes v7. Exact parity against candidate_v7 passed36 representative asset/date/format calls at17 paths with MODEL_CONFIG deliberately set to an incompatible scale9; serving ignores it. All paths finite. The first parity script failed before simulation because default_store_root requires an asset argument; corrected to resolve the snapshot root from default_store_root(asset). No model result was inferred from that failed script.

## Iteration42 — extra-period COM refinement checks

Hypothesis: fine/refreshed/Sobol COM behavior and quiet-session corrections generalize to the added Sep23–30 period. Score v7, v7×0.95, v7 with global0.02 floor, and v7 with weekday0.10/weekend0.02 floor, each independently against v7,1000 paths,float32. These are the same registered settings, not a new parameter search. Existing v5 results provide the v7 comparator. Require the added period to support any setting selected from full windows; report all outcomes, including failed candidates. Reserve four workers (maximum14 including existing median4, quiet2, late-scale4). No new GPU job.

Artifact decision: the repository hook refused the184.6MB small checkpoint (50MiB blob cap). Since small failed selection and serving does not use it, leave its local weights untracked and retain its public immutable revision, configuration, benchmark and CRPS artifacts. Tiny remains committed. Do not raise the blob cap for a rejected model. This changes packaging only, not the measured experiment.

Iteration36 added-period result: scale0.95 improves all five crypto24h assets (mean0.4812%),7/8 v5 COM assets (0.1247%), and4/5 crypto1h assets (0.0105%). The one-hour raw equal-asset mean is effectively tied,923.0057→923.0065, despite favorable average percentage change. KEEP0.95 for crypto24h; defer1h until median interaction finishes. COM must pass its v7-specific validation.

Iteration40 full global-floor result:0.02 improves all16 v7 COM cells, mean gains0.3767%/0.6814%, CRPS2081.8772/1786.1357. By forecast-origin New York weekday/weekend, mean per-asset gains are0.3178%/0.7666% in September and0.4858%/2.1655% in October; all8 assets improve in each stratum. A24h path can cross session types, so these are descriptive origin strata, not interval attribution. KEEP pending added-period check and weekend-only comparator. Unlike the earlier coarse model, weekday gains are favorable too.

Iteration42 first result: v7 improves all8 added-period COM cells versus v5, mean0.4658%, CRPS1878.6437→1869.7969. Fine/refresh/Sobol gains survive these additional market dates.

## Iteration43 — learned versus robust hourly level in the selected model

The first four full September median cells are all worse than tiny despite the earlier9/10 small-window advantage; do not promote the cheap control from partial evidence. Finish the registered median full-window tests. Independently score tiny Chronos on v7 one-hour behavior (stronger event kernels), then with the already-declared0.95 width, at1000 paths,float32,both full windows. This tests whether the best fully validated learned model remains useful with subsequent changes. Require both means and predominantly favorable asset cells for each increment; no model capacity escalation.

Use the cached-prompt official scorer mode for these new runs. Its exact per-prompt parity with ordinary backtest was already verified; omitting field rank avoids expensive reward smoothing irrelevant to the CRPS objective. Existing queued jobs continue unchanged. Reserve one CPU/GPU worker, maximum11CPU and oneGPU; no concurrent GPU job.

## Iteration44 — identify floor interactions and avoid double narrowing

Hypothesis A: renewed low-floor gains arise from the refined COM distribution, rather than merely using longer validation windows. Test the same0.02 floor on v5 COM across both full windows, against existing v5 controls. Compare its incremental gain with the already-completed v7 floor test; this is an ablation, not a new floor value.

Hypothesis B: combining uniform0.95 narrowing with reduced quiet-session variance may overcorrect. Add only scale0.95 to v7+floor0.02 and compare with floor02_v7_float32,1000 paths,float32. Require both window means and most asset cells before keeping this combination; late-September confirmation would follow. Do not assume the favorable isolated effects add. Use official cached scoring, three workers, total maximum14 and oneGPU.

Iteration42 low-floor confirmation: all8 added-period COM cells improve, mean0.6644%, CRPS1869.7969→1857.8117. Combined with16/16 full-window cells and19/20 daily means, KEEP global0.02 floor. Descriptive two-day block intervals for daily percentage changes are[-1.145,-0.141]% and[-2.587,-0.274]%; these do not correct repeated selection. Weekend-only and uniform-width interaction comparisons still determine the final combination.

## Iteration45 — state-dependent volatility uncertainty

Hypothesis: disagreement between recent and long-memory variance forecasts should widen uncertainty about volatility itself, rather than simply shift the central variance. On v7 crypto1h only, replace fixed log-volatility std0.35 with sqrt(0.35^2 +0.10*log(initial/long)^2), capped0.55. Preserve the mixture's unit second moment, all variance forecasts and random draws. This changes tail/central-mass shape when recent and long variance disagree; it does not use future observations or train a residual target.

Test exactly this one functional form,1000 paths,float32,both full windows against unchanged v7(=v4 for1h). Require both configuration means and most asset cells to improve; otherwise discard without tuning the coefficient/cap. A favorable result would still need combination with the selected hourly predictor and additional-period validation. Queue three CPU workers after the floor-interaction queue, preserving maximum14 and oneGPU.

Iteration44 ablation result: low floor already improves16/16 v5 cells, means0.3338%/0.6749%, close to v7's0.3767%/0.6814%. Therefore the renewed effect cannot be credited mainly to fine bins/Sobol. To separate broader-window sampling from model evolution, subset the already-official full-window scores to the original two-day dates. On September's original dates, old seasonal+low-floor regressed0.0500%(1/8 wins), while v5+low-floor improves0.1973%(7/8) and v7 improves0.2829%(7/8). October original dates improve0.0970%/0.1216%/0.1235% for original/v5/v7. The added dates have larger gains, but model evolution also changes the original-date signs. Do not attribute the interaction to one component without an additional factorial ablation. Results:floor_window_attribution.csv.

## Iteration46 — forecast decay versus estimation memory

Hypothesis: the selected15-minute estimation half-life need not imply the correct four-hour forward variance-reversion half-life. The latter has remained fixed while estimation memory changed. For crypto1h only, test a60-minute forward half-life and an effectively constant forecast (half-life1e12 minutes), each against v7's240-minute control. All estimates, distributions, event layers and random paths stay identical. This brackets stronger versus negligible forward reversion without changing the history estimator.

Score both full windows,1000 paths,float32,official cached scorer. Require both means and most asset cells; reject an option on either mean regression. No finer sweep follows. Any retained decay must be measured with the selected hourly predictor and on added dates. Reserve the two CPU workers of the quiet-floor queue after its completion; total maximum14CPU/oneGPU unchanged.

Iteration39/41 full result: median144h loses9/10 cells against tiny Chronos, mean0.0574%/0.3262% worse, CRPS924.2963/692.2007 versus923.8133/690.0451. DISCARD median as the replacement for tiny despite its9/10 short-window wins. The already-queued v7 interactions finish transparently, but the model-capacity conclusion now favors tiny over this matched robust control on broader dates. Neither short-window outcome alone was reliable.

Iteration40 weekend-only result: relative to global0.02, weekend-only adjustment loses15/16 full-window cells (+0.1808%/+0.1557%) and6/8 added-period cells (+0.1951%). DISCARD weekend-only variant; KEEP global0.02. There is no reason to add its weekday/weekend branching to serving.

## Iteration47 — final COM scale interaction confirmation

The0.95+floor0.02 full-window run is nearing completion with favorable means and most cells. Score that exact combination on the additional Sep23–30 period against the completed floor0.02 control,1000 paths,float32. Require a favorable configuration mean and most assets before retaining both; otherwise retain only floor0.02. Allocate the four CPU workers released by the now-completed COM extra-period queue. Use cached official scoring and no GPU.

## Iteration48 — variance-clock Brownian bridge sampling

Hypothesis: Sobol's limited low-dimensional accuracy should prioritize cumulative path returns, not successive minute shocks. Generate Gaussian paths using a Brownian bridge on the model's deterministic cumulative-variance clock, assigning the first Sobol coordinate to path-level volatility and the remaining early coordinates to terminal/coarse path structure. This preserves the independent-normal increment law analytically while changing finite-ensemble integration error. Existing direct Sobol crypto failures do not test this coordinate construction.

Before CRPS, verify the bridge's exact linear covariance identity at1,7,60,288 steps with strongly heterogeneous variances; this checks distribution preservation without a sampling-based tolerance. Then compare v3+bridge against existing v3 controls on both initial windows and seeds11/37, all three configurations,1000 paths,float32. Require all four seed/window means and most cells within a configuration. No claim from a single lucky fan. Retained groups need default-seed full-window tests against the selected generator, giving a third independent seed and current-model interaction check. Allocate the four CPU workers released by the completed added-period scale test. OneGPU job continues independently; maximum14CPU.

Iteration44/47 scale interaction completed:0.95 on low-floor COM improves13/16 full-window cells, mean0.0753%/0.1574%, CRPS2080.2779/1783.6086. Added period improves7/8 cells, mean0.0925%, CRPS1855.6890. KEEP both low-floor and0.95 scaling for COM. Together with the independently confirmed crypto24h0.95 scale, freeze candidate v8 with these24h changes; one-hour behavior stays v7 while learned/adaptive candidates finish.

Serving promotion: v8 is frozen in modeling.py; exact candidate_v8 parity passed36 representative calls at17 paths, finite outputs, MODEL_CONFIG ignored. Canonical1000-path composed CRPS September crypto24h4201.6155,COM2080.2779,crypto1h925.0679; October3406.1522,1783.6086,691.3428. Sources are scale95_v5 crypto24h, floor02_scale95_v7 COM, unchanged v4 crypto1h. All36 cells improve versus seeded starter. Updated report/figure identify the composed sources; no CHAMPION while useful experiments remain and envelope is normal.

## Iteration49 — state-dependent mixture confirmation and width interaction

The adaptive-mixture full-window test has nine favorable completed cells, with all five September assets improving (mean0.1641%); no promotion before the tenth cell. Prepare two independent follow-ups: adaptive mixture on the added Sep23–30 period against the unchanged v5/v7 one-hour control, then adaptive mixture plus0.95 scale on both full windows against adaptive alone. Require both full-window means and most cells for the interaction, and favorable added-period mean/majority for the shape change. Keep the formula fixed; no coefficient fitting to the scored dates. Three workers wait for the registered adaptive suite, preserving the14-worker ceiling.

Iteration45 completed: adaptive mixture improves all10 full-window crypto1h cells, mean0.1641%/0.1885%, CRPS923.4354/690.1485. KEEP pending added-period and learned-model interaction checks. Iteration46 decay60 completed:September mean improves0.0411%(3/5) but October worsens0.2227%(0/5); DISCARD faster forward reversion. The registered constant-forecast comparison continues.

## Iteration50 — learned level versus adaptive distribution interaction

Hypothesis: tiny's hourly-level correction and state-dependent mixture capture different uncertainty and can improve together. Compare tiny+adaptive mixture with both completed/ongoing individual changes on the full windows,1000 paths,float32. Then score tiny alone and tiny+adaptive on the added Sep23–30 period. Require favorable incremental means in both full windows and the added period, with mostly favorable asset cells, before retaining both. A combined regression favors the better simpler component rather than stacking them by default.

Queue oneGPU/CPU worker behind the exact current tiny full+scale queue PID54625. This preserves the oneGPU rule and CPU allocation. Pretrained paths remain offline; no additional model weights are fetched.

## Iteration51 — matched constant-mixture control

Hypothesis: adaptive-mixture gains reflect state dependence rather than merely a higher average log-volatility variance. Compute its mean squared mixture width from prevalidation June08–August31 contexts at six-hour intervals, pooled equally over the five crypto assets. Use the square root of that training-only mean as a single constant-mixture control. This parameter is determined without official validation outcomes; no grid or per-asset choice is permitted.

Compare the resulting constant with both v7 fixed0.35 and the adaptive model on both full windows,1000 paths,float32. Retain a claim of adaptive benefit only if it beats the matched control on both means and most cells. Record the training width distribution. Fit and score with two CPU workers after the forecast-decay queue PID56474; noGPU and unchanged total allocation.

Iteration48 multi-seed result: Brownian bridge improves19/20 one-hour cells, all four seed/window means (seed11:0.0373%/0.0283%; seed37:0.0375%/0.0632%). KEEP crypto1h for default-seed full validation. Crypto24h regresses every October cell at both seeds; DISCARD. COM fails one of the four mean criteria and, against selected direct Sobol, worsens October and most cells at seed11; DISCARD COM bridge. Exact bridge covariance errors were below2.8e-15, so these signs reflect finite-ensemble scoring, not a changed target covariance.

## Iteration52 — bridge in the current one-hour distributions

Test bridge sampling on unchanged v8 one-hour and separately on v8+adaptive mixture across both full windows,1000 paths,float32,default seed. Compare each to its matching random-sampling control. Require both means and most cells before selecting bridge within either distribution; no blanket promotion to24h. This supplies the third seed and tests interaction with adaptive tail shape. Use the four workers released by completed bridge seed studies.

## Iteration53 — causal historical adaptive-mixture validation

Hypothesis: adaptive-mixture gains persist on older market dates with artifacts fitted before those dates. Fit a May–June within-hour clock using the existing May–June seasonal table. Combine it with the already-causal pre-July event table; retain v8's fixed one-hour parameters. Score two two-day windows endingJuly22/August22, both with the latest formula, using cached official prompts and no ranks. Compare unchanged mixture0.35 against the same adaptive formula,1000 paths,float32. Require both means and most cells. This is a retrospective robustness check, not an untouched holdout.

No September-fitted clock or seasonality is allowed into this historical experiment. Add optional clock-table selection with unchanged defaults. These studies queue sequentially after iteration52, retaining its four-worker allocation. Current learned GPU studies remain one-at-a-time.

Iteration46 complete: constant forecast worsens September0.0388%(1/5 wins), improves October0.0613%(4/5). DISCARD both faster and constant alternatives; retain the240-minute forward half-life. The estimation-memory change did not warrant changing forecast decay.

Iteration49: adaptive mixture improves4/5 added-period cells, mean0.0337%, CRPS923.0057→922.7662. Adding0.95 on full windows improves9/10 cells beyond adaptive, mean0.0340%/0.1207%, CRPS923.0706/689.3662. KEEP the width interaction provisionally; additional-period confirmation remains necessary because isolated one-hour narrowing was nearly tied there.

## Iteration54 — additional-period one-hour interaction checks

Score adaptive+0.95, bridge alone, and bridge+adaptive on the existing added Sep23–30 period at1000 paths,float32. Compare each against its matching completed control; retain a component only when full-window and added-period evidence agree. This does not add parameter values or broaden sampling to rejected24h groups. Three CPU workers are released by the completed adaptive-followup queue; other allocations remain unchanged.

Clock-table plumbing preserves exact default lookup values on10 asset/lag cases. Historical studies disable lookahead by explicitly selecting the pre-July table; Chronos is not used in those historical runs because its current helper still assumes the original seasonal table.

Iteration51 fitted control:1,700 training contexts yield constant log-volatility std0.3842461321. Contexts are spaced six hours at minute17, avoiding an exclusive round-hour diagnostic. Per-asset mean adaptive width ranges0.3731–0.3872. The control is fitted before scoring and is not selected from a width grid.

## Iteration55 — adaptive center versus extra tail uncertainty

Hypothesis: adaptive mixture gains require preserving second-moment tail uncertainty, rather than only narrowing the central fan when fast/long variance disagree. Keep mixture0.35 and apply a conditional width multiplier exp(0.35^2-adaptive_width^2). This exactly matches the adaptive mixture's median path-volatility factor while reducing its unconditional variance; all parameters derive from the already-fixed adaptive formula. Compare this center-only control with adaptive mixture and v8 on both full windows,1000 paths,float32.

Require both means and most cells for any selection; report an adaptive advantage only if the tail-preserving form wins. No refitting or new shrinkage coefficient. Two CPU workers queue after the matched-control study PID58988, retaining the overall14-worker/oneGPU ceiling.

Iteration52 bridge-alone full result: all10 cells improve, means0.1625%/0.1058%, CRPS923.5471/690.6868. KEEP pending added-period and adaptive interaction. Iteration54 adaptive+0.95 reverses on added dates: mean+0.0241%, only2/5 cells improve, CRPS922.7662→923.0809. DISCARD that width interaction; keep scale1 for the adaptive one-hour distribution despite its favorable full-window means.

## Iteration56 — verified research forecast reuse

Repeated Chronos combinations use identical hourly inputs and seeds. Add an opt-in research disk cache keyed by the complete input values, asset, origin, seed, model provenance and forecasting-module source hash. Cache only the25-hour forecast array; context changes cannot reuse a forecast. Writes are atomic, malformed entries miss, and serving remains uncached unless explicitly opted in. Generated forecast files stay untracked.

Before enabling this cache for scored experiments, queue an exact uncached-versus-disk-reloaded forecast parity check behind the current GPU queue. Then enable it only on subsequent registered interaction runs, preserving oneGPU scheduling. This changes computation reuse, not the model, prompt paths, training or scorer. Final serving latency must still be measured without the research cache.

Research cache adds an optional cache-only mode that raises on a missing/different-seed key instead of running inference. This will allow later purely cached comparisons to use CPU workers without risking a second GPU job. Its parity gate also checks that changing the seed misses; cache-only runs are not scheduled until the required forecasts actually exist.

Iteration51 control result: adaptive mixture beats the training-matched constant0.384246 on all10 cells, mean0.1175%/0.1139%; matched constant CRPS924.6111/690.8603. KEEP state dependence; a higher average fixed mixture width does not reproduce the improvement.

Iteration52/54 combined confirmation: bridge+adaptive improves all10 full-window cells over adaptive alone (0.1580%/0.1028%), CRPS921.9590/689.5160. On added dates, bridge alone improves5/5 versus v8; adaptive added to bridge improves4/5, mean0.0297%, CRPS922.5773→922.3751. KEEP the combination at scale1. No learned-model addition is implied by this CPU-only result.

## Iteration57 — final learned/bridge/adaptive interaction

After the registered Chronos interaction queue completes and populates verified caches, compare Chronos+adaptive+bridge with both Chronos+adaptive and adaptive+bridge, first on both full windows and then the added period. Keep the three-way combination only if it beats the matching two-way controls on the required means and predominantly favorable cells. Use1000 paths,float32,three CPU workers and cache-only mode, which cannot start GPU inference on a miss. The queue waits for PID58610; its three workers are released by completed extra-period interaction tests. No new GPU allocation.

Iteration53 historical counterexample: adaptive mixture improves9/10 cells but fails the registered both-means requirement. July mean improves0.0750%; August worsens0.0713%, driven by XRP+0.69%. XRP regresses on both days (+23.70/+10.76 raw CRPS daily means), with only105/288 prompts improved; this is not a single isolated prompt. Do not claim historical confirmation. The provisional v9 adaptive+bridge freezing step passed36-call exact parity before this result arrived, but adaptive selection is now unresolved and must not be nominated on the favorable September/October evidence alone.

## Iteration58 — historical sampling interaction audit

Test the actual bridge-based mixture comparison on the same causal historical periods: bridge-only versus bridge+adaptive, all five assets,both windows,1000 paths,float32. This isolates whether the historical failure persists in the proposed serving generator; no parameters or dates change. If either mean still fails, withdraw adaptive from serving and retain bridge-only pending genuinely new evidence. Record both the original failure and this interaction result; do not overwrite either. Four CPU workers are released by the completed historical queue.

Iteration55 center-only control beats the adaptive tail-preserving form8/10 cells, means0.0137%/0.0534%; CRPS923.2969/689.7943. This weakens a tail-uncertainty interpretation: conditional central narrowing is at least as useful on the development windows. Neither form earns promotion from those windows after the historical counterexample.

Causal diagnosis of the five worst historical XRP prompts: initial/long variance ratios5.92,5.86,10.56,6.17,8.99, all saturating adaptive mixture at0.55. The failure occurs in already-hot states, not calm ones; it persists over both days. These are context-only diagnostics attached to previously scored errors, not new forecasts or proof of causality.

## Iteration59 — asymmetric calm-state correction

Hypothesis: the adaptive formula should respond to a stale high long-memory floor when recent volatility has declined, without narrowing an already-hot forecast. Keep the same0.10 coefficient and0.55 cap, but use min(log(initial/long),0) in the mixture formula. No per-asset exception or cap tuning. This is a NEW post-failure hypothesis, not a reinterpretation of the failed symmetric model.

Test on both full development windows; on the same two causal historical windows; and on NEW two-day historical windows endingJuly29/August29, with matched fixed-mixture controls and all artifacts fitted beforeJuly1. Require both means and most assets in each paired stage; additional late-September and bridge/learned interaction checks follow only if it survives. Dates already used for diagnosing the failure are not called holdouts. Allocate two workers released by the completed center-only study, preserving the hardware cap.

Iteration58 confirms the counterexample under bridge sampling: July improves0.0723%, August regresses0.0871%, with XRP+0.73% and only3/5 August cells improved. DISCARD symmetric adaptive mixture from serving. Preserve it as a research candidate and keep its completed favorable and unfavorable results. V9 is revised to v8+one-hour bridge ONLY before nomination; the earlier adaptive freezing step is explicitly withdrawn. The asymmetric calm-state hypothesis is a separate registered experiment and earns no promotion from this failure.

Iteration57 scope update BEFORE those cached tests run: symmetric adaptation has failed historical selection. Add Chronos+bridge WITHOUT adaptive as the primary current-serving comparison, on full and added periods. Preserve the already-registered symmetric combinations as research results, but do not promote them without resolving their historical failure. Cached forecasts are independent of these distribution choices.

## Iteration60 — measure standardized return autocorrelation

Hypothesis: serial dependence in base-step returns changes the variance-growth curve across scored horizons, something a uniform width multiplier cannot fix. Measure non-overlapping adjacent standardized returns at1-minute lag for crypto1h and5-minute lag for all24h assets, using only June–August prices. Remove the appropriate coarse/fine seasonal profile and a strictly lagged slow volatility level; never interpolate NaN prices. Report early-training versus August correlations and a50%-shrunk/capped-at0.10 coefficient as a possible conservative model input.

These are diagnostics, not CRPS. Before adding correlated Gaussian increments, inspect sign stability and the implied long-horizon variance factor. No autoregressive mean forecast is implied: a covariance-law experiment would initially keep zero mean. Use one CPU worker within the currently freed slots; all serving choices remain unchanged.

Iteration60 diagnostics: most adjacent standardized returns have small negative correlations. After fixed50% shrinkage, implied long-horizon standard-deviation factors are0.9856–1.0008 for crypto24h,0.9850–1.0091 for COM, and0.9895–0.9961 for crypto1h. XAU is consistently positive; BTC/XRP minute correlations and HYPE five-minute correlation change sign across the two training subperiods. This is a modest horizon-shape hypothesis, not evidence for a large forecast gain.

## Iteration61 — covariance law at fixed base-step variance

Use the prevalidation shrunken correlations in a stationary Gaussian AR(1) innovation law, preserving unit variance at every individual step and zero forecast mean. Keep the observed-price start, volatility forecasts, mixtures and sampling unchanged. No conditional-return drift is added. Apply only when the requested step matches the fitted1-/5-minute lag; other step sizes retain independent innovations.

Validate exact covariance algebra before official scoring, then compare against v9 on all three configurations across both full windows,1000 paths,float32. Require both means and most asset cells within a configuration; no per-asset score-based selection. A retained group needs added-period confirmation. Four CPU workers fit within the current10-worker maximum including the reserved cached-interaction workers.

V9 serving verification after withdrawing adaptation: exact36-call reference parity, valid17- and1000-path shapes, finite/positive paths, correct starts, exact repeats and seed sensitivity passed. Warm1000-path latency:BTC1h0.0179s,AAPLX24h0.0565s,XAU24h0.0607s. The latter contexts include3715/898 missing closes, respectively. Results in candidate_v9_serving_check.json.

Iteration59 old-period result: calm-only adaptation produces tiny historical regressions in BOTH means (+0.0005%/+0.0006%), improving4/10 cells. It fails the registered criterion despite the small magnitude; do not promote or add bridge/learned interactions. Finish the already-registered new historical dates to report them transparently, without changing the threshold after seeing these signs.

## Iteration62 — causal historical learned-model check

Before any Chronos serving promotion, compare Chronos+bridge with the completed bridge-only control on both causal historical windows endingJuly22/August22,1000 paths,float32. Require both means and most cells; a failure rejects the learned addition under the same standard that withdrew symmetric adaptation.

Extend the hourly helper to accept the selected seasonal table, preserving its original default exactly. Historical calls explicitly pass the pre-July seasonal table alongside pre-July clock/event tables. Forecast-cache keys include the resulting input values and module source hash, so the different transform cannot reuse September-profile forecasts. Queue oneGPU worker behind the existing Chronos interaction queue PID58610; cached CPU-only tests may coexist because they raise rather than infer on a cache miss. No future market outcomes enter any fit.

Current frozen v9 is bridge-only for1h, with no symmetric/calm adaptation or learned forecast. Its canonical composed CRPS: September crypto24h4201.6155,COM2080.2779,crypto1h923.5471; October3406.1522,1783.6086,690.6868. All36 asset/window cells improve versus starter. Historical bridge-only also improves all10 cells, means0.2485%/0.5228%. Report/figure now reflect this actual serving state and explicitly describe the withdrawn adaptive variant.

Iteration59 new-date result also fails:July mean+0.0059%(0/5 wins),August-0.1580%(3/5). DISCARD calm-only adaptation; do not rescue it using the favorable August average. This completes the planned asymmetric follow-up without a new serving change.

Iteration61 complete: return covariance improves crypto24h9/10 cells, means0.0104%/0.0346%, and crypto1h10/10, means0.0066%/0.0158%. COM reverses sign (+0.0019%/-0.0026%) with only8/16 wins. DISCARD COM dependence; KEEP crypto as a small candidate pending extra-period and causal historical checks. Covariance-law algebra passed eight exact unit-variance/covariance cases.

## Iteration63 — validate crypto covariance beyond development windows

Score the unchanged fitted AR covariance on the added Sep23–30 period, both crypto formats, against v9. Separately refit correlation coefficients using only May–June, the pre-July seasonal/clock tables, and the same50% shrinkage/cap. Evaluate the historicalJuly22/August22 windows with matched causal controls. Crypto1h controls already exist; generate the corresponding24h controls. Require both historical means and the added-period mean plus predominantly favorable cells before selecting a group. This tests a fitting policy, not September-trained coefficients on earlier data.

Allocate four workers for the added period and four for sequential historical fitting/scoring. With the oneGPU job, two-worker matched-mixture study planned next, and three reserved cached-comparison workers, the maximum remains14.

## Iteration64 — retain the matched constant as its own candidate

Although it is weaker than symmetric adaptation on development dates, the training-derived constant0.384246 beats fixed0.35 on all10 full-window random-sampling cells (0.0469%/0.0749%). Test it as a simpler candidate with the selected bridge generator, without selecting a new mixture value from scores. Require full-window, added-period and causal historical agreement.

For historical tests, independently refit the SAME training-moment rule on May–June contexts with pre-July seasonality/clock; using the September-fitted constant in July would leak future inputs. Score both old and new historical date pairs against matching bridge controls. Freeze the rule and do not alter its coefficient or cap. Two CPU workers plus the two four-worker covariance studies, oneGPU worker and three reserved cached-comparison workers total14.

Iteration63 result: added-period covariance gains are0.0334% crypto24h and0.0009% crypto1h, each4/5 cells. Causal historical refits fail all four configuration/window mean criteria:crypto24h+0.0013%/+0.0248%,crypto1h+0.0008%/+0.0027%. DISCARD fitted return dependence from serving. Small measured autocorrelations do not establish useful future covariance adjustments under this CRPS model.

## Iteration65 — price expectation versus log-price centering

Hypothesis: zero-mean Gaussian log returns induce an upward arithmetic-price expectation, particularly for volatile24h paths. Test subtracting one half of each path's cumulative conditional log-return variance, so the underlying Gaussian-mixture process has flat conditional arithmetic-price expectation. Preserve every volatility forecast, random draw, mixture and sampler. This is one centering convention, not a fitted directional signal or drift grid.

Use v9 without return correlation, Student innovations or empirical innovations; those require different conditional-moment arguments and are explicitly incompatible with this experiment. Compare all three configurations on both full windows,1000 paths,float32. Require both means and most cells before retaining a group; any retained group needs added-period validation. Four workers are released by completed covariance studies.

## Iteration66 — temporal audit of the selected24h width

The covariance rejection raises a separate question about transferring the already-selected5% crypto24h width reduction to older regimes. Score scale1 with the exact causal historical v9 crypto24h model and compare with existing scale0.95 controls onJuly22/August22. This audits the fixed, previously selected width; no new scale is tried. Report any counterexample alongside the larger three recent windows and revisit selection if the reversal is material rather than claiming universal calibration. Four workers are released by the other completed covariance study.

Iteration66 width audit is mixed:scale1 versus0.95 regresses July0.1646% but improves August0.3154%. Thus0.95 has a material old high-volatility counterexample and is not universally calibrated. The audit compares the fixed choice with all three larger recent periods; pooled paired results are recorded in width_temporal_audit.csv. This audit was registered as temporal sensitivity/material-reversal review, not the strict all-historical-window veto used for new adaptive/covariance models. Do not conflate those standards or hide the counterexample.

## Iteration67 — independent-seed width audit

Hypothesis: the selected crypto24h0.95 width gain survives Monte Carlo fan variation. Run v9 crypto24h at scales1 and0.95 on both full windows under seeds11 and37,1000 paths,float32. Require favorable means in all four seed/window comparisons and most cells. No new width values are allowed. This directly checks the possibility that narrowing mainly compensates one fixed random fan's dispersion, raised by the sampler experiments and historical reversal. Four CPU workers use the allocation released by the completed width-history audit.

Iteration56 cache gate FAILED before cached scoring: tiny records its immutable checkpoint in revision.txt, while the cache assumed the small checkpoint's provenance.json filename. The parity job stopped; dependent cache-only and historical jobs also failed before any CRPS artifact. This exposed that process-exit dependencies release resources but do not certify prerequisite success. No failed result is treated as a comparison.

Fix: accept the model's committed provenance.json or revision.txt, still hashing it together with source and full inputs. Preserve failed logs. Restart the parity gate, all forecast-populating studies, historical GPU test, and cached CPU studies in ONE sequential fail-fast queue. This prevents an upstream failure from releasing studies that depend on populated caches. No other GPU job is active at restart; maximum concurrent CPU usage remains within14.

Iteration56 corrected cache gate PASSED: exact uncached-versus-disk forecast equality, reload without inference, and changed-seed cache miss. The sequential recovery queue is now running; the original metadata failure remains archived.

Iteration65 result: price-mean correction worsens both COM means (+0.0068%/+0.0067%) and reverses sign for crypto24h (+0.0104%/-0.0158%) and1h (+0.0060%/-0.0034%). DISCARD all groups. The mathematical centering convention does not improve this empirical objective consistently.

Iteration64 matched constant fails added September (+0.0027%) and new historical July (+0.0205%,0/5 wins), despite favorable full-window results. DISCARD. Historical August's per-asset percentage mean slightly improves while raw configuration CRPS worsens; report this distinction, not an unqualified win.

Iteration67 width seed audit PASSED: all20 additional-seed cells improve. Seed11 mean reductions0.2052%/0.5239%; seed37 0.2090%/0.5270%. Across the five disjoint default-seed windows,0.95 reduces pooled official CRPS4231.2708→4219.2935 over3840 prompts (0.2831%); duration-weighted mean cell gain0.3482%. KEEP the simple current width based on this broader loss reduction and independent-seed evidence, while explicitly retaining August's0.3170% counterexample. This is not a universal-calibration claim or a retroactive pass of the stricter adaptive-model veto.

## Iteration68 — relax narrowing only in hot24h states

Hypothesis: the historical width counterexample reflects hot regimes where the long-memory variance forecast lags. For crypto24h only, smoothly relax the selected0.95 width toward1 when initial/long variance rises from1 to2, and decay that relaxation with the existing forward-variance half-life. Formula:scale(t)=0.95+0.05*clip(log(initial/long)/log(2),0,1)*decay(t). Do not tune the threshold, endpoints or half-life.

Test this single state-based policy against v9 on both full windows, added September, the old causal historical windows and the new historical date pair (with matched new24h controls). Require favorable full-window and added-period means and historical agreement before selecting it; otherwise keep the fixed width with its documented limitation. No new scalar-width grid. Four workers are released by completed seed tests.

## Iteration69 — shorten learned decoding only if exactly equivalent

The selected learned horizon is one hour, yet the helper decodes25 hourly values inherited from the rejected24h experiment. At ordinary issuance boundaries, three hourly predictions cover the full requested hour. Test whether seeded three-step Chronos decoding exactly reproduces the first three values/samples of25-step decoding on real BTC/HYPE contexts, including an hour-boundary case. If any prefix or averaged forecast differs, do not treat this as a transparent speed optimization. No CRPS is measured here.

Run as oneGPU process only after the complete recovery queue PID67090 exits, so cache-populating and cached comparisons are unaffected. Record inference times separately from the existing serving benchmarks. Do not change forecasting code or invalidate caches before the current studies finish.

Iteration68 full result: hot-state width relaxation loses all10 full-window cells, means+0.0288%/+0.0131%. DISCARD without changing its threshold or decay. The registered extra-date runs finish for the record; the fixed width remains selected with its documented historical exception.

## Iteration70 — an independent public forecasting architecture

Hypothesis: Google's point-forecast TimesFM-1.0-200m may improve hourly log-variance forecasts beyond Chronos and the selected bridge baseline. Public metadata identifies google/timesfm-1.0-200m-pytorch revision0581e2c56cb06feb51cfd98fc2b4005b74f7187b, last modified2024-08-29, Apache2.0, before every validation period. Only model/package metadata and weights are fetched, never market data.

The old timesfm1.3.0 package excludes Python3.12, but its standalone PyTorch decoder imports only standard-library modules and torch. Vendor that exact unmodified public decoder with its Apache license and wheel SHA256, rather than changing workspace Python or numerical dependencies. This is a research compatibility choice to verify, not a claim that the full old package supports3.12.

Use the same144 complete hourly deseasonalized variances, log-relative-to-median input, native point forecast, inverse exp clipped to[-3,3],25% variance blend, and0.25–4 ratio clipping. Its experimental quantile heads are explicitly uncalibrated per the model card and will not be used. The native point-in-log-space forecast differs from Chronos's mean of exponentiated samples; compare end-to-end utility without attributing a win solely to architecture. Require a finite contiguous suffix of at least48 hours; otherwise retain EWMA, without filling prices.

Download now; queue GPU latency/repeatability gates after existing GPU work. Cold response must remain<60s and warm1000-path generation<5s, with GPU memory capped. Only then score full one-hour windows against v9 and Chronos+bridge at1000 paths,float32. Require both means and most cells; any retained model also needs added-period and causal historical confirmation. Unselected large weights remain untracked; if selected they must be committed for offline serving.

TimesFM public download completed:814,370,168-byte checkpoint, pinned2024 revision; exact26KB standalone decoder and Apache license are preserved with wheel SHA256. No workspace dependency or Python version was changed.

Protocol refinement BEFORE TimesFM scoring: benchmark the native float32 CPU decoder first. If cold<60s and warm uncached1000-path<5s with adequate memory headroom, use CPU inference for the experiment and avoid occupying the already-busy GPU. Otherwise retain the registered deferred GPU gate. This changes resource choice before any TimesFM CRPS is observed, not the accuracy threshold.

## Iteration71 — resume interrupted learned-model comparisons

Budget reread:62.32/100 USD,14.6h remaining,normal phase. Process audit confirms no research jobs survived interruption. Resume the existing fail-fast Chronos recovery queue with artifact-aware skipping; append prefix parity to the same sequential queue to preserve the one-GPU rule without stale PID dependencies. Preserve completed results. No serving change.

TimesFM CPU gate PASSED: cold8.0825s,warm0.1813/0.1830s,peakRSS2.413GiB,repeatable1000-path output. Launch the registered two-full-window one-hour official CRPS study with four CPU workers. Combined peak workers are seven (four TimesFM plus at most three cached Chronos); GPU inference remains sequential. Accuracy thresholds remain those registered in iteration70. Hypothesis and decision unchanged; results pending.

Iteration68 remaining historical results complete: hot-state relaxation changes mean per-asset CRPS by+0.0047%/-0.0530% on the original July/August pair (1/5 and3/5 wins), and-0.0055%/0.0000% on the new pair (4/5 wins and all ties). These mixed historical results do not overturn the full-window10/10 losses. DISCARD remains the verdict. Lesson: recognizing a hot-state counterexample does not establish a profitable universal relaxation policy.

## Iteration72 — smooth fine seasonal estimates without changing their mean

Hypothesis: ten-minute COM seasonal estimates retain sampling noise between adjacent bins; a fixed circular[0.25,0.5,0.25] average may improve the calibration tradeoff while preserving each weekday/weekend profile mean. Apply this same smoothed variance shape to past-return normalization and future variance, after the existing seasonal-floor transform. Keep refresh dates, robust estimator, scale, events, mixture and Sobol unchanged. This is a single smoothing policy, not a kernel-width sweep. Crypto and learned-model studies are unaffected.

Measure profile roughness before/after, verify positive values and exact mean preservation, then score all eight COM assets on both full windows at1000 paths,float32 against v9. Require both configuration means and most cells; retain only with additional September confirmation. Reject if smoothing removes useful cash-open structure. Four workers plus existing four TimesFM and at most three cached Chronos workers stay below14.

Iteration72 result:32 profile checks preserve mean and positivity; adjacent-bin squared differences fall to28.74% of their original value on average. Official CRPS nevertheless worsens ALL16 cells. September COM mean2080.2779→2082.2865(mean per-asset+0.0974%);October1783.6086→1785.4575(+0.1025%). DISCARD smoothing with no kernel-width rescue. Lesson: fine seasonal transitions carry useful forecast information; mathematical smoothness is not an empirical calibration objective.

Iteration70/71 TimesFM full result PASSED:10/10 cells improve versus v9 bridge. September mean923.5471→922.1419(mean per-asset-0.1360%);October690.6868→689.3561(-0.1988%). This is a retained research candidate, not yet serving.

## Iteration73 — temporal validation of TimesFM

Budget63.90/100USD,14.5h remaining,normal. Keep the exact pretrained point-forecast25% blend and all settings fixed. Score the additional Sep23–30 period against sep_late_bridge and the causal July20–21/Aug20–21 periods against historical_bridge_1h, using pre-July seasonality,clock,event tables for both historical forecasts. Require the added-period mean and BOTH historical means to improve, with most cells favorable; no historical tuning if the criterion fails. Four CPU workers per study, with at most three workers in the sequential Chronos queue, total at most11. Full-window results are not interpreted as untouched-test evidence.

Iteration73 historical result FAILS:July mean767.0085→765.8563(mean per-asset-0.1567%,5/5 wins),August1681.5828→1682.7293(+0.0547%,1/5 wins). DISCARD TimesFM from serving under the registered both-historical-means criterion. Finish the already-running additional September study for completeness. The favorable10/10 development cells and19/20 daily gains do not override this failure. Descriptive two-day block95% intervals on daily mean percentage changes were[-0.2411,-0.0670] and[-0.4844,-0.0975]; these do not adjust for model selection and cannot establish temporal transfer. Large unselected weights remain ignored.

## Iteration74 — direct pretrained quantile forecasting

Hypothesis: Chronos-Bolt-tiny's direct multi-horizon forecast may provide a useful and cheaper hourly variance predictor than autoregressive Chronos or TimesFM. Public metadata for amazon/chronos-bolt-tiny pins revision a0e552de83495b5c28c14c71c374f3e33280b340,last modified2025-11-21,Apache2.0,before all evaluation periods. Download only public weights/config/model card. Use CPU float32,existing chronos-forecasting research dependency,offline local loading,and environment-derived seed.

Keep TimesFM's144 complete-clock-hour log-variance inputs and contiguous finite suffix policy, use native median log prediction, exponentiate clipped[-3,3], and the same25% blend/ratio clipping. This matches TimesFM's point-in-log-space aggregation more closely than sampled Chronos's mean-exp aggregation; do not claim architectural causality. No quantile-tail extrapolation is needed. Gate cold<60s,warm<5s,repeatability1000 paths before official scoring. Require both full-window means and most cells, then unchanged added-period and causal historical criteria. No serving change from model metadata or forecast proxy metrics.

Iteration73 additional-period result:TimesFM mean922.5773→921.7783(mean per-asset-0.0956%,4/5 wins). Historical August rejection remains decisive under the registered criterion.

Iteration74 CPU gate PASSED:10.814s cold,0.0398/0.0373s warm uncached1000-path generation,1.152GiB peakRSS,exact repeats. Public checkpoint34,622,352 bytes. Launch full official one-hour comparison against bridge v9 with four CPU workers, while the sequential original-Chronos queue continues. No GPU use by Bolt.

## Iteration75 — principal-component Gaussian integration

Budget65.45/100USD,14.4h remaining,normal. Hypothesis: ordering Sobol coordinates by the eigenvalues of the cumulative-price Gaussian covariance can improve finite1000-path integration beyond the selected bridge/direct/random samplers. This is an exact linear representation of the same Gaussian increments on each prompt's deterministic variance clock, not a fitted return-correlation model. Use descending eigenvalues and canonical eigenvector signs; reserve coordinate0 for the unchanged path-volatility mixture as in the bridge sampler.

Verify cumulative covariance and increment identity algebra on heterogeneous clocks before scoring, plus latency at60/288 steps. Then compare all36 full-window cells against composed v9,1000 paths,float32,four CPU workers. Require both means and most cells per configuration. Any retained group must survive independent seeds11/37 and additional-period/historical checks. No per-asset sampler selection. Existing Bolt4workers and sequential Chronos at most3 leave total11.

Iteration75 algebra gate PASSED eight clocks/sizes:maximum increment covariance error1.49e-9,cumulative covariance matches the intended Brownian clock,and exact repeats.1000-path transform timing approximately0.002–0.003s for60steps and0.024–0.025s for288steps with one BLAS thread. Launch full comparison; these algebra/timing checks do not predict CRPS benefit.

Iteration74 full result PASSED:10/10 cells improve versus v9. September923.5471→922.0844(mean per-asset-0.1379%);October690.6868→689.2155(-0.2215%). Bolt is slightly stronger in aggregate than TimesFM on these dates, but this is not sufficient for selection.

## Iteration76 — transfer check for direct median forecasting

Budget66.44/100USD,14.4h remaining,normal. Evaluate the unchanged Bolt median25% blend on addedSeptember and the causal pre-July-profile historical windows, using exactly the TimesFM criterion: added mean plus BOTH historical means favorable and most cells improving. No coefficient or aggregation adjustment based on historical outcomes. Allocate three workers per study plus PCA4 and sequential Chronos at most3,total13. Candidate remains unselected pending these checks.

Iteration76 historical result FAILS:Bolt July767.0085→766.2502(mean per-asset-0.1007%,5/5wins);August1681.5828→1682.5864(+0.0487%,2/5wins). DISCARD from serving; complete addedSeptember for the record. Native median forecasts from two distinct public architectures share favorable recent behavior and historical mean reversal; do not present either as a universally better volatility estimator.

Iteration75 crypto24h full result PASSED:10/10cells improve,September4201.6155→4199.1576(mean per-asset-0.0601%),October3406.1522→3404.2744(-0.0578%). One-hour loses9/10cells,both means(+0.0071%/+0.0094%),so DISCARD that sampler change. COM September fails(+0.0092%);finish October but do not promote this group.

## Iteration77 — independent integration and temporal checks

Budget67.11/100USD,14.3h remaining,normal. Test PCA crypto24h on seeds11/37 using existing width_seed11_narrow/width_seed37_narrow controls (same v9,scale0.95). Require all four seed/window means favorable and most cells. Also test the unchanged sampler on addedSeptember against sep_late_scale95 and causal historicalJuly/August against historical_v9_crypto24h. All date means must improve before promotion. Sequential queue,four CPU workers; with remaining PCA4,Bolt3 and reserved Chronos3,the temporary maximum is14. No coefficient refitting or selection by asset.

Iteration75 completed36-cell result:COM also worsens October(+0.0025%,4/8wins),confirming rejection for that group. Crypto24h is the sole retained PCA candidate pending iteration77. Iteration76 additionalSeptember Bolt mean922.5773→921.7095(mean per-asset-0.0995%,4/5wins);historicalAugust rejection remains unchanged.

Iteration77 independent-seed gate FAILS:seed11 September improves0.0422%(5/5wins)butOctober regresses0.0025%(2/5wins;raw3404.2178→3404.3154). Therefore DISCARD PCA promotion under the registered all-four-means criterion. Seed37 and the queued extra-date checks finish transparently; do not redefine the gate from pooled gains after seeing the reversal. Lesson: the target Gaussian law is unchanged, but finite-fan CRPS rankings can still depend on the environment seed. Exact covariance preservation and default-seed10/10wins do not establish a robust improvement.

Iteration62 completed Chronos+bridge historical result FAILS:July767.0085→766.4017(mean per-asset-0.0783%,5/5wins),August1681.5828→1682.3025(+0.0163%,3/5wins). DISCARD original Chronos addition under the same historical criterion. Remaining cached interaction/full studies document its recent performance but cannot override this gate. Iteration77 seed37 also fails October(+0.0241%,2/5wins),despite September-0.0487%(5/5wins). PCA remains rejected.

## Iteration78 — nonlinear residual information at the current fast baseline

Budget68.30/100USD,14.3h remaining,normal. Hypothesis: interactions among the original six context features can predict conditional scale residuals beyond the current15-minute EWMA even though the earlier linear correction failed against an older baseline. Compare one fixed depth2 boosted-tree model (80trees,learning_rate0.05,min_leaf100,squared loss) with matched ridge on the same features/targets. No hyperparameter sweep. Fit uses snapshot prices only; CPU only. Export trees as plain JSON for NumPy inference, so sklearn is training-only.

First chronological diagnostic: profiles/clock/calendar parameters endingJuly1,training originsApril–July,August proxy holdout,one origin every3h and rotating minute offsets to avoid round-hour-only sampling. Target is log ratio of the next hour's mean absolute one-minute return to the exact v9 forecast mean absolute increment proxy, with the actual two-minute issuance lag and supplied finite context. Bound labels to the training cutoff. Clip target to[-1.5,1.5] before both fits. Compare centered residual predictions (no fitted global intercept) to zero and matched ridge. A tree must improve August proxy MSE over both before official scoring. Proxy is not CRPS.

If passed, refit the same policy throughSeptember1 with current pre-September profiles,freeze half-strength conditional correction clipped±0.25 in log scale (same conservative residual strength as earlier),and score both full windows against v9 at1000paths. Require both means/most cells,then added-period and separately refitted causal historical validation. Diagnose feature/target parity before scoring. No model chosen from historical score failures.

Iteration77 additional dates complete:PCA addedSeptember improves5/5cells(mean per-asset-0.0454%,4269.6556→4267.7677),historicalJuly/August improve10/10cells(-0.1017%/-0.0658%,3365.7251→3362.2591 and9025.5106→9019.9199). These default-seed gains coexist with BOTH independent-seedOctober reversals. The predeclared seed criterion still rejects promotion; report all evidence rather than implying temporal failure.

Iteration78 proxy gate PASSED:4,878training origins,1,240August holdout origins;centered log-scale residual MSE zero0.099300,ridge0.093759,depth2boost0.086646. Tree improves over both controls. JSON traversal reproduces sklearn centered predictions to2.50e-16 across all holdout rows. This is surrogate evidence only,not a CRPS claim. Refit the unchanged tree and matched ridge throughSeptember1 using the current pre-September profiles before official tests. Fit remains one CPU process; cached Chronos queue uses at most3workers.

## Iteration79 — official nonlinear versus matched linear correction

Refit completed on6,118pre-September origins. Its August subset is IN-SAMPLE and is explicitly labeled training_subset_not_validation;only the earlier chronological diagnostic gate supports proceeding. JSON export error4.997e-16. Exact baseline parity passed onBTC/ETH/HYPE actual contexts. Both tree/ridge produce repeatable,finite,positive1000-path outputs with correct starts;warm times0.0216–0.0249s. No sklearn is needed for inference.

Score tree and matched ridge independently against bridge v9 on both full windows at1000paths,float32,four workers each. Require both means and most cells for either model;tree must additionally beat the matched ridge to substantiate nonlinear value. Any retained model requires addedSeptember and a separately refitted causal historical check. Total with the cached Chronos queue is11workers. The strength0.5,±0.25clip,features and fitting choices stay frozen.

Iteration79 full result PASSED for both residual candidates:tree923.5471→921.8547 and690.6868→689.2966(mean per-asset-0.1732%/-0.2180%,10/10wins);ridge923.3670/690.1056(-0.0211%/-0.0933%,9/10wins). Tree also beats matched ridge in10/10cells. Retain both pending temporal tests;do not discard a simpler positive control merely because its development gains are smaller.

## Iteration80 — causal refit of the residual fitting policy

Budget72.22/100USD,14.0h remaining,normal. Evaluate both frozen residual models on addedSeptember against sep_late_bridge. Separately refit the identical fixed hyperparameters,features,target,centering and shrinkage onApril–June origins,using only labels beforeJuly1 and pre-July seasonal/clock/event artifacts. Score both resulting models on the July/August historical windows against historical_bridge_1h. Require addedSeptember and BOTH historical means plus most cells favorable for either model. Tree must retain its advantage over ridge to justify its complexity. No historical hyperparameter or correction-strength search. Two independent sequential queues,4CPU workers each at scoring;historical fitting itself uses one core. All Chronos/PCA queues have finished.

Iteration69 prefix gate PASSED on both registered real contexts:all32sample prefixes and mean-exp prefixes match exactly for3versus25decoded values,maximum difference0. Tested contexts require only2values. HYPE forecast decoding0.1935s→0.0313s;BTC25step timing includes initial warmup. This is a research optimization opportunity,not promotion evidence:Chronos remains rejected by the historical CRPS criterion. Leave its source unchanged for reproducibility of cache-dependent completed studies.

## Iteration81 — age of the latest variance observation

Budget72.55/100USD,14.0h remaining,normal. Hypothesis: conditional variance reversion should account for elapsed time since the latest usable return,including issuance lead and missing trades. Official context_end is2minutes before start for1h and3minutes for24h. EWMA normalization cancels a common age offset,while the current forward decay starts at forecast time. Test adding max(age_of_last_valid_return,0) to every forward reversion elapsed time. This is a forecast-timing hypothesis,not a claim that an empirically calibrated EWMA is a literal latent-state filter.

Preserve all observations,estimator weights,seasonality,calendar,path starting prices,mixture and sampler. Do not insert extra price shocks or interpolate missing prices. Compare v9 alone with this single timing policy across all36 full-window cells,1000paths,float32,four workers. Require both means and most cells per configuration;retained groups need added-period/causal historical checks. No combined nonlinear model before the independent effect is established. Residual validation queues reserve8workers,so maximum12.

## Iteration82 — directly measure the unselected learned-mixture interaction

The completed cached Chronos+bridge+symmetric-mixture combination improves10/10full cells(mean September920.9922,October688.7061;mean per-asset-0.2508%/-0.3069%)and4/5addedSeptember cells(mean922.0894,-0.0660%). The components each fail historicalAugust,but that does not mathematically establish the combination's historical result. Resolve this uncertainty directly rather than implying an unmeasured combined failure. This does not reverse either component's rejection or permit recent scores to override their gates.

Run the unchanged combination with pre-July tables on the sameJuly/August historicalwindows,using existing exact-input cached forecasts and cache-only mode. TwoCPU workers,zeroGPU inference,total maximum14with the residual and timing studies. Require BOTH historicalmeans andmost cells to improve against bridge v9;any retained combination additionally needs the newJuly/August date pair before selection. No coefficient,timing,blend or threshold changes. This is a distinct interaction test,not tuning the failed components against the historical losses.

Iteration80 completed:tree addedSeptember improves5/5cells(mean922.0836,-0.0653%),historicalJuly improves5/5(mean766.5914,-0.0579%)butAugust regresses(+0.0142%,1681.8958,2/5wins). Ridge addedSeptember improves3/5(mean922.4911,-0.0166%),July5/5(mean766.5141,-0.0652%)butAugust regresses(+0.0958%,1682.3433,2/5wins). DISCARD both under the unchanged historical criterion. The tree's stronger proxy and development results do not establish regime transfer.

Iteration82 direct combined test also FAILS:July improves5/5(mean766.1906,-0.1117%),August worsens(mean1686.2416,+0.1706%,2/5wins),includingXRP+0.98%. DISCARD the combined Chronos/adaptive bridge candidate on its own measured evidence. Its failure is now observed,not inferred from component failures.

## Iteration83 — supervise conditional correction with official loss slopes

Budget74.14/100USD,13.9h remaining,normal. Hypothesis: the mean-absolute-return surrogate omits features of the official price/volatility CRPS objective. Reuse the same six features,fixed depth2/80tree learner and matched ridge,but supervise with finite-difference official CRPS slopes around the unchanged baseline. This is a new target,not retuning the failed proxy correction against historical outcomes.

First collect paired1000-path,float32 official CRPS at log-width shifts-0.1,0,+0.1 on the existing April–August rotating-origin diagnostic dataset,using pre-July seasonal/clock/event tables. Each origin's three fans use the same benchmark seed. Scoring calls agent.predict.generate and agent.predict.cached_backtest(the official synth-lib worker);no metric is reimplemented. Use only complete snapshot reference paths at these exact origins,record excluded missing-reference counts,and never fill prices. No field rank is computed. Four CPU workers,one asset per worker;each asset processes its three widths sequentially.

Fit onApril–July only. Divide each official finite-difference slope by that asset's training mean baseline CRPS;target=-2*slope,clipped[-3,3]. The fixed factor2 is a gradient-step convention,not a fitted score optimum. Center both predictions over training data to isolate conditional correction,then retain the same half-strength±0.25log-scale serving clip. Compare both models by ACTUAL official CRPS onAugust origins,not by squared gradient prediction error. Require lower aggregate mean and most assets;tree additionally must beat ridge. Only a passing model earns a September-profile refit and full/recent/causal historical/independent-seed tests. All fits,means and normalizers for this first gate excludeAugust labels. Repeated use ofAugust as a development gate will be disclosed;it is not an untouched test.

Iteration83 causality amendment BEFORE any label scoring:pre-July artifacts are unavailable atApril–June origins. Even training-loss predictions must obey the campaign's prediction-time rule. Restrict the first official-label dataset toJuly–August,train onJuly,and evaluate onAugust. Keep the same rotating origins and all preregistered learner/target settings. A passing model will refit onJuly–August labels generated by this CAUSAL pre-July forecasting policy,then be tested against v9 with its current profiles;no retrospective scoring with September-fitted profiles onJuly inputs. A causal historical fitting-policy test,if reached,will first need independently fitted pre-April artifacts andApril–June official labels. No results have been observed under the abandoned April-label plan.

Iteration81 full result FAILS allgroups:September mean per-asset changes crypto24h-0.0005%,crypto1h-0.0018%,COM approximately-0.0000%;October reverses to+0.0004%,+0.0052%,+0.0004%. DISCARD observation-age adjustment without changing the reversion rate or adding price shocks. The effects are small and inconsistent;an appealing timing interpretation alone does not improve the empirical model.

Iteration83 label collection completed:2,476exact origins×3widths=7,428official1000-path scores. July training has1,236origins;August holdout has1,240. Bothnormalizers and learner fitting useJuly only. JSON traversal parity maxerror2.78e-16. ActualAugust officialCRPS evaluation oftree/ridge is running with fourCPU workers and exactpaired baseline subsets. No decision is based on gradient-prediction MSE.

Iteration83 ACTUAL official August holdout gate PASSED for both models: baseline787.3886,gradient-tree786.8555(mean per-asset-0.0876%,4/5wins),gradient-ridge787.2394(-0.0402%,4/5wins). Tree beats ridge in4/5cells(-0.0473%). XRP worsens for both; no per-asset exception is allowed. These are exact-origin complete-reference development holdout scores, not field ranks or the unseen evaluation.

## Iteration84 — refit loss-supervised correction before recent validation

Refit the identical slope target, normalization rule, tree/ridge settings and centering on all2,476July–August official labels. Every label was generated with artifacts available byJuly1; no retrospective use ofSeptember-fitted forecast parameters. Serving residual features retain the pre-July table recorded by their artifact, while the underlying forecast uses the current v9 profiles. This profile-update transfer is part of the policy being tested, not assumed equivalent to the training baseline.

Verify exported inference and1000-path serving, then score both full windows against v9, fourCPU workers per model. Require both means/most cells for each;tree must additionally beat ridge before preferring it. Any retained model needs addedSeptember, independent seeds11/37, and a truly causal historical fitting-policy audit using pre-April artifacts andApril–June official loss labels. No hyperparameter changes after seeing these outcomes.

Iteration84 refit completed on2,476official-label origins; exported inference error4.44e-16. Current-base parity and all six real-context tree/ridge1000-path checks pass, including exact repeats and positive/finite paths. Warm generation0.0216–0.0247s. Both full-window studies now run with four workers each; serving remains v9 pending their results.

## Iteration85 — another causal three-month COM refresh

Budget78.19/100USD,13.7h remaining,normal. Hypothesis: the already-selected rolling seasonal estimator benefits from a further update three days after its latest tested refresh as these markets mature. Fit the identical30-minute parent and50%-shrunk10-minute child onJune17–September16 andJuly8–October7. Add these September17/October8 cutoffs to the COM-only refresh schedule; all other v9 settings remain fixed.

Score both variant and matched v9 control on two fully affected scored-day windows:September18–19 andOctober9–10(two days endingSep20/Oct11),1000paths,float32. Require both means and most asset cells. If retained, test a separately preregisteredSeptember25 refresh(June25–September24) on the addedSeptember period. Do not select one favorable refresh date or tune lookback/binning. This tests the existing fitting policy at newer cutoffs, not a new seasonal estimator. Only after temporal confirmation would a final cutoff refit be considered. Sequential fitting/control/candidate queue uses at most4workers; the two gradient studies use8,maximum12.

Iteration84 full result: gradient-tree PASSES both means,923.5471→922.4810 and690.6868→690.1906(mean per-asset-0.1015%/-0.0819%,8/10wins),and beats ridge10/10. Gradient-ridge FAILS October(+0.0802%,0/5wins)despiteSeptember-0.0324%;DISCARD ridge. Tree remains provisional.

## Iteration86 — seed and causal-history audit of loss supervision

Freeze the retained gradient-tree. Test addedSeptember against sep_late_bridge,and independent seeds11/37 against newly scored frozen-v9 controls on both full windows. Require added mean plus all four independent-seed means and most cells favorable.

For the historical fitting-policy audit,fit fresh crypto seasonal/clock artifacts onFebruary–March endingApril1. Fit event multipliers fromOctober2025–March2026 using only the supplied calendar and snapshot,endingApril1;this supplies enough prior releases without borrowing future event fits. Collect official±0.1log-width labels onApril–June origins using those pre-April artifacts and1000paths. Recompute all six features with the pre-April table;do NOT reuse the pre-July-normalized feature columns from the origin schedule. Fit the same gradient-tree onthese labels with training-only normalizers,then evaluate July/August using the current causal pre-July baseline and the artifact's pre-April feature transform. This mirrors the recent training-policy/profile-update transfer and avoids retrospective label lookahead.

Require both historical means and most cells to improve against historical_bridge_1h. No target,tree,shrinkage or clipping changes. Run one four-worker seed/late queue and one sequential historical-fitting/label/scoring queue capped at four,alongside the COM refresh queue capped at four;maximum12. No GPU is needed.

Iteration85 refresh result FAILS:September1772.4210→1771.0679(mean per-asset-0.0880%,6/8wins),October1757.6093→1757.9295(+0.0067%,4/8wins). DISCARD the extra refresh policy;do not keep justSeptember17 or perform the conditionalSeptember25 test. The selected COM schedule remains unchanged,and this result cautions against an unmeasured final-cutoff refresh.

## Iteration87 — separate ordinary seasonality from scheduled-event windows

Hypothesis: ordinary seasonal estimates can absorb scheduled-event variance that is then added again by the selected calendar forecast. Refit only the COM seasonal profiles after excluding five-minute returns whose real-time intervals overlap an already-selected non-unit event kernel. Event types,scopes,dates and kernel windows come from the supplied calendar and the committed event_multipliers.json; do not invent dates or tune exclusions. Keep event multipliers themselves fixed to isolate this single fitting-policy change.

Apply the same exclusion to parent30-minute and child10-minute fits at all existing selected cutoffs(Sep1,Sep14,Oct1,Oct5),with unchanged three-month windows,parent shrinkage,estimator and final floor. Record excluded fractions before scoring. Params must be available by each fit cutoff. This is a hypothesis about double-counting,not an assertion that the current calibrated model is wrong.

Compare all16 full-window COM cells against v9,1000paths,float32. Require both means and most cells; any retained policy must pass the addedSeptember period before serving. Do not combine with the rejected extra-refresh schedule or refit event multipliers in this experiment. Sequential fitting/scoring queue uses at most4workers;gradient recent/history queues reserve4each,total12.

Iteration86 added-period result FAILS the raw-CRPS mean criterion:gradient-tree922.5773→922.6097,despite a favorable mean per-asset percentage change(-0.0055%)and3/5wins. These aggregations answer different questions. The declared added configuration mean is raw official CRPS; do not substitute the favorable percentage average after seeing the result. DISCARD promotion. Finish the already-registered independent-seed and causal historical audits for the record. No claim that loss supervision robustly improves v9 is warranted from the two development windows and August gate alone.

Iteration86 causal-history result also FAILS:gradient-treeJuly767.0085→766.6364(mean per-asset-0.0518%,5/5wins),August1681.5828→1684.2893(+0.0745%,3/5wins;XRP+0.68%). Pre-April fits andApril–June official-label provenance are retained. Seed11 recent means improve0.1020%/0.0822%(8/10wins),so the observed temporal failure is not explained by that alternative fan seed. Seed37 completes as registered.

Iteration87 event-window exclusion FAILS both COM means:2080.2779→2081.6011(+0.0817%,0/8wins),1783.6086→1784.0238(+0.0242%,2/8wins). Excluded return fractions span0.545%–0.792% across32asset/profile fits; overlap boundary gate passed. DISCARD this seasonal-only change with event multipliers fixed. This does not establish the result of a jointly refitted seasonal/event model.

## Iteration88 — match the supervision comparison

The proxy-versus-gradient studies differ in training dates and profile transforms as well as target. Do not attribute their performance difference solely to loss supervision. Fit the original log-mean-absolute-return proxy on the exact1,236July/1,240August complete-reference origin set used for the gradient study,with the same pre-July features,tree80/depth2/min_leaf100 settings,centering and half-strength clipping. Refit onthe same2,476July–August origins for full-window comparison. Only target construction changes within this matched comparison.

Score theJuly-trained proxy tree onAugust exact origins,and its refit onboth full current-v9 windows,1000paths,float32. Compare to the corresponding gradient-tree and v9 controls. This is primarily a controlled attribution experiment. Any standalone retained candidate still requires favorable holdout/full means and most cells,then added-period/causal-history/seed validation;do not select it solely because it beats a rejected gradient model. TwoCPU scoring workers; no GPU.

Iteration88 August matched-target result:proxy mean786.8545 versusgradient786.8555,essentially tied in raw CRPS;proxy improves all5assets versusbaseline787.3886. Againstgradient it wins3/5assets,but its mean per-asset percentage difference is+0.0074%. These differing aggregates and tiny raw difference do not support attributing the earlier August gain specifically to official-loss supervision. Full matched comparison continues.

## Iteration89 — re-estimate calendar kernels with matched seasonal backgrounds

Hypothesis: the failed event-window exclusion may require calendar multipliers estimated against its changed background. Fit two fresh COM event parameter sets using the same pre-September data,calendar,event-date pooling,shrinkage and leave-one-event-out diagnostics: one against the unmasked fine seasonal profile at its serving floor0.02,and one against the masked fine profile atfloor0.02. Keep event power1.5 fixed. The unmasked kernel refit is a necessary control because the original kernels used a different seasonal normalization.

Score both models against v9 on all16full-window COM cells,1000paths,float32. Also compare masked+refit versusunmasked+refit to isolate the exclusion policy. Require favorable raw means and most cells in BOTH windows;retained candidates need addedSeptember. Do not credit exclusion for a gain shared by the unmasked kernel refit. No iterative remasking,new event types,window tuning or fitted power change. Sequential diagnostic/fitting/scoring queue uses at most4CPUworkers;other active queues remain below14total.

## Landing — finalize v9; no new experiments

The harness issued its non-negotiable LANDING ORDER at$85.00 and4.6h elapsed. Stop new experiments. Select the previously frozen v9 policy because it completed its registered validation. Do not promote late candidates that have only development/holdout evidence, and do not perform an unmeasured cutoff refit.

Iteration86 seed37 completed: recent mean per-asset improvements0.1019%/0.0819%,8/10wins. Both additional seeds support the recent gradient-tree effect, but its added-period raw-mean and causal historical failures remain decisive.

Iteration88 completed its already-running matched comparison during finalization:matched-proxy full means922.1880/689.5235,10/10wins versusv9,8/10wins versusgradient-tree. August raw CRPS essentially ties the gradient target. Verdict:RESEARCH CANDIDATE ONLY,not nominated;additional-period,historical and seed checks were not conducted before landing. Do not label it a failed temporal model or a fully validated improvement.

Iteration89 completed:unmasked fine-floor kernel refit worsens both COM means(+0.0491%/+0.0177%,1/16wins). Masked-profile plus kernel refit also worsens both(+0.1040%/+0.0328%,2/16wins;means2081.9863/1784.1507). DISCARD both. No new jobs were launched after the landing order;previously running jobs finished while deliverables were finalized.

Final selected serving behavior remains v9:crypto24h seasonal/calendar EWMA with0.95width and ordinary Gaussian sampling;COM robust fine-seasonal refreshed EWMA,0.02floor,0.95width,0.35mixture andSobol;crypto1h15-minute fast memory,240-minute variance reversion,0.35mixture andSobol Brownian bridge. No adaptive mixture,pretrained forecast,residual learner,PCA sampler,extra refresh or event-exclusion policy is active.

Final verification uses the exact official context slice. Exact36-call reference parity passed,as did17-/1000-path contracts,positive finite paths,correct starts,exact repeats andseed sensitivity. Warm1000-path times:BTC1h0.0163s,AAPLX24h0.0560s,XAU24h0.0578s. Sparse contexts contained3713/898NaN closes. All10required serving files are tracked/present,total485085bytes;no pretrained weights are required. Default dependency graph confirmsSciPy via properscoring/synth-subnet/synth-lib.

Final canonical aggregation checks confirm1000paths,float32,matching prompt counts,and36/36improved cells. September starter→champion:crypto24h4296.6726→4201.6155,COM2316.0261→2080.2779,crypto1h960.4266→923.5471. October:3531.2042→3406.1522,2065.4112→1783.6086,741.2317→690.6868. Ratios of raw configuration means improve2.2123%,10.1790%,3.8399% and3.5413%,13.6439%,6.8190%,respectively. Final report distinguishes these from averages of per-asset percentage changes and documents the retained width's historical exception.

Landing interface repair: the harness rejected the initial nomination because its file-based loader could not resolve the sibling qmc_bridge import. Changed only module loading to a cached import resolved against modeling.py's own location. An isolated file-loader check passes, and the full serving verification again passes all 36 exact reference calls plus arbitrary/1000-path contracts, repeatability and seed sensitivity. Forecast behavior is unchanged. Refreshed artifact hashes and amend the nominated commit; no research experiment was launched.
