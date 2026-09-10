# Research journal — campaign-3

## Final reserved validation — model unchanged

The sole frozen-model final replay completes at 1000 paths, with August 8–18 supplying prior simulation history and reported reward rounds restricted to August 18–28 (exclusive end). The historical Score estimate is **19.802644078667015**, substantially below the warmed development estimate of 45.09516889117868.

- COM_EQU: reward ratio **0.028641953742380485**, cumulative rank **162/257**, final-round rank 165; 1,529 rounds, August 18 00:18–August 27 23:54 UTC.
- Crypto hourly: ratio **0.46019129972065925**, cumulative rank **42/257**, final-round rank 135; 3,435 rounds, August 18 00:03–August 27 23:58 UTC.
- Crypto daily: ratio **0.10524606889697077**, cumulative rank **47/257**, final-round rank 38; 1,034 rounds, August 18 00:31–August 27 23:58 UTC.

Verdict: nominate the model frozen at `9bc60e9adff582d9d58a223ecaff6396aec23e2e`, without post-holdout parameter or candidate changes. Source and seven serving-archive hashes match the pre-evaluation manifest. Additional offline checks pass 1/2/17/999/1000/1001 paths, equivalent timestamp units, NaNs and future exclusion, deterministic RNG behavior, invalid-input errors and artifact-free pre-weight fallback. Import and serving need no network or training runtimes.

Lesson: the much stronger development reward level does not transfer to this reserved window. The result is compatible with selection optimism and market/field changes, but does not identify their causal contributions. An improved ordinal crypto-daily rank accompanies a sharply smaller reward ratio, reinforcing the need to report both. Do not present the earlier 45.095169 development estimate as the final validation result or the future campaign Score. Unconfirmed iterations 53–54 remain excluded.

## Landing — freeze the fully confirmed model

The harness issues its landing order at 85.29/100 USD. Stop new experiments. The frozen serving model uses daily multiscale stratification, daily-crypto fan-mean centering and hourly prefix stratification, with the previously confirmed seven numerical weight archives. Its dev1 short-window Score proxy is **51.544567** and its explicitly warmed August 8–18 estimate is **45.095169**. The latter has COM_EQU ratio 0.0510693/rank 161 of 252, hourly crypto 0.9206018/rank 9 of 257, and daily crypto 0.3811840/rank 61 of 257.

Iteration 53 ends with 5,284 training and 566 diagnostic curves. Ridge 0.001 wins for both designs. Actual diagnostic CRPS ratios for the richer design are 1.0004714 versus identity and 0.9998064 versus control in crypto: reject crypto. COM_EQU ratios are 0.9990719 versus identity and 0.9997139 versus control: promising, but no development or reward confirmation exists. Do not deploy it. The simpler control is not independently promoted.

Iteration 54 completes its dense COM_EQU interaction evaluation: ratios 0.9999696 in dev1 and 0.9998832 in warmed dev2 versus multiscale alone. Joint reward confirmation has not occurred. Despite the positive CRPS result and separate weekly-shape reward gains, omit the weekly weights from the champion. This is deferred validation, not evidence that the candidate fails.

The final serving audit passes all 18 active asset/format combinations at 17 and 1000 paths, plus 45 centered-crypto comparisons over three dates and 1/17/1000 paths. Numerical archives load with pickle disabled and have recorded hashes. No research jobs remain. Run only the already reserved final scoring window, August 18–28, with ten prior simulation days and 1000 paths. Freeze before inspecting any final outcomes and do not tune afterward. The final window is indexed by scoring time; overlapping daily paths and prior-history smoothing limit statistical independence from adjacent development days.

### Iteration 52 promotion — daily crypto only

Dense mean CRPS ratios are 0.999948/0.999971 against multiscale-only daily crypto. Full reward ratios improve from 0.4607167 to **0.5121465** in dev1 and from 0.3792333 to **0.3811840** in explicitly warmed dev2; ranks remain 32/257 and 61/257. Promote unconditional fan-mean centering only for daily crypto. COM_EQU and hourly keep their prior samplers without this transformation. The improvement is much larger in dev1 than in warmed dev2; do not overstate its stability or size.

Iteration 53's score curves remain anchored to its preregistered uncentered multiscale model, not retroactively changed by this promotion. Any surviving fitted correction must later be checked in interaction with the actually deployed sampler/centering stack.

### Iteration 49 reward verdict and interface audit

Weekly lead shape improves COM_EQU full-round reward ratios to 0.0565473/0.0344222, versus 0.0564549/0.0332406 for its unchanged-sampler baseline; ranks are 111/200 of 257. It loses dev1 crypto rewards (0.3939104 versus 0.395023), despite a tiny positive mean-CRPS result, so reject its crypto branch. The equity branch requires interaction confirmation with the promoted sampler before deployment.

The serving audit identifies and fixes a genuine input-type bug: NumPy integer Unix seconds were interpreted as nanoseconds by pandas. Python integers, NumPy integers, floats, ISO strings, naive UTC timestamps and equivalent timezone-aware timestamps now yield identical paths. Reward aggregation now filters unnecessary earlier rounds before official computation and keys its cache by the cutoff. Complete official row comparisons and the exact 42.803008 baseline confirm unchanged retained results.

## Iteration 54 — equity weekly-shape/sampler interaction (preregistered)

- Hypothesis: the separately confirmed weekly lead shape remains useful with the promoted multiscale daily fan.
- Change: combine exactly full-strength weekly lead shape with the fixed two-sweep multiscale sampler, for COM_EQU only. Evaluate every daily development prompt and the entire warmed dev2 input period at 1000 paths, against saved multiscale-only scores. No new strengths or calendar harmonics are introduced.
- Result: pending.
- Verdict: require both-window mean CRPS gains and full reward improvement before adding the weekly weights to serving. Otherwise retain the simpler promoted sampler.
- Lesson: pending.

### Iterations 49 and 52 follow-up

Weekly-shape dense ratios are COM_EQU 0.999970/0.999884 and crypto 0.999998/0.999903. The dev1 crypto gain is negligible, so it must not be portrayed as a material success; the fixed candidate now receives full reward comparison against its original unchanged-sampler baseline.

Fan-mean centering fails COM_EQU (1.000039/1.000047) and hourly crypto (0.999939/1.000037); reject those branches. Daily crypto passes the 0.00001 screen gate with ratios 0.999947/0.999956. Continue only daily crypto to every-prompt confirmation, including the warmed dev2 input period. No additional centering strengths are tested.

## Iteration 53 — multi-minute information in official-score calibration (preregistered)

- Hypothesis: the fitted numerical design describes minute-return volatility but does not directly encode how returns aggregate over 5/15/30 minutes. Nine ratios across 1h/6h/24h memories may reveal conditional microstructure effects relevant to daily scoring.
- Change: rebuild the three-point 0.92/1/1.08 scale-response curves around the promoted multiscale daily model, using the same pre-field training anchors and 1000-path official scorer. Apply scaling inside the generator before the lattice, correcting the old calibration experiment's approximate post-generation scaling. Fit matched ridge Newton controls with and without nine multi-minute/minute absolute-volatility ratios. Select identity or penalty 0.001/0.01/0.1/1 solely on the chronological quadratic diagnostic; keep +/-8% bounds. Then rescore both selected fits and identity through the official scorer on the actual diagnostic contexts.
- Result: pending. This is a controlled feature ablation on a refreshed, correctly projected score-response objective, not reuse of the old score curves as if the serving model were unchanged.
- Verdict: a daily competition proceeds to development only if the richer design improves actual diagnostic CRPS by at least 0.0001 against both identity and its matched control. Require both development windows and full rewards before promotion. Do not adopt or retune the control after a failed added-feature gate.
- Lesson: pending.

### Iteration 51 verdict — reject added stale-quote variance

Only 14/480 and 36/480 screened COM_EQU prompts have a stale last close; their maximum ages are six and five minutes. Group CRPS ratios are approximately 1.000000/1.000005, failing the material two-window gate. Neither screened crypto format contains an affected prompt, so the hourly branch also fails its sample-size gate. Reject the adjustment. Missing observations alone do not justify assuming extra latent variance; this is not evidence that all full-history crypto prompts have zero age.

## Iteration 52 — unconditional fan-mean centering (preregistered)

- Hypothesis: zero-centered log returns imply a positively shifted arithmetic-price mean. Removing that finite-fan convexity shift may improve the validator's price/percentage-return objective without adding a directional predictor.
- Change: after the promoted daily multiscale or hourly prefix sampler, subtract each horizon's log-mean-exp across paths from all log paths at that horizon, then apply the same price-lattice projection. Compare this single full-strength transformation with identity at 1000 paths in both development windows. There is no extra quantile grid, seed selection, fitted drift or half-strength arm.
- Result: pending. This enforces flat unconditional fan means before lattice rounding; it does not establish a conditional martingale or claim that real prices should be driftless.
- Verdict: continue a competition only if its asset-mean CRPS ratio improves by at least 0.00001 in both windows, then require dense CRPS and full rewards. Do not salvage individual assets after the group result.
- Lesson: pending.

### Iterations 47–48 promotion and iteration 50 rejection

Promote daily two-sweep multiscale stratification and hourly prefix stratification. The combined short-window dev1 Score proxy is **49.830242**. The explicitly warmed August 8–18 Score estimate is **45.030144**, up from 42.803008: COM_EQU reward ratio **0.0510693**, rank **161/252**; hourly crypto **0.9206018**, rank **9/257**; daily crypto **0.3792333**, rank **61/257**. All use 1000 paths and every official reward round. The self-contained production implementation passes 24 exact research/serving comparisons and the context/fallback edge checks.

Iteration 50 uses 13,080 crypto training and 965 diagnostic rows. Identity loss is 1.042706146, the best matched control is 1.039308816, and the best fast-feature fit is 1.038942660. Both select 200 stages, but the added-feature advantage is only 0.000366156, below the preregistered 0.0005 gate. Reject before field scoring, without relaxing the threshold.

## Iteration 51 — uncertainty since the last observed trade (preregistered)

- Hypothesis: when a prompt's last finite close predates its timestamp, the model starts diffusion too late. Accounting for that unobserved elapsed time in the first forecast increment may improve sparse-quote assets while leaving the mandatory initial point unchanged.
- Change: around the promoted sampler, multiply only the first increment's diffusion scale by sqrt(1 + last-close-age / forecast-step-duration). Use full observed age with no new cap, fit, seed or strength grid. The rest of the path construction remains unchanged. Compare identity and this single fixed rule at 1000 paths in both development windows, recording the number of affected prompts. Do not impute any missing historical close.
- Result: pending. Assets with a close exactly at the prompt are unchanged by construction.
- Verdict: retain daily COM_EQU only if both group ratios improve by at least 0.00001 and official rewards confirm. Consider hourly crypto only if at least 20 affected prompts exist in each window and both-window CRPS and rewards improve; otherwise reject that branch as underpowered. Daily crypto is not a promotion target for this sparse-quote experiment.
- Lesson: pending.

### Iterations 48–49 confirmation progress

At every daily prompt, multiscale stratification improves against increment-only stratification with COM_EQU CRPS ratios 0.999873/0.999773 and crypto 0.999822/0.999828. Its dev1 full reward ratios are 0.0622061 for COM_EQU (rank 108/257) and 0.4607167 for crypto (rank 32/257), both above increment-only 0.0598841/0.4206283. Explicitly warmed dev2 rewards are now running.

The weekly-shape screen gives COM_EQU ratios 0.999963/0.999884 and crypto 0.999987/0.999904. All four are below one but the dev1 crypto gain is especially small. Continue only this fixed full-strength model to dense confirmation against the unchanged sampler; any later combination with stratification must receive a separate interaction check.

## Iteration 50 — sub-hour information for hourly volatility (preregistered)

- Hypothesis: the temporal encoder bins returns hourly and the fastest static memory is 30 minutes. Past 5/15/30-minute absolute scale, RMS, largest normalized move and quote coverage may contain useful immediate volatility information for the one-hour competition.
- Change: restrict fitting to crypto and the existing hourly target, around the frozen half-strength static plus temporal model. Compare identical seven-leaf Poisson residual trees with and without twelve fast features, using learning rate 0.03, minimum leaf size 150, L2 10, seed 95 and 200 stages. Select identity/50/100/200 stages only on the pre-field chronological diagnostic. Bound corrections to +/-0.25 log scale. Require the best fast design to beat the best control by at least 0.0005 before field evaluation; if it passes, compare both selected fits and identity at full strength and 1000 paths in both hourly development windows.
- Result: pending. The features use only finite positive past closes and never interpolate; missing sub-hour observations carry explicit coverage information. No auxiliary candle fields or new market series are required.
- Verdict: no field scoring after a failed controlled likelihood gate. Require both-window improvement over identity and the matched control, followed by all-round rewards, before adoption.
- Lesson: pending.

### Iteration 45 final verdict — reject both phase corrections

Despite better mean CRPS, both phase candidates lose dev1 rewards against the deployed daily crypto model's 0.395023 ratio. The aligned control gives 0.384724/0.246129, ranks 39/82; shifted training gives 0.352926/0.270314, ranks 44/78. Dev2 improves against 0.241880, but the preregistered both-window reward gate fails. Discard both corrections entirely, with no asset-specific salvage. This is another concrete counterexample to equating asset-mean CRPS improvement with emissions improvement; it also prevents an unsupported claim that shifted training clocks solve the field problem.

### Warmed replay reconstruction and day-level stability

Reconstructing all three warmed incumbent competitions from their saved official CRPS gives exactly the original ratios, ranks and retained intervals: COM_EQU 0.04673687382620357, hourly crypto 0.8953589120219363, daily crypto 0.34199446651969584. Thus candidate reward replay has an exact golden baseline rather than merely a similar aggregate. The stratified candidate is still running.

On August 8–18, increment/prefix stratification improves every asset's aggregate CRPS. Its asset-mean daily ratio improves on 9/10 COM_EQU days, 9/10 hourly-crypto days and 10/10 daily-crypto days. These are descriptive paired-day checks, not independent-sample significance tests; overlapping forecasts remain dependent. The publisher's full text confirms the finite-ensemble quantile reference; no market data was accessed online.

### Iteration 49 likelihood result

The weekly mean-preserving correction improves chronological diagnostic loss from 0.983648284 to 0.983480924, a gain of 0.000167360. Ridge 0.01 wins among the three predeclared penalties, on 23,337 training and 2,508 diagnostic rows; all fits converge. It clears the 0.0001 gate and proceeds to the single full-strength daily field screen. This internal objective is not the validator's CRPS and omits other serving-clock interactions, so field consistency is essential.

### Iteration 45 dense crypto confirmation

Across all daily crypto development prompts, the aligned-clock control gives asset-mean CRPS ratios 0.999679/0.999825 and the shifted-clock model gives 0.999829/0.999542. Both pass two-window mean consistency, but neither dominates the other. HYPE reverses in dev1 under both fits, and BTC reverses in dev2. Replay both fixed candidates against the field before choosing or discarding them; do not cherry-pick asset branches after these results. The clock-sampling attribution remains unresolved.

## Iteration 49 — smooth weekly lead shape (preregistered)

- Hypothesis: the fitted periodic level correction depends on the prompt's clock, but not how variance evolves through the coming day. A smooth mean-preserving weekly shape may supplement the existing context-derived daily profile without assuming market closures.
- Change: fit a bounded +/-0.25 log-bucket correction using integrated weekly sine/cosine harmonics at frequencies 1 and 2, pooled plus asset interactions. Center its features across the five lead buckets, and preserve the current static/linear/nonlinear lead forecast's duration-weighted mean. Exclude daily harmonics because the empirical intraday clock already models that variation. Select ridge penalty 0.01/0.1/1 only on the existing pre-field chronological likelihood diagnostic; require at least 0.0001 improvement. Test only the selected full-strength correction against identity in both daily competition groups. Keep the sampler unchanged for this one-variable screen.
- Result: pending. No sessions, weekend gates, external calendars, new seeds, or post-cutoff data are introduced.
- Verdict: abandon before field scoring if the likelihood gate fails; otherwise require both-window official CRPS and reward gains per retained competition.
- Lesson: pending.

### Iteration 48 first field result

The fixed multiscale projection improves daily CRPS against increment-only stratification in every preregistered group/window: COM_EQU ratios 0.999919/0.999821, crypto 0.999749/0.999822. It passes the 0.00001 continuation threshold. Relative to unchanged Sobol, ratios are COM_EQU 0.999795/0.999693 and crypto 0.999651/0.999679. Continue this one algorithm to every-prompt 1000-path confirmation, including ten warmup days for dev2; do not change its sweep count or order. Terminal marginal quantiles match their analytical targets at 1/17/1000 paths.

### Iteration 47 first full reward confirmation

Dev1 all-round reward replay confirms stratification in all three competitions: COM_EQU ratio 0.0598841, rank 111/257; hourly crypto 0.9719845, rank 21/257; daily crypto 0.4206283, rank 32/257. The corresponding short-window Score proxy is 48.41656, versus approximately 46.748 for the deployed model. This is still a comparative short-window result, not a steady-state estimate. The warmed dev2 incumbent replay has exactly reproduced its COM_EQU golden ratio and rank; remaining competitions and the stratified candidate are running.

The knowledge-only literature check finds the same midpoint-quantile rationale in Zamo and Naveau (2018), DOI `10.1007/s11004-017-9709-7`, Section 3.2.1. It does not validate our joint path transformation; only the official historical comparisons can do that. No external prices or post-cutoff field data were requested.

## Iteration 48 — joint daily interval stratification (preregistered)

- Hypothesis: daily scoring uses nested 5-minute, 30-minute, 3-hour and terminal increments. Marginal correction of the finest shocks leaves sampling error in larger scored blocks; a fixed multiscale projection may improve them without fitting market parameters.
- Change: use the same midpoint Student-5 quantiles as iteration 47, not a new quantile grid. Starting from the same Sobol fan, perform exactly two fine-to-coarse sweeps over blocks of 1/6/36/288 steps. Rank-remap each block sum to its analytical marginal scale and spread its adjustment across constituent increments in proportion to their forecast variances. Compare this one fixed algorithm with unchanged and increment-only fans at 1000 paths, every fourth daily prompt in both development windows. No additional strengths, sweep counts, orderings or seeds may be selected on field results.
- Result: pending. This is a new joint-path projection, not a claim that all overlapping marginals can simultaneously equal their targets. No new fitted weights or prices are used.
- Verdict: continue only if both competition groups improve in both windows against increment-only stratification by at least 0.00001 in asset-mean CRPS ratio; require dense and full-reward confirmation before deployment.
- Lesson: pending.

### Serving-contract audit before sampler promotion

The current deployed model passes 24 serving/research comparisons across BTC/HYPE/SP500/AAPLX, both formats, and 1/17/1000 paths at relative tolerance 1e-12. It also passes raw-versus-dropped NaNs, missing auxiliary columns, injected future rows, unsorted rows, constant/single-close histories, zero horizon and pre-weight fallback checks. An initial audit mismatch was caused by the audit passing an inclusive seven-day left boundary to the research helper; both actual research evaluation and production use the same open-left boundary. Correcting the audit input restores equivalence; no serving change was needed.

### Iteration 46 steady-state result

The explicitly warmed production replay for August 8–18 gives Score **42.803008**. COM_EQU reward ratio is **0.0467369**, rank **161/252**, 1,469 rounds from August 8 00:36 to August 17 23:56 UTC. Crypto-hourly is **0.8953589**, rank **9/257**, 3,400 rounds from August 8 00:08 to August 17 23:57. Daily crypto is **0.3419945**, rank **63/257**, 1,043 rounds from August 8 00:48 to August 17 23:31. All use 1000 paths and the official backtester. The active miner count and effective intervals differ from short-window selection; retain the distinction. The model is commit `97b60be` (subsequent experimental sampler changes are not deployed in that replay).

### Iteration 47 dense confirmation

Every-prompt, 1000-path CRPS confirms increment stratification daily: COM_EQU ratios 0.999876/0.999916 and crypto 0.999892/0.999880. Prefix stratification hourly gives 0.999869/0.999845 over 14,400 prompts. Confirm dev1 rewards on its existing selection interval and dev2 rewards with the explicit ten-day warmup, including an exact replay check of the warmed incumbent baseline. Do not substitute cold-window ratios for the warmed Score.

### Recovered iteration 29 normalization result

The archived one-variable normalization arms show COM_EQU ratios 1.000313/0.999745 and crypto 0.999966/0.999946. Equity is inconsistent and crypto gains are negligible. This resolves the old report's stale "ongoing" description without claiming an unperformed current-stack experiment. The actual product-of-profiles mean need not equal one.

### Iterations 45 and 47 first field results

The shifted-clock correction fails COM_EQU consistency (CRPS ratios 1.000325/0.998388); reject that branch. For crypto it improves both windows (0.999796/0.999604), but the aligned control also improves (0.999628/0.999871). Therefore the field screen does not establish uniform superiority of shifted sampling. Confirm both crypto corrections densely and retain that attribution caveat.

Marginal stratification shows a horizon-specific pattern. Full increment remapping improves daily COM_EQU 0.999876/0.999872 and crypto 0.999902/0.999857. Prefix remapping is inconsistent daily, but full prefix remapping improves hourly 0.999829/0.999758 and dominates its half-strength alternative in both windows. Continue only full increment remapping daily and full prefix remapping hourly at every prompt and 1000 paths. Zero-strength controls agree with production, and 1/17/1000-path tests recover the intended sorted quantile nodes exactly.

### Iteration 45 paired diagnostic result

There are 23,452 paired training anchors and 2,496 paired diagnostic anchors. On the same shifted diagnostic set, identity likelihood is 1.009628, the best aligned-clock control is 1.008114, and the shifted-clock fit is 1.007578. The shifted advantage of 0.000536 narrowly clears the preregistered 0.0005 gate; 200 stages is selected. Export the already-selected aligned control as well and compare both full-strength corrections against identity through official development CRPS. This distinguishes a sampling-policy gain from the generic effect of adding another residual model.

## Iteration 47 — rank-based marginal stratification (preregistered)

- Hypothesis: even the Sobol fan retains finite-sample marginal error. Replacing each sampled marginal with evenly spaced quantiles of the intended Student distribution may reduce this error without fitting another predictor.
- Change: preserve sample ranks and all current scale/shape forecasts. Separately remap standardized single-step shocks, or cumulative log-price deviations using their known cumulative diffusion scale, onto midpoint quantiles of the five-degree Student family. Test half/full remapping strengths against exact zero-strength control at 1000 paths, both development windows and all three competitions. The ordering comes only from our own simulated samples, never from future prices or other miners.
- Result: pending official CRPS. This changes the finite fan's joint behavior as well as targeted marginals; it is not assumed to preserve every scored increment distribution exactly.
- Verdict: require both-window improvement and official rewards before promotion. Do not optimize random seeds or add further quantile grids after seeing field results.
- Lesson: pending.

### Iteration 44 full equity reward confirmation

The nonlinear lead extension improves COM_EQU full-round reward ratios from 0.048053/0.023665 to **0.056455/0.033241**, with cumulative ranks **111/200 of 257** and final-round ranks 107/193. Promote it only for commodities/equities. Daily crypto keeps the linear lead correction and hourly remains unchanged. The extra encoder is small and exported to NumPy; it does not require Torch during serving.

## Iteration 46 — explicit warmup audit (preregistered)

- Hypothesis: a clean steady-state reward estimate may differ materially from the short-window development replays used for comparative selection.
- Change: the pinned `_trim_warmup` is explicitly a no-op if trimming would leave no rounds. Because daily smoothing is ten days, the ten-day development runs are not a guaranteed steady-state sample, despite using all available rounds. Run the current frozen production model through `agent/predict.py`'s official backtest wrapper with ten extra days of simulations before dev2, then report only August 8–18. All simulated starts are after the July 26 weight-availability date, all data are pre-cutoff, and the final window remains untouched. Limit concurrent per-asset jobs to four to stay within available disk space for transient 1000-path JSON files.
- Result: pending. Earlier development ratios remain real comparable selection results, but must not be presented as clean steady-state post-cutoff Score estimates.
- Verdict: this is an evaluation audit, not a model change or a new source of market data. Final nomination must include an explicitly warmed replay.
- Lesson: pending. The code's ten-/five-day windows override the constitution's imprecise 24-hour smoothing description.

### Iteration 44 dense gate

At all daily prompts, full nonlinear lead correction gives COM_EQU CRPS ratios 0.999388/0.998831. The half-strength crypto extension is 1.000004/0.999689: its effectively zero dev1 screen gain changes sign with denser sampling. Reject the crypto extension under the consistency gate rather than rationalize the tiny reversal. Only COM_EQU proceeds to full reward confirmation; hourly and daily crypto remain unchanged.

## Iteration 45 — randomized training-clock coverage (preregistered)

- Hypothesis: fitting only at 00:00, 03:00, …, 21:00 UTC leaves a clock-distribution mismatch. A bounded timestamp-only audit of archived July 28–30 prompts confirms that BTC, SP500 and GOOGLX are requested at all 24 hours and many distinct minutes.
- Change: pair every existing three-hour training anchor with a deterministic timestamp-hashed offset of 0–179 minutes inside its stratum. Recompute both control and shifted contexts/24-hour labels identically from the snapshot; retain a pair only if both labels and contexts are usable and both targets finish by July 26. Fit the same seven-leaf, minimum-150-row, L2-10, learning-rate-0.03 daily residual trees around the frozen joint stack. Select 0/50/100/200 stages on one common shifted July 1–25 diagnostic set. No extra label days or larger training sample is introduced.
- Result: pending. Require the shifted fit to improve likelihood over both identity and the best aligned-anchor control by at least 0.0005 before field evaluation.
- Verdict: reject the sampling claim if the controlled gate fails; broader clock coverage alone is not evidence of better forecasts.
- Lesson: pending.

### Joint serving audit and iteration 44 screen

The first joint serving comparison found a one-tick discrepancy. Root cause: converting the student's NumPy float64 multiplier to a Python float allowed NumPy's weak-scalar promotion to keep an intermediate scale at float32. Preserve the strong float64 scalar instead. Production now matches the official research candidate at rtol=1e-12 for BTC, HYPE, SP500 and GOOGLX, at 17/1000 paths and both formats. Warm daily 1000-path calls take about 0.095–0.103 seconds; hourly 0.029 seconds in this audit. The earlier 36 constant/sparse/single-observation shape and positivity cases also passed before joint integration and will be repeated on the final model.

The nonlinear lead screen gives COM_EQU CRPS ratios 0.999583/0.999279 at half strength and 0.999397/0.998908 at full strength. Crypto full strength fails consistency (1.000120/0.999507); half strength is 0.999997/0.999668, effectively flat in dev1. Confirm full strength for COM_EQU and half strength for crypto at every prompt; no promotion follows from that negligible crypto screening gain alone.

### Iteration 44 chronological selection

The nonlinear lead encoder improves its diagnostic likelihood from 0.988871 to 0.982625, with epoch 42 selected within the fixed 120-epoch budget. It now proceeds to the preregistered half/full-strength official development screen around the confirmed joint stack. No second seed or epoch extension is introduced.

### Iteration 41 full reward confirmation and promotion

The selected daily combination passes every official reward round at 1000 paths. COM_EQU reward/best-field ratios reach **0.048053/0.023665**, with cumulative ranks **111/209 of 257** and final-round ranks 128/193. Daily crypto reaches **0.395023/0.241880**, with cumulative ranks **33/86 of 257** and final-round ranks 81/45. Both improve the promoted temporal-lead model in both windows. Promote the numerical student, daily GCD projection, and half-periodic COM_EQU correction together with the already confirmed temporal lead. Production evaluates only the student's daily tree head, reuses temporal inputs/embedding, and needs no Chronos, Torch or sklearn inference. Hourly remains unchanged. Combined with the hourly incumbent, development Score estimates are about 46.468/41.063; these are not held-out or post-cutoff scores.

### Iterations 42–43 rejection

Reject Newton calibration after the preregistered development check: relative to temporal lead, daily CRPS ratios are COM_EQU 0.998421/1.000013 and crypto 1.000146/1.002063. The small actual pre-field diagnostic improvement does not transfer consistently. Do not rescue it by choosing another penalty from field outcomes.

Reject the quote-activity extension before field scoring. The best activity likelihood is 1.007615 versus 1.007667 for the same model without activity: improvement 0.000053, below the preregistered 0.0001 attribution threshold. Both improve the uncorrected 1.008593 diagnostic, so most of the apparent gain comes from another residual tree rather than the added activity features.

## Iteration 44 — nonlinear temporal lead-shape fine-tuning (preregistered)

- Hypothesis: the frozen linear lead residual may leave useful sequence information inaccessible; a small nonlinear shape-specific encoder can improve the relative early/late volatility profile without changing its mean level.
- Change: initialize a separate copy of the existing temporal convolutional encoder and 48-unit hidden layer, with a zero-initialized five-bucket output. Fit a bounded +/-0.5 log-shape residual around the promoted static-plus-linear lead forecast, preserving its duration-weighted mean. Use seed 73, AdamW learning rate 0.0002, weight decay 0.1, 120 epochs, batch size 512, residual penalty 0.05 and adjacent-residual penalty 0.02. Select the best existing chronological diagnostic epoch, including identity at epoch -1. If nonzero wins, test half/full strengths around the selected daily joint stack, keeping its student, periodic and lattice components fixed.
- Result: pending. This is the only GPU job; GPU allocation is capped at 35%.
- Verdict: require both-window official gains and full rewards before promotion. Do not add seeds or extend epochs after a failed field result.
- Lesson: pending.

### Iterations 40–42 confirmation progress

Across all 6,240 daily prompts, the student/teacher asset-mean CRPS ratios are 1.000053/1.000008. Median measured generation time is 0.0645 seconds for the numerical student versus 0.0750 for the pretrained teacher; p95 is 0.1132 versus 0.1203 seconds under concurrent load. Thus artifact savings are much larger than this particular measured latency improvement; do not claim an order-of-magnitude speedup.

All-prompt interaction CRPS supports the complete combination. Relative to promoted temporal lead, student+lead+projection gives crypto ratios 0.999437/0.998863. Adding half-periodic correction for COM_EQU gives 0.998236/0.997585. Every-12th-round official reward screens support the same choice: crypto reward ratios 0.384824/0.249229 and COM_EQU 0.050905/0.019840, versus lead controls 0.255910/0.172249 and 0.017727/0.012669 on those exact thinned rounds. These are selection diagnostics, not final full-round numbers. The selected joint candidate now receives all-round confirmation.

The local Newton fit uses 5,284 training and 566 diagnostic contexts, with 1000-path official score differences. Penalty 0.001 wins the quadratic diagnostic. Independent actual official rescoring of the diagnostic contexts gives weighted CRPS ratios crypto 0.999991 and COM_EQU 0.999040. It passes the preregistered strict two-group gate, although the crypto gain is negligible. Continue only the selected full-strength correction to the development screen; do not search the rejected penalties on field outcomes.

## Iteration 43 — close-only quote-activity residual (preregistered)

- Hypothesis: missing-trade density, repeated quotes, quote-grid coarseness and short-lag sign behavior contain information not represented by the existing volatility moments or hourly temporal encoder. This remains usable when optional OHLCV columns are entirely missing.
- Change: compute eight past-only activity statistics over 1h/6h/24h/7d plus relative GCD-grid size and last-trade age. Fit a small Poisson-loss daily scale residual around the frozen static/temporal/student forecast, using either the current numerical design alone or that design plus activity. Fix seven leaves, minimum leaf size 150, learning rate 0.03 and L2 penalty 10. Choose zero/50/100/200 stages only on the pre-field chronological likelihood. Require the activity design to beat the best no-activity control by at least 0.0001 before field testing.
- Result: pending. No interpolation, assumed sessions, extra market series, external prices or field miner paths are used.
- Verdict: reject the extra features if they fail the controlled chronological gate; a lower fitting loss is not a CRPS result.
- Lesson: pending.

### Iteration 34 full reward result

The independent daily refinements pass full replay. GCD projection improves reward ratios from 0.218222/0.126951 to 0.226401/0.133456 for crypto, and from 0.014075/0.009365 to 0.014211/0.009528 for COM_EQU. Adding half-periodic correction with projection reaches COM_EQU 0.024937/0.011444, ranks 126/214 of 257. These controls predate the newly promoted temporal lead correction; retain them for the explicitly preregistered joint experiment rather than assume their gains add to the new model.

## Iteration 42 — local official-score Newton calibration (preregistered)

- Hypothesis: earlier piecewise-linear score-curve fits could not resolve smooth local improvements near a grid knot. A regularized quadratic approximation from official scores may provide a more informative scale correction.
- Change: around the promoted temporal-lead daily fan, compute official 1000-path CRPS at relative scales 0.92/1/1.08 on the existing pre-field curve contexts. Fit a linear correction from static inputs, temporal summaries, smooth clock features and the frozen temporal embedding. Use finite-difference gradients/curvatures, floor negative or tiny curvatures at a training-only constant for stability, and bound relative scale changes to +/-8%. Select zero or ridge penalties 0.001/0.01/0.1/1 only on the existing chronological diagnostic. Asset weighting uses validator coefficients and constant training-set competition normalizers, never outcome-dependent prompt-score normalization.
- Result: pending. Quadratic losses are explicitly surrogate diagnostics, not official CRPS evaluations. A selected nonzero candidate must be rescored with the official scorer on the diagnostic contexts before development field testing.
- Verdict: require lower actual diagnostic CRPS in both daily competitions; otherwise reject before field scoring. These curves surround fitted parameters and are meta-training data, not causal historical forecast claims.
- Lesson: pending.

### Iteration 40 field fidelity

The numerical student preserves the teacher's daily gains at 1000 paths. Student CRPS ratios are COM_EQU 0.998859/0.997996 and crypto 0.999453/0.999113, versus teacher 0.998860/0.997989 and 0.999325/0.999171. The student artifact is about 1.7 MiB versus the pretrained cache's 183 MiB, and it does not invoke pretrained inference. Proceed to dense confirmation rather than declare equivalence from a sparse mean alone.

## Iteration 41 — daily component interactions (preregistered)

- Hypothesis: a tree correction to overall scale and a temporal correction to relative lead shape are complementary; the separate periodic level correction may instead duplicate clock information in the tree.
- Change: use the distilled tree, promoted temporal lead and GCD projection. Evaluate every daily development prompt at 1000 paths for the student alone, lead plus projection, and student plus lead plus projection. For COM_EQU only, also test adding the already-screened half-periodic correction to each lead-containing combination. No hourly changes, new fitting labels or final-window inspection.
- Result: pending dense official CRPS and reward replay.
- Verdict: retain components only if their incremental joint effect survives both regimes. Screen rewards with thinned rounds to avoid replaying clearly inferior combinations, then confirm the selected joint model at every official round before promotion.
- Lesson: pending. Individual improvements are not assumed additive.

### Iteration 38 full reward confirmation and promotion

At all official reward rounds and 1000 paths, temporal lead-shape correction raises COM_EQU reward/best-field ratios from 0.014075/0.009365 to **0.016611/0.015254**, with cumulative ranks 129/211 of 257. Daily crypto improves from 0.218222/0.126951 to **0.262694/0.166763**, with ranks 57/104. Final-round ranks are COM_EQU 130/210 and crypto 93/68. Promote this correction in the self-contained serving model, reusing the temporal embedding so it does not run the encoder twice. The hourly forecast remains unchanged. This is still development validation, not the reserved final or post-cutoff campaign Score.

### Iterations 39–40 results

Reject finite Student-scale persistence. Six-hour correlation gives COM_EQU CRPS ratios 0.999783/1.000271 and crypto 0.999931/1.000272; 24-hour correlation gives 0.999932/1.000067 and 0.999932/1.000134. Tiny gains in dev1 reverse in dev2. The constant path-scale model remains the simpler supported choice.

Distillation passes its preregistered fidelity gate: the selected 400-stage student has chronological daily log-correction RMSE 0.012203, or 18.65% of teacher standard deviation, versus 0.017310/0.013476 at 100/200 stages. Batch teacher traversal matches single-row traversal in numerical checks. The student now receives official daily CRPS evaluation; no field outcomes selected its stage count.

## Iteration 40 — distill the daily tree correction (preregistered)

- Hypothesis: the promising pretrained-feature tree can be approximated by a small student that uses only existing numerical context features, avoiding the frozen foundation model at serving time.
- Change: use only our own fitted teacher's bounded log-scale corrections as distillation targets, never another miner's predictions. Fit 15-leaf histogram regressors to static inputs, temporal summaries, smooth clock features and the frozen 48-dimensional temporal embedding. Fix learning rate 0.05, minimum leaf size 64 and L2 penalty 5. Select 100/200/400 stages using chronological daily teacher-prediction RMSE alone; no field-dependent tree architecture selection. Require held-out daily RMSE below 0.02 log units and below half of the teacher-output standard deviation before field confirmation.
- Result: pending. Fitting and teacher diagnostics use only the existing pre-July-26 dataset.
- Verdict: retain only if the student preserves both-window official daily gains and meaningfully reduces latency/artifact burden; teacher fit fidelity alone is insufficient.
- Lesson: pending. This is regression to our own fitted teacher, not imitation of any field miner.

### Iteration 38 dense confirmation

Every-prompt evaluation confirms full-strength temporal lead gains at 1000 paths: asset-mean relative CRPS is COM_EQU 0.999108/0.998654 and daily crypto 0.999849/0.999229. Official reward replay is now running. The baseline rows are reused from the identical-clock, identical-path-count dense stack run, matched on window/asset/timestamp; they are not re-estimated with a different random fan.

The checkpoint smoke test computes one official 17-path BTC hourly score, then resumes that exact task without recomputation. Persistent-scale controls preserve all first-hour shocks exactly and the infinite-persistence option equals the baseline for 1, 17 and 1000 requested paths.

### Iteration 37 official result and confirmation gate

The completed repeat finds consistent daily improvements but hourly regressions. Half/full tree corrections give COM_EQU CRPS ratios 0.999331/0.998725 and 0.998860/0.997989; daily crypto 0.999576/0.999293 and 0.999325/0.999171. Hourly ratios are 1.000779/1.000708 and 1.002167/1.002146, respectively. Reject hourly trees. Full daily strength dominates half strength in both windows and proceeds to every-prompt, 1000-path confirmation. The selected tree requires frozen Chronos features at inference, so offline artifact size and latency must be accounted for before any promotion.

## Iteration 39 — persistent Student scale process (preregistered)

- Hypothesis: a single shared Student scale for an entire 24-hour path overstates the persistence of uncertainty; earlier IID-tail and lognormal-volatility experiments also changed the one-step marginal family, so they did not isolate this question.
- Change: retain the exact baseline Sobol Gaussian increments and five-degree Student scale marginal. Evolve the scale's Gaussian-copula latent once per forecast hour, with exponential correlation times of 6 or 24 hours, using a separately scrambled Sobol stream. Every step retains the same marginal scale distribution; only scale dependence across hours changes. Keep static/temporal forecasts, clock, lead shape and field windows fixed.
- Result: pending 1000-path daily official CRPS on both development windows.
- Verdict: require consistent gains; otherwise retain the simpler shared path scale. No claim that matching one-step marginals guarantees matching longer-horizon distributions.
- Lesson: pending.

### Iteration 38 first official result

The frozen temporal lead residual passes the initial two-window daily CRPS gate. Half-strength ratios are COM_EQU 0.999367/0.999117 and crypto 0.999815/0.999482; full strength improves them to 0.999007/0.998602 and 0.999761/0.999171. Continue full-strength confirmation at every daily prompt with 1000 paths. These small improvements are materially weaker than the likelihood diagnostic suggested and still require reward replay.

### Iterations 34 and 38 progress

At every daily prompt and 1000 paths, GCD projection again improves asset-mean relative CRPS: COM_EQU 0.999963/0.999903 and crypto 0.999886/0.999813. Commodity/equity half-periodic correction gives 0.999003/0.998568; combining it with projection gives 0.998964/0.998466. The interaction preserves the small gains. Full official reward replay is running; these are not emissions estimates.

The frozen temporal lead correction improves chronological likelihood from 0.997370 to 0.988871. All three penalties improve the baseline, and the preregistered selector chooses 0.01. Training/diagnostic rows are 23,337/2,508, with labels ending by July 26. Proceed to the half/full official daily screen, rather than promoting from likelihood alone. Extracting the existing temporal embedding preserves the previous residual outputs exactly in a numerical audit.

### Evaluation durability

Future research screens checkpoint completed asset/format/window tasks atomically and support `--resume`. Checkpoint identities include task settings plus the research code and local model-weight fingerprints, preventing accidental reuse after a model change. Earlier running experiments keep their existing execution and are not restarted merely to acquire this optimization.

## Iteration 38 — frozen temporal lead-shape residual (preregistered)

- Hypothesis: the sequence encoder that improves overall scale also contains information about the relative volatility of early versus late daily forecast buckets, which the existing static lead network cannot use.
- Change: freeze the current temporal encoder and lead network. Fit a small regularized linear correction from the encoder's 48 hidden features and 30 standardized static inputs to the five lead buckets. Bound corrections to +/-0.5 log units and preserve the duration-weighted mean of the incumbent bucket forecast, so training changes shape rather than level. Select zero correction or one of L2 penalties 0.01/0.1/1 solely on the existing July 1–25 chronological diagnostic, with no new labels or field dates. If nonzero wins, compare half/full correction strengths on both daily development windows through the official scorer.
- Result: pending.
- Verdict: reject unless chronological likelihood and both development windows support the shape correction; confirm emissions before serving it.
- Lesson: pending. A stronger scale encoder need not improve within-path timing, so zero correction is an explicit candidate.

### Campaign resumption — September 9, 2026

The workspace resumes from commit `f76e842`, not from a new starter. `uv sync` succeeds; the snapshot cutoff remains August 29 and no external prices or post-cutoff field data are used. The interrupted iteration-37 screen has no completed result artifact and no surviving process, so repeat its preregistered half/full tree comparison at 1000 paths with one GPU worker. This avoids concurrent GPU model copies. Preserve the incumbent until official results support a change.

### Iteration 34 dense interaction confirmation (preregistered)

- Hypothesis: the small, consistent daily lattice gains and commodity/equity half-strength periodic gains survive every-prompt evaluation and their combination.
- Change: evaluate `stacked` and `stacktick:gcd` for both daily competitions; additionally evaluate `stackperiodic:0.5` and `stacktick:gcd:0.5` for commodities/equities. Use 1000 paths, all development prompts, unchanged context, weights and random draws. Reuse the existing two development windows; do not inspect the final window during selection.
- Result: pending official CRPS and full reward replay.
- Verdict: require consistent development reward gains before promotion. Reject a combination if its interaction erases the constituent gains.
- Lesson: pending; small CRPS differences must not be interpreted as emissions without the official aggregation.

### Iteration 37 pre-field selection

Shallow trees improve the chronological likelihood from 1.007904 to 1.006335 with static inputs, 1.004610 with temporal summaries/harmonics, and 1.004544 after adding pretrained summaries. The preregistered selector chooses the pretrained-feature model at 200 stages. Most of the improvement clearly comes from the non-pretrained temporal/clock summaries; the marginal pretrained gain is only 0.000066. The selected model now proceeds to half/full-strength official CRPS, without field-based tree hyperparameter selection.

### Hourly projection serving audit correction

The first ad-hoc comparison accidentally supplied an inclusive seven-day-left-endpoint frame to the research feature helper, whereas production and all field evaluations use a strict left boundary. The extra old observation changes long-memory neural features and breaks exact path equality. Correct the audit frame, not the already-correct field results or serving filter. This is distinct from the earlier temporal-cache boundary mismatch.

## Iteration 37 — regularized tree residuals (preregistered)

- Hypothesis: shallow boosted trees may capture robust threshold interactions that the small neural forecasters miss, while a nonlinear calibration may recover information from the otherwise rejected pretrained features.
- Change: fit Poisson-loss histogram boosting to future absolute-return scale divided by the frozen static/temporal forecast. Use seven-leaf trees, at least 150 rows per leaf, learning rate 0.03 and L2 penalty 10. Compare static-only, static plus temporal-window summaries/smooth clock harmonics, and the same features plus frozen Chronos summaries. The chronology and labels stay fixed. Select a common 0/50/100/200-stage checkpoint using only pre-field likelihood; zero stages means no correction. Bound corrections to +/-0.5 log units.
- Result: pending pre-development selection, then half/full-strength official field screens only for a selected nonzero model.
- Verdict: no field-based choice among individual tree hyperparameters or pretrained variants. Require consistent field gains before adoption.
- Lesson: pending. Poisson regression is used to estimate a positive conditional mean scale, not because returns are Poisson counts.

### Iteration 32 full reward confirmation and promotion

At every reward round and 1000 paths, GCD lattice projection raises hourly reward/best-field ratios from 0.869436/0.872234 to **0.950966/0.966341**. Cumulative ranks are 25/9 out of 257; final-round ranks are 55/1. Promote hourly projection only, while daily projection remains under confirmation. This is a roughly 0.05–0.09% asset-mean CRPS improvement with a much larger emissions effect, not evidence that CRPS and rewards are interchangeable.

### Iteration 36 result

Frozen Chronos-Bolt forecasts were generated for all 25,965 exact-context training/diagnostic rows. The existing stack's mean chronological likelihood is 1.007904. Neither intercept-only calibration (best tested 1.008133) nor pretrained-feature calibration (best tested 1.008303) improves it. Both selectors correctly retain zero coefficients. Reject before redundant field scoring. This bounded failure does not establish that all foundation models or fine-tuning would fail; it says this pretrained volatility representation adds no validated benefit with the tested small calibration.

### Iterations 34–35 screen results

The modern-stack Student alternatives fail across regimes: df4 daily CRPS ratios are COM_EQU 1.002025/0.997312 and crypto 1.000855/0.998581; df3/df8 reverse gains even more strongly. Keep df5. Daily GCD projection gives small consistent gains (COM_EQU 0.999963/0.999905; crypto 0.999888/0.999812). Half periodic correction improves COM_EQU 0.999097/0.998738 but not crypto consistently (1.000638/0.999577); continue only commodities/equities, with a joint lattice check.

Temporal seeds 41/109 improve their internal likelihood diagnostic to 1.006364/1.006247 (best epochs 27/31), versus original 1.006739. Nevertheless, the three-seed ensemble fails the official field screen: full replacement gives COM_EQU 0.999551/1.000542, hourly 1.000175/0.999423 and daily crypto 1.000089/1.000571. Reject the ensemble and do not cherry-pick either extra seed from field results. Better internal likelihood does not guarantee better field CRPS.

### Chronos setup audit

The installed Torch attempts an eager Triton matrix kernel but the container has no C compiler. Disable only those optional Triton overrides for this research pipeline, retaining standard Torch CUDA kernels. Also explicitly convert the pretrained hourly forecast from the six-hour normalization used by temporal inputs to the one-hour reference used by hourly labels. No pretrained scores had been produced before these setup corrections.

## Iteration 36 — pretrained volatility representation (preregistered)

- Hypothesis: a generic pretrained temporal model may supply complementary volatility information not captured by the small supervised CNN.
- Change: use Amazon Chronos-Bolt-small, pinned to its initial November 13, 2024 Hugging Face commit `0e96a1377adca89eebae3bbde4f859dc3b9f4b46`. Only model weights/configuration are downloaded, never prices. Feed the exact-context 168-hour log absolute-return sequence, preserving unobserved bins as NaN, and forecast 24 hourly values. Freeze pretrained weights. Fit bounded, regularized linear calibration over forecast median, quantile-mean and spread relative to the existing temporal/static baseline; compare against a same-data intercept-only control. Select only on the pre-development chronological diagnostic, with the established overlap caveat.
- Result: pending. Dependency is pinned to chronos-forecasting 1.5.3, and serving would require local committed pretrained weights if selected.
- Verdict: reject before field evaluation unless the extra representation improves likelihood over both the baseline and calibration-only control. Any retained candidate must then pass 1000-path official CRPS, reward and offline latency checks.
- Lesson: pending. Generic pretrained uncertainty is not automatically a calibrated price distribution; it is used only as a candidate volatility feature.

## Iteration 35 — exact-context temporal seed ensemble (preregistered)

- Hypothesis: averaging independently initialized temporal residuals may reduce finite-training noise that survived the failed static-network ensemble.
- Change: fit two extra seeds, 41 and 109, with exactly the current exact-context cache, 100-epoch limit, loss, regularization and chronological selector. Average log corrections with the original seed 73; test halfway and full replacement of the single temporal residual. No new labels, architecture, random path seed or clock parameters.
- Result: pending pre-field diagnostics followed by official 1000-path development CRPS.
- Verdict: require consistent gains and official reward confirmation. A worse ensemble is rejected rather than selectively choosing whichever extra seed looks best on field data.
- Lesson: pending. Internal timestamp-split label overlap remains the documented likelihood-diagnostic limitation.

## Iteration 34 — distribution and periodic interactions (preregistered)

- Hypothesis: the stronger temporal/Sobol stack can change which finite-fan tail shape or discretization is calibrated, and a previously weak periodic residual may complement rather than duplicate the new temporal predictor.
- Change: hold the stack fixed and separately test daily GCD quote projection; Sobol Student degrees 3, 4 and 8 at unchanged RMS; and the existing periodic residual at half/full strengths. Preserve draws, train cutoffs and all clock/lead settings. No new field dates or miner paths are examined.
- Result: pending 1000-path official CRPS on both development windows; promising daily refresh also receives denser dev2 confirmation with its old-data control.
- Verdict: demand consistent gains before reward replay. Do not combine successful individual modifications without checking their interaction.
- Lesson: pending. This is a controlled interaction check, not permission for unbounded tail grids.

### Resumption audit and completed calibration extension

The promoted daily stack passes production/research agreement at rtol=1e-12 for BTC, HYPE, SP500 and GOOGLX in both formats, including 17/1000 paths, NaNs and future-row invariance. Representative latency is 0.028–0.046 seconds hourly and 0.075–0.083 daily. The 600-epoch calibration extension again fails transfer: half-strength CRPS ratios dev1/dev2 are COM_EQU 0.998685/1.001499, hourly 0.999887/0.999928 and daily crypto 0.999719/1.002424. Full strength worsens every dev2 competition. Final rejection; no more epoch extensions.

### Iterations 32–33 first results

At every hourly prompt and 1000 paths, lattice projection improves asset-mean relative CRPS to 0.999492/0.999145. Minimum-difference and GCD candidates have identical aggregate scores; the safer GCD candidate proceeds to full official reward confirmation. The refreshed static model improves dev2 daily CRPS slightly over the old-data control: full strength gives COM_EQU 0.998713 versus 0.999199 and crypto 0.999012 versus 0.999491, relative to the stack. Hourly regresses to 1.000468. Continue only daily refresh confirmation; these small one-window gains do not yet justify promotion.

### Full daily refinement confirmation and promotion

The full-temporal/Sobol stack with a 30-minute, 0.75-strength crypto clock reaches daily-crypto reward ratios 0.218222/0.126951, ranks 62/163 out of 257. A 15-minute, full-strength COM_EQU clock reaches 0.014075/0.009365, ranks 129/216. Both improve the earlier lead-only daily model in both windows at 1000 paths and all reward rounds. The half-temporal crypto alternative gives 0.249621/0.099636: a marginally higher two-window mean, but a substantially weaker minimum-window result. Prefer the common full temporal strength for robustness and fewer competition-specific strength parameters; do not claim it maximizes the two-window arithmetic mean. Promote the full stack for both daily competitions. Together with the confirmed hourly model, development Score estimates are approximately 36.724 and 33.618.

## Iteration 33 — chronological checkpoint refresh (preregistered)

- Hypothesis: additional completed labels from late July/early August may improve calibration on newer venues more than context-only bias correction did.
- Change: append snapshot-only training rows with every target ending before August 7, then repeat the already tested fixed 40-epoch, 60-day-recency fine-tune from the original static network. Compare against both the unchanged strong stack and the old-data 40-epoch fine-tune, at half/full strengths. Keep temporal weights and path construction fixed. Activate either fine-tune no earlier than August 7; consequently dev1 is unchanged by construction and only dev2 measures refresh effects. No final-window prices or labels enter this experiment.
- Result: pending. This is a one-window causal refresh diagnostic, not two independent confirmations. A successful policy would require a separately frozen rolling-origin validation before final nomination.
- Verdict: require improvement over the old-data control, not just over the unfine-tuned baseline; otherwise do not attribute any gain to fresher labels.
- Lesson: pending.

## Iteration 32 — price-lattice projection on the hourly finalist (preregistered)

- Hypothesis: projecting continuous fans onto a price grid inferred from observed quotes may improve calibration at the scored price precision. The earlier quantization experiment used a weaker hourly model and did not receive final reward confirmation.
- Change: preserve the exact-temporal/Sobol hourly forecast and compare two past-context grid estimators: minimum positive close difference, and a greatest-common-divisor lattice after controlled decimal conversion. Project only generated prices, never repair or round historical market data. Do not choose a new random seed or change volatility.
- Result: pending dense 1000-path official CRPS and rewards in both development windows.
- Verdict: require consistent gains and verify inferred-grid safety; otherwise retain continuous paths.
- Lesson: pending. A minimum observed price change is not automatically the exchange tick size, so the estimators are explicit hypotheses.

### Iteration 31 result

Context-only adaptive calibration does not pass the consistency gate. Quarter-strength gives COM_EQU CRPS ratios 1.00027/0.99849, hourly 1.00064/1.00012, and daily crypto 1.00002/0.99932; half strength increases regressions. Reject. The completed-volatility moment ratio is not a reliable replacement for the fixed conditional forecast.

### Fine-grid calibration development result

The 300-epoch fine-grid correction still fails across regimes: half strength gives COM_EQU 0.99864/1.00152, hourly 1.00003/0.99980, daily crypto 0.99968/1.00222. Full strength is worse in dev2. Reject this checkpoint. The independently preregistered 600-epoch run selects epoch 388 (pre-field objective 0.799594), with its development screen pending; no further extensions are planned.

### Fine-grid convergence check (before development inspection)

The 1000-path fine-grid fit improves the pre-field objective from 0.800481 to 0.799637, with its best epoch at the fixed limit of 299. Before inspecting its new development results, preregister one extension to 600 epochs with all other settings unchanged, preserving the 300-epoch artifact. If the selected epoch remains below 300, skip duplicate development scoring. No further epoch extensions are planned for this experiment.

### Updated serving audit

The promoted hourly implementation imports no Torch and remains self-contained. Production/research paths match to relative tolerance 1e-12 after aligning floating-point multiplication order; an initial approximately 1e-11 discrepancy was numerical rather than a model difference. Representative 1000-path calls take 0.028–0.047 seconds hourly and 0.059–0.062 seconds daily. NaN-contaminated contexts, 17 requested paths, and injected future rows pass the shape, positivity, finiteness and invariance checks.

### Full exact-temporal/Sobol hourly confirmation and promotion

At every official reward round and 1000 paths, full exact-context temporal correction plus direct scrambled Sobol improves hourly reward ratios from 0.629991 to 0.869436 in dev1 and from 0.704092 to 0.872234 in dev2. Cumulative ranks improve 87→25 and 135→14 out of 257. Final-round ranks are 60 and 1, respectively; the latter is not the cumulative rank. Promote these hourly components with self-contained NumPy/SciPy inference in `modeling.py`. Daily remains the confirmed lead-shaped model pending its own finer-clock confirmation. Development Score estimates are now approximately 34.052 and 31.044; the reserved final window is still untouched.

## Iteration 31 — context-only adaptive calibration (preregistered)

- Hypothesis: a fixed pooled model can retain a slow-moving calibration bias on newer venues. Recently completed volatility forecasts inside the supplied seven-day context may identify that bias without retraining or external data.
- Change: reconstruct nine completed static-neural volatility forecasts using only observations available at each historical anchor, compare their predicted scales with subsequently observed absolute-return scales, and shrink the ratio of means by strengths 0.25/0.5. Hourly anchors cover the past nine completed hours; daily anchors span the past three days with six-hour spacing. Skip anchors preceding weight availability and require at least three usable forecasts. Cap the unshrunk ratio to [0.5,2].
- Result: pending official development CRPS against the lead-shaped incumbent. Overlapping daily labels are not treated as independent observations. Historical subcontexts are necessarily shorter than seven days and this approximation must be reported.
- Verdict: retain only if adaptation improves both regimes without destabilizing other competitions; otherwise prefer the fixed forecaster.
- Lesson: pending. Every calibration target ends by the actual prediction time, and no past served miner paths are used.

### Iteration 30 result and bounded refinement

The shape-matched experiment computes 47,875 official CRPS values at 1000 paths over 6,081 contexts. No fitted epoch improves the pre-development validation objective of 0.800481; the selector retains the exact zero correction. Reject this fit without redundant development scoring. One bounded follow-up is preregistered: resolve the scale grid more finely around 1, because a coarse piecewise-linear surrogate can place its minimum at a grid knot even when the underlying smooth expected score has a nearby optimum. Preserve the original artifact and do not call the interpolated objective an official CRPS evaluation.

## Iteration 30 — shape-matched official-score calibration (preregistered)

- Hypothesis: the previous score-curve fit used a flat classical path family at 128 paths, while deployment now includes learned scale and clock/lead profiles. Training and inference objectives therefore differed even after correcting raw-score weighting.
- Change: recompute five scale-response points around the lead-shaped incumbent at 1000 paths on the same pre-field training contexts, using the official scorer and its realized-path slicing helper. Fit a small residual log-scale network, initialized to no correction, using raw asset-weighted CRPS and constant competition-level numerical scaling. Keep the existing chronological split and bounded corrections.
- Result: pending. These are in-sample/meta-training response curves around fitted parameters, not historical causal backtests; only later development-window evaluations are performance claims. No field-rank calculation uses these training dates.
- Verdict: require consistent development CRPS and official reward gains before adopting a calibration correction.
- Lesson: pending; fixing objective alignment is necessary to test an idea, not evidence that it will win.

## Iteration 29 — finer-clock interaction and joint normalization (preregistered)

- Hypothesis: finer empirical clocks may complement the temporal residual. Separately, normalizing clock and lead components individually does not guarantee that their product has unit mean; the earlier description of perfectly preserved average scale was too strong.
- Change: around full exact-temporal plus half-lead, compare 30-/15-minute clocks, full clock strength at 15-minute resolution, and explicit unit-mean normalization of the final product. Each comparison has an adjacent one-variable control. All forecasts remain positive and continuous-trading, with no sessions or zero-volatility gates.
- Result: a 16-context diagnostic finds joint mean multipliers from 0.9992 to 1.0252 despite unit-mean components. Existing held-out forecasts and rewards remain valid; only the previous interpretation needs correction. The simpler clock screen gives consistent equity gains at 30-/15-minute resolution (ratios 0.99761/0.99799 and 0.99659/0.99745). Crypto favors the more conservative 30-minute profile (0.99929/0.999994). Lowering the floor alone is effectively a null change in this sample.
- Verdict: adopt only on fresh official CRPS and reward evidence, not merely because unit-mean normalization sounds cleaner.
- Lesson: component normalizations do not commute with multiplication. Document the actual forecast, not the intended interpretation.

## Iteration 28 — empirical clock resolution and shrinkage (preregistered)

- Hypothesis: the current hourly clock and three-bin smoothing may blur repeatable volatility bursts. Its 25% profile floor and 25% shrinkage also enforce substantial dispersion in empirically quiet periods. These are modeling choices, not venue constraints.
- Change: against the lead-shaped incumbent, change one feature at a time: 30-minute versus hourly bins, 15-minute versus hourly bins, full versus 0.75 profile strength, or a 0.05 versus 0.25 positive profile floor. Keep the same within-day normalization, smoothing kernel, maximum profile value and seven-day context. Every bin remains strictly positive; no trading sessions, closures or external calendars are introduced.
- Result: pending both development windows. A score-only field diagnostic shows the largest weighted daily gaps are AAPLX, TSLAX, GOOGLX and NVDAX, not SP500 despite its higher coefficient. This diagnostic compares public aggregate score levels, never miners' served paths.
- Verdict: require consistent official CRPS and reward gains before combining with the temporal candidate.
- Lesson: pending.

### Iteration 27 result

Forecast-error mixtures fail the consistency gate at 1000 paths. Half conditional replacement gives COM_EQU CRPS ratios 0.99987/1.00065, hourly 1.00007/1.00038, daily crypto 1.00015/1.00099. Full replacement regresses further, and conditioning on the recent volatility trend barely differs from pooling. Discard these mixtures. The simpler Student mixture remains better calibrated than this empirical future-volatility uncertainty proxy.

## Iteration 27 — forecast-error volatility mixtures (preregistered)

- Hypothesis: uncertainty about future average volatility may differ by horizon and recent volatility trend; a fixed Student scale mixture ignores this structure.
- Change: pool pre-July-1 ratios of future realized absolute-return scale to the static forecast, separately by crypto/noncrypto and horizon, optionally conditioning on falling/stable/rising recent volatility. Store 101 positive empirical quantiles, normalize each pool to the same expected path multiplier as the df5 incumbent, then replace half/all latent Student scales with these forecast-error scales. This changes distribution shape, not mean absolute movement, and does not bootstrap raw price returns.
- Result: pending 1000-path official CRPS screens in both development windows.
- Verdict: retain only a consistent gain; otherwise keep the simpler Student mixture. The temporal and periodic mean-scale models remain separate experiments.
- Lesson: pending.

## Iteration 26 — smooth data-fitted periodic scale (preregistered)

- Hypothesis: continuously traded markets can have smooth activity cycles even without sessions or closures. Past-week clock shape does not explicitly forecast how average future volatility depends on the current phase of those cycles.
- Change: fit a strongly regularized positive scale residual from daily/weekly sine and cosine harmonics, shared and interacted with asset identity. Fit only the pre-July-1 rows and select regularization on July 1–25; use the confirmed half-hourly/full-daily neural baseline. Bound log corrections at +/-0.5 and test half/full strengths. There are no exchange calendars, market-hour gates, weekend exclusions, or zero-volatility bins.
- Result: pending chronological likelihood and official development CRPS against the lead-shaped incumbent.
- Verdict: require consistent gains; reject rather than invent a trading-session explanation if the data do not support this periodic predictor.
- Lesson: pending.

### Iteration 25 result

All three ridge penalties worsen chronological validation mean-square error versus zero location: baseline 1.82452; penalties 100/1000/10000 give 1.84238/1.84066/1.83205 on 2,208 validation rows (20,215 training rows). The selector correctly exports zero coefficients. Reject before spending on redundant CRPS screens, because every proposed location path would equal the incumbent. Keep zero log drift; no evidence supports a directional adjustment.

### Temporal boundary audit and exact-context refit

The cached temporal training sequence includes one return at the seven-day left boundary that runtime cannot form from its supplied frame. Three sampled contexts show differences only in the first hourly bin: maximum 0.0167–0.0284, with every later bin exactly equal. This is older-context mismatch, not future leakage; all development predictions already used the exact runtime context. Nonetheless, refit a separately named `temporal_exact` artifact using the runtime feature function for every training row before promotion. Preserve the original artifact/results rather than silently changing their provenance.

## Iteration 25 — conditional price location (preregistered)

- Hypothesis: zero conditional log drift may leave small mean-reversion or momentum information unused. This is distinct from the previous experiments on volatility reversion and serial covariance.
- Change: fit a heavily regularized ridge model for standardized future log returns at 5, 60 and 1440 minutes, from the original static inputs plus ten trailing signed-return horizons and their availability indicators. Select the ridge penalty only on the pre-July-26 chronological holdout, compare to a zero forecast, cap each predicted location at 0.3 reference standard deviations, and test quarter/half strengths against the promoted lead-shaped incumbent. No external prices or directional labels after the training cutoff.
- Result: pending likelihood-independent mean-square diagnostic and official development CRPS.
- Verdict: discard unless location helps both development windows and does not harm another competition materially; no directional story will substitute for measured performance.
- Lesson: pending.

### Full daily lead confirmation and promotion

At 1000 paths and every official reward round, half lead shape improves daily crypto reward ratios 0.135626→0.151344 (rank 118→97) and 0.037376→0.058647 (rank 230→227), and COM_EQU 0.000498→0.000785 (rank unchanged at 174) and 0.000156→0.000442 (rank 231→230). There are 1,499/1,474 equity rounds and 1,015/1,044 daily-crypto rounds in dev1/dev2. Promote half lead shape with its pre-July-26 weights. Combined with the confirmed hourly correction, development Score estimates become 26.071 and 25.439. The remaining daily reward gap is still large, so continue the campaign.

### Iteration 23 first result

The fixed four-seed ensemble yields only tiny daily changes (full-strength COM_EQU 1.00026/0.99928, daily crypto 1.00005/0.99990), while hourly improves 0.99953/0.99982. Discard as a general daily replacement. Retain the hourly ensemble only for a dense reward screen against the temporal correction; no individual seed is cherry-picked.

## Iteration 24 — scale/shape interaction (preregistered)

- Hypothesis: the temporal residual changes mean forecast scale while the lead-time network redistributes scale over the day, so their gains may be complementary rather than redundant.
- Change: at 1000 paths, compare tuned, half/full temporal residual, and each temporal strength combined with half lead shape, on every archived daily prompt in both development windows. Hourly is handled separately because lead shape is inactive there.
- Result: pending. The standalone temporal screen improves all six competition/window cells at half strength (COM_EQU 0.99792/0.99387, hourly 0.99965/0.99959, daily crypto 0.99970/0.99808). Full temporal improves equities further but is less stable elsewhere. Its pre-development likelihood improves 1.01930→1.00675, selected at epoch 24; NumPy/Torch outputs agree within 2e-6 and future-row invariance passes.
- Verdict: select a combined candidate only if dense official CRPS and rewards improve both windows; otherwise keep the simpler component.
- Lesson: pending; held-out likelihood and actual rewards remain distinct diagnostics.

### Full hourly confirmation and promotion

Every official reward round confirms half-strength neural hourly correction at 1000 paths. Dev1 reward ratio improves 0.368394→0.629991, rank 136→87 out of 257 over 1,624 post-warmup rounds. Dev2 improves 0.446012→0.704092, rank 198→135 over 1,847 rounds. Final-round ranks are 140 and 37, distinct from cumulative reward ranks. Promote this hourly correction in `modeling.py`; no daily change yet. Combining these hourly results with the existing full daily incumbent gives development campaign Score estimates 25.54 and 24.72, up from 16.82 and 16.12. These remain development estimates, not the reserved final test.

### Sampling seed robustness

Three additional preregistered seeds support the small direct-Sobol gain in all four daily competition/window combinations (ratios 0.99947–0.99993 daily crypto and 0.99957–0.99984 COM_EQU). Hourly improves five of six seed/window combinations; the exception is 1.00023 in dev1 seed 3. The gain is therefore not universal for any individual scramble, but daily robustness and pooled hourly direction justify dense 1000-path reward confirmation. No seed is chosen based on these results; keep the original prompt seed.

## Iteration 23 — predeclared neural seed ensemble (preregistered)

- Hypothesis: the compact volatility model may have material optimization variance despite its small architecture. Averaging independent fits could improve robustness without new features or data.
- Change: repeat the original likelihood training with seeds 73, 420 and 2026; average their log-scale forecasts with the original model using fixed equal weights. Keep all datasets, splits, architecture and 150-epoch selection procedure unchanged. No choosing individual seeds by development performance; test only the predetermined ensemble at half/full strength.
- Result: pending. This GPU job starts only after the temporal job exits.
- Verdict: retain only on consistent development and official reward gains; otherwise prefer the smaller single network.
- Lesson: pending.

## Iteration 22 — temporal residual volatility network (preregistered)

- Hypothesis: eight EWMA summaries discard the ordering of recent volatility bursts. A small convolutional encoder of the previous seven days may predict volatility beyond the static feature model.
- Change: summarize observed gap-normalized returns into 168 trailing hourly bins of absolute scale, RMS, signed mean and coverage. A three-layer temporal convolution predicts a regularized residual to the frozen original neural log-scale, using the same price-only training rows and chronological split. Missing bins receive neutral feature values plus zero coverage, not fabricated prices. NumPy-only inference is required before any adoption.
- Result: pending held-out likelihood, serving latency and official development CRPS. Limit GPU memory to 35% and run only one GPU job.
- Verdict: reject unless it improves both development windows beyond trivial noise and supports the 1000-path serving contract.
- Lesson: pending. Extra architecture is not presumed better than the compact EWMA model.

## Iteration 21 — horizon-matched volatility labels (preregistered)

- Hypothesis: training a daily model on future one-minute absolute returns can miss the five-minute microstructure that dominates daily CRPS. Predicting future five-minute absolute returns directly may improve calibration, even though changing the historical feature lag alone did not pass the earlier consistency gate.
- Change: keep the original feature rows, model architecture, optimizer and chronological split; replace only the daily target with the mean future five-minute absolute log-return divided by sqrt(5), on overlapping observed endpoint pairs within the next 24 hours. No interpolation, and all target endpoints remain before July 26. Keep the original hourly target for joint training but do not deploy its changed head in this experiment.
- Result: pending official development CRPS at half/full strength relative to `tuned`.
- Verdict: require a consistent gain and confirm rewards at 1000 paths; otherwise reject the target modification.
- Lesson: pending.

## Iteration 20 — finite-ensemble variance reduction (preregistered)

- Hypothesis: independent Student mixing scales currently break exact antithetic symmetry. Paired scales or low-discrepancy joint draws may improve finite-ensemble CRPS without changing the intended predictive law.
- Change: compare shared antithetic pair scales, stratified scale quantiles, scrambled Sobol direct increments, and scrambled Sobol Brownian-bridge increments against `tuned`. Screen at the actual 1000-path serving count, since the power-of-two behavior of Sobol would make a 256-path comparison misleading. Retain arbitrary-count support and deterministic prompt seeds.
- Result: pending official CRPS and reward checks in both development windows.
- Verdict: require reproducible improvements across both windows, then verify full prompt coverage and seed sensitivity.
- Lesson: the bridge construction is a hypothesis, not a promised gain. Its increment transform was verified orthonormal at 60 and 288 steps. The installed SciPy Sobol documentation explicitly warns that non-power-of-two samples lose full balance properties. We use 1000 because the validator requires it, not 1024.

### Iteration 19 result

The five-bucket supervised validation loss improves from 1.01379 to 0.99737 (23,337 training and 2,508 validation rows). At 256 paths, half-strength lead shape gives COM_EQU relative CRPS 0.99769/0.99543 and daily crypto 0.999995/0.99852. Full strength has a small dev1 crypto regression. End-round sparse reward checks support half strength: COM_EQU reward ratios 0.000178→0.000447 and 0.0000564→0.000162; daily crypto 0.07663→0.07770 and 0.02316→0.03599. Retain half strength for 1000-path dense confirmation, not yet production.

### Iteration 20 first result

At 1000 paths on 2160 sampled prompts, direct scrambled Sobol is the only sampler that improves every competition in both windows: mean asset-relative CRPS 0.99980/0.99964 COM_EQU, 0.99990/0.99994 hourly crypto, 0.99976/0.99980 daily crypto. These gains are very small; dense and seed-sensitivity confirmation is required. Mean generation time is 0.029 seconds versus 0.0069 for `tuned`, safely under the latency limit. Pure shared pair scales are mixed; Brownian bridge is mixed daily; stratification alone is mixed hourly. Retain direct Sobol only as an experimental candidate.

### Iteration 21 result

Five-minute targets are typically 2–8% larger than one-minute targets after diffusion rescaling on the newer venues, confirming that the targets differ. This does not translate into robust forecasting gains: half-strength COM_EQU CRPS ratios 0.99896/1.00084 and daily crypto 0.99959/1.00124; full strength regresses further in dev2. Discard. Like direct raw-CRPS fitting, it moves calibration in the wrong direction in the calmer development window.

### Iteration 18 result

Correct raw-score weighting improves the held-out curve objective only slightly, from 0.795870 to 0.795527 (epoch 28). Half-strength official CRPS ratios to `tuned` are COM_EQU 0.99895/1.00073, hourly 1.00045/1.00010, daily crypto 0.99967/1.00111. Full-strength regresses further in dev2. Sparse end-round official reward checks confirm the lack of consistency: daily-crypto ratios change 0.07663→0.09110 in dev1 but 0.02316→0.01649 in dev2; hourly changes 0.86748→0.72852 and 0.53715→0.47545. These single-round, reduced-path numbers are screening diagnostics, not campaign Score estimates. Discard the fitted correction, retain the objective-rule correction as an important methodological lesson.

## Iteration 19 — learned lead-time volatility shape (preregistered)

- Hypothesis: a flat learned daily scale cannot represent conditional volatility relaxation after a recent shock. Predicting positive volatility in five future buckets can improve the path's allocation of uncertainty without changing its average scale.
- Change: train a pooled five-output network on snapshot-only future absolute-return scales in minutes 0–60, 60–180, 180–360, 360–720, and 720–1440. Reuse the original 30 features and chronological training/validation split, with all targets ending before July 26. Normalize the predicted bucket scales to unit duration-weighted mean, then blend only the shape with the incumbent daily local-clock profile.
- Result: pending official development CRPS and reward replay; hourly behavior remains unchanged.
- Verdict: require consistent development improvements before full 1000-path confirmation; otherwise discard.
- Lesson: pending. This isolates conditional lead time rather than revisiting fixed exchange sessions or hand-coded volatility decay.

## Iteration 0 — preregistration and evaluation audit

- Hypothesis: the seven-day Gaussian starter is miscalibrated in changing volatility regimes; correcting evaluation mechanics must precede model comparisons.
- Change: installed the pinned public libraries with `uv sync`; read the validator's CRPS, competition configuration, weighting and official backtester. Replaced the starter's strict context loader with raw parquet loading, preserving NaNs. Generate at archived prompt timestamps rather than approximate hourly timestamps. Enforce the frozen cutoff and separate candidate output directories.
- Result: pending baseline. Snapshot contains 13 live assets; both 289-point and 61-point formats require 1000 served paths. The daily score sums 288 five-minute, 48 thirty-minute, 8 three-hour changes and one absolute endpoint; the hourly score additionally includes twelve start-to-horizon gap terms.
- Verdict: keep infrastructure changes, not a model claim.
- Lesson: the campaign describes a lenient starter loader, but the provided implementation actually calls strict `load_range`. Also, installed competition windows are 10 days (daily) and 5 days (hourly), not a 24-hour window; use the pinned validator as truth.

## Research protocol

All prices come from the snapshot. Never inspect data after 2026-08-29 or individual miners' served predictions. Training targets end before validation starts. Field comparisons start after 2026-06-23; SP500 ranks only after 2026-07-19. Initial development windows: scored 2026-07-20 to 2026-07-30 and 2026-08-04 to 2026-08-14; reserve 2026-08-18 to 2026-08-28 for final validation. Inspect volatility to determine whether those regimes genuinely differ rather than asserting they do.

Planned sequence: reproducible baseline; recent versus seven-day volatility; robust scale and heavy-tailed mixtures; horizon-dependent volatility reversion; fitted volatility forecasting if simpler calibration stalls. Each substantive change needs lower paired official CRPS across both development windows (target at least 0.5% mean gain), no competition-level regression exceeding 1%, and final field-relative validation at 1000 paths. These are provisional selection gates, not claims of statistical significance. All CRPS and reward calculations use `synth_lib.backtester` functions through `agent/predict.py`.

### Coverage correction and baseline probe

The archived field starts on July 27, not June 23. Thus development windows are now scored July 28–August 7 and August 8–18, and final validation August 18–28. This correction precedes candidate evaluation. A 256-path BTC daily smoke test on 49 archived prompts ending July 30 produces official mean CRPS 2738.3321 and final per-asset smoothed score 61.7331. This is a reduced-path infrastructure check, not a nominated rank.

## Iteration 1 — recent versus robust volatility (preregistered)

- Hypothesis: shorter volatility half-lives improve crypto-hourly calibration; mean absolute returns reduce Gaussian overdispersion from isolated jumps on both formats.
- Change: compare deterministic antithetic Gaussian starter, EWMA RMS half-lives 6h/24h, EWMA absolute-return scales 6h/24h. Normalize nonadjacent observed returns by square root elapsed time; preserve missing observations. Fixed timestamp-derived seeds keep paired comparisons reproducible.
- Result: pending 256-path screening, 4-hourly daily prompts and approximately 4-hourly hourly prompts across both development windows and all live assets.
- Verdict: pending. Select only with paired gains across both windows; screen then confirm at 1000 paths.
- Lesson: pending.

### Iteration 3 result and regime check

Replacing one-minute calibration with five-minute calibration has mixed mean relative CRPS: COM_EQU_24H 0.9982/1.0010, CRYPTO_1H 1.0001/1.0016, CRYPTO_24H 0.9993/0.9958. Longer lags generally hurt hourly. Discard as a universal change; daily-crypto lag-five remains an exploratory candidate, below the preregistered consistency gate.

Snapshot realized daily RMS confirms development regime contrast: dev1/dev2 mean daily volatility BTC 1.502%/0.961%, ETH 2.059%/1.283%, SP500 0.713%/0.444%, AAPLX 2.124%/0.963%. The second period is substantially calmer on these four diverse assets; neither is asserted to be a bull/bear regime.

## Iteration 4 — mean-reverting forecast volatility (preregistered)

- Hypothesis: a short-memory near-term scale that relaxes to a longer-memory scale over the path is better calibrated than a flat diffusion.
- Change: log-scale exponential reversion from 30-minute (hourly) or 60-minute (daily) absolute-return EWMA to slower 6h/24h targets, with 1h/6h/24h relaxation. Preserve df=5 and zero drift.
- Result: pending; compare to auto:0:1:5 on both development windows.
- Verdict: pending; require the established paired gain gate.
- Lesson: pending.

## Iteration 5 — pooled learned volatility (preregistered)

- Hypothesis: pooling the long price history can learn persistence from multiscale absolute/RMS return features better than a single manually chosen half-life.
- Change: fit ridge and a small neural forecaster to future mean absolute returns, using only targets ending before July 26. Chronological model-selection holdout begins July 1, strictly before the field development windows. Inputs use exactly the seven-day context; output is a multiplicative correction to the incumbent hourly/daily scale. Training loss is a Laplace scale likelihood, not a separately reimplemented CRPS. Evaluate all final comparisons through official CRPS.
- Result: pending; fixed training seed, 150 epochs maximum, at most 35% GPU memory, one GPU job. Compare ridge/neural plus a half-strength blend to the incumbent on both development windows.
- Verdict: pending; discard if validation CRPS does not meet the paired improvement gate.
- Lesson: pending.

### Iteration 4 result

Volatility reversion does not clear the paired >0.5% gate. The conservative 6h target/1h reversion has mean relative ratios COM_EQU_24H 0.9995/0.9990, hourly 0.9987/0.9997, daily crypto 0.9999/0.9998. Longer reversion to 24h helps the calm window but hurts the earlier volatile window. Discard for now: the flat horizon-specific robust scale is simpler and comparably calibrated.

## Iteration 6 — temporal organization of heavy tails (preregistered)

- Hypothesis: tails generated independently at each step or by evolving stochastic volatility may match multi-horizon scoring better than one constant latent scale per path.
- Change: compare iid Student innovations (df 3/5/8), lognormal stochastic volatility (log-scale SD 0.4/0.7, persistence 1h/6h), and mean-centered empirical-return bootstrap to the incumbent pathwise df=5 model. Keep RMS diffusion scale fixed; this screen diagnoses distribution shape, not a final parameter search.
- Result: pending; two development windows, 256 paths, identical prompt sample.
- Verdict: pending; require consistent gains to justify the added path dynamics.
- Lesson: pending.

### Iteration 5 implementation failure, caught before CRPS evaluation

The first training target distribution had median scale ratios near 30 instead of near one. Root cause: pandas/Arrow timestamps are microsecond-indexed, while target code divided `.asi8` differences as nanoseconds, inflating targets by sqrt(1000). Runtime research features use `total_seconds()` and are unaffected. Discard the invalid training artifacts; explicitly convert target timestamps to nanoseconds, add a minute-spacing invariant, rebuild the dataset, and rerun training. No invalid learned model was evaluated or retained.

### Iteration 5 corrected training diagnostics

Corrected dataset: 23,457 training contexts and 2,508 chronological holdout contexts across all 13 assets. Baseline holdout Laplace scale losses are 1.04008 hourly and 1.09789 daily; ridge gives 1.02366/1.05288; the neural model reaches mean 1.01930 at epoch 129. These are likelihood diagnostics, not CRPS or emission claims. Exported NumPy weights use CPU-only inference and were selected before the development windows. Proceed to official CRPS validation.

### Iteration 6 result

None of iid Student innovations, empirical bootstrap, or evolving lognormal volatility clears the improvement gate. The mild 0.4 log-volatility/6h-persistence model gives ratios 0.9987/0.9997 COM_EQU_24H, 1.0004/1.0008 hourly, 0.9992/0.9998 daily crypto. The empirical bootstrap consistently hurts; larger stochastic-volatility dispersion hurts crypto by approximately 1–1.6%. Keep the simple constant pathwise Student mixture and discard these dynamics.

### Iteration 5 official CRPS result

Neural daily volatility passes: relative to the robust Student incumbent, full neural correction produces COM_EQU_24H 0.9860/0.9892 and CRYPTO_24H 0.9943/0.9841. Hourly full neural correction regresses 1.0012/1.0012, so retain the classical hourly forecaster. Half-strength daily neural correction has COM_EQU_24H 0.9903/0.9889 and daily crypto 0.9957/0.9881. AAPLX and SP500 favor stronger shrinkage, but avoid asset-specific tuning; compare full versus half-strength correction at the competition-family level with denser, 1000-path official scoring. Ridge is weaker than neural for daily and does not clear the hourly gain gate.

## Iteration 7 — dense 1000-path confirmation (preregistered)

- Hypothesis: the daily neural gain survives a fourfold denser prompt sample and the true 1000-path serving size; neural corrections should be shrunk for commodities/equities if coefficient-weighted ranks favor that choice.
- Change: compare robust Student incumbent, full daily neural correction, and half-strength correction for COM_EQU_24H only. Hourly remains classical for all three. Score every daily prompt and every sixth hourly prompt on both development windows.
- Result: pending.
- Verdict: require consistent competition-level gains; otherwise revert the learned component. No final-window tuning.
- Lesson: pending.

### Iteration 7 result and serving checks

The dense experiment evaluates 8,640 prompts and 25,920 forecasts at 1000 paths. Neural gains persist: full daily correction versus robust Student gives COM_EQU_24H 0.98692/0.98838 and CRYPTO_24H 0.99503/0.98414; half-strength commodity/equity correction gives 0.99085/0.98851. Hourly is identical by construction. A sampled end-of-window official reward replay will resolve the coefficient-weighted blend choice before the full field replay. Provisional production implementation matches the screened model within floating-point precision, returns the required 1000x289/1000x61 shapes, accepts arbitrary positive path counts and NaN-contaminated frames, ignores injected future rows, and takes 0.026–0.047 seconds for representative 1000-path prompts.

## Iteration 8 — score-aware fine-tuning (preregistered)

- Hypothesis: likelihood improvements need not improve CRPS (as the hourly result shows). Training on the actual official scorer's response to volatility changes may better align the learned correction with the competition.
- Change: sample a deterministic quarter of the pre-July-26 training contexts; tabulate ten multiplicative volatility scales at 128 paths using only `predict.score_paths` and the official backtester scorer. Fine-tune the existing small network against piecewise-linear interpolation of those stored score values, regularized toward modest corrections. This is not a replacement CRPS implementation; every response-curve value is generated by the official scorer. Keep the same July-1 chronological validation split and no field-window fitting.
- Result: pending; interpolation loss is only a training diagnostic. Deployment requires fresh official CRPS at 1000 paths and unchanged development windows.
- Verdict: pending; reject if gains are inconsistent or only appear at the reduced training path count.
- Lesson: pending.

### Field-relative reality check

The sampled official final-round replay favors full daily neural correction over half-strength COM_EQU correction in both windows, so use full correction provisionally. However, low CRPS improvements do NOT imply field competitiveness: daily crypto sampled reward ratios are only 0.0250/0.0070 and COM_EQU about 1.2e-6/9.9e-7. Hourly ratios are 0.206/0.582. These are sampled end-round diagnostics, not the final cumulative-emission Score. The remaining daily gap is material, so early nomination is not justified. Continue research.

## Iteration 9 — empirical periodic volatility, not sessions (preregistered)

- Hypothesis: a flat daily variance forecast misses repeating information-flow patterns even on a venue that trades continuously. A weak periodic profile estimated exclusively from the supplied past seven days may improve daily paths.
- Change: normalize past absolute returns within each observed calendar day, estimate a smoothed 24-hour profile, and shrink toward a flat positive curve. Test strengths 0.25/0.5/0.75/1 against the full daily-neural incumbent. This has NO exchange calendar, open/close boundary, weekend switch, or zero-volatility session. Every hour remains active; this experiment tests whether a periodic pattern exists rather than assuming one.
- Result: pending. Price NaNs remain dropped, never interpolated; smoothing applies only to estimated volatility parameters.
- Verdict: retain only if both development windows show consistent official CRPS gains; otherwise discard the periodic hypothesis.
- Lesson: pending.

### Iteration 9 result

The empirical positive volatility profile clears the gate. Relative to the daily-neural incumbent, strength 0.75 gives COM_EQU_24H 0.98653/0.98038 and daily crypto 0.99567/0.99463; strength 1 gives 0.98496/0.97800 and 0.99577/0.99467. Hourly is unchanged. Thus continuous trading does not imply a flat variance clock in this snapshot. Prefer the slightly shrunk 0.75 profile provisionally and validate full field emissions at 1000 paths; do not reinterpret this as exchange-session closures.

### Iteration 8 training result

Computed 95,750 official CRPS values at 128 paths over 6,081 pre-field contexts, with 9,575 valid context/horizon pairs. Score-aware fine-tuning improves the chronological interpolated validation ratio only from 0.992970 to 0.992561, peaking at epoch 5; longer fitting overfits. The new model is now undergoing fresh official development-window CRPS screening. This tiny training gain alone does not justify deployment.

### Iteration 8 held-out result

Reject score-aware fine-tuning: full correction relative to the likelihood-trained hybrid gives COM_EQU_24H 1.00084/0.99619, hourly 1.00094/1.00123, and daily crypto 0.99995/0.99953. The small training-objective gain does not transfer consistently, and hourly still regresses. Keep the likelihood-trained daily model and classical hourly model.

## Iteration 10 — moment-matched tail calibration (preregistered)

- Hypothesis: previous shape comparisons fixed RMS, which changes expected absolute movement across Student degrees of freedom. Matching the absolute first moment may expose a better shape/scale pair, especially for peaked commodity/equity returns.
- Change: hold the neural-plus-0.75-cycle forecaster fixed; compare Student df 3/5/8 and Gaussian shapes at expected absolute movements 0.8/0.9/1.0/1.1 times the predicted absolute scale. The Student normalizing factor is analytic; every forecast CRPS is still computed only by the official scorer.
- Result: pending. Screening at 256 paths precedes 1000-path confirmation and coefficient-weighted field selection.
- Verdict: retain only consistent cross-window gains, not an isolated grid winner.
- Lesson: pending.

### Iteration 10 result

Moment matching does not reveal a consistent material improvement. Student df3 at matched absolute scale 1.0 gives hourly 0.99920/0.99939 but daily-crypto 1.00044/1.00166; other scale choices trade gains in the calmer window for losses in the earlier window. Retain df5 at the existing scale. This confirms that the prior simple mixture was near a broad calibration optimum; it does not prove all heavy-tailed models are equivalent.

## Iteration 11 — periodic-profile stability (preregistered)

- Hypothesis: six or seven observed days make the local variance profile noisy. A strictly pre-field historical profile can regularize it without any assumed exchange calendar or zero-variance periods.
- Change: fit positive per-asset hourly profiles from April 1 (or first trade) through July 26, using the same within-day normalization and circular smoothing. Blend 25/50/75/100% historical prior into the local profile; keep total cycle strength 0.75. Separately check 15/30/60/120-minute robust half-lives for hourly forecasts, changing nothing else.
- Result: pending official development-window CRPS. The historical profile never reads validation-period prices when fitted.
- Verdict: require consistent gains across both windows; otherwise prefer the local-only profile and 60-minute hourly scale.
- Lesson: pending.

### Iteration 11 result

Keep local-only profile and 60-minute hourly memory. Historical profile shrinkage has <0.1% gains and mixed COM_EQU performance (100% prior 0.99930/1.00120). Hourly 15/30/120-minute half-lives versus 60-minute produce 1.00186/1.00422, 1.00008/1.00028, and 1.00135/1.00211. The additional parameters are not justified.

## Iteration 12 — chronological refitting and recency weighting (preregistered)

- Hypothesis: the original network's optimizer only fitted labels before July 1; July 1–25 was used for model selection. Refitting on all labels already available before the field windows, with recency weighting, may better match newer venues without violating no-lookahead.
- Change: initialize from the likelihood model, make 40 fixed-epoch passes over all pre-July-26 labels, and compare uniform versus 60/120-day exponential recency weights. No development or final-window labels enter the fit. Daily correction remains full strength with the 0.75 local profile; hourly stays classical. Compare optional half-strength shrinkage only if full corrections regress.
- Result: pending official development-window CRPS. Training-batch likelihood is only diagnostic, not evidence of improvement.
- Verdict: keep only a consistent cross-window gain; otherwise retain the original frozen weights.
- Lesson: pending.

### Full baseline reward replay

The 1000-path deterministic starter has official cumulative-emission Score estimates 0.32375 on dev1 and 0.42456 on dev2. Reward ranks (COM_EQU_24H / CRYPTO_1H / CRYPTO_24H) are 239/216/236 of 257 on dev1 and 248/227/257 on dev2. These are full archived-prompt, all-asset replays, not the earlier sampled terminal-rank diagnostics. The two concepts must not be conflated.

### Evaluation throughput change

Reward replay is now optionally partitioned by reward timestamps across workers, each calling the unchanged official `compute_combined_smoothed_scores` with the full score history; the unchanged official warmup trim is applied again after concatenation. Verify equality with the serial implementation, including a split across warmup, before using it for nomination. This is parallel scheduling, not a new score/rank formula.

### Iteration 12 result

Recency refitting gives small but directionally consistent gains. The 60-day weighted refit gives COM_EQU_24H 0.99942/0.99812 and CRYPTO_24H 0.99926/0.99901 versus the current neural-cycle incumbent. These are below the original 0.5% CRPS screen threshold, so do not replace the weights yet. Because emissions are nonlinear, retain this as a candidate for coefficient-weighted reward confirmation rather than treating the threshold as proof of no utility.

## Iteration 13 — cross-horizon return dependence (preregistered)

- Hypothesis: even a well-calibrated one-step distribution can misprice longer scored intervals if returns are not independent. Mild serial dependence may improve multi-horizon CRPS without changing one-step marginal scale.
- Change: preserve the full neural-plus-local-cycle incumbent and df5 marginal mixture; generate stationary AR(1) Gaussian innovations at correlations -0.2/-0.1/+0.1/+0.2, plus a half-shrunk correlation estimated from valid adjacent context returns at the served step size. No price interpolation, drift signal, or future information is used.
- Result: pending official development-window CRPS.
- Verdict: require consistent cross-window improvement; otherwise retain independent increments.
- Lesson: pending.

### Iteration 13 result

Reject serial correlation as a general improvement. Half-shrunk empirical correlation gives ratios within 0.02% of the independent model; fixed nonzero correlations trade gains between windows and horizons. For example rho=-0.1 yields hourly 1.00132/1.00298, whereas rho=+0.1 yields 1.00069/0.99878. The data-driven estimate is effectively a null result, and introducing a fixed correlation would be tuning noise.

### Parallel replay verification

Serial and three-worker official combined replay are exactly equal across all columns on 1,542 result rows, with input reward times straddling the hourly warmup boundary. The scheduling optimization is validated.

## Iteration 14 — emission-sensitive hourly selection (preregistered)

- Hypothesis: the softmax can make small, consistent CRPS gains economically relevant even when they fall below a generic 0.5% screen gate. The remaining hourly candidates should be compared on archived reward aggregation, not discarded solely on unweighted CRPS averages.
- Change: score every archived hourly prompt in both development windows at 1000 paths for the incumbent, moment-matched df3, half-strength ridge/neural/score-fitted corrections, 30-minute memory, mild volatility reversion, and +/-3% scale adjustments. Reuse only reward timestamps from the full baseline backtest; recompute prompt scores and combined weights using the official functions. Thin reward rounds by a fixed factor for initial selection, then replay every round for finalists.
- Result: pending. All candidates were specified before this dense comparison; no final-window observations enter selection.
- Verdict: promote only a candidate whose reward-ratio gain is consistent across both windows and survives full-round replay. Report this adjusted selection criterion transparently.
- Lesson: pending.

## Iteration 15 — use available intraminute information (preregistered)

- Hypothesis: close-to-close volatility throws away intraminute range information, which may improve short-horizon scale estimates. The snapshot documentation incorrectly describes non-close columns as empty.
- Change: direct inspection on July 25 shows usable OHLCV/trade counts for BTC/ETH/SOL/XRP (100% of minutes), HYPE/SP500 (~99.4%), and NVDAX (~53.6%, matching its close density that day). Compute a recent mean log(high/low) scale and geometrically blend it with the close-based scale. Compare 50/100% blend and 0.9/1.0/1.1 range calibration. If high/low are absent or unusable, fall back to the close-only forecast. No online prices are used.
- Result: pending official CRPS on both development windows. A direct blend is a diagnostic; retraining with range information would be a separate experiment.
- Verdict: require consistent gains and robustness to NaN/absent OHLC inputs.
- Lesson: pending.

### Iteration 15 result and candle audit

Reject direct range substitution. A 50% range blend at calibration 1.0 yields COM_EQU_24H 1.01133/0.99371, hourly 1.00323/1.00662, and daily crypto 1.00361/0.99953. The range scale is much smaller than the close-based scale (e.g. 0.635 for BTC and 0.540 for AAPLX on the July-25 context). This is not evidence that OHLC is corrupted: all inspected complete candles satisfy low <= open/close <= high; BTC's open is essentially the previous close, while sparse Hyperliquid assets have material between-candle opening moves. Naive continuous-diffusion range calibration is not sufficient here.

## Iteration 16 — learn from optional candle and activity features (preregistered)

- Hypothesis: a learned model can use range/close discrepancies and changes in volume/trade intensity without forcing an inappropriate fixed range calibration.
- Change: add 19 range, coverage, volume, and trade-count features to the 30 existing inputs. Initialize from the close-only network with zero weights on the new inputs; preserve the original pre-July-1 training / July-1–25 selection split and 150-epoch cap. Compare full and half-strength auxiliary corrections on both development windows. Use the incumbent when auxiliary data are unusable.
- Result: pending. Optional-field availability is checked explicitly; no model may depend on interpolated or invented OHLCV values.
- Verdict: require out-of-sample official CRPS/emission gains, not just better training likelihood.
- Lesson: pending.

## Iteration 17 — observed atoms and price discreteness (preregistered)

- Hypothesis: a continuous diffusion misses the atom at zero in observed returns. The earlier empirical bootstrap subtracted its sample mean and normalized RMS, which both obscured this question.
- Change: test a symmetrized empirical step-return bootstrap normalized by mean absolute magnitude, a zero-inflated increment model using past valid zero-return frequency, and simulated-price quantization at the smallest reliable observed price increment. None treats NaN as zero or assumes a closed trading session. Keep the same neural scale and positive local volatility profile.
- Result: pending. A pre-field July-25 context has zero-return fractions (1m/5m) BTC 3.7%/0.2%, SOL 14.2%/6.4%, XAU 27.8%/11.3%, and SP500 12.2%/5.2%; these motivate a measured test, not a presumed gain.
- Verdict: retain only consistent official CRPS/reward improvements, with arbitrary path-count and positive-price contracts preserved.
- Lesson: pending.

### Historical-score availability safety check

One public score-only request explicitly bounded to July 1–3, 2026 returned HTTP 200 but failed the timestamp-bound validator. The response was rejected without storing or studying CRPS values; no out-of-window information was used. Do not repeat or broaden that probe. Continue exclusively with the frozen offline field bundle. The evaluator now refuses to run if `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is absent. This does not change any CRPS or reward formula.

### Full incumbent replay, first development window

The neural-plus-0.75-local-cycle model reaches official cumulative Score estimate 16.8173 on dev1, versus 0.32375 for the starter. Reward ranks / ratios to the field's best: COM_EQU_24H 174/257 and 0.0004981; CRYPTO_1H 136/257 and 0.368394; CRYPTO_24H 118/257 and 0.135626. This is substantial progress but not field leadership, especially for commodities/equities. Dev2 full replay is still running.

### Full incumbent replay, second development window

Dev2 finishes at Score estimate 16.1181 versus starter 0.42456. Reward ranks / ratios are COM_EQU_24H 231/257 and 0.00015596; CRYPTO_1H 198/257 and 0.446012; CRYPTO_24H 230/257 and 0.0373755. The model improves both regimes, but the remaining daily gap is large.

### Iteration 16 and 17 results

The auxiliary network does not improve the chronological likelihood diagnostic (best 1.019339 versus 1.019301 for close-only), and its official CRPS gains are tiny: full auxiliary daily ratios COM_EQU 0.99944/0.99990 and crypto 0.99975/0.99981. Keep it experimental, not deployed. Symmetric empirical bootstrap and zero inflation also fail the consistency gate. Price quantization has small consistent gains (hourly 0.99947/0.99911), so retain it only for later reward-based confirmation.

## Iteration 18 — align training with the actual prompt-score rule (preregistered)

- Hypothesis: the previous score-curve fit divided each prompt by its realized baseline CRPS, inadvertently giving different weights to different future outcomes. The actual pinned validator does not do that: valid prompt scores are raw CRPS minus the prompt minimum. The 95th percentile is used only to fill missed responses; valid scores are neither capped nor divided by a percentile range. Legacy helper names/comments are misleading, so the executed `compute_prompt_scores` source is decisive.
- Change: reuse the same official CRPS response curves, network, split, and optimizer, but weight raw scores by the validator asset coefficients and balance historical asset sample counts. Normalize only by a constant training-set scale per competition for numerical conditioning. This avoids outcome-dependent per-prompt normalization. No additional field history is needed, because a common prompt-minimum subtraction cancels in complete-coverage miner comparisons.
- Result: pending fresh official development-window CRPS and rewards. Compare full and half-strength corrections against a tuned incumbent with the promising half-strength hourly neural correction.
- Verdict: adopt only if both development windows and full reward replays support the change.
- Lesson: pending; this experiment corrects an objective mismatch, not the official scorer.

### Iteration 2 half-life result

The 1h half-life beats 6h for crypto-hourly in both development windows: ratios 0.9935 and 0.9870. The same change hurts daily crypto (1.0083/1.0049) and daily commodities/equities (1.0143/1.0094). Keep 1h half-life for hourly, 6h for daily. This is a horizon-specific result, not an asset-specific fitted exception. Next isolate scale and path-mixture effects around this `auto` baseline.

### Iteration 2 mixture result

At 256 paths, a five-degree Student variance mixture with unchanged RMS scale improves mean relative CRPS versus auto-Gaussian: COM_EQU_24H 0.9966/0.9876, CRYPTO_1H 0.9964/0.9986, CRYPTO_24H 0.9982/0.9953 (dev1/dev2). Increasing Gaussian scale hurts every competition; reducing by 10% has mixed gains. Keep df=5 provisionally, but the sub-0.5% hourly gains require denser and 1000-path confirmation. HYPE does not benefit on the pooled screen; do not introduce an asset exception without further evidence.

## Iteration 3 — sampling frequency (preregistered)

- Hypothesis: minute-scale microstructure and illiquidity can distort the diffusion estimate for daily five-minute paths. Estimating scale from actual 5/15/30-minute differences may better match scored horizons.
- Change: retain identical auto half-life and df=5 mixture; vary only the return lag used for scale estimation. Missing endpoint pairs are dropped, never filled or interpolated.
- Result: pending official CRPS screen on both development windows.
- Verdict: pending; require a consistent >0.5% gain or reject extra complexity.
- Lesson: pending.

### Iteration 1 result

Official CRPS screening uses 60 prompts per asset/format/window (2160 prompts total), 256 paths, five candidates. Six-hour absolute-return scale versus deterministic starter: mean asset-relative CRPS ratios are dev1/dev2 0.9750/0.9002 for COM_EQU_24H, 0.9729/0.9438 for CRYPTO_1H, and 0.9933/0.9761 for CRYPTO_24H. It is the best hourly variant for all five crypto assets in both windows. Keep robust recent scale as the next baseline. The windows differ materially in calibration improvement, although this alone does not establish opposite market regimes. Full official 1000-path field replays are starting; reduced-path averages are not emission estimates.

## Iteration 2 — scale and mixture calibration (preregistered)

- Hypothesis: absolute-return scaling still understates the peaked, heavy-tailed return distribution. A pathwise variance mixture can improve both near-zero sharpness and tail coverage without a directional forecast.
- Change: screen 1h/3h/6h/12h absolute-return half-lives, then multiplier and Student-t path-scale grids against av:360. Do not change multiple dimensions in the first half-life experiment.
- Result: pending; retain candidates only if both development windows improve by at least 0.5% or a competition improves without a >1% regression elsewhere.
- Verdict: pending.
- Lesson: pending.
