# Better volatility forecasts are not the same as better miner rewards

**Final outcome:** the frozen model scores **19.802644** on the reserved historical validation, versus **45.095169** on explicitly warmed development. It supports both required formats and passes offline/interface audits, but its development reward level does not transfer to the reserved window. No model changes follow the final result.

## Context

Synth SN50 asks a miner for a distribution of future price paths, not a single price target. This campaign covers all three active competitions: five crypto assets at one-minute resolution for one hour; the same assets at five-minute resolution for one day; and eight commodity/equity assets at five-minute resolution for one day. SPYX is retired and is neither trained as an active asset nor evaluated.

The validator requests **1000 paths**. Each includes its observed starting price: 61 points hourly or 289 daily. The implementation also honors other requested positive path counts. All market data comes from the frozen venue-specific snapshot with research cutoff **August 29, 2026**. No subsequent prices, scores or rewards are used. No other miner's served predictions or operator-private research is examined.

CRPS and emissions are different objectives. The validator converts raw CRPS to prompt scores, applies asset coefficients, smooths each miner's scores, normalizes and applies a softmax. In the pinned implementation, valid prompt scores are raw CRPS minus the prompt minimum; the percentile statistic fills missed responses, not a percentile-range division of every valid response. An unweighted mean of asset CRPS ratios is therefore a screening diagnostic, not an emissions estimate.

The campaign's Score is **100 times the mean across competitions of min(1, our cumulative reward weight / the best other miner's cumulative reward weight)**. Each competition contributes one third. Every reported CRPS and reward replay uses the official backtester through `agent/predict.py`, never a replacement scorer. Sources of truth are synth-lib commit `782a44fe7585a4e7b0388e22b99c57f13d56f5ee` and its synth-subnet dependency `0d014cbcec376b28e887403b84025ebf1a89f60e`.

## Approach

### Data and chronology

Context loading reads existing snapshot partitions, preserves missing observations and checks coverage before evaluation. Serving drops nonfinite/nonpositive closes, excludes future rows and observations outside the preceding seven days, and never interpolates prices. Returns spanning observation gaps are normalized by the square root of elapsed minutes. Auxiliary OHLCV columns are not required by the champion.

The initial static model has 23,457 training contexts before July 1 and 2,508 July 1–25 chronological diagnostic contexts. Later models have their own recorded row counts. All deployed fitted targets end by **July 26, 2026, 00:00 UTC**, and learned weights activate no earlier than that cutoff. Earlier calls use a context-only fallback. Chronological diagnostics are model-selection tools, not claims of unbiased historical performance.

The offline field archive starts around July 27. Development uses scoring-time windows July 28–August 7 and August 8–18; August 18–28 is reserved for final validation. All window ends are exclusive. The development regimes differ in realized volatility: average daily RMS for BTC is approximately 1.50% versus 0.96%, ETH 2.06% versus 1.28%, SP500 0.71% versus 0.44%, and AAPLX 2.12% versus 0.96%. These are more-volatile and calmer periods, not labels for bull and bear markets.

### Frozen serving model

The model is frozen at commit **`9bc60e9adff582d9d58a223ecaff6396aec23e2e`** before examining final validation outcomes. Its components are:

1. **Robust recent scale.** Exponentially weighted absolute returns, with a one-hour half-life hourly and a six-hour half-life daily, supply the base volatility.
2. **Small learned corrections.** A 48/32-unit static network uses 30 volatility/asset features, at half strength hourly and full strength daily. A temporal convolutional residual uses 168 hourly bins of absolute returns, RMS returns, signed returns and observed coverage. Daily forecasts additionally use a numerical tree student distilled from our own pretrained-feature teacher.
3. **Positive daily timing profiles.** A seven-day empirical clock uses 30-minute bins at 0.75 strength for crypto and 15-minute bins at full strength for commodities/equities. There are no opening/closing gates or zero-volatility sessions. A five-bucket lead model, corrected by frozen temporal features, redistributes volatility through the day. Commodities/equities also use a confirmed nonlinear lead encoder and a half-strength smooth periodic *level* adjustment.
4. **Finite-fan construction.** Antithetic scrambled Sobol draws initialize a five-degree Student scale mixture. Daily fans receive exactly two fine-to-coarse projection sweeps over 1/6/36/288-step blocks. Each sweep rank-remaps block sums to midpoint Student quantiles and allocates adjustments across constituent increments in proportion to forecast variance. Hourly fans instead stratify cumulative log-price deviations.
5. **Daily crypto centering and price lattice.** Daily crypto removes each horizon's finite-fan log-mean-exp shift before exponentiation; this flattens unconditional price means before safety clipping and rounding, not conditional expected returns. All formats project onto a plausible GCD-derived historical price grid, anchored at the last observed close. The initial point remains unchanged.

There is no learned directional price predictor. Clock and lead components have unit mean separately, but their product generally does not. Explicitly forcing their joint mean to one was tested and rejected rather than silently changing the confirmed implementation.

Serving is self-contained NumPy/pandas/SciPy code. It does not require Torch, scikit-learn, a pretrained-model runtime, a GPU or network access. Its seven required archives are `volatility`, `temporal_exact`, `lead`, `lead_temporal`, `tree_student`, `periodic` and `lead_finetune`, each ending in `_weights.npz`. The final contract audit records their hashes. Other research archives in the repository are not serving dependencies.

### Why the sampler matters

For a fixed continuous predictive distribution with a finite first moment, equally weighted midpoint quantiles minimize expected empirical-ensemble CRPS when observations follow that distribution. Zamo and Naveau (2018), *Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts*, Mathematical Geosciences, DOI `10.1007/s11004-017-9709-7`, discuss this distinction in Section 3.2.1 and attribute the optimal-quantile result to Bröcker (2012).

That univariate fact does **not** establish that our joint path transformation improves Synth's multi-interval objective. Correcting price-level marginals can disturb increment marginals, and vice versa. Daily prefix remapping fails consistency; daily multiscale and hourly prefix remapping pass actual official scoring and rewards. Two projection sweeps do not make every overlapping marginal exact. The quantile grid, sweep count and ordering are not extended after inspecting their field results.

## Experiments

The journal preserves hypotheses, preregistered gates, numerical outcomes and keep/discard decisions. The most useful findings are:

| Experiment | Outcome |
|---|---|
| Recent RMS versus absolute-return scale | Robust recent scale wins; the seven-day starter adapts too slowly. |
| Horizon-specific memory | Hourly and daily forecasts need different memories. |
| Gaussian, Student mixtures, IID tails and stochastic volatility | A modest Student mixture wins; the tested added dynamics do not transfer consistently. |
| Static and exact-context temporal networks | Retained at separately confirmed strengths. |
| Five-minute targets, range/OHLCV substitutions and auxiliary networks | No consistently superior deployed alternative. |
| Directional ridge forecasts and serial-correlation adjustments | Rejected for negligible or regime-dependent gains. |
| Official-score-curve fine-tuning, moment recalibration and Newton corrections | Better internal objectives do not reliably produce better development rewards. |
| Positive empirical clock and temporal lead corrections | Retained after dense CRPS and full rewards. |
| Nonlinear temporal lead encoder | Retained for COM_EQU; rejected for crypto. |
| Pretrained-feature trees and numerical distillation | Daily gains survive distillation and joint reward confirmation; hourly use is rejected. |
| Finite Student-scale persistence | Tiny first-window gains reverse in the second window. |
| Quote-activity residual | Added-feature likelihood gain 0.000053 misses its 0.0001 gate; no field scoring. |
| Randomized training clocks | Both shifted and aligned fits improve mean crypto CRPS, but both lose first-window rewards. Neither is deployed. |
| Fast 5/15/30-minute scale features | Gain 0.000366 beyond the matched control misses its 0.0005 gate; no field scoring. |
| Extra uncertainty for a stale last close | COM_EQU ratios approximately 1.000000/1.000005 fail; sampled crypto has too few affected prompts. |
| Multiscale daily and prefix hourly stratification | Retained after dense and explicitly warmed reward confirmation. |
| Unconditional fan-mean centering | Retained only for daily crypto; COM_EQU loses and hourly reverses across windows. |

Distillation reduces the numerical student's archive to about 1.7 MiB while avoiding the pretrained serving pipeline; the experiment's teacher-feature cache is about 183 MiB. On 6,240 daily prompts, student/teacher CRPS ratios are 1.000053/1.000008. The historical concurrent-load generation medians are 0.0645 versus 0.0750 seconds: an artifact/dependency reduction, not an order-of-magnitude latency claim.

The emissions counterexamples are important. The shifted-clock model improves daily crypto mean CRPS by approximately 0.017%/0.046%, yet its first-window reward ratio falls from 0.395023 to 0.352926. A tiny positive weekly-shape crypto CRPS result likewise loses first-window rewards. We do not salvage individual assets after failed group gates.

Two ideas remain **unconfirmed and excluded** at landing. A smooth weekly *lead-shape* correction passes separate COM_EQU rewards and dense interaction CRPS, with multiscale-relative ratios 0.999970/0.999883, but lacks joint reward confirmation. A nine-feature multi-minute Newton ablation improves actual chronological COM_EQU diagnostic CRPS to 0.999072 versus identity and 0.999714 versus its matched control, but has no development-field confirmation; crypto fails identity. Neither is represented as a champion improvement.

Implementation failures are also recorded. Microsecond timestamps were initially mistaken for nanoseconds during label building; distribution checks caught this before affected learned candidates were evaluated, and the labels were rebuilt. A later interface audit finds and fixes NumPy integer Unix seconds being interpreted as nanoseconds. Exact serving/research comparison also catches a float32/float64 promotion issue. These checks matter when one-tick differences can affect ranking.

## Results

### Development and warmup

The starter's full 1000-path short-window Score estimates are **0.32375** and **0.42456**. Under the same first-window protocol, the frozen model reaches **51.544567**:

| Competition | Dev1 reward rank | Reward / best field reward |
|---|---:|---:|
| COM_EQU_24H | 108 / 257 | 0.0622061 |
| CRYPTO_1H | 21 / 257 | 0.9719845 |
| CRYPTO_24H | 32 / 257 | 0.5121465 |

This is a **selection proxy**, not a clean steady-state estimate. The pinned validator uses ten-day daily and five-day hourly smoothing, not the campaign prose's 24 hours. Its warmup trimmer is a no-op when trimming would remove every round. Short daily runs can therefore look complete without actually removing warmup.

With ten explicit prior simulation days and rewards restricted to August 8–18, the frozen model gives **45.095169**:

| Competition | Reward rank | Reward / best field reward | Reward rounds |
|---|---:|---:|---:|
| COM_EQU_24H | 161 / 252 | 0.0510693 | 1,469 |
| CRYPTO_1H | 9 / 257 | 0.9206018 | 3,400 |
| CRYPTO_24H | 61 / 257 | 0.3811840 | 1,043 |

The retained intervals are August 8 00:36–August 17 23:56 UTC for COM_EQU, August 8 00:08–August 17 23:57 for hourly crypto, and August 8 00:48–August 17 23:31 for daily crypto. An earlier deployed stack gives 42.803008 on this exact warmed protocol. Reconstruction from its saved official CRPS reproduces all three baseline ratios, ranks and intervals exactly before candidate comparison. Different warmup protocols and active field counts must not be averaged or silently substituted for one another.

### Reserved final validation

The frozen model's reserved August 18–28, 2026 scoring-time replay gives **19.802644**, using 1000 paths and ten prior simulation days:

| Competition | Cumulative reward rank | Reward / best field reward | Reward rounds | Final-round rank |
|---|---:|---:|---:|---:|
| COM_EQU_24H | 162 / 257 | 0.0286420 | 1,529 | 165 |
| CRYPTO_1H | 42 / 257 | 0.4601913 | 3,435 | 135 |
| CRYPTO_24H | 47 / 257 | 0.1052461 | 1,034 | 38 |

Retained reward timestamps are August 18 00:18–August 27 23:54 UTC for COM_EQU, August 18 00:03–August 27 23:58 for hourly crypto, and August 18 00:31–August 27 23:58 for daily crypto. The machine-readable result is `agent/results/frozen_final/final/competition_metrics.json`.

This is substantially weaker than the warmed development estimate of 45.095169. Every competition's reward ratio declines. Daily crypto even improves its ordinal cumulative rank from 61 to 47 while its reward ratio falls from 0.381184 to 0.105246: ranks alone are not comparable measures of reward share across windows. The experiment does not separate development-selection optimism, market-regime changes and changes in field performance. It therefore does not establish robust transfer of the development reward level. We preserve the frozen model rather than tune or select another candidate on this result.

### Interface and reproducibility

Before final evaluation, all 18 active asset/format combinations pass serving/research equivalence at 17 and 1000 paths, relative tolerance 1e-12. Daily crypto additionally passes 45 comparisons across three dates and 1/17/1000 paths. Audits cover NaN-contaminated frames, missing auxiliary fields, future-row exclusion, unsorted inputs, constant/single-close contexts, zero horizon, pre-weight fallback and equivalent timestamp representations. A context without any finite positive close raises a clear error rather than inventing a starting price.

Additional synthetic interface fixtures pass at 1/2/17/999/1000/1001 paths. Import and simulation pass with network access blocked, without importing Torch, scikit-learn or Chronos. Microsecond/nanosecond index representations agree, global RNG changes do not alter predictions, and the context-only fallback does not load fitted artifacts. The frozen source and all seven required archive hashes remain unchanged after final evaluation.

Run evaluation from the workspace root, using the pinned environment and offline score bundle. The final command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run agent/field_eval.py \
  --name frozen_final --windows final --warmup-days 10 \
  --score-start 2026-08-18 --workers 4
```

The numerical serving code and seven required weight archives are frozen before this command. Evaluation metadata and reports may be completed afterward without modifying the model.

## Limitations

- The effective sample is days and market regimes, not thousands of independent overlapping forecasts. Repeated development screens create selection risk.
- Windows are indexed by scoring time. Daily paths at boundaries overlap adjacent days, and warmed reward smoothing deliberately includes earlier history; the reserved reward window is not ten independent days isolated from development observations.
- Internal chronological diagnostics are repeatedly used for model selection; some daily targets overlap split boundaries. They are not causal field backtests. Deployed fitting labels end before field-scored predictions.
- Likelihood and local quadratic score surrogates are imperfect. Actual official rescoring and full rewards repeatedly overturn attractive internal gains.
- COM_EQU remains the weakest competition. This univariate-context interface does not reproduce cross-asset information, news or other inputs available to some live miners.
- The official backtester may fill NaN-holed realized paths from archived validator truth. We preserve that behavior and do not invent missing market observations.
- A bounded score-only API probe returned out-of-request timestamps and was rejected before score analysis or storage. Evaluation is strictly offline and fails closed without its offline-bundle setting.
- All numerical Scores in this report are historical estimates, **not the campaign's eventual post-cutoff Score or a guarantee of future emissions**.

## What we would do with more budget

First finish reward confirmation of the already specified COM_EQU weekly-shape interaction and development confirmation of the controlled multi-minute calibration. Then test their interaction with the frozen model, without silently combining individually promising components. A genuinely new untouched chronological window would be needed for further selection after final validation.

Infrastructure improvements are equally valuable: make smoothing/warmup semantics explicit, distinguish simulation and reporting intervals, enforce client-side historical bounds, and ship raw-NaN and timestamp-type contract checks. The landing order stops new research; unconfirmed candidates remain excluded rather than being promoted on internal metrics alone.
