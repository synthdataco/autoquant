from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x=pd.read_csv('agent/results/full_asset_table.csv')
p=x.pivot(index=['configuration','asset','horizon'],columns='model',values='crps').reset_index();p['improvement']=100*(1-p.candidate/p.baseline)
fig,axes=plt.subplots(1,3,figsize=(13,5),gridspec_kw={'width_ratios':[1.35,1,1]})
for ax,c,color in zip(axes,['COM_EQU_24H','CRYPTO_24H','CRYPTO_1H'],['#167d8d','#3267ac','#7854a6']):
 g=p[p.configuration==c].sort_values('improvement')
 ax.barh(g.asset,g.improvement,color=color);ax.set_title(c);ax.set_xlabel('CRPS reduction versus starter (%)');ax.set_xlim(0,max(g.improvement)*1.22);ax.spines[['top','right']].set_visible(False)
 for i,val in enumerate(g.improvement):ax.text(val+.1,i,f'{val:.1f}%',va='center',fontsize=9)
fig.suptitle('Official replay: September 4–13 scored window, 1000 paths, complete realized coverage',fontsize=12)
fig.tight_layout();fig.savefig('agent/results/crps_improvement.png',dpi=180);plt.close(fig)
c=x.groupby(['configuration','model']).crps.mean().unstack()
summary='| Configuration | Starter CRPS | Champion CRPS | Reduction |\n|---|---:|---:|---:|\n'
for cfg in ['CRYPTO_24H','COM_EQU_24H','CRYPTO_1H']:
 row=c.loc[cfg];summary+=f'| {cfg} | {row.baseline:.4f} | {row.candidate:.4f} | {(1-row.candidate/row.baseline)*100:.2f}% |\n'
assets='| Configuration | Asset | Starter | Champion | Reduction |\n|---|---|---:|---:|---:|\n'
for row in p.itertuples():assets+=f'| {row.configuration} | {row.asset} | {row.baseline:.3f} | {row.candidate:.3f} | {row.improvement:.2f}% |\n'
report='''# Seasonal scale matters more than a more complicated price model

## Context

Synth evaluates distributions of simulated prices with CRPS: a forecast must be neither too concentrated nor too diffuse. We tested all three supported configurations and all13 active assets, ignoring the retired SPYX feed. The starter calibrated independent Gaussian log returns from seven days of minute-close volatility.

All market observations came from the frozen snapshot ending September14,2026. There was no online market-price download and no access to another miner's served paths. Historical score/reward APIs were used only to repair a missing comparison bundle, with bounded dates and immediate client-side filtering. Predictions always used context available before issuance.

## Approach

The champion is a small, offline model with three components:

1. **Seasonal scale.** Robust absolute returns estimate a half-hour-of-week activity profile, with shrinkage toward the same hour on weekdays or weekends. This captures the underlying cash-session pattern while recognizing that these perpetual venues trade around the clock.
2. **A heterogeneous autoregression of volatility.** A regularized regression predicts future mean absolute returns from past15-minute,1-hour,6-hour,1-day,3-day and7-day summaries. The target is one hour for minute paths and24hours for five-minute paths. This is a HAR-style model; its training loss is a surrogate, while official CRPS decides whether it is useful.
3. **Uncertainty in future scale.** Each path receives a shared lognormal scale multiplier with log standard deviation0.4 and mean1. Conditional innovations are Gaussian and antithetic. This gives a sharper center and broader tails than a single Gaussian forecast without adding a fitted directional bet.

Missing prices remain missing. If a short feature window has no observed returns, the model uses a longer observed return scale. An entirely unobserved context cannot supply a starting price and raises a clear error. Every path begins at the last positive observed close, and the model honors the requested path count and grid. All simulation randomness derives from `SYNTH_BENCHMARK_SEED`.

The serving artifact registry contains dated checkpoints through July20,July31,August31 andSeptember14. A checkpoint is eligible only after its last training minute has closed and before the prompt's issuance time. The final refresh is therefore unavailable at September15 midnight and first becomes eligible at00:01 for1h and00:02 for24h. September historical replay selects the August31 checkpoint; no historical prediction uses the September14 fit.

## Experiments, including failures

Initial development used August1–7 andAugust17–23, with28 prompts per asset-format per week. Baseline crypto24h CRPS was more than twice as high in the second week, providing a useful regime contrast. An unused August8–16 interval added72 prompts per asset-format. A separate September4–13 confirmation used80 prompts per asset-format. All reported model comparisons use1000 paths and identical realized observations.

The largest improvement came from modeling session timing for commodities and equities. Five-minute resampling alone explained little of it. Absolute-return scale estimates were less sensitive to isolated jumps, and a learned volatility forecast worked better when paired with uncertainty in future scale. Half-hour profiles improved on hourly profiles; quarter-hour profiles did not help reliably.

The following alternatives failed prespecified promotion thresholds:

- Faster adaptation and blanket volatility shrinkage helped calm conditions but sometimes lost ground in the busier week.
- Learned return drift had positive training R² but worsened held-out CRPS.
- Tree models and a small, GPU-trained neural residual ensemble reduced fitting error without reliable CRPS gains; their busy-window crypto performance was especially weak.
- Extra range/volume features, blockwise future-volatility forecasts, empirical innovations and residual block resampling offered only small or inconsistent gains.
- Fitted return dependence appeared slightly helpful at one seed but reversed with additional seeds. Wider volatility mixtures also lost precision.
- A direct official-CRPS calibration grid on a separate July training week selected values near the existing settings. Its small apparent optimum did not transfer consistently to August.
- An exploratory FOMC-calendar adjustment was not robust across assets and multiplier sizes. It is not in the champion.
- Adding older2025 crypto history or excluding incomplete final candles did not reliably improve validation.

The journal and prompt-level CSV files preserve the hypotheses, numbers and keep/discard decisions. No field-relative rank was optimized.

## Results

### Full official replay

The table below uses `synth_lib.backtester.orchestration.backtest`, exact validator prompt timestamps,1000 paths, and100% realized coverage. The scored-time window is September4 throughSeptember13,2026;24h prompts start one day earlier. There are1726 commodity/equity,1080 crypto24h and6480 crypto1h prompts per model. Each configuration value is the equal-weight average of its assets' mean CRPS.

'''+summary+'''
Every asset-format improved. Coefficient-weighted CRPS reductions are13.27% for commodities/equities,3.61% for crypto24h and5.32% for crypto1h. Those are weighted CRPS comparisons, not reward or rank claims.

![Per-asset official CRPS improvement](results/crps_improvement.png)

'''+assets+'''
### Current-rule replay and chronology

The pinned library activates the1h volatility component on September22, after the observation cutoff. Full historical orchestration above correctly uses the rules active on the replayed dates. Separately, the official backtester scorer applied the current published formula to historical observations, without reading future prices. On the ten-day3h-grid confirmation, starter versus August31-fit champion CRPS was:

| Configuration | Starter | Champion |
|---|---:|---:|
| CRYPTO_24H |3561.5627|3399.9605|
| COM_EQU_24H |1920.4177|1646.3891|
| CRYPTO_1H, including volatility |853.3771|805.9141|

Grid and exact-prompt results should not be compared directly: their prompt samples differ, and the grid's mean realized coverage was98.8718%. Within each comparison, coverage is identical. The official metric was never reimplemented.

Refreshing the same recipe from July31 toAugust31 improved commodity/equity CRPS slightly but worsened crypto by about0.25–0.32% on the grid. Both checkpoints beat the starter broadly. More recent fitting is not a guaranteed improvement. The September14 refresh has passed contract and latency checks but cannot have a post-cutoff CRPS measurement within this campaign's permitted data.

Serving checks across all18 asset-formats generated1000 valid paths in about0.02–0.04seconds per prompt on this hardware, well below the practical5-second ceiling. Seed reproducibility, arbitrary path counts, raw NaN gaps, a single observed close, and artifact-availability boundaries were checked. Representative September outputs from the simplified serving code were bit-for-bit equal to the validated research implementation.

## Limitations

Overlapping prompts are not independent observations, and one September window cannot establish performance in all regimes. Day-cluster bootstrap checks support the broad baseline improvements, but not tiny differences among near-tied variants. Hyperparameter search across many families can still overfit; this is why weak gains and single-seed effects were rejected.

The seasonal profile is venue-specific and can miss holidays, listing changes and unusual events. Early thin-market data are not equivalent to liquid recent trading. The final model does not model cross-asset dependence, because its serving input contains only one asset's context. It also does not claim that log-price median centering makes arithmetic prices a strict martingale.

The report measures historical CRPS, not the unseen September15–24 evaluation outcome. The newest artifact refresh is an application of the validated training recipe, not an independently scored future result.

## What we would do with more budget

The most valuable extension would be additional independent regimes and event-level holdouts. The tested complexity families have reached a validation plateau; more tuning on these same weeks risks fitting noise. A future campaign could examine a genuinely different information source available at issuance, or study scheduled-event volatility on a larger, precommitted event sample. The remaining compute budget alone is not evidence that another layer or wider parameter grid will improve expected CRPS.
'''
# Keep published prose readable when generated numeric tokens meet words.
import re
report=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', report)
report=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', report)
report=report.replace('CRYPTO_24 H','CRYPTO_24H').replace('COM_EQU_24 H','COM_EQU_24H').replace('CRYPTO_1 H','CRYPTO_1H').replace('SP 500','SP500')
report=re.sub(r',(?=\S)', ', ', report)
report=report.replace('andSeptember','and September').replace('andAugust','and August').replace('throughSeptember','through September').replace('toAugust','to August')
Path('agent/report.md').write_text(report)
