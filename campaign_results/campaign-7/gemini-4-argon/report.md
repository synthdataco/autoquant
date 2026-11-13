# Synth (Bittensor SN50) Probabilistic Price Forecasting — Research Campaign Report

## 1. Context & Problem Formulation

In the **Synth (Bittensor Subnet 50)** probabilistic forecasting competition, miners generate an ensemble of $M = 1{,}000$ simulated future price paths $P^{(m)}_{0:T}$ over a horizon $T$ at discrete step increments $\Delta t$ for 13 assets across three active scoring configurations:

1. **`CRYPTO_24H`** (`BTC`, `ETH`, `SOL`, `XRP`, `HYPE`): $T = 86{,}400\text{ s}$ (24 hours), $\Delta t = 300\text{ s}$ (5 minutes, 289 nodes per path), scored every 60 minutes across `5min`, `30min`, `3hour`, and `24hour_abs` basis-point change intervals.
2. **`COM_EQU_24H`** (`XAU`, `NVDAX`, `GOOGLX`, `TSLAX`, `AAPLX`, `WTIOIL`, `SPCX`, `SP500`): $T = 86{,}400\text{ s}$ (24 hours), $\Delta t = 300\text{ s}$ (5 minutes, 289 nodes per path), scored every 60 minutes across `5min`, `30min`, `3hour`, and `24hour_abs` basis-point change intervals.
3. **`CRYPTO_1H`** (`BTC`, `ETH`, `SOL`, `XRP`, `HYPE`): $T = 3{,}600\text{ s}$ (1 hour), $\Delta t = 60\text{ s}$ (1 minute, 61 nodes per path), scored every 12 minutes across `1min`, `5min`, `15min`, `30min`, and `60min_abs` basis-point change intervals (using the active `HIGH_FREQUENCY` post-`2026-09-22` interval set).

### What the Multi-Horizon CRPS Objective Actually Rewards

For any target change or terminal level $y$ and empirical ensemble $\{x_1, \dots, x_M\}$, the Continuous Ranked Probability Score is:
$$\operatorname{CRPS}(F_M, y) = \frac{1}{M}\sum_{m=1}^{M} |x_m - y| - \frac{1}{2M^2}\sum_{m=1}^{M}\sum_{m'=1}^{M} |x_m - x_{m'}|$$
Minimizing summed CRPS across all sub-intervals simultaneously imposes four strict mathematical requirements on the path generator:
1. **Exact $L_1$ Dispersion Calibration ($\mathbb{E}|X - X'|$ vs. $\sigma_2$)**: Unlike log-likelihood or MSE (which depend on second moments $\mathbb{E}[X^2]$), CRPS trades off absolute forecast error $\mathbb{E}|X - y|$ directly against ensemble $L_1$ spread $\frac{1}{2}\mathbb{E}|X - X'|$. If heavy-tailed shocks (e.g., Student-$t$) are normalized to unit variance ($\mathbb{E}[Z^2] = 1$) instead of matching the Gaussian $L_1$ norm ($\mathbb{E}|Z| = \sqrt{2/\pi}$), 1-step increments suffer systematic under-dispersion in the core while multi-step sums (which converge toward Gaussianity via the Central Limit Theorem) become over-dispersed.
2. **Martingale Symmetry & Zero Ensemble Drift**: Over 1-minute to 24-hour horizons, predictable mean returns are orders of magnitude smaller than return volatility ($\text{SNR} \ll 0.02$). Any nonzero sample mean drift in the ensemble inflates $\mathbb{E}|X - y|$ linearly without increasing $\mathbb{E}|X - X'|$, strictly degrading CRPS.
3. **Time-of-Day, Day-of-Week, and Scheduled-Event Clock Alignment**: Because CRPS is evaluated on every `5min` (or `1min`), `30min`, and `3hour` increment throughout the 24-hour trajectory, an ensemble with the right 24-hour total variance but a flat intraday volatility profile gets heavily penalized during both quiet periods (overnight/weekends on `COM_EQU_24H`, Asian afternoon lull on crypto) and high-volatility windows (09:30–10:30 NY equity open, Sunday 18:00 NY CME/Globex open, FOMC/CPI/NFP/EIA releases, and after-hours corporate earnings).
4. **Multi-Horizon Scaling & Stochastic Volatility Clustering**: Under an independent-increment random walk, $k$-step variance equals $k \sigma_1^2$. In real markets, intraday returns exhibit two competing effects: slight negative autocorrelation / mean reversion at 5m–3h horizons ($VR(k) = \operatorname{Var}(r_{t:t+k}) / (k \operatorname{Var}(r_1)) \approx 0.85\text{--}0.95$) alongside fat-tailed **volatility clustering** over 3h–24h horizons (`24hour_abs`). Simultaneously matching `5min` increments and `24hour_abs` terminal distributions requires combining multi-horizon variance-ratio scaling with path- and block-level stochastic volatility.

### Diagnosing the Starter Baseline (`Iteration 0`)

The starter model in `agent/modeling.py` (`CONTEXT_DAYS = 7`, flat Student-$t$ random walk using raw 5-minute returns over 7 days) suffered from five structural bottlenecks:
1. **Wrong Volatility Clock (`COM_EQU_24H`)**: Computing a flat standard deviation across 7 days averages Regular Trading Hours (RTH, `09:30–16:00 NY`) with overnight and 65-hour weekend dead zones. Consequently, the baseline over-predicted weekend and overnight volatility by $3\times\text{--}8\times$ while under-predicting the `09:30 NY` equity open by $2.5\times$.
2. **Weekend Amnesia in Calendar-Time EWMAs**: Standard calendar-time EWMAs with 1h–24h half-lives decay to near-zero over the 65-hour Friday-to-Monday equity weekend, causing Monday morning (`09:30 NY`) forecasts to under-predict RTH volatility until several hours after Monday's open.
3. **Outlier Contamination of Seasonality & Scale**: Single macro jumps or corporate earnings gaps (`NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`, `SPCX`) inflated trailing variance estimates for days after the event, while the model remained completely blind to upcoming scheduled events in `$SYNTH_BENCHMARK_CALENDAR`.
4. **Unpaired Monte Carlo Sampling Noise**: Independent Student-$t$ draws at $M = 1{,}000$ paths exhibit non-trivial sample mean drift and sample dispersion variance across steps.
5. **1-Minute Sampling Misalignment**: Taking `diff()` of `pct_change(step_min)` without contiguous timestamp alignment diluted step variance whenever missing data or non-contiguous bars occurred.

---

## 2. Architecture of the Champion Model (`Iteration 7`)

Our champion model in [`modeling.py`](file:///workspace/agent/modeling.py) (`CONTEXT_DAYS = 28`, runtime $\sim 0.08\text{ s}$ per 24H call and $\sim 0.02\text{ s}$ per 1H call for $1{,}000$ paths on CPU) is a **zero-training-loop, strictly causal econometric state-space simulator** composed of six tightly integrated modules:

```
context_prices (28d 1m closes)
   │
   ├─► 1. Strict NaN-Drop & Gap-Normalized Step Returns (5m & step_min grids)
   │
   ├─► 2. US-DST-Aware America/New_York 7x48 Seasonality Surface
   │      • Centered 7-day rolling L1 normalization
   │      • Within-slot 96th-percentile clipped L1 scale (_robust_slot_l1)
   │      • 48 weekday half-hour slots + 5-regime weekend curve (Sat/Sun/Globex open)
   │      • Empirical shrinkage toward pre-Aug-2026 asset priors (PRIOR_NY_SEASONALITY)
   │
   ├─► 3. Dual-Role Scheduled Event Calendar ($SYNTH_BENCHMARK_CALENDAR)
   │      • Historical damping (is_ewma_hist=True): damps discrete jump spikes only
   │      • Forward projection (is_ewma_hist=False): full multi-stage trajectory
   │        for FOMC, CPI, PPI, NFP, NYSE_CLOSED/EARLY_CLOSE, and EARNINGS (-390m to +1380m)
   │
   ├─► 4. Intrinsic Activity-Clock GLS HAR-EWMA Volatility Term Structure
   │      • Precision weights act_w = min(1, (s_ny / 0.85)^2)
   │      • Activity-clock time step d_tau = (1 - btau) + btau * (w / mean(w))
   │      • 7 horizons (15m, 1h, 6h, 24h, 3d, 7d, 28d) + 5-phase overlapping 60m
   │      • 24/7 crypto unweighted short-horizon blend (v_15m_raw, v_1h_raw)
   │
   ├─► 5. Multi-Horizon Variance Ratio (vr_scale), Tail Shrinkage & Breakout Filter
   │      • Empirical VR(30m, 3h) for 24H and VR(5m, 15m) for 1H
   │      • L1/L2 heavy-tail shrinkage + (0.35*v_24h + 0.65*v_3d)/v_28d breakout
   │
   └─► 6. Paired Antithetic L1-Normalized Student-t(ν=6) + Path/Block SV
          • Exact L1-matched Student-t quantiles: E|q| = sqrt(2/pi)
          • Antithetic pairs (z_i, -z_i) share identical path (p_mult) & block (b_mult) SV
          • Exact martingale drift correction: -0.5 * step_sig^2
```

### 2.1 Strict `NaN` Dropping & Gap-Normalized Step Return Extraction
In [`_extract_closes_and_timestamps`](file:///workspace/agent/modeling.py#L440-L463) and [`simulate`](file:///workspace/agent/modeling.py#L466-L520), `NaN` or non-positive prices in `context_prices` are strictly dropped (`raw_closes[finite_mask]`, `ts_sec[finite_mask]`) without ever interpolating or backfilling (`.ffill()` / `.bfill()`). If an asset has fewer than 28 days of history (leading `NaN`s) or interior `NaN` gaps, dropping `NaN`s prevents synthetic zero returns from diluting historical volatility, while any step return spanning a dropped `NaN` timestamp gap $\Delta t > \Delta t_{\text{step}}$ is normalized by $\sqrt{\Delta t_{\text{step}} / \Delta t}$.

### 2.2 US-DST-Aware `America/New_York` Seasonality with Within-Slot Clipped $L_1$ Scale
Financial market liquidity and volatility follow the **`America/New_York` wall clock** (shifting by 1 hour relative to UTC on the second Sunday of March and the first Sunday of November). Our vectorized [`_utc_to_ny_dow_slot`](file:///workspace/agent/modeling.py#L221-L246) maps any array of Unix timestamps into New York day-of-week `dow` $\in \{0,\dots,6\}$ and half-hour `slot` $\in \{0,\dots,47\}$ in microseconds.

A critical discovery of our campaign (Iteration 6) is that **global-threshold $L_2$ RMS seasonality estimation creates severe diurnal selection bias**. In tokenized equities and commodities (`COM_EQU_24H`), RTH open/close slots have $5\times\text{--}10\times$ higher volatility than overnight or Saturday slots. Applying a global outlier threshold (such as $4.5 \times \text{rolling RMS}$ or the global 99.5th percentile) clips regular RTH open bars at $\sim 2.1\sigma_{\text{RTH}}$ while leaving weekend/overnight glitches unclipped up to $20\sigma_{\text{Sat}}$, flattening the estimated diurnal ratio by $25\text{--}40\%$.

Instead, both our offline pre-August-2026 prior table [`PRIOR_NY_SEASONALITY`](file:///workspace/agent/modeling.py#L31-L163) and our online adaptive estimator [`_build_ny_seasonality_table`](file:///workspace/agent/modeling.py#L343-L447) use:
1. **Centered 7-Day Rolling $L_1$ Local Regime Normalization**: Each 5-minute return is divided by its centered 7-day (`win = 2016` five-minute bars) rolling mean absolute deviation ($\times \sqrt{\pi/2}$) after damping known scheduled event jumps. Because a 7-day window always contains exactly 5 weekdays and 2 weekend days, it never drifts across the weekly cycle.
2. **Within-Slot 96th-Percentile Clipped $L_1$ Scale ([`_robust_slot_l1`](file:///workspace/agent/modeling.py#L326-L340))**: Within every individual half-hour slot (and each of the 5 weekend regimes: Saturday `00:00–12:00`, Saturday `12:00–24:00`, Sunday `00:00–12:00`, Sunday `12:00–18:00`, and Sunday `18:00–24:00` NY Globex/futures open), values above that specific slot's 96th percentile are winsorized before computing the Gaussian-consistent $L_1$ scale $\sqrt{\pi/2}\,\mathbb{E}|u|$.
3. **Bayesian Shrinkage to Pre-Cutoff Asset Priors**: The online 28-day seasonality curve is circularly smoothed (`[0.20, 0.60, 0.20]`) and shrunk (`w_emp = n_days / (n_days + 14.0)`) toward `PRIOR_NY_SEASONALITY[asset]`.

### 2.3 Dual-Role Scheduled-Event Multipliers (`$SYNTH_BENCHMARK_CALENDAR`)
Using the official event calendar (`FOMC`, `CPI`, `PPI`, `NFP`, `NYSE_CLOSED`, `NYSE_EARLY_CLOSE`, and corporate `EARNINGS` for `NVDAX`, `TSLAX`, `GOOGLX`, `AAPLX`, `SPCX`), [`_get_event_multipliers(asset, ts_sec, is_ewma_hist)`](file:///workspace/agent/modeling.py#L144-L223) separates event handling into two distinct econometric roles (Iteration 7):
- **Role 1 — Historical EWMA & Seasonality Deseasonalization (`is_ewma_hist=True`)**: When deseasonalizing historical returns (`r_deseas = r_grid / (s_ny * ev_hist)`), we damp only the immediate discrete announcement spike (`-30..240m` after-hours jump `1.85x / 3.80x / 4.30x / 2.60x` and next-day `09:00–11:00 NY` opening jump `1.60x / 1.38x` for earnings). Crucially, we do **not** divide daytime pre-earnings or post-earnings RTH returns by large multipliers in `ev_hist`, allowing genuine post-event market activity to feed naturally into `v_1h`, `v_6h`, and `v_24h`.
- **Role 2 — Forward Volatility Trajectory Projection (`is_ewma_hist=False`)**: When projecting future step volatility (`s_fut = s_table[dow_fut, slot_fut] * ev_fut`), we apply the full pre-August-2026 empirical multi-stage trajectory relative to the outlier-cleaned diurnal prior: for `EARNINGS` (which occur at `16:00 NY` after-hours close, where normal `s_table` is only `~0.42`), this spans pre-release RTH (`-390..-30m`: `1.20x`), immediate pre-close (`-30..0m`: `2.20x`), after-hours release (`0..30m`: `8.50x`, `30..90m`: `9.00x`, `90..150m`: `5.50x`, `150..240m`: `2.60x`), overnight consolidation (`240..900m`: `1.35x`), next-day pre-market (`900..1050m`: `1.45x`), next-day RTH open (`1050..1140m`: `1.45x`), and next-day RTH follow-through (`1140..1380m`: `1.15x`).

### 2.4 Intrinsic Activity-Clock GLS HAR-EWMA & 24/7 Crypto Fast Horizons
In [`_weighted_ewma_rms`](file:///workspace/agent/modeling.py#L249-L266), each deseasonalized squared return $r_{\text{deseas},t}^2$ receives a Generalized Least Squares (GLS) precision weight $w_t = \min(1, (s_{\text{NY},t} / 0.85)^2)$ so noisy low-liquidity overnight/weekend ticks do not corrupt the baseline volatility state.

To prevent the EWMA from forgetting Friday's RTH volatility over the 65-hour equity/commodity weekend, we advance the EWMA clock in **normalized activity time**:
$$\Delta \tau_t = (1 - \beta_\tau) + \beta_\tau \frac{w_t}{\bar{w}}, \qquad \alpha_t = 1 - \exp(-\ln(2)\,\Delta \tau_t / H)$$
with $\beta_\tau = 0.75$ for `COM_EQU_24H` and $\beta_\tau = 0.0$ for 24/7 crypto. Because $\frac{1}{N}\sum_{t=1}^N \Delta \tau_t \equiv 1$, every half-life $H \in \{15\text{m}, 1\text{h}, 6\text{h}, 24\text{h}, 3\text{d}, 7\text{d}, 28\text{d}\}$ retains its exact calendar meaning over full weekly cycles while weekend dead zones advance the clock at only $0.25\times$ speed.

For 5-minute grids (`step_min == 5`), we also average the 15m and 1h EWMAs across all 5 staggered 1-minute phases over the final 60 minutes (`r_last60 = log(raw_closes[-60:] / raw_closes[-65:-5])`), eliminating 5-minute grid phase alignment luck. For 24/7 crypto (`CRYPTO_1H` and `CRYPTO_24H`), we blend unweighted, lightly-clipped (`5.0 * clip_ref`) fast EWMAs (`v_15m_raw`, `v_1h_raw`) so non-US hours (`s_ny < 0.85`) do not over-shrink live crypto breakouts toward the 28-day anchor.

### 2.5 Multi-Horizon Variance Ratio (`vr_scale`), Tail Shrinkage & Breakout Adaptation
To reconcile fat-tailed 1-step shocks with slightly mean-reverting intraday returns ($VR < 1$), we compute:
- **Heavy-Tail $L_1/L_2$ Shrinkage (`tail_shrink`)**: Based on the empirical ratio $\text{MAD}_{\text{deseas}} / \text{RMS}_{\text{deseas}} \in [0.78, 1.0]$.
- **Empirical Multi-Horizon Variance Ratio (`vr_scale`)**: $\sqrt{\operatorname{clip}(0.45\,VR(30\text{m}) + 0.55\,VR(3\text{h}), 0.78, \text{vr\_cap})}$ over the trailing 14 days for 24H forecasts (`vr_cap = 1.00` for crypto, `1.04` for equities/commodities), and $\sqrt{\operatorname{clip}(0.50\,VR(5\text{m}) + 0.50\,VR(15\text{m}), 0.80, 1.04)}$ over the trailing 7 days for 1H forecasts.
- **Fast Breakout Multiplier (`brk_mult`, `COM_EQU_24H`)**: When recent 1–3 day volatility breaks out above the 28-day baseline ($(0.35\,v_{24\text{h}} + 0.65\,v_{3\text{d}}) / v_{28\text{d}} > 1$), a damped multiplier $\operatorname{clip}(\cdot, 1.0, 1.32)^{0.25}$ counteracts historical mean-reversion damping during regime shifts.

### 2.6 Paired Antithetic $L_1$-Normalized Student-$t(\nu=6)$ Quadrature + Path/Block SV
To eliminate Monte Carlo sampling noise at $M = 1{,}000$ paths while matching both 1-step heavy tails and 24-hour volatility clustering:
1. **$L_1$-Normalized Stratified Student-$t(\nu=6)$ Innovations**: At each step $s$, we permute $M/2 = 500$ stratified quantiles of $t(\nu=6)$ normalized so that $\frac{1}{M/2}\sum_{i=1}^{M/2} |q_i| = \sqrt{2/\pi} \approx 0.79788456$ (matching the exact $L_1$ dispersion of a standard normal variable while preserving $\nu=6$ heavy tails).
2. **Paired Antithetic Path & Block Stochastic Volatility**: We draw stratified path-level log-normal volatility multipliers `p_half` ($\sigma_{\text{path}} = 0.22$ for 24H, $0.26$ for 1H, normalized to unit mean) and block-level log-normal multipliers `b_half` ($\sigma_{\text{block}} = 0.14$ over 1-hour blocks for 24H, $0.16$ over 5-minute blocks for 1H, normalized to unit column mean). Crucially, each antithetic shock pair $(z_i, -z_i)$ is assigned the **exact same** path and block volatility multipliers (`p_full = [p_half, p_half]`, `b_full = [b_half, b_half]`). This preserves exact zero-mean symmetry ($\sum_m z_{m,s}\,\sigma_{m,s} \equiv 0$ at every step $s$) while widening 3-hour and 24-hour unconditional kurtosis to match empirical return distributions.

---

## 3. Experimental Progression & Key Lessons (Including Failures)

Every iteration was evaluated at full production scale (`num_simulations = 1000`, `stride = 5` giving 34 prompts per 7-day window per asset) using `synth_lib.backtester.backtest` across two primary 7-day regimes (`R1_Aug` ending `2026-08-20` and `R2_Nov` ending `2026-11-07`, 36 competition-asset-regime cells totaling 1,224 forecast prompts), and then validated out-of-sample across all four calendar months (`R1_Aug`, `R2_Sep`, `R3_Oct`, `R4_Nov`, 72 cells totaling 2,448 forecast prompts).

### 3.1 Summary of Campaign Iterations (2-Regime Benchmark: `R1_Aug` & `R2_Nov`, `N = 1000`)

| Iteration | Commit | Core Change | `CRYPTO_24H` Mean `cw_crps` (`unw`) | `COM_EQU_24H` Mean `cw_crps` (`unw`) | `CRYPTO_1H` Mean `cw_crps` (`unw`) | Cells Improved vs. Prev (vs. Iter 0) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iter 0** | `dee4d2d` | Starter baseline (`CONTEXT_DAYS = 7`, flat Student-$t$ random walk) | `3156.06` (`3350.24`) | `2280.53` (`2806.43`) | `857.58` (`908.73`) | — | Baseline |
| **Iter 1** | `d84dcb0` | `CONTEXT_DAYS = 28`, contiguous log-returns, 6-horizon HAR-EWMA, empirical VR scaling, antithetic paths | `3072.81` (`3265.17`) | `2219.99` (`2727.44`) | `828.99` (`878.88`) | **34 / 36** (34 / 36) | **KEEP** |
| **Iter 2** | `4bc628c` | US-DST `America/New_York` 48-slot weekday + 5-regime weekend seasonality, GLS precision-weighted deseasonalized EWMAs | `3045.80` (`3237.76`) | `2113.86` (`2591.29`) | `828.08` (`877.96`) | **36 / 36** (36 / 36) | **KEEP** |
| **Iter 3** | *(discard)* | $L_1$-normalized Student-$t$ + unpaired path/block SV + static `comp_scale = 1.15` on equities | `3026.74` (`3219.01`) | `2087.17` (`2554.72`) | `826.72` (`876.64`) | 25 / 36 (`R1_Aug` equities degraded `+1.05%`) | **DISCARD** |
| **Iter 4** | `3a02e64` | Paired antithetic path & block SV (`p_full`, `b_full`), remove static `comp_scale`, faster HAR weights with `v_3d` | `3027.14` (`3219.45`) | `2074.47` (`2539.75`) | `826.61` (`876.50`) | **33 / 36** vs Iter 2 (36 / 36) | **KEEP** |
| **Iter 5** | `503f3fb` | Exact 7-day rolling variance in seasonality table, linear RMS weekend blend, `$SYNTH_BENCHMARK_CALENDAR` event multipliers | `3020.47` (`3212.16`) | `2063.84` (`2527.17`) | `826.20` (`876.04`) | **33 / 36** (36 / 36) | **KEEP** |
| **Iter 6** | `a073e1c` | Within-slot 96th-pct clipped $L_1$ seasonality (`_robust_slot_l1`), intrinsic activity-clock EWMA (`btau = 0.75`), breakout `brk_mult` | `3015.80` (`3206.97`) | `2049.08` (`2509.01`) | `825.55` (`875.36`) | **33 / 36** (36 / 36) | **KEEP** |
| **Iter 6b** | *(discard)* | Negatively autocorrelated MA(1)/MA(5) step innovations & Permanent+Transient noise decomposition | *(degraded 1-step CRPS)* | *(degraded `+4.66`)* | *(degraded `+0.58`)* | — | **DISCARD** |
| **Iter 7** | `5fc0bf4` | Dual-role event multipliers (`is_ewma_hist`), 24/7 crypto fast-horizon blend (`v_15m_raw`, `v_1h_raw`), `(v_24h, v_3d)/v_28d` breakout, strict `NaN` drop | **`3015.33`** (**`3206.44`**) | **`2046.08`** (**`2506.39`**) | **`824.73`** (**`874.51`**) | **29 / 36**, 6 tied (36 / 36) | **KEEP (Champion)** |

### 3.2 Analysis of Discarded Experiments & Negative Results

1. **Iteration 3 (Static `comp_scale = 1.15` & Unpaired Path SV — Discarded)**:
   - *What failed*: In Iteration 3, we noticed that `COM_EQU_24H` in volatile `R2_Nov` had a higher optimal scale than in calm `R1_Aug`, so we tested a static `1.15x` multiplier on non-crypto assets alongside path-level stochastic volatility where all 1,000 paths received independent `p_mult` draws. While `R2_Nov` improved by `-3.65%`, calm `R1_Aug` degraded by `+1.05%` (`6 / 8` assets worse), and assigning different `p_mult` values to antithetic twins $(z_i, -z_i)$ broke exact zero-mean symmetry.
   - *How it informed later iterations*: First, pairing each antithetic twin $(i, i + M/2)$ with the exact same `p_mult` and `b_mult` (`p_full = [p_half, p_half]`) in Iteration 4 immediately restored exact martingale symmetry. Second, investigation into *why* `COM_EQU_24H` needed higher RTH scale in Iteration 3 revealed two structural root causes—global-threshold $L_2$ outlier clipping was compressing the RTH-to-overnight seasonality contrast, and calendar-time EWMAs were decaying across the weekend—which we solved structurally in Iteration 6 without any static scale multiplier.

2. **Iteration 6b (MA(1)/MA(5) Step Innovations & Permanent-Transient Decomposition — Discarded)**:
   - *What failed*: Because empirical 1-minute and 5-minute returns have variance ratios below 1.0 ($VR(6) \approx 0.90$, $VR(36) \approx 0.85$), we tested generating negatively autocorrelated moving-average step innovations $\tilde{z}_t = (z_t + \theta \bar{z}_{t-k:t-1})/\sqrt{1 + \theta^2/k}$ ($\theta < 0$) and permanent-plus-transient price paths ($\log P_t = \sum_{s \le t} \sigma_{\text{perm}} z_s + \sigma_{\text{trans}} \eta_t$). Both degraded CRPS across `CRYPTO_1H` (`+0.58` mean `cw_crps`) and `COM_EQU_24H` (`+4.66` mean `cw_crps`).
   - *Why it failed (The CLT Kurtosis Trap)*: Any linear combination of independent random variables across adjacent steps ($z_t + \theta z_{t-1}$ or $z_t + \eta_t - \eta_{t-1}$) makes the 1-step increment a sum of multiple independent draws. By the Central Limit Theorem, summing independent draws reduces 1-step excess kurtosis toward a Gaussian distribution, directly hurting 1-step (`1min` and `5min`) CRPS—which carries over 55% of the competition weight—more than matching the multi-step variance ratio helps. Keeping each step innovation as a pure $L_1$-normalized Student-$t(\nu=6)$ draw while calibrating multi-horizon dispersion via `vr_scale` preserves exact 1-step fat tails.

3. **Iteration 7 Ablation `C1` (Symmetric vs. Dual-Role Earnings Multipliers — Discarded in Favor of `C2`)**:
   - *What failed*: When we updated `_get_event_multipliers` to reflect the full pre-August-2026 empirical earnings release curve (`8.50x / 9.00x / 5.50x / 2.60x` during after-hours `0..240m` and `1.35x..1.45x` across overnight and next-day RTH), applying that full curve symmetrically to both forward projection (`ev_fut`) and historical EWMA deseasonalization (`ev_hist`) improved pre-earnings prompts (`-36h..0h`) by `-100` to `-600` CRPS points, but **degraded** post-earnings prompts (`+13h..+28h`) by `+50` to `+100` CRPS points.
   - *Why it failed & how `is_ewma_hist` solved it*: Dividing historical post-earnings RTH returns by `1.45..9.00` inside `ev_hist` artificially depressed `v_1h`, `v_6h`, and `v_24h` right after an earnings shock! Separating `_get_event_multipliers(asset, ts_sec, is_ewma_hist)` so `is_ewma_hist=True` damps only the discrete announcement jump while `is_ewma_hist=False` projects the full multi-stage curve improved **10 / 10** pre- and post-earnings windows across all 5 equity earnings events (`NVDAX`, `SPCX`, `TSLAX`, `GOOGLX`, `AAPLX`).

---

## 4. Full Benchmark Results (`N = 1000` Paths)

We report the official `synth_lib.backtester.backtest` scores (`num_simulations = 1000`, `stride = 5`) across all **4 non-overlapping 7-day regimes** (`R1_Aug`: `2026-08-13..2026-08-20`, `R2_Sep`: `2026-09-11..2026-09-18`, `R3_Oct`: `2026-10-08..2026-10-15`, `R4_Nov`: `2026-10-31..2026-11-07`). Note that **`R2_Sep` and `R3_Oct` (36 cells, 1,224 prompts) were completely untouched holdout regimes** never inspected during model development.

### 4.1 Executive Summary by Competition (2-Regime & 4-Regime Benchmarks)

| Competition | 2-Regime (`Aug`, `Nov`) Baseline `cw_crps` (`unw`) | 2-Regime (`Aug`, `Nov`) Champion `cw_crps` (`unw`) | 2-Regime Change (`Cells Won`) | 4-Regime (`Aug..Nov`) Baseline `cw_crps` (`unw`) | 4-Regime (`Aug..Nov`) Champion `cw_crps` (`unw`) | 4-Regime Change (`Cells Won`) | Holdout (`Sep`, `Oct`) Cells Won |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CRYPTO_24H`** | `3156.06` (`3350.24`) | **`3015.33`** (**`3206.44`**) | **`-4.46%`** (`10 / 10`) | `3537.09` (`3756.58`) | **`3405.67`** (**`3623.72`**) | **`-3.72%`** (`20 / 20`) | **`10 / 10`** (`100%`) |
| **`COM_EQU_24H`** | `2280.53` (`2806.43`) | **`2046.08`** (**`2506.39`**) | **`-10.28%`** (`16 / 16`) | `2020.37` (`2472.22`) | **`1809.17`** (**`2202.03`**) | **`-10.45%`** (`32 / 32`) | **`16 / 16`** (`100%`) |
| **`CRYPTO_1H`** | `857.58` (`908.73`) | **`824.73`** (**`874.51`**) | **`-3.83%`** (`10 / 10`) | `886.15` (`940.55`) | **`850.07`** (**`903.37`**) | **`-4.07%`** (`20 / 20`) | **`10 / 10`** (`100%`) |
| **Total / Mean** | — | — | **`36 / 36` (`100%`)** | — | — | **`72 / 72` (`100%`)** | **`36 / 36` (`100%`)** |

Across all **72 competition-asset-regime cells (2,448 forecast prompts)**, the Champion Model (`Iteration 7`) beats the Starter Baseline (`Iteration 0`) on **72 out of 72 cells (`100.0%`)**, achieving an average relative CRPS reduction of **`-6.98%`** across all cells (`-10.45%` on `COM_EQU_24H`, `-4.07%` on `CRYPTO_1H`, and `-3.72%` on `CRYPTO_24H`).

### 4.2 Per-Regime Breakdown Across All 4 Months

| Competition | Metric | `R1_Aug` (`Aug 13–20`) | `R2_Sep` (`Sep 11–18`, Holdout) | `R3_Oct` (`Oct 08–15`, Holdout) | `R4_Nov` (`Oct 31–Nov 07`) | 4-Regime Mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`CRYPTO_24H`** | Baseline `cw_crps` (`unw`) | `2235.67` (`2385.15`) | `2016.29` (`2122.46`) | `5821.02` (`6193.75`) | `4075.39` (`4324.97`) | `3537.09` (`3756.58`) |
| **`CRYPTO_24H`** | **Champion `cw_crps` (`unw`)** | **`2143.35`** (`2288.24`) | **`1956.92`** (`2060.63`) | **`5635.09`** (`6021.39`) | **`3887.32`** (`4124.64`) | **`3405.67`** (`3623.72`) |
| **`CRYPTO_24H`** | Relative Change (`Cells Won`) | **`-4.13%`** (`5/5`) | **`-2.94%`** (`5/5`) | **`-3.19%`** (`5/5`) | **`-4.61%`** (`5/5`) | **`-3.72%`** (`20/20`) |
| **`COM_EQU_24H`** | Baseline `cw_crps` (`unw`) | `1621.25` (`1987.31`) | `1347.90` (`1638.07`) | `2172.52` (`2638.01`) | `2939.81` (`3625.51`) | `2020.37` (`2472.22`) |
| **`COM_EQU_24H`** | **Champion `cw_crps` (`unw`)** | **`1408.75`** (`1720.21`) | **`1189.03`** (`1446.63`) | **`1955.50`** (`2348.72`) | **`2683.41`** (`3292.58`) | **`1809.17`** (`2202.03`) |
| **`COM_EQU_24H`** | Relative Change (`Cells Won`) | **`-13.11%`** (`8/8`) | **`-11.79%`** (`8/8`) | **`-9.99%`** (`8/8`) | **`-8.72%`** (`8/8`) | **`-10.45%`** (`32/32`) |
| **`CRYPTO_1H`** | Baseline `cw_crps` (`unw`) | `691.77` (`732.16`) | `604.60` (`640.53`) | `1224.81` (`1304.22`) | `1023.40` (`1085.29`) | `886.15` (`940.55`) |
| **`CRYPTO_1H`** | **Champion `cw_crps` (`unw`)** | **`658.91`** (`697.81`) | **`580.66`** (`615.05`) | **`1170.18`** (`1249.40`) | **`990.55`** (`1051.20`) | **`850.07`** (`903.37`) |
| **`CRYPTO_1H`** | Relative Change (`Cells Won`) | **`-4.75%`** (`5/5`) | **`-3.96%`** (`5/5`) | **`-4.46%`** (`5/5`) | **`-3.21%`** (`5/5`) | **`-4.07%`** (`20/20`) |

### 4.3 Per-Asset 72-Cell Breakdown (`cw_crps`, Baseline -> Champion, % Change, and Historical Field Rank)

#### `CRYPTO_24H` (20 / 20 Cells Improved)
| Asset | `R1_Aug` (Iter 0 -> Iter 7, Rank) | `R2_Sep` Holdout (Iter 0 -> Iter 7, Rank) | `R3_Oct` Holdout (Iter 0 -> Iter 7, Rank) | `R4_Nov` (Iter 0 -> Iter 7, Rank) | 4-Regime Mean Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BTC`** | `1048.10 -> 1003.71` (`-4.24%`, **#2**) | `917.24 -> 888.81` (`-3.10%`, **#1**) | `2385.56 -> 2299.13` (`-3.62%`, #42) | `1940.48 -> 1861.71` (`-4.06%`, **#5**) | **`-3.75%`** |
| **`ETH`** | `2485.07 -> 2380.42` (`-4.21%`, #22) | `1693.96 -> 1638.94` (`-3.25%`, **#1**) | `5395.24 -> 5176.59` (`-4.05%`, #60) | `3692.19 -> 3467.85` (`-6.08%`, #8) | **`-4.40%`** |
| **`SOL`** | `2468.82 -> 2354.51` (`-4.63%`, #9) | `2449.57 -> 2368.18` (`-3.32%`, #10) | `6813.09 -> 6628.89` (`-2.70%`, #70) | `4181.20 -> 3972.37` (`-4.99%`, #24) | **`-3.91%`** |
| **`XRP`** | `1947.27 -> 1848.87` (`-5.05%`, **#3**) | `1573.32 -> 1506.99` (`-4.22%`, **#4**) | `6641.21 -> 6437.39` (`-3.07%`, #118) | `3838.67 -> 3678.00` (`-4.19%`, #20) | **`-4.13%`** |
| **`HYPE`** | `3887.64 -> 3726.08` (`-4.16%`, **#1**) | `3447.37 -> 3381.66` (`-1.91%`, #16) | `7870.00 -> 7633.44` (`-3.01%`, #26) | `6724.38 -> 6456.66` (`-3.98%`, #27) | **`-3.26%`** |

#### `COM_EQU_24H` (32 / 32 Cells Improved)
| Asset | `R1_Aug` (Iter 0 -> Iter 7, Rank) | `R2_Sep` Holdout (Iter 0 -> Iter 7, Rank) | `R3_Oct` Holdout (Iter 0 -> Iter 7, Rank) | `R4_Nov` (Iter 0 -> Iter 7, Rank) | 4-Regime Mean Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`XAU`** | `647.27 -> 563.16` (`-12.99%`, **#3**) | `931.76 -> 843.25` (`-9.50%`, #8) | `1876.62 -> 1661.27` (`-11.48%`, #39) | `1693.96 -> 1538.96` (`-9.15%`, #25) | **`-10.78%`** |
| **`NVDAX`** | `2040.57 -> 1811.43` (`-11.23%`, #14) | `1840.23 -> 1612.23` (`-12.39%`, **#2**) | `3378.83 -> 3067.62` (`-9.21%`, #45) | `4245.54 -> 3790.70` (`-10.71%`, **#1**) | **`-10.88%`** |
| **`GOOGLX`** | `1556.83 -> 1300.68` (`-16.45%`, **#2**) | `1887.07 -> 1679.22` (`-11.01%`, #9) | `2902.52 -> 2528.86` (`-12.87%`, **#3**) | `4009.40 -> 3787.05` (`-5.55%`, #15) | **`-11.47%`** |
| **`TSLAX`** | `2533.17 -> 2128.44` (`-15.98%`, #7) | `2642.06 -> 2404.14` (`-9.01%`, #51) | `3838.68 -> 3461.18` (`-9.83%`, #9) | `4447.82 -> 3832.17` (`-13.84%`, **#2**) | **`-12.16%`** |
| **`AAPLX`** | `1495.44 -> 1304.05` (`-12.80%`, #15) | `1721.84 -> 1479.82` (`-14.06%`, **#2**) | `2087.84 -> 1923.81` (`-7.86%`, #25) | `2826.09 -> 2694.43` (`-4.66%`, #34) | **`-9.84%`** |
| **`WTIOIL`** | `1744.71 -> 1581.53` (`-9.35%`, **#2**) | `1649.15 -> 1482.58` (`-10.10%`, **#4**) | `2574.16 -> 2387.92` (`-7.24%`, #63) | `2481.28 -> 2231.01` (`-10.09%`, #84) | **`-9.19%`** |
| **`SPCX`** | `3695.07 -> 3272.24` (`-11.44%`, **#3**) | `1804.60 -> 1585.82` (`-12.12%`, **#2**) | `2664.08 -> 2368.18` (`-11.11%`, #11) | `6247.34 -> 5762.23` (`-7.77%`, #10) | **`-10.61%`** |
| **`SP500`** | `635.72 -> 551.94` (`-13.18%`, **#1**) | `627.84 -> 527.14` (`-16.04%`, **#1**) | `1162.82 -> 1044.97` (`-10.13%`, #60) | `1027.99 -> 905.33` (`-11.93%`, **#3**) | **`-12.82%`** |

#### `CRYPTO_1H` (20 / 20 Cells Improved)
| Asset | `R1_Aug` (Iter 0 -> Iter 7) | `R2_Sep` Holdout (Iter 0 -> Iter 7) | `R3_Oct` Holdout (Iter 0 -> Iter 7) | `R4_Nov` (Iter 0 -> Iter 7) | 4-Regime Mean Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BTC`** | `337.98 -> 320.22` (`-5.25%`) | `278.57 -> 266.22` (`-4.43%`) | `559.17 -> 532.39` (`-4.79%`) | `462.66 -> 446.63` (`-3.46%`) | **`-4.48%`** |
| **`ETH`** | `723.09 -> 686.09` (`-5.12%`) | `484.33 -> 465.24` (`-3.94%`) | `1029.66 -> 982.34` (`-4.60%`) | `799.21 -> 771.22` (`-3.50%`) | **`-4.29%`** |
| **`SOL`** | `772.97 -> 735.12` (`-4.90%`) | `714.48 -> 683.90` (`-4.28%`) | `1414.21 -> 1344.28` (`-4.95%`) | `1023.79 -> 986.78` (`-3.61%`) | **`-4.44%`** |
| **`XRP`** | `655.34 -> 624.80` (`-4.66%`) | `503.92 -> 483.39` (`-4.07%`) | `1350.00 -> 1291.77` (`-4.31%`) | `964.61 -> 934.04` (`-3.17%`) | **`-4.05%`** |
| **`HYPE`** | `1171.40 -> 1122.84` (`-4.15%`) | `1041.72 -> 1004.56` (`-3.57%`) | `1771.03 -> 1700.09` (`-4.01%`) | `1866.73 -> 1814.07` (`-2.82%`) | **`-3.64%`** |

---

## 5. Limitations & Known Risks

1. **Unscheduled Geopolitical & Macro Shocks (`WTIOIL`, `XAU`, `CRYPTO_24H`)**:
   While `$SYNTH_BENCHMARK_CALENDAR` covers scheduled FOMC, CPI, NFP, EIA Petroleum Status Reports, and corporate earnings, commodity and crypto markets remain exposed to unscheduled weekend geopolitical announcements (e.g., OPEC+/Middle East headlines affecting Sunday 18:00 NY `WTIOIL` and `XAU` opens, or sudden exchange liquidation cascades in `R3_Oct` crypto). Because `context_prices` provides only past prices of the target asset, an unscheduled jump that begins *after* `start_time` cannot be anticipated from univariate price history alone.
2. **Close-Only Context Interface**:
   The `simulate(..., context_prices=...)` interface supplies only 1-minute `close` prices (no 1-minute high/low range or volume). While 1-minute closes allow us to construct 5-phase overlapping 5-minute EWMAs and 1-minute variance ratios, intra-minute Parkinson/Garman-Klass range estimators cannot be computed directly at inference time.
3. **Homogeneous Event Multiplier Curve Across Equities**:
   Our forward `EARNINGS` multiplier curve (`ev_fut`) uses the cross-equity empirical pre-August-2026 trajectory for all five equities (`NVDAX`, `TSLAX`, `GOOGLX`, `AAPLX`, `SPCX`). In practice, high-beta single names (`TSLAX`, `NVDAX`) occasionally experience larger post-earnings RTH drift/volatility than mega-cap lower-beta names (`AAPLX`), though `s_table * deseas_schedule` naturally scales the curve by each asset's baseline volatility.

---

## 6. What We Would Do With More Budget

1. **Asset-Specific Empirical Bayesian Event-Response Calibration**:
   With additional research iterations, we would estimate an asset-specific event-beta scalar $\kappa_{\text{asset}}$ from historical pre-cutoff earnings/EIA releases (shrinking each asset's past 2–3 quarterly earnings responses toward the cross-sectional prior) so `TSLAX`/`NVDAX` vs. `AAPLX` have tailored after-hours jump vs. next-day RTH follow-through ratios.
2. **Regime-Switching Jump-Diffusion Tail Mixture for Commodities (`WTIOIL` & `XAU`)**:
   In `WTIOIL` and `XAU`, empirical returns during high-volatility geopolitical weeks (`R3_Oct` and `R4_Nov`) exhibit occasional discrete jumps during the Sunday 18:00 NY Globex opening and London/NY transition slots. Adding a small Poisson jump mixture active specifically during transition slots when `v_3d / v_28d > 1.15` could further tighten `WTIOIL` and `XAU` tail calibration.
3. **Horizon-Dependent Heavy-Tail Degrees of Freedom ($\nu_\tau$)**:
   Currently, step innovations use a uniform $L_1$-normalized Student-$t(\nu=6)$ distribution across all steps, relying on path/block SV to shape multi-hour kurtosis. Allowing $\nu$ to vary between RTH active slots ($\nu \approx 7$) and illiquid overnight/transition slots ($\nu \approx 4.5$) could further refine 5-minute vs. 3-hour quantile calibration on `COM_EQU_24H`.
