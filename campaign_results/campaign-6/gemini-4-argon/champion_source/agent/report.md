# Calibrating Multi-Horizon Probabilistic Price Simulations on Bittensor SN50 (Synth)

## 1. Context & Problem Statement

In Bittensor Subnet 50 (**Synth**), miners generate probabilistic price forecasts as an ensemble ("fan") of $N = 1,000$ simulated price paths starting from the most recent observed price $P_0$. Validators score these ensembles against realized future prices using the **Continuous Ranked Probability Score (CRPS)** summed across multiple return horizons and (for 1-hour prompts) realized block volatilities:

| Competition | Assets | Horizon / Step | Scored Intervals & Weights |
| :--- | :--- | :--- | :--- |
| **`CRYPTO_24H`** | `BTC, ETH, SOL, XRP, HYPE` | 24h (`86,400s`), 5m (`300s`, 289 pts) | `5min`, `30min`, `3hour`, `24hour_abs` |
| **`COM_EQU_24H`** | `XAU, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL, SPCX, SP500` | 24h (`86,400s`), 5m (`300s`, 289 pts) | `5min`, `30min`, `3hour`, `24hour_abs` |
| **`CRYPTO_1H`** | `BTC, ETH, SOL, XRP, HYPE` | 1h (`3,600s`), 1m (`60s`, 61 pts) | `1min`, `2min`, `5min`, `15min`, `30min`, `60min_abs`, `10min..60min_gaps`, plus `VOL_CRPS_LAMBDA = 5.25` × (`vol_60min`, `vol_15min`, `vol_5min`) |

For a forecast cumulative distribution $F$ and realized return $y$ (in basis points),
$$	ext{CRPS}(F, y) = \mathbb{E}_F|X - y| - \frac{1}{2}\mathbb{E}_F|X - X'|.$$
Near the optimum $\sigma^*$, a Gaussian fan with scale $\sigma$ incurs an expected CRPS penalty of approximately
$$\frac{\mathbb{E}[	ext{CRPS}(\sigma)]}{\mathbb{E}[	ext{CRPS}(\sigma^*)]} \approx 1 + \frac{1}{4}\left(\ln \frac{\sigma}{\sigma^*}ight)^2.$$
Consequently, any model that uses a constant volatility over a 24-hour horizon on tokenized equities or commodities — where per-minute realized volatility swings **4–5x** between the US cash open (`13:30–14:00 UTC`) and overnight/weekend sessions — is simultaneously punished for being far too wide overnight and on weekends, and far too narrow at the US cash open. Similarly, a homoscedastic i.i.d. Gaussian path generator in `CRYPTO_1H` produces almost zero cross-path dispersion in realized path volatility (`vol_60min`, `vol_15min`, `vol_5min`), incurring a severe penalty on the `5.25x`-weighted volatility CRPS terms.

### Validation Protocol
- **No-Lookahead Calibration**: All structural tables (weekly seasonality, within-hour and within-half-hour modulation, long-term volatility anchors, and scheduled event multipliers) were estimated strictly on pre-validation history (`2025-08-01` to `2026-09-13`, with primary fit window `2026-05-18` to `2026-09-07` and a 60-day pre-validation split `2026-07-16..2026-08-14` vs `2026-08-15..2026-09-13`).
- **Two Disjoint 10-Day Out-of-Sample Regimes (at 1,000 paths via `synth_lib.backtester`)**:
  - **Window A (`2026-09-15` → `2026-09-25`, 10 days)**: High-volatility regime (includes the September 16–17 FOMC meeting and crypto liquidation spikes; 240 prompts/asset for 24h, 1,440 prompts/asset for 1h).
  - **Window B (`2026-10-01` → `2026-10-11`, 10 days)**: Calmer / mixed recent regime immediately preceding the data cutoff (includes the October 2 NFP release; 240 prompts/asset for 24h, 1,440 prompts/asset for 1h).
- **Acceptance Rule**: Every retained change had to reduce CRPS across both Window A and Window B and improve the majority of the 36 `(asset, window)` evaluation cells.

---

## 2. Approach & Model Architecture

Our final model in `agent/modeling.py` is a fast (~18–25 ms per 1,000-path prompt on CPU), self-contained, zero-lookahead probabilistic simulator built around six core components:

### 2.1. Separated Backward-$L^2$ Deseasonalization & Forward-CRPS Seasonal Projection (`336` Weekly Half-Hour Slots + Sub-Slot Modulation)
1. **336-Slot Weekly Grid**: Each UTC week is partitioned into $7 	imes 48 = 336$ half-hour slots starting Monday `00:00 UTC`.
2. **Why Backward and Forward Seasonality Must Be Separated**:
   If a single seasonality curve $S(s)$ is used for both backward deseasonalization and forward projection, then for any prompt whose forecast horizon stays in the same session as its recent context (e.g., Saturday or weekday overnight), the ratio $S_{	ext{fwd}}(s) / S_{	ext{back}}(s) \approx 1$ cancels out in fast EWMAs! Because weekend and overnight returns are zero-inflated and heavily leptokurtic, their $L^2$ RMS is **30–65% higher** than their CRPS-optimal scale $c^*$. We therefore maintain:
   - `_S2_BACK_336`: Backward deseasonalization profile so deseasonalized squared returns $r_t^2 / \Delta S_{	ext{back}}^2$ are stationary across sessions.
   - `_S2_FWD_336`: Forward CRPS-optimal variance profile calibrated across weekly sessions on pre-validation data (`2026-06-22` to `2026-09-07`). For `CRYPTO_1H`, blending `0.35 * s2_back + 0.45 * s2_fwd_norm[asset] + 0.20 * s2_fwd_norm['BTC']` captures weekend quietness without over-damping intraday peaks.
3. **Sub-Half-Hour Modulation (Normalized to Unit Mean per Half-Hour)**:
   - **`MIN_OF_HOUR_S2` (`CRYPTO_1H`)**: 1-minute crypto returns exhibit a **1.74x sawtooth wave** within every half-hour — minutes `:00–:04` and `:30–:34` run at `1.23x–1.25x` mean variance while `:25–:29` and `:55–:59` drop to `0.71x–0.83x`.
   - **`WEEKDAY_SUB288_S2` (`COM_EQU_24H` equities & `WTIOIL`)**: Within the `13:30–14:00 UTC` NYSE cash open half-hour, 5-minute variance decays by more than **2x** from `13:30–13:35` to `13:55–14:00`.

### 2.2. Exact Full-Grid Seasonal Integral Over NaN Gaps
In `context_prices`, missing (`NaN`) closes represent untraded minutes, not missing clock time. Taking `.dropna()` and treating consecutive trade prices as 1-minute returns inflates variance on sparse assets (`AAPLX` has 41% NaN minutes). Instead, for each valid return $r_k = \ln(P_{t_k} / P_{t_{k-1}})$, we compute the exact seasonal variance integral over all elapsed clock minutes:
$$\Delta S_k^2 = \sum_{u = t_{k-1}+1}^{t_k} S_{	ext{back}}^2(u),$$
and evaluate the EWMA denominator over the **full active minute grid** $\sum_{u} w(u) S_{	ext{back}}^2(u)$ so untraded minutes properly contribute zero realized movement over their elapsed seasonal time.

### 2.3. Scheduled Calendar Event Multipliers (`$SYNTH_BENCHMARK_CALENDAR`)
Leave-one-event-out analysis on `2025-08-05` to `2026-09-07` showed that 5 scheduled event types have 100% consistent directional variance effects:
- **`FOMC`** (`+0` to `+60m` after release): `3.40x` variance on crypto, `XAU`, and `SP500`; `2.10x` on single-stock equities.
- **`NFP`** (`+0` to `+30m`): `2.10x` variance on crypto, `XAU`, and `SP500`.
- **`CPI`** (`+0` to `+30m`): `1.80x` variance across all assets except `WTIOIL`.
- **`EARNINGS`** (`+0` to `+90m` for the specific stock): `12.0x` variance.
- **`NYSE_CLOSED` / `NYSE_EARLY_CLOSE`**: `0.05x` (`0.25x` on `SP500`) during the closed cash session (`390m`) and `0.12x` (`0.30x` on `SP500`) during early-close afternoons (`120m`).
Applying these multipliers in both `grid_s2` (backward deseasonalization, preventing a past event from polluting post-event EWMAs) and `fwd_s2_min` (forward projection) sharply reduced event-window CRPS gaps.

### 2.4. Multi-Scale 6-Half-Life Hybrid Volatility Filter (`L2 + L1 + 5m Block MAD + 1m True-Range OHLC`)
Across 6 exponential half-lives (`5m, 15m, 60m, 240m, 1440m, 10080m`), we combine four complementary deseasonalized log-volatility estimators:
1. **Winsorized $L^2$ (`log_l2`)**: 98.5th-percentile clipped deseasonalized variance rate $r_k^2 / \Delta S_k^2$.
2. **Deseasonalized $L^1$ (`log_l1`)**: $\sqrt{\pi/2} |r_k| \sqrt{\Delta S_k^2}$ integrated against the seasonal denominator.
3. **5-Minute Block MAD (`_compute_5m_mad_ewma`)**: Bins dropped-NaN trade returns into 5-minute clock blocks via `np.bincount` (never interpolating NaNs) so changes in 1-minute sparsity do not bias 5-minute step dispersion.
4. **1-Minute True-Range Parkinson Estimator (`_compute_ohlc_log_ewma`)**: Uses `high`, `low`, and consecutive previous `close` ($H_t^* = \max(H_t, C_{t-1})$, $L_t^* = \min(L_t, C_{t-1})$) to capture close-to-open jumps between 1-minute bars, anchored at the 7-day half-life (`log_l2[5]`).

Forward base volatility mean-reverts exponentially in log-space from $\sigma_{	ext{short}}$ to $\sigma_{	ext{long}}$ with horizon time constant $	au = 30	ext{ min}$ (`CRYPTO_1H`) and $	au = 360	ext{ min}$ (`24H`), modulated on `CRYPTO_24H` by the 24-hour Kaufman Efficiency Ratio ($	ext{ER}_{24	ext{h}} = |R_{24	ext{h}}| / 	ext{RV}_{24	ext{h}}$) to contract in range-bound chop ($	ext{ER}_{24	ext{h}} < 0.35$) and expand during active directional breakouts ($	ext{ER}_{24	ext{h}} > 0.55$ with $\sigma_{	ext{short}} > \sigma_{	ext{long}}$).

### 2.5. Stratified Path/Step Log-Normal Scale Mixture & Orthogonal Horizon Secant Projection
To simultaneously calibrate heavy-tailed step returns, multi-hour cumulative returns, and `CRYPTO_1H` path volatility distributions (`vol_60min`, `vol_15min`, `vol_5min`):
1. **Stratified Path Multipliers**: The $N/2 = 500$ antithetic pairs receive deterministic inverse-normal stratified log-normal volatility multipliers $m_k = \exp(\sigma_{	ext{path}} \Phi^{-1}(\frac{k+0.5}{N/2}) - \frac{1}{2}\sigma_{	ext{path}}^2)$, paired with step-level log-normal multipliers $\exp(\sigma_{	ext{step}}\eta_{k, t} - \frac{1}{2}\sigma_{	ext{step}}^2)$ and exact per-step column standardization.
2. **Orthogonal Horizon Secant Projection (`_apply_secant_shocks`)**:
   Decomposing each path increment into its Brownian bridge component and variance-weighted block/horizon secant component:
   $$\Delta 	ilde{X}_{k, j} = \Delta X_{k, j} + (\eta - 1)\frac{\sigma_j^2}{\sum_m \sigma_m^2}\sum_m \Delta X_{k, m}$$
   allows independent calibration of multi-step cumulative dispersion vs single-step dispersion (`eta_tot = 1.09` / `1.05` on `COM_EQU_24H` to account for 5-minute zero-return inflation vs 0% 24h zeros; `eta_blk = 0.95` on 15-minute blocks in `CRYPTO_1H` to compensate for CLT kurtosis averaging).

---

## 3. Progression of Experiments (Including Negative Results)

| Iteration | Description | Verdict | Key Empirical Finding |
| :--- | :--- | :---: | :--- |
| **Iter 0** | Starter baseline (`np.std(diff(log(close.dropna())))`) + vectorized `predict.py` | Baseline | Severe NaN-gap inflation and no seasonality (`COM_EQU_24H` gap +30% to +47%). |
| **Iter 1** | 336-slot weekly seasonality + full-grid deseasonalized HAR-EWMA + antithetic sampling | **Keep** | Improved 6/6 competition-windows (`COM_EQU_24H` -53.6 / -75.0 CRPS; `CRYPTO_1H` -11.9 / -33.9). |
| **Iter 2** | Stratified path/step log-normal scale mixture + CRPS scale calibration + fast indexing | **Keep** | Improved 32/36 cells over Iter 1; cut `simulate()` latency 3x; gave `CRYPTO_1H` realistic block vol spread. |
| **Iter 3** | Fix Pandas `Timestamp.as_unit('us').value` nanosecond bug + 5m MAD/RMS session curves | **Keep** | **36/36 cells improved** (`CRYPTO_24H` -81.4 / -69.7; `COM_EQU_24H` -161.1 / -165.9; `CRYPTO_1H` -18.8 / -14.1). |
| **Iter 4** | Separate backward-$L^2$ (`_S2_BACK_336`) from forward-CRPS (`_S2_FWD_336`) + 6-scale L1/L2 EWMA | **Keep** | Eliminated weekend/overnight EWMA cancellation bias; 34/36 cells improved over Iter 3. |
| **Iter 5** | 60m within-hour sawtooth (`MIN_OF_HOUR_S2`) + 5m weekday sub-slots + calendar event multipliers | **Keep** | 35/36 cells improved; FOMC/NFP event CRPS gaps cut by 50–80%. |
| **Iter 6** | 5m block MAD (`_compute_5m_mad_ewma`) + scale-anchored OHLC Parkinson range (`_compute_ohlc_log_ewma`) | **Keep** | Improved 10/10 `CRYPTO_1H` cells and both `COM_EQU_24H` windows; rejected GJR leverage after ablation. |
| **Iter 7** | Lower `CRYPTO_24H` global `c_scale` (`0.935` vs `0.96`) | **Discard** | Improved calm Window B (-2.65) but hurt high-vol Window A (+3.05); rejected per cross-regime rule. |
| **Iter 8** | Orthogonal horizon secant projection + `CRYPTO_1H` forward seasonality + True-Range OHLC + Efficiency Ratio | **Keep** | Improved **18/18 pre-val pairs** and **6/6 competition-window totals** (`CRYPTO_1H` Win B -1.11 CRPS). |
| **Iter 9** | Discrete zero-return point mass (10–15% delta at 0) & 7-session CV residual table | **Discard** | Putting discrete point mass at 0 shrinks $\mathbb{E}|X-X'|$ and worsened CRPS across all tested assets (`+0.4` to `+16.8`). |
| **Iter 10** | Pre-path column standardization (`pre_col_std`) + 3-level hierarchical secant (`BTC/ETH` 24h) + `SPCX/SP500` off-hours EWMA damping | **Keep** | Improved **36/36 `(asset, window)` cells** over Iter 8 (`CRYPTO_24H` -0.40/-0.79; `COM_EQU_24H` -0.23/-1.12; `CRYPTO_1H` -0.03/-0.04). |

---

## 4. Detailed Results Across Validation Regimes (1,000 Paths)

### 4.1. Competition-Level Summary

| Competition | Window | Starter (Iter 0) Mean CRPS | Champion (Iter 10) Mean CRPS | Absolute CRPS Reduction | Relative CRPS Reduction | Starter Coef CRPS | Champion Coef CRPS | Starter Mean Gap | Champion Mean Gap |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`CRYPTO_24H`** | **Win A** (`09-15..09-25`) | 4906.568 | **4805.635** | **-100.933** | **-2.06%** | 4597.614 | **4504.083** | +3.70% | **+1.19%** |
| **`CRYPTO_24H`** | **Win B** (`10-01..10-11`) | 3528.842 | **3359.914** | **-168.928** | **-4.79%** | 3326.500 | **3161.400** | +10.56% | **+2.07%** |
| **`COM_EQU_24H`** | **Win A** (`09-15..09-25`) | 2161.553 | **1892.577** | **-268.976** | **-12.44%** | 1780.843 | **1563.528** | +30.32% | **+2.14%** |
| **`COM_EQU_24H`** | **Win B** (`10-01..10-11`) | 2062.542 | **1762.511** | **-300.031** | **-14.55%** | 1686.077 | **1437.689** | +47.25% | **+3.83%** |
| **`CRYPTO_1H`** | **Win A** (`09-15..09-25`) | 1060.074 | **1024.292** | **-35.782** | **-3.38%** | 992.517 | **959.482** | +10.88% | **+4.92%** |
| **`CRYPTO_1H`** | **Win B** (`10-01..10-11`) | 740.814 | **686.073** | **-54.741** | **-7.39%** | 699.090 | **645.561** | +23.44% | **+4.64%** |

### 4.2. Per-Asset Breakdown (36/36 Asset-Window Cells Improved)

| Competition | Asset | Coef | Win A Starter | Win A Champion | Win A $\Delta$ | Win B Starter | Win B Champion | Win B $\Delta$ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`CRYPTO_24H`** | `BTC` | 1.000 | 2997.592 | **2926.149** | -71.443 | 2372.879 | **2239.760** | -133.119 |
| **`CRYPTO_24H`** | `ETH` | 0.651 | 4037.465 | **3941.386** | -96.079 | 2844.582 | **2683.714** | -160.868 |
| **`CRYPTO_24H`** | `SOL` | 0.588 | 4892.707 | **4830.269** | -62.438 | 3773.623 | **3606.588** | -167.035 |
| **`CRYPTO_24H`** | `XRP` | 0.495 | 6635.389 | **6565.020** | -70.369 | 3955.433 | **3696.604** | -258.829 |
| **`CRYPTO_24H`** | `HYPE` | 0.258 | 5969.685 | **5765.354** | -204.331 | 4697.692 | **4572.901** | -124.791 |
| **`COM_EQU_24H`** | `SP500` | 3.440 | 759.702 | **709.241** | -50.461 | 764.191 | **690.604** | -73.587 |
| **`COM_EQU_24H`** | `XAU` | 2.261 | 1645.031 | **1499.962** | -145.069 | 1388.882 | **1231.250** | -157.632 |
| **`COM_EQU_24H`** | `AAPLX` | 1.855 | 1587.664 | **1273.828** | -313.836 | 1591.018 | **1271.620** | -319.398 |
| **`COM_EQU_24H`** | `NVDAX` | 1.389 | 1813.673 | **1658.029** | -155.644 | 1875.547 | **1618.842** | -256.705 |
| **`COM_EQU_24H`** | `GOOGLX` | 1.351 | 2177.442 | **1922.858** | -254.584 | 1993.709 | **1656.029** | -337.680 |
| **`COM_EQU_24H`** | `TSLAX` | 1.282 | 2522.052 | **2060.626** | -461.426 | 2272.693 | **1857.086** | -415.607 |
| **`COM_EQU_24H`** | `WTIOIL` | 1.186 | 3814.934 | **3608.290** | -206.644 | 3415.241 | **3096.037** | -319.204 |
| **`COM_EQU_24H`** | `SPCX` | 0.900 | 2971.928 | **2407.782** | -564.146 | 3199.057 | **2678.618** | -520.439 |
| **`CRYPTO_1H`** | `BTC` | 1.000 | 644.713 | **622.072** | -22.641 | 499.038 | **454.486** | -44.552 |
| **`CRYPTO_1H`** | `ETH` | 0.651 | 873.722 | **842.640** | -31.082 | 603.320 | **552.288** | -51.032 |
| **`CRYPTO_1H`** | `SOL` | 0.588 | 1054.975 | **1029.690** | -25.285 | 794.986 | **741.061** | -53.925 |
| **`CRYPTO_1H`** | `XRP` | 0.495 | 1419.772 | **1381.282** | -38.490 | 833.288 | **753.538** | -79.750 |
| **`CRYPTO_1H`** | `HYPE` | 0.258 | 1307.187 | **1245.776** | -61.411 | 973.435 | **928.990** | -44.445 |

---

## 5. Limitations & What We Would Do With More Budget

1. **Single-Asset Context Constraint (`simulate()` API)**:
   Because `simulate()` receives only the target asset's 7-day `context_prices`, cross-asset lead-lag signals (e.g., `BTC` leading `SOL/XRP/HYPE` during sudden liquidations, or `SP500` leading thin single-stock perps like `AAPLX` and `SPCX` when those stocks have 10–30 minute trade gaps) cannot be used at inference time.
2. **Daylight Saving Time (EDT vs EST) Transitions in Static UTC Seasonality Tables**:
   Our 336-slot weekly seasonality profiles (`SEASONALITY_336` and `FORWARD_S_336`) are indexed in UTC and calibrated on summer/early-autumn data (EDT, `UTC-4`, where NYSE opens at `13:30 UTC`). When US Daylight Saving Time ends in November (EST, `UTC-5`), the NYSE open shifts to `14:30 UTC` while the scheduled calendar events (`events.parquet`) automatically shift in `time_utc`. In a year-round deployment, indexing the US equity/commodity seasonality profile in `America/New_York` local clock time rather than fixed UTC slots would make the model invariant to DST transitions.
3. **Fine-Grained Intraday Volatility Autocorrelation Along Simulated Paths**:
   While our stratified path-level (`sig_path_vv`) and step-level (`sig_step_vv`) log-normal mixture with 15-minute secant projection (`_apply_secant_shocks`) captures first-order path volatility dispersion in `CRYPTO_1H`, replacing independent step multipliers with a correlated Markov-switching or rough-volatility path process could further tighten the `vol_15min` and `vol_5min` CRPS terms.
