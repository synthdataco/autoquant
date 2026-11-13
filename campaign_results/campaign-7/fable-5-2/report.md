# Calibrating price fans for Synth (SN50): a seasonal, session-aware stochastic-volatility simulator fitted on CRPS

*Campaign-7, model fable-5-2. Data cutoff 2026-11-08. All CRPS numbers below come from the official
synth-lib scorer through `agent/predict.py` at 1000 paths.*

## 1. Context

A Synth miner answers each prompt with 1000 price paths; the validator sums ensemble CRPS over many
interval returns. Reading `synth.validator.crps_calculation` gives the anatomy that drove every decision:

- **24h configs (5-min steps).** 288 five-minute returns carry ~61 % of a calibrated fan's CRPS, 48
  thirty-minute returns 25 %, 8 three-hour returns 10 %, the absolute 24h price 4 %.
- **crypto-1h (1-min steps).** 1-min returns 26 %, 2-min 18 %, 5-min 12 %, 15/30-min 12 %, the twelve
  "0→5k min" gaps 29 %, the 60-min level 3 %, plus ~1 % for the CRPS of realised block volatilities.

So the score is mostly about the *scale and shape of the next few steps*, at every step of the horizon.
The field is extremely tight: over a month the best miner beats the median miner by 0.3–0.5 % of CRPS,
and a naive constant-volatility Gaussian (the starter) is only 3 % (crypto 24h) to 11 % (equities) behind
the best miner. Every idea is worth tenths of a percent; experiments must be paired on identical prompts.

## 2. Approach

One model for all 13 assets and both horizons (`agent/core.py`, numpy only; ~60 fitted numbers per
asset and format in `agent/params/`):

1. **Seasonal variance of every minute** = New-York-clock weekly profile (10 080 minutes) x UTC-clock daily
   profile x scheduled-event multipliers. The weekly profile is estimated on the scale CRPS rewards,
   (E|r|)^2 rather than E[r^2], pools weekdays at one-minute resolution with a jump-preserving smoother,
   keeps a minute-of-hour factor, and gives Saturdays, Sundays and NYSE holidays their own shape. Event
   multipliers (CPI, NFP, PPI, retail sales, GDP, PCE, FOMC, EIA, earnings) are tables by minutes since the
   event, read from the calendar file only.
2. **Level.** From the context: deseasonalised EWMAs of squared returns, absolute returns and
   Parkinson ranges at nine half-lives (2 min … 7 days), the context mean, and EWMAs over *the target
   step's own session* (core / shoulder / night / weekend). The forecast for step h is a power mean of
   these estimators with softmax weights that move with the horizon, a discount on fast estimators when
   the step lies in another session than the present, per-session offsets, and small linear terms
   (signed and absolute recent-return z-scores, a jump indicator, the share of zero-return minutes).
3. **Paths.** Gaussian shocks times exp(log-vol), where log-vol = a permanent factor whose loading grows
   with the horizon + an OU factor + white noise whose size depends on the session and on how thin the
   market is (this is what produces fat tails and the right dispersion of realised volatility), and a
   small MA(1). Zero drift.
4. **Fitting on the metric.** `research/train.py` replays the validator's CRPS (all intervals, gaps and
   volatility blocks, with the N=1000 ensemble term) in torch on GPU-simulated paths and runs Adam on
   all parameters at once. Nothing is tuned by hand.
5. **Sampling.** Latin-hypercube shocks; then the 1000 served paths are moved, rank by rank and from the
   finest block size to the coarsest, onto the quantiles of an 8x larger reference sample of the same
   model; finally prices are snapped to the venue's tick, read off the context.

## 3. Experiments (in order; failures included)

| # | Change | Evidence | Verdict |
|---|---|---|---|
| 0 | Starter: constant 7-day sigma, Gaussian | last 30 d: crypto-24h 3785.58, com-equ 1988.95, 1h 833.37 (+3.4 %, +10.8 %, +5.7 % vs best miner) | baseline |
| 1 | v1: variance profile + multi-scale level + SV paths, CRPS-fitted | fold A (fit < 2026-07-24, 100 d out of sample): 3665.11 / 1686.69 / 780.30; 1st on all 1h assets and 5 of 13 24h assets | keep |
| 2a | Seasonal on (E|r|)^2, session-dependent tail weight and offsets | fold A com-equ 1686.69 → 1676.97 (AAPLX 1531.60 → 1511.94, SP500 751.44 → 744.98) | keep |
| 2b | Latin-hypercube shocks | paired: BTC 24h 2461.54 → 2459.93, BTC 1h 578.77 → 578.48 (antithetic: half of that) | keep |
| 2c | Hierarchical quantile matching | paired: -0.02 … -0.03 % everywhere, +0.45 s per 24h prompt | keep (serving only) |
| 3a | Tick-grid snapping | paired: SOL 1h -0.04 %, XRP 1h -0.03 %, XAU 24h -0.02 %, BTC 0 | keep |
| 3b | Session-level features + cross-session discount | fold B validation loss without them: SP500 +0.19 %, WTI +0.09 %, AAPLX +0.09 %, crypto 0 | keep |
| 3c | Power mean instead of geometric pooling | +0.00 … +0.05 % without it | keep, marginal |
| 3d | UTC-clock component (two clocks) | post-DST days, same parameters: crypto -0.03 %, com-equ +0.01 % | keep, marginal |
| 3e | Whole v3 vs v2 on fold B (fit < 2026-10-01, 36 d) | com-equ 1699.70 → 1696.93, 1h 761.26 → 760.73, crypto-24h 3535.16 → 3535.55 | keep |
| – | Variance profile on E[r^2] (v1) | weekend steps 40 % too wide at 12–24 h, core session 10–30 % too narrow | replaced |
| – | 1-min weekend buckets for thin perps | zero-inflated buckets exploded the deseasonalised returns out of sample (AAPLX loss x200) | replaced by 5-min means + floor |
| – | Stale parameters | fit < July scored on October: com-equ +0.33 % worse than the September fit (XAU +0.8 %, SPCX +1.8 %); crypto unchanged | refit close to the cutoff |

Diagnostics that paid for themselves: a per-cell CRPS-optimal width multiplier (`research/diag_scale.py`,
"price a residual before fitting it"), per-prompt joins with the field by hour, weekday, volatility
decile, ex-ante state and event type (`research/where_lose*.py`, `vs_top*.py`, `event_split.py`).

## 4. Results

**Out of sample, fold B** (everything fitted before 2026-10-01; validator prompts scored 2026-10-02 →
2026-11-07; quantile matching off). "Paired" compares mean CRPS with each miner on the prompts both answered.

| config | AGG CRPS | vs best miner | vs median miner | rank 1 on |
|---|---|---|---|---|
| CRYPTO_24H | 3535.55 | -0.03 % | -0.26 % | BTC, ETH, SOL, XRP (HYPE 85/231) |
| COM_EQU_24H | 1696.93 | -0.08 % | -0.73 % | XAU, NVDAX, TSLAX, SP500 (AAPLX 2, GOOGLX 32, SPCX 44, WTIOIL 191) |
| CRYPTO_1H | 760.73 | -0.06 % | -0.33 % | BTC, ETH, SOL, XRP (HYPE 31/223) |

**Champion configuration on the starter's window** (validator prompts scored 2026-10-08 → 2026-11-07,
quantile matching and tick grid on; out of sample for the served parameters, see section 5). Paired against the field on these prompts: -0.02 % / -0.01 % / -0.06 % vs the best miner per asset and -0.26 % / -0.72 % / -0.33 % vs the median miner:

| config | starter | champion | change |
|---|---|---|---|
| CRYPTO_24H | 3785.58 | 3663.19 | -3.23 % |
| COM_EQU_24H | 1988.95 | 1799.38 | -9.53 % |
| CRYPTO_1H | 833.37 | 791.12 | -5.07 % |

Where the edge comes from, against the median miner on fold B: scheduled events (crypto CPI prompts
-1.5 %, equity earnings prompts -6 %, 1h prompts spanning the CPI minute -15 %), the US open, and quiet
sessions of the thin equity perps. Where it does not exist: WTIOIL (+0.5 % on ordinary prompts) and HYPE.

## 5. Limitations

- **Small margins, short windows.** The out-of-sample advantage over the best miner is a few hundredths of
  a percent on 36 days; on the last 12 days alone, against the five miners that turned out best there, the
  model is 0.2–0.3 % behind on crypto, all of it in high-volatility hours that followed quiet states.
- **Served parameters are the fold-B fit (data before 2026-10-01).** They are the ones validated out of sample, but they are five weeks older than the cutoff; the freshness experiment says the short-history assets (SPCX, SP500, XAU, WTIOIL) would gain from a refit through 2026-11-08, which the credit budget did not let me validate. On the starter's window below, 2026-10-08 → 2026-11-07, these parameters are therefore still out of sample.**
- The UTC component and the event tables rest on few observations (one earlier winter; 10–16 events per
  type). PCE/GDP multipliers look too high.
- Weekend re-openings are modelled as seasonal variance, not as the gap risk they are.
- Only one asset's context is visible; no cross-asset information is used.
- Latency: ~0.5 s per 24h prompt, ~0.1 s per 1h prompt on this box, dominated by the reference sample.

## 6. With more budget

1. A small pooled network for the level (inputs: the same features), trained end-to-end on the same loss.
2. State-dependent volatility-of-volatility (wider after quiet states), the pattern the last 12 days show.
3. Re-openings and earnings as explicit jump mixtures with a random jump minute.
4. Walk-forward refits (monthly) to measure the decay of a profile and of the parameters properly.
5. WTIOIL and HYPE post-mortems; volume/trade-count features; the unfinished ablations (extras, v2-like).

## Appendix — champion per asset (2026-10-08 → 2026-11-07, CRPS and paired rank among miners)

24h: XAU 1255.88 (1/228), NVDAX 2313.13 (1/230), GOOGLX 2114.14 (55/236), TSLAX 2812.17 (1/232), AAPLX 1529.61 (5/233), WTIOIL 2919.58 (225/233), SPCX 3772.52 (43/225), SP500 853.07 (1/235), BTC 2495.26 (1/235), ETH 3502.18 (1/233), SOL 4214.10 (1/229), XRP 3946.66 (1/233), HYPE 5280.17 (88/234).

1h: BTC 538.15 (1/226), ETH 752.41 (1/226), SOL 916.59 (1/226), XRP 856.93 (1/226), HYPE 1133.73 (34/226).
