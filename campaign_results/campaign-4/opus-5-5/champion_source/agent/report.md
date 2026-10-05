# Synth SN50 price-path model: campaign report

Audience: Synth miners. What worked, what didn't, and by how much. Every number below is a
CRPS measured with synth-lib's own scoring (`agent/predict.py`, or `calculate_crps_for_miner` /
`calculate_vol_crps_for_miner` via `agent/harness.py` on the field's own prompt times). Lower is
better. Paired "% change" numbers compare two models on identical prompts.

## 1. Context

- Three competitions are in scope. CRYPTO_24H (BTC, ETH, SOL, XRP, HYPE; 289 points, 5-min steps),
  COM_EQU_24H (XAU, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL, SPCX, SP500) and CRYPTO_1H (crypto;
  61 points, 1-min steps).
- Data: a frozen 1-min OHLCV snapshot ending 2026-09-14. No lookahead; everything runs offline.
- 24h score: CRPS on 5-min, 30-min and 3-h changes (bps) plus the 24-h absolute price. **The
  5-min term is about 60% of the total.**
- 1h score: 1/2/5/15/30-min changes, 60-min abs and 12 gap intervals. From 2026-09-22 there is
  also a vol-CRPS term, but it is only about 1% of the 1h score.
- Validation: two out-of-sample windows. For each, the artifacts are fitted only on data before
  the window starts.
  - V1 = 08-01→08-15.
  - V2 = 08-29→09-13.
  - Paired runs use the field's prompt times (stride 2–3 for tuning).

## 2. Final model (`agent/modeling.py`)

The per-minute variance is v(t) = scale · L(h) · S(t) · E(t). Innovations are fat-tailed with a
per-path vol multiplier:

1. **S(t): weekly seasonal profile.** 2016 bins of 5 minutes, one week of **New York local time**
   (so DST is handled). Fitted per asset from 1-min squared returns (`fit_season.py`).
   - Crypto: returns normalised by a centered 3-day mean, 180 days.
   - Equities/commodities: returns normalised by a centered **7-day (whole-week)** mean. The
     profile is built from mean |r|, squared, which is robust to rare jumps in thin bins. It
     uses 30-day half-life recency weights.
   - Mon–Thu are 50% pooled. Friday keeps its own shape. 3-tap smoothing.
2. **E(t): scheduled FOMC releases.** At 14:00 NY, variance is multiplied over 0–30, 30–90 and
   90–240 minutes after the release, per asset class. The factors are applied to the context
   too, so the level is not double counted after the event.
   - crypto 4 / 2.5 / 1.3
   - XAU and SP500 4.5 / 3.5 / 1.8
   - WTIOIL 1.4
   - xStocks 2 / 1.5 / 1
3. **L(h): deseasonalised variance level.** A horizon-bucketed log-HAR (`harfit.py`), fitted by
   QLIKE and pooled per class (crypto / equities).
   - L_B = l7d · exp(β_B · [1, log(EWMA_hl / l7d)]).
   - EWMA half-lives are 15, 60, 240, 720 and 1440 min.
   - Buckets are 0–5, 5–15, 15–60, 60–180, 180–480 and 480–1440 min; interpolation is in log
     horizon.
4. **Innovations.** Student-t(5) per step times a per-path lognormal multiplier exp(0.4·Z − 0.16).
   The −½σ² drift makes each path a martingale in price.
5. **Per-competition calibration.**

   | Competition | scale | sv  |
   |-------------|-------|-----|
   | C24         | 0.75  | 0.4 |
   | E24         | 0.9   | 0.4 |
   | C1H         | 1.0   | 0.4 |

Runtime is 0.01–0.06 s per prompt at 1000 paths, with no network. The model survives NaN
minutes (dropped, never interpolated), short context and an empty context (fallback variance).
The seed comes from `SYNTH_BENCHMARK_SEED`.

Artifacts (`agent/artifacts/season.npz`, `har.json`) are refitted on all data up to 2026-09-14.

## 3. Results

**Out-of-sample, V2 (08-29→09-13), artifacts fitted before 08-29, 1000 paths.** Field
references are the per-prompt median and 10th percentile of all miners' CRPS.

| config | asset | champion | starter | vs starter | vs field median | vs field p10 |
|---|---|---|---|---|---|---|
| 24h | BTC | 2283.1 | 2443.8 | −6.6% | +0.11% | +1.2% |
| 24h | ETH | 3097.0 | 3272.3 | −5.4% | +0.02% | +1.1% |
| 24h | SOL | 3963.7 | 4140.6 | −4.3% | −0.02% | +1.1% |
| 24h | XRP | 4136.7 | 4424.3 | −6.5% | −0.19% | +1.3% |
| 24h | HYPE | 4829.2 | 4968.6 | −2.8% | +0.19% | +1.2% |
| 24h | XAU | 1382.2 | 1607.2 | −14.0% | +0.07% | +1.8% |
| 24h | NVDAX | 1660.3 | 2063.1 | −19.5% | −0.67% | +1.5% |
| 24h | GOOGLX | 1377.3 | 1608.7 | −14.4% | −0.34% | +1.5% |
| 24h | TSLAX | 2255.0 | 2749.1 | −18.0% | −0.09% | +1.8% |
| 24h | AAPLX | 1452.4 | 1788.1 | −18.8% | −0.63% | +1.6% |
| 24h | WTIOIL | 2968.1 | 3171.7 | −6.4% | +0.72% | +2.0% |
| 24h | SPCX | 2453.2 | 3038.6 | −19.3% | −0.13% | +2.6% |
| 24h | SP500 | 628.6 | 676.7 | −7.1% | −0.26% | +1.5% |
| 1h | BTC | 470.6 | 517.9 | −9.1% | −0.02% | +1.9% |
| 1h | ETH | 665.2 | 711.7 | −6.5% | +0.08% | +2.1% |
| 1h | SOL | 854.5 | 903.8 | −5.5% | +0.04% | +1.9% |
| 1h | XRP | 905.7 | 980.5 | −7.6% | +0.03% | +2.0% |
| 1h | HYPE | 1007.9 | 1059.9 | −4.9% | −0.04% | +1.8% |

**The per-prompt field p10 is not a single miner.** It is an order statistic across ~240 miners,
so it sits 1–2% under the median by construction. Against *consistent single miners* (≥95%
prompt coverage over V2), the best miner beats this model by only:
- 0.3–1.6% on E24
- 0.5–0.8% on crypto 24h
- 0.2–0.4% on crypto 1h

On most asset-configs the model beats the median miner.

**Official `agent/predict.py` backtests of the final committed model**, 1000 paths. These are
in-sample for the refitted artifacts:
- CRYPTO_24H, 7 days to 09-13:
  - BTC 2121.0 (starter 2275.0)
  - ETH 2983.7 (3143.4)
  - SOL 3750.9 (3875.0)
  - XRP 3871.0 (4070.6)
- CRYPTO_1H, 3 days to 09-12:
  - BTC 529.2 (544.7)
  - ETH 737.6 (759.6)
  - SOL 882.9 (900.6)
  - XRP 935.2 (958.7)
- COM_EQU_24H and HYPE could not be run through `predict.py` (see Limitations). They were scored
  with `harness.py` on 09-06→09-13 field prompts:
  - XAU 1341.6 (1559.9)
  - NVDAX 1572.9 (1787.1)
  - GOOGLX 1463.5 (1641.2)
  - TSLAX 1905.5 (2406.7)
  - AAPLX 1537.9 (1888.4)
  - WTIOIL 3128.6 (3373.9)
  - SPCX 2671.7 (3268.6)
  - SP500 620.1 (668.9)

## 4. Experiments, in order (details in `journal.md`)

| # | idea | result (paired % CRPS) | verdict |
|---|---|---|---|
| 1 | v1: NY-time weekly profile + EWMA level blend + t(5) + per-path vol | vs starter −2…−16% on every asset-config | base |
| 2 | ablations: flat seasonality | +7.5% E24, +2% C24, +1.5% C1H | seasonality is the #1 ingredient |
| 3 | QLIKE log-HAR level forecast | OOS QLIKE better on every asset and bucket; CRPS E24 −0.95, C1H −0.61, C24 ≈0 until scale 0.75 (−0.4) | keep |
| 4 | HAR fitted directly on Gaussian CRPS | no gain over QLIKE + one scalar; worse on E24 | discard |
| 5 | OU (within-path) log-vol | ≈ same as a scale change | discard |
| 6 | oracle: realised future level (research only) | best −0.8% C24, −0.1% C1H | level forecast is near saturation |
| 7 | **whole-week profile normalisation** (the 3-day window overstated weekend variance 2–3×) | E24 −0.4…−0.9% | keep |
| 8 | zero-return atom (stale / closed markets, tick size) | ±0.05% | discard |
| 9 | Student-t → filtered historical simulation (empirical standardised residuals) | ≈0 at best | discard |
| 10 | scale ±11%, nu 4/8, sv 0.3/0.55, horizon-tilted scale | all within ±0.3%, sign flips between windows | calibration is at a flat optimum |
| 11 | variance-ratio check for microstructure mean reversion | VR ≈ 0.9–1.1 | nothing to exploit |
| 12 | **FOMC release multipliers** | on overlapping prompts: 24h −0.7% (crypto −1…−1.7, SP500/XAU up to −3.5), 1h −1.2% | keep |

## 5. Lessons for miners

1. **Get the weekly seasonality right, in exchange-local time.**
   - It is the biggest single lever: 2–8% of CRPS.
   - For equities and commodities, normalise returns by a *whole-week* rolling level. A 3-day
     window silently inflates weekend variance 2–3×.
2. **A better conditional-mean variance does not mean a better CRPS.**
   - CRPS prefers a predictive variance *below* the mean forecast variance when vol is uncertain.
   - Calibrate one scalar per competition: 0.75 for crypto-24h.
3. **Past a decent HAR, the level forecast is nearly saturated.** Even a perfect oracle of the
   future realised level buys < 1%.
4. **Distribution shape barely matters.** t(5) × lognormal(0.4), empirical residuals and a
   point mass at zero all land within ±0.1%.
5. **Scheduled macro events are free information.** The FOMC calendar is published a year ahead.
   Crypto vol in the first 30 min after the statement is 4–6× normal.
6. **Weekend and closed-market prompts dominate relative losses but not the mean.**
   - Their absolute CRPS is small, so tuning for them hardly moves the competition score.
   - Weekday active-hour prompts dominate the mean.
7. **The field is tightly clustered.** A good vol model reaches the median; the best consistent
   miner is under 1–2% ahead.

## 6. Limitations

- The live backtester has no offline bundle: `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` is unset.
  - `predict.py` needs `--eval-end ≤ 09-13` (24h) or `≤ 09-12` (1h).
  - `predict.py` cannot score any equity, or HYPE, because synth-lib's `MinutePriceStore.load_range`
    rejects windows with NaN minutes, and those assets have NaN minutes every week. They were
    scored through the same synth-lib CRPS functions in `harness.py`.
- The FOMC multipliers are fitted on 5 events, and their test is partly in-sample. The factors
  were rounded down on purpose.
- Tuning covered only two 2-week windows. Differences under ~0.3% flip sign between windows and
  should be read as noise.
- Artifacts are static. They are refitted to 09-14 and not updated during the 10-day evaluation.
- The vol-CRPS term (from 2026-09-22) was measured but not tuned for (it is about 1% of the 1h
  score).

## 7. With more budget

- Live, rolling refits of the profile and HAR. Recency-weighted profiles helped equities.
- A full economic calendar (CPI, NFP, PPI, retail sales, crypto-specific unlocks) with
  per-event multipliers, validated leave-one-event-out.
- Cross-asset level features (BTC → alts; SP500 → xStocks), and range-based (Parkinson/
  Garman-Klass) inputs for the 1h level.
- A per-prompt ensemble of 2–3 calibrations. The field's p10 advantage suggests there are
  per-regime gains that a single scalar cannot capture.
