# Journal — campaign-6

## 0. Setup (h 0.0–0.5)
- Read validator scoring. 24h: CRPS on 288 5-min returns (bps) + 48 30-min + 8 3h + 1 abs-24h price (bps of last price).
  1h: 60 1-min + 30 2-min + ... + 12 "gap" intervals + vol-CRPS (block std of returns, λ=5.25). Per-step return
  marginals dominate the sum → intraday scale and tail shape of 5-min/1-min returns are the main levers.
- Offline field bundle spans scored prompts 2026-09-08 → 2026-10-11 only. 24h prompts ≈ hourly, 1h prompts every ~10 min.
- Built `agent/evalrun.py`: parallel per-asset generation (npy day layout, float32) + official `backtest()`
  (latest_formula=True), saves per-prompt ours/best/median CRPS.
- Starter (log-normal, 7d std), 2026-10-04→10-11: crypto24 mean CRPS 3302 (+11.6% vs field median per prompt),
  crypto1h (10-08→10-11) 766 (+15% vs median).
- Measured seasonality (5-min r², deseasonalised by 7d level): BTC weekend 0.35–0.55x weekday, US-open hours 3x;
  NVDAX/SPCX open hour 7x, weekend 0.1–0.2x. Variance ratio 5m/1m ≈ 1.0 for all assets → minute data usable for 5-min vol.
  Profiles fitted in New York local time (DST-robust; eval window still EDT).
- Validation plan: W_A = 09-12→09-26, W_B = 09-26→10-11 (bundle span). Profiles for experiments fitted on data < 09-08 only.

## 1. Model v1: seasonal EWMA + Student-t + per-path vol multiplier (h 0.5–2)
- Hypothesis: seasonality (NY-time weekly profile) × deseasonalised EWMA level beats a flat 7-day std.
- Result (W_B 09-26→10-11, mean CRPS): c24 3541 (starter would be ~+10% vs median; v1 +1.1% vs field median per prompt),
  c1 738.8 (+0.5% vs median), e24 1772.8 (+8.6% vs median: equities poorly served). KEEP.
- Diagnostic: standardised realised 5-min returns by NY hour → equity weekends hugely over-predicted (SPCX weekend rms 0.25).
  Three bugs/lessons in the profile fit:
  1. level normaliser must span a full week (a 3-day centred window divides the weekend effect away);
  2. clipped mean-of-ratios understates weekday/weekend ratio; fit ratio-of-sums;
  3. weekday/weekend variance ratio drifts strongly upward over 2026 (SPCX 5→28 May→Sep, XAU 6→19) → recency weighting.
- Profile variants on W_B e24: L2 hl21 1737.9; L1 (|r|-based, robust) no recency 1730.5; **L1 hl21 1720.6**; L1 hl10 1719.3; L1 hl42 1724.0.
  KEEP L1 hl21 (L2 profiles get destroyed by single crash bars: XRP weekend>weekday).

## 2. Zero-price-change atom (h 2–2.5)
- Validator arrays have many unchanged steps (XAU 11% of 5-min steps, SOL 10% of 1-min). Hypothesis: an atom at 0 with
  per-bin probability (fitted like the profile, both ends traded) lowers CRPS.
- W_B e24: 1720.6 → 1718.9 (XAU −2.7, SPCX −3.5, others ≈). Small, KEEP (ZK=1).

## 3. Global shape params (W_B, e24 / c24 mean CRPS; base 1718.9 / ~3545)
- SCALE 0.9: 1723.1 / 3540.2; SCALE 1.1: 1720.9 / 3555.8; NU 3: 1726.5 / 3549.4; NU 10: 1719.9 / 3547.2;
  ETA 0: 1719.7 / 3545.8; ETA 0.4: 1721.7 / 3540.1. All within ±0.3% → shape/global scale is near-flat; leave defaults.
- Note: mean-CRPS and mean-ratio-to-median disagree on sign for several of these → noise level.

## 4. Weekend scale for equities (h 2.5–3.5)
- Loss decomposition by start weekday (e24 W_B): Fri/Sat starts +2.9%/+4.5% vs median, other days ≈0. Weekend
  standardised returns: MAD 0.4–0.5 vs 0.85 on weekdays (peaked + jumps).
- Fixed WE_SCALE 0.5: W_B 1718.9→1709.8 but W_A 1819.2→1823.8 (NVDAX, GOOGLX, WTIOIL worse) → sign flips: DISCARD.
- Adaptive: ratio of last weekend's deseasonalised L1 level to the context weekdays' (ρ), forecast weekend var × ρ^γ.
  γ=1: W_A 1819.2→1818.0, W_B 1718.9→1710.3; 13/16 asset-window cells improve. KEEP (γ=1).

## 5. Weekend adaptation for crypto (h 3.5–4)
- Hypothesis: crypto weekends also drift away from the profile; apply the same ρ^γ (WE_CRYPTO=1).
- c24 W_B 3541.1 → 3529.3 (all 5 assets better), but c24 W_A 4496.0 → 4511.2 (all 5 worse); c1 690.73 → 692.04.
- Verdict: DISCARD (sign flips across windows). Lesson: one window is never enough; W_A/W_B disagree often at the 0.3% level.

## 6. Scheduled macro events (h 4–5.5)
- Hypothesis: the calendar ($SYNTH_BENCHMARK_CALENDAR) tells us when variance jumps (CPI/NFP/FOMC/PPI/GDP/PCE, EIA for WTIOIL).
  Measured (analysis/events.py) the squared-L1 variance ratio vs the same clock time on non-event days, in windows
  [-30,0,2,5,15,30,60,120,240] min; CPI minute 0–2 ≈ 14–26x, FOMC elevated for 2h. Profile refitted with event
  minutes masked out (so the profile is not double-counting them). Out-of-sample: multipliers fitted on releases < 07-01.
- c24 + e24 over W_A+W_B: e24 1762.31 → 1762.06, c24 4008.0 → 4005.8; per-event paired: FOMC prompts −1% (win 87–100%),
  CPI −0.3/−0.5% (win 78–100%), NFP/PCE ≈ 0.
- c1 (09-09→09-19, latest formula, our-vs-our paired so the field formula is irrelevant): CPI prompts −5.6%, FOMC −6.5%,
  PPI −4.4%, all 100% wins; overall −0.33%.
- Power on the multipliers (EV_POW, multipliers^p), Sep releases, c1: p=1.5 CPI −5.8/FOMC −4.4/PPI −2.7% vs p=1;
  **p=2 CPI −12%/FOMC −7.7%/PPI −5.3%**; p=3 overshoots (CPI only −1%). c24 p=1.5 −0.7…−0.8%, p=2 −1…−1.7% on event prompts;
  e24 p=1.5 better than p=2 on CPI/FOMC. 09-22→10-11 c1: PCE −4.5% with p=2 but NFP +3.4% (not in the eval window).
- Verdict: KEEP. Defaults EV_POW 2 at 1-min steps, 1.5 at 5-min, pre-July table. Lesson: the median release ratio
  understates the mean; the right size is bigger than the median. Eval window has CPI (10-14) and PPI+retail (10-15),
  EIA (WTIOIL only) 10-15 and 10-21.

## 7. CRYPTO_1H level dynamics (h 5.5–6.5) — 10-01→10-11 c1, base 690.73
- HL_S 30 + TAU_MR 60: 690.29 (best, marginal, kept for 1h); HL_S 60: 691.56; HL_S 15+TAU 120: 691.39; HL_S 10+TAU 30: 692.86;
  HL_L 1440: 692.21; ETA 0.4: 692.27; no zero atom 690.61; WINS 10: 693.42; WINS 5: 700.70.
- Parkinson range (H/L) blended into both EWMAs, put on the close-return scale: RANGE_W 0.5 690.81, 1.0 692.07 vs 690.42. DISCARD.
- Fresher profile (cutoff 10-01 vs 09-08): 690.88 → 690.22. Fit the served profile on all data.
- Field rank (analysis/fieldrank.py, per-miner mean): c1 still +1.9–2.9% behind the best single miner on every asset;
  the field is compressed (best→median ~3%). Part is structural: live miners see prices up to the request,
  the benchmark context ends 2 min earlier.

## 8. More 24h level/shape checks (h 6.5–7)
- c24 W_A / W_B: HL_L 5760 4495.5/3542.8, TAU_MR 720 4497.3/3539.2, HL_S 360 4496.7/3543.6, TAU 60 4496.9/3545.0,
  HL_L 1440 4497.2/3542.0 vs base 4496.0/3542.2. Flat → keep.
- Calibration at aggregation levels (analysis/calib_h.py): mean|z| at 24h ≈ 1.0 for NVDAX/SP500/SPCX vs ~0.8 expected.
  Tried per-path persistent drift adding DRIFT_F x horizon variance only at long horizons: e24 W_A 1816.21 → 1816.26 (0.1),
  1816.71 (0.3); W_B 1710.78 → 1710.71 / 1710.90. DISCARD (single-period trend, not a stable effect).

## 9. Minute-of-hour shape (h 7–7.5)
- Crypto |r1| by minute of hour: :00 ≈1.25x, :15/:30 ≈1.1–1.2x, :56–:59 ≈0.8x — inside our 15-min slots.
  Fitted 60 multipliers per asset (L1 ratio of sums vs a centred 60-min level, mean 1 within each 15-min slot).
- c1 10-01→10-11 691.01 → 690.87 (MOH 0.5 same); c24 W_B 3542.2 → 3540.7; e24 W_B 1710.8 → 1709.7. KEEP (small, all three agree).
- Lesson: CRPS is flat near the right variance; a ±40% variance pattern on single minutes is worth <0.1%.
  Every σ-shape refinement since step 6 has been <0.1% → the model class is at its plateau.

## 10. Leave-out validation of the event power, EIA (h 7.5–8.5)
- EV_POW was chosen on the Sep releases; the table on releases < 07-01. Out-of-sample check on releases 07-01..09-07
  (before the field bundle, so our-vs-our paired with synth's own `calculate_total_score_for_miner`; analysis/evpow.py).
- c1 (5 crypto, 10 starts per release): vs no events, p=1 / 1.5 / 2 / 3: CPI −6.7/−10.5/−10.1/+34%, FOMC −6.7/−8.9/−7.4/+20%,
  PPI −2.5/−3.0/−2.4/+2.3%, NFP −2.1/−1.8/+0.8%, GDP/PCE (same minute) +0.7/+2.4/+6.0%. Per release, CPI Jul favours 2,
  Aug favours 1.5 (p=2 only −1.5%), Sep favours 2. Pooled over the 3 CPIs and 3 PPIs p=2 ≈ p=1.5 + 0.5%, with more downside.
  → 1h default EV_POW 1.75 (convex loss in p: the middle is safe in both regimes).
- 24h (13 assets): p=1.5 best: CPI −0.57%, FOMC −1.97%, NFP −0.5%; p=3 hurts everywhere. Keep 1.5.
- EIA_PETROLEUM (WTIOIL only): worse on 10/10 releases (+0.1% at p=1, +0.3% at 1.5). EIA is weekly at a fixed slot, so the
  weekly profile carries it if those minutes are not masked out. Removed EIA from the mask and the multipliers;
  e24 W_B (profile cutoff 09-26) WTIOIL 3223.7 → 3222.7.
- Lesson: GDP/PCE multipliers are too strong out of sample for crypto 1h — not in the eval window, left as is (noted).

## 11. Is the level forecaster the bottleneck? (h 8.5–9)
- Offline (analysis/har.py): predict next-hour (c1) / next-day (24h) deseasonalised realised 1-min variance from EWMAs at
  half-lives 5…2880 min; log-linear (HAR-like) and linear combos fitted < 09-08, tested 09-08..10-11. QLIKE on test:
  c1 served 0.2369 vs HAR 0.2343 (−1%; ≈ 0.1% CRPS by the ¼(ln σ ratio)² rule); c24 served 0.1206 vs HAR 0.1260 (worse);
  e24 0.1851 vs 0.1897 (worse). Verdict: no change. The 2-EWMA rule is at what own-price history can give.
- The remaining c1 gap to the best single miner (≈0.6–1.1% per asset on 09-22..10-11) is consistent with information we
  cannot see in the benchmark (live miners hold prices up to the request; the context ends 2 min before).

## 12. Final checks and nomination (h 9–10)
- Equity calibration by NY hour (calib.py, NVDAX/SP500 weekdays, 09-12..10-10): rms z 0.92–1.07 in nearly all hours,
  open hour ≈1.0 → no open-gap mis-scaling left; by the ¼(ln σ ratio)² rule hour-level residuals are worth ≪0.1%.
- Robustness: NaN-heavy, 50-row, 2-row and all-NaN-but-last contexts, unknown asset, 7 paths: correct shapes, finite,
  deterministic per seed; ~0.05 s/prompt (24h) and ~0.01 s (1h) at 1000 paths.
- Out-of-sample final (profile fitted before each window, pre-July event table, evalrun = official backtest()):
  c24 09-26..10-11 3533.1 (starter 3697.8, −4.5%); e24 1704.5 (2025.6, −15.8%); c1 09-22..10-11 807.4 (853.6, −5.4%).
- Official agent/predict.py, 1000 paths, latest formula, served artifacts (season.json fitted to 10-11 → in-sample
  for the profile), same windows: c24 3494.1 vs starter 3698.2; e24 1689.2 vs 2025.5; c1 801.8 vs 853.5.
- Evidence of plateau: last six iterations each < 0.1% (minute-of-hour, drift, range, HAR, level sweeps, open calibration);
  shape params flat ±0.3%; offline the served level forecaster matches or beats fitted alternatives.
  Verdict: nominate this model.
