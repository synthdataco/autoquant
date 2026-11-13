# Journal — campaign-7 (opus-5-5)

Conventions: every CRPS below comes from synth-lib (`score_crps` / `backtest`), via
`agent/predict.py` or my batch harness `agent/research/evalrun.py` (same scoring calls, many
prompts at once, saves per-prompt rows + the field's best/p10/p50 on each prompt from the offline
bundle). "vs_p10%" = sum(my CRPS)/sum(field p10 CRPS) - 1 over the same prompts (lower = better).
Field-relative columns are only for orientation; decisions are made on CRPS.

## Iteration 0 — setup and reading (2026-11-10, h0-h1)
- Scoring read from synth-subnet: 24h = CRPS (bps) of 288 5-min, 48 30-min, 8 3-h returns + final
  price (bps of final price). 1h = 60x1m, 30x2m, 12x5m, 4x15m, 2x30m, 1 abs, 12 "gap" returns
  (start->5,10..60 min) + vol term 5.25*(mean over block sizes of per-block CRPS of realised
  std of 1-min returns, blocks 60/15/5 min).
- Real path point 0 = close of bar labelled start_time; context ends at bar start-3min (24h) /
  start-2min (1h).
- Field (offline bundle, scored since 08-03): median miner is only +1.5..3% (24h) / +5.5% (1h)
  above the per-prompt best; p10 is ~+0.7% (24h) / +3.5% (1h). Very tight packing.
- Data facts: weekend/weekday variance ratio 0.13-0.26 for tokenised stocks, 0.3-0.5 XAU/SP500/
  WTI, 0.55-0.72 crypto. NY-clock hourly profiles more peaked than UTC ones for every asset
  (incl. crypto: peak 10:00 NY). 1-min variance ratios VR(k) ~ 1 for crypto (no bounce at 1 min).
  Zero 1-min changes: BTC 5.6%, SOL 9.7%, XRP 7.8%, ETH 2.8%, HYPE 1.6% (real, not stale data).

## Iteration 1 — v1: seasonal EWMA scale mixture
Hypothesis: a weekly NY-time vol profile (minute of week) x EWMA level beats the constant-vol
starter on every configuration, by >5% on COM_EQU_24H.
Change: modeling.py v1 (profiles fitted on all data — IN-SAMPLE for the window below; to be
redone walk-forward), fast/slow EWMA (4h/2d half-lives) blended with reversion over 6h, t(5)
innovations, per-path lognormal variance multiplier sd 0.35. CONTEXT_DAYS=14.
Result (validator prompt times, scored 2026-10-01..10-15 for 24h, 10-01..10-08 for 1h, mean CRPS):

| comp | asset | starter | v1 | v1 vs field p10 |
|---|---|---|---|---|
| crypto-24h | BTC/ETH/HYPE/SOL/XRP | 2323/2920/4575/3893/3879 | 2202/2791/4459/3753/3669 | +1.2/+1.4/+0.6/+1.0/+1.7% |
| com-equ-24h | AAPLX/GOOGLX/NVDAX/SP500/SPCX/TSLAX/WTIOIL/XAU | 1456/1916/1838/699/3028/2065/3184/1309 | 1230/1612/1610/638/2556/1715/2902/1172 | +3.9/+2.4/+1.2/+2.0/+3.4/+2.0/+2.4/+1.2% |
| crypto-1h | BTC/ETH/HYPE/SOL/XRP | 503/582/989/757/818 | 464/538/950/705/738 | +2.7/+2.8/+1.8/+2.4/+2.8% |

Verdict: keep (all 18 asset cells improve, 3-16%). Lesson: seasonality is the big first lever
for equities; v1 sits near the field median, ~1-3% behind the field's p10.

## Validation protocol (fixed from iteration 2 on)
Walk-forward profiles: window A = profiles fitted on data < 2026-08-01, scored 2026-08-05..08-26
(1h: ..08-15); window B = profiles < 2026-10-01, scored 2026-10-08..10-29 (1h: ..10-18).
Validator prompt times. `compare.py` = paired per-asset % change of mean CRPS, "better_cells"
= assets improved, t_day = day-clustered t-stat. A change is kept only if it helps on both
windows in most cells. NB: 1h field numbers before 2026-09-22 compare our vol-term CRPS with the
field's no-vol-term CRPS (~1% bias against us); window B is clean.

## Iteration 2 — one-at-a-time sweep of v1 (crypto-24h, window A)
Result: all effects within +-0.35%: longer persistence of the fast level (t_mr 1440: -0.34%),
shorter slow half-life (1d: -0.14%), more mixing (mix_sd 0.5: -0.05%), scale 1.15 worse (+0.17%).
Lesson: global shape knobs are worth tenths of a percent; the level term structure matters more.

## Iteration 3 — v2: HAR-style lead-dependent level (QLIKE-fitted)
Hypothesis: a convex mix of 3 EWMAs (fast/med/slow) of deseasonalised variance, with weights
depending on the lead, forecasts better than 2 EWMAs. Exploration (QLIKE on variance, NOT CRPS,
data < 08-01): crypto 5-min best mix at leads 0-1h/1-4h/4-12h/12-24h over EWMA(1h,6h,8d) =
(.45,.40,.15) -> (0,.70,.30); equities (.65,.25,.10) -> (.05,.45,.50); crypto 1-min for the 1h
prompt over EWMA(10m,1h,8h) = (.55,.20,.25) at 0-5 min -> (.20,.25,.55) at 30-60 min. v1's 1h
half-lives (4h/2d) were far too slow.
Result (window A, CRPS vs v1): crypto-24h -0.62% (5/5), com-equ-24h -6.71% (8/8; v1 was +10%
vs field p10 there), crypto-1h -1.99% (5/5). Verdict: keep.

## Iteration 4 — exactly-zero returns (staleness mass)
Data: zero 5-min returns on Saturdays: XAU 42%, tokenised stocks 10-17%; zero 1-min returns on
Saturdays BTC 15%, SOL 15%. Hypothesis: a point mass at 0 (profiled share per minute of week,
variance moved to the moving steps) cuts CRPS in calm periods.
Result (A): crypto-1h -0.02%, com-equ -0.15% (8/8, XAU -0.41%), crypto-24h 0.00%. Verdict: keep
(cheap, never hurts), but it is NOT the explanation of the calm-period gap.

## Iteration 5 — where is the gap? weekends; robust (absolute-return) level
Gap to field p10 by weekday: com-equ Saturday-start prompts +8.5% (vs ~2% other days), XAU
Saturday prompts up to +46%. Inspected XAU 2026-08-22 07:43: model 5-min sigma 2-3.5 bps vs
realised rms 0.3-1.3 bps (53% zero steps): Friday-night activity inflated the level, then the
weekend went dead. Weekend activity is extremely skewed across weekends (Sat 5-min rms 0.15-4
bps for XAU), so a MEAN-square profile overstates the typical weekend (Sat/weekday variance
ratio: mean-square profile 0.094 vs mean-|r| profile 0.036 for XAU).
- weekend-only EWMA level on weekend steps: no gain (A: +0.01..+0.10%). Discard.
- est="abs": level from EWMAs of deseasonalised |r|, profile from mean |r| (squared), sigma =
  sqrt(pi/2 * abs_scale) * level. A: com-equ -0.79% (7/8), crypto-24h +0.16%; abs_scale 1.2:
  A com-equ -0.71% (8/8), crypto -0.04%; B com-equ -0.45% (7/8), crypto -0.22% (5/5); 1h A -0.11%
  (5/5). Hybrid (squared level, abs profile) slightly weaker. Verdict: keep est=abs, scale 1.2.
- fast intra-path stochastic vol (to fix a peaked+U-shaped 1-min PIT for BTC): worse on every
  cell (+0.05..+0.5%). The PIT shape comes from level-forecast dispersion, not from the
  innovation law. Discard.

## Benchmark: the best SINGLE miner, not the per-prompt p10
No individual miner reaches the per-prompt p10 (it mixes different miners per prompt): the best
single miner (full coverage) in window A is +2.1% vs p10 in 1h, +1.1% in crypto-24h, +1.2% in
com-equ-24h, and the top of the field is made of clones with identical scores. evalrun now
reports vs_bestminer% (best single miner over the whole competition, chosen ex post).
v3 (committed) vs best single miner — window A: crypto-24h +0.11%, com-equ -0.17%, 1h +0.64%
(biased, see above); window B: crypto-24h +0.31%, com-equ +0.20%, crypto-1h +0.42%.

## Iteration 6 — CONTEXT_DAYS and lead-growing mixture (B / A+B)
- CONTEXT_DAYS 10/14/30 vs 21: all within +-0.05% (B). Keep 21 (the 8-day slow EWMA needs it).
- per-path log-variance random walk (mix uncertainty growing with lead): A com-equ -0.08%,
  B com-equ +0.07%, crypto +0.03/-0.05: sign flips between windows -> noise. Discard.

## Iteration 7 — scheduled events from the calendar (walk-forward, leave-window-out)
Exploration: per event, mean |r|/sqrt(profile) in windows after the release over the previous
24h level, median over events (pooled over assets of a group), fitted only on events before
each validation window (events_<cutoff>.json). Variance multipliers (median^2), crypto/equ:
CPI first minute 101x/58x, 1-5 min 9x/6x, 5-15 min 3.9x/2.0x, 15-30 min 2.4x/1.6x; FOMC 30-60 min
7x/4.6x (press conference); NFP 1-5 min 5.9x/3.7x; EARNINGS (equ) 0-2h after the release 60-130x,
2-6h 6.8x, overnight+next open 3.3x; GDP/PCE/PPI/RETAIL ~1.5-3x at 1-15 min; EIA ~1 (noisy).
Profiles re-fitted EXCLUDING event windows (else the 08:30 bucket double-counts). Context returns
inside event windows are divided by the multiplier before entering the level EWMAs.
Result vs v3, prompts whose horizon holds the event (A / B / C):
  CPI: 1h -5.7% / -10.4%; crypto-24h -0.36% / -0.90%; com-equ -0.14% / -0.67%
  EARNINGS (com-equ): -3.9% / -7.8% / -9.9%;  FOMC com-equ -1.7% (B) / -3.0% (C), crypto +0.5/+0.1%
  NFP mixed (+0.1..+0.6% crypto C, -1.2% com-equ A); EIA mixed (+0.6% A) -> EIA dropped.
  All prompts: A crypto-24h -0.02, com-equ -0.10, 1h -0.07; B -0.07 / -0.29 / -0.15;
  C +0.09 / -0.49 / -0.01. The trimmed-mean multipliers are no better than the medians.
Verdict: keep (v4: median multipliers, no EIA). The eval window holds CPI 11-10, PPI 11-13,
RETAIL_SALES 11-17 and NVDA earnings 11-18 21:00 UTC.
v4 vs best single miner, window C (10-29..11-08, after the NY DST switch): crypto-24h +0.32%,
com-equ +0.62%, crypto-1h +0.54%.

## Iteration 8 — minute-of-hour pattern; knob re-check under v4
- Profiles with the minute-of-hour ratio restored after smoothing (first minute of each hour/half
  hour: |r| +20-25% for crypto, +35-50% for SP500/NVDAX): A 1h -0.01%, B 1h -0.04% (5/5), 24h
  -0.00/-0.01% (all cells same sign). Keep (v4.1).
- One-at-a-time knobs around v4.1 on A and B (abs_scale 1.1/1.3, mix_sd 0.25/0.5, nu 4/7): every
  change within +-0.16%, and the com-equ winners on A (scale 1.1 -0.09%, mix 0.5 -0.07%, nu 4
  -0.05%) all flip sign on B (+0.10/+0.04/+0.10%). The current knobs sit at the optimum the
  windows can resolve. No change. Lesson: stop tuning knobs; only structure can still pay.

## Iteration 9 — high-low range in the level measure
Hypothesis: mixing the bar range (Parkinson-style, self-calibrated to the |r| level over the
context) into the absolute-return measure cuts the noise of the fast EWMAs (1h prompts).
Result (range weight 0.3 / 0.6): 1h A -0.00/+0.01%, B -0.03/-0.04%; crypto-24h A -0.03/-0.04%,
B +0.01/+0.02%; com-equ A +0.01/+0.02%, B -0.01/-0.01%. Verdict: discard (option left off).
Lesson: as the constitution warned, the range adds nothing the EWMA of realised moves has not
already priced.

## Iteration 10 — the price grid (tick rounding) for the 1h prompts
Hypothesis: SOL (tick 1.16 bp) and XRP (0.9 bp) print on a coarse grid relative to a 1-min sigma
of 3-8 bp, so their realised 1-min returns are lattice-valued; rounding the simulated prices to
the grid seen in the context (power of ten, applied only when tick/price > 0.2 bp) should match
it better. The grid creates zeros by itself, so the explicit zero mass is switched off when
rounding (zero_mult_tick).
Result (SOL, XRP only — the others are below the 0.2 bp threshold):
  rounding + profiled zeros: A -0.13%, B -0.02%; rounding, no explicit zeros: A SOL -0.21%,
  XRP -0.11%; B SOL -0.11%, XRP -0.06% (t_day -15.5 / -5.3). Verdict: keep for 1h.
24h prompts (A / B): rounding + half the profiled zeros: com-equ -0.01/-0.01%, crypto -0.02/
-0.01% (SOL/XRP only); without explicit zeros XAU loses (+0.17/+0.10%). Verdict: keep, tiny.
v5 = tick rounding (24h: zero_mult_tick 0.5; 1h: 0).

## Iteration 11 — is the profile shape / weekend level right? (A / B)
- profile exponent 0.9 / 1.1 (flatter / more peaked day): com-equ +0.09/+0.08% (A), +0.18/-0.03%
  (B); crypto -0.04/+0.07 (A), +0.01/+0.03 (B). The fitted shape (exponent 1) is right.
- weekend variance x0.8 / x1.25: com-equ -0.06/+0.13 (A), -0.02/+0.07 (B, only 3/8 cells better
  at 0.8); crypto +0.02/+0.08, 0.00/+0.10. Weekend level right within ~10%. No change.

## Iteration 12 — the Friday-afternoon prompts (worst com-equ prompts in window C)
Worst prompts vs the field median in C are SP500/AAPLX prompts starting Friday 16:00-23:00 NY:
+30-40%. Inspected SP500 Fri 11-06 16:15 NY: model sigma 3.3-4.1 bp across the horizon,
realised 0.45-1.8 bp — the hot Friday session (last-6h rms 10 bp) inflated the level, and the
weekend does not follow it.
- weekend steps use slow-heavy EWMA weights ([0,.3,.7] or [0,0,1]) for every prompt: A com-equ
  +0.06/+0.15%, crypto +0.56/+1.43%; B +0.05/+0.13%, crypto +0.01/+0.05%; C -0.02/-0.05%. Much
  worse: inside a weekend, the weekend's own recent minutes are the best information. Discard.
- only for prompts that START on a weekday (the regime-crossing case): com-equ A +0.01/+0.03%,
  B -0.01/0.00%, C -0.04/-0.09%; crypto A +0.30/+0.59%, B -0.01%, C -0.09/-0.11%. Sign flips
  between August and late October: whether a hot Friday is followed by a dead weekend is itself
  regime-dependent. Discard (option `we_wts` left off).

## Iteration 13 — official numbers for the nomination; state-dependent mixing; stratified draws
Official agent/predict.py (1000 paths, validator prompt times, 10 days scored before 2026-11-08;
NB in-sample for the v5 artifacts), mean CRPS over assets — starter vs v5:
  CRYPTO_24H 4606.8 -> 4439.6 (-3.6%); COM_EQU_24H 3304.0 -> 3039.0 (-8.0%);
  CRYPTO_1H 988.4 -> 926.8 (-6.2%). v5 nominated (commit 4a9c5ff) as a safety net.
- mix_state (mixing sd grows with |ln fast/slow|: more level uncertainty when regimes move):
  0.3 / 0.6 -> A 1h 0.00/0.00, crypto-24h +0.01/+0.08, com-equ -0.01/-0.02; B 0.00/-0.02,
  -0.00/+0.01. Nothing. Discard.
- stratified (Latin-hypercube) draws: the per-path multiplier and every step's innovations cover
  all N strata (random pairing across steps). Rationale: the plug-in ensemble CRPS of N iid draws
  is biased up by E|X-X'|/(2N) ~ 0.1% of CRPS at N=1000; stratified marginals remove most of it.
  A: 1h -0.08% (5/5, t_day -5.7), crypto-24h -0.06% (5/5, t -10.4), com-equ -0.07% (8/8,
  t -10.4); B: -0.06% (5/5), -0.07% (5/5), -0.07% (8/8). Verdict: keep — free, and every cell
  agrees. Implemented with a logit-spaced quantile table (|err| < 1e-7, 0.1 s per 24h prompt)
  and exact-count stale steps.
  (Window C of this sweep was contaminated: I edited modeling.py while it ran, so its later
  configs loaded the new default. sweep.py now freezes a snapshot of the model per sweep.)

## Iteration 14 — half-lives; where does the best 1h miner beat us?
- EWMA half-lives around the QLIKE fit: 1h fast 5/20 (vs 10): +0.05/-0.02% (A), +0.03/+0.03% (B);
  1h slow 1440 (vs 480): -0.04% (A), 0.00% (B); 24h fast 30, medium 240, slow 4d: within
  +-0.06%, signs flip A->B. The fitted half-lives are the optimum. No change.
- Per asset vs the best single miner (window C): com-equ AAPLX -0.02, GOOGLX +1.35 (earnings
  week), NVDAX -0.62, SP500 +1.19, SPCX +0.05, TSLAX +0.73, WTIOIL +1.04, XAU +0.57%; 1h: +0.24
  (XRP) .. +0.57% (ETH), uniform -> structural.
- (ours - best 1h miner)/p50 by quintile of realised/forecast vol (C): +2.31, +0.78, +0.17,
  -0.19, +0.13%. They win when the hour turns out calmer than we forecast. (Binning on our own
  error favours the other miner by construction; the asymmetry is the informative part.)
- Trade count does not help: OLS of log next-hour |r| on log HAR forecast (slope 0.95-1.00,
  R2 0.66-0.78) gains only +0.0002..+0.003 R2 from the trade-count surprise (coef 0.02-0.07).
- PIT of the realised 60-min / 15-min volatility (the 1h vol term's statistic) inside our paths'
  distribution of it (B, BTC/SOL/HYPE): U-shaped, 14.4/11.3 .. 9.7/12.5% at the deciles: our
  simulated realised vol is too concentrated (the per-path multiplier is the only source of
  vol dispersion within a path), and slightly high.
- Wider per-path dispersion with thinner step tails for 1h (keeps the step marginal, widens the
  distribution of each path's realised vol): mix_sd/nu 0.5/6: A -0.03% (4/5), B -0.04% (5/5);
  0.6/7: A -0.04% (4/5), B -0.05% (5/5); 0.6/5: -0.02/-0.01%; 0.7/8: -0.03/-0.04%.
  Verdict: keep 0.6/7 for 1h (v7). Small, consistent.

## Iteration 15 — final evaluation of v7 (official + untouched holdout)
Official agent/predict.py, 10 days scored before 2026-11-08 (in-sample for the artifacts),
starter -> v5 -> v7 mean CRPS: CRYPTO_24H 4606.76 -> 4439.62 -> 4436.95 (-3.69%);
COM_EQU_24H 3304.04 -> 3039.04 -> 3037.12 (-8.08%); CRYPTO_1H 988.40 -> 926.78 -> 925.40 (-6.37%).
Holdout S (artifacts fitted < 2026-09-01; scored 09-05..09-26, 1h 09-23..10-03; never used for a
decision): v7 vs starter 4148.5 vs 4284.6 (-3.2%), 1803.6 vs 2066.0 (-12.7%), 917.2 vs 959.2
(-4.4%); vs the best single field miner -0.01% / +0.00% / -0.16% (vs field p50: -0.45/-0.29/-0.28%).
Harness interface check: ok (24h + 1h), CONTEXT_DAYS 21, 0.2 s / 0.0 s per call.
Nominated v7 (commit 6f1e6cd). Report written.

## Iteration 16 — weekend uncertainty: a per-path "dead or active weekend" draw
Hypothesis: weekend activity varies ~10x from one weekend to the next (XAU Saturday 5-min rms
0.15-4 bp), far more than weekday levels, so weekend steps need their own, wider per-path
variance factor (stratified log-normal, mean 1) on top of the common one.
Result (we_mix_sd 0.5 / 0.8, all assets; A / B / C):
  com-equ: 0.5 -> -0.04% (8/8) / -0.02% (4/8) / -0.05% (8/8); 0.8 -> -0.08 (7/8) / -0.04 (3/8) /
  -0.11% (8/8). crypto-24h: 0.5 -> -0.02 (3/5) / -0.02 (5/5) / -0.10% (5/5); 0.8 -> +0.01 /
  0.00 / -0.21%. Verdict: keep, 0.5 for crypto and 0.8 for the equity/commodity perps (24h only;
  1h untested -> off). v8.
Official (10 days before 2026-11-08): v8 CRYPTO_24H 4432.82 (v7 4436.95), COM_EQU_24H 3035.10
(v7 3037.12); CRYPTO_1H unchanged (925.40). Nominated v8 (commit 8a06e83).

## Iteration 17 — the same idea for the weekday off-session (equity perps)
Hypothesis: overnight activity, like the weekend's, varies more between nights than the level
says. Own per-path factor (sd 0.4 / 0.7) on weekday 16:00-09:30 NY steps of the equity perps.
Result: A -0.06/-0.09% (7/8), B +0.05/+0.20% (0/8), C +0.09/+0.31% (0/8). Discard (option off).
Lesson: weekends are a separate regime whose level the context cannot pin down; overnights
follow the session closely enough that extra dispersion only blurs the fan.

## Iteration 18 — per-path intensity of release reactions; 1h weekend factor
- ev_mix_sd (each path draws how much of a release's EXCESS variance it gets, mean kept):
  0.7 / 1.2 -> 1h A 0.00/+0.01%, B +0.01/+0.03%; on CPI-spanning 1h prompts +1.3% (A) / +2.6% (B)
  worse; earnings-spanning com-equ prompts -2.0% (A, n=20); 24h overall ~0. Discard: a release's
  reaction is better described by the median multiplier with t tails than by a
  surprise/no-surprise mixture.
- 1h_we_mix_sd 0.3: A -0.00% (4/5), B -0.01% (5/5). Consistent but negligible; not adopted (it
  would need a new official 1h run for ~0.01%).
- Equity weekend factor 1.1 / 1.4 (vs 0.8): A 0.00/+0.07%, B +0.02/+0.10%. 0.8 stays.
- HAR weights refitted with an absolute-return QLIKE (equities: (.7,.15,.15) -> (.15,.25,.6)):
  +0.05% (A), +0.01% (B), +0.03% (C). The squared-QLIKE weights stay. (Crypto abs fit within
  0.1 of the current weights at every lead; not tested.)

## Where the campaign stands (h7.3)
v8 nominated (commit 8a06e83). The last eight experiments (iterations 16-19, eight variants)
moved CRPS by < +-0.1% or made it worse, and every global knob sits at an optimum agreed by two
or three walk-forward windows; on the untouched holdout S v7 already matched the best single
field miner (-0.01 / +0.00 / -0.16%). Remaining budget is kept as margin. Decision: stop here.
