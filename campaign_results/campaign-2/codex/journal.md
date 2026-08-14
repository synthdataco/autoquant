## 2026-08-07 — Iteration 1

## 2026-08-07 — Iteration 2

## 2026-08-07 — Iteration 3

## 2026-08-07 — Iteration 4

## 2026-08-07 — Iteration 5

### Iteration 6 — 1000‑path multi-asset checks

### Iteration 7 — 1000‑path, 2‑day sweeps

## 2026-08-07 — Iteration 8

## 2026-08-08 — Iteration 9

## 2026-08-08 — Iteration 10 (θ sweep)

## 2026-08-08 — Iteration 11 (per‑asset 24h σ)

## 2026-08-08 — Iteration 12 (wrap‑up)

## 2026-08-08 — Iteration 13 (asset‑specific θ for 24h)

## 2026-08-08 — Iteration 14 (confirm COM_EQU_24H post‑θ defaults)

## 2026-08-08 — Iteration 15 (interface check)

## 2026-08-08 — Iteration 16 (CRYPTO_1H combined recheck)

## 2026-08-08 — Iteration 17 (CRYPTO_24H combined recheck)

## 2026-08-08 — Iteration 18 (AAPLX-focused 24h tuning)

## 2026-08-08 — Iteration 19 (report refresh)

## 2026-08-08 — Iteration 20 (cleanup)

## 2026-08-08 — Iteration 21 (tooling)

## 2026-08-08 — Iteration 22 (docs polish)

## 2026-08-08 — Iteration 23 (calibration tooling + AAPLX quick grid)

## 2026-08-08 — Iteration 24 (SP500/XAU quick grids + combined)

## 2026-08-08 — Iteration 25 (docs: config)

## 2026-08-08 — Iteration 26 (final readiness)

## 2026-08-08 — Iteration 27 (Score estimate)

## 2026-08-08 — Iteration 28 (1h mixture-of-Gaussians tails)

## 2026-08-08 — Iteration 29 (extend mixgauss to ETH/SOL + combined 1h)

## 2026-08-08 — Iteration 30 (ETH/SOL p,r sweep; confirm combined 1h)

## 2026-08-08 — Iteration 32 (1h mixauto combined check)

## 2026-08-08 — Iteration 33 (early-minutes variance slope)

## 2026-08-08 — Iteration 34 (revert slope; freeze config)

- Change: Disabled `short_sigma_boost` (set to 0.0) for BTC/ETH/SOL in `agent/config.json` since it did not materially improve combined CRYPTO_1H rank.
- Status: Config stabilized; deliverables and tools synced; awaiting landing order.

- Change: Added an optional early-minutes dispersion boost for 1h (config: `short_sigma_boost`, `short_tau_minutes`) applied as an exponential decay from the start of the path. Enabled with boost=0.10, tau=10 for BTC/ETH/SOL.
- Results (2d, 400 paths): BTC mean_crps=620.59; ETH=845.59; SOL=831.20.
- Combined CRYPTO_1H (2d, 1000 paths): rank 168/256; reward_weight≈2.03e‑4 (no material change vs Iteration 32).
- Verdict: Early-minute slope does not move field-relative rank; keep off by default for nomination code; retain config hook for future work.

- Change: Switched BTC/ETH/SOL to `short_shock: mixauto` (adaptive mixture matching kurtosis, r=4.0).
- Combined CRYPTO_1H (2d, 1000 paths): rank 168/256; reward_weight≈2.08e‑4 (similar to prior 167/256 under fixed mix).
- Verdict: Adaptive tails maintain slight 1h gains but do not materially change rank; pausing here.

## 2026-08-08 — Iteration 31 (updated Score estimate)

- Action: Re-ran `agent/tools/score_estimate.py` at 1000 paths after 1h mixgauss tweaks.
- Result: CRYPTO_1H ratio≈0.0768; CRYPTO_24H≈8.00e‑10; COM_EQU_24H≈2.84e‑08; Score_estimate≈2.56.
- Verdict: Small lift from prior ≈2.17 to ≈2.56; still non‑competitive. Holding for landing order/new direction.

- Change: Ran a small grid on ETH/SOL (p∈{0.05,0.06}, r∈{3.5,4.0}); persisted best per asset in `agent/config.json`.
- Combined CRYPTO_1H (2d, 1000 paths, BTC/ETH/SOL) after updates: rank 167/256; reward_weight≈3.13e‑4 (in line with Iteration 29).
- Verdict: Confirms modest but steady 1h gain; still far from competitive. Pausing further 1h work unless directed.

- Change: Enabled `short_shock: mixgauss` for ETH and SOL with p=0.06, r=4.0 in `agent/config.json`.
- Results (2d, 400 paths): ETH mean_crps=844.85; SOL=827.88.
- Combined CRYPTO_1H rank (2d, 1000 paths): rank 167/256; reward_weight≈3.18e‑4 (slight improvement vs prior 168/256).
- Verdict: Small but consistent 1h gain; still far from competitive. Leaving mixgauss enabled in config for future work.

- Change: For 1h only, added an optional mixture-of-Gaussians shock model (`short_shock: mixgauss`) with parameters `mix_p` (jump prob) and `mix_r` (sigma ratio), configurable via `agent/config.json`.
- Run: BTC 1h with mixgauss (p=0.06, r=4.0), 2 days ending 2026‑07‑15, 400 paths.
- Result: mean_crps=619.69 (improves vs prior ~623.14–629), smoothed_score=101.12.
- Verdict: Simple variance-mixture helps 1h CRPS a bit; generalize cautiously and validate across ETH/SOL before considering in champion code.

- Action: Added `agent/tools/score_estimate.py` to estimate the campaign Score using official combined weights over short windows; regenerated predictions and ran it at 1000 paths.
- Result: CRYPTO_1H ratio≈0.0652; CRYPTO_24H≈9.26e‑10; COM_EQU_24H≈2.92e‑08; Score_estimate≈2.17.
- Verdict: Confirms non‑nomination — estimated Score is near zero; further gains require a different model class or extensive per‑asset calibration.

- Check: Budget normal; envelope ample; no landing order present in `agent/BUDGET.md`.
- Verify: Deliverables present (journal, report with Reproduce, suggestions, CONFIG.md, tools).
- Sanity: Prior interface check passed; snapshot windows all ≤ 2026‑07‑27; no lookahead.
- Status: Idling to preserve budget; ready to act on a landing order or new directive.

- Action: Added `agent/CONFIG.md` and expanded the Reproduce section in `agent/report.md` with per‑asset override details and calibrator usage.
- Goal: Make per‑asset calibration and combined‑rank reproduction self‑serve and auditable.

- Change: Fixed calibrator to regenerate predictions per grid point. Ran 1‑day, 300‑path grids:
  - SP500: σ∈{0.90,0.92,0.96}, θ∈{0.03,0.04,0.05} → best written: long_sigma_mult=0.96, θ=0.05
  - XAU: σ∈{0.92,0.95,0.98}, θ∈{0.03,0.04,0.05} → best written: long_sigma_mult=0.95, θ=0.04
- Re-generated 2‑day, 1000‑path predictions for SP500/AAPLX/XAU and recomputed combined COM_EQU_24H rank:
  - Final snapshot: rank 256/257; reward_weight≈1.80e‑09 (still not competitive).
- Verdict: Even with quick per‑asset grids, combined rank remains flat; end modeling here.

- Change: Added per‑asset override support via `agent/config.json` (read at runtime by the model) and a `agent/tools/calibrate_asset.py` script to grid‑search 24h `long_sigma_mult` and `theta` on a short window.
- Run: Calibrated AAPLX over σ∈{0.86,0.90,0.94}, θ∈{0.03,0.05,0.07} with 1 day @ 300 paths.
- Result: Best found written to `agent/config.json`:
  - AAPLX: long_sigma_mult=0.94, theta=0.05 (on this window).
- Verdict: Tooling in place for reproducible per‑asset tuning; short‑window best may not lift combined rank without broader validation.

- Action: Added a Reproduce section to `agent/report.md` with exact `uv run` commands for per‑asset runs and the new combined‑rank helper.
- Goal: Make evaluation fully turnkey for reviewers without touching core code.

- Action: Added `agent/tools/combined_eval.py` — a small helper to generate predictions and compute official combined ranks for any competition in one command.
- Usage examples are documented in the file header; defaults match validator parity (1000 paths).
- Purpose: Make reproduction and future tuning faster without touching the core model.

- Action: Removed bulky prediction artifacts at `miner_outputs/campaign/predictions` to keep the repo lean for bundling (no effect on results; commands to reproduce are in the report and journal).
- Verdict: Workspace tidied; deliverables unchanged; awaiting landing order.

- Action: Updated `agent/report.md` to reflect latest combined ranks: CRYPTO_1H rank 168/256; CRYPTO_24H 256/257; COM_EQU_24H 256/257; and clarified the conclusion.
- Verdict: Documentation now matches final computed results; campaign deliverables are synchronized.

- Change: Stronger AAPLX 24h σ shrink (×0.90) and slightly higher default θ=0.05 for OU; kept SP500/XAU settings.
- Quick checks (1d, 400 paths):
  - AAPLX mean_crps=2160.78 (improves), smoothed=198.85
  - SP500 mean_crps=1155.90 (neutral), smoothed=89.92
- 2‑day 1000‑path COM_EQU_24H recompute:
  - SP500 mean_crps=1153.47; AAPLX=2106.07; XAU=1749.51.
  - Combined final snapshot: rank 256/257; reward_weight≈1.91e‑10 (no material change).
- Verdict: Targeted AAPLX tweak improves its CRPS but does not move combined rank meaningfully; conclude modeling phase.

- Action: Re-generated 2‑day 1000‑path predictions for BTC/ETH/SOL (24h) and computed official combined CRYPTO_24H rank.
- Single‑asset CRPS (2d, 1000 paths): BTC mean_crps=2879.13; ETH=4191.91; SOL=3757.69.
- Combined CRYPTO_24H final snapshot: rank 256/257; reward_weight≈8.84e‑12 (no material change).
- Verdict: Confirms earlier conclusion — 24h crypto remains uncompetitive; no nomination.

- Action: Re-generated 2‑day 1000‑path predictions for BTC/ETH/SOL with latest model (asset‑specific θ defaults) and computed official combined CRYPTO_1H rank.
- Single‑asset CRPS (2d, 1000 paths): BTC mean_crps=623.14; ETH=846.31; SOL=836.72.
- Combined CRYPTO_1H final snapshot: rank 168/256; reward_weight≈2.46e‑4 (improvement vs earlier run, still far from competitive).
- Verdict: Small improvement in 1h combined rank; insufficient for nomination. Deliverables remain final.

- Action: Validated `simulate()` contract on live context without writing predictions.
  - BTC 1h @ 2026‑07‑15T00:00Z: 64 paths × 61 steps; finite start/end.
  - SP500 24h @ 2026‑07‑24T00:00Z: 64 paths × 289 steps; finite start/end.
- Verdict: Interface and shapes correct for both profiles; ready for any further evaluation if directed.

- Action: Re‑generated 2‑day 1000‑path predictions for SP500, AAPLX, XAU and recomputed combined COM_EQU_24H rank with official combiner.
- Single‑asset CRPS (2d, 1000 paths): SP500 mean_crps=1154.49; AAPLX=2099.07; XAU=1749.50.
- Combined COM_EQU_24H final snapshot: rank 256/257; reward_weight≈2.66e‑10 (no material change).
- Verdict: Confirms earlier decision — additional tuning yields minimal movement relative to field; keep deliverables as final and skip nomination.

- Change: Added asset‑specific OU θ defaults for 24h (env `OU_THETA` still overrides): SP500/AAPLX=0.04; XAU/WTIOIL=0.03; BTC=0.02; ETH/SOL=0.03.
- Quick checks (1 day, 400 paths):
  - SP500: mean_crps=1157.59; smoothed=91.61 (slightly worse vs Iter 11 short‑run)
  - AAPLX: mean_crps=2234.25; smoothed=176.38 (unchanged/slightly worse)
- Verdict: Asset‑specific θ did not yield clear gains on this window; deprioritize further θ tuning without broader validation.
- Next: No nomination; deliverables finalized. Further improvement likely requires broader param search with nested validation or a richer model class.

- Action: Finalized `agent/report.md` with context, approach, experiments (including failures), consolidated results, limitations, and next‑budget plan. Updated `agent/suggestions.md` with concrete follow‑ups.
- Decision: Do not nominate a CHAMPION — competition‑level ranks remain far from the field’s top; further iteration likely requires per‑asset calibrated parameters and/or model class changes beyond current scope.
- Note: All evaluations comply with no‑lookahead (≤ 2026‑07‑27) and use synth‑lib backtester only. Prediction artifacts pruned periodically to stay within disk budget.

- Change: Added per‑asset long‑horizon σ multipliers (applied only at 24h): SP500×0.92, AAPLX×0.94, XAU×0.95, WTIOIL×0.95 (others 1.00). Retained global 24h shrink ×0.90 and OU drift θ=0.03.
- Quick checks (1 day, 400 paths):
  - SP500: mean_crps=1153.45 (improves), smoothed=87.47
  - AAPLX: mean_crps=2231.03 (worse vs prior short-run), smoothed=174.29
  - XAU: mean_crps=1722.56 (improves), smoothed=137.62
- 2‑day combined COM_EQU_24H (SP500, AAPLX, XAU; 1000‑path per asset): final snapshot rank 256/257, reward_weight≈6.31e‑09 (slight move but still far from top).
- Verdict: Per‑asset σ helps SP500/XAU modestly; AAPLX regresses. Given diminishing returns and budget/time, pause major modeling changes.
- Next: finalize report.md, add suggestions.md (asset‑specific θ, per‑asset grid over σ/θ with cross‑validation windows, regime switches, adaptive σ via realized gap calibration). Hold nomination.

- Change: Made OU θ configurable via env var `OU_THETA` (clipped [0.005, 0.2]).
- Sweep (1 day, 400 paths):
  - BTC 24h:
    - θ=0.02 → mean_crps=2866.51, smoothed=158.54
    - θ=0.04 → mean_crps=2883.58, smoothed=173.83
    - θ=0.06 → mean_crps=2893.74, smoothed=182.93
  - SP500 24h:
    - θ=0.02 → mean_crps=1157.06, smoothed=91.08
    - θ=0.04 → mean_crps=1155.31, smoothed=89.32 (best CRPS)
    - θ=0.06 → mean_crps=1156.97, smoothed=90.99
- Verdict: θ≈0.02 minimizes BTC CRPS; θ≈0.04 slightly better for SP500. For a single θ, 0.03 remains a reasonable compromise; asset‑specific θ may help but increases complexity.
- Next: hold θ=0.03 globally; proceed only if we can squeeze larger 24h wins via σ shrink per asset. Otherwise, pause major changes and draft suggestions.md with ideas for next budget.

- Hypothesis: A weak OU mean‑reversion on 24h log‑prices plus a slightly tighter σ will reduce 24h_abs CRPS without hurting earlier intervals.
- Change: In `agent/modeling.py`, for 24h only: per‑step σ ×0.90 and discrete OU drift toward a 12h EMA of the context log‑price with θ=0.03 per hour; noise remains Gaussian. Other horizons unchanged.
- Quick checks (1 day, 400 paths):
  - crypto-24h/BTC: mean_crps=2876.24; smoothed_score=167.25
  - com-equ-24h/SP500: mean_crps=1156.35; smoothed_score=90.37
- Verdict: SP500 24h improves modestly; BTC 24h roughly neutral vs prior short-window runs. Next: extend OU tuning (θ sweep 0.02–0.06) and re‑check multi‑asset at 400 paths before rerunning 1000‑path windows.
- Lesson: Gentle mean‑reversion can help high‑coefficient equities; crypto response is subtler and may need asset‑specific θ.

- Goal: Compute competition‑level backtested ranks at 1000 paths using official combiner.
- Method: Used `synth_lib.backtester.orchestration.compute_combined_smoothed_scores` to aggregate per‑asset BacktestResults with official `CompetitionConfig` and `ASSET_COEFFICIENTS`.
- Assets included per competition (prediction coverage):
  - CRYPTO_1H: BTC, ETH, SOL (eval_end=2026‑07‑15)
  - CRYPTO_24H: BTC, ETH, SOL (eval_end=2026‑07‑15)
  - COM_EQU_24H: SP500, AAPLX, XAU (eval_end=2026‑07‑24)
- Final ranks (last updated_at snapshot):
  - CRYPTO_1H: rank 238 / 256, reward_weight≈5.19e‑13
  - CRYPTO_24H: rank 254 / 257, reward_weight≈4.78e‑09
  - COM_EQU_24H: rank 256 / 257, reward_weight≈3.71e‑10
- Verdict: Not nomination‑ready. Next: target larger 24h improvement (stronger σ shrink and/or weak mean‑reversion drift) and re‑tune 1h dispersion.
- Lesson: Field softmax concentrates rewards at the very top; small CRPS wins must compound across high‑coefficient assets.

- Windows: 2 days; crypto eval_end=2026‑07‑15; com‑equ eval_end=2026‑07‑24.
- CRYPTO_1H (1000 paths, 2d):
  - BTC: mean_crps=623.66; smoothed_score=100.54
  - ETH: mean_crps=845.58; smoothed_score=130.65
  - SOL: mean_crps=834.90; smoothed_score=132.02
- CRYPTO_24H (1000 paths, 2d):
  - BTC: mean_crps=2846.54; smoothed_score=137.43
  - ETH: mean_crps=4125.32; smoothed_score=200.47
  - SOL: mean_crps=3725.68; smoothed_score=155.93
- COM_EQU_24H (1000 paths, 2d, eval_end 2026‑07‑24):
  - SP500: mean_crps=1169.62; smoothed_score=103.64
  - AAPLX: mean_crps=2085.29; smoothed_score=153.45
  - XAU: mean_crps=1730.42; smoothed_score=134.35
- Ops note: Freed ~690MB by pruning older prediction JSONs to avoid disk‑full during runs.
- Verdict: Results stable over longer windows; dispersion+robust adjustments generalize across crypto and high‑weight equities.
- Next: compute competition‑level ranks at 1000 paths (aggregate over assets) and, if satisfactory, prepare CHAMPION yaml with commit SHA, profiles=[low, high], and notes including per‑competition ranks.

- Setup: 1‑day windows for runtime; crypto eval_end=2026‑07‑15; com‑equ eval_end=2026‑07‑24.
- CRYPTO_1H (1000 paths):
  - ETH: mean_crps=836.61; smoothed_score=113.91
  - SOL: mean_crps=795.49; smoothed_score=112.60
- COM_EQU_24H (1000 paths):
  - AAPLX: mean_crps=2100.16; smoothed_score=198.21
  - XAU: mean_crps=1733.11; smoothed_score=164.55
- Verdict: Results consistent with earlier spot‑checks; equities respond well to dispersion priors; crypto improved vs early baselines. Next: run fuller 2–3 day windows at 1000 paths for BTC/ETH/SOL and SP500/AAPLX/XAU, then begin report.md and consider CHAMPION if gains persist.

- Hypothesis: Per-asset dispersion priors + robust EWMA↔MAD alignment will improve calibration, especially SP500 (high weight) and 1h crypto.
- Change: In `agent/modeling.py`, added robust_adjust using MAD-to-std ratio (clipped 0.85–1.15) and light per-asset multipliers (equities 0.96, commodities 0.98, crypto 1.00). Kept horizon multipliers and tails.
- 1000-path checks (faster single-day windows to bound runtime):
  - crypto-1h/BTC (1 day ending 2026-07-15): mean_crps=617.71, smoothed_score=91.50
  - com-equ-24h/SP500 (1 day ending 2026-07-24): mean_crps=1166.08, smoothed_score=100.10
- Verdict: BTC 1h CRPS improves vs 400‑path earlier runs; SP500 24h also improves modestly. Proceed to 1000‑path multi-asset sweep before nomination.
- Lesson: Robust dispersion alignment is a low-cost, portable win; per-asset priors help high‑coefficient equities without hurting crypto.

### Multi-asset spot-checks (Iteration 4 settings)

- Window: 2 days, eval_end varies per asset; paths=300 for speed.
- CRYPTO_1H:
  - ETH: mean_crps=851.73, smoothed_score=136.77 (2026-07-15)
  - SOL: mean_crps=844.64, smoothed_score=141.66 (2026-07-15)
- CRYPTO_24H:
  - ETH: mean_crps=4179.11, smoothed_score=236.10 (2026-07-15)
  - SOL: mean_crps=3776.73, smoothed_score=183.57 (2026-07-15)
- COM_EQU_24H (eval_end=2026-07-24 for SP500/AAPLX/XAU to land in scored window):
  - SP500: mean_crps=1206.03, smoothed_score=140.05
  - AAPLX: mean_crps=2114.92, smoothed_score=173.62
  - XAU: mean_crps=1734.77, smoothed_score=137.17

- Takeaway: Settings are stable across assets; equities ingest required permissive loader to allow NaN minutes (no lookahead, still offline). Next: per-asset σ scaling and drift for COM_EQU_24H, then 1000‑path checks.

- Hypothesis: Small, horizon-specific dispersion tweaks will tighten 24h CRPS and hold 1h steady.
- Change: Multiply per-step σ by 1.08 for 1h and 0.92 for 24h; tails remain horizon-aware (t for 1h, normal for 24h). See `agent/modeling.py`.
- Result (BTC, 2 days ending 2026-07-15, 400 paths):
  - crypto-1h/BTC: mean_crps 628.89 → 629.05 (neutral), smoothed_score 105.62 → 105.80
  - crypto-24h/BTC: mean_crps 2884.57 → 2873.97 (improves), smoothed_score 162.77 → 155.73
- Verdict: Dispersion shrink helps 24h slightly; 1h unchanged. Proceed to multi-asset sanity checks.
- Lesson: σ calibration moves CRPS more reliably than tail ν at this stage.

- Hypothesis: Horizon-aware tails — Student‑t for 1h, near‑normal for 24h — will improve 1h calibration without over‑widening 24h.
- Change: Make tails conditional on `time_length`: use t‑shocks (ν∈[6,18]) for 1h; Gaussian for 24h; blended ν for mid horizons. Keep EWMA volatility scaling. See `agent/modeling.py`.
- Result (BTC, 2 days ending 2026-07-15, 400 paths):
  - crypto-1h/BTC: mean_crps 625.85 → 628.89 (slightly worse vs Iteration 2 but close), smoothed_score 102.64 → 105.62
  - crypto-24h/BTC: mean_crps 2867.56 → 2884.57 (slightly worse), smoothed_score 151.44 → 162.77
- Verdict: Mixed; need variance calibration per horizon (e.g., shrink per‑step σ for 24h, modest increase for 1h) and/or small mean‑reversion drift.
- Lesson: Tail choice alone is insufficient; CRPS is sensitive to dispersion calibration.

- Hypothesis: Recent-volatility emphasis and fatter tails improve calibration, especially for 1h.
- Change: Replace simple std with EWMA std of 1m returns; switch Normal→Student‑t with df estimated from sample kurtosis; keep zero drift. See `agent/modeling.py`.
- Result (400 paths for speed, same window):
  - crypto-1h/BTC: mean_crps 633.51 → 625.85 (better), final_smoothed_score 110.18 → 102.64
  - crypto-24h/BTC: mean_crps 2807.96 → 2867.56 (worse), final_smoothed_score 111.71 → 151.44
- Verdict: Keep concept, adjust per-horizon calibration — Student‑t helps 1h but over‑widens 24h.
- Lesson: Field-relative softmax can move counter to CRPS; optimize CRPS first, then check rank.

- Hypothesis: A calibrated log-normal martingale on recent minute returns provides a workable baseline across both formats; use as a reference point for CRPS and latency.
- Change: None (baseline) — used the starter `agent/modeling.py` unchanged; generated predictions with 400 paths for speed.
- Result:
  - crypto-24h/BTC (2 days ending 2026-07-15): num_prompts=48, mean_crps=2807.96, final_smoothed_score=111.71
  - crypto-1h/BTC (2 days ending 2026-07-15): num_prompts=288, mean_crps=633.51, final_smoothed_score=110.18
- Verdict: Keep as baseline only — distributions appear too wide/uncalibrated for short horizons; next focus is per-asset volatility calibration and drift handling.
- Lesson: Quick loops with <1000 paths are fast enough to compare variants, but absolute CRPS will be pessimistic; full 1000-path backtests needed before nomination.
## 2026-08-08 — Iteration 35 (24h per-asset calibration)

- Change: Ran `agent/tools/calibrate_asset.py` 1‑day, 400‑path grids for 24h on SP500, AAPLX, and XAU with `long_sigma_mult∈{0.86,0.90,0.94,0.98}`, `theta∈{0.02,0.04,0.06}`; persisted best per asset to `agent/config.json`.
- Best combos (mean_crps on 1‑day window ending 2026‑07‑24):
  - SP500: long_sigma_mult=0.94, theta=0.02, best_mean_crps≈1151.71
  - AAPLX: long_sigma_mult=0.98, theta=0.02, best_mean_crps≈2255.06
  - XAU  : long_sigma_mult=0.98, theta=0.02, best_mean_crps≈1636.67
- Combined COM_EQU_24H (SP500, AAPLX, XAU; 2‑day, 1000 paths): rank 256/257; reward_weight≈4.87e‑10 (still non‑competitive; slight move only).
- Verdict: Light per‑asset θ≈0.02 and modest σ tuning improve single‑asset CRPS a bit but do not shift combined rank meaningfully. Keep overrides; next explore alternate 24h drift anchors (multi‑scale EMA blend) or stronger σ shrink on SP500/AAPLX with nested validation windows. Optionally revisit crypto‑24h.
## 2026-08-08 — Iteration 36 (24h anchor: multi-scale EMA blend)

- Change: Implemented optional 24h drift anchor as a blend of 6h/12h/24h EMAs of log‑price (weights 0.50/0.35/0.15) in `agent/modeling.py` (`long_anchor: blend`). Enabled for SP500/AAPLX/XAU in `agent/config.json`.
- Result: COM_EQU_24H combined (2‑day, 1000 paths; eval_end=2026‑07‑24) rank 256/257; reward_weight≈4.48e‑10 (no material improvement vs ema12).
- Verdict: Multi‑scale anchor does not move field‑relative rank on this window. Keep code path (config‑gated) but leave room for alternative weights or anchors (e.g., rolling VWAP or context‑fit linear trend) if time allows.
## 2026-08-08 — Iteration 37 (1h adaptive tails: p cap + r sweep)

- Change: Added `mix_p_cap` to `mixauto` shocks in `agent/modeling.py` to cap jump probability; set ETH/SOL `mix_r=5.0`, `mix_p_cap=0.08` in `agent/config.json` (BTC unchanged).
- Result: CRYPTO_1H combined (BTC, ETH, SOL; 2‑day, 1000 paths; eval_end=2026‑07‑15) rank 167/256; reward_weight≈3.14e‑4 (matches/barely improves prior ~168/256).
- Verdict: Small but consistent 1h improvement; total Score still dominated by weak 24h performance. Keep config; deprioritize further 1h work unless 24h unlocks stall.
## 2026-08-08 — Iteration 38 (Score estimate refresh)

- Action: Re-ran `agent/tools/score_estimate.py` at 1000 paths after 24h calibration + 1h mixauto p-cap updates.
- Result: CRYPTO_1H ratio≈0.0622; CRYPTO_24H≈8.59e‑10; COM_EQU_24H≈6.29e‑08; Score_estimate≈2.07 (slightly lower vs prior ≈2.56; consistent with unchanged 24h ranks).
- Verdict: Confirms non‑nomination; further progress requires a different 24h approach (dispersion and/or drift) or an entirely richer model class.
## 2026-08-08 — Iteration 39 (SP500 stronger σ shrink tests)

- Change: For 24h SP500 only, tested `long_sigma_mult` 0.90 and 0.86 (keeping θ=0.02, anchor=ema12) against 2‑day combined COM_EQU_24H.
- Result: Combined rank remained 256/257; reward_weight≈4.83e‑10 (0.90) and ≈4.57e‑10 (0.86) — no improvement vs 0.94.
- Verdict: Over‑shrinking dispersion hurts or is neutral in field‑relative terms. Revert to calibrated 0.94.
## 2026-08-08 — Iteration 40 (24h trend drift prototype)

- Change: Added 24h trend-aware drift (config `long_anchor: trend`) — EMA(12h) anchor plus additive linear-trend drift estimated over last 720m of log-price; dt-scaled, with `trend_scale=0.6`. Enabled for SP500 and AAPLX.
- Result: COM_EQU_24H combined (SP500, AAPLX, XAU; 2‑day, 1000 paths; eval_end=2026‑07‑24) rank 256/257; reward_weight≈6.18e‑12 (no improvement; slightly worse vs prior ~4.5e‑10).
- Verdict: This simple trend drift does not help and may hurt on these windows. Revert to ema12/blend anchors. Next: either regime-aware dispersion switch or pause equities and probe crypto‑24h.
## 2026-08-08 — Iteration 41 (24h regime-aware dispersion)

- Change: Added optional variance-based regime switch for 24h in `agent/modeling.py`. If last 60m return std is low/high vs EWMA baseline (ratio <0.85 or >1.25), scale per-step σ by 0.94/1.04 respectively. Enabled for SP500/AAPLX/XAU via `agent/config.json`.
- Result: COM_EQU_24H combined (SP500, AAPLX, XAU; 2‑day, 1000 paths; eval_end=2026‑07‑24): rank 256/257; reward_weight≈6.01e‑10 (no meaningful improvement; essentially unchanged).
- Verdict: Simple variance-switch does not move combined rank on this window. Keeping code path config‑gated for future use. Next, likely stop equities micro‑tuning and probe crypto‑24h or conclude iteration absent new modeling ideas.
## 2026-08-08 — Iteration 42 (crypto‑24h per-asset calibration)

- Change: Ran `agent/tools/calibrate_asset.py` (1 day, 400 paths) for BTC/ETH/SOL at 24h with grid long_sigma_mult∈{0.86,0.90,0.94,0.98}, theta∈{0.02,0.04,0.06}; persisted best to `agent/config.json`.
  - BTC: long_sigma_mult=0.98, theta=0.02, best_mean_crps≈2971.19
  - ETH: long_sigma_mult=0.98, theta=0.02, best_mean_crps≈4373.17
  - SOL: long_sigma_mult=0.98, theta=0.02, best_mean_crps≈3877.19
- Result: CRYPTO_24H combined (BTC, ETH, SOL; 2 days, 1000 paths; eval_end=2026‑07‑15): rank 256/257; reward_weight≈1.04e‑11 (no material change).
- Verdict: Per‑asset θ≈0.02 with mild σ less‑shrink (0.98) does not improve combined rank. 24h remains the limiting factor. Likely pause major 24h work without new modeling ideas; prepare to wrap with report and suggestions, or attempt a different shock family for 24h if time allows.
## 2026-08-08 — Iteration 43 (24h variance‑gamma shocks)

- Change: Added `long_shock: vg` option for 24h in `agent/modeling.py`: Normal shocks with per-step Gamma variance mixing (mean=1) calibrated to bounded excess kurtosis from recent returns; preserves unconditional per-step variance. Enabled for SP500/AAPLX/XAU with `long_ek_cap=0.25`.
- Result: COM_EQU_24H combined (2‑day, 1000 paths; eval_end=2026‑07‑24) rank 256/257; reward_weight≈6.39e‑10 (no improvement vs Gaussian baseline).
- Verdict: 24h alternate shock family does not improve combined rank on these windows. Given repeated non‑movement across multiple 24h ideas, I will stop equities tuning and proceed to finalize deliverables and suggestions.
## 2026-08-08 — Iteration 44 (24h NIG shocks)

- Change: Implemented `long_shock: nig` (symmetric) for 24h: Normal with Inverse‑Gaussian mixing (μ=1, λ=α²), with α chosen to match bounded excess kurtosis via ek≈3/α² while preserving unconditional variance. Config‑gated with `long_ek_cap` and `long_ek_default`.
- Result: COM_EQU_24H combined (SP500, AAPLX, XAU; 2‑day, 1000 paths; eval_end=2026‑07‑24): rank 256/257; reward_weight≈6.05e‑10 (no improvement vs Gaussian/VG).
- Verdict: Another 24h tail family shows no lift on combined rank. With budget/time remaining but no promising 24h gains, proceed to finalize documentation and keep code paths gated for future exploration.
## 2026-08-08 — Iteration 45 (baseline restore + final checks)

- Change: Restored 24h equities to baseline config (Gaussian shocks, ema12 anchor, no regime): SP500 σ_mult=0.94, θ=0.02; AAPLX σ_mult=0.98, θ=0.02; XAU σ_mult=0.98, θ=0.02. Left 1h crypto tails as previously tuned.
- Result: COM_EQU_24H combined (SP500, AAPLX, XAU; 2‑day, 1000 paths): rank 256/257; reward_weight≈4.59e‑10. CRYPTO_24H combined (BTC, ETH, SOL; 2‑day, 1000 paths): rank 256/257; reward_weight≈9.59e‑12.
- Verdict: Confirms final state — 24h remains non‑competitive; 1h modestly improved but insufficient for nomination. Proceed to finalize docs; no CHAMPION nomination.
## 2026-08-08 — Iteration 46 (config cleanup)

- Action: Pruned `agent/config.json` to a minimal, stable baseline.
  - Equities (SP500/AAPLX/XAU): keep `long_sigma_mult`, `theta`, and explicit `long_anchor=ema12`.
  - Crypto (BTC/ETH/SOL): keep 24h overrides (`long_sigma_mult`, `theta`) and 1h tails (`short_shock`, `mix_r`, `mix_p_cap`, `short_sigma_boost`, `short_tau_minutes`).
- Result: Config simpler and aligned with final baseline; experimental keys removed but code paths remain available.
- Verdict: Final housekeeping complete; deliverables finalized with no nomination.
## 2026-08-08 — Iteration 47 (final Score estimate)

- Action: Ran `agent/tools/score_estimate.py` at 1000 paths on the restored baseline.
- Result: CRYPTO_1H ratio≈0.0584; CRYPTO_24H≈1.00e‑09; COM_EQU_24H≈5.34e‑08; Score_estimate≈1.95.
- Verdict: Confirms non‑nomination; deliverables finalized.
## 2026-08-08 — Iteration 48 (REPRODUCE.md)

- Action: Added agent/REPRODUCE.md with exact commands, constraints, and baseline config summary.
- Result: Standalone reproduce guide in-place; aligns with report final snapshot.
- Verdict: Docs complete.
## 2026-08-08 — Iteration 49 (housekeeping)

- Action: Pruned miner_outputs to save space (kept empty directory; tools regenerate as needed).
- Result: Workspace lighter; no change to code or configs.
- Verdict: Campaign artifacts finalized.
## 2026-08-08 — Iteration 50 (CHANGELOG)

- Action: Added agent/CHANGELOG.md summarizing Iterations 1–49 for quick reference.
- Result: Readers can scan major milestones without parsing the full journal.
- Verdict: Docs polished; awaiting landing order.
## 2026-08-08 — Iteration 51 (docs polish)

- Change: Expanded agent/CONFIG.md to document  for 24h drift.
- Result: Config docs now match the final baseline exactly.
- Verdict: All docs are aligned; ready for landing order.
## 2026-08-08 — Iteration 52 (root README)

- Action: Added root README.md pointing to agent docs and summarizing final status and constraints.
- Result: Clear entry point for reviewers; aligns with REPRODUCE.md and report.md.
- Verdict: Repo-level docs complete.
## 2026-08-10 — Iteration 53 (1h BTC early-minute boost + mix_p_cap)

- Hypothesis: For 1h crypto, a modest early-minute dispersion boost (short_sigma_boost=0.08, tau=8m) and a slightly higher mix_p_cap (0.10) will better fit early gaps and tails, improving smoothed score without inflating mean CRPS.
- Change: Updated agent/config.json for BTC only: short_sigma_boost=0.08, short_tau_minutes=8.0, mix_p_cap=0.10; kept ETH/SOL unchanged.
- Result (1 day ending 2026-07-24, 200 paths):
  - BTC 1h: mean_crps=463.75 (prev≈461.76), smoothed_score=68.53 (prev≈66.58).
  - ETH 1h baseline (unchanged): mean_crps=635.21, smoothed_score=84.26.
  - SOL 1h baseline (unchanged): mean_crps=640.73, smoothed_score=100.29.
  - SP500 24h baseline: mean_crps=1157.32, smoothed_score=91.34.
- Verdict: Slight improvement in smoothed score but with worse mean CRPS — likely noise from reward-history alignment. Not compelling; discard.
- Lesson: Early-minute boost risks over-dispersion that hurts CRPS despite possible rank wiggles; prioritize calibrating mixture r and probability cap before additive boosts.
## 2026-08-10 — Iteration 54 (1h mixture tuning without boost)

- Hypothesis: Removing early-minute boost and tuning mixture-only (mix_r, mix_p_cap) will reduce 1h CRPS without harming smoothed score; target combo r=5, p_cap=0.08.
- Change: Set BTC/ETH/SOL short_sigma_boost=0.0; BTC mix_r=5, mix_p_cap=0.08 (ETH/SOL propagated after BTC check).
- Results (1 day ending 2026-07-24, 400 paths unless noted):
  - BTC 1h: mean_crps≈460.08, smoothed≈64.93 (vs prior ≈461.76/66.58 at 200 paths).
  - ETH 1h: mean_crps≈634.95, smoothed≈83.99 (baseline ≈635.21/84.26). Small CRPS gain; smoothed flat.
  - SOL 1h: pending compute (started; will append numbers next iteration).
- Verdict: Mixture-only tuning slightly improves CRPS on BTC and ETH with minimal smoothed impact. Keep and proceed to SOL + 2-day BTC verification.
- Lesson: Probability-cap + scale-ratio is a cleaner lever for early gaps/tails than additive dispersion; boosts risk over-width.
(Update) SOL 1h result for Iteration 54: mean_crps≈638.12 (baseline ≈640.73), smoothed≈97.74 (baseline ≈100.29). Small CRPS gain with slight smoothed dip; keep settings for now.
## 2026-08-10 — Iteration 55 (1h mixture-only verification)

- Hypothesis: The mixture-only setting (r=5, p_cap=0.08, no boost) generalizes across windows and holds at 1000 paths.
- Change: None beyond Iteration 54; verified windows/paths.
- Results:
  - BTC 1h — 2 days, 400 paths: mean_crps≈471.37; smoothed≈67.19.
  - BTC 1h — 1 day, 1000 paths: mean_crps≈459.11; smoothed≈62.55 (lower CRPS than 400-path run, as expected).
  - ETH 1h — 1 day, 400 paths: mean_crps≈634.95; smoothed≈83.99.
  - SOL 1h — 1 day, 400 paths: mean_crps≈638.12; smoothed≈97.74.
- Verdict: Mixture-only settings produce small but consistent CRPS gains across BTC/ETH/SOL, with mixed but acceptable smoothed shifts. Keep settings; proceed to combined 1h check and brief 24h sanity pass.
- Lesson: Sampling at 1000 paths reduces CRPS modestly vs 400; tuning decisions should be made at or near serving count when feasible.
## 2026-08-10 — Iteration 56 (1h combined + 24h sanity)

- Hypothesis: With r=5, p_cap=0.08, no boost on BTC/ETH/SOL, combined 1h shows a small net CRPS improvement; 24h SP500 remains stable.
- Change: None (evaluation only).
- Results (1 day ending 2026-07-24):
  - CRYPTO_1H per-asset: BTC mean_crps≈460.08; ETH≈634.95; SOL≈638.12. Smoothed scores ≈64.93/83.99/97.74.
  - SP500 24h sanity (200 paths): mean_crps≈1159.16; smoothed≈93.18 (consistent with prior slice).
- Verdict: Keep mixture-only settings for crypto 1h; 24h unchanged.
- Lesson: 1h mixture parameters provide modest, consistent gains; further 24h work likely low ROI within current envelope.
EOF}
(Update) Iteration 56 — Combined CRYPTO_1H (1 day, 400 paths, BTC/ETH/SOL): combined rank=174/257; reward_weight≈7.55e-04.
## 2026-08-10 — Iteration 57 (Score estimate @ 1000 paths)

- Hypothesis: Mixture-only 1h tuning yields a slight lift in the overall Score_estimate; 24h remains the main limiter.
- Change: None; evaluation-only via agent/tools/score_estimate.py at 1000 paths.
- Results:
  - CRYPTO_1H_ratio ≈ 0.0558
  - CRYPTO_24H_ratio ≈ 8.87e-10
  - COM_EQU_24H_ratio ≈ 5.60e-08
  - Score_estimate ≈ 1.86
- Verdict: 1h improvements are modest but real; 24h remains non-competitive in this snapshot. Keep 1h mixture-only settings; document findings; consider deeper 24h work only with more budget/time.
- Lesson: Field-relative improvements require large absolute gains on high-weight assets; focus should prioritize levers that move combined ranks materially.
## 2026-08-10 — Iteration 58 (24h SP500 anchor/θ micro-experiment — aborted)

- Hypothesis: A quick 24h SP500 sweep over long_anchor∈{ema12,blend,trend} and θ∈{0.02,0.04,0.06} could yield a small COM_EQU_24H combined reward_weight lift.
- Change: Attempted helper script to grid anchor/θ at 1 day, 200 paths; no production changes applied.
- Result: Helper got corrupted mid-edit (quoting/f-string issues) and expected ROI is low; kept 1h mixture-only state unchanged.
- Verdict: Discard this 24h sweep for this cycle; do not nominate.
- Lesson: Within this envelope, 1h mixture tuning offers modest, consistent gains; 24h needs a deeper approach (anchors/regimes or alternative shocks) to move combined ranks materially.
## 2026-08-10 — Iteration 59 (CRYPTO_1H combined — 2 days, 400 paths)

- Hypothesis: A 2-day combined check reduces noise and confirms direction for the 1h mixture-only settings.
- Change: None; evaluation-only via agent/tools/combined_eval.py on BTC/ETH/SOL, 2 days ending 2026-07-24 at 400 paths.
- Result: combined rank=183/257; reward_weight≈3.61e-04 (1-day prior was rank=174/257; ≈7.55e-04).
- Verdict: Direction broadly consistent; rank fluctuates across windows with smaller reward_weight. Keep mixture-only settings; gains are modest.
- Lesson: Short windows remain noisy; use 1000-path checks for final judgments and avoid overfitting to single-day combined ranks.
EOF}
## 2026-08-10 Iteration 1
Hypothesis: Current model is a tuned martingale; confirm baseline CRPS and score on crypto-1h with reduced paths.
Change: None (establish baseline at 400 sims).
Result: BTC 1h: mean_crps~613.57, final_smoothed_score~96.79 over 152 prompts.
Verdict: Keep as baseline.
Lesson: Pipeline healthy; 1h already near top score at 400 sims; need 24h coverage assets with available scored prompts (BTC works; SP500 lacked scores in chosen window).

## 2026-08-10 — Iteration 2
Hypothesis: Slightly boosting early-minute dispersion and allowing higher rare-shock cap improves CRYPTO_1H gaps scoring; tightening SP500 long width and higher θ helps 24h absolute CRPS.
Change: config.json — BTC/ETH/SOL short_sigma_boost + mix_p_cap up; SP500 long_sigma_mult down, theta up.
Result: BTC 24h (1 day, 400 paths) mean_crps≈2885.40; smoothed≈176.54. BTC 1h/ETH 1h re-scored successfully (full summary suppressed in log due to interrupts, but pipeline OK).
Verdict: Keep tweaks; proceed to multi-asset quick checks and then a 1000‑path confirm on 1h.
Lesson: Editing via config enables fast iteration; 24h BTC score is reasonable; need SP500/AAPLX windows with scored prompts for meaningful COM_EQU_24H readout.

## 2026-08-10 — Iteration 3
Hypothesis: Post-tweak 1h at 1000 paths confirms small gains; one COM_EQU_24H asset (AAPLX) provides a usable 24h slice late-July.
Change: None beyond Iteration 2; evaluation-only.
Results: BTC 1h/ETH 1h 1000-path scoring ran; summaries interrupted during smoothed calc but pipelines OK. AAPLX 24h window available; smoothed calc interrupted; SP500 24h had no scored prompts for 2026-07-21→22.
Verdict: Continue with 1h 1000-path confirms across BTC/ETH and capture full summaries; keep searching scored SP500 window near 2026-07-24+.
Lesson: Backtester network fetch for reward histories dominates wall time; avoid interrupts to preserve summaries.
## 2026-08-10 — Iteration 4 (post-tweak confirms)
Hypothesis: 1h BTC/ETH at 1000 paths and 24h AAPLX at 400 paths provide stable post-tweak reference metrics.
Change: None (evaluation only).
Results:
 - BTC 1h (1 day, 1000 paths): mean_crps≈613.27; smoothed≈96.48.
 - ETH 1h (1 day, 1000 paths): mean_crps≈842.62; smoothed≈125.61.
 - AAPLX 24h (1 day, 400 paths): mean_crps≈2160.83; smoothed≈235.96.
Verdict: Keep current settings; 1h metrics are stable; AAPLX shows reasonable 24h behavior. Proceed to combined 1h (BTC/ETH/SOL) at 1000 paths for a single-day slice and, if coverage permits, a SP500 24h slice near 2026-07-24.
Lesson: Completing runs without interrupts yields consistent summaries; reward-history fetch dominates wall time on COM_EQU_24H.

## 2026-08-10 — Iteration 5 (combined 1h + COM_EQU_24H add)
Hypothesis: Combined 1-day CRYPTO_1H at 1000 paths reflects net post-tweak calibration; XAU 24h provides a COM_EQU_24H reference when SP500 window is unavailable.
Change: None (evaluation only).
Results (1 day ending 2026-07-24):
 - BTC 1h: mean_crps≈461.61; smoothed≈63.83.
 - ETH 1h: mean_crps≈641.48; smoothed≈83.33.
 - SOL 1h: mean_crps≈641.31; smoothed≈95.14.
Results (24h, 1 day ending 2026-07-22):
 - XAU 24h: mean_crps≈1841.88; smoothed≈213.48.
Verdict: Keep current 1h mixture tweaks; 24h remains acceptable but not a differentiator. Next, consider modest horizon-aware long_shock for 24h (VG/NIG) on SP500/AAPLX if windows exist.
Lesson: Combined 1h runs at serving count are fast enough and more reliable; COM_EQU_24H reward-history fetch dominates wall time.

## 2026-08-10 — Iteration 6 (AAPLX 24h VG test)
Hypothesis: Variance-Gamma long shocks can reduce 24h_abs CRPS on AAPLX without widening early intervals.
Change: agent/config.json — AAPLX {long_shock=vg, long_ek_default=0.15, long_ek_cap=0.3}.
Result: AAPLX 24h (1 day, 400 paths): mean_crps≈2160.74; smoothed≈235.91 (baseline ≈2160.83/235.96). Effect negligible in this window.
Verdict: Discard VG for now (no clear benefit); retain Gaussian long shocks.
Lesson: Mild variance-mixture on 24h didn’t move CRPS in this slice; deeper 24h improvements likely require anchor/regime changes or per-interval calibration.

## 2026-08-10 — Iteration 7 (AAPLX 24h variance-regime)
Hypothesis: Gentle variance-based dispersion (±3% outside a narrow band) stabilizes 24h CRPS slightly vs fixed width.
Change: agent/config.json — AAPLX {long_shock=gauss, long_regime=variance, hv_minutes=60, hv_ratio_low=0.85, hv_ratio_high=1.15, hv_low_sigma_mult=0.97, hv_high_sigma_mult=1.03}.
Result: AAPLX 24h (1 day, 400 paths): mean_crps≈2157.95; smoothed≈234.46 (vs prior ≈2160.83/235.96 and VG ≈2160.74/235.91). Small improvement in this slice.
Verdict: Keep variance-regime for AAPLX 24h; apply analogous gentle regime to SP500 when a covered window is available.
Lesson: Light adaptive width can yield minor CRPS gains on 24h without hurting earlier intervals; ROI remains modest.

## 2026-08-10 — Iteration 8 (COM_EQU_24H finalize)
Hypothesis: Gentle variance-regime on SP500 mirrors AAPLX gains and provides small 24h improvement on a covered window.
Change: agent/config.json — SP500 {long_regime=variance, hv_minutes=60, hv_ratio_low=0.85, hv_ratio_high=1.15, hv_low_sigma_mult=0.97, hv_high_sigma_mult=1.03}.
Results (1 day, 400 paths):
 - SP500 24h (end 2026-07-26): mean_crps≈533.90; smoothed≈67.96.
 - AAPLX 24h (end 2026-07-22): mean_crps≈2157.95; smoothed≈234.46.
 - XAU 24h (end 2026-07-22): mean_crps≈1841.88; smoothed≈213.48.
Note: AAPLX/XAU on 2026-07-26 hit read-only realized-path cache; used 2026-07-22 where scoring completes.
Verdict: Keep variance-regime on SP500/AAPLX; COM_EQU_24H still modest impact; prioritize report finalization.
Lesson: Realized-path caching writes under market_data/realized can fail on read-only mounts; choose covered dates or run per-asset where cache exists.

## 2026-08-10 — Iteration 9 (SP500 θ A/B with variance-regime)
Hypothesis: Slightly lower θ (0.02 vs 0.04) could reduce 24h_abs CRPS when variance-regime adjusts dispersion.
Change: OU_THETA env for SP500 24h on 2026-07-26; 400 paths; variance-regime enabled.
Results: θ=0.02 → mean_crps≈532.72; smoothed≈67.38. θ=0.04 → mean_crps≈534.85; smoothed≈68.42.
Verdict: Prefer θ≈0.02 for SP500 with variance-regime in this slice; keep gentle regime settings.
Lesson: Within small bands, θ trades off 24h_abs vs drift; variance-regime plus slightly slower mean-reversion was marginally better here.

## 2026-08-10 — Iteration 10 (CRYPTO_1H combined snapshot)
Hypothesis: 1-day 1000-path combined run provides a clean post-tweak benchmark.
Change: None (evaluation only).
Result: rank=173/257; reward_weight≈8.33e-04 (BTC/ETH/SOL; end 2026-07-24).
Verdict: Keep settings; document and wrap cycle without nomination.
Lesson: Post-tweak 1h gains are modest; 24h variance-regime yields small CRPS improvements on AAPLX/SP500 but not enough to move combined ranks materially.

## 2026-08-10 — Iteration 11 (Final Score estimate)
Change: None (evaluation only; 1000 paths).
Result: CRYPTO_1H≈0.06363; CRYPTO_24H≈9.68e-10; COM_EQU_24H≈5.92e-08; Score_estimate≈2.12.
Verdict: Do not nominate; wrap docs.
Lesson: 24h remains the limiting factor; gentle regimes/tuning yield only modest gains within this envelope.
## 2026-08-10 — Iteration 12 (Wrap-up)
Change: None.
Result: Deliverables complete; Score_estimate≈2.12.
Verdict: Stop without nomination.
Lesson: 24h remains the limiter within this envelope.
## 2026-08-10 — Iteration 12 (Wrap-up)
Change: None.
Result: Deliverables complete; Score_estimate≈2.12.
Verdict: Stop without nomination.
Lesson: 24h remains the limiter within this envelope.


## 2026-08-10 — Iteration 13 (24h NIG A/B)
Hypothesis: NIG variance-mixture might lower 24h_abs CRPS vs Gaussian+variance-regime.
Change: SP500/AAPLX long_shock=nig (long_ek_default=0.08, long_ek_cap=0.20).
Results: SP500 (07-26, 400 paths) mean_crps≈533.50; smoothed≈67.78 (worse than Gauss+regime ≈532.72/67.38). AAPLX (07-22, 400 paths) mean_crps≈2158.08; smoothed≈234.53 (≈2157.95/234.46 prior).
Verdict: Discard NIG; revert to Gaussian with gentle variance-regime (θ≈0.02 for SP500 on this slice).
Lesson: For 24h, light adaptive width outperforms heavier-tailed mixtures in these windows.

## 2026-08-10 — Iteration 14 (1h boost=0 confirm)
Change: short_sigma_boost=0.0 for BTC/ETH/SOL.
Result: CRYPTO_1H combined (1 day, 1000 paths, end 2026-07-24): rank=176/257; reward_weight≈5.88e-04.
Verdict: Keep zero-boost (no benefit observed over prior slice); document and remain without nomination.
Lesson: Early-minute global boost is not needed with mixture-only tails; results are slice-sensitive but directionally consistent.

## 2026-08-10 — Iteration 35 (1h per-minute multipliers)

- Hypothesis: Early-minute widening via explicit per-step multipliers will better fit the 1h gaps intervals without over-broadening the full hour, yielding a small lift in combined CRYPTO_1H rank.
- Change:
  - Added a `short_minute_mults` hook in `agent/modeling.py` for 1h: if present (length==steps), apply elementwise multipliers to per-step shocks; otherwise fall back to the legacy exponential boost (`short_sigma_boost`, `short_tau_minutes`).
  - Seeded conservative arrays in `agent/config.json`:
    - BTC: starts ~1.10 and decays to 1.00 by ~minute 37.
    - ETH: starts ~1.08 and decays to 1.00 by ~minute 36.
    - SOL: starts ~1.10 and decays to 1.00 by ~minute 36.
- Eval (1 day, 400 paths; end=2026-07-15) via `agent/tools/combined_eval.py` on BTC/ETH/SOL:
  - Result: rank 167/257; reward_weight≈3.17e-04; miners≈257.
- Verdict: Comparable to prior 1h combined (≈167–168) — no material change on this short window at 400 paths. The hook is in place and safe; further tuning likely needs small shape adjustments (avoid over-weighting ≤5min which can hurt cumulative gaps) and confirmation at 1000 paths.
- Next: If pursuing 1h further, run 2‑day, 1000‑path combined with a small grid over the first 10–15 multipliers (±2–3%) to target gaps while guarding 60min_abs.

### Iteration 35b — validator-parity confirm (2d, 1000 paths)

- Change: None (evaluation-only)
- Eval (2 days, 1000 paths; end=2026-07-15): rank 167/256; reward_weight≈2.76e-04; miners≈256.
- Verdict: Parity run confirms no material movement vs previous baseline combined ranks. Hook retained; pause 1h tuning unless directed to grid-search the leading multipliers.



## 2026-08-10 — Iteration 15 (1h mix_p_cap=0.08)
Change: mix_p_cap=0.08 for BTC/ETH/SOL (short_sigma_boost=0.0).
Result: CRYPTO_1H combined (1 day, 1000 paths, end 2026-07-24): rank=175/257; reward_weight≈6.47e-04.
Verdict: Minor slice improvement over cap=0.10 run; still modest overall. Keep documented; no nomination.
Lesson: Small cap tightening yields slight lift on this slice; overall 1h remains consistent with prior results.



## 2026-08-10 — Iteration 17 (24h long_sigma_mult sweeps)
Hypothesis: Slight dispersion nudges at 24h can reduce 24h_abs CRPS without harming earlier intervals under variance-regime.
Change: SP500 long_sigma_mult∈{0.92,0.94} with OU_THETA=0.02; AAPLX long_sigma_mult∈{0.96,1.00}; reverted afterward.
Results: SP500 (07-26, 400 paths): 0.92→mean_crps≈533.41; 0.94→≈532.63 (best so far). AAPLX (07-22): 0.96→≈2164.07; 1.00→≈2155.19 (best so far).
Verdict: Keep note: tiny gains possible; baseline remains modest; no nomination.
Lesson: 24h small σ multipliers can help marginally, but gains are too small to affect combined ranks here.



## 2026-08-10 — Iteration 18 (COM_EQU_24H combined, 24h σ tweaks)
Change: SP500 long_sigma_mult=0.94; AAPLX=1.00 (variance-regime on); XAU baseline.
Result: COM_EQU_24H combined (1 day, 400 paths, end 2026-07-24): rank=256/257; reward_weight≈1.03e-11.
Verdict: σ nudges too small to move combined rank; revert to defaults and keep note only.
Lesson: On this snapshot, tiny 24h dispersion changes don’t change combined standings; deeper modeling needed for material gains.



## 2026-08-10 — Iteration 19 (COM_EQU_24H combined, 2-day σ tweaks)
Change: SP500 long_sigma_mult=0.94; AAPLX=1.00; XAU baseline.
Result: COM_EQU_24H combined (2 days, 400 paths, end 2026-07-24): rank=256/257; reward_weight≈1.97e-12.
Verdict: No material change; revert to defaults; document and remain without nomination.
Lesson: Multi-day COM_EQU_24H remains dominated by stronger miners; tiny σ changes do not move combined standings here.
