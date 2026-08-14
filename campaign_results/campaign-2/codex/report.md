# Campaign Report — SN50 Calibration Experiments

## Context
- Goal: Improve CRPS across CRYPTO_1H, CRYPTO_24H, and COM_EQU_24H under Synth SN50 rules.
- Data cutoff: 2026‑07‑27 (no lookahead). All evaluation via synth‑lib backtester.
- Paths: 1000 per prompt for final checks (validator parity); fewer for iteration speed.

## Approach
- Baseline: log‑normal martingale on minute returns.
- Additions and calibrations:
  - EWMA volatility on 1‑minute log returns (recent emphasis).
  - Horizon‑aware tails: Student‑t for 1h; Gaussian for 24h; blended for mid horizons.
  - Dispersion tuning: σ multipliers by horizon (1h ×1.08; 24h ×0.92 → later tightened to ×0.90).
  - Robust alignment: scale EWMA std to MAD (clip to [0.85, 1.15]).
  - Per‑asset priors: equities ×0.96; commodities ×0.98; crypto ×1.00.
  - 24h drift: weak OU mean‑reversion toward a 12h EMA anchor on log‑price (θ default 0.03; env‑tunable `OU_THETA`).
  - Long‑horizon per‑asset σ (24h only): SP500×0.92, AAPLX×0.94, XAU×0.95, WTIOIL×0.95.

## Experiments (highlights)
- Iter 1–3: EWMA + fat tails improved 1h but widened 24h; moved to horizon‑aware tails.
- Iter 4–5: σ tuning + robust EWMA↔MAD improved stability; first 1000‑path BTC/SP500 gains.
- Iter 6–7: 1000‑path checks across BTC/ETH/SOL and SP500/AAPLX/XAU; 2‑day sweeps for stability.
- Iter 8: Official cross‑asset combiner for competition‑level ranks (with asset coefficients).
- Iter 9–10: Add 24h OU drift; θ sweep (BTC favored 0.02; SP500 ~0.04; kept global 0.03 for simplicity).
- Iter 11: Per‑asset 24h σ multipliers — modest wins on SP500/XAU; AAPLX regressed.

## Results
- CRYPTO_1H (1 day ending 2026‑07‑15; 1000 paths):
  - BTC mean_crps=617.71; ETH=836.61; SOL=795.49.
- CRYPTO_24H (2 days ending 2026‑07‑15; 1000 paths):
  - BTC mean_crps=2846.54; ETH=4125.32; SOL=3725.68.
- COM_EQU_24H (2 days ending 2026‑07‑24; 1000 paths):
  - SP500 mean_crps=1169.62; AAPLX=2085.29; XAU=1730.42.
- Competition‑level backtested ranks (official combiner, final snapshot):
  - CRYPTO_1H: rank 168/256; reward_weight ≈ 2.46e‑4 (latest).
  - CRYPTO_24H: rank 256/257; reward_weight ≈ 8.84e‑12.
  - COM_EQU_24H: rank 256/257; reward_weight ≈ 1.91e‑10.

## Limitations
- Closed‑form simulator; no learned drift/regime logic beyond OU and σ multipliers.
- Heuristic priors (σ, θ) — modest improvements but insufficient vs field top.
- Short validation windows (1–2 days) for iteration speed; multi‑week sweeps may refine parameters but risk overfitting without nested validation.

## Conclusion
- Not nomination‑ready. Despite small CRPS gains (notably SP500/XAU 24h) and a modest CRYPTO_1H improvement to rank 168/256, competition‑level ranks remain far from the reward‑earning top of field.
- We are stopping iteration here to preserve budget and provide a clear, reproducible report.

## Next With More Budget
- Asset‑specific θ and σ learned via rolling cross‑validated calibration.
- Regime awareness: simple volatility‑state switch to adapt dispersion/tails in real time.
- Improved drift anchors (multi‑scale EMA blend) and gap‑targeted calibration for 1h.
- Alternative shock families for 1h (variance‑gamma / NIG) with Gaussian retained for 24h.

## Reproduce
- Env setup: from the workspace root run `uv sync` once; then always use `uv run`.
- Quick per‑asset backtest (example):
  - 1h BTC, 2 days ending 2026‑07‑15 at 1000 paths:
    `uv run agent/predict.py --asset BTC --days 2 --eval-end 2026-07-15 --time-length 3600 --time-increment 60 --num-simulations 1000`
  - 24h SP500, 2 days ending 2026‑07‑24 at 1000 paths:
    `uv run agent/predict.py --asset SP500 --days 2 --eval-end 2026-07-24 --num-simulations 1000`
- Combined ranks with validator coefficients (helper):
  - CRYPTO_1H (BTC ETH SOL):
    `uv run python agent/tools/combined_eval.py --competition CRYPTO_1H --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000`
  - COM_EQU_24H (SP500 AAPLX XAU):
    `uv run python agent/tools/combined_eval.py --competition COM_EQU_24H --assets SP500 AAPLX XAU --days 2 --eval-end 2026-07-24 --paths 1000`

### Per‑Asset Overrides
- The model reads optional overrides from `agent/config.json` per asset symbol (uppercase):
  - `long_sigma_mult` (float): additional 24h dispersion multiplier after global scaling.
  - `theta` (float): 24h OU mean‑reversion strength (per‑hour).
- Example file:
  `{ "SP500": { "long_sigma_mult": 0.96, "theta": 0.05 }, "AAPLX": { "long_sigma_mult": 0.94, "theta": 0.05 } }`
- Calibrate quickly on a short window with:
  `uv run python agent/tools/calibrate_asset.py --asset AAPLX --eval-end 2026-07-24 --days 1 --paths 300 --grid-sigma 0.90 0.94 --grid-theta 0.03 0.05`


Update (2026-08-08): Added 24h variants (multi-scale/linear-trend anchors, variance regime, variance-gamma shocks). None improved COM_EQU_24H combined rank (≈256/257 at 1000 paths). CRYPTO_1H remains ≈167/256 after adaptive tails. Score estimate ≈2.07. No nomination.

Final check (2026-08-08): Reverted to baseline 24h (Gaussian, ema12, no regime). COM_EQU_24H≈256/257; CRYPTO_24H≈256/257; CRYPTO_1H≈167/256. No nomination.

Final Score estimate (1000 paths): CRYPTO_1H≈0.0584, CRYPTO_24H≈1.00e-09, COM_EQU_24H≈5.34e-08; overall ≈1.95.

## Final Snapshot (2026-08-08 UTC)

Reproduce the final baseline configuration and results.

Commands (run from repo root after `uv sync`):

- COM_EQU_24H (SP500 AAPLX XAU, 2 days, 1000 paths, eval_end=2026-07-24)
  `uv run python agent/tools/combined_eval.py --competition COM_EQU_24H --assets SP500 AAPLX XAU --days 2 --eval-end 2026-07-24 --paths 1000`
- CRYPTO_24H (BTC ETH SOL, 2 days, 1000 paths, eval_end=2026-07-15)
  `uv run python agent/tools/combined_eval.py --competition CRYPTO_24H --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000`
- CRYPTO_1H (BTC ETH SOL, 2 days, 1000 paths, eval_end=2026-07-15)
  `uv run python agent/tools/combined_eval.py --competition CRYPTO_1H --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000`
- Score estimate
  `uv run python agent/tools/score_estimate.py --paths 1000`

Final config used (`agent/config.json`):

```json
{
  "AAPLX": {
    "long_anchor": "ema12",
    "long_sigma_mult": 0.98,
    "theta": 0.02
  },
  "BTC": {
    "long_sigma_mult": 0.98,
    "mix_r": 4.0,
    "short_shock": "mixauto",
    "short_sigma_boost": 0.0,
    "short_tau_minutes": 10.0,
    "theta": 0.02
  },
  "ETH": {
    "long_sigma_mult": 0.98,
    "mix_p_cap": 0.08,
    "mix_r": 5.0,
    "short_shock": "mixauto",
    "short_sigma_boost": 0.0,
    "short_tau_minutes": 10.0,
    "theta": 0.02
  },
  "SOL": {
    "long_sigma_mult": 0.98,
    "mix_p_cap": 0.08,
    "mix_r": 5.0,
    "short_shock": "mixauto",
    "short_sigma_boost": 0.0,
    "short_tau_minutes": 10.0,
    "theta": 0.02
  },
  "SP500": {
    "long_anchor": "ema12",
    "long_sigma_mult": 0.94,
    "theta": 0.02
  },
  "XAU": {
    "long_anchor": "ema12",
    "long_sigma_mult": 0.98,
    "theta": 0.02
  }
}
```


## Addendum (2026-08-10 UTC) — 1h Mixture-Only Tuning

- Change: For CRYPTO_1H only, removed early-minute dispersion boosts and tuned a simple Gaussian mixture for shocks with two knobs: 'mix_r' (sigma ratio) and 'mix_p_cap' (cap on big-sigma probability); kept 24h unchanged.
- Settings adopted: 'mix_r'=5, 'mix_p_cap'=0.08, 'short_sigma_boost'=0.0 for BTC/ETH/SOL.
- Windows/paths: primarily 1-day at 400 paths for iteration; verified BTC on 2-day (400) and 1-day (1000) to gauge sampling bias; all via synth-lib backtester.
- Per-asset outcomes (1 day ending 2026-07-24, 400 paths): BTC mean_crps≈460.08; ETH≈634.95; SOL≈638.12 with smoothed≈64.93/83.99/97.74 respectively (small CRPS gains vs prior slice).
- Combined CRYPTO_1H (BTC/ETH/SOL, 1 day, 400 paths): rank 174/257; reward_weight≈7.55e-04.
- 24h sanity (SP500, 1 day, 200 paths): mean_crps≈1159.16; smoothed≈93.18 (unchanged trend).
- Updated Score_estimate at 1000 paths (official tool, short windows): CRYPTO_1H≈0.0558; CRYPTO_24H≈8.87e-10; COM_EQU_24H≈5.60e-08; overall ≈1.86.

Takeaway: Mixture-only tuning yields modest, consistent 1h CRPS improvements without destabilizing 24h. Field-relative emissions remain dominated by 24h underperformance; further 24h work likely requires more ambitious modeling (anchors/regimes or learned components) beyond this envelope.

## Addendum (2026-08-10 UTC) — Latest Runs and Findings

- CRYPTO_1H (1 day, 1000 paths, end 2026-07-24):
  - BTC mean_crps≈461.61; smoothed≈63.83.
  - ETH mean_crps≈641.48; smoothed≈83.33.
  - SOL mean_crps≈641.31; smoothed≈95.14.
- COM_EQU_24H (24h, 1 day, 400 paths):
  - SP500 (end 2026-07-26), variance-regime on: mean_crps≈533.90; smoothed≈67.96.
  - AAPLX (end 2026-07-22), variance-regime on: mean_crps≈2157.95; smoothed≈234.46.
  - XAU (end 2026-07-22): mean_crps≈1841.88; smoothed≈213.48.
- 24h variance-regime (gentle, hv_minutes=60, band 0.85–1.15, mult 0.97/1.03) yields small CRPS gains on AAPLX and SP500 in covered slices; VG shocks had negligible effect.
- Practical note: combined runs on some 2026-07-26 assets attempted to write realized paths under market_data/realized/ (read-only mount) and failed; per‑asset runs on covered dates (e.g., 2026-07-22) completed successfully.

## Final Score Estimate (2026-08-10 UTC)
- CRYPTO_1H_ratio ≈ 0.06363
- CRYPTO_24H_ratio ≈ 9.68e-10
- COM_EQU_24H_ratio ≈ 5.92e-08
- Score_estimate ≈ 2.12

Finalization: No nomination filed; cycle wrapped 2026-08-10 UTC.
- 24h NIG shocks (conservative): neutral-to-worse vs Gaussian + variance-regime on SP500/AAPLX; kept Gaussian.

### 24h Dispersion Micro-sweeps (2026-08-10 UTC)
- SP500 (end 2026-07-26, OU_THETA=0.02, 400 paths): long_sigma_mult=0.92 → mean_crps≈533.41; 0.94 → ≈532.63 (best in slice).
- AAPLX (end 2026-07-22, 400 paths): long_sigma_mult=0.96 → mean_crps≈2164.07; 1.00 → ≈2155.19 (best in slice).
Note: Gains are small; reverted config to documented defaults.

### COM_EQU_24H Combined — 2-day σ Tweaks (2026-08-10 UTC)
- Settings: SP500 long_sigma_mult=0.94; AAPLX=1.00; XAU baseline.
- Window: 2 days ending 2026-07-24; 400 paths.
- Result: rank=256/257; reward_weight≈1.97e-12.
Note: No material change; reverted to documented defaults.
