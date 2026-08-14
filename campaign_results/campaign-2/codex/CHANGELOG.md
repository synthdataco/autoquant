# CHANGELOG — SN50 Research Campaign

This changelog summarizes key iterations and outcomes. See `agent/journal.md` for full detail.

- Iter 1–3: Baseline → EWMA vol; horizon‑aware tails; early 1h vs 24h tradeoffs.
- Iter 4–7: Robust dispersion (EWMA↔MAD); horizon σ tweaks; 1000‑path multi‑asset checks.
- Iter 8: Added official combined‑rank tooling; first field‑relative ranks.
- Iter 9–13: 24h OU drift toward 12h EMA; θ sweeps; per‑asset σ for equities.
- Iter 14–21: Validation scripts, documentation polish, and reproducibility helpers.
- Iter 22–27: Config docs; initial Score estimate (≈2.17), confirming non‑nomination.
- Iter 28–34: 1h mixture tails (fixed and adaptive); small lift to CRYPTO_1H (~167–168/256).
- Iter 35: 24h per‑asset calibration (SP500/AAPLX/XAU); combined rank unchanged (~256/257).
- Iter 36: Multi‑scale 24h drift anchor (blend); no rank movement.
- Iter 37: 1h `mixauto` with p‑cap; CRYPTO_1H holds ~167/256.
- Iter 38: Score estimate refresh (≈2.07).
- Iter 39: Stronger SP500 shrink probes; neutral/worse → revert.
- Iter 40: Trend‑aware 24h drift; worse → revert to ema12.
- Iter 41: Regime‑aware 24h dispersion (variance switch); no change.
- Iter 42: Crypto‑24h per‑asset σ/θ; combined rank ~256/257.
- Iter 43: 24h variance‑gamma; no change.
- Iter 44: 24h NIG; no change.
- Iter 45: Baseline restore; final combined checks.
- Iter 46: Config pruning to minimal baseline + 1h tails.
- Iter 47: Final Score estimate ≈1.95.
- Iter 48: Added REPRODUCE.md.
- Iter 49: Pruned miner_outputs; deliverables finalized. No CHAMPION.

Outcome: CRYPTO_1H modestly improved; 24h variants did not move field‑relative rank. Score estimate remains ≈1.95; no nomination.
