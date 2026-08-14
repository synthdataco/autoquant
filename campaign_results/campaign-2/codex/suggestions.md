## Follow‑ups and Suggestions

- Asset‑specific OU θ and σ: Learn θ, per‑asset 24h σ multipliers using a rolling calibration on pre‑cutoff windows; constrain via cross‑validation to avoid overfit.
- Regime awareness: Add simple volatility‑state switch (e.g., realized 5–15m variance threshold) to adjust dispersion and/or tails on the fly.
- Drift anchor variants: Test OU anchor as multi‑scale EMA blend (3h/12h) and as recent VWAP proxy from minute closes.
- Gap‑targeted calibration for 1h: Directly minimize gaps CRPS terms via small multiplicative adjustments to early‑minute dispersion.
- Distribution families: Try variance‑gamma or normal‑inverse‑Gaussian shocks for 1h; keep Gaussian for 24h.
- Latency audit: Profile per‑prompt latency at 1000 paths to keep <5 s ceiling; vectorize OU loop further (precompute eps and update in one shot).
- Data quality: Maintain permissive loader but track NaN density per asset/time; skip prompts with missing anchor context rather than patching last point.

- 24h shock families beyond Gaussian: test NIG or tempered stable with efficient samplers; calibrate to realized intraday ek and 24h_abs sensitivity; ensure unconditional variance preserved.
- Per-asset cross-validated σ/θ: nested validation with 2–3 disjoint windows post-2026-07-19 (SP500) using the official combiner to avoid overfitting.
- Regime modeling: discrete states keyed by realized variance/RSI/long-short EMA spread with per-state dispersion and drift priors; verify latency ≤5 s per prompt at 1000 paths.
- Crypto-specific 24h: investigate basis changes and weekend effects (still 24/7 perps) via context-derived adjustments that do not leak future info.

### Next-Budget Priorities (post Iterations 54–57)

- CRYPTO_1H per-asset grids: BTC/ETH/SOL × mix_r∈{4,5,6,7} × mix_p_cap∈{0.04,0.06,0.08,0.10,0.12}; select via combined_eval; confirm winners at 1000 paths.
- Early-minute calibration without global boost: fit per-asset minute-1..5 multipliers by minimizing gaps CRPS terms on a calibration window; strong regularization; verify no degradation on non-gap intervals.
- 24h anchors and θ: compare ema12 vs blend vs trend anchors; per-asset discrete-OU θ from cross-validated sweeps; guardrails via 24h_abs CRPS and combined COM_EQU_24H rank.
- 24h shock families: variance-gamma or NIG with variance preservation and kurtosis-bounded mixing; empirical check that 24h_abs doesn’t regress.
- Latency + throughput: profile numba/vectorization hotspots; pre-allocate arrays; ensure ≤5 s per 1000-path prompt; add a micro-benchmark to journal.
- Robustness checks: expand windows (± different regimes within snapshot); enforce no-lookahead; include ablation table in report.
EOF}

### Next-Budget Priorities (post Iterations 54–57)

- CRYPTO_1H per-asset grids: BTC/ETH/SOL × mix_r∈{4,5,6,7} × mix_p_cap∈{0.04,0.06,0.08,0.10,0.12}; select via combined_eval; confirm winners at 1000 paths.
- Early-minute calibration without global boost: fit per-asset minute-1..5 multipliers by minimizing gaps CRPS terms on a calibration window; strong regularization; verify no degradation on non-gap intervals.
- 24h anchors and θ: compare ema12 vs blend vs trend anchors; per-asset discrete-OU θ from cross-validated sweeps; guardrails via 24h_abs CRPS and combined COM_EQU_24H rank.
- 24h shock families: variance-gamma or NIG with variance preservation and kurtosis-bounded mixing; empirical check that 24h_abs doesn’t regress.
- Latency + throughput: profile numba/vectorization hotspots; pre-allocate arrays; ensure ≤5 s per 1000-path prompt; add a micro-benchmark to journal.
- Robustness checks: expand windows (± different regimes within snapshot); enforce no-lookahead; include ablation table in report.
EOF}

- Per-asset 24h dispersion: small improvements near SP500 long_sigma_mult≈0.94 and AAPLX≈1.00; validate across multi-day windows before adopting.
