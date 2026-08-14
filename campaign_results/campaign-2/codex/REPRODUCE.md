# Reproduce Final Results (2026-08-08 UTC)

This repository is self-contained and runs fully offline with the bundled snapshot (prices through 2026-07-27). Always run from the workspace root and use `uv`.

- One-time setup:
  - `uv sync`

- Combined ranks at 1000 paths (validator parity):
  - COM_EQU_24H (SP500 AAPLX XAU; 2 days; eval_end=2026-07-24)
    - `uv run python agent/tools/combined_eval.py --competition COM_EQU_24H --assets SP500 AAPLX XAU --days 2 --eval-end 2026-07-24 --paths 1000`
  - CRYPTO_24H (BTC ETH SOL; 2 days; eval_end=2026-07-15)
    - `uv run python agent/tools/combined_eval.py --competition CRYPTO_24H --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000`
  - CRYPTO_1H (BTC ETH SOL; 2 days; eval_end=2026-07-15)
    - `uv run python agent/tools/combined_eval.py --competition CRYPTO_1H --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000`

- Final Score estimate (1000 paths):
  - `uv run python agent/tools/score_estimate.py --paths 1000`

- Baseline configuration in effect (excerpt from `agent/config.json`):
  - Equities (SP500, AAPLX, XAU): `long_anchor=ema12`, per-asset `long_sigma_mult`, `theta=0.02`.
  - Crypto 1h tails: `short_shock=mixauto` with ETH/SOL `mix_r=5.0`, `mix_p_cap=0.08`.

- Hard constraints (audited):
  - No-lookahead: do not query any API or data after 2026-07-27.
  - Prices: use the local snapshot only; do not fetch market data online.
  - Scoring: use synth-lib backtester only (already wired in tools).
  - Run from repo root so the backtester resolves the local store.

- Notes:
  - Expect combined ranks near: COM_EQU_24H ≈ 256/257, CRYPTO_24H ≈ 256/257, CRYPTO_1H ≈ 167/256 (1000 paths).
  - `agent/modeling.py` includes additional 24h shock/anchor options gated by config; the baseline keeps them off.

