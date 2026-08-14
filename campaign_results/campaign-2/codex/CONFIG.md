# Model Configuration (agent/config.json)

- Location: `agent/config.json`
- Purpose: Optional per‑asset overrides consumed at runtime by `agent/modeling.py`.

Keys (per asset, uppercase):
- `long_sigma_mult` (float): Additional 24h dispersion multiplier applied after global scaling.
- `theta` (float): 24h OU mean‑reversion strength (per‑hour), used only for 24h horizons.
- `long_anchor` (str, optional): 24h drift anchor; baseline uses `ema12` (12h EMA of log‑price).
- `short_shock` (str, optional): 1h shock family; one of `t`, `mixgauss`, `mixauto`. Default `t`.
- `mix_r` (float, optional): For 1h mixtures, ratio of large to small sigma; e.g., 4.0–5.0.
- `mix_p_cap` (float, optional): For `mixauto`, upper cap on jump probability (e.g., 0.08).
- `short_sigma_boost` (float, optional): Extra early‑minute dispersion (fraction), default 0.0.
- `short_tau_minutes` (float, optional): Decay horizon for the early‑minute boost.

Example:

```
{
  "SP500": { "long_sigma_mult": 0.96, "theta": 0.05 },
  "AAPLX": { "long_sigma_mult": 0.94, "theta": 0.05 },
  "XAU": { "long_sigma_mult": 0.95, "theta": 0.04 }
}
```

Notes:
- Absent keys fall back to built‑in defaults.
- Changes take effect immediately for subsequent runs; no code changes required.
