# Campaign plan

Objective: lower official CRPS in CRYPTO_24H, COM_EQU_24H and CRYPTO_1H, with valid, fast responses under short and gappy contexts.

All prices come from the frozen store. Calendar information comes from the supplied environment path. Cutoff: 2026-11-08. No online market-data requests. All Python runs use `uv run` from `/workspace`. No other agents are used.

1. Measure the seeded starter at 1000 paths; explain scores by asset, horizon, session and date. Read the exact validator scoring and benchmark context contracts.
2. Test robust multi-timescale volatility estimates against the seven-day constant Gaussian starter.
3. Measure local-clock intraday and weekly variance seasonality. Test causal profiles with shrinkage; explicitly compare New York, London and UTC clocks.
4. Test heavy-tailed innovations and stochastic volatility, with particular attention to the 1h volatility CRPS term. Compare changes on disjoint windows across the snapshot.
5. Test scheduled events and market closures using leave-one-event-out analysis. Fit only effects large and repeatable enough to justify their complexity.
6. Validate selected candidates on actual field prompt times, at 1000 simulations, across every asset/configuration. Diagnose individual failures and stress the interface and latency.
7. On the landing order, nominate a committed champion, write the public report and limitations, and commit all deliverables.

Primary decisions use mean CRPS per configuration and paired percent changes per asset-window cell. Avoid rank optimization. Initial screening uses fewer prompts, not fewer paths when practical. Require gains in both development windows and a majority of asset-window cells; confirm on another regime before selecting a small effect. Changes that improve one configuration can be isolated to it.

Initial development scored windows: 2026-08-17 through 2026-08-23 and 2026-10-05 through 2026-10-11. Recent confirmation: 2026-11-01 through 2026-11-07. Older tests will use CRPS-only with the latest 1h formula and fully covered prices. The regime characterization may lead to additional windows; record choices before scoring.
