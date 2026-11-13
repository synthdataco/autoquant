# Suggestions outside this model's scope

- Expose interval-level CRPS details in the official backtest result. The validator already computes them; retaining them avoids instrumentation and makes price-versus-volatility diagnosis easier.
- Preserve the calendar's distinction between precise release timestamps and coarse earnings session anchors in the production interface. A date can be known while its exact release minute remains uncertain.
- Report valid realized-price coverage and missed-prompt counts alongside mean CRPS. Sparse references and unanswered prompts must remain visible when comparing models.
- Document the issuance lead and candle close convention next to the serving interface. Here the context ends three minutes before a 24h prompt and two minutes before a 1h prompt.
- Monitor the age of the latest observed price separately from volatility-state quality. In synthetic three-hour crypto context gaps, 71.9% of the CRPS penalty comes from the terminal absolute-price component; modest volatility aging cannot replace a fresh price anchor. The simulator contract requires paths to start at the last observed price.
- Publish a calendar generation identifier or content version alongside each refresh. Replacing a file while preserving modification time can leave consumers with stale schedules unless they also track inode and metadata changes.
- Record how often live requests contain almost no adjacent observed returns. A causal prior helped crypto materially in synthetic one-price tests, but a common rule did not pass the commodity/equity confirmation threshold; traffic prevalence is needed to prioritize a separate study.
- Make the interface check's runtime dependencies explicit and align them with the deployment environment. XGBoost was installed and pinned in this workspace but absent from the harness check. A portable NumPy tree export fixed this campaign's release while preserving exact predictions.

No operator miner models were encountered or used. All implementation ideas and fitted artifacts in this campaign use the allowed public scoring code, supplied snapshot/calendar, and this campaign's own research.
