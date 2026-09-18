# AutoQuant - The final boss of Quant benchmarks

**A quant research benchmark whose leaderboard is a live market.**

Give a coding agent a sandbox, $100 of credit and a day. Its task: build a model that beats
professional quants who are being paid, right now, to beat it.

The task is real work — the kind that normally takes a skilled quant months of open-ended research.
The opponents are ~250 live miners on [Synth SN50](https://synthdata.co), financially incentivised to
win. The scorer is the subnet's own validator code. The metric is money.

**Still unsolved.** No frontier model, given one shot inside that envelope, has taken the top spot.
The best correctly-scored champion earns 25% of what the field's best miner earns, and its strongest
finish is 24th of 257 — it leads no competition. Most rows below predate two harness corrections and
are not comparable to it; see the footnote.

## Leaderboard

<!-- LEADERBOARD:BEGIN -->
| Model | Campaign | Score | Mean rank | Spend | crypto-1h | crypto-24h | com-equ-24h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 | campaign-2 | **30.1** \* | 78 | 3.8 h / $25.67 | 1/257 (μ60.5) | 81/257 (μ132.6) | 152/191 (μ181.8) |
| gpt-6-astra | campaign-3 | **24.7** | 98 | 3.2 h / $93.60 | 24/257 (μ38.8) | 46/257 (μ37.6) | 224/257 (μ192) |
| gpt-5 | campaign-2 | **10.2** \* | 211 | — | 194/257 (μ163.3) | 252/257 (μ253.8) | 187/191 (μ217.6) |
| gemini-3.1-pro-preview | campaign-2 | **8.1** \* | 210 | — | 206/257 (μ173.9) | 238/257 (μ242.2) | 186/191 (μ217) |
| claude-fable-5-1 | campaign-3 | **7.4** | 180 | 2.9 h / $58.55 | 194/257 (μ175.5) | 168/257 (μ172.6) | 178/257 (μ188.5) |
| glm-5.2 | campaign-1 | **6.6** \* | 207.7 | 2.4 h / $5.78 | 195/257 (μ185.3) | 243/257 (μ246.4) | 185/191 (μ211.8) |
| gpt-5.6-sol | campaign-3 | **2.7** \* | 212.3 | 4.3 h / $23.55 | 136/257 (μ222.8) | 247/257 (μ249.4) | 254/257 (μ252) |
| kimi-k3 | campaign-1 | **2.1** \* | 197.7 | 2.4 h / $4.96 | 185/257 (μ203.4) | 231/257 (μ235.3) | 177/191 (μ193.8) |
| gemini-3.8-flash | campaign-3 | **0.2** \* | 249.3 | 0.3 h / $1.51 | 247/257 (μ249.9) | 247/257 (μ246.1) | 254/257 (μ252.9) |

Score = 100 × mean over the three competitions of min(1, the champion's simulated emissions ÷
the best other miner's) — see [docs/scoring.md](docs/scoring.md). Rank cells are the final-round
rank out of the field, with the mean rank over the window's scoring rounds in parentheses.
One row per model — its best Score, and the campaign that produced it. Spend is wall time and
credits from the leg's own budget file, against an envelope of $100 and 18 h (24 h in
campaign-1); a leg that was restarted reports only its final attempt.

\* scored before two harness corrections: the generation lead-in, which had been charging
the candidate a horizon of worst-score rounds it was never given the chance to answer, and
uniform realized truth, without which missing minutes in the operator's price store shrank
the CRPS sum and flattered the champion. They pull in opposite directions and neither is
uniform across champions, so a marked row is not comparable to an unmarked one, and the
ordering between them is unreliable rather than merely optimistic or conservative.
<!-- LEADERBOARD:END -->

Generated with `synth-bench-leaderboard --write README.md`; every number is read from an archived
verdict, never typed by hand. `--tag is-0620` renders a diagnostic window instead.

## Why this benchmark is hard to fake

- **The opponents are paid to win.** Not a held-out test set: a live field of professional quants
  competing for real emissions, who improve every week.
- **It cannot be trained for.** Scoring runs on market data that did not exist when the model was
  written, and the field moves. There is no answer key to memorise and no static set to overfit.
- **It is ongoing.** Every campaign is scored on a fresh window, so a result is a claim about a
  specific ten days that anyone can re-verify.
- **The skill transfers.** What it takes to win — calibrated uncertainty on real price paths under a
  deadline — is what any serious modelling job needs. A system that wins here is a system you can
  point at other well-defined modelling work.
- **Anyone can enter.** Bring your own model, agent framework or harness; the engine is public and a
  dry run costs nothing. A champion that reaches the top of the field earns subnet emissions — and
  the bragging rights of having beaten paid quants with an agent.

Beat it and you have a runnable, provably-scored AI quant, plus the receipts.

## How it works

**The task.** One _campaign_ runs one _leg_ per model, sequentially on identical hardware. A leg gets
a deliberately minimal starter model, a frozen snapshot of 1-minute prices, the subnet's official
backtester, and a constitution: no lookahead, prices from the snapshot only, deliverables (journal,
report, champion commit), and the scoring formula — pre-registered, so agents design for the metric
instead of guessing it. [docs/task.md](docs/task.md).

**The envelope.** A fixed credit budget and wall clock, metered through one proxy ledger so every
model is measured with the same instrument. At 85% of either the harness issues a non-negotiable
landing order; nominating is final and forfeits the rest. Where an agent chooses to stop is part of
what is measured.

**The score.** The champion is injected into the archived field and run through the validator's own
aggregation. Because the subnet pays via a softmax, the Score is deliberately top-heavy: mid-pack
earns ≈ 0, so **ranks are published beside it** — per competition and per asset — to show how close a
zero was. [docs/scoring.md](docs/scoring.md).

**Reproducibility.** Every campaign publishes its config, the constitution each agent read, its
journal and report, the champion's source at the nominated commit, and the verdict JSON.
Rescore any champion yourself: [docs/reproduce.md](docs/reproduce.md). Or deploy one as a live
miner: [docs/deploy-a-champion.md](docs/deploy-a-champion.md).

## Results

| Campaign                                   | Panel                                                        | Envelope              |
| ------------------------------------------ | ------------------------------------------------------------ | --------------------- |
| [campaign-1](campaign_results/campaign-1/) | gemini-3.1-pro-preview, glm-5.2, kimi-k3                     | $100 / 24 h per model |
| [campaign-2](campaign_results/campaign-2/) | claude-opus-5, gpt-5, revised constitution                   | $100 / 18 h per model |
| [campaign-3](campaign_results/campaign-3/) | claude-fable-5-1, gpt-5.6-sol, gpt-6-astra, gemini-3.8-flash | $100 / 18 h per model |

Each leg directory **is** the champion: the agent's whole workspace at the commit it nominated —
`modeling.py` and the calibration artifacts it loads, `report.md` (the agent's own write-up, the most
interesting artifact here), `journal.md` (one entry per iteration, failures included), its research
scripts — plus `CHAMPION` (the nomination record), `verdict.json` and any `verdict-<tag>.json`
diagnostic windows, and the serving shell `miner.py` / `entrypoint.sh` that runs it as a live miner.

campaign-2 also carries three older **in-sample** windows per leg. They are the overfitting check, not
a second leaderboard: they showed the 1-hour edge decaying with distance from the data cutoff while
the 24-hour model held up. Read them with [docs/scoring.md](docs/scoring.md#the-evaluation-window).

## Enter it yourself

The engine is [`synth_lib.benchmark`](https://github.com/synthdataco/synth-lib); this repo is the
campaigns and results. `uv sync`, then [docs/run-a-campaign.md](docs/run-a-campaign.md). A full panel
needs a GPU-capable box, provider keys and a few hundred dollars; a `cli: fake` dry run costs nothing.

## Caveats worth reading before quoting a number

- **The field is alive.** Scores are relative to whoever was mining that week; only legs scored on the
  _same_ window compare cleanly. The table above mixes two: campaign-3 was scored on 2026-08-30 →
  09-09, campaign-1 and campaign-2 on 07-28 → 08-07. Rows from different windows are not a
  head-to-head.
- **Windows are short.** Ten days of scoring rounds; per-asset ranks are noisier than they look.
- **One box, sequential legs.** No interference between models, but no control over regime differences
  between them either.
- Constitutions, journals and reports are published **verbatim**, as the agents received and wrote
  them, with one exception: rule 5 of the constitution named a directory in the operator's own
  research repository, and that path is redacted here. Nothing else was edited.

## License

MIT (this repo). Agent-authored reports and journals are published as written.
