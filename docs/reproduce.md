# Reproduce a published verdict

Every scored champion in this repo can be rescored from what is published here: the champion's source
at its nominated commit (`campaign_results/<campaign>/<leg>/champion_source/`), the campaign config,
and the window recorded in its `verdict.json`. The field's scores and realized paths come from the
public Synth API, so you do not need our snapshot to reproduce a *score* — only to reproduce an
agent's own backtests.

## What you need

- Python 3.12+, `uv`, Docker (the generation step runs network-isolated in a container).
- A local minute-price store covering the window plus 7 days of context before it — see the engine's
  [install-and-first-run.md](install-and-first-run.md) for how to fill one (Binance for the cryptos; Hyperliquid retains only ~3.5 days, so the
  tokenised equity/commodity assets need either accumulated history or Hyperliquid's S3 archive).
- Network access to `api.synthdata.co` for the field scores and realized paths.

## Steps

```bash
uv sync                                   # pulls synth-lib[benchmark], the engine
docker build -t synth-bench-sandbox "$(python -c 'import synth_lib.benchmark.sandbox as s, os; print(os.path.dirname(s.__file__))')"

# the window is in the verdict you are reproducing
python -c "import json; d=json.load(open('campaign_results/campaign-1/claude/verdict.json')); print(d['window'])"

uv run python -m synth_lib.benchmark.verdict.run_verdict \
    --campaign campaign-1 --legs claude \
    --window-start <start> --window-end <end> \
    --tag repro                            # writes verdict-repro.json, never overwrites ours
```

Compare `verdict-repro.json` with the published `verdict.json`: Score, per-competition ranks, per-asset
mean ranks.

## Why your numbers may differ slightly

- **Unseeded models.** Most champions draw from an unseeded RNG, so CRPS moves by a small amount run
  to run. Champions that measured it report a noise floor in their `CHAMPION` notes — for the ones
  that did, it is a few thousandths of a Score term.
- **The field's archived scores can change.** The API serves the validator's stored history; late
  arrivals or corrections shift a rank by a place or two.
- **Your price store is not our snapshot.** Model *context* comes from your local store. If your
  coverage differs (especially on Hyperliquid-routed assets), the champion sees slightly different
  inputs. Scoring truth is unaffected — that comes from the validator's realized paths.
- **Assets that cannot be scored** (retired feed, market not yet listed) land in `assets_failed`.
  Compare that list first; a different failure set explains a different competition score entirely.

## Scoring a champion on a different window

That is the interesting experiment: in-sample versus out-of-sample, or a later window to see whether
an edge persisted. Use `--tag <name>` so each window's verdict is its own file, and remember the
backtester refuses an unsettled window (it needs `now − horizon − 1 day`). Ranks per *asset* are the
comparable ones if your window crosses a change in how the subnet aggregates competitions.
