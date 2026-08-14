# Deploy a champion as a live miner

A published champion is a runnable miner, not just a paper result. `synth_lib.serving` turns one into
a Bittensor SN50 miner: it unpacks the champion, wraps its `simulate()` in the live response contract,
and keeps a local price store warm so each request gets a real 7-day context.

## Unpack

```bash
uv sync
uv run python -m synth_lib.serving.unpack_champion \
    --campaign campaign-1 --leg claude \
    --results-dir campaign_results --dest champions
```

This copies the champion's `modeling.py` (plus any artifacts it needs), generates a `ChampionMiner`
subclass, an `entrypoint.sh` and a `PROVENANCE.md` — and **gates the champion through the validator's
own `validate_responses`** on both response formats before writing anything. A champion that fails the
live contract is not unpacked.

The gate exists because the campaign contract and the live contract differ: the validator requires
integer metadata fields and prices that round-trip through 8 significant digits, which raw float64
output from a backtested model does not satisfy. The wrapper fixes exactly that, and the gate proves
it per champion.

## Warm the price store

```bash
uv run python -m synth_lib.preparation.market_data --assets BTC ETH SOL XRP --start-days-ago 30
uv run python -m synth_lib.preparation.market_data --recent-days 3        # the rest, venue-limited
```

Hyperliquid keeps ~3.5 days of minutes, so the tokenised equity/commodity assets start with a short
context and fill in as the miner's background refresh accumulates days. Run it a few days before you
care about those competitions' scores.

## Run

```bash
NETWORK=finney NETUID=50 COLDKEY_NAME=<coldkey> HOTKEY_NAME=<hotkey> PORT=8091 \
  ./champions/<name>/entrypoint.sh
```

The miner warms up from the venues, starts a background refresh thread, and serves every competition
asset that has a live venue. Retired-feed assets are refused loudly rather than served from a dead
source — a crash you can see beats a silently wrong fan.

There are **no defensive guards** on the request path: a data hole or an exploding path fails the
request instead of quietly degrading, so your monitoring sees it.

## Before you trust it

- Measure per-prompt latency on your hardware. The validator allows roughly the prompt's lead time
  (about a minute), but a champion above a few seconds per prompt at 1000 paths will strain both your
  box and any backtesting you do.
- Champions are frozen research artifacts: none of them adapt, retrain, or watch the field. Treat one
  as a starting point you now own, not a product.
