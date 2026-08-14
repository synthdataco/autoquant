# Deploy a champion as a live miner

A published champion is a runnable miner, not just a paper result. `synth_lib.serving` wraps its
`simulate()` in the live response contract and keeps a local price store warm so each request gets a
real 7-day context.

## Take a copy

Every leg in `campaign_results/` is **already unpacked** — `modeling.py`, the `ChampionMiner` subclass
in `miner.py`, and `entrypoint.sh` sit flat in the leg directory, alongside any artifacts the model
loads. Deploying one is a copy:

```bash
uv sync
cp -a campaign_results/campaign-2/claude champions/claude_opus_5
chmod +x champions/claude_opus_5/entrypoint.sh
```

Model artifacts (`.npz`, `.parquet`) are Git LFS objects. Run `git lfs install && git lfs pull` in
this repo first, or you will copy 130-byte pointer files that start fine and then fail when the
champion loads its calibration data mid-prompt:

```bash
find champions/claude_opus_5 \( -name '*.npz' -o -name '*.npy' -o -name '*.parquet' \) \
  -exec grep -l 'version https://git-lfs' {} +      # must print nothing
```

## Re-gate it

The published leg passed the live contract when it was unpacked. Re-run that gate against the
`synth-lib` you are actually deploying with, since the pinned contract can move under a champion that
has not changed:

```bash
uv run python -c "
from pathlib import Path
from synth_lib.serving.unpack_champion import contract_gate
contract_gate(Path('champions/claude_opus_5/modeling.py'), ('low', 'high'))
print('contract gate OK')"
```

Both response formats, regardless of which profiles the champion nominated — a deployed miner
receives every prompt the validator sends.

The gate exists because the campaign contract and the live contract differ: the validator requires
integer metadata fields and prices that round-trip through 8 significant digits, which raw float64
output from a backtested model does not satisfy. `miner.py` fixes exactly that, and the gate proves
it per champion.

> `synth_lib.serving.unpack_champion` produced these legs, but do not point it at one. It reads the
> **archive** layout — a `champion/` subtree, or a `workspace.bundle` checked out at `CHAMPION.sha` —
> which is how the private results repo stores a leg, not how this repo publishes one. Against a
> published leg it fails with `no champion source in ...: expected champion/ or workspace.bundle`.

## Warm the price store

```bash
for A in BTC ETH SOL XRP; do uv run python -m synth_lib.preparation.market_data --asset $A --days 30; done
for A in HYPE XAU NVDAX TSLAX AAPLX GOOGLX SP500 SPCX WTIOIL; do
  uv run python -m synth_lib.preparation.market_data --asset $A --days 3   # the rest, venue-limited
done
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
