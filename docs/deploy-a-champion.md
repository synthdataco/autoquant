# Deploy a champion as a live miner

A published champion is a runnable miner, not just a paper result. `synth_lib.serving` wraps its
`simulate()` in the live response contract and keeps a local price store warm so each request gets a
real context of the length the champion chose.

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

## Its context length

From campaign-7, a champion's `modeling.py` may set `CONTEXT_DAYS`: a plain integer at module level,
from 3 to 30. It is how many days of minute prices before each prompt `simulate()` receives, in the
champion's own backtests, in its verdict and live. A champion without it gets 7 days, as every
earlier champion did.

```bash
grep -n '^CONTEXT_DAYS' champions/claude_opus_5/modeling.py    # prints nothing: 7 days
```

Any other value, `14.0` or `True` included, is refused: the contract gate fails, and so does the
miner.

## Warm the price store

The miner keeps one store per asset, `market_data/prices/<ASSET>/1m/date=YYYY-MM-DD.parquet` under
the directory it runs from. Each request reads `CONTEXT_DAYS + 1` days of it: a window of D days
touches D + 1 daily partitions, the oldest and today's both partial. The miner warms the store at
startup and refreshes it every minute, from the venue each asset is scored against:

| Assets                                                      | Venue       | History the venue serves               | Before the first start                                |
| ----------------------------------------------------------- | ----------- | -------------------------------------- | ----------------------------------------------------- |
| BTC, ETH, SOL, XRP                                          | Binance     | all of it                              | nothing: the warm-up fetches every day                |
| HYPE, XAU, NVDAX, TSLAX, AAPLX, GOOGLX, SP500, SPCX, WTIOIL | Hyperliquid | the last 5,000 minutes, about 3.5 days | with `CONTEXT_DAYS` above 3, seed it from the archive |

Unseeded, a Hyperliquid-routed asset is served with about 3.5 days of prices and NaN before them,
and gains one day for each day the miner runs: a 30-day champion would wait four weeks for its full
context.

### Seed the Hyperliquid days from the archive

`scripts/ingest_reservoir.py` builds them from Hydromancer Reservoir's 1-second archive, the same
way it builds the campaign's store, which the champion was fitted and scored on. Run it before the
miner first starts, for the `CONTEXT_DAYS + 1` days up to yesterday; 31 days covers any champion.
Run it, and the miner, from the champion's own folder: its `market_data/` is then apart from the
campaign store in the repo root, and `uv` still finds the repo's project from there.

```bash
uv sync
cd champions/<name>
AWS_PROFILE=<profile> uv run python ../../scripts/ingest_reservoir.py \
  --start <today - CONTEXT_DAYS - 1> --end <yesterday>
```

- The bucket is requester-pays: the reads are billed to your AWS account, which must be in the
  commercial partition.
- **Seed before the first start.** The miner's warm-up stores the oldest day the venue still partly
  holds as settled, and the miner never fetches a settled day again.
- The day before the context's oldest day is seeded for its last close: the ingest carries it into
  the next day's minutes before their first trade.
- The archive publishes a day some time after it ends, and the venue covers the last three days, so
  seed on the day the miner starts.
- **Start each champion on an empty store**, as a new champion folder is. The ingest skips a day
  already on disk, and the miner skips a day it has settled, so a store reused from the campaign,
  another champion or an older engine keeps serving its old minutes.

## Run

From the champion's folder, which holds its store:

```bash
cd champions/<name>
NETWORK=finney NETUID=50 COLDKEY_NAME=<coldkey> HOTKEY_NAME=<hotkey> PORT=8091 ./entrypoint.sh
```

The miner warms up from the venues, starts a background refresh thread, and serves every competition
asset that has a live venue. Retired-feed assets are refused loudly rather than served from a dead
source — a crash you can see beats a silently wrong fan.

A second thread refreshes the event calendar at startup and daily into `market_data/calendar/`, and
points the champion at it through `SYNTH_BENCHMARK_CALENDAR`, as in its campaign. Set
`FRED_API_KEY` in the miner's environment. Without it the US macro releases are missing from the
live calendar, a champion that reads them serves without them, and the miner logs a warning at
each refresh.

There are **no defensive guards** on the request path: a data hole or an exploding path fails the
request instead of quietly degrading, so your monitoring sees it.

## Before you trust it

- Measure per-prompt latency on your hardware, at the champion's context length. The validator
  allows roughly the prompt's lead time (about a minute), but a champion above a few seconds per
  prompt at 1000 paths will strain both your box and any backtesting you do.
- Champions are frozen research artifacts: none of them adapt, retrain, or watch the field. Treat one
  as a starting point you now own, not a product.
- **Daylight saving.** The US session and the US releases keep New York time, and European hours
  keep London time. In UTC they move an hour twice a year: New York on the second Sunday of March and
  the first Sunday of November, London on the last Sundays of March and October.
  - A champion keyed on UTC hours keeps the profile it was fitted on. Fitted on summer months, it
    sits an hour early on the US session all winter; fitted on both regimes, it smears the session
    over two hours.
  - Check how yours keys time of day (`tz_convert`, `ZoneInfo`, or a bare UTC hour). Score it on live
    prompts after a switch before you rely on it.
