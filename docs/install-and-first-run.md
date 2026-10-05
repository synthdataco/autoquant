# Install, get data, smoke, run, score

Assumes a box provisioned per [provision-a-box.md](provision-a-box.md). Every command runs from the
root of this repo: `sandbox/`, `metering/` and `campaigns/` are here, and the engine
(`synth_lib.benchmark`) comes from the pinned `synth-lib` that `uv sync` installs.

## 1. Install

```bash
uv sync   # synth-lib[benchmark] is a required dependency; there is no extra to ask for

# Proxy: one master key + your provider keys. NEVER commit this file.
cp metering/.env.proxy.example metering/.env.proxy
$EDITOR metering/.env.proxy          # LITELLM_MASTER_KEY=$(openssl rand -hex 32), provider keys

docker compose -f metering/docker-compose.yaml up -d --force-recreate
curl -s localhost:4000/health/liveliness           # -> I'm alive!

docker build -t synth-bench-sandbox sandbox/
```

The proxy's model list is `metering/litellm_config.yaml`. Every alias a campaign names must exist
there; the file ships with examples for several providers, and each one you add needs explicit
prices under `model_info` (a model with no price meters as **zero spend**, which silently disables
the budget cap). Probe any new provider or CLI with [probe-a-new-cli.md](probe-a-new-cli.md) first.

## 2. Data

A campaign needs two different things, from two different places.

| What                                                       | Purpose                                                                                                             | Source                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1-minute closes per asset, in a local `market_data/` store | The **model's input**: the 7-day context each prompt hands to `simulate()`, and the price series agents backtest on | Venue clients in `synth_lib.preparation` (Binance, Hyperliquid), or your own archive |
| Realized paths + field scores/rewards                      | The **scoring ground truth**: the arrays the validator's own CRPS consumed, and the field to rank against           | The public Synth API, via `synth_lib.backtester.scripts.build_offline_bundle`        |

### Minute prices

`synth_lib.preparation` routes each asset to the venue it is scored against and writes daily parquet
partitions (`market_data/prices/{ASSET}/1m/date=YYYY-MM-DD.parquet`, a full 1440-row minute grid with
NaN where no trade occurred):

```bash
# Binance-routed assets: deep history straight from the venue. --days N ends today.
for A in BTC ETH SOL XRP; do uv run python -m synth_lib.preparation.market_data --asset $A --days 400; done

# Hyperliquid-routed assets: the venue holds ~3.5 days, so ask for 2 and no more (see below).
for A in HYPE XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500; do
  uv run python -m synth_lib.preparation.market_data --asset $A --days 2
done
```

**History depth differs by venue, and this is the step that decides your launch date.** Binance
serves deep minute history for the crypto assets. Hyperliquid's candle endpoint returns at most ~5000
minutes (about 3.5 days), so for HYPE and every tokenised equity/commodity perp the venue can only
fill the last three days. A campaign whose `data_cutoff` is older than that gets **no** Hyperliquid
history from `market_data` at all — and `com-equ-24h` is a third of the Score.

Hence `--days 2`, not 3. A request the venue cannot serve returns nothing, and `ingest_day` persists
that as a 1440-row all-NaN partition marked final: `build_snapshot` links it (it only errors when
*no* partition matches), agents read it as "listed, and traded zero times all day", and every later
ingest skips it.

Everything older comes from the archive:

```bash
uv sync
AWS_PROFILE=<profile> uv run python scripts/ingest_reservoir.py \
    --start 2025-08-01 --end <data_cutoff>          # --dry-run first
```

`scripts/ingest_reservoir.py` reads Hydromancer's Reservoir, an S3 archive of Hyperliquid history as
1-second candles partitioned by dex and day:

```
s3://hydromancer-reservoir/by_dex/{dex}/candles/1s/date=YYYY-MM-DD/candles.parquet
```

`{dex}` is the part of the coin id before the colon (`xyz:SP500` → `xyz`); native perps and spot
indices are under `hyperliquid`. The bucket is in `ap-northeast-1` and **requester-pays**: reads are
billed to your own AWS account, need credentials arranged with the provider, and carry
`RequestPayer=requester` on every call. Those credentials must be in the commercial partition — an
AWS European Sovereign Cloud account cannot reach the bucket, and no IAM policy bridges partitions.

Four things decide whether archive-derived minutes are usable. The script handles all four; they are
here because they are what to check if you write your own, or if a day looks wrong:

- **Listing dates bound the range.** The HIP-3 markets these tokenised assets trade on appear only
  from the day their market launched, so there is no history before that — a fact about the market,
  not a gap in the archive. The per-asset first-trade dates are in the script's `FIRST_TRADE`.
- **Aggregate to the last close in each minute.** `last(close)` over the 1-second candles inside a
  minute reproduces the venue's own 1-minute candle close. Aggregate a trade/candle close series;
  an order-book mid is a different series and will not match.
- **Watch the numeric type.** Prices arrive as fixed-point decimals, and decimal → float64
  conversion is not always correctly rounded (7413.9 can land on 7413.900000000001), which leaves
  an archive-derived minute one ULP from the same price fetched as JSON. Cast via string when the
  two must agree exactly.
- **Mirror the partition contract**: one `date=YYYY-MM-DD.parquet` per asset-day, a full 1440-row
  minute grid, NaN where no trade occurred. That is what `MinutePriceStore.ingest_day` produces and
  what the snapshot builder and `generate_predictions.py` expect.

Reads are grouped by `(dex, day)` and column-projected, so the nine assets cost two reads a day rather
than nine. Writing your own: `pyarrow.fs.S3FileSystem` has no `request_payer` option, so a projected
read of a requester-pays object needs a boto3 range-read wrapper.

When validating archive-derived minutes against minutes swept live from the API, compare against a
bulk-fetched window: a sweep that requests data up to `now` receives the current, still-forming
candle, whose close is only the last trade so far, so disagreement there reflects the live-swept
side rather than the aggregation.

### Audit the data before you spend anything

Two checks, and they read inside the partitions rather than counting filenames — `build_snapshot`
links the files that exist and `manifest.json` records only their names, so a store of half-written
days passes every structural check there is:

```bash
uv run python scripts/audit_store.py --cutoff <data_cutoff> --require-from 2026-02-01
uv run python scripts/audit_realized.py --cutoff <data_cutoff> --days 32
```

`audit_store.py` separates DAMAGE — a defect with a known repair — from review lines, which are
minutes in which nothing traded. Every asset trades 24/7, so those are a liquidity fact rather than
a session gap, and how many matter at a 24-hour horizon is your call, not the script's.

It also walks back from the cutoff and reports the oldest day on which EVERY asset has an
unbroken run — the real floor for any backtest that uses the whole field. `--require-from`
fails below a date you name, and separates an asset whose history simply begins there from one
with a hole to fill.

`audit_realized.py` fails when an asset has no cached validator paths: the backtester then scores it
from the local slice alone, and because a CRPS is a sum with NaN points dropped, a holed store
yields a smaller sum and flatters whatever is being scored.

Re-run `audit_store.py` against the snapshot after `setup` — that is what the agents actually mount:

```bash
uv run python scripts/audit_store.py --cutoff <data_cutoff> \
    --store campaign_runs/<name>/snapshot/prices
```

### Realized paths (before `setup`)

We cache the realized paths so the agent can backtest offline instead of fetching one request per
scored prompt from the live API, on its own clock. They are also what makes scoring work where your
store is thin: the backtester slices realized prices out of the same local store, and any prompt whose
slice contains NaN — every HL-routed asset older than ~3.5 days — is scored from the validator's own
array instead, verbatim, so local CRPS matches the live network. They must land in
`market_data/realized/` **before `setup`**, because that is what `build_snapshot` hard-links into the
snapshot the sandbox mounts.

```bash
for SLUG in crypto-24h com-equ-24h crypto-1h; do
  uv run python -m synth_lib.backtester.scripts.build_offline_bundle \
    --competition "$SLUG" --days 32 --eval-end <data_cutoff>
done
```

### Field scores (after `setup`, before `run`)

The same script also writes scores/rewards/pool parquets, which land **outside** `market_data/` and
therefore never reach the sandbox. Without them, agents fetch field scores from the live API — which
works, but is paginated one day per request, so a month-long backtest is a lot of traffic and
concurrent agents rate-limit each other. Build that half directly into the snapshot instead:

```bash
BUNDLE=campaign_runs/<name>/snapshot/offline_data
for SLUG in crypto-24h com-equ-24h crypto-1h; do
  uv run python -m synth_lib.backtester.scripts.build_offline_bundle \
    --competition "$SLUG" --days 32 --eval-end <data_cutoff> --out "$BUNDLE" --no-realized-paths
done
```

`run_campaign` then exports `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` into every sandbox — but only if
that directory exists. The slug is in the filenames, so one directory serves all three competitions.
Note the bundle is not covered by `manifest.json` (which tracks only `date=*.parquet`), so
`verify_snapshot` cannot detect tampering with it.

### Before `setup`

```bash
ls market_data/prices/BTC/1m | tail -3      # does coverage reach your data_cutoff?
```

`build_snapshot` errors only when _no_ partition matches the window; it cannot detect a `data_cutoff`
that overshoots your real coverage, so an optimistic cutoff produces a quietly incomplete snapshot
that fails later, inside an agent's budget. Keep `data_cutoff` at least `horizon + 1d` behind today
or the backtester rejects the window as unsettled. Hard links are made **at setup**: data ingested
afterwards is invisible to a campaign already set up.

## 3. Smoke first, always

Two smokes before a real budget. The first costs nothing:

```bash
# 1. fake CLI: exercises setup, workspace, budget file, landing, nomination probe, collection
#    (a campaign yaml with `- {id: dry, cli: fake, model: fake-model}`)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/dry-run.yaml
uv run python -m synth_lib.benchmark.run_campaign run   --campaign campaigns/dry-run.yaml

# 2. real CLIs, tiny envelope (a few dollars, minutes): every adapter, auth, and the sandbox
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/smoke.yaml
uv run python -m synth_lib.benchmark.run_campaign run   --campaign campaigns/smoke.yaml
```

Export **only** the master key. With provider keys also in your shell, some CLIs send their own
credential alongside the proxy's and bypass metering.

What the real smoke must prove, per leg:

1. The container starts and the agent finds `CAMPAIGN.md` at the workspace root.
2. `uv sync` succeeds **inside** the sandbox (it needs the `bridge` network for pinned packages).
3. `uv run agent/predict.py --asset BTC --days 2 --eval-end <date>` prints a `summary:` — proof the
   read-only snapshot is readable and synth-lib scoring works in-container.
4. `BUDGET.md` is rewritten with climbing spend and the landing order arrives.
5. When `CHAMPION` appears, `check interface:` in `BUDGET.md` reads `ok (24h + 1h)`.
6. Isolation: inside the container, `ls /workspace` shows no operator code, and
   `ls /workspace/market_data/prices/BTC/1m | tail` stops at `data_cutoff`.

## 4. Run a campaign

```bash
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/<name>.yaml
nohup uv run python -m synth_lib.benchmark.run_campaign run --campaign campaigns/<name>.yaml \
    >> ~/<name>.log 2>&1 &
tail -f ~/<name>.log
```

`setup` is not idempotent (it refuses a non-empty workspace). Legs run sequentially in yaml order.
The run log narrates every launch, phase change, spend decile and exit decision; the agents' own
output goes to `campaign_runs/<name>/artifacts/<leg>/transcript-N.log`.

Campaign yaml essentials: `models` (each `{id, cli, model}` where `model` **must** match a served
alias), `budget_usd_per_model`, `deadline_hours_per_model`, `data_start`/`data_cutoff`,
`forward_window_days`, `hardware` (free text, shown to the agent), `gpu`, `sandbox_cpus`,
`sandbox_memory_gb`, `poll_seconds`. Per-turn CLIs wait up to one poll interval between turns, so
keep `poll_seconds` small (30–60) when a leg's CLI exits after each turn.

Optional per-model keys, part of what a leg measures:

- `effort`: the CLI's own reasoning-effort level — claude-code: `low`, `medium`, `high`, `xhigh`,
  `max`; codex: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` (`ultra` also
  turns on proactive sub-agent delegation). Which levels a model accepts is up to the model.
  When omitted, each CLI applies its default for the model, which differs between models from the same
  vendor.
- `ultracode: true` (claude-code only): Claude plans a dynamic workflow for every substantive task.
- `goal: true` (claude-code only): an evaluator on the leg's model checks, at every stop, that the
  constitution allows it — the landing order carried out, or an evidenced early nomination.

## 5. Verdict

After the scoring window has passed (and settled — the backtester refuses a window younger than
`horizon + 1d`):

```bash
uv run python -m synth_lib.benchmark.verdict.run_verdict --campaign <name> \
    --window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>
```

Per leg it clones the champion at `CHAMPION.sha`, builds its venv in one bridge-network sandbox pass,
generates every (competition, asset) in `--network none` sandboxes against the read-only store, and
scores on the host. Results land in `campaign_results/<campaign>/<leg>/verdict.json`. Flags worth
knowing: `--legs` (subset), `--tag` (an extra window without overwriting the canonical verdict),
`--force` (rescore), `--skip-baseline`, `--no-gpu`, `--keep-work`.

**Disk**: predictions are generated at the validator's 1000 paths, so one leg's full set is on the
order of 10 GB. The runner deletes each leg's set after scoring unless `--keep-work`. Do not lower
`--num-simulations` to save space — the field's CRPS was computed at 1000 paths, and a smaller
sample biases the candidate's CRPS upward.
