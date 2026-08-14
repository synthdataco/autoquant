# Run your own campaign

The engine is `synth_lib.benchmark` (in [synth-lib](https://github.com/synthdataco/synth-lib)); this
repo holds everything you deploy and everything a campaign produces: the sandbox `Dockerfile`, the
metering proxy's compose file and config, the campaign yamls, the results. This page is the map; the
detail is in [provision-a-box.md](provision-a-box.md), [install-and-first-run.md](install-and-first-run.md)
(install → data → smoke → run → verdict), [operate-a-campaign.md](operate-a-campaign.md) (what to run
while it is live) and [probe-a-new-cli.md](probe-a-new-cli.md) (before a new CLI or alias carries a
paid leg).

## What it costs

A full five-model panel at $100 per leg is a few hundred dollars of model spend plus a GPU box for a
few days. A `cli: fake` dry run costs nothing and exercises the whole machinery. A real smoke campaign
(a few dollars, minutes per leg) is **highly recommended** before a real budget: it exercises every
adapter, the auth path and the sandbox against live providers, which is where a misconfiguration would
otherwise surface hours into a paid leg.

## The path

```bash
uv sync

# 1. a box: >=16 vCPU, >=64 GB RAM, >=200 GB disk, one 24 GB GPU if your campaign sets gpu: true

# 2. the metering proxy: one master key + your provider keys, never committed
cp metering/.env.proxy.example metering/.env.proxy
docker compose -f metering/docker-compose.yaml up -d
docker build -t synth-bench-sandbox sandbox/

# 3. data: a minute store for the model's context, plus realized paths and the field-scores bundle
#    for scoring (install-and-first-run.md §2 — the retention asymmetry is the part to read twice)

# 4. write a campaign yaml (copy one from campaigns/) and dry-run it with cli: fake

# 5. smoke, then run
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/<name>.yaml
nohup uv run python -m synth_lib.benchmark.run_campaign run --campaign campaigns/<name>.yaml >> ~/<name>.log 2>&1 &

# 6. after the window settles: score, then regenerate the leaderboard
uv run python -m synth_lib.benchmark.verdict.run_verdict --campaign <name> \
    --window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>
synth-bench-leaderboard --write README.md
```

## Campaign design choices that matter

- **`data_cutoff` = launch day − 11.** The champion is then scored on the 10 days following the
  cutoff, which matches the validator's smoothing window and avoids the new-miner penalty. The
  trade-off is that the evaluation window exists while agents work, so the constitution's ban on
  querying any API past the cutoff is what protects it — check the transcripts.
- **Budget and deadline.** Wall clock binds sooner than money for most models. In the published
  campaigns no agent used more than about eight hours productively, so 18 h is generous; a shorter
  deadline biases against strategies that need to train something.
- **Model aliases must match the proxy's served names exactly**, and every custom alias needs explicit
  prices in the proxy config — a model with no price meters as zero spend and its budget cap never
  binds.
- **Poll interval.** CLIs that exit after every turn wait up to one poll interval to be relaunched, so
  30–60 s keeps that overhead small.

## Publishing results back here

Archive per [operate-a-campaign.md](operate-a-campaign.md): the curated per-leg artifacts, the constitution each agent
read, the redacted `state.json` (it holds the proxy keys), and the champion's source unpacked at its
nominated commit. Then regenerate the leaderboard — it reads the verdicts, so the table cannot drift
from them.
