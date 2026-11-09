# Changelog

What changed from one campaign to the next. Each entry covers the agents' task, the panel and
envelope, the harness, the data, and how champions are scored. Entries are newest first and start
at campaign-3.

The engine is [synth-lib](https://github.com/synthdataco/synth-lib). Each entry names the engine
commit its legs ran and the PRs merged since the previous campaign.

Scoring changes reach every campaign, because older champions are rescored on the current verdict.
An entry's results are the ones reported for that campaign. The [README](README.md) leaderboard
holds the current numbers.

| Campaign       | First leg  | Cutoff → scored window   | Engine    | GPU        |
| -------------- | ---------- | ------------------------ | --------- | ---------- |
| campaign-6     | 2026-10-21 | 2026-10-11 → 10-12…10-22 | `6117938` | T4 (16 GB) |
| campaign-5-gpu | not run    | —                        | —         | —          |
| campaign-4     | 2026-09-29 | 2026-09-14 → 09-15…09-25 | `56fa368` | L4 (24 GB) |
| campaign-3     | 2026-09-08 | 2026-08-29 → 08-30…09-09 | `782a44f` | L4 (24 GB) |

## Unreleased: campaign-7

The pin is at `71504c8`. Nothing below has run in a campaign yet.

### Merged

- **A live champion gets the verdict's context** ([synthdataco/synth-lib#77]). It is pinned in
  [synthdataco/autoquant#21].
  - Serving keeps NaN minutes in the context, as the verdict does. Gaps used to be forward- and
    back-filled, which narrowed fans on a half-empty Hyperliquid window and collapsed some
    champions' fans altogether.
  - The price store re-reads only unsettled days, once a minute, instead of re-fetching settled
    days every 5 minutes.
- **Campaign-6 feedback** ([synthdataco/synth-lib#72]):
  - The probe and the verdict put the champion's folder first on the import path, as serving does.
    A champion can now import a file beside `modeling.py`.
  - A nomination whose interface check fails is handed back once, while budget remains.
  - `run --legs` runs a subset of the campaign's legs.
  - The driver passes proxy keys in a header, never in a URL.
  - There is one Synth API connection per thread.
- **The offline bundle is stored one file per day** ([synthdataco/synth-lib#81]).
  - Field scores, rewards and the miner pool live under
    `miner_scores/{competition}/{asset}/`, `rewards_history/{competition}/` and `miner_pool_usd/`,
    one `date=YYYY-MM-DD.parquet` per day.
  - A wider window fetches only its new days. A day that has not settled is kept as `.partial` and
    fetched again by the next build.
  - Verdicts share one bundle, under `offline_data/field`.
  - `--force-refresh` replaces `--chunk-days`. Bundles of single files still load.
- **CRPS without the field, on any window of the store** ([synthdataco/synth-lib#82]).
  - `predict.py --crps-only` answers an even grid, by default every 60 minutes for the 24h
    competitions and every 10 for crypto-1h, and scores each prediction against the store's prices.
    It needs no bundle and reports CRPS only: no field, no rank.
  - Every prompt is scored with the current crypto-1h formula.
  - It compares models on the same grid. It is not the validator's sample of start times, so it
    gives no absolute level.
  - Before 2026-07-23 the store holds today's instruments, not the Pyth prices the validator scored
    against.
  - The constitution points agents at it.
- **Hyperliquid minutes are the venue's 1-minute candles** ([synthdataco/autoquant#22] and
  [synthdataco/synth-lib#81]).
  - **Open, high and low come from the archive's own 1-second fields.** `ingest_reservoir.py` used
    to derive them from the 1-second closes, which matched the venue's candle on 81–94% of minutes.
    Built from the 1-second fields, they match on 100%.
  - **A minute without a trade is a flat candle**, as the venue prints it: open, high, low and close
    equal the previous close, and volume and trade count are 0. It used to be NaN. The validator
    scores against the same candles.
  - **The Hyperliquid client keeps those candles too.** [synthdataco/synth-lib#77] had dropped
    them. The store, the live context and the validator's realized path now see the same minutes.
  - **What this changes for agents:** a NaN close now means there was no price at all, and
    `trade_count == 0` means there was no trade. DATA.md and the constitution say so.
  - **A day with a venue-wide pause longer than 15 minutes is refused,** instead of being filled.
  - **Re-ingest the archive range** before building campaign-7's snapshot: with `--force-refresh`,
    or into a fresh store. A champion built on gaps keeps being scored on gaps.
- **The constitution names the live miner, and asks agents to explain their CRPS**
  ([synthdataco/synth-lib#84]).
  - Champions mine live after the campaign. Emission rewards every prompt answered (a miss scores
    as the field's 95th percentile) and a model that holds up on contexts rougher than the
    snapshot. CRPS stays the target.
  - A first method lesson: break a backtest's CRPS down by asset, configuration, hour and day, read
    the worst prompts against the field, and explain each anomaly before tuning.
- **A campaign names its kind** ([synthdataco/synth-lib#85]).
  - `kind:` in the campaign yaml picks the constitution, workspace, snapshot and interface check.
    No `kind:` means `monte-carlo`, the sampled-paths competitions, so existing yamls are
    unchanged and set up byte-identical.
  - Another package can register a kind. The verdict scores `monte-carlo` only.
- **A live champion starts on its stored calendar** ([synthdataco/synth-lib#86]). A calendar on
  the volume no older than a day is served from startup, and the first refresh replaces it. Without
  one, the assets wait for that refresh as before.
- **Daylight saving** ([synthdataco/synth-lib#87]).
  - The US session and the US releases keep New York time, and European hours London time, so in
    UTC they move an hour at each switch. The snapshot holds both regimes.
  - DATA.md lists the New York and London switches inside the snapshot and the next one after it,
    and the NYSE session in UTC in each regime.
  - The constitution says a profile can be keyed on New York or London time, and asks agents to
    measure which clock each asset follows. It does not prescribe one.
- **Field data from 2026-07-24 instead of the 32 days before the cutoff**
  ([synthdataco/synth-lib#88]). Campaign-6's agents had 32 days of field data and tuned mostly on
  the last few.
  - The bundle covers 2026-07-24 to the cutoff (`DAYS` in `docs/install-and-first-run.md`), and
    `scripts/audit_realized.py` needs `--days`.
  - Field-relative numbers start on each competition's first ranked round in DATA.md: 2026-08-03 in
    the 24h competitions, 2026-07-29 in crypto-1h. From there on:
    - the field is scored on the prices the store holds, and so are the rounds its moving average
      covers;
    - the competitions and assets are today's;
    - com-equ-24h's field history is complete.

    An earlier window has no field rank the engine can reproduce. `--crps-only` covers it.
  - The constitution explains the warm-up trim: `--days N` ranks only the rounds after a window's
    first `window_days`. With N no longer than `window_days`, nothing is dropped and the ranks rest
    on a truncated history.
  - The agent chooses its in-sample and out-of-sample windows. There is no holdout rule. A method
    lesson: validate a change across regimes over the whole span, not on the last weeks before the
    cutoff alone.
  - The constitution's history table gains the 2026-07-23 feed switch, the 2026-09-04 outlier cap
    and the 2026-10-28 prompt cadence.
  - DATA.md gives each competition's first ranked round, its scored prompts a day by period, and
    com-equ-24h's fixed daily start times.
- **`ingest_reservoir.py` reads a stored day again until it is settled** ([synthdataco/autoquant#27]).
  - It skips a stored day only when the partition says it is settled (`is_final`), as the price
    store does. It used to skip any stored day, so a partition a venue top-up wrote while its day
    was running, and that no later top-up completed, stayed partial.
  - Minutes the archive leaves without a price, before the day's first trade when no previous day
    is stored, keep the stored partition's candles.
  - A partition is written to a temporary file and renamed into place, so a killed run never leaves
    a truncated one.
  - A stalled route to the bucket fails the read sooner: 10 s to connect, 60 s per read, 3 attempts.
  - `--keep-going` logs a day the dex-pause check refuses and goes on with the next days, for
    seeding a live miner's store. Without it, the refusal stops the run as before.
- **The proxy prices the cache writes the gateway bills** ([synthdataco/autoquant#29]). The gateway
  reports and bills cache writes for gpt-6-astra ($12.50 / 1M) and Kimi K3.1 Pro (the input rate).
  Their entries priced cached reads only, so LiteLLM metered written tokens at zero: on a call writing
  a 5k-token prefix, astra metered $0.0003 where the gateway charged $0.068. With the write prices,
  the metered cost equals the gateway's.

### Planned, not built

CLIs:

- **Kimi Code 2.5.2.** The image install and the `kimi-k3.1-pro` alias are built here. The
  harness side, `effort` on kimi-code legs and the session lock, lands with the next synth-lib pin.
  - The image installs `@moonshot-ai/kimi-code@2.5.2` from npm, with the other CLIs. The
    `kimi-k3.1-pro` alias serves Kimi K3.1 Pro through the Vercel AI Gateway, on Moonshot's own
    hosting.
  - The harness writes `effort` to the leg's `~/.kimi-code/config.toml` as `[thinking] effort`, and
    kimi-code sends it as `reasoning_effort`: `low`, `high` or `max`, the K3 models' levels. Unset,
    kimi-code sends no effort at first, then `medium`, so a Kimi K3.1 Pro leg must set `effort`.
  - The harness turns kimi-code's session lock off. A lock left by a killed sandbox run would block
    every `-c`.
- **Antigravity CLI 1.5.2 (`agy`) for Gemini legs.** The image install is built here. The harness
  side, `antigravity-cli` legs in place of gemini-cli's, lands with the next synth-lib pin.
  - No gemini-cli release knows a Gemini 4 model, so campaign-6's gemini-4-argon leg ran at the
    API's default thinking level, `medium`. gemini-cli 0.61.0 stays in the image for archived
    campaigns.
  - The image installs `agy` from its release tarball, checked against its sha256, with
    auto-update off.
  - `agy` reaches the Gemini API through the proxy's `/gemini` passthrough, as gemini-cli does. The
    leg's proxy key is minted without a budget cap, as for gemini-cli, and the driver enforces the
    budget from the key's spend.
  - `effort` on an `antigravity-cli` leg is passed as `--effort`: `low`, `medium` or `high`, the
    levels of agy's Gemini models. agy refuses at launch a level the model lacks. Gemini 4 Argon's
    top level is `high`.
  - Some calls leave the leg's model. `agy`'s flash-tier subagents run on Gemini 3.8 Flash, now
    priced with its cached rate. Web search runs on its own model. The proxy does not meter Google
    Search grounding fees.
  - No documented setting turns `agy`'s telemetry off in a headless run.

Task:

- **The champion chooses its context length, with the next synth-lib pin.**
  - `CONTEXT_DAYS` in `modeling.py`, a plain integer from 3 to 30, is how many days of minute
    prices before each prompt `simulate()` receives. A champion without it gets 7, so every archived
    champion keeps its context.
  - The agent's backtests, the interface check, the verdict and the live miner read the same value.
    Any other value is refused by all of them, from the agent's first backtest on.
  - A context reaching back before an asset's first day in the store starts with NaN. A day missing
    after that first day still fails.
  - A live miner needs `CONTEXT_DAYS + 1` days in its store. Hyperliquid serves about 3.5 days, so a
    longer context needs the miner's store seeded from the archive before its first start
    ([docs/deploy-a-champion.md](docs/deploy-a-champion.md)).

Harness and verdict:

- **An interface check on live-like contexts,** with gaps and a short history.
- **A Score covers different rounds depending on the window's length.**
  - Without simulated registration, a backtest drops each competition's first `window_days` of
    reward rounds: 10 days for the 24h competitions, 5 for crypto-1h. The validator's moving average
    in those rounds reads scores from before the window, which the backtest does not fetch.
  - An 11-day window therefore scores the 24h competitions on its last day only.
  - The trim stays. Generating the champion over one smoothing window before the scored window was
    considered and dropped: on a verdict that window is before the cutoff, so it is in sample.
  - The constitution explains the trim to agents ([synthdataco/synth-lib#88]).
  - Still open: how to run a short interim verdict. `--simulate-registration` scores every round, as
    a miner registered when the window opens.
- **Calendar.**
  - Label or drop the earnings dates Nasdaq lists as estimates.
  - Add NYSE special closures, which are still added by hand.
  - Build the calendar daily from before the cutoff, so an event found later is not backdated.
- **Smoke.** The smoke test should cover:
  - a calendar read without a guard;
  - a `predict.py` window that contains events;
  - a champion that imports a file beside `modeling.py`.
- **Operations.**
  - Delete each leg's proxy key when the leg ends.
  - A leg takes about 12 GB of disk: share one wheel cache across legs.

### To decide

- The objective's window: 10 days or 30, now that the leaderboard reads 30.
- The panel.
- Whether a campaign with open internet access is campaign-7 or a separate campaign. In it, the
  agents could use any data source and their champions would define their own live context. It
  would need its own leaderboard.

## campaign-6

Panel:

- claude-code: opus-5-5 and muse-wonder (`muse-wonder-1.0`).
- codex: gpt-6-astra.
- gemini-cli: gemini-4-argon.

$100 / 18 h, engine `6117938`, $178.30 spent over the four legs. campaign-5-gpu did not run, so the
engine work done for it ran here first.

### Panel and envelope

- **opus-5 and gpt-6-sol are out;** gemini-4-argon and muse-wonder are in.
  - A second gemini-4-argon leg, on claude-code through the Vercel AI Gateway's Anthropic API, was
    planned and then dropped before it started.
- **The GPU is one T4** (16 GB, Turing, no bf16) on a 16 vCPU / 60 GB host. campaign-4 had one L4
  (24 GB) on 64 GB. The sandbox is unchanged: 14 CPU, 48 GB.
- **Effort is unpinned again.** The `effort`, `ultracode` and `goal` keys exist
  ([synthdataco/synth-lib#58]), but this campaign's yaml sets none of them.
- **CLI versions:** the image pins claude-code 2.1.296, codex 0.156.1 and gemini-cli 0.61.0, but
  both claude-code legs' transcripts record 2.1.281.

### Task

- **An event calendar** ([synthdataco/synth-lib#70], [synthdataco/synth-lib#71]).
  - The file is `market_data/calendar/events.parquet`, at the path in `$SYNTH_BENCHMARK_CALENDAR`.
  - It covers:
    - US macro releases: CPI, NFP, PPI, PCE, retail sales and GDP;
    - FOMC statements;
    - NYSE closures and early closes;
    - the EIA weekly petroleum report;
    - the earnings of the companies behind the tokenised stocks.
  - It holds every date published before the cutoff, upcoming ones included.
  - A champion reads the same file live, refreshed daily.
  - A new rule: event dates come only from this file.
- **`predict.py` answers the prompts the validator scored, at their real start times,** read from
  the bundle ([synthdataco/synth-lib#64]).
  - `--days N --eval-end D` takes any day up to the cutoff, and runs can go in parallel.
  - `--cadence-minutes` gives an even grid instead.
  - `--latest-formula` scores older prompts with the current crypto-1h formula.
- **New constitution notes:**
  - a fourth structural break: the crypto-1h volatility term, from 2026-09-22;
  - look a competition up by `(asset, time_length)`;
  - the API's `to` names a whole UTC day, inclusive;
  - parallel work (subagents, workflows, background jobs) spends from the same envelope.
- **"Lessons earlier agents paid for"**, five method notes:
  - price a residual before fitting it;
  - test a feature against a fast EWMA of realized volatility;
  - validate an event effect leave-one-event-out;
  - check what `simulate()` can see;
  - count the asset-window cells where a change helps.
- **DATA.md** adds measured facts (the index unit, flat stretches in scored paths) and a calendar
  section.

### Harness

From [synthdataco/synth-lib#58], [synthdataco/synth-lib#59] and [synthdataco/synth-lib#64]:

- **`effort`, `ultracode` and `goal` can be set per leg.**
- **The offline bundle is required.** `run` refuses to start without it, and so does `predict.py`.
- **A backtest reads prices only up to the last prompt's end.** A day missing from the read-only
  store raises before any fetch.
- **Each `predict.py` run gets its own predictions directory.**
- **`BUDGET.md` reports GPU-busy minutes and peak GPU memory.** `min_gpu_minutes` exists, and this
  campaign does not set it.

### Scoring

- **The validator's moving average is replayed exactly** ([synthdataco/synth-lib#60],
  [synthdataco/synth-lib#61], [synthdataco/synth-lib#62]):
  - the moving average covers the window the validator's update reads;
  - a re-registered uid scores as the two miners the validator sees;
  - late joiners are backfilled from the prompt the validator saved first;
  - `verdict.json` records `engine_rev`.
- **Predictions are matched backward** ([synthdataco/synth-lib#64]).
  - A prompt is answered by the latest prediction at or before it, within an hour.
  - It used to be the nearest prediction within ±30 minutes, so a later one could answer it.
- **Late joiners are gated** ([synthdataco/synth-lib#64]): a miner is out of an update until it has
  a scored prompt inside the window, and an absent candidate ranks last.
- **The leaderboard has one 30-day window per campaign** ([synthdataco/autoquant#19]).
  - Each champion is registered at its cutoff (`cutoff-30d`).
  - This replaces the 10-day `lead-in-2` table.
  - Row marks come from the verdict's own record ([synthdataco/autoquant#18]).

### Results

- **Interim verdict**, 10-11 → 10-21, live prompts:

  | Leg            | Score | Spend           |
  | -------------- | ----- | --------------- |
  | gemini-4-argon | 40.0  | 15.9 h / $70.95 |
  | opus-5-5       | 21.9  | 3.1 h / $11.78  |
  | gpt-6-astra    | 12.2  | 4.8 h / $88.41  |
  | muse-wonder    | 4.4   | 3.7 h / $7.16   |

  The window is 11 days, so the 24h competitions are scored on 10-21 alone (see
  [Planned, not built](#planned-not-built)).

- **gemini-4-argon with simulated registration on 10-25:** 25.9.
- **`cutoff-30d`** (10-11 → 11-10) can be scored from about 11-12. Until it is, campaign-6 has no
  leaderboard row.
- **The archive keeps the calendar the legs read:** `campaign_results/campaign-6/calendar/`.
- **gpt-6-astra's first nomination failed the 1h interface check.**
  - Its 1h path imported a file beside `modeling.py`, which the probe's loader could not find.
  - It fixed the import itself before landing, so its verdicts were unaffected.
  - The loader fix is [synthdataco/synth-lib#72].

## campaign-5-gpu

Not run, so it has no results. The design, committed as
[`campaigns/campaign-5-gpu.yaml`](campaigns/campaign-5-gpu.yaml) in [synthdataco/autoquant#18]:

- **Panel:** campaign-4's two strongest models, opus-5-5 (effort `max`, `goal`) and gpt-6-astra
  (effort `xhigh`).
- **A GPU-trained model is mandatory.**
- **Proof the GPU part matters:**
  - an ablation without the GPU-trained part;
  - every GPU run logged in `agent/gpu.md`;
  - `min_gpu_minutes: 60`.
- **Serving stays on CPU,** at about 5 s per prompt.

Its engine work ([synthdataco/synth-lib#64]) first ran in campaign-6.

## campaign-4

Panel:

- claude-code: opus-5 and opus-5-5.
- codex: gpt-6-astra and gpt-6-sol.

$100 / 18 h, engine `56fa368`, with claude-code 2.1.281 and codex 0.156.1. `56fa368` is the tip of
[synthdataco/synth-lib#57]'s branch, merged after the campaign started, so that PR's changes are
listed here.

### Panel and envelope

- **claude-fable-5-1 is off the panel.** Its safety classifier declines the campaign prompt under
  `cyber`, and the API serves the turn from a fallback model. Rewording the constitution did not
  clear it.
- **gpt-5.6-sol and gemini-3.8-flash are out;** opus-5, opus-5-5 and gpt-6-sol are in, and
  gpt-6-astra returns.
- **Effort is unpinned.** The CLIs' defaults applied: high for opus-5, medium for opus-5-5, low for
  gpt-6-astra and medium for gpt-6-sol.
- **Envelope and hardware are unchanged.**

### Task

- **The objective moves from emissions to CRPS** ([synthdataco/synth-lib#57]). It said "your Score
  is your simulated emissions relative to the field's best", and now says "a calibration problem:
  build the model with the lowest CRPS".
  - The constitution explains why the field position is a poor target. The per-prompt score is
    CRPS minus the prompt's best CRPS, so 0.1% of CRPS can move a rank by a hundred places.
  - A nomination reports CRPS per configuration, beside the starting model's, instead of ranks.
- **"The venue never closes; the underlying does"** ([synthdataco/synth-lib#50]). This replaces
  "do not build a session-of-day structure".
  - The tokenised equities' and gold's volatility swings 4–5× across the UTC day.
  - Weekdays run 2–4× hotter than weekends.
- **A champion is mandatory.** Ending without `agent/CHAMPION` scores 0.
- **New rules:**
  - keep binary and encoded data out of terminal output, a known classifier trigger;
  - seed any randomness from `SYNTH_BENCHMARK_SEED` ([synthdataco/synth-lib#55]).
- **The tone changes.** The constitution is reworded as "a research engineer working on your own",
  and the hard rules become "constraints that make the result valid".
- **DATA.md** gives the span of the scored prompts in the bundle, and the span of the cached
  realized paths.

### Harness

- **The agent workspace pins the engine the host runs** ([synthdataco/synth-lib#47]).
- **The resume prompt presses the agent to continue,** and a champion is required
  ([synthdataco/synth-lib#47]).
- **The driver detects a turn served by a substituted model** ([synthdataco/synth-lib#57]).
- **codex aliases in the proxy config are real model ids.** Under an invented alias codex cannot
  find the context window, which livelocked a campaign-3 leg.

### Scoring

- **Lead-in** ([synthdataco/synth-lib#49]).
  - The verdict generates one horizon before the window, so the candidate is never backfilled as a
    late joiner.
  - On campaign-3's gpt-6-astra, crypto-24h reward against the top miner went from 0.043 to 0.752.
- **One source of realized truth** ([synthdataco/synth-lib#51]).
- **Bundle files record their window,** checked before a cached file is reused
  ([synthdataco/synth-lib#52]).
- **A rank-evolution chart per competition** ([synthdataco/synth-lib#53]).
- **The crypto-1h volatility term** ([synthdataco/synth-lib#47]).
  - Each prompt is scored by the formula that was live for it.
  - synth-subnet moves to v1.13.0.
  - A non-finite path is a miss, not a NaN score.
- **The verdict sets `SYNTH_BENCHMARK_SEED` and records it** ([synthdataco/synth-lib#56]).
- **New verdict modes** ([synthdataco/synth-lib#57]):
  - `--live-prompts` answers the start times the validator actually scored;
  - `--simulate-registration` scores a champion as a miner deployed on a given day;
  - a verdict is a directory, with per-prompt scores and field rewards beside `verdict.json`;
  - generated predictions are cached across re-scores.
- **The leaderboard has one row per champion,** not per model. Otherwise campaign-4's gpt-6-astra
  would hide campaign-3's.

### Results

`lead-in-2`, rescored on `298a95d`:

| Leg         | Score | Spend           |
| ----------- | ----- | --------------- |
| opus-5-5    | 48.1  | 2.0 h / $8.10   |
| gpt-6-astra | 32.4  | 0.8 h / $18.32  |
| opus-5      | 7.6   | 12.2 h / $86.68 |
| gpt-6-sol   | 4.1   | 15.4 h / $4.72  |

Defects it exposed, fixed in campaign-6's engine:

- **No offline bundle reached the legs.** Every backtest called the live API, and its padding
  reached past the cutoff. Each such run crashed on the read-only partition, so no post-cutoff data
  reached a champion.
- **`champion/` and `CHAMPION` are one path on a case-insensitive filesystem.** Champions are now
  published under `champion_source/` ([synthdataco/synth-lib#59], [synthdataco/autoquant#15]).
- **The baseline verdict crashed** ([synthdataco/synth-lib#61]).
- **Prediction matching took the nearest prediction within ±30 minutes.**

## campaign-3

Panel:

- claude-code: claude-fable-5-1.
- codex: gpt-5.6-sol and gpt-6-astra.
- gemini-cli: gemini-3.8-flash.

$100 / 18 h, engine `782a44f`, with claude-code 2.1.263, codex 0.146.0 and gemini-cli 0.53.1.

### Panel and envelope

- **The panel is new.** campaign-2 ran claude-opus-5, gpt-5 and gemini-3.1-pro-preview.
- **gemini-3.8-flash was added mid-flight.**
- **Envelope and hardware are unchanged:** 14 CPU, 48 GB, one L4.

### Data

- **The context is minute OHLCV.** `context_prices` is a DataFrame with the columns `open`, `high`,
  `low`, `close`, `volume` and `trade_count`, where it used to be a Series of closes. Any column may
  hold NaN.
- **The Hyperliquid-routed assets come from Hydromancer's Reservoir 1-second archive**
  (`scripts/ingest_reservoir.py`, [synthdataco/autoquant#6]).
  - Open, high and low are the first, highest and lowest 1-second close.
  - Volume and trade counts are summed.
  - campaign-7 changes this: see [Unreleased: campaign-7](#unreleased-campaign-7).
- **DATA.md is generated at setup:** the schema, the NaN semantics, and coverage per asset by era.
- **Bundle fixes made during setup:**
  - one oversized score no longer aborts a bundle;
  - bundle requests are no longer paced at half a request per second;
  - venue ingest retries transient failures, a day at a time.

### Task

- **Loader expectations:** the strict `load_range` raises on any NaN, and `predict.py` shows the
  lenient pattern.
- **Removed from the constitution:**
  - the asset-coefficient example values;
  - the line that only near the best earns;
  - the list of 1h intervals.
- **Commit every deliverable before the deadline.**

### Results

`lead-in-2`, rescored on `298a95d`:

| Leg              | Score | Spend          |
| ---------------- | ----- | -------------- |
| gpt-6-astra      | 30.4  | 3.2 h / $93.60 |
| fable-5-1        | 8.1   | 2.9 h / $58.55 |
| gpt-5-6-sol      | 0.4   | 4.3 h / $23.55 |
| gemini-3-8-flash | 0.1   | 0.3 h / $1.51  |

- **gemini-3.8-flash nominated after 19 minutes.**
- **What it exposed:**
  - **A codex alias that is not a real model id livelocks the leg.** gpt-5.6-sol was restarted
    clean.
  - **Vercel AI Gateway credit is per team.** gpt-6-astra died on a 402 and resumed once its key
    moved to a team with credit.
  - **The verdict lacked a lead-in,** which penalised strong champions as late joiners. Fixed for
    campaign-4.
  - **Missing minutes flatter CRPS** instead of failing a run.

[synthdataco/synth-lib#47]: https://github.com/synthdataco/synth-lib/pull/47
[synthdataco/synth-lib#49]: https://github.com/synthdataco/synth-lib/pull/49
[synthdataco/synth-lib#50]: https://github.com/synthdataco/synth-lib/pull/50
[synthdataco/synth-lib#51]: https://github.com/synthdataco/synth-lib/pull/51
[synthdataco/synth-lib#52]: https://github.com/synthdataco/synth-lib/pull/52
[synthdataco/synth-lib#53]: https://github.com/synthdataco/synth-lib/pull/53
[synthdataco/synth-lib#55]: https://github.com/synthdataco/synth-lib/pull/55
[synthdataco/synth-lib#56]: https://github.com/synthdataco/synth-lib/pull/56
[synthdataco/synth-lib#57]: https://github.com/synthdataco/synth-lib/pull/57
[synthdataco/synth-lib#58]: https://github.com/synthdataco/synth-lib/pull/58
[synthdataco/synth-lib#59]: https://github.com/synthdataco/synth-lib/pull/59
[synthdataco/synth-lib#60]: https://github.com/synthdataco/synth-lib/pull/60
[synthdataco/synth-lib#61]: https://github.com/synthdataco/synth-lib/pull/61
[synthdataco/synth-lib#62]: https://github.com/synthdataco/synth-lib/pull/62
[synthdataco/synth-lib#64]: https://github.com/synthdataco/synth-lib/pull/64
[synthdataco/synth-lib#70]: https://github.com/synthdataco/synth-lib/pull/70
[synthdataco/synth-lib#71]: https://github.com/synthdataco/synth-lib/pull/71
[synthdataco/synth-lib#72]: https://github.com/synthdataco/synth-lib/pull/72
[synthdataco/synth-lib#77]: https://github.com/synthdataco/synth-lib/pull/77
[synthdataco/synth-lib#81]: https://github.com/synthdataco/synth-lib/pull/81
[synthdataco/synth-lib#82]: https://github.com/synthdataco/synth-lib/pull/82
[synthdataco/synth-lib#84]: https://github.com/synthdataco/synth-lib/pull/84
[synthdataco/synth-lib#85]: https://github.com/synthdataco/synth-lib/pull/85
[synthdataco/synth-lib#86]: https://github.com/synthdataco/synth-lib/pull/86
[synthdataco/synth-lib#87]: https://github.com/synthdataco/synth-lib/pull/87
[synthdataco/synth-lib#88]: https://github.com/synthdataco/synth-lib/pull/88
[synthdataco/autoquant#6]: https://github.com/synthdataco/autoquant/pull/6
[synthdataco/autoquant#15]: https://github.com/synthdataco/autoquant/pull/15
[synthdataco/autoquant#18]: https://github.com/synthdataco/autoquant/pull/18
[synthdataco/autoquant#19]: https://github.com/synthdataco/autoquant/pull/19
[synthdataco/autoquant#21]: https://github.com/synthdataco/autoquant/pull/21
[synthdataco/autoquant#22]: https://github.com/synthdataco/autoquant/pull/22
[synthdataco/autoquant#27]: https://github.com/synthdataco/autoquant/pull/27
[synthdataco/autoquant#29]: https://github.com/synthdataco/autoquant/pull/29
