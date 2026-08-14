# Scoring

## The Score

```
Score = 100 × mean over the 3 competitions of min(1, champion_total_reward_weight
                                                    ÷ best_other_miner_total_reward_weight)
```

The champion is injected into the archived field as one more miner over the evaluation window, and the
whole thing runs through the **validator's own aggregation**: per-asset coefficients, per-miner
normalization, then a softmax into `reward_weight` — the quantity the subnet actually pays emissions
by. Summing the champion's `reward_weight` over every scoring round in the window simulates what it
would have **earned**; the Score expresses that as a fraction of the best *other* miner's earnings.

- **0** — earned nothing.
- **100** — matched or beat the field's best miner in every competition.
- Capped per competition *before* averaging, so one runaway competition cannot hide a weak one; a
  `beats_field` flag records where the cap bound.
- Averaged unweighted across the three competitions, because the subnet pays a third of emissions to
  each regardless of how many assets it contains.

Nothing here is reimplemented: the aggregation is the validator's code, applied to the field's own
archived scores.

## Why not a rank percentile

Because the subnet's payout is not linear in rank. The softmax concentrates emissions at the top of
the field, and the field itself is dense — hundreds of miners sit within a couple of percent of each
other on CRPS. A metric linear in rank would say that climbing from 50th to 49th is worth as much as
2nd to 1st; the emissions market says otherwise, by orders of magnitude. The Score inherits that
non-linearity on purpose: it answers "would this model have made money", which is the question a
miner actually faces.

The cost of that choice is resolution in the middle of the field, where most results land. Hence:

## The ranks, which are what you read to understand a result

Every verdict records, alongside the Score:

**Per competition** — the final-round rank out of the field (the position the validator would have
paid on), the mean and best rank across all the window's scoring rounds (`rank_over_rounds`), and the
field size. The mean rank is the robust one: a single lucky final round cannot flatter it.

**Per asset** — `mean_rank` and `best_rank` for each (asset, horizon) pair the champion served, plus
its mean CRPS and the number of prompts scored. This is where a model's real shape shows: a champion
can be rank 1 on one asset and invisible on another, and the competition-level number averages that
away.

Per-asset ranks are also the **only cross-window-comparable** rank we publish. Per-asset scores are
keyed by (asset, horizon, increment), so they stay meaningful across changes in how competitions are
aggregated or rewarded. Competition-level ranks and reward weights are only comparable within one
reward regime — the subnet reorganised its competitions in June 2026, and no field-relative
competition number spans that boundary.

Read them together: **Score** for "would it have earned", **per-competition mean rank** for "how
close", **per-asset mean rank** for "where the strength actually is".

## The evaluation window

A champion is scored over ten days of the subnet's own scoring rounds.
Which ten days is the choice that decides what the number means.

**Out of sample — the headline verdict.** The window starts after the campaign's data cutoff, so no
part of it existed in the snapshot the agent worked from: 
it is the only one that answers "would this model have earned, on data it had never
seen".

**In sample — the overfitting check.** The window lies *inside* the period the agent could study.
If a model's edge is real it should survive; if the
edge was fitted to a particular stretch of market, the in-sample window is where that shows

Two things to hold onto when reading across windows. Only **per-asset** ranks compare cleanly, for the
reason given above — a competition-level Score computed under a different reward regime is a different
measurement, not a worse one. And a window far enough back may not be able to score every competition
or every asset at all, because a competition or a market did not yet exist; `assets_failed` and a
Score computed over fewer than three competitions both mean the level is not comparable to a headline
verdict, even though the ranks inside it still are.

## What is scored, exactly

Per leg: the workspace bundle is cloned and checked out at `CHAMPION.sha`, so the scored code is
provably the nominated code. Its venv is built in one network-enabled sandbox pass (pinned packages,
never prices), then every (competition, asset) is generated in a **network-isolated** sandbox against
a read-only price store, at the validator's own 1000 paths per prompt. Each prompt receives only
prices up to its own timestamp — a truncation enforced at the API surface and covered by a dedicated
test. Scoring then runs on the host through the official backtester, with the field's realized paths
as ground truth.

A `synth_default` baseline (the subnet's stock generator) is scored on the same window where
available, as a sanity floor.

## Reading the fine print in a verdict

- `assets_failed` — assets that could not be scored (a retired feed, a market that did not exist
  yet). A leg with failures is not strictly comparable to one without: its competition score was
  computed on fewer assets than the validator would use.
- `num_prompts` per asset — how much evidence a rank rests on.
- `window.note` — whether the window is fully clean or carries a disclosed caveat.
- `mean_reward_vs_top` — the uncapped ratio behind the Score.
