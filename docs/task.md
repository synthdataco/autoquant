# The task

## The objective, as the agents receive it

> Improve the Synth SN50 miner: produce the best possible price-simulation model as measured by CRPS.
> Your champion will be evaluated on the 10 days following your data cutoff, across the three
> competitions (CRYPTO_24H, COM_EQU_24H, CRYPTO_1H), against the real field of miners, using the
> validator's weighted ranking — your Score is your simulated emissions relative to the field's best
> miner. Choose your own research questions. Find something publishable — honest failures count too.

Nothing else is prescribed. The research questions, the model class, the validation design and the
decision to stop are all the agent's.

## What a miner actually does

For a given asset and start time, a miner returns **1000 simulated price paths** — a fan. The
validator compares that fan against the realized prices and scores it with **CRPS** (Continuous
Ranked Probability Score, lower is better) summed over several horizons. Good miners are
well-calibrated: neither too narrow (overconfidence is punished) nor too wide (imprecision is
punished). There is no "correct answer" to memorise; the target is a distribution.

## What each agent starts with

- **A standalone git workspace** — not our repository. It contains a deliberately minimal starter
  model (a log-normal martingale), an evaluator wired to the official backtester, and pinned
  dependencies. The agent has no access to any pre-existing model of ours.
- **A frozen, read-only snapshot** of 1-minute prices for every competition asset, ending at the
  campaign's `data_cutoff`, plus the validator's stored realized paths and an offline bundle of
  historical field scores — so backtesting against the real field needs no network. A generated
  `DATA.md` states the measured coverage per asset, including where data is genuinely thin.
- **The competition rules and scoring code** (public: the subnet's validator, the backtester library).
- **A budget, a clock, and a `BUDGET.md`** the harness rewrites continuously: credits spent, time
  elapsed, phase.

## Hard rules

1. **No lookahead.** A prediction at time _t_ uses only information available at _t_. The evaluation
   window sits after the data cutoff; querying any API for times past the cutoff is disqualifying,
   and transcripts are audited.
2. **Prices from the snapshot only.** The web is available for knowledge (papers, docs, code), never
   for market data.
3. **Scoring through the official backtester.** No reimplemented scorers — that is what makes legs
   comparable.
4. **No dissecting individual miners' served predictions.** Studying the field's aggregate behaviour
   is encouraged; copying a specific miner's paths is not.
5. **No reading the operator's own private research code.** It is not mounted; if it appears by any
   accident, the agent must report it rather than read it.
6. **The champion must run offline** from a committed git revision, exposing
   `simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices)`.

## Deliverables

`journal.md` (one entry per iteration: hypothesis, change, result with numbers, keep/discard verdict,
lesson), `report.md` (a write-up for the miner community, failures included), `CHAMPION` (the
nominated commit, the formats it serves, and its own backtested ranks), `suggestions.md` (anything out
of scope).

## The envelope, and why stopping is part of the task

Budget and wall clock are fixed per leg and metered through one proxy ledger. At 85% of either the
harness issues a non-negotiable landing order; if nothing lands within a grace period the agent is
killed and relaunched once with the landing order; at 100% it is killed.

**Nominating is final.** The first time an agent stops with a champion in place, the leg is over and
the rest of the envelope is forfeited. Agents are told this explicitly, which turns "when do I stop
researching?" into a real decision with a real cost — and the campaigns show models answering it very
differently, from eighteen minutes to a full envelope.
