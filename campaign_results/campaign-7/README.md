# campaign-7

Six legs, $100 / 18 h each, data cutoff 2026-11-08. The champions run as live miners on Synth SN50
instead of through a verdict, so campaign-7 has no Score. The constitution told the agents their
champion would be scored over the 10 days after the cutoff; that verdict was not run. What changed
since campaign-6 is in the [CHANGELOG](../../CHANGELOG.md#campaign-7).

| Leg             | CLI and effort                   | Spend          | Ended by      | Champion  | CONTEXT_DAYS | Serving set       |
| --------------- | -------------------------------- | -------------- | ------------- | --------- | ------------ | ----------------- |
| opus-5-5        | claude-code, `max`, goal         | 7.4 h / $76.60 | its own stop  | `8a06e83` | 21           | 3 files, 2.2 MB   |
| gpt-6-astra     | codex, `max`                     | 4.8 h / $94.85 | landing order | `5fbf857` | 7            | 22 files, 19.2 MB |
| gemini-4-argon  | antigravity-cli, `high`          | 4.7 h / $20.31 | its own stop  | `a1345d9` | 28           | 1 file, 35 KB     |
| kimi-k3-1-pro   | kimi-code, `max`                 | 1.4 h / $80.04 | its own stop  | `933c67a` | 14           | 1 file, 8 KB      |
| fable-5-2       | claude-code, `max`, goal         | 4.2 h / $98.13 | landing order | `6e3fc56` | 30           | 33 files, 1.6 MB  |
| gemini-4-carbon | antigravity-cli, `high`, gateway | 7.4 h / $25.30 | its own stop  | `225754d` | 30           | 2 files, 0.2 MB   |

- fable-5-2 and gemini-4-carbon were added after the first four legs had run, each from the same
  snapshot. gemini-4-carbon ran on engine `11fded2`, the others on `39bd62d`.
- A first fable-5-2 launch was stopped after $3.48 because its driver log was discarded. Its
  workspace was removed and the leg started again; the table reports the second launch.

## A leg directory

- `CHAMPION`, `journal.md`, `report.md`, `suggestions.md`: the agent's deliverables, verbatim.
- `CAMPAIGN.md` (the constitution the agent read), `BUDGET.md` (its last envelope report) and
  `transcript-*.log`.
- `champion_source/`: the files the champion reads while serving, at the nominated commit.
  `SERVING.json` lists each with its hash and why it is there, and the calls that proved the set:
  108 `simulate` calls (13 assets at 24h and 5 at 1h, six starts on both sides of the cutoff), run
  in the campaign's sandbox without network. Served alone, the set gives byte-identical output to
  the whole tree at the champion's commit on every call.
- `research_source/`: the research and training code behind the champion, and the files its
  report cites, with `RESEARCH.json`. It is not deployed.
- `workspace.bundle`: everything the agent committed.
  `git clone workspace.bundle ws && git -C ws checkout <CHAMPION sha>` gives the whole tree at the
  champion's commit.

## Publisher's notes

Agent-authored files are published as written. Each leg was audited against the constitution's
constraints twice, once from its own record and once from its transcript and git history, and each
finding was then checked by two reviewers asked to refute it. The findings below were upheld by
both. The antigravity-cli and kimi-code transcripts do not record every command in full, so the
no-lookahead audit is weaker on the gemini-4-argon, gemini-4-carbon and kimi-k3-1-pro legs.

- **gpt-6-astra:** no finding.
- **opus-5-5 and kimi-k3-1-pro:** no breach. Both stopped with envelope left, and the reviewers
  split on whether their journals show that more work would not have helped. opus-5-5's diagnosis
  scripts `research_source/agent/research/breakdown.py` and `inspect_prompt.py` call the
  validator's CRPS function directly; its experiments were scored with the backtester.
- **gemini-4-argon: `report.md` and `CHAMPION` give figures no run produced (constraint 4).** Its
  journal's CRPS figures match its logs, though the journal's protocol misstates the prompt counts
  too.
  - `CHAMPION` gives crypto-24h R2_Sep 1956.92 and R3_Oct 5635.09. The backtester printed 4045.45
    and 3546.56 (`transcript-0.log` line 1183). Each pair has the same sum, and the other two
    competitions' pairs agree to within 0.02: the 4-regime means are right, the split between
    regimes is not.
  - In `report.md` §4.3, 68 of the 72 per-asset champion values differ from the only run that
    scored all 72 cells.
  - The protocol is misstated. The backtester scored 168 prompts per 24h cell (70 in the eight
    com-equ-24h R4_Nov cells) and 576 per 1h cell, 19,472 in all, not 34 per cell (`stride = 5`)
    and 2,448 in all.
  - R2_Sep and R3_Oct are called holdout regimes "never inspected during model development". The
    transcript shows the field's scores over windows overlapping them printed before Iteration 1
    (line 126), and the champion's own scores on them printed before its last `modeling.py` edits
    (line 1183).
  - It nominated with most of its envelope left and no journal evidence that more work would not
    help. It scored many variants by calling the validator's CRPS function directly instead of the
    backtester (constraint 3).
- **fable-5-2: a torch copy of the metric judged experiments (constraint 3).**
  `research_source/research/train.py` replays the validator's CRPS in torch as a training loss.
  The constitution says "NEVER reimplement the metric" beside the backtester (`CAMPAIGN.md` line
  195), and its numbered constraints do not mention training losses. Its validation output also
  decided ablations and diagnoses, which constraint 3 reserves for the backtester. Every CRPS in
  `CHAMPION` comes from the backtester.
- **gemini-4-carbon: a torch copy of the metric judged experiments (constraint 3), disclosed by
  the agent.**
  - `research_source/research/train_iter6_gpu_direct_crps.py` (lines 443–488) rebuilds the
    validator's CRPS in PyTorch. It is the training loss of the served `mlpeq24` network. In
    Iterations 6–8 it decided which candidates went on to the backtester, which then dropped one of
    them (`mlpc24`). Iterations 3–5 called the validator's scoring function directly. Every CRPS in
    `CHAMPION` comes from the backtester.
  - It nominated at 7.4 h and $25.30. The journal's only evidence that more work would not help is
    that a few parameters sat at their local optimum (`journal.md` lines 278 and 293), while
    `report.md` §7 lists directions it did not test.
  - `CHAMPION` sections A and B, labelled strict out-of-sample, measure the calibration fitted up to
    2026-08-10, and their three windows decided what was kept. The nominated calibration, refitted
    up to the cutoff, was checked only on W3 (section C), which its fit covers.
  - `journal.md` line 30 reports a "Live Validator Format Bug in Starter". There is none: the live
    miner's wrapper converts the timestamp and the time increment and rounds the paths before the
    validator reads them.
  - Six Iteration 5 scripts were never committed, and are in neither the bundle nor the transcript.
