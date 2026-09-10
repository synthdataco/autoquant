# campaign_3_fable-5-1 — provenance

- Campaign: `campaign-3`, leg `fable-5-1`
- Champion sha: `9157d051638cbbb283c9f8ee98897d0313633bcc` (from `campaign_results/campaign-3/fable-5-1/CHAMPION`)
- Source: the champion's whole `agent/` tree at that sha, copied verbatim
- Runtime data present: `results/starter.json`, `results/v1.json`, `results/v1b.json`, `results/v2.json`, `results/v2_com-equ-24h_prompts.parquet`, `results/v2_crypto-1h_prompts.parquet`, `results/v2_crypto-24h_prompts.parquet`, `results/v3.json`, `results/v3_com-equ-24h_prompts.parquet`, `results/v3_crypto-1h_prompts.parquet`, `results/v3_crypto-24h_prompts.parquet`, `results/v4.json`, `results/v4_com-equ-24h_prompts.parquet`, `results/v4_crypto-1h_prompts.parquet`, `results/v4_crypto-24h_prompts.parquet`, `results/v4_verdict10.json`, `results/v4_verdict10_com-equ-24h_prompts.parquet`, `results/v4_verdict10_crypto-1h_prompts.parquet`, `results/v4_verdict10_crypto-24h_prompts.parquet`, `results/v5.json`, `results/v5_com-equ-24h_prompts.parquet`, `results/v5_crypto-1h_prompts.parquet`, `results/v5_crypto-24h_prompts.parquet`, `results/v5_verdict10.json`, `results/v5_verdict10_com-equ-24h_prompts.parquet`, `results/v5_verdict10_crypto-1h_prompts.parquet`, `results/v5_verdict10_crypto-24h_prompts.parquet`, `season.json`, `season_tune0601.json`, `season_tune0601_z.json`, `season_v2.json`, `season_v3.json`, `season_v4.json`
- Referenced but absent: none
- Added by the unpacker: `CHAMPION`, `miner.py`, `entrypoint.sh`, `PROVENANCE.md`, `workspace.bundle`, `CHAMPION.agent` (nothing else was modified)
- Nomination: `CHAMPION` is the archive's, the sha above. The agent's own nomination file is kept verbatim as `CHAMPION.agent`; it names an earlier commit, because a commit cannot contain its own sha.
- History: `workspace.bundle` — the champion's own git history. Verify this tree against the sha it claims:
  `git clone workspace.bundle /tmp/verify && git -C /tmp/verify rev-parse 9157d051638cbbb283c9f8ee98897d0313633bcc`
- Profiles: low, high
- Unpacked: 2026-09-10
- Contract gate at unpack: PASSED (`validate_responses` == CORRECT per profile, synthetic context)

The champion's research report, journal and full nomination notes live in
`campaign_results/campaign-3/fable-5-1/`.
