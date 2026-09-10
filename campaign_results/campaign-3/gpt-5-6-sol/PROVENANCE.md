# campaign_3_gpt-5-6-sol — provenance

- Campaign: `campaign-3`, leg `gpt-5-6-sol`
- Champion sha: `5de025a38bf85c73e6866578b83ba7979f199778` (from `campaign_results/campaign-3/gpt-5-6-sol/CHAMPION`)
- Source: the champion's whole `agent/` tree at that sha, copied verbatim
- Runtime data present: none (code-only champion)
- Referenced but absent: none
- Added by the unpacker: `CHAMPION`, `miner.py`, `entrypoint.sh`, `PROVENANCE.md`, `workspace.bundle` (nothing else was modified)
- Nomination: `CHAMPION` is the archive's, the sha above. This agent committed no nomination file of its own.
- History: `workspace.bundle` — the champion's own git history. Verify this tree against the sha it claims:
  `git clone workspace.bundle /tmp/verify && git -C /tmp/verify rev-parse 5de025a38bf85c73e6866578b83ba7979f199778`
- Profiles: low, high
- Unpacked: 2026-09-10
- Contract gate at unpack: PASSED (`validate_responses` == CORRECT per profile, synthetic context)

The champion's research report, journal and full nomination notes live in
`campaign_results/campaign-3/gpt-5-6-sol/`.
