# campaign_3_gemini-3-8-flash — provenance

- Campaign: `campaign-3`, leg `gemini-3-8-flash`
- Champion sha: `f1c010f9040fe1be283fff96affebcf5d9a5a3fe` (from `campaign_results/campaign-3/gemini-3-8-flash/CHAMPION`)
- Source: the champion's whole `agent/` tree at that sha, copied verbatim
- Runtime data present: none (code-only champion)
- Referenced but absent: none
- Added by the unpacker: `CHAMPION`, `miner.py`, `entrypoint.sh`, `PROVENANCE.md`, `workspace.bundle` (nothing else was modified)
- Nomination: `CHAMPION` is the archive's, the sha above. This agent committed no nomination file of its own.
- History: `workspace.bundle` — the champion's own git history. Verify this tree against the sha it claims:
  `git clone workspace.bundle /tmp/verify && git -C /tmp/verify rev-parse f1c010f9040fe1be283fff96affebcf5d9a5a3fe`
- Profiles: low, high
- Unpacked: 2026-09-10
- Contract gate at unpack: PASSED (`validate_responses` == CORRECT per profile, synthetic context)

The champion's research report, journal and full nomination notes live in
`campaign_results/campaign-3/gemini-3-8-flash/`.
