# Suggestions for the Synth research harness

- Expose an official exact-start generation helper. The backtester's 30-minute prediction-file
  tolerance silently permits a regular-grid forecast to be scored against a different start
  price, which can invalidate model selection.
- Remove retired SPYX from the pinned current competition configuration. Researchers otherwise
  need to construct an active-asset copy before combined COM/EQU replay.
- Make the snapshot-safe realization boundary explicit. The backtester prefetches beyond the
  nominal horizon, so a 1h evaluation ending on the data cutoff can request a later partition.
- Provide an option to reuse already computed per-asset score streams in combined replay.
  Nomination-grade 1000-path scoring is currently recomputed, which adds substantial campaign
  time without changing forecasts.
- Surface reward weights with scientific-notation precision in the standard total summary;
  six-decimal formatting hides meaningful differences among low-ranked candidates.
- Keep the generated `DATA.md` guidance about NaN semantics. It prevented interpolation of
  legitimate no-trade gaps and should remain a prominent part of future campaigns.

No private operator research code was encountered during this campaign.
