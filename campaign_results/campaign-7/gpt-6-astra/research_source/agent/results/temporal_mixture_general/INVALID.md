This partial run is excluded from research comparisons. Its implementation
paired two separately scrambled Sobol engines by row index, inducing strong
finite-fan cross-correlations. It was stopped during the daily-crypto jobs as
soon as the sampling audit detected the error. The original implementation is
preserved as `agent/temporal_mixture_invalid.py` for diagnosis.

The corrected comparison uses one joint engine and a static reference with
identical coordinates. See iteration 41 in the journal.
