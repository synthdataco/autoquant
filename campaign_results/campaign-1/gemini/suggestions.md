# Suggestions

- `synth-lib`'s `load_range` currently raises a `ValueError` if there are any missing minute price records (NaNs) in the requested context window. Since some assets (like SPCX) intrinsically have structural gaps in the provided parquet snapshots, using `load_range` blindly will crash the evaluation script for some assets. It would be beneficial to add a `fill_missing=False` flag or gracefully warn rather than crashing so miners can evaluate equities out-of-the-box.
- Adding a simple unified drift estimator to the validator's baseline models might improve the ecosystem's baseline performance, as pure random walks get disproportionately punished during strong trend days.
