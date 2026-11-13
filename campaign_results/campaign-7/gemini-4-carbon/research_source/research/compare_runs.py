"""Compare two 3-window prompt parquet sets cell-by-cell (54 cells) and break down residual gaps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from synth.validator.moving_average import ASSET_COEFFICIENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-prefix", required=True, help="e.g. .../baseline")
    ap.add_argument("--curr-prefix", required=True, help="e.g. .../iter1")
    args = ap.parse_args()

    windows = [("W1", "w1"), ("W2", "w2"), ("W3", "w3")]
    cell_rows = []
    comp_summary = []

    for w_label, w_tag in windows:
        p_prev = Path(f"{args.prev_prefix}_{w_tag}_prompts.parquet")
        p_curr = Path(f"{args.curr_prefix}_{w_tag}_prompts.parquet")
        df0 = pd.read_parquet(p_prev)
        df1 = pd.read_parquet(p_curr)
        df0["scored_time"] = pd.to_datetime(df0["scored_time"], utc=True)
        df1["scored_time"] = pd.to_datetime(df1["scored_time"], utc=True)
        m = df0.merge(
            df1[["competition", "asset", "scored_time", "crps"]],
            on=["competition", "asset", "scored_time"],
            suffixes=("_prev", "_curr"),
        )

        for (comp, asset), g in m.groupby(["competition", "asset"], sort=False):
            c0 = float(g["crps_prev"].mean())
            c1 = float(g["crps_curr"].mean())
            fmin = float(g["field_min"].mean()) if "field_min" in g else float("nan")
            fmed = float(g["field_median"].mean()) if "field_median" in g else float("nan")
            pct = (c1 / c0 - 1.0) * 100.0
            cell_rows.append(
                {
                    "window": w_label,
                    "competition": comp,
                    "asset": asset,
                    "coef": ASSET_COEFFICIENTS.get(asset, 1.0),
                    "prev_crps": c0,
                    "curr_crps": c1,
                    "delta_pct": pct,
                    "field_min": fmin,
                    "field_med": fmed,
                    "beats_med": c1 < fmed,
                }
            )

        sub_cells = pd.DataFrame([r for r in cell_rows if r["window"] == w_label])
        for comp, grp in sub_cells.groupby("competition", sort=False):
            w0 = float(np.average(grp["prev_crps"], weights=grp["coef"]))
            w1 = float(np.average(grp["curr_crps"], weights=grp["coef"]))
            u0 = float(grp["prev_crps"].mean())
            u1 = float(grp["curr_crps"].mean())
            fmed_w = float(np.average(grp["field_med"], weights=grp["coef"]))
            comp_summary.append(
                {
                    "window": w_label,
                    "competition": comp,
                    "prev_w": w0,
                    "curr_w": w1,
                    "delta_w_pct": (w1 / w0 - 1.0) * 100.0,
                    "prev_unw": u0,
                    "curr_unw": u1,
                    "field_med_w": fmed_w,
                    "cells_improved": f"{int((grp['curr_crps'] < grp['prev_crps']).sum())}/{len(grp)}",
                }
            )

    cells_df = pd.DataFrame(cell_rows)
    comp_df = pd.DataFrame(comp_summary)

    print("=== COMPETITION-WINDOW SUMMARY ===")
    print(comp_df.to_string(index=False))

    print("\n=== 3-WINDOW POOLED COMPETITION SUMMARY ===")
    for comp, grp in comp_df.groupby("competition", sort=False):
        p_w = grp["prev_w"].mean()
        c_w = grp["curr_w"].mean()
        fm_w = grp["field_med_w"].mean()
        sub_c = cells_df[cells_df["competition"] == comp]
        wins = int((sub_c["curr_crps"] < sub_c["prev_crps"]).sum())
        print(
            f"  {comp:<12} | prev_3w={p_w:9.3f} -> curr_3w={c_w:9.3f} ({(c_w/p_w - 1)*100:+.2f}%) | "
            f"field_med_3w={fm_w:9.3f} | cells_improved={wins}/{len(sub_c)}"
        )

    tot_wins = int((cells_df["curr_crps"] < cells_df["prev_crps"]).sum())
    print(f"\nTOTAL CELLS IMPROVED: {tot_wins} / {len(cells_df)}")

    # Residual diagnosis by NY day-of-week on current run (gap to field_median)
    print("\n=== RESIDUAL DIAGNOSIS: CURRENT RATIO TO FIELD_MEDIAN BY NY START DAY-OF-WEEK ===")
    all_curr = []
    for w_label, w_tag in windows:
        d = pd.read_parquet(f"{args.curr_prefix}_{w_tag}_prompts.parquet")
        d["window"] = w_label
        all_curr.append(d)
    big = pd.concat(all_curr, ignore_index=True)
    big["scored_time"] = pd.to_datetime(big["scored_time"], utc=True)
    horizon_s = np.where(big["competition"] == "CRYPTO_1H", 3600, 86400)
    start_utc = big["scored_time"] - pd.to_timedelta(horizon_s, unit="s")
    ny = start_utc.dt.tz_convert("America/New_York")
    big["ny_dow"] = ny.dt.day_of_week
    big["ratio_med"] = big["crps"] / np.maximum(big["field_median"], 1e-6)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for comp in ["CRYPTO_24H", "COM_EQU_24H", "CRYPTO_1H"]:
        sub = big[big["competition"] == comp]
        parts = []
        for d_idx, d_name in enumerate(names):
            g = sub[sub["ny_dow"] == d_idx]
            if len(g) > 0:
                parts.append(f"{d_name}:{g['ratio_med'].mean():.3f}")
        print(f"  {comp:<12} | " + "  ".join(parts))

    print("\n=== WORST CELLS VS FIELD_MEDIAN (TOP 12) ===")
    cells_df["vs_med_pct"] = (cells_df["curr_crps"] / cells_df["field_med"] - 1.0) * 100.0
    print(
        cells_df.sort_values("vs_med_pct", ascending=False)[
            ["window", "competition", "asset", "prev_crps", "curr_crps", "delta_pct", "field_med", "vs_med_pct"]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
