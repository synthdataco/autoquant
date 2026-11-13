"""Merge validated GPU Direct-CRPS residual MLP heads into agent/calibration_params.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--calib", type=Path, default=WORKSPACE / "agent" / "calibration_params.npz")
    ap.add_argument(
        "--only",
        nargs="*",
        default=["mlpeq24"],
        help="Subset of GPU heads to merge (default: mlpeq24 only).",
    )
    args = ap.parse_args()

    calib_d = {
        k: v
        for k, v in np.load(args.calib, allow_pickle=False).items()
        if not k.startswith(("mlp1h_", "mlpc24_", "mlpeq24_", "lowdof1h_"))
    }
    meta = json.loads(str(calib_d["meta_json"]))

    heads_d = dict(np.load(args.heads, allow_pickle=False).items())
    kept = json.loads(str(heads_d["heads_json"]))
    allowed = set(args.only) if args.only else {"mlpc24", "mlpeq24", "mlp1h"}
    kept = [h for h in kept if h in allowed]

    for k, v in heads_d.items():
        if k != "heads_json" and any(k.startswith(f"{h}_") for h in kept):
            calib_d[k] = v.astype(np.float32)

    meta["gpu_residual_heads"] = kept
    calib_d["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(args.calib, **calib_d)
    print(f"Merged GPU residual heads {kept} into {args.calib} ({args.calib.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
