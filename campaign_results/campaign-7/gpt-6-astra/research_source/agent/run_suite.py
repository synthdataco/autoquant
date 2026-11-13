"""Bounded parallel runs of predict.py; every score uses synth-lib's backtester."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
OTHER = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="modeling")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ends", nargs="+", default=["2026-08-24", "2026-10-12"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--cadence", type=int, default=120)
    ap.add_argument("--offset", type=int, default=17)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--configs", nargs="+", default=["crypto24", "other24", "crypto1"])
    ap.add_argument("--assets", nargs="+")
    ap.add_argument("--field", action="store_true", help="CRPS-only at exact field times")
    ap.add_argument("--rank", action="store_true", help="full official backtest including field")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--event-types", nargs="+")
    ap.add_argument("--event-leads", nargs="+", type=int)
    args = ap.parse_args()
    directory = ROOT / "agent/results" / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for config in args.configs:
        assets = OTHER if config == "other24" else CRYPTO
        horizon, step = (3600, 60) if config == "crypto1" else (86400, 300)
        for end in args.ends:
            for asset in assets:
                if args.assets and asset not in args.assets:
                    continue
                jobs.append((config, end, asset, horizon, step))

    def run(job):
        config, end, asset, horizon, step = job
        prefix = directory / f"{config}_{asset}_{end}"
        if prefix.with_suffix(".json").exists() and not args.force:
            return json.loads(prefix.with_suffix(".json").read_text())
        cmd = ["uv", "run", "agent/predict.py", "--model-module", args.model,
               "--asset", asset, "--eval-end", end, "--days", str(args.days),
               "--time-length", str(horizon), "--time-increment", str(step),
               "--num-simulations", str(args.paths), "--output", str(prefix)]
        if args.details:
            cmd += ["--details"]
        if args.event_types:
            cmd += ["--event-types", *args.event_types]
        if args.event_leads:
            cmd += ["--event-leads", *map(str,args.event_leads)]
        if not args.rank:
            cmd += ["--crps-only"]
            if args.field:
                cmd += ["--field-times", "--stride", str(args.stride)]
            else:
                cmd += ["--cadence-minutes", str(args.cadence), "--offset-minutes", str(args.offset)]
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
        with prefix.with_suffix(".log").open("w") as f:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode:
            return {"error": str(prefix.with_suffix('.log')), "returncode": proc.returncode}
        return json.loads(prefix.with_suffix(".json").read_text())

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            if "error" in row:
                print(json.dumps(row), flush=True)
            else:
                print(f"{row['asset']:6s} {row['time_length']:5d} {row['eval_end']} CRPS={row['mean_crps']:.5f} n={row['num_prompts']} s/prompt={row['seconds_per_prompt']:.3f}", flush=True)
    (directory / "suite.json").write_text(json.dumps(rows, indent=2) + "\n")
    if any("error" in row for row in rows):
        sys.exit(1)
    for config in args.configs:
        horizon = 3600 if config == "crypto1" else 86400
        assets = OTHER if config == "other24" else CRYPTO
        selected = [r for r in rows if r['time_length'] == horizon and r['asset'] in assets]
        if selected:
            n = sum(r['num_prompts'] for r in selected)
            print(f"{config}: mean CRPS={sum(r['mean_crps']*r['num_prompts'] for r in selected)/n:.6f}, n={n}")


if __name__ == "__main__":
    main()
