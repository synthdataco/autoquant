"""Run paired, native-cadence official backtests and save their summaries."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "agent" / "results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="ASSET:TIME_LENGTH")
    parser.add_argument("--eval-end", action="append", required=True)
    parser.add_argument("--model-module", action="append", default=None)
    parser.add_argument("--predictions-dir", default=None, help="isolated prediction store for this batch")
    parser.add_argument("--file-format", choices=["json", "npy"], default="json")
    parser.add_argument("--experiment-label", default=None)
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()
    modules = args.model_module or ["starter", "modeling"]
    for eval_end in args.eval_end:
        for case in args.case:
            asset, length_text = case.split(":")
            length = int(length_text)
            increment = 60 if length == 3600 else 300
            for module in modules:
                cmd = ["uv", "run", "agent/predict.py", "--asset", asset, "--days", str(args.days),
                       "--eval-end", eval_end, "--time-length", str(length),
                       "--time-increment", str(increment),
                       "--model-module", module]
                if args.predictions_dir:
                    cmd.extend(["--predictions-dir", args.predictions_dir])
                if args.file_format != "json":
                    cmd.extend(["--file-format", args.file_format])
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if result.returncode:
                    print(result.stdout[-4000:], flush=True)
                    print(result.stderr[-4000:], flush=True)
                    raise RuntimeError(f"Backtest failed: {case} {eval_end} {module}")
                summary_line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("summary:"))
                summary = ast.literal_eval(summary_line.removeprefix("summary: "))
                record = {"run_at": datetime.now(timezone.utc).isoformat(), "asset": asset,
                          "time_length": length, "time_increment": increment,
                          "prompt_selection": "exact historical starts", "file_format": args.file_format,
                          "eval_end": eval_end, "days": args.days,
                          "model_module": module, "summary": summary}
                if args.experiment_label:
                    record["experiment_label"] = args.experiment_label
                with RESULTS.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({"asset": asset, "length": length, "end": eval_end,
                                  "model": module, "crps": summary["mean_crps"],
                                  "prompts": summary["num_prompts"],
                                  "coverage": summary["realized_coverage"]}), flush=True)


if __name__ == "__main__":
    main()
