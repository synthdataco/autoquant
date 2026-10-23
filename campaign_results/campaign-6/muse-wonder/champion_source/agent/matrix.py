"""Parallel CRPS matrix via the official predict.py. Usage: uv run python agent/matrix.py TAG"""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
ENVS = {}
if ":" in TAG:  # TAG like "v2:SYNTH_MODEL_NU=5,SYNTH_MODEL_VOLSD=0.35"
    TAG, spec = TAG.split(":", 1)
    ENVS = dict(kv.split("=") for kv in spec.split(","))
JOBS = []  # (asset, days, end, time_length, time_increment, latest, env)
WINDOW_A = ("2026-09-30", 3)
WINDOW_B = ("2026-10-11", 3)  # second window; scored prompts exist Sep..Oct per bundle span
for end, days in [WINDOW_A, WINDOW_B]:
    for a in ["BTC", "ETH", "SOL", "XRP", "HYPE"]:
        JOBS.append((a, days, end, 86400, 300, False, ENVS))
        JOBS.append((a, days, end, 3600, 60, True, ENVS))
    for a in ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]:
        JOBS.append((a, days, end, 86400, 300, False, ENVS))

def run(job):
    a, days, end, tl, ti, latest, extra = job
    cmd = ["uv", "run", "python", "agent/predict.py", "--asset", a, "--days", str(days),
           "--eval-end", end, "--time-length", str(tl), "--time-increment", str(ti)]
    if latest: cmd.append("--latest-formula")
    env = dict(os.environ); env.update(extra)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env, cwd="/workspace")
        line = [l for l in out.stdout.splitlines() if "mean_crps" in l]
        crps = line[0].split("'mean_crps':")[1].split(",")[0].strip() if line else "FAIL:" + out.stdout[-200:] + out.stderr[-200:]
    except Exception as e:
        crps = f"ERR {e}"
    cfg = "1h" if tl == 3600 else "24h"
    return f"{TAG}\t{cfg}\t{a}\t{end}\t{crps}"

with ThreadPoolExecutor(max_workers=3) as ex:
    for res in ex.map(run, JOBS):
        print(res, flush=True)
