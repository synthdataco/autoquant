"""Cache field CRPS (per miner, per prompt) from the public Synth API, HARD-CAPPED at the data cutoff.

No query ever extends past CUTOFF (2026-09-14T00:00Z): anything later is our evaluation window.
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from synth_lib.backtester.loading import get_miner_scores

CUTOFF = datetime(2026, 9, 14, tzinfo=timezone.utc)
OUT = Path(__file__).resolve().parent.parent / "cache" / "field"
JOBS = [(a, 86400, 300) for a in ["BTC", "ETH", "SOL", "XRP", "HYPE"]] + \
       [(a, 3600, 60) for a in ["BTC", "ETH", "SOL", "XRP", "HYPE"]] + \
       [(a, 86400, 300) for a in ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]]


def job(a, tl, ti, start):
    end = CUTOFF
    assert end <= CUTOFF
    df = get_miner_scores(start, end, a, tl, ti)
    df = df[[c for c in ["miner_uid", "asset", "crps", "scored_time", "start_time", "time_length"] if c in df]]
    df["crps"] = df["crps"].astype(float)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{a}_{tl}.parquet")
    return a, tl, len(df)


if __name__ == "__main__":
    start = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=timezone.utc)
    with ThreadPoolExecutor(4) as ex:
        todo = [j for j in JOBS if not (OUT / f"{j[0]}_{j[1]}.parquet").exists()]
        for r in ex.map(lambda j: job(*j, start), todo):
            print(r, flush=True)
