"""Paired test of the FOMC event multipliers on past FOMC days (prompts overlapping the release).

Usage: uv run agent/evtest.py
"""
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, "agent")
from harness import evaluate, CRYPTO, EQU  # noqa: E402
from data import load_asset  # noqa: E402

EVENTS = ["2026-03-18 14:00", "2026-04-29 14:00", "2026-06-17 14:00", "2026-07-29 14:00"]
if __name__ == "__main__":
    snap = f"results/snap_{int(time.time()*1000)}.py"
    shutil.copy("agent/modeling.py", snap)
    base = {"season_file": "agent/artifacts/seasonMF_2026-08-01.npz", "har_file": "agent/artifacts/har_2026-08-01.json",
            "har_file_1h": "agent/artifacts/har_2026-08-01.json"}
    rows = []
    with ProcessPoolExecutor(14) as ex:
        for e in EVENTS:
            t = pd.Timestamp(e, tz="America/New_York").tz_convert("UTC").tz_localize(None)
            for asset in CRYPTO + EQU:
                if load_asset(asset).index[0] > pd.Timestamp(t, tz="UTC") - pd.Timedelta(days=8):
                    continue
                cfgs = [(86400, t - pd.Timedelta(hours=24), t + pd.Timedelta(hours=1), "60min")]
                if asset in CRYPTO:
                    cfgs.append((3600, t - pd.Timedelta(minutes=60), t + pd.Timedelta(minutes=240), "5min"))
                for tl, s, en, g in cfgs:
                    for v in ("off", "on"):
                        r = evaluate(snap, asset, tl, s, en, {**base, "use_events": v == "on"}, 500, pool=ex, grid=g)
                        r["asset"], r["tl"], r["event"], r["var"] = asset, tl, e[:10], v
                        rows.append(r)
    d = pd.concat(rows)
    d.to_parquet("results/evtest.parquet")
    m = d.groupby(["tl", "asset", "event", "var"]).crps.mean().unstack("var")
    m["pct"] = (m["on"] / m["off"] - 1) * 100
    print(m.pct.unstack("event").round(2).to_string())
    print((m.groupby("tl").on.sum() / m.groupby("tl").off.sum() - 1) * 100)
