"""Leave-out check of the event multipliers' power on releases 2026-07-01..09-07 (outside the field bundle, so scored
with the official synth crps_calculation on our own paths: our-vs-our paired comparison, no field involved).
uv run agent/analysis/evpow.py 3600|86400"""
import sys, os, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.backtester.config import competition_for
from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices
TL = int(sys.argv[1])
POWS = ["off", 1.0, 1.5, 2.0, 3.0]
CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
ALL = CRYPTO + ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]


def run(asset):
    os.environ["SEASON_FILE"] = "season_pre0908_hl21_l1_ev_moh.json"
    spec = importlib.util.spec_from_file_location("m", "/workspace/agent/modeling.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ev = pd.read_parquet(os.environ["SYNTH_BENCHMARK_CALENDAR"])
    ev["t"] = pd.to_datetime(ev.time_utc, utc=True)
    ev = ev[(ev.t >= "2026-07-01") & (ev.t < "2026-09-07") & ev.event.isin(list(m.EV_MULT))]
    ev = ev[(ev.event != "EIA_PETROLEUM") | (asset == "WTIOIL")]
    comp = competition_for(asset, TL); ti = comp.time_increment
    pr = load_minute_prices(Path("/workspace/market_data"), asset, pd.Timestamp("2026-06-20", tz="UTC"), pd.Timestamp("2026-09-10", tz="UTC"))
    rows = []
    for t0, name in zip(ev.t, ev.event):
        starts = [t0 - pd.Timedelta(minutes=k) for k in (range(2, 56, 6) if TL == 3600 else range(7, 23 * 60, 97))]
        for t in starts:
            grid = t + pd.to_timedelta(np.arange(TL // ti + 1) * ti, unit="s")
            real = pr["close"].reindex(grid - pd.Timedelta(seconds=60)).to_numpy().astype(float)
            if np.isnan(real).any():
                continue
            ctx = pr.loc[:context_end(t, TL)][-10080:]
            r = {"asset": asset, "event": name, "t0": t0}
            for p in POWS:
                os.environ["EVENTS"] = "0" if p == "off" else "1"
                os.environ["EV_POW"] = "1" if p == "off" else str(p)
                sims = np.asarray(m.simulate(asset, t.isoformat(), ti, TL, 1000, ctx)[2:], dtype=np.float32).astype(float)
                r[str(p)] = calculate_total_score_for_miner(sims, real, ti, comp.scoring_intervals, comp.vol_scoring_blocks)[0]
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(13) as ex:
        d = pd.concat(list(ex.map(run, CRYPTO if TL == 3600 else ALL)))
    d.to_csv(f"/workspace/agent/results/evpow_{TL}.csv", index=False)
    cols = [str(p) for p in POWS]
    rel = d[cols].div(d["off"], axis=0) - 1
    rel["event"] = d.event
    print("mean relative CRPS change vs no events, by event type (n prompts)")
    print(pd.concat([rel.groupby("event").mean().round(4), d.groupby("event").size().rename("n")], axis=1).to_string())
    print("by release (p=1 vs p=1.5 vs p=2):")
    r2 = rel.assign(t0=d.t0.values).groupby(["event", "t0"])[["1.0", "1.5", "2.0", "3.0"]].mean().round(4)
    print(r2.to_string())
