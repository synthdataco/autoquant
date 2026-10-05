"""Paired variant comparison: runs harness.evaluate for base + variants over the same prompts
(same seed -> common random numbers) and prints the % CRPS change per asset and per group."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from harness import evaluate, CRYPTO, EQU  # noqa: E402

WINDOWS = {"V1": ("2026-08-01", "2026-08-15", "2026-08-01"),
           "V2": ("2026-08-29", "2026-09-13", "2026-08-29")}
GROUPS = {"C24": (CRYPTO, 86400), "E24": (EQU, 86400), "C1H": (CRYPTO, 3600)}


def run(variants: dict, groups=("C24", "E24", "C1H"), windows=("V1", "V2"), stride=2, model="agent/modeling.py",
        base_params=None, nsim=1000):
    base_params = base_params or {}
    # snapshot the model so edits made while this runs cannot leak into running workers
    import shutil, time as _t
    snap = f"results/snap_{int(_t.time()*1000)}.py"
    shutil.copy(model, snap)
    model = snap
    tasks = []
    for wname in windows:
        s, e, sf = WINDOWS[wname]
        for g in groups:
            assets, tl = GROUPS[g]
            for a in assets:
                for vname, vp in variants.items():
                    tasks.append((wname, g, a, tl, s, e, vname, {"season_file": f"agent/artifacts/season_{sf}.npz", "har_file": f"agent/artifacts/har_{sf}.json",
                                  "har_file_1h": f"agent/artifacts/har_{sf}.json",
                                  **base_params, **vp}))

    def one(t):
        wname, g, a, tl, s, e, vname, p = t
        p = {k: (v.replace("{W}", WINDOWS[wname][2]) if isinstance(v, str) else v) for k, v in p.items()}
        r = evaluate(model, a, tl, s, e, p, nsim=nsim, stride=stride, pool=ex)
        return dict(win=wname, grp=g, asset=a, var=vname, crps=r.crps.mean(),
                    vol=r.crps_vol.mean() if "crps_vol" in r else float("nan"), f_med=r.f_med.mean())

    from concurrent.futures import ThreadPoolExecutor
    with ProcessPoolExecutor(14) as ex, ThreadPoolExecutor(8) as tx:
        rows = list(tx.map(one, tasks))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["win", "grp", "asset"], columns="var", values="crps")
    base = list(variants)[0]
    rel = 100 * (piv.div(piv[base], axis=0) - 1)
    pd.set_option("display.width", 250)
    print(rel.round(2).to_string())
    print("--- mean % change by window/group")
    print(rel.groupby(level=[0, 1]).mean().round(3).to_string())
    print("--- overall mean by group")
    print(rel.groupby(level=1).mean().round(3).to_string())
    return df


if __name__ == "__main__":
    variants = json.loads(sys.argv[1])
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    run(variants, **kw)
