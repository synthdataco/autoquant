"""Diagnostics. Two questions:

(a) Where does the CRPS actually go, interval by interval?
(b) How much is left on the table by imperfect volatility forecasting?

(b) is answered with ORACLE models that peek at the realized path. They are diagnostics only —
they establish an upper bound on what a causal vol forecast can buy, and they never appear in
`modeling.py`. Anything labelled `oracle_*` here is lookahead by construction.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import build_prompts  # noqa: E402
from synth.validator.crps_calculation import calculate_total_score_for_miner  # noqa: E402


def breakdown(paths, real, comp):
    _, detail = calculate_total_score_for_miner(
        np.asarray(paths, float), np.asarray(real, float), comp.time_increment,
        comp.scoring_intervals, comp.vol_scoring_blocks,
    )
    return {d["Interval"]: d["CRPS"] for d in detail}


def realized_step_sigma(real: np.ndarray) -> float:
    r = np.diff(np.log(real))
    r = r[np.isfinite(r)]
    return float(np.std(r)) if r.size > 5 else np.nan


def run(asset, window, tl=86_400, ti=300, cadence=120, n_sims=1000, seed=7,
        model_spec=("models", "iid_model", {"dist": "t", "scale": 0.9, "nu": 4.0}),
        oracle=None, oracle_scale=1.0, n_max=None):
    import importlib

    mod = importlib.import_module(model_spec[0])
    model = getattr(mod, model_spec[1])(**model_spec[2])
    ps = build_prompts(asset, window[0], window[1], tl, ti, cadence)
    comp = ps.competition
    steps = tl // ti + 1
    agg = defaultdict(float)
    n = 0
    tot = []
    idxs = range(len(ps.starts)) if n_max is None else range(min(n_max, len(ps.starts)))
    for i in idxs:
        rng = np.random.default_rng(seed + i * 7919)
        real = ps.reals[i]
        if oracle is None:
            paths = model(asset=asset, start_time=ps.starts[i], time_increment=ti, time_length=tl,
                          num_simulations=n_sims, context_prices=ps.contexts[i], rng=rng)
        elif oracle == "level":
            # knows the realized average step-vol of the very window it predicts
            s = realized_step_sigma(real)
            if not np.isfinite(s):
                continue
            z = rng.standard_t(4.0, size=(n_sims, steps - 1)) / np.sqrt(2.0)
            sh = np.zeros((n_sims, steps))
            sh[:, 1:] = s * oracle_scale * z
            paths = float(pd.Series(ps.contexts[i]["close"]).dropna().iloc[-1]) * np.exp(np.cumsum(sh, axis=1))
        elif oracle == "path":
            # knows the realized |return| at every single future step
            r = np.abs(np.diff(np.log(real)))
            r = np.where(np.isfinite(r), r, np.nanmean(r))
            sig = pd.Series(r).rolling(6, min_periods=1, center=True).mean().to_numpy()
            sig = sig / np.nanmean(sig) * realized_step_sigma(real)
            z = rng.standard_t(4.0, size=(n_sims, steps - 1)) / np.sqrt(2.0)
            sh = np.zeros((n_sims, steps))
            sh[:, 1:] = sig[None, :] * oracle_scale * z
            paths = float(pd.Series(ps.contexts[i]["close"]).dropna().iloc[-1]) * np.exp(np.cumsum(sh, axis=1))
        else:
            raise ValueError(oracle)
        paths = np.asarray(paths, np.float32).astype(float)
        b = breakdown(paths, real, comp)
        for k, v in b.items():
            agg[k] += v
        tot.append(b["Overall"])
        n += 1
    return pd.Series({k: v / n for k, v in agg.items()}), n, float(np.mean(tot))


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    tl = int(sys.argv[2]) if len(sys.argv) > 2 else 86_400
    ti = 300 if tl == 86_400 else 60
    cad = 120 if tl == 86_400 else 30
    win = ("2026-07-05", "2026-07-23")
    rows = {}
    kw = dict(tl=tl, ti=ti, cadence=cad)
    rows["model"], n, _ = run(asset, win, **kw)
    for os_ in (0.8, 0.9, 1.0):
        rows[f"oracle_level_{os_}"], _, _ = run(asset, win, oracle="level", oracle_scale=os_, **kw)
    for os_ in (0.8, 0.9, 1.0):
        rows[f"oracle_path_{os_}"], _, _ = run(asset, win, oracle="path", oracle_scale=os_, **kw)
    tab = pd.DataFrame(rows).T
    tab["rel_%"] = tab["Overall"] / tab.loc["model", "Overall"] * 100 - 100
    print(f"\n{asset}  n={n} prompts   window={win}")
    print(tab.round(1).to_string())
