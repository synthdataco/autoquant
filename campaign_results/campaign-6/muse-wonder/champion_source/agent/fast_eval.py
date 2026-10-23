"""Fast offline CRPS-proxy tuner (NOT the official metric — for screening only).

Replicates the validator's interval construction at a reduced path count and with
analytic-ish empirical CRPS on log-return changes, so a (NU, VOLSD, FASTW) grid can be
screened in minutes. Survivors are verified with agent/predict.py.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "agent")

def crps_ens(obs, ens):
    ens = np.sort(ens); n = len(ens)
    return float(np.mean(np.abs(ens - obs)) - np.sum((2*np.arange(1, n+1) - n - 1) * ens) / (n*n))

def simulate_fast(sigma_path, nu, n_paths, rng, seas):
    """sigma_path: (minutes,) deterministic median vol (incl seasonality); SV adds dispersion."""
    T = len(sigma_path)
    z = rng.standard_t(nu, size=(n_paths, T)) / np.sqrt(nu/(nu-2))
    return sigma_path[None,:] * z  # 1-min log returns; SV handled by caller via extra factor

WS = Path(".").resolve()
from synth_lib.benchmark.generate_predictions import load_minute_prices
import json
ART = json.loads(Path("agent/artifacts.json").read_text())

def build(asset, days_eval=("2026-09-01","2026-09-28")):
    t0 = pd.Timestamp("2026-08-20", tz="UTC")
    t1 = pd.Timestamp("2026-09-29", tz="UTC")
    p = load_minute_prices(WS/"market_data", asset, t0, t1)
    close = p["close"]
    return close

def eval_cfg(asset, close, nu, volsd, fastw, n_paths=300, n_prompts=40, seed=7):
    rng = np.random.default_rng(seed)
    prof = np.asarray(ART[asset]["profile"])
    idx = close.index
    p = close.values.astype(float)
    ok = np.isfinite(p) & (p > 0)
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_all = np.diff(lp)
    # prompt starts: every 12h in eval window
    starts = pd.date_range("2026-09-02", "2026-09-28", freq="12h", tz="UTC")[:n_prompts]
    intervals = [(1,5),(6,30),(36,180),(288,1440)]  # steps of 5min -> horizons count
    total = 0.0; cnt = 0
    pos = {t:i for i,t in enumerate(idx)}
    for st in starts:
        i = pos.get(st)
        if i is None or i < 7*1440 or i+1440 >= len(p): continue
        ctx_r = r_all[i-7*1440:i]
        fut_idx = idx[i:i+1440]
        ctx_idx = idx[i-7*1440:i]
        valid = np.isfinite(ctx_r)
        if valid.sum() < 6000: continue
        seas_r = prof[ctx_idx.weekday[valid], ctx_idx.hour[valid]] if False else None
        # deseasonalise using timestamp-aligned profile
        cr = ctx_r.copy()
        ci = ctx_idx
        seas = prof[ci.weekday, ci.hour]
        u = np.where(valid, cr/np.where(valid, seas, 1.0), np.nan)
        u = u[np.isfinite(u)]
        cap = np.quantile(np.abs(u), 0.999); u = np.clip(u, -cap, cap)
        sq = pd.Series(u*u)
        fast = float(sq.ewm(halflife=360, adjust=False).mean().iloc[-1])
        slow = max(float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1]), 1e-12)
        mu = 0.5*np.log(slow); start = mu + fastw*(0.5*np.log(max(fast,1e-12))-mu)
        import os as _os; _w = float(_os.environ.get("W","1.0")); mu += np.log(_w); start += np.log(_w)
        seas_f = prof[fut_idx.weekday, fut_idx.hour]
        # simulate: AR(1) log vol at 1-min, t shocks
        phi = 0.5**(1/360)
        log_sig = np.full(n_paths, start)
        eps_sd = volsd*np.sqrt(1-phi*phi)
        rets = np.empty((n_paths, 1440))
        for j in range(1440):
            rets[:,j] = np.exp(log_sig)*seas_f[j]*(rng.standard_t(nu, size=n_paths)/np.sqrt(nu/(nu-2)))
            log_sig = mu + phi*(log_sig-mu) + eps_sd*rng.standard_normal(n_paths)
        real = lp[i:i+1441] - lp[i-1] if False else None
        real_lp = lp[i-1:i+1440]  # real_lp[0] = price at start-1... use diffs from start price
        # realized log-price relative to start (index i-1 is last context close? close[i] is first future)
        base = lp[i-1]
        real_cum = np.concatenate([[0.0], np.cumsum(r_all[i:i+1440])])
        sim_cum = np.concatenate([np.zeros((n_paths,1)), np.cumsum(rets, axis=1)], axis=1)
        # score at 5min steps: changes over horizons in interval_steps of 5min
        s5_sim = sim_cum[:, ::5]; s5_real = real_cum[::5]
        for steps_h in [1, 6, 36]:
            sc = np.diff(s5_sim[:, ::1], axis=1)  # placeholder
        # proper: changes over k*5min
        for k in [1, 6, 36]:
            sim_ch = (s5_sim[:, k:] - s5_sim[:, :-k]) * 1e4
            real_ch = (s5_real[k:] - s5_real[:-k]) * 1e4
            for t in range(sim_ch.shape[1]):
                if np.isfinite(real_ch[t]):
                    total += crps_ens(real_ch[t], sim_ch[:, t]* (s5_sim[0,0]*0+1)); cnt += 1
        # terminal absolute (scaled like validator: /last*1e4) — approximate with log diff
        total += crps_ens(real_cum[-1]*1e4, sim_cum[:,-1]*1e4); cnt += 1
    return total/max(cnt,1)

if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv)>1 else "BTC"
    close = build(asset)
    grid = [(5,0.35,0.85,1.0),(5,0.35,0.85,1.05),(5,0.35,0.85,1.10),(5,0.35,0.85,0.95),
            (6,0.25,1.0,1.05),(5,0.25,1.0,1.05),(8,0.25,1.0,1.05),(5,0.35,1.0,1.05)]
    for nu, volsd, fastw, width in grid:
        import os; os.environ["W"]=str(width)
        print(asset, f"nu={nu} volsd={volsd} fastw={fastw} width={width}",
              round(eval_cfg(asset, close, nu, volsd, fastw),3), flush=True)
