"""Screen vol mean-reversion speed x volsd on the fast proxy (24h), BTC/SOL/XAU."""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd, json
from pathlib import Path
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
ART = json.loads(Path("agent/artifacts.json").read_text())

def crps_ens(obs, ens):
    ens = np.sort(ens); n = len(ens)
    return float(np.mean(np.abs(ens - obs)) - np.sum((2*np.arange(1, n+1) - n - 1) * ens) / (n*n))

def run(asset, close, nu=5, volsd=0.35, fastw=0.85, phi_hl=360, n_paths=300, seed=7):
    rng = np.random.default_rng(seed)
    prof = np.asarray(ART[asset]["profile"])
    idx = close.index; p = close.values.astype(float)
    ok = np.isfinite(p) & (p > 0)
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_all = np.diff(lp); H = 1440
    phi = 0.5 ** (1/phi_hl)
    total = 0.0; cnt = 0
    for st in pd.date_range("2026-09-02", "2026-09-28", freq="12h", tz="UTC"):
        i = {t: k for k, t in enumerate(idx)}.get(st)
        if i is None or i < 7*1440 or i + H >= len(p): continue
        ctx_r = r_all[i-7*1440:i]; ci = idx[i-7*1440:i]
        u = ctx_r / prof[ci.weekday, ci.hour]; u = u[np.isfinite(u)]
        cap = np.quantile(np.abs(u), 0.999); uc = np.clip(u, -cap, cap)
        sq = pd.Series(uc*uc)
        fast = max(float(sq.ewm(halflife=360, adjust=False).mean().iloc[-1]), 1e-12)
        slow = max(float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1]), 1e-12)
        mu = 0.5*np.log(slow); start = mu + fastw*(0.5*np.log(fast)-mu)
        fut = idx[i:i+H]; seas_f = prof[fut.weekday, fut.hour]
        log_sig = np.full(n_paths, start); eps_sd = volsd*np.sqrt(1-phi*phi)
        rets = np.empty((n_paths, H))
        for j in range(H):
            rets[:, j] = np.exp(log_sig)*seas_f[j]*(rng.standard_t(nu, size=n_paths)/np.sqrt(nu/(nu-2)))
            log_sig = mu + phi*(log_sig-mu) + eps_sd*rng.standard_normal(n_paths)
        rc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r_all[i:i+H]))])
        sim_cum = np.concatenate([np.zeros((n_paths,1)), np.cumsum(rets, axis=1)], axis=1)
        s5s, s5r = sim_cum[:, ::5], rc[::5]
        raw = r_all[i:i+H]
        for k in [1, 6, 36]:
            sim_ch = (s5s[:, k:] - s5s[:, :-k])*1e4; real_ch = (s5r[k:] - s5r[:-k])*1e4
            for t in range(sim_ch.shape[1]):
                if np.isfinite(raw[t*5:(t+k)*5]).all():
                    total += crps_ens(real_ch[t], sim_ch[:, t]); cnt += 1
        total += crps_ens(rc[-1]*1e4, sim_cum[:, -1]*1e4); cnt += 1
    return total/max(cnt,1)

if __name__ == "__main__":
    for asset in ["BTC", "SOL", "XAU"]:
        close = load_minute_prices(WS/"market_data", asset, pd.Timestamp("2026-08-20", tz="UTC"),
                                   pd.Timestamp("2026-09-29", tz="UTC"))["close"]
        for name, kw in [("hl360 sd.35", {}), ("hl720 sd.35", {"phi_hl":720}), ("hl720 sd.5", {"phi_hl":720,"volsd":0.5}),
                         ("hl1440 sd.5", {"phi_hl":1440,"volsd":0.5}), ("hl1440 sd.65", {"phi_hl":1440,"volsd":0.65}),
                         ("hl360 sd.5", {"volsd":0.5}), ("hl2880 sd.5", {"phi_hl":2880,"volsd":0.5})]:
            print(asset, name, round(run(asset, close, **kw), 3), flush=True)
