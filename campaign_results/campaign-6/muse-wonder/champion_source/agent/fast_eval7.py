"""Out-of-window check on the fast proxy: July prompts (pre-bundle), BTC/SOL/XAU.
Compares width 1.0 / 0.95 / 0.90 / 0.85 under identical seeds."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd, json
from pathlib import Path
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
ART = json.loads(Path("agent/artifacts.json").read_text())
PHI = 0.5 ** (1/2880)

def crps_ens(obs, ens):
    ens = np.sort(ens); n = len(ens)
    return float(np.mean(np.abs(ens - obs)) - np.sum((2*np.arange(1, n+1) - n - 1) * ens) / (n*n))

def run(asset, close, width, window, n_paths=300, seed=7):
    rng = np.random.default_rng(seed)
    prof = np.asarray(ART[asset]["profile"])
    idx = close.index; p = close.values.astype(float)
    ok = np.isfinite(p) & (p > 0)
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_all = np.diff(lp); H = 1440
    total = 0.0; cnt = 0
    pos = {t: k for k, t in enumerate(idx)}
    anc = ART[asset].get("anchor")
    for st in pd.date_range(window[0], window[1], freq="12h", tz="UTC"):
        i = pos.get(st)
        if i is None or i < 7*1440 or i + H >= len(p): continue
        ctx_r = r_all[i-7*1440:i]; ci = idx[i-7*1440:i]
        if anc == "NY":
            loc = ci.tz_convert("America/New_York"); seas_c = prof[loc.weekday, loc.hour]
        else:
            seas_c = prof[ci.weekday, ci.hour]
        u = ctx_r / seas_c; u = u[np.isfinite(u)]
        cap = np.quantile(np.abs(u), 0.999); uc = np.clip(u, -cap, cap)
        sq = pd.Series(uc*uc)
        fast = max(float(sq.ewm(halflife=360, adjust=False).mean().iloc[-1]), 1e-12)
        slow = max(float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1]), 1e-12)
        mu = 0.5*np.log(slow) + np.log(width); start = mu + 0.85*(0.5*np.log(fast)+np.log(width)-mu)
        fut = idx[i:i+H]
        if anc == "NY":
            loc = fut.tz_convert("America/New_York"); seas_f = prof[loc.weekday, loc.hour]
        else:
            seas_f = prof[fut.weekday, fut.hour]
        log_sig = np.full(n_paths, start); eps_sd = 0.35*np.sqrt(1-PHI*PHI)
        rets = np.empty((n_paths, H))
        for j in range(H):
            rets[:, j] = np.exp(log_sig)*seas_f[j]*(rng.standard_t(5, size=n_paths)/np.sqrt(5/3))
            log_sig = mu + PHI*(log_sig-mu) + eps_sd*rng.standard_normal(n_paths)
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
        start = {"XAU": "2026-06-20"}.get(asset, "2026-06-20")
        close = load_minute_prices(WS/"market_data", asset, pd.Timestamp("2026-06-12", tz="UTC"),
                                   pd.Timestamp("2026-07-30", tz="UTC"))["close"]
        for w in [1.0, 0.95, 0.90, 0.85]:
            print(asset, "Jul w=", w, round(run(asset, close, w, ("2026-07-01","2026-07-28")), 3), flush=True)
