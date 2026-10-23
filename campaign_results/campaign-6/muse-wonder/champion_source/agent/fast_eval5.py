"""Bootstrap-style simulation: resample blocks of deseasonalised context returns.
Hybrid: empirical shocks (no t assumption) + SV overlay optional."""
import sys, warnings
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

def run(asset, close, mode="t", nu=5, volsd=0.35, fastw=1.0, block=30, n_paths=300, seed=7, H=1440):
    rng = np.random.default_rng(seed)
    prof = np.asarray(ART[asset]["profile"])
    idx = close.index; p = close.values.astype(float)
    ok = np.isfinite(p) & (p > 0)
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_all = np.diff(lp)
    phi = 0.5 ** (1/2880)
    total = 0.0; cnt = 0
    pos = {t: k for k, t in enumerate(idx)}
    for st in pd.date_range("2026-09-02", "2026-09-28", freq="12h", tz="UTC"):
        i = pos.get(st)
        if i is None or i < 7*1440 or i + H >= len(p): continue
        ctx_r = r_all[i-7*1440:i]; ci = idx[i-7*1440:i]
        # NY anchor for equities not needed for BTC screen; use stored profile as-is for crypto
        seas_c = prof[ci.weekday, ci.hour]
        u_all = ctx_r / seas_c
        m = np.isfinite(u_all)
        u = u_all[m]; uidx = np.where(m)[0]
        if len(u) < 5000: continue
        cap = np.quantile(np.abs(u), 0.999); uc = np.clip(u, -cap, cap)
        sq = pd.Series(uc*uc)
        fast = max(float(sq.ewm(halflife=360, adjust=False).mean().iloc[-1]), 1e-12)
        slow = max(float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1]), 1e-12)
        mu = 0.5*np.log(slow); start = mu + fastw*(0.5*np.log(fast)-mu)
        fut = idx[i:i+H]; seas_f = prof[fut.weekday, fut.hour]
        # standardised residuals for bootstrap (divide by slow sigma)
        resid = uc / np.sqrt(slow)
        log_sig = np.full(n_paths, start); eps_sd = volsd*np.sqrt(1-phi*phi)
        rets = np.empty((n_paths, H))
        if mode == "boot":
            # block bootstrap of resid
            nb = H // block + 1
            starts_b = rng.integers(0, len(resid)-block, size=(n_paths, nb))
            seq = np.concatenate([resid[s:s+block] for s in starts_b.ravel()]).reshape(n_paths, nb*block)[:, :H]
            vol_path = np.exp(log_sig)[:, None] * np.sqrt(slow)  # scalar start vol (per path = same start)
            # add SV path per path
            ls = np.full((n_paths, H), start);
            shocks_v = rng.standard_normal((n_paths, H))
            for j in range(1, H):
                ls[:, j] = mu + phi*(ls[:, j-1]-mu) + eps_sd*shocks_v[:, j]
            rets = np.exp(ls) * seas_f[None, :] * seq
        else:
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
    for asset in ["BTC", "SOL"]:
        close = load_minute_prices(WS/"market_data", asset, pd.Timestamp("2026-08-20", tz="UTC"),
                                   pd.Timestamp("2026-09-29", tz="UTC"))["close"]
        for name, kw in [("t5", {}), ("boot30", {"mode":"boot","block":30}),
                         ("boot10", {"mode":"boot","block":10}), ("boot60", {"mode":"boot","block":60})]:
            print(asset, name, round(run(asset, close, **kw), 3), flush=True)
