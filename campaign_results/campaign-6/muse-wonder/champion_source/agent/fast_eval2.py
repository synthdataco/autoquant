"""Screen drift / jump-diffusion / seasonal-shape variants on the fast proxy (BTC+XAU, Sep)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd, json
from pathlib import Path
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
ART = json.loads(Path("agent/artifacts.json").read_text())
PHI = 0.5 ** (1/360)

def crps_ens(obs, ens):
    ens = np.sort(ens); n = len(ens)
    return float(np.mean(np.abs(ens - obs)) - np.sum((2*np.arange(1, n+1) - n - 1) * ens) / (n*n))

def load(asset, start):
    p = load_minute_prices(WS/"market_data", asset, pd.Timestamp(start, tz="UTC"),
                           pd.Timestamp("2026-09-29", tz="UTC"))
    return p["close"]

def run(asset, close, nu=5, volsd=0.35, fastw=0.85, drift_k=0.0, jump_p=0.0, jump_sd=3.0,
        seas_pow=1.0, n_paths=300, seed=7, horizon_h=24):
    rng = np.random.default_rng(seed)
    prof = np.asarray(ART[asset]["profile"]); prof = prof ** seas_pow; prof /= prof.mean()
    idx = close.index; p = close.values.astype(float)
    ok = np.isfinite(p) & (p > 0)
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_all = np.diff(lp); H = horizon_h * 60
    starts = pd.date_range("2026-09-02", "2026-09-28", freq="12h", tz="UTC")
    pos = {t: i for i, t in enumerate(idx)}
    total = 0.0; cnt = 0
    for st in starts:
        i = pos.get(st)
        if i is None or i < 7*1440 or i + H >= len(p): continue
        ctx_r = r_all[i-7*1440:i]; ci = idx[i-7*1440:i]
        valid = np.isfinite(ctx_r)
        if valid.sum() < 6000: continue
        seas = prof[ci.weekday, ci.hour]
        u = ctx_r / np.where(valid, seas, 1.0); u = u[np.isfinite(u)]
        cap = np.quantile(np.abs(u), 0.999); uc = np.clip(u, -cap, cap)
        sq = pd.Series(uc*uc)
        fast = max(float(sq.ewm(halflife=360, adjust=False).mean().iloc[-1]), 1e-12)
        slow = max(float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1]), 1e-12)
        mu = 0.5*np.log(slow); start = mu + fastw*(0.5*np.log(fast)-mu)
        fut = idx[i:i+H]; seas_f = prof[fut.weekday, fut.hour]
        drift = drift_k * (float(np.nanmean(u[-1440:])) if len(u) else 0.0)
        log_sig = np.full(n_paths, start); eps_sd = volsd*np.sqrt(1-PHI*PHI)
        rets = np.empty((n_paths, H))
        for j in range(H):
            z = rng.standard_t(nu, size=n_paths)/np.sqrt(nu/(nu-2))
            step = np.exp(log_sig)*seas_f[j]*z + drift
            if jump_p > 0:
                m = rng.random(n_paths) < jump_p
                step = step + m*np.exp(log_sig)*seas_f[j]*jump_sd*rng.standard_normal(n_paths)
            rets[:, j] = step
            log_sig = mu + PHI*(log_sig-mu) + eps_sd*rng.standard_normal(n_paths)
        real_cum = np.concatenate([[0.0], np.nan_to_num(np.cumsum(np.where(np.isfinite(r_all[i:i+H]), r_all[i:i+H], np.nan)))])
        rc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r_all[i:i+H]))])
        sim_cum = np.concatenate([np.zeros((n_paths,1)), np.cumsum(rets, axis=1)], axis=1)
        s5s, s5r = sim_cum[:, ::5], rc[::5]
        for k in ([1,6,36] if H==1440 else [1,2,5,15]):
            sim_ch = (s5s[:, k:] - s5s[:, :-k])*1e4; real_ch = (s5r[k:] - s5r[:-k])*1e4
            raw = r_all[i:i+H]
            for t in range(sim_ch.shape[1]):
                seg = raw[t*5:(t+k)*5]
                if np.isfinite(seg).all():
                    total += crps_ens(real_ch[t], sim_ch[:, t]); cnt += 1
        total += crps_ens(rc[-1]*1e4, sim_cum[:, -1]*1e4); cnt += 1
    return total/max(cnt,1)

if __name__ == "__main__":
    variants = [("base", {}), ("drift0.5", {"drift_k":0.5}), ("drift1", {"drift_k":1.0}),
                ("jump", {"jump_p":0.002,"jump_sd":3.0}), ("jump+", {"jump_p":0.005,"jump_sd":4.0}),
                ("seaspow0.8", {"seas_pow":0.8}), ("seaspow1.2", {"seas_pow":1.2}),
                ("volsd0.2", {"volsd":0.2}), ("volsd0.5", {"volsd":0.5}), ("nu4", {"nu":4}), ("nu8", {"nu":8})]
    for asset, st in [("BTC","2026-08-20"), ("XAU","2026-08-20")]:
        close = load(asset, st)
        for name, kw in variants:
            print(asset, name, round(run(asset, close, **kw), 3), flush=True)
