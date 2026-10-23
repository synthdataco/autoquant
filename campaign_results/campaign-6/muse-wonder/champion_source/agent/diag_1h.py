"""1h: decompose score into price-CRPS vs vol-CRPS share using validator functions directly,
and test whether fan block-RV dispersion matches realised (per block size)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
from pathlib import Path
import modeling
WS = Path(".").resolve()
from synth_lib.benchmark.generate_predictions import load_minute_prices, context_end
from synth.validator.crps_calculation import calculate_total_score_for_miner, calculate_crps_for_miner, calculate_vol_crps_for_miner
from synth.validator.competition_config import CRYPTO_1H

def run(asset, starts):
    t0 = min(starts) - pd.Timedelta(days=8); t1 = max(starts) + pd.Timedelta(days=2)
    prices = load_minute_prices(WS/"market_data", asset, t0, t1)
    tp, tv, tt = [], [], []
    for st in starts:
        ctx = prices.loc[:context_end(st, 3600)][-7*24*60:]
        out = modeling.simulate(asset, st.isoformat(), 60, 3600, 400, ctx)
        fan = np.array(out[2:])
        fut = prices.loc[st: st + pd.Timedelta(hours=1), "close"]
        real = fut.resample("1min").last()
        real = np.concatenate([[fan[0,0]], real.values[:60]])
        if len(real) < 61 or np.isnan(real).all(): continue
        ps, _ = calculate_crps_for_miner(fan, real, 60, CRYPTO_1H.scoring_intervals)
        vs, _ = calculate_vol_crps_for_miner(fan, real, 60, CRYPTO_1H.vol_scoring_blocks)
        if ps > 0: tp.append(ps); tv.append(vs); tt.append(ps+vs)
    print(asset, f"n={len(tp)} price={np.mean(tp):.1f} vol={np.mean(tv):.1f} volshare={np.mean(tv)/np.mean(tt):.2f}", flush=True)

if __name__ == "__main__":
    starts = list(pd.date_range("2026-09-20", "2026-09-27", freq="6h", tz="UTC"))
    for a in ["BTC","ETH","SOL"]:
        run(a, starts)
