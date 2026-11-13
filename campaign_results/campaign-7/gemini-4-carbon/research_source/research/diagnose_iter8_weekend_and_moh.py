"""Iteration 8 fast pre-2026-08-10 diagnostic of remaining residuals:
1. Inspect `moh_{asset}` & `week1m_{asset}` around 09:30 NY and Weekends (`<= 2026-08-10`).
2. Test CRYPTO_1H Weekend vs Weekday scale (`c_wknd`) & `tau_vov_wknd` & un-smeared 09:30 NY `week1m` on `<= 2026-08-10` (8,490 prompts, M=1000).
3. Test COM_EQU_24H Saturday/Weekend dispersion (`wknd_sigma_mult`, `tau_vov_wknd`, `wknd_ou_kappa` multiplier) & `WTIOIL` on `<= 2026-08-10` (3,384 prompts, M=1000, `mlpeq24` + `LHS_ALL` active!).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from agent import modeling
from research.check_eq24_lhs_gpu import _extract_eq24_worker
from research.train_iter6_gpu_direct_crps import (
    ASSET_COEFFICIENTS,
    CRYPTO_ASSETS,
    EQUITY_COM_ASSETS,
    make_zi_mask_gpu,
    total_crps_1h_torch,
    total_crps_24h_torch,
)
from research.tune_iter7_joint_gpu import gen_shocks_gpu_v2


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut_cache: dict = {}

    calib = modeling._load_calibration()
    arrays = calib["arrays"]
    print("=== 1. INSPECT moh_BTC & week1m_BTC (<= 2026-08-10) ===")
    moh_btc = arrays["moh_BTC"]
    print("  moh_BTC top minutes:", [(m, round(float(moh_btc[m]), 4)) for m in [0, 1, 2, 15, 29, 30, 31, 45, 59]])
    w1_btc = arrays["week1m_BTC"]
    print("  week1m_BTC Mon slots 111..117 (09:15..09:45 NY):", [round(float(x), 3) for x in w1_btc[0, 111:118]])
    print("  week1m_BTC mean RMS by DOW (Mon..Sun):", [round(float(np.sqrt(np.mean(w1_btc[d]**2))), 3) for d in range(7)])

    # Load cached 8,490 CRYPTO_1H prompts (<= 2026-08-10)
    cache_pkl = Path("/root/.gemini/antigravity-cli/brain/21e1ac75-e1b1-4477-a35e-c76db6ee850d/scratch/iter7_diag_cache.pkl")
    with open(cache_pkl, "rb") as f:
        res_1h, _ = pickle.load(f)
    rows_1h = [r for _, _, sub in res_1h for r in sub]

    p0_1h = torch.tensor(np.array([r["p0"] for r in rows_1h], dtype=np.float32), device=device)
    rp_1h = torch.tensor(np.stack([r["real_prices"] for r in rows_1h], axis=0), device=device)
    sig1h = torch.tensor(np.stack([r["sig_gk35"] for r in rows_1h], axis=0), device=device) * 1e-4
    is_wk_1h = np.array([r["is_wknd_prompt"] for r in rows_1h], dtype=bool)
    assets_1h = np.array([r["asset"] for r in rows_1h])
    ts_1h = np.array([r["start_ts"] for r in rows_1h])
    cut_mid = int(pd.Timestamp("2026-04-15T00:00:00Z").timestamp())

    # Check realized/predicted RMS ratio on 1H Weekends vs Weekdays (<= 2026-08-10)!
    rp_np = np.stack([r["real_prices"] for r in rows_1h], axis=0)
    r1_real = (np.log(rp_np[:, 1:]) - np.log(rp_np[:, :-1])) * 1e4
    sig_np = np.stack([r["sig_gk35"] for r in rows_1h], axis=0)
    z_real = np.clip(r1_real / np.maximum(sig_np, 0.1), -8.0, 8.0)
    print("\n=== 2. CRYPTO_1H STANDARDIZED RESIDUAL RMS (|r1|/sig & sqrt(E[z^2])) BY WEEKEND VS WEEKDAY (<= 2026-08-10) ===")
    for a in CRYPTO_ASSETS:
        m_wd = (assets_1h == a) & (~is_wk_1h)
        m_wk = (assets_1h == a) & is_wk_1h
        rms_wd = float(np.sqrt(np.mean(z_real[m_wd] ** 2)))
        rms_wk = float(np.sqrt(np.mean(z_real[m_wk] ** 2)))
        mad_wd = float(np.mean(np.abs(z_real[m_wd])))
        mad_wk = float(np.mean(np.abs(z_real[m_wk])))
        # Also block vol std ratio!
        sd_real_wd = float(np.mean(np.std(r1_real[m_wd], axis=1, ddof=1) / np.sqrt(np.mean(sig_np[m_wd]**2, axis=1))))
        sd_real_wk = float(np.mean(np.std(r1_real[m_wk], axis=1, ddof=1) / np.sqrt(np.mean(sig_np[m_wk]**2, axis=1))))
        print(f"  {a:<5} | Wkdy: rms={rms_wd:.3f}, mad={mad_wd:.3f}, sd60_ratio={sd_real_wd:.3f} | Wknd: rms={rms_wk:.3f}, mad={mad_wk:.3f}, sd60_ratio={sd_real_wk:.3f}")

    @torch.no_grad()
    def eval_1h_wk_mod(wk_scale: float, wk_tau: float, wk_nu: float) -> np.ndarray:
        n = len(rows_1h)
        out = np.empty(n, dtype=np.float64)
        bs = 50
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            wk_b = is_wk_1h[i0:i1]
            sc_vec = torch.tensor(np.where(wk_b, wk_scale, 1.0), dtype=torch.float32, device=device).view(-1, 1)
            tau_vec = torch.tensor(np.where(wk_b, wk_tau, 0.32), dtype=torch.float32, device=device)
            # Use wk_nu only if all equal else blend; test uniform nu or split by batch
            z_wd = gen_shocks_gpu_v2(i1 - i0, 1000, 60, 12.0, tau_vec, 25.0, -0.18, 1.005, device, 777000 + i0, mode="lhs_all", lut_cache=lut_cache)
            if abs(wk_nu - 12.0) > 1e-3:
                z_wk = gen_shocks_gpu_v2(i1 - i0, 1000, 60, wk_nu, tau_vec, 25.0, -0.18, 1.005, device, 777000 + i0, mode="lhs_all", lut_cache=lut_cache)
                m_t = torch.tensor(wk_b, device=device).view(-1, 1, 1)
                z = torch.where(m_t, z_wk, z_wd)
            else:
                z = z_wd
            cum_y = torch.cumsum(z * (sig1h[i0:i1] * sc_vec).unsqueeze(1), dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_1h[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_1h[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_1h_torch(paths, rp_1h[i0:i1]).cpu().numpy()
        return out

    sc1_base = eval_1h_wk_mod(1.0, 0.32, 12.0)
    print("\n=== 3. CRYPTO_1H WEEKEND SCALE / TAU / NU TESTS ON <= 2026-08-10 ===")
    for ws, wt, wn in (
        (0.95, 0.32, 12.0),
        (0.97, 0.32, 12.0),
        (1.03, 0.32, 12.0),
        (1.06, 0.32, 12.0),
        (1.00, 0.38, 9.0),
        (1.00, 0.26, 14.0),
    ):
        sc_c = eval_1h_wk_mod(ws, wt, wn)
        wk_b = float(np.average([np.mean(sc1_base[(assets_1h==a) & is_wk_1h]) for a in CRYPTO_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in CRYPTO_ASSETS]))
        wk_c = float(np.average([np.mean(sc_c[(assets_1h==a) & is_wk_1h]) for a in CRYPTO_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in CRYPTO_ASSETS]))
        wins = sum(np.mean(sc_c[(assets_1h==a) & is_wk_1h]) < np.mean(sc1_base[(assets_1h==a) & is_wk_1h]) for a in CRYPTO_ASSETS)
        print(f"  wk_scale={ws:.2f}, wk_tau={wt:.2f}, wk_nu={wn:4.1f} | Wknd CRPS: {wk_b:.3f} -> {wk_c:.3f} ({(wk_c/wk_b-1)*100:+.3f}%, {wins}/5)")

    # 4. Now inspect COM_EQU_24H Saturday & WTIOIL residuals on <= 2026-08-10!
    eq_starts = {
        "NVDAX": ("2026-02-01", "2026-02-20"),
        "TSLAX": ("2026-02-01", "2026-02-20"),
        "GOOGLX": ("2026-02-01", "2026-02-20"),
        "AAPLX": ("2026-02-15", "2026-03-05"),
        "XAU": ("2026-03-01", "2026-03-20"),
        "WTIOIL": ("2026-03-01", "2026-03-20"),
        "SP500": ("2026-04-01", "2026-04-20"),
        "SPCX": ("2026-06-01", "2026-06-12"),
    }
    tasks_eq = [(a, eq_starts[a][0], eq_starts[a][1], "2026-08-09T23:00:00Z") for a in EQUITY_COM_ASSETS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res_eq = list(ex.map(_extract_eq24_worker, tasks_eq))
    rows_eq = [r for sub in res_eq for r in sub]

    p0_eq = torch.tensor(np.array([r["p0"] for r in rows_eq], dtype=np.float32), device=device)
    rp_eq = torch.tensor(np.stack([r["real_prices"] for r in rows_eq], axis=0), device=device)
    sig_eq = torch.tensor(np.stack([r["sig_bps"] for r in rows_eq], axis=0), device=device) * 1e-4
    kap_eq = torch.tensor(np.stack([r["step_kappas"] for r in rows_eq], axis=0), device=device)
    tau_eq = torch.tensor(np.array([r["tau_eff"] for r in rows_eq], dtype=np.float32), device=device)
    rho_eq = torch.tensor(np.array([r["rho_lev"] for r in rows_eq], dtype=np.float32), device=device)
    pz_eq = torch.tensor(np.array([r["p_zero"] for r in rows_eq], dtype=np.float32), device=device)
    wk_eq = torch.tensor(np.stack([r["is_wk_step"] for r in rows_eq], axis=0), device=device)
    assets_eq = np.array([r["asset"] for r in rows_eq])
    ny_d_eq = np.array([int(modeling._utc_sec_to_ny_parts(np.array([r["start_ts"]], dtype=np.int64))[0][0]) for r in rows_eq])

    @torch.no_grad()
    def eval_eq_mod(wk_sig_mult: float, kap_mult: float, zi_mult: float) -> np.ndarray:
        n = len(rows_eq)
        out = np.empty(n, dtype=np.float64)
        bs = 24
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            b_sz = i1 - i0
            z = gen_shocks_gpu_v2(b_sz, 1000, 288, 6.0, tau_eq[i0:i1], 48.0, rho_eq[i0:i1], 1.000, device, 555000 + i0, mode="lhs_all", lut_cache=lut_cache)
            gen_zi = torch.Generator(device=device)
            gen_zi.manual_seed(605000 + i0)
            pz_m = torch.clamp(pz_eq[i0:i1] * zi_mult, 0.0, 0.70)
            zi_m = make_zi_mask_gpu(pz_m, wk_eq[i0:i1], 1000, gen_zi)
            if zi_m is not None:
                zm = z * zi_m
                z = zm / torch.sqrt(torch.clamp(torch.mean(zm * zm, dim=1, keepdim=True), min=1e-8))
            sig_m = torch.where(wk_eq[i0:i1], sig_eq[i0:i1] * wk_sig_mult, sig_eq[i0:i1])
            kap_m = torch.clamp(kap_eq[i0:i1] * kap_mult, 0.0, 0.45)
            curr = torch.zeros((b_sz, 1000), device=device)
            anchor = torch.zeros((b_sz, 1000), device=device)
            in_wk = torch.zeros((b_sz, 1), device=device, dtype=torch.bool)
            cols = []
            for k in range(288):
                kap = kap_m[:, k : k + 1]
                act_ou = kap > 0.0
                anchor = torch.where(act_ou & (~in_wk), curr, anchor)
                in_wk = act_ou
                ou_sc = torch.sqrt(torch.clamp(1.0 - 0.5 * kap, min=0.2))
                next_ou = anchor + (1.0 - kap) * (curr - anchor) + z[:, :, k] * (sig_m[:, k : k + 1] * ou_sc)
                next_rw = curr + z[:, :, k] * sig_m[:, k : k + 1]
                curr = torch.where(act_ou, next_ou, next_rw)
                cols.append(curr)
            cum_y = torch.stack(cols, dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_eq[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_eq[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_24h_torch(paths, rp_eq[i0:i1]).cpu().numpy()
        return out

    eq_base = eval_eq_mod(1.0, 1.0, 1.0)
    print("\n=== 4. COM_EQU_24H CLOSED-WEEKEND SIGMA / KAPPA / ZI TESTS ON <= 2026-08-10 ===")
    sat_m = (ny_d_eq == 4) | (ny_d_eq == 5)
    for wsm, km, zm in (
        (0.88, 1.0, 1.0),
        (0.93, 1.0, 1.0),
        (1.07, 1.0, 1.0),
        (1.00, 1.30, 1.0),
        (1.00, 0.75, 1.0),
        (0.92, 1.25, 1.15),
    ):
        eq_c = eval_eq_mod(wsm, km, zm)
        b_all = float(np.average([np.mean(eq_base[assets_eq == a]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        c_all = float(np.average([np.mean(eq_c[assets_eq == a]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        b_sat = float(np.average([np.mean(eq_base[(assets_eq == a) & sat_m]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        c_sat = float(np.average([np.mean(eq_c[(assets_eq == a) & sat_m]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        wins = sum(np.mean(eq_c[assets_eq == a]) < np.mean(eq_base[assets_eq == a]) for a in EQUITY_COM_ASSETS)
        print(f"  wk_sig={wsm:.2f}, kap_m={km:.2f}, zi_m={zm:.2f} | All: {b_all:.3f}->{c_all:.3f} ({(c_all/b_all-1)*100:+.3f}%, {wins}/8) | Fri+Sat: {(c_sat/b_sat-1)*100:+.3f}%")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
