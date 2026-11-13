"""Quick 30s Stage-2C check: COM_EQU_24H LHS_ALL with mlpeq24 active + nu=11.5..14.0 bowl check on CRYPTO_1H (<= 2026-08-10)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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
from research.diagnose_iter7_pre_aug10 import summarize_comp
from research.train_iter6_gpu_direct_crps import (
    ASSET_COEFFICIENTS,
    CRYPTO_ASSETS,
    EQUITY_COM_ASSETS,
    load_asset_arrays,
    make_zi_mask_gpu,
    total_crps_1h_torch,
    total_crps_24h_torch,
)
from research.tune_iter7_joint_gpu import gen_shocks_gpu_v2


def _extract_eq24_worker(args: tuple) -> list[dict]:
    asset, start_fit, prompt_start_iso, prompt_end_iso = args
    arr = load_asset_arrays(asset, start_fit, "2026-08-10")
    ts_all, c_all = arr["ts_sec"], arr["close"]
    o_all, h_all, l_all, v_all, tc_all = arr["open"], arr["high"], arr["low"], arr["volume"], arr["trade_count"]

    ts0 = pd.Timestamp(prompt_start_iso)
    ts1 = pd.Timestamp(prompt_end_iso)
    t_s = ts0.tz_localize("UTC") if ts0.tzinfo is None else ts0.tz_convert("UTC")
    t_e = ts1.tz_localize("UTC") if ts1.tzinfo is None else ts1.tz_convert("UTC")
    grid = pd.date_range(t_s, t_e, freq="8h")
    rows = []
    for t_pd in grid:
        st = int(t_pd.timestamp())
        target_ts = st + 300 * np.arange(0, 289, dtype=np.int64)
        pos_r = np.searchsorted(ts_all, target_ts)
        if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
            continue
        rp = c_all[pos_r]
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
            continue
        lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
        hi = int(np.searchsorted(ts_all, st - 180, side="right"))
        if hi - lo < 2000:
            continue
        closes = c_all[lo:hi]
        ts_sec = ts_all[lo:hi]
        p0 = float(closes[-1]) if (np.isfinite(closes[-1]) and closes[-1] > 0.0) else float(rp[0])
        ohlcv_extra = (o_all[lo:hi], h_all[lo:hi], l_all[lo:hi], v_all[lo:hi], tc_all[lo:hi])
        sig_bps, step_kappas, vov_mult = modeling._estimate_step_dynamics(
            asset, closes, ts_sec, st, 300, 288, ohlcv_extra=ohlcv_extra, p0=p0, return_vov_mult=True
        )
        zi = modeling._estimate_weekend_zi_probs(asset, closes, ts_sec, st, 300, 288)
        rows.append(
            {
                "asset": asset,
                "start_ts": st,
                "p0": np.float32(p0),
                "real_prices": rp.astype(np.float32),
                "sig_bps": sig_bps.astype(np.float32),
                "step_kappas": step_kappas.astype(np.float32),
                "tau_eff": np.float32(np.clip(0.30 * vov_mult, 0.12, 0.65)),
                "rho_lev": np.float32(-0.18 if asset == "SP500" else (-0.08 if asset in ("XAU", "WTIOIL") else -0.12)),
                "p_zero": np.float32(zi[0] if zi is not None else 0.0),
                "is_wk_step": zi[1] if zi is not None else np.zeros(288, dtype=bool),
            }
        )
    return rows


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut_cache: dict = {}

    cache_pkl = Path("/root/.gemini/antigravity-cli/brain/21e1ac75-e1b1-4477-a35e-c76db6ee850d/scratch/iter7_diag_cache.pkl")
    with open(cache_pkl, "rb") as f:
        res_1h, _ = pickle.load(f)
    rows_1h = [r for _, _, sub in res_1h for r in sub]
    p0_1h = torch.tensor(np.array([r["p0"] for r in rows_1h], dtype=np.float32), device=device)
    rp_1h = torch.tensor(np.stack([r["real_prices"] for r in rows_1h], axis=0), device=device)
    sig1h_base = torch.tensor(np.stack([r["sig_base"] for r in rows_1h], axis=0), device=device) * 1e-4
    sig1h_gk35 = torch.tensor(np.stack([r["sig_gk35"] for r in rows_1h], axis=0), device=device) * 1e-4

    @torch.no_grad()
    def run_1h(sig_t: torch.Tensor, nu: float, tau: float, mode: str, seed_b: int) -> np.ndarray:
        n = len(rows_1h)
        out = np.empty(n, dtype=np.float64)
        bs = 50
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            z = gen_shocks_gpu_v2(i1 - i0, 1000, 60, nu, tau, 25.0, -0.18, 1.005, device, seed_b + i0, mode=mode, lut_cache=lut_cache)
            cum_y = torch.cumsum(z * sig_t[i0:i1].unsqueeze(1), dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_1h[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_1h[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_1h_torch(paths, rp_1h[i0:i1]).cpu().numpy()
        return out

    sc_b = 0.5 * (run_1h(sig1h_base, 9.0, 0.40, "prng", 777000) + run_1h(sig1h_base, 9.0, 0.40, "prng", 999000))
    for nu_c, tau_c in ((11.5, 0.325), (12.0, 0.32), (14.0, 0.32)):
        sc_c = 0.5 * (run_1h(sig1h_gk35, nu_c, tau_c, "lhs_all", 777000) + run_1h(sig1h_gk35, nu_c, tau_c, "lhs_all", 999000))
        summarize_comp(rows_1h, sc_b, sc_c, f"1H GK35+nu={nu_c},t={tau_c},lhs_all")

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

    @torch.no_grad()
    def run_eq(mode: str, seed_b: int) -> np.ndarray:
        n = len(rows_eq)
        out = np.empty(n, dtype=np.float64)
        bs = 24
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            b_sz = i1 - i0
            z = gen_shocks_gpu_v2(b_sz, 1000, 288, 6.0, tau_eq[i0:i1], 48.0, rho_eq[i0:i1], 1.000, device, seed_b + i0, mode=mode, lut_cache=lut_cache)
            gen_zi = torch.Generator(device=device)
            gen_zi.manual_seed(seed_b + 50000 + i0)
            zi_m = make_zi_mask_gpu(pz_eq[i0:i1], wk_eq[i0:i1], 1000, gen_zi)
            if zi_m is not None:
                zm = z * zi_m
                z = zm / torch.sqrt(torch.clamp(torch.mean(zm * zm, dim=1, keepdim=True), min=1e-8))
            curr = torch.zeros((b_sz, 1000), device=device)
            anchor = torch.zeros((b_sz, 1000), device=device)
            in_wk = torch.zeros((b_sz, 1), device=device, dtype=torch.bool)
            cols = []
            for k in range(288):
                kap = kap_eq[i0:i1, k : k + 1]
                act_ou = kap > 0.0
                anchor = torch.where(act_ou & (~in_wk), curr, anchor)
                in_wk = act_ou
                ou_sc = torch.sqrt(torch.clamp(1.0 - 0.5 * kap, min=0.2))
                next_ou = anchor + (1.0 - kap) * (curr - anchor) + z[:, :, k] * (sig_eq[i0:i1, k : k + 1] * ou_sc)
                next_rw = curr + z[:, :, k] * sig_eq[i0:i1, k : k + 1]
                curr = torch.where(act_ou, next_ou, next_rw)
                cols.append(curr)
            cum_y = torch.stack(cols, dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_eq[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_eq[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_24h_torch(paths, rp_eq[i0:i1]).cpu().numpy()
        return out

    print(f"\n=== STAGE 2C: COM_EQU_24H LHS CHECK WITH mlpeq24 ACTIVE (n={len(rows_eq)}, <= 2026-08-10) ===")
    eq_p = 0.5 * (run_eq("prng", 555000) + run_eq("prng", 655000))
    for md in ("lhs", "lhs_all"):
        eq_l = 0.5 * (run_eq(md, 555000) + run_eq(md, 655000))
        b_w = float(np.average([np.mean(eq_p[assets_eq == a]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        c_w = float(np.average([np.mean(eq_l[assets_eq == a]) for a in EQUITY_COM_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in EQUITY_COM_ASSETS]))
        wins = sum(np.mean(eq_l[assets_eq == a]) < np.mean(eq_p[assets_eq == a]) for a in EQUITY_COM_ASSETS)
        per_a = " ".join(f"{a}:{(np.mean(eq_l[assets_eq==a])/np.mean(eq_p[assets_eq==a])-1)*100:+.3f}%" for a in EQUITY_COM_ASSETS)
        print(f"  COM_EQU_24H {md:<8} (2-seed) | {b_w:8.3f} -> {c_w:8.3f} ({(c_w/b_w-1)*100:+.3f}%, {wins}/8) | {per_a}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
