"""Iteration 7 Stage-2 Joint GPU Sweep at M=1000 paths strictly on <= 2026-08-10.

Uses cached `iter7_diag_cache.pkl` (8,490 CRYPTO_1H prompts + 5,640 CRYPTO_24H prompts across Nov 2025 - Aug 10, 2026)
plus COM_EQU_24H prompts (with `mlpeq24` active!) to jointly evaluate:
  1. Causal 30d Scale-Matched Garman-Klass + Close-to-Close hybrid (`w_gk = 0.35` on 1H, `w_gk = 0.25/0.30` on C24).
  2. Joint `(nu, tau_vov, hl_vov, c_scale)` grid at M=1000 on CRYPTO_1H and CRYPTO_24H, requiring consistency across
     BOTH H1 (Nov 2025 - Apr 15, 2026) and H2 (Apr 15 - Aug 10, 2026) and >= 4/5 assets!
  3. Randomized LHS Stratified Quantiles (`t_mat + h0` vs `t_mat + h0 + xi`) across all 3 competitions (including COM_EQU_24H with `mlpeq24`!).
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
from scipy.stats import norm, t as student_t
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


@torch.no_grad()
def gen_shocks_gpu_v2(
    b_sz: int,
    m_paths: int,
    n_steps: int,
    nu: float,
    tau_vov_vec: torch.Tensor | float,
    hl_vov: float,
    rho_lev_vec: torch.Tensor | float,
    c_scale: float,
    device: torch.device,
    seed: int,
    mode: str = "prng",
    lut_cache: dict | None = None,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    half = m_paths // 2

    if isinstance(tau_vov_vec, (int, float)):
        tau_t = torch.full((b_sz, 1), float(tau_vov_vec), device=device)
    else:
        tau_t = tau_vov_vec.view(b_sz, 1)

    if isinstance(rho_lev_vec, (int, float)):
        rho_t = torch.full((b_sz, 1), float(rho_lev_vec), device=device)
    else:
        rho_t = rho_lev_vec.view(b_sz, 1)

    if mode == "prng":
        g_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)
        rate = torch.full((b_sz, half, n_steps), 0.5, device=device)
        conc = torch.full((b_sz, half, n_steps), nu * 0.5, device=device)
        chi_h = torch._standard_gamma(conc, generator=gen) / rate
        t_h = torch.clamp(g_h * math.sqrt(nu - 2.0) / torch.sqrt(torch.clamp(chi_h, min=1e-6)), -10.0, 10.0)
        z_drv_h = torch.clamp(g_h, -4.0, 4.0)
        h0_u = torch.randn((b_sz, half), device=device, generator=gen)
        xi_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)
    else:
        grid_n = 4096
        if lut_cache is not None and nu in lut_cache:
            t_lut, z_lut = lut_cache[nu]
        else:
            p_grid = 0.5 + 0.5 * (np.arange(grid_n, dtype=np.float64) + 0.5) / grid_n
            t_lut = torch.tensor(np.clip(student_t.ppf(p_grid, df=nu), 0.0, 10.0), dtype=torch.float32, device=device)
            z_lut = torch.tensor(np.clip(norm.ppf(p_grid), 0.0, 4.0), dtype=torch.float32, device=device)
            if lut_cache is not None:
                lut_cache[nu] = (t_lut, z_lut)

        strata = torch.arange(half, device=device, dtype=torch.float32).view(1, half, 1)
        u_pos = (strata + torch.rand((b_sz, half, n_steps), device=device, generator=gen)) / float(half)
        idx_f = torch.clamp(u_pos * grid_n - 0.5, 0.0, float(grid_n - 2))
        i0 = torch.clamp(idx_f.to(torch.long), 0, grid_n - 2)
        frac = idx_f - i0.to(torch.float32)
        t_mag = t_lut[i0] * (1.0 - frac) + t_lut[i0 + 1] * frac
        z_mag = z_lut[i0] * (1.0 - frac) + z_lut[i0 + 1] * frac

        signs = torch.where(torch.rand((b_sz, half, n_steps), device=device, generator=gen) < 0.5, -1.0, 1.0)
        perm = torch.argsort(torch.rand((b_sz, half, n_steps), device=device, generator=gen), dim=1)
        t_h = torch.gather(t_mag * signs, 1, perm)
        z_drv_h = torch.gather(z_mag * signs, 1, perm)

        u_h0 = (torch.arange(half, device=device, dtype=torch.float32).view(1, half) + torch.rand((b_sz, half), device=device, generator=gen)) / float(half)
        idx_h0 = torch.clamp(u_h0 * grid_n - 0.5, 0.0, float(grid_n - 2))
        ih0 = torch.clamp(idx_h0.to(torch.long), 0, grid_n - 2)
        fh0 = idx_h0 - ih0.to(torch.float32)
        h0_mag = z_lut[ih0] * (1.0 - fh0) + z_lut[ih0 + 1] * fh0
        s_h0 = torch.where(torch.rand((b_sz, half), device=device, generator=gen) < 0.5, -1.0, 1.0)
        p_h0 = torch.argsort(torch.rand((b_sz, half), device=device, generator=gen), dim=1)
        h0_u = torch.gather(h0_mag * s_h0, 1, p_h0)

        if mode == "lhs_all":
            u_xi = (strata + torch.rand((b_sz, half, n_steps), device=device, generator=gen)) / float(half)
            ix_f = torch.clamp(u_xi * grid_n - 0.5, 0.0, float(grid_n - 2))
            ix0 = torch.clamp(ix_f.to(torch.long), 0, grid_n - 2)
            fx = ix_f - ix0.to(torch.float32)
            xi_mag = z_lut[ix0] * (1.0 - fx) + z_lut[ix0 + 1] * fx
            s_xi = torch.where(torch.rand((b_sz, half, n_steps), device=device, generator=gen) < 0.5, -1.0, 1.0)
            p_xi = torch.argsort(torch.rand((b_sz, half, n_steps), device=device, generator=gen), dim=1)
            xi_h = torch.gather(xi_mag * s_xi, 1, p_xi)
        else:
            xi_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)

    t_mat = torch.cat([t_h, -t_h], dim=1)
    z_drv = torch.cat([z_drv_h, -z_drv_h], dim=1)
    t_mat = t_mat / torch.clamp(t_mat.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    xi = torch.cat([xi_h, -xi_h], dim=1)
    xi = xi / torch.clamp(xi.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    h = torch.cat([h0_u, -h0_u], dim=1)
    h = h * (tau_t / torch.clamp(h.std(dim=1, unbiased=False, keepdim=True), min=1e-6))

    phi = math.exp(-math.log(2.0) / hl_vov)
    eta = tau_t * math.sqrt(max(1.0 - phi * phi, 1e-6))
    s_rho = torch.sqrt(torch.clamp(1.0 - rho_t * rho_t, min=0.0))

    m_cols = []
    for k in range(n_steps):
        if k > 0:
            h = phi * h + eta * (rho_t * z_drv[:, :, k - 1] + s_rho * xi[:, :, k])
        m_c = torch.exp(torch.clamp(h, -2.2, 2.2))
        m_c = m_c / torch.sqrt(torch.clamp(torch.mean(m_c * m_c, dim=1, keepdim=True), min=1e-8))
        m_cols.append(m_c)
    m_mat = torch.stack(m_cols, dim=2)
    return (c_scale * m_mat) * t_mat


def _extract_eq24_worker(args: tuple) -> list[dict]:
    asset, start_fit, prompt_start_iso, prompt_end_iso = args
    arr = load_asset_arrays(asset, start_fit, "2026-08-10")
    ts_all, c_all = arr["ts_sec"], arr["close"]
    o_all, h_all, l_all, v_all, tc_all = arr["open"], arr["high"], arr["low"], arr["volume"], arr["trade_count"]

    t_s = pd.Timestamp(prompt_start_iso).tz_localize("UTC")
    t_e = pd.Timestamp(prompt_end_iso).tz_localize("UTC")
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
        res_1h, rows_c24 = pickle.load(f)
    rows_1h = [r for _, _, sub in res_1h for r in sub]
    print(f"Loaded cached 1H={len(rows_1h)}, C24={len(rows_c24)} in {time.time()-t0:.1f}s")

    # Pre-load 1H tensors onto GPU
    p0_1h = torch.tensor(np.array([r["p0"] for r in rows_1h], dtype=np.float32), device=device)
    rp_1h = torch.tensor(np.stack([r["real_prices"] for r in rows_1h], axis=0), device=device)
    sig1h_base = torch.tensor(np.stack([r["sig_base"] for r in rows_1h], axis=0), device=device) * 1e-4
    sig1h_gk35 = torch.tensor(np.stack([r["sig_gk35"] for r in rows_1h], axis=0), device=device) * 1e-4

    @torch.no_grad()
    def run_1h(sig_t: torch.Tensor, nu: float, tau: float, hl: float, cs: float, mode: str = "prng", seed_b: int = 777000) -> np.ndarray:
        n = len(rows_1h)
        out = np.empty(n, dtype=np.float64)
        bs = 50
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            z = gen_shocks_gpu_v2(i1 - i0, 1000, 60, nu, tau, hl, -0.18, cs, device, seed_b + i0, mode=mode, lut_cache=lut_cache)
            cum_y = torch.cumsum(z * sig_t[i0:i1].unsqueeze(1), dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_1h[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_1h[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_1h_torch(paths, rp_1h[i0:i1]).cpu().numpy()
        return out

    print("\n=== STAGE 2A: CRYPTO_1H JOINT (GK35 + nu + tau_vov + hl_vov + c_scale) AT M=1000 (<= 2026-08-10) ===")
    sc_base = run_1h(sig1h_base, 9.0, 0.40, 25.0, 1.005, mode="prng")
    assets_1h = np.array([r["asset"] for r in rows_1h])

    # Fine sweep around (tau_vov in [0.26, 0.30, 0.33, 0.36], nu in [9.0, 10.5, 12.0], c_scale in [1.005, 1.015, 1.022], hl in [20, 25, 32])
    candidates_1h = [
        (9.0, 0.36, 25.0, 1.005),
        (9.0, 0.33, 25.0, 1.005),
        (9.0, 0.30, 25.0, 1.005),
        (9.0, 0.27, 25.0, 1.005),
        (10.5, 0.33, 25.0, 1.005),
        (10.5, 0.30, 25.0, 1.005),
        (12.0, 0.32, 25.0, 1.005),
        (10.5, 0.32, 25.0, 1.014),
        (10.5, 0.32, 25.0, 1.020),
        (10.5, 0.30, 25.0, 1.018),
        (10.5, 0.28, 25.0, 1.018),
        (12.0, 0.30, 25.0, 1.018),
        (10.5, 0.30, 20.0, 1.018),
        (10.5, 0.30, 32.0, 1.018),
    ]
    best_cfg = (9.0, 0.40, 25.0, 1.005)
    best_sc = sc_base
    best_mean = 1e9
    for nu_c, tau_c, hl_c, cs_c in candidates_1h:
        sc_c = run_1h(sig1h_gk35, nu_c, tau_c, hl_c, cs_c, mode="prng")
        lbl = f"GK35+nu={nu_c},t={tau_c},hl={hl_c:.0f},c={cs_c}"
        summarize_comp(rows_1h, sc_base, sc_c, lbl)
        per_a = " ".join(f"{a}:{(np.mean(sc_c[assets_1h==a])/np.mean(sc_base[assets_1h==a])-1)*100:+.3f}%" for a in CRYPTO_ASSETS)
        print(f"      per-asset: {per_a}")
        w_m = float(np.average([np.mean(sc_c[assets_1h == a]) for a in CRYPTO_ASSETS], weights=[ASSET_COEFFICIENTS[a] for a in CRYPTO_ASSETS]))
        if w_m < best_mean:
            best_mean = w_m
            best_cfg = (nu_c, tau_c, hl_c, cs_c)
            best_sc = sc_c

    print(f"\n  --> Best 1H PRNG config: {best_cfg} | Now testing +LHS vs +LHS_ALL (2-seed avg):")
    sc_base_2 = run_1h(sig1h_base, 9.0, 0.40, 25.0, 1.005, mode="prng", seed_b=999000)
    sc_best_prng2 = run_1h(sig1h_gk35, *best_cfg, mode="prng", seed_b=999000)
    sc_best_lhs1 = run_1h(sig1h_gk35, *best_cfg, mode="lhs", seed_b=777000)
    sc_best_lhs2 = run_1h(sig1h_gk35, *best_cfg, mode="lhs", seed_b=999000)
    sc_best_lhsa1 = run_1h(sig1h_gk35, *best_cfg, mode="lhs_all", seed_b=777000)
    sc_best_lhsa2 = run_1h(sig1h_gk35, *best_cfg, mode="lhs_all", seed_b=999000)
    summarize_comp(rows_1h, 0.5 * (sc_base + sc_base_2), 0.5 * (best_sc + sc_best_prng2), "Best 1H (PRNG 2-seed)")
    summarize_comp(rows_1h, 0.5 * (sc_base + sc_base_2), 0.5 * (sc_best_lhs1 + sc_best_lhs2), "Best 1H + LHS(t,h0) (2-seed)")
    summarize_comp(rows_1h, 0.5 * (sc_base + sc_base_2), 0.5 * (sc_best_lhsa1 + sc_best_lhsa2), "Best 1H + LHS_ALL (2-seed)")

    # Stage 2B: CRYPTO_24H Joint Sweep
    p0_24 = torch.tensor(np.array([r["p0"] for r in rows_c24], dtype=np.float32), device=device)
    rp_24 = torch.tensor(np.stack([r["real_prices"] for r in rows_c24], axis=0), device=device)
    sig24_base = torch.tensor(np.stack([r["sig_base"] for r in rows_c24], axis=0), device=device) * 1e-4
    sig24_gk30 = torch.tensor(np.stack([r["sig_gk30"] for r in rows_c24], axis=0), device=device) * 1e-4

    @torch.no_grad()
    def run_c24(sig_t: torch.Tensor, nu: float, tau: float, cs: float, mode: str = "prng", seed_b: int = 888000) -> np.ndarray:
        n = len(rows_c24)
        out = np.empty(n, dtype=np.float64)
        bs = 24
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            z = gen_shocks_gpu_v2(i1 - i0, 1000, 288, nu, tau, 48.0, -0.18, cs, device, seed_b + i0, mode=mode, lut_cache=lut_cache)
            cum_y = torch.cumsum(z * sig_t[i0:i1].unsqueeze(1), dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_24[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_24[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = total_crps_24h_torch(paths, rp_24[i0:i1]).cpu().numpy()
        return out

    print("\n=== STAGE 2B: CRYPTO_24H JOINT (GK30 + nu + tau_vov + c_scale + LHS) AT M=1000 (<= 2026-08-10) ===")
    sc24_b1 = run_c24(sig24_base, 6.0, 0.34, 1.000, mode="prng", seed_b=888000)
    sc24_b2 = run_c24(sig24_base, 6.0, 0.34, 1.000, mode="prng", seed_b=988000)
    sc24_b = 0.5 * (sc24_b1 + sc24_b2)
    for nu_c, tau_c, cs_c, md in (
        (6.0, 0.34, 1.000, "lhs"),
        (6.0, 0.34, 1.000, "lhs_all"),
        (6.0, 0.30, 1.000, "lhs"),
        (6.0, 0.30, 1.008, "lhs"),
        (7.0, 0.30, 1.008, "lhs"),
        (7.0, 0.32, 1.005, "lhs"),
        (6.0, 0.34, 1.008, "lhs"),
    ):
        s1 = run_c24(sig24_gk30, nu_c, tau_c, cs_c, mode=md, seed_b=888000)
        s2 = run_c24(sig24_gk30, nu_c, tau_c, cs_c, mode=md, seed_b=988000)
        summarize_comp(rows_c24, sc24_b, 0.5 * (s1 + s2), f"GK30+nu={nu_c},t={tau_c},c={cs_c},{md}")

    # Stage 2C: Verify LHS on COM_EQU_24H (WITH mlpeq24 active!) strictly <= 2026-08-10
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
        print(f"  COM_EQU_24H {md:<8} (2-seed) | {b_w:8.3f} -> {c_w:8.3f} ({(c_w/b_w-1)*100:+.3f}%, {wins}/8 assets improved)")

    print(f"\nStage-2 GPU sweep finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
