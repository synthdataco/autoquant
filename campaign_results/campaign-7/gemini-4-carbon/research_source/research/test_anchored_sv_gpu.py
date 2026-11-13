"""Test Spot-Anchored Stochastic Volatility (h_0 ~ N(0, tau_0^2) growing to tau_inf^2 over k=0..N-1,
exact Heston/Hull-White filtered-spot physics!) vs Stationary h_0 ~ N(0, tau_inf^2) strictly on <= 2026-08-10!
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from synth.validator.moving_average import ASSET_COEFFICIENTS

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "agent"))
from research.fit_seasonality_v1 import CRYPTO_ASSETS, EQUITY_COM_ASSETS  # noqa: E402
from research.train_iter6_gpu_direct_crps import (  # noqa: E402
    _build_asset_dataset_worker,
    make_zi_mask_gpu,
    pack_tensors,
    total_crps_1h_torch,
    total_crps_24h_torch,
)


@torch.no_grad()
def eval_sv_curve(
    va: dict,
    va_gpu: dict,
    is_1h: bool,
    nu: float,
    tau0: float,
    tau_inf: float,
    hl_vov: float,
    c_scale: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    m_val = 350 if is_1h else 260
    v_bs = 128 if is_1h else 48
    n_va = va["feats"].shape[0]
    n_steps = va["n_steps"]
    crps_fn = total_crps_1h_torch if is_1h else total_crps_24h_torch

    gen = torch.Generator(device=device)
    gen.manual_seed(20260810)
    scores = np.empty(n_va, dtype=np.float64)

    phi = math.exp(-math.log(2.0) / hl_vov)
    eta = tau_inf * math.sqrt(max(1.0 - phi * phi, 1e-6))

    for i0 in range(0, n_va, v_bs):
        i1 = min(i0 + v_bs, n_va)
        bs = i1 - i0
        half = m_val // 2

        g_h = torch.randn((bs, half, n_steps), device=device, generator=gen)
        conc = torch.full((bs, half, n_steps), nu * 0.5, device=device)
        chi_h = torch._standard_gamma(conc, generator=gen) / 0.5
        t_h = torch.clamp(g_h * math.sqrt(nu - 2.0) / torch.sqrt(torch.clamp(chi_h, min=1e-6)), -10.0, 10.0)
        z_drv = torch.cat([torch.clamp(g_h, -4.0, 4.0), -torch.clamp(g_h, -4.0, 4.0)], dim=1)
        t_mat = torch.cat([t_h, -t_h], dim=1)
        t_mat = t_mat / torch.clamp(t_mat.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

        xi_h = torch.randn((bs, half, n_steps), device=device, generator=gen)
        xi = torch.cat([xi_h, -xi_h], dim=1)
        xi = xi / torch.clamp(xi.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

        h0_h = torch.randn((bs, half), device=device, generator=gen)
        h = torch.cat([h0_h, -h0_h], dim=1)
        h = h * (tau0 / torch.clamp(h.std(dim=1, unbiased=False, keepdim=True), min=1e-6))

        rho = va_gpu["rho_lev"][i0:i1].view(bs, 1)
        s_rho = torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))

        m_cols = []
        for k in range(n_steps):
            if k > 0:
                h = phi * h + eta * (rho * z_drv[:, :, k - 1] + s_rho * xi[:, :, k])
            mc = torch.exp(torch.clamp(h, -2.4, 2.4))
            mc = mc / torch.sqrt(torch.clamp(torch.mean(mc * mc, dim=1, keepdim=True), min=1e-8))
            m_cols.append(mc)
        m_mat = torch.stack(m_cols, dim=2)
        z = (c_scale * m_mat) * t_mat

        zi_m = make_zi_mask_gpu(va_gpu["p_zero"][i0:i1], va_gpu["is_wk_step"][i0:i1], m_val, gen)
        if zi_m is not None:
            zm = z * zi_m
            z = zm / torch.sqrt(torch.clamp(torch.mean(zm * zm, dim=1, keepdim=True), min=1e-8))

        k_idx = torch.arange(n_steps, device=device, dtype=torch.float32).view(1, n_steps)
        decay = torch.exp(-math.log(2.0) * k_idx / va_gpu["hl_base"][i0:i1].unsqueeze(1))
        v_des = torch.maximum(
            va_gpu["v_long"][i0:i1].unsqueeze(1) + (va_gpu["v_short"][i0:i1] - va_gpu["v_long"][i0:i1]).unsqueeze(1) * decay,
            0.04 * va_gpu["prior_v"][i0:i1].unsqueeze(1),
        )
        sigma_log = torch.sqrt(v_des * va_gpu["fut_seas_var"][i0:i1]) * 1e-4

        step_kap = va_gpu["step_kappas"][i0:i1]
        if not bool((step_kap > 0.0).any().item()):
            cum_y = torch.cumsum(z * sigma_log.unsqueeze(1), dim=2)
        else:
            curr = torch.zeros((bs, m_val), device=device)
            anchor = torch.zeros((bs, m_val), device=device)
            in_wk = torch.zeros((bs, 1), device=device, dtype=torch.bool)
            cols = []
            for k in range(n_steps):
                kap = step_kap[:, k : k + 1]
                act = kap > 0.0
                anchor = torch.where(act & (~in_wk), curr, anchor)
                in_wk = act
                eps_ou = z[:, :, k] * (sigma_log[:, k : k + 1] * torch.sqrt(torch.clamp(1.0 - 0.5 * kap, min=0.2)))
                curr = torch.where(act, anchor + (1.0 - kap) * (curr - anchor) + eps_ou, curr + z[:, :, k] * sigma_log[:, k : k + 1])
                cols.append(curr)
            cum_y = torch.stack(cols, dim=2)

        exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
        p0 = va_gpu["p0"][i0:i1]
        p_fut = exp_y * (p0.view(bs, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
        paths = torch.cat([p0.view(bs, 1, 1).expand(bs, m_val, 1), p_fut], dim=2)
        scores[i0:i1] = crps_fn(paths, va_gpu["real_prices"][i0:i1]).cpu().numpy()

    per_a = {}
    a_means, a_coefs = [], []
    for a in sorted(set(va["assets"].tolist())):
        m_a = va["assets"] == a
        val_a = float(np.mean(scores[m_a]))
        per_a[a] = val_a
        a_means.append(val_a)
        a_coefs.append(ASSET_COEFFICIENTS.get(a, 1.0))
    return float(np.average(a_means, weights=a_coefs)), per_a


def main():
    device = torch.device("cuda")
    tasks_1h = [(a, "2026-05-15", "2026-06-22T00:00:00Z", "2026-08-10T22:00:00Z", "2h", 60, 3600) for a in CRYPTO_ASSETS]
    tasks_c24 = [(a, "2026-05-15", "2026-06-22T00:00:00Z", "2026-08-09T23:00:00Z", "4h", 300, 86400) for a in CRYPTO_ASSETS]
    tasks_eq24 = [(a, "2026-05-15" if a != "SPCX" else "2026-06-01", "2026-06-27T00:00:00Z", "2026-08-09T23:00:00Z", "3h", 300, 86400) for a in EQUITY_COM_ASSETS]

    with ProcessPoolExecutor(max_workers=12) as ex:
        r_1h = [r for sub in ex.map(_build_asset_dataset_worker, tasks_1h) for r in sub]
        r_c24 = [r for sub in ex.map(_build_asset_dataset_worker, tasks_c24) for r in sub]
        r_eq24 = [r for sub in ex.map(_build_asset_dataset_worker, tasks_eq24) for r in sub]

    for label, rows, tinc, grid_cfgs in (
        (
            "CRYPTO_1H",
            r_1h,
            60,
            [
                ("Iter5 Stationary (tau0=0.40, t_inf=0.40, hl=25)", 9.0, 0.40, 0.40, 25.0, 1.005),
                ("Anchored A (tau0=0.20, t_inf=0.46, hl=20)", 9.0, 0.20, 0.46, 20.0, 1.005),
                ("Anchored B (tau0=0.22, t_inf=0.50, hl=18)", 9.0, 0.22, 0.50, 18.0, 1.005),
                ("Anchored C (tau0=0.16, t_inf=0.48, hl=15)", 10.0, 0.16, 0.48, 15.0, 1.005),
                ("Anchored D (tau0=0.25, t_inf=0.48, hl=22)", 8.0, 0.25, 0.48, 22.0, 1.005),
                ("HigherStat (tau0=0.44, t_inf=0.44, hl=25)", 9.0, 0.44, 0.44, 25.0, 1.005),
            ],
        ),
        (
            "CRYPTO_24H",
            r_c24,
            300,
            [
                ("Iter5 Stationary (tau0=0.34, t_inf=0.34, hl=48)", 6.0, 0.34, 0.34, 48.0, 1.000),
                ("Anchored A (tau0=0.16, t_inf=0.38, hl=36)", 6.0, 0.16, 0.38, 36.0, 1.000),
                ("Anchored B (tau0=0.18, t_inf=0.42, hl=36)", 6.5, 0.18, 0.42, 36.0, 1.000),
                ("Anchored C (tau0=0.22, t_inf=0.38, hl=42)", 6.0, 0.22, 0.38, 42.0, 1.000),
            ],
        ),
        (
            "COM_EQU_24H",
            r_eq24,
            300,
            [
                ("Iter5 Stationary (tau0=0.30, t_inf=0.30, hl=48)", 6.0, 0.30, 0.30, 48.0, 1.000),
                ("Anchored A (tau0=0.15, t_inf=0.35, hl=36)", 6.0, 0.15, 0.35, 36.0, 1.000),
                ("Anchored B (tau0=0.18, t_inf=0.38, hl=36)", 6.5, 0.18, 0.38, 36.0, 1.000),
                ("Anchored C (tau0=0.20, t_inf=0.34, hl=42)", 6.0, 0.20, 0.34, 42.0, 1.000),
            ],
        ),
    ):
        va = pack_tensors(rows, tinc)
        va_gpu = {
            k: torch.tensor(va[k], device=device)
            for k in ("v_short", "v_long", "prior_v", "hl_base", "fut_seas_var", "step_kappas", "p0", "real_prices", "p_zero", "is_wk_step", "rho_lev")
        }
        print(f"\n=== {label} SPOT-ANCHORED vs STATIONARY STOCH-VOL (<= 2026-08-10, n={len(rows)}) ===")
        base_w = None
        base_pa = None
        for name, nu, t0, tinf, hl, csc in grid_cfgs:
            w, pa = eval_sv_curve(va, va_gpu, tinc <= 60, nu, t0, tinf, hl, csc, device)
            if base_w is None:
                base_w = w
                base_pa = pa
                print(f"  {name:<50}: {w:9.3f} (BASE)")
            else:
                wins = sum(1 for a in pa if pa[a] < base_pa[a])
                print(f"  {name:<50}: {w:9.3f} ({(w/base_w-1)*100:+6.3f}%, wins={wins}/{len(pa)})")


if __name__ == "__main__":
    main()
