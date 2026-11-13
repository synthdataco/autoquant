"""Calibrate Iteration 3 strictly on <= 2026-08-10 against exact validator CRPS:
- Heavy-tailed Student-t(nu) step innovations
- Intrapath stationary AR(1) log-volatility cascade (tau_vov, hl_vov)
- Return-volatility leverage correlation (rho_lev)
- CRPS-optimal dispersion scale factor (c_scale)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.benchmark.generate_predictions import context_slice, load_minute_prices

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "agent"))
import modeling  # noqa: E402


def simulate_paths_parametric(
    rng: np.random.Generator,
    p0: float,
    sigma_log: np.ndarray,
    step_kappas: np.ndarray,
    num_sim: int,
    nu: float,
    tau_vov: float,
    hl_steps: float,
    rho_lev: float,
    c_scale: float,
) -> np.ndarray:
    """Fast vectorized simulation with Student-t(nu) + stationary AR(1) log-vol + leverage."""
    n_steps = len(sigma_log)
    half = num_sim // 2

    # 1. Antithetic unit-variance Student-t(nu) draws
    if nu < 100.0:
        g_half = rng.standard_normal((half, n_steps))
        chi_half = rng.chisquare(nu, size=(half, n_steps))
        t_half = g_half * np.sqrt((nu - 2.0) / np.maximum(chi_half, 1e-6))
        t_half = np.clip(t_half, -10.0, 10.0)
        # Standardized normal proxy for leverage driver
        z_drv_half = np.clip(g_half, -4.0, 4.0)
    else:
        t_half = rng.standard_normal((half, n_steps))
        z_drv_half = t_half

    t_mat = np.vstack((t_half, -t_half))
    t_mat /= np.maximum(np.std(t_mat, axis=0, keepdims=True), 1e-6)
    z_drv = np.vstack((z_drv_half, -z_drv_half))

    # 2. Stationary AR(1) log-volatility multiplier m_{i,k} with E[m^2] == 1
    if tau_vov > 1e-3:
        phi = math.exp(-math.log(2.0) / max(hl_steps, 0.5))
        eta = tau_vov * math.sqrt(max(1.0 - phi * phi, 1e-6))
        xi_half = rng.standard_normal((half, n_steps))
        xi = np.vstack((xi_half, -xi_half))
        xi /= np.maximum(np.std(xi, axis=0, keepdims=True), 1e-6)

        h0_half = rng.standard_normal(half) * tau_vov
        h = np.concatenate((h0_half, -h0_half))
        h *= tau_vov / max(float(np.std(h)), 1e-6)

        m_mat = np.empty((num_sim, n_steps), dtype=np.float64)
        s_rho = math.sqrt(max(1.0 - rho_lev * rho_lev, 0.0))
        for k in range(n_steps):
            if k > 0:
                innov_v = rho_lev * z_drv[:, k - 1] + s_rho * xi[:, k]
                h = phi * h + eta * innov_v
            h_c = np.clip(h, -2.2, 2.2)
            m_col = np.exp(h_c)
            # Normalize column so E[m_col^2 * t_col^2] == 1.0 before c_scale!
            # Wait: normalizing m_col alone by sqrt(mean(m_col^2)) preserves exact marginal variance!
            m_col /= math.sqrt(max(float(np.mean(m_col * m_col)), 1e-8))
            m_mat[:, k] = m_col
        shocks = (c_scale * m_mat) * t_mat
    else:
        shocks = c_scale * t_mat

    cum_y = modeling._build_cum_log_returns(shocks, sigma_log, step_kappas)
    np.clip(cum_y, -10.0, 10.0, out=cum_y)
    paths = np.empty((num_sim, n_steps + 1), dtype=np.float32)
    paths[:, 0] = np.float32(p0)
    exp_y = np.exp(cum_y)
    exp_y *= p0 / np.maximum(np.mean(exp_y, axis=0, keepdims=True), 1e-12)
    paths[:, 1:] = exp_y.astype(np.float32)
    return paths


def precompute_prompt_dataset(assets: list[str], start_str: str, end_str: str, freq_str: str, time_inc: int, time_len: int):
    num_steps = time_len // time_inc
    dataset = []
    for asset in assets:
        p = load_minute_prices(
            WORKSPACE / "market_data",
            asset,
            pd.Timestamp(start_str) - pd.Timedelta(days=32),
            pd.Timestamp(end_str) + pd.Timedelta(days=2),
        )
        times = pd.date_range(start_str, end_str, freq=freq_str)
        for t0 in times:
            fut_idx = pd.date_range(t0, t0 + pd.Timedelta(seconds=time_len), freq=f"{time_inc}s")
            fut = p["close"].reindex(fut_idx).to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(fut)) or np.any(fut <= 0):
                continue
            ctx = context_slice(p, t0, time_len, 30)
            closes = ctx["close"].to_numpy(dtype=np.float64)
            if len(closes) < 500 or not np.isfinite(closes[-1]):
                continue
            ts_sec = modeling._extract_timestamps_sec(ctx.index, len(closes))
            sig_bps, kappas = modeling._estimate_step_dynamics(
                asset, closes, ts_sec, int(t0.timestamp()), time_inc, num_steps
            )
            dataset.append((asset, int(t0.timestamp()), float(closes[-1]), sig_bps * 1e-4, kappas, fut))
    return dataset


def _score_one(args):
    idx, p0, sig_log, kappas, fut, time_inc, scoring_intervals, vol_blocks, nu, tau_vov, hl_steps, rho_lev, c_scale, num_sim = args
    rng = np.random.default_rng(10000 + idx)
    paths = simulate_paths_parametric(rng, p0, sig_log, kappas, num_sim, nu, tau_vov, hl_steps, rho_lev, c_scale)
    score, _ = calculate_total_score_for_miner(
        paths,
        fut,
        time_inc,
        scoring_intervals,
        vol_blocks,
    )
    return float(score)


def eval_dataset(executor, dataset, comp_cfg, nu, tau_vov, hl_steps, rho_lev, c_scale, num_sim=400):
    tasks = [
        (idx, p0, sig_log, kappas, fut, comp_cfg.time_increment, comp_cfg.scoring_intervals, comp_cfg.vol_scoring_blocks, nu, tau_vov, hl_steps, rho_lev, c_scale, num_sim)
        for idx, (_asset, _ts, p0, sig_log, kappas, fut) in enumerate(dataset)
    ]
    scores = list(executor.map(_score_one, tasks, chunksize=16))
    return float(np.mean(scores))


def main():
    with ProcessPoolExecutor(max_workers=12) as ex:
        print("1. Precomputing <= 2026-08-10 calibration prompts for CRYPTO_1H...")
        ds_1h = precompute_prompt_dataset(
            ["BTC", "ETH", "SOL", "XRP", "HYPE"], "2026-04-01T00:00:00Z", "2026-08-09T20:00:00Z", "31h", 60, 3600
        )
        print(f"   Loaded {len(ds_1h)} CRYPTO_1H prompts (<= 2026-08-10).")
        base_1h = eval_dataset(ex, ds_1h, CRYPTO_1H, nu=200.0, tau_vov=0.0, hl_steps=20.0, rho_lev=0.0, c_scale=1.0)
        print(f"   CRYPTO_1H Gaussian constant-path baseline CRPS = {base_1h:.3f}")

        for params in [
            (6.0, 0.00, 30.0, 0.00, 0.98),
            (6.0, 0.35, 25.0, -0.15, 0.98),
            (5.5, 0.42, 30.0, -0.18, 0.98),
            (5.5, 0.48, 35.0, -0.18, 1.00),
            (5.0, 0.45, 30.0, -0.18, 0.96),
            (6.5, 0.50, 40.0, -0.18, 1.00),
            (8.0, 0.48, 35.0, -0.18, 1.00),
            (6.0, 0.55, 45.0, -0.18, 1.00),
        ]:
            sc = eval_dataset(ex, ds_1h, CRYPTO_1H, *params)
            print(f"   params (nu={params[0]}, tau={params[1]}, hl={params[2]}, rho={params[3]}, c={params[4]}) -> CRPS={sc:.3f} ({(sc/base_1h-1)*100:+.2f}%)")

        print("\n2. Precomputing <= 2026-08-10 calibration prompts for CRYPTO_24H...")
        ds_c24 = precompute_prompt_dataset(
            ["BTC", "ETH", "SOL", "XRP", "HYPE"], "2026-04-01T00:00:00Z", "2026-08-09T00:00:00Z", "43h", 300, 86400
        )
        print(f"   Loaded {len(ds_c24)} CRYPTO_24H prompts (<= 2026-08-10).")
        base_c24 = eval_dataset(ex, ds_c24, CRYPTO_24H, nu=200.0, tau_vov=0.0, hl_steps=36.0, rho_lev=0.0, c_scale=1.0)
        print(f"   CRYPTO_24H Gaussian baseline CRPS = {base_c24:.3f}")
        for params in [
            (5.5, 0.00, 36.0, 0.00, 0.98),
            (5.5, 0.28, 36.0, -0.18, 1.00),
            (6.0, 0.35, 48.0, -0.18, 1.00),
            (5.0, 0.32, 48.0, -0.18, 0.98),
            (6.5, 0.38, 60.0, -0.20, 1.02),
            (6.0, 0.42, 60.0, -0.18, 1.00),
        ]:
            sc = eval_dataset(ex, ds_c24, CRYPTO_24H, *params)
            print(f"   params (nu={params[0]}, tau={params[1]}, hl={params[2]}, rho={params[3]}, c={params[4]}) -> CRPS={sc:.3f} ({(sc/base_c24-1)*100:+.2f}%)")

        print("\n3. Precomputing <= 2026-08-10 calibration prompts for COM_EQU_24H...")
        ds_eq24 = precompute_prompt_dataset(
            ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"],
            "2026-06-05T00:00:00Z",
            "2026-08-09T00:00:00Z",
            "37h",
            300,
            86400,
        )
        print(f"   Loaded {len(ds_eq24)} COM_EQU_24H prompts (<= 2026-08-10).")
        base_eq24 = eval_dataset(ex, ds_eq24, COM_EQU_24H, nu=200.0, tau_vov=0.0, hl_steps=36.0, rho_lev=0.0, c_scale=1.0)
        print(f"   COM_EQU_24H Gaussian baseline CRPS = {base_eq24:.3f}")
        for params in [
            (5.5, 0.00, 36.0, 0.00, 0.98),
            (5.0, 0.25, 36.0, -0.12, 0.98),
            (5.5, 0.30, 48.0, -0.12, 1.00),
            (6.0, 0.32, 48.0, -0.12, 1.00),
            (5.0, 0.32, 48.0, -0.15, 0.96),
            (6.0, 0.25, 36.0, -0.10, 1.02),
        ]:
            sc = eval_dataset(ex, ds_eq24, COM_EQU_24H, *params)
            print(f"   params (nu={params[0]}, tau={params[1]}, hl={params[2]}, rho={params[3]}, c={params[4]}) -> CRPS={sc:.3f} ({(sc/base_eq24-1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
