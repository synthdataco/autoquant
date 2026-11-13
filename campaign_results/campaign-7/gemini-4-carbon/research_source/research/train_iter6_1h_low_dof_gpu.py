"""Low-DOF (28-parameter) GPU Differentiable Direct-CRPS HAR + Self-Normalized 1m Garman-Klass + Bipower
+ Cooling/GJR Forecaster for CRYPTO_1H (strictly <= 2026-08-10: Inner Train <= Jun 20, Inner Val Jun 22..Aug 10).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from synth.validator.moving_average import ASSET_COEFFICIENTS

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "agent"))
import modeling  # noqa: E402
from research.fit_seasonality_v1 import CRYPTO_ASSETS  # noqa: E402
from research.train_iter6_gpu_direct_crps import (  # noqa: E402
    OUT_NPZ,
    load_asset_arrays,
    sample_base_shocks_gpu,
    total_crps_1h_torch,
)

# 11 positive variance components in exact deseasonalized 1m variance units:
# [cc_3.5, cc_10, cc_45, cc_180, cc_1440, cc_10080, gk_5, gk_30, gk_180, bv_15, prior_v]
INIT_W_SHORT = np.array([0.002, 0.474, 0.218, 0.178, 0.050, 0.050, 0.002, 0.002, 0.002, 0.002, 0.020], dtype=np.float32)
INIT_W_LONG = np.array([0.002, 0.236, 0.248, 0.298, 0.090, 0.090, 0.002, 0.002, 0.002, 0.002, 0.030], dtype=np.float32)


def _extract_1h_worker(asset: str) -> list[dict]:
    arr = load_asset_arrays(asset, "2025-10-01", "2026-08-10")
    ts_all = arr["ts_sec"]
    c_all = arr["close"]
    o_all = arr["open"]
    h_all = arr["high"]
    l_all = arr["low"]

    calib = modeling._load_calibration()
    arrays = calib["arrays"]
    cal_data = modeling._load_calendar_data()
    asset_events = modeling._get_asset_events(cal_data, asset)
    week1m = arrays[f"week1m_{asset}"]
    moh = arrays[f"moh_{asset}"]
    prior_1m = float(calib["meta"]["assets"][asset]["prior_sigma_1m_bps"])
    prior_v = prior_1m * prior_1m

    grid = pd.date_range(pd.Timestamp("2025-11-01T00:00:00Z"), pd.Timestamp("2026-08-10T22:00:00Z"), freq="2h")
    rows = []
    for t_pd in grid:
        st = int(t_pd.timestamp())
        target_ts = st + 60 * np.arange(0, 61, dtype=np.int64)
        pos_r = np.searchsorted(ts_all, target_ts)
        if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
            continue
        rp = c_all[pos_r]
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
            continue

        lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
        hi = int(np.searchsorted(ts_all, st - 120, side="right"))
        if hi - lo < 3000:
            continue

        c_ctx = c_all[lo:hi]
        pos_c = c_ctx[np.isfinite(c_ctx) & (c_ctx > 0.0)]
        if len(pos_c) < 200:
            continue
        p0 = float(pos_c[-1])

        fut_min_ts = st + 60 * np.arange(1, 61, dtype=np.int64)
        fut_dow, fut_slot5, _ = modeling._utc_sec_to_ny_parts(fut_min_ts)
        fut_moh = (fut_min_ts // 60) % 60
        ev_fut_var = modeling._event_var_mult_1m(fut_min_ts, asset, True, calib, asset_events)
        fut_s_1m = week1m[fut_dow, fut_slot5] * moh[fut_moh] * np.sqrt(ev_fut_var)
        fut_seas_var = (fut_s_1m * fut_s_1m).astype(np.float32)

        n_tail = min(hi - lo, 25_000)
        c_s = c_ctx[-n_tail:]
        o_s = o_all[lo:hi][-n_tail:]
        h_s = h_all[lo:hi][-n_tail:]
        l_s = l_all[lo:hi][-n_tail:]
        ts_s = ts_all[lo:hi][-n_tail:]

        ok = np.isfinite(c_s[1:]) & np.isfinite(c_s[:-1]) & (c_s[1:] > 0.0) & (c_s[:-1] > 0.0) & ((ts_s[1:] - ts_s[:-1]) == 60)
        if int(np.sum(ok)) < 500:
            continue

        r1 = (np.log(c_s[1:][ok]) - np.log(c_s[:-1][ok])) * 1e4
        ts_v = ts_s[1:][ok]
        c_dow, c_slot5, _ = modeling._utc_sec_to_ny_parts(ts_v)
        c_moh = (ts_v // 60) % 60
        ev_hist_1m = np.sqrt(modeling._event_var_mult_1m(ts_v, asset, True, calib, asset_events))
        s_hist = np.maximum(week1m[c_dow, c_slot5] * moh[c_moh] * ev_hist_1m, 0.08)

        u_raw = r1 / s_hist
        local_scale = modeling._causal_rolling_rms(np.clip(u_raw, -75.0, 75.0) ** 2, window=1440, prior_val=prior_v)
        u_clip = np.clip(u_raw, -4.0 * local_scale, 4.0 * local_scale)
        u_hub_sq = u_clip ** 2

        evs_cc = modeling._ewma_terminals(u_hub_sq, (3.5, 10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)

        # Causal 24h-self-normalized 1m Garman-Klass & Bipower on recent 4,320 valid bars (3 days)
        m_t = min(len(u_hub_sq), 4320)
        u_sq_t = u_hub_sq[-m_t:]
        u_cl_t = u_clip[-m_t:]
        sc_t = local_scale[-m_t:]
        s_t = s_hist[-m_t:]
        o_v = o_s[1:][ok][-m_t:]
        h_v = h_s[1:][ok][-m_t:]
        l_v = l_s[1:][ok][-m_t:]
        c_v = c_s[1:][ok][-m_t:]

        ohlc_ok = np.isfinite(o_v) & np.isfinite(h_v) & np.isfinite(l_v) & (o_v > 0.0) & (l_v > 0.0) & (h_v >= l_v)
        hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h_v, 1e-12)) - np.log(np.maximum(l_v, 1e-12))) * 1e4, 0.0)
        co_bps = np.where(ohlc_ok, (np.log(np.maximum(c_v, 1e-12)) - np.log(np.maximum(o_v, 1e-12))) * 1e4, 0.0)
        gk_u = np.minimum(np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_t * s_t), 16.0 * sc_t * sc_t)

        abs_u = np.sqrt(u_sq_t)
        bv_u = np.empty_like(u_sq_t)
        bv_u[0] = u_sq_t[0]
        bv_u[1:] = np.minimum((math.pi / 2.0) * abs_u[1:] * abs_u[:-1], 2.0 * u_sq_t[1:] + sc_t[1:] ** 2)

        gk_ev = modeling._ewma_terminals(gk_u, (5.0, 30.0, 180.0, 1440.0), prior_v)
        bv_ev = modeling._ewma_terminals(bv_u, (15.0, 1440.0), prior_v)
        cc_1440 = float(evs_cc[4])

        if gk_ev[3] > 1e-4:
            gk_ratio = float(np.clip(cc_1440 / gk_ev[3], 0.25, 4.0))
            gk_5, gk_30, gk_180 = float(gk_ev[0]) * gk_ratio, float(gk_ev[1]) * gk_ratio, float(gk_ev[2]) * gk_ratio
        else:
            gk_5, gk_30, gk_180 = float(evs_cc[1]), float(evs_cc[2]), float(evs_cc[3])

        if bv_ev[1] > 1e-4:
            bv_15 = float(bv_ev[0]) * float(np.clip(cc_1440 / bv_ev[1], 0.25, 4.0))
        else:
            bv_15 = float(evs_cc[1])

        comps = np.array(
            [evs_cc[0], evs_cc[1], evs_cc[2], evs_cc[3], evs_cc[4], evs_cc[5], gk_5, gk_30, gk_180, bv_15, prior_v],
            dtype=np.float32,
        )

        # Low-DOF state signals (4 clean scalars)
        cool_slope = float(np.clip(math.log((evs_cc[1] + 1e-6) / (evs_cc[4] + 1e-6)), -2.5, 2.5))
        z_sig = u_cl_t / np.maximum(sc_t, 1e-4)
        zr_240 = float(np.sum(z_sig[-240:])) / math.sqrt(240.0) if len(z_sig) >= 240 else 0.0
        gjr_neg = float(np.clip(max(-zr_240, 0.0) - 0.39, -0.39, 2.6))
        if len(u_sq_t) >= 360:
            blk = np.mean(u_sq_t[-360:].reshape(24, 15), axis=1)
            vov_s = float(np.clip(np.std(np.log(np.maximum(blk, 1e-4))) - 0.85, -1.2, 1.2))
        else:
            vov_s = 0.0
        jump_s = float(np.clip(math.log((bv_15 + 1e-6) / (evs_cc[1] + 1e-6)), -2.0, 2.0))

        # Also record exact Iter 5 v_short & v_long for exact baseline comparison
        vs_iter5 = float(0.48 * evs_cc[1] + 0.22 * evs_cc[2] + 0.18 * evs_cc[3] + 0.05 * evs_cc[4] + 0.05 * evs_cc[5] + 0.02 * prior_v)
        vl_iter5 = float(0.24 * evs_cc[1] + 0.25 * evs_cc[2] + 0.30 * evs_cc[3] + 0.09 * evs_cc[4] + 0.09 * evs_cc[5] + 0.03 * prior_v)

        rows.append(
            {
                "asset": asset,
                "start_ts": st,
                "comps": comps,
                "aux": np.array([cool_slope, gjr_neg, vov_s, jump_s], dtype=np.float32),
                "vs_iter5": vs_iter5,
                "vl_iter5": vl_iter5,
                "prior_v": prior_v,
                "fut_seas_var": fut_seas_var,
                "p0": p0,
                "real_prices": rp.astype(np.float32),
            }
        )
    return rows


class LowDOF1HModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_w_short_0", torch.tensor(np.log(INIT_W_SHORT)))
        self.register_buffer("log_w_long_0", torch.tensor(np.log(INIT_W_LONG)))
        self.delta_w_short = nn.Parameter(torch.zeros(11))
        self.delta_w_long = nn.Parameter(torch.zeros(11))
        # 6 scalar adjustments: [global_scale, hl_shift, cool_short, gjr_mult, vov_base, vov_slope]
        self.scalars = nn.Parameter(torch.zeros(6))

    def forward(self, comps: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ws = torch.softmax(self.log_w_short_0 + self.delta_w_short, dim=0)
        wl = torch.softmax(self.log_w_long_0 + self.delta_w_long, dim=0)
        g_scale = torch.exp(0.08 * torch.tanh(self.scalars[0]))
        mod_short = torch.exp(torch.clamp(0.18 * torch.tanh(self.scalars[2]) * aux[:, 0] + 0.18 * torch.tanh(self.scalars[3]) * aux[:, 1] + 0.15 * torch.tanh(self.scalars[5]) * aux[:, 3], -0.25, 0.25))
        mod_long = torch.exp(torch.clamp(0.18 * torch.tanh(self.scalars[3]) * aux[:, 1], -0.22, 0.22))
        vs = (comps @ ws) * g_scale * mod_short
        vl = (comps @ wl) * g_scale * mod_long
        hl = 18.0 * torch.exp(0.45 * torch.tanh(self.scalars[1]))
        tau = 0.40 * torch.exp(torch.clamp(0.12 * torch.tanh(self.scalars[4]) + 0.16 * torch.tanh(self.scalars[5]) * aux[:, 2], -0.20, 0.20))
        return vs, vl, hl.expand(comps.shape[0]), tau


def sim_1h_gpu(
    vs: torch.Tensor,
    vl: torch.Tensor,
    hl: torch.Tensor,
    tau: torch.Tensor,
    prior_v: torch.Tensor,
    fut_seas_var: torch.Tensor,
    p0: torch.Tensor,
    t_mat: torch.Tensor,
    h_unit: torch.Tensor,
) -> torch.Tensor:
    b_sz, m_paths, n_steps = t_mat.shape
    k_idx = torch.arange(n_steps, device=t_mat.device, dtype=t_mat.dtype).view(1, n_steps)
    decay = torch.exp(-math.log(2.0) * k_idx / hl.unsqueeze(1))
    v_des = torch.maximum(vl.unsqueeze(1) + (vs - vl).unsqueeze(1) * decay, 0.04 * prior_v.unsqueeze(1))
    sigma_log = torch.sqrt(v_des * fut_seas_var) * 1e-4

    m_raw = torch.exp(torch.clamp(tau.view(b_sz, 1, 1) * h_unit, -2.2, 2.2))
    m_rms = torch.sqrt(torch.clamp(torch.mean(m_raw * m_raw, dim=1, keepdim=True), min=1e-8))
    z = (1.005 * (m_raw / m_rms)) * t_mat
    cum_y = torch.cumsum(z * sigma_log.unsqueeze(1), dim=2)
    exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
    p_fut = exp_y * (p0.view(b_sz, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
    return torch.cat([p0.view(b_sz, 1, 1).expand(b_sz, m_paths, 1), p_fut], dim=2)


def main():
    t0 = time.time()
    torch.manual_seed(20260810)
    device = torch.device("cuda")
    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(_extract_1h_worker, CRYPTO_ASSETS))
    rows = [r for sub in res for r in sub]

    cut_tr = int(pd.Timestamp("2026-06-20T00:00:00Z").timestamp())
    cut_va = int(pd.Timestamp("2026-06-22T00:00:00Z").timestamp())
    tr_r = [r for r in rows if r["start_ts"] <= cut_tr]
    va_r = [r for r in rows if r["start_ts"] >= cut_va]
    print(f"Extracted CRYPTO_1H in {time.time() - t0:.1f}s: train={len(tr_r)}, inner_val={len(va_r)}")

    def to_gpu(sub_rows: list[dict]) -> dict:
        return {
            "comps": torch.tensor(np.stack([r["comps"] for r in sub_rows]), device=device),
            "aux": torch.tensor(np.stack([r["aux"] for r in sub_rows]), device=device),
            "vs5": torch.tensor([r["vs_iter5"] for r in sub_rows], dtype=torch.float32, device=device),
            "vl5": torch.tensor([r["vl_iter5"] for r in sub_rows], dtype=torch.float32, device=device),
            "prior_v": torch.tensor([r["prior_v"] for r in sub_rows], dtype=torch.float32, device=device),
            "fut_seas": torch.tensor(np.stack([r["fut_seas_var"] for r in sub_rows]), device=device),
            "p0": torch.tensor([r["p0"] for r in sub_rows], dtype=torch.float32, device=device),
            "rp": torch.tensor(np.stack([r["real_prices"] for r in sub_rows]), device=device),
            "coefs": torch.tensor([ASSET_COEFFICIENTS.get(r["asset"], 1.0) for r in sub_rows], dtype=torch.float32, device=device),
            "rho": torch.full((len(sub_rows),), -0.18, dtype=torch.float32, device=device),
            "assets": np.array([r["asset"] for r in sub_rows]),
        }

    tr = to_gpu(tr_r)
    va = to_gpu(va_r)
    model = LowDOF1HModel().to(device)

    @torch.no_grad()
    def eval_val(use_iter5: bool = False, blend: float = 1.0) -> tuple[float, np.ndarray]:
        gen = torch.Generator(device=device)
        gen.manual_seed(20260810)
        n_va = va["p0"].shape[0]
        scores = np.empty(n_va, dtype=np.float64)
        for i0 in range(0, n_va, 128):
            i1 = min(i0 + 128, n_va)
            bs = i1 - i0
            t_m, h_u = sample_base_shocks_gpu(bs, 350, 60, 9.0, 25.0, va["rho"][i0:i1], device, gen)
            if use_iter5:
                vs = va["vs5"][i0:i1]
                vl = va["vl5"][i0:i1]
                hl = torch.full((bs,), 18.0, device=device)
                tau = torch.full((bs,), 0.40, device=device)
            else:
                vs_m, vl_m, hl_m, tau_m = model(va["comps"][i0:i1], va["aux"][i0:i1])
                vs = blend * vs_m + (1.0 - blend) * va["vs5"][i0:i1]
                vl = blend * vl_m + (1.0 - blend) * va["vl5"][i0:i1]
                hl = blend * hl_m + (1.0 - blend) * 18.0
                tau = blend * tau_m + (1.0 - blend) * 0.40
            paths = sim_1h_gpu(vs, vl, hl, tau, va["prior_v"][i0:i1], va["fut_seas"][i0:i1], va["p0"][i0:i1], t_m, h_u)
            scores[i0:i1] = total_crps_1h_torch(paths, va["rp"][i0:i1]).cpu().numpy()
        a_means, a_coefs = [], []
        for a in CRYPTO_ASSETS:
            m_a = va["assets"] == a
            a_means.append(float(np.mean(scores[m_a])))
            a_coefs.append(ASSET_COEFFICIENTS.get(a, 1.0))
        return float(np.average(a_means, weights=a_coefs)), scores

    base_w, base_scores = eval_val(use_iter5=True)
    init_w, _ = eval_val(use_iter5=False)
    print(f"  Exact Iter 5 Inner-Val CRPS = {base_w:.4f} | Init LowDOF CRPS = {init_w:.4f}")

    # Precompute detached train normalization scale
    with torch.no_grad():
        gen0 = torch.Generator(device=device)
        gen0.manual_seed(999)
        n_tr = tr["p0"].shape[0]
        tr_ref = torch.empty(n_tr, device=device)
        for i0 in range(0, n_tr, 128):
            i1 = min(i0 + 128, n_tr)
            bs = i1 - i0
            t_m, h_u = sample_base_shocks_gpu(bs, 64, 60, 9.0, 25.0, tr["rho"][i0:i1], device, gen0)
            p_b = sim_1h_gpu(tr["vs5"][i0:i1], tr["vl5"][i0:i1], torch.full((bs,), 18.0, device=device), torch.full((bs,), 0.40, device=device), tr["prior_v"][i0:i1], tr["fut_seas"][i0:i1], tr["p0"][i0:i1], t_m, h_u)
            tr_ref[i0:i1] = torch.clamp(total_crps_1h_torch(p_b, tr["rp"][i0:i1]), min=20.0)

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=24, eta_min=1e-3)
    best_w = base_w
    best_sd = None
    best_ep = 0
    t_gen = torch.Generator(device=device)
    t_gen.manual_seed(7777)

    for ep in range(1, 25):
        perm = torch.randperm(tr["p0"].shape[0], device=device, generator=t_gen)
        ep_l, nb = 0.0, 0
        for b0 in range(0, tr["p0"].shape[0], 128):
            idx = perm[b0 : b0 + 128]
            if len(idx) < 16:
                continue
            bs = len(idx)
            opt.zero_grad(set_to_none=True)
            vs, vl, hl, tau = model(tr["comps"][idx], tr["aux"][idx])
            t_m, h_u = sample_base_shocks_gpu(bs, 128, 60, 9.0, 25.0, tr["rho"][idx], device, t_gen)
            paths = sim_1h_gpu(vs, vl, hl, tau, tr["prior_v"][idx], tr["fut_seas"][idx], tr["p0"][idx], t_m, h_u)
            sc = total_crps_1h_torch(paths, tr["rp"][idx])
            rel = (tr["coefs"][idx] * (sc / tr_ref[idx])).mean() / tr["coefs"][idx].mean()
            reg = 0.008 * (model.delta_w_short.pow(2).mean() + model.delta_w_long.pow(2).mean() + model.scalars.pow(2).mean())
            loss = rel + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_l += float(rel.item())
            nb += 1
        sched.step()
        if ep % 2 == 0:
            vw, _ = eval_val(use_iter5=False, blend=1.0)
            mk = ""
            if vw < best_w - 1e-4:
                best_w = vw
                best_ep = ep
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mk = " *BEST*"
            print(f"  Epoch {ep:02d} | train_rel={ep_l/nb:.5f} | inner_val_w={vw:.4f} ({(vw/base_w-1)*100:+.3f}%){mk}")

    if best_sd is not None:
        model.load_state_dict(best_sd)
        best_bl = 1.0
        best_sc = base_scores
        for bl in (0.70, 0.85, 1.00):
            vw, vsc = eval_val(use_iter5=False, blend=bl)
            if vw <= best_w + 1e-6:
                best_w = vw
                best_bl = bl
                best_sc = vsc
        wins = 0
        print(f"  Best Epoch {best_ep:02d} (blend={best_bl:.2f}) Inner-Val Per-Asset Breakdown:")
        for a in CRYPTO_ASSETS:
            m_a = va["assets"] == a
            b0_a = float(np.mean(base_scores[m_a]))
            b1_a = float(np.mean(best_sc[m_a]))
            if b1_a < b0_a:
                wins += 1
            print(f"    {a:<5} (n={int(m_a.sum()):3d}): {b0_a:8.3f} -> {b1_a:8.3f} ({(b1_a/b0_a-1)*100:+6.3f}%)")

        ws = torch.softmax(model.log_w_short_0 + model.delta_w_short, dim=0).detach().cpu().numpy()
        wl = torch.softmax(model.log_w_long_0 + model.delta_w_long, dim=0).detach().cpu().numpy()
        sc_np = model.scalars.detach().cpu().numpy()
        print("  Learned w_short:", np.round(ws, 3))
        print("  Learned w_long :", np.round(wl, 3))
        print("  Learned scalars:", np.round(sc_np, 3))

        rel_pct = (best_w / base_w - 1.0) * 100.0
        if rel_pct < -0.025 and wins >= 3:
            # Append lowdof1h weights to iter6_gpu_heads.npz!
            prev = dict(np.load(OUT_NPZ, allow_pickle=False).items()) if OUT_NPZ.exists() else {}
            heads = json.loads(str(prev["heads_json"])) if "heads_json" in prev else []
            if "lowdof1h" not in heads:
                heads.append("lowdof1h")
            prev["heads_json"] = np.array(json.dumps(heads))
            prev["lowdof1h_ws"] = ws.astype(np.float32)
            prev["lowdof1h_wl"] = wl.astype(np.float32)
            prev["lowdof1h_scalars"] = sc_np.astype(np.float32)
            prev["lowdof1h_blend"] = np.array([best_bl], dtype=np.float32)
            np.savez_compressed(OUT_NPZ, **prev)
            print(f"  [VERDICT CRYPTO_1H] KEEP ({base_w:.3f} -> {best_w:.3f}, {rel_pct:+.3f}%, wins={wins}/5) -> saved to {OUT_NPZ}!")
            return
    print("  [VERDICT CRYPTO_1H] DISCARD on Inner Val.")


if __name__ == "__main__":
    main()
