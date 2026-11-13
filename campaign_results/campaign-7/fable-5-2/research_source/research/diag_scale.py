"""CRPS-optimal width correction per cell (price a residual before fitting it).

For every (prompt, step) of a validation window, the step's own marginal (v_h, total log-vol sd) is sampled and
its CRPS against the realised step return is computed at several width multipliers s. Summing per cell gives
the CRPS-optimal multiplier of the cell and the gain a perfect per-cell rescale would bring, in % of the
window's step-level CRPS. Cells: session x horizon bucket, and hour of the NY week.
Training-loss arithmetic (torch), a diagnostic only.
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
import numpy as np, torch, core, pandas as pd
from pathlib import Path
import train as T
pdir, ds, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
vs, ve = T.minute_stamp(sys.argv[5]), T.minute_stamp(sys.argv[6])
S = np.array([0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.9])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def best(c):                      # c [cells, len(S)] summed CRPS -> (s*, gain)
    i = c.argmin(1); return S[i], c[:, 4] - c.min(1)
for a in sys.argv[4].split(","):
    path = ds / f"{a}_{tag}.npz"
    z0 = np.load(path); sel = (z0["t0"] >= vs) & (z0["t0"] < ve)
    d = T.Data(path, tag, lambda t: (t >= vs) & (t < ve))
    par = json.loads((pdir / f"model_{a}_{tag}.json").read_text())
    m = T.Model(tag).to(T.DEV); m.load_raw(par["raw"])
    inc = T.CFG[tag]["inc"]; H = m.H; N = 400
    gen = torch.Generator(device=T.DEV); gen.manual_seed(1)
    out = np.zeros((d.n, H, len(S)), dtype=np.float32)
    with torch.no_grad():
        A = m.c("A0") + m.c("A1") * m.phi
        rho = m.c("rho") ** (inc / 5.0); hh = torch.arange(1, H + 1, device=T.DEV).float()
        base_var = A * A + m.c("c") ** 2 * (1 - rho ** (2 * hh))
        for i in range(0, d.n, 64):
            idx = torch.arange(i, min(i + 64, d.n), device=T.DEV)
            v = m.variance(d, idx); lam = m.lam(d, idx)
            vg = base_var[None, :] + lam * lam                                   # [B,H]
            g = torch.randn(len(idx), N, H, device=T.DEV, generator=gen) * vg.sqrt()[:, None, :]
            e = v.sqrt()[:, None, :] * torch.exp(g - vg[:, None, :]) * torch.randn(len(idx), N, H, device=T.DEV, generator=gen)
            y = torch.log1p(d.ret[1][idx] / 1e4)
            for j, s in enumerate(S):
                out[i:i + len(idx), :, j] = (T.Model.crps((e * s) * 1e4, y * 1e4)).cpu().numpy()
    sess = d.ss.cpu().numpy()
    hb = [(0, 12), (12, 48), (48, 144), (144, 288)] if tag == "24h" else [(0, 5), (5, 15), (15, 30), (30, 60)]
    total = out[:, :, 4].sum()
    print(f"== {a} {tag} n={d.n} step-CRPS total at s=1: {total / d.n:.1f} per prompt; global s*: {S[out.sum((0, 1)).argmin()]}")
    cells = np.array([[out[:, lo:hi][sess[:, lo:hi] == c].sum(0) for lo, hi in hb] for c in range(4)]).reshape(16, len(S))
    sb, gain = best(cells)
    print(" s* by session(core,shoulder,night,weekend) x horizon:\n", sb.reshape(4, 4), "\n gain % :", (gain.reshape(4, 4) / total * 100), " total", round(gain.sum() / total * 100, 3))
    t0 = z0["t0"][sel]
    end_min = t0[:, None] + (np.arange(H)[None, :] + 1) * inc
    wall = core.ny_wall_minutes(pd.DatetimeIndex(end_min.ravel().astype("datetime64[m]")).tz_localize("UTC")).reshape(end_min.shape)
    how = core.minute_of_week(wall) // 60
    cells = np.array([out[how == k].sum(0) for k in range(168)])
    sb, gain = best(cells)
    print(" hour-of-week: total gain % from per-hour rescale:", round(gain.sum() / total * 100, 3))
    print(" s* (rows Mon..Sun):\n", sb.reshape(7, 24))
