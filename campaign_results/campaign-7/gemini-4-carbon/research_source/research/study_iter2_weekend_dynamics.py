"""Calibrate Iteration 2 strictly on <= 2026-08-10:
1. Online Empirical-Bayes 30d context session ratio adaptation shrinkage.
2. Closed-exchange weekend Ornstein-Uhlenbeck mean-reversion rate kappa_wknd matching 5m->30m->3h VR.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.fit_seasonality_v1 import ALL_ASSETS, CRYPTO_ASSETS, MIN_FIT_START, load_1m_closes

WORKSPACE = Path(__file__).resolve().parents[1]


def theoretical_ou_vr(kappa: float, m: int) -> float:
    """For stationary OU x_k = (1-kappa)*x_{k-1} + eps_k, step diff dx_k = x_k - x_{k-1}:
    Var(x_m - x_0) / (m * Var(x_1 - x_0)).
    If phi = 1 - kappa (0 < phi < 1):
      Var(x) = sigma_e^2 / (1 - phi^2)
      Var(x_m - x_0) = 2 * Var(x) * (1 - phi^m)
      Var(x_1 - x_0) = 2 * Var(x) * (1 - phi)
      => VR(m) = (1 - phi^m) / (m * (1 - phi))!
    Wait: check also transient starting from x_0 = 0 (at Friday close):
      Var(x_m) = sigma_e^2 * (1 - phi^{2m}) / (1 - phi^2), step 1 = sigma_e^2.
      Inside steady weekend (say 12h into Saturday), stationary VR(m) = (1 - phi^m) / (m * (1 - phi))!
    """
    if kappa <= 1e-7:
        return 1.0
    phi = max(0.0, 1.0 - kappa)
    return float((1.0 - phi ** m) / (m * (1.0 - phi)))


def fit_kappa_from_vrs(vr6: float, vr36: float) -> float:
    best_k = 0.0
    best_err = 1e9
    for k in np.linspace(0.0, 0.35, 351):
        p6 = theoretical_ou_vr(k, 6)
        p36 = theoretical_ou_vr(k, 36)
        err = (p6 - vr6) ** 2 + 1.5 * (p36 - vr36) ** 2
        if err < best_err:
            best_err = err
            best_k = float(k)
    return best_k


def main():
    npz = np.load(WORKSPACE / "agent" / "calibration_params.npz", allow_pickle=False)
    print("=== WEEKEND (Fri 17:00 NY -> Sun 18:00 NY) VARIANCE RATIOS & OU KAPPA ON <= 2026-08-10 ===")
    kappas = {}
    for asset in ALL_ASSETS:
        c = load_1m_closes(asset, MIN_FIT_START[asset], "2026-08-10")
        lc = np.log(c) * 1e4
        # Sample 5m grid
        c5 = lc.iloc[::5]
        ny = c5.index.tz_convert("America/New_York")
        dow = ny.day_of_week.to_numpy()
        hr = ny.hour.to_numpy()

        # Pure Saturday [00:00, 24:00) NY + Sunday [00:00, 17:00) NY interior blocks
        r5 = c5 - c5.shift(1)
        r30 = c5 - c5.shift(6)
        r180 = c5 - c5.shift(36)

        sat_m5 = (dow == 5) | ((dow == 6) & (hr < 16))
        # Ensure entire 3h lookback stayed inside weekend (after Fri 18:00 NY)
        sat_m30 = ((dow == 5) & (hr >= 1)) | ((dow == 6) & (hr < 16))
        sat_m180 = ((dow == 5) & (hr >= 3)) | ((dow == 6) & (hr < 16))

        v5 = float(np.nanmean(np.clip(r5[sat_m5], -150, 150) ** 2))
        v30 = float(np.nanmean(np.clip(r30[sat_m30], -350, 350) ** 2))
        v180 = float(np.nanmean(np.clip(r180[sat_m180], -800, 800) ** 2))

        vr6 = v30 / (6.0 * v5)
        vr36 = v180 / (36.0 * v5)
        raw_k = fit_kappa_from_vrs(vr6, vr36)
        # Shrink conservatively by 0.70 so we never over-tether if a real weekend macro headline hits
        k_shrunk = round(0.70 * raw_k, 4) if vr36 < 0.90 else 0.0
        kappas[asset] = k_shrunk
        print(
            f"  {asset:7s} | VR(30m/5m)={vr6:5.2f} | VR(3h/5m)={vr36:5.2f} | "
            f"raw_kappa={raw_k:.4f} -> shrunk_kappa={k_shrunk:.4f} "
            f"(pred VR6={theoretical_ou_vr(k_shrunk,6):.2f}, VR36={theoretical_ou_vr(k_shrunk,36):.2f})"
        )
    print("Fitted weekend OU kappas:", json.dumps(kappas))


if __name__ == "__main__":
    main()
