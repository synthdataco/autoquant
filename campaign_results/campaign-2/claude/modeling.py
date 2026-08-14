"""Synth SN50 miner: filtered historical simulation with seasonal HAR volatility.

Design follows directly from what the validator actually scores (see agent/journal.md):

  * On a 24h prompt the 5-min interval term is ~61% of total CRPS and 5min+30min is ~86%;
    on a 1h prompt the 1/2/5-min terms plus the cumulative gaps dominate. So the object that
    matters is the SHORT-HORIZON conditional return density, not the 24h drift.
  * CRPS at step t only sees the ensemble's MARGINAL distribution at t. Cross-step path
    structure is invisible to it. Effort therefore goes into (a) the scale of each step's
    return distribution and (b) its shape.
  * Empirically (agent/research/shape.py) resampling real standardized returns beats Gaussian
    innovations by 2.0-5.7% on the 5-min term, and Student-t(3) is within ~0.1-0.7% of it.
    Gaussian is never competitive.

The model, per prompt:
  1. sigma forecast: HAR-style blend of trailing realized vols (1h/3h/6h/24h/3d/7d) on
     SEASONALLY DESEASONALIZED minute returns, shrunk toward the asset's long-run vol.
  2. seasonality: a 168-bucket hour-of-week multiplier (fitted on the snapshot) is applied
     forward along the path, so quiet weekend / overnight hours get narrower fans and the
     13:00-15:00 UTC activity peak gets wider ones.
  3. innovations: bootstrap from the asset's pool of deseasonalized standardized minute
     returns (fat tails, correct asymmetry) with a Student-t(3) fallback.
  4. martingale: zero drift. The validator scores changes against a near-random-walk; any
     drift we add is uncompensated risk.

simulate() honours whatever (time_increment, time_length, num_simulations) it is given, so the
same code serves CRYPTO_24H (300/86400), COM_EQU_24H (300/86400) and CRYPTO_1H (60/3600).
Runs fully offline from agent/artifacts/calib.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ART = Path(__file__).resolve().parent / "artifacts" / "calib.npz"

# HAR windows in MINUTES (must match the order fitted in agent/research/fit_calib.py).
_HAR_WINDOWS = (60, 180, 360, 1440, 4320, 10080)

# ---- tunable knobs (swept in agent/research/*_sweep.py; module-level so the sweeps are
# ---- picklable across the harness's process pool) ----
PARAMS = {
    "scale": 1.0,        # global multiplier on the forecast sigma
    "use_kcal": 1.0,     # 1 = apply the per-(asset,increment) calibrated scale from the artifact
    "hurst": 0.5,        # sigma(t) ~ t**hurst per-step time scaling (0.5 = random walk)
    # swept: 1.0 beats 0.85 by 0.06% and 0.70 by 0.46% on crypto-1h training windows. The HAR
    # blend already shrinks internally through its 7-day term, so extra shrinkage just adds bias.
    "shrink": 1.0,       # log-space weight on the HAR forecast vs the long-run vol
    "season_pow": 1.0,   # exponent on the seasonal amplitude (0 = off, 1 = as fitted)
    "ewma_mix": 0.0,     # log-space weight on the trailing EWMA sigma vs the HAR forecast
    # --- stochastic volatility ---
    # CRPS at step t scores the MARGINAL, which integrates over the unknown future vol path.
    # A static sigma cannot produce dispersion that GROWS with horizon; an explicit log-vol
    # random walk can. "filtered" switches to the properly-standardized residual pool, which
    # is only correct once the vol dispersion is modelled explicitly rather than smuggled in
    # through a globally-standardized pool.
    "sv_vol": 0.0,       # per-step sd of the log-vol random walk (0 = static sigma)
    "sv_level": 0.0,     # sd of a per-path constant log-vol offset (HAR forecast-error scale)
    "filtered": 0.0,     # 1 = use the conditionally-standardized pool (poolc*)
    # --- zero-inflation / tick discreteness (equity-like assets only) ---
    "zi_mix": 0.0,       # weight on the state-dependent zero atom (0 = purely continuous)
    "tick": 0.0,         # 1 = round simulated prices to the asset's observed tick size
    # --- mean reversion (MA(1)) on the increments ---
    # perinterval.py found the CRPS-optimal dispersion multiplier is MONOTONE in horizon:
    # 1min wants 1.10, 2min 1.05, 5min 0.95, 15-60min and all the *_gaps want ~0.90. One global
    # scale cannot express that -- it splits the difference and loses at both ends (~0.4% of
    # crypto-1h CRPS, weighted). A single scale is only right if increments are independent.
    # The fix is at the level of the INCREMENT CORRELATION, not the scale: an MA(1) filter
    # e_t - theta*e_{t-1} with theta > 0 raises the one-step variance by (1+theta^2) while the
    # h-step cumulative variance grows only like (1-theta)^2*h + O(1), so it inflates the short
    # marginal and deflates the cumulative ones simultaneously. That matches the measured
    # microstructure: first-order return autocorrelation is -0.001 to -0.011 (bid-ask bounce).
    # Normalized to hold the ONE-STEP variance at step_sigma so kcal stays valid.
    # Swept on TRAINING windows disjoint from every reporting window. Optima: crypto-1h (1-min
    # increments) 0.10 (-0.12%); crypto-24h (5-min) 0.20 (-0.30%). The increment dependence is
    # expected -- a 5-min step aggregates five minutes of bounce, so a larger theta is needed to
    # produce the same cumulative-variance deficit.
    #
    # EQUITIES GET theta = 0. The crypto-24h sweep initially suggested 0.20 for the equity-like
    # assets too, but that did NOT replicate: on the reporting window it was neutral (+4.792% ->
    # +4.800%) and over three further windows the apparent gain was ~0.1% with the sign flipping per
    # asset (TSLAX, WTIOIL and SPCX all got worse). That is inside the noise floor of these windows,
    # so it is not claimed. This is also mechanistically consistent: these feeds are zero-inflated
    # and stale (P(5-min change = 0) = 0.105 on AAPLX vs 0.002 on BTC), and staleness induces
    # POSITIVE serial correlation, which works against the negative MA(1) bounce.
    "ma1_1": 0.10,       # theta at 1-minute increments (crypto-1h)
    "ma1_5": 0.20,       # theta at 5-minute increments (crypto-24h)
    "ma1_equ": 0.0,      # theta for equity-like assets at any increment (see above)
    # ma2 was the obvious follow-up to ma1 -- MA(1) is one parameter fitted to a smooth monotone
    # curve, so a second lag should bend what MA(1) can only tilt. Swept -0.05..0.10 on the training
    # windows: FLAT (17.443 / 17.406 / 17.420 / 17.478), optimum at 0 with a spread of 0.02% = noise.
    # So MA(1) already captures the exploitable part of the horizon profile and the remaining
    # per-interval curvature is not worth a parameter. Kept at 0; the code path stays because it is
    # the natural place for anyone with more statistical power to look again.
    "ma2": 0.0,          # second MA lag (crypto only); REFUTED as an improvement, see above
    # --- session-conditional innovation SHAPE (equity-like assets only) ---
    # The tokenized equity/commodity perps trade 24/7 but their UNDERLYING cash market does not, so
    # US market hours are a real structural feature of the process (unlike crypto, where there is no
    # session structure -- BTC is kept on the pooled path deliberately).
    # After the 168-bucket amplitude has equalized the SCALE (residual sd 0.86 open vs 1.05 weekend),
    # the residual SHAPE still differs enormously: excess kurtosis 8.6 open vs 173.9 weekend, and
    # P(exact zero) 2.6% open vs 33.6% weekend, averaged over the five equity-like assets. One pooled
    # set of innovations is dominated 4:1 by the closed sessions (8130 closed minutes per week vs
    # 1950 open), so every open-market step draws innovations that are far too spiky and far too
    # often exactly zero. CRPS is marginal-only, so each step's shape is scored directly.
    # Corroborated independently by equwhen.py: the com-equ deficit is a UNIFORM shift (median prompt
    # ratio ~= mean; worst 10% of prompts carry only 22-24% of the excess) and is worse on Mon-Thu
    # starts (NVDAX 1.158) than Fri-Sun starts (NVDAX 1.034) -- Mon-Thu horizons being the ones
    # densest in open-market minutes.
    # Pools live in the separate artifact sess.npz and are each standardized to unit variance, so
    # this is a pure SHAPE change: sigma still comes wholly from the forecast x amplitude path and
    # kcal stays valid. Sessions with too few observations fall back to the global pool.
    "sess_pool": 1.0,    # 1 = session-conditional innovation pools for equity-like assets
    # --- seasonal amplitude clip ---
    # fit_calib3 clipped the hour-of-week amplitude to [0.25, 3.0]. That guardrail was sized for
    # crypto (BTC's amplitude spans only 0.51-1.86, so it never binds) and was silently applied to
    # assets whose underlying market actually SHUTS. On XAU the floor binds on 33 of 168 buckets --
    # a fifth of the week -- where the true amplitude goes down to 0.152. For those 33 hours we were
    # simulating a fan ~64% too wide, on every path, at every horizon crossing them. TSLAX hits the
    # ceiling on 4 buckets (true max 3.16).
    # This is a SCALE error, distinct from the SHAPE error that sess_pool fixes, and season_pow
    # cannot reach it: raising a flat-bottomed curve to any power leaves it flat-bottomed.
    # season2.npz refits with clip [0.05, 5.0], everything else byte-identical to fit_calib3.
    # Only XAU (39% max change) and TSLAX (5%) differ at all; the other 11 assets are unchanged
    # by construction, which is why this is safe to enable globally.
    #
    # MEASURED EFFECT: NIL (+5.305% -> +5.306% over both com-equ windows; XAU itself 6.80 -> 6.77).
    # The reason is worth recording, because it is a general property of this design rather than a
    # quirk of XAU: the curve is applied TWICE and the two uses cancel. It divides the context's
    # returns to produce sigma_min, then multiplies forward to shape the path. Understating the
    # weekend amplitude inflates the deseasonalized context vol by almost exactly the factor by
    # which it then deflates the forward amplitude (XAU Saturday start: sigma_min 1.808 -> 1.888bp
    # while mean forward amp 0.257 -> 0.228, product 0.4639 -> 0.4305, a 7% net move from a 39%
    # curve error). A multiplicative seasonal factor is self-normalizing to first order.
    # Kept ON anyway: it is the correct curve, costs nothing at serve time, and leaves the model
    # honest for any future change that breaks the cancellation (e.g. a horizon-dependent sigma, or
    # any use of the curve that is not paired with a matching deseasonalization).
    "season2": 1.0,      # 1 = use the wider-clip seasonal curve from season2.npz (effect ~nil)
    # --- per-asset dispersion corrections on top of kcal (com-equ only) ---
    # pit.py showed the com-equ assets are mis-dispersed in OPPOSITE directions, which is why the
    # global `scale` sweep looked flat (0.85->1.08 moved the mean 0.8% against an 11.9% gap): it was
    # averaging two real errors of opposite sign, not finding a correct level.
    #   NVDAX  PIT at 30min 12.0/23.7/26.8/22.0/13.4 and at 180min 12.0/22.3/37.7/17.6/10.4 -- a
    #          clean central hump at EVERY horizon past 5min, i.e. systematically TOO WIDE. Its
    #          fitted kcal5 is 1.245, the largest of any asset; that was an over-correction.
    #   XAU    same direction, less extreme.
    #   GOOGLX/TSLAX are too NARROW at 24h (PIT 37.5/... and 51.5/..., 12.5% and 15.2% outside the
    #          ensemble's support) -- the opposite defect, so no single scale can serve both.
    # A too-wide ensemble at every horizon is a uniform per-prompt cost, which matches equwhen.py
    # (median prompt ratio = mean), is invisible to a drift test (nvdiag.py), and cannot be fixed by
    # better vol FORECASTING (oracle.py: perfect foresight worth 0.57% against NVDAX's 11.3% gap)
    # because it is a level error rather than a state-tracking error. That is the whole explanation
    # for the largest and most stubborn gap in the competition.
    # Fitted by kcal_equ.py on 07-16..07-21 and REQUIRED to replicate on the disjoint 07-21..07-26
    # holdout before adoption -- the test iteration 17's kcal refit failed. Only two of seven did:
    #   NVDAX k*=0.94  fit +0.25  holdout +0.28   ADOPT
    #   XAU   k*=0.88  fit +0.37  holdout +0.28   ADOPT
    #   GOOGLX 1.12 (holdout -0.06), WTIOIL 0.88 (holdout -0.51), SPCX 1.06 (-0.01) REJECTED as
    #   window-fitting; TSLAX and AAPLX were already optimal at 1.00.
    # Applied as a multiplier on top of kcal so the original artifact stays untouched.
    "kadj": 1.0,         # 0 = ignore the per-asset adjustments below, 1 = apply them
    # A LATE, WINDOW-INDEPENDENT re-test of the crypto 5-min dispersion level (iteration 39).
    # perinterval.py on crypto-24h found every one of the four scored intervals, on all five crypto
    # assets, wanting k > 1 -- 5min 1.076, 30min 1.100, 3hour 1.032, 24hour_abs 1.092. That is a
    # FLAT positive pattern, not the monotone tilt that MA(1) addresses, so it is a pure level
    # signal: kcal5 is calibrated slightly too tight for this regime. Confirmed by direct sweep:
    #   window A (fit 07-02..07-14 / hold 07-14..07-26) at 1.04: all five assets better on BOTH,
    #     BTC -0.036/-0.120, ETH -0.066/-0.070, SOL -0.043/-0.034, XRP -0.092/-0.158, HYPE -0.018/-0.095
    #   window B (fit 06-24..07-02) at 1.04: all five better on fit, -0.16% to -0.32%
    # Unlike iteration 30's per-asset refit (three of five reversed sign out of sample), the sign
    # here is uniform across every asset and every window, which is what distinguishes a real level
    # bias from window-fitting. Applied ONLY to the 5-minute increment: crypto-1h's 1-min kcal was
    # swept separately and sits at its own optimum, and iteration 38 showed that competition's
    # remaining headroom is elsewhere.
    # Refined from 1.04 to 1.09 (iteration 41). Re-running perinterval.py on the 1.04 predictions
    # showed the profile STILL asking for more (5min 1.068, 30min 1.100, 24hour_abs 1.054), i.e. the
    # first step was conservative. A finer sweep at nsim=500 confirmed 1.00 is worse than 1.04 on
    # every asset and BOTH windows (so the adoption itself replicates at higher precision), while
    # 1.09 beat 1.04 on 4 of 5 holdouts -- by margins inside the CRPS noise floor, which is why the
    # tie was broken on the OFFICIAL metric instead: rank 87 -> 76/257, Score term 0.1881 -> 0.1949.
    "kc5_crypto": 1.09,  # multiplier on kcal5 for crypto assets only (see above)
}

# Per-asset dispersion adjustments that replicated out of sample (see PARAMS["kadj"]).
_KADJ = {"NVDAX": 0.94, "XAU": 0.88}

# Assets whose feeds are zero-inflated / stale rather than continuously traded. These take
# PARAMS["ma1_equ"] instead of the crypto thetas.
_EQUITY_LIKE = frozenset({
    "AAPLX", "TSLAX", "GOOGLX", "NVDAX", "SPCX", "SP500", "XAU", "WTIOIL", "SPYX",
})


def _ma1(asset: str, step_min: float) -> float:
    """MA(1) coefficient for this asset and increment (see PARAMS notes)."""
    if asset in _EQUITY_LIKE:
        return float(PARAMS["ma1_equ"])
    return float(PARAMS["ma1_5"] if step_min >= 3.0 else PARAMS["ma1_1"])

_CACHE: dict | None = None


def _calib() -> dict:
    global _CACHE
    if _CACHE is None:
        if _ART.exists():
            z = np.load(_ART, allow_pickle=False)
            _CACHE = {k: z[k] for k in z.files}
        else:
            _CACHE = {}
    return _CACHE


def _get(asset: str, key: str, default=None):
    c = _calib()
    k = f"{asset}_{key}"
    return c[k] if k in c else default


_SESS_ART = _ART.parent / "sess.npz"
_SESS_CACHE: dict | None = None


def _sess_pools() -> dict:
    """Session-conditional innovation pools, from the separate sess.npz artifact."""
    global _SESS_CACHE
    if _SESS_CACHE is None:
        if _SESS_ART.exists():
            z = np.load(_SESS_ART, allow_pickle=False)
            _SESS_CACHE = {k: z[k] for k in z.files}
        else:
            _SESS_CACHE = {}
    return _SESS_CACHE


# US cash-equity session boundaries in UTC (regular hours 13:30-20:00 Mon-Fri).
_SESSIONS = ("open", "extended", "closed", "weekend")


def _session_of(ts: pd.Timestamp, n: int, step_minutes: float) -> np.ndarray:
    """Session label for each of n future steps starting at ts. Mirrors fit_sess.session_of."""
    base = ts.dayofweek * 24.0 + ts.hour + ts.minute / 60.0
    offs = base + np.arange(n) * (step_minutes / 60.0)
    dow = np.floor(offs / 24.0).astype(np.int64) % 7
    hod = offs % 24.0
    s = np.full(n, "closed", dtype=object)
    s[dow >= 5] = "weekend"
    wd = dow < 5
    s[wd & (hod >= 13.5) & (hod < 20.0)] = "open"
    s[wd & (((hod >= 9.5) & (hod < 13.5)) | ((hod >= 20.0) & (hod < 24.0)))] = "extended"
    return s


_SEASON2_ART = _ART.parent / "season2.npz"
_SEASON2_CACHE: dict | None = None


def _season2() -> dict:
    """Wider-clip hour-of-week curves (see PARAMS['season2'])."""
    global _SEASON2_CACHE
    if _SEASON2_CACHE is None:
        if _SEASON2_ART.exists():
            z = np.load(_SEASON2_ART, allow_pickle=False)
            _SEASON2_CACHE = {k: z[k] for k in z.files}
        else:
            _SEASON2_CACHE = {}
    return _SEASON2_CACHE


def _season_curve(asset: str) -> np.ndarray:
    if float(PARAMS["season2"]) > 0:
        s2 = _season2().get(f"{asset}_season")
        if s2 is not None and len(s2) == 168:
            return np.asarray(s2, dtype=float)
    s = _get(asset, "season")
    if s is None or len(s) != 168:
        return np.ones(168)
    return np.asarray(s, dtype=float)


def _hw_index(ts: pd.Timestamp, n: int, step_minutes: float) -> np.ndarray:
    """hour-of-week bucket for each of n future steps starting at ts."""
    base = ts.dayofweek * 24 + ts.hour + ts.minute / 60.0
    offs = base + np.arange(n) * (step_minutes / 60.0)
    return (np.floor(offs).astype(np.int64)) % 168


def _ewma_sigma(rets_deseas: np.ndarray, lam: float) -> float:
    """Trailing EWMA conditional sigma of the deseasonalized returns.

    Must mirror exactly how fit_calib3.py standardized the residual pool, otherwise the
    bootstrap's unit scale does not correspond to this sigma and the fan is mis-sized.
    """
    r = rets_deseas[np.isfinite(rets_deseas)]
    if r.size < 20:
        return float("nan")
    n = min(r.size, 4000)
    r = r[-n:]
    w = lam ** np.arange(n - 1, -1, -1)
    w /= w.sum()
    return float(np.sqrt(np.sum(w * r**2)))


def _forecast_sigma(asset: str, rets_deseas: np.ndarray) -> float:
    """Per-minute deseasonalized sigma: weighted blend of trailing RVs, shrunk to long-run."""
    weights = _get(asset, "har_w")
    long_run = _get(asset, "sd")
    long_run = float(long_run[0]) if long_run is not None else None

    rv = []
    for w in _HAR_WINDOWS:
        seg = rets_deseas[-w:]
        seg = seg[np.isfinite(seg)]
        # winsorize: a single feed glitch should not blow up the whole fan
        if seg.size >= 20:
            q = np.quantile(np.abs(seg), 0.995)
            seg = np.clip(seg, -max(q, 1e-9), max(q, 1e-9))
            rv.append(float(np.sqrt(np.mean(seg**2))))
        else:
            rv.append(np.nan)
    rv = np.asarray(rv)
    ok = np.isfinite(rv) & (rv > 0)
    if not ok.any():
        return long_run if long_run else 1e-4

    if weights is not None and len(weights) == len(_HAR_WINDOWS) + 1:
        # log-space linear model: log sigma = w0 + sum wi * log rv_i  (missing -> longest available)
        lrv = np.log(np.where(ok, rv, np.nan))
        fill = np.nanmax(lrv) if np.isfinite(lrv).any() else np.log(long_run or 1e-4)
        lrv = np.where(np.isfinite(lrv), lrv, fill)
        sig = float(np.exp(weights[0] + np.dot(weights[1:], lrv)))
    else:
        sig = float(np.exp(np.mean(np.log(rv[ok]))))

    if long_run and long_run > 0:
        # shrink in log space toward the asset's long-run vol; caps runaway forecasts
        k = float(PARAMS["shrink"])
        sig = float(np.exp(k * np.log(sig) + (1.0 - k) * np.log(long_run)))
        sig = float(np.clip(sig, 0.25 * long_run, 4.0 * long_run))
    return max(sig, 1e-9) * float(PARAMS["scale"])


def _pool_for(asset: str, step_min: float):
    """Residual pool built AT the scored increment, plus its scaling ratio vs sqrt-time.

    Bootstrapping minute-shaped innovations and scaling by sqrt(step) mis-specifies the marginal
    that CRPS actually sees: temporal aggregation thins tails, so a 1-min pool scaled to 5 min
    implied kurtosis 329 vs a real 93 on SP500 (1435 vs 230 on AAPLX). We therefore key the pool
    on the increment. Only 1m and 5m are ever requested (CRYPTO_1H / the two 24h competitions).
    """
    key = 5 if step_min >= 3 else 1
    prefix = "poolc" if float(PARAMS["filtered"]) > 0 else "pool"
    pool = _get(asset, f"{prefix}{key}")
    if pool is None:
        pool = _get(asset, f"pool{key}")
    ratio = _get(asset, f"ratio{key}")
    ratio = float(ratio[0]) if ratio is not None else 1.0
    if pool is None:  # legacy artifact
        pool = _get(asset, "pool")
    return pool, ratio


def _kcal(asset: str, step_min: float) -> float:
    """Per-(asset, increment) CRPS-optimal dispersion multiplier, fitted out-of-sample.

    Two separate facts justify this, and they point in OPPOSITE directions, so no single global
    `scale` can capture them:

      * On crypto the HAR forecast is biased HIGH. realized/forecast sigma has median 0.85 in
        every half-month from 2026-03 to 2026-07 (never above 0.96) -- a level bias, not a
        property of the quiet late-July window. CRPS-optimal scale is 0.85 (crypto-24h) / 0.80
        (crypto-1h), worth 0.8-1.3%.
      * On the equity-like assets it is biased LOW (realized/forecast median 1.00, CRPS-optimal
        ~1.15), because their returns are zero-inflated and tick-discrete: a third of SP500's
        5-min changes are under 1bp, which drags the RV estimate below the dispersion the
        continuous part of the distribution actually needs.

    CRPS is also asymmetric -- under-dispersion is punished harder than over-dispersion -- so the
    fit deliberately targets the CRPS optimum directly rather than a variance match.
    Fitted by agent/research/fit_kcal.py on a TRAINING window disjoint from every window used to
    report results.
    """
    if float(PARAMS["use_kcal"]) <= 0:
        return 1.0
    k = _get(asset, f"kcal{5 if step_min >= 3 else 1}")
    if k is None:
        return 1.0
    k = float(k[0])
    if float(PARAMS["kadj"]) > 0:
        k *= _KADJ.get(asset, 1.0)
    # 5-minute crypto level correction (iteration 39). Equities are excluded: their dispersion bias
    # runs the other way (zero-inflated, tick-discrete feeds) and is already handled by kcal/_KADJ.
    if step_min >= 3 and asset not in _EQUITY_LIKE:
        k *= float(PARAMS["kc5_crypto"])
    return k


def _draw_innovations(pool, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Unit-scale fat-tailed innovations, STRATIFIED across paths at each step.

    This is a pure estimator-variance win, available because of how the validator scores. Each
    scoring interval compares the realized change over a non-overlapping block against the
    ensemble's distribution of that same block, and CRPS is computed from the ensemble's EMPIRICAL
    CDF. A finite random sample's CDF is rougher than the distribution it came from, which biases
    CRPS upward by O(1/M) -- we measured ~0.45% at M=250.

    Drawing each step's M innovations as a stratified sample (one from each 1/M probability bin,
    randomly permuted across paths) makes every step's marginal a low-discrepancy representation
    of the target law instead of a random one, at identical cost. The marginals are what is
    scored, so this removes most of that bias.

    Stratification is per-step and the permutation is independent per step, so increments stay
    independent across time: the sums the coarser intervals score (30min = 6 steps, 3hour = 36)
    are unaffected in distribution, and inherit a smaller variance reduction of their own.
    """
    nsim, n_future = shape
    # stratified uniforms: bin i gets one draw from [i/M, (i+1)/M), independently permuted per step
    u = (rng.random((nsim, n_future)) + np.arange(nsim)[:, None]) / nsim
    for j in range(n_future):
        rng.shuffle(u[:, j])

    if pool is not None and len(pool) >= 5000:
        p = np.sort(np.asarray(pool, dtype=float))
        # empirical quantile function of the residual pool
        z = p[np.clip((u * len(p)).astype(np.int64), 0, len(p) - 1)]
    else:
        from scipy.stats import t as _t
        nu = 3.0
        z = _t.ppf(u, nu) / np.sqrt(nu / (nu - 2.0))
    return z


def _zero_prob(asset: str, rets: np.ndarray, step_min: float) -> float:
    """P(next change is EXACTLY zero), conditioned on how stale the feed has just been.

    The equity-like feeds are tick-discrete AND go stale: the unconditional P(5-min change = 0)
    is 0.105 on AAPLX and 0.115 on XAU, far above what tick rounding alone implies (tick 0.33bp
    against a 9.7bp sd would give ~0.03). Crucially the atom is highly PREDICTABLE -- conditioning
    on the number of zero returns in the previous 12 bars moves AAPLX's P(zero) from 0.043 to
    0.514 -- so a state-dependent mixture weight is worth much more than a constant one.

    A continuous ensemble pays ~0.234*sigma of CRPS at every one of those points, and the 5-min
    term is ~61% of the 24h score across 288 points, so this is the largest single structural
    misspecification left for AAPLX/XAU.

    Returns a probability blended between the asset's unconditional rate (from the artifact) and
    the trailing empirical rate, then damped by `zi_mix`.
    """
    w = float(PARAMS["zi_mix"])
    if w <= 0:
        return 0.0
    q0 = _get(asset, f"zq{5 if step_min >= 3 else 1}")
    if q0 is None:
        return 0.0
    q_uncond = float(q0[0])
    if q_uncond < 0.005:            # crypto: no meaningful atom, skip the machinery entirely
        return 0.0
    # The recent rate MUST be measured on the increment we are about to simulate: the 1-min zero
    # rate is 0.43 on AAPLX against 0.13 at 5-min, so reusing the minute series here inflates q
    # by ~3x and makes the mixture actively harmful.
    step = max(int(round(step_min)), 1)
    r = rets[np.isfinite(rets)]
    if r.size >= 12 * step:
        agg = r[-12 * step:].reshape(12, step).sum(axis=1)   # minute log-rets -> step log-rets
        recent = float(np.mean(agg == 0.0))
        q = 0.5 * q_uncond + 0.5 * recent
    else:
        q = q_uncond
    return float(np.clip(w * q, 0.0, 0.9))


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = int(time_length) // int(time_increment) + 1
    n_future = steps - 1

    ctx = pd.Series(context_prices).dropna()
    last = float(ctx.iloc[-1])

    # ---- deseasonalize the context's minute returns before measuring vol ----
    season = _season_curve(asset)
    if isinstance(ctx.index, pd.DatetimeIndex) and len(ctx) > 1:
        lp = np.log(ctx.to_numpy(dtype=float))
        rets = np.diff(lp)
        idx = ctx.index[1:]
        hw = (idx.dayofweek * 24 + idx.hour).to_numpy()
        # per-minute returns: divide out the seasonal amplitude
        fac = season[hw % 168]
        rets_d = rets / np.where(fac > 0, fac, 1.0)
        start_ts = pd.Timestamp(idx[-1])
    else:
        rets = np.diff(np.log(np.asarray(ctx, dtype=float))) if len(ctx) > 1 else np.array([])
        rets_d = rets
        start_ts = pd.Timestamp(start_time)

    rets_d = rets_d[np.isfinite(rets_d)]
    sigma_min = _forecast_sigma(asset, rets_d)

    # The residual pool was standardized by a trailing EWMA conditional vol, so blending that
    # same statistic in keeps the bootstrap's unit scale meaningful. HAR alone predicts the
    # NEXT 24h better; EWMA matches the pool's construction. Mix in log space.
    mix = float(PARAMS["ewma_mix"])
    if mix > 0:
        lam = _get(asset, "lam")
        lam = float(lam[0]) if lam is not None else 0.97
        se = _ewma_sigma(rets_d, lam)
        if np.isfinite(se) and se > 0:
            sigma_min = float(np.exp((1 - mix) * np.log(sigma_min) + mix * np.log(se)))

    # ---- forward seasonal amplitude per step of the requested grid ----
    step_min = float(time_increment) / 60.0
    hw_f = _hw_index(start_ts, n_future, step_min)
    amp = season[hw_f] ** float(PARAMS["season_pow"])  # multiplier on per-minute sigma
    # per-step sigma: t**hurst scaling of the deseasonalized per-minute vol (0.5 = random walk),
    # times the empirical correction for how this asset's vol really scales to this increment.
    pool, ratio = _pool_for(asset, step_min)
    step_sigma = sigma_min * amp * step_min ** float(PARAMS["hurst"]) * ratio * _kcal(asset, step_min)
    # sigma of the (unobserved) step just BEFORE the horizon, for the MA(1) lag term below
    step_sigma_1step = float(step_sigma[0])

    rng = np.random.default_rng()
    nsim = int(num_simulations)
    z = _draw_innovations(pool, (nsim, n_future), rng)

    # --- session-conditional innovation shape (equity-like assets only; see PARAMS) ---
    # Redraw each step from the pool of its OWN session. Pure shape substitution: every session
    # pool is unit-variance, so step_sigma (and therefore kcal) is untouched. Steps whose session
    # has no fitted pool keep the global draw above.
    if float(PARAMS["sess_pool"]) > 0 and asset in _EQUITY_LIKE and n_future > 0:
        sp = _sess_pools()
        pkey = 5 if step_min >= 3 else 1
        labels = _session_of(start_ts, n_future, step_min)
        for name in _SESSIONS:
            cols = np.flatnonzero(labels == name)
            if cols.size == 0:
                continue
            spool = sp.get(f"{asset}_pool{pkey}_{name}")
            if spool is None or len(spool) < 5000:
                continue  # fall back to the global pool for this session
            z[:, cols] = _draw_innovations(spool, (nsim, cols.size), rng)

    # --- stochastic volatility: per-path log-vol level offset + a log-vol random walk ---
    # Both are variance-neutral (mean-1 in level space) so the ensemble's total variance is
    # unchanged; they widen the tails of the MARGINAL at each step, and the random walk makes
    # that widening grow with horizon -- which is exactly what a static sigma cannot do.
    sv_v = float(PARAMS["sv_vol"])
    sv_l = float(PARAMS["sv_level"])
    if sv_v > 0 or sv_l > 0:
        logm = np.zeros((nsim, n_future))
        if sv_v > 0:
            logm += np.cumsum(rng.normal(0.0, sv_v, size=(nsim, n_future)), axis=1)
        if sv_l > 0:
            logm += rng.normal(0.0, sv_l, size=(nsim, 1))
        logm -= 0.5 * np.var(logm, axis=0, keepdims=True)  # keep per-step variance neutral
        step_sigma = step_sigma[None, :] * np.exp(logm)
        incr = z * step_sigma
    else:
        incr = z * step_sigma[None, :]

    # --- zero atom: a fraction q of increments are set to EXACTLY zero ---
    # Applied on the SCORED grid, because that is the marginal CRPS sees. The survivors are
    # re-inflated by 1/sqrt(1-q) so the per-step variance is unchanged: the mixture reallocates
    # probability mass to the atom rather than shrinking the distribution.
    q = _zero_prob(asset, rets, step_min)
    if q > 0:
        keep = rng.random(incr.shape) >= q
        incr = incr * keep / np.sqrt(1.0 - q)

    # --- MA(1) mean reversion: inflate the 1-step marginal, deflate the cumulative windows ---
    # incr is already scaled to step_sigma, so filter and renormalize by sqrt(1+theta^2) to leave
    # the one-step variance exactly where kcal calibrated it. The lagged term needs one extra
    # innovation before step 0 so that step 0 is filtered like every other step (otherwise the
    # first increment -- 27.6% of the crypto-1h score via the 1min interval -- keeps the
    # unfiltered variance and the whole point is lost on the highest-weight term).
    # ma2 extends this to a second lag: MA(1) is one parameter fitted to a smooth monotone curve
    # (optimal k runs 1.10 at 1min -> 0.90 at 60min), so a second lag can bend the profile where
    # MA(1) can only tilt it. Same variance normalization, now sqrt(1 + th^2 + th2^2).
    th = _ma1(asset, step_min)
    th2 = 0.0 if asset in _EQUITY_LIKE else float(PARAMS["ma2"])
    if th != 0.0 or th2 != 0.0:
        nlag = 2 if th2 != 0.0 else 1
        lag = _draw_innovations(pool, (nsim, nlag), rng) * step_sigma_1step
        ext = np.concatenate([lag, incr], axis=1)          # nlag pre-horizon innovations
        out = ext[:, nlag:] - th * ext[:, nlag - 1: -1]
        if th2 != 0.0:
            out = out - th2 * ext[:, nlag - 2: -2]
        incr = out / np.sqrt(1.0 + th * th + th2 * th2)

    logp = np.log(last) + np.cumsum(incr, axis=1)
    paths = np.empty((int(num_simulations), steps), dtype=float)
    paths[:, 0] = last
    paths[:, 1:] = np.exp(logp)

    return (start_time, time_increment, *[p.tolist() for p in paths])
