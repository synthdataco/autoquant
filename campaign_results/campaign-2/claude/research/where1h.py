"""WHERE do we lose to the crypto leaders? Per-prompt decomposition against the real field.

MOTIVATION. equwhen.py answered this for com-equ-24h and the answer (a UNIFORM shift, not a few
bad prompts) is what redirected that competition's work from "find the bad prompts" to "fix the
dispersion level". The same diagnostic has NEVER been run on crypto -- which, per iteration 27, is
where the entire remaining Score gradient lives.

The question is worth asking because the two possible answers imply completely different work:

  * UNIFORM excess (our CRPS/leader ratio roughly constant across prompts) => a calibration-level
    defect, addressable by a better marginal, and essentially what we have already been doing.
  * CONCENTRATED excess (a minority of prompts carrying most of the deficit) => a STATE-dependent
    defect. We would be fine in normal conditions and badly wrong in an identifiable regime, and
    the fix is a conditional model, not a better constant.

We measure against the top individual miners on the FULL-COVERAGE panel (never `field_min`, which
report 7.1 shows is an unachievable order statistic sitting 14-17% below the best real miner).

Outputs, per asset:
  * our mean CRPS, the leader's, the ratio;
  * how the excess distributes over prompts -- the share of total excess carried by the worst
    decile / quartile of prompts, vs the 10%/25% a uniform defect would give;
  * the ratio by quintile of a set of candidate STATE variables known at prompt time (no
    lookahead): trailing realized vol, the vol RATIO (recent/forecast, i.e. is vol rising or
    falling), and the level of our own forecast sigma.

That last block is the actionable part: if the excess concentrates in, say, the top quintile of
"vol is rising", then there is a conditional defect with a named trigger.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/agent/research")
sys.path.insert(0, "/workspace/agent")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE")
ap.add_argument("--start", default="2026-07-14")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--nsim", type=int, default=1000)
ap.add_argument("--ntop", type=int, default=5, help="how many top miners define 'the leaders'")
ap.add_argument("--mod", default="modeling")
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]


def state_vars(asset: str, t: pd.Timestamp) -> dict:
    """Candidate conditioning variables, all computable at prompt time (strictly no lookahead)."""
    ctx = harness.context(asset, t)
    if len(ctx) < 2000:
        return {}
    lp = np.log(ctx.to_numpy(dtype=float))
    r = np.diff(lp)
    r = r[np.isfinite(r)]
    if r.size < 2000:
        return {}
    # trailing realized vol over the last hour and the last day
    rv_1h = float(np.sqrt(np.mean(r[-60:] ** 2)))
    rv_1d = float(np.sqrt(np.mean(r[-1440:] ** 2)))
    rv_7d = float(np.sqrt(np.mean(r**2)))
    return {
        "rv_1h": rv_1h,
        "rv_1d": rv_1d,
        # is volatility RISING or FALLING going into the prompt? The single most likely
        # state-dependent defect: a trailing-window forecast lags a turning point in both
        # directions, and CRPS punishes the under-dispersed direction much harder.
        "vol_ratio_1h_1d": rv_1h / rv_1d if rv_1d > 0 else np.nan,
        "vol_ratio_1d_7d": rv_1d / rv_7d if rv_7d > 0 else np.nan,
    }


print(f"=== {a.slug}  {a.start} -> {a.end}  nsim={a.nsim} ===\n")
for asset in a.assets.split(","):
    res = harness.evaluate(mod.simulate, a.slug, asset, a.start, a.end, nsim=a.nsim)
    if res is None or res.empty:
        print(f"{asset}: no results")
        continue
    res = res.dropna(subset=["crps"]).copy()
    res["start_time"] = pd.to_datetime(res.start_time, utc=True)

    panel = harness.field_panel(a.slug, asset, prompts=res.start_time)
    common = panel.index.intersection(pd.DatetimeIndex(res.start_time))
    if len(common) < 20 or panel.shape[1] < 20:
        print(f"{asset}: panel too thin (n={len(common)}, miners={panel.shape[1]})")
        continue
    panel = panel.loc[common]
    ours = res.set_index("start_time").loc[common, "crps"]

    means = panel.mean(axis=0).sort_values()
    leaders = list(means.index[: a.ntop])
    # "the leader" = the single best miner by mean over these prompts; also a top-N average,
    # which is a lower-variance target and less likely to be the luckiest of a pack of equals.
    lead1 = panel[means.index[0]]
    leadN = panel[leaders].mean(axis=1)

    rank = int((means.to_numpy() < ours.mean()).sum()) + 1
    print(f"--- {asset}  n={len(common)} prompts, panel={panel.shape[1]} miners ---")
    print(f"  ours {ours.mean():.4f}   best {means.iloc[0]:.4f}  (+{(ours.mean()/means.iloc[0]-1)*100:.3f}%)"
          f"   top{a.ntop} {leadN.mean():.4f}  (+{(ours.mean()/leadN.mean()-1)*100:.3f}%)"
          f"   rank {rank}/{len(means)}")

    # ---- how is the excess distributed over prompts? ----
    excess = (ours - leadN).to_numpy()
    tot = excess.sum()
    order = np.argsort(excess)[::-1]          # worst prompts first
    n = len(excess)
    for frac in (0.10, 0.25, 0.50):
        k = max(int(round(frac * n)), 1)
        share = excess[order[:k]].sum() / tot if tot != 0 else np.nan
        print(f"    worst {int(frac*100):>2d}% of prompts carry {share*100:6.1f}% of the excess"
              f"   (uniform would be {int(frac*100)}%)")
    # a uniform-shift defect also implies the per-prompt RATIO is centred and tight
    ratio = (ours / leadN).to_numpy()
    ratio = ratio[np.isfinite(ratio)]
    print(f"    per-prompt ratio: mean {ratio.mean():.4f}  median {np.median(ratio):.4f}"
          f"  p10 {np.quantile(ratio,0.10):.4f}  p90 {np.quantile(ratio,0.90):.4f}"
          f"  frac>1 {np.mean(ratio>1)*100:.1f}%")

    # ---- does the excess condition on a state variable known at prompt time? ----
    sv = {t: state_vars(asset, t) for t in common}
    sv = {t: v for t, v in sv.items() if v}
    if len(sv) > 40:
        svdf = pd.DataFrame(sv).T
        exc = (ours - leadN).loc[svdf.index]
        rat = (ours / leadN).loc[svdf.index]
        for col in svdf.columns:
            q = pd.qcut(svdf[col], 5, labels=False, duplicates="drop")
            byq = pd.DataFrame({"q": q, "exc": exc.to_numpy(), "rat": rat.to_numpy()})
            g = byq.groupby("q")
            rr = g["rat"].mean()
            ee = g["exc"].sum() / exc.sum()
            print(f"    {col:>16s}  ratio by quintile: " + " ".join(f"{v:.4f}" for v in rr)
                  + "   excess share: " + " ".join(f"{v*100:4.1f}%" for v in ee))
    print()
