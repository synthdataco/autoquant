"""Do DESEASONALIZED innovations have a different SHAPE by market session, for equity-like assets?

Motivation. The tokenized equity perps (AAPLX, NVDAX, TSLAX, GOOGLX) and SPCX/SP500 trade 24/7, but
their UNDERLYING cash market does not. The campaign notes warn against session-of-day structure --
that warning is about crypto, where it is genuinely absent. For a token that tracks NVDA, US market
hours are a real structural feature of the price process, not a calendar artifact.

modeling.py already handles the SCALE of this: a 168-bucket hour-of-week amplitude narrows the fan
overnight and on weekends. What it does NOT handle is SHAPE. Innovations are bootstrapped from ONE
global pool of deseasonalized standardized minute returns, so a step that falls inside a closed-market
hour can draw an innovation that actually happened at the opening bell, and vice versa.

That matters here specifically because CRPS is MARGINAL-ONLY: the score at step t sees only the
ensemble's marginal distribution at t. Getting each step's marginal SHAPE right is scored directly;
there is no cross-step structure to hide behind. If closed-hour returns are (after dividing out their
own smaller amplitude) much more zero-inflated and more leptokurtic than open-hour returns, then the
global pool is simultaneously too fat overnight and too thin at the open, on every single path.

This script only measures. It computes, per asset, the deseasonalized standardized minute returns
exactly the way modeling.py builds its pool, then splits them by session and compares the shape
statistics that CRPS actually feels:
  * excess kurtosis and the zero fraction -- the two ways a marginal can be mis-shaped at fixed sigma
  * the 90/99th percentile of |z| -- the tail region where CRPS pays most
  * residual sd by session -- a sanity check that the 168-bucket deseasonalization already did its
    job on SCALE. If sd is ~1 in every session, any remaining difference is purely shape, which is
    exactly the part the current model cannot express.

Sessions (UTC): US cash equities 13:30-20:00 Mon-Fri.
  open    = weekday 13:30-20:00
  premarket/afterhours = weekday, adjacent 4h either side
  closed  = weekday, rest
  weekend = Sat/Sun
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='NVDAX,TSLAX,GOOGLX,AAPLX,SPCX,XAU,WTIOIL,BTC')
ap.add_argument('--end', default='2026-07-27')
a = ap.parse_args()


def session_of(idx: pd.DatetimeIndex) -> np.ndarray:
    dow = idx.dayofweek.to_numpy()
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    s = np.full(len(idx), 'closed', dtype=object)
    s[(dow >= 5)] = 'weekend'
    wd = dow < 5
    s[wd & (hod >= 13.5) & (hod < 20.0)] = 'open'
    s[wd & (((hod >= 9.5) & (hod < 13.5)) | ((hod >= 20.0) & (hod < 24.0)))] = 'extended'
    return s


print("deseasonalized standardized minute returns, split by US cash session")
print("sd~1 everywhere => the 168-bucket amplitude already fixed SCALE; what remains is SHAPE\n")
print(f"{'asset':8s}{'session':10s}{'n':>9s}{'sd':>7s}{'exkurt':>9s}{'zero%':>8s}"
      f"{'p90|z|':>8s}{'p99|z|':>8s}")

summary = []
for asset in a.assets.split(','):
    try:
        px = harness.context(asset, pd.Timestamp(a.end, tz='UTC'))
    except Exception as e:
        print(f"{asset:8s} context failed: {e}"); continue
    if px is None or len(px) < 5000:
        print(f"{asset:8s} too little context"); continue
    s = px['price'] if isinstance(px, pd.DataFrame) else px
    s = s.dropna()
    idx = s.index
    r = s.pct_change().to_numpy()[1:]
    idx = idx[1:]
    # divide out the same 168-bucket seasonal amplitude modeling.py uses
    try:
        curve = modeling._season_curve(asset)
        hw = (idx.dayofweek * 24 + idx.hour).to_numpy()
        amp = curve[hw % len(curve)]
        z = r / np.where(amp > 0, amp, 1.0)
    except Exception:
        z = r
    z = z / np.nanstd(z)
    sess = session_of(idx)
    for name in ['open', 'extended', 'closed', 'weekend']:
        m = (sess == name) & np.isfinite(z)
        n = int(m.sum())
        if n < 500:
            continue
        v = z[m]
        sd = float(np.std(v))
        k = float(((v - v.mean()) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else np.nan
        zero = float((v == 0).mean() * 100)
        av = np.abs(v)
        print(f"{asset:8s}{name:10s}{n:9d}{sd:7.2f}{k:9.1f}{zero:7.1f}%"
              f"{np.percentile(av,90):8.2f}{np.percentile(av,99):8.2f}")
        summary.append(dict(asset=asset, sess=name, n=n, sd=sd, kurt=k, zero=zero))
    print()

if summary:
    d = pd.DataFrame(summary)
    eq = d[d.asset.isin(['NVDAX', 'TSLAX', 'GOOGLX', 'AAPLX', 'SPCX'])]
    print("=" * 74)
    print("equity-like average by session (the pooled model gives all of these ONE shape):")
    if not eq.empty:
        g = eq.groupby('sess')[['sd', 'kurt', 'zero']].mean()
        print(g.round(2).to_string())
        if 'open' in g.index and 'weekend' in g.index:
            print(f"\nzero-fraction open->weekend: {g.loc['open','zero']:.1f}% -> "
                  f"{g.loc['weekend','zero']:.1f}%")
            print(f"residual sd  open->weekend: {g.loc['open','sd']:.2f} -> {g.loc['weekend','sd']:.2f}")
            print("\nIf sd is already ~equal but zero%/kurt differ a lot, a session-conditional")
            print("innovation POOL is the missing piece and costs nothing at serve time.")
            print("If sd itself still differs a lot, the 168-bucket amplitude is underfitting SCALE,")
            print("which would be a simpler and even cheaper fix.")
    print("\ncrypto control (BTC) -- expect NO session structure; if BTC shows the same pattern")
    print("the effect is an artifact of my deseasonalization, not a market-hours effect:")
    print(d[d.asset == 'BTC'][['sess', 'sd', 'kurt', 'zero']].round(2).to_string(index=False))
