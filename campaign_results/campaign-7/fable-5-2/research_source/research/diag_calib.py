"""Calibration of the fitted variance path by session x horizon (validation period): mean(|r|/sqrt(v)) per cell
relative to the asset's overall mean. 1.00 = same relative width everywhere; >1 = fan too narrow there."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
import numpy as np, torch, core
from pathlib import Path
import train as T
pdir, ds, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assets = sys.argv[4].split(",")
vs, ve = T.minute_stamp(sys.argv[5]), T.minute_stamp(sys.argv[6])
np.set_printoptions(linewidth=200, precision=3, suppress=True)
hb = [(0, 12), (12, 48), (48, 144), (144, 288)] if tag == "24h" else [(0, 5), (5, 15), (15, 30), (30, 60)]
for a in assets:
    d = T.Data(ds / f"{a}_{tag}.npz", tag, lambda t: (t >= vs) & (t < ve))
    par = json.loads((pdir / f"model_{a}_{tag}.json").read_text())
    m = T.Model(tag).to(T.DEV)
    m.load_raw(par["raw"])
    with torch.no_grad():
        v = m.variance(d, torch.arange(d.n, device=T.DEV)).cpu().numpy()
    inc = T.CFG[tag]["inc"]
    r = d.ret[1].cpu().numpy() / 1e4
    sess = d.sess.cpu().numpy()[:, inc - 1::inc]
    z = np.abs(r) / np.sqrt(v)
    base = z.mean()
    tab = np.array([[z[:, lo:hi][sess[:, lo:hi] == c].mean() / base if (sess[:, lo:hi] == c).sum() > 500 else np.nan for lo, hi in hb] for c in range(4)])
    share = np.array([np.abs(r)[sess == c].sum() / np.abs(r).sum() for c in range(4)])
    print(f"{a} {tag} n={d.n} E|z|={base:.3f} rms z={np.sqrt((z**2).mean()):.3f} | rows=session(core,shoulder,night,weekend) cols=horizon buckets | |r| share by session {share}")
    print(tab)
