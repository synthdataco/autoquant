"""Fit the simulator's parameters on the scoring rule itself (differentiable simulation, GPU).

The loss replays synth.validator.crps_calculation in torch (same intervals, same ensemble-CRPS
formula, the 1/N spread term set to the validator's N=1000) on simulated paths; it is a TRAINING
loss only. Every reported CRPS comes from agent/predict.py.

Usage: uv run python research/train.py --ds scratch/ds_oos --out scratch/params_oos --train-end 2026-07-01 --val-start 2026-07-24
"""
import sys, json, argparse, time, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
import numpy as np, torch
from pathlib import Path
import core

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F = core.N_FEAT; NG = core.N_GLOB; NE = core.N_EXTRA
CFG = {"24h": dict(inc=5, T=1440, ks=(1, 6, 36)), "1h": dict(inc=1, T=60, ks=(1, 2, 5, 15, 30))}
VOL_BLOCKS = ((60, 5.25 / 3), (15, 5.25 / 12), (5, 5.25 / 36))

def minute_stamp(day):
    return int(np.datetime64(day, "m").astype(np.int64))

class Data:
    def __init__(self, path, tag, sel):
        z = np.load(path)
        cfg = CFG[tag]; inc = cfg["inc"]
        idx = np.flatnonzero(sel(z["t0"]))
        self.n = len(idx); self.tag = tag
        lr = z["lr"][idx]
        g = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)
        self.x, self.xs = g(z["x"][idx]), g(z["xs"][idx])
        self.seas, self.sess = g(z["seas"][idx]), g(z["sess"][idx].astype(np.int64), torch.long)
        self.ss = self.sess[:, inc - 1::inc].contiguous()          # session of each step (its last minute)
        self.xzs = torch.gather(g(z["xz"][idx]), 1, self.ss)       # zero-return share of that session in the context
        xsl = g(z["xsl"][idx])                                     # [n, 4, 2]
        self.xsl = torch.gather(xsl, 1, self.ss[:, :, None].expand(-1, -1, xsl.shape[2]))   # [n, H, 2] the step's own session level
        self.cross = self.ss != g(z["cur"][idx].astype(np.int64), torch.long)[:, None]
        self.xe = g(z["xe"][idx])
        self.ret = {k: g(np.expm1(lr[:, k::k] - lr[:, :-k:k]) * 1e4) for k in cfg["ks"]}
        self.lrF = g(lr[:, -1])
        if tag == "1h":
            self.gap = g(np.expm1(lr[:, 5::5] - lr[:, :1]) * 1e4)
            r1 = np.expm1(np.diff(lr, axis=1)) * 1e4
            self.vol = {bs: g(r1.reshape(len(idx), 60 // bs, bs).std(axis=2, ddof=1)) for bs, _ in VOL_BLOCKS}

class Model(torch.nn.Module):
    def __init__(self, tag, init=None, freeze=None, mask_feats=None):
        super().__init__()
        cfg = CFG[tag]; self.tag = tag; self.inc = cfg["inc"]; self.H = cfg["T"] // cfg["inc"]; self.ks = cfg["ks"]
        a0 = np.full(F, -3.0)
        for fam in range(3): a0[fam * core.N_HL + 2: fam * core.N_HL + 7] = 0.0
        a0[NG:] = -1.0
        inv_sp = lambda v: math.log(math.expm1(v))
        raw = dict(alpha=a0, beta=np.zeros(F), fam=np.zeros(2), b0=0.0, b1=0.0, kappa=0.0,
                   A0=inv_sp(0.15), A1=inv_sp(0.05), c=inv_sp(0.2), rho=2.0, lam=np.full(4, inv_sp(0.3)), theta=0.0,
                   dlt=np.zeros(3), dlt1=np.zeros(3), gz=0.0, gl=0.0, p=0.1, gamma=inv_sp(0.5), ce0=np.zeros(NE), ce1=np.zeros(NE))
        if init: raw.update(init)
        self.p = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(np.atleast_1d(v), dtype=torch.float32)) for k, v in raw.items()})
        self.freeze = freeze or {}
        self.mask = torch.zeros(F, device=DEV)
        if mask_feats is not None: self.mask[torch.tensor(mask_feats, device=DEV)] = -30.0
        self.fast = torch.tensor(core.FAST_MASK, dtype=torch.float32, device=DEV)
        ahead = (torch.arange(self.H, device=DEV) + 0.5) * self.inc
        self.phi = torch.log1p(ahead / core.PHI_SCALE)

    def c(self, name):
        """Constrained value of a scalar parameter."""
        if name in self.freeze: return torch.tensor(self.freeze[name], device=DEV)
        v = self.p[name]
        if name in ("A0", "A1", "c"): return torch.nn.functional.softplus(v)
        if name == "rho": return torch.sigmoid(v)
        if name == "kappa": return 1.5 * torch.sigmoid(v) - 0.75 + 0.0 * v
        if name == "theta": return 0.5 * torch.tanh(v)
        if name == "gamma": return torch.nn.functional.softplus(v)
        if name == "p":
            pv = 1.5 * torch.tanh(v)
            return torch.where(pv.abs() < 0.02, torch.full_like(pv, 0.02), pv)
        return v

    def export(self):
        g = lambda t: [float(v) for v in t.detach().cpu().numpy().ravel()]
        out = {"alpha": g(self.p["alpha"] + self.mask), "beta": g(self.p["beta"]), "fam_bias": [0.0] + g(self.p["fam"]),
               "dlt": [0.0] + g(self.dl("dlt")), "dlt1": [0.0] + g(self.dl("dlt1")), "lam_raw": g(self.p["lam"])}
        for k in ("b0", "b1", "kappa", "A0", "A1", "c", "rho", "theta", "gz", "gl", "p", "gamma"): out[k] = float(self.c(k))
        out["ce0"], out["ce1"] = g(self.dl("ce0")), g(self.dl("ce1"))
        out["raw"] = {k: g(v) for k, v in self.p.items()}
        return out

    def load_raw(self, raw):
        with torch.no_grad():
            for k, v in raw.items(): self.p[k].copy_(torch.tensor(v, device=DEV).view_as(self.p[k]))

    def dl(self, name):
        return self.p[name] * (0.0 if (name in self.freeze or name[:2] in self.freeze) else 1.0)

    def lam(self, d, idx):
        return torch.nn.functional.softplus(self.p["lam"][d.ss[idx]] + self.c("gz") * d.xzs[idx])

    def variance(self, d, idx):
        x, xs, seas, sess = d.x[idx], d.xs[idx], d.seas[idx], d.sess[idx]
        B = len(idx)
        mv = seas * torch.exp(self.c("kappa") * torch.gather(xs, 1, sess))
        s = mv.view(B, self.H, self.inc).sum(-1)
        z1 = torch.zeros(1, device=DEV)
        fam = torch.cat([torch.zeros(core.N_HL, device=DEV), self.p["fam"][0].repeat(core.N_HL), self.p["fam"][1].repeat(core.N_HL), z1])
        feat = torch.cat([(x + fam)[:, None, :].expand(B, self.H, NG), d.xsl[idx]], -1)          # [B,H,F]
        logits = (self.p["alpha"] + self.mask)[None, :] + self.p["beta"][None, :] * self.phi[:, None]
        lw_same = torch.log_softmax(logits, -1)
        lw_cross = torch.log_softmax(logits - self.c("gamma") * self.fast[None, :], -1)
        lw = torch.where(d.cross[idx][:, :, None], lw_cross[None], lw_same[None])
        p = self.c("p")
        level = torch.logsumexp(lw + p * feat, -1) / p + self.c("b0") + self.c("b1") * self.phi
        ss = d.ss[idx]
        level = level + torch.cat([z1, self.dl("dlt")])[ss] + torch.cat([z1, self.dl("dlt1")])[ss] * self.phi + self.c("gl") * d.xzs[idx]
        xe = d.xe[idx]
        level = level + (xe @ self.dl("ce0"))[:, None] + (xe @ self.dl("ce1"))[:, None] * self.phi
        return s * torch.exp(level)

    def paths(self, v, lam, N, gen):
        B, H = v.shape
        rn = lambda *s: torch.randn(*s, device=DEV, generator=gen)
        rho = self.c("rho") ** (self.inc / 5.0)
        hh = torch.arange(1, H + 1, device=DEV)
        lag = (hh[:, None] - hh[None, :]).clamp(min=0).float()
        K = torch.tril(rho ** lag) * torch.sqrt(1 - rho * rho)
        ou = rn(B, N, H) @ K.T
        A = self.c("A0") + self.c("A1") * self.phi
        cc, theta = self.c("c"), self.c("theta")
        g = A * rn(B, N, 1) + cc * ou + lam[:, None, :] * rn(B, N, H)
        var_g = (A * A + cc * cc * (1 - rho ** (2 * hh.float())))[None, :] + lam * lam
        sd = torch.sqrt(v)[:, None, :] * torch.exp(g - var_g[:, None, :])
        e = sd * rn(B, N, H)
        e = torch.cat([e[..., :1], (e[..., 1:] + theta * e[..., :-1]) / torch.sqrt(1 + theta * theta)], -1)
        lp = torch.cumsum(e - 0.5 * sd * sd, -1)
        return torch.cat([torch.zeros(B, N, 1, device=DEV), lp], -1)

    @staticmethod
    def crps(x, y):
        """Ensemble CRPS along dim 1 with the validator's N=1000 spread term. x [B,N,M], y [B,M]."""
        N = x.shape[1]
        xs, _ = torch.sort(x, dim=1)
        coef = (2 * torch.arange(1, N + 1, device=DEV, dtype=x.dtype) - N - 1).view(1, N, 1)
        spread = (coef * xs).sum(1) / (N * (N - 1))
        return (x - y[:, None, :]).abs().mean(1) - (1 - 1 / 1000) * spread

    def loss(self, d, idx, N, gen, parts=False):
        lp = self.paths(self.variance(d, idx), self.lam(d, idx), N, gen)
        out = {}
        for k in self.ks:
            sim = torch.expm1(lp[..., k::k] - lp[..., :-k:k]) * 1e4
            out[f"r{k}"] = self.crps(sim, d.ret[k][idx]).sum(-1)
        simF = torch.expm1(lp[..., -1] - d.lrF[idx][:, None]) * 1e4
        out["abs"] = self.crps(simF[..., None], torch.zeros(len(idx), 1, device=DEV)).sum(-1)
        if self.tag == "1h":
            out["gaps"] = self.crps(torch.expm1(lp[..., 5::5]) * 1e4, d.gap[idx]).sum(-1)
            r1 = torch.expm1(lp[..., 1:] - lp[..., :-1]) * 1e4
            vol = 0.0
            for bs, wgt in VOL_BLOCKS:
                sv = r1.reshape(r1.shape[0], r1.shape[1], 60 // bs, bs).std(dim=-1)
                vol = vol + wgt * self.crps(sv, d.vol[bs][idx]).sum(-1)
            out["vol"] = vol
        total = sum(out.values())
        return (total, out) if parts else total

@torch.no_grad()
def evaluate(model, d, N=512, chunk=128, max_n=6000, seed=123):
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    idx_all = torch.arange(d.n, device=DEV)
    if d.n > max_n: idx_all = idx_all[torch.linspace(0, d.n - 1, max_n, device=DEV).long()]
    tot, parts, n = 0.0, {}, 0
    for i in range(0, len(idx_all), chunk):
        idx = idx_all[i:i + chunk]
        t, p = model.loss(d, idx, N, gen, parts=True)
        tot += float(t.sum()); n += len(idx)
        for k, v in p.items(): parts[k] = parts.get(k, 0.0) + float(v.sum())
    return tot / n, {k: v / n for k, v in parts.items()}

def fit(asset, tag, args, init=None, freeze=None, mask_feats=None, verbose=True):
    te, vs = minute_stamp(args.train_end), minute_stamp(args.val_start)
    ts = minute_stamp(args.train_start) if args.train_start else 0
    path = Path(args.ds) / f"{asset}_{tag}.npz"
    horizon = CFG[tag]["T"]
    dtr = Data(path, tag, lambda t: (t >= ts) & (t + horizon < te))
    dva = Data(path, tag, lambda t: (t >= vs) & (t < minute_stamp(args.val_end)))
    torch.manual_seed(args.seed)
    model = Model(tag, init, freeze, mask_feats).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 20)
    gen = torch.Generator(device=DEV); gen.manual_seed(args.seed)
    B = args.batch if tag == "24h" else args.batch * 3
    t0 = time.time()
    v0 = evaluate(model, dva)[0] if dva.n else float("nan")
    for step in range(args.steps):
        idx = torch.randint(0, dtr.n, (B,), device=DEV, generator=gen)
        loss = model.loss(dtr, idx, args.paths, gen).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    tr = evaluate(model, dtr, max_n=3000)[0]
    va, parts = evaluate(model, dva) if dva.n else (float("nan"), {})
    out = model.export()
    out.update({"asset": asset, "tag": tag, "train_end": args.train_end, "n_train": dtr.n, "n_val": dva.n, "val": va, "train": tr})
    if verbose:
        print(f"{asset:7s} {tag} ntr={dtr.n} nva={dva.n} init_val={v0:.2f} train={tr:.2f} val={va:.2f} "
              f"parts={ {k: round(v, 1) for k, v in parts.items()} } | b0={out['b0']:.3f} b1={out['b1']:.3f} kappa={out['kappa']:.2f} "
              f"A0={out['A0']:.3f} A1={out['A1']:.3f} c={out['c']:.3f} rho={out['rho']:.3f} theta={out['theta']:.3f} "
              f"lam_raw={np.round(out['lam_raw'], 2)} gz={out['gz']:.2f} gl={out['gl']:.2f} dlt={np.round(out['dlt'], 2)} dlt1={np.round(out['dlt1'], 2)} "
              f"fam={np.round(out['fam_bias'], 2)} p={out['p']:.2f} gamma={out['gamma']:.2f} ce0={np.round(out['ce0'], 3)} ce1={np.round(out['ce1'], 3)} ({time.time() - t0:.0f}s)", flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="all"); ap.add_argument("--tags", default="24h,1h")
    ap.add_argument("--train-start", default=None); ap.add_argument("--train-end", default="2026-07-01")
    ap.add_argument("--val-start", default="2026-07-24"); ap.add_argument("--val-end", default="2026-11-09")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=160)
    ap.add_argument("--paths", type=int, default=256); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--suffix", default="")
    ap.add_argument("--freeze", default="", help="name=value,... (constrained scale)")
    ap.add_argument("--mask-fam", default="", help="families to drop: sq,abs,pk")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    from common import ALL_ASSETS, ASSETS_CRYPTO
    assets = ALL_ASSETS if args.assets == "all" else args.assets.split(",")
    freeze = {k: float(v) for k, v in (kv.split("=") for kv in args.freeze.split(",") if kv)}
    fams = {"sq": 0, "abs": 1, "pk": 2}
    mask = [fams[f] * core.N_HL + i for f in args.mask_fam.split(",") if f in fams for i in range(core.N_HL)]
    if "sess" in args.mask_fam.split(","): mask += [NG, NG + 1]
    mask = mask or None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for tag in args.tags.split(","):
        for a in assets:
            if tag == "1h" and a not in ASSETS_CRYPTO: continue
            res = fit(a, tag, args, freeze=freeze, mask_feats=mask)
            if not args.no_save: (out / f"model_{a}_{tag}{args.suffix}.json").write_text(json.dumps(res))
