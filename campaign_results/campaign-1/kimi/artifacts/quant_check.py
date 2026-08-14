"""One-off: is the midpoint quantile grid CRPS-optimal for a Student-t?

Compares the served midpoint-quantile ensemble against gradient-optimized atoms
on the exact CRPS objective (quadrature). Diagnostic only.
"""
import numpy as np
from scipy.stats import t as st

df, n = 4.5, 1000
F = st(df=df)
sd = st.std(df)
yy = F.ppf(np.linspace(2e-5, 1 - 2e-6, 50001)) / sd
pdf = F.pdf(yy * sd) / sd
pdf /= np.trapezoid(pdf, yy)
YS = np.array_split(yy, 10)
PS = np.array_split(pdf, 10)


def crps_of(x):
    x = np.sort(x)
    t1 = 0.0
    for ys, ps in zip(YS, PS):
        t1 += np.trapezoid(np.mean(np.abs(x[:, None] - ys[None, :]), axis=0) * ps, ys)
    t2 = 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    return t1 - t2


def grad(x):
    x = np.sort(x)
    g = np.zeros_like(x)
    for ys, ps in zip(YS, PS):
        g += np.trapezoid(np.sign(x[:, None] - ys[None, :]) * ps[None, :], ys, axis=1)
    return g / n - np.mean(np.sign(x[:, None] - x[None, :]), axis=1) / n


x_mid = F.ppf((np.arange(n) + 0.5) / n) / sd
c_mid = crps_of(x_mid)
print(f"midpoint quantile grid CRPS: {c_mid:.6f}", flush=True)

x = x_mid.copy()
for _ in range(200):
    x = np.sort(x - 0.025 * n * grad(x))
c_opt = crps_of(x)
print(f"optimized atoms CRPS:        {c_opt:.6f}")
print(f"relative gap: {100 * (c_mid - c_opt) / c_mid:.4f}%")
