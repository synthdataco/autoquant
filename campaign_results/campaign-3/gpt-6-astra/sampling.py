"""Deterministic finite-fan samplers for the same scale-mixture process."""

from collections import deque
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import chi2, qmc, t


def bridge_increments(normals):
    count, steps = normals.shape
    paths = np.zeros((count, steps + 1))
    if steps == 0:
        return normals.copy()
    paths[:, steps] = np.sqrt(steps) * normals[:, 0]
    intervals = deque([(0, steps)])
    coordinate = 1
    while intervals:
        left, right = intervals.popleft()
        middle = (left + right) // 2
        if middle == left:
            continue
        span = right - left
        paths[:, middle] = ((right - middle) * paths[:, left] + (middle - left) * paths[:, right]) / span
        paths[:, middle] += np.sqrt((middle - left) * (right - middle) / span) * normals[:, coordinate]
        coordinate += 1
        intervals.extend([(left, middle), (middle, right)])
    return np.diff(paths, axis=1)


def mixture_shocks(count, steps, seed, mode, degrees=5.):
    rng = np.random.default_rng(seed)
    pairs = (count + 1) // 2
    if mode in ['sobol', 'bridge']:
        engine = qmc.Sobol(steps + 1, scramble=True, seed=int(seed % (2**32)))
        uniforms = engine.random_base2(int(np.ceil(np.log2(max(1, pairs)))))[:pairs]
        scales = np.sqrt((degrees - 2) / chi2.ppf(np.clip(uniforms[:, :1], 1e-12, 1 - 1e-12), degrees))
        normals = ndtri(np.clip(uniforms[:, 1:], 1e-12, 1 - 1e-12))
        if mode == 'bridge':
            normals = bridge_increments(normals)
    else:
        normals = rng.standard_normal((pairs, steps))
        if mode == 'paired':
            scales = np.sqrt(3 / rng.chisquare(5, size=(pairs, 1)))
        elif mode == 'stratified':
            quantiles = (np.arange(pairs) + .5) / pairs
            scales = np.sqrt(3 / chi2.ppf(rng.permutation(quantiles)[:, None], 5))
        else:
            raise ValueError(f'Unknown sampler {mode}')
    shocks = normals * scales
    return np.concatenate([shocks, -shocks], axis=0)[:count]


def persistent_shocks(count, steps, seed, increment, persistence_minutes):
    if steps == 0 or np.isinf(persistence_minutes):
        return mixture_shocks(count, steps, seed, 'sobol')
    pairs = (count+1)//2
    power = int(np.ceil(np.log2(max(1, pairs))))
    engine = qmc.Sobol(steps+1, scramble=True, seed=int(seed % (2**32)))
    uniforms = engine.random_base2(power)[:pairs]
    normals = ndtri(np.clip(uniforms[:, 1:], 1e-12, 1-1e-12))
    block_indices = np.arange(steps)*increment//3600
    blocks = int(block_indices[-1]+1)
    scale = np.empty((pairs, blocks))
    scale[:, 0] = np.sqrt(3/chi2.ppf(np.clip(uniforms[:, 0], 1e-12, 1-1e-12), 5))
    if blocks > 1:
        innovations = qmc.Sobol(blocks-1, scramble=True, seed=int((seed ^ 0x6A09E667) % (2**32)))
        innovations = ndtri(np.clip(innovations.random_base2(power)[:pairs], 1e-12, 1-1e-12))
        latent = ndtri(np.clip(uniforms[:, 0], 1e-12, 1-1e-12))
        correlation = np.exp(-60/persistence_minutes)
        for block in range(1, blocks):
            latent = correlation*latent+np.sqrt(1-correlation**2)*innovations[:, block-1]
            scale[:, block] = np.sqrt(3/chi2.ppf(np.clip(ndtr(latent), 1e-12, 1-1e-12), 5))
    shocks = normals*scale[:, block_indices]
    return np.concatenate([shocks, -shocks], axis=0)[:count]


@lru_cache(maxsize=8)
def student_quantiles(count, degrees=5.):
    probabilities = (np.arange(count)+.5)/count
    return np.sqrt((degrees-2)/degrees)*t.ppf(probabilities, degrees)


def stratified_marginals(values, scales=1.):
    if values.shape[1] == 0:
        return values.copy()
    order = np.argsort(values, axis=0, kind='stable')
    output = np.empty_like(values)
    np.put_along_axis(output, order, student_quantiles(len(values))[:, None], axis=0)
    return output*np.broadcast_to(scales, (values.shape[1],))


def multiscale_stratification(increments, scales):
    steps = increments.shape[1]
    adjusted = increments.copy()
    variances = np.broadcast_to(scales, (steps,))**2
    for sweep in range(2):
        for width in [1, 6, 36, 288]:
            blocks = steps//width
            if blocks == 0:
                continue
            used = blocks*width
            block_values = adjusted[:,:used].reshape(len(adjusted),blocks,width)
            block_variances = variances[:used].reshape(blocks,width)
            totals = block_values.sum(axis=2)
            total_variance = block_variances.sum(axis=1)
            targets = stratified_marginals(totals,np.sqrt(total_variance))
            allocation = np.divide(block_variances,total_variance[:,None],
                                   out=np.zeros_like(block_variances),where=total_variance[:,None]>0)
            block_values += (targets-totals)[:,:,None]*allocation[None]
    return adjusted
