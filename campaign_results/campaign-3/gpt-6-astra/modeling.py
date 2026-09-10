"""Offline probabilistic paths with robust, horizon-specific volatility forecasts."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtri
from scipy.stats import chi2, qmc, t as student_t

ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500")
HALF_LIVES = (30, 60, 180, 360, 720, 1440, 2880, 10080)
COMMODITY_EQUITY_BLEND = 1.0
CYCLE_STRENGTH = 0.75
LEAD_STRENGTH = 0.5
WEIGHTS_AVAILABLE_FROM = pd.Timestamp('2026-07-26',tz='UTC')


@lru_cache(maxsize=8)
def _weights(kind='volatility'):
    with np.load(Path(__file__).with_name(f"{kind}_weights.npz"), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _features(closes, timestamp):
    minutes = (closes.index - timestamp).total_seconds().to_numpy() / 60
    gaps = np.diff(minutes)
    returns = np.diff(np.log(closes.to_numpy(dtype=float)))
    valid = (gaps > 0) & np.isfinite(returns)
    normalized = returns[valid] / np.sqrt(gaps[valid])
    age = -minutes[1:][valid]
    absolute, root_square = {}, {}
    for half_life in HALF_LIVES:
        if not normalized.size:
            absolute[half_life] = root_square[half_life] = 1e-8
            continue
        weights = np.exp(-np.log(2) * age / half_life)
        absolute[half_life] = max(float(np.average(np.abs(normalized), weights=weights) * np.sqrt(np.pi / 2)), 1e-8)
        root_square[half_life] = max(float(np.sqrt(np.average(normalized ** 2, weights=weights))), 1e-8)
    return absolute, root_square


def _feature_values(absolute, root_square, asset):
    reference = absolute[360]
    values = [np.log(max(estimator[half_life], 1e-9) / reference)
              for estimator in (absolute, root_square) for half_life in HALF_LIVES]
    values += [np.log(reference), *[float(asset == name) for name in ASSETS]]
    return np.asarray(values,dtype=np.float32)


def _network_output(values, weights):
    inputs = np.clip((values - weights["center"]) / weights["spread"], -8, 8)
    for layer in (0, 2):
        inputs = weights[f"{layer}.weight"] @ inputs + weights[f"{layer}.bias"]
        inputs = inputs / (1 + np.exp(-inputs))
    return weights["4.weight"] @ inputs + weights["4.bias"]


def _learned_multiplier(absolute, root_square, asset, hourly):
    output = _network_output(_feature_values(absolute,root_square,asset),_weights())
    strength = .5 if hourly else 1.0 if asset in ASSETS[:5] else COMMODITY_EQUITY_BLEND
    return float(np.exp(np.clip(output[0 if hourly else 1], -2, 2))) ** strength


def _lead_profile(absolute, root_square, asset, steps, increment, embedding, sequence):
    if steps == 0:
        return np.ones(0)
    weights = _weights('lead')
    buckets = np.exp(np.clip(_network_output(_feature_values(absolute,root_square,asset),weights),-2,2))
    correction_weights = _weights('lead_temporal')
    standardized = np.clip((_feature_values(absolute,root_square,asset)-correction_weights['static_center'])/correction_weights['static_spread'],-8,8)
    design = np.r_[embedding,standardized]
    design = np.r_[1.,np.clip((design-correction_weights['center'])/correction_weights['spread'],-8,8)]
    adjusted = buckets*np.exp(np.clip(design @ correction_weights['coefficients'],-.5,.5))
    duration = correction_weights['duration']
    buckets = adjusted*(buckets @ duration)/(adjusted @ duration)
    if asset not in ASSETS[:5]:
        fine = _weights('lead_finetune')
        hidden = _sequence_embedding(sequence,_feature_values(absolute,root_square,asset),fine)
        correction = np.clip(fine['head.2.weight'] @ hidden+fine['head.2.bias'],-.5,.5)
        adjusted = buckets*np.exp(correction)
        duration = fine['duration']
        buckets = adjusted*(buckets @ duration)/(adjusted @ duration)
    centers = (weights['edges'][:-1]+weights['edges'][1:])/2
    shape = np.interp(np.arange(1,steps+1)*increment/60,centers,buckets)
    shape = np.clip(shape/shape.mean(),.4,2.5)**LEAD_STRENGTH
    return shape/shape.mean()


def _temporal_embedding(closes, timestamp, absolute, root_square, asset):
    weights = _weights('temporal_exact')
    reference = absolute[360]
    gaps = np.diff(closes.index.as_unit('ns').asi8)/60e9
    changes = np.diff(np.log(closes.to_numpy()))/np.sqrt(gaps)
    observed = pd.DataFrame({'absolute':np.abs(changes),'square':changes**2,'signed':changes},index=closes.index[1:])
    grouped = observed.resample('1h',origin=timestamp.floor('min'),closed='right',label='right')
    rows = grouped.mean()
    rows['coverage'] = grouped['absolute'].count()/60
    rows = rows.reindex(pd.date_range(end=timestamp.floor('min'),periods=168,freq='1h'))
    values = np.stack([np.log(np.maximum(rows.absolute.to_numpy()*np.sqrt(np.pi/2),1e-9)/reference),
                       np.log(np.maximum(np.sqrt(rows.square.to_numpy()),1e-9)/reference),
                       rows.signed.to_numpy()/reference,rows.coverage.to_numpy()])
    values = np.clip(np.nan_to_num(values,nan=0.,posinf=5.,neginf=-5.),-5,5).astype(np.float32)
    return _sequence_embedding(values,_feature_values(absolute,root_square,asset),weights), values


def _sequence_embedding(sequence, static, weights):
    values = sequence
    for layer in [0,2,4]:
        padded = np.pad(values,((0,0),(2,2)))
        windows = np.lib.stride_tricks.sliding_window_view(padded,5,axis=1)[:,::2]
        values = np.einsum('ock,ctk->ot',weights[f'encoder.{layer}.weight'],windows)+weights[f'encoder.{layer}.bias'][:,None]
        values = values/(1+np.exp(-values))
    static = np.clip((static-weights['center'])/weights['spread'],-8,8)
    values = np.r_[values.reshape(-1),static]
    values = weights['head.0.weight'] @ values+weights['head.0.bias']
    values = values/(1+np.exp(-values))
    return values


def _temporal_adjustment(embedding):
    weights = _weights('temporal_exact')
    return np.clip(weights['head.2.weight'] @ embedding+weights['head.2.bias'],-.8,.8)


def _student_multiplier(sequence, embedding, absolute, root_square, asset, timestamp, residual):
    static = _feature_values(absolute,root_square,asset)
    baseline = np.log([_learned_multiplier(absolute,root_square,asset,True),
                       _learned_multiplier(absolute,root_square,asset,False)])+residual
    summaries = np.concatenate([sequence[:,-width:].mean(axis=1) for width in [1,3,6,12,24,48,96,168]])
    day = (timestamp.hour+timestamp.minute/60)/24
    week = (timestamp.dayofweek+day)/7
    clock = np.array([function(2*np.pi*frequency*phase) for phase in [day,week]
                      for frequency in [1,2] for function in [np.sin,np.cos]])
    design = np.r_[static,baseline,summaries,clock,embedding]
    weights = _weights('tree_student')
    correction = float(weights['base'][1])
    for stage in range(int(weights['stages'])):
        nodes = weights[f'1_{stage}']
        index = 0
        while not nodes[index]['is_leaf']:
            node = nodes[index]
            value = design[int(node['feature_idx'])]
            left = bool(node['missing_go_to_left']) if np.isnan(value) else value<=node['num_threshold']
            index = int(node['left'] if left else node['right'])
        correction += nodes[index]['value']
    return np.exp(np.clip(correction,-.5,.5))


def _periodic_multiplier(asset, timestamp):
    day = (timestamp.hour+timestamp.minute/60+timestamp.second/3600)/24
    week = (timestamp.dayofweek+day)/7
    harmonic = np.array([function(2*np.pi*frequency*phase) for phase in [day,week]
                         for frequency in [1,2] for function in [np.sin,np.cos]])
    indicator = np.array([float(asset==name) for name in ASSETS])
    design = np.r_[harmonic,(harmonic[:,None]*indicator[None,:]).reshape(-1)]
    output = design @ _weights('periodic')['coefficients']
    return float(np.exp(np.clip(output[1],-.5,.5)))**.5


def _sobol_shocks(count, steps, seed):
    pairs = (count+1)//2
    engine = qmc.Sobol(steps+1,scramble=True,seed=int(seed%(2**32)))
    uniforms = engine.random_base2(int(np.ceil(np.log2(max(1,pairs)))))[:pairs]
    scales = np.sqrt(3/chi2.ppf(np.clip(uniforms[:,:1],1e-12,1-1e-12),5))
    shocks = ndtri(np.clip(uniforms[:,1:],1e-12,1-1e-12))*scales
    return np.concatenate([shocks,-shocks],axis=0)[:count]


@lru_cache(maxsize=8)
def _student_quantiles(count):
    probabilities = (np.arange(count)+.5)/count
    return np.sqrt(3/5)*student_t.ppf(probabilities,5)


def _stratified_marginals(values, scales=1.):
    if values.shape[1]==0:
        return values.copy()
    order = np.argsort(values,axis=0,kind='stable')
    output = np.empty_like(values)
    np.put_along_axis(output,order,_student_quantiles(len(values))[:,None],axis=0)
    return output*np.broadcast_to(scales,(values.shape[1],))


def _multiscale_stratification(increments, scales):
    steps = increments.shape[1]
    adjusted = increments.copy()
    variances = np.broadcast_to(scales,(steps,))**2
    for sweep in range(2):
        for width in [1,6,36,288]:
            blocks = steps//width
            if blocks==0:
                continue
            used = blocks*width
            block_values = adjusted[:,:used].reshape(len(adjusted),blocks,width)
            block_variances = variances[:used].reshape(blocks,width)
            totals = block_values.sum(axis=2)
            total_variance = block_variances.sum(axis=1)
            targets = _stratified_marginals(totals,np.sqrt(total_variance))
            allocation = np.divide(block_variances,total_variance[:,None],
                                   out=np.zeros_like(block_variances),where=total_variance[:,None]>0)
            block_values += (targets-totals)[:,:,None]*allocation[None]
    return adjusted


def _volatility_profile(closes, bins=24):
    if len(closes) < 2:
        return np.ones(bins)
    gaps = np.diff(closes.index.as_unit('ns').asi8) / 60e9
    absolute = np.abs(np.diff(np.log(closes.to_numpy(dtype=float)))) / np.sqrt(gaps)
    minute = closes.index[1:].hour*60+closes.index[1:].minute
    clock = pd.DataFrame({'absolute':absolute,'bin':minute*bins//1440,'day':closes.index[1:].date})
    daily_mean = clock.groupby('day').absolute.transform('mean').clip(lower=1e-9)
    clock['relative'] = clock.absolute / daily_mean
    profile = clock.groupby('bin').relative.mean().reindex(range(bins),fill_value=1).to_numpy()
    profile = .25*np.roll(profile,1) + .5*profile + .25*np.roll(profile,-1)
    profile = np.clip(profile,.25,3)
    return profile / profile.mean()


def _quote_tick(closes, last):
    digits = min(8,max(0,12-int(np.floor(np.log10(last)))))
    factor = 10.**digits
    if len(closes)<2 or closes.max()*factor>=np.iinfo(np.int64).max/4:
        return 0.
    integer_prices = np.rint(closes.to_numpy()*factor).astype(np.int64)
    return float(np.gcd.reduce(np.abs(np.diff(integer_prices)))/factor)


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    if time_increment <= 0 or time_length < 0 or num_simulations < 1:
        raise ValueError("Positive increment/path count and nonnegative horizon are required")
    timestamp = pd.to_datetime(start_time, unit="s", utc=True) if isinstance(start_time, (int, float, np.integer, np.floating)) else pd.Timestamp(start_time)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    if context_prices is None or "close" not in context_prices:
        raise ValueError("A context frame containing close prices is required")
    closes = pd.to_numeric(context_prices["close"], errors="coerce").copy()
    closes.index = pd.to_datetime(closes.index, utc=True)
    closes = closes.loc[(closes.index > timestamp - pd.Timedelta(days=7)) & (closes.index <= timestamp)]
    closes = closes.loc[np.isfinite(closes) & (closes > 0)].sort_index()
    closes = closes.loc[~closes.index.duplicated(keep="last")]
    if closes.empty:
        raise ValueError("No finite positive close exists in the seven-day context")
    last = float(closes.iloc[-1])
    absolute, root_square = _features(closes, timestamp)
    hourly = time_length <= 3600
    sigma = absolute[60 if hourly else 360]
    learned_scale = 1.0
    if timestamp >= WEIGHTS_AVAILABLE_FROM:
        learned_scale = _learned_multiplier(absolute, root_square, asset, hourly)
        if not hourly and asset not in ASSETS[:5]:
            learned_scale *= _periodic_multiplier(asset,timestamp)
        embedding, sequence = _temporal_embedding(closes,timestamp,absolute,root_square,asset)
        residual = _temporal_adjustment(embedding)
        learned_scale *= np.exp(residual[0 if hourly else 1])
        if not hourly:
            learned_scale *= _student_multiplier(sequence,embedding,absolute,root_square,asset,timestamp,residual)
    sigma *= learned_scale*np.sqrt(time_increment / 60)
    seed = int.from_bytes(hashlib.blake2s(f"{asset}:{timestamp}".encode(), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    steps = time_length // time_increment
    if not hourly:
        bins = (48 if asset in ASSETS[:5] else 96) if timestamp>=WEIGHTS_AVAILABLE_FROM else 24
        cycle_strength = 1.0 if asset not in ASSETS[:5] and timestamp>=WEIGHTS_AVAILABLE_FROM else CYCLE_STRENGTH
        profile = _volatility_profile(closes,bins)
        future_times = timestamp + pd.to_timedelta(np.arange(1,steps+1)*time_increment,unit='s')
        clock_indices = (future_times.hour*60+future_times.minute)*bins//1440
        sigma *= 1 + cycle_strength*(profile[clock_indices]-1)
        if timestamp >= WEIGHTS_AVAILABLE_FROM:
            sigma *= _lead_profile(absolute,root_square,asset,steps,time_increment,embedding,sequence)
    if timestamp >= WEIGHTS_AVAILABLE_FROM:
        shocks = _sobol_shocks(num_simulations,steps,seed)
    else:
        shocks = rng.standard_normal(((num_simulations + 1) // 2, steps))
        shocks = np.concatenate([shocks, -shocks], axis=0)[:num_simulations]
        shocks *= np.sqrt(3 / rng.chisquare(5, size=(num_simulations, 1)))
    increments = shocks*sigma
    if timestamp>=WEIGHTS_AVAILABLE_FROM and not hourly:
        increments = _multiscale_stratification(increments,sigma)
    log_paths = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(increments, axis=1)], axis=1)
    if timestamp>=WEIGHTS_AVAILABLE_FROM and hourly:
        cumulative_scale = np.sqrt(np.cumsum(np.broadcast_to(sigma,(steps,))**2))
        log_paths[:,1:] = _stratified_marginals(log_paths[:,1:],cumulative_scale)
    if timestamp>=WEIGHTS_AVAILABLE_FROM and not hourly and asset in ASSETS[:5]:
        log_paths -= logsumexp(log_paths,axis=0)-np.log(num_simulations)
        log_paths[:,0] = 0.
    paths = last * np.exp(np.clip(log_paths, -40, 40))
    tick = _quote_tick(closes,last)
    if 0<tick<last*.01:
        paths = np.maximum(tick,last+np.rint((paths-last)/tick)*tick)
        paths[:,0] = last
    return (start_time, time_increment, *paths.tolist())
