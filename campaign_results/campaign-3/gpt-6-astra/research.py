"""Frozen-data experiments using only the official scorer exposed by predict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from predict import AGENT_DIR, CUTOFF, ensure_workspace_cwd, load_context, score_paths
from auxiliary import auxiliary_features
from synth_lib.backtester.config import slug_for, competition_for
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.preparation import _slice_real_prices, _fill_gaps_from_realized_paths

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
EQUITY = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
WINDOWS = {"dev1": ("2026-07-28", "2026-08-07"), "dev2": ("2026-08-08", "2026-08-18"), "final": ("2026-08-18", "2026-08-28"), "warm_dev2": ("2026-07-29", "2026-08-18")}


def seed_for(asset, start_time):
    return int.from_bytes(hashlib.blake2s(f"{asset}:{start_time}".encode(), digest_size=8).digest(), "little")


@lru_cache(maxsize=8)
def learned_weights(kind='neural'):
    filename = 'score_weights.npz' if kind == 'curve' else 'volatility_weights.npz'
    if kind.startswith('refit_'):
        filename = f'{kind}_weights.npz'
    if kind=='aux':
        filename='aux_weights.npz'
    if kind=='weighted':
        filename='weighted_score_weights.npz'
    if kind=='lead':
        filename='lead_weights.npz'
    if kind=='fiveminute':
        filename='fiveminute_weights.npz'
    if kind=='ensemble':
        filename='ensemble_weights.npz'
    if kind=='uncertainty':
        filename='uncertainty_weights.npz'
    if kind=='calibration':
        filename='shape_score_fine_weights.npz'
    if kind=='calibration600':
        filename='shape_score_fine_600_weights.npz'
    if kind=='refresh':
        filename='refresh_weights.npz'
    return dict(np.load(AGENT_DIR / filename))


def learned_multiplier(feature, asset, length, kind):
    from train_vol import feature_vector

    weights = learned_weights(kind)
    vector=feature_vector(feature,asset)
    if kind=='aux':
        vector=np.r_[vector,feature['auxiliary']]
    inputs = np.clip((vector - weights['center']) / weights['spread'], -8, 8)
    head = 0 if length == 3600 else 1
    if kind=='ensemble':
        for layer in [0,2]:
            inputs = np.einsum('noi,ni->no',weights[f'{layer}.weight'],inputs)+weights[f'{layer}.bias']
            inputs = inputs/(1+np.exp(-inputs))
        output = np.einsum('noi,ni->no',weights['4.weight'],inputs)+weights['4.bias']
        return float(np.exp(np.clip(output[:,head],-2,2).mean()))
    if kind == 'ridge':
        return float(np.clip(1 + np.r_[1, inputs] @ weights['ridge'][:,head], 0.2, 4))
    for layer in [0, 2]:
        inputs = weights[f'{layer}.weight'] @ inputs + weights[f'{layer}.bias']
        inputs = inputs / (1 + np.exp(-inputs))
    output = weights['4.weight'] @ inputs + weights['4.bias']
    limits = weights['limits'] if kind.startswith('calibration') else [-2,2]
    return float(np.exp(np.clip(output[head], *limits)))


def features(context, start_time, increment):
    close = pd.to_numeric(context["close"], errors="coerce")
    close = close.loc[np.isfinite(close) & (close > 0)]
    log_prices = np.log(close.to_numpy())
    minutes = (close.index - pd.Timestamp(start_time)).total_seconds().to_numpy() / 60
    gaps = np.diff(minutes)
    returns = np.diff(log_prices)
    valid = (gaps > 0) & np.isfinite(returns)
    returns, gaps, age = returns[valid], gaps[valid], -minutes[1:][valid]
    normalized = returns / np.sqrt(gaps)
    output = {"last": float(close.iloc[-1]), "age_minutes": max(0.,float(-minutes[-1])), "starter": float(np.std(returns)), "rv": {}, "av": {}, "range": {}, 'context': context}
    log_ranges, range_age = np.array([]), np.array([])
    if {'high','low'}.issubset(context.columns):
        high = pd.to_numeric(context['high'],errors='coerce').to_numpy(dtype=float)
        low = pd.to_numeric(context['low'],errors='coerce').to_numpy(dtype=float)
        range_valid = np.isfinite(high) & np.isfinite(low) & (low>0) & (high>=low)
        log_ranges = np.log(high[range_valid]/low[range_valid])
        range_age = (pd.Timestamp(start_time)-context.index[range_valid]).total_seconds().to_numpy()/60
    for half_life in [15, 30, 60, 120, 180, 360, 720, 1440, 2880, 10080]:
        weights = np.exp(-np.log(2) * age / half_life)
        output["rv"][half_life] = float(np.sqrt(np.average(normalized ** 2, weights=weights)))
        output["av"][half_life] = float(np.average(np.abs(normalized), weights=weights) * np.sqrt(np.pi / 2))
        output['range'][half_life] = output['av'][half_life]
        if len(log_ranges)>20 and np.any(log_ranges>0):
            range_weights = np.exp(-np.log(2)*range_age/half_life)
            output['range'][half_life] = float(np.average(log_ranges,weights=range_weights)*np.sqrt(np.pi/8))
    raw = pd.to_numeric(context["close"], errors="coerce")
    raw_log = np.log(raw.where(raw > 0))
    raw_age = (pd.Timestamp(start_time) - raw.index).total_seconds().to_numpy() / 60
    for lag in [5, 15, 30, 180]:
        changes = raw_log.diff(lag).to_numpy() / np.sqrt(lag)
        valid_changes = np.isfinite(changes)
        output[f"av{lag}"] = {}
        for half_life in [30, 60, 180, 360, 720, 1440, 2880, 10080]:
            weights = np.exp(-np.log(2) * raw_age[valid_changes] / half_life)
            output[f"av{lag}"][half_life] = float(np.average(np.abs(changes[valid_changes]), weights=weights) * np.sqrt(np.pi / 2))
    output["returns"] = normalized
    clock = pd.DataFrame({'absolute':np.abs(normalized),'hour':close.index[1:][valid].hour,
                          'day':close.index[1:][valid].date})
    daily_mean = clock.groupby('day').absolute.transform('mean').clip(lower=1e-9)
    clock['relative'] = clock.absolute / daily_mean
    profile = clock.groupby('hour').relative.mean().reindex(range(24),fill_value=1).to_numpy()
    profile = (.25*np.roll(profile,1) + .5*profile + .25*np.roll(profile,-1))
    profile = np.clip(profile,.25,3)
    output['cycle'] = profile / profile.mean()
    sampled = raw_log.iloc[::-max(1,increment//60)].iloc[::-1].diff()
    output['step_returns']=sampled.replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
    differences=np.round(np.abs(np.diff(close.to_numpy())),max(0,10-int(np.floor(np.log10(output['last'])))))
    positive_differences=differences[differences>0]
    output['tick']=float(positive_differences.min()) if len(positive_differences) else 0.
    digits = min(8,max(0,12-int(np.floor(np.log10(output['last'])))))
    factor = 10.**digits
    output['gcd_tick'] = 0.
    if len(close)>1 and close.max()*factor<np.iinfo(np.int64).max/4:
        integer_prices = np.rint(close.to_numpy()*factor).astype(np.int64)
        output['gcd_tick'] = float(np.gcd.reduce(np.abs(np.diff(integer_prices)))/factor)
    pairs = pd.concat([sampled,sampled.shift(1)],axis=1).dropna()
    output['correlation'] = float(pairs.corr().iloc[0,1]) if len(pairs)>20 and (pairs.std()>0).all() else 0.
    output['auxiliary'],output['auxiliary_usable']=auxiliary_features(context,start_time,max(output['av'][360],1e-8),output['av'])
    return output


def generate_variant(feature, asset, start_time, increment, length, count, variant):
    parts = variant.split(":")
    weekly_strength = float(parts[1]) if parts[0]=='stackweekly' else 0.
    age_strength = float(parts[1]) if parts[0]=='stackage' else 0.
    center_strength = float(parts[1]) if parts[0]=='stackcenter' else 0.
    micro_parameters = parts[1:] if parts[0]=='stackmicro' else None
    if parts[0]=='stackweekly':
        parts = ['stackrank',parts[2],'1'] if len(parts)>2 else ['stackrank','increments','0']
    if parts[0] in ['stackage','stackcenter','stackmicro']:
        parts = ['stackrank','prefix' if length==3600 else 'multiscale','1']
    stacked_mode = parts[0] in ['stacked','refresh','refresh_old','stacktick','stackdf','stackperiodic','stackensemble','stacktree','stacklead','stackpersist','stackstudent','stackmix','stacknewton','stackfine','stackrank','stackphase']
    phase_strength = float(parts[2]) if parts[0]=='stackphase' else None
    phase_kind = 'phase_control' if parts[0]=='stackphase' and parts[1]=='control' else 'phase'
    rank_mode = parts[1] if parts[0]=='stackrank' else None
    rank_strength = float(parts[2]) if rank_mode else 0.
    fine_strength = (1. if asset not in CRYPTO and length>3600 else None) if parts[0] in ['stackrank','stackphase'] else float(parts[1]) if parts[0]=='stackfine' else None
    newton_strength = float(parts[1]) if parts[0]=='stacknewton' else None
    sampler_persistence = float(parts[1]) if parts[0]=='stackpersist' else None
    temporal_lead_strength = 1. if parts[0] in ['stacknewton','stackfine','stackrank','stackphase'] else float(parts[2]) if parts[0]=='stackmix' else float(parts[1]) if parts[0]=='stacklead' else None
    tree_strength = (1. if length>3600 else None) if parts[0] in ['stackrank','stackphase'] else 1. if parts[0]=='stackfine' else float(parts[1]) if parts[0] in ['stacktree','stackstudent','stackmix'] else None
    if tree_strength == 0:
        tree_strength = None
    tree_kind = 'tree_student' if parts[0] in ['stackstudent','stackmix','stackfine','stackrank','stackphase'] else 'tree'
    temporal_ensemble_strength = float(parts[1]) if parts[0]=='stackensemble' else None
    sampler_degrees = float(parts[1]) if parts[0]=='stackdf' else 5.
    refresh_strength = float(parts[1]) if parts[0] in ['refresh','refresh_old'] else None
    refresh_kind = 'refit_recent60' if parts[0]=='refresh_old' else 'refresh'
    offset = int(parts[1]) if parts[0]=='resample' else int(parts[2]) if parts[0]=='sampler' and len(parts)>2 else 0
    draw_seed = (seed_for(asset,start_time)+offset) % (2**64)
    rng = np.random.default_rng(draw_seed)
    steps = length // increment
    lead_strength=.5 if stacked_mode or parts[0] in ['location','periodic','uncertainty','clock','sharp','adaptive','calibration','calibration600','ticked'] else float(parts[2]) if parts[0] in ['combined','combined_exact'] else float(parts[1]) if parts[0]=='lead' else None
    sampler='sobol' if stacked_mode or parts[0]=='ticked' else parts[5] if parts[0]=='sharp' and len(parts)>5 else parts[3] if parts[0] in ['combined','combined_exact'] and len(parts)>3 else parts[1] if parts[0]=='sampler' else None
    fiveminute_strength=float(parts[1]) if parts[0]=='fiveminute' else None
    temporal_strength=1. if stacked_mode or parts[0]=='ticked' else float(parts[2]) if parts[0]=='sharp' else float(parts[1]) if parts[0] in ['temporal','temporal_exact','combined','combined_exact'] else None
    temporal_kind='temporal_exact' if stacked_mode or parts[0] in ['temporal_exact','combined_exact','sharp','ticked'] else 'temporal'
    ensemble_strength=float(parts[1]) if parts[0]=='ensemble' else None
    location_strength=float(parts[1]) if parts[0]=='location' else None
    periodic_strength=(.5 if asset not in CRYPTO and length>3600 else 0.) if parts[0] in ['stackfine','stackrank','stackphase'] else float(parts[3]) if parts[0]=='stackmix' else float(parts[1]) if parts[0] in ['periodic','stackperiodic'] else float(parts[2]) if parts[0]=='stacktick' and len(parts)>2 else None
    uncertainty_parameters=parts[1:] if parts[0]=='uncertainty' else None
    clock_parameters=(['48','.75','.25'] if asset in CRYPTO else ['96','1','.25']) if stacked_mode else [parts[1],parts[3],'0.25'] if parts[0]=='sharp' else parts[1:] if parts[0]=='clock' else None
    joint_normalize=bool(int(parts[4])) if parts[0]=='sharp' else False
    adaptive_strength=float(parts[1]) if parts[0]=='adaptive' else None
    calibration_strength=float(parts[1]) if parts[0] in ['calibration','calibration600'] else None
    calibration_kind=parts[0] if calibration_strength is not None else 'calibration'
    weighted_strength=float(parts[1]) if parts[0]=='weighted' else None
    tuned_mode=parts[0] in ['tuned','resample'] or lead_strength is not None or sampler is not None or fiveminute_strength is not None or temporal_strength is not None or ensemble_strength is not None
    tick_kind='gcd' if parts[0] in ['stackmix','stackfine','stackrank','stackphase'] else parts[1] if parts[0] in ['ticked','stacktick'] else 'min'
    atom_mode='quantize' if parts[0] in ['ticked','stacktick','stackmix','stackfine','stackrank','stackphase'] else parts[0] if parts[0] in ['symboot','hurdle','quantize'] else None
    atom_parameter=1. if parts[0] in ['ticked','stacktick','stackmix','stackfine','stackrank','stackphase'] else float(parts[1]) if atom_mode else 0.
    auxiliary_strength=float(parts[1]) if parts[0]=='aux' else None
    range_parameters = parts[1:] if parts[0] == 'range' else None
    correlation_parameter = parts[1] if parts[0] == 'corr' else None
    prior_strength = float(parts[1]) if parts[0] == 'prior' else 0.
    refit_parameters = parts[1:] if parts[0] == 'refit' else None
    if parts[0] == 'quick':
        parts = ['av',parts[1],'1','5']
    shape_parameters = parts[1:] if parts[0] == "shape" else None
    cycle_strength = float(parts[1]) if parts[0] == "cycle" else 0.
    if prior_strength:
        cycle_strength = .75
        parts = ['hybrid','1']
    if refit_parameters:
        cycle_strength = .75
        parts = ['auto','0','1','5'] if length == 3600 else ['ml','refit_'+refit_parameters[0],refit_parameters[1]]
    if correlation_parameter is not None:
        cycle_strength = .75
        parts = ['hybrid','1']
    if range_parameters:
        cycle_strength = .75
        parts = ['hybrid','1']
    if auxiliary_strength is not None:
        cycle_strength=.75
        parts=['auto','0','1','5']
    if atom_mode:
        cycle_strength=.75
        parts=['hybrid','1']
    if tuned_mode or weighted_strength is not None:
        cycle_strength=.75
        parts=['auto','0','1','5']
    if shape_parameters:
        cycle_strength = .75
        parts = ["hybrid", "1"]
    if parts[0] == "cycle":
        parts = ["hybrid", "1"]
    if parts[0] == "hybrid":
        strength = 1.0 if asset in CRYPTO else float(parts[1])
        parts = ["auto", "0", "1", "5"] if length == 3600 else ["ml", "neural", str(strength)]
    schedule = None
    learned_scale = 1.0
    if tuned_mode or weighted_strength is not None:
        incumbent=learned_multiplier(feature,asset,length,'neural')**(.5 if length==3600 else 1.)
        learned_scale=incumbent
        if refresh_strength is not None and pd.Timestamp(start_time)>=pd.Timestamp('2026-08-07',tz='UTC'):
            refreshed = learned_multiplier(feature,asset,length,refresh_kind)**(.5 if length==3600 else 1.)
            learned_scale *= (refreshed/incumbent)**refresh_strength
        if calibration_strength is not None:
            learned_scale *= learned_multiplier(feature,asset,length,calibration_kind)**calibration_strength
        if adaptive_strength is not None:
            from adaptive import adaptive_ratio

            corrections = feature.setdefault('adaptive',{})
            if length not in corrections:
                corrections[length] = adaptive_ratio(feature['context'],start_time,asset,length)
            learned_scale *= corrections[length]**adaptive_strength
        if periodic_strength is not None:
            from periodic import periodic_multiplier

            learned_scale *= periodic_multiplier((CRYPTO+EQUITY).index(asset),start_time,length)**periodic_strength
        if ensemble_strength is not None:
            ensemble=learned_multiplier(feature,asset,length,'ensemble')**(.5 if length==3600 else 1.)
            learned_scale=incumbent**(1-ensemble_strength)*ensemble**ensemble_strength
        if temporal_strength is not None:
            from temporal import context_sequence, temporal_residual
            from train_vol import feature_vector

            if 'temporal' not in feature:
                feature['temporal'] = context_sequence(feature['context'],start_time,max(feature['av'][360],1e-8))
            residual = temporal_residual(feature['temporal'],feature_vector(feature,asset),temporal_kind)
            if temporal_ensemble_strength is not None:
                ensemble = np.mean([residual] + [temporal_residual(feature['temporal'],feature_vector(feature,asset),name)
                                               for name in ['temporal_seed41','temporal_seed109']],axis=0)
                residual = residual*(1-temporal_ensemble_strength)+ensemble*temporal_ensemble_strength
            learned_scale *= np.exp(temporal_strength*residual[0 if length==3600 else 1])
            if tree_strength is not None:
                from chronos_vol import forecast_features
                from tree_vol import tree_design, tree_residual

                if tree_kind=='tree' and 'chronos' not in feature:
                    feature['chronos'] = forecast_features(feature['temporal'][None])
                baseline_logs = np.log([learned_multiplier(feature,asset,3600,'neural')**.5,
                                        learned_multiplier(feature,asset,86400,'neural')])+residual
                designs = tree_design(feature_vector(feature,asset)[None],feature['temporal'][None],
                                      [start_time],baseline_logs[None],feature.get('chronos'))
                if tree_kind=='tree':
                    design = designs['pretrained'][0]
                else:
                    from temporal import temporal_embedding

                    design = np.r_[designs['temporal'][0], temporal_embedding(feature['temporal'],feature_vector(feature,asset),'temporal_exact')]
                learned_scale *= np.exp(tree_strength*tree_residual(design,tree_kind)[0 if length==3600 else 1])
        if fiveminute_strength is not None and length>3600:
            learned_scale=incumbent**(1-fiveminute_strength)*learned_multiplier(feature,asset,length,'fiveminute')**fiveminute_strength
        if weighted_strength is not None:
            learned_scale=incumbent**(1-weighted_strength)*learned_multiplier(feature,asset,length,'weighted')**weighted_strength
    if auxiliary_strength is not None:
        incumbent=1. if length==3600 else learned_multiplier(feature,asset,length,'neural')
        learned_scale=incumbent
        if feature['auxiliary_usable']:
            learned_scale=incumbent**(1-auxiliary_strength)*learned_multiplier(feature,asset,length,'aux')**auxiliary_strength
    if newton_strength is not None and length>3600:
        from newton_score import newton_multiplier

        learned_scale *= 1+newton_strength*(newton_multiplier(feature,asset,start_time)-1)
    if phase_strength and length>3600:
        from newton_score import newton_design
        from tree_vol import tree_residual

        learned_scale *= np.exp(phase_strength*tree_residual(newton_design(feature,asset,start_time),phase_kind)[1])
    if micro_parameters:
        if micro_parameters[0]=='constant':
            learned_scale *= float(micro_parameters[1])
        else:
            from micro_score import micro_multiplier

            learned_scale *= micro_multiplier(feature,asset,start_time,micro_parameters[0])
    innovation = parts[0] if parts[0] in ["iid", "sv", "boot"] else None
    innovation_parameters = parts[1:]
    if innovation:
        parts = ["auto", "0", "1"]
    if parts[0] == "ml":
        learned_scale = learned_multiplier(feature, asset, length, parts[1]) ** float(parts[2])
        parts = ["auto", "0", "1", "5"]
    if parts[0] == "revert":
        slow_half_life, decay_minutes = int(parts[1]), float(parts[2])
        initial = feature["av"][30 if length == 3600 else 60]
        target = feature["av"][slow_half_life]
        weight = np.exp(-np.arange(1, steps + 1) * increment / 60 / decay_minutes)
        schedule = np.exp(weight * np.log(initial) + (1 - weight) * np.log(target))
        parts = ["auto", "0", "1", "5"]
    if parts[0].startswith("auto"):
        parts[0] = "av" + parts[0][4:]
        parts[1] = "60" if length == 3600 else "360"
    if parts[0] == "starter":
        sigma = feature["starter"]
    else:
        sigma = feature[parts[0]][int(parts[1])]
    multiplier = float(parts[2]) if len(parts) > 2 else 1.0
    sigma *= multiplier * learned_scale * np.sqrt(increment / 60)
    if range_parameters:
        half_life = 60 if length==3600 else 360
        ratio = feature['range'][half_life]*float(range_parameters[1])/max(feature['av'][half_life],1e-8)
        sigma *= ratio**float(range_parameters[0])
    if schedule is not None:
        sigma = schedule * np.sqrt(increment / 60)
    flat_sigma = sigma
    if cycle_strength and length > 3600:
        future_times = pd.Timestamp(start_time) + pd.to_timedelta(np.arange(1,steps+1)*increment,unit='s')
        profile = feature['cycle']
        clock_indices = future_times.hour
        if clock_parameters:
            from clock_profile import estimate_profile

            bins,cycle_strength,floor = int(clock_parameters[0]),float(clock_parameters[1]),float(clock_parameters[2])
            profiles = feature.setdefault('clock_profiles',{})
            if (bins,floor) not in profiles:
                profiles[(bins,floor)] = estimate_profile(feature['context'],start_time,bins,floor)
            profile = profiles[(bins,floor)]
            clock_indices = (future_times.hour*60+future_times.minute)*bins//1440
        if prior_strength:
            with np.load(AGENT_DIR/'clock_profiles.npz') as archive:
                profile = (1-prior_strength)*profile + prior_strength*archive[asset]
        sigma *= 1 + cycle_strength*(profile[clock_indices]-1)
    if lead_strength is not None and length > 3600:
        from train_vol import feature_vector

        weights = learned_weights('lead')
        values = np.clip((feature_vector(feature,asset)-weights['center'])/weights['spread'],-8,8)
        for layer in [0,2]:
            values = weights[f'{layer}.weight'] @ values + weights[f'{layer}.bias']
            values = values / (1+np.exp(-values))
        bucket_scale = np.exp(np.clip(weights['4.weight'] @ values + weights['4.bias'],-2,2))
        if temporal_lead_strength is not None:
            from lead_temporal import corrected_buckets

            bucket_scale = corrected_buckets(feature['temporal'], feature_vector(feature,asset), bucket_scale, temporal_lead_strength)
        if fine_strength is not None:
            from temporal import temporal_residual, temporal_weights

            correction = np.clip(temporal_residual(feature['temporal'],feature_vector(feature,asset),'lead_finetune'),-.5,.5)
            adjusted = bucket_scale*np.exp(fine_strength*correction)
            duration = temporal_weights('lead_finetune')['duration']
            bucket_scale = adjusted*(bucket_scale @ duration)/(adjusted @ duration)
        if weekly_strength:
            from weekly_lead import weekly_buckets

            bucket_scale = weekly_buckets(bucket_scale,start_time,(CRYPTO+EQUITY).index(asset))
        minutes = np.arange(1,steps+1)*increment/60
        centers = (weights['edges'][:-1]+weights['edges'][1:])/2
        shape = np.interp(minutes,centers,bucket_scale)
        shape = np.clip(shape/shape.mean(),.4,2.5)**lead_strength
        sigma *= shape/shape.mean()
    if joint_normalize and length>3600 and np.mean(sigma)>0:
        sigma *= flat_sigma/np.mean(sigma)
    if age_strength and steps>0 and feature['age_minutes']>0:
        sigma = np.broadcast_to(sigma,(steps,)).copy()
        sigma[0] *= np.sqrt(1+age_strength*feature['age_minutes']/(increment/60))
    shocks = rng.standard_normal(((count + 1) // 2, steps))
    shocks = np.concatenate([shocks, -shocks], axis=0)[:count]
    if correlation_parameter is not None:
        correlation = np.clip(.5*feature['correlation'],-.25,.25) if correlation_parameter=='auto' else float(correlation_parameter)
        orthogonal = np.sqrt(1-correlation**2)
        for step in range(1,steps):
            shocks[:,step] = correlation*shocks[:,step-1] + orthogonal*shocks[:,step]
    if atom_mode=='symboot':
        absolute_returns=np.abs(feature['step_returns'])
        if len(absolute_returns) and absolute_returns.mean()>0:
            magnitudes=rng.choice(absolute_returns,size=((count+1)//2,steps))
            magnitudes=np.concatenate([magnitudes,magnitudes],axis=0)[:count]
            shocks=np.sign(shocks)*magnitudes/absolute_returns.mean()*np.sqrt(2/np.pi)*atom_parameter
    if atom_mode=='hurdle':
        probability=float(np.clip(np.mean(feature['step_returns']==0)*atom_parameter,0,.8))
        shocks *= (rng.random(shocks.shape)>=probability)/(1-probability)
    if innovation == "iid":
        degrees = float(innovation_parameters[0])
        shocks *= np.sqrt((degrees - 2) / rng.chisquare(degrees, size=shocks.shape))
    elif innovation == "sv":
        spread, persistence = map(float, innovation_parameters)
        correlation = np.exp(-increment / 60 / persistence)
        state = rng.standard_normal(count) * spread
        for step in range(steps):
            state = correlation * state + np.sqrt(1-correlation**2) * spread * rng.standard_normal(count)
            shocks[:,step] *= np.exp(state-spread**2)
    elif innovation == "boot":
        empirical = feature['returns']
        sampled = rng.choice(empirical, size=shocks.shape)
        sampled = sampled - np.mean(empirical)
        shocks = sampled / max(np.std(empirical), 1e-9)
    if len(parts) > 3:
        degrees = float(shape_parameters[0]) if shape_parameters else float(parts[3])
        if shape_parameters:
            mean_absolute = np.sqrt(2/np.pi) if degrees >= 100 else np.exp(np.log(2)+.5*np.log(degrees-2)+math.lgamma((degrees+1)/2)-np.log(degrees-1)-.5*np.log(np.pi)-math.lgamma(degrees/2))
            shocks *= float(shape_parameters[1]) * np.sqrt(2/np.pi) / mean_absolute
        if 2 < degrees < 100 and atom_mode!='symboot':
            mixing = np.sqrt((degrees - 2) / rng.chisquare(degrees, size=(count, 1)))
            shocks *= mixing
            if uncertainty_parameters:
                weights = learned_weights('uncertainty')
                head,category = int(length>3600),int(asset not in CRYPTO)
                quantiles = weights['pooled'][head,category]
                if uncertainty_parameters[0]=='conditional':
                    state = np.digitize(np.log(feature['av'][60]/feature['av'][360]),[-.2,.2])
                    quantiles = weights['conditional'][head,category,state]
                replacement = rng.choice(quantiles,size=(count,1))
                selected = rng.random((count,1))<float(uncertainty_parameters[1])
                shocks *= np.where(selected,replacement,mixing)/mixing
    if sampler:
        from sampling import mixture_shocks

        shocks = mixture_shocks(count, steps, draw_seed, sampler, sampler_degrees)
        if sampler_persistence is not None:
            from sampling import persistent_shocks

            shocks = persistent_shocks(count, steps, draw_seed, increment, sampler_persistence)
    if rank_mode=='increments' and rank_strength>0:
        from sampling import stratified_marginals

        shocks = (1-rank_strength)*shocks+rank_strength*stratified_marginals(shocks)
    increments = shocks*sigma
    if rank_mode=='multiscale' and rank_strength>0:
        from sampling import multiscale_stratification

        increments = (1-rank_strength)*increments+rank_strength*multiscale_stratification(increments,sigma)
    log_paths = np.concatenate([np.zeros((count, 1)), np.cumsum(increments, axis=1)], axis=1)
    if rank_mode=='prefix' and rank_strength>0:
        from sampling import stratified_marginals

        cumulative_scale = np.sqrt(np.cumsum(np.broadcast_to(sigma,(steps,))**2))
        log_paths[:,1:] = (1-rank_strength)*log_paths[:,1:]+rank_strength*stratified_marginals(log_paths[:,1:],cumulative_scale)
    if center_strength:
        from scipy.special import logsumexp

        log_paths -= center_strength*(logsumexp(log_paths,axis=0)-np.log(count))
        log_paths[:,0] = 0.
    if location_strength is not None:
        from location import location_curve
        from train_vol import feature_vector

        log_paths += location_strength*location_curve(feature['context'],start_time,max(feature['av'][360],1e-8),
                                                      feature_vector(feature,asset),np.arange(steps+1)*increment/60)
    paths=feature["last"] * np.exp(log_paths)
    tick = feature['gcd_tick'] if tick_kind=='gcd' else feature['tick']
    if atom_mode=='quantize' and 0<tick<feature['last']*.01:
        tick *= atom_parameter
        paths=np.maximum(tick,feature['last']+np.rint((paths-feature['last'])/tick)*tick)
        paths[:,0]=feature['last']
    return paths


def evaluate_asset(task):
    asset, length, window, variants, count, stride = task
    ensure_workspace_cwd()
    begin, end = [pd.Timestamp(value, tz="UTC") for value in WINDOWS[window]]
    increment = 60 if length == 3600 else 300
    field = get_miner_scores(begin, end, asset, length, increment)
    times = pd.DatetimeIndex(field.start_time.unique()).sort_values()[::stride]
    prices = load_context(asset, times.min() - pd.Timedelta(days=7), min(end, times.max() + pd.Timedelta(seconds=length)))
    prompts = []
    for timestamp in times:
        prompts.append({"asset": asset, "start_time": timestamp, "time_length": length, "time_increment": increment,
                        "real_prices": _slice_real_prices(prices[["close"]], timestamp, length, increment)})
    _fill_gaps_from_realized_paths(prompts, asset)
    rows = []
    for prompt in prompts:
        timestamp = prompt["start_time"]
        context = prices.loc[(prices.index > timestamp - pd.Timedelta(days=7)) & (prices.index <= timestamp)]
        feature = features(context, timestamp, increment)
        for variant in variants:
            started = perf_counter()
            paths = generate_variant(feature, asset, timestamp, increment, length, count, variant)
            elapsed = perf_counter() - started
            score, detail = score_paths(paths, prompt["real_prices"], increment, asset, length)
            row = {"asset": asset, "length": length, "window": window, "time": timestamp,
                   "variant": variant, "crps": score, "latency": elapsed, "paths": count,
                   "age_minutes":feature['age_minutes']}
            row.update({item["Interval"]: item["CRPS"] for item in detail})
            rows.append(row)
    print(f"{window} {asset} {length}: {len(prompts)} prompts, {len(variants)} variants", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variants", default="starter,rv:360,rv:1440,av:360,av:1440")
    parser.add_argument("--variants-equity")
    parser.add_argument("--variants-high")
    parser.add_argument("--windows", default="dev1,dev2")
    parser.add_argument("--assets", default=",".join(CRYPTO + EQUITY))
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--stride-low", type=int, default=4)
    parser.add_argument("--stride-high", type=int, default=24)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--format", choices=['all','low','high'],default='all')
    args = parser.parse_args()
    print((AGENT_DIR / "BUDGET.md").read_text(), flush=True)
    variants = args.variants.split(",")
    tasks = [(asset, length, window, args.variants_high.split(',') if length==3600 and args.variants_high else args.variants_equity.split(',') if asset in EQUITY and args.variants_equity else variants,
              args.paths, args.stride_high if length == 3600 else args.stride_low)
             for window in args.windows.split(",") for asset in args.assets.split(",")
             for length in ([86400, 3600] if asset in CRYPTO else [86400])
             if args.format=='all' or (args.format=='low' and length==86400) or (args.format=='high' and length==3600)]
    target = AGENT_DIR / "results" / f"{args.name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = target.with_suffix('.parts')
    checkpoints.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256()
    for source in sorted([*AGENT_DIR.glob('*.py'), *AGENT_DIR.glob('*_weights.npz')]):
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for task in tasks:
            identity = json.dumps([task, fingerprint.hexdigest()]).encode()
            checkpoint = checkpoints / f'{hashlib.sha256(identity).hexdigest()}.parquet'
            if args.resume and checkpoint.exists():
                results.append(pd.read_parquet(checkpoint))
                print(f'Resumed {task[:3]}', flush=True)
            else:
                pending[executor.submit(evaluate_asset, task)] = checkpoint
        for completed in as_completed(pending):
            batch = pd.DataFrame(completed.result())
            checkpoint = pending[completed]
            temporary = checkpoint.with_suffix('.tmp')
            batch.to_parquet(temporary)
            temporary.replace(checkpoint)
            results.append(batch)
    result = pd.concat(results, ignore_index=True).sort_values(['window', 'asset', 'length', 'time', 'variant'])
    result.to_parquet(target)
    print(result.groupby(["window", "length", "asset", "variant"]).crps.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
