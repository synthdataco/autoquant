"""Diagnose Sobol stream coupling without evaluating a forecasting score."""
import json
import os
import numpy as np
from sampling_models import normal_coordinates
from run_suite import ROOT


def main():
    rows = []
    for seed in [int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108')), 91117]:
        for d in [61, 289]:
            count = 1000
            child = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
            first = normal_coordinates(seed, count, d, True)
            separate = normal_coordinates(child, count, d, True)
            joint = normal_coordinates(seed, count, 2*d, True)
            variance = float(np.mean(first[:, 0]**2))
            wrong = np.mean(first*separate, axis=0)/variance
            corrected = np.mean(joint[:, :d]*joint[:, d:], axis=0)/variance
            # Match total marginal log-scale variance in the corrected process.
            rho = np.exp2(-(1 if d == 61 else 5)/(30 if d == 61 else 180))
            state = .4*np.sqrt(.5)*joint[:, d]
            persistent = .4*np.sqrt(.5)*joint[:, d-1]
            means, deviations = [], []
            for j in range(d-1):
                state = rho*state+.4*np.sqrt(.5)*np.sqrt(1-rho*rho)*joint[:, d+j+1]
                log_scale = persistent+state
                means.append(float(np.exp(log_scale-.5*.4**2).mean()))
                deviations.append(float(log_scale.std()))
            rows.append({'seed': seed, 'dimensions_per_process': d, 'paths': count,
                         'invalid_separate_engine_correlations_min_max': [float(wrong.min()), float(wrong.max())],
                         'joint_same_coordinate_correlations_min_max': [float(corrected.min()), float(corrected.max())],
                         'joint_scale_mean_min_max': [min(means), max(means)],
                         'joint_log_scale_sd_min_max': [min(deviations), max(deviations)],
                         'population_scale_mean': 1., 'population_log_scale_sd': .4})
    (ROOT/'agent/results/temporal_sampling_audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
