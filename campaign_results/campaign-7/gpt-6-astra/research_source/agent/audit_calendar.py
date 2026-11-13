"""Check production calendar fallbacks and refresh without calculating CRPS."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from candidate_models import candidate_forecast
from integrated_models import draw_logpaths
from run_suite import ROOT
from synth_lib.benchmark.generate_predictions import context_slice, load_minute_prices
from synth_lib.preparation.config import MARKET_DATA_DIR


def main():
    os.chdir(ROOT)
    model = importlib.import_module('modeling')
    supplied_path = os.environ['SYNTH_BENCHMARK_CALENDAR']
    supplied = pd.read_parquet(supplied_path)
    eligible = supplied[(supplied.event == 'CPI') &
                        (supplied.time_utc < pd.Timestamp('2026-11-09', tz='UTC'))]
    event = eligible.sort_values('time_utc').iloc[-1]
    start = event.time_utc - pd.Timedelta(minutes=15)
    prices = load_minute_prices(MARKET_DATA_DIR, 'BTC',
                               start-pd.Timedelta(days=model.CONTEXT_DAYS), start)
    context = context_slice(prices, start, 3600, model.CONTEXT_DAYS)
    unchanged = context.copy(deep=True)
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    os.environ['SYNTH_BENCHMARK_SEED'] = str(seed)
    original_connect = socket.socket.connect
    original_variant = os.environ.get('SYNTH_RESEARCH_VARIANT')
    rows = []

    def no_network(*args, **kwargs):
        raise AssertionError('Serving attempted a network connection')

    def invoke(name, count=1000):
        began = perf_counter()
        result = model.simulate('BTC', start.isoformat(), 60, 3600, count, context)
        elapsed = perf_counter()-began
        paths = np.asarray(result[2:], dtype=float)
        assert paths.shape == (count, 61), (name, paths.shape)
        assert np.isfinite(paths).all() and (paths > 0).all()
        assert np.all(paths[:, 0] == float(context.close.dropna().iloc[-1]))
        assert elapsed < 5, (name, elapsed)
        pd.testing.assert_frame_equal(unchanged, context)
        rows.append({'case': name, 'paths': count, 'seconds': elapsed, 'passed': True})
        return paths

    socket.socket.connect = no_network
    try:
        last, variance = candidate_forecast('BTC', start, 60, 3600, context,
                                             shape=True, events=False)
        no_events = last*np.exp(draw_logpaths(seed, 1000, variance, 60, mixture=.4))
        os.environ.pop('SYNTH_BENCHMARK_CALENDAR', None)
        np.testing.assert_array_equal(invoke('absent_env'), no_events)
        with TemporaryDirectory(prefix='calendar-audit-') as directory:
            path = Path(directory)/'events.parquet'
            os.environ['SYNTH_BENCHMARK_CALENDAR'] = str(path)
            np.testing.assert_array_equal(invoke('missing_file'), no_events)
            supplied.iloc[:0].to_parquet(path, index=False)
            np.testing.assert_array_equal(invoke('empty_calendar'), no_events)
            first_mtime = path.stat().st_mtime_ns
            selected = supplied[supplied.time_utc == event.time_utc].copy()
            selected.to_parquet(path, index=False)
            # Model the production refresh explicitly even on coarse filesystems.
            os.utime(path, ns=(first_mtime+1_000_000, first_mtime+1_000_000))
            with_event = invoke('refresh_at_same_path')
            assert not np.array_equal(with_event, no_events)
            np.testing.assert_array_equal(invoke('repeat_refreshed_calendar'), with_event)
            second_mtime = path.stat().st_mtime_ns
            supplied.iloc[:0].to_parquet(path, index=False)
            os.utime(path, ns=(second_mtime+1_000_000, second_mtime+1_000_000))
            np.testing.assert_array_equal(invoke('remove_event_at_same_path'), no_events)
            preserved_mtime = path.stat().st_mtime_ns
            replacement = Path(directory)/'replacement.parquet'
            selected.to_parquet(replacement, index=False)
            os.utime(replacement, ns=(preserved_mtime, preserved_mtime))
            replacement.replace(path)
            np.testing.assert_array_equal(invoke('atomic_refresh_preserved_mtime'), with_event)
            supplied.iloc[:0].to_parquet(path, index=False)
            os.utime(path, ns=(preserved_mtime, preserved_mtime))
            np.testing.assert_array_equal(invoke('in_place_refresh_preserved_mtime'), no_events)
            unknown = selected.copy()
            unknown['event'] = 'UNRECOGNIZED_AUDIT_TYPE'
            unknown.to_parquet(path, index=False)
            third_mtime = path.stat().st_mtime_ns
            os.utime(path, ns=(third_mtime+1_000_000, third_mtime+1_000_000))
            np.testing.assert_array_equal(invoke('unknown_type'), no_events)
        os.environ['SYNTH_BENCHMARK_CALENDAR'] = supplied_path
        current = invoke('supplied_calendar')
        np.testing.assert_array_equal(current, with_event)
        for count in [1, 3, 999, 1000, 1001]:
            first = invoke(f'count_{count}', count)
            np.testing.assert_array_equal(invoke(f'repeat_count_{count}', count), first)
        for variant in ['candidate_basic', 'prof_abs5m60', 'event_clean', 'integrated_sv']:
            os.environ['SYNTH_RESEARCH_VARIANT'] = variant
            np.testing.assert_array_equal(invoke('ignored_'+variant), current)
    finally:
        socket.socket.connect = original_connect
        os.environ['SYNTH_BENCHMARK_CALENDAR'] = supplied_path
        if original_variant is None:
            os.environ.pop('SYNTH_RESEARCH_VARIANT', None)
        else:
            os.environ['SYNTH_RESEARCH_VARIANT'] = original_variant
    result = {'event_from_supplied_calendar': event.time_utc.isoformat(),
              'context_ends': context.index[-1].isoformat(), 'seed': seed,
              'calendar_absent_matches_no_event_model': True,
              'refresh_changes_forecast': True, 'network_blocked': True,
              'research_flag_invariant': True, 'cases': rows,
              'max_seconds': max(r['seconds'] for r in rows)}
    destination = ROOT/'agent/results/calendar_audit.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print('Passed', len(rows), 'calendar, count and serving-flag cases; max seconds',
          result['max_seconds'])


if __name__ == '__main__':
    main()
