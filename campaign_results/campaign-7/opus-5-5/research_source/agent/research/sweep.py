"""Run a list of param configs through evalrun on a standard window, then compare to the first.
  sweep.py --comp crypto-24h --win A --configs '{"ref": {}, "s085": {"scale": 0.85}}' [--workers 12]
Windows: A = profiles fitted < 2026-08-01, B = < 2026-10-01 (walk-forward)."""
import argparse, json, subprocess, sys
WIN = {
    'A': {'prof': 'research/profiles_2026-08-01.npz', '24h': ('2026-08-05', '2026-08-26'), '1h': ('2026-08-05', '2026-08-15')},
    'B': {'prof': 'research/profiles_2026-10-01.npz', '24h': ('2026-10-08', '2026-10-29'), '1h': ('2026-10-08', '2026-10-18')},
    'S': {'prof': 'research/profilesxm_2026-09-01.npz', '24h': ('2026-09-05', '2026-09-26'), '1h': ('2026-09-23', '2026-10-03')},
    'C': {'prof': 'research/profilesxm_2026-10-01.npz', '24h': ('2026-10-29', '2026-11-08'), '1h': ('2026-10-29', '2026-11-08')},
}
ap = argparse.ArgumentParser()
ap.add_argument('--comp', required=True); ap.add_argument('--win', required=True)
ap.add_argument('--configs', required=True); ap.add_argument('--workers', type=int, default=12)
ap.add_argument('--model', default='agent/modeling.py'); ap.add_argument('--prefix', default='')
ap.add_argument('--verbose', action='store_true'); ap.add_argument('--assets', default=None)
a = ap.parse_args()
cfgs = json.loads(a.configs)
# freeze the model for the whole sweep: editing agent/modeling.py mid-sweep must not leak into it
import os, shutil, atexit
if a.model == 'agent/modeling.py':
    snap = f'agent/_snap_{os.getpid()}.py'
    shutil.copy('agent/modeling.py', snap)
    atexit.register(lambda: os.path.exists(snap) and os.remove(snap))
    a.model = snap
w = WIN[a.win]; s, e = w['1h' if a.comp == 'crypto-1h' else '24h']
tags = []
for name, params in cfgs.items():
    params = dict(params); params.setdefault('profile_file', w['prof'])
    tag = f"{a.prefix}{a.win}_{name}"
    tags.append(tag)
    r = subprocess.run([sys.executable, '-I', 'agent/research/evalrun.py', '--comp', a.comp, '--start', s, '--end', e,
                        '--mode', 'field', '--tag', tag, '--model', a.model, '--params', json.dumps(params),
                        '--workers', str(a.workers)] + (['--assets', a.assets] if a.assets else []), capture_output=True, text=True)
    if r.returncode != 0:
        print(name, 'FAILED', r.stderr[-2000:]); continue
    lines = r.stdout.splitlines()
    i = [j for j, l in enumerate(lines) if l.startswith('[')][0]
    print('\n'.join(lines[i:] if a.verbose else lines[i:i + 1]), flush=True)
if len(tags) > 1:
    subprocess.run([sys.executable, '-I', 'agent/research/compare.py', a.comp, *tags])
