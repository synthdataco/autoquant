"""Run a pre-registered family serially, checking the campaign budget between jobs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--model', default='vol_models')
    ap.add_argument('--suffix', default='dev')
    ap.add_argument('--workers', type=int, default=6)
    args, extra = ap.parse_known_args()
    for variant in args.variants:
        budget = (ROOT/'agent/BUDGET.md').read_text()
        if 'LANDING ORDER' in budget:
            print(budget, flush=True)
            break
        print(f'Running {variant}; budget checked.', flush=True)
        tag = variant + '_' + args.suffix
        log = ROOT/'agent/results'/f'{tag}.log'
        env = os.environ.copy()
        env['SYNTH_RESEARCH_VARIANT'] = variant
        with log.open('w') as out:
            code = subprocess.run(['uv','run','agent/run_suite.py','--model',args.model,'--tag',tag,'--workers',str(args.workers),*extra],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
        print('\n'.join(log.read_text().splitlines()[-4:]), flush=True)
        if code:
            raise SystemExit(code)

if __name__ == '__main__':
    main()
