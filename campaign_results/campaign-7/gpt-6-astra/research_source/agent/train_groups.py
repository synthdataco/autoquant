"""Run the two competition-specific fits serially on the single GPU."""
import os
import subprocess
from run_suite import ROOT, CRYPTO, OTHER


def main():
    os.chdir(ROOT)
    for name, assets in [('ml_crypto_daily', CRYPTO), ('ml_other_daily', OTHER)]:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            print('Landing order: stop fitting.', flush=True)
            return
        subprocess.run(['uv', 'run', 'agent/train_ml.py', '--output', name,
                        '--kinds', 'vol', '--horizons', '86400', '--assets', *assets],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
