"""Verify committed portable serving dependencies without importing XGBoost."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

import modeling
from ml_models import manifest
from portable_trees import portable_booster


def main():
    root = Path(__file__).resolve().parent.parent
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    required = [Path('agent/artifacts')/name for name in
                ['seasonality.json', 'seasonality_abs5m.json', 'events.json',
                 'closures.json', 'ml/manifest.json']]
    fits = [row for row in manifest()['fits'] if row['kind'] == 'vol']
    for fit in fits:
        assert fit['cutoff'] <= '2026-11-09'
        tree = portable_booster(fit['file'])
        prediction = tree.inplace_predict(np.zeros((1, len(fit['features'])), dtype=np.float32))
        assert np.isfinite(prediction).all()
        required.append(Path('agent/artifacts/ml_portable')/Path(fit['file']).with_suffix('.npz'))
    sources = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(root/'agent'):
                sources.add(path.relative_to(root))
    assert 'xgboost' not in sys.modules
    rows = []
    for relative in sorted(set(required)|sources):
        path = root/relative
        assert path.is_file() and str(relative) in tracked, str(relative)
        data = path.read_bytes()
        rows.append({'path': str(relative), 'bytes': len(data), 'tracked': True,
                     'sha256': hashlib.sha256(data).hexdigest()})
    result = {'passed': True, 'context_days': modeling.CONTEXT_DAYS,
              'loaded_volatility_models': len(fits), 'xgboost_imported': False,
              'latest_fit': max(fit['cutoff'] for fit in fits),
              'artifact_bytes': sum((root/path).stat().st_size for path in required),
              'source_and_artifacts': rows}
    destination = root/'agent/results/release_portable_artifact_audit.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print({k: v for k, v in result.items() if k != 'source_and_artifacts'})


if __name__ == '__main__':
    main()
