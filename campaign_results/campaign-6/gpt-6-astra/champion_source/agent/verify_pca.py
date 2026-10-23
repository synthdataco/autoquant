"""Algebraic covariance/repeatability gate, plus 1000-path transform timing."""
import os,time,json
from pathlib import Path
import numpy as np
from qmc_pca import pca_normals

rng=np.random.default_rng(int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011')))
rows=[]
for n in [1,7,60,288]:
    for kind in ['constant','heterogeneous']:
        sig=np.ones(n) if kind=='constant' else np.exp(rng.normal(size=n))
        z=pca_normals(np.eye(n),sig)
        error=np.max(np.abs(z.T@z-np.eye(n)))
        assert error<1e-8,(n,error)
        paths=np.cumsum(z*sig,axis=1);clock=np.cumsum(sig**2)
        assert np.allclose(paths.T@paths,np.minimum.outer(clock,clock),rtol=1e-10,atol=1e-9)
        normal=rng.normal(size=(1000,n));started=time.perf_counter()
        result=pca_normals(normal,sig);elapsed=time.perf_counter()-started
        assert np.array_equal(result,pca_normals(normal,sig))
        rows.append(dict(steps=n,clock=kind,increment_covariance_max_error=float(error),seconds1000=elapsed))
Path('agent/results/pca_algebra_gate.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
