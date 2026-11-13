"""A dated scale prior for genuinely sparse observed return histories."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from ml_features import one_features
from ml_models import scale_adjustment
from vol_models import valid_returns


@lru_cache(maxsize=1)
def fits():
    return json.loads((Path(__file__).resolve().parent/'artifacts/volatility_prior.json').read_text())['fits']


def prior_factor(asset, start, horizon, context, *, no_returns_only=False):
    _, returns, _ = valid_returns(context)
    count = int(np.isfinite(returns).sum())
    if count >= (1 if no_returns_only else 30):
        return 1.
    asof = str(context.index[-1].date())
    dates = [date for date in fits() if date <= asof]
    if not dates:
        return 1.
    matches = [row for row in fits()[max(dates)]
               if row['asset'] == asset and row['time_length'] == horizon]
    if not matches:
        return 1.
    prior = matches[0]['scale']
    _, base = one_features(asset, start, horizon, context)
    current = base*scale_adjustment(asset, start, horizon, context, 'vol', 1.)
    return float(np.exp((1-count/30)*np.log(prior/max(current, 1e-12))))
