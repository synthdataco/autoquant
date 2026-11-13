"""Train/serve consistency: features and variance path of simulate()'s code path vs the dataset builder's."""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
import core
from synth_lib.benchmark.generate_predictions import context_slice
pdir = Path(sys.argv[1]); ds = Path(sys.argv[2])
for asset, tag, tl, inc in [("BTC", "24h", 86400, 5), ("NVDAX", "24h", 86400, 5), ("SOL", "1h", 3600, 1)]:
    z = np.load(ds / f"{asset}_{tag}.npz")
    df = load_asset(asset)
    params = core.load_params(asset, tl, pdir)
    events = core.asset_events(asset, core.load_calendar())
    worst = 0.0
    for i in [5, len(z["t0"]) // 2, len(z["t0"]) - 7]:
        t0 = pd.Timestamp(int(z["t0"][i]) * 60, unit="s", tz="UTC")
        ctx = context_slice(df, t0, tl, 30)
        arr = core.context_arrays(ctx, asset, params, events)
        x, xs, _ = core.features_from_arrays(arr["r"], arr["pk"], arr["seas"], arr["sess"])
        seas, sess = core.future_seasonal(int(z["t0"][i]), tl // 60, asset, params, events)
        worst = max(worst, np.abs(x - z["x"][i]).max(), np.abs(xs - z["xs"][i]).max(), np.abs(seas - z["seas"][i]).max(),
                    np.abs(sess - z["sess"][i]).max(), abs(arr["price"] - z["p0"][i]))
    print(asset, tag, "max abs diff features/seasonal/price:", worst)
