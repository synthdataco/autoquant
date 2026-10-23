"""1h-horizon screen: nu / volsd / fastw matter differently for block-RV CRPS."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
from fast_eval2 import run, load

if __name__ == "__main__":
    for asset in ["BTC", "ETH", "SOL"]:
        close = load(asset, "2026-08-20")
        for name, kw in [("base", {}), ("nu4", {"nu":4}), ("nu8", {"nu":8}), ("nu12", {"nu":12}),
                         ("volsd0.2", {"volsd":0.2}), ("volsd0.55", {"volsd":0.55}),
                         ("fastw0.5", {"fastw":0.5}), ("fastw1.0", {"fastw":1.0})]:
            print(asset, "1h", name, round(run(asset, close, horizon_h=1, **kw), 3), flush=True)
