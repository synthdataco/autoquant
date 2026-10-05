"""Final acceptance test for the champion's declared interface.

Everything the harness is entitled to assume about `agent/modeling.py:simulate` is asserted
here, on the real snapshot, with the network physically disabled. Run before nominating:

    uv run agent/verify_interface.py

Checks, per asset and per scoring configuration:
  * signature `(asset, start_time, time_increment, time_length, num_simulations, context_prices)`
  * return `(start_timestamp, time_increment, *paths)` with exactly num_simulations paths
  * each path `time_length // time_increment + 1` finite floats
  * each path starts at the LAST price in the context (the contract's anchor)
  * determinism: same SYNTH_BENCHMARK_SEED -> identical paths; different seed -> different
  * a context containing NaN is accepted (dropped, never interpolated)
  * no socket is ever opened
"""

from __future__ import annotations

import os
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

import predict as P  # noqa: E402

P.ensure_workspace_cwd()


class _Offline(socket.socket):
    def __init__(self, *a, **k):
        raise RuntimeError("the model opened a socket: it is not offline")


def go_offline() -> None:
    socket.socket = _Offline
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("the model opened a connection: it is not offline")
    )


CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
COMEQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
START = datetime(2026, 9, 10, 13, 17, tzinfo=timezone.utc)   # not a round hour
NSIM = 8


def context_for(asset: str, start: datetime):
    end = start - timedelta(minutes=1)
    return P.load_context_frame(asset, end - timedelta(minutes=P.CONTEXT_MINUTES), end)


def check(simulate, asset, start, inc, length, ctx, label):
    n_expect = length // inc + 1
    out = simulate(asset, start, inc, length, NSIM, ctx)
    assert isinstance(out, (list, tuple)), f"{label}: return is {type(out)}"
    ts, got_inc, *paths = out
    assert got_inc == inc, f"{label}: echoed increment {got_inc} != {inc}"
    assert len(paths) == NSIM, f"{label}: {len(paths)} paths, asked for {NSIM}"
    anchor = float(ctx["close"].dropna().iloc[-1])
    for i, p in enumerate(paths):
        a = np.asarray(p, dtype=float)
        assert a.shape == (n_expect,), f"{label}: path {i} has {a.shape}, want ({n_expect},)"
        assert np.all(np.isfinite(a)), f"{label}: path {i} has non-finite values"
        assert np.all(a > 0), f"{label}: path {i} has a non-positive price"
        assert abs(a[0] - anchor) <= 1e-9 * max(1.0, abs(anchor)), (
            f"{label}: path {i} starts at {a[0]}, contract says {anchor}"
        )
    return ts, np.asarray(paths, dtype=float)


def main() -> int:
    go_offline()
    from modeling import simulate

    configs = [("24h", 300, 86400, CRYPTO + COMEQU), ("1h", 60, 3600, CRYPTO)]
    for name, inc, length, assets in configs:
        for asset in assets:
            ctx = context_for(asset, START)
            nan_frac = float(ctx["close"].isna().mean())
            label = f"{name}/{asset}"

            os.environ["SYNTH_BENCHMARK_SEED"] = "424242"
            ts, a = check(simulate, asset, START, inc, length, ctx, label)
            _, b = check(simulate, asset, START, inc, length, ctx, label)
            assert np.array_equal(a, b), f"{label}: same seed gave different paths"

            os.environ["SYNTH_BENCHMARK_SEED"] = "424243"
            _, c = check(simulate, asset, START, inc, length, ctx, label)
            assert not np.array_equal(a, c), f"{label}: the seed does nothing"

            del os.environ["SYNTH_BENCHMARK_SEED"]
            _, d = check(simulate, asset, START, inc, length, ctx, label)
            assert np.all(np.isfinite(d)), f"{label}: no-seed run is broken"

            # A context that is mostly holes must still produce a legal prediction.
            holed = ctx.copy()
            holed.iloc[: int(len(holed) * 0.8)] = np.nan
            check(simulate, asset, START, inc, length, holed, label + "/holed")

            print(f"ok  {label:>14}  ts={ts}  nan={nan_frac:6.1%}  "
                  f"spread={np.std(np.log(a[:, -1] / a[:, 0])):.4f}")

    print("\nALL INTERFACE CHECKS PASSED (offline, seeded, NaN-tolerant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
