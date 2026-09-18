"""Rebuild a scoreable workspace bundle from a published champion tree.

`run_verdict` clones `<results>/<campaign>/<leg>/workspace.bundle` and checks out `CHAMPION.sha`,
which proves the scored code is the nominated code. Campaigns 1 and 2 were archived without their
bundles — the operator runbook says to drop them before publishing — and their campaign_runs
directories are gone, so those legs cannot currently be rescored at all.

This reconstructs a bundle from what WAS published. The published leg is the unpack_champion
layout: `modeling.py` and its artifacts flat in the leg directory. A bundle needs the workspace
layout instead — `agent/modeling.py` beside `agent/<artifacts>`, and a `pyproject.toml` at the root
for the sandbox's `uv sync`.

    uv run python scripts/reconstruct_bundle.py --campaign campaign-2 --leg claude \\
        --out rescore --synth-lib-rev <the rev the verdict will run on>

WHAT THIS COSTS. A git commit's sha is a hash of its content and metadata, so the rebuilt commit
cannot carry the original one: the emitted CHAMPION names the new sha. The guarantee therefore
weakens from "the scored code is provably the commit the agent nominated" to "the scored code is
the tree that was published". Those should be the same tree — publishing is a checkout of that
commit — but it is an assumption rather than a proof, and a verdict produced this way should say so.

Writes to a separate results root (`--out`) on purpose. The published archive is the record, and on
at least one leg it is also what a live miner was deployed from; nothing here should touch it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# Added by unpack_champion when the leg was published, or written by the harness around it. None of
# it belongs in a reconstructed workspace: the serving shell is not the champion, and a verdict
# reads CHAMPION from the results directory rather than from inside the bundle.
NOT_WORKSPACE = {
    "miner.py",
    "entrypoint.sh",
    "CHAMPION",
    "CHAMPION.agent",
    "PROVENANCE.md",
    "BUDGET.md",
    "CAMPAIGN.md",
    "workspace.bundle",
    "__pycache__",
}

PYPROJECT = """[project]
name = "campaign-workspace"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = ["synth-lib", "numpy>=2.2", "pandas>=2.3"]

[tool.uv.sources]
synth-lib = {{ git = "https://github.com/synthdataco/synth-lib", rev = "{rev}" }}
"""


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def reconstruct(leg_dir: Path, dest: Path, synth_lib_rev: str) -> tuple[Path, str]:
    """Build `<dest>/workspace.bundle` + `<dest>/CHAMPION` from a published leg."""
    if not (leg_dir / "modeling.py").exists():
        raise SystemExit(f"{leg_dir} has no modeling.py — is it a published leg?")
    dest.mkdir(parents=True, exist_ok=True)

    work = dest / "_workspace"
    shutil.rmtree(work, ignore_errors=True)
    agent = work / "agent"
    agent.mkdir(parents=True)

    for item in sorted(leg_dir.iterdir()):
        if item.name in NOT_WORKSPACE or item.name.startswith("verdict"):
            continue
        target = agent / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, target)
    (work / "pyproject.toml").write_text(PYPROJECT.format(rev=synth_lib_rev))

    run(["git", "init", "-q", "-b", "main"], work)
    run(["git", "add", "-A"], work)
    run(
        [
            "git",
            "-c", "user.name=autoquant",
            "-c", "user.email=noreply@synthdata.co",
            "commit", "-q", "-m",
            f"reconstructed from the published {leg_dir.parent.name}/{leg_dir.name} tree",
        ],
        work,
    )
    sha = run(["git", "rev-parse", "HEAD"], work)

    bundle = dest / "workspace.bundle"
    run(["git", "bundle", "create", str(bundle.resolve()), "--all"], work)
    shutil.rmtree(work, ignore_errors=True)

    original = (leg_dir / "CHAMPION").read_text() if (leg_dir / "CHAMPION").exists() else ""
    original_sha = next(
        (line.split(":", 1)[1].strip() for line in original.splitlines() if line.startswith("sha:")),
        "unknown",
    )
    (dest / "CHAMPION").write_text(
        f"sha: {sha}\n"
        "agent_dir: agent\n"
        "profiles: [low, high]\n"
        "notes: |\n"
        f"  Reconstructed from the published {leg_dir.parent.name}/{leg_dir.name} tree, whose own\n"
        f"  CHAMPION names sha {original_sha}. That commit is not recoverable — the workspace it\n"
        "  lived in was deleted and the bundle was never published — so this names the rebuilt\n"
        "  commit instead. The scored code is the published tree, which should be that commit\n"
        "  checked out, but this is an assumption rather than a proof. Say so beside any number\n"
        "  produced from it.\n"
    )
    return bundle, sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("campaign_results"))
    ap.add_argument("--out", type=Path, default=Path("rescore"), help="separate results root to write into")
    ap.add_argument(
        "--synth-lib-rev",
        required=True,
        help="rev for the workspace's own pyproject. The champion here imports only numpy/pandas, "
        "but the sandbox runs `uv sync` against this file, so it must resolve",
    )
    args = ap.parse_args()

    leg_dir = args.results_dir / args.campaign / args.leg
    dest = args.out / args.campaign / args.leg
    bundle, sha = reconstruct(leg_dir, dest, args.synth_lib_rev)
    size_mb = bundle.stat().st_size / 1e6
    print(f"wrote {bundle} ({size_mb:.1f} MB) at {sha[:7]}")
    print(f"wrote {dest / 'CHAMPION'}")
    print(f"\nscore it with:  --results-dir {args.out} --campaign {args.campaign} --legs {args.leg}")


if __name__ == "__main__":
    main()
