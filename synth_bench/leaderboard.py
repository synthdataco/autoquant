"""Render the leaderboard from the archived verdicts.

The published table is generated, never hand-maintained: every number comes from a
`campaign_results/<campaign>/<leg>/verdict*.json` written by the scoring runner. A hand-written table
drifts from the verdicts within one campaign; this cannot.

Cost is deliberately not a column: a leg restarted after an interruption gets a fresh budget file, so
the archive records only its last attempt. Measured spend and wall time are in each campaign's report,
accounted across attempts.

    synth-bench-leaderboard                    # markdown to stdout
    synth-bench-leaderboard --tag is-0620      # a diagnostic window instead of the canonical one
    synth-bench-leaderboard --write README.md  # replace the block between the marker comments
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BEGIN = "<!-- LEADERBOARD:BEGIN -->"
END = "<!-- LEADERBOARD:END -->"


def model_label(alias: str) -> str:
    """The model behind a proxy-served alias: aliases carry provider prefixes and local suffixes
    (`engy-glm-5.2`, `claude-opus-5-model`) that name how it was served, not what ran."""
    return re.sub(r"-model$", "", re.sub(r"^engy-", "", alias))


def model_of(leg_dir: Path, campaign_dir: Path) -> str:
    """The model a leg ran, from the campaign config (the leg id is not the model)."""
    config = campaign_dir / "campaign.yaml"
    if not config.exists():
        return leg_dir.name
    for line in config.read_text().splitlines():
        if f"id: {leg_dir.name}" in line:
            match = re.search(r"model:\s*([A-Za-z0-9._-]+)", line)
            if match:
                return model_label(match.group(1))
    return leg_dir.name


def rows(results_dir: Path, tag: str | None) -> list[dict]:
    name = f"verdict-{tag}.json" if tag else "verdict.json"
    out: list[dict] = []
    for campaign_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for leg_dir in sorted(p for p in campaign_dir.iterdir() if p.is_dir() and p.name != "baselines"):
            champion = (leg_dir / "CHAMPION").exists()
            verdict = leg_dir / name
            row = {
                "campaign": campaign_dir.name,
                "leg": leg_dir.name,
                "model": model_of(leg_dir, campaign_dir),
                "landed": champion,
            }
            if verdict.exists():
                data = json.loads(verdict.read_text())
                row["score"] = data.get("score")
                row["mean_rank"] = data.get("mean_competition_rank")
                row["per_competition"] = data.get("per_competition") or {}
                row["window"] = (data.get("window") or {}).get("start", "")
            else:
                row["score"] = row["mean_rank"] = None
                row["per_competition"] = {}
                row["window"] = ""
            out.append(row)
    # Scored legs first, best Score first; unscored legs keep campaign order at the bottom.
    out.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0), r["campaign"], r["leg"]))
    return out


def competition_cell(per_competition: dict, slug: str) -> str:
    entry = per_competition.get(slug)
    if not entry:
        return "—"
    rank, size = entry.get("rank"), entry.get("field_size")
    over_rounds = entry.get("rank_over_rounds") or {}
    mean = over_rounds.get("mean_rank")
    if rank is None:
        return "n/a"
    cell = f"{rank}/{size}" if size else str(rank)
    return f"{cell} (μ{mean:g})" if mean else cell


def render(results_dir: Path, tag: str | None) -> str:
    data = rows(results_dir, tag)
    lines = [
        "| Model | Campaign | Score | Mean rank | crypto-1h | crypto-24h | com-equ-24h |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data:
        score = f"**{r['score']:g}**" if r["score"] is not None else ("pending" if r["landed"] else "no champion")
        mean_rank = f"{r['mean_rank']:g}" if r["mean_rank"] is not None else "—"
        lines.append(
            f"| {r['model']} | {r['campaign']} | {score} | {mean_rank} "
            f"| {competition_cell(r['per_competition'], 'crypto-1h')} "
            f"| {competition_cell(r['per_competition'], 'crypto-24h')} "
            f"| {competition_cell(r['per_competition'], 'com-equ-24h')} |"
        )
    lines += [
        "",
        "Score = 100 × mean over the three competitions of min(1, the champion's simulated emissions ÷",
        "the best other miner's) — see [docs/scoring.md](docs/scoring.md). Rank cells are the final-round",
        "rank out of the field, with the mean rank over the window's scoring rounds in parentheses.",
        "Measured spend and wall time per leg are in each campaign's report.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("campaign_results"))
    ap.add_argument("--tag", default=None, help="read verdict-<tag>.json instead of verdict.json")
    ap.add_argument("--write", type=Path, default=None, help="replace the LEADERBOARD block in this file")
    args = ap.parse_args()

    table = render(args.results_dir, args.tag)
    if args.write is None:
        print(table)
        return
    text = args.write.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{args.write} has no {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    args.write.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}")
    print(f"leaderboard written into {args.write}")


if __name__ == "__main__":
    main()
