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
LEGS_BEGIN = "<!-- LEGS:BEGIN -->"
LEGS_END = "<!-- LEGS:END -->"


def model_label(alias: str) -> str:
    """The model behind a proxy-served alias: aliases carry provider prefixes and local suffixes
    (`engy-glm-5.2`, `gateway-gpt-6-astra`, `claude-opus-5-model`) that name how it was served,
    not what ran."""
    return re.sub(r"-model$", "", re.sub(r"^(engy|gateway)-", "", alias))


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


def best_per_model(data: list[dict]) -> list[dict]:
    """One row per model: its best Score, and the campaign that produced it.

    A model that entered more than one campaign has more than one champion, and listing all of them
    reads as several competitors rather than one entrant's best attempt. The rows are already sorted
    best-first, so the first occurrence of a model wins; an unscored leg is kept only when that model
    has no scored one, so a "pending" row never displaces a real result.
    """
    seen: set[str] = set()
    kept = []
    for row in data:
        if row["model"] in seen:
            continue
        seen.add(row["model"])
        kept.append(row)
    return kept


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


def budget(leg_dir: Path) -> tuple[str, str]:
    """(wall time, spend) from the leg's archived BUDGET.md — the harness's own last poll.

    It records the LAST attempt, so a leg that was restarted reports that attempt only, and a leg
    that resumed on its original key reports cumulative spend against a clock that restarted. Both
    shapes are in the archive; neither is inferable from this file, which is why the table is
    footnoted rather than silently reconciled.
    """
    path = leg_dir / "BUDGET.md"
    if not path.exists():
        return "—", "—"
    text = path.read_text()
    spend = re.search(r"Credits:\s*([0-9.]+)\s*/", text)
    hours = re.search(r"Time\s*:\s*([0-9.]+)\s*h elapsed", text)
    # A scored champion cannot have been produced in zero elapsed time: the file is a fresh budget
    # from a restart that never got a poll. Report nothing rather than six cents.
    if hours and float(hours.group(1)) == 0.0:
        return "—", "—"
    return (f"{hours.group(1)} h" if hours else "—", f"${spend.group(1)}" if spend else "—")


def render_legs(results_dir: Path, tag: str | None) -> str:
    """Every leg, with what it cost — the per-leg companion to the one-row-per-model table."""
    lines = [
        "| Model | Campaign | Wall time | Spend | crypto-1h | crypto-24h | com-equ-24h |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows(results_dir, tag):
        hours, spend = budget(results_dir / r["campaign"] / r["leg"])
        lines.append(
            f"| {r['model']} | {r['campaign']} | {hours} | {spend} "
            f"| {competition_cell(r['per_competition'], 'crypto-1h')} "
            f"| {competition_cell(r['per_competition'], 'crypto-24h')} "
            f"| {competition_cell(r['per_competition'], 'com-equ-24h')} |"
        )
    lines += [
        "",
        "Wall time and spend are the harness's own last poll of `BUDGET.md`, against an envelope of",
        "$100 and 18 h (24 h in campaign-1). A leg that was restarted reports only its final attempt,",
        "so these are lower bounds where a campaign report says a leg ran twice.",
    ]
    return "\n".join(lines)


def render(results_dir: Path, tag: str | None, all_legs: bool = False) -> str:
    data = rows(results_dir, tag)
    if not all_legs:
        data = best_per_model(data)
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
        "One row per model — its best Score, and the campaign that produced it.",
        "Measured spend and wall time per leg are in each campaign's report.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("campaign_results"))
    ap.add_argument("--tag", default=None, help="read verdict-<tag>.json instead of verdict.json")
    ap.add_argument("--write", type=Path, default=None, help="replace the LEADERBOARD block in this file")
    ap.add_argument("--all-legs", action="store_true", help="every champion, not just each model's best")
    args = ap.parse_args()

    table = render(args.results_dir, args.tag, args.all_legs)
    if args.write is None:
        print(table)
        return
    text = args.write.read_text()
    for begin, end, block in ((BEGIN, END, table), (LEGS_BEGIN, LEGS_END, render_legs(args.results_dir, args.tag))):
        if begin not in text or end not in text:
            if begin == BEGIN:
                raise SystemExit(f"{args.write} has no {begin} / {end} markers")
            continue  # the per-leg block is optional
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{block}\n{end}{tail}"
    args.write.write_text(text)
    print(f"leaderboard written into {args.write}")


if __name__ == "__main__":
    main()
