"""Render the leaderboard from the archived verdicts.

The published table is generated, never hand-maintained: every number comes from a
`campaign_results/<campaign>/<leg>/verdict*.json` written by the scoring runner. A hand-written table
drifts from the verdicts within one campaign; this cannot.

Two tables come out of one pass over the same rows: what each champion earned, and where it placed.
They share Model + Campaign as their key.

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

COMPETITIONS = ("crypto-1h", "crypto-24h", "com-equ-24h")

# Assets the subnet retired, and the day it stopped prompting them. A failure on one of these in
# a window that starts after that day is a smaller competition; the same failure in an earlier
# window, or on any other asset, means the champion was scored on fewer assets than the validator
# uses and its Score is not the Score.
RETIRED_ASSETS = {"SPYX": "2026-07-23"}

# Marks a row scored before the harness corrections. It is appended in both tables and explained
# once, after them.
FALLBACK_MARK = " \\*"


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


def verdict_path(leg_dir: Path, tag: str | None) -> Path | None:
    """The verdict file for a tag, or None. A verdict is a directory holding its JSON, charts and
    frames; archived ones were written as a single flat file beside the leg."""
    stem = f"verdict-{tag}" if tag else "verdict"
    for candidate in (leg_dir / stem / "verdict.json", leg_dir / f"{stem}.json"):
        if candidate.exists():
            return candidate
    return None


def rows(results_dir: Path, tag: str | None) -> list[dict]:
    """One row per leg. With a tag, a leg that has no tagged verdict falls back to its canonical
    one and is flagged, so a partially rescored archive still renders — visibly mixed, not
    silently."""
    out: list[dict] = []
    for campaign_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for leg_dir in sorted(p for p in campaign_dir.iterdir() if p.is_dir() and p.name != "baselines"):
            champion = (leg_dir / "CHAMPION").exists()
            verdict = verdict_path(leg_dir, tag)
            fallback = bool(tag) and verdict is None and verdict_path(leg_dir, None) is not None
            if fallback:
                verdict = verdict_path(leg_dir, None)
            row = {
                "campaign": campaign_dir.name,
                "leg": leg_dir.name,
                "model": model_of(leg_dir, campaign_dir),
                "landed": champion,
                "fallback": fallback,
            }
            if verdict is not None:
                data = json.loads(verdict.read_text())
                window = data.get("window") or {}
                row["score"] = data.get("score")
                row["mean_rank"] = data.get("mean_competition_rank")
                row["per_competition"] = data.get("per_competition") or {}
                row["window_start"] = window.get("start", "")
                row["window_end"] = window.get("end", "")
            else:
                row["score"] = row["mean_rank"] = None
                row["per_competition"] = {}
                row["window_start"] = row["window_end"] = ""
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


def blocking_failures(row: dict) -> list[str]:
    """Failed assets that are not a retirement the window is entirely after."""
    out = []
    window_start = row.get("window_start") or ""
    for slug, entry in (row.get("per_competition") or {}).items():
        for asset in entry.get("assets_failed") or []:
            retired = RETIRED_ASSETS.get(asset)
            if retired is not None and window_start >= retired:
                continue
            error = ((entry.get("per_asset") or {}).get(asset) or {}).get("error", "")
            out.append(f"{row['campaign']}/{row['leg']} {slug}/{asset}: {error or 'no error recorded'}")
    return out


def competition_cell(per_competition: dict, slug: str) -> str:
    """Mean rank over the window's scoring rounds, with the final round in parentheses.

    The mean leads because the Score does: reward weight is summed over every round, so a single
    round's rank is the wrong companion to it — and on a 10-day window that is one draw out of a
    few hundred."""
    entry = per_competition.get(slug)
    if not entry:
        return "—"
    rank, size = entry.get("rank"), entry.get("field_size")
    mean = (entry.get("rank_over_rounds") or {}).get("mean_rank")
    if rank is None and mean is None:
        return "n/a"
    if mean is None:
        return f"{rank}/{size}" if size else str(rank)
    # The mean carries no denominator: field_size is the FINAL round's, and a mean taken while the
    # field was larger can exceed it — campaign-1 and -2 show means above their 191.
    final = f"{rank}/{size}" if size else f"{rank}"
    return f"μ{mean:g} (final {final})" if rank is not None else f"μ{mean:g}"


def budget(leg_dir: Path) -> str:
    """What the leg used of its envelope, as `<hours> h / $<spend>`, from its archived BUDGET.md.

    The file is the harness's own last poll, so a restarted leg reports only its final attempt and
    a leg resumed on its original key reports cumulative spend against a clock that restarted.
    Neither shape is inferable from the file, hence the footnote rather than a silent reconciliation.
    """
    path = leg_dir / "BUDGET.md"
    if not path.exists():
        return "—"
    text = path.read_text()
    spend = re.search(r"Credits:\s*([0-9.]+)\s*/", text)
    hours = re.search(r"Time\s*:\s*([0-9.]+)\s*h elapsed", text)
    # A scored champion cannot have been produced in zero elapsed time: the file is a fresh budget
    # from a restart that never got a poll. Report nothing rather than six cents.
    if not (spend and hours) or float(hours.group(1)) == 0.0:
        return "—"
    return f"{hours.group(1)} h / ${spend.group(1)}"


def render_scores(data: list[dict], results_dir: Path, all_legs: bool) -> str:
    """What each champion earned."""
    lines = [
        "| Model | Campaign | Score | Mean final rank | Spend |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in data:
        score = f"**{r['score']:g}**" if r["score"] is not None else ("pending" if r["landed"] else "no champion")
        if r["fallback"]:
            score += FALLBACK_MARK
        mean_rank = f"{r['mean_rank']:g}" if r["mean_rank"] is not None else "—"
        lines.append(
            f"| {r['model']} | {r['campaign']} | {score} | {mean_rank} "
            f"| {budget(results_dir / r['campaign'] / r['leg'])} |"
        )
    selection = (
        "One row per champion: a model that entered more than one campaign has a row for each."
        if all_legs
        else "One row per model: its best Score, and the campaign that produced it."
    )
    lines += [
        "",
        "Score = 100 × mean over the three competitions of min(1, the champion's simulated emissions ÷",
        "the best other miner's) — see [docs/scoring.md](docs/scoring.md).",
        selection,
        "Mean final rank averages the three competitions' final-round ranks — not the μ figures below,",
        "which average over rounds within one competition. Spend is wall time and credits from the",
        "leg's own budget file, against an envelope of $100 and 18 h (24 h in campaign-1); a leg that",
        "was restarted reports only its final attempt.",
    ]
    return "\n".join(lines)


def render_ranks(data: list[dict]) -> str:
    """Where each champion placed, and over which window."""
    lines = [
        "| Model | Campaign | Start | End | crypto-1h | crypto-24h | com-equ-24h |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data:
        campaign = r["campaign"] + (FALLBACK_MARK if r["fallback"] else "")
        cells = " ".join(f"| {competition_cell(r['per_competition'], slug)}" for slug in COMPETITIONS)
        lines.append(
            f"| {r['model']} | {campaign} | {r['window_start'] or '—'} | {r['window_end'] or '—'} " f"{cells} |"
        )
    lines += [
        "",
        "Start and End bound the verdict window: the forward, out-of-sample period the champion was",
        "scored over, which begins the day after its campaign's data cutoff. campaign-1 and -2 share a",
        "window, and campaign-3 and campaign-4 each ran on a later one, so two ranks are comparable only",
        "where the dates match. campaign-4's champions answered the start times the validator actually",
        "scored; the earlier campaigns answered an even grid of start times, which costs crypto-1h ranks",
        "most. Each cell is the MEAN rank over that window's scoring rounds, with the final round in",
        "parentheses out of the field it faced. The Score sums reward weight over every round, so the",
        "mean is what it corresponds to; the mean carries no denominator because the field size shown",
        "is the final round's, and the field was larger earlier in the window.",
    ]
    return "\n".join(lines)


def render(results_dir: Path, tag: str | None, all_legs: bool = False) -> str:
    data = rows(results_dir, tag)
    if not all_legs:
        data = best_per_model(data)
    # After the selection, not before: the objection is that an incomplete Score is not comparable
    # to a complete one, which only bites for a row that is actually rendered. A leg superseded by
    # a better one from the same model is compared to nothing.
    blocked = [line for row in data for line in blocking_failures(row)]
    if blocked:
        raise SystemExit(
            "refusing to render: a champion was scored on fewer assets than the competition has, "
            "so its Score is not comparable to a complete one.\n  " + "\n  ".join(blocked)
        )
    return "\n".join(
        [
            render_scores(data, results_dir, all_legs),
            "",
            "### Where each champion placed",
            "",
            render_ranks(data),
            "",
            "\\* scored before two harness corrections: the generation lead-in, which had been charging",
            "the candidate a horizon of worst-score rounds it was never given the chance to answer, and",
            "uniform realized truth, without which missing minutes in the operator's price store shrank",
            "the CRPS sum and flattered the champion. They pull in opposite directions and neither is",
            "uniform across champions, so a marked row is not comparable to an unmarked one, and the",
            "ordering between them is unreliable rather than merely optimistic or conservative.",
        ]
    )


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
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{args.write} has no {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    args.write.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}")
    print(f"leaderboard written into {args.write}")


if __name__ == "__main__":
    main()
