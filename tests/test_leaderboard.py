"""The leaderboard is generated from verdicts, so its format is pinned by a test.

If this breaks, the published README table and the archived verdicts have diverged — regenerate,
do not hand-edit.
"""

import json
import re
import sys

import pytest

from synth_bench.leaderboard import BEGIN, END, best_per_model, competition_cell, main, model_label, render, rows


def _squash(text: str) -> str:
    """Table cells without their padding, so a test reads `| cell |` whatever the column width."""
    return re.sub(r" {2,}\|", " |", text)


def _leg(results, campaign, leg, *, score=None, mean_rank=None, champion=True, budget=True, current=True):
    d = results / campaign / leg
    d.mkdir(parents=True)
    (results / campaign / "campaign.yaml").write_text(
        f"name: {campaign}\nmodels:\n  - {{id: {leg}, cli: claude-code, model: some-model-5}}\n"
    )
    if champion:
        (d / "CHAMPION").write_text("sha: abc1234\nagent_dir: agent\nprofiles: [low, high]\n")
    if budget:
        (d / "BUDGET.md").write_text("- Credits: 12.34 / 100.00 USD (12%)\n- Time   : 3.5 h elapsed\n")
    if score is not None:
        (d / "verdict.json").write_text(
            json.dumps(
                {
                    "score": score,
                    "mean_competition_rank": mean_rank,
                    "per_competition": {
                        "crypto-1h": {"rank": 7, "field_size": 250, "rank_over_rounds": {"mean_rank": 11.5}},
                        "crypto-24h": {"rank": 200, "field_size": 250, "rank_over_rounds": {"mean_rank": 190.0}},
                        "com-equ-24h": {"rank": None, "field_size": None, "rank_over_rounds": None},
                    },
                    "window": {"start": "2026-07-28", "end": "2026-08-07"},
                    **({"engine_rev": "298a95d", "live_prompts": True} if current else {}),
                }
            )
        )
    return d


def test_model_label_strips_alias_decoration():
    """Aliases name how a model was served, not what ran."""
    assert model_label("claude-opus-5-model") == "claude-opus-5"
    assert model_label("engy-glm-5.2") == "glm-5.2"
    assert model_label("gpt-5") == "gpt-5"
    assert model_label("gateway-gpt-6-astra") == "gpt-6-astra"


def test_competition_cell_leads_with_the_mean_rank():
    """The Score sums reward weight over every round, so the mean is its companion, not one round."""
    per = {"crypto-1h": {"rank": 7, "field_size": 250, "rank_over_rounds": {"mean_rank": 11.5}}}
    assert competition_cell(per, "crypto-1h") == "μ11.5 (final 7/250)"
    assert competition_cell(per, "missing") == "—"
    assert competition_cell({"x": {"rank": None}}, "x") == "n/a"
    # A verdict predating rank_over_rounds still renders, with the only rank it has.
    assert competition_cell({"x": {"rank": 7, "field_size": 250}}, "x") == "7/250"


def test_rows_sort_scored_first_and_carry_model_names(tmp_path):
    _leg(tmp_path, "camp-a", "low", score=4.2, mean_rank=180)
    _leg(tmp_path, "camp-a", "high", score=30.1, mean_rank=78)
    _leg(tmp_path, "camp-b", "unscored")  # champion but no verdict yet

    data = rows(tmp_path, tag=None)
    assert [r["leg"] for r in data] == ["high", "low", "unscored"]
    assert data[0]["score"] == 30.1 and data[0]["model"] == "some-model-5"
    assert data[-1]["score"] is None and data[-1]["landed"] is True


def test_render_marks_pending_and_no_champion(tmp_path):
    _leg(tmp_path, "camp", "scored", score=30.1, mean_rank=78)
    _leg(tmp_path, "camp", "pending")
    _leg(tmp_path, "camp", "failed", champion=False)

    table = _squash(render(tmp_path, tag=None))
    assert "| **30.1** |" in table and "μ11.5 (final 7/250)" in table
    assert "pending" in table and "no champion" in table
    assert table.count("|") > 20  # a table, not a stub
    assert "Spend" in table and "3.5 h / $12.34" in table


def test_spend_is_absent_rather_than_wrong(tmp_path):
    """No budget file, or one claiming a champion took zero hours, prints nothing.

    A restart leaves a fresh budget file that never got a poll; published verbatim it advertises a
    real Score for a few cents of spend.
    """
    from synth_bench.leaderboard import budget

    assert budget(_leg(tmp_path, "c", "nofile", budget=False)) == "—"
    stale = _leg(tmp_path, "c", "stale")
    (stale / "BUDGET.md").write_text("- Credits: 0.06 / 100.00 USD (0%)\n- Time   : 0.0 h elapsed\n")
    assert budget(stale) == "—"


def test_tagged_window_reads_its_own_file(tmp_path):
    d = _leg(tmp_path, "camp", "leg", score=30.1)
    (d / "verdict-is-0620.json").write_text(json.dumps({"score": 1.5, "per_competition": {}}))
    assert rows(tmp_path, tag="is-0620")[0]["score"] == 1.5
    assert rows(tmp_path, tag=None)[0]["score"] == 30.1


def test_one_row_per_model_keeps_its_best_score(tmp_path):
    """A model that entered two campaigns has two champions; listing both reads as two competitors
    rather than one entrant's best attempt. The rows arrive best-first, so the first wins."""
    _leg(tmp_path, "camp-a", "gemini", score=6.7)
    _leg(tmp_path, "camp-b", "gemini", score=8.1)

    kept = best_per_model(rows(tmp_path, tag=None))
    assert [(r["campaign"], r["score"]) for r in kept] == [("camp-b", 8.1)]

    scores, ranks = _squash(render(tmp_path, tag=None)).split("### Where each champion placed")
    assert scores.count("| some-model-5 |") == 1 and ranks.count("| some-model-5 |") == 1
    assert "8.1" in scores and "6.7" not in scores
    assert "One row per model" in scores


def test_all_legs_shows_every_champion(tmp_path):
    _leg(tmp_path, "camp-a", "gemini", score=6.7)
    _leg(tmp_path, "camp-b", "gemini", score=8.1)
    table = render(tmp_path, tag=None, all_legs=True)
    assert "8.1" in table and "6.7" in table
    assert "One row per champion" in table and "One row per model" not in table


def test_a_pending_leg_never_displaces_a_scored_one(tmp_path):
    """Same model, one campaign scored and one still pending: the score is what belongs on the table."""
    _leg(tmp_path, "camp-a", "gemini", score=8.1)
    _leg(tmp_path, "camp-b", "gemini")  # champion, no verdict yet

    kept = best_per_model(rows(tmp_path, tag=None))
    assert [(r["campaign"], r["score"]) for r in kept] == [("camp-a", 8.1)]
    assert "pending" not in render(tmp_path, tag=None)


def test_ranks_move_to_their_own_table_with_the_window(tmp_path):
    """A rank means nothing without the window it was measured over, and the campaigns differ."""
    _leg(tmp_path, "camp", "leg", score=30.1, mean_rank=78)

    scores, ranks = _squash(render(tmp_path, tag=None)).split("### Where each champion placed")
    assert "Spend" in scores and "crypto-1h" not in scores
    assert "| 2026-07-28 | 2026-08-07 |" in ranks
    assert "μ11.5 (final 7/250)" in ranks and "μ190 (final 200/250)" in ranks


def test_a_fallback_row_is_marked_in_both_tables(tmp_path):
    """The corrections moved ranks as well as Scores, so the mark has to follow the rows."""
    _leg(tmp_path, "camp", "leg", score=30.1, mean_rank=78)  # verdict.json only

    scores, ranks = _squash(render(tmp_path, tag="lead-in-2")).split("### Where each champion placed")
    assert "**30.1** \\*" in scores
    assert "| camp \\* |" in ranks


@pytest.mark.parametrize(
    "verdict", [{"engine_rev": None, "live_prompts": True}, {"engine_rev": "abc", "live_prompts": False}]
)
def test_a_verdict_not_scored_by_the_current_harness_is_marked(tmp_path, verdict):
    """The mark comes from what the verdict records, not from which campaign it belongs to."""
    d = _leg(tmp_path, "camp", "leg", score=30.1, mean_rank=78)
    v = json.loads((d / "verdict.json").read_text())
    (d / "verdict.json").write_text(json.dumps({**v, **verdict}))

    scores, ranks = _squash(render(tmp_path, tag=None)).split("### Where each champion placed")
    assert "**30.1** \\*" in scores and "| camp \\* |" in ranks


def test_a_current_verdict_is_not_marked(tmp_path):
    _leg(tmp_path, "camp", "leg", score=30.1, mean_rank=78)
    assert "\\*" not in render(tmp_path, tag=None).split("\n\n\\*")[0]


def test_both_tables_are_padded_to_their_widest_cell(tmp_path):
    _leg(tmp_path, "camp", "short", score=3.0, mean_rank=7)
    _leg(tmp_path, "another-campaign", "a-much-longer-leg-name", score=30.1, mean_rank=78)
    for part in render(tmp_path, tag=None).split("### Where each champion placed"):
        lines = [line for line in part.splitlines() if line.startswith("|")]
        assert len({len(line) for line in lines}) == 1, lines
        assert lines[1].startswith("| ---") and set(lines[1]) <= {"|", " ", "-"}


def test_write_leaves_a_blank_line_inside_each_marker(tmp_path, monkeypatch):
    _leg(tmp_path / "results", "camp", "leg", score=30.1, mean_rank=78)
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{BEGIN}\nold\n{END}\n\nAfter.\n")
    monkeypatch.setattr(
        sys, "argv", ["leaderboard", "--results-dir", str(tmp_path / "results"), "--write", str(readme)]
    )
    main()

    text = readme.read_text()
    block = text.split(BEGIN)[1].split(END)[0]
    assert block.startswith("\n\n| Model") and block.endswith(".\n\n")
    assert text.endswith(f"{END}\n\nAfter.\n")


def _with_failure(results, error, asset="XAU"):
    d = _leg(results, "camp", "leg", score=30.1, mean_rank=78)
    v = json.loads((d / "verdict.json").read_text())
    v["per_competition"]["crypto-24h"]["assets_failed"] = [asset]
    v["per_competition"]["crypto-24h"]["per_asset"] = {asset: {"error": error}}
    (d / "verdict.json").write_text(json.dumps(v))
    return d


def test_a_failed_asset_blocks_the_leaderboard(tmp_path):
    """A champion scored on fewer assets than the competition has does not have that Score."""
    _with_failure(tmp_path, "ArrowInvalid('cannot convert')")
    with pytest.raises(SystemExit, match="fewer assets"):
        render(tmp_path, tag=None)


def test_a_retired_asset_is_not_a_failure_after_it_retired(tmp_path):
    """SPYX stopped being prompted on 2026-07-23; the fixture window starts after that."""
    _with_failure(tmp_path, "NoScoresAvailable('no scored prompts')", asset="SPYX")
    assert "**30.1**" in render(tmp_path, tag=None)


def test_a_retired_asset_still_blocks_a_window_that_precedes_its_retirement(tmp_path):
    """Before 2026-07-23 the field WAS scored on SPYX, so a failure there is a real gap."""
    d = _with_failure(tmp_path, "NoScoresAvailable('no scored prompts')", asset="SPYX")
    v = json.loads((d / "verdict.json").read_text())
    v["window"]["start"] = "2026-07-01"
    (d / "verdict.json").write_text(json.dumps(v))
    with pytest.raises(SystemExit, match="SPYX"):
        render(tmp_path, tag=None)


def test_another_asset_failing_for_the_same_reason_still_blocks(tmp_path):
    """The old check keyed on the error string, so a genuine no-scores gap on a live asset passed."""
    _with_failure(tmp_path, "NoScoresAvailable('no scored prompts')", asset="XAU")
    with pytest.raises(SystemExit, match="XAU"):
        render(tmp_path, tag=None)


def test_a_hidden_leg_with_a_failed_asset_does_not_block(tmp_path):
    """One row per model is rendered; a leg the selection dropped is compared to nothing."""
    _leg(tmp_path, "camp-b", "gemini", score=8.1, mean_rank=40)  # the displayed best
    d = _with_failure(tmp_path, "ArrowInvalid('cannot convert')")  # camp/leg, score 30.1
    v = json.loads((d / "verdict.json").read_text())
    v["score"] = 1.0  # …and now the worst, so best_per_model drops it
    (d / "verdict.json").write_text(json.dumps(v))

    assert "**8.1**" in render(tmp_path, tag=None)
    with pytest.raises(SystemExit, match="XAU"):
        render(tmp_path, tag=None, all_legs=True)


def test_a_verdict_written_as_a_directory_is_read(tmp_path):
    """The runner now writes verdict-<tag>/ — the JSON with its charts and frames. The archived
    campaigns kept their flat verdict-<tag>.json, and both have to render."""
    d = _leg(tmp_path, "camp", "leg", score=30.1, mean_rank=78)
    tagged = d / "verdict-w-0917-0924"
    tagged.mkdir()
    (tagged / "verdict.json").write_text(json.dumps({"score": 42.0, "per_competition": {}}))

    assert rows(tmp_path, tag="w-0917-0924")[0]["score"] == 42.0
    assert rows(tmp_path, tag=None)[0]["score"] == 30.1  # the flat one is still the canonical
