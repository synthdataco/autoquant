"""The leaderboard is generated from verdicts, so its format is pinned by a test.

If this breaks, the published README table and the archived verdicts have diverged — regenerate,
do not hand-edit.
"""

import json

from synth_bench.leaderboard import best_per_model, competition_cell, model_label, render, rows


def _leg(results, campaign, leg, *, score=None, mean_rank=None, champion=True, budget=True):
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


def test_competition_cell_shows_final_rank_and_mean():
    per = {"crypto-1h": {"rank": 7, "field_size": 250, "rank_over_rounds": {"mean_rank": 11.5}}}
    assert competition_cell(per, "crypto-1h") == "7/250 (μ11.5)"
    assert competition_cell(per, "missing") == "—"
    assert competition_cell({"x": {"rank": None}}, "x") == "n/a"


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

    table = render(tmp_path, tag=None)
    assert "| **30.1** |" in table and "7/250 (μ11.5)" in table
    assert "pending" in table and "no champion" in table
    assert table.count("|") > 20  # a table, not a stub
    assert "Spend" not in table  # cost is per-report, not per-row: a restarted leg resets its budget file


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

    table = render(tmp_path, tag=None)
    assert table.count("| some-model-5 |") == 1
    assert "8.1" in table and "6.7" not in table


def test_all_legs_shows_every_champion(tmp_path):
    _leg(tmp_path, "camp-a", "gemini", score=6.7)
    _leg(tmp_path, "camp-b", "gemini", score=8.1)
    table = render(tmp_path, tag=None, all_legs=True)
    assert "8.1" in table and "6.7" in table


def test_a_pending_leg_never_displaces_a_scored_one(tmp_path):
    """Same model, one campaign scored and one still pending: the score is what belongs on the table."""
    _leg(tmp_path, "camp-a", "gemini", score=8.1)
    _leg(tmp_path, "camp-b", "gemini")  # champion, no verdict yet

    kept = best_per_model(rows(tmp_path, tag=None))
    assert [(r["campaign"], r["score"]) for r in kept] == [("camp-a", 8.1)]
    assert "pending" not in render(tmp_path, tag=None)
