"""Tests for the variable-history retained phase-process evaluator core."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import learned_ai.evaluation.retained_phase_process_generalization as diagnostic
import tools.serve_retained_phase_process_generalization as web
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / ("docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json")


def _corpus() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _history(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def spec() -> dict:
    records = diagnostic.load_corpus_records(_corpus())
    schedule = diagnostic.build_schedule(records)
    body = {
        "schema_version": diagnostic.SPEC_SCHEMA,
        "diagnostic_id": "test-phase-process",
        "plan": {"identity": "p" * 64},
        "authorization": {"identity": "a" * 64},
        "implementation": {"commit": "c" * 40},
        "corpus": {"identity": _corpus()["corpus_identity"]},
        "runtime": {"seed": 42},
        "protocol": {
            "horizon_post_start_logical_plies": (
                diagnostic.HORIZON_POST_START_LOGICAL_PLIES
            ),
            "max_post_start_logical_plies": (diagnostic.MAX_POST_START_LOGICAL_PLIES),
        },
        "schedule": schedule,
    }
    return {**body, "spec_identity": canonical_sha256(body)}


def _strict_state(
    *,
    logical_ply: int,
    terminal: bool,
    history: str,
    no_capture_count: int,
) -> dict:
    return {
        "status": "terminal" if terminal else "ongoing",
        "fen": f"sanmill-fen-{logical_ply}",
        "terminal": terminal,
        "logical_ply_count": logical_ply,
        "no_capture_count": no_capture_count,
        "repetition_current_count": 1,
        "repetition_history_length": logical_ply + 1,
        "history_sha256": history,
        "outcome": {
            "terminal": terminal,
            "winner": "white" if terminal else None,
            "reason": "loseFewerThanThree" if terminal else "ongoing",
        },
    }


def _search(turn: dict) -> dict:
    return {
        "status": "ok",
        "full_turn_actions": list(turn["actions"]),
        "logical_move_id": "logical:test",
        "model_action": dict(turn["move"]),
        "logical_ply_delta": 1,
        "resulting_fen": turn["sanmill_fen_after"],
        "resulting_side_to_move": "B",
        "terminal": turn["terminal"],
        "winner": "white" if turn["terminal"] else None,
        "outcome_reason": turn["outcome_reason"],
        "effective_depth": 12,
        "completed_depth": 12,
        "score_kind": "cp",
        "score": 0,
        "score_perspective": "side-to-move",
        "node_budget": diagnostic.SANMILL_NODE_CEILING,
        "primary_nodes": 100,
        "removal_nodes": 0,
        "total_nodes": 100,
        "search_calls": 1,
    }


def _synthetic_ledger_record(
    spec: dict,
    ordinal: int,
    previous_hash: str | None,
    *,
    length: int,
) -> dict:
    item = spec["schedule"][ordinal]
    start_ply = int(item["start_logical_ply"])
    start_turn = str(item["start_turn"])
    candidate_color = str(item["candidate_color"])
    start_history = str(item["expected_start_history_sha256"])
    turns = []
    before = start_history
    for index in range(1, length + 1):
        mover = diagnostic._expected_mover(start_turn, index)
        actor = "candidate" if mover == candidate_color else "sanmill"
        terminal = index == length
        after = _history(f"{ordinal}:{index}")
        move = {
            "from": None,
            "to": "a7" if mover == "W" else "d7",
            "capture": None,
        }
        turn = {
            "post_start_logical_ply": index,
            "absolute_logical_ply": start_ply + index,
            "mover_color": mover,
            "actor": actor,
            "move": move,
            "actions": [move["to"]],
            "before_history_sha256": before,
            "after_history_sha256": after,
            "local_fen_after": f"local-fen-{ordinal}-{index}",
            "sanmill_fen_after": f"sanmill-fen-{ordinal}-{index}",
            "terminal": terminal,
            "outcome_reason": "loseFewerThanThree" if terminal else "ongoing",
            "no_capture_count": index,
            "repetition_current_count": 1,
            "repetition_history_length": start_ply + index + 1,
            "candidate_malom_delta": 0.0 if actor == "candidate" else None,
            "search": None,
        }
        if actor == "sanmill":
            turn["search"] = _search(turn)
        turns.append(turn)
        before = after

    survives = length > diagnostic.HORIZON_POST_START_LOGICAL_PLIES
    snapshot = None
    if survives:
        horizon = turns[diagnostic.HORIZON_POST_START_LOGICAL_PLIES - 1]
        snapshot_state = _strict_state(
            logical_ply=(start_ply + diagnostic.HORIZON_POST_START_LOGICAL_PLIES),
            terminal=False,
            history=horizon["after_history_sha256"],
            no_capture_count=diagnostic.HORIZON_POST_START_LOGICAL_PLIES,
        )
        snapshot = {
            "post_start_logical_ply": (diagnostic.HORIZON_POST_START_LOGICAL_PLIES),
            "absolute_logical_ply": (
                start_ply + diagnostic.HORIZON_POST_START_LOGICAL_PLIES
            ),
            "local_fen": horizon["local_fen_after"],
            "history_sha256": horizon["after_history_sha256"],
            "strict_referee_state": snapshot_state,
            "malom_theoretical": {
                "history_aware": False,
                "queryable": True,
                "side_to_move": "W",
                "side_to_move_wdl": "D",
                "candidate_color": candidate_color,
                "candidate_perspective_wdl": "D",
            },
        }
    start = {
        "start_id": item["start_id"],
        "start_record_identity": item["start_record_identity"],
        "expected_history_sha256": start_history,
        "observed_history_sha256": start_history,
        "logical_ply_count": start_ply,
        "action_token_count": start_ply,
        "logical_plies_by_side": [
            (start_ply + 1) // 2,
            start_ply // 2,
        ],
        "no_capture_count": 0,
        "repetition_current_count": 0,
        "repetition_history_length": start_ply + 1,
        "final_nmm_fen": "start-local",
        "final_sanmill_fen": "start-sanmill",
    }
    final = _strict_state(
        logical_ply=start_ply + length,
        terminal=True,
        history=before,
        no_capture_count=length,
    )
    return {
        "schema_version": diagnostic.GAME_SCHEMA,
        "spec_identity": spec["spec_identity"],
        "ordinal": ordinal,
        "unit_index": item["unit_index"],
        "game_id": item["game_id"],
        "match_key": item["match_key"],
        "candidate_id": item["candidate_id"],
        "candidate_color": candidate_color,
        "start_id": item["start_id"],
        "phase": item["phase"],
        "start": start,
        "ongoing_after_post_start_logical_ply_108": survives,
        "post_start_ply_108_snapshot": snapshot,
        "post_start_logical_plies": length,
        "total_logical_plies": start_ply + length,
        "termination_class": "rules_terminal",
        "outcome_reason": "loseFewerThanThree",
        "winner": "W",
        "candidate_score": 1.0 if candidate_color == "W" else 0.0,
        "final_state": final,
        "history_process": diagnostic._history_process(start, snapshot, final),
        "turns": turns,
        "candidate_malom": diagnostic._candidate_malom_summary(turns),
        "sanmill_search": diagnostic._search_summary(turns),
        "game_elapsed_seconds": 0.25,
        "cumulative_active_seconds": float(ordinal + 1),
        "complete_diagnostic": True,
        "previous_record_sha256": previous_hash,
    }


def _summary_record(item: dict, *, survives: bool) -> dict:
    start_ply = int(item["start_logical_ply"])
    length = 109 if survives else 30
    start_process = {
        "no_capture_count": 2,
        "repetition_current_count": 1,
        "repetition_history_length": start_ply + 1,
    }
    horizon_process = (
        {
            "no_capture_count": 50,
            "repetition_current_count": 1,
            "repetition_history_length": start_ply + 109,
        }
        if survives
        else None
    )
    final_process = {
        "no_capture_count": 60,
        "repetition_current_count": 1,
        "repetition_history_length": start_ply + length + 1,
    }
    snapshot = (
        {
            "malom_theoretical": {
                "candidate_perspective_wdl": "D",
            }
        }
        if survives
        else None
    )
    return {
        **item,
        "ongoing_after_post_start_logical_ply_108": survives,
        "post_start_ply_108_snapshot": snapshot,
        "post_start_logical_plies": length,
        "total_logical_plies": start_ply + length,
        "termination_class": "rules_terminal",
        "outcome_reason": "drawFiftyMove",
        "candidate_score": 0.5,
        "candidate_malom": {
            "candidate_turns": 10,
            "queryable_turns": 10,
            "unqueryable_turns": 0,
            "preserving_turns": 10,
            "one_step_downgrade_turns": 0,
            "two_step_downgrade_turns": 0,
            "preserving_rate_given_queryable": 1.0,
        },
        "history_process": {
            "start": start_process,
            "horizon": horizon_process,
            "final": final_process,
        },
    }


def test_schedule_pairs_both_candidates_inside_start_and_colour(spec) -> None:
    schedule = spec["schedule"]
    assert len(schedule) == diagnostic.EXPECTED_GAMES == 156
    assert {item["phase"] for item in schedule} == {
        "placement",
        "movement",
        "flying",
    }
    for unit_index in range(diagnostic.EXPECTED_MATCHED_COLOUR_UNITS):
        v3, v4 = schedule[unit_index * 2 : unit_index * 2 + 2]
        assert v3["unit_index"] == v4["unit_index"] == unit_index
        assert v3["match_key"] == v4["match_key"]
        assert v3["candidate_color"] == v4["candidate_color"]
        assert v3["candidate_id"] == diagnostic.EXPECTED_CANDIDATES[0]
        assert v4["candidate_id"] == diagnostic.EXPECTED_CANDIDATES[1]


def test_relative_horizon_uses_start_ply_not_absolute_ply() -> None:
    record = _corpus()["records"][0]
    start_ply = record["strict_start"]["logical_ply_count"]
    absolute = start_ply + diagnostic.HORIZON_POST_START_LOGICAL_PLIES

    class Malom:
        @staticmethod
        def query_state(_board):
            return "W"

    state_record = _strict_state(
        logical_ply=absolute,
        terminal=False,
        history=_history("relative-horizon"),
        no_capture_count=17,
    )
    state = type(
        "State",
        (),
        {
            "logical_ply_count": absolute,
            "terminal": False,
            "history_sha256": state_record["history_sha256"],
            "portable_record": staticmethod(lambda: state_record),
        },
    )()
    board = diagnostic.BoardState.from_fen_string(record["fen"])
    snapshot = diagnostic._snapshot_at_horizon(
        board=board,
        state=state,
        candidate_color="B",
        malom=Malom(),
        start_logical_ply=start_ply,
    )
    assert snapshot["post_start_logical_ply"] == 108
    assert snapshot["absolute_logical_ply"] == absolute
    assert snapshot["strict_referee_state"]["no_capture_count"] == 17
    assert snapshot["malom_theoretical"]["history_aware"] is False


def test_complete_report_clusters_both_colours_at_start_level(spec) -> None:
    records = []
    for item in spec["schedule"]:
        records.append(
            _summary_record(
                item,
                survives=(item["candidate_id"] == diagnostic.EXPECTED_CANDIDATES[1]),
            )
        )
    report = diagnostic.summarize_records(spec, records, "f" * 64)
    primary = report["paired"]["primary_start_clustered_108_ply_survival_v4_minus_v3"]
    assert report["status"] == "completed"
    assert primary["support"] == 39
    assert primary["mean"] == 1.0
    assert primary["interval"] == [1.0, 1.0]
    assert primary["distribution"] == {"1.0": 39}
    assert primary["decision"] == ("v4_higher_108_post_start_ply_survival")
    assert (
        report["by_candidate"][diagnostic.EXPECTED_CANDIDATES[1]]["history_process"][
            "horizon_no_capture"
        ]["support"]
        == 78
    )
    assert report["claim_boundary"]["held_out"] is False
    assert report["claim_boundary"]["playing_strength_claim"] is False


def test_partial_ledger_round_trips_variable_start_and_history(tmp_path, spec) -> None:
    path = tmp_path / "games.jsonl"
    first = _synthetic_ledger_record(spec, 0, None, length=109)
    tail = diagnostic.append_game_record(path, first, must_create=True)
    second = _synthetic_ledger_record(spec, 1, tail, length=30)
    diagnostic.append_game_record(path, second, must_create=False)

    records, observed_tail = diagnostic.load_game_ledger(spec, path)
    report = diagnostic.recompute_report(spec, path)
    assert records == [first, second]
    assert observed_tail == canonical_sha256(second)
    assert report["completed_games"] == 2
    assert report["paired"]["matched_colour_units_complete"] == 1
    assert report["paired"]["start_units_complete"] == 0
    assert (
        report["paired"]["primary_start_clustered_108_ply_survival_v4_minus_v3"][
            "decision"
        ]
        == "pending"
    )


def test_ledger_rejects_a_skipped_relative_horizon(tmp_path, spec) -> None:
    record = _synthetic_ledger_record(spec, 0, None, length=109)
    record["ongoing_after_post_start_logical_ply_108"] = False
    record["post_start_ply_108_snapshot"] = None
    record["history_process"] = diagnostic._history_process(
        record["start"],
        None,
        record["final_state"],
    )
    path = tmp_path / "bad.jsonl"
    diagnostic.append_game_record(path, record, must_create=True)
    with pytest.raises(
        diagnostic.RetainedPhaseProcessError,
        match="long game omitted relative-horizon survival",
    ):
        diagnostic.load_game_ledger(spec, path)


def test_ledger_uses_the_frozen_start_turn_for_actor_order(tmp_path, spec) -> None:
    record = _synthetic_ledger_record(spec, 0, None, length=30)
    record["turns"][0]["mover_color"] = (
        "B" if record["turns"][0]["mover_color"] == "W" else "W"
    )
    path = tmp_path / "bad-turn.jsonl"
    diagnostic.append_game_record(path, record, must_create=True)
    with pytest.raises(
        diagnostic.RetainedPhaseProcessError,
        match="actor order differs",
    ):
        diagnostic.load_game_ledger(spec, path)


def test_web_does_not_invent_metrics_before_a_spec_exists(tmp_path) -> None:
    assert web.build_payload(
        tmp_path,
        heldout_plan_path=None,
        heldout_output_root=None,
    ) == {
        "available": False,
        "status": "not_started",
        "message": "诊断尚未启动；没有伪造或预填结果。",
        "expected_games": 156,
        "expected_starts": 39,
        "heldout_score": None,
    }
    assert "相对 108 手" in web.HTML
    assert "不是和棋" in web.HTML
    assert "history-free" in web.HTML
    assert "不是 held-out" in web.HTML
    assert "cap" in web.HTML
    assert "跨语料复现" in web.HTML
    assert "事后工程描述" in web.HTML
    assert "配对得分主指标" in web.HTML
    assert "真正 held-out 候选盲源池" in web.HTML
    assert "源池可用" in web.HTML
    assert "held-out 高精度得分方案" in web.HTML
    assert "跨 0 只能判“不确定”" in web.HTML


def test_web_exposes_only_validated_heldout_pool_summary() -> None:
    source = web._heldout_pool_payload(web.DEFAULT_HELDOUT_POOL)

    assert source is not None
    assert source["available"] is True
    assert source["pool_identity"] == (
        "2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7"
    )
    assert source["independent_starts"] == 361
    assert source["phase_counts"] == {
        "flying": 56,
        "movement": 152,
        "placement": 153,
    }
    assert source["strict_replay"] == {
        "repeat_passes": 2,
        "fresh_process_count": 722,
        "accepted_count": 361,
        "excluded_count": 0,
    }
    assert "records" not in source


def test_web_shows_selected_heldout_plan_without_inventing_results(
    tmp_path,
) -> None:
    body = {
        "schema_version": "nmm.retained-heldout-score-plan.v1",
        "workload": {
            "games": 1012,
            "unique_starts": 253,
            "max_active_hours": 4.0,
        },
        "corpus": {"phase_counts": {"placement": 99, "movement": 98, "flying": 56}},
        "analysis": {"engineering_interval": {"maximum_primary_half_width": 0.015}},
    }
    plan = {**body, "plan_identity": canonical_sha256(body)}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    summary = web._heldout_score_payload(plan_path, tmp_path / "output")

    assert summary["status"] == "awaiting_authorization"
    assert summary["selected_starts"] == 253
    assert summary["expected_games"] == 1012
    assert summary["maximum_primary_half_width"] == 0.015
    assert summary["authorization_present"] is False
    assert summary["primary"] is None


def test_web_clusters_score_by_start_before_computing_precision() -> None:
    records = []
    candidate_ids = diagnostic.EXPECTED_CANDIDATES
    values = {
        ("start-a", "W"): (0.0, 1.0),
        ("start-a", "B"): (0.5, 0.5),
        ("start-b", "W"): (1.0, 0.0),
        ("start-b", "B"): (0.5, 0.5),
    }
    for (start_id, colour), scores in values.items():
        for candidate_id, score in zip(candidate_ids, scores, strict=True):
            records.append(
                {
                    "start_id": start_id,
                    "candidate_color": colour,
                    "candidate_id": candidate_id,
                    "candidate_score": score,
                    "termination_class": "rules_terminal",
                }
            )

    result = web._start_clustered_precision(
        records,
        start_key="start_id",
        value_key="candidate_score",
        require_rules_terminal=True,
    )
    assert result["support"] == 2
    assert result["matched_colour_units"] == 4
    assert result["mean"] == 0.0
    assert result["distribution"] == {"-0.5": 1, "0.5": 1}


def test_web_cross_corpus_contrast_and_conservative_score_budget() -> None:
    development = {
        "support": 64,
        "mean": 0.078125,
        "standard_error": 0.03729898172468536,
    }
    phase = {
        "support": 39,
        "mean": -0.02564102564102564,
        "standard_error": 0.017890787562163092,
    }
    contrast = web._independent_fixed_corpus_contrast(phase, development)
    assert contrast["mean"] == pytest.approx(-0.10376602564102563)
    assert contrast["interval"] == pytest.approx(
        [-0.18484690038539592, -0.022685150896655362]
    )
    assert contrast["post_hoc"] is True

    planning = web._score_planning_budgets([0.08249203304485238, 0.12149262874514737])
    assert planning == {
        "conservative_sample_standard_deviation": 0.12149262874514737,
        "rows": [
            {"target_half_width": 0.03, "starts": 64, "games": 256},
            {"target_half_width": 0.02, "starts": 142, "games": 568},
            {"target_half_width": 0.015, "starts": 253, "games": 1012},
            {"target_half_width": 0.01, "starts": 568, "games": 2272},
        ],
        "planning_only": True,
    }


def test_embedded_phase_process_web_javascript_parses_with_node() -> None:
    script = web.HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    result = subprocess.run(
        ["node", "--check", "-"],
        input=script,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr


def test_web_recomputes_partial_ledger_and_start_support(tmp_path, spec) -> None:
    (tmp_path / "spec.json").write_bytes(canonical_json_bytes(spec) + b"\n")
    first = _synthetic_ledger_record(spec, 0, None, length=109)
    tail = diagnostic.append_game_record(
        tmp_path / "games.jsonl",
        first,
        must_create=True,
    )
    second = _synthetic_ledger_record(spec, 1, tail, length=30)
    diagnostic.append_game_record(
        tmp_path / "games.jsonl",
        second,
        must_create=False,
    )

    payload = web.build_payload(tmp_path)
    assert payload["available"] is True
    assert payload["report"]["completed_games"] == 2
    assert payload["report"]["paired"]["matched_colour_units_complete"] == 1
    assert payload["report"]["paired"]["start_units_complete"] == 0
    assert payload["precision"]["fixed_width_budgets"] == []
    assert payload["mechanism"] is None
    assert payload["identities"]["corpus_identity"] == _corpus()["corpus_identity"]


def test_web_rejects_a_tampered_spec_identity(tmp_path, spec) -> None:
    tampered = {**spec, "diagnostic_id": "tampered"}
    (tmp_path / "spec.json").write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(ValueError, match="spec identity differs"):
        web.build_payload(tmp_path)


def test_web_reads_only_an_identity_bound_zero_game_mechanism_report(
    tmp_path,
    spec,
    monkeypatch,
) -> None:
    (tmp_path / "spec.json").write_bytes(canonical_json_bytes(spec) + b"\n")
    ledger = tmp_path / "games.jsonl"
    ledger.write_bytes(b"fixed-ledger\n")
    source_result = "r" * 64
    primary = {
        "support": 39,
        "mean": 0.0,
        "sample_standard_deviation": 0.0,
        "standard_error": 0.0,
        "interval": [0.0, 0.0],
        "half_width": 0.0,
        "distribution": {"0.0": 39},
    }
    monkeypatch.setattr(
        web,
        "load_game_ledger",
        lambda _spec, _ledger: ([], None),
    )
    monkeypatch.setattr(
        web,
        "summarize_records",
        lambda _spec, _records, _tail: {
            "completed_games": 156,
            "result_identity": source_result,
            "paired": {
                "primary_start_clustered_108_ply_survival_v4_minus_v3": primary,
            },
        },
    )
    body = {
        "schema_version": web.MECHANISM_SCHEMA,
        "source": {
            "diagnostic_id": spec["diagnostic_id"],
            "spec_identity": spec["spec_identity"],
            "ledger_sha256": hashlib.sha256(b"fixed-ledger\n").hexdigest(),
            "result_identity": source_result,
            "games": 156,
            "new_games": 0,
        },
        "by_candidate": {},
        "paired": {},
    }
    mechanism = {**body, "result_identity": canonical_sha256(body)}
    (tmp_path / "mechanism-report.json").write_bytes(
        canonical_json_bytes(mechanism) + b"\n"
    )
    payload = web.build_payload(tmp_path)
    assert payload["mechanism"] == mechanism

    mechanism["source"]["new_games"] = 1
    changed_body = {
        key: value for key, value in mechanism.items() if key != "result_identity"
    }
    mechanism["result_identity"] = canonical_sha256(changed_body)
    (tmp_path / "mechanism-report.json").write_bytes(
        canonical_json_bytes(mechanism) + b"\n"
    )
    with pytest.raises(ValueError, match="source binding differs"):
        web.build_payload(tmp_path)
