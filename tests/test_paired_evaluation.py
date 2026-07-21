from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation import paired_protocol
from learned_ai.evaluation.paired_protocol import EvaluationError, EvaluationSpec, freeze_evaluation_spec, recompute_evaluation
from learned_ai.training.run_contract import canonical_sha256


def _spec() -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id="eval-1", candidate_bundle="a" * 64, baseline_bundle="b" * 64,
        start_positions=("........................|W|0|0",), pairs=1, seed=7,
        work_budget={"lookahead_rollouts_per_move": 0}, max_ply=10,
        rules_version="nmm-v4-corrected", confidence_z=1.96,
        acceptance_margin=0.0, rejection_margin=0.0, runtime={"device": "single"},
    )


def _records(path: Path, spec: EvaluationSpec) -> None:
    previous = None
    rows = []
    for game, score in enumerate((1.0, 0.0)):
        game_id = "eval-game:" + canonical_sha256({"spec": spec.spec_identity, "pair": 0, "game": game})
        record = {"schema_version": "nmm.evaluation-game.v1", "spec_identity": spec.spec_identity,
            "pair": 0, "game": game, "game_id": game_id, "seed": game,
            "start_fen": spec.start_positions[0], "candidate_color": "W" if game == 0 else "B",
            "winner": "W", "candidate_score": score, "ply": 5, "terminal_reason": "rules_terminal",
            "complete": True, "previous_record_sha256": previous}
        record_hash = canonical_sha256(record)
        rows.append(json.dumps({"record": record, "record_sha256": record_hash}, sort_keys=True))
        previous = record_hash
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_recompute_validates_complete_paired_ledger(tmp_path: Path) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    freeze_evaluation_spec(spec_path, spec)
    _records(records_path, spec)

    result = recompute_evaluation(spec_path, records_path)

    assert result["games"] == 2
    assert result["decision"] == "inconclusive"
    assert len(result["result_identity"]) == 64


def test_recompute_rejects_duplicate_or_missing_game(tmp_path: Path) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    freeze_evaluation_spec(spec_path, spec)
    _records(records_path, spec)
    first = records_path.read_text(encoding="utf-8").splitlines()[0]
    records_path.write_text(first + "\n" + first + "\n", encoding="utf-8")

    with pytest.raises(EvaluationError):
        recompute_evaluation(spec_path, records_path)


def test_spec_rejects_non_frozen_work_budget() -> None:
    with pytest.raises(EvaluationError, match="zero-rollout"):
        EvaluationSpec(
            evaluation_id="eval", candidate_bundle="a" * 64, baseline_bundle="b" * 64,
            start_positions=("........................|W|0|0",), pairs=1, seed=1,
            work_budget={"lookahead_rollouts_per_move": 1}, max_ply=10,
            rules_version="v4", confidence_z=1.96, acceptance_margin=0.0,
            rejection_margin=0.0, runtime={},
        )


def _patch_runner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    spec: EvaluationSpec,
    policy_type: type,
    engine_type: type,
) -> None:
    def fake_load_bundle(path: str | Path, *, device: str):
        identity = spec.candidate_bundle if str(path) == "candidate" else spec.baseline_bundle
        return object(), {"bundle_identity": identity}

    monkeypatch.setattr(paired_protocol, "load_bundle_model", fake_load_bundle)
    monkeypatch.setattr(paired_protocol, "_BundlePolicy", policy_type)
    monkeypatch.setattr(paired_protocol, "GameEngine", engine_type)
    monkeypatch.setattr(paired_protocol, "is_terminal", lambda _board: (False, None))


@pytest.mark.parametrize("draw_reason", ("repetition", "50-move rule"))
def test_run_stops_on_engine_level_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draw_reason: str,
) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    freeze_evaluation_spec(spec_path, spec)

    class ConstantPolicy:
        def __init__(self, _model, _device: str) -> None:
            pass

        def choose_move(self, _board) -> dict[str, str]:
            return {"to": "a7"}

    class DrawEngine:
        def __init__(self, *, human_color) -> None:
            self.board = None
            self.finished = False
            self.winner = None
            self.draw_reason = None

        def apply_move(self, _move: dict[str, str]) -> None:
            if self.finished:
                raise ValueError("Game is already over.")
            self.finished = True
            self.winner = None
            self.draw_reason = draw_reason

    _patch_runner_dependencies(monkeypatch, spec, ConstantPolicy, DrawEngine)

    result = paired_protocol.run_paired_evaluation(
        spec_path, "candidate", "baseline", records_path
    )

    assert result["games"] == 2
    assert result["draws"] == 2
    records = [json.loads(line)["record"] for line in records_path.read_text().splitlines()]
    assert [record["terminal_reason"] for record in records] == [draw_reason, draw_reason]
    assert [record["candidate_score"] for record in records] == [0.5, 0.5]


def test_run_resumes_valid_partial_ledger_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    partial_path = Path(f"{records_path}.partial")
    freeze_evaluation_spec(spec_path, spec)
    state = {"choose_calls": 0, "fail_on_call": 2}

    class InterruptiblePolicy:
        def __init__(self, _model, _device: str) -> None:
            pass

        def choose_move(self, _board) -> dict[str, str]:
            state["choose_calls"] += 1
            if state["choose_calls"] == state["fail_on_call"]:
                raise RuntimeError("simulated interruption")
            return {"to": "a7"}

    class OneMoveWinEngine:
        def __init__(self, *, human_color) -> None:
            self.board = None
            self.finished = False
            self.winner = None
            self.draw_reason = None

        def apply_move(self, _move: dict[str, str]) -> None:
            self.finished = True
            self.winner = "W"
            self.draw_reason = None

    _patch_runner_dependencies(monkeypatch, spec, InterruptiblePolicy, OneMoveWinEngine)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        paired_protocol.run_paired_evaluation(
            spec_path, "candidate", "baseline", records_path
        )

    assert not records_path.exists()
    assert partial_path.exists()
    assert len(partial_path.read_text(encoding="utf-8").splitlines()) == 1

    state["choose_calls"] = 0
    state["fail_on_call"] = -1
    result = paired_protocol.run_paired_evaluation(
        spec_path, "candidate", "baseline", records_path
    )

    assert state["choose_calls"] == 1
    assert result["games"] == 2
    assert records_path.exists()
    assert not partial_path.exists()


def test_run_rejects_malformed_partial_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    partial_path = Path(f"{records_path}.partial")
    freeze_evaluation_spec(spec_path, spec)
    malformed = '{"record": {}}\n'
    partial_path.write_text(malformed, encoding="utf-8")

    class ConstantPolicy:
        def __init__(self, _model, _device: str) -> None:
            pass

        def choose_move(self, _board) -> dict[str, str]:
            return {"to": "a7"}

    class OneMoveWinEngine:
        def __init__(self, *, human_color) -> None:
            self.board = None
            self.finished = False
            self.winner = None
            self.draw_reason = None

        def apply_move(self, _move: dict[str, str]) -> None:
            self.finished = True
            self.winner = "W"
            self.draw_reason = None

    _patch_runner_dependencies(monkeypatch, spec, ConstantPolicy, OneMoveWinEngine)

    with pytest.raises(EvaluationError, match="malformed game record"):
        paired_protocol.run_paired_evaluation(
            spec_path, "candidate", "baseline", records_path
        )

    assert not records_path.exists()
    assert partial_path.read_text(encoding="utf-8") == malformed
