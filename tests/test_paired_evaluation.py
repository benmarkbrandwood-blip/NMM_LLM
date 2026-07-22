from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation import paired_protocol
from learned_ai.evaluation.paired_protocol import EvaluationError, EvaluationSpec, freeze_evaluation_spec, recompute_evaluation
from learned_ai.training.run_contract import canonical_sha256


def _spec(*, runtime: dict[str, str] | None = None) -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id="eval-1", candidate_bundle="a" * 64, baseline_bundle="b" * 64,
        start_positions=("........................|W|0|0",), pairs=1, seed=7,
        work_budget={"lookahead_rollouts_per_move": 0}, max_ply=10,
        rules_version="nmm-v4-corrected", confidence_z=1.96,
        acceptance_margin=0.0, rejection_margin=0.0,
        runtime={"device": "single"} if runtime is None else runtime,
    )


def _bound_runtime(*, device: str = "cpu", commit: str = "c" * 40) -> dict[str, str]:
    return {
        "schema_version": "nmm.paired-runtime.v1",
        "git_commit": commit,
        "git_tree": "clean",
        "platform": "test-platform",
        "pytorch": "test-pytorch",
        "device": device,
        "device_index": "none" if device == "cpu" else "0",
        "device_name": "test-device",
        "precision": "float32",
        "route": "policy-argmax-v1",
        "components": (
            "sentinel=off,value_net=off,gap_net=off,"
            "human_db=off,specialist_db=off"
        ),
        "lookahead_features": "zeroed-72",
    }


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


def test_spec_rejects_more_pairs_than_unique_starts() -> None:
    with pytest.raises(EvaluationError, match="cannot exceed unique starts"):
        EvaluationSpec(
            evaluation_id="eval", candidate_bundle="a" * 64,
            baseline_bundle="b" * 64,
            start_positions=("........................|W|0|0",),
            pairs=2, seed=1,
            work_budget={"lookahead_rollouts_per_move": 0}, max_ply=10,
            rules_version="v4", confidence_z=1.96,
            acceptance_margin=0.0, rejection_margin=0.0,
            runtime={},
        )


def test_spec_rejects_duplicate_start_positions() -> None:
    fen = "........................|W|0|0"
    with pytest.raises(EvaluationError, match="start positions must be unique"):
        EvaluationSpec(
            evaluation_id="eval", candidate_bundle="a" * 64,
            baseline_bundle="b" * 64,
            start_positions=(fen, fen), pairs=1, seed=1,
            work_budget={"lookahead_rollouts_per_move": 0}, max_ply=10,
            rules_version="v4", confidence_z=1.96,
            acceptance_margin=0.0, rejection_margin=0.0,
            runtime={},
        )


def test_spec_rejects_unknown_bound_runtime_fields() -> None:
    runtime = {**_bound_runtime(), "unexpected": "value"}
    with pytest.raises(EvaluationError, match="runtime contract fields"):
        EvaluationSpec(
            evaluation_id="eval", candidate_bundle="a" * 64,
            baseline_bundle="b" * 64,
            start_positions=("........................|W|0|0",),
            pairs=1, seed=1,
            work_budget={"lookahead_rollouts_per_move": 0}, max_ply=10,
            rules_version="v4", confidence_z=1.96,
            acceptance_margin=0.0, rejection_margin=0.0,
            runtime=runtime,
        )


def test_build_runtime_identity_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "c" * 40,
        ("status", "--short", "--untracked-files=all"): " M tracked.py",
    }
    monkeypatch.setattr(
        paired_protocol,
        "_git_output",
        lambda *args: outputs[args],
    )

    with pytest.raises(EvaluationError, match="clean Git tree"):
        paired_protocol.build_runtime_identity("cpu")


def test_run_rejects_legacy_unbound_runtime_before_loading_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.json"
    freeze_evaluation_spec(spec_path, _spec())

    def unexpected_bundle_load(*_args, **_kwargs):
        raise AssertionError("legacy run reached bundle loading")

    monkeypatch.setattr(
        paired_protocol,
        "load_bundle_model",
        unexpected_bundle_load,
    )

    with pytest.raises(EvaluationError, match="legacy.*runtime"):
        paired_protocol.run_paired_evaluation(
            spec_path,
            "candidate",
            "baseline",
            tmp_path / "games.jsonl",
            device="cpu",
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
    monkeypatch.setattr(
        paired_protocol,
        "build_runtime_identity",
        lambda _device: spec.runtime,
    )


def test_run_rejects_bound_device_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(runtime=_bound_runtime(device="cpu"))
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    freeze_evaluation_spec(spec_path, spec)

    class EmptyPolicy:
        def __init__(self, _model, _device: str) -> None:
            pass

        def choose_move(self, _board) -> dict:
            return {}

    class MinimalEngine:
        def __init__(self, *, human_color) -> None:
            self.board = None
            self.finished = False
            self.winner = None
            self.draw_reason = None

    _patch_runner_dependencies(monkeypatch, spec, EmptyPolicy, MinimalEngine)

    with pytest.raises(EvaluationError, match="device.*frozen"):
        paired_protocol.run_paired_evaluation(
            spec_path,
            "candidate",
            "baseline",
            records_path,
            device="cuda",
        )


def test_run_rejects_bound_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _bound_runtime(device="cpu")
    spec = _spec(runtime=runtime)
    spec_path = tmp_path / "spec.json"
    freeze_evaluation_spec(spec_path, spec)
    monkeypatch.setattr(
        paired_protocol,
        "build_runtime_identity",
        lambda _device: {**runtime, "pytorch": "changed-pytorch"},
    )

    with pytest.raises(EvaluationError, match="frozen spec.*pytorch"):
        paired_protocol.run_paired_evaluation(
            spec_path,
            "candidate",
            "baseline",
            tmp_path / "games.jsonl",
            device="cpu",
        )


def test_run_accepts_matching_bound_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _bound_runtime(device="cpu")
    spec = _spec(runtime=runtime)
    spec_path = tmp_path / "spec.json"
    records_path = tmp_path / "games.jsonl"
    freeze_evaluation_spec(spec_path, spec)

    class EmptyPolicy:
        def __init__(self, _model, _device: str) -> None:
            pass

        def choose_move(self, _board) -> dict:
            return {}

    class MinimalEngine:
        def __init__(self, *, human_color) -> None:
            self.board = None
            self.finished = False
            self.winner = None
            self.draw_reason = None

    _patch_runner_dependencies(monkeypatch, spec, EmptyPolicy, MinimalEngine)
    result = paired_protocol.run_paired_evaluation(
        spec_path,
        "candidate",
        "baseline",
        records_path,
        device="cpu",
    )

    assert result["games"] == 2
    assert records_path.exists()


@pytest.mark.parametrize("draw_reason", ("repetition", "50-move rule"))
def test_run_stops_on_engine_level_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draw_reason: str,
) -> None:
    spec = _spec(runtime=_bound_runtime())
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
    spec = _spec(runtime=_bound_runtime())
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
    spec = _spec(runtime=_bound_runtime())
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
