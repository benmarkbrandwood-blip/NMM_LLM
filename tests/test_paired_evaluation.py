from __future__ import annotations

import json
from pathlib import Path

import pytest

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
