from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import PositionalSafetyFilter
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.training import classical_a_pos_corpus as corpus_module
from learned_ai.training.classical_a_pos_corpus import (
    CorpusContractError,
    CorpusLayout,
    CorpusRecordSealer,
    _build_frozen_corpus_manifest,
    _load_frozen_corpus,
    load_frozen_corpus,
    seal_singleton_ledger,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


MINI_LAYOUT = CorpusLayout(
    counts={
        "train": {
            "placement": {"W": 1, "B": 1},
            "movement": {"W": 0, "B": 0},
            "flying": {"W": 0, "B": 0},
        },
        "dev": {
            "placement": {"W": 0, "B": 0},
            "movement": {"W": 0, "B": 0},
            "flying": {"W": 0, "B": 0},
        },
    }
)


def _sha(char: str) -> str:
    return char * 64


def _source() -> dict:
    return {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "implementation_sha256": {
            "corpus_builder": _sha("1"),
            "scaffolded_encoder": _sha("2"),
            "scaffolded_net": _sha("3"),
            "game_ai": _sha("4"),
            "heuristics": _sha("5"),
            "native_extension": _sha("6"),
            "positional_safety_gate": _sha("7"),
        },
    }


def _resources() -> dict[str, str]:
    return {
        "evolved_weights": _sha("1"),
        "fullgame_db": _sha("2"),
        "endgame_db": _sha("3"),
        "phase_value_place": _sha("4"),
        "phase_value_move": _sha("5"),
        "phase_value_fly": _sha("6"),
        "gap_net": _sha("7"),
        "malom_manifest": _sha("8"),
        "malom_content": _sha("9"),
    }


def _oracle_identity(
    *,
    label_version: str = "sector-corrected-v1",
    manifest_sha256: str | None = None,
    content_sha256: str | None = None,
) -> dict[str, str]:
    return {
        "label_version": label_version,
        "manifest_sha256": manifest_sha256 or _sha("8"),
        "content_sha256": content_sha256 or _sha("9"),
    }


def _record(*, game_id: str, game_index: int, history: list[dict]) -> dict:
    board = BoardState.new_game()
    for move in history:
        assert move in get_all_legal_moves(board)
        board = board.apply_move(move)
    legal = [dict(move) for move in get_all_legal_moves(board)]
    teacher = legal[0]
    return {
        "example_id": canonical_sha256(
            {
                "history_sha256": canonical_sha256(history),
                "board_fen": board.to_fen_string(),
            }
        ),
        "game_id": game_id,
        "game_index": game_index,
        "split": "train",
        "stratum": "placement",
        "candidate_color": board.turn,
        "logical_ply": len(history),
        "history_moves": history,
        "history_sha256": canonical_sha256(history),
        "sanmill_history_sha256": _sha("a" if not history else "b"),
        "board_fen": board.to_fen_string(),
        "teacher_action": teacher,
        "teacher_evidence": {
            "difficulty": 9,
            "depth": 14,
            "threads": 1,
            "node_cap": 13_887_000,
            "main_search_nodes": 0,
            "restricted_rerank_nodes": 0,
            "positive_search": False,
            "teacher_instance_id": canonical_sha256(
                {"game_id": game_id, "logical_ply": len(history)}
            ),
            "gate_status": "applied",
            "gate_source": "classical-coordinator",
            "gate_selection_rule": "original-already-in-A_pos",
            "gate_selection_error": None,
            "original_action": teacher,
            "selected_action": teacher,
        },
    }


def _inventory_mask(_board: BoardState, legal: list[dict] | tuple[dict, ...]):
    return tuple(index < 2 for index in range(len(legal)))


def _malom_value(outcome: str) -> SimpleNamespace:
    return SimpleNamespace(
        outcome=outcome,
        sector="test-sector",
        sector_value=0,
        perspective="side-to-move",
    )


class _TwoSafeMalomOracle:
    def __init__(self) -> None:
        first = get_all_legal_moves(BoardState.new_game())[0]
        parents = [
            BoardState.new_game(),
            BoardState.new_game().apply_move(first),
        ]
        self._outcomes: dict[str, str] = {}
        for parent in parents:
            for index, move in enumerate(get_all_legal_moves(parent)):
                child = parent.apply_move(move)
                self._outcomes[child.to_fen_string()] = "W" if index < 2 else "L"
        for parent in parents:
            self._outcomes[parent.to_fen_string()] = "W"

    def query_value(self, board: BoardState) -> SimpleNamespace:
        return _malom_value(self._outcomes[board.to_fen_string()])

    @staticmethod
    def move_value(
        _parent: SimpleNamespace,
        child: SimpleNamespace,
    ) -> SimpleNamespace:
        return child

    @staticmethod
    def terminal_move_value(
        _parent: SimpleNamespace,
        _rules_outcome: str,
    ) -> SimpleNamespace:
        return _malom_value("W")


def _real_safety_filter(
    *,
    manifest_sha256: str = "8" * 64,
    content_sha256: str = "9" * 64,
) -> PositionalSafetyFilter:
    return PositionalSafetyFilter(
        _TwoSafeMalomOracle(),
        label_version=CURRENT_MALOM_LABEL_VERSION,
        manifest_sha256=manifest_sha256,
        content_sha256=content_sha256,
    )


def _seal_records(payloads: list[dict]) -> tuple[list[dict], list[dict]]:
    sealer = CorpusRecordSealer(
        source=_source(),
        resources=_resources(),
        inventory_verifier=_inventory_mask,
    )
    states: list[dict] = []
    labels: list[dict] = []
    for payload in payloads:
        unsealed = dict(payload)
        teacher_action = unsealed.pop("teacher_action")
        teacher_evidence = unsealed.pop("teacher_evidence")
        state = sealer.freeze_state(unsealed)
        label = sealer.seal_teacher_label(
            state,
            teacher_action=teacher_action,
            teacher_evidence=teacher_evidence,
        )
        states.append(state)
        labels.append(label)
    return states, labels


def _write_mini_corpus(
    tmp_path: Path,
    states: list[dict],
    records: list[dict],
) -> Path:
    state_split = tmp_path / "frozen-state-split.jsonl"
    state_split.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in states)
    )
    examples = tmp_path / "examples.jsonl"
    examples.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )
    ledger = tmp_path / "singleton-encounters.json"
    ledger.write_bytes(canonical_json_bytes(seal_singleton_ledger([])) + b"\n")
    manifest = _build_frozen_corpus_manifest(
        corpus_id="mini-corpus",
        source=_source(),
        resources_before=_resources(),
        resources_after=_resources(),
        state_split_path=state_split,
        examples_path=examples,
        singleton_ledger_path=ledger,
        collection_games=16,
        teacher_active_seconds=0.0,
        initial_policy_state_sha256=_sha("c"),
        sanmill_runtime_identity=_sha("d"),
        layout=MINI_LAYOUT,
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _valid_artifacts() -> tuple[list[dict], list[dict]]:
    first = get_all_legal_moves(BoardState.new_game())[0]
    return _seal_records(
        [
            _record(game_id="game-w", game_index=2, history=[]),
            _record(game_id="game-b", game_index=3, history=[first]),
        ]
    )


def _resign_state_and_teacher_artifacts(
    states: list[dict],
    records: list[dict],
) -> None:
    previous = "0" * 64
    shared_fields = {
        "example_id",
        "game_id",
        "game_index",
        "split",
        "stratum",
        "candidate_color",
        "logical_ply",
        "history_moves",
        "history_sha256",
        "sanmill_history_sha256",
        "board_fen",
        "legal_actions",
        "a_pos_mask",
        "a_pos_verification",
    }
    for state, record in zip(states, records, strict=True):
        proof = state["a_pos_verification"]
        proof["inventory_sha256"] = canonical_sha256(
            {
                "board_fen": state["board_fen"],
                "legal_actions": state["legal_actions"],
                "a_pos_mask": state["a_pos_mask"],
            }
        )
        proof["previous_verification_sha256"] = previous
        proof["verification_sha256"] = canonical_sha256(
            {key: value for key, value in proof.items() if key != "verification_sha256"}
        )
        previous = proof["verification_sha256"]
        state["state_record_identity"] = canonical_sha256(
            {
                key: value
                for key, value in state.items()
                if key != "state_record_identity"
            }
        )
        for field in shared_fields:
            record[field] = state[field]
        record["state_record_identity"] = state["state_record_identity"]
        record["record_sha256"] = canonical_sha256(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )


def test_strict_loader_replays_full_history_and_binds_atomic_order(
    tmp_path: Path,
) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)
    query_count = 0

    def verifier(board: BoardState, legal: list[dict] | tuple[dict, ...]):
        nonlocal query_count
        query_count += 1
        return _inventory_mask(board, legal)

    corpus = _load_frozen_corpus(
        manifest,
        layout=MINI_LAYOUT,
        inventory_verifier=verifier,
        oracle_identity=_oracle_identity(),
        sanmill_runtime_identity=_sha("d"),
    )

    assert len(corpus.examples) == 2
    assert corpus.examples[0].board.turn == "W"
    assert corpus.examples[1].board.turn == "B"
    assert corpus.examples[0].a_pos_mask.count(True) == 2
    assert corpus.manifest["heldout"]["consumed"] is False
    assert query_count == 2
    assert corpus.manifest["a_pos_verifier"]["loader_requeries_malom"] is True
    assert corpus.manifest["state_split_artifact"]["teacher_fields_present"] is False
    assert corpus.manifest["referee"]["sanmill_tree"] == (
        "17b9b0fd51ee8dac54c0454a6935978a47d19e0c"
    )
    assert corpus.manifest["referee"]["binary_size"] == 5_641_216
    assert corpus.manifest["referee"]["strict_referee"]["profile"] == (
        "mif-stable-moving-v1"
    )


def test_action_order_mismatch_fails_closed(tmp_path: Path) -> None:
    states, records = _valid_artifacts()
    states[0]["legal_actions"][0], states[0]["legal_actions"][1] = (
        states[0]["legal_actions"][1],
        states[0]["legal_actions"][0],
    )
    records[0]["teacher_index"] = 1
    _resign_state_and_teacher_artifacts(states, records)
    manifest = _write_mini_corpus(tmp_path, states, records)

    with pytest.raises(CorpusContractError, match="legal action order"):
        _load_frozen_corpus(
            manifest,
            layout=MINI_LAYOUT,
            inventory_verifier=_inventory_mask,
            oracle_identity=_oracle_identity(),
            sanmill_runtime_identity=_sha("d"),
        )


def test_a_pos_mask_tampering_breaks_build_time_state_integrity(
    tmp_path: Path,
) -> None:
    states, records = _valid_artifacts()
    states[0]["a_pos_mask"][1] = False
    states[0]["a_pos_mask"][2] = True
    with pytest.raises(CorpusContractError, match="frozen state record identity"):
        _write_mini_corpus(tmp_path, states, records)


def test_independent_oracle_rejects_wrong_mask_after_full_resigning(
    tmp_path: Path,
) -> None:
    states, records = _valid_artifacts()
    states[0]["a_pos_mask"][1] = False
    states[0]["a_pos_mask"][2] = True
    _resign_state_and_teacher_artifacts(states, records)
    manifest = _write_mini_corpus(tmp_path, states, records)

    with pytest.raises(CorpusContractError, match="independent A_pos inventory"):
        _load_frozen_corpus(
            manifest,
            layout=MINI_LAYOUT,
            inventory_verifier=_inventory_mask,
            oracle_identity=_oracle_identity(),
            sanmill_runtime_identity=_sha("d"),
        )


def test_state_split_artifact_rejects_teacher_fields(tmp_path: Path) -> None:
    states, records = _valid_artifacts()
    states[0]["teacher_action"] = records[0]["teacher_action"]

    with pytest.raises(CorpusContractError, match="contain teacher data"):
        _write_mini_corpus(tmp_path, states, records)


def test_referee_runtime_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)

    with pytest.raises(CorpusContractError, match="referee contract"):
        _load_frozen_corpus(
            manifest,
            layout=MINI_LAYOUT,
            inventory_verifier=_inventory_mask,
            oracle_identity=_oracle_identity(),
            sanmill_runtime_identity=_sha("e"),
        )


def test_resource_drift_before_and_after_labeling_fails_closed(
    tmp_path: Path,
) -> None:
    states, records = _valid_artifacts()
    state_split = tmp_path / "states.jsonl"
    state_split.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in states)
    )
    examples = tmp_path / "examples.jsonl"
    examples.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )
    ledger = tmp_path / "singletons.json"
    ledger.write_bytes(canonical_json_bytes(seal_singleton_ledger([])) + b"\n")
    after = _resources()
    after["malom_content"] = _sha("0")

    with pytest.raises(CorpusContractError, match="resources changed"):
        _build_frozen_corpus_manifest(
            corpus_id="drifted",
            source=_source(),
            resources_before=_resources(),
            resources_after=after,
            state_split_path=state_split,
            examples_path=examples,
            singleton_ledger_path=ledger,
            collection_games=16,
            teacher_active_seconds=0.0,
            initial_policy_state_sha256=_sha("c"),
            sanmill_runtime_identity=_sha("d"),
            layout=MINI_LAYOUT,
        )


def test_public_loader_never_accepts_an_incomplete_prefix(tmp_path: Path) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)

    with pytest.raises(CorpusContractError, match="real PositionalSafetyFilter"):
        load_frozen_corpus(manifest)


@pytest.mark.parametrize(
    ("manifest_identity", "content_identity", "expected_error"),
    [
        (_sha("f"), _sha("9"), "Malom manifest identity"),
        (_sha("8"), _sha("e"), "Malom content identity"),
    ],
)
def test_public_loader_rejects_real_filter_with_wrong_malom_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_identity: str,
    content_identity: str,
    expected_error: str,
) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)
    monkeypatch.setattr(corpus_module, "FROZEN_CORPUS_LAYOUT", MINI_LAYOUT)
    safety_filter = _real_safety_filter(
        manifest_sha256=manifest_identity,
        content_sha256=content_identity,
    )

    with pytest.raises(CorpusContractError, match=expected_error):
        load_frozen_corpus(
            manifest,
            safety_filter=safety_filter,
            sanmill_runtime_identity=_sha("d"),
        )


def test_public_loader_accepts_only_matching_real_filter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)
    monkeypatch.setattr(corpus_module, "FROZEN_CORPUS_LAYOUT", MINI_LAYOUT)

    corpus = load_frozen_corpus(
        manifest,
        safety_filter=_real_safety_filter(),
        sanmill_runtime_identity=_sha("d"),
    )

    assert len(corpus.examples) == 2


def test_internal_loader_rejects_non_current_malom_label_version(
    tmp_path: Path,
) -> None:
    states, records = _valid_artifacts()
    manifest = _write_mini_corpus(tmp_path, states, records)

    with pytest.raises(CorpusContractError, match="sector-corrected-v1"):
        _load_frozen_corpus(
            manifest,
            layout=MINI_LAYOUT,
            inventory_verifier=_inventory_mask,
            oracle_identity=_oracle_identity(label_version="legacy-unversioned"),
            sanmill_runtime_identity=_sha("d"),
        )
