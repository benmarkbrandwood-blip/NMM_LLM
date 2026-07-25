from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.sanmill_data_query import (
    SanmillDataQueryError,
    SanmillDataQueryProtocolError,
    SanmillDataQuerySession,
    parse_data_query_response,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_uci import inspect_sanmill_installation


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"
_BOOK_SHA256 = "cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5"


def _state(
    *,
    action_count: int = 0,
    logical_count: int = 0,
    side_counts: list[int] | None = None,
    history_sha256: str = "a" * 64,
) -> dict[str, object]:
    if side_counts is None:
        side_counts = [0, 0]
    return {
        "current_fen": (
            "********/********/******** w p p 0 9 0 9 0 0 "
            "-1 -1 -1 -1 0 0 1 ids:nodes"
        ),
        "side_to_move": "white" if logical_count % 2 == 0 else "black",
        "phase": "placing",
        "pending_removal": False,
        "pending_removals": [0, 0],
        "no_capture_plies": 0,
        "action_token_count": action_count,
        "logical_ply_count": logical_count,
        "logical_plies_by_side": side_counts,
        "snapshot_history_len": action_count,
        "repetition_history_len": action_count,
        "history_sha256": history_sha256,
        "outcome": {"kind": "ongoing", "reason": "ongoing"},
    }


def _candidate(kind: str = "book") -> dict[str, object]:
    payload: dict[str, object] = {
        "logical_move_id": f"{kind}:" + "b" * 64,
        "stable_index": 0,
        "mapped_notation": "d2",
        "full_turn_actions": ["d2"],
        "remaining_actions": ["d2"],
        "contains_removal": False,
        "logical_ply_delta": 1,
        "turn_prefix_complete": True,
    }
    if kind == "book":
        payload.update(
            {
                "source_group_id": "book-rank-1",
                "source_rank": 1,
                "raw_notation": "d2",
            }
        )
    elif kind == "perfect":
        payload["perfect"] = {
            "category": "draw",
            "wdl": 0,
            "steps": 1,
            "mode": "strict_steps",
        }
    elif kind == "human":
        payload.update(
            {
                "raw_notation": "d2",
                "human": {
                    "wins": 12,
                    "losses": 8,
                    "draws": 10,
                    "total": 30,
                    "frequency_numerator": 30,
                    "frequency_denominator": 40,
                    "relative_frequency": 0.75,
                    "empirical_win_rate": 0.4,
                    "empirical_draw_rate": 1 / 3,
                    "empirical_loss_rate": 8 / 30,
                    "legacy_experience_score": 0.466,
                    "moves_to_end_sum": 60.0,
                    "average_moves_to_end": 2.0,
                    "malom_wdl_after": "D",
                    "malom_dtw_after": 1,
                },
            }
        )
    return payload


def _book_source() -> dict[str, object]:
    return {
        "candidate_order": "source_array",
        "canonical_fen": (
            "********/********/******** w p p 0 9 0 9 0 0 "
            "-1 -1 -1 -1 0 0 1 ids:nodes"
        ),
        "identity": {
            "byte_length": 107245,
            "kind": "opening_book",
            "oracle_positions": 109,
            "oracle_records": 437,
            "schema_version": 1,
            "sha256": _BOOK_SHA256,
            "source": "bundled",
            "symmetry": "ring16",
            "variant": "nmm",
        },
        "selection_weight": {
            "formula": "ratio^(rank-1)",
            "kind": "geometric_rank",
            "ratio": 0.6,
        },
        "transform_to_canonical": 0,
    }


def _perfect_source() -> dict[str, object]:
    return {
        "identity": {
            "kind": "perfect_database",
            "database_format": "malom-sector",
            "sector_format_version": 2,
            "variant": "std",
            "root": "X:\\local\\perfect",
            "secval_sha256": "c" * 64,
            "fast_manifest_sha256": "d" * 64,
            "manifest_algorithm": "sha256(names,sizes,headers)-v1",
            "declared_sector_count": 498,
            "available_sector_count": 498,
            "placement_sector_count": 449,
            "settled_sector_count": 49,
            "flying_related_sector_count": 13,
            "fully_available": True,
        },
        "query_mode": "strict_steps",
        "candidate_order": "full_turn_uci_lexicographic",
        "fallback": "none",
        "coverage": {
            "placing": True,
            "moving": True,
            "flying": True,
            "pending_removal": "resolved_by_legal_continuation",
        },
    }


def _human_source() -> dict[str, object]:
    return {
        "identity": {
            "kind": "human_database",
            "database_format": "nmm-llm-human-db",
            "path": "X:\\local\\human_db.sqlite",
            "sha256": "e" * 64,
            "file_size": 4096,
            "schema_version": "2",
            "schema_sha256": "f" * 64,
            "build_date": "2026-07-25T00:00:00Z",
            "total_games": 40,
            "position_count": 1,
            "move_count": 2,
            "read_only": True,
            "immutable": True,
            "sidecars_absent": True,
            "malom_label_version": "sector-corrected-v1",
            "malom_trusted": True,
            "malom_trust_reason": "trusted version",
            "meta": [
                {"key": "schema_version", "value": "2"},
                {
                    "key": "malom_label_version",
                    "value": "sector-corrected-v1",
                },
            ],
        },
        "state_key": "........................|W|place|0|0|0|0",
        "symmetry_index": 0,
        "candidate_order": (
            "total_desc_then_canonical_notation_then_mapped_notation"
        ),
        "frequency_denominator_scope": "all_state_candidates",
        "total_matching_candidates": 2,
        "eligible_candidate_count": 2,
        "returned_candidate_count": 1,
        "candidate_limit": 1,
        "min_total": 0,
        "position": {
            "total_games": 40,
            "wins": 20,
            "losses": 10,
            "draws": 10,
            "malom_wdl": "D",
            "malom_dtw": 2,
            "canonical_winning_move": "d2",
        },
        "fallback": "none",
    }


def _response(
    operation: str,
    source: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "request_id": "request-1",
        "operation": operation,
        "status": "available",
        "state": _state(),
        "source": source,
        "candidates": [candidate],
    }


def _line(payload: dict[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def test_parse_book_response_and_portable_identity() -> None:
    response = parse_data_query_response(
        _line(_response("query_book", _book_source(), _candidate())),
        expected_operation="query_book",
        expected_request_id="request-1",
    )

    assert response.available
    assert response.state is not None
    assert response.state.logical_plies_by_side == (0, 0)
    assert response.candidates is not None
    assert response.candidates[0].full_turn_actions == ("d2",)
    portable = portable_source_identity(response)
    assert portable["kind"] == "book"
    assert portable["identity"]["sha256"] == _BOOK_SHA256
    assert len(portable["identity_sha256"]) == 64


def test_parse_perfect_and_human_source_specific_data() -> None:
    perfect = parse_data_query_response(
        _line(
            _response(
                "query_perfect_db",
                _perfect_source(),
                _candidate("perfect"),
            )
        )
    )
    human = parse_data_query_response(
        _line(
            _response(
                "query_human_db",
                _human_source(),
                _candidate("human"),
            )
        )
    )

    assert perfect.candidates is not None
    assert perfect.candidates[0].perfect is not None
    assert perfect.candidates[0].perfect.mode == "strict_steps"
    perfect_portable = portable_source_identity(
        perfect,
        path_lookup_key="malom_db_path",
    )
    assert "root" not in perfect_portable["identity"]
    assert perfect_portable["identity"]["path_lookup_key"] == "malom_db_path"

    assert human.candidates is not None
    assert human.candidates[0].human is not None
    assert human.candidates[0].human.frequency_numerator == 30
    human_portable = portable_source_identity(
        human,
        path_lookup_key="human_db_path",
    )
    assert "path" not in human_portable["identity"]
    assert human_portable["identity"]["path_lookup_key"] == "human_db_path"


@pytest.mark.parametrize(
    "location",
    ["response", "state", "candidate", "source"],
)
def test_parser_rejects_unknown_fields(location: str) -> None:
    payload = _response("query_book", _book_source(), _candidate())
    target: dict[str, object]
    if location == "response":
        target = payload
    elif location == "state":
        target = payload["state"]  # type: ignore[assignment]
    elif location == "candidate":
        target = payload["candidates"][0]  # type: ignore[index,assignment]
    else:
        target = payload["source"]  # type: ignore[assignment]
    target["unexpected"] = True

    with pytest.raises(SanmillDataQueryError, match="unknown"):
        parse_data_query_response(_line(payload))


def test_parser_preserves_versioned_protocol_error() -> None:
    payload = {
        "protocol_version": 1,
        "request_id": "bad-history",
        "operation": "query_book",
        "status": "error",
        "error": {
            "code": "history_illegal_action",
            "message": "action is illegal",
            "action_index": 3,
        },
    }

    with pytest.raises(SanmillDataQueryProtocolError) as raised:
        parse_data_query_response(
            _line(payload),
            expected_operation="query_book",
            expected_request_id="bad-history",
        )

    assert raised.value.code == "history_illegal_action"
    assert raised.value.action_index == 3


def test_parser_rejects_non_strict_perfect_candidate_or_source() -> None:
    payload = _response(
        "query_perfect_db",
        _perfect_source(),
        _candidate("perfect"),
    )
    payload["source"]["query_mode"] = "random"  # type: ignore[index]

    with pytest.raises(SanmillDataQueryError, match="strict"):
        parse_data_query_response(_line(payload))

    payload = _response(
        "query_perfect_db",
        _perfect_source(),
        _candidate("perfect"),
    )
    payload["candidates"][0]["perfect"]["mode"] = "wdl"  # type: ignore[index]

    with pytest.raises(SanmillDataQueryError, match="StrictSteps"):
        parse_data_query_response(_line(payload))


def test_parser_rejects_incomplete_or_duplicate_logical_turns() -> None:
    payload = _response("query_book", _book_source(), _candidate())
    payload["candidates"][0]["logical_ply_delta"] = 0  # type: ignore[index]

    with pytest.raises(SanmillDataQueryError, match="complete logical ply"):
        parse_data_query_response(_line(payload))

    first = _candidate()
    second = copy.deepcopy(first)
    second["stable_index"] = 1
    second["logical_move_id"] = "book:" + "c" * 64
    payload = _response("query_book", _book_source(), first)
    payload["candidates"].append(second)  # type: ignore[union-attr]

    with pytest.raises(SanmillDataQueryError, match="logical turns are duplicated"):
        parse_data_query_response(_line(payload))


def test_parser_rejects_removal_before_the_primary_action() -> None:
    payload = _response("query_book", _book_source(), _candidate())
    candidate = payload["candidates"][0]  # type: ignore[index]
    candidate.update(  # type: ignore[union-attr]
        {
            "full_turn_actions": ["xd6", "d2"],
            "remaining_actions": ["xd6", "d2"],
            "contains_removal": True,
            "removal_action": "xd6",
        }
    )

    with pytest.raises(SanmillDataQueryError, match="primary-plus-removal"):
        parse_data_query_response(_line(payload))


def test_history_summary_selected_count_must_match_state() -> None:
    payload = {
        "protocol_version": 1,
        "request_id": "history",
        "operation": "history_summary",
        "status": "available",
        "state": _state(
            action_count=2,
            logical_count=2,
            side_counts=[1, 1],
        ),
        "result": {"count_mode": "logical", "selected_count": 1},
    }

    with pytest.raises(SanmillDataQueryError, match="selected count"):
        parse_data_query_response(_line(payload))


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_book_query_is_byte_stable_and_stream_remains_aligned() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    runs: list[tuple[str, str]] = []

    for _ in range(2):
        with SanmillDataQuerySession(installation) as session:
            book = session.query_book(
                (),
                request_id="book-cross-process",
            )
            assert book.candidates is not None
            assert len(book.candidates) == 8
            summary = session.history_summary(
                book.candidates[0].full_turn_actions,
                request_id="history-cross-process",
            )
            assert summary.state is not None
            assert summary.state.logical_ply_count == 1
            assert summary.state.logical_plies_by_side == (1, 0)
            assert summary.state.action_token_count == 1
            assert len(session.transcript) == 4
            runs.append((book.raw_line, summary.raw_line))

    assert runs[0] == runs[1]
