from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import pytest

from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryOutcome,
    DataQueryResponse,
    DataQueryState,
)
from learned_ai.evaluation.sanmill_prefix import (
    PrefixSourceSpec,
    SanmillPrefixError,
    assign_source_kind,
    generate_paired_prefix,
)
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_SHA256,
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    SanmillInstallation,
    inspect_sanmill_installation,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"
_TOKENS = (
    ("a7", "d7"),
    ("g7", "d6"),
    ("g1", "f6"),
    ("a1", "f4"),
    ("b6", "d2"),
    ("f2", "b4"),
    ("a4", "c5"),
    ("g4", "e5"),
)


def _fake_installation() -> SanmillInstallation:
    return SanmillInstallation(
        checkout=Path("X:/Sanmill"),
        commit=PINNED_SANMILL_COMMIT,
        checkout_head=PINNED_SANMILL_COMMIT,
        tree=PINNED_SANMILL_TREE,
        binary=Path("X:/Sanmill/target/release/tgf.exe"),
        binary_sha256="a" * 64,
        binary_size=1,
        license_sha256="b" * 64,
    )


def _history_digest(actions: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(actions).encode("ascii")).hexdigest()


def _logical_count(actions: Sequence[str]) -> int:
    return sum(not action.startswith("x") for action in actions)


def _state(actions: Sequence[str], *, drift: bool = False) -> DataQueryState:
    count = _logical_count(actions)
    digest = "f" * 64 if drift else _history_digest(actions)
    return DataQueryState(
        current_fen=f"fake-fen-{digest}",
        side_to_move="white" if count % 2 == 0 else "black",
        phase="placing",
        pending_removal=False,
        pending_removals=(0, 0),
        no_capture_plies=0,
        action_token_count=len(actions),
        logical_ply_count=count,
        logical_plies_by_side=((count + 1) // 2, count // 2),
        snapshot_history_len=len(actions),
        repetition_history_len=len(actions),
        history_sha256=digest,
        outcome=DataQueryOutcome(kind="ongoing", winner=None, reason="ongoing"),
    )


def _book_source() -> dict[str, Any]:
    return {
        "identity": {
            "kind": "opening_book",
            "schema_version": 1,
            "variant": "nmm",
            "symmetry": "ring16",
            "sha256": EXPECTED_OPENING_BOOK_SHA256,
            "byte_length": 107245,
            "oracle_positions": 109,
            "oracle_records": 437,
            "source": "bundled",
        }
    }


def _candidate(
    *,
    logical_ply: int,
    stable_index: int,
    token: str,
    compound: bool = False,
) -> DataQueryCandidate:
    actions = (token, "xa7") if compound else (token,)
    digest = hashlib.sha256(
        f"{logical_ply}:{stable_index}:{actions}".encode("ascii")
    ).hexdigest()
    return DataQueryCandidate(
        logical_move_id=f"book:{digest}",
        source_group_id=f"book-rank-{stable_index + 1}",
        stable_index=stable_index,
        source_rank=stable_index + 1,
        raw_notation=token,
        mapped_notation=token,
        full_turn_actions=actions,
        remaining_actions=actions,
        contains_removal=compound,
        removal_action="xa7" if compound else None,
        logical_ply_delta=1,
        turn_prefix_complete=True,
        perfect=None,
        human=None,
    )


class _FakeSession:
    def __init__(
        self,
        *,
        miss_at: int | None = None,
        drift_summary_at: int | None = None,
    ) -> None:
        self.miss_at = miss_at
        self.drift_summary_at = drift_summary_at
        self.calls: list[str] = []

    def query_book(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
    ) -> DataQueryResponse:
        del request_id
        self.calls.append("book")
        state = _state(actions)
        del expected_current_fen
        logical_ply = state.logical_ply_count
        if logical_ply == self.miss_at:
            return DataQueryResponse(
                protocol_version=1,
                request_id=None,
                operation="query_book",
                status="book_miss",
                state=state,
                source=_book_source(),
                candidates=(),
                result=None,
                raw_line="{}",
            )
        first, second = _TOKENS[logical_ply]
        candidates = (
            _candidate(
                logical_ply=logical_ply,
                stable_index=0,
                token=first,
                compound=logical_ply == 2,
            ),
            _candidate(
                logical_ply=logical_ply,
                stable_index=1,
                token=second,
                compound=logical_ply == 2,
            ),
        )
        return DataQueryResponse(
            protocol_version=1,
            request_id=None,
            operation="query_book",
            status="available",
            state=state,
            source=_book_source(),
            candidates=candidates,
            result=None,
            raw_line="{}",
        )

    def history_summary(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
        count_mode: str = "logical",
    ) -> DataQueryResponse:
        del request_id, expected_current_fen
        self.calls.append("summary")
        logical_ply = _logical_count(actions)
        state = _state(
            actions,
            drift=logical_ply == self.drift_summary_at,
        )
        return DataQueryResponse(
            protocol_version=1,
            request_id=None,
            operation="history_summary",
            status="available",
            state=state,
            source=None,
            candidates=None,
            result={
                "count_mode": count_mode,
                "selected_count": logical_ply,
            },
            raw_line="{}",
        )


def test_source_assignment_requires_explicit_order_independent_weights() -> None:
    forward = assign_source_kind(
        experiment_id="eval-v1",
        pair_id="pair-17",
        seed=42,
        weights={"book": 75, "perfect_db": 25},
    )
    reverse = assign_source_kind(
        experiment_id="eval-v1",
        pair_id="pair-17",
        seed=42,
        weights={"perfect_db": 25, "book": 75},
    )

    assert forward == reverse
    assert forward in {"book", "perfect_db"}
    with pytest.raises(SanmillPrefixError, match="explicit weights"):
        assign_source_kind(
            experiment_id="eval-v1",
            pair_id="pair-17",
            seed=42,
            weights={},
        )


def test_prefix_source_spec_has_no_implicit_human_or_database_source() -> None:
    book = PrefixSourceSpec(
        kind="book",
        candidate_policy="source_declared",
    )

    assert book.portable_record() == {
        "kind": "book",
        "candidate_policy": "source_declared",
    }
    with pytest.raises(SanmillPrefixError, match="pinned portable identity"):
        PrefixSourceSpec(
            kind="human_db",
            candidate_policy="human_frequency",
            database_path=Path("X:/human.sqlite"),
            path_lookup_key="human_db_path",
        )


def test_generate_prefix_is_deterministic_and_shared_by_both_games() -> None:
    spec = PrefixSourceSpec(
        kind="book",
        candidate_policy="source_declared",
    )
    first = generate_paired_prefix(
        _FakeSession(),  # type: ignore[arg-type]
        _fake_installation(),
        experiment_id="eval-v1",
        pair_id="pair-3",
        seed=42,
        source_spec=spec,
    )
    second = generate_paired_prefix(
        _FakeSession(),  # type: ignore[arg-type]
        _fake_installation(),
        experiment_id="eval-v1",
        pair_id="pair-3",
        seed=42,
        source_spec=spec,
    )

    assert first.to_dict() == second.to_dict()
    assert first.prefix_identity == second.prefix_identity
    assert len(first.steps) == 8
    assert first.steps[-1].output_logical_plies_by_side == (4, 4)
    assert first.to_dict()["shared_by_pair_games"] == [0, 1]
    assert len(first.action_tokens) == 9
    assert any(len(step.action_tokens) == 2 for step in first.steps)


def test_source_miss_fails_without_querying_another_provider() -> None:
    session = _FakeSession(miss_at=3)

    with pytest.raises(SanmillPrefixError, match="no fallback"):
        generate_paired_prefix(
            session,  # type: ignore[arg-type]
            _fake_installation(),
            experiment_id="eval-v1",
            pair_id="pair-miss",
            seed=42,
            source_spec=PrefixSourceSpec(
                kind="book",
                candidate_policy="uniform_candidate",
            ),
        )

    assert session.calls == [
        "book",
        "summary",
        "book",
        "summary",
        "book",
        "summary",
        "book",
    ]


def test_history_identity_drift_is_rejected_at_the_next_boundary() -> None:
    session = _FakeSession(drift_summary_at=2)

    with pytest.raises(SanmillPrefixError, match="preceding history summary"):
        generate_paired_prefix(
            session,  # type: ignore[arg-type]
            _fake_installation(),
            experiment_id="eval-v1",
            pair_id="pair-drift",
            seed=42,
            source_spec=PrefixSourceSpec(
                kind="book",
                candidate_policy="uniform_candidate",
            ),
        )


def test_source_identity_pin_mismatch_is_rejected_before_selection() -> None:
    session = _FakeSession()

    with pytest.raises(SanmillPrefixError, match="differs from its pin"):
        generate_paired_prefix(
            session,  # type: ignore[arg-type]
            _fake_installation(),
            experiment_id="eval-v1",
            pair_id="pair-source-drift",
            seed=42,
            source_spec=PrefixSourceSpec(
                kind="book",
                candidate_policy="uniform_candidate",
                expected_identity_sha256="0" * 64,
            ),
        )

    assert session.calls == ["book"]


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_book_prefix_is_byte_stable_across_fresh_processes() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    records: list[dict[str, Any]] = []

    for _ in range(2):
        from learned_ai.evaluation.sanmill_data_query import (
            SanmillDataQuerySession,
        )

        with SanmillDataQuerySession(installation) as session:
            prefix = generate_paired_prefix(
                session,
                installation,
                experiment_id="book-prefix-black-box-v1",
                pair_id="pair-12",
                seed=42,
                source_spec=PrefixSourceSpec(
                    kind="book",
                    candidate_policy="source_declared",
                ),
            )
        records.append(prefix.to_dict())

    assert records[0] == records[1]
    assert records[0]["logical_ply_count"] == 8
    assert records[0]["logical_plies_by_side"] == [4, 4]
