from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

import pytest

from learned_ai.evaluation.sanmill_book_paths import (
    BookPathCorpus,
    SanmillBookPathError,
    enumerate_complete_book_paths,
    freeze_book_path_corpus,
    load_book_path_corpus,
)
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryOutcome,
    DataQueryResponse,
    DataQueryState,
    SanmillDataQuerySession,
)
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_SHA256,
    EXPECTED_SANMILL_BINARY_SHA256,
    EXPECTED_SANMILL_BINARY_SIZE,
    EXPECTED_SANMILL_LICENSE_SHA256,
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    SanmillInstallation,
    inspect_sanmill_installation,
)
from learned_ai.training.run_contract import canonical_sha256


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"
_GENERATOR_COMMIT = "f" * 40
_PRIMARY_BY_DEPTH = ("d2", "d6", "f4", "b6", "g7", "g4", "f2", "a1")


def _fake_installation() -> SanmillInstallation:
    return SanmillInstallation(
        checkout=Path("X:/Sanmill"),
        commit=PINNED_SANMILL_COMMIT,
        checkout_head=PINNED_SANMILL_COMMIT,
        tree=PINNED_SANMILL_TREE,
        binary=Path("X:/Sanmill/target/release/tgf.exe"),
        binary_sha256=EXPECTED_SANMILL_BINARY_SHA256,
        binary_size=EXPECTED_SANMILL_BINARY_SIZE,
        license_sha256=EXPECTED_SANMILL_LICENSE_SHA256,
    )


def _history_digest(actions: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(actions).encode("ascii")).hexdigest()


def _logical_count(actions: Sequence[str]) -> int:
    return sum(not action.startswith("x") for action in actions)


def _state(
    actions: Sequence[str],
    *,
    history_drift: bool = False,
) -> DataQueryState:
    logical_count = _logical_count(actions)
    digest = "0" * 64 if history_drift else _history_digest(actions)
    return DataQueryState(
        current_fen=f"fake-fen-{digest}",
        side_to_move="white" if logical_count % 2 == 0 else "black",
        phase="placing",
        pending_removal=False,
        pending_removals=(0, 0),
        no_capture_plies=0,
        action_token_count=len(actions),
        logical_ply_count=logical_count,
        logical_plies_by_side=(
            (logical_count + 1) // 2,
            logical_count // 2,
        ),
        snapshot_history_len=len(actions),
        repetition_history_len=len(actions),
        history_sha256=digest,
        outcome=DataQueryOutcome(
            kind="ongoing",
            winner=None,
            reason="ongoing",
        ),
    )


def _book_identity(*, drift: bool = False) -> dict[str, Any]:
    identity = {
        "kind": "opening_book",
        "schema_version": 1,
        "variant": "nmm",
        "symmetry": "ring16",
        "sha256": EXPECTED_OPENING_BOOK_SHA256,
        "byte_length": 107245 + int(drift),
        "oracle_positions": 109,
        "oracle_records": 437,
        "source": "bundled",
    }
    return {"identity": identity}


def _candidate(
    *,
    depth: int,
    stable_index: int,
    token: str,
    compound: bool = False,
) -> DataQueryCandidate:
    actions = (token, "xa7") if compound else (token,)
    digest = hashlib.sha256(
        f"{depth}:{stable_index}:{actions}".encode("ascii")
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


class _FakeBookSession:
    def __init__(
        self,
        *,
        source_drift_at: int | None = None,
        summary_drift_at: int | None = None,
        duplicate_at: int | None = None,
    ) -> None:
        self.source_drift_at = source_drift_at
        self.summary_drift_at = summary_drift_at
        self.duplicate_at = duplicate_at
        self.calls: list[str] = []

    def query_book(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
    ) -> DataQueryResponse:
        del request_id
        self.calls.append("query_book")
        state = _state(actions)
        del expected_current_fen
        depth = state.logical_ply_count
        source = _book_identity(drift=depth == self.source_drift_at)
        if depth == 3 and "d6" in actions:
            return DataQueryResponse(
                protocol_version=1,
                request_id=None,
                operation="query_book",
                status="book_miss",
                state=state,
                source=source,
                candidates=(),
                result=None,
                raw_line="{}",
            )
        if depth == 1:
            candidates = (
                _candidate(
                    depth=depth,
                    stable_index=0,
                    token="d6",
                ),
                _candidate(
                    depth=depth,
                    stable_index=1,
                    token="b4",
                ),
            )
        else:
            token = _PRIMARY_BY_DEPTH[depth]
            candidates = (
                _candidate(
                    depth=depth,
                    stable_index=0,
                    token=token,
                    compound=depth == 2,
                ),
            )
        if depth == self.duplicate_at:
            candidates = (
                candidates[0],
                _candidate(
                    depth=depth,
                    stable_index=1,
                    token=candidates[0].full_turn_actions[0],
                    compound=candidates[0].contains_removal,
                ),
            )
        return DataQueryResponse(
            protocol_version=1,
            request_id=None,
            operation="query_book",
            status="available",
            state=state,
            source=source,
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
        self.calls.append("history_summary")
        logical_count = _logical_count(actions)
        state = _state(
            actions,
            history_drift=logical_count == self.summary_drift_at,
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
                "selected_count": logical_count,
            },
            raw_line="{}",
        )


def _fake_corpus(**session_options: Any) -> BookPathCorpus:
    return enumerate_complete_book_paths(
        _FakeBookSession(**session_options),  # type: ignore[arg-type]
        _fake_installation(),
        generator_commit=_GENERATOR_COMMIT,
    )


def test_enumeration_prunes_miss_and_preserves_compound_logical_turn() -> None:
    session = _FakeBookSession()
    corpus = enumerate_complete_book_paths(
        session,  # type: ignore[arg-type]
        _fake_installation(),
        generator_commit=_GENERATOR_COMMIT,
    )

    assert len(corpus.paths) == 1
    assert len(corpus.incomplete_leaves) == 1
    assert corpus.incomplete_leaves[0].kind == "book_miss"
    assert corpus.incomplete_leaves[0].logical_ply_count == 3
    assert len(corpus.paths[0].steps) == 8
    assert len(corpus.paths[0].action_tokens) == 9
    assert corpus.paths[0].steps[2].compound_turn
    assert corpus.paths[0].steps[2].action_tokens == ("f4", "xa7")
    assert corpus.depth_audit[2].compound_edge_count == 2
    assert session.calls.count("query_book") == 10
    assert session.calls.count("history_summary") == 10
    assert set(session.calls) == {"query_book", "history_summary"}


def test_enumeration_is_deterministic_for_same_inputs() -> None:
    first = _fake_corpus()
    second = _fake_corpus()

    assert first.to_dict() == second.to_dict()
    assert first.corpus_identity == second.corpus_identity


def test_enumeration_rejects_source_history_and_duplicate_drift() -> None:
    with pytest.raises(SanmillBookPathError, match="source identity changed"):
        _fake_corpus(source_drift_at=2)
    with pytest.raises(
        SanmillBookPathError,
        match="preceding history summary",
    ):
        _fake_corpus(summary_drift_at=2)
    with pytest.raises(SanmillBookPathError, match="duplicate child history"):
        _fake_corpus(duplicate_at=0)


def test_strict_loader_recomputes_identities_and_rejects_unknown_fields() -> None:
    payload = _fake_corpus().to_dict()

    assert BookPathCorpus.from_dict(copy.deepcopy(payload)).to_dict() == payload

    tampered = copy.deepcopy(payload)
    tampered["paths"][0]["steps"][0]["candidate"]["source_group_id"] = (
        "tampered-group"
    )
    with pytest.raises(SanmillBookPathError, match="identity mismatch"):
        BookPathCorpus.from_dict(tampered)

    unknown = copy.deepcopy(payload)
    unknown["source_identity"]["identity"]["unexpected"] = True
    unknown["source_identity"]["identity_sha256"] = canonical_sha256(
        unknown["source_identity"]["identity"]
    )
    unknown["corpus_identity"] = canonical_sha256(
        {key: value for key, value in unknown.items() if key != "corpus_identity"}
    )
    with pytest.raises(SanmillBookPathError, match="unknown unexpected"):
        BookPathCorpus.from_dict(unknown)


def test_freeze_is_exclusive_and_load_round_trips(tmp_path: Path) -> None:
    corpus = _fake_corpus()
    target = tmp_path / "book-path-corpus.json"

    freeze_book_path_corpus(target, corpus)

    assert load_book_path_corpus(target).to_dict() == corpus.to_dict()
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_book_path_corpus(target, corpus)


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_complete_book_corpus_is_stable_across_fresh_processes() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    records: list[dict[str, Any]] = []

    for _ in range(2):
        with SanmillDataQuerySession(installation) as session:
            corpus = enumerate_complete_book_paths(
                session,
                installation,
                generator_commit=_GENERATOR_COMMIT,
            )
        records.append(corpus.to_dict())

    assert records[0] == records[1]
    assert records[0]["summary"] == {
        "complete_path_count": 192,
        "incomplete_leaf_count": 508,
        "book_miss_leaf_count": 508,
        "terminal_leaf_count": 0,
        "compound_edge_count": 176,
    }
    assert [
        item["unique_child_history_count"]
        for item in records[0]["depth_audit"]
    ] == [8, 40, 76, 140, 264, 232, 128, 192]
