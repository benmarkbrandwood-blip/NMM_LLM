"""Source-only exposure audit for a frozen held-out evaluation corpus."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Protocol, Sequence

from game.board import BoardState
from learned_ai.training.run_contract import canonical_sha256


HELDOUT_EXPOSURE_SCHEMA = "nmm.heldout-corpus-exposure-audit.v1"


class HeldoutExposureError(ValueError):
    """Raised when a corpus or exposure record violates the frozen contract."""


class _HumanPosition(Protocol):
    total_games: int


class _HumanDatabase(Protocol):
    def query_position(self, board: BoardState) -> _HumanPosition | None: ...


class _SpecialistEvidence(Protocol):
    empirical_counts: tuple[int, int, int]
    theoretical_wdl: Any


class _SpecialistDatabase(Protocol):
    def query_wdl_evidence(
        self,
        board: BoardState,
        min_samples: int = 0,
    ) -> _SpecialistEvidence | None: ...


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeldoutExposureError(f"{context} must be an object")
    return value


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HeldoutExposureError("corpus records must be an array")
    result = []
    for index, item in enumerate(value):
        result.append(_mapping(item, context=f"corpus record {index}"))
    return result


def validate_executable_corpus(
    payload: Mapping[str, Any],
    *,
    expected_corpus_identity: str,
    expected_records_identity: str,
) -> list[Mapping[str, Any]]:
    """Validate the frozen artifact identities without regenerating its sources."""
    corpus = _mapping(payload.get("corpus"), context="corpus")
    records = _records(corpus.get("records"))
    if corpus.get("records_identity") != expected_records_identity:
        raise HeldoutExposureError("executable record identity differs from the pin")
    if canonical_sha256(records) != expected_records_identity:
        raise HeldoutExposureError("executable records have drifted")

    identity_body = {
        "input_identities": payload.get("input_identities"),
        "input_identities_identity": payload.get("input_identities_identity"),
        "corpus": corpus,
    }
    if payload.get("executable_corpus_identity") != expected_corpus_identity:
        raise HeldoutExposureError("executable corpus identity differs from the pin")
    if canonical_sha256(identity_body) != expected_corpus_identity:
        raise HeldoutExposureError("executable corpus identity has drifted")

    ids = [str(record.get("source_core_id", "")) for record in records]
    if len(records) != 64 or len(set(ids)) != 64 or any(not item for item in ids):
        raise HeldoutExposureError("executable corpus must contain 64 unique IDs")
    return records


def classify_exposure(
    records: Sequence[Mapping[str, Any]],
    *,
    human_db: _HumanDatabase,
    specialist_db: _SpecialistDatabase,
) -> list[dict[str, Any]]:
    """Classify D4-canonical database exposure without loading a candidate."""
    result: list[dict[str, Any]] = []
    for raw in records:
        execution = _mapping(raw.get("execution_record"), context="execution record")
        final = _mapping(execution.get("final"), context="execution final state")
        fen = final.get("nmm_fen")
        if not isinstance(fen, str):
            raise HeldoutExposureError("execution final state has no NMM FEN")
        board = BoardState.from_fen_string(fen)
        if board.to_fen_string() != fen:
            raise HeldoutExposureError("execution final NMM FEN is not canonical")

        human = human_db.query_position(board)
        specialist = specialist_db.query_wdl_evidence(board, min_samples=0)
        specialist_counts = (
            tuple(int(value) for value in specialist.empirical_counts)
            if specialist is not None
            else (0, 0, 0)
        )
        human_games = int(human.total_games) if human is not None else 0
        human_exposed = human is not None
        specialist_exposed = specialist is not None
        result.append(
            {
                "source_core_id": str(raw.get("source_core_id")),
                "stratum": str(raw.get("stratum")),
                "human_db_d4_exposed": human_exposed,
                "human_db_games": human_games,
                "specialist_db_d4_exposed": specialist_exposed,
                "specialist_db_empirical_samples": sum(specialist_counts),
                "specialist_db_has_theoretical_label": bool(
                    specialist is not None
                    and specialist.theoretical_wdl is not None
                ),
                "strict_independence_subset": not (
                    human_exposed or specialist_exposed
                ),
            }
        )
    return result


def build_exposure_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    human_db: _HumanDatabase,
    specialist_db: _SpecialistDatabase,
    corpus_identity: str,
    records_identity: str,
    human_db_identity: str,
    specialist_db_identity: str,
) -> dict[str, Any]:
    """Build a portable audit and its strict-independence sensitivity subset."""
    exposure = classify_exposure(
        records,
        human_db=human_db,
        specialist_db=specialist_db,
    )
    strict_ids = [
        item["source_core_id"]
        for item in exposure
        if item["strict_independence_subset"]
    ]
    strata = sorted({item["stratum"] for item in exposure})
    summary = {
        "record_count": len(exposure),
        "human_db_d4_exposed_count": sum(
            item["human_db_d4_exposed"] for item in exposure
        ),
        "specialist_db_d4_exposed_count": sum(
            item["specialist_db_d4_exposed"] for item in exposure
        ),
        "strict_independence_count": len(strict_ids),
        "strict_independence_stratum_counts": dict(
            sorted(
                Counter(
                    item["stratum"]
                    for item in exposure
                    if item["strict_independence_subset"]
                ).items()
            )
        ),
        "per_stratum": {
            stratum: {
                "records": sum(item["stratum"] == stratum for item in exposure),
                "human_db_d4_exposed": sum(
                    item["stratum"] == stratum
                    and item["human_db_d4_exposed"]
                    for item in exposure
                ),
                "specialist_db_d4_exposed": sum(
                    item["stratum"] == stratum
                    and item["specialist_db_d4_exposed"]
                    for item in exposure
                ),
            }
            for stratum in strata
        },
    }
    body = {
        "schema_version": HELDOUT_EXPOSURE_SCHEMA,
        "status": "source_only_no_candidate_loaded_no_games_played",
        "candidate_loaded": False,
        "games_played": 0,
        "source_identities": {
            "executable_corpus": corpus_identity,
            "executable_records": records_identity,
            "human_db": human_db_identity,
            "specialist_db": specialist_db_identity,
        },
        "records": exposure,
        "records_identity": canonical_sha256(exposure),
        "strict_independence_source_core_ids": strict_ids,
        "strict_independence_identity": canonical_sha256(strict_ids),
        "summary": summary,
    }
    return {**body, "audit_identity": canonical_sha256(body)}
