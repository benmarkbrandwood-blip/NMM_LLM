"""Freeze and characterize the official F0-H0 Design B2 split.

The split is derived from the sealed F0-D0 manifest before any F0-H0
screening statistic is computed.  The final-test membership is visible only
as session identifiers and is guarded from raw, decision, and feature access.
Malom is used only by the bounded cost benchmark in this module; no oracle
outcome, safe-set, support, concentration, or product-effect statistic is
published here.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from ai.malom_db import MalomDB
from game.rules import get_game_phase
from learned_ai.evaluation.human_f0h0_design_b_supplement import (
    PHASE_NAME,
    PHASES,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
    F0H0Error,
    ReplayedDecision,
    _oracle_inventory,
    _read_raw_game,
    replay_game,
    verify_malom_snapshot,
)
from learned_ai.evaluation.human_f0h0_split_retest import (
    EXPECTED_CORPUS_IDENTITY,
    EXPECTED_F0D0_FILE_SHA256,
    EXPECTED_F0D0_MANIFEST_IDENTITY,
    Boundary,
    GameRecord,
    SplitRetestError,
    _load_sealed_json,
    canonical_sha256,
    nearest_rank_quantiles,
    sha256_file,
    write_sealed_json,
)


PLAN_SCHEMA = "nmm.f0-h0-design-b2-freeze-plan.v1"
MEMBERSHIP_SCHEMA = "nmm.f0-h0-design-b2-frozen-membership.v1"
RESULT_SCHEMA = "nmm.f0-h0-design-b2-characterization-result.v1"
PARTITIONS = ("train", "selection", "confirmation", "final-test")
NONFINAL_PARTITIONS = ("train", "selection", "confirmation")
EXPECTED_COUNTS = {
    "train": 36_949,
    "selection": 887,
    "confirmation": 386,
    "final-test": 847,
}
MALOM_MANIFEST_FILE_SHA256 = (
    "f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747"
)
MALOM_CONTENT_SHA256 = (
    "c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544"
)


class B2FreezeError(RuntimeError):
    """Raised when the frozen B2 contract or an access boundary differs."""


class FinalTestAccessError(B2FreezeError):
    """Raised before any final-test raw record or derivative can be read."""


T = TypeVar("T")


def _translate_error(exc: Exception) -> B2FreezeError:
    return B2FreezeError(str(exc))


def _players(games: Sequence[GameRecord]) -> set[str]:
    return {player for game in games for player in game.players}


def _games_before(games: Sequence[GameRecord], cut: date) -> list[GameRecord]:
    return [game for game in games if game.played_on < cut]


def _strong_segment(
    games: Sequence[GameRecord],
    *,
    start: date,
    end: date | None,
) -> list[GameRecord]:
    prior_players = _players(_games_before(games, start))
    return [
        game
        for game in games
        if game.played_on >= start
        and (end is None or game.played_on < end)
        and game.white_player not in prior_players
        and game.black_player not in prior_players
    ]


def partition_b2_games(
    games: Sequence[GameRecord],
    *,
    train_cut: date,
    confirmation_cut: date,
    final_cut: date,
) -> dict[str, list[GameRecord]]:
    """Apply the owner-selected B2 definitions without looking at outcomes."""
    if not train_cut < confirmation_cut < final_cut:
        raise B2FreezeError("B2 cut order is invalid")
    return {
        "train": _games_before(games, train_cut),
        "selection": _strong_segment(
            games,
            start=train_cut,
            end=confirmation_cut,
        ),
        "confirmation": _strong_segment(
            games,
            start=confirmation_cut,
            end=final_cut,
        ),
        "final-test": _strong_segment(games, start=final_cut, end=None),
    }


def _hash_rank(namespace: str, seed: str, *values: object) -> str:
    payload = "\0".join((namespace, seed, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decision_sample(
    games: Sequence[GameRecord],
    *,
    size: int,
    seed: str,
) -> list[list[Any]]:
    references = [
        (game.session_id, logical_ply)
        for game in games
        for logical_ply in range(game.move_count)
    ]
    ranked = sorted(
        references,
        key=lambda row: (
            _hash_rank("f0-h0-b2-malom-cost-decision-v1", seed, *row),
            row,
        ),
    )
    if len(ranked) < size:
        raise B2FreezeError("Malom benchmark population is too small")
    return [[session, logical_ply] for session, logical_ply in ranked[:size]]


def _fallback_games(
    train: Sequence[GameRecord],
    selection: Sequence[GameRecord],
    *,
    total_games: int,
    seed: str,
) -> list[str]:
    if total_games < len(selection):
        raise B2FreezeError("fallback sample cannot retain all selection games")
    train_needed = total_games - len(selection)
    ranked_train = sorted(
        train,
        key=lambda game: (
            _hash_rank(
                "f0-h0-b2-fallback-train-game-v1",
                seed,
                game.session_id,
            ),
            game.session_id,
        ),
    )
    if len(ranked_train) < train_needed:
        raise B2FreezeError("fallback sample population is too small")
    return sorted(
        [game.session_id for game in selection]
        + [game.session_id for game in ranked_train[:train_needed]]
    )


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        plan, file_sha = _load_sealed_json(
            path,
            schema=PLAN_SCHEMA,
            identity_field="plan_identity",
        )
    except SplitRetestError as exc:
        raise _translate_error(exc) from exc
    expected_boundary = {
        "corpus_identity": EXPECTED_CORPUS_IDENTITY,
        "manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
        "behavior_games": 92_226,
        "behavior_logical_plies": 4_394_220,
        "behavior_player_keys": 4_994,
        "outcome_games": 37_866,
    }
    if plan.get("input_boundary") != expected_boundary:
        raise B2FreezeError("B2 plan F0-D0 boundary differs")
    split = plan.get("split")
    if not isinstance(split, Mapping) or split.get("cuts") != {
        "train_end_exclusive": "2026-03-01",
        "selection_end_exclusive": "2026-04-01",
        "confirmation_end_exclusive": "2026-05-01",
    }:
        raise B2FreezeError("B2 plan cuts differ")
    if split.get("expected_games") != EXPECTED_COUNTS:
        raise B2FreezeError("B2 plan expected counts differ")
    scope = plan.get("scope")
    forbidden = (
        "independent_support",
        "modifiable_state_reachability",
        "concentration",
        "product_effect_upper_bound",
    )
    if not isinstance(scope, Mapping) or any(
        scope.get(key) is not False for key in forbidden
    ):
        raise B2FreezeError("B2 plan permits a screening dimension")
    benchmark = plan.get("malom_cost_benchmark")
    if not isinstance(benchmark, Mapping):
        raise B2FreezeError("B2 Malom benchmark contract is absent")
    required_benchmark = {
        "sample_decisions": 256,
        "maximum_projected_active_seconds": 7_200.0,
        "projection_safety_multiplier": 1.25,
        "fallback_sample_games": 10_000,
        "fallback_selection_policy": "all_887_selection_plus_hash_ranked_train",
        "trusted_label_version": "sector-corrected-v1",
    }
    if any(benchmark.get(key) != value for key, value in required_benchmark.items()):
        raise B2FreezeError("B2 Malom benchmark thresholds differ")
    guard = plan.get("final_test_guard")
    if not isinstance(guard, Mapping) or guard.get("default") != "deny":
        raise B2FreezeError("B2 final-test guard is not fail closed")
    return plan, file_sha


def verify_implementation_artifacts(
    repository_root: str | Path,
    plan: Mapping[str, Any],
) -> None:
    root = Path(repository_root)
    artifacts = plan.get("implementation_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise B2FreezeError("B2 implementation artifact inventory is absent")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise B2FreezeError("B2 implementation artifact is invalid")
        path = root / str(artifact.get("path"))
        if sha256_file(path) != artifact.get("sha256"):
            raise B2FreezeError(f"B2 implementation artifact differs: {path}")


def build_membership(
    boundary: Boundary,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build membership from F0-D0 metadata only; no raw game is opened."""
    cuts = plan["split"]["cuts"]
    partitions = partition_b2_games(
        boundary.games,
        train_cut=date.fromisoformat(cuts["train_end_exclusive"]),
        confirmation_cut=date.fromisoformat(cuts["selection_end_exclusive"]),
        final_cut=date.fromisoformat(cuts["confirmation_end_exclusive"]),
    )
    measured = {name: len(partitions[name]) for name in PARTITIONS}
    if measured != plan["split"]["expected_games"]:
        raise B2FreezeError(
            f"B2 membership count differs: expected={EXPECTED_COUNTS}, measured={measured}"
        )
    session_sets = {
        name: {game.session_id for game in partitions[name]} for name in PARTITIONS
    }
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            if session_sets[left] & session_sets[right]:
                raise B2FreezeError(f"B2 sessions overlap: {left} and {right}")
    test_players = {name: _players(partitions[name]) for name in PARTITIONS[1:]}
    player_overlap: dict[str, int] = {}
    for left_index, left in enumerate(PARTITIONS[1:]):
        for right in PARTITIONS[left_index + 2 :]:
            key = f"{left}::{right}"
            player_overlap[key] = len(test_players[left] & test_players[right])
    if any(player_overlap.values()):
        raise B2FreezeError("B2 test-segment player sets overlap")

    membership_rows = {
        name: sorted(session_sets[name]) for name in PARTITIONS
    }
    included = set().union(*session_sets.values())
    omitted = sorted(
        game.session_id for game in boundary.games if game.session_id not in included
    )
    benchmark = plan["malom_cost_benchmark"]
    analysis_games = partitions["train"] + partitions["selection"]
    benchmark_refs = _decision_sample(
        analysis_games,
        size=int(benchmark["sample_decisions"]),
        seed=str(benchmark["sample_seed"]),
    )
    fallback_games = _fallback_games(
        partitions["train"],
        partitions["selection"],
        total_games=int(benchmark["fallback_sample_games"]),
        seed=str(benchmark["fallback_sample_seed"]),
    )
    return {
        "schema_version": MEMBERSHIP_SCHEMA,
        "freeze_id": plan["freeze_id"],
        "status": "official_b2_membership_frozen_before_screening",
        "plan_identity": plan["plan_identity"],
        "input_boundary": {
            "corpus_identity": boundary.corpus_identity,
            "manifest_identity": boundary.manifest_identity,
            "manifest_file_sha256": boundary.manifest_file_sha256,
            "behavior_raw_subset_identity": boundary.raw_subset_identity,
        },
        "definition": plan["split"],
        "partitions": {
            name: {
                "games": len(membership_rows[name]),
                "session_ids": membership_rows[name],
                "session_ids_identity": canonical_sha256(membership_rows[name]),
            }
            for name in PARTITIONS
        },
        "pairwise_session_intersections": {
            f"{left}::{right}": 0
            for left_index, left in enumerate(PARTITIONS)
            for right in PARTITIONS[left_index + 1 :]
        },
        "test_segment_player_isolation": {
            "player_keys": {
                name: len(test_players[name]) for name in PARTITIONS[1:]
            },
            "player_key_identities": {
                name: canonical_sha256(sorted(test_players[name]))
                for name in PARTITIONS[1:]
            },
            "pairwise_intersections": player_overlap,
            "verified_disjoint": True,
        },
        "unassigned_behavior_games": {
            "games": len(omitted),
            "session_ids_identity": canonical_sha256(omitted),
            "reason": "outside the four selected B2 membership definitions",
        },
        "malom_cost_preregistration": {
            "benchmark_decision_references": benchmark_refs,
            "benchmark_decision_references_identity": canonical_sha256(
                benchmark_refs
            ),
            "fallback_game_session_ids": fallback_games,
            "fallback_game_session_ids_identity": canonical_sha256(
                fallback_games
            ),
        },
        "access_state": {
            "membership_built_from_f0d0_manifest_only": True,
            "raw_game_files_opened": 0,
            "final_test_raw_game_files_opened": 0,
            "final_test_decisions_loaded": 0,
            "final_test_derived_features_loaded": 0,
        },
    }


def load_membership(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        value, file_sha = _load_sealed_json(
            path,
            schema=MEMBERSHIP_SCHEMA,
            identity_field="membership_identity",
        )
    except SplitRetestError as exc:
        raise _translate_error(exc) from exc
    if value.get("status") != "official_b2_membership_frozen_before_screening":
        raise B2FreezeError("B2 membership status differs")
    partitions = value.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITIONS):
        raise B2FreezeError("B2 membership partitions differ")
    observed: dict[str, set[str]] = {}
    for name in PARTITIONS:
        row = partitions[name]
        sessions = row.get("session_ids") if isinstance(row, Mapping) else None
        if (
            not isinstance(sessions, list)
            or sessions != sorted(sessions)
            or len(sessions) != len(set(sessions))
            or row.get("games") != EXPECTED_COUNTS[name]
            or canonical_sha256(sessions) != row.get("session_ids_identity")
        ):
            raise B2FreezeError(f"B2 {name} membership differs")
        observed[name] = set(sessions)
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            if observed[left] & observed[right]:
                raise B2FreezeError("B2 membership sets are not disjoint")
    isolation = value.get("test_segment_player_isolation")
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("verified_disjoint") is not True
    ):
        raise B2FreezeError("B2 player isolation is not sealed")
    prereg = value.get("malom_cost_preregistration")
    if not isinstance(prereg, Mapping):
        raise B2FreezeError("B2 cost sample membership is absent")
    refs = prereg.get("benchmark_decision_references")
    fallback = prereg.get("fallback_game_session_ids")
    if (
        not isinstance(refs, list)
        or len(refs) != 256
        or canonical_sha256(refs)
        != prereg.get("benchmark_decision_references_identity")
        or not isinstance(fallback, list)
        or len(fallback) != 10_000
        or canonical_sha256(fallback)
        != prereg.get("fallback_game_session_ids_identity")
    ):
        raise B2FreezeError("B2 cost sample identities differ")
    return value, file_sha


@dataclass(frozen=True)
class FrozenSplitAccess:
    """The only supported content accessor for the frozen B2 corpus."""

    final_sessions: frozenset[str]

    @classmethod
    def from_membership(cls, membership: Mapping[str, Any]) -> "FrozenSplitAccess":
        partitions = membership.get("partitions")
        if not isinstance(partitions, Mapping):
            raise B2FreezeError("frozen membership is absent")
        final = partitions.get("final-test")
        sessions = final.get("session_ids") if isinstance(final, Mapping) else None
        if not isinstance(sessions, list) or len(sessions) != EXPECTED_COUNTS["final-test"]:
            raise B2FreezeError("final-test membership is unavailable")
        return cls(final_sessions=frozenset(str(value) for value in sessions))

    def assert_allowed(self, session_id: str, *, access_kind: str) -> None:
        if session_id in self.final_sessions:
            raise FinalTestAccessError(
                "final-test is sealed; separate one-time authorization is "
                f"required before {access_kind}: {session_id}"
            )

    def read_raw_game(
        self,
        repository_root: Path,
        record: CorpusRecord,
        boundary: F0D0Boundary,
    ) -> Mapping[str, Any]:
        self.assert_allowed(record.session_id, access_kind="raw game read")
        try:
            return _read_raw_game(repository_root, record, boundary)
        except F0H0Error as exc:
            raise _translate_error(exc) from exc

    def load_decisions(
        self,
        repository_root: Path,
        record: CorpusRecord,
        boundary: F0D0Boundary,
    ) -> list[ReplayedDecision]:
        self.assert_allowed(record.session_id, access_kind="decision load")
        raw = self.read_raw_game(repository_root, record, boundary)
        try:
            return replay_game(raw, record)
        except F0H0Error as exc:
            raise _translate_error(exc) from exc

    def derive_features(
        self,
        session_id: str,
        producer: Callable[[], T],
    ) -> T:
        self.assert_allowed(session_id, access_kind="derived feature load")
        return producer()


def _outcome_map(f0d0_boundary: F0D0Boundary) -> dict[str, str | None]:
    manifest = f0d0_boundary.manifest
    encoding = manifest.get("record_encoding")
    rows = manifest.get("game_records")
    if not isinstance(encoding, Mapping) or not isinstance(rows, list):
        raise B2FreezeError("F0-D0 result encoding is absent")
    fields = encoding.get("fields")
    values = encoding.get("outcome_values")
    if not isinstance(fields, list) or not isinstance(values, list):
        raise B2FreezeError("F0-D0 outcome table is absent")
    positions = {field: index for index, field in enumerate(fields)}
    required = {
        "session_id",
        "independent_outcome_index",
        "outcome_analysis_eligible",
    }
    if not required.issubset(positions):
        raise B2FreezeError("F0-D0 independent outcome fields are absent")
    result: dict[str, str | None] = {}
    for row in rows:
        session = row[positions["session_id"]]
        if not isinstance(session, str):
            raise B2FreezeError("F0-D0 session identifier is invalid")
        if not row[positions["outcome_analysis_eligible"]]:
            result[session] = None
            continue
        index = row[positions["independent_outcome_index"]]
        if isinstance(index, bool) or not isinstance(index, int):
            raise B2FreezeError("F0-D0 outcome index is invalid")
        try:
            outcome = values[index]
        except IndexError as exc:
            raise B2FreezeError("F0-D0 outcome index is out of range") from exc
        if outcome not in {"W", "B", "D"}:
            raise B2FreezeError("F0-D0 eligible outcome is invalid")
        result[session] = outcome
    return result


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise B2FreezeError("cannot summarize an empty distribution")
    return {
        "quantile_method": "nearest-rank",
        "quantiles": nearest_rank_quantiles(values),
        "mean": sum(values) / len(values),
    }


def _player_game_distribution(games: Sequence[GameRecord]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for game in games:
        counts.update(game.players)
    values = list(counts.values())
    return {
        **_distribution(values),
        "tail": [
            {
                "at_least_games": threshold,
                "player_keys": sum(value >= threshold for value in values),
            }
            for threshold in (1, 2, 5, 10, 20, 50, 100, 250, 500)
        ],
    }


def _length_distribution(games: Sequence[GameRecord]) -> dict[str, Any]:
    values = [game.move_count for game in games]
    return {
        **_distribution(values),
        "bins": {
            "1-18": sum(1 <= value <= 18 for value in values),
            "19-40": sum(19 <= value <= 40 for value in values),
            "41-60": sum(41 <= value <= 60 for value in values),
            "61-100": sum(61 <= value <= 100 for value in values),
            "101+": sum(value >= 101 for value in values),
        },
    }


def characterize_metadata(
    games: Sequence[GameRecord],
    outcomes: Mapping[str, str | None],
) -> dict[str, Any]:
    white_players = {game.white_player for game in games}
    black_players = {game.black_player for game in games}
    outcome_counts = Counter(outcomes[game.session_id] for game in games)
    return {
        "games": len(games),
        "player_keys": len(_players(games)),
        "player_game_distribution": _player_game_distribution(games),
        "logical_plies": sum(game.move_count for game in games),
        "strict_outcome_eligible_games": len(games) - outcome_counts[None],
        "strict_outcome_distribution": {
            outcome: outcome_counts[outcome] for outcome in ("W", "B", "D")
        },
        "color_distribution": {
            "white_player_game_incidences": len(games),
            "black_player_game_incidences": len(games),
            "unique_white_player_keys": len(white_players),
            "unique_black_player_keys": len(black_players),
        },
        "game_length_logical_plies": _length_distribution(games),
        "date_range": {
            "minimum": min(game.played_on for game in games).isoformat(),
            "maximum": max(game.played_on for game in games).isoformat(),
        },
    }


def replay_characterization(
    *,
    repository_root: Path,
    games_by_partition: Mapping[str, Sequence[GameRecord]],
    raw_boundary: F0D0Boundary,
    access: FrozenSplitAccess,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    raw_records = {record.session_id: record for record in raw_boundary.records}
    phase_counts = {name: Counter() for name in NONFINAL_PARTITIONS}
    color_counts = {name: Counter() for name in NONFINAL_PARTITIONS}
    expected_games = sum(
        len(games_by_partition[name]) for name in NONFINAL_PARTITIONS
    )
    opened = 0
    raw_bytes = 0
    for name in NONFINAL_PARTITIONS:
        for game in games_by_partition[name]:
            record = raw_records.get(game.session_id)
            if record is None or not record.behavior_eligible:
                raise B2FreezeError("nonfinal raw record is absent")
            decisions = access.load_decisions(repository_root, record, raw_boundary)
            if len(decisions) != game.move_count:
                raise B2FreezeError("nonfinal decision count differs")
            for decision in decisions:
                phase = PHASE_NAME.get(
                    get_game_phase(decision.board, decision.board.turn)
                )
                if phase not in PHASES:
                    raise B2FreezeError("nonfinal phase is unsupported")
                phase_counts[name][phase] += 1
                color_counts[name][decision.board.turn] += 1
            opened += 1
            raw_bytes += raw_boundary.raw_size_by_path[record.canonical_file]
            if progress is not None:
                progress(opened, expected_games)
    return (
        {
            name: {
                "decisions_by_phase": {
                    phase: phase_counts[name][phase] for phase in PHASES
                },
                "decisions_by_actor_color": {
                    color: color_counts[name][color] for color in ("W", "B")
                },
            }
            for name in NONFINAL_PARTITIONS
        },
        {"raw_files_opened": opened, "raw_bytes_read": raw_bytes},
    )


def _benchmark_states(
    *,
    repository_root: Path,
    references: Sequence[Sequence[Any]],
    raw_boundary: F0D0Boundary,
    access: FrozenSplitAccess,
) -> tuple[list[ReplayedDecision], dict[str, Any]]:
    wanted: dict[str, set[int]] = defaultdict(set)
    for row in references:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 2
            or not isinstance(row[0], str)
            or isinstance(row[1], bool)
            or not isinstance(row[1], int)
        ):
            raise B2FreezeError("benchmark decision reference is invalid")
        wanted[row[0]].add(row[1])
    records = {record.session_id: record for record in raw_boundary.records}
    selected: dict[tuple[str, int], ReplayedDecision] = {}
    opened = 0
    raw_bytes = 0
    started = time.perf_counter()
    for session in sorted(wanted):
        record = records.get(session)
        if record is None or not record.behavior_eligible:
            raise B2FreezeError("benchmark session is absent")
        decisions = access.load_decisions(repository_root, record, raw_boundary)
        for logical_ply in wanted[session]:
            if logical_ply < 0 or logical_ply >= len(decisions):
                raise B2FreezeError("benchmark logical ply is out of range")
            selected[(session, logical_ply)] = decisions[logical_ply]
        opened += 1
        raw_bytes += raw_boundary.raw_size_by_path[record.canonical_file]
    elapsed = time.perf_counter() - started
    ordered = [selected[(str(row[0]), int(row[1]))] for row in references]
    if len(ordered) != len(references):
        raise B2FreezeError("benchmark state membership differs")
    return ordered, {
        "state_construction_seconds": elapsed,
        "raw_files_opened": opened,
        "raw_bytes_read": raw_bytes,
    }


def _upper_mean_95(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise B2FreezeError("cost benchmark has insufficient observations")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean + 1.959963984540054 * math.sqrt(variance / len(values))


def project_cost(
    *,
    total_decisions: int,
    state_build_seconds: float,
    sample_decisions: int,
    query_counts: Sequence[int],
    query_seconds: Sequence[float],
    safety_multiplier: float,
) -> dict[str, Any]:
    if total_decisions <= 0 or sample_decisions <= 1 or safety_multiplier < 1.0:
        raise B2FreezeError("cost projection inputs are invalid")
    queries_upper = _upper_mean_95([float(value) for value in query_counts])
    seconds_upper = _upper_mean_95(query_seconds)
    replay_per_state = state_build_seconds / sample_decisions
    projected_queries = math.ceil(
        total_decisions * queries_upper * safety_multiplier
    )
    projected_seconds = total_decisions * (
        replay_per_state + seconds_upper
    ) * safety_multiplier
    return {
        "logical_decisions": total_decisions,
        "queries_per_decision_upper_mean_95": queries_upper,
        "query_seconds_per_decision_upper_mean_95": seconds_upper,
        "state_construction_seconds_per_sample_decision": replay_per_state,
        "safety_multiplier": safety_multiplier,
        "projected_queries": projected_queries,
        "projected_active_seconds": projected_seconds,
    }


def run_malom_cost_benchmark(
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
    membership: Mapping[str, Any],
    games_by_partition: Mapping[str, Sequence[GameRecord]],
    raw_boundary: F0D0Boundary,
    access: FrozenSplitAccess,
    malom_path: str | Path,
    malom_manifest_path: str | Path,
) -> dict[str, Any]:
    spec = plan["malom_cost_benchmark"]
    if sha256_file(malom_manifest_path) != MALOM_MANIFEST_FILE_SHA256:
        raise B2FreezeError("Malom manifest file SHA-256 differs")
    try:
        snapshot = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=malom_manifest_path,
            full_hash=False,
        )
    except F0H0Error as exc:
        raise _translate_error(exc) from exc
    if (
        snapshot.get("trust_level") != "sector-corrected-v1"
        or snapshot.get("content_sha256") != MALOM_CONTENT_SHA256
    ):
        raise B2FreezeError("Malom snapshot trust identity differs")
    references = membership["malom_cost_preregistration"][
        "benchmark_decision_references"
    ]
    states, construction = _benchmark_states(
        repository_root=repository_root,
        references=references,
        raw_boundary=raw_boundary,
        access=access,
    )
    database = MalomDB(malom_path)
    if not database.is_available():
        raise B2FreezeError("Malom tablebase is unavailable")
    pass_rows: list[dict[str, Any]] = []
    first_query_counts: list[int] = []
    first_state_seconds: list[float] = []
    try:
        for pass_name in ("sector_cold_first_pass", "sector_warm_repeat_pass"):
            start = time.perf_counter()
            query_count = 0
            legal_actions = 0
            state_seconds: list[float] = []
            state_query_counts: list[int] = []
            cache_before = len(database._cache)  # noqa: SLF001 - benchmark audit
            for decision in states:
                state_start = time.perf_counter()
                try:
                    _tier, inventory, queries = _oracle_inventory(
                        decision.board,
                        database,
                    )
                except F0H0Error as exc:
                    raise _translate_error(exc) from exc
                state_seconds.append(time.perf_counter() - state_start)
                query_count += queries
                state_query_counts.append(queries)
                legal_actions += len(inventory)
            elapsed = time.perf_counter() - start
            cache_after = len(database._cache)  # noqa: SLF001 - benchmark audit
            pass_rows.append(
                {
                    "pass": pass_name,
                    "states": len(states),
                    "queries": query_count,
                    "legal_actions": legal_actions,
                    "elapsed_seconds": elapsed,
                    "queries_per_second": query_count / elapsed,
                    "sector_cache_entries_before": cache_before,
                    "sector_cache_entries_after": cache_after,
                    "sector_cache_entries_added": cache_after - cache_before,
                }
            )
            if pass_name == "sector_cold_first_pass":
                first_query_counts = state_query_counts
                first_state_seconds = state_seconds
    finally:
        database.close()
    if pass_rows[0]["queries"] != pass_rows[1]["queries"]:
        raise B2FreezeError("Malom repeated query inventory differs")
    full_decisions = sum(
        game.move_count
        for name in ("train", "selection")
        for game in games_by_partition[name]
    )
    projection = project_cost(
        total_decisions=full_decisions,
        state_build_seconds=float(construction["state_construction_seconds"]),
        sample_decisions=len(states),
        query_counts=first_query_counts,
        query_seconds=first_state_seconds,
        safety_multiplier=float(spec["projection_safety_multiplier"]),
    )
    threshold = float(spec["maximum_projected_active_seconds"])
    decision = "full" if projection["projected_active_seconds"] <= threshold else "sample"
    if decision == "full":
        selected_sessions = sorted(
            game.session_id
            for name in ("train", "selection")
            for game in games_by_partition[name]
        )
    else:
        selected_sessions = membership["malom_cost_preregistration"][
            "fallback_game_session_ids"
        ]
    by_session = {
        game.session_id: game
        for name in ("train", "selection")
        for game in games_by_partition[name]
    }
    selected_decisions = sum(
        by_session[session].move_count for session in selected_sessions
    )
    selected_projection = {
        **projection,
        "logical_decisions": selected_decisions,
        "projected_queries": math.ceil(
            projection["projected_queries"] * selected_decisions / full_decisions
        ),
        "projected_active_seconds": (
            projection["projected_active_seconds"]
            * selected_decisions
            / full_decisions
        ),
    }
    return {
        "purpose": "cost_only_no_screening_statistic",
        "label_boundary": "positional-only A_pos; never A_allow",
        "malom_snapshot": {
            key: snapshot[key]
            for key in (
                "content_sha256",
                "dataset_id",
                "manifest_file_sha256",
                "trust_level",
            )
        },
        "sample": {
            "decision_references": len(references),
            "decision_references_identity": membership[
                "malom_cost_preregistration"
            ]["benchmark_decision_references_identity"],
            **construction,
        },
        "passes": pass_rows,
        "projection": {
            **projection,
            "maximum_projected_active_seconds": threshold,
            "decision": decision,
            "selected_game_sessions": len(selected_sessions),
            "selected_game_sessions_identity": canonical_sha256(selected_sessions),
            "selected_projection": selected_projection,
        },
        "published_oracle_outcomes": 0,
        "published_safe_sets": 0,
    }


def run_characterization(
    *,
    repository_root: str | Path,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    membership: Mapping[str, Any],
    membership_file_sha256: str,
    boundary: Boundary,
    raw_boundary: F0D0Boundary,
    malom_path: str | Path,
    malom_manifest_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if membership.get("plan_identity") != plan.get("plan_identity"):
        raise B2FreezeError("B2 membership plan lineage differs")
    game_by_session = {game.session_id: game for game in boundary.games}
    games_by_partition: dict[str, list[GameRecord]] = {}
    for name in PARTITIONS:
        sessions = membership["partitions"][name]["session_ids"]
        try:
            games_by_partition[name] = [
                game_by_session[session] for session in sessions
            ]
        except KeyError as exc:
            raise B2FreezeError("B2 membership session is outside F0-D0") from exc
    access = FrozenSplitAccess.from_membership(membership)
    benchmark = run_malom_cost_benchmark(
        repository_root=Path(repository_root),
        plan=plan,
        membership=membership,
        games_by_partition=games_by_partition,
        raw_boundary=raw_boundary,
        access=access,
        malom_path=malom_path,
        malom_manifest_path=malom_manifest_path,
    )
    outcomes = _outcome_map(raw_boundary)
    metadata = {
        name: characterize_metadata(games_by_partition[name], outcomes)
        for name in NONFINAL_PARTITIONS
    }
    replay, replay_access = replay_characterization(
        repository_root=Path(repository_root),
        games_by_partition=games_by_partition,
        raw_boundary=raw_boundary,
        access=access,
        progress=progress,
    )
    for name in NONFINAL_PARTITIONS:
        metadata[name].update(replay[name])
        if sum(replay[name]["decisions_by_phase"].values()) != metadata[name][
            "logical_plies"
        ]:
            raise B2FreezeError("B2 phase accounting differs")
        if sum(replay[name]["decisions_by_actor_color"].values()) != metadata[
            name
        ]["logical_plies"]:
            raise B2FreezeError("B2 color accounting differs")
    if sum(
        metadata[name]["strict_outcome_eligible_games"]
        for name in NONFINAL_PARTITIONS
    ) != sum(
        1
        for name in NONFINAL_PARTITIONS
        for game in games_by_partition[name]
        if outcomes[game.session_id] is not None
    ):
        raise B2FreezeError("B2 strict outcome accounting differs")
    return {
        "schema_version": RESULT_SCHEMA,
        "freeze_id": plan["freeze_id"],
        "status": "b2_frozen_and_nonfinal_characterization_completed",
        "lineage": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha256,
            "membership_identity": membership["membership_identity"],
            "membership_file_sha256": membership_file_sha256,
            "f0d0_corpus_identity": boundary.corpus_identity,
            "f0d0_manifest_identity": boundary.manifest_identity,
            "f0d0_manifest_file_sha256": boundary.manifest_file_sha256,
        },
        "scope": {
            "official_split_frozen": True,
            "nonfinal_characterization_only": True,
            "independent_support_computed": False,
            "modifiable_state_reachability_computed": False,
            "concentration_computed": False,
            "product_effect_upper_bound_computed": False,
            "models_loaded": 0,
            "games_started": 0,
            "search_batches_started": 0,
            "training_updates": 0,
        },
        "claim_boundaries": {
            "safety": "positional-only A_pos; never A_allow",
            "population": "observed PlayOK-like source domain only",
            "not_product_ui_or_new_population_generalization": True,
            "history_selection_attenuation": {
                "excluded_games": 1_751,
                "excluded_draws": 35,
                "retained_games": 92_789,
                "retained_draws": 26_157,
                "nonrandom": True,
            },
            "unverified_terminal_basis_games": 54_923,
        },
        "partition_characterization": metadata,
        "final_test": {
            "games": membership["partitions"]["final-test"]["games"],
            "session_ids_identity": membership["partitions"]["final-test"][
                "session_ids_identity"
            ],
            "sealed": True,
            "content_statistics": None,
        },
        "malom_cost_benchmark": benchmark,
        "access_audit": {
            **replay_access,
            "benchmark_raw_files_opened": benchmark["sample"]["raw_files_opened"],
            "benchmark_raw_bytes_read": benchmark["sample"]["raw_bytes_read"],
            "final_test_raw_game_files_opened": 0,
            "final_test_decisions_loaded": 0,
            "final_test_derived_features_loaded": 0,
            "human_db_reads": 0,
            "database_writes": 0,
            "source_pool_2eb04f54_reads": 0,
            "source_pool_2eb04f54_records_consumed": 0,
        },
    }


def load_result(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        return _load_sealed_json(
            path,
            schema=RESULT_SCHEMA,
            identity_field="result_identity",
        )
    except SplitRetestError as exc:
        raise _translate_error(exc) from exc


__all__ = [
    "B2FreezeError",
    "EXPECTED_COUNTS",
    "FinalTestAccessError",
    "FrozenSplitAccess",
    "MEMBERSHIP_SCHEMA",
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "build_membership",
    "characterize_metadata",
    "load_membership",
    "load_plan",
    "load_result",
    "partition_b2_games",
    "project_cost",
    "run_characterization",
    "verify_implementation_artifacts",
    "write_sealed_json",
]
