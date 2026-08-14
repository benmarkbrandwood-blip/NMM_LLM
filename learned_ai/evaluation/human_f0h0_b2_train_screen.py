"""Train-only F0-H0 positional human-behaviour rejection screen.

The official B2 membership and its 10,000-game cost fallback are immutable
inputs.  This module intersects that fallback with the train partition and
never opens selection, confirmation, or final-test content.  All oracle
claims are positional-only and use ``A_pos``; no result from this module is a
full-history ``A_allow`` claim or an approval of a later experiment gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from ai.malom_db import MalomDB, OracleMoveValue, compare_oracle_move_values
from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase, terminal_wdl
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
    F0H0Error,
    ReplayedDecision,
    _read_raw_game,
    canonical_sha256,
    concentration,
    quantiles,
    replay_game,
    sha256_file,
    verify_malom_snapshot,
    wilson_interval,
    write_sealed_json,
)
from learned_ai.evaluation.oracle_corpus import _D4, _transform_board


PLAN_SCHEMA = "nmm.f0-h0-b2-train-rejection-plan.v1"
RESULT_SCHEMA = "nmm.f0-h0-b2-train-rejection-result.v1"

EXPECTED_CORPUS_IDENTITY = (
    "4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29"
)
EXPECTED_F0D0_MANIFEST_IDENTITY = (
    "bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7"
)
EXPECTED_F0D0_FILE_SHA256 = (
    "0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6"
)
EXPECTED_MEMBERSHIP_IDENTITY = (
    "06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b"
)
EXPECTED_MEMBERSHIP_FILE_SHA256 = (
    "06c3be92c87927d506dc36eb908aec3064220f4ead2ebb3b5ff3dfb7bf5032cb"
)
EXPECTED_CHARACTERIZATION_IDENTITY = (
    "183a39ab29ddfbec76a7188606b0a1297ffbdb845346a05753807f2c609b65e6"
)
EXPECTED_CHARACTERIZATION_FILE_SHA256 = (
    "7ab7f68b29072e0de132525970b9cbcbbf68b58d07bda8ed36117e54c45da779"
)
EXPECTED_FALLBACK_IDENTITY = (
    "d43ee042514d9dea389849e943a5fb9d0f2d6218f6e226a980afc9354e9c8cd4"
)
EXPECTED_PARTITION_COUNTS = {
    "train": 36_949,
    "selection": 887,
    "confirmation": 386,
    "final-test": 847,
}
EXPECTED_SAMPLE_COMPOSITION = {
    "train": 9_113,
    "selection": 887,
    "confirmation": 0,
    "final-test": 0,
}
PROTECTED_PARTITIONS = frozenset({"selection", "confirmation", "final-test"})
WDL_RANK = {"L": 0, "D": 1, "W": 2}
DOWNGRADE_TYPES = ("W->D", "W->L", "D->L")
PHASE_NAMES = {"place": "placement", "move": "movement", "fly": "flying"}

T = TypeVar("T")


class TrainScreenError(RuntimeError):
    """Raised when a train-screen input or invariant fails closed."""


class ProtectedPartitionAccessError(TrainScreenError):
    """Raised before protected B2 content can be opened or derived."""


class OracleCoverageAbstention(RuntimeError):
    """Raised only when a required positional tablebase entry is absent."""

    def __init__(self, reason: str, query_count: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.query_count = query_count


def _load_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainScreenError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise TrainScreenError(f"JSON root is not an object: {source}")
    return value, raw


def _load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    value, raw = _load_json(path)
    if value.get("schema_version") != schema:
        raise TrainScreenError(f"schema differs: {Path(path)}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise TrainScreenError(f"identity is absent: {Path(path)}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise TrainScreenError(f"identity differs: {Path(path)}")
    return value, hashlib.sha256(raw).hexdigest()


def load_screen_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate the immutable train-screen preregistration."""
    plan, file_sha = _load_sealed(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    boundary = plan.get("input_boundary")
    expected_boundary = {
        "f0d0_corpus_identity": EXPECTED_CORPUS_IDENTITY,
        "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "f0d0_manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
        "b2_membership_identity": EXPECTED_MEMBERSHIP_IDENTITY,
        "b2_membership_file_sha256": EXPECTED_MEMBERSHIP_FILE_SHA256,
        "b2_characterization_identity": EXPECTED_CHARACTERIZATION_IDENTITY,
        "b2_characterization_file_sha256": EXPECTED_CHARACTERIZATION_FILE_SHA256,
        "fallback_sample_identity": EXPECTED_FALLBACK_IDENTITY,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) != expected for key, expected in expected_boundary.items()
    ):
        raise TrainScreenError("train-screen input identity differs")
    sample = plan.get("sample")
    if (
        not isinstance(sample, Mapping)
        or sample.get("frozen_fallback_games") != 10_000
        or sample.get("membership_composition") != EXPECTED_SAMPLE_COMPOSITION
        or sample.get("analysis_partition") != "train"
        or sample.get("resampling_allowed") is not False
        or not isinstance(sample.get("train_session_ids_identity"), str)
        or len(sample["train_session_ids_identity"]) != 64
    ):
        raise TrainScreenError("train-screen sample contract differs")
    thresholds = plan.get("thresholds")
    required_thresholds = {
        "independent_support",
        "modifiable_reachability",
        "concentration",
        "estimability",
        "product_effect",
        "oracle_coverage",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != required_thresholds:
        raise TrainScreenError("train-screen threshold contract differs")
    if plan.get("statistics_partitions") != ["train"]:
        raise TrainScreenError("train-screen statistics partition differs")
    if plan.get("safe_set") != "A_pos" or plan.get("a_allow_claim") is not False:
        raise TrainScreenError("train-screen positional claim boundary differs")
    if plan.get("four_b_execution_rule") != "run_only_if_four_a_passes":
        raise TrainScreenError("train-screen estimability stop rule differs")
    return plan, file_sha


def load_screen_result(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and identity-check a completed train-screen manifest."""
    return _load_sealed(
        path,
        schema=RESULT_SCHEMA,
        identity_field="result_identity",
    )


def verify_implementation_artifacts(
    repository_root: str | Path,
    plan: Mapping[str, Any],
) -> None:
    root = Path(repository_root)
    artifacts = plan.get("implementation_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TrainScreenError("implementation artifact inventory is absent")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise TrainScreenError("implementation artifact row is invalid")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TrainScreenError("implementation artifact identity is invalid")
        if sha256_file(root / relative) != expected:
            raise TrainScreenError(f"implementation artifact differs: {relative}")


def load_characterization_identity(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load only the sealed B2 characterization identity and cost branch."""
    value, raw = _load_json(path)
    identity = value.get("result_identity")
    body = dict(value)
    body.pop("result_identity", None)
    file_sha = hashlib.sha256(raw).hexdigest()
    if (
        identity != EXPECTED_CHARACTERIZATION_IDENTITY
        or canonical_sha256(body) != EXPECTED_CHARACTERIZATION_IDENTITY
        or file_sha != EXPECTED_CHARACTERIZATION_FILE_SHA256
    ):
        raise TrainScreenError("B2 characterization identity differs")
    projection = value.get("malom_cost_benchmark", {}).get("projection", {})
    if (
        projection.get("decision") != "sample"
        or projection.get("selected_game_sessions_identity")
        != EXPECTED_FALLBACK_IDENTITY
    ):
        raise TrainScreenError("B2 frozen cost branch differs")
    return value, file_sha


@dataclass
class TrainOnlyAccess:
    """Fail-closed content accessor for one frozen B2 train sample."""

    partition_by_session: Mapping[str, str]
    allowed_train_sessions: frozenset[str]
    successful_accesses: Counter[tuple[str, str]] = field(default_factory=Counter)
    denied_attempts: Counter[tuple[str, str]] = field(default_factory=Counter)

    @classmethod
    def from_membership(
        cls,
        membership: Mapping[str, Any],
        allowed_train_sessions: Sequence[str],
    ) -> "TrainOnlyAccess":
        partitions = membership.get("partitions")
        if not isinstance(partitions, Mapping):
            raise TrainScreenError("B2 partitions are absent")
        partition_by_session: dict[str, str] = {}
        for name in EXPECTED_PARTITION_COUNTS:
            row = partitions.get(name)
            sessions = row.get("session_ids") if isinstance(row, Mapping) else None
            if not isinstance(sessions, list):
                raise TrainScreenError(f"B2 {name} membership is absent")
            for session in sessions:
                if session in partition_by_session:
                    raise TrainScreenError("B2 session belongs to multiple partitions")
                partition_by_session[str(session)] = name
        allowed = frozenset(str(value) for value in allowed_train_sessions)
        if not allowed or any(
            partition_by_session.get(value) != "train" for value in allowed
        ):
            raise TrainScreenError("analysis membership is not train-only")
        return cls(
            partition_by_session=partition_by_session, allowed_train_sessions=allowed
        )

    def assert_train(self, session_id: str, *, access_kind: str) -> None:
        partition = self.partition_by_session.get(
            session_id, "outside-frozen-membership"
        )
        if partition != "train" or session_id not in self.allowed_train_sessions:
            self.denied_attempts[(partition, access_kind)] += 1
            raise ProtectedPartitionAccessError(
                "train-only F0-H0 screen denied "
                f"{access_kind} for {partition} session {session_id}"
            )

    def read_raw_game(
        self,
        repository_root: Path,
        record: CorpusRecord,
        boundary: F0D0Boundary,
        *,
        reader: Callable[[Path, CorpusRecord, F0D0Boundary], Mapping[str, Any]]
        | None = None,
    ) -> Mapping[str, Any]:
        self.assert_train(record.session_id, access_kind="raw_game")
        selected_reader = _read_raw_game if reader is None else reader
        try:
            value = selected_reader(repository_root, record, boundary)
        except F0H0Error as exc:
            raise TrainScreenError(str(exc)) from exc
        self.successful_accesses[("train", "raw_game")] += 1
        return value

    def load_decisions(
        self,
        repository_root: Path,
        record: CorpusRecord,
        boundary: F0D0Boundary,
    ) -> list[ReplayedDecision]:
        self.assert_train(record.session_id, access_kind="decisions")
        raw = self.read_raw_game(repository_root, record, boundary)
        try:
            decisions = replay_game(raw, record)
        except F0H0Error as exc:
            raise TrainScreenError(str(exc)) from exc
        self.successful_accesses[("train", "decisions")] += 1
        return decisions

    def derive_features(self, session_id: str, producer: Callable[[], T]) -> T:
        self.assert_train(session_id, access_kind="derived_features")
        value = producer()
        self.successful_accesses[("train", "derived_features")] += 1
        return value


def derive_train_sample(
    membership: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Intersect the immutable 10,000-game fallback with B2 train."""
    partitions = membership.get("partitions")
    prereg = membership.get("malom_cost_preregistration")
    if not isinstance(partitions, Mapping) or not isinstance(prereg, Mapping):
        raise TrainScreenError("B2 sample membership is absent")
    fallback = prereg.get("fallback_game_session_ids")
    if (
        not isinstance(fallback, list)
        or len(fallback) != 10_000
        or canonical_sha256(fallback) != EXPECTED_FALLBACK_IDENTITY
    ):
        raise TrainScreenError("B2 fallback sample identity differs")
    sets: dict[str, set[str]] = {}
    for name, count in EXPECTED_PARTITION_COUNTS.items():
        row = partitions.get(name)
        sessions = row.get("session_ids") if isinstance(row, Mapping) else None
        if not isinstance(sessions, list) or len(sessions) != count:
            raise TrainScreenError(f"B2 {name} membership differs")
        sets[name] = set(str(value) for value in sessions)
    fallback_set = set(str(value) for value in fallback)
    if len(fallback_set) != 10_000:
        raise TrainScreenError("B2 fallback sample contains duplicates")
    composition = {name: len(fallback_set & values) for name, values in sets.items()}
    if composition != EXPECTED_SAMPLE_COMPOSITION:
        raise TrainScreenError("B2 fallback sample composition differs")
    train_sessions = sorted(fallback_set & sets["train"])
    if plan is not None:
        expected_identity = plan["sample"]["train_session_ids_identity"]
        if canonical_sha256(train_sessions) != expected_identity:
            raise TrainScreenError("train analysis membership identity differs")
    return train_sessions, composition


def exact_state_key(board: BoardState) -> str:
    """Return the exact positional predecessor FEN used by ``A_pos``."""
    return board.to_fen_string()


def ring16_state_key(board: BoardState) -> str:
    """Canonicalize one positional predecessor under the repository ring16."""
    return ring16_canonical_transition(board.to_fen_string(), None)[0]


def _split_fen(fen: str) -> tuple[str, str]:
    fields = fen.split("|")
    if len(fields) != 4 or len(fields[0]) != 24:
        raise TrainScreenError("NMM positional FEN is invalid")
    return fields[0], "|".join(fields[1:])


def ring16_canonical_transition(
    before_fen: str,
    after_fen: str | None,
) -> tuple[str, str | None]:
    """Canonicalize a state or settled transition with one shared transform.

    Canonicalizing the predecessor and successor separately can collapse two
    distinct actions.  This function applies the same D4/ring-swap operation
    to both boards, preserving the action relation within a ring16 class.
    """
    before_board, before_tail = _split_fen(before_fen)
    after_board: str | None = None
    after_tail: str | None = None
    if after_fen is not None:
        after_board, after_tail = _split_fen(after_fen)
    candidates: list[tuple[str, str | None]] = []
    for swapped in (False, True):
        source_before = (
            before_board[16:24] + before_board[8:16] + before_board[0:8]
            if swapped
            else before_board
        )
        source_after = None
        if after_board is not None:
            source_after = (
                after_board[16:24] + after_board[8:16] + after_board[0:8]
                if swapped
                else after_board
            )
        for matrix in _D4:
            canonical_before = (
                f"{_transform_board(source_before, matrix)}|{before_tail}"
            )
            canonical_after = (
                None
                if source_after is None or after_tail is None
                else f"{_transform_board(source_after, matrix)}|{after_tail}"
            )
            candidates.append((canonical_before, canonical_after))
    return min(candidates, key=lambda item: (item[0], item[1] or ""))


@dataclass
class SupportCell:
    observations: int = 0
    players: set[str] = field(default_factory=set)
    games: set[str] = field(default_factory=set)

    def observe(self, player: str, game: str) -> None:
        self.observations += 1
        self.players.add(player)
        self.games.add(game)


def support_summary(
    cells: Mapping[str, SupportCell],
    *,
    minimum_players: int,
    minimum_games: int,
    tail_thresholds: Sequence[int],
) -> tuple[dict[str, Any], set[str]]:
    if not cells:
        raise TrainScreenError("state-support population is empty")
    supported = {
        key
        for key, cell in cells.items()
        if len(cell.players) >= minimum_players and len(cell.games) >= minimum_games
    }
    total = sum(cell.observations for cell in cells.values())
    supported_observations = sum(cells[key].observations for key in supported)
    observations = [cell.observations for cell in cells.values()]
    players = [len(cell.players) for cell in cells.values()]
    games = [len(cell.games) for cell in cells.values()]
    return (
        {
            "states_or_classes": len(cells),
            "observations": total,
            "observation_count_quantiles": quantiles(observations),
            "independent_player_count_quantiles": quantiles(players),
            "independent_game_count_quantiles": quantiles(games),
            "observation_count_tails": {
                str(value): sum(count >= value for count in observations)
                for value in tail_thresholds
            },
            "support_floor": {
                "minimum_independent_players": minimum_players,
                "minimum_independent_games": minimum_games,
            },
            "supported_states_or_classes": len(supported),
            "supported_state_or_class_fraction": len(supported) / len(cells),
            "supported_observations": supported_observations,
            "supported_decision_fraction": supported_observations / total,
        },
        supported,
    )


def clustered_proportion(
    successes_by_game: Mapping[str, int],
    totals_by_game: Mapping[str, int],
) -> dict[str, Any]:
    """Fixed-sample proportion with a whole-game cluster-robust interval."""
    games = [game for game, total in totals_by_game.items() if total > 0]
    if not games:
        raise TrainScreenError("clustered proportion input is empty")
    successes = sum(int(successes_by_game.get(game, 0)) for game in games)
    total = sum(int(totals_by_game[game]) for game in games)
    if successes < 0 or successes > total:
        raise TrainScreenError("clustered proportion counts are invalid")
    point = successes / total
    wilson = wilson_interval(successes, total)
    if len(games) < 2:
        lower, upper, standard_error = 0.0, 1.0, None
    else:
        score_squares = sum(
            (successes_by_game.get(game, 0) - point * totals_by_game[game]) ** 2
            for game in games
        )
        variance = len(games) / (len(games) - 1) * score_squares / total**2
        standard_error = math.sqrt(max(0.0, variance))
        lower = max(0.0, point - 1.959963984540054 * standard_error)
        upper = min(1.0, point + 1.959963984540054 * standard_error)
    return {
        "point": point,
        "successes": successes,
        "observations": total,
        "independent_game_clusters": len(games),
        "lower_95": min(lower, wilson["lower_95"]),
        "upper_95": max(upper, wilson["upper_95"]),
        "cluster_robust_standard_error": standard_error,
        "fixed_membership_wilson": wilson,
        "method": "envelope_of_game_cluster_normal_and_fixed_membership_wilson",
    }


def _oracle_inventory_positional(
    board: BoardState,
    database: MalomDB,
) -> tuple[str, list[tuple[Mapping[str, Any], OracleMoveValue]], int]:
    parent = database.query_value(board)
    if parent is None:
        raise OracleCoverageAbstention("parent_position_unavailable", 1)
    if parent.outcome not in WDL_RANK:
        raise TrainScreenError("parent Malom outcome is invalid")
    results: list[tuple[Mapping[str, Any], OracleMoveValue]] = []
    query_count = 1
    for move in get_all_legal_moves(board):
        after = board.apply_move(move)
        rules_value = terminal_wdl(after)
        if rules_value is not None:
            value = database.terminal_move_value(parent, rules_value)
        else:
            child = database.query_value(after)
            query_count += 1
            if child is None:
                raise OracleCoverageAbstention(
                    "successor_position_unavailable",
                    query_count,
                )
            value = database.move_value(parent, child)
        if value.outcome not in WDL_RANK:
            raise TrainScreenError("successor Malom outcome is invalid")
        results.append((dict(move), value))
    if not results:
        raise TrainScreenError("human decision has no legal positional action")
    best_tier = max((value.outcome for _move, value in results), key=WDL_RANK.get)
    if best_tier != parent.outcome:
        raise TrainScreenError("candidate inventory contradicts parent Malom tier")
    contexts = {
        (value.sector, value.sector_value, value.perspective)
        for _move, value in results
    }
    if len(contexts) != 1:
        raise TrainScreenError("candidate inventory has mixed Malom contexts")
    return parent.outcome, results, query_count


def _move_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field) for field in ("from", "to", "capture")
    )


@dataclass(frozen=True)
class OracleDecision:
    exact_key: str
    ring16_key: str
    exact_action: str
    ring16_action: str
    exact_a_pos_actions: frozenset[str]
    ring16_a_pos_actions: frozenset[str]
    parent_tier: str
    chosen_tier: str
    a_pos_cardinality: int
    chosen_preserves: bool
    within_tier_regret: bool
    normalized_within_tier_regret: float
    legal_actions: int
    query_count: int
    phase: str
    color: str
    actor_player_key: str
    game_id: str


def label_oracle_decision(
    decision: ReplayedDecision,
    database: MalomDB,
) -> OracleDecision:
    parent_tier, inventory, query_count = _oracle_inventory_positional(
        decision.board,
        database,
    )
    chosen_rows = [
        (move, value) for move, value in inventory if _move_matches(move, decision.move)
    ]
    if len(chosen_rows) != 1:
        raise TrainScreenError("observed human action is absent from Malom inventory")
    chosen_move, chosen_value = chosen_rows[0]
    a_pos = [(move, value) for move, value in inventory if value.outcome == parent_tier]
    if not a_pos:
        raise TrainScreenError("A_pos is empty")
    before_fen = decision.board.to_fen_string()
    exact_actions: set[str] = set()
    ring_actions: set[str] = set()
    chosen_exact = ""
    chosen_ring = ""
    for move, _value in a_pos:
        after_fen = decision.board.apply_move(move).to_fen_string()
        exact_action = after_fen
        ring_before, ring_after = ring16_canonical_transition(before_fen, after_fen)
        if ring_after is None:
            raise TrainScreenError("ring16 transition successor is absent")
        ring_action = canonical_sha256([ring_before, ring_after])
        exact_actions.add(exact_action)
        ring_actions.add(ring_action)
        if _move_matches(move, chosen_move):
            chosen_exact = exact_action
            chosen_ring = ring_action
    chosen_preserves = chosen_value.outcome == parent_tier
    if chosen_preserves and (not chosen_exact or not chosen_ring):
        raise TrainScreenError("preserving observed action is absent from A_pos")
    if not chosen_preserves:
        after_fen = decision.board.apply_move(chosen_move).to_fen_string()
        chosen_exact = after_fen
        ring_before, ring_after = ring16_canonical_transition(before_fen, after_fen)
        if ring_after is None:
            raise TrainScreenError("ring16 observed transition successor is absent")
        chosen_ring = canonical_sha256([ring_before, ring_after])
    best_full = max(inventory, key=lambda item: item[1].ordering_key())[1]
    within_tier_regret = bool(
        chosen_preserves and compare_oracle_move_values(chosen_value, best_full) < 0
    )
    levels = sorted(
        {value.ordering_key() for _move, value in a_pos},
        reverse=True,
    )
    chosen_level = chosen_value.ordering_key()
    normalized_regret = (
        levels.index(chosen_level) / (len(levels) - 1)
        if chosen_preserves and len(levels) > 1
        else 0.0
    )
    ring_before, _ = ring16_canonical_transition(before_fen, None)
    raw_phase = get_game_phase(decision.board, decision.board.turn)
    if raw_phase not in PHASE_NAMES:
        raise TrainScreenError("human decision phase is invalid")
    return OracleDecision(
        exact_key=before_fen,
        ring16_key=ring_before,
        exact_action=chosen_exact,
        ring16_action=chosen_ring,
        exact_a_pos_actions=frozenset(exact_actions),
        ring16_a_pos_actions=frozenset(ring_actions),
        parent_tier=parent_tier,
        chosen_tier=chosen_value.outcome,
        a_pos_cardinality=len(a_pos),
        chosen_preserves=chosen_preserves,
        within_tier_regret=within_tier_regret,
        normalized_within_tier_regret=normalized_regret,
        legal_actions=len(inventory),
        query_count=query_count,
        phase=PHASE_NAMES[raw_phase],
        color=decision.board.turn,
        actor_player_key=decision.actor_player_key,
        game_id=decision.game_id,
    )


def downgrade_type(label: OracleDecision) -> str | None:
    if WDL_RANK[label.chosen_tier] >= WDL_RANK[label.parent_tier]:
        return None
    transition = f"{label.parent_tier}->{label.chosen_tier}"
    if transition not in DOWNGRADE_TYPES:
        raise TrainScreenError("unexpected positional downgrade transition")
    return transition


@dataclass
class EstimabilityCell:
    exposures: int = 0
    games: set[str] = field(default_factory=set)
    players: set[str] = field(default_factory=set)
    action_counts: Counter[str] = field(default_factory=Counter)
    fold_action_totals: dict[int, Counter[str]] = field(
        default_factory=lambda: {0: Counter(), 1: Counter()}
    )
    fold_action_events: dict[int, dict[str, Counter[str]]] = field(
        default_factory=lambda: {
            0: defaultdict(Counter),
            1: defaultdict(Counter),
        }
    )

    def observe(
        self,
        *,
        game: str,
        players: Sequence[str],
        action: str,
        fold: int,
        event: str | None,
    ) -> None:
        self.exposures += 1
        self.games.add(game)
        self.players.update(players)
        self.action_counts[action] += 1
        self.fold_action_totals[fold][action] += 1
        if event is not None:
            self.fold_action_events[fold][action][event] += 1


def _fold_for_game(seed: str, game: str) -> int:
    digest = hashlib.sha256(f"f0-h0-crossfit-v1\0{seed}\0{game}".encode()).digest()
    return digest[0] & 1


def estimability_summary(
    cells: Mapping[str, EstimabilityCell],
    *,
    supported_keys: set[str],
    total_analysis_decisions: int,
    minimum_observations_k: int,
    minimum_per_action_m: int,
) -> tuple[dict[str, Any], set[str]]:
    repeated = {
        key
        for key, cell in cells.items()
        if key in supported_keys and cell.exposures >= minimum_observations_k
    }
    varied = {
        key
        for key in repeated
        if sum(
            count >= minimum_per_action_m for count in cells[key].action_counts.values()
        )
        >= 2
    }
    covered_decisions = sum(cells[key].exposures for key in varied)
    games = set().union(*(cells[key].games for key in varied)) if varied else set()
    players = set().union(*(cells[key].players for key in varied)) if varied else set()
    return (
        {
            "support_qualified_classes": sum(key in supported_keys for key in cells),
            "classes_with_observations_at_least_k": len(repeated),
            "classes_with_two_observed_safe_actions_each_at_least_m": len(varied),
            "covered_exposures": covered_decisions,
            "covered_decision_fraction": (
                covered_decisions / total_analysis_decisions
                if total_analysis_decisions
                else 0.0
            ),
            "covered_games": len(games),
            "covered_independent_players": len(players),
            "k": minimum_observations_k,
            "m": minimum_per_action_m,
        },
        varied,
    )


def _weighted_mean(rows: Sequence[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _value, weight in rows)
    if denominator <= 0:
        return None
    return sum(value * weight for value, weight in rows) / denominator


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise TrainScreenError("bootstrap distribution is empty")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def estimate_action_effects(
    cells: Mapping[str, EstimabilityCell],
    *,
    eligible_keys: set[str],
    minimum_per_action_m: int,
    selection_minimum: int,
    evaluation_minimum: int,
    minimum_crossfit_classes: int,
    bootstrap_replicates: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    """Report naive max-min and cross-fitted Jeffreys-shrunk lift."""
    results: dict[str, Any] = {}
    for event in DOWNGRADE_TYPES:
        event_exposures = sum(
            cells[key].exposures for key in eligible_keys if key in cells
        )
        observed_events = sum(
            cells[key].fold_action_events[fold][action][event]
            for key in eligible_keys
            if key in cells
            for fold in (0, 1)
            for action in cells[key].fold_action_totals[fold]
        )
        eligible_varied_classes = sum(
            sum(
                count >= minimum_per_action_m
                for count in cells[key].action_counts.values()
            )
            >= 2
            for key in eligible_keys
            if key in cells
        )
        if observed_events == 0:
            if event_exposures <= 0:
                raise TrainScreenError("zero-event transition has no exposures")
            event_rate_interval = wilson_interval(0, event_exposures)
            results[event] = {
                "status": "no_observed_transition_events",
                "observed_transition_events": 0,
                "eligible_exposures": event_exposures,
                "uncorrected_weighted_within_class_max_minus_min": 0.0,
                "uncorrected_classes": eligible_varied_classes,
                "corrected_crossfit_classes": 0,
                "minimum_crossfit_classes": minimum_crossfit_classes,
                "corrected_point": 0.0,
                "conservative_lower_95": 0.0,
                "conservative_upper_95": event_rate_interval["upper_95"],
                "upper_bound_method": (
                    "fixed-membership Wilson upper bound on the zero-event "
                    "exposure rate; no action contrast is asserted"
                ),
                "bootstrap_replicates": 0,
            }
            continue
        naive_rows: list[tuple[float, float]] = []
        corrected_by_class: dict[str, tuple[float, float]] = {}
        for key in sorted(eligible_keys):
            cell = cells[key]
            actions = [
                action
                for action, count in cell.action_counts.items()
                if count >= minimum_per_action_m
            ]
            if len(actions) < 2:
                continue
            rates = []
            for action in actions:
                total = cell.action_counts[action]
                successes = sum(
                    cell.fold_action_events[fold][action][event] for fold in (0, 1)
                )
                rates.append((successes / total, action, total))
            maximum = max(rates, key=lambda row: (row[0], row[1]))
            minimum = min(rates, key=lambda row: (row[0], row[1]))
            naive_rows.append((maximum[0] - minimum[0], sum(row[2] for row in rates)))

            fold_rows: list[tuple[float, float]] = []
            for evaluation_fold in (0, 1):
                selection_fold = 1 - evaluation_fold
                selection_actions = [
                    action
                    for action in actions
                    if cell.fold_action_totals[selection_fold][action]
                    >= selection_minimum
                ]
                if len(selection_actions) < 2:
                    continue

                def selection_rate(action: str) -> float:
                    total = cell.fold_action_totals[selection_fold][action]
                    successes = cell.fold_action_events[selection_fold][action][event]
                    return (successes + 0.5) / (total + 1.0)

                high = max(
                    selection_actions,
                    key=lambda action: (selection_rate(action), action),
                )
                low = min(
                    selection_actions,
                    key=lambda action: (selection_rate(action), action),
                )
                if high == low:
                    continue
                high_total = cell.fold_action_totals[evaluation_fold][high]
                low_total = cell.fold_action_totals[evaluation_fold][low]
                if high_total < evaluation_minimum or low_total < evaluation_minimum:
                    continue
                high_success = cell.fold_action_events[evaluation_fold][high][event]
                low_success = cell.fold_action_events[evaluation_fold][low][event]
                high_rate = (high_success + 0.5) / (high_total + 1.0)
                low_rate = (low_success + 0.5) / (low_total + 1.0)
                harmonic_weight = (
                    2.0 * high_total * low_total / (high_total + low_total)
                )
                fold_rows.append((high_rate - low_rate, harmonic_weight))
            point = _weighted_mean(fold_rows)
            if point is not None:
                corrected_by_class[key] = (
                    sum(value * weight for value, weight in fold_rows),
                    sum(weight for _value, weight in fold_rows),
                )

        naive = _weighted_mean(naive_rows)
        class_count = len(corrected_by_class)
        if class_count < minimum_crossfit_classes:
            results[event] = {
                "status": "insufficient_crossfit_state_classes",
                "uncorrected_weighted_within_class_max_minus_min": naive,
                "uncorrected_classes": len(naive_rows),
                "corrected_crossfit_classes": class_count,
                "minimum_crossfit_classes": minimum_crossfit_classes,
                "corrected_point": None,
                "conservative_lower_95": None,
            }
            continue
        contributions = list(corrected_by_class.values())
        corrected_point = sum(row[0] for row in contributions) / sum(
            row[1] for row in contributions
        )
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"{bootstrap_seed}\0{event}".encode()).digest()[:8]
            )
        )
        draws: list[float] = []
        for _index in range(bootstrap_replicates):
            selected = [
                contributions[rng.randrange(class_count)] for _ in range(class_count)
            ]
            denominator = sum(row[1] for row in selected)
            if denominator > 0:
                draws.append(sum(row[0] for row in selected) / denominator)
        results[event] = {
            "status": "estimated",
            "uncorrected_weighted_within_class_max_minus_min": naive,
            "uncorrected_classes": len(naive_rows),
            "corrected_crossfit_classes": class_count,
            "minimum_crossfit_classes": minimum_crossfit_classes,
            "corrected_point": corrected_point,
            "conservative_lower_95": _percentile(draws, 0.025),
            "conservative_upper_95": _percentile(draws, 0.975),
            "bootstrap_replicates": len(draws),
        }
    return results


def _serialize_counter(counter: Mapping[Any, int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda row: str(row[0]))
    }


def _input_file_rows(
    records: Sequence[CorpusRecord],
    boundary: F0D0Boundary,
) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": record.canonical_file,
            "size_bytes": boundary.raw_size_by_path[record.canonical_file],
            "sha256": boundary.raw_sha256_by_path[record.canonical_file],
        }
        for record in sorted(records, key=lambda item: item.canonical_file)
    ]


def run_train_screen(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    membership: Mapping[str, Any],
    membership_file_sha256: str,
    characterization_file_sha256: str,
    malom_path: str | Path,
    malom_manifest_path: str | Path,
    ruleset_path: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute the frozen 9,113-game train-only rejection screen."""
    notify = progress if progress is not None else (lambda _message: None)
    root = Path(repository_root)
    if membership_file_sha256 != EXPECTED_MEMBERSHIP_FILE_SHA256:
        raise TrainScreenError("official B2 membership file SHA-256 differs")
    if characterization_file_sha256 != EXPECTED_CHARACTERIZATION_FILE_SHA256:
        raise TrainScreenError("B2 characterization file SHA-256 differs")
    if boundary.file_sha256 != EXPECTED_F0D0_FILE_SHA256:
        raise TrainScreenError("F0-D0 manifest file SHA-256 differs")
    train_sessions, sample_composition = derive_train_sample(membership, plan)
    by_session = {record.session_id: record for record in boundary.records}
    try:
        records = [by_session[session] for session in train_sessions]
    except KeyError as exc:
        raise TrainScreenError("train sample session is absent from F0-D0") from exc
    if any(not record.behavior_eligible for record in records):
        raise TrainScreenError("train sample includes a behavior-ineligible game")
    access = TrainOnlyAccess.from_membership(membership, train_sessions)
    support_spec = plan["thresholds"]["independent_support"]
    exact_support: dict[str, SupportCell] = defaultdict(SupportCell)
    ring_support: dict[str, SupportCell] = defaultdict(SupportCell)
    player_decisions: Counter[str] = Counter()
    game_decisions: Counter[str] = Counter()
    first_pass_decisions = 0
    for index, record in enumerate(records, 1):
        decisions = access.load_decisions(root, record, boundary)
        game_decisions[record.session_id] = len(decisions)
        first_pass_decisions += len(decisions)
        for decision in decisions:
            exact = exact_state_key(decision.board)
            ring = ring16_state_key(decision.board)
            exact_support[exact].observe(decision.actor_player_key, record.session_id)
            ring_support[ring].observe(decision.actor_player_key, record.session_id)
            player_decisions[decision.actor_player_key] += 1
        if index % 500 == 0 or index == len(records):
            notify(f"support-pass {index}/{len(records)} games")
    if first_pass_decisions != sum(record.move_count for record in records):
        raise TrainScreenError("strict replay decision count differs from F0-D0")
    tails = support_spec["observation_tail_thresholds"]
    exact_summary, exact_supported = support_summary(
        exact_support,
        minimum_players=int(support_spec["minimum_independent_players"]),
        minimum_games=int(support_spec["minimum_independent_games"]),
        tail_thresholds=tails,
    )
    ring_summary, ring_supported = support_summary(
        ring_support,
        minimum_players=int(support_spec["minimum_independent_players"]),
        minimum_games=int(support_spec["minimum_independent_games"]),
        tail_thresholds=tails,
    )

    snapshot = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=malom_manifest_path,
        full_hash=False,
    )
    expected_malom = plan["input_boundary"]["malom"]
    for name in (
        "content_sha256",
        "dataset_id",
        "manifest_file_sha256",
        "trust_level",
    ):
        if snapshot.get(name) != expected_malom.get(name):
            raise TrainScreenError("Malom snapshot differs from preregistration")
    if snapshot.get("trust_level") != "sector-corrected-v1":
        raise TrainScreenError("Malom label version is not sector-corrected-v1")
    database = MalomDB(malom_path)
    if not database.is_available():
        raise TrainScreenError("Malom tablebase is unavailable")

    a_pos_counts: Counter[int] = Counter()
    strata: Counter[tuple[str, str, str, str]] = Counter()
    oracle_abstentions: Counter[str] = Counter()
    oracle_queries = 0
    covered_decisions = 0
    game_totals: Counter[str] = Counter()
    game_modifiable: Counter[str] = Counter()
    game_supported_modifiable: Counter[str] = Counter()
    supported_modifiable_players: Counter[str] = Counter()
    ring_a_pos_signatures: dict[str, frozenset[str]] = {}
    exact_a_pos_signatures: dict[str, frozenset[str]] = {}
    exact_estimability: dict[str, EstimabilityCell] = defaultdict(EstimabilityCell)
    ring_estimability: dict[str, EstimabilityCell] = defaultdict(EstimabilityCell)
    first_downgrades: Counter[str] = Counter()
    guided_first_downgrades: Counter[str] = Counter()
    supported_guided_first_downgrades: Counter[str] = Counter()
    fully_covered_games = 0
    first_downgrade_games = 0
    within_tier_regret_count = 0
    preserving_count = 0
    normalized_regret_sum = 0.0
    exposure_count = 0
    exposure_support_abstentions = 0
    try:
        for index, (record, session) in enumerate(
            zip(records, train_sessions, strict=True), 1
        ):
            decisions = access.load_decisions(root, record, boundary)
            labels: list[OracleDecision | None] = []
            for decision in decisions:
                game_totals[session] += 1
                try:
                    label = label_oracle_decision(decision, database)
                except OracleCoverageAbstention as exc:
                    oracle_abstentions[exc.reason] += 1
                    oracle_queries += exc.query_count
                    labels.append(None)
                    continue
                labels.append(label)
                oracle_queries += label.query_count
                covered_decisions += 1
                a_pos_counts[label.a_pos_cardinality] += 1
                modifiable = label.a_pos_cardinality > 1
                ring_qualified = label.ring16_key in ring_supported
                game_modifiable[session] += int(modifiable)
                game_supported_modifiable[session] += int(modifiable and ring_qualified)
                if modifiable and ring_qualified:
                    supported_modifiable_players[label.actor_player_key] += 1
                strata[
                    (
                        label.phase,
                        label.color,
                        label.parent_tier,
                        "modifiable" if modifiable else "forced",
                    )
                ] += 1
                prior_exact = exact_a_pos_signatures.setdefault(
                    label.exact_key,
                    label.exact_a_pos_actions,
                )
                prior_ring = ring_a_pos_signatures.setdefault(
                    label.ring16_key,
                    label.ring16_a_pos_actions,
                )
                if prior_exact != label.exact_a_pos_actions:
                    raise TrainScreenError("exact-state A_pos signature changed")
                if prior_ring != label.ring16_a_pos_actions:
                    raise TrainScreenError("ring16-state A_pos signature changed")
                if label.chosen_preserves:
                    preserving_count += 1
                    normalized_regret_sum += label.normalized_within_tier_regret
                    within_tier_regret_count += int(label.within_tier_regret)

            if all(label is not None for label in labels):
                fully_covered_games += 1
                covered_labels = [label for label in labels if label is not None]
                first_index: int | None = None
                first_event: str | None = None
                for label_index, label in enumerate(covered_labels):
                    event = downgrade_type(label)
                    if event is not None:
                        first_index = label_index
                        first_event = event
                        break
                if first_event is not None and first_index is not None:
                    first_downgrades[first_event] += 1
                    first_downgrade_games += 1
                    if first_index > 0:
                        predecessor = covered_labels[first_index - 1]
                        if (
                            predecessor.chosen_preserves
                            and predecessor.a_pos_cardinality > 1
                        ):
                            guided_first_downgrades[first_event] += 1
                            if predecessor.ring16_key in ring_supported:
                                supported_guided_first_downgrades[first_event] += 1

                exposure_end = (
                    first_index if first_index is not None else len(covered_labels)
                )
                for label_index in range(max(0, exposure_end)):
                    if label_index + 1 >= len(covered_labels):
                        break
                    label = covered_labels[label_index]
                    if not label.chosen_preserves or label.a_pos_cardinality <= 1:
                        continue
                    exposure_count += 1
                    response_event = (
                        first_event
                        if first_index is not None and label_index + 1 == first_index
                        else None
                    )
                    fold = _fold_for_game(
                        str(plan["thresholds"]["estimability"]["crossfit_seed"]),
                        session,
                    )
                    players = record.player_keys
                    if label.exact_key in exact_supported:
                        exact_estimability[label.exact_key].observe(
                            game=session,
                            players=players,
                            action=label.exact_action,
                            fold=fold,
                            event=response_event,
                        )
                    if label.ring16_key in ring_supported:
                        ring_estimability[label.ring16_key].observe(
                            game=session,
                            players=players,
                            action=label.ring16_action,
                            fold=fold,
                            event=response_event,
                        )
                    else:
                        exposure_support_abstentions += 1
            if index % 250 == 0 or index == len(records):
                notify(f"oracle-pass {index}/{len(records)} games")
    finally:
        database.close()

    oracle_decisions = first_pass_decisions
    coverage_spec = plan["thresholds"]["oracle_coverage"]
    coverage_fraction = covered_decisions / oracle_decisions
    modifiable_interval = clustered_proportion(game_modifiable, game_totals)
    supported_modifiable_interval = clustered_proportion(
        game_supported_modifiable,
        game_totals,
    )
    one_per_game = {session: 1 for session in train_sessions}
    supported_modifiable_game_reach = clustered_proportion(
        {
            session: int(game_supported_modifiable.get(session, 0) > 0)
            for session in train_sessions
        },
        one_per_game,
    )
    estimability_spec = plan["thresholds"]["estimability"]
    exact_estimability_summary, exact_estimable_keys = estimability_summary(
        exact_estimability,
        supported_keys=exact_supported,
        total_analysis_decisions=oracle_decisions,
        minimum_observations_k=int(estimability_spec["k"]),
        minimum_per_action_m=int(estimability_spec["m"]),
    )
    ring_estimability_summary, ring_estimable_keys = estimability_summary(
        ring_estimability,
        supported_keys=ring_supported,
        total_analysis_decisions=oracle_decisions,
        minimum_observations_k=int(estimability_spec["k"]),
        minimum_per_action_m=int(estimability_spec["m"]),
    )
    four_a_gates = {
        "minimum_ring16_estimable_classes": (
            ring_estimability_summary[
                "classes_with_two_observed_safe_actions_each_at_least_m"
            ]
            >= int(estimability_spec["minimum_ring16_classes"])
        ),
        "minimum_ring16_covered_decision_fraction": (
            ring_estimability_summary["covered_decision_fraction"]
            >= float(estimability_spec["minimum_ring16_covered_decision_fraction"])
        ),
        "minimum_ring16_covered_games": (
            ring_estimability_summary["covered_games"]
            >= int(estimability_spec["minimum_ring16_covered_games"])
        ),
        "minimum_ring16_covered_players": (
            ring_estimability_summary["covered_independent_players"]
            >= int(estimability_spec["minimum_ring16_covered_players"])
        ),
    }
    four_a_passes = all(four_a_gates.values())
    action_effects = None
    if four_a_passes:
        action_effects = estimate_action_effects(
            ring_estimability,
            eligible_keys=ring_estimable_keys,
            minimum_per_action_m=int(estimability_spec["m"]),
            selection_minimum=int(estimability_spec["crossfit_selection_minimum"]),
            evaluation_minimum=int(estimability_spec["crossfit_evaluation_minimum"]),
            minimum_crossfit_classes=int(estimability_spec["minimum_crossfit_classes"]),
            bootstrap_replicates=int(estimability_spec["bootstrap_replicates"]),
            bootstrap_seed=str(estimability_spec["bootstrap_seed"]),
        )

    product_spec = plan["thresholds"]["product_effect"]
    guided_total = sum(supported_guided_first_downgrades.values())
    mechanism_availability_observed = wilson_interval(
        guided_total,
        fully_covered_games,
    )
    mechanism_availability = wilson_interval(guided_total, len(records))
    action_effect_gate = bool(
        action_effects is not None
        and any(
            row.get("status") == "estimated"
            and row.get("conservative_lower_95") is not None
            and row.get("conservative_upper_95") is not None
            and row["conservative_upper_95"]
            >= float(product_spec["minimum_signable_absolute_effect"])
            for row in action_effects.values()
        )
    )

    player_concentration = concentration(list(player_decisions.values()))
    game_concentration = concentration(list(game_decisions.values()))
    exact_state_concentration = concentration(
        [cell.observations for cell in exact_support.values()]
    )
    ring_state_concentration = concentration(
        [cell.observations for cell in ring_support.values()]
    )
    supported_player_concentration = (
        concentration(list(supported_modifiable_players.values()))
        if supported_modifiable_players
        else None
    )
    concentration_spec = plan["thresholds"]["concentration"]
    support_gates = {
        "minimum_ring16_supported_classes": (
            ring_summary["supported_states_or_classes"]
            >= int(support_spec["minimum_ring16_supported_classes"])
        ),
        "minimum_ring16_supported_decision_fraction": (
            ring_summary["supported_decision_fraction"]
            >= float(support_spec["minimum_ring16_supported_decision_fraction"])
        ),
    }
    reach_spec = plan["thresholds"]["modifiable_reachability"]
    reach_gates = {
        "minimum_modifiable_decisions": (
            modifiable_interval["successes"]
            >= int(reach_spec["minimum_modifiable_decisions"])
        ),
        "minimum_modifiable_fraction_lcb": (
            modifiable_interval["lower_95"]
            >= float(reach_spec["minimum_modifiable_fraction_lcb"])
        ),
        "minimum_supported_modifiable_fraction_lcb": (
            supported_modifiable_interval["lower_95"]
            >= float(reach_spec["minimum_supported_modifiable_fraction_lcb"])
        ),
        "minimum_supported_modifiable_game_reach_lcb": (
            supported_modifiable_game_reach["lower_95"]
            >= float(reach_spec["minimum_supported_modifiable_game_reach_lcb"])
        ),
    }
    concentration_gates = {
        "maximum_player_top_1_percent_share": (
            player_concentration["top_1_percent_share"]
            <= float(concentration_spec["maximum_player_top_1_percent_share"])
        ),
        "maximum_player_top_5_percent_share": (
            player_concentration["top_5_percent_share"]
            <= float(concentration_spec["maximum_player_top_5_percent_share"])
        ),
        "maximum_player_top_10_percent_share": (
            player_concentration["top_10_percent_share"]
            <= float(concentration_spec["maximum_player_top_10_percent_share"])
        ),
        "maximum_player_gini": (
            player_concentration["gini"]
            <= float(concentration_spec["maximum_player_gini"])
        ),
        "minimum_player_kish_effective_units": (
            player_concentration["kish_effective_units"]
            >= float(concentration_spec["minimum_player_kish_effective_units"])
        ),
        "minimum_supported_modifiable_players": (
            len(supported_modifiable_players)
            >= int(concentration_spec["minimum_supported_modifiable_players"])
        ),
        "maximum_supported_modifiable_player_top_5_share": (
            supported_player_concentration is not None
            and supported_player_concentration["top_5_percent_share"]
            <= float(
                concentration_spec[
                    "maximum_supported_modifiable_player_top_5_percent_share"
                ]
            )
        ),
        "maximum_ring16_state_top_1_percent_share": (
            ring_state_concentration["top_1_percent_share"]
            <= float(concentration_spec["maximum_ring16_state_top_1_percent_share"])
        ),
    }
    product_gates = {
        "four_a_state_level_estimability": four_a_passes,
        "minimum_supported_guided_first_downgrade_upper_bound": (
            mechanism_availability["upper_95"]
            >= float(product_spec["minimum_signable_absolute_effect"])
        ),
        "minimum_corrected_action_effect_upper_bound": action_effect_gate,
    }
    coverage_gates = {
        "minimum_oracle_coverage": (
            coverage_fraction >= float(coverage_spec["minimum_decision_coverage"])
        ),
        "maximum_oracle_abstention_fraction": (
            1.0 - coverage_fraction
            <= float(coverage_spec["maximum_abstention_fraction"])
        ),
    }
    gate_results = {
        "independent_support": support_gates,
        "modifiable_reachability": reach_gates,
        "concentration": concentration_gates,
        "estimability": four_a_gates,
        "product_effect": product_gates,
        "oracle_coverage": coverage_gates,
    }
    all_gates_pass = all(
        bool(value)
        for dimension in gate_results.values()
        for value in dimension.values()
    )
    decision = "not_rejected_at_f0_h0" if all_gates_pass else "stop_condition_triggered"

    train_partition = membership["partitions"]["train"]
    train_set = set(train_partition["session_ids"])
    full_train_records = [
        by_session[session] for session in sorted(train_set) if session in by_session
    ]
    if len(full_train_records) != EXPECTED_PARTITION_COUNTS["train"]:
        raise TrainScreenError("full train membership is absent from F0-D0")
    strict_outcomes = Counter(
        record.recorded_outcome
        for record in full_train_records
        if record.outcome_eligible
    )
    strict_outcome_players = {
        player
        for record in full_train_records
        if record.outcome_eligible
        for player in record.player_keys
    }
    if any(key not in {"W", "B", "D"} for key in strict_outcomes):
        raise TrainScreenError("strict train outcome value is invalid")

    access_audit = {
        "statistics_partitions": ["train"],
        "successful_accesses": {
            f"{partition}:{kind}": count
            for (partition, kind), count in sorted(access.successful_accesses.items())
        },
        "denied_attempts": {
            f"{partition}:{kind}": count
            for (partition, kind), count in sorted(access.denied_attempts.items())
        },
        "selection_raw_games_or_decisions_or_features_read": 0,
        "confirmation_raw_games_or_decisions_or_features_read": 0,
        "final_test_raw_games_or_decisions_or_features_read": 0,
        "source_pool_2eb04f54_records_read_or_consumed": 0,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "screen_id": plan["screen_id"],
        "status": "completed_read_only_train_rejection_screen",
        "decision": decision,
        "decision_semantics": {
            "rejection_only": True,
            "not_rejected_is_not_approval": True,
            "cannot_approve_or_start_e0_or_any_later_gate": True,
        },
        "claim_boundary": {
            "source_domain": "observed PlayOK-like source only",
            "state_safety": "positional-only",
            "safe_set": "A_pos",
            "a_allow_claim": False,
            "full_history_safety_claim": False,
            "state_novelty_claim": False,
            "product_ui_or_new_population_generalization": False,
        },
        "lineage": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha256,
            "f0d0_corpus_identity": EXPECTED_CORPUS_IDENTITY,
            "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
            "f0d0_manifest_file_sha256": boundary.file_sha256,
            "b2_membership_identity": membership["membership_identity"],
            "b2_membership_file_sha256": membership_file_sha256,
            "b2_characterization_identity": EXPECTED_CHARACTERIZATION_IDENTITY,
            "b2_characterization_file_sha256": characterization_file_sha256,
            "frozen_fallback_identity": EXPECTED_FALLBACK_IDENTITY,
            "train_analysis_membership_identity": canonical_sha256(train_sessions),
        },
        "preregistered_thresholds": plan["thresholds"],
        "sample": {
            "frozen_fallback_games": 10_000,
            "membership_composition": sample_composition,
            "analysis_games": len(records),
            "analysis_partition": "train",
            "resampled": False,
            "analysis_decisions": oracle_decisions,
            "analysis_session_ids_identity": canonical_sha256(train_sessions),
        },
        "input_files": {
            "raw_games": _input_file_rows(records, boundary),
            "raw_game_files": len(records),
            "raw_game_inventory_identity": canonical_sha256(
                _input_file_rows(records, boundary)
            ),
            "malom_manifest": {
                "relative_path": Path(malom_manifest_path).as_posix(),
                "sha256": sha256_file(malom_manifest_path),
                "snapshot": snapshot,
            },
            "ruleset": {
                "relative_path": Path(ruleset_path).as_posix(),
                "sha256": sha256_file(ruleset_path),
            },
        },
        "access_audit": access_audit,
        "bases": {
            "behavior_global_inherited": {
                "games": 92_226,
                "logical_plies": 4_394_220,
                "player_keys": 4_994,
            },
            "behavior_train_full_membership": {
                "games": len(full_train_records),
                "logical_plies": sum(
                    record.move_count for record in full_train_records
                ),
                "player_keys": len(
                    {
                        player
                        for record in full_train_records
                        for player in record.player_keys
                    }
                ),
                "content_opened": False,
            },
            "behavior_train_frozen_sample": {
                "games": len(records),
                "logical_plies": oracle_decisions,
                "player_keys": len(player_decisions),
            },
            "oracle_positional_sample": {
                "decisions": oracle_decisions,
                "covered_decisions": covered_decisions,
                "coverage_fraction": coverage_fraction,
                "abstained_decisions": oracle_decisions - covered_decisions,
                "abstention_reasons": _serialize_counter(oracle_abstentions),
                "queries": oracle_queries,
            },
            "strict_product_train_full_membership": {
                "games": sum(strict_outcomes.values()),
                "player_keys": len(strict_outcome_players),
            },
        },
        "dimensions": {
            "independent_support": {
                "exact_positional_state": exact_summary,
                "ring16_positional_state_class": ring_summary,
                "exact_state_definition": "exact predecessor BoardState FEN",
                "ring16_definition": "repository D4 x abstract ring swap",
            },
            "modifiable_state_reachability": {
                "a_pos_cardinality_counts": _serialize_counter(a_pos_counts),
                "a_pos_cardinality_greater_than_one": modifiable_interval,
                "support_qualified_ring16_and_modifiable": (
                    supported_modifiable_interval
                ),
                "support_qualified_modifiable_game_reach": (
                    supported_modifiable_game_reach
                ),
                "phase_color_tier_counts": [
                    {
                        "phase": key[0],
                        "color": key[1],
                        "tier": key[2],
                        "choice_class": key[3],
                        "decisions": count,
                    }
                    for key, count in sorted(strata.items())
                ],
                "positional_only": True,
            },
            "concentration": {
                "players": player_concentration,
                "games": game_concentration,
                "exact_positional_states": exact_state_concentration,
                "ring16_positional_state_classes": ring_state_concentration,
                "support_qualified_modifiable_players": (
                    supported_player_concentration
                ),
            },
            "product_effect_upper_bound": {
                "four_a_estimability": {
                    "decision": (
                        "state_level_empirically_estimable"
                        if four_a_passes
                        else "state_level_not_empirically_estimable"
                    ),
                    "exact_positional_state": exact_estimability_summary,
                    "ring16_positional_state_class": ring_estimability_summary,
                    "gates": four_a_gates,
                    "eligible_exposure_definition": (
                        "support-qualified modifiable A_pos choice before the "
                        "game's first positional tier loss with an observed "
                        "immediate opponent reply"
                    ),
                    "eligible_exposures": exposure_count,
                    "support_abstained_exposures": exposure_support_abstentions,
                },
                "four_b_state_conditioned_effect": {
                    "status": (
                        "estimated_with_crossfit_and_shrinkage"
                        if action_effects is not None
                        else "skipped_because_four_a_failed"
                    ),
                    "event_definition": (
                        "immediate opponent reply is the game's first "
                        "positional tier loss"
                    ),
                    "uncorrected_and_corrected_by_transition": action_effects,
                    "winner_curse_correction": (
                        "whole-game deterministic two-fold action selection, "
                        "Jeffreys shrinkage, and ring16-state cluster bootstrap"
                    ),
                },
                "mechanism_scope": {
                    "fully_oracle_covered_games": fully_covered_games,
                    "games_with_first_theory_downgrade": first_downgrade_games,
                    "first_downgrade_counts": {
                        event: first_downgrades[event] for event in DOWNGRADE_TYPES
                    },
                    "modifiable_predecessor_counts": {
                        event: guided_first_downgrades[event]
                        for event in DOWNGRADE_TYPES
                    },
                    "support_qualified_modifiable_predecessor_counts": {
                        event: supported_guided_first_downgrades[event]
                        for event in DOWNGRADE_TYPES
                    },
                    "support_qualified_guided_fraction_among_fully_covered_games": (
                        mechanism_availability_observed
                    ),
                    "conservative_support_qualified_guided_fraction_all_sample_games": (
                        mechanism_availability
                    ),
                    "coverage_abstentions_are_not_counted_as_successes": True,
                    "maximum_per_game_score_swing_for_upper_bound": 1.0,
                },
                "product_scope": {
                    "denominator": "full B2 train strict-outcome subset",
                    "strict_outcome_games": sum(strict_outcomes.values()),
                    "independent_player_keys": len(strict_outcome_players),
                    "recorded_and_independently_replayed_outcomes": {
                        outcome: strict_outcomes[outcome] for outcome in ("W", "B", "D")
                    },
                    "not_extrapolated_to_mechanism_sample": True,
                },
            },
        },
        "secondary_difficulty": {
            "within_tier_complete_comparator_regret_decisions": (
                within_tier_regret_count
            ),
            "preserving_decisions": preserving_count,
            "rate_among_preserving": (
                within_tier_regret_count / preserving_count
                if preserving_count
                else None
            ),
            "mean_normalized_regret_among_preserving": (
                normalized_regret_sum / preserving_count if preserving_count else None
            ),
            "not_combined_with_wdl_loss": True,
        },
        "gate_results": gate_results,
        "known_biases": {
            "history_attrition_nonrandom": {
                "excluded_games": 1_751,
                "excluded_draws": 35,
                "retained_history_games": 92_789,
                "retained_draws": 26_157,
            },
            "unverifiable_terminal_basis_games": 54_923,
            "missing_conditions": [
                "UI orientation",
                "time control",
                "exact source rules variant",
                "explicit import batch",
                "upstream source-file identity",
            ],
            "transport": (
                "no inference to product UI, other time controls, or new people"
            ),
            "state_overlap": (
                "diagnostic only; game and player membership own contamination"
            ),
        },
        "prohibited_operations_observed": {
            "games_started": 0,
            "search_batches_started": 0,
            "models_loaded": 0,
            "training_updates": 0,
            "database_writes_or_rebuilds": 0,
            "protected_partition_content_reads": 0,
            "source_pool_records_read_or_consumed": 0,
        },
    }
    if (
        access.denied_attempts
        or membership_file_sha256 != EXPECTED_MEMBERSHIP_FILE_SHA256
        or boundary.file_sha256 != EXPECTED_F0D0_FILE_SHA256
    ):
        raise TrainScreenError("protected access or lineage invariant differs")
    return result


def write_screen_result(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Exclusively create one sealed machine-readable screen result."""
    return write_sealed_json(
        path,
        result,
        identity_field="result_identity",
    )


__all__ = [
    "EXPECTED_SAMPLE_COMPOSITION",
    "PLAN_SCHEMA",
    "ProtectedPartitionAccessError",
    "RESULT_SCHEMA",
    "TrainOnlyAccess",
    "TrainScreenError",
    "clustered_proportion",
    "derive_train_sample",
    "estimate_action_effects",
    "estimability_summary",
    "exact_state_key",
    "label_oracle_decision",
    "load_characterization_identity",
    "load_screen_plan",
    "load_screen_result",
    "ring16_canonical_transition",
    "ring16_state_key",
    "run_train_screen",
    "support_summary",
    "verify_implementation_artifacts",
    "write_screen_result",
]
