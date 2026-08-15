"""Player-isolated design support for featureized human deviation research.

This module defines a new question after the immutable F0-H0 rejection.  It
does not re-estimate exact-state or ring16 human frequencies.  The observable
unit is a complete human choice set plus the chosen action.  Human-facing
features are deliberately low-dimensional and never include Malom values,
state identities, player identities, or protected-partition information.

Malom is used only after the choice has been reconstructed, to label the
positional ``A_pos`` set and positional tier loss.  All such labels are
positional-only; this module never constructs or claims ``A_allow``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.malom_db import MalomDB, OracleMoveValue
from game.board import ADJACENCY, MILLS, BoardState
from game.rules import get_game_phase
from learned_ai.evaluation.human_f0h0_b2_train_screen import (
    OracleCoverageAbstention,
    _move_matches,
    _oracle_inventory_positional,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
    F0H0Error,
    ReplayedDecision,
    _read_raw_game,
    canonical_sha256,
    concentration,
    replay_game,
    sha256_file,
    wilson_interval,
)


PLAN_SCHEMA = "nmm.human-feature-deviation-plan.v1"
SPLIT_SCHEMA = "nmm.human-feature-deviation-train-split.v1"
EXPLORATION_SCHEMA = "nmm.human-feature-deviation-exploration.v1"

EXPECTED_F0D0_CORPUS_IDENTITY = (
    "4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29"
)
EXPECTED_F0D0_MANIFEST_IDENTITY = (
    "bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7"
)
EXPECTED_F0D0_FILE_SHA256 = (
    "0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6"
)
EXPECTED_B2_MEMBERSHIP_IDENTITY = (
    "06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b"
)
EXPECTED_B2_MEMBERSHIP_FILE_SHA256 = (
    "06c3be92c87927d506dc36eb908aec3064220f4ead2ebb3b5ff3dfb7bf5032cb"
)
EXPECTED_B2_COUNTS = {
    "train": 36_949,
    "selection": 887,
    "confirmation": 386,
    "final-test": 847,
}
OFFICIAL_PROTECTED = frozenset({"selection", "confirmation", "final-test"})
RESEARCH_PARTITIONS = (
    "research-exploration",
    "research-confirmation",
    "cross-player-discard",
)
WDL_RANK = {"L": 0, "D": 1, "W": 2}
TRANSITIONS = ("W->D", "W->L", "D->L")
PHASE_NAMES = {"place": "placement", "move": "movement", "fly": "flying"}

# Ten named, actor-normalized, board-visible action features.  The first three
# are the frozen geometry control.  The remaining seven are the tactical
# heuristic panel.  State-only features are intentionally absent because they
# cancel inside a conditional choice set.
FEATURE_NAMES = (
    "source_degree",
    "destination_degree",
    "capture_degree",
    "closes_mill",
    "blocks_immediate_mill",
    "creates_double_mill",
    "new_own_potential_mills",
    "own_mobility_delta",
    "material_balance_after",
    "captured_opponent_threat_lines",
)
GEOMETRY_FEATURES = FEATURE_NAMES[:3]
TACTICAL_FEATURES = FEATURE_NAMES[3:]


class FeatureDeviationError(RuntimeError):
    """Raised when a frozen identity or required research input differs."""


class ProtectedResearchAccessError(FeatureDeviationError):
    """Raised before non-exploration content can be opened or derived."""


def _load_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureDeviationError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise FeatureDeviationError(f"JSON root is not an object: {source}")
    return value, raw


def _load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    value, raw = _load_json(path)
    if value.get("schema_version") != schema:
        raise FeatureDeviationError(f"schema differs: {Path(path)}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise FeatureDeviationError(f"identity is absent: {Path(path)}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise FeatureDeviationError(f"identity differs: {Path(path)}")
    return value, hashlib.sha256(raw).hexdigest()


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate the feature-deviation preregistration."""
    plan, file_sha = _load_sealed(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    boundary = plan.get("input_boundary")
    expected = {
        "f0d0_corpus_identity": EXPECTED_F0D0_CORPUS_IDENTITY,
        "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "f0d0_manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
        "b2_membership_identity": EXPECTED_B2_MEMBERSHIP_IDENTITY,
        "b2_membership_file_sha256": EXPECTED_B2_MEMBERSHIP_FILE_SHA256,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) != value for key, value in expected.items()
    ):
        raise FeatureDeviationError("feature-deviation input boundary differs")
    split = plan.get("train_internal_split")
    if (
        not isinstance(split, Mapping)
        or split.get("assignment_unit") != "source_domain_player_key"
        or split.get("confirmation_first_byte_values") != list(range(64))
        or split.get("cross_arm_game_policy") != "discard"
        or split.get("exploration_pilot_games") != 128
    ):
        raise FeatureDeviationError("feature-deviation split contract differs")
    feature_spec = plan.get("feature_dictionary")
    if (
        not isinstance(feature_spec, Mapping)
        or tuple(feature_spec.get("ordered_features", ())) != FEATURE_NAMES
        or tuple(feature_spec.get("geometry_control", ())) != GEOMETRY_FEATURES
        or tuple(feature_spec.get("tactical_panel", ())) != TACTICAL_FEATURES
    ):
        raise FeatureDeviationError("feature dictionary differs")
    if plan.get("safe_set") != "A_pos" or plan.get("a_allow_claim") is not False:
        raise FeatureDeviationError("positional claim boundary differs")
    if plan.get("statistics_partitions") != ["train"]:
        raise FeatureDeviationError("statistics partition differs")
    if plan.get("confirmation_execution_authorized") is not False:
        raise FeatureDeviationError("confirmation execution boundary differs")
    return plan, file_sha


def verify_implementation_artifacts(
    repository_root: str | Path,
    plan: Mapping[str, Any],
) -> None:
    root = Path(repository_root)
    rows = plan.get("implementation_artifacts")
    if not isinstance(rows, list) or not rows:
        raise FeatureDeviationError("implementation artifact inventory is absent")
    for row in rows:
        if not isinstance(row, Mapping):
            raise FeatureDeviationError("implementation artifact row is invalid")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise FeatureDeviationError("implementation artifact identity is invalid")
        if sha256_file(root / relative) != expected:
            raise FeatureDeviationError(f"implementation artifact differs: {relative}")


def _player_arm(player_key: str, *, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}\0{player_key}".encode("utf-8")).digest()
    return "research-confirmation" if digest[0] < 64 else "research-exploration"


def _rank_sessions(session_ids: Sequence[str], *, seed: str) -> list[str]:
    return sorted(
        (str(value) for value in session_ids),
        key=lambda session: (
            hashlib.sha256(f"{seed}\0{session}".encode("utf-8")).digest(),
            session,
        ),
    )


def build_train_internal_split(
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic player split from manifest structure only."""
    if boundary.file_sha256 != EXPECTED_F0D0_FILE_SHA256:
        raise FeatureDeviationError("F0-D0 boundary file identity differs")
    if (
        official_membership.get("membership_identity")
        != EXPECTED_B2_MEMBERSHIP_IDENTITY
    ):
        raise FeatureDeviationError("official B2 membership identity differs")
    partitions = official_membership.get("partitions")
    if not isinstance(partitions, Mapping):
        raise FeatureDeviationError("official B2 partitions are absent")
    train_row = partitions.get("train")
    train_ids = train_row.get("session_ids") if isinstance(train_row, Mapping) else None
    if not isinstance(train_ids, list) or len(train_ids) != EXPECTED_B2_COUNTS["train"]:
        raise FeatureDeviationError("official B2 train membership differs")
    if len(set(train_ids)) != len(train_ids):
        raise FeatureDeviationError("official B2 train membership has duplicates")

    record_by_id = {record.session_id: record for record in boundary.records}
    try:
        train_records = [record_by_id[str(session)] for session in train_ids]
    except KeyError as exc:
        raise FeatureDeviationError("B2 train session is absent from F0-D0") from exc
    if any(not record.behavior_eligible for record in train_records):
        raise FeatureDeviationError("B2 train contains behavior-ineligible games")

    split_spec = plan["train_internal_split"]
    seed = str(split_spec["player_hash_seed"])
    pilot_seed = str(split_spec["exploration_pilot_hash_seed"])
    players = sorted({key for record in train_records for key in record.player_keys})
    arm_by_player = {player: _player_arm(player, seed=seed) for player in players}

    sessions: dict[str, list[str]] = {name: [] for name in RESEARCH_PARTITIONS}
    decisions: Counter[str] = Counter()
    for record in train_records:
        left = arm_by_player[record.player_keys[0]]
        right = arm_by_player[record.player_keys[1]]
        arm = left if left == right else "cross-player-discard"
        sessions[arm].append(record.session_id)
        decisions[arm] += record.move_count
    for values in sessions.values():
        values.sort()
    if sum(len(values) for values in sessions.values()) != len(train_records):
        raise FeatureDeviationError("research split does not cover B2 train")

    exploration_players = sorted(
        player for player, arm in arm_by_player.items() if arm == "research-exploration"
    )
    confirmation_players = sorted(
        player
        for player, arm in arm_by_player.items()
        if arm == "research-confirmation"
    )
    if set(exploration_players) & set(confirmation_players):
        raise FeatureDeviationError("research player sets overlap")

    pilot_count = int(split_spec["exploration_pilot_games"])
    ranked = _rank_sessions(sessions["research-exploration"], seed=pilot_seed)
    if len(ranked) < pilot_count:
        raise FeatureDeviationError("research exploration arm is smaller than pilot")
    pilot = sorted(ranked[:pilot_count])

    return {
        "schema_version": SPLIT_SCHEMA,
        "status": "frozen_before_feature_statistics",
        "plan_identity": plan["plan_identity"],
        "input_boundary": {
            "f0d0_corpus_identity": EXPECTED_F0D0_CORPUS_IDENTITY,
            "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
            "f0d0_manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
            "b2_membership_identity": EXPECTED_B2_MEMBERSHIP_IDENTITY,
            "b2_membership_file_sha256": EXPECTED_B2_MEMBERSHIP_FILE_SHA256,
        },
        "assignment_rule": {
            "unit": "source_domain_player_key",
            "hash": "SHA-256(seed NUL player_key)",
            "research_confirmation": "first digest byte in [0,63]",
            "research_exploration": "first digest byte in [64,255]",
            "cross_arm_game_policy": "discard",
            "player_hash_seed": seed,
        },
        "partitions": {
            name: {
                "games": len(sessions[name]),
                "logical_plies": int(decisions[name]),
                "session_ids_identity": canonical_sha256(sessions[name]),
                "session_ids": sessions[name],
            }
            for name in RESEARCH_PARTITIONS
        },
        "player_membership": {
            "research-exploration": {
                "players": len(exploration_players),
                "player_keys_identity": canonical_sha256(exploration_players),
                "player_keys": exploration_players,
            },
            "research-confirmation": {
                "players": len(confirmation_players),
                "player_keys_identity": canonical_sha256(confirmation_players),
                "player_keys": confirmation_players,
            },
            "pairwise_player_overlap": 0,
        },
        "exploration_pilot": {
            "selection": "lowest SHA-256(seed NUL session_id) ranks",
            "seed": pilot_seed,
            "games": len(pilot),
            "session_ids_identity": canonical_sha256(pilot),
            "session_ids": pilot,
        },
        "access_state": {
            "built_from_f0d0_and_b2_membership_only": True,
            "raw_game_files_opened": 0,
            "research_confirmation_content_reads": 0,
            "selection_content_reads": 0,
            "confirmation_content_reads": 0,
            "final_test_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
    }


def load_research_split(path: str | Path) -> tuple[dict[str, Any], str]:
    split, file_sha = _load_sealed(
        path,
        schema=SPLIT_SCHEMA,
        identity_field="split_identity",
    )
    if split.get("status") != "frozen_before_feature_statistics":
        raise FeatureDeviationError("research split status differs")
    if split.get("input_boundary", {}).get("b2_membership_identity") != (
        EXPECTED_B2_MEMBERSHIP_IDENTITY
    ):
        raise FeatureDeviationError("research split B2 lineage differs")
    partitions = split.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(
        RESEARCH_PARTITIONS
    ):
        raise FeatureDeviationError("research split partitions differ")
    seen: set[str] = set()
    for name in RESEARCH_PARTITIONS:
        row = partitions[name]
        values = row.get("session_ids") if isinstance(row, Mapping) else None
        if (
            not isinstance(values, list)
            or values != sorted(values)
            or len(values) != len(set(values))
            or row.get("games") != len(values)
            or row.get("session_ids_identity") != canonical_sha256(values)
        ):
            raise FeatureDeviationError(f"research split {name} differs")
        if seen & set(values):
            raise FeatureDeviationError("research split session overlap")
        seen.update(values)
    pilot = split.get("exploration_pilot")
    pilot_ids = pilot.get("session_ids") if isinstance(pilot, Mapping) else None
    exploration = set(partitions["research-exploration"]["session_ids"])
    if (
        not isinstance(pilot_ids, list)
        or pilot_ids != sorted(pilot_ids)
        or len(pilot_ids) != 128
        or not set(pilot_ids).issubset(exploration)
        or pilot.get("session_ids_identity") != canonical_sha256(pilot_ids)
    ):
        raise FeatureDeviationError("research exploration pilot differs")
    player_membership = split.get("player_membership")
    if not isinstance(player_membership, Mapping):
        raise FeatureDeviationError("research player membership is absent")
    left = set(player_membership["research-exploration"]["player_keys"])
    right = set(player_membership["research-confirmation"]["player_keys"])
    if left & right or player_membership.get("pairwise_player_overlap") != 0:
        raise FeatureDeviationError("research player sets overlap")
    return split, file_sha


@dataclass
class ExplorationOnlyAccess:
    """Fail closed before any non-pilot content read or derivation."""

    official_partition_by_session: Mapping[str, str]
    research_partition_by_session: Mapping[str, str]
    pilot_sessions: frozenset[str]
    successful_accesses: Counter[tuple[str, str]] = field(default_factory=Counter)
    denied_attempts: Counter[tuple[str, str]] = field(default_factory=Counter)

    @classmethod
    def from_memberships(
        cls,
        official_membership: Mapping[str, Any],
        research_split: Mapping[str, Any],
    ) -> "ExplorationOnlyAccess":
        official: dict[str, str] = {}
        for name in EXPECTED_B2_COUNTS:
            row = official_membership["partitions"][name]
            for session in row["session_ids"]:
                if session in official:
                    raise FeatureDeviationError("official session partition overlap")
                official[str(session)] = name
        research: dict[str, str] = {}
        for name in RESEARCH_PARTITIONS:
            for session in research_split["partitions"][name]["session_ids"]:
                if session in research:
                    raise FeatureDeviationError("research session partition overlap")
                research[str(session)] = name
        pilot = frozenset(
            str(value) for value in research_split["exploration_pilot"]["session_ids"]
        )
        if not pilot or any(
            official.get(session) != "train"
            or research.get(session) != "research-exploration"
            for session in pilot
        ):
            raise FeatureDeviationError("pilot is not train exploration only")
        return cls(official, research, pilot)

    def assert_pilot(self, session_id: str, *, access_kind: str) -> None:
        official = self.official_partition_by_session.get(
            session_id, "outside-frozen-membership"
        )
        research = self.research_partition_by_session.get(session_id, "not-train")
        if (
            official != "train"
            or research != "research-exploration"
            or session_id not in self.pilot_sessions
        ):
            label = f"{official}:{research}"
            self.denied_attempts[(label, access_kind)] += 1
            raise ProtectedResearchAccessError(
                f"exploration-only access denied {access_kind} for {label} "
                f"session {session_id}"
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
        self.assert_pilot(record.session_id, access_kind="raw_game")
        selected_reader = _read_raw_game if reader is None else reader
        try:
            value = selected_reader(repository_root, record, boundary)
        except F0H0Error as exc:
            raise FeatureDeviationError(str(exc)) from exc
        self.successful_accesses[("research-exploration", "raw_game")] += 1
        return value

    def load_decisions(
        self,
        repository_root: Path,
        record: CorpusRecord,
        boundary: F0D0Boundary,
    ) -> list[ReplayedDecision]:
        self.assert_pilot(record.session_id, access_kind="decisions")
        raw = self.read_raw_game(repository_root, record, boundary)
        try:
            decisions = replay_game(raw, record)
        except F0H0Error as exc:
            raise FeatureDeviationError(str(exc)) from exc
        self.successful_accesses[("research-exploration", "decisions")] += 1
        return decisions

    def derive_features(self, session_id: str, producer: Callable[[], Any]) -> Any:
        self.assert_pilot(session_id, access_kind="derived_features")
        value = producer()
        self.successful_accesses[("research-exploration", "derived_features")] += 1
        return value


def _piece_count(board: BoardState, color: str) -> int:
    return sum(value == color for value in board.positions.values())


def _potential_mills(board: BoardState, color: str) -> int:
    return sum(
        1
        for line in MILLS
        if [board.positions[pos] for pos in line].count(color) == 2
        and [board.positions[pos] for pos in line].count("") == 1
    )


def _mobility(board: BoardState, color: str) -> int:
    if board.phase == "place":
        return len(board.legal_placements(color))
    return len(board.legal_moves(color))


def _blocks_immediate_mill(
    board: BoardState,
    move: Mapping[str, Any],
    actor: str,
) -> bool:
    destination = move.get("to")
    if destination is None:
        return False
    opponent = "B" if actor == "W" else "W"
    for line in MILLS:
        if destination not in line:
            continue
        values = [board.positions[pos] for pos in line]
        if values.count(opponent) == 2 and values.count("") == 1:
            return True
    return False


def _captured_threat_lines(
    board: BoardState,
    capture: str | None,
    opponent: str,
) -> int:
    if capture is None:
        return 0
    count = 0
    for line in MILLS:
        if capture not in line:
            continue
        values = [board.positions[pos] for pos in line]
        if values.count(opponent) >= 2:
            count += 1
    return count


def action_feature_scores(
    board: BoardState,
    move: Mapping[str, Any],
) -> dict[str, float]:
    """Return the ten frozen human-visible action features."""
    actor = board.turn
    opponent = "B" if actor == "W" else "W"
    normalized = {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }
    after = board.apply_move(normalized)
    destination = normalized["to"]
    source = normalized["from"]
    capture = normalized["capture"]
    closes_mill = bool(destination and after.is_mill(str(destination), actor))
    double_mill = False
    if destination is not None:
        double_mill = (
            sum(
                1
                for line in MILLS
                if destination in line
                and all(after.positions[pos] == actor for pos in line)
            )
            >= 2
        )
    values = {
        "source_degree": len(ADJACENCY.get(source, ())) / 4.0 if source else 0.0,
        "destination_degree": (
            len(ADJACENCY.get(destination, ())) / 4.0 if destination else 0.0
        ),
        "capture_degree": (len(ADJACENCY.get(capture, ())) / 4.0 if capture else 0.0),
        "closes_mill": float(closes_mill),
        "blocks_immediate_mill": float(
            _blocks_immediate_mill(board, normalized, actor)
        ),
        "creates_double_mill": float(double_mill),
        "new_own_potential_mills": float(
            _potential_mills(after, actor) - _potential_mills(board, actor)
        ),
        "own_mobility_delta": float(
            (_mobility(after, actor) - _mobility(board, actor)) / 24.0
        ),
        "material_balance_after": float(
            (_piece_count(after, actor) - _piece_count(after, opponent)) / 9.0
        ),
        "captured_opponent_threat_lines": float(
            _captured_threat_lines(board, capture, opponent) / 2.0
        ),
    }
    if tuple(values) != FEATURE_NAMES or any(
        not math.isfinite(value) for value in values.values()
    ):
        raise FeatureDeviationError("human-visible action feature contract differs")
    return values


@dataclass
class FeatureOpportunity:
    varied: int = 0
    possible_conflict: int = 0
    strict_conflict: int = 0
    mixed_maximum: int = 0
    followed_maximum: int = 0
    unsafe_follow: int = 0
    strict_conflict_follow: int = 0
    varied_players: set[str] = field(default_factory=set)
    conflict_players: set[str] = field(default_factory=set)
    unsafe_follow_players: set[str] = field(default_factory=set)
    varied_games: set[str] = field(default_factory=set)
    conflict_games: set[str] = field(default_factory=set)
    unsafe_follow_games: set[str] = field(default_factory=set)
    by_phase: Counter[str] = field(default_factory=Counter)
    by_tier: Counter[str] = field(default_factory=Counter)
    by_color: Counter[str] = field(default_factory=Counter)
    player_varied: Counter[str] = field(default_factory=Counter)
    player_unsafe_follow: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        *,
        scores: Sequence[float],
        safe_indices: set[int],
        chosen_index: int,
        player: str,
        game: str,
        phase: str,
        tier: str,
        color: str,
    ) -> None:
        if not scores:
            raise FeatureDeviationError("feature choice set is empty")
        maximum = max(scores)
        minimum = min(scores)
        if math.isclose(maximum, minimum, rel_tol=0.0, abs_tol=1e-12):
            return
        maximizers = {
            index
            for index, value in enumerate(scores)
            if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
        }
        unsafe_maximizers = maximizers - safe_indices
        safe_maximizers = maximizers & safe_indices
        self.varied += 1
        self.varied_players.add(player)
        self.varied_games.add(game)
        self.player_varied[player] += 1
        if unsafe_maximizers:
            self.possible_conflict += 1
            self.conflict_players.add(player)
            self.conflict_games.add(game)
            self.by_phase[phase] += 1
            self.by_tier[tier] += 1
            self.by_color[color] += 1
        if unsafe_maximizers and not safe_maximizers:
            self.strict_conflict += 1
        if unsafe_maximizers and safe_maximizers:
            self.mixed_maximum += 1
        if chosen_index in maximizers:
            self.followed_maximum += 1
        if chosen_index in unsafe_maximizers:
            self.unsafe_follow += 1
            self.unsafe_follow_players.add(player)
            self.unsafe_follow_games.add(game)
            self.player_unsafe_follow[player] += 1
            if not safe_maximizers:
                self.strict_conflict_follow += 1


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {
            "point": None,
            "successes": successes,
            "observations": total,
            "lower_95": None,
            "upper_95": None,
            "zero_events_not_smoothed": successes == 0,
        }
    if successes < 0 or successes > total:
        raise FeatureDeviationError("rate counts are invalid")
    interval = wilson_interval(successes, total)
    return {
        "point": successes / total,
        "successes": successes,
        "observations": total,
        "lower_95": interval["lower_95"],
        "upper_95": interval["upper_95"],
        "zero_events_not_smoothed": successes == 0,
    }


def _average_unique_player_rate(
    successes: Mapping[str, int],
    totals: Mapping[str, int],
) -> dict[str, Any]:
    players = sorted(player for player, total in totals.items() if total > 0)
    if not players:
        return {"point": None, "players": 0}
    rates = [successes.get(player, 0) / totals[player] for player in players]
    return {"point": sum(rates) / len(rates), "players": len(players)}


def _serialize_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _feature_summary(value: FeatureOpportunity, decisions: int) -> dict[str, Any]:
    return {
        "action_score_varies": _rate(value.varied, decisions),
        "possible_conflict_given_variation": _rate(
            value.possible_conflict, value.varied
        ),
        "strict_conflict_given_variation": _rate(value.strict_conflict, value.varied),
        "mixed_safe_unsafe_maximum_given_variation": _rate(
            value.mixed_maximum, value.varied
        ),
        "observed_follow_rate_given_variation": _rate(
            value.followed_maximum, value.varied
        ),
        "direct_unsafe_heuristic_follow_given_variation": _rate(
            value.unsafe_follow, value.varied
        ),
        "strict_conflict_follow_given_strict_conflict": _rate(
            value.strict_conflict_follow, value.strict_conflict
        ),
        "average_unique_player_unsafe_follow_given_variation": (
            _average_unique_player_rate(
                value.player_unsafe_follow,
                value.player_varied,
            )
        ),
        "independent_support": {
            "varied_players": len(value.varied_players),
            "conflict_players": len(value.conflict_players),
            "unsafe_follow_players": len(value.unsafe_follow_players),
            "varied_games": len(value.varied_games),
            "conflict_games": len(value.conflict_games),
            "unsafe_follow_games": len(value.unsafe_follow_games),
        },
        "possible_conflict_counts_by_phase": _serialize_counter(value.by_phase),
        "possible_conflict_counts_by_tier": _serialize_counter(value.by_tier),
        "possible_conflict_counts_by_color": _serialize_counter(value.by_color),
        "multiple_recommendations_semantics": (
            "strict conflict means every tied score maximum lies outside A_pos; "
            "mixed maxima are reported separately"
        ),
    }


def run_exploration(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    research_split: Mapping[str, Any],
    plan: Mapping[str, Any],
    database: MalomDB,
    malom_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only the frozen 128-game exploration pilot."""
    root = Path(repository_root)
    if research_split.get("plan_identity") != plan.get("plan_identity"):
        raise FeatureDeviationError("research split does not bind the plan")
    if malom_snapshot.get("trust_level") != "sector-corrected-v1":
        raise FeatureDeviationError("Malom trust level differs")
    if not database.is_available():
        raise FeatureDeviationError("required Malom database is unavailable")

    access = ExplorationOnlyAccess.from_memberships(
        official_membership,
        research_split,
    )
    record_by_id = {record.session_id: record for record in boundary.records}
    pilot_ids = research_split["exploration_pilot"]["session_ids"]
    try:
        records = [record_by_id[session] for session in pilot_ids]
    except KeyError as exc:
        raise FeatureDeviationError("pilot session is absent from F0-D0") from exc

    opportunities = {name: FeatureOpportunity() for name in FEATURE_NAMES}
    phase_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    player_decisions: Counter[str] = Counter()
    games_with_decisions: set[str] = set()
    a_pos_modifiable = 0
    covered = 0
    abstained = 0
    abstention_reasons: Counter[str] = Counter()
    query_count = 0
    expected_decisions = 0
    started = time.perf_counter()

    for record in records:
        decisions = access.load_decisions(root, record, boundary)
        expected_decisions += record.move_count
        if len(decisions) != record.move_count:
            raise FeatureDeviationError("pilot replay decision count differs")
        for decision in decisions:
            try:
                parent_tier, inventory, queries = _oracle_inventory_positional(
                    decision.board,
                    database,
                )
            except OracleCoverageAbstention as exc:
                abstained += 1
                query_count += exc.query_count
                abstention_reasons[exc.reason] += 1
                continue
            query_count += queries
            chosen_rows = [
                index
                for index, (move, _value) in enumerate(inventory)
                if _move_matches(move, decision.move)
            ]
            if len(chosen_rows) != 1:
                raise FeatureDeviationError("observed action is absent from inventory")
            chosen_index = chosen_rows[0]
            safe_indices = {
                index
                for index, (_move, value) in enumerate(inventory)
                if value.outcome == parent_tier
            }
            if not safe_indices:
                raise FeatureDeviationError("A_pos is empty")
            if len(safe_indices) > 1:
                a_pos_modifiable += 1
            chosen_value: OracleMoveValue = inventory[chosen_index][1]
            if WDL_RANK[chosen_value.outcome] < WDL_RANK[parent_tier]:
                transition = f"{parent_tier}->{chosen_value.outcome}"
                if transition not in TRANSITIONS:
                    raise FeatureDeviationError("unexpected positional transition")
                transition_counts[transition] += 1
            raw_phase = get_game_phase(decision.board, decision.board.turn)
            if raw_phase not in PHASE_NAMES:
                raise FeatureDeviationError("pilot phase is invalid")
            phase = PHASE_NAMES[raw_phase]
            action_features = [
                action_feature_scores(decision.board, move)
                for move, _value in inventory
            ]
            for name in FEATURE_NAMES:
                opportunities[name].observe(
                    scores=[row[name] for row in action_features],
                    safe_indices=safe_indices,
                    chosen_index=chosen_index,
                    player=decision.actor_player_key,
                    game=decision.game_id,
                    phase=phase,
                    tier=parent_tier,
                    color=decision.board.turn,
                )
            covered += 1
            phase_counts[phase] += 1
            tier_counts[parent_tier] += 1
            color_counts[decision.board.turn] += 1
            player_decisions[decision.actor_player_key] += 1
            games_with_decisions.add(decision.game_id)

    elapsed = time.perf_counter() - started
    if covered + abstained != expected_decisions:
        raise FeatureDeviationError("pilot oracle accounting differs")
    access_audit = {
        "successful_accesses": {
            f"{partition}:{kind}": int(count)
            for (partition, kind), count in sorted(access.successful_accesses.items())
        },
        "denied_attempts": {
            f"{partition}:{kind}": int(count)
            for (partition, kind), count in sorted(access.denied_attempts.items())
        },
        "research_confirmation_content_reads": 0,
        "selection_content_reads": 0,
        "confirmation_content_reads": 0,
        "final_test_content_reads": 0,
        "source_pool_2eb04f54_records_read_or_consumed": 0,
        "human_db_reads": 0,
        "database_writes": 0,
        "games_or_search_batches": 0,
        "model_loads_or_training_updates": 0,
    }
    if access.denied_attempts:
        raise FeatureDeviationError("protected access attempt occurred")
    return {
        "schema_version": EXPLORATION_SCHEMA,
        "status": "exploration_only_no_confirmatory_decision",
        "plan_identity": plan["plan_identity"],
        "split_identity": research_split["split_identity"],
        "claim_boundary": {
            "new_question_not_f0_h0_replay": True,
            "f0_h0_stop_condition_remains_effective": True,
            "safe_set": "A_pos",
            "state_safety": "positional-only",
            "a_allow_claim": False,
            "causal_inducement_claim": False,
            "product_effect_claim": False,
            "source_domain": "observed PlayOK-like source only",
        },
        "input_identities": {
            "f0d0_corpus_identity": EXPECTED_F0D0_CORPUS_IDENTITY,
            "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
            "b2_membership_identity": EXPECTED_B2_MEMBERSHIP_IDENTITY,
            "malom": dict(malom_snapshot),
        },
        "sample": {
            "partition": "research-exploration within official B2 train",
            "games": len(records),
            "games_with_covered_decisions": len(games_with_decisions),
            "session_ids_identity": research_split["exploration_pilot"][
                "session_ids_identity"
            ],
            "expected_decisions": expected_decisions,
            "covered_decisions": covered,
            "abstained_decisions": abstained,
            "independent_players": len(player_decisions),
            "player_decision_concentration": concentration(
                list(player_decisions.values())
            ),
        },
        "oracle": {
            "queries": query_count,
            "elapsed_seconds": elapsed,
            "queries_per_second": query_count / elapsed if elapsed > 0 else None,
            "coverage": _rate(covered, expected_decisions),
            "abstention_reasons": _serialize_counter(abstention_reasons),
        },
        "positional_labels": {
            "a_pos_cardinality_greater_than_one": _rate(a_pos_modifiable, covered),
            "tier_counts": _serialize_counter(tier_counts),
            "phase_counts": _serialize_counter(phase_counts),
            "color_counts": _serialize_counter(color_counts),
            "chosen_tier_loss_counts": {
                transition: int(transition_counts[transition])
                for transition in TRANSITIONS
            },
            "zero_events_not_smoothed": True,
        },
        "heuristic_opportunity_screen": {
            name: _feature_summary(opportunities[name], covered)
            for name in FEATURE_NAMES
        },
        "interpretation": {
            "exploratory_only": True,
            "no_feature_selected_or_removed": True,
            "no_model_fit": True,
            "confirmation_membership_unopened": True,
            "separate_rates_not_multiplied": True,
            "direct_joint_unsafe_follow_reported": True,
        },
        "access_audit": access_audit,
    }


__all__ = [
    "EXPLORATION_SCHEMA",
    "FEATURE_NAMES",
    "GEOMETRY_FEATURES",
    "PLAN_SCHEMA",
    "SPLIT_SCHEMA",
    "TACTICAL_FEATURES",
    "ExplorationOnlyAccess",
    "FeatureDeviationError",
    "FeatureOpportunity",
    "ProtectedResearchAccessError",
    "action_feature_scores",
    "build_train_internal_split",
    "load_plan",
    "load_research_split",
    "run_exploration",
    "verify_implementation_artifacts",
]
