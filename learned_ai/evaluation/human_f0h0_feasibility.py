"""Read-only F0-H0 rejection screen for the recovered human corpus.

The module deliberately separates three stages:

* freeze a player-component split without opening raw game records;
* measure tablebase cost and freeze either full or sampled membership; and
* run the preregistered screen on train and selection only.

Confirmation and final-test source records are never opened by the screen.
All Malom claims in this module are positional-only and use ``A_pos``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.malom_db import MalomDB, OracleMoveValue, compare_oracle_move_values
from game.board import BoardState
from game.rules import (
    get_all_legal_moves,
    get_game_phase,
    terminal_result,
    terminal_wdl,
)
from learned_ai.data.data_contract import (
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from game.draw_rules import StandardDrawTracker


PLAN_SCHEMA = "nmm.f0-h0-read-only-rejection-plan.v1"
SPLIT_SCHEMA = "nmm.f0-h0-player-component-split.v1"
COST_SCHEMA = "nmm.f0-h0-malom-cost-decision.v1"
RESULT_SCHEMA = "nmm.f0-h0-read-only-rejection-result.v1"

F0D0_SCHEMA = "nmm.f0-d0-human-raw-reconstructability.v2"
EXPECTED_F0D0_FILE_SHA256 = (
    "0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6"
)
EXPECTED_F0D0_MANIFEST_IDENTITY = (
    "bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7"
)
EXPECTED_CORPUS_IDENTITY = (
    "4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29"
)

PARTITIONS = (
    "train",
    "selection",
    "one-time-confirmation",
    "final-test",
)
ANALYSIS_PARTITIONS = frozenset({"train", "selection"})
PROTECTED_PARTITIONS = frozenset({"one-time-confirmation", "final-test"})
WDL_RANK = {"L": 0, "D": 1, "W": 2}


class F0H0Error(RuntimeError):
    """Raised when a required F0-H0 input or invariant fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the finite canonical JSON representation used for identities."""
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise F0H0Error("payload cannot be represented as canonical JSON") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F0H0Error(f"invalid JSON input: {source}") from exc
    if not isinstance(value, dict):
        raise F0H0Error(f"JSON input is not an object: {source}")
    return value, raw


def _load_sealed_json(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    value, raw = _load_json(path)
    if value.get("schema_version") != schema:
        raise F0H0Error(f"schema differs for {Path(path)}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise F0H0Error(f"identity is absent for {Path(path)}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise F0H0Error(f"identity differs for {Path(path)}")
    return value, hashlib.sha256(raw).hexdigest()


def write_sealed_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    """Write one new canonical evidence object without overwriting."""
    target = Path(path)
    if target.exists():
        raise F0H0Error(f"refusing to overwrite evidence: {target}")
    body = dict(payload)
    body.pop(identity_field, None)
    sealed = {**body, identity_field: canonical_sha256(body)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(sealed))
    return sealed


def load_result(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and verify a sealed F0-H0 result manifest."""

    return _load_sealed_json(
        path,
        schema=RESULT_SCHEMA,
        identity_field="result_identity",
    )


@dataclass(frozen=True)
class CorpusRecord:
    session_id: str
    canonical_file: str
    move_count: int
    recorded_outcome: str | None
    player_keys: tuple[str, str]
    behavior_eligible: bool
    outcome_eligible: bool


@dataclass(frozen=True)
class F0D0Boundary:
    manifest: Mapping[str, Any]
    file_sha256: str
    records: tuple[CorpusRecord, ...]
    raw_sha256_by_path: Mapping[str, str]
    raw_size_by_path: Mapping[str, int]


def _table_value(table: Sequence[Any], index: Any, *, field: str) -> Any:
    if isinstance(index, bool) or not isinstance(index, int):
        raise F0H0Error(f"encoded {field} index is invalid")
    try:
        return table[index]
    except IndexError as exc:
        raise F0H0Error(f"encoded {field} index is out of range") from exc


def load_f0d0_boundary(path: str | Path) -> F0D0Boundary:
    """Independently verify and decode the exact F0-D0 source boundary."""
    manifest, raw = _load_json(path)
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != EXPECTED_F0D0_FILE_SHA256:
        raise F0H0Error("F0-D0 manifest file SHA-256 differs")
    if manifest.get("schema_version") != F0D0_SCHEMA:
        raise F0H0Error("F0-D0 manifest schema differs")
    recorded_manifest_identity = manifest.get("manifest_identity")
    body = dict(manifest)
    body.pop("manifest_identity", None)
    manifest_identity = canonical_sha256(body)
    if (
        recorded_manifest_identity != EXPECTED_F0D0_MANIFEST_IDENTITY
        or manifest_identity != EXPECTED_F0D0_MANIFEST_IDENTITY
    ):
        raise F0H0Error("F0-D0 manifest identity differs")

    input_encoding = manifest.get("input_file_encoding")
    input_rows = manifest.get("input_files")
    if not isinstance(input_encoding, Mapping) or not isinstance(input_rows, list):
        raise F0H0Error("F0-D0 input-file encoding is absent")
    if input_encoding.get("fields") != [
        "relative_path",
        "role_index",
        "byte_length",
        "sha256",
        "session_id",
        "status_index",
        "failure_index",
    ]:
        raise F0H0Error("F0-D0 input-file fields differ")
    roles = input_encoding.get("role_values")
    if not isinstance(roles, list):
        raise F0H0Error("F0-D0 input role table is absent")
    raw_sha_by_path: dict[str, str] = {}
    raw_size_by_path: dict[str, int] = {}
    raw_identity_rows: list[dict[str, Any]] = []
    for row in input_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise F0H0Error("F0-D0 input-file row width differs")
        if _table_value(roles, row[1], field="role") is not None:
            continue
        relative_path, byte_length, digest = row[0], row[2], row[3]
        if (
            not isinstance(relative_path, str)
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or not isinstance(digest, str)
        ):
            raise F0H0Error("F0-D0 raw-file row is invalid")
        raw_sha_by_path[relative_path] = digest
        raw_size_by_path[relative_path] = byte_length
        raw_identity_rows.append(
            {
                "relative_path": relative_path,
                "byte_length": byte_length,
                "sha256": digest,
            }
        )

    identities = manifest.get("identities")
    if not isinstance(identities, Mapping):
        raise F0H0Error("F0-D0 nested identities are absent")
    raw_identity = canonical_sha256(raw_identity_rows)
    if raw_identity != identities.get("raw_files_identity"):
        raise F0H0Error("F0-D0 raw-file identity differs")

    encoding = manifest.get("record_encoding")
    game_rows = manifest.get("game_records")
    if not isinstance(encoding, Mapping) or not isinstance(game_rows, list):
        raise F0H0Error("F0-D0 game-record encoding is absent")
    fields = encoding.get("fields")
    if not isinstance(fields, list):
        raise F0H0Error("F0-D0 game-record fields are absent")
    positions = {field: index for index, field in enumerate(fields)}
    required = {
        "session_id",
        "canonical_file",
        "imported_at",
        "move_count",
        "recorded_outcome_index",
        "player_key_indices",
        "behavior_replay_eligible",
        "outcome_analysis_eligible",
    }
    if not required.issubset(positions):
        raise F0H0Error("F0-D0 game-record fields are incomplete")
    player_table = encoding.get("player_keys")
    outcome_table = encoding.get("outcome_values")
    if not isinstance(player_table, list) or not isinstance(outcome_table, list):
        raise F0H0Error("F0-D0 record tables are incomplete")

    records: list[CorpusRecord] = []
    session_identity_rows: list[dict[str, Any]] = []
    for row in game_rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise F0H0Error("F0-D0 game-record row width differs")
        session_id = row[positions["session_id"]]
        canonical_file = row[positions["canonical_file"]]
        move_count = row[positions["move_count"]]
        player_indices = row[positions["player_key_indices"]]
        if (
            not isinstance(session_id, str)
            or not isinstance(canonical_file, str)
            or isinstance(move_count, bool)
            or not isinstance(move_count, int)
            or not isinstance(player_indices, list)
            or len(player_indices) != 2
        ):
            raise F0H0Error("F0-D0 game-record row is invalid")
        if canonical_file not in raw_sha_by_path:
            raise F0H0Error("canonical raw file is absent from F0-D0 inputs")
        player_keys = tuple(
            _table_value(player_table, item, field="player key")
            for item in player_indices
        )
        if any(not isinstance(item, str) for item in player_keys):
            raise F0H0Error("decoded player key is invalid")
        outcome = _table_value(
            outcome_table,
            row[positions["recorded_outcome_index"]],
            field="recorded outcome",
        )
        records.append(
            CorpusRecord(
                session_id=session_id,
                canonical_file=canonical_file,
                move_count=move_count,
                recorded_outcome=outcome,
                player_keys=(player_keys[0], player_keys[1]),
                behavior_eligible=bool(
                    row[positions["behavior_replay_eligible"]]
                ),
                outcome_eligible=bool(
                    row[positions["outcome_analysis_eligible"]]
                ),
            )
        )
        session_identity_rows.append(
            {
                "session_id": session_id,
                "canonical_file": canonical_file,
                "file_sha256": raw_sha_by_path[canonical_file],
                "imported_at": row[positions["imported_at"]],
            }
        )
    session_identity = canonical_sha256(session_identity_rows)
    if session_identity != identities.get("session_source_identity"):
        raise F0H0Error("F0-D0 session-source identity differs")
    corpus_identity = canonical_sha256(
        {
            "schema_version": "nmm.human-raw-corpus.v1",
            "raw_files_identity": raw_identity,
            "session_source_identity": session_identity,
            "imported_manifest_sha256": manifest["imported_manifest"]["sha256"],
            "unique_sessions": len(records),
        }
    )
    if (
        corpus_identity != EXPECTED_CORPUS_IDENTITY
        or corpus_identity != identities.get("corpus_identity")
    ):
        raise F0H0Error("F0-D0 corpus identity differs")
    return F0D0Boundary(
        manifest=manifest,
        file_sha256=file_sha,
        records=tuple(records),
        raw_sha256_by_path=raw_sha_by_path,
        raw_size_by_path=raw_size_by_path,
    )


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = _load_sealed_json(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    if plan.get("claim_boundary", {}).get("safe_set_name") != "A_pos":
        raise F0H0Error("F0-H0 plan does not bind positional-only A_pos")
    if plan.get("claim_boundary", {}).get("full_rule_safety_claim") is not False:
        raise F0H0Error("F0-H0 plan does not reject a full-rule safety claim")
    boundary = plan.get("input_boundary")
    expected_boundary = {
        "corpus_identity": EXPECTED_CORPUS_IDENTITY,
        "manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) != expected
        for field, expected in expected_boundary.items()
    ):
        raise F0H0Error("F0-H0 plan input identity differs")
    if boundary.get("behavior") != {
        "games": 92_226,
        "logical_plies": 4_394_220,
        "player_keys": 4_994,
    }:
        raise F0H0Error("F0-H0 plan behavior base differs")
    if boundary.get("factual") != {
        "games": 37_866,
        "player_keys": 3_392,
    }:
        raise F0H0Error("F0-H0 plan factual base differs")
    if plan.get("protected_access", {}).get("final-test") != "unopened":
        raise F0H0Error("F0-H0 plan does not keep final-test unopened")
    return plan, file_sha


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _hash_fraction(namespace: str, seed: str, value: str) -> float:
    raw = f"{namespace}\0{seed}\0{value}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return integer / 2**64


def _fraction_partition(value: float, ratios: Mapping[str, float]) -> str:
    cumulative = 0.0
    for partition in PARTITIONS:
        cumulative += float(ratios[partition])
        if value < cumulative or partition == PARTITIONS[-1]:
            return partition
    raise AssertionError("partition ratios did not cover the unit interval")


def _component_identity(
    players: Sequence[str],
    sessions: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "players": sorted(players),
            "sessions": sorted(sessions),
        }
    )


def build_split(
    *,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen split using manifest structure only, never raw games."""
    split_spec = plan["split"]
    ratios = split_spec["ratios"]
    seed = split_spec["seed"]
    if tuple(ratios) != PARTITIONS:
        raise F0H0Error("split partition order differs")
    if not math.isclose(sum(float(value) for value in ratios.values()), 1.0):
        raise F0H0Error("split ratios do not sum to one")

    records = [record for record in boundary.records if record.behavior_eligible]
    players = sorted({key for record in records for key in record.player_keys})
    expected = plan["input_boundary"]["behavior"]
    if (
        len(records) != int(expected["games"])
        or sum(record.move_count for record in records)
        != int(expected["logical_plies"])
        or len(players) != int(expected["player_keys"])
    ):
        raise F0H0Error("behavior eligibility base differs from preregistration")

    dsu = _DisjointSet(players)
    provisional: dict[str, str] = {}
    for player in players:
        provisional[player] = _fraction_partition(
            _hash_fraction("f0-h0-provisional-player-v1", seed, player),
            ratios,
        )
    conflict_games = 0
    for record in records:
        white, black = record.player_keys
        dsu.union(white, black)
        if provisional[white] != provisional[black]:
            conflict_games += 1

    component_players: dict[str, list[str]] = defaultdict(list)
    for player in players:
        component_players[dsu.find(player)].append(player)
    component_records: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        root = dsu.find(record.player_keys[0])
        if root != dsu.find(record.player_keys[1]):
            raise F0H0Error("player component construction failed")
        component_records[root].append(record)

    components: list[dict[str, Any]] = []
    for root, player_values in component_players.items():
        record_values = component_records[root]
        sessions = [record.session_id for record in record_values]
        identity = _component_identity(player_values, sessions)
        components.append(
            {
                "component_identity": identity,
                "players": sorted(player_values),
                "records": sorted(record_values, key=lambda item: item.session_id),
                "games": len(record_values),
                "logical_plies": sum(item.move_count for item in record_values),
            }
        )
    components.sort(
        key=lambda item: (-int(item["games"]), item["component_identity"])
    )

    targets = {partition: len(records) * float(ratios[partition]) for partition in PARTITIONS}
    assigned_games = {partition: 0 for partition in PARTITIONS}
    component_partition: dict[str, str] = {}
    for component in components:
        games = int(component["games"])

        def score(partition: str) -> tuple[float, str]:
            normalized = (assigned_games[partition] + games) / targets[partition]
            tie = canonical_sha256(
                {
                    "component": component["component_identity"],
                    "partition": partition,
                    "seed": seed,
                }
            )
            return normalized, tie

        chosen = min(PARTITIONS, key=score)
        component_partition[component["component_identity"]] = chosen
        assigned_games[chosen] += games

    game_rows: list[list[str]] = []
    player_rows: list[list[str]] = []
    counts = {
        partition: {"games": 0, "logical_plies": 0, "player_keys": 0}
        for partition in PARTITIONS
    }
    for component in components:
        identity = component["component_identity"]
        partition = component_partition[identity]
        counts[partition]["games"] += component["games"]
        counts[partition]["logical_plies"] += component["logical_plies"]
        counts[partition]["player_keys"] += len(component["players"])
        game_rows.extend(
            [record.session_id, partition, identity]
            for record in component["records"]
        )
        player_rows.extend(
            [player, partition, identity] for player in component["players"]
        )
    game_rows.sort()
    player_rows.sort()
    game_identity = canonical_sha256(game_rows)
    player_identity = canonical_sha256(player_rows)
    component_summary = [
        {
            "component_identity": component["component_identity"],
            "games": component["games"],
            "logical_plies": component["logical_plies"],
            "partition": component_partition[component["component_identity"]],
            "player_keys": len(component["players"]),
        }
        for component in components
    ]
    return {
        "schema_version": SPLIT_SCHEMA,
        "screen_id": plan["screen_id"],
        "plan_identity": plan["plan_identity"],
        "f0d0_boundary": {
            "corpus_identity": EXPECTED_CORPUS_IDENTITY,
            "file_sha256": boundary.file_sha256,
            "manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        },
        "algorithm": split_spec["algorithm"],
        "ratios": ratios,
        "seed": seed,
        "component_count": len(components),
        "largest_component": component_summary[0],
        "provisional_cross_partition_games": conflict_games,
        "conflict_resolution": (
            "the entire connected player-game component receives one "
            "deterministic target-balanced partition"
        ),
        "counts": counts,
        "component_summary": component_summary,
        "game_membership_fields": [
            "session_id",
            "partition",
            "component_identity",
        ],
        "game_membership": game_rows,
        "game_membership_identity": game_identity,
        "player_membership_fields": [
            "player_key",
            "partition",
            "component_identity",
        ],
        "player_membership": player_rows,
        "player_membership_identity": player_identity,
        "access_state": {
            "one-time-confirmation_raw_record_reads": 0,
            "one-time-confirmation_statistics_reads": 0,
            "final-test_raw_record_reads": 0,
            "final-test_statistics_reads": 0,
            "membership_only_was_generated": True,
        },
    }


def verify_split(
    split: Mapping[str, Any],
    *,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
) -> None:
    if split.get("schema_version") != SPLIT_SCHEMA:
        raise F0H0Error("split schema differs")
    if split.get("plan_identity") != plan.get("plan_identity"):
        raise F0H0Error("split plan identity differs")
    games = split.get("game_membership")
    players = split.get("player_membership")
    if not isinstance(games, list) or not isinstance(players, list):
        raise F0H0Error("split membership arrays are absent")
    if canonical_sha256(games) != split.get("game_membership_identity"):
        raise F0H0Error("game membership identity differs")
    if canonical_sha256(players) != split.get("player_membership_identity"):
        raise F0H0Error("player membership identity differs")
    behavior_sessions = {
        record.session_id for record in boundary.records if record.behavior_eligible
    }
    session_partition = {row[0]: row[1] for row in games}
    if set(session_partition) != behavior_sessions or len(games) != len(session_partition):
        raise F0H0Error("split does not cover each behavior game exactly once")
    player_partition: dict[str, str] = {}
    for row in players:
        if row[0] in player_partition:
            raise F0H0Error("a player appears more than once in split membership")
        player_partition[row[0]] = row[1]
    for record in boundary.records:
        if not record.behavior_eligible:
            continue
        partition = session_partition[record.session_id]
        if any(player_partition[player] != partition for player in record.player_keys):
            raise F0H0Error("a player crosses split partitions")
    access = split.get("access_state")
    if not isinstance(access, Mapping) or any(
        access.get(field) != 0
        for field in (
            "one-time-confirmation_raw_record_reads",
            "one-time-confirmation_statistics_reads",
            "final-test_raw_record_reads",
            "final-test_statistics_reads",
        )
    ):
        raise F0H0Error("protected split access state is not sealed")


def load_split(
    path: str | Path,
    *,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    split, file_sha = _load_sealed_json(
        path,
        schema=SPLIT_SCHEMA,
        identity_field="split_identity",
    )
    verify_split(split, boundary=boundary, plan=plan)
    return split, file_sha


def split_session_partitions(split: Mapping[str, Any]) -> dict[str, str]:
    return {str(row[0]): str(row[1]) for row in split["game_membership"]}


def _read_raw_game(
    repository_root: Path,
    record: CorpusRecord,
    boundary: F0D0Boundary,
) -> Mapping[str, Any]:
    path = repository_root / record.canonical_file
    raw = path.read_bytes()
    if len(raw) != boundary.raw_size_by_path[record.canonical_file]:
        raise F0H0Error(f"raw game size differs: {record.canonical_file}")
    if hashlib.sha256(raw).hexdigest() != boundary.raw_sha256_by_path[
        record.canonical_file
    ]:
        raise F0H0Error(f"raw game SHA-256 differs: {record.canonical_file}")
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise F0H0Error(f"raw game framing differs: {record.canonical_file}")
    try:
        payload = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F0H0Error(f"raw game JSON differs: {record.canonical_file}") from exc
    if not isinstance(payload, Mapping) or payload.get("session_id") != record.session_id:
        raise F0H0Error(f"raw game session differs: {record.canonical_file}")
    return payload


def _notation(move: Mapping[str, Any]) -> str:
    source = move.get("from")
    target = move.get("to")
    base = str(target) if source is None else f"{source}-{target}"
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


@dataclass(frozen=True)
class ReplayedDecision:
    logical_ply: int
    board: BoardState
    move: Mapping[str, Any]
    actor_player_key: str
    game_id: str


def replay_game(
    raw_game: Mapping[str, Any],
    record: CorpusRecord,
) -> list[ReplayedDecision]:
    """Revalidate one F0-D0-eligible game and retain decision states."""
    moves = raw_game.get("moves")
    if not isinstance(moves, list) or len(moves) != record.move_count:
        raise F0H0Error(f"raw move inventory differs: {record.session_id}")
    board = BoardState.new_game()
    tracker = StandardDrawTracker(board)
    terminal_seen = False
    decisions: list[ReplayedDecision] = []
    for index, raw_move in enumerate(moves):
        if terminal_seen or not isinstance(raw_move, Mapping):
            raise F0H0Error(f"strict replay framing differs: {record.session_id}")
        if (
            raw_move.get("board_fen_before") != board.to_fen_string()
            or raw_move.get("color") != board.turn
            or raw_move.get("turn") != index // 2 + 1
            or raw_move.get("type") != get_game_phase(board, board.turn)
        ):
            raise F0H0Error(f"strict replay metadata differs: {record.session_id}")
        expected = {
            "from": raw_move.get("from"),
            "to": raw_move.get("to"),
            "capture": raw_move.get("capture"),
        }
        legal = get_all_legal_moves(board)
        matches = [
            move
            for move in legal
            if all(move.get(field) == expected[field] for field in expected)
        ]
        if len(matches) != 1 or raw_move.get("notation") != _notation(matches[0]):
            raise F0H0Error(f"strict replay move differs: {record.session_id}")
        actor_index = 0 if board.turn == "W" else 1
        decisions.append(
            ReplayedDecision(
                logical_ply=index,
                board=board,
                move=dict(matches[0]),
                actor_player_key=record.player_keys[actor_index],
                game_id=record.session_id,
            )
        )
        after = board.apply_move(matches[0])
        draw_reason = tracker.observe(board, matches[0], after)
        is_terminal, _winner, _reason = terminal_result(after)
        terminal_seen = bool(is_terminal or draw_reason is not None)
        board = after
    return decisions


def _legal_count_bucket(count: int) -> str:
    if count <= 0:
        raise F0H0Error("a behavior decision has no legal action")
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17+"


def coarse_state_class(board: BoardState, legal_count: int) -> str:
    actor = board.turn
    opponent = "B" if actor == "W" else "W"
    return "|".join(
        (
            f"phase={get_game_phase(board, actor)}",
            f"color={actor}",
            f"own_on={board.pieces_on_board[actor]}",
            f"opp_on={board.pieces_on_board[opponent]}",
            f"own_unplaced={9 - board.pieces_placed[actor]}",
            f"opp_unplaced={9 - board.pieces_placed[opponent]}",
            f"legal={_legal_count_bucket(legal_count)}",
        )
    )


def positional_state_identity(board: BoardState) -> str:
    return hashlib.sha256(
        f"f0-h0-positional-state-v1\0{board.to_fen_string()}".encode("utf-8")
    ).hexdigest()


def _oracle_inventory(
    board: BoardState,
    database: MalomDB,
) -> tuple[str, list[tuple[Mapping[str, Any], OracleMoveValue]], int]:
    """Return every legal action's complete positional value or fail closed."""
    parent = database.query_value(board)
    if parent is None or parent.outcome not in WDL_RANK:
        raise F0H0Error("required parent Malom value is unavailable")
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
                raise F0H0Error("required successor Malom value is unavailable")
            value = database.move_value(parent, child)
        if value.outcome not in WDL_RANK:
            raise F0H0Error("candidate Malom value has an invalid outcome")
        results.append((dict(move), value))
    if not results:
        raise F0H0Error("behavior decision has no positional candidates")
    best_tier = max(
        (value.outcome for _move, value in results),
        key=WDL_RANK.get,
    )
    if best_tier != parent.outcome:
        raise F0H0Error("candidate inventory contradicts parent Malom tier")
    contexts = {
        (value.sector, value.sector_value, value.perspective)
        for _move, value in results
    }
    if len(contexts) != 1:
        raise F0H0Error("candidate inventory has mixed Malom contexts")
    return parent.outcome, results, query_count


def verify_malom_snapshot(
    *,
    malom_path: str | Path,
    manifest_path: str | Path,
    full_hash: bool,
) -> dict[str, Any]:
    manifest = load_dataset_manifest(manifest_path)
    if (
        manifest.logical_name != "malom_tablebase"
        or manifest.trust_level != CURRENT_MALOM_LABEL_VERSION
        or manifest.trust_level != "sector-corrected-v1"
        or "theoretical_wdl" not in manifest.label_kinds
    ):
        raise F0H0Error("Malom manifest trust boundary differs")
    snapshot = verify_dataset_snapshot(malom_path, manifest, full_hash=full_hash)
    return {
        **snapshot,
        "content_sha256": manifest.content_sha256,
        "dataset_id": manifest.dataset_id,
        "manifest_file_sha256": sha256_file(manifest_path),
        "trust_level": manifest.trust_level,
        "components": [component.to_dict() for component in manifest.components],
    }


def _ranked_sessions(
    records: Sequence[CorpusRecord],
    *,
    namespace: str,
    seed: str,
) -> list[CorpusRecord]:
    return sorted(
        records,
        key=lambda record: (
            canonical_sha256(
                {
                    "namespace": namespace,
                    "seed": seed,
                    "session_id": record.session_id,
                }
            ),
            record.session_id,
        ),
    )


def build_cost_decision(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
    split: Mapping[str, Any],
    malom_path: str | Path,
    malom_manifest_path: str | Path,
) -> dict[str, Any]:
    """Run the bounded preregistered cost pilot and freeze analysis members."""
    cost_spec = plan["cost"]
    partitions = split_session_partitions(split)
    by_session = {record.session_id: record for record in boundary.records}
    train_records = [
        by_session[session]
        for session, partition in partitions.items()
        if partition == "train"
    ]
    pilot_games = _ranked_sessions(
        train_records,
        namespace="f0-h0-cost-pilot-game-v1",
        seed=cost_spec["pilot_seed"],
    )[: int(cost_spec["pilot_games"])]
    if len(pilot_games) != int(cost_spec["pilot_games"]):
        raise F0H0Error("cost pilot has insufficient train games")

    snapshot = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=malom_manifest_path,
        full_hash=bool(cost_spec["require_full_malom_hash"]),
    )
    database = MalomDB(malom_path)
    if not database.is_available():
        raise F0H0Error("Malom tablebase is unavailable for cost pilot")
    root = Path(repository_root)
    selected: list[ReplayedDecision] = []
    raw_inputs: list[dict[str, Any]] = []
    try:
        per_game_states = int(cost_spec["pilot_states_per_game"])
        for record in pilot_games:
            raw_game = _read_raw_game(root, record, boundary)
            decisions = replay_game(raw_game, record)
            ranked = sorted(
                decisions,
                key=lambda decision: canonical_sha256(
                    {
                        "logical_ply": decision.logical_ply,
                        "seed": cost_spec["pilot_seed"],
                        "session_id": record.session_id,
                    }
                ),
            )
            selected.extend(ranked[:per_game_states])
            raw_inputs.append(
                {
                    "relative_path": record.canonical_file,
                    "sha256": boundary.raw_sha256_by_path[record.canonical_file],
                    "size_bytes": boundary.raw_size_by_path[record.canonical_file],
                }
            )
        expected_states = len(pilot_games) * per_game_states
        if len(selected) != expected_states:
            raise F0H0Error("cost pilot has insufficient decision states")
        start = time.perf_counter()
        query_count = 0
        state_queries: list[int] = []
        legal_actions = 0
        state_seconds: list[float] = []
        for decision in selected:
            state_start = time.perf_counter()
            _tier, inventory, queries = _oracle_inventory(decision.board, database)
            state_seconds.append(time.perf_counter() - state_start)
            query_count += queries
            state_queries.append(queries)
            legal_actions += len(inventory)
        elapsed = time.perf_counter() - start
    finally:
        database.close()

    analysis_plies = sum(
        int(split["counts"][partition]["logical_plies"])
        for partition in ANALYSIS_PARTITIONS
    )
    mean_queries = query_count / len(selected)
    mean_seconds = elapsed / len(selected)

    def upper_mean(values: Sequence[int | float]) -> float:
        mean = sum(values) / len(values)
        if len(values) < 2:
            raise F0H0Error("cost pilot has insufficient observations")
        variance = sum((float(value) - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        return mean + 1.959963984540054 * math.sqrt(variance / len(values))

    queries_per_state_upper_95 = upper_mean(state_queries)
    seconds_per_state_upper_95 = upper_mean(state_seconds)
    projected_queries = math.ceil(queries_per_state_upper_95 * analysis_plies)
    projected_seconds = seconds_per_state_upper_95 * analysis_plies
    full_allowed = (
        projected_queries <= int(cost_spec["maximum_full_queries"])
        and projected_seconds <= float(cost_spec["maximum_full_wall_seconds"])
    )
    mode = "full" if full_allowed else "sample"
    eligible_records = [
        by_session[session]
        for session, partition in partitions.items()
        if partition in ANALYSIS_PARTITIONS
    ]
    if mode == "full":
        analysis_records = sorted(eligible_records, key=lambda item: item.session_id)
    else:
        analysis_records = _ranked_sessions(
            eligible_records,
            namespace="f0-h0-oracle-analysis-game-v1",
            seed=cost_spec["sample_seed"],
        )[: int(cost_spec["sample_games"])]
    if not analysis_records:
        raise F0H0Error("cost decision selected no analysis games")
    selected_plies = sum(record.move_count for record in analysis_records)
    selected_projected_queries = math.ceil(
        queries_per_state_upper_95 * selected_plies
    )
    selected_projected_seconds = seconds_per_state_upper_95 * selected_plies
    if (
        selected_projected_queries > int(cost_spec["maximum_sample_queries"])
        or selected_projected_seconds
        > float(cost_spec["maximum_sample_wall_seconds"])
    ):
        raise F0H0Error("frozen sample exceeds its preregistered cost bound")
    analysis_membership = [record.session_id for record in analysis_records]
    return {
        "schema_version": COST_SCHEMA,
        "screen_id": plan["screen_id"],
        "plan_identity": plan["plan_identity"],
        "split_identity": split["split_identity"],
        "claim_boundary": {
            "labels": "positional-only",
            "safe_set_name": "A_pos",
            "full_rule_safety_claim": False,
        },
        "malom_snapshot": snapshot,
        "pilot": {
            "game_membership": [record.session_id for record in pilot_games],
            "game_membership_identity": canonical_sha256(
                [record.session_id for record in pilot_games]
            ),
            "raw_inputs": sorted(raw_inputs, key=lambda item: item["relative_path"]),
            "states": len(selected),
            "queries": query_count,
            "legal_actions": legal_actions,
            "elapsed_seconds": elapsed,
            "mean_queries_per_state": mean_queries,
            "mean_seconds_per_state": mean_seconds,
            "queries_per_state_upper_95": queries_per_state_upper_95,
            "seconds_per_state_upper_95": seconds_per_state_upper_95,
            "maximum_state_seconds": max(state_seconds),
        },
        "projection": {
            "analysis_eligible_logical_plies": analysis_plies,
            "projected_full_queries": projected_queries,
            "projected_full_wall_seconds": projected_seconds,
            "maximum_full_queries": int(cost_spec["maximum_full_queries"]),
            "maximum_full_wall_seconds": float(
                cost_spec["maximum_full_wall_seconds"]
            ),
            "selected_analysis_logical_plies": selected_plies,
            "selected_projected_queries": selected_projected_queries,
            "selected_projected_wall_seconds": selected_projected_seconds,
            "maximum_sample_queries": int(cost_spec["maximum_sample_queries"]),
            "maximum_sample_wall_seconds": float(
                cost_spec["maximum_sample_wall_seconds"]
            ),
        },
        "decision": mode,
        "analysis_game_membership": analysis_membership,
        "analysis_game_membership_identity": canonical_sha256(analysis_membership),
        "analysis_games": len(analysis_membership),
        "selection_rule": (
            "all train+selection games" if mode == "full" else cost_spec["sample_rule"]
        ),
        "screening_statistics_observed_before_decision": False,
        "protected_partition_raw_reads": {
            "one-time-confirmation": 0,
            "final-test": 0,
        },
    }


def load_cost_decision(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    split: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    value, file_sha = _load_sealed_json(
        path,
        schema=COST_SCHEMA,
        identity_field="cost_decision_identity",
    )
    if (
        value.get("plan_identity") != plan.get("plan_identity")
        or value.get("split_identity") != split.get("split_identity")
    ):
        raise F0H0Error("cost decision lineage differs")
    members = value.get("analysis_game_membership")
    if not isinstance(members, list) or canonical_sha256(members) != value.get(
        "analysis_game_membership_identity"
    ):
        raise F0H0Error("cost decision membership identity differs")
    if value.get("protected_partition_raw_reads") != {
        "one-time-confirmation": 0,
        "final-test": 0,
    }:
        raise F0H0Error("cost decision opened a protected partition")
    return value, file_sha


def quantiles(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def percentile(probability: float) -> float:
        index = probability * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def gini(values: Sequence[int | float]) -> float:
    positive = sorted(float(value) for value in values if value >= 0)
    if not positive or sum(positive) == 0:
        return 0.0
    n = len(positive)
    weighted = sum((index + 1) * value for index, value in enumerate(positive))
    return (2.0 * weighted) / (n * sum(positive)) - (n + 1.0) / n


def concentration(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted((int(value) for value in values), reverse=True)
    total = sum(ordered)
    if not ordered or total <= 0:
        raise F0H0Error("concentration input is empty")

    def top_share(fraction: float) -> float:
        count = max(1, math.ceil(len(ordered) * fraction))
        return sum(ordered[:count]) / total

    squared = sum((value / total) ** 2 for value in ordered)
    return {
        "units": len(ordered),
        "observations": total,
        "maximum_share": ordered[0] / total,
        "top_1_percent_share": top_share(0.01),
        "top_5_percent_share": top_share(0.05),
        "top_10_percent_share": top_share(0.10),
        "gini": gini(ordered),
        "kish_effective_units": 1.0 / squared,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise F0H0Error("Wilson interval input is invalid")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "point": proportion,
        "lower_95": max(0.0, center - radius),
        "upper_95": min(1.0, center + radius),
    }


def component_robust_interval(
    observations: Sequence[tuple[str, bool]],
    z: float = 1.959963984540054,
) -> dict[str, Any]:
    """Return a game-weighted interval robust to player-network components.

    The reported interval is the envelope of a fixed-membership Wilson
    interval and a connected-component cluster-robust normal interval.  If
    fewer than two independent components are observed, the interval fails
    closed to [0, 1].
    """
    if not observations:
        raise F0H0Error("component interval input is empty")
    by_component: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for component, success in observations:
        by_component[component][0] += int(success)
        by_component[component][1] += 1
    successes = sum(value[0] for value in by_component.values())
    total = sum(value[1] for value in by_component.values())
    point = successes / total
    wilson = wilson_interval(successes, total, z=z)
    component_count = len(by_component)
    largest_share = max(value[1] for value in by_component.values()) / total
    if component_count < 2:
        return {
            "point": point,
            "successes": successes,
            "observations": total,
            "lower_95": 0.0,
            "upper_95": 1.0,
            "method": "fail-closed-insufficient-independent-components",
            "independent_components": component_count,
            "largest_component_game_share": largest_share,
            "cluster_robust_standard_error": None,
            "fixed_membership_wilson": wilson,
        }
    score_squares = sum(
        (component_successes - point * component_total) ** 2
        for component_successes, component_total in by_component.values()
    )
    variance = component_count / (component_count - 1) * score_squares / total**2
    standard_error = math.sqrt(max(0.0, variance))
    cluster_lower = max(0.0, point - z * standard_error)
    cluster_upper = min(1.0, point + z * standard_error)
    return {
        "point": point,
        "successes": successes,
        "observations": total,
        "lower_95": min(wilson["lower_95"], cluster_lower),
        "upper_95": max(wilson["upper_95"], cluster_upper),
        "method": (
            "envelope-of-fixed-membership-wilson-and-player-network-"
            "component-cluster-robust-normal"
        ),
        "independent_components": component_count,
        "largest_component_game_share": largest_share,
        "cluster_robust_standard_error": standard_error,
        "cluster_robust_normal": {
            "lower_95": cluster_lower,
            "upper_95": cluster_upper,
        },
        "fixed_membership_wilson": wilson,
    }


@dataclass(frozen=True)
class PositionalDecisionLabel:
    parent_tier: str
    chosen_tier: str
    a_pos_cardinality: int
    chosen_preserves_tier: bool
    within_tier_full_regret: bool
    legal_actions: int
    query_count: int


def label_positional_decision(
    decision: ReplayedDecision,
    database: MalomDB,
) -> PositionalDecisionLabel:
    parent_tier, inventory, query_count = _oracle_inventory(decision.board, database)
    best_rank = max(WDL_RANK[value.outcome] for _move, value in inventory)
    a_pos = [
        (move, value)
        for move, value in inventory
        if WDL_RANK[value.outcome] == best_rank
    ]
    chosen = [
        value for move, value in inventory if dict(move) == dict(decision.move)
    ]
    if len(chosen) != 1:
        raise F0H0Error("observed action is absent from Malom inventory")
    chosen_value = chosen[0]
    best_full = max(inventory, key=lambda item: item[1].ordering_key())[1]
    comparison = compare_oracle_move_values(chosen_value, best_full)
    preserves = chosen_value.outcome == parent_tier
    return PositionalDecisionLabel(
        parent_tier=parent_tier,
        chosen_tier=chosen_value.outcome,
        a_pos_cardinality=len(a_pos),
        chosen_preserves_tier=preserves,
        within_tier_full_regret=bool(preserves and comparison < 0),
        legal_actions=len(inventory),
        query_count=query_count,
    )


def _support_summary(
    state_support: Mapping[str, Mapping[str, Any]],
    *,
    min_players: int,
    min_games: int,
) -> dict[str, Any]:
    player_counts = [len(value["players"]) for value in state_support.values()]
    game_counts = [len(value["games"]) for value in state_support.values()]
    supported = [
        key
        for key, value in state_support.items()
        if len(value["players"]) >= min_players
        and len(value["games"]) >= min_games
    ]
    total_decisions = sum(int(value["decisions"]) for value in state_support.values())
    supported_decisions = sum(
        int(state_support[key]["decisions"]) for key in supported
    )
    return {
        "states_or_classes": len(state_support),
        "independent_players_quantiles": quantiles(player_counts),
        "independent_games_quantiles": quantiles(game_counts),
        "support_floor": {
            "minimum_independent_players": min_players,
            "minimum_independent_games": min_games,
        },
        "supported_states_or_classes": len(supported),
        "supported_state_or_class_fraction": (
            len(supported) / len(state_support) if state_support else 0.0
        ),
        "supported_decision_fraction": (
            supported_decisions / total_decisions if total_decisions else 0.0
        ),
        "supported_keys": supported,
    }


def run_screen(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    split: Mapping[str, Any],
    split_file_sha256: str,
    cost: Mapping[str, Any],
    cost_file_sha256: str,
    malom_path: str | Path,
    malom_manifest_path: str | Path,
    ruleset_path: str | Path,
) -> dict[str, Any]:
    """Execute the frozen train+selection rejection screen."""
    session_partition = split_session_partitions(split)
    component_by_session = {
        str(row[0]): str(row[2]) for row in split["game_membership"]
    }
    by_session = {record.session_id: record for record in boundary.records}
    analysis_records = [
        by_session[session]
        for session, partition in session_partition.items()
        if partition in ANALYSIS_PARTITIONS
    ]
    oracle_members = set(cost["analysis_game_membership"])
    if any(session_partition[session] not in ANALYSIS_PARTITIONS for session in oracle_members):
        raise F0H0Error("oracle membership includes a protected partition")

    support_spec = plan["thresholds"]["support"]
    player_decisions: Counter[str] = Counter()
    game_decisions: Counter[str] = Counter()
    coarse_support: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"decisions": 0, "players": set(), "games": set()}
    )
    replayed: dict[str, list[ReplayedDecision]] = {}
    raw_inputs: list[dict[str, Any]] = []
    root = Path(repository_root)
    for record in sorted(analysis_records, key=lambda item: item.session_id):
        raw_game = _read_raw_game(root, record, boundary)
        decisions = replay_game(raw_game, record)
        if record.session_id in oracle_members:
            replayed[record.session_id] = decisions
        raw_inputs.append(
            {
                "relative_path": record.canonical_file,
                "sha256": boundary.raw_sha256_by_path[record.canonical_file],
                "size_bytes": boundary.raw_size_by_path[record.canonical_file],
            }
        )
        game_decisions[record.session_id] += len(decisions)
        for decision in decisions:
            legal_count = len(get_all_legal_moves(decision.board))
            key = coarse_state_class(decision.board, legal_count)
            player_decisions[decision.actor_player_key] += 1
            support = coarse_support[key]
            support["decisions"] += 1
            support["players"].add(decision.actor_player_key)
            support["games"].add(decision.game_id)

    coarse_summary = _support_summary(
        coarse_support,
        min_players=int(support_spec["coarse_minimum_players"]),
        min_games=int(support_spec["coarse_minimum_games"]),
    )
    supported_coarse = set(coarse_summary.pop("supported_keys"))

    snapshot = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=malom_manifest_path,
        full_hash=False,
    )
    expected_snapshot = cost["malom_snapshot"]
    for field in (
        "manifest_sha256",
        "content_sha256",
        "component_count",
        "size_bytes",
        "trust_level",
    ):
        if snapshot[field] != expected_snapshot[field]:
            raise F0H0Error("Malom snapshot differs from frozen cost decision")

    database = MalomDB(malom_path)
    if not database.is_available():
        raise F0H0Error("Malom tablebase is unavailable for F0-H0")
    a_pos_counts: Counter[int] = Counter()
    by_phase_color_tier: Counter[tuple[str, str, str, str]] = Counter()
    exact_support: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"decisions": 0, "players": set(), "games": set()}
    )
    first_downgrades: Counter[str] = Counter()
    first_downgrade_games = 0
    steerable_supported_games = 0
    steerable_unsupported_games = 0
    modifiable_reach_games = 0
    supported_modifiable_reach_games = 0
    within_tier_regret = 0
    preserving_decisions = 0
    oracle_decisions = 0
    oracle_queries = 0
    factual_sample_outcomes: Counter[str] = Counter()
    factual_modifiable_reach = 0
    factual_supported_modifiable_reach = 0
    factual_supported_steerable = 0
    factual_supported_steerable_outcomes: Counter[str] = Counter()
    per_game_facts: list[dict[str, Any]] = []
    interval_observations: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    try:
        for session in cost["analysis_game_membership"]:
            record = by_session[session]
            decisions = replayed.get(session)
            if decisions is None:
                raise F0H0Error("frozen oracle member was not replayed")
            labels: list[tuple[ReplayedDecision, PositionalDecisionLabel, str]] = []
            game_modifiable = False
            game_supported_modifiable = False
            for decision in decisions:
                label = label_positional_decision(decision, database)
                legal_count = label.legal_actions
                coarse_key = coarse_state_class(decision.board, legal_count)
                state_key = positional_state_identity(decision.board)
                support = exact_support[state_key]
                support["decisions"] += 1
                support["players"].add(decision.actor_player_key)
                support["games"].add(decision.game_id)
                labels.append((decision, label, coarse_key))
                oracle_decisions += 1
                oracle_queries += label.query_count
                a_pos_counts[label.a_pos_cardinality] += 1
                modifiable = label.a_pos_cardinality > 1
                supported = coarse_key in supported_coarse
                by_phase_color_tier[
                    (
                        get_game_phase(decision.board, decision.board.turn),
                        decision.board.turn,
                        label.parent_tier,
                        "modifiable" if modifiable else "forced",
                    )
                ] += 1
                game_modifiable = game_modifiable or modifiable
                game_supported_modifiable = game_supported_modifiable or (
                    modifiable and supported
                )
                if label.chosen_preserves_tier:
                    preserving_decisions += 1
                    if label.within_tier_full_regret:
                        within_tier_regret += 1
            if game_modifiable:
                modifiable_reach_games += 1
            if game_supported_modifiable:
                supported_modifiable_reach_games += 1

            first: tuple[int, str] | None = None
            for index, (_decision, label, _coarse_key) in enumerate(labels):
                if WDL_RANK[label.chosen_tier] < WDL_RANK[label.parent_tier]:
                    transition = f"{label.parent_tier}->{label.chosen_tier}"
                    if transition not in {"W->D", "W->L", "D->L"}:
                        raise F0H0Error("unexpected positional downgrade transition")
                    first = index, transition
                    break
            steering = "none"
            if first is not None:
                first_downgrade_games += 1
                index, transition = first
                first_downgrades[transition] += 1
                if index > 0:
                    _previous_decision, previous, previous_coarse = labels[index - 1]
                    if previous.chosen_preserves_tier and previous.a_pos_cardinality > 1:
                        if previous_coarse in supported_coarse:
                            steerable_supported_games += 1
                            steering = "supported_positional_predecessor"
                        else:
                            steerable_unsupported_games += 1
                            steering = "unsupported_positional_predecessor_abstain"
            component = component_by_session[session]
            interval_observations["modifiable_reach"].append(
                (component, game_modifiable)
            )
            interval_observations["supported_modifiable_reach"].append(
                (component, game_supported_modifiable)
            )
            interval_observations["first_downgrade"].append(
                (component, first is not None)
            )
            interval_observations["supported_steerable"].append(
                (component, steering == "supported_positional_predecessor")
            )
            if first is not None:
                interval_observations["steerable_given_first_downgrade"].append(
                    (
                        component,
                        steering == "supported_positional_predecessor",
                    )
                )
            for transition in ("W->D", "W->L", "D->L"):
                interval_observations[f"first_{transition}"].append(
                    (component, first is not None and first[1] == transition)
                )
            if record.outcome_eligible:
                if record.recorded_outcome not in {"W", "B", "D"}:
                    raise F0H0Error("strict factual outcome is unavailable")
                factual_sample_outcomes[record.recorded_outcome] += 1
                factual_modifiable_reach += int(game_modifiable)
                factual_supported_modifiable_reach += int(
                    game_supported_modifiable
                )
                is_factual_supported_steerable = (
                    steering == "supported_positional_predecessor"
                )
                factual_supported_steerable += int(
                    is_factual_supported_steerable
                )
                if is_factual_supported_steerable:
                    factual_supported_steerable_outcomes[
                        record.recorded_outcome
                    ] += 1
                interval_observations["factual_modifiable_reach"].append(
                    (component, game_modifiable)
                )
                interval_observations[
                    "factual_supported_modifiable_reach"
                ].append((component, game_supported_modifiable))
                interval_observations["factual_supported_steerable"].append(
                    (component, is_factual_supported_steerable)
                )
            per_game_facts.append(
                {
                    "session_id_sha256": hashlib.sha256(
                        f"f0-h0-game-report-v1\0{session}".encode("utf-8")
                    ).hexdigest(),
                    "first_downgrade": first[1] if first is not None else None,
                    "modifiable_reach": game_modifiable,
                    "supported_modifiable_reach": game_supported_modifiable,
                    "steering_class": steering,
                    "strict_outcome_eligible": record.outcome_eligible,
                }
            )
    finally:
        database.close()

    exact_summary = _support_summary(
        exact_support,
        min_players=int(support_spec["exact_minimum_players"]),
        min_games=int(support_spec["exact_minimum_games"]),
    )
    exact_summary.pop("supported_keys")
    sample_games = len(cost["analysis_game_membership"])
    factual_games = sum(factual_sample_outcomes.values())
    modifiable_interval = component_robust_interval(
        interval_observations["modifiable_reach"]
    )
    supported_modifiable_interval = component_robust_interval(
        interval_observations["supported_modifiable_reach"]
    )
    first_downgrade_interval = component_robust_interval(
        interval_observations["first_downgrade"]
    )
    steerable_interval = component_robust_interval(
        interval_observations["supported_steerable"]
    )
    steerable_given_first_interval = (
        component_robust_interval(
            interval_observations["steerable_given_first_downgrade"]
        )
        if first_downgrade_games
        else None
    )
    transition_intervals = {
        transition: component_robust_interval(
            interval_observations[f"first_{transition}"]
        )
        for transition in ("W->D", "W->L", "D->L")
    }
    factual_modifiable_interval = (
        component_robust_interval(
            interval_observations["factual_modifiable_reach"]
        )
        if factual_games
        else None
    )
    factual_supported_interval = (
        component_robust_interval(
            interval_observations["factual_supported_modifiable_reach"]
        )
        if factual_games
        else None
    )
    factual_supported_steerable_interval = (
        component_robust_interval(
            interval_observations["factual_supported_steerable"]
        )
        if factual_games
        else None
    )

    all_outcome_records = [
        record
        for record in analysis_records
        if record.outcome_eligible
    ]
    all_outcomes = Counter(record.recorded_outcome for record in all_outcome_records)
    all_outcome_players = {
        player for record in all_outcome_records for player in record.player_keys
    }
    analysis_component_games = Counter(
        component_by_session[record.session_id] for record in analysis_records
    )
    analysis_component_metrics = concentration(
        list(analysis_component_games.values())
    )
    thresholds = plan["thresholds"]
    concentration_metrics = {
        "players": concentration(list(player_decisions.values())),
        "games": concentration(list(game_decisions.values())),
        "coarse_state_classes": concentration(
            [int(value["decisions"]) for value in coarse_support.values()]
        ),
        "sampled_exact_positional_states": concentration(
            [int(value["decisions"]) for value in exact_support.values()]
        ),
    }
    gates = {
        "analysis_independent_components": (
            len(analysis_component_games)
            >= int(thresholds["dependence"]["minimum_analysis_components"])
        ),
        "analysis_largest_component_game_share": (
            analysis_component_metrics["maximum_share"]
            <= float(
                thresholds["dependence"][
                    "maximum_analysis_component_game_share"
                ]
            )
        ),
        "oracle_independent_components": (
            supported_modifiable_interval["independent_components"]
            >= int(thresholds["dependence"]["minimum_oracle_components"])
        ),
        "oracle_largest_component_game_share": (
            supported_modifiable_interval["largest_component_game_share"]
            <= float(
                thresholds["dependence"][
                    "maximum_oracle_component_game_share"
                ]
            )
        ),
        "coarse_supported_decision_fraction": (
            coarse_summary["supported_decision_fraction"]
            >= float(support_spec["minimum_supported_decision_fraction"])
        ),
        "modifiable_state_fraction": (
            sum(count for cardinality, count in a_pos_counts.items() if cardinality > 1)
            / oracle_decisions
            >= float(thresholds["reachability"]["minimum_modifiable_state_fraction"])
        ),
        "supported_modifiable_game_reach_lcb": (
            supported_modifiable_interval["lower_95"]
            >= float(thresholds["reachability"]["minimum_supported_game_reach_lcb"])
        ),
        "player_top_1_percent_share": (
            concentration_metrics["players"]["top_1_percent_share"]
            <= float(thresholds["concentration"]["maximum_top_1_percent_share"])
        ),
        "player_gini": (
            concentration_metrics["players"]["gini"]
            <= float(thresholds["concentration"]["maximum_player_gini"])
        ),
        "player_kish_effective_units": (
            concentration_metrics["players"]["kish_effective_units"]
            >= float(thresholds["concentration"]["minimum_player_kish_ess"])
        ),
        "product_effect_upper_bound": (
            factual_supported_steerable_interval is not None
            and factual_supported_steerable_interval["upper_95"]
            >= float(thresholds["product_effect"]["minimum_signable_score_effect"])
        ),
    }
    decision = "未被否决" if all(gates.values()) else "触发停止条件"
    ruleset = Path(ruleset_path)
    ruleset_sha = sha256_file(ruleset)
    result = {
        "schema_version": RESULT_SCHEMA,
        "screen_id": plan["screen_id"],
        "status": "completed_read_only_rejection_screen",
        "decision": decision,
        "gate_results": gates,
        "claim_boundary": {
            "source_domain": "observed PlayOK-like source only",
            "labels": "positional-only",
            "safe_set_name": "A_pos",
            "not_a_allow": True,
            "full_rule_safety_claim": False,
            "cannot_approve_e0_or_later": True,
        },
        "lineage": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha256,
            "split_identity": split["split_identity"],
            "split_file_sha256": split_file_sha256,
            "cost_decision_identity": cost["cost_decision_identity"],
            "cost_file_sha256": cost_file_sha256,
            "f0d0_corpus_identity": EXPECTED_CORPUS_IDENTITY,
            "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
            "f0d0_manifest_file_sha256": boundary.file_sha256,
        },
        "input_files": {
            "raw_files_opened": sorted(raw_inputs, key=lambda item: item["relative_path"]),
            "raw_files_opened_count": len(raw_inputs),
            "raw_file_inventory_inherited_from_f0d0": True,
            "malom_manifest": {
                "relative_path": Path(malom_manifest_path).as_posix(),
                "sha256": sha256_file(malom_manifest_path),
                "snapshot": snapshot,
            },
            "ruleset": {
                "relative_path": Path(ruleset_path).as_posix(),
                "sha256": ruleset_sha,
            },
        },
        "access_audit": {
            "statistics_partitions": sorted(ANALYSIS_PARTITIONS),
            "one-time-confirmation_raw_record_reads": 0,
            "one-time-confirmation_statistics_reads": 0,
            "final-test_raw_record_reads": 0,
            "final-test_statistics_reads": 0,
            "source_pool_2eb04f54_records_read": 0,
        },
        "bases": {
            "behavior_global_inherited": {
                "games": 92_226,
                "logical_plies": 4_394_220,
                "player_keys": 4_994,
            },
            "behavior_train_selection": {
                "games": len(analysis_records),
                "logical_plies": sum(record.move_count for record in analysis_records),
                "player_keys": len(
                    {player for record in analysis_records for player in record.player_keys}
                ),
            },
            "oracle_analysis": {
                "mode": cost["decision"],
                "games": sample_games,
                "decisions": oracle_decisions,
                "queries": oracle_queries,
            },
            "factual_global_inherited": {
                "games": 37_866,
                "player_keys": 3_392,
                "unverifiable_result_games": 54_923,
            },
            "factual_train_selection": {
                "games": len(all_outcome_records),
                "player_keys": len(all_outcome_players),
            },
            "factual_oracle_sample": {
                "games": factual_games,
            },
        },
        "dimensions": {
            "independent_support": {
                "coarse_state_classes_full_train_selection": coarse_summary,
                "exact_positional_states_oracle_members": exact_summary,
            },
            "modifiable_state_reachability": {
                "a_pos_cardinality_counts": dict(sorted(a_pos_counts.items())),
                "modifiable_state_fraction": (
                    sum(
                        count
                        for cardinality, count in a_pos_counts.items()
                        if cardinality > 1
                    )
                    / oracle_decisions
                ),
                "game_reach": modifiable_interval,
                "supported_game_reach": supported_modifiable_interval,
                "phase_color_tier_counts": [
                    {
                        "phase": key[0],
                        "color": key[1],
                        "tier": key[2],
                        "choice_class": key[3],
                        "decisions": count,
                    }
                    for key, count in sorted(by_phase_color_tier.items())
                ],
            },
            "concentration": concentration_metrics,
            "player_network_component_dependence": {
                "full_train_selection": analysis_component_metrics,
                "interval_method": supported_modifiable_interval["method"],
                "thresholds": thresholds["dependence"],
            },
            "product_effect_upper_bound": {
                "signed_product_effect": thresholds["product_effect"],
                "mechanism": {
                    "denominator_scope": "oracle analysis games",
                    "first_theory_downgrade": first_downgrade_interval,
                    "first_transition_counts": {
                        transition: first_downgrades.get(transition, 0)
                        for transition in ("W->D", "W->L", "D->L")
                    },
                    "first_transition_intervals": transition_intervals,
                    "supported_steerable_predecessor": steerable_interval,
                    "supported_steerable_share_given_first_downgrade": (
                        steerable_given_first_interval
                    ),
                    "unsupported_steerable_predecessor_abstentions": (
                        steerable_unsupported_games
                    ),
                    "coarse_whole_game_score_bound": supported_modifiable_interval,
                    "maximum_per_game_score_swing": 1.0,
                },
                "factual": {
                    "denominator_scope": "strict factual result subset only",
                    "full_train_selection_outcome_counts": {
                        outcome: all_outcomes.get(outcome, 0)
                        for outcome in ("W", "B", "D")
                    },
                    "oracle_sample_outcome_counts": {
                        outcome: factual_sample_outcomes.get(outcome, 0)
                        for outcome in ("W", "B", "D")
                    },
                    "modifiable_game_reach": factual_modifiable_interval,
                    "supported_modifiable_game_reach_score_bound": (
                        factual_supported_interval
                    ),
                    "supported_steerable_game_score_bound": (
                        factual_supported_steerable_interval
                    ),
                    "supported_steerable_outcome_counts": {
                        outcome: factual_supported_steerable_outcomes.get(
                            outcome, 0
                        )
                        for outcome in ("W", "B", "D")
                    },
                    "maximum_per_game_score_swing": 1.0,
                },
            },
        },
        "secondary_difficulty": {
            "within_tier_complete_comparator_regret_decisions": within_tier_regret,
            "preserving_decisions": preserving_decisions,
            "rate_among_preserving": (
                within_tier_regret / preserving_decisions
                if preserving_decisions
                else None
            ),
            "not_combined_with_wdl_loss": True,
        },
        "known_biases": {
            "history_attrition_nonrandom": {
                "excluded_games": 1_751,
                "excluded_draws": 35,
                "retained_history_games": 92_789,
                "retained_draws": 26_157,
            },
            "missing_conditions": [
                "UI orientation",
                "time control",
                "exact source rules variant",
                "explicit import batch",
                "upstream source-file identity",
            ],
            "transport": "no inference to product UI, other time controls, or new people",
            "unverifiable_result_games": 54_923,
        },
        "per_oracle_game": per_game_facts,
        "prohibited_operations_observed": {
            "games_started": 0,
            "search_batches_started": 0,
            "models_loaded": 0,
            "training_updates": 0,
            "database_writes": 0,
            "source_pool_records_consumed": 0,
        },
    }
    return result
