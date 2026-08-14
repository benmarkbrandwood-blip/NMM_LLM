"""Corrected read-only F0-H0 split-feasibility measurements.

The v1 screen incorrectly required every game of every connected player to
remain in one partition.  This module measures cut-tolerant alternatives.  It
does not select a final split, query Malom, open a model, or compute any F0-H0
scientific endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx

from game.board import BoardState
from game.draw_rules import StandardDrawTracker
from game.rules import get_all_legal_moves, get_game_phase, terminal_result
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen


PLAN_SCHEMA = "nmm.f0-h0-corrected-split-measurement-plan.v1"
RESULT_SCHEMA = "nmm.f0-h0-corrected-split-measurement-result.v1"
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
EXPECTED_V1_PLAN_IDENTITY = (
    "95a802625867906ab453ed7a52bbba1e0202b08473b10f897ba81c87fb59d530"
)
EXPECTED_V1_SPLIT_IDENTITY = (
    "e41da5fbf1a2ba60441273664c6834dafcb54bcb79d541f3349a948f7cac5dd4"
)
EXPECTED_V1_RESULT_IDENTITY = (
    "714627f8be20bc45a267c97752171644040fc1273a24f82a570a7cb83512fe82"
)

PARTITIONS = (
    "train",
    "selection",
    "one-time-confirmation",
    "final-test",
)
PAIR_INDICES = tuple(
    (left, right)
    for left in range(len(PARTITIONS))
    for right in range(left + 1, len(PARTITIONS))
)
DISTANCE_BUCKETS = ("1", "2", "3-4", "5-8", "9-16", "17+")


class SplitRetestError(RuntimeError):
    """Raised when a required measurement input or invariant differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SplitRetestError("payload cannot be canonicalized") from exc
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
        raise SplitRetestError(f"invalid JSON input: {source}") from exc
    if not isinstance(value, dict):
        raise SplitRetestError(f"JSON input is not an object: {source}")
    return value, raw


def _load_sealed_json(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    value, raw = _load_json(path)
    if value.get("schema_version") != schema:
        raise SplitRetestError(f"schema differs for {Path(path)}")
    recorded = value.get(identity_field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise SplitRetestError(f"identity is absent for {Path(path)}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != recorded:
        raise SplitRetestError(f"identity differs for {Path(path)}")
    return value, hashlib.sha256(raw).hexdigest()


def write_sealed_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise SplitRetestError(f"refusing to overwrite evidence: {target}")
    body = dict(payload)
    body.pop(identity_field, None)
    sealed = {**body, identity_field: canonical_sha256(body)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(sealed))
    return sealed


def _table_value(table: Sequence[Any], index: Any, *, field: str) -> Any:
    if isinstance(index, bool) or not isinstance(index, int):
        raise SplitRetestError(f"encoded {field} index is invalid")
    try:
        return table[index]
    except IndexError as exc:
        raise SplitRetestError(f"encoded {field} index is out of range") from exc


@dataclass(frozen=True)
class GameRecord:
    session_id: str
    canonical_file: str
    file_sha256: str
    file_size: int
    played_on: date
    move_count: int
    white_player: str
    black_player: str

    @property
    def players(self) -> tuple[str, str]:
        return self.white_player, self.black_player


@dataclass(frozen=True)
class Boundary:
    manifest_file_sha256: str
    manifest_identity: str
    corpus_identity: str
    games: tuple[GameRecord, ...]
    raw_subset_identity: str


def load_boundary(path: str | Path) -> Boundary:
    """Verify and decode only the F0-D0 behavior-eligible source boundary."""
    manifest, raw = _load_json(path)
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != EXPECTED_F0D0_FILE_SHA256:
        raise SplitRetestError("F0-D0 manifest file SHA-256 differs")
    if manifest.get("schema_version") != F0D0_SCHEMA:
        raise SplitRetestError("F0-D0 manifest schema differs")
    body = dict(manifest)
    recorded_manifest_identity = body.pop("manifest_identity", None)
    if (
        recorded_manifest_identity != EXPECTED_F0D0_MANIFEST_IDENTITY
        or canonical_sha256(body) != EXPECTED_F0D0_MANIFEST_IDENTITY
    ):
        raise SplitRetestError("F0-D0 manifest identity differs")
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or identities.get(
        "corpus_identity"
    ) != EXPECTED_CORPUS_IDENTITY:
        raise SplitRetestError("F0-D0 corpus identity differs")

    input_encoding = manifest.get("input_file_encoding")
    input_rows = manifest.get("input_files")
    if not isinstance(input_encoding, Mapping) or not isinstance(input_rows, list):
        raise SplitRetestError("F0-D0 input encoding is absent")
    if input_encoding.get("fields") != [
        "relative_path",
        "role_index",
        "byte_length",
        "sha256",
        "session_id",
        "status_index",
        "failure_index",
    ]:
        raise SplitRetestError("F0-D0 input fields differ")
    roles = input_encoding.get("role_values")
    if not isinstance(roles, list):
        raise SplitRetestError("F0-D0 role table is absent")
    raw_inputs: dict[str, tuple[int, str]] = {}
    raw_identity_rows: list[dict[str, Any]] = []
    for row in input_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise SplitRetestError("F0-D0 input row width differs")
        if _table_value(roles, row[1], field="role") is not None:
            continue
        relative_path, byte_length, digest = row[0], row[2], row[3]
        if (
            not isinstance(relative_path, str)
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise SplitRetestError("F0-D0 raw input row is invalid")
        raw_inputs[relative_path] = byte_length, digest
        raw_identity_rows.append(
            {
                "relative_path": relative_path,
                "byte_length": byte_length,
                "sha256": digest,
            }
        )
    if canonical_sha256(raw_identity_rows) != identities.get("raw_files_identity"):
        raise SplitRetestError("F0-D0 raw-file identity differs")

    encoding = manifest.get("record_encoding")
    rows = manifest.get("game_records")
    if not isinstance(encoding, Mapping) or not isinstance(rows, list):
        raise SplitRetestError("F0-D0 record encoding is absent")
    fields = encoding.get("fields")
    players = encoding.get("player_keys")
    if not isinstance(fields, list) or not isinstance(players, list):
        raise SplitRetestError("F0-D0 record tables are absent")
    positions = {field: index for index, field in enumerate(fields)}
    required = {
        "session_id",
        "canonical_file",
        "date",
        "move_count",
        "player_key_indices",
        "behavior_replay_eligible",
    }
    if not required.issubset(positions):
        raise SplitRetestError("F0-D0 record fields are incomplete")

    games: list[GameRecord] = []
    raw_subset_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise SplitRetestError("F0-D0 game row width differs")
        if not row[positions["behavior_replay_eligible"]]:
            continue
        session_id = row[positions["session_id"]]
        canonical_file = row[positions["canonical_file"]]
        played_on_raw = row[positions["date"]]
        move_count = row[positions["move_count"]]
        player_indices = row[positions["player_key_indices"]]
        if (
            not isinstance(session_id, str)
            or not isinstance(canonical_file, str)
            or not isinstance(played_on_raw, str)
            or isinstance(move_count, bool)
            or not isinstance(move_count, int)
            or move_count <= 0
            or not isinstance(player_indices, list)
            or len(player_indices) != 2
            or canonical_file not in raw_inputs
        ):
            raise SplitRetestError("F0-D0 behavior row is invalid")
        try:
            played_on = date.fromisoformat(played_on_raw)
        except ValueError as exc:
            raise SplitRetestError("F0-D0 behavior date is invalid") from exc
        decoded_players = tuple(
            _table_value(players, index, field="player key")
            for index in player_indices
        )
        if any(not isinstance(player, str) for player in decoded_players):
            raise SplitRetestError("F0-D0 player key is invalid")
        file_size, digest = raw_inputs[canonical_file]
        games.append(
            GameRecord(
                session_id=session_id,
                canonical_file=canonical_file,
                file_sha256=digest,
                file_size=file_size,
                played_on=played_on,
                move_count=move_count,
                white_player=decoded_players[0],
                black_player=decoded_players[1],
            )
        )
        raw_subset_rows.append([session_id, canonical_file, file_size, digest])
    games.sort(key=lambda game: game.session_id)
    raw_subset_rows.sort()
    unique_players = {player for game in games for player in game.players}
    if (
        len(games) != 92_226
        or sum(game.move_count for game in games) != 4_394_220
        or len(unique_players) != 4_994
    ):
        raise SplitRetestError("F0-D0 behavior base differs")
    return Boundary(
        manifest_file_sha256=file_sha,
        manifest_identity=recorded_manifest_identity,
        corpus_identity=EXPECTED_CORPUS_IDENTITY,
        games=tuple(games),
        raw_subset_identity=canonical_sha256(raw_subset_rows),
    )


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = _load_sealed_json(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    boundary = plan.get("input_boundary", {})
    if boundary != {
        "corpus_identity": EXPECTED_CORPUS_IDENTITY,
        "manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
        "behavior_games": 92_226,
        "behavior_logical_plies": 4_394_220,
        "behavior_player_keys": 4_994,
    }:
        raise SplitRetestError("corrected plan input boundary differs")
    if plan.get("scope", {}).get("select_final_design") is not False:
        raise SplitRetestError("corrected plan selects a final design")
    if plan.get("prohibited_operations", {}).get("malom_queries") is not True:
        raise SplitRetestError("corrected plan does not prohibit Malom")
    return plan, file_sha


@dataclass
class GraphData:
    graph: nx.Graph
    edge_games: Counter[tuple[str, str]]
    player_games: Counter[str]
    opponents: dict[str, set[str]]
    self_games: int


def build_player_graph(games: Sequence[GameRecord]) -> GraphData:
    edge_games: Counter[tuple[str, str]] = Counter()
    player_games: Counter[str] = Counter()
    opponents: dict[str, set[str]] = defaultdict(set)
    self_games = 0
    for game in games:
        white, black = game.players
        for player in set(game.players):
            player_games[player] += 1
        if white == black:
            self_games += 1
            edge_games[(white, black)] += 1
            continue
        edge = tuple(sorted((white, black)))
        edge_games[edge] += 1
        opponents[white].add(black)
        opponents[black].add(white)
    graph = nx.Graph()
    graph.add_nodes_from(sorted(player_games))
    for (left, right), count in sorted(edge_games.items()):
        if left != right:
            graph.add_edge(left, right, games=count)
    return GraphData(
        graph=graph,
        edge_games=edge_games,
        player_games=player_games,
        opponents=opponents,
        self_games=self_games,
    )


def nearest_rank_quantiles(values: Sequence[int]) -> dict[str, int]:
    if not values:
        raise SplitRetestError("quantile input is empty")
    ordered = sorted(int(value) for value in values)

    def at(probability: float) -> int:
        rank = max(1, math.ceil(probability * len(ordered)))
        return ordered[rank - 1]

    return {
        "min": ordered[0],
        "p01": at(0.01),
        "p05": at(0.05),
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _tail_counts(values: Iterable[int], thresholds: Sequence[int]) -> list[dict[str, int]]:
    materialized = list(values)
    return [
        {
            "at_least": threshold,
            "players": sum(value >= threshold for value in materialized),
        }
        for threshold in thresholds
    ]


def graph_structure(data: GraphData, games: Sequence[GameRecord]) -> dict[str, Any]:
    components = [set(component) for component in nx.connected_components(data.graph)]
    component_games: Counter[str] = Counter()
    component_players: dict[str, set[str]] = {}
    player_component: dict[str, str] = {}
    for players in components:
        identity = canonical_sha256(sorted(players))
        component_players[identity] = players
        for player in players:
            player_component[player] = identity
    for game in games:
        identity = player_component[game.white_player]
        if player_component[game.black_player] != identity:
            raise SplitRetestError("connected-component assignment differs")
        component_games[identity] += 1
    rows = []
    for identity, players in component_players.items():
        subgraph = data.graph.subgraph(players)
        rows.append(
            {
                "component_identity": identity,
                "player_keys": len(players),
                "games": component_games[identity],
                "unique_opponent_pairs": subgraph.number_of_edges(),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["player_keys"]),
            -int(row["games"]),
            str(row["component_identity"]),
        )
    )
    opponent_counts = [len(data.opponents[player]) for player in data.graph]
    game_counts = [data.player_games[player] for player in data.graph]
    return {
        "player_keys": data.graph.number_of_nodes(),
        "games": len(games),
        "unique_opponent_pairs": data.graph.number_of_edges(),
        "self_games": data.self_games,
        "connected_components": len(rows),
        "giant_component": rows[0],
        "non_giant_components": rows[1:],
        "degree": {
            "distinct_opponents": {
                "quantiles": nearest_rank_quantiles(opponent_counts),
                "tail": _tail_counts(opponent_counts, (5, 10, 25, 50, 100, 250)),
            },
            "games_per_player": {
                "quantiles": nearest_rank_quantiles(game_counts),
                "tail": _tail_counts(game_counts, (10, 25, 50, 100, 250, 500, 1000)),
            },
        },
    }


def detect_communities(
    data: GraphData,
    *,
    seed: int,
    resolution: float,
) -> tuple[list[set[str]], dict[str, Any]]:
    communities = [
        set(community)
        for community in nx.community.louvain_communities(
            data.graph,
            weight="games",
            resolution=resolution,
            threshold=1e-7,
            seed=seed,
        )
    ]
    communities.sort(key=lambda value: (-len(value), canonical_sha256(sorted(value))))
    membership: dict[str, int] = {}
    for index, community in enumerate(communities):
        for player in community:
            if player in membership:
                raise SplitRetestError("Louvain communities overlap")
            membership[player] = index
    if set(membership) != set(data.graph):
        raise SplitRetestError("Louvain communities do not cover the graph")
    internal_games = [0] * len(communities)
    boundary_games = [0] * len(communities)
    internal_edges = [0] * len(communities)
    for (left, right), count in data.edge_games.items():
        left_index = membership[left]
        right_index = membership[right]
        if left_index == right_index:
            internal_games[left_index] += count
            if left != right:
                internal_edges[left_index] += 1
        else:
            boundary_games[left_index] += count
            boundary_games[right_index] += count
    rows = [
        {
            "community_index": index,
            "community_identity": canonical_sha256(sorted(community)),
            "player_keys": len(community),
            "internal_games": internal_games[index],
            "internal_unique_opponent_pairs": internal_edges[index],
            "boundary_games": boundary_games[index],
        }
        for index, community in enumerate(communities)
    ]
    return communities, {
        "method": "networkx-louvain",
        "networkx_version": nx.__version__,
        "seed": seed,
        "resolution": resolution,
        "weight": "game count per unique player pair",
        "communities": len(communities),
        "modularity": nx.community.modularity(
            data.graph,
            communities,
            weight="games",
            resolution=resolution,
        ),
        "player_size_quantiles": nearest_rank_quantiles(
            [len(community) for community in communities]
        ),
        "community_rows": rows,
    }


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def time_structure(
    games: Sequence[GameRecord],
    *,
    cut_dates: Sequence[str],
) -> dict[str, Any]:
    weekly_games: Counter[date] = Counter()
    weekly_players: dict[date, set[str]] = defaultdict(set)
    for game in games:
        week = _week_start(game.played_on)
        weekly_games[week] += 1
        weekly_players[week].update(game.players)
    weekly = [
        {
            "week_start": week.isoformat(),
            "games": weekly_games[week],
            "player_keys": len(weekly_players[week]),
        }
        for week in sorted(weekly_games)
    ]
    cuts = []
    for raw_cut in cut_dates:
        try:
            cut = date.fromisoformat(raw_cut)
        except ValueError as exc:
            raise SplitRetestError("candidate time cut is invalid") from exc
        pre_games = [game for game in games if game.played_on < cut]
        post_games = [game for game in games if game.played_on >= cut]
        pre_players = {player for game in pre_games for player in game.players}
        post_players = {player for game in post_games for player in game.players}
        pre_only = pre_players - post_players
        post_only = post_players - pre_players
        spanning = pre_players & post_players
        player_classes = {
            "pre_only": pre_only,
            "post_only": post_only,
            "spanning": spanning,
        }
        touching = {
            label: sum(bool(set(game.players) & members) for game in games)
            for label, members in player_classes.items()
        }
        incidences = {
            label: sum(
                sum(player in members for player in set(game.players))
                for game in games
            )
            for label, members in player_classes.items()
        }
        strongest = [
            game
            for game in post_games
            if set(game.players).issubset(post_only)
        ]
        strongest_players = {
            player for game in strongest for player in game.players
        }
        cuts.append(
            {
                "cut_date": cut.isoformat(),
                "pre_games": len(pre_games),
                "post_games_time_only": len(post_games),
                "post_players_time_only": len(post_players),
                "player_classes": {
                    label: len(members)
                    for label, members in player_classes.items()
                },
                "games_touching_player_class": touching,
                "player_game_incidences": incidences,
                "post_games_both_players_unseen_pre_cut": len(strongest),
                "players_in_strong_post_games": len(strongest_players),
                "strong_post_membership_identity": canonical_sha256(
                    sorted(game.session_id for game in strongest)
                ),
            }
        )
    return {
        "date_min": min(game.played_on for game in games).isoformat(),
        "date_max": max(game.played_on for game in games).isoformat(),
        "weekly": weekly,
        "candidate_cuts": cuts,
    }


def _hash_rank(namespace: str, seed: str, value: str) -> str:
    return hashlib.sha256(
        f"{namespace}\0{seed}\0{value}".encode("utf-8")
    ).hexdigest()


def _initial_holdout(
    nodes: set[str],
    communities: Sequence[set[str]],
    *,
    target: int,
    seed: str,
    restart: int,
) -> set[str]:
    ranked_communities = sorted(
        communities,
        key=lambda community: _hash_rank(
            "f0-h0-a-community-order-v1",
            f"{seed}:{restart}",
            canonical_sha256(sorted(community)),
        ),
    )
    selected: set[str] = set()
    for community in ranked_communities:
        if len(selected) + len(community) <= target:
            selected.update(community)
    if len(selected) < target:
        remaining = sorted(
            nodes - selected,
            key=lambda player: (
                _hash_rank(
                    "f0-h0-a-player-fill-v1",
                    f"{seed}:{restart}",
                    player,
                ),
                player,
            ),
        )
        selected.update(remaining[: target - len(selected)])
    if len(selected) != target:
        raise SplitRetestError("design A initialization has wrong size")
    return selected


def _cut_games(data: GraphData, holdout: set[str]) -> int:
    return sum(
        count
        for (left, right), count in data.edge_games.items()
        if (left in holdout) != (right in holdout)
    )


def _classify_games(
    games: Sequence[GameRecord],
    holdout: set[str],
) -> dict[str, int]:
    counts = Counter()
    for game in games:
        white_holdout = game.white_player in holdout
        black_holdout = game.black_player in holdout
        if white_holdout and black_holdout:
            counts["holdout_internal"] += 1
        elif not white_holdout and not black_holdout:
            counts["train_internal"] += 1
        else:
            counts["cross_cut_discard"] += 1
    return {
        "holdout_internal": counts["holdout_internal"],
        "train_internal": counts["train_internal"],
        "cross_cut_discard": counts["cross_cut_discard"],
    }


def measure_design_a(
    data: GraphData,
    games: Sequence[GameRecord],
    communities: Sequence[set[str]],
    *,
    targets: Sequence[float],
    seed: str,
    restarts: int,
    maximum_kl_iterations: int,
) -> list[dict[str, Any]]:
    nodes = set(data.graph)
    results = []
    for target_ratio in targets:
        target_players = round(len(nodes) * target_ratio)
        candidates = []
        for restart in range(restarts):
            initial = _initial_holdout(
                nodes,
                communities,
                target=target_players,
                seed=seed,
                restart=restart,
            )
            left, right = nx.community.kernighan_lin_bisection(
                data.graph,
                partition=(initial, nodes - initial),
                max_iter=maximum_kl_iterations,
                weight="games",
                seed=restart,
            )
            if len(left) == target_players:
                holdout = set(left)
            elif len(right) == target_players:
                holdout = set(right)
            else:
                raise SplitRetestError("Kernighan-Lin changed partition size")
            membership_identity = canonical_sha256(sorted(holdout))
            candidates.append(
                {
                    "restart": restart,
                    "holdout": holdout,
                    "cut_games": _cut_games(data, holdout),
                    "membership_identity": membership_identity,
                }
            )
        chosen = min(
            candidates,
            key=lambda item: (
                int(item["cut_games"]),
                str(item["membership_identity"]),
            ),
        )
        holdout = chosen["holdout"]
        classified = _classify_games(games, holdout)
        if classified["cross_cut_discard"] != chosen["cut_games"]:
            raise SplitRetestError("design A cut accounting differs")
        community_classes = Counter()
        for community in communities:
            selected = len(community & holdout)
            if selected == 0:
                community_classes["train_only"] += 1
            elif selected == len(community):
                community_classes["holdout_only"] += 1
            else:
                community_classes["split"] += 1
        results.append(
            {
                "target_holdout_player_fraction": target_ratio,
                "target_holdout_players": target_players,
                "measured_holdout_players": len(holdout),
                "measured_holdout_player_fraction": len(holdout) / len(nodes),
                "holdout_internal_games": classified["holdout_internal"],
                "train_internal_games": classified["train_internal"],
                "cross_cut_discard_games": classified["cross_cut_discard"],
                "cross_cut_discard_fraction": (
                    classified["cross_cut_discard"] / len(games)
                ),
                "holdout_internal_game_fraction": (
                    classified["holdout_internal"] / len(games)
                ),
                "train_internal_game_fraction": (
                    classified["train_internal"] / len(games)
                ),
                "candidate_membership_identity": chosen["membership_identity"],
                "selected_restart_for_measurement": chosen["restart"],
                "restart_cut_game_counts": [
                    {
                        "restart": item["restart"],
                        "cut_games": item["cut_games"],
                    }
                    for item in candidates
                ],
                "louvain_community_disposition": dict(community_classes),
            }
        )
    return results


def _largest_remainder_counts(
    total: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    raw = {partition: total * float(ratios[partition]) for partition in PARTITIONS}
    counts = {partition: math.floor(raw[partition]) for partition in PARTITIONS}
    remaining = total - sum(counts.values())
    ranked = sorted(
        PARTITIONS,
        key=lambda partition: (-(raw[partition] - counts[partition]), PARTITIONS.index(partition)),
    )
    for partition in ranked[:remaining]:
        counts[partition] += 1
    return counts


def decision_player_partition(
    players: Iterable[str],
    *,
    ratios: Mapping[str, float],
    seed: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    values = sorted(set(players))
    counts = _largest_remainder_counts(len(values), ratios)
    ranked = sorted(
        values,
        key=lambda player: (
            _hash_rank("f0-h0-c-player-rank-v1", seed, player),
            player,
        ),
    )
    membership: dict[str, str] = {}
    offset = 0
    for partition in PARTITIONS:
        next_offset = offset + counts[partition]
        for player in ranked[offset:next_offset]:
            membership[player] = partition
        offset = next_offset
    if len(membership) != len(values):
        raise SplitRetestError("design C player partition is incomplete")
    rows = [[player, membership[player]] for player in sorted(membership)]
    return membership, {
        "ratios": dict(ratios),
        "seed": seed,
        "player_counts": counts,
        "membership_identity": canonical_sha256(rows),
    }


def measure_design_c_counts(
    games: Sequence[GameRecord],
    membership: Mapping[str, str],
) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    game_sets: dict[str, set[str]] = {partition: set() for partition in PARTITIONS}
    cross_game_pairs: Counter[tuple[str, str]] = Counter()
    same_partition_games = 0
    for game in games:
        white_partition = membership[game.white_player]
        black_partition = membership[game.black_player]
        white_decisions = (game.move_count + 1) // 2
        black_decisions = game.move_count // 2
        decision_counts[white_partition] += white_decisions
        decision_counts[black_partition] += black_decisions
        if white_decisions:
            game_sets[white_partition].add(game.session_id)
        if black_decisions:
            game_sets[black_partition].add(game.session_id)
        if white_partition == black_partition:
            same_partition_games += 1
        else:
            pair = tuple(sorted((white_partition, black_partition), key=PARTITIONS.index))
            cross_game_pairs[pair] += 1
    return {
        "partition_counts": {
            partition: {
                "player_keys": sum(value == partition for value in membership.values()),
                "decisions": decision_counts[partition],
                "games_with_at_least_one_decision": len(game_sets[partition]),
            }
            for partition in PARTITIONS
        },
        "same_partition_games": same_partition_games,
        "cross_partition_games": len(games) - same_partition_games,
        "cross_partition_game_pairs": [
            {
                "left": pair[0],
                "right": pair[1],
                "games": count,
            }
            for pair, count in sorted(
                cross_game_pairs.items(),
                key=lambda item: (
                    PARTITIONS.index(item[0][0]),
                    PARTITIONS.index(item[0][1]),
                ),
            )
        ],
    }


def _move_notation(move: Mapping[str, Any]) -> str:
    source = move.get("from")
    target = move.get("to")
    base = str(target) if source is None else f"{source}-{target}"
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


def _read_raw_game(root: Path, game: GameRecord) -> Mapping[str, Any]:
    path = root / game.canonical_file
    raw = path.read_bytes()
    if len(raw) != game.file_size:
        raise SplitRetestError(f"raw game size differs: {game.canonical_file}")
    if hashlib.sha256(raw).hexdigest() != game.file_sha256:
        raise SplitRetestError(f"raw game SHA-256 differs: {game.canonical_file}")
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SplitRetestError(f"raw game framing differs: {game.canonical_file}")
    try:
        payload = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitRetestError(f"raw game JSON differs: {game.canonical_file}") from exc
    if not isinstance(payload, Mapping) or payload.get("session_id") != game.session_id:
        raise SplitRetestError(f"raw game session differs: {game.canonical_file}")
    return payload


def _distance_bucket(distance: int) -> str:
    if distance <= 0:
        raise SplitRetestError("ply distance must be positive")
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if distance <= 4:
        return "3-4"
    if distance <= 8:
        return "5-8"
    if distance <= 16:
        return "9-16"
    return "17+"


def _minimum_distance(left: Sequence[int], right: Sequence[int]) -> int:
    left_index = 0
    right_index = 0
    best: int | None = None
    while left_index < len(left) and right_index < len(right):
        distance = abs(left[left_index] - right[right_index])
        if distance == 0:
            raise SplitRetestError("one decision belongs to two partitions")
        best = distance if best is None else min(best, distance)
        if left[left_index] < right[right_index]:
            left_index += 1
        else:
            right_index += 1
    if best is None:
        raise SplitRetestError("distance input is empty")
    return best


def _pair_name(left: int, right: int) -> str:
    return f"{PARTITIONS[left]}__{PARTITIONS[right]}"


def _pair_bit(left: int, right: int) -> int:
    try:
        index = PAIR_INDICES.index((left, right))
    except ValueError as exc:
        raise SplitRetestError("partition pair is invalid") from exc
    return 1 << index


_ORBIT_MASK_BITS = 4
_SAME_GAME_PAIR_BITS = 6
_COUNT_WIDTH = 24
_COUNT_START = _ORBIT_MASK_BITS + _SAME_GAME_PAIR_BITS


def _increment_orbit(value: int, partition_index: int) -> int:
    value |= 1 << partition_index
    value += 1 << (_COUNT_START + _COUNT_WIDTH * partition_index)
    return value


def _orbit_count(value: int, partition_index: int) -> int:
    return (
        value >> (_COUNT_START + _COUNT_WIDTH * partition_index)
    ) & ((1 << _COUNT_WIDTH) - 1)


def _same_game_pair_mask(value: int) -> int:
    return (value >> _ORBIT_MASK_BITS) & ((1 << _SAME_GAME_PAIR_BITS) - 1)


def _set_same_game_pair(value: int, left: int, right: int) -> int:
    return value | (_pair_bit(left, right) << _ORBIT_MASK_BITS)


def _trajectory_distance_counts(move_count: int) -> dict[str, int]:
    counts = Counter()
    for distance in range(1, move_count):
        if distance % 2 == 1:
            counts[_distance_bucket(distance)] += move_count - distance
    return {bucket: counts[bucket] for bucket in DISTANCE_BUCKETS}


def measure_design_c_ring16(
    *,
    repository_root: str | Path,
    games: Sequence[GameRecord],
    membership: Mapping[str, str],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Strictly replay all behavior games and measure ring16 split leakage."""
    root = Path(repository_root)
    partition_index = {partition: index for index, partition in enumerate(PARTITIONS)}
    orbit_values: dict[str, int] = {}
    same_game_orbit_distance: dict[str, Counter[str]] = {
        _pair_name(left, right): Counter() for left, right in PAIR_INDICES
    }
    trajectory_distance: dict[str, Counter[str]] = {
        _pair_name(left, right): Counter() for left, right in PAIR_INDICES
    }
    trajectory_cross_games: Counter[str] = Counter()
    raw_bytes = 0
    decisions = 0
    terminal_games = 0
    for game_number, game in enumerate(games, start=1):
        raw_game = _read_raw_game(root, game)
        raw_bytes += game.file_size
        moves = raw_game.get("moves")
        if not isinstance(moves, list) or len(moves) != game.move_count:
            raise SplitRetestError(f"raw move count differs: {game.session_id}")
        board = BoardState.new_game()
        tracker = StandardDrawTracker(board)
        terminal_seen = False
        game_orbits: dict[str, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for logical_ply, raw_move in enumerate(moves):
            if terminal_seen or not isinstance(raw_move, Mapping):
                raise SplitRetestError(
                    f"strict replay framing differs: {game.session_id}"
                )
            if (
                raw_move.get("board_fen_before") != board.to_fen_string()
                or raw_move.get("color") != board.turn
                or raw_move.get("turn") != logical_ply // 2 + 1
                or raw_move.get("type") != get_game_phase(board, board.turn)
            ):
                raise SplitRetestError(
                    f"strict replay metadata differs: {game.session_id}"
                )
            expected = {
                "from": raw_move.get("from"),
                "to": raw_move.get("to"),
                "capture": raw_move.get("capture"),
            }
            matches = [
                move
                for move in get_all_legal_moves(board)
                if all(move.get(field) == value for field, value in expected.items())
            ]
            if len(matches) != 1 or raw_move.get("notation") != _move_notation(
                matches[0]
            ):
                raise SplitRetestError(f"strict replay move differs: {game.session_id}")
            actor = game.white_player if board.turn == "W" else game.black_player
            actor_partition = membership[actor]
            actor_index = partition_index[actor_partition]
            orbit = ring16_canonical_fen(board.to_fen_string())
            orbit_values[orbit] = _increment_orbit(
                orbit_values.get(orbit, 0),
                actor_index,
            )
            game_orbits[orbit][actor_index].append(logical_ply)
            decisions += 1

            after = board.apply_move(matches[0])
            draw_reason = tracker.observe(board, matches[0], after)
            is_terminal, _winner, _reason = terminal_result(after)
            terminal_seen = bool(is_terminal or draw_reason is not None)
            board = after
        terminal_games += int(terminal_seen)

        for orbit, occurrences in game_orbits.items():
            present = sorted(occurrences)
            if len(present) < 2:
                continue
            value = orbit_values[orbit]
            for left_position, left in enumerate(present):
                for right in present[left_position + 1 :]:
                    value = _set_same_game_pair(value, left, right)
                    distance = _minimum_distance(
                        occurrences[left],
                        occurrences[right],
                    )
                    same_game_orbit_distance[_pair_name(left, right)][
                        _distance_bucket(distance)
                    ] += 1
            orbit_values[orbit] = value

        white_index = partition_index[membership[game.white_player]]
        black_index = partition_index[membership[game.black_player]]
        if white_index != black_index:
            left, right = sorted((white_index, black_index))
            pair = _pair_name(left, right)
            trajectory_cross_games[pair] += 1
            for bucket, count in _trajectory_distance_counts(game.move_count).items():
                trajectory_distance[pair][bucket] += count
        if progress is not None:
            progress(game_number, len(games))

    if decisions != sum(game.move_count for game in games):
        raise SplitRetestError("strict replay decision total differs")

    unique_by_partition = [0] * len(PARTITIONS)
    shared_unique_by_partition = [0] * len(PARTITIONS)
    decisions_by_partition = [0] * len(PARTITIONS)
    shared_decisions_by_partition = [0] * len(PARTITIONS)
    pair_intersections: Counter[tuple[int, int]] = Counter()
    pair_same_game: Counter[tuple[int, int]] = Counter()
    for value in orbit_values.values():
        mask = value & ((1 << _ORBIT_MASK_BITS) - 1)
        same_mask = _same_game_pair_mask(value)
        shared = mask.bit_count() > 1
        for index in range(len(PARTITIONS)):
            count = _orbit_count(value, index)
            if count:
                unique_by_partition[index] += 1
                decisions_by_partition[index] += count
                if shared:
                    shared_unique_by_partition[index] += 1
                    shared_decisions_by_partition[index] += count
        for pair_index, (left, right) in enumerate(PAIR_INDICES):
            if mask & (1 << left) and mask & (1 << right):
                pair_intersections[(left, right)] += 1
                if same_mask & (1 << pair_index):
                    pair_same_game[(left, right)] += 1

    partition_rows = []
    for index, partition in enumerate(PARTITIONS):
        partition_rows.append(
            {
                "partition": partition,
                "decisions": decisions_by_partition[index],
                "unique_ring16_orbits": unique_by_partition[index],
                "shared_ring16_orbits": shared_unique_by_partition[index],
                "unique_orbit_overlap_rate": (
                    shared_unique_by_partition[index] / unique_by_partition[index]
                    if unique_by_partition[index]
                    else 0.0
                ),
                "decisions_on_shared_ring16_orbits": (
                    shared_decisions_by_partition[index]
                ),
                "decision_weighted_overlap_rate": (
                    shared_decisions_by_partition[index]
                    / decisions_by_partition[index]
                    if decisions_by_partition[index]
                    else 0.0
                ),
            }
        )
    pair_rows = []
    for left, right in PAIR_INDICES:
        intersection = pair_intersections[(left, right)]
        union = unique_by_partition[left] + unique_by_partition[right] - intersection
        same_game = pair_same_game[(left, right)]
        pair = _pair_name(left, right)
        pair_rows.append(
            {
                "left": PARTITIONS[left],
                "right": PARTITIONS[right],
                "shared_unique_ring16_orbits": intersection,
                "jaccard": intersection / union if union else 0.0,
                "share_of_left_unique_orbits": (
                    intersection / unique_by_partition[left]
                    if unique_by_partition[left]
                    else 0.0
                ),
                "share_of_right_unique_orbits": (
                    intersection / unique_by_partition[right]
                    if unique_by_partition[right]
                    else 0.0
                ),
                "shared_orbits_seen_within_same_game": same_game,
                "shared_orbits_cross_game_only": intersection - same_game,
                "same_game_orbit_group_minimum_ply_distance": {
                    bucket: same_game_orbit_distance[pair][bucket]
                    for bucket in DISTANCE_BUCKETS
                },
                "cross_partition_trajectory_games": trajectory_cross_games[pair],
                "cross_partition_trajectory_decision_pairs_by_ply_distance": {
                    bucket: trajectory_distance[pair][bucket]
                    for bucket in DISTANCE_BUCKETS
                },
            }
        )
    return {
        "state_identity": (
            "repository ring16: D4 x abstract inner/outer-ring swap over the "
            "pre-decision NMM FEN"
        ),
        "raw_files_opened": len(games),
        "raw_bytes_read": raw_bytes,
        "strict_replayed_games": len(games),
        "strict_replayed_decisions": decisions,
        "strict_terminal_games": terminal_games,
        "unique_ring16_orbits_all_partitions": len(orbit_values),
        "partition_overlap": partition_rows,
        "pairwise_overlap": pair_rows,
    }


def _verify_sealed_identity(
    path: Path,
    *,
    identity_field: str,
    expected_identity: str,
    expected_sha256: str,
) -> dict[str, Any]:
    value, raw = _load_json(path)
    file_sha = hashlib.sha256(raw).hexdigest()
    body = dict(value)
    recorded = body.pop(identity_field, None)
    if (
        file_sha != expected_sha256
        or recorded != expected_identity
        or canonical_sha256(body) != expected_identity
    ):
        raise SplitRetestError(f"frozen v1 artifact differs: {path}")
    return {
        "path": path.as_posix(),
        "file_sha256": file_sha,
        identity_field: recorded,
    }


def verify_v1_artifacts(
    root: Path,
    specification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = {
        "plan": ("plan_identity", EXPECTED_V1_PLAN_IDENTITY),
        "split": ("split_identity", EXPECTED_V1_SPLIT_IDENTITY),
        "result": ("result_identity", EXPECTED_V1_RESULT_IDENTITY),
    }
    rows = []
    for role in ("plan", "split", "result"):
        item = specification.get(role)
        if not isinstance(item, Mapping):
            raise SplitRetestError("v1 preservation input is absent")
        identity_field, identity = expected[role]
        if item.get("identity") != identity:
            raise SplitRetestError("v1 preservation identity differs")
        path = root / str(item.get("path"))
        rows.append(
            {
                "role": role,
                **_verify_sealed_identity(
                    path,
                    identity_field=identity_field,
                    expected_identity=identity,
                    expected_sha256=str(item.get("file_sha256")),
                ),
            }
        )
    return rows


def _claim_impact() -> dict[str, Any]:
    return {
        "design_a_player_cut": {
            "supports": [
                "development measurement on held-out source player keys after cross-cut games are discarded",
                "a source-domain unseen-account claim for those held-out keys",
            ],
            "does_not_support": [
                "future-time generalization unless time is separately isolated",
                "real-person identity beyond the recovered source account key",
                "product-UI transport",
            ],
            "leakage": (
                "cross-cut games are discarded, but source, calendar, opponent-network, "
                "and repeated-position dependence can remain"
            ),
            "endpoint_effect": (
                "mechanism estimates can target held-out actors; product outcomes lose "
                "all discarded cross-cut games and may change the natural-game estimand"
            ),
        },
        "design_b_time_holdout": {
            "supports": [
                "future-period source traffic under the time-only definition",
                "future-period unseen-account traffic under the stronger both-players-new definition",
            ],
            "does_not_support": [
                "unseen-player generalization for returning-player time-only games",
                "product-UI transport",
            ],
            "leakage": (
                "returning players carry identity and learning history across the cut; "
                "the strong subset removes that overlap but changes the traffic population"
            ),
            "endpoint_effect": (
                "mechanism endpoints can be estimated on either named population; factual "
                "product endpoints on the strong subset no longer represent all later games"
            ),
        },
        "design_c_decision_owner": {
            "supports": [
                "held-out decision-maker keys for actor-choice behavior estimates",
            ],
            "does_not_support": [
                "independent complete-game outcomes",
                "future-time generalization",
                "an untouched trajectory or state distribution",
                "product-UI transport",
            ],
            "leakage": (
                "one game can contribute adjacent trajectory states to multiple partitions; "
                "ring16-equivalent states can also recur across games and partitions"
            ),
            "endpoint_effect": (
                "actor-level mechanism labels remain uniquely owned, but standard errors "
                "must cluster by game/player; complete-game product endpoints cannot be "
                "assigned without a separate game-level rule"
            ),
        },
    }


def run_measurement(
    *,
    repository_root: str | Path,
    boundary: Boundary,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    f0d0_manifest_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root)
    v1_artifacts = verify_v1_artifacts(root, plan["superseded_v1"])
    data = build_player_graph(boundary.games)
    graph_metrics = graph_structure(data, boundary.games)
    community_spec = plan["graph_measurement"]["community_detection"]
    communities, community_metrics = detect_communities(
        data,
        seed=int(community_spec["seed"]),
        resolution=float(community_spec["resolution"]),
    )
    time_metrics = time_structure(
        boundary.games,
        cut_dates=plan["graph_measurement"]["time_cut_dates"],
    )
    design_a_spec = plan["design_a"]
    design_a = measure_design_a(
        data,
        boundary.games,
        communities,
        targets=design_a_spec["target_holdout_player_fractions"],
        seed=str(design_a_spec["seed"]),
        restarts=int(design_a_spec["restarts"]),
        maximum_kl_iterations=int(design_a_spec["maximum_kl_iterations"]),
    )
    players = {player for game in boundary.games for player in game.players}
    design_c_spec = plan["design_c"]
    membership, membership_summary = decision_player_partition(
        players,
        ratios=design_c_spec["player_ratios"],
        seed=str(design_c_spec["seed"]),
    )
    design_c_counts = measure_design_c_counts(boundary.games, membership)
    design_c_ring = measure_design_c_ring16(
        repository_root=root,
        games=boundary.games,
        membership=membership,
        progress=progress,
    )
    if sum(
        row["decisions"] for row in design_c_counts["partition_counts"].values()
    ) != design_c_ring["strict_replayed_decisions"]:
        raise SplitRetestError("design C decision accounting differs")

    design_b = {
        "method": "calendar cut; post-cut starts on the cut date",
        "time_only_all_post_games": True,
        "strong_player_and_time_isolation": (
            "both players in a post-cut game must be absent from every pre-cut game"
        ),
        "candidate_cuts": time_metrics["candidate_cuts"],
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "measurement_id": plan["measurement_id"],
        "status": "completed_measurement_only_no_split_selection",
        "decision": None,
        "recommendation": None,
        "scope": {
            "graph_time_and_candidate_split_scale_only": True,
            "f0_h0_scientific_dimensions_run": False,
            "final_split_selected": False,
            "malom_queries": 0,
            "games_started": 0,
            "models_loaded": 0,
            "training_updates": 0,
        },
        "lineage": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha256,
            "f0d0_manifest_identity": boundary.manifest_identity,
            "f0d0_manifest_file_sha256": boundary.manifest_file_sha256,
            "f0d0_corpus_identity": boundary.corpus_identity,
            "behavior_raw_subset_identity": boundary.raw_subset_identity,
        },
        "supersession": {
            "v1_status": "superseded_by_corrected_split_design",
            "reason": (
                "v1 combined complete per-player game isolation with bilateral games, "
                "which algebraically collapses each connected component and tests only "
                "whether a zero-cut partition exists"
            ),
            "v1_artifacts_verified_unchanged": v1_artifacts,
            "replacement_scope": "split-feasibility measurement only",
        },
        "inputs": {
            "f0d0_manifest": {
                "path": Path(f0d0_manifest_path).as_posix(),
                "file_sha256": boundary.manifest_file_sha256,
                "manifest_identity": boundary.manifest_identity,
                "corpus_identity": boundary.corpus_identity,
            },
            "raw_behavior_games": {
                "count": len(boundary.games),
                "identity_from_f0d0": boundary.raw_subset_identity,
                "all_sizes_and_sha256_verified_before_use": True,
                "reason_opened": (
                    "F0-D0 does not retain per-ply predecessor states required for exact "
                    "design C ring16 overlap"
                ),
            },
        },
        "graph_structure": graph_metrics,
        "community_structure": community_metrics,
        "time_structure": time_metrics,
        "candidate_designs": {
            "design_a_player_cut": {
                "method": (
                    "Louvain community initialization plus fixed-size weighted "
                    "Kernighan-Lin refinement; each target is measured independently"
                ),
                "measurements": design_a,
            },
            "design_b_time_holdout": design_b,
            "design_c_decision_owner": {
                "player_partition": membership_summary,
                "scale": design_c_counts,
                "ring16_leakage": design_c_ring,
            },
        },
        "claim_impact": _claim_impact(),
        "access_audit": {
            "f0d0_manifest_read": True,
            "raw_behavior_files_opened": design_c_ring["raw_files_opened"],
            "raw_behavior_bytes_read": design_c_ring["raw_bytes_read"],
            "humandb_reads": 0,
            "database_writes": 0,
            "malom_queries": 0,
            "source_pool_2eb04f54_artifact_reads": 0,
            "source_pool_records_consumed": 0,
        },
        "prohibited_operations_observed": {
            "database_writes": 0,
            "games_started": 0,
            "search_batches_started": 0,
            "models_loaded": 0,
            "training_updates": 0,
            "f0_h0_scientific_dimensions": 0,
            "final_split_selections": 0,
        },
    }


def load_result(path: str | Path) -> tuple[dict[str, Any], str]:
    return _load_sealed_json(
        path,
        schema=RESULT_SCHEMA,
        identity_field="result_identity",
    )
