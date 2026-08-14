"""Read-only F0-H0 Design B support and second-level split measurements.

This module measures support, player-cut structure, calendar subsegments, and
ring16 overlap for preregistered candidate designs.  It selects no final split
and performs no Malom query, model load, game, search, training, or database
operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx

from game.board import BoardState
from game.draw_rules import StandardDrawTracker
from game.rules import get_all_legal_moves, get_game_phase, terminal_result
from learned_ai.evaluation.human_f0h0_split_retest import (
    EXPECTED_CORPUS_IDENTITY,
    EXPECTED_F0D0_FILE_SHA256,
    EXPECTED_F0D0_MANIFEST_IDENTITY,
    F0D0_SCHEMA,
    Boundary,
    GraphData,
    GameRecord,
    SplitRetestError,
    _load_sealed_json,
    _move_notation,
    _read_raw_game,
    build_player_graph,
    canonical_sha256,
    detect_communities,
    graph_structure,
    load_boundary,
    load_result as load_split_retest_result,
    measure_design_a,
    nearest_rank_quantiles,
    sha256_file,
    write_sealed_json,
)
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen


PLAN_SCHEMA = "nmm.f0-h0-design-b-supplement-plan.v1"
RESULT_SCHEMA = "nmm.f0-h0-design-b-supplement-result.v1"
MEASUREMENT_STATUS = "completed_measurement_only_no_final_split_selection"
PHASES = ("placement", "movement", "flying")
B_PARTITIONS = (
    "train",
    "selection",
    "one-time-confirmation",
    "final-test",
)
PHASE_NAME = {"place": "placement", "move": "movement", "fly": "flying"}
OUTCOME_ELIGIBLE_GAMES = 37_866
PREVIOUS_RESULT_IDENTITY = (
    "cbfa6d43fa31e9644bae169e6b6d42232aa008e54921c96a46fbdddb73a95931"
)
PREVIOUS_RESULT_FILE_SHA256 = (
    "eb0ed05a458b282a88b6bce12824a9744780238601609f446d7772b886dba77a"
)


@dataclass(frozen=True)
class B2Candidate:
    candidate_id: str
    cut_one: date
    cut_two: date


@dataclass(frozen=True)
class MeasurementInputs:
    boundary: Boundary
    outcome_eligible: Mapping[str, bool]
    previous_result: Mapping[str, Any]
    previous_result_file_sha256: str


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitRetestError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise SplitRetestError(f"JSON input is not an object: {path}")
    return value, raw


def load_outcome_eligibility(path: str | Path) -> dict[str, bool]:
    """Decode F0-D0 outcome eligibility without inventing result support."""
    source = Path(path)
    manifest, raw = _read_json_object(source)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_F0D0_FILE_SHA256:
        raise SplitRetestError("F0-D0 manifest file SHA-256 differs")
    if manifest.get("schema_version") != F0D0_SCHEMA:
        raise SplitRetestError("F0-D0 manifest schema differs")
    body = dict(manifest)
    recorded = body.pop("manifest_identity", None)
    if (
        recorded != EXPECTED_F0D0_MANIFEST_IDENTITY
        or canonical_sha256(body) != recorded
    ):
        raise SplitRetestError("F0-D0 manifest identity differs")
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or identities.get(
        "corpus_identity"
    ) != EXPECTED_CORPUS_IDENTITY:
        raise SplitRetestError("F0-D0 corpus identity differs")

    encoding = manifest.get("record_encoding")
    rows = manifest.get("game_records")
    if not isinstance(encoding, Mapping) or not isinstance(rows, list):
        raise SplitRetestError("F0-D0 record encoding is absent")
    fields = encoding.get("fields")
    if not isinstance(fields, list):
        raise SplitRetestError("F0-D0 record fields are absent")
    positions = {field: index for index, field in enumerate(fields)}
    required = {
        "session_id",
        "behavior_replay_eligible",
        "outcome_analysis_eligible",
    }
    if not required.issubset(positions):
        raise SplitRetestError("F0-D0 outcome fields are absent")

    outcome: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise SplitRetestError("F0-D0 game row width differs")
        if not row[positions["behavior_replay_eligible"]]:
            continue
        session_id = row[positions["session_id"]]
        eligible = row[positions["outcome_analysis_eligible"]]
        if not isinstance(session_id, str) or not isinstance(eligible, bool):
            raise SplitRetestError("F0-D0 outcome row is invalid")
        if session_id in outcome:
            raise SplitRetestError("F0-D0 behavior session is duplicated")
        outcome[session_id] = eligible
    if len(outcome) != 92_226 or sum(outcome.values()) != OUTCOME_ELIGIBLE_GAMES:
        raise SplitRetestError("F0-D0 outcome base differs")
    return outcome


def load_measurement_inputs(
    *,
    f0d0_path: str | Path,
    previous_result_path: str | Path,
) -> MeasurementInputs:
    boundary = load_boundary(f0d0_path)
    outcome = load_outcome_eligibility(f0d0_path)
    if set(outcome) != {game.session_id for game in boundary.games}:
        raise SplitRetestError("F0-D0 outcome and behavior memberships differ")
    previous_result, previous_sha = load_split_retest_result(previous_result_path)
    if previous_result.get("result_identity") != PREVIOUS_RESULT_IDENTITY:
        raise SplitRetestError("previous split result identity differs")
    if previous_sha != PREVIOUS_RESULT_FILE_SHA256:
        raise SplitRetestError("previous split result file SHA-256 differs")
    return MeasurementInputs(
        boundary=boundary,
        outcome_eligible=outcome,
        previous_result=previous_result,
        previous_result_file_sha256=previous_sha,
    )


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = _load_sealed_json(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    boundary = plan.get("input_boundary")
    if boundary != {
        "corpus_identity": EXPECTED_CORPUS_IDENTITY,
        "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
        "f0d0_manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
        "behavior_games": 92_226,
        "behavior_logical_plies": 4_394_220,
        "behavior_player_keys": 4_994,
        "outcome_games": OUTCOME_ELIGIBLE_GAMES,
        "previous_split_result_identity": PREVIOUS_RESULT_IDENTITY,
        "previous_split_result_file_sha256": PREVIOUS_RESULT_FILE_SHA256,
    }:
        raise SplitRetestError("supplement plan input boundary differs")
    scope = plan.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) is not False
        for field in (
            "select_final_split",
            "make_feasibility_decision",
            "run_f0_h0_scientific_dimensions",
        )
    ):
        raise SplitRetestError("supplement plan scope differs")
    if plan.get("status") != "frozen_before_supplement_statistics":
        raise SplitRetestError("supplement plan status differs")
    support = plan.get("support_measurement")
    if not isinstance(support, Mapping) or support.get("cut_dates") != [
        "2026-03-01",
        "2026-05-01",
    ]:
        raise SplitRetestError("supplement support cuts differ")
    b1 = plan.get("b1_player_resplit")
    if not isinstance(b1, Mapping) or b1.get(
        "source_membership_identity"
    ) != "cf65a0030d9051ecdf6fe3a07e0693c05ba9cfc833d1b32c398f5cd8268ffe6a":
        raise SplitRetestError("supplement B1 source identity differs")
    if b1.get("independent_cut_player_fractions") != [
        0.25,
        0.3333333333333333,
        0.5,
    ]:
        raise SplitRetestError("supplement B1 cut fractions differ")
    b2 = plan.get("b2_time_resplit")
    if not isinstance(b2, Mapping) or b2.get("base_cut") != "2026-03-01":
        raise SplitRetestError("supplement B2 base cut differs")
    if [
        (row.candidate_id, row.cut_one.isoformat(), row.cut_two.isoformat())
        for row in _parse_b2_candidates(plan)
    ] != [
        ("equal-calendar-span", "2026-04-17", "2026-06-03"),
        ("early-month-boundaries", "2026-04-01", "2026-05-01"),
        ("later-month-boundaries", "2026-05-01", "2026-06-01"),
    ]:
        raise SplitRetestError("supplement B2 cut candidates differ")
    ring16 = plan.get("ring16_comparison")
    if not isinstance(ring16, Mapping):
        raise SplitRetestError("supplement ring16 plan differs")
    random_spec = ring16.get("random_baseline")
    if not isinstance(random_spec, Mapping) or (
        random_spec.get("left_games"), random_spec.get("right_games")
    ) != (36_949, 4_577):
        raise SplitRetestError("supplement random baseline scale differs")
    prohibited = plan.get("prohibited_operations")
    expected_prohibited = {
        "database_writes",
        "games",
        "search_batches",
        "model_loads",
        "training",
        "malom_queries",
        "source_pool_2eb04f54_reads_or_consumption",
        "final_split_selection",
        "feasibility_decision",
        "existing_frozen_record_changes",
    }
    if (
        not isinstance(prohibited, Mapping)
        or set(prohibited) != expected_prohibited
        or any(value is not True for value in prohibited.values())
    ):
        raise SplitRetestError("supplement prohibited operations differ")
    implementation = plan.get("implementation_artifacts")
    expected_paths = {
        "learned_ai/evaluation/human_f0h0_design_b_supplement.py",
        "scripts/measure_human_f0h0_design_b_supplement.py",
        "tests/test_human_f0h0_design_b_supplement.py",
    }
    if not isinstance(implementation, list) or {
        row.get("path") for row in implementation if isinstance(row, Mapping)
    } != expected_paths:
        raise SplitRetestError("supplement implementation artifacts differ")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        for row in implementation
    ):
        raise SplitRetestError("supplement implementation hashes differ")
    return plan, file_sha


def load_result(path: str | Path) -> tuple[dict[str, Any], str]:
    return _load_sealed_json(
        path,
        schema=RESULT_SCHEMA,
        identity_field="result_identity",
    )


def _players(games: Sequence[GameRecord]) -> set[str]:
    return {player for game in games for player in game.players}


def _games_before(games: Sequence[GameRecord], cut: date) -> list[GameRecord]:
    return [game for game in games if game.played_on < cut]


def _strong_games(
    games: Sequence[GameRecord],
    *,
    start: date,
    end: date | None = None,
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


def _game_count_distribution(games: Sequence[GameRecord]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for game in games:
        counts[game.white_player] += 1
        counts[game.black_player] += 1
    values = list(counts.values())
    if not values:
        return {
            "quantiles": None,
            "tail": [],
            "total_player_game_incidences": 0,
        }
    return {
        "quantiles": nearest_rank_quantiles(values),
        "tail": [
            {
                "at_least_games": threshold,
                "player_keys": sum(value >= threshold for value in values),
            }
            for threshold in (1, 2, 5, 10, 20, 50, 100, 250, 500)
        ],
        "total_player_game_incidences": sum(values),
    }


def _support_row(
    games: Sequence[GameRecord],
    outcome_eligible: Mapping[str, bool],
) -> dict[str, Any]:
    outcome_games = [game for game in games if outcome_eligible[game.session_id]]
    return {
        "games": len(games),
        "player_keys": len(_players(games)),
        "decisions": sum(game.move_count for game in games),
        "outcome_eligible_games": len(outcome_games),
        "outcome_eligible_player_keys": len(_players(outcome_games)),
        "player_game_distribution": _game_count_distribution(games),
        "membership_identity": canonical_sha256(
            sorted(game.session_id for game in games)
        ),
    }


def measure_support_metadata(
    games: Sequence[GameRecord],
    outcome_eligible: Mapping[str, bool],
    *,
    cut_dates: Sequence[str],
) -> tuple[dict[str, Any], dict[str, list[GameRecord]]]:
    result: dict[str, Any] = {}
    subsets: dict[str, list[GameRecord]] = {}
    for raw_cut in cut_dates:
        cut = date.fromisoformat(raw_cut)
        train = _games_before(games, cut)
        strong = _strong_games(games, start=cut)
        subsets[f"support::{raw_cut}::train"] = train
        subsets[f"support::{raw_cut}::strong"] = strong
        result[raw_cut] = {
            "train": _support_row(train, outcome_eligible),
            "strong_post": _support_row(strong, outcome_eligible),
        }
    return result, subsets


def _component_rows(
    games: Sequence[GameRecord],
) -> tuple[dict[str, Any], Any]:
    graph_data = build_player_graph(games)
    structure = graph_structure(graph_data, games)
    rows = [structure["giant_component"], *structure["non_giant_components"]]
    return {
        **structure,
        "all_component_rows": rows,
        "zero_cut_status": (
            "single_component_zero_cut_collapse"
            if structure["connected_components"] == 1
            else "multiple_indivisible_components_measured"
        ),
    }, graph_data


def _largest_remainder_counts(
    total: int,
    ratios: Mapping[str, float],
    order: Sequence[str],
) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in order}
    counts = {name: math.floor(raw[name]) for name in order}
    missing = total - sum(counts.values())
    ranked = sorted(
        order,
        key=lambda name: (-(raw[name] - counts[name]), order.index(name)),
    )
    for name in ranked[:missing]:
        counts[name] += 1
    return counts


def _hash_rank(namespace: str, seed: str, value: str) -> str:
    return hashlib.sha256(
        f"{namespace}\0{seed}\0{value}".encode("utf-8")
    ).hexdigest()


def _initial_subset(
    nodes: set[str],
    communities: Sequence[set[str]],
    *,
    target: int,
    seed: str,
) -> set[str]:
    ranked = sorted(
        communities,
        key=lambda community: _hash_rank(
            "f0-h0-b1-community-v1",
            seed,
            canonical_sha256(sorted(community)),
        ),
    )
    selected: set[str] = set()
    for community in ranked:
        if len(selected) + len(community) <= target:
            selected.update(community)
    if len(selected) < target:
        remaining = sorted(
            nodes - selected,
            key=lambda player: (
                _hash_rank("f0-h0-b1-fill-v1", seed, player),
                player,
            ),
        )
        selected.update(remaining[: target - len(selected)])
    if len(selected) != target:
        raise SplitRetestError("B1 initial subset has wrong size")
    return selected


def _kl_exact_subset(
    graph: nx.Graph,
    communities: Sequence[set[str]],
    *,
    target: int,
    seed: str,
    max_iterations: int,
    nx_seed: int,
) -> set[str]:
    nodes = set(graph)
    initial = _initial_subset(nodes, communities, target=target, seed=seed)
    if graph.number_of_edges() == 0:
        return initial
    left, right = nx.community.kernighan_lin_bisection(
        graph,
        partition=(initial, nodes - initial),
        max_iter=max_iterations,
        weight="games",
        seed=nx_seed,
    )
    if len(left) == target:
        return set(left)
    if len(right) == target:
        return set(right)
    raise SplitRetestError("B1 Kernighan-Lin changed partition size")


def _classify_three_way(
    games: Sequence[GameRecord],
    membership: Mapping[str, str],
) -> dict[str, Any]:
    internal = Counter()
    cross_pairs = Counter()
    for game in games:
        white = membership[game.white_player]
        black = membership[game.black_player]
        if white == black:
            internal[white] += 1
        else:
            cross_pairs[tuple(sorted((white, black)))] += 1
    discarded = sum(cross_pairs.values())
    return {
        "internal_games": {
            partition: internal[partition]
            for partition in (
                "selection",
                "one-time-confirmation",
                "final-test",
            )
        },
        "cross_partition_discard_games": discarded,
        "cross_partition_discard_fraction": discarded / len(games),
        "cross_partition_pair_games": [
            {"left": pair[0], "right": pair[1], "games": count}
            for pair, count in sorted(cross_pairs.items())
        ],
    }


def measure_b1_three_way(
    graph_data: GraphData,
    games: Sequence[GameRecord],
    *,
    ratios: Mapping[str, float],
    seed: str,
    outer_restarts: int,
    inner_restarts: int,
    max_iterations: int,
    louvain_seed: int,
) -> dict[str, Any]:
    order = ("selection", "one-time-confirmation", "final-test")
    counts = _largest_remainder_counts(len(graph_data.graph), ratios, order)
    full_communities, _ = detect_communities(
        graph_data,
        seed=louvain_seed,
        resolution=1.0,
    )
    candidates = []
    for outer in range(outer_restarts):
        selection = _kl_exact_subset(
            graph_data.graph,
            full_communities,
            target=counts["selection"],
            seed=f"{seed}:outer:{outer}",
            max_iterations=max_iterations,
            nx_seed=louvain_seed + outer,
        )
        remainder = set(graph_data.graph) - selection
        remainder_graph = graph_data.graph.subgraph(remainder).copy()
        remainder_data = type(graph_data)(
            graph=remainder_graph,
            edge_games={
                edge: count
                for edge, count in graph_data.edge_games.items()
                if edge[0] in remainder and edge[1] in remainder
            },
            player_games=Counter(
                {
                    player: graph_data.player_games[player]
                    for player in remainder
                }
            ),
            opponents={
                player: set(remainder_graph.neighbors(player))
                for player in remainder
            },
            self_games=0,
        )
        if remainder_graph.number_of_edges() == 0:
            remainder_communities = [{player} for player in sorted(remainder)]
        else:
            remainder_communities, _ = detect_communities(
                remainder_data,
                seed=louvain_seed + 10_000 + outer,
                resolution=1.0,
            )
        for inner in range(inner_restarts):
            confirmation = _kl_exact_subset(
                remainder_graph,
                remainder_communities,
                target=counts["one-time-confirmation"],
                seed=f"{seed}:outer:{outer}:inner:{inner}",
                max_iterations=max_iterations,
                nx_seed=louvain_seed + 20_000 + outer * 100 + inner,
            )
            final_test = remainder - confirmation
            membership = {
                **{player: "selection" for player in selection},
                **{
                    player: "one-time-confirmation"
                    for player in confirmation
                },
                **{player: "final-test" for player in final_test},
            }
            classified = _classify_three_way(games, membership)
            membership_identity = canonical_sha256(
                [[player, membership[player]] for player in sorted(membership)]
            )
            candidates.append(
                {
                    "outer_restart": outer,
                    "inner_restart": inner,
                    "membership": membership,
                    "membership_identity": membership_identity,
                    **classified,
                }
            )
    chosen = min(
        candidates,
        key=lambda row: (
            row["cross_partition_discard_games"],
            row["membership_identity"],
        ),
    )
    return {
        "player_counts": counts,
        "measured_player_counts": Counter(chosen["membership"].values()),
        "membership_identity": chosen["membership_identity"],
        "selected_outer_restart_for_measurement": chosen["outer_restart"],
        "selected_inner_restart_for_measurement": chosen["inner_restart"],
        "restart_discard_game_counts": [
            {
                "outer_restart": row["outer_restart"],
                "inner_restart": row["inner_restart"],
                "discard_games": row["cross_partition_discard_games"],
            }
            for row in candidates
        ],
        "internal_games": chosen["internal_games"],
        "cross_partition_discard_games": chosen[
            "cross_partition_discard_games"
        ],
        "cross_partition_discard_fraction": chosen[
            "cross_partition_discard_fraction"
        ],
        "cross_partition_pair_games": chosen["cross_partition_pair_games"],
    }


def measure_b1(
    games: Sequence[GameRecord],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    structure, graph_data = _component_rows(games)
    spec = plan["b1_player_resplit"]
    communities, summary = detect_communities(
        graph_data,
        seed=int(spec["louvain_seed"]),
        resolution=1.0,
    )
    independent = measure_design_a(
        graph_data,
        games,
        communities,
        targets=spec["independent_cut_player_fractions"],
        seed=spec["seed"],
        restarts=int(spec["independent_cut_restarts"]),
        maximum_kl_iterations=int(spec["maximum_kl_iterations"]),
    )
    three_way = measure_b1_three_way(
        graph_data,
        games,
        ratios=spec["three_way_player_ratios"],
        seed=spec["seed"],
        outer_restarts=int(spec["three_way_outer_restarts"]),
        inner_restarts=int(spec["three_way_inner_restarts"]),
        max_iterations=int(spec["maximum_kl_iterations"]),
        louvain_seed=int(spec["louvain_seed"]),
    )
    return {
        "graph": structure,
        "community_structure": summary,
        "independent_edge_cut_measurements": independent,
        "simultaneous_three_way_edge_cut_measurement": three_way,
    }


def _parse_b2_candidates(plan: Mapping[str, Any]) -> list[B2Candidate]:
    candidates = []
    for row in plan["b2_time_resplit"]["candidate_cut_pairs"]:
        candidate = B2Candidate(
            candidate_id=row["candidate_id"],
            cut_one=date.fromisoformat(row["cut_one"]),
            cut_two=date.fromisoformat(row["cut_two"]),
        )
        if not date(2026, 3, 1) < candidate.cut_one < candidate.cut_two:
            raise SplitRetestError("B2 cut order is invalid")
        candidates.append(candidate)
    if len(candidates) < 3 or len({row.candidate_id for row in candidates}) != len(
        candidates
    ):
        raise SplitRetestError("B2 candidate set is invalid")
    return candidates


def build_b2_profiles(
    games: Sequence[GameRecord],
    outcome_eligible: Mapping[str, bool],
    *,
    candidates: Sequence[B2Candidate],
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, list[GameRecord]]]:
    base = date(2026, 3, 1)
    train = _games_before(games, base)
    march_test_pool = _strong_games(games, start=base)
    measurements: dict[str, Any] = {}
    profiles: dict[str, dict[str, str]] = {}
    subsets: dict[str, list[GameRecord]] = {}
    for candidate in candidates:
        bounds = (
            ("selection", base, candidate.cut_one),
            ("one-time-confirmation", candidate.cut_one, candidate.cut_two),
            ("final-test", candidate.cut_two, None),
        )
        membership = {game.session_id: "train" for game in train}
        segments: dict[str, Any] = {}
        subsets[f"b2::{candidate.candidate_id}::train"] = train
        for partition, start, end in bounds:
            all_segment = [
                game
                for game in march_test_pool
                if game.played_on >= start
                and (end is None or game.played_on < end)
            ]
            strong = _strong_games(games, start=start, end=end)
            subsets[f"b2::{candidate.candidate_id}::{partition}"] = strong
            for game in strong:
                if game.session_id in membership:
                    raise SplitRetestError("B2 game belongs to two partitions")
                membership[game.session_id] = partition
            support = _support_row(strong, outcome_eligible)
            segments[partition] = {
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat() if end is not None else None,
                "all_segment_games": len(all_segment),
                "all_segment_player_keys": len(_players(all_segment)),
                "all_segment_membership_identity": canonical_sha256(
                    sorted(game.session_id for game in all_segment)
                ),
                "both_players_unseen_before_segment": support,
            }
        profiles[candidate.candidate_id] = membership
        measurements[candidate.candidate_id] = {
            "cut_one": candidate.cut_one.isoformat(),
            "cut_two": candidate.cut_two.isoformat(),
            "march_test_pool_games": len(march_test_pool),
            "march_test_pool_player_keys": len(_players(march_test_pool)),
            "march_test_pool_membership_identity": canonical_sha256(
                sorted(game.session_id for game in march_test_pool)
            ),
            "train": _support_row(train, outcome_eligible),
            "segments": segments,
            "included_game_membership_identity": canonical_sha256(
                [[session, membership[session]] for session in sorted(membership)]
            ),
        }
    return measurements, profiles, subsets


def build_random_baseline(
    games: Sequence[GameRecord],
    *,
    left_games: int,
    right_games: int,
    seed: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    if left_games <= 0 or right_games <= 0 or left_games + right_games > len(games):
        raise SplitRetestError("random baseline sizes are invalid")
    ranked = sorted(
        games,
        key=lambda game: (
            _hash_rank("f0-h0-b-ring16-random-v1", seed, game.session_id),
            game.session_id,
        ),
    )
    left = ranked[:left_games]
    right = ranked[left_games : left_games + right_games]
    membership = {
        **{game.session_id: "random-left" for game in left},
        **{game.session_id: "random-right" for game in right},
    }
    return membership, {
        "seed": seed,
        "left_games": left_games,
        "right_games": right_games,
        "sets_disjoint": True,
        "membership_identity": canonical_sha256(
            [[session, membership[session]] for session in sorted(membership)]
        ),
    }


def _increment_packed(value: int, group_index: int, group_count: int) -> int:
    width = 32
    value |= 1 << group_index
    value += 1 << (group_count + width * group_index)
    return value


def _packed_count(value: int, group_index: int, group_count: int) -> int:
    width = 32
    return (value >> (group_count + width * group_index)) & ((1 << width) - 1)


def summarize_overlap(
    orbit_values: Mapping[str, int],
    groups: Sequence[str],
) -> dict[str, Any]:
    group_count = len(groups)
    unique = [0] * group_count
    shared_unique = [0] * group_count
    decisions = [0] * group_count
    shared_decisions = [0] * group_count
    intersections: Counter[tuple[int, int]] = Counter()
    for value in orbit_values.values():
        mask = value & ((1 << group_count) - 1)
        is_shared = mask.bit_count() > 1
        for index in range(group_count):
            count = _packed_count(value, index, group_count)
            if not count:
                continue
            unique[index] += 1
            decisions[index] += count
            if is_shared:
                shared_unique[index] += 1
                shared_decisions[index] += count
        for left in range(group_count):
            for right in range(left + 1, group_count):
                if mask & (1 << left) and mask & (1 << right):
                    intersections[(left, right)] += 1
    partition_rows = []
    for index, group in enumerate(groups):
        partition_rows.append(
            {
                "partition": group,
                "decisions": decisions[index],
                "unique_ring16_orbits": unique[index],
                "shared_ring16_orbits": shared_unique[index],
                "unique_orbit_overlap_rate": (
                    shared_unique[index] / unique[index] if unique[index] else 0.0
                ),
                "decisions_on_shared_ring16_orbits": shared_decisions[index],
                "decision_weighted_overlap_rate": (
                    shared_decisions[index] / decisions[index]
                    if decisions[index]
                    else 0.0
                ),
            }
        )
    pair_rows = []
    for left in range(group_count):
        for right in range(left + 1, group_count):
            intersection = intersections[(left, right)]
            union = unique[left] + unique[right] - intersection
            pair_rows.append(
                {
                    "left": groups[left],
                    "right": groups[right],
                    "shared_unique_ring16_orbits": intersection,
                    "jaccard": intersection / union if union else 0.0,
                    "share_of_left_unique_orbits": (
                        intersection / unique[left] if unique[left] else 0.0
                    ),
                    "share_of_right_unique_orbits": (
                        intersection / unique[right] if unique[right] else 0.0
                    ),
                }
            )
    return {
        "state_identity": (
            "repository ring16: D4 x abstract inner/outer-ring swap over the "
            "pre-decision NMM FEN"
        ),
        "partition_overlap": partition_rows,
        "pairwise_overlap": pair_rows,
        "unique_ring16_orbits_all_partitions": len(orbit_values),
    }


def _phase_labels(
    subsets: Mapping[str, Sequence[GameRecord]],
) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    for label, games in subsets.items():
        for game in games:
            labels[game.session_id].append(label)
    return labels


def replay_required_games(
    *,
    repository_root: str | Path,
    games: Sequence[GameRecord],
    outcome_eligible: Mapping[str, bool],
    phase_subsets: Mapping[str, Sequence[GameRecord]],
    profile_memberships: Mapping[str, Mapping[str, str]],
    profile_groups: Mapping[str, Sequence[str]],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root)
    phase_labels = _phase_labels(phase_subsets)
    assignments: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for profile, membership in profile_memberships.items():
        groups = list(profile_groups[profile])
        for session, group in membership.items():
            try:
                group_index = groups.index(group)
            except ValueError as exc:
                raise SplitRetestError("ring16 profile group is invalid") from exc
            assignments[session].append((profile, group_index))
    required_ids = set(phase_labels) | set(assignments)
    required_games = [game for game in games if game.session_id in required_ids]
    if len(required_games) != len(required_ids):
        raise SplitRetestError("raw replay requirement membership differs")

    orbit_tables: dict[str, dict[str, int]] = {
        profile: {} for profile in profile_memberships
    }
    phase_counts: dict[str, Counter[str]] = {
        label: Counter() for label in phase_subsets
    }
    raw_bytes = 0
    decisions = 0
    terminal_games = 0
    for game_number, game in enumerate(required_games, start=1):
        raw_game = _read_raw_game(root, game)
        raw_bytes += game.file_size
        moves = raw_game.get("moves")
        if not isinstance(moves, list) or len(moves) != game.move_count:
            raise SplitRetestError(f"raw move count differs: {game.session_id}")
        board = BoardState.new_game()
        tracker = StandardDrawTracker(board)
        terminal_seen = False
        game_assignments = assignments.get(game.session_id, [])
        game_phase_labels = phase_labels.get(game.session_id, [])
        for logical_ply, raw_move in enumerate(moves):
            if terminal_seen or not isinstance(raw_move, Mapping):
                raise SplitRetestError(
                    f"strict replay framing differs: {game.session_id}"
                )
            phase_raw = get_game_phase(board, board.turn)
            if (
                raw_move.get("board_fen_before") != board.to_fen_string()
                or raw_move.get("color") != board.turn
                or raw_move.get("turn") != logical_ply // 2 + 1
                or raw_move.get("type") != phase_raw
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
            phase = PHASE_NAME.get(phase_raw)
            if phase is None:
                raise SplitRetestError("strict replay phase is unsupported")
            for label in game_phase_labels:
                phase_counts[label][phase] += 1
            if game_assignments:
                orbit = ring16_canonical_fen(board.to_fen_string())
                for profile, group_index in game_assignments:
                    table = orbit_tables[profile]
                    table[orbit] = _increment_packed(
                        table.get(orbit, 0),
                        group_index,
                        len(profile_groups[profile]),
                    )
            decisions += 1
            after = board.apply_move(matches[0])
            draw_reason = tracker.observe(board, matches[0], after)
            is_terminal, _winner, _reason = terminal_result(after)
            terminal_seen = bool(is_terminal or draw_reason is not None)
            board = after
        if terminal_seen != outcome_eligible[game.session_id]:
            raise SplitRetestError(
                f"strict terminal eligibility differs: {game.session_id}"
            )
        terminal_games += int(terminal_seen)
        if progress is not None:
            progress(game_number, len(required_games))

    return {
        "raw_files_opened": len(required_games),
        "raw_bytes_read": raw_bytes,
        "strict_replayed_decisions": decisions,
        "strict_terminal_games": terminal_games,
        "phase_counts": {
            label: {phase: phase_counts[label][phase] for phase in PHASES}
            for label in phase_subsets
        },
        "ring16_profiles": {
            profile: summarize_overlap(orbit_tables[profile], profile_groups[profile])
            for profile in profile_memberships
        },
    }


def _attach_phase_counts(
    support: dict[str, Any],
    b2: dict[str, Any],
    phase_counts: Mapping[str, Mapping[str, int]],
) -> None:
    for cut in support:
        support[cut]["train"]["decisions_by_phase"] = phase_counts[
            f"support::{cut}::train"
        ]
        support[cut]["strong_post"]["decisions_by_phase"] = phase_counts[
            f"support::{cut}::strong"
        ]
    for candidate_id, candidate in b2.items():
        candidate["train"]["decisions_by_phase"] = phase_counts[
            f"b2::{candidate_id}::train"
        ]
        for partition, segment in candidate["segments"].items():
            segment["both_players_unseen_before_segment"][
                "decisions_by_phase"
            ] = phase_counts[f"b2::{candidate_id}::{partition}"]


def run_measurement(
    *,
    repository_root: str | Path,
    inputs: MeasurementInputs,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    f0d0_manifest_path: str | Path,
    previous_result_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root)
    for artifact in plan["implementation_artifacts"]:
        artifact_path = root / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise SplitRetestError(
                f"supplement implementation artifact differs: {artifact_path}"
            )
    games = list(inputs.boundary.games)
    support, support_subsets = measure_support_metadata(
        games,
        inputs.outcome_eligible,
        cut_dates=plan["support_measurement"]["cut_dates"],
    )
    expected_support = plan["support_measurement"]["preexisting_scale_checks"]
    for cut, expected in expected_support.items():
        measured = support[cut]
        if {
            "train_games": measured["train"]["games"],
            "strong_post_games": measured["strong_post"]["games"],
            "strong_post_player_keys": measured["strong_post"]["player_keys"],
            "strong_post_membership_identity": measured["strong_post"][
                "membership_identity"
            ],
        } != expected:
            raise SplitRetestError(f"supplement pre-existing scale differs: {cut}")
    march_strong = support_subsets["support::2026-03-01::strong"]
    b1 = measure_b1(march_strong, plan=plan)

    b2_candidates = _parse_b2_candidates(plan)
    b2, b2_profiles, b2_subsets = build_b2_profiles(
        games,
        inputs.outcome_eligible,
        candidates=b2_candidates,
    )
    if any(
        sum(
            segment["all_segment_games"]
            for segment in candidate["segments"].values()
        )
        != len(march_strong)
        for candidate in b2.values()
    ):
        raise SplitRetestError("supplement B2 March pool accounting differs")
    baseline_spec = plan["ring16_comparison"]["random_baseline"]
    random_membership, random_summary = build_random_baseline(
        games,
        left_games=int(baseline_spec["left_games"]),
        right_games=int(baseline_spec["right_games"]),
        seed=baseline_spec["seed"],
    )

    coarse_membership = {
        **{
            game.session_id: "train"
            for game in support_subsets["support::2026-03-01::train"]
        },
        **{game.session_id: "test" for game in march_strong},
    }
    profile_memberships: dict[str, Mapping[str, str]] = {
        "design_b_coarse_march": coarse_membership,
        **{f"design_b2::{name}": membership for name, membership in b2_profiles.items()},
        "random_baseline": random_membership,
    }
    profile_groups: dict[str, Sequence[str]] = {
        "design_b_coarse_march": ("train", "test"),
        **{f"design_b2::{name}": B_PARTITIONS for name in b2_profiles},
        "random_baseline": ("random-left", "random-right"),
    }
    phase_subsets = {**support_subsets, **b2_subsets}
    replay = replay_required_games(
        repository_root=repository_root,
        games=games,
        outcome_eligible=inputs.outcome_eligible,
        phase_subsets=phase_subsets,
        profile_memberships=profile_memberships,
        profile_groups=profile_groups,
        progress=progress,
    )
    _attach_phase_counts(support, b2, replay["phase_counts"])
    for cut, measured in support.items():
        for subset in ("train", "strong_post"):
            if sum(measured[subset]["decisions_by_phase"].values()) != measured[
                subset
            ]["decisions"]:
                raise SplitRetestError(
                    f"supplement phase accounting differs: {cut} {subset}"
                )

    c_ring = inputs.previous_result["candidate_designs"][
        "design_c_decision_owner"
    ]["ring16_leakage"]
    return {
        "schema_version": RESULT_SCHEMA,
        "measurement_id": plan["measurement_id"],
        "status": MEASUREMENT_STATUS,
        "decision": None,
        "recommendation": None,
        "scope": {
            "design_b_support_and_second_level_scale_only": True,
            "design_c_comparator_only": True,
            "final_split_selected": False,
            "feasibility_decision_made": False,
            "f0_h0_scientific_dimensions_run": False,
            "malom_queries": 0,
            "models_loaded": 0,
            "games_started": 0,
            "training_updates": 0,
        },
        "lineage": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha256,
            "f0d0_corpus_identity": inputs.boundary.corpus_identity,
            "f0d0_manifest_identity": inputs.boundary.manifest_identity,
            "f0d0_manifest_file_sha256": inputs.boundary.manifest_file_sha256,
            "behavior_raw_subset_identity": inputs.boundary.raw_subset_identity,
            "previous_split_result_identity": PREVIOUS_RESULT_IDENTITY,
            "previous_split_result_file_sha256": (
                inputs.previous_result_file_sha256
            ),
        },
        "inputs": {
            "f0d0_manifest": {
                "path": Path(f0d0_manifest_path).as_posix(),
                "file_sha256": inputs.boundary.manifest_file_sha256,
                "manifest_identity": inputs.boundary.manifest_identity,
                "corpus_identity": inputs.boundary.corpus_identity,
            },
            "previous_split_result": {
                "path": Path(previous_result_path).as_posix(),
                "file_sha256": inputs.previous_result_file_sha256,
                "result_identity": PREVIOUS_RESULT_IDENTITY,
            },
            "raw_behavior_games": {
                "reason_opened": (
                    "F0-D0 does not retain per-ply phase or predecessor states "
                    "required for exact phase and ring16 measurements"
                ),
                "all_opened_sizes_sha256_and_strict_replay_verified": True,
            },
        },
        "support_measurement": support,
        "b1_player_resplit": b1,
        "b2_time_resplit": b2,
        "ring16_comparison": {
            "metric_definition": (
                "for each partition, the fraction of its decisions whose "
                "ring16 orbit occurs in at least one other partition"
            ),
            "design_b_profiles": {
                name: value
                for name, value in replay["ring16_profiles"].items()
                if name.startswith("design_b")
            },
            "design_c_previous_result": {
                "result_identity": PREVIOUS_RESULT_IDENTITY,
                "partition_overlap": c_ring["partition_overlap"],
                "pairwise_overlap": c_ring["pairwise_overlap"],
            },
            "random_baseline": {
                **random_summary,
                "overlap": replay["ring16_profiles"]["random_baseline"],
            },
            "interpretation": None,
        },
        "raw_replay": {
            field: replay[field]
            for field in (
                "raw_files_opened",
                "raw_bytes_read",
                "strict_replayed_decisions",
                "strict_terminal_games",
            )
        },
        "access_audit": {
            "f0d0_manifest_read": True,
            "previous_split_result_read": True,
            "raw_behavior_files_opened": replay["raw_files_opened"],
            "raw_behavior_bytes_read": replay["raw_bytes_read"],
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
            "malom_queries": 0,
            "f0_h0_scientific_dimensions": 0,
            "final_split_selections": 0,
            "feasibility_decisions": 0,
        },
    }


__all__ = [
    "B2Candidate",
    "MEASUREMENT_STATUS",
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "build_b2_profiles",
    "build_random_baseline",
    "load_measurement_inputs",
    "load_plan",
    "load_result",
    "measure_b1_three_way",
    "measure_support_metadata",
    "replay_required_games",
    "run_measurement",
    "summarize_overlap",
    "write_sealed_json",
]
