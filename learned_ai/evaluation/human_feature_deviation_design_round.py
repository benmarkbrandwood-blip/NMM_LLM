"""Structural rebalancing and extended exploration for human deviation.

The structural half of this module is deliberately incapable of opening raw
game content.  It consumes only frozen session membership, player keys, and
move counts.  The exploratory half may replay only a separately frozen sample
from the research-exploration arm.  Official B2 holdouts and the research
confirmation arm remain protected.

All oracle labels are positional-only ``A_pos`` labels.  Nothing here creates
or claims the full-history ``A_allow`` set.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx

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
    canonical_sha256,
    concentration,
)
from learned_ai.evaluation.human_feature_deviation import (
    EXPECTED_B2_COUNTS,
    EXPECTED_B2_MEMBERSHIP_IDENTITY,
    EXPECTED_F0D0_CORPUS_IDENTITY,
    EXPECTED_F0D0_FILE_SHA256,
    EXPECTED_F0D0_MANIFEST_IDENTITY,
    PHASE_NAMES,
    TRANSITIONS,
    WDL_RANK,
    ExplorationOnlyAccess,
    FeatureDeviationError,
    FeatureOpportunity,
    _feature_summary,
    _potential_mills,
    _rate,
    _serialize_counter,
    action_feature_scores,
)


DESIGN_PLAN_SCHEMA = "nmm.human-feature-deviation-design-round.v1"
REBALANCE_SCHEMA = "nmm.human-feature-deviation-rebalance.v1"
SPLIT_V2_SCHEMA = "nmm.human-feature-deviation-train-split.v2"
SPLIT_V3_SCHEMA = "nmm.human-feature-deviation-train-split.v3"
EXTENSION_SCHEMA = "nmm.human-feature-deviation-exploration-extension.v1"

V1_PLAN_IDENTITY = "04177a73ca5b9a1aa8cc8352477f2050759e6a742cee049f1191d3064ae5d662"
V1_SPLIT_IDENTITY = "fa74650c1afdffeb0d30f334b2b7859538f81b0e502c17a64092bfdcd99a06dd"
V1_EXPLORATION_IDENTITY = (
    "c489dca91c00569491d2b50a879bd014081e0109e0899afcc2bf2f13d584d7d6"
)

V2_FEATURE_NAMES = (
    "source_degree",
    "destination_degree",
    "capture_degree",
    "closes_mill",
    "opponent_immediate_mill_destinations_removed",
    "creates_mill_fork",
    "new_own_potential_mills",
    "own_mobility_delta",
    "opponent_mobility_reduction",
    "captured_opponent_threat_lines",
)


def _hash_rank(seed: str, value: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest(), value


def _load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureDeviationError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise FeatureDeviationError(f"schema differs: {source}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise FeatureDeviationError(f"identity absent: {source}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise FeatureDeviationError(f"identity differs: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def load_design_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = _load_sealed(
        path,
        schema=DESIGN_PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    if plan.get("v1_identities") != {
        "plan": V1_PLAN_IDENTITY,
        "split": V1_SPLIT_IDENTITY,
        "exploration": V1_EXPLORATION_IDENTITY,
    }:
        raise FeatureDeviationError("design-round v1 lineage differs")
    rebalance = plan.get("rebalance")
    if (
        not isinstance(rebalance, Mapping)
        or rebalance.get("candidate_confirmation_player_fractions") != [0.3, 0.4, 0.5]
        or rebalance.get("outcome_variables_allowed") is not False
        or rebalance.get("lock_v1_pilot_players_to_exploration") is not True
    ):
        raise FeatureDeviationError("design-round rebalance contract differs")
    extension = plan.get("exploration_extension")
    if (
        not isinstance(extension, Mapping)
        or extension.get("total_games") != 1024
        or extension.get("includes_v1_pilot") is not True
        or tuple(extension.get("candidate_feature_dictionary", ())) != V2_FEATURE_NAMES
    ):
        raise FeatureDeviationError("design-round extension contract differs")
    access = plan.get("prohibited_content")
    if not isinstance(access, Mapping) or any(
        access.get(name) is not True
        for name in (
            "research_confirmation",
            "official_selection",
            "official_confirmation",
            "official_final_test",
            "source_pool_2eb04f54",
        )
    ):
        raise FeatureDeviationError("design-round protected boundary differs")
    return plan, file_sha


def load_split_v2(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        raw_value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureDeviationError(f"invalid JSON: {source}") from exc
    schema = raw_value.get("schema_version") if isinstance(raw_value, dict) else None
    if schema not in {SPLIT_V2_SCHEMA, SPLIT_V3_SCHEMA}:
        raise FeatureDeviationError(f"schema differs: {source}")
    split, file_sha = _load_sealed(
        path, schema=str(schema), identity_field="split_identity"
    )
    if split.get("v1_split_remains_immutable") is not True:
        raise FeatureDeviationError("v2 split lineage boundary differs")
    partitions = split.get("partitions")
    expected_names = {
        "research-exploration",
        "research-confirmation",
        "cross-player-discard",
    }
    if not isinstance(partitions, Mapping) or set(partitions) != expected_names:
        raise FeatureDeviationError("v2 split partitions differ")
    seen: set[str] = set()
    for name in sorted(expected_names):
        row = partitions[name]
        sessions = row.get("session_ids") if isinstance(row, Mapping) else None
        if (
            not isinstance(sessions, list)
            or sessions != sorted(sessions)
            or len(sessions) != len(set(sessions))
            or row.get("games") != len(sessions)
            or row.get("session_ids_identity") != canonical_sha256(sessions)
            or seen & set(sessions)
        ):
            raise FeatureDeviationError(f"v2 split partition differs: {name}")
        seen.update(sessions)
    players = split.get("player_membership")
    if not isinstance(players, Mapping):
        raise FeatureDeviationError("v2 player membership absent")
    exploration_players = set(players["research-exploration"]["player_keys"])
    confirmation_players = set(players["research-confirmation"]["player_keys"])
    if exploration_players & confirmation_players:
        raise FeatureDeviationError("v2 player membership overlaps")
    sample = split.get("exploration_sample")
    sample_ids = sample.get("session_ids") if isinstance(sample, Mapping) else None
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != 1024
        or not set(sample_ids).issubset(
            set(partitions["research-exploration"]["session_ids"])
        )
        or sample.get("session_ids_identity") != canonical_sha256(sample_ids)
    ):
        raise FeatureDeviationError("v2 exploration sample differs")
    return split, file_sha


def _decision_counts(record: CorpusRecord) -> tuple[int, int]:
    return (record.move_count + 1) // 2, record.move_count // 2


def _player_decisions(
    records: Sequence[CorpusRecord],
) -> Counter[str]:
    values: Counter[str] = Counter()
    for record in records:
        white, black = _decision_counts(record)
        values[record.player_keys[0]] += white
        values[record.player_keys[1]] += black
    return values


def _arm_metrics(
    records: Sequence[CorpusRecord],
    assigned_players: set[str],
) -> dict[str, Any]:
    decisions = _player_decisions(records)
    values = list(decisions.values())
    return {
        "assigned_players": len(assigned_players),
        "participating_players": len(decisions),
        "games": len(records),
        "decisions": sum(values),
        "player_decision_concentration": concentration(values),
    }


def _split_records(
    records: Sequence[CorpusRecord],
    confirmation_players: set[str],
) -> tuple[list[CorpusRecord], list[CorpusRecord], list[CorpusRecord]]:
    exploration: list[CorpusRecord] = []
    confirmation: list[CorpusRecord] = []
    discarded: list[CorpusRecord] = []
    for record in records:
        left = record.player_keys[0] in confirmation_players
        right = record.player_keys[1] in confirmation_players
        if left and right:
            confirmation.append(record)
        elif not left and not right:
            exploration.append(record)
        else:
            discarded.append(record)
    return exploration, confirmation, discarded


def _player_graph(records: Sequence[CorpusRecord]) -> nx.Graph:
    graph = nx.Graph()
    for record in records:
        left, right = record.player_keys
        graph.add_node(left)
        graph.add_node(right)
        if left == right:
            continue
        previous = graph.get_edge_data(left, right, {}).get("games", 0)
        graph.add_edge(left, right, games=previous + 1)
    return graph


def _initial_confirmation(
    graph: nx.Graph,
    communities: Sequence[set[str]],
    *,
    target_count: int,
    locked_exploration: set[str],
    seed: str,
) -> set[str]:
    """Build a community-stratified exact-size initialization."""
    eligible = set(graph) - locked_exploration
    if target_count > len(eligible):
        raise FeatureDeviationError("confirmation target exceeds unlocked players")
    ranked_communities = sorted(
        communities,
        key=lambda group: _hash_rank(seed, canonical_sha256(sorted(group))),
    )
    selected: set[str] = set()
    remaining_eligible = len(eligible)
    remaining_target = target_count
    for index, community in enumerate(ranked_communities):
        candidates = sorted(
            community & eligible,
            key=lambda player: _hash_rank(f"{seed}:{index}", player),
        )
        if not candidates:
            continue
        count = round(len(candidates) * remaining_target / remaining_eligible)
        count = min(len(candidates), max(0, count))
        selected.update(candidates[:count])
        remaining_target -= count
        remaining_eligible -= len(candidates)
    if remaining_target:
        unused = sorted(
            eligible - selected,
            key=lambda player: _hash_rank(f"{seed}:fill", player),
        )
        selected.update(unused[:remaining_target])
    if len(selected) != target_count or selected & locked_exploration:
        raise FeatureDeviationError("initial rebalanced membership differs")
    return selected


def _largest_remainder_community_targets(
    communities: Sequence[set[str]],
    *,
    target_count: int,
    locked_exploration: set[str],
) -> list[int]:
    unlocked = [len(group - locked_exploration) for group in communities]
    total_players = sum(len(group) for group in communities)
    ratio = target_count / total_players
    raw = [len(group) * ratio for group in communities]
    targets = [
        min(unlocked[index], math.floor(value)) for index, value in enumerate(raw)
    ]
    remaining = target_count - sum(targets)
    ranked = sorted(
        range(len(communities)),
        key=lambda index: (
            -(raw[index] - math.floor(raw[index])),
            canonical_sha256(sorted(communities[index])),
        ),
    )
    while remaining:
        progressed = False
        for index in ranked:
            if targets[index] >= unlocked[index]:
                continue
            targets[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise FeatureDeviationError("community targets cannot reach exact size")
    return targets


def _activity_balanced_confirmation(
    communities: Sequence[set[str]],
    player_decisions: Mapping[str, int],
    *,
    target_count: int,
    locked_exploration: set[str],
    seed: str,
) -> set[str]:
    """Balance decision mass within every community at an exact player count."""
    targets = _largest_remainder_community_targets(
        communities,
        target_count=target_count,
        locked_exploration=locked_exploration,
    )
    total_players = sum(len(group) for group in communities)
    global_ratio = target_count / total_players
    selected: set[str] = set()
    for index, community in enumerate(communities):
        candidates = sorted(
            community - locked_exploration,
            key=lambda player: (
                -int(player_decisions.get(player, 0)),
                _hash_rank(f"{seed}:{index}", player),
            ),
        )
        required = targets[index]
        selected_weight = 0
        processed_weight = 0
        for offset, player in enumerate(candidates):
            weight = int(player_decisions.get(player, 0))
            remaining_slots = required - sum(
                candidate in selected for candidate in candidates[:offset]
            )
            remaining_players = len(candidates) - offset
            must_select = remaining_slots == remaining_players
            may_select = remaining_slots > 0
            target_weight = (processed_weight + weight) * global_ratio
            if must_select or (
                may_select
                and abs(selected_weight + weight - target_weight)
                <= abs(selected_weight - target_weight)
            ):
                selected.add(player)
                selected_weight += weight
            processed_weight += weight
        if len(selected & community) != required:
            raise FeatureDeviationError("activity-balanced community size differs")
    if len(selected) != target_count or selected & locked_exploration:
        raise FeatureDeviationError("activity-balanced membership differs")
    return selected


def _flip_delta(graph: nx.Graph, player: str, side: set[str]) -> int:
    delta = 0
    player_in_side = player in side
    for opponent, data in graph[player].items():
        weight = int(data.get("games", 1))
        same = player_in_side == (opponent in side)
        delta += weight if same else -weight
    return delta


def _improve_fixed_size_cut(
    graph: nx.Graph,
    confirmation: set[str],
    *,
    locked_exploration: set[str],
    maximum_iterations: int,
    shortlist: int,
) -> tuple[set[str], int]:
    """Deterministically reduce cut weight with fixed-size pair swaps."""
    side = set(confirmation)
    iterations = 0
    for _ in range(maximum_iterations):
        confirmation_ranked = sorted(
            ((_flip_delta(graph, player, side), player) for player in side),
            key=lambda row: (row[0], row[1]),
        )[:shortlist]
        exploration_ranked = sorted(
            (
                (_flip_delta(graph, player, side), player)
                for player in set(graph) - side - locked_exploration
            ),
            key=lambda row: (row[0], row[1]),
        )[:shortlist]
        best: tuple[int, str, str] | None = None
        for left_delta, left in confirmation_ranked:
            for right_delta, right in exploration_ranked:
                edge = graph.get_edge_data(left, right, {}).get("games", 0)
                candidate = (left_delta + right_delta + 2 * int(edge), left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] >= 0:
            break
        _delta, leave_confirmation, enter_confirmation = best
        side.remove(leave_confirmation)
        side.add(enter_confirmation)
        iterations += 1
    if side & locked_exploration:
        raise FeatureDeviationError("locked pilot player moved to confirmation")
    return side, iterations


def _cut_games(graph: nx.Graph, confirmation: set[str]) -> int:
    return sum(
        int(data.get("games", 1))
        for left, right, data in graph.edges(data=True)
        if (left in confirmation) != (right in confirmation)
    )


def _candidate(
    records: Sequence[CorpusRecord],
    graph: nx.Graph,
    communities: Sequence[set[str]],
    *,
    ratio: float,
    locked_exploration: set[str],
    player_decisions: Mapping[str, int],
    seed: str,
    algorithm: str,
    maximum_iterations: int,
    shortlist: int,
) -> dict[str, Any]:
    players = set(graph)
    target = round(len(players) * ratio)
    if algorithm == "cut_only_pair_swap_v1":
        initial = _initial_confirmation(
            graph,
            communities,
            target_count=target,
            locked_exploration=locked_exploration,
            seed=seed,
        )
        confirmation, iterations = _improve_fixed_size_cut(
            graph,
            initial,
            locked_exploration=locked_exploration,
            maximum_iterations=maximum_iterations,
            shortlist=shortlist,
        )
    elif algorithm == "community_activity_balance_v2":
        confirmation = _activity_balanced_confirmation(
            communities,
            player_decisions,
            target_count=target,
            locked_exploration=locked_exploration,
            seed=seed,
        )
        iterations = 0
    else:
        raise FeatureDeviationError("unknown rebalance algorithm")
    exploration_records, confirmation_records, discarded = _split_records(
        records, confirmation
    )
    exploration_players = players - confirmation
    exploration_metrics = _arm_metrics(exploration_records, exploration_players)
    confirmation_metrics = _arm_metrics(confirmation_records, confirmation)
    minimum_kish = min(
        exploration_metrics["player_decision_concentration"]["kish_effective_units"],
        confirmation_metrics["player_decision_concentration"]["kish_effective_units"],
    )
    return {
        "candidate_id": f"community-cut-confirmation-{int(round(ratio * 100))}",
        "target_confirmation_player_fraction": ratio,
        "optimization_seed": seed,
        "algorithm": algorithm,
        "pair_swap_iterations": iterations,
        "cut_games_graph_weight": _cut_games(graph, confirmation),
        "research-exploration": exploration_metrics,
        "research-confirmation": confirmation_metrics,
        "cross-player-discard": {
            "games": len(discarded),
            "decisions": sum(record.move_count for record in discarded),
            "game_fraction": len(discarded) / len(records),
        },
        "selection_score_minimum_kish_effective_players": minimum_kish,
        "confirmation_player_keys": sorted(confirmation),
    }


def _existing_split_metrics(
    records: Sequence[CorpusRecord],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    players = split["player_membership"]
    exploration_players = set(players["research-exploration"]["player_keys"])
    confirmation_players = set(players["research-confirmation"]["player_keys"])
    exploration, confirmation, discarded = _split_records(records, confirmation_players)
    result = {
        "candidate_id": "frozen-v1-hash-75-25",
        "research-exploration": _arm_metrics(exploration, exploration_players),
        "research-confirmation": _arm_metrics(confirmation, confirmation_players),
        "cross-player-discard": {
            "games": len(discarded),
            "decisions": sum(record.move_count for record in discarded),
            "game_fraction": len(discarded) / len(records),
        },
    }
    result["selection_score_minimum_kish_effective_players"] = min(
        result["research-exploration"]["player_decision_concentration"][
            "kish_effective_units"
        ],
        result["research-confirmation"]["player_decision_concentration"][
            "kish_effective_units"
        ],
    )
    return result


def _precision_analysis(metrics: Mapping[str, Any]) -> dict[str, Any]:
    confirmation = metrics["research-confirmation"]
    actual_players = int(confirmation["participating_players"])
    kish = float(confirmation["player_decision_concentration"]["kish_effective_units"])
    z_sum = 1.959963984540054 + 0.8416212335729143

    def row(n: float) -> dict[str, Any]:
        coefficient = z_sum / math.sqrt(n)
        return {
            "effective_n": n,
            "mde_per_unit_player_level_standard_deviation": coefficient,
            "maximum_sd_for_0_01_nat_true_effect_to_reach_80_percent_power": (
                0.01 / coefficient
            ),
            "D_to_L_true_difference_needed_for_2pp_lower_bound": {
                "if_sd_0_25": 0.02 + coefficient * 0.25,
                "if_sd_0_50": 0.02 + coefficient * 0.50,
                "if_sd_1_00": 0.02 + coefficient,
            },
        }

    return {
        "method": (
            "normal-approximation planning coefficient "
            "(z_0.975 + z_0.80) / sqrt(N); no outcome variance read"
        ),
        "equal_player_upper_structural_count": row(float(actual_players)),
        "decision_weighted_kish_conservative_proxy": row(kish),
        "preexisting_thresholds": {
            "minimum_log_loss_reduction_nats": 0.01,
            "log_loss_lower_95_must_exceed_zero": True,
            "minimum_D_to_L_top_bottom_difference": 0.02,
            "D_to_L_lower_95_must_meet_minimum": True,
        },
        "structural_reachability_decision": (
            "not_certified_fail_closed_without_outcome_variance_or_D_to_L_support"
        ),
        "mathematically_impossible_claim": False,
        "reason": (
            "log loss is unbounded without a frozen probability floor, and event "
            "support plus player-level variance are prohibited result variables"
        ),
    }


def analyze_and_rebalance(
    *,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    v1_split: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure v1 and frozen structural candidates without raw content."""
    if boundary.file_sha256 != EXPECTED_F0D0_FILE_SHA256:
        raise FeatureDeviationError("F0-D0 manifest identity differs")
    if official_membership.get("membership_identity") != (
        EXPECTED_B2_MEMBERSHIP_IDENTITY
    ):
        raise FeatureDeviationError("B2 membership identity differs")
    if v1_split.get("split_identity") != V1_SPLIT_IDENTITY:
        raise FeatureDeviationError("v1 research split identity differs")
    if plan.get("schema_version") != DESIGN_PLAN_SCHEMA:
        raise FeatureDeviationError("design-round plan schema differs")

    train_ids = official_membership["partitions"]["train"]["session_ids"]
    if len(train_ids) != EXPECTED_B2_COUNTS["train"]:
        raise FeatureDeviationError("official train membership differs")
    by_session = {record.session_id: record for record in boundary.records}
    try:
        records = [by_session[str(session)] for session in train_ids]
    except KeyError as exc:
        raise FeatureDeviationError("train record absent from F0-D0") from exc
    if any(not record.behavior_eligible for record in records):
        raise FeatureDeviationError("train record is behavior-ineligible")

    pilot_ids = set(v1_split["exploration_pilot"]["session_ids"])
    locked_players = {
        player
        for record in records
        if record.session_id in pilot_ids
        for player in record.player_keys
    }
    if len(pilot_ids) != 128:
        raise FeatureDeviationError("v1 pilot size differs")

    graph = _player_graph(records)
    all_player_decisions = _player_decisions(records)
    rebalance = plan["rebalance"]
    communities = [
        set(value)
        for value in nx.community.louvain_communities(
            graph,
            weight="games",
            resolution=float(rebalance["louvain_resolution"]),
            threshold=1e-7,
            seed=int(rebalance["louvain_seed"]),
        )
    ]
    communities.sort(key=lambda group: (-len(group), canonical_sha256(sorted(group))))
    candidates = []
    for ratio in rebalance["candidate_confirmation_player_fractions"]:
        candidates.append(
            _candidate(
                records,
                graph,
                communities,
                ratio=float(ratio),
                locked_exploration=locked_players,
                player_decisions=all_player_decisions,
                seed=f"{rebalance['candidate_seed']}:{ratio}",
                algorithm=str(
                    rebalance.get("algorithm_version", "cut_only_pair_swap_v1")
                ),
                maximum_iterations=int(rebalance["maximum_pair_swap_iterations"]),
                shortlist=int(rebalance["pair_swap_shortlist_per_side"]),
            )
        )
    v1_metrics = _existing_split_metrics(records, v1_split)
    candidates.sort(
        key=lambda row: (
            -float(row["selection_score_minimum_kish_effective_players"]),
            int(row["cross-player-discard"]["games"]),
            str(row["candidate_id"]),
        )
    )
    eligible_candidates = list(candidates)
    if rebalance.get("v1_comparator_eligible_for_selection") is True:
        eligible_candidates.append(v1_metrics)
    eligible_candidates.sort(
        key=lambda row: (
            -float(row["selection_score_minimum_kish_effective_players"]),
            int(row["cross-player-discard"]["games"]),
            str(row["candidate_id"]),
        )
    )
    chosen = eligible_candidates[0]
    for row in candidates:
        if row.get("candidate_id") != chosen.get("candidate_id"):
            row.pop("confirmation_player_keys", None)
    if chosen["candidate_id"] == "frozen-v1-hash-75-25":
        chosen_confirmation = set(
            v1_split["player_membership"]["research-confirmation"]["player_keys"]
        )
    else:
        chosen_confirmation = set(chosen.pop("confirmation_player_keys"))
    for row in candidates:
        row.pop("confirmation_player_keys", None)
    all_players = set(graph)
    chosen_exploration = all_players - chosen_confirmation
    exploration, confirmation, discarded = _split_records(records, chosen_confirmation)
    extension = plan["exploration_extension"]
    ranked_exploration = sorted(
        (record.session_id for record in exploration),
        key=lambda session: _hash_rank(str(extension["sample_seed"]), session),
    )
    old_pilot = sorted(pilot_ids)
    extra_needed = int(extension["total_games"]) - len(old_pilot)
    extras = [session for session in ranked_exploration if session not in pilot_ids][
        :extra_needed
    ]
    sample = sorted(old_pilot + extras)
    if len(sample) != int(extension["total_games"]):
        raise FeatureDeviationError("extended exploration sample is undersized")

    split_payload = {
        "schema_version": rebalance.get("output_split_schema", SPLIT_V2_SCHEMA),
        "status": "frozen_before_extended_exploration_or_confirmation",
        "design_round_plan_identity": plan["plan_identity"],
        "supersedes_for_future_work_only": V1_SPLIT_IDENTITY,
        "v1_split_remains_immutable": True,
        "input_boundary": {
            "f0d0_corpus_identity": EXPECTED_F0D0_CORPUS_IDENTITY,
            "f0d0_manifest_identity": EXPECTED_F0D0_MANIFEST_IDENTITY,
            "f0d0_manifest_file_sha256": EXPECTED_F0D0_FILE_SHA256,
            "b2_membership_identity": EXPECTED_B2_MEMBERSHIP_IDENTITY,
        },
        "assignment_rule": {
            "unit": "source_domain_player_key",
            "algorithm": "community-stratified fixed-size cut with deterministic pair-swap improvement",
            "selected_candidate": chosen["candidate_id"],
            "selection_rule": rebalance["selection_rule"],
            "old_128_pilot_players_locked_to_exploration": True,
            "cross_arm_game_policy": "discard",
        },
        "partitions": {
            "research-exploration": {
                "games": len(exploration),
                "logical_plies": sum(record.move_count for record in exploration),
                "session_ids_identity": canonical_sha256(
                    sorted(record.session_id for record in exploration)
                ),
                "session_ids": sorted(record.session_id for record in exploration),
            },
            "research-confirmation": {
                "games": len(confirmation),
                "logical_plies": sum(record.move_count for record in confirmation),
                "session_ids_identity": canonical_sha256(
                    sorted(record.session_id for record in confirmation)
                ),
                "session_ids": sorted(record.session_id for record in confirmation),
            },
            "cross-player-discard": {
                "games": len(discarded),
                "logical_plies": sum(record.move_count for record in discarded),
                "session_ids_identity": canonical_sha256(
                    sorted(record.session_id for record in discarded)
                ),
                "session_ids": sorted(record.session_id for record in discarded),
            },
        },
        "player_membership": {
            "research-exploration": {
                "players": len(chosen_exploration),
                "player_keys_identity": canonical_sha256(sorted(chosen_exploration)),
                "player_keys": sorted(chosen_exploration),
            },
            "research-confirmation": {
                "players": len(chosen_confirmation),
                "player_keys_identity": canonical_sha256(sorted(chosen_confirmation)),
                "player_keys": sorted(chosen_confirmation),
            },
            "pairwise_player_overlap": 0,
        },
        "exploration_sample": {
            "games": len(sample),
            "includes_v1_pilot_games": len(old_pilot),
            "additional_games": len(extras),
            "selection": "retain v1 pilot, then lowest SHA-256(seed NUL session_id) ranks",
            "seed": extension["sample_seed"],
            "session_ids_identity": canonical_sha256(sample),
            "session_ids": sample,
        },
        "access_state": {
            "built_from_f0d0_and_memberships_only": True,
            "raw_game_files_opened": 0,
            "human_action_or_feature_reads": 0,
            "research_confirmation_content_reads": 0,
            "selection_content_reads": 0,
            "confirmation_content_reads": 0,
            "final_test_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
    }
    manifest = {
        "schema_version": REBALANCE_SCHEMA,
        "status": "structural_only_confirmation_not_opened",
        "design_round_plan_identity": plan["plan_identity"],
        "v1_identities": {
            "plan": V1_PLAN_IDENTITY,
            "split": V1_SPLIT_IDENTITY,
            "exploration": V1_EXPLORATION_IDENTITY,
        },
        "outcome_blindness_contract": {
            "inputs_used": [
                "session_id",
                "player_keys",
                "move_count",
                "official train membership",
                "v1 pilot membership",
            ],
            "inputs_not_read": [
                "raw game files",
                "human chosen actions",
                "Malom tiers",
                "tier-loss events",
                "action feature values",
                "recorded or replayed outcomes",
            ],
        },
        "v1_precision_reachability": {
            "structural_metrics": v1_metrics,
            "precision": _precision_analysis(v1_metrics),
        },
        "graph": {
            "players": graph.number_of_nodes(),
            "unique_opponent_pairs": graph.number_of_edges(),
            "communities": len(communities),
            "modularity": nx.community.modularity(
                graph,
                communities,
                weight="games",
                resolution=float(rebalance["louvain_resolution"]),
            ),
            "community_player_sizes": sorted(
                (len(group) for group in communities), reverse=True
            ),
            "old_pilot_locked_players": len(locked_players),
        },
        "candidate_selection_rule": rebalance["selection_rule"],
        "v1_comparator": v1_metrics,
        "candidates_ranked": candidates,
        "selected_candidate": chosen["candidate_id"],
        "selected_precision_reachability": _precision_analysis(chosen),
        "access_audit": {
            "raw_game_files_opened": 0,
            "human_outcome_or_action_variables_read": 0,
            "malom_queries": 0,
            "research_confirmation_content_reads": 0,
            "selection_content_reads": 0,
            "confirmation_content_reads": 0,
            "final_test_content_reads": 0,
            "source_pool_2eb04f54_records_read_or_consumed": 0,
        },
    }
    return manifest, split_payload


def _immediate_mill_destinations(board: BoardState, color: str) -> set[str]:
    result: set[str] = set()
    for line in MILLS:
        values = [board.positions[position] for position in line]
        if values.count(color) == 2 and values.count("") == 1:
            result.add(line[values.index("")])
    return result


def _mobility(board: BoardState, color: str) -> int:
    if board.phase == "place":
        return len(board.legal_placements(color))
    return len(board.legal_moves(color))


def _captured_threat_lines(
    board: BoardState, capture: str | None, opponent: str
) -> int:
    if capture is None:
        return 0
    return sum(
        1
        for line in MILLS
        if capture in line
        and [board.positions[position] for position in line].count(opponent) >= 2
    )


def extended_action_feature_scores(
    board: BoardState, move: Mapping[str, Any]
) -> dict[str, float]:
    """Return the revised ten-term visible exploratory feature panel."""
    actor = board.turn
    opponent = "B" if actor == "W" else "W"
    normalized = {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }
    before_actor_threats = _immediate_mill_destinations(board, actor)
    before_opponent_threats = _immediate_mill_destinations(board, opponent)
    before_actor_mobility = _mobility(board, actor)
    before_opponent_mobility = _mobility(board, opponent)
    after = board.apply_move(normalized)
    after_actor_threats = _immediate_mill_destinations(after, actor)
    after_opponent_threats = _immediate_mill_destinations(after, opponent)
    destination = normalized["to"]
    source = normalized["from"]
    capture = normalized["capture"]
    closes_mill = bool(destination and after.is_mill(str(destination), actor))
    values = {
        "source_degree": len(ADJACENCY.get(source, ())) / 4.0 if source else 0.0,
        "destination_degree": (
            len(ADJACENCY.get(destination, ())) / 4.0 if destination else 0.0
        ),
        "capture_degree": len(ADJACENCY.get(capture, ())) / 4.0 if capture else 0.0,
        "closes_mill": float(closes_mill),
        "opponent_immediate_mill_destinations_removed": float(
            len(before_opponent_threats - after_opponent_threats)
        ),
        "creates_mill_fork": float(
            len(after_actor_threats) >= 2
            and len(after_actor_threats - before_actor_threats) >= 1
        ),
        "new_own_potential_mills": float(
            _potential_mills(after, actor) - _potential_mills(board, actor)
        ),
        "own_mobility_delta": float(
            (_mobility(after, actor) - before_actor_mobility) / 24.0
        ),
        "opponent_mobility_reduction": float(
            (before_opponent_mobility - _mobility(after, opponent)) / 24.0
        ),
        "captured_opponent_threat_lines": float(
            _captured_threat_lines(board, capture, opponent) / 2.0
        ),
    }
    if tuple(values) != V2_FEATURE_NAMES or any(
        not math.isfinite(value) for value in values.values()
    ):
        raise FeatureDeviationError("extended visible feature contract differs")
    return values


def _query_inventory(
    board: BoardState, database: MalomDB
) -> tuple[str, list[tuple[Mapping[str, Any], OracleMoveValue]], int]:
    """Keep the positional oracle's board-first interface explicit and tested."""
    return _oracle_inventory_positional(board, database)


@dataclass
class CollinearityAudit:
    choice_sets: int = 0
    exact_affine_choice_sets: int = 0
    maximum_within_choice_residual_range: float = 0.0

    def observe(self, rows: Sequence[Mapping[str, float]]) -> None:
        residuals = [
            row["material_balance_after"] - row["closes_mill"] / 9.0 for row in rows
        ]
        width = max(residuals) - min(residuals)
        self.choice_sets += 1
        self.maximum_within_choice_residual_range = max(
            self.maximum_within_choice_residual_range, abs(width)
        )
        if math.isclose(width, 0.0, rel_tol=0.0, abs_tol=1e-12):
            self.exact_affine_choice_sets += 1


def run_extended_exploration(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    split: Mapping[str, Any],
    plan: Mapping[str, Any],
    database: MalomDB,
    malom_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only the frozen extended research-exploration sample."""
    if split.get("schema_version") not in {SPLIT_V2_SCHEMA, SPLIT_V3_SCHEMA}:
        raise FeatureDeviationError("research split schema differs")
    if split.get("design_round_plan_identity") != plan.get("plan_identity"):
        raise FeatureDeviationError("v2 split does not bind design plan")
    if malom_snapshot.get("trust_level") != "sector-corrected-v1":
        raise FeatureDeviationError("Malom trust level differs")
    if not database.is_available():
        raise FeatureDeviationError("required Malom database is unavailable")
    expected = int(plan["exploration_extension"]["total_games"])
    sample_ids = split["exploration_sample"]["session_ids"]
    if len(sample_ids) != expected:
        raise FeatureDeviationError("extended sample size differs")

    guard_split = {
        "partitions": split["partitions"],
        "exploration_pilot": {"session_ids": sample_ids},
    }
    access = ExplorationOnlyAccess.from_memberships(official_membership, guard_split)
    by_session = {record.session_id: record for record in boundary.records}
    records = [by_session[session] for session in sample_ids]
    expected_decisions = sum(record.move_count for record in records)
    budget = plan["exploration_extension"]["hard_budget"]
    if expected_decisions > int(budget["maximum_decisions"]):
        raise FeatureDeviationError("frozen sample exceeds decision budget")

    old_features = {
        name: FeatureOpportunity()
        for name in action_feature_scores(
            BoardState.new_game(), {"from": None, "to": "a7", "capture": None}
        )
    }
    new_features = {name: FeatureOpportunity() for name in V2_FEATURE_NAMES}
    player_decisions: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    abstentions: Counter[str] = Counter()
    games_covered: set[str] = set()
    collinearity = CollinearityAudit()
    simultaneous_double_mill_choices = 0
    mill_fork_choices = 0
    covered = 0
    abstained = 0
    query_count = 0
    a_pos_modifiable = 0
    started = time.perf_counter()
    root = Path(repository_root)
    for record in records:
        decisions = access.load_decisions(root, record, boundary)
        if len(decisions) != record.move_count:
            raise FeatureDeviationError("replayed decision count differs")
        for decision in decisions:
            if query_count > int(budget["maximum_queries"]):
                raise FeatureDeviationError("oracle query budget exceeded")
            if time.perf_counter() - started > float(budget["maximum_seconds"]):
                raise FeatureDeviationError("active-time budget exceeded")
            try:
                parent_tier, inventory, queries = _query_inventory(
                    decision.board, database
                )
            except OracleCoverageAbstention as exc:
                abstained += 1
                query_count += exc.query_count
                abstentions[exc.reason] += 1
                continue
            query_count += queries
            chosen_rows = [
                index
                for index, (move, _value) in enumerate(inventory)
                if _move_matches(move, decision.move)
            ]
            if len(chosen_rows) != 1:
                raise FeatureDeviationError("observed action absent from inventory")
            chosen_index = chosen_rows[0]
            safe_indices = {
                index
                for index, (_move, value) in enumerate(inventory)
                if value.outcome == parent_tier
            }
            if not safe_indices:
                raise FeatureDeviationError("A_pos is empty")
            a_pos_modifiable += len(safe_indices) > 1
            chosen_value: OracleMoveValue = inventory[chosen_index][1]
            if WDL_RANK[chosen_value.outcome] < WDL_RANK[parent_tier]:
                transition = f"{parent_tier}->{chosen_value.outcome}"
                if transition not in TRANSITIONS:
                    raise FeatureDeviationError("unexpected tier transition")
                transitions[transition] += 1
            raw_phase = get_game_phase(decision.board, decision.board.turn)
            if raw_phase not in PHASE_NAMES:
                raise FeatureDeviationError("invalid phase")
            phase = PHASE_NAMES[raw_phase]
            old_rows = [
                action_feature_scores(decision.board, move)
                for move, _value in inventory
            ]
            new_rows = [
                extended_action_feature_scores(decision.board, move)
                for move, _value in inventory
            ]
            collinearity.observe(old_rows)
            simultaneous_double_mill_choices += any(
                row["creates_double_mill"] > 0 for row in old_rows
            )
            mill_fork_choices += any(row["creates_mill_fork"] > 0 for row in new_rows)
            for name, opportunity in old_features.items():
                opportunity.observe(
                    scores=[row[name] for row in old_rows],
                    safe_indices=safe_indices,
                    chosen_index=chosen_index,
                    player=decision.actor_player_key,
                    game=decision.game_id,
                    phase=phase,
                    tier=parent_tier,
                    color=decision.board.turn,
                )
            for name, opportunity in new_features.items():
                opportunity.observe(
                    scores=[row[name] for row in new_rows],
                    safe_indices=safe_indices,
                    chosen_index=chosen_index,
                    player=decision.actor_player_key,
                    game=decision.game_id,
                    phase=phase,
                    tier=parent_tier,
                    color=decision.board.turn,
                )
            covered += 1
            player_decisions[decision.actor_player_key] += 1
            phase_counts[phase] += 1
            tier_counts[parent_tier] += 1
            color_counts[decision.board.turn] += 1
            games_covered.add(decision.game_id)
    elapsed = time.perf_counter() - started
    if covered + abstained != expected_decisions:
        raise FeatureDeviationError("extended oracle accounting differs")
    if query_count > int(budget["maximum_queries"]):
        raise FeatureDeviationError("oracle query budget exceeded")

    return {
        "schema_version": EXTENSION_SCHEMA,
        "status": "extended_exploration_only_confirmation_unopened",
        "design_round_plan_identity": plan["plan_identity"],
        "split_identity": split["split_identity"],
        "claim_boundary": {
            "exploratory_only": True,
            "safe_set": "A_pos",
            "state_safety": "positional-only",
            "a_allow_claim": False,
            "F0_H0_stop_remains_effective": True,
            "confirmatory_evidence": False,
        },
        "sample": {
            "games": len(records),
            "games_with_covered_decisions": len(games_covered),
            "session_ids_identity": split["exploration_sample"]["session_ids_identity"],
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
            "queries_per_second": query_count / elapsed if elapsed else None,
            "coverage": _rate(covered, expected_decisions),
            "abstention_reasons": _serialize_counter(abstentions),
            "malom_snapshot": dict(malom_snapshot),
        },
        "positional_labels": {
            "a_pos_cardinality_greater_than_one": _rate(a_pos_modifiable, covered),
            "phase_counts": _serialize_counter(phase_counts),
            "tier_counts": _serialize_counter(tier_counts),
            "color_counts": _serialize_counter(color_counts),
            "chosen_tier_loss_counts": {
                transition: int(transitions[transition]) for transition in TRANSITIONS
            },
            "zero_events_not_smoothed": True,
        },
        "v1_feature_panel": {
            name: _feature_summary(value, covered)
            for name, value in old_features.items()
        },
        "v2_candidate_feature_panel": {
            name: _feature_summary(value, covered)
            for name, value in new_features.items()
        },
        "structural_diagnostics": {
            "simultaneous_double_mill_choice_sets": simultaneous_double_mill_choices,
            "mill_fork_choice_sets": mill_fork_choices,
            "closes_mill_material_balance_after": {
                "choice_sets": collinearity.choice_sets,
                "exact_affine_choice_sets": collinearity.exact_affine_choice_sets,
                "maximum_within_choice_residual_range": (
                    collinearity.maximum_within_choice_residual_range
                ),
                "tested_identity": (
                    "material_balance_after - closes_mill / 9 is constant "
                    "within each complete atomic choice set"
                ),
            },
        },
        "access_audit": {
            "successful_accesses": {
                f"{partition}:{kind}": int(value)
                for (partition, kind), value in sorted(
                    access.successful_accesses.items()
                )
            },
            "denied_attempts": {
                f"{partition}:{kind}": int(value)
                for (partition, kind), value in sorted(access.denied_attempts.items())
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
        },
    }


__all__ = [
    "DESIGN_PLAN_SCHEMA",
    "EXTENSION_SCHEMA",
    "REBALANCE_SCHEMA",
    "SPLIT_V2_SCHEMA",
    "SPLIT_V3_SCHEMA",
    "V2_FEATURE_NAMES",
    "CollinearityAudit",
    "analyze_and_rebalance",
    "extended_action_feature_scores",
    "load_design_plan",
    "load_split_v2",
    "run_extended_exploration",
]
