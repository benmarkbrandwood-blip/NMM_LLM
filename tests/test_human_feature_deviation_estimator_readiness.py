from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    ChoiceObservation,
    EstimatorAccess,
    EstimatorReadinessError,
    NumericalContract,
    _objective_and_gradient,
    _stable_probabilities,
    _weighted_quantile,
    build_community_crossfit_structure,
    build_crossfit_structure,
    canonicalize_choice_inventory,
    diagnose_separation,
    fit_conditional_logit,
    load_effective_readiness_plan,
    player_cluster_bootstrap,
    standardize_from_training_fold,
)


def _choice(
    *,
    player: str,
    features: list[list[float]],
    chosen: int,
    fold: int = 0,
    outcomes: tuple[str, ...] | None = None,
) -> ChoiceObservation:
    rows = np.asarray(features, dtype=np.float64)
    return ChoiceObservation(
        player_key=player,
        game_id=f"game-{player}",
        decision_index=0,
        fold=fold,
        features=rows,
        chosen_index=chosen,
        parent_tier="D",
        action_outcomes=outcomes or tuple("D" for _ in features),
        phase="movement",
        color="W",
    )


def test_extreme_softmax_is_finite_and_uses_frozen_reporting_bounds() -> None:
    probabilities, log_sum = _stable_probabilities(
        np.asarray([1000.0, 0.0, -1000.0]),
        floor=1e-12,
        ceiling=0.999999999999,
    )

    assert np.all(np.isfinite(probabilities))
    assert math.isfinite(log_sum)
    assert probabilities[0] == 0.999999999999
    assert probabilities[2] == 1e-12


def test_choice_inventory_is_lexical_and_duplicate_is_fatal() -> None:
    moves = [
        {"from": "d1", "to": "d2", "capture": None},
        {"from": "a1", "to": "d1", "capture": "g1"},
    ]
    rows = [{"x": 2.0}, {"x": 1.0}]

    ordered_moves, ordered_rows, chosen = canonicalize_choice_inventory(
        moves,
        rows,
        moves[0],
        feature_names=("x",),
        maximum_actions=768,
    )

    assert ordered_moves[0]["from"] == "a1"
    assert ordered_rows[:, 0].tolist() == [1.0, 2.0]
    assert chosen == 1
    with pytest.raises(EstimatorReadinessError, match="duplicate"):
        canonicalize_choice_inventory(
            [moves[0], dict(moves[0])],
            rows,
            moves[0],
            feature_names=("x",),
            maximum_actions=768,
        )


@pytest.mark.parametrize(
    ("moves", "rows", "chosen", "message"),
    [
        ([], [], {"from": None, "to": "a7", "capture": None}, "empty"),
        (
            [{"from": None, "to": "a7", "capture": None}],
            [{"x": float("nan")}],
            {"from": None, "to": "a7", "capture": None},
            "nonfinite",
        ),
        (
            [{"from": None, "to": "a7", "capture": None}],
            [{"x": 1.0}],
            {"from": None, "to": "d7", "capture": None},
            "observed",
        ),
    ],
)
def test_invalid_choice_inputs_fail_closed(
    moves: list[dict[str, str | None]],
    rows: list[dict[str, float]],
    chosen: dict[str, str | None],
    message: str,
) -> None:
    with pytest.raises(EstimatorReadinessError, match=message):
        canonicalize_choice_inventory(
            moves,
            rows,
            chosen,
            feature_names=("x",),
            maximum_actions=768,
        )


def test_single_and_oversized_choice_sets_are_not_neutral_defaults() -> None:
    single = _choice(player="p", features=[[1.0]], chosen=0)
    assert single.is_degenerate is True
    moves = [{"from": None, "to": f"x{index}", "capture": None} for index in range(769)]
    rows = [{"x": float(index)} for index in range(769)]
    with pytest.raises(EstimatorReadinessError, match="maximum"):
        canonicalize_choice_inventory(
            moves,
            rows,
            moves[0],
            feature_names=("x",),
            maximum_actions=768,
        )


def test_standardization_uses_only_training_fold_and_rejects_zero_scale() -> None:
    training = [
        _choice(player="a", features=[[0.0], [2.0]], chosen=0),
        _choice(player="b", features=[[2.0], [4.0]], chosen=1),
    ]
    held_out = [_choice(player="c", features=[[1000.0], [2000.0]], chosen=0)]

    mean, scale = standardize_from_training_fold(training, columns=(0,))

    assert np.allclose(mean, [2.0])
    assert np.allclose(scale, [math.sqrt(2.0)])
    assert not np.isclose(mean[0], np.vstack([c.features for c in held_out]).mean())
    with pytest.raises(EstimatorReadinessError, match="scale"):
        standardize_from_training_fold(
            [_choice(player="z", features=[[1.0], [1.0]], chosen=0)],
            columns=(0,),
        )


def test_objective_is_equal_player_normalized_and_penalizes_all_terms() -> None:
    choices = [
        _choice(player="heavy", features=[[0.0], [1.0]], chosen=1),
        _choice(player="heavy", features=[[0.0], [1.0]], chosen=1),
        _choice(player="light", features=[[0.0], [1.0]], chosen=0),
    ]
    beta = np.asarray([0.5])
    mean = np.asarray([0.0])
    scale = np.asarray([1.0])

    objective, gradient = _objective_and_gradient(
        beta,
        choices,
        columns=(0,),
        mean=mean,
        scale=scale,
        ridge_lambda=0.01,
    )
    p1 = math.exp(0.5) / (1.0 + math.exp(0.5))
    expected = (-math.log(p1) - math.log(1.0 - p1)) / 2.0
    expected += 0.5 * 0.01 * 0.5**2

    assert math.isclose(objective, expected, rel_tol=0.0, abs_tol=1e-12)
    assert gradient.shape == (1,)
    assert math.isfinite(gradient[0])


def test_optimizer_converges_deterministically_on_identified_fixture() -> None:
    contract = NumericalContract.for_tests(maximum_iterations=100)
    choices = [
        _choice(player=f"p{index}", features=[[0.0], [1.0]], chosen=index % 2)
        for index in range(20)
    ]
    mean, scale = standardize_from_training_fold(choices, columns=(0,))

    first = fit_conditional_logit(
        choices,
        columns=(0,),
        mean=mean,
        scale=scale,
        contract=contract,
    )
    second = fit_conditional_logit(
        choices,
        columns=(0,),
        mean=mean,
        scale=scale,
        contract=contract,
    )

    assert first.converged is True
    assert np.allclose(first.coefficients, second.coefficients, atol=1e-10)
    assert math.isclose(first.objective, second.objective, abs_tol=1e-12)


def test_optimizer_line_search_failure_is_not_swallowed() -> None:
    contract = NumericalContract.for_tests(
        maximum_iterations=2,
        maximum_line_search_steps=0,
    )
    choices = [
        _choice(player="p", features=[[0.0], [1.0]], chosen=1),
        _choice(player="q", features=[[0.0], [1.0]], chosen=1),
    ]
    mean, scale = standardize_from_training_fold(choices, columns=(0,))

    with pytest.raises(EstimatorReadinessError, match="line search"):
        fit_conditional_logit(
            choices,
            columns=(0,),
            mean=mean,
            scale=scale,
            contract=contract,
        )


def test_separation_and_nonfinite_coefficients_fail_closed() -> None:
    with pytest.raises(EstimatorReadinessError, match="coefficient"):
        diagnose_separation(
            coefficients=np.asarray([21.0]),
            information=np.asarray([[1.0]]),
            chosen_probabilities=np.asarray([0.5]),
            contract=NumericalContract.for_tests(),
        )
    with pytest.raises(EstimatorReadinessError, match="finite"):
        diagnose_separation(
            coefficients=np.asarray([float("nan")]),
            information=np.asarray([[1.0]]),
            chosen_probabilities=np.asarray([0.5]),
            contract=NumericalContract.for_tests(),
        )


def test_crossfit_structure_is_player_isolated_and_discards_cross_games() -> None:
    players = [f"p{index}" for index in range(10)]
    games = [
        (f"g{index}", players[index], players[(index + 1) % len(players)], 10)
        for index in range(len(players))
    ]

    result = build_crossfit_structure(
        assigned_players=players,
        games=games,
        folds=5,
        capacities=(2, 2, 2, 2, 2),
        fold_seed="fold-test",
        sample_seed="sample-test",
        maximum_games_per_fold=2,
    )

    membership = result["player_fold"]
    assert set(membership) == set(players)
    assert all(0 <= fold < 5 for fold in membership.values())
    assert result["cross_fold_games"] + result["same_fold_games"] == 10
    for row in result["sample_games"]:
        assert membership[row["white"]] == membership[row["black"]]


def test_corrected_community_folds_keep_communities_whole() -> None:
    players = [f"p{index}" for index in range(20)]
    games: list[tuple[str, str, str, int]] = []
    for group in range(5):
        members = players[group * 4 : group * 4 + 4]
        for index in range(3):
            games.append((f"g-{group}-{index}", members[index], members[index + 1], 10))

    result = build_community_crossfit_structure(
        assigned_players=players,
        games=games,
        folds=5,
        community_resolution=2.0,
        community_seed=170816,
        sample_seed="sample-test",
        maximum_games_per_fold=10,
    )

    assert result["cross_fold_games"] == 0
    assert result["sample_players"] == 20
    assert all(row["games"] == 3 for row in result["fold_metrics"])


def test_v2_plan_inherits_the_frozen_numerical_contract() -> None:
    root = Path(__file__).resolve().parent.parent

    effective, identities = load_effective_readiness_plan(
        root / "docs/experiments/human-feature-deviation-estimator-readiness-v2.json",
        inherited_v1_path=(
            root
            / "docs/experiments/human-feature-deviation-estimator-readiness-v1.json"
        ),
    )

    assert identities["v2_plan_identity"].startswith("246eef0a")
    assert effective["numerical_contract"]["regularization"]["lambda"] == 0.01
    assert effective["cross_fit_contract"]["community_resolution"] == 2.0
    assert effective["confirmation_execution_authorized"] is False


def test_tracked_crossfit_structure_is_sealed_and_protected() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.pop("structure_identity")

    assert canonical_sha256(value) == identity
    assert (
        identity == "b2ab654856a13d17fbc5256b6395c078e6cd13db9114da390daf720d952e6ae4"
    )
    assert value["structure"]["sample_game_count"] == 6400
    assert value["structure"]["sample_players"] == 980
    assert all(row["games"] == 1280 for row in value["structure"]["fold_metrics"])
    assert value["access_audit"]["research_confirmation_content_reads"] == 0
    assert value["access_audit"]["official_final_test_content_reads"] == 0


def test_access_whitelist_denies_protected_content_before_reader(
    tmp_path: Path,
) -> None:
    access = EstimatorAccess(
        official_partition_by_session={
            "allowed": "train",
            "internal": "train",
            "selection": "selection",
            "official-confirm": "confirmation",
            "final": "final-test",
        },
        research_partition_by_session={
            "allowed": "research-exploration",
            "internal": "research-confirmation",
        },
        allowed_sessions=frozenset({"allowed"}),
    )
    called = False

    def reader() -> str:
        nonlocal called
        called = True
        return str(tmp_path)

    for session in ("internal", "selection", "official-confirm", "final"):
        with pytest.raises(EstimatorReadinessError, match="denied"):
            access.derive(session, access_kind="result", producer=reader)
    assert called is False
    assert access.derive("allowed", access_kind="result", producer=reader)
    assert called is True


def test_weighted_quantile_and_zero_event_bootstrap_do_not_add_events() -> None:
    assert (
        _weighted_quantile(
            np.asarray([0.1, 0.2, 0.9]),
            np.asarray([1.0, 1.0, 1.0]),
            0.8,
        )
        == 0.9
    )
    values = {"a": 0.0, "b": 0.0, "c": 0.0}
    result = player_cluster_bootstrap(
        values,
        replicates=100,
        seed="zero-event-test",
        statistic="mean_and_sd",
    )

    assert result["point_mean"] == 0.0
    assert result["mean_interval"] == [0.0, 0.0]
    assert result["zero_events_not_smoothed"] is True
