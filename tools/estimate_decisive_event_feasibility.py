"""Estimate decisive-event yield for a proposed no-update evaluation.

This is a planning diagnostic, not a paired-effect power calculator. It treats
an observed decisive-game rate as a plug-in Bernoulli rate and reports how many
decisive games a future sample might contain. A strength comparison still
needs a frozen paired estimand, discordance model, effect threshold, corpus and
error/precision rule.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist
from typing import Any


def wilson_interval(events: int, games: int, confidence: float) -> tuple[float, float]:
    if games <= 0 or not 0 <= events <= games:
        raise ValueError("events and games must satisfy 0 <= events <= games")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    proportion = events / games
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / games
    centre = (proportion + z * z / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / games
            + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def probability_at_least(
    *, sample_games: int, event_rate: float, target_events: int
) -> float:
    if sample_games < 0 or target_events < 0:
        raise ValueError("sample_games and target_events must be non-negative")
    if not 0.0 <= event_rate <= 1.0:
        raise ValueError("event_rate must be between zero and one")
    if target_events == 0:
        return 1.0
    if target_events > sample_games or event_rate == 0.0:
        return 0.0
    if event_rate == 1.0:
        return 1.0

    # Sum P(X < target_events) recursively. The target counts used for this
    # diagnostic are deliberately small, avoiding a dependency on SciPy.
    failure_rate = 1.0 - event_rate
    term = failure_rate**sample_games
    below_target = term
    odds = event_rate / failure_rate
    for count in range(target_events - 1):
        term *= (sample_games - count) / (count + 1) * odds
        below_target += term
    return max(0.0, min(1.0, 1.0 - below_target))


def minimum_games_for_event_yield(
    *, event_rate: float, target_events: int, target_probability: float
) -> int | None:
    if target_events <= 0:
        raise ValueError("target_events must be positive")
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be between zero and one")
    if event_rate == 0.0:
        return None
    if not 0.0 < event_rate <= 1.0:
        raise ValueError("event_rate must be between zero and one")

    lower = target_events
    upper = target_events
    while (
        probability_at_least(
            sample_games=upper,
            event_rate=event_rate,
            target_events=target_events,
        )
        < target_probability
    ):
        upper *= 2
    while lower < upper:
        midpoint = (lower + upper) // 2
        if (
            probability_at_least(
                sample_games=midpoint,
                event_rate=event_rate,
                target_events=target_events,
            )
            >= target_probability
        ):
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def build_report(
    *,
    label: str,
    observed_events: int,
    observed_games: int,
    sample_sizes: list[int],
    target_event_counts: list[int],
    confidence: float,
    target_probability: float,
) -> dict[str, Any]:
    lower, upper = wilson_interval(observed_events, observed_games, confidence)
    rate = observed_events / observed_games
    return {
        "schema": "nmm.decisive-event-feasibility.v1",
        "scope": {
            "planning_diagnostic_only": True,
            "not_paired_effect_power": True,
            "not_launch_authority": True,
        },
        "label": label,
        "observed": {
            "events": observed_events,
            "games": observed_games,
            "eventRate": rate,
            "wilsonConfidence": confidence,
            "wilsonInterval": [lower, upper],
        },
        "candidateSamples": [
            {
                "games": games,
                "expectedEventsPlugIn": games * rate,
                "probabilityZeroEventsPlugIn": (1.0 - rate) ** games,
            }
            for games in sample_sizes
        ],
        "eventYieldThresholds": [
            {
                "targetEvents": target,
                "targetProbability": target_probability,
                "minimumGamesPlugIn": minimum_games_for_event_yield(
                    event_rate=rate,
                    target_events=target,
                    target_probability=target_probability,
                ),
            }
            for target in target_event_counts
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--observed-events", type=int, required=True)
    parser.add_argument("--observed-games", type=int, required=True)
    parser.add_argument("--sample-sizes", default="16,128,256,512,1024")
    parser.add_argument("--target-event-counts", default="10,20,30")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target-probability", type=float, default=0.90)
    args = parser.parse_args()
    report = build_report(
        label=args.label,
        observed_events=args.observed_events,
        observed_games=args.observed_games,
        sample_sizes=[int(value) for value in args.sample_sizes.split(",")],
        target_event_counts=[
            int(value) for value in args.target_event_counts.split(",")
        ],
        confidence=args.confidence,
        target_probability=args.target_probability,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
