"""Deterministic, policy-explicit paired opening prefixes from Sanmill."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryResponse,
    DataQueryState,
    SanmillDataQueryError,
    SanmillDataQuerySession,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_uci import SanmillInstallation
from learned_ai.training.run_contract import canonical_sha256


PAIRED_PREFIX_SCHEMA = "nmm.sanmill-paired-prefix.v1"
PREFIX_SELECTION_SCHEMA = "nmm.sanmill-prefix-selection.v1"
PREFIX_LOGICAL_PLIES = 8
_SOURCE_KINDS = frozenset({"book", "perfect_db", "human_db"})
_CANDIDATE_POLICIES = frozenset(
    {"uniform_candidate", "source_declared", "human_frequency"}
)
_SOURCE_OPERATION = {
    "book": "query_book",
    "perfect_db": "query_perfect_db",
    "human_db": "query_human_db",
}


class SanmillPrefixError(SanmillDataQueryError):
    """Raised when prefix policy, identity, replay, or evidence drifts."""


@dataclass(frozen=True)
class PrefixSourceSpec:
    """One explicit source; mixture assignment happens before generation."""

    kind: str
    candidate_policy: str
    database_path: Path | None = None
    path_lookup_key: str | None = None
    expected_identity_sha256: str | None = None
    cache_sectors: int | None = None
    human_candidate_limit: int | None = None
    human_min_total: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise SanmillPrefixError(f"unsupported prefix source {self.kind!r}")
        if self.candidate_policy not in _CANDIDATE_POLICIES:
            raise SanmillPrefixError(
                f"unsupported candidate policy {self.candidate_policy!r}"
            )
        allowed_policies = {
            "book": {"uniform_candidate", "source_declared"},
            "perfect_db": {"uniform_candidate"},
            "human_db": {"uniform_candidate", "human_frequency"},
        }[self.kind]
        if self.candidate_policy not in allowed_policies:
            raise SanmillPrefixError(
                f"{self.candidate_policy!r} is invalid for {self.kind}"
            )
        if self.expected_identity_sha256 is not None and (
            len(self.expected_identity_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.expected_identity_sha256
            )
        ):
            raise SanmillPrefixError(
                "expected source identity must be a lowercase SHA-256"
            )
        if self.kind == "book":
            if any(
                value is not None
                for value in (
                    self.database_path,
                    self.path_lookup_key,
                    self.cache_sectors,
                    self.human_candidate_limit,
                )
            ) or self.human_min_total != 0:
                raise SanmillPrefixError("book source has database-only settings")
        else:
            if self.database_path is None or not self.database_path.is_absolute():
                raise SanmillPrefixError(
                    f"{self.kind} needs an absolute machine-local database path"
                )
            if not self.path_lookup_key:
                raise SanmillPrefixError(f"{self.kind} needs a path lookup key")
            if self.expected_identity_sha256 is None:
                raise SanmillPrefixError(
                    f"{self.kind} needs a pinned portable identity SHA-256"
                )
        if self.kind != "perfect_db" and self.cache_sectors is not None:
            raise SanmillPrefixError("cache_sectors applies only to Perfect DB")
        if self.cache_sectors is not None and self.cache_sectors <= 0:
            raise SanmillPrefixError("cache_sectors must be positive")
        if self.kind != "human_db" and (
            self.human_candidate_limit is not None or self.human_min_total != 0
        ):
            raise SanmillPrefixError(
                "HumanDB query limits apply only to HumanDB"
            )
        if (
            self.human_candidate_limit is not None
            and self.human_candidate_limit <= 0
        ):
            raise SanmillPrefixError("HumanDB candidate limit must be positive")
        if self.human_min_total < 0:
            raise SanmillPrefixError("HumanDB minimum total cannot be negative")

    def portable_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": self.kind,
            "candidate_policy": self.candidate_policy,
        }
        if self.kind != "book":
            record.update(
                {
                    "path_lookup_key": self.path_lookup_key,
                    "expected_identity_sha256": self.expected_identity_sha256,
                }
            )
        if self.kind == "perfect_db":
            record["cache_sectors"] = self.cache_sectors
        elif self.kind == "human_db":
            record.update(
                {
                    "candidate_limit": self.human_candidate_limit,
                    "min_total": self.human_min_total,
                }
            )
        return record


@dataclass(frozen=True)
class PrefixStep:
    logical_ply: int
    side: str
    input_fen: str
    input_history_sha256: str
    candidate_pool_identity: str
    candidate_pool_size: int
    selection_trace: tuple[Mapping[str, Any], ...]
    selected_logical_move_id: str
    selected_stable_index: int
    selected_source_group_id: str | None
    selected_source_rank: int | None
    action_tokens: tuple[str, ...]
    output_fen: str
    output_history_sha256: str
    output_action_token_count: int
    output_logical_ply_count: int
    output_logical_plies_by_side: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_ply": self.logical_ply,
            "side": self.side,
            "input": {
                "fen": self.input_fen,
                "history_sha256": self.input_history_sha256,
            },
            "candidate_pool": {
                "identity": self.candidate_pool_identity,
                "size": self.candidate_pool_size,
            },
            "selection_trace": [dict(item) for item in self.selection_trace],
            "selected_candidate": {
                "logical_move_id": self.selected_logical_move_id,
                "stable_index": self.selected_stable_index,
                "source_group_id": self.selected_source_group_id,
                "source_rank": self.selected_source_rank,
                "action_tokens": list(self.action_tokens),
            },
            "output": {
                "fen": self.output_fen,
                "history_sha256": self.output_history_sha256,
                "action_token_count": self.output_action_token_count,
                "logical_ply_count": self.output_logical_ply_count,
                "logical_plies_by_side": list(
                    self.output_logical_plies_by_side
                ),
            },
        }


@dataclass(frozen=True)
class PairedPrefix:
    experiment_id: str
    pair_id: str
    seed: int
    source_spec: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    sanmill: Mapping[str, Any]
    action_tokens: tuple[str, ...]
    steps: tuple[PrefixStep, ...]
    final_fen: str
    final_history_sha256: str
    prefix_identity: str = ""

    def __post_init__(self) -> None:
        if len(self.steps) != PREFIX_LOGICAL_PLIES:
            raise SanmillPrefixError("paired prefix does not have eight steps")
        if [step.logical_ply for step in self.steps] != list(
            range(PREFIX_LOGICAL_PLIES)
        ):
            raise SanmillPrefixError("paired prefix step indices are not contiguous")
        if [step.side for step in self.steps] != [
            "white" if index % 2 == 0 else "black"
            for index in range(PREFIX_LOGICAL_PLIES)
        ]:
            raise SanmillPrefixError("paired prefix sides do not alternate")
        flattened = tuple(
            token for step in self.steps for token in step.action_tokens
        )
        if flattened != self.action_tokens:
            raise SanmillPrefixError("paired prefix action history is inconsistent")
        for index, step in enumerate(self.steps):
            if (
                step.output_logical_ply_count != index + 1
                or step.output_logical_plies_by_side
                != _expected_side_counts(index + 1)
            ):
                raise SanmillPrefixError("paired prefix step counts are inconsistent")
            if index > 0 and (
                step.input_fen != self.steps[index - 1].output_fen
                or step.input_history_sha256
                != self.steps[index - 1].output_history_sha256
            ):
                raise SanmillPrefixError("paired prefix step chain is inconsistent")
        final_step = self.steps[-1]
        if (
            self.final_fen != final_step.output_fen
            or self.final_history_sha256 != final_step.output_history_sha256
        ):
            raise SanmillPrefixError("paired prefix final state is inconsistent")
        if self.source_spec.get("kind") != self.source_identity.get("kind"):
            raise SanmillPrefixError("paired prefix source kind is inconsistent")
        expected = canonical_sha256(self._identity_body())
        if self.prefix_identity and self.prefix_identity != expected:
            raise SanmillPrefixError("paired prefix identity mismatch")
        object.__setattr__(self, "prefix_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": PAIRED_PREFIX_SCHEMA,
            "selection_schema": PREFIX_SELECTION_SCHEMA,
            "experiment_id": self.experiment_id,
            "pair_id": self.pair_id,
            "seed": self.seed,
            "source_spec": dict(self.source_spec),
            "source_identity": dict(self.source_identity),
            "sanmill": dict(self.sanmill),
            "logical_ply_count": PREFIX_LOGICAL_PLIES,
            "logical_plies_by_side": [4, 4],
            "action_tokens": list(self.action_tokens),
            "steps": [step.to_dict() for step in self.steps],
            "final": {
                "fen": self.final_fen,
                "history_sha256": self.final_history_sha256,
            },
            "shared_by_pair_games": [0, 1],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_body(),
            "prefix_identity": self.prefix_identity,
        }


def _selection_payload(
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    source_kind: str,
    logical_ply: int,
    purpose: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": PREFIX_SELECTION_SCHEMA,
        "experiment_id": experiment_id,
        "pair_id": pair_id,
        "seed": seed,
        "source_kind": source_kind,
        "logical_ply": logical_ply,
        "purpose": purpose,
        "attempt": attempt,
    }


def _weighted_index(
    weights: Sequence[int],
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    source_kind: str,
    logical_ply: int,
    purpose: str,
) -> tuple[int, dict[str, Any]]:
    if not weights or any(
        not isinstance(weight, int)
        or isinstance(weight, bool)
        or weight < 0
        for weight in weights
    ):
        raise SanmillPrefixError("selection weights must be non-negative integers")
    total = sum(weights)
    if total <= 0:
        raise SanmillPrefixError("selection weights have no positive mass")
    ceiling = 1 << 256
    unbiased_limit = ceiling - (ceiling % total)
    attempt = 0
    while True:
        payload = _selection_payload(
            experiment_id=experiment_id,
            pair_id=pair_id,
            seed=seed,
            source_kind=source_kind,
            logical_ply=logical_ply,
            purpose=purpose,
            attempt=attempt,
        )
        digest = hashlib.sha256(
            (
                canonical_sha256(payload)
                + ":nmm.sanmill-prefix-weighted-draw.v1"
            ).encode("ascii")
        ).hexdigest()
        draw = int(digest, 16)
        if draw < unbiased_limit:
            target = draw % total
            cumulative = 0
            for index, weight in enumerate(weights):
                cumulative += weight
                if target < cumulative:
                    return index, {
                        "purpose": purpose,
                        "attempt": attempt,
                        "draw_sha256": digest,
                        "weight_total": total,
                        "selected_index": index,
                    }
            raise AssertionError("weighted selection target was not consumed")
        attempt += 1


def assign_source_kind(
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    weights: Mapping[str, int],
) -> str:
    """Assign a source from explicit integer weights; there is no default mix."""
    _validate_identity_inputs(experiment_id, pair_id, seed)
    if not isinstance(weights, Mapping) or not weights:
        raise SanmillPrefixError("source assignment needs explicit weights")
    unknown = sorted(set(weights) - _SOURCE_KINDS)
    if unknown:
        raise SanmillPrefixError(
            "source assignment has unknown kinds: " + ", ".join(unknown)
        )
    ordered = sorted(weights)
    values = [weights[kind] for kind in ordered]
    index, _ = _weighted_index(
        values,
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_kind="source_assignment",
        logical_ply=-1,
        purpose="pair_source",
    )
    return ordered[index]


def _validate_identity_inputs(
    experiment_id: str,
    pair_id: str,
    seed: int,
) -> None:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise SanmillPrefixError("experiment_id must be non-empty")
    if not isinstance(pair_id, str) or not pair_id:
        raise SanmillPrefixError("pair_id must be non-empty")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SanmillPrefixError("prefix seed must be a non-negative integer")


def _candidate_record(candidate: DataQueryCandidate) -> dict[str, Any]:
    record: dict[str, Any] = {
        "logical_move_id": candidate.logical_move_id,
        "stable_index": candidate.stable_index,
        "source_group_id": candidate.source_group_id,
        "source_rank": candidate.source_rank,
        "full_turn_actions": list(candidate.full_turn_actions),
    }
    if candidate.perfect is not None:
        record["perfect"] = {
            "category": candidate.perfect.category,
            "wdl": candidate.perfect.wdl,
            "steps": candidate.perfect.steps,
            "mode": candidate.perfect.mode,
        }
    if candidate.human is not None:
        record["human"] = {
            "total": candidate.human.total,
            "frequency_numerator": candidate.human.frequency_numerator,
            "frequency_denominator": candidate.human.frequency_denominator,
        }
    return record


def _uniform_candidate(
    candidates: Sequence[DataQueryCandidate],
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    source_kind: str,
    logical_ply: int,
) -> tuple[DataQueryCandidate, tuple[Mapping[str, Any], ...]]:
    index, trace = _weighted_index(
        [1] * len(candidates),
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_kind=source_kind,
        logical_ply=logical_ply,
        purpose="uniform_candidate",
    )
    return candidates[index], (trace,)


def _source_declared_book_candidate(
    candidates: Sequence[DataQueryCandidate],
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    logical_ply: int,
) -> tuple[DataQueryCandidate, tuple[Mapping[str, Any], ...]]:
    groups: dict[str, list[DataQueryCandidate]] = {}
    ranks: dict[str, int] = {}
    for candidate in candidates:
        if candidate.source_group_id is None or candidate.source_rank is None:
            raise SanmillPrefixError("book candidate lacks group or rank")
        groups.setdefault(candidate.source_group_id, []).append(candidate)
        previous = ranks.setdefault(
            candidate.source_group_id,
            candidate.source_rank,
        )
        if previous != candidate.source_rank:
            raise SanmillPrefixError("book source group has inconsistent ranks")
    ordered_groups = sorted(groups, key=lambda group: (ranks[group], group))
    if len(set(ranks.values())) != len(ranks):
        raise SanmillPrefixError("book source ranks are duplicated across groups")
    max_rank = max(ranks.values())
    group_weights = [
        (3 ** (ranks[group] - 1)) * (5 ** (max_rank - ranks[group]))
        for group in ordered_groups
    ]
    group_index, group_trace = _weighted_index(
        group_weights,
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_kind="book",
        logical_ply=logical_ply,
        purpose="source_declared_group",
    )
    selected_group = sorted(
        groups[ordered_groups[group_index]],
        key=lambda candidate: candidate.stable_index,
    )
    candidate_index, candidate_trace = _weighted_index(
        [1] * len(selected_group),
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_kind="book",
        logical_ply=logical_ply,
        purpose="source_group_continuation",
    )
    return selected_group[candidate_index], (group_trace, candidate_trace)


def _human_frequency_candidate(
    candidates: Sequence[DataQueryCandidate],
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    logical_ply: int,
) -> tuple[DataQueryCandidate, tuple[Mapping[str, Any], ...]]:
    if any(candidate.human is None for candidate in candidates):
        raise SanmillPrefixError("HumanDB candidate lacks frequency data")
    weights = [
        candidate.human.frequency_numerator  # type: ignore[union-attr]
        for candidate in candidates
    ]
    index, trace = _weighted_index(
        weights,
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_kind="human_db",
        logical_ply=logical_ply,
        purpose="human_frequency",
    )
    return candidates[index], (trace,)


def _select_candidate(
    candidates: Sequence[DataQueryCandidate],
    *,
    source_spec: PrefixSourceSpec,
    experiment_id: str,
    pair_id: str,
    seed: int,
    logical_ply: int,
) -> tuple[DataQueryCandidate, tuple[Mapping[str, Any], ...]]:
    if not candidates:
        raise SanmillPrefixError("available source returned no candidates")
    if source_spec.candidate_policy == "uniform_candidate":
        return _uniform_candidate(
            candidates,
            experiment_id=experiment_id,
            pair_id=pair_id,
            seed=seed,
            source_kind=source_spec.kind,
            logical_ply=logical_ply,
        )
    if source_spec.candidate_policy == "source_declared":
        return _source_declared_book_candidate(
            candidates,
            experiment_id=experiment_id,
            pair_id=pair_id,
            seed=seed,
            logical_ply=logical_ply,
        )
    return _human_frequency_candidate(
        candidates,
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        logical_ply=logical_ply,
    )


def _expected_side_counts(logical_ply_count: int) -> tuple[int, int]:
    return ((logical_ply_count + 1) // 2, logical_ply_count // 2)


def _validate_boundary_state(
    state: DataQueryState,
    *,
    logical_ply_count: int,
    action_token_count: int,
    expected_previous: DataQueryState | None,
) -> None:
    if state.outcome.terminal:
        raise SanmillPrefixError("prefix reached a terminal state before completion")
    if state.pending_removal or sum(state.pending_removals) != 0:
        raise SanmillPrefixError("prefix boundary has a pending removal")
    if state.logical_ply_count != logical_ply_count:
        raise SanmillPrefixError("prefix logical-ply count drifted")
    if state.logical_plies_by_side != _expected_side_counts(logical_ply_count):
        raise SanmillPrefixError("prefix per-side logical-ply counts drifted")
    if state.action_token_count != action_token_count:
        raise SanmillPrefixError("prefix action-token count drifted")
    expected_side = "white" if logical_ply_count % 2 == 0 else "black"
    if state.side_to_move != expected_side:
        raise SanmillPrefixError("prefix side to move drifted")
    if expected_previous is not None and state != expected_previous:
        raise SanmillPrefixError(
            "source query state differs from the preceding history summary"
        )


def _query_source(
    session: SanmillDataQuerySession,
    source_spec: PrefixSourceSpec,
    *,
    actions: Sequence[str],
    request_id: str,
    expected_current_fen: str | None,
) -> DataQueryResponse:
    if source_spec.kind == "book":
        return session.query_book(
            actions,
            request_id=request_id,
            expected_current_fen=expected_current_fen,
        )
    if source_spec.kind == "perfect_db":
        assert source_spec.database_path is not None
        return session.query_perfect_db(
            actions,
            database_path=source_spec.database_path,
            request_id=request_id,
            expected_current_fen=expected_current_fen,
            cache_sectors=source_spec.cache_sectors,
        )
    assert source_spec.database_path is not None
    return session.query_human_db(
        actions,
        database_path=source_spec.database_path,
        request_id=request_id,
        expected_current_fen=expected_current_fen,
        candidate_limit=source_spec.human_candidate_limit,
        min_total=source_spec.human_min_total,
    )


def _request_id(
    experiment_id: str,
    pair_id: str,
    logical_ply: int,
    purpose: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema": PAIRED_PREFIX_SCHEMA,
            "experiment_id": experiment_id,
            "pair_id": pair_id,
        }
    )
    return f"prefix-{digest[:20]}-{logical_ply:02d}-{purpose}"


def generate_paired_prefix(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    experiment_id: str,
    pair_id: str,
    seed: int,
    source_spec: PrefixSourceSpec,
) -> PairedPrefix:
    """Generate one immutable eight-ply prefix shared by both pair games."""
    _validate_identity_inputs(experiment_id, pair_id, seed)
    actions: tuple[str, ...] = ()
    previous_state: DataQueryState | None = None
    steps: list[PrefixStep] = []
    bound_source_identity: dict[str, Any] | None = None

    for logical_ply in range(PREFIX_LOGICAL_PLIES):
        source_response = _query_source(
            session,
            source_spec,
            actions=actions,
            request_id=_request_id(
                experiment_id,
                pair_id,
                logical_ply,
                "source",
            ),
            expected_current_fen=(
                previous_state.current_fen if previous_state is not None else None
            ),
        )
        if source_response.status != "available":
            raise SanmillPrefixError(
                f"{source_spec.kind} is {source_response.status} at logical ply "
                f"{logical_ply}; no fallback is permitted"
            )
        if source_response.state is None or source_response.candidates is None:
            raise SanmillPrefixError("available source response is incomplete")
        _validate_boundary_state(
            source_response.state,
            logical_ply_count=logical_ply,
            action_token_count=len(actions),
            expected_previous=previous_state,
        )
        source_identity = portable_source_identity(
            source_response,
            path_lookup_key=source_spec.path_lookup_key,
        )
        if (
            source_spec.expected_identity_sha256 is not None
            and source_identity["identity_sha256"]
            != source_spec.expected_identity_sha256
        ):
            raise SanmillPrefixError("prefix source identity differs from its pin")
        if bound_source_identity is None:
            bound_source_identity = source_identity
        elif source_identity != bound_source_identity:
            raise SanmillPrefixError("prefix source identity changed during sampling")

        pool = tuple(source_response.candidates)
        pool_identity = canonical_sha256(
            [_candidate_record(candidate) for candidate in pool]
        )
        selected, trace = _select_candidate(
            pool,
            source_spec=source_spec,
            experiment_id=experiment_id,
            pair_id=pair_id,
            seed=seed,
            logical_ply=logical_ply,
        )
        if selected.remaining_actions != selected.full_turn_actions:
            raise SanmillPrefixError(
                "stable prefix boundary returned a partial logical turn"
            )
        next_actions = actions + selected.full_turn_actions
        summary = session.history_summary(
            next_actions,
            request_id=_request_id(
                experiment_id,
                pair_id,
                logical_ply,
                "summary",
            ),
            count_mode="logical",
        )
        if summary.status != "available" or summary.state is None:
            raise SanmillPrefixError(
                "selected prefix action became terminal or unavailable"
            )
        _validate_boundary_state(
            summary.state,
            logical_ply_count=logical_ply + 1,
            action_token_count=len(next_actions),
            expected_previous=None,
        )
        if (
            summary.state.history_sha256
            == source_response.state.history_sha256
        ):
            raise SanmillPrefixError("selected action did not change history identity")
        steps.append(
            PrefixStep(
                logical_ply=logical_ply,
                side=source_response.state.side_to_move or "",
                input_fen=source_response.state.current_fen,
                input_history_sha256=source_response.state.history_sha256,
                candidate_pool_identity=pool_identity,
                candidate_pool_size=len(pool),
                selection_trace=trace,
                selected_logical_move_id=selected.logical_move_id,
                selected_stable_index=selected.stable_index,
                selected_source_group_id=selected.source_group_id,
                selected_source_rank=selected.source_rank,
                action_tokens=selected.full_turn_actions,
                output_fen=summary.state.current_fen,
                output_history_sha256=summary.state.history_sha256,
                output_action_token_count=summary.state.action_token_count,
                output_logical_ply_count=summary.state.logical_ply_count,
                output_logical_plies_by_side=summary.state.logical_plies_by_side,
            )
        )
        actions = next_actions
        previous_state = summary.state

    if previous_state is None or bound_source_identity is None:
        raise AssertionError("eight-ply prefix loop produced no state")
    if (
        previous_state.logical_ply_count != PREFIX_LOGICAL_PLIES
        or previous_state.logical_plies_by_side != (4, 4)
    ):
        raise SanmillPrefixError("final paired-prefix logical counts are invalid")
    return PairedPrefix(
        experiment_id=experiment_id,
        pair_id=pair_id,
        seed=seed,
        source_spec=source_spec.portable_record(),
        source_identity=bound_source_identity,
        sanmill=installation.portable_record(),
        action_tokens=actions,
        steps=tuple(steps),
        final_fen=previous_state.current_fen,
        final_history_sha256=previous_state.history_sha256,
    )
