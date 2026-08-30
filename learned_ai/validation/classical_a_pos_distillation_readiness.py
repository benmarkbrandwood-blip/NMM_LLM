"""Authorization-free readiness contract for classical ``A_pos`` distillation.

This module can describe and inspect the frozen offline-supervised route, but
it cannot launch it.  In particular, resource recommendations are deliberately
separate from the all-zero authorization envelope, and a successful structural
inspection still returns ``fatal_stop`` until a future controller supplies the
missing issuer, smoke, runtime, Git, path, and product-authority evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from learned_ai.models.scaffolded_encoder import MOVE_FEAT_DIM, VALUE_INPUT_DIM
from learned_ai.training.classical_a_pos_corpus import (
    CORPUS_SCHEMA,
    FROZEN_CORPUS_LAYOUT,
    RESOURCE_IDENTITY_KEYS,
    STATE_SPLIT_ARTIFACT_SCHEMA,
    TRAINING_PROFILE,
    FrozenCorpus,
)
from learned_ai.training.classical_a_pos_supervised import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    PRODUCTION_BATCHES_PER_EPOCH,
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_EPOCHS,
    PRODUCTION_POLICY_HIDDEN,
    PRODUCTION_SEEDS,
    PRODUCTION_TRAIN_EXAMPLES,
    PRODUCTION_UPDATES_PER_SEED,
    TRAINER_ID,
    production_loop_config,
)
from learned_ai.training.run_contract import canonical_sha256


PROFILE_SCHEMA = "nmm.classical-a-pos-distillation-profile.v1"
RUN_MANIFEST_SCHEMA = "nmm.classical-a-pos-supervised-run-proposal.v1"
REQUESTED_RESOURCE_SCHEMA = "nmm.classical-a-pos-requested-resource.v1"
ZERO_AUTHORIZATION_SCHEMA = "nmm.classical-a-pos-zero-authorization.v1"
READINESS_RECEIPT_SCHEMA = "nmm.classical-a-pos-readiness-receipt.v1"

_TRAINING_SEMANTICS = "offline-supervised-hard-a-pos-masked-one-hot-cross-entropy"
_HEX = frozenset("0123456789abcdef")
_RUN_MANIFEST_BODY_KEYS = {
    "schema_version",
    "proposal_status",
    "launch_status",
    "authorization_status",
    "executable",
    "profile_identity",
    "training_profile",
    "training_semantics",
    "schemas",
    "corpus",
    "corpus_identity",
    "split_identity",
    "seeds",
    "model",
    "optimizer",
    "optimizer_updates_per_seed",
    "rl",
    "forbidden",
    "device",
    "requested_resource_package",
    "authorization_envelope",
}


class DistillationReadinessError(RuntimeError):
    """A proposed distillation contract failed a fail-closed readiness check."""


def _freeze(value: Any, *, field: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DistillationReadinessError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DistillationReadinessError(f"{field} contains a non-string key")
            copied[key] = _freeze(item, field=f"{field}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        )
    raise DistillationReadinessError(
        f"{field} contains unsupported {type(value).__name__} data"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_exact_contract(value: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            raise DistillationReadinessError(f"{field} must be an object")
        actual_keys = set(value)
        expected_keys = set(expected)
        unknown = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        if unknown:
            raise DistillationReadinessError(
                f"{field} has unknown keys: {', '.join(unknown)}"
            )
        if missing:
            raise DistillationReadinessError(
                f"{field} has missing keys: {', '.join(missing)}"
            )
        for key in expected:
            _require_exact_contract(
                value[key],
                expected[key],
                field=f"{field}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected,
        (str, bytes, bytearray),
    ):
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise DistillationReadinessError(f"{field} must be an array")
        if len(value) != len(expected):
            raise DistillationReadinessError(f"{field} length differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_contract(
                actual_item,
                expected_item,
                field=f"{field}[{index}]",
            )
        return
    if type(value) is not type(expected) or value != expected:
        if field == "profile" or field.startswith("profile."):
            raise DistillationReadinessError(
                f"profile differs at {field} from the frozen contract"
            )
        raise DistillationReadinessError(f"{field} differs from the frozen contract")


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise DistillationReadinessError(
            f"{field} must be a 64-character lowercase SHA-256"
        )
    return value


def _sum_split(split: str) -> int:
    return sum(
        count
        for colours in FROZEN_CORPUS_LAYOUT.to_dict()[split].values()
        for count in colours.values()
    )


_CORPUS_TOTAL = FROZEN_CORPUS_LAYOUT.total
_TRAIN_TOTAL = _sum_split("train")
_DEV_TOTAL = _sum_split("dev")
if (
    _CORPUS_TOTAL != 16_384
    or _TRAIN_TOTAL != PRODUCTION_TRAIN_EXAMPLES
    or _TRAIN_TOTAL != 14_336
    or _DEV_TOTAL != 2_048
    or PRODUCTION_BATCH_SIZE * PRODUCTION_BATCHES_PER_EPOCH != PRODUCTION_TRAIN_EXAMPLES
    or PRODUCTION_EPOCHS * PRODUCTION_BATCHES_PER_EPOCH != PRODUCTION_UPDATES_PER_SEED
):
    raise RuntimeError("classical A_pos corpus and supervised constants disagree")

_LOOP_CONTRACT = production_loop_config(PRODUCTION_SEEDS[0]).to_dict()
for _seed in PRODUCTION_SEEDS:
    _seed_contract = production_loop_config(_seed).to_dict()
    if {
        key: value for key, value in _seed_contract.items() if key != "permutation_seed"
    } != {
        key: value for key, value in _LOOP_CONTRACT.items() if key != "permutation_seed"
    }:
        raise RuntimeError("classical A_pos per-seed loop contracts disagree")

_PROFILE_TEMPLATE: dict[str, Any] = {
    "schema_version": PROFILE_SCHEMA,
    "profile": TRAINING_PROFILE,
    "training_semantics": _TRAINING_SEMANTICS,
    "schemas": {
        "corpus": CORPUS_SCHEMA,
        "feature": FEATURE_SCHEMA,
        "label": LABEL_SCHEMA,
        "trainer": TRAINER_ID,
    },
    "corpus": {
        "total_examples": _CORPUS_TOTAL,
        "train_examples": _TRAIN_TOTAL,
        "dev_examples": _DEV_TOTAL,
        "whole_game_split": True,
    },
    "seeds": list(PRODUCTION_SEEDS),
    "model": {
        "move_feature_dim": MOVE_FEAT_DIM,
        "policy_hidden": list(PRODUCTION_POLICY_HIDDEN),
        "value_hidden": [],
        "dropout": 0.0,
        "value_input_dim": VALUE_INPUT_DIM,
        "value_branch_in_optimizer": False,
    },
    "optimizer": {
        "kind": _LOOP_CONTRACT["optimizer"]["kind"],
        "parameter_scope": _LOOP_CONTRACT["optimizer"]["parameter_scope"],
        "lr": _LOOP_CONTRACT["optimizer"]["lr"],
        "betas": _LOOP_CONTRACT["optimizer"]["betas"],
        "eps": _LOOP_CONTRACT["optimizer"]["eps"],
        "weight_decay": _LOOP_CONTRACT["optimizer"]["weight_decay"],
        "schedule": _LOOP_CONTRACT["optimizer"]["schedule"],
        "batch_size": PRODUCTION_BATCH_SIZE,
        "epochs": PRODUCTION_EPOCHS,
        "batches_per_epoch": PRODUCTION_BATCHES_PER_EPOCH,
        "grad_clip_norm": _LOOP_CONTRACT["grad_clip_norm"],
    },
    "optimizer_updates_per_seed": PRODUCTION_UPDATES_PER_SEED,
    "rl": _LOOP_CONTRACT["rl"],
    "forbidden": {
        "online_teacher_queries": False,
        "human_db": False,
        "specialist_db": False,
        "advisors": False,
    },
    "device": {
        "kind": "cpu",
        "cuda": False,
        "throughput_claim": None,
    },
}

FROZEN_TRAINING_PROFILE: Mapping[str, Any] = _freeze(
    _PROFILE_TEMPLATE,
    field="profile",
)


def validate_training_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an independent immutable copy of the exact frozen profile."""
    if not isinstance(profile, Mapping):
        raise DistillationReadinessError("profile must be an object")
    _require_exact_contract(profile, _PROFILE_TEMPLATE, field="profile")
    return _freeze(_thaw(profile), field="profile")


_REQUESTED_RESOURCE_PACKAGE: dict[str, Any] = {
    "schema_version": REQUESTED_RESOURCE_SCHEMA,
    "recommendation_only": True,
    "counts_as_authorization": False,
    "state_generation": {
        "games": 1_024,
        "active_seconds": 3_600,
    },
    "teacher": {
        "active_seconds": 14_400,
        "nodes": 4_000_000_000,
        "positive_search_labels": 1_280,
    },
    "smoke": {"active_seconds": 3_600},
    "seed": {
        "per_seed_active_seconds": 21_600,
        "aggregate_active_seconds": 64_800,
    },
    "sequence_active_seconds": 86_400,
    "evaluation_games": 0,
}

_ZERO_AUTHORIZATION: dict[str, Any] = {
    "schema_version": ZERO_AUTHORIZATION_SCHEMA,
    "status": "unauthorized",
    "authorization_identity": None,
    "authorized_by": None,
    "issued_at_utc": None,
    "expires_at_utc": None,
    "standing_delegation_identity": None,
    "consumption_limit": 0,
    "allowed_operations": [],
    "allow_exact_resume": False,
    "authorized_resources": {
        "games": 0,
        "active_seconds": 0,
        "teacher_nodes": 0,
        "positive_search_labels": 0,
        "evaluation_games": 0,
    },
}


def validate_requested_resource_package(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationReadinessError("requested resource package must be an object")
    _require_exact_contract(
        value,
        _REQUESTED_RESOURCE_PACKAGE,
        field="requested resource package",
    )
    return _freeze(_thaw(value), field="requested resource package")


def validate_authorization_envelope(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationReadinessError("authorization envelope must be an object")
    _require_exact_contract(
        value,
        _ZERO_AUTHORIZATION,
        field="authorization envelope",
    )
    return _freeze(_thaw(value), field="authorization envelope")


def _manifest_body(
    profile: Mapping[str, Any],
    *,
    corpus_identity: str,
    split_identity: str,
) -> dict[str, Any]:
    copied_profile = _thaw(profile)
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "proposal_status": "proposed",
        "launch_status": "unlaunched",
        "authorization_status": "unauthorized",
        "executable": False,
        "profile_identity": canonical_sha256(copied_profile),
        "training_profile": copied_profile["profile"],
        "training_semantics": copied_profile["training_semantics"],
        "schemas": copied_profile["schemas"],
        "corpus": copied_profile["corpus"],
        "corpus_identity": corpus_identity,
        "split_identity": split_identity,
        "seeds": copied_profile["seeds"],
        "model": copied_profile["model"],
        "optimizer": copied_profile["optimizer"],
        "optimizer_updates_per_seed": copied_profile["optimizer_updates_per_seed"],
        "rl": copied_profile["rl"],
        "forbidden": copied_profile["forbidden"],
        "device": copied_profile["device"],
        "requested_resource_package": _thaw(_REQUESTED_RESOURCE_PACKAGE),
        "authorization_envelope": _thaw(_ZERO_AUTHORIZATION),
    }


def build_supervised_run_manifest(
    profile: Mapping[str, Any],
    corpus_identity: str,
    split_identity: str,
) -> Mapping[str, Any]:
    """Build a canonical, immutable, explicitly unauthorized proposal."""
    checked_profile = validate_training_profile(profile)
    checked_corpus = _require_sha256(corpus_identity, field="corpus_identity")
    checked_split = _require_sha256(split_identity, field="split_identity")
    body = _manifest_body(
        checked_profile,
        corpus_identity=checked_corpus,
        split_identity=checked_split,
    )
    manifest = {**body, "manifest_identity": canonical_sha256(body)}
    return validate_supervised_run_manifest(manifest)


def validate_supervised_run_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise DistillationReadinessError("run manifest must be an object")
    expected_keys = _RUN_MANIFEST_BODY_KEYS | {"manifest_identity"}
    actual_keys = set(manifest)
    unknown = sorted(actual_keys - expected_keys)
    missing = sorted(expected_keys - actual_keys)
    if unknown:
        raise DistillationReadinessError(
            f"run manifest has unknown keys: {', '.join(unknown)}"
        )
    if missing:
        raise DistillationReadinessError(
            f"run manifest has missing keys: {', '.join(missing)}"
        )
    corpus_identity = _require_sha256(
        manifest["corpus_identity"],
        field="manifest corpus_identity",
    )
    split_identity = _require_sha256(
        manifest["split_identity"],
        field="manifest split_identity",
    )
    validate_requested_resource_package(manifest["requested_resource_package"])
    validate_authorization_envelope(manifest["authorization_envelope"])
    expected_body = _manifest_body(
        FROZEN_TRAINING_PROFILE,
        corpus_identity=corpus_identity,
        split_identity=split_identity,
    )
    observed_body = {
        key: _thaw(value)
        for key, value in manifest.items()
        if key != "manifest_identity"
    }
    _require_exact_contract(observed_body, expected_body, field="run manifest")
    identity = _require_sha256(
        manifest["manifest_identity"],
        field="manifest_identity",
    )
    if identity != canonical_sha256(observed_body):
        raise DistillationReadinessError("run manifest canonical identity differs")
    return _freeze(_thaw(manifest), field="run manifest")


def _expected_split_contract() -> dict[str, Any]:
    return {
        "strategy": "whole-game",
        "algorithm": {
            "game_index_origin": 0,
            "block_size_games": 16,
            "candidate_colour_by_parity": {"even": "W", "odd": "B"},
            "dev_remainders": [0, 1],
            "train_remainders": list(range(2, 16)),
            "stop_only_after_complete_block": True,
            "maximum_games": 1_024,
        },
        "counts": FROZEN_CORPUS_LAYOUT.to_dict(),
    }


_CORPUS_MANIFEST_KEYS = {
    "schema_version",
    "corpus_id",
    "profile",
    "status",
    "source",
    "generator",
    "referee",
    "teacher",
    "selection",
    "a_pos_verifier",
    "resources_before",
    "resources_after",
    "state_split_artifact",
    "examples",
    "singleton_ledger",
    "collection_games",
    "split",
    "split_identity",
    "heldout",
    "corpus_identity",
}

_TEACHER_KEYS = {
    "route",
    "identity_claim",
    "product_tt_trajectory_equivalent",
    "fresh_instance_per_label",
    "difficulty",
    "depth",
    "threads",
    "node_cap_per_label",
    "label",
    "online_queries_during_training",
    "active_seconds",
    "active_seconds_limit",
    "main_search_nodes",
    "restricted_rerank_nodes",
    "aggregate_nodes",
    "aggregate_nodes_limit",
    "positive_search_labels",
    "positive_search_labels_limit",
    "completed_labels",
}


def _require_exact_keys(value: Any, expected: set[str], *, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DistillationReadinessError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        raise DistillationReadinessError(f"{field} keys differ ({'; '.join(details)})")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DistillationReadinessError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_loaded_corpus(corpus: Any) -> FrozenCorpus:
    if type(corpus) is not FrozenCorpus:
        raise DistillationReadinessError(
            "corpus loader must return the strict FrozenCorpus result"
        )
    manifest = _require_exact_keys(
        corpus.manifest,
        _CORPUS_MANIFEST_KEYS,
        field="corpus manifest",
    )
    observed_identity = _require_sha256(
        corpus.corpus_identity,
        field="corpus identity",
    )
    manifest_identity = _require_sha256(
        manifest["corpus_identity"],
        field="corpus manifest identity",
    )
    body = {key: value for key, value in manifest.items() if key != "corpus_identity"}
    if observed_identity != manifest_identity or manifest_identity != canonical_sha256(
        body
    ):
        raise DistillationReadinessError("corpus identity differs")
    for key, expected_value in {
        "schema_version": CORPUS_SCHEMA,
        "profile": TRAINING_PROFILE,
        "status": "complete",
    }.items():
        _require_exact_contract(
            manifest[key],
            expected_value,
            field=f"corpus manifest.{key}",
        )
    if len(corpus.examples) != _CORPUS_TOTAL:
        raise DistillationReadinessError(
            "corpus must contain exactly 16,384 loaded examples"
        )

    collection_games = manifest["collection_games"]
    if (
        isinstance(collection_games, bool)
        or not isinstance(collection_games, int)
        or not 0 < collection_games <= 1_024
        or collection_games % 16 != 0
    ):
        raise DistillationReadinessError("corpus collection game ledger differs")

    resources_before = _require_exact_keys(
        manifest["resources_before"],
        set(RESOURCE_IDENTITY_KEYS),
        field="corpus resources_before",
    )
    resources_after = _require_exact_keys(
        manifest["resources_after"],
        set(RESOURCE_IDENTITY_KEYS),
        field="corpus resources_after",
    )
    checked_before = {
        key: _require_sha256(
            resources_before[key],
            field=f"corpus resources_before.{key}",
        )
        for key in sorted(RESOURCE_IDENTITY_KEYS)
    }
    checked_after = {
        key: _require_sha256(
            resources_after[key],
            field=f"corpus resources_after.{key}",
        )
        for key in sorted(RESOURCE_IDENTITY_KEYS)
    }
    if checked_before != checked_after:
        raise DistillationReadinessError("corpus resource ledger differs")

    split = manifest["split"]
    expected_split = _expected_split_contract()
    _require_exact_contract(split, expected_split, field="corpus split")
    observed_split_identity = canonical_sha256(split)
    expected_split_identity = canonical_sha256(expected_split)
    object_split_identity = _require_sha256(
        corpus.split_identity,
        field="corpus split identity",
    )
    manifest_split_identity = _require_sha256(
        manifest["split_identity"],
        field="corpus manifest split identity",
    )
    if (
        observed_split_identity != manifest_split_identity
        or object_split_identity != manifest_split_identity
        or manifest_split_identity != expected_split_identity
    ):
        raise DistillationReadinessError("corpus split identity differs")

    examples = _require_exact_keys(
        manifest["examples"],
        {"path", "size", "sha256", "count"},
        field="corpus examples",
    )
    _require_exact_contract(
        examples["count"],
        _CORPUS_TOTAL,
        field="corpus examples.count",
    )
    _require_sha256(examples["sha256"], field="corpus examples SHA-256")

    heldout = manifest["heldout"]
    _require_exact_contract(
        heldout,
        {"consumed": False, "sources": []},
        field="corpus heldout",
    )

    teacher = _require_exact_keys(
        manifest["teacher"],
        _TEACHER_KEYS,
        field="corpus teacher",
    )
    fixed_teacher = {
        "route": "current-head-exact-D9-post-ProductPositionalSafetyGate",
        "identity_claim": "fresh-instance-current-D9-positional-teacher",
        "product_tt_trajectory_equivalent": False,
        "fresh_instance_per_label": True,
        "difficulty": 9,
        "depth": 14,
        "threads": 1,
        "node_cap_per_label": 13_887_000,
        "label": "one-hot-atomic-action-inside-A_pos",
        "online_queries_during_training": False,
        "active_seconds_limit": 14_400,
        "aggregate_nodes_limit": 4_000_000_000,
        "positive_search_labels_limit": 1_280,
        "completed_labels": _CORPUS_TOTAL,
    }
    for key, expected_value in fixed_teacher.items():
        _require_exact_contract(
            teacher[key],
            expected_value,
            field=f"corpus teacher.{key}",
        )
    active_seconds = teacher["active_seconds"]
    if (
        isinstance(active_seconds, bool)
        or not isinstance(active_seconds, (int, float))
        or not math.isfinite(float(active_seconds))
        or not 0.0 <= float(active_seconds) <= 14_400.0
    ):
        raise DistillationReadinessError("corpus teacher active time differs")
    main_nodes = _require_int(
        teacher["main_search_nodes"],
        field="corpus teacher main-search nodes",
    )
    rerank_nodes = _require_int(
        teacher["restricted_rerank_nodes"],
        field="corpus teacher restricted-rerank nodes",
    )
    aggregate_nodes = _require_int(
        teacher["aggregate_nodes"],
        field="corpus teacher aggregate nodes",
    )
    if aggregate_nodes != main_nodes + rerank_nodes or aggregate_nodes > 4_000_000_000:
        raise DistillationReadinessError("corpus teacher aggregate nodes differ")
    if (
        _require_int(
            teacher["positive_search_labels"],
            field="corpus teacher positive-search labels",
        )
        > 1_280
    ):
        raise DistillationReadinessError("corpus teacher positive-search labels differ")

    verifier = _require_exact_keys(
        manifest["a_pos_verifier"],
        {
            "boundary",
            "identity",
            "record_count",
            "verification_chain_head",
            "loader_requeries_malom",
        },
        field="corpus A_pos verifier",
    )
    for key, expected_value in {
        "boundary": "build-time-chain-plus-load-time-real-PositionalSafetyFilter",
        "record_count": _CORPUS_TOTAL,
        "loader_requeries_malom": True,
    }.items():
        _require_exact_contract(
            verifier[key],
            expected_value,
            field=f"corpus A_pos verifier.{key}",
        )
    _require_sha256(verifier["identity"], field="corpus A_pos verifier identity")
    _require_sha256(
        verifier["verification_chain_head"],
        field="corpus A_pos verification chain",
    )

    state_artifact = _require_exact_keys(
        manifest["state_split_artifact"],
        {
            "schema_version",
            "file",
            "identity",
            "teacher_fields_present",
            "controller_freeze_event_required_before_teacher",
        },
        field="corpus state/split artifact",
    )
    state_file = _require_exact_keys(
        state_artifact["file"],
        {"path", "size", "sha256", "count"},
        field="corpus state/split file",
    )
    for key, expected_value in {
        "schema_version": STATE_SPLIT_ARTIFACT_SCHEMA,
        "teacher_fields_present": False,
        "controller_freeze_event_required_before_teacher": True,
    }.items():
        _require_exact_contract(
            state_artifact[key],
            expected_value,
            field=f"corpus state/split artifact.{key}",
        )
    _require_exact_contract(
        state_file["count"],
        _CORPUS_TOTAL,
        field="corpus state/split file.count",
    )
    _require_sha256(state_artifact["identity"], field="state/split identity")
    _require_sha256(state_file["sha256"], field="state/split file SHA-256")

    selection = manifest["selection"]
    if not isinstance(selection, Mapping):
        raise DistillationReadinessError("corpus selection must be an object")
    _require_exact_contract(
        selection.get("teacher_results_visible_to_selection"),
        False,
        field="corpus selection.teacher_results_visible_to_selection",
    )
    return corpus


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(value: str | Path) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DistillationReadinessError("corpus manifest path is missing") from exc
    if not path.is_file():
        raise DistillationReadinessError("corpus manifest path is not a file")
    return path


def _resolve_output_path(value: str | Path, *, corpus_root: Path) -> tuple[Path, str]:
    try:
        path = Path(value).resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DistillationReadinessError("output path is invalid") from exc
    if (
        path == corpus_root
        or path.is_relative_to(corpus_root)
        or corpus_root.is_relative_to(path)
    ):
        raise DistillationReadinessError(
            "output path overlaps the frozen corpus bundle"
        )
    if not path.exists():
        return path, "absent"
    if not path.is_dir():
        raise DistillationReadinessError("output path must be a directory")
    try:
        first = next(path.iterdir(), None)
    except OSError as exc:
        raise DistillationReadinessError(
            "output directory cannot be inspected"
        ) from exc
    if first is not None:
        raise DistillationReadinessError("output directory is not empty")
    return path, "empty"


def run_distillation_preflight(
    profile: Mapping[str, Any],
    *,
    corpus_manifest_path: str | Path,
    output_dir: str | Path,
    corpus_loader: Callable[[Path], FrozenCorpus],
) -> Mapping[str, Any]:
    """Inspect the offline proposal without creating output or granting launch."""
    # This must remain the first operation: invalid semantics may not touch paths
    # or call a loader that could open large or forbidden assets.
    checked_profile = validate_training_profile(profile)

    manifest_path = _resolve_manifest_path(corpus_manifest_path)
    resolved_output, output_state = _resolve_output_path(
        output_dir,
        corpus_root=manifest_path.parent,
    )
    if not callable(corpus_loader):
        raise DistillationReadinessError("corpus_loader must be callable")
    manifest_file_sha256 = _sha256_file(manifest_path)
    try:
        corpus = corpus_loader(manifest_path)
    except DistillationReadinessError:
        raise
    except Exception as exc:
        raise DistillationReadinessError("strict corpus loading failed") from exc
    checked_corpus = _validate_loaded_corpus(corpus)
    if _sha256_file(manifest_path) != manifest_file_sha256:
        raise DistillationReadinessError(
            "corpus manifest changed during read-only preflight"
        )
    if output_state == "absent" and resolved_output.exists():
        raise DistillationReadinessError(
            "output path was created during read-only preflight"
        )
    if output_state == "empty":
        try:
            output_changed = next(resolved_output.iterdir(), None) is not None
        except OSError as exc:
            raise DistillationReadinessError(
                "output directory cannot be inspected after corpus loading"
            ) from exc
        if output_changed:
            raise DistillationReadinessError(
                "output directory changed during read-only preflight"
            )

    proposal = build_supervised_run_manifest(
        checked_profile,
        corpus_identity=checked_corpus.corpus_identity,
        split_identity=checked_corpus.split_identity,
    )
    unresolved = [
        "production_d9_teacher_issuer",
        "supervised_plan_issuer",
        "state_freeze_teacher_order_event_chain",
        "supervised_controller",
        "supervised_smoke_authority",
        "production_corpus_loader_binding",
        "device_throughput_measurement",
        "launch_path_binding",
        "managed_git_state",
        "product_owner_authorization",
    ]
    body = {
        "schema_version": READINESS_RECEIPT_SCHEMA,
        "verdict": "fatal_stop",
        "launch_status": "unlaunched",
        "authorization_status": "unauthorized",
        "executable": False,
        "profile_identity": proposal["profile_identity"],
        "proposed_manifest_identity": proposal["manifest_identity"],
        "corpus_identity": checked_corpus.corpus_identity,
        "split_identity": checked_corpus.split_identity,
        "corpus_manifest_path": str(manifest_path),
        "corpus_manifest_file_sha256": manifest_file_sha256,
        "output_dir": str(resolved_output),
        "output_state": output_state,
        "checks": {
            "profile_exact": True,
            "frozen_corpus_contract_rechecked": True,
            "production_corpus_loader_bound": False,
            "whole_game_split_exact": True,
            "heldout_unconsumed": True,
            "online_teacher_during_training": False,
            "output_isolated_and_unwritten": True,
        },
        "unresolved": unresolved,
    }
    receipt = {**body, "readiness_identity": canonical_sha256(body)}
    return _freeze(receipt, field="readiness receipt")
