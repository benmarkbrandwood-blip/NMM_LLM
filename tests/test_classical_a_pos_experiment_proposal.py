from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.classical_a_pos_distillation_readiness import (
    FROZEN_TRAINING_PROFILE,
    REQUESTED_RESOURCE_PACKAGE,
    UNRESOLVED_BINDINGS,
    ZERO_AUTHORIZATION_ENVELOPE,
    validate_authorization_envelope,
    validate_requested_resource_package,
    validate_training_profile,
)
from learned_ai.validation.classical_a_pos_experiment_proposal import (
    build_classical_a_pos_experiment_proposal,
    load_classical_a_pos_experiment_proposal,
    verify_classical_a_pos_experiment_proposal,
)


PROPOSAL_PATH = (
    Path(__file__).parents[1]
    / "docs"
    / "experiments"
    / "classical-a-pos-offline-distillation-v1.proposal.json"
)
PROFILE_IDENTITY = "bfa8d2f8e19b1c24641e24e4765844f678cb9288838e5eda3f8782d7ace9cbe0"
BASE_COMMIT = "e210112540f8869c15f5d4737631a29a9a66341b"
TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "proposal_status",
    "launch_status",
    "authorization_status",
    "executable",
    "reviewed_base_commit",
    "profile_identity",
    "training_profile",
    "requested_resource_package",
    "authorization_envelope",
    "objective",
    "operation_order",
    "claim_boundaries",
    "stop_conditions",
    "prohibited_actions",
    "unresolved_bindings",
    "proposal_identity",
}


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _resign(proposal: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in proposal.items() if key != "proposal_identity"}
    proposal["proposal_identity"] = canonical_sha256(body)
    return proposal


def _canonical_file_bytes(proposal: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(proposal) + b"\n"


def test_checked_in_proposal_is_exact_canonical_builder_output() -> None:
    built = build_classical_a_pos_experiment_proposal()
    loaded = load_classical_a_pos_experiment_proposal(PROPOSAL_PATH)

    assert loaded == built
    assert PROPOSAL_PATH.read_bytes() == _canonical_file_bytes(built)
    assert canonical_sha256(FROZEN_TRAINING_PROFILE) == PROFILE_IDENTITY


def test_proposal_exact_status_identity_and_base_contract() -> None:
    proposal = build_classical_a_pos_experiment_proposal()

    assert set(proposal) == TOP_LEVEL_KEYS
    assert (
        proposal["schema_version"]
        == "nmm.classical-a-pos-offline-distillation-experiment-proposal.v1"
    )
    assert proposal["experiment_id"] == "classical-a-pos-offline-distillation-v1"
    assert proposal["proposal_status"] == "proposed"
    assert proposal["launch_status"] == "unlaunched"
    assert proposal["authorization_status"] == "unauthorized"
    assert proposal["executable"] is False
    assert proposal["reviewed_base_commit"] == BASE_COMMIT
    assert proposal["profile_identity"] == PROFILE_IDENTITY
    body = {key: value for key, value in proposal.items() if key != "proposal_identity"}
    assert proposal["proposal_identity"] == canonical_sha256(body)


def test_proposal_semantic_literals_and_order_are_exact() -> None:
    proposal = build_classical_a_pos_experiment_proposal()

    assert proposal["objective"] == {
        "objective_id": "offline-classical-a-pos-policy-distillation-v1",
        "teacher_boundary": "current-head-exact-D9-post-ProductPositionalSafetyGate",
        "student_boundary": "C1-FROZEN_TRAINING_PROFILE",
        "result_boundary": "three-fresh-supervised-seed-checkpoints",
    }
    assert proposal["operation_order"] == (
        "runtime-plan-freeze",
        "technical-readiness-freeze",
        "single-use-authorization-consumption",
        "state-generation",
        "state-freeze",
        "offline-d9-teacher-labeling",
        "corpus-freeze",
        "supervised-smoke",
        "seed-2026083001",
        "seed-2026083002",
        "seed-2026083003",
    )
    assert proposal["claim_boundaries"] == {
        "permitted": (
            "proposal-contract-is-frozen-and-authorization-free",
            "future-completed-run-may-report-frozen-corpus-imitation-metrics-only",
        ),
        "excluded": (
            "playing-strength-improvement",
            "human-specialization",
            "heldout-generalization",
            "product-safety-or-product-readiness",
            "checkpoint-promotion",
            "publication-or-release",
        ),
    }
    assert proposal["stop_conditions"] == (
        "proposal-or-c1-contract-drift",
        "any-unresolved-binding",
        "teacher-before-state-freeze",
        "missing-or-drifting-data-checkpoint-path-git-device-or-smoke-provenance",
        "quota-or-resource-cap-exceeded",
        "authorization-missing-expired-revoked-consumed-or-plan-mismatched",
        "nonfinite-or-failed-operation",
        "output-not-isolated",
        "prohibited-action-requested",
    )
    assert proposal["prohibited_actions"] == (
        "launch-smoke-or-training-from-this-proposal",
        "online-teacher-queries-during-student-training",
        "rl-a2c-ppo-value-or-entropy-training",
        "human-db-specialist-db-or-advisor-inputs",
        "heldout-evaluation-games",
        "automatic-retry",
        "checkpoint-or-corpus-relabeling",
        "post-hoc-corpus-shrink-or-quota-change",
        "product-route-switch",
        "checkpoint-promotion-publication-or-release",
    )


def test_nested_contracts_derive_from_and_pass_c1_validators() -> None:
    proposal = build_classical_a_pos_experiment_proposal()

    assert proposal["training_profile"] == FROZEN_TRAINING_PROFILE
    assert proposal["requested_resource_package"] == REQUESTED_RESOURCE_PACKAGE
    assert proposal["authorization_envelope"] == ZERO_AUTHORIZATION_ENVELOPE
    validate_training_profile(proposal["training_profile"])
    validate_requested_resource_package(proposal["requested_resource_package"])
    validate_authorization_envelope(proposal["authorization_envelope"])
    assert proposal["unresolved_bindings"] == UNRESOLVED_BINDINGS


def test_built_and_verified_proposals_are_deeply_immutable() -> None:
    built = build_classical_a_pos_experiment_proposal()
    verified = verify_classical_a_pos_experiment_proposal(_mutable(built))

    for proposal in (built, verified):
        with pytest.raises(TypeError):
            proposal["executable"] = True  # type: ignore[index]
        with pytest.raises(TypeError):
            proposal["objective"]["objective_id"] = "changed"  # type: ignore[index]
        with pytest.raises(TypeError):
            proposal["operation_order"][0] = "changed"  # type: ignore[index]


def test_caller_mutation_after_verify_cannot_change_result() -> None:
    source = _mutable(build_classical_a_pos_experiment_proposal())
    verified = verify_classical_a_pos_experiment_proposal(source)

    source["objective"]["objective_id"] = "changed-after-verification"
    source["operation_order"].reverse()

    assert verified == build_classical_a_pos_experiment_proposal()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proposal_status", "approved"),
        ("launch_status", "launchable"),
        ("authorization_status", "authorized"),
        ("executable", True),
        ("reviewed_base_commit", "f" * 40),
        ("profile_identity", "f" * 64),
    ],
)
def test_status_commit_or_identity_upgrade_is_rejected_even_when_resigned(
    field: str,
    value: Any,
) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal[field] = value

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "authorized"),
        (("authorization_identity",), "a" * 64),
        (("authorized_by",), "product-owner"),
        (("allowed_operations",), ["training"]),
        (("allow_exact_resume",), True),
        (("consumption_limit",), 1),
        (("authorized_resources", "games"), 1),
        (("authorized_resources", "active_seconds"), 1),
        (("authorized_resources", "teacher_nodes"), 1),
    ],
)
def test_any_positive_authorization_is_rejected_even_when_resigned(
    path: tuple[str, ...],
    value: Any,
) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    target = proposal["authorization_envelope"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


def test_resource_and_authorization_contracts_cannot_be_swapped() -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal["requested_resource_package"], proposal["authorization_envelope"] = (
        proposal["authorization_envelope"],
        proposal["requested_resource_package"],
    )

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


@pytest.mark.parametrize(
    "field",
    [
        "objective",
        "operation_order",
        "claim_boundaries",
        "stop_conditions",
        "prohibited_actions",
        "unresolved_bindings",
    ],
)
def test_semantic_body_drift_is_rejected_even_when_resigned(field: str) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    value = proposal[field]
    if isinstance(value, list):
        value[0] = f"changed-{value[0]}"
    else:
        first_key = next(iter(value))
        value[first_key] = f"changed-{value[first_key]}"

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


@pytest.mark.parametrize(
    "field",
    ["operation_order", "stop_conditions", "prohibited_actions", "unresolved_bindings"],
)
def test_semantic_array_reordering_is_rejected_even_when_resigned(field: str) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal[field][0], proposal[field][1] = proposal[field][1], proposal[field][0]

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


def test_claim_boundary_array_reordering_is_rejected_even_when_resigned() -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal["claim_boundaries"]["excluded"].reverse()

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


def test_identity_mutation_and_unsigned_body_mutation_are_rejected() -> None:
    identity_mutation = _mutable(build_classical_a_pos_experiment_proposal())
    identity_mutation["proposal_identity"] = "f" * 64
    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(identity_mutation)

    unsigned_mutation = _mutable(build_classical_a_pos_experiment_proposal())
    unsigned_mutation["objective"]["objective_id"] = "changed"
    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(unsigned_mutation)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_plan",
        "authorization",
        "command",
        "path",
        "checkpoint",
        "controller",
        "cli",
        "issuer",
        "corpus_identity",
        "split_identity",
    ],
)
def test_forbidden_runtime_or_execution_field_is_rejected(field: str) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal[field] = "forbidden"

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


@pytest.mark.parametrize("missing", sorted(TOP_LEVEL_KEYS))
def test_every_missing_top_level_key_is_rejected(missing: str) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    del proposal[missing]
    if missing != "proposal_identity":
        _resign(proposal)

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(proposal)


def test_unknown_nested_key_is_rejected_even_when_resigned() -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    proposal["objective"]["runtime"] = "injected"

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


@pytest.mark.parametrize(
    ("path", "spoof"),
    [
        (("executable",), 0),
        (("requested_resource_package", "state_generation", "games"), True),
        (("authorization_envelope", "consumption_limit"), False),
        (("authorization_envelope", "authorized_resources", "games"), False),
    ],
)
def test_bool_int_type_spoofs_are_rejected_even_when_resigned(
    path: tuple[str, ...],
    spoof: Any,
) -> None:
    proposal = _mutable(build_classical_a_pos_experiment_proposal())
    target = proposal
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = spoof

    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal(_resign(proposal))


def test_verify_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        verify_classical_a_pos_experiment_proposal([])  # type: ignore[arg-type]


def test_loader_rejects_bom_duplicate_keys_and_non_json_constants(
    tmp_path: Path,
) -> None:
    canonical = _canonical_file_bytes(build_classical_a_pos_experiment_proposal())
    variants = {
        "bom": b"\xef\xbb\xbf" + canonical,
        "duplicate": canonical.replace(
            b"{",
            b'{"schema_version":"duplicate",',
            1,
        ),
        "nan": canonical.replace(b'"executable":false', b'"executable":NaN'),
        "infinity": canonical.replace(b'"executable":false', b'"executable":Infinity'),
        "negative-infinity": canonical.replace(
            b'"executable":false', b'"executable":-Infinity'
        ),
    }

    for name, payload in variants.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            load_classical_a_pos_experiment_proposal(path)


@pytest.mark.parametrize(
    "transform",
    [
        lambda payload: payload.removesuffix(b"\n"),
        lambda payload: payload.removesuffix(b"\n") + b"\r\n",
        lambda payload: b" " + payload,
        lambda payload: payload.replace(b'":', b'": ', 1),
        lambda payload: payload + b"\n",
    ],
)
def test_loader_rejects_noncanonical_bytes(tmp_path: Path, transform: Any) -> None:
    canonical = _canonical_file_bytes(build_classical_a_pos_experiment_proposal())
    path = tmp_path / "proposal.json"
    path.write_bytes(transform(canonical))

    with pytest.raises(ValueError):
        load_classical_a_pos_experiment_proposal(path)


def test_loader_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "proposal.json"
    path.write_bytes(b"[]\n")

    with pytest.raises(ValueError):
        load_classical_a_pos_experiment_proposal(path)


def test_public_surface_has_zero_arg_builder_and_no_mutating_operation() -> None:
    import learned_ai.validation.classical_a_pos_experiment_proposal as module

    assert module.__all__ == (
        "build_classical_a_pos_experiment_proposal",
        "verify_classical_a_pos_experiment_proposal",
        "load_classical_a_pos_experiment_proposal",
    )
    assert (
        tuple(inspect.signature(build_classical_a_pos_experiment_proposal).parameters)
        == ()
    )
    assert tuple(
        inspect.signature(verify_classical_a_pos_experiment_proposal).parameters
    ) == ("proposal",)
    assert tuple(
        inspect.signature(load_classical_a_pos_experiment_proposal).parameters
    ) == ("path",)
    assert not any(
        forbidden in name.lower()
        for name in module.__all__
        for forbidden in ("publish", "launch", "authorize", "execute", "write")
    )


def test_build_verify_and_load_have_no_write_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_bytes(
        _canonical_file_bytes(build_classical_a_pos_experiment_proposal())
    )
    original = proposal_path.read_bytes()

    def reject_write(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("proposal API attempted a write")

    monkeypatch.setattr(Path, "write_bytes", reject_write)
    monkeypatch.setattr(Path, "write_text", reject_write)
    monkeypatch.setattr(Path, "mkdir", reject_write)
    monkeypatch.setattr(Path, "touch", reject_write)

    built = build_classical_a_pos_experiment_proposal()
    verified = verify_classical_a_pos_experiment_proposal(built)
    loaded = load_classical_a_pos_experiment_proposal(proposal_path)

    assert built == verified == loaded
    assert proposal_path.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (proposal_path,)


def test_loaded_contract_is_deeply_immutable() -> None:
    loaded = load_classical_a_pos_experiment_proposal(PROPOSAL_PATH)

    with pytest.raises(TypeError):
        loaded["launch_status"] = "launched"  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded["authorization_envelope"]["consumption_limit"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded["unresolved_bindings"][0] = "resolved"  # type: ignore[index]
