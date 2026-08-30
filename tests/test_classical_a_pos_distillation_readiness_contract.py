from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from learned_ai.training.classical_a_pos_corpus import (
    CORPUS_SCHEMA,
    FROZEN_CORPUS_LAYOUT,
    STATE_SPLIT_ARTIFACT_SCHEMA,
    TRAINING_PROFILE,
    FrozenCorpus,
)
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.classical_a_pos_distillation_readiness import (
    FROZEN_TRAINING_PROFILE,
    DistillationReadinessError,
    build_supervised_run_manifest,
    run_distillation_preflight,
    validate_authorization_envelope,
    validate_requested_resource_package,
    validate_supervised_run_manifest,
    validate_training_profile,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _resource_identities() -> dict[str, str]:
    return {
        "evolved_weights": "1" * 64,
        "fullgame_db": "2" * 64,
        "endgame_db": "3" * 64,
        "phase_value_place": "4" * 64,
        "phase_value_move": "5" * 64,
        "phase_value_fly": "6" * 64,
        "gap_net": "7" * 64,
        "malom_manifest": "8" * 64,
        "malom_content": "9" * 64,
    }


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _resign_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in manifest.items() if key != "manifest_identity"}
    manifest["manifest_identity"] = canonical_sha256(body)
    return manifest


def _split_contract() -> dict[str, Any]:
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


def _valid_corpus_manifest() -> dict[str, Any]:
    split = _split_contract()
    manifest: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": "classical-a-pos-corpus-under-test",
        "profile": TRAINING_PROFILE,
        "status": "complete",
        "source": {"verified_by_strict_loader": True},
        "generator": {"offline_state_generation": True},
        "referee": {"verified_by_strict_loader": True},
        "teacher": {
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
            "active_seconds": 0.0,
            "active_seconds_limit": 14_400,
            "main_search_nodes": 0,
            "restricted_rerank_nodes": 0,
            "aggregate_nodes": 0,
            "aggregate_nodes_limit": 4_000_000_000,
            "positive_search_labels": 0,
            "positive_search_labels_limit": 1_280,
            "completed_labels": 16_384,
        },
        "selection": {"teacher_results_visible_to_selection": False},
        "a_pos_verifier": {
            "boundary": "build-time-chain-plus-load-time-real-PositionalSafetyFilter",
            "identity": "c" * 64,
            "record_count": 16_384,
            "verification_chain_head": "d" * 64,
            "loader_requeries_malom": True,
        },
        "resources_before": _resource_identities(),
        "resources_after": _resource_identities(),
        "state_split_artifact": {
            "schema_version": STATE_SPLIT_ARTIFACT_SCHEMA,
            "file": {
                "path": "states.jsonl",
                "size": 1,
                "sha256": "e" * 64,
                "count": 16_384,
            },
            "identity": "f" * 64,
            "teacher_fields_present": False,
            "controller_freeze_event_required_before_teacher": True,
        },
        "examples": {
            "path": "examples.jsonl",
            "size": 1,
            "sha256": "1" * 64,
            "count": 16_384,
        },
        "singleton_ledger": {
            "path": "singletons.json",
            "size": 1,
            "sha256": "2" * 64,
        },
        "collection_games": 16,
        "split": split,
        "split_identity": canonical_sha256(split),
        "heldout": {"consumed": False, "sources": []},
    }
    manifest["corpus_identity"] = canonical_sha256(manifest)
    return manifest


def _valid_corpus(**manifest_changes: Any) -> FrozenCorpus:
    manifest = _valid_corpus_manifest()
    for field, value in manifest_changes.items():
        manifest[field] = value
    if manifest_changes:
        body = {
            key: value for key, value in manifest.items() if key != "corpus_identity"
        }
        manifest["corpus_identity"] = canonical_sha256(body)
    return FrozenCorpus(
        manifest=manifest,
        examples=(None,) * 16_384,  # type: ignore[arg-type]
        corpus_identity=manifest["corpus_identity"],
        split_identity=manifest["split_identity"],
    )


def _corpus_with_split_origin(
    value: Any, *, resign_split: bool = False
) -> FrozenCorpus:
    split = _split_contract()
    split["algorithm"]["game_index_origin"] = value
    changes: dict[str, Any] = {"split": split}
    if resign_split:
        changes["split_identity"] = canonical_sha256(split)
    return _valid_corpus(**changes)


def _corpus_with_teacher_field(field: str, value: Any) -> FrozenCorpus:
    teacher = _valid_corpus_manifest()["teacher"]
    teacher[field] = value
    return _valid_corpus(teacher=teacher)


def _corpus_with_examples_count(value: Any) -> FrozenCorpus:
    examples = _valid_corpus_manifest()["examples"]
    examples["count"] = value
    return _valid_corpus(examples=examples)


def _corpus_with_state_count(value: Any) -> FrozenCorpus:
    state = _valid_corpus_manifest()["state_split_artifact"]
    state["file"]["count"] = value
    return _valid_corpus(state_split_artifact=state)


def _corpus_with_verifier_count(value: Any) -> FrozenCorpus:
    verifier = _valid_corpus_manifest()["a_pos_verifier"]
    verifier["record_count"] = value
    return _valid_corpus(a_pos_verifier=verifier)


def _corpus_with_verifier_field(field: str, value: Any) -> FrozenCorpus:
    verifier = _valid_corpus_manifest()["a_pos_verifier"]
    verifier[field] = value
    return _valid_corpus(a_pos_verifier=verifier)


def _corpus_with_state_field(field: str, value: Any) -> FrozenCorpus:
    state = _valid_corpus_manifest()["state_split_artifact"]
    state[field] = value
    return _valid_corpus(state_split_artifact=state)


def _corpus_with_selection_field(field: str, value: Any) -> FrozenCorpus:
    selection = _valid_corpus_manifest()["selection"]
    selection[field] = value
    return _valid_corpus(selection=selection)


def _manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus" / "manifest.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    return path


def test_profile_and_manifest_are_deeply_immutable() -> None:
    checked = validate_training_profile(_mutable(FROZEN_TRAINING_PROFILE))
    source = _mutable(FROZEN_TRAINING_PROFILE)
    validated_copy = validate_training_profile(source)
    source["model"]["policy_hidden"][0] = 999

    with pytest.raises(TypeError):
        checked["training_semantics"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        checked["optimizer"]["lr"] = 2e-3  # type: ignore[index]
    assert validated_copy["model"]["policy_hidden"] == (128, 64)

    manifest = build_supervised_run_manifest(
        checked,
        corpus_identity=SHA_A,
        split_identity=SHA_B,
    )
    with pytest.raises(TypeError):
        manifest["executable"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest["requested_resource_package"]["teacher"]["nodes"] = 1  # type: ignore[index]


def test_profile_binds_current_corpus_model_optimizer_and_seed_constants() -> None:
    profile = validate_training_profile(FROZEN_TRAINING_PROFILE)

    assert profile["corpus"] == {
        "total_examples": 16_384,
        "train_examples": 14_336,
        "dev_examples": 2_048,
        "whole_game_split": True,
    }
    assert profile["seeds"] == (2026083001, 2026083002, 2026083003)
    assert profile["model"] == {
        "move_feature_dim": 62,
        "policy_hidden": (128, 64),
        "value_hidden": (),
        "dropout": 0.0,
        "value_input_dim": 23,
        "value_branch_in_optimizer": False,
    }
    assert profile["optimizer"]["kind"] == "Adam"
    assert profile["optimizer"]["lr"] == 1e-3
    assert profile["optimizer"]["batch_size"] == 128
    assert profile["optimizer"]["epochs"] == 20
    assert profile["optimizer"]["grad_clip_norm"] == 1.0
    assert profile["optimizer_updates_per_seed"] == 2_240
    assert profile["device"] == {
        "kind": "cpu",
        "cuda": False,
        "throughput_claim": None,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.pop("model"), "missing.*model"),
        (lambda value: value.update({"unknown_asset": "x"}), "unknown.*unknown_asset"),
        (
            lambda value: value["model"].update({"unknown": 1}),
            "unknown.*unknown",
        ),
        (
            lambda value: value["optimizer"].update({"lr": 2e-3}),
            "profile differs",
        ),
        (
            lambda value: value["forbidden"].update({"online_teacher_queries": True}),
            "profile differs",
        ),
        (lambda value: value["rl"].update({"a2c": True}), "profile differs"),
    ],
)
def test_profile_rejects_missing_unknown_and_nested_drift(
    mutation: Any,
    match: str,
) -> None:
    profile = _mutable(FROZEN_TRAINING_PROFILE)
    mutation(profile)
    with pytest.raises(DistillationReadinessError, match=match):
        validate_training_profile(profile)


def test_manifest_separates_resource_recommendation_from_zero_authority() -> None:
    manifest = build_supervised_run_manifest(
        FROZEN_TRAINING_PROFILE,
        corpus_identity=SHA_A,
        split_identity=SHA_B,
    )
    assert manifest["proposal_status"] == "proposed"
    assert "status" not in manifest
    resources = manifest["requested_resource_package"]
    authority = manifest["authorization_envelope"]

    assert set(resources) == {
        "schema_version",
        "recommendation_only",
        "counts_as_authorization",
        "state_generation",
        "teacher",
        "smoke",
        "seed",
        "sequence_active_seconds",
        "evaluation_games",
    }
    assert resources["schema_version"] == "nmm.classical-a-pos-requested-resource.v1"
    assert resources["recommendation_only"] is True
    assert resources["counts_as_authorization"] is False
    assert resources["state_generation"] == {
        "games": 1_024,
        "active_seconds": 3_600,
    }
    assert resources["teacher"] == {
        "active_seconds": 14_400,
        "nodes": 4_000_000_000,
        "positive_search_labels": 1_280,
    }
    assert resources["smoke"]["active_seconds"] == 3_600
    assert resources["seed"] == {
        "per_seed_active_seconds": 21_600,
        "aggregate_active_seconds": 64_800,
    }
    assert resources["sequence_active_seconds"] == 86_400
    assert resources["evaluation_games"] == 0

    assert set(authority) == {
        "schema_version",
        "status",
        "authorization_identity",
        "authorized_by",
        "issued_at_utc",
        "expires_at_utc",
        "standing_delegation_identity",
        "consumption_limit",
        "allowed_operations",
        "allow_exact_resume",
        "authorized_resources",
    }
    assert authority["schema_version"] == ("nmm.classical-a-pos-zero-authorization.v1")
    assert authority["status"] == "unauthorized"
    assert authority["authorization_identity"] is None
    assert authority["authorized_by"] is None
    assert authority["issued_at_utc"] is None
    assert authority["expires_at_utc"] is None
    assert authority["standing_delegation_identity"] is None
    assert authority["allowed_operations"] == ()
    assert authority["consumption_limit"] == 0
    assert authority["allow_exact_resume"] is False
    assert authority["authorized_resources"] == {
        "games": 0,
        "active_seconds": 0,
        "teacher_nodes": 0,
        "positive_search_labels": 0,
        "evaluation_games": 0,
    }
    with pytest.raises(DistillationReadinessError):
        validate_authorization_envelope(resources)
    with pytest.raises(DistillationReadinessError):
        validate_requested_resource_package(authority)


@pytest.mark.parametrize("delta", [-1, 1])
def test_manifest_rejects_resource_recommendation_drift(delta: int) -> None:
    manifest = _mutable(
        build_supervised_run_manifest(
            FROZEN_TRAINING_PROFILE,
            corpus_identity=SHA_A,
            split_identity=SHA_B,
        )
    )
    manifest["requested_resource_package"]["teacher"]["nodes"] += delta
    _resign_manifest(manifest)
    with pytest.raises(DistillationReadinessError, match="requested resource package"):
        validate_supervised_run_manifest(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["authorization_envelope"].update(
            {"authorization_identity": "c" * 64}
        ),
        lambda value: value["authorization_envelope"].update(
            {"allowed_operations": ["train"]}
        ),
        lambda value: value["authorization_envelope"]["authorized_resources"].update(
            {"teacher_nodes": 1}
        ),
        lambda value: value["authorization_envelope"].update({"consumption_limit": 1}),
        lambda value: value["authorization_envelope"].update(
            {"allow_exact_resume": True}
        ),
        lambda value: value.update({"executable": True}),
    ],
)
def test_manifest_rejects_any_authority_or_executable_escalation(
    mutation: Any,
) -> None:
    manifest = _mutable(
        build_supervised_run_manifest(
            FROZEN_TRAINING_PROFILE,
            corpus_identity=SHA_A,
            split_identity=SHA_B,
        )
    )
    mutation(manifest)
    _resign_manifest(manifest)
    with pytest.raises(DistillationReadinessError):
        validate_supervised_run_manifest(manifest)


def test_zero_authorization_status_cannot_be_escalated() -> None:
    manifest = _mutable(
        build_supervised_run_manifest(
            FROZEN_TRAINING_PROFILE,
            corpus_identity=SHA_A,
            split_identity=SHA_B,
        )
    )
    authority = manifest["authorization_envelope"]
    authority["status"] = "authorized"

    with pytest.raises(
        DistillationReadinessError,
        match=r"authorization envelope\.status differs",
    ):
        validate_authorization_envelope(authority)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.pop("model"), "missing.*model"),
        (lambda value: value.update({"unknown": 1}), "unknown.*unknown"),
        (
            lambda value: value["optimizer"].update({"unknown": 1}),
            "unknown.*unknown",
        ),
        (
            lambda value: value["optimizer"].update({"batch_size": 127}),
            "run manifest",
        ),
    ],
)
def test_manifest_rejects_missing_unknown_and_nested_drift(
    mutation: Any,
    match: str,
) -> None:
    manifest = _mutable(
        build_supervised_run_manifest(
            FROZEN_TRAINING_PROFILE,
            corpus_identity=SHA_A,
            split_identity=SHA_B,
        )
    )
    mutation(manifest)
    _resign_manifest(manifest)
    with pytest.raises(DistillationReadinessError, match=match):
        validate_supervised_run_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("corpus_identity", "A" * 64),
        ("corpus_identity", "a" * 63),
        ("split_identity", "B" * 64),
        ("split_identity", "b" * 65),
    ],
)
def test_manifest_requires_exact_lowercase_sha256_identities(
    field: str,
    value: str,
) -> None:
    kwargs = {"corpus_identity": SHA_A, "split_identity": SHA_B, field: value}
    with pytest.raises(DistillationReadinessError, match="lowercase SHA-256"):
        build_supervised_run_manifest(FROZEN_TRAINING_PROFILE, **kwargs)


def test_manifest_requires_its_canonical_identity_even_after_semantic_validation() -> (
    None
):
    manifest = _mutable(
        build_supervised_run_manifest(
            FROZEN_TRAINING_PROFILE,
            corpus_identity=SHA_A,
            split_identity=SHA_B,
        )
    )
    manifest["manifest_identity"] = "c" * 64
    with pytest.raises(DistillationReadinessError, match="canonical identity"):
        validate_supervised_run_manifest(manifest)


def test_preflight_validates_profile_before_paths_or_loader() -> None:
    profile = _mutable(FROZEN_TRAINING_PROFILE)
    profile["forbidden"]["human_db"] = True
    calls: list[Path] = []

    class ExplodingPath:
        def __fspath__(self) -> str:
            raise AssertionError("paths must not be touched before profile validation")

    def loader(path: Path) -> FrozenCorpus:
        calls.append(path)
        raise AssertionError("loader must not run")

    with pytest.raises(DistillationReadinessError, match="profile differs"):
        run_distillation_preflight(
            profile,
            corpus_manifest_path=ExplodingPath(),  # type: ignore[arg-type]
            output_dir=ExplodingPath(),  # type: ignore[arg-type]
            corpus_loader=loader,
        )
    assert calls == []


@pytest.mark.parametrize(
    "corpus_factory,match",
    [
        (
            lambda: FrozenCorpus(
                manifest=_valid_corpus_manifest(),
                examples=(None,) * 16_384,  # type: ignore[arg-type]
                corpus_identity="3" * 64,
                split_identity=_valid_corpus_manifest()["split_identity"],
            ),
            "corpus identity",
        ),
        (
            lambda: _valid_corpus(split_identity="3" * 64),
            "split identity",
        ),
        (
            lambda: FrozenCorpus(
                manifest=_valid_corpus_manifest(),
                examples=(None,) * 16_383,  # type: ignore[arg-type]
                corpus_identity=_valid_corpus_manifest()["corpus_identity"],
                split_identity=_valid_corpus_manifest()["split_identity"],
            ),
            "16,384",
        ),
        (
            lambda: _valid_corpus(heldout={"consumed": True, "sources": []}),
            "heldout",
        ),
        (
            lambda: _valid_corpus(
                teacher={
                    **_valid_corpus_manifest()["teacher"],
                    "online_queries_during_training": True,
                }
            ),
            r"corpus teacher\.online_queries_during_training",
        ),
        (
            lambda: _valid_corpus(
                examples={
                    **_valid_corpus_manifest()["examples"],
                    "count": 16_383,
                }
            ),
            r"corpus examples\.count",
        ),
        (
            lambda: _valid_corpus(status="building"),
            r"corpus manifest\.status",
        ),
    ],
)
def test_preflight_rejects_wrong_corpus_contract(
    tmp_path: Path,
    corpus_factory: Any,
    match: str,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    with pytest.raises(DistillationReadinessError, match=match):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=tmp_path / "isolated-output",
            corpus_loader=lambda _path: corpus_factory(),
        )


@pytest.mark.parametrize(
    "corpus_factory,match",
    [
        (
            lambda: _corpus_with_split_origin(False),
            r"corpus split\.algorithm\.game_index_origin",
        ),
        (
            lambda: _corpus_with_split_origin(0.0, resign_split=True),
            r"corpus split\.algorithm\.game_index_origin",
        ),
        (
            lambda: _valid_corpus(heldout={"consumed": 0, "sources": []}),
            r"corpus heldout\.consumed",
        ),
        (
            lambda: _corpus_with_teacher_field("difficulty", 9.0),
            r"corpus teacher\.difficulty",
        ),
        (
            lambda: _corpus_with_teacher_field("threads", True),
            r"corpus teacher\.threads",
        ),
        (
            lambda: _corpus_with_teacher_field("completed_labels", 16_384.0),
            r"corpus teacher\.completed_labels",
        ),
        (
            lambda: _corpus_with_examples_count(16_384.0),
            r"corpus examples\.count",
        ),
        (
            lambda: _corpus_with_state_count(16_384.0),
            r"corpus state/split file\.count",
        ),
        (
            lambda: _corpus_with_verifier_count(16_384.0),
            r"corpus A_pos verifier\.record_count",
        ),
        (
            lambda: _corpus_with_verifier_field("loader_requeries_malom", 1),
            r"corpus A_pos verifier\.loader_requeries_malom",
        ),
        (
            lambda: _corpus_with_state_field("teacher_fields_present", 0),
            r"corpus state/split artifact\.teacher_fields_present",
        ),
        (
            lambda: _corpus_with_selection_field(
                "teacher_results_visible_to_selection", 0
            ),
            r"corpus selection\.teacher_results_visible_to_selection",
        ),
    ],
)
def test_preflight_rejects_python_numeric_type_spoofs(
    tmp_path: Path,
    corpus_factory: Any,
    match: str,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    with pytest.raises(DistillationReadinessError, match=match):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=tmp_path / "isolated-output",
            corpus_loader=lambda _path: corpus_factory(),
        )


@pytest.mark.parametrize("collection_games", [0, 15, 17, 1_025, True])
def test_preflight_rechecks_forged_collection_game_ledger(
    tmp_path: Path,
    collection_games: Any,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    with pytest.raises(DistillationReadinessError, match="collection game ledger"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=tmp_path / "isolated-output",
            corpus_loader=lambda _path: _valid_corpus(
                collection_games=collection_games
            ),
        )


def test_preflight_rechecks_forged_resource_identity_ledger(tmp_path: Path) -> None:
    manifest_path = _manifest_file(tmp_path)
    changed = _resource_identities()
    changed["gap_net"] = "a" * 64

    with pytest.raises(DistillationReadinessError, match="resource ledger"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=tmp_path / "isolated-output",
            corpus_loader=lambda _path: _valid_corpus(resources_after=changed),
        )


def test_preflight_rejects_nonempty_and_overlapping_output_paths(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "evidence.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(DistillationReadinessError, match="not empty"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=nonempty,
            corpus_loader=lambda _path: _valid_corpus(),
        )
    with pytest.raises(DistillationReadinessError, match="overlap"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=manifest_path.parent,
            corpus_loader=lambda _path: _valid_corpus(),
        )


def test_preflight_rejects_missing_manifest_before_calling_loader(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def loader(path: Path) -> FrozenCorpus:
        calls.append(path)
        return _valid_corpus()

    with pytest.raises(DistillationReadinessError, match="manifest path is missing"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=tmp_path / "missing.json",
            output_dir=tmp_path / "output",
            corpus_loader=loader,
        )
    assert calls == []


def test_preflight_wraps_loader_failure_and_writes_nothing(tmp_path: Path) -> None:
    manifest_path = _manifest_file(tmp_path)
    output = tmp_path / "isolated-output"

    def loader(_path: Path) -> FrozenCorpus:
        raise ValueError("simulated strict-loader failure")

    with pytest.raises(DistillationReadinessError, match="strict corpus loading"):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=output,
            corpus_loader=loader,
        )
    assert not output.exists()


def test_existing_empty_output_is_inspected_but_not_written(tmp_path: Path) -> None:
    manifest_path = _manifest_file(tmp_path)
    output = tmp_path / "isolated-output"
    output.mkdir()

    receipt = run_distillation_preflight(
        FROZEN_TRAINING_PROFILE,
        corpus_manifest_path=manifest_path,
        output_dir=output,
        corpus_loader=lambda _path: _valid_corpus(),
    )

    assert receipt["output_state"] == "empty"
    assert list(output.iterdir()) == []
    assert receipt["verdict"] == "fatal_stop"


def test_output_second_inspection_oserror_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    output = tmp_path / "isolated-output"
    output.mkdir()
    target = output.resolve()
    real_iterdir = Path.iterdir
    calls = 0

    def flaky_iterdir(path: Path):
        nonlocal calls
        if path.resolve() == target:
            calls += 1
            if calls == 2:
                raise OSError("simulated second inspection failure")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", flaky_iterdir)
    with pytest.raises(
        DistillationReadinessError,
        match="output directory cannot be inspected",
    ):
        run_distillation_preflight(
            FROZEN_TRAINING_PROFILE,
            corpus_manifest_path=manifest_path,
            output_dir=output,
            corpus_loader=lambda _path: _valid_corpus(),
        )


def test_successful_structural_preflight_still_returns_fatal_stop_without_writes(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest_file(tmp_path)
    output = tmp_path / "isolated-output"
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    calls: list[Path] = []

    def loader(path: Path) -> FrozenCorpus:
        calls.append(path)
        return _valid_corpus()

    receipt = run_distillation_preflight(
        FROZEN_TRAINING_PROFILE,
        corpus_manifest_path=manifest_path,
        output_dir=output,
        corpus_loader=loader,
    )

    assert calls == [manifest_path.resolve()]
    assert receipt["verdict"] == "fatal_stop"
    assert receipt["launch_status"] == "unlaunched"
    assert receipt["authorization_status"] == "unauthorized"
    assert receipt["executable"] is False
    assert len(receipt["readiness_identity"]) == 64
    assert receipt["readiness_identity"] == receipt["readiness_identity"].lower()
    assert receipt["unresolved"] == (
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
    )
    assert receipt["checks"]["frozen_corpus_contract_rechecked"] is True
    assert "strict_frozen_corpus_loaded" not in receipt["checks"]
    assert not output.exists()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    with pytest.raises(TypeError):
        receipt["executable"] = True  # type: ignore[index]
