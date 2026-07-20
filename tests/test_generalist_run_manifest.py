"""Tests for Generalist run manifest construction and lifecycle publication."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.training.generalist_run_manifest import (
    RUN_EVENT_LEDGER_NAME,
    RUN_MANIFEST_NAME,
    append_run_lifecycle_event,
    build_generalist_run_manifest,
    publish_initial_run_contract,
)
from learned_ai.training.run_contract import (
    ContractValidationError,
    canonical_sha256,
    load_run_events,
    load_run_manifest,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        no_sentinel=True,
        no_value_net=True,
        no_gap_net=True,
        no_s1a_warmstart=True,
        no_imitation_mix=True,
        ppo=False,
        start_mode="fresh",
        out_dir=str(tmp_path / "run"),
        specialist_db=str(tmp_path / "specialist.sqlite"),
    )


def _report() -> dict:
    config = {
        "max_games": 1,
        "batch_games": 1,
        "seed": 42,
        "preflight_mode": "smoke",
    }
    return {
        "mode": "smoke",
        "verdict": "ready_for_smoke",
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "resolved_config": config,
        "config_sha256": canonical_sha256(config),
        "checks": {
            "malom": {"identity": "malom-identity"},
            "specialist_db": {
                "identity": "specialist-identity",
                "label_version": "sector-corrected-v1",
                "trust": "trusted",
            },
            "human_db": {
                "identity": "human-identity",
                "label_version": None,
                "trust": "empirical_frequencies_and_outcomes",
            },
        },
    }


def _manifest(tmp_path: Path):
    return build_generalist_run_manifest(
        _args(tmp_path),
        report=_report(),
        root=tmp_path,
        command=("python", "scripts/train_s_gen_v2.py", "--launch", "smoke"),
        run_id="run-001",
        experiment_id="dev-v4-corrected",
        created_at_utc="2026-07-20T09:00:00Z",
        environment={"python": "3.13.1", "cuda_available": False},
    )


def test_manifest_binds_preflight_assets_components_and_outputs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    assert manifest.status == "preflight_passed"
    assert manifest.seeds == {"run": 42}
    assert manifest.components == {
        "sentinel": False,
        "value_net": False,
        "gap_net": False,
        "ppo": False,
        "imitation_warmstart": False,
        "imitation_mix": False,
    }
    assert [asset.logical_name for asset in manifest.assets] == [
        "malom_tablebase",
        "specialist_db",
        "human_db",
    ]
    assert manifest.outputs["run_directory"] == "run"
    assert manifest.checkpoint_policy["start_mode"] == "fresh"


def test_manifest_rejects_nonpassing_or_inconsistent_preflight(tmp_path: Path) -> None:
    report = _report()
    report["verdict"] = "fatal_stop"
    with pytest.raises(ContractValidationError, match="cannot build"):
        build_generalist_run_manifest(
            _args(tmp_path),
            report=report,
            root=tmp_path,
            command=("python", "trainer.py"),
            run_id="run-001",
            experiment_id="experiment",
        )

    report = _report()
    report["config_sha256"] = "0" * 64
    with pytest.raises(ContractValidationError, match="hash is inconsistent"):
        build_generalist_run_manifest(
            _args(tmp_path),
            report=report,
            root=tmp_path,
            command=("python", "trainer.py"),
            run_id="run-001",
            experiment_id="experiment",
        )


def test_weights_only_manifest_records_source_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.start_mode = "weights-only"
    report = _report()
    report["checks"]["checkpoint"] = {
        "identity": "source-identity",
        "format": "legacy-pytorch-weights",
    }

    manifest = build_generalist_run_manifest(
        args,
        report=report,
        root=tmp_path,
        command=("python", "trainer.py"),
        run_id="run-weights",
        experiment_id="experiment",
        created_at_utc="2026-07-20T09:00:00Z",
        environment={"python": "3.13.1"},
    )

    assert manifest.checkpoint_policy["start_mode"] == "weights-only"
    source = next(
        asset for asset in manifest.assets if asset.logical_name == "source_checkpoint"
    )
    assert source.identity == "source-identity"
    assert source.trust_level == "lineage_labeled_weights_only"


def test_exact_resume_manifest_records_full_state_continuation(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.start_mode = "exact-resume"
    report = _report()
    report["checks"]["checkpoint"] = {
        "identity": "source-identity",
        "format": "checkpoint-envelope-v2",
    }

    manifest = build_generalist_run_manifest(
        args,
        report=report,
        root=tmp_path,
        command=("python", "trainer.py"),
        run_id="run-resume",
        experiment_id="experiment",
        created_at_utc="2026-07-20T09:00:00Z",
        environment={"python": "3.13.1"},
    )

    source = next(
        asset for asset in manifest.assets if asset.logical_name == "source_checkpoint"
    )
    assert source.trust_level == "integrity_verified_exact_resume"
    assert source.intended_use == "complete_training_state_continuation"


def test_initial_contract_publication_is_atomic_and_no_overwrite(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = Path(manifest.outputs["run_directory"])
    output = tmp_path / output

    publish_initial_run_contract(output, manifest)

    assert load_run_manifest(output / RUN_MANIFEST_NAME) == manifest
    events = load_run_events(output / RUN_EVENT_LEDGER_NAME)
    assert len(events) == 1
    assert events[0].status == "preflight_passed"
    assert events[0].details["manifest_sha256"] == manifest.manifest_sha256
    assert not list(tmp_path.glob(".*.contract.*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        publish_initial_run_contract(output, manifest)


def test_lifecycle_events_extend_the_hash_chain(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "run"
    publish_initial_run_contract(output, manifest)

    running = append_run_lifecycle_event(
        output,
        run_id=manifest.run_id,
        status="running",
        event_type="training_started",
        timestamp_utc="2026-07-20T09:00:01Z",
    )
    completed = append_run_lifecycle_event(
        output,
        run_id=manifest.run_id,
        status="completed",
        event_type="training_completed",
        timestamp_utc="2026-07-20T09:00:02Z",
    )

    assert running.sequence == 1
    assert completed.sequence == 2
    assert completed.previous_event_sha256 == running.event_sha256
    assert load_run_events(output / RUN_EVENT_LEDGER_NAME)[-1] == completed


def test_lifecycle_rejects_wrong_run_id(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "run"
    publish_initial_run_contract(output, manifest)

    with pytest.raises(ContractValidationError, match="run ID"):
        append_run_lifecycle_event(
            output,
            run_id="different-run",
            status="running",
            event_type="training_started",
        )
