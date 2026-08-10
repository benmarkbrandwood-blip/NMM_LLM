"""Tests for strict, read-only Generalist v2 preflight checks."""

from __future__ import annotations

import json
import hashlib
import random
import sqlite3
from pathlib import Path

import pytest
import torch

from learned_ai.training import generalist_preflight as preflight_module
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.data.data_contract import publish_dataset_manifest
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.training.generalist_preflight import (
    GitState,
    PreflightConfigurationError,
    _file_sha256,
    configure_generalist_paths,
    load_training_settings,
    resolved_resume_config,
    resume_config_sha256,
    run_generalist_preflight,
    validate_generalist_configuration,
)
from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    CheckpointPayload,
    capture_rng_state,
    save_checkpoint,
)
from learned_ai.training.managed_generalist import (
    ManagedPlan,
    authorize_plan,
    publish_managed_plan,
)
from scripts import train_s_gen_v2 as trainer
from scripts.build_malom_dataset_manifest import build_manifest


def _write_specialist_db(path: Path, version: str | None) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE positions (
            pos_hash TEXT PRIMARY KEY,
            malom_label TEXT
        );
        CREATE TABLE winning_lines (id INTEGER PRIMARY KEY);
        CREATE TABLE preferred_plays (id INTEGER PRIMARY KEY);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    if version is not None:
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('malom_label_version', ?)",
            (version,),
        )
    connection.commit()
    connection.close()


def _write_human_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE positions (id INTEGER PRIMARY KEY);
        CREATE TABLE moves (id INTEGER PRIMARY KEY);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    connection.commit()
    connection.close()


def _write_malom(path: Path) -> None:
    path.mkdir()
    (path / "std.secval").write_text(
        "virt_loss_val: -299\nvirt_win_val: 299\n",
        encoding="ascii",
    )
    (path / "std_test.sec2").write_bytes(b"")
    publish_dataset_manifest(path.parent / "malom.manifest.json", build_manifest(path))


def _smoke_args(tmp_path: Path):
    parser = trainer._build_argument_parser()
    return parser.parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "new-output"),
            "--malom",
            str(tmp_path / "malom"),
            "--malom-manifest",
            str(tmp_path / "malom.manifest.json"),
            "--human-db",
            str(tmp_path / "human.sqlite"),
            "--specialist-db",
            str(tmp_path / "specialist.sqlite"),
            "--ruleset-manifest",
            str(
                Path(trainer.__file__).resolve().parents[1]
                / "data"
                / "rulesets"
                / "nmm-training-core@2.json"
            ),
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
            "--no-opening-forcing",
            "--max-games",
            "1",
            "--batch-games",
            "1",
        ]
    )


def test_settings_loader_rejects_duplicate_and_unknown_local_keys(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    local = data / "paths.json"
    local.write_text(
        '{"malom_db_path":"one","malom_db_path":"two"}', encoding="utf-8"
    )

    with pytest.raises(PreflightConfigurationError, match="duplicate JSON key"):
        load_training_settings(tmp_path, str(local))

    local.write_text('{"unexpected_path":"value"}', encoding="utf-8")
    with pytest.raises(PreflightConfigurationError, match="unknown training path"):
        load_training_settings(tmp_path, str(local))


def test_settings_loader_accepts_documented_lookup_only_keys(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    local = data / "paths.json"
    registry_only = {
        "human_db_prefix12_snapshot_path": "snapshot.sqlite",
        "human_db_prefix12_source_manifest_path": "source.jsonl",
        "human_db_prefix12_history_ledger_path": "history.jsonl",
        "human_games_imported_manifest_path": "imported.json",
        "sanmill_prefix12_checkout": "sanmill-prefix-runtime",
        "mif_checkout": "mif-runtime",
    }
    local.write_text(json.dumps(registry_only), encoding="utf-8")

    settings = load_training_settings(tmp_path, str(local))

    assert {key: settings.values[key] for key in registry_only} == registry_only


def test_path_resolution_records_cli_environment_config_and_disable_sources(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.json").write_text(
        json.dumps({"human_db_path": "shared-human.sqlite"}), encoding="utf-8"
    )
    local = data / "paths.json"
    local.write_text(
        json.dumps(
            {
                "malom_db_path": "local-malom",
                "sanmill_training_checkout": "local-sanmill-runtime",
            }
        ),
        encoding="utf-8",
    )
    args = _smoke_args(tmp_path)
    args.out_dir = "cli-output"
    args.malom = None
    args.human_db = None
    args.specialist_db = None
    settings = load_training_settings(tmp_path, str(local))

    sources = configure_generalist_paths(
        args,
        root=tmp_path,
        settings=settings,
        environ={"NMM_SPECIALIST_DB": "environment.sqlite"},
    )

    assert sources["out_dir"] == "cli"
    assert sources["malom"] == "local_path_config:malom_db_path"
    assert sources["human_db"] == "shared_config:human_db_path"
    assert sources["specialist_db"] == "environment:NMM_SPECIALIST_DB"
    assert sources["sanmill_runtime"] == (
        "local_path_config:sanmill_training_checkout"
    )
    assert sources["sentinel"] == "cli:no_sentinel"
    assert Path(args.malom) == (tmp_path / "local-malom").resolve()
    assert Path(args.sanmill_runtime) == (
        tmp_path / "local-sanmill-runtime"
    ).resolve()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_games", 0, "positive integer"),
        ("lr", float("nan"), "finite"),
        ("gamma_td", 1.1, "between zero and one"),
        ("self_play_ratio", -0.1, "between zero and one"),
        ("time_budget", 0.0, "-1 or a positive"),
        ("heuristic_node_budget", 0, "positive integer"),
        ("batch_games", 2, "must not exceed max_games"),
    ],
)
def test_configuration_validation_rejects_invalid_values(
    tmp_path: Path, field: str, value, message: str
) -> None:
    args = _smoke_args(tmp_path)
    setattr(args, field, value)

    with pytest.raises(PreflightConfigurationError, match=message):
        validate_generalist_configuration(args)


def test_configuration_rejects_unproven_parallel_rollouts(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    args.max_games = 2
    args.batch_games = 2

    with pytest.raises(PreflightConfigurationError, match="shared rollout state"):
        validate_generalist_configuration(args)


def test_configuration_rejects_mixed_heuristic_work_budgets(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.heuristic_node_budget = 25_000
    args.time_budget = 1.0

    with pytest.raises(PreflightConfigurationError, match="mutually exclusive"):
        validate_generalist_configuration(args)


def _enable_sanmill_contract(args, runtime: Path) -> None:
    args.referee_engine = "sanmill"
    args.opponent_engine = "sanmill"
    args.sanmill_runtime = str(runtime)
    args.diff_max = 1
    args.curriculum_advance_policy = "disabled"
    args.sanmill_node_ladder = (1_000,)
    args.sanmill_stage_games = None
    args.sanmill_search_depth = None
    args.minimal_rollouts = True
    args.no_recovery = True


def test_configuration_accepts_fixed_sanmill_resource_schedule(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    _enable_sanmill_contract(args, tmp_path / "runtime")
    args.max_games = 5_000
    args.diff_max = 5
    args.curriculum_advance_policy = "fixed-resource"
    args.sanmill_node_ladder = (1_000, 5_000, 25_000, 100_000, 500_000)
    args.sanmill_stage_games = (500, 500, 500, 1_000, 2_500)

    validate_generalist_configuration(args)


@pytest.mark.parametrize(
    ("stage_games", "message"),
    [
        (None, "requires sanmill_stage_games"),
        ((500, 500), "exactly diff_max"),
        ((500, 500, 500, 1_000, 2_499), "must sum to max_games"),
    ],
)
def test_configuration_rejects_invalid_sanmill_resource_schedule(
    tmp_path: Path,
    stage_games,
    message: str,
) -> None:
    args = _smoke_args(tmp_path)
    _enable_sanmill_contract(args, tmp_path / "runtime")
    args.max_games = 5_000
    args.diff_max = 5
    args.curriculum_advance_policy = "fixed-resource"
    args.sanmill_node_ladder = (1_000, 5_000, 25_000, 100_000, 500_000)
    args.sanmill_stage_games = stage_games

    with pytest.raises(PreflightConfigurationError, match=message):
        validate_generalist_configuration(args)


def test_configuration_rejects_resource_schedule_on_local_route(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.curriculum_advance_policy = "fixed-resource"
    args.sanmill_stage_games = (1,)

    with pytest.raises(PreflightConfigurationError, match="Sanmill engine pair"):
        validate_generalist_configuration(args)


def test_configuration_requires_paired_fail_closed_sanmill_controls(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.referee_engine = "sanmill"

    with pytest.raises(PreflightConfigurationError, match="paired"):
        validate_generalist_configuration(args)

    _enable_sanmill_contract(args, tmp_path / "runtime")
    args.minimal_rollouts = False
    with pytest.raises(PreflightConfigurationError, match="minimal_rollouts"):
        validate_generalist_configuration(args)

    args.minimal_rollouts = True
    args.no_recovery = False
    with pytest.raises(PreflightConfigurationError, match="no_recovery"):
        validate_generalist_configuration(args)


def test_sanmill_smoke_preflight_binds_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.experiment_id = "dev-v4-sanmill-refereed-fresh-v1"
    _enable_sanmill_contract(args, tmp_path / "runtime")
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)
    runtime_identity = "b" * 64

    monkeypatch.setattr(
        preflight_module,
        "probe_sanmill_training_runtime",
        lambda *_args, **_kwargs: {
            "identity": runtime_identity,
            "commit": "a" * 40,
            "binary_sha256": "c" * 64,
            "strict_referee": {
                "semanticDigest": "sha256:" + "d" * 64,
            },
            "probe": {"deterministic": True},
        },
    )

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={"sanmill_runtime": "cli"},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_smoke"
    assert report["checks"]["sanmill_training"]["identity"] == runtime_identity
    assert report["checks"]["components"]["referee_engine"] == "sanmill"
    assert report["experimentDigest"]


def test_sanmill_resource_schedule_smoke_is_bounded_and_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.experiment_id = "dev-v4-sanmill-refereed-fresh-v1"
    _enable_sanmill_contract(args, tmp_path / "runtime")
    args.max_games = 5
    args.segment_games = 5
    args.self_play_ratio = 0.0
    args.diff_max = 5
    args.curriculum_advance_policy = "fixed-resource"
    args.sanmill_node_ladder = (1_000, 5_000, 25_000, 100_000, 500_000)
    args.sanmill_stage_games = (1, 1, 1, 1, 1)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)

    monkeypatch.setattr(
        preflight_module,
        "probe_sanmill_training_runtime",
        lambda *_args, **_kwargs: {
            "identity": "b" * 64,
            "commit": "a" * 40,
            "binary_sha256": "c" * 64,
            "strict_referee": {"semanticDigest": "sha256:" + "d" * 64},
            "probe": {"deterministic": True},
        },
    )

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={"sanmill_runtime": "cli"},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_smoke"
    assert report["errors"] == []
    assert report["resolved_config"]["sanmill_stage_games"] == [1, 1, 1, 1, 1]


def test_segment_boundary_does_not_change_resume_semantics(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    args.max_games = 2
    args.segment_games = 1
    segmented = resume_config_sha256(args)
    args.segment_games = 2

    assert resume_config_sha256(args) == segmented


def test_imitation_mix_control_changes_resume_semantics(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    disabled = resume_config_sha256(args)
    args.no_imitation_mix = False

    assert resume_config_sha256(args) != disabled


def test_fixed_auxiliary_defaults_preserve_legacy_resume_config_shape(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)

    config = resolved_resume_config(args)

    assert "malom_policy_aux_mode" not in config
    assert "malom_policy_aux_target_ratio" not in config
    assert "malom_policy_aux_coef_cap" not in config
    assert "malom_policy_aux_denominator_floor" not in config


def test_normalized_auxiliary_settings_bind_resume_semantics(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    fixed = resume_config_sha256(args)
    args.malom_policy_aux_mode = "policy-head-normalized"
    args.mill_bonus_mode = "malom-preserving-only"

    config = resolved_resume_config(args)

    assert config["malom_policy_aux_mode"] == "policy-head-normalized"
    assert config["malom_policy_aux_target_ratio"] == 0.25
    assert config["malom_policy_aux_coef_cap"] == 0.25
    assert config["malom_policy_aux_denominator_floor"] == 1e-12
    assert resume_config_sha256(args) != fixed


def test_learning_rate_mode_preserves_default_and_binds_fixed_semantics(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    adaptive = resume_config_sha256(args)

    assert "lr_adaptation_mode" not in resolved_resume_config(args)

    args.lr_adaptation_mode = "fixed"

    assert resolved_resume_config(args)["lr_adaptation_mode"] == "fixed"
    assert resume_config_sha256(args) != adaptive


def test_main_rejects_duplicate_cli_options_before_training(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        trainer.main(["--max-games", "1", "--max-games=2"])

    assert raised.value.code == 2
    assert "specified more than once" in capsys.readouterr().err


def test_smoke_preflight_is_read_only_and_ready_for_corrected_baseline(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)
    before = {
        path: path.stat().st_mtime_ns
        for path in (Path(args.human_db), Path(args.specialist_db))
    }

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={"out_dir": "cli"},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_smoke"
    assert report["errors"] == []
    assert report["checks"]["specialist_db"]["trust"] == "trusted"
    assert report["checks"]["specialist_db"]["identity"]
    assert report["checks"]["malom"]["identity"]
    assert report["checks"]["human_db"]["identity"]
    assert report["checks"]["human_db"]["malom_columns_policy"] == (
        "masked_historical_labels"
    )
    assert not Path(args.out_dir).exists()
    assert before == {
        path: path.stat().st_mtime_ns
        for path in (Path(args.human_db), Path(args.specialist_db))
    }


def test_fresh_preflight_does_not_create_specialist_db_sidecars(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    database = SpecialistDB(args.specialist_db)
    database.close()
    wal = Path(f"{args.specialist_db}-wal")
    shm = Path(f"{args.specialist_db}-shm")
    assert not wal.exists()
    assert not shm.exists()

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={"out_dir": "cli"},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_smoke"
    assert not wal.exists()
    assert not shm.exists()


def test_fresh_preflight_rejects_existing_specialist_db_sidecars(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    database = SpecialistDB(args.specialist_db)
    database.close()
    Path(f"{args.specialist_db}-wal").touch()

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={"out_dir": "cli"},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "fatal_stop"
    assert any("SQLite sidecars" in error for error in report["errors"])


def test_smoke_preflight_rejects_existing_output_and_legacy_specialist_db(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    Path(args.out_dir).mkdir()
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), None)

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "fatal_stop"
    assert "fresh output path must not already exist" in report["errors"]
    assert any("trusted Malom label version" in error for error in report["errors"])


def test_fresh_preflight_rejects_reused_specialist_database(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)
    connection = sqlite3.connect(args.specialist_db)
    connection.execute(
        "INSERT INTO positions(pos_hash, malom_label) VALUES ('old', NULL)"
    )
    connection.execute(
        "INSERT INTO meta(key, value) VALUES "
        "('training_lineage_root_run_id', 'old-run')"
    )
    connection.commit()
    connection.close()

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "fatal_stop"
    assert any("empty isolated database" in item for item in report["errors"])
    assert any("already bound" in item for item in report["errors"])


def test_smoke_preflight_rejects_malom_inventory_drift(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)
    (Path(args.malom) / "unexpected.sec2").write_bytes(b"")

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "fatal_stop"
    assert any("component inventory has changed" in item for item in report["errors"])


def test_weights_only_preflight_accepts_compatible_legacy_weights(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.start_mode = "weights-only"
    args.experiment_id = "weights-import-test"
    model = trainer.ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=args.policy_hidden,
    )
    source = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.get_config(),
            "stage": trainer.STAGE_TAG,
        },
        source,
    )
    args.resume = str(source)
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)

    report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        expected_move_feature_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        expected_value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_smoke"
    checkpoint = report["checks"]["checkpoint"]
    assert checkpoint["format"] == "legacy-pytorch-weights"
    assert checkpoint["checkpoint_id"].startswith("legacy-sha256:")

    args.experiment_id = "dev-v4-malom-corrected-fresh-v1"
    blocked = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        expected_move_feature_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        expected_value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert blocked["verdict"] == "fatal_stop"
    assert (
        "weights-only imports require an explicit non-fresh experiment ID"
        in blocked["errors"]
    )


def test_exact_resume_preflight_binds_semantics_and_specialist_db(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.experiment_id = "dev-v4-malom-corrected-fresh-v1"
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    specialist_path = Path(args.specialist_db)
    _write_specialist_db(specialist_path, CURRENT_MALOM_LABEL_VERSION)
    specialist_sha256 = _file_sha256(specialist_path)
    model = trainer.ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=args.policy_hidden,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    payload = CheckpointPayload(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state({"game": random.Random(args.seed).getstate()}),
        trainer_state={
            "game_count": 0,
            "batch_count": 0,
            "update_count": 0,
            "difficulty": trainer.DIFF_START,
            "temperature": args.temp_start,
            "rolling_metrics": {},
            "curriculum": {},
            "target_network": {},
            "recovery_state": {},
            "model_config": model.get_config(),
        },
        data_state={
            "cursor": {"completed_games": 0},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {
                "specialist_db": {"sha256": specialist_sha256}
            },
        },
    )
    fresh_report = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        expected_move_feature_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        expected_value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )
    assert fresh_report["verdict"] == "ready_for_smoke"
    descriptor = CheckpointDescriptor(
        checkpoint_id="source:checkpoint:00000001",
        run_id="source",
        experiment_id=args.experiment_id,
        parent_checkpoint_id=None,
        role="latest",
        save_reason="test",
        created_at_utc="2026-07-20T11:00:00Z",
        config_sha256=resume_config_sha256(args),
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        label_schema_version=trainer.LABEL_SCHEMA_VERSION,
        database_schema_versions={"specialist_db": trainer.LABEL_SCHEMA_VERSION},
        asset_identities={"specialist_db": specialist_sha256},
        implementation={
            "trainer": trainer.STAGE_TAG,
            "framework": "pytorch",
            "experiment_digest": fresh_report["experimentDigest"],
            "mif_suite_tag": fresh_report["mifSuite"]["tag"],
            "mif_release_commit": fresh_report["mifSuite"]["releaseCommit"],
            "mif_suite_jcs_sha256": fresh_report["mifSuite"]["suiteJcsSha256"],
            "ruleset_semantic_digest": fresh_report["ruleset"]["semanticDigest"],
        },
    )
    source = tmp_path / "source.pt"
    save_checkpoint(source, descriptor, payload)
    args.start_mode = "exact-resume"
    args.resume = str(source)

    compatible = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        expected_move_feature_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        expected_value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert compatible["verdict"] == "ready_for_smoke"
    assert compatible["errors"] == []

    connection = sqlite3.connect(specialist_path)
    connection.execute("INSERT INTO winning_lines(id) VALUES (1)")
    connection.commit()
    connection.close()
    changed = run_generalist_preflight(
        args,
        mode="smoke",
        root=tmp_path,
        path_sources={},
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        expected_move_feature_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        expected_value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert changed["verdict"] == "fatal_stop"
    assert "checkpoint: SpecialistDB content identity has changed" in changed["errors"]


def test_long_run_preflight_remains_needs_decision(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    args.preflight = "long-run"
    args.max_games = 5_000
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)

    report = run_generalist_preflight(
        args,
        mode="long-run",
        root=tmp_path,
        path_sources={},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "needs_decision"
    assert report["errors"] == []
    assert report["unresolved_decisions"]


def test_authorized_managed_long_run_is_ready(tmp_path: Path) -> None:
    args = _smoke_args(tmp_path)
    args.preflight = None
    args.launch = "long-run"
    args.run_id = "managed-v4-test-segment-0001"
    args.experiment_id = "dev-v4-managed-baseline-v1"
    args.max_games = 5_000
    args.segment_games = 250
    args.segment_stop_game = 250
    args.heuristic_node_budget = 500_000
    control_dir = tmp_path / "control"
    args.out_dir = str(control_dir / "segments" / "segment-0001")
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    args.paths_config = str(paths_config)
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    args.managed_plan = str(plan_path)
    args.managed_authorization = str(authorization_path)
    plan = ManagedPlan(
        plan_id="managed-v4-test",
        created_at_utc="2026-07-20T12:00:00Z",
        objective="corrected-v4-single-GPU-baseline",
        experiment_id=args.experiment_id,
        git_commit="a" * 40,
        control_dir=str(control_dir.resolve()),
        paths_config=str(paths_config.resolve()),
        paths_config_sha256=hashlib.sha256(paths_config.read_bytes()).hexdigest(),
        resume_config_sha256=resume_config_sha256(args),
        max_games=args.max_games,
        segment_games=args.segment_games,
        max_wall_hours=12.0,
        common_trainer_args=("--max-games", "5000"),
        allow_safe_exact_resume=True,
        publication_allowed=False,
        promotion_allowed=False,
    )
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved for the bounded local run.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    _write_malom(Path(args.malom))
    _write_human_db(Path(args.human_db))
    _write_specialist_db(Path(args.specialist_db), CURRENT_MALOM_LABEL_VERSION)

    report = run_generalist_preflight(
        args,
        mode="long-run",
        root=tmp_path,
        path_sources={},
        git_state=GitState(commit="a" * 40, dirty=False, diff_sha256=None),
    )

    assert report["verdict"] == "ready_for_long_run"
    assert report["errors"] == []
    assert report["unresolved_decisions"] == []
