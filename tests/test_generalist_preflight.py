"""Tests for strict, read-only Generalist v2 preflight checks."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import pytest
import torch

from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.training.generalist_preflight import (
    GitState,
    PreflightConfigurationError,
    _file_sha256,
    configure_generalist_paths,
    load_training_settings,
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
from scripts import train_s_gen_v2 as trainer


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
            "--human-db",
            str(tmp_path / "human.sqlite"),
            "--specialist-db",
            str(tmp_path / "specialist.sqlite"),
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--no-s1a-warmstart",
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
        json.dumps({"malom_db_path": "local-malom"}), encoding="utf-8"
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
    assert sources["sentinel"] == "cli:no_sentinel"
    assert Path(args.malom) == (tmp_path / "local-malom").resolve()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_games", 0, "positive integer"),
        ("lr", float("nan"), "finite"),
        ("gamma_td", 1.1, "between zero and one"),
        ("self_play_ratio", -0.1, "between zero and one"),
        ("time_budget", 0.0, "-1 or a positive"),
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


def test_exact_resume_preflight_binds_semantics_and_specialist_db(
    tmp_path: Path,
) -> None:
    args = _smoke_args(tmp_path)
    args.experiment_id = "exact-resume-test"
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
        implementation={"trainer": trainer.STAGE_TAG, "framework": "pytorch"},
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

    assert compatible["verdict"] == "needs_decision"
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
