"""Focused tests for fail-closed mill-bonus ablation preparation."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.validation import mill_bonus_ablation_readiness as readiness_module
from learned_ai.validation.mill_bonus_ablation_readiness import (
    DEFAULT_CONTRACT,
    MillBonusAblationReadinessError,
    _normalised_pair_semantics,
    assert_preparation_outputs_ignored,
    assert_preparation_targets_absent,
    build_prepare_commands,
    inspect_template,
    inspect_preparation_evidence,
    load_ablation_contract,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / DEFAULT_CONTRACT
PATHS_CONFIG = ROOT / "data/training_paths.local.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_probe(
    root: Path,
    *,
    summary: dict[str, object],
    module_path: Path,
    script_path: Path,
    commit: str = "a" * 40,
    tree: str = "b" * 40,
) -> tuple[dict[str, object], dict[str, object]]:
    spec = {
        "path": "out/probe.json",
        "schema_version": "nmm.malom-downgrade-penalty-no-update-probe.v1",
        "source_probe_identity": "c" * 64,
        "source_probe_sha256": "d" * 64,
        "module_path": module_path.relative_to(root).as_posix(),
        "script_path": script_path.relative_to(root).as_posix(),
        "expected_summary": summary,
    }
    per_state = [
        {
            "ordinal": 1,
            "mills_formed": 1,
            "malom_quality": -1.0,
            "phase": "move",
            "stratum": "book",
            "rewards": {
                "control": {"total": 0.0},
                "treatment": {"total": -0.25},
            },
        }
    ]
    body = {
        "schema_version": spec["schema_version"],
        "source_probe": {
            "probe_identity": spec["source_probe_identity"],
            "sha256": spec["source_probe_sha256"],
        },
        "auditor": {
            "implementation_commit": commit,
            "implementation_tree": tree,
            "module_sha256": _sha256(module_path),
            "script_sha256": _sha256(script_path),
            "tracked_worktree_clean": True,
        },
        "summary": summary,
        "per_state": per_state,
        "claim_boundary": {
            "candidate_policy_loaded": False,
            "new_games": False,
            "optimizer_created": False,
            "weights_updated": False,
            "actions_changed_between_modes": False,
            "states_changed_between_modes": False,
            "reward_component_only": True,
            "causal_training_effect_proven": False,
        },
    }
    probe = {
        **body,
        "probe_identity": readiness_module.canonical_sha256(body),
    }
    path = root / spec["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(readiness_module.canonical_json_bytes(probe))
    contract = {
        "preparation_evidence": {
            "downgrade_penalty_no_update_probe": spec
        }
    }
    return contract, probe


def test_required_probe_binds_source_summary_and_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "module.py"
    script_path = tmp_path / "script.py"
    module_path.write_text("module\n", encoding="utf-8")
    script_path.write_text("script\n", encoding="utf-8")
    summary = {
        "states": 1,
        "affected_states": 1,
        "mill_forming_states": 1,
        "non_mill_states": 0,
        "quality_rank_counts": {"1": 1},
        "phase_counts": {"move": 1},
        "stratum_counts": {"book": 1},
        "control_reward_total": 0.0,
        "treatment_reward_total": -0.25,
        "treatment_minus_control": -0.25,
    }
    contract, probe = _write_probe(
        tmp_path,
        summary=summary,
        module_path=module_path,
        script_path=script_path,
    )

    monkeypatch.setattr(
        readiness_module,
        "_git_output",
        lambda _root, *args: "b" * 40
        if args == ("rev-parse", f"{'a' * 40}^{{tree}}")
        else "",
    )
    monkeypatch.setattr(
        readiness_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    result = inspect_preparation_evidence(
        tmp_path, contract, source={"head": "e" * 40}
    )

    assert result is not None
    assert result["probe_identity"] == probe["probe_identity"]
    assert result["summary"] == summary


def test_required_probe_rejects_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "module.py"
    script_path = tmp_path / "script.py"
    module_path.write_text("module\n", encoding="utf-8")
    script_path.write_text("script\n", encoding="utf-8")
    contract, _probe = _write_probe(
        tmp_path,
        summary={
            "states": 1,
            "affected_states": 1,
            "mill_forming_states": 1,
            "non_mill_states": 0,
            "quality_rank_counts": {"1": 1},
            "phase_counts": {"move": 1},
            "stratum_counts": {"book": 1},
            "control_reward_total": 0.0,
            "treatment_reward_total": -0.25,
            "treatment_minus_control": -0.25,
        },
        module_path=module_path,
        script_path=script_path,
    )
    contract["preparation_evidence"][
        "downgrade_penalty_no_update_probe"
    ]["expected_summary"] = {"states": 2}
    monkeypatch.setattr(
        readiness_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    with pytest.raises(
        MillBonusAblationReadinessError,
        match="summary differs",
    ):
        inspect_preparation_evidence(
            tmp_path, contract, source={"head": "e" * 40}
        )


def test_required_probe_rejects_unpublished_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "module.py"
    script_path = tmp_path / "script.py"
    module_path.write_text("module\n", encoding="utf-8")
    script_path.write_text("script\n", encoding="utf-8")
    contract, _probe = _write_probe(
        tmp_path,
        summary={
            "states": 1,
            "affected_states": 1,
            "mill_forming_states": 1,
            "non_mill_states": 0,
            "quality_rank_counts": {"1": 1},
            "phase_counts": {"move": 1},
            "stratum_counts": {"book": 1},
            "control_reward_total": 0.0,
            "treatment_reward_total": -0.25,
            "treatment_minus_control": -0.25,
        },
        module_path=module_path,
        script_path=script_path,
    )

    def run(command, *_args, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if command[1:3] == ["merge-base", "--is-ancestor"] else 0,
        )

    monkeypatch.setattr(readiness_module.subprocess, "run", run)

    with pytest.raises(
        MillBonusAblationReadinessError,
        match="not in the published lineage",
    ):
        inspect_preparation_evidence(
            tmp_path, contract, source={"head": "e" * 40}
        )


def test_prepare_commands_encode_all_six_unlaunched_arms() -> None:
    contract = load_ablation_contract(CONTRACT_PATH)
    ordered_arms = sorted(
        contract["arms"], key=lambda arm: int(arm["launch_order"])
    )
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=PATHS_CONFIG,
        python_executable="python-under-test",
    )

    assert len(commands) == 6
    semantics_by_seed: dict[int, set[str]] = {}
    for arm, command in zip(ordered_arms, commands, strict=True):
        assert command[:3] == [
            "python-under-test",
            str(ROOT / "scripts/manage_generalist_run.py"),
            "prepare",
        ]
        assert "authorize" not in command
        assert "run-next" not in command
        args = manager._build_parser().parse_args(command[2:])
        assert args.plan_id == arm["plan_id"]
        assert args.experiment_id == arm["experiment_id"]
        assert args.seed == arm["seed"]
        assert args.mill_bonus_mode == arm["mill_bonus_mode"]
        assert args.max_games == 5000
        assert args.completion_game_bound == 500
        assert args.segment_games == 500
        assert args.max_wall_hours == 1.0
        assert args.max_ply == 120
        assert args.engine_profile == "sanmill-fixed-resource"
        assert args.policy_health_gate
        assert args.policy_health_device == "cuda"
        common_args = manager._common_trainer_args(args, PATHS_CONFIG)
        parsed = trainer._build_argument_parser().parse_args(
            ["--preflight", "long-run", *common_args]
        )
        assert parsed.referee_engine == "sanmill"
        assert parsed.opponent_engine == "sanmill"
        assert parsed.curriculum_advance_policy == "fixed-resource"
        assert parsed.sanmill_node_ladder == (
            1000,
            5000,
            25000,
            100000,
            500000,
        )
        assert parsed.sanmill_stage_games == (500, 500, 500, 1000, 2500)
        assert parsed.self_play_ratio == 0.60
        assert parsed.no_recovery
        assert parsed.no_sentinel
        assert parsed.no_value_net
        assert parsed.no_gap_net
        assert parsed.no_s1a_warmstart
        assert parsed.no_imitation_mix
        assert parsed.no_s1b_refresher
        assert parsed.no_opening_forcing
        assert not parsed.ppo
        semantics_by_seed.setdefault(arm["seed"], set()).add(
            _normalised_pair_semantics(parsed)
        )
    assert all(len(identities) == 1 for identities in semantics_by_seed.values())


def test_template_audit_is_immutable_and_rejects_sidecars(tmp_path: Path) -> None:
    database_path = tmp_path / "template.sqlite"
    database = SpecialistDB(str(database_path))
    database.close()
    expected = {
        "path": "template.sqlite",
        "byte_length": database_path.stat().st_size,
        "sha256": _sha256(database_path),
        "quick_check": "ok",
        "label_version": "sector-corrected-v1",
        "positions": 0,
        "winning_lines": 0,
        "preferred_plays": 0,
    }
    contract = {"data_contract": {"specialist_db_initial_template": expected}}

    result = inspect_template(tmp_path, contract)

    assert result["sha256"] == expected["sha256"]
    assert result["counts"] == {
        "positions": 0,
        "winning_lines": 0,
        "preferred_plays": 0,
    }
    assert not Path(f"{database_path}-wal").exists()
    assert not Path(f"{database_path}-shm").exists()

    Path(f"{database_path}-wal").touch()
    with pytest.raises(MillBonusAblationReadinessError, match="sidecars"):
        inspect_template(tmp_path, contract)


def test_preparation_refuses_any_existing_target(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_ablation_contract(CONTRACT_PATH))
    for index, arm in enumerate(contract["arms"]):
        arm["control_dir"] = f"out/arm-{index}"
        arm["specialist_db"] = f"data/arm-{index}.sqlite"
    existing = tmp_path / contract["arms"][2]["control_dir"]
    existing.mkdir(parents=True)

    with pytest.raises(
        MillBonusAblationReadinessError,
        match="preparation targets already exist",
    ):
        assert_preparation_targets_absent(
            tmp_path,
            contract,
            report_path=tmp_path / "out/readiness.json",
        )


def test_failed_command_preserves_stdout_diagnostic(tmp_path: Path) -> None:
    diagnostic = '{"verdict":"fatal_stop","errors":["wrong output"]}'

    def failed_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python", "trainer.py"],
            returncode=2,
            stdout=diagnostic,
            stderr="",
        )

    with pytest.raises(
        MillBonusAblationReadinessError,
        match="fatal_stop",
    ):
        readiness_module._run_checked(
            ["python", "trainer.py"],
            root=tmp_path,
            runner=failed_runner,
        )


def test_expected_decision_exit_can_be_inspected(tmp_path: Path) -> None:
    diagnostic = '{"verdict":"needs_decision"}'

    def decision_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python", "trainer.py"],
            returncode=2,
            stdout=diagnostic,
            stderr="",
        )

    result = readiness_module._run_checked(
        ["python", "trainer.py"],
        root=tmp_path,
        runner=decision_runner,
        accepted_return_codes=(2,),
    )

    assert result.stdout == diagnostic


def test_fresh_preflight_targets_first_managed_segment(tmp_path: Path) -> None:
    plan = SimpleNamespace(
        common_trainer_args=["--max-games", "5000"],
        control_dir=str(tmp_path / "control"),
        game_bound=500,
        plan_id="six-arm-seed-42-legacy",
        segment_games=500,
    )

    command = readiness_module._build_fresh_preflight_command(
        plan,
        root=tmp_path,
        python_executable="python-under-test",
    )
    parsed = trainer._build_argument_parser().parse_args(command[2:])

    assert command[:2] == [
        "python-under-test",
        str(tmp_path / "scripts/train_s_gen_v2.py"),
    ]
    assert parsed.preflight == "long-run"
    assert parsed.launch is None
    assert parsed.start_mode == "fresh"
    assert parsed.run_id == "six-arm-seed-42-legacy-segment-0001"
    assert parsed.out_dir == str(tmp_path / "control/segments/segment-0001")
    assert parsed.segment_games == 500
    assert parsed.segment_stop_game == 500
    assert parsed.managed_plan is None
    assert parsed.managed_authorization is None


def test_unlaunched_preflight_has_only_the_product_gate_open(
    tmp_path: Path,
) -> None:
    plan = SimpleNamespace(
        control_dir=str(tmp_path / "control"),
        experiment_id="experiment",
        game_bound=500,
        plan_id="six-arm-seed-42-legacy",
        resume_config_sha256="5" * 64,
        segment_games=500,
    )
    source_commit = "a" * 40
    preflight = {
        "schema_version": "nmm.generalist-preflight.v1",
        "mode": "long-run",
        "verdict": "needs_decision",
        "errors": [],
        "unresolved_decisions": [
            readiness_module.PRODUCT_AUTHORIZATION_DECISION
        ],
        "resume_config_sha256": plan.resume_config_sha256,
        "git": {
            "commit": source_commit,
            "dirty": False,
        },
        "resolved_config": {
            "experiment_id": plan.experiment_id,
            "run_id": f"{plan.plan_id}-segment-0001",
            "out_dir": str(tmp_path / "control/segments/segment-0001"),
            "segment_games": 500,
            "segment_stop_game": 500,
            "start_mode": "fresh",
        },
        "checks": {
            "output": {
                "exists": False,
                "isolated": True,
                "kind": "run_directory",
            }
        },
    }

    readiness_module._validate_unlaunched_preflight(
        preflight,
        plan=plan,
        source_commit=source_commit,
        arm_id="s42-legacy",
    )

    preflight["unresolved_decisions"].append("another decision")
    with pytest.raises(
        MillBonusAblationReadinessError,
        match="decisions differ",
    ):
        readiness_module._validate_unlaunched_preflight(
            preflight,
            plan=plan,
            source_commit=source_commit,
            arm_id="s42-legacy",
        )


def test_real_preparation_outputs_are_all_git_ignored() -> None:
    contract = load_ablation_contract(CONTRACT_PATH)

    assert_preparation_outputs_ignored(
        ROOT,
        contract,
        report_path=ROOT / "out/mill-bonus-ablation-smoke-v1/readiness.json",
    )


def test_contract_loader_rejects_semantic_tampering(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["common_training_contract"]["max_logical_plies"] = 119
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        MillBonusAblationReadinessError,
        match="plan identity differs",
    ):
        load_ablation_contract(path)
