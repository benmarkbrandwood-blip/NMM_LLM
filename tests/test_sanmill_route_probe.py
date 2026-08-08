"""Focused tests for the no-update integrated Sanmill route probe."""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from learned_ai.training.run_contract import canonical_sha256
from ai.human_db import HumanDB
from learned_ai.training.sanmill_referee import (
    SanmillBoardMirrorError,
    SanmillTrainingGame,
    SanmillTrainingOpponent,
)
from learned_ai.validation import sanmill_route_probe as probe
from scripts import probe_sanmill_integrated_route as probe_runner
from scripts import train_s_gen_v2 as trainer


_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_probe_plan_freezes_the_complete_bounded_matrix() -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)

    assert plan.identity == (
        "7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb"
    )
    assert plan.node_budgets == (1_000, 5_000, 25_000, 100_000, 500_000)
    assert len(plan.schedule) == 36
    assert sum(game.opponent_kind == "sanmill" for game in plan.schedule) == 30
    assert sum(game.opponent_kind == "frozen_target" for game in plan.schedule) == 6
    assert plan.payload["bounded_work"] == {
        "complete_games": 36,
        "search_opponent_games": 30,
        "frozen_target_games": 6,
        "maximum_logical_plies": 4_320,
        "maximum_search_calls": 1_800,
        "maximum_requested_search_node_ceilings": 227_160_000,
    }


def test_probe_plan_identity_rejects_a_changed_budget(tmp_path: Path) -> None:
    payload = json.loads(
        (_ROOT / probe.DEFAULT_PLAN_RELATIVE).read_text(encoding="utf-8")
    )
    payload["schedule"][0]["node_budget"] = 999
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(probe.SanmillRouteProbeError, match="identity mismatch"):
        probe.load_probe_plan(changed)


def test_local_read_only_snapshots_match_the_frozen_plan() -> None:
    paths = _ROOT / probe.DEFAULT_PATHS_RELATIVE
    if not paths.is_file():
        pytest.skip("requires ignored local probe snapshot paths")
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    inputs = probe.resolve_probe_inputs(paths)

    record = probe.verify_probe_inputs(
        plan,
        inputs,
        verify_malom_components=False,
    )

    assert record["human_db"]["sidecars"] == []
    assert record["specialist_db"]["sidecars"] == []
    assert record["specialist_db"]["metadata"]["malom_label_version"] == (
        "sector-corrected-v1"
    )


def test_schedule_defaults_are_the_production_route_seams() -> None:
    signature = inspect.signature(probe.execute_probe_schedule)

    assert signature.parameters["rollout_fn"].default is trainer._rollout
    assert signature.parameters["game_factory"].default is SanmillTrainingGame
    assert signature.parameters["opponent_factory"].default is SanmillTrainingOpponent


class _FakeGame:
    def __init__(self, _installation, *, timing_observer, **_kwargs) -> None:
        self.observer = timing_observer
        self.state = SimpleNamespace(
            fen="fake",
            history_sha256="a" * 64,
            logical_ply_count=2,
            terminal=False,
            winner=None,
            outcome_reason_code="ongoing",
        )

    def __enter__(self):
        self.observer("sanmill_process_startup", 0.001)
        return self

    def __exit__(self, *_exc) -> None:
        return None


def test_schedule_uses_no_update_controls_and_completes_exactly(
    monkeypatch,
) -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    sink = probe._TimingSink()
    runtime = probe.ProbeRuntime(
        device=torch.device("cpu"),
        model=object(),
        frozen_opponent=object(),
        lookahead_advisor=object(),
        human_db=object(),
        specialist_db=object(),
        malom_db=object(),
        human_route=object(),
        specialist_route=object(),
        malom_route=object(),
        installation=object(),
        timing_sink=sink,
    )
    calls: list[dict] = []
    forbidden: list[str] = []

    monkeypatch.setattr(
        torch.Tensor,
        "backward",
        lambda *_args, **_kwargs: forbidden.append("backward"),
    )
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda *_args, **_kwargs: forbidden.append("checkpoint"),
    )

    def fake_opponent(_game, *, node_budget, depth):
        return SimpleNamespace(node_budget=node_budget, depth=depth)

    def fake_rollout(**kwargs):
        calls.append(kwargs)
        kwargs["timing_observer"]("learner_encode", 0.002)
        budget = getattr(kwargs["opponent"], "node_budget", None)
        observations = (
            [{"nodes": min(17, budget), "depth": 3}] if budget is not None else []
        )
        return SimpleNamespace(
            trajectory=[],
            outcome=trainer.DRAW_LONG,
            termination_reason="max-ply-truncation",
            ply=2,
            phase_ply_counts={"place": 2},
            compound_turn_count=0,
            opponent_search_observations=observations,
            opponent_search_calls=len(observations),
            opponent_search_nodes=sum(item["nodes"] for item in observations),
            opponent_search_depth_sum=sum(item["depth"] for item in observations),
        )

    samples = probe.execute_probe_schedule(
        plan,
        runtime,
        game_factory=_FakeGame,
        opponent_factory=fake_opponent,
        rollout_fn=fake_rollout,
    )

    assert len(samples) == len(calls) == 36
    assert [sample["game_id"] for sample in samples] == [
        game.game_id for game in plan.schedule
    ]
    assert all(call["persist_rollout_evidence"] is False for call in calls)
    assert all(call["record_branches"] is False for call in calls)
    assert all(call["retry_ply"] == 0 for call in calls)
    assert all(call["specialist_db"] is runtime.specialist_route for call in calls)
    assert all(call["malom_db"] is runtime.malom_route for call in calls)
    assert forbidden == []


def test_schedule_failure_preserves_prefix_and_bridge_context() -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    sink = probe._TimingSink()
    runtime = probe.ProbeRuntime(
        device=torch.device("cpu"),
        model=object(),
        frozen_opponent=object(),
        lookahead_advisor=object(),
        human_db=object(),
        specialist_db=object(),
        malom_db=object(),
        human_route=object(),
        specialist_route=object(),
        malom_route=object(),
        installation=object(),
        timing_sink=sink,
    )
    call_count = 0
    bridge_diagnostic = {
        "schema_version": "nmm.sanmill-board-mirror-diagnostic.v1",
        "local_board_fen": "local",
        "projected_board_fen": "projected",
    }

    def fake_opponent(_game, *, node_budget, depth):
        return SimpleNamespace(node_budget=node_budget, depth=depth)

    def fake_rollout(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise SanmillBoardMirrorError(bridge_diagnostic)
        return SimpleNamespace(
            trajectory=[],
            outcome=trainer.DRAW_LONG,
            termination_reason="max-ply-truncation",
            ply=2,
            phase_ply_counts={"place": 2},
            compound_turn_count=0,
            opponent_search_observations=[{"nodes": 17, "depth": 3}],
            opponent_search_calls=1,
            opponent_search_nodes=17,
            opponent_search_depth_sum=3,
        )

    with pytest.raises(probe.SanmillRouteProbeScheduleError) as raised:
        probe.execute_probe_schedule(
            plan,
            runtime,
            game_factory=_FakeGame,
            opponent_factory=fake_opponent,
            rollout_fn=fake_rollout,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic["schema_version"] == probe.SCHEDULE_FAILURE_SCHEMA
    assert diagnostic["completed_sample_count"] == 1
    assert diagnostic["completed_samples"][0]["game_id"] == (
        plan.schedule[0].game_id
    )
    assert len(diagnostic["completed_samples"][0]["sample_identity"]) == 64
    assert diagnostic["failed_schedule"] == probe._probe_game_record(
        plan.schedule[1]
    )
    assert diagnostic["exception"]["bridge_diagnostic"] == bridge_diagnostic
    assert diagnostic["exception"]["message"] == (
        "Sanmill and NMM board mirrors diverged"
    )


def test_run_failure_captures_after_identities_and_stays_incomplete(
    monkeypatch,
) -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    learner = object()
    frozen_model = object()
    runtime = SimpleNamespace(
        model=learner,
        frozen_opponent=SimpleNamespace(_model=frozen_model),
        device=torch.device("cpu"),
        installation=object(),
        close=lambda: None,
    )
    data = {"human_db": {"sha256": "a" * 64}}
    source = {"branch": "dev", "commit": "b" * 40}
    schedule_failure = probe.SanmillRouteProbeScheduleError(
        {
            "schema_version": probe.SCHEDULE_FAILURE_SCHEMA,
            "failed_schedule": probe._probe_game_record(plan.schedule[0]),
            "completed_sample_count": 0,
            "completed_samples": [],
            "exception": {
                "type": "example.Error",
                "message": "failed closed",
            },
        }
    )
    monkeypatch.setattr(probe, "verify_probe_inputs", lambda *_args: data)
    monkeypatch.setattr(probe, "_build_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        probe,
        "model_state_sha256",
        lambda model: "c" * 64 if model is learner else "d" * 64,
    )
    monkeypatch.setattr(probe, "_host_record", lambda *_args: {"host": "fixed"})
    monkeypatch.setattr(
        probe,
        "execute_probe_schedule",
        lambda *_args: (_ for _ in ()).throw(schedule_failure),
    )
    monkeypatch.setattr(
        probe,
        "inspect_published_source",
        lambda **_kwargs: dict(source),
    )
    monkeypatch.setattr(
        probe,
        "training_installation_record",
        lambda *_args, **_kwargs: {"identity": "e" * 64},
    )

    with pytest.raises(probe.SanmillRouteProbeRunFailure) as raised:
        probe.run_probe(
            plan,
            object(),
            source=source,
            run_id="diagnostic-test",
            invocation=["python", "probe"],
        )

    report = raised.value.report
    assert report["schema_version"] == probe.FAILURE_SCHEMA
    assert report["status"] == "failed_closed"
    assert report["failure"] == schedule_failure.diagnostic
    assert report["data_unchanged"] is True
    assert report["source_unchanged"] is True
    assert report["model"]["learner_unchanged"] is True
    assert report["model"]["frozen_unchanged"] is True
    assert "samples" not in report
    body = dict(report)
    identity = body.pop("report_identity")
    assert identity == canonical_sha256(body)


def test_timed_proxy_delegates_to_the_same_read_only_object() -> None:
    class Target:
        marker = "unchanged"

        def query(self, value):
            return value + 1

    sink = probe._TimingSink()
    collector = probe._TimingCollector()
    sink.current = collector
    target = Target()
    proxy = probe._TimedReadOnlyProxy(target, sink, {"query": "db_query"})

    assert proxy.target is target
    assert proxy.marker == "unchanged"
    assert proxy.query(4) == 5
    assert len(collector.samples["db_query"]) == 1


def test_immutable_human_snapshot_open_creates_no_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "human.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE positions (state_key TEXT PRIMARY KEY);
        CREATE TABLE moves (state_key TEXT, notation TEXT, total INTEGER);
        INSERT INTO meta(key, value) VALUES ('total_games', '0');
        """
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()

    database = HumanDB(path, read_only=True, immutable=True)
    assert database.is_available()
    database.close()

    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()


def _completed_report(plan: probe.ProbePlan) -> dict:
    body = {
        "schema_version": probe.RESULT_SCHEMA,
        "status": "completed_no_update_measurement",
        "samples": [{"game_id": game.game_id} for game in plan.schedule],
    }
    return {**body, "report_identity": canonical_sha256(body)}


def _failed_report(plan: probe.ProbePlan) -> dict:
    body = {
        "schema_version": probe.FAILURE_SCHEMA,
        "status": "failed_closed",
        "plan": {"identity": plan.identity},
        "failure": {
            "schema_version": probe.SCHEDULE_FAILURE_SCHEMA,
            "failed_schedule": probe._probe_game_record(plan.schedule[0]),
            "completed_sample_count": 0,
            "completed_samples": [],
            "exception": {
                "type": "example.Error",
                "message": "failed closed",
            },
        },
    }
    return {**body, "report_identity": canonical_sha256(body)}


def test_atomic_publisher_refuses_incomplete_or_existing_evidence(
    tmp_path: Path,
) -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    incomplete = _completed_report(plan)
    incomplete["samples"] = incomplete["samples"][:-1]
    body = dict(incomplete)
    body.pop("report_identity")
    incomplete["report_identity"] = canonical_sha256(body)

    with pytest.raises(probe.SanmillRouteProbeError, match="incomplete"):
        probe.publish_probe_result(tmp_path / "bad.json", incomplete, plan)

    target = tmp_path / "result.json"
    report = _completed_report(plan)
    probe.publish_probe_result(target, report, plan)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        probe.publish_probe_result(target, report, plan)


def test_failure_publisher_is_atomic_distinct_and_no_overwrite(
    tmp_path: Path,
) -> None:
    plan = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    completed = tmp_path / "probe.json"
    target = probe.probe_failure_output(completed)
    report = _failed_report(plan)

    probe.publish_probe_failure(target, report, plan)

    assert not completed.exists()
    assert target.name == "probe.failure.json"
    assert json.loads(target.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        probe.publish_probe_failure(target, report, plan)

    invalid = dict(report)
    invalid["status"] = "completed_no_update_measurement"
    with pytest.raises(probe.SanmillRouteProbeError, match="status"):
        probe.publish_probe_failure(tmp_path / "invalid.json", invalid, plan)


def test_runner_quarantines_failure_without_publishing_result(
    monkeypatch,
) -> None:
    plan_path = _ROOT / probe.DEFAULT_PLAN_RELATIVE
    plan = probe.load_probe_plan(plan_path)
    output = _ROOT / "out/diagnostics/test-route-probe-failure.json"
    report = _failed_report(plan)
    report["run_id"] = "diagnostic-test"
    report["failure"]["completed_sample_count"] = 0
    published: dict[str, object] = {}
    monkeypatch.setattr(
        probe_runner,
        "parse_args",
        lambda: SimpleNamespace(
            preflight=False,
            launch="probe",
            plan=plan_path,
            paths_config=_ROOT / probe.DEFAULT_PATHS_RELATIVE,
            run_id="diagnostic-test",
            output=output,
        ),
    )
    monkeypatch.setattr(probe_runner, "load_probe_plan", lambda _path: plan)
    monkeypatch.setattr(probe_runner, "tracked_plan_record", lambda _plan: {})
    monkeypatch.setattr(
        probe_runner,
        "inspect_published_source",
        lambda **_kwargs: {"commit": "a" * 40},
    )
    monkeypatch.setattr(probe_runner, "resolve_probe_inputs", lambda _path: object())
    monkeypatch.setattr(
        probe_runner,
        "validate_probe_output",
        lambda path: Path(path).resolve(strict=False),
    )
    monkeypatch.setattr(
        probe_runner,
        "run_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            probe.SanmillRouteProbeRunFailure(report)
        ),
    )
    monkeypatch.setattr(
        probe_runner,
        "publish_probe_failure",
        lambda path, observed, observed_plan: published.update(
            {"path": path, "report": observed, "plan": observed_plan}
        ),
    )
    monkeypatch.setattr(
        probe_runner,
        "publish_probe_result",
        lambda *_args: pytest.fail("completed result must not be published"),
    )

    assert probe_runner.main() == 1
    assert published == {
        "path": probe.probe_failure_output(output),
        "report": report,
        "plan": plan,
    }


def test_output_is_confined_and_cannot_be_reused(tmp_path: Path) -> None:
    with pytest.raises(probe.SanmillRouteProbeError, match="out/diagnostics"):
        probe.validate_probe_output(tmp_path / "outside.json")

    target = _ROOT / "out/diagnostics/test-route-probe-output.json"
    assert probe.validate_probe_output(target) == target.resolve(strict=False)


def test_probe_does_not_import_or_copy_the_reference_runner() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")

    assert "reference.mif1" not in source
    assert "NMM_Std" not in source
    assert "torch.optim" not in source
    assert "save_checkpoint(" not in source
    assert "scaffolded_a2c_update" not in source
