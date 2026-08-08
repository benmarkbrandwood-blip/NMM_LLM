"""Focused tests for the one-entry Sanmill route diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation import sanmill_route_probe as probe
from scripts import diagnose_sanmill_integrated_route as diagnostic_runner


_ROOT = Path(__file__).resolve().parents[1]


def _write_reidentified(payload: dict, target: Path) -> None:
    body = dict(payload)
    body.pop("plan_identity", None)
    payload["plan_identity"] = canonical_sha256(body)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_tracked_diagnostic_selects_only_parent_index_zero() -> None:
    diagnostic = probe.load_probe_diagnostic_plan(
        _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    )
    effective = probe.diagnostic_probe_plan(diagnostic)

    assert diagnostic.identity == (
        "5554489e3278dca88cc4f816e97ced1bdf17e7a89b0e4c02991c808d7087e4b0"
    )
    assert diagnostic.parent.identity == (
        "7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb"
    )
    assert diagnostic.selected == diagnostic.parent.schedule[0]
    assert effective.schedule == (diagnostic.parent.schedule[0],)
    assert effective.node_budgets == (1_000,)
    assert effective.payload["bounded_work"] == {
        "complete_games": 1,
        "search_opponent_games": 1,
        "frozen_target_games": 0,
        "maximum_logical_plies": 120,
        "maximum_search_calls": 60,
        "maximum_requested_search_node_ceilings": 60_000,
    }


def test_diagnostic_rejects_a_different_parent_schedule_entry(
    tmp_path: Path,
) -> None:
    path = _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_schedule_entry"] = json.loads(
        (_ROOT / probe.DEFAULT_PLAN_RELATIVE).read_text(encoding="utf-8")
    )["schedule"][1]
    changed = tmp_path / "changed.json"
    _write_reidentified(payload, changed)

    with pytest.raises(
        probe.SanmillRouteProbeError,
        match="preserve parent schedule index zero exactly",
    ):
        probe.load_probe_diagnostic_plan(changed)


def test_diagnostic_rejects_parent_byte_identity_drift(tmp_path: Path) -> None:
    path = _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parent_plan"]["raw_sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    _write_reidentified(payload, changed)

    with pytest.raises(probe.SanmillRouteProbeError, match="parent bytes drifted"):
        probe.load_probe_diagnostic_plan(changed)


def test_diagnostic_preflight_keeps_launch_unauthorized(monkeypatch) -> None:
    diagnostic = probe.load_probe_diagnostic_plan(
        _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    )
    parent_report = {
        "schema_version": probe.PREFLIGHT_SCHEMA,
        "status": "ready_for_authorized_probe",
        "launch_authorized": False,
        "plan": {"identity": diagnostic.parent.identity},
        "source": {"published": True},
        "bounded_work": dict(diagnostic.parent.payload["bounded_work"]),
        "next_gate": "old",
    }
    monkeypatch.setattr(probe, "preflight_probe", lambda *_args, **_kwargs: parent_report)
    monkeypatch.setattr(
        probe,
        "tracked_plan_record",
        lambda plan: {"identity": plan.identity, "raw_sha256": plan.raw_sha256},
    )

    report = probe.preflight_probe_diagnostic(
        diagnostic.path,
        _ROOT / probe.DEFAULT_PATHS_RELATIVE,
    )

    assert report["status"] == "ready_for_authorized_minimal_diagnostic"
    assert report["launch_authorized"] is False
    assert report["plan"]["identity"] == diagnostic.identity
    assert report["parent_probe_plan"] == parent_report["plan"]
    assert report["selected_schedule_entry"]["scheduled_index"] == 0
    assert report["bounded_work"]["complete_games"] == 1


def test_diagnostic_report_reframes_success_without_strength_claim(
    monkeypatch,
) -> None:
    diagnostic = probe.load_probe_diagnostic_plan(
        _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    )
    source_report = {
        "schema_version": probe.RESULT_SCHEMA,
        "status": "completed_no_update_measurement",
        "samples": [{"game_id": diagnostic.selected.game_id}],
        "interpretation": {"strength_measured": True},
        "report_identity": "old",
    }
    observed: dict[str, probe.ProbePlan] = {}

    def complete(plan, *_args, **_kwargs):
        observed["plan"] = plan
        return source_report

    monkeypatch.setattr(probe, "run_probe", complete)
    monkeypatch.setattr(
        probe,
        "tracked_plan_record",
        lambda plan: {"identity": plan.identity},
    )

    report = probe.run_probe_diagnostic(
        diagnostic,
        SimpleNamespace(),
        source={"commit": "a" * 40},
        run_id="diagnostic-test",
        invocation=["python", "diagnose"],
    )

    assert report["diagnostic"]["outcome"] == (
        "selected_entry_completed_without_mirror_mismatch"
    )
    assert observed["plan"].schedule == (diagnostic.parent.schedule[0],)
    assert report["diagnostic"]["historical_failure_index_known"] is False
    assert report["interpretation"]["completed_measurement"] is False
    assert report["interpretation"]["strength_measured"] is False
    body = dict(report)
    identity = body.pop("report_identity")
    assert identity == canonical_sha256(body)


def test_diagnostic_report_preserves_structured_failure(monkeypatch) -> None:
    diagnostic = probe.load_probe_diagnostic_plan(
        _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    )
    source_report = {
        "schema_version": probe.FAILURE_SCHEMA,
        "status": "failed_closed",
        "failure": {
            "failed_schedule": probe._probe_game_record(diagnostic.selected),
            "exception": {"bridge_diagnostic": {"local_board_fen": "local"}},
        },
        "interpretation": {},
        "report_identity": "old",
    }

    def fail(*_args, **_kwargs):
        raise probe.SanmillRouteProbeRunFailure(source_report)

    monkeypatch.setattr(probe, "run_probe", fail)
    monkeypatch.setattr(
        probe,
        "tracked_plan_record",
        lambda plan: {"identity": plan.identity},
    )

    with pytest.raises(probe.SanmillRouteProbeRunFailure) as raised:
        probe.run_probe_diagnostic(
            diagnostic,
            SimpleNamespace(),
            source={"commit": "a" * 40},
            run_id="diagnostic-test",
            invocation=["python", "diagnose"],
        )

    report = raised.value.report
    assert report["failure"] == source_report["failure"]
    assert report["diagnostic"]["outcome"] == "failed_closed_with_diagnostic"
    assert report["interpretation"]["retry_authorized"] is False


def test_runner_preflight_cannot_supply_launch_authority(monkeypatch, capsys) -> None:
    plan_path = _ROOT / probe.DEFAULT_DIAGNOSTIC_PLAN_RELATIVE
    monkeypatch.setattr(
        diagnostic_runner,
        "parse_args",
        lambda: SimpleNamespace(
            preflight=True,
            launch=None,
            plan=plan_path,
            paths_config=_ROOT / probe.DEFAULT_PATHS_RELATIVE,
            run_id=None,
            output=None,
        ),
    )
    monkeypatch.setattr(
        diagnostic_runner,
        "preflight_probe_diagnostic",
        lambda *_args, **_kwargs: {
            "status": "ready_for_authorized_minimal_diagnostic",
            "launch_authorized": False,
        },
    )

    assert diagnostic_runner.main() == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["launch_authorized"] is False
