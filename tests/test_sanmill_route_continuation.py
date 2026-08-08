"""Focused tests for the bounded Sanmill route continuation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation import sanmill_route_continuation as continuation
from learned_ai.validation import sanmill_route_probe as probe


_ROOT = Path(__file__).resolve().parents[1]
_TRACKED_PLAN = (
    _ROOT
    / "docs/experiments/sanmill-no-update-integrated-route-continuation-v1.json"
)


def _payload() -> dict:
    parent = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    payload = {
        "schema_version": continuation.CONTINUATION_PLAN_SCHEMA,
        "status": "prepared_unlaunched",
        "experiment_id": "continuation-test",
        "claim_boundary": "test only",
        "parent_plan": {
            "path": probe.DEFAULT_PLAN_RELATIVE.as_posix(),
            "raw_sha256": parent.raw_sha256,
            "plan_identity": parent.identity,
        },
        "schedule_range": {
            "start_scheduled_index": 7,
            "end_scheduled_index_exclusive": 36,
        },
        "bounded_work": {
            "complete_games": 29,
            "search_opponent_games": 23,
            "frozen_target_games": 6,
            "maximum_logical_plies": 3480,
            "maximum_search_calls": 1380,
            "maximum_requested_search_node_ceilings": 226_500_000,
        },
        "decision_rules": {
            "diagnosis_only": True,
            "execution_requires_explicit_authority": True,
            "no_automatic_escalation": True,
            "no_retry": True,
            "preserve_parent_schedule_identity": True,
            "publish_success_or_failure_atomically": True,
            "refuse_output_overwrite": True,
            "training_launch": False,
        },
    }
    payload["plan_identity"] = canonical_sha256(payload)
    return payload


def _write(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tracked_continuation_plan_binds_remaining_parent_schedule() -> None:
    loaded = continuation.load_probe_continuation_plan(_TRACKED_PLAN)
    effective = continuation.continuation_probe_plan(loaded)

    assert loaded.identity == (
        "807fcae96ee03634d5abb61b9982fcfaf364b07ab1c139142ad8cc1cffdadb08"
    )
    assert loaded.raw_sha256 == (
        "b10ba116f43468567f053ef965d72371cc570ea999bbd00234c5aa284ad6ad75"
    )
    assert [game.scheduled_index for game in effective.schedule] == list(
        range(7, 36)
    )
    assert effective.payload["bounded_work"] == _payload()["bounded_work"]


def test_continuation_preserves_parent_indices_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "continuation.json"
    _write(_payload(), path)

    loaded = continuation.load_probe_continuation_plan(path)
    effective = continuation.continuation_probe_plan(loaded)

    assert len(loaded.schedule) == 29
    assert loaded.schedule[0] == loaded.parent.schedule[7]
    assert loaded.schedule[-1] == loaded.parent.schedule[35]
    assert [game.scheduled_index for game in effective.schedule] == list(
        range(7, 36)
    )
    assert effective.node_budgets == (5_000, 25_000, 100_000, 500_000)
    assert effective.payload["bounded_work"] == _payload()["bounded_work"]


def test_continuation_rejects_invalid_or_empty_range(tmp_path: Path) -> None:
    payload = _payload()
    payload["schedule_range"]["start_scheduled_index"] = 36
    body = dict(payload)
    body.pop("plan_identity")
    payload["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "continuation.json"
    _write(payload, path)

    with pytest.raises(probe.SanmillRouteProbeError, match="range is invalid"):
        continuation.load_probe_continuation_plan(path)


def test_continuation_rejects_incorrect_bounded_work(tmp_path: Path) -> None:
    payload = _payload()
    payload["bounded_work"]["maximum_search_calls"] -= 1
    body = dict(payload)
    body.pop("plan_identity")
    payload["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "continuation.json"
    _write(payload, path)

    with pytest.raises(probe.SanmillRouteProbeError, match="bounded work drifted"):
        continuation.load_probe_continuation_plan(path)


def test_continuation_preflight_never_authorizes_launch(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "continuation.json"
    _write(_payload(), path)
    parent = probe.load_probe_plan(_ROOT / probe.DEFAULT_PLAN_RELATIVE)
    monkeypatch.setattr(
        continuation,
        "preflight_probe",
        lambda *_args, **_kwargs: {
            "schema_version": probe.PREFLIGHT_SCHEMA,
            "status": "ready_for_authorized_probe",
            "launch_authorized": False,
            "plan": {"identity": parent.identity},
            "bounded_work": dict(parent.payload["bounded_work"]),
        },
    )
    monkeypatch.setattr(
        continuation,
        "tracked_plan_record",
        lambda plan: {"identity": plan.identity},
    )

    report = continuation.preflight_probe_continuation(
        path,
        _ROOT / probe.DEFAULT_PATHS_RELATIVE,
    )

    assert report["status"] == "ready_for_authorized_continuation_probe"
    assert report["launch_authorized"] is False
    assert report["schedule_range"] == {
        "start_scheduled_index": 7,
        "end_scheduled_index_exclusive": 36,
    }
    assert report["bounded_work"]["complete_games"] == 29
