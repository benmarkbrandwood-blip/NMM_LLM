#!/usr/bin/env python3
"""Create the one-time direct product-owner authorization record."""

# ruff: noqa: E402

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import write_sealed_json
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    AUTHORIZATION_SCHEMA,
    load_plan,
    sha256_file,
)


PLAN_PATH = _ROOT / "docs/experiments/sanmill-trained-model-baseline-v1.json"
AUTH_PATH = (
    _ROOT
    / "docs/experiments/sanmill-trained-model-baseline-v1/authorization.json"
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def main() -> int:
    if AUTH_PATH.exists():
        raise SystemExit("authorization already exists")
    if _git("branch", "--show-current") != "dev":
        raise SystemExit("authorization requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        raise SystemExit("tracked worktree must be clean before authorization")
    plan, plan_sha = load_plan(PLAN_PATH)
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized_once_measurement_unconsumed",
        "operator": "product-owner-direct",
        "authorized_on": "2026-08-16",
        "grant_count": 1,
        "authorization_basis": (
            "Direct product-owner authorization in the task titled '以零训练的 "
            "Malom 安全随机为基线，测量已训练模型的实际得分'."
        ),
        "plan": {
            "path": str(PLAN_PATH.relative_to(_ROOT)).replace("\\", "/"),
            "identity": plan["plan_identity"],
            "file_sha256": plan_sha,
        },
        "start_pool": {
            "path": plan["start_pool"]["path"],
            "identity": plan["start_pool"]["pool_identity"],
            "file_sha256": plan["start_pool"]["pool_file_sha256"],
            "formal_starts": plan["experiment"]["starts"],
            "formal_membership_identity": plan["start_pool"][
                "formal_membership_identity"
            ],
            "excluded_start_ids": plan["start_pool"]["excluded_start_ids"],
        },
        "baseline": plan["baseline"],
        "candidate_bindings": {
            "retained_v4_bundle_identity": plan["candidate_runtime"][
                "retained_v4"
            ]["bundle"]["identity"],
            "retained_v4_checkpoint_sha256": plan["candidate_runtime"][
                "retained_v4"
            ]["checkpoint"]["sha256"],
            "retained_v4_checkpoint_payload_sha256": plan["candidate_runtime"][
                "retained_v4"
            ]["checkpoint"]["payload_sha256"],
            "retained_v4_specialist_db_identity": plan["candidate_runtime"][
                "retained_v4"
            ]["specialist_db"]["identity"],
            "active_specialist_runtime_identity": plan["candidate_runtime"][
                "active_specialists"
            ]["runtime_identity"],
        },
        "execution_scope": plan["resource_envelope"],
        "permitted": [
            "one ten-game non-evidence technical rehearsal",
            "one zero-formal-game preflight with bounded deterministic canaries",
            "one exact 2032-game formal execution after every gate passes",
            "read-only policy-model loading",
            "read-only sector-corrected-v1 Malom, HumanDB, SpecialistDB, and Sanmill use",
        ],
        "prohibited": [
            "automatic retry, resume, recovery, batching around a limit, or extension",
            "a second formal execution",
            "training, fitting, tuning, or weight updates",
            "checkpoint edits, copies, renames, or alias changes",
            "database writes or rebuilds",
            "promotion, deployment, publication, release, or new training authorization",
        ],
        "consumption_rule": (
            "consumed when the fresh measurement-started marker is durably created "
            "before the first formal game"
        ),
        "host_interruption": (
            "no recovery authorized; any missing-suffix continuation requires a new "
            "direct product-owner authorization"
        ),
        "rehearsal_and_preflight_do_not_consume_formal_execution": True,
        "source_commit_at_authorization": _git("rev-parse", "HEAD"),
        "source_tree_at_authorization": _git("rev-parse", "HEAD^{tree}"),
        "implementation_files": {
            path: sha256_file(_ROOT / path)
            for path in (
                "learned_ai/evaluation/training_aligned_policy.py",
                "learned_ai/evaluation/sanmill_trained_model_baseline.py",
                "scripts/rehearse_sanmill_trained_model_baseline.py",
                "scripts/preflight_sanmill_trained_model_baseline.py",
                "scripts/run_sanmill_trained_model_baseline.py",
            )
        },
        "claim_boundary": plan["claim_boundary"],
    }
    sealed = write_sealed_json(
        AUTH_PATH,
        payload,
        identity_field="authorization_identity",
    )
    print(sealed["authorization_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
