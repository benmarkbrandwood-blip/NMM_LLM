#!/usr/bin/env python3
"""Prepare the immutable no-update target-refresh direct cross-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.data.data_contract import load_dataset_manifest  # noqa: E402
from learned_ai.evaluation.phase_corpus import validate_phase_corpus  # noqa: E402
from learned_ai.evaluation.phase_replay_development_corpus import (  # noqa: E402
    validate_phase_replay_development_corpus,
    validate_phase_replay_sanmill_audit,
)
from learned_ai.evaluation.target_refresh_direct_crossplay import (  # noqa: E402
    READINESS_SCHEMA,
    DirectCrossplayError,
    build_direct_crossplay_schedule,
    load_direct_crossplay_plan,
)
from learned_ai.training.generalist_preflight import _probe_human_db  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import SanmillTrainingGame  # noqa: E402
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)
from scripts.report_target_refresh_equal_transition_diagnostic import (  # noqa: E402
    _arm_by_cell,
    _load_candidate_pair,
    _load_fork,
    _prefix_by_seed,
    _strict_json,
)  # noqa: E402
from scripts.report_target_refresh_schedule_isolation_diagnostic import (  # noqa: E402
    _read_only_observations,
)  # noqa: E402
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    load_equal_transition_contract,
)


DEFAULT_PLAN = ROOT / (
    "docs/experiments/sanmill-target-refresh-direct-crossplay-v1.json"
)
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"
DEFAULT_OUTPUT = ROOT / "out/target-refresh-direct-crossplay-v1/readiness.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_direct_crossplay_human_db(path: Path) -> dict[str, Any]:
    """Probe the same immutable HumanDB main-file view used at runtime."""
    return _probe_human_db(path, immutable=True)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise DirectCrossplayError("evidence path is outside the repository") from exc


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DirectCrossplayError("could not inspect Git evidence") from exc


def _resolve_setting(settings: dict[str, Any], key: str) -> Path:
    value = Path(str(settings[key]))
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _validate_source(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    if _git("status", "--short", "--untracked-files=all"):
        raise DirectCrossplayError("readiness requires a clean tracked worktree")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_dev = _git("rev-parse", "origin/dev")
    if branch != "dev" or head != origin_dev:
        raise DirectCrossplayError("readiness requires published dev source")
    implementation_commit = plan["implementation"]["commit"]
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, head],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DirectCrossplayError("implementation commit is not an ancestor") from exc
    changed = sorted(
        path
        for path in _git(
            "diff", "--name-only", f"{implementation_commit}..{head}", "--"
        ).splitlines()
        if path
    )
    allowed = {
        _relative(plan_path),
        _relative(plan_path.with_suffix(".md")),
    }
    if set(changed) - allowed:
        raise DirectCrossplayError(
            "post-implementation tracked paths are not contract-only"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "implementation_commit": implementation_commit,
        "post_implementation_paths": changed,
        "tracked_clean": True,
        "published": True,
    }


def _validate_implementation(plan: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for prefix in ("module", "prepare", "runner"):
        path = ROOT / plan["implementation"][f"{prefix}_path"]
        if not path.is_file():
            raise DirectCrossplayError(f"{prefix} implementation is absent")
        sha256 = _sha256(path)
        if sha256 != plan["implementation"][f"{prefix}_sha256"]:
            raise DirectCrossplayError(f"{prefix} implementation identity differs")
        observed[prefix] = {"path": _relative(path), "sha256": sha256}
    return observed


def _result_boundary_record(result: dict[str, Any], seed: int) -> dict[str, Any]:
    records = result["policy_distribution"]["by_seed"][str(seed)]["boundaries"]
    matches = [
        item for item in records if item["post_fork_consumed_transitions"] == 8192
    ]
    if len(matches) != 1:
        raise DirectCrossplayError("source result final boundary differs")
    return matches[0]


def _validate_checkpoints(
    plan: dict[str, Any],
    schedule_contract: dict[str, Any],
    source_result: dict[str, Any],
) -> dict[str, Any]:
    anchors = {item["seed"]: item for item in plan["checkpoint_contract"]["anchors"]}
    candidates = {
        (item["seed"], item["condition"]): item
        for item in plan["checkpoint_contract"]["candidates"]
    }
    prefixes = _prefix_by_seed(schedule_contract)
    arms = _arm_by_cell(schedule_contract)
    observed: dict[str, Any] = {}
    device = torch.device("cpu")
    for seed in plan["measurement_contract"]["seeds"]:
        _, fork_record = _load_fork(prefixes[seed])
        expected_anchor = anchors[seed]
        expected_anchor_record = {
            key: value for key, value in expected_anchor.items() if key != "seed"
        }
        if fork_record != expected_anchor_record:
            raise DirectCrossplayError("anchor checkpoint identity differs")
        anchor_path = ROOT / expected_anchor["path"]
        if _sha256(anchor_path) != expected_anchor["file_sha256"]:
            raise DirectCrossplayError("anchor checkpoint bytes differ")
        models, records = _load_candidate_pair(
            arms,
            seed=seed,
            boundary=8192,
            fork_record=fork_record,
            device=device,
        )
        if set(models) != {"refresh", "no-refresh"}:
            raise DirectCrossplayError("candidate model cells differ")
        boundary = _result_boundary_record(source_result, seed)
        for condition in ("refresh-once", "no-refresh"):
            expected = candidates[(seed, condition)]
            expected_record = {
                key: value
                for key, value in expected.items()
                if key not in {"seed", "condition"}
            }
            record = records[condition]
            source_record = boundary["checkpoints"][condition]
            if record != expected_record or source_record != expected_record:
                raise DirectCrossplayError("candidate checkpoint identity differs")
            candidate_path = ROOT / expected["path"]
            if _sha256(candidate_path) != expected["file_sha256"]:
                raise DirectCrossplayError("candidate checkpoint bytes differ")
        observed[str(seed)] = {
            "anchor": expected_anchor,
            "candidates": {
                condition: candidates[(seed, condition)]
                for condition in ("refresh-once", "no-refresh")
            },
            "loaded_on_cpu": True,
        }
    return observed


def build_readiness(
    *,
    plan_path: Path,
    paths_config_path: Path,
    malom_manifest_path: Path,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = load_direct_crossplay_plan(plan_path)
    source = _validate_source(plan, plan_path)
    implementation = _validate_implementation(plan)

    source_result_path = ROOT / plan["source"]["source_result_path"]
    schedule_contract_path = ROOT / plan["source"]["schedule_contract_path"]
    for path, expected, label in (
        (
            source_result_path,
            plan["source"]["source_result_sha256"],
            "source result",
        ),
        (
            schedule_contract_path,
            plan["source"]["schedule_contract_sha256"],
            "schedule contract",
        ),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise DirectCrossplayError(f"{label} identity differs")
    source_result = _strict_json(source_result_path)
    if source_result.get("result_identity") != plan["source"][
        "source_result_identity"
    ]:
        raise DirectCrossplayError("source result semantic identity differs")
    if (
        source_result.get("decision", {}).get("classification")
        != "no_material_paired_outcome_effect"
        or source_result.get("policy_distribution", {})
        .get("decision", {})
        .get("classification")
        != "inconclusive_late_onset"
    ):
        raise DirectCrossplayError("source result does not require this successor")
    schedule_contract = load_equal_transition_contract(schedule_contract_path)
    if schedule_contract["plan_identity"] != plan["source"][
        "schedule_plan_identity"
    ]:
        raise DirectCrossplayError("schedule contract plan identity differs")

    data = plan["data_contract"]
    policy_corpus_path = ROOT / data["policy_corpus_path"]
    replay_corpus_path = ROOT / data["replay_corpus_path"]
    replay_audit_path = ROOT / data["replay_audit_path"]
    for path, expected, label in (
        (policy_corpus_path, data["policy_corpus_sha256"], "policy corpus"),
        (replay_corpus_path, data["replay_corpus_sha256"], "replay corpus"),
        (replay_audit_path, data["replay_audit_sha256"], "replay audit"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise DirectCrossplayError(f"{label} identity differs")
    policy_corpus = _strict_json(policy_corpus_path)
    replay_corpus = _strict_json(replay_corpus_path)
    replay_audit = _strict_json(replay_audit_path)
    validate_phase_corpus(policy_corpus)
    validate_phase_replay_development_corpus(replay_corpus)
    validate_phase_replay_sanmill_audit(replay_audit, corpus=replay_corpus)
    if (
        replay_corpus["corpus_identity"] != data["replay_corpus_identity"]
        or replay_audit["audit_identity"] != data["replay_audit_identity"]
    ):
        raise DirectCrossplayError("replay semantic identity differs")

    settings = _strict_json(paths_config_path.resolve())
    human_path = _resolve_setting(settings, "human_db_path")
    malom_path = _resolve_setting(settings, "malom_db_path")
    human_report = _probe_direct_crossplay_human_db(human_path)
    if (
        human_report.get("error")
        or human_report.get("identity") != data["human_db_identity"]
        or human_report.get("malom_columns_policy")
        != data["human_db_malom_policy"]
    ):
        raise DirectCrossplayError("HumanDB identity or trust policy differs")
    manifest = load_dataset_manifest(malom_manifest_path.resolve())
    if manifest.manifest_sha256 != data["malom_manifest_identity"]:
        raise DirectCrossplayError("Malom manifest identity differs")
    std_anchor = next(
        (
            item
            for item in manifest.components
            if item.relative_path == "std.secval"
        ),
        None,
    )
    if std_anchor is None or _sha256(malom_path / "std.secval") != std_anchor.sha256:
        raise DirectCrossplayError("Malom tablebase identity differs")

    installation = load_local_installation(paths_config_path.resolve())
    before = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    with SanmillTrainingGame(installation, seed=42) as game:
        if game.state.terminal or game.state.logical_ply_count != 0:
            raise DirectCrossplayError("strict Sanmill referee canary differs")
        identity = game.state.strict_referee_identity
        if identity is None:
            raise DirectCrossplayError("strict Sanmill referee identity is absent")
        referee = {
            "rules_identity_sha256": game.state.rules_identity_sha256,
            "strict_referee_semantic_digest": identity.semantic_digest,
            "initial_history_sha256": game.state.history_sha256,
        }
    checkpoints = _validate_checkpoints(plan, schedule_contract, source_result)
    after = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    if before != after:
        raise DirectCrossplayError("read-only source observations changed")

    schedule = build_direct_crossplay_schedule(plan)
    readiness_core = {
        "schema_version": READINESS_SCHEMA,
        "state": "ready_for_product_authorization",
        "launch_authorized": False,
        "claim_boundary": plan["claim_boundary"],
        "plan": {
            "path": _relative(plan_path),
            "sha256": _sha256(plan_path),
            "plan_identity": plan["plan_identity"],
        },
        "source": source,
        "implementation": implementation,
        "source_result": {
            "path": _relative(source_result_path),
            "sha256": _sha256(source_result_path),
            "result_identity": source_result["result_identity"],
            "outcome_classification": source_result["decision"]["classification"],
            "policy_classification": source_result["policy_distribution"][
                "decision"
            ]["classification"],
        },
        "data": {
            "human_db_identity": human_report["identity"],
            "human_db_malom_policy": human_report["malom_columns_policy"],
            "malom_manifest_identity": manifest.manifest_sha256,
            "replay_corpus_identity": replay_corpus["corpus_identity"],
            "replay_audit_identity": replay_audit["audit_identity"],
            "read_only_observations": {"before": before, "after": after},
        },
        "referee": referee,
        "checkpoints": checkpoints,
        "schedule": {
            "identity": canonical_sha256(schedule),
            "pairs": len(schedule) // 2,
            "games": len(schedule),
            "seeds": plan["measurement_contract"]["seeds"],
            "record_indices": plan["measurement_contract"]["record_indices"],
            "replicates_per_start": plan["measurement_contract"][
                "replicates_per_start"
            ],
        },
        "resource_envelope": plan["resource_envelope"],
        "prohibited_operations": plan["prohibited_operations"],
    }
    readiness = {
        **readiness_core,
        "readiness_identity": canonical_sha256(readiness_core),
    }
    return readiness


def validate_readiness(
    readiness: dict[str, Any],
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise DirectCrossplayError("direct cross-play readiness schema differs")
    identity = readiness.get("readiness_identity")
    if not isinstance(identity, str) or identity != canonical_sha256(
        {key: readiness[key] for key in readiness if key != "readiness_identity"}
    ):
        raise DirectCrossplayError("direct cross-play readiness identity differs")
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
        or readiness.get("plan", {}).get("plan_identity")
        != plan["plan_identity"]
        or readiness.get("resource_envelope") != plan["resource_envelope"]
    ):
        raise DirectCrossplayError("direct cross-play readiness content differs")
    return readiness


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"readiness output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary readiness output exists: {temporary}")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_direct_crossplay_plan(args.plan.resolve())
    expected_output = (ROOT / plan["output_contract"]["readiness"]).resolve()
    if args.output.resolve() != expected_output:
        raise DirectCrossplayError("readiness output differs from the frozen plan")
    readiness = build_readiness(
        plan_path=args.plan,
        paths_config_path=args.paths_config,
        malom_manifest_path=args.malom_manifest,
    )
    _write_exclusive(args.output.resolve(), readiness)
    print(json.dumps(readiness, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
