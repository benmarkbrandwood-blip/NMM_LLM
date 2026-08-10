"""Publish the read-only equal-transition target-refresh diagnostic result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.data.data_contract import load_dataset_manifest  # noqa: E402
from learned_ai.evaluation.common_anchor_policy_distribution import (  # noqa: E402
    DEFAULT_DIVERGENCE_THRESHOLDS,
    PRIMARY_TEMPERATURE,
)
from learned_ai.evaluation.phase_corpus import validate_phase_corpus  # noqa: E402
from learned_ai.evaluation.target_refresh_equal_transition_result import (  # noqa: E402
    EXPECTED_BOUNDARIES,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    RESULT_SCHEMA,
    classify_transition_policy_divergence,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.checkpoint_envelope import (  # noqa: E402
    CheckpointEnvelope,
    load_checkpoint,
)
from learned_ai.training.generalist_preflight import _probe_human_db  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.analyze_common_anchor_policy_distribution import (  # noqa: E402
    CommonAnchorAnalysisError,
    _build_feature_corpus,
    _compare_checkpoint_pair,
    _load_policy,
    _open_immutable_human_db,
    _read_only_observations,
    _state_dict_sha256,
)


CONTRACT_SCHEMA = "nmm.target-refresh-equal-transition-diagnostic-plan.v1"
READINESS_SCHEMA = "nmm.target-refresh-equal-transition-readiness.v1"
DEFAULT_CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-equal-transition-diagnostic-v1.json"
)
DEFAULT_READINESS = ROOT / (
    "out/target-refresh-equal-transition-diagnostic-v1/readiness.json"
)
DEFAULT_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"
DEFAULT_OUTPUT = ROOT / (
    "out/target-refresh-equal-transition-diagnostic-v1/result.json"
)
EXPECTED_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)


class EqualTransitionReportError(RuntimeError):
    """Raised when immutable equal-transition result evidence is incomplete."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EqualTransitionReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EqualTransitionReportError(
                    f"non-finite JSON value in {path.name}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EqualTransitionReportError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EqualTransitionReportError(f"JSON root is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise EqualTransitionReportError(
            "result evidence path is outside the repository"
        ) from exc


def _git_identity(expected_training_commit: str) -> dict[str, Any]:
    def output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True
        ).strip()

    status = output("status", "--porcelain", "--untracked-files=no")
    branch = output("branch", "--show-current")
    head = output("rev-parse", "HEAD")
    origin_dev = output("rev-parse", "origin/dev")
    training_is_ancestor = (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                expected_training_commit,
                head,
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if (
        status
        or branch != "dev"
        or head != origin_dev
        or not training_is_ancestor
    ):
        raise EqualTransitionReportError(
            "result publication requires a clean published analysis source "
            "descending from the exact training source"
        )
    return {
        "branch": branch,
        "head": expected_training_commit,
        "origin_dev": origin_dev,
        "tracked_clean": True,
        "published": True,
        "analysis_head": head,
        "analysis_origin_dev": origin_dev,
        "training_source_is_ancestor": True,
    }


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = _strict_json(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise EqualTransitionReportError("equal-transition contract schema differs")
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise EqualTransitionReportError("equal-transition plan identity differs")
    prefixes = contract.get("prefixes")
    arms = contract.get("arms")
    if not isinstance(prefixes, list) or len(prefixes) != len(EXPECTED_SEEDS):
        raise EqualTransitionReportError("shared-prefix cells differ")
    if not isinstance(arms, list) or len(arms) != (
        len(EXPECTED_SEEDS) * len(EXPECTED_CONDITIONS)
    ):
        raise EqualTransitionReportError("treatment-arm cells differ")
    if {int(prefix["seed"]) for prefix in prefixes} != set(EXPECTED_SEEDS):
        raise EqualTransitionReportError("shared-prefix seeds differ")
    observed_cells = {
        (int(arm["seed"]), str(arm["condition"])) for arm in arms
    }
    expected_cells = {
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    }
    if observed_cells != expected_cells:
        raise EqualTransitionReportError("treatment-arm cells differ")
    measurement = contract.get("measurement_contract", {})
    if (
        measurement.get("transition_boundaries")
        != list(EXPECTED_BOUNDARIES)
        or measurement.get("fixed_phase_corpus_sha256")
        != EXPECTED_CORPUS_SHA256
        or measurement.get("temperatures") != [1.0, PRIMARY_TEMPERATURE]
        or measurement.get("training_games") != 0
        or measurement.get("optimizer_updates") != 0
    ):
        raise EqualTransitionReportError("measurement contract differs")
    return contract


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise EqualTransitionReportError("readiness schema differs")
    if readiness.get("contract", {}).get("plan_identity") != contract[
        "plan_identity"
    ]:
        raise EqualTransitionReportError("readiness plan identity differs")
    source = readiness.get("source", {})
    if not source.get("published") or not source.get("tracked_clean"):
        raise EqualTransitionReportError("readiness source was not clean and published")
    return readiness


def _prefix_by_seed(contract: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(prefix["seed"]): prefix for prefix in contract["prefixes"]}


def _arm_by_cell(
    contract: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(arm["seed"]), str(arm["condition"])): arm
        for arm in contract["arms"]
    }


def _segment(control_dir: str) -> Path:
    path = ROOT / control_dir / "segments" / "segment-0001"
    if not path.is_dir():
        raise EqualTransitionReportError(f"completed segment is absent: {control_dir}")
    return path


def _load_fork(
    prefix: Mapping[str, Any],
) -> tuple[CheckpointEnvelope, dict[str, Any]]:
    path = _segment(str(prefix["control_dir"])) / "target-refresh-fork.pt"
    envelope = load_checkpoint(path, map_location="cpu")
    state = envelope.payload.trainer_state
    recovery = state.get("recovery_state", {})
    fork = recovery.get("target_refresh_fork_state", {})
    if (
        envelope.descriptor.role != "target_refresh_fork"
        or envelope.descriptor.experiment_id != prefix["experiment_id"]
        or state.get("game_count") != 50
        or fork.get("captured") is not True
        or fork.get("fork_game") != 50
        or fork.get("treatment") is not None
        or fork.get("post_fork_transition_origin") is not None
    ):
        raise EqualTransitionReportError(
            f"shared fork semantics differ: seed {prefix['seed']}"
        )
    return envelope, {
        "path": _relative(path),
        "file_sha256": _sha256_file(path),
        "checkpoint_id": envelope.descriptor.checkpoint_id,
        "payload_sha256": envelope.payload_sha256,
        "model_state_sha256": _state_dict_sha256(envelope.payload.model_state),
        "game_count": int(state["game_count"]),
        "update_count": int(state["update_count"]),
        "optimizer_consumed_transition_count": int(
            recovery["optimizer_consumed_transition_count"]
        ),
        "pending_transition_count": len(recovery["pending_steps"]),
    }


def _load_candidate_pair(
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    seed: int,
    boundary: int,
    fork_record: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    records: dict[str, Any] = {}
    model_config: dict[str, Any] | None = None
    immutable_assets: dict[str, Any] | None = None
    for condition in EXPECTED_CONDITIONS:
        arm = arms[(seed, condition)]
        branch_resume = (
            ROOT / str(arm["control_dir"]) / "initial-target-refresh-fork.pt"
        ).resolve()
        branch = load_checkpoint(branch_resume, map_location=device)
        branch_implementation = dict(branch.descriptor.implementation)
        expected_branch_id = f"{fork_record['checkpoint_id']}:branch:{condition}"
        if (
            branch.descriptor.role != "target_refresh_fork"
            or branch.descriptor.experiment_id != arm["experiment_id"]
            or branch.descriptor.checkpoint_id != expected_branch_id
            or branch.descriptor.parent_checkpoint_id != fork_record["checkpoint_id"]
            or branch.payload_sha256 != fork_record["payload_sha256"]
            or branch_implementation.get("target_refresh_branch_kind")
            != "target-refresh-fork-v1"
            or branch_implementation.get("target_refresh_branch_source_checkpoint_id")
            != fork_record["checkpoint_id"]
            or branch_implementation.get("target_refresh_branch_source_payload_sha256")
            != fork_record["payload_sha256"]
            or branch_implementation.get("target_refresh_branch_treatment") != condition
        ):
            raise EqualTransitionReportError(
                f"branch checkpoint semantics differ: seed {seed}, {condition}"
            )
        path = _segment(str(arm["control_dir"])) / (
            f"transition-{boundary:08d}.pt"
        )
        envelope = load_checkpoint(path, map_location=device)
        state = envelope.payload.trainer_state
        recovery = state.get("recovery_state", {})
        fork = recovery.get("target_refresh_fork_state", {})
        implementation = dict(envelope.descriptor.implementation)
        expected_runtime_implementation = {
            key: value
            for key, value in branch_implementation.items()
            if not key.startswith("target_refresh_branch_")
        }
        origin = fork.get("post_fork_transition_origin")
        consumed = recovery.get("optimizer_consumed_transition_count")
        if (
            envelope.descriptor.role != "transition_diagnostic_candidate"
            or envelope.descriptor.experiment_id != arm["experiment_id"]
            or envelope.descriptor.config_sha256 != branch.descriptor.config_sha256
            or implementation != expected_runtime_implementation
            or fork.get("captured") is not True
            or fork.get("fork_game") != 50
            or fork.get("treatment") != condition
            or not isinstance(origin, int)
            or not isinstance(consumed, int)
            or consumed - origin != boundary
            or len(recovery.get("pending_steps", [])) >= 64
        ):
            raise EqualTransitionReportError(
                f"candidate semantics differ: seed {seed}, {condition}, {boundary}"
            )
        if str(recovery.get("source_checkpoint")) != str(branch_resume):
            raise EqualTransitionReportError(
                f"candidate fork source differs: seed {seed}, {condition}"
            )
        current_config = dict(state["model_config"])
        if model_config is None:
            model_config = current_config
        elif current_config != model_config:
            raise EqualTransitionReportError(
                f"paired model configurations differ: seed {seed}"
            )
        assets = dict(envelope.descriptor.asset_identities)
        current_immutable = {
            key: assets[key]
            for key in (
                "human_db",
                "malom_tablebase",
                "mif_suite_1_0",
                "sanmill_training_runtime",
                "training_ruleset",
            )
        }
        if immutable_assets is None:
            immutable_assets = current_immutable
        elif current_immutable != immutable_assets:
            raise EqualTransitionReportError(
                f"paired immutable assets differ: seed {seed}"
            )
        model_key = "refresh" if condition == "refresh-once" else "no-refresh"
        models[model_key] = _load_policy(envelope, device=device)
        records[condition] = {
            "path": _relative(path),
            "file_sha256": _sha256_file(path),
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "model_state_sha256": _state_dict_sha256(
                envelope.payload.model_state
            ),
            "game_count": int(state["game_count"]),
            "update_count": int(state["update_count"]),
            "optimizer_consumed_transition_count": consumed,
            "post_fork_consumed_transition_count": boundary,
            "pending_transition_count": len(recovery["pending_steps"]),
            "fork_checkpoint_id": fork_record["checkpoint_id"],
            "immutable_asset_identities": current_immutable,
        }
    return models, records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = {
        "contract": args.contract.resolve(),
        "readiness": args.readiness.resolve(),
        "corpus": args.corpus.resolve(),
        "paths_config": args.paths_config.resolve(),
        "malom_manifest": args.malom_manifest.resolve(),
        "output": args.output.resolve(),
    }
    for label in (
        "contract",
        "readiness",
        "corpus",
        "paths_config",
        "malom_manifest",
    ):
        if not paths[label].is_file():
            raise EqualTransitionReportError(f"{label} is not an existing file")
    if paths["output"].exists():
        raise EqualTransitionReportError("result output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise EqualTransitionReportError("requested CUDA device is unavailable")
    device = torch.device(args.device)

    contract = _validate_contract(paths["contract"])
    readiness = _validate_readiness(paths["readiness"], contract=contract)
    source = _git_identity(str(readiness["source"]["head"]))
    corpus_sha256 = _sha256_file(paths["corpus"])
    if corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise EqualTransitionReportError("fixed phase corpus SHA-256 differs")
    corpus = _strict_json(paths["corpus"])
    validate_phase_corpus(corpus)

    settings = _strict_json(paths["paths_config"])
    human_value = Path(str(settings["human_db_path"]))
    human_path = (
        human_value.resolve()
        if human_value.is_absolute()
        else (ROOT / human_value).resolve()
    )
    malom_value = Path(str(settings["malom_db_path"]))
    malom_path = (
        malom_value.resolve()
        if malom_value.is_absolute()
        else (ROOT / malom_value).resolve()
    )
    if not human_path.is_file() or not malom_path.is_dir():
        raise EqualTransitionReportError("read-only data source is unavailable")
    human_report = _probe_human_db(human_path)
    if (
        human_report.get("error")
        or human_report.get("identity")
        != contract["data_contract"]["human_db_identity"]
        or human_report.get("malom_columns_policy")
        != "masked_historical_labels"
    ):
        raise EqualTransitionReportError("HumanDB identity or label policy differs")
    malom_manifest = load_dataset_manifest(paths["malom_manifest"])
    if malom_manifest.manifest_sha256 != contract["data_contract"][
        "malom_manifest_identity"
    ]:
        raise EqualTransitionReportError("Malom manifest identity differs")
    std_anchor = next(
        (
            component
            for component in malom_manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if std_anchor is None or _sha256_file(malom_path / "std.secval") != (
        std_anchor.sha256
    ):
        raise EqualTransitionReportError("Malom std.secval identity differs")

    before = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    human_db = _open_immutable_human_db(human_path)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    if not malom.is_available():
        human_db.close()
        raise EqualTransitionReportError("Malom dependency is unavailable")

    prefixes = _prefix_by_seed(contract)
    arms = _arm_by_cell(contract)
    seed_reports: dict[str, Any] = {}
    summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    try:
        for seed in EXPECTED_SEEDS:
            fork_envelope, fork_record = _load_fork(prefixes[seed])
            anchor_model = _load_policy(fork_envelope, device=device)
            try:
                states, feature_record = _build_feature_corpus(
                    corpus=corpus,
                    anchor_model=anchor_model,
                    human_db=human_db,
                    malom=malom,
                    device=device,
                )
            except CommonAnchorAnalysisError as exc:
                raise EqualTransitionReportError(str(exc)) from exc
            boundary_reports: list[dict[str, Any]] = []
            summaries[str(seed)] = {}
            for boundary in EXPECTED_BOUNDARIES:
                models, checkpoints = _load_candidate_pair(
                    arms,
                    seed=seed,
                    boundary=boundary,
                    fork_record=fork_record,
                    device=device,
                )
                try:
                    state_records, summary = _compare_checkpoint_pair(
                        states=states,
                        models=models,
                        device=device,
                    )
                except CommonAnchorAnalysisError as exc:
                    raise EqualTransitionReportError(str(exc)) from exc
                summaries[str(seed)][str(boundary)] = summary
                boundary_reports.append(
                    {
                        "post_fork_consumed_transitions": boundary,
                        "checkpoints": checkpoints,
                        "summary": summary,
                        "states": state_records,
                    }
                )
            seed_reports[str(seed)] = {
                "fork": fork_record,
                "feature_corpus": feature_record,
                "boundaries": boundary_reports,
            }
    finally:
        human_db.close()
        malom.close()

    after = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    if before != after:
        raise EqualTransitionReportError(
            "read-only data source observations changed during analysis"
        )
    decision = classify_transition_policy_divergence(
        summaries,
        thresholds=DEFAULT_DIVERGENCE_THRESHOLDS,
    )
    report_core = {
        "schema_version": RESULT_SCHEMA,
        "scope": {
            "read_only": True,
            "training_games": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "database_writes": 0,
            "candidate_models_loaded": True,
            "held_out_strength_claim": False,
            "promotion_publication_or_long_run_authority": False,
        },
        "identities": {
            "source": source,
            "contract": {
                "path": _relative(paths["contract"]),
                "sha256": _sha256_file(paths["contract"]),
                "plan_identity": contract["plan_identity"],
            },
            "readiness": {
                "path": _relative(paths["readiness"]),
                "sha256": _sha256_file(paths["readiness"]),
                "readiness_identity": readiness["readiness_identity"],
            },
            "fixed_phase_corpus": {
                "path": _relative(paths["corpus"]),
                "sha256": corpus_sha256,
                "corpus_identity": corpus["corpus_identity"],
            },
            "paths_config_sha256": _sha256_file(paths["paths_config"]),
            "human_db": {
                "lookup_key": "human_db_path",
                "identity": human_report["identity"],
                "historical_malom_labels": "masked",
            },
            "malom": {
                "lookup_key": "malom_db_path",
                "manifest_path": _relative(paths["malom_manifest"]),
                "manifest_file_sha256": _sha256_file(paths["malom_manifest"]),
                "manifest_identity": malom_manifest.manifest_sha256,
                "std_secval_sha256": std_anchor.sha256,
            },
        },
        "comparison_contract": {
            "seeds": list(EXPECTED_SEEDS),
            "conditions": list(EXPECTED_CONDITIONS),
            "post_fork_consumed_transition_boundaries": list(
                EXPECTED_BOUNDARIES
            ),
            "temperatures": [1.0, PRIMARY_TEMPERATURE],
            "primary_temperature": PRIMARY_TEMPERATURE,
            "state_weighting": "each of the 64 fixed positions has equal weight",
            "classification_thresholds": DEFAULT_DIVERGENCE_THRESHOLDS,
        },
        "read_only_observations": {"before": before, "after": after},
        "by_seed": seed_reports,
        "decision": decision,
        "interpretation_boundaries": [
            "full legal-action logits use one fixed feature matrix per seed",
            "top-1 changes alone do not establish material policy divergence",
            "the phase corpus is development evidence, not held-out strength evidence",
            "no supervised train or validation curves exist for this online RL diagnostic",
            "this result cannot select, promote, publish or start a model",
        ],
    }
    report = {**report_core, "result_identity": canonical_sha256(report_core)}
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output"].with_name(paths["output"].name + ".tmp")
    if temporary.exists():
        raise EqualTransitionReportError("temporary result output already exists")
    temporary.write_bytes(canonical_json_bytes(report))
    temporary.replace(paths["output"])
    print(f"report={_relative(paths['output'])}")
    print(f"sha256={_sha256_file(paths['output'])}")
    print(f"result_identity={report['result_identity']}")
    print(f"classification={decision['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
