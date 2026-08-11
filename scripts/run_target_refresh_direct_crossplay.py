#!/usr/bin/env python3
"""Authorize and run one frozen, no-update target-refresh cross-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.human_db import HumanDB  # noqa: E402
from game.board import BoardState  # noqa: E402
from learned_ai.evaluation.phase_replay_development_corpus import (  # noqa: E402
    replay_record_into_sanmill_game,
)
from learned_ai.evaluation.target_refresh_direct_crossplay import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    LEDGER_SCHEMA,
    DirectCrossplayError,
    build_direct_crossplay_schedule,
    load_direct_crossplay_plan,
    summarize_direct_crossplay,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.models.scaffolded_encoder import (  # noqa: E402
    encode_position_with_lookahead,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.generalist_run_manifest import utc_now_text  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import SanmillTrainingGame  # noqa: E402
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    load_equal_transition_contract,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402
from scripts.analyze_common_anchor_policy_distribution import (  # noqa: E402
    _load_policy,
    _open_immutable_human_db,
    _read_only_observations,
)
from scripts.prepare_target_refresh_direct_crossplay import (  # noqa: E402
    DEFAULT_MALOM_MANIFEST,
    DEFAULT_PATHS_CONFIG,
    build_readiness,
    validate_readiness,
)
from scripts.report_target_refresh_equal_transition_diagnostic import (  # noqa: E402
    _arm_by_cell,
    _load_candidate_pair,
    _load_fork,
    _prefix_by_seed,
    _strict_json,
)


DEFAULT_PLAN = ROOT / (
    "docs/experiments/sanmill-target-refresh-direct-crossplay-v1.json"
)
LAUNCH_SCHEMA = "nmm.target-refresh-direct-crossplay-launch.v1"
COMPLETION_SCHEMA = "nmm.target-refresh-direct-crossplay-completion.v1"
FAILURE_SCHEMA = "nmm.target-refresh-direct-crossplay-failure.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (ROOT / path).resolve(strict=False)
    )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise DirectCrossplayError(f"{field} is outside the repository") from exc
    return resolved


def _relative(path: Path) -> str:
    return path.resolve(strict=False).relative_to(ROOT.resolve()).as_posix()


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    target = _inside_root(path, field="output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DirectCrossplayError(f"output already exists: {target}") from exc


def _canonical_identity(
    value: Mapping[str, Any], *, field: str, identity_field: str
) -> str:
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if identity != canonical_sha256(body):
        raise DirectCrossplayError(f"{field} identity differs")
    return str(identity)


def _resolve_setting(settings: Mapping[str, Any], key: str) -> Path:
    path = Path(str(settings[key]))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _outputs(plan: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _inside_root(value, field=f"{name} output")
        for name, value in plan["output_contract"].items()
    }


def build_authorization(
    *,
    plan: Mapping[str, Any],
    readiness_identity: str,
    authorized_at_utc: str,
    decision_note: str,
) -> dict[str, Any]:
    """Build the only authorization accepted by the one-shot runner."""
    if not decision_note.strip():
        raise DirectCrossplayError("authorization decision note is required")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_at_utc": authorized_at_utc,
        "authorized_by": "product-owner",
        "operator": "product-owner-delegated-agent",
        "decision_note": decision_note.strip(),
        "plan_identity": plan["plan_identity"],
        "readiness_identity": readiness_identity,
        "resource_envelope": plan["resource_envelope"],
        "permitted_operations": [
            "load-frozen-checkpoints-read-only",
            "run-288-direct-crossplay-games-once",
            "use-sanmill-as-strict-referee-only",
            "publish-development-ledger-and-result-once",
        ],
        "prohibited_operations": plan["prohibited_operations"],
        "one_attempt": True,
        "expiry": "consumed when the one direct-crossplay launch starts",
        "claim_boundary": plan["claim_boundary"],
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def _load_readiness(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    readiness = _strict_json(path)
    return validate_readiness(readiness, plan=dict(plan))


def _validate_authorization(
    path: Path,
    *,
    plan: Mapping[str, Any],
    readiness_identity: str,
) -> tuple[dict[str, Any], str]:
    authorization = _strict_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise DirectCrossplayError("direct cross-play authorization schema differs")
    identity = _canonical_identity(
        authorization,
        field="direct cross-play authorization",
        identity_field="authorization_identity",
    )
    expected = build_authorization(
        plan=plan,
        readiness_identity=readiness_identity,
        authorized_at_utc=str(authorization.get("authorized_at_utc", "")),
        decision_note=str(authorization.get("decision_note", "")),
    )
    if authorization != expected:
        raise DirectCrossplayError("direct cross-play authorization differs")
    return authorization, identity


def record_authorization(
    *,
    plan_path: Path,
    readiness_path: Path,
    expected_readiness_identity: str,
    decision_note: str,
) -> dict[str, Any]:
    plan = load_direct_crossplay_plan(plan_path.resolve(strict=True))
    outputs = _outputs(plan)
    if readiness_path.resolve() != outputs["readiness"]:
        raise DirectCrossplayError("readiness path differs from the frozen plan")
    readiness = _load_readiness(readiness_path.resolve(strict=True), plan=plan)
    if readiness["readiness_identity"] != expected_readiness_identity:
        raise DirectCrossplayError("expected readiness identity differs")
    authorization = build_authorization(
        plan=plan,
        readiness_identity=expected_readiness_identity,
        authorized_at_utc=utc_now_text(),
        decision_note=decision_note,
    )
    _publish_exclusive(outputs["authorization"], authorization)
    return authorization


def _build_policy_generators(
    scheduled: Mapping[str, Any],
) -> dict[str, torch.Generator]:
    generators: dict[str, torch.Generator] = {}
    for colour, field in (
        ("W", "policy_seed_white"),
        ("B", "policy_seed_black"),
    ):
        seed = scheduled.get(field)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DirectCrossplayError(f"{field} is absent or invalid")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        generators[colour] = generator
    return generators


def _sample_policy_move(
    *,
    board: BoardState,
    model: torch.nn.Module,
    advisor: LookaheadAdvisor,
    generator: torch.Generator,
    temperature: float,
) -> dict[str, Any]:
    encoded = encode_position_with_lookahead(
        board,
        board.turn,
        sentinel_advisor=None,
        db=None,
        value_net=None,
        lookahead_advisor=advisor,
        specialist_db=None,
        sdb_min_samples=3,
        strict=True,
    )
    if encoded is None or not encoded.legal_moves:
        raise DirectCrossplayError("ongoing strict-referee state is not encodable")
    features = np.asarray(encoded.feat_matrix, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(encoded.legal_moves):
        raise DirectCrossplayError("policy feature matrix shape differs")
    if not np.isfinite(features).all():
        raise DirectCrossplayError("policy feature matrix is non-finite")
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        logits = model.policy_logits(feature_tensor)
        _, probabilities = trainer._policy_distribution(logits, temperature)
    # One scalar draw per logical turn makes the colour-specific random stream
    # comparable across the two swapped games even when action counts differ.
    draw = float(torch.rand((), generator=generator).item())
    cumulative = torch.cumsum(probabilities.detach().cpu(), dim=0)
    chosen_index = int(torch.searchsorted(cumulative, draw, right=False).item())
    chosen_index = min(chosen_index, len(encoded.legal_moves) - 1)
    return dict(encoded.legal_moves[chosen_index])


def _normalised_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "from": None if move.get("from") is None else str(move["from"]),
        "to": None if move.get("to") is None else str(move["to"]),
        "capture": (
            None if move.get("capture") is None else str(move["capture"])
        ),
    }


def _run_game(
    *,
    scheduled: Mapping[str, Any],
    record: Mapping[str, Any],
    models: Mapping[str, torch.nn.Module],
    advisor: LookaheadAdvisor,
    installation: Any,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    generators = _build_policy_generators(scheduled)
    colour_condition = {
        str(scheduled["no_refresh_colour"]): "no-refresh",
        str(scheduled["refresh_once_colour"]): "refresh-once",
    }
    max_plies = int(
        plan["measurement_contract"]["max_post_start_logical_plies"]
    )
    temperature = float(plan["measurement_contract"]["temperature"])
    moves: list[dict[str, str | None]] = []
    with SanmillTrainingGame(
        installation, seed=int(scheduled["referee_seed"])
    ) as game:
        board = replay_record_into_sanmill_game(record, game)
        start_state = game.state
        for _ in range(max_plies):
            if game.state.terminal:
                break
            player = board.turn
            condition = colour_condition[player]
            move = _sample_policy_move(
                board=board,
                model=models[condition],
                advisor=advisor,
                generator=generators[player],
                temperature=temperature,
            )
            game.apply_nmm_move(board, move)
            board = board.apply_move(move)
            moves.append(_normalised_move(move))
        end_state = game.state

    post_start = int(end_state.logical_ply_count - start_state.logical_ply_count)
    if post_start != len(moves):
        raise DirectCrossplayError("strict referee and ledger ply counts differ")
    if end_state.terminal:
        winner = end_state.winner
        termination_reason = end_state.outcome_reason_code
    else:
        if post_start != max_plies:
            raise DirectCrossplayError("direct game stopped without a terminal reason")
        winner = None
        termination_reason = "max-ply-truncation"
    no_refresh_name = (
        "white" if scheduled["no_refresh_colour"] == "W" else "black"
    )
    score = 0.5 if winner is None else 1.0 if winner == no_refresh_name else 0.0
    outcome_class = {0.0: "loss", 0.5: "draw", 1.0: "win"}[score]
    return {
        "schema_version": LEDGER_SCHEMA,
        "plan_identity": plan["plan_identity"],
        **dict(scheduled),
        "phase": str(record["phase"]),
        "no_refresh_score": score,
        "outcome_class": outcome_class,
        "winner": winner,
        "termination_reason": termination_reason,
        "post_start_logical_plies": post_start,
        "start_history_sha256": start_state.history_sha256,
        "end_history_sha256": end_state.history_sha256,
        "moves": moves,
    }


def _load_runtime(
    *,
    plan: Mapping[str, Any],
    paths_config_path: Path,
) -> tuple[
    Any,
    HumanDB,
    ExternalSolvedDB,
    dict[int, dict[str, torch.nn.Module]],
    dict[int, LookaheadAdvisor],
    dict[int, Mapping[str, Any]],
]:
    settings = _strict_json(paths_config_path)
    human_path = _resolve_setting(settings, "human_db_path")
    malom_path = _resolve_setting(settings, "malom_db_path")
    human_db = _open_immutable_human_db(human_path)
    malom: ExternalSolvedDB | None = None
    try:
        malom = ExternalSolvedDB(str(malom_path), strict=True)
        if not malom.is_available():
            raise DirectCrossplayError("strict Malom decoder is unavailable")
        installation = load_local_installation(paths_config_path)
        source_contract = load_equal_transition_contract(
            ROOT / plan["source"]["schedule_contract_path"]
        )
        prefixes = _prefix_by_seed(source_contract)
        arms = _arm_by_cell(source_contract)
        models_by_seed: dict[int, dict[str, torch.nn.Module]] = {}
        advisors: dict[int, LookaheadAdvisor] = {}
        anchor_by_seed = {
            int(item["seed"]): item
            for item in plan["checkpoint_contract"]["anchors"]
        }
        candidates_by_cell = {
            (int(item["seed"]), str(item["condition"])): item
            for item in plan["checkpoint_contract"]["candidates"]
        }
        for seed in plan["measurement_contract"]["seeds"]:
            anchor_envelope, anchor_record = _load_fork(prefixes[seed])
            expected_anchor = {
                key: value
                for key, value in anchor_by_seed[seed].items()
                if key != "seed"
            }
            if anchor_record != expected_anchor:
                raise DirectCrossplayError(
                    f"loaded anchor checkpoint differs for seed {seed}"
                )
            anchor_model = _load_policy(anchor_envelope, device=torch.device("cpu"))
            models, candidate_records = _load_candidate_pair(
                arms,
                seed=seed,
                boundary=8192,
                fork_record=expected_anchor,
                device=torch.device("cpu"),
            )
            for condition in ("refresh-once", "no-refresh"):
                expected_candidate = {
                    key: value
                    for key, value in candidates_by_cell[(seed, condition)].items()
                    if key not in {"seed", "condition"}
                }
                if candidate_records[condition] != expected_candidate:
                    raise DirectCrossplayError(
                        "loaded candidate checkpoint differs for "
                        f"seed {seed}, {condition}"
                    )
            models_by_seed[seed] = {
                "refresh-once": models["refresh"],
                "no-refresh": models["no-refresh"],
            }
            advisors[seed] = LookaheadAdvisor(
                sentinel=None,
                evaluate_fn=trainer._simple_evaluate,
                value_net=None,
                gap_net=None,
                human_db=human_db,
                use_sentinel=True,
                endgame_db=malom,
                ply_depth=12,
                frozen_model=anchor_model,
                frozen_device=torch.device("cpu"),
                sim_ply_depth=5,
                strict=True,
            )
        replay = _strict_json(ROOT / plan["data_contract"]["replay_corpus_path"])
        records = {int(item["record_index"]): item for item in replay["records"]}
        if sorted(records) != list(range(1, 13)):
            raise DirectCrossplayError("replay record index set differs")
        return installation, human_db, malom, models_by_seed, advisors, records
    except Exception:
        if malom is not None:
            malom.close()
        human_db.close()
        raise


def _write_ledger_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(canonical_json_bytes(row) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def launch_once(
    *,
    plan_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    paths_config_path: Path,
    malom_manifest_path: Path,
    expected_readiness_identity: str,
    run_id: str,
) -> dict[str, Any]:
    if not run_id.strip():
        raise DirectCrossplayError("direct cross-play run id is required")
    plan = load_direct_crossplay_plan(plan_path.resolve(strict=True))
    outputs = _outputs(plan)
    if readiness_path.resolve() != outputs["readiness"]:
        raise DirectCrossplayError("readiness path differs from the frozen plan")
    if authorization_path.resolve() != outputs["authorization"]:
        raise DirectCrossplayError("authorization path differs from the frozen plan")
    readiness = _load_readiness(readiness_path.resolve(strict=True), plan=plan)
    if readiness["readiness_identity"] != expected_readiness_identity:
        raise DirectCrossplayError("expected readiness identity differs")
    rebuilt = build_readiness(
        plan_path=plan_path,
        paths_config_path=paths_config_path,
        malom_manifest_path=malom_manifest_path,
    )
    if rebuilt != readiness:
        raise DirectCrossplayError("direct cross-play readiness has drifted")
    _, authorization_identity = _validate_authorization(
        authorization_path.resolve(strict=True),
        plan=plan,
        readiness_identity=expected_readiness_identity,
    )
    occupied = [
        name
        for name in ("launch", "ledger", "result", "completion", "failure")
        if outputs[name].exists()
    ]
    if occupied:
        raise DirectCrossplayError(
            "direct cross-play attempt outputs already exist: " + ", ".join(occupied)
        )

    launch_body = {
        "schema_version": LAUNCH_SCHEMA,
        "status": "started_once",
        "run_id": run_id,
        "started_at_utc": utc_now_text(),
        "plan_identity": plan["plan_identity"],
        "readiness_identity": expected_readiness_identity,
        "authorization_identity": authorization_identity,
        "source_commit": readiness["source"]["head"],
        "training_games": 0,
    }
    launch = {**launch_body, "launch_identity": canonical_sha256(launch_body)}
    _publish_exclusive(outputs["launch"], launch)
    started = time.monotonic()
    human_db: HumanDB | None = None
    malom: ExternalSolvedDB | None = None
    try:
        before = _read_only_observations(
            human_db_path=_resolve_setting(_strict_json(paths_config_path), "human_db_path"),
            malom_path=_resolve_setting(_strict_json(paths_config_path), "malom_db_path"),
        )
        (
            installation,
            human_db,
            malom,
            models_by_seed,
            advisors,
            records,
        ) = _load_runtime(plan=plan, paths_config_path=paths_config_path)
        schedule = build_direct_crossplay_schedule(plan)
        outputs["ledger"].parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        with outputs["ledger"].open("xb") as ledger:
            for scheduled in schedule:
                elapsed_hours = (time.monotonic() - started) / 3600.0
                if elapsed_hours >= float(
                    plan["resource_envelope"]["maximum_active_wall_hours"]
                ):
                    raise DirectCrossplayError(
                        "direct cross-play active wall-time limit reached"
                    )
                seed = int(scheduled["seed"])
                row = _run_game(
                    scheduled=scheduled,
                    record=records[int(scheduled["record_index"])],
                    models=models_by_seed[seed],
                    advisor=advisors[seed],
                    installation=installation,
                    plan=plan,
                )
                rows.append(row)
                _write_ledger_row(ledger, row)
                print(
                    f"[{len(rows):03d}/{len(schedule)}] seed={seed} "
                    f"start={scheduled['record_index']} "
                    f"no-refresh={row['no_refresh_colour']} "
                    f"{row['outcome_class']} ply={row['post_start_logical_plies']}",
                    flush=True,
                )
        result = summarize_direct_crossplay(plan, rows)
        result_without_identity = dict(result)
        result_without_identity.pop("result_identity", None)
        result_without_identity.update(
            {
                "run_id": run_id,
                "source_commit": readiness["source"]["head"],
                "readiness_identity": expected_readiness_identity,
                "authorization_identity": authorization_identity,
                "launch_identity": launch["launch_identity"],
                "ledger": {
                    "path": _relative(outputs["ledger"]),
                    "sha256": _sha256(outputs["ledger"]),
                    "rows": len(rows),
                },
            }
        )
        after = _read_only_observations(
            human_db_path=_resolve_setting(_strict_json(paths_config_path), "human_db_path"),
            malom_path=_resolve_setting(_strict_json(paths_config_path), "malom_db_path"),
        )
        if after != before:
            raise DirectCrossplayError("read-only source observations changed")
        result_without_identity["read_only_observations"] = {
            "before": before,
            "after": after,
        }
        result = {
            **result_without_identity,
            "result_identity": canonical_sha256(result_without_identity),
        }
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours > float(
            plan["resource_envelope"]["maximum_active_wall_hours"]
        ):
            raise DirectCrossplayError(
                "direct cross-play exceeded the active wall-time limit"
            )
        _publish_exclusive(outputs["result"], result)
        completion_body = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "completed_once",
            "run_id": run_id,
            "completed_at_utc": utc_now_text(),
            "elapsed_hours": elapsed_hours,
            "plan_identity": plan["plan_identity"],
            "readiness_identity": expected_readiness_identity,
            "authorization_identity": authorization_identity,
            "launch_identity": launch["launch_identity"],
            "result": {
                "path": _relative(outputs["result"]),
                "sha256": _sha256(outputs["result"]),
                "result_identity": result["result_identity"],
                "classification": result["decision"]["classification"],
            },
            "ledger": result["ledger"],
            "resource_accounting": {
                **plan["resource_envelope"],
                "elapsed_active_wall_hours": elapsed_hours,
            },
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        completion = {
            **completion_body,
            "completion_identity": canonical_sha256(completion_body),
        }
        _publish_exclusive(outputs["completion"], completion)
        return completion
    except Exception as exc:
        failure_body = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "run_id": run_id,
            "failed_at_utc": utc_now_text(),
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "plan_identity": plan["plan_identity"],
            "readiness_identity": expected_readiness_identity,
            "authorization_identity": authorization_identity,
            "launch_identity": launch["launch_identity"],
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "retry_or_recovery_authorized": False,
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        failure = {
            **failure_body,
            "failure_identity": canonical_sha256(failure_body),
        }
        _publish_exclusive(outputs["failure"], failure)
        raise
    finally:
        if malom is not None:
            malom.close()
        if human_db is not None:
            human_db.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record-authorization", action="store_true")
    action.add_argument("--launch", choices=("once",))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST
    )
    parser.add_argument("--expected-readiness-identity")
    parser.add_argument("--decision-note")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_direct_crossplay_plan(args.plan.resolve(strict=True))
    outputs = _outputs(plan)
    readiness_path = (
        args.readiness.resolve() if args.readiness else outputs["readiness"]
    )
    authorization_path = (
        args.authorization.resolve()
        if args.authorization
        else outputs["authorization"]
    )
    if not args.expected_readiness_identity:
        raise DirectCrossplayError("expected readiness identity is required")
    if args.record_authorization:
        authorization = record_authorization(
            plan_path=args.plan,
            readiness_path=readiness_path,
            expected_readiness_identity=args.expected_readiness_identity,
            decision_note=args.decision_note or "",
        )
        print(json.dumps(authorization, sort_keys=True))
        return 0
    completion = launch_once(
        plan_path=args.plan,
        readiness_path=readiness_path,
        authorization_path=authorization_path,
        paths_config_path=args.paths_config.resolve(strict=True),
        malom_manifest_path=args.malom_manifest.resolve(strict=True),
        expected_readiness_identity=args.expected_readiness_identity,
        run_id=args.run_id or "",
    )
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
