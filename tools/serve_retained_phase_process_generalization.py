"""Read-only local web report for the retained phase-process confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.retained_phase_process_generalization import (  # noqa: E402
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    load_game_ledger,
    summarize_records,
)
from learned_ai.evaluation.retained_passivity_diagnostic import (  # noqa: E402
    EXPECTED_GAMES as DEVELOPMENT_EXPECTED_GAMES,
    load_game_ledger as load_development_game_ledger,
    summarize_diagnostic_records as summarize_development_records,
)
from learned_ai.evaluation.retained_late_import_heldout_pool import (  # noqa: E402
    validate_retained_late_import_pool,
)
from learned_ai.evaluation.retained_heldout_score import (  # noqa: E402
    EXPECTED_GAMES as HELDOUT_EXPECTED_GAMES,
    EXPECTED_STARTS as HELDOUT_EXPECTED_STARTS,
    PLAN_SCHEMA as HELDOUT_PLAN_SCHEMA,
    load_game_ledger as load_heldout_game_ledger,
    summarize_records as summarize_heldout_records,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402


MECHANISM_SCHEMA = "nmm.retained-phase-process-mechanism-audit-result.v1"
DEFAULT_HELDOUT_POOL = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)
DEFAULT_HELDOUT_PLAN = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-heldout-score-v1.json"
)
DEFAULT_HELDOUT_OUTPUT = _ROOT / (
    "learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-heldout-score-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_spec_identity(spec: dict[str, Any]) -> None:
    identity = spec.get("spec_identity")
    body = {key: value for key, value in spec.items() if key != "spec_identity"}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ValueError("phase-process spec identity differs")


def _fixed_width_budgets(primary: dict[str, Any]) -> list[dict[str, Any]]:
    deviation = primary.get("sample_standard_deviation")
    if deviation is None:
        return []
    return [
        {
            "target_half_width": width,
            "starts": max(1, math.ceil((1.96 * float(deviation) / width) ** 2)),
            "games": max(1, math.ceil((1.96 * float(deviation) / width) ** 2)) * 4,
            "planning_only": True,
        }
        for width in (0.10, 0.075, 0.05)
    ]


def _start_clustered_precision(
    records: list[dict[str, Any]],
    *,
    start_key: str,
    value_key: str,
    require_rules_terminal: bool = False,
) -> dict[str, Any]:
    """Average both colours within a start before computing an interval."""
    by_colour: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        if (
            require_rules_terminal
            and record.get("termination_class") != "rules_terminal"
        ):
            continue
        value = record.get(value_key)
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
        else:
            continue
        key = (str(record[start_key]), str(record["candidate_color"]))
        by_colour.setdefault(key, {})[str(record["candidate_id"])] = numeric

    by_start: dict[str, dict[str, float]] = {}
    matched_colour_units = 0
    for (start_id, colour), candidates in by_colour.items():
        if set(candidates) != set(EXPECTED_CANDIDATES):
            continue
        matched_colour_units += 1
        by_start.setdefault(start_id, {})[colour] = (
            candidates[EXPECTED_CANDIDATES[1]] - candidates[EXPECTED_CANDIDATES[0]]
        )

    differences = [
        (colours["W"] + colours["B"]) / 2.0
        for _, colours in sorted(by_start.items())
        if set(colours) == {"W", "B"}
    ]
    support = len(differences)
    mean = sum(differences) / support if support else None
    if support:
        deviation = statistics.stdev(differences) if support > 1 else 0.0
        standard_error = deviation / math.sqrt(support)
        half_width = 1.96 * standard_error
        interval: list[float | None] = [mean - half_width, mean + half_width]
    else:
        deviation = standard_error = half_width = None
        interval = [None, None]
    distribution = Counter(differences)
    return {
        "support": support,
        "matched_colour_units": matched_colour_units,
        "mean": mean,
        "sample_standard_deviation": deviation,
        "standard_error": standard_error,
        "half_width": half_width,
        "interval": interval,
        "distribution": {
            str(value): distribution[value] for value in sorted(distribution)
        },
    }


def _independent_fixed_corpus_contrast(
    phase: dict[str, Any], development: dict[str, Any]
) -> dict[str, Any]:
    """Describe phase minus development fixed-corpus effects post hoc."""
    if not phase["support"] or not development["support"]:
        return {
            "mean": None,
            "standard_error": None,
            "half_width": None,
            "interval": [None, None],
        }
    mean = float(phase["mean"]) - float(development["mean"])
    standard_error = math.sqrt(
        float(phase["standard_error"]) ** 2 + float(development["standard_error"]) ** 2
    )
    half_width = 1.96 * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "half_width": half_width,
        "interval": [mean - half_width, mean + half_width],
        "development_starts": development["support"],
        "phase_starts": phase["support"],
        "post_hoc": True,
    }


def _score_planning_budgets(deviations: list[float]) -> dict[str, Any]:
    """Use the larger observed start-level score SD as a planning input."""
    conservative_deviation = max(deviations)
    rows = []
    for target_half_width in (0.03, 0.02, 0.015, 0.01):
        starts = max(
            1,
            math.ceil((1.96 * conservative_deviation / target_half_width) ** 2),
        )
        rows.append(
            {
                "target_half_width": target_half_width,
                "starts": starts,
                "games": starts * 4,
            }
        )
    return {
        "conservative_sample_standard_deviation": conservative_deviation,
        "rows": rows,
        "planning_only": True,
    }


@lru_cache(maxsize=4)
def _heldout_pool_payload_cached(
    path_text: str,
    _size: int,
    _modified_ns: int,
) -> dict[str, Any]:
    pool_path = Path(path_text)
    payload = _read_json(pool_path)
    records = validate_retained_late_import_pool(payload)
    source = payload["source_audit"]
    exposure = payload["exposure_audit"]
    strict = payload["strict_replay_audit"]
    return {
        "available": True,
        "status": payload["status"],
        "pool_identity": payload["pool_identity"],
        "records_identity": payload["records_identity"],
        "independent_starts": len(records),
        "phase_counts": payload["selection_contract"]["strict_phase_counts"],
        "source_audit": {
            "late_import_count": source["late_import_count"],
            "locally_valid_source_count": source["locally_valid_source_count"],
            "invalid_source_count": source["invalid_source_count"],
            "late_import_timestamp_range": source["late_import_timestamp_range"],
        },
        "exposure_audit": {
            "candidate_state_count": exposure["candidate_state_count"],
            "eligible_state_count": exposure["eligible_state_count"],
            "eligible_source_game_count": exposure["eligible_source_game_count"],
            "rejection_hits_nonexclusive": exposure["rejection_hits_nonexclusive"],
        },
        "strict_replay": {
            "repeat_passes": strict["repeat_passes"],
            "fresh_process_count": strict["fresh_process_count"],
            "accepted_count": strict["accepted_count"],
            "excluded_count": strict["excluded_count"],
        },
        "nested_precision_prefixes": payload["nested_precision_prefixes"],
        "claim_boundary": payload["claim_boundaries"],
    }


def _heldout_pool_payload(path: str | Path | None) -> dict[str, Any] | None:
    """Return validated source-only pool facts without exposing full histories."""
    if path is None:
        return None
    pool_path = Path(path).resolve()
    if not pool_path.is_file():
        return {
            "available": False,
            "status": "source_pool_absent",
            "message": "候选盲 held-out 源池尚未冻结。",
        }
    stat = pool_path.stat()
    return _heldout_pool_payload_cached(str(pool_path), stat.st_size, stat.st_mtime_ns)


def _heldout_score_payload(
    plan_path: str | Path | None,
    output_root: str | Path | None,
) -> dict[str, Any] | None:
    """Return plan/progress/result facts without loading either candidate."""
    if plan_path is None or output_root is None:
        return None
    frozen_path = Path(plan_path).resolve()
    if not frozen_path.is_file():
        return {
            "available": False,
            "status": "plan_not_frozen",
            "message": "held-out 得分计划尚未冻结。",
        }
    plan = _read_json(frozen_path)
    plan_identity = plan.get("plan_identity")
    plan_body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if (
        plan.get("schema_version") != HELDOUT_PLAN_SCHEMA
        or not isinstance(plan_identity, str)
        or canonical_sha256(plan_body) != plan_identity
        or plan.get("workload", {}).get("games") != HELDOUT_EXPECTED_GAMES
        or plan.get("workload", {}).get("unique_starts") != HELDOUT_EXPECTED_STARTS
    ):
        raise ValueError("held-out score plan binding differs")

    root = Path(output_root).resolve()
    spec_path = root / "spec.json"
    authorization_present = (root / "authorization.json").is_file()
    if not spec_path.is_file():
        return {
            "available": True,
            "status": "awaiting_authorization",
            "plan_identity": plan_identity,
            "selected_starts": HELDOUT_EXPECTED_STARTS,
            "expected_games": HELDOUT_EXPECTED_GAMES,
            "phase_counts": plan["corpus"]["phase_counts"],
            "maximum_primary_half_width": plan["analysis"]["engineering_interval"][
                "maximum_primary_half_width"
            ],
            "max_active_hours": plan["workload"]["max_active_hours"],
            "authorization_present": authorization_present,
            "authorization_consumed": False,
            "completed_games": 0,
            "active_seconds": 0.0,
            "primary": None,
            "by_candidate": None,
        }

    spec = _read_json(spec_path)
    _validate_spec_identity(spec)
    if spec.get("plan", {}).get("identity") != plan_identity:
        raise ValueError("held-out runtime plan differs")
    ledger_path = root / "games.jsonl"
    if ledger_path.is_file():
        records, tail = load_heldout_game_ledger(spec, ledger_path)
    else:
        records, tail = [], None
    report = summarize_heldout_records(spec, records, tail)
    progress_path = root / "progress.json"
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    status = (
        "completed"
        if (root / "completion.json").is_file()
        else "failed" if (root / "failure.json").is_file() else "running"
    )
    return {
        "available": True,
        "status": status,
        "plan_identity": plan_identity,
        "spec_identity": spec["spec_identity"],
        "selected_starts": HELDOUT_EXPECTED_STARTS,
        "expected_games": HELDOUT_EXPECTED_GAMES,
        "phase_counts": plan["corpus"]["phase_counts"],
        "maximum_primary_half_width": plan["analysis"]["engineering_interval"][
            "maximum_primary_half_width"
        ],
        "max_active_hours": plan["workload"]["max_active_hours"],
        "authorization_present": authorization_present,
        "authorization_consumed": (root / "launch.json").is_file(),
        "completed_games": len(records),
        "active_seconds": float(progress.get("active_seconds") or 0.0),
        "primary": report["paired"]["primary_start_clustered_score_v4_minus_v3"],
        "by_candidate": report["by_candidate"],
        "result_identity": report["result_identity"],
    }


def _candidate_route_signature(spec: dict[str, Any]) -> list[dict[str, Any]]:
    signature = []
    for candidate in spec.get("candidates", []):
        signature.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "bundle_identity": candidate.get("bundle", {}).get("identity"),
                "bundle_manifest_sha256": candidate.get("bundle", {}).get(
                    "manifest_sha256"
                ),
                "checkpoint_file_sha256": candidate.get("checkpoint", {}).get(
                    "file_sha256"
                ),
                "checkpoint_payload_sha256": candidate.get("checkpoint", {}).get(
                    "payload_sha256"
                ),
                "specialist_db_file_sha256": candidate.get("specialist_db", {}).get(
                    "file_sha256"
                ),
            }
        )
    return sorted(signature, key=lambda row: str(row["candidate_id"]))


def _validate_cross_corpus_comparability(
    phase_spec: dict[str, Any],
    development_spec: dict[str, Any],
    development_records: list[dict[str, Any]],
) -> None:
    if _candidate_route_signature(phase_spec) != _candidate_route_signature(
        development_spec
    ):
        raise ValueError("cross-corpus candidate route identities differ")
    if phase_spec.get("runtime") != development_spec.get("runtime"):
        raise ValueError("cross-corpus deterministic runtime differs")

    phase_protocol = phase_spec.get("protocol", {})
    development_protocol = development_spec.get("protocol", {})
    for key in (
        "candidate_move_selection",
        "color_swap",
        "result_based_early_stop",
        "safety_cap_disposition",
        "sanmill_node_ceiling_per_turn",
        "strict_referee",
    ):
        if phase_protocol.get(key) != development_protocol.get(key):
            raise ValueError(f"cross-corpus protocol differs for {key}")
    if (
        development_protocol.get("horizon_total_logical_ply") != 120
        or phase_protocol.get("horizon_post_start_logical_plies") != 108
        or development_protocol.get("max_post_prefix_logical_plies")
        != phase_protocol.get("max_post_start_logical_plies")
    ):
        raise ValueError("cross-corpus horizon or safety ceiling differs")
    prefix_plies = {
        record.get("prefix", {}).get("logical_ply_count")
        for record in development_records
    }
    if prefix_plies != {12}:
        raise ValueError("development prefixes are not all 12 logical plies")


def _development_stamp(root: Path) -> tuple[tuple[str, int, int], ...]:
    names = ("spec.json", "games.jsonl", "completion.json")
    result = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"development comparison is missing {name}")
        stat = path.stat()
        result.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(result)


@lru_cache(maxsize=4)
def _load_development_evidence_cached(
    root_text: str, _stamp: tuple[tuple[str, int, int], ...]
) -> dict[str, Any]:
    root = Path(root_text)
    spec = _read_json(root / "spec.json")
    _validate_spec_identity(spec)
    records, tail = load_development_game_ledger(spec, root / "games.jsonl")
    report = summarize_development_records(spec, records, tail)
    completion = _read_json(root / "completion.json")
    completion_body = {
        key: value for key, value in completion.items() if key != "completion_identity"
    }
    if (
        report.get("completed_games") != DEVELOPMENT_EXPECTED_GAMES
        or report.get("status") != "completed"
        or canonical_sha256(completion_body) != completion.get("completion_identity")
        or completion.get("diagnostic_id") != spec.get("diagnostic_id")
        or completion.get("spec_identity") != spec.get("spec_identity")
        or completion.get("result_identity") != report.get("result_identity")
        or completion.get("completed_games") != DEVELOPMENT_EXPECTED_GAMES
        or completion.get("ledger_sha256") != _sha256_file(root / "games.jsonl")
        or completion.get("ledger_tail_record_sha256") != tail
    ):
        raise ValueError("development completion binding differs")
    return {
        "spec": spec,
        "records": records,
        "result_identity": report.get("result_identity"),
    }


def _cross_corpus_payload(
    phase_spec: dict[str, Any],
    phase_records: list[dict[str, Any]],
    development_output_root: str | Path,
) -> dict[str, Any]:
    root = Path(development_output_root).resolve()
    development = _load_development_evidence_cached(str(root), _development_stamp(root))
    development_spec = development["spec"]
    development_records = development["records"]
    _validate_cross_corpus_comparability(
        phase_spec, development_spec, development_records
    )

    phase_survival = _start_clustered_precision(
        phase_records,
        start_key="start_id",
        value_key="ongoing_after_post_start_logical_ply_108",
    )
    development_survival = _start_clustered_precision(
        development_records,
        start_key="source_core_id",
        value_key="ongoing_after_total_logical_ply_120",
    )
    phase_score = _start_clustered_precision(
        phase_records,
        start_key="start_id",
        value_key="candidate_score",
        require_rules_terminal=True,
    )
    development_score = _start_clustered_precision(
        development_records,
        start_key="source_core_id",
        value_key="candidate_score",
        require_rules_terminal=True,
    )
    score_deviations = [
        float(item["sample_standard_deviation"])
        for item in (development_score, phase_score)
        if item["sample_standard_deviation"] is not None
    ]
    return {
        "available": True,
        "comparison_basis": {
            "same_candidate_route_identities": True,
            "same_deterministic_runtime": True,
            "same_strict_referee_and_sanmill_work": True,
            "comparable_post_start_horizon_plies": 108,
            "development_corpus_reused": True,
            "phase_corpus_project_visible": True,
            "development_result_identity": development["result_identity"],
        },
        "survival": {
            "development": development_survival,
            "phase": phase_survival,
            "phase_minus_development": _independent_fixed_corpus_contrast(
                phase_survival, development_survival
            ),
        },
        "score": {
            "development": development_score,
            "phase": phase_score,
            "phase_minus_development": _independent_fixed_corpus_contrast(
                phase_score, development_score
            ),
            "fixed_width_planning": (
                _score_planning_budgets(score_deviations)
                if len(score_deviations) == 2
                else None
            ),
        },
        "claim_boundary": {
            "post_hoc_fixed_corpus_description": True,
            "population_inference": False,
            "held_out": False,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "equivalence_claim": False,
            "authorization_for_more_games": False,
        },
    }


def build_payload(
    output_root: str | Path,
    development_output_root: str | Path | None = None,
    heldout_pool_path: str | Path | None = DEFAULT_HELDOUT_POOL,
    heldout_plan_path: str | Path | None = DEFAULT_HELDOUT_PLAN,
    heldout_output_root: str | Path | None = DEFAULT_HELDOUT_OUTPUT,
) -> dict[str, Any]:
    """Build current progress and independently recomputed process metrics."""
    root = Path(output_root).resolve()
    spec_path = root / "spec.json"
    if not spec_path.is_file():
        return {
            "available": False,
            "status": "not_started",
            "message": "诊断尚未启动；没有伪造或预填结果。",
            "expected_games": EXPECTED_GAMES,
            "expected_starts": EXPECTED_STARTS,
            "heldout_score": _heldout_score_payload(
                heldout_plan_path, heldout_output_root
            ),
        }

    spec = _read_json(spec_path)
    _validate_spec_identity(spec)
    ledger_path = root / "games.jsonl"
    if ledger_path.is_file():
        records, tail = load_game_ledger(spec, ledger_path)
    else:
        records, tail = [], None
    report = summarize_records(spec, records, tail)
    progress_path = root / "progress.json"
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    status = (
        "completed"
        if (root / "completion.json").is_file()
        else "failed" if (root / "failure.json").is_file() else "running"
    )
    report["status"] = status
    primary = report["paired"]["primary_start_clustered_108_ply_survival_v4_minus_v3"]
    mechanism = None
    mechanism_path = root / "mechanism-report.json"
    if mechanism_path.is_file():
        mechanism = _read_json(mechanism_path)
        mechanism_body = {
            key: value for key, value in mechanism.items() if key != "result_identity"
        }
        source = mechanism.get("source")
        if (
            mechanism.get("schema_version") != MECHANISM_SCHEMA
            or canonical_sha256(mechanism_body) != mechanism.get("result_identity")
            or not isinstance(source, dict)
            or source.get("diagnostic_id") != spec.get("diagnostic_id")
            or source.get("spec_identity") != spec.get("spec_identity")
            or source.get("result_identity") != report.get("result_identity")
            or source.get("ledger_sha256") != _sha256_file(ledger_path)
            or source.get("games") != EXPECTED_GAMES
            or source.get("new_games") != 0
        ):
            raise ValueError("phase-process mechanism source binding differs")
    cross_corpus = (
        _cross_corpus_payload(spec, records, development_output_root)
        if development_output_root is not None
        else None
    )
    return {
        "available": True,
        "status": status,
        "report": report,
        "progress": {
            "completed_games": int(
                progress.get("completed_games") or report["completed_games"]
            ),
            "expected_games": EXPECTED_GAMES,
            "current_game_ordinal": progress.get("current_game_ordinal"),
            "current_stage": progress.get("current_stage"),
            "current_stage_ply": progress.get("current_stage_ply"),
            "active_seconds": progress.get("active_seconds"),
        },
        "identities": {
            "diagnostic_id": spec["diagnostic_id"],
            "spec_identity": spec["spec_identity"],
            "plan_identity": spec["plan"]["identity"],
            "implementation_commit": spec["implementation"]["commit"],
            "corpus_identity": spec["corpus"]["identity"],
        },
        "precision": {
            "start_clustered_primary": primary,
            "fixed_width_budgets": _fixed_width_budgets(primary),
        },
        "mechanism": mechanism,
        "cross_corpus": cross_corpus,
        "heldout_pool": _heldout_pool_payload(heldout_pool_path),
        "heldout_score": _heldout_score_payload(heldout_plan_path, heldout_output_root),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NMM_LLM · v3/v4 阶段过程确认</title>
<style>
:root{color-scheme:dark;--bg:#06111f;--panel:#0b1b2d;--panel2:#10243a;--line:#24415d;--text:#edf6ff;--muted:#96adc2;--cyan:#52c7ef;--amber:#f2b84b;--green:#6dd7a0;--red:#ff7c8a}*{box-sizing:border-box}body{margin:0;background:linear-gradient(160deg,#06111f,#081522 52%,#06101b);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}main{max-width:1380px;margin:auto;padding:26px}h1{font-size:23px;margin:0 0 4px}h2{font-size:16px;margin:0 0 10px}.sub,.help{color:var(--muted)}.sub{margin-bottom:18px}.help{font-size:12px;margin:-5px 0 12px}.notice{border:1px solid var(--amber);background:#2b2112;color:#ffe3a2;padding:12px 14px;border-radius:8px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card,.panel{border:1px solid var(--line);background:linear-gradient(160deg,var(--panel2),var(--panel));border-radius:7px;padding:14px}.card .k{color:var(--muted);font-size:12px}.card .v{font-size:24px;font-weight:700;margin:3px 0}.card .d{color:var(--muted);font-size:12px}.panel{margin-top:12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:190px 1fr 72px;gap:10px;align-items:center}.track{height:10px;background:#071320;border-radius:20px;overflow:hidden}.fill{height:100%;background:var(--cyan)}.fill.v4{background:var(--amber)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}.badge{display:inline-block;border:1px solid var(--line);padding:2px 7px;border-radius:10px;color:var(--muted)}details{border-top:1px solid var(--line);padding-top:10px;margin-top:10px}summary{cursor:pointer;color:var(--cyan)}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#b9ddf0;word-break:break-all}.empty{max-width:760px;margin:20vh auto;text-align:center}.empty .panel{padding:30px}.bad{color:var(--red)}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.barrow{grid-template-columns:130px 1fr 62px}}@media(max-width:520px){main{padding:14px}.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main id="app"><div class="empty"><div class="panel"><h1>正在读取阶段过程证据…</h1></div></div></main>
<script>
const C={v3:'retained-v3-refresh50',v4:'retained-v4-no-refresh'};
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const pp=v=>v==null?'—':`${v>0?'+':''}${(100*v).toFixed(2)}pp`;
const num=(v,d=1)=>v==null?'—':Number(v).toFixed(d);
const integer=v=>v==null?'—':Math.round(v).toLocaleString('zh-CN');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const card=(k,v,d)=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
function bar(label,value,klass=''){const w=value==null?0:Math.max(0,Math.min(100,value*100));return `<div class="barrow"><span>${label}</span><div class="track"><div class="fill ${klass}" style="width:${w}%"></div></div><b>${pct(value)}</b></div>`}
function decisionText(value){return ({pending:'等待 39 个完整起点',inconclusive:'区间跨 0：不确定',inconclusive_precision:'半宽超门：精度不足',v4_higher_108_post_start_ply_survival:'v4 的相对窗口存活率更高',v3_higher_108_post_start_ply_survival:'v3 的相对窗口存活率更高'})[value]||value}
function phaseRows(phases){return ['placement','movement','flying'].map(phase=>{const row=phases[phase]||{},a=row[C.v3],b=row[C.v4];return `<tr><td>${phase}</td><td>${a?integer(a.games):'—'}</td><td>${a?pct(a.horizon_108_post_start.survival_rate):'—'}</td><td>${b?integer(b.games):'—'}</td><td>${b?pct(b.horizon_108_post_start.survival_rate):'—'}</td></tr>`}).join('')}
function processRows(a,b){const rows=[['起点 no-capture','start_no_capture'],['窗口 no-capture','horizon_no_capture'],['窗口 − 起点 no-capture','horizon_minus_start_no_capture'],['终局 no-capture','final_no_capture'],['起点当前重复计数','start_repetition_current'],['窗口当前重复计数','horizon_repetition_current'],['终局当前重复计数','final_repetition_current']];return rows.map(([label,key])=>{const x=a.history_process[key],y=b.history_process[key];return `<tr><td>${label}</td><td>${integer(x.support)}</td><td>${num(x.mean)}</td><td>${integer(y.support)}</td><td>${num(y.mean)}</td></tr>`}).join('')}
function reasonRows(a,b){const keys=[...new Set([...Object.keys(a.outcome_reasons||{}),...Object.keys(b.outcome_reasons||{})])].sort();if(!keys.length)return '<tr><td>暂无规则终局</td><td>0</td><td>0</td></tr>';return keys.map(key=>`<tr><td>${esc(key)}</td><td>${integer(a.outcome_reasons[key]||0)}</td><td>${integer(b.outcome_reasons[key]||0)}</td></tr>`).join('')}
function precisionBlock(x){if(!x.start_clustered_primary.support)return `<div class="panel"><h2>起点聚类精度</h2><p class="help">等待同一起点的候选执白、执黑两个颜色单元都形成完整 v3/v4 配对。</p></div>`;const p=x.start_clustered_primary,iv=p.interval||[null,null],dist=Object.entries(p.distribution||{}).sort((a,b)=>Number(a[0])-Number(b[0])).map(([v,n])=>`<tr><td>${pp(Number(v))}</td><td>${integer(n)}</td></tr>`).join(''),budgets=x.fixed_width_budgets.map(row=>`<tr><td>${pp(row.target_half_width)}</td><td>${integer(row.starts)}</td><td>${integer(row.games)}</td></tr>`).join('');return `<div class="panel"><h2>起点聚类精度与差值分布</h2><p class="help">先在每个起点内平均候选执白、执黑的两个差值，再跨独立起点计算工程区间；颜色单元不能当成独立样本。预算只是用已观测标准差做的固定半宽说明，不会自动扩展本次 39 起点合同。</p><div class="two"><table><thead><tr><th>两色平均差</th><th>起点</th></tr></thead><tbody>${dist}</tbody></table><table><thead><tr><th>目标半宽</th><th>估计起点</th><th>估计局数</th></tr></thead><tbody>${budgets}</tbody></table></div></div>`}
function crossCorpusBlock(x){if(!x)return '';const s=x.survival,d=s.development,p=s.phase,c=s.phase_minus_development,sc=x.score,ds=sc.development,ps=sc.phase,cs=sc.phase_minus_development,ci=c.interval||[null,null],dsi=d.interval||[null,null],psi=p.interval||[null,null],dc=ds.interval||[null,null],pc=ps.interval||[null,null],cc=cs.interval||[null,null],planning=sc.fixed_width_planning,rows=planning?planning.rows.map(row=>`<tr><td>${pp(row.target_half_width)}</td><td>${integer(row.starts)}</td><td>${integer(row.games)}</td></tr>`).join(''):'';return `<div class="panel"><h2>跨语料复现：原存活方向未复现</h2><p class="help">两批使用相同候选 route、checkpoint、SpecialistDB、确定性 CPU float32、严格裁判与 500,000 节点 Sanmill。开发集从 12 手前缀到总第 120 手，等价于阶段集的相对 108 手。两批起点不同且都已对项目可见。</p><table><thead><tr><th>固定语料</th><th>独立起点</th><th>存活差 v4−v3</th><th>95% 工程区间</th></tr></thead><tbody><tr><td>复用开发集</td><td>${integer(d.support)}</td><td>${pp(d.mean)}</td><td>${pp(dsi[0])} … ${pp(dsi[1])}</td></tr><tr><td>阶段历史集</td><td>${integer(p.support)}</td><td>${pp(p.mean)}</td><td>${pp(psi[0])} … ${pp(psi[1])}</td></tr><tr><td>阶段 − 开发（事后）</td><td>${integer(c.phase_starts)} + ${integer(c.development_starts)}</td><td>${pp(c.mean)}</td><td>${pp(ci[0])} … ${pp(ci[1])}</td></tr></tbody></table><p class="help"><b>结论边界：</b>开发集上的正方向没有在阶段集复现。最后一行只是两个固定语料效应的事后工程描述，不是预注册方向门、总体推断、refresh 因果或棋力结论，也不授权追加样本。</p><div class="two"><div><h2>配对得分（仅规划）</h2><table><thead><tr><th>固定语料</th><th>起点</th><th>得分差 v4−v3</th><th>工程区间</th></tr></thead><tbody><tr><td>复用开发集</td><td>${integer(ds.support)}</td><td>${pp(ds.mean)}</td><td>${pp(dc[0])} … ${pp(dc[1])}</td></tr><tr><td>阶段历史集</td><td>${integer(ps.support)}</td><td>${pp(ps.mean)}</td><td>${pp(pc[0])} … ${pp(pc[1])}</td></tr><tr><td>阶段 − 开发（事后）</td><td>${integer(cs.phase_starts)} + ${integer(cs.development_starts)}</td><td>${pp(cs.mean)}</td><td>${pp(cc[0])} … ${pp(cc[1])}</td></tr></tbody></table></div><div><h2>未来 held-out 固定半宽预算</h2><table><thead><tr><th>目标半宽</th><th>估计起点</th><th>总局数</th></tr></thead><tbody>${rows}</tbody></table></div></div><p class="help">预算采用两批已观测起点级得分标准差中较大的 ${planning?pct(planning.conservative_sample_standard_deviation):'—'}，每个起点四局。它只是保守的工程规划输入，不是总体方差保证、等效界值或新评测授权；真正 held-out 必须预先冻结配对得分主指标、语料、目标半宽/最小效应/等效界值及资源上限。</p></div>`}
function heldoutPoolBlock(x){if(!x)return '';if(!x.available)return `<div class="panel"><h2>真正 held-out 源池</h2><p class="help">${esc(x.message||'候选盲源池尚未冻结。')}</p></div>`;const widths={64:'3.0pp',142:'2.0pp',253:'1.5pp',568:'1.0pp'},rows=x.nested_precision_prefixes.map(row=>`<tr><td>${widths[row.target_starts]||'—'}</td><td>${integer(row.target_starts)}</td><td>${integer(row.target_games)}</td><td>${row.available?'可用':'不足'}</td><td>${integer(row.phase_counts.placement||0)} / ${integer(row.phase_counts.movement||0)} / ${integer(row.phase_counts.flying||0)}</td></tr>`).join(''),s=x.source_audit,e=x.exposure_audit,r=x.strict_replay,p=x.phase_counts;return `<div class="panel"><h2>真正 held-out 候选盲源池（已冻结，尚未评测）</h2><p class="help">406 局迟到导入的 PlayOK 人类棋谱未进入两条 retained route 共用的活动 HumanDB。选择不读取人类胜负，也不加载 v3/v4 policy；每个源局最多一个起点，并要求起点 ring16 轨道唯一。</p><div class="grid">${card('冻结独立起点',integer(x.independent_starts),`placement ${integer(p.placement)} · movement ${integer(p.movement)} · flying ${integer(p.flying)}`)}${card('合法源局',`${integer(s.locally_valid_source_count)} / ${integer(s.late_import_count)}`,`${integer(s.invalid_source_count)} 局 fail-closed 排除`)}${card('零暴露状态',integer(e.eligible_state_count),`扫描 ${integer(e.candidate_state_count)} 个第 12 手后状态`)}${card('严格历史重放',`${integer(r.accepted_count)} / ${integer(x.independent_starts)}`,`${integer(r.fresh_process_count)} 个新裁判进程；排除 ${integer(r.excluded_count)}`)}</div><table><thead><tr><th>规划目标半宽</th><th>起点</th><th>总局数</th><th>源池</th><th>P / M / F 起点</th></tr></thead><tbody>${rows}</tbody></table><p class="help"><b>帮助：</b>“源池可用”只表示存在足够的候选盲、训练库未暴露、严格非终局历史；它不等于已经完成 held-out，也不授权任何对局。1pp 方案需要 568 起点，当前只有 361，因此不能按该功效目标冻结。下一步仍须由产品负责人选择固定半宽、方向最小效应或等效框架，再冻结子集、计划、资源上限与 readiness。</p><code>held-out source pool ${esc(x.pool_identity)} · records ${esc(x.records_identity)}</code></div>`}
function mechanismBlock(m){if(!m)return `<div class="panel"><h2>安全推进与完整排序复算</h2><p class="help">等待完整逐手账本的身份绑定零新对局复算。网页不会从普通吃子率、粗 W/D/L 或存活率猜测安全吃子机会与完整 Malom 排序。</p></div>`;const a=m.by_candidate[C.v3],b=m.by_candidate[C.v4],sa=a.safe_progress.all_candidate_turns,sb=b.safe_progress.all_candidate_turns,sha=a.safe_progress.after_relative_horizon_candidate_turns,shb=b.safe_progress.after_relative_horizon_candidate_turns,oa=a.complete_order.all_candidate_turns,ob=b.complete_order.all_candidate_turns,oha=a.complete_order.after_relative_horizon_candidate_turns,ohb=b.complete_order.after_relative_horizon_candidate_turns,ps=m.paired.start_clustered_missed_safe_capture_share_v4_minus_v3,po=m.paired.start_clustered_mean_order_regret_v4_minus_v3,siv=ps.interval||[null,null],oiv=po.interval||[null,null];return `<div class="panel"><h2>安全吃子与后缀重访（零新对局复算）</h2><p class="help">安全吃子机会要求至少一个完整合法吃子动作保持当前 Malom 粗 W/D/L，并会重置严格无吃子计数。机会内选择率的分母是安全吃子机会；错过份额与棋盘重访率的分母是全部候选回合。棋盘重访只检查冻结起点及已记录 post-start 后缀，不等于严格三次重复。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>安全机会</th><th>机会内保值吃子</th><th>错过/候选回合</th><th>重访/候选回合</th><th>窗口后重访</th></tr></thead><tbody><tr><td>v3</td><td>${integer(sa.candidate_turns)}</td><td>${integer(sa.safe_capture_opportunity_turns)}</td><td>${pct(sa.safe_capture_selection_rate_given_opportunity)}</td><td>${pct(sa.missed_safe_capture_share_per_candidate_turn)}</td><td>${pct(sa.chosen_board_revisit_rate)}</td><td>${sha.chosen_board_revisit_turns} / ${sha.candidate_turns}</td></tr><tr><td>v4</td><td>${integer(sb.candidate_turns)}</td><td>${integer(sb.safe_capture_opportunity_turns)}</td><td>${pct(sb.safe_capture_selection_rate_given_opportunity)}</td><td>${pct(sb.missed_safe_capture_share_per_candidate_turn)}</td><td>${pct(sb.chosen_board_revisit_rate)}</td><td>${shb.chosen_board_revisit_turns} / ${shb.candidate_turns}</td></tr></tbody></table><p class="help">起点聚类的错过份额差 v4−v3：${pp(ps.mean)}，工程区间 ${pp(siv[0])} … ${pp(siv[1])}，支持 ${ps.support} / 39 个完整起点。该指标是探索性机制证据，没有方向性验收门。</p></div><div class="panel"><h2>完整 Malom 保值集合排序（零新对局复算）</h2><p class="help">只在粗 W/D/L 保值动作集合内、且每个保值动作都有完整可比 OracleMoveValue 时排序。序位后悔 0 表示选最高等级、1 表示选最低不同等级；分母是可完整排序回合。它是 history-free 位置排序，不是终局距离、活性或棋力。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>可排序覆盖</th><th>有不同等级</th><th>机会内选最高等级</th><th>序位后悔*</th><th>窗口后后悔*</th></tr></thead><tbody><tr><td>v3</td><td>${integer(oa.candidate_turns)}</td><td>${pct(oa.within_wdl_orderable_coverage_per_candidate_turn)}</td><td>${integer(oa.full_order_choice_opportunity_turns)}</td><td>${pct(oa.chosen_full_order_best_rate_given_opportunity)}</td><td>${pct(oa.mean_normalised_ordinal_regret_given_orderable)}</td><td>${pct(oha.mean_normalised_ordinal_regret_given_orderable)}</td></tr><tr><td>v4</td><td>${integer(ob.candidate_turns)}</td><td>${pct(ob.within_wdl_orderable_coverage_per_candidate_turn)}</td><td>${integer(ob.full_order_choice_opportunity_turns)}</td><td>${pct(ob.chosen_full_order_best_rate_given_opportunity)}</td><td>${pct(ob.mean_normalised_ordinal_regret_given_orderable)}</td><td>${pct(ohb.mean_normalised_ordinal_regret_given_orderable)}</td></tr></tbody></table><p class="help">起点聚类的平均序位后悔差 v4−v3：${pp(po.mean)}，工程区间 ${pp(oiv[0])} … ${pp(oiv[1])}，支持 ${po.support} / 39 个完整起点。条件版本不会用更大的候选回合分母摊薄；支持不足时必须显示而不能补零。</p></div>`}
function render(payload){const app=document.getElementById('app');if(!payload.available){app.innerHTML=`<div class="empty"><div class="panel"><h1>v3/v4 阶段过程确认</h1><p class="sub">${esc(payload.message)}</p><div class="notice">没有精确计划和授权时，网页只显示“未启动”，不会预填或推测结果。</div></div></div>`;return}const r=payload.report,a=r.by_candidate[C.v3],b=r.by_candidate[C.v4],p=r.paired.primary_start_clustered_108_ply_survival_v4_minus_v3,iv=p.interval||[null,null],active=payload.progress.active_seconds;
app.innerHTML=`<h1>NMM_LLM · retained-v3 / no-refresh-v4 阶段过程确认</h1><div class="sub">${esc(payload.identities.diagnostic_id)} · <span class="badge">${esc(payload.status)}</span></div><div class="notice"><b>固定项目可见语料的过程确认，不是 held-out 棋力评测。</b> 结果不能归因 refresh，不能用于晋级、发布或释放。</div>
<div class="grid">${card('完成进度',`${payload.progress.completed_games} / ${payload.progress.expected_games}`,payload.progress.current_stage?`game ${Number(payload.progress.current_game_ordinal)+1} · ${payload.progress.current_stage} ply ${payload.progress.current_stage_ply}`:'当前无在途对局')}${card('完整起点',`${r.paired.start_units_complete} / ${r.paired.start_units_expected}`,`${r.paired.matched_colour_units_complete} / ${r.paired.matched_colour_units_expected} 个颜色配对`)}${card('主差值 v4 − v3',pp(p.mean),iv[0]==null?'等待完整起点':`工程区间 ${pp(iv[0])} … ${pp(iv[1])}`)}${card('主判决',decisionText(p.decision),`半宽 ${pp(p.half_width)}；门限 10.00pp`)}${card('v3: 相对 108 手仍在进行',pct(a.horizon_108_post_start.survival_rate),`${a.horizon_108_post_start.survived} / ${a.games} 局`)}${card('v4: 相对 108 手仍在进行',pct(b.horizon_108_post_start.survival_rate),`${b.horizon_108_post_start.survived} / ${b.games} 局`)}${card('活动用时',active==null?'—':num(active/60)+' min','只计 evaluator active time；上限 2 h')}${card('报告身份',r.result_identity?esc(r.result_identity.slice(0,12)):'—','实时从规范账本独立复算')}</div>
<div class="panel"><h2>相对 108 手 continuation survival</h2><p class="help">从每个冻结历史起点再走 108 个完整逻辑手后，严格裁判仍未终局。它不是和棋、不是胜率，也不预测最终结果；不同起点的绝对手数不同。</p><div class="bars">${bar('retained-v3 refresh-50',a.horizon_108_post_start.survival_rate)}${bar('retained-v4 no-refresh',b.horizon_108_post_start.survival_rate,'v4')}</div></div>
${precisionBlock(payload.precision)}
${crossCorpusBlock(payload.cross_corpus)}
${heldoutPoolBlock(payload.heldout_pool)}
<div class="panel"><h2>按起始阶段分层</h2><p class="help">分母是各阶段已经完成的候选对局数；placement / movement / flying 的固定支持分别来自 18 / 14 / 7 个起点。</p><table><thead><tr><th>阶段</th><th>v3 局数</th><th>v3 存活率</th><th>v4 局数</th><th>v4 存活率</th></tr></thead><tbody>${phaseRows(r.by_phase)}</tbody></table></div>
<div class="two"><div class="panel"><h2>无吃子与重复过程</h2><p class="help">每一行都显示自己的支持数；窗口行只含到达相对 108 手的局。严格无吃子/三次重复历史由 Sanmill 裁判持有，不能由 Malom 棋盘值替代。</p><table><thead><tr><th>指标</th><th>v3 n</th><th>v3 均值</th><th>v4 n</th><th>v4 均值</th></tr></thead><tbody>${processRows(a,b)}</tbody></table></div><div class="panel"><h2>规则终止原因</h2><p class="help">1,536 post-start 是故障安全 cap；命中时记 incomplete，绝不转成和棋。</p><table><thead><tr><th>原因</th><th>v3</th><th>v4</th></tr></thead><tbody>${reasonRows(a,b)}</tbody></table></div></div>
${mechanismBlock(payload.mechanism)}
<div class="two"><div class="panel"><h2>长度（post-start）</h2><table><thead><tr><th>候选</th><th>支持</th><th>均值</th><th>中位</th><th>P90</th><th>最大</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>`<tr><td>${label}</td><td>${x.lengths.post_start.support}</td><td>${num(x.lengths.post_start.mean)}</td><td>${num(x.lengths.post_start.median,0)}</td><td>${num(x.lengths.p90_post_start,0)}</td><td>${integer(x.lengths.post_start.max)}</td></tr>`).join('')}</tbody></table></div><div class="panel"><h2>Malom 候选动作过程</h2><p class="help">query coverage 的分母是候选回合；保值/降级率的分母仅是可查询候选回合。Malom 是 history-free 位置理论值。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>覆盖率</th><th>保值率*</th><th>降级率*</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>`<tr><td>${label}</td><td>${integer(x.candidate_malom_moves.candidate_turns)}</td><td>${pct(x.candidate_malom_moves.query_coverage)}</td><td>${pct(x.candidate_malom_moves.preserving_rate_given_queryable)}</td><td>${pct(x.candidate_malom_moves.downgrade_rate_given_queryable)}</td></tr>`).join('')}</tbody></table></div></div>
<div class="two"><div class="panel"><h2>相对窗口 Malom 理论 W/D/L</h2><p class="help">仅含到达窗口的快照，从候选视角投影；它不携带三次重复与无吃子历史，不是严格终局裁定。</p><table><thead><tr><th>候选</th><th>快照</th><th>可查</th><th>W</th><th>D</th><th>L</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>{const m=x.malom_at_horizon_candidate_perspective;return `<tr><td>${label}</td><td>${m.snapshot_support}</td><td>${m.queryable}</td><td>${m.wins}</td><td>${m.draws}</td><td>${m.losses}</td></tr>`}).join('')}</tbody></table></div><div class="panel"><h2>最终严格规则 W/D/L（描述性）</h2><p class="help">只含规则终局，cap 单列剔除。该端点未按稀有胜负设计功效，不能产生 held-out 棋力、等效或晋级主张。</p><table><thead><tr><th>候选</th><th>支持</th><th>胜</th><th>和</th><th>负</th><th>cap</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>{const w=x.eventual_rules_wdl;return `<tr><td>${label}</td><td>${w.support}</td><td>${w.wins}</td><td>${w.draws}</td><td>${w.losses}</td><td>${w.safety_cap_excluded}</td></tr>`}).join('')}</tbody></table></div></div>
<div class="panel"><h2>指标帮助与身份边界</h2><details open><summary>这些数能回答什么？</summary><p class="help">只能回答两个命名 final route 在这 39 个固定、项目已可见、起点处对两候选数据库均无 D4 命中的阶段历史上，过程指标是否复现。存活率方向本身不表示棋力；W/D/L 是次要描述；差异不能归因 target refresh。</p></details><details><summary>安全吃子与完整排序指标在哪里？</summary><p class="help">它们必须从完整逐手账本做身份绑定的零新对局复算，并显示各自机会分母与 query coverage。在该复算产物存在前，网页不会从普通吃子率或粗 W/D/L 猜测结果。</p></details><code>plan ${esc(payload.identities.plan_identity)} · spec ${esc(payload.identities.spec_identity)} · corpus ${esc(payload.identities.corpus_identity)} · source ${esc(payload.identities.implementation_commit)}</code></div>`}
async function tick(){try{const response=await fetch('/api/diagnostic',{cache:'no-store'});if(!response.ok)throw new Error(await response.text());render(await response.json())}catch(error){document.getElementById('app').innerHTML=`<div class="empty"><div class="panel"><h1>网页读取失败</h1><p class="bad">${esc(error.message)}</p></div></div>`}}
tick();setInterval(tick,3000);
</script></body></html>"""

HELDOUT_SCORE_SCRIPT = r"""
function heldoutScoreDecision(value){return ({pending:'等待完整配对',inconclusive:'区间跨 0：不确定',inconclusive_precision:'实际半宽超过 1.5pp',inconclusive_incomplete_safety_cap:'存在未到规则终局的 cap：主结论无效',v4_higher_fixed_heldout_score:'v4 在冻结 held-out 语料上得分更高',v3_higher_fixed_heldout_score:'v3 在冻结 held-out 语料上得分更高'})[value]||value}
function heldoutScoreBlock(x){if(!x)return '';if(!x.available)return `<div class="panel"><h2>held-out 高精度得分方案</h2><p class="help">${esc(x.message||'计划尚未冻结。')}</p></div>`;const p=x.primary,iv=p&&p.interval?p.interval:[null,null],a=x.by_candidate&&x.by_candidate[C.v3],b=x.by_candidate&&x.by_candidate[C.v4],phase=x.phase_counts||{};return `<div class="panel"><h2>held-out 高精度得分方案：253 起点 / 1,012 局 / ±1.5pp</h2><p class="help">产品已选固定半宽描述框架。每个独立起点由 v3、v4 各执白和执黑共四局组成；先在起点内平均两种颜色的 v4−v3 得分差，再跨 253 个起点计算区间。胜/和/负按 1 / 0.5 / 0 计分。</p><div class="grid">${card('状态',esc(x.status),x.authorization_present?(x.authorization_consumed?'授权已消费':'授权文件已存在，尚未消费'):'尚无启动授权')}${card('完成进度',`${integer(x.completed_games)} / ${integer(x.expected_games)}`,`活动上限 ${num(x.max_active_hours,1)} h`)}${card('独立起点',integer(x.selected_starts),`P ${integer(phase.placement)} · M ${integer(phase.movement)} · F ${integer(phase.flying)}`)}${card('目标半宽',pp(x.maximum_primary_half_width),'95% 起点聚类工程区间')}${card('主得分差 v4−v3',p?pp(p.mean):'—',p&&iv[0]!=null?`${pp(iv[0])} … ${pp(iv[1])}`:'尚未产生评测对局')}${card('主判决',p?heldoutScoreDecision(p.decision):'等待授权',p?`实际半宽 ${pp(p.half_width)}；支持 ${integer(p.support)} 起点`:'计划冻结不等于可以启动')}${card('v3 严格终局计分',a?pct(a.eventual_rules_wdl.score_rate):'—',a?`${integer(a.eventual_rules_wdl.support)} 局；cap ${integer(a.eventual_rules_wdl.safety_cap_excluded)}`:'等待对局')}${card('v4 严格终局计分',b?pct(b.eventual_rules_wdl.score_rate):'—',b?`${integer(b.eventual_rules_wdl.support)} 局；cap ${integer(b.eventual_rules_wdl.safety_cap_excluded)}`:'等待对局')}</div><p class="help"><b>怎么读：</b>区间完全高于 0 且半宽不超过 1.5pp，才判 v4 在这批冻结 held-out 起点上得分更高；完全低于 0 才判 v3 更高；跨 0 只能判“不确定”，绝不等同于“两模型等效”。若任何对局命中 1,536 post-start 安全 cap，该局不按和棋计，方向性主结论 fail closed。108 手存活、无吃子计时、重复、阶段和 Malom 指标都是解释性过程指标，不能替换配对得分主指标。该结果也不能隔离 target refresh 因果、外推 Elo/总体棋力或自动触发晋级、发布、释放。</p><code>plan ${esc(x.plan_identity)}${x.spec_identity?` · spec ${esc(x.spec_identity)}`:''}${x.result_identity?` · result ${esc(x.result_identity)}`:''}</code></div>`}
"""

HTML = (
    HTML.replace(
        "function heldoutPoolBlock",
        HELDOUT_SCORE_SCRIPT + "\nfunction heldoutPoolBlock",
    )
    .replace(
        "${heldoutPoolBlock(payload.heldout_pool)}",
        (
            "${heldoutScoreBlock(payload.heldout_score)}\n"
            "${heldoutPoolBlock(payload.heldout_pool)}"
        ),
    )
    .replace(
        (
            "下一步仍须由产品负责人选择固定半宽、方向最小效应或等效框架，"
            "再冻结子集、计划、资源上限与 readiness。"
        ),
        "当前实际选择与状态以本页“held-out 高精度得分方案”卡片为准。",
    )
)


class Handler(BaseHTTPRequestHandler):
    output_root: Path
    development_output_root: Path | None
    heldout_pool_path: Path | None
    heldout_plan_path: Path | None
    heldout_output_root: Path | None

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        try:
            if route == "/":
                self._send(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    HTML.encode("utf-8"),
                )
                return
            if route == "/api/diagnostic":
                body = json.dumps(
                    build_payload(
                        self.output_root,
                        self.development_output_root,
                        self.heldout_pool_path,
                        self.heldout_plan_path,
                        self.heldout_output_root,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send(
                    HTTPStatus.OK,
                    "application/json; charset=utf-8",
                    body,
                )
                return
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")
        except Exception as exc:
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "text/plain; charset=utf-8",
                str(exc).encode("utf-8"),
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--development-output-root",
        type=Path,
        help=(
            "optional completed retained passivity diagnostic root for an "
            "identity-checked, zero-new-game cross-corpus comparison"
        ),
    )
    parser.add_argument(
        "--heldout-pool",
        type=Path,
        default=DEFAULT_HELDOUT_POOL,
        help=(
            "validated candidate-blind source-only pool to show beside the "
            "completed development diagnostics"
        ),
    )
    parser.add_argument(
        "--heldout-plan",
        type=Path,
        default=DEFAULT_HELDOUT_PLAN,
        help="frozen high-precision held-out score plan",
    )
    parser.add_argument(
        "--heldout-output-root",
        type=Path,
        default=DEFAULT_HELDOUT_OUTPUT,
        help="read-only runtime evidence root for the held-out score plan",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser


def main() -> int:
    args = _parser().parse_args()
    handler = type(
        "PhaseProcessHandler",
        (Handler,),
        {
            "output_root": args.output_root,
            "development_output_root": args.development_output_root,
            "heldout_pool_path": args.heldout_pool,
            "heldout_plan_path": args.heldout_plan,
            "heldout_output_root": args.heldout_output_root,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
