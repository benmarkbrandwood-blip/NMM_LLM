#!/usr/bin/env python3
"""Independently recompute the frozen product-route primary endpoints."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (  # noqa: E402
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_product_route_heldout import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    GAME_SCHEMA,
    MATERIAL_LOWER_BOUND,
    MAXIMUM_HALF_WIDTH,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    load_sealed,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (  # noqa: E402
    sha256_file,
)


RECOMPUTE_SCHEMA = "nmm.sanmill-product-route-heldout-independent-recompute.v1"


def _read_raw_chain(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise RuntimeError("raw ledger is incomplete")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for line in raw.splitlines():
        wrapper = json.loads(line)
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "record",
            "record_sha256",
        }:
            raise RuntimeError("raw ledger wrapper differs")
        body = wrapper["record"]
        digest = wrapper["record_sha256"]
        if (
            not isinstance(body, dict)
            or body.get("previous_record_sha256") != previous
            or canonical_sha256(body) != digest
            or body.get("schema_version") != GAME_SCHEMA
        ):
            raise RuntimeError("raw ledger chain differs")
        record = dict(body)
        record.pop("previous_record_sha256")
        records.append(record)
        previous = str(digest)
    return records, previous


def _interval(values: list[float]) -> dict[str, Any]:
    if len(values) != EXPECTED_STARTS:
        raise RuntimeError("independent start support differs")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "support": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _primary(records: list[dict[str, Any]], start_ids: list[str]) -> dict[str, Any]:
    if len(records) != EXPECTED_GAMES:
        raise RuntimeError("independent game count differs")
    if any(row["termination_class"] != "rules_terminal" for row in records):
        raise RuntimeError("independent strict result is incomplete")
    grouped: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        key = (
            int(record["difficulty"]),
            str(record["route"]),
            str(record["start_id"]),
        )
        color = str(record["candidate_color"])
        if color in grouped[key]:
            raise RuntimeError("independent color unit is duplicated")
        grouped[key][color] = float(record["candidate_score"])
    primary: dict[str, Any] = {}
    for difficulty in (9, 10):
        differences: list[float] = []
        for start_id in start_ids:
            classical = grouped[(difficulty, "classical-first", start_id)]
            specialist = grouped[(difficulty, "specialist-first", start_id)]
            if set(classical) != {"W", "B"} or set(specialist) != {"W", "B"}:
                raise RuntimeError("independent both-color unit differs")
            differences.append(
                statistics.fmean(classical.values())
                - statistics.fmean(specialist.values())
            )
        interval = _interval(differences)
        if interval["half_width"] > MAXIMUM_HALF_WIDTH:
            decision = "precision_inadequate_stop"
        elif interval["lower"] >= MATERIAL_LOWER_BOUND:
            decision = "classical_first_material_route_candidate"
        else:
            decision = "no_classical_first_route_change_supported"
        primary[f"difficulty_{difficulty}_classical_minus_specialist"] = {
            **interval,
            "maximum_half_width": MAXIMUM_HALF_WIDTH,
            "minimum_material_lower_bound": MATERIAL_LOWER_BOUND,
            "precision_adequate": interval["half_width"] <= MAXIMUM_HALF_WIDTH,
            "decision": decision,
            "directional_note": (
                "specialist_first_higher" if interval["upper"] < 0.0 else None
            ),
            "difference_distribution": dict(
                sorted(Counter(str(value) for value in differences).items())
            ),
        }
    return primary


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _compare(left: Any, right: Any, path: str = "primary") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f"{path}:keys"]
        differences: list[str] = []
        for key in sorted(left):
            differences.extend(_compare(left[key], right[key], f"{path}.{key}"))
        return differences
    return [] if _same_number(left, right) else [path]


def run(args: argparse.Namespace) -> int:
    plan, plan_sha = load_sealed(
        ROOT / args.plan, schema=PLAN_SCHEMA, identity_field="plan_identity"
    )
    authorization, authorization_sha = load_sealed(
        ROOT / args.authorization,
        schema=AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )
    result, result_sha = load_sealed(
        ROOT / args.result, schema=RESULT_SCHEMA, identity_field="result_identity"
    )
    if (
        authorization["plan_identity"] != plan["plan_identity"]
        or result["plan_identity"] != plan["plan_identity"]
    ):
        raise RuntimeError("independent frozen binding differs")
    ledger_path = ROOT / result["raw_ledger"]["path"]
    if sha256_file(ledger_path) != result["raw_ledger"]["file_sha256"]:
        raise RuntimeError("independent raw ledger identity differs")
    records, tail = _read_raw_chain(ledger_path)
    if tail != result["raw_ledger"]["tail_record_sha256"]:
        raise RuntimeError("independent raw ledger tail differs")
    primary = _primary(records, list(plan["source_pool"]["suffix_start_ids"]))
    differences = _compare(primary, result["analysis"]["primary"])
    if differences:
        raise RuntimeError(f"independent primary differs: {differences}")
    body = {
        "schema_version": RECOMPUTE_SCHEMA,
        "status": "independent_recompute_matched",
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_sha,
        "result_identity": result["result_identity"],
        "result_file_sha256": result_sha,
        "raw_ledger_file_sha256": result["raw_ledger"]["file_sha256"],
        "raw_ledger_tail_record_sha256": tail,
        "records": len(records),
        "primary": primary,
        "comparison_differences": differences,
        "shared_analysis_function_used": False,
    }
    sealed = write_sealed_json(
        ROOT / args.output, body, identity_field="recompute_identity"
    )
    print(json.dumps(sealed, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-product-route-heldout-v1.json"
    )
    parser.add_argument(
        "--authorization",
        default="docs/experiments/sanmill-product-route-heldout-v1/authorization.json",
    )
    parser.add_argument(
        "--result",
        default=(
            "docs/evidence/sanmill-product-route-heldout-v1-manifest-2026-08-21.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-product-route-heldout-v1-independent-"
            "recompute-2026-08-21.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
