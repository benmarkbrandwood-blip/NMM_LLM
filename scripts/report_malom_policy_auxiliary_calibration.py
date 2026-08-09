#!/usr/bin/env python3
"""Validate and publish the four-arm policy-auxiliary calibration result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.malom_policy_auxiliary_calibration_result import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_READINESS_REPORT,
    DEFAULT_RESULT,
    MalomPolicyAuxiliaryCalibrationResultError,
    analyze_calibration_result,
    publish_result,
)
from learned_ai.training.managed_generalist import (  # noqa: E402
    ManagedContractError,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def _require_ignored_output(path: Path) -> None:
    try:
        relative = path.resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "result output must stay inside the repository's ignored area"
        ) from exc
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "result output must be ignored by Git"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    output = _resolve(args.output)
    try:
        _require_ignored_output(output)
        report = analyze_calibration_result(
            root=ROOT,
            contract_path=_resolve(args.contract),
            readiness_path=_resolve(args.readiness),
            paths_config=_resolve(args.paths_config),
        )
        publish_result(output, report)
    except (
        MalomPolicyAuxiliaryCalibrationResultError,
        ManagedContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "state": "not_reportable",
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "state": "result_published",
                "verdict": report["decision"]["verdict"],
                "selected_coefficient": report["decision"][
                    "selected_coefficient"
                ],
                "result_identity": report["result_identity"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
