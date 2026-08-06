from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ADAPTER = Path("tools/nmm_llm_mif_adapter.py").resolve()


def _run(wire: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-B", str(ADAPTER)],
        input=wire,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_process_emits_one_lf_only_response() -> None:
    request = {
        "protocol": "MIF-INTEROP/1",
        "kind": "request",
        "requestId": "capabilities",
        "operation": "capabilities",
        "payload": {},
    }
    completed = _run(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout.endswith(b"\n")
    assert b"\r\n" not in completed.stdout
    assert json.loads(completed.stdout)["status"] == "ok"


def test_process_rejects_duplicate_members_after_unescape() -> None:
    wire = (
        b'{"protocol":"MIF-INTEROP/1","kind":"request",'
        b'"requestId":"one","requestId":"two",'
        b'"operation":"capabilities","payload":{}}\n'
    )
    completed = _run(wire)
    response = json.loads(completed.stdout)
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"][0]["code"] == (
        "duplicate-member-after-unescape"
    )


def test_process_rejects_crlf_transport() -> None:
    completed = _run(
        b'{"protocol":"MIF-INTEROP/1","kind":"request",'
        b'"requestId":"x","operation":"capabilities","payload":{}}\r\n'
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"LF-only" in completed.stderr


def test_process_reports_oversized_request_as_resource_diagnostic() -> None:
    limit = 16_777_216
    completed = _run((b"x" * (limit + 1)) + b"\n")
    assert completed.returncode == 0
    assert completed.stderr == b""
    response = json.loads(completed.stdout)
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"] == [
        {
            "category": "resource",
            "code": "resource-limit",
            "resourceLimit": {
                "name": "interop-request-bytes",
                "limit": limit,
                "actual": limit + 1,
            },
            "message": "interop-request-bytes resource limit exceeded",
        }
    ]


def test_process_resynchronizes_after_oversized_record() -> None:
    limit = 16_777_216
    valid_request = {
        "protocol": "MIF-INTEROP/1",
        "kind": "request",
        "requestId": "after-limit",
        "operation": "capabilities",
        "payload": {},
    }
    wire = (
        (b"x" * (limit + 1))
        + b"\n"
        + json.dumps(valid_request, separators=(",", ":")).encode()
        + b"\n"
    )
    completed = _run(wire)
    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["status"] for response in responses] == ["error", "ok"]
    assert responses[1]["requestId"] == "after-limit"
