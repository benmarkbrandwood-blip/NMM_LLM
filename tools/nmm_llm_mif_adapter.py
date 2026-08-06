#!/usr/bin/env python3
"""Run the independent NMM_LLM MIF-INTEROP/1 NDJSON adapter."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter  # noqa: E402
from learned_ai.interop.mif_v1.common import (  # noqa: E402
    MAX_INTEROP_REQUEST_BYTES,
    MifError,
    jcs_bytes,
    parse_ijson,
)


MAX_REQUEST_BYTES = MAX_INTEROP_REQUEST_BYTES


def _framing_error(message: str) -> None:
    print(f"nmm-llm-mif-adapter: {message}", file=sys.stderr)


def _error_response(error: MifError) -> dict[str, object]:
    return {
        "protocol": "MIF-INTEROP/1",
        "kind": "response",
        "requestId": "invalid-request",
        "operation": "capabilities",
        "status": "error",
        "diagnostics": error.as_diagnostics(),
    }


def _write_response(response: dict[str, object]) -> None:
    sys.stdout.buffer.write(jcs_bytes(response) + b"\n")
    sys.stdout.buffer.flush()


def _drain_oversized_line(first_chunk: bytes) -> int:
    """Consume one oversized record with bounded memory and count its payload."""

    actual = len(first_chunk)
    if first_chunk.endswith(b"\n"):
        return actual - 1
    while True:
        chunk = sys.stdin.buffer.readline(65_536)
        if chunk == b"":
            return actual
        actual += len(chunk)
        if chunk.endswith(b"\n"):
            return actual - 1


def main() -> int:
    adapter = MifInteropAdapter()
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 2)
        if line == b"":
            return 0
        payload_bytes = len(line) - 1 if line.endswith(b"\n") else len(line)
        if payload_bytes > MAX_REQUEST_BYTES:
            actual = _drain_oversized_line(line)
            _write_response(
                _error_response(
                    MifError(
                        "resource",
                        "resource-limit",
                        "interop-request-bytes resource limit exceeded",
                        resource_limit={
                            "name": "interop-request-bytes",
                            "limit": MAX_REQUEST_BYTES,
                            "actual": actual,
                        },
                    )
                )
            )
            continue
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            _framing_error("input must be LF-only terminated")
            return 2
        try:
            request = parse_ijson(line[:-1])
            response = adapter.handle(request)
        except MifError as exc:
            response = _error_response(exc)
        _write_response(response)


if __name__ == "__main__":
    raise SystemExit(main())
