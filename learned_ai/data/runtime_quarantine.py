"""Append-only quarantine storage for untrusted runtime game records."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


RUNTIME_QUARANTINE_SCHEMA = "nmm.runtime-quarantine.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


class RuntimeGameQuarantine:
    """Durably append raw runtime evidence without trusting it for training."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence, self._previous_sha256 = self._inspect_existing()

    def _inspect_existing(self) -> tuple[int, str | None]:
        if not self.path.exists():
            return 0, None
        previous = None
        sequence = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"invalid runtime quarantine record at line {line_number}"
                    ) from exc
                declared = record.pop("record_sha256", None)
                if record.get("schema_version") != RUNTIME_QUARANTINE_SCHEMA:
                    raise RuntimeError("unsupported runtime quarantine schema")
                if record.get("sequence") != sequence:
                    raise RuntimeError("runtime quarantine sequence is discontinuous")
                if record.get("previous_record_sha256") != previous:
                    raise RuntimeError("runtime quarantine hash chain is broken")
                observed = canonical_sha256(record)
                if declared != observed:
                    raise RuntimeError("runtime quarantine record hash is invalid")
                previous = observed
                sequence += 1
        return sequence, previous

    def append_game(
        self,
        game_record: dict[str, Any],
        *,
        source: str,
        reason_code: str = "unreviewed_runtime_game",
    ) -> str:
        """Append one untrusted game and return its quarantine record ID."""
        if not isinstance(game_record, dict):
            raise TypeError("game_record must be a dictionary")
        if not source or not reason_code:
            raise ValueError("source and reason_code must be non-empty")
        payload = json.loads(canonical_json_bytes(game_record))
        with self._lock:
            record_id = f"runtime-game:{uuid4().hex}"
            record = {
                "schema_version": RUNTIME_QUARANTINE_SCHEMA,
                "record_id": record_id,
                "sequence": self._sequence,
                "received_at_utc": _utc_now(),
                "source": source,
                "reason_code": reason_code,
                "trust_level": "quarantined_unreviewed",
                "allowed_consumers": ["telemetry", "explicit_snapshot_import"],
                "payload_sha256": canonical_sha256(payload),
                "payload": payload,
                "previous_record_sha256": self._previous_sha256,
            }
            record_sha256 = canonical_sha256(record)
            published = dict(record)
            published["record_sha256"] = record_sha256
            with self.path.open("ab") as handle:
                handle.write(canonical_json_bytes(published) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._previous_sha256 = record_sha256
            self._sequence += 1
            return record_id
