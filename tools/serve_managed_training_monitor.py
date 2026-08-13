"""Read-only local dashboard for one managed Generalist run.

The dashboard never imports trainer code, opens SQLite, or modifies
trainer-owned evidence. Every request re-reads immutable contracts and
append-only JSONL evidence from disk. Host GPU samples are written beneath the
selected run's ``local-monitor`` directory only during controller-confirmed
managed training windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MAX_CHART_POINTS = 700
GPU_SAMPLE_INTERVAL_SECONDS = 5.0
SOURCE_ROLLING_WINDOW = 200
SEGMENT_PATTERN = re.compile(r"^segment-(\d{4})$")
_GPU_SAMPLE_LOCK = threading.Lock()
_GPU_LAST_SAMPLE_MONOTONIC = 0.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    if not path.is_file():
        return rows, malformed
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A writer can be between bytes when the dashboard polls.
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    rows.append(value)
                else:
                    malformed += 1
    except OSError:
        malformed += 1
    return rows, malformed


def _event_payload(record: dict[str, Any]) -> dict[str, Any]:
    event = record.get("event")
    return event if isinstance(event, dict) else record


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _downsample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= MAX_CHART_POINTS:
        return rows
    stride = math.ceil(len(rows) / MAX_CHART_POINTS)
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def _tail_text(path: Path, limit: int = 20) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines[-limit:] if line.strip()]


def _redact_warning(line: str) -> str:
    """Hide machine-local absolute paths before a warning reaches the LAN."""
    return re.sub(r"[A-Za-z]:\\[^\r\n]*$", "<local-path>", line)


def _process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def _lock_pid(control_dir: Path) -> int | None:
    path = control_dir / "controller.lock"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not text.startswith("pid="):
        return None
    try:
        return int(text.removeprefix("pid="))
    except ValueError:
        return None


def _series_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "game",
        "difficulty",
        "temperature",
        "mixed_win_rate",
        "mixed_draw_rate",
        "mixed_window_games",
        "best_win_rate",
        "vs_frozen_score_rate_200",
        "vs_sanmill_score_rate_200",
        "policy_top1_rate",
        "heuristic_top1_rate",
        "malom_win_move_rate",
        "entropy_mean",
        "entropy_mean_smooth50",
        "chosen_prob_mean",
        "chosen_prob_mean_smooth50",
        "policy_top1_rate_smooth50",
        "heuristic_top1_rate_smooth50",
        "malom_win_move_rate_smooth50",
        "malom_known_rate_smooth50",
        "reward_total_mean",
        "reward_total_mean_smooth50",
        "reward_heuristic_mean",
        "reward_heuristic_mean_smooth50",
        "reward_retro_mean",
        "reward_retro_mean_smooth50",
        "ply",
        "ply_smooth50",
        "lr",
        "lr_smooth50",
        "lr_x1e4",
    )
    return {key: row.get(key) for key in keys}


def _update_series_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("game", "policy_loss", "value_loss", "entropy", "lr", "batch_steps")
    return {key: row.get(key) for key in keys}


def _learning_rate_step_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    """Keep exact LR change boundaries plus the final observed endpoint."""
    result: list[dict[str, float | int]] = []
    last_lr: float | None = None
    final_point: dict[str, float | int] | None = None
    for row in rows:
        game = _integer(row.get("game"))
        lr = _number(row.get("lr"))
        if game is None or lr is None:
            continue
        point: dict[str, float | int] = {
            "game": game,
            "lr_x1e4": lr * 10_000.0,
        }
        if last_lr is None or lr != last_lr:
            result.append(point)
        last_lr = lr
        final_point = point
    if final_point is not None and result[-1]["game"] != final_point["game"]:
        result.append(final_point)
    return result


def _chart_game_rows(
    rows: list[dict[str, Any]], *, rolling_win: int | None
) -> list[dict[str, Any]]:
    """Add display-only rolling metrics without altering trainer evidence."""
    smooth_keys = (
        "entropy_mean",
        "chosen_prob_mean",
        "policy_top1_rate",
        "heuristic_top1_rate",
        "malom_win_move_rate",
        "malom_unknown_rate",
        "reward_total_mean",
        "reward_heuristic_mean",
        "reward_retro_mean",
        "ply",
        "lr",
    )
    windows = {key: deque(maxlen=50) for key in smooth_keys}
    mixed_win_window: deque[float] = deque(maxlen=rolling_win or 1)
    mixed_draw_window: deque[float] = deque(maxlen=rolling_win or 1)
    source_score_windows = {
        "vs_frozen": deque(maxlen=SOURCE_ROLLING_WINDOW),
        "vs_sanmill": deque(maxlen=SOURCE_ROLLING_WINDOW),
    }
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for key, window in windows.items():
            value = _number(source.get(key))
            if value is not None:
                window.append(value)
            row[f"{key}_smooth50"] = (
                sum(window) / len(window) if len(window) == 50 else None
            )
        unknown_rate = _number(row.get("malom_unknown_rate_smooth50"))
        row["malom_known_rate_smooth50"] = (
            1.0 - unknown_rate if unknown_rate is not None else None
        )
        outcome = _number(source.get("outcome"))
        if rolling_win is not None:
            mixed_win_window.append(1.0 if outcome == 1.5 else 0.0)
            mixed_draw_window.append(0.0 if outcome in (1.5, -1.0) else 1.0)
        row["mixed_win_rate"] = (
            sum(mixed_win_window) / len(mixed_win_window)
            if rolling_win is not None and len(mixed_win_window) == rolling_win
            else None
        )
        row["mixed_draw_rate"] = (
            sum(mixed_draw_window) / len(mixed_draw_window)
            if rolling_win is not None and len(mixed_draw_window) == rolling_win
            else None
        )
        row["mixed_window_games"] = rolling_win
        if rolling_win is None or len(mixed_win_window) < rolling_win:
            row["best_win_rate"] = None
        source_name = str(source.get("game_type") or "")
        source_window = source_score_windows.get(source_name)
        if source_window is not None:
            source_window.append(
                1.0 if outcome == 1.5 else 0.0 if outcome == -1.0 else 0.5
            )
        for name, window in source_score_windows.items():
            row[f"{name}_score_rate_200"] = (
                sum(window) / len(window)
                if len(window) == SOURCE_ROLLING_WINDOW
                else None
            )
        smooth_lr = _number(row.get("lr_smooth50"))
        row["lr_x1e4"] = smooth_lr * 10_000.0 if smooth_lr is not None else None
        result.append(row)
    return result


_TERMINATION_KEYS = (
    "win_fewer_than_three",
    "win_no_legal_moves",
    "draw_repetition",
    "draw_no_progress",
    "max_ply_truncation",
    "lose_no_legal_moves",
    "lose_fewer_than_three",
    "other",
)


def _normalized_termination_reason(value: Any) -> str:
    reason = str(value or "other")
    reason = {
        "draw_threefold_repetition": "draw_repetition",
        "threefold-repetition": "draw_repetition",
        "draw_no_progress": "draw_no_progress",
        "draw_fifty_move": "draw_no_progress",
        "fifty-move": "draw_no_progress",
        "no-progress": "draw_no_progress",
        "max-ply-truncation": "max_ply_truncation",
        "max_ply": "max_ply_truncation",
        "DRAW_LONG": "max_ply_truncation",
    }.get(reason, reason)
    return reason if reason in _TERMINATION_KEYS else "other"


def _termination_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    window: deque[str] = deque(maxlen=50)
    result: list[dict[str, Any]] = []
    for row in rows:
        reason = _normalized_termination_reason(row.get("termination_reason"))
        window.append(reason)
        counts = Counter(window)
        if len(window) == 50:
            item: dict[str, Any] = {"game": row.get("game")}
            item.update({key: counts[key] / len(window) for key in _TERMINATION_KEYS})
            result.append(item)
    return _downsample(result)


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    terminations = Counter(
        _normalized_termination_reason(row.get("termination_reason"))
        for row in rows
    )
    for row in rows:
        outcome = _number(row.get("outcome"))
        if outcome == 1.5:
            counts["win"] += 1
        elif outcome == -1.0:
            counts["loss"] += 1
        else:
            counts["draw"] += 1
    total = sum(counts.values())
    return {
        "total": total,
        "win": counts["win"],
        "draw": counts["draw"],
        "loss": counts["loss"],
        "winRate": counts["win"] / total if total else None,
        "scoreRate": (
            (counts["win"] + 0.5 * counts["draw"]) / total if total else None
        ),
        "ruleDraw": (
            terminations["draw_repetition"] + terminations["draw_no_progress"]
        ),
        "maxPly": terminations["max_ply_truncation"],
    }


def _opponent_outcomes(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for source in sorted({str(row.get("game_type") or "unknown") for row in rows}):
        source_rows = [
            row for row in rows if str(row.get("game_type") or "unknown") == source
        ]
        colors = {}
        for color in ("W", "B"):
            color_rows = [
                row for row in source_rows if str(row.get("learner_color")) == color
            ]
            if color_rows:
                colors[color] = _outcome_summary(color_rows)
        by_source[source] = {
            "overall": _outcome_summary(source_rows),
            "recentSourceGames": {
                **_outcome_summary(source_rows[-SOURCE_ROLLING_WINDOW:]),
                "window": min(SOURCE_ROLLING_WINDOW, len(source_rows)),
                "configuredWindow": SOURCE_ROLLING_WINDOW,
            },
            "byLearnerColor": colors,
        }

    ladder_text = _trainer_option(plan, "--sanmill-node-ladder") or ""
    try:
        ladder = [int(value) for value in ladder_text.split(",") if value]
    except ValueError:
        ladder = []
    sanmill_rows = [
        row for row in rows if str(row.get("game_type")) == "vs_sanmill"
    ]
    by_level = []
    for level in sorted(
        {
            value
            for row in sanmill_rows
            if (value := _integer(row.get("difficulty"))) is not None
        }
    ):
        level_rows = [
            row for row in sanmill_rows if _integer(row.get("difficulty")) == level
        ]
        budgets = [
            value
            for row in level_rows
            if (value := _integer(row.get("opponent_node_budget"))) is not None
        ]
        by_level.append(
            {
                "level": level,
                "nodeBudget": (
                    Counter(budgets).most_common(1)[0][0]
                    if budgets
                    else (ladder[level - 1] if 0 < level <= len(ladder) else None)
                ),
                **_outcome_summary(level_rows),
            }
        )
    return {
        "overall": _outcome_summary(rows),
        "bySource": by_source,
        "sanmillByLevel": by_level,
    }


def _trainer_option(plan: dict[str, Any], option: str) -> str | None:
    args = plan.get("common_trainer_args")
    if not isinstance(args, list):
        return None
    try:
        index = args.index(option)
    except ValueError:
        return None
    return str(args[index + 1]) if index + 1 < len(args) else None


def _rolling_window_from_manifests(segment_dirs: list[Path]) -> int | None:
    """Read the trainer's real rolling window and reject segment drift."""
    observed: set[int] = set()
    for segment_dir in segment_dirs:
        manifest_path = segment_dir / "run-manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        resolved = manifest.get("resolved_config")
        value = resolved.get("rolling_win") if isinstance(resolved, dict) else None
        window = _integer(value)
        if window is None or window <= 0:
            raise ValueError(f"{manifest_path} has invalid resolved rolling_win")
        observed.add(window)
    if len(observed) > 1:
        raise ValueError(f"managed segments disagree on rolling_win: {sorted(observed)}")
    return next(iter(observed), None)


def _health_status(
    *,
    state: str,
    controller_alive: bool,
    authorization_matches: bool,
    observed_game: int,
    completed_games: int,
    max_games: int,
    completed_segments: int,
    segment_summaries: list[dict[str, Any]],
    games: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    warnings: list[str],
    warnings_superseded: bool,
    malformed: int,
) -> dict[str, Any]:
    """Return transparent infrastructure health, never a strength estimate."""
    stops: list[str] = []
    cautions: list[str] = []
    if not authorization_matches:
        stops.append("authorization_identity_mismatch")
    if state in {"failed", "stopped"}:
        stops.append("controller_reported_stop")
    if state == "running" and not controller_alive:
        stops.append("running_controller_offline")
    if malformed:
        cautions.append("malformed_jsonl_tail")

    numeric_game_fields = (
        "outcome",
        "temperature",
        "lr",
        "entropy_mean",
        "chosen_prob_mean",
    )
    numeric_update_fields = ("policy_loss", "value_loss", "entropy", "lr")
    if any(
        field in row and _number(row.get(field)) is None
        for row in games
        for field in numeric_game_fields
    ) or any(
        field in row and _number(row.get(field)) is None
        for row in updates
        for field in numeric_update_fields
    ):
        stops.append("non_finite_training_metric")

    completed_segment_rows = segment_summaries[:completed_segments]
    if any(
        not summary.get("checkpoint") or not summary.get("manifest")
        for summary in completed_segment_rows
    ):
        stops.append("completed_segment_missing_evidence")
    previous_last = 0
    for summary in segment_summaries:
        first = _integer(summary.get("firstGame"))
        last = _integer(summary.get("lastGame"))
        if first is None or last is None:
            continue
        if first != previous_last + 1:
            stops.append("segment_game_discontinuity")
            break
        previous_last = last

    if state == "completed":
        if observed_game != max_games or completed_games != max_games:
            stops.append("completion_count_mismatch")
        if len(games) != max_games:
            stops.append("completion_log_count_mismatch")

    expected_warning_prefixes = ("HumanDB Malom labels are disabled:",)
    unexpected_warnings = [
        warning
        for warning in warnings
        if not warning.startswith(expected_warning_prefixes)
    ]
    if unexpected_warnings and not warnings_superseded:
        cautions.append("unexpected_supervisor_warning")

    stops = list(dict.fromkeys(stops))
    cautions = list(dict.fromkeys(cautions))
    if stops:
        name = "stop"
    elif cautions:
        name = "warning"
    elif state == "completed":
        name = "complete"
    else:
        name = "healthy"
    return {
        "name": name,
        "stopReasons": stops,
        "cautionReasons": cautions,
        "unexpectedWarningCount": (
            0 if warnings_superseded else len(unexpected_warnings)
        ),
        "historicalWarningCount": (
            len(unexpected_warnings) if warnings_superseded else 0
        ),
        "scope": "infrastructure-only",
    }


def _managed_training_windows(
    events: list[dict[str, Any]],
) -> list[tuple[datetime, datetime | None]]:
    """Return managed-segment activity windows from the controller ledger."""
    windows: list[tuple[datetime, datetime | None]] = []
    active_start: datetime | None = None
    for event in events:
        event_type = str(event.get("event_type") or "")
        event_at = _timestamp(event.get("timestamp_utc"))
        if event_at is None:
            continue
        if event_type == "managed_segment_started":
            if active_start is not None and event_at >= active_start:
                windows.append((active_start, event_at))
            active_start = event_at
            continue
        closes_training_window = event_type in {
            "managed_segment_completed",
            "managed_segment_interrupted",
            "managed_plan_completed",
        } or any(token in event_type for token in ("failed", "stopped"))
        if (
            active_start is not None
            and closes_training_window
            and event_at >= active_start
        ):
            windows.append((active_start, event_at))
            active_start = None
    if active_start is not None:
        windows.append((active_start, None))
    return windows


def _gpu_rows_in_training_windows(
    rows: list[dict[str, Any]],
    training_windows: list[tuple[datetime, datetime | None]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_at = _timestamp(row.get("timestampUtc"))
        if sample_at is None:
            continue
        if any(
            sample_at >= start and (end is None or sample_at <= end)
            for start, end in training_windows
        ):
            result.append(row)
    return result


def _gpu_status(
    control_dir: Path,
    observed_game: int,
    *,
    training_windows: list[tuple[datetime, datetime | None]],
    sampling_active: bool,
) -> dict[str, Any]:
    """Collect whole-device telemetry only during managed training windows."""
    global _GPU_LAST_SAMPLE_MONOTONIC

    telemetry_path = control_dir / "local-monitor" / "gpu-telemetry.jsonl"
    sample_error: str | None = None
    now_monotonic = time.monotonic()
    with _GPU_SAMPLE_LOCK:
        if (
            sampling_active
            and now_monotonic - _GPU_LAST_SAMPLE_MONOTONIC
            >= GPU_SAMPLE_INTERVAL_SECONDS
        ):
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi.exe",
                        "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                first_line = next(
                    line for line in completed.stdout.splitlines() if line.strip()
                )
                fields = [field.strip() for field in first_line.split(",")]
                if len(fields) != 6:
                    raise ValueError("unexpected nvidia-smi field count")
                memory_used = float(fields[4])
                memory_total = float(fields[5])
                sample = {
                    "game": observed_game,
                    "timestampUtc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "gpuUtilPct": float(fields[3]),
                    "memoryUsedMiB": memory_used,
                    "memoryTotalMiB": memory_total,
                    "memoryUtilPct": (
                        100.0 * memory_used / memory_total if memory_total else 0.0
                    ),
                    "telemetryScope": "managed-training-window-whole-device",
                }
                telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
                _GPU_LAST_SAMPLE_MONOTONIC = now_monotonic
            except (OSError, subprocess.SubprocessError, StopIteration, ValueError) as exc:
                sample_error = f"{type(exc).__name__}: {exc}"

        rows, malformed = _read_jsonl(telemetry_path)

    training_rows = _gpu_rows_in_training_windows(rows, training_windows)
    latest = training_rows[-1] if training_rows else {}
    return {
        "available": bool(training_rows),
        "sampleIntervalSeconds": GPU_SAMPLE_INTERVAL_SECONDS,
        "scope": "managed-training-window-whole-device",
        "samplingActive": sampling_active,
        "sampleError": sample_error,
        "malformedLinesIgnored": malformed,
        "sampleCount": len(training_rows),
        "excludedOutsideTrainingWindow": len(rows) - len(training_rows),
        "latest": latest,
        "series": _downsample(training_rows),
    }


def collect_status(control_dir: Path) -> dict[str, Any]:
    plan = _read_json(control_dir / "plan.json")
    authorization = _read_json(control_dir / "authorization.json")
    event_records, event_malformed = _read_jsonl(
        control_dir / "controller-events.jsonl"
    )
    events = [_event_payload(record) for record in event_records]

    all_games: dict[int, dict[str, Any]] = {}
    all_updates: dict[int, dict[str, Any]] = {}
    segment_summaries: list[dict[str, Any]] = []
    malformed = event_malformed
    segments_root = control_dir / "segments"
    segment_dirs = []
    if segments_root.is_dir():
        segment_dirs = sorted(
            (
                path
                for path in segments_root.iterdir()
                if path.is_dir() and SEGMENT_PATTERN.match(path.name)
            ),
            key=lambda path: path.name,
        )

    recovered_segment_indexes = {
        _integer(event.get("details", {}).get("segment_index"))
        for event in events
        if event.get("event_type") == "managed_segment_started"
        and event.get("details", {}).get("recovery") is True
    }
    completed_segment_indexes = {
        _integer(event.get("details", {}).get("segment_index"))
        for event in events
        if event.get("event_type") == "managed_segment_completed"
    }
    recovery_prefixes: dict[int, tuple[Path, int]] = {}
    quarantine_root = (control_dir / "quarantine").resolve()
    for event in events:
        if event.get("event_type") != "managed_segment_interrupted":
            continue
        details = event.get("details", {})
        segment_index = _integer(details.get("segment_index"))
        resume_game_count = _integer(details.get("resume_game_count"))
        incomplete_output = details.get("incomplete_output")
        if (
            segment_index is None
            or segment_index not in recovered_segment_indexes
            or segment_index not in completed_segment_indexes
            or resume_game_count is None
            or resume_game_count <= 0
            or not isinstance(incomplete_output, str)
        ):
            continue
        try:
            prefix_dir = Path(incomplete_output).resolve(strict=True)
        except OSError:
            continue
        if not prefix_dir.is_relative_to(quarantine_root):
            continue
        recovery_prefixes[segment_index] = (prefix_dir, resume_game_count)

    for segment in segment_dirs:
        match = SEGMENT_PATTERN.match(segment.name)
        segment_index = int(match.group(1)) if match is not None else None
        game_rows, bad_games = _read_jsonl(segment / "train_log.jsonl")
        update_rows, bad_updates = _read_jsonl(segment / "update_log.jsonl")
        malformed += bad_games + bad_updates
        recovered_game_rows: list[dict[str, Any]] = []
        recovered_update_rows: list[dict[str, Any]] = []
        recovered_prefix_name: str | None = None
        if segment_index in recovery_prefixes:
            prefix_dir, resume_game_count = recovery_prefixes[segment_index]
            prefix_games, bad_prefix_games = _read_jsonl(
                prefix_dir / "train_log.jsonl"
            )
            prefix_updates, bad_prefix_updates = _read_jsonl(
                prefix_dir / "update_log.jsonl"
            )
            malformed += bad_prefix_games + bad_prefix_updates
            recovered_game_rows = [
                row
                for row in prefix_games
                if (_integer(row.get("game")) or 0) <= resume_game_count
            ]
            recovered_update_rows = [
                row
                for row in prefix_updates
                if (_integer(row.get("game")) or 0) <= resume_game_count
            ]
            recovered_prefix_name = prefix_dir.name

        logical_game_rows: dict[int, dict[str, Any]] = {}
        for row in (*recovered_game_rows, *game_rows):
            game = _integer(row.get("game"))
            if game is not None:
                logical_game_rows[game] = row
        logical_update_rows: dict[int, dict[str, Any]] = {}
        for row in (*recovered_update_rows, *update_rows):
            game = _integer(row.get("game"))
            if game is not None:
                logical_update_rows[game] = row

        game_numbers: list[int] = []
        for game, row in sorted(logical_game_rows.items()):
            all_games[game] = row
            game_numbers.append(game)
        update_numbers: list[int] = []
        for game, row in sorted(logical_update_rows.items()):
            all_updates[game] = row
            update_numbers.append(game)
        segment_summaries.append(
            {
                "name": segment.name,
                "gameRows": len(logical_game_rows),
                "updateRows": len(logical_update_rows),
                "firstGame": min(game_numbers) if game_numbers else None,
                "lastGame": max(game_numbers) if game_numbers else None,
                "checkpoint": (segment / "latest.pt").is_file(),
                "manifest": (segment / "run-manifest.json").is_file(),
                "recoveredPrefixRows": len(recovered_game_rows),
                "recoveredPrefix": recovered_prefix_name,
            }
        )

    games = [all_games[key] for key in sorted(all_games)]
    updates = [all_updates[key] for key in sorted(all_updates)]
    latest = games[-1] if games else {}
    latest_update = updates[-1] if updates else {}

    max_games = int(plan.get("max_games", 0))
    completed_events = [
        event
        for event in events
        if event.get("event_type") == "managed_segment_completed"
    ]
    completed_games = max(
        (
            int(event.get("details", {}).get("completed_games", 0))
            for event in completed_events
        ),
        default=0,
    )
    completed_segments = len(completed_events)
    completed_seconds = sum(
        float(event.get("details", {}).get("elapsed_seconds", 0.0))
        for event in completed_events
    )
    observed_game = max(
        completed_games,
        _integer(latest.get("game")) or 0,
        _integer(latest_update.get("game")) or 0,
    )

    last_event = events[-1] if events else {}
    last_event_type = str(last_event.get("event_type", "unknown"))
    last_status = str(last_event.get("status", "unknown"))
    state = last_status
    if completed_games >= max_games > 0:
        state = "completed"
    elif last_event_type == "managed_segment_started":
        state = "running"
    elif last_event_type == "managed_segment_completed":
        state = "between_segments"
    elif "failed" in last_event_type or "stopped" in last_status:
        state = "stopped"

    active_elapsed = 0.0
    if state == "running":
        started = _timestamp(last_event.get("timestamp_utc"))
        if started is not None:
            active_elapsed = max(
                0.0,
                (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds(),
            )

    outcomes = Counter()
    terminations = Counter()
    opponents = Counter()
    for row in games:
        outcome = _number(row.get("outcome"))
        if outcome == 1.5:
            outcomes["win"] += 1
        elif outcome == -1.0:
            outcomes["loss"] += 1
        else:
            outcomes["draw"] += 1
        terminations[str(row.get("termination_reason") or "unknown")] += 1
        opponents[str(row.get("game_type") or "unknown")] += 1

    pid = _lock_pid(control_dir)
    controller_alive = _process_exists(pid)
    stderr_path = control_dir / "supervisor.stderr.log"
    warnings = [
        _redact_warning(line)
        for line in _tail_text(stderr_path)
    ]
    warnings_superseded = False
    last_event_timestamp = _timestamp(last_event.get("timestamp_utc"))
    if state == "completed" and warnings and last_event_timestamp is not None:
        try:
            stderr_timestamp = datetime.fromtimestamp(
                stderr_path.stat().st_mtime,
                timezone.utc,
            )
        except OSError:
            pass
        else:
            warnings_superseded = stderr_timestamp <= last_event_timestamp
    rolling_win = _rolling_window_from_manifests(segment_dirs)
    chart_games = _chart_game_rows(games, rolling_win=rolling_win)
    learning_rate_steps = _learning_rate_step_rows(games)
    latest_chart = chart_games[-1] if chart_games else {}
    sampled_games = _downsample(chart_games)
    sampled_updates = _downsample(updates)
    progress = (100.0 * observed_game / max_games) if max_games else 0.0
    gpu = _gpu_status(
        control_dir,
        observed_game,
        training_windows=_managed_training_windows(events),
        sampling_active=state == "running" and controller_alive,
    )
    segment_markers = sorted(
        {
            int(event.get("details", {}).get("completed_games", 0))
            for event in completed_events
            if int(event.get("details", {}).get("completed_games", 0)) > 0
        }
    )
    resource_markers: list[int] = []
    stage_text = _trainer_option(plan, "--sanmill-stage-games")
    if stage_text:
        try:
            cumulative = 0
            stage_values = [int(value) for value in stage_text.split(",")]
            for value in stage_values[:-1]:
                cumulative += value
                resource_markers.append(cumulative)
        except ValueError:
            resource_markers = []
    ladder_text = _trainer_option(plan, "--sanmill-node-ladder") or ""
    try:
        node_ladder = [int(value) for value in ladder_text.split(",") if value]
    except ValueError:
        node_ladder = []
    current_level = _integer(latest.get("difficulty"))
    current_node_budget = (
        node_ladder[current_level - 1]
        if current_level is not None and 0 < current_level <= len(node_ladder)
        else None
    )
    next_resource_game = next(
        (game for game in resource_markers if game > observed_game),
        None,
    )

    authorization_matches = authorization.get("plan_sha256") == plan.get(
        "plan_sha256"
    )
    opponent_outcomes = _opponent_outcomes(games, plan)
    health = _health_status(
        state=state,
        controller_alive=controller_alive,
        authorization_matches=authorization_matches,
        observed_game=observed_game,
        completed_games=completed_games,
        max_games=max_games,
        completed_segments=completed_segments,
        segment_summaries=segment_summaries,
        games=games,
        updates=updates,
        warnings=warnings,
        warnings_superseded=warnings_superseded,
        malformed=malformed,
    )

    return {
        "schema": "nmm-local-training-monitor-v3",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "identity": {
            "planId": plan.get("plan_id"),
            "planSha256": plan.get("plan_sha256"),
            "experimentId": plan.get("experiment_id"),
            "gitCommit": plan.get("git_commit"),
            "authorizationMatches": authorization_matches,
        },
        "state": {
            "name": state,
            "lastEventType": last_event_type,
            "lastEventStatus": last_status,
            "controllerPid": pid,
            "controllerAlive": controller_alive,
            "completedGames": completed_games,
            "observedGame": observed_game,
            "maxGames": max_games,
            "progressPct": progress,
            "completedSegments": completed_segments,
            "observedSegments": len(segment_dirs),
            "segmentGames": plan.get("segment_games"),
            "completedActiveHours": completed_seconds / 3600.0,
            "estimatedActiveHours": (completed_seconds + active_elapsed) / 3600.0,
            "maxWallHours": plan.get("max_wall_hours"),
            "currentSegment": segment_dirs[-1].name if segment_dirs else None,
            "currentNodeBudget": current_node_budget,
            "nextResourceGame": next_resource_game,
            "rollingWin": rolling_win,
            "sourceRollingWindow": SOURCE_ROLLING_WINDOW,
        },
        "latest": {
            "game": latest.get("game"),
            "difficulty": latest.get("difficulty"),
            "temperature": latest.get("temperature"),
            "mixedWinRate": latest_chart.get("mixed_win_rate"),
            "bestWinRate": latest.get("best_win_rate"),
            "ply": latest.get("ply"),
            "terminationReason": latest.get("termination_reason"),
            "gameType": latest.get("game_type"),
            "policyTop1": latest.get("policy_top1_rate"),
            "heuristicTop1": latest.get("heuristic_top1_rate"),
            "malomWinMove": latest.get("malom_win_move_rate"),
            "entropy": latest.get("entropy_mean"),
            "chosenProbability": latest.get("chosen_prob_mean"),
            "lr": latest.get("lr"),
            "updateGame": latest_update.get("game"),
            "policyLoss": latest_update.get("policy_loss"),
            "valueLoss": latest_update.get("value_loss"),
            "updateEntropy": latest_update.get("entropy"),
            "mixedDrawRate": latest_chart.get("mixed_draw_rate"),
            "rewardTotal": latest_chart.get("reward_total_mean_smooth50"),
            "rewardHeuristic": latest_chart.get(
                "reward_heuristic_mean_smooth50"
            ),
            "rewardRetro": latest_chart.get("reward_retro_mean_smooth50"),
        },
        "components": {
            "sentinel": "disabled",
            "recovery": "disabled",
            "restore": "disabled",
            "resurrect": "disabled",
            "curriculum": "fixed-resource",
        },
        "health": health,
        "opponentOutcomes": opponent_outcomes,
        "counts": {
            "loggedGames": len(games),
            "updates": len(updates),
            "outcomes": dict(outcomes),
            "terminations": dict(terminations.most_common()),
            "opponents": dict(opponents.most_common()),
            "malformedLinesIgnored": malformed,
            "warnings": len(warnings),
            "warningsSuperseded": warnings_superseded,
        },
        "segments": segment_summaries,
        "warnings": warnings,
        "gpu": gpu,
        "markers": {
            "segments": segment_markers,
            "resources": resource_markers,
        },
        "series": {
            "games": [_series_row(row) for row in sampled_games],
            "updates": [_update_series_row(row) for row in sampled_updates],
            "learningRate": learning_rate_steps,
            "terminations50": _termination_series(games),
        },
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>NMM_LLM 训练监控</title>
  <style>
    :root { color-scheme: dark; --bg:#07111f; --panel:#0e1b2c; --line:#23344a;
      --text:#e7eef8; --muted:#91a4bd; --green:#45d483; --blue:#55a7ff;
      --orange:#ffb454; --red:#ff6b75; --violet:#bd8cff; --cyan:#58d6d6; }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(150deg,#07111f,#0a1422 55%,#07111f);
      color:var(--text); font:14px/1.45 system-ui,"Segoe UI",sans-serif; }
    main { max-width:1500px; margin:auto; padding:22px; }
    header { display:flex; justify-content:space-between; gap:18px; align-items:flex-start;
      margin-bottom:18px; }
    h1 { font-size:24px; margin:0 0 5px; letter-spacing:.2px; }
    .muted { color:var(--muted); }
    .header-actions { display:flex; align-items:center; justify-content:flex-end; gap:10px;
      flex-wrap:wrap; }
    .language-select { min-width:118px; padding:7px 30px 7px 10px; border:1px solid var(--line);
      border-radius:10px; color:var(--text); background:#0b1727; cursor:pointer; font:inherit; }
    .language-select:hover,.language-select:focus-visible { border-color:var(--blue); outline:none;
      box-shadow:0 0 0 3px rgba(85,167,255,.15); }
    .action-button { padding:7px 11px; border:1px solid #39506c; border-radius:10px;
      color:var(--text); background:#122238; cursor:pointer; font:inherit; }
    .action-button:hover,.action-button:focus-visible { border-color:var(--blue); outline:none;
      box-shadow:0 0 0 3px rgba(85,167,255,.15); }
    .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
      overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
    .badge { display:inline-flex; align-items:center; gap:7px; padding:7px 11px;
      border:1px solid var(--line); border-radius:999px; background:#0b1727; }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); }
    .running .dot { background:var(--cyan); box-shadow:0 0 12px var(--cyan); }
    .completed .dot { background:var(--blue); }
    .between_segments .dot { background:var(--orange); }
    .stopped .dot { background:var(--magenta,#cc79a7); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
    .card,.panel { background:rgba(14,27,44,.94); border:1px solid var(--line);
      border-radius:13px; box-shadow:0 8px 30px rgba(0,0,0,.18); }
    .card { padding:13px 14px; min-height:84px; position:relative; }
    .card.health-healthy { border-color:rgba(86,180,233,.75); }
    .card.health-complete { border-color:rgba(85,167,255,.7); }
    .card.health-warning { border-color:rgba(255,180,84,.75); }
    .card.health-stop { border-color:rgba(204,121,167,.9); }
    .card-head,.panel-head { display:flex; align-items:flex-start; justify-content:space-between;
      gap:10px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.55px; }
    .value { margin-top:5px; font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
    .sub { margin-top:2px; color:var(--muted); font-size:12px; }
    .progress { height:12px; margin:15px 0 20px; overflow:hidden; background:#101f32;
      border:1px solid var(--line); border-radius:999px; }
    .progress > div { height:100%; width:0; background:linear-gradient(90deg,var(--blue),var(--green));
      transition:width .35s ease; }
    .evidence-key { display:flex; flex-wrap:wrap; gap:8px 16px; margin:-8px 2px 17px;
      color:var(--muted); font-size:12px; }
    .evidence-key span::before { content:'●'; margin-right:6px; color:var(--blue); }
    .evidence-key span:last-child::before { content:'┊'; color:#91a4bd; }
    .grid { display:grid; grid-template-columns:minmax(0,1fr); gap:12px; margin-top:12px; }
    .panel { padding:14px; min-width:0; position:relative; }
    .panel-head { margin-bottom:8px; }
    .panel h2 { font-size:15px; margin:0; font-weight:600; }
    .help-button { flex:0 0 auto; width:24px; height:24px; display:inline-grid;
      place-items:center; padding:0; border:1px solid #39506c; border-radius:50%;
      color:#b9c9dc; background:#122238; cursor:pointer; font:700 13px/1 system-ui,sans-serif; }
    .help-button:hover,.help-button:focus-visible { color:#fff; border-color:var(--blue);
      background:#173252; outline:none; box-shadow:0 0 0 3px rgba(85,167,255,.15); }
    canvas { width:100%; height:230px; display:block; }
    .wide { grid-column:1 / -1; }
    .bars { display:grid; gap:8px; }
    .bar-row { display:grid; grid-template-columns:minmax(110px,1fr) 4fr 55px; gap:9px; align-items:center; }
    .bar-track { height:9px; border-radius:999px; overflow:hidden; background:#18283b; }
    .bar-fill { height:100%; border-radius:999px; background:var(--blue); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th,td { padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    .sub-row td:first-child { padding-left:26px; color:var(--muted); }
    .table-note { margin:8px 2px 13px; color:var(--muted); font-size:12px; }
    .table-section-title { margin:16px 0 6px; color:#bcd7f8; font-size:12px;
      text-transform:uppercase; letter-spacing:.55px; }
    .warning { padding:9px 10px; color:#ffd59b; background:#2b2215; border:1px solid #59401f;
      border-radius:8px; overflow-wrap:anywhere; }
    footer { display:flex; justify-content:space-between; gap:12px; margin:18px 2px 4px;
      color:var(--muted); font-size:12px; }
    .help-panel { position:fixed; z-index:100; top:0; right:0; bottom:0; width:min(410px,36vw);
      overflow:auto; padding:20px; border-left:1px solid #304762; background:#0d1b2d;
      box-shadow:-18px 0 55px rgba(0,0,0,.38); }
    .help-panel[hidden] { display:none; }
    body.help-open main { max-width:calc(100vw - 410px); margin-left:0; margin-right:410px; }
    .help-active { outline:2px solid var(--blue); outline-offset:2px; }
    .modal-head { display:flex; align-items:flex-start; justify-content:space-between;
      gap:18px; padding-bottom:12px; border-bottom:1px solid var(--line); }
    .modal-head h2 { margin:0; font-size:19px; }
    .modal-close { width:30px; height:30px; border:1px solid var(--line); border-radius:8px;
      color:var(--text); background:#13243a; cursor:pointer; font:22px/1 system-ui,sans-serif; }
    .modal-close:hover,.modal-close:focus-visible { border-color:var(--blue); outline:none; }
    .help-sections { display:grid; gap:14px; margin-top:15px; }
    .help-section { padding:12px 13px; border:1px solid var(--line); border-radius:10px;
      background:#0a1727; }
    .help-section h3 { margin:0 0 5px; color:#bcd7f8; font-size:12px;
      text-transform:uppercase; letter-spacing:.55px; }
    .help-section p { margin:0; color:#d6e1ef; }
    @media (max-width:850px) { .grid { grid-template-columns:1fr; } .wide { grid-column:auto; }
      header { flex-direction:column; } .header-actions { width:100%; justify-content:space-between; }
      footer { flex-direction:column; } .help-panel { width:min(92vw,410px); }
      body.help-open main { max-width:1500px; margin:auto; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1 data-i18n="pageTitle">NMM_LLM · Generalist 受管训练</h1><div id="identity" class="muted" data-i18n="identityLoading">正在读取计划身份…</div></div>
    <div class="header-actions">
      <label><span class="sr-only" data-i18n="languageLabel">界面语言</span><select id="languageSelect" class="language-select" data-i18n-aria="languageLabel"><option value="zh">中文</option><option value="en">English</option></select></label>
      <button type="button" id="exportPng" class="action-button" data-i18n="exportPng">导出 PNG</button>
      <div id="stateBadge" class="badge"><span class="dot"></span><span id="stateText" data-i18n="connecting">连接中</span></div>
    </div>
  </header>
  <section class="cards">
    <div id="healthCard" class="card"><div class="card-head"><div class="label" data-i18n="cardHealth">训练健康度</div><button type="button" class="help-button" data-help-key="health">?</button></div><div id="health" class="value">—</div><div id="healthSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardGames">观测局数</div><button type="button" class="help-button" data-help-key="games">?</button></div><div id="games" class="value">—</div><div id="gamesSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardDifficulty">难度</div><button type="button" class="help-button" data-help-key="difficulty">?</button></div><div id="difficulty" class="value">—</div><div id="temp" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardFrozenRecent">冻结臂 · 最近 200 来源局</div><button type="button" class="help-button" data-help-key="sourceRecent">?</button></div><div id="frozenRecent" class="value">—</div><div id="frozenRecentSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardSanmillRecent">Sanmill · 最近 200 来源局</div><button type="button" class="help-button" data-help-key="sourceRecent">?</button></div><div id="sanmillRecent" class="value">—</div><div id="sanmillRecentSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div id="trainingWindowLabel" class="label" data-i18n="cardTrainingWindow">混合训练窗口</div><button type="button" class="help-button" data-help-key="trainingWindow">?</button></div><div id="mixedWindow" class="value">—</div><div id="mixedWindowSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardRuleDraws">规则和棋</div><button type="button" class="help-button" data-help-key="terminationSplit">?</button></div><div id="ruleDraws" class="value">—</div><div id="ruleDrawsSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardMaxPly">max-ply 截断</div><button type="button" class="help-button" data-help-key="terminationSplit">?</button></div><div id="maxPly" class="value">—</div><div id="maxPlySub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardLatestUpdate">最近更新</div><button type="button" class="help-button" data-help-key="latestUpdate">?</button></div><div id="update" class="value">—</div><div id="loss" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardSegments">分段</div><button type="button" class="help-button" data-help-key="segments">?</button></div><div id="segments" class="value">—</div><div id="segmentsSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardActiveTime">Active time</div><button type="button" class="help-button" data-help-key="activeTime">?</button></div><div id="hours" class="value">—</div><div id="hoursSub" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardGpu">训练期 GPU 遥测</div><button type="button" class="help-button" data-help-key="gpuCard">?</button></div><div id="gpuCurrent" class="value">—</div><div id="gpuMemory" class="sub">—</div></div>
    <div class="card"><div class="card-head"><div class="label" data-i18n="cardComponents">实验开关</div><button type="button" class="help-button" data-help-key="components">?</button></div><div id="components" class="value">—</div><div id="componentsSub" class="sub">—</div></div>
  </section>
  <div class="progress"><div id="progressBar"></div></div>
  <div class="evidence-key"><span data-i18n="observedEvidenceKey">曲线和条形：仅已观测日志/遥测</span><span data-i18n="planBoundaryKey">虚线与上限：冻结计划边界，不是预测</span></div>

  <section class="grid">
    <div class="panel wide"><div class="panel-head"><h2 data-i18n="panelOpponentOutcomes">按对手来源的胜 / 和 / 负</h2><button type="button" class="help-button" data-help-key="opponentOutcomes">?</button></div><div class="table-note" data-i18n="opponentOutcomeNote">规则和棋与 max-ply 截断分别计数；得分率仅为训练诊断。</div><div style="overflow:auto"><table><thead><tr><th data-i18n="tableSource">来源</th><th data-i18n="tableGames">局数</th><th data-i18n="tableWins">胜</th><th data-i18n="tableDraws">和</th><th data-i18n="tableRuleDraws">规则和棋</th><th data-i18n="tableMaxPly">max-ply</th><th data-i18n="tableLosses">负</th><th data-i18n="tableScoreRate">得分率</th></tr></thead><tbody id="sourceOutcomeRows"></tbody></table></div><div class="table-section-title" data-i18n="sanmillByLevel">Sanmill 按节点档位</div><div class="table-note" data-i18n="nodeTimingNote">参考搜索耗时为本机持久进程的校准中位数 / P90，仅包含 Sanmill 搜索。</div><div style="overflow:auto"><table><thead><tr><th data-i18n="tableLevel">级别</th><th data-i18n="tableNodes">节点上限</th><th data-i18n="tableSearchTime">参考毫秒（中位 / P90）</th><th data-i18n="tableGames">局数</th><th data-i18n="tableWins">胜</th><th data-i18n="tableDraws">和</th><th data-i18n="tableRuleDraws">规则和棋</th><th data-i18n="tableMaxPly">max-ply</th><th data-i18n="tableLosses">负</th><th data-i18n="tableScoreRate">得分率</th></tr></thead><tbody id="sanmillLevelRows"></tbody></table></div></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelOutcomes">胜 / 和 / 负</h2><button type="button" class="help-button" data-help-key="outcomes">?</button></div><div id="outcomeBars" class="bars"></div></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelOpponents">对手来源</h2><button type="button" class="help-button" data-help-key="opponents">?</button></div><div id="opponentBars" class="bars"></div></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelWinTrend">胜率趋势</h2><button type="button" class="help-button" data-help-key="winTrend">?</button></div><canvas id="winChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelDifficultyPly">难度与对局长度</h2><button type="button" class="help-button" data-help-key="difficultyPly">?</button></div><canvas id="depthChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelTop1">策略、启发式与 Malom Top-1</h2><button type="button" class="help-button" data-help-key="top1">?</button></div><canvas id="topChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelEntropy">Entropy 与选择概率（50 局平滑）</h2><button type="button" class="help-button" data-help-key="entropyTrend">?</button></div><canvas id="entropyChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelExploration">温度与选择概率</h2><button type="button" class="help-button" data-help-key="exploration">?</button></div><canvas id="exploreChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelRewards">奖励信号（50 局平滑）</h2><button type="button" class="help-button" data-help-key="rewards">?</button></div><canvas id="rewardChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelLosses">Policy / Value loss</h2><button type="button" class="help-button" data-help-key="losses">?</button></div><canvas id="lossChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelLr">Learning rate × 10⁴（实际值·阶梯）</h2><button type="button" class="help-button" data-help-key="learningRate">?</button></div><div id="lrChartNote" class="table-note">原始执行值；阶梯线不做平滑或线性插值。</div><canvas id="lrChart"></canvas></div>
    <div class="panel wide"><div class="panel-head"><h2 data-i18n="panelTerminationTrend">终止原因构成（滚动 50 局）</h2><button type="button" class="help-button" data-help-key="terminationTrend">?</button></div><canvas id="terminationChart"></canvas></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelTerminations">终止原因</h2><button type="button" class="help-button" data-help-key="terminations">?</button></div><div id="terminationBars" class="bars"></div></div>
    <div class="panel"><div class="panel-head"><h2 data-i18n="panelGpu">训练期间 GPU 与显存遥测（整卡）</h2><button type="button" class="help-button" data-help-key="gpuTrend">?</button></div><div class="table-note" data-i18n="gpuTrainingNote">仅接受受管分段运行窗口内的整卡样本；不是训练进程独占读数。</div><canvas id="gpuChart"></canvas></div>
    <div class="panel wide"><div class="panel-head"><h2 data-i18n="panelSegmentEvidence">分段证据</h2><button type="button" class="help-button" data-help-key="segmentEvidence">?</button></div><div style="overflow:auto"><table><thead><tr><th data-i18n="tableSegment">分段</th><th data-i18n="tableFirstGame">首局</th><th data-i18n="tableLastGame">末局</th><th data-i18n="tableGameRows">对局行</th><th data-i18n="tableUpdates">更新</th><th data-i18n="tableCheckpoint">Checkpoint</th></tr></thead><tbody id="segmentRows"></tbody></table></div></div>
    <div class="panel wide"><div class="panel-head"><h2 data-i18n="panelWarnings">警告（只读）</h2><button type="button" class="help-button" data-help-key="warnings">?</button></div><div id="warnings" class="muted" data-i18n="none">无</div></div>
  </section>
  <footer><span id="refresh">—</span><span data-i18n="footer">只读监控面板 · 每 5 秒刷新 · 不控制训练</span></footer>
</main>
<aside id="helpPanel" class="help-panel" role="complementary" aria-labelledby="helpTitle" hidden>
  <article>
    <div class="modal-head"><h2 id="helpTitle">—</h2><button type="button" id="helpClose" class="modal-close" aria-label="关闭">×</button></div>
    <div class="help-sections">
      <section class="help-section"><h3 data-i18n="helpPurpose">作用</h3><p id="helpPurposeText"></p></section>
      <section class="help-section"><h3 data-i18n="helpRead">怎么看</h3><p id="helpReadText"></p></section>
      <section class="help-section"><h3 data-i18n="helpExpected">预期趋势</h3><p id="helpExpectedText"></p></section>
      <section class="help-section"><h3 data-i18n="helpWatch">需要注意</h3><p id="helpWatchText"></p></section>
    </div>
  </article>
</aside>
<script>
const COLORS = {blue:'#56b4e9',green:'#009e73',orange:'#e69f00',red:'#cc79a7',violet:'#b794f4',cyan:'#00d5e7',yellow:'#f0e442',magenta:'#cc79a7',gpu:'#00d5e7',vram:'#ff9f43'};
const OUTCOME_BAR_ORDER=Object.freeze(['win','draw','loss']);
const NODE_TIMING_MS = Object.freeze({1000:[0.21,0.30],5000:[0.63,0.80],25000:[2.44,2.85],100000:[9.91,11.34],500000:[52.85,60.77]});
const I18N = {
  zh: {
    documentTitle:'NMM_LLM 训练监控', pageTitle:'NMM_LLM · Generalist 受管训练',
    identityLoading:'正在读取计划身份…', connecting:'连接中', languageLabel:'界面语言', exportPng:'导出 PNG', exportFailed:'PNG 导出失败',
    cardHealth:'训练健康度', cardGames:'观测局数', cardSegments:'分段', cardActiveTime:'活跃运行时间', cardDifficulty:'节点级别',
    cardFrozenRecent:'冻结臂 · 最近 200 来源局', cardSanmillRecent:'Sanmill · 最近 200 来源局', cardTrainingWindow:'混合训练窗口', cardRuleDraws:'规则和棋', cardMaxPly:'max-ply 截断', cardLatestUpdate:'最近更新', cardGpu:'训练期 GPU 遥测', cardComponents:'实验开关', panelWinTrend:'按对手来源的 200 局得分率趋势',
    panelTop1:'策略、启发式与 Malom Top-1（50 局平滑）', panelExploration:'温度与选择概率',
    panelLosses:'Policy / Value loss', panelEntropy:'Entropy（50 局平滑）', panelRewards:'奖励信号（50 局平滑）', panelLr:'Learning rate × 10⁴（实际值·阶梯）', lrChartNote:'原始执行值；阶梯线不做平滑或线性插值。', panelDifficultyPly:'对局长度（50 局平均）', panelGpu:'训练期间 GPU 与显存遥测（整卡）', gpuTrainingNote:'仅接受受管分段运行窗口内的整卡样本；不是训练进程独占读数。', panelTerminationTrend:'终止原因构成（滚动 50 局）', panelTerminations:'终止原因',
    panelOpponentOutcomes:'按对手来源的胜 / 和 / 负', opponentOutcomeNote:'规则和棋与 max-ply 截断分别计数；得分率仅为训练诊断。', sanmillByLevel:'Sanmill 按节点档位', nodeTimingNote:'参考搜索耗时为本机持久进程的校准中位数 / P90，仅包含 Sanmill 搜索。',
    panelOutcomes:'全部胜 / 和 / 负', panelOpponents:'对手来源', panelSegmentEvidence:'分段证据',
    panelWarnings:'警告（只读）', tableSegment:'分段', tableFirstGame:'首局', tableLastGame:'末局',
    tableGameRows:'对局行', tableUpdates:'更新', tableCheckpoint:'Checkpoint', tableSource:'来源', tableGames:'局数', tableWins:'胜', tableDraws:'和', tableRuleDraws:'规则和棋', tableMaxPly:'max-ply', tableLosses:'负', tableScoreRate:'得分率', tableLevel:'级别', tableNodes:'节点上限', tableSearchTime:'参考毫秒（中位 / P90）', referenceSearch:'参考搜索', none:'无',
    footer:'只读监控面板 · 每 5 秒刷新 · 仅显示观测与冻结计划，不绘制预测', observedEvidenceKey:'曲线和条形：仅已观测日志/遥测', planBoundaryKey:'虚线与上限：冻结计划边界，不是预测', helpPurpose:'作用', helpRead:'怎么看',
    helpExpected:'常见情况（非预测）', helpWatch:'需要注意', helpButton:'查看说明', closeHelp:'关闭说明',
    noData:'暂无数据', controllerConfirmed:'控制器确认', loggedRows:'日志点', limit:'上限',
    temperature:'温度', best:'最佳', game:'局', gamesUnit:'局', level:'级别', nodes:'节点', scoreRate:'得分率', winRate:'胜率', sourceSample:'来源样本', trainingDiagnosticOnly:'混合来源训练诊断，不是棋力 KPI', frozenShort:'冻结臂', sanmillShort:'Sanmill', fullWindow:'完整 200 局窗口', updatedAt:'更新于', malformedTail:'忽略损坏尾行',
    loadFailed:'读取失败', yes:'是', online:'控制器在线', offline:'控制器离线', controllerExited:'控制器已正常退出', noInfrastructureStop:'无基础设施停止信号', healthIssueCount:'项需检查', learnerWhite:'执白', learnerBlack:'执黑', allSources:'全部来源', noTrainingGpuTelemetry:'无训练期遥测', postTrainingGpuSamplesIgnored:'已忽略训练窗口外样本', sampleCount:'样本数',
    healthStates:{healthy:'正常',complete:'完整完成',warning:'需注意',stop:'停止信号'},
    states:{running:'运行中',between_segments:'分段交接中',completed:'已完成',complete:'已完成',stopped:'已停止',failed:'失败',unknown:'未知'},
    chart:{win200:'200 局胜率',draw200:'200 局和棋率',best:'最佳',frozenScore200:'冻结模型得分率',sanmillScore200:'Sanmill 得分率',policy:'策略',heuristic:'启发式',malom:'Malom',
      temperature:'温度',chosenProbability:'选中概率',entropy:'Entropy',policyLoss:'Policy loss',valueLoss:'Value loss',totalReward:'总奖励',heuristicReward:'启发式奖励',retroReward:'Retro 奖励',learningRate:'实际 LR × 10⁴',difficulty:'难度',ply:'50 局平均 ply',gpuUtil:'GPU 利用率',vramUtil:'显存占用率',malomKnown:'Malom 标签覆盖率'},
    values:{win:'胜',draw:'和',loss:'负','fewer-than-three':'少于三子','no-legal-move':'无合法着',
      repetition:'重复局面','threefold-repetition':'三次重复','draw_repetition':'重复和棋','max-ply':'达到 ply 上限','max-ply-truncation':'ply 上限截断',
      max_ply:'达到 ply 上限',DRAW_LONG:'长局截断','no-progress':'无进展和棋','vs_heuristic':'启发式对手',
      'vs_sanmill':'Sanmill 搜索对手','vs_frozen':'冻结模型对手',heur:'启发式对手',frozen:'冻结模型对手',
      win_fewer_than_three:'胜：对方少于三子',win_no_legal_moves:'胜：对方无合法着',draw_repetition:'重复和棋',draw_threefold_repetition:'三次重复和棋',draw_fifty_move:'五十步和棋',draw_no_progress:'无进展和棋',max_ply_truncation:'达到 ply 上限','max-ply-truncation':'达到 ply 上限',lose_no_legal_moves:'负：无合法着',lose_fewer_than_three:'负：少于三子',other:'其他'}
  },
  en: {
    documentTitle:'NMM_LLM Training Monitor', pageTitle:'NMM_LLM · Managed Generalist Training',
    identityLoading:'Reading plan identity…', connecting:'Connecting', languageLabel:'Interface language', exportPng:'Export PNG', exportFailed:'PNG export failed',
    cardHealth:'Training health', cardGames:'Observed games', cardSegments:'Segments', cardActiveTime:'Active time', cardDifficulty:'Node level',
    cardFrozenRecent:'Frozen arm · latest 200 source games', cardSanmillRecent:'Sanmill · latest 200 source games', cardTrainingWindow:'Mixed training window', cardRuleDraws:'Rules draws', cardMaxPly:'max-ply truncations', cardLatestUpdate:'Latest update', cardGpu:'Training-window GPU telemetry', cardComponents:'Experiment switches', panelWinTrend:'200-game score-rate trend by opponent source',
    panelTop1:'Policy, heuristic, and Malom Top-1 (50-game smoothed)', panelExploration:'Temperature and chosen probability',
    panelLosses:'Policy / Value loss', panelEntropy:'Entropy (50-game smoothed)', panelRewards:'Reward signals (50-game smoothed)', panelLr:'Learning rate × 10⁴ (actual steps)', lrChartNote:'Actual executed values; step line with no smoothing or linear interpolation.', panelDifficultyPly:'Game length (50-game mean)', panelGpu:'Training-window GPU and VRAM telemetry (whole device)', gpuTrainingNote:'Only whole-device samples inside managed segment activity windows are accepted; this is not process-exclusive usage.', panelTerminationTrend:'Termination mix (rolling 50 games)', panelTerminations:'Termination reasons',
    panelOpponentOutcomes:'Wins / draws / losses by opponent source', opponentOutcomeNote:'Rules draws and max-ply truncations are counted separately; score rate is a training diagnostic only.', sanmillByLevel:'Sanmill by node level', nodeTimingNote:'Reference search time is this host’s warm-process calibration median / P90 and includes Sanmill search only.',
    panelOutcomes:'All wins / draws / losses', panelOpponents:'Opponent mix', panelSegmentEvidence:'Segment evidence',
    panelWarnings:'Warnings (read only)', tableSegment:'Segment', tableFirstGame:'First game', tableLastGame:'Last game',
    tableGameRows:'Game rows', tableUpdates:'Updates', tableCheckpoint:'Checkpoint', tableSource:'Source', tableGames:'Games', tableWins:'Wins', tableDraws:'Draws', tableRuleDraws:'Rules draws', tableMaxPly:'max-ply', tableLosses:'Losses', tableScoreRate:'Score rate', tableLevel:'Level', tableNodes:'Node ceiling', tableSearchTime:'Reference ms (median / P90)', referenceSearch:'reference search', none:'None',
    footer:'Read-only monitor · refreshes every 5 seconds · observed data and frozen plan only; no forecast', observedEvidenceKey:'Lines and bars: observed logs/telemetry only', planBoundaryKey:'Dashed markers and limits: frozen plan boundaries, not forecasts', helpPurpose:'Purpose', helpRead:'How to read it',
    helpExpected:'Typical pattern (not a forecast)', helpWatch:'Watch for', helpButton:'Show explanation', closeHelp:'Close explanation',
    noData:'No data', controllerConfirmed:'Controller confirmed', loggedRows:'logged points', limit:'limit',
    temperature:'temperature', best:'best', game:'game', gamesUnit:'games', level:'level', nodes:'nodes', scoreRate:'score rate', winRate:'win rate', sourceSample:'source sample', trainingDiagnosticOnly:'mixed-source training diagnostic, not a strength KPI', frozenShort:'frozen', sanmillShort:'Sanmill', fullWindow:'full 200-game window', updatedAt:'Updated', malformedTail:'malformed tail lines ignored',
    loadFailed:'Read failed', yes:'yes', online:'controller online', offline:'controller offline', controllerExited:'controller exited normally', noInfrastructureStop:'no infrastructure stop signal', healthIssueCount:'items need review', learnerWhite:'learner White', learnerBlack:'learner Black', allSources:'all sources', noTrainingGpuTelemetry:'No training-window telemetry', postTrainingGpuSamplesIgnored:'samples outside training windows ignored', sampleCount:'samples',
    healthStates:{healthy:'healthy',complete:'complete',warning:'warning',stop:'stop signal'},
    states:{running:'running',between_segments:'between segments',completed:'completed',complete:'completed',stopped:'stopped',failed:'failed',unknown:'unknown'},
    chart:{win200:'win 200',draw200:'draw 200',best:'best',frozenScore200:'frozen-model score',sanmillScore200:'Sanmill score',policy:'policy',heuristic:'heuristic',malom:'Malom',temperature:'temperature',
      chosenProbability:'chosen probability',entropy:'entropy',policyLoss:'policy loss',valueLoss:'value loss',totalReward:'total reward',heuristicReward:'heuristic reward',retroReward:'retro reward',learningRate:'actual LR × 10⁴',difficulty:'difficulty',ply:'50-game mean ply',gpuUtil:'GPU utilization',vramUtil:'VRAM utilization',malomKnown:'Malom label coverage'},
    values:{win:'win',draw:'draw',loss:'loss','fewer-than-three':'fewer than three','no-legal-move':'no legal move',
      repetition:'repetition','threefold-repetition':'threefold repetition','draw_repetition':'repetition draw','max-ply':'ply limit','max-ply-truncation':'ply-limit truncation',
      max_ply:'ply limit',DRAW_LONG:'long-game truncation','no-progress':'no-progress draw','vs_heuristic':'heuristic opponent',
      'vs_sanmill':'Sanmill search opponent','vs_frozen':'frozen-model opponent',heur:'heuristic opponent',frozen:'frozen-model opponent',
      win_fewer_than_three:'win: opponent <3',win_no_legal_moves:'win: opponent blocked',draw_repetition:'repetition draw',draw_threefold_repetition:'threefold repetition draw',draw_fifty_move:'fifty-move draw',draw_no_progress:'no-progress draw',max_ply_truncation:'ply-limit truncation','max-ply-truncation':'ply-limit truncation',lose_no_legal_moves:'loss: blocked',lose_fewer_than_three:'loss: <3',other:'other'}
  }
};

const HELP = {
  health: {
    zh:{title:'训练健康度',purpose:'用透明、可复核的基础设施门槛汇总运行状态；它不预测晋级概率，也不评价最终棋力。',read:'完整完成/正常表示计划身份、控制器状态、分段连续性、checkpoint/manifest、日志完整性和数值有限性未发现停止信号。需注意会列出非致命证据问题；停止信号表示应隔离运行。',expected:'运行中保持正常，全部计划局数和分段闭合后变为完整完成。当前冻结资源课程不按胜率晋级，所以没有统计上成立的晋级概率。',watch:'身份不一致、控制器异常离线、非有限指标、分段缺口、已完成分段缺证据、完成计数不一致或未知 stderr 警告。健康不等于模型变强；棋力信号请看按对手来源 W/D/L。'},
    en:{title:'Training health',purpose:'Summarizes transparent, auditable infrastructure gates. It neither predicts promotion probability nor judges final playing strength.',read:'Complete/healthy means no stop signal was found in plan identity, controller state, segment continuity, checkpoint/manifest presence, log integrity, or metric finiteness. Warning lists non-fatal evidence concerns; stop means quarantine the run.',expected:'It stays healthy while running and becomes complete after all planned games and segments close. This fixed-resource curriculum does not advance by win rate, so no statistically defined promotion probability exists.',watch:'Identity drift, an unexpectedly dead controller, non-finite metrics, segment gaps, missing completed evidence, completion-count mismatch, or unknown stderr warnings. Health is not strength; use source-split W/D/L for playing signals.'}
  },
  games: {
    zh:{title:'观测局数',purpose:'显示监控器从所有分段观测到的局数、计划总局数、控制器已确认的完成局数和可用日志点数。',read:'分子是目前观测到的最高局号，分母是冻结计划中的总局数。“控制器确认”以分段完成证据为准；日志点可能是抽样，不等于每局一行。',expected:'运行时应单调增长并最终到达计划上限。活跃分段内，观测值短暂领先控制器确认值属正常。',watch:'控制器离线、长时间不增长、计数倒退，或观测值与分段证据长期不一致。'},
    en:{title:'Observed games',purpose:'Shows the games seen across all segments, the planned total, controller-confirmed completion, and the number of usable log points.',read:'The numerator is the highest observed game; the denominator is the frozen plan total. Controller-confirmed games come from completed-segment evidence. Logged points may be sampled and need not equal one row per game.',expected:'It should increase monotonically while training runs and eventually reach the plan total. The observed count may briefly lead the confirmed count inside the active segment.',watch:'A dead controller, a prolonged stall, a decreasing count, or a lasting mismatch between observed progress and segment evidence.'}
  },
  segments: {
    zh:{title:'分段',purpose:'跟踪每 250 局一个进程段的完成情况，用于精确续训、资源审计和故障隔离。',read:'主数字是已完成分段数/计划分段总数；下方是目前活跃的 segment 目录。',expected:'每完成 250 局增加 1，监控页会自动跟随新分段，无需重启。',watch:'分段序号跳号、重叠，或已完成分段缺少 manifest/checkpoint。活跃分段在结束前没有最终 checkpoint 是正常的。'},
    en:{title:'Segments',purpose:'Tracks the 250-game process segments used for exact resume, resource audits, and failure isolation.',read:'The main value is completed segments over planned segments; the subtitle names the active segment directory.',expected:'It advances by one every 250 games. The dashboard should follow each new segment automatically without a restart.',watch:'Skipped or overlapping segment numbers, or a completed segment without its manifest/checkpoint. The active segment normally lacks its final checkpoint until it closes.'}
  },
  activeTime: {
    zh:{title:'活跃运行时间',purpose:'估算已完成分段加当前分段的实际活跃训练时间，并对照 12 小时资源上限。',read:'它是训练活跃时间，不是包含关机、暂停或人工等待的日历时间。',expected:'训练正常运行时稳定增长；分段交接时可有短暂平台。',watch:'局数增长但活跃时间不动、时间突然倒退，或在未完成计划时超出冻结资源上限。'},
    en:{title:'Active time',purpose:'Estimates active training time from completed segments plus the current segment and compares it with the 12-hour resource ceiling.',read:'This is active training time, not calendar time including shutdowns, pauses, or waiting.',expected:'It should rise steadily while training is active, with brief plateaus during segment hand-off.',watch:'Games advancing while active time stays frozen, time moving backwards, or the run exceeding its frozen resource limit before completion.'}
  },
  difficulty: {
    zh:{title:'节点级别',purpose:'显示当前 Sanmill 固定节点课程级别，并给出本机校准的参考搜索耗时。',read:'级别 1–5 分别对应每次搜索 1千、5千、2.5万、10万、50万节点上限。参考值依次为 0.21/0.30、0.63/0.80、2.44/2.85、9.91/11.34、52.85/60.77 毫秒（中位数/P90）。它们来自持久 Sanmill 进程的八局面引擎校准，只包含搜索，不是整步或整局耗时，也不是固定时限。',expected:'只应在全局第 500、1000、1500、2500 局后按计划升档，不依赖胜率。同一节点上限的实际耗时仍会随局面、提前完成深度和开局深度限制变化。',watch:'不要把参考毫秒当作强制思考时限或棋力等级。注意提前或延迟升档、节点记录不符、搜索失败，或耗时在同一硬件上出现数量级异常。'},
    en:{title:'Node level',purpose:'Shows the current fixed-node Sanmill curriculum level with this host’s calibrated reference search time.',read:'Levels 1–5 cap search at 1k, 5k, 25k, 100k, and 500k nodes. Warm-process median/P90 values are 0.21/0.30, 0.63/0.80, 2.44/2.85, 9.91/11.34, and 52.85/60.77 ms. They come from an eight-root engine calibration and include search only—not the whole logical turn or game, and not a fixed time limit.',expected:'Levels change only after global games 500, 1000, 1500, and 2500, independent of win rate. Actual time at one ceiling still varies with position, completed depth, and the opening-depth policy.',watch:'Do not treat the reference milliseconds as a forced think time or strength rating. Watch for early/late level changes, mismatched node records, search failures, or order-of-magnitude timing drift on the same host.'}
  },
  sourceRecent: {
    zh:{title:'按对手来源的最近窗口',purpose:'分别显示冻结臂与 Sanmill 臂最近 200 个同来源训练局的胜率、得分率和 W/D/L，避免混合对手分布。',read:'主数字是胜率；副标题同时给出得分率、W/D/L 和实际样本数。得分率把和棋计半分。窗口按来源独立推进，所以两臂对应的全局局号区间不完全相同。',expected:'冻结臂与 Sanmill 臂可以有完全不同的量级。只能在同一来源、节点、颜色和温度背景下解释变化。',watch:'冻结臂结果不是绝对棋力，Sanmill 训练结果也不是 held-out。不要把高冻结臂胜率与 Sanmill 胜率相加或平均后作模型选择。'},
    en:{title:'Recent window by opponent source',purpose:'Separately shows win rate, score rate, and W/D/L over the latest 200 same-source training games for the frozen and Sanmill arms.',read:'The headline is win rate; the subtitle also reports score rate, W/D/L, and actual sample size. Draws earn half a point in score rate. Each source window advances independently, so their global-game spans differ.',expected:'The two arms may live at entirely different scales. Interpret changes only within the same source, node level, learner color, and temperature context.',watch:'Frozen-arm results are not absolute strength, and Sanmill training games are not held out. Never add or average the two headline rates for model selection.'}
  },
  trainingWindow: {
    zh:{title:'混合训练窗口',purpose:'显示 manifest 的 resolved_config.rolling_win 所定义的真实混合来源训练胜率窗口；页面标签动态显示局数，不相信旧字段名暗示的 200 局。',read:'窗口长度从每个分段 manifest 读取并要求完全一致。主数字只计胜局，不给和棋半分；它混合冻结臂和 Sanmill 臂。',expected:'可作为训练调度和日志连续性诊断，但会随近期对手抽样比例剧烈波动。',watch:'这不是棋力 KPI、held-out 结果或晋级概率。若分段 manifest 的窗口不一致，监控器会停止生成状态而不是猜测。'},
    en:{title:'Mixed training window',purpose:'Shows the real mixed-source training win-rate window from manifest resolved_config.rolling_win. The label displays the dynamic game count instead of trusting the legacy field name.',read:'The window is read from every segment manifest and must agree. The headline counts wins only, gives no half point for a draw, and mixes frozen and Sanmill arms.',expected:'It is useful for training-schedule and log-continuity diagnostics but can move sharply with the recent opponent sample mix.',watch:'This is not a strength KPI, held-out result, or promotion probability. If segment manifests disagree, status generation fails instead of guessing.'}
  },
  terminationSplit: {
    zh:{title:'规则和棋与 max-ply 截断',purpose:'把棋规实际裁定的和棋与监控上限截断分开，避免把未完成对局写成规则和棋。',read:'规则和棋包括三次重复和五十步/无进展裁定；max-ply 是达到实验安全上限后停止，结果不完整。主数字给出全量计数和占比，副标题按冻结臂与 Sanmill 拆分。',expected:'两类都可能随节点档位和策略变化。规则和棋可作为结果；截断只能作为不完整性和被动性诊断。',watch:'不要用 Malom 单局面 W/D/L 覆盖严格裁判结果，也不要把截断计半分后宣称真实得分率。截断占比高时，强度结论需要更高安全上限的独立评测。'},
    en:{title:'Rules draws versus max-ply truncations',purpose:'Separates draws actually awarded by the rules from games stopped at the monitoring cap, so incomplete games are not relabelled as rules draws.',read:'Rules draws include threefold repetition and fifty-move/no-progress decisions. max-ply means the experiment safety limit stopped an incomplete game. The headline gives total count and share; the subtitle splits frozen and Sanmill arms.',expected:'Both can change with node level and policy. Rules draws are outcomes; truncations are incompleteness and passivity diagnostics only.',watch:'Do not overwrite strict-referee outcomes with a single-position Malom W/D/L, and do not award half points to truncations as true score. A high truncation share requires an independent evaluation with a larger safety cap for strength claims.'}
  },
  latestUpdate: {
    zh:{title:'最近更新',purpose:'显示最近一次优化器更新对应的局号，以及 Policy/Value loss。',read:'Policy loss 受目标和优势符号影响，可以为负；不要只用“越小越好”解读单点。应结合更新频率、非有限值和长期尺度。',expected:'局号应持续跟进；loss 可噪声很大，但应保持有限，且不应持续爆炸。',watch:'NaN/Inf、长期数量级上升、数值完全冻结，或对局在增长但更新局号长时间不动。'},
    en:{title:'Latest update',purpose:'Shows the game index of the latest optimiser update plus policy and value losses.',read:'Policy loss can be negative because of the objective and advantage signs. Do not interpret one point as simply “lower is better”; inspect update cadence, finiteness, and longer-term scale.',expected:'The update game should keep advancing. Losses may be noisy but should remain finite and should not explode persistently.',watch:'NaN/Inf, sustained orders-of-magnitude growth, perfectly frozen values, or games advancing while the update index remains stale.'}
  },
  components: {
    zh:{title:'实验开关',purpose:'明确显示本次冻结基线没有启用哨兵网络（Sentinel）与恢复机制，避免把默认占位值或缺少事件误读为有效信号。',read:'“关”是计划要求，不是组件加载失败。课程为固定资源模式，只按全局局号切换 Sanmill 节点档位。',expected:'整个训练谱系中，哨兵网络、恢复（restore）、复活（resurrect）和热探索（hot-explore）都应保持关闭；资源档位按计划变化。',watch:'日志中出现恢复、复活或热探索事件、哨兵网络的非占位奖励，或开关在续训后改变。'},
    en:{title:'Experiment switches',purpose:'Makes explicit that Sentinel and recovery are disabled, so placeholder values or absent events are not mistaken for real signals.',read:'OFF is the frozen plan, not a load failure. The fixed-resource curriculum changes Sanmill node levels only by global game index.',expected:'Sentinel, restore, resurrect, and hot-explore remain off for the lineage while resource levels follow the plan.',watch:'Any restore/resurrect/hot-explore event, non-placeholder Sentinel reward, or a switch changing after resume.'}
  },
  winTrend: {
    zh:{title:'按对手来源的 200 局得分率趋势',purpose:'分别显示冻结模型和 Sanmill 搜索对手的最近 200 个同来源训练局得分率，避免混合来源。',read:'得分率=(胜+0.5×和)/局数。每条线在该来源积满 200 局后才出现。横轴是全局局号；灰色短虚线是分段边界，绿色长虚线是节点升档。',expected:'同一来源、节点档位和颜色下的中期变化更可解释。冻结目标遵循当前运行 manifest 绑定的刷新计划，Sanmill 节点档位也会变，因此曲线都不是固定对手评测。',watch:'不要跨节点档位或刷新计划直接归因，也不要将任一线当晋升概率。关注无解释断层、来源消失、长期贴近零，或策略更果断但得分率不改善。'},
    en:{title:'200-game score rate by opponent source',purpose:'Separately plots the latest 200 same-source training-game score rates for frozen-model and Sanmill-search opponents.',read:'Score rate=(wins+0.5×draws)/games. Each line begins only after that source has a full 200-game window. The x-axis is global game; short grey dashes are segment boundaries and long green dashes are node transitions.',expected:'Medium-term changes are most interpretable within one source, node level, and learner color. The frozen target follows the refresh schedule bound by this run’s manifest, while Sanmill node work changes by stage, so neither line is a fixed-opponent evaluation.',watch:'Do not compare across node levels or refresh schedules without qualification or treat either line as promotion probability. Watch for unexplained discontinuities, a missing source, sustained near-zero scores, or decisiveness without score improvement.'}
  },
  top1: {
    zh:{title:'策略、启发式与 Malom Top-1',purpose:'对比学习策略、启发式首选着和可用 Malom 标签之间的 Top-1 匹配率，并同时显示 Malom 标签覆盖率。',read:'均为完整 50 局等权平均，第 50 局前不画。Policy 与 heuristic 接近表示模型更像启发式；Malom 命中率必须和标签覆盖率一起看。一致率不等于棋力。',expected:'策略可与启发式分离，同时维持或提高可信 Malom 好着命中；具体方向不必单调。',watch:'续训后突然全为 0/1、覆盖率很低却解读 Malom 命中率、或策略更像启发式但来源分组结果不改善。'},
    en:{title:'Policy, heuristic, and Malom Top-1',purpose:'Compares learned-policy, heuristic-first-choice, and trusted-Malom Top-1 rates while also plotting Malom-label coverage.',read:'All are equal-weight full 50-game means and are hidden before game 50. Policy/heuristic convergence means more copying. Read Malom hit rate together with coverage. Agreement is not strength by itself.',expected:'The policy may depart from the heuristic while preserving or improving trusted Malom-good-move agreement; no line must be monotonic.',watch:'Abrupt all-zero/all-one values after resume, interpreting Malom hit rate under low coverage, or increasing heuristic imitation without source-split result improvement.'}
  },
  exploration: {
    zh:{title:'温度与选择概率',purpose:'同时展示控制探索的 temperature 与模型对实际选中着的平均概率。',read:'温度是逐局记录的已观测调度值；chosen probability 使用完整 50 局等权平均，第 50 局前不绘制。温度越高，抽样越分散；chosen probability 越高，模型对自己选择越果断。两条线不是简单的镜像。',expected:'本基线温度应按冻结调度总体下降；学到稳定偏好时，chosen probability 可逐步上升，但不要求单调。',watch:'分段边界温度重置、概率迅速接近 1 但胜率不升（自信地选错），或长期极低概率表明策略仍无区分度。'},
    en:{title:'Temperature and chosen probability',purpose:'Shows exploration temperature together with the mean probability assigned to the action actually chosen.',read:'Temperature is the observed per-game schedule value. Chosen probability uses a full equal-weight 50-game mean and is hidden before game 50. Higher temperature spreads sampling; higher chosen probability means the policy is more decisive. The two lines are not simple mirror images.',expected:'This baseline temperature should decline according to its frozen schedule. Chosen probability may rise as preferences become clearer, but it need not be monotonic.',watch:'Temperature resets at segment boundaries, chosen probability racing toward one while win rate stays flat (confidently wrong), or persistently tiny probabilities indicating little policy separation.'}
  },
  entropyTrend: {
    zh:{title:'Entropy',purpose:'用完整 50 局等权平均观察策略分布的不确定性；第 50 局前不绘制。',read:'Entropy 越低通常表示策略越集中，但不说明集中在正确着法。选中概率已单独放在温度图中，避免不同量纲共轴。',expected:'随学习和降温可总体下降，也会随节点档位和局面波动；这只是常见现象，不是预测目标。',watch:'突然接近零但按来源得分不改善、分段边界跳变、非有限值，或长期完全冻结。'},
    en:{title:'Entropy',purpose:'Uses a full equal-weight 50-game mean to show policy uncertainty; it is hidden before game 50.',read:'Lower entropy usually means a more concentrated policy, not that it concentrates on correct moves. Chosen probability remains in the temperature chart so unlike units do not share this axis.',expected:'It may decline with learning and cooling and fluctuate with node levels and positions; that is a common pattern, not a forecast target.',watch:'Collapse toward zero without source-split score improvement, resume jumps, non-finite values, or a completely frozen line.'}
  },
  losses: {
    zh:{title:'Policy / Value loss',purpose:'跟踪每次优化更新的策略目标和价值预测误差，主要用于发现数值不稳定或更新停滞。',read:'看长期范围、噪声带和是否有非有限值。Policy loss 可为负；不同阶段的绝对值不一定可直接比。',expected:'小批量 RL loss 通常很噪，可上下振荡。健康信号是数值有限、尺度可控、更新持续，而不是每步下降。',watch:'NaN/Inf、持续爆炸、周期性与分段续训完全对齐的异常跳变，或很长时间没有新更新点。'},
    en:{title:'Policy / Value loss',purpose:'Tracks policy-objective and value-prediction losses at optimiser updates, mainly to detect numerical instability or stalled learning.',read:'Inspect long-term scale, noise, and finiteness. Policy loss may be negative, and absolute values across different phases are not always directly comparable.',expected:'Small-batch RL losses are usually noisy. Healthy means finite, bounded in scale, and continually updated—not decreasing at every step.',watch:'NaN/Inf, sustained explosion, discontinuities exactly aligned with resumes, or a long period with no new update points.'}
  },
  rewards: {
    zh:{title:'奖励信号',purpose:'用 50 局滑动平均显示总奖励、启发式奖励与胜负回传奖励，帮助判断最终胜负回传是否主导学习，以及形状奖励是否异常。',read:'三条线均使用完整 50 局等权平均，第 50 局前不绘制。总奖励是实际训练目标中的合成结果；启发式奖励是很小的辅助形状项；胜负回传奖励反映最终对局结果。本实验哨兵网络（Sentinel）关闭，因此不绘制哨兵网络奖励。',expected:'曲线会很噪。胜局增加时，胜负回传奖励和总奖励可能上升；启发式奖励应保持较小且有限。',watch:'奖励出现非有限值、启发式奖励突然主导总奖励、胜负回传奖励与结果统计方向矛盾，或续训边界产生无解释偏移。'},
    en:{title:'Reward signals',purpose:'Shows 50-game moving averages for total, heuristic, and retro rewards to reveal whether outcome propagation and shaping behave coherently.',read:'All three lines use full equal-weight 50-game means and are hidden before game 50. Total is the combined training reward, heuristic is a small shaping term, and retro carries outcome feedback. Sentinel reward is omitted because Sentinel is disabled.',expected:'Noise is normal. Retro/total may improve with more wins, while heuristic reward should remain small and finite.',watch:'Non-finite reward, heuristic dominating total, retro contradicting outcomes, or unexplained resume-boundary shifts.'}
  },
  learningRate: {
    zh:{title:'Learning rate',purpose:'显示训练器实际执行的学习率，纵轴为原值乘以 10⁴，便于观察很小的数。',read:'这是未经平滑的原始控制量。水平段表示学习率保持不变，垂直跳变表示从该局起启用新值；不会把离散降档画成连续衰减。例如图上的 0.5 代表实际 LR=0.00005。',expected:'可按冻结的适配规则离散变化，但 exact-resume 后应连续，不应在分段边界无原因重置。',watch:'变为零、非有限、超出预期范围，或恰好在每个 250 局边界反复重置。'},
    en:{title:'Learning rate',purpose:'Shows the optimizer learning rate multiplied by 10⁴ so small values remain readable.',read:'This is the unsmoothed executed control value. Horizontal runs mean the rate was held; a vertical step marks the game where the new value took effect. Discrete changes are never drawn as gradual decay. A plotted value of 0.5 means an actual LR of 0.00005.',expected:'Bounded discrete steps are possible, but exact resume should remain continuous instead of resetting at segment boundaries.',watch:'Zero, non-finite or out-of-range LR, or repeated resets exactly at every 250-game boundary.'}
  },
  difficultyPly: {
    zh:{title:'对局长度',purpose:'显示完整 50 局窗口内的平均 rollout 逻辑 ply；第 50 局前不绘制。',read:'纵轴从 0 开始并按 10 ply 整数刻度显示。蓝色竖虚线是冻结计划中的节点升档边界，不是难度曲线或未来预测。',expected:'对局长度会随局面大幅波动，节点档位变化后可能出现分布改变。',watch:'大量对局粘在 120-ply 截断上限、分段边界无解释跳变，或所有长度突然相同。'},
    en:{title:'Game length',purpose:'Shows mean rollout logical plies over a full 50-game window; it is hidden before game 50.',read:'The y-axis starts at zero with integer 10-ply ticks. Blue vertical dashes are frozen-plan node-transition boundaries, not a difficulty curve or future forecast.',expected:'Game length varies with positions and its distribution may change after a node-level transition.',watch:'Many games pinned to the 120-ply truncation ceiling, unexplained resume-boundary jumps, or suddenly identical lengths.'}
  },
  gpuCard: {
    zh:{title:'训练期 GPU 遥测',purpose:'显示受管训练分段运行窗口内最近一条 NVIDIA 设备 0 整卡样本，而不是查看网页时的实时负载。',read:'主数字是最近训练期样本的 GPU 利用率；副标题显示整卡显存和训练期样本数。整卡读数可能包含其他程序，不等于训练进程独占值。本次运行若未在训练时采集，就明确显示“无训练期遥测”，不能事后重建。',expected:'神经网络更新会形成短峰；Sanmill CPU 搜索和环境推进期间 GPU 较低是正常的。显存通常较稳定。',watch:'缺失不等于 0%。训练仍在更新但 GPU 长期为 0、显存突然大幅增加或逼近 100%、训练期采样停止，或 NVIDIA 查询错误。'},
    en:{title:'Training-window GPU telemetry',purpose:'Shows the latest whole-device NVIDIA device 0 sample recorded inside a managed training segment, not the live load when the page is viewed.',read:'The headline is the latest training-window GPU utilization; the subtitle shows whole-device VRAM and training-window sample count. Other applications may contribute, so this is not process-exclusive. If the run was not sampled during training, it says no telemetry instead of reconstructing it afterward.',expected:'Neural-network updates produce short peaks, while CPU-side Sanmill search and environment work can leave GPU utilization low. VRAM should usually be stable.',watch:'Missing does not mean 0%. Watch for zero utilization while updates continue, sudden or near-full VRAM, stopped in-window sampling, or NVIDIA query errors.'}
  },
  gpuTrend: {
    zh:{title:'训练期间 GPU 与显存遥测（整卡）',purpose:'仅在控制器确认受管训练分段正在运行时每 5 秒采样整张 NVIDIA 设备，并按当时的全局局号绘图。',read:'纵轴均为 0–100%；青色实线是整卡 GPU 计算利用率，橙色虚线是整卡显存占用率。训练窗口外的实时样本会被忽略；若没有训练期样本，图保持无数据。该遥测不能归因到单个进程。',expected:'GPU 通常呈脉冲状，因为规则推进和 Sanmill 搜索主要在 CPU；显存线应较平稳，分段交接期间不采样。',watch:'不要把训练后实时负载解释成训练负载。关注训练期间持续 100% 且训练停滞、显存不断爬升、非有限值或遥测停止。'},
    en:{title:'Training-window GPU and VRAM telemetry (whole device)',purpose:'Samples the whole NVIDIA device every five seconds only while the controller confirms a managed training segment is running, and plots it against the then-observed global game.',read:'Both use a 0–100% axis. Cyan is whole-device compute utilization and orange dashed is whole-device VRAM utilization. Live samples outside training windows are ignored; the chart stays empty when no in-window evidence exists. The data is not attributable to one process.',expected:'GPU is usually bursty because rules and Sanmill search are CPU-side. VRAM should be comparatively stable; no samples are taken between segments.',watch:'Never interpret post-training live load as training load. Watch for sustained 100% with stalled training, steadily rising VRAM, non-finite values, or in-window telemetry stopping.'}
  },
  terminationTrend: {
    zh:{title:'终止原因构成',purpose:'以滚动 50 局百分比堆叠图显示胜负原因、规则和棋和 120-ply 截断如何随训练变化。',read:'仅在积满完整 50 局后绘制；每个时点各颜色合计为 100%。规则重复和棋与 max-ply 截断严格分开；截断不是棋规和棋。',expected:'构成可随模型和节点档位变化，但合法原因应可解释，截断不应长期占主导。',watch:'截断比例持续上升、未知原因增多、某合法类别无解释消失，或与累计终止条形图对不上。'},
    en:{title:'Termination mix',purpose:'Shows a rolling 50-game percentage stack for win/loss causes, rules draws, and 120-ply truncations.',read:'The stack is hidden until a full 50-game window exists; colors then sum to 100% at each game. Repetition draws remain separate from max-ply truncation; truncation is not a rules draw.',expected:'The mix may change with policy and node level, but reasons should remain explainable and truncation should not dominate persistently.',watch:'A rising truncation share, growing unknown reasons, unexplained disappearance of a legal category, or disagreement with aggregate termination counts.'}
  },
  terminations: {
    zh:{title:'终止原因',purpose:'统计可用日志行中对局如何结束，例如少于三子、无合法着、重复或达到 ply 上限。',read:'条形长度是相对计数。这是已记录样本的分布；如果训练器抽样记录，它不是全部对局的完整普查。',expected:'正常应有多种合法终止原因。重复和棋可出现；max-ply/长局截断不是规则和棋，应单独解读。',watch:'截断比例异常升高、某个合法终止类型突然消失，或出现无法识别的新原因。'},
    en:{title:'Termination reasons',purpose:'Counts how logged games ended, such as fewer than three pieces, no legal move, repetition, or the ply cap.',read:'Bar lengths are relative counts over available log rows. If the trainer samples game logs, this is not a complete census of every game.',expected:'Several legal terminal reasons should appear. Repetition draws may occur; max-ply/long-game truncation is not a rules-based draw and must remain distinguishable.',watch:'A rising truncation share, the sudden disappearance of a legal terminal type, or an unknown new reason.'}
  },
  opponentOutcomes: {
    zh:{title:'按对手来源的 W/D/L',purpose:'把冻结模型和 Sanmill 搜索对手的结果分开，避免 60/40 混合总数掩盖两类训练环境的差异。',read:'每个来源显示样本数、胜和负、得分率、规则和棋和 max-ply 截断，并按学习者执白/执黑细分；Sanmill 另按节点档位列出本机校准的搜索中位数/P90。参考毫秒只包含持久进程中的引擎搜索。得分率把日志中的全部和棋计半分，所以必须同时查看截断列。',expected:'各层样本数应与对手调度、颜色和固定节点阶段相符。来源内和同一节点档位内的变化比混合总数更可解释。',watch:'不要把 frozen 结果当绝对棋力：冻结目标遵循计划绑定的刷新周期。Sanmill 是较稳定的外部锚点，但节点档位随阶段变化。两者的训练结果都不是冻结正式评测或晋升证据；max-ply 也不是真实规则和棋。'},
    en:{title:'Source-split W/D/L',purpose:'Separates frozen-model and Sanmill-search outcomes so the 60/40 aggregate cannot hide differences between the two training environments.',read:'Each source shows sample size, wins, draws, losses, score rate, rules draws, max-ply truncations, and learner-White/Black splits. Sanmill levels also show this host’s calibrated search median/P90. Those milliseconds include engine search only. Logged score gives every recorded draw half a point, so read it with the truncation column.',expected:'Sample sizes should match the opponent schedule, colors, and fixed-node stages. Within-source and within-level changes are more interpretable than the mixed total.',watch:'Frozen results are not absolute strength: the target follows the plan-bound refresh cadence. Sanmill is a steadier external anchor, but its node ceiling changes by stage. Neither training result is a frozen formal evaluation or promotion result, and max-ply is not a true rules draw.'}
  },
  outcomes: {
    zh:{title:'胜 / 和 / 负',purpose:'展示训练器按其记录视角标注的胜、和、负样本数。',read:'与对手来源和难度一起看。条形是日志样本计数，不是冻结基线的正式对局结果。',expected:'构成会随对手、难度和探索变化，不应要求胜率单调上升。和棋必须结合终止原因区分规则和截断。',watch:'长期极端单一结果、结果计数与终止记录无法对齐，或将此图误当成候选模型的棋力晋升证据。'},
    en:{title:'Wins / draws / losses',purpose:'Shows win, draw, and loss samples from the viewpoint recorded by the trainer.',read:'Read it with opponent mix and difficulty. These are logged training samples, not formal candidate-versus-frozen-baseline results.',expected:'The mix may change with opponent, difficulty, and exploration and need not improve monotonically. Use termination reasons to separate rules draws from truncations.',watch:'A prolonged single-outcome collapse, counts that cannot be reconciled with termination logs, or treating this panel as promotion evidence.'}
  },
  opponents: {
    zh:{title:'对手来源',purpose:'检查训练局是否来自冻结模型或 Sanmill 搜索对手，以验证冻结的 60/40 调度。',read:'条形表示两类来源在已记录样本中的数量。短窗口受随机抽样影响，不会精确等于比例。',expected:'样本足够多时应接近 60% 冻结模型、40% Sanmill，两类对手都应持续出现。',watch:'一类对手长期消失、比例持续极端偏斜，或分段续训后调度状态重置。'},
    en:{title:'Opponent mix',purpose:'Checks frozen-model versus Sanmill-search games against the frozen 60/40 scheduler.',read:'Bars count each source among logged samples. Short windows are noisy and will not exactly match the target ratio.',expected:'Over enough samples the split should approach 60% frozen model and 40% Sanmill, with both sources continuing.',watch:'One source disappearing, persistent extreme skew, or scheduler state unexpectedly resetting after resume.'}
  },
  segmentEvidence: {
    zh:{title:'分段证据',purpose:'列出每个进程分段的局号边界、日志行、优化更新、manifest 和 checkpoint，是续训完整性的快速视图。',read:'表格按最新分段在上显示。首/末局用于查重叠或缺口；checkpoint 表示该分段是否已有可续训产物。',expected:'已完成的 250 局分段应边界连续、有 manifest 且 checkpoint=是。最新活跃分段在结束前可显示 checkpoint=—。',watch:'已完成分段缺 checkpoint/manifest、局号重叠或跳过、空日志，或更新数在相似工作量下突然异常。'},
    en:{title:'Segment evidence',purpose:'Lists game bounds, log rows, optimiser updates, manifests, and checkpoints for each process segment as a quick exact-resume integrity view.',read:'Newest segments appear first. First/last games reveal gaps or overlaps; checkpoint shows whether a resumable artefact exists.',expected:'Every completed 250-game segment should be contiguous and have a manifest plus checkpoint=yes. The active segment may show checkpoint=— until it finishes.',watch:'A completed segment missing checkpoint/manifest, overlapping or skipped games, empty logs, or an unexplained update-count discontinuity.'}
  },
  warnings: {
    zh:{title:'警告（只读）',purpose:'显示 supervisor/trainer 标准错误尾部中的警告，不会向训练进程发送任何命令。',read:'目前已知的 HumanDB Malom 提示会每个进程分段出现一次：它表示未版本化的历史 Malom 列被屏蔽，人类频率/结果仍可用。重复条目可来自多个分段。',expected:'除已审核的 HumanDB 标签屏蔽提示外，应尽量保持稳定，不出现新的基础设施错误。',watch:'非有限值、Malom/DB 身份变化、checkpoint 损坏、CUDA 错误、规则身份不一致，或 controller offline。这些属于停止并隔离候选信号。'},
    en:{title:'Warnings (read only)',purpose:'Shows warning lines from the supervisor/trainer stderr tail without sending commands to training.',read:'The known HumanDB Malom notice appears once per process segment: unversioned historical Malom columns are masked, while human frequencies/outcomes remain usable. Repeated entries can therefore come from different segments.',expected:'Apart from the reviewed HumanDB label-masking notice, the list should remain stable with no new infrastructure errors.',watch:'Non-finite values, Malom/DB identity changes, checkpoint corruption, CUDA errors, rules-identity mismatches, or controller offline. These are stop-and-quarantine signals.'}
  }
};

let currentLanguage = localStorage.getItem('nmm-monitor-language') || 'zh';
if (!Object.hasOwn(I18N,currentLanguage)) currentLanguage='zh';
let lastData=null, activeHelpKey=null, lastFocusedElement=null, activeHelpSource=null;

function t(key){let value=I18N[currentLanguage];for(const part of key.split('.'))value=value?.[part];return value??key;}
const finiteNumber = v => (v===null||v===undefined||v==='') ? null : (Number.isFinite(Number(v)) ? Number(v) : null);
const pct = v => finiteNumber(v)===null ? '—' : `${(finiteNumber(v)*100).toFixed(1)}%`;
const num = (v,d=2) => finiteNumber(v)===null ? '—' : finiteNumber(v).toFixed(d);
const integer = v => finiteNumber(v)===null ? '—' : Math.round(finiteNumber(v)).toLocaleString(currentLanguage==='zh'?'zh-CN':'en-US');
const valueLabel = key => I18N[currentLanguage].values[key] || String(key).replaceAll('_',' ');
function nodeTimingLabel(nodeBudget){const timing=NODE_TIMING_MS[Math.round(Number(nodeBudget))];return timing?`${timing[0].toFixed(2)} / ${timing[1].toFixed(2)} ms`:'—';}
function learningRateNote(rows){const points=rows||[],changes=points.slice(1).filter((point,index)=>finiteNumber(point.lr_x1e4)!==finiteNumber(points[index].lr_x1e4));if(!points.length)return t('noData');const start=points[0];if(!changes.length)return currentLanguage==='zh'?`原始执行值：第 ${integer(start.game)} 局起 ${num(start.lr_x1e4,2)}，此后未变化；无平滑。`:`Actual executed value: ${num(start.lr_x1e4,2)} from game ${integer(start.game)}, unchanged thereafter; unsmoothed.`;return currentLanguage==='zh'?`原始执行值：第 ${integer(start.game)} 局起 ${num(start.lr_x1e4,2)}；${changes.map(point=>`第 ${integer(point.game)} 局起 ${num(point.lr_x1e4,2)}`).join('；')}。阶梯线，无平滑。`:`Actual executed value: ${num(start.lr_x1e4,2)} from game ${integer(start.game)}; ${changes.map(point=>`${num(point.lr_x1e4,2)} from game ${integer(point.game)}`).join('; ')}. Step line; unsmoothed.`;}

function applyStaticTranslations(){
  document.documentElement.lang=currentLanguage==='zh'?'zh-CN':'en';document.title=t('documentTitle');
  document.querySelectorAll('[data-i18n]').forEach(node=>{node.textContent=t(node.dataset.i18n);});
  document.querySelectorAll('[data-i18n-aria]').forEach(node=>{node.setAttribute('aria-label',t(node.dataset.i18nAria));});
  document.getElementById('languageSelect').value=currentLanguage;
  document.querySelectorAll('.help-button').forEach(button=>{const title=HELP[button.dataset.helpKey]?.[currentLanguage]?.title||'';button.setAttribute('aria-label',`${t('helpButton')}: ${title}`);button.title=`${t('helpButton')}: ${title}`;});
  document.getElementById('helpClose').setAttribute('aria-label',t('closeHelp'));
  if(activeHelpKey)renderHelp(activeHelpKey);
}

function renderHelp(key){const content=HELP[key]?.[currentLanguage];if(!content)return;
  document.getElementById('helpTitle').textContent=content.title;
  document.getElementById('helpPurposeText').textContent=content.purpose;
  document.getElementById('helpReadText').textContent=content.read;
  document.getElementById('helpExpectedText').textContent=content.expected;
  document.getElementById('helpWatchText').textContent=content.watch;
}
function openHelp(key,source){if(activeHelpSource)activeHelpSource.classList.remove('help-active');activeHelpKey=key;lastFocusedElement=source;activeHelpSource=source.closest('.card,.panel');if(activeHelpSource)activeHelpSource.classList.add('help-active');renderHelp(key);const panel=document.getElementById('helpPanel');panel.hidden=false;document.body.classList.add('help-open');if(lastData)render(lastData);document.getElementById('helpClose').focus();}
function closeHelp(){const panel=document.getElementById('helpPanel');panel.hidden=true;document.body.classList.remove('help-open');if(activeHelpSource)activeHelpSource.classList.remove('help-active');activeHelpSource=null;activeHelpKey=null;if(lastData)render(lastData);if(lastFocusedElement)lastFocusedElement.focus();}
function setLanguage(language){if(!Object.hasOwn(I18N,language))return;currentLanguage=language;localStorage.setItem('nmm-monitor-language',language);applyStaticTranslations();if(lastData)render(lastData);}

function drawMarkers(c,markers,X,xmin,xmax,pad,h){for(const marker of markers||[]){const game=Number(marker.game);if(!Number.isFinite(game)||game<xmin||game>xmax)continue;c.save();c.strokeStyle=marker.color;c.lineWidth=1;c.setLineDash(marker.dash||[4,4]);c.beginPath();c.moveTo(X(game),pad.t);c.lineTo(X(game),h-pad.b);c.stroke();c.restore();}}
function niceStep(range,target=5){if(!Number.isFinite(range)||range<=0)return 1;const rough=range/target,power=10**Math.floor(Math.log10(rough)),fraction=rough/power;return (fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*power;}
function axisNumber(value,step){if(step>=1)return integer(value);const digits=Math.max(1,Math.min(4,-Math.floor(Math.log10(step))));return num(value,digits);}
function drawYAxisLabel(c,label,x,y){c.save();c.textAlign='right';c.fillText(label,x,y);c.restore();}

function lineChart(id,rows,specs,fixedDomain=null,markers=[],tickStep=null,xDomain=null){
  const canvas=document.getElementById(id),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  const w=Math.max(320,rect.width),h=230;canvas.width=w*dpr;canvas.height=h*dpr;
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);const pad={l:48,r:15,t:25,b:27};
  const all=[];for(const s of specs)for(const r of rows){const x=finiteNumber(r.game),y=finiteNumber(r[s.key]);if(x!==null&&y!==null)all.push([x,y]);}
  if(!all.length){c.fillStyle='#91a4bd';c.font='11px system-ui';c.fillText(t('noData'),pad.l,pad.t+20);return;}
  let xmin=xDomain?xDomain[0]:Math.min(...all.map(p=>p[0])),xmax=xDomain?xDomain[1]:Math.max(...all.map(p=>p[0]));if(xmax===xmin)xmax=xmin+1;
  let ymin=fixedDomain?fixedDomain[0]:Math.min(...all.map(p=>p[1])),ymax=fixedDomain?fixedDomain[1]:Math.max(...all.map(p=>p[1]));
  let autoStep=null;if(tickStep&&!fixedDomain){ymin=0;ymax=Math.max(tickStep,Math.ceil(ymax/tickStep)*tickStep);}else if(!fixedDomain){if(ymax===ymin){const d=Math.max(Math.abs(ymax)*.1,1e-6);ymax+=d;ymin-=d;}autoStep=niceStep(ymax-ymin);ymin=Math.floor(ymin/autoStep)*autoStep;ymax=Math.ceil(ymax/autoStep)*autoStep;}
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  c.font='11px system-ui';c.strokeStyle='#23344a';c.fillStyle='#91a4bd';c.lineWidth=1;
  const effectiveStep=tickStep||autoStep;const tickCount=effectiveStep?Math.max(1,Math.round((ymax-ymin)/effectiveStep)):4;for(let i=0;i<=tickCount;i++){const y=pad.t+i*(h-pad.t-pad.b)/tickCount;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();const val=ymax-i*(ymax-ymin)/tickCount;const label=fixedDomain&&ymax===1?`${Math.round(val*100)}%`:fixedDomain&&ymax===100?`${Math.round(val)}%`:axisNumber(val,effectiveStep||(ymax-ymin)/tickCount);drawYAxisLabel(c,label,pad.l-8,y+4);}
  drawMarkers(c,markers,X,xmin,xmax,pad,h);
  c.fillText(integer(xmin),pad.l,h-7);const xmaxText=integer(xmax);c.fillText(xmaxText,w-pad.r-c.measureText(xmaxText).width,h-7);
  let lx=pad.l;for(const s of specs){c.save();c.strokeStyle=s.color;c.lineWidth=s.width||2;c.setLineDash(s.dash||[]);c.beginPath();c.moveTo(lx,pad.t-15);c.lineTo(lx+11,pad.t-15);c.stroke();c.restore();c.fillStyle='#c8d5e6';c.fillText(s.label,lx+15,pad.t-12);lx+=c.measureText(s.label).width+37;
    c.save();c.strokeStyle=s.color;c.lineWidth=s.width||2;c.setLineDash(s.dash||[]);c.beginPath();let started=false,previousPy=null;for(const r of rows){const x=finiteNumber(r.game),y=finiteNumber(r[s.key]);if(x===null||y===null){started=false;previousPy=null;continue;}const px=X(x),py=Y(y);if(!started){c.moveTo(px,py);started=true;}else{if(s.stepped)c.lineTo(px,previousPy);c.lineTo(px,py);}previousPy=py;}c.stroke();c.restore();}
}

function stackedAreaChart(id,rows,specs,markers=[],xDomain=null){
  const canvas=document.getElementById(id),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  const w=Math.max(320,rect.width),h=230;canvas.width=w*dpr;canvas.height=h*dpr;
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);const pad={l:48,r:15,t:25,b:27};
  if(!rows.length){c.fillStyle='#91a4bd';c.font='11px system-ui';c.fillText(t('noData'),pad.l,pad.t+20);return;}
  let xmin=xDomain?xDomain[0]:Math.min(...rows.map(r=>Number(r.game))),xmax=xDomain?xDomain[1]:Math.max(...rows.map(r=>Number(r.game)));if(xmax===xmin)xmax=xmin+1;
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>h-pad.b-y*(h-pad.t-pad.b);
  c.font='11px system-ui';c.strokeStyle='#23344a';c.fillStyle='#91a4bd';c.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad.t+i*(h-pad.t-pad.b)/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();drawYAxisLabel(c,`${100-i*25}%`,pad.l-8,y+4);}
  const cumulative=new Array(rows.length).fill(0);for(const spec of specs){c.beginPath();for(let i=0;i<rows.length;i++){const top=cumulative[i]+(Number(rows[i][spec.key])||0);const x=X(Number(rows[i].game)),y=Y(top);if(i===0)c.moveTo(x,y);else c.lineTo(x,y);}for(let i=rows.length-1;i>=0;i--)c.lineTo(X(Number(rows[i].game)),Y(cumulative[i]));c.closePath();c.fillStyle=spec.color;c.globalAlpha=.72;c.fill();c.globalAlpha=1;for(let i=0;i<rows.length;i++)cumulative[i]+=Number(rows[i][spec.key])||0;}
  drawMarkers(c,markers,X,xmin,xmax,pad,h);c.fillStyle='#91a4bd';c.fillText(integer(xmin),pad.l,h-7);const xmaxText=integer(xmax);c.fillText(xmaxText,w-pad.r-c.measureText(xmaxText).width,h-7);
  let lx=pad.l;for(const spec of specs){c.fillStyle=spec.color;c.fillRect(lx,pad.t-17,10,6);c.fillStyle='#c8d5e6';const label=valueLabel(spec.key);c.fillText(label,lx+14,pad.t-12);lx+=c.measureText(label).width+34;if(lx>w-120){lx=pad.l;}}
}

function bars(id,values,colors={},preferredOrder=[]){const host=document.getElementById(id),source=values||{},preferred=new Set(preferredOrder),names=[...preferredOrder.filter(name=>Object.hasOwn(source,name)),...Object.keys(source).filter(name=>!preferred.has(name))],entries=names.map(name=>[name,source[name]]);host.replaceChildren();
  if(!entries.length){host.textContent=t('noData');host.className='muted';return;}host.className='bars';const max=Math.max(1,...entries.map(([,v])=>Number(v)||0));
  for(const [name,value] of entries){const row=document.createElement('div');row.className='bar-row';const label=document.createElement('span');label.textContent=valueLabel(name);const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${100*Number(value)/max}%`;fill.style.background=colors[name]||COLORS.blue;track.append(fill);const count=document.createElement('span');count.textContent=integer(value);count.style.textAlign='right';row.append(label,track,count);host.append(row);}}

function appendOutcomeRow(tbody,label,summary,sub=false){const tr=document.createElement('tr');if(sub)tr.className='sub-row';const values=[label,integer(summary?.total),integer(summary?.win),integer(summary?.draw),integer(summary?.ruleDraw),integer(summary?.maxPly),integer(summary?.loss),pct(summary?.scoreRate)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);}
function renderOutcomeTables(data){const split=data.opponentOutcomes||{};const sourceBody=document.getElementById('sourceOutcomeRows');sourceBody.replaceChildren();appendOutcomeRow(sourceBody,t('allSources'),split.overall||{});for(const source of ['vs_frozen','vs_sanmill']){const group=split.bySource?.[source];if(!group)continue;appendOutcomeRow(sourceBody,valueLabel(source),group.overall);for(const color of ['W','B']){const summary=group.byLearnerColor?.[color];if(summary)appendOutcomeRow(sourceBody,color==='W'?t('learnerWhite'):t('learnerBlack'),summary,true);}}
  const levelBody=document.getElementById('sanmillLevelRows');levelBody.replaceChildren();for(const row of split.sanmillByLevel||[]){const tr=document.createElement('tr');const values=[`${t('level')} ${integer(row.level)}`,integer(row.nodeBudget),nodeTimingLabel(row.nodeBudget),integer(row.total),integer(row.win),integer(row.draw),integer(row.ruleDraw),integer(row.maxPly),integer(row.loss),pct(row.scoreRate)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}levelBody.append(tr);}}

function renderRecentSourceCard(data,source,valueId,subId){const recent=data.opponentOutcomes?.bySource?.[source]?.recentSourceGames||{};document.getElementById(valueId).textContent=pct(recent.winRate);document.getElementById(subId).textContent=`${t('scoreRate')} ${pct(recent.scoreRate)} · W/D/L ${integer(recent.win)}/${integer(recent.draw)}/${integer(recent.loss)} · n=${integer(recent.window)}`;}

function translateWarning(warning){if(currentLanguage==='zh'&&warning.startsWith('HumanDB Malom labels are disabled:'))return '已禁用 HumanDB 中未版本化的 Malom 标签；人类频率和对局结果仍可用（本机路径已隐藏）。';return warning;}
function render(data){lastData=data;const s=data.state,l=data.latest,g=data.gpu||{},gl=g.latest||{},h=data.health||{};const markers=[...(data.markers?.segments||[]).map(game=>({game,color:'#63758b',dash:[3,5]})),...(data.markers?.resources||[]).map(game=>({game,color:COLORS.blue,dash:[7,4]}))];const sharedGameDomain=[1,Math.max(2,Number(s.observedGame)||2)];
  document.getElementById('identity').textContent=`${data.identity.experimentId} · ${String(data.identity.planSha256).slice(0,16)}… · ${String(data.identity.gitCommit).slice(0,10)}`;
  const stateName=I18N[currentLanguage].states[s.name]||s.name||t('states.unknown');const badge=document.getElementById('stateBadge');badge.className=`badge ${s.name}`;const processText=s.name==='completed'&&!s.controllerAlive?t('controllerExited'):(s.controllerAlive?t('online'):t('offline'));document.getElementById('stateText').textContent=`${stateName} · PID ${s.controllerPid??'—'} · ${processText}`;
  const healthCard=document.getElementById('healthCard');healthCard.className=`card health-${h.name||'warning'}`;document.getElementById('health').textContent=t(`healthStates.${h.name||'warning'}`);const healthIssues=(h.stopReasons?.length||0)+(h.cautionReasons?.length||0);document.getElementById('healthSub').textContent=healthIssues?`${integer(healthIssues)} ${t('healthIssueCount')}`:t('noInfrastructureStop');
  document.getElementById('games').textContent=`${integer(s.observedGame)} / ${integer(s.maxGames)}`;document.getElementById('gamesSub').textContent=`${t('controllerConfirmed')} ${integer(s.completedGames)} · ${t('loggedRows')} ${integer(data.counts.loggedGames)}`;
  const totalSegments=Number(s.segmentGames)>0?Math.ceil(Number(s.maxGames)/Number(s.segmentGames)):'—';document.getElementById('segments').textContent=`${integer(s.completedSegments)} / ${integer(totalSegments)}`;document.getElementById('segmentsSub').textContent=s.currentSegment||'—';
  document.getElementById('hours').textContent=`${num(s.estimatedActiveHours,2)} h`;document.getElementById('hoursSub').textContent=`${t('limit')} ${num(s.maxWallHours,1)} h`;
  document.getElementById('difficulty').textContent=`L${integer(l.difficulty)} · ${integer(s.currentNodeBudget)} ${t('nodes')}`;document.getElementById('temp').textContent=`${t('referenceSearch')} ${nodeTimingLabel(s.currentNodeBudget)}`;
  renderRecentSourceCard(data,'vs_frozen','frozenRecent','frozenRecentSub');renderRecentSourceCard(data,'vs_sanmill','sanmillRecent','sanmillRecentSub');
  document.getElementById('trainingWindowLabel').textContent=`${t('cardTrainingWindow')} · ${integer(s.rollingWin)} ${t('gamesUnit')}`;document.getElementById('mixedWindow').textContent=pct(l.mixedWinRate);document.getElementById('mixedWindowSub').textContent=t('trainingDiagnosticOnly');
  const overall=data.opponentOutcomes?.overall||{},bySource=data.opponentOutcomes?.bySource||{},frozenOverall=bySource.vs_frozen?.overall||{},sanmillOverall=bySource.vs_sanmill?.overall||{},totalGames=Number(overall.total)||0;
  document.getElementById('ruleDraws').textContent=`${integer(overall.ruleDraw)} · ${pct(totalGames?Number(overall.ruleDraw)/totalGames:null)}`;document.getElementById('ruleDrawsSub').textContent=`${t('frozenShort')} ${integer(frozenOverall.ruleDraw)} · ${t('sanmillShort')} ${integer(sanmillOverall.ruleDraw)}`;
  document.getElementById('maxPly').textContent=`${integer(overall.maxPly)} · ${pct(totalGames?Number(overall.maxPly)/totalGames:null)}`;document.getElementById('maxPlySub').textContent=`${t('frozenShort')} ${integer(frozenOverall.maxPly)} · ${t('sanmillShort')} ${integer(sanmillOverall.maxPly)}`;
  document.getElementById('update').textContent=`${t('game')} ${integer(l.updateGame)}`;document.getElementById('loss').textContent=`P ${num(l.policyLoss,3)} · V ${num(l.valueLoss,3)}`;
  document.getElementById('gpuCurrent').textContent=g.available?`${num(gl.gpuUtilPct,0)}%`:t('noTrainingGpuTelemetry');document.getElementById('gpuMemory').textContent=g.available?`${num(finiteNumber(gl.memoryUsedMiB)/1024,1)} / ${num(finiteNumber(gl.memoryTotalMiB)/1024,1)} GB · ${num(gl.memoryUtilPct,1)}% · ${t('sampleCount')} ${integer(g.sampleCount)}`:`${t('postTrainingGpuSamplesIgnored')} ${integer(g.excludedOutsideTrainingWindow)}`;
  document.getElementById('components').textContent=currentLanguage==='zh'?'哨兵网络（Sentinel）：关':'Sentinel OFF';document.getElementById('componentsSub').textContent=currentLanguage==='zh'?'恢复机制：关 · 固定资源课程':'Recovery OFF · fixed-resource';
  document.getElementById('progressBar').style.width=`${Math.max(0,Math.min(100,Number(s.progressPct)||0))}%`;
  lineChart('winChart',data.series.games,[{key:'vs_frozen_score_rate_200',label:t('chart.frozenScore200'),color:COLORS.blue,width:2.4},{key:'vs_sanmill_score_rate_200',label:t('chart.sanmillScore200'),color:COLORS.orange,dash:[8,5],width:2.4}],[0,1],markers,null,sharedGameDomain);
  lineChart('topChart',data.series.games,[{key:'policy_top1_rate_smooth50',label:t('chart.policy'),color:COLORS.blue,width:2.3},{key:'heuristic_top1_rate_smooth50',label:t('chart.heuristic'),color:COLORS.orange,dash:[8,5]},{key:'malom_win_move_rate_smooth50',label:t('chart.malom'),color:COLORS.magenta,dash:[3,4]},{key:'malom_known_rate_smooth50',label:t('chart.malomKnown'),color:COLORS.cyan,dash:[11,4,2,4]}],[0,1],markers,null,sharedGameDomain);
  lineChart('exploreChart',data.series.games,[{key:'temperature',label:t('chart.temperature'),color:COLORS.orange,dash:[8,5]},{key:'chosen_prob_mean_smooth50',label:t('chart.chosenProbability'),color:COLORS.blue,width:2.3}],[0,1],markers,null,sharedGameDomain);
  lineChart('entropyChart',data.series.games,[{key:'entropy_mean_smooth50',label:t('chart.entropy'),color:COLORS.blue,width:2.3}],null,markers,null,sharedGameDomain);
  lineChart('lossChart',data.series.updates,[{key:'policy_loss',label:t('chart.policyLoss'),color:COLORS.blue,width:2.2},{key:'value_loss',label:t('chart.valueLoss'),color:COLORS.magenta,dash:[8,5],width:2.2}],null,markers,null,sharedGameDomain);
  lineChart('rewardChart',data.series.games,[{key:'reward_total_mean_smooth50',label:t('chart.totalReward'),color:COLORS.blue,width:2.3},{key:'reward_heuristic_mean_smooth50',label:t('chart.heuristicReward'),color:COLORS.orange,dash:[8,5]},{key:'reward_retro_mean_smooth50',label:t('chart.retroReward'),color:COLORS.magenta,dash:[3,4]}],null,markers,null,sharedGameDomain);
  document.getElementById('lrChartNote').textContent=learningRateNote(data.series.learningRate||[]);
  lineChart('lrChart',data.series.learningRate||[],[{key:'lr_x1e4',label:t('chart.learningRate'),color:COLORS.blue,width:2.3,stepped:true}],null,markers,null,sharedGameDomain);
  lineChart('depthChart',data.series.games,[{key:'ply_smooth50',label:t('chart.ply'),color:COLORS.blue,width:2.3}],null,markers,10,sharedGameDomain);
  lineChart('gpuChart',g.series||[],[{key:'gpuUtilPct',label:t('chart.gpuUtil'),color:COLORS.gpu,width:2.4},{key:'memoryUtilPct',label:t('chart.vramUtil'),color:COLORS.vram,dash:[8,5],width:2.4}],[0,100],markers,null,sharedGameDomain);
  const terminationSpecs=[{key:'win_fewer_than_three',color:'#56b4e9'},{key:'win_no_legal_moves',color:'#0072b2'},{key:'draw_repetition',color:'#f0e442'},{key:'draw_no_progress',color:'#e69f00'},{key:'max_ply_truncation',color:'#8f9fb3'},{key:'lose_no_legal_moves',color:'#cc79a7'},{key:'lose_fewer_than_three',color:'#d55e00'},{key:'other',color:'#b794f4'}].filter(spec=>(data.series.terminations50||[]).some(row=>Number(row[spec.key])>0));stackedAreaChart('terminationChart',data.series.terminations50||[],terminationSpecs,markers,sharedGameDomain);
  renderOutcomeTables(data);bars('terminationBars',data.counts.terminations);bars('outcomeBars',data.counts.outcomes,{win:COLORS.blue,draw:COLORS.yellow,loss:COLORS.magenta},OUTCOME_BAR_ORDER);bars('opponentBars',data.counts.opponents,{'vs_frozen':COLORS.blue,'vs_sanmill':COLORS.orange});
  const tbody=document.getElementById('segmentRows');tbody.replaceChildren();for(const seg of [...data.segments].reverse()){const tr=document.createElement('tr');for(const value of [seg.name,seg.firstGame??'—',seg.lastGame??'—',seg.gameRows,seg.updateRows,seg.checkpoint?t('yes'):'—']){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);}
  const warningHost=document.getElementById('warnings');warningHost.replaceChildren();if(data.warnings.length){const grouped=new Map();for(const warning of data.warnings){const translated=translateWarning(warning);grouped.set(translated,(grouped.get(translated)||0)+1);}for(const [warning,count] of grouped){const div=document.createElement('div');div.className='warning';div.textContent=count>1?`${warning} × ${integer(count)}`:warning;warningHost.append(div);}}else warningHost.textContent=t('none');
  const locale=currentLanguage==='zh'?'zh-CN':'en-US';document.getElementById('refresh').textContent=`${t('updatedAt')} ${new Date(data.generatedAt).toLocaleString(locale)} · ${t('malformedTail')} ${integer(data.counts.malformedLinesIgnored)}`;
}

function exportDashboardPng(){if(!lastData)return;try{const width=1600,margin=40,gap=16,cardColumns=3,cardHeight=105,cardWidth=(width-2*margin-(cardColumns-1)*gap)/cardColumns;const cards=[...document.querySelectorAll('.cards .card')];const cardRows=Math.ceil(cards.length/cardColumns);const tableHeight=285;const charts=[...document.querySelectorAll('.grid canvas')];const chartColumns=1,chartWidth=width-2*margin,chartHeight=285,chartRows=charts.length;const headerHeight=125,summaryHeight=210;const height=headerHeight+cardRows*(cardHeight+gap)+tableHeight+chartRows*(chartHeight+gap)+summaryHeight+margin;const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const c=canvas.getContext('2d');c.fillStyle='#07111f';c.fillRect(0,0,width,height);c.fillStyle='#e7eef8';c.font='700 28px system-ui';c.fillText(t('pageTitle'),margin,48);c.font='14px system-ui';c.fillStyle='#91a4bd';c.fillText(document.getElementById('identity').textContent,margin,76);c.fillText(`${t('observedEvidenceKey')} · ${t('planBoundaryKey')}`,margin,101);let y=headerHeight;
    cards.forEach((card,index)=>{const col=index%cardColumns,row=Math.floor(index/cardColumns),x=margin+col*(cardWidth+gap),cy=y+row*(cardHeight+gap);c.fillStyle='#0e1b2c';c.fillRect(x,cy,cardWidth,cardHeight);c.strokeStyle='#304762';c.strokeRect(x+.5,cy+.5,cardWidth-1,cardHeight-1);c.fillStyle='#91a4bd';c.font='12px system-ui';c.fillText(card.querySelector('.label')?.textContent||'',x+16,cy+24);c.fillStyle='#e7eef8';c.font='700 22px system-ui';c.fillText(card.querySelector('.value')?.textContent||'—',x+16,cy+56);c.fillStyle='#91a4bd';c.font='12px system-ui';c.fillText((card.querySelector('.sub')?.textContent||'').slice(0,68),x+16,cy+82);});y+=cardRows*(cardHeight+gap)+10;
    const split=lastData.opponentOutcomes||{},tableWidth=(width-2*margin-gap)/2,leftX=margin,rightX=margin+tableWidth+gap;
    const numericRight=[315,380,445,515,580,640,735];
    const headerLabels=[t('tableGames'),t('tableWins'),t('tableDraws'),t('tableRuleDraws'),t('tableMaxPly'),t('tableLosses'),t('tableScoreRate')];
    const drawTableFrame=(x,title,firstHeader)=>{
      c.fillStyle='#e7eef8';c.font='700 17px system-ui';c.textAlign='left';c.fillText(title,x,y+22);
      c.fillStyle='#122238';c.fillRect(x,y+35,tableWidth,29);
      c.strokeStyle='#304762';c.strokeRect(x+.5,y+35.5,tableWidth-1,tableHeight-51);
      c.beginPath();c.moveTo(x,y+64.5);c.lineTo(x+tableWidth,y+64.5);c.stroke();
      c.fillStyle='#91a4bd';c.font='600 11px system-ui';c.fillText(firstHeader,x+12,y+54);
      c.textAlign='right';headerLabels.forEach((label,index)=>c.fillText(label,x+numericRight[index],y+54));c.textAlign='left';
    };
    const drawWdlRow=(x,yy,item,index)=>{
      if(index%2===1){c.fillStyle='rgba(30,48,70,.28)';c.fillRect(x+1,yy-18,tableWidth-2,26);}
      c.fillStyle=item.emphasis?'#e7eef8':'#c8d5e6';c.font=item.emphasis?'600 12px system-ui':'12px system-ui';c.textAlign='left';c.fillText(item.label,x+12+(item.indent||0),yy);
      const row=item.row||{};const values=[integer(row.total),integer(row.win),integer(row.draw),integer(row.ruleDraw),integer(row.maxPly),integer(row.loss),pct(row.scoreRate)];
      c.textAlign='right';values.forEach((value,valueIndex)=>c.fillText(value,x+numericRight[valueIndex],yy));c.textAlign='left';
    };
    const sourceRows=[{label:t('allSources'),row:split.overall,emphasis:true}];
    for(const source of ['vs_frozen','vs_sanmill']){const group=split.bySource?.[source];if(!group)continue;sourceRows.push({label:valueLabel(source),row:group.overall,emphasis:true});for(const color of ['W','B']){const row=group.byLearnerColor?.[color];if(row)sourceRows.push({label:color==='W'?t('learnerWhite'):t('learnerBlack'),row,indent:18});}}
    const levelRows=(split.sanmillByLevel||[]).map(row=>({label:`L${integer(row.level)} · ${integer(row.nodeBudget)} ${t('nodes')} · ${nodeTimingLabel(row.nodeBudget)}`,row,emphasis:true}));
    drawTableFrame(leftX,t('panelOpponentOutcomes'),t('tableSource'));drawTableFrame(rightX,t('sanmillByLevel'),t('tableLevel'));
    sourceRows.forEach((item,index)=>drawWdlRow(leftX,y+86+index*27,item,index));levelRows.forEach((item,index)=>drawWdlRow(rightX,y+86+index*27,item,index));
    y+=tableHeight;
    charts.forEach((chart,index)=>{const col=index%chartColumns,row=Math.floor(index/chartColumns),x=margin+col*(chartWidth+gap),cy=y+row*(chartHeight+gap),title=chart.closest('.panel')?.querySelector('h2')?.textContent||'';c.fillStyle='#0e1b2c';c.fillRect(x,cy,chartWidth,chartHeight);c.strokeStyle='#304762';c.strokeRect(x+.5,cy+.5,chartWidth-1,chartHeight-1);c.fillStyle='#e7eef8';c.font='600 15px system-ui';c.fillText(title,x+14,cy+24);c.drawImage(chart,x+12,cy+36,chartWidth-24,chartHeight-48);});y+=chartRows*(chartHeight+gap)+12;
    c.fillStyle='#0e1b2c';c.fillRect(margin,y,width-2*margin,summaryHeight-25);c.strokeStyle='#304762';c.strokeRect(margin+.5,y+.5,width-2*margin-1,summaryHeight-26);c.fillStyle='#e7eef8';c.font='700 16px system-ui';c.fillText(t('panelSegmentEvidence'),margin+16,y+26);c.fillStyle='#c8d5e6';c.font='13px system-ui';const summaryLines=[`${t('cardSegments')}: ${integer(lastData.state.completedSegments)} / ${integer(lastData.state.observedSegments)}`,`${t('panelOutcomes')}: W ${integer(lastData.counts.outcomes?.win)} · D ${integer(lastData.counts.outcomes?.draw)} · L ${integer(lastData.counts.outcomes?.loss)}`,`${t('panelOpponents')}: ${Object.entries(lastData.counts.opponents||{}).map(([k,v])=>`${valueLabel(k)} ${integer(v)}`).join(' · ')}`,`${t('panelTerminations')}: ${Object.entries(lastData.counts.terminations||{}).map(([k,v])=>`${valueLabel(k)} ${integer(v)}`).join(' · ')}`,`${t('panelWarnings')}: ${integer(lastData.counts.warnings)} · ${t('malformedTail')} ${integer(lastData.counts.malformedLinesIgnored)}`];summaryLines.forEach((line,index)=>c.fillText(line.slice(0,190),margin+16,y+55+index*25));canvas.toBlob(blob=>{if(!blob)throw new Error('canvas encode failed');const link=document.createElement('a'),stamp=new Date(lastData.generatedAt).toISOString().replaceAll(':','-');link.href=URL.createObjectURL(blob);link.download=`${lastData.identity.planId}-${stamp}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);},'image/png');}catch(error){alert(`${t('exportFailed')}: ${error}`);}}

async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error(`${response.status}`);render(await response.json());}catch(error){document.getElementById('stateText').textContent=`${t('loadFailed')}: ${error}`;document.getElementById('stateBadge').className='badge stopped';}}
document.getElementById('languageSelect').addEventListener('change',event=>setLanguage(event.target.value));
document.getElementById('exportPng').addEventListener('click',exportDashboardPng);
document.querySelectorAll('.help-button').forEach(button=>button.addEventListener('click',()=>openHelp(button.dataset.helpKey,button)));
document.getElementById('helpClose').addEventListener('click',closeHelp);
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!document.getElementById('helpPanel').hidden)closeHelp();});
applyStaticTranslations();refresh();setInterval(refresh,5000);window.addEventListener('resize',()=>{if(lastData)render(lastData);});
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    control_dir: Path

    def _send(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode("utf-8"))
            return
        if path == "/api/status":
            try:
                status = collect_status(self.control_dir)
                payload = json.dumps(
                    status, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            except Exception as exc:  # dashboard must report, never touch training
                payload = json.dumps(
                    {"error": type(exc).__name__, "message": str(exc)},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "application/json; charset=utf-8",
                    payload,
                )
                return
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", payload)
            return
        if path == "/health":
            self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, "image/x-icon", b"")
            return
        self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        self._send(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "text/plain; charset=utf-8",
            b"read only\n",
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress five-second polling noise. Unexpected errors still reach stderr.
        if args and str(args[1]) not in {"200", "204"}:
            super().log_message(fmt, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="Permit a non-loopback bind such as 0.0.0.0",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    control_dir = args.control_dir.resolve(strict=True)
    if not (control_dir / "plan.json").is_file():
        parser.error("control directory does not contain plan.json")
    if args.once:
        # ASCII escaping keeps the diagnostic mode safe in legacy Windows shells.
        print(json.dumps(collect_status(control_dir), indent=2, ensure_ascii=True))
        return 0
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not args.allow_lan:
        parser.error("non-loopback binding requires --allow-lan")
    DashboardHandler.control_dir = control_dir
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"NMM_LLM read-only monitor: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
