#!/usr/bin/env python3
"""Run the authorized strict Sanmill UCI bridge validation smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.sanmill_uci import (  # noqa: E402
    SanmillBridgeError,
    SanmillInstallation,
    SanmillUciSession,
    assert_pending_removal_parity,
    assert_stable_legal_parity,
    atomic_move_for_actions,
    inspect_sanmill_installation,
    inspect_sanmill_opening_book,
    project_stable_sanmill_fen,
    runtime_record,
    strict_contract_record,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402


NO_CAPTURE_DRAW_FEN = (
    "***OOO**/***@@@**/******** w m s 3 0 3 0 0 0 -1 -1 -1 -1 0 100 1 ids:nodes"
)
FEWER_THAN_THREE_FEN = (
    "**O**O**/**@**@**/******** w m s 2 0 2 0 0 0 -1 -1 -1 -1 0 0 1 ids:nodes"
)
THREEFOLD_PREFIX = tuple(
    "d6 f4 d2 b4 e4 d5 c4 d3 g4 d7 a4 d1 e5 e3 c3 c5 f6 b6 "
    "a4-a7 b4-a4 c4-b4 c5-c4 g4-g1 d7-g7 g1-g4 g7-d7 "
    "g4-g1 d7-g7 g1-g4".split()
)
THREEFOLD_FINAL = "g7-d7"
STAGED_CAPTURE_PREFIX = tuple(
    "d6 f4 d2 b4 g4 d7 a4 d1 d5 d3 e4 f6 f2 b2 b6 g7 a7 c3 "
    "d5-c5 c3-c4 e4-e5 c4-c3".split()
)

PERFORMANCE_POSITIONS: tuple[tuple[str, str, tuple[str, ...] | str], ...] = (
    ("placement", "startpos", ("d6", "f4", "d2", "b4")),
    ("movement", "startpos", THREEFOLD_PREFIX),
    (
        "flying",
        "fen",
        "***OOO**/***@@@**/******** w m s 3 0 3 0 0 0 -1 -1 -1 -1 0 0 1 ids:nodes",
    ),
)

_EVIDENCE_SOURCE_FILES = (
    "learned_ai/evaluation/sanmill_uci.py",
    "scripts/audit_sanmill_uci_bridge.py",
    "tests/test_sanmill_uci.py",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all budgets must be positive")
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SanmillBridgeError(f"NMM_LLM Git inspection failed: {detail}")
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_nmm_llm_source() -> dict[str, Any]:
    """Bind durable evidence to committed bridge code, tests, and auditor."""
    records: list[dict[str, str]] = []
    for relative in _EVIDENCE_SOURCE_FILES:
        _git_output("ls-files", "--error-unmatch", "--", relative)
        dirty = subprocess.run(
            ["git", "-C", str(_ROOT), "diff", "--quiet", "HEAD", "--", relative],
            check=False,
        )
        if dirty.returncode == 1:
            raise SanmillBridgeError(
                f"NMM_LLM evidence source differs from HEAD: {relative}"
            )
        if dirty.returncode != 0:
            raise SanmillBridgeError(
                f"cannot compare NMM_LLM evidence source with HEAD: {relative}"
            )
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(_ROOT / relative),
            }
        )
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "tree": _git_output("rev-parse", "HEAD^{tree}"),
        "scoped_worktree": "clean",
        "files": records,
    }


def _fen_counter(fen: str, index: int) -> int:
    fields = fen.split()
    try:
        return int(fields[index])
    except (IndexError, ValueError) as exc:
        raise SanmillBridgeError(f"cannot read FEN counter {index}: {fen}") from exc


def run_rule_probes(
    installation: SanmillInstallation,
    *,
    node_budget: int,
) -> dict[str, Any]:
    with SanmillUciSession(installation) as session:
        session.new_game()
        session.position_startpos()
        opening = session.search_fixed_nodes(max(node_budget, 100_000))
        if opening.depth != 1:
            raise SanmillBridgeError(
                "DrawOnHumanExperience opening-depth probe did not select depth 1"
            )

        session.new_game()
        session.position_fen(NO_CAPTURE_DRAW_FEN)
        no_capture_state = session.position_state()
        no_capture_search = session.probe_terminal_draw(node_budget)
        if _fen_counter(no_capture_state.fen, 15) != 100:
            raise SanmillBridgeError("no-capture draw FEN lost its 100-ply counter")
        if (
            no_capture_state.outcome.winner != "draw"
            or no_capture_state.outcome.reason != "drawFiftyMove"
        ):
            raise SanmillBridgeError("no-capture fixture has the wrong Sanmill outcome")

        session.new_game()
        session.position_startpos(THREEFOLD_PREFIX)
        repetition_before = session.position_state()
        if repetition_before.terminal:
            raise SanmillBridgeError("threefold fixture became terminal too early")
        session.position_startpos((*THREEFOLD_PREFIX, THREEFOLD_FINAL))
        repetition_after = session.position_state()
        repetition_search = session.probe_terminal_draw(node_budget)
        if "current_count=2" not in repetition_after.history:
            raise SanmillBridgeError(
                "threefold history did not retain two prior matches"
            )
        if repetition_after.outcome.reason != "drawThreefoldRepetition":
            raise SanmillBridgeError("threefold fixture has the wrong Sanmill outcome")

        session.new_game()
        session.position_startpos(STAGED_CAPTURE_PREFIX)
        capture_before = session.position_state()
        capture_board = project_stable_sanmill_fen(capture_before.fen)
        capture_nmm_moves = assert_stable_legal_parity(
            capture_board,
            capture_before.legal_actions,
        )
        capture_primary = session.search_fixed_nodes(node_budget)
        if capture_primary.bestmove != "d6-d5":
            raise SanmillBridgeError(
                "staged-capture fixture did not choose its pinned mill-forming move"
            )
        capture_actions = (*STAGED_CAPTURE_PREFIX, capture_primary.bestmove)
        session.position_startpos(capture_actions)
        capture_pending = session.position_state()
        if not capture_pending.removal_pending:
            raise SanmillBridgeError("mill-forming move did not stage a removal")
        assert_pending_removal_parity(
            capture_nmm_moves,
            capture_primary.bestmove,
            capture_pending.legal_actions,
        )
        capture_removal = session.search_fixed_nodes(node_budget)
        if not capture_removal.bestmove.startswith("x"):
            raise SanmillBridgeError("staged-capture probe did not choose a removal")
        atomic_move_for_actions(
            capture_nmm_moves,
            capture_primary.bestmove,
            capture_removal.bestmove,
        )
        session.position_startpos((*capture_actions, capture_removal.bestmove))
        capture_after = session.position_state()
        if capture_after.terminal or _fen_counter(capture_after.fen, 15) != 0:
            raise SanmillBridgeError("capture did not reset the no-capture counter")
        if (
            "root_reset=true" not in capture_after.history
            or "len=0" not in capture_after.history
        ):
            raise SanmillBridgeError("capture did not reset repetition history")

        session.new_game()
        session.position_fen(FEWER_THAN_THREE_FEN)
        terminal_state = session.position_state()
        if not terminal_state.terminal or terminal_state.legal_actions:
            raise SanmillBridgeError("fewer-than-three fixture is not terminal")
        if (
            terminal_state.outcome.winner != "black"
            or terminal_state.outcome.reason != "loseFewerThanThree"
        ):
            raise SanmillBridgeError("fewer-than-three fixture has the wrong outcome")

        return {
            "opening_depth_policy": {
                "node_ceiling": max(node_budget, 100_000),
                "result": opening.semantic_record(),
                "interpretation": (
                    "depth 1 proves the ordinary Sanmill opening-depth table is "
                    "active; SkillLevel=30 would select depth 30 if bypassed"
                ),
            },
            "no_capture_draw": {
                "state": no_capture_state.portable_record(),
                "search": no_capture_search.semantic_record(),
            },
            "threefold_draw": {
                "before": repetition_before.portable_record(),
                "after": repetition_after.portable_record(),
                "search": repetition_search.semantic_record(),
            },
            "staged_capture_and_reset": {
                "before": capture_before.portable_record(),
                "primary_search": capture_primary.semantic_record(),
                "pending": capture_pending.portable_record(),
                "removal_search": capture_removal.semantic_record(),
                "after": capture_after.portable_record(),
            },
            "fewer_than_three": {
                "state": terminal_state.portable_record(),
                "search_attempted": False,
                "reason": (
                    "the fail-closed build rejects MOVE_NONE before Sanmill's "
                    "release fallback; terminal phase and empty legal set are "
                    "queried before every search"
                ),
            },
        }


def run_selfplay(
    installation: SanmillInstallation,
    *,
    node_budget: int,
    max_turns: int,
) -> dict[str, Any]:
    actions: list[str] = []
    turns: list[dict[str, Any]] = []
    with SanmillUciSession(installation) as session:
        session.new_game()
        for turn_index in range(1, max_turns + 1):
            session.position_startpos(actions)
            before = session.position_state()
            if before.terminal:
                break
            if before.removal_pending:
                raise SanmillBridgeError("selfplay turn began with a pending removal")
            board = project_stable_sanmill_fen(before.fen)
            nmm_moves = assert_stable_legal_parity(board, before.legal_actions)

            primary = session.search_fixed_nodes(node_budget)
            if primary.terminal_token:
                raise SanmillBridgeError("ongoing selfplay returned a terminal token")
            actions.append(primary.bestmove)
            session.position_startpos(actions)
            staged = session.position_state()

            removal_result = None
            removal_action = None
            if staged.removal_pending:
                assert_pending_removal_parity(
                    nmm_moves,
                    primary.bestmove,
                    staged.legal_actions,
                )
                removal_result = session.search_fixed_nodes(node_budget)
                if not removal_result.bestmove.startswith("x"):
                    raise SanmillBridgeError("removal search did not choose a removal")
                removal_action = removal_result.bestmove
                actions.append(removal_action)
                session.position_startpos(actions)
            elif staged.terminal:
                if staged.legal_actions:
                    raise SanmillBridgeError("terminal staged state advertised actions")

            atomic_move_for_actions(nmm_moves, primary.bestmove, removal_action)
            after = session.position_state()
            if not after.terminal:
                after_board = project_stable_sanmill_fen(after.fen)
                assert_stable_legal_parity(after_board, after.legal_actions)

            turns.append(
                {
                    "turn": turn_index,
                    "before_fen": before.fen,
                    "before_outcome": before.outcome.portable_record(),
                    "primary": primary.semantic_record(),
                    "removal": (
                        removal_result.semantic_record()
                        if removal_result is not None
                        else None
                    ),
                    "after_fen": after.fen,
                    "after_outcome": after.outcome.portable_record(),
                    "terminal": after.terminal,
                }
            )
            if after.terminal:
                break

    semantic = {
        "node_ceiling_per_search": node_budget,
        "max_complete_turns": max_turns,
        "completed_turns": len(turns),
        "staged_actions": actions,
        "turns": turns,
        "stopped_at_ceiling": bool(
            turns and len(turns) == max_turns and not turns[-1]["terminal"]
        ),
    }
    return {**semantic, "semantic_identity": canonical_sha256(semantic)}


def run_performance_probes(
    installation: SanmillInstallation,
    *,
    budgets: Sequence[int],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    with SanmillUciSession(installation, search_timeout=300.0) as session:
        for phase, source_kind, source in PERFORMANCE_POSITIONS:
            for budget in budgets:
                session.new_game()
                if source_kind == "startpos":
                    assert isinstance(source, tuple)
                    session.position_startpos(source)
                else:
                    assert isinstance(source, str)
                    session.position_fen(source)
                state = session.position_state()
                if state.terminal or state.removal_pending:
                    raise SanmillBridgeError(
                        f"performance fixture is not a stable {phase} position"
                    )
                result = session.search_fixed_nodes(budget)
                nps = (
                    result.nodes / result.elapsed_seconds
                    if result.elapsed_seconds > 0
                    else None
                )
                samples.append(
                    {
                        "phase": phase,
                        "fen": state.fen,
                        "node_ceiling": budget,
                        "result": result.semantic_record(),
                        "elapsed_seconds": result.elapsed_seconds,
                        "nodes_per_second": nps,
                    }
                )

    by_phase: dict[str, dict[str, Any]] = {}
    for phase, _, _ in PERFORMANCE_POSITIONS:
        phase_samples = [sample for sample in samples if sample["phase"] == phase]
        rates = [
            float(sample["nodes_per_second"])
            for sample in phase_samples
            if sample["nodes_per_second"] is not None
        ]
        by_phase[phase] = {
            "samples": len(phase_samples),
            "median_nodes_per_second": statistics.median(rates),
        }
    return {"samples": samples, "summary": by_phase}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=_ROOT / "data" / "training_paths.local.json",
    )
    parser.add_argument("--node-budget", type=_positive_int, default=10_000)
    parser.add_argument("--max-turns", type=_positive_int, default=60)
    parser.add_argument(
        "--performance-budgets",
        type=_csv_positive_ints,
        default=(1_000, 10_000, 100_000, 500_000),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "out" / "diagnostics" / "sanmill-uci-bridge-smoke.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nmm_llm_source = inspect_nmm_llm_source()
    installation = inspect_sanmill_installation(args.paths_config)
    opening_book = inspect_sanmill_opening_book(installation)
    rule_probes = run_rule_probes(installation, node_budget=args.node_budget)
    first = run_selfplay(
        installation,
        node_budget=args.node_budget,
        max_turns=args.max_turns,
    )
    second = run_selfplay(
        installation,
        node_budget=args.node_budget,
        max_turns=args.max_turns,
    )
    if first != second:
        raise SanmillBridgeError(
            "fresh-process selfplays differ after timing fields were excluded"
        )
    performance = run_performance_probes(
        installation,
        budgets=args.performance_budgets,
    )
    payload = {
        "schema_version": "nmm.sanmill-strict-uci-smoke-result.v1",
        "status": "passed",
        "claim_boundary": (
            "bridge/rule/reproducibility/performance evidence only; no candidate "
            "was loaded and no playing-strength evaluation was run"
        ),
        "runtime": runtime_record(),
        "nmm_llm_source": nmm_llm_source,
        "installation": installation.portable_record(),
        "contract": strict_contract_record(),
        "opening_book_gate": opening_book.portable_record(),
        "rule_probes": rule_probes,
        "reproducibility": {
            "fresh_process_runs": 2,
            "equal": True,
            "semantic_identity": first["semantic_identity"],
            "run": first,
        },
        "performance": performance,
    }
    payload["evidence_identity"] = canonical_sha256(payload)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "evidence_identity": payload["evidence_identity"],
                "completed_turns": first["completed_turns"],
                "selfplay_identity": first["semantic_identity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
