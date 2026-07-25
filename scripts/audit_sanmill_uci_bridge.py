#!/usr/bin/env python3
"""Run the authorized strict Sanmill logical-turn bridge validation smoke."""

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
    assert_stable_legal_parity,
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
    "docs/experiments/sanmill-strict-uci-bridge-smoke-v2.md",
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


def run_rule_probes(
    installation: SanmillInstallation,
    *,
    node_budget: int,
) -> dict[str, Any]:
    with SanmillUciSession(installation) as session:
        session.new_game()
        session.position_startpos()
        opening_state = session.state_json()
        opening = session.search_logical_turn(max(node_budget, 100_000))
        if opening.effective_depth != 1 or opening.completed_depth != 1:
            raise SanmillBridgeError(
                "DrawOnHumanExperience opening-depth probe did not select depth 1"
            )

        session.new_game()
        session.position_fen(NO_CAPTURE_DRAW_FEN)
        no_capture_state = session.state_json()
        no_capture_search = session.search_logical_turn(node_budget)
        if no_capture_state.no_capture_count != 100:
            raise SanmillBridgeError("no-capture draw FEN lost its 100-ply counter")
        if (
            no_capture_state.winner is not None
            or no_capture_state.outcome_reason != "drawFiftyMove"
            or no_capture_search.status != "terminal"
            or no_capture_search.total_nodes != 0
        ):
            raise SanmillBridgeError("no-capture fixture has the wrong Sanmill outcome")

        session.new_game()
        session.position_startpos(THREEFOLD_PREFIX)
        repetition_before = session.state_json()
        if repetition_before.terminal:
            raise SanmillBridgeError("threefold fixture became terminal too early")
        session.position_startpos((*THREEFOLD_PREFIX, THREEFOLD_FINAL))
        repetition_after = session.state_json()
        repetition_search = session.search_logical_turn(node_budget)
        if repetition_before.repetition_current_count != 2:
            raise SanmillBridgeError(
                "threefold fixture did not retain two prior occurrences"
            )
        if (
            repetition_after.repetition_current_count != 3
            or repetition_after.outcome_reason != "drawThreefoldRepetition"
            or repetition_search.status != "terminal"
        ):
            raise SanmillBridgeError("threefold fixture has the wrong Sanmill outcome")

        session.new_game()
        session.position_startpos(STAGED_CAPTURE_PREFIX)
        capture_before = session.state_json()
        capture_board = project_stable_sanmill_fen(capture_before.fen)
        assert_stable_legal_parity(
            capture_board,
            capture_before.legal_actions,
        )
        capture_budget = max(node_budget, 500_000)
        capture_turn = session.search_logical_turn(capture_budget, depth=8)
        if capture_turn.full_turn_actions[0] != "d6-d5":
            raise SanmillBridgeError(
                "staged-capture fixture did not choose its pinned mill-forming move"
            )
        if (
            len(capture_turn.full_turn_actions) != 2
            or not capture_turn.full_turn_actions[1].startswith("x")
        ):
            raise SanmillBridgeError(
                "logical capture probe did not return its required removal"
            )
        session.position_startpos(
            (*STAGED_CAPTURE_PREFIX, *capture_turn.full_turn_actions)
        )
        capture_after = session.state_json()
        if capture_turn.resulting_fen != capture_after.fen:
            raise SanmillBridgeError(
                "logical capture result differs from authoritative replay"
            )
        if capture_after.terminal or capture_after.no_capture_count != 0:
            raise SanmillBridgeError("capture did not reset the no-capture counter")
        if (
            capture_after.repetition_history_length != 0
            or capture_after.repetition_current_count != 0
        ):
            raise SanmillBridgeError("capture did not reset repetition history")
        if (
            capture_after.logical_ply_count
            != capture_before.logical_ply_count + 1
        ):
            raise SanmillBridgeError("capture did not advance one logical ply")

        session.new_game()
        session.position_fen(FEWER_THAN_THREE_FEN)
        terminal_state = session.state_json()
        terminal_search = session.search_logical_turn(node_budget)
        if not terminal_state.terminal or terminal_state.legal_actions:
            raise SanmillBridgeError("fewer-than-three fixture is not terminal")
        if (
            terminal_state.winner != "black"
            or terminal_state.outcome_reason != "loseFewerThanThree"
            or terminal_search.status != "terminal"
        ):
            raise SanmillBridgeError("fewer-than-three fixture has the wrong outcome")

        return {
            "opening_depth_policy": {
                "node_ceiling": max(node_budget, 100_000),
                "before": opening_state.portable_record(),
                "logical_turn": opening.semantic_record(),
                "interpretation": (
                    "depth 1 proves the ordinary Sanmill opening-depth table is "
                    "active; SkillLevel=30 would select depth 30 if bypassed"
                ),
            },
            "no_capture_draw": {
                "state": no_capture_state.portable_record(),
                "logical_turn": no_capture_search.semantic_record(),
            },
            "threefold_draw": {
                "before": repetition_before.portable_record(),
                "after": repetition_after.portable_record(),
                "logical_turn": repetition_search.semantic_record(),
            },
            "compound_capture_and_reset": {
                "before": capture_before.portable_record(),
                "logical_turn": capture_turn.semantic_record(),
                "after": capture_after.portable_record(),
            },
            "fewer_than_three": {
                "state": terminal_state.portable_record(),
                "logical_turn": terminal_search.semantic_record(),
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
            before = session.state_json()
            if before.terminal:
                break
            if before.removal_pending:
                raise SanmillBridgeError("selfplay turn began with a pending removal")
            board = project_stable_sanmill_fen(before.fen)
            assert_stable_legal_parity(board, before.legal_actions)

            logical_turn = session.search_logical_turn(node_budget)
            if logical_turn.status != "ok":
                raise SanmillBridgeError(
                    "ongoing selfplay returned a terminal logical response"
                )
            actions.extend(logical_turn.full_turn_actions)
            session.position_startpos(actions)
            after = session.state_json()
            if after.fen != logical_turn.resulting_fen:
                raise SanmillBridgeError(
                    "selfplay replay differs from logical-turn resulting FEN"
                )
            if after.logical_ply_count != before.logical_ply_count + 1:
                raise SanmillBridgeError(
                    "selfplay logical-ply count did not advance exactly once"
                )
            if (
                after.action_token_count
                != before.action_token_count + len(logical_turn.full_turn_actions)
            ):
                raise SanmillBridgeError(
                    "selfplay action-token count differs from returned turn"
                )
            if after.history_sha256 == before.history_sha256:
                raise SanmillBridgeError(
                    "selfplay history identity did not change after a turn"
                )
            if after.terminal != logical_turn.terminal:
                raise SanmillBridgeError(
                    "logical-turn and replay terminal flags disagree"
                )
            if after.outcome_reason != logical_turn.outcome_reason:
                raise SanmillBridgeError(
                    "logical-turn and replay outcome reasons disagree"
                )
            if not after.terminal:
                after_board = project_stable_sanmill_fen(after.fen)
                assert_stable_legal_parity(after_board, after.legal_actions)

            turns.append(
                {
                    "turn": turn_index,
                    "before_fen": before.fen,
                    "before_history_sha256": before.history_sha256,
                    "before_logical_ply_count": before.logical_ply_count,
                    "logical_turn": logical_turn.semantic_record(),
                    "after_fen": after.fen,
                    "after_history_sha256": after.history_sha256,
                    "after_logical_ply_count": after.logical_ply_count,
                    "after_outcome": after.outcome.portable_record(),
                    "terminal": after.terminal,
                }
            )
            if after.terminal:
                break

    semantic = {
        "node_ceiling_per_logical_turn": node_budget,
        "max_complete_logical_turns": max_turns,
        "completed_turns": len(turns),
        "replayed_action_tokens": actions,
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
                state = session.state_json()
                if state.terminal or state.removal_pending:
                    raise SanmillBridgeError(
                        f"performance fixture is not a stable {phase} position"
                    )
                result = session.search_logical_turn(budget)
                nps = (
                    result.total_nodes / result.elapsed_seconds
                    if result.elapsed_seconds > 0
                    else None
                )
                samples.append(
                    {
                        "phase": phase,
                        "fen": state.fen,
                        "node_ceiling": budget,
                        "logical_turn": result.semantic_record(),
                        "elapsed_seconds": result.elapsed_seconds,
                        "nodes_per_second": nps,
                    }
                )

        compound_budget = max(max(budgets), 500_000)
        session.new_game()
        session.position_startpos(STAGED_CAPTURE_PREFIX)
        compound_state = session.state_json()
        compound = session.search_logical_turn(compound_budget, depth=8)
        if (
            len(compound.full_turn_actions) != 2
            or not compound.full_turn_actions[1].startswith("x")
        ):
            raise SanmillBridgeError(
                "compound performance fixture did not include its removal"
            )
        samples.append(
            {
                "phase": "compound_mill",
                "fen": compound_state.fen,
                "node_ceiling": compound_budget,
                "logical_turn": compound.semantic_record(),
                "elapsed_seconds": compound.elapsed_seconds,
                "nodes_per_second": (
                    compound.total_nodes / compound.elapsed_seconds
                    if compound.elapsed_seconds > 0
                    else None
                ),
            }
        )

    by_phase: dict[str, dict[str, Any]] = {}
    for phase in sorted({sample["phase"] for sample in samples}):
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
        default=(
            _ROOT
            / "out"
            / "diagnostics"
            / "sanmill-strict-uci-bridge-smoke-v2.json"
        ),
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
        "schema_version": "nmm.sanmill-strict-uci-smoke-result.v2",
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
