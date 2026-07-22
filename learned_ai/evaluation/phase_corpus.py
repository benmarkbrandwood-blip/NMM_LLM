"""Draft phase-covered starts from a pinned Sanmill rules-replay fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.evaluation.oracle_corpus import (
    _nmm_board,
    _parse_board_nodes,
    render_review_assets,
    ring16_canonical_fen,
    validate_review_manifest,
)
from learned_ai.training.run_contract import canonical_sha256


PHASE_CORPUS_SCHEMA = "nmm.phase-covered-corpus.v1"
PHASE_CORPUS_ID = "dev-v4-phase-covered-corpus-v1"
PHASE_CORPUS_STATUS = "draft_for_owner_and_product_review_not_frozen"
EXPECTED_SANMILL_COMMIT = "f8be034d10ba7b293f9c10d661b1cfd81e9be096"
SOURCE_ASSET = PurePosixPath("crates/tgf-mill/testdata/legacy_oracle/0.json")
EXPECTED_SOURCE_BLOB = "a12eb55c32b5b6ccda0e1fbbde3761d384b453af"
EXPECTED_SOURCE_SHA256 = (
    "4b0ffc4e0d754e9cf2b3275726c366109173d303b1fa2dc0965c76d2db2b1f03"
)

_PHASE_ORDER = {"placement": 0, "movement": 1, "flying": 2}
_QUOTAS = {
    "placement": {
        ("W", "D"): 4,
        ("B", "D"): 4,
        ("W", "L"): 3,
        ("B", "L"): 3,
        ("W", "W"): 4,
        ("B", "W"): 4,
    },
    "movement": {
        ("W", "D"): 4,
        ("B", "D"): 3,
        ("W", "L"): 3,
        ("B", "L"): 4,
        ("W", "W"): 3,
        ("B", "W"): 4,
    },
    "flying": {
        ("W", "D"): 4,
        ("B", "D"): 5,
        ("W", "L"): 5,
        ("B", "L"): 4,
        ("W", "W"): 2,
        ("B", "W"): 1,
    },
}


class PhaseCorpusError(RuntimeError):
    """Raised when the phase corpus cannot be reproduced exactly."""


@dataclass(frozen=True)
class ProjectedStep:
    fen: str
    action: str
    tgf_phase: str


@dataclass(frozen=True)
class _Candidate:
    fen: str
    phase: str
    wdl: str
    transform: str
    source: dict[str, Any]

    @property
    def source_identity(self) -> str:
        return str(self.source["source_identity"])

    @property
    def selection_key(self) -> str:
        return hashlib.sha256(
            (
                f"{PHASE_CORPUS_ID}|{self.phase}|{self.wdl}|"
                f"{self.transform}|{self.fen}"
            ).encode("utf-8")
        ).hexdigest()


def _run_git(checkout: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={checkout.resolve().as_posix()}",
        "-C",
        str(checkout),
        *arguments,
    ]
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PhaseCorpusError("cannot read the pinned Sanmill fixture") from exc


def load_pinned_tgf_fixture(
    paths_config: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact standard-NMM TGF fixture through the local path registry."""
    try:
        config = json.loads(Path(paths_config).read_text(encoding="utf-8"))
        checkout = Path(config["sanmill_checkout"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise PhaseCorpusError("cannot resolve sanmill_checkout") from exc
    if not checkout.is_dir():
        raise PhaseCorpusError("sanmill_checkout is not a directory")

    object_name = f"{EXPECTED_SANMILL_COMMIT}:{SOURCE_ASSET.as_posix()}"
    blob = _run_git(checkout, "rev-parse", object_name).decode().strip()
    if blob != EXPECTED_SOURCE_BLOB:
        raise PhaseCorpusError("Sanmill source blob identity changed")
    source_bytes = _run_git(checkout, "show", object_name)
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise PhaseCorpusError("Sanmill source content identity changed")
    try:
        fixture = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseCorpusError("cannot parse the pinned TGF fixture") from exc
    if fixture.get("rule_idx") != 0 or fixture.get("rule_name") != (
        "Nine Men's Morris"
    ):
        raise PhaseCorpusError("pinned TGF fixture is not standard NMM")
    source = {
        "path_lookup_key": "sanmill_checkout",
        "sanmill_commit": EXPECTED_SANMILL_COMMIT,
        "asset_path": SOURCE_ASSET.as_posix(),
        "git_blob": EXPECTED_SOURCE_BLOB,
        "asset_sha256": EXPECTED_SOURCE_SHA256,
        "fixture_version": fixture.get("version"),
        "fixture_generator_git_sha": fixture.get("generator_git_sha"),
        "rule_idx": fixture.get("rule_idx"),
        "rule_name": fixture.get("rule_name"),
    }
    return fixture, source


def project_tgf_fen(tgf_fen: str) -> ProjectedStep | None:
    """Project one stable TGF state; pending-removal states return ``None``."""
    fields = tgf_fen.split()
    if len(fields) < 8:
        raise PhaseCorpusError("TGF FEN has too few fields")
    board_text, side, tgf_phase, action = fields[:4]
    if action == "r":
        return None
    if action not in ("p", "s"):
        raise PhaseCorpusError(f"unsupported stable TGF action: {action!r}")
    if side not in ("w", "b") or tgf_phase not in ("p", "m"):
        raise PhaseCorpusError("TGF side or phase is unsupported")
    try:
        white_on_board = int(fields[4])
        white_in_hand = int(fields[5])
        black_on_board = int(fields[6])
        black_in_hand = int(fields[7])
    except ValueError as exc:
        raise PhaseCorpusError("TGF piece counters are invalid") from exc
    nmm_board = _nmm_board(_parse_board_nodes(board_text))
    if nmm_board.count("W") != white_on_board:
        raise PhaseCorpusError("TGF white on-board count is inconsistent")
    if nmm_board.count("B") != black_on_board:
        raise PhaseCorpusError("TGF black on-board count is inconsistent")
    white_placed = 9 - white_in_hand
    black_placed = 9 - black_in_hand
    if not 0 <= white_placed <= 9 or not 0 <= black_placed <= 9:
        raise PhaseCorpusError("TGF reserve counters are outside standard NMM")
    turn = "W" if side == "w" else "B"
    fen = f"{nmm_board}|{turn}|{white_placed}|{black_placed}"
    board = BoardState.from_fen_string(fen)
    expected_phase = "place" if tgf_phase == "p" else "move"
    if board.phase != expected_phase:
        raise PhaseCorpusError("TGF and NMM phases disagree")
    return ProjectedStep(fen=fen, action=action, tgf_phase=tgf_phase)


def swap_colors_and_turn(fen: str) -> str:
    """Apply the standard color involution to a compact NMM FEN."""
    board_text, turn, white_placed, black_placed = fen.split("|")
    swapped_board = board_text.translate(str.maketrans({"W": "B", "B": "W"}))
    swapped_turn = "B" if turn == "W" else "W"
    return (
        f"{swapped_board}|{swapped_turn}|{black_placed}|{white_placed}"
    )


def _phase(board: BoardState) -> str | None:
    if board.phase == "place":
        return "placement"
    if board.pieces_on_board[board.turn] == 3:
        return "flying"
    if 3 in board.pieces_on_board.values():
        return None
    return "movement"


def _source_record(
    trajectory_index: int,
    trajectory: Mapping[str, Any],
    step_index: int,
    step: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "trajectory_index": trajectory_index,
        "trajectory_seed": trajectory.get("seed"),
        "step_index": step_index,
        "ply": step.get("ply"),
        "tgf_fen": step.get("fen"),
        "side_to_move": step.get("side_to_move"),
        "phase_tag": step.get("phase_tag"),
        "picked_uci": step.get("picked_uci"),
        "legal_uci_count": len(step.get("legal_uci", [])),
    }
    return {**body, "source_identity": canonical_sha256(body)}


def _resource_candidate(
    *,
    fen: str,
    phase: str,
    transform: str,
    source: dict[str, Any],
    policy,
) -> _Candidate | None:
    board = BoardState.from_fen_string(fen)
    if policy.human_db.query_position(board) is not None:
        return None
    if policy.specialist_db.query_wdl_evidence(
        board,
        min_samples=0,
    ) is not None:
        return None
    wdl = policy.malom.query(board)
    if wdl not in ("W", "D", "L"):
        return None
    return _Candidate(
        fen=fen,
        phase=phase,
        wdl=wdl,
        transform=transform,
        source=source,
    )


def _select_candidates(pools: Sequence[_Candidate]) -> list[_Candidate]:
    selected: list[_Candidate] = []
    used_sources: set[str] = set()
    used_orbits: set[str] = set()
    for phase in ("placement", "movement", "flying"):
        for (turn, wdl), required in _QUOTAS[phase].items():
            eligible = sorted(
                (
                    candidate
                    for candidate in pools
                    if candidate.phase == phase
                    and BoardState.from_fen_string(candidate.fen).turn == turn
                    and candidate.wdl == wdl
                    and candidate.source_identity not in used_sources
                    and ring16_canonical_fen(candidate.fen) not in used_orbits
                ),
                key=lambda candidate: candidate.selection_key,
            )
            chosen = eligible[:required]
            if len(chosen) != required:
                raise PhaseCorpusError(
                    f"insufficient {phase} {turn}/{wdl} candidates: "
                    f"need {required}, found {len(chosen)}"
                )
            selected.extend(chosen)
            used_sources.update(item.source_identity for item in chosen)
            used_orbits.update(ring16_canonical_fen(item.fen) for item in chosen)
    return sorted(
        selected,
        key=lambda item: (_PHASE_ORDER[item.phase], item.selection_key),
    )


def build_phase_corpus(
    fixture: Mapping[str, Any],
    source: Mapping[str, Any],
    policy,
    *,
    review_asset_directory: str,
) -> dict[str, Any]:
    """Build the deterministic 22/21/21 draft and its provenance evidence."""
    trajectories = fixture.get("trajectories")
    if not isinstance(trajectories, Sequence):
        raise PhaseCorpusError("TGF fixture trajectories are missing")

    unique_steps: dict[str, tuple[str, dict[str, Any]]] = {}
    audit = {
        "trajectory_count": len(trajectories),
        "raw_step_count": 0,
        "pending_removal_count": 0,
        "stable_step_count": 0,
        "terminal_or_unplayable_count": 0,
        "duplicate_ring16_count": 0,
    }
    for trajectory_index, trajectory in enumerate(trajectories, 1):
        steps = trajectory.get("steps")
        if not isinstance(steps, Sequence):
            raise PhaseCorpusError("TGF trajectory steps are missing")
        for step_index, step in enumerate(steps, 1):
            audit["raw_step_count"] += 1
            projected = project_tgf_fen(str(step.get("fen", "")))
            if projected is None:
                audit["pending_removal_count"] += 1
                continue
            audit["stable_step_count"] += 1
            board = BoardState.from_fen_string(projected.fen)
            if is_terminal(board)[0] or not get_all_legal_moves(board):
                audit["terminal_or_unplayable_count"] += 1
                continue
            orbit = ring16_canonical_fen(projected.fen)
            if orbit in unique_steps:
                audit["duplicate_ring16_count"] += 1
                continue
            unique_steps[orbit] = (
                projected.fen,
                _source_record(
                    trajectory_index,
                    trajectory,
                    step_index,
                    step,
                ),
            )

    pools: list[_Candidate] = []
    phase_pool_counts = {"placement": 0, "movement": 0, "flying": 0}
    skipped_opponent_flying = 0
    for fen, source_record in unique_steps.values():
        board = BoardState.from_fen_string(fen)
        phase = _phase(board)
        if phase is None:
            skipped_opponent_flying += 1
            continue
        phase_pool_counts[phase] += 1
        direct = _resource_candidate(
            fen=fen,
            phase=phase,
            transform="identity",
            source=source_record,
            policy=policy,
        )
        if direct is not None:
            pools.append(direct)
        if phase == "flying":
            swapped_fen = swap_colors_and_turn(fen)
            swapped = _resource_candidate(
                fen=swapped_fen,
                phase=phase,
                transform="swap-colors-and-turn",
                source=source_record,
                policy=policy,
            )
            if swapped is not None:
                pools.append(swapped)

    selected = _select_candidates(pools)
    entries: list[dict[str, Any]] = []
    review_root = PurePosixPath(review_asset_directory)
    for index, candidate in enumerate(selected, 1):
        board = BoardState.from_fen_string(candidate.fen)
        source_item = {
            **candidate.source,
            "color_transform": candidate.transform,
            "oracle_moves": [],
            "nmm_illegal_oracle_moves": [],
        }
        entries.append(
            {
                "index": index,
                "fen": candidate.fen,
                "ring16_canonical_fen": ring16_canonical_fen(candidate.fen),
                "phase": candidate.phase,
                "turn": board.turn,
                "malom_wdl_for_side_to_move": candidate.wdl,
                "legal_move_count": len(get_all_legal_moves(board)),
                "pieces": {
                    "white_placed": board.pieces_placed["W"],
                    "black_placed": board.pieces_placed["B"],
                    "white_on_board": board.pieces_on_board["W"],
                    "black_on_board": board.pieces_on_board["B"],
                },
                "human_db_exact_overlap": False,
                "specialist_db_exact_overlap": False,
                "sources": [source_item],
                "review_title_prefix": "Phase candidate",
                "review_detail_lines": [
                    f"phase {candidate.phase} | Malom {candidate.wdl} | "
                    f"trajectory {source_item['trajectory_index']} "
                    f"step {source_item['step_index']}",
                    f"transform {candidate.transform} | HumanDB/SpecialistDB "
                    "exact overlap no/no",
                ],
                "review_footer": (
                    "DRAFT ONLY — legal seeded replay, not expert play"
                ),
                "review_png": (
                    review_root
                    / "positions"
                    / f"phase-{index:03d}.png"
                ).as_posix(),
            }
        )

    starts = [entry["fen"] for entry in entries]
    route_resources = {
        name: value["identity"]
        for name, value in policy.manifest["resources"].items()
    }
    payload = {
        "schema_version": PHASE_CORPUS_SCHEMA,
        "corpus_id": PHASE_CORPUS_ID,
        "evaluation_id": "dev-v4-training-aligned-paired-eval-v1-draft",
        "status": PHASE_CORPUS_STATUS,
        "image_prefix": "phase",
        "review_sheet_title": "phase candidates",
        "source": dict(source),
        "source_characterization": {
            "purpose": "deterministic standard-NMM rules-replay fixture",
            "selection_signal": "seeded picked legal actions",
            "expert_quality": False,
            "strength_oracle": False,
        },
        "route_bundle_identity": policy.bundle_identity,
        "resource_identities": route_resources,
        "selection_contract": {
            "algorithm": "sha256-stratified-first-v1",
            "phase_counts": {"placement": 22, "movement": 21, "flying": 21},
            "turn_counts": {"W": 32, "B": 32},
            "quotas": {
                phase: {
                    f"{turn}-{wdl}": count
                    for (turn, wdl), count in quotas.items()
                }
                for phase, quotas in _QUOTAS.items()
            },
            "flying_definition": "side-to-move-has-exactly-three-pieces",
            "color_transform": (
                "flying candidates may swap both colors and the side to move"
            ),
            "required_db_overlap": "absent-from-HumanDB-and-final-SpecialistDB",
        },
        "pool_audit": {
            **audit,
            "unique_stable_ring16_count": len(unique_steps),
            "phase_pool_counts": phase_pool_counts,
            "opponent_only_flying_skipped": skipped_opponent_flying,
            "eligible_variant_count": len(pools),
        },
        "claim_boundaries": [
            "draft-not-frozen",
            "requires-human-board-review",
            "requires-baseline-and-work-budget-decision",
            "not-expert-play",
            "not-strength-or-promotion-evidence",
        ],
        "start_positions": starts,
        "start_positions_sha256": canonical_sha256(starts),
        "entries": entries,
    }
    validate_phase_corpus(payload, require_identity=False)
    return payload


def _json_document(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_phase_review_package(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
    start_positions_path: str | Path,
    asset_root: str | Path,
) -> dict[str, Any]:
    """Persist a non-overwriting draft JSON, FEN list, and PNG review package."""
    output = Path(output_path)
    starts_output = Path(start_positions_path)
    assets = Path(asset_root)
    for target in (output, starts_output, assets):
        if target.exists():
            raise FileExistsError(f"phase corpus target exists: {target}")
    try:
        starts_relative = starts_output.resolve().relative_to(
            output.parent.resolve()
        )
    except ValueError as exc:
        raise PhaseCorpusError(
            "start-position list must be beside the corpus JSON"
        ) from exc
    starts = list(payload["start_positions"])
    start_bytes = _json_document(starts)
    review_assets = render_review_assets(payload, assets)
    complete = {
        **dict(payload),
        "start_positions_artifact": {
            "path": starts_relative.as_posix(),
            "file_sha256": hashlib.sha256(start_bytes).hexdigest(),
        },
        "review_assets": review_assets,
    }
    complete = {**complete, "corpus_identity": canonical_sha256(complete)}
    validate_phase_corpus(complete)
    validate_review_manifest(
        assets,
        expected_individuals=64,
        expected_sheets=6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    starts_output.parent.mkdir(parents=True, exist_ok=True)
    starts_output.write_bytes(start_bytes)
    output.write_bytes(_json_document(complete))
    return complete


def validate_phase_corpus(
    payload: Mapping[str, Any],
    *,
    require_identity: bool = True,
) -> dict[str, Any]:
    """Validate the frozen facts that do not require machine-local databases."""
    if payload.get("schema_version") != PHASE_CORPUS_SCHEMA:
        raise PhaseCorpusError("unsupported phase corpus schema")
    if payload.get("corpus_id") != PHASE_CORPUS_ID:
        raise PhaseCorpusError("phase corpus ID changed")
    if payload.get("status") != PHASE_CORPUS_STATUS:
        raise PhaseCorpusError("phase corpus is not an unfrozen review draft")
    entries = payload.get("entries")
    starts = payload.get("start_positions")
    if not isinstance(entries, Sequence) or not isinstance(starts, Sequence):
        raise PhaseCorpusError("phase corpus entries or starts are missing")
    if len(entries) != 64 or len(starts) != 64:
        raise PhaseCorpusError("phase corpus must contain 64 starts")
    if list(starts) != [entry.get("fen") for entry in entries]:
        raise PhaseCorpusError("entry and start order differ")
    if canonical_sha256(list(starts)) != payload.get("start_positions_sha256"):
        raise PhaseCorpusError("phase start identity mismatch")
    if require_identity:
        body = dict(payload)
        identity = body.pop("corpus_identity", None)
        if canonical_sha256(body) != identity:
            raise PhaseCorpusError("phase corpus identity mismatch")

    observed_quotas: dict[str, dict[tuple[str, str], int]] = {
        phase: {} for phase in _QUOTAS
    }
    exact: set[str] = set()
    orbits: set[str] = set()
    source_ids: set[str] = set()
    for expected_index, entry in enumerate(entries, 1):
        if entry.get("index") != expected_index:
            raise PhaseCorpusError("phase corpus indices are not contiguous")
        fen = str(entry.get("fen"))
        board = BoardState.from_fen_string(fen)
        if board.to_fen_string() != fen:
            raise PhaseCorpusError("phase corpus FEN does not round-trip")
        if is_terminal(board)[0] or not get_all_legal_moves(board):
            raise PhaseCorpusError("phase corpus contains an unplayable start")
        phase = str(entry.get("phase"))
        if _phase(board) != phase:
            raise PhaseCorpusError("phase classification is incorrect")
        wdl = str(entry.get("malom_wdl_for_side_to_move"))
        if wdl not in ("W", "D", "L"):
            raise PhaseCorpusError("phase corpus WDL is invalid")
        key = (board.turn, wdl)
        observed_quotas[phase][key] = observed_quotas[phase].get(key, 0) + 1
        orbit = ring16_canonical_fen(fen)
        if entry.get("ring16_canonical_fen") != orbit:
            raise PhaseCorpusError("phase corpus ring16 identity is incorrect")
        if fen in exact or orbit in orbits:
            raise PhaseCorpusError("phase corpus is not exact/ring16 unique")
        exact.add(fen)
        orbits.add(orbit)
        if entry.get("human_db_exact_overlap") is not False:
            raise PhaseCorpusError("HumanDB overlap claim changed")
        if entry.get("specialist_db_exact_overlap") is not False:
            raise PhaseCorpusError("SpecialistDB overlap claim changed")
        sources = entry.get("sources")
        if not isinstance(sources, Sequence) or len(sources) != 1:
            raise PhaseCorpusError("phase source provenance is incomplete")
        source = sources[0]
        source_body = {
            key: source.get(key)
            for key in (
                "trajectory_index",
                "trajectory_seed",
                "step_index",
                "ply",
                "tgf_fen",
                "side_to_move",
                "phase_tag",
                "picked_uci",
                "legal_uci_count",
            )
        }
        source_identity = canonical_sha256(source_body)
        if source.get("source_identity") != source_identity:
            raise PhaseCorpusError("phase source identity mismatch")
        if source_identity in source_ids:
            raise PhaseCorpusError("one TGF source step was selected twice")
        source_ids.add(source_identity)
        projected = project_tgf_fen(str(source.get("tgf_fen")))
        if projected is None:
            raise PhaseCorpusError("pending removal was used as a start")
        transform = source.get("color_transform")
        expected_fen = (
            projected.fen
            if transform == "identity"
            else swap_colors_and_turn(projected.fen)
            if transform == "swap-colors-and-turn"
            else None
        )
        if expected_fen != fen:
            raise PhaseCorpusError("phase source projection changed")
    if observed_quotas != _QUOTAS:
        raise PhaseCorpusError("phase/turn/WDL quotas changed")
    return {
        "starts": len(entries),
        "ring16_orbits": len(orbits),
        "source_steps": len(source_ids),
        "phase_counts": {
            phase: sum(counts.values())
            for phase, counts in observed_quotas.items()
        },
    }
