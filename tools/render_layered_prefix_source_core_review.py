"""Render and verify the frozen 64-member source-core review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.oracle_corpus import (
    _load_fonts,
    render_position_image,
)
from learned_ai.training.run_contract import canonical_sha256


REVIEW_SCHEMA = "nmm.layered-opening-prefix-source-core-review.v1"
REVIEW_STATUS = "source_membership_review_only_not_execution_authority"
DEFAULT_SOURCE = Path(
    "docs/experiments/"
    "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
)
DEFAULT_OUTPUT = Path(
    "docs/experiments/assets/"
    "sanmill-layered-opening-prefix-v2-source-core-2026-08-01"
)
STRATUM_LABELS = {
    "book": "Book",
    "human_db": "HumanDB",
    "perfect_db": "Perfect DB",
}


class SourceCoreReviewError(RuntimeError):
    """Raised when the source-core review package does not verify."""


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SourceCoreReviewError(
            f"review input is outside the repository: {path}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_png(image: Image.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, compress_level=9)
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "width": image.width,
        "height": image.height,
    }


def _contact_sheet(
    panels: Sequence[Image.Image],
    *,
    title: str,
) -> Image.Image:
    if not panels:
        raise SourceCoreReviewError("cannot render an empty contact sheet")
    columns = 4
    thumb_size = (360, 420)
    title_height = 58
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], title_height + rows * thumb_size[1]),
        "#ddd5c8",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), title, fill="#2b2722", font=_load_fonts()["sheet"])
    for index, panel in enumerate(panels):
        thumbnail = panel.resize(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0]
        y = title_height + (index // columns) * thumb_size[1]
        sheet.paste(thumbnail, (x, y))
    return sheet


def _detail_line(record: Mapping[str, Any]) -> str:
    stratum = str(record["stratum"])
    if stratum == "book":
        return (
            f"{record['family']} | {record['source_name']} | "
            f"{record['source_member_id']}"
        )
    if stratum == "human_db":
        return (
            f"ledger rank {record['ledger_rank']} | "
            f"{record['distinct_game_count']} distinct games | "
            f"{record['occurrence_count']} occurrences"
        )
    if stratum == "perfect_db":
        theory = record["theory_summary"]
        return (
            f"{record['route_id']} | seed {record['route_seed']} | "
            f"tied/single {theory['tied_best_step_count']}/"
            f"{theory['single_best_step_count']}"
        )
    raise SourceCoreReviewError(f"unexpected stratum: {stratum}")


def _panel_entry(
    record: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    fen = str(record["final"]["nmm_fen"])
    board = BoardState.from_fen_string(fen)
    stratum = str(record["stratum"])
    label = STRATUM_LABELS.get(stratum)
    if label is None:
        raise SourceCoreReviewError(f"unexpected stratum: {stratum}")
    return {
        "index": index,
        "fen": fen,
        "pieces": {
            "white_placed": board.pieces_placed["W"],
            "black_placed": board.pieces_placed["B"],
            "white_on_board": board.pieces_on_board["W"],
            "black_on_board": board.pieces_on_board["B"],
        },
        "legal_move_count": len(get_all_legal_moves(board)),
        "sources": [],
        "review_title_prefix": f"{label} · {record['stratum_member_id']}",
        "review_detail_lines": [
            (
                f"{record['source_core_id']} | {record['source_subtype']} | "
                "12 logical plies"
            ),
            _detail_line(record),
        ],
        "review_footer": (
            "Membership review only · execution: "
            f"{record['execution_record_status']}"
        ),
    }


def build_review_assets(
    source_path: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Render all 64 positions plus six stratum contact sheets."""
    if output.exists():
        raise FileExistsError(f"review asset directory exists: {output}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    core = source["source_core"]
    records = list(core["records"])
    if len(records) != 64:
        raise SourceCoreReviewError("source core does not contain 64 records")
    ordered_fens = [str(item["final"]["nmm_fen"]) for item in records]
    ordered_fens_identity = canonical_sha256(ordered_fens)
    output.mkdir(parents=True)

    assets: list[dict[str, Any]] = []
    individuals: list[dict[str, Any]] = []
    panels_by_stratum: dict[str, list[tuple[Mapping[str, Any], Image.Image]]] = (
        defaultdict(list)
    )
    for index, record in enumerate(records, 1):
        entry = _panel_entry(record, index=index)
        panel = render_position_image(
            entry,
            total=len(records),
            start_positions_sha256=ordered_fens_identity,
        )
        relative = Path("positions") / f"{record['source_core_id']}.png"
        metadata = _save_png(panel, output / relative)
        asset = {
            **metadata,
            "path": relative.as_posix(),
            "source_core_id": record["source_core_id"],
            "stratum": record["stratum"],
        }
        assets.append(asset)
        individuals.append(asset)
        panels_by_stratum[str(record["stratum"])].append((record, panel))

    sheets: list[dict[str, Any]] = []
    per_sheet = 12
    for stratum in ("book", "human_db", "perfect_db"):
        grouped = panels_by_stratum[stratum]
        sheet_count = math.ceil(len(grouped) / per_sheet)
        for sheet_index in range(sheet_count):
            chunk = grouped[
                sheet_index * per_sheet : (sheet_index + 1) * per_sheet
            ]
            first = str(chunk[0][0]["stratum_member_id"])
            last = str(chunk[-1][0]["stratum_member_id"])
            sheet = _contact_sheet(
                [panel for _record, panel in chunk],
                title=(
                    "Twelve-ply source core · "
                    f"{STRATUM_LABELS[stratum]} · {first}–{last}"
                ),
            )
            relative = (
                Path("contact-sheets")
                / f"{stratum}-{sheet_index + 1:02d}.png"
            )
            metadata = _save_png(sheet, output / relative)
            asset = {
                **metadata,
                "path": relative.as_posix(),
                "stratum": stratum,
                "sheet_index": sheet_index + 1,
                "source_core_ids": [
                    record["source_core_id"] for record, _panel in chunk
                ],
            }
            assets.append(asset)
            sheets.append(asset)

    renderer = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "status": REVIEW_STATUS,
        "source": {
            "path": _repository_relative(source_path),
            "sha256": _sha256_file(source_path),
            "source_membership_identity": core["source_membership_identity"],
        },
        "renderer": {
            "path": renderer.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(renderer),
            "pillow_version": PILLOW_VERSION,
        },
        "rendered_positions": {
            "count": len(ordered_fens),
            "ordered_fens_identity": ordered_fens_identity,
        },
        "summary": {
            "individual_image_count": len(individuals),
            "contact_sheet_count": len(sheets),
            "asset_count": len(assets),
            "stratum_counts": {"book": 22, "human_db": 21, "perfect_db": 21},
        },
        "individual_images": individuals,
        "contact_sheets": sheets,
        "assets": assets,
    }
    payload["manifest_identity"] = canonical_sha256(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_review_assets(
    source_path: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, int]:
    """Verify source binding, image identities, dimensions, and coverage."""
    manifest_path = output / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REVIEW_SCHEMA:
        raise SourceCoreReviewError("unexpected review schema")
    if payload.get("status") != REVIEW_STATUS:
        raise SourceCoreReviewError("unexpected review status")
    identity = payload.pop("manifest_identity", None)
    if identity != canonical_sha256(payload):
        raise SourceCoreReviewError("review manifest identity mismatch")
    payload["manifest_identity"] = identity

    source = json.loads(source_path.read_text(encoding="utf-8"))
    core = source["source_core"]
    if payload["source"] != {
        "path": _repository_relative(source_path),
        "sha256": _sha256_file(source_path),
        "source_membership_identity": core["source_membership_identity"],
    }:
        raise SourceCoreReviewError("review source binding drifted")
    renderer = Path(__file__).resolve()
    if payload["renderer"] != {
        "path": renderer.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(renderer),
        "pillow_version": PILLOW_VERSION,
    }:
        raise SourceCoreReviewError("review renderer identity drifted")

    records = list(core["records"])
    ordered_fens = [str(item["final"]["nmm_fen"]) for item in records]
    if payload["rendered_positions"] != {
        "count": 64,
        "ordered_fens_identity": canonical_sha256(ordered_fens),
    }:
        raise SourceCoreReviewError("rendered position identity drifted")
    if payload["summary"] != {
        "individual_image_count": 64,
        "contact_sheet_count": 6,
        "asset_count": 70,
        "stratum_counts": {"book": 22, "human_db": 21, "perfect_db": 21},
    }:
        raise SourceCoreReviewError("review asset summary drifted")

    expected_ids = [item["source_core_id"] for item in records]
    if [item["source_core_id"] for item in payload["individual_images"]] != (
        expected_ids
    ):
        raise SourceCoreReviewError("individual image order drifted")
    sheet_ids = [
        source_core_id
        for sheet in payload["contact_sheets"]
        for source_core_id in sheet["source_core_ids"]
    ]
    if sheet_ids != expected_ids:
        raise SourceCoreReviewError("contact-sheet coverage drifted")

    seen_paths: set[str] = set()
    for asset in payload["assets"]:
        relative = str(asset["path"])
        if relative in seen_paths:
            raise SourceCoreReviewError(f"duplicate asset path: {relative}")
        seen_paths.add(relative)
        path = output / relative
        if not path.is_file() or _sha256_file(path) != asset["sha256"]:
            raise SourceCoreReviewError(f"asset hash mismatch: {relative}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise SourceCoreReviewError(f"asset is not PNG: {relative}")
            if [image.width, image.height] != [
                int(asset["width"]),
                int(asset["height"]),
            ]:
                raise SourceCoreReviewError(
                    f"asset dimensions mismatch: {relative}"
                )
    actual_pngs = {
        path.relative_to(output).as_posix() for path in output.rglob("*.png")
    }
    if actual_pngs != seen_paths:
        raise SourceCoreReviewError("untracked or missing PNG assets")
    return {
        "individual_images": 64,
        "contact_sheets": 6,
        "assets": 70,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        summary = verify_review_assets(args.source, args.output)
    else:
        build_review_assets(args.source, args.output)
        summary = verify_review_assets(args.source, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
