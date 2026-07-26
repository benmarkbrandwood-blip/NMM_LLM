"""Render the expert Book parent/child semantic-review boards.

The source audit remains immutable.  This tool assigns presentation-only
P01..P14 aliases, renders all exact eight-ply parent variants, and renders one
comparison sheet for every exact parent that has multiple twelve-ply
children.
"""

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


REVIEW_ASSET_SCHEMA = "nmm.layered-expert-book-review-assets.v1"
REVIEW_STATUS = "semantic_review_only_not_corpus_freeze"
DEFAULT_AUDIT = Path(
    "docs/evidence/"
    "sanmill-layered-expert-book-source-audit-2026-07-26.json"
)
DEFAULT_OUTPUT = Path(
    "docs/experiments/assets/"
    "sanmill-layered-expert-book-parent-review-2026-07-26"
)


class ExpertBookReviewAssetError(RuntimeError):
    """Raised when review assets cannot be built or verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _turn_text(turn: Sequence[str]) -> str:
    return "".join(str(token) for token in turn)


def _history_text(record: Mapping[str, Any], start: int, stop: int) -> str:
    turns = record["resolved_logical_turns"][start:stop]
    return " ".join(_turn_text(turn) for turn in turns)


def _source_label(record: Mapping[str, Any]) -> str:
    row = int(record["source_row"])
    label = str(record.get("label", "")).strip()
    if row == 1 and label:
        return f"row-{row:03d}-{label.lower().replace(' ', '-')}"
    return f"row-{row:03d}"


def _sorted_parent_groups(
    audit: Mapping[str, Any],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in audit["records"]:
        groups[str(record["parent8"]["ring16_canonical_fen"])].append(record)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            min(int(record["source_row"]) for record in item[1]),
            item[0],
        ),
    )
    return [
        (
            f"P{index:02d}",
            sorted(
                records,
                key=lambda record: (
                    int(record["source_row"]),
                    str(record["variation_id"]),
                ),
            ),
        )
        for index, (_orbit, records) in enumerate(ordered, 1)
    ]


def build_review_model(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Build stable presentation aliases without changing audit identities."""
    parent_groups = _sorted_parent_groups(audit)
    parent_variants: list[dict[str, Any]] = []
    child_comparisons: list[dict[str, Any]] = []

    for group_id, records in parent_groups:
        exact_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            exact_groups[str(record["parent8"]["exact_history_sha256"])].append(
                record
            )
        ordered_exact = sorted(
            exact_groups.items(),
            key=lambda item: (
                min(int(record["source_row"]) for record in item[1]),
                item[0],
            ),
        )
        for variant_index, (history_sha256, variant_records) in enumerate(
            ordered_exact
        ):
            variant_records = sorted(
                variant_records,
                key=lambda record: (
                    int(record["source_row"]),
                    str(record["variation_id"]),
                ),
            )
            suffix = (
                ""
                if len(ordered_exact) == 1
                else f"-{chr(ord('A') + variant_index)}"
            )
            review_id = f"{group_id}{suffix}"
            representative = variant_records[0]
            source_rows = sorted(
                {int(record["source_row"]) for record in variant_records}
            )
            parent_variants.append(
                {
                    "review_id": review_id,
                    "group_id": group_id,
                    "exact_history_sha256": history_sha256,
                    "source_rows": source_rows,
                    "history": _history_text(representative, 0, 8),
                    "nmm_fen": str(representative["parent8"]["nmm_fen"]),
                    "ring16_canonical_fen": str(
                        representative["parent8"]["ring16_canonical_fen"]
                    ),
                    "record_count": len(variant_records),
                }
            )
            if len(variant_records) > 1:
                child_comparisons.append(
                    {
                        "review_id": review_id,
                        "group_id": group_id,
                        "exact_history_sha256": history_sha256,
                        "parent_history": _history_text(
                            representative, 0, 8
                        ),
                        "records": [
                            {
                                "source_row": int(record["source_row"]),
                                "source_label": _source_label(record),
                                "variation_id": str(record["variation_id"]),
                                "label": str(record.get("label", "")),
                                "continuation": _history_text(record, 8, 12),
                                "nmm_fen": str(
                                    record["prefix_record"]["final"]["nmm_fen"]
                                ),
                                "exact_history_sha256": str(
                                    record["exact_history_sha256"]
                                ),
                            }
                            for record in variant_records
                        ],
                    }
                )

    return {
        "parent_group_count": len(parent_groups),
        "parent_variants": parent_variants,
        "child_comparisons": child_comparisons,
    }


def _panel_entry(
    *,
    index: int,
    fen: str,
    title: str,
    details: Sequence[str],
    footer: str,
) -> dict[str, Any]:
    board = BoardState.from_fen_string(fen)
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
        "review_title_prefix": title,
        "review_detail_lines": list(details),
        "review_footer": footer,
    }


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
    columns: int,
) -> Image.Image:
    if not panels:
        raise ExpertBookReviewAssetError("cannot render an empty contact sheet")
    thumb_size = (360, 420)
    title_height = 52
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], title_height + rows * thumb_size[1]),
        "#ddd5c8",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), title, fill="#2b2722", font=_load_fonts()["sheet"])
    for index, panel in enumerate(panels):
        thumbnail = panel.resize(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0]
        y = title_height + (index // columns) * thumb_size[1]
        sheet.paste(thumbnail, (x, y))
    return sheet


def _relative_asset(
    metadata: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    absolute = Path(str(metadata["path"]))
    return {
        **metadata,
        "path": absolute.relative_to(output).as_posix(),
    }


def build_review_assets(
    audit_path: Path = DEFAULT_AUDIT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Render review panels and write a content-bound manifest."""
    if output.exists():
        raise FileExistsError(f"review asset directory exists: {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    model = build_review_model(audit)
    output.mkdir(parents=True)
    audit_identity = str(audit["audit_identity"])
    rendered_fens = [
        str(parent["nmm_fen"]) for parent in model["parent_variants"]
    ] + [
        str(record["nmm_fen"])
        for comparison in model["child_comparisons"]
        for record in comparison["records"]
    ]
    rendered_positions_sha256 = canonical_sha256(rendered_fens)
    assets: list[dict[str, Any]] = []

    parent_panels: list[Image.Image] = []
    parent_manifest: list[dict[str, Any]] = []
    parent_total = len(model["parent_variants"])
    for index, parent in enumerate(model["parent_variants"], 1):
        rows = ", ".join(str(row) for row in parent["source_rows"])
        entry = _panel_entry(
            index=index,
            fen=str(parent["nmm_fen"]),
            title=f"{parent['review_id']} parent",
            details=[
                f"source rows {rows} | first 8 logical plies",
                f"history: {parent['history']}",
            ],
            footer="Review 1–3: family, grouping, and coverage priority",
        )
        panel = render_position_image(
            entry,
            total=parent_total,
            start_positions_sha256=rendered_positions_sha256,
        )
        target = output / "parents" / f"{parent['review_id']}.png"
        asset = _relative_asset(_save_png(panel, target), output=output)
        assets.append(asset)
        parent_panels.append(panel)
        parent_manifest.append({**parent, "panel": asset})

    parent_overview_image = _contact_sheet(
        parent_panels,
        title="Eight-ply expert Book parents (14 groups / 15 exact variants)",
        columns=4,
    )
    parent_overview = _relative_asset(
        _save_png(parent_overview_image, output / "parent-overview.png"),
        output=output,
    )
    assets.append(parent_overview)

    comparison_manifest: list[dict[str, Any]] = []
    child_panel_count = 0
    for comparison in model["child_comparisons"]:
        child_panels: list[Image.Image] = []
        child_records: list[dict[str, Any]] = []
        records = comparison["records"]
        for index, record in enumerate(records, 1):
            title = (
                f"{comparison['review_id']} "
                f"{record['source_label'].replace('row-', 'row ')}"
            )
            entry = _panel_entry(
                index=index,
                fen=str(record["nmm_fen"]),
                title=title,
                details=[
                    "same exact 8-ply parent | final position after ply 12",
                    f"plies 9–12: {record['continuation']}",
                ],
                footer=(
                    "Review 4: primary, distinct additional, or same-plan"
                ),
            )
            panel = render_position_image(
                entry,
                total=len(records),
                start_positions_sha256=rendered_positions_sha256,
            )
            target = (
                output
                / "children"
                / str(comparison["review_id"])
                / f"{record['source_label']}.png"
            )
            asset = _relative_asset(_save_png(panel, target), output=output)
            assets.append(asset)
            child_panels.append(panel)
            child_records.append({**record, "panel": asset})
            child_panel_count += 1

        columns = min(4, len(child_panels))
        sheet_image = _contact_sheet(
            child_panels,
            title=(
                f"{comparison['review_id']} children — "
                "same exact eight-ply parent"
            ),
            columns=columns,
        )
        sheet = _relative_asset(
            _save_png(
                sheet_image,
                output
                / "child-overviews"
                / f"{comparison['review_id']}.png",
            ),
            output=output,
        )
        assets.append(sheet)
        comparison_manifest.append(
            {
                **comparison,
                "records": child_records,
                "sheet": sheet,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": REVIEW_ASSET_SCHEMA,
        "status": REVIEW_STATUS,
        "source": {
            "path": audit_path.as_posix(),
            "sha256": _sha256_file(audit_path),
            "audit_identity": audit_identity,
        },
        "renderer": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(Path(__file__)),
            "pillow_version": PILLOW_VERSION,
        },
        "rendered_positions": {
            "count": len(rendered_fens),
            "ordered_fens_sha256": rendered_positions_sha256,
        },
        "summary": {
            "parent_group_count": int(model["parent_group_count"]),
            "exact_parent_variant_count": len(parent_manifest),
            "multi_child_exact_parent_count": len(comparison_manifest),
            "child_panel_count": child_panel_count,
            "asset_count": len(assets),
        },
        "parent_overview": parent_overview,
        "parent_variants": parent_manifest,
        "child_comparisons": comparison_manifest,
        "assets": assets,
    }
    payload["manifest_identity"] = canonical_sha256(payload)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_review_assets(
    audit_path: Path = DEFAULT_AUDIT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, int]:
    """Verify the manifest, source binding, model counts, and every PNG."""
    manifest_path = output / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REVIEW_ASSET_SCHEMA:
        raise ExpertBookReviewAssetError("unexpected review asset schema")
    if payload.get("status") != REVIEW_STATUS:
        raise ExpertBookReviewAssetError("unexpected review asset status")
    identity = payload.pop("manifest_identity", None)
    if identity != canonical_sha256(payload):
        raise ExpertBookReviewAssetError("manifest identity mismatch")
    payload["manifest_identity"] = identity

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = payload["source"]
    if source["sha256"] != _sha256_file(audit_path):
        raise ExpertBookReviewAssetError("source audit file hash mismatch")
    if source["audit_identity"] != audit["audit_identity"]:
        raise ExpertBookReviewAssetError("source audit identity mismatch")
    renderer = payload.get("renderer")
    expected_renderer = {
        "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(Path(__file__)),
        "pillow_version": PILLOW_VERSION,
    }
    if renderer != expected_renderer:
        raise ExpertBookReviewAssetError("review renderer identity mismatch")
    model = build_review_model(audit)
    rendered_fens = [
        str(parent["nmm_fen"]) for parent in model["parent_variants"]
    ] + [
        str(record["nmm_fen"])
        for comparison in model["child_comparisons"]
        for record in comparison["records"]
    ]
    if payload.get("rendered_positions") != {
        "count": len(rendered_fens),
        "ordered_fens_sha256": canonical_sha256(rendered_fens),
    }:
        raise ExpertBookReviewAssetError(
            "rendered review-position identity mismatch"
        )
    expected_summary = {
        "parent_group_count": int(model["parent_group_count"]),
        "exact_parent_variant_count": len(model["parent_variants"]),
        "multi_child_exact_parent_count": len(model["child_comparisons"]),
        "child_panel_count": sum(
            len(item["records"]) for item in model["child_comparisons"]
        ),
        "asset_count": len(payload["assets"]),
    }
    if payload["summary"] != expected_summary:
        raise ExpertBookReviewAssetError("review asset summary mismatch")

    seen: set[str] = set()
    for asset in payload["assets"]:
        relative = str(asset["path"])
        if relative in seen:
            raise ExpertBookReviewAssetError(f"duplicate asset path: {relative}")
        seen.add(relative)
        path = output / relative
        if not path.is_file() or _sha256_file(path) != asset["sha256"]:
            raise ExpertBookReviewAssetError(f"asset hash mismatch: {relative}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ExpertBookReviewAssetError(
                    f"asset is not PNG: {relative}"
                )
            if [image.width, image.height] != [
                int(asset["width"]),
                int(asset["height"]),
            ]:
                raise ExpertBookReviewAssetError(
                    f"asset dimensions mismatch: {relative}"
                )
    return {
        "parent_groups": expected_summary["parent_group_count"],
        "parent_variants": expected_summary["exact_parent_variant_count"],
        "child_comparisons": expected_summary[
            "multi_child_exact_parent_count"
        ],
        "child_panels": expected_summary["child_panel_count"],
        "assets": expected_summary["asset_count"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing asset directory without rewriting it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        summary = verify_review_assets(args.audit, args.output)
    else:
        build_review_assets(args.audit, args.output)
        summary = verify_review_assets(args.audit, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
