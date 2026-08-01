from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
)
REVIEW_DOC = SOURCE.with_name(
    "sanmill-layered-opening-prefix-v2-source-core-review-2026-08-01.md"
)
ASSETS = (
    ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01"
)
RENDERER = ROOT / "tools" / "render_layered_prefix_source_core_review.py"


def _module():
    spec = importlib.util.spec_from_file_location("source_core_renderer", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_manifest_identity_and_source_binding() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.pop("manifest_identity")

    assert identity == canonical_sha256(manifest)
    assert manifest["status"] == (
        "source_membership_review_only_not_execution_authority"
    )
    assert manifest["source"]["source_membership_identity"] == (
        source["source_core"]["source_membership_identity"]
    )


def test_review_package_covers_all_members_and_verifies() -> None:
    module = _module()
    assert module.verify_review_assets(SOURCE, ASSETS) == {
        "individual_images": 64,
        "contact_sheets": 6,
        "assets": 70,
    }
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"] == {
        "individual_image_count": 64,
        "contact_sheet_count": 6,
        "asset_count": 70,
        "stratum_counts": {"book": 22, "human_db": 21, "perfect_db": 21},
    }


def test_review_document_embeds_every_contact_sheet() -> None:
    document = REVIEW_DOC.read_text(encoding="utf-8")
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))

    for sheet in manifest["contact_sheets"]:
        target = (
            "assets/sanmill-layered-opening-prefix-v2-source-core-"
            f"2026-08-01/{sheet['path']}"
        )
        assert f"({target})" in document
        assert (REVIEW_DOC.parent / target).is_file()
