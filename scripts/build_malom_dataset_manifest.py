"""Build a frozen component-hash manifest for one Malom tablebase directory."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.data.data_contract import (
    DatasetComponent,
    DatasetManifest,
    publish_dataset_manifest,
)
from learned_ai.training.run_contract import canonical_sha256


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source: Path) -> DatasetManifest:
    """Hash every component and return a path-portable Malom manifest."""
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or not (source / "std.secval").is_file():
        raise RuntimeError("source is not a populated Malom tablebase directory")
    components: list[DatasetComponent] = []
    total = len(files)
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        print(f"[{index}/{total}] hashing {relative}", flush=True)
        components.append(
            DatasetComponent(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=_hash_file(path),
            )
        )
    component_inventory = [item.to_dict() for item in components]
    return DatasetManifest(
        dataset_id="malom-standard-ultra-strong-sector-corrected-v1",
        logical_name="malom_tablebase",
        role="training_oracle",
        source="Malom Standard Ultra-strong 1.1.0 local validated import",
        schema_version="malom-ultra-strong-sec2",
        content_sha256=canonical_sha256(component_inventory),
        size_bytes=sum(item.size_bytes for item in components),
        created_at_utc=datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        creation_process="full SHA-256 component inventory audit",
        trust_level="sector-corrected-v1",
        allowed_consumers=("generalist_preflight", "malom_oracle"),
        validation=(
            "full component SHA-256 inventory",
            "sector-corrected decoder regression",
        ),
        exclusions=(
            "historical unversioned persisted labels",
            "machine-specific absolute source path",
        ),
        label_kinds=("theoretical_wdl",),
        components=tuple(components),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.source)
    publish_dataset_manifest(args.output, manifest)
    print(f"manifest_sha256={manifest.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
