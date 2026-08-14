#!/usr/bin/env python3
"""Create or audit immutable inputs for the retained held-out score plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.delivery.training_route_bundle import (  # noqa: E402
    verify_training_route_bundle,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from tools.prepare_retained_phase_process_inputs import (  # noqa: E402
    build_manifest as build_source_manifest,
)


SOURCE_SNAPSHOT_IDENTITY = (
    "b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac"
)
SOURCE_ROOT = _ROOT / (
    "learned_ai/checkpoints/evaluation/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1/inputs"
)
TARGET_ROOT = _ROOT / (
    "learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-heldout-score-v1/inputs"
)
MANIFEST_SCHEMA = "nmm.retained-heldout-score-input-snapshots.v1"
MANIFEST_NAME = "manifest.json"
CANDIDATES = {
    "retained-v3-refresh50": {
        "source_bundle": SOURCE_ROOT / "v3-route-bundle",
        "target_bundle": "v3-route-bundle",
        "bundle_identity": (
            "b6d7ecf62ea9aeba893eff51e794d9307c444f361f54c9e1e832ac5b5d7bc5a0"
        ),
        "source_specialist_db": SOURCE_ROOT / "v3-specialist-db-snapshot.sqlite",
        "target_specialist_db": "v3-specialist-db-snapshot.sqlite",
        "specialist_db_sha256": (
            "82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe"
        ),
    },
    "retained-v4-no-refresh": {
        "source_bundle": SOURCE_ROOT / "v4-route-bundle",
        "target_bundle": "v4-route-bundle",
        "bundle_identity": (
            "817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f"
        ),
        "source_specialist_db": SOURCE_ROOT / "v4-specialist-db-snapshot.sqlite",
        "target_specialist_db": "v4-specialist-db-snapshot.sqlite",
        "specialist_db_sha256": (
            "3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed"
        ),
    },
}


class HeldoutScoreInputError(RuntimeError):
    """Raised when an input snapshot differs from the frozen source."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_ROOT).as_posix()
    except ValueError as exc:
        raise HeldoutScoreInputError("snapshot path leaves the repository") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, base: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HeldoutScoreInputError("snapshot input must be a regular file")
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _bundle_files(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_dir():
        raise HeldoutScoreInputError("route bundle directory is absent")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise HeldoutScoreInputError("route bundle contains a symbolic link")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return [_file_record(item, base=path) for item in files]


def _assert_no_sqlite_sidecars(path: Path) -> None:
    sidecars = [
        Path(str(path) + suffix)
        for suffix in ("-journal", "-shm", "-wal")
        if Path(str(path) + suffix).exists()
    ]
    if sidecars:
        raise HeldoutScoreInputError(
            "SpecialistDB sidecars exist: " + ", ".join(item.name for item in sidecars)
        )


def _candidate_record(
    candidate_id: str,
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    bundle = root / str(config["target_bundle"])
    specialist_db = root / str(config["target_specialist_db"])
    files = _bundle_files(bundle)
    verified = verify_training_route_bundle(bundle, device="cpu")
    if verified.get("bundle_identity") != config["bundle_identity"]:
        raise HeldoutScoreInputError(f"{candidate_id} route identity differs")
    _assert_no_sqlite_sidecars(specialist_db)
    specialist_hash = _sha256_file(specialist_db)
    if specialist_hash != config["specialist_db_sha256"]:
        raise HeldoutScoreInputError(f"{candidate_id} SpecialistDB differs")
    return {
        "candidate_id": candidate_id,
        "route_bundle": {
            "path": _relative(TARGET_ROOT / str(config["target_bundle"])),
            "identity": config["bundle_identity"],
            "files": files,
            "files_identity": canonical_sha256(files),
            "read_only_files": all(
                not bool((bundle / str(item["path"])).stat().st_mode & stat.S_IWRITE)
                for item in files
            ),
        },
        "specialist_db": {
            "path": _relative(TARGET_ROOT / str(config["target_specialist_db"])),
            "bytes": specialist_db.stat().st_size,
            "sha256": specialist_hash,
            "sidecars_absent": True,
            "read_only_file": not bool(specialist_db.stat().st_mode & stat.S_IWRITE),
        },
    }


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    """Audit one complete successor-owned input directory."""
    root = TARGET_ROOT if root is None else root
    if root.resolve() != TARGET_ROOT.resolve():
        raise HeldoutScoreInputError("snapshot root differs")
    if not root.is_dir():
        raise HeldoutScoreInputError("successor snapshot root is absent")
    source = build_source_manifest()
    if source.get("snapshot_identity") != SOURCE_SNAPSHOT_IDENTITY:
        raise HeldoutScoreInputError("source snapshot identity differs")
    candidates = [
        _candidate_record(candidate_id, config, root=root)
        for candidate_id, config in CANDIDATES.items()
    ]
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "source_snapshot_identity": SOURCE_SNAPSHOT_IDENTITY,
        "target_root": _relative(root),
        "candidates": candidates,
        "copy_semantics": {
            "successor_owned": True,
            "byte_exact": True,
            "source_paths_reused_at_runtime": False,
            "sqlite_sidecars_absent": True,
            "files_marked_read_only": True,
        },
    }
    result = {**body, "snapshot_identity": canonical_sha256(body)}
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        try:
            persisted = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HeldoutScoreInputError("snapshot manifest is invalid") from exc
        if canonical_json_bytes(persisted) != raw or persisted != result:
            raise HeldoutScoreInputError("snapshot manifest differs")
    return result


def _mark_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)


def prepare_inputs() -> dict[str, Any]:
    """Copy exact inputs once and atomically publish the verified directory."""
    if TARGET_ROOT.exists():
        raise FileExistsError(f"successor snapshot root already exists: {TARGET_ROOT}")
    source = build_source_manifest()
    if source.get("snapshot_identity") != SOURCE_SNAPSHOT_IDENTITY:
        raise HeldoutScoreInputError("source snapshot identity differs")
    TARGET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".heldout-score-inputs-", dir=TARGET_ROOT.parent)
    )
    try:
        for config in CANDIDATES.values():
            source_bundle = Path(config["source_bundle"])
            source_db = Path(config["source_specialist_db"])
            _assert_no_sqlite_sidecars(source_db)
            if _sha256_file(source_db) != config["specialist_db_sha256"]:
                raise HeldoutScoreInputError("source SpecialistDB differs")
            shutil.copytree(
                source_bundle,
                temporary / str(config["target_bundle"]),
                copy_function=shutil.copy2,
            )
            shutil.copy2(source_db, temporary / str(config["target_specialist_db"]))
        _mark_read_only(temporary)
        os.replace(temporary, TARGET_ROOT)
        manifest = build_manifest()
        manifest_path = TARGET_ROOT / MANIFEST_NAME
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        manifest_path.chmod(stat.S_IREAD)
        return build_manifest()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit"))
    args = parser.parse_args()
    result = prepare_inputs() if args.command == "prepare" else build_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
