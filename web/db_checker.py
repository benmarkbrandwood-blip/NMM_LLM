"""web/db_checker.py — Startup database status checker.

Runs once at server startup in the background:
  1. Reads data/db_manifest.json (bundled source of truth).
  2. Checks sentinel files to determine which bundles are installed.
  3. Optionally fetches the remote manifest from Internet Archive (3 s timeout,
     silent on network failure) to detect available updates.
  4. Writes result to _db_status — readable via get_db_status().

Also manages data/db_manifest_local.json, written by the installer/DB manager
after each successful download so SHA comparisons survive across restarts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_MANIFEST_PATH       = _ROOT / "data" / "db_manifest.json"
_LOCAL_MANIFEST_PATH = _ROOT / "data" / "db_manifest_local.json"

log = logging.getLogger(__name__)

# Populated by run_startup_check(); read by get_db_status() / /api/db/status.
_db_status: dict[str, Any] = {}


# ── Manifest I/O ─────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read db_manifest.json: %s", exc)
        return {}


def load_local_manifest() -> dict:
    try:
        return json.loads(_LOCAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Could not read db_manifest_local.json: %s", exc)
        return {}


def save_local_manifest(data: dict) -> None:
    try:
        _LOCAL_MANIFEST_PATH.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Could not write db_manifest_local.json: %s", exc)


def mark_bundle_installed(bundle_id: str, sha256: str) -> None:
    """Record a successful install in the local manifest."""
    local = load_local_manifest()
    local[bundle_id] = {"sha256": sha256}
    save_local_manifest(local)


# ── Remote manifest fetch ─────────────────────────────────────────────────────

async def _fetch_remote_manifest(base_url: str, timeout: float = 3.0) -> dict | None:
    url = f"{base_url.rstrip('/')}/db_manifest.json"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        log.debug("Remote manifest fetch failed (%s): %s", url, exc)
        return None


# ── Per-bundle presence check ─────────────────────────────────────────────────

def _bundle_present(bundle: dict) -> bool:
    """True if all sentinel files for this bundle exist on disk."""
    sentinels = bundle.get("sentinel_files", [])
    if not sentinels:
        return False
    return all((_ROOT / s).exists() for s in sentinels)


# ── Main startup check ────────────────────────────────────────────────────────

async def run_startup_check() -> None:
    global _db_status

    manifest = load_manifest()
    local    = load_local_manifest()
    bundles  = manifest.get("bundles", {})
    base_url = manifest.get("ia_base_url", "")

    remote         = await _fetch_remote_manifest(base_url) if base_url else None
    remote_bundles = (remote or {}).get("bundles", {})
    remote_ok      = remote is not None

    bundle_status: dict[str, Any] = {}
    updates_available: list[str]  = []
    missing_required: list[str]   = []

    for bid, bdata in bundles.items():
        present    = _bundle_present(bdata)
        local_sha  = local.get(bid, {}).get("sha256", "")
        repo_sha   = bdata.get("sha256", "")
        remote_sha = remote_bundles.get(bid, {}).get("sha256", "") if remote_ok else ""

        # An update is available when the remote manifest advertises a different
        # SHA than our local copy — only meaningful if we know we're installed.
        update_available = bool(
            present and remote_ok and remote_sha and remote_sha != repo_sha
        )

        optional = bdata.get("optional", True)
        bundle_status[bid] = {
            "present":          present,
            "update_available": update_available,
            "state":            _bundle_state(present, update_available, optional),
            "repo_sha":         repo_sha,
            "local_sha":        local_sha,
            "remote_sha":       remote_sha,
            "optional":         optional,
            "label":            bdata.get("label", bid),
            "description":      bdata.get("description", ""),
            "size_display":     bdata.get("size_display", ""),
            "filename":         bdata.get("filename", ""),
            "version":          bdata.get("version", ""),
            "extract_to":       bdata.get("extract_to", ""),
            "extract_type":     bdata.get("extract_type", ""),
            "replaces":         bdata.get("replaces"),
        }

        if update_available:
            updates_available.append(bid)
        if not present and not bdata.get("optional", True):
            missing_required.append(bid)

    _db_status = {
        "bundles":           bundle_status,
        "updates_available": updates_available,
        "missing_required":  missing_required,
        "remote_reachable":  remote_ok,
        "ia_base_url":       base_url,
        "project_root":      str(_ROOT),
    }

    if missing_required:
        log.warning("Required DB bundles missing: %s", missing_required)
    if updates_available:
        log.info("DB updates available: %s", updates_available)
    else:
        log.info("DB status check complete — all bundles current.")


def _bundle_state(present: bool, update_available: bool, optional: bool) -> str:
    if present:
        return "update_available" if update_available else "current"
    return "missing" if optional else "missing_required"


def get_db_status() -> dict:
    return _db_status
