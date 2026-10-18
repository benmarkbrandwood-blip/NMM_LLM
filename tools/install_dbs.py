#!/usr/bin/env python3
"""tools/install_dbs.py — Interactive database downloader.

Called by install.sh / install.ps1 after the venv is ready.
Usage:  python tools/install_dbs.py <project_root>

Downloads, verifies (SHA256), and extracts IA bundles, then optionally
configures the Malom perfect-database path in settings.json.
"""
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# ANSI colours — disabled on Windows unless ANSICON / Windows Terminal is active
_NO_COLOUR = sys.platform == "win32" and not os.environ.get("ANSICON") and not os.environ.get("WT_SESSION")
GREEN  = "" if _NO_COLOUR else "\033[0;32m"
YELLOW = "" if _NO_COLOUR else "\033[1;33m"
RED    = "" if _NO_COLOUR else "\033[0;31m"
NC     = "" if _NO_COLOUR else "\033[0m"


def info(msg):  print(f"{GREEN}[NMM]{NC} {msg}", flush=True)
def warn(msg):  print(f"{YELLOW}[NMM]{NC} {msg}", flush=True)
def err(msg):   print(f"{RED}[NMM]{NC} {msg}", file=sys.stderr, flush=True)


ROOT                = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
MANIFEST_PATH       = ROOT / "data" / "db_manifest.json"
LOCAL_MANIFEST_PATH = ROOT / "data" / "db_manifest_local.json"
SETTINGS_PATH       = ROOT / "data" / "settings.json"

# Bundles included in the recommended "Y" set
_RECOMMENDED = {"nmm_nets", "nmm_endgame_lite", "nmm_fullgame"}


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"Could not read db_manifest.json: {e}")
        return {}


def _load_local() -> dict:
    try:
        return json.loads(LOCAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_local(data: dict) -> None:
    LOCAL_MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bundle_present(b: dict) -> bool:
    return all((ROOT / s).exists() for s in b.get("sentinel_files", []))


# ── SHA256 ────────────────────────────────────────────────────────────────────

def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Download + extract ────────────────────────────────────────────────────────

def _download_bundle(b: dict, base_url: str) -> "str | None":
    """Download, verify, and extract one bundle.  Returns actual SHA256 or None on failure."""
    label    = b.get("label", b.get("id", "?"))
    filename = b.get("filename", "")
    url      = f"{base_url.rstrip('/')}/{filename}"
    expected = b.get("sha256", "")

    info(f"Downloading {label} ({b.get('size_display', '?')})…")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=f"_{filename}")
    tmp = Path(tmp_name)
    try:
        os.close(tmp_fd)

        total_bytes = b.get("size_bytes", 0)
        downloaded  = 0
        h           = hashlib.sha256()

        req = urllib.request.Request(url, headers={"User-Agent": "NMM-Installer/1.0"})
        try:
            with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
                content_length = int(resp.headers.get("Content-Length") or 0)
                if content_length:
                    total_bytes = content_length
                last_pct = -1
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = min(100, int(downloaded * 100 // total_bytes))
                        if pct != last_pct:
                            bar      = "█" * (pct // 4) + "░" * (25 - pct // 4)
                            mb_done  = downloaded / 1_048_576
                            mb_total = total_bytes / 1_048_576
                            print(
                                f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB ",
                                end="", flush=True,
                            )
                            last_pct = pct
                    else:
                        print(f"\r  {downloaded / 1_048_576:.1f} MB downloaded…", end="", flush=True)
            print()  # end progress line
        except urllib.error.URLError as e:
            print()
            err(f"Download failed for {label}: {e}")
            return None

        actual = h.hexdigest()
        if expected and actual != expected:
            err(
                f"Checksum mismatch for {label}.\n"
                f"  expected: {expected}\n"
                f"  got:      {actual}"
            )
            return None

        info("Extracting…")
        etype      = b.get("extract_type", "tar")
        extract_to = b.get("extract_to", ".")

        if etype == "tar":
            with tarfile.open(tmp) as tf:
                try:
                    tf.extractall(ROOT, filter="data")
                except TypeError:
                    tf.extractall(ROOT)
        elif etype == "gzip":
            dest = ROOT / extract_to
            dest.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(tmp, "rb") as gz, dest.open("wb") as out:
                shutil.copyfileobj(gz, out)
        else:
            err(f"Unknown extract_type '{etype}' for {label}.")
            return None

        info(f"{label} installed successfully.")
        return actual

    except Exception as e:
        err(f"Unexpected error installing {label}: {e}")
        return None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ── Malom prompt ──────────────────────────────────────────────────────────────

def _prompt_malom() -> None:
    print()
    print("  ── Malom Perfect Database (optional) ──────────────────────")
    print("  Provides perfect Win/Draw/Loss for every position.")
    print("  Download via BitTorrent (Standard Ultra-strong v1.1.0):")
    print("    https://www.inf.u-szeged.hu/~danner/mills/")
    print("  (12 GB torrent · 78 GB extracted · contains .sec2 sector files)")
    print()
    try:
        ans = input("  Do you have the Malom database installed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        info("Malom skipped — configure later at /databases in the game.")
        return

    try:
        raw = input("  Path to the folder containing .sec2 files: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not raw:
        return

    p = Path(raw).expanduser()
    if not p.exists():
        warn(f"Path does not exist: {raw}")
        return
    if not p.is_dir():
        warn(f"Not a directory: {raw}")
        return
    sec2 = list(p.glob("*.sec2"))
    if not sec2:
        warn(f"No .sec2 files found in {raw}")
        return

    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) if SETTINGS_PATH.exists() else {}
        settings["malom_db_path"] = str(p)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        info(f"Malom configured: {len(sec2)} sector files found.")
    except Exception as e:
        warn(f"Could not save Malom path: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    manifest = _load_manifest()
    if not manifest:
        return

    bundles  = manifest.get("bundles", {})
    base_url = manifest.get("ia_base_url", "")
    if not bundles or not base_url:
        warn("Manifest is empty or missing ia_base_url — skipping.")
        return

    # Annotate each bundle with its install state
    items = [{**b, "_id": bid, "_present": _bundle_present(b)} for bid, b in bundles.items()]

    # Show table
    print()
    print("  Available databases:")
    print()
    width = max(len(b.get("label", b["_id"])) for b in items) + 2
    for i, b in enumerate(items):
        label = b.get("label", b["_id"])
        size  = b.get("size_display", "?")
        bid   = b["_id"]
        if b["_present"]:
            tag = "✓ installed"
        elif not b.get("optional", True):
            tag = "* required"
        elif bid in _RECOMMENDED:
            tag = "↓ recommended"
        else:
            tag = "  optional"
        print(f"  [{i+1}] {label:<{width}} {size:>10}   {tag}")

    total_rec_mb = sum(
        b.get("size_bytes", 0) for b in items
        if b["_id"] in _RECOMMENDED and not b["_present"]
    ) / 1_048_576
    print()
    if total_rec_mb > 0:
        print(f"  Y = recommended set ({total_rec_mb:.0f} MB: AI models + endgame tables + full-game DB)")
    else:
        print("  Y = recommended set (all already installed)")
    print("  n = skip all   |   s = choose individually")
    print()

    try:
        choice = input("  Download databases? [Y/n/s] ").strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        print()
        info("Skipping database download.")
        _prompt_malom()
        return

    selected: list[dict] = []

    if choice in ("y", "yes", ""):
        required_ids = {b["_id"] for b in items if not b.get("optional", True)}
        dl_ids       = _RECOMMENDED | required_ids
        selected     = [b for b in items if not b["_present"] and b["_id"] in dl_ids]
        if not selected:
            info("Recommended set already installed — nothing to download.")
    elif choice in ("s", "select"):
        print()
        for b in items:
            label = b.get("label", b["_id"])
            size  = b.get("size_display", "?")
            try:
                if b["_present"]:
                    ans = input(f"  {label} (already installed) — re-download? [y/N] ").strip().lower()
                else:
                    ans = input(f"  {label} ({size}) — download? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if ans in ("y", "yes"):
                selected.append(b)
    else:
        info("Skipping database download. Download later at /databases in the game.")
        _prompt_malom()
        return

    if not selected:
        info("Nothing to download.")
        _prompt_malom()
        return

    local  = _load_local()
    failed = []
    for b in selected:
        sha = _download_bundle(b, base_url)
        if sha:
            local[b["_id"]] = {"sha256": sha}
        else:
            failed.append(b.get("label", b["_id"]))

    _save_local(local)

    if failed:
        warn(f"Some downloads failed: {', '.join(failed)}")
        warn("Retry later at /databases in the game.")
    else:
        info("All downloads complete.")

    _prompt_malom()


if __name__ == "__main__":
    main()
