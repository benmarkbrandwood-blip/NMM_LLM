#!/usr/bin/env python3
"""tools/update_manifest_sha.py — Print SHA256 and byte size for IA bundle files.

Usage:
    python tools/update_manifest_sha.py nmm_nets.tar.gz nmm_endgame_lite.tar.gz ...

Paste the output values into data/db_manifest.json after each IA upload.
"""
import hashlib
import sys
from pathlib import Path


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)

for arg in sys.argv[1:]:
    p = Path(arg)
    if not p.exists():
        print(f"  MISSING: {arg}")
        continue
    sha, size = sha256_and_size(p)
    print(f'  "{p.name}":')
    print(f'    sha256:       "{sha}"')
    print(f'    size_bytes:   {size}')
    print(f'    size_display: "{human_size(size)}"')
    print()
