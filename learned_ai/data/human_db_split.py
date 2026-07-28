"""learned_ai/data/human_db_split.py — canonical position-level split (§V2, §H1, §H3).

Every trainer / eval script that samples from human_db.sqlite should go through
`in_val_bucket(state_key, val_fraction)` so:

  * a given state_key is always in the same slice (deterministic SHA-256
    hash), never in both train and val,
  * every consumer (ValueNet v2 training + eval, HumanPrefNet training + eval,
    Sentinel v2 dataset builder) agrees on which state_keys are val.

Splitting at the state_key level is stricter than splitting at pair-level or
per-example-level: it prevents any position-mediated leakage between train and
val.  It's slightly weaker than a game-level split (a state_key can be reached
from multiple games), but for the v2 pipeline we treat state_keys as the
canonical training-unit boundary.
"""
from __future__ import annotations

import hashlib
from typing import Iterable


# Bump this if the hash function or manifest semantics change.
MANIFEST_VERSION = "v1"

DEFAULT_VAL_FRACTION = 0.20


def state_key_bucket(state_key: str) -> int:
    """Return the 0..99 bucket for a state_key (§V2/§H1/§H3 canonical hash)."""
    h = hashlib.sha256(state_key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def in_val_bucket(state_key: str, val_fraction: float = DEFAULT_VAL_FRACTION) -> bool:
    """True iff this state_key is in the held-out validation slice.

    val_fraction is honoured as an integer cut on the 0..99 bucket space, so
    e.g. val_fraction=0.15 → buckets 0..14 are val.
    """
    upper = int(round(val_fraction * 100))
    return state_key_bucket(state_key) < upper


def partition(
    state_keys: Iterable[str],
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Partition an iterable of state_keys into (train, val) lists."""
    train: list[str] = []
    val:   list[str] = []
    for sk in state_keys:
        (val if in_val_bucket(sk, val_fraction) else train).append(sk)
    return train, val


__all__ = [
    "MANIFEST_VERSION",
    "DEFAULT_VAL_FRACTION",
    "state_key_bucket",
    "in_val_bucket",
    "partition",
]
