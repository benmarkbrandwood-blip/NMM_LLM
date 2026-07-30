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

v2 additions (HumanMovePolicyNet Phase 4b / GapNet v3 Stage B):
  * `three_way_split(state_key)` → "test"|"val"|"train"
      test  = buckets  0..4   (5 %)
      val   = buckets  5..19  (15 %)
      train = buckets 20..99  (80 %)
    The "old val" (buckets 0..19) is unchanged — the new split merely
    carves a held-out *test* slice from the bottom of it.  Old consumers
    that call `in_val_bucket` are unaffected.
  * `game_level_split(session_id)` → "test"|"val"|"train"
    Applies the same 5/15/80 cut via a SHA-256 hash of the session_id
    string.  Used by `tools/build_session_index.py` to build
    `game_split_mask` diagnostic arrays.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Literal


# Bump this if the hash function or manifest semantics change.
MANIFEST_VERSION = "v2"

DEFAULT_VAL_FRACTION = 0.20

# v2 three-way split boundaries (bucket space 0..99)
_TEST_UPPER  = 5   # buckets 0..4  → test  (5 %)
_VAL_UPPER   = 20  # buckets 5..19 → val   (15 %)
#                    buckets 20..99 → train (80 %)

Split = Literal["train", "val", "test"]


def state_key_bucket(state_key: str) -> int:
    """Return the 0..99 bucket for a state_key (§V2/§H1/§H3 canonical hash)."""
    h = hashlib.sha256(state_key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def in_val_bucket(state_key: str, val_fraction: float = DEFAULT_VAL_FRACTION) -> bool:
    """True iff this state_key is in the held-out validation slice.

    val_fraction is honoured as an integer cut on the 0..99 bucket space, so
    e.g. val_fraction=0.15 → buckets 0..14 are val.

    Unchanged from v1 — all existing consumers (ValueNet v2, HumanPrefNet,
    Sentinel) continue to call this without modification.
    """
    upper = int(round(val_fraction * 100))
    return state_key_bucket(state_key) < upper


def three_way_split(state_key: str) -> Split:
    """Return "test", "val", or "train" for a state_key.

    Boundaries (bucket 0..99):
      test  =  0..4   (5 %)
      val   =  5..19  (15 %)
      train = 20..99  (80 %)
    """
    b = state_key_bucket(state_key)
    if b < _TEST_UPPER:
        return "test"
    if b < _VAL_UPPER:
        return "val"
    return "train"


def game_level_split(session_id: str) -> Split:
    """Return "test", "val", or "train" for a session_id string.

    Uses the same 5/15/80 bucket thresholds as `three_way_split`.
    Deterministic: identical session_ids always map to the same split.
    """
    h = hashlib.sha256(session_id.encode("utf-8")).digest()
    b = int.from_bytes(h[:4], "big") % 100
    if b < _TEST_UPPER:
        return "test"
    if b < _VAL_UPPER:
        return "val"
    return "train"


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
    "Split",
    "state_key_bucket",
    "in_val_bucket",
    "three_way_split",
    "game_level_split",
    "partition",
]
