"""learned_ai/data/elo_binning.py — Shared Elo binning helper.

Single source of truth for the audit script and the HumanDB v3 builder.
The reviewer flagged (rev-2, §6) that fixed band columns cannot be
reallocated once counts are aggregated: v3 therefore stores raw 50-Elo
bin totals and applies configurations (Option A etc.) at read time.

Both the audit and the builder must agree on the bin boundary formula
so a mover with Elo=1250 lives in the same bin in both.  This module
is that agreement.

Option A (this project's first frozen training configuration) is
included as a pure function.  Additional configurations can be added
here without touching either the audit or the builder.
"""
from __future__ import annotations

from typing import Optional

# Bin width in Elo points.  Any change here is a schema-version change
# for the candidate HumanDB and must bump `elo_bin_size` in `meta`.
ELO_BIN_SIZE = 50


def elo_bin(elo: Optional[int]) -> Optional[int]:
    """Map an Elo rating to its bin identifier (the lower edge of the
    50-Elo bucket).  Returns None if `elo` is None.

    Examples:
        elo_bin(1200) -> 1200
        elo_bin(1237) -> 1200
        elo_bin(1250) -> 1250
        elo_bin(None) -> None
    """
    if elo is None:
        return None
    return int(elo) // ELO_BIN_SIZE * ELO_BIN_SIZE


# ── Option A: PlayOK amateur-corpus strata (not universal strength) ─────────

# Cut-offs are inclusive on the upper edge of each band.  These match
# tools/audit_human_blunders.py::_elo_band exactly.  Bin-aligned so
# every 50-Elo bin belongs to exactly one band (no boundary bins
# straddle two bands).
OPTION_A_NAME = "option_a_1149_1249"
_OPTION_A_LOWER_MAX  = 1149
_OPTION_A_MIDDLE_MAX = 1249


def option_a_band(elo: Optional[int]) -> str:
    """Return 'lower' / 'middle' / 'upper' / 'unknown' under Option A.

    Strata within the PlayOK amateur corpus (see
    docs/human_blunder_audit_phase1.md); not universal strength labels.
    """
    if elo is None:
        return "unknown"
    if elo <= _OPTION_A_LOWER_MAX:
        return "lower"
    if elo <= _OPTION_A_MIDDLE_MAX:
        return "middle"
    return "upper"


def option_a_band_from_bin(bin_start: Optional[int]) -> str:
    """Given a 50-Elo bin start, return the Option A band membership.

    Uses the bin's *upper* edge (`bin_start + ELO_BIN_SIZE - 1`) as the
    representative — a bin fully contained in one band always maps to
    that band, so no boundary-straddling is possible with the current
    boundaries (1150 and 1250 are exact bin edges).
    """
    if bin_start is None:
        return "unknown"
    return option_a_band(bin_start + ELO_BIN_SIZE - 1)
