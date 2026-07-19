"""learned_ai/sentinel/db_teacher.py — read-only external solved-DB teacher adapter.

This adapter targets the **Malom ultra-strong NMM database** (ggevay/malom,
GPL-3 by Gabor E. Gevay and Gabor Danner) stored at the configured
``external_db_path`` (default: ``/mnt/windows/NMM_DB/strong/``).  It is used
as a *training-time teacher only* to label observed game trajectories with
ground-truth WDL.

IMPORTANT
---------
This is NOT the project's internal ``ai/endgame_solved_db.py`` (the engine's own
retrograde endgame DB). The two must never be merged. This adapter is read-only
and is designed to mirror that module's query surface so training code can swap
between internal and external teachers.

GRACEFUL UNAVAILABILITY (hard requirement)
------------------------------------------
The external DB is not present in the repo and frequently absent at runtime.
Every public method is therefore non-fatal:
  * construction never raises, even on a bad/missing path;
  * when the DB is unavailable, ``is_available()`` returns False and all
    ``query_*`` methods return ``None`` (or a list of ``None`` for trajectories);
  * a single clear warning is logged the first time an unavailable DB is queried.

MALOM DATABASE FORMAT
---------------------
The Malom .sec2 files store solved WDL for each board sector (W, B, WF, BF)
where W/B = pieces on board, WF/BF = pieces still to place.  The hash function
is a two-part combinatorial index over canonical symmetry orbits of White pieces
and compressed Black piece positions.  Full details are in ai/malom_db.py.

The previous stub (targeting database.dat + preCalculatedVars.dat) has been
replaced by the working MalomDB adapter in ai/malom_db.py.
  3. Attempts to read a small header from ``preCalculatedVars.dat`` and records
     its byte length + first bytes as ``self.format_probe`` for later analysis.
  4. The MalomDB adapter (ai/malom_db.py) implements the full hash function and
     decodes the .sec2 entries directly.  is_available() returns True when .sec2
     files are found; query() returns WDL results for any position in the DB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from game.rules import get_all_legal_moves, terminal_wdl

logger = logging.getLogger(__name__)

try:
    from ai.malom_db import MalomDB as _MalomDB
except ImportError:
    _MalomDB = None  # type: ignore[assignment,misc]


class ExternalSolvedDB:
    """Read-only adapter for the Malom ultra-strong NMM database."""

    def __init__(self, db_path: str = "", enabled: bool = True) -> None:
        """Open the Malom DB at db_path. Never raises.

        Parameters
        ----------
        db_path : path to the directory containing std_*.sec2 files
                  (e.g. /mnt/windows/NMM_DB/strong).
                  Empty string => unavailable.
        enabled : when False the adapter is forced unavailable regardless of path
                  (used to honour ``external_db_enabled: false`` in config).
        """
        self.db_path: str = db_path or ""
        self._enabled = bool(enabled)
        self._warned = False
        self._malom: Optional[_MalomDB] = None  # type: ignore[type-arg]
        self.db_dir: Optional[Path] = None
        self.format_probe: Dict[str, Any] = {}

        try:
            self._probe()
        except Exception as exc:  # absolutely never fatal
            logger.warning("[ExternalSolvedDB] probe failed (non-fatal): %s", exc)

    # ── Probing ──────────────────────────────────────────────────────────────

    def _probe(self) -> None:
        if not self._enabled or not self.db_path:
            return

        p = Path(self.db_path)
        if p.is_file():
            self.db_dir = p.parent
        elif p.is_dir():
            self.db_dir = p
        else:
            return

        self.format_probe = {
            "db_dir": str(self.db_dir),
            "available": False,
        }
        if _MalomDB is None:
            logger.warning("[ExternalSolvedDB] ai.malom_db not importable; DB unavailable")
            return

        self._malom = _MalomDB(self.db_dir)
        self.format_probe["available"] = self._malom.is_available()
        if self._malom.is_available():
            logger.info("[ExternalSolvedDB] Malom DB ready at %s", self.db_dir)

    # ── Availability ───────────────────────────────────────────────────────────

    @property
    def _available(self) -> bool:
        return self._malom is not None and self._malom.is_available()

    def is_available(self) -> bool:
        """True when the Malom DB files are present and queryable."""
        return self._available

    def _warn_unavailable_once(self) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "[ExternalSolvedDB] unavailable (path=%r, enabled=%s) — all "
                "queries return None; training falls back to outcome-proxy "
                "supervision.",
                self.db_path, self._enabled,
            )

    # ── Lookup ──────────────────────────────────────────────────────────────────

    def _lookup(self, board) -> Optional[str]:
        """Return "W"|"L"|"D" for the current mover, or None."""
        if self._malom is None:
            return None
        try:
            rules_result = terminal_wdl(board)
            if rules_result is not None:
                return rules_result
            result = self._malom.query(board)
            return result["outcome"] if result else None
        except Exception as exc:
            logger.debug("[ExternalSolvedDB] lookup error: %s", exc)
            return None

    # ── Public query surface (mirrors ai/endgame_solved_db.py style) ────────────

    def query_state(self, board) -> Optional[str]:
        """Return "W" | "L" | "D" for the side to move, or None if unavailable."""
        if not self._available:
            self._warn_unavailable_once()
            return None
        return self._lookup(board)

    def query(self, board) -> Optional[str]:
        """Alias of ``query_state`` matching EndgameSolvedDB.query()."""
        return self.query_state(board)

    @staticmethod
    def _validated_atomic_move(board, move: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a normalized complete legal move, or ``None``.

        Malom stores only settled positions.  A move that forms a Mill is not
        complete until its mandatory capture has been selected, so validation
        must happen before either the current or successor state is queried.
        ``get_all_legal_moves`` is the project's authoritative atomic-action
        enumerator and rejects missing, spurious, or illegal captures.
        """
        if not isinstance(move, dict):
            return None
        if any(field not in move for field in ("from", "to", "capture")):
            return None

        atomic_move = {
            "from": move["from"],
            "to": move["to"],
            "capture": move["capture"],
        }
        try:
            if atomic_move not in get_all_legal_moves(board):
                return None
        except Exception:
            return None
        return atomic_move

    def query_move_quality(self, board, move: Dict[str, Any]) -> Optional[float]:
        """Quality delta of ``move`` from ``board``: + good, - bad, None unknown.

        ``move`` must be a complete legal atomic action with explicit ``from``,
        ``to``, and ``capture`` fields.  A Mill-forming move without its
        mandatory capture, or a move with a spurious/illegal capture, returns
        ``None`` without querying Malom.

        The score compares WDL before the action with WDL after the fully
        settled action, from the original mover's perspective.
        """
        if not self._available:
            self._warn_unavailable_once()
            return None

        atomic_move = self._validated_atomic_move(board, move)
        if atomic_move is None:
            logger.debug(
                "[ExternalSolvedDB] rejected incomplete or illegal atomic move: %r",
                move,
            )
            return None

        try:
            before = self._lookup(board)
            after_board = board.apply_move(atomic_move)
            after = self._lookup(after_board)
        except Exception:
            return None
        if before is None or after is None:
            return None
        # Mover's perspective: after applying the move it is the opponent's turn,
        # so an opponent "L" (they lose) is good for the mover.
        rank = {"W": 1.0, "D": 0.0, "L": -1.0}
        before_v = rank.get(before, 0.0)
        after_opp_v = rank.get(after, 0.0)
        after_mover_v = -after_opp_v
        return after_mover_v - before_v

    # Outcome from the mover's perspective after applying a move flips to the
    # opponent's perspective (it becomes their turn). Negate to score the move
    # for the player who made it.
    _NEGATE_WDL = {"W": "loss", "L": "win", "D": "draw"}
    _PROJECT_WDL = {"W": "win", "L": "loss", "D": "draw"}

    def _enumerate_legal_moves(self, board, player) -> List[Dict[str, Any]]:
        """All legal apply-move dicts {from,to,capture} for ``player`` on ``board``.

        Delegate to the rules engine's authoritative atomic-action enumerator.
        Asking for a player other than the side to move fails closed; applying
        such a move would otherwise move the wrong colour in ``BoardState``.
        """
        if player != board.turn:
            return []
        return [dict(move) for move in get_all_legal_moves(board)]

    def query_all_moves(self, board, player) -> List[Dict[str, Any]]:
        """Return legal moves with coarse labels and complete Malom values.

        Returns a list of dicts, one per legal move::

            {"move": {"from","to","capture"}, "wdl": "win"|"draw"|"loss"|"unknown",
             "dtm": int | None, "oracle_value": OracleMoveValue | None}

        WDL and ``oracle_value`` use ``player``'s perspective. ``dtm`` exposes
        the transformed candidate key2 only for decisive WDL values; draw
        ordering requires the complete two-key comparator.  Returns an empty
        list when the DB is unavailable (never raises).
        """
        if not self._available:
            self._warn_unavailable_once()
            return []
        out: List[Dict[str, Any]] = []
        try:
            legal = self._enumerate_legal_moves(board, player)
        except Exception as exc:
            logger.debug("[ExternalSolvedDB] move enumeration error: %s", exc)
            return []

        query_value = getattr(self._malom, "query_value", None)
        parent_value = None
        if callable(query_value):
            try:
                parent_value = query_value(board)
            except Exception:
                parent_value = None

        for mv in legal:
            wdl = "unknown"
            dtm: Optional[int] = None
            oracle_value = None
            child_outcome = None
            child_key2 = None
            terminal_child = False
            try:
                after = board.apply_move(mv)
                rules_result = terminal_wdl(after)
                if rules_result is not None:
                    terminal_child = True
                    child_outcome = rules_result
                    make_terminal_value = getattr(
                        self._malom,
                        "terminal_move_value",
                        None,
                    )
                    if parent_value is not None and callable(make_terminal_value):
                        oracle_value = make_terminal_value(
                            parent_value,
                            rules_result,
                        )
                else:
                    child_value = query_value(after) if callable(query_value) else None
                    if child_value is not None:
                        child_outcome = child_value.outcome
                        child_key2 = child_value.key2
                        make_move_value = getattr(self._malom, "move_value", None)
                        if parent_value is not None and callable(make_move_value):
                            oracle_value = make_move_value(
                                parent_value,
                                child_value,
                            )
                    elif self._malom is not None:
                        result = self._malom.query(after)
                        if result:
                            child_outcome = result.get("outcome")
                            child_key2 = result.get("dtw")
            except Exception:
                child_outcome = None
                oracle_value = None

            if oracle_value is not None:
                wdl = self._PROJECT_WDL.get(oracle_value.outcome, "unknown")
                if wdl in ("win", "loss"):
                    dtm = oracle_value.key2
            elif child_outcome is not None:
                wdl = self._NEGATE_WDL.get(child_outcome, "unknown")
                if terminal_child:
                    dtm = 0
                elif wdl in ("win", "loss") and isinstance(
                    child_key2,
                    (int, float),
                ):
                    dtm = int(child_key2)

            out.append(
                {
                    "move": mv,
                    "wdl": wdl,
                    "dtm": dtm,
                    "oracle_value": oracle_value,
                }
            )
        return out

    @staticmethod
    def best_move_result(
        results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the strongest result using full values when available."""
        known = [row for row in results if row.get("wdl") != "unknown"]
        if not known:
            return None

        valued = [row for row in known if row.get("oracle_value") is not None]
        if len(valued) == len(known):
            contexts = {
                (
                    value.sector,
                    value.sector_value,
                    value.perspective,
                )
                for value in (row["oracle_value"] for row in valued)
            }
            if len(contexts) == 1:
                return max(
                    valued,
                    key=lambda row: row["oracle_value"].ordering_key(),
                )
            logger.error(
                "[ExternalSolvedDB] cannot compare values from mixed contexts"
            )

        # Compatibility fallback for test doubles and older adapters that only
        # expose WDL.  Draws deliberately tie: bare key2 cannot order them.
        def coarse_key(row: Dict[str, Any]) -> tuple[int, int]:
            rank = {"win": 2, "draw": 1, "loss": 0}.get(row.get("wdl"), -1)
            raw_distance = row.get("dtm")
            distance = (
                abs(int(raw_distance))
                if isinstance(raw_distance, (int, float))
                else 0
            )
            if row.get("wdl") == "win":
                secondary = -distance
            elif row.get("wdl") == "loss":
                secondary = distance
            else:
                secondary = 0
            return rank, secondary

        return max(known, key=coarse_key)

    def query_trajectory(self, states: List[Any]) -> List[Optional[str]]:
        """Return a WDL (or None) for each state in a trajectory.

        Always returns a list of the same length as ``states`` (all None when
        unavailable) so callers can zip without length checks.
        """
        if not self._available:
            self._warn_unavailable_once()
            return [None] * len(states)
        out: List[Optional[str]] = []
        for s in states:
            try:
                out.append(self._lookup(s))
            except Exception:
                out.append(None)
        return out

    def close(self) -> None:
        """Release cached sector data."""
        if self._malom is not None:
            self._malom.close()

    def __repr__(self) -> str:
        return (
            f"ExternalSolvedDB(path={self.db_path!r}, enabled={self._enabled}, "
            f"available={self._available})"
        )


def open_external_db(config) -> ExternalSolvedDB:
    """Convenience constructor from a SentinelConfig.

    Honours both ``external_db_enabled`` and ``external_db_path``. Falls back to
    the ``NMM_EXTERNAL_DB`` environment variable when the config path is empty.
    """
    path = getattr(config, "external_db_path", "") or os.environ.get("NMM_EXTERNAL_DB", "")
    enabled = bool(getattr(config, "external_db_enabled", False))
    return ExternalSolvedDB(db_path=path, enabled=enabled)
