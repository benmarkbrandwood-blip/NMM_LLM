"""Side-effect-bounded resource loading for the Web product startup path."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ai.human_db import HumanDB
from ai.ngram_opponent_model import NGramOpponentModel


log = logging.getLogger("nmm")


def load_cached_ngram_model(
    cache_path: Path | str,
    *,
    model_factory: Callable[[], Any] = NGramOpponentModel,
) -> tuple[Any | None, dict[str, Any]]:
    """Load only an existing N-gram cache; never build from game records."""
    path = Path(cache_path)
    status: dict[str, Any] = {
        "available": False,
        "enabled": False,
        "mode": "cache-only",
        "cache_path": str(path),
        "cache_valid": False,
        "game_count": 0,
        "disabled_reason": None,
        "raw_corpus_scan_attempted": False,
    }
    if not path.is_file():
        status["disabled_reason"] = "cache file is missing"
        log.warning(
            "NGramOpponentModel disabled: cache file is missing at %s; "
            "raw game scanning is disabled for Web startup.",
            path,
        )
        return None, status

    try:
        model = model_factory()
        model.load(path)
        game_count = int(model.game_count)
        if game_count <= 0:
            raise ValueError("cache contains no games")
    except Exception as exc:
        status["disabled_reason"] = f"cache load failed: {exc}"
        log.warning(
            "NGramOpponentModel disabled: cache load failed at %s: %s; "
            "raw game scanning is disabled for Web startup.",
            path,
            exc,
        )
        return None, status

    status.update(
        {
            "available": True,
            "enabled": True,
            "cache_valid": True,
            "game_count": game_count,
        }
    )
    log.info(
        "NGramOpponentModel: loaded read-only cache with %d games from %s",
        game_count,
        path,
    )
    return model, status


def open_web_human_db(
    db_path: Path | str,
    *,
    db_factory: Callable[..., Any] = HumanDB,
) -> tuple[Any | None, dict[str, Any]]:
    """Open the Web HumanDB query snapshot in immutable read-only mode."""
    path = Path(db_path)
    status: dict[str, Any] = {
        "available": False,
        "path": str(path),
        "read_only": True,
        "immutable": True,
        "entry_count": 0,
        "game_count": 0,
        "disabled_reason": None,
    }
    try:
        database = db_factory(path, read_only=True, immutable=True)
        if not database.is_available():
            status["disabled_reason"] = "database is unavailable"
            log.warning(
                "HumanDB unavailable at %s; Web access remains read-only and "
                "raw-corpus startup fallback is disabled.",
                path,
            )
            return None, status
    except Exception as exc:
        status["disabled_reason"] = f"immutable open failed: {exc}"
        log.warning("HumanDB immutable read-only open failed at %s: %s", path, exc)
        return None, status

    status.update(
        {
            "available": True,
            "entry_count": int(database.entry_count),
            "game_count": int(database.game_count),
        }
    )
    log.info(
        "HumanDB: immutable read-only snapshot loaded with %d positions and "
        "%d games from %s",
        status["entry_count"],
        status["game_count"],
        path,
    )
    return database, status


class ReadOnlyHumanDBRuntimeView:
    """Delegate HumanDB queries while making runtime update attempts explicit."""

    def __init__(self, database: Any) -> None:
        self._database = database

    def __getattr__(self, name: str) -> Any:
        return getattr(self._database, name)

    def add_game(self, _record: dict) -> None:
        log.info(
            "HumanDB immutable runtime snapshot was not mutated; the completed "
            "game remains in the normal game-record persistence path."
        )
