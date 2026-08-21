from __future__ import annotations

import ast
import importlib
import logging
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _startup_resources():
    return importlib.import_module("web.startup_resources")


def test_web_startup_source_never_builds_ngram_from_game_corpora() -> None:
    source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_from_games"
        for node in ast.walk(tree)
    )
    assert "_human_games_dir" not in source
    assert '"data" / "human_games"' not in source


def test_web_startup_source_uses_immutable_human_db_loader() -> None:
    source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    loader_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open_web_human_db"
    ]
    assert len(loader_calls) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HumanDB"
        for node in ast.walk(tree)
    )


def test_overseer_status_exposes_web_startup_resource_state() -> None:
    source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    endpoint = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "overseer_status"
    )
    endpoint_source = ast.unparse(endpoint)

    assert "ngram_opponent_model" in endpoint_source
    assert "human_db_runtime" in endpoint_source


def test_missing_ngram_cache_disables_without_constructing_model(
    tmp_path: Path,
    caplog,
) -> None:
    resources = _startup_resources()
    constructed = []

    def _must_not_construct():
        constructed.append(True)
        raise AssertionError("missing cache must not construct or train a model")

    caplog.set_level(logging.WARNING)
    model, status = resources.load_cached_ngram_model(
        tmp_path / "missing-ngram.json",
        model_factory=_must_not_construct,
    )

    assert model is None
    assert constructed == []
    assert status == {
        "available": False,
        "enabled": False,
        "mode": "cache-only",
        "cache_path": str(tmp_path / "missing-ngram.json"),
        "cache_valid": False,
        "game_count": 0,
        "disabled_reason": "cache file is missing",
        "raw_corpus_scan_attempted": False,
    }
    assert "cache file is missing" in caplog.text


def test_invalid_ngram_cache_disables_without_corpus_fallback(
    tmp_path: Path,
    caplog,
) -> None:
    resources = _startup_resources()
    cache = tmp_path / "invalid-ngram.json"
    cache.write_text("not a model", encoding="utf-8")

    class _InvalidModel:
        game_count = 0

        def load(self, path: Path) -> None:
            assert path == cache
            raise ValueError("invalid cache fixture")

        def load_from_games(self, _path: Path) -> None:
            raise AssertionError("raw corpus fallback must never run")

    caplog.set_level(logging.WARNING)
    model, status = resources.load_cached_ngram_model(
        cache,
        model_factory=_InvalidModel,
    )

    assert model is None
    assert status["enabled"] is False
    assert status["cache_valid"] is False
    assert status["raw_corpus_scan_attempted"] is False
    assert status["disabled_reason"].startswith("cache load failed:")
    assert "invalid cache fixture" in caplog.text


def test_valid_ngram_cache_is_loaded_read_only(tmp_path: Path) -> None:
    resources = _startup_resources()
    cache = tmp_path / "ngram.json"
    original = b"frozen-cache"
    cache.write_bytes(original)

    class _CachedModel:
        game_count = 17

        def __init__(self) -> None:
            self.loaded = []

        def load(self, path: Path) -> None:
            self.loaded.append(path)

        def load_from_games(self, _path: Path) -> None:
            raise AssertionError("cached startup must never inspect raw games")

    model, status = resources.load_cached_ngram_model(
        cache,
        model_factory=_CachedModel,
    )

    assert model is not None
    assert model.loaded == [cache]
    assert cache.read_bytes() == original
    assert status["available"] is True
    assert status["enabled"] is True
    assert status["cache_valid"] is True
    assert status["game_count"] == 17
    assert status["disabled_reason"] is None
    assert status["raw_corpus_scan_attempted"] is False


def test_web_human_db_is_opened_read_only_and_immutable(tmp_path: Path) -> None:
    resources = _startup_resources()
    path = tmp_path / "human.sqlite"
    path.write_bytes(b"fixture-not-opened-by-fake")
    calls = []

    class _FakeHumanDB:
        entry_count = 23
        game_count = 7

        def __init__(
            self,
            db_path: Path,
            *,
            read_only: bool,
            immutable: bool,
        ) -> None:
            calls.append((db_path, read_only, immutable))

        def is_available(self) -> bool:
            return True

    database, status = resources.open_web_human_db(
        path,
        db_factory=_FakeHumanDB,
    )

    assert database is not None
    assert calls == [(path, True, True)]
    assert status == {
        "available": True,
        "path": str(path),
        "read_only": True,
        "immutable": True,
        "entry_count": 23,
        "game_count": 7,
        "disabled_reason": None,
    }


def test_real_immutable_human_db_open_leaves_main_and_sidecars_unchanged(
    tmp_path: Path,
) -> None:
    resources = _startup_resources()
    path = tmp_path / "human.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE positions(state_key TEXT PRIMARY KEY);
        CREATE TABLE moves(state_key TEXT, notation TEXT);
        INSERT INTO meta(key, value) VALUES ('total_games', '0');
        INSERT INTO meta(key, value)
        VALUES ('malom_label_version', 'sector-corrected-v1');
        """
    )
    connection.commit()
    connection.close()
    before_bytes = path.read_bytes()
    before_mtime_ns = path.stat().st_mtime_ns
    wal = Path(f"{path}-wal")
    shm = Path(f"{path}-shm")
    assert not wal.exists()
    assert not shm.exists()

    database, status = resources.open_web_human_db(path)

    assert database is not None
    assert database.is_available()
    assert status["read_only"] is True
    assert status["immutable"] is True
    assert database.query_line is not None
    database.close()
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime_ns
    assert not wal.exists()
    assert not shm.exists()


def test_read_only_runtime_view_never_forwards_mutations(caplog) -> None:
    resources = _startup_resources()

    class _Database:
        def query(self) -> str:
            return "read-result"

        def add_game(self, _record: dict) -> None:
            raise AssertionError("immutable HumanDB must never receive add_game")

    caplog.set_level(logging.INFO)
    view = resources.ReadOnlyHumanDBRuntimeView(_Database())

    assert view.query() == "read-result"
    view.add_game({"winner": "W"})
    assert "immutable runtime snapshot was not mutated" in caplog.text
