"""tests/test_human_db_malom_resolution.py — Locks the priority order
of `_resolve_malom_path` in `tools/_human_db_build.py`.

Reviewer §12: the Malom path must be caller-supplied, never a hard-
coded absolute path in the plan doc.  Priority order (highest first):

    1. --no-malom → skip
    2. --malom-db CLI argument
    3. NMM_MALOM_DB environment variable
    4. malom_db_path in data/training_paths.local.json
    5. learned_ai.sentinel.config default (empty)
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools._human_db_build import _resolve_malom_path  # noqa: E402


class TestMalomPathResolution(unittest.TestCase):
    """Locks the priority order.  Uses monkeypatch to isolate from the
    real environment so the tests pass whether or not the developer's
    checkout carries training_paths.local.json / NMM_MALOM_DB."""

    def _clean_env(self) -> dict:
        """Return a clean env with NMM_MALOM_DB removed."""
        env = dict(os.environ)
        env.pop("NMM_MALOM_DB", None)
        return env

    def test_no_malom_flag_returns_empty(self):
        # Even with everything else set, --no-malom wins.
        with mock.patch.dict(os.environ, {"NMM_MALOM_DB": "/from/env"}):
            self.assertEqual(_resolve_malom_path("/from/cli", no_malom=True), "")

    def test_cli_path_wins_over_env(self):
        with mock.patch.dict(os.environ, {"NMM_MALOM_DB": "/from/env"}):
            self.assertEqual(
                _resolve_malom_path("/from/cli", no_malom=False),
                "/from/cli",
            )

    def test_env_used_when_no_cli(self):
        with mock.patch.dict(os.environ, self._clean_env() | {"NMM_MALOM_DB": "/from/env"}):
            self.assertEqual(_resolve_malom_path("", no_malom=False), "/from/env")

    def test_falls_back_to_empty_when_nothing_resolves(self):
        """No CLI, no env, and no readable local config → empty.  Uses a
        temporary rename of training_paths.local.json (if present) so we
        don't depend on the developer's local checkout."""
        local_cfg = _ROOT / "data" / "training_paths.local.json"
        temp_hide = _ROOT / "data" / "training_paths.local.json.hidden_for_test"
        hidden = False
        if local_cfg.exists():
            local_cfg.rename(temp_hide)
            hidden = True
        try:
            with mock.patch.dict(os.environ, self._clean_env(), clear=True):
                self.assertEqual(_resolve_malom_path("", no_malom=False), "")
        finally:
            if hidden:
                temp_hide.rename(local_cfg)

    def test_local_config_used_when_no_env_or_cli(self):
        """Write a synthetic training_paths.local.json (backing up any
        existing one), assert the resolver picks up its malom_db_path."""
        local_cfg = _ROOT / "data" / "training_paths.local.json"
        temp_hide = _ROOT / "data" / "training_paths.local.json.hidden_for_test"
        hidden = False
        if local_cfg.exists():
            local_cfg.rename(temp_hide)
            hidden = True
        try:
            local_cfg.write_text(
                json.dumps({"malom_db_path": "/from/local_config"}),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, self._clean_env(), clear=True):
                self.assertEqual(
                    _resolve_malom_path("", no_malom=False),
                    "/from/local_config",
                )
        finally:
            if local_cfg.exists():
                local_cfg.unlink()
            if hidden:
                temp_hide.rename(local_cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
