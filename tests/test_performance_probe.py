from __future__ import annotations

import pytest

from learned_ai.validation.performance_probe import LOADER_WAIT_TRIGGER, probe_model_bundle


def test_probe_rejects_unbounded_or_empty_work() -> None:
    with pytest.raises(ValueError):
        probe_model_bundle("unused", iterations=0)


def test_loader_optimization_threshold_is_frozen() -> None:
    assert LOADER_WAIT_TRIGGER == 0.10
