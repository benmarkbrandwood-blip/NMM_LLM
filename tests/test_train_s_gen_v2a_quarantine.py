from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import train_s_gen_v2a


def test_main_lineage_v2a_is_quarantined_before_runtime_initialisation() -> None:
    with pytest.raises(RuntimeError, match="quarantined"):
        train_s_gen_v2a.run(SimpleNamespace())
