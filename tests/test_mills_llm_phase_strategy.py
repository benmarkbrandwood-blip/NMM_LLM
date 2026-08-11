from __future__ import annotations

import pathlib
import runpy


def test_phase_strategy_is_read_as_utf8(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_read_text(
        path: pathlib.Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        calls.append({"path": path, "args": args, "kwargs": kwargs})
        return "## Phase A\nCaf\u00e9 strategy\n"

    module_path = pathlib.Path(__file__).parents[1] / "ai" / "mills_llm.py"
    monkeypatch.setattr(type(module_path), "read_text", fake_read_text)
    monkeypatch.setattr(type(module_path), "exists", lambda _path: True)
    namespace = runpy.run_path(
        str(module_path),
        run_name="__mills_llm_phase_strategy_encoding_test__",
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"] == {"encoding": "utf-8"}
    assert namespace["_PHASE_STRATEGY"]["A"].endswith("Caf\u00e9 strategy")
