from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tools.mif_interop_adapter_commands as commands


def _fake_repositories(tmp_path: Path) -> tuple[Path, Path]:
    mif_root = tmp_path / "mif"
    sanmill_root = tmp_path / "sanmill"
    (mif_root / "tools").mkdir(parents=True)
    (mif_root / "tools" / "mif_1_0_reference_adapter.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    sanmill_root.mkdir()
    (sanmill_root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    return mif_root, sanmill_root


def test_generator_pins_formal_and_execution_source_identities() -> None:
    assert commands.MIF_COMMIT == "f37ddfeb5fb8479991fa38eeb03c797bef8ae408"
    assert commands.MIF_PINNED_FILES == {
        "mif-1.0.md": (
            "330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995"
        ),
        "docs/zh-CN/mif-1.0.md": (
            "9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1"
        ),
        "artifacts/mif-1.0/index.json": (
            "3849a70897829d6d994c790b64e63484469483a940887fe828a1a0d421d78e90"
        ),
        "artifacts/mif-1.0/corpus/executable/reference-cases.json": (
            "a48c50352caebce30deb1de11f8f73dbc4540ee538651c3a139d9bcb166ba983"
        ),
        "interop/adapter-protocol-v1.md": (
            "a59e5e5af3e948f6c7cac6a39a490c6eae6338151741b6c7fcdde5c88d991e2d"
        ),
        "interop/cases/smoke-v1.json": (
            "a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53"
        ),
    }


def test_three_party_command_arrays_are_process_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, sanmill_root = _fake_repositories(tmp_path)
    monkeypatch.setattr(commands, "_git_head", lambda _: commands.MIF_COMMIT)
    monkeypatch.setattr(commands, "_git_worktree_changes", lambda _: "")
    monkeypatch.setattr(commands, "_verify_mif_sources", lambda _: None)
    config = commands.command_config(
        mif_root=mif_root,
        sanmill_root=sanmill_root,
        sanmill_binary=None,
    )
    adapters = {item["name"]: item for item in config["adapters"]}
    assert list(adapters) == ["mif-reference", "nmm-llm", "sanmill"]
    assert adapters["mif-reference"]["command"] == [
        "{python}",
        "-B",
        "tools/mif_1_0_reference_adapter.py",
    ]
    assert Path(adapters["nmm-llm"]["command"][-1]).name == (
        "nmm_llm_mif_adapter.py"
    )
    assert adapters["sanmill"]["command"][-2:] == ["mill", "mif-interop"]
    assert all(item["workingDirectory"] == "{repo}" for item in adapters.values())


def test_command_generator_rejects_a_floating_mif_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, sanmill_root = _fake_repositories(tmp_path)
    monkeypatch.setattr(commands, "_git_head", lambda _: "0" * 40)
    with pytest.raises(ValueError, match="MIF checkout must be exactly"):
        commands.command_config(
            mif_root=mif_root,
            sanmill_root=sanmill_root,
            sanmill_binary=None,
        )


def test_command_generator_rejects_modified_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, _ = _fake_repositories(tmp_path)
    source = mif_root / "contract.txt"
    source.write_bytes(b"locked\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        commands,
        "MIF_PINNED_FILES",
        {"contract.txt": expected},
    )

    commands._verify_mif_sources(mif_root)
    source.write_bytes(b"dirty\n")
    with pytest.raises(ValueError, match="MIF source hash mismatch"):
        commands._verify_mif_sources(mif_root)


def test_command_generator_rejects_mif_worktree_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, sanmill_root = _fake_repositories(tmp_path)
    monkeypatch.setattr(commands, "_git_head", lambda _: commands.MIF_COMMIT)
    monkeypatch.setattr(
        commands,
        "_git_worktree_changes",
        lambda _: "?? local-shadow-module.py",
    )
    monkeypatch.setattr(commands, "_verify_mif_sources", lambda _: None)
    with pytest.raises(ValueError, match="worktree changes"):
        commands.command_config(
            mif_root=mif_root,
            sanmill_root=sanmill_root,
            sanmill_binary=None,
        )
