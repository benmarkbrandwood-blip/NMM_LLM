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
    assert commands.MIF_SUITE_COMMIT == (
        "3ee7e57c7d4c7208be91f62914f344a587fb0f70"
    )
    assert commands.MIF_WIRE_COMMIT == (
        "7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978"
    )
    assert commands.MIF_PINNED_FILES == {
        "mif-1.0.md": (
            "330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995"
        ),
        "docs/zh-CN/mif-1.0.md": (
            "9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1"
        ),
        "artifacts/mif-1.0/index.json": (
            "5acbb714bed77e24eaac72fa5f24d2e54d1e17aaf568a8b60718c840281a6541"
        ),
        "artifacts/mif-1.0/corpus/executable/reference-cases.json": (
            "350b7ff02772e820a57431e11c4e2f15a874d0779fb6e7afb01e9b16f6992741"
        ),
        "interop/adapter-protocol-v1.md": (
            "253c1d201ea1db625e0c534da445ca4ecaa0b07597dfc7dbf59fbd6adf89874f"
        ),
        "interop/cases/smoke-v1.json": (
            "a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53"
        ),
        "interop/cases/deterministic-v1.json": (
            "d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82"
        ),
        "interop/differential-candidate-4-v1.json": (
            "560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e"
        ),
        "mif-suite-1.0.json": (
            "088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4"
        ),
        "release/mif-1.0-release-manifest.json": (
            "b721cb2bd22e404ef2cac1ff570c7ea4d0b4859c97cbaba94a8acce241a00057"
        ),
        "LICENSE": (
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
        ),
    }


def test_three_party_command_arrays_are_process_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, sanmill_root = _fake_repositories(tmp_path)
    monkeypatch.setattr(
        commands, "_git_head", lambda _: commands.MIF_SUITE_COMMIT
    )
    monkeypatch.setattr(commands, "_git_worktree_changes", lambda _: "")
    monkeypatch.setattr(commands, "_verify_mif_sources", lambda _: None)
    monkeypatch.setattr(commands, "_verify_mif_suite", lambda _: None)
    config = commands.command_config(
        mif_root=mif_root,
        sanmill_root=sanmill_root,
        sanmill_binary=None,
    )
    adapters = {item["name"]: item for item in config["adapters"]}
    assert list(adapters) == [
        "mif-reference",
        "nmm-llm-python",
        "sanmill-rust",
    ]
    assert adapters["mif-reference"]["command"] == [
        "{python}",
        "-B",
        "tools/mif_1_0_reference_adapter.py",
    ]
    assert Path(adapters["nmm-llm-python"]["command"][-1]).name == (
        "nmm_llm_mif_adapter.py"
    )
    assert adapters["sanmill-rust"]["command"][-2:] == ["mill", "mif-interop"]
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


def test_command_generator_rejects_wrong_suite_jcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, _ = _fake_repositories(tmp_path)
    (mif_root / "mif-suite-1.0.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(commands, "sha256_digest", lambda _: "sha256:" + "0" * 64)

    with pytest.raises(ValueError, match="MIF suite JCS hash mismatch"):
        commands._verify_mif_suite(mif_root)


def test_command_generator_rejects_mif_worktree_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mif_root, sanmill_root = _fake_repositories(tmp_path)
    monkeypatch.setattr(
        commands, "_git_head", lambda _: commands.MIF_SUITE_COMMIT
    )
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
