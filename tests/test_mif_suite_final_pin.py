"""MIF Suite 1.0 final independent-adapter pin regressions."""

from __future__ import annotations

import tools.mif_interop_adapter_commands as commands
from learned_ai.interop.mif_v1.adapter import capabilities


SUITE_COMMIT = "3ee7e57c7d4c7208be91f62914f344a587fb0f70"
WIRE_COMMIT = "7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978"
SUITE_JCS_SHA256 = (
    "sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f"
)
SUITE_RAW_SHA256 = (
    "sha256:088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4"
)
RULESET_DIGESTS = [
    "sha256:173caf8189defd1ab7d4a3e8b9e26688a07fd77976bf09d56bff5fe0c273e1a1",
    "sha256:224f7e368e322a4cc8c1225a025fb548d5b41eb096d34b7ae0543182d1aa9393",
]
TESTED_CLASSES = [
    "identity",
    "key",
    "position",
    "replay",
    "ruleset",
    "transform",
]


def test_capability_publishes_exact_suite_tested_domain() -> None:
    value = capabilities()
    classes = {item["id"]: item["level"] for item in value["classes"]}

    assert value["suites"] == [SUITE_JCS_SHA256]
    assert classes == {
        "conversion": "none",
        **{identifier: "tested" for identifier in TESTED_CLASSES},
    }
    assert "full" not in classes
    assert value["conversions"] == []
    assert sorted(item["semanticDigest"] for item in value["rulesets"]) == (
        RULESET_DIGESTS
    )
    assert {item["level"] for item in value["rulesets"]} == {"tested"}


def test_capability_binds_suite_commit_raw_and_jcs_identities() -> None:
    value = capabilities()

    assert value["implementation"]["version"].endswith(SUITE_COMMIT[:12])
    assert value["annotations"]["wireCommit"] == WIRE_COMMIT
    assert value["annotations"]["suiteCandidateCommit"] == SUITE_COMMIT
    assert value["annotations"]["suiteJcsSha256"] == SUITE_JCS_SHA256
    assert value["annotations"]["suiteRawSha256"] == SUITE_RAW_SHA256
    assert value["testedCorpora"][-1]["classes"] == TESTED_CLASSES


def test_command_generator_locks_suite_candidate_and_release_inputs() -> None:
    assert commands.MIF_SUITE_COMMIT == SUITE_COMMIT
    assert commands.MIF_WIRE_COMMIT == WIRE_COMMIT
    assert commands.MIF_SUITE_JCS_SHA256 == SUITE_JCS_SHA256
    assert commands.MIF_PINNED_FILES["mif-suite-1.0.json"] == SUITE_RAW_SHA256[7:]
    assert commands.MIF_PINNED_FILES[
        "release/mif-1.0-release-manifest.json"
    ] == "b721cb2bd22e404ef2cac1ff570c7ea4d0b4859c97cbaba94a8acce241a00057"
    assert commands.MIF_PINNED_FILES[
        "interop/differential-candidate-4-v1.json"
    ] == "560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e"
    assert commands.MIF_PINNED_FILES["LICENSE"] == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )
