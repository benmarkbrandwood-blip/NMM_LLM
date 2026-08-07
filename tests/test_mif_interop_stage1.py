from __future__ import annotations

import ast
from pathlib import Path

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter, capabilities
from learned_ai.interop.mif_v1.common import (
    MIF_SUITE_JCS_SHA256,
    MifError,
    sha256_digest,
)
from learned_ai.interop.mif_v1.engine import execute, replay, resumption_state
from learned_ai.interop.mif_v1.model import (
    canonicalize_mfen,
    canonicalize_mpk,
    resolve_manifest,
)
from tests.mif_interop_fixtures import (
    DOCUMENT_DIGEST,
    EMPTY_ORIGIN,
    MIF_COMMIT,
    SEMANTIC_DIGEST,
    clone,
    empty_observation,
    example_manifest,
    mill_removal_events,
    offer_r1_mstate,
)


def test_pinned_manifest_identities_and_source_independence() -> None:
    manifest = resolve_manifest(example_manifest())
    assert manifest.semantic_digest == SEMANTIC_DIGEST
    assert manifest.document_digest == DOCUMENT_DIGEST

    source_root = Path("learned_ai/interop/mif_v1")
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert all("reference" not in name for name in imported)


def test_capabilities_pin_contract_and_tested_suite_domain() -> None:
    value = capabilities()
    assert value["suites"] == [MIF_SUITE_JCS_SHA256]
    assert value["annotations"]["contractCommit"] == MIF_COMMIT
    assert all(record["level"] == "tested" for record in value["rulesets"])
    assert [record["id"] for record in value["rulesets"]] == [
        "example-morris",
        "x-origin-stabilization",
    ]


def test_canonicalize_mfen_and_structural_mpk() -> None:
    manifest = example_manifest()
    assert canonicalize_mfen(EMPTY_ORIGIN, manifest) == EMPTY_ORIGIN
    mpk = (
        "MPK/1.0 mill24-state-v1 example-morris@1 "
        f"{SEMANTIC_DIGEST} structural-d4-v1 "
        "........................ b p 9,9"
    )
    assert canonicalize_mpk(mpk, manifest) == mpk


def test_execute_initial_boundary_matches_frozen_identity() -> None:
    execution = execute(example_manifest(), EMPTY_ORIGIN, [], [], [])
    assert execution.state.serialize() == EMPTY_ORIGIN
    assert execution.result["final"]["decisionDigest"] == (
        "sha256:3e178ce2bd4583f8a24b28b648b88f2ab1575d210e6fb295faca13e8cca47b46"
    )
    assert execution.repetition_history == [
        {"source": "origin", "key": empty_observation()}
    ]


def test_execute_placing_cycle_observes_all_stable_primary_boundaries() -> None:
    manifest = example_manifest()
    manifest["placing"]["movementAllowed"] = True
    events = [
        {"actor": "w", "from": "a7", "seq": 1, "to": "a4", "type": "move"},
        {"actor": "b", "from": "b6", "seq": 2, "to": "b4", "type": "move"},
        {"actor": "w", "from": "a4", "seq": 3, "to": "a7", "type": "move"},
        {"actor": "b", "from": "b4", "seq": 4, "to": "b6", "type": "move"},
    ]
    execution = execute(
        manifest,
        "MFEN/1.0 mill24-state-v1 W......./B......./........ w p p 8,8 - 0 0 -",
        events,
        [],
        [],
    )
    assert execution.state.serialize() == (
        "MFEN/1.0 mill24-state-v1 W......./B......./........ w p p 8,8 - 0 4 -"
    )
    assert len(execution.repetition_history) == 5
    assert len(execution.trace) == 5


def test_execute_mill_and_forced_remove_matches_frozen_reference() -> None:
    execution = execute(
        example_manifest(),
        EMPTY_ORIGIN,
        mill_removal_events(),
        [],
        [],
    )
    assert execution.state.serialize() == (
        "MFEN/1.0 mill24-state-v1 B.....BB/.W....../........ w p p 7,6 - 0 5 -"
    )
    assert execution.result["final"]["decisionDigest"] == (
        "sha256:d9b65701fc1490ce8d7d65d92c454db4480fe8bc1e66796bfda51fc5799793e6"
    )
    assert execution.trace[5]["decisionState"]["action"] == "r"
    assert execution.trace[6]["decisionState"]["action"] == "p"


def test_malformed_remove_target_returns_protocol_error_instead_of_crashing() -> None:
    events = mill_removal_events()
    events[-1]["target"] = 5
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "bad-target",
            "operation": "execute",
            "payload": {
                "manifest": example_manifest(),
                "origin": EMPTY_ORIGIN,
                "events": events,
                "repetitionSeed": [],
                "preOriginClaims": [],
            },
        }
    )
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"] == [
        {
            "category": "syntax",
            "code": "x-event-shape",
            "eventSeq": 6,
            "message": "invalid structured remove target",
        }
    ]


def test_flying_mill_remove_reaches_frozen_material_terminal() -> None:
    execution = execute(
        example_manifest(),
        (
            "MFEN/1.0 mill24-state-v1 BW...B.B/.W..W.../........ "
            "b m m 0,0 - 0 18 -"
        ),
        [
            {"actor": "b", "from": "d1", "seq": 1, "to": "a1", "type": "move"},
            {
                "actor": "b",
                "seq": 2,
                "target": {"at": "d7", "zone": "board"},
                "type": "remove",
            },
        ],
        [],
        [],
    )
    assert execution.state.serialize() == (
        "MFEN/1.0 mill24-state-v1 B.....BB/.W..W.../........ "
        "- o o 0,0 - 0 19 b:fewer-than-minimum"
    )
    assert execution.result["final"]["decisionDigest"] == (
        "sha256:a95ca567724ea188e7626d02f85bce2b42f2305226a7e2226a706dde891cf4a2"
    )


def test_replay_reference_and_portable_envelopes_match_frozen_digests() -> None:
    _, reference = replay(offer_r1_mstate(), example_manifest())
    _, portable = replay(offer_r1_mstate(portable=True), None)
    assert reference == portable
    assert reference["decisionDigest"] == (
        "sha256:f25cfb5dae617feba90fc1cbd48fb5d526727c8a3fad65910400a72a03657d19"
    )
    assert reference["resumptionDigest"] == (
        "sha256:1abb022db99a0959d00c90ca5ba6a946b99d183c8a811e01f242ef081bf5d5b3"
    )


def test_resumption_prefix_preserves_raw_pre_origin_claims() -> None:
    mstate = clone(offer_r1_mstate())
    mstate["events"] = [
        {
            "actor": "b",
            "offerEventSeq": 0,
            "seq": 1,
            "type": "decline-draw",
        }
    ]
    mstate["preOriginClaims"] = [
        {"actor": "w", "kind": "draw-offer", "status": "open"}
    ]
    mstate["claims"] = [
        {
            "source": "pre-origin",
            "actor": "w",
            "kind": "draw-offer",
            "status": "declined",
            "resolvedEventSeq": 1,
        }
    ]

    execution, _ = replay(mstate, example_manifest())
    value, digest = resumption_state(execution)
    expected_prefix = {
        "origin": EMPTY_ORIGIN,
        "preOriginRepetition": [],
        "preOriginClaims": [
            {"actor": "w", "kind": "draw-offer", "status": "open"}
        ],
        "events": mstate["events"],
    }
    assert value["replayPrefixDigest"] == sha256_digest(expected_prefix)
    assert digest == (
        "sha256:41680b36c378977be30b09c7e17c44086381464229a08e3f31f1d7ff904ff314"
    )
    assert execution.claims == mstate["claims"]


def test_replay_rejects_checkpoint_instead_of_repairing_it() -> None:
    mstate = clone(offer_r1_mstate())
    mstate["current"] = mstate["current"].replace(" 0 0 -", " 0 1 -")
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "bad-checkpoint",
            "operation": "replay",
            "payload": {"mstate": mstate, "manifest": example_manifest()},
        }
    )
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"][0]["code"] == "checkpoint-mismatch"


def test_reference_mstate_requires_caller_manifest() -> None:
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "missing-manifest",
            "operation": "replay",
            "payload": {"mstate": offer_r1_mstate()},
        }
    )
    assert response["status"] == "error"
    assert response["diagnostics"] == {
        "format": "MIFDIAG/1.0",
        "errors": [
            {
                "category": "integrity",
                "code": "manifest-missing",
                "message": "reference ruleset requires caller resolver",
            }
        ],
    }


def test_unsupported_variant_fails_closed() -> None:
    manifest = example_manifest()
    manifest["captures"]["leap"]["enabled"] = True
    with pytest.raises(MifError) as caught:
        resolve_manifest(manifest)
    assert getattr(caught.value, "code", None) == "unsupported-profile"
