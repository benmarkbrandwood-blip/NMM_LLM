from __future__ import annotations

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter
from learned_ai.interop.mif_v1.engine import replay
from learned_ai.interop.mif_v1.transform import transform_payload
from learned_ai.interop.mif_v1.turns import project_logical_turns
from tests.mif_interop_fixtures import (
    example_manifest,
    mill_removal_mstate,
    offer_r1_mstate,
    origin_manifest,
    origin_mstate,
)


def test_transform_origin_history_replays_to_frozen_transformed_identity() -> None:
    result = transform_payload(
        {
            "kind": "mstate",
            "document": origin_mstate(),
            "manifest": origin_manifest(),
            "transform": "r90ccw",
            "verifyReplay": True,
            "requireEquivalence": False,
        }
    )
    assert result["document"]["current"] == (
        "MFEN/1.0 mill24-state-v1 WBWBWB../BWBWBWBW/WBWBWBWB w m m 0,0 - 0 24 -"
    )
    assert result["decisionDigest"] == (
        "sha256:d2c733493b3e7d863884ada1902e7e6b49c6728ba24250f3d9b0bd0f9133eb86"
    )
    assert result["resumptionDigest"] == (
        "sha256:9e254c5e286c4d5bc7fe7619797c6a108826ffa72751465cf907563a5bcb20ec"
    )


def test_decision_transform_requires_materialized_repetition_history() -> None:
    _, replay_result = replay(offer_r1_mstate(), example_manifest())
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "missing-history",
            "operation": "transform",
            "payload": {
                "kind": "decision-state",
                "document": replay_result["decisionState"],
                "manifest": example_manifest(),
                "transform": "r90ccw",
                "verifyReplay": False,
                "requireEquivalence": False,
            },
        }
    )
    assert response["diagnostics"] == {
        "format": "MIFDIAG/1.0",
        "errors": [
            {
                "category": "ineligible",
                "code": "insufficient-transform-history",
                "message": "decision repetition root requires its materialized active history",
            }
        ],
    }


def test_transform_equivalence_requires_exact_declaration() -> None:
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "missing-invariance",
            "operation": "transform",
            "payload": {
                "kind": "mstate",
                "document": origin_mstate(),
                "manifest": origin_manifest(),
                "transform": "r90ccw",
                "verifyReplay": True,
                "requireEquivalence": True,
            },
        }
    )
    error = response["diagnostics"]["errors"][0]
    assert (error["category"], error["code"]) == (
        "ineligible",
        "transform-invariance-undeclared",
    )


def test_project_draw_negotiation_produces_no_logical_turn() -> None:
    result = project_logical_turns(
        {"mstate": offer_r1_mstate(), "manifest": example_manifest()}
    )
    assert result["document"]["fragments"] == []
    assert result["document"]["sourceResumptionDigest"] == (
        "sha256:1abb022db99a0959d00c90ca5ba6a946b99d183c8a811e01f242ef081bf5d5b3"
    )


def test_project_origin_stabilization_groups_both_removals() -> None:
    result = project_logical_turns(
        {"mstate": origin_mstate(), "manifest": origin_manifest()}
    )
    assert result["document"]["fragments"] == [
        {
            "kind": "origin-stabilization",
            "removeEventSeqs": [1, 2],
            "status": "complete",
        }
    ]


def test_project_primary_and_compulsory_remove_as_one_logical_turn() -> None:
    result = project_logical_turns(
        {"mstate": mill_removal_mstate(), "manifest": example_manifest()}
    )
    assert result["document"] == {
        "format": "MIFTURN/1.0",
        "profile": "logical-turn-v1",
        "sourceResumptionDigest": (
            "sha256:ba3a054ea30ce02023b6f72aa49896fef70fabc667ef0f3cd79ff81d09cff164"
        ),
        "fragments": [
            {
                "kind": "logical-turn",
                "primaryEventSeq": seq,
                "removeEventSeqs": [6] if seq == 5 else [],
                "status": "complete",
            }
            for seq in range(1, 6)
        ],
    }
