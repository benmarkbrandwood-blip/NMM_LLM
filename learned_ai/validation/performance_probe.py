"""Fixed-work local model probe with an explicit loader optimization trigger."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from learned_ai.delivery.model_bundle import load_bundle_model
from learned_ai.training.run_contract import canonical_sha256


PROBE_SCHEMA = "nmm.local-performance-probe.v1"
LOADER_WAIT_TRIGGER = 0.10


def probe_model_bundle(
    bundle: str | Path,
    *,
    device: str = "cpu",
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, Any]:
    """Measure deterministic forward work without mutating training state."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("probe iteration counts are invalid")
    model, manifest = load_bundle_model(bundle, device=device)
    move_dim = int(manifest["architecture"]["parameters"]["move_feat_dim"])
    value_dim = int(manifest["architecture"]["parameters"]["value_input_dim"])
    generator = torch.Generator(device="cpu").manual_seed(20260720)
    move_input = torch.randn(24, move_dim, generator=generator).to(device)
    value_input = torch.randn(1, value_dim, generator=generator).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def synchronize() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(move_input, value_input)
        synchronize()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(move_input, value_input)
            synchronize()
            samples.append(time.perf_counter() - started)
    mean_seconds = statistics.fmean(samples)
    report = {
        "schema_version": PROBE_SCHEMA,
        "bundle_identity": manifest["bundle_identity"],
        "device": device,
        "fixed_work": {"legal_actions": 24, "policy_forwards": iterations, "value_forwards": iterations},
        "latency_ms": {"mean": mean_seconds * 1000.0, "median": statistics.median(samples) * 1000.0, "maximum": max(samples) * 1000.0},
        "forwards_per_second": 1.0 / mean_seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "data_path": {
            "kind": "online-serial-rollout",
            "persistent_loader": False,
            "loader_wait_ratio": None,
            "optimization_trigger": LOADER_WAIT_TRIGGER,
            "optimization_activated": False,
            "reason": "no persistent replay or offline loader exists in the measured path",
        },
    }
    report["probe_identity"] = canonical_sha256(report)
    return report
