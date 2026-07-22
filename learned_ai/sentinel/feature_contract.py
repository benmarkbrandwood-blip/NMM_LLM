"""Sentinel feature availability shared by training and evaluation.

The live Sentinel does not query Malom. Features populated only from the
database must therefore be zero during training and offline evaluation, even
when Malom is used as the label teacher.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from learned_ai.sentinel.feature_builder import FEATURE_DIM


# Counterfactual slots 41:46 and 48:58 come from the solved database. Slots
# 46 and 47 are heuristic rank/score fields and remain available at runtime.
DB_FEATURE_SLOTS = tuple(range(41, 46)) + tuple(range(48, 58))


def db_free_numpy_mask() -> np.ndarray:
    """Return the required float32 mask for NumPy Sentinel features."""
    mask = np.ones(FEATURE_DIM, dtype=np.float32)
    mask[list(DB_FEATURE_SLOTS)] = 0.0
    return mask


def db_free_torch_mask(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Return the required float32 mask for Torch Sentinel features."""
    mask = torch.ones(FEATURE_DIM, dtype=torch.float32, device=device)
    mask[list(DB_FEATURE_SLOTS)] = 0.0
    return mask
