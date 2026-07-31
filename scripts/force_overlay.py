"""Renderer-independent force-arrow timing and screen-space geometry."""
from __future__ import annotations
import numpy as np

FORCE_ARROW_LENGTH_PX = 120.0


def force_active_at_elapsed(elapsed_s: float, force_duration_s: float = 2.0) -> bool:
    return 0.0 <= elapsed_s < force_duration_s


def fixed_arrow_endpoint(anchor: tuple[int, int], projected_direction: tuple[int, int]) -> tuple[int, int]:
    vector = np.asarray(projected_direction, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0:
        raise ValueError("projected force direction is degenerate")
    endpoint = np.asarray(anchor, dtype=np.float64) + vector / norm * FORCE_ARROW_LENGTH_PX
    return tuple(endpoint.round().astype(int))
