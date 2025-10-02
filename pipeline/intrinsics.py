from typing import Tuple
import numpy as np
from .config import ESTIMATED_FOV_DEG


def estimate_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """Estimate reasonable camera intrinsics for a depth sensor.

    Assumes a pinhole camera model with approximately ESTIMATED_FOV_DEG field of view.
    Returns (fx, fy, cx, cy).
    """
    fx = fy = width / (2 * np.tan(np.radians(ESTIMATED_FOV_DEG) / 2))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


