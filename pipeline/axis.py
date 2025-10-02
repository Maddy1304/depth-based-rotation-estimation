from typing import List
import numpy as np


def estimate_rotation_axis_robust(normals: List[np.ndarray]) -> np.ndarray:
    if len(normals) < 3:
        return np.array([0.0, 0.0, 1.0])

    N = np.asarray(normals)
    median = np.median(N, axis=0)
    distances = np.linalg.norm(N - median, axis=1)
    threshold = np.median(distances) + 2 * np.std(distances)
    filtered = N[distances < threshold]
    if len(filtered) < 3:
        filtered = N

    mean = filtered.mean(axis=0)
    centered = filtered - mean
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    axis = eigenvectors[:, np.argmin(eigenvalues)]
    axis = axis / np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
    return axis


