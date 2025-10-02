from typing import Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d

from .config import (
    PLANE_DISTANCE_THRESHOLD_M,
    MIN_PLANE_POINTS,
    RANSAC_ITERATIONS,
)


def segment_largest_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold_m: float = PLANE_DISTANCE_THRESHOLD_M,
    ransac_n: int = 3,
    num_iterations: int = RANSAC_ITERATIONS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(pcd.points) < MIN_PLANE_POINTS:
        return None, None

    model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold_m,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    if len(inliers) < MIN_PLANE_POINTS:
        return None, None

    normal = np.array(model[:3])
    inlier_pts = np.asarray(pcd.select_by_index(inliers).points)
    return normal, inlier_pts


def compute_plane_area_3d(points3d: np.ndarray, normal: np.ndarray) -> float:
    if points3d is None or normal is None or len(points3d) < 3:
        return 0.0

    centroid = points3d.mean(axis=0)
    n = normal / np.linalg.norm(normal)

    arbitrary = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    coords2d = (points3d - centroid) @ np.vstack([u, v]).T

    if len(coords2d) > 1000:
        idx = np.random.choice(len(coords2d), 1000, replace=False)
        coords2d = coords2d[idx]

    try:
        hull = ConvexHull(coords2d)
        return float(hull.volume)
    except Exception:
        return 0.0


def angle_with_camera_normal(normal: np.ndarray) -> Tuple[float, np.ndarray]:
    camera_axis = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    if np.dot(n, camera_axis) < 0:
        n = -n
    cos_angle = float(np.clip(np.dot(n, camera_axis), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_angle)))
    return angle, n


def process_depth_frame(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_to_points_fn,
    build_pcd_fn,
    denoise_fn,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    pts = depth_to_points_fn(depth_m, fx, fy, cx, cy)
    if len(pts) < MIN_PLANE_POINTS:
        return None, None, None

    pcd = build_pcd_fn(pts)
    pcd = denoise_fn(pcd)

    normal, inliers = segment_largest_plane(pcd)
    if normal is None or inliers is None:
        return None, None, None

    area = compute_plane_area_3d(inliers, normal)
    angle, corrected_normal = angle_with_camera_normal(normal)
    return angle, area, corrected_normal


