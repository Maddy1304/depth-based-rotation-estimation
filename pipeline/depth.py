from typing import Tuple
import numpy as np
import open3d as o3d

from .config import MAX_VALID_DEPTH_M, STAT_OUTLIER_NB_NEIGHBORS, STAT_OUTLIER_STD_RATIO


def depth_to_points_3d(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    #Project a depth image to XYZ (meters)

    #depth: metric depth image (m). fx, fy, cx, cy: pinhole intrinsics
    
    h, w = depth.shape
    u = np.arange(0, w)
    v = np.arange(0, h)
    uu, vv = np.meshgrid(u, v)

    mask = (depth > 0) & (depth < MAX_VALID_DEPTH_M)
    Z = depth[mask]
    X = (uu[mask] - cx) * Z / fx
    Y = (vv[mask] - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return pts


def build_point_cloud(points_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    #Create an Open3D PointCloud from XYZ numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    return pcd


def remove_statistical_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    #Simple statistical denoise. Helps plane fitting on raw depth clouds
    if len(pcd.points) == 0:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_OUTLIER_NB_NEIGHBORS, std_ratio=STAT_OUTLIER_STD_RATIO
    )
    return pcd


