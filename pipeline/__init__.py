"""Pipeline package for modular depth processing.

Modules:
- config: constants and thresholds
- intrinsics: camera intrinsics estimation
- depth: depth image to 3D point cloud conversion
- planes: plane segmentation, area, angle, per-frame processing
- axis: robust rotation axis estimation
- rosio: rosbag reading utilities
"""

__all__ = [
    "config",
    "intrinsics",
    "depth",
    "planes",
    "axis",
    "rosio",
]


