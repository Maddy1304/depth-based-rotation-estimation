import numpy as np

# -------------------------------
# Configuration constants
# -------------------------------

# Estimated camera intrinsics defaults (60° FOV)
ESTIMATED_FOV_DEG: float = 60.0

# Plane segmentation params
PLANE_DISTANCE_THRESHOLD_M: float = 0.02  # meters
MIN_PLANE_POINTS: int = 100
RANSAC_ITERATIONS: int = 2000

# Depth filtering
MAX_VALID_DEPTH_M: float = 10.0

# Outlier removal
STAT_OUTLIER_NB_NEIGHBORS: int = 20
STAT_OUTLIER_STD_RATIO: float = 2.0

# Misc
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)


