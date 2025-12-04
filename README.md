# Depth-Based Rotation Estimation

A ROS2-based perception pipeline for estimating plane rotation angles and rotation axis from depth sensor data. This project processes depth images from ROS2 bags to extract geometric information about rotating objects using computer vision and 3D perception techniques.

## 🎯 Overview

This system analyzes depth images from a rotating cuboid to:
- Extract plane normals and compute rotation angles for each frame
- Calculate visible plane areas in metric units (m²)
- Estimate the global rotation axis using robust statistical methods
- Output results in CSV format and rotation axis vector

## ✨ Features

- **ROS2 Bag Processing**: Reads and processes depth images from ROS2 bag files
- **3D Point Cloud Generation**: Converts depth images to metric 3D point clouds using pinhole camera model
- **Robust Plane Segmentation**: RANSAC-based plane fitting with statistical outlier removal
- **Rotation Analysis**: Computes angles between plane normals and camera optical axis
- **Global Axis Estimation**: PCA-based rotation axis estimation with outlier filtering
- **Modular Architecture**: Clean, maintainable codebase with separated concerns

## 📋 Requirements

- Python 3.8+
- ROS2 (for bag file format compatibility)
- See `depth/requiremnets.txt` for Python dependencies

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/depth-based-rotation-estimation.git
cd depth-based-rotation-estimation
```

2. Install dependencies:
```bash
pip install -r depth/requiremnets.txt
```

## 📖 Usage

### Basic Usage

Run the main processing script:
```bash
python process.py
```

### Input Format

The script expects a ROS2 bag directory named `depth/` containing:
- `depth.db3`: SQLite3 database with depth image messages
- `metadata.yaml`: Bag metadata file
- Topic: `/depth` (sensor_msgs/msg/Image)
- Format: 16-bit depth images (480x640 resolution)

### Output

Results are saved in the `outputs/` directory:

1. **`results.csv`**: Per-frame metrics
   - `frame_id`: Frame index
   - `timestamp_ns`: ROS timestamp in nanoseconds
   - `angle_deg`: Rotation angle in degrees
   - `area_m2`: Visible plane area in square meters

2. **`axis.txt`**: Global rotation axis vector
   - Format: `[X Y Z]` unit vector in camera frame
   - Represents the axis around which the object rotates

## 📁 Project Structure

```
New_assesment/
├── process.py                 # Main orchestrator script
├── pipeline/                  # Modular perception pipeline
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants
│   ├── intrinsics.py         # Camera intrinsics estimation
│   ├── depth.py              # Depth to 3D point cloud conversion
│   ├── planes.py             # Plane segmentation and analysis
│   ├── axis.py               # Rotation axis estimation
│   └── rosio.py              # ROS2 bag I/O utilities
├── depth/                     # ROS2 bag data
│   ├── depth.db3             # Bag database file
│   ├── metadata.yaml         # Bag metadata
│   └── requiremnets.txt      # Python dependencies
└── outputs/                   # Results directory
    ├── results.csv            # Per-frame results
    └── axis.txt               # Rotation axis vector
```

## 🔬 Algorithm Details

### 1. Camera Intrinsics Estimation
- Assumes 60° field of view for depth sensor
- Calculates: `fx = fy = width/(2*tan(30°))`, `cx = width/2`, `cy = height/2`
- Used when real camera calibration is unavailable

### 2. Depth to 3D Point Cloud
- Pinhole camera projection: `X = (u-cx)*Z/fx`, `Y = (v-cy)*Z/fy`
- Filters depth range: 0-10 meters
- Converts depth units from mm to m if needed

### 3. Point Cloud Processing
- **Statistical Outlier Removal**: Removes noisy points (20 neighbors, 2.0 std ratio)
- Improves plane segmentation robustness

### 4. Plane Segmentation
- **RANSAC Algorithm**: Robust plane fitting
  - Distance threshold: 0.02m (2cm tolerance)
  - Iterations: 2000
  - Minimum inliers: 100 points
- Returns plane normal and inlier points

### 5. Area Computation
- Projects inlier points onto 2D plane coordinates
- Computes convex hull area in square meters
- Downsamples to 1000 points for efficiency

### 6. Angle Calculation
- Computes angle between plane normal and camera optical axis [0,0,1]
- Ensures normal points toward camera for consistency
- Returns angle in degrees

### 7. Rotation Axis Estimation
- Collects plane normals from all frames
- Applies median-based outlier filtering
- Uses PCA to find direction of minimum variance
- Returns unit vector with positive Z component

## 🔧 Configuration

Key parameters can be adjusted in `pipeline/config.py`:

```python
ESTIMATED_FOV_DEG = 60.0              # Camera field of view assumption
PLANE_DISTANCE_THRESHOLD_M = 0.02      # RANSAC distance threshold (meters)
MIN_PLANE_POINTS = 100                 # Minimum points for plane fitting
RANSAC_ITERATIONS = 2000                # RANSAC iterations
MAX_VALID_DEPTH_M = 10.0               # Maximum valid depth (meters)
STAT_OUTLIER_NB_NEIGHBORS = 20         # Outlier removal neighbors
STAT_OUTLIER_STD_RATIO = 2.0           # Outlier removal std ratio
```

## 📊 Results Interpretation

### Example Output

**results.csv:**
```
frame_id,timestamp_ns,angle_deg,area_m2
0,1702944981696402893,65.03,13.09
1,1702944983557733535,15.40,0.90
...
```

**axis.txt:**
```
# Rotation axis vector [X Y Z] in camera frame
-0.998938 0.029735 0.035201
```

### Understanding Results

- **Angle (degrees)**: 0° = plane parallel to image plane, 90° = plane perpendicular
- **Area (m²)**: Actual 3D area of visible plane surface
- **Rotation Axis**: Unit vector indicating rotation direction in camera frame

## 🛠️ Technical Stack

- **ROS2**: Bag file format and message handling
- **Open3D**: Point cloud processing and RANSAC plane segmentation
- **NumPy**: Numerical computations and array operations
- **SciPy**: Convex hull computation
- **Pandas**: Results table management
- **rosbags**: ROS2 bag reading library

## 🔍 ROS2 Integration

### Message Types
- **Topic**: `/depth`
- **Message Type**: `sensor_msgs/msg/Image`
- **Encoding**: 16-bit depth (uint16)
- **Serialization**: CDR (Common Data Representation)

### Bag Format
- **Storage**: SQLite3 database
- **Format**: ROS2 bag2
- **Version**: 8

## 🎓 Key Concepts

- **Pinhole Camera Model**: Standard computer vision projection
- **RANSAC**: Robust plane fitting with outlier rejection
- **PCA**: Principal component analysis for rotation axis
- **Convex Hull**: 2D area computation on plane projection
- **Statistical Filtering**: Median + 2σ outlier removal

## 🚧 Limitations & Future Improvements

### Current Limitations
- Camera intrinsics are estimated (not calibrated)
- Single plane detection per frame
- No temporal smoothing between frames
- Offline processing only (no real-time capability)

### Potential Improvements
- Real camera calibration integration
- Multi-plane detection
- Temporal consistency checks
- Real-time ROS2 node implementation
- GPU acceleration for point cloud processing
- Explicit occlusion handling

## 📝 License

[Specify your license here]

## 👤 Author

[Your Name]

## 🙏 Acknowledgments

- ROS2 community for bag format and tools
- Open3D team for excellent point cloud processing library
- 10x ConstructionAI for the perception challenge

## 📚 References

- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [Pinhole Camera Model](https://en.wikipedia.org/wiki/Pinhole_camera_model)

---

**Note**: This project was developed for a perception assignment focusing on depth-based rotation estimation from ROS2 sensor data.
