import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pipeline.intrinsics import estimate_intrinsics
from pipeline.depth import depth_to_points_3d, build_point_cloud, remove_statistical_outliers
from pipeline.planes import segment_largest_plane, compute_plane_area_3d, angle_with_camera_normal, process_depth_frame
from pipeline.axis import estimate_rotation_axis_robust
from pipeline.rosio import open_depth_bag, iterate_depth_frames, close_reader

# -------------------------------
# Configuration
# -------------------------------
# Backward-compatible alias for threshold naming in the refactor

# -------------------------------
# Utility functions
# -------------------------------

# (moved to pipeline.intrinsics)

# (moved to pipeline.depth)

# (moved to pipeline.planes)

# (moved to pipeline.planes)

# (moved to pipeline.planes)

# (moved to pipeline.axis)

# moved to pipeline.planes.process_depth_frame

# -------------------------------
# Main processing
# -------------------------------

def main():
    print("=" * 60)
    print("Cuboid Rotation Angle Estimation")
    print("=" * 60)
    
    results = []
    normals = []
    
    print("\n📂 Reading ROS bag...")
    try:
        reader, connections = open_depth_bag('depth', '/depth')
    except Exception:
        print("❌ Failed to open bag directory 'depth' or list connections!")
        return
    if not connections:
        print("❌ No /depth topic found in bag file!")
        close_reader(reader)
        return
    print(f"✓ Found {len(connections)} connection(s) for /depth topic")

    idx = -1
    try:
        for idx, (h, w, timestamp, depth) in enumerate(iterate_depth_frames(reader, connections)):
            print(f"\n🔄 Processing Frame {idx}...")
            fx, fy, cx, cy = estimate_intrinsics(w, h)
            angle, area, normal = process_depth_frame(
                depth, fx, fy, cx, cy,
                depth_to_points_fn=depth_to_points_3d,
                build_pcd_fn=build_point_cloud,
                denoise_fn=remove_statistical_outliers,
            )
            if angle is not None:
                results.append({'frame_id': idx, 'timestamp_ns': timestamp, 'angle_deg': angle, 'area_m2': area})
                normals.append(normal)
                print(f"  ✓ Angle: {angle:.2f}°")
                print(f"  ✓ Area: {area:.4f} m²")
            else:
                print("  ⚠️  Failed to segment plane")
    finally:
        close_reader(reader)
    
    # Save results
    if not results:
        print("\n❌ No valid frames processed!")
        return
    
    print("\n" + "=" * 60)
    print(f"📊 Processed {len(results)} frames successfully")
    print("=" * 60)
    
    # Save results table
    df = pd.DataFrame(results)
    df.to_csv("outputs/results.csv", index=False)
    print("\n✅ Saved outputs/results.csv")
    print(f"\nResults Preview:")
    print(df.to_string(index=False))
    
    # Estimate and save rotation axis
    if normals:
        axis = estimate_rotation_axis_robust(normals)
        np.savetxt("outputs/axis.txt", axis.reshape(1, 3), fmt="%.6f", 
                   header="Rotation axis vector [X Y Z] in camera frame")
        print(f"\n✅ Saved outputs/axis.txt")
        print(f"   Rotation Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        print(f"   Magnitude: {np.linalg.norm(axis):.6f} (should be 1.0)")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()