# In-Pipe SLAM Dataset

This repository provides simulation assets, configuration files, and evaluation tools for benchmarking SLAM methods in confined in-pipe environments.

The full real-world and simulated ROS 2 bag datasets are publicly available on Zenodo.

---

## 📦 Dataset Download

The dataset is hosted on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.18615583

Please download the ROS 2 bag sequences directly from Zenodo.

**Format:**
- ROS 2 bag (SQLite3 `.db3`)
- Compatible with ROS 2 Humble (rosbag2)

---

## 🧭 Recorded Sensor Topics

The real-world sequences include RGB-D, IMU, LiDAR, and robot proprioceptive data.

### RGB Camera
- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`

### Depth (Aligned to Color)
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/aligned_depth_to_color/camera_info`

### Raw Depth
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/depth/camera_info`

### Point Cloud
- `/camera/camera/depth/color/points`

### Infrared
- `/camera/camera/infra1/image_rect_raw`
- `/camera/camera/infra2/image_rect_raw`

### IMU
- `/camera/camera/imu`
- `/camera/camera/accel/sample`
- `/camera/camera/gyro/sample`

### LiDAR
- `/scan`

### Robot Data
- `/encoders`
- `/cmd_vel`

### TF & Navigation
- `/tf`
- `/tf_static`
- `/trajectory`
- `/initialpose`
- `/goal_pose`

---

## ▶️ Playing a Sequence

After downloading a sequence:

```bash
ros2 bag play rouen8_localisation_withemm/rouen8_localisation_withemm_0.db3 
```
## 📂 Repository Contents

This GitHub repository contains:

Gazebo simulation models

Robot URDF description

Launch files

SLAM evaluation scripts

Benchmark configuration files

Large ROS 2 bag files are hosted externally on Zenodo.




