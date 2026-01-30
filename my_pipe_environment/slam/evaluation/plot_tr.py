import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# File paths
# -----------------------
gt_file = "gt_output.csv"
slam_file = "slam_log.csv"
output_png = "xy_trajectories.png"

# -----------------------
# Load CSV files
# -----------------------
gt = pd.read_csv(gt_file)
slam = pd.read_csv(slam_file)

# Convert to NumPy explicitly (IMPORTANT)
gt_x = gt["x"].to_numpy()
gt_y = gt["y"].to_numpy()

slam_x = slam["x"].to_numpy()
slam_y = slam["y"].to_numpy()

# -----------------------
# Plot trajectories
# -----------------------
plt.figure(figsize=(8, 6))

plt.plot(gt_x, gt_y, label="Ground Truth", linewidth=2)
plt.plot(slam_x, slam_y, label="SLAM", linewidth=2)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("XY Trajectories")
plt.legend()
plt.axis("equal")
plt.grid(True)

# -----------------------
# Save plot
# -----------------------
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved trajectory plot to: {output_png}")

