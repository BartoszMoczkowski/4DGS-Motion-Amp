import cv2
import numpy as np
from pathlib import Path

scene = "lego_turbine_2_missing"
cam = "cam01"
frames_dir = Path(f"data/multipleview/{scene}/{cam}")
frame_files = sorted(list(frames_dir.glob("*.jpg")))
print(f"Total frames for {cam}: {len(frame_files)}")

# Read first 30 frames downsampled for fast ROI detection
sample_frames = []
for f in frame_files[:30]:
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    sample_frames.append(img)

stack = np.stack(sample_frames, axis=0).astype(np.float32)
diff = np.abs(stack[1:] - stack[:-1])
mean_diff = np.mean(diff, axis=0)

# Threshold to find motion region
thresh = mean_diff > (np.mean(mean_diff) + 3 * np.std(mean_diff))
y_indices, x_indices = np.where(thresh)
if len(y_indices) > 0:
    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()
    print(f"Motion bounds: X=[{xmin}, {xmax}], Y=[{ymin}, {ymax}], W={xmax-xmin}, H={ymax-ymin}")
    print(f"Center: ({ (xmin+xmax)//2 }, { (ymin+ymax)//2 })")
else:
    print("No motion region found with simple threshold!")
