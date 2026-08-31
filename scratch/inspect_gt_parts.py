import json
import os
import numpy as np
from pxr import Usd, UsdGeom

# Load points3D_multipleview.ply or gt_segmentation.npz or open the USD file
usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
print(f"Opening USD: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# Let's collect part names in order of traversal, exactly as _sample_pointcloud in omni_capture.py does!
label_names = {}
lid = 0
for prim in stage.Traverse():
    if prim.GetTypeName() != "Mesh":
        continue
    parent = prim.GetParent()
    name = parent.GetName() if (parent and parent.IsValid() and parent.GetName()) else prim.GetName()
    if name not in label_names:
        label_names[name] = lid
        lid += 1

print(f"Total parts found in USD traversal: {len(label_names)}")
print("First 10 label mappings:", list(label_names.items())[:10])
print("Last 10 label mappings:", list(label_names.items())[-10:])

# Save mapping dictionary for verification
with open("scratch/part_label_mapping.json", "w") as f:
    json.dump(label_names, f, indent=2)
