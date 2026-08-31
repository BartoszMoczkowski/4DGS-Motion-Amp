from pxr import Usd, UsdGeom
import numpy as np
import os, sys

usd_path = 'omniverse-pipeline/data/scenes/grid/pump_A40mm_M8_scene.usd'
stage = Usd.Stage.Open(usd_path)
print('Opened stage:', usd_path, flush=True)

cache_def = UsdGeom.XformCache(Usd.TimeCode.Default())
cache_0 = UsdGeom.XformCache(Usd.TimeCode(0))

diffs = []
centroids_def = []
centroids_0 = []
part_names = []

for p in stage.Traverse():
    if p.GetTypeName() == 'Mesh':
        parent = p.GetParent()
        pname = parent.GetName() if (parent and parent.IsValid()) else p.GetName()
        pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
        if pts is None or len(pts) == 0:
            continue
        
        M_def = np.array(cache_def.GetLocalToWorldTransform(p))
        M_0 = np.array(cache_0.GetLocalToWorldTransform(p))
        
        P_def = np.array([M_def.T[:3, :3] @ pt + M_def.T[:3, 3] for pt in pts])
        P_0 = np.array([M_0.T[:3, :3] @ pt + M_0.T[:3, 3] for pt in pts])
        
        diff = np.max(np.abs(P_def - P_0))
        diffs.append(diff)
        part_names.append(pname)
        centroids_def.append(P_def.mean(axis=0))
        centroids_0.append(P_0.mean(axis=0))

centroids_def = np.array(centroids_def)
centroids_0 = np.array(centroids_0)
print(f'Total meshes evaluated: {len(part_names)}', flush=True)
print(f'Max diff between Default() and TimeCode(0) across all points: {max(diffs):.6f}', flush=True)
print('Centroids_def min:', np.round(centroids_def.min(0), 3), 'max:', np.round(centroids_def.max(0), 3), 'mean:', np.round(centroids_def.mean(0), 3), flush=True)
print('Centroids_0 min:', np.round(centroids_0.min(0), 3), 'max:', np.round(centroids_0.max(0), 3), 'mean:', np.round(centroids_0.mean(0), 3), flush=True)
print(f'Number of unique centroid positions: {len(np.unique(np.round(centroids_def, 3), axis=0))}', flush=True)
