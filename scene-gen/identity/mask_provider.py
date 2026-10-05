"""scene-gen/identity/mask_provider.py — Multi-view mask provider for Gaussian Grouping.

Supports two modes with lazy on-demand loading to prevent RAM exhaustion:
1. "oracle": Loads GT instance segmentation on demand from Isaac Sim render directory.
2. "sam": Extracts 2D masks via SAM (ViT-B), tracks them, and unifies mask IDs across views.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class MaskProvider:
    """Base class for on-demand 2D mask providers."""

    def get_mask(self, cam_name: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def cam_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def n_frames(self) -> int:
        raise NotImplementedError


class OracleMaskProvider(MaskProvider):
    """Lazily loads and maps Oracle GT instance segmentation masks from render_dir."""

    def __init__(
        self,
        render_dir: str,
        cam_names: List[str],
        n_frames: int,
        cache_size: int = 32,
    ):
        self._render_dir = render_dir
        self._cam_names = cam_names
        self._n_frames = n_frames
        self._cache_size = cache_size
        self._cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()

        # Parse class_to_id from label_names.json if available
        self.class_to_id: Dict[str, int] = {}
        label_names_path = os.path.join(render_dir, "label_names.json")
        if os.path.isfile(label_names_path):
            with open(label_names_path, "r") as f:
                raw_labels = json.load(f)
                self.class_to_id = {name: int(idx) + 1 for idx, name in raw_labels.items()}
            self._num_classes = len(self.class_to_id) + 1
        else:
            # Check 1 sample mapping to find classes
            sample_map_path = os.path.join(render_dir, cam_names[0], "instance_segmentation", "instance_segmentation_semantics_mapping_0000.json")
            if os.path.isfile(sample_map_path):
                with open(sample_map_path, "r") as f:
                    sm = json.load(f)
                max_id = 1
                for k, v in sm.items():
                    c_name = v.get("class", "")
                    if "cube_01" in c_name:
                        max_id = max(max_id, 1)
                    elif "cube_02" in c_name:
                        max_id = max(max_id, 2)
                    elif c_name.startswith("part_"):
                        try:
                            max_id = max(max_id, int(c_name.split("_")[1]))
                        except Exception:
                            max_id = max(max_id, int(k))
                self._num_classes = max_id + 1
            else:
                self._num_classes = 108

        print(f"[OracleMaskProvider] Initialized for {len(cam_names)} cameras, {n_frames} frames, {self._num_classes} classes")

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def cam_names(self) -> List[str]:
        return self._cam_names

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get_mask(self, cam_name: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        # Check cache
        cache_key = (cam_name, frame_idx)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            mapped = self._cache[cache_key]
            if target_size is not None and (mapped.shape[1], mapped.shape[0]) != target_size:
                return cv2.resize(mapped, target_size, interpolation=cv2.INTER_NEAREST)
            return mapped

        # Load from disk
        cam_inst_dir = os.path.join(self._render_dir, cam_name, "instance_segmentation")
        img_path = os.path.join(cam_inst_dir, f"instance_segmentation_{frame_idx:04d}.png")
        map_path = os.path.join(cam_inst_dir, f"instance_segmentation_semantics_mapping_{frame_idx:04d}.json")

        if not os.path.isfile(img_path) or not os.path.isfile(map_path):
            # Fallback to frame 0
            img_path = os.path.join(cam_inst_dir, "instance_segmentation_0000.png")
            map_path = os.path.join(cam_inst_dir, "instance_segmentation_semantics_mapping_0000.json")

        raw_im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw_im is None:
            # Return dummy zeros
            h, w = (target_size[1], target_size[0]) if target_size else (900, 1600)
            return np.zeros((h, w), dtype=np.int32)

        with open(map_path, "r") as f:
            sem_map = json.load(f)

        val_to_global: Dict[int, int] = {}
        for px_str, data in sem_map.items():
            px_val = int(px_str)
            cls_name = data.get("class", "BACKGROUND") if isinstance(data, dict) else str(data)
            base_cls_name = cls_name.split("/")[-1]
            if base_cls_name in ("BACKGROUND", "UNLABELLED", ""):
                val_to_global[px_val] = 0
            elif base_cls_name in self.class_to_id:
                val_to_global[px_val] = self.class_to_id[base_cls_name]
            elif cls_name in self.class_to_id:
                val_to_global[px_val] = self.class_to_id[cls_name]
            else:
                if "cube_01" in base_cls_name:
                    val_to_global[px_val] = 1
                elif "cube_02" in base_cls_name:
                    val_to_global[px_val] = 2
                elif base_cls_name.startswith("part_"):
                    try:
                        val_to_global[px_val] = int(base_cls_name.split("_")[1])
                    except Exception:
                        val_to_global[px_val] = px_val
                else:
                    val_to_global[px_val] = px_val

        max_val = int(raw_im.max())
        lut = np.zeros(max_val + 1, dtype=np.int32)
        for v, gid in val_to_global.items():
            if v <= max_val:
                lut[v] = gid

        mapped = lut[raw_im]

        # Put into cache
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = mapped

        if target_size is not None and (mapped.shape[1], mapped.shape[0]) != target_size:
            return cv2.resize(mapped, target_size, interpolation=cv2.INTER_NEAREST)

        return mapped


class SAMMaskProvider(MaskProvider):
    """Provides multi-view consistent SAM instance masks."""

    def __init__(
        self,
        unified_masks: Dict[str, np.ndarray],
        num_classes: int,
        cam_names: List[str],
        n_frames: int,
    ):
        self._unified_masks = unified_masks
        self._num_classes = num_classes
        self._cam_names = cam_names
        self._n_frames = n_frames

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def cam_names(self) -> List[str]:
        return self._cam_names

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get_mask(self, cam_name: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        matched_k = next((k for k in self._unified_masks if k in cam_name or cam_name in k), None)
        if matched_k is None:
            matched_k = list(self._unified_masks.keys())[0]

        mask = self._unified_masks[matched_k]
        if target_size is not None and (mask.shape[1], mask.shape[0]) != target_size:
            return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask


def create_sam_mask_provider(
    cam_dirs: Dict[str, str],
    sam_checkpoint: str,
    output_cache_dir: str,
    pc_xyz: np.ndarray,
    cameras: List[any],
    min_area: int = 200,
    points_per_side: int = 32,
    device: str = "cuda",
) -> SAMMaskProvider:
    """Extract SAM masks on reference frames, match across views via 3D projection, and build SAMMaskProvider."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    os.makedirs(output_cache_dir, exist_ok=True)
    cache_file = os.path.join(output_cache_dir, "sam_unified_masks.npz")

    cam_names = list(cam_dirs.keys())
    n_frames = 60

    if os.path.isfile(cache_file):
        print(f"[SAMMaskProvider] Loading cached SAM masks from {cache_file}")
        cached = np.load(cache_file)
        unified_masks = {k: cached[k] for k in cached.files}
        max_cls = max(int(m.max()) for m in unified_masks.values())
        return SAMMaskProvider(unified_masks, max_cls + 1, cam_names, n_frames)

    print(f"[SAMMaskProvider] Initializing SAM ({sam_checkpoint}) on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.94,
        crop_n_layers=0,
        min_mask_region_area=min_area,
    )

    per_cam_ref_masks: Dict[str, List[np.ndarray]] = {}
    H, W = 0, 0

    for cam, cdir in cam_dirs.items():
        frame0_candidates = [
            os.path.join(cdir, "rgb_0000.png"),
            os.path.join(cdir, "rgb_0001.png"),
            os.path.join(cdir, "frame_00001.jpg"),
            os.path.join(cdir, "frame_00000.jpg"),
            os.path.join(cdir, "frame_00001.png"),
            os.path.join(cdir, "frame_00000.png"),
        ]
        img_path = next((p for p in frame0_candidates if os.path.isfile(p)), None)
        if img_path is None:
            # Also try listing directory for any image
            if os.path.isdir(cdir):
                all_imgs = sorted([f for f in os.listdir(cdir) if f.endswith((".png", ".jpg", ".jpeg"))])
                if all_imgs:
                    img_path = os.path.join(cdir, all_imgs[0])
        if img_path is None:
            continue

        im_bgr = cv2.imread(img_path)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        H, W = im_rgb.shape[:2]

        print(f"[SAMMaskProvider] Generating SAM masks for {cam}...")
        raw_masks = mask_gen.generate(im_rgb)

        sorted_masks = sorted(raw_masks, key=lambda x: x["area"], reverse=True)
        total_px = H * W
        filtered = [m["segmentation"] for m in sorted_masks if m["area"] < 0.75 * total_px and m["area"] > min_area]
        per_cam_ref_masks[cam] = filtered
        print(f"[SAMMaskProvider]   {cam}: retained {len(filtered)} candidate masks")

    print("[SAMMaskProvider] Associating SAM masks across views via 3D point projection...")
    ref_cameras: Dict[str, any] = {}
    for cam_name in cam_names:
        for i in range(len(cameras)):
            if hasattr(cameras, "dataset") and hasattr(cameras.dataset, "image_paths"):
                path_str = str(cameras.dataset.image_paths[i])
                if cam_name in path_str and ("00000" in path_str or "00001" in path_str):
                    c = cameras[i]
                    c.original_image = None
                    ref_cameras[cam_name] = c
                    break
            else:
                c = cameras[i]
                c.original_image = None
                ref_cameras[cam_name] = c
                break

    nodes = []
    node_to_idx = {}
    for cam, mask_list in per_cam_ref_masks.items():
        for m_idx in range(len(mask_list)):
            node = (cam, m_idx)
            node_to_idx[node] = len(nodes)
            nodes.append(node)

    n_nodes = len(nodes)
    co_occurrence = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    pts_sub = pc_xyz[::4]
    pts_homo = np.concatenate([pts_sub, np.ones((len(pts_sub), 1))], axis=1)

    pt_cam_masks: Dict[int, Dict[str, int]] = {i: {} for i in range(len(pts_sub))}

    for cam_key, cam_obj in ref_cameras.items():
        if cam_key not in per_cam_ref_masks:
            continue

        masks_list = per_cam_ref_masks[cam_key]
        full_proj = cam_obj.full_proj_transform.cpu().numpy()

        p_proj = pts_homo @ full_proj
        w = p_proj[:, 3:4]
        valid = (w > 0.1).reshape(-1)
        ndc = p_proj / np.maximum(w, 1e-6)
        u = ((ndc[:, 0] + 1.0) * 0.5 * W).astype(np.int32)
        v = ((ndc[:, 1] + 1.0) * 0.5 * H).astype(np.int32)

        in_screen = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for pt_idx in np.where(in_screen)[0]:
            ui, vi = u[pt_idx], v[pt_idx]
            for m_idx, m_arr in reversed(list(enumerate(masks_list))):
                if m_arr[vi, ui]:
                    pt_cam_masks[pt_idx][cam_key] = m_idx
                    break

    for pt_idx, hits in pt_cam_masks.items():
        cam_items = list(hits.items())
        for i in range(len(cam_items)):
            c1, m1 = cam_items[i]
            n1 = node_to_idx.get((c1, m1))
            for j in range(i + 1, len(cam_items)):
                c2, m2 = cam_items[j]
                n2 = node_to_idx.get((c2, m2))
                if n1 is not None and n2 is not None:
                    co_occurrence[n1, n2] += 1
                    co_occurrence[n2, n1] += 1

    from scipy.sparse import csgraph
    adj = co_occurrence >= 15
    np.fill_diagonal(adj, False)
    n_components, labels = csgraph.connected_components(adj, directed=False)
    print(f"[SAMMaskProvider] Unified {n_nodes} view-specific masks into {n_components} global instances")

    node_to_global = {}
    for node, comp_id in zip(nodes, labels):
        node_to_global[node] = int(comp_id + 1)

    unified_masks: Dict[str, np.ndarray] = {}
    for cam, mask_list in per_cam_ref_masks.items():
        int_mask = np.zeros((H, W), dtype=np.int32)
        sorted_indices = sorted(range(len(mask_list)), key=lambda i: mask_list[i].sum(), reverse=True)
        for m_idx in sorted_indices:
            gid = node_to_global.get((cam, m_idx), 0)
            int_mask[mask_list[m_idx]] = gid
        unified_masks[cam] = int_mask

    np.savez_compressed(cache_file, **unified_masks)
    print(f"[SAMMaskProvider] Saved unified SAM masks to {cache_file}")

    return SAMMaskProvider(unified_masks, n_components + 1, cam_names, n_frames)
