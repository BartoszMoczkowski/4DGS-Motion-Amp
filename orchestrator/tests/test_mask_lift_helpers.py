"""Sandbox tests for T22 ``mask_lift.py`` helper functions (no GPU needed).

Tests the pure-numpy/scipy pieces: k-NN graph building, graph dilation, mask path
resolution, and mask image loading.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


def test_build_knn_edges_shape_and_symmetry():
    """_build_knn_edges returns an undirected edge set with no self-loops."""
    from pipeline.vendored.cuda.mask_lift import _build_knn_edges

    rng = np.random.RandomState(0)
    xyz = rng.randn(50, 3).astype(np.float32)
    edges = _build_knn_edges(xyz, k=6)

    assert edges.ndim == 2
    assert edges.shape[1] == 2
    # No self-loops
    assert (edges[:, 0] != edges[:, 1]).all()
    # Symmetric: for every (u,v) there is a (v,u)
    edge_set = set(map(tuple, edges.tolist()))
    for u, v in edges.tolist():
        assert (v, u) in edge_set


def test_dilate_graph_expands_roi():
    """_dilate_graph expands a bool mask along k-NN edges."""
    from pipeline.vendored.cuda.mask_lift import _build_knn_edges, _dilate_graph

    rng = np.random.RandomState(1)
    xyz = rng.randn(100, 3).astype(np.float32)
    edges = _build_knn_edges(xyz, k=6)

    # Seed a single point
    mask = np.zeros(100, dtype=bool)
    mask[0] = True

    dilated = _dilate_graph(mask, edges, hops=1)
    assert dilated.sum() > mask.sum()
    # The seed point should still be True
    assert dilated[0]

    # Full dilation eventually reaches everything (connected graph)
    fully = _dilate_graph(mask, edges, hops=100)
    assert fully.all()


def test_find_mask_path_priority():
    """_find_mask_path checks candidates in the expected order."""
    from pipeline.vendored.cuda.mask_lift import _find_mask_path

    import tempfile
    with tempfile.TemporaryDirectory() as masks_dir:
        # Create only the static mask
        static_path = os.path.join(masks_dir, "cam01.png")
        Image.new("L", (10, 10), color=255).save(static_path)

        # Frame-specific mask does not exist
        assert _find_mask_path(masks_dir, "cam01", frame_name="frame_00001.png") == static_path

        # Now create the frame-specific mask — it should take priority
        frame_dir = os.path.join(masks_dir, "cam01")
        os.makedirs(frame_dir)
        frame_path = os.path.join(frame_dir, "frame_00001.png")
        Image.new("L", (10, 10), color=255).save(frame_path)
        assert _find_mask_path(masks_dir, "cam01", frame_name="frame_00001.png") == frame_path


def test_find_mask_path_none():
    """_find_mask_path returns None when no mask exists."""
    from pipeline.vendored.cuda.mask_lift import _find_mask_path

    import tempfile
    with tempfile.TemporaryDirectory() as masks_dir:
        assert _find_mask_path(masks_dir, "cam01") is None


def test_load_mask_grayscale_threshold():
    """_load_mask reads grayscale and thresholds at 128."""
    from pipeline.vendored.cuda.mask_lift import _load_mask

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # White image -> all True
        white = Image.new("L", (4, 4), color=255)
        white_path = os.path.join(tmp, "white.png")
        white.save(white_path)
        assert _load_mask(white_path).all()

        # Black image -> all False
        black = Image.new("L", (4, 4), color=0)
        black_path = os.path.join(tmp, "black.png")
        black.save(black_path)
        assert not _load_mask(black_path).any()

        # Gray at 127 -> False, 129 -> True
        gray = Image.new("L", (2, 2), color=127)
        gray_path = os.path.join(tmp, "gray127.png")
        gray.save(gray_path)
        assert not _load_mask(gray_path).any()

        gray2 = Image.new("L", (2, 2), color=129)
        gray2_path = os.path.join(tmp, "gray129.png")
        gray2.save(gray2_path)
        assert _load_mask(gray2_path).all()


def test_load_mask_resize():
    """_load_mask resizes to target_size when provided."""
    from pipeline.vendored.cuda.mask_lift import _load_mask

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        im = Image.new("L", (10, 10), color=255)
        path = os.path.join(tmp, "mask.png")
        im.save(path)

        mask = _load_mask(path, target_size=(4, 4))
        assert mask.shape == (4, 4)


def test_load_mask_missing():
    """_load_mask returns None for a missing file."""
    from pipeline.vendored.cuda.mask_lift import _load_mask

    assert _load_mask("/nonexistent/path.png") is None
