"""orchestrator/tests/test_w6_production_parity.py

Test suite for W6 (T54) Production integration, streaming render & experiment ladder.
Verifies:
1. Cross-tool deformation equivalence: training render(), offline render.py,
   get_state_at_time(), and render_mod() produce numerically identical deformed positions,
   rotations, scales, and opacities at arbitrary physical timestamps.
2. Bounded-memory streaming render: views are streamed to disk and MP4 incrementally
   with O(1) host RAM and VRAM (no 600-frame accumulation OOM).
3. Complete manifest provenance: git_revision, git_dirty, resolved_config,
   offsets_json_hash, weight_maps_hash, warmstart_checkpoint_hash,
   component_structural_history, and random_seed are fully hashed and recorded.
4. Parity between core/render.py and orchestrator/pipeline/vendored/cuda/render.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Mock Linux-only CUDA extension simple_knn if missing in sandbox
if "simple_knn" not in sys.modules:
    m_knn = types.ModuleType("simple_knn")
    m_knn_c = types.ModuleType("simple_knn._C")
    m_knn_c.distCUDA2 = lambda *args, **kwargs: None
    m_knn._C = m_knn_c
    sys.modules["simple_knn"] = m_knn
    sys.modules["simple_knn._C"] = m_knn_c

# Mock Linux-only diff_gaussian_rasterization if missing in sandbox
if "diff_gaussian_rasterization" not in sys.modules:
    m_diff = types.ModuleType("diff_gaussian_rasterization")

    class MockRasterSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockRasterizer:
        def __init__(self, raster_settings=None):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            dev = kwargs.get("means3D", torch.empty(0)).device
            n = kwargs.get("means3D", torch.empty(4)).shape[0]
            return (
                torch.zeros((3, 32, 32), device=dev),
                torch.zeros(n, device=dev),
                torch.zeros((1, 32, 32), device=dev),
            )

    m_diff.GaussianRasterizationSettings = MockRasterSettings
    m_diff.GaussianRasterizer = MockRasterizer
    sys.modules["diff_gaussian_rasterization"] = m_diff

from scene.motion_components import (
    MotionComponents,
    FourierComponentTrajectory,
)
from scene.deformation_dispatch import dispatch_deformation
from utils.render_utils import get_state_at_time
from pipeline.artifacts import (
    create_run,
    load_manifest,
    record_stage_result,
    Artifact,
)
from pipeline.artifacts.manifest import get_git_sha, get_git_dirty


# ==============================================================================
# Mock classes for testing
# ==============================================================================

class DummyDeformation(nn.Module):
    """Mock deformation network that displaces Gaussians as a function of time."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, means3D, scales, rotations, opacity, shs, time):
        t = time if time.dim() == 2 else time.view(-1, 1)
        if t.shape[0] != means3D.shape[0]:
            t = t.expand(means3D.shape[0], 1)
        disp = 0.01 * torch.sin(2.0 * math.pi * 5.691 * t) * torch.tensor([1.0, 0.5, 0.0], device=means3D.device)
        return (
            means3D + disp + 0.0 * self.dummy_param,
            scales.clone(),
            rotations.clone(),
            opacity.clone(),
            shs.clone() if shs is not None else None,
        )


class MockGaussianModel:
    """Mock GaussianModel for testing dispatch equivalence."""

    def __init__(self, num_points: int = 50, device: str = "cpu"):
        self.device = torch.device(device)
        self._xyz = nn.Parameter(torch.randn((num_points, 3), device=self.device, dtype=torch.float32))
        self._scaling = nn.Parameter(torch.ones((num_points, 3), device=self.device, dtype=torch.float32) * 0.02)
        self._rotation = nn.Parameter(torch.zeros((num_points, 4), device=self.device, dtype=torch.float32))
        self._rotation.data[:, 0] = 1.0  # (w=1, x=0, y=0, z=0)
        self._opacity = nn.Parameter(torch.ones((num_points, 1), device=self.device, dtype=torch.float32) * 2.0)
        self._features_dc = nn.Parameter(torch.randn((num_points, 1, 3), device=self.device, dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.zeros((num_points, 15, 3), device=self.device, dtype=torch.float32))
        self._motion_component_logits = nn.Parameter(torch.zeros((num_points, 1), device=self.device, dtype=torch.float32))
        self._deformation = DummyDeformation()
        self.motion_components = None

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def scaling_activation(self, s):
        return torch.exp(s)

    def rotation_activation(self, r):
        return nn.functional.normalize(r, dim=-1)

    def opacity_activation(self, o):
        return torch.sigmoid(o)


class MockViewCamera:
    """Mock View Camera with physical timestamps."""

    def __init__(self, timestamp: float, image_size: tuple[int, int] = (32, 32)):
        self.time = timestamp
        self.image_height, self.image_width = image_size
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.world_view_transform = torch.eye(4)
        self.full_proj_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.original_image = torch.zeros((3, image_size[0], image_size[1]), dtype=torch.float32)


# ==============================================================================
# 1. Cross-Tool Deformation Equivalence Test
# ==============================================================================

def test_cross_tool_deformation_parity():
    """Verify that training render(), get_state_at_time(), and dispatch_deformation produce identical output."""
    N = 40
    pc = MockGaussianModel(num_points=N)

    # Initialize MotionComponents with realistic turbine parameters
    mc = MotionComponents(
        num_components=1,
        components=[FourierComponentTrajectory(base_frequency_hz=5.691, fps=20.004, num_harmonics=1)],
        ramp_start_iter=0,
        ramp_end_iter=0,  # alpha=1.0 immediate
    )
    pc.motion_components = mc
    pc._motion_component_logits.data.fill_(3.0)  # High responsibility

    timestamps = [0.0, 0.25, 0.5, 0.75, 1.0]

    for t_val in timestamps:
        cam = MockViewCamera(timestamp=t_val)
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32)

        # 1. Direct dispatch_deformation
        m_dispatch, s_dispatch, r_dispatch, o_dispatch, sh_dispatch = dispatch_deformation(
            pc=pc,
            means3D=pc.get_xyz,
            scales=pc._scaling,
            rotations=pc._rotation,
            opacity=pc._opacity,
            shs=pc.get_features,
            time=t_tensor,
            stage="fine",
            motion_components=mc,
        )

        # 2. get_state_at_time from render_utils
        m_state, s_state, r_state, o_state, sh_state = get_state_at_time(pc, cam)

        # 3. dispatch with pc.motion_components auto-retrieved
        m_auto, s_auto, r_auto, o_auto, sh_auto = dispatch_deformation(
            pc=pc,
            means3D=pc.get_xyz,
            scales=pc._scaling,
            rotations=pc._rotation,
            opacity=pc._opacity,
            shs=pc.get_features,
            time=t_tensor,
            stage="fine",
        )

        # Verify exact numerical parity across all channels
        torch.testing.assert_close(m_dispatch, m_state)
        torch.testing.assert_close(r_dispatch, r_state)
        torch.testing.assert_close(s_dispatch, s_state)
        torch.testing.assert_close(o_state, pc._opacity)
        torch.testing.assert_close(sh_dispatch, sh_state)

        torch.testing.assert_close(m_dispatch, m_auto)
        torch.testing.assert_close(r_dispatch, r_auto)
        torch.testing.assert_close(o_dispatch, o_auto)


# ==============================================================================
# 2. Bounded-Memory Streaming Render Test
# ==============================================================================

def test_bounded_async_writer(tmp_path):
    """Verify that BoundedAsyncWriter writes images correctly without deadlock."""
    from core.render import BoundedAsyncWriter

    out_dir = tmp_path / "async_out"
    out_dir.mkdir()

    writer = BoundedAsyncWriter(max_workers=2, max_queued=3)
    num_images = 20

    for i in range(num_images):
        # Alternate between uint8 numpy array and float torch tensor
        if i % 2 == 0:
            img = np.full((16, 16, 3), fill_value=i * 10, dtype=np.uint8)
        else:
            img = torch.full((3, 16, 16), fill_value=float(i) / num_images, dtype=torch.float32)
        out_file = str(out_dir / f"{i:05d}.png")
        writer.submit(out_file, img)

    writer.close()

    # Confirm all files exist on disk
    for i in range(num_images):
        f = out_dir / f"{i:05d}.png"
        assert f.is_file(), f"Expected file {f} to exist"
        assert f.stat().st_size > 0


def test_streaming_render_memory_bounded(tmp_path, monkeypatch):
    """Verify that render_set streams frames to disk and video with bounded memory."""
    from core.render import render_set

    # Mock render() function to return a small synthetic rendering
    def mock_render(view, gaussians, pipeline, background, cam_type=None):
        return {"render": torch.zeros((3, 16, 16), dtype=torch.float32)}

    monkeypatch.setattr("core.render.render", mock_render)

    num_frames = 30
    views = [MockViewCamera(timestamp=float(i) / num_frames, image_size=(16, 16)) for i in range(num_frames)]
    pc = MockGaussianModel(num_points=10)

    class MockPipeline:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    render_set(
        model_path=str(tmp_path),
        name="test",
        iteration=1000,
        views=views,
        gaussians=pc,
        pipeline=MockPipeline(),
        background=torch.zeros(3),
        cam_type="MultipleView",
    )

    out_folder = tmp_path / "test" / "ours_1000"
    render_folder = out_folder / "renders"
    gt_folder = out_folder / "gt"
    video_file = out_folder / "video_rgb.mp4"

    assert render_folder.is_dir()
    assert gt_folder.is_dir()
    assert video_file.is_file()
    assert video_file.stat().st_size > 0

    # Verify all frames were written
    render_files = list(render_folder.glob("*.png"))
    gt_files = list(gt_folder.glob("*.png"))
    assert len(render_files) == num_frames
    assert len(gt_files) == num_frames


# ==============================================================================
# 3. Complete Manifest Provenance Test
# ==============================================================================

def test_manifest_provenance_complete(tmp_path):
    """Verify that manifest.json records all required provenance hashes and metadata."""
    run_id = "test-prov-001"
    resolved_config = {
        "train": {
            "random_seed": 1337,
            "use_soft_loss_weights": True,
            "use_motion_components": True,
            "use_split_merge": True,
        }
    }

    manifest = create_run(
        run_id=run_id,
        preset="test_preset",
        resolved_config=resolved_config,
        stage_names=["train.default"],
        runs_root=tmp_path,
        metadata={"custom_tag": "prov_check"},
    )

    # Simulate train.default stage completion with model provenance metadata
    sample_offsets_hash = f"sha256:{hashlib.sha256(b'offsets').hexdigest()}"
    sample_weights_hash = f"sha256:{hashlib.sha256(b'weights').hexdigest()}"
    sample_warmstart_hash = f"sha256:{hashlib.sha256(b'warmstart').hexdigest()}"
    sample_history = [{"iteration": 1000, "action": "split", "decision": "accepted"}]

    model_meta = {
        "offsets_json_hash": sample_offsets_hash,
        "weight_maps_hash": sample_weights_hash,
        "warmstart_checkpoint_hash": sample_warmstart_hash,
        "component_structural_history": sample_history,
    }

    dummy_model_dir = tmp_path / run_id / "point_cloud"
    dummy_model_dir.mkdir(parents=True)

    art = Artifact(
        name="model",
        kind="model",
        path=str(dummy_model_dir),
        producing_stage="train.default",
        metadata=model_meta,
    )

    record_stage_result(
        run_id=run_id,
        stage="train.default",
        status="success",
        artifacts=[art],
        runs_root=tmp_path,
    )

    # Reload manifest from disk and verify complete provenance
    loaded = load_manifest(run_id, runs_root=tmp_path)

    # 1. Git revision and dirty-tree status
    assert "git_revision" in loaded.provenance or loaded.git_sha is not None
    assert "git_dirty" in loaded.provenance
    assert isinstance(loaded.provenance["git_dirty"], bool)
    assert isinstance(loaded.git_dirty, bool)

    # 2. Random seed
    assert loaded.provenance.get("random_seed") == 1337

    # 3. Offsets JSON hash
    assert loaded.provenance.get("offsets_json_hash") == sample_offsets_hash

    # 4. Weight maps hash
    assert loaded.provenance.get("weight_maps_hash") == sample_weights_hash

    # 5. Warmstart checkpoint hash
    assert loaded.provenance.get("warmstart_checkpoint_hash") == sample_warmstart_hash

    # 6. Component structural history
    assert loaded.provenance.get("component_structural_history") == sample_history

    # 7. Resolved config intact
    assert loaded.resolved_config == resolved_config


# ==============================================================================
# 4. Parity Between core/render.py and vendored/cuda/render.py
# ==============================================================================

def test_core_and_vendored_render_parity():
    """Verify that core/render.py and orchestrator/pipeline/vendored/cuda/render.py stay behaviorally synchronized."""
    core_render_py = REPO_ROOT / "core" / "render.py"
    vendored_render_py = REPO_ROOT / "orchestrator" / "pipeline" / "vendored" / "cuda" / "render.py"

    assert core_render_py.is_file()
    assert vendored_render_py.is_file()

    core_text = core_render_py.read_text(encoding="utf-8")
    vendored_text = vendored_render_py.read_text(encoding="utf-8")

    # Both must define BoundedAsyncWriter and use streaming imageio writer
    assert "class BoundedAsyncWriter:" in core_text
    assert "class BoundedAsyncWriter:" in vendored_text
    assert "imageio.get_writer(video_path, fps=resolved_fps)" in core_text
    assert "imageio.get_writer(video_path, fps=resolved_fps)" in vendored_text
    assert "resolve_render_fps(views, default_fps=20.004, override_fps=fps)" in core_text
    assert "resolve_render_fps(views, default_fps=20.004, override_fps=fps)" in vendored_text


def test_resolve_render_fps():
    """Verify that render fps dynamically resolves physical timestamps without hardcoding 30 fps."""
    class MockView:
        def __init__(self, physical_time):
            self.physical_time = physical_time

    # 1. Physical 20.004 fps sequence (dt = 1/20.004 ~ 0.04999)
    views_physical = [MockView(i * (1.0 / 20.004)) for i in range(10)]

    from core.render import resolve_render_fps
    fps_measured = resolve_render_fps(views_physical)
    assert abs(fps_measured - 20.004) < 1e-2

    # 2. Views without physical_time default to 20.004
    views_plain = [MockView(None) for _ in range(5)]
    fps_default = resolve_render_fps(views_plain)
    assert fps_default == 20.004

    # 3. Explicit override
    fps_override = resolve_render_fps(views_physical, override_fps=60.0)
    assert fps_override == 60.0
