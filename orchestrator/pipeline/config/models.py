"""Pydantic models for the unified pipeline config.

One typed schema covering every stage's settings, replacing three scattered sources:
``omniverse-pipeline/omniverse_pipeline/capture_config_pump.yaml``, ``core/arguments/multipleview/<name>.py``, and the
CLI flags baked into ``omniverse-pipeline/train_pump.sh`` / ``motion-seg/motion_seg/run.sh`` / ``core/render_amp.py``. See
``pipeline/config/MIGRATION.md`` for the exact old-setting -> new-field mapping and
``planning/tasks/T02-config-schema-and-presets.md`` for scope.

Design notes:
- Every model forbids unknown fields (``extra="forbid"``) so a typo or stale key fails fast at
  validation time instead of being silently ignored.
- Field names intentionally mirror the original CLI flag / dict key names (including the
  inconsistent casing already in the codebase, e.g. ``convert_SHs_python``) so migrating a
  legacy setting is a rename-free copy — see MIGRATION.md rather than "cleaning up" names here.
- This module only *describes* settings; it does not execute anything (T04+) and does not know
  about artifact paths / run directories (T03) beyond the same defaults the old scripts had.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for every config model: unknown fields are a hard error, not silently dropped."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- 4DGS core param groups (arguments/__init__.py: ModelParams/PipelineParams/
#     ModelHiddenParams/OptimizationParams) --------------------------------------------------


class KPlanesConfig(StrictModel):
    """``ModelHiddenParams.kplanes_config`` — not CLI-settable upstream (dict-typed), config-only."""

    grid_dimensions: int = 2
    input_coordinate_dim: int = 4
    output_coordinate_dim: int = 32
    # last entry is the TIME resolution of the deformation grid.
    resolution: list[int] = Field(default_factory=lambda: [64, 64, 64, 25])


class ModelHiddenParams(StrictModel):
    """Deformation-network hyperparameters (``arguments/__init__.py: ModelHiddenParams``)."""

    net_width: int = 64
    timebase_pe: int = 4
    defor_depth: int = 1
    posebase_pe: int = 10
    scale_rotation_pe: int = 2
    opacity_pe: int = 2
    timenet_width: int = 64
    timenet_output: int = 32
    bounds: float = 1.6
    plane_tv_weight: float = 0.0001
    time_smoothness_weight: float = 0.01
    l1_time_planes: float = 0.0001
    kplanes_config: KPlanesConfig = Field(default_factory=KPlanesConfig)
    multires: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    no_dx: bool = False
    no_grid: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = True
    no_dshs: bool = True
    empty_voxel: bool = False
    grid_pe: int = 0
    static_mlp: bool = False
    apply_rotation: bool = False


class OptimizationParams(StrictModel):
    """Training-only hyperparameters (``arguments/__init__.py: OptimizationParams``)."""

    dataloader: bool = False
    zerostamp_init: bool = False
    custom_sampler: Optional[str] = None
    iterations: int = 30_000
    coarse_iterations: int = 3000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 20_000
    deformation_lr_init: float = 0.00016
    deformation_lr_final: float = 0.000016
    grid_lr_init: float = 0.0016
    grid_lr_final: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0
    lambda_lpips: float = 0
    weight_constraint_init: float = 1
    weight_constraint_after: float = 0.2
    weight_decay_iteration: int = 5000
    opacity_reset_interval: int = 3000
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold_coarse: float = 0.0002
    densify_grad_threshold_fine_init: float = 0.0002
    densify_grad_threshold_after: float = 0.0002
    pruning_from_iter: int = 500
    pruning_interval: int = 100
    opacity_threshold_coarse: float = 0.005
    opacity_threshold_fine_init: float = 0.005
    opacity_threshold_fine_after: float = 0.005
    batch_size: int = 1
    add_point: bool = False


class ModelParams(StrictModel):
    """Scene/model params (``arguments/__init__.py: ModelParams``).

    ``source_path``/``model_path`` are normally *derived* from the scene name by the DAG's
    artifact wiring (T03/T05), not hand-set in a preset — they're exposed here so a preset can
    still pin them explicitly (matches how ``train_pump.sh`` invokes ``train.py -s ... --expname
    ...`` today) until that wiring lands.
    """

    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = True
    data_device: str = "cuda"
    eval: bool = True
    render_process: bool = False
    add_points: bool = False
    extension: str = ".png"
    llffhold: int = 8


class PipelineParams(StrictModel):
    """Renderer params (``arguments/__init__.py: PipelineParams``)."""

    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


# --- train / render / seg_extract (thin wrappers around the shared param groups above) -----


class TrainConfig(StrictModel):
    """``train.py`` extra flags not covered by the shared param groups."""

    ip: str = "127.0.0.1"
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = False
    test_iterations: list[int] = Field(default_factory=lambda: [3000, 7000, 14000])
    save_iterations: list[int] = Field(
        default_factory=lambda: [14000, 20000, 30000, 45000, 60000]
    )
    quiet: bool = False
    checkpoint_iterations: list[int] = Field(default_factory=list)
    start_checkpoint: Optional[str] = None
    expname: str = ""
    configs: str = ""


class RenderConfig(StrictModel):
    """``render.py`` flags (also shared, minus ``--configs``, by ``extract_trajectories.py``)."""

    iteration: int = -1
    skip_train: bool = False
    skip_test: bool = False
    quiet: bool = False
    skip_video: bool = False
    configs: Optional[str] = None


class SegExtractConfig(StrictModel):
    """``motion-seg/motion_seg/extract_trajectories.py`` flags (beyond the shared param groups)."""

    iteration: int = -1
    configs: Optional[str] = None
    n_times: int = 60
    # "" => script computes "<model_path>/trajectories.npz" itself.
    out: str = ""


# --- segmentation: role "segment" with two selectable implementations ----------------------


class SegmentRigidConfig(StrictModel):
    """``motion-seg/motion_seg/segment_rigid.py`` (segmentation Option B: rigidity-graph clustering)."""

    k: int = 12
    min_size: int = 15
    threshold_mult: float = 1.0
    opacity_thresh: float = 0.1
    # None => script computes "<out-without-ext>_preview.png"; "" explicitly skips.
    preview_png: Optional[str] = None


class SegmentMbsConfig(StrictModel):
    """``motion-seg/motion_seg/mbs_infer.py`` (segmentation Option A: MultiBodySync MotNet inference).

    ``checkpoint`` has no sensible default (it's a required CLI flag pointing at a downloaded
    ``.pth.tar``); left empty here and enforced by :meth:`SegmentConfig._check_impl_ready`.
    """

    checkpoint: str = ""
    n_points: int = 4000
    n_views: int = 4
    n_sub: int = 256
    opacity_thresh: float = 0.1
    alpha: float = 0.05
    seed: int = 0


class SegmentConfig(StrictModel):
    """Role ``segment``: picks + configures one of the two segmentation implementations.

    This is the "role -> impl selection" the config layer owns (T02 scope) — the registry
    (T04) resolves ``impl`` to an actual registered stage class named e.g. ``segment.rigid`` /
    ``segment.mbs``; this model only validates that the choice is coherent.
    """

    impl: Literal["rigid", "mbs"] = "rigid"
    rigid: SegmentRigidConfig = Field(default_factory=SegmentRigidConfig)
    mbs: SegmentMbsConfig = Field(default_factory=SegmentMbsConfig)

    @model_validator(mode="after")
    def _check_impl_ready(self) -> "SegmentConfig":
        if self.impl == "mbs" and not self.mbs.checkpoint:
            raise ValueError(
                "segment.impl == 'mbs' requires segment.mbs.checkpoint to be set "
                "(path to a downloaded MultiBodySync checkpoint)"
            )
        return self


class SegEvalConfig(StrictModel):
    """``motion-seg/motion_seg/evaluate_segmentation.py`` flags."""

    drop_floaters: bool = False
    recolored_ply: Optional[str] = None
    # None => script computes "<pred-without-ext>_vs_gt.png"; "" explicitly skips.
    comparison_png: Optional[str] = None
    top_n: int = 15


# --- amplification ----------------------------------------------------------------------------

#: Fixed channel order used by core/render_amp.py / amp-ui/amp_ui/ampUI.py (positional indexing into
#: amp_factors/freq_cutoffs lists) — this is the canonical channel list for the new schema.
AMP_CHANNELS: tuple[str, ...] = (
    "pos3d",
    "pos2d",
    "rotation",
    "scale",
    "opacity",
    "SHs",
    "color",
    "cov3D",
)

#: amp-ui/amp_ui/ampUI.py's Streamlit method labels map to different underlying functions than core/render_amp.py's
#: CLI ``--method`` strings. The new schema standardizes on the CLI names; this alias table lets
#: a future UI/wrapper (T09/T15) translate the Streamlit label a user picks back to a canonical
#: ``AmpConfig.method`` value. See MIGRATION.md for the full reconciliation.
AMP_METHOD_ALIASES: dict[str, str] = {
    "base": "eulerian",
    "base segmented": "eulerian_mod",
    "abs": "eulerian_abs",
    "abs segmented": "eulerian_abs_mod",
}


class AmpChannelConfig(StrictModel):
    """Per-channel amplification settings. ``factor == -1`` means "don't amplify this channel"."""

    factor: float = -1.0
    freq_low: float = 0.0
    freq_high: float = 1.0


class AmpConfig(StrictModel):
    """``core/render_amp.py`` / ``amp-ui/amp_ui/ampUI.py`` flags."""

    method: Literal["eulerian", "eulerian_abs", "eulerian_mod", "eulerian_abs_mod"] = "eulerian"
    low_vram_mode: bool = False
    frozen_cam: bool = False
    video_path: str = "render.mp4"
    video_fps: int = 20
    skip_train: bool = False
    skip_test: bool = False
    skip_video: bool = False
    quiet: bool = False
    iteration: int = -1
    configs: Optional[str] = None
    channels: dict[str, AmpChannelConfig] = Field(
        default_factory=lambda: {c: AmpChannelConfig() for c in AMP_CHANNELS}
    )

    @model_validator(mode="after")
    def _check_channels(self) -> "AmpConfig":
        unknown = set(self.channels) - set(AMP_CHANNELS)
        if unknown:
            raise ValueError(
                f"unknown amp channel(s) {sorted(unknown)}; must be one of {AMP_CHANNELS}"
            )
        return self


# --- Isaac capture + prep -------------------------------------------------------------------


class LightingConfig(StrictModel):
    """``capture_config*.yaml: lighting`` — a headless render is black without this."""

    enabled: bool = True
    force: bool = False
    add_dome: bool = True
    dome_intensity: float = 1000.0
    # None | "generate" | a texture path.
    dome_texture: Optional[str] = None
    bg_base: list[int] = Field(default_factory=lambda: [6, 7, 11])
    bg_line: list[int] = Field(default_factory=lambda: [28, 32, 44])
    bg_dot: list[int] = Field(default_factory=lambda: [48, 54, 70])
    bg_step: int = 64
    add_distant: bool = True
    distant_intensity: float = 4000.0
    distant_elev_deg: float = 45.0
    distant_azim_deg: float = -45.0


class CaptureSceneConfig(StrictModel):
    usd_path: str = ""
    subject_prim: Optional[str] = None
    semantic_roots: list[str] = Field(default_factory=list)


class CaptureRigConfig(StrictModel):
    layout: str = "dome"
    n_cameras: int = 10
    radius_scale: float = 2.5
    n_rings: int = 3
    min_elev_deg: float = 15.0
    max_elev_deg: float = 75.0


class CaptureFrameConfig(StrictModel):
    """``capture_config*.yaml: capture`` section (renders as ``CaptureConfig.capture``)."""

    width: int = 1600
    height: int = 900
    vfov_deg: float = 45.0
    num_frames: int = 60
    rt_subframes: int = 8
    # None => script defaults to radius*0.05 / radius*4.0 at runtime.
    near: Optional[float] = None
    far: Optional[float] = None


class CaptureOutputConfig(StrictModel):
    capture_dir: str = ""
    instance_segmentation: bool = True
    semantic_segmentation: bool = True
    num_init_points: int = 100_000


class CaptureConfig(StrictModel):
    """``omni_capture.py`` + its ``capture_config*.yaml`` (stage ``capture.isaac``)."""

    headless: bool = True
    scene: CaptureSceneConfig = Field(default_factory=CaptureSceneConfig)
    rig: CaptureRigConfig = Field(default_factory=CaptureRigConfig)
    capture: CaptureFrameConfig = Field(default_factory=CaptureFrameConfig)
    output: CaptureOutputConfig = Field(default_factory=CaptureOutputConfig)
    lighting: LightingConfig = Field(default_factory=LightingConfig)


class ConvertConfig(StrictModel):
    """``omni_to_4dgs.py`` flags (stage ``convert``)."""

    name: str = "omni_scene"
    near: float = 0.1
    far: float = 1000.0
    target_radius: float = 4.0


class PrepSplitConfig(StrictModel):
    """``split_mesh.py`` flags (stage ``prep_split.default``)."""

    group: str = "CONJUNTO_BOMBAS"
    min_faces: int = 1
    preview: Optional[str] = None
    no_usd: bool = False
    weld_tol: float = 1e-5


class PrepMotionConfig(StrictModel):
    """``add_motion.py`` flags (stage ``prep_motion.default``).

    ``trans_amp_mm``/``rot_surface_mm`` stay in millimetres here (as authored); the script's
    internal conversion to stage units via the USD's ``metersPerUnit`` is a runtime detail, not
    a config concern.
    """

    group: str = "CONJUNTO_BOMBAS"
    num_frames: int = 60
    fps: float = 24.0
    trans_amp_mm: tuple[float, float] = (1.0, 4.0)
    rot_surface_mm: tuple[float, float] = (0.5, 3.0)
    rot_deg_max: float = 3.0
    freq: tuple[int, int] = (2, 5)
    # 0 = each part gets its own independent motion (finest GT segments); K>0 clusters parts
    # into K shared-motion groups.
    groups: int = 0
    exclude: list[str] = Field(default_factory=lambda: ["frame_base"])
    seed: int = 0
    plot: Optional[str] = None


# --- top-level ---------------------------------------------------------------------------------


class PipelineConfig(StrictModel):
    """The whole resolved, validated config for one run.

    Built by layering YAML presets (``base <- scene <- experiment``, see
    ``pipeline/config/resolver.py``) and validating the merged result against this schema.
    """

    name: str = "base"

    prep_split: PrepSplitConfig = Field(default_factory=PrepSplitConfig)
    prep_motion: PrepMotionConfig = Field(default_factory=PrepMotionConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    convert: ConvertConfig = Field(default_factory=ConvertConfig)

    model: ModelParams = Field(default_factory=ModelParams)
    pipeline_params: PipelineParams = Field(default_factory=PipelineParams)
    hidden: ModelHiddenParams = Field(default_factory=ModelHiddenParams)
    optim: OptimizationParams = Field(default_factory=OptimizationParams)
    train: TrainConfig = Field(default_factory=TrainConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    seg_extract: SegExtractConfig = Field(default_factory=SegExtractConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    seg_eval: SegEvalConfig = Field(default_factory=SegEvalConfig)

    amp: AmpConfig = Field(default_factory=AmpConfig)
