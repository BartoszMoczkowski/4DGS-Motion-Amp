"""Unified pydantic config schema and layered presets (base <- scene <- experiment).

See ``planning/tasks/T02-config-schema-and-presets.md`` for scope and ``MIGRATION.md`` (next to
this file) for the field-by-field mapping from the old scattered config sources
(``capture_config_pump.yaml``, ``arguments/multipleview/*.py``, ``.sh`` flags) to this schema.
"""

from __future__ import annotations

from .bridge import render_bridge_source, write_bridge
from .loader import (
    load_legacy_capture_yaml,
    load_legacy_hyperparams,
    merge_legacy_sources,
)
from .models import (
    AMP_CHANNELS,
    AMP_METHOD_ALIASES,
    AmpChannelConfig,
    AmpConfig,
    CaptureConfig,
    CaptureFrameConfig,
    CaptureOutputConfig,
    CaptureRigConfig,
    CaptureSceneConfig,
    ConvertConfig,
    KPlanesConfig,
    LightingConfig,
    ModelHiddenParams,
    ModelParams,
    OptimizationParams,
    PipelineConfig,
    PipelineParams,
    PrepMotionConfig,
    PrepSplitConfig,
    RenderConfig,
    SegEvalConfig,
    SegExtractConfig,
    SegmentConfig,
    SegmentMbsConfig,
    SegmentRigidConfig,
    TrainConfig,
)
from .resolver import list_presets, resolve_preset, validate_config

__all__ = [
    # top-level
    "PipelineConfig",
    # 4DGS core param groups
    "ModelParams",
    "PipelineParams",
    "ModelHiddenParams",
    "OptimizationParams",
    "KPlanesConfig",
    # train/render/seg_extract
    "TrainConfig",
    "RenderConfig",
    "SegExtractConfig",
    # segmentation (role -> impl)
    "SegmentConfig",
    "SegmentRigidConfig",
    "SegmentMbsConfig",
    "SegEvalConfig",
    # amplification
    "AmpConfig",
    "AmpChannelConfig",
    "AMP_CHANNELS",
    "AMP_METHOD_ALIASES",
    # isaac capture + prep
    "CaptureConfig",
    "CaptureSceneConfig",
    "CaptureRigConfig",
    "CaptureFrameConfig",
    "CaptureOutputConfig",
    "LightingConfig",
    "ConvertConfig",
    "PrepSplitConfig",
    "PrepMotionConfig",
    # resolver
    "list_presets",
    "resolve_preset",
    "validate_config",
    # migration helpers
    "load_legacy_capture_yaml",
    "load_legacy_hyperparams",
    "merge_legacy_sources",
    # T09 CUDA-stage config bridge
    "render_bridge_source",
    "write_bridge",
]
