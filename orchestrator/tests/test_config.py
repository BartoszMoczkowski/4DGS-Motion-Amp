"""Tests for T02 (unified config schema & presets).

Round-trips the `pump01` preset against the actual legacy files it was migrated from
(`omniverse_pipeline/capture_config_pump.yaml`, `arguments/multipleview/pump01.py`) rather than
trusting a hand-typed copy, and checks that the schema fails fast on invalid config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fields the legacy `arguments/multipleview/*.py` files redundantly restate inside their
# `ModelHiddenParams` dict even though they're really `ModelParams`/`OptimizationParams`-owned
# settings the schema already covers elsewhere (harmless duplication upstream — mmengine's merge
# just re-sets the same value). Excluded from the hidden-section round-trip diff; see
# MIGRATION.md's "known upstream quirks" section.
_LEGACY_HIDDEN_DUPLICATE_KEYS = {"render_process"}


def test_list_presets_includes_all_shipped_presets():
    from pipeline.config import list_presets

    assert {"base", "pump01", "pump01_segB_tuned"} <= set(list_presets())


def test_base_preset_validates_and_matches_multipleview_default():
    from pipeline.config import validate_config

    cfg = validate_config("base")
    assert cfg.name == "base"
    assert cfg.hidden.net_width == 128
    assert cfg.hidden.kplanes_config.resolution == [64, 64, 64, 150]
    assert cfg.optim.iterations == 15000
    assert cfg.optim.opacity_reset_interval == 3000  # left at the class default, on purpose


def test_pump01_extends_base_and_overrides_only_what_it_should():
    from pipeline.config import validate_config

    cfg = validate_config("pump01")
    assert cfg.optim.opacity_reset_interval == 60000  # pump01's one diff from base
    assert cfg.hidden.net_width == 128  # inherited from base, unchanged
    assert cfg.train.port == 6017
    assert cfg.train.expname == "multipleview/pump01"


def test_pump01_roundtrips_legacy_capture_yaml():
    from pipeline.config import validate_config
    from pipeline.config.loader import load_legacy_capture_yaml

    legacy = load_legacy_capture_yaml(
        REPO_ROOT / "omniverse-pipeline" / "omniverse_pipeline" / "capture_config_pump.yaml"
    )["capture"]
    cfg = validate_config("pump01").capture

    assert cfg.headless == legacy["headless"]
    assert cfg.scene.usd_path == legacy["scene"]["usd_path"]
    assert cfg.scene.semantic_roots == legacy["scene"]["semantic_roots"]
    assert cfg.rig.n_cameras == legacy["rig"]["n_cameras"]
    assert cfg.rig.radius_scale == legacy["rig"]["radius_scale"]
    assert cfg.capture.rt_subframes == legacy["capture"]["rt_subframes"]
    assert cfg.capture.num_frames == legacy["capture"]["num_frames"]
    assert cfg.output.capture_dir == legacy["output"]["capture_dir"]
    assert cfg.output.num_init_points == legacy["output"]["num_init_points"]
    assert cfg.lighting.dome_intensity == legacy["lighting"]["dome_intensity"]
    assert cfg.lighting.bg_base == legacy["lighting"]["bg_base"]


def test_pump01_roundtrips_legacy_hyperparams():
    from pipeline.config import validate_config
    from pipeline.config.loader import load_legacy_hyperparams

    legacy = load_legacy_hyperparams(REPO_ROOT / "core" / "arguments" / "multipleview" / "pump01.py")
    cfg = validate_config("pump01")

    legacy_hidden = {
        k: v for k, v in legacy["hidden"].items() if k not in _LEGACY_HIDDEN_DUPLICATE_KEYS
    }
    resolved_hidden = cfg.hidden.model_dump()
    for key, value in legacy_hidden.items():
        assert resolved_hidden[key] == value, f"hidden.{key} mismatch"

    resolved_optim = cfg.optim.model_dump()
    for key, value in legacy["optim"].items():
        assert resolved_optim[key] == value, f"optim.{key} mismatch"


def test_experiment_preset_overrides_segment_config():
    from pipeline.config import validate_config

    cfg = validate_config("pump01_segB_tuned")
    assert cfg.segment.impl == "rigid"
    assert cfg.segment.rigid.threshold_mult == 2.0
    assert cfg.segment.rigid.opacity_thresh == 0.2
    # inherited through pump01 <- base, untouched by the experiment preset
    assert cfg.optim.opacity_reset_interval == 60000


def test_unknown_preset_raises_filenotfounderror():
    from pipeline.config import validate_config

    with pytest.raises(FileNotFoundError):
        validate_config("does-not-exist")


def test_unknown_segment_impl_fails_fast():
    from pipeline.config import PipelineConfig

    with pytest.raises(Exception):
        PipelineConfig(segment={"impl": "not_a_real_impl"})


def test_missing_required_mbs_checkpoint_fails_fast():
    from pipeline.config import PipelineConfig

    with pytest.raises(Exception):
        PipelineConfig(segment={"impl": "mbs"})  # no checkpoint set


def test_bad_type_fails_fast():
    from pipeline.config import PipelineConfig

    with pytest.raises(Exception):
        PipelineConfig(optim={"iterations": "not-a-number"})


def test_unknown_top_level_field_fails_fast():
    from pipeline.config import PipelineConfig

    with pytest.raises(Exception):
        PipelineConfig(this_field_does_not_exist=True)


def test_unknown_amp_channel_fails_fast():
    from pipeline.config import PipelineConfig

    with pytest.raises(Exception):
        PipelineConfig(amp={"channels": {"not_a_channel": {"factor": 1.0}}})
