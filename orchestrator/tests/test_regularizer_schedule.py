"""Tests for T35 (Regularizer schedule presets - R1).

Verifies:
1. Iteration-gated weight switch logic (step and linear schedules, edge cases, partial overrides).
2. Compatibility with dict, Namespace, and ModelHiddenParams input objects.
3. Named presets under `orchestrator/pipeline/config/presets/` resolve the retuned schedule.
4. Bridge file generation carries the regularizer schedule parameters into the container.
"""

from __future__ import annotations

import types
from pathlib import Path
import pytest

from pipeline.config import validate_config, ModelHiddenParams
from pipeline.config.bridge import render_bridge_source
from pipeline.stages.train import get_regularization_weights


class DummyHyper:
    """Mock object mimicking argparse / ParamGroup hyperparameter Namespace."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_schedule_inactive_when_anneal_iter_zero():
    hyper = DummyHyper(
        time_smoothness_weight=0.001,
        l1_time_planes=0.0001,
        plane_tv_weight=0.0002,
        regularizer_anneal_iter=0,
        time_smoothness_weight_final=0.0001,
        l1_time_planes_final=0.0,
    )
    for it in [1, 500, 3000, 15000]:
        ts, l1, ptv = get_regularization_weights(it, hyper)
        assert ts == pytest.approx(0.001)
        assert l1 == pytest.approx(0.0001)
        assert ptv == pytest.approx(0.0002)


def test_schedule_inactive_when_anneal_iter_negative():
    hyper = {"time_smoothness_weight": 0.001, "l1_time_planes": 0.0001, "plane_tv_weight": 0.0002, "regularizer_anneal_iter": -1}
    ts, l1, ptv = get_regularization_weights(5000, hyper)
    assert ts == pytest.approx(0.001)
    assert l1 == pytest.approx(0.0001)
    assert ptv == pytest.approx(0.0002)


def test_iteration_gated_weight_switch():
    hyper = {
        "time_smoothness_weight": 0.001,
        "l1_time_planes": 0.0001,
        "plane_tv_weight": 0.0002,
        "regularizer_anneal_iter": 3000,
        "regularizer_anneal_type": "step",
        "time_smoothness_weight_final": 0.0001,
        "l1_time_planes_final": 0.0,
        "plane_tv_weight_final": None,
    }

    # Before gate (< 3000): stock values preserved while geometry stabilizes
    for it in [1, 100, 1500, 2999]:
        ts, l1, ptv = get_regularization_weights(it, hyper)
        assert ts == pytest.approx(0.001), f"failed at iter {it}"
        assert l1 == pytest.approx(0.0001), f"failed at iter {it}"
        assert ptv == pytest.approx(0.0002), f"failed at iter {it}"

    # At exact gate (3000): anneals to target values
    ts, l1, ptv = get_regularization_weights(3000, hyper)
    assert ts == pytest.approx(0.0001)
    assert l1 == 0.0
    assert ptv == pytest.approx(0.0002)

    # After gate (> 3000): target values persist throughout fine training
    for it in [3001, 5000, 10000, 15000]:
        ts, l1, ptv = get_regularization_weights(it, hyper)
        assert ts == pytest.approx(0.0001), f"failed at iter {it}"
        assert l1 == 0.0, f"failed at iter {it}"
        assert ptv == pytest.approx(0.0002), f"failed at iter {it}"


def test_partial_overrides_and_sentinels():
    # Only time_smoothness_weight_final specified; l1_time_planes_final is -1 (sentinel)
    hyper = DummyHyper(
        time_smoothness_weight=0.001,
        l1_time_planes=0.0001,
        plane_tv_weight=0.0002,
        regularizer_anneal_iter=3000,
        time_smoothness_weight_final=0.0002,
        l1_time_planes_final=-1.0,  # negative sentinel should keep initial
        plane_tv_weight_final=None,
    )
    ts, l1, ptv = get_regularization_weights(4000, hyper)
    assert ts == pytest.approx(0.0002)
    assert l1 == pytest.approx(0.0001)  # kept initial
    assert ptv == pytest.approx(0.0002)  # kept initial


def test_linear_anneal_mode():
    hyper = {
        "time_smoothness_weight": 0.001,
        "l1_time_planes": 0.0001,
        "plane_tv_weight": 0.0002,
        "regularizer_anneal_iter": 3000,
        "regularizer_anneal_type": "linear",
        "regularizer_anneal_end_iter": 5000,
        "time_smoothness_weight_final": 0.0001,
        "l1_time_planes_final": 0.0,
    }

    # Before start
    ts, l1, ptv = get_regularization_weights(2000, hyper)
    assert ts == pytest.approx(0.001)
    assert l1 == pytest.approx(0.0001)

    # At start of anneal
    ts, l1, ptv = get_regularization_weights(3000, hyper)
    assert ts == pytest.approx(0.001)
    assert l1 == pytest.approx(0.0001)

    # Midpoint: iter 4000 (alpha = 0.5)
    ts, l1, ptv = get_regularization_weights(4000, hyper)
    assert ts == pytest.approx(0.5 * 0.001 + 0.5 * 0.0001)
    assert l1 == pytest.approx(0.5 * 0.0001 + 0.5 * 0.0)

    # End and beyond
    ts, l1, ptv = get_regularization_weights(5000, hyper)
    assert ts == pytest.approx(0.0001)
    assert l1 == pytest.approx(0.0)

    ts, l1, ptv = get_regularization_weights(6000, hyper)
    assert ts == pytest.approx(0.0001)
    assert l1 == pytest.approx(0.0)


def test_works_with_model_hidden_params_instance():
    params = ModelHiddenParams(
        plane_tv_weight=0.0002,
        time_smoothness_weight=0.001,
        l1_time_planes=0.0001,
        regularizer_anneal_iter=3000,
        time_smoothness_weight_final=0.0001,
        l1_time_planes_final=0.0,
    )
    assert get_regularization_weights(100, params) == (0.001, 0.0001, 0.0002)
    assert get_regularization_weights(3000, params) == (0.0001, 0.0, 0.0002)


@pytest.mark.parametrize(
    "preset_name",
    [
        "lego_turbine_2_missing",
        "lego_turbine_base",
        "lego_turbine_5_missing",
        "turbine_r1_missing_6",
        "lego_turbine_2_missing_r1",
    ],
)
def test_turbine_presets_resolve_schedule_r1(preset_name: str):
    cfg = validate_config(preset_name)
    assert cfg.hidden.plane_tv_weight == pytest.approx(0.0002)
    assert cfg.hidden.time_smoothness_weight == pytest.approx(0.001)
    assert cfg.hidden.l1_time_planes == pytest.approx(0.0001)
    assert cfg.hidden.regularizer_anneal_iter == 3000
    assert cfg.hidden.time_smoothness_weight_final == pytest.approx(0.0001)
    assert cfg.hidden.l1_time_planes_final == pytest.approx(0.0)


def test_lego_turbine_2_missing_stock_has_disabled_schedule():
    cfg = validate_config("lego_turbine_2_missing_stock")
    assert cfg.hidden.regularizer_anneal_iter == 0
    assert cfg.hidden.time_smoothness_weight == pytest.approx(0.001)
    assert cfg.hidden.l1_time_planes == pytest.approx(0.0001)


def test_base_preset_defaults_schedule_disabled():
    cfg = validate_config("base")
    assert cfg.hidden.regularizer_anneal_iter == 0
    assert cfg.hidden.time_smoothness_weight_final is None
    assert cfg.hidden.l1_time_planes_final is None


def test_bridge_rendering_includes_regularizer_schedule():
    cfg = validate_config("lego_turbine_2_missing")
    bridge_src = render_bridge_source(cfg.model_dump())

    assert "ModelHiddenParams = {" in bridge_src
    # Execute the bridge Python source in an isolated namespace to verify it parses and merges
    ns: dict = {}
    exec(bridge_src, ns)
    assert "ModelHiddenParams" in ns
    hp = ns["ModelHiddenParams"]
    assert hp["regularizer_anneal_iter"] == 3000
    assert hp["time_smoothness_weight_final"] == pytest.approx(0.0001)
    assert hp["l1_time_planes_final"] == pytest.approx(0.0)
    assert hp["plane_tv_weight"] == pytest.approx(0.0002)
