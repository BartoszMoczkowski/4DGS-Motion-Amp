# T35 — Regularizer schedule presets (R1)

**Status:** done
**Phase:** 12 / Phase 1 (config & loss quick wins)
**Depends on:** —
**Remedy:** R1

## Goal

Retune the temporal regularizers for small coherent periodic motion, as a
*schedule* rather than a constant.

## Change points

- Current values (verified): `plane_tv_weight = 2e-4`,
  `time_smoothness_weight = 1e-3`, `l1_time_planes = 1e-4` —
  `core/arguments/multipleview/lego_turbine_base.py:12-14` (and siblings),
  applied every fine-stage iteration at `core/train.py:278-285`
  (implementation `core/scene/gaussian_model.py:576-577`).
- Target: stock values while geometry stabilizes, then anneal after ~3 000
  fine-stage iterations to `time_smoothness_weight = 1e-4`,
  `l1_time_planes = 0`.
- Land as **named presets** under
  `orchestrator/pipeline/config/presets/lego_turbine_*.yaml` (one per ladder
  step), version-controlled — not shell history.

## Caution

Do not ablate R1 separately from R2 (T36) in early runs — a stronger motion
signal makes the regularizers bite harder; they land together in Phase 1 and
are untangled only in the post-Phase-2 single-factor ladder.

## Verification

- `uv run amp train lego_turbine_2_missing --dry-run` shows the resolved
  schedule values (`regularizer_anneal_iter: 3000`, `time_smoothness_weight_final: 0.0001`, `l1_time_planes_final: 0.0`).
- Sandbox test of the schedule logic (iteration-gated weight switch): `orchestrator/tests/test_regularizer_schedule.py` (15 passing tests).

## Deliverables

1. **Schedule implementation:**
   - `get_regularization_weights(iteration, hyper)` in `pipeline.stages.train` and `core/utils/general_utils.py`.
   - Loss site in `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py` updated to query scheduled weights at fine stage.
   - `compute_regulation` in `core/scene/gaussian_model.py` optimized to skip computing zero-weighted terms.
2. **Schema & Argument Group:**
   - `ModelHiddenParams` in `orchestrator/pipeline/config/models.py` and `core/arguments/__init__.py` extended with `regularizer_anneal_iter`, `regularizer_anneal_type`, `regularizer_anneal_end_iter`, `time_smoothness_weight_final`, `l1_time_planes_final`, `plane_tv_weight_final`.
3. **Preset YAMLs & Configs:**
   - Retuned presets: `lego_turbine_2_missing.yaml`, `lego_turbine_base.yaml`, `lego_turbine_5_missing.yaml`, `turbine_r1_missing_6.yaml`.
   - Explicit ladder step preset: `lego_turbine_2_missing_r1.yaml`.
   - Ablation baseline preset: `lego_turbine_2_missing_stock.yaml`.
   - Upstream reference configs: `core/arguments/multipleview/lego_turbine_*.py`.
4. **Sandbox tests:**
   - `orchestrator/tests/test_regularizer_schedule.py` covering step switch, linear mode, sentinels, partial overrides, preset resolution, and bridge serialization.

