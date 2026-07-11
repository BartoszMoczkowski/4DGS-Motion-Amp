# T02 — Unified config schema & presets

- Status: todo
- Phase: 0
- Depends on: T01
- Environment: host

## Goal
One typed (pydantic) config covering **all** stage settings, with layered presets, replacing the
three scattered sources. This is the highest-value, most tedious task — do it carefully.

## In scope
- Pydantic models per stage grouping the current knobs:
  - capture: from `omniverse_pipeline/capture_config_pump.yaml` (app/scene/rig/capture/output/lighting).
  - convert: `omni_to_4dgs.py` flags (name, target-radius, etc.).
  - train/render: `arguments/multipleview/<name>.py` params + `train_pump.sh` flags (port, expname).
  - segment: role + impl params (`segment_rigid.py`: k, min-size, threshold-mult, opacity-thresh;
    `mbs_infer.py`: working-set, checkpoint).
  - amp: `render_amp.py` / `ampUI.py` params (amp_factors, freq_cutoffs, method, low_vram_mode).
- Layered composition `base ← scene ← experiment`; a resolver that merges and validates.
- Presets: `base`, `pump01` (scene), plus one example `experiment` (e.g. `pump01_segB_tuned`).
- Loader that reads existing YAML/py so we can *migrate* current settings, not retype them.
- `role → impl` selection lives here (which segmentation backend, etc.).

## Out of scope
Executing anything; the registry itself (T04) — config only *names* impls.

## Deliverables
`pipeline/config/` models + `presets/` files; `validate_config(preset)` returns a resolved,
validated config object; a migration note mapping every old setting → new field.

## Acceptance criteria
- Round-trip: loading `pump01` reproduces the exact settings currently in `capture_config_pump.yaml`
  + `arguments/multipleview/pump01.py` + the `.sh` flags (documented field-by-field).
- Invalid config (bad type, unknown impl, missing required) fails fast with a clear error.

## Relevant existing files
`omniverse_pipeline/capture_config_pump.yaml`, `arguments/multipleview/*.py`,
`omniverse_pipeline/train_pump.sh`, `motion_seg/run.sh`, `render_amp.py`, `ampUI.py`.

## Notes / gotchas
`arguments/multipleview/*.py` are executable Python config (4DGS convention) — decide whether to
keep generating them for `train.py` from the unified config (likely yes: emit a temp `.py` the
wrapped script reads) rather than changing `train.py`. Document that bridge.
