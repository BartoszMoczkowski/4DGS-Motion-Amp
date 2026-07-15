# Migration: old scattered config -> unified `PipelineConfig` schema

Field-by-field map from the three old sources (`capture_config_pump.yaml`,
`arguments/multipleview/*.py`, and the CLI flags baked into `train_pump.sh` / `motion_seg/run.sh`
/ `render_amp.py` / `ampUI.py`) to `pipeline/config/models.py`. Presets in `presets/*.yaml` are
the actual migrated result; this document is the reasoning behind each mapping, and the notes on
upstream quirks that the new schema deliberately resolves or has to reconcile.

Everything below is validated by `tests/test_config.py`, which loads the *actual* legacy files via
`pipeline/config/loader.py` and asserts the resolved `pump01` preset matches them field-by-field —
this doc should never drift from that test.

## 1. `omniverse_pipeline/capture_config_pump.yaml` -> `PipelineConfig.capture`

One-to-one for every section except `app`:

| Old path | New path | Notes |
|---|---|---|
| `app.headless` | `capture.headless` | Flattened — `app` only ever held this one key. |
| `scene.usd_path` | `capture.scene.usd_path` | |
| `scene.subject_prim` | `capture.scene.subject_prim` | |
| `scene.semantic_roots` | `capture.scene.semantic_roots` | |
| `rig.*` | `capture.rig.*` | `layout`, `n_cameras`, `radius_scale`, `n_rings`, `min_elev_deg`, `max_elev_deg` — `world_up` is *not* a field (script auto-detects it from the USD stage's up-axis, never authored). |
| `capture.*` | `capture.capture.*` | `width`, `height`, `vfov_deg`, `num_frames`, `rt_subframes`, `near`, `far`. Yes, `capture.capture` — the YAML's `capture:` section became `CaptureFrameConfig` nested under the stage's own `CaptureConfig`, which is also named `capture`. |
| `output.*` | `capture.output.*` | `capture_dir`, `instance_segmentation`, `semantic_segmentation`, `num_init_points`. |
| `lighting.*` | `capture.lighting.*` | All 12 keys, unchanged. |

CLI-only overrides of `omni_capture.py` (`--usd`, `--out`, `--n-cameras`, `--headless`,
`--frames`) are **not** separate schema fields — they were always just alternate ways to set
`capture.scene.usd_path` / `capture.output.capture_dir` / `capture.rig.n_cameras` /
`capture.headless` / `capture.capture.num_frames` respectively. The stage wrapper (T09/T11) is
expected to pass the resolved config values as those flags rather than exposing a second,
parallel set of override fields.

## 2. `arguments/multipleview/pump01.py` (and `default.py`) -> `hidden` / `optim`

`ModelHiddenParams` dict -> `PipelineConfig.hidden` (`ModelHiddenParams` model), `OptimizationParams`
dict -> `PipelineConfig.optim` (`OptimizationParams` model). Key names are unchanged 1:1 **except**:

- `kplanes_config` (a nested dict in the legacy file) -> `hidden.kplanes_config`, now its own
  `KPlanesConfig` model (`grid_dimensions`, `input_coordinate_dim`, `output_coordinate_dim`,
  `resolution`) instead of an untyped dict.
- **`render_process` is dropped from `hidden`.** Both `default.py` and `pump01.py` redundantly
  restate `render_process=False` inside their `ModelHiddenParams` dict, but `render_process` is
  actually a `ModelParams` field (`arguments/__init__.py`), not a `ModelHiddenParams` one —
  `mmengine`'s merge just applies it by attribute name regardless of which dict it came from in
  the legacy file, so the duplication was harmless there. The new schema has exactly one field for
  it: `PipelineConfig.model.render_process` (default `False`, matching every legacy file). This is
  the one intentional de-duplication in the migration — see `tests/test_config.py`'s
  `_LEGACY_HIDDEN_DUPLICATE_KEYS` for where the round-trip test excludes it accordingly.

**Layering `base <- pump01` for this section specifically:** `arguments/multipleview/default.py`
*already* overrides several `arguments/__init__.py` class defaults (finer time resolution, wider
net, lighter TV regularization, a 15k-iteration run instead of 30k). The `base.yaml` preset
captures those `default.py` values, not the raw `arguments/__init__.py` ones — every
`multipleview/*` scene preset should extend `base`, matching how every `arguments/multipleview/*`
file today is itself a diff against `default.py`, not against `arguments/__init__.py` directly.

`pump01.py`'s only actual diff from `default.py` is `optim.opacity_reset_interval: 60000` (raised
from the implicit class default of `3000` to avoid a coarse/fine opacity-reset collision that
reliably NaN's the loss — see the comment in `pump01.py` and in `presets/pump01.yaml`). That is the
only key `presets/pump01.yaml` sets under `optim:`/`hidden:`.

Everything in `arguments/__init__.py`'s `ModelParams`/`PipelineParams`/`OptimizationParams` classes
that neither `default.py` nor `pump01.py` overrides keeps its class default via the corresponding
pydantic model's own default (`models.py`'s `ModelParams`/`PipelineParams`/`OptimizationParams`) —
nothing was retyped, just left unset in the preset so the model default applies.

## 3. `train_pump.sh` -> `PipelineConfig.model` / `PipelineConfig.train` / `PipelineConfig.convert`

| Old | New | Notes |
|---|---|---|
| `NAME` (positional, default `pump01`) | `convert.name`, and the derived `model.source_path` / `model.model_path` | The script computes `data/multipleview/$NAME` and `output/multipleview/$NAME` from `NAME`; the preset pins these directly in `model.source_path`/`model.model_path` until path derivation from `convert.name` is wired into the DAG's artifact resolution (T03/T05/T06) — at that point these two fields become derivable rather than hand-set. |
| `PORT` (positional, default `6017`) | `train.port` | |
| `--expname "multipleview/$NAME"` | `train.expname` | |
| `--configs "arguments/multipleview/$NAME.py"` | *(none — superseded)* | The old bridge was: generate/point at a per-scene `arguments/*.py` file for `train.py`'s `--configs` merge. The new schema **is** that merged result; a stage wrapper (T09) can still emit a temp `.py`/mmengine-config-shaped file from the resolved `PipelineConfig` for `train.py`/`render.py`/etc. to consume unchanged (see "Bridge to `--configs`" below), so `train.py` itself needs no code change. |

## 4. `motion_seg/run.sh` -> `PipelineConfig.seg_extract` / `.segment` / `.seg_eval`

| Old | New |
|---|---|
| `extract_trajectories.py --model_path ... --configs ...` | `seg_extract.configs`, `seg_extract.iteration`, `seg_extract.n_times`, `seg_extract.out` (`model_path` is derived, not a config field — same "derived path" note as above) |
| `segment_rigid.py --trajectories/--out/--k/--min-size/--threshold-mult/--opacity-thresh` | `segment.rigid.*` (`k`, `min_size`, `threshold_mult`, `opacity_thresh`, `preview_png`); `--trajectories`/`--out` are run-artifact paths, not config (T03) |
| `evaluate_segmentation.py --pred/--gt/--drop-floaters/--recolored-ply/--comparison-png/--top-n` | `seg_eval.*` (`drop_floaters`, `recolored_ply`, `comparison_png`, `top_n`); `--pred`/`--gt` are artifact paths |
| The script's own two-implementation choice (rigid vs. `mbs_infer.py`, not actually in `run.sh` but documented alongside it) | `segment.impl: "rigid" \| "mbs"` — this is the schema's role -> impl selection point; `segment.mbs.*` holds Option-A's own knobs (`checkpoint`, `n_points` — the "working-set size", `n_views`, `n_sub`, `opacity_thresh`, `alpha`, `seed`) |

`presets/pump01_segB_tuned.yaml` demonstrates exactly the re-tune-without-re-extracting use case
from `run.sh`'s header comment (`./motion_seg/run.sh pump01 --threshold-mult 2 --opacity-thresh
0.2`): an experiment preset that extends `pump01` and overrides only `segment.rigid.threshold_mult`
/ `segment.rigid.opacity_thresh`.

**Missing-required fails fast:** `segment.impl == "mbs"` without `segment.mbs.checkpoint` set is
rejected by a model validator (`SegmentConfig._check_impl_ready`) — mirrors `mbs_infer.py`'s own
`--checkpoint` being a required CLI flag with no default.

## 5. `render_amp.py` / `ampUI.py` -> `PipelineConfig.amp`

| Old | New |
|---|---|
| `--amp_factors` (list, positionally indexed) | `amp.channels[channel].factor`, keyed by channel name instead of position |
| `--freq_low` / `--freq_high` (lists, positionally indexed, then zipped) | `amp.channels[channel].freq_low` / `.freq_high` |
| `--method` (`eulerian`/`eulerian_abs`/`eulerian_mod`/`eulerian_abs_mod`) | `amp.method`, same four values (canonical) |
| `--low_vram` | `amp.low_vram_mode` |
| `--frozen_cam` | `amp.frozen_cam` |
| `--video_path` / `--video_fps` | `amp.video_path` / `amp.video_fps` |
| `--skip_train`/`--skip_test`/`--skip_video`/`--quiet` | `amp.skip_train`/`.skip_test`/`.skip_video`/`.quiet` |
| `--iteration` / `--configs` | `amp.iteration` / `amp.configs` |

**Positional -> named channels.** The 8-channel order `[pos3d, pos2d, rotation, scale, opacity,
SHs, color, cov3D]` (confirmed against `render_amp.py`'s `render_set_amp` construction) is
preserved as `AMP_CHANNELS` in `models.py`, but the schema addresses each channel **by name**
(`amp.channels["opacity"].factor`) instead of by position in a flat list — eliminates an entire
class of "which index was that again" bugs the old CLI/UI both had. A stage wrapper (T09) rebuilds
the old positional lists from `AMP_CHANNELS` order when invoking `render_amp.py` unchanged.

**`--amp_factors` was `type=int` in the CLI despite factors being conceptually float** (an existing
script quirk — `2.0` would fail/truncate under the old `type=int`). The schema's
`AmpChannelConfig.factor` is a proper `float`; the stage wrapper is responsible for whatever
int/float coercion `render_amp.py`'s argparse still needs until that's fixed upstream (out of scope
for T02 — config only).

**`ampUI.py`'s method labels don't match `render_amp.py`'s CLI strings.** The Streamlit UI offers
`"base"` / `"base segmented"` / `"abs"` / `"abs segmented"`, which map to the *same* four
underlying functions as the CLI's `"eulerian"` / `"eulerian_mod"` / `"eulerian_abs"` /
`"eulerian_abs_mod"`, just with different labels. The schema standardizes on the CLI's names
(`amp.method`); `AMP_METHOD_ALIASES` in `models.py` is the reconciliation table a future UI
(T15) or wrapper uses to translate a picked Streamlit label into a canonical `amp.method` value:

```python
AMP_METHOD_ALIASES = {
    "base": "eulerian",
    "base segmented": "eulerian_mod",
    "abs": "eulerian_abs",
    "abs segmented": "eulerian_abs_mod",
}
```

`ampUI.py`'s hardcoded, not-exposed constants (`fps = 20` for the render loop — separate from the
video-encode `video_fps` — and the `frame[:, 200, :]` space-time visualization slice index) are
**not** schema fields; they're UI-internal rendering details, not pipeline configuration.

## 6. `split_mesh.py` / `add_motion.py` -> `PipelineConfig.prep_split` / `.prep_motion`

Straight 1:1 flag copies (`prep_split.group/min_faces/preview/no_usd/weld_tol`,
`prep_motion.group/num_frames/fps/trans_amp_mm/rot_surface_mm/rot_deg_max/freq/groups/exclude/
seed/plot`). `--in`/`--out` are artifact paths (T03), not config. `trans_amp_mm`/`rot_surface_mm`
stay in millimetres in the schema, matching how they're authored on the CLI — the script's
internal conversion to USD stage units via `metersPerUnit` remains a runtime detail.

`prep_motion.groups` (0 = each part independently animated -> finest possible GT segmentation
labels; K>0 = K shared-motion clusters) is the single knob controlling ground-truth segmentation
granularity, and is the schema's most direct mapping of a "new idea" experiment knob — swapping it
per-preset requires zero code changes, exactly per T02's goal.

## 7. Bridge to `train.py`/`render.py`/etc.'s own `--configs` mmengine files

`train.py` does not (and, per the "wrap don't rewrite" ground rule in `planning/INSTRUCTIONS.md`,
should not) read `PipelineConfig` directly — it only understands `--configs <path-to-a-python-file
with ModelHiddenParams=dict(...)/OptimizationParams=dict(...)>` merged via
`utils.params_utils.merge_hparams`. The stage wrapper that will invoke `train.py`/`render.py`/
`render_amp.py`/`extract_trajectories.py` (T09) is expected to serialize the resolved
`PipelineConfig.hidden`/`.optim` back out to a temp file in exactly that shape before invoking the
script — i.e. the inverse of `pipeline.config.loader.load_legacy_hyperparams`. That
serializer is out of scope for T02 (config schema only); this doc records the expectation so T09
doesn't have to re-derive it.

## 8. Reusing the migration for a future scene

`pipeline/config/loader.py` provides `load_legacy_capture_yaml(path)` and
`load_legacy_hyperparams(path)`, the same functions `tests/test_config.py` uses to verify `pump01`.
Migrating a second scene's legacy config is: run both loaders against its
`capture_config_<scene>.yaml` and `arguments/multipleview/<scene>.py`, `merge_legacy_sources(...)`
the fragments, add an `extends: base` (or `extends: <existing-scene>`) key, and save it as
`presets/<scene>.yaml` — not a repeat of this document's manual field-by-field port.
