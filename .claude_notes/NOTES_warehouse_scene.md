# Warehouse scene composition (pump in industrial factory)

Date: 2026-07-27. Goal: the pump capture should happen inside an industrial warehouse
background instead of the generated dark-grid dome texture.

## What exists now

- **`omniverse-pipeline/omniverse_pipeline/compose_scene.py`** (new reference script,
  plain python + usd-core, `--selftest` passes): composites an animated subject USD into
  a static environment USD via pure references (no flattening). Subject lands at
  `/World/<group>` with optional translate/rotateX/scale; env under `--env-root`.
  `--src-prim` allows renaming the subject in the output (used to finally drop the
  `CONJUNTO_BOMBAS` name). Animation and timeline (0–59 @ 24 fps) ride through the
  reference untouched — verified programmatically.
- **`omniverse-pipeline/data/scenes/pump_warehouse.usd`** (new, in-repo): the composed
  scene. `/World/Factory` → references
  `Q:/Omniverse/assets/physAI_start/Factory/Factory.usd` (36×61 m hall, 12.8 m tall,
  Z-up, cm, **18 sphere ceiling lights**); `/World/pump` → references
  `Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd` `/World/CONJUNTO_BOMBAS`
  (all 107 parts, animated). Pump placement mirrors `physAI_start/SceneAssembly.usd`,
  which places this exact pump in this exact factory: translate `(-130.29, -437.17, 0.35)`,
  scale `0.2` → pump world bbox ≈ 2.4 × 5.1 × 4.1 m, sitting on the floor (z ≈ 0.35 cm).

  Note: the file only holds *references* to the two Q: assets — it is tiny and breaks if
  Q: is unmounted. That's deliberate (animation stays live; re-running add_motion.py
  needs no re-compose).

## Key facts discovered while exploring

- `SceneAssembly.usd` (physAI_start) already combines factory + this pump — its pump
  transform is the artist-intended placement, which is why we copied it exactly.
- Units/axes mess: the pump USD claims `metersPerUnit=0.01` (cm) but its raw extents
  (1195×2542×2039) only make sense as mm-ish; SceneAssembly's scale 0.2 makes it a
  plausible ~5 m machine skid. Factory is true cm, Z-up; pump USD claims Y-up but
  SceneAssembly applies **no rotation** (orient identity, rotateX 0) — we replicated
  that. **Open question:** whether the pump orientation reads correctly in the factory
  still needs a visual check in Isaac Sim (first capture will show it immediately).
- `omni_capture.py` specifics that make the composition slot in cleanly:
  - `scene.subject_prim` scopes BOTH the rig bbox and the init point cloud sampling
    (`_sample_pointcloud` uses the same root). Setting `subject_prim: "/World/pump"`
    keeps cameras and `points3D_gt.ply` pump-only, factory excluded.
  - `scene.semantic_roots` (`["/World/pump"]` after rename) tags only pump parts →
    factory gets no instance labels. Good.
  - `_setup_lighting` skips adding the dome/distant pair if the stage already has lights
    (unless `lighting.force`). The factory ships 18 sphere lights, so with the current
    preset (`force: false`) the capture uses factory lighting and the generated grid dome
    disappears — the factory shell itself becomes the visible background. **Open
    question:** exposure may need tuning (`force: true` adds our dome+distant on top).
- Camera geometry still works: subject bbox radius ≈ 3.6 m → `radius_scale 2.5` puts the
  10 cameras ~9 m out, well inside the hall (walls at ±17–30 m). near=90 / far=7000 cm
  still fine.

## What was deliberately NOT done (deferred, per user)

- **Orchestrator integration.** The plan was a `prep_compose` DAG stage (animated_mesh →
  scene_mesh, vendored `compose_scene.py`, `capture.isaac` input renamed, preset
  `prep_compose` section defaulting to pass-through for other presets). User decided this
  is not needed for now — capture can be pointed at the composed scene manually.
- **Config updates.** `capture_config_pump.yaml` / `pump01.yaml` still point at the bare
  animated USD. To capture the warehouse scene with the reference script, set:
  `scene.usd_path: <repo>/omniverse-pipeline/data/scenes/pump_warehouse.usd`,
  `scene.subject_prim: "/World/pump"`, `scene.semantic_roots: ["/World/pump"]`.
- **Renaming everywhere.** The rename to `pump` applies only inside the new composed
  scene (source asset, split/add_motion `--group`, and all configs still say
  `CONJUNTO_BOMBAS`).

## Gotcha encountered

Git Bash (MSYS) mangles CLI args starting with `/` (e.g. `--env-root /World/Factory`
became garbage and `DefinePrim` failed with "Path must be an absolute path: <>"). Calling
`compose()` from `python -c` avoids it; in the orchestrator/container (Linux) it's a
non-issue.
