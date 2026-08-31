# Master Technical Post-Mortem & Standardized Resolutions

This report compiles all technical issues, root causes, operational friction points, and permanent architectural resolutions identified during synthetic scene generation, Isaac Sim multi-view capture, and dataset conversion.

---

## 1. Vulkan VRAM Exhaustion & High-Resolution Rendering ($1600\times900$ / $1920\times1080$)

### ❌ Problem
- **Symptom**: Isaac Sim crashed on startup with exit code `3221225477` (Vulkan TDR / `ERROR_DEVICE_LOST`).
- **Root Cause**: `gen_spinning_cubes.py` generated capture configs with $1600 \times 900$ resolution. `omni_capture.py` defined 16 cameras and attached **all 16 render products simultaneously** to a single Replicator `BasicWriter`. Rendering 16 simultaneous $1600\times900$ framebuffers exceeded the GPU memory bandwidth and 2-second Windows WDDM driver TDR timeout.

### ✅ Permanent Resolution
1. **Camera Chunking / Batching in `omni_capture.py`**:
   - Instead of attaching all 16 cameras at once, render cameras in **chunks of 4**:
     ```python
     chunk_size = 4
     for i in range(0, len(cameras), chunk_size):
         batch_rp = [rep.create.render_product(c, (1920, 1080)) for c in cameras[i:i+chunk_size]]
         writer.attach(batch_rp)
         # Render animation timecodes for this batch
         writer.detach()
     ```
   - **Benefit**: VRAM footprint drops by 75%, allowing **Full HD ($1920 \times 1080$)** captures without GPU driver resets.
2. **Disable Non-Essential Annotators**:
   - Set `semantic_segmentation: false` and `colorize_instance_segmentation: false` in YAML capture configs. Unused SDGPipeline graph nodes are omitted, freeing ~30% VRAM overhead.
3. **Windows TDR Delay (Optional Machine Setup)**:
   - Set registry key `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` = `10` (decimal) to allow complex path tracing subframes to complete.

---

## 2. Vulkan Driver Race Condition Between Sequential Processes

### ❌ Problem
- **Symptom**: When `run_cubes_capture.py` launched `python.bat` in a rapid Python loop, the second scene crashed immediately on startup.
- **Root Cause**: Windows WDDM driver requires ~3–5 seconds after `kit.exe` terminates to tear down Vulkan context buffers and free VRAM. Immediate re-launching triggered driver initialization failures.

### ✅ Permanent Resolution
- Insert an explicit **5-second GPU cooldown pause** (`time.sleep(5.0)`) between subprocess invocations in `run_cubes_capture.py`.

---

## 3. Subprocess Exit Code Masking in Isaac Sim `python.bat`

### ❌ Problem
- **Symptom**: `run_cubes_capture.py` proceeded to launch dataset conversion on failed captures, crashing later with `FileNotFoundError: cam01`.
- **Root Cause**: Isaac Sim's native Windows `python.bat` wrapper returns exit code `0` even when internal `kit.exe` suffers a fatal C++ exception or breakpad crash during shutdown. `subprocess.run(check=True)` evaluated the command as successful.

### ✅ Permanent Resolution
- Add **Post-Capture Output Validation** in `run_cubes_capture.py`:
  ```python
  def validate_capture(capture_dir: Path) -> bool:
      cam01_rgb = capture_dir / "cam01" / "rgb"
      return cam01_rgb.is_dir() and len(list(cam01_rgb.glob("*.png"))) >= 120
  ```
  If validation fails, the batch runner logs an explicit error and halts before attempting conversion.

---

## 4. Stale Artifact Globs & Cascade Failures

### ❌ Problem
- **Symptom**: `run_cubes_capture.py` picked up old test files like `capture_config_cubes_k2_ring_k2.yaml` and failed during dataset conversion.
- **Root Cause**: Overly permissive glob matching (`capture_config_cubes_k2*.yaml`) matched duplicate, corrupted test YAML files.

### ✅ Permanent Resolution
- **Manifest-Driven Execution**: Read scene names explicitly from `dataset_manifest.json` instead of loose file globs.

---

## 5. API Signature Incompatibilities During In-Flight Fixes

### ❌ Problem
- **Symptom**: `AttributeError: 'UsdContext' object has no attribute 'is_stage_loading'` crashed the pipeline.
- **Root Cause**: Adding unverified API assumptions (`while omni.usd.get_context().is_stage_loading():`) without verifying object methods in the target version of Isaac Sim (6.0.1 / Kit 106+).

### ✅ Permanent Resolution
- Use verified USD context methods (`omni.usd.get_context().open_stage()`) and wait for payload resolution using standard application updates (`simulation_app.update()`).

---

## 6. Code Duplication & Maintenance Drift

### ❌ Problem
- **Symptom**: Fixes applied to `omniverse-pipeline/omniverse_pipeline/omni_capture.py` were missing in `orchestrator/pipeline/vendored/isaac/omni_capture.py`.
- **Root Cause**: Maintaining identical duplicate script files across packages creates dual-maintenance overhead and code drift.

### ✅ Permanent Resolution
- **Single Importable Source of Truth**: Refactor orchestrator stages to import directly from `omniverse_pipeline.omni_capture` rather than maintaining vendored file duplicates.

---

## 7. Ad-Hoc Inline Progress Polling

### ❌ Problem
- **Symptom**: Repeated execution of 15+ ad-hoc `python -c "from pathlib import Path..."` commands to count PNG files across 16 camera directories.
- **Root Cause**: Heavy stdout buffering in headless Isaac Sim obscured live progress updates, forcing external disk checking.

### ✅ Permanent Resolution
1. **Dedicated Progress CLI Tool (`scene-gen/check_capture_progress.py`)**:
   - Standardize external progress checking into a single CLI script that outputs camera counts, frame completion percentages, and GT artifact status.
2. **Structured `status.json` Runtime File**:
   - `omni_capture.py` writes a lightweight `status.json` file inside `capture_dir` every 10 frames:
     ```json
     {
       "scene_name": "cubes_k4_both",
       "current_frame": 90,
       "total_frames": 120,
       "status": "rendering"
     }
     ```
