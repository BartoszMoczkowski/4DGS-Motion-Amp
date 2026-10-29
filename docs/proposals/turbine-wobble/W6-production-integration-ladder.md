# W6 (T54) — Production integration, streaming render, and single-factor experiment ladder

**Status:** todo  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W1–W5 (T48–T53)  
**Experiments:** Full single-factor ladder execution (`E1a` → `E1b` → `E2` → `E3` → `E4`)  
**Gate:** Training, offline render, `seg_extract`, and `amp` outputs are numerically equivalent at arbitrary physical timestamps; long sequences stream with bounded RAM/VRAM (no OOM); every ladder run is single-factor and fully provenance-hashed in its manifest.

---

## 1. Objective

Ensure complete production parity across training (`core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`), offline rendering (`core/render.py` and `vendored/cuda/render.py`), trajectory extraction (`seg_extract.py`), and motion amplification (`render_amp.py` / `vendored/cuda/amp.py`), eliminate multi-camera frame-accumulation OOMs via streaming I/O, and execute the frozen single-factor experiment ladder (`E1a`–`E4`).

## 2. Implementation Steps

1. **Unified deformation dispatch across all 4 pipeline consumers:**
   - Verify and enforce a single shared deformation path across:
     1. `core/gaussian_renderer/__init__.py:render()`
     2. `core/utils/render_utils.py:get_state_at_time()` (used by `seg_extract.py` and `export_perframe_3DGS.py`)
     3. `core/motion_amp/renderer.py:render_mod()` (used by `render_amp.py` and `vendored/cuda/amp.py`)
     4. Both `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`
2. **Bounded-memory streaming render & export:**
   - Refactor `core/render.py` and `orchestrator/pipeline/vendored/cuda/render.py` so views are rendered, written to disk, and streamed to MP4 one frame at a time without holding a full list of 6 MP images/tensors in RAM or VRAM (fixing the 600-frame train-split exit-137 OOM).
3. **Complete manifest provenance:**
   - Record in every run's `manifest.json`:
     - `git_revision` and dirty-tree status
     - `resolved_config`
     - `offsets_json_hash` (SHA-256)
     - `weight_maps_hash` (SHA-256, when W3 is active)
     - `warmstart_checkpoint_hash` (SHA-256, when W4/W5 warm-start is used)
     - `component_structural_history` (accepted/rejected split/merge ledger)
     - `random_seed`
4. **Execute the frozen single-factor experiment ladder (real GPU):**
   - **Comparison A (`E1a` vs `E1b`):** Fixed sidecar timestamps vs guarded learnable offsets (native HexPlane everywhere, soft weights OFF, components OFF, trajectory loss OFF).
   - **Comparison B (`Best A` vs `E2`):** Best timing config vs best timing + W3 continuous soft loss weights.
   - **Comparison C (`Best B` vs `E3`):** Best W3 config vs best W3 + W4 fixed-$K$ soft motion components ($K=1$, $f_0 = 5.691\text{ Hz}$, 1 harmonic, persistent HexPlane residual).
   - **Comparison D (`Best C` vs `E4`):** Best W4 config vs best W4 + W5 reversible split/merge.

## 3. Verification & Tests

- **Cross-tool deformation equivalence test:** Given a trained checkpoint with learned offsets and soft components, verify at 5 distinct timestamps that training `render()`, offline `render.py`, `get_state_at_time()`, and `render_mod()` produce identical deformed 3D positions and rotations.
- **Streaming memory ceiling test:** Render a multi-hundred-frame sequence and assert host RAM and CUDA allocated memory remain bounded ($O(1)$ in frame count).
- **Manifest provenance test:** Verify all required SHA-256 hashes and metadata fields are populated on run completion.

## 4. Deliverables & Stop Conditions

- **Deliverables:** Streaming render/export implementation, parity & memory tests (`orchestrator/tests/test_w6_production_parity.py`), single-factor comparison tables with W2 metrics and manifest links, and `PASS` / `FAIL` / `INCONCLUSIVE` verdicts for each gate.
- **Stop conditions:** Any mismatch between training-time and offline deformation, any frame-accumulation OOM, or any multi-factor config change between adjacent ladder steps blocks progression.

