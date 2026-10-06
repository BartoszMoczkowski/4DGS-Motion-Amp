# Correctness review — motion-amplification path (2026-10-05)

Scope: `core/render_amp.py`, `core/motion_amp/renderer.py`, `amp-ui/amp_ui/{ampUI.py,run_renders_auto.py,cameras.py}`, cross-checked against `core/gaussian_renderer/`, `core/scene/`, `core/arguments/`, and the `depth-diff-gaussian-rasterization` CUDA fork. Three independent reviews, deduplicated and merged here.

The vendored copy `orchestrator/pipeline/vendored/cuda/amp.py` is byte-identical in amplification logic, so **every `render_amp.py` bug below also exists in the orchestrator's vendored stage.**

---

## CRITICAL / HIGH — silently wrong results

### 1. FFT wrap-around sample contaminates every frame (`eulerian` / `eulerian_mod`)
`core/render_amp.py:135, 213` — `values_delta = values_tensor.roll(-1,-1) - values_tensor` produces `delta[T-1] = v[0] - v[T-1]`, a spurious jump discontinuity. Fed into `rfft`, it spreads broadband energy across all bins; no band-pass filter can remove it. Scales with `|v[0]-v[T-1]|` — even "periodic" captures rarely have exactly matching endpoints.
**Fix:** zero `delta[..., -1]` before the FFT.

### 2. `eulerian*` amplifies velocity, not displacement — effective gain ≪ `a`
`core/render_amp.py:155-157` — after re-roll, `out[t] = v[t] + (a-1)·(v[t]-v[t-1])_filtered`. For a sinusoid, gain ≈ `1 + (a-1)·ω` (ω per frame). A 60-frame, one-period sequence (the pump01 regime) at `a=10` yields only ~2× displacement amplification, plus a phase shift. The `eulerian_abs*` variants are the defensible Eulerian method. Identity at `a=1` passes, which is likely why this went unnoticed.
**Decision needed:** which method the thesis claims; document or reimplement.

### 3. In-place mutation: the Streamlit UI re-amplifies on every render click
`amp-ui/amp_ui/ampUI.py:52-73` + `core/render_amp.py:164` — `amplify_frame_data_*` mutate `values_list` in place; `AMPUI.render()` passes `self.values` directly. After the first click, `self.values` holds amplified data; every subsequent click amplifies the previous output again. Interactive tuning comparisons after the first click are all corrupted. `run_renders_auto.py` avoids this only by reloading the config every repeat.
**Fix:** deep-copy before amplification, or reload per render.

### 4. Benchmark timing is systematically biased across VRAM modes (no GPU sync)
`amp-ui/amp_ui/run_renders_auto.py:52-64` — no `torch.cuda.synchronize()` before stopping the timer. The amplify functions are async CUDA; non-low-VRAM timing misses the kernel tail, while low-VRAM mode's `.cpu()` forces near-full sync. The `vram_modes=[False,True]` comparison — the harness's central purpose — is not measuring the same thing in both arms.
**Fix:** `torch.cuda.synchronize()` before `time.time_ns()`.

### 5. Amplified quaternions never re-normalized; the CUDA fork doesn't normalize either
`core/motion_amp/renderer.py:99` returns normalized quats, but amplification perturbs them and `submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.cu:127` has normalization **commented out**. Non-unit quaternions → non-orthonormal "rotation" matrices → silently wrong Gaussian shape/orientation. `Sigma` stays PSD so nothing crashes.
**Fix:** re-normalize quaternions after amplification (in the amplify functions or before rasterize).

### 6. UI channel labels are swapped: "rotation" drives scales and vice versa
`amp-ui/amp_ui/ampUI.py:96` lists `["pos3d","pos2d","rotation","scale","opacity","SHs","color","cov3D"]`, but `values_list` order (`render_amp.py:51-60`) is `[means3D, means2D, scales, rotations, ...]`. Index 2 ↔ 3 swapped. A user enabling "scale" amplification is actually amplifying rotations — and hits bug 5 without knowing it.
**Fix:** correct the label order.

### 7. Frame-0 reset indexes the wrong axis for SHs (4-D tensor)
`core/render_amp.py:157, 241` — `amped_values_rerolled[:,:,0] = ...` assumes `[N, features, T]`. SHs are `[N, 16, 3, T]`, so `[:,:,0]` selects **color channel 0 across all frames**: the R-channel SH amplification is silently reverted for every frame, and frame 0 is never reset. Live whenever `amp_factors[5] != -1`; pump configs have `no_dshs=False`, so SHs genuinely vary over time.
**Fix:** use `[..., 0]` indexing.

### 8. `load_config` crashes with pinned mmcv 2.2.0 (UI + harness broken end-to-end)
`core/render_amp.py:652-654` — uses `mmcv.Config.fromfile`; `mmcv==2.2.0` moved `Config` to `mmengine`. Unconditional `AttributeError` whenever `--configs` is set via `load_config`. The `__main__` path was fixed (`:684-687`, mmengine first) and the vendored copy documents the fix (T11), but it was never backported to `load_config`, which is what `ampUI.py:31` and `run_renders_auto.py:26` call.
**Fix:** backport the mmengine-first import to `load_config`.

### 9. Stale relative paths after the repo restructure (harness/UI cannot run)
`run_renders_auto.py:112`, `ampUI.py:84, 90, 114` — both assume cwd contains `./output` **and** `./arguments`. Repo root has `output/` but `arguments/` moved to `core/`. The documented launch command crashes at `os.listdir("./arguments")`. Also: hardcoded Windows paths and a stale `models` list referencing `multipleview\test_hand_2` (not a current benchmark scene).
**Fix:** resolve paths relative to repo root (or use `pipeline/paths.py` conventions).

---

## MEDIUM

### 10. `irfft` without `n=n_frames` crashes on odd frame counts
`core/render_amp.py:150, 222, 299, 372` — output length defaults to `2*(bins-1)` = T-1 for odd T → broadcast `RuntimeError`. 60-frame pump scenes dodge it by luck. One-argument fix: `n=n_frames`.

### 11. Post-activation amplification leaves parameter domain
`core/motion_amp/renderer.py:98-100` applies activations before returning, and no clamp/re-normalization happens after amplification:
- opacity can leave [0,1]: below 0 the Gaussian silently vanishes (`forward.cu:346` `alpha < 1/255` cutoff);
- scales amplified in linear space can go ≤ 0; Sigma is quadratic in scale so negatives render as their positive mirror — plausible-looking, wrong;
- rotations: see bug 5.

### 12. Filter band-edge semantics inconsistent across the four methods
`render_amp.py:146` inclusive `>=`/`<=` vs. `:218, 295, 357` strict `>`/`<`. With default `freq_low=0.0, freq_high=1.0`, `eulerian` includes DC+Nyquist, the other three exclude both. Same CLI values → different filters per method.

### 13. `abs` variants anchor to the LAST frame, comment says "initial element"
`render_amp.py:283, 302, 347, 386` — `torch.narrow(values_tensor, -1, -1, 1)` takes the final frame. Textbook Eulerian reference is the canonical (undeformed) positions or the temporal mean. Also: `abs` methods with `a=1` full-band do **not** reproduce the input video (unlike `eulerian`). Comment-or-code mismatch; decide the intended reference.

### 14. Silent truncation on argument-length mismatches
`render_amp.py:694` (`zip(freq_low, freq_high)`) and `:117` (`zip(values_list, amp_factors, freq_cutoffs)`) silently truncate to the shortest. E.g. 8 amp factors + 1 freq pair → only parameter 0 is ever visited; a non-`-1` factor elsewhere is silently dropped. Add length validation. Related: `a == -1` is the skip sentinel, so true motion inversion (−1×) is unexpressible; the UI allows `min_value=-1.0`.

### 15. `results.csv` schema incompatible with its own writer; non-atomic, end-of-run only
`run_renders_auto.py:131-132` — `pd.DataFrame(results)` without `columns=` writes integer headers; the archived CSV has named headers. `to_csv` writes directly to the final path once at the very end — a crash loses the entire run. Fix: explicit columns, incremental append or write-to-temp-then-rename.

### 16. `cameras.py`: untrustworthy timebase and cross-camera alignment
- Writer stamps fixed 20 fps while frames are paced by `sleep(0.05)` + variable read latency; no timestamps or frame counters recorded — corrupts the frequency axis for FFT analysis, undetectably.
- `sync_event.set(); clear()` is a non-blocking pulse; busy workers miss it → frame k in camera1.mp4 ≠ frame k in camera2.mp4.
- Unverified `cap.set(3/4)` resolution: if rejected, frames don't match the hardcoded writer size and OpenCV **silently discards every frame**.
- One camera dying produces a shorter file while others record on, silently.
- A worker blocked in `cap.read()` on a stalled camera never observes `stop_event`; `join()` hangs.

### 17. `low_vram_mode` not forwarded to `generate_frame_data`
`render_amp.py:558` — the largest memory object (8 params × N Gaussians × T frames) is accumulated on GPU during extraction even in low-VRAM mode. Memory-only impact, results unaffected.

---

## LOW

### 18. Unknown `--method` silently renders an unamplified video
`render_amp.py:561-569` — if/elif chain has no `else`; a typo produces a plausible-looking unamplified video. Add an `else: raise`.

### 19. Dead retry in `multithread_write`
`render_amp.py:521-523` — `status` is a `Future`, never `== False`; failed PNG writes are silently swallowed (mp4 still written).

### 20. `override_color` branch missing `shs_final = None`
`core/motion_amp/renderer.py:118-119` — diverges from upstream `gaussian_renderer/__init__.py:117`; latent today (no caller passes `override_color`), crashes if ever used.

### 21. Cosmetic FPS print off-by-one
`render_amp.py:579` — `(len(views)-1)/Δt` should be `len(views)`.

---

## Scope finding (not a bug, but important for the thesis)

**There is no per-motion-segment amplification in `render_amp.py` at all.** `amp_factors` is per *parameter channel* (8 slots), not per part. "Segmented" method variants mean memory chunking (1024-Gaussian splits), not segmentation masks. Per-part amplification currently exists only via the orchestrator's separate seg pipeline. If per-part amplification is a thesis claim, this gap needs to be closed or explicitly scoped.

## Verified correct (selected)

- FFT normalization (`norm="ortho"` both ways) — round-trip is exact.
- Roll/re-roll frame alignment — identity at `a=1` all-pass (verified algebraically).
- Deformation queried at correct per-view timesteps; no state leakage between frames (`forward_dynamic` is pure).
- Quaternion sign-flip artifacts unlikely (`apply_rotation=False` everywhere; additive residuals anchor to canonical sign).
- Opacity squeeze `[N,1]→[N]` safe (rasterizer reads a flat pointer).
- Chunked FFT along Gaussian axis mathematically identical to full-tensor FFT.
- `run_renders_auto.py` reloading config every repeat (saves it from bug 3).

## Unresolved / needs empirical settling

1. Timing-bias magnitude (bug 4): add `torch.cuda.synchronize()` and diff against archived `results.csv` rows.
2. `_record_memory_history(enabled=True)` per `AMPUI()` construction may accumulate trace buffers across repeats, skewing later repeats.
3. `frozen_cam=True` (harness) vs `False` (UI) — intentional?
4. Frequency bounds are fractions of Nyquist of the *frame count*, not real time — non-portable across scenes with different frame counts/fps; worth a sentence in the thesis docs.
5. Reference frame for `abs` methods (bug 13): compare against deviation-from-canonical (`pc.get_xyz` before deformation).

## Fix priority

1. Bug 1 (zero wrap-around delta before FFT)
2. Bug 2 (decide increment vs displacement; thesis claim depends on it)
3. Bug 5 + 6 (quaternion re-normalization + UI label swap)
4. Bug 8 (one-line mmengine backport — unbreaks UI and harness)
5. Bug 9 (path fixes — harness runnable again)
6. Bug 3 (in-place mutation — UI trustworthy)
7. Bug 4 (GPU sync — benchmark numbers publishable)
8. Bugs 7, 10, 12, 13, 14 (correctness hardening + validation)

---

## Fixes applied

### amp-ui fixes applied 2026-10-05

- **Bug 3** — `ampUI.py` and `run_renders_auto.py` now deep-copy the per-frame parameter structure via a `_clone_values()` helper (tensor `.clone()`, list-of-lists rebuilt, non-tensor entries like `colors_precomp`/`cov3D_precomp` passed through) at render time only, and pass the copy into `amplify_frame_data_*`; `self.values` always retains raw extracted data.
- **Bug 4** — `torch.cuda.synchronize()` (guarded by `torch.cuda.is_available()`) added before starting and before stopping the amplification timer in both `AMPUI.render()` copies (UI and harness), so both VRAM modes are timed identically.
- **Bug 6** — `ampUI.py` channel labels corrected to `["pos3d","pos2d","scale","rotation","opacity","SHs","color","cov3D"]`, matching the `values_list` order in `core/render_amp.py`.
- **Bug 9** — both files now resolve `REPO_ROOT` from `__file__` (two levels up) and use `<root>/output` and `<root>/core/arguments`; `<root>/core` is inserted into `sys.path` so `render_amp` imports regardless of cwd. The hardcoded backslash `models` list in `run_renders_auto.py` is replaced by `discover_models()`, which scans `output/` for trained models (dirs containing `cfg_args`) and pairs each with `<dataset>/<name>.py`, falling back to `<dataset>/default.py` / `<dataset>_default.py`; if nothing is found it exits with an error listing the model dirs on disk. Verified on disk: 6 models discovered (dnerf/bouncingballs, dnerf/lego, multipleview/pump01, test1, test6, test_hand_2).
- **Bug 15** — `results.csv` now uses explicit columns `["model","method","low_vram","mem_alloc","mem_seg","time","error_msg"]` (matching the archived file's named headers), no index column, written to `<root>/results.csv`; `save_results()` rewrites from accumulated results after **every** benchmark combo via temp file + `os.replace` for atomicity, so a crash preserves completed rows.
- **Bug 16** — `cameras.py` rewritten: (a) per-camera sidecar CSV (`<name>_timestamps.csv`: frame_idx, camera_index, wall_time_ns, monotonic_ns) is the authoritative timebase, documented in the module docstring; (b) `sync_event` pulse replaced by a `threading.Barrier` across workers + controller so every camera captures the same cycle, with loud logging on a broken barrier; (c) requested 1280×720 verified against the actual frame shape, VideoWriter adjusted to the real size with a warning; (d) a failed `read()` sets a shared error flag and aborts the barrier, stopping all cameras and naming the culprit; (e) `CAP_PROP_READ_TIMEOUT_MSEC` set where supported, barrier aborted on stop so waiting workers wake, joins use a 5 s timeout with a loud warning.

**Not fixed here (owned by the parallel core/ effort):** `load_config`'s mmcv/mmengine crash (bug 8) — once that lands, both amp-ui entry points should run end-to-end without further changes on the amp-ui side; no other dependency on the core fix was found.

### core fixes applied 2026-10-05

Applied to `core/render_amp.py` and mirrored byte-for-byte in the amplification logic of
`orchestrator/pipeline/vendored/cuda/amp.py` (vendored header updated to document the divergence
from the "verbatim copy" policy); `core/motion_amp/renderer.py` for bug 20.

- **Bug 1 + Bug 2** (`render_amp.py:184-264, 266-351`) — `eulerian`/`eulerian_mod` reimplemented as
  textbook Eulerian displacement amplification: `out[t] = mean + a * filter(v[t] - mean)` with the
  per-Gaussian temporal mean as reference. The rolled first difference (and with it the wrap-around
  sample) no longer exists, so bug 1 is fixed by elimination rather than by zeroing `delta[..., -1]`.
  CLI `--method` names unchanged; docstrings and `--method` help text document the semantics change.
- **Bug 13** (`render_amp.py:353-430, 432-520`) — `eulerian_abs*` now use the per-Gaussian temporal
  mean as reference instead of the last frame (`torch.narrow(..., -1, -1, 1)`); comments fixed. All
  four methods now share identical mean-anchored semantics (`*_mod` = Gaussian-chunked, `*_abs` names
  kept for CLI compatibility); `a=1` full-band is an exact identity for all four.
- **Bug 7** — fixed by elimination: the frame-0 reset (`[:,:,0]`, wrong axis for 4-D SHs) was only
  needed by the old first-difference formulation and no longer exists in any method.
- **Bug 10** (`render_amp.py:246, 328, 412, 500`) — all four `irfft` calls pass `n=n_frames`; odd
  frame counts verified working.
- **Bug 11 + Bug 5 (Python side)** — new `_sanitize_amplified_values()` (`render_amp.py:113-129`)
  applied right before results are written back (`:253, 345, 419, 514`): opacity clamped to [0, 1],
  scales clamped to >= 1e-8, rotation quaternions re-normalized along the feature axis. CUDA fork
  untouched, as instructed.
- **Bug 12** (`render_amp.py:242, 308, 408, 483`) — all four methods use inclusive `>=`/`<=` band edges.
- **Bug 14** — new `validate_amp_args()` (`render_amp.py:132-158`) called at the top of all four
  amplify functions and of `render_set_amp` (`:693`): `amp_factors` must be exactly length 8, `-1` is
  the only allowed negative; `freq_cutoffs` length 1 broadcasts to 8, other lengths raise.
  New `build_freq_cutoffs()` (`:161-181`) used in `__main__` (`:849`): `--freq_low`/`--freq_high`
  must have equal length 1 or 8, with length-1 broadcast. Orchestrator CLI already passes 8/8, so
  existing configs are unaffected.
- **Bug 17** (`render_amp.py:697`) — `low_vram_mode` forwarded to `generate_frame_data`.
- **Bug 18** (`render_amp.py:709-712`) — unknown `--method` raises `ValueError` listing valid methods.
- **Bug 19** (`render_amp.py:650-657`) — `multithread_write` calls `.result()` on each future; failed
  writes retry once synchronously, then raise `RuntimeError`.
- **Bug 21** (`render_amp.py:722`) — FPS print uses `len(views)`.
- **Bug 8** (`render_amp.py:795-803`) — `load_config` now tries `mmengine.config.Config` first and
  falls back to `mmcv`, matching the `__main__` path; same change in the vendored copy (whose
  `mmengine`-only call sites were likewise converted to try/fallback for parity).
- **Bug 20** (`core/motion_amp/renderer.py:120`) — `shs_final = None` restored in the
  `override_color` branch, matching upstream `core/gaussian_renderer/__init__.py:117`.

**Deliberate extra change (behavior-preserving):** `.cuda()` calls in the amplify functions are
guarded by `torch.cuda.is_available()` and filter masks use `.to(<tensor>.device)` instead of
`.cuda()`, so the amplification math is unit-testable on CPU-only machines. No-op on CUDA machines.

**Verification (CPU, torch 2.14.1+cpu, throwaway venv — script not committed):** the four amplify
functions were extracted from the real files via AST and exercised on synthetic data (N=100
Gaussians, T=60/61 frames, sinusoidal motion, 8 parameter slots incl. 4-D SHs `[N,16,3]`):
77/77 checks passed for both `core/render_amp.py` and the vendored copy — (a) `a=1` full-band
identity for all methods; (b) `a=5` yields measured displacement gain 5.000 (was ~1+(a-1)·w before);
(c) odd T=61 runs and stays identity; (d) quaternions unit-norm at `a=8`; (e) opacity in [0,1] and
scales >= 1e-8 at `a=8`; plus chunked-vs-full agreement and all bug-14 validation/broadcast cases.
`ast.parse` passes for all three edited files. Orchestrator sandbox suite (`pytest -q`, MCP tests
excluded for missing optional deps): 224 passed, 9 skipped; 10 failures in
`test_roi_motion_gate.py`/`test_segment_kabsch.py`/`test_stages_isaac.py` are pre-existing and
unrelated — those tests only import `pipeline.vendored.host.*`/`isaac.*`, never the files edited here.

**Not fixed (out of scope / other owner):** bugs 3, 4, 6, 9, 15, 16 (amp-ui, fixed by the parallel
effort above); the vendored `--amp_factors type=int` quirk (intentionally preserved, documented in
the vendored header); the "no per-motion-segment amplification" scope finding (thesis scoping
decision, not a bug fix); the unresolved empirical items in the review.
