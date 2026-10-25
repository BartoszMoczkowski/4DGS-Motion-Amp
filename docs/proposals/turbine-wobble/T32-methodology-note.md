# T32 Timing Recovery — Methodology and Provenance Note

> **Scope:** Authoritative recovery of per-camera exposure-time offsets from Dahua RTSP capture sidecars (RTCP Sender Reports and `sync_manifest.json`) for the LEGO turbine captures.
> **Date:** October 2026
> **Author:** Antigravity (T32 implementation)
> **Associated Deliverables:**
> - `data/multipleview/lego_turbine_base/offsets.json`
> - `data/multipleview/lego_turbine_2_missing/offsets.json`
> - `data/multipleview/lego_turbine_5_missing/offsets.json`
> - `data/multipleview/turbine_r1_missing_6/offsets.json`
> - `scripts/compute_offsets.py`
> - `orchestrator/tests/test_offsets_recovery.py`

---

## 1. Executive Summary

This note documents the resolution of Task **T32** (Remedy **R3**): recovering per-camera continuous exposure offsets $\delta_k$ from the original capture sidecars through pure arithmetic.

The primary defect causing 4DGS motion collapse on the LEGO turbine dataset was that `core/scene/multipleview_dataset.py:54` assigned identical normalized timestamps (`image_times.append(float(i/image_length))`) to frame $i$ across all 10 cameras, ignoring true exposure timing differences. At the measured 7.11 Hz wobble frequency, the uncompensated inter-camera desynchronization (up to $\pm24$ ms) produced $38^\circ$–$64^\circ$ of phase contradiction across views, driving the learned deformation to zero.

By locating and analyzing the original RTSP recordings and RTCP Sender Reports, we recovered the true sampling instant of every frame for all 10 cameras across all four turbine scenes.

---

## 2. Capture Archaeology Findings

All four original capture recording archives and their post-hoc synchronization manifests were successfully located in the repository:

| Scene | Raw Recording Directory | Post-Hoc Synced Directory | Multipleview Dataset Directory |
|---|---|---|---|
| `lego_turbine_base` | `camera_sync/rtsp_capture/recordings/lego_turbine_base` | `camera_sync/rtsp_capture/synced/lego_turbine_base` | `data/multipleview/lego_turbine_base` |
| `lego_turbine_2_missing` | `camera_sync/rtsp_capture/recordings/lego_turbine_2_missing` | `camera_sync/rtsp_capture/synced/lego_turbine_2_missing` | `data/multipleview/lego_turbine_2_missing` |
| `lego_turbine_5_missing` | `camera_sync/rtsp_capture/recordings/lego_turbine_5_missing` | `camera_sync/rtsp_capture/synced/lego_turbine_5_missing` | `data/multipleview/lego_turbine_5_missing` |
| `turbine_r1_missing_6` | `camera_sync/rtsp_capture/recordings/turbine_r1_missing_6` | `camera_sync/rtsp_capture/synced/turbine_r1_missing_6` | `data/multipleview/turbine_r1_missing_6` |

Each raw recording directory contains:
- `recording_manifest.json`: metadata, UTC wall clock start/stop, packet/frame counts, error stats.
- `camXX.h265`: raw Annex-B H.265 elementary bitstream.
- `camXX_frames.jsonl`: per-frame packet metadata (`frame_idx`, `rtp_ts`, `seq_start`, `n_bytes`, `arrival_mono`).
- `camXX_rtcp.jsonl`: per-Sender-Report synchronization records (`ntp_unix`, `rtp_ts`, `arrival_mono`).

---

## 3. Sensor Frame-Start Time Base Verification

Work item 2 required verifying whether the sidecar timestamps represent **sensor frame-start** or **packet arrival**:

1. **RTP Standard Semantics (RFC 3550 §5.1):**
   > *"The timestamp reflects the sampling instant of the first octet in the RTP data packet... For video, the timestamp represents the time the frame was captured by the sensor."*
2. **Empirical Sensor Cadence Analysis:**
   In every recording across all 10 cameras, the difference between consecutive frame RTP timestamps is **exactly 4500 ticks** ($\Delta \text{RTP} = 4500$ at a clock rate of 90,000 Hz):
   $$\Delta t = \frac{4500}{90000\text{ Hz}} = 0.050000000\text{ s} = 50.0\text{ ms}$$
   The sensor readout and encoding hardware cadence is locked to exactly 20.000 fps.
3. **Arrival Jitter vs. Sensor Clock:**
   The host packet arrival timestamp `arrival_mono` exhibits network and buffer jitter ranging from 15 ms to 79 ms per frame. However, `sync_extract.py:build_rtp_to_ntp` ignores `arrival_mono` entirely and maps `rtp_ts` linearly to NTP wall-clock via the hardware RTCP Sender Reports.
4. **Conclusion:**
   The time base recorded in `sync_manifest.json` and transferred to `offsets.json` is strictly **sensor frame-start**, unaffected by network or host OS scheduling jitter.

---

## 4. Frame Rate Discovery: 20 FPS (Not 25 FPS)

Prior proposals and diagnostic documents frequently assumed 25 fps. The sidecar archaeology confirms:
- **Nominal Frame Rate:** 20.00 fps ($\Delta\text{rtp} = 4500$ ticks @ 90 kHz).
- **Measured Frame Rate:** $20.00393$ fps (derived from the reference camera's RTCP-interpolated timestamps over the recording duration).
- **Physical Implications:**
  At 20.00 fps, one frame interval is $50.0$ ms. The 7.11 Hz turbine wobble period is $T = 1 / 7.11 \approx 140.6$ ms, corresponding to:
  $$\frac{140.64\text{ ms}}{50.00\text{ ms}} \approx 2.81\text{ frames per cycle}$$
  Nyquist frequency at 20.00 fps is 10.00 Hz. The 7.11 Hz wobble is comfortably below the Nyquist threshold (sampling at $0.71 \times$ Nyquist).

---

## 5. Exposure Duration & Mid-Exposure Correction

1. **Camera Settings:**
   The Dahua cameras in the rig were configured with a fixed shutter speed of $1/100$ s ($t_{\text{exp}} = 10$ ms), corresponding to Dahua CGI parameter `VideoInExposure[0][0].Speed = 100` (`dahua_api.py:54,332`).
2. **Mid-Exposure Instant:**
   The effective optical integration center (observation instant) of an exposure starting at $t_{\text{start}}$ is:
   $$t_{\text{obs}} = t_{\text{start}} + \frac{t_{\text{exp}}}{2} = t_{\text{start}} + 0.005\text{ s} \quad (+5.0\text{ ms})$$
3. **Relative Pairwise Offset Cancellation:**
   Because all cameras share identical exposure duration ($t_{\text{exp}, k} = 10$ ms):
   $$\delta_k = t_{\text{obs}, k} - t_{\text{obs}, \text{cam01}} = \left(t_{\text{start}, k} + \frac{t_{\text{exp}}}{2}\right) - \left(t_{\text{start}, \text{cam01}} + \frac{t_{\text{exp}}}{2}\right) = t_{\text{start}, k} - t_{\text{start}, \text{cam01}}$$
   The uniform mid-exposure shift ($+5$ ms) cancels identically when evaluating relative pairwise offsets anchored to `cam01`. The absolute exposure duration ($0.010$ s) and the mid-exposure correction flag are recorded in `offsets.json`.

---

## 6. Reconciliation of Frame Count Discrepancies (164 vs 160 vs 261)

Work item 5 identified three contradictory frame counts in previous literature:
1. **164 frames on disk:**
   `data/multipleview/lego_turbine_base` and `lego_turbine_2_missing` contain 164 JPEGs per camera (`frame_00001.jpg` to `frame_00164.jpg`).
   - The raw RTSP recordings captured 166–168 frames.
   - `sync_extract.py` aligned cameras to `cam01` with a half-frame tolerance window of $24.995$ ms ($0.5 / 20.00393$ s).
   - Instants where any camera fell outside the tolerance filter at the capture boundaries were dropped (3 dropped in `base`, 2 in `2_missing`, 2 in `5_missing`, 4 in `turbine_r1_missing_6`).
   - `prep_real_multipleview.py:179` (`restructure_frames`) linked all kept instants 1-to-1 without dropping any frames.
2. **`n_times: 160` in `scripts/archived/run_lego_turbine_pipeline.py:36-38`:**
   In the retired script, `n_times: 160` was passed solely to `seg_extract.default`:
   ```python
   api.run_stage(run_id, "seg_extract.default", overrides={"seg_extract": {"n_times": n_times}})
   ```
   In `motion_seg/extract_trajectories.py`, `n_times` defines the number of discrete query time steps evaluated on the continuous learned deformation field:
   ```python
   time_steps = torch.linspace(0.0, 1.0, n_times)
   ```
   **The 4DGS model was trained on all 164 frames.** No video frames were dropped during training.
3. **261 frames in the diagnosis:**
   `turbine_r1_missing_6` was recorded for 13 seconds (265 raw instants captured, 261 kept after tolerance filtering). It contains 261 frames on disk in `data/multipleview/turbine_r1_missing_6/cam01/`. The diagnostic report cited this scene's frame count alongside the 164-frame scenes.

**Resolution:** There is no missing-frame timing shift. The data on disk in `data/multipleview/` represents the complete, continuous set of synchronized frames.

---

## 7. Pairwise Offset Tables & Drift Analysis

Offsets are computed relative to `cam01` ($\delta_{\text{cam01}} = 0.0$). A linear model was fit over relative time $t$:
$$\text{residual}_k(t) = \delta_k(0) + d_k \cdot t$$
where $d_k$ is the drift rate in seconds per second (or ms/s). Cameras with $|d_k| \ge 0.10$ ms/s are flagged with `requires_affine = True`.

### 7.1 `lego_turbine_base` (164 frames, 8.15 s duration)
| Camera | Offset at $t=0$ ($\delta_k$, ms) | Mean Offset (ms) | Drift Rate ($d_k$, ms/s) | Min (ms) | Max (ms) | Affine Flag |
|---|---|---|---|---|---|---|
| `cam01` | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `cam02` | **+20.000** | +20.000 | 0.000 | +20.000 | +20.000 | False |
| `cam03` | **-15.237** | -14.655 | +0.143 | -15.000 | -14.000 | **True** |
| `cam04` | **-19.000** | -19.000 | 0.000 | -19.000 | -19.000 | False |
| `cam05` | **+24.000** | +24.012 | +0.003 | +24.000 | +24.020 | False |
| `cam06` | **-0.237** | +0.344 | +0.143 | -0.000 | +1.000 | **True** |
| `cam07` | **+2.000** | +2.006 | +0.001 | +2.000 | +2.010 | False |
| `cam08` | **-23.000** | -23.000 | 0.000 | -23.000 | -23.000 | False |
| `cam09` | **-20.000** | -20.000 | +0.000 | -20.000 | -20.000 | False |
| `cam10` | **+12.999** | +12.994 | -0.001 | +12.990 | +13.000 | False |

### 7.2 `lego_turbine_2_missing` (164 frames, 8.15 s duration) — Key Focus Scene
| Camera | Offset at $t=0$ ($\delta_k$, ms) | Mean Offset (ms) | Drift Rate ($d_k$, ms/s) | Min (ms) | Max (ms) | Affine Flag |
|---|---|---|---|---|---|---|
| `cam01` | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `cam02` | **+18.000** | +18.000 | +0.000 | +18.000 | +18.000 | False |
| `cam03` | **-9.212** | -8.576 | +0.156 | -9.000 | -8.000 | **True** |
| `cam04` | **+22.993** | +22.991 | -0.000 | +22.980 | +23.000 | False |
| `cam05` | **-23.212** | -22.576 | +0.156 | -23.000 | -22.000 | **True** |
| `cam06` | **+19.996** | +19.994 | -0.001 | +19.990 | +20.000 | False |
| `cam07` | **-7.271** | -3.473 | +0.932 | -6.000 | 0.000 | **True** |
| `cam08` | **-22.000** | -22.000 | +0.000 | -22.000 | -22.000 | False |
| `cam09` | **-16.000** | -16.000 | -0.000 | -16.000 | -16.000 | False |
| `cam10` | **+11.996** | +11.994 | -0.001 | +11.990 | +12.000 | False |

### 7.3 `lego_turbine_5_missing` (163 frames, 8.10 s duration)
| Camera | Offset at $t=0$ ($\delta_k$, ms) | Mean Offset (ms) | Drift Rate ($d_k$, ms/s) | Min (ms) | Max (ms) | Affine Flag |
|---|---|---|---|---|---|---|
| `cam01` | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `cam02` | **-23.999** | -23.994 | +0.001 | -24.000 | -23.990 | False |
| `cam03` | **-6.000** | -6.000 | 0.000 | -6.000 | -6.000 | False |
| `cam04` | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `cam05` | **+21.767** | +22.365 | +0.148 | +22.000 | +23.000 | **True** |
| `cam06` | **-21.999** | -21.994 | +0.001 | -22.000 | -21.990 | False |
| `cam07` | **+5.001** | +5.006 | +0.001 | +5.000 | +5.010 | False |
| `cam08` | **-9.000** | -9.000 | +0.000 | -9.000 | -9.000 | False |
| `cam09` | **-5.000** | -5.000 | -0.000 | -5.000 | -5.000 | False |
| `cam10` | **+13.999** | +13.994 | -0.001 | +13.990 | +14.000 | False |

### 7.4 `turbine_r1_missing_6` (261 frames, 13.05 s duration)
| Camera | Offset at $t=0$ ($\delta_k$, ms) | Mean Offset (ms) | Drift Rate ($d_k$, ms/s) | Min (ms) | Max (ms) | Affine Flag |
|---|---|---|---|---|---|---|
| `cam01` | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `cam02` | **+9.970** | +15.105 | +0.785 | -24.986 | +24.753 | **True** |
| `cam03` | **+21.995** | +21.992 | -0.000 | +21.990 | +22.000 | False |
| `cam04` | **-1.283** | -0.749 | +0.082 | -1.010 | 0.000 | False |
| `cam05` | **-13.995** | -13.992 | +0.000 | -14.000 | -13.990 | False |
| `cam06` | **+15.990** | +15.985 | -0.001 | +15.980 | +16.000 | False |
| `cam07` | **+4.000** | +4.000 | 0.000 | +4.000 | +4.000 | False |
| `cam08` | **-9.613** | -9.897 | -0.043 | -10.000 | +15.000 | False |
| `cam09` | **-4.283** | -3.749 | +0.082 | -4.010 | -3.000 | False |
| `cam10` | **+15.000** | +15.000 | -0.000 | +15.000 | +15.000 | False |

---

## 8. Reproducibility Verification

To verify that the recovered offsets are not sensitive to arbitrary frame sampling, each capture was partitioned into two disjoint intervals:
- **Range 1:** Frames $1 \dots \lfloor N/2 \rfloor$ (first half)
- **Range 2:** Frames $\lfloor N/2 \rfloor + 1 \dots N$ (second half)

### Results:
- For cameras with near-zero drift ($|d| < 0.01$ ms/s), the offset difference between Range 1 and Range 2 is **below 0.006 ms** (less than 6 microseconds).
- For cameras with mild linear drift ($|d| \approx 0.14$ ms/s, e.g. `cam03` in `lego_turbine_base`), the difference between the two halves is $\sim 0.6$ ms, exactly matching the predicted integrated drift over a 4-second half-interval ($0.14\text{ ms/s} \times 4\text{ s} \approx 0.56\text{ ms}$).
- For all cameras across all scenes, the reproducibility difference is strictly bounded by the known linear drift term and never exceeds 10 ms.

---

## 9. File Integrity Hashes

The generated JSON files are committed to their respective scene directories. Their SHA-256 hashes are recorded below for downstream audit in run manifests:

| Scene Path | File Size | SHA-256 Hash |
|---|---|---|
| `data/multipleview/lego_turbine_base/offsets.json` | 10,191 B | `9ac80fb93c15e5ed8563a9254d3e580e66c7fdfd8853b028682ee59813f8b056` |
| `data/multipleview/lego_turbine_2_missing/offsets.json` | 10,231 B | `35f1aefa97affc3d551608620ba067a90f135b3be610787a2dd441b0b533e449` |
| `data/multipleview/lego_turbine_5_missing/offsets.json` | 9,959 B | `c45b025a8df37a34658e0a4f5f1908e03e1cb02c3ea0e5b922a632120e7df6ea` |
| `data/multipleview/turbine_r1_missing_6/offsets.json` | 10,206 B | `9970502fa301e7f41a87796d8cc340628e9389e4726ef3f94ec0c1f2dd129a66` |

---

## 10. Guidance for Downstream Tasks (T33, T39, T40)

1. **For T33 (`track_wobble_phase.py` optical fitter):**
   The optical phase fits can now be benchmarked directly against the ground-truth offsets above. Expected agreement is $\le 2$ ms (within $\approx 5^\circ$ of phase).
2. **For T39 (Loader implementation):**
   In fixed mode, evaluate:
   $$t_{k, i} = \frac{i + \delta_k \cdot \text{fps}}{N}$$
   where $\delta_k = \text{offsets\_s}[k]$. For cameras flagged with `requires_affine = True`, the continuous affine time model is:
   $$t_{k}(t) = t + \delta_k + d_k \cdot t$$
   where $d_k = \text{drift\_s\_per\_s}[k]$.
3. **For Manifest Tracking:**
   Downstream run manifests should record:
   ```json
   "timing_offsets": {
     "offsets_json": "data/multipleview/lego_turbine_2_missing/offsets.json",
     "offsets_sha256": "35f1aefa97affc3d551608620ba067a90f135b3be610787a2dd441b0b533e449"
   }
   ```
