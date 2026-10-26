# T41 — Fourier-rigid deformation head (R9)

**Status:** done
**Phase:** 12 / Phase 3 (structured motion)
**Depends on:** T39
**Remedy:** R9

## Goal

Give the wobble a parameterization that matches its physics instead of asking
~300 000 independent decoder queries to rediscover one rigid sinusoid.

## Design

- New module `core/scene/fourier_rigid.py`, gated to the turbine ROI by a
  mask (from `motion-seg` clusters or a 3D bounding box `roi_bbox`).
- All ROI Gaussians share one rigid transform $(R(t), T(t))$:
  - Multi-harmonic translation:
    $$T(t) = \sum_{n=1}^{N_h} a_n \mathbf{u} \sin(2\pi n f t + \phi_n) + \sum_{n=1}^{N_h} b_n \mathbf{v} \sin(2\pi n f t + \psi_n)$$
    along learnable dominant axis $\mathbf{u}$ and transverse axis $\mathbf{v}$ with $N_h=3$ harmonics;
  - Small-angle Rodrigues rotation:
    $$R(t) = I + \sin\theta \hat{\mathbf{w}} + (1 - \cos\theta) \hat{\mathbf{w}}^2$$
    about a learnable 3D pivot $\mathbf{p}$, with rotation quaternion applied to Gaussian covariance via `batch_quaternion_multiply`;
  - Base frequency anchored at $f = 7.112$ Hz with optional learnable residual offset $\Delta f$;
  - Optical track initialization via `init_from_optical_tracks()` reading `trajectory_tracks.json` or `optical_offsets.json`.
- ~25 parameters instead of ~10⁶ — resists the conflicting-gradient regime and
  makes the motion **fully auditable** (frequency, 3D dominant axis, harmonic amplitudes in mm, and phases in deg readable via `get_summary()`).
- Additive hybrid mode (`pure_rigid=False`) preserves HexPlane deformations outside the ROI and adds rigid transform inside; pure rigid mode (`pure_rigid=True`) strictly overrides deformation with rigid motion on the ROI.
- Staged unfreezing schedule (`fourier_head_freeze_iter = 500`) keeps the head frozen during early coarse/fine densification.

## Verification

- `orchestrator/tests/test_fourier_rigid_head.py`:
  - `test_fourier_rigid_head_init_and_parameter_count`: confirms parameter count is $\sim 25$ ($< 40$).
  - `test_rodrigues_rotation_exactness`: validates exact agreement with analytical 90-degree axis rotation.
  - `test_fourier_translation_periodicity`: verifies exact periodic translation $T(t + 1/f) = T(t)$.
  - `test_fourier_head_roi_gating`: confirms points outside ROI undergo 0 displacement while points inside receive rigid transform.
  - `test_fourier_head_pure_rigid_mode`: verifies strict override in pure rigid mode.
  - `test_fourier_head_auditability_summary`: verifies `get_summary()` JSON output contains frequency, dominant axis, amplitudes, and phases.
  - `test_init_from_optical_tracks`: verifies initialization from optical tracking files.
  - `test_fourier_head_gradient_flow`: confirms non-zero gradient flow back to translation amplitudes, phases, axes, pivot, and frequency.
  - `test_fourier_head_freeze_schedule`: validates freeze schedule gating during early iterations.
  - `test_optimization_params_schema_r9`: verifies Pydantic strict model schema serialization and defaults.
  - `test_preset_resolution_and_bridge_generation_r9`: verifies preset inheritance and CLI argument bridge.
  - All 11 tests passed in 8.29s.
- Full regression suite: 538 passed, 9 skipped in 106.32s.

## Deliverables

1. `core/scene/fourier_rigid.py`: `FourierRigidHead` module and Rodrigues rotation math.
2. `core/gaussian_renderer/__init__.py`: ROI motion evaluation and quaternion covariance updates.
3. `core/arguments/__init__.py` & `orchestrator/pipeline/config/models.py`: R9 configuration schema.
4. `core/train.py` & `orchestrator/pipeline/vendored/cuda/train.py`: Freeze schedule, optimizer step, and `fourier_head.pth` / `fourier_head_state.json` persistence.
5. `orchestrator/pipeline/config/presets/lego_turbine_2_missing_r9.yaml`: Phase 3 preset.
6. `orchestrator/tests/test_fourier_rigid_head.py`: Comprehensive test suite.

