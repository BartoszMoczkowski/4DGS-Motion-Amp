# W5 (T53) — Reversible outer-loop component split and merge

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W4 (T52)  
**Experiment:** E4 (Best W4 configuration + reversible split/merge)  
**Gate before W6:** Synthetic two-mover scene initialized at $K=1$ discovers $K=2$ with soft responsibilities matching ground-truth movers; static scene stays at $K=0/1$ without inventing components; rejected split restores exact pre-proposal model/optimizer/scheduler state. All gate tests verified in `orchestrator/tests/test_w5_component_split_merge.py`.

---

## 1. Objective

Extend the fixed-$K$ soft motion component model (W4) with an outer-loop **reversible split/merge controller** that treats structural changes ($K \to K+1$ or $K \to K-1$) as testable hypotheses validated on held-out views, with exact rollback if held-out reconstruction or motion coherence fails to improve.

## 2. Implementation Steps

1. **Proposal trigger & guards:**
   - Run only after component warm-up (`split_merge_start_iter`) at configurable intervals (`split_merge_interval`), subject to `max_components` ($K_{\max}$), minimum responsibility mass (`min_component_support`), and proposal cooldown.
   - **Split criterion:** Component $k$ has support $\sum_i q_{ik} \ge N_{\min}$ and statistically significant structured residual $h_i(t)$ (spatial coherence + bimodal/multi-modal residual displacement direction, amplitude, or phase) that is not explained by a single rigid transform.
   - **Merge criterion:** Two components $(k_1, k_2)$ have near-identical rigid trajectory parameters and overlapping support, or a component's support drops below the pruning floor.
2. **Soft candidate initialization:**
   - For a split of component $k$, cluster the responsibility-weighted residuals $h_i(t)$ softly into two children $(k_a, k_b)$ so $q_{i, k_a} + q_{i, k_b} = q_{ik}$ (every Gaussian retains fractional responsibilities; no binary thresholding) and initialize child trajectories from parent trajectory + residual fit while preserving the HexPlane state.
3. **Held-out acceptance window & exact rollback:**
   - Before applying a candidate split or merge, snapshot `(gaussians.capture(), motion_components.state_dict(), optimizers, schedulers, rng_state)` in memory/disk.
   - Train for a short evaluation window (`proposal_eval_window_iters`, e.g. 200–500 steps) and evaluate held-out view PSNR/L1 and residual motion coherence against predeclared improvement margins.
   - If accepted: commit the structural change and log the event.
   - If rejected: restore the exact pre-proposal snapshot, mark the candidate signature in a cooldown/rejection ledger so resumed or subsequent intervals do not repeat the same rejected split, and log the rollback.

## 3. Verification & Tests

- **Synthetic $K=1 \to K=2$ discovery test:** Two distinct rigid movers initialized with a single component split into two components and assign soft responsibilities matching the two movers.
- **Static scene guardrail test:** A purely static scene triggers zero accepted splits.
- **Non-rigid mover guardrail test:** A smoothly deforming non-rigid object retains its free-form HexPlane residual rather than shattering into $K_{\max}$ rigid components.
- **Exact rollback test:** Force a split rejection and verify all model tensors, responsibilities, component parameters, and optimizer states match the pre-proposal snapshot bit-for-bit.
- **Merge & checkpoint resume test:** Redundant duplicate components merge cleanly; checkpoint save/resume preserves structural history and rejection cooldown state.

## 4. Deliverables & Stop Conditions

- **Deliverables:** Reversible split/merge controller, structured proposal/rollback log schema in run manifest, unit and synthetic pilot tests (`orchestrator/tests/test_w5_component_split_merge.py`), and `E4` preset.
- **Stop conditions:** Stop if static geometry acquires moving components, if rollback is not bit-exact, or if splits are accepted on training loss while degrading held-out metrics.

