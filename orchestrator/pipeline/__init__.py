"""Layer 1 execution engine for the 4DGS motion-amp orchestrator.

Pure-Python DAG runner that wraps the existing pipeline scripts (Isaac capture,
4DGS train/render, motion segmentation, amplification) behind a stage registry,
typed artifacts, and a run manifest.

This top-level package must stay import-light: no torch/CUDA/Isaac imports at
module scope anywhere under ``pipeline/``. Those only happen inside a stage's
``run()`` body, which executes inside the relevant container/env (``cuda`` or
``isaac``), never inside the CPU-only host process that imports this package.
"""

__version__ = "0.1.0"
