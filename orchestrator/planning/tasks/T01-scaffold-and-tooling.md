# T01 — Subproject scaffold & tooling

- Status: done
- Phase: 0
- Depends on: —
- Environment: host

## Goal
Stand up the `orchestrator/pipeline/` Python package skeleton and dev tooling so every later task
has a home and a consistent import path.

## In scope
- Create `orchestrator/pipeline/` package (`__init__.py`, submodule stubs: `config/`, `stages/`,
  `dag/`, `artifacts/`, `containers/`, `resources/`, `api.py`).
- Wire package into the repo build (`pyproject.toml`: add package + any pure-Python deps —
  pydantic, docker SDK, pynvml — as an optional `[orchestrator]` extra so it doesn't disturb the
  existing CUDA env).
- Minimal `pipeline/api.py` with typed stub signatures (raise `NotImplementedError`) matching the
  public API in ARCHITECTURE.md.
- Test harness: `pytest` config + a `tests/` dir with one trivial import test.

## Out of scope
Any real logic; containers; config content (T02).

## Deliverables
Importable `pipeline` package; `python -c "import pipeline"` works in `.venv`; `pytest` green.

## Acceptance criteria
- Package imports with no side effects and no CUDA/torch import at top level.
- Extra installs cleanly (`uv sync` / `pip install -e .[orchestrator]`) without perturbing the
  existing training deps.

## Relevant existing files
`pyproject.toml`, `.venv/`, `main.py`.

## Notes / gotchas
Keep top-level imports light — the package must import fine in the CPU-only sandbox (no torch).
Torch/CUDA only inside stage `run()` bodies that execute in the `cuda` env.
