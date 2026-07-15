# T15 — UI (Streamlit over Layer 1 API)

- Status: todo
- Phase: 5
- Depends on: T09
- Environment: host

## Goal
A thin UI for Bartosz to run, tune, and review — over the *same* Layer 1 API (no logic
duplication). Deprioritized. (Milestone M5.)

## In scope
- Streamlit app (reuse `ampUI.py`'s amplification-param panel as one view).
- Views: pick/edit a preset; launch a run; watch per-stage progress + live logs + GPU meter;
  browse artifacts/previews (renders, segmentation PNGs, amp videos); compare runs.
- Talks to Layer 1 directly (or the T14 HTTP server) — pick one and document.

## Out of scope
New capabilities beyond what Layer 1 exposes.

## Deliverables
`orchestrator/ui/` Streamlit app + run instructions.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- Launch and monitor a run from the UI; see previews and compare two runs.
- No pipeline logic lives in the UI — it only calls the API.

## Relevant existing files
`ampUI.py` (existing Streamlit amp UI to fold in), Layer 1 API, manifest/artifacts.

## Notes / gotchas
Explicitly the least important layer — keep it thin. If the T14 tool surface already covers
Bartosz's day-to-day via Claude, this can stay minimal.
