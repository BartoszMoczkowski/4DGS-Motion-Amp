# Superseded — see WINDOWS_SETUP.md

**This guide is retired (2026-07-14).** It assumed the orchestrator had to run *from inside* a
WSL2 Linux distro. That assumption was revised: the runtime host is now native Windows (Docker
Desktop is reachable directly from Windows, no WSL2 shell needed) — see
`planning/INSTRUCTIONS.md`'s locked "Runtime host" decision and
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Runtime host moved off WSL2" entry.

Use **`planning/WINDOWS_SETUP.md`** instead for first-time machine setup.

Bundling a proper WSL2/Docker environment as an alternative way to run this project is deferred,
unscheduled future work — see `planning/tasks/T16-wsl-docker-bundling.md`. If that ever gets
picked back up, this file's original content (WSL2 distro setup, `nvidia-ctk`, `/mnt/q` assets
mounting, etc.) is still in the repo's git history as a starting point.
