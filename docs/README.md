# Documentation index

Compiled documentation for the thesis project **Motion Amplification for 4D Gaussian Splatting** (Bartosz Moczkowski, Technology University of Lodz). These files summarize the work done in the repo; the deeper, chronological working notes live in `.claude_notes/` and `orchestrator/planning/`.

| Document | Contents |
|---|---|
| [overview.md](overview.md) | Project goal, repo map, current status at a glance |
| [motion-segmentation.md](motion-segmentation.md) | MultiBodySync ↔ 4DGS analysis, adaptation options A/B/C, the `motion-seg/motion_seg/` implementation and results |
| [omniverse-pipeline.md](omniverse-pipeline.md) | Synthetic-data testing pipeline: Isaac Sim capture, pump asset preparation, conversion to 4DGS format, bugs found and fixed |
| [orchestrator.md](orchestrator.md) | The three-layer orchestration system (DAG execution module, MCP server, Streamlit UI), task history T01–T17, real-hardware debugging saga, end-to-end milestone |
| [memory-dump.md](memory-dump.md) | Dump of the assistant's persistent project memory (Cowork-tooling-specific entries excluded) |
| [viewer_usage.md](viewer_usage.md) | (pre-existing) Using the SIBR viewer with 4DGS |

Side note: `.claude_notes/NOTES_viseron_setup.md` documents a Viseron NVR setup unrelated to the thesis; it is not covered here.
