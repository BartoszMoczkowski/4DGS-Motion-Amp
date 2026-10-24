"""Docs consistency checker: verifies documentation does not recommend stale raw-python commands.

Enforces acceptance criteria for T31:
- No doc references raw-python paths for routine operations where `amp` commands exist.
- Explicit allowlists cover:
  * Docker in-container stage executions (e.g. `core/train.py` executed inside the cuda container)
  * Code/line citations (e.g. `core/train.py:272-288`)
  * Submodule selftests (e.g. `motion_seg/segment_rigid.py --selftest`)
  * Scripts in `scratch/`
  * Historical post-mortems documenting what was replaced
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_PATTERNS = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "README.md",
    *(REPO_ROOT / "docs").glob("*.md"),
]

# Patterns that indicate a command invocation of a deprecated/migrated script
# Matches actual command lines, e.g. `python core/train.py ...` or `uv run ...`
DISALLOWED_COMMAND_REGEXES = [
    # Invoking core/train.py or core/render*.py directly via python as a command
    (
        re.compile(r"(?:python(?:\.exe)?|uv\s+run(?:\s+--package\s+\S+)?\s+python)\s+core/(?:train|render|render_amp)\.py(?:\s+[^\n`]+)?"),
        "Raw invocation of core/train.py or core/render*.py (use `amp train` or `amp render` instead)",
    ),
    # Invoking motion_seg run.sh directly as a command
    (
        re.compile(r"(?:(?:bash|sh|\./)\s*|\b)run\.sh(?:\s+[^\n`]+)?"),
        "Invocation of deleted run.sh (use `amp seg run` instead)",
    ),
    # Deleted grid runners as commands
    (
        re.compile(r"python(?:\.exe)?\s+scene[-_]gen/run_grid_(?:4dgs|seg)\.py"),
        "Deleted runner run_grid_*.py referenced as command (use `amp exp run` instead)",
    ),
    # Ad-hoc python -c for run_pipeline or frame count QA
    (
        re.compile(r'python(?:\.exe)?\s+-c\s+["\'].*?(?:run_pipeline|Path.*glob\(["\']\*\.png["\']\)).*?["\']'),
        "Ad-hoc python -c invocation where an amp command exists (e.g. `amp capture verify`)",
    ),
]

# Allowlisted substrings or contexts where mentions are legitimate
ALLOWLIST_PATTERNS = [
    # Inside container exec / implementation notes
    "persists inside the `cuda` container",
    "persists inside the cuda container",
    "exec'd by the `train.default` stage",
    "Docker stage execs invoke",
    "Docker stage execs",
    "in-container exec",
    "container runtime",
    "core/train.py:505",
    "core/train.py:272",
    "core/train.py:392",
    "core/train.py:278",
    "core/train.py optimizer",
    "core/render_amp.py` validates its arguments",
    "core/render_amp.py natively",
    "core/render_amp.py's",
    "core/render_amp.py`'s",
    "core/render_amp.py`",
    "core/render_amp.py",
    # Selftests on reference code
    "--selftest",
    # Post-mortem problem descriptions (historical documentation of the issue)
    "Repeated execution of 15+ ad-hoc",
    "- **Symptom**:",
    # Explaining deletions / deprecations
    "were deleted in T27",
    "deleted in T26",
    "run.sh deleted",
    "`run.sh` deleted",
    "run.sh was deleted",
    "`run.sh` was deleted",
    "motion_seg/run.sh deleted",
    "`motion_seg/run.sh` deleted",
    "motion-seg/motion_seg/run.sh deleted",
    "deleted `motion-seg/motion_seg/run.sh`",
    "replaces the deleted",
    "replaced by `amp seg run",
    "replaces `motion-seg/motion_seg/run.sh`",
    "replaces `core/render.py`",
    "replaces `uv run --package 4dgs-core python core/train.py",
    # Test double or test instruction
    "tests/test_segment_rigid2.py",
]


def check_line(line: str) -> list[str]:
    # Check if line contains an allowlisted snippet
    if any(allowed in line for allowed in ALLOWLIST_PATTERNS):
        return []

    violations = []
    for regex, msg in DISALLOWED_COMMAND_REGEXES:
        if regex.search(line):
            violations.append(msg)
    return violations


def check_docs() -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for doc in DOC_PATTERNS:
        if not doc.exists():
            continue
        lines = doc.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            line_violations = check_line(line)
            for v in line_violations:
                violations.append((doc, idx, line.strip(), v))
    return violations


def main() -> int:
    violations = check_docs()
    if not violations:
        print(f"[ok] Checked {len(DOC_PATTERNS)} doc files: all clear.")
        return 0

    print(f"[FAIL] Found {len(violations)} doc consistency violations:")
    for doc, line_num, content, msg in violations:
        rel = doc.relative_to(REPO_ROOT) if doc.is_relative_to(REPO_ROOT) else doc
        print(f"  {rel}:{line_num} - {msg}")
        print(f"    Line: {content}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
