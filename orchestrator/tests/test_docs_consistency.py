"""Test documentation consistency (T31).

Ensures that documentation (AGENTS.md, README.md, docs/*.md):
- Does not recommend raw-python command invocations for operations where `amp` commands exist.
- Documents the unified run-ID convention and canonical `runs/` layout.
- Keeps gotchas and command references consistent with current reality.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is 2 levels up from orchestrator/tests
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_docs_consistency import check_docs


def test_no_raw_python_command_references_in_docs() -> None:
    """Acceptance criterion: No doc references raw-python paths for routine operations."""
    violations = check_docs()
    if violations:
        msg = "\n".join(
            f"{doc.relative_to(REPO_ROOT)}:{line_num} - {err} (Line: {line})"
            for doc, line_num, line, err in violations
        )
        assert False, f"Found {len(violations)} doc consistency violations:\n{msg}"


def test_agents_md_documents_amp_commands_and_canonical_roots() -> None:
    """Acceptance criterion: AGENTS.md documents amp commands, run ID convention, and runs/ root."""
    agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "amp explore" in agents_md
    assert "amp train" in agents_md
    assert "amp render" in agents_md
    assert "amp seg" in agents_md
    assert "amp exp run" in agents_md
    assert "runs/<run_id>/" in agents_md
    assert "<experiment>-<cell>-<8hex>" in agents_md
    assert "migration_legacy_output.md" in agents_md


def test_readme_quickstart_walkthrough() -> None:
    """Acceptance criterion: Fresh-clone walkthrough documented in README.md."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv sync" in readme
    assert "uv run amp --help" in readme
    assert "uv run amp exp run" in readme
    assert "uv run amp mlflow serve" in readme
    assert "uv run amp explore" in readme


def test_legacy_output_migration_doc_and_script_exist() -> None:
    """Acceptance criterion: Legacy-dir migration note and utility exist."""
    assert (REPO_ROOT / "docs" / "migration_legacy_output.md").exists()
    assert (REPO_ROOT / "scripts" / "migrate_legacy_output.py").exists()
