"""T25: `amp` CLI skeleton — group listing + lazy-import discipline.

Runs in a clean subprocess: pytest's own process is polluted by conftest and
other test modules, so heavy-module absence can only be asserted on a fresh
interpreter.
"""

import subprocess
import sys
import textwrap

HEAVY_MODULES = ["torch", "docker", "mcp", "streamlit", "cv2", "numpy"]
GROUPS = [
    "capture",
    "train",
    "render",
    "seg",
    "exp",
    "orch",
    "bench",
    "mlflow",
    "report",
]

PROBE = textwrap.dedent(
    """
    import sys
    from typer.testing import CliRunner

    from amp.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    groups = {groups!r}
    for group in groups:
        assert group in result.output, f"group {{group}} missing from amp --help"
        sub = runner.invoke(app, [group, "--help"])
        assert sub.exit_code == 0, f"amp {{group}} --help failed: {{sub.output}}"

    heavy = {heavy!r}
    loaded = [m for m in heavy if m in sys.modules]
    assert not loaded, f"heavy modules imported by amp --help: {{loaded}}"
    print("AMP CLI PROBE OK")
    """
).format(groups=GROUPS, heavy=HEAVY_MODULES)


def test_amp_help_lists_groups_without_heavy_imports():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "AMP CLI PROBE OK" in result.stdout
