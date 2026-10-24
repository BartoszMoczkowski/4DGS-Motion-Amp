"""Archive redundant batch drivers into scripts/archived with clear deprecation headers."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARCHIVED_DIR = SCRIPTS_DIR / "archived"
ARCHIVED_DIR.mkdir(parents=True, exist_ok=True)

DRIVERS = [
    ("batch_capture_series_a.py", "experiments/capture_limitation.yaml"),
    ("batch_capture_series_r.py", "experiments/capture_limitation.yaml"),
    ("batch_capture_series_tf.py", "experiments/capture_limitation.yaml"),
    ("batch_train_capture_limitation.py", "experiments/capture_limitation.yaml"),
    ("batch_seg_capture_limitation.py", "experiments/capture_limitation.yaml"),
    ("run_lego_turbine_pipeline.py", "experiments/lego_turbine.yaml"),
]

HEADER_TEMPLATE = """\"\"\"
[DEPRECATED / ARCHIVED — T33]
This ad-hoc batch driver has been retired.
Its workflow is now executed declaratively via:
    amp exp run {spec}

Retained for historical audit and reference only.
\"\"\"
"""

for fname, spec in DRIVERS:
    src = SCRIPTS_DIR / fname
    dst = ARCHIVED_DIR / fname
    if src.is_file():
        content = src.read_text(encoding="utf-8")
        # Prepend header if not already present
        if "[DEPRECATED / ARCHIVED — T33]" not in content:
            new_content = HEADER_TEMPLATE.format(spec=spec) + "\n" + content
        else:
            new_content = content
        dst.write_text(new_content, encoding="utf-8")
        src.unlink()
        print(f"Archived {fname} -> scripts/archived/{fname}")
    elif dst.is_file():
        print(f"Already archived: scripts/archived/{fname}")
