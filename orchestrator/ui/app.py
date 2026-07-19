"""T15 — a thin Streamlit panel over Layer 1's ``pipeline.api``.

Views (per ``planning/tasks/T15-ui.md``'s scope): pick/edit a preset (folds in ``ampUI.py``'s
amp-parameter panel), launch a run and watch per-stage progress/logs, browse artifacts/previews,
compare two runs. No pipeline logic lives here — every action calls into ``layer1_client``, which
itself only calls ``pipeline.api`` (+ the same read-only helpers Layer 2's MCP server uses).

Run with (see ``README.md`` for the full setup):

    cd orchestrator && streamlit run ui/app.py
"""

from __future__ import annotations

import time
from typing import Any, Optional

import streamlit as st

import layer1_client as client  # local module; also bootstraps `pipeline`/`mcp_server` on sys.path

from pipeline.config import AMP_CHANNELS, AMP_METHOD_ALIASES  # noqa: E402 (after path bootstrap)

_METHOD_LABEL_BY_VALUE = {v: k for k, v in AMP_METHOD_ALIASES.items()}


st.set_page_config(page_title="4DGS Motion-Amp Orchestrator", layout="wide")
st.title("4DGS Motion-Amp Orchestrator")
st.caption(
    "Thin UI over Layer 1 (`pipeline.api`) — direct in-process import, no MCP/HTTP hop involved. "
    "See `ui/README.md` for why."
)


# --- shared helpers ------------------------------------------------------------------------------


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _diff_configs(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    fa, fb = _flatten(a), _flatten(b)
    rows = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, "<missing>"), fb.get(key, "<missing>")
        if va != vb:
            rows.append({"key": key, "run_a": va, "run_b": vb})
    return rows


def _render_status(run_id: str, *, key_prefix: str) -> None:
    """Per-stage status table + job_error + log-tailing/cancel controls for one run."""
    try:
        status = client.get_status(run_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Couldn't fetch status for {run_id!r}: {exc}")
        return

    if status.get("job_error"):
        st.error(f"Background job error (run never got past its first stage):\n{status['job_error']}")

    st.write(f"**status:** {status['status']} · **preset:** {status['preset']} · **updated:** {status['updated_at']}")

    stages = status.get("stages", {})
    if stages:
        rows = [{"stage": name, **rec} for name, rec in stages.items()]
        st.dataframe(rows, use_container_width=True, key=f"{key_prefix}_stage_table")
    else:
        st.info("No stages recorded yet.")

    col1, col2 = st.columns([2, 1])
    stage_names = list(stages.keys())
    if stage_names:
        stage_pick = col1.selectbox(
            "Tail logs for stage", stage_names, key=f"{key_prefix}_log_stage"
        )
        if col1.button("Tail last 200 lines", key=f"{key_prefix}_tail_btn"):
            try:
                log = client.tail_logs(run_id, stage_pick, max_lines=200)
                st.code("\n".join(log["lines"]) or "(empty log)", language="text")
                if log["truncated"]:
                    st.caption(f"showing last 200 of {log['line_count']} lines — {log['path']}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Couldn't read log: {exc}")

    if col2.button("Cancel run (best-effort)", key=f"{key_prefix}_cancel_btn"):
        try:
            result = client.cancel_run(run_id)
            (st.success if result.get("cancelled") else st.warning)(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cancel failed: {exc}")


tab_presets, tab_launch, tab_runs, tab_compare, tab_machine = st.tabs(
    ["Presets", "Launch & Monitor", "Runs & Artifacts", "Compare Runs", "GPU / Containers"]
)


# --- Presets --------------------------------------------------------------------------------


with tab_presets:
    st.subheader("Pick / validate a preset")
    try:
        presets = client.list_presets()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't list presets: {exc}")
        presets = []

    preset: Optional[str] = st.selectbox("Preset", presets, key="preset_select") if presets else None

    if preset and st.button("Validate / resolve", key="validate_btn"):
        try:
            st.session_state["resolved_cfg"] = client.validate_config(preset)
            st.session_state["resolved_cfg_preset"] = preset
        except Exception as exc:  # noqa: BLE001
            st.error(f"{preset!r} failed to validate: {exc}")
            st.session_state.pop("resolved_cfg", None)

    resolved = (
        st.session_state.get("resolved_cfg")
        if st.session_state.get("resolved_cfg_preset") == preset
        else None
    )

    if resolved:
        with st.expander("Full resolved config (JSON)"):
            st.json(resolved)

        st.markdown("#### Amplification-parameter panel (folded in from `ampUI.py`)")
        amp = resolved.get("amp", {})
        channels_cfg = amp.get("channels", {})

        current_method_label = _METHOD_LABEL_BY_VALUE.get(amp.get("method", "eulerian"), "base")
        # Widget keys are namespaced by `preset` (not just the field name) — Streamlit ignores a
        # widget's `value=` on every rerun after its key first appears in session_state, so without
        # this, switching the Preset dropdown above would keep showing the *previous* preset's amp
        # values instead of the newly-resolved one.
        method_label = st.selectbox(
            "Method", list(AMP_METHOD_ALIASES.keys()),
            index=list(AMP_METHOD_ALIASES.keys()).index(current_method_label),
            key=f"amp_method_{preset}",
        )

        channels_payload: dict[str, dict[str, float]] = {}
        for ch in AMP_CHANNELS:
            cfg = channels_cfg.get(ch, {})
            c1, c2, c3 = st.columns(3)
            factor = c1.number_input(
                f"{ch} — amp factor", min_value=-1.0, max_value=100.0,
                value=float(cfg.get("factor", -1.0)), step=0.01, key=f"amp_factor_{preset}_{ch}",
            )
            freq_low = c2.number_input(
                f"{ch} — freq low cutoff", min_value=0.0, max_value=100.0,
                value=float(cfg.get("freq_low", 0.0)), key=f"amp_flow_{preset}_{ch}",
            )
            freq_high = c3.number_input(
                f"{ch} — freq high cutoff", min_value=0.0, max_value=100.0,
                value=float(cfg.get("freq_high", 1.0)), key=f"amp_fhigh_{preset}_{ch}",
            )
            channels_payload[ch] = {"factor": factor, "freq_low": freq_low, "freq_high": freq_high}

        st.markdown("#### Save as a new preset")
        st.caption(
            "Writes a new `pipeline/config/presets/<name>.yaml` with `extends: <this preset>` "
            "plus the amp overrides above — config, not code, per INSTRUCTIONS.md's rule."
        )
        new_name = st.text_input("New preset name", key="new_preset_name")
        overwrite = st.checkbox("Overwrite if it already exists", key="new_preset_overwrite")
        if st.button("Save preset variant", key="save_preset_btn"):
            if not new_name:
                st.warning("Give the new preset a name first.")
            else:
                try:
                    path = client.save_preset_variant(
                        new_name,
                        extends=preset,
                        amp_method=AMP_METHOD_ALIASES[method_label],
                        amp_channels=channels_payload,
                        overwrite=overwrite,
                    )
                    st.success(f"Saved {path}. Reselect the Preset dropdown above to see it.")
                except FileExistsError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Couldn't save preset: {exc}")


# --- Launch & Monitor ----------------------------------------------------------------------


with tab_launch:
    st.subheader("Launch a run")
    try:
        launch_presets = client.list_presets()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't list presets: {exc}")
        launch_presets = []

    launch_preset = (
        st.selectbox("Preset", launch_presets, key="launch_preset") if launch_presets else None
    )

    lc1, lc2 = st.columns(2)
    from_stage = lc1.text_input("from_stage (optional)", key="from_stage") or None
    to_stage = lc2.text_input("to_stage (optional)", key="to_stage") or None
    only_raw = st.text_input("only these stages (comma-separated, optional)", key="only_stages")
    force = st.checkbox("force (ignore cache)", key="force_run")

    with st.expander("External artifacts (advanced — e.g. prep_split.default's raw_mesh)"):
        raw_mesh_path = st.text_input(
            "raw_mesh path (leave blank unless this preset's DAG needs it)", key="raw_mesh_path"
        )

    if st.button("Launch run", type="primary", key="launch_btn", disabled=not launch_preset):
        try:
            external = None
            if raw_mesh_path:
                from pipeline.artifacts import Artifact

                external = {
                    "raw_mesh": Artifact(
                        name="raw_mesh", kind="usd", path=raw_mesh_path, producing_stage="external"
                    )
                }
            only = [s.strip() for s in only_raw.split(",") if s.strip()] or None
            run_id = client.start_pipeline_run(
                launch_preset,
                external_artifacts=external,
                from_stage=from_stage,
                to_stage=to_stage,
                only=only,
                force=force,
            )
            st.session_state["last_run_id"] = run_id
            st.success(f"Launched run_id={run_id} (running in the background — see Monitor below).")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Launch failed: {exc}")

    st.divider()
    st.subheader("Monitor a run")
    monitor_run_id = st.text_input(
        "run_id", value=st.session_state.get("last_run_id", ""), key="monitor_run_id"
    )
    auto_refresh = st.checkbox("Auto-refresh every 5s", key="auto_refresh")
    if monitor_run_id:
        _render_status(monitor_run_id, key_prefix="monitor")
        if auto_refresh:
            time.sleep(5)
            st.rerun()


# --- Runs & Artifacts ------------------------------------------------------------------------


with tab_runs:
    st.subheader("All runs")
    try:
        runs = client.list_runs()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't list runs: {exc}")
        runs = []

    if runs:
        st.dataframe(runs, use_container_width=True, key="runs_table")
    else:
        st.info("No runs yet — launch one from the Launch & Monitor tab.")

    run_ids = [r["run_id"] for r in runs]
    inspect_run = st.selectbox("Inspect run", run_ids, key="inspect_run") if run_ids else None

    if inspect_run:
        _render_status(inspect_run, key_prefix="inspect")

        st.markdown("#### Artifacts")
        try:
            artifacts = client.list_artifacts(inspect_run)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't list artifacts: {exc}")
            artifacts = []

        if artifacts:
            st.dataframe(artifacts, use_container_width=True, key="artifacts_table")
            art_names = [a["name"] for a in artifacts]
            art_sel = st.selectbox("Preview / summarize artifact", art_names, key="inspect_artifact")
            if art_sel:
                try:
                    preview = client.artifact_preview_info(inspect_run, art_sel)
                    if preview["kind"] == "image":
                        st.image(preview["path"], caption=art_sel)
                    elif preview["kind"] == "video":
                        st.video(preview["path"])
                    summary = client.read_artifact_summary(inspect_run, art_sel)
                    with st.expander("Artifact summary (JSON)"):
                        st.json(summary)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Couldn't load artifact: {exc}")
        else:
            st.info("No artifacts produced yet.")


# --- Compare Runs -----------------------------------------------------------------------------


with tab_compare:
    st.subheader("Compare two runs")
    try:
        cmp_runs = client.list_runs()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't list runs: {exc}")
        cmp_runs = []

    cmp_ids = [r["run_id"] for r in cmp_runs]
    if len(cmp_ids) < 2:
        st.info("Need at least two runs to compare.")
    else:
        cc1, cc2 = st.columns(2)
        run_a = cc1.selectbox("Run A", cmp_ids, index=0, key="cmp_a")
        run_b = cc2.selectbox("Run B", cmp_ids, index=1, key="cmp_b")

        col_a, col_b = st.columns(2)
        for col, rid in ((col_a, run_a), (col_b, run_b)):
            with col:
                st.markdown(f"**{rid}**")
                try:
                    status = client.get_status(rid)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Couldn't fetch {rid}: {exc}")
                    continue
                st.write(f"status: {status['status']} · preset: {status['preset']}")
                rows = [{"stage": name, **rec} for name, rec in status.get("stages", {}).items()]
                if rows:
                    st.dataframe(rows, use_container_width=True, key=f"cmp_stages_{rid}")
                try:
                    arts = client.list_artifacts(rid)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Couldn't list artifacts for {rid}: {exc}")
                    arts = []
                for art in arts:
                    if art.get("kind") == "png":
                        try:
                            preview = client.artifact_preview_info(rid, art["name"])
                            st.image(preview["path"], caption=art["name"])
                        except Exception:  # noqa: BLE001 - best-effort preview, don't block compare
                            pass

        st.markdown("#### Resolved-config diff")
        try:
            cfg_a = client.get_resolved_config(run_a)
            cfg_b = client.get_resolved_config(run_b)
            diff_rows = _diff_configs(cfg_a, cfg_b)
            if diff_rows:
                st.dataframe(diff_rows, use_container_width=True, key="cmp_diff_table")
            else:
                st.info("Resolved configs are identical.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't diff configs: {exc}")


# --- GPU / Containers --------------------------------------------------------------------------


with tab_machine:
    st.subheader("GPU / RAM")
    try:
        st.json(client.gpu_status())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't fetch GPU status: {exc}")

    st.subheader("Containers")
    try:
        containers = client.list_containers()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't list containers: {exc}")
        containers = []

    if containers:
        st.dataframe(containers, use_container_width=True, key="containers_table")
    else:
        st.info("No managed containers.")

    mc1, mc2 = st.columns(2)
    start_env = mc1.selectbox("Environment", ["cuda", "isaac"], key="start_env")
    if mc1.button("Start / warm container", key="start_container_btn"):
        try:
            container_id = client.start_container(start_env)
            st.success(f"container: {container_id}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Start failed: {exc}")

    stop_id = mc2.text_input("Container id to stop", key="stop_container_id")
    if mc2.button("Stop container", key="stop_container_btn"):
        if not stop_id:
            st.warning("Enter a container id first.")
        else:
            try:
                client.stop_container(stop_id)
                st.success("stopped")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Stop failed: {exc}")
