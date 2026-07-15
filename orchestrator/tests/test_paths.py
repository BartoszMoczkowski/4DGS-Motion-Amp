"""Tests for T06 (path-translation module) — revised for the 2-space (host/container) model
after the runtime host moved off WSL2 (see ``pipeline/paths.py``'s own module docstring and
``.claude_notes/NOTES_pipeline_orchestration.md``).

Table-driven over representative (host, container) pairs for both roots (repo, assets), each
checked pairwise in every direction, plus dedicated coverage for backslash-tolerant/case-
insensitive host-path *input* (real Windows users type these) and the mount-spec builder.

Host-side results are compared as ``Path`` objects, not raw strings — ``Path``'s own equality is
structural (flavour-aware), so the same assertion holds whether this file actually runs on Windows
(the real target) or this sandbox's Linux (str() would otherwise render with the "wrong"
separator on whichever OS didn't construct the literal). Container-side results and
``MountSpec.source`` stay plain string comparisons since those are always POSIX / explicitly
forward-slash-normalized regardless of host OS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.paths import (
    MountSpec,
    Roots,
    container_mounts,
    get_roots,
    to_container,
    to_host,
)

ROOTS = Roots(
    repo_root_host=Path("C:/Users/bartosz/Code/4DGS-Motion-Amp"),
    assets_root_host=Path("Q:/Omniverse"),
)

# (host, container) pairs covering both roots, root-level and nested. Host forms use forward
# slashes -- Windows accepts "/" as an alternate separator, so this stays meaningful (and
# joinable via pathlib) regardless of which OS actually runs the test.
CASES = [
    ("C:/Users/bartosz/Code/4DGS-Motion-Amp", "/workspace"),
    (
        "C:/Users/bartosz/Code/4DGS-Motion-Amp/orchestrator/pipeline/paths.py",
        "/workspace/orchestrator/pipeline/paths.py",
    ),
    ("Q:/Omniverse", "/omniverse"),
    (
        "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd",
        "/omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd",
    ),
    ("Q:/Omniverse/renders/capture_pump", "/omniverse/renders/capture_pump"),
]


@pytest.mark.parametrize("host,container", CASES)
def test_to_host_from_either_space(host, container):
    assert to_host(host, roots=ROOTS) == Path(host)
    assert to_host(container, roots=ROOTS) == Path(host)


@pytest.mark.parametrize("host,container", CASES)
def test_to_container_from_either_space(host, container):
    assert str(to_container(host, roots=ROOTS)) == container
    assert str(to_container(container, roots=ROOTS)) == container


@pytest.mark.parametrize("host,container", CASES)
def test_to_container_accepts_env(host, container):
    assert str(to_container(host, "cuda", roots=ROOTS)) == container
    assert str(to_container(host, "isaac", roots=ROOTS)) == container


@pytest.mark.parametrize("host,container", CASES)
def test_round_trip_host_container_host(host, container):
    assert to_host(to_container(host, roots=ROOTS), roots=ROOTS) == Path(host)


@pytest.mark.parametrize("host,container", CASES)
def test_round_trip_container_host_container(host, container):
    assert str(to_container(to_host(container, roots=ROOTS), roots=ROOTS)) == container


def test_to_host_accepts_backslash_input():
    # Real Windows users type backslash paths; matching is backslash-tolerant even though the
    # roots themselves were constructed with forward slashes.
    result = to_host("C:\\Users\\bartosz\\Code\\4DGS-Motion-Amp\\orchestrator", roots=ROOTS)
    assert result == Path("C:/Users/bartosz/Code/4DGS-Motion-Amp/orchestrator")


def test_to_host_matching_is_case_insensitive():
    result = to_host("c:/users/bartosz/code/4dgs-motion-amp/orchestrator", roots=ROOTS)
    assert result == Path("C:/Users/bartosz/Code/4DGS-Motion-Amp/orchestrator")


def test_unknown_root_raises():
    with pytest.raises(ValueError):
        to_host("/some/unrelated/path", roots=ROOTS)
    with pytest.raises(ValueError):
        to_host("D:\\Elsewhere\\file.txt", roots=ROOTS)


def test_to_container_rejects_bad_env():
    with pytest.raises(ValueError):
        to_container("C:/Users/bartosz/Code/4DGS-Motion-Amp", "not-an-env", roots=ROOTS)


def test_roots_host_properties_match_construction():
    assert ROOTS.repo_root_host == Path("C:/Users/bartosz/Code/4DGS-Motion-Amp")
    assert ROOTS.assets_root_host == Path("Q:/Omniverse")


def test_container_mounts_shape_matches_devcontainer_convention():
    mounts = container_mounts("cuda", roots=ROOTS)
    assert mounts == [
        MountSpec(source="C:/Users/bartosz/Code/4DGS-Motion-Amp", target="/workspace"),
        MountSpec(source="Q:/Omniverse", target="/omniverse"),
    ]
    assert (
        mounts[1].as_docker_mount_string()
        == "source=Q:/Omniverse,target=/omniverse,type=bind,consistency=cached"
    )


def test_container_mounts_rejects_bad_env():
    with pytest.raises(ValueError):
        container_mounts("not-an-env", roots=ROOTS)


def test_get_roots_env_var_overrides(monkeypatch):
    monkeypatch.setenv("PIPELINE_REPO_ROOT", "D:/repo")
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", "E:/Omniverse")
    roots = get_roots()
    assert roots.repo_root_host == Path("D:/repo")
    assert roots.assets_root_host == Path("E:/Omniverse")


def test_get_roots_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("PIPELINE_REPO_ROOT", raising=False)
    monkeypatch.delenv("PIPELINE_ASSETS_ROOT", raising=False)
    roots = get_roots()
    assert roots.assets_root_host == Path("Q:/Omniverse")
