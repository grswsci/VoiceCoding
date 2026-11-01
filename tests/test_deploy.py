"""Contract tests for voicecoding.deploy.

Groups: render_run_bat content (env prefix, python exe, '-m voicecoding.main
run'), render_startup_vbs content (lnk + bat paths), --dry-run writing
nothing, preset profiles mapping to VOICECODING_* variables, and autostart
via schtasks command construction only. The public entry point is
run_deploy(args) fed through configure_deploy_parser; every subprocess call
is mocked and APPDATA points into tmp_path so nothing outside the test tree
is touched.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fake_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            calls.append([str(part) for part in cmd])
        else:
            calls.append([str(cmd)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


def _point_appdata_at_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def _run_deploy(argv: list[str]):
    import voicecoding.deploy as deploy_mod

    parser = argparse.ArgumentParser(prog="voicecoding deploy")
    deploy_mod.configure_deploy_parser(parser)
    args = parser.parse_args(list(argv))
    return deploy_mod.run_deploy(args)


def _preset_registry() -> dict:
    import voicecoding.deploy as deploy_mod

    return deploy_mod._PRESET_OVERRIDES


def _run_bat_path(appdata: Path) -> Path:
    return appdata / "voicecoding" / "run.bat"


# ---------------------------------------------------------------------------
# render_run_bat
# ---------------------------------------------------------------------------


def test_render_run_bat_contains_env_prefix_exe_and_module_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voicecoding.deploy as deploy_mod

    env_overrides = dict(_preset_registry()["stable"])
    exe = str(tmp_path / "venv" / "Scripts" / "python.exe")

    text = deploy_mod.render_run_bat(exe, env_overrides)

    assert "VOICECODING_" in text
    assert exe in text
    assert "-m voicecoding.main run" in text
    for key in env_overrides:
        assert key in text


# ---------------------------------------------------------------------------
# render_startup_vbs
# ---------------------------------------------------------------------------


def test_render_startup_vbs_references_bat_and_lnk_paths(
    tmp_path: Path,
) -> None:
    import voicecoding.deploy as deploy_mod

    bat_path = tmp_path / "voicecoding" / "run.bat"
    lnk_path = (
        tmp_path
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / "voicecoding.lnk"
    )

    text = deploy_mod.render_startup_vbs(str(bat_path), str(lnk_path))

    assert str(bat_path) in text
    assert str(lnk_path) in text


# ---------------------------------------------------------------------------
# Deploy run / dry-run
# ---------------------------------------------------------------------------


def test_full_deploy_writes_run_bat_and_creates_shortcut_via_cscript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_appdata_at_tmp(monkeypatch, tmp_path)
    calls = _fake_subprocess_run(monkeypatch)
    # The cscript invocation only happens on the Windows branch.
    monkeypatch.setattr(sys, "platform", "win32")

    _run_deploy([])

    run_bat = _run_bat_path(tmp_path)
    assert run_bat.is_file()
    text = run_bat.read_text(encoding="utf-8", errors="replace")
    assert "VOICECODING_" in text
    assert sys.executable in text
    assert "-m voicecoding.main run" in text

    assert any("cscript" in str(call).lower() for call in calls)


def test_dry_run_writes_nothing_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_appdata_at_tmp(monkeypatch, tmp_path)
    calls = _fake_subprocess_run(monkeypatch)

    _run_deploy(["--dry-run"])

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert calls == []


def test_schtasks_autostart_only_builds_scheduled_task_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_appdata_at_tmp(monkeypatch, tmp_path)
    calls = _fake_subprocess_run(monkeypatch)
    # The schtasks invocation only happens on the Windows branch.
    monkeypatch.setattr(sys, "platform", "win32")

    _run_deploy(["--autostart", "schtasks"])

    assert _run_bat_path(tmp_path).is_file()
    schtasks_calls = [call for call in calls if "schtasks" in str(call).lower()]
    assert schtasks_calls, "expected a schtasks command to be constructed"
    assert all(
        "schtasks" in str(call).lower() for call in calls
    ), f"unexpected non-schtasks subprocess calls: {calls}"


def test_skip_autostart_writes_bat_without_shortcut_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_appdata_at_tmp(monkeypatch, tmp_path)
    calls = _fake_subprocess_run(monkeypatch)

    _run_deploy(["--skip-autostart"])

    assert _run_bat_path(tmp_path).is_file()
    assert calls == []


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_map_to_voicecoding_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = _preset_registry()
    name = next(iter(registry))
    payload = registry[name]

    _point_appdata_at_tmp(monkeypatch, tmp_path)
    _fake_subprocess_run(monkeypatch)
    _run_deploy(["--preset", name])

    text = _run_bat_path(tmp_path).read_text(encoding="utf-8", errors="replace")

    if isinstance(payload, dict):
        assert payload, f"preset {name!r} carries no values"
        for key in payload:
            assert str(key).startswith("VOICECODING_"), key
            assert str(key) in text
    else:
        assert "VOICECODING_" in text
