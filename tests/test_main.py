"""Contract tests for voicecoding.main.

Groups: the CLI exposes prog 'voicecoding' with the run/doctor/deploy
subcommands; run wires into VoiceMouseApp; doctor/deploy route into their
modules (targets patched wherever the dispatcher resolves them, covering
both top-level and lazy imports); the package version is 1.0.0; and no
package module may import a heavy library at import time (the lazy-import
rule that keeps this lean-environment suite runnable).
"""
from __future__ import annotations

import importlib

import pytest

from conftest import (
    FakeInputStream,
    FakeKeyboardListener,
    FakeMouseListener,
    FakeWhisperModel,
)

PACKAGE_MODULES = (
    "config",
    "audio",
    "transcriber",
    "mouse_listener",
    "keyboard_listener",
    "output",
    "system_integration",
    "doctor",
    "deploy",
    "app",
    "main",
)

HEAVY_NAMES = {
    "faster_whisper",
    "WhisperModel",
    "ctranslate2",
    "sounddevice",
    "soundfile",
    "pynput",
}


class _RecordingStub:
    """Callable/class hybrid recording every invocation path.

    Works as a replacement for entry-point functions such as run_doctor and
    run_deploy: calls are recorded and the conventional process exit code 0
    is returned, and attribute access yields further stubs so dispatchers
    can probe it.
    """

    def __init__(self, calls: list, label: str) -> None:
        self._calls = calls
        self._label = label

    def __call__(self, *args, **kwargs):
        self._calls.append(f"{self._label}()")
        return 0

    def run(self):
        self._calls.append(f"{self._label}.run")
        return 0

    def __iter__(self):
        return iter([])

    def __getattr__(self, name: str):
        return _RecordingStub(self._calls, f"{self._label}.{name}")


def _patch_entry(monkeypatch: pytest.MonkeyPatch, module, names, calls, label):
    """Replace existing callables on module with recording stubs.

    Returns the names actually patched so tests can assert a target existed
    to mock, independently of whether it was invoked.
    """
    patched: list[str] = []
    for name in names:
        if callable(getattr(module, name, None)):
            monkeypatch.setattr(
                module, name, _RecordingStub(calls, f"{label}.{name}")
            )
            patched.append(name)
    return patched


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_package_version_is_1_0_0() -> None:
    import voicecoding

    assert voicecoding.__version__ == "1.0.0"


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


def test_help_prints_prog_and_subcommands(monkeypatch, capsys) -> None:
    import voicecoding.main as main_module

    with pytest.raises(SystemExit) as excinfo:
        main_module.main(["--help"])

    assert excinfo.value.code in (0, None, "")
    out = capsys.readouterr().out
    assert "voicecoding" in out
    for subcommand in ("run", "doctor", "deploy"):
        assert subcommand in out


# ---------------------------------------------------------------------------
# Subcommand wiring
# ---------------------------------------------------------------------------


def test_run_subcommand_builds_and_runs_app(monkeypatch) -> None:
    import voicecoding.app as app_module
    import voicecoding.main as main_module

    calls: list[str] = []

    class StubApp:
        def __init__(self, *args, **kwargs) -> None:
            calls.append("app-init")

        def run(self):
            calls.append("app-run")
            return 0

    monkeypatch.setattr(main_module, "VoiceMouseApp", StubApp, raising=False)
    monkeypatch.setattr(app_module, "VoiceMouseApp", StubApp, raising=False)

    result = main_module.main(["run"])

    assert "app-init" in calls
    assert "app-run" in calls
    assert result in (0, None)


def test_doctor_subcommand_routes_into_doctor_module(monkeypatch) -> None:
    import voicecoding.doctor as doctor_module
    import voicecoding.main as main_module

    calls: list[str] = []
    patched = _patch_entry(
        monkeypatch,
        main_module,
        ("run_doctor", "cmd_doctor", "Doctor"),
        calls,
        "main",
    )
    patched += _patch_entry(
        monkeypatch,
        doctor_module,
        ("run_doctor", "cmd_doctor", "Doctor"),
        calls,
        "doctor",
    )
    assert patched, "no doctor entry point found to mock"

    app_calls: list[str] = []

    class StubApp:
        def __init__(self, *args, **kwargs) -> None:
            app_calls.append("init")

    monkeypatch.setattr(main_module, "VoiceMouseApp", StubApp, raising=False)

    result = main_module.main(["doctor"])

    assert calls, "the doctor dispatcher never invoked the mocked target"
    assert app_calls == [], "doctor subcommand must not start the app"
    assert result in (0, None)


def test_deploy_subcommand_routes_into_deploy_module(monkeypatch) -> None:
    import voicecoding.deploy as deploy_module
    import voicecoding.main as main_module

    calls: list[str] = []
    patched = _patch_entry(
        monkeypatch,
        main_module,
        ("run_deploy", "cmd_deploy"),
        calls,
        "main",
    )
    patched += _patch_entry(
        monkeypatch,
        deploy_module,
        ("main", "run", "cli", "run_deploy", "cmd_deploy"),
        calls,
        "deploy",
    )
    assert patched, "no deploy entry point found to mock"

    app_calls: list[str] = []

    class StubApp:
        def __init__(self, *args, **kwargs) -> None:
            app_calls.append("init")

    monkeypatch.setattr(main_module, "VoiceMouseApp", StubApp, raising=False)

    result = main_module.main(["deploy"])

    assert calls, "the deploy dispatcher never invoked the mocked target"
    assert app_calls == [], "deploy subcommand must not start the app"
    assert result in (0, None)


# ---------------------------------------------------------------------------
# Lazy-import rule
# ---------------------------------------------------------------------------


def test_package_modules_import_without_heavy_libraries() -> None:
    import voicecoding

    assert voicecoding.__package__ == "voicecoding"

    for name in PACKAGE_MODULES:
        module = importlib.import_module(f"voicecoding.{name}")
        leaked = HEAVY_NAMES & set(vars(module))
        assert not leaked, (
            f"voicecoding.{name} binds heavy libraries at import time: "
            f"{sorted(leaked)}"
        )

    # No fake constructor may have run during a plain import either.
    assert FakeWhisperModel.instances == []
    assert FakeInputStream.instances == []
    assert FakeMouseListener.instances == []
    assert FakeKeyboardListener.instances == []
