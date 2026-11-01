"""Contract tests for voicecoding.app.VoiceMouseApp.

The app is constructed for real — every heavy dependency resolves to a
conftest fake at construction time — and then its collaborators (stub
WhisperTranscriber, fake recorder, output and integration doubles) are bound
onto the instance. ``threading.Thread`` is patched to a synchronous fake so
worker paths run deterministically. The frozen AppConfig is customized via
``dataclasses.replace``.

Groups: front button toggles recording, rear button stops and routes the
transcript through inject_or_clipboard, rear-while-idle sends Enter, the
workspace gesture reaches switch_workspace, and the status file is written
atomically as JSON.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import SyncThread, load_contract_config


class FakeRecorder:
    def __init__(self, recording) -> None:
        self.is_recording = False
        self.started = 0
        self.stopped = 0
        self._recording = recording

    def start(self) -> None:
        self.started += 1
        self.is_recording = True

    def stop_and_save(self):
        self.stopped += 1
        self.is_recording = False
        return self._recording


class StubTranscriber:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, audio_path: str) -> str:
        self.calls.append(str(audio_path))
        return "transcribed text"

    def prewarm(self) -> None:
        return None


class _OutputDouble:
    def __init__(self) -> None:
        self.injected: list[tuple[str, dict]] = []
        self.enter_calls: list[dict] = []

    def inject_or_clipboard(self, text: str, **kwargs) -> str:
        self.injected.append((text, dict(kwargs)))
        return "pasted"

    def send_enter(self, **kwargs) -> None:
        self.enter_calls.append(dict(kwargs))


@pytest.fixture()
def patch_threads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(threading, "Thread", SyncThread)
    import voicecoding.app as app_module

    if hasattr(app_module, "Thread"):
        monkeypatch.setattr(app_module, "Thread", SyncThread)


def _make_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    from voicecoding.app import VoiceMouseApp

    config = load_contract_config(monkeypatch)
    status_file = tmp_path / "status.json"
    config = replace(
        config,
        status_file=status_file,
        temp_dir=tmp_path / "recordings",
    )

    # The real constructor wires listeners, recorder and transcriber; the
    # conftest fakes keep it free of heavy backends. Collaborators that the
    # tests observe are then swapped for doubles.
    app = VoiceMouseApp(config)

    recording = SimpleNamespace(
        duration_s=1.0, path=tmp_path / "clip.wav"
    )
    recorder = FakeRecorder(recording)
    setattr(app, "_recorder", recorder)
    setattr(app, "_transcriber", StubTranscriber())

    output = _OutputDouble()
    setattr(app, "_output", output)

    integration = SimpleNamespace(
        switch_workspace=lambda direction: (
            integration.workspace_calls.append(direction) or True
        )
    )
    integration.workspace_calls: list[str] = []
    setattr(app, "_system_integration", integration)

    return app, recorder, output, integration, status_file


def _gesture_handler(app):
    for name in ("_on_gesture", "_handle_gesture", "_dispatch_gesture"):
        handler = getattr(app, name, None)
        if callable(handler):
            return handler
    pytest.fail("VoiceMouseApp exposes no gesture handler")


# ---------------------------------------------------------------------------
# Front button: toggles recording
# ---------------------------------------------------------------------------


def test_front_press_starts_then_stops_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, recorder, _output, _integration, _status = _make_app(
        monkeypatch, tmp_path
    )
    front_press = getattr(app, "_on_front_press")

    front_press()
    assert recorder.started == 1
    assert recorder.is_recording is True

    front_press()
    assert recorder.stopped == 1
    assert recorder.is_recording is False


def test_front_press_writes_valid_status_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, _recorder, _output, _integration, status_file = _make_app(
        monkeypatch, tmp_path
    )

    getattr(app, "_on_front_press")()

    assert status_file.is_file()
    data = json.loads(status_file.read_text(encoding="utf-8"))
    assert any(
        "record" in str(key).lower() and value for key, value in data.items()
    )


def test_status_json_is_written_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, _recorder, _output, _integration, status_file = _make_app(
        monkeypatch, tmp_path
    )

    replace_calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _spy_replace(src, dst, *args, **kwargs):
        replace_calls.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _spy_replace)

    getattr(app, "_on_front_press")()

    assert any(dst == str(status_file) for _src, dst in replace_calls)
    assert not [
        p for p in tmp_path.iterdir() if p.is_file() and p.suffix == ".tmp"
    ]


# ---------------------------------------------------------------------------
# Rear button: stop and route the transcript
# ---------------------------------------------------------------------------


def test_rear_press_while_recording_injects_transcribed_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, recorder, output, _integration, _status = _make_app(
        monkeypatch, tmp_path
    )

    recorder.is_recording = True
    getattr(app, "_on_rear_press")()

    assert recorder.stopped == 1
    assert output.injected, "transcript was never injected"
    text, _kwargs = output.injected[0]
    assert text == "transcribed text"


def test_rear_press_while_idle_sends_enter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, recorder, output, _integration, _status = _make_app(
        monkeypatch, tmp_path
    )

    recorder.is_recording = False
    getattr(app, "_on_rear_press")()

    assert recorder.stopped == 0
    assert output.enter_calls, "idle rear press should send Enter"
    assert output.injected == []


# ---------------------------------------------------------------------------
# Workspace gesture
# ---------------------------------------------------------------------------


def test_workspace_gesture_calls_switch_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, patch_threads
) -> None:
    app, _recorder, _output, integration, _status = _make_app(
        monkeypatch, tmp_path
    )
    # Route the left gesture to the workspace action for this test; the
    # frozen config is customized via dataclasses.replace.
    setattr(
        app,
        "_config",
        replace(app._config, gesture_left_action="workspace_left"),
    )

    handler = _gesture_handler(app)
    handler("left")

    assert integration.workspace_calls == ["left"]
