"""Shared test doubles for the voicecoding suite.

The suite must run in a lean environment that ships only pytest, numpy and
pyperclip, so every heavy third-party library the package touches is replaced
here by a small, call-recording fake. The fakes are injected into
``sys.modules`` at conftest import time — before any test module imports the
package — so function-level imports inside package code resolve to fakes.

Every fake exposes a module-level ``STATE`` singleton that individual tests
can configure through ``monkeypatch`` or direct mutation; an autouse fixture
resets all state between tests.
"""
from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# pynput fakes
# ---------------------------------------------------------------------------


class FakeButton:
    """Namespace mirroring pynput.mouse.Button members."""

    left = "left"
    right = "right"
    middle = "middle"
    x1 = "x1"
    x2 = "x2"


class FakeKey:
    """Namespace mirroring the pynput.keyboard.Key members the package uses.

    Members are plain lowercase strings so assertions can compare via
    ``str(key)`` without depending on enum internals.
    """

    enter = "enter"
    esc = "esc"
    space = "space"
    tab = "tab"
    backspace = "backspace"
    v = "v"

    ctrl = "ctrl"
    ctrl_l = "ctrl"
    ctrl_r = "ctrl"

    shift = "shift"
    shift_l = "shift"
    shift_r = "shift"

    alt = "alt"
    alt_l = "alt"
    alt_r = "alt"
    alt_gr = "altgr"

    cmd = "cmd"
    cmd_l = "cmd"
    cmd_r = "cmd"

    left = "left"
    right = "right"
    up = "up"
    down = "down"


class FakeKeyCode:
    """Minimal pynput.keyboard.KeyCode stand-in carrying a virtual key code."""

    def __init__(self, vk: int | None = None, char: str | None = None) -> None:
        self.vk = vk
        self.char = char

    @classmethod
    def from_vk(cls, vk: int) -> "FakeKeyCode":
        return cls(vk=vk)

    @classmethod
    def from_char(cls, char: str) -> "FakeKeyCode":
        return cls(char=char)

    def __repr__(self) -> str:
        return f"KeyCode(vk={self.vk!r}, char={self.char!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeKeyCode):
            return (self.vk, self.char) == (other.vk, other.char)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.vk, self.char))


class FakeMouseListener:
    """pynput.mouse.Listener replacement: records kwargs, spawns nothing."""

    instances: list["FakeMouseListener"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.running = False
        self.suppress_count = 0
        FakeMouseListener.instances.append(self)

    def start(self) -> "FakeMouseListener":
        self.running = True
        return self

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        return None

    def suppress_event(self) -> None:
        self.suppress_count += 1

    def __enter__(self) -> "FakeMouseListener":
        return self.start()

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False


class FakeKeyboardListener:
    """pynput.keyboard.Listener replacement: records kwargs, spawns nothing."""

    instances: list["FakeKeyboardListener"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.running = False
        FakeKeyboardListener.instances.append(self)

    def start(self) -> "FakeKeyboardListener":
        self.running = True
        return self

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        return None

    def __enter__(self) -> "FakeKeyboardListener":
        return self.start()

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False


class FakeMouseController:
    """pynput.mouse.Controller replacement that records every action."""

    instances: list["FakeMouseController"] = []

    def __init__(self) -> None:
        self.position: tuple[float, float] = (0.0, 0.0)
        self.events: list[tuple] = []
        FakeMouseController.instances.append(self)

    def move(self, x: float, y: float) -> None:
        self.position = (x, y)
        self.events.append(("move", x, y))

    def press(self, button) -> None:
        self.events.append(("press", button))

    def release(self, button) -> None:
        self.events.append(("release", button))

    def click(self, button, count: int = 1) -> None:
        self.events.append(("click", button, count))


class FakeKeyboardController:
    """pynput.keyboard.Controller replacement that records every action."""

    instances: list["FakeKeyboardController"] = []

    def __init__(self) -> None:
        self.events: list[tuple] = []
        FakeKeyboardController.instances.append(self)

    def press(self, key) -> None:
        self.events.append(("press", key))

    def release(self, key) -> None:
        self.events.append(("release", key))

    def type(self, text: str) -> None:
        self.events.append(("type", text))

    def pressed(self, *keys):
        """Context-manager form of a modified tap (press all, release reversed)."""
        controller = self

        class _Pressed:
            def __enter__(self):
                for key in keys:
                    controller.press(key)
                return controller

            def __exit__(self, *exc_info) -> bool:
                for key in reversed(keys):
                    controller.release(key)
                return False

        return _Pressed()


pynput_mouse_mod = types.ModuleType("pynput.mouse")
pynput_mouse_mod.Listener = FakeMouseListener
pynput_mouse_mod.Controller = FakeMouseController
pynput_mouse_mod.Button = FakeButton

pynput_keyboard_mod = types.ModuleType("pynput.keyboard")
pynput_keyboard_mod.Listener = FakeKeyboardListener
pynput_keyboard_mod.Controller = FakeKeyboardController
pynput_keyboard_mod.Key = FakeKey
pynput_keyboard_mod.KeyCode = FakeKeyCode

pynput_mod = types.ModuleType("pynput")
pynput_mod.mouse = pynput_mouse_mod
pynput_mod.keyboard = pynput_keyboard_mod


# ---------------------------------------------------------------------------
# sounddevice / soundfile fakes
# ---------------------------------------------------------------------------

#: Mixed device list: two plausible capture devices plus OS-reserved entries
#: (pulse/pipewire/sysdefault-style) that device filtering must exclude.
DEFAULT_INPUT_DEVICES = [
    {
        "name": "Microphone (Realtek Audio)",
        "index": 0,
        "max_input_channels": 2,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "pulse",
        "index": 1,
        "max_input_channels": 32,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "pipewire",
        "index": 2,
        "max_input_channels": 32,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "sysdefault",
        "index": 3,
        "max_input_channels": 128,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "default",
        "index": 4,
        "max_input_channels": 128,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "USB Condenser Microphone",
        "index": 5,
        "max_input_channels": 1,
        "default_samplerate": 16000.0,
        "hostapi": 0,
    },
]


class FakeInputStream:
    """sounddevice.InputStream replacement.

    ``start()`` only marks the stream active — it deliberately does NOT
    deliver blocks through the registered callback. Real PortAudio invokes
    the callback from a separate OS thread after ``start()`` returns, while
    this fake would call it synchronously on the caller's thread; that
    re-enters ``AudioRecorder._callback`` while the recorder still holds its
    non-reentrant lock inside its own ``start()``/``open_stream()`` and
    deadlocks. Tests that need capture audio feed blocks explicitly through
    the recorded ``kwargs["callback"]`` instead. All constructor kwargs are
    recorded for assertions.
    """

    instances: list["FakeInputStream"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.active = False
        self.blocks_read = 0
        FakeInputStream.instances.append(self)

    def _frames(self) -> int:
        return int(self.kwargs.get("blocksize") or 1024)

    def _empty_block(self, frames: int | None = None):
        frames = frames or self._frames()
        channels = int(self.kwargs.get("channels") or 1)
        dtype = self.kwargs.get("dtype") or "float32"
        return np.zeros((frames, channels), dtype=dtype)

    def deliver_blocks(self, count: int) -> None:
        """Push ``count`` silent blocks through the registered callback."""
        callback = self.kwargs.get("callback")
        if callback is None:
            return
        for _ in range(count):
            callback(self._empty_block(), self._frames(), None, None)
            self.blocks_read += 1

    def start(self) -> "FakeInputStream":
        self.active = True
        return self

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False

    def read(self, frames: int):
        block = self._empty_block(frames)
        self.blocks_read += 1
        return block, False

    def __enter__(self) -> "FakeInputStream":
        return self.start()

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False


sounddevice_mod = types.ModuleType("sounddevice")
sounddevice_mod.STATE = SimpleNamespace(devices=None)
sounddevice_mod.InputStream = FakeInputStream
sounddevice_mod.PortAudioError = type("PortAudioError", (Exception,), {})
sounddevice_mod.default = SimpleNamespace(
    device=None, samplerate=48000.0, channels=1
)


def _fake_query_devices(index=None):
    devices = sounddevice_mod.STATE.devices or DEFAULT_INPUT_DEVICES
    if index is not None:
        return devices[index]
    return [dict(device) for device in devices]


sounddevice_mod.query_devices = _fake_query_devices


soundfile_mod = types.ModuleType("soundfile")
soundfile_mod.STATE = SimpleNamespace(writes=[])


def _fake_soundfile_write(path, data, samplerate, **kwargs) -> None:
    soundfile_mod.STATE.writes.append(
        {
            "path": str(path),
            "data": np.asarray(data),
            "samplerate": samplerate,
            "kwargs": dict(kwargs),
        }
    )


soundfile_mod.write = _fake_soundfile_write
soundfile_mod.info = lambda path: SimpleNamespace(
    samplerate=16000, frames=0, channels=1
)


# ---------------------------------------------------------------------------
# faster-whisper / ctranslate2 fakes
# ---------------------------------------------------------------------------


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTranscriptionInfo:
    def __init__(self, language: str = "en", duration: float = 1.0) -> None:
        self.language = language
        self.duration = duration


def default_transcribe_result():
    # Real Whisper segment texts carry leading separators between segments,
    # so the joiner produces " hello world" which .strip() turns into
    # "hello world" — mirroring production output.
    return (
        [FakeSegment(" hello"), FakeSegment(" world")],
        FakeTranscriptionInfo(language="en"),
    )


class FakeWhisperModel:
    """faster_whisper.WhisperModel replacement.

    Records every constructor call (model path, device, compute type) and
    every transcribe call. Behaviour is driven by the module-level ``STATE``:

    - ``transcribe_errors``: deque of exceptions (or ``None``) consumed one
      per transcribe call; a non-``None`` entry is raised.
    - ``results``: optional deque of ``(segments, info)`` overrides; when
      empty the default two-segment result is returned.
    """

    instances: list["FakeWhisperModel"] = []

    def __init__(
        self,
        model_size_or_path,
        device: str = "auto",
        compute_type: str = "default",
        **kwargs,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.kwargs = dict(kwargs)
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        faster_whisper_mod.STATE.transcribe_calls.append((str(audio), dict(kwargs)))
        pending = faster_whisper_mod.STATE.transcribe_errors
        if pending:
            error = pending.popleft()
            if error is not None:
                raise error
        if faster_whisper_mod.STATE.results:
            return faster_whisper_mod.STATE.results.popleft()
        return default_transcribe_result()


faster_whisper_mod = types.ModuleType("faster_whisper")
faster_whisper_mod.WhisperModel = FakeWhisperModel
faster_whisper_mod.STATE = SimpleNamespace(
    transcribe_calls=[],
    transcribe_errors=deque(),
    results=deque(),
)

ctranslate2_mod = types.ModuleType("ctranslate2")
ctranslate2_mod.STATE = SimpleNamespace(cuda_count=1)


def _fake_get_cuda_device_count() -> int:
    return ctranslate2_mod.STATE.cuda_count


ctranslate2_mod.get_cuda_device_count = _fake_get_cuda_device_count


# ---------------------------------------------------------------------------
# injection — must happen before any voicecoding import
# ---------------------------------------------------------------------------

sys.modules["pynput"] = pynput_mod
sys.modules["pynput.mouse"] = pynput_mouse_mod
sys.modules["pynput.keyboard"] = pynput_keyboard_mod
sys.modules["sounddevice"] = sounddevice_mod
sys.modules["soundfile"] = soundfile_mod
sys.modules["faster_whisper"] = faster_whisper_mod
sys.modules["ctranslate2"] = ctranslate2_mod


# ---------------------------------------------------------------------------
# shared helpers available to test modules
# ---------------------------------------------------------------------------


class SyncThread:
    """``threading.Thread`` replacement that runs the target synchronously.

    Patching ``threading.Thread`` with this class makes worker-thread code
    paths deterministic: the target body executes inside the constructor, so
    tests can assert on its effects immediately.
    """

    instances: list["SyncThread"] = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self.daemon = daemon
        self.name = name
        SyncThread.instances.append(self)
        if target is not None:
            target(*self._args, **self._kwargs)

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


_MODIFIER_ALIASES = {
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "shift_l": "shift",
    "shift_r": "shift",
    "cmd_l": "cmd",
    "cmd_r": "cmd",
    "alt_l": "alt",
}


def key_name(key) -> str:
    """Normalize any fake key object to a lowercase canonical name."""
    name = str(key).strip("<>").lower()
    return _MODIFIER_ALIASES.get(name, name)


def normalize_key_events(events) -> list[tuple[str, str]]:
    """Map recorded (action, key) pairs to (action, canonical-name) pairs."""
    return [(action, key_name(key)) for action, key in events]


def purge_voicecoding_env(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Remove every VOICECODING_* variable so tests start from clean env."""
    import os

    for name in [key for key in os.environ if key.startswith("VOICECODING_")]:
        monkeypatch.delenv(name, raising=False)


def load_contract_config(monkeypatch: "pytest.MonkeyPatch | None" = None):
    """Build the AppConfig the way the package exposes it.

    Tries the documented loading entry points in order (module-level loader,
    ``AppConfig.from_env`` / ``AppConfig.load``) and falls back to the plain
    constructor, so the suite tracks the configuration contract rather than
    one specific function name.
    """
    if monkeypatch is not None:
        purge_voicecoding_env(monkeypatch)

    from voicecoding import config as config_mod
    from voicecoding.config import AppConfig

    module_loader = getattr(config_mod, "load_config", None)
    if callable(module_loader):
        return module_loader()

    for classmethod_name in ("from_env", "load"):
        factory = getattr(AppConfig, classmethod_name, None)
        if callable(factory):
            return factory()

    return AppConfig()


def reset_fake_state() -> None:
    """Reset every fake library to its pristine state."""
    FakeWhisperModel.instances.clear()
    faster_whisper_mod.STATE.transcribe_calls.clear()
    faster_whisper_mod.STATE.transcribe_errors.clear()
    faster_whisper_mod.STATE.results.clear()

    FakeInputStream.instances.clear()
    sounddevice_mod.STATE.devices = None

    soundfile_mod.STATE.writes.clear()

    ctranslate2_mod.STATE.cuda_count = 1

    FakeMouseListener.instances.clear()
    FakeKeyboardListener.instances.clear()
    FakeMouseController.instances.clear()
    FakeKeyboardController.instances.clear()
    SyncThread.instances.clear()


@pytest.fixture(autouse=True)
def isolated_fake_libraries():
    """Guarantee each test starts and ends with pristine fake libraries."""
    reset_fake_state()
    yield
    reset_fake_state()
