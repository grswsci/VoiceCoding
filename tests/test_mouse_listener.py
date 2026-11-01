"""Contract tests for voicecoding.mouse_listener.SideButtonListener.

Groups kept analogous to the legacy behaviour: gesture classification with
threshold edges, debounce, right-button gesture trigger decision, anchor
pinning via move_cursor, and the Windows-specific button alias routing and
win32_event_filter suppression contract.

The listener is constructed with the package's explicit signature, pulling
every tunable from a real AppConfig. The package maps front=X2 / rear=X1 by
default, so routing tests pin FRONT/REAR via env to keep the x1-is-front
scenario; driver alias spellings (button8/x_button1) are rejected by the
config layer and therefore passed straight to the listener, which owns alias
normalization. Click dispatch runs on a worker thread; the worker is made
synchronous by patching threading.Thread only after the listener's own
thread is up (the synchronous fake would inline the listener's event loop
and never return).
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable

import pytest

from conftest import (
    FakeButton,
    FakeMouseListener,
    SyncThread,
    load_contract_config,
    purge_voicecoding_env,
)

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C


class _Handlers:
    """Call recorder passed as the listener callbacks."""

    def __init__(self) -> None:
        self.front: list[int] = []
        self.rear: list[int] = []
        self.gestures: list[str] = []

    def on_front_press(self) -> None:
        self.front.append(1)

    def on_rear_press(self) -> None:
        self.rear.append(1)

    def on_gesture(self, direction: str) -> None:
        self.gestures.append(direction)


class FakeIntegration:
    """System-integration double capturing cursor/workspace side effects."""

    def __init__(self) -> None:
        self.cursor_calls: list[tuple[float, float]] = []
        self.workspace_calls: list[str] = []

    def move_cursor(self, x: float, y: float) -> bool:
        self.cursor_calls.append((x, y))
        return True

    def switch_workspace(self, direction: str) -> bool:
        self.workspace_calls.append(direction)
        return True

    def is_text_input_focused(self) -> bool:
        return False


def _make_listener(
    monkeypatch: pytest.MonkeyPatch,
    handlers: _Handlers,
    integration: FakeIntegration | None = None,
    *,
    front_button: str | None = None,
    rear_button: str | None = None,
    **env: str,
):
    from voicecoding.mouse_listener import SideButtonListener

    purge_voicecoding_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(f"VOICECODING_{key}", value)
    config = load_contract_config()
    return SideButtonListener(
        on_front_press=handlers.on_front_press,
        on_rear_press=handlers.on_rear_press,
        on_gesture=handlers.on_gesture,
        front_button=(
            front_button if front_button is not None else config.front_button
        ),
        rear_button=(
            rear_button if rear_button is not None else config.rear_button
        ),
        debounce_s=config.button_debounce_ms / 1000.0,
        gestures_enabled=config.gestures_enabled,
        gesture_trigger_button=config.gesture_trigger_button,
        gesture_threshold_px=config.gesture_threshold_px,
        gesture_freeze_pointer=config.gesture_freeze_pointer,
        gesture_restore_cursor=config.gesture_restore_cursor,
        system_integration=integration,
    )


def _start_listener(monkeypatch: pytest.MonkeyPatch, listener) -> None:
    listener.start()
    # Wait for the listener's own thread to register the fake pynput
    # Listener before handing callbacks back to the test.
    deadline = time.perf_counter() + 5.0
    while not FakeMouseListener.instances and time.perf_counter() < deadline:
        time.sleep(0.01)
    # Make click-dispatch workers synchronous, but only after the listener's
    # own thread is up: SyncThread would run the listener's event loop inline
    # and never return.
    monkeypatch.setattr(threading, "Thread", SyncThread)


def _registered_callbacks(listener) -> dict:
    assert FakeMouseListener.instances, "listener did not create a pynput Listener"
    fake = FakeMouseListener.instances[-1]
    return fake.kwargs


# ---------------------------------------------------------------------------
# Gesture classification (static classifier, threshold edges)
# ---------------------------------------------------------------------------


def _classify(dx: float, dy: float, threshold_px: int) -> str | None:
    from voicecoding.mouse_listener import SideButtonListener

    classify: Callable = getattr(SideButtonListener, "_classify_gesture")
    return classify(dx, dy, threshold_px)


def test_classify_returns_none_below_threshold() -> None:
    assert _classify(20, 10, 120) is None
    assert _classify(119, 0, 120) is None


def test_classify_hits_direction_exactly_at_threshold() -> None:
    assert _classify(120, 0, 120) == "right"
    assert _classify(-120, 0, 120) == "left"
    assert _classify(0, -120, 120) == "up"
    assert _classify(0, 120, 120) == "down"


def test_classify_picks_dominant_axis() -> None:
    assert _classify(200, 30, 120) == "right"
    assert _classify(-220, 10, 120) == "left"
    assert _classify(20, -250, 120) == "up"
    assert _classify(30, 240, 120) == "down"


def test_classify_ignores_small_diagonal_jitter() -> None:
    assert _classify(30, 30, 120) is None


# ---------------------------------------------------------------------------
# Front/rear button routing through pynput on_click
# ---------------------------------------------------------------------------


def test_front_button_press_routes_to_front_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener = _make_listener(
        monkeypatch, handlers, FRONT_BUTTON="x1", REAR_BUTTON="x2"
    )
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](10.0, 10.0, FakeButton.x1, True)
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, False)

    assert len(handlers.front) == 1
    assert handlers.rear == []
    assert handlers.gestures == []
    listener.stop()


def test_rear_button_press_routes_to_rear_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener = _make_listener(
        monkeypatch, handlers, FRONT_BUTTON="x1", REAR_BUTTON="x2"
    )
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](10.0, 10.0, FakeButton.x2, True)

    assert len(handlers.rear) == 1
    assert handlers.front == []
    listener.stop()


@pytest.mark.parametrize("alias", ["button8", "x_button1", "x1"])
def test_front_button_aliases_route_to_x1(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    handlers = _Handlers()
    # Alias spellings are normalized by the listener itself, not the config
    # layer, so they bypass env and go straight to the constructor.
    listener = _make_listener(
        monkeypatch, handlers, front_button=alias, rear_button="x2"
    )
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](0.0, 0.0, FakeButton.x1, True)

    assert len(handlers.front) == 1
    listener.stop()


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_debounce_swallows_rapid_repeat_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voicecoding.mouse_listener as mouse_listener_module

    clock = FakeClock()
    monkeypatch.setattr(mouse_listener_module.time, "time", clock)
    monkeypatch.setattr(mouse_listener_module.time, "monotonic", clock)

    handlers = _Handlers()
    listener = _make_listener(
        monkeypatch, handlers, FRONT_BUTTON="x1", REAR_BUTTON="x2"
    )
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](10.0, 10.0, FakeButton.x1, True)  # fires
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, False)
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, True)  # inside window
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, False)

    assert len(handlers.front) == 1

    clock.advance(0.5)  # well past the 150 ms debounce window
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, True)
    callbacks["on_click"](10.0, 10.0, FakeButton.x1, False)

    assert len(handlers.front) == 2
    listener.stop()


# ---------------------------------------------------------------------------
# Right-button gesture trigger decision
# ---------------------------------------------------------------------------


def _gesture_listener(monkeypatch: pytest.MonkeyPatch, handlers: _Handlers):
    return _make_listener(
        monkeypatch,
        handlers,
        integration=FakeIntegration(),
        GESTURES_ENABLED="true",
        GESTURE_TRIGGER_BUTTON="right",
    )


def test_right_trigger_small_move_fires_no_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener = _gesture_listener(monkeypatch, handlers)
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](100.0, 100.0, FakeButton.right, True)
    callbacks["on_move"](130.0, 100.0)  # 30 px < 120 px threshold
    callbacks["on_click"](130.0, 100.0, FakeButton.right, False)

    assert handlers.gestures == []
    listener.stop()


def test_right_trigger_large_move_fires_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener = _gesture_listener(monkeypatch, handlers)
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](100.0, 100.0, FakeButton.right, True)
    callbacks["on_move"](240.0, 100.0)  # 140 px past the 120 px threshold
    callbacks["on_click"](240.0, 100.0, FakeButton.right, False)

    assert handlers.gestures == ["right"]
    listener.stop()


def test_gesture_without_trigger_button_held_fires_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener = _gesture_listener(monkeypatch, handlers)
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_move"](600.0, 600.0)  # large move, no trigger held

    assert handlers.gestures == []
    listener.stop()


# ---------------------------------------------------------------------------
# Freeze: anchor pinning via move_cursor
# ---------------------------------------------------------------------------


def test_active_gesture_pins_cursor_to_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration = FakeIntegration()
    handlers = _Handlers()
    # Pointer pinning applies to the side-button triggers (front/rear); the
    # right button keeps its native click and never pins.
    listener = _make_listener(
        monkeypatch,
        handlers,
        integration=integration,
        GESTURES_ENABLED="true",
        FRONT_BUTTON="x1",
        REAR_BUTTON="x2",
    )
    _start_listener(monkeypatch, listener)
    callbacks = _registered_callbacks(listener)

    callbacks["on_click"](100, 100, FakeButton.x2, True)  # rear trigger held
    callbacks["on_move"](160, 90)  # still below threshold, gesture active

    assert (100, 100) in integration.cursor_calls
    assert handlers.gestures == []
    listener.stop()


# ---------------------------------------------------------------------------
# win32_event_filter contract
# ---------------------------------------------------------------------------


def _win32_listener(monkeypatch: pytest.MonkeyPatch, handlers: _Handlers):
    monkeypatch.setattr(sys, "platform", "win32")
    listener = _make_listener(monkeypatch, handlers)
    _start_listener(monkeypatch, listener)
    fake = FakeMouseListener.instances[-1]
    return listener, fake


def test_win32_platform_passes_event_filter_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener, fake = _win32_listener(monkeypatch, handlers)

    event_filter = fake.kwargs.get("win32_event_filter")
    assert callable(event_filter)
    listener.stop()


def test_event_filter_suppresses_xbutton_messages_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    listener, fake = _win32_listener(monkeypatch, handlers)
    event_filter = fake.kwargs["win32_event_filter"]

    x1_data = SimpleNamespace(mouseData=0x0001, pt=SimpleNamespace(x=10, y=20))
    x2_data = SimpleNamespace(mouseData=0x0002, pt=SimpleNamespace(x=30, y=40))

    event_filter(WM_XBUTTONDOWN, x1_data)
    event_filter(WM_XBUTTONUP, x1_data)
    event_filter(WM_XBUTTONDOWN, x2_data)
    event_filter(WM_XBUTTONUP, x2_data)
    assert fake.suppress_count == 4

    # pynput 1.8 short-circuits its own dispatch when the filter suppresses,
    # so the filter routes each side-button press through on_click itself;
    # the handlers must have seen each press exactly once. Dispatch runs on
    # worker threads, so flush them before asserting.
    deadline = time.perf_counter() + 2.0
    while (len(handlers.front) < 1 or len(handlers.rear) < 1) and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert handlers.front == [1]
    assert handlers.rear == [1]

    # Ordinary mouse messages must flow untouched: no suppression, and the
    # app-level callbacks stay silent.
    for message in (
        WM_MOUSEMOVE,
        WM_LBUTTONDOWN,
        WM_LBUTTONUP,
        WM_RBUTTONDOWN,
        WM_RBUTTONUP,
        WM_MBUTTONDOWN,
    ):
        event_filter(message, x1_data)
    assert fake.suppress_count == 4
    assert handlers.front == []
    assert handlers.rear == []
    listener.stop()


def test_non_windows_platform_omits_event_filter_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _Handlers()
    monkeypatch.setattr(sys, "platform", "linux")
    listener = _make_listener(monkeypatch, handlers)
    _start_listener(monkeypatch, listener)
    fake = FakeMouseListener.instances[-1]

    assert "win32_event_filter" not in fake.kwargs
    listener.stop()
