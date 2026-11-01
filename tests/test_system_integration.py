"""Contract tests for voicecoding.system_integration.

Groups: create_system_integration platform dispatch (win32 ->
WindowsSystemIntegration, otherwise the no-op integration), and the
switch_workspace key choreography (press cmd -> ctrl -> arrow, release in
reverse) driven against the fake pynput keyboard controller.
"""
from __future__ import annotations

import sys

import pytest

from conftest import (
    FakeKeyboardController,
    normalize_key_events,
)


def _noop_class():
    import voicecoding.system_integration as integration_module

    for name in ("Noop", "NoopSystemIntegration", "NoopIntegration"):
        candidate = getattr(integration_module, name, None)
        if isinstance(candidate, type):
            return candidate
    pytest.fail("system_integration exposes no no-op integration class")


def _make_windows_integration():
    from voicecoding.system_integration import WindowsSystemIntegration

    return WindowsSystemIntegration()


def _bound_controller(integration) -> FakeKeyboardController:
    controller = getattr(integration, "_keyboard", None)
    if isinstance(controller, FakeKeyboardController):
        return controller
    assert FakeKeyboardController.instances, "no keyboard controller was created"
    return FakeKeyboardController.instances[-1]


# ---------------------------------------------------------------------------
# Platform dispatch
# ---------------------------------------------------------------------------


def test_windows_platform_builds_windows_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicecoding.system_integration import (
        create_system_integration,
        WindowsSystemIntegration,
    )

    monkeypatch.setattr(sys, "platform", "win32")
    integration = create_system_integration()

    assert isinstance(integration, WindowsSystemIntegration)


def test_non_windows_platform_builds_noop_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicecoding.system_integration import create_system_integration

    monkeypatch.setattr(sys, "platform", "linux")
    integration = create_system_integration()

    assert isinstance(integration, _noop_class())


# ---------------------------------------------------------------------------
# switch_workspace key order
# ---------------------------------------------------------------------------


def test_switch_workspace_left_presses_cmd_ctrl_left_in_order() -> None:
    integration = _make_windows_integration()

    integration.switch_workspace("left")

    events = normalize_key_events(_bound_controller(integration).events)
    assert events == [
        ("press", "cmd"),
        ("press", "ctrl"),
        ("press", "left"),
        ("release", "left"),
        ("release", "ctrl"),
        ("release", "cmd"),
    ]


def test_switch_workspace_right_presses_cmd_ctrl_right_in_order() -> None:
    integration = _make_windows_integration()

    integration.switch_workspace("right")

    events = normalize_key_events(_bound_controller(integration).events)
    assert events == [
        ("press", "cmd"),
        ("press", "ctrl"),
        ("press", "right"),
        ("release", "right"),
        ("release", "ctrl"),
        ("release", "cmd"),
    ]


def test_switch_workspace_releases_every_key_it_pressed() -> None:
    integration = _make_windows_integration()

    integration.switch_workspace("left")

    events = normalize_key_events(_bound_controller(integration).events)
    pressed = [name for action, name in events if action == "press"]
    released = [name for action, name in events if action == "release"]
    assert sorted(pressed) == sorted(released)
    assert released == list(reversed(pressed))
