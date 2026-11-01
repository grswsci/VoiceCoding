"""Contract tests for voicecoding.output.TextOutput.

Groups: clipboard routing with and without auto-paste (pyperclip copy plus a
single Ctrl+V tap), Enter delivery modes via the private tap helpers, a
broken paste tap still leaving the text on the clipboard, and the focus
probe resolving an "unknown" answer to False.

TextOutput is constructed for real: the pynput keyboard backend is imported
lazily by _ensure_keyboard on first use, which under the conftest fakes
resolves to the recording FakeKeyboardController. No real input hooks are
touched.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pyperclip

from conftest import (
    FakeKeyboardController,
    normalize_key_events,
)


class _TapRecorder:
    """Stand-in for the private tap helpers, recording every call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    def text(self) -> str:
        return repr(self.calls)


def _make_subject(system_integration=None):
    from voicecoding.output import TextOutput

    return TextOutput(system_integration=system_integration)


# ---------------------------------------------------------------------------
# Clipboard routing
# ---------------------------------------------------------------------------


def test_clipboard_route_without_auto_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _make_subject()

    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", lambda text: copied.append(text))

    subject.inject_or_clipboard("  hello  ", auto_paste=False)

    assert copied == ["hello"]
    # The clipboard-only route never synthesizes keystrokes: no keyboard
    # controller is even constructed.
    assert FakeKeyboardController.instances == []


def test_auto_paste_route_copies_and_taps_ctrl_v(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _make_subject()

    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", lambda text: copied.append(text))

    subject.inject_or_clipboard("hello", auto_paste=True)

    assert copied == ["hello"]
    assert len(FakeKeyboardController.instances) == 1
    events = normalize_key_events(FakeKeyboardController.instances[-1].events)
    assert events == [
        ("press", "ctrl"),
        ("press", "v"),
        ("release", "v"),
        ("release", "ctrl"),
    ]


def test_paste_tap_failure_still_copies_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken paste tap must not lose the transcript from the clipboard."""
    subject = _make_subject()

    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", lambda text: copied.append(text))

    def failing_tap(*args, **kwargs) -> None:
        raise RuntimeError("paste tap refused")

    setattr(subject, "_tap_key", failing_tap)
    setattr(subject, "_tap_modified_key", failing_tap)

    route = subject.inject_or_clipboard("hello", auto_paste=True)

    assert copied == ["hello"]
    assert route == "clipboard"


# ---------------------------------------------------------------------------
# Enter delivery modes
# ---------------------------------------------------------------------------


def _bind_enter_taps(subject):
    plain = _TapRecorder("tap_key")
    modified = _TapRecorder("tap_modified_key")
    setattr(subject, "_tap_key", plain)
    setattr(subject, "_tap_modified_key", modified)
    return plain, modified


def test_send_enter_mode_taps_plain_enter() -> None:
    subject = _make_subject()
    plain, modified = _bind_enter_taps(subject)

    send_enter: Callable = getattr(subject, "send_enter")
    send_enter(mode="enter")

    assert len(plain.calls) == 1
    assert modified.calls == []
    assert "enter" in plain.text().lower()


def test_send_enter_ctrl_mode_uses_modified_tap() -> None:
    subject = _make_subject()
    plain, modified = _bind_enter_taps(subject)

    send_enter: Callable = getattr(subject, "send_enter")
    send_enter(mode="ctrl_enter")

    assert plain.calls == []
    assert len(modified.calls) == 1
    assert "ctrl" in modified.text().lower()
    assert "enter" in modified.text().lower()


def test_send_enter_shift_mode_uses_modified_tap() -> None:
    subject = _make_subject()
    plain, modified = _bind_enter_taps(subject)

    send_enter: Callable = getattr(subject, "send_enter")
    send_enter(mode="shift_enter")

    assert plain.calls == []
    assert len(modified.calls) == 1
    assert "shift" in modified.text().lower()
    assert "enter" in modified.text().lower()


def test_send_enter_none_mode_taps_nothing() -> None:
    subject = _make_subject()
    plain, modified = _bind_enter_taps(subject)

    send_enter: Callable = getattr(subject, "send_enter")
    send_enter(mode="none")

    assert plain.calls == []
    assert modified.calls == []


# ---------------------------------------------------------------------------
# Focus probe
# ---------------------------------------------------------------------------


def test_is_text_input_focused_resolves_unknown_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An integration answer of None means "unknown"; the output layer must
    # resolve that to False rather than typing blindly into the desktop.
    integration = SimpleNamespace(is_text_input_focused=lambda: None)
    subject = _make_subject(system_integration=integration)

    is_focused: Callable[[], bool] = getattr(subject, "is_text_input_focused")
    assert is_focused() is False
