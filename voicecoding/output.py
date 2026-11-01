"""Local text output: clipboard staging plus synthesized keystroke injection.

All keyboard synthesis goes through pynput's keyboard controller, imported
lazily on first use so this module can be constructed (and unit-tested)
without an input backend present.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

import pyperclip

from voicecoding.system_integration import (
    SystemIntegration,
    create_system_integration,
)

_LOG = logging.getLogger(__name__)

# Settle time between press and release so target applications observe the
# synthesized keystroke reliably.
_TAP_DELAY_S = 0.012


class TextOutput:
    """Delivers transcribed text locally: typed, pasted, or clipboard only."""

    def __init__(
        self,
        *,
        system_integration: SystemIntegration | None = None,
    ) -> None:
        self._kb: Any = None
        self._enter_key: object = None
        self._ctrl_key: object = None
        self._shift_key: object = None
        self._system_integration: SystemIntegration = (
            system_integration
            if system_integration is not None
            else create_system_integration()
        )

    def send_enter(self, *, mode: str = "enter") -> None:
        """Send an Enter keystroke according to ``mode``.

        Supported modes: ``enter``, ``ctrl_enter``, ``shift_enter``, ``none``.
        """
        normalized = mode.strip().lower()
        if normalized == "none":
            return
        self._ensure_keyboard()
        if normalized == "enter":
            self._tap_key(self._enter_key)
            return
        if normalized == "ctrl_enter":
            self._tap_modified_key(self._ctrl_key, self._enter_key)
            return
        if normalized == "shift_enter":
            self._tap_modified_key(self._shift_key, self._enter_key)
            return
        raise ValueError(f"Unsupported enter mode: {mode!r}")

    def inject_or_clipboard(self, text: str, *, auto_paste: bool = False) -> str:
        """Deliver ``text`` and report the route taken.

        Routes: ``empty`` (nothing to send), ``typed`` (into the focused
        control), ``pasted`` (clipboard + Ctrl+V), ``clipboard`` (clipboard
        only, or paste failed).
        """
        normalized = text.strip()
        if not normalized:
            return "empty"

        if self.is_text_input_focused():
            try:
                self._ensure_keyboard().type(normalized)
                return "typed"
            except Exception:
                _LOG.warning(
                    "Direct typing failed; falling back to clipboard",
                    exc_info=True,
                )

        pyperclip.copy(normalized)
        if auto_paste:
            try:
                self._paste_clipboard()
                return "pasted"
            except Exception:
                _LOG.warning("Paste keystroke failed; text left on clipboard")
                return "clipboard"
        return "clipboard"

    def is_text_input_focused(self) -> bool:
        """Whether a text control currently holds focus.

        The Windows integration answers False, so the clipboard + paste path
        is the default delivery route.
        """
        try:
            focused = self._system_integration.is_text_input_focused()
        except Exception:
            focused = None
        return focused is True

    def _paste_clipboard(self) -> None:
        self._ensure_keyboard()
        self._tap_modified_key(self._ctrl_key, "v")

    def _ensure_keyboard(self) -> Any:
        """Import and construct the pynput keyboard controller on first use."""
        if self._kb is None:
            try:
                keyboard_module = importlib.import_module("pynput.keyboard")
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load keyboard control dependencies: {error}"
                ) from error
            controller_ctor = getattr(keyboard_module, "Controller")
            key_holder = getattr(keyboard_module, "Key")
            self._kb = controller_ctor()
            self._enter_key = key_holder.enter
            self._ctrl_key = key_holder.ctrl
            self._shift_key = key_holder.shift
        return self._kb

    def _tap_key(self, key: object) -> None:
        kb = self._kb
        kb.press(key)
        time.sleep(_TAP_DELAY_S)
        kb.release(key)

    def _tap_modified_key(self, modifier: object, key: object) -> None:
        pressed_modifier = False
        pressed_key = False
        kb = self._kb
        try:
            kb.press(modifier)
            pressed_modifier = True
            kb.press(key)
            pressed_key = True
            time.sleep(_TAP_DELAY_S)
        finally:
            if pressed_key:
                try:
                    kb.release(key)
                except Exception:
                    pass
            if pressed_modifier:
                try:
                    kb.release(modifier)
                except Exception:
                    pass
