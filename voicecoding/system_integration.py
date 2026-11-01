"""Desktop integration surface: pointer control, window probes, virtual desktops.

The Windows backend talks to user32 through ctypes (standard library only).
Every heavy dependency (pynput for synthesized chords) is imported lazily
inside the method that needs it, so importing this module stays side-effect
free on any platform.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Protocol

_LOG = logging.getLogger(__name__)


class SystemIntegration(Protocol):
    """Integration surface consumed by the listeners, output layer, and app."""

    def active_window(self) -> dict[str, object] | None: ...

    def cursor_position(self) -> tuple[int, int] | None: ...

    def move_cursor(self, *, x: int, y: int) -> bool: ...

    def switch_workspace(self, direction: str) -> bool: ...

    def is_text_input_focused(self) -> bool | None: ...


class NoopSystemIntegration:
    """Stub for non-Windows platforms: every probe answers 'unknown'/'failed'."""

    def active_window(self) -> dict[str, object] | None:
        return None

    def cursor_position(self) -> tuple[int, int] | None:
        return None

    def move_cursor(self, *, x: int, y: int) -> bool:
        del x, y
        return False

    def switch_workspace(self, direction: str) -> bool:
        del direction
        return False

    def is_text_input_focused(self) -> bool | None:
        return None


class WindowsSystemIntegration:
    """Windows desktop integration via the ctypes user32 API."""

    def active_window(self) -> dict[str, object] | None:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None

            buf_title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf_title, 512)

            buf_class = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buf_class, 256)

            class_name = buf_class.value
            return {
                "class": class_name,
                "initialClass": class_name,
                "title": buf_title.value,
            }
        except Exception:
            _LOG.debug("active_window failed on Windows", exc_info=True)
            return None

    def cursor_position(self) -> tuple[int, int] | None:
        try:
            import ctypes
            import ctypes.wintypes

            point = ctypes.wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):  # type: ignore[attr-defined]
                return point.x, point.y
            return None
        except Exception:
            return None

    def move_cursor(self, *, x: int, y: int) -> bool:
        try:
            import ctypes

            return bool(ctypes.windll.user32.SetCursorPos(x, y))  # type: ignore[attr-defined]
        except Exception:
            return False

    def switch_workspace(self, direction: str) -> bool:
        """Switch virtual desktops by synthesizing Win+Ctrl+Left/Right."""
        if direction not in {"left", "right"}:
            return False
        try:
            keyboard_module = importlib.import_module("pynput.keyboard")
        except Exception:
            _LOG.debug(
                "pynput.keyboard unavailable; cannot switch workspace",
                exc_info=True,
            )
            return False

        controller_ctor = getattr(keyboard_module, "Controller")
        key_holder = getattr(keyboard_module, "Key")
        arrow = key_holder.left if direction == "left" else key_holder.right
        controller = controller_ctor()
        try:
            controller.press(key_holder.cmd)
            controller.press(key_holder.ctrl)
            controller.press(arrow)
            controller.release(arrow)
            controller.release(key_holder.ctrl)
            controller.release(key_holder.cmd)
        except Exception:
            _LOG.debug(
                "switch_workspace failed for direction=%s", direction, exc_info=True
            )
            return False
        return True

    def is_text_input_focused(self) -> bool | None:
        # Windows has no cheap "is a text caret focused" probe; return False so
        # the output layer uses the clipboard + paste path instead of typing
        # blindly into whatever control holds focus.
        return False


def create_system_integration() -> SystemIntegration:
    """Return the platform integration: Windows backend, else the Noop stub."""
    if sys.platform == "win32":
        return WindowsSystemIntegration()
    return NoopSystemIntegration()
