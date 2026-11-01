"""Recording hotkey listener built on the pynput keyboard backend.

Fires ``on_hotkey`` when every configured virtual-key code is held at the
same time. The latch guarantees one callback per physical press: the combo
must be fully released before it can fire again, and a debounce window
suppresses key-repeat chatter.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger(__name__)

HotkeyCallback = Callable[[], None]


class KeyboardHotkeyListener:
    """Watches a virtual-keycode chord and invokes the callback once per press."""

    def __init__(
        self,
        *,
        on_hotkey: HotkeyCallback,
        keycodes: tuple[int, ...],
        debounce_s: float = 0.15,
    ) -> None:
        if not keycodes:
            raise ValueError("keycodes must not be empty")
        self._on_hotkey: HotkeyCallback = on_hotkey
        self._combo: frozenset[int] = frozenset(keycodes)
        self._debounce_s: float = max(0.0, debounce_s)
        self._state_lock: threading.Lock = threading.Lock()
        self._pressed: set[int] = set()
        self._combo_latched: bool = False
        self._last_fire_monotonic: float = 0.0
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynput_listener: Any = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            keyboard_module = importlib.import_module("pynput.keyboard")
        except Exception as error:
            _LOG.warning("Keyboard hotkey listener unavailable: %s", error)
            return

        listener_ctor = getattr(keyboard_module, "Listener")
        listener = listener_ctor(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self._pynput_listener = listener
        listener.start()
        try:
            while not self._stop.wait(timeout=0.2):
                pass
        finally:
            self._reset_pressed_state()
            listener.stop()
            self._pynput_listener = None

    def on_press(self, key: object) -> None:
        keycode = self._keycode_of(key)
        if keycode is None:
            return
        if self._process_key_event(keycode, 1):
            self._invoke_hotkey()

    def on_release(self, key: object) -> None:
        keycode = self._keycode_of(key)
        if keycode is None:
            return
        self._process_key_event(keycode, 0)

    def _invoke_hotkey(self) -> None:
        try:
            self._on_hotkey()
        except Exception:
            _LOG.exception("Hotkey callback failed")

    @staticmethod
    def _keycode_of(key: object) -> int | None:
        keycode = getattr(key, "vk", None)
        return keycode if isinstance(keycode, int) else None

    def _reset_pressed_state(self) -> None:
        with self._state_lock:
            self._pressed.clear()
            self._combo_latched = False

    def _process_key_event(self, keycode: int, value: int) -> bool:
        """Latch logic: press adds, release removes, full release re-arms."""
        with self._state_lock:
            if value == 1:
                self._pressed.add(keycode)
            elif value == 0:
                self._pressed.discard(keycode)
            else:
                return False

            if self._combo_latched and not self._combo.issubset(self._pressed):
                self._combo_latched = False

            now = time.monotonic()
            if (
                not self._combo_latched
                and self._combo.issubset(self._pressed)
                and now - self._last_fire_monotonic >= self._debounce_s
            ):
                self._combo_latched = True
                self._last_fire_monotonic = now
                return True
            return False
