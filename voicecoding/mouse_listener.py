"""Side-button mouse listener: dedicated voice keys plus drag gestures.

Windows-native design (single resident backend):

- Side buttons X1/X2 are swallowed system-wide through the pynput low-level
  hook (``win32_event_filter``), so they work as dedicated push-to-talk keys
  without triggering browser back/forward.
- Gestures are classified from accumulated pointer travel between press and
  release of the configured trigger button. Since Windows has no input-device
  grab, ``gesture_freeze_pointer`` pins the cursor to the gesture-start anchor
  by re-issuing ``SetCursorPos`` after each sampled move; the anchor is
  restored on release when ``gesture_restore_cursor`` is set.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from voicecoding.system_integration import (
    SystemIntegration,
    create_system_integration,
)

ButtonCallback = Callable[[], None]
GestureCallback = Callable[[str], None]
_LOG = logging.getLogger(__name__)

# WM_XBUTTONDOWN / WM_XBUTTONUP: the only messages the low-level hook
# intercepts. Everything else passes through untouched.
_WM_XBUTTONDOWN = 0x020B
_WM_XBUTTONUP = 0x020C

# Driver alias normalization: pynput names the side buttons "x1"/"x2", but
# some drivers surface the raw button numbers or legacy alias spellings.
_BUTTON_ALIASES: dict[str, frozenset[str]] = {
    "x1": frozenset({"x1", "x_button1", "button8"}),
    "x2": frozenset({"x2", "x_button2", "button9"}),
}
_RIGHT_CANDIDATES = frozenset({"right", "button2"})

# Force-finish a gesture whose button-up event was never delivered; without
# this the cursor would stay pinned to the anchor forever.
_GESTURE_TIMEOUT_S = 1.2


def _normalize_side_button(name: str) -> str:
    lowered = name.strip().lower()
    for canonical, aliases in _BUTTON_ALIASES.items():
        if lowered in aliases:
            return canonical
    raise ValueError(
        f"Unknown side button {name!r}; expected one of: x1, x2"
    )


class SideButtonListener:
    """Dispatches front/rear side-button presses and drag-gesture directions."""

    def __init__(
        self,
        on_front_press: ButtonCallback,
        on_rear_press: ButtonCallback,
        front_button: str,
        rear_button: str,
        debounce_s: float = 0.15,
        on_gesture: GestureCallback | None = None,
        gestures_enabled: bool = False,
        gesture_trigger_button: str = "rear",
        gesture_threshold_px: int = 120,
        gesture_freeze_pointer: bool = True,
        gesture_restore_cursor: bool = True,
        system_integration: SystemIntegration | None = None,
        rescan_interval_s: float = 2.0,
    ) -> None:
        del rescan_interval_s  # retained for call-site compatibility
        if gesture_trigger_button not in {"front", "rear", "right"}:
            raise ValueError(
                "gesture_trigger_button must be one of: front, rear, right"
            )
        normalized_front = _normalize_side_button(front_button)
        normalized_rear = _normalize_side_button(rear_button)
        if normalized_front == normalized_rear:
            raise ValueError("Front and rear side buttons must be different")

        self._on_front_press: ButtonCallback = on_front_press
        self._on_rear_press: ButtonCallback = on_rear_press
        self._on_gesture: GestureCallback | None = on_gesture
        self._front_button: str = normalized_front
        self._rear_button: str = normalized_rear
        self._debounce_s: float = max(0.0, debounce_s)
        self._gestures_enabled: bool = gestures_enabled
        self._gesture_trigger_button: str = gesture_trigger_button
        self._gesture_threshold_px: int = max(1, gesture_threshold_px)
        self._gesture_freeze_pointer: bool = gesture_freeze_pointer
        self._gesture_restore_cursor: bool = gesture_restore_cursor
        self._system_integration: SystemIntegration = (
            system_integration
            if system_integration is not None
            else create_system_integration()
        )
        self._last_front_press_monotonic: float = 0.0
        self._last_rear_press_monotonic: float = 0.0
        self._debounce_lock: threading.Lock = threading.Lock()
        self._gesture_lock: threading.Lock = threading.Lock()
        self._gesture_active: bool = False
        self._gesture_dx: int = 0
        self._gesture_dy: int = 0
        self._gesture_last_position: tuple[int, int] | None = None
        self._gesture_anchor_cursor: tuple[int, int] | None = None
        self._gesture_started_since: float | None = None
        self._gesture_trigger_label: str | None = None
        # True while a re-anchoring SetCursorPos echo is still in flight; the
        # matching on_move event must not be accumulated as travel.
        self._gesture_pin_pending: bool = False
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        # Holder for the live pynput listener; the win32 hook filter needs it
        # to call suppress_event(), and it only exists after construction.
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
        self._run_pynput()

    def _run_pynput(self, *, timeout_s: float | None = None) -> None:
        try:
            mouse_module = importlib.import_module("pynput.mouse")
        except Exception as error:
            raise RuntimeError("pynput.mouse is not available") from error

        listener_ctor = getattr(mouse_module, "Listener")
        listener_kwargs: dict[str, object] = {
            "on_click": self.on_click,
            "on_move": self.on_move,
        }
        if sys.platform == "win32":
            # pynput's docs call this option "win32_event_filter", but the
            # implementation reads options under "event_filter" (both names
            # appear across released versions); pass both — unknown options
            # are kept in _options and ignored harmlessly.
            listener_kwargs["win32_event_filter"] = self._win32_event_filter
            listener_kwargs["event_filter"] = self._win32_event_filter

        listener = listener_ctor(**listener_kwargs)
        self._pynput_listener = listener
        _LOG.info("Mouse listener using pynput backend")
        listener.start()
        deadline: float | None = None
        if timeout_s is not None:
            deadline = time.monotonic() + max(0.2, timeout_s)
        try:
            while not self._stop.is_set():
                self._release_stale_gesture_capture()
                if deadline is not None and time.monotonic() >= deadline:
                    return
                time.sleep(0.2)
        finally:
            self._gesture_pin_pending = False
            listener.stop()
            self._pynput_listener = None

    def _win32_event_filter(self, msg: int, data: object) -> None:
        """Low-level mouse hook: swallow X1/X2 so side buttons stay dedicated.

        Runs inside the WH_MOUSE_LL hook callback. pynput 1.8.x implements
        ``suppress_event()`` by raising SystemHook.SuppressException, which
        short-circuits the remaining dispatch — calling it from the filter
        would swallow the event but on_click would never fire. So this
        filter performs the routing itself (a full on_click dispatch) and
        suppresses afterwards; pynput's hook handler catches the raised
        suppression exception and returns 1 to stop system propagation.
        Zero I/O, zero sleeps: slow callbacks get the hook removed by the
        system and cause global pointer stutter.
        """
        if msg not in (_WM_XBUTTONDOWN, _WM_XBUTTONUP):
            return
        pressed = msg == _WM_XBUTTONDOWN
        pt = getattr(data, "pt", None)
        if pt is None:
            return
        xbutton = (getattr(data, "mouseData", 0) >> 16) & 0xFFFF
        mouse_module = importlib.import_module("pynput.mouse")
        button = getattr(mouse_module.Button, "x1" if xbutton == 1 else "x2")
        try:
            self.on_click(pt.x, pt.y, button, pressed)
        finally:
            listener = self._pynput_listener
            if listener is not None:
                listener.suppress_event()

    def on_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        btn_name = str(button).lower().split(".")[-1]
        button_label: str | None = None
        if btn_name in _BUTTON_ALIASES[self._front_button]:
            button_label = "front"
        elif btn_name in _BUTTON_ALIASES[self._rear_button]:
            button_label = "rear"
        elif (
            self._gestures_enabled
            and self._gesture_trigger_button == "right"
            and btn_name in _RIGHT_CANDIDATES
        ):
            button_label = "right"

        if button_label is None:
            return

        if self._gestures_enabled and button_label == self._gesture_trigger_button:
            if pressed:
                self._start_gesture_capture(
                    initial_position=(x, y),
                    button_label=button_label,
                )
            else:
                self._finish_gesture_capture(button_label)
            return

        if pressed:
            self._dispatch_click_async(button_label)

    def on_move(self, x: int, y: int) -> None:
        if not self._gestures_enabled:
            return
        if self._consume_reanchor_echo(x, y):
            return
        self._accumulate_gesture_position(x, y)
        if self._gesture_freeze_pointer:
            self._pin_cursor_to_anchor()

    def _consume_reanchor_echo(self, x: int, y: int) -> bool:
        """Drop the on_move echo produced by our own re-anchoring cursor move."""
        with self._gesture_lock:
            if not self._gesture_pin_pending:
                return False
            self._gesture_pin_pending = False
            anchor = self._gesture_anchor_cursor
            active = self._gesture_active
        return active and anchor is not None and anchor == (x, y)

    def _pin_cursor_to_anchor(self) -> None:
        with self._gesture_lock:
            if not self._gesture_active or self._gesture_pin_pending:
                return
            anchor = self._gesture_anchor_cursor
            last = self._gesture_last_position
        if anchor is None or last is None or last == anchor:
            return
        try:
            moved = bool(
                self._system_integration.move_cursor(x=anchor[0], y=anchor[1])
            )
        except Exception:
            moved = False
        if not moved:
            return
        # After the pin the OS cursor sits at the anchor, so future travel
        # must be measured from there, not from the pre-pin position.
        with self._gesture_lock:
            if self._gesture_active:
                self._gesture_last_position = anchor
                self._gesture_pin_pending = True

    def _start_gesture_capture(
        self,
        *,
        initial_position: tuple[int, int] | None = None,
        button_label: str | None = None,
    ) -> None:
        # The right trigger keeps native click behavior, so it never gets a
        # pin anchor: no freeze, no restore.
        capture_anchor = (
            self._gesture_freeze_pointer or self._gesture_restore_cursor
        ) and button_label != "right"
        anchor: tuple[int, int] | None = initial_position if capture_anchor else None
        if capture_anchor and anchor is None:
            anchor = self._read_cursor_position()
        with self._gesture_lock:
            self._gesture_active = True
            self._gesture_dx = 0
            self._gesture_dy = 0
            self._gesture_last_position = initial_position
            self._gesture_started_since = time.monotonic()
            self._gesture_trigger_label = button_label
            self._gesture_pin_pending = False
            self._gesture_anchor_cursor = anchor

    def _accumulate_gesture_position(self, x: int, y: int) -> None:
        with self._gesture_lock:
            if not self._gesture_active:
                return
            if self._gesture_last_position is None:
                self._gesture_last_position = (x, y)
                return
            last_x, last_y = self._gesture_last_position
            self._gesture_dx += x - last_x
            self._gesture_dy += y - last_y
            self._gesture_last_position = (x, y)

    def _finish_gesture_capture(self, button_label: str) -> None:
        with self._gesture_lock:
            if not self._gesture_active:
                return
            dx = self._gesture_dx
            dy = self._gesture_dy
            self._gesture_active = False
            self._gesture_dx = 0
            self._gesture_dy = 0
            self._gesture_last_position = None
            self._gesture_started_since = None
            self._gesture_trigger_label = None
            self._gesture_pin_pending = False
            anchor_cursor = self._gesture_anchor_cursor
            self._gesture_anchor_cursor = None

        direction = self._classify_gesture(dx, dy, self._gesture_threshold_px)
        _LOG.debug(
            "Gesture capture finished: button=%s dx=%s dy=%s direction=%s",
            button_label,
            dx,
            dy,
            direction,
        )
        if direction is None:
            if button_label == "right":
                # The right button is not hooked, so the native click already
                # reached the application; nothing to replay.
                return
            self._dispatch_click_async(button_label)
            return
        self._dispatch_gesture(direction)
        if anchor_cursor is not None:
            self._restore_cursor_position(anchor_cursor)

    def _release_stale_gesture_capture(self) -> None:
        with self._gesture_lock:
            if not self._gesture_active:
                return
            started_since = self._gesture_started_since
            if started_since is None:
                return
            elapsed_s = time.monotonic() - started_since
            if elapsed_s < _GESTURE_TIMEOUT_S:
                return
            self._gesture_active = False
            self._gesture_dx = 0
            self._gesture_dy = 0
            self._gesture_last_position = None
            self._gesture_anchor_cursor = None
            self._gesture_started_since = None
            self._gesture_trigger_label = None
            self._gesture_pin_pending = False
        _LOG.warning(
            "Gesture capture timeout reached (%.2fs); discarding gesture",
            elapsed_s,
        )

    def _dispatch_gesture(self, direction: str) -> None:
        callback = self._on_gesture
        if callback is None:
            return
        try:
            callback(direction)
        except Exception:
            _LOG.exception("Gesture callback failed: direction=%s", direction)

    def _restore_cursor_position(self, position: tuple[int, int]) -> None:
        try:
            self._system_integration.move_cursor(x=position[0], y=position[1])
        except Exception:
            _LOG.debug("Failed to restore cursor position", exc_info=True)

    def _read_cursor_position(self) -> tuple[int, int] | None:
        try:
            return self._system_integration.cursor_position()
        except Exception:
            return None

    def _dispatch_click(self, button_label: str) -> None:
        if button_label == "front":
            self._dispatch_front_press()
            return
        if button_label == "rear":
            self._dispatch_rear_press()

    def _dispatch_click_async(self, button_label: str) -> None:
        worker = threading.Thread(
            target=self._invoke_click_callback,
            args=(button_label,),
            daemon=True,
        )
        worker.start()

    def _invoke_click_callback(self, button_label: str) -> None:
        try:
            self._dispatch_click(button_label)
        except Exception:
            _LOG.exception("Mouse click callback failed: button=%s", button_label)

    def _dispatch_front_press(self) -> None:
        if self._should_fire_front():
            self._on_front_press()

    def _dispatch_rear_press(self) -> None:
        if self._should_fire_rear():
            self._on_rear_press()

    def _should_fire_front(self) -> bool:
        now = time.monotonic()
        with self._debounce_lock:
            if now - self._last_front_press_monotonic < self._debounce_s:
                return False
            self._last_front_press_monotonic = now
            return True

    def _should_fire_rear(self) -> bool:
        now = time.monotonic()
        with self._debounce_lock:
            if now - self._last_rear_press_monotonic < self._debounce_s:
                return False
            self._last_rear_press_monotonic = now
            return True

    @staticmethod
    def _classify_gesture(dx: int, dy: int, threshold_px: int) -> str | None:
        if max(abs(dx), abs(dy)) < threshold_px:
            return None
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"
