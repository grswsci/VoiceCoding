"""Application wiring: config, recorder, transcriber, output, and listeners.

Front side button toggles recording; rear button (or the recording-submit
hotkey) stops recording and delivers the transcription locally. All text
output is pure local injection or clipboard staging — nothing leaves the
machine.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
from pathlib import Path

from voicecoding.audio import AudioRecorder, AudioRecording
from voicecoding.config import AppConfig
from voicecoding.keyboard_listener import KeyboardHotkeyListener
from voicecoding.mouse_listener import SideButtonListener
from voicecoding.output import TextOutput
from voicecoding.system_integration import (
    SystemIntegration,
    create_system_integration,
)
from voicecoding.transcriber import WhisperTranscriber

_LOG = logging.getLogger(__name__)


class VoiceMouseApp:
    """Owns the recording pipeline and routes input events to it."""

    def __init__(self, config: AppConfig) -> None:
        if config.front_button == config.rear_button:
            raise ValueError("Front and rear side buttons must be different")

        self._config: AppConfig = config
        self._system_integration: SystemIntegration = create_system_integration()
        self._recorder: AudioRecorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
            temp_dir=config.temp_dir,
            pre_buffer_s=config.pre_buffer_s,
        )
        self._transcriber: WhisperTranscriber = WhisperTranscriber(config)
        self._output: TextOutput = TextOutput(
            system_integration=self._system_integration,
        )
        self._listener: SideButtonListener = SideButtonListener(
            on_front_press=self._on_front_press,
            on_rear_press=self._on_rear_press,
            on_gesture=self._on_gesture,
            front_button=config.front_button,
            rear_button=config.rear_button,
            debounce_s=config.button_debounce_ms / 1000.0,
            gestures_enabled=config.gestures_enabled,
            gesture_trigger_button=config.gesture_trigger_button,
            gesture_threshold_px=config.gesture_threshold_px,
            gesture_freeze_pointer=config.gesture_freeze_pointer,
            gesture_restore_cursor=config.gesture_restore_cursor,
            system_integration=self._system_integration,
        )
        record_keycodes = tuple(config.record_hotkey_keycodes)
        self._keyboard_listener: KeyboardHotkeyListener | None = None
        if record_keycodes:
            self._keyboard_listener = KeyboardHotkeyListener(
                on_hotkey=self._on_front_press,
                keycodes=record_keycodes,
                debounce_s=config.button_debounce_ms / 1000.0,
            )
        self._recording_submit_listener = KeyboardHotkeyListener(
            on_hotkey=self._on_recording_submit_press,
            keycodes=(config.recording_submit_keycode,),
            debounce_s=config.button_debounce_ms / 1000.0,
        )
        self._stop_event: threading.Event = threading.Event()
        self._transcribe_lock: threading.Lock = threading.Lock()
        self._workers_lock: threading.Lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._prewarm_started: bool = False

    def run(self) -> None:
        # Initialize the pynput package tree on the main thread BEFORE any
        # listener thread starts. Two threads importing pynput.mouse and
        # pynput.keyboard concurrently can deadlock on Python's per-module
        # import locks (pynput -> pynput.keyboard while pynput.keyboard's
        # parent chain waits on pynput); after this one-time import the
        # in-thread imports hit sys.modules without touching the locks.
        importlib.import_module("pynput")
        self._listener.start()
        if self._keyboard_listener is not None:
            self._keyboard_listener.start()
        if self._recording_submit_listener is not None:
            self._recording_submit_listener.start()
        self._recorder.open_stream()
        self._set_recording_status(False)
        _LOG.info(
            "VoiceCoding ready. "
            f"auto_paste={self._config.auto_paste}, "
            f"enter_mode={self._config.enter_mode}, "
            f"debounce_ms={self._config.button_debounce_ms}, "
            f"front_button={self._config.front_button}, "
            f"rear_button={self._config.rear_button}, "
            f"record_hotkey_keycodes={self._config.record_hotkey_keycodes}, "
            f"recording_submit_keycode={self._config.recording_submit_keycode}, "
            f"gestures_enabled={self._config.gestures_enabled}, "
            f"gesture_trigger={self._config.gesture_trigger_button}, "
            f"gesture_threshold_px={self._config.gesture_threshold_px}, "
            f"gesture_freeze_pointer={self._config.gesture_freeze_pointer}, "
            f"gesture_restore_cursor={self._config.gesture_restore_cursor}, "
            f"prewarm_on_start={self._config.prewarm_on_start}, "
            f"prewarm_delay_s={self._config.prewarm_delay_s}. "
            "Front button toggles recording; rear button stops recording or "
            "sends Enter when idle."
        )
        self._maybe_prewarm_transcriber()
        try:
            _ = self._stop_event.wait()
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._listener.stop()
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
        if self._recording_submit_listener is not None:
            self._recording_submit_listener.stop()
        self._recorder.cancel()
        self._recorder.close_stream()
        self._set_recording_status(False)
        with self._workers_lock:
            workers = list(self._workers)
        still_running: list[threading.Thread] = []
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                still_running.append(worker)
        if still_running:
            _LOG.warning(
                "Shutdown warning: %d transcription worker(s) are still running",
                len(still_running),
            )

    def _on_front_press(self) -> None:
        if not self._recorder.is_recording:
            try:
                self._recorder.start()
                self._set_recording_status(True)
                _LOG.info("Recording started")
            except Exception as error:
                self._set_recording_status(False)
                _LOG.exception("Failed to start recording: %s", error)
            return

        try:
            recording = self._stop_recording()
        except Exception as error:
            _LOG.exception("Failed to stop recording: %s", error)
            return

        if recording is None:
            return

        self._start_transcription_worker(recording)

    def _on_rear_press(self) -> None:
        if self._recorder.is_recording:
            try:
                recording = self._stop_recording()
            except Exception as error:
                _LOG.exception("Failed to stop recording from rear button: %s", error)
                return

            if recording is None:
                return

            _LOG.info("Recording stopped by rear button")
            self._start_transcription_worker(recording)
            return

        try:
            self._output.send_enter(mode=self._config.enter_mode)
            if self._config.enter_mode == "none":
                _LOG.info("Enter key handling disabled (enter_mode=none)")
            else:
                _LOG.info("Enter key sent")
        except Exception as error:
            _LOG.exception("Failed to send Enter: %s", error)

    def _on_recording_submit_press(self) -> None:
        if not self._recorder.is_recording:
            return
        _LOG.info("Recording submit hotkey pressed, routing to rear-button logic")
        self._on_rear_press()

    def _on_gesture(self, direction: str) -> None:
        action = self._resolve_gesture_action(direction)
        if action == "noop":
            _LOG.info("Gesture '%s' recognized with no action configured", direction)
            return

        if action == "record_toggle":
            _LOG.info("Gesture '%s' -> toggle recording", direction)
            self._on_front_press()
            return

        if action == "send_enter":
            mode = self._config.enter_mode
            if mode == "none":
                mode = "enter"
            try:
                self._output.send_enter(mode=mode)
                _LOG.info("Gesture '%s' -> send enter (%s)", direction, mode)
            except Exception as error:
                _LOG.exception(
                    "Gesture '%s' failed to send enter: %s", direction, error
                )
            return

        if action == "workspace_left":
            if self._switch_workspace("left"):
                _LOG.info("Gesture '%s' -> switch workspace left", direction)
            else:
                _LOG.warning("Gesture '%s' failed to switch workspace left", direction)
            return

        if action == "workspace_right":
            if self._switch_workspace("right"):
                _LOG.info("Gesture '%s' -> switch workspace right", direction)
            else:
                _LOG.warning(
                    "Gesture '%s' failed to switch workspace right", direction
                )
            return

        _LOG.warning("Gesture '%s' mapped to unknown action '%s'", direction, action)

    def _resolve_gesture_action(self, direction: str) -> str:
        mapping = {
            "up": self._config.gesture_up_action,
            "down": self._config.gesture_down_action,
            "left": self._config.gesture_left_action,
            "right": self._config.gesture_right_action,
        }
        return mapping.get(direction, "noop")

    def _switch_workspace(self, direction: str) -> bool:
        try:
            return bool(self._system_integration.switch_workspace(direction))
        except Exception:
            _LOG.debug("Workspace switch failed", exc_info=True)
            return False

    def _stop_recording(self) -> AudioRecording | None:
        try:
            recording = self._recorder.stop_and_save()
        except Exception as error:
            self._set_recording_status(False)
            raise RuntimeError(error) from error

        self._set_recording_status(False)
        if recording is None:
            _LOG.info("Recording was empty and has been discarded")
            return None
        return recording

    def _start_transcription_worker(
        self,
        recording: AudioRecording,
    ) -> None:
        worker = threading.Thread(
            target=self._transcribe_and_output,
            args=(recording,),
            daemon=True,
        )
        with self._workers_lock:
            self._workers.add(worker)
        worker.start()

    def _transcribe_and_output(
        self,
        recording: AudioRecording,
    ) -> None:
        current = threading.current_thread()
        try:
            _LOG.info(
                "Recording stopped (%.1fs), transcribing...", recording.duration_s
            )
            with self._transcribe_lock:
                text = self._transcriber.transcribe(recording.path)

            if not text:
                _LOG.info("No speech recognized")
                return

            route = self._output.inject_or_clipboard(
                text,
                auto_paste=self._config.auto_paste,
            )

            if route == "typed":
                _LOG.info("Transcription typed into focused input")
            elif route == "pasted":
                _LOG.info("Transcription pasted via clipboard")
            elif route == "clipboard":
                _LOG.info("Transcription copied to clipboard")
            else:
                _LOG.warning("Transcription produced no output (route=%s)", route)
        except Exception as error:
            _LOG.exception("Transcription failed: %s", error)
        finally:
            self._safe_unlink(recording.path)
            with self._workers_lock:
                self._workers.discard(current)

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            _LOG.warning("Failed to remove temp audio file %s: %s", path, error)

    def _maybe_prewarm_transcriber(self) -> None:
        if not self._config.prewarm_on_start or self._prewarm_started:
            return
        self._prewarm_started = True

        worker = threading.Thread(
            target=self._prewarm_transcriber,
            args=(self._config.prewarm_delay_s,),
            daemon=True,
        )
        worker.start()

    def _prewarm_transcriber(self, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            _LOG.info("Transcriber prewarm scheduled in %.1fs", delay_s)
            if self._stop_event.wait(timeout=delay_s):
                return

        try:
            self._transcriber.prewarm()
            _LOG.info("Transcriber prewarm complete")
        except Exception as error:
            _LOG.warning("Transcriber prewarm skipped: %s", error)

    def _set_recording_status(self, is_recording: bool) -> None:
        payload = {
            "recording": is_recording,
            "state": "recording" if is_recording else "idle",
        }
        path = self._config.status_file
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_path, path)
        except Exception as error:
            _LOG.warning("Failed to update recording status file %s: %s", path, error)
