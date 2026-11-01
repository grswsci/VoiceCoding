from __future__ import annotations

import importlib
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

_LOG = logging.getLogger(__name__)

AudioFrame = NDArray[np.float32]


@dataclass
class AudioRecording:
    path: Path
    duration_s: float


class _AudioStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _SoundDeviceModule(Protocol):
    def InputStream(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        device: int | str | None,
        callback: Callable[[AudioFrame, int, object, object], None],
    ) -> _AudioStream: ...

    def query_devices(self) -> object: ...


class _SoundFileModule(Protocol):
    def write(self, file: str | Path, data: AudioFrame, samplerate: int) -> None: ...


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        temp_dir: Path,
        pre_buffer_s: float = 0.5,
    ) -> None:
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._temp_dir: Path = temp_dir
        self._pre_buffer_s: float = max(pre_buffer_s, 0.0)
        self._sd: _SoundDeviceModule | None = None
        self._sf: _SoundFileModule | None = None
        self._lock: threading.Lock = threading.Lock()
        self._frames: list[AudioFrame] = []
        self._stream: _AudioStream | None = None
        self._recording: bool = False
        self._selected_input_device: int | str | None = None
        self._active_sample_rate: int = sample_rate
        self._persistent_stream: _AudioStream | None = None
        self._ring: deque[AudioFrame] = deque()
        self._ring_samples: int = 0
        self._max_ring_samples: int = int(sample_rate * self._pre_buffer_s)

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def open_stream(self) -> None:
        """Open a persistent audio stream that feeds the ring buffer."""
        if self._pre_buffer_s <= 0:
            _LOG.info("Pre-buffer disabled (pre_buffer_s=0)")
            return
        self._ensure_audio_modules()
        with self._lock:
            if self._persistent_stream is not None:
                return
            if self._sd is None:
                raise RuntimeError("Audio input module not initialized")
            device = self._resolve_input_device()
            samplerate = self._sample_rate
            try:
                stream = self._sd.InputStream(
                    samplerate=samplerate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=device,
                    callback=self._callback,
                )
            except Exception as first_error:
                fallback_rate = self._resolve_device_sample_rate(device)
                if fallback_rate is None or fallback_rate == samplerate:
                    raise
                stream = self._sd.InputStream(
                    samplerate=fallback_rate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=device,
                    callback=self._callback,
                )
                samplerate = fallback_rate
                _LOG.warning(
                    "Persistent stream samplerate fallback: %s -> %s (%s)",
                    self._sample_rate,
                    fallback_rate,
                    first_error,
                )
            stream.start()
            self._persistent_stream = stream
            self._active_sample_rate = samplerate
            self._max_ring_samples = int(samplerate * self._pre_buffer_s)
            _LOG.info(
                "Audio stream opened (persistent): device=%s sample_rate=%s pre_buffer_s=%.2f",
                device,
                samplerate,
                self._pre_buffer_s,
            )

    def close_stream(self) -> None:
        """Close the persistent audio stream."""
        with self._lock:
            stream = self._persistent_stream
            self._persistent_stream = None
            self._ring.clear()
            self._ring_samples = 0
        if stream is not None:
            stream.stop()
            stream.close()
            _LOG.info("Persistent audio stream closed")

    def start(self) -> None:
        self._ensure_audio_modules()
        with self._lock:
            if self._recording:
                return
            try:
                self._temp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise RuntimeError(
                    f"Failed to create temp audio directory {self._temp_dir}: {error}"
                ) from error

            if self._persistent_stream is not None:
                # Use the already-open persistent stream; prepend ring buffer
                self._frames = list(self._ring)
                self._ring.clear()
                self._ring_samples = 0
                self._recording = True
                _LOG.info(
                    "Recording started (persistent stream, pre-buffered %d frames)",
                    len(self._frames),
                )
                return

            self._frames = []
            if self._sd is None:
                raise RuntimeError("Audio input module not initialized")
            device = self._resolve_input_device()
            samplerate = self._sample_rate
            try:
                stream = self._sd.InputStream(
                    samplerate=samplerate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=device,
                    callback=self._callback,
                )
            except Exception as first_error:
                fallback_rate = self._resolve_device_sample_rate(device)
                if fallback_rate is None or fallback_rate == samplerate:
                    raise
                stream = self._sd.InputStream(
                    samplerate=fallback_rate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=device,
                    callback=self._callback,
                )
                samplerate = fallback_rate
                _LOG.warning(
                    "Audio samplerate fallback applied: %s -> %s (%s)",
                    self._sample_rate,
                    fallback_rate,
                    first_error,
                )
            self._stream = stream
            self._recording = True
            self._active_sample_rate = samplerate
        # The PortAudio callback thread takes self._lock on its first invocation,
        # so the stream must be started with the lock released; a blocked audio
        # callback drops input frames. Roll back state if the device fails here.
        try:
            stream.start()
        except Exception:
            with self._lock:
                if self._stream is stream:
                    self._stream = None
                    self._recording = False
            stream.close()
            raise
        _LOG.info(
            "Audio recording stream started: device=%s sample_rate=%s channels=%s",
            device,
            samplerate,
            self._channels,
        )

    def stop_and_save(self) -> AudioRecording | None:
        with self._lock:
            if not self._recording:
                return None
            stream = self._stream
            self._stream = None
            self._recording = False

        if stream is not None:
            stream.stop()
            stream.close()
            _LOG.info("Audio recording stream stopped")
        else:
            _LOG.info("Recording stopped (persistent stream remains open)")

        with self._lock:
            if not self._frames:
                return None
            audio = np.concatenate(self._frames, axis=0)
            self._frames = []

        out_path = self._temp_dir / f"recording_{uuid4().hex}.wav"
        if self._sf is None:
            raise RuntimeError("Audio write module not initialized")
        try:
            self._sf.write(out_path, audio, self._active_sample_rate)
        except Exception as error:
            raise RuntimeError(
                f"Failed to write recording to {out_path}: {error}"
            ) from error
        duration = float(len(audio) / self._active_sample_rate)
        _LOG.info("Audio recording saved: path=%s duration_s=%.2f", out_path, duration)
        return AudioRecording(path=out_path, duration_s=duration)

    def cancel(self) -> None:
        with self._lock:
            if not self._recording:
                self._frames = []
                return
            stream = self._stream
            self._stream = None
            self._recording = False
            self._frames = []

        if stream is not None:
            stream.stop()
            stream.close()
        # persistent stream stays open

    def _callback(
        self, indata: AudioFrame, frames: int, time_data: object, status: object
    ) -> None:
        del frames
        del time_data
        del status
        frame = indata.copy()
        with self._lock:
            if self._recording:
                self._frames.append(frame)
            elif self._persistent_stream is not None and self._max_ring_samples > 0:
                self._ring.append(frame)
                self._ring_samples += len(frame)
                while self._ring_samples > self._max_ring_samples and self._ring:
                    removed = self._ring.popleft()
                    self._ring_samples -= len(removed)

    def _ensure_audio_modules(self) -> None:
        if self._sd is not None and self._sf is not None:
            return
        try:
            sounddevice_module = importlib.import_module("sounddevice")
            soundfile_module = importlib.import_module("soundfile")
        except Exception as error:
            raise RuntimeError(
                "Audio dependencies missing. Install sounddevice and soundfile."
            ) from error

        self._sd = cast(_SoundDeviceModule, cast(object, sounddevice_module))
        self._sf = cast(_SoundFileModule, cast(object, soundfile_module))

    def _resolve_input_device(self) -> int | str | None:
        if self._selected_input_device is not None:
            return self._selected_input_device
        if self._sd is None:
            return None

        query_devices = getattr(self._sd, "query_devices", None)
        if not callable(query_devices):
            return None

        try:
            devices_obj = query_devices()
        except Exception:
            return None

        devices = _coerce_device_list(devices_obj)
        if devices is None:
            return None

        default_index: int | None = None
        default_attr = getattr(self._sd, "default", None)
        default_device = getattr(default_attr, "device", None)
        if isinstance(default_device, list | tuple) and default_device:
            candidate = default_device[0]
            if isinstance(candidate, int):
                default_index = candidate

        def _name(index: int) -> str:
            if index < 0 or index >= len(devices):
                return ""
            item = devices[index]
            if not isinstance(item, Mapping):
                return ""
            raw = item.get("name", "")
            return raw if isinstance(raw, str) else ""

        if default_index is not None:
            default_name = _name(default_index).lower()
            if "monitor" not in default_name:
                self._selected_input_device = default_index
                return default_index

        virtual_names = {
            "default",
        }
        for idx, entry in enumerate(devices):
            if not isinstance(entry, Mapping):
                continue
            max_inputs = entry.get("max_input_channels", 0)
            if not isinstance(max_inputs, int | float) or max_inputs <= 0:
                continue
            name_obj = entry.get("name", "")
            if not isinstance(name_obj, str):
                continue
            name = name_obj.lower()
            if "monitor" in name or name.strip() in virtual_names:
                continue
            self._selected_input_device = idx
            return idx

        return None

    def _resolve_device_sample_rate(self, device: int | str | None) -> int | None:
        if self._sd is None:
            return None
        query_devices = getattr(self._sd, "query_devices", None)
        if not callable(query_devices):
            return None
        try:
            devices_obj = query_devices()
        except Exception:
            return None
        devices = _coerce_device_list(devices_obj)
        if devices is None:
            return None

        if isinstance(device, int) and 0 <= device < len(devices):
            entry = devices[device]
            if isinstance(entry, Mapping):
                raw = entry.get("default_samplerate")
                if isinstance(raw, int | float) and raw > 0:
                    return int(raw)
        return None


def _coerce_device_list(devices_obj: object) -> list[object] | None:
    if isinstance(devices_obj, list):
        return devices_obj
    if isinstance(devices_obj, tuple):
        return list(devices_obj)
    if isinstance(devices_obj, Iterable):
        return list(devices_obj)
    return None
