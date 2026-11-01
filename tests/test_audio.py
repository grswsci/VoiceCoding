"""Contract tests for voicecoding.audio.AudioRecorder.

Groups: input-device selection (OS-reserved entries — loopback/monitor
sinks and the generic "default" endpoint — must never be picked, while
usable Realtek/USB microphones must be), stream opening at the configured
sample rate even when a device reports an unusable rate, wav output written
through the fake soundfile, and the pre-buffer keeping rolling audio so the
first word is not clipped.

All capture runs against the conftest sounddevice/soundfile fakes; no real
audio device is opened. The fake stream delivers nothing on its own (real
PortAudio feeds callbacks from a separate thread), so tests push capture
blocks through the recorded stream callback explicitly.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import (
    FakeInputStream,
    load_contract_config,
    purge_voicecoding_env,
    sounddevice_mod,
    soundfile_mod,
)


def _make_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str):
    purge_voicecoding_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(f"VOICECODING_{key}", value)
    config = load_contract_config()
    return replace(config, temp_dir=tmp_path / "recordings")


def _make_recorder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str):
    from voicecoding.audio import AudioRecorder

    config = _make_config(monkeypatch, tmp_path, **env)
    return AudioRecorder(
        sample_rate=config.sample_rate,
        channels=config.channels,
        dtype=config.dtype,
        temp_dir=config.temp_dir,
        pre_buffer_s=config.pre_buffer_s,
    )


def _device(name: str, index: int, inputs: int, rate: float) -> dict:
    return {
        "name": name,
        "index": index,
        "max_input_channels": inputs,
        "default_samplerate": rate,
        "hostapi": 0,
    }


def _feed_blocks(blocks: int, frames: int = 1024) -> None:
    """Deliver capture blocks through the callback the recorder registered."""
    assert FakeInputStream.instances, "no input stream was opened"
    stream = FakeInputStream.instances[-1]
    assert callable(stream.kwargs.get("callback")), (
        "recorder registered no capture callback"
    )
    stream.kwargs["blocksize"] = frames  # the package passes no blocksize
    stream.deliver_blocks(blocks)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def test_device_selection_excludes_os_reserved_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reserved = [
        _device("default", 0, 128, 48000.0),
        _device("PulseAudio Monitor of Mic", 1, 32, 48000.0),
    ]
    sounddevice_mod.STATE.devices = reserved + [
        _device("Microphone (Realtek Audio)", 2, 2, 48000.0),
        _device("USB Condenser Microphone", 3, 1, 16000.0),
    ]

    recorder = _make_recorder(monkeypatch, tmp_path)
    recorder.start()

    assert FakeInputStream.instances, "no input stream was opened"
    selected = FakeInputStream.instances[-1].kwargs.get("device")
    assert selected == 2, "an OS-reserved device entry was selected"

    # With the Realtek entry gone, the USB microphone (now sitting right
    # after a reserved entry) is the usable pick.
    sounddevice_mod.STATE.devices = [
        _device("default", 0, 128, 48000.0),
        _device("USB Condenser Microphone", 1, 1, 16000.0),
    ]
    recorder = _make_recorder(monkeypatch, tmp_path)
    recorder.start()

    assert FakeInputStream.instances[-1].kwargs.get("device") == 1


def test_device_selection_keeps_working_with_only_reserved_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sounddevice_mod.STATE.devices = [
        _device("default", 0, 128, 48000.0),
        _device("PulseAudio Monitor of Mic", 1, 32, 48000.0),
    ]
    recorder = _make_recorder(monkeypatch, tmp_path)

    # No usable named entry: the recorder must still open a stream and let
    # the OS default device answer, rather than crash.
    recorder.start()

    assert FakeInputStream.instances, "recorder failed to open any stream"
    assert FakeInputStream.instances[-1].kwargs.get("device") is None


# ---------------------------------------------------------------------------
# Sample-rate handling
# ---------------------------------------------------------------------------


def test_zero_device_sample_rate_falls_back_to_configured_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sounddevice_mod.STATE.devices = [
        _device("Broken Rate Microphone", 0, 1, 0),
    ]
    recorder = _make_recorder(monkeypatch, tmp_path)

    recorder.start()

    assert FakeInputStream.instances, "no input stream was opened"
    used_rate = FakeInputStream.instances[-1].kwargs.get("samplerate")
    assert used_rate == 16000


# ---------------------------------------------------------------------------
# Wav output
# ---------------------------------------------------------------------------


def test_recording_writes_a_wav_via_soundfile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _make_recorder(monkeypatch, tmp_path)

    recorder.start()
    _feed_blocks(blocks=10, frames=1600)
    result = recorder.stop_and_save()

    assert soundfile_mod.STATE.writes, "no wav data was written"
    written = soundfile_mod.STATE.writes[-1]
    assert written["path"].endswith(".wav")
    assert written["samplerate"] > 0

    result_path = Path(str(getattr(result, "path", result)))
    assert result_path.suffix == ".wav"
    assert str(tmp_path / "recordings") in str(result_path)


def test_stop_and_save_reports_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _make_recorder(monkeypatch, tmp_path)

    recorder.start()
    _feed_blocks(blocks=4, frames=1600)
    result = recorder.stop_and_save()

    duration = getattr(result, "duration_s", None)
    assert duration is not None
    assert duration >= 0


# ---------------------------------------------------------------------------
# Pre-buffer
# ---------------------------------------------------------------------------


def test_pre_buffer_keeps_audio_before_the_record_press(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _make_recorder(
        monkeypatch, tmp_path, PRE_BUFFER_S="0.5", SAMPLE_RATE="16000"
    )

    # The persistent stream keeps a rolling ring buffer alive before the
    # record press; blocks arriving then must survive into the saved wav.
    # The ring trims itself to its 8000-frame capacity by dropping whole
    # blocks, so 512-frame blocks are used: sixteen of them leave 7680
    # frames retained, comfortably above the 0.45 s floor.
    recorder.open_stream()
    _feed_blocks(blocks=16, frames=512)
    recorder.start()
    recorder.stop_and_save()

    assert soundfile_mod.STATE.writes, "no wav data was written"
    data = soundfile_mod.STATE.writes[-1]["data"]
    frames = int(getattr(data, "shape", (0,))[0]) if data.ndim else int(len(data))
    expected_floor = int(16000 * 0.5 * 0.9)  # 0.5 s pre-roll, small tolerance
    assert frames >= expected_floor, (
        f"expected at least the {expected_floor}-frame pre-buffer, got {frames}"
    )

