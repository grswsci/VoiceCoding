"""Contract tests for voicecoding.transcriber.WhisperTranscriber.

Nine groups: CUDA device/float16 selection, no-CUDA CPU fallback, model-name
mapping, transcribe keyword pass-through, segment joining, NVIDIA DLL
registration, prewarm idempotence, CUDA failure retried exactly once on CPU,
and the hard failure when CUDA is required but absent.

All model/device behaviour runs against the faster_whisper / ctranslate2
fakes installed by conftest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from conftest import (
    FakeWhisperModel,
    faster_whisper_mod,
    load_contract_config,
    purge_voicecoding_env,
)

MODEL_MAP_ENTRY = "deepdml/faster-whisper-large-v3-turbo-ct2"


def _make_config(monkeypatch: pytest.MonkeyPatch, **env: str):
    # Purge before setting overrides: load_contract_config() with a
    # monkeypatch argument purges the environment itself, which would wipe
    # any variable set beforehand.
    purge_voicecoding_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(f"VOICECODING_{key}", value)
    return load_contract_config()


def _make_transcriber(monkeypatch: pytest.MonkeyPatch, **env: str):
    from voicecoding.transcriber import WhisperTranscriber

    return WhisperTranscriber(_make_config(monkeypatch, **env))


# ---------------------------------------------------------------------------
# 1. CUDA available -> requested device + float16
# ---------------------------------------------------------------------------


def test_cuda_available_uses_cuda_and_float16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctranslate2_mod = sys.modules["ctranslate2"]
    ctranslate2_mod.STATE.cuda_count = 1

    transcriber = _make_transcriber(monkeypatch)

    assert transcriber.device_in_use == "cuda"
    assert transcriber.backend_in_use == "faster_whisper"

    transcriber.prewarm()
    assert len(FakeWhisperModel.instances) == 1
    model = FakeWhisperModel.instances[0]
    assert model.device == "cuda"
    assert model.compute_type == "float16"


# ---------------------------------------------------------------------------
# 2. No CUDA + fallback enabled -> cpu / int8
# ---------------------------------------------------------------------------


def test_no_cuda_falls_back_to_cpu_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    ctranslate2_mod = sys.modules["ctranslate2"]
    ctranslate2_mod.STATE.cuda_count = 0

    transcriber = _make_transcriber(monkeypatch, FALLBACK_CPU="true")

    transcriber.prewarm()
    assert len(FakeWhisperModel.instances) == 1
    model = FakeWhisperModel.instances[0]
    assert model.device == "cpu"
    assert model.compute_type == "int8"
    assert transcriber.device_in_use == "cpu"


# ---------------------------------------------------------------------------
# 3. Model-name mapping
# ---------------------------------------------------------------------------


def test_turbo_model_name_maps_to_ct2_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber = _make_transcriber(monkeypatch)

    transcriber.prewarm()
    assert len(FakeWhisperModel.instances) == 1
    assert (
        FakeWhisperModel.instances[0].model_size_or_path == MODEL_MAP_ENTRY
    )


# ---------------------------------------------------------------------------
# 4. Transcribe keyword pass-through
# ---------------------------------------------------------------------------


def test_transcribe_passes_language_beam_and_vad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcriber = _make_transcriber(monkeypatch)

    text = transcriber.transcribe(str(tmp_path / "clip.wav"))

    assert text == "hello world"
    assert len(faster_whisper_mod.STATE.transcribe_calls) == 1
    _, kwargs = faster_whisper_mod.STATE.transcribe_calls[0]
    assert "language" in kwargs
    assert kwargs["language"] is None  # config language "auto" -> None
    assert kwargs.get("beam_size") == 5
    assert kwargs.get("vad_filter") is True


# ---------------------------------------------------------------------------
# 5. Segment joining
# ---------------------------------------------------------------------------


def test_segments_are_joined_into_one_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import FakeSegment, FakeTranscriptionInfo

    faster_whisper_mod.STATE.results.append(
        (
            [FakeSegment("good "), FakeSegment("morning"), FakeSegment(" world")],
            FakeTranscriptionInfo(language="en"),
        )
    )

    transcriber = _make_transcriber(monkeypatch)
    text = transcriber.transcribe(str(tmp_path / "clip.wav"))

    assert " ".join(text.split()) == "good morning world"
    assert text.strip() == text


# ---------------------------------------------------------------------------
# 6. NVIDIA DLL directory registration
# ---------------------------------------------------------------------------


def _install_fake_dll_dir_registrar(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    registered: list[str] = []

    def _fake_add_dll_directory(path):
        registered.append(str(path))
        from types import SimpleNamespace

        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        os, "add_dll_directory", _fake_add_dll_directory, raising=False
    )
    return registered


def test_register_nvidia_dll_dirs_adds_each_bin_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sysconfig

    registered = _install_fake_dll_dir_registrar(monkeypatch)

    # The registration globs <purelib>/nvidia/*/bin, so point sysconfig at
    # the fake site directory.
    site_dir = tmp_path / "site-packages"
    for package in ("cublas", "cudnn"):
        bin_dir = site_dir / "nvidia" / package / "bin"
        bin_dir.mkdir(parents=True)

    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site_dir)})

    transcriber = _make_transcriber(monkeypatch)
    transcriber.prewarm()

    expected = {
        str((site_dir / "nvidia" / "cublas" / "bin").resolve()),
        str((site_dir / "nvidia" / "cudnn" / "bin").resolve()),
    }
    registered_resolved = {str(Path(p).resolve()) for p in registered}
    assert expected <= registered_resolved


def test_register_nvidia_dll_dirs_noops_without_nvidia_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sysconfig

    registered = _install_fake_dll_dir_registrar(monkeypatch)

    empty_site = tmp_path / "empty-site"
    empty_site.mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(empty_site)})

    transcriber = _make_transcriber(monkeypatch)
    transcriber.prewarm()

    assert registered == []


# ---------------------------------------------------------------------------
# 7. Prewarm idempotence (double-checked model construction)
# ---------------------------------------------------------------------------


def test_prewarm_builds_model_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber = _make_transcriber(monkeypatch)

    transcriber.prewarm()
    transcriber.prewarm()
    transcriber.prewarm()

    assert len(FakeWhisperModel.instances) == 1


def test_transcribe_after_prewarm_reuses_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcriber = _make_transcriber(monkeypatch)

    transcriber.prewarm()
    transcriber.transcribe(str(tmp_path / "one.wav"))
    transcriber.transcribe(str(tmp_path / "two.wav"))

    assert len(FakeWhisperModel.instances) == 1
    assert len(faster_whisper_mod.STATE.transcribe_calls) == 2


# ---------------------------------------------------------------------------
# 8. CUDA transcribe failure -> CPU rebuild, retried exactly once
# ---------------------------------------------------------------------------


def test_cuda_failure_rebuilds_on_cpu_and_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    faster_whisper_mod.STATE.transcribe_errors.append(
        RuntimeError("CUDA error: out of memory")
    )

    transcriber = _make_transcriber(monkeypatch)
    text = transcriber.transcribe(str(tmp_path / "clip.wav"))

    assert text == "hello world"
    assert len(FakeWhisperModel.instances) == 2
    assert FakeWhisperModel.instances[0].device == "cuda"
    assert FakeWhisperModel.instances[1].device == "cpu"
    assert FakeWhisperModel.instances[1].compute_type == "int8"
    assert len(faster_whisper_mod.STATE.transcribe_calls) == 2
    assert transcriber.device_in_use == "cpu"


def test_second_cuda_failure_propagates_after_single_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    faster_whisper_mod.STATE.transcribe_errors.append(
        RuntimeError("CUDA error: first failure")
    )
    faster_whisper_mod.STATE.transcribe_errors.append(
        RuntimeError("CPU error: second failure")
    )

    transcriber = _make_transcriber(monkeypatch)
    with pytest.raises(RuntimeError):
        transcriber.transcribe(str(tmp_path / "clip.wav"))

    # Exactly one retry: two models built, two transcribe attempts, no third.
    assert len(FakeWhisperModel.instances) == 2
    assert len(faster_whisper_mod.STATE.transcribe_calls) == 2


# ---------------------------------------------------------------------------
# 9. No CUDA + fallback disabled -> RuntimeError
# ---------------------------------------------------------------------------


def test_no_cuda_without_fallback_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctranslate2_mod = sys.modules["ctranslate2"]
    ctranslate2_mod.STATE.cuda_count = 0

    config = _make_config(monkeypatch, FALLBACK_CPU="false")

    from voicecoding.transcriber import WhisperTranscriber

    # The error may surface at construction or at first model build; both
    # satisfy the contract.
    with pytest.raises(RuntimeError):
        transcriber = WhisperTranscriber(config)
        transcriber.prewarm()
