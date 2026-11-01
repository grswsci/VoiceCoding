from __future__ import annotations

import logging
import os
import sysconfig
from pathlib import Path
from threading import Lock

from voicecoding.config import AppConfig

_LOG = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "large-v3-turbo"

MODEL_REPO_ALIASES: dict[str, str] = {
    DEFAULT_MODEL_NAME: "deepdml/faster-whisper-large-v3-turbo-ct2",
}


def _register_nvidia_dll_dirs() -> None:
    """Make pip-installed CUDA runtime DLLs visible to the process on Windows."""
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    nvidia_root = Path(purelib) / "nvidia"
    if not nvidia_root.is_dir():
        return
    for entry in sorted(nvidia_root.iterdir()):
        bin_dir = entry / "bin"
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (AttributeError, OSError):
            continue


def _resolve_device(config: AppConfig) -> tuple[str, str]:
    """Pick (device, compute_type) for the model load."""
    if config.device != "cuda":
        return config.device, config.compute_type
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", config.compute_type
    if config.fallback_to_cpu:
        _LOG.warning(
            "CUDA device requested but no CUDA device is available; "
            "falling back to CPU (compute_type=int8)"
        )
        return "cpu", "int8"
    raise RuntimeError(
        "CUDA device requested but no CUDA device is available, "
        "and CPU fallback is disabled"
    )


class WhisperTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self._config: AppConfig = config
        self._model: object | None = None
        self._model_lock: Lock = Lock()
        self.device_in_use: str = config.device
        self.backend_in_use: str = "faster_whisper"

    def transcribe(self, audio_path: Path) -> str:
        self._ensure_model_loaded()
        language = None if self._config.language == "auto" else self._config.language
        try:
            segments = self._run_transcribe(audio_path, language)
        except Exception as error:
            if self.device_in_use != "cuda" or not self._config.fallback_to_cpu:
                raise
            _LOG.warning(
                "CUDA transcription failed (%s); rebuilding model on CPU "
                "and retrying once",
                error,
            )
            with self._model_lock:
                self._install_model("cpu", "int8")
            segments = self._run_transcribe(audio_path, language)
        return "".join(segment.text for segment in segments).strip()

    def prewarm(self) -> None:
        self._ensure_model_loaded()

    def _run_transcribe(self, audio_path: Path, language: str | None):
        model = self._model
        if model is None:
            raise RuntimeError("Whisper model is not initialized")
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._config.beam_size,
            vad_filter=self._config.vad_enabled,
        )
        return segments

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            device, compute_type = _resolve_device(self._config)
            self._install_model(device, compute_type)

    def _install_model(self, device: str, compute_type: str) -> None:
        _register_nvidia_dll_dirs()
        repo = MODEL_REPO_ALIASES.get(self._config.model_name, self._config.model_name)
        from faster_whisper import WhisperModel

        self._model = WhisperModel(repo, device=device, compute_type=compute_type)
        self.device_in_use = device
        _LOG.info(
            "Loaded whisper model %s (device=%s, compute_type=%s)",
            repo,
            device,
            compute_type,
        )
