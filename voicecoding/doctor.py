from __future__ import annotations

import argparse
import importlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from voicecoding.config import AppConfig, load_config

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def run_doctor(*, apply_fixes: bool = False) -> int:
    if apply_fixes:
        _apply_doctor_fixes()

    checks: list[DoctorCheck] = []

    config_check, config = _check_config_load()
    checks.append(config_check)

    checks.append(_check_transcriber())
    checks.append(_check_audio_input(config))
    checks.append(_check_autostart())

    _print_checks(checks)

    fail_count = sum(1 for check in checks if check.status == "fail")
    warn_count = sum(1 for check in checks if check.status == "warn")
    print(f"Doctor summary: {len(checks)} checks, {fail_count} fail, {warn_count} warn")
    return 1 if fail_count else 0


def _apply_doctor_fixes() -> None:
    _fix_autostart()


def _fix_autostart() -> None:
    """Rebuild the autostart artifacts by delegating to the deploy module."""
    try:
        from voicecoding.deploy import configure_deploy_parser, run_deploy
    except Exception as error:
        _LOG.warning("Cannot load deploy module for autostart repair: %s", error)
        return

    parser = argparse.ArgumentParser(prog="voicecoding deploy")
    configure_deploy_parser(parser)
    args = parser.parse_args([])
    args.dry_run = False
    args.skip_autostart = False

    try:
        run_deploy(args)
    except Exception as error:
        _LOG.warning("Autostart repair failed: %s", error)


def _check_config_load() -> tuple[DoctorCheck, AppConfig | None]:
    try:
        config = load_config()
    except Exception as error:
        return (
            DoctorCheck(
                name="config",
                status="fail",
                detail=f"failed to load config: {error}",
            ),
            None,
        )

    return (
        DoctorCheck(
            name="config",
            status="ok",
            detail=(
                "loaded "
                + f"front={config.front_button}, rear={config.rear_button}, "
                + f"model={config.model_name}, device={config.device}"
            ),
        ),
        config,
    )


def _check_transcriber() -> DoctorCheck:
    try:
        faster_whisper = importlib.import_module("faster_whisper")
        ctranslate2 = importlib.import_module("ctranslate2")
    except Exception as error:
        return DoctorCheck(
            name="transcriber",
            status="fail",
            detail=f"cannot import transcription stack (faster-whisper / ctranslate2): {error}",
        )

    fw_version = getattr(faster_whisper, "__version__", "unknown")
    ct_version = getattr(ctranslate2, "__version__", "unknown")
    versions = f"faster-whisper={fw_version}, ctranslate2={ct_version}"

    try:
        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception as error:
        return DoctorCheck(
            name="transcriber",
            status="warn",
            detail=f"{versions}; CUDA probe failed: {error}",
        )

    if cuda_devices <= 0:
        return DoctorCheck(
            name="transcriber",
            status="warn",
            detail=(
                f"{versions}; no CUDA device detected "
                "(transcription will fall back to CPU if enabled)"
            ),
        )

    cached = _find_cached_whisper_model()
    if cached is None:
        return DoctorCheck(
            name="transcriber",
            status="warn",
            detail=(
                f"{versions}; CUDA devices={cuda_devices}; "
                "model not found in the local cache — the first run downloads "
                "~1.6 GB (set HF_ENDPOINT=https://hf-mirror.com to speed up the download)"
            ),
        )

    return DoctorCheck(
        name="transcriber",
        status="ok",
        detail=f"{versions}; CUDA devices={cuda_devices}; model cached at {cached}",
    )


def _hf_cache_candidates() -> list[Path]:
    candidates: list[Path] = []
    hub_cache_env = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if hub_cache_env:
        candidates.append(Path(hub_cache_env))
    hf_home_env = os.environ.get("HF_HOME")
    if hf_home_env:
        candidates.append(Path(hf_home_env) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return candidates


def _find_cached_whisper_model() -> Path | None:
    from voicecoding.transcriber import DEFAULT_MODEL_NAME, MODEL_REPO_ALIASES

    repo_id = MODEL_REPO_ALIASES.get(DEFAULT_MODEL_NAME, DEFAULT_MODEL_NAME)
    prefix = "models--" + repo_id.replace("/", "--")
    for root in _hf_cache_candidates():
        if not root.is_dir():
            continue
        try:
            for entry in root.iterdir():
                if entry.name.startswith(prefix):
                    return entry
        except OSError:
            continue
    return None


def autostart_run_bat_path() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "voicecoding" / "run.bat"


def autostart_shortcut_path() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return (
        Path(appdata)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / "voicecoding.lnk"
    )


def _check_autostart() -> DoctorCheck:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return DoctorCheck(
            name="autostart",
            status="warn",
            detail="APPDATA is not set; cannot verify autostart",
        )

    run_bat = autostart_run_bat_path()
    shortcut = autostart_shortcut_path()
    missing = [str(path) for path in (run_bat, shortcut) if not path.exists()]
    if not missing:
        return DoctorCheck(
            name="autostart",
            status="ok",
            detail=f"autostart installed ({run_bat.name}, {shortcut.name})",
        )
    return DoctorCheck(
        name="autostart",
        status="warn",
        detail="autostart incomplete; missing: "
        + ", ".join(missing)
        + "; run `voicecoding deploy` to install",
    )


def _check_audio_input(config: AppConfig | None) -> DoctorCheck:
    try:
        sounddevice = importlib.import_module("sounddevice")
    except Exception as error:
        return DoctorCheck(
            name="audio-input",
            status="fail",
            detail=f"cannot import sounddevice: {error}",
        )

    query_devices = getattr(sounddevice, "query_devices", None)
    if not callable(query_devices):
        return DoctorCheck(
            name="audio-input",
            status="fail",
            detail="sounddevice.query_devices is unavailable",
        )

    try:
        devices_obj = query_devices()
    except Exception as error:
        return DoctorCheck(
            name="audio-input",
            status="fail",
            detail=f"failed to query audio devices: {error}",
        )

    device_entries = _coerce_device_entries(devices_obj)
    if device_entries is None:
        return DoctorCheck(
            name="audio-input",
            status="warn",
            detail="unexpected audio device payload shape",
        )

    input_devices: list[Mapping[str, object]] = []
    for item in device_entries:
        max_inputs = _to_float(item.get("max_input_channels", 0.0))
        if max_inputs > 0:
            input_devices.append(item)
    if not input_devices:
        return DoctorCheck(
            name="audio-input",
            status="fail",
            detail="no input-capable microphone device detected",
        )

    default_index = _read_default_input_device_index(sounddevice)
    check_input_settings = getattr(sounddevice, "check_input_settings", None)
    if default_index is not None and callable(check_input_settings):
        sample_rate = float(config.sample_rate) if config is not None else 16000.0
        channels = config.channels if config is not None else 1
        try:
            _ = check_input_settings(
                device=default_index,
                channels=max(1, int(channels)),
                samplerate=sample_rate,
            )
        except Exception as error:
            return DoctorCheck(
                name="audio-input",
                status="warn",
                detail=f"default input exists but validation failed: {error}",
            )

    return DoctorCheck(
        name="audio-input",
        status="ok",
        detail=f"detected {len(input_devices)} input-capable device(s)",
    )


def _read_default_input_device_index(sounddevice: object) -> int | None:
    default = getattr(sounddevice, "default", None)
    device = getattr(default, "device", None)
    if isinstance(device, list | tuple) and device:
        candidate = device[0]
        if isinstance(candidate, int):
            return candidate
    return None


def _coerce_device_entries(devices_obj: object) -> list[Mapping[str, object]] | None:
    if isinstance(devices_obj, list | tuple):
        entries: list[Mapping[str, object]] = []
        for item in devices_obj:
            if not isinstance(item, Mapping):
                return None
            entries.append(item)
        return entries
    return None


def _to_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _print_checks(checks: list[DoctorCheck]) -> None:
    for check in checks:
        badge = {
            "ok": "[OK]",
            "warn": "[WARN]",
            "fail": "[FAIL]",
        }.get(check.status, "[INFO]")
        print(f"{badge} {check.name}: {check.detail}")
