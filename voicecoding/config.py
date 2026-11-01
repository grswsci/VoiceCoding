from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _read_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a float, got {raw!r}") from error


def _read_button(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in {"x1", "x2"}:
        raise ValueError(f"{name} must be either 'x1' or 'x2', got {value!r}")
    return value


def _require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _require_non_negative(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _require_non_negative_float(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative float, got {value}")
    return value


def _read_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {options}; got {value!r}")
    return value


@dataclass(frozen=True)
class AppConfig:
    sample_rate: int
    channels: int
    dtype: str
    model_name: str
    log_level: str
    device: str
    language: str
    compute_type: str
    beam_size: int
    vad_enabled: bool
    fallback_to_cpu: bool
    button_debounce_ms: int
    gestures_enabled: bool
    gesture_trigger_button: str
    gesture_threshold_px: int
    gesture_freeze_pointer: bool
    gesture_restore_cursor: bool
    gesture_up_action: str
    gesture_down_action: str
    gesture_left_action: str
    gesture_right_action: str
    enter_mode: str
    auto_paste: bool
    prewarm_on_start: bool
    prewarm_delay_s: float
    status_file: Path
    front_button: str
    rear_button: str
    record_hotkey_keycodes: tuple[int, ...]
    recording_submit_keycode: int
    pre_buffer_s: float
    temp_dir: Path


def load_config() -> AppConfig:
    temp_dir = Path(
        os.getenv(
            "VOICECODING_TEMP_DIR", str(Path(tempfile.gettempdir()) / "voicecoding")
        )
    )
    status_file = Path(
        os.getenv(
            "VOICECODING_STATUS_FILE",
            os.path.join(tempfile.gettempdir(), "voicecoding-status.json"),
        )
    )

    sample_rate = _require_positive(
        "VOICECODING_SAMPLE_RATE", _read_int("VOICECODING_SAMPLE_RATE", 16000)
    )
    channels = _require_positive(
        "VOICECODING_CHANNELS", _read_int("VOICECODING_CHANNELS", 1)
    )
    front_button = _read_button("VOICECODING_FRONT_BUTTON", "x2")
    rear_button = _read_button("VOICECODING_REAR_BUTTON", "x1")
    record_hotkey_keycodes = tuple(
        dict.fromkeys(
            (
                _require_non_negative(
                    "VOICECODING_RECORD_HOTKEY_CODE_1",
                    _read_int("VOICECODING_RECORD_HOTKEY_CODE_1", 162),
                ),
                _require_non_negative(
                    "VOICECODING_RECORD_HOTKEY_CODE_2",
                    _read_int("VOICECODING_RECORD_HOTKEY_CODE_2", 160),
                ),
                _require_non_negative(
                    "VOICECODING_RECORD_HOTKEY_CODE_3",
                    _read_int("VOICECODING_RECORD_HOTKEY_CODE_3", 32),
                ),
            )
        )
    )
    if len(record_hotkey_keycodes) != 3:
        raise ValueError("VOICECODING_RECORD_HOTKEY_CODE_1/2/3 must be distinct")
    recording_submit_keycode = _read_optional_int("VOICECODING_RECORDING_SUBMIT_KEYCODE")
    if recording_submit_keycode is None:
        recording_submit_keycode = 13
    else:
        recording_submit_keycode = _require_non_negative(
            "VOICECODING_RECORDING_SUBMIT_KEYCODE",
            recording_submit_keycode,
        )
    if front_button == rear_button:
        raise ValueError("VOICECODING_FRONT_BUTTON and VOICECODING_REAR_BUTTON must differ")
    button_debounce_ms = _require_non_negative(
        "VOICECODING_BUTTON_DEBOUNCE_MS",
        _read_int("VOICECODING_BUTTON_DEBOUNCE_MS", 150),
    )
    gestures_enabled = _read_bool("VOICECODING_GESTURES_ENABLED", False)
    gesture_trigger_button = _read_choice(
        "VOICECODING_GESTURE_TRIGGER_BUTTON",
        "rear",
        {"front", "rear", "right"},
    )
    gesture_threshold_px = _require_positive(
        "VOICECODING_GESTURE_THRESHOLD_PX",
        _read_int("VOICECODING_GESTURE_THRESHOLD_PX", 120),
    )
    gesture_freeze_pointer = _read_bool("VOICECODING_GESTURE_FREEZE_POINTER", True)
    gesture_restore_cursor = _read_bool("VOICECODING_GESTURE_RESTORE_CURSOR", True)
    gesture_actions = {
        "record_toggle",
        "send_enter",
        "workspace_left",
        "workspace_right",
        "noop",
    }
    gesture_up_action = _read_choice(
        "VOICECODING_GESTURE_UP_ACTION",
        "record_toggle",
        gesture_actions,
    )
    gesture_down_action = _read_choice(
        "VOICECODING_GESTURE_DOWN_ACTION",
        "noop",
        gesture_actions,
    )
    gesture_left_action = _read_choice(
        "VOICECODING_GESTURE_LEFT_ACTION",
        "noop",
        gesture_actions,
    )
    gesture_right_action = _read_choice(
        "VOICECODING_GESTURE_RIGHT_ACTION",
        "send_enter",
        gesture_actions,
    )
    enter_mode = _read_choice(
        "VOICECODING_ENTER_MODE",
        "enter",
        {"enter", "ctrl_enter", "shift_enter", "none"},
    )
    prewarm_delay_s = _require_non_negative_float(
        "VOICECODING_PREWARM_DELAY_S",
        _read_float("VOICECODING_PREWARM_DELAY_S", 0.0),
    )
    pre_buffer_s = _require_non_negative_float(
        "VOICECODING_PRE_BUFFER_S",
        _read_float("VOICECODING_PRE_BUFFER_S", 0.5),
    )
    beam_size = _require_positive(
        "VOICECODING_BEAM_SIZE", _read_int("VOICECODING_BEAM_SIZE", 5)
    )

    return AppConfig(
        sample_rate=sample_rate,
        channels=channels,
        dtype=os.getenv("VOICECODING_DTYPE", "float32").strip(),
        model_name=os.getenv("VOICECODING_MODEL", "large-v3-turbo").strip(),
        log_level=_read_choice(
            "VOICECODING_LOG_LEVEL",
            "info",
            {"debug", "info", "warning", "error", "critical"},
        ).upper(),
        device=os.getenv("VOICECODING_DEVICE", "cuda").strip().lower(),
        language=os.getenv("VOICECODING_LANGUAGE", "auto").strip().lower(),
        compute_type=os.getenv("VOICECODING_COMPUTE_TYPE", "float16").strip().lower(),
        beam_size=beam_size,
        vad_enabled=_read_bool("VOICECODING_VAD_ENABLED", True),
        fallback_to_cpu=_read_bool("VOICECODING_FALLBACK_CPU", True),
        button_debounce_ms=button_debounce_ms,
        gestures_enabled=gestures_enabled,
        gesture_trigger_button=gesture_trigger_button,
        gesture_threshold_px=gesture_threshold_px,
        gesture_freeze_pointer=gesture_freeze_pointer,
        gesture_restore_cursor=gesture_restore_cursor,
        gesture_up_action=gesture_up_action,
        gesture_down_action=gesture_down_action,
        gesture_left_action=gesture_left_action,
        gesture_right_action=gesture_right_action,
        enter_mode=enter_mode,
        auto_paste=_read_bool("VOICECODING_AUTO_PASTE", True),
        prewarm_on_start=_read_bool("VOICECODING_PREWARM_ON_START", True),
        prewarm_delay_s=prewarm_delay_s,
        pre_buffer_s=pre_buffer_s,
        status_file=status_file,
        front_button=front_button,
        rear_button=rear_button,
        record_hotkey_keycodes=record_hotkey_keycodes,
        recording_submit_keycode=recording_submit_keycode,
        temp_dir=temp_dir,
    )
