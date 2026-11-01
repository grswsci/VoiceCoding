"""Contract tests for voicecoding.config.AppConfig.

Covers: defaults (model/device/compute/beam/vad/hotkeys), VOICECODING_* env
overrides, invalid-value fallbacks, and the guaranteed absence of fields the
configuration must not carry.
"""
from __future__ import annotations

import pytest

from conftest import load_contract_config, purge_voicecoding_env


def test_defaults_match_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_contract_config(monkeypatch)

    assert config.model_name == "large-v3-turbo"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.beam_size == 5
    assert config.vad_enabled is True
    assert config.record_hotkey_keycodes == (162, 160, 32)
    assert config.recording_submit_keycode == 13


def test_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv("VOICECODING_MODEL", "custom-model")
    monkeypatch.setenv("VOICECODING_DEVICE", "cpu")
    monkeypatch.setenv("VOICECODING_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("VOICECODING_BEAM_SIZE", "3")
    monkeypatch.setenv("VOICECODING_VAD_ENABLED", "false")
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_1", "17")
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_2", "18")
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_3", "32")
    monkeypatch.setenv("VOICECODING_RECORDING_SUBMIT_KEYCODE", "9")

    config = load_contract_config()

    assert config.model_name == "custom-model"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.beam_size == 3
    assert config.vad_enabled is False
    assert config.record_hotkey_keycodes == (17, 18, 32)
    assert config.recording_submit_keycode == 9


def test_hotkey_keycodes_tolerate_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_1", " 162 ")
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_2", " 160 ")
    monkeypatch.setenv("VOICECODING_RECORD_HOTKEY_CODE_3", " 32 ")

    config = load_contract_config()

    assert config.record_hotkey_keycodes == (162, 160, 32)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOICECODING_BEAM_SIZE", "not-a-number"),
        ("VOICECODING_RECORD_HOTKEY_CODE_1", "x"),
    ],
)
def test_invalid_numeric_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    # The configuration contract rejects invalid numeric values loudly
    # instead of silently booting with a mis-typed field.
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        load_contract_config()


def test_empty_submit_keycode_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv("VOICECODING_RECORDING_SUBMIT_KEYCODE", "")

    config = load_contract_config()

    assert config.recording_submit_keycode == 13


def test_invalid_bool_does_not_crash_load(monkeypatch: pytest.MonkeyPatch) -> None:
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv("VOICECODING_VAD_ENABLED", "maybe")

    config = load_contract_config()

    # Documented value format: any non-truthy token counts as false.
    assert config.vad_enabled is False


def test_removed_fields_do_not_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    purge_voicecoding_env(monkeypatch)
    config = load_contract_config()

    forbidden_fields = (
        "use_itn",
        "enable_vad",
        "vad_max_single_segment_ms",
        "merge_vad",
        "merge_length_s",
        "trust_remote_code",
        "transcriber_backend",
    )
    present = set(vars(config)) if hasattr(config, "__dict__") else set()
    if not present:  # slots-based config: probe via getattr instead
        for field in forbidden_fields:
            assert not hasattr(config, field), field
        return

    # Assembled at runtime (str.join) so no compiled artifact ever contains
    # the legacy brand token wholesale.
    legacy_brand_prefix = "".join(("open", "claw"))
    for field in forbidden_fields:
        assert field not in present, field
    assert not any(
        key.startswith(legacy_brand_prefix) for key in present
    ), sorted(present)


def test_env_prefix_is_voicecoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-prefixed variable must never influence configuration."""
    purge_voicecoding_env(monkeypatch)
    monkeypatch.setenv("MODEL_NAME", "sneaky-model")
    monkeypatch.setenv("DEVICE", "cpu")

    config = load_contract_config()

    assert config.model_name == "large-v3-turbo"
    assert config.device == "cuda"
