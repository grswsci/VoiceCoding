"""Contract tests for voicecoding.doctor.

The doctor surface is the public run_doctor() entry point plus the
module-level check helpers it composes (_check_transcriber /
_check_autostart, each returning a DoctorCheck dataclass). Groups: the
transcriber check against the fake ctranslate2 CUDA count (0 -> warn, >= 1
with a cached model -> ok, missing cache -> warn), the autostart check keyed
off APPDATA and the Startup-folder artifacts, and the public entry point
returning 0 unless a check hard-fails.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

MODEL_CACHE_DIR = "models--deepdml--faster-whisper-large-v3-turbo-ct2"


def _fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(
        pathlib.Path, "home", classmethod(lambda cls: home)
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HF_HOME", str(home / ".cache" / "huggingface"))
    return home


def _install_hf_model_cache(home: pathlib.Path) -> None:
    snapshot = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / MODEL_CACHE_DIR
        / "snapshots"
    )
    snapshot.mkdir(parents=True)


def _transcriber_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    *,
    cuda_count: int,
    with_cache: bool,
):
    import voicecoding.doctor as doctor_mod

    sys.modules["ctranslate2"].STATE.cuda_count = cuda_count
    home = _fake_home(monkeypatch, tmp_path)
    if with_cache:
        _install_hf_model_cache(home)
    return doctor_mod._check_transcriber()


def _autostart_check():
    import voicecoding.doctor as doctor_mod

    return doctor_mod._check_autostart()


# ---------------------------------------------------------------------------
# _check_transcriber
# ---------------------------------------------------------------------------


def test_transcriber_check_warns_when_no_cuda_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    check = _transcriber_check(
        monkeypatch, tmp_path, cuda_count=0, with_cache=True
    )

    assert check.status == "warn"


def test_transcriber_check_passes_with_cuda_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    check = _transcriber_check(
        monkeypatch, tmp_path, cuda_count=2, with_cache=True
    )

    assert check.status == "ok"


def test_transcriber_check_warns_when_model_cache_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    check = _transcriber_check(
        monkeypatch, tmp_path, cuda_count=2, with_cache=False
    )

    assert check.status == "warn"
    assert "cache" in check.detail.lower()


# ---------------------------------------------------------------------------
# _check_autostart
# ---------------------------------------------------------------------------


def test_autostart_check_warns_when_entry_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))

    check = _autostart_check()

    assert check.status == "warn"
    assert "missing" in check.detail.lower()


def test_autostart_check_passes_when_shortcut_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # The check requires both autostart artifacts: launcher and shortcut.
    (tmp_path / "voicecoding").mkdir()
    (tmp_path / "voicecoding" / "run.bat").write_text("", encoding="utf-8")
    startup = (
        tmp_path
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    startup.mkdir(parents=True)
    (startup / "voicecoding.lnk").write_bytes(b"")

    check = _autostart_check()

    assert check.status == "ok"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def test_run_doctor_reports_checks_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from conftest import purge_voicecoding_env

    import voicecoding.doctor as doctor_mod

    purge_voicecoding_env(monkeypatch)

    result = doctor_mod.run_doctor()

    out = capsys.readouterr().out
    assert result == 0  # warns never fail the doctor; only hard failures do
    assert "Doctor summary" in out
