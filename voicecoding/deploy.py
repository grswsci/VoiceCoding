"""Windows deployment: launcher batch file and Startup shortcut creation.

The deploy flow writes ``%APPDATA%\\voicecoding\\run.bat`` (Python executable
captured at deploy time), renders a VBS shortcut script, and invokes it once
via ``cscript`` to create the Startup .lnk. ``schtasks`` exists only as a
fallback registration mechanism; its command string is constructed but never
executed off-Windows or in dry-run mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_LOG = logging.getLogger(__name__)

_PRESET_OVERRIDES: dict[str, dict[str, str]] = {
    "stable": {
        "VOICECODING_AUTO_PASTE": "true",
        "VOICECODING_BUTTON_DEBOUNCE_MS": "220",
        "VOICECODING_PREWARM_ON_START": "true",
    },
    "fast": {
        "VOICECODING_AUTO_PASTE": "true",
        "VOICECODING_BUTTON_DEBOUNCE_MS": "120",
        "VOICECODING_PREWARM_ON_START": "true",
    },
    "low-resource": {
        "VOICECODING_AUTO_PASTE": "false",
        "VOICECODING_BUTTON_DEBOUNCE_MS": "250",
        "VOICECODING_PREWARM_ON_START": "false",
    },
}

_AUTOSTART_CHOICES = ("startup", "schtasks", "none")
_TASK_NAME = "VoiceCoding"


def _appdata_base() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base)
    # Portable fallback for environments where APPDATA is unset.
    return Path.home() / "AppData" / "Roaming"


def _appdata_dir() -> Path:
    """Config and launcher directory: %APPDATA%\\voicecoding."""
    return _appdata_base() / "voicecoding"


def _startup_dir() -> Path:
    """Per-user Startup folder that receives the autostart shortcut."""
    return (
        _appdata_base()
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def render_run_bat(python_exe: str, env_overrides: dict[str, str]) -> str:
    """Render the launcher batch file as a pure string (no filesystem access)."""
    lines = ["@echo off", "rem VoiceCoding launcher"]
    for key in sorted(env_overrides):
        lines.append(f'set "{key}={env_overrides[key]}"')
    lines.append(f'"{python_exe}" -m voicecoding.main run')
    return "\r\n".join(lines) + "\r\n"


def render_startup_vbs(target_bat: str, lnk_path: str) -> str:
    """Render the shortcut-creation script as a pure string (no filesystem access)."""
    working_dir = Path(target_bat).parent
    lines = [
        'Set shell = CreateObject("WScript.Shell")',
        f'Set shortcut = shell.CreateShortcut("{lnk_path}")',
        f'shortcut.TargetPath = "{target_bat}"',
        f'shortcut.WorkingDirectory = "{working_dir}"',
        'shortcut.Description = "VoiceCoding"',
        "shortcut.Save",
    ]
    return "\r\n".join(lines) + "\r\n"


def build_schtasks_command(run_bat: Path) -> str:
    """Construct (never execute here) the schtasks fallback registration command."""
    return f'schtasks /Create /TN {_TASK_NAME} /SC ONLOGON /TR "\\"{run_bat}\\""'


def configure_deploy_parser(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--preset",
        choices=sorted(_PRESET_OVERRIDES.keys()),
        default="stable",
        help="deployment preset profile",
    )
    _ = parser.add_argument(
        "--autostart",
        choices=list(_AUTOSTART_CHOICES),
        default="startup",
        help="autostart mechanism (Startup shortcut preferred; schtasks is the backup)",
    )
    _ = parser.add_argument(
        "--skip-autostart",
        action="store_true",
        help="write the launcher without creating any autostart entry",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned actions and rendered files without writing anything",
    )
    _ = parser.add_argument(
        "--install-dir",
        default=None,
        help="override the launcher directory (default: %APPDATA%\\voicecoding)",
    )


def run_deploy(args: argparse.Namespace) -> int:
    preset: str = args.preset
    if preset not in _PRESET_OVERRIDES:
        _LOG.error("Unknown preset: %s", preset)
        return 2

    autostart: str = "none" if args.skip_autostart else str(args.autostart)
    install_dir = Path(args.install_dir) if args.install_dir else _appdata_dir()
    run_bat = install_dir / "run.bat"
    vbs_path = install_dir / "create_startup.vbs"
    lnk_path = _startup_dir() / "voicecoding.lnk"
    # Capture the interpreter at deploy time so the launcher keeps working
    # after the console environment that created it is gone.
    python_exe = sys.executable

    print(f"Preset: {preset}")
    print(f"Launcher: {run_bat}")
    print(f"Python: {python_exe}")
    print(f"Autostart: {autostart}")

    if args.dry_run:
        print("--- run.bat ---")
        print(render_run_bat(python_exe, _PRESET_OVERRIDES[preset]), end="")
        if autostart == "startup":
            print("--- create_startup.vbs ---")
            print(render_startup_vbs(str(run_bat), str(lnk_path)), end="")
        elif autostart == "schtasks":
            print("--- schtasks command ---")
            print(build_schtasks_command(run_bat))
        return 0

    install_dir.mkdir(parents=True, exist_ok=True)
    run_bat.write_text(
        render_run_bat(python_exe, _PRESET_OVERRIDES[preset]),
        encoding="utf-8",
    )
    _LOG.info("Wrote launcher: %s", run_bat)

    if autostart == "startup":
        vbs_path.write_text(
            render_startup_vbs(str(run_bat), str(lnk_path)),
            encoding="utf-8",
        )
        if sys.platform == "win32":
            result = subprocess.run(
                ["cscript", "//nologo", str(vbs_path)],
                check=False,
            )
            if result.returncode != 0:
                _LOG.error(
                    "Startup shortcut creation failed: cscript exited %s",
                    result.returncode,
                )
                return 1
            _LOG.info("Startup shortcut created: %s", lnk_path)
        else:
            _LOG.warning(
                "Not on Windows; VBS written but not executed: %s", vbs_path
            )
    elif autostart == "schtasks":
        command = build_schtasks_command(run_bat)
        if sys.platform == "win32":
            result = subprocess.run(command, shell=True, check=False)
            if result.returncode != 0:
                _LOG.error(
                    "Scheduled task creation failed: schtasks exited %s",
                    result.returncode,
                )
                return 1
            _LOG.info("Scheduled task created: %s", _TASK_NAME)
        else:
            _LOG.warning(
                "Not on Windows; schtasks command not executed: %s", command
            )
    else:
        _LOG.info("Autostart disabled; only the launcher was written")

    return 0
