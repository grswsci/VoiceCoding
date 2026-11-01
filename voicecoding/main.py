from __future__ import annotations

import argparse

from voicecoding.config import load_config
from voicecoding.doctor import run_doctor
from voicecoding.logging_setup import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicecoding")
    subparsers = parser.add_subparsers(dest="command")
    _ = subparsers.add_parser("run", help="run the voice-input daemon")
    doctor_parser = subparsers.add_parser("doctor", help="run environment diagnostics")
    _ = doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="apply safe auto-remediations before running checks",
    )
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="generate run scripts and install autostart entries",
    )
    from voicecoding.deploy import configure_deploy_parser

    configure_deploy_parser(deploy_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw_command = getattr(args, "command", None)
    command = raw_command if isinstance(raw_command, str) else "run"
    if command == "doctor":
        apply_fixes = bool(getattr(args, "fix", False))
        return run_doctor(apply_fixes=apply_fixes)
    if command == "deploy":
        from voicecoding.deploy import run_deploy

        return run_deploy(args)

    from voicecoding.app import VoiceMouseApp

    config = load_config()
    configure_logging(config.log_level)
    app = VoiceMouseApp(config)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
