# VoiceCoding

Push-to-talk voice input for Windows, powered by a local faster-whisper model on your NVIDIA GPU.

中文文档：[`README.zh-CN.md`](./README.zh-CN.md)

Detailed setup guide: [`docs/WINDOWS_SETUP.md`](./docs/WINDOWS_SETUP.md)

## What This Project Does

VoiceCoding turns speech into text on Windows, fully offline:

- Mouse front side button: start/stop recording
- Mouse rear side button while recording: stop recording and type the transcribed text locally at the cursor (clipboard + Ctrl+V when auto-paste is enabled)
- Keyboard hotkey (default Ctrl+Shift+Space): start/stop recording without touching the mouse
- Optional gestures: hold a trigger button and move the pointer past a threshold to fire direction actions (record, Enter, virtual desktop switching)
- Transcription runs locally with faster-whisper large-v3-turbo (float16, ≈2 GB VRAM); nothing leaves your machine

Core goals are low friction, stable daily use, pure local text output, and graceful fallback when any subsystem fails.

## Requirements

- Windows 10/11 x64
- Python >= 3.10
- NVIDIA GPU with a driver supporting CUDA 12 (validated on an 8 GB-class laptop GPU; the float16 model footprint is ≈2 GB VRAM)
- A microphone
- No NVIDIA GPU? CPU-only mode works too — see the CPU install alternative below

## Install

GPU mode (recommended):

```powershell
cd VoiceCoding
pip install -U pip
pip install -e ".[gpu]"
```

The `gpu` extra installs `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`, which are required for CUDA inference.

CPU-only alternative:

```powershell
pip install -e .
set VOICECODING_DEVICE=cpu
```

Shortcut: `scripts/setup.ps1` creates a `.venv`, installs the GPU build, and prints next steps:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

### Model download (first run)

The large-v3-turbo model (CTranslate2 repo `deepdml/faster-whisper-large-v3-turbo-ct2`, ~1.6 GB) is downloaded automatically from Hugging Face on first start.

Slow network or mainland China? Set the mirror before the first run:

```bat
:: cmd
set HF_ENDPOINT=https://hf-mirror.com
```

```powershell
# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## Run

```powershell
voicecoding run
```

When you see `VoiceCoding ready.`, the listener is live. Press Ctrl+C to stop.

## Default Mapping and State Logic

- `VOICECODING_FRONT_BUTTON` default: `x2`
- `VOICECODING_REAR_BUTTON` default: `x1`

| Input | State | Action |
|---|---|---|
| Front side button | idle | start recording |
| Front side button | recording | stop recording |
| Rear side button | recording | stop recording and type the transcript at the cursor (clipboard + Ctrl+V when auto-paste is enabled) |
| Rear side button | idle | send Enter (`VOICECODING_ENTER_MODE`) |
| Record hotkey (default Ctrl+Shift+Space) | any | start/stop recording |
| Enter (default VK_RETURN=13, see `VOICECODING_RECORDING_SUBMIT_KEYCODE`) | recording | stop and submit the transcript |

If your hardware labels are reversed:

```bat
set VOICECODING_FRONT_BUTTON=x1
set VOICECODING_REAR_BUTTON=x2
```

### Gestures (optional)

Enable with `VOICECODING_GESTURES_ENABLED=true`. Hold the configured trigger button (default: right mouse button) and move the pointer past `VOICECODING_GESTURE_THRESHOLD_PX` to fire a direction action (`VOICECODING_GESTURE_UP/DOWN/LEFT/RIGHT_ACTION`: `record_toggle`, `send_enter`, `workspace_left`, `workspace_right`, `noop`). `workspace_left` / `workspace_right` switch the Windows virtual desktop (Win+Ctrl+Left/Right). The pointer is pinned during the gesture when `VOICECODING_GESTURE_FREEZE_POINTER` is enabled, and returned to its original position when `VOICECODING_GESTURE_RESTORE_CURSOR` is enabled.

## Frequently Used Variables

| Variable | Default | Purpose |
|---|---|---|
| `VOICECODING_MODEL_NAME` | `deepdml/faster-whisper-large-v3-turbo-ct2` | faster-whisper (CTranslate2) model repo; the default is the large-v3-turbo conversion |
| `VOICECODING_DEVICE` | `cuda` | Inference device (`cuda` / `cpu`); `cpu` is the escape hatch when CUDA is unavailable or VRAM is tight |
| `VOICECODING_COMPUTE_TYPE` | `float16` | Model compute type; `int8_float16` roughly halves VRAM usage |
| `VOICECODING_LANGUAGE` | `auto` | Spoken language; `auto` lets the model detect it |
| `VOICECODING_FALLBACK_TO_CPU` | `true` | Retry on CPU when CUDA initialization fails |
| `VOICECODING_BEAM_SIZE` | `5` | Beam-search width for decoding (`1` = greedy) |
| `VOICECODING_VAD_ENABLED` | `true` | Skip silence via voice activity detection |
| `VOICECODING_SAMPLE_RATE` | `16000` | Microphone capture sample rate (Hz) |
| `VOICECODING_CHANNELS` | `1` | Capture channel count |
| `VOICECODING_DTYPE` | `float32` | Capture sample format |
| `VOICECODING_ENTER_MODE` | `enter` | Rear-button submit mode while idle (`enter`, `ctrl_enter`, `shift_enter`, `none`) |
| `VOICECODING_AUTO_PASTE` | `true` | Deliver transcripts via clipboard + Ctrl+V instead of synthetic typing |
| `VOICECODING_FRONT_BUTTON` | `x2` | Button bound to record start/stop (`x1` / `x2`) |
| `VOICECODING_REAR_BUTTON` | `x1` | Button bound to stop-and-type / idle Enter (`x1` / `x2`) |
| `VOICECODING_GESTURES_ENABLED` | `false` | Enable hold-and-move gestures |
| `VOICECODING_GESTURE_TRIGGER_BUTTON` | `right` | Gesture trigger button (`front`, `rear`, `right`) |
| `VOICECODING_GESTURE_THRESHOLD_PX` | `120` | Pointer travel (px) required to fire a direction action |
| `VOICECODING_GESTURE_FREEZE_POINTER` | `true` | Pin the pointer during gesture capture |
| `VOICECODING_GESTURE_RESTORE_CURSOR` | `true` | Move the cursor back to its original position after the gesture |
| `VOICECODING_GESTURE_UP_ACTION` | `record_toggle` | Direction action (see gesture section for allowed values) |
| `VOICECODING_GESTURE_DOWN_ACTION` | `noop` | Direction action |
| `VOICECODING_GESTURE_LEFT_ACTION` | `noop` | Direction action |
| `VOICECODING_GESTURE_RIGHT_ACTION` | `send_enter` | Direction action |
| `VOICECODING_RECORD_HOTKEY_KEYCODES` | `162,160,32` | Comma-separated Windows VK codes of the record hotkey; default = LeftCtrl+LeftShift+Space (VK_LCONTROL=162, VK_LSHIFT=160, VK_SPACE=32) |
| `VOICECODING_RECORDING_SUBMIT_KEYCODE` | `13` | VK code that stops recording and submits the transcript while recording (default VK_RETURN); leave unset to disable |
| `VOICECODING_PREWARM_ON_START` | `true` | Load the model at startup to cut first-use latency |
| `VOICECODING_PREWARM_DELAY_S` | `0.0` | Seconds to wait after startup before prewarming |
| `VOICECODING_PRE_BUFFER_S` | `0.5` | Seconds of rolling audio kept before the record press, so the first word is not clipped |
| `VOICECODING_BUTTON_DEBOUNCE_MS` | `150` | Side-button debounce window (ms) |
| `VOICECODING_STATUS_FILE` | `<temp>/voicecoding-status.json` | Runtime status JSON for widgets/scripts |
| `VOICECODING_TEMP_DIR` | `<temp>/voicecoding` | Directory for temporary recordings |

Notes on value formats:

- Booleans accept `true` / `1` / `yes` / `on` (case-insensitive); any other value counts as false.
- Numeric variables take plain integers/decimals as strings (e.g. `120`, `0.5`).
- All settings are environment variables with the `VOICECODING_` prefix.

Full configuration source of truth: `voicecoding/config.py`.

## Built-in Doctor

Run diagnostics:

```powershell
voicecoding doctor
```

Checks include:

- Config load validity
- Microphone input availability
- Transcriber backend health (`faster_whisper` / `ctranslate2` importable, CUDA device count, model cache presence)
- Autostart entry state

Rebuild the autostart entry:

```powershell
voicecoding doctor --fix
```

## Deploy and Autostart

> **Important:** whichever Python environment you run `voicecoding deploy` in is the interpreter autostart will use. Activate the environment you actually want before deploying.

`voicecoding deploy` prepares autostart:

- Writes `%APPDATA%\voicecoding\run.bat` (your env vars + the absolute `python.exe` path of the environment deploy ran in)
- Writes `%APPDATA%\voicecoding\create_startup.vbs`
- Runs `cscript //nologo` once to create the Startup shortcut `voicecoding.lnk`

Useful flags:

- `--preset <name>` — apply a named defaults profile
- `--dry-run` — show what would be written without changing anything
- `--skip-autostart` — write `run.bat` but skip the Startup shortcut
- `--autostart startup|schtasks|none` — choose the autostart mechanism; `schtasks` (ONLOGON task) is the backup when the Startup-folder shortcut is unavailable, `none` skips

To remove autostart later, delete `%APPDATA%\voicecoding` and the `voicecoding.lnk` shortcut in your Startup folder (`shell:startup`).

## Troubleshooting Shortlist

### Side buttons act like browser back/forward

If the low-level mouse hook cannot be installed, the side buttons keep their OS default behavior (browser back/forward). VoiceCoding logs a warning and stays usable via the keyboard hotkey. Run `voicecoding doctor` to check the environment, then restart.

### Text does not reach elevated (run-as-admin) windows

Windows UIPI blocks synthetic keys (Enter, Ctrl+V, Win+Ctrl+arrows) from reaching elevated windows. A non-admin VoiceCoding cannot type into a run-as-admin window; this is an OS security boundary, not a bug. Work inside a normal-privilege window, or accept that elevated targets must be filled manually.

### GPU out of memory

Browsers and games can consume most of the VRAM. If you hit CUDA OOM, either switch to CPU:

```bat
set VOICECODING_DEVICE=cpu
```

or shrink the GPU footprint:

```bat
set VOICECODING_COMPUTE_TYPE=int8_float16
```

### First model download is slow

The first start downloads ~1.6 GB from Hugging Face. On slow networks or in mainland China, set `HF_ENDPOINT=https://hf-mirror.com` before the first run (see Install).

## License

Source code is licensed under Apache-2.0, copyright "Copyright 2026 VoiceCoding Contributors". See `LICENSE`.

Project derivation and third-party/model asset notices: [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
