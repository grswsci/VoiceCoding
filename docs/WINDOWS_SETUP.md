# VoiceCoding — Windows Setup Guide

This guide walks through a full Windows install: NVIDIA driver, Python
environment, GPU dependencies, the model download, first run, diagnostics, and
autostart. For a quick overview see [`../README.md`](../README.md).

## 1. What You Need

- Windows 10/11 x64
- Python >= 3.10
- NVIDIA GPU with a driver supporting CUDA 12
  - Validated on an 8 GB-class laptop GPU; the float16 model footprint is
    ≈2 GB VRAM
  - No NVIDIA GPU? Follow the CPU-only alternative in step 4
- A working microphone
- ~4 GB free disk space (dependencies + ~1.6 GB model weights)

## 2. NVIDIA Driver

Install/upgrade the NVIDIA driver from https://www.nvidia.com/drivers and
pick the latest Game Ready or Studio driver for your GPU. Any recent driver
that supports CUDA 12 works — you do **not** need to install the CUDA Toolkit
itself; the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels bring the CUDA
runtime libraries.

Verify the driver sees the GPU:

```bat
nvidia-smi
```

## 3. Python Environment

Install Python 3.10+ from https://www.python.org/downloads/ (tick "Add
python.exe to PATH"), or use Miniconda/Anaconda.

### Option A: venv via the helper script

```powershell
cd VoiceCoding
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The script creates `.venv`, installs the GPU build, and prints next steps.

### Option B: manual venv

```bat
cd VoiceCoding
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[gpu]"
```

### Option C: conda

```bat
conda create -n voicecoding python=3.10
conda activate voicecoding
pip install -U pip
pip install -e ".[gpu]"
```

## 4. Install

GPU mode (recommended):

```bat
pip install -e ".[gpu]"
```

The `gpu` extra installs `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`, required
for CUDA inference.

CPU-only alternative (no NVIDIA GPU, or GPU intentionally unused):

```bat
pip install -e .
set VOICECODING_DEVICE=cpu
```

## 5. Model Download (First Run)

The large-v3-turbo model — CTranslate2 repo
`deepdml/faster-whisper-large-v3-turbo-ct2` (~1.6 GB) — is downloaded
automatically from Hugging Face the first time the app starts.

Slow network or mainland China? Set the mirror **before the first run**:

```bat
:: cmd (current session)
set HF_ENDPOINT=https://hf-mirror.com

:: cmd (persistent)
setx HF_ENDPOINT https://hf-mirror.com
```

```powershell
# PowerShell (current session)
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 6. First Check: Doctor

```bat
voicecoding doctor
```

Checks include:

- Config load validity
- Microphone input availability
- Transcriber backend health (`faster_whisper` / `ctranslate2` importable,
  CUDA device count, model cache presence)
- Autostart entry state

Rebuild the autostart entry:

```bat
voicecoding doctor --fix
```

## 7. Run

```bat
voicecoding run
```

- The first start downloads the model (~1.6 GB), then preloads it.
- When you see `VoiceCoding ready.`, the listener is live.
- Ctrl+C stops the app.

Controls recap (full table in the README):

- Mouse front side button: start/stop recording
- Mouse rear side button while recording: stop and type the transcript at the
  cursor; while idle: send Enter
- Ctrl+Shift+Space (default): start/stop recording
- Enter while recording: stop and submit the transcript

## 8. Autostart (Deploy)

> **Important:** whichever Python environment you run `voicecoding deploy` in
> is the interpreter autostart will use. Activate the environment you actually
> want before deploying — the generated `run.bat` bakes in its absolute
> `python.exe` path.

```bat
voicecoding deploy
```

What it does:

1. Writes `%APPDATA%\voicecoding\run.bat` — your env vars plus the absolute
   `python.exe` path of the environment deploy ran in.
2. Writes `%APPDATA%\voicecoding\create_startup.vbs`.
3. Runs `cscript //nologo` once to create the Startup shortcut
   `voicecoding.lnk` (`shell:startup`).

Useful flags:

- `--preset <name>` — apply a named defaults profile
- `--dry-run` — show what would be written without changing anything
- `--skip-autostart` — write `run.bat` but skip the Startup shortcut
- `--autostart startup|schtasks|none` — choose the autostart mechanism;
  `schtasks` (ONLOGON task) is the backup when the Startup-folder shortcut is
  unavailable, `none` skips autostart entirely

To remove autostart later: delete `%APPDATA%\voicecoding` and the
`voicecoding.lnk` shortcut in your Startup folder (`shell:startup`).

## 9. GPU Memory Notes

- Default `float16` needs ≈2 GB VRAM on top of whatever your desktop and
  browser already reserve.
- Browsers with hardware acceleration, games, or CUDA-based tools can squeeze
  out the remaining VRAM.
- If you hit CUDA OOM:
  - shrink the footprint: `set VOICECODING_COMPUTE_TYPE=int8_float16`, or
  - leave the GPU entirely: `set VOICECODING_DEVICE=cpu`.
- `VOICECODING_FALLBACK_TO_CPU=true` (default) retries on CPU when CUDA
  initialization fails, so a busy GPU degrades speed, not functionality.

## 10. Troubleshooting

### Side buttons act like browser back/forward

If the low-level mouse hook cannot be installed, the side buttons keep their
OS default behavior (browser back/forward). VoiceCoding logs a warning and
stays usable via the keyboard hotkey. Run `voicecoding doctor`, fix what it
reports, then restart.

### Text does not reach elevated (run-as-admin) windows

Windows UIPI blocks synthetic keys (Enter, Ctrl+V, Win+Ctrl+arrows) from
reaching elevated windows. A non-admin VoiceCoding cannot type into a
run-as-admin window — this is an OS security boundary, not a bug. Work in
normal-privilege windows, or fill elevated targets manually.

### GPU out of memory

See GPU Memory Notes above: `VOICECODING_DEVICE=cpu` or
`VOICECODING_COMPUTE_TYPE=int8_float16`.

### First model download is slow

Set `HF_ENDPOINT=https://hf-mirror.com` before the first run (step 5).

### Autostart starts the wrong environment

The Startup shortcut launches the interpreter recorded at deploy time. Re-run
`voicecoding deploy` from the environment you want, or rebuild the entry with
`voicecoding doctor --fix`.

## 11. Uninstall

1. Deactivate the app: `Ctrl+C` in its console, or just log off.
2. Remove autostart: delete `%APPDATA%\voicecoding` and `voicecoding.lnk` in
   `shell:startup` (or the schtasks entry if you used `--autostart schtasks`).
3. Remove the environment: delete `.venv` (or `conda env remove -n
   voicecoding`), then delete the repository folder.
