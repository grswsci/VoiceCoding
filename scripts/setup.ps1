# VoiceCoding Windows setup helper.
# Creates .venv, installs VoiceCoding with GPU dependencies, and prints next steps.
# Compatible with Windows PowerShell 5.1+; no admin rights required.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

Write-Host "=== VoiceCoding setup ==="
Write-Host "Repo root: $repoRoot"

# 1) Virtual environment
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    Write-Host "[1/3] .venv already exists, reusing it."
} else {
    Write-Host "[1/3] Creating virtual environment .venv ..."
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "python -m venv failed. Make sure Python 3.10+ is on PATH."
    }
}

# 2) Install VoiceCoding with the GPU extra (nvidia-cublas-cu12 + nvidia-cudnn-cu12)
Write-Host "[2/3] Installing VoiceCoding (GPU build) ..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed. Check the output above."
}
& $venvPython -m pip install -e ".[gpu]"
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed. Check the output above."
}

# 3) Model download hint + next steps
Write-Host ""
Write-Host "[3/3] Install complete."
Write-Host ""
Write-Host "First run downloads the large-v3-turbo model (~1.6 GB) from Hugging Face."
Write-Host "Slow network or mainland China? Set the mirror BEFORE the first run:"
Write-Host '  cmd:        set HF_ENDPOINT=https://hf-mirror.com'
Write-Host '  PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"'
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate the environment:  .\.venv\Scripts\Activate.ps1"
Write-Host "     (if activation is blocked: Set-ExecutionPolicy -Scope Process Bypass)"
Write-Host "  2. Self-check:                voicecoding doctor"
Write-Host "  3. Start the listener:        voicecoding run"
Write-Host "     wait for:                  VoiceCoding ready."
Write-Host "  4. Enable autostart:          voicecoding deploy"
Write-Host ""
Write-Host "Note: whichever Python environment you run 'voicecoding deploy' in becomes"
Write-Host "the interpreter autostart will use."
