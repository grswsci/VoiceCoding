# Third-Party Notices

This document summarizes third-party components used by VoiceCoding and their
declared licenses.

Last reviewed: 2026-11-01.

## 1) Project License

VoiceCoding source code is licensed under Apache-2.0, copyright
"Copyright 2026 VoiceCoding Contributors". See `LICENSE`.

## 2) Direct Python Dependencies

The following are direct runtime dependencies declared in `pyproject.toml`.

| Package | Declared License | Upstream |
|---|---|---|
| `numpy` | BSD-3-Clause | https://numpy.org |
| `sounddevice` | MIT | https://python-sounddevice.readthedocs.io/ |
| `soundfile` | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| `pynput` | LGPL-3.0 (or later) | https://github.com/moses-palmer/pynput |
| `pyperclip` | BSD | https://github.com/asweigart/pyperclip |
| `faster-whisper` | MIT | https://github.com/SYSTRAN/faster-whisper |
| `nvidia-cublas-cu12` | NVIDIA software license | https://pypi.org/project/nvidia-cublas-cu12/ |
| `nvidia-cudnn-cu12` | NVIDIA software license | https://pypi.org/project/nvidia-cudnn-cu12/ |

Notes:

- `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` form the `[gpu]` extra. They
  provide the NVIDIA CUDA 12 / cuDNN 9 runtime libraries used for GPU
  inference and are subject to NVIDIA's license terms shipped inside each
  wheel.
- `sounddevice` binds to PortAudio, and `soundfile` binds to libsndfile; both
  native libraries carry their own licenses and remain subject to them.
- `pynput` is LGPL-licensed. If you redistribute packaged binaries, ensure
  LGPL obligations are satisfied (license notice, relinking conditions where
  applicable, and source availability requirements for the LGPL-covered
  components).
- Transitive dependencies are not exhaustively listed here. They remain
  subject to their own licenses.

## 3) Speech Backend Components

These components are pulled in through `faster-whisper` and used at runtime.

| Component | Declared License | Upstream | Used for |
|---|---|---|---|
| `CTranslate2` | MIT | https://github.com/OpenNMT/CTranslate2 | Inference engine that executes the Whisper model |
| Silero VAD | MIT | https://github.com/snakers4/silero-vad | Voice activity detection models/code bundled with faster-whisper |

## 4) Model Assets and Weights

VoiceCoding defaults to the model ID:

- `deepdml/faster-whisper-large-v3-turbo-ct2` — the Whisper large-v3-turbo
  weights converted to CTranslate2 (faster-whisper) format.

At review time, the model card declares the MIT license. The original Whisper
models by OpenAI are likewise MIT-licensed.

References:

- https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
- https://github.com/openai/whisper

Important caveats:

- The default weights auto-download (~1.6 GB) from Hugging Face on first run.
  When a mirror endpoint is configured via `HF_ENDPOINT`, re-verify that the
  mirror serves unmodified artifacts before any redistribution.
- If you switch to other model IDs, mirror-hosted weights, or bundled model
  artifacts, re-verify the exact model license/terms before redistribution.
- Do not assume all speech models under the same ecosystem share identical
  license terms.

## 5) Attribution and Compliance Guidance

When distributing VoiceCoding (source or binaries):

1. Keep `LICENSE` and this notice file.
2. Preserve upstream copyright and license notices for bundled components.
3. Re-check model card licenses when changing model IDs or revisions.
4. Do not assume all speech models under the same ecosystem share identical
   license terms.
