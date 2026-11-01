# VoiceCoding

面向 Windows 的按键发音（push-to-talk）语音输入工具，由你 NVIDIA GPU 上的本地 faster-whisper 模型驱动。

English README: [`README.md`](./README.md)

详细安装指南：[`docs/WINDOWS_SETUP.md`](./docs/WINDOWS_SETUP.md)

## 这个项目解决什么问题

VoiceCoding 在 Windows 上把语音变成文字，全程离线：

- 鼠标前侧键：开始 / 结束录音
- 录音中按鼠标后侧键：停止录音，并把转写文本在光标处本地输入（开启自动粘贴时走剪贴板 + Ctrl+V）
- 键盘热键（默认 Ctrl+Shift+Space）：不碰鼠标也能开始 / 结束录音
- 可选手势：按住触发键并移动指针超过阈值，触发方向动作（录音、回车、虚拟桌面切换）
- 转写由本地 faster-whisper large-v3-turbo 完成（float16，约 2 GB 显存），数据不出本机

核心目标是低摩擦、可日常稳定使用、纯本地文本输出，并且每个环节失败时都有回退路径。

## 环境要求

- Windows 10/11 x64
- Python >= 3.10
- NVIDIA GPU，驱动需支持 CUDA 12（已在 8 GB 级笔记本 GPU 上验证；float16 模型占用约 2 GB 显存）
- 一个麦克风
- 没有 NVIDIA GPU？纯 CPU 模式也可用，见下方的 CPU 安装方案

## 安装

GPU 模式（推荐）：

```powershell
cd VoiceCoding
pip install -U pip
pip install -e ".[gpu]"
```

`gpu` extra 会安装 `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`，CUDA 推理必需。

纯 CPU 方案：

```powershell
pip install -e .
set VOICECODING_DEVICE=cpu
```

快捷方式：`scripts/setup.ps1` 会创建 `.venv`、安装 GPU 版本并打印后续步骤：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

### 模型下载（首次运行）

large-v3-turbo 模型（CTranslate2 仓库 `deepdml/faster-whisper-large-v3-turbo-ct2`，约 1.6 GB）会在首次启动时自动从 Hugging Face 下载。

网络慢或在中国大陆？首次运行前先设置镜像：

```bat
:: cmd
set HF_ENDPOINT=https://hf-mirror.com
```

```powershell
# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 运行

```powershell
voicecoding run
```

看到 `VoiceCoding ready.` 说明监听已就绪。按 Ctrl+C 停止。

## 默认映射与状态逻辑

- `VOICECODING_FRONT_BUTTON` 默认：`x2`
- `VOICECODING_REAR_BUTTON` 默认：`x1`

| 输入 | 状态 | 动作 |
|---|---|---|
| 前侧键 | 空闲 | 开始录音 |
| 前侧键 | 录音中 | 结束录音 |
| 后侧键 | 录音中 | 停止录音并在光标处输入转写文本（开启自动粘贴时走剪贴板 + Ctrl+V） |
| 后侧键 | 空闲 | 发送 Enter（`VOICECODING_ENTER_MODE`） |
| 录音热键（默认 Ctrl+Shift+Space） | 任意 | 开始 / 结束录音 |
| Enter（默认 VK_RETURN=13，见 `VOICECODING_RECORDING_SUBMIT_KEYCODE`） | 录音中 | 停止并提交转写 |

如果你的鼠标物理定义相反：

```bat
set VOICECODING_FRONT_BUTTON=x1
set VOICECODING_REAR_BUTTON=x2
```

### 手势（可选）

用 `VOICECODING_GESTURES_ENABLED=true` 开启。按住配置的触发键（默认：鼠标右键）并把指针移动超过 `VOICECODING_GESTURE_THRESHOLD_PX`，即可触发方向动作（`VOICECODING_GESTURE_UP/DOWN/LEFT/RIGHT_ACTION`：`record_toggle`、`send_enter`、`workspace_left`、`workspace_right`、`noop`）。`workspace_left` / `workspace_right` 切换 Windows 虚拟桌面（Win+Ctrl+左/右）。启用 `VOICECODING_GESTURE_FREEZE_POINTER` 时指针在手势期间被固定；启用 `VOICECODING_GESTURE_RESTORE_CURSOR` 时手势结束后光标回到原位。

## 常用配置项

| 变量 | 默认值 | 作用 |
|---|---|---|
| `VOICECODING_MODEL_NAME` | `deepdml/faster-whisper-large-v3-turbo-ct2` | faster-whisper（CTranslate2）模型仓库；默认即 large-v3-turbo 转换版 |
| `VOICECODING_DEVICE` | `cuda` | 推理设备（`cuda` / `cpu`）；CUDA 不可用或显存吃紧时 `cpu` 是逃生通道 |
| `VOICECODING_COMPUTE_TYPE` | `float16` | 模型计算精度；`int8_float16` 约省一半显存 |
| `VOICECODING_LANGUAGE` | `auto` | 说话语言；`auto` 由模型自动检测 |
| `VOICECODING_FALLBACK_TO_CPU` | `true` | CUDA 初始化失败时回退到 CPU |
| `VOICECODING_BEAM_SIZE` | `5` | 解码 beam 宽度（`1` = 贪心） |
| `VOICECODING_VAD_ENABLED` | `true` | 用语音活动检测跳过静音 |
| `VOICECODING_SAMPLE_RATE` | `16000` | 麦克风采样率（Hz） |
| `VOICECODING_CHANNELS` | `1` | 采集声道数 |
| `VOICECODING_DTYPE` | `float32` | 采集采样格式 |
| `VOICECODING_ENTER_MODE` | `enter` | 空闲时后侧键的提交模式（`enter`、`ctrl_enter`、`shift_enter`、`none`） |
| `VOICECODING_AUTO_PASTE` | `true` | 转写结果走剪贴板 + Ctrl+V，而不是合成按键输入 |
| `VOICECODING_FRONT_BUTTON` | `x2` | 绑定录音开始/结束的按键（`x1` / `x2`） |
| `VOICECODING_REAR_BUTTON` | `x1` | 绑定停止并输入 / 空闲回车的按键（`x1` / `x2`） |
| `VOICECODING_GESTURES_ENABLED` | `false` | 是否启用手势 |
| `VOICECODING_GESTURE_TRIGGER_BUTTON` | `right` | 手势触发键（`front`、`rear`、`right`） |
| `VOICECODING_GESTURE_THRESHOLD_PX` | `120` | 触发方向动作所需的指针位移（px） |
| `VOICECODING_GESTURE_FREEZE_POINTER` | `true` | 手势期间固定指针 |
| `VOICECODING_GESTURE_RESTORE_CURSOR` | `true` | 手势结束后光标回到原位 |
| `VOICECODING_GESTURE_UP_ACTION` | `record_toggle` | 方向动作（可选值见手势章节） |
| `VOICECODING_GESTURE_DOWN_ACTION` | `noop` | 方向动作 |
| `VOICECODING_GESTURE_LEFT_ACTION` | `noop` | 方向动作 |
| `VOICECODING_GESTURE_RIGHT_ACTION` | `send_enter` | 方向动作 |
| `VOICECODING_RECORD_HOTKEY_KEYCODES` | `162,160,32` | 录音热键的 Windows VK 码，逗号分隔；默认 = 左Ctrl+左Shift+空格（VK_LCONTROL=162、VK_LSHIFT=160、VK_SPACE=32） |
| `VOICECODING_RECORDING_SUBMIT_KEYCODE` | `13` | 录音中停止并提交转写的 VK 码（默认 VK_RETURN）；不设置则禁用 |
| `VOICECODING_PREWARM_ON_START` | `true` | 启动时预加载模型，降低首次识别延迟 |
| `VOICECODING_PREWARM_DELAY_S` | `0.0` | 启动后延迟多少秒再预热 |
| `VOICECODING_PRE_BUFFER_S` | `0.5` | 按下录音键前保留的滚动音频秒数，避免吃掉第一个字 |
| `VOICECODING_BUTTON_DEBOUNCE_MS` | `150` | 侧键去抖窗口（毫秒） |
| `VOICECODING_STATUS_FILE` | `<temp>/voicecoding-status.json` | 运行状态 JSON（状态栏/脚本读取） |
| `VOICECODING_TEMP_DIR` | `<temp>/voicecoding` | 临时录音目录 |

取值格式说明：

- 布尔值接受 `true` / `1` / `yes` / `on`（不区分大小写）；其他任何值都视为 false。
- 数值变量用普通整数字符串/小数字符串（如 `120`、`0.5`）。
- 所有配置都是带 `VOICECODING_` 前缀的环境变量。

完整配置以 `voicecoding/config.py` 为准。

## 内置自检 Doctor

运行：

```powershell
voicecoding doctor
```

检查项包括：

- 配置加载是否有效
- 麦克风输入设备可用性
- 转写后端健康（`faster_whisper` / `ctranslate2` 可导入、CUDA 设备数、模型缓存）
- 自启动项状态

重建自启动项：

```powershell
voicecoding doctor --fix
```

## Deploy 与自启动

> **重要：** 你在哪个 Python 环境里运行 `voicecoding deploy`，自启动用的就是那个解释器。部署前先激活你真正想用的环境。

`voicecoding deploy` 会准备好自启动：

- 写入 `%APPDATA%\voicecoding\run.bat`（你的环境变量 + deploy 运行环境的 `python.exe` 绝对路径）
- 写入 `%APPDATA%\voicecoding\create_startup.vbs`
- 运行一次 `cscript //nologo`，在启动文件夹创建快捷方式 `voicecoding.lnk`

常用参数：

- `--preset <name>` — 应用预设配置档
- `--dry-run` — 只显示将要写入的内容，不实际修改
- `--skip-autostart` — 只写 `run.bat`，跳过启动快捷方式
- `--autostart startup|schtasks|none` — 选择自启动机制；启动文件夹快捷方式不可用时可用 `schtasks`（ONLOGON 计划任务）兜底，`none` 跳过

以后想取消自启动：删除 `%APPDATA%\voicecoding` 和启动文件夹（`shell:startup`）里的 `voicecoding.lnk`。

## 故障排查（短版）

### 侧键变成浏览器的"前进/后退"

如果低级鼠标钩子安装失败，侧键会保持系统默认行为（浏览器前进/后退）。VoiceCoding 会记录一条警告，键盘热键仍然可用。先跑 `voicecoding doctor` 检查环境，然后重启程序。

### 文本进不去管理员权限（run-as-admin）窗口

Windows UIPI 会拦截合成按键（Enter、Ctrl+V、Win+Ctrl+方向键），非管理员运行的 VoiceCoding 无法向管理员窗口输入文字。这是系统安全边界，不是 bug。请在普通权限窗口中使用，或接受提权窗口需要手动输入。

### GPU 显存不足

浏览器和游戏可能占掉大部分显存。遇到 CUDA OOM 时，要么切到 CPU：

```bat
set VOICECODING_DEVICE=cpu
```

要么缩小 GPU 占用：

```bat
set VOICECODING_COMPUTE_TYPE=int8_float16
```

### 首次模型下载慢

首次启动会从 Hugging Face 下载约 1.6 GB。网络慢或在中国大陆时，首次运行前设置 `HF_ENDPOINT=https://hf-mirror.com`（见安装章节）。

## License

项目源码采用 Apache-2.0，版权声明 "Copyright 2026 VoiceCoding Contributors"，详见 `LICENSE`。

项目来源与第三方依赖、模型资产声明见 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。
