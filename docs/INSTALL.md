# 安装与移植（INSTALL）

目标：把「本地 AI 服务层」装到另一台 Windows 机器（或另一个目录），
只改配置/环境变量，不改代码。

---

## 0. 需要什么

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10/11 x64（Linux 也能跑，ffmpeg/路径按需调整） |
| Python | **3.11+**（本机验证 3.13.9） |
| 显卡 | NVIDIA，建议 ≥ 8GB 显存（RTX 3070 已验证） |
| 磁盘 | 模型约 9GB + 输出/日志预留几 GB |
| ffmpeg | 需要 `ffmpeg.exe` + `ffprobe.exe` |
| 网络 | 首次装依赖/下模型需要；运行时可离线 |

仓库里**不含**：模型权重、音色、输出、日志、虚拟环境。
这些都在 `LOCALAI_HOME` 下，按本文重建/拷贝即可。

---

## 1. 放置代码

把仓库放到任意目录，例如 `D:\LocalAIService`。目录长这样：

```
LocalAIService\
  pyproject.toml
  src\localai\          ← 服务层（Python 包）
  engine\               ← 原生推理实现（engine.py / asr_engine.py / ...）
  engine\indextts_vendor\  ← IndexTTS-2.5 推理源码（约 3.6MB，已随仓库）
  svc\service.py        ← 旧入口兼容壳
  start_ai_service.ps1  ← 正式启动脚本（UTF-8 BOM）
  docs\ tests\ integrations\
  models\ voices\ outputs\ logs\   ← 数据，不进 git
```

> 数据目录可以不在代码目录里：设置 `LOCALAI_HOME` 指向数据根即可。
> 例如代码放 `D:\apps\localai`，模型/音色放 `E:\LocalAIService`：
> `set LOCALAI_HOME=E:\LocalAIService`。

`engine\` 是随仓库分发的原生引擎目录；如果单独放，设置
`LOCALAI_ENGINE_DIR` 指向它。服务层通过环境变量把路径传给原生模块，
不写死 `E:\`。

---

## 2. 创建虚拟环境

```powershell
cd E:\LocalAIService
py -3.13 -m venv engine\.venv
engine\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
```

> 虚拟环境放在 `engine\.venv` 是启动脚本的默认约定；换位置就改
> `start_ai_service.ps1` 里的 `$python`，或直接用自己的 python。

---

## 3. 安装服务层（轻量依赖）

```powershell
engine\.venv\Scripts\python.exe -m pip install -e .
```

这会装上 `fastapi / uvicorn / httpx / python-multipart / pydantic /
soundfile / numpy`，并注册 `localai` 命令。

验证：

```powershell
engine\.venv\Scripts\localai.exe version
engine\.venv\Scripts\localai.exe --help
```

> 没装成也能用源码跑：设置 `PYTHONPATH=<仓库>\src`，然后
> `python -m localai ...`。启动脚本已经自动设置了 `PYTHONPATH`。

---

## 4. 安装 GPU 引擎依赖（torch 单独装）

**不要**直接 `pip install torch`，会拿到 CPU 版。按 CUDA 版本用官方索引：

```powershell
# CUDA 13.0（本机 torch 2.11+cu130）
engine\.venv\Scripts\python.exe -m pip install torch ^
  --index-url https://download.pytorch.org/whl/cu130

# 或者 CUDA 12.8
# --index-url https://download.pytorch.org/whl/cu128
```

然后装推理依赖：

```powershell
engine\.venv\Scripts\python.exe -m pip install ^
  funasr demucs transformers soundfile librosa scipy ^
  modelscope huggingface_hub
```

如果需要与当前机器完全一致的版本，参考
`engine\requirements.lock.txt`（这是本机 `pip freeze` 的结果，
可以 `pip install -r engine\requirements.lock.txt`，但要注意 CUDA 版本匹配）。

验证：

```powershell
engine\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. 准备模型

先体检，看缺什么：

```powershell
engine\.venv\Scripts\localai.exe models list
```

自动下载（**ModelScope 优先**，失败回退 HuggingFace）：

```powershell
engine\.venv\Scripts\localai.exe models download all
# 或单个：
engine\.venv\Scripts\localai.exe models download tts
engine\.venv\Scripts\localai.exe models download asr
engine\.venv\Scripts\localai.exe models download speaker
engine\.venv\Scripts\localai.exe models download demucs
```

| 能力 | 模型 | 目标目录（默认） | 大小 |
| --- | --- | --- | --- |
| TTS | IndexTTS-2.5（含 hf_cache 辅助模型） | `models\TTS\IndexTTS-2.5` | ≈ 8 GB |
| ASR | SenseVoiceSmall | `models\SenseVoiceSmall` | ≈ 0.9 GB |
| 声纹 | campplus | `models\campplus` | ≈ 27 MB |
| 分离 | Demucs htdemucs / htdemucs_6s | `models\torch\hub\checkpoints` | ≈ 130 MB |

说明：

- **htdemucs_ft** 是 4 个模型的 bag，本项目默认不下载；缺失是正常的，
  `models list` / `doctor` 会明确标注「可选未下载」。
- 从旧机器迁移：直接拷贝 `models\`、`voices\` 两个目录即可，不需要重新下载。
- 模型目录不在默认位置时，用 `LOCALAI_MODEL_DIR` / `LOCALAI_ASR_MODEL_DIR` /
  `LOCALAI_CAMPPUS_DIR` / `LOCALAI_DEMUCS_REPO` 覆盖。

---

## 6. ffmpeg

探测顺序：

1. `LOCALAI_FFMPEG`（完整路径，优先）；
2. 本机默认 `C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe`；
3. `PATH` 里的 `ffmpeg`。

`ffprobe` 同目录优先；也可用 `LOCALAI_FFPROBE` 单独指定。

建议把 ffmpeg 的 `bin` 加进 PATH，或设置：

```powershell
$env:LOCALAI_FFMPEG = "D:\tools\ffmpeg\bin\ffmpeg.exe"
$env:LOCALAI_FFPROBE = "D:\tools\ffmpeg\bin\ffprobe.exe"
```

---

## 7. 启动

### 7.1 前台（调试）

```powershell
engine\.venv\Scripts\localai.exe serve
# 或
engine\.venv\Scripts\python.exe -m localai serve
```

默认 `127.0.0.1:7880`；`--host` / `--port` 或 `LOCALAI_HOST` / `LOCALAI_PORT` 可改。

### 7.2 正式启动脚本

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1
```

参数名保持兼容：

```powershell
-IdleSeconds 900            # TTS 空闲卸载
-AsrIdleSeconds 900
-SpeakerIdleSeconds 900
-SeparateIdleSeconds 900
```

设为 0 表示关闭对应能力的空闲卸载。脚本会：

- 设置 `LOCALAI_HOME` / `PYTHONPATH` / `TORCH_HOME` / 各空闲秒数；
- 用 `engine\.venv\Scripts\python.exe -m localai serve` 启动；
- 日志写到 `logs\service.out.log` / `logs\service.err.log`。

### 7.3 后台/隐藏启动（推荐）

```powershell
$py = "E:\LocalAIService\engine\.venv\Scripts\python.exe"
$cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File E:\LocalAIService\start_ai_service.ps1"
$startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance(); $startup.ShowWindow = 0
[void]([wmiclass]"Win32_Process").Create($cmd, "E:\LocalAIService", $startup)
```

然后：

```powershell
curl.exe -s http://127.0.0.1:7880/health
```

不要用会阻塞前台的 `Start-Process -Wait` 来“重启”；先结束监听 7880 的进程，
再用上面的 WMI 方式拉起。

### 7.4 开机自启

任选一种：

- 任务计划程序：触发器「登录时」，操作
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File E:\LocalAIService\start_ai_service.ps1`；
- 启动文件夹放一个 `.vbs`/快捷方式，调用上面的命令；
- 用 NSSM 注册为服务（注意工作目录设为仓库根）。

---

## 8. 验证

```powershell
engine\.venv\Scripts\localai.exe doctor
Invoke-RestMethod http://127.0.0.1:7880/health
Invoke-RestMethod http://127.0.0.1:7880/capabilities
```

跑测试：

```powershell
engine\.venv\Scripts\python.exe -m pytest -q
```

手动回归（首次会加载模型）：

```powershell
curl.exe -s -X POST http://127.0.0.1:7880/tts -H "Content-Type: application/json" -d "{\"text\":\"测试\",\"voice\":\"default\"}" -o test.wav
curl.exe -s -X POST "http://127.0.0.1:7880/asr" -F "file=@test.wav"
curl.exe -s -X POST http://127.0.0.1:7880/media/probe -F "file=@test.wav"
```

---

## 9. 环境变量总表

新变量优先，旧的 `INDEX_TTS_*` 仍兼容。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `LOCALAI_HOME` | 项目根 | 数据根目录；音色/输出/日志/模型都从它推导 |
| `LOCALAI_HOST` / `LOCALAI_PORT` | `127.0.0.1` / `7880` | 监听地址 |
| `LOCALAI_ENGINE_DIR` | `<HOME>\engine` | 原生引擎目录 |
| `LOCALAI_TTS_BACKEND` | `engine` | `engine` 或 `comfy` |
| `LOCALAI_MODEL_DIR` | `<HOME>\models\TTS\IndexTTS-2.5` | TTS 权重 |
| `LOCALAI_ASR_MODEL_DIR` | `<HOME>\models\SenseVoiceSmall` | ASR 权重 |
| `LOCALAI_CAMPPUS_DIR` | `<HOME>\models\campplus` | 声纹权重 |
| `LOCALAI_VOICES_DIR` | `<HOME>\voices` | 音色目录 |
| `LOCALAI_OUTPUT_DIR` | `<HOME>\outputs` | 输出/临时/抽帧 |
| `LOCALAI_LOGS_DIR` | `<HOME>\logs` | 日志与显存估算 |
| `LOCALAI_TORCH_HOME` | `<HOME>\models\torch` | Demucs/torch 缓存 |
| `LOCALAI_DEMUCS_REPO` | `<TORCH_HOME>\hub\checkpoints` | Demucs 权重 |
| `LOCALAI_VRAM_ESTIMATES` | `<HOME>\logs\vram_estimates.json` | 显存估算校准 |
| `LOCALAI_VRAM_MARGIN_MB` | `512` | 加载时预留显存余量 |
| `LOCALAI_IDLE_UNLOAD` | `900` | TTS 空闲卸载秒数，0=关 |
| `LOCALAI_ASR_IDLE_UNLOAD` | `900` | ASR 空闲卸载 |
| `LOCALAI_SPEAKER_IDLE_UNLOAD` | `900` | 声纹空闲卸载 |
| `LOCALAI_SEPARATE_IDLE_UNLOAD` | `900` | 分离空闲卸载 |
| `LOCALAI_FFMPEG` / `LOCALAI_FFPROBE` | 自动探测 | ffmpeg 路径 |
| `LOCALAI_TOKEN` | 空 | 可选 Bearer 鉴权 |
| `LOCALAI_LOG_LEVEL` | `INFO` | 日志级别 |
| `LOCALAI_COMFY_BASE` | `http://127.0.0.1:8188` | ComfyUI 地址（comfy 后端） |
| `LOCALAI_COMFY_MODEL_DIR` | `D:\Comfy-Desktop\视频\ComfyUI\models\TTS\IndexTTS-2.5` | comfy 后端节点权重目录 |

---

## 10. 从旧机器迁移清单

1. 拷贝整个代码目录（不含 `.venv` 也行，重装更快）；
2. 拷贝 `models\`、`voices\`；
3. 新机器装 Python + venv + 依赖（第 2~4 节）；
4. 设置 `LOCALAI_HOME`（如果数据不在代码目录）；
5. `localai doctor` → 全绿；
6. 启动并跑第 8 节验证。

`logs\vram_estimates.json` 可以一起拷贝（显存估算会更快收敛），
不拷贝也会自动重新校准。

---

## 11. 排错

| 现象 | 处理 |
| --- | --- |
| `localai` 命令找不到 | 用 `python -m localai`；或重新 `pip install -e .` |
| `engine` 模块导入失败 | 检查 `LOCALAI_ENGINE_DIR` 是否有 `engine.py` |
| 模型缺失 | `localai models list` / `models download all` |
| ffmpeg missing | 设 `LOCALAI_FFMPEG` 或加 PATH |
| 端口被占 | `Get-NetTCPConnection -LocalPort 7880`；先停旧进程 |
| 首次 TTS 很慢 | 正常（加载 ~6GB 模型）；看 `logs\service.*.log` |
| 507 显存不足 | `POST /vram/unload {"all":true}`，关闭其它 GPU 程序 |
| 中文日志乱码 | 设 `PYTHONUTF8=1`（启动脚本已设） |
| torch CUDA 不可用 | 重装 CUDA 版 torch（第 4 节） |
| `models download` 失败 | 装 `modelscope`；或手动按第 5 节目录放置 |

体检报告示例：

```powershell
engine\.venv\Scripts\localai.exe doctor
```
