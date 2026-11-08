# 本地 AI 服务管理器（TTS + ASR + 声纹 + 媒体 + 下载）

> 项目原名 **IndexTTS**，因能力已扩展到 TTS / ASR / 声纹 / 视频抽帧，改名为
> **LocalAIService（本地 AI 服务管理器）**。真实目录 `E:\LocalAIService`，
> `E:\IndexTTS` 保留为指向它的 **目录 junction**，旧脚本、旧快捷方式、旧命令行
> 全部照旧可用。迁移细节和回滚方法见文末「15. 改名与迁移」。

把 IndexTTS-2.5 的推理从「ComfyUI 节点」改成**独立 Python 引擎**，再用原来的
FastAPI 服务对外提供 HTTP 接口；同时接入 **SenseVoiceSmall** 做中文/多语种语音识别，
用 **campplus** 做 192 维声纹识别（判断「是谁在说话」），
并用 **ffmpeg** 提供媒体探测、抽帧、抽音轨、视频转写（`/media/*`），
以及 **yt-dlp** 的通用视频/音频下载与「链接 -> 字幕时间轴 + 镜头边界」分析
（`/media/download*`、`/media/analyze`，见文末第 18 章）。
TTS / ASR / 声纹 / 人声分离 四者都采用「第一次请求才加载、空闲自动卸载」策略，
各有一把推理锁、一个看门狗，互不干扰；四个能力加载前再统一经过
`src\localai\core\vram.py` 做**按需腾显存（LRU + 安全余量）**，启动时仍然是 0 模型。
TTS 接口契约保持不变：

- 监听：`127.0.0.1:7880`
- 后端：`TTS_BACKEND=engine`（默认）/ `comfy`（旧路径，保留但当前环境不可用）
- TTS 模型：`E:\LocalAIService\models\TTS\IndexTTS-2.5`（约 6 GB，含 `hf_cache` 辅助模型）
- ASR 模型：`E:\LocalAIService\models\SenseVoiceSmall`（约 936 MB）
- 声纹模型：`E:\LocalAIService\models\campplus`（`campplus_cn_common.bin` 约 27 MB）
- 声纹库：`E:\LocalAIService\speakers`（只存 192 维向量 + 名字，**不存原始音频**）
- 音色：`E:\LocalAIService\voices`，每次请求重新扫描，新增文件不用重启
- 输出：`E:\LocalAIService\outputs`，每次成功合成留一份 WAV；ASR / 声纹临时文件放
  `outputs\asr_tmp\`、`outputs\speaker_tmp\`，媒体接口临时文件放 `outputs\media_tmp\`，
  处理完立即删除
- 抽帧结果：`E:\LocalAIService\outputs\frames\<时间戳>\`，文件名带时间点，留档不自动删
- ffmpeg：默认 `C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe`，
  可用 `INDEX_TTS_FFMPEG` / `INDEX_TTS_FFPROBE` 覆盖

当前状态：**已跑通**。`curl 127.0.0.1:7880/health` 返回
`backend=engine`、`model_ready=true`；TTS / ASR / 声纹 / 人声分离 都是按需加载，
启动后不占显存；四个能力加载前统一由显存管家按 LRU 腾地方，显存不足返回 507。

---

## 0. 两层架构（v2.2 重构）

从 v2.2 起，项目拆成两层，**服务层可移植、可分享；插件层只走 HTTP**：

```
第二层  DSH 插件（本仓库只定契约，不实现）
          │  只通过 HTTP：/capabilities /health /tts /asr /voices /vram ...
          ▼
第一层  本地 AI 服务层（src\localai，独立 Python 包 localai-service）
          │  统一 FastAPI 装配 + 能力注册表 + 显存管家 + 空闲看门狗
          ▼
        原生引擎（engine\*.py，按需惰性 import，不写死在服务层里）
```

- **新入口**：`python -m localai serve`（或安装后的 `localai serve`）；
  `start_ai_service.ps1` 已改为调用它，参数名 `-IdleSeconds` 等保持兼容。
- **旧入口**：`svc\service.py` 变成薄壳，`from localai.legacy import app, main`，
  旧调用不会断。
- **配置集中**：`LOCALAI_HOME`（默认项目根）推导所有目录；换机器只改环境变量，
  代码里没有写死 `E:\`。完整变量表见 [`docs/INSTALL.md`](docs/INSTALL.md)。
- **能力自发现**：新增 `GET /capabilities`，返回 tts/asr/speaker/separate/media
  的 ready/loaded/模型名/空闲秒数/监听地址，任意程序可直接接入。
- **命令**：`localai doctor`（体检）、`localai models list|download`（查/下权重）、
  `localai version`。
- **文档**：
  - 接口文档（给调用方）：[`docs/SERVICE_API.md`](docs/SERVICE_API.md)
  - 插件契约（给未来 DSH 插件）：[`docs/PLUGIN_CONTRACT.md`](docs/PLUGIN_CONTRACT.md)
  - 安装/移植：[`docs/INSTALL.md`](docs/INSTALL.md)

```bash
curl http://127.0.0.1:7880/capabilities
localai doctor
```

### 0.1 管理界面（WebUI）

服务自带一个纯静态管理页面（无 npm、无 CDN、暗色中文界面）：

```text
http://127.0.0.1:7880/ui
```

五个页面：**概览 / 模型 / 设置 / 任务 / 系统**。可查看 GPU 与能力状态、下载或
切换模型档位、按分组修改配置（写 `<LOCALAI_HOME>/config.json`，只写改动项）、
取消任务、启停开机自启、查看日志、二次确认后重启服务。

任务页支持**双击任务行打开详情弹窗**（也可点「详情」按钮）：概要、请求参数原文、
结果 JSON 与音频/图片预览、完整错误原文、created → started → finished 时间线，
并可在弹窗内取消 queued/running 任务；弹窗打开期间列表自动刷新会暂停。
细节见 [`docs/WEBUI.md`](docs/WEBUI.md#11-任务详情弹窗)。

- 接口文档：[`docs/WEBUI.md`](docs/WEBUI.md)（含 `/admin/*` 清单与设置白名单）
- `/admin/*` 受 `LOCALAI_TOKEN` 保护；`/ui` 与静态资源免鉴权，令牌只存浏览器 localStorage
- 默认只监听 `127.0.0.1`；非回环地址时页面顶部会显示安全警告

---

## 1. 目录结构

```
E:\LocalAIService\                  # 真实目录（E:\IndexTTS 是指向这里的 junction）
├─ pyproject.toml                  # 包定义：localai-service；console_scripts: localai
├─ src\localai\                    # 【第一层】服务层（可移植 Python 包）
│  ├─ config.py                    # 唯一配置入口（LOCALAI_HOME / 路径 / idle / ffmpeg）
│  ├─ constants.py                 # 与部署无关的业务常量（格式/模型文件/下载源）
│  ├─ cli.py                       # localai serve | doctor | models list|download | version
│  ├─ core\                        # registry / vram（显存管家）/ idle（看门狗）/ errors / voices / audio
│  ├─ engines\                     # 能力适配器：tts_indextts / asr_sensevoice / speaker_campplus /
│  │                               #              separate_demucs / media_ffmpeg（+ tts_comfy 兼容）
│  ├─ api\                         # app.py 装配 + routers\ 按能力拆分 + schemas
│  └─ legacy.py                    # 旧入口兼容（svc\service.py 转发到这里）
├─ engine\                         # 原生推理实现（服务层通过 adapter 惰性 import）
│  ├─ .venv\                       # Python 3.13.9 + torch 2.11 cu130 + transformers 4.49
│  ├─ engine.py                    # TTS 引擎：加载模型 / synthesize / 命令行自检
│  ├─ asr_engine.py                # ASR 引擎：懒加载 SenseVoice / 富标签解析 / unload
│  ├─ speaker_engine.py            # 声纹引擎：campplus 192 维向量 / 登记 / 识别 / 未知聚类
│  ├─ separator_engine.py          # 人声分离引擎：Demucs 懒加载 / 分离 / unload
│  ├─ ffmpeg_tool.py               # 媒体工具：probe / 转 16k wav / 抽帧 / 抽音轨
│  ├─ indextts_vendor\             # vendored 推理源码（约 3.6MB，已随仓库）
│  ├─ requirements.lock.txt        # 当前引擎 venv 的完整包版本（pip freeze）
│  └─ start_engine_service.ps1     # 旧启动壳（仍可用，内部仍指向 svc\service.py）
├─ svc\service.py                  # 兼容壳：import 新包并导出 app（不再含业务逻辑）
├─ docs\                           # SERVICE_API.md / PLUGIN_CONTRACT.md / INSTALL.md
├─ tests\                          # pytest 冒烟测试（不加载模型）
├─ integrations\dsh\README.md      # 【第二层】DSH 插件占位（只文档，不写实现）
├─ models\TTS\IndexTTS-2.5\        # TTS 主权重 + hf_cache 辅助模型
├─ models\SenseVoiceSmall\         # ASR 权重（model.pt 936 MB + 配置/词表）
├─ models\campplus\                # 声纹权重（campplus_cn_common.bin + config.yaml）
├─ speakers\                       # 声纹库：<名字>.json + unknown.json（只有向量，无音频）
├─ voices\                         # 参考音色（default.mp3 / multi.wav / ...）
├─ outputs\                        # 合成留档；asr_tmp\ / speaker_tmp\ / media_tmp\ 是临时目录
│  └─ frames\<时间戳>\             # /media/frames 抽出的帧（留档，不自动删）
├─ logs\                           # service.*.log / vram_estimates.json
├─ start_ai_service.ps1            # 正式启动脚本（调用 python -m localai serve）
└─ start_service_engine.ps1        # 旧名字兼容壳（转交给 start_ai_service.ps1）
```

> `models\`、`voices\`、`outputs\`、`logs\`、`speakers\`、`engine\.venv\`
> 都不进 git（见 `.gitignore`）；`engine\indextts_vendor\` 约 3.6MB，随仓库分发。


引擎 venv 里的关键版本：

| 包 | 版本 |
|---|---|
| Python | 3.13.9 |
| torch | **2.11.0+cu130** |
| torchaudio | **2.11.0+cu130** |
| transformers | **4.49.0** |
| tokenizers | 0.21.4 |
| funasr | **1.4.24** |
| kaldiio / hydra-core | 2.18.1 / 1.3.8 |
| fastapi / uvicorn | 0.162.2 / 0.54.0 |
| soundfile | 0.14.0 |

> 为什么是 torch 2.11 而不是 Comfy 的 2.12.1：PyTorch 官方 cu130 索引里
> `torchaudio` 只出到 2.11.0，torch/torchaudio 必须成对，所以引擎用 2.11.0。
> 为什么是 transformers 4.49：vendored 代码 import 了
> `transformers.generation.beam_constraints`，4.49 还有这个模块；5.x 已移除。

---

## 2. 启动服务

### 本机当前使用：start_ai_service.ps1（TTS + ASR + 声纹 + 媒体）

```powershell
# 默认：TTS / ASR / 声纹 / 人声分离 都是空闲 900 秒卸载
powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1

# 调试：四边都改成 60 秒空闲卸载
powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 `
  -IdleSeconds 60 -AsrIdleSeconds 60 -SpeakerIdleSeconds 60 -SeparateIdleSeconds 60

# 关闭某一侧的空闲卸载（0 = 关闭）
powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 `
  -IdleSeconds 0 -AsrIdleSeconds 0 -SpeakerIdleSeconds 0 -SeparateIdleSeconds 0
```

> 旧名字 `start_service_engine.ps1` 仍然存在，是一个只转交参数的同名壳，
> 旧计划任务/旧命令行不用改；两个脚本都在服务根目录下，参数完全一致。

参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `-IdleSeconds` | `900` | TTS 空闲多少秒卸载；`0` = 关闭 |
| `-AsrIdleSeconds` | `900` | ASR 空闲多少秒卸载；`0` = 关闭 |
| `-SpeakerIdleSeconds` | `900` | 声纹空闲多少秒卸载；`0` = 关闭 |
| `-SeparateIdleSeconds` | `900` | 人声分离空闲多少秒卸载；`0` = 关闭 |

脚本内部把它们分别写成环境变量 `LOCALAI_IDLE_UNLOAD`、
`LOCALAI_ASR_IDLE_UNLOAD`、`LOCALAI_SPEAKER_IDLE_UNLOAD`、
`LOCALAI_SEPARATE_IDLE_UNLOAD`（同时写旧的 `INDEX_TTS_*` 兼容名），
然后执行 `python -m localai serve`。
四个能力加载前的显存安全余量另由 `LOCALAI_VRAM_MARGIN_MB` 控制
（默认 512MB，详见第 17 章「显存管理」）。
脚本用 `Start-Process ... -Wait` 守着 python；从自动化工具里调用时，
建议用 WMI / `cmd /c start` 放后台，再在下一轮 `curl /health`。

### 推荐：后台启动（带健康检查）

```powershell
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Background
```

脚本会：

1. 检查 `E:\LocalAIService\engine\.venv\Scripts\python.exe`；
2. 检查 fastapi / uvicorn / soundfile / torch，缺 Web 依赖会自动补装；
3. 设置 `TTS_BACKEND=engine`、`INDEX_TTS_PORT=7880`；
4. 后台启动 `E:\LocalAIService\svc\service.py`，日志写到
   `E:\LocalAIService\logs\service.out.log` / `service.err.log`；
5. 轮询 `/health`，就绪后打印 JSON。

### 前台启动（调试用）

```powershell
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1
```

### 覆盖旧服务

```powershell
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Restart -Background
```

`-Restart` 只会结束命令行里带 `service.py` 的进程，**不会碰 ComfyUI 的 python**。
如果 7880 被别的程序占用，脚本会拒绝并打印对方的 PID/命令行。

### 手动启动（等价命令）

```powershell
# 新入口（推荐）；在仓库根目录执行，或先 pip install -e .
$env:LOCALAI_HOME = 'E:\LocalAIService'
$env:PYTHONPATH = 'E:\LocalAIService\src'
E:\LocalAIService\engine\.venv\Scripts\python.exe -m localai serve

# 或安装后的命令
E:\LocalAIService\engine\.venv\Scripts\localai.exe serve
```

### 停止

```powershell
Get-NetTCPConnection -LocalPort 7880 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 3. 后端切换

`service.py` 读取环境变量 `TTS_BACKEND`：

| 值 | 行为 |
|---|---|
| `engine`（默认） | 直接调用 `E:\LocalAIService\engine\engine.py` 的常驻 `IndexTTS2` 实例，不经过 ComfyUI |
| `comfy` | 保留原来的「上传参考音频 → `/prompt` → 轮询 `/history` → 下载 FLAC」链路 |

切回旧链路：

```powershell
$env:TTS_BACKEND = 'comfy'
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Restart -Background
```

> 当前环境 `comfy` 仍会报
> `ModuleNotFoundError: No module named 'transformers.generation.beam_constraints'`，
> 原因见第 10 节。不要为了它降级 ComfyUI 的 transformers。

---

## 4. 接口（旧契约不变，新增 `/media/*`）

### GET /health

engine 模式返回：

```json
{
  "ok": true,
  "backend": "engine",
  "model_ready": true,
  "engine_loaded": true,
  "voices": 6,
  "model_detail": "独立引擎模型文件已就绪（模型已加载）",
  "model_dir": "E:\\LocalAIService\\models\\TTS\\IndexTTS-2.5",
  "engine_dir": "E:\\LocalAIService\\engine",
  "idle_unload_seconds": 900,
  "idle_seconds": 42.1,
  "asr": {"ready": true, "loaded": false, "idle_unload_seconds": 900},
  "speaker": {"ready": true, "loaded": false, "idle_unload_seconds": 900},
  "media": {
    "ffmpeg": "ok",
    "path": "C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\ffmpeg.exe",
    "ffprobe": "ok",
    "ffprobe_path": "C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\ffprobe.exe"
  },
  "comfy": "disabled"
}
```

- `ok` / `model_ready`：主权重 + `hf_cache` 辅助模型是否齐全。
- `engine_loaded`：TTS 模型是否已经加载（第一次 `/tts` 之前是 `false`）。
- `asr.ready`：SenseVoice 模型文件是否齐全。
- `asr.loaded`：ASR 模型是否已经加载（第一次 `/asr` 之前是 `false`）。
- `speaker.ready`：campplus 声纹模型文件是否齐全。
- `speaker.loaded`：声纹模型是否已经加载（第一次声纹请求 / `/asr` 之前是 `false`）。
- `media.ffmpeg`：`ok` 表示 ffmpeg/ffprobe 可用；`missing` 时会额外给出
  `expected`（期望路径）和 `detail`（怎么修），见「14. ffmpeg 媒体接口」。
- `backend`：`engine` 或 `comfy`。

### GET /voices

```json
{
  "voices": [
    {"name": "default", "path": "E:\\LocalAIService\\voices\\default.mp3", "filename": "default.mp3", "format": "mp3", "duration": 3.157},
    {"name": "multi", "path": "E:\\LocalAIService\\voices\\multi.wav", "filename": "multi.wav", "format": "wav", "duration": 10.348}
  ],
  "count": 2,
  "directory": "E:\\LocalAIService\\voices"
}
```

音色名 = 文件名去扩展名；同名不同扩展名时第二个起自动加 `_2`。

### POST /tts

```powershell
$json = '{"text":"主人，我是大肥鱼。","voice":"default"}'
[System.IO.File]::WriteAllText("$env:TEMP\tts.json", $json, [System.Text.UTF8Encoding]::new($false))

curl.exe -s -D - -X POST http://127.0.0.1:7880/tts `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$env:TEMP\tts.json" `
  --output E:\LocalAIService\outputs\test.wav
```

成功返回 `HTTP 200`、`Content-Type: audio/wav`，响应体是 16bit PCM WAV。
响应头保留：

- `X-Seed`：本次实际种子
- `X-Language`：本次语言
- `X-Audio-Path` / `X-Audio-Filename`：留档路径和文件名

字段：

| 字段 | 说明 |
|---|---|
| `text` | 必填，1~5000 字符 |
| `voice` | 音色名，默认 `default`；不存在返回 404 |
| `speed` | 0.5~2.0；内部换算成上游 `duration_factor = 1/speed` |
| `emotion` | 8 个 0~1 数字 → 情感向量；文本 → 需要 `INDEX_TTS_QWEN_EMO=1` |
| `seed` | 0~4294967295；不填随机 |
| `language` | `zh/en/ja/es/ar`；不填自动判断 |

### POST /v1/audio/speech（OpenAI 兼容）

```powershell
$body = '{"model":"indextts","input":"主人，我是大肥鱼。","voice":"default","response_format":"wav"}'
[System.IO.File]::WriteAllText("$env:TEMP\openai.json", $body, [System.Text.UTF8Encoding]::new($false))

curl.exe -s -X POST http://127.0.0.1:7880/v1/audio/speech `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$env:TEMP\openai.json" `
  --output E:\LocalAIService\outputs\openai.wav
```

只支持 `response_format=wav`；错误按 OpenAI 的 `{"error": {...}}` 形状返回。

### POST /asr（语音识别）

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | 是 | **音频或视频**：wav/mp3/flac/ogg/opus/m4a/aac/wma/amr/aiff/ape，或 mp4/mkv/mov/avi/flv/webm/m4v/ts/mpg/wmv/3gp/rmvb/vob，上限 200 MB |
| `language` | 否 | `auto`（默认）/ `zh` / `en` / `ja` / `ko` / `yue` / `nospeech` |

> 非 wav 的文件会先用 ffmpeg 转成 16kHz 单声道 wav 再识别，临时文件处理完立即删除；
> 返回体里会多 `source_format`（上传的原扩展名）和 `converted`（是否转过）两个字段。
> 细节见「14. ffmpeg 媒体接口」。

查询参数 `?speaker=0` 可以关掉声纹（默认 `1` 开启）：关掉时**不会加载 campplus**，
返回体里的 `speaker` 是 `null`，适合只需要文字、想省显存的时候。

```powershell
curl.exe -s -F "file=@E:\LocalAIService\outputs\pick_female-A.wav" `
  http://127.0.0.1:7880/asr

# 只要文字，不加载声纹
curl.exe -s -F "file=@E:\LocalAIService\outputs\pick_female-A.wav" `
  "http://127.0.0.1:7880/asr?speaker=0"
```

返回：

```json
{
  "ok": true,
  "text": "主人，我是大肥鱼，这条声音是我自己挑的，你听听看圣儿的话，我以后就用它跟你说话。",
  "language": "zh",
  "emotion": "ANGRY",
  "events": ["Speech"],
  "duration": 5.7,
  "elapsed_ms": 2616,
  "loaded": true,
  "speaker": {"name": "主人", "score": 0.83, "known": true}
}
```

- `text`：去掉 SenseVoice 富标签后的正文。
- `language`：模型识别出的语言（`zh` / `en` / `ja` ...）。
- `emotion`：`HAPPY` / `SAD` / `ANGRY` / `NEUTRAL` / `FEARFUL` /
  `DISGUSTED` / `SURPRISED` / `OTHER` / `EMO_UNKNOWN`。
- `events`：音频事件，常见 `Speech`，也可能是 `BGM` / `Laughter` /
  `Applause` / `Cry` / `Cough` / `Sneeze` / `Breath` / `Sing`。
- `duration`：音频时长（秒）。
- `elapsed_ms`：本次请求总耗时（首次包含模型加载，约 10 秒；之后约 0.1~0.4 秒）。
- `loaded`：请求结束后 ASR 是否处于已加载状态。
- `speaker`：声纹判断结果。`known=true` 时 `name` 是已登记的名字；
  `known=false` 时是未知池里的稳定编号（`未知-1`、`未知-2`…），同一个人多次
  请求会落到同一个编号。声纹算失败时这里是 `null`，并多一个 `speaker_error` 字段，
  **不影响文字结果**。
- `source_format` / `converted`：上传文件的扩展名、是否经过 ffmpeg 转 16k
  （wav 直接识别时为 `false`）。

### POST /api/v1/asr

`/asr` 的别名，参数和返回完全一致，方便别的软件接入：

```powershell
curl.exe -s -F "file=@E:\LocalAIService\outputs\pick_female-A.wav" `
  -F "language=zh" http://127.0.0.1:7880/api/v1/asr
```

### GET /asr/status

```json
{
  "ready": true,
  "loaded": false,
  "idle_seconds": null,
  "idle_unload_seconds": 900,
  "detail": "SenseVoice 模型文件已就绪（首次请求时加载）"
}
```

`idle_seconds` 在还没用过 ASR 时为 `null`；用过之后表示「距离上次请求过了多少秒」。

---

## 5. 语音识别的按需加载 / 空闲卸载

### 为什么这样做

RTX 3070 只有 8 GB。TTS 加载后约占 5~6 GB，SenseVoiceSmall 再占约 1 GB；
两边都常驻时可用余量只剩 0.6~0.7 GB。为了不和 ComfyUI、其它任务抢显存，
ASR 也照 TTS 的模式做：

1. 服务启动只挂监听：`service.py` 只 import `asr_engine.py` 的配置和函数，
   **不 import funasr、不加载模型**；
2. 第一次 `POST /asr` 才 `AutoModel(model=<本地目录>, device="cuda:0")`；
3. 空闲超过 `INDEX_TTS_ASR_IDLE_UNLOAD` 秒，看门狗卸载模型并
   `gc.collect()` + `torch.cuda.empty_cache()`；
4. 再来请求自动重新加载。

### 互不干扰的锁

- `_ASR_INFER_LOCK`（asyncio）：识别和卸载互斥，正在识别时不会被看门狗卸掉；
- `_ENGINE_INFER_LOCK`（asyncio）：TTS 自己的推理/卸载锁，和 ASR 完全独立；
- `_SPEAKER_INFER_LOCK`（asyncio）：声纹自己的推理/卸载锁，和上面两把都独立；
- `asr_engine.py` 内部用 `_LOAD_LOCK`（threading，保护加载/卸载）+
  `_InferenceGate`（容量可配的推理信号量，默认 1 = 串行；设
  `LOCALAI_ASR_PARALLEL=2~4` 才放开并发）；
  `speaker_engine.py` 内部有 `_LOAD_LOCK`（threading），防止并发重复加载；
- 三个看门狗任务各自检查自己的空闲时间，只动自己的模型。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---:|---|
| `INDEX_TTS_ASR_IDLE_UNLOAD` | `900` | ASR 空闲卸载秒数；`0` = 关闭 |
| `LOCALAI_ASR_PARALLEL` | `1` | ASR 引擎并发容量（1=串行，2~4=可并发；默认 1） |
| `INDEX_TTS_ASR_MODEL_DIR` | `E:\LocalAIService\models\SenseVoiceSmall` | 模型目录 |
| `INDEX_TTS_ASR_DEVICE` | `cuda:0` | 推理设备；排错可设 `cpu` |
| `INDEX_TTS_ASR_ALLOW_CPU_FALLBACK` | `1` | CUDA 显存不足时是否回退 CPU |
| `INDEX_TTS_ASR_BATCH_SIZE_S` | `60` | 动态批总时长（秒） |
| `INDEX_TTS_ASR_MERGE_LENGTH_S` | `15` | VAD 分段合并长度（秒） |
| `INDEX_TTS_ASR_USE_ITN` | `1` | 是否做逆文本归一化（数字/标点更自然） |

### 实测（RTX 3070 8G，2026-11-08）

| 场景 | 数值 |
|---|---|
| 服务启动后（TTS/ASR 都没加载） | `nvidia-smi` 约 **761~870 MiB** |
| 第一次 `/asr`（含加载） | 墙钟 **9.95 s**，`elapsed_ms=9919` |
| ASR 加载后显存 | **1811~2034 MiB**（比空载 +约 1.2 GB） |
| 第二次 `/asr`（模型复用） | **0.09 s**（短音频），长音频约 0.3~0.4 s |
| 空闲阈值 60 s，等待 80 s | `loaded=false`，显存回落到 **982 MiB** |
| 卸载后再请求 | 自动重载，**2.47 s**（funasr 已 import，比首次快） |
| TTS + ASR 同时加载 | 显存 **7231~7274 MiB**，两边推理都能跑通 |
| 模型自带 `example\zh.mp3` | 「开放时间早上9点至下午5点。」（完全正确） |
| 默认音色「测试一下语音合成。」 | 识别完全正确 |

> 首次加载约 10 秒主要是 import funasr + CUDA 初始化 + 读 936 MB 权重；
> 之后重新加载只要 2~3 秒。

---

## 6. 声纹识别（campplus，192 维）

声纹解决的是「**这段语音是谁说的**」：给每个人登记一条（或几条）语音，之后拿新音频来比，
算余弦相似度，够像就认成这个人，不够像就扔进「未知」池并给一个稳定编号。

### 6.1 能干什么 / 不能干什么

**能：**

- 区分不同的人：同一句话、两个不同音色，实测相似度 **-0.073**（几乎不相关）；
- 认已登记的人：同一条音频再算一次自比 **1.000**；登记后拿同一段来认，score **1.000**；
- 给不认识的声音一个**跨请求稳定**的编号（`未知-1`、`未知-2`…），同一段音频反复打
  会一直落在同一个编号上，不会每次新建；
- 只占约 27 MB 权重，和 TTS / ASR 完全独立加载、独立卸载。

**不能：**

- **不做实时多人分离对话**：一次请求只判断「这一整段音频最像谁」，
  不会把一段多人对话切成「A 说了 3 秒、B 说了 2 秒」。要分离得先做说话人分离
  （diarization），那是另一个模型；
- **不承诺方言鲁棒性**：campplus 是普通话 16k 通用模型，重口音、方言、变声、
  极短音频（< 2 秒）、带 BGM 的音频都会掉分；
- 不做活体检测 / 防录音重放，不能当身份认证（security）用；
- 不做跨语种音色一致性判断，也不判断「像不像某个明星」——只对**你登记过的人**负责。

### 6.2 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/speakers` | 已登记的人：名字、样本数、首次/最近更新时间 |
| `GET` | `/speakers/status` | `ready` / `loaded` / `idle_seconds` / `idle_unload_seconds` / `threshold` |
| `GET` | `/speakers/unknown` | 未知聚类：名字、样本数、首次/最近出现时间 |
| `POST` | `/speakers/enroll` | multipart：`file` 音频 + `name` 名字 |
| `POST` | `/speakers/identify` | multipart：`file` 音频 |
| `DELETE` | `/speakers/{name}` | 删掉某个人；不存在返回 `deleted:false`（不报错） |

```powershell
# 1) 登记：同一个名字可以多次登记，样本累加后取平均，越多样本越稳
curl.exe -s -X POST http://127.0.0.1:7880/speakers/enroll `
  -F "file=@E:\LocalAIService\outputs\pick_female-A.wav" -F "name=主人"
# {"ok":true,"name":"主人","samples":1,"created_at":"2026-11-08T13:53:15"}

# 2) 认人
curl.exe -s -X POST http://127.0.0.1:7880/speakers/identify `
  -F "file=@E:\LocalAIService\outputs\pick_female-A.wav"
# {"ok":true,"matches":[{"name":"主人","score":1.0}],"best":"主人",
#  "score":1.0,"threshold":0.65,"known":true}

curl.exe -s -X POST http://127.0.0.1:7880/speakers/identify `
  -F "file=@E:\LocalAIService\outputs\pick_female-B.wav"
# {"ok":true,"matches":[{"name":"主人","score":-0.0729}],"best":null,
#  "score":-0.0729,"threshold":0.65,"known":false}

# 3) 列名单 / 看未知池
curl.exe -s http://127.0.0.1:7880/speakers
curl.exe -s http://127.0.0.1:7880/speakers/unknown
# {"ok":true,"unknown":[{"name":"未知-1","samples":2,
#   "created_at":"2026-11-08T13:53:23","last_seen":"2026-11-08T13:53:23"}],"count":1}

# 4) 删除（中文名字记得 URL 编码；curl 直接传也行）
curl.exe -s -X DELETE http://127.0.0.1:7880/speakers/主人
# {"ok":true,"deleted":true}
```

`/speakers/identify` 返回的 `matches` 最多 5 个候选，按分数从高到低；
`best` 只在 `known=true` 时有名字，否则是 `null`（但 `score` 仍是最高分，方便调阈值）。

### 6.3 阈值含义

分数是 **L2 归一化向量的余弦相似度**，范围 -1~1：

| 分数 | 一般含义 |
|---|---|
| `> 0.7` | 同一个人（不同录音、不同设备通常也有 0.7~0.9） |
| `0.6 ~ 0.7` | 灰色地带，可能是同一个人换了麦克风/环境 |
| `< 0.6` | 基本是不同的人 |

两个环境变量：

| 变量 | 默认 | 说明 |
|---|---:|---|
| `INDEX_TTS_SPEAKER_THRESHOLD` | `0.65` | ≥ 这个分数才判「已知」 |
| `INDEX_TTS_SPEAKER_UNKNOWN_THRESHOLD` | `0.62` | 未知池内部聚类阈值；≥ 这个分数复用旧编号 |

> 未知阈值**故意比已知阈值低一点**：认人宁可保守（不认识的别乱认），
> 但聚类要尽量把同一个陌生人归到一起，别一个人变出十个 `未知-N`。

其余声纹相关环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `INDEX_TTS_SPEAKER_IDLE_UNLOAD` | `900` | 声纹空闲卸载秒数；`0` = 关闭 |
| `INDEX_TTS_SPEAKERS_DIR` | `E:\LocalAIService\speakers` | 声纹库目录 |
| `INDEX_TTS_CAMPPUS_DIR` | `E:\LocalAIService\models\campplus` | 声纹模型目录 |

### 6.4 隐私：只存向量，不存音频

```
E:\LocalAIService\speakers\
├─ 主人.json      # {"name","created_at","updated_at","embeddings":[[192 个浮点数], ...],
│                 #  "source_files":["pick_female-A.wav"]}
└─ unknown.json   # {"next_id":1,"clusters":[{"name":"未知-1","embeddings":[...]}]}
```

- **只存 192 维 float 向量和名字**，`source_files` 只记原始文件名（方便你自己辨认），
  **不保存、不复制原始音频**；
- 上传的音频只落在 `E:\LocalAIService\outputs\speaker_tmp\`，算完向量立刻 `unlink`；
- 全部在本机磁盘，不上传任何地方；
- 想彻底删掉某个人：`DELETE /speakers/{name}`；想清空未知池：把
  `E:\LocalAIService\speakers\unknown.json` 内容改成 `{"next_id": 1, "clusters": []}`。
- 注意：向量本身仍然可以用于「重新识别」，删音频 ≠ 删向量，删人要删 json。

### 6.5 懒加载 / 空闲卸载

- 服务启动只 import `speaker_engine.py`（numpy + 环境变量），**不 import funasr、
  不加载 campplus**；`/health` 里 `speaker.loaded=false`；
- 第一次声纹请求（`/speakers/*` 或 `/asr`）才 `AutoModel(model=<campplus 目录>)`；
- 空闲超过 `INDEX_TTS_SPEAKER_IDLE_UNLOAD` 秒，独立看门狗调用 `unload()`，
  `gc.collect()` + `torch.cuda.empty_cache()`；
- 三把锁：`_ENGINE_INFER_LOCK`（TTS）、`_ASR_INFER_LOCK`、`_SPEAKER_INFER_LOCK`，
  各自串行「推理 / 卸载」，互不干扰。

### 6.6 实测（RTX 3070 8G，2026-11-08）

| 场景 | 数值 |
|---|---|
| 启动后（三个都没加载） | `nvidia-smi` **733~776 MiB**（含 ComfyUI 自占的 ~680 MiB） |
| 只加载 campplus | **966 MiB**（增加约 **233 MiB**，含 torch CUDA context） |
| campplus 加载耗时 | **7.47~7.93 s**（首轮） |
| 同一条音频自比 | **1.000** |
| 两个不同音色（A vs B） | **-0.073** |
| 登记 A 后认 A | **1.000**，`known=true` |
| 认 B | **-0.073**，`known=false` |
| ASR 加载后（SenseVoice 约 1.1 GB） | **1915 MiB** |
| ASR + 声纹都加载 | **1942 MiB**（campplus 在已有 torch context 时只多约 27 MB） |
| TTS + ASR + 声纹全加载 | **7233 MiB** |
| 声纹空闲阈值 60 s，等 85 s | `speaker.loaded=false`，显存 966 → **923 MiB** |
| `?speaker=0` 打 `/asr` | 文字正常返回，`speaker.loaded` 仍是 **false** |

> 卸载后显存只回落 40 多 MB 而不是 233 MB：campplus 权重只有 27 MB，
> 那 200 MB 是 torch 的 CUDA context，只要进程还活着就不会释放。
> 想完全归零就重启服务（重启后回到 733 MiB 左右）。

---

## 7. 加音色

1. 把 `wav / mp3 / flac / ogg / opus / m4a / aac` 放进 `E:\LocalAIService\voices\`；
2. 文件名（去扩展名）就是音色名，例如 `小美.wav` → `"voice": "小美"`；
3. 直接调 `/voices` 或 `/tts`，**不用重启**。

engine 模式直接把文件路径交给 `librosa.load()`，不再上传到 ComfyUI 的 input
目录。参考音频建议 3~10 秒、干净人声、单声道/立体声均可。

---

## 8. 引擎内部做了什么

`E:\LocalAIService\engine\engine.py`：

- **复制** ComfyUI 节点的 `indextts` 源码到 `indextts_vendor\`，不改 Comfy 目录；
- 首次 `synthesize()` 时构造 `IndexTTS2(cfg_path=..., model_dir=..., use_bf16=True, low_vram 自动, use_cuda_kernel=False, use_deepspeed=False)`；
- 之后所有请求复用同一个实例（`_LOAD_LOCK` + `_INFER_LOCK`）；
- 8G 显存：bf16 + 上游 `low_vram` 自动切块 + 每次推理后 `gc.collect()` / `torch.cuda.empty_cache()`；
- 输出用 `soundfile` 写 16bit PCM WAV。

### 对 vendored 代码做的一个补丁

`indextts_vendor\indextts\infer_v2_5.py` 里原本用 `torchaudio.save()` 写 WAV。
torchaudio 2.11 的 `save()` 改成必须依赖 `torchcodec`（又需要系统 ffmpeg），
否则报：

```
ImportError: TorchCodec is required for save_with_torchcodec.
```

所以复制件里加了 `_save_wav_soundfile()`，直接用 `soundfile` 写 WAV。
**ComfyUI 原目录没有改**，只是引擎自己的副本。

---

## 9. 实测数据（RTX 3070 8G，2026-11-08）

| 请求 | 客户端墙钟 | 服务端合成耗时 | 音频时长 | 文件大小 |
|---|---:|---:|---:|---:|
| 首次 `/tts`（含加载模型 28.2s） | 37.12 s | 4.67 s | 2.148 s | 94,764 B |
| 第二次 `/tts`（模型复用） | 3.27 s | 3.33 s | 1.869 s | 82,476 B |
| 再一次同参数 | 8.37 s* | 2.98 s | 1.869 s | 82,476 B |
| `/v1/audio/speech` | 6.32 s | 4.44 s | 2.067 s | 91,180 B |
| 引擎命令行（长句，含加载 27.8s） | 36.67 s | 8.86 s | 3.065 s | 135,212 B |

\* 当时有其它客户端并发请求，`_ENGINE_INFER_LOCK` 会排队，所以墙钟偏大；
服务端单次合成仍是 3 秒左右。

显存（`nvidia-smi`，服务常驻、模型已加载、空闲时）：

```
6664 MiB used / 1354 MiB free / 8192 MiB total
torch: allocated 5.16 GB, reserved 5.26 GB
```

模型加载约 **28 秒**；之后单句 2 秒音频约 **3~5 秒** 合成。
连续两次同参数请求输出字节完全一致（同 seed）。

---

## 10. 为什么弃用 ComfyUI 这条路

ComfyUI 的 venv 里 `transformers` 是 **5.16.1**，而 `index-tts-2.5` 节点
vendored 的 `indextts/gpt/transformers_generation_utils.py` 仍然：

```python
from transformers.generation.beam_constraints import ...
```

这个模块在 transformers 5.x 已被移除，节点执行时直接：

```
ModuleNotFoundError: No module named 'transformers.generation.beam_constraints'
```

可选方案只有两个：

1. 降级 ComfyUI 的 transformers 到 4.x —— 会影响 ComfyUI 里其它所有节点；
2. 把 IndexTTS 拆成独立环境 —— **本方案**，ComfyUI 完全不受影响。

独立引擎里装的是 transformers 4.49.0，vendored 代码 import 正常；
torch 2.11 + cu130 也能在 RTX 3070 上跑 bf16。

---

## 11. 排错

### `/health` 返回 `model_ready:false`

看 `model_detail` 里的 `missing`。主权重在
`E:\LocalAIService\models\TTS\IndexTTS-2.5`，辅助模型在它的 `hf_cache\` 下：
`w2v-bert-2.0`、`campplus_cn_common.bin`、`bigvgan`。缺任何一个都会触发上游
联网下载（本机要求不重下），所以先补齐文件。

### 首次请求很慢 / 超时

正常：加载约 28 秒（3.1GB GPT + 2.3GB w2v-bert + s2mel/codec/bigvgan）。
之后复用，单句 3~5 秒。不要把 `INDEX_TTS_TIMEOUT` 设得太小。

### CUDA out of memory / 第二次请求进程消失

8G 卡同时跑 ComfyUI 图像/视频模型时容易 OOM。处理：

1. 先释放 ComfyUI 占用的显存（在 ComfyUI 里卸载模型，或停掉正在跑的任务）；
2. 确认 `INDEX_TTS_BF16=1`（默认）；
3. 重启服务，让引擎重新加载；
4. 长文本会由上游 `low_vram` 自动按标点切块，不要一次塞几千字。

### `emotion` 文本报「没有加载 Qwen 情感模型」

engine 默认 `use_qwen_emo=False`（省 1.2GB 显存）。要用情感文本：

```powershell
$env:INDEX_TTS_QWEN_EMO = '1'
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Restart -Background
```

8 维情感向量不需要 Qwen，直接可用。

### `/asr` 返回 503 / 加载失败

先看 `service.err.log` 和 `/asr/status` 的 `ready`：

1. `ready=false`：模型文件不全，检查 `E:\LocalAIService\models\SenseVoiceSmall`
   下是否有 `model.pt`、`config.yaml`、`am.mvn`、`tokens.json`、
   `chn_jpn_yue_eng_ko_spectok.bpe.model`；
2. CUDA OOM：`asr_engine.py` 默认会回退 CPU（`INDEX_TTS_ASR_ALLOW_CPU_FALLBACK=1`），
   识别会慢一些但能用；也可以先等 TTS 空闲卸载，或临时把
   `INDEX_TTS_ASR_DEVICE=cpu`；
3. funasr 没装好：`python -c "import funasr; print(funasr.__version__)"`。

### 声纹接口返回 503 / 认不出人

1. `GET /speakers/status` 里 `ready=false`：检查
   `E:\LocalAIService\models\campplus\` 下是否有 `campplus_cn_common.bin` 和 `config.yaml`；
2. `load_error` 非空：把 `service.err.log` 里的原始报错发出来（funasr 版本和
   campplus 结构对不上时，通常是 `AutoModel` 加载阶段报错）；
3. 认不出已登记的人：先用**登记时那段音频**打 `/speakers/identify`，正常应该 ~1.0；
   如果这都低，说明登记时音频是静音/太短/格式不支持；
4. 不同人也能认成同一个人：说明阈值太低，把 `INDEX_TTS_SPEAKER_THRESHOLD`
   调高到 `0.7` 再试；
5. `DELETE /speakers/名字` 返回 `deleted:false`：名字里含 `\ / : * ? " < > |` 会被
   过滤，或者这个名字本来就不存在。

### 识别文字有错字

SenseVoiceSmall 是 2 亿参数左右的 CTC 小模型，对 TTS 合成音、口音、轻声
（例如「顺耳」听成「圣儿」、「卸载」听成「歇子」）会有错字，这是模型能力
边界，不是接口问题。模型自带的 `example\zh.mp3` 可以识别成
「开放时间早上9点至下午5点。」，说明链路正常。对准确率要求高时，建议换更大
的 ASR 模型，或在业务侧做关键词纠正。

### funasr 安装时的 protobuf 冲突警告

`funasr 1.4.24` 会把 `protobuf` 升到 7.x，pip 会提示
`descript-audiotools 0.7.2 requires protobuf<3.20`。当前 IndexTTS 推理链路
不 import `descript_audiotools`，TTS 回归已验证正常；如果以后某个节点因此
报错，再把 protobuf 降回 3.19.6 并重装 funasr（`--no-deps`）。

### 端口 7880 被占用

```powershell
Get-NetTCPConnection -LocalPort 7880 -State Listen
```

用启动脚本的 `-Restart`，或手动停掉占用进程。不要改端口（客户端都连 7880）。

### 中文乱码 / 合成成英文

提交 JSON 用 UTF-8（无 BOM），并带 `charset=utf-8`。PowerShell 5.1 直接
`-Body $body` 可能按本地编码发送，按上面的 `UTF8.GetBytes()` / 写文件方式做。

### 想确认引擎本身是否正常（不经过 HTTP）

```powershell
E:\LocalAIService\engine\.venv\Scripts\python.exe E:\LocalAIService\engine\engine.py `
  --text "主人，我是大肥鱼，这条声音是我自己挑的。" `
  --voice E:\LocalAIService\voices\default.mp3 `
  --out E:\LocalAIService\outputs\engine_selftest.wav
```

连续两次复用测试：

```powershell
E:\LocalAIService\engine\.venv\Scripts\python.exe E:\LocalAIService\engine\try_two.py
```

### 看日志

```powershell
Get-Content E:\LocalAIService\logs\service.err.log -Tail 80 -Encoding UTF8
Get-Content E:\LocalAIService\logs\engine_install.log -Tail 80 -Encoding UTF8
```

---

## 12. 重新安装引擎环境（换机/重建时）

```powershell
$py = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe'
& $py -m venv E:\LocalAIService\engine\.venv
$pip = 'E:\LocalAIService\engine\.venv\Scripts\python.exe'

# 1) torch + torchaudio（cu130，必须成对）
& $pip install torch==2.11.0+cu130 torchaudio==2.11.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130

# 2) transformers 4.49 + 其余依赖
& $pip install "transformers==4.49.0" tokenizers accelerate safetensors huggingface_hub `
  numpy librosa soundfile scipy omegaconf einops sentencepiece tiktoken openai-whisper `
  modelscope munch pandas matplotlib numba json5 textstat jieba cn2an g2p-en ffmpeg-python `
  descript-audiotools PyYAML wetext fugashi unidic-lite

# 3) Web 服务依赖
& $pip install fastapi uvicorn httpx python-multipart

# 4) ASR：funasr + SenseVoiceSmall（Python 3.13 / torch 2.11 实测可用）
& $pip install funasr==1.4.24
# 会顺带装 kaldiio / hydra-core / omegaconf / RapidFuzz / torch_complex 等；
# 日志见 E:\LocalAIService\logs\asr_install.log

# 5) 从 ModelScope 下模型（HF 直连不通，用 modelscope 快）
& $pip -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir=r'E:\LocalAIService\models\SenseVoiceSmall')"

# 6) 检查
& $pip -c "import torch, torchaudio, transformers; print(torch.__version__, torchaudio.__version__, transformers.__version__)"
& $pip -c "import funasr; print('funasr', funasr.__version__)"
```

> `fugashi` + `unidic-lite` 不在节点 requirements.txt 的必装列表里，但
> `IndexTTS2.__init__` 会构造 `JapaneseG2PProcessor`，没有它们会直接
> `ImportError`（即使只合成中文）。所以必须装。

---

## 13. 已知限制

- 服务只监听 `127.0.0.1`，无鉴权，不要直接暴露公网。
- 只支持 `response_format=wav`，不做流式返回。
- engine 模式同一时刻只处理一个合成请求（GPU 串行，排队等待）。
- ASR 也串行处理（`_ASR_INFER_LOCK`），但和历史 TTS 请求可以同时在显存里；
  两边都加载时 8G 卡余量只有 0.6~0.7 GB，长文本 TTS 仍可能 OOM。
- SenseVoiceSmall 对合成音/口音会有错字；情感/事件标签是模型判断，不一定
  符合人的直觉（例如平静语气被判成 `ANGRY`）。
- 声纹只回答「这一整段音频最像谁」，**不做多人分离/说话人日志**；一段多人对话
  只会得到一个「最像的人」，不会切成 A 说了几秒、B 说了几秒。
- 声纹对**方言/重口音/变声/极短音频（< 2 秒）/带背景音乐**不承诺效果；
  campplus 是普通话 16k 通用模型，遇到这些情况分数会明显下降。
- 声纹不是身份认证：没有活体检测，录音重放也能骗过，别拿它当门锁。
- 声纹库里的 192 维向量本身仍可用来识别，删了原始音频不等于匿名；
  不想留就把 `E:\LocalAIService\speakers\*.json` 一起删掉。
- 三个模型全加载时 8G 卡会到 **7.2 GB** 左右（TTS 5.6 + ASR 1.1 + 声纹 27 MB +
  CUDA context），此时再让 ComfyUI 跑大模型容易 OOM，靠空闲卸载让位。
- Qwen 情感文本默认不加载；需要时用 `INDEX_TTS_QWEN_EMO=1`，并注意 8G 显存。
- 每次成功合成都会在 `E:\LocalAIService\outputs` 留档，不会自动清理。
- ComfyUI 旧链路仍保留代码，但当前 transformers 5.16.1 下不可用；不要降级
  ComfyUI 的依赖。
- **ffmpeg 路径写死在本机**：默认 `C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\`。
  换机 / 挪动 ffmpeg 时设 `INDEX_TTS_FFMPEG`（和可选的 `INDEX_TTS_FFPROBE`），
  否则 `/media/*` 与「非 wav 转写」会返回 503，`/health.media.ffmpeg` 会显示 `missing`。
- 媒体接口的上传上限是 200MB（`MEDIA_MAX_BYTES`），大视频请先自己裁一段。
- 抽帧一次最多 200 张（`MAX_FRAMES`），按 fps 抽时会先估算，超了直接报错。
- ffmpeg 单条命令默认 300 秒超时（`INDEX_TTS_FFMPEG_TIMEOUT` 可调）。
- 项目改名后 venv 实测可直接用（`python.exe` 能 import torch/funasr/fastapi）；
  但 `Scripts\*.exe`（pip.exe、uvicorn.exe 等）内部记录的绝对路径仍是旧路径，
  如果它们失效，统一改用 `python.exe -m pip ...` / `python.exe -m uvicorn ...`。
- `E:\IndexTTS` 是 junction，不是副本；**不要**对它做「删除目录」类操作，
  要删只能 `rmdir E:\IndexTTS`（只删链接，不动目标目录内容）。

---

## 14. ffmpeg 媒体接口（新增）

实现位置：`E:\LocalAIService\engine\ffmpeg_tool.py`（纯 stdlib + 外部 ffmpeg，
不 import torch，所以调用它不会占显存、也不会触发模型加载）。

### 14.1 ffmpeg 定位

按顺序找：

1. 环境变量 `INDEX_TTS_FFMPEG`（ffprobe 用 `INDEX_TTS_FFPROBE`，不设时自动找
   同目录的 `ffprobe.exe`）；
2. 本机默认 `C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe`；
3. PATH 里的 `ffmpeg` / `ffprobe`。

都找不到时，`/health` 返回：

```json
"media": {
  "ffmpeg": "missing",
  "path": null,
  "expected": "C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\ffmpeg.exe",
  "detail": "找不到 ffmpeg。请设置环境变量 INDEX_TTS_FFMPEG，或确认默认路径存在：..."
}
```

正常时：

```json
"media": {
  "ffmpeg": "ok",
  "path": "C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\ffmpeg.exe",
  "ffprobe": "ok",
  "ffprobe_path": "C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\ffprobe.exe"
}
```

### 14.2 接口清单

| 方法 | 路径 | 入参 | 返回 |
|---|---|---|---|
| POST | `/media/probe` | `file` | `{ok, duration, size, bit_rate, format{...}, audio{...}, video{...}}` |
| POST | `/media/to_wav` | `file` | `audio/wav` 二进制（16k / 单声道 / PCM16） |
| POST | `/media/frames` | `file` + `count`/`fps`/`times`/`width` | `{ok, count, dir, frames[], preview_b64, preview_mime}` |
| POST | `/media/audio` | `file` + `format=wav\|mp3` | `audio/wav` 或 `audio/mpeg` 二进制 |
| POST | `/asr` | `file`（音频**或视频**） | 识别结果（非 wav 会先自动转 16k） |

`/media/frames` 的参数：

- `count=3`：整段均匀抽 3 张（取每段中点，避开片头黑场）；
- `times=1,2.5,4`：在这些秒数各抽 1 张；
- `fps=0.5`：每 2 秒抽 1 张；
- 三者只能给一个；都不给时抽中间 1 张；
- `width=640`：输出宽度，高度按比例、自动取偶数；
- 文件落在 `outputs\frames\<时间戳>_<随机6位>\`，命名 `frame_0001_12.500s.jpg`；
- 返回里的 `preview_b64` 是**第一帧**的 base64（jpg/png 原图，最多 1 张），
  方便调用方不读文件也能直接看。

`/asr` 现在接受的格式：`wav / mp3 / flac / ogg / opus / m4a / aac / wma / amr /
aiff / ape` 以及 `mp4 / mkv / mov / avi / flv / webm / m4v / ts / mpg / mpeg /
wmv / 3gp / rmvb / vob`。wav 直接识别，其它格式先用 ffmpeg 转 16k 单声道 wav
（临时文件处理完立即删除）。返回里多了两个字段：`source_format`、`converted`。

### 14.3 curl 示例

```bash
# 1) 探测：时长 / 容器 / 音频参数 / 视频分辨率帧率
curl -s -F "file=@D:\test.mp4" http://127.0.0.1:7880/media/probe

# 2) 转 16k 单声道 wav（ASR 标准输入）
curl -s -o test16k.wav -F "file=@D:\test.mp4" http://127.0.0.1:7880/media/to_wav

# 3) 均匀抽 3 张、宽 640
curl -s -F "file=@D:\test.mp4" -F "count=3" -F "width=640" \
  http://127.0.0.1:7880/media/frames

# 4) 按时间点抽 2 张
curl -s -F "file=@D:\test.mp4" -F "times=1,3.5" http://127.0.0.1:7880/media/frames

# 5) 每 2 秒抽 1 张
curl -s -F "file=@D:\test.mp4" -F "fps=0.5" http://127.0.0.1:7880/media/frames

# 6) 只抽音轨成 mp3（保留原采样率/声道）
curl -s -o audio.mp3 -F "file=@D:\test.mp4" -F "format=mp3" \
  http://127.0.0.1:7880/media/audio

# 7) 视频直接转写（不加声纹，省一次 campplus 加载）
curl -s -F "file=@D:\test.mp4" "http://127.0.0.1:7880/asr?speaker=0"
```

PowerShell 里用 `curl.exe`（不是 `curl` 别名）：

```powershell
curl.exe -s -F "file=@D:\test.mp4" -F "count=3" http://127.0.0.1:7880/media/frames
# 结果里的 preview_b64 可以直接落成图片：
$r = curl.exe -s -F "file=@D:\test.mp4" -F "count=1" http://127.0.0.1:7880/media/frames | ConvertFrom-Json
[IO.File]::WriteAllBytes("D:\preview.jpg", [Convert]::FromBase64String($r.preview_b64))
```

错误码：文件类型不支持/参数不合法 → `400`；ffmpeg 找不到 → `503`；
ffmpeg 超时 → `504`；上传超过 200MB → `413`。失败时 `detail` 里带 ffmpeg
stderr 的最后 20 行，方便直接定位（比如「moov atom not found」说明文件损坏）。

### 14.4 本机实测（2026-11-08，测试视频 640x360 / 25fps / 5 秒 / 带语音）

| 操作 | 结果 |
|---|---|
| `/media/probe` | `duration=5.0`，`format=mov,mp4...`，视频 `h264 640x360 25.0fps 125 帧`，音频 `aac 22050Hz 2ch 119.6kbps` |
| `/media/frames count=3 width=320` | 3 张（0.833s / 2.500s / 4.167s），320x180，服务端耗时 **243ms** |
| `/media/frames times=1,3.5` | 2 张（1.000s / 3.500s），640x360 |
| `/media/frames fps=2` | 10 张（0.000s ~ 4.500s，间隔 0.5s） |
| `/media/to_wav` | `16000Hz / 1 声道 / PCM_16 / 5.016s / 160576 字节`，客户端总耗时 **134ms** |
| `/media/audio format=mp3` | 44357 字节，mp3 22050Hz 2 声道；`format=wav` 为 pcm_s16le 22050Hz 2 声道 |
| `/asr`（直接传 mp4） | 成功：`converted=true`，`duration=5.016`，`elapsed_ms≈10980`（含首次加载 SenseVoice），识别出日语文本 |

---

## 15. 改名与迁移（IndexTTS -> LocalAIService）

### 15.1 现在的状态

| 项目 | 值 |
|---|---|
| 真实目录 | `E:\LocalAIService` |
| 兼容别名 | `E:\IndexTTS` → 目录 junction → `E:\LocalAIService` |
| 启动脚本 | `E:\LocalAIService\start_ai_service.ps1`（旧名 `start_service_engine.ps1` 是转交壳） |
| 计划任务 | 名字仍是 `IndexTTS-Service`（为了不破坏按名字查询的旧调用），指向新脚本 |
| venv | `E:\LocalAIService\engine\.venv`（随目录移动，实测可直接 import torch/funasr/fastapi） |
| 服务 | `127.0.0.1:7880`，启动时 0 模型 |

代码里所有默认路径都改成「从 `service.py` / `engine.py` 自身位置推服务根目录」，
不再写死盘符，所以 `E:\LocalAIService` 和 `E:\IndexTTS`（junction）两种启动方式
都会解析到同一个真实目录；`/health` 里的 `model_dir` / `engine_dir` 显示的是
`E:\LocalAIService\...`。

### 15.2 当初怎么改的（留档，换机可照做）

```powershell
# 0) 先停服务（只杀监听 7880 的 python）
Get-NetTCPConnection -LocalPort 7880 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# 1) 同盘改名（秒级；如果提示“访问被拒绝”，先用资源监视器/句柄工具
#    确认没有进程把它当工作目录，过几秒重试）
Rename-Item E:\IndexTTS E:\LocalAIService

# 2) 建兼容 junction（旧路径继续可用；/J 是目录联接，不占空间）
cmd /c mklink /J "E:\IndexTTS" "E:\LocalAIService"

# 3) 验证 junction
(Get-Item E:\IndexTTS).LinkType     # Junction
(Get-Item E:\IndexTTS).Target       # E:\LocalAIService

# 4) 验证 venv（移动后应仍然可用）
E:\LocalAIService\engine\.venv\Scripts\python.exe -c "import torch, funasr, fastapi; print('ok')"

# 5) 更新计划任务（本机保留任务名，只换脚本路径；描述已改成中文说明）
#    可用「任务计划程序」图形界面，或 schtasks /change /tn IndexTTS-Service /tr "..."
```

### 15.3 怎么改回去（回滚）

```powershell
# 1) 停服务
Get-NetTCPConnection -LocalPort 7880 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# 2) 先删 junction —— rmdir 只删链接，不会删 E:\LocalAIService 里的内容
cmd /c rmdir E:\IndexTTS

# 3) 目录改回原名
Rename-Item E:\LocalAIService E:\IndexTTS

# 4) 计划任务指回旧脚本（旧脚本会转交给 start_ai_service.ps1，仍在原目录）
schtasks /change /tn "IndexTTS-Service" /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"E:\IndexTTS\start_service_engine.ps1\""
```

> 回滚后代码依然能用：所有路径都是相对服务根目录推导的，`E:\IndexTTS`
> 重新变成真实目录也能正常解析。如果曾经设置过 `INDEX_TTS_HOME`，
> 记得把它删掉或改成新的根目录。

### 15.4 这次改名动过的文件

| 文件 | 改动 |
|---|---|
| `E:\LocalAIService\svc\service.py` | 路径常量改为 `SERVICE_ROOT` 推导；新增 `/media/*`；`/asr` 支持视频；`/health` 加 `media`；FastAPI 标题/描述改名 |
| `E:\LocalAIService\engine\ffmpeg_tool.py` | **新增**：probe / to_wav16k / extract_frames / extract_audio / status |
| `E:\LocalAIService\engine\engine.py` | `SERVICE_ROOT` 推导 + 模型/命令行默认路径 |
| `E:\LocalAIService\engine\asr_engine.py` | 同上 |
| `E:\LocalAIService\engine\speaker_engine.py` | 同上（campplus / speakers 目录） |
| `E:\LocalAIService\engine\try_two.py` | 排错脚本路径改为相对推导 |
| `E:\LocalAIService\start_ai_service.ps1` | **新增**：正式启动脚本（UTF-8 带 BOM） |
| `E:\LocalAIService\start_service_engine.ps1` | 改成转交壳（参数不变，UTF-8 带 BOM） |
| `E:\LocalAIService\engine\start_engine_service.ps1` | 路径改为 `$PSScriptRoot` 推导 |
| `E:\LocalAIService\README.md` / `svc\README.md` | 改名说明、迁移/回滚、媒体接口文档、实测数据 |
| 计划任务 `IndexTTS-Service` | 动作改为 `start_ai_service.ps1`，描述更新；触发条件（登录时）不变 |

---

## 16. 人声分离（Demucs）

### 16.1 能做什么

用 Meta 的 **Demucs** 把一段音频拆成人声和伴奏，典型用途：

- 从带 BGM 的视频/歌曲里抽 **纯人声**，做 TTS 参考音色；
- 抽伴奏做翻唱、二创；
- 给 ASR 做前置降噪（人声轨通常比原轨识别更准）。

本机装的是 `demucs==4.1.0`（Python 3.13 / torch 2.11 实测可用），走的是
`demucs.api.Separator`，不是命令行子进程。代码在
`E:\LocalAIService\engine\separator_engine.py`，接口在
`E:\LocalAIService\svc\service.py`。

| 模型 | 输出 stem | 说明 |
|---|---|---|
| `htdemucs`（默认） | vocals / drums / bass / other | 4 轨，速度和质量最均衡；本机已备好权重 |
| `htdemucs_6s` | 上面 4 轨 + guitar / piano | 6 轨；本机已备好权重 |
| `htdemucs_ft` | vocals / drums / bass / other | 4 个模型 bag，质量略好、耗时约 4 倍；**权重未下全**，要用先补文件 |

### 16.2 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/separate` | multipart：`file`、`model`、`stems` |
| POST | `/api/v1/separate` | 同上，给别的软件用的别名 |
| GET | `/separate/status` | `{ready, loaded, model, idle_seconds, idle_unload_seconds, ...}` |

`/health` 里也新增了一块：

```json
"separate": {"ready": true, "loaded": false, "idle_unload_seconds": 900, "model": "htdemucs"}
```

参数：

- `file`：音频或视频都行（内部先用 ffmpeg 转 44.1kHz / 立体声 wav）；
- `model`：`htdemucs`（默认）/ `htdemucs_6s` / `htdemucs_ft`；
- `stems`：逗号分隔，默认 `vocals,drums,bass,other`；
  6s 模型还可以加 `guitar,piano`，只要其中一条时传 `stems=vocals` 即可。

```bash
# 从带背景音的视频里抽人声（最常用）
curl -X POST http://127.0.0.1:7880/separate ^
  -F "file=@D:\dsh_ws\大肥鱼音源.mp4" ^
  -F "model=htdemucs" ^
  -F "stems=vocals,drums,bass,other"

# 只要人声，省一点落盘
curl -X POST http://127.0.0.1:7880/separate ^
  -F "file=@D:\dsh_ws\_voice_src\source_44k_stereo.wav" ^
  -F "stems=vocals"

# 6 轨模型
curl -X POST http://127.0.0.1:7880/separate ^
  -F "file=@D:\music\song.mp3" ^
  -F "model=htdemucs_6s" ^
  -F "stems=vocals,drums,bass,other,guitar,piano"

# 看状态
curl http://127.0.0.1:7880/separate/status
```

返回示例（实测 9.752 秒素材）：

```json
{
  "ok": true,
  "model": "htdemucs",
  "sample_rate": 44100,
  "duration": 9.752,
  "elapsed": 3.38,
  "separation_elapsed": 0.74,
  "stems": {
    "vocals": "E:\\LocalAIService\\outputs\\separate\\20261108_143817_37f04d\\vocals.wav",
    "drums":  "E:\\LocalAIService\\outputs\\separate\\20261108_143817_37f04d\\drums.wav",
    "bass":   "E:\\LocalAIService\\outputs\\separate\\20261108_143817_37f04d\\bass.wav",
    "other":  "E:\\LocalAIService\\outputs\\separate\\20261108_143817_37f04d\\other.wav"
  },
  "dir": "E:\\LocalAIService\\outputs\\separate\\20261108_143817_37f04d",
  "loaded": true,
  "released_models": []
}
```

- `elapsed`：整个 HTTP 请求耗时（含转 44.1k、首次加载模型）；
- `separation_elapsed`：纯 Demucs 推理耗时；
- `released_models`：为腾显存被卸载的模型（`tts` / `asr` / `speaker`），空数组表示没卸。

输出目录固定为：

```
E:\LocalAIService\outputs\separate\<yyyyMMdd_HHmmss>_<6位随机>\<stem>.wav
```

每个 stem 都是 **44.1kHz / 立体声 / PCM16 wav**，文件名固定，下游拼路径即可。

### 16.3 模型权重在哪（离线、不散落用户目录）

```
E:\LocalAIService\models\torch\hub\checkpoints\
├── 955717e8-8726e21a.th      84,141,911 字节   htdemucs 主模型
├── 5c90dfd2-34c22ccb.th      54,996,327 字节   htdemucs_6s
├── htdemucs.yaml
├── htdemucs_6s.yaml
└── htdemucs_ft.yaml          （只有 bag 描述，4 个 .th 还没下）
```

- `TORCH_HOME` 默认指到 `E:\LocalAIService\models\torch`，启动脚本里也显式设置；
- Demucs 用本地 `repo` 加载，**不会联网**，也不会写
  `C:\Users\<用户>\.cache\torch` 或 HuggingFace 缓存；
- 换机/重装时按 `htdemucs.yaml` 里的签名从
  `https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/` 下对应的
  `.th` 放进这个目录即可；
- 安装与下载尝试顺序记录在 `E:\LocalAIService\logs\demucs_install.log`。

> 第一次装的时候先装了 `demucs==4.0.1`，发现它没有 `demucs.api`；
> 为了用官方 Python API（而不是每次起子进程）升级到了 `4.1.0`，
> 依赖都能在 Python 3.13 下装，日志里有完整记录。

### 16.4 懒加载 / 空闲卸载

- 服务启动、`import separator_engine` 都**不加载** torch / Demucs；
- 第一次 `/separate` 才把权重读进显存（本机约 2.6 秒）；
- 空闲超过 `INDEX_TTS_SEPARATE_IDLE_UNLOAD` 秒（默认 900，0 = 关闭）自动
  `unload()`，显存还给别的程序；下次请求再自动加载；
- 有自己的 `_SEPARATE_INFER_LOCK` 和看门狗，和 TTS / ASR / 声纹互不影响；
- 同一时刻只保留一个 Demucs 模型：请求 6s / ft 时会先卸掉当前模型；
- **（已归档，旧逻辑）** 以前 `/separate` 单独检查「空闲 < 2800MB 就卸掉
  TTS / ASR / 声纹」，只保护分离这一条路，其它能力加载前没有显存检查；
  这套逻辑已删除，统一改由第 17 章「显存管理」的 `vram_manager` 处理。
  下面的旧实测数字只作历史参考，当前行为以第 17 章为准。

启动脚本新增参数：

```powershell
# 默认 900 秒
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1

# 调试：分离 60 秒就卸载
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SeparateIdleSeconds 60

# 关闭分离空闲卸载
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SeparateIdleSeconds 0
```

### 16.5 本机实测（RTX 3070 8G，2026-11-08）

素材：`D:\dsh_ws\_voice_src\source_44k_stereo.wav`（44.1kHz 立体声，9.752 秒）。

| 项目 | 结果 |
|---|---|
| 首次 `/separate`（含加载） | 总耗时 **3.38s**，其中纯分离 **0.74s** |
| 第二次 `/api/v1/separate`（只要 vocals） | 总耗时 **0.48s**，纯分离 **0.35s** |
| 显存 | 加载后整卡 **1670MiB / 8192MiB**（服务基线约 800MiB，Demucs 增量约 0.9GB） |
| 单次推理 torch 峰值分配 | 约 **552MiB** |
| 输出 | 4 个 stem 各 9.752s，44.1kHz 立体声 |

**ASR 粗验证**（同一段素材，`?speaker=0`）：

| 输入 | 识别文本 | 事件标签 |
|---|---|---|
| 原始带 BGM | 小明就叫饭团，白白胖胖，软乎乎的，跟白米饭一样，看着就让人开心好养活，而且饿了的时候念叨一下名字，就当解馋了嘛。 | `BGM` |
| 分离后 vocals | 小明就叫饭团，白白胖胖，软乎乎的，跟白米饭一样，看着就让人开心好养活。而且饿了的时候念叨一下名字，就当解馋了嘛。 | `Speech` |

文字基本一致，但人声轨的标点和断句更干净，事件标签也从 `BGM` 变成了
`Speech`，说明背景音确实被压下去了。

### 16.6 用分离结果做音色（本例 `dafeiyu`）

```powershell
# 1) 抽人声
curl -X POST http://127.0.0.1:7880/separate ^
  -F "file=@D:\dsh_ws\大肥鱼音源.mp4" -F "stems=vocals"

# 2) 用 ffmpeg / soundfile 裁头尾，转 44.1k 单声道，放到 voices\dafeiyu.wav
#    文件名必须是 ASCII；音色名就是文件名去扩展名。
```

本次实际参数：

| 项目 | 值 |
|---|---|
| 文件 | `E:\LocalAIService\voices\dafeiyu.wav` |
| 时长 | **9.400s**（从 vocals 的 0.30s 到 9.70s，去掉头尾静音） |
| 采样率 / 声道 | 44100Hz / 单声道 / PCM16 |
| F0 中位数（librosa.pyin） | **339.41Hz**（P10 268.77 / P90 464.72，女声） |
| RMS | **0.1094**（峰值 0.6029） |

```powershell
# 3) 用这个音色合成
curl -X POST http://127.0.0.1:7880/tts ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"主人，我是大肥鱼。以后我就用这个声音跟你说话了——你听听，像不像你想要的那个样子。\",\"voice\":\"dafeiyu\",\"speed\":1.0}" ^
  --output E:\LocalAIService\outputs\sample_dafeiyu.wav
```

本次样音：`E:\LocalAIService\outputs\sample_dafeiyu.wav`，8.49 秒、
22050Hz 单声道、374,432 字节；首次合成（含加载 TTS）39.87 秒。

### 16.7 许可提醒

- **Demucs 代码**是 MIT，可自由使用/修改；
- **预训练权重**（`htdemucs` 系列）受训练数据（MUSDB18-HQ 等）影响，
  一般视为**非商用 / 研究用途**。要商用请先自行确认 Meta 的模型许可，
  或换成许可明确的分离模型（如 UVR / MDX 系列里标注可商用的权重）；
- 本服务只在本机跑，不上传音频；输出文件在 `outputs\separate\`，
  要不要保留、保留多久由调用方决定。

### 16.8 常见问题

**`/separate` 返回 503：Demucs 权重不完整**

看 `/separate/status` 里的 `missing` 列表，把缺的 `.th` 按 16.3 放到
`models\torch\hub\checkpoints\`。`htdemucs_ft` 需要 4 个 `.th`，少一个都会报这个。

**首次请求慢**

首次要加载权重（本机约 2.6s），之后 0.5~1s 一条 10 秒素材。不是联网卡住。

**CUDA out of memory**

- 把 `INDEX_TTS_SEPARATE_SEGMENT` 调小（如 5）；
- 确认没有别的模型同时占卡（TTS 常驻约 5~6GB 时，Demucs 可能挤不进去）；
- 临时用 `INDEX_TTS_SEPARATE_DEVICE=cpu`，会慢很多但能跑。

**`/separate` 400：不支持的 model / stem**

`model` 只能是 `htdemucs` / `htdemucs_ft` / `htdemucs_6s`；
`guitar` / `piano` 只有 `htdemucs_6s` 有。

**输出太大**

每个 stem 都是 44.1k 立体声 PCM16，10 秒约 1.7MB。只要人声就传
`stems=vocals`，或定期清理 `outputs\separate\`。

**想确认引擎本身是否正常（不经过 HTTP）**

```powershell
E:\LocalAIService\engine\.venv\Scripts\python.exe E:\LocalAIService\engine\separator_engine.py --status
E:\LocalAIService\engine\.venv\Scripts\python.exe E:\LocalAIService\engine\separator_engine.py ^
  --audio D:\dsh_ws\_voice_src\source_44k_stereo.wav ^
  --out E:\LocalAIService\outputs\separate_probe --stems vocals
```

---

## 17. 显存管理（统一按需腾显存）

> 2026-11-08 新增。之前只有 `/separate` 里有一处「显存不够就卸 TTS/ASR/声纹」
> 的零散判断，TTS / ASR / 声纹自己加载前**没有**腾显存逻辑。现在四个能力统一
> 走 `src\localai\core\vram.py`（旧的 `engine\vram_manager.py` 现在是转发壳），
> `/separate` 里那段旧逻辑已删除（见 17.8 归档）。

### 17.1 为什么 8G 卡必须这样

RTX 3070 只有 8GB 显存，Windows 桌面 / 浏览器 / QQ 等还会先占掉
0.6~1.8GB。四个能力的量级差别很大：

| 能力 | 典型占用 | 说明 |
|---|---:|---|
| TTS IndexTTS-2.5 | **约 5.2~6.4GB** | 最大头；加载后常驻，推理时还有临时峰值 |
| ASR SenseVoice | **约 0.94GB** | 第二次加载后实测 940MB |
| 声纹 campplus | **约 22~240MB** | 在已有 CUDA context 上增量很小 |
| 人声分离 Demucs | **约 0.8~0.9GB** | 权重加载增量小，推理峰值和缓存更占地方 |

TTS 一个就能吃掉大半张卡。如果四个能力各自「想加载就加载」，8G 卡在
Windows WDDM 下不一定会抛出可捕获的 `CUDA out of memory`，更常见的是
驱动层反复换页、整个桌面卡住，甚至进程直接消失。所以策略不是「等 OOM
再处理」，而是**在加载前就确认放得下；放不下先按 LRU 卸别人**。

服务启动仍然 **0 模型**：管家只登记卸载函数和推理锁，不预热、不 import torch；
第一次请求才会加载。四个原有的空闲看门狗继续保留，空闲 900 秒各自卸载。

### 17.2 接口与文件

新增文件：

```
E:\LocalAIService\src\localai\core\vram.py    # 管家实现（engine\vram_manager.py 是转发壳）
E:\LocalAIService\logs\vram_estimates.json     # 估算校准结果（首次加载后生成）
```

#### GET /vram

返回当前显存、各能力 loaded / last_used / estimate，以及管家实际在用的估算值。
示例（TTS 已加载、Demucs 已卸载）：

```json
{
  "free_mb": 1263,
  "total_mb": 8192,
  "used_mb": 6928,
  "torch_allocated_mb": 5470,
  "torch_reserved_mb": 5654,
  "margin_mb": 512,
  "estimates": {"tts": 6200, "asr": 900, "speaker": 123, "separate": 800},
  "models": [
    {"name": "tts", "loaded": true, "last_used": "2026-11-08T15:02:16+08:00",
     "idle_seconds": 0.1, "estimate_mb": 6200, "measured_mb": 5176,
     "samples": 8, "active": 0, "loader": "IndexTTS-2.5（TTS，常驻）"},
    {"name": "asr", "loaded": false, "last_used": null, "idle_seconds": null,
     "estimate_mb": 900, "measured_mb": 818, "samples": 3, "active": 0,
     "loader": "SenseVoiceSmall（ASR）"},
    {"name": "speaker", "loaded": false, "last_used": null, "idle_seconds": null,
     "estimate_mb": 123, "measured_mb": 44, "samples": 4, "active": 0,
     "loader": "campplus（声纹，在已有 CUDA context 上增量很小）"},
    {"name": "separate", "loaded": false, "last_used": "2026-11-08T15:01:54+08:00",
     "idle_seconds": null, "estimate_mb": 800, "measured_mb": 182,
     "samples": 4, "active": 0, "loader": "Demucs htdemucs（人声分离）"}
  ]
}
```

- `last_used`：最近一次请求结束时间；`idle_seconds`：已空闲秒数；
- `active`：正在用这个能力的请求数（>0 时管家不会卸它）；
- `measured_mb`：最近一次「加载前后空闲差」；`samples`：校准样本数；
- `free_mb/total_mb`：torch 已初始化时用 `mem_get_info()`，否则用 nvidia-smi。

#### POST /vram/unload

```powershell
# 手动卸指定能力
curl -X POST http://127.0.0.1:7880/vram/unload ^
  -H "Content-Type: application/json" ^
  -d "{\"models\":[\"tts\",\"asr\"],\"all\":false}"

# 全部卸载（恢复到 0 模型）
curl -X POST http://127.0.0.1:7880/vram/unload ^
  -H "Content-Type: application/json" ^
  -d "{\"all\":true}"
```

返回：

```json
{"ok": true, "released": ["asr", "speaker"], "free_mb": 6997, "total_mb": 8192}
```

卸载会先等对应推理锁；正在推理的请求跑完后才卸，不会卸到一半。响应里的
`released` 是**真正卸掉**的模型（没加载的不会列进去）。

#### GET /health 里的 vram

`/health` 新增一个轻量字段，1 秒缓存、不主动初始化 CUDA：

```json
"vram": {"free_mb": 1283, "total_mb": 8192}
```

### 17.3 管家策略（LRU + margin + 校准）

#### 加载流程

每个能力加载前都走同一套：

```
拿到该能力的推理锁
  -> begin_use(name)                  # active+1，防止别的请求卸它
  -> 已加载？ 是：touch 后直接推理
  -> 否：ensure_capacity(name)
       需要的量 = 该能力估算值 + margin（默认 512MB）
       空闲够？ 是：直接加载
       不够：按 LRU 卸「其它已加载、active=0」的模型
             每卸一个重新查一次真实空闲，够了就停
       还不够：返回 507，不硬加载
  -> 加载成功：记录加载前后空闲差，校准估算，写 vram_estimates.json
  -> 推理
  -> end_use(name) + touch
```

#### LRU 和“不卸正在用的”

- 候选按 `last_used_mono` 从老到新排序，最久没用过的先卸；
- 绝对不卸当前请求的目标能力；
- 正在推理的模型 `active > 0`，直接跳过；
- 卸载动作必须拿到该能力的 `asyncio.Lock`，和推理互斥；请求路径的加锁顺序
  固定为「先自己的推理锁，再管家的 op lock」，手动卸载不拿 op lock，
  避免两边反过来加锁造成死锁。

#### margin 是什么

`needed = estimate_mb + INDEX_TTS_VRAM_MARGIN_MB`。默认 512MB 是留给：

- CUDA context / cuBLAS workspace；
- 推理时的临时张量和显存碎片；
- 显存统计误差（WDDM 下 nvidia-smi 与 torch 的数字并不完全一致）。

调大更保守，调小更容易让多个模型共存：

```powershell
# 默认 512；如果仍然偶尔 OOM，改成 768 / 1024 再重启
$env:INDEX_TTS_VRAM_MARGIN_MB = "1024"
powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1
```

不要低于 256；8G 卡上 TTS 推理的临时峰值不小。

#### 估算校准（含保守下限）

每次加载成功后，记录「加载前空闲 - 加载后空闲」，并按以下规则更新估算：

- 每次最多向实测值修正 **±20%**，避免一次显存抖动把估算打得太低；
- 结果持久化到 `logs/vram_estimates.json`，服务重启后直接复用；
- **TTS 下限 6200MB**、**ASR 下限 900MB**、**Demucs 下限 800MB**：
  WDDM 会复用缓存，单纯用 `mem_get_info` 差值可能只测到 140MB（Demucs）、
  818MB（ASR）或 5.1GB（TTS），但推理峰值 / 与其它模型共存所需的空间更大，
  所以不能让校准无限下探。

日志示例：

```
显存估算校准：tts 6400 MB -> 6200 MB（本次实测增量 5176 MB，样本 1）
显存估算校准：asr 1200 MB -> 960 MB（本次实测增量 940 MB，样本 1）
显存估算校准：separate 1000 MB -> 800 MB（本次实测增量 140 MB，样本 1）
```

#### 显存不足时的错误

加载前放不下、且没有可卸的模型时，返回 **HTTP 507**，不会让进程 OOM：

```json
{
  "ok": false,
  "error": "insufficient_vram",
  "detail": "显存不足：空闲 1018 MB，需要 1312 MB（模型估算 800 MB + 安全余量 512 MB），已卸载其它模型 无，仍差 295 MB。请关闭占用显卡的程序后重试。",
  "need_mb": 1312,
  "free_mb": 1018,
  "released": [],
  "shortfall_mb": 295
}
```

服务继续可用；稍后释放显存或调小 margin 再重试即可。

### 17.4 实测（RTX 3070 8G，2026-11-08）

以下 nvidia-smi 数字均为 `used / free`（MiB），括号内是脚本记录的
`released`。测试素材：`voices\dafeiyu.wav`（9.4 秒人声）。

#### a) 空载 -> TTS -> 分离 -> TTS

| 步骤 | HTTP | nvidia-smi 结果 | 管家动作 / released |
|---|---:|---|---|
| 空载 | - | 618 / 7400 | 0 模型 |
| `POST /tts`（首次，29.3s） | 200 | 6506 / 1512 | 加载 TTS，无卸载；估算 6400→6200 |
| `POST /separate`（1.3s） | 200 | 1546 / 6472 | **卸 TTS**；`released_models:["tts"]` |
| `POST /tts`（21.6s） | 200 | 6488 / 1530 | **卸 Demucs** 后重载 TTS；日志 `腾出 separate` |

关键日志：

```
腾显存：卸掉 tts（释放约 6200 MB），当前空闲 6999 MB / 需要 1512 MB
腾显存：卸掉 separate（释放约 800 MB），当前空闲 6999 MB / 需要 6712 MB
```

#### b) ASR + 声纹 -> TTS -> ASR

| 步骤 | HTTP | nvidia-smi 结果 | 管家动作 / released |
|---|---:|---|---|
| 手动卸 TTS 后 | - | 876 / 7142 | 0 模型 |
| `POST /asr`（4.0s） | 200 | 1840 / 6178 | 加载 ASR（940MB）+ 声纹（22MB） |
| `POST /speakers/identify` | 200 | 1840 / 6178 | 复用声纹，无加载 |
| `POST /tts`（20.6s） | 200 | 6530 / 1488 | **卸 ASR** 后加载 TTS；`released:["asr"]` |
| `POST /asr`（3.0s） | 200 | 1840 / 6178 | **先卸声纹、再卸 TTS**，重载 ASR；`released:["speaker","tts"]` |

> 第 4 步只卸了 ASR、没卸声纹：ASR 释放后空闲 6931MB，已经 ≥ TTS 需要的
> 6712MB，LRU 就停了。声纹只有 22MB，留着不影响安全；后续请求需要时它会
> 被正常复用或卸载（第 5 步就按 LRU 先卸了它）。

#### c) 请求缺权重的 htdemucs_ft

```text
HTTP 503
{"ok":false,"error":"separate_error",
 "detail":"Demucs 权重不完整（htdemucs_ft），缺少：f7e0c4bc-ba3fe64a.th, d12395a8-e57c48e6.th, 92cfc3b6-ef3bcb9c.th, 04573f0d-f3cf25b2.th"}
```

权重检查发生在腾显存之前，所以不会为了一个注定失败的请求先卸掉别的模型；
服务状态完全不变。

#### d) /vram 与手动卸载

- `GET /vram` 正确显示 loaded、last_used、idle_seconds、estimate、measured；
- `POST /vram/unload {"models":["asr","speaker"]}` 返回
  `{"ok":true,"released":["asr","speaker"],...}`；
- 卸载后 nvidia-smi 从 1840 / 6178 回到 **873 / 7145**。

#### e) 重启后复用估算

重启前 `vram_estimates.json` 里已有 tts=6200、asr=900、speaker=123、
separate=800。重启日志：

```
已读取显存估算校准：...（tts=6200MB、asr=900MB、speaker=123MB、separate=800MB）
显存管家注册：tts，估算 6200 MB（来自持久化校准）
...
```

重启后 `/vram.estimates` 仍是这四个值，且四个模型 `loaded=false`（0 模型）。

#### 显存不足（507）的实测

先加载 TTS，然后发一个长文本 TTS 让它持续推理，同时请求 `/separate`。
此时 TTS `active=1` 不能卸，空闲不足：

```text
HTTP 507
{"ok":false,"error":"insufficient_vram",
 "detail":"显存不足：空闲 1018 MB，需要 1312 MB（模型估算 800 MB + 安全余量 512 MB），
           已卸载其它模型 无，仍差 295 MB。请关闭占用显卡的程序后重试。",
 "need_mb":1312,"free_mb":1018,"released":[],"shortfall_mb":295}
```

长 TTS 随后正常返回 200，`/health` 仍为 true；没有进程崩溃、没有整机卡死。

### 17.5 8G 卡常见组合

| 组合 | 实测占用（nvidia-smi） | 建议 |
|---|---:|---|
| 0 模型 | 0.6GB / 7.4GB 空闲 | 默认启动状态 |
| TTS | 6.5GB / 1.5GB 空闲 | 单独跑最稳 |
| ASR + 声纹 | 1.84GB / 6.18GB 空闲 | 可同时常驻，适合转写+认人 |
| Demucs | 1.55GB / 6.47GB 空闲 | 单独跑；TTS 在时由管家先卸 TTS |
| TTS + ASR + 声纹 | 约 7.3GB | 很紧，管家会按实际空闲决定是否卸；不建议依赖 |
| TTS + Demucs | 不推荐 | 当前策略让 TTS 下限 6200+margin，加载 TTS 时会先卸 Demucs |

原则：**不要手工用 nvidia-smi 的「空闲」去推算能不能再塞一个模型**。
Windows 的 WDDM 会把一部分显存分页/复用，nvidia-smi 和 torch 的统计不一致；
统一以 `/vram` 和管家日志为准。

### 17.6 调参与排错

| 环境变量 | 默认 | 说明 |
|---|---:|---|
| `INDEX_TTS_VRAM_MARGIN_MB` | `512` | 加载前安全余量（MB） |
| `INDEX_TTS_VRAM_ESTIMATES` | `logs\vram_estimates.json` | 校准文件路径（一般不用改） |

**想更保守**：margin 改 768/1024，重启服务。  
**想清空校准**：停服务，删除 `logs\vram_estimates.json`，再启动（会回到
6400/1200/300/1000 的初始估算）。  
**遇到 507**：

1. `curl http://127.0.0.1:7880/vram` 看 `free_mb`、哪个模型 `active>0`；
2. 不需要的模型用 `POST /vram/unload` 手动卸；
3. 或等空闲看门狗 / 另一个请求结束；
4. 确认不是 ComfyUI 等外部程序占着卡。

### 17.7 锁与并发

- 每个能力仍有自己的推理闸门（`_ENGINE_INFER_LOCK` 等），容量默认 1 = 串行；
  ASR 可显式设 `LOCALAI_ASR_PARALLEL=2` 放开两路（`docs/SERVICE_API.md` 有实测）；
- 管家另有 `_VRAM_LOAD_LOCK`，保证「判断显存 + 加载」不会两个请求同时做；
- `ensure_capacity` 内部的 `_op_lock` 串行化「查显存 + 卸模型」；
- 请求顺序固定：**先拿自己的推理锁 → begin_use → 再进管家**；
- 卸载顺序固定：**先拿目标推理锁 → 再卸**；
- 手动卸载不拿 `_op_lock`，避免和请求路径形成 ABBA 死锁。

### 17.8 归档：以前 `/separate` 的单独判断

旧版（2026-11-08 之前）只在 `_handle_separate_request` 里调用
`_release_gpu_for_separation()`：分离前查一次空闲，`< 2800MB` 就无条件
卸掉 TTS / ASR / 声纹，响应里的 `released_models` 列出被卸的模型。
它有三个局限：

1. 只保护 `/separate`，TTS / ASR / 声纹加载前没有同类保护；
2. 阈值写死 2800MB，不随实测和模型变化；
3. 一次性全卸，不区分 LRU，也不管模型是否正在推理。

现在这段代码和阈值已删除，`released_models` 字段保留，值来自
`ensure_capacity()` 的 `released`。README 16.4 里“低于约 2800MB 会主动卸载”
属于旧说明，**以本章为准**。

---

## 18. 通用下载与链接分析（v2.4 新增）

给一个 http/https 链接，就能下载视频/音频，并拿到**字幕时间轴 + 镜头边界**：

```bash
# 异步下载（进度看 /jobs/{id}）
curl -X POST http://127.0.0.1:7880/media/download \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.bilibili.com/video/BV1UY4G6pEJg","quality":"720p"}'

# 小文件同步下载（≤200MB）
curl -X POST http://127.0.0.1:7880/media/download/sync \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/a.mp4"}'

# 链接分析：下载 -> 站点字幕/ASR -> 场景边界 -> 可选抽帧
curl -X POST http://127.0.0.1:7880/media/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.bilibili.com/video/BV1UY4G6pEJg","max_frames":3}'

curl http://127.0.0.1:7880/media/download/status
```

- 引擎：`yt-dlp`（装在 `engine/.venv`，未装时其余能力不受影响）+ 直链兜底；
- 不占显存、不加载模型，服务仍 0 模型启动；
- 产物在 `data/downloads/<日期>/`（`.gitignore` 已排除）；
- 只允许 http/https，私有/回环/链路本地地址与 `file://` 一律拒绝（SSRF 防护）；
- B站自动读取本机 `bili_credential.json` 生成**临时** cookie 文件，用完即删，
  不写日志、不进仓库；其它站点需显式传 Netscape cookie 文件；
- 单文件上限 `LOCALAI_DOWNLOAD_MAX_MB`（默认 2048），同步接口上限
  `LOCALAI_DOWNLOAD_SYNC_MAX_MB`（默认 200），下载/分析超时
  `LOCALAI_DOWNLOAD_TIMEOUT`（默认 1800 秒）。

接口参数、SSRF 策略、实测数据与已知限制见
[`docs/SERVICE_API.md`](docs/SERVICE_API.md) 第 14 章。

---

## 19. 已经下过模型的用户怎么复用（v2.5 新增）

不用重新下载。打开管理界面 `/ui` →「模型」页：

1. **添加搜索路径**：例如 `D:\models`、`E:\ComfyUI\models`，或本服务以前的
   `E:\LocalAIService\models`。程序会递归检查子目录，最多 20 个路径。默认是空列表，
   不会写死任何本机目录；本机测试时在页面里临时添加即可。
2. 点「扫描已有模型」，等进度结束；结果会列出能力 / 模型 / 路径 / 大小 / 置信度。
3. 找到目标后点「使用这个」：服务只会把该目录写进 `config.json`（配置指向），
   **不移动、不复制、不创建 junction / symlink**，也不会删除原文件。下次加载该能力时
   直接从原目录读权重；已加载的旧模型不会被强行打断，可等空闲卸载或手动卸载。

也可以直接调接口：

```bash
# 默认没有搜索路径，先添加（只接受存在且可读的目录）
curl -X PUT http://127.0.0.1:7880/admin/models/search-paths \
  -H "Content-Type: application/json" \
  -d '{"add":"D:\\models"}'

# 后台扫描（不传 paths 就用已保存的路径；空列表会 400）
curl -X POST http://127.0.0.1:7880/admin/models/scan \
  -H "Content-Type: application/json" -d '{"max_depth":6}'

# 用任务 id 查进度和结果
curl http://127.0.0.1:7880/admin/models/scan/<task_id>

# 采用其中一条（示例：whisper small）
curl -X POST http://127.0.0.1:7880/admin/models/adopt \
  -H "Content-Type: application/json" \
  -d '{"capability":"whisper","path":"D:\\models\\faster-whisper-small","name":"small"}'
```

- 环境变量 `LOCALAI_MODEL_SEARCH_PATHS`（`;` 分隔）优先于 `config.json`；
- 采用写入的键：`tts_model_dir` / `asr_model_dir` / `whisper_model_dirs` /
  `translate_model_dir` / `speaker_model_dir` / `demucs_repo` /
  `vision_model_dir` / `vad_model_dir`，都只影响本服务配置；
- 置信度 `partial` 说明缺文件或大小/哈希对不上（结果里有 `missing` / `detail`）；
  缺签名文件时采用接口会 400，不会写入半截配置；
- 详细签名表、接口字段与默认行为见
  [`docs/SERVICE_API.md`](docs/SERVICE_API.md) 第 16 章。
