# 服务层接口文档（SERVICE API）

本文写给**任何调用方**：DSH 插件、脚本、其他软件、别的机器上的程序。
调用方只需要一个 base URL，不需要 import 本项目的代码，也不需要知道模型在哪。

- 默认地址：`http://127.0.0.1:7880`
- 交互式文档：`/docs`（Swagger UI）
- OpenAPI：`/openapi.json`
- 能力发现：`GET /capabilities`
- HTTP 契约版本：`api_version = "1"`（不兼容变更才 +1）

所有请求/响应均为 UTF-8 JSON，音频接口返回二进制。
模型是**懒加载**的：服务刚启动时 0 模型，第一次调用某能力才会进显存；
空闲超时自动卸载。首次调用会慢（加载几秒~几十秒），之后复用。

---

## 0. 快速开始

```bash
curl http://127.0.0.1:7880/health
curl http://127.0.0.1:7880/capabilities

curl -X POST http://127.0.0.1:7880/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，我是本地 AI 服务。","voice":"default","speed":1.0}' \
  --output hello.wav

curl -X POST "http://127.0.0.1:7880/asr?speaker=0" \
  -F "file=@hello.wav" -F "language=auto"
```

`/tts` 第一次调用会加载 IndexTTS-2.5（约 6 GB 显存），可能等待 10~60 秒；
之后再调用通常是秒级。8G 卡上如果同时要用别的 GPU 程序，先看 `/vram`。

---

## 1. 能力发现 `/capabilities`

任意程序启动时先打这个接口，就能知道：

- 服务版本、HTTP 契约版本、监听地址；
- 每个能力是否 `ready`（模型文件齐全）、是否需要显存、当前是否 `loaded`；
- 模型名、模型目录、空闲卸载秒数；
- 显存余量、上传大小限制、默认超时。

示例（字段有删减）：

```json
{
  "ok": true,
  "api_version": "1",
  "service": {
    "name": "localai-service",
    "version": "2.2.0",
    "python": "3.13.9",
    "platform": "Windows-11-10.0.26200-SP0",
    "started_at": "2026-11-08T15:40:00+08:00"
  },
  "listen": {"host": "127.0.0.1", "port": 7880, "base_url": "http://127.0.0.1:7880"},
  "backend": "engine",
  "asr": {
    "engine": "sensevoice",
    "engines": ["sensevoice", "whisper"],
    "whisper": {
      "model": "small",
      "device": "cuda",
      "compute_type": "int8_float16",
      "weights_dir": "E:\\LocalAIService\\models\\whisper\\small",
      "idle_unload_seconds": 900
    }
  },
  "auth": {"required": false, "scheme": "bearer"},
  "gpu": {"available": true, "free_mb": 7369, "total_mb": 8192, "margin_mb": 512},
  "capabilities": {
    "tts": {
      "name": "tts",
      "title": "语音合成（TTS）",
      "ready": true,
      "loaded": false,
      "requires_gpu": true,
      "model": "IndexTTS-2.5",
      "model_dir": "E:\\LocalAIService\\models\\TTS\\IndexTTS-2.5",
      "idle_unload_seconds": 900,
      "idle_seconds": null,
      "estimate_mb": 6200,
      "endpoints": ["/tts", "/v1/audio/speech", "/api/v1/tts/tasks"]
    }
  },
  "concurrency": {
    "upload_max_mb": 200,
    "audio_upload_max_mb": 50,
    "tts_max_text_chars": 5000,
    "job_timeout_seconds": 180.0
  }
}
```

> 判断「能不能用」看 `capabilities.<name>.ready`；
> 判断「现在有没有占显存」看 `loaded`；
> 判断「要不要先腾显存」看 `gpu.free_mb` 与 `estimate_mb`。

---

## 2. 健康检查与显存

### `GET /health`

不会加载模型，也不主动初始化 CUDA。关键字段：

| 字段 | 含义 |
| --- | --- |
| `ok` | TTS 模型文件就绪（engine 后端） |
| `model_ready` / `engine_loaded` | 文件齐全 / 当前是否已加载 |
| `voices` | 音色数量 |
| `asr` / `speaker` / `separate` | 各能力的 `ready`、`loaded`、空闲秒数 |
| `asr.engine` / `asr.vad` | 默认识别引擎；SenseVoice 分段用的 fsmn-vad 状态 |
| `whisper` | faster-whisper 的 `ready` / `loaded` / `model` / 空闲秒数（不加载模型） |
| `vram` | `free_mb` / `total_mb`（nvidia-smi，1 秒缓存） |
| `media` | ffmpeg/ffprobe 探测结果 |
| `comfy` | engine 后端固定为 `disabled` |

### `GET /vram`

```json
{
  "free_mb": 7361,
  "total_mb": 8192,
  "used_mb": 831,
  "torch_allocated_mb": null,
  "torch_reserved_mb": null,
  "models": [
    {
      "name": "tts",
      "loaded": false,
      "last_used": null,
      "idle_seconds": null,
      "estimate_mb": 6200,
      "measured_mb": 5176,
      "samples": 8,
      "active": 0,
      "loader": "IndexTTS-2.5（TTS，常驻）"
    }
  ],
  "margin_mb": 512,
  "estimates": {"tts": 6200, "asr": 900, "speaker": 123, "separate": 800}
}
```

### `POST /vram/unload`

手动释放显存，不影响后续请求（会按需重新加载）。

```bash
curl -X POST http://127.0.0.1:7880/vram/unload \
  -H "Content-Type: application/json" -d '{"models":["tts","separate"]}'
curl -X POST http://127.0.0.1:7880/vram/unload \
  -H "Content-Type: application/json" -d '{"all":true}'
```

返回：`{"ok":true,"released":["tts"],"free_mb":7360,"total_mb":8192}`。
可选名字：`tts` / `asr` / `whisper` / `speaker` / `separate`；未知名字返回 400。

---

## 3. TTS（语音合成）

### 3.1 `POST /tts`

请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `text` | string | 必填 | 1~5000 字 |
| `voice` | string | `default` | 音色名（`<HOME>/voices` 下文件名去扩展名） |
| `speed` | float | 1.0 | 0.5~2.0 |
| `emotion` | string | 无 | 8 维向量或情感文本（文本需 Qwen 情感模型） |
| `seed` | int | 随机 | 0~2^32-1 |
| `language` | string | 自动 | `zh`/`en`/`ja`/`es`/`ar` |

响应：`audio/wav`（16bit PCM），带响应头：

- `X-Seed`：实际使用的种子
- `X-Language`：实际语言
- `X-Audio-Filename` / `X-Audio-Path`：服务端留档文件

```bash
curl -X POST http://127.0.0.1:7880/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气不错。","voice":"default","speed":1.1,"seed":42}' \
  -o out.wav
```

### 3.2 `POST /v1/audio/speech`（OpenAI 兼容）

```json
{"model":"indextts","input":"要合成的文本","voice":"default","response_format":"wav","speed":1.0}
```

只支持 `response_format="wav"`（统一返回 16bit PCM WAV）；
其他值返回 400 `unsupported_response_format`。

### 3.3 DSH 语音插件兼容接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/voices` | `{"voices":["default","female-A"],"count":2}` |
| POST | `/api/v1/tts/tasks` | 请求体 `{text,prompt_audio,speed,seed,language}`，返回 WAV |
| POST | `/api/v1/upload` | multipart `file`，存进音色目录 |

`prompt_audio` 可以传音色名、文件名或完整路径；不认识的名字会回落到第一个音色。

### 3.4 音色列表 `GET /voices`

```json
{
  "voices": [{"name":"default","path":"...","filename":"default.mp3","format":"mp3","duration":3.2}],
  "count": 7,
  "directory": "E:\\LocalAIService\\voices"
}
```

音色目录每次请求重新扫描，新增文件无需重启。

---

## 4. ASR（语音识别）

服务内置两个识别引擎，注册成两个并列能力，各自懒加载、各自占显存、按 LRU 互相腾退：

| 引擎 | 注册能力 | 模型 | 时间戳 | 说明 |
| --- | --- | --- | --- | --- |
| `sensevoice`（默认） | `asr` | SenseVoiceSmall | 无原生时间戳；`/asr/segments` 用 fsmn-vad 切句（`timestamp_source=vad`） | 中文短音频最快，带情感/事件标签与声纹 |
| `whisper` | `whisper` | faster-whisper（`tiny/base/small/medium/large-v3`） | 原生 segment 时间戳（`timestamp_source=native`） | 需要精确时间戳、多语种；默认档 `small` |

### `POST /asr`，别名 `POST /api/v1/asr`

multipart：

| 字段 | 说明 |
| --- | --- |
| `file` | 音频或视频（wav/mp3/m4a/ogg/flac/mp4/mkv/webm…，≤200MB） |
| `language` | `auto`/`zh`/`en`/`ja`/`ko`/`yue`/`nospeech` |
| `speaker` | `1`（默认，识别声纹）/ `0`（跳过，连 campplus 都不加载） |
| `engine` | `sensevoice` / `whisper` / `auto`；**空 = 按配置默认**（`LOCALAI_ASR_ENGINE`） |
| `timestamps` | `0`（默认）/ `1`；只影响 `engine=auto` 的选择：要时间戳时选 whisper |

`engine=auto` 规则：`timestamps=1` 走 whisper，否则走配置默认。
旧的默认行为完全不变（不带 `engine` → sensevoice + 声纹）。

非 WAV 会先用 ffmpeg 归一成 **16kHz 单声道 WAV** 再识别；采样率不用自己处理。
音频容器转码失败会回退直接识别；视频转码失败直接报错。

响应：

```json
{
  "ok": true,
  "engine": "sensevoice",
  "text": "识别出的文字",
  "language": "zh",
  "emotion": "NEUTRAL",
  "events": [],
  "duration": 3.2,
  "elapsed_ms": 850,
  "loaded": true,
  "speaker": {"name": "主人", "score": 0.8123, "known": true},
  "source_format": ".wav",
  "converted": false
}
```

`engine` 是本次真正用到的引擎（`sensevoice` / `whisper`）。声纹失败不会影响转写，
会多一个 `speaker_error` 字段。

```bash
curl -X POST "http://127.0.0.1:7880/asr?speaker=0" \
  -F "file=@meeting.mp4" -F "language=auto"

# 指定 whisper（原生时间戳，但 /asr 只返回文本；要分段用 /asr/segments）
curl -X POST "http://127.0.0.1:7880/asr" \
  -F "file=@meeting.wav" -F "engine=whisper" -F "language=zh"
```

### `GET /asr/status`

```json
{
  "ready": true,
  "loaded": false,
  "engine": "sensevoice",
  "engines": {
    "sensevoice": {"model": "SenseVoiceSmall", "ready": true, "loaded": false,
                    "model_dir": "...", "idle_unload_seconds": 900, "vad": {"ready": true, "loaded": false}},
    "whisper": {"model": "faster-whisper-small", "ready": true, "loaded": false,
                "model_dir": "E:\\LocalAIService\\models\\whisper\\small",
                "device": "cuda", "compute_type": "int8_float16",
                "size": "small", "size_mb": 463.7, "idle_unload_seconds": 900}
  },
  "whisper": {"...": "同上，另带 idle_seconds"},
  "idle_seconds": null,
  "parallel": {"capacity": 2, "in_use": 0, "peak": 0, "env": "LOCALAI_ASR_PARALLEL"}
}
```

### `POST /asr/segments`（带时间戳的分段转写）

multipart：`file`（音频/视频）、`language`、`engine=sensevoice|whisper|auto`。
`engine=auto` 在**本接口固定选 whisper**（本接口就是要时间戳）。

```bash
curl -X POST http://127.0.0.1:7880/asr/segments \
  -F "file=@speech.wav" -F "language=zh" -F "engine=whisper"
```

响应：

```json
{
  "ok": true,
  "engine": "whisper",
  "language": "zh",
  "duration": 12.3,
  "segments": [
    {"start": 0.0, "end": 2.4, "text": "第一句"},
    {"start": 2.6, "end": 5.1, "text": "第二句"}
  ],
  "text": "第一句第二句",
  "timestamp_source": "native",
  "elapsed": 0.8,
  "device": "cuda",
  "converted": false,
  "source_format": ".wav"
}
```

保证：

- 分段**按时间升序、互不重叠、`end > start`**（服务端统一整理）；
- `timestamp_source` 区分来源：whisper 是 `native`，sensevoice 是 `vad`；
- sensevoice 的 `start/end` 取 fsmn-vad 的语音段边界（段间静音如实保留为间隙）；
- 分段之间的静音不会被硬塞进某一段，`duration` 是整段音频时长。

### Whisper 引擎（faster-whisper）

| 项目 | 值 |
| --- | --- |
| Python 包 | `faster-whisper==1.2.1` + `ctranslate2==4.8.5` + `av==17.0.1` + `onnxruntime==1.31.1` |
| 显存运行库 | `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` / `nvidia-cuda-runtime-cu12`（本机 torch 是 cu130，两者互不覆盖） |
| 权重目录 | `LOCALAI_HOME/models/whisper/<size>/`（默认 `E:\LocalAIService\models\whisper\small`） |
| 支持档位 | `tiny` / `base` / `small`（默认）/ `medium` / `large-v3` |
| 下载顺序 | 本地已有 → 本机 HF 缓存 → ModelScope（`Systran/faster-whisper-<size>`）→ HF 镜像 `hf-mirror.com` |
| 设备 / 精度 | `cuda` + `int8_float16`（默认）；CPU 回退 `int8` |
| 加载策略 | 服务启动 0 模型；第一次请求才下载/加载；`unload()` 释放模型 + gc + empty_cache |
| 空闲卸载 | `LOCALAI_WHISPER_IDLE_UNLOAD`（默认 900 秒） |

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCALAI_ASR_ENGINE` | `sensevoice` | 识别默认引擎（`sensevoice` / `whisper`）；任务级 `engine=` 可覆盖 |
| `LOCALAI_WHISPER_MODEL` | `small` | 档位 |
| `LOCALAI_WHISPER_DEVICE` | `cuda` | 设备；设 `cpu` 走 CPU |
| `LOCALAI_WHISPER_COMPUTE` | `int8_float16` | 计算类型 |
| `LOCALAI_WHISPER_DIR` | `<HOME>/models/whisper` | 权重根目录 |
| `LOCALAI_WHISPER_IDLE_UNLOAD` | `900` | 空闲卸载秒数（0=关闭） |
| `LOCALAI_WHISPER_PROMPT` | `以下是普通话的句子。` | 中文提示词；实测把 small 的 CER 从 16.6% 降到 0.9%，设空可关 |
| `LOCALAI_WHISPER_ALLOW_CPU_FALLBACK` | `1` | CUDA 失败时是否回退 CPU |

> 权重不会散到用户目录：`WhisperModel` 直接用 `models/whisper/<size>/` 本地目录，
> 下载由服务层先落盘再加载。`models/` 已在 `.gitignore` 中。

### 识别实测（RTX 3070 8G，2026-11）

评测集：8 条 TTS 合成的**已知文本**中文短音频（合计 25.6 秒，单条 2.7~3.6 秒），
语言固定 `zh`，whisper `beam_size=5`、`vad_filter=True`、`initial_prompt=以下是普通话的句子。`。
CER = 字符错误率（去标点/空白后的编辑距离 / 参考字符数）。

| 引擎 | 模型 | CER | 单条延迟（均值） | RTF（音频时长/耗时） | 冷加载 |
| --- | --- | --- | --- | --- | --- |
| sensevoice | SenseVoiceSmall | **1.73%** | 0.165 s | 19.45 | 10.84 s |
| whisper | tiny | 17.03% | 0.232 s | 13.81 | 0.41 s |
| whisper | base | 9.16% | 0.273 s | 11.74 | 0.44 s |
| whisper | **small** | **0.89%** | 0.328 s | 9.75 | 1.18 s |
| whisper | medium | **0.89%** | 0.646 s | 4.95 | 3.28 s |
| whisper | small（CPU/int8） | — | 2.19 s | 1.46 | 1.82 s |

> RTF 按任务书定义 = 音频时长 / 处理耗时，越大越快。CPU small 仍快于实时（1.46×）。

显存（整卡 `nvidia-smi` used，加载期间每 100ms 采样；基线约 1.4~1.6GB 含桌面与
CUDA context）：

| 模型 | 加载后整卡 used | 加载增量 | 卸载后 |
| --- | --- | --- | --- |
| whisper tiny | 1615 MB | +220 MB | 1619 MB |
| whisper base | 1698 MB | +96 MB | 1596 MB |
| whisper small | 1854 MB | +288 MB | 1576 MB |
| whisper medium | 2534 MB | +960 MB | 1597 MB |
| sensevoice | 2470 MB | +893 MB | 1528 MB |

> whisper `small` 常驻只占约 0.3GB（int8_float16），完全能和 TTS 共存；
> `medium` 约 1GB，仍可与 TTS 同时驻留（8G 卡余量足够），紧张时显存管家会按 LRU 先卸 TTS。

长音频（46.35 秒、12 句、句间 0.6 秒静音）：

| 项目 | 结果 |
| --- | --- |
| 引擎 / 档位 | whisper `small`（cuda / int8_float16） |
| 处理耗时 | 2.95 秒（RTF 15.72） |
| segments | 12 段（与 12 句一一对应） |
| 抽查 3 段裁剪再识别 | 段 0（0~4.0s）、段 6（23.0~27.0s）、段 11（42.0~46.0s），whisper 与 sensevoice 复识 CER 均 **0.00%** |
| 与真实句子起点误差 | 最大 0.64 秒（whisper 的段边界略偏早，裁剪片段内容仍完整） |

> 结论：需要时间戳用 whisper `small`；`medium` 的 CER 与 small 相同但慢 2 倍、多占
> 0.7GB，默认不选。`tiny/base` 中文 CER 偏高（17%/9%），只适合快速预览。

---

## 5. 声纹（campplus）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/speakers` | 已登记列表 |
| GET | `/speakers/status` | 引擎状态 + 阈值 |
| GET | `/speakers/unknown` | 未知声纹聚类 |
| POST | `/speakers/enroll` | multipart `file` + `name`，同名累加样本 |
| POST | `/speakers/identify` | multipart `file`，返回 matches/best/known |
| DELETE | `/speakers/{name}` | 删除，不存在时 `deleted=false` |

隐私：`<HOME>/speakers/*.json` 只存 192 维向量和名字，**不存原始音频**；
上传音频只落临时目录，算完立即删除。

```bash
curl -X POST http://127.0.0.1:7880/speakers/enroll \
  -F "file=@voice.wav" -F "name=主人"
curl -X POST http://127.0.0.1:7880/speakers/identify -F "file=@test.wav"
```

---

## 6. 人声分离（Demucs）

### `POST /separate`，别名 `POST /api/v1/separate`

multipart：

| 字段 | 说明 |
| --- | --- |
| `file` | 音频或视频（≤200MB） |
| `model` | `htdemucs`（默认）/ `htdemucs_6s`（多 guitar/piano）/ `htdemucs_ft`（需额外权重） |
| `stems` | 逗号分隔；默认 `vocals,drums,bass,other` |

响应（stem 已落盘到 `outputs/separate/<时间戳>_<6位>/`）：

```json
{
  "ok": true,
  "model": "htdemucs",
  "sample_rate": 44100,
  "duration": 12.3,
  "elapsed": 18.4,
  "stems": {"vocals": {"path": "...\\vocals.wav", "duration": 12.3}},
  "dir": "E:\\LocalAIService\\outputs\\separate\\20261108_154000_ab12cd",
  "loaded": true,
  "released_models": ["tts"]
}
```

`released_models` 表示为了腾显存被 LRU 卸掉的其它模型；下次调用会自动重载。

### `GET /separate/status`

`{ready, loaded, model, idle_seconds, idle_unload_seconds, available, weights_dir, device, detail}`。

---

## 7. 媒体工具（ffmpeg）

全部 multipart，字段名统一 `file`；上传上限 200MB。

### `POST /media/probe`

```json
{"ok":true,"duration":12.34,"format":"mov,mp4,m4a","audio":{"codec":"aac","sample_rate":44100,"channels":2},"video":{"codec":"h264","width":1920,"height":1080,"fps":30.0}}
```

### `POST /media/to_wav`

返回 `audio/wav`（16k 单声道 PCM16）；响应头 `X-Duration`、`X-Sample-Rate=16000`、`X-Channels=1`。

### `POST /media/frames`

表单：`count` / `fps` / `times`（三选一）、`width`（可选）。
返回：

```json
{"ok":true,"count":3,"dir":"...\\outputs\\frames\\20261108_154000_ab12cd","elapsed_ms":420,
 "frames":[{"path":"...\\frame_0001.jpg","time":0.0}],"preview_b64":"...","preview_mime":"image/jpeg"}
```

### `POST /media/audio`

表单 `format=wav|mp3`；只抽音轨，保留源采样率与声道，返回音频二进制。

### `POST /media/scenes`（场景 / 镜头边界检测）

multipart：`file`（视频）、`threshold`（默认 `0.3`）、`max_scenes`（默认 `200`）、
`min_gap`（默认 `1.0` 秒）。

实现：`ffmpeg -vf "select='gt(scene,TH)',metadata=mode=print:file=-,showinfo" -f null -`，
解析 stderr 的 `pts_time` 与 stdout 的 `lavfi.scene_score`，再做「最小间隔去抖」
（间隔小于 `min_gap` 的只保留前一个）。

```bash
curl -X POST http://127.0.0.1:7880/media/scenes \
  -F "file=@cut.mp4" -F "threshold=0.3" -F "min_gap=1.0"
```

```json
{
  "ok": true,
  "count": 2,
  "scenes": [
    {"time": 1.52, "score": 1.0},
    {"time": 3.04, "score": 1.0}
  ],
  "duration": 4.56,
  "threshold": 0.3,
  "max_scenes": 200,
  "min_gap": 1.0,
  "json_path": "E:\\LocalAIService\\outputs\\scenes\\20261108_173056_247.json",
  "elapsed_ms": 92
}
```

- 结果同时落盘到 `<OUTPUT>/scenes/<时间戳>.json`（含 source/threshold/scenes）；
- `time` 单位秒，`score` 是 ffmpeg 的 scene 分数（0~1）；
- 没有视频流的文件返回 400；`threshold` 越界 / `max_scenes<=0` / `min_gap<0` 也返回 400。

实测（合成 4.56 秒黑→白→蓝切换视频，真实切换点 1.50s / 3.00s）：

| 检测边界 | 真实边界 | 误差 | score | 耗时 |
| --- | --- | --- | --- | --- |
| 1.52 s | 1.50 s | +0.02 s | 1.0 | 0.092 s（整段） |
| 3.04 s | 3.00 s | +0.04 s | 1.0 | — |

> 误差来自帧级时间戳（25fps = 0.04s/帧），不是算法抖动。

---

## 8. ComfyUI（仅 `TTS_BACKEND=comfy`）

- `GET /comfy/status`：队列状态；ComfyUI 不可达时 502。
- `TTS_BACKEND=comfy` 时 `/health` 的 `ok` 表示 ComfyUI 可达，`comfy` 为 `up`/`down`。

当前默认后端是 `engine`，一般用不到这一节。

---

## 9. 错误码与错误体

普通接口：

```json
{"ok": false, "error": "insufficient_vram", "detail": "显存不足：……", "need_mb": 6912, "free_mb": 6000, "released": ["asr"], "shortfall_mb": 912}
```

`/v1/*` 走 OpenAI 风格：

```json
{"error": {"message": "...", "type": "upstream_error", "code": "engine_error"}}
```

| 状态码 | `error` | 触发条件 | 调用方建议 |
| --- | --- | --- | --- |
| 400 | `voice_not_found` / `invalid_*` | 参数/音色/格式不对 | 修正参数，不要重试 |
| 401 | `unauthorized` | 服务端设了 `LOCALAI_TOKEN` | 带 `Authorization: Bearer` |
| 404 | `voice_not_found` | 音色不存在 | 先 `/voices` |
| 404 | `job_not_found` | 任务 id 不存在 | 检查 id |
| 409 | `job_canceled` / `not_cancelable` | 任务已取消 / 已结束 | 不要重复取消 |
| 413 | — | 文件超过 200MB | 先压缩/裁剪 |
| 502 | `comfy_error` | ComfyUI 上游错误 | 检查 ComfyUI |
| 503 | `engine_error` / `asr_error` / `speaker_error` / `separate_error` | 引擎加载/推理失败 | 看 `detail`，可重试 |
| 504 | `comfy_error` / `job_timeout` | 等待上游超时 / 任务超时 | 缩短任务后重试 |
| 507 | `insufficient_vram` | 腾完显存仍不够 | 用 `/vram/unload` 或关闭占卡程序 |

**507 是正常保护，不是崩溃**：服务不会因为显存不足退出，卸载后仍可继续服务。

---

## 10. 并发、超时与首次加载

- 所有 TTS / ASR / 人声分离任务都进同一个持久化队列，由**单 worker 派发**；
  默认每类并行度为 1；ASR 引擎的可配置并发见 12.2.1（默认 2，可设 1 回到完全串行），
  所以默认实际仍是一条一条跑。声纹/媒体接口仍是直接调用。
- 每个 GPU 能力的推理闸门容量可调；默认 1，语义与旧锁相同。
- 加载前统一经过显存管家：按 LRU 卸载其它空闲模型，仍不够则 507。
- 首次调用 TTS 约 10~60 秒；ASR/声纹/分离通常几秒~几十秒。
- 客户端建议：同步 TTS 超时 ≥ 180 秒；ASR/分离 ≥ 300 秒；连接超时 5 秒。
  任务队列接口另有 `LOCALAI_JOB_TIMEOUT`（默认 600 秒）兜底。
- 长文本：TTS 单次上限 5000 字；更长的文本请调用方自行分段（见插件契约）。
- 空闲卸载默认 900 秒；`/capabilities` 里有每个能力的实际值。

---

## 11. 鉴权（预留，默认关闭）

服务端设置 `LOCALAI_TOKEN` 后，除 `/health`、`/capabilities`、`/docs`、
`/openapi.json` 外都要求：

```http
Authorization: Bearer <LOCALAI_TOKEN>
```

也接受 `X-LocalAI-Token: <token>`。未设置时完全放行（默认）。

---

## 12. 任务队列（持久化 + 三层排序 + 同类并行）

从 2.2.0 起，**所有 TTS / ASR / 人声分离请求都会先变成一条任务记录**，
再交给单 worker 调度器派发（默认每类并行 1，实际一条一条跑）：

- 旧同步接口（`/tts`、`/v1/audio/speech`、`/api/v1/tts/tasks`、`/asr`、
  `/api/v1/asr`、`/separate`、`/api/v1/separate`）**响应体、状态码、二进制
  格式完全不变**，只是内部改成「提交任务 → 等完成」；
- 新增的 `/jobs` 系列接口让调用方可以「提交后立刻返回、自己轮询」，
  适合长文本、批量转写、分离这类耗时任务；
- 任务记录落在 SQLite：`<LOCALAI_HOME>/data/jobs.db`，服务重启不丢。

### 12.1 任务表结构

表名 `jobs`（WAL 模式；索引 `(state, priority DESC, created_at)`）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | TEXT PRIMARY KEY | uuid hex |
| `kind` | TEXT | `tts` / `asr` / `speaker` / `separate` / `media` |
| `action` | TEXT | `tts.synthesize` / `asr.transcribe` / `separate.split` / `speaker.enroll` / `media.frames` … |
| `priority` | INTEGER | 默认 0，越大越先（负数=让路） |
| `state` | TEXT | `queued` / `running` / `done` / `failed` / `canceled` |
| `created_at` | TEXT | 提交时间（ISO8601 带时区） |
| `started_at` | TEXT | 开始执行时间 |
| `finished_at` | TEXT | 进入终态时间 |
| `payload` | TEXT | JSON：文本 / 上传文件本地路径 / 参数 |
| `result` | TEXT | JSON：输出路径、时长、seed、模型等 |
| `error` | TEXT | 失败原因（成功时为空） |
| `attempts` | INTEGER | 被调度器领取的次数（重启 requeue 会累加） |
| `client_ip` | TEXT | 调用方 IP |
| `meta` | TEXT | JSON 预留扩展；取消 running 任务时写 `cancel_requested` |

### 12.2 调度规则（严格三层，无 aging）

从高到低**只有三层**，没有老化、没有自动提权：

1. **同类型优先**（避免频繁加载/卸载模型）
   如果当前已加载的能力（调度器的 `current_kind`）在队列里还有任务，就先跑它们；
   其它类型优先级再高也**不许**切走。
2. **优先级**：任务提交时的可选参数 `priority`（整数，默认 0，越大越先）。
3. **提交时间**：`created_at` 早的先；同一毫秒内按入库顺序（rowid）FIFO。

实现方式：

```text
current_kind 有效、且该 kind 队列非空
    -> 只在这个 kind 里按 (priority DESC, created_at ASC, rowid ASC) 取
否则
    -> 在所有 kind 里按同样的键选出「下一个任务」，用它的 kind 作为新的 current_kind
```

**同类清空是唯一的切换出口。** 明确接受「同类型持续涌入会饿死其它类型」
这个代价（主人要求如此）：只要 tts 队列一直有任务，priority=999 的 separate
也只能等着。已经 running 的任务不能被抢占，优先级只影响还在 `queued` 的任务。

### 12.2.1 同类型并行（显存允许时）

同一 `current_kind` 的多个任务可以同时跑；**不同 kind 仍然串行**（避免同一张
卡上叠两套模型）。控制项：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCALAI_JOBS_PARALLEL_TTS` | `1` | TTS 同类并行上限 |
| `LOCALAI_JOBS_PARALLEL_ASR` | `1` | ASR 同类并行上限（兼容旧名） |
| `LOCALAI_ASR_PARALLEL` | `2` | ASR 引擎真实并发容量，范围 1~4；新名优先（设 1 = 完全串行） |
| `LOCALAI_ASR_UNLOAD_TIMEOUT` | `30` | 卸载等待推理退出的超时（秒） |
| `LOCALAI_JOBS_PARALLEL_SEPARATE` | `1` | 人声分离同类并行上限 |
| `LOCALAI_JOBS_PARALLEL_EXTRA_MB` | `600` | 每新增一个并行任务，要求额外预留的显存 |
| `LOCALAI_JOBS_MAX_PARALLEL` | `1` | 总并发硬上限；`1` = 不额外限制，由每类上限决定 |

派发规则：

1. 第一个任务不检查显存（它负责按需加载模型，加载本身由显存管家把关）；
2. 在已派发 1 个之后，每新增 1 个并行任务前检查：
   `vram.snapshot().free_mb - margin_mb >= LOCALAI_JOBS_PARALLEL_EXTRA_MB`，
   不够就只跑已经派发的那些，不新开（等已有任务结束再试）；
3. 适配器的 `infer_lock` 已从单把 `asyncio.Lock` 换成**容量可调的并发闸门**：
   容量 1 语义完全不变；容量 >1 时允许 N 个并发进入推理；
4. 是否真的允许同模型并发由适配器的 `CAN_INFER_IN_PARALLEL` 决定。没有实测
   确认的类型一律强制 1，并在启动日志与 `/capabilities` 里说明原因；
5. 撞 CUDA OOM：立即把该 kind 的并行**临时降到 1**、卸载其它模型腾空间、
   只重试这一条任务一次；仍失败则任务 `failed`，进程不崩。

#### 识别并发（SenseVoice，默认 2；设 1 = 完全串行）

**开关与优先级**

1. `LOCALAI_ASR_PARALLEL`（新名，推荐）：识别引擎信号量容量，`1~4`，非法值
   按 1 处理；
2. `LOCALAI_JOBS_PARALLEL_ASR`（旧名）：只设置它也能生效；
3. 两者同时设置且不一致时，**新名 `LOCALAI_ASR_PARALLEL` 优先**。
   `Config.from_env()` 会把最终值同时写回 `LOCALAI_ASR_PARALLEL` 和
   `LOCALAI_JOBS_PARALLEL_ASR`，保证「调度器开几路」与「引擎放几路」永远同值。

**锁模型**

- `engine/asr_engine.py` 原来的模块级 `_INFER_LOCK` 换成了容量可配的
  `_InferenceGate`（内部 `threading.BoundedSemaphore`）；
- 容量 1：语义与旧锁完全一致，同一时刻只有一个推理；
- 容量 >1：最多 N 个推理同时进入；推理路径**只拿信号量、不嵌套独占锁**；
- 模型加载 / 卸载是结构性操作：加载只在模型为空时发生（此时不可能有推理），
  并受 `_LOAD_LOCK` 保护；卸载先收干全部推理槽位，再释放模型和 CUDA 缓存。
  等待超过 `LOCALAI_ASR_UNLOAD_TIMEOUT`（默认 30 秒）只记错误日志、放弃本次
  卸载，模型保持加载，不会拆掉正在推理的模型，也不会打乱信号量计数。

**实测数据（RTX 3070 8G，2026-11-08）**

方法：服务重启后先跑 1 条预热（把 SenseVoice 加载进显存），再用同一批 8 条
3.07~9.69 秒 wav，通过真实 HTTP `POST /jobs` 并发提交，轮询完成；逐条延迟取
任务结果里的 `elapsed_ms`，总耗时为客户端墙钟，吞吐 = 条数 / 总耗时；
每 100ms 调一次 `nvidia-smi` 采整卡 used，取峰值。测量脚本见
`tools/measure_asr_parallel.py`，原始 JSON 在 `logs/asr_parallel_p*.json`。

| 并行度 | 单条均值 / 中位 | 总耗时 | 吞吐 | 峰值显存（整卡 used） | 失败 |
| --- | --- | --- | --- | --- | --- |
| 1（默认） | 0.109 s / 0.106 s | 0.896 s | 8.93 条/s | 2693 MB | 0 |
| 2 | 0.135 s / 0.136 s | 0.585 s | 13.68 条/s（+53%） | 2692 MB | 0 |
| 3 | 0.201 s / 0.200 s | 0.606 s | 13.21 条/s（+48%） | 2704 MB | 0 |

说明：峰值是整卡 used（含系统/其它程序约 1.4GB）；SenseVoice 加载前后基线的
增量约 1.2GB。并行后单条延迟会因抢 GPU 略微升高，但总吞吐明显上升；3 路已经
进入饱和区，反而比 2 路略慢。

**10 条正确性比对（并行 2 vs 串行 1）**

同一批 10 条不同内容（3.07~9.69 秒），并行 2 连续提交。逐条比较
`result.text`：**10/10 完全一致，0 条串结果、0 条空文本**。例如：

| wav | 串行文本 | 并行 2 文本 |
| --- | --- | --- |
| `...3148330195.wav` | 再生成一句，看看会不会自动加载回来。 | 同左 |
| `...4101165473.wav` | 这是open AI金融接口的回归测试。 | 同左 |
| `...2726196196.wav` | 这是DSH插线接口的回归测试。 | 同左 |
| `...4232563416.wav` | 接口对齐了，DSC插件现在能直接用我的声音朗读。 | 同左 |

另外 6 条（含 5.70 / 7.63 / 8.49 / 9.69 秒长音频）也全部一致。SenseVoice 是
无状态 CTC 推理，每次 `generate()` 使用独立的 `cache={}`，并发不会共享解码状态。

**30 条稳定性（并行 2）**

连续提交 30 条（8 条素材循环）：**30/30 `done`，无异常、无 5xx、无超时**；
总耗时 2.334 s，吞吐 12.85 条/s，峰值显存 2733 MB；批量前 2657 MB、批量后
2713 MB，前后差 **+56 MB（< 200 MB）**，没有显存泄漏。调度器
`/jobs/stats` 的 `asr` 闸门峰值为 2，引擎 `/asr/status` 的 `parallel.peak`
也为 2，确认两路真的同时进入了引擎。

**OOM / 异常兜底**

调度器已有的机制对新并发路径仍然生效：`AsrError` 的文本包含
`CUDA out of memory` 时，`is_cuda_oom()` 能识别；`_degrade_after_oom()` 会把
ASR 闸门容量设回 1、把该 kind 记入 `parallel_degraded`、卸载其它模型，然后
只重试当前任务一次；再失败才置 `failed`，进程不退出。`tests/test_jobs.py` 的
`test_parallel_oom_degrades_to_serial_and_retries_once` 覆盖了这条路径。

**默认值结论**

并行 2 稳定、可正确识别、吞吐提升 53%（>20%），技术上达到「可以默认开 2」的
条件（实测 +53%，10/10 结果一致、30/30 稳定、无显存泄漏），因此默认值已从 1 改为 **2**。
想更保守/省显存：显式设置 `LOCALAI_ASR_PARALLEL=1` 回到完全串行即可；
并行 3 不建议作为默认值。

**已知风险**

- 并发会抬高瞬时显存和单条延迟；8G 卡同时跑 TTS 等大模型时，仍可能 OOM，
  此时调度器会降级重试；
- 本机 10 条比对没有出现结果串扰，但并发能力依赖 SenseVoice/funasr 的线程安全；
  升级 funasr、torch 或替换模型后应重新跑 10 条比对和 30 条稳定性；
- 默认 2 会走并发路径；设 1 时与旧行为完全一致（同一时刻只跑一条识别）。

**回滚方式**

把 `LOCALAI_ASR_PARALLEL` 和 `LOCALAI_JOBS_PARALLEL_ASR` 都设为 `1`（或直接
删掉这两个变量），重启服务即可回到完全串行；不需要改数据、模型或声纹。
代码级回滚直接 revert 对应 commit。

TTS / 人声分离的旧实测数据保留如下（它们仍固定为 1）：

| 类型 | 并行度 | 总耗时 | 峰值显存（整卡 used） | 是否有收益 |
| --- | --- | --- | --- | --- |
| TTS（4 条短句） | 1 | 8.53 s | 6439 MB | — |
| TTS（4 条短句） | 2 | 8.06 s（-5.5%） | 6459 MB | 否（<10%，噪声级） |
| 分离（2 条约 10s） | 1 | 0.862 s（3 次中位） | 1633 MB | — |
| 分离（2 条约 10s） | 2 | 0.717~0.825 s（不稳定） | 1633 MB | 否（不稳定） |

- **TTS = 1**：并行 2 的总耗时只降 5.5%，没有超过 10% 的吞吐收益；而且
  `engine/engine.py` 内部有 `_INFER_LOCK`、随机种子是模块级状态，并发不安全。
- **分离 = 1**：并行 2 的收益不稳定，且引擎内部同样有锁。

> 调度层的并行能力本身已经实现并用测试验证：用假执行器（不加载模型）时，
> `LOCALAI_JOBS_PARALLEL_ASR=2` 下 4 个 ASR 任务的 `running` 峰值确实到 2，
> 显存检查不通过时自动回落到 1（见 `tests/test_jobs.py` 的 d 组）。
> 真实模型的 ASR 并发现在由引擎信号量真正放开，默认仍为 1。

### 12.2.2 超时、崩溃恢复与其它开关

5. **超时**：`LOCALAI_JOB_TIMEOUT`（默认 600 秒）到点置 `failed`，
   并等底层 GPU 调用真正退出后才释放名额（GPU 调用无法从线程里强杀）。
6. **崩溃恢复**：启动时把上次遗留的 `running` 按
   `LOCALAI_JOBS_RECOVER=requeue|fail`（默认 `requeue`）重新排队或判失败。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCALAI_JOBS_RECOVER` | `requeue` | 重启时 running 的处理方式 |
| `LOCALAI_JOBS_MAX_PARALLEL` | `1` | 总并发硬上限（`1`=不额外限制） |
| `LOCALAI_JOBS_PARALLEL_TTS` | `1` | TTS 同类并行上限 |
| `LOCALAI_JOBS_PARALLEL_ASR` | `2` | ASR 同类并行上限（兼容旧名，跟随 `LOCALAI_ASR_PARALLEL`） |
| `LOCALAI_ASR_PARALLEL` | `1` | ASR 引擎真实并发容量，范围 1~4 |
| `LOCALAI_ASR_UNLOAD_TIMEOUT` | `30` | 卸载等待推理退出的超时（秒） |
| `LOCALAI_JOBS_PARALLEL_SEPARATE` | `1` | 分离同类并行上限 |
| `LOCALAI_JOBS_PARALLEL_EXTRA_MB` | `600` | 每个新增并行任务预留显存（MB） |
| `LOCALAI_JOB_TIMEOUT` | `600` | 单任务超时（秒） |
| `LOCALAI_JOBS_KEEP_UPLOADS` | `0` | 任务完成后是否保留上传文件 |
| `LOCALAI_JOBS_POLL` | `0.2` | 调度器空转轮询间隔（秒） |

### 12.3 `POST /jobs`（通用 JSON 提交）

```json
{"kind": "tts", "priority": 5, "payload": {"text": "你好", "voice": "default"}}
```

返回：

```json
{"ok": true, "id": "6f2d…", "state": "queued"}
```

`action` 可省略，按 kind 取默认值。`speaker` / `media` 也支持通过这里提交，
`payload.file` 必须是**服务端本地路径**（高级用法；常规上传请用下面的 multipart 入口）。

### 12.4 `POST /jobs/tts`（JSON）

```http
POST /jobs/tts
Content-Type: application/json

{"text":"你好","voice":"default","speed":1.0,"priority":0}
```

### 12.5 `POST /jobs/asr`、`POST /jobs/separate`（multipart）

```bash
curl -X POST http://127.0.0.1:7880/jobs/asr \
  -F "file=@hello.wav" -F "language=auto" -F "speaker=1" -F "priority=0"

# 任务级指定引擎 / 要时间戳（engine=auto 时选 whisper）
curl -X POST http://127.0.0.1:7880/jobs/asr \
  -F "file=@hello.wav" -F "engine=whisper" -F "timestamps=1"

curl -X POST http://127.0.0.1:7880/jobs/separate \
  -F "file=@song.mp3" -F "model=htdemucs" -F "stems=vocals,drums,bass,other"
```

`POST /jobs/asr` 新增可选表单 `engine=sensevoice|whisper|auto`、`timestamps=0|1`；
任务结果的 JSON 里同样带 `"engine"` 字段。通用 `POST /jobs` 的 `payload` 里也可以放
`engine` / `timestamps`。

上传文件先落到 `<HOME>/data/uploads/<job_id>/input.<ext>`；任务结束（done /
failed / canceled）后默认删除，`LOCALAI_JOBS_KEEP_UPLOADS=1` 可保留。

### 12.6 查询 / 取消 / 取结果

| 接口 | 说明 |
| --- | --- |
| `GET /jobs?state=&kind=&limit=&offset=` | 列表，创建时间倒序；`total` 为过滤后总数 |
| `GET /jobs/{id}` | 详情（含 `result` / `error`） |
| `POST /jobs/{id}/cancel` | queued 立即 `canceled`；running 标记 `cancel_requested`，跑完不写结果 |
| `GET /jobs/{id}/result` | done 后取结果；TTS 这类音频直接返回文件，其它返回 JSON |
| `GET /jobs/stats` | 队列统计 + 调度器状态 |

列表响应：

```json
{"ok": true, "total": 2, "limit": 50, "offset": 0,
 "jobs": [{"id": "…", "kind": "tts", "state": "done", "result": {"path": "…"}}]}
```

取消响应（running）：

```json
{"ok": true, "id": "…", "state": "running", "cancel_requested": true}
```

任务结束后的取消返回 409：

```json
{"ok": false, "id": "…", "state": "done", "error": "not_cancelable"}
```

### 12.7 `GET /jobs/stats`

```json
{
  "ok": true,
  "total": 6,
  "queued": 1,
  "running": 1,
  "done": 4,
  "failed": 0,
  "canceled": 0,
  "by_state": {"queued": 1, "running": 1, "done": 4, "failed": 0, "canceled": 0},
  "by_kind": {"tts": 4, "separate": 2},
  "queued_by_kind": {"separate": 1},
  "running_by_kind": {"tts": 1},
  "avg_wait_ms": 120.5,
  "avg_run_ms": 830.2,
  "oldest_queued_seconds": 12.3,
  "scheduler": {
    "current_kind": "tts",
    "batch_streak": 4,
    "loaded_model": "IndexTTS-2.5",
    "running_job_id": "…",
    "running_job_ids": ["…"],
    "running": 1,
    "max_parallel": 1,
    "per_kind_parallel": {"tts": 1, "asr": 1, "separate": 1, "speaker": 1, "media": 1},
    "asr_parallel": 1,
    "asr_engine": "sensevoice",
    "asr_engines": {
      "default": "sensevoice",
      "sensevoice": {"adapter": "asr", "loaded": false},
      "whisper": {"adapter": "whisper", "loaded": true, "model": "faster-whisper-small",
                  "device": "cuda", "compute_type": "int8_float16"}
    },
    "parallel_extra_mb": 600,
    "parallel_degraded": {},
    "concurrency": {
      "tts": {"name": "tts", "capacity": 1, "in_use": 1, "peak": 1, "reason": ""}
    },
    "timeout_seconds": 600.0,
    "recover_mode": "requeue",
    "paused": false
  }
}
```

- `by_kind` 是**全部历史任务**按 kind 计数（不是只统计排队中的）；
  需要看排队/运行中的分布用 `queued_by_kind` / `running_by_kind`；
- `avg_wait_ms` = 平均（started_at - created_at）；`avg_run_ms` = 平均
  （finished_at - started_at）；没有样本时为 0；
- `per_kind_parallel` 是**当前生效**的每类并行度（配置 + 能力限制 + OOM 降级后的值）；
- `parallel_degraded` 记录撞过 CUDA OOM、被临时降为 1 的 kind 及原因；
- `concurrency` 是每个能力推理闸门的实时状态（容量 / 占用 / 峰值）；
- **没有 `aging_seconds`**（老化逻辑已彻底删除）。

### 12.8 观测

- 日志（`logs/service.*.log`）每个任务一行开始/结束：
  `id / kind / action / model / 等待 ms / 执行 ms / 显存 free/total / 并发 n/N`；
- 切换 kind 时记一行：
  `调度切换：tts→separate（批次 5/2；同类队列已清空是唯一切换出口）`；
- 并行撞 OOM 时记一行：`显存不足降级：asr 同类并行临时降为 1 …`；
- `GET /health` 新增：

```json
"jobs": {"queued": 2, "running": 1, "current_kind": "tts"}
```

- `GET /capabilities` 新增：

```json
"jobs": {"enabled": true, "sync_endpoints_use_queue": true,
         "order": "kind_first > priority > created_at（无 aging）",
         "max_parallel": 1,
         "per_kind_parallel": {"tts": 1, "asr": 1, "separate": 1, "speaker": 1,
                               "media": 1, "translate": 1, "download": 1, "analyze": 1},
         "parallel_extra_mb": 600,
         "timeout_seconds": 600.0, "recover": "requeue"}
```

### 12.9 注意

- 同步旧接口现在也要经过同一个队列，因此**会在队列里留下记录**；
  并发调用时它们按调度规则排队，不再只靠能力锁。
- 同步接口等待上限 = 对应任务类型超时 + 30 秒（下载 / 分析用
  `LOCALAI_DOWNLOAD_TIMEOUT`，默认 1800 秒；其它能力仍是 600 秒）。
- 参数错误（格式/语言/模型/stems）仍然在入队前返回 400，不会产生任务记录。
- 声纹 `/speakers/*` 与媒体 `/media/*` 仍是直接调用、不入队（它们不是
  长任务，语义保持不变）。
- **饿死风险**：同类型持续涌入时，其它类型会一直排队（没有 aging 提权）。
  这是主人明确要求的取舍；调用方如果在意延迟，应自己控制提交节奏，
  或把任务拆到不同服务实例。

---

## 13. 翻译（NLLB-200）

新增的**轻量快速翻译**能力：`facebook/nllb-200-distilled-600M`（约 2.37 GB 权重，
GPU fp16 约占 1.2 GB 显存）。和 TTS / ASR 一样是**懒加载 + 空闲卸载 + 显存 LRU**：
服务启动 0 模型，第一次 `/translate` 才加载；空闲 900 秒（可配）自动卸载；
显存不够时和 TTS / whisper / Demucs 按 LRU 互相腾地方。

- 能力名：`translate`（`/capabilities` 里可发现）
- 引擎名：`nllb`（请求里的 `engine` 字段，目前唯一可选值）
- 权重目录：`LOCALAI_HOME/models/nllb/nllb-200-distilled-600M/`
  （可用 `LOCALAI_TRANSLATE_DIR` 改根目录）
- 安装日志：`logs/nllb_install.log`（每次下载尝试与失败原因都写在这里）

### 13.1 接口一览

| 接口 | 用途 | 是否入任务队列 |
| --- | --- | --- |
| `POST /translate` | 单条文本翻译（≤ 5000 字），长文本自动切分后拼回 | 否（直接调用） |
| `POST /translate/batch` | 一批文本（≤ 500 条 / ≤ 200k 字），内部批量推理 | 否（直接调用） |
| `POST /jobs/translate` | 超大批量进队列，立刻返回 job id | 是（`kind=translate`） |
| `GET /translate/status` | 模型是否就绪/已加载、空闲秒数、语言表、权重信息 | 否 |
| `GET /jobs/{id}/result` | 取 `/jobs/translate` 的结果 | — |

`/translate` 与 `/translate/batch` 不写任务库（短请求不留垃圾记录）；
需要持久化、排队、或单批超过 500 条时用 `/jobs/translate`。

### 13.2 单条：`POST /translate`

```bash
curl -X POST http://127.0.0.1:7880/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气不错，我们出去走走吧。","source":"zh","target":"en"}'
```

```json
{
  "ok": true,
  "engine": "nllb",
  "source": "zho_Hans",
  "target": "eng_Latn",
  "text": "The weather is fine today, let's go for a walk.",
  "elapsed_ms": 7509,
  "chunks": 1
}
```

- 第一次调用包含加载时间（本机约 5~8 秒），之后单条短句约 **0.23 秒**；
- `source` 可写 `auto`、常见别名（`zh`/`en`/`ja`…）或 NLLB 码（`zho_Hans`）；
- `target` 同上（不接受 `auto`）；
- `chunks` 是这条文本被切成了多少块（长文本会 >1）；
- 可选字段：`engine`（默认 `nllb`）、`max_len`（每块最大 token，默认 400）、
  `batch_size`（内部批大小，默认 8）。

### 13.3 批量：`POST /translate/batch`

```bash
curl -X POST http://127.0.0.1:7880/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["你吃饭了吗？","这个方案还得再讨论一下。","人工智能正在改变生活。"],
       "source":"zh","target":"en","batch_size":8}'
```

```json
{
  "ok": true,
  "engine": "nllb",
  "source": "zho_Hans",
  "target": "eng_Latn",
  "texts": [
    "Have you eaten?",
    "This plan needs further discussion.",
    "Artificial intelligence is changing life."
  ],
  "count": 3,
  "elapsed_ms": 269,
  "chunks": 3
}
```

- 返回顺序与 `texts` 严格一致；空字符串原样返回空字符串；
- 内部按 `batch_size` 分批跑 GPU，**比逐条调用快很多**（500 条约 31~36 秒）；
- 上限：单条 5000 字、单批 500 条、单批总量 200,000 字；
  超限返回 **400**（不是 422），说明里会告诉你超了多少、该改用哪个接口。

### 13.4 超大批量：`POST /jobs/translate`

```bash
curl -X POST http://127.0.0.1:7880/jobs/translate \
  -H "Content-Type: application/json" \
  -d '{"texts":["第一条……","第二条……"],"source":"zh","target":"en","batch_size":8,"priority":0}'
```

```json
{"ok": true, "id": "6f7e0bb4e249463da456e7252142fdbe", "state": "queued",
 "count": 500, "source": "zho_Hans", "target": "eng_Latn"}
```

结果：

```bash
curl http://127.0.0.1:7880/jobs/6f7e0bb4e249463da456e7252142fdbe/result
```

```json
{
  "ok": true,
  "id": "6f7e0bb4e249463da456e7252142fdbe",
  "state": "done",
  "result": {
    "ok": true,
    "engine": "nllb",
    "model": "nllb-200-distilled-600M",
    "source": "zho_Hans",
    "target": "eng_Latn",
    "texts": ["...", "..."],
    "count": 500,
    "chunks": 500,
    "elapsed_ms": 34000,
    "loaded": true
  }
}
```

- 调度器按「同类型优先」把连续的 translate 任务排在一起，模型只加载一次；
- 语言/长度等参数错误在**入队前**返回 400，不会产生失败任务；
- 单任务超时默认 600 秒（`LOCALAI_JOB_TIMEOUT`），500 条远低于上限。

### 13.5 状态：`GET /translate/status`

```json
{
  "ok": true,
  "ready": true,
  "loaded": true,
  "idle_seconds": 0.5,
  "idle_unload_seconds": 900,
  "model": "nllb-200-distilled-600M",
  "model_dir": "E:\\LocalAIService\\models\\nllb\\nllb-200-distilled-600M",
  "device": "cuda",
  "dtype": "float16",
  "fp16": true,
  "size_mb": 2367.6,
  "weights_source": "local",
  "missing": [],
  "load_seconds": 5.1,
  "load_error": null,
  "install_log": "E:\\LocalAIService\\logs\\nllb_install.log",
  "languages": [{"code": "zho_Hans", "name": "简体中文"}],
  "detail": "nllb-200-distilled-600M 权重已就绪（模型已加载）"
}
```

`/health` 与 `/capabilities` 也各有一块 `translate`（轻量字段）；
`/capabilities` 的 `translate.languages` 是完整语言表，插件可直接用它做下拉框。

### 13.6 语言码

入口同时接受三种写法：

1. **NLLB 码**（推荐，含义最明确）：`zho_Hans`、`eng_Latn`、`jpn_Jpan` …；
2. **常见别名**：`zh`、`zh-cn`、`en`、`ja`、`ko`、`fr`、`de`、`es`、`ru`、`ar` …；
3. **ISO 风格**：`zh-cn`、`eng-latn`（大小写与 `-`/`_` 不敏感）。

常用对照（完整 39 种见 `/translate/status` 的 `languages`）：

| NLLB 码 | 语言 | 常见别名 |
| --- | --- | --- |
| `zho_Hans` | 简体中文 | `zh` / `zh-cn` / `chinese` / `cn` |
| `zho_Hant` | 繁体中文 | `zh-tw` / `zh-hant` |
| `yue_Hant` | 粤语 | `yue` / `cantonese` |
| `eng_Latn` | 英语 | `en` / `english` |
| `jpn_Jpan` | 日语 | `ja` / `jp` |
| `kor_Hang` | 韩语 | `ko` / `kr` |
| `fra_Latn` | 法语 | `fr` |
| `deu_Latn` | 德语 | `de` |
| `spa_Latn` | 西班牙语 | `es` |
| `rus_Cyrl` | 俄语 | `ru` |
| `ara_Arab` | 阿拉伯语 | `ar` |
| `por_Latn` / `ita_Latn` / `nld_Latn` | 葡/意/荷 | `pt` / `it` / `nl` |
| `pol_Latn` / `tur_Latn` / `ukr_Cyrl` | 波/土/乌 | `pl` / `tr` / `uk` |
| `vie_Latn` / `tha_Thai` / `ind_Latn` | 越/泰/印尼 | `vi` / `th` / `id` |
| `hin_Deva` / `ben_Beng` / `tam_Taml` | 印地/孟加拉/泰米尔 | `hi` / `bn` / `ta` |
| `heb_Hebr` / `fas_Arab` / `urd_Arab` | 希伯来/波斯/乌尔都 | `he` / `fa` / `ur` |
| `swe_Latn` / `dan_Latn` / `nor_Latn` / `fin_Latn` | 北欧四语 | `sv` / `da` / `no` / `fi` |
| `ces_Latn` / `ron_Latn` / `hun_Latn` / `ell_Grek` / `bul_Cyrl` / `srp_Cyrl` | 捷/罗/匈/希/保/塞 | 对应两字母码 |

未知语言会返回 400，并在 `detail` 里列出**全部可用 NLLB 码与别名**。

### 13.7 `source=auto` 的边界

`auto` 用 `langdetect`（纯 Python，无额外模型下载）判断，再映射到 NLLB 码：

- 只覆盖上表里的**常见语种**；检测到表外语种会返回 400，要求显式给 `source`；
- 文本太短（一两个词）、纯数字/符号、或中英混排时检测可能不准；
- 一批文本只检测**第一条非空文本**，整批按同一源语言处理；
- 要稳定结果就显式给 `source`（尤其是术语、代码、商品名这类文本）。

错误示例：

```json
{"detail": "检测到的语言不在支持列表里（tl(78%)）；auto 只覆盖常见语种，请显式指定 source。可用 NLLB 码：……"}
```

### 13.8 长文本策略

一条请求里的长文本会这样处理：

1. **按段落切**：`\n` / `\r\n` 原样保留为分隔块，不送进模型；
2. **按句子切**：中文 `。！？；…`、英文 `. ! ? ;` 都认；英文句号只在后面是空白/结尾时切，
   不会切开 `3.14`；句尾的引号/括号归到同一句；
3. **超长句再切**：先按逗号/顿号/冒号等次级标点切，仍超过 `max_len` 的按字符二分硬切；
4. **每块 ≤ `max_len` token**（默认 400，真正用 tokenizer 计数）；
5. 翻译后**按原结构拼回**：段落间的换行原样保留；同一段内被切开的块用空格连接
   （中文/日文目标不加空格，避免句间多出空格）；硬切开的同一句用空字符串连接。

结构保证：段落数、空行数不丢；每个切块都会出现在结果里（内部有数量校验，
不一致会直接报错而不是悄悄丢句）。注意：**句子之间的原始空格**不会逐字保留
（由目标语言习惯决定连接符）。

### 13.9 实测（本机 RTX 3070 8GB / torch 2.11+cu130 / fp16）

**质量抽检**（默认 `num_beams=4`，人工阅读后给评语）：

| 方向 | 原文 | 译文 | 评语 |
| --- | --- | --- | --- |
| 中→英 | 今天天气不错，我们出去走走吧。 | The weather is fine today, let's go for a walk. | 通顺，口语自然 |
| 中→英 | 请把这份文件打印三份，下午开会要用。 | Please print this document in three copies for the afternoon meeting. | 完整，语序正确 |
| 中→英 | 该系统采用分布式架构，支持水平扩展。 | The system uses a distributed architecture to support level expansion. | 可读，但“水平扩展”译成 level expansion，术语不准 |
| 中→英 | 请在提交代码前运行单元测试，确保没有回归。 | Please run the unit test before submitting the code to make sure there is no return. | “回归”误译为 return，术语错 |
| 中→英 | 这家餐厅的招牌菜是红烧肉，味道很地道。 | The restaurant's signature dish is roasted meat, which tastes delicious. | 可读，“红烧肉”丢失 |
| 英→中 | Please print three copies of this document for the afternoon meeting. | 请为下午会议打印这份文件的三个副本。 | 完整通顺 |
| 英→中 | The system uses a distributed architecture and supports horizontal scaling. | 该系统采用分布式架构，支持水平扩展。 | 术语准确 |
| 英→中 | Sorry, I may have misheard you. Could you say that again? | 抱歉，我可能听不到你，你能再说吗？ | “misheard”误译成“听不到” |
| 英→中 | The restaurant's signature dish is braised pork, and it tastes authentic. | 这家餐厅的标志性菜是烤的猪肉， | 术语错（braised pork→红烧肉），句尾略缺 |
| 英→中 | Urban rail transit passenger flow peaks during the morning and evening rush hours. | 城市铁路运输乘客流量在凌晨和晚上高峰。 | “早晚高峰”译成“凌晨和晚上”，语义偏差 |
| 中→日 | 你吃饭了吗？我有点饿了。 | あなたは食べましたか? 私は少し腹が減りました。 | 自然 |
| 中→日 | 他昨天加班到很晚，今天状态不太好。 | 彼は昨日遅くまで働いていたので，今日も不調です。 | 基本准确 |
| 中→日 | 这家餐厅的招牌菜是红烧肉，味道很地道。 | このレストランの標識は赤い焼肉で，美味しい。 | “招牌菜”误成“標識”，术语错 |
| 中→日 | 该系统采用分布式架构，支持水平扩展。 | このシステムは分布式構造を採用し，レベル拡張をサポートします。 | 中文词“分布式/扩展”残留 |
| 中→日 | 如果有任何问题，请随时联系客服。 | 顧客サービスにご連絡ください。 | 条件从句丢失 |

总体：**日常口语、书面陈述句可读性较好**；**专业术语、文化专有名词
（红烧肉/招牌菜/回归/水平扩展）准确率一般**；小模型偶有漏译。适合“先看懂大意”，
不适合法务/医疗等要求逐字准确的场景。

**速度**（已加载后，batch_size=8）：

| 场景 | 配置 | 耗时 | 吞吐 |
| --- | --- | --- | --- |
| 单条短句 | beams=4 | 0.223~0.235 s（首次预热 0.66 s） | ~4.3 条/s |
| 20 句短句 | beams=1 | 0.792 s | 25.3 条/s |
| 20 句短句 | beams=2 | 1.098 s | 18.2 条/s |
| 20 句短句 | **beams=4（默认）** | 1.333 s | 15.0 条/s |
| 500 条，引擎直调 | beams=2 | 24.68 s | 20.3 条/s / 351 字/s |
| 500 条，引擎直调 | **beams=4（默认）** | 31.65 s | 15.8 条/s / 274 字/s |
| 500 条，`/translate/batch`（含首次加载） | beams=4 | 36.56 s | 13.7 条/s / 237 字/s |
| 500 条，`/jobs/translate`（排队到完成） | beams=4 | 34.07 s | 14.7 条/s |

“RTF 类似指标”：按**输入字符**算约 **237~274 字符/秒**（beam 2 时约 351）。
500 条与 20 条的单位吞吐接近，说明批量接口没有明显额外开销。

**长文本结构**（每段 327~336 字，3 段，空行分段）：

| 文档 | 输入 | 输出块数 | 段落（入→出） | 空行（入→出） | 耗时 | 结构 |
| --- | --- | --- | --- | --- | --- | --- |
| 长文 1（AI 行业） | 336 字 | 3 | 3 → 3 | 2 → 2 | 1.33 s | ✅ |
| 长文 2（公司故事） | 327 字 | 3 | 3 → 3 | 2 → 2 | 1.27 s | ✅ |
| 长文 3（睡眠研究） | 334 字 | 3 | 3 → 3 | 2 → 2 | 1.30 s | ✅ |

人工核对：每段都有对应译文，没有整段丢失；句子数量与语义基本对应
（个别小模型漏译属于质量问题，不是切分丢句）。

**显存与共存**（8GB 卡，桌面/驱动基础占用约 1.2 GB）：

| 状态 | 整卡空闲 | 说明 |
| --- | --- | --- |
| 加载前 | 7097 MB | 服务 0 模型 |
| 加载 NLLB 后 | 5913 MB | 增量 **1184 MB**（torch 已分配 1177 MB） |
| NLLB + whisper-small 同时加载 | 4949 MB | 可以共存（约 1.6 GB） |
| 再加载 TTS | 1221 MB | **LRU 自动卸掉 translate + whisper**，TTS 独占 |
| 再调 `/translate` | 5753 MB | **LRU 反过来卸掉 TTS**，翻译重新加载 |
| `/vram/unload {"all":true}` | 6965 MB | 全部卸载，回到基线 |
| 500 条批次后 | 5499 MB | 比加载后多 414 MB（PyTorch 保留缓存，非泄漏） |
| 卸载后 | 7031 MB | 相对基线差 66 MB，属统计噪声 |

结论：NLLB 可以和 whisper/ASR 这类小模型共存；和 TTS（6.2 GB）**不能同时**，
由显存管家按 LRU 自动二选一。`/vram` 里 translate 的估算会按实测校准
（初始 1500 MB，本机校准到 1200 MB）。

**稳定性**：500 条批次连续跑完无报错、无 OOM；进程显存前后差在噪声范围内；
重复调用与卸载/重载均正常。

### 13.10 默认参数与理由

| 参数 | 默认 | 理由 |
| --- | --- | --- |
| `num_beams` | **4** | 批量下 beams=2 会出现明显漏译（如“三份副本…下午开会”只译前半句），beams=4 能修正；速度只比 beams=2 慢约 19%（20 句 1.10s→1.33s），值得 |
| `batch_size` | **8** | 8GB 卡上 8 条一批吞吐已接近饱和，再大收益有限、显存峰值更高 |
| `max_len` | **400 token** | 覆盖绝大多数段落；超长会切块，不会截断丢内容 |
| fp16 | **开**（`LOCALAI_TRANSLATE_FP16=1`） | 3070 支持；显存减半、速度更快，实测质量无明显下降；CPU 自动回退 fp32 |
| 空闲卸载 | **900 秒**（`LOCALAI_TRANSLATE_IDLE_UNLOAD`） | 与其它能力一致；0 = 关闭 |
| 并发 | **1** | `transformers.generate` 未验证同模型并发安全；吞吐靠批内 batch_size，而不是多请求并发 |

### 13.11 错误与限制

| 情况 | HTTP | 说明 |
| --- | --- | --- |
| text 超 5000 字 / 批超 500 条 / 超 20 万字 | 400 | `detail` 说明超了多少、建议改用 `/jobs/translate` |
| 语言码不认识 / `auto` 检测失败 | 400 | `detail` 列出可用 NLLB 码与别名 |
| 引擎名不认识 | 400 | 目前只支持 `nllb` |
| 权重未下载且下载失败 | 503 | `detail` 含各下载源失败原因；看 `logs/nllb_install.log` |
| 显存不足（LRU 腾完仍不够） | 507 | 与其它能力一致，`released` 里是已卸载的模型 |
| 任务超时（队列） | 504 | 默认 600 秒 |

已知限制：

- NLLB-200-distilled-600M 是**小模型**：日常文本可读，专业术语/文化词/长难句准确率有限，
  且没有术语表/自定义词典；需要稳定术语时建议调用方后处理或换更大模型。
- 短而抽象的技术短语容易误译（例如“显存互相腾挪的验证”可能译成完全无关的句子），
  给足上下文会明显改善。
- 一次请求只按一种源语言处理；中英混排请拆成多条并分别给 `source`。
- 标点会跟随模型输出（中文常输出半角逗号/句号），不会强制转成全角。
- 不支持流式；长文本按块串行生成，首字延迟 = 整条耗时。
- `source=auto` 依赖 `langdetect`，对超短文本/代码/符号串不可靠。
- GPU 与 TTS 不能同时驻留（8GB 卡），切换时会有一次加载等待。

---

## 14. 下载与链接分析（yt-dlp + 直链兜底）

新增的通用能力：给一个 **http/https** 链接，就能下载视频/音频，并进一步拿到
**字幕时间轴 + 镜头边界**。不占显存、不加载模型，服务启动仍然是 0 模型。

- 能力名：`download`（`/capabilities`、`/admin/models` 里可发现）
- 引擎：`yt-dlp`（默认），普通直链自动兜底到 httpx 流式下载
- 下载目录：`LOCALAI_HOME/data/downloads/<YYYY-MM-DD>/`（已被 `.gitignore` 排除）
- 字幕：优先站点字幕（中文 > 英文），没有站内字幕时由 `analyze` 跑 ASR

### 14.1 接口一览

| 接口 | 用途 | 是否入任务队列 |
| --- | --- | --- |
| `POST /media/download` | 提交异步下载，立刻返回 job id | 是（`kind=download`） |
| `POST /media/download/sync` | 小文件同步下载（≤ 同步上限） | 否（线程池直跑） |
| `GET /media/download/status` | yt-dlp 版本 / aria2 / 目录 / 限额 / 当前任务数 | 否 |
| `POST /media/analyze` | 下载 + 字幕/ASR + 场景边界（同步等结果） | 内部走 `kind=analyze` |
| `POST /jobs/analyze` | 异步链接分析，立刻返回 job id | 是（`kind=analyze`） |
| `POST /jobs/{id}/cancel` | 取消下载/分析（运行中尽力而为） | — |

### 14.2 异步下载 `POST /media/download`

```bash
curl -X POST http://127.0.0.1:7880/media/download \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.bilibili.com/video/BV1UY4G6pEJg","quality":"720p","subtitles":true,"priority":0}'
```

```json
{"ok": true, "job_id": "0f4c...", "id": "0f4c...", "state": "queued"}
```

参数：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `url` | 必填 | 只允许 http/https |
| `quality` | `720p` | `720p` / `best` / `audio`；audio 转 m4a |
| `subtitles` | `true` | 尝试站内字幕，下不到不报错 |
| `cookies` | 空 | 可选 Netscape cookie 文件；B站不填时自动读取本机凭证 |
| `priority` | `0` | 越大越先（同类型优先 > 优先级 > 提交时间） |

执行中 `GET /jobs/{id}` 的 `meta.progress` 形如：

```json
{"status":"downloading","percent":42.3,"downloaded_bytes":3251200,
 "total_bytes":7680000,"speed":980000.0,"eta":5,"elapsed_seconds":3.3}
```

完成后 `result`：

```json
{"ok":true,"path":"...\\data\\downloads\\2026-11-08\\xxx.mp4","title":"...",
 "duration":431.268,"site":"BiliBili","subtitles":["...\\xxx.zh.srt"],
 "size_mb":7.8,"quality":"720p","elapsed_seconds":2.5}
```

### 14.3 同步下载 `POST /media/download/sync`

参数同上，但**没有 `priority`**。只允许不超过 `LOCALAI_DOWNLOAD_SYNC_MAX_MB`
（默认 200MB）的文件；超限返回 **413**，`detail` 建议改用 `/media/download`。

```json
{"ok":true,"path":"...mp4","title":"Big_Buck_Bunny_720_10s_1MB",
 "duration":10.0,"site":"Generic","subtitles":[],"size_mb":0.92,
 "quality":"720p","sync_limit_mb":200}
```

### 14.4 状态 `GET /media/download/status`

```json
{
  "ok": true,
  "yt_dlp": {"installed": true, "version": "2026.10.26", "path": "...\\yt_dlp\\__init__.py"},
  "aria2c": {"available": false, "path": null},
  "download_dir": "E:\\LocalAIService\\data\\downloads",
  "today_dir": "E:\\LocalAIService\\data\\downloads\\2026-11-08",
  "jobs": {"queued": 0, "running": 0, "analyze_running": 0},
  "limits": {"max_mb": 2048, "sync_max_mb": 200, "timeout_seconds": 1800.0,
             "qualities": ["720p", "best", "audio"]},
  "ssrf": {"schemes": ["http", "https"], "check_redirects": true}
}
```

### 14.5 链接分析 `POST /media/analyze`

```bash
curl -X POST http://127.0.0.1:7880/media/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.bilibili.com/video/BV1UY4G6pEJg","engine":"auto","max_frames":3}'
```

编排：**下载 →（站点字幕 | ASR 分段）→ 场景边界 → 可选抽帧**。

- `engine`：`whisper` / `sensevoice` / `auto`；只在站内没有可用字幕时使用；
  `auto` 在需要时间戳时选 `whisper`。
- `max_frames`：0（默认，不抽帧）~ 200。抽帧点均匀分布，并吸附到最近的镜头
  边界（2 秒内）。
- 返回结构：

```json
{
  "ok": true,
  "title": "TTS模型从夯到拉排行！9款热门语音模型，谁是真神？",
  "duration": 431.268,
  "path": "...\\data\\downloads\\2026-11-08\\xxx.mp4",
  "site": "BiliBili",
  "subtitles_from": "site",
  "subtitle_path": "...\\xxx.zh.srt",
  "segments": [{"start": 0.08, "end": 0.68, "text": "本期盘点"}],
  "scenes": [{"time": 6.4, "score": 1.0}],
  "frames": [{"path": "...\\outputs\\frames\\...\\frame_0001_71.878s.jpg", "time_sec": 71.878}],
  "scenes_json": "...\\outputs\\scenes\\....json",
  "asr": null,
  "download": {"size_mb": 7.8, "quality": "720p", "elapsed_seconds": 1.5},
  "elapsed_seconds": 6.7
}
```

`subtitles_from` 取值：`site`（站点字幕）或 `asr`（识别结果）。
`segments` 已保证升序、无重叠、`end > start`（与 `/asr/segments` 同一整理逻辑）。

### 14.6 限额与环境变量

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCALAI_DOWNLOAD_MAX_MB` | 2048 | 单文件大小上限；超限中止并删除半成品 |
| `LOCALAI_DOWNLOAD_SYNC_MAX_MB` | 200 | 同步接口上限，超过要求走异步 |
| `LOCALAI_DOWNLOAD_TIMEOUT` | 1800 | 下载/分析任务超时秒数（队列按 kind 使用） |
| `LOCALAI_BILI_COOKIE` | 本机 bili-duty 凭证 | B站 cookie JSON 路径 |
| `LOCALAI_DOWNLOAD_ALLOW_PRIVATE` | 空 | **仅测试**：1=允许私有地址；生产不要设 |

### 14.7 SSRF 策略

- 只允许 `http://` / `https://`；`file://`、`ftp://`、`data:` 等一律 400。
- 解析域名后拒绝：`127.0.0.0/8`、`10/8`、`172.16/12`、`192.168/16`、
  `169.254/16`、`::1`、`fc00::/7`、`fe80::/10`，以及其它非公网地址。
- 直链下载**不自动跟随重定向**：逐跳校验 `Location`，最多 5 跳。
- yt-dlp 下载前会复查探测到的最终页面地址；`webpage_url` 也按同一策略校验。
- 测试本机临时 HTTP 服务器时临时设 `LOCALAI_DOWNLOAD_ALLOW_PRIVATE=1`。

### 14.8 Cookie 说明

- `cookies` 参数接受 Netscape 格式文件路径，仅本次下载使用。
- 不传且 URL 属于 `bilibili.com` / `b23.tv` 时，服务读取本机
  `C:\Users\Administrator\.dsh\bili-duty\bili_credential.json`（可用
  `LOCALAI_BILI_COOKIE` 覆盖），生成**临时** Netscape 文件，用完立即删除。
- 字段名大小写不敏感：`SESSDATA` / `bili_jct` / `buvid3` / `DedeUserID` 等。
- **cookie 值不写日志、不进仓库**；`bili_credential.json` 已在 `.gitignore`。
- 其它站点不会自动携带任何 cookie，需要登录的内容请显式传 `cookies`。

### 14.9 队列、进度与取消

- `kind=download` / `kind=analyze` 与其它任务共用 SQLite 队列；排序仍是
  **同类型优先 > 优先级 > 提交时间**。
- 下载 / 分析各自固定同类并行 1（同站请求克制，不并发抓取）。
- 运行中的取消：进度钩子每个检查点查一次取消标记，命中就停；已下完的
  文件保留，半成品 `.part/.ytdl` 删除。服务重启后 running 任务按
  `LOCALAI_JOBS_RECOVER`（默认 requeue）重新排队。

### 14.10 实测数据（2026-11-08，本机 8GB 卡）

| 场景 | 结果 |
| --- | --- |
| B站 `BV1UY4G6pEJg`（720p） | yt-dlp **2026.10.26**；耗时 **2.5s**；**7.8MB**；时长 431.268s；站内中文字幕 `xxx.zh.srt`；路径 `data\downloads\2026-11-08\...mp4` |
| 一般站 `test-videos.co.uk`（10s mp4） | 耗时 **2.7s**；**0.92MB**；时长 10.0s；`site=Generic` |
| 直链（本机临时 HTTP 服务器，ffmpeg testsrc 5s） | 落盘成功；**0.07MB**；probe 时长 **5.0s**；`site=127.0.0.1` |
| analyze 端到端（B站，max_frames=3） | 总耗时 **6.7s**；`subtitles_from=site`；**209** 段；**2** 个场景边界；**3** 张帧 |
| SSRF `http://127.0.0.1:...` | 拒绝：`DownloadBlocked`（SSRF 防护） |
| SSRF `file:///C:/x.mp4` | 拒绝：只允许 http/https |
| 超限（2MB 文件，上限 1MB） | 中止并删除 `.part`，无残留 |

B站字幕抽查（与音频内容一致）：

| 时间 | 文本 |
| --- | --- |
| 0.08~0.68 | 本期盘点 |
| 0.68~2.40 | 我自制的9款主流TTS |
| 2.40~3.70 | 语音生成整合包 |

### 14.11 已知限制

- **B站风控**：无 cookie 或 cookie 过期时可能返回 403 / 需要验证码；不要反复
  重试，更新 `bili_credential.json` 后再试。
- **yt-dlp 需要常更新**：站点改版后旧版本会失效。升级：
  `engine\.venv\Scripts\python -m pip install -U yt-dlp`；WebUI「系统」页可看版本。
- 站点字幕是**站方提供**的，时间轴/断句质量由站方决定；没有字幕时才跑 ASR，
  速度取决于所选识别引擎。
- 直链兜底只针对 yt-dlp 明确报 `Unsupported URL` 的情况，避免把登录页/HTML
  当视频保存。
- 不做站内爬取 / 批量抓取，一次只下载用户明确给出的一个公开链接。

---

## 15. 视觉 / 关键帧挑选（Qwen2.5-VL-3B）

本地小视觉模型（VLM）+ 「按字幕和时长循环抽帧，直到凑够 N 张有效关键帧」的代理。
启动仍然 0 模型：第一次视觉请求才拉起 ``llama.cpp`` 的 ``llama-server.exe``
子进程（只监听 ``127.0.0.1:7891``），和 TTS / ASR / 翻译一样由显存管家按 LRU
互卸；空闲 ``LOCALAI_VISION_IDLE_UNLOAD``（默认 900 秒）后自动停掉。

### 15.1 接口一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/vision/describe` | 单张图片理解（multipart），默认用单帧评分提示词 |
| POST | `/vision/scan` | 多图批量：multipart 上传或 JSON 路径列表；`task=score/caption/ocr` |
| POST | `/vision/frames` | 按字幕 + 场景 + VLM 循环抽帧，凑够 N 张 |
| POST | `/jobs/vision` | 异步版 `/vision/frames`（kind=vision，可查进度/取消） |
| GET | `/vision/status` | 模型、进程、显存估算、默认 planner、提示词文件 |
| POST | `/media/analyze` | 旧接口新增可选 `filter="vision"` + `target_frames` |

### 15.2 `POST /vision/describe`

multipart，字段名固定 ``file``；可选 ``prompt``（覆盖用户消息）与 ``time``（秒）。
系统提示词固定读 ``docs/prompts/vlm_frame_score.md``，所以返回通常是评分 JSON。

```powershell
curl.exe -s -X POST http://127.0.0.1:7880/vision/describe `
  -F "file=@outputs\frames\xxx\frame_0002_215.600s.jpg"
```

```json
{
  "ok": true,
  "text": "```json\n{\n  \"usable\": false,\n  \"kind\": \"blank\",\n  \"score\": 1,\n  \"ocr_text\": \"\",\n  \"summary\": \"视频片段空白，无信息\",\n  \"why\": \"视频片段空白，无信息\"\n}\n```",
  "parsed": {"usable": false, "kind": "blank", "score": 1, "ocr_text": "", "summary": "视频片段空白，无信息", "why": "视频片段空白，无信息"},
  "elapsed_ms": 985,
  "model": "Qwen2.5-VL-3B-Instruct-Q4_K_M",
  "prompt_file": "docs/prompts/vlm_frame_score.md"
}
```

- 只接受真正的图片（JPEG/PNG/WebP/GIF/BMP，按文件头判断），其它一律 400；
- 上传文件在处理完立即删除；
- 模型偶尔会包 markdown 围栏，服务端同时返回 `parsed`（容错解析结果）。

### 15.3 `POST /vision/scan`

两种请求体：

```powershell
# 1) multipart 多图（字段名随意，可多张）
curl.exe -s -X POST http://127.0.0.1:7880/vision/scan -F "task=score" -F "file=@a.jpg" -F "file=@b.jpg"

# 2) JSON 路径列表（服务端本机文件）
curl.exe -s -X POST http://127.0.0.1:7880/vision/scan -H "Content-Type: application/json" `
  -d "{\"task\":\"score\",\"images\":[\"E:/a.jpg\",\"E:/b.jpg\"]}"
```

返回 ``{ok,task,count,results:[{index,path,text,parsed}],elapsed_ms,model}``。
``task`` 决定提示词：`score`=评分 JSON、`caption`=一句话描述 JSON、`ocr`=纯文字。
一次最多 16 张；模型只加载一次，逐张推理（llama-server 单 slot）。

### 15.4 `POST /vision/frames`

```json
{
  "video_path": "E:/LocalAIService/data/downloads/2026-11-08/xxx.mp4",
  "target_frames": 3,
  "planner": "hybrid",
  "max_rounds": 6,
  "subtitles": [{"start": 0.08, "end": 0.68, "text": "本期盘点"}],
  "scene_times": [1.52, 3.04]
}
```

``video_path`` 与 ``job_id`` 二选一（``job_id`` 会从任务结果的 ``path`` 取视频）。
``subtitles`` / ``scene_times`` 可选：不给就找同目录站点字幕（``*.zh.srt`` 等），
没字幕再跑 ``/asr/segments`` 同款分段识别；场景点没给就现场 ``/media/scenes``。

响应：

```json
{
  "ok": true,
  "video": "...",
  "duration": 431.267,
  "target": 3,
  "rounds_used": 1,
  "frames": [
    {"time": 5.085, "path": "...", "score": 4, "kind": "chart",
     "ocr_text": "...", "summary": "...", "why": "..."}
  ],
  "stats": {"candidates": 12, "vlm_calls": 8, "blank_skipped": 0,
            "duplicates_skipped": 4, "elapsed_ms": 20653,
            "planner": "hybrid", "max_rounds": 6, "subtitles": 209},
  "subtitles_from": "site"
}
```

字段：``target`` 1~10；``planner`` ``hybrid``（默认）/``agent``；``max_rounds`` 1~12。
凑不够时如实少给，并带 ``detail``/``stats.shortfall``；VLM 不可用时带
``degraded=true`` + ``degraded_reason``，帧仍来自确定性回退。

**两种 planner 的区别**

| | `hybrid`（默认） | `agent` |
| --- | --- | --- |
| 规划 | 字幕关键词 + 场景点 + 均匀补充（确定性） | 先用 `vlm_frame_plan.md` 让模型规划，再按代理协议循环 |
| 评分 | 每帧一次 `vlm_frame_score.md`，取 `usable && score>=3` 的前 N | 同左；模型每轮输出 `{"action":"extract","times":[...]}` 或 `{"action":"final","frames":[...]}` |
| 轮数 | 分数不够就补抽 4~8 个新点，最多 `max_rounds` | 模型自己决定下一批点；`round>=max_rounds` 强制结束 |
| 速度 | 快、可控 | 多几次纯文本调用，适合复杂讲解视频 |
| 解析容错 | 去 markdown 围栏 + 第一个 `{` 到最后一个 `}` + 容忍尾逗号 | 同左 |

**算法（两条 planner 共用）**

1. ffprobe 取时长；字幕按「调用方 > 同目录站点字幕 > ASR」；
2. 生成候选点（字幕关键词位 > 场景切换点 > 均匀补充，点间距 ≥3 秒）；
3. 先抽 8~16 张，做**感知哈希去重**（dHash 汉明距离，默认阈值 6）+ **亮度/方差预筛**
   （近纯色、低方差直接判 `blank`，不调 VLM）；
4. 对非空白帧调 VLM 评分；同一张幻灯片按「感知哈希 / OCR 相似度 / summary」
   只留一张；
5. 不够就补抽；VLM 连续失败两次提前终止，退回方差 + 场景点的确定性结果。

抽帧宽度固定 768px（够看清幻灯片正文，又把视觉 token 控制在 ~1000）。
结果图片落在 ``outputs/frames/vision_<时间戳>_<随机>/``。

### 15.5 `POST /jobs/vision`

请求体同 `/vision/frames`，额外 ``priority``；提交后轮询 ``GET /jobs/{id}``。
``kind=vision`` 在队列里固定同类并行 1；进度写 ``meta.progress``（stage/percent），
``POST /jobs/{id}/cancel`` 在下一轮检查点退出。

```powershell
curl.exe -s -X POST http://127.0.0.1:7880/jobs/vision -H "Content-Type: application/json" `
  -d "{\"video_path\":\"E:/.../a.mp4\",\"target_frames\":3,\"planner\":\"hybrid\"}"
```

### 15.6 `/media/analyze` 的 `filter="vision"`

```json
{"url": "https://...", "filter": "vision", "target_frames": 3}
```

在「下载 -> 字幕/ASR -> 场景边界」之后，直接复用当前字幕与场景点挑关键帧，
结果里多出：

```json
"vision_frames": [{"time": 5.085, "path": "...", "score": 4, "kind": "chart", "...": "..."}],
"vision": {"planner": "hybrid", "rounds_used": 1, "vlm_calls": 8,
           "degraded": false, "degraded_reason": null}
```

不传 ``filter``（或传空）时行为与旧版完全一致，``frames`` 仍是原来的均匀抽帧。

### 15.7 `GET /vision/status`

```json
{
  "ok": true, "ready": true, "loaded": false,
  "model": "Qwen2.5-VL-3B-Instruct-Q4_K_M",
  "runtime": "llama.cpp", "runtime_version": "b12202", "port": 7891,
  "model_size_mb": 1840.5, "mmproj_size_mb": 805.6,
  "vram_estimate_mb": 3400, "planner_default": "hybrid",
  "prompts": {"dir": "...", "score": true, "agent": true, "plan": true},
  "endpoints": ["/vision/describe", "/vision/scan", "/vision/frames",
                "/vision/status", "/jobs/vision"],
  "limits": {"target_frames": [1, 10], "max_rounds": [1, 12]}
}
```

该接口**不会加载模型**：只读文件与进程状态。

### 15.8 模型、显存与降级

| 项目 | 值 |
| --- | --- |
| 路线 | llama.cpp 子进程（Vulkan 构建）+ Qwen2.5-VL-3B-Instruct GGUF |
| 文本模型 | `models/vision/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`（约 1.84GB） |
| 视觉投影 | `models/vision/mmproj-Q8_0.gguf`（约 0.81GB；由 F16 用 llama-quantize 压到 Q8） |
| 运行时 | `models/vision/llama-server.exe` + DLL（约 37MB zip，解压后 ~100MB） |
| 显存 | 权重约 2.65GB + 上下文/计算缓冲，整卡增量约 **3.4GB**（≤3.5GB 目标） |
| 单帧耗时 | 预热后约 **0.8~2 秒**（768px，400 token 输出） |
| 空闲卸载 | `LOCALAI_VISION_IDLE_UNLOAD`（默认 900，0=关） |
| 端口 | `LOCALAI_VISION_PORT`（默认 7891，只监听 127.0.0.1） |

- 安装：``POST /admin/models/download {"capability":"vision"}``，或把上面三个文件
  手动放进 ``models/vision/``；``localai doctor`` 会显示版本与缺失文件。
- 8G 卡上 VLM 与 TTS 不能同时驻留：加载前显存管家会按 LRU 卸掉最久没用的模型；
  视觉模型在独立子进程里，卸掉即释放，不会留下 torch 碎片。
- **降级**：模型缺失/进程崩溃/请求超时/VLM 全部失败时，``/vision/frames`` 返回
  ``degraded=true`` + 方差筛选结果；``/vision/describe`` 返回 503（``vision_error``）。
  进程不会因为视觉失败而崩；下一次请求会自动重拉 llama-server。

**相关环境变量**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCALAI_VISION_DIR` | `<HOME>/models/vision` | 二进制 + GGUF 目录 |
| `LOCALAI_VISION_PORT` | 7891 | llama-server 端口（仅回环） |
| `LOCALAI_VISION_CTX` | 4096 | 上下文长度（agent 协议需要放完整提示词） |
| `LOCALAI_VISION_GPU_LAYERS` | 99 | 卸载到 GPU 的层数 |
| `LOCALAI_VISION_TIMEOUT` | 120 | 单次推理超时秒数 |
| `LOCALAI_VISION_IDLE_UNLOAD` | 900 | 空闲卸载秒数，0=关 |
| `LOCALAI_VISION_PLANNER` | hybrid | 默认 planner |
| `LOCALAI_VISION_PHASH_DISTANCE` | 6 | 感知哈希去重阈值 |
| `LOCALAI_VISION_MODEL` / `_MMPROJ` / `_SERVER` | 见上表 | 覆盖文件名 |

### 15.9 实测（RTX 3070 8GB，2026-11-08）

| 场景 | 结果 |
| --- | --- |
| 431 秒 B 站视频（209 段站点字幕），hybrid，target=3 | **3 帧 / 1 轮 / VLM 8 次 / 20.7s**；选点 **5.085s、163.725s、224.265s** |
| 同上，agent，target=3 | **3 帧 / 2 轮 / VLM 14 次 / 27.1s**（先按 `vlm_frame_plan.md` 规划，再走 `extract` 补抽与 `final` 决策；选点与 hybrid 相同：5.085 / 163.725 / 224.265） |
| 10 秒短视频（Big Buck Bunny，无音轨），hybrid，target=2 | **2 帧 / 1 轮 / VLM 3 次 / 9.5s**，无死循环；ASR 因无音轨失败后按无字幕继续 |
| 空章节页 `/vision/describe` | `usable=false`、`kind=blank`、`score=1` |
| 显存 | 加载后整卡 used 约 **4.6GB**（基线约 1.2GB，增量约 **3.4GB**）；卸载后回落到基线 |
| 二次复用 | 同一进程第二次挑帧不再加载模型，llama-server 保持常驻，单帧约 1 秒 |

**人工核对（431 秒视频，target=3）**

- 旧均匀抽帧：71.9s（RTF 排行图）/ 215.6s（**空章节页**）/ 359.4s（总结页）；
- 新 hybrid：5.085s（工具总览 UI）、163.725s（「同一句词，四种语气」对比页）、
  224.265s（商用许可清单，正文完整）。空章节页被排除，后两张明显比均匀抽帧有信息量；
  第一张是片头工具总览，信息量一般但不再是纯章节页/转场。

### 15.10 已知限制

- 3B 小模型会**幻觉 OCR**：字幕提示词在场时，偶尔把字幕内容写进 `ocr_text`；
  关键帧本身仍按画面筛，建议调用方把 `ocr_text` 当辅助信息。
- Qwen2.5-VL 的视觉投影层用 Q8 量化以压到 3.5GB 显存，极端小字 OCR 可能弱于 F16；
  需要更高精度可设 `LOCALAI_VISION_MMPROJ=mmproj-F16.gguf`（显存约 +0.5GB）。
- 首次在新机器上跑视觉图会现场编译 Vulkan shader（约 8~10 秒），之后进入
  1~2 秒/帧；服务启动时会自动做一次小图预热。
- `agent` planner 的质量取决于小模型是否按协议输出 JSON；解析失败会退回
  `hybrid` 的确定性补抽，不会卡死。
- 一次最多 10 张关键帧、12 轮；超过请分多次或先缩小视频区间。
