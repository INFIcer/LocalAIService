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
可选名字：`tts` / `asr` / `speaker` / `separate`；未知名字返回 400。

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

### `POST /asr`，别名 `POST /api/v1/asr`

multipart：

| 字段 | 说明 |
| --- | --- |
| `file` | 音频或视频（wav/mp3/m4a/ogg/flac/mp4/mkv/webm…，≤200MB） |
| `language` | `auto`/`zh`/`en`/`ja`/`ko`/`yue`/`nospeech` |
| `?speaker=0` | 关闭声纹（默认开启；关闭后连 campplus 都不加载） |

非 WAV 会先用 ffmpeg 归一成 **16kHz 单声道 WAV** 再识别；采样率不用自己处理。
音频容器转码失败会回退直接识别；视频转码失败直接报错。

响应：

```json
{
  "ok": true,
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

声纹失败不会影响转写，会多一个 `speaker_error` 字段。

```bash
curl -X POST "http://127.0.0.1:7880/asr?speaker=0" \
  -F "file=@meeting.mp4" -F "language=auto"
```

### `GET /asr/status`

`{ready, loaded, idle_seconds, idle_unload_seconds, detail}`。

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

curl -X POST http://127.0.0.1:7880/jobs/separate \
  -F "file=@song.mp3" -F "model=htdemucs" -F "stems=vocals,drums,bass,other"
```

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
         "per_kind_parallel": {"tts": 1, "asr": 1, "separate": 1, "speaker": 1, "media": 1},
         "parallel_extra_mb": 600,
         "timeout_seconds": 600.0, "recover": "requeue"}
```

### 12.9 注意

- 同步旧接口现在也要经过同一个队列，因此**会在队列里留下记录**；
  并发调用时它们按调度规则排队，不再只靠能力锁。
- 同步接口等待上限 = `LOCALAI_JOB_TIMEOUT + 30` 秒。
- 参数错误（格式/语言/模型/stems）仍然在入队前返回 400，不会产生任务记录。
- 声纹 `/speakers/*` 与媒体 `/media/*` 仍是直接调用、不入队（它们不是
  长任务，语义保持不变）。
- **饿死风险**：同类型持续涌入时，其它类型会一直排队（没有 aging 提权）。
  这是主人明确要求的取舍；调用方如果在意延迟，应自己控制提交节奏，
  或把任务拆到不同服务实例。

