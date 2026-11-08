# 第二层：DSH 插件契约（PLUGIN CONTRACT）

> 状态：**只定契约，本仓库不实现插件**。
> 未来插件放在独立仓库/独立目录，通过 HTTP 调本服务层。
> 本文件是插件作者与「本地 AI 服务层」之间的约定，改动前先改这里。

---

## 0. 边界：插件不碰模型

硬性约束：

1. 插件**只通过 HTTP** 调用服务层；**禁止** `import torch`、`import funasr`、
   `import demucs`、`import index_tts`，也**禁止**把模型代码复制进插件。
2. 插件**不负责**模型加载/卸载。所有加载都发生在服务层：第一次请求按需加载，
   空闲超时自动卸载。插件不应尝试“预热”或“常驻”。
3. 插件**不读**服务层的模型目录、不解析权重文件；需要知道能力/模型版本时
   调 `GET /capabilities`。
4. 服务层不可用时，插件必须**降级而不是阻塞/崩溃**（见第 3 节）。
5. 插件不持久化服务端路径（`X-Audio-Path` 等）作为长期引用；输出文件可能被清理。

一句话：**服务层负责“有没有、能不能、占多少显存”；插件负责“什么时候用、怎么呈现”。**

---

## 1. 启动与能力发现

### 1.1 配置

插件只需要一个配置项：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:7880` | 服务层地址 |
| `token` | 空 | 与 `LOCALAI_TOKEN` 对应；空=不鉴权 |
| `enabled` | true | 总开关 |
| `requestTimeoutMs` | 180000（TTS）/300000（ASR、分离） | 见第 3.4 节 |

### 1.2 发现

插件启动时（或每 60 秒）调用：

```http
GET /capabilities
```

必须校验：

- `api_version`：插件支持的契约版本。当前为 `"1"`。
  若主版本不认识 → 插件进入「只读/禁用」状态并提示用户升级，不要瞎调。
- `capabilities.tts.ready` / `capabilities.asr.ready` 等：只有 `ready=true` 才展示对应功能。
- `auth.required`：为 true 时所有非豁免请求带 `Authorization: Bearer <token>`。
- `listen.base_url`：可用于展示/日志，但不要用它覆盖用户配置（反代场景可能不同）。

建议缓存能力结果 30~60 秒；不要每个按键都打一次。

### 1.3 健康

```http
GET /health
```

- `ok`：TTS 模型就绪（engine 后端）。
- `vram.free_mb` / `vram.total_mb`：轻量（nvidia-smi，1 秒缓存）。
- `asr/speaker/separate.loaded`：当前是否占用显存。

健康检查失败（连接拒绝/超时 2 秒）→ 视为「服务不在」，进入降级。

### 1.4 显存

```http
GET /vram
```

- 需要判断“现在能不能合成/分离”时用它：看 `free_mb` 与目标能力的 `estimate_mb`。
- 不要自己计算显存；不要因为 `loaded=false` 就认为不可用（懒加载是正常的）。
- 需要主动让路时：`POST /vram/unload {"models":["tts"]}` 或 `{"all":true}`。
  插件**不应**在用户没操作时随意卸载（会拖慢下一次请求）。

---

## 2. 场景一：勾选后自动朗读

### 2.1 触发

插件在 DSH 里勾选「自动朗读」后，对新增的助手消息调用 TTS。
建议：**只朗读最终回复，不朗读工具调用/思考过程/代码块**（或按用户设置过滤）。

### 2.2 接口选择

优先：

```http
POST /tts
Content-Type: application/json

{"text":"要朗读的文本","voice":"default","speed":1.0,"language":"zh"}
```

返回 `audio/wav` 二进制，直接播放/缓存。

备选（旧插件兼容）：

```http
POST /api/v1/tts/tasks
{"text":"...","prompt_audio":"default","speed":1.0}
```

两者语义相同；新插件用 `/tts`，因为字段更明确（`voice`/`emotion`/`seed`）。

### 2.3 长文本分段

服务端单次上限 **5000 字**。插件负责分段：

- 优先按段落/句号/换行切，单段建议 ≤ 300 字（播放更跟手，失败重试代价小）；
- 不要把一句话从中间切断；
- 段与段之间串行请求（同一能力在服务端本来就是串行锁），或最多 2 路并发；
- 每段可复用同一个 `voice`，`seed` 固定可让音色更稳定（可选）；
- 用户停止朗读时，取消后续段落请求（HTTP 客户端 abort），不要继续堆积。

### 2.4 并发与超时

| 参数 | 建议值 |
| --- | --- |
| 同时进行的 TTS 请求 | ≤ 2（服务端串行，多了只会排队） |
| 连接超时 | 3~5 秒 |
| 单段读取超时 | 180 秒（首次加载可能 10~60 秒） |
| 重试 | 仅对 503/504/网络错误重试 1 次，间隔 1~2 秒；400/404/507 不重试 |

507 表示显存不足：不要循环重试；提示用户「显存不足」或先 `/vram/unload`。

### 2.5 失败降级

- 服务不在（连接拒绝/超时）→ **静默跳过朗读**，消息照常显示，不弹错误窗；
- 单段失败 → 跳过该段，继续后面的段；连续 2 段失败 → 停止本次朗读；
- 插件应在状态栏/设置页显示「语音服务未启动」，而不是打断对话；
- 不允许因为 TTS 失败而阻塞消息流或重发消息。

### 2.6 播放

- 响应是完整 WAV（16bit PCM），可以直接写临时文件播放；
- 音频可能带 `X-Audio-Path`（服务端留档路径），**不要依赖它长期存在**；
- 缓存建议按 `sha256(text+voice+speed+language)` 做内存/磁盘 LRU，别永久堆积。

---

## 3. 场景二：语音条转文字

### 3.1 接口

```http
POST /asr?speaker=0
Content-Type: multipart/form-data

file=<音频/视频>
language=auto
```

别名：`POST /api/v1/asr`（行为完全一致）。

### 3.2 上传要求

- 支持容器：wav / mp3 / flac / ogg / opus / m4a / aac，
  以及 mp4 / mkv / mov / webm 等视频（服务端用 ffmpeg 抽音轨）；
- 大小上限：**200MB**（超过返回 413 或 400，视入口而定）；
- 采样率/声道：**不需要插件转码**。服务端统一转 16kHz 单声道再识别；
- 如果是超长录音，建议插件先按静音/时长切片（如 10 分钟一段），
  避免单次请求过大；服务端本身支持长音频但耗时线性增长。

### 3.3 声纹字段

- `?speaker=1`（默认）：返回 `speaker: {name, score, known}`；
  首次会额外加载 campplus（约 0.3GB 增量），可能多等几秒。
- `?speaker=0`：不加载 campplus，响应里 `speaker=null`。
- 用户没登记声纹时 `speaker` 可能是 `未知-N` 或 `null`；插件不要把 `known=false` 当错误。
- 声纹失败只影响 `speaker`/`speaker_error`，**文字照常返回**。

### 3.4 响应与展示

```json
{"ok":true,"text":"...","language":"zh","duration":12.3,"elapsed_ms":850,
 "speaker":{"name":"主人","score":0.81,"known":true},"converted":true}
```

- 以 `text` 为准；`language` 可用于展示；
- `converted=true` 表示服务端做了转码，不影响结果；
- 空文本是合法结果（静音/噪声），插件展示「（未识别到语音）」，不要报错。

### 3.5 失败降级

- 服务不在 → 语音条仍保留，显示「转写服务未启动」，允许重试；
- 400 → 展示具体原因（格式不支持/太大/语言不支持）；
- 507 → 提示显存不足，建议关闭其它 GPU 程序；
- 超时（建议 300 秒）→ 提示可重试，不要自动无限重试。

---

## 4. 场景三：声音选择与音色管理

### 4.1 列出音色

```http
GET /voices
```

返回 `{voices:[{name, path, filename, format, duration}], count, directory}`。

- `name` 是合成时 `voice` 字段该传的值；
- 音色目录每次请求重新扫描，新上传的文件马上可见；
- 插件只展示 `name`/`duration`，不要把绝对路径写进 UI（可能含用户名）。

### 4.2 上传音色

```http
POST /api/v1/upload
Content-Type: multipart/form-data

file=@my_voice.wav
```

- 支持 wav/mp3/flac/ogg/opus/m4a/aac，≤50MB；
- 同名文件会覆盖（服务端按文件名保存）；
- 返回 `{ok, name, filename, size}`；之后用 `name` 作为 `voice`。

也可以让用户把文件放进 `<HOME>/voices/`；插件调 `/voices` 刷新即可。

### 4.3 声纹（可选功能）

| 操作 | 接口 |
| --- | --- |
| 列出已登记 | `GET /speakers` |
| 登记 | `POST /speakers/enroll`（multipart `file` + `name`） |
| 识别 | `POST /speakers/identify`（multipart `file`） |
| 未知聚类 | `GET /speakers/unknown` |
| 删除 | `DELETE /speakers/{name}` |

注意：

- 声纹只存 192 维向量与名字，不存原音频；插件不要把原音频再传回服务端长期保存；
- `name` 会被用作文件名，服务端会做安全化；插件仍应限制长度与非法字符；
- 登记建议 3~10 秒清晰单人语音；同名重复登记会累加样本。

### 4.4 默认音色

- 插件应保存用户选择的音色名；服务端没有“全局默认音色”状态。
- 若保存的音色被删除，合成会 404 `voice_not_found`；插件应回落到 `/voices` 第一个并提示。

---

## 5. 版本协商

- 服务端在 `/capabilities` 返回 `api_version`（当前 `"1"`）和 `service.version`（如 `"2.2.0"`）。
- 插件只依赖 `api_version` 的主版本；`service.version` 仅用于展示/日志。
- 新增字段是兼容变更，插件必须忽略未知字段。
- 若 `api_version` 不认识：
  - 不要调用写接口（TTS/上传/声纹登记）；
  - 可继续调用 `/health` 做状态展示；
  - 提示用户升级插件或服务。

---

## 6. 错误处理约定

| 场景 | 服务端 | 插件行为 |
| --- | --- | --- |
| 参数错误 | 400 | 提示用户，修正后重试；不自动重试 |
| 鉴权失败 | 401 | 提示检查 `token` 配置；不重试 |
| 音色不存在 | 404 | 刷新 `/voices`，回落默认音色 |
| 文件过大 | 413 | 提示压缩/裁剪 |
| 服务/引擎失败 | 503 | 指数退避重试 1 次；仍失败则降级 |
| 上游超时 | 504 | 提示重试；不要无限重试 |
| 显存不足 | 507 | 提示关闭 GPU 程序或 `/vram/unload`；不重试 |

统一读取：

- 普通接口：`detail` / `error`；
- `/v1/*`：`error.message`、`error.code`。

日志里不要打印完整 token、完整音频 base64、用户隐私文本。

---

## 7. 鉴权预留

服务端通过环境变量 `LOCALAI_TOKEN` 开启（默认关闭）：

```http
Authorization: Bearer <token>
```

- 豁免：`/health`、`/capabilities`、`/docs`、`/openapi.json`；
- 插件把 token 存在自己的安全配置里，不要写进日志/URL；
- 未开启时不要主动发空 Authorization 头。

---

## 8. 插件不该做的事

- ❌ 自己加载/卸载模型、自己管理显存；
- ❌ 直接读 `models/`、`speakers/`、`outputs/` 目录；
- ❌ 假设服务一定在 `127.0.0.1:7880`（应可配置）；
- ❌ 在服务不可用时阻塞 UI 或反复重试；
- ❌ 把服务端返回的绝对路径当作长期文件引用；
- ❌ 调用未在 `/capabilities` 里声明为 `ready` 的能力。

---

## 9. 最小实现清单（未来插件）

1. `GET /capabilities` → 校验 `api_version`、缓存能力；
2. 自动朗读：`POST /tts`，分段、串行、180s 超时、失败静默跳过；
3. 语音转写：`POST /asr`（大文件分片、`?speaker=0` 可选）；
4. 音色管理：`GET /voices`、`POST /api/v1/upload`；
5. 健康/显存：`GET /health`、`GET /vram`；
6. 错误映射：400/401/404/413/503/504/507；
7. 配置：`baseUrl`、`token`、超时、总开关；
8. 日志脱敏。
