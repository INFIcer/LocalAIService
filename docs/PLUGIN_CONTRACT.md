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

### 3.6 选引擎与要时间戳（新增）

服务现在有两个识别引擎，**普通语音条转文字不需要改任何代码**（默认仍是 sensevoice，
中文最快、带声纹）。只有下面几种情况才需要显式选择：

| 需求 | 用哪个接口 / 参数 |
| --- | --- |
| 普通语音条转文字 | `POST /asr`（不传 `engine`，默认 sensevoice） |
| 需要每句的 `start/end`（字幕、定位、按句切片） | `POST /asr/segments` |
| 想用 whisper 做整段转写 | `POST /asr`，`engine=whisper` |
| 不确定但反正要时间戳 | `/asr/segments` 不传或 `engine=auto`（固定选 whisper） |
| 异步任务 | `POST /jobs/asr` 同样支持 `engine` / `timestamps` |

`/asr` 的响应新增 `engine` 字段（`sensevoice` / `whisper`），插件可以不展示，但建议
写进日志方便排错。

`/asr/segments` 契约：

```json
{
  "ok": true,
  "engine": "whisper",
  "language": "zh",
  "duration": 12.3,
  "segments": [{"start": 0.0, "end": 2.4, "text": "..."}],
  "text": "全文",
  "timestamp_source": "native"
}
```

插件要注意：

- 分段保证**升序、无重叠、`end > start`**，可以直接拿来切音频/做字幕；
- `timestamp_source=native` 是 whisper 原生时间戳；`vad` 是 sensevoice 用
  fsmn-vad 切句得到的边界（段间静音保留为间隙，不保证逐字对齐）；
- 静音/纯噪声可能返回 `segments: []`，这是合法结果，展示「（未识别到语音）」；
- **不要默认给所有请求加 `engine=whisper`**：whisper 首次使用要下载
  `models/whisper/<size>/`（几十秒到几分钟），而 sensevoice 默认已就绪；
- 需要 whisper 时把超时放宽到 **≥600 秒**（首次下载 + 加载），之后单条 3 秒音频
  约 0.3 秒；显存不足仍是 507，按 3.5 的降级处理。

实测参考（RTX 3070 8G，中文 CER）：sensevoice 1.73%，whisper small 0.89%、
medium 0.89%、base 9.16%、tiny 17.03%。默认组合是「sensevoice 转文字 + whisper small
出时间戳」。

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

## 5. 异步任务接口（推荐插件用）

长任务（长文本 TTS、长录音转写、整首歌分离）推荐用任务队列接口：
**提交后立刻拿到 `id`，再轮询或取结果**。好处是插件不必一直占着一条
HTTP 长连接，服务重启后任务记录也还在。

> 短请求（一句话朗读、几秒语音条）继续用 `/tts`、`/asr` 即可；
> 它们内部也走同一个队列，只是会等任务完成再返回。

### 5.1 提交

TTS（JSON）：

```http
POST /jobs/tts
Content-Type: application/json

{"text":"要朗读的整段文本","voice":"default","speed":1.0,"priority":0}
```

ASR（multipart）：

```http
POST /jobs/asr
Content-Type: multipart/form-data

file=<音频/视频>
language=auto
speaker=1
priority=0
```

人声分离（multipart）：

```http
POST /jobs/separate

file=<音频/视频>
model=htdemucs
stems=vocals,drums,bass,other
priority=0
```

通用入口（高级）：`POST /jobs`，body
`{"kind":"tts","action":"tts.synthesize","priority":0,"payload":{...}}`。

统一响应：

```json
{"ok": true, "id": "6f2d…", "state": "queued"}
```

- `priority`：默认 0，越大越先；负数表示「不着急」。
- 排序严格三层：**同类型优先 > `priority` > 提交时间**（没有 aging 提权）。
  服务端会先把当前 kind 的排队任务跑完才切换类型，所以提交顺序不等于完成顺序；
  同类任务一直有积压时，其它类型会被饿死（这是有意的取舍）。
- 已经 `running` 的任务不会被后来的高优先级任务抢占；`priority` 只影响 `queued`。
- 上传文件落在服务端 `data/uploads/<job_id>/`，任务结束后默认删除；
  插件**不要**依赖这些路径长期存在。

### 5.2 轮询 / 取详情

```http
GET /jobs/{id}
```

```json
{"ok": true, "id": "…", "kind": "tts", "state": "done",
 "result": {"path": "…", "seed": 12345, "language": "zh",
            "duration_seconds": 4.2, "model": "IndexTTS-2.5"},
 "error": null}
```

- `state`：`queued` / `running` / `done` / `failed` / `canceled`；
- 推荐节奏：首次 300~500ms，之后逐步退避到 2~5 秒；
  不要 50ms 一次地打，也不要无限轮询（超过 `LOCALAI_JOB_TIMEOUT` 会失败）。
- 只想看最近任务：`GET /jobs?state=queued&kind=tts&limit=20`；
- 队列压力：`GET /jobs/stats`（`queued`、`running`、`by_kind`、
  `oldest_queued_seconds`、`scheduler.current_kind`、
  `scheduler.per_kind_parallel`；没有 `aging_seconds`）。

### 5.3 取结果

```http
GET /jobs/{id}/result
```

- TTS：直接返回 `audio/wav` 文件（与 `/tts` 的音频内容同源）；
- ASR / 分离：返回 JSON（`result.text` / `result.stems` / `result.dir` …）；
- 任务未完成时返回 **409** `{"ok":false,"error":"job_not_done",...}`。

### 5.4 取消

```http
POST /jobs/{id}/cancel
```

- `queued`：立即变成 `canceled`，不会再执行；
- `running`：返回 `{"state":"running","cancel_requested":true}`，
  服务端会在 GPU 调用退出后置 `canceled`，**不会写 result**；
- 已经结束的任务返回 **409** `not_cancelable`。

插件在用户点「停止」、关闭面板、退出程序时，应取消自己提交的 queued 任务，
避免服务端积压。

### 5.5 失败与重试

| 情况 | 表现 | 插件行为 |
| --- | --- | --- |
| 参数错误 | 提交时 400 | 修正参数，不重试 |
| 显存不足 | 任务 `failed`，`error` 含「显存不足」 | 提示用户，不循环重试 |
| 引擎失败 | 任务 `failed`，`error` 含异常类型 | 可重试 1 次（新提交一个任务） |
| 超时 | 任务 `failed`，`error` 含「任务超时」 | 提示；长文本先分段再提交 |
| 服务重启 | requeue 模式下自动重跑；fail 模式变 `failed` | 以 `GET /jobs/{id}` 为准 |

**不要**拿同一个 `id` 重复提交；重试要新建任务。任务 `id` 可以短期持久化
（比如存进插件自己的缓存），但不要假设服务端永久保留记录。

### 5.6 同步接口还要不要用？

要。两者共用同一条执行链路：

- 一句话朗读、几秒语音条：`/tts`、`/asr` 更简单；
- 长文本 / 批量 / 需要进度与取消：`/jobs/*`；
- 不要为了「异步」而把短请求也拆成轮询，会增加延迟和无谓请求。

---

## 6. 版本协商

- 服务端在 `/capabilities` 返回 `api_version`（当前 `"1"`）和 `service.version`（如 `"2.2.0"`）。
- 插件只依赖 `api_version` 的主版本；`service.version` 仅用于展示/日志。
- 新增字段是兼容变更，插件必须忽略未知字段。
- 若 `api_version` 不认识：
  - 不要调用写接口（TTS/上传/声纹登记）；
  - 可继续调用 `/health` 做状态展示；
  - 提示用户升级插件或服务。

---

## 7. 错误处理约定

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

## 8. 鉴权预留

服务端通过环境变量 `LOCALAI_TOKEN` 开启（默认关闭）：

```http
Authorization: Bearer <token>
```

- 豁免：`/health`、`/capabilities`、`/docs`、`/openapi.json`；
- 插件把 token 存在自己的安全配置里，不要写进日志/URL；
- 未开启时不要主动发空 Authorization 头。

---

## 9. 翻译：什么时候调 `/translate` 而不是自己翻译

### 9.1 总原则

插件**不要自己下载、加载或运行翻译模型**（也不要把大段文本塞给在线翻译 API）。
服务层已经有 `translate` 能力（NLLB-200-distilled-600M）：懒加载、空闲卸载、
和 TTS/ASR/Demucs 共用显存管家。插件只负责“把文本递过去、把译文显示出来”。

判断口诀：

- **要“看懂大意”的跨语言文本** → 调 `/translate`；
- **要求逐字准确 / 有术语表 / 法务医疗** → 不要只依赖 NLLB，走人工或专业翻译；
- **纯代码、变量名、URL、邮箱、订单号** → 不翻译，原样展示；
- **一两个词的短文本** → 不值得进模型（延迟与检测都不稳），能显示原文就显示原文。

### 9.2 用哪个接口

| 场景 | 接口 | 说明 |
| --- | --- | --- |
| 单条消息 / 一段字幕 / 一个段落 | `POST /translate` | 直接返回译文；长文本服务端自动切分 |
| 10~500 条短文本（列表、批量字幕） | `POST /translate/batch` | 一次 HTTP、内部批量推理，最快 |
| 超过 500 条 / 不希望阻塞 UI | `POST /jobs/translate` + `GET /jobs/{id}/result` | 入队后轮询，模型只加载一次 |
| 判断能力是否可用 | `GET /translate/status` 或 `/capabilities` | `ready` + `loaded`，不加载模型 |

**不要**在循环里逐条调 `/translate`：500 条逐条调用会多出 500 次排队/序列化开销，
批量接口实测快一个数量级。

### 9.3 请求要点

```json
POST /translate
{"text": "今天天气不错，我们出去走走吧。", "source": "auto", "target": "zh"}
```

- `source` 优先写明确语言（`zh`/`en`/`ja`… 或 `zho_Hans`/`eng_Latn`）；
  只有“不知道用户语言”时才用 `auto`。`auto` 只覆盖常见语种，超短文本会失败（400）。
- `target` 必须是明确语言，不能 `auto`。
- 一次请求只能有一种源语言；中英混排请拆成多条，分别给 `source`。
- 服务端已经会按段落/句子切分并拼回，**插件不要为了“怕超长”把段落拆碎**；
  单条上限 5000 字，超过就分条或走队列。
- 返回的 `source` / `target` 是实际使用的 NLLB 码，可直接展示或缓存。
- 可选 `engine: "nllb"`；以后扩展引擎时用它切换，不要写死模型名。

### 9.4 语言与展示

- `GET /capabilities` 的 `translate.languages` 是完整语言表（`code` + 中文名），
  可直接用来做下拉框；常见别名见 `docs/SERVICE_API.md` 13.6。
- `source=auto` 的检测结果不可靠时，给用户一个“手动选源语言”的入口，
  然后带着明确的 `source` 重试；不要反复用 `auto` 重试。
- 译文标点由模型决定（中文常输出半角标点），插件不要做“全角/半角”强转，
  否则会和原文引用对不上。

### 9.5 延迟、加载与缓存

- 第一次翻译会加载模型，本机约 **5~8 秒**；之后单条短句约 **0.23 秒**，
  500 条批量约 **31~36 秒**。
- UI 上要显示“翻译中…”而不是无响应；首次加载可以把超时放宽到 30 秒以上。
- 建议按 `(text, source, target, engine)` 做**客户端缓存**：同一段文本重复翻译很常见，
  命中缓存可以完全避免请求。不要缓存“临时状态”（比如正在加载）。
- 空闲 900 秒后模型会被卸载；下一次请求又会加载一次，这是正常行为，
  不要因此认为服务挂了。

### 9.6 失败降级

| 返回 | 插件行为 |
| --- | --- |
| 400（语言/长度/引擎参数错） | 修正参数或提示用户；**不要原样重试** |
| 503（引擎/权重不可用） | 隐藏或禁用翻译按钮，记录一次日志；不要循环重试 |
| 507（显存不足） | 提示“显卡忙”，稍后重试；不要自动卸载其它能力 |
| 504（队列超时） | 把批次拆小，或改用同步 `/translate/batch` |
| 网络错误 | 静默降级为原文展示 |

翻译是**增强功能**：失败时应该继续显示原文，不能阻塞消息/字幕/播放主流程。

### 9.7 隐私

翻译全程在本机 GPU 上完成，文本不会离开本机；插件不要把待翻译文本
写进日志或上报。缓存译文时也按普通用户内容对待。

---

## 10. 插件不该做的事

- ❌ 自己加载/卸载模型、自己管理显存；
- ❌ 直接读 `models/`、`speakers/`、`outputs/` 目录；
- ❌ 假设服务一定在 `127.0.0.1:7880`（应可配置）；
- ❌ 在服务不可用时阻塞 UI 或反复重试；
- ❌ 把服务端返回的绝对路径当作长期文件引用；
- ❌ 调用未在 `/capabilities` 里声明为 `ready` 的能力。

---

## 11. 最小实现清单（未来插件）

1. `GET /capabilities` → 校验 `api_version`、缓存能力；
2. 自动朗读：`POST /tts`，分段、串行、180s 超时、失败静默跳过；
3. 语音转写：`POST /asr`（大文件分片、`?speaker=0` 可选）；
4. 翻译：`POST /translate` / `POST /translate/batch`（`source=auto` 失败就要求手选，
   单条 ≤ 5000 字、批 ≤ 500 条；超大批量走 `/jobs/translate`）；
5. 音色管理：`GET /voices`、`POST /api/v1/upload`；
6. 健康/显存：`GET /health`、`GET /vram`；
7. 错误映射：400/401/404/413/503/504/507；
8. 配置：`baseUrl`、`token`、超时、总开关；
9. 日志脱敏。

---

## 12. 让服务去下载 / 分析链接

### 12.1 总原则

插件**不要自己装 yt-dlp、不要自己抓页面、不要自己解析站点**。服务层已经提供
`download` 能力（yt-dlp + 直链兜底 + SSRF 防护 + cookie 管理 + 限额）。
插件只负责把用户给的**公开链接**递过去，然后展示进度与结果。

- 一次只处理**用户明确给出的一个链接**；不要做站内爬取、批量遍历、自动翻页。
- 下载/分析产物在服务端 `data/downloads/<日期>/`，插件不要假设长期存在，
  需要长期保存就自己复制走。
- 需要字幕时间轴/镜头边界时，优先用 `/media/analyze`，不要自己拆成
  「先下载再分别调 ffmpeg」——服务端已经编排好字幕优先级与场景检测。

### 12.2 选哪个接口

| 场景 | 接口 | 说明 |
| --- | --- | --- |
| 用户明确要「下载这个视频/音频」 | `POST /media/download` | 入队，立刻返回 `job_id`；轮询进度 |
| 小文件（≤ 200MB）想直接拿路径 | `POST /media/download/sync` | 同步返回；超限 413，改异步 |
| 要「字幕时间轴 + 镜头边界」 | `POST /media/analyze` | 同步等结果；长视频建议用 `/jobs/analyze` |
| 不希望阻塞 UI | `POST /jobs/analyze` + `GET /jobs/{id}` | 与下载任务同一条队列 |
| 判断能不能下 | `GET /media/download/status` | yt-dlp 版本、限额、目录 |
| 用户点「停止」 | `POST /jobs/{id}/cancel` | 运行中尽力取消；半成品会清理 |

### 12.3 下载任务

```json
POST /media/download
{"url": "https://www.bilibili.com/video/BV...", "quality": "720p",
 "subtitles": true, "priority": 0}
```

响应 `{"ok":true,"job_id":"…","state":"queued"}`。轮询 `GET /jobs/{id}`：

- `state`：`queued/running/done/failed/canceled`；
- `meta.progress`：`{percent, downloaded_bytes, total_bytes, speed, eta}`；
- 完成后的 `result`：`{path,title,duration,site,subtitles[],size_mb}`。

轮询节奏 500ms~2s；下载可能几十分钟（`LOCALAI_DOWNLOAD_TIMEOUT` 默认 1800s），
不要用 TTS 的 180s 超时。UI 上进度条用 `percent`，没有 `total_bytes` 时用
`downloaded_bytes` 显示「已下载 xx MB」。

### 12.4 链接分析

```json
POST /media/analyze
{"url": "https://...", "engine": "whisper|sensevoice|auto", "max_frames": 0}
```

返回：

```json
{"ok":true,"title":"…","duration":123.4,"path":"…",
 "subtitles_from":"site|asr",
 "segments":[{"start":0.08,"end":0.68,"text":"…"}],
 "scenes":[{"time":6.4,"score":1.0}],
 "frames":[]}
```

- `segments` 已升序、无重叠，可直接做字幕轴/时间戳跳转；
- `subtitles_from="site"` 表示站内字幕，`"asr"` 表示服务端识别；
- `max_frames=0` 不抽帧；需要缩略图时给 1~200，返回 `frames[].path`；
- `scenes` 是镜头切换点（`time` 秒 + `score`），可直接做章节/关键帧。

### 12.5 错误映射

| 返回 | 场景 | 插件行为 |
| --- | --- | --- |
| 400 | URL 非 http/https / 私有地址 / 参数非法 | 提示用户换链接，不要重试 |
| 413 | 超过单文件或同步上限 | 提示「文件太大」，改用异步接口 |
| 409 | 任务取消 / 正在取消 | 停止轮询 |
| 502 | yt-dlp 站点错误、风控、需要登录 | 展示 `detail`；不要连续重试 |
| 504 | 下载/分析超时 | 提示；可换更短链接或调大 `LOCALAI_DOWNLOAD_TIMEOUT` |

### 12.6 Cookie 与隐私

- B站链接**不需要插件传 cookie**：服务端按域名自动读取本机凭证，生成临时
  cookie 文件、用完即删；
- 其它需要登录的站点，插件可让用户选择 Netscape cookie 文件路径，通过
  `cookies` 字段传给服务端；**不要**把 cookie 内容写进日志/界面；
- 服务端不会把 cookie 值写日志，也不会提交进仓库；
- 只下载用户明确给出的公开链接；失败一次就如实展示错误，不要反复重试到被封。

---

## 13. 场景四：插件要关键帧就调 `/vision/frames`

### 13.1 什么时候用

- 用户上传/转发一段视频，想要「几张能代表内容的截图」；
- 自动化报告、视频摘要、封面候选：需要避开章节页、转场、空镜；
- 已经下载过视频（`/media/download` 或 `/media/analyze` 返回了 `path`），
  不要自己抽均匀帧——把路径交给服务端。

### 13.2 推荐调用（异步）

```json
POST /jobs/vision
{
  "video_path": "E:/LocalAIService/data/downloads/2026-11-08/xxx.mp4",
  "target_frames": 3,
  "planner": "hybrid",
  "max_rounds": 6
}
```

返回 `{ok:true,id,state:"queued"}`；轮询 `GET /jobs/{id}`：

```json
{"state":"done","result":{
  "ok":true,"duration":431.267,"target":3,"rounds_used":1,
  "frames":[{"time":5.085,"path":"...","score":4,"kind":"chart",
             "ocr_text":"...","summary":"...","why":"..."}],
  "stats":{"candidates":12,"vlm_calls":8,"elapsed_ms":20653},
  "subtitles_from":"site"}}
```

- 也可以直接 `POST /vision/frames` 同步等（讲解视频 20~60 秒，长视频更久）；
- 刚跑完 `/jobs/analyze` 时，直接传 `{"job_id":"<id>","target_frames":3}`，
  服务端会从上个任务结果的 `path` 取视频，不用自己拼路径。

### 13.3 参数怎么选

- `target_frames`：1~10；封面/摘要一般 3，长教程最多 5~6；
- `planner`：默认 `hybrid`（快、稳定）；视频里图表/演示很多时再试 `agent`；
- `max_rounds`：默认 6；短视频 2~3 就够，长视频不要超过 8；
- `subtitles` / `scene_times`：如果插件自己已经有（例如 `/asr/segments`、
  `/media/scenes` 的结果），直接传，能省一次识别/检测；
- 不传 `subtitles` 时，服务端优先找视频同目录的 `*.srt/*.vtt`，其次跑 ASR。

### 13.4 结果怎么用

- `frames[].path` 是服务端本机绝对路径，插件可以直接读文件或转成 `file://`；
- `time` 是精确到毫秒的时间点，可做「点击跳到视频对应位置」；
- `kind` 可用于挑封面：`chart/table/code/demo` 优先，`blank/transition` 已排除；
- `ocr_text` 是小模型识别结果，**可能幻觉**，展示时可以标「自动识别」；
- `score>=3` 且 `usable=true` 才会返回；凑不够时 `frames` 会少于 `target_frames`，
  不要重试到死，按实际数量展示。

### 13.5 降级与错误

| 情况 | 响应 | 插件行为 |
| --- | --- | --- |
| 模型未安装 / 进程崩溃 / 全部超时 | `ok:true` + `degraded:true` + `degraded_reason`，帧来自方差回退 | 正常展示，提示「自动挑帧（降级）」 |
| VLM 单次调用失败 | 该帧跳过；连续两次失败提前结束并降级 | 同上 |
| `video_path` 不存在 / 不是视频 | 400 | 提示用户换文件，不要重试 |
| 显存不足（TTS 正忙且腾不出来） | 507 `insufficient_vram` | 提示稍后重试，或先 `/vram/unload` |
| 任务取消 | 409 / `state=canceled` | 停止轮询 |

### 13.6 不要做的事

- 不要自己按 0%/50%/100% 抽帧再让用户选——章节页/转场会浪费用户注意力；
- 不要并发提交多个 `/vision/frames`：视觉模型单 slot，且和 TTS 抢显存；
- 不要把 `models/vision/` 里的 GGUF 复制进插件目录；路径由服务端
  `/vision/status` 和 `/capabilities` 给出；
- 不要在插件里起第二个 llama-server：端口 7891 由服务管理。
