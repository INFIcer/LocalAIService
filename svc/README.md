# 本地 AI 服务管理器（原名 IndexTTS-2.5）—— svc 目录说明

> **2026-11 更新：项目已改名为 LocalAIService（本地 AI 服务管理器）。**
> 真实目录 `E:\LocalAIService`，`E:\IndexTTS` 是兼容 junction；
> 服务现在同时提供 TTS / ASR / 声纹 / ffmpeg 媒体接口，启动时 0 模型。
>
> 默认后端已改为独立引擎（`TTS_BACKEND=engine`）。
>
> 启动方式、目录结构、迁移说明、媒体接口、排错说明以
> [`E:\LocalAIService\README.md`](E:\LocalAIService\README.md) 为准。
>
> 新启动命令（旧名 `start_service_engine.ps1` 仍然可用）：
>
> ```powershell
> powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1
> powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Restart -Background
> ```
>
> 本文档下半部分保留的是旧版 ComfyUI 链路的细节（API 契约仍然一致），
> 其中「上传参考音频到 ComfyUI」「transformers 5.x 阻塞」等描述仅适用于
> `TTS_BACKEND=comfy`。engine 模式直接读 `E:\LocalAIService\voices`，不上传。

把 ComfyUI 的 `IndexTTS2_5Node` 包成一个简单的本地 HTTP 服务：提交文本，直接拿回 WAV。

- 监听：`127.0.0.1:7880`
- ComfyUI：`http://127.0.0.1:8188`（已在运行，服务不会启动/重启它）
- 工作流：`LoadAudio -> IndexTTS2_5Node -> SaveAudio`，模板写在 `service.py` 里，不依赖外部 JSON
- 输出：ComfyUI 的 `SaveAudio` 目前固定写 FLAC，服务端用 `soundfile` 转成 16bit PCM WAV 再返回

## 目录

```
E:\LocalAIService\
├─ svc\
│  ├─ service.py      # FastAPI 服务（全部逻辑）
│  ├─ selftest.py     # 不加载模型的最小链路自检
│  ├─ start.ps1       # 建/复用 venv、装依赖、前台启动
│  ├─ README.md
│  └─ .venv\          # start.ps1 自动创建
├─ voices\            # 参考音色目录（每次 /voices 都重新扫描）
│  ├─ default.mp3
│  └─ multi.wav
├─ outputs\           # 服务端留档：每次成功合成一份 wav
├─ models\TTS\IndexTTS-2.5\   # 权重下载目录（下载脚本使用）
└─ logs\download.log          # 权重下载日志
```

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File E:\LocalAIService\svc\start.ps1
```

`start.ps1` 会：

1. 优先用 `C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe` 建 venv（找不到再用 `C:\Python314\python.exe` / PATH 里的 python）；
2. venv 固定在 `E:\LocalAIService\svc\.venv`，已存在就复用；
3. 只有 `import fastapi, uvicorn, httpx, soundfile` 失败时才 `pip install`，重复运行不会重复装；
4. 前台启动 `service.py`，日志直接打在控制台。

第一次运行需要联网装依赖（约 1 分钟）。服务本身只监听 `127.0.0.1:7880`，不会占用别的端口。

## 接口

### GET /health

```json
{
  "ok": true,
  "comfy": "up",
  "model_ready": false,
  "voices": 2,
  "model_detail": "模型权重尚未就绪，E:\\LocalAIService\\models\\TTS\\IndexTTS-2.5 缺少 N 个文件",
  "comfy_url": "http://127.0.0.1:8188"
}
```

- `ok`：ComfyUI 是否可达（`/system_stats`）。
- `model_ready`：节点实际读取的目录里，8 个必需模型文件是否都存在且非空。
- `voices`：`E:\LocalAIService\voices` 下的音色数量。

### GET /voices

每次请求都重新扫描音色目录，新增文件后**不用重启**。

```json
{
  "voices": [
    {"name": "default", "path": "E:\\LocalAIService\\voices\\default.mp3", "filename": "default.mp3", "format": "mp3", "duration": 3.42},
    {"name": "multi", "path": "E:\\LocalAIService\\voices\\multi.wav", "filename": "multi.wav", "format": "wav", "duration": 12.1}
  ],
  "count": 2,
  "directory": "E:\\LocalAIService\\voices"
}
```

音色名 = 文件名去掉扩展名。同名不同扩展名时，按文件名排序，第二个起自动叫 `xxx_2`。

### POST /tts

请求：

```json
{
  "text": "你好，欢迎使用 IndexTTS 2.5。",
  "voice": "default",
  "speed": 1.0,
  "emotion": null,
  "seed": null
}
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | 是 | 要合成的文本，最长 5000 字符 |
| `voice` | 否 | 音色名，默认 `default`；不存在返回 404 |
| `speed` | 否 | 语速，`0.5 ~ 2.0`，1.0 正常；内部换算成节点的 `duration_factor = 1/speed` |
| `emotion` | 否 | 8 个 0~1 的数字（`0.8,0,0,0,0,0,0,0`）走 `emo_vector`；其它文本走 `emo_text`（需要 Qwen 情感模型） |
| `seed` | 否 | 0 ~ 4294967295；不填则随机，实际值在响应头 `X-Seed` |
| `language` | 否 | `zh/en/ja/es/ar`；不填自动判断（有中文→zh，有假名→ja，否则 en） |

成功：`HTTP 200`，`Content-Type: audio/wav`，响应体就是 WAV 字节。

响应头里带诊断信息：

- `X-Seed`：本次实际使用的种子
- `X-Language`：本次使用的语言
- `X-Audio-Path`：服务端留档的绝对路径（如 `E:\LocalAIService\outputs\20261108-123456_default_123.wav`）
- `X-Comfy-Filename`：ComfyUI output 里的原始 FLAC 文件名

失败：JSON（含 ComfyUI 的 execution error 原文），例如：

```json
{
  "ok": false,
  "error": "comfy_error",
  "detail": "ComfyUI 执行失败（详见 comfy_error）",
  "comfy_error": "execution_error: 节点 IndexTTS2_5Node 抛出 FileNotFoundError: ...",
  "comfy_messages": [...],
  "prompt_id": "..."
}
```

### POST /v1/audio/speech（OpenAI 兼容）

请求：

```json
{
  "model": "indextts",
  "input": "你好，这是 OpenAI 兼容接口。",
  "voice": "default",
  "response_format": "wav",
  "speed": 1.0
}
```

- `input` 对应 `/tts` 的 `text`；`model` 只做展示，不校验。
- `response_format` **只支持 `wav`**；其它值返回 `400`，错误里说明原因。
- 额外支持 `language`、`emotion`、`seed`，含义同上。
- 成功同样返回 `audio/wav` 二进制，响应头同上；错误按 OpenAI 的 `{"error": {...}}` 形状返回。

### GET /comfy/status

透传 ComfyUI `/queue` 的摘要：

```json
{
  "ok": true,
  "comfy": "up",
  "running": 0,
  "pending": 0,
  "queue_running": [],
  "queue_pending": []
}
```

## 调用示例

### curl.exe（cmd / PowerShell 7 都可以）

健康检查：

```powershell
curl.exe -s http://127.0.0.1:7880/health
```

音色列表：

```powershell
curl.exe -s http://127.0.0.1:7880/voices
```

合成（中文建议把 JSON 先写成 UTF-8 无 BOM 文件，避免控制台编码把中文弄坏）：

```powershell
$json = '{"text":"你好，这是本地 TTS 服务。","voice":"default","speed":1.0}'
[System.IO.File]::WriteAllText("$env:TEMP\tts.json", $json, [System.Text.UTF8Encoding]::new($false))
curl.exe -s -X POST http://127.0.0.1:7880/tts `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$env:TEMP\tts.json" `
  -o E:\LocalAIService\outputs\demo.wav
```

看响应头（确认 `X-Audio-Path`、`X-Seed`）：

```powershell
curl.exe -s -D - -X POST http://127.0.0.1:7880/tts `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$env:TEMP\tts.json" `
  -o E:\LocalAIService\outputs\demo.wav
```

OpenAI 兼容：

```powershell
curl.exe -s -X POST http://127.0.0.1:7880/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"indextts\",\"input\":\"Hello from IndexTTS.\",\"voice\":\"default\",\"response_format\":\"wav\"}" `
  -o E:\LocalAIService\outputs\openai_demo.wav
```

### PowerShell Invoke-RestMethod

```powershell
# 健康检查
Invoke-RestMethod -Uri http://127.0.0.1:7880/health | ConvertTo-Json

# 音色列表
(Invoke-RestMethod -Uri http://127.0.0.1:7880/voices).voices | Format-Table name, format, duration, path

# 合成并保存
$body = @{
    text  = "你好，这是 PowerShell 调用的测试。"
    voice = "default"
    speed = 1.0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:7880/tts `
    -Method Post `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -OutFile E:\LocalAIService\outputs\ps_demo.wav
```

> Windows PowerShell 5.1 里如果直接 `-Body $body`，中文可能被按 ISO-8859-1 编码；上面用 `UTF8.GetBytes()` 是为了稳妥。PowerShell 7 默认就是 UTF-8。

### 自检（不依赖 TTS 模型）

```powershell
cd E:\LocalAIService\svc
.\.venv\Scripts\python.exe selftest.py
```

它跑 `LoadAudio -> SaveAudio` 最小工作流，验证「上传 → 提交 → 轮询 → 下载 → 转 WAV」整条链路，输出：

```json
{
  "ok": true,
  "prompt_id": "...",
  "elapsed_seconds": 1.234,
  "comfy_output": {"filename": "svc_selftest_00001.flac", "subfolder": "", "type": "output"},
  "comfy_bytes": 12345,
  "wav_bytes": 45678,
  "wav_path": "E:\\LocalAIService\\outputs\\selftest_minimal.wav"
}
```

模型没下载完时，用这个脚本就能确认服务本身没问题。

## 换参考音色

1. 把音频文件（`wav / mp3 / flac / ogg / opus / m4a / aac`）放进 `E:\LocalAIService\voices\`；
2. 文件名（去扩展名）就是音色名，例如 `小美.wav` → `voice: "小美"`；
3. 直接调 `/voices` 或 `/tts`，**每次请求都会重新扫描目录，不用重启服务**。

服务第一次用到某个音色时，会通过 ComfyUI 的 `/upload/image` 接口把文件上传到 ComfyUI 的 input 目录（`D:\ComfyUI\input`），之后按内容哈希缓存，同一进程内不重复上传。文件内容改了会生成新的上传名。

> **engine 模式不需要这一步**：`TTS_BACKEND=engine` 时直接把
> `E:\LocalAIService\voices\xxx` 的绝对路径交给引擎的 `librosa.load()`，
> 不经过 ComfyUI，也不产生上传缓存。

**为什么不是直接给 LoadAudio 传绝对路径？**
这版 ComfyUI 的 `LoadAudio` 用 `folder_paths.get_annotated_filepath()` 做路径穿越检查，`E:\LocalAIService\voices\...` 不在 input 目录内会被拒绝（`Invalid file path`）。所以服务改成「先上传、再把 input 目录里的文件名交给 LoadAudio」，音色文件仍然只维护在 `E:\LocalAIService\voices\`。

## 配置（环境变量）

都有默认值，一般不用改。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `COMFY_BASE` | `http://127.0.0.1:8188` | ComfyUI 地址 |
| `INDEX_TTS_VOICES` | `E:\LocalAIService\voices` | 音色目录 |
| `INDEX_TTS_OUTPUT` | `E:\LocalAIService\outputs` | 服务端留档目录 |
| `INDEX_TTS_MODEL_DIR` | `E:\LocalAIService\models\TTS\IndexTTS-2.5` | 权重下载目录（`/health` 检查用） |
| `INDEX_TTS_COMFY_MODEL_DIR` | `D:\Comfy-Desktop\视频\ComfyUI\models\TTS\IndexTTS-2.5` | 节点实际读取的模型目录 |
| `INDEX_TTS_TIMEOUT` | `180` | 单次任务最长等待秒数 |
| `INDEX_TTS_POLL` | `0.4` | 轮询间隔秒数 |
| `INDEX_TTS_HOST` / `INDEX_TTS_PORT` | `127.0.0.1` / `7880` | 监听地址和端口 |

## 模型状态与目录映射（重要）

`index-tts-2.5` 节点在代码里硬编码读取：

```
<ComfyUI>\models\TTS\IndexTTS-2.5
```

即：

```
D:\Comfy-Desktop\视频\ComfyUI\models\TTS\IndexTTS-2.5
```

但下载脚本把权重放在：

```
E:\LocalAIService\models\TTS\IndexTTS-2.5
```

如果下载完成后 `ComfyUI\models\TTS` 还不存在，节点会报 `FileNotFoundError: IndexTTS-2.5 模型目录缺少文件 ...`，即使 `E:` 盘的权重已经齐了。

两种处理方式（任选其一，推荐 junction，不占额外空间）：

```powershell
# 方式一：目录联接（推荐；只需执行一次，无需管理员）
New-Item -ItemType Junction `
  -Path 'D:\Comfy-Desktop\视频\ComfyUI\models\TTS' `
  -Target 'E:\LocalAIService\models\TTS'
```

```powershell
# 方式二：复制（占约 6GB 额外空间）
Copy-Item 'E:\LocalAIService\models\TTS' 'D:\Comfy-Desktop\视频\ComfyUI\models\TTS' -Recurse
```

节点是在每次执行时按文件系统路径找文件，不是启动时缓存，所以建好 junction 后**不需要重启 ComfyUI**。`/health` 的 `model_ready` 看的是节点那个目录；`model_detail` 会区分「还在下载」和「下载完了但没映射」。

> 本机已经创建好上述 junction（`D:\Comfy-Desktop\视频\ComfyUI\models\TTS` -> `E:\LocalAIService\models\TTS`），
> 权重也已下载完成，`/health` 现在返回 `model_ready: true`。换机器或目录被删掉时按上面的命令重建即可。

### 当前环境已知阻塞：ComfyUI 的 transformers 版本过新

ComfyUI 自带的 `transformers` 是 **5.16.1**，这个版本已经移除了
`transformers.generation.beam_constraints`；而 `index-tts-2.5` 的 vendored 代码仍会 import 它，
所以真正执行 `IndexTTS2_5Node` 时会报：

```
ModuleNotFoundError: No module named 'transformers.generation.beam_constraints'
```

`/tts` 会把这个原始错误完整放进 `comfy_error`，链路本身是通的。按本次任务约束，
**没有修改 ComfyUI 节点目录，也没有安装/降级 ComfyUI 侧的依赖**。要真正合成，需要二选一：

1. 把 `index-tts-2.5` 自定义节点更新到兼容 `transformers 5.x` 的版本（推荐，影响面小）；
2. 或在 ComfyUI 的 `.venv` 里把 `transformers` 降到 4.x（会影响整个 ComfyUI，需单独评估）。

这个问题修好后，直接跑本文档里的 `/tts` 示例即可，服务代码不用改。


下载进度：

```powershell
Get-Content E:\LocalAIService\logs\download.log -Tail 20
```

## 常见错误排查

### 1. `/health` 里 `comfy: "down"`

```powershell
Invoke-RestMethod http://127.0.0.1:8188/system_stats
```

如果这条也不通，说明 ComfyUI 没在跑或被防火墙拦了。本服务不会自动拉起 ComfyUI。

### 2. `model_ready: false`

- `model_detail` 说「缺少 N 个文件」：权重还在下载，等 `download.log` 出现 `=== 全部结束 ===` 且目录总大小约 6GB。
- `model_detail` 说「权重已下载完成，但节点读取的目录里没有」：按上一节建 junction。

### 3. `FileNotFoundError: ... 模型目录缺少文件`

同上，节点读的是 `ComfyUI\models\TTS\IndexTTS-2.5`，不是 `E:` 盘那个目录。

### 4. `comfy_error` 里有 `execution_error`

服务会把 ComfyUI 的原始错误（节点名、异常类型、异常消息、traceback 尾部）原样放进响应的 `comfy_error` / `comfy_messages`。常见原因：

- 本机当前的 `transformers 5.16.1` 与 `index-tts-2.5` 不兼容（`No module named 'transformers.generation.beam_constraints'`），见上一节；
- 模型/情感模型没下完；
- 参考音频损坏或格式不受支持（换 `wav`/`mp3` 试）；
- 文本太长触发 `max_mel_tokens` 截断（节点默认只提示不报错；如果开了 `strict_max_mel_tokens` 才会报错）。

### 5. 中文变成乱码或合成出英文

用 UTF-8 提交 JSON，并带 `charset=utf-8`。PowerShell 5.1 直接 `-Body $body` 可能把中文按本地编码发出去，按上面示例用 `UTF8.GetBytes()` 或写 UTF-8 文件给 `curl.exe --data-binary`。

### 6. `400`：`只支持 response_format='wav'`

OpenAI 接口目前只出 WAV。要 MP3 就在客户端转，或者等后续扩展（ComfyUI 的 `SaveAudioMP3` 可以接，但当前没实现）。

### 7. `404`：音色不存在

`GET /voices` 看实际名字。注意名字是**文件名去扩展名**，且区分大小写。

### 8. 端口 7880 被占用

```powershell
Get-NetTCPConnection -LocalPort 7880 -ErrorAction SilentlyContinue
```

停掉占用进程，或临时用 `$env:INDEX_TTS_PORT=7881` 换端口（换完记得同步客户端地址）。

### 9. 合成很慢 / 超时

- 第一次合成要加载模型（约 10~60 秒），属正常；
- 单次任务上限 180 秒（`INDEX_TTS_TIMEOUT`），长文本建议拆段；
- 8G 显存同时跑图像/视频模型时可能 OOM，先释放显存再合成。

## 已知限制

- **以下限制主要描述旧版 ComfyUI 后端。** 默认的 engine 后端已可正常合成；
  它的启动/排错/性能数据见 `E:\LocalAIService\README.md`。
- 旧的 `TTS_BACKEND=comfy` 链路在当前环境的 ComfyUI 里仍被
  `transformers 5.16.1` 阻塞（`No module named 'transformers.generation.beam_constraints'`）。
- 只支持 `response_format=wav`；不做流式返回。
- 不做鉴权，只监听 `127.0.0.1`，不要直接暴露到公网。
- 服务端会留档每次合成的 WAV 到 `E:\LocalAIService\outputs`，不会自动清理。
- `SaveAudio` 是 ComfyUI 里标记为 DEPRECATED 的节点，但当前仍可用；它固定输出 FLAC，所以服务端统一转 WAV。
- 参考音频上传后会留在 ComfyUI 的 `D:\ComfyUI\input`，文件名带内容哈希；换音色/改文件不会覆盖旧文件。
