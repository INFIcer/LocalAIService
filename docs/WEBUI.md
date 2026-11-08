# 管理界面（WebUI）与 `/admin/*` 接口

管理界面是服务层自带的一个**纯静态单页**：没有 npm、没有构建步骤、没有 CDN，
所有 HTML / CSS / JS 都随 Python 包分发。打开浏览器访问：

```text
http://127.0.0.1:7880/ui
```

> 默认只监听 `127.0.0.1`。如果把 `LOCALAI_HOST` 改成 `0.0.0.0`，页面顶部会显示
> 警告：管理接口可能被局域网内其它机器访问，请务必设置 `LOCALAI_TOKEN`。

---

## 1. 页面结构

左侧导航 + 右侧内容，窄屏（≤820px）自动变成顶部横向导航：

| 页面 | 内容 |
| --- | --- |
| **概览** | GPU 显存进度条、各能力卡片（就绪/已加载/显存估算/当前模型/空闲卸载）、任务队列数字、最近 10 条任务、最近日志；每 3 秒自动刷新，可暂停 |
| **模型** | 按能力分组的模型表格：名称、是否已下载、大小、是否当前档位、缺失文件；支持「下载」「切换档位」，下载中显示已写入字节/大小 |
| **设置** | 按分组表单（服务 / 显存 / 空闲卸载 / 并发 / TTS / ASR / 翻译 / 媒体）；每项显示默认值、来源（env/file/default）、是否需要重启；只保存改动项 |
| **任务** | `/jobs` 队列列表（状态 / 类型 / 优先级 / 等待 / 耗时）、状态筛选、取消、**双击任务行打开详情弹窗**（请求参数 / 结果文件 / 错误原文 / 时间线） |
| **系统** | 版本 / Python / GPU / 磁盘占用 / 监听地址、开机自启开关、重启服务（输入 `RESTART` 二次确认）、日志查看器（out/err、行数） |

交互约定：

- 所有操作都有 loading / 成功 / 失败提示；失败直接展示后端 JSON 的 `detail`；
- 令牌只存在浏览器 `localStorage`，页面源码里没有真实 token；
- 下载任务轮询 `GET /admin/models/tasks/{task_id}`，同一时刻只允许一个下载。

### 1.1 任务详情弹窗

**打开方式**：在「任务」页**双击任意任务行**；也可以点该行「操作」列里的「详情」按钮，
或先用 Tab 把焦点移到任务行再按 Enter / 空格。概览页「最近任务」同样双击打开。
移动端没有双击手势，用「详情」按钮即可。

弹窗内容（纵向滚动的五个分区）：

| 分区 | 显示内容 |
| --- | --- |
| **概要** | 任务 ID（一键复制）、类型 / 动作（如 `tts.synthesize`）、状态彩色徽章（失败/取消/超时会额外标注）、优先级、创建 / 开始 / 结束时间、等待时长、执行时长、尝试次数、客户端 IP；`meta.cancel_requested` 时显示「取消请求已发出」 |
| **请求参数** | `payload` 里的文本类字段（`text` / `prompt` / `subtitle` / `ocr` / `transcript` / `content` …）**直接显示原文**并保留换行，可单独复制；下面折叠一份格式化 JSON，长内容可滚动 |
| **结果** | `result` 的格式化 JSON；递归识别 `path` / `output_path` / `output` / `audio_path` / `file` / `stems` / `frames[]` 等字段里的本地绝对路径：**音频**（.wav/.mp3/.m4a …）给 `<audio controls>`，数据源是 `GET /jobs/{id}/result`；**图片**（.jpg/.png/.webp …）给缩略图（优先 `GET /jobs/{id}/result`，失败回退 `GET /vision/frame?path=…`）；**其它文件**只显示路径 + 「复制路径」。文件不可达时只显示路径，不会让整个弹窗报错 |
| **错误** | `error` 原文，等宽字体、保留换行、一键复制；没有错误时显示「没有错误」。自动化测试产生的 `voice: "__definitely_missing__"` 之类在这里能完整看到，便于区分测试与真实故障；错误里含「超时 / timeout」时概要区会加「超时」徽章 |
| **时间线** | created → started → finished 三段，未开始/进行中/已取消都有对应文案；取消的任务会提示「主动取消时 error 为空是正常的」 |

**交互细节**：

- 右上角 `×`、`Esc`、点击遮罩都能关闭；`Tab` 焦点被限制在弹窗内，关闭后焦点回到原任务行；
- 弹窗打开期间**暂停任务列表的 3 秒自动刷新**（顶部「自动刷新」开关仍然有效），关闭后自动恢复；
  弹窗内点「刷新详情」只刷新当前任务；
- 状态是 `queued` / `running` 时底部显示「取消任务 / 请求取消」，调用
  `POST /jobs/{id}/cancel`，成功后同时刷新详情和列表；终态任务隐藏该按钮；
- 详情数据来自 `GET /jobs/{id}`；该请求失败时用列表行里已有的完整记录兜底，并在弹窗顶部
  显示错误原因，不会整页崩；
- 窄屏（≤820px）弹窗铺满全屏；深色主题与主界面一致；
- 服务设置了 `LOCALAI_TOKEN` 时，音频/图片通过带 `Authorization` 头的 `fetch` 取成
  Blob 再播放/预览，不把令牌拼进 URL。


---

## 2. 配置文件 `config.json`

管理界面把改动写入：

```text
<LOCALAI_HOME>/config.json
```

规则：

1. **只写改动项**：没改的键不写；值改回默认值时把该键从文件里删除；
2. 保留文件里不认识的键（不整份覆盖）；
3. 优先级：**环境变量 > config.json > 代码默认值**；
4. 如果某个键被启动脚本的环境变量覆盖，保存仍会成功，但接口返回
   `source: "env"` 与 `warnings`，明确提示「仍以环境变量为准」；
5. `config.json` 已加入 `.gitignore`，属于机器级配置，不入库。

`LOCALAI_HOME` 默认是项目根 `E:\LocalAIService`。

---

## 3. 接口清单

所有 `/admin/*` 都受全局 `LOCALAI_TOKEN` 中间件保护（设置了令牌时要求
`Authorization: Bearer <token>`）。`/ui` 与静态资源本身免鉴权，否则页面无法
打开输入令牌。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/admin/settings` | 当前生效配置，按分组返回；每项带 `value/default/source/requires_restart/description` |
| PUT | `/admin/settings` | 保存白名单设置；返回 `applied` / `requires_restart` / `warnings` |
| GET | `/admin/models` | 各能力模型清单（是否下载、大小、缺失文件、当前档位） |
| POST | `/admin/models/download` | 异步下载；`{"capability":"whisper","name":"medium"}` |
| GET | `/admin/models/tasks/{task_id}` | 下载进度（状态 / 已写入字节 / 错误） |
| POST | `/admin/models/activate` | 切换默认档位（whisper / ASR 引擎），写配置不重启进程 |
| GET | `/admin/system` | 版本 / Python / GPU / 磁盘 / 进程启动时间 / 计划任务 / 监听地址 / 是否启用令牌 |
| GET | `/admin/autostart` | 计划任务 `LocalAIService-Service` 状态 |
| POST | `/admin/autostart` | `{"enabled":true|false}` 启用/禁用开机自启（非 Windows 返回 501） |
| POST | `/admin/restart` | `{"confirm":true}` 才执行；返回 202 后延迟约 2 秒重启 |
| GET | `/admin/logs?lines=200&file=service` | 日志尾部；自动探测 UTF-8(BOM) / UTF-16 / GBK |

### 设置白名单

`GET /admin/settings` 的 `settings` 字段就是完整白名单，共 26 项：

- 服务：`host`、`port`、`token`（需重启；token 只回显是否已设置）
- 显存：`vram_margin_mb`
- 空闲卸载：`idle_unload`、`asr_idle_unload`、`whisper_idle_unload`、
  `speaker_idle_unload`、`separate_idle_unload`、`translate_idle_unload`
- 并发：`asr_parallel`、`jobs_parallel_tts`、`jobs_parallel_separate`、
  `jobs_parallel_extra_mb`、`jobs_max_parallel`
- TTS：`tts_backend`（需重启）、`tts_default_voice`
- ASR：`asr_engine`、`whisper_model`、`whisper_device`（需重启）、
  `whisper_compute`（需重启）
- 翻译：`translate_fp16`（需重启）、`translate_num_beams`、`translate_batch_size`、
  `translate_max_len`
- 媒体：`ffmpeg`

---

## 4. 下载 / 切换 / 重启的行为

- **下载**：后台线程复用已有下载代码——
  whisper 用 `engine/whisper_engine.py::ensure_weights()`，
  翻译用 `engine/translate_engine.py::ensure_weights()`，
  TTS/ASR/声纹/Demucs 用 `localai.cli` 中 `models download` 的同一批函数，
  fsmn-vad 用 `engine/asr_engine.py::_ensure_vad_weights()`。
  进度以「目标目录当前字节 - 开始时字节」表示（上游 snapshot 下载没有统一总字节回调）。
- **切换**：只改 `config.json` 并更新当前进程配置，不主动卸载已加载的旧模型；
  下次请求按新档位加载，旧模型由空闲看门狗 / 显存管家回收。
- **重启**：`POST /admin/restart` 先返回 202，延迟 1~2 秒后用脱离进程执行
  `tools/restart_service.ps1`：结束监听 7880 的 python 进程，再用 WMI 后台执行
  `start_ai_service.ps1`。重启期间接口短暂不可用。
- **开机自启**：Windows 下操作计划任务 `LocalAIService-Service`；不存在时创建
  「登录时」触发任务。命令全部走 `powershell -NoProfile -NonInteractive`，
  不弹 UAC；权限不足时接口返回明确的 PowerShell stderr。

---

## 5. 手动验收

```powershell
# 页面
curl.exe -s http://127.0.0.1:7880/ui | Select-String "LocalAIService 控制台"

# 接口
curl.exe -s http://127.0.0.1:7880/admin/settings
curl.exe -s http://127.0.0.1:7880/admin/models
curl.exe -s http://127.0.0.1:7880/admin/system

# 任务详情弹窗（前端资源里应包含这些函数/选择器）
curl.exe -s http://127.0.0.1:7880/ui/app.js | Select-String "openJobDetail","modal-job","dblclick"

# 任务详情接口
curl.exe -s http://127.0.0.1:7880/jobs/<job_id>
curl.exe -s -o result.bin -D - http://127.0.0.1:7880/jobs/<job_id>/result

# 带令牌时
curl.exe -s -H "Authorization: Bearer <token>" http://127.0.0.1:7880/admin/settings
```

手动验收任务详情：在「任务」页双击一条 `done` 的 TTS 任务，确认弹窗里能看到
`payload.text` 原文、音频播放器和 result JSON；再双击一条 `failed` 任务，确认
`error` 原文完整（换行没丢）。打开弹窗后等待 5 秒以上，确认列表没有因自动刷新而
重绘；关闭弹窗后列表恢复刷新。

测试：`python -m pytest -q`（`tests/test_admin.py` 覆盖设置读写、白名单与范围、
模型清单字段、档位切换、`/ui`、鉴权、restart 必须 confirm；
`tests/test_jobs_detail.py` 覆盖任务详情字段、`/jobs/{id}/result` 的音频/图片/其它
文件下发、目录白名单与 403/409、以及 `app.js` 里的详情弹窗标记）。
