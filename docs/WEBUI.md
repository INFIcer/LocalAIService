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
| **任务** | `/jobs` 队列列表（状态 / 类型 / 优先级 / 等待 / 耗时）、状态筛选、取消、点击查看详情与结果 JSON |
| **系统** | 版本 / Python / GPU / 磁盘占用 / 监听地址、开机自启开关、重启服务（输入 `RESTART` 二次确认）、日志查看器（out/err、行数） |

交互约定：

- 所有操作都有 loading / 成功 / 失败提示；失败直接展示后端 JSON 的 `detail`；
- 令牌只存在浏览器 `localStorage`，页面源码里没有真实 token；
- 下载任务轮询 `GET /admin/models/tasks/{task_id}`，同一时刻只允许一个下载。

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

# 带令牌时
curl.exe -s -H "Authorization: Bearer <token>" http://127.0.0.1:7880/admin/settings
```

测试：`python -m pytest -q`（`tests/test_admin.py` 覆盖设置读写、白名单与范围、
模型清单字段、档位切换、`/ui`、鉴权、restart 必须 confirm）。
