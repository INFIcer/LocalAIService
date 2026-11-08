# 启动本地 AI 服务（TTS + ASR + 声纹 + 媒体 + 人声分离）
#
# 新入口：python -m localai serve（服务层代码在 src\localai\）
# 本脚本只负责：设好环境变量 -> 用 engine\.venv 的 python 后台跑服务 -> 日志落 logs\。
#
# 特点：
#   - 常驻服务只挂监听，不预热模型；TTS / ASR / 声纹 / Demucs 都是第一次请求才加载；
#   - 空闲超过 -IdleSeconds 秒（默认 900 = 15 分钟）就自动卸载 TTS 模型；
#   - 空闲超过 -AsrIdleSeconds 秒（默认 900）就自动卸载 SenseVoice；
#   - 空闲超过 -SpeakerIdleSeconds 秒（默认 900）就自动卸载 campplus 声纹模型；
#   - 空闲超过 -SeparateIdleSeconds 秒（默认 900）就自动卸载 Demucs 分离模型；
#   - 卸载后再来请求会自动重新加载，四边各自独立、互不影响。
#
# 目录全部从脚本位置推导（本脚本在服务根目录下），所以：
#   - 真实目录 E:\LocalAIService 能用；
#   - 兼容别名 E:\IndexTTS（junction）也能用，旧调用方式不用改。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -IdleSeconds 1800
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -IdleSeconds 0   # 关闭 TTS 空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -AsrIdleSeconds 0 # 关闭 ASR 空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SpeakerIdleSeconds 0 # 关闭声纹空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SeparateIdleSeconds 0 # 关闭分离空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -IdleSeconds 60 -AsrIdleSeconds 60 -SpeakerIdleSeconds 60 -SeparateIdleSeconds 60  # 调试
param(
    [int]$IdleSeconds = 900,
    [int]$AsrIdleSeconds = 900,
    [int]$SpeakerIdleSeconds = 900,
    [int]$SeparateIdleSeconds = 900
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 数据根目录与源码目录：换机器/换盘只需改这里，或从外部设置 LOCALAI_HOME。
$env:LOCALAI_HOME = $root
$env:PYTHONPATH = Join-Path $root 'src'

$env:TTS_BACKEND = 'engine'
$env:LOCALAI_IDLE_UNLOAD = [string]$IdleSeconds
$env:LOCALAI_ASR_IDLE_UNLOAD = [string]$AsrIdleSeconds
$env:LOCALAI_SPEAKER_IDLE_UNLOAD = [string]$SpeakerIdleSeconds
$env:LOCALAI_SEPARATE_IDLE_UNLOAD = [string]$SeparateIdleSeconds
# 兼容旧变量名，方便旧脚本/旧文档继续生效
$env:INDEX_TTS_IDLE_UNLOAD = [string]$IdleSeconds
$env:INDEX_TTS_ASR_IDLE_UNLOAD = [string]$AsrIdleSeconds
$env:INDEX_TTS_SPEAKER_IDLE_UNLOAD = [string]$SpeakerIdleSeconds
$env:INDEX_TTS_SEPARATE_IDLE_UNLOAD = [string]$SeparateIdleSeconds
# 让 python 的日志按 UTF-8 写，中文在 service.*.log 里不会变成乱码
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Demucs / torch 的权重缓存固定放项目里，不散落到用户目录。
$env:TORCH_HOME = Join-Path $root 'models\torch'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir 'service.out.log'
$errLog = Join-Path $logDir 'service.err.log'
"[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] 启动服务 (入口=localai serve, backend=engine, TTS空闲卸载=${IdleSeconds}s, ASR空闲卸载=${AsrIdleSeconds}s, 声纹空闲卸载=${SpeakerIdleSeconds}s, 分离空闲卸载=${SeparateIdleSeconds}s)" | Out-File -FilePath $outLog -Append -Encoding UTF8

$python = Join-Path $root 'engine\.venv\Scripts\python.exe'

# 用 Start-Process 起子进程并重定向输出：这样 python 往 stderr 写日志也不会被
# PowerShell 当成 NativeCommandError 把脚本打断（之前踩过这个坑）。
$proc = Start-Process -FilePath $python -ArgumentList @('-m', 'localai', 'serve') -WorkingDirectory $root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
exit $proc.ExitCode