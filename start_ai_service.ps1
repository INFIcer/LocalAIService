# 启动本地 AI 服务（TTS + ASR + 声纹 + 媒体 + 人声分离）
#
# 新入口：python -m localai serve（服务层代码在 src\localai\）
# 本脚本只负责：设好环境变量 -> 用 engine\.venv 的 python 后台跑服务 -> 日志落 logs\。
#
# 特点：
#   - 常驻服务只挂监听，不预热模型；TTS / ASR / 声纹 / Demucs 都是第一次请求才加载；
#   - 空闲超过 -IdleSeconds 秒（默认 900 = 15 分钟）就自动卸载 TTS 模型；
#   - 空闲超过 -AsrIdleSeconds 秒（默认 900）就自动卸载 SenseVoice；
#   - 空闲超过 -WhisperIdleSeconds 秒（默认 900）就自动卸载 faster-whisper；
#   - 空闲超过 -SpeakerIdleSeconds 秒（默认 900）就自动卸载 campplus 声纹模型；
#   - 空闲超过 -SeparateIdleSeconds 秒（默认 900）就自动卸载 Demucs 分离模型；
#   - 空闲超过 -TranslateIdleSeconds 秒（默认 900）就自动卸载 NLLB 翻译模型；
#   - 卸载后再来请求会自动重新加载，五边各自独立、互不影响。
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
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -WhisperIdleSeconds 0 # 关闭 whisper 空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SpeakerIdleSeconds 0 # 关闭声纹空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -SeparateIdleSeconds 0 # 关闭分离空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -TranslateIdleSeconds 0 # 关闭翻译空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -VisionIdleSeconds 0 # 关闭视觉空闲卸载
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_ai_service.ps1 -IdleSeconds 60 -AsrIdleSeconds 60 -SpeakerIdleSeconds 60 -SeparateIdleSeconds 60 -TranslateIdleSeconds 60  # 调试
param(
    [int]$IdleSeconds = 900,
    [int]$AsrIdleSeconds = 900,
    [int]$WhisperIdleSeconds = 900,
    [int]$SpeakerIdleSeconds = 900,
    [int]$SeparateIdleSeconds = 900,
    [int]$TranslateIdleSeconds = 900,
    [int]$VisionIdleSeconds = 900
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 同一个服务只允许一个启动脚本跑起来。两次重启挨得太近时，两个 python 会同时
# 往 logs\service.out.log 追加，文件里会留下成片的 NUL 空洞，WebUI 读日志就成乱码。
$启动锁 = New-Object System.Threading.Mutex($false, 'Global\LocalAIService-Launch')
$已持锁 = $false
try {
    $已持锁 = $启动锁.WaitOne(60000)
    if (-not $已持锁) {
        Write-Host '另一个启动脚本还在运行（服务可能已启动），本次跳过。' -ForegroundColor Yellow
        exit 0
    }
    $探测 = New-Object System.Net.Sockets.TcpClient
    try {
        $探测.Connect('127.0.0.1', 7880)
        $已在运行 = $探测.Connected
    } catch {
        $已在运行 = $false
    } finally {
        $探测.Dispose()
    }
    if ($已在运行) {
        Write-Host '服务已经在 127.0.0.1:7880 运行，本次跳过。' -ForegroundColor Yellow
        exit 0
    }

# 数据根目录与源码目录：换机器/换盘只需改这里，或从外部设置 LOCALAI_HOME。
$env:LOCALAI_HOME = $root
$env:PYTHONPATH = Join-Path $root 'src'

$env:TTS_BACKEND = 'engine'
$env:LOCALAI_IDLE_UNLOAD = [string]$IdleSeconds
$env:LOCALAI_ASR_IDLE_UNLOAD = [string]$AsrIdleSeconds
$env:LOCALAI_WHISPER_IDLE_UNLOAD = [string]$WhisperIdleSeconds
$env:LOCALAI_SPEAKER_IDLE_UNLOAD = [string]$SpeakerIdleSeconds
$env:LOCALAI_SEPARATE_IDLE_UNLOAD = [string]$SeparateIdleSeconds
$env:LOCALAI_TRANSLATE_IDLE_UNLOAD = [string]$TranslateIdleSeconds
$env:LOCALAI_VISION_IDLE_UNLOAD = [string]$VisionIdleSeconds
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
"[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] 启动服务 (入口=localai serve, backend=engine, TTS空闲卸载=${IdleSeconds}s, ASR空闲卸载=${AsrIdleSeconds}s, whisper空闲卸载=${WhisperIdleSeconds}s, 声纹空闲卸载=${SpeakerIdleSeconds}s, 分离空闲卸载=${SeparateIdleSeconds}s, 翻译空闲卸载=${TranslateIdleSeconds}s, 视觉空闲卸载=${VisionIdleSeconds}s)" | Out-File -FilePath $outLog -Append -Encoding UTF8

$python = Join-Path $root 'engine\.venv\Scripts\python.exe'

    # 用 Start-Process 起子进程并重定向输出：这样 python 往 stderr 写日志也不会被
    # PowerShell 当成 NativeCommandError 把脚本打断（之前踩过这个坑）。
    $proc = Start-Process -FilePath $python -ArgumentList @('-m', 'localai', 'serve') -WorkingDirectory $root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
    exit $proc.ExitCode
} finally {
    if ($已持锁) { [void]$启动锁.ReleaseMutex() }
    $启动锁.Dispose()
}