# =============================================================================
# 本地 AI 服务（TTS + ASR + 声纹 + 媒体）启动脚本
# =============================================================================
# 用 <服务根目录>\engine\.venv 里的 Python 启动 <服务根目录>\svc\service.py，
# 并把 TTS_BACKEND 固定为 engine（不经过 ComfyUI）。
#
# 目录从脚本自身位置推出来：本脚本在 engine\ 下，所以服务根目录 = engine 的上一级。
# 改名（IndexTTS -> LocalAIService）或从兼容 junction E:\IndexTTS 启动都能正常工作。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1
#       前台启动，日志直接打在窗口里，Ctrl+C 停止。
#
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Background
#       后台启动，日志写 <服务根目录>\logs\service.out.log / service.err.log。
#
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\engine\start_engine_service.ps1 -Restart -Background
#       先停掉占用 7880 的旧 service.py，再后台启动（只杀命令行里带 service.py 的进程）。
#
# 参数：
#   -Background  后台运行
#   -Restart     启动前停掉旧的 service.py（不会碰 ComfyUI 的 python）
#   -Port        监听端口，默认 7880（本次任务要求固定 7880）
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Background,
    [switch]$Restart,
    [int]$Port = 7880
)

$ErrorActionPreference = 'Stop'

# 控制台按 UTF-8 输出，避免 Windows PowerShell 5.1 把中文显示成乱码。
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

# $PSScriptRoot 是脚本所在目录（engine），上一级就是服务根目录。
# 注意：从 junction 路径调用时 $PSScriptRoot 会是 E:\IndexTTS\engine，
# 但下面的路径照样能打开（junction 透明），日志/模型都落在真实目录里。
$EngineRoot  = $PSScriptRoot
$ServiceRoot = Split-Path -Parent $EngineRoot
$Python      = Join-Path $EngineRoot '.venv\Scripts\python.exe'
$ServicePy   = Join-Path $ServiceRoot 'svc\service.py'
$ServiceDir  = Join-Path $ServiceRoot 'svc'
$LogDir      = Join-Path $ServiceRoot 'logs'
$PidFile     = Join-Path $LogDir 'service.pid'
$OutLog      = Join-Path $LogDir 'service.out.log'
$ErrLog      = Join-Path $LogDir 'service.err.log'

function Write-Step([string]$Message) {
    Write-Host "[LocalAIService] $Message" -ForegroundColor Cyan
}

function Test-ServiceProcess([int]$ProcessId) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    return ($null -ne $proc -and $proc.CommandLine -like "*service.py*")
}

# ---------------------------------------------------------------------------
# 1. 环境检查
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $Python)) {
    throw "找不到引擎 venv：$Python`n请先按 README 的『安装引擎环境』一节创建并安装依赖。"
}
if (-not (Test-Path -LiteralPath $ServicePy)) {
    throw "找不到服务文件：$ServicePy"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Step "检查运行依赖（fastapi / uvicorn / soundfile / torch）..."
& $Python -c "import fastapi, uvicorn, soundfile, torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Step "缺少 Web/音频依赖，尝试补装 fastapi uvicorn httpx soundfile ..."
    & $Python -m pip install --disable-pip-version-check --progress-bar off fastapi uvicorn httpx soundfile
    if ($LASTEXITCODE -ne 0) { throw "依赖安装失败，请查看上面的 pip 输出。" }
    & $Python -c "import fastapi, uvicorn, soundfile, torch"
    if ($LASTEXITCODE -ne 0) { throw "依赖仍不完整；torch 请参考 README 用 cu130 索引单独安装。" }
}

# ---------------------------------------------------------------------------
# 2. 端口检查 / 可选重启旧服务
# ---------------------------------------------------------------------------
$env:TTS_BACKEND = 'engine'
$env:INDEX_TTS_PORT = "$Port"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $oldPid = [int]$listener.OwningProcess
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    $cmdline = if ($proc) { $proc.CommandLine } else { '(无法读取命令行)' }

    if ($Restart -and (Test-ServiceProcess $oldPid)) {
        Write-Step "端口 $Port 被旧 service.py 占用（PID=$oldPid），按 -Restart 停掉它..."
        Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    elseif ($Restart) {
        throw "端口 $Port 被 PID=$oldPid 占用，但命令行不像 service.py，拒绝结束它：`n$cmdline"
    }
    else {
        throw "端口 $Port 已被占用（PID=$oldPid）。`n命令行：$cmdline`n如果这是旧服务，加 -Restart 参数；否则先手动停掉占用进程。"
    }
}

# ---------------------------------------------------------------------------
# 3. 启动
# ---------------------------------------------------------------------------
Write-Step "backend=engine  port=$Port"
Write-Step "python=$Python"
Write-Step "service=$ServicePy"

if ($Background) {
    $process = Start-Process -FilePath $Python `
        -ArgumentList @($ServicePy) `
        -WorkingDirectory $ServiceDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -PassThru

    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
    Write-Step "已后台启动，PID=$($process.Id)"
    Write-Step "日志：$OutLog / $ErrLog"

    # 等 /health 可用（最多 30 秒；模型本身是首次请求才加载，所以启动很快）
    $healthUrl = "http://127.0.0.1:$Port/health"
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
            if ($response.ok) {
                $ready = $true
                Write-Step "服务已就绪：$healthUrl"
                $response | ConvertTo-Json -Depth 5
                break
            }
        }
        catch {
            if ($process.HasExited) {
                throw "服务进程已退出（退出码 $($process.ExitCode)），请查看 $ErrLog"
            }
        }
    }
    if (-not $ready) {
        Write-Warning "30 秒内 /health 还没返回 ok，请查看 $ErrLog 和 $OutLog"
    }
}
else {
    Write-Step "前台启动（Ctrl+C 停止）..."
    & $Python $ServicePy
}
