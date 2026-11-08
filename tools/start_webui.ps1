<#
.SYNOPSIS
    LocalAIService 管理界面启动器。

.DESCRIPTION
    1) 先探测服务是否已在运行（默认 http://127.0.0.1:7880/health）；
    2) 已在运行 → 直接用默认浏览器打开 /ui；
    3) 没运行 → 后台启动服务（start_ai_service.ps1），轮询等它就绪，再打开 /ui；
    4) 等不到就打印日志位置并提示排错。

.PARAMETER Port
    服务端口，默认 7880（也可用环境变量 LOCALAI_PORT）。

.PARAMETER NoBrowser
    只启动/检查服务，不打开浏览器（自测用）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File E:\LocalAIService\tools\start_webui.ps1
#>
param(
    [int]$Port = 0,
    [switch]$NoBrowser,
    [int]$WaitSeconds = 90
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root 'start_ai_service.ps1'))) { $root = 'E:\LocalAIService' }
if ($Port -le 0) { $Port = if ($env:LOCALAI_PORT) { [int]$env:LOCALAI_PORT } else { 7880 } }
$base = "http://127.0.0.1:$Port"
$healthUrl = "$base/health"
$uiUrl = "$base/ui"

function Test-Health {
    try {
        $r = Invoke-WebRequest $healthUrl -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch {
        return $false
    }
}

function Open-Ui {
    if ($NoBrowser) { Write-Host "[跳过] 打开浏览器：$uiUrl" -ForegroundColor DarkGray; return }
    Write-Host "打开管理界面：$uiUrl" -ForegroundColor Green
    Start-Process $uiUrl
}

Write-Host "LocalAIService 管理界面启动器" -ForegroundColor Cyan
Write-Host "服务地址：$base"

if (Test-Health) {
    Write-Host "服务已在运行 ✓" -ForegroundColor Green
    Open-Ui
    exit 0
}

Write-Host "服务没在运行，开始后台启动……" -ForegroundColor Yellow
$launcher = Join-Path $root 'start_ai_service.ps1'
if (-not (Test-Path $launcher)) {
    Write-Host "找不到启动脚本：$launcher" -ForegroundColor Red
    exit 2
}
try {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', "`"$launcher`""
    ) -WindowStyle Hidden | Out-Null
} catch {
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 3
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 800
    if (Test-Health) { $ready = $true; break }
    Write-Host "." -NoNewline
}
Write-Host ""

if ($ready) {
    Write-Host "服务已就绪 ✓" -ForegroundColor Green
    Open-Ui
    exit 0
}

Write-Host "等了 $WaitSeconds 秒服务还没起来。" -ForegroundColor Red
$logDir = Join-Path $root 'logs'
Write-Host "查日志： $logDir\service.out.log / service.err.log" -ForegroundColor Yellow
Write-Host "手动启动： powershell -ExecutionPolicy Bypass -File `"$launcher`"" -ForegroundColor Yellow
exit 4
