<#
.SYNOPSIS
    启动 IndexTTS-2.5 本地 TTS 服务（首次运行会自动建 venv 并装依赖）。

.DESCRIPTION
    - venv 固定在 E:\IndexTTS\svc\.venv，已存在就直接复用；
    - 优先用 Python 3.13 创建 venv，找不到再退回 PATH 里的 python；
    - 依赖只在 import 失败时才安装，重复运行不会重复装；
    - 最后前台启动 service.py（监听 127.0.0.1:7880）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File E:\IndexTTS\svc\start.ps1
#>

$ErrorActionPreference = 'Stop'

$svcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $svcDir

$venvDir = Join-Path $svcDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

function Write-Step([string]$message) {
    Write-Host "[IndexTTS] $message" -ForegroundColor Cyan
}

# ---- 1. 选择创建 venv 用的 Python（优先 3.13） ----
$pythonCandidates = @(
    'C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe',
    'C:\Python314\python.exe'
)
$bootstrapPython = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path $candidate) {
        $bootstrapPython = $candidate
        break
    }
}
if (-not $bootstrapPython) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $bootstrapPython = $cmd.Source }
}
if (-not $bootstrapPython) {
    throw '找不到 Python。请安装 Python 3.13，或把 python.exe 加进 PATH。'
}

# ---- 2. 建/复用 venv ----
if (-not (Test-Path $venvPython)) {
    Write-Step "创建 venv：$venvDir（使用 $bootstrapPython）"
    & $bootstrapPython -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "创建 venv 失败（退出码 $LASTEXITCODE）。"
    }
}
else {
    Write-Step "复用已有 venv：$venvDir"
}

# ---- 3. 检查依赖，缺了才装 ----
Write-Step '检查依赖 fastapi / uvicorn / httpx / soundfile ...'
& $venvPython -c "import fastapi, uvicorn, httpx, soundfile" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Step '安装依赖（首次运行需要联网，约 1 分钟）...'
    & $venvPython -m pip install --disable-pip-version-check --no-input fastapi "uvicorn[standard]" httpx soundfile
    if ($LASTEXITCODE -ne 0) {
        throw "安装依赖失败（退出码 $LASTEXITCODE）。请检查网络或代理后重试。"
    }
    & $venvPython -c "import fastapi, uvicorn, httpx, soundfile"
    if ($LASTEXITCODE -ne 0) {
        throw '依赖安装后仍无法 import，请查看上面的 pip 输出。'
    }
}
else {
    Write-Step '依赖已安装，跳过。'
}

# ---- 4. 前台启动 ----
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    # 让控制台按 UTF-8 显示中文日志（不影响服务内部编码）。
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
}
catch {
    # 某些宿主（如把输出重定向到文件）不允许改编码，忽略即可。
}

Write-Step '启动服务：http://127.0.0.1:7880（Ctrl+C 停止）'
& $venvPython (Join-Path $svcDir 'service.py')
exit $LASTEXITCODE
