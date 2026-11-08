# 重启本地 AI 服务
#
# 由 POST /admin/restart 以「脱离进程」的方式调用：
#   1. 等待约 1 秒，让 HTTP 响应先发出去；
#   2. 结束监听指定端口（默认 7880）的进程；
#   3. 用 WMI 在后台重新执行 start_ai_service.ps1（不阻塞、不弹窗）。
#
# 手动用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\tools\restart_service.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File E:\LocalAIService\tools\restart_service.ps1 -Port 7880
param(
    [int]$Port = 7880,
    [string]$StartScript = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($StartScript)) {
    $StartScript = Join-Path $root 'start_ai_service.ps1'
}

# 重启是后台动作，页面看不到输出；所有关键步骤写进 logs/restart.log 方便排错。
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir 'restart.log'
function Write-RestartLog([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] $Message"
    Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
}

Write-RestartLog "收到重启请求：端口=$Port 启动脚本=$StartScript"

# 给接口留出发送响应的时间。
Start-Sleep -Seconds 1

$pids = @()
try {
    $pids = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
} catch {
    # 极老系统没有 Get-NetTCPConnection 时退回 netstat。
    Write-RestartLog "Get-NetTCPConnection 不可用，改用 netstat：$($_.Exception.Message)"
    $pids = netstat -ano | Select-String ":$Port\s+.*LISTENING" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Select-Object -Unique
}
Write-RestartLog "监听端口 $Port 的进程：$($pids -join ', ')"

foreach ($processId in $pids) {
    if (-not $processId) { continue }
    try {
        $proc = Get-Process -Id $processId -ErrorAction Stop
        if ($proc.ProcessName -match 'python') {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-RestartLog "已结束进程 $processId（$($proc.ProcessName)）"
        } else {
            Write-RestartLog "跳过进程 $processId（$($proc.ProcessName)），不是 python"
        }
    } catch {
        # 进程可能已经退出；继续下一步。
        Write-RestartLog "结束进程 $processId 失败（可能已退出）：$($_.Exception.Message)"
    }
}

# 等端口真正释放，再拉起新进程。
Start-Sleep -Milliseconds 800

$cmd = 'powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $StartScript + '"'
$startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$startup.ShowWindow = 0
[void]([wmiclass]"Win32_Process").Create($cmd, (Split-Path -Parent $StartScript), $startup)
Write-RestartLog "已通过 WMI 拉起启动脚本：$cmd"
