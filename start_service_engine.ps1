# 兼容壳：旧脚本名 start_service_engine.ps1
# ---------------------------------------------------------------
# 项目从 IndexTTS 改名为 LocalAIService 后，启动脚本新名字是
# start_ai_service.ps1（同目录）。为了不让旧计划任务、旧快捷方式、
# 旧命令行调用失败，保留这个同名壳，原样转交参数。
#
# 旧调用方式继续有效：
#   powershell -ExecutionPolicy Bypass -File E:\IndexTTS\start_service_engine.ps1
#   powershell -ExecutionPolicy Bypass -File E:\LocalAIService\start_service_engine.ps1
param(
    [int]$IdleSeconds = 900,
    [int]$AsrIdleSeconds = 900,
    [int]$SpeakerIdleSeconds = 900
)

$target = Join-Path $PSScriptRoot 'start_ai_service.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "找不到启动脚本：$target"
}

& $target -IdleSeconds $IdleSeconds -AsrIdleSeconds $AsrIdleSeconds -SpeakerIdleSeconds $SpeakerIdleSeconds
exit $LASTEXITCODE
