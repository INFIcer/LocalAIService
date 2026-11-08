@echo off
chcp 65001 >nul
setlocal
set "SCRIPT=%~dp0tools\start_webui.ps1"
if not exist "%SCRIPT%" set "SCRIPT=E:\LocalAIService\tools\start_webui.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
if errorlevel 1 pause
endlocal
