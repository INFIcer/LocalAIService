' LocalAIService tray launcher.
' Double-click this file: starts the service in the background and shows a
' notification-area icon. No console window stays open.
'
' pythonw.exe has no console, so a crash before the Python side can show its
' dialog would look like "nothing happened". That is why this script waits a
' few seconds and checks that a tray process is really alive. A Python-side
' failure keeps its process running while its error dialog is open, so the two
' checks never report the same failure twice.
Option Explicit

Const WATCH_MS = 6000

Dim fso, shell, root, pythonw, command, env, pythonpath
Dim errorLog, startupLog
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\engine\.venv\Scripts\pythonw.exe"
errorLog = root & "\logs\tray-error.log"
startupLog = root & "\logs\tray.log"

If Not fso.FileExists(pythonw) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "See docs\INSTALL.md to prepare it first.", vbExclamation, "LocalAIService"
    WScript.Quit 1
End If

' Same as start_ai_service.ps1: source checkout works even without pip install -e .
Set env = shell.Environment("PROCESS")
pythonpath = root & "\src"
If Len(env("PYTHONPATH")) > 0 Then pythonpath = pythonpath & ";" & env("PYTHONPATH")
env("PYTHONPATH") = pythonpath
env("PYTHONUTF8") = "1"
env("PYTHONIOENCODING") = "utf-8"

shell.CurrentDirectory = root
command = Chr(34) & pythonw & Chr(34) & " -m localai tray"

On Error Resume Next
shell.Run command, 0, False
If Err.Number <> 0 Then
    MsgBox "Could not start the tray:" & vbCrLf & Err.Description & vbCrLf & vbCrLf & _
           "Error log:" & vbCrLf & errorLog, vbCritical, "LocalAIService"
    WScript.Quit 1
End If
Err.Clear
On Error GoTo 0

' A second double-click is fine: the running tray keeps the count above zero,
' and the new process explains itself with its own dialog.
WScript.Sleep WATCH_MS
If TrayProcessCount() = 0 Then
    MsgBox "The tray did not stay running." & vbCrLf & vbCrLf & _
           "Error log:" & vbCrLf & errorLog & vbCrLf & _
           "Startup log:" & vbCrLf & startupLog, vbCritical, "LocalAIService"
    WScript.Quit 1
End If

' Counts pythonw processes running "localai tray". Returns -1 when WMI is not
' available, in which case the caller must not read it as "no tray running".
Function TrayProcessCount()
    Dim wmi, items, item, total
    total = 0
    On Error Resume Next
    Set wmi = GetObject("winmgmts:\\.\root\cimv2")
    Set items = wmi.ExecQuery("SELECT ProcessId FROM Win32_Process WHERE Name='pythonw.exe' AND CommandLine LIKE '%localai tray%'")
    For Each item In items
        total = total + 1
    Next
    If Err.Number <> 0 Then
        total = -1
        Err.Clear
    End If
    On Error GoTo 0
    TrayProcessCount = total
End Function
