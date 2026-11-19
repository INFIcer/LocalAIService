' LocalAIService tray launcher.
' Double-click this file: starts the service in the background and shows a
' notification-area icon. No console window stays open.
Option Explicit

Dim fso, shell, root, pythonw, command, env, pythonpath
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\engine\.venv\Scripts\pythonw.exe"

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
shell.Run command, 0, False
