@echo off
rem Double-clickable launcher for sync_results.ps1 (PowerShell scripts cannot
rem be started directly from Explorer / cmd `start`). %~dp0 = this file's dir,
rem so it works from any location. Extra args pass through (e.g. -DryRun).
powershell -NoProfile -File "%~dp0sync_results.ps1" %*
echo.
pause
