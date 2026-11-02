@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv-train\Scripts\python.exe" goto missing
".venv-train\Scripts\python.exe" tools\bowl_model.py validate %*
set RESULT=%ERRORLEVEL%
pause
exit /b %RESULT%
:missing
echo Run windows\setup_training.cmd first.
pause
exit /b 1
