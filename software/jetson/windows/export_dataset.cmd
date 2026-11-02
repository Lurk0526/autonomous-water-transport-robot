@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv-labelme\Scripts\python.exe" goto missing
".venv-labelme\Scripts\python.exe" tools\export_bowl_dataset.py %*
set RESULT=%ERRORLEVEL%
pause
exit /b %RESULT%
:missing
echo Run windows\setup_annotation.cmd first.
pause
exit /b 1
