@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv-labelme\Scripts\labelme.exe" goto missing
set "IMAGE_FOLDER=%CD%\datasets\target_bowl\source"
if not "%~1"=="" set "IMAGE_FOLDER=%~1"
".venv-labelme\Scripts\labelme.exe" "%IMAGE_FOLDER%" --config "%CD%\annotation\labelme-bowl.yaml"
if errorlevel 1 pause
exit /b
:missing
echo Run windows\setup_annotation.cmd first.
pause
exit /b 1
