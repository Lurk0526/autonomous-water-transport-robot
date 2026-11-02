@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
py -3.12 -c "import struct,sys; assert struct.calcsize('P')==8; print(sys.version)"
if errorlevel 1 goto fail
if not exist ".venv-labelme\Scripts\python.exe" py -3.12 -m venv .venv-labelme
if errorlevel 1 goto fail
".venv-labelme\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv-labelme\Scripts\python.exe" -m pip install "third_party\labelme-7.8.1-py3-none-any.whl"
if errorlevel 1 goto fail
".venv-labelme\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
echo Annotation environment ready. Run windows\start_annotation.cmd
pause
exit /b 0
:fail
echo Installation failed. Install 64-bit Python 3.12 and ensure py -3.12 works.
pause
exit /b 1
