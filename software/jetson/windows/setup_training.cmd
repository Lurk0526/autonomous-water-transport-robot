@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set TORCH_INDEX=https://download.pytorch.org/whl/cu128
set CUDA_CHECK=--require-cuda
if /i "%~1"=="cpu" set TORCH_INDEX=https://download.pytorch.org/whl/cpu
if /i "%~1"=="cpu" set CUDA_CHECK=
py -3.12 -c "import struct,sys; assert struct.calcsize('P')==8; print(sys.version)"
if errorlevel 1 goto fail
if not exist ".venv-train\Scripts\python.exe" py -3.12 -m venv .venv-train
if errorlevel 1 goto fail
".venv-train\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv-train\Scripts\python.exe" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url %TORCH_INDEX%
if errorlevel 1 goto fail
".venv-train\Scripts\python.exe" -m pip install "third_party\ultralytics-8.4.178-py3-none-any.whl" -r windows\requirements-training.txt
if errorlevel 1 goto fail
".venv-train\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
".venv-train\Scripts\python.exe" tools\check_training_env.py %CUDA_CHECK%
if errorlevel 1 goto fail
echo Training environment ready. RTX 4060 default: imgsz 320, batch 8, workers 0.
pause
exit /b 0
:fail
echo Installation or GPU check failed. Read the error above and README_WINDOWS.md.
pause
exit /b 1
