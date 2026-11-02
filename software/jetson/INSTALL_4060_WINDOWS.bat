@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
python -m pip install -r requirements-road-4060.txt
if errorlevel 1 goto fail
python tools\check_training_bundle.py --gpu
if errorlevel 1 goto fail
echo 安装与 GPU 检查完成。接下来请按 RTX4060训练指南.md 运行。
pause
exit /b 0
:fail
echo 安装或检查失败，请保留上方报错，不要继续训练。
pause
exit /b 1
