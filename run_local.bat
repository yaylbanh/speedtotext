@echo off
chcp 65001 >nul
title Speed To Text - LOCAL (GPU)
cd /d "%~dp0"

REM Chay local: khong dung tunnel gradio, mo thang trinh duyet 127.0.0.1
set STT_SHARE=0
set PYTHONUTF8=1

REM --- Tim Python 3.12 ---
set "PYCMD="
py -3.12 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.12"
if not defined PYCMD if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not defined PYCMD (
  echo [!] Khong tim thay Python 3.12.
  echo     Cai Python 3.12, hoac sua duong dan trong file run_local.bat nay.
  pause
  exit /b 1
)

echo [*] Python: %PYCMD%
echo [*] Dang kiem tra/cai thu vien (faster-whisper, gradio)...
%PYCMD% -m pip install -q -r requirements.txt

echo.
echo [*] Khoi dong Speed To Text (chay LOCAL tren GPU cua may ban)...
echo [*] Trinh duyet se tu mo http://127.0.0.1:7860
echo [*] Lan dau se tai model large-v3 (~3GB) ve may, vui long doi.
echo.
%PYCMD% app.py

echo.
echo [!] App da dong hoac co loi. Xem log ben tren.
pause
