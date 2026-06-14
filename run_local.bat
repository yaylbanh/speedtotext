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

REM Chi cai khi THIEU thu vien (khoi cai lai moi lan -> mo nhanh)
%PYCMD% -c "import faster_whisper, gradio, ctranslate2; import importlib.util as u; assert u.find_spec('nvidia.cublas') and u.find_spec('nvidia.cudnn') and u.find_spec('nvidia.cuda_runtime')" >nul 2>&1
if errorlevel 1 (
  echo [*] Thieu thu vien -^> dang cai dat lan dau...
  %PYCMD% -m pip install -q -r requirements.txt
) else (
  echo [*] Thu vien da day du -^> bo qua cai dat.
)

echo.
echo [*] Khoi dong Speed To Text (chay LOCAL tren GPU cua may ban)...
echo [*] Trinh duyet se tu mo http://127.0.0.1:7860
echo [*] Lan dau se tai model large-v3 (~3GB) ve may, vui long doi.
echo.
%PYCMD% app.py

echo.
echo [!] App da dong hoac co loi. Xem log ben tren.
pause
