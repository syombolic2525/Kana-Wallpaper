@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM =============================================================
REM Kana Wallpaper - One-click runner (v4)
REM  - Double-click this BAT:
REM      1) If venv/deps are missing -> run setup automatically
REM      2) Then launch the latest kana_wallpaper_launcher*.py via venv python
REM =============================================================

set "SETUP_SCRIPT=kana_wallpaper_env_setup.py"

if not exist "%SETUP_SCRIPT%" (
  echo [ERROR] Setup script not found: %SETUP_SCRIPT%
  echo         Place this BAT in the same folder as the setup script.
  pause
  exit /b 1
)

REM --- Choose python command (python -> py -3) ---
set "PY_CMD=python"
python -V >nul 2>nul
if errorlevel 1 (
  set "PY_CMD=py -3"
  py -3 -V >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python is not available. Install Python 3 and try again.
    pause
    exit /b 1
  )
)

REM --- Quick dependency check (skip setup if OK) ---
set "NEED_SETUP=1"
if exist ".venv\Scripts\python.exe" (
  if exist "_kana_state\models\face_detection_yunet_2023mar.onnx" (
    if exist "_kana_state\models\lbpcascade_animeface.xml" (
      if exist "_kana_state\models\yolov8x6_animeface.pt" (
        ".venv\Scripts\python.exe" -c "import numpy, cv2; from PIL import Image; import ultralytics" >nul 2>nul
        if not errorlevel 1 set "NEED_SETUP=0"
      )
    )
  )
)
if "%NEED_SETUP%"=="0" goto :DETECT_LAUNCHER

REM --- If model files are missing, auto-download them during setup ---
set "NEED_SMALL=0"
if not exist "_kana_state\models\face_detection_yunet_2023mar.onnx" set "NEED_SMALL=1"
if not exist "_kana_state\models\lbpcascade_animeface.xml" set "NEED_SMALL=1"

set "NEED_YOLO=0"
if not exist "_kana_state\models\yolov8x6_animeface.pt" set "NEED_YOLO=1"

set "DL_FLAGS="
if "%NEED_SMALL%"=="1" set "DL_FLAGS=!DL_FLAGS! --download-models"
if "%NEED_YOLO%"=="1" set "DL_FLAGS=!DL_FLAGS! --download-yolo"

echo [INFO] Running setup (first time may take a while)...
call %PY_CMD% "%SETUP_SCRIPT%" %DL_FLAGS% --no-pause --no-run-scripts

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] venv python not found after setup: .venv\Scripts\python.exe
  pause
  exit /b 1
)

:DETECT_LAUNCHER
set "LAUNCHER="

REM --- Prefer PowerShell to pick max _vNNN, then newest timestamp ---
for /f "usebackq delims=" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$files = Get-ChildItem -File -Filter 'kana_wallpaper_launcher*.py' -ErrorAction SilentlyContinue; if(-not $files){ exit 2 }; $best = $files | Sort-Object @{Expression={ if($_.Name -match '_v(\d+)'){ [int]$matches[1] } else { -1 } }; Descending=$true}, @{Expression={$_.LastWriteTime}; Descending=$true} | Select-Object -First 1; $best.Name"`) do set "LAUNCHER=%%A"

REM --- Fallback: pick newest by modified time using DIR ---
if "%LAUNCHER%"=="" (
  for /f "delims=" %%F in ('dir /b /a:-d /o:-d "kana_wallpaper_launcher*.py" 2^>nul') do (
    set "LAUNCHER=%%F"
    goto :FOUND_LAUNCHER
  )
)

:FOUND_LAUNCHER
if "%LAUNCHER%"=="" (
  echo [ERROR] Launcher not found: kana_wallpaper_launcher*.py
  echo.
  echo [HINT] Ensure the launcher file exists in this folder ^(same as this BAT^).
  echo        Example: kana_wallpaper_launcher_STABLE_YYYYMMDD_vNNN_....py
  echo.
  echo [DEBUG] Python files in this folder:
  dir /b "*.py"
  echo.
  pause
  exit /b 1
)

echo [INFO] Launching: %LAUNCHER%
".venv\Scripts\python.exe" "%LAUNCHER%"

pause
endlocal