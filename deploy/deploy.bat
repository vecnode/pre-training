@echo off
rem -----------------------------------------------------------------------------
rem Deploy the trained LoRA adapter inference server (FastAPI + uvicorn).
rem Bootstraps the local uv environment, then serves deploy/app.py.
rem Can be started standalone, or remotely by the metaagent controller
rem (POST /api/adapter/launch -> METAAGENT_ADAPTER_LAUNCH_CMD).
rem
rem Usage:  deploy.bat [host] [port]      (defaults: 127.0.0.1 8008)
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve deploy/ dir and the project root (its parent).
pushd "%~dp0" >nul
set "DEPLOY_DIR=%CD%"
popd >nul
pushd "%~dp0.." >nul
set "PROJECT_DIR=%CD%"
popd >nul

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"

rem Optional host/port overrides.
set "APP_HOST=%~1"
set "APP_PORT=%~2"
if "%APP_HOST%"=="" set "APP_HOST=127.0.0.1"
if "%APP_PORT%"=="" set "APP_PORT=8008"

title LoRA Adapter Inference Server

rem Ensure the local uv environment + CUDA-ready torch are present.
call "%PROJECT_DIR%\uv_setup.bat"
if errorlevel 1 (
    echo.
    echo UV setup failed. Cannot start the inference server.
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo.
    echo Local venv python not found at "%VENV_PY%".
    echo Run uv_setup.bat from the project root first.
    exit /b 1
)

echo.
echo Starting adapter inference server on http://%APP_HOST%:%APP_PORT%
echo   model source: deploy/merged_model (if present) else training/runs adapter
echo   front-end: http://%APP_HOST%:%APP_PORT%/   API: /api/summarize, /api/health
echo.
"%VENV_PY%" "%DEPLOY_DIR%\app.py" --host %APP_HOST% --port %APP_PORT%

endlocal
