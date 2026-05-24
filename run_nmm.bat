@echo off
setlocal EnableDelayedExpansion
title Nine Men's Morris

set "NMM_DIR=%~dp0"
set "VENV_UV=%NMM_DIR%.venv\Scripts\uvicorn.exe"
set "HOST=127.0.0.1"
set "PORT=8000"

if not exist "%VENV_UV%" (
    echo [NMM] ERROR: .venv not found. Run install.ps1 first.
    pause & exit /b 1
)

rem -- Start Ollama if installed but not yet running --
where ollama >nul 2>&1
if %errorlevel%==0 (
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [NMM] Starting Ollama service...
        start /B "" ollama serve
        timeout /t 3 /nobreak >nul
    ) else (
        echo [NMM] Ollama already running.
    )
)

rem -- Fall back to port 8080 if 8000 is busy --
netstat -an | findstr /C:":%PORT% " | findstr /C:"LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [NMM] Port %PORT% is in use, switching to 8080...
    set "PORT=8080"
)

echo [NMM] Starting Nine Men's Morris at http://%HOST%:%PORT% ...
cd /d "%NMM_DIR%"

rem -- Launch uvicorn in this console window --
start /B "" "%VENV_UV%" web.app:app --host %HOST% --port %PORT%

rem -- Wait for server then open browser --
timeout /t 2 /nobreak >nul
echo [NMM] Opening browser at http://%HOST%:%PORT%
start "" "http://%HOST%:%PORT%"

echo.
echo [NMM] Server is running. Close this window to stop.
echo.
pause >nul
