@echo off
rem ============================================================================
rem  Nine Men's Morris -- Windows installer (batch wrapper)
rem
rem  This double-clickable wrapper exists so users do not have to fight with
rem  PowerShell's Execution Policy. It launches install.ps1 with -ExecutionPolicy
rem  Bypass for THIS process only -- no permanent system change is made.
rem
rem  Usage:
rem     install.bat                 (interactive; asks about Ollama)
rem     install.bat /noollama       (skip Ollama install)
rem     install.bat /yes            (install Ollama without prompting)
rem     install.bat /model NAME     (override LLM model, e.g. mistral:7b)
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
title Nine Men's Morris -- Installer

set "NMM_DIR=%~dp0"
rem Strip trailing backslash for nicer display, but keep original for paths.
set "NMM_DIR_DISP=%NMM_DIR%"
if "%NMM_DIR_DISP:~-1%"=="\" set "NMM_DIR_DISP=%NMM_DIR_DISP:~0,-1%"

set "PS_SCRIPT=%NMM_DIR%install.ps1"

if not exist "%PS_SCRIPT%" (
    echo [NMM] ERROR: install.ps1 not found next to install.bat.
    echo [NMM] Expected at: %PS_SCRIPT%
    pause
    exit /b 1
)

rem -- Translate batch-style flags into PowerShell parameters ----------------
set "PS_ARGS="
:parse
if "%~1"=="" goto runps
if /I "%~1"=="/noollama"  set "PS_ARGS=!PS_ARGS! -NoOllama" & shift & goto parse
if /I "%~1"=="-noollama"  set "PS_ARGS=!PS_ARGS! -NoOllama" & shift & goto parse
if /I "%~1"=="--noollama" set "PS_ARGS=!PS_ARGS! -NoOllama" & shift & goto parse
if /I "%~1"=="/yes"       set "PS_ARGS=!PS_ARGS! -Yes" & shift & goto parse
if /I "%~1"=="-yes"       set "PS_ARGS=!PS_ARGS! -Yes" & shift & goto parse
if /I "%~1"=="/y"         set "PS_ARGS=!PS_ARGS! -Yes" & shift & goto parse
if /I "%~1"=="/model"     set "PS_ARGS=!PS_ARGS! -Model ""%~2""" & shift & shift & goto parse
if /I "%~1"=="-model"     set "PS_ARGS=!PS_ARGS! -Model ""%~2""" & shift & shift & goto parse
rem Unknown flag -- pass it through verbatim.
set "PS_ARGS=!PS_ARGS! %~1"
shift
goto parse

:runps
rem -- Pick the PowerShell host: prefer pwsh (PS 7+), fall back to powershell --
set "PS_EXE="
where pwsh >nul 2>&1 && set "PS_EXE=pwsh"
if "%PS_EXE%"=="" (
    where powershell >nul 2>&1 && set "PS_EXE=powershell"
)
if "%PS_EXE%"=="" (
    echo [NMM] ERROR: Neither 'pwsh' nor 'powershell' was found on PATH.
    echo [NMM] PowerShell ships with Windows; if this is missing your system is unusual.
    pause
    exit /b 1
)

echo [NMM] Using %PS_EXE% to run install.ps1 ...
echo.

rem Quote the script path so spaces in the install directory survive.
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %PS_ARGS%
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo [NMM] Installer exited with code %RC%.
) else (
    echo [NMM] Installer finished successfully.
)
echo.
echo Press any key to close this window...
pause >nul
exit /b %RC%
