@echo off
REM ============================================================================
REM NEURECTOMY Desktop Application - One-Click Launcher
REM ============================================================================
REM This script launches the pre-built NEURECTOMY desktop application
REM No build required - just click and run!
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                                                                ║
echo ║         NEURECTOMY - Dimensional Forge v0.1.0                 ║
echo ║              Desktop Application Launcher                      ║
echo ║                                                                ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Check for the MSI installer
set "MSI_PATH=%~dp0apps\spectrum-workspace\src-tauri\target\release\bundle\msi\NEURECTOMY_0.1.0_x64_en-US.msi"
set "EXE_PATH=%~dp0apps\spectrum-workspace\src-tauri\target\release\bundle\nsis\NEURECTOMY_0.1.0_x64-setup.exe"

echo [*] Checking for NEURECTOMY installation...

REM First, try to find an installed version
for /f "tokens=*" %%A in ('powershell -NoProfile -Command "Get-ItemProperty -Path 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like '*NEURECTOMY*' } | Select-Object -ExpandProperty InstallLocation | Select-Object -First 1 2>/dev/null"') do set "INSTALLED_PATH=%%A"

if defined INSTALLED_PATH (
    echo [✓] Found installed version at: !INSTALLED_PATH!
    echo.
    echo [*] Launching NEURECTOMY...
    start "" "!INSTALLED_PATH!\NEURECTOMY.exe"
    timeout /t 2 /nobreak
    exit /b 0
)

REM If not installed, check for installer
echo [!] NEURECTOMY not installed. Installing now...
echo.

if exist "!MSI_PATH!" (
    echo [*] Found installer: !MSI_PATH!
    echo [*] Launching installer...
    msiexec /i "!MSI_PATH!" /quiet /norestart
    echo.
    echo [✓] Installation started. Please wait...
    timeout /t 10 /nobreak
    
    REM Launch the app after installation
    echo [*] Launching NEURECTOMY...
    start "" "C:\Program Files\NEURECTOMY\NEURECTOMY.exe"
) else if exist "!EXE_PATH!" (
    echo [*] Found installer: !EXE_PATH!
    echo [*] Launching installer...
    start "" "!EXE_PATH!"
    echo.
    echo [✓] Installation started. Please follow the installer prompts...
) else (
    echo.
    echo [✗] ERROR: Could not find NEURECTOMY installer!
    echo.
    echo     Expected location:
    echo     !MSI_PATH!
    echo.
    echo     or
    echo.
    echo     !EXE_PATH!
    echo.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                 Launcher Complete                              ║
echo ║   NEURECTOMY is starting... (allow 10-30 seconds)             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

endlocal
