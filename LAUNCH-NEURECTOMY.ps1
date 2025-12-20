#!/usr/bin/env pwsh
<#
.SYNOPSIS
    NEURECTOMY Desktop Application - One-Click Launcher (PowerShell)
.DESCRIPTION
    Launches the pre-built NEURECTOMY desktop application.
    No build required - fully packaged and ready to run!
.EXAMPLE
    .\LAUNCH-NEURECTOMY.ps1
#>

param()

$ErrorActionPreference = 'Continue'

# Display banner
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "║         NEURECTOMY - Dimensional Forge v0.1.0                 ║" -ForegroundColor Cyan
Write-Host "║              Desktop Application Launcher                      ║" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Paths to installers
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$msiPath = Join-Path $scriptPath "apps\spectrum-workspace\src-tauri\target\release\bundle\msi\NEURECTOMY_0.1.0_x64_en-US.msi"
$exePath = Join-Path $scriptPath "apps\spectrum-workspace\src-tauri\target\release\bundle\nsis\NEURECTOMY_0.1.0_x64-setup.exe"

Write-Host "[*] Checking for NEURECTOMY installation..." -ForegroundColor Yellow

# Check if already installed
$installed = Get-ItemProperty -Path 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | 
Where-Object { $_.DisplayName -like '*NEURECTOMY*' } | 
Select-Object -First 1

if ($installed) {
    Write-Host "[✓] Found installed version" -ForegroundColor Green
    $installPath = $installed.InstallLocation
    $appPath = Join-Path $installPath "NEURECTOMY.exe"
    
    if (Test-Path $appPath) {
        Write-Host "[*] Launching NEURECTOMY..." -ForegroundColor Yellow
        Write-Host ""
        & $appPath
        exit 0
    }
}

# Check for installer
Write-Host "[!] NEURECTOMY not installed. Installing now..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path $msiPath) {
    Write-Host "[*] Found MSI installer" -ForegroundColor Yellow
    Write-Host "[*] Launching installer..." -ForegroundColor Yellow
    
    $msiArgs = @("/i", $msiPath, "/quiet", "/norestart")
    & msiexec @msiArgs
    
    Write-Host ""
    Write-Host "[✓] Installation started. Please wait..." -ForegroundColor Green
    Start-Sleep -Seconds 10
    
    # Try to launch installed app
    $appPath = "C:\Program Files\NEURECTOMY\NEURECTOMY.exe"
    if (Test-Path $appPath) {
        Write-Host "[*] Launching NEURECTOMY..." -ForegroundColor Yellow
        & $appPath
    }
}
elseif (Test-Path $exePath) {
    Write-Host "[*] Found NSIS installer" -ForegroundColor Yellow
    Write-Host "[*] Launching installer..." -ForegroundColor Yellow
    Write-Host ""
    
    & $exePath
    Write-Host "[✓] Installation started. Please follow the installer prompts..." -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[✗] ERROR: Could not find NEURECTOMY installer!" -ForegroundColor Red
    Write-Host ""
    Write-Host "     Expected locations:" -ForegroundColor Yellow
    Write-Host "     - $msiPath" -ForegroundColor Gray
    Write-Host "     - $exePath" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                 Launcher Complete                              ║" -ForegroundColor Cyan
Write-Host "║   NEURECTOMY is starting... (allow 10-30 seconds)             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
