#!/usr/bin/env pwsh
<#
.SYNOPSIS
Build NEURECTOMY Desktop Application with updated port configuration (16xxx scheme)
.DESCRIPTION
This script:
1. Validates the environment
2. Installs/updates dependencies
3. Builds the Tauri desktop app
4. Creates a build summary
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NEURECTOMY Desktop Application Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Color functions
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Blue
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Step 1: Check pnpm
Write-Info "Checking pnpm installation..."
$pnpmVersion = pnpm --version
if ($pnpmVersion) {
    Write-Success "pnpm $pnpmVersion"
}
else {
    Write-Error "pnpm not found. Please install pnpm first."
    exit 1
}

# Step 2: Check Node.js
Write-Info "Checking Node.js installation..."
$nodeVersion = node --version
if ($nodeVersion) {
    Write-Success "Node.js $nodeVersion"
}
else {
    Write-Error "Node.js not found. Please install Node.js first."
    exit 1
}

# Step 3: Check Rust/Cargo
Write-Info "Checking Rust/Cargo installation..."
$cargoVersion = cargo --version
if ($cargoVersion) {
    Write-Success "$cargoVersion"
}
else {
    Write-Warning "Rust/Cargo not found. Tauri build will require Rust."
}

# Step 4: Navigate to spectrum-workspace
Write-Info "Navigating to spectrum-workspace..."
$appDir = ".\apps\spectrum-workspace"
if (Test-Path $appDir) {
    Write-Success "Found spectrum-workspace"
}
else {
    Write-Error "spectrum-workspace directory not found"
    exit 1
}

# Step 5: Install dependencies
Write-Info "Installing dependencies..."
Write-Info "This may take a few minutes..."
pnpm install
if ($LASTEXITCODE -eq 0) {
    Write-Success "Dependencies installed"
}
else {
    Write-Error "Failed to install dependencies"
    exit 1
}

# Step 6: Build the Tauri desktop app
Write-Info "Building Tauri desktop application..."
Write-Info "Updated configuration:"
Write-Info "  - Spectrum Workspace (devUrl): http://localhost:16000"
Write-Info "  - API Gateway: http://localhost:16080"
Write-Info "  - ML Service: http://localhost:16081"
Write-Info "  - WebSocket: ws://localhost:16083"
Write-Info ""

Push-Location $appDir
pnpm tauri build
$buildSuccess = $LASTEXITCODE -eq 0
Pop-Location

if ($buildSuccess) {
    Write-Success "Desktop application build completed successfully!"
    Write-Info ""
    Write-Info "Build artifacts:"
    Write-Info "  - Windows (MSI): apps/spectrum-workspace/src-tauri/target/release/bundle/msi/"
    Write-Info "  - macOS (DMG): apps/spectrum-workspace/src-tauri/target/release/bundle/dmg/"
    Write-Info "  - Linux (AppImage): apps/spectrum-workspace/src-tauri/target/release/bundle/appimage/"
}
else {
    Write-Error "Desktop application build failed"
    exit 1
}

# Step 7: Create build summary
Write-Info "Creating build summary..."
$summaryPath = "DESKTOP_BUILD_SUMMARY.md"
@"
# Desktop Application Build Summary

**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Status:** ✅ SUCCESS

## Build Configuration

### Port Scheme (Updated to 16xxx)
- **Spectrum Workspace (Frontend):** http://localhost:16000
- **API Gateway:** http://localhost:16080  
- **ML Service:** http://localhost:16081
- **WebSocket Server:** ws://localhost:16083
- **GraphQL Endpoint:** http://localhost:16080/graphql

## Files Updated

### Frontend Configuration Files
- `apps/spectrum-workspace/src/lib/api.ts` - API_BASE_URL: 16080, ML_API_URL: 16081
- `apps/spectrum-workspace/src/lib/graphql.ts` - GRAPHQL_URL: 16080/graphql
- `apps/spectrum-workspace/src/hooks/useWebSocket.ts` - WebSocket: 16083
- `apps/spectrum-workspace/src/services/__tests__/ryot-service.test.ts` - Ryot endpoint: 46080

### Tauri Configuration
- `apps/spectrum-workspace/src-tauri/tauri.conf.json` - devUrl: 16000

## Build Artifacts

### Windows
- **Location:** `apps/spectrum-workspace/src-tauri/target/release/bundle/msi/`
- **Format:** MSI Installer

### macOS
- **Location:** `apps/spectrum-workspace/src-tauri/target/release/bundle/dmg/`
- **Format:** DMG Installer

### Linux
- **Location:** `apps/spectrum-workspace/src-tauri/target/release/bundle/appimage/`
- **Format:** AppImage

## Development Setup

To run the desktop app in development mode with hot reload:

\`\`\`bash
cd apps/spectrum-workspace
pnpm tauri dev
\`\`\`

This will:
1. Start the Vite dev server on port 16000
2. Launch the Tauri desktop window
3. Connect to backend services on the new 16xxx ports

## Testing Instructions

1. **Launch the app:**
   - Development: \`pnpm tauri dev\` from apps/spectrum-workspace
   - Production: Run the built installer for your platform

2. **Verify connectivity:**
   - Open DevTools (Ctrl+Shift+I)
   - Check Network tab for requests to:
     - \`http://localhost:16080\` (API calls)
     - \`http://localhost:16081\` (ML service)
     - \`ws://localhost:16083\` (WebSocket)

3. **Check logs:**
   - Frontend console logs in DevTools
   - Backend service logs in terminal running the dev server

## Troubleshooting

### If desktop app can't connect to backend:
1. Verify backend services are running on correct ports
2. Check Windows Firewall isn't blocking connections
3. Review CSP (Content Security Policy) in tauri.conf.json

### If build fails:
1. Ensure Rust and Cargo are installed (required for Tauri)
2. Try: \`cargo update\` in \`apps/spectrum-workspace/src-tauri/\`
3. Clear build cache: \`cargo clean\`

## Next Steps

1. ✅ Desktop app built and ready
2. Start backend services on new 16xxx ports
3. Test desktop app connectivity
4. Deploy or distribute built installers

---

**Build System:** Tauri 2.0 with Vite + React + Rust Backend
**Last Updated:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@ | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Success "Build summary created: $summaryPath"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Start backend services on 16xxx ports" -ForegroundColor Yellow
Write-Host "2. Test desktop app: pnpm tauri dev" -ForegroundColor Yellow
Write-Host "3. Verify API connectivity in DevTools" -ForegroundColor Yellow
Write-Host ""
