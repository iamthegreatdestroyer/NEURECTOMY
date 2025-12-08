# Start Tauri dev server with proper error handling
Set-Location $PSScriptRoot
Write-Host "🚀 Starting NEURECTOMY Desktop..." -ForegroundColor Cyan
Write-Host "📍 Working directory: $PWD" -ForegroundColor Gray
Write-Host ""

try {
    & pnpm run tauri:dev
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
