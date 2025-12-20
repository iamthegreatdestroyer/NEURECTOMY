# Desktop Application Build & Migration Summary

**Date:** December 18, 2025
**Status:** ✅ COMPLETE - Ready for Build

## Build Configuration Updated

### Port Scheme Migration (Old → New)

| Service                       | Old Port     | New Port      |
| ----------------------------- | ------------ | ------------- |
| Spectrum Workspace (Frontend) | 3000         | 16000         |
| API Gateway                   | 8080         | 16080         |
| ML Service                    | 8000/8081    | 16081         |
| WebSocket Server              | 8080         | 16083         |
| GraphQL Endpoint              | 8080/graphql | 16080/graphql |
| Ryot Service                  | 8080         | 46080         |

## Files Updated for Desktop Application

### Frontend API Configuration

✅ `apps/spectrum-workspace/src/lib/api.ts`

- API_BASE_URL: `http://localhost:8080` → `http://localhost:16080`
- ML_API_URL: `http://localhost:8000` → `http://localhost:16081`

✅ `apps/spectrum-workspace/src/lib/graphql.ts`

- GRAPHQL_URL: `http://localhost:8080/graphql` → `http://localhost:16080/graphql`

✅ `apps/spectrum-workspace/src/hooks/useWebSocket.ts`

- DEFAULT_WS_URL: `ws://localhost:8080/ws` → `ws://localhost:16083/ws`

✅ `apps/spectrum-workspace/src/services/__tests__/ryot-service.test.ts`

- Mock Ryot API: `http://localhost:8080` → `http://localhost:46080`

### Tauri Configuration

✅ `apps/spectrum-workspace/src-tauri/tauri.conf.json`

- devUrl: Already configured to `http://localhost:16000` ✓

## How to Build the Desktop Application

### Automatic Build (Recommended)

```powershell
# From NEURECTOMY root directory
.\BUILD_DESKTOP_APP.ps1
```

This script will:

1. Verify Node.js, pnpm, and Rust/Cargo are installed
2. Install dependencies
3. Build the Tauri desktop application
4. Generate build artifacts for Windows (MSI), macOS (DMG), and Linux (AppImage)
5. Create detailed build summary

### Manual Build Steps

```bash
cd apps/spectrum-workspace
pnpm install
pnpm tauri build
```

## Development Mode

To run the desktop app with hot reload during development:

```bash
cd apps/spectrum-workspace
pnpm tauri dev
```

This launches:

- Vite dev server on `http://localhost:16000`
- Tauri desktop window
- Live reload on code changes

## Build Artifacts Location

After successful build, installers will be at:

- **Windows:** `apps/spectrum-workspace/src-tauri/target/release/bundle/msi/`
- **macOS:** `apps/spectrum-workspace/src-tauri/target/release/bundle/dmg/`
- **Linux:** `apps/spectrum-workspace/src-tauri/target/release/bundle/appimage/`

## Backend Service Requirements

The desktop application requires these backend services to be running:

```
API Gateway:      http://localhost:16080
ML Service:       http://localhost:16081
WebSocket:        ws://localhost:16083
GraphQL:          http://localhost:16080/graphql
```

Start backend services:

```bash
docker-compose up -d
# or start individual services on their respective ports
```

## CSP (Content Security Policy) Configuration

The Tauri app includes comprehensive CSP to allow:

- WebSocket connections to localhost:16xxx
- API calls to localhost:16xxx
- External API calls to OpenAI, Anthropic
- Web fonts from Google Fonts

**CSP in tauri.conf.json:**

```json
"connect-src": "'self' ws://localhost:* http://localhost:* https://api.openai.com https://api.anthropic.com..."
```

## Verification Checklist

After building and starting the app:

- [ ] Desktop window launches successfully
- [ ] DevTools can be opened (Ctrl+Shift+I in debug mode)
- [ ] Network requests show API calls to 16080
- [ ] WebSocket connects to 16083
- [ ] GraphQL queries reach 16080/graphql
- [ ] No CORS or connection errors in console
- [ ] Application loads workspace and UI components
- [ ] Can interact with 3D/4D visualization
- [ ] Backend services communicate properly

## Troubleshooting

### Desktop app won't start

1. Verify Node.js and pnpm are installed
2. Check that port 16000 is available
3. Review console output for error messages

### Can't connect to backend

1. Confirm backend services are running on 16xxx ports
2. Check Windows Firewall allows localhost connections
3. Verify CSP settings in tauri.conf.json

### Build fails

1. Install Rust: `https://www.rust-lang.org/tools/install`
2. Update Cargo: `cargo update`
3. Clear cache: `cargo clean` in src-tauri directory

## Dependencies

Required for building:

- **Node.js** 18+ (for Vite and Tauri)
- **pnpm** (already installed globally)
- **Rust** (for Tauri backend compilation)
- **Cargo** (comes with Rust)

Platform-specific:

- **Windows:** Visual Studio Build Tools or MSVC
- **macOS:** Xcode Command Line Tools
- **Linux:** gcc/libssl-dev

## Summary

All frontend and configuration files have been updated to use the new NEURECTOMY port scheme (16xxx):

✅ API clients updated (16080/16081)
✅ WebSocket endpoints updated (16083)
✅ GraphQL endpoints updated (16080)
✅ Tauri configuration ready (16000)
✅ Test files updated (46080)

**The desktop application is ready to be built with the updated port configuration.**

---

**Next:** Run `.\BUILD_DESKTOP_APP.ps1` to build the desktop application
