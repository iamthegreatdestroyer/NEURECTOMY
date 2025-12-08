# NEURECTOMY Desktop Application Guide

**NEURECTOMY Spectrum Workspace** is a fully **bundled native desktop application** built with **Tauri 2.0** - providing a VS Code-like IDE experience with ALL backend services embedded in a single executable.

---

## 🎯 **One-Click Desktop App**

NEURECTOMY is a **complete all-in-one desktop application** that bundles:

- ✅ **Frontend UI** (React + Vite)
- ✅ **API Gateway** (Embedded Rust HTTP server on port 16080)
- ✅ **ML Service** (Embedded or spawned Python service on port 16081)
- ✅ **Database** (Embedded SQLite in app data folder)
- ✅ **Native System Integration** (File system, clipboard, notifications)

**Everything starts with one command** - no separate backend setup required!

---

## 🖥️ **Desktop vs Web Mode**

### Desktop Mode (Recommended - Now with Embedded Backend!)

- ✅ **Native window** with OS integration
- ✅ **Embedded API server** - starts automatically on port 16080
- ✅ **Embedded ML service** - starts automatically on port 16081
- ✅ **Embedded database** - SQLite in app data folder
- ✅ **File system access** for reading/writing agent files
- ✅ **System clipboard** integration
- ✅ **Native notifications**
- ✅ **Faster performance** (Rust backend)
- ✅ **Auto-updates** capability
- ✅ **Native menus and shortcuts**
- ✅ **Single executable** - no external services needed!

### Web Mode (Development Only - Requires External Backend)

- ⚠️ Limited to browser sandbox
- ⚠️ No direct file system access
- ⚠️ Requires separate API/ML services running
- ⚠️ Slower performance
- ✅ Quick testing without building

---

## 🚀 **Running the Desktop Application (One Command!)**

### Option 1: Desktop Development Mode (Recommended)

```powershell
# From project root
cd C:\Users\sgbil\NEURECTOMY

# Run as native desktop app with embedded backend
cd apps/spectrum-workspace
pnpm desktop

# OR use the full command
pnpm tauri:dev
```

**What this does (all automatically!):**

1. ✅ Starts Vite dev server on port 16000
2. ✅ Initializes embedded SQLite database
3. ✅ Starts embedded API Gateway on port 16080
4. ✅ Spawns ML Service on port 16081 (Python subprocess)
5. ✅ Launches native desktop window
6. ✅ Enables hot reload for development
7. ✅ Opens DevTools for debugging

**No separate backend commands needed!** Everything launches together.

---

### Option 2: Build Native Installer

```powershell
cd C:\Users\sgbil\NEURECTOMY\apps\spectrum-workspace

# Build production desktop app
pnpm tauri:build
```

**Output:**

- 📦 **Windows:** `src-tauri/target/release/NEURECTOMY.exe`
- 📦 **Installer:** `src-tauri/target/release/bundle/msi/NEURECTOMY_0.1.0_x64_en-US.msi`

---

## 📋 **Port Configuration (All Embedded!)**

All NEURECTOMY services use the **16000 series** and **start automatically** when you launch the desktop app:

| Service                      | Port  | URL                    | Status         |
| ---------------------------- | ----- | ---------------------- | -------------- |
| **Desktop App (Vite)**       | 16000 | http://localhost:16000 | ✅ Auto-starts |
| **API Gateway (Rust)**       | 16080 | http://localhost:16080 | ✅ Embedded    |
| **ML Service (Python/Rust)** | 16081 | http://localhost:16081 | ✅ Embedded    |
| **WebSocket**                | 16080 | ws://localhost:16080   | ✅ Embedded    |
| **Database (SQLite)**        | N/A   | App Data Folder        | ✅ Embedded    |

**Embedded Architecture:**

- `devUrl`: http://localhost:16000 ✅
- Frontend loads from Vite dev server
- API Gateway starts automatically in Rust backend
- ML Service spawns as subprocess (Python) or runs in Rust
- Database initialized automatically in app data folder
- **No Docker, no separate terminals, no manual service management!**

---

## 🏗️ **Embedded Backend Architecture**

### How It Works

When you run `pnpm desktop`, the Tauri application:

1. **Initializes Tauri Runtime**
   - Loads Rust backend
   - Sets up window management
   - Initializes plugin system

2. **Starts Embedded Services (Automatic!)**

   ```rust
   // Happens in src-tauri/src/lib.rs setup()
   let service_manager = ServiceManager::new();
   service_manager.start_all().await?;
   ```

   - **Database**: Creates SQLite DB in `%APPDATA%/neurectomy/neurectomy.db`
   - **API Gateway**: Starts Axum HTTP server on port 16080
   - **ML Service**: Spawns Python subprocess on port 16081

3. **Launches Frontend**
   - Vite dev server on port 16000
   - Opens desktop window
   - Frontend connects to embedded API on 16080

### Service Lifecycle

```
Desktop App Launches
        ↓
[Rust Backend Initializes]
        ↓
[Database] → SQLite created in app data folder
        ↓
[API Gateway] → Axum server starts on :16080
        ↓
[ML Service] → Python subprocess spawns on :16081
        ↓
[Frontend] → Vite connects to embedded backend
        ↓
✅ Ready to use!

Desktop App Closes
        ↓
[Services Shutdown] → All embedded services stop gracefully
        ↓
[Database] → SQLite connections closed
        ↓
✅ Clean exit
```

### File Structure

```
apps/spectrum-workspace/src-tauri/
├── Cargo.toml              # Dependencies (axum, sqlx, tokio)
├── src/
│   ├── lib.rs              # Main entry - starts services
│   ├── main.rs             # Executable entry point
│   ├── commands.rs         # Tauri commands (file operations)
│   ├── gpu.rs              # GPU detection & WebGPU support
│   ├── state.rs            # Application state management
│   └── services/           # 🆕 EMBEDDED BACKEND
│       ├── mod.rs          # ServiceManager orchestrator
│       ├── api_gateway.rs  # REST API server (Axum on :16080)
│       ├── database.rs     # SQLite initialization & migrations
│       └── ml_service.rs   # ML service integration
├── capabilities/           # Tauri permissions
│   └── default.json
└── tauri.conf.json         # Tauri configuration
```

### API Gateway Endpoints

The embedded API Gateway (`src-tauri/src/services/api_gateway.rs`) provides:

| Endpoint            | Method | Description                    |
| ------------------- | ------ | ------------------------------ |
| `/health`           | GET    | Health check                   |
| `/api/health`       | GET    | API health check               |
| `/api/projects`     | GET    | List all projects              |
| `/api/projects`     | POST   | Create new project             |
| `/api/projects/:id` | GET    | Get project details            |
| `/api/agents`       | GET    | List all agents                |
| `/api/agents/:id`   | GET    | Get agent details              |
| `/graphql`          | POST   | GraphQL endpoint (coming soon) |

**All endpoints automatically available when desktop app launches!**

### Database Schema

SQLite database automatically created with:

```sql
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT
);

CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE agent_runs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL,
    input TEXT,
    output TEXT,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata TEXT,
    created_at INTEGER NOT NULL
);
```

**Database location**: `C:\Users\<username>\AppData\Roaming\neurectomy\neurectomy.db`

---

## 🎯 **Desktop Features**

### Window Configuration

```json
{
  "title": "NEURECTOMY - Dimensional Forge",
  "width": 1600,
  "height": 900,
  "minWidth": 1280,
  "minHeight": 720,
  "theme": "Dark"
}
```

### File System Access

```typescript
import { readTextFile, writeTextFile } from "@tauri-apps/plugin-fs";

// Read agent.py from disk
const content = await readTextFile("agents/agent.py");

// Save edited agent
await writeTextFile("agents/agent.py", newContent);
```

### Native Dialogs

```typescript
import { open, save } from "@tauri-apps/plugin-dialog";

// Open file picker
const selected = await open({
  multiple: false,
  filters: [
    {
      name: "Python",
      extensions: ["py"],
    },
  ],
});
```

### Clipboard Integration

```typescript
import { readText, writeText } from "@tauri-apps/plugin-clipboard-manager";

// Copy to system clipboard
await writeText("code snippet");

// Paste from clipboard
const clipboardContent = await readText();
```

### System Notifications

```typescript
import { sendNotification } from "@tauri-apps/plugin-notification";

sendNotification({
  title: "Agent Ready",
  body: "Your agent has been deployed successfully!",
});
```

---

## 🔧 **Troubleshooting**

### Issue: Port 16000 Already in Use

**Solution 1: Kill existing processes**

```powershell
Get-Process | Where-Object {$_.ProcessName -like "*node*"} | Stop-Process -Force
```

**Solution 2: Change port temporarily**

```typescript
// vite.config.ts
server: {
  port: 16001  // or any available port
}

// Update tauri.conf.json accordingly
"devUrl": "http://localhost:16001"
```

### Issue: Tauri Window Doesn't Open

**Check:**

1. ✅ Vite dev server is running (check terminal)
2. ✅ Port matches in both configs
3. ✅ No firewall blocking localhost connections

**Debug:**

```powershell
# Check if Vite is accessible
Invoke-WebRequest http://localhost:16000
```

### Issue: Hot Reload Not Working

**Solution:** Ensure `beforeDevCommand` starts Vite server:

```json
"beforeDevCommand": "pnpm dev"
```

### Issue: File System Permissions Denied

**Check:** `tauri.conf.json` has correct scopes:

```json
"fs": {
  "scope": [
    "$APPDATA", "$APPDATA/**",
    "$DOCUMENT", "$DOCUMENT/**",
    "$DOWNLOAD", "$DOWNLOAD/**"
  ]
}
```

---

## 🏗️ **Project Structure**

```
apps/spectrum-workspace/
├── src/                      # React application
│   ├── components/
│   │   └── editors/
│   │       ├── MonacoEditor.tsx      # Monaco editor component
│   │       └── EditorManager.tsx     # Multi-file management
│   ├── stores/
│   │   └── editor-store.ts           # Editor state (Zustand)
│   └── features/
│       └── agent-editor/
│           └── AgentEditor.tsx       # Main IDE interface
├── src-tauri/                # Tauri native backend
│   ├── src/
│   │   └── main.rs           # Rust entry point
│   ├── tauri.conf.json       # Tauri configuration
│   ├── Cargo.toml            # Rust dependencies
│   └── icons/                # Desktop app icons
└── vite.config.ts            # Vite bundler config
```

---

## 🎨 **Desktop vs Browser Experience**

### Monaco Editor in Desktop Mode

- ✅ **Full keyboard shortcuts** (no browser conflicts)
- ✅ **Native file dialogs** for open/save
- ✅ **Direct file system access** (no need for upload/download)
- ✅ **System clipboard** integration
- ✅ **Native context menus**

### Desktop-Only Features (Planned)

- 🔄 **System tray integration**
- 🔄 **Global hotkeys**
- 🔄 **Deep OS integration** (Windows Terminal, SSH)
- 🔄 **Local AI model** execution
- 🔄 **Git integration** with system Git

---

## 📦 **Distribution**

### Building for Production

```powershell
# Build optimized desktop app
cd apps/spectrum-workspace
pnpm tauri:build

# Outputs:
# - NEURECTOMY.exe (portable executable)
# - NEURECTOMY_0.1.0_x64_en-US.msi (installer)
```

### Installer Features

- ✅ Start menu shortcuts
- ✅ Desktop icon
- ✅ File associations (.nrct files)
- ✅ Automatic updates (when configured)
- ✅ Uninstaller

---

## 🔐 **Security**

Tauri is more secure than Electron:

- ✅ **Smaller attack surface** (Rust backend)
- ✅ **Content Security Policy** (CSP) enforced
- ✅ **Capability-based permissions**
- ✅ **No Node.js runtime** in frontend
- ✅ **IPC validation** between frontend/backend

**CSP Configuration:**

```json
"csp": "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; connect-src 'self' ws://localhost:* http://localhost:* https://api.openai.com https://api.anthropic.com;"
```

---

## 🚀 **Next Steps**

1. **Run Desktop Mode:**

   ```powershell
   cd C:\Users\sgbil\NEURECTOMY
   pnpm --filter @neurectomy/spectrum-workspace desktop
   ```

2. **Test Monaco Editor:**
   - Native window should open automatically
   - Navigate to Agent Editor
   - Test file operations with direct file system access

3. **Configure Auto-Updates:**
   - Set up update server
   - Configure `updater` plugin in Tauri

4. **Build Production Installer:**
   - Test on clean Windows machine
   - Sign executable for Windows SmartScreen

---

## 📚 **Resources**

- **Tauri Docs:** https://v2.tauri.app/
- **Tauri API:** https://v2.tauri.app/reference/javascript/api/
- **Monaco Editor:** https://microsoft.github.io/monaco-editor/
- **Vite:** https://vitejs.dev/

---

**NEURECTOMY is a DESKTOP APPLICATION, not a web app!** 🚀

Run it with `pnpm desktop` and experience the full native IDE experience.
