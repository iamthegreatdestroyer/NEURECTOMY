# 🚀 NEURECTOMY IDE - Quick Start Guide

**Welcome to NEURECTOMY** - The Ultimate Agent Development & Orchestration Platform

This guide will help you start building the IDE and understand the architecture.

---

## 📋 Prerequisites

- **Node.js:** >= 20.0.0
- **pnpm:** 8.15.0 (Package Manager)
- **Rust:** Latest stable (for Tauri)
- **Python:** 3.11+ (for ML services)
- **Docker:** Latest (for services)
- **Git:** Latest

---

## 🏗️ Project Setup

### 1. Clone and Install Dependencies

```bash
# Clone repository
git clone https://github.com/your-org/NEURECTOMY.git
cd NEURECTOMY

# Install all dependencies (monorepo)
pnpm install

# Build all packages
pnpm build
```

### 2. Start Development Environment

```bash
# Start all services (Docker)
pnpm docker:up

# In separate terminal: Start Tauri desktop app
pnpm desktop

# Or start web version only
pnpm dev
```

### 3. Verify Setup

Open browser to `http://localhost:5173` (web) or Tauri window should open automatically.

You should see the NEURECTOMY Spectrum Workspace with:

- Left sidebar navigation
- Top menu bar
- Center workspace area
- Right properties panel (collapsed)
- Bottom terminal panel (collapsed)

---

## 📁 Project Structure

```
NEURECTOMY/
├── apps/
│   └── spectrum-workspace/      # Main Tauri desktop application
│       ├── src/
│       │   ├── components/      # Reusable UI components
│       │   ├── features/        # Feature modules (Forge, Command, etc.)
│       │   ├── stores/          # Zustand state management ✅ NEW
│       │   ├── layouts/         # Layout components
│       │   ├── hooks/           # Custom React hooks
│       │   └── lib/             # Utilities
│       └── src-tauri/           # Tauri Rust backend
│
├── packages/
│   ├── 3d-engine/               # Three.js/WebGPU visualization ✅ ENHANCED
│   ├── api-client/              # GraphQL/REST API client ✅ NEW
│   ├── container-command/       # Docker/K8s orchestration
│   ├── core/                    # Core utilities
│   ├── types/                   # Shared TypeScript types
│   └── ui/                      # Shared UI components
│
├── services/
│   ├── rust-core/               # High-performance Rust backend
│   └── ml-service/              # Python ML microservice
│
├── docker/                      # Docker configurations
├── k8s/                         # Kubernetes manifests
└── docs/                        # Documentation
```

---

## 🎨 Key Concepts

### 1. **State Management (Zustand)**

All global state is managed through Zustand stores:

```typescript
// Import stores
import { useWorkspaceStore, useAgentStore, useContainerStore } from '@/stores';

// In your component
function MyComponent() {
  // Access workspace state
  const { layout, addTab, toggleSidebar } = useWorkspaceStore();

  // Access agent state
  const { workflows, activeWorkflowId, createWorkflow } = useAgentStore();

  // Access container state
  const { clusters, activeClusterId, selectPod } = useContainerStore();

  // Use state and actions
  return (
    <div>
      <button onClick={() => toggleSidebar('left')}>Toggle Sidebar</button>
      <button onClick={() => createWorkflow({
        id: 'workflow-1',
        name: 'My Workflow',
        nodes: [],
        connections: [],
        status: 'idle'
      })}>
        Create Workflow
      </button>
    </div>
  );
}
```

**Available Stores:**

- `useWorkspaceStore` - Layout, tabs, panels, theme
- `useAgentStore` - Agent workflows, nodes, connections
- `useContainerStore` - Docker containers, K8s clusters

### 2. **3D Visualization**

Use the 3D engine components for CAD-like visualizations:

```typescript
import { AgentNodeMesh, ConnectionLine } from '@/components/3d';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

function MyVisualization() {
  const { workflows, activeWorkflowId } = useAgentStore();
  const workflow = workflows.find(w => w.id === activeWorkflowId);

  return (
    <Canvas>
      <OrbitControls />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} />

      {workflow?.nodes.map(node => (
        <AgentNodeMesh
          key={node.id}
          node={node}
          isSelected={/* selection logic */}
          onClick={(node) => console.log('Clicked:', node.name)}
        />
      ))}

      {workflow?.connections.map(conn => (
        <ConnectionLine
          key={conn.id}
          connection={conn}
          startPosition={/* calculate from node */}
          endPosition={/* calculate from node */}
        />
      ))}
    </Canvas>
  );
}
```

### 3. **API Communication**

Use the GraphQL client for backend communication:

```typescript
import { query, mutation, subscribe } from "@neurectomy/api-client";

// Query example
async function fetchAgents() {
  const { data, error } = await query(`
    query GetAgents {
      agents {
        id
        name
        status
      }
    }
  `);

  if (error) {
    console.error("Failed to fetch agents:", error);
    return;
  }

  return data.agents;
}

// Mutation example
async function createAgent(name: string) {
  const { data, error } = await mutation(
    `
    mutation CreateAgent($name: String!) {
      createAgent(name: $name) {
        id
        name
      }
    }
  `,
    { name }
  );

  return data?.createAgent;
}

// Subscription example (real-time updates)
function subscribeToAgentUpdates() {
  const unsubscribe = subscribe(
    `
      subscription AgentUpdates {
        agentUpdated {
          id
          status
          metrics {
            cpu
            memory
          }
        }
      }
    `,
    undefined,
    (data) => {
      console.log("Agent updated:", data.agentUpdated);
      // Update store with new data
    },
    (error) => {
      console.error("Subscription error:", error);
    }
  );

  // Cleanup
  return unsubscribe;
}
```

---

## 🧩 Creating New Features

### Step 1: Create Feature Module

```bash
# Create feature directory
mkdir apps/spectrum-workspace/src/features/my-feature

# Create main component
touch apps/spectrum-workspace/src/features/my-feature/MyFeature.tsx
```

### Step 2: Implement Component

```typescript
// apps/spectrum-workspace/src/features/my-feature/MyFeature.tsx
export default function MyFeature() {
  return (
    <div className="h-full flex flex-col">
      <div className="h-12 px-4 flex items-center border-b border-border">
        <h1>My Feature</h1>
      </div>
      <div className="flex-1 p-4">
        {/* Feature content */}
      </div>
    </div>
  );
}
```

### Step 3: Add Route

```typescript
// apps/spectrum-workspace/src/App.tsx
import { lazy } from 'react';

const MyFeature = lazy(() => import('./features/my-feature/MyFeature'));

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        {/* ...existing routes */}
        <Route
          path="my-feature"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <MyFeature />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}
```

### Step 4: Add Navigation

```typescript
// apps/spectrum-workspace/src/components/sidebar/Sidebar.tsx
const menuItems = [
  // ...existing items
  {
    id: 'my-feature',
    label: 'My Feature',
    icon: <MyIcon />,
    path: '/my-feature',
  },
];
```

---

## 🎨 Styling Guidelines

NEURECTOMY uses **Tailwind CSS** with a custom dark theme:

```typescript
// Use Tailwind utility classes
<div className="flex items-center gap-2 p-4 bg-card rounded-lg border border-border">
  <span className="text-sm text-muted-foreground">Status:</span>
  <span className="text-primary font-medium">Active</span>
</div>

// Custom colors are defined in tailwind.config.ts
// Use semantic color names:
// - bg-primary, bg-secondary, bg-tertiary (backgrounds)
// - text-primary, text-secondary (text)
// - border (borders)
// - accent-primary, accent-secondary (highlights)
// - success, warning, error (status colors)
```

---

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run tests with coverage
pnpm test:coverage

# Run tests in watch mode
pnpm test --watch

# Run tests for specific package
pnpm --filter @neurectomy/3d-engine test
```

### Example Test

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MyComponent } from './MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('handles clicks', async () => {
    const { user } = render(<MyComponent />);
    const button = screen.getByRole('button');

    await user.click(button);

    expect(screen.getByText('Clicked')).toBeInTheDocument();
  });
});
```

---

## 🚀 Building for Production

```bash
# Build all packages
pnpm build

# Build Tauri desktop application
pnpm desktop:build

# Output: apps/spectrum-workspace/src-tauri/target/release/
```

---

## 🐛 Common Issues

### Issue: "Cannot find module @neurectomy/..."

**Solution:** Build packages first

```bash
pnpm build
```

### Issue: Tauri fails to start

**Solution:** Ensure Rust is installed

```bash
rustup update
cargo --version
```

### Issue: WebGPU not working

**Solution:** Use a Chromium-based browser or enable WebGPU flags:

- Chrome: `chrome://flags/#enable-unsafe-webgpu`
- Edge: `edge://flags/#enable-unsafe-webgpu`

### Issue: Docker services not starting

**Solution:** Check Docker daemon

```bash
docker ps
pnpm docker:up
```

---

## 📚 Additional Resources

- **Architecture Docs:** `docs/architecture/`
- **API Reference:** `docs/api/`
- **Component Library:** `packages/ui/src/`
- **Tutorials:** `docs/tutorials/`

---

## 🎯 Next Steps

1. ✅ Explore the codebase
2. ✅ Run the development environment
3. ✅ Review the state management stores
4. ✅ Try creating a simple agent workflow
5. ✅ Visualize it in the Dimensional Forge
6. ✅ Read the Implementation Progress doc

---

**Need Help?** Check `docs/` or raise an issue on GitHub.

**Happy Building!** 🚀
