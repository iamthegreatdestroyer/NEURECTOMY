# NEURECTOMY IDE Components

This document describes the custom IDE components integrated into the Spectrum Workspace IDE view.

## Table of Contents

- [NEURECTOMY IDE Components](#neurectomy-ide-components)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Components](#components)
    - [Agent Execution Graph](#agent-execution-graph)
      - [Props](#props)
      - [Types](#types)
      - [Usage](#usage)
    - [GitOps Overlay](#gitops-overlay)
      - [Props](#props-1)
      - [Types](#types-1)
      - [Usage](#usage-1)
    - [Experiment Sidebar](#experiment-sidebar)
      - [Props](#props-2)
      - [Types](#types-2)
      - [Usage](#usage-2)
    - [Enhanced Status Bar](#enhanced-status-bar)
      - [Props](#props-3)
      - [Usage](#usage-3)
  - [Motion Utilities](#motion-utilities)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
    - [Available Shortcuts](#available-shortcuts)
    - [Usage](#usage-4)
    - [Getting Shortcut Info](#getting-shortcut-info)
  - [Integration Guide](#integration-guide)
    - [1. Import Components](#1-import-components)
    - [2. Add State Management](#2-add-state-management)
    - [3. Add Keyboard Shortcuts](#3-add-keyboard-shortcuts)
    - [4. Render Components](#4-render-components)
  - [File Structure](#file-structure)

## Overview

The NEURECTOMY IDE extends VS Code-style interfaces with AI-native features:

| Component             | Inspiration | Purpose                         |
| --------------------- | ----------- | ------------------------------- |
| Agent Execution Graph | Dust.tt     | Visualize AI agent workflows    |
| GitOps Overlay        | Flux/ArgoCD | Deployment status visualization |
| Experiment Sidebar    | MLflow      | ML experiment tracking          |
| Enhanced Status Bar   | VS Code     | Multi-segment status display    |

## Components

### Agent Execution Graph

**Path:** `@/components/agent-graph/AgentExecutionGraph.tsx`

A Dust.tt-style node visualization showing agent workflow with interactive nodes, edges, and reactive layout.

#### Props

\`\`\`typescript
interface AgentExecutionGraphProps {
nodes: AgentNode[];
onNodeSelect?: (node: AgentNode) => void;
className?: string;
showMinimap?: boolean;
interactive?: boolean;
}
\`\`\`

#### Types

\`\`\`typescript
type AgentStatus = "pending" | "running" | "success" | "error" | "paused";

interface AgentNode {
id: string;
name: string;
type: "trigger" | "agent" | "tool" | "decision" | "output";
status: AgentStatus;
x: number;
y: number;
metadata?: {
model?: string;
tokens?: number;
duration?: number;
provider?: string;
error?: string;
};
connections: string[];
}
\`\`\`

#### Usage

\`\`\`tsx
import { AgentExecutionGraph, mockExecutionNodes } from "@/components";

<AgentExecutionGraph
nodes={mockExecutionNodes}
onNodeSelect={(node) => console.log("Selected:", node.name)}
showMinimap={true}
/>
\`\`\`

---

### GitOps Overlay

**Path:** `@/components/gitops/GitOpsOverlay.tsx`

Flux-style deployment status visualization showing pipeline stages and environment sync status.

#### Props

\`\`\`typescript
interface GitOpsOverlayProps {
environments: DeploymentEnvironment[];
pipeline: PipelineRun;
onEnvironmentSelect?: (env: DeploymentEnvironment) => void;
onPipelineAction?: (action: string) => void;
className?: string;
}
\`\`\`

#### Types

\`\`\`typescript
type EnvironmentStatus = "synced" | "syncing" | "pending" | "failed" | "unknown";
type PipelineStage = "checkout" | "build" | "test" | "scan" | "deploy";

interface DeploymentEnvironment {
name: string;
namespace: string;
cluster: string;
status: EnvironmentStatus;
lastSync: string;
revision: string;
resources: { total: number; synced: number; failed: number };
}

interface PipelineRun {
id: string;
branch: string;
commit: string;
author: string;
message: string;
startedAt: string;
status: "pending" | "running" | "success" | "failed";
stages: PipelineStageStatus[];
}
\`\`\`

#### Usage

\`\`\`tsx
import { GitOpsOverlay, mockEnvironments, mockPipeline } from "@/components";

<GitOpsOverlay
environments={mockEnvironments}
pipeline={mockPipeline}
onEnvironmentSelect={(env) => console.log("Selected:", env.name)}
/>
\`\`\`

---

### Experiment Sidebar

**Path:** `@/components/experiments/ExperimentSidebar.tsx`

MLflow-style experiment tracking interface with runs, metrics, and comparison features.

#### Props

\`\`\`typescript
interface ExperimentSidebarProps {
experiments: Experiment[];
onRunSelect?: (run: ExperimentRun) => void;
onCompare?: (runs: ExperimentRun[]) => void;
className?: string;
}
\`\`\`

#### Types

\`\`\`typescript
type RunStatus = "running" | "completed" | "failed" | "scheduled";

interface ExperimentRun {
id: string;
name: string;
status: RunStatus;
startTime: string;
endTime?: string;
metrics: Record<string, number>;
params: Record<string, string>;
tags: string[];
}

interface Experiment {
id: string;
name: string;
description?: string;
createdAt: string;
runs: ExperimentRun[];
}
\`\`\`

#### Usage

\`\`\`tsx
import { ExperimentSidebar, mockExperiments } from "@/components";

<ExperimentSidebar
experiments={mockExperiments}
onRunSelect={(run) => console.log("Selected run:", run.id)}
onCompare={(runs) => console.log("Compare:", runs.length, "runs")}
/>
\`\`\`

---

### Enhanced Status Bar

**Path:** `@/components/status-bar/EnhancedStatusBar.tsx`

Multi-segment status bar with real-time metrics, Git status, agent activity, and deployment indicators.

#### Props

\`\`\`typescript
interface StatusBarProps {
gitBranch: string;
gitSynced: boolean;
metrics: { cpu: number; gpu: number; memory: number };
activeAgents: number;
deploymentStatus: "idle" | "deploying" | "success" | "failed";
agentGraphOpen: boolean;
terminalOpen: boolean;
onToggleAgentGraph: () => void;
onToggleTerminal: () => void;
onGitClick?: () => void;
onAgentsClick?: () => void;
}
\`\`\`

#### Usage

\`\`\`tsx
import { EnhancedStatusBar } from "@/components";

<EnhancedStatusBar
gitBranch="main"
gitSynced={true}
metrics={{ cpu: 45, gpu: 62, memory: 8.4 }}
activeAgents={3}
deploymentStatus="success"
agentGraphOpen={false}
terminalOpen={true}
onToggleAgentGraph={() => setAgentGraphOpen(prev => !prev)}
onToggleTerminal={() => setTerminalOpen(prev => !prev)}
/>
\`\`\`

---

## Motion Utilities

**Path:** `@/lib/motion.tsx`

Wrapper components for Framer Motion v10+ compatibility with className support.

\`\`\`tsx
import { MotionDiv, MotionSpan } from "@/lib/motion";

// Use instead of motion.div when you need className
<MotionDiv
className="my-class"
initial={{ opacity: 0 }}
animate={{ opacity: 1 }}
onClick={() => console.log("Clicked")}

> Content
> </MotionDiv>
> \`\`\`

---

## Keyboard Shortcuts

**Path:** `@/hooks/useIDEKeyboardShortcuts.ts`

IDE-specific keyboard shortcuts for panel toggles, sidebar navigation, and file operations.

### Available Shortcuts

| Shortcut         | Action             | Category   |
| ---------------- | ------------------ | ---------- |
| \`Ctrl+\`\`      | Toggle Terminal    | Panel      |
| \`Ctrl+Shift+A\` | Toggle Agent Graph | Panel      |
| \`Ctrl+B\`       | Toggle Sidebar     | Panel      |
| \`Ctrl+Shift+E\` | Explorer Panel     | Sidebar    |
| \`Ctrl+Shift+F\` | Search Panel       | Sidebar    |
| \`Ctrl+Shift+G\` | Git Panel          | Sidebar    |
| \`Ctrl+Shift+I\` | AI Agents Panel    | Sidebar    |
| \`Ctrl+Shift+X\` | Experiments Panel  | Sidebar    |
| \`Ctrl+Shift+P\` | Extensions Panel   | Sidebar    |
| \`Ctrl+,\`       | Settings           | Sidebar    |
| \`Ctrl+W\`       | Close Active File  | File       |
| \`Ctrl+S\`       | Save File          | File       |
| \`Ctrl+P\`       | Command Palette    | Command    |
| \`Ctrl+1\`       | Focus Editor       | Navigation |

### Usage

\`\`\`tsx
import { useIDEKeyboardShortcuts } from "@/hooks";

function MyIDEComponent() {
const handlers = {
toggleTerminal: () => setTerminalOpen(prev => !prev),
switchToExplorer: () => setActiveActivity("explorer"),
closeActiveFile: () => closeFile(activeFileId),
};

useIDEKeyboardShortcuts(handlers);

return <div>...</div>;
}
\`\`\`

### Getting Shortcut Info

\`\`\`tsx
import { ideShortcuts, formatShortcut, getShortcutsByCategory } from "@/hooks";

// Get formatted shortcut string
const shortcutStr = formatShortcut(ideShortcuts.toggleTerminal);
// Result: "Ctrl+`" (or "⌘`" on Mac)

// Get all shortcuts grouped by category
const grouped = getShortcutsByCategory();
// { panel: [...], sidebar: [...], file: [...], ... }
\`\`\`

---

## Integration Guide

### 1. Import Components

\`\`\`tsx
import {
AgentExecutionGraph,
GitOpsOverlay,
ExperimentSidebar,
EnhancedStatusBar,
} from "@/components";
\`\`\`

### 2. Add State Management

\`\`\`tsx
const [terminalOpen, setTerminalOpen] = useState(true);
const [agentGraphOpen, setAgentGraphOpen] = useState(false);
const [activeActivity, setActiveActivity] = useState<ActivityBarItem>("explorer");
\`\`\`

### 3. Add Keyboard Shortcuts

\`\`\`tsx
import { useIDEKeyboardShortcuts } from "@/hooks";

const shortcutHandlers = useMemo(() => ({
toggleTerminal: () => setTerminalOpen(prev => !prev),
toggleAgentGraph: () => setAgentGraphOpen(prev => !prev),
// ... other handlers
}), []);

useIDEKeyboardShortcuts(shortcutHandlers);
\`\`\`

### 4. Render Components

\`\`\`tsx
// In activity bar
<ActivityBarIcon
icon={FlaskConical}
label="Experiments"
active={activeActivity === "experiments"}
onClick={() => setActiveActivity("experiments")}
/>

// In sidebar
{activeActivity === "experiments" && <ExperimentSidebar experiments={experiments} />}

// Agent graph panel (collapsible)
{agentGraphOpen && (

  <div className="h-48 border-t">
    <AgentExecutionGraph nodes={nodes} />
  </div>
)}

// Status bar at bottom
<EnhancedStatusBar
gitBranch="main"
agentGraphOpen={agentGraphOpen}
onToggleAgentGraph={() => setAgentGraphOpen(prev => !prev)}
// ...
/>
\`\`\`

---

## File Structure

\`\`\`
src/
├── components/
│ ├── index.ts # Main barrel export
│ ├── agent-graph/
│ │ ├── index.ts
│ │ └── AgentExecutionGraph.tsx
│ ├── gitops/
│ │ ├── index.ts
│ │ └── GitOpsOverlay.tsx
│ ├── experiments/
│ │ ├── index.ts
│ │ └── ExperimentSidebar.tsx
│ └── status-bar/
│ ├── index.ts
│ └── EnhancedStatusBar.tsx
├── hooks/
│ ├── index.ts
│ ├── useKeyboardShortcuts.ts
│ └── useIDEKeyboardShortcuts.ts
├── lib/
│ └── motion.tsx # Framer Motion wrappers
└── features/
└── ide/
└── IDEView.tsx # Main IDE component
\`\`\`
