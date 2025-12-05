# @neurectomy/types API Reference

Core TypeScript type definitions shared across the NEURECTOMY platform.

## Installation

```bash
pnpm add @neurectomy/types
```

## Type Categories

### Agent Types

#### AgentStatus

```typescript
type AgentStatus =
  | "idle"
  | "running"
  | "paused"
  | "error"
  | "completed"
  | "active";
```

#### AgentTier

```typescript
type AgentTier =
  | "foundational"
  | "specialist"
  | "innovator"
  | "meta"
  | "domain"
  | "emerging"
  | "human-centric"
  | "enterprise";
```

#### Agent

```typescript
interface Agent {
  id: string;
  name: string;
  codename: string;
  tier: AgentTier;
  status: AgentStatus;
  description: string;
  philosophy: string;
  capabilities: string[];
  version: string;
  createdAt: Date;
  updatedAt: Date;
  metadata: AgentMetadata;
}
```

#### AgentMetadata

```typescript
interface AgentMetadata {
  icon?: string;
  color?: string;
  tags?: string[];
  dependencies?: string[];
  performance?: AgentPerformanceMetrics;
}
```

#### AgentConfig

Configuration for creating/updating agents:

```typescript
interface AgentConfig {
  name: string;
  codename: string;
  tier: AgentTier;
  description: string;
  philosophy: string;
  capabilities: string[];
  metadata?: Partial<AgentMetadata>;
}
```

#### AgentState

Runtime state for agents:

```typescript
interface AgentState {
  status: AgentStatus;
  currentTask?: string;
  progress?: number;
  lastError?: string;
  metrics?: AgentPerformanceMetrics;
}
```

#### AgentTemplate

Pre-configured agent templates:

```typescript
interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  tier: AgentTier;
  defaultCapabilities: string[];
  requiredInputs: string[];
  optionalInputs?: string[];
  category: string;
}
```

---

### Workflow Types

#### WorkflowStatus

```typescript
type WorkflowStatus = "draft" | "active" | "paused" | "completed" | "failed";
```

#### Workflow

```typescript
interface Workflow {
  id: string;
  name: string;
  description?: string;
  status: WorkflowStatus;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, unknown>;
}
```

#### WorkflowNode

```typescript
interface WorkflowNode {
  id: string;
  type:
    | "agent"
    | "condition"
    | "input"
    | "output"
    | "transform"
    | "parallel"
    | "loop";
  position: Position3D;
  data: WorkflowNodeData;
  status?: AgentStatus;
}
```

#### WorkflowEdge

```typescript
interface WorkflowEdge {
  id: string;
  source: string;
  sourceHandle?: string;
  target: string;
  targetHandle?: string;
  label?: string;
  animated?: boolean;
}
```

#### TaskDefinition

```typescript
interface TaskDefinition {
  name: string;
  description?: string;
  agentId: string;
  inputs: Record<string, unknown>;
  timeout?: number;
  priority?: "low" | "normal" | "high" | "critical";
  retryPolicy?: {
    maxRetries: number;
    backoffMs: number;
    exponential?: boolean;
  };
}
```

---

### 3D/4D Visualization Types

#### Position3D

```typescript
interface Position3D {
  x: number;
  y: number;
  z: number;
}
```

#### Quaternion

```typescript
interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}
```

#### Transform3D

```typescript
interface Transform3D {
  position: Position3D;
  rotation: Quaternion;
  scale: Position3D;
}
```

#### Timeline

```typescript
interface Timeline {
  id: string;
  name: string;
  duration: number;
  fps: number;
  keyframes: Map<string, TimelineKeyframe[]>;
  markers: TimelineMarker[];
}
```

#### TimelineKeyframe

```typescript
interface TimelineKeyframe<T = unknown> {
  time: number;
  value: T;
  easing?: "linear" | "ease-in" | "ease-out" | "ease-in-out" | "cubic-bezier";
  bezierHandles?: [number, number, number, number];
}
```

---

### Container Types

#### ContainerStatus

```typescript
type ContainerStatus = "created" | "running" | "paused" | "exited" | "dead";
```

#### Container

```typescript
interface Container {
  id: string;
  name: string;
  image: string;
  status: ContainerStatus;
  ports: PortMapping[];
  volumes: VolumeMount[];
  environment: Record<string, string>;
  labels: Record<string, string>;
  createdAt: Date;
  startedAt?: Date;
  stoppedAt?: Date;
  health?: ContainerHealth;
}
```

#### PortMapping

```typescript
interface PortMapping {
  hostPort: number;
  containerPort: number;
  protocol: "tcp" | "udp";
}
```

#### VolumeMount

```typescript
interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readOnly?: boolean;
}
```

---

### Event Types

#### NeurectomyEvent

```typescript
interface NeurectomyEvent<T = unknown> {
  id: string;
  type: string;
  timestamp: Date;
  source: string;
  data: T;
  metadata?: Record<string, unknown>;
}
```

#### EventHandler

```typescript
type EventHandler<T = unknown> = (
  event: NeurectomyEvent<T>
) => void | Promise<void>;
```

---

### API Types

#### ApiResponse

```typescript
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta?: ApiMeta;
}
```

#### ApiError

```typescript
interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}
```

#### ApiMeta

```typescript
interface ApiMeta {
  page?: number;
  pageSize?: number;
  totalCount?: number;
  totalPages?: number;
}
```

#### PaginationParams

```typescript
interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}
```

---

### User & Auth Types

#### User

```typescript
interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: UserRole;
  preferences: UserPreferences;
  createdAt: Date;
  lastLoginAt?: Date;
}
```

#### UserRole

```typescript
type UserRole = "admin" | "developer" | "viewer";
```

#### UserPreferences

```typescript
interface UserPreferences {
  theme: "light" | "dark" | "system";
  language: string;
  notifications: NotificationPreferences;
  editor: EditorPreferences;
}
```

---

## Usage Examples

### Type Guards

```typescript
import type { Agent, AgentStatus } from "@neurectomy/types";

function isActiveAgent(agent: Agent): boolean {
  return agent.status === "running" || agent.status === "active";
}

function isErrorStatus(status: AgentStatus): status is "error" {
  return status === "error";
}
```

### Creating Type-Safe Objects

```typescript
import type { AgentConfig, TaskDefinition } from "@neurectomy/types";

const config: AgentConfig = {
  name: "My Agent",
  codename: "MYAGENT",
  tier: "specialist",
  description: "A custom agent",
  philosophy: "Do one thing well",
  capabilities: ["task-1", "task-2"],
};

const task: TaskDefinition = {
  name: "Process Data",
  agentId: "agent-123",
  inputs: { data: [1, 2, 3] },
  timeout: 30000,
  priority: "normal",
  retryPolicy: {
    maxRetries: 3,
    backoffMs: 1000,
    exponential: true,
  },
};
```

### Working with Generics

```typescript
import type { ApiResponse, NeurectomyEvent } from "@neurectomy/types";

// Typed API response
type AgentListResponse = ApiResponse<Agent[]>;

// Typed event
interface TaskCompleted {
  taskId: string;
  result: unknown;
}

type TaskCompletedEvent = NeurectomyEvent<TaskCompleted>;
```

---

## Module Organization

All types are exported from the package root:

```typescript
import type {
  // Agent types
  Agent,
  AgentConfig,
  AgentState,
  AgentStatus,
  AgentTier,
  AgentTemplate,
  AgentMetrics,
  AgentMetadata,
  AgentPerformanceMetrics,

  // Workflow types
  Workflow,
  WorkflowNode,
  WorkflowEdge,
  WorkflowStatus,
  WorkflowDefinition,
  WorkflowNodeData,
  TaskDefinition,
  PortDefinition,

  // 3D/4D types
  Position3D,
  Quaternion,
  Transform3D,
  Timeline,
  TimelineKeyframe,
  TimelineMarker,

  // Container types
  Container,
  ContainerStatus,
  ContainerHealth,
  PortMapping,
  VolumeMount,

  // Event types
  NeurectomyEvent,
  EventHandler,

  // API types
  ApiResponse,
  ApiError,
  ApiMeta,
  PaginationParams,

  // User types
  User,
  UserRole,
  UserPreferences,
  NotificationPreferences,
  EditorPreferences,
} from "@neurectomy/types";
```
