/**
 * NEURECTOMY Type Definitions
 *
 * Core type definitions shared across the platform.
 */

// =============================================================================
// Agent Types
// =============================================================================

export type AgentStatus = 'idle' | 'running' | 'paused' | 'error' | 'completed';

export type AgentTier = 'foundational' | 'specialist' | 'innovator' | 'meta' | 'domain' | 'emerging' | 'human-centric' | 'enterprise';

export interface Agent {
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

export interface AgentMetadata {
  icon?: string;
  color?: string;
  tags?: string[];
  dependencies?: string[];
  performance?: AgentPerformanceMetrics;
}

export interface AgentPerformanceMetrics {
  avgResponseTime: number;
  successRate: number;
  totalInvocations: number;
  lastInvocation?: Date;
}

// =============================================================================
// Workflow Types
// =============================================================================

export type WorkflowStatus = 'draft' | 'active' | 'paused' | 'completed' | 'failed';

export interface Workflow {
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

export interface WorkflowNode {
  id: string;
  type: 'agent' | 'condition' | 'input' | 'output' | 'transform' | 'parallel' | 'loop';
  position: Position3D;
  data: WorkflowNodeData;
  status?: AgentStatus;
}

export interface WorkflowNodeData {
  label: string;
  agentId?: string;
  config?: Record<string, unknown>;
  inputs?: PortDefinition[];
  outputs?: PortDefinition[];
}

export interface WorkflowEdge {
  id: string;
  source: string;
  sourceHandle?: string;
  target: string;
  targetHandle?: string;
  label?: string;
  animated?: boolean;
}

export interface PortDefinition {
  id: string;
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'any';
  required?: boolean;
  default?: unknown;
}

// =============================================================================
// 3D/4D Visualization Types
// =============================================================================

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Transform3D {
  position: Position3D;
  rotation: Quaternion;
  scale: Position3D;
}

export interface TimelineKeyframe<T = unknown> {
  time: number;
  value: T;
  easing?: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'cubic-bezier';
  bezierHandles?: [number, number, number, number];
}

export interface Timeline {
  id: string;
  name: string;
  duration: number;
  fps: number;
  keyframes: Map<string, TimelineKeyframe[]>;
  markers: TimelineMarker[];
}

export interface TimelineMarker {
  time: number;
  name: string;
  color?: string;
}

// =============================================================================
// Container Types
// =============================================================================

export type ContainerStatus = 'created' | 'running' | 'paused' | 'exited' | 'dead';

export interface Container {
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

export interface PortMapping {
  hostPort: number;
  containerPort: number;
  protocol: 'tcp' | 'udp';
}

export interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readOnly?: boolean;
}

export interface ContainerHealth {
  status: 'healthy' | 'unhealthy' | 'starting' | 'none';
  failingStreak: number;
  lastCheck?: Date;
}

// =============================================================================
// Event Types
// =============================================================================

export interface NeurectomyEvent<T = unknown> {
  id: string;
  type: string;
  timestamp: Date;
  source: string;
  data: T;
  metadata?: Record<string, unknown>;
}

export type EventHandler<T = unknown> = (event: NeurectomyEvent<T>) => void | Promise<void>;

// =============================================================================
// API Types
// =============================================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta?: ApiMeta;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ApiMeta {
  page?: number;
  pageSize?: number;
  totalCount?: number;
  totalPages?: number;
}

export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// =============================================================================
// User & Auth Types
// =============================================================================

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: UserRole;
  preferences: UserPreferences;
  createdAt: Date;
  lastLoginAt?: Date;
}

export type UserRole = 'admin' | 'developer' | 'viewer';

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: NotificationPreferences;
  editor: EditorPreferences;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  desktop: boolean;
}

export interface EditorPreferences {
  fontSize: number;
  fontFamily: string;
  tabSize: number;
  insertSpaces: boolean;
  wordWrap: boolean;
  minimap: boolean;
}

// =============================================================================
// Re-exports
// =============================================================================

export * from './agent';
export * from './workflow';
export * from './container';
export * from './visualization';
