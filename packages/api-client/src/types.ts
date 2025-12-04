/**
 * API response types for Neurectomy services.
 */

export interface Agent {
  id: string;
  name: string;
  type: "ai" | "tool" | "composite" | "workflow";
  version?: string;
  description?: string;
  capabilities: string[];
  status: "active" | "inactive" | "error";
  metadata?: Record<string, string>;
  createdAt: string;
  updatedAt: string;
}

export interface Task {
  id: string;
  name: string;
  description?: string;
  agentId: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  startedAt?: string;
  completedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  version?: string;
  tasks: Task[];
  status: "draft" | "active" | "archived";
  variables?: Record<string, unknown>;
  metadata?: Record<string, string>;
  createdAt: string;
  updatedAt: string;
}

export interface ExecutionResult<T = unknown> {
  id: string;
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  metrics?: {
    durationMs: number;
    tokensUsed?: number;
    cost?: number;
  };
  logs?: Array<{
    level: "debug" | "info" | "warn" | "error";
    message: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
  }>;
  createdAt: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  uptime: number;
  checks: Record<
    string,
    {
      status: "pass" | "fail";
      message?: string;
      latencyMs?: number;
    }
  >;
}
