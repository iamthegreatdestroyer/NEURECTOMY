/**
 * @neurectomy/test-config - Test Factories
 *
 * Factory functions for creating test data with sensible defaults.
 */

import { vi } from "vitest";

// ============================================================================
// ID Generators
// ============================================================================

let idCounter = 0;

/**
 * Generate a unique test ID
 */
export function testId(prefix = "test"): string {
  return `${prefix}-${++idCounter}-${Date.now()}`;
}

/**
 * Generate a UUID v4 for testing
 */
export function testUuid(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Reset ID counter between tests
 */
export function resetIdCounter(): void {
  idCounter = 0;
}

// ============================================================================
// Agent Factory
// ============================================================================

export interface TestAgent {
  id: string;
  name: string;
  type: "autonomous" | "assistant" | "worker" | "orchestrator";
  status: "idle" | "running" | "paused" | "error" | "terminated";
  capabilities: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface CreateTestAgentOptions {
  id?: string;
  name?: string;
  type?: TestAgent["type"];
  status?: TestAgent["status"];
  capabilities?: string[];
  metadata?: Record<string, unknown>;
}

/**
 * Create a test agent with sensible defaults
 */
export function createTestAgent(
  options: CreateTestAgentOptions = {}
): TestAgent {
  const id = options.id ?? testUuid();
  const now = new Date().toISOString();

  return {
    id,
    name: options.name ?? `Test Agent ${id.slice(0, 8)}`,
    type: options.type ?? "worker",
    status: options.status ?? "idle",
    capabilities: options.capabilities ?? ["execute", "monitor"],
    metadata: options.metadata ?? {},
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Create multiple test agents
 */
export function createTestAgents(
  count: number,
  options: CreateTestAgentOptions = {}
): TestAgent[] {
  return Array.from({ length: count }, (_, i) =>
    createTestAgent({ ...options, name: `Agent ${i + 1}` })
  );
}

// ============================================================================
// Workflow Factory
// ============================================================================

export interface TestWorkflowStep {
  id: string;
  name: string;
  type: "action" | "condition" | "loop" | "parallel";
  config: Record<string, unknown>;
  next?: string | string[];
}

export interface TestWorkflow {
  id: string;
  name: string;
  description: string;
  version: string;
  status: "draft" | "active" | "paused" | "archived";
  steps: TestWorkflowStep[];
  triggers: Array<{ type: string; config: Record<string, unknown> }>;
  createdAt: string;
  updatedAt: string;
}

export interface CreateTestWorkflowOptions {
  id?: string;
  name?: string;
  description?: string;
  version?: string;
  status?: TestWorkflow["status"];
  steps?: TestWorkflowStep[];
}

/**
 * Create a test workflow with sensible defaults
 */
export function createTestWorkflow(
  options: CreateTestWorkflowOptions = {}
): TestWorkflow {
  const id = options.id ?? testUuid();
  const now = new Date().toISOString();

  return {
    id,
    name: options.name ?? `Test Workflow ${id.slice(0, 8)}`,
    description: options.description ?? "A test workflow for unit testing",
    version: options.version ?? "1.0.0",
    status: options.status ?? "draft",
    steps: options.steps ?? [
      {
        id: testUuid(),
        name: "Start",
        type: "action",
        config: { action: "start" },
        next: undefined,
      },
    ],
    triggers: [],
    createdAt: now,
    updatedAt: now,
  };
}

// ============================================================================
// Container Factory
// ============================================================================

export interface TestContainer {
  id: string;
  name: string;
  image: string;
  status: "created" | "running" | "paused" | "exited" | "dead";
  ports: Array<{ host: number; container: number; protocol: "tcp" | "udp" }>;
  environment: Record<string, string>;
  labels: Record<string, string>;
  createdAt: string;
}

export interface CreateTestContainerOptions {
  id?: string;
  name?: string;
  image?: string;
  status?: TestContainer["status"];
  ports?: TestContainer["ports"];
  environment?: Record<string, string>;
}

/**
 * Create a test container with sensible defaults
 */
export function createTestContainer(
  options: CreateTestContainerOptions = {}
): TestContainer {
  const id = options.id ?? testId("container");
  const now = new Date().toISOString();

  return {
    id,
    name: options.name ?? `test-container-${id.slice(-8)}`,
    image: options.image ?? "neurectomy/test:latest",
    status: options.status ?? "running",
    ports: options.ports ?? [{ host: 8080, container: 80, protocol: "tcp" }],
    environment: options.environment ?? { NODE_ENV: "test" },
    labels: { "neurectomy.test": "true" },
    createdAt: now,
  };
}

// ============================================================================
// Event Factory
// ============================================================================

export interface TestEvent<T = unknown> {
  id: string;
  type: string;
  source: string;
  subject?: string;
  data: T;
  time: string;
  specversion: "1.0";
}

export interface CreateTestEventOptions<T = unknown> {
  id?: string;
  type?: string;
  source?: string;
  subject?: string;
  data?: T;
}

/**
 * Create a CloudEvents-compatible test event
 */
export function createTestEvent<T = unknown>(
  options: CreateTestEventOptions<T> = {}
): TestEvent<T> {
  return {
    id: options.id ?? testUuid(),
    type: options.type ?? "test.event",
    source: options.source ?? "/neurectomy/test",
    subject: options.subject,
    data: options.data ?? ({} as T),
    time: new Date().toISOString(),
    specversion: "1.0",
  };
}

// ============================================================================
// User Factory
// ============================================================================

export interface TestUser {
  id: string;
  email: string;
  name: string;
  role: "admin" | "user" | "viewer";
  permissions: string[];
  createdAt: string;
}

export interface CreateTestUserOptions {
  id?: string;
  email?: string;
  name?: string;
  role?: TestUser["role"];
  permissions?: string[];
}

/**
 * Create a test user with sensible defaults
 */
export function createTestUser(options: CreateTestUserOptions = {}): TestUser {
  const id = options.id ?? testUuid();

  return {
    id,
    email: options.email ?? `test-${id.slice(0, 8)}@neurectomy.test`,
    name: options.name ?? `Test User ${id.slice(0, 8)}`,
    role: options.role ?? "user",
    permissions: options.permissions ?? ["read", "write"],
    createdAt: new Date().toISOString(),
  };
}

// ============================================================================
// Response Factory
// ============================================================================

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

/**
 * Create a paginated API response
 */
export function createPaginatedResponse<T>(
  items: T[],
  options: { page?: number; pageSize?: number; total?: number } = {}
): PaginatedResponse<T> {
  const page = options.page ?? 1;
  const pageSize = options.pageSize ?? 10;
  const total = options.total ?? items.length;
  const totalPages = Math.ceil(total / pageSize);

  return {
    data: items,
    pagination: {
      page,
      pageSize,
      total,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    },
  };
}

// ============================================================================
// Mock Function Factories
// ============================================================================

/**
 * Create a mock function that resolves with a value
 */
export function mockResolve<T>(value: T) {
  return vi.fn().mockResolvedValue(value);
}

/**
 * Create a mock function that rejects with an error
 */
export function mockReject(error: Error | string) {
  return vi
    .fn()
    .mockRejectedValue(typeof error === "string" ? new Error(error) : error);
}

/**
 * Create a mock function that returns different values on consecutive calls
 */
export function mockSequence<T>(...values: T[]) {
  const mock = vi.fn();
  values.forEach((value, index) => {
    mock.mockReturnValueOnce(value);
  });
  return mock;
}

/**
 * Create a mock function that resolves with different values on consecutive calls
 */
export function mockAsyncSequence<T>(...values: T[]) {
  const mock = vi.fn();
  values.forEach((value) => {
    mock.mockResolvedValueOnce(value);
  });
  return mock;
}

// ============================================================================
// Date/Time Helpers
// ============================================================================

/**
 * Create a date relative to now
 */
export function relativeDate(offset: {
  days?: number;
  hours?: number;
  minutes?: number;
  seconds?: number;
}): Date {
  const date = new Date();

  if (offset.days) date.setDate(date.getDate() + offset.days);
  if (offset.hours) date.setHours(date.getHours() + offset.hours);
  if (offset.minutes) date.setMinutes(date.getMinutes() + offset.minutes);
  if (offset.seconds) date.setSeconds(date.getSeconds() + offset.seconds);

  return date;
}

/**
 * Create a fixed date for consistent testing
 */
export function fixedDate(isoString = "2024-01-15T12:00:00.000Z"): Date {
  return new Date(isoString);
}

/**
 * Mock Date.now() to return a fixed timestamp
 */
export function mockDateNow(timestamp: number | Date): () => void {
  const original = Date.now;
  const ts = typeof timestamp === "number" ? timestamp : timestamp.getTime();

  Date.now = vi.fn(() => ts);

  return () => {
    Date.now = original;
  };
}
