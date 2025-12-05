/**
 * @neurectomy/test-config - API Mocks
 *
 * Mock implementations for API clients and network requests.
 * Uses MSW (Mock Service Worker) for realistic API mocking.
 */

import { http, HttpResponse, delay } from "msw";
import { setupServer } from "msw/node";
import { vi } from "vitest";

// ============================================================================
// Types
// ============================================================================

export interface MockResponseOptions {
  status?: number;
  delay?: number;
  headers?: Record<string, string>;
}

export interface MockGraphQLResponse<T = unknown> {
  data?: T;
  errors?: Array<{ message: string; path?: string[] }>;
}

// ============================================================================
// MSW Server Setup
// ============================================================================

// Default handlers - can be overridden in tests
const defaultHandlers = [
  // Health check endpoint
  http.get("/api/health", () => {
    return HttpResponse.json({ status: "ok", timestamp: Date.now() });
  }),

  // Agent endpoints
  http.get("/api/agents", () => {
    return HttpResponse.json({
      agents: [],
      total: 0,
      page: 1,
      pageSize: 10,
    });
  }),

  http.get("/api/agents/:id", ({ params }) => {
    const { id } = params;
    return HttpResponse.json({
      id,
      name: `Test Agent ${id}`,
      status: "idle",
      createdAt: new Date().toISOString(),
    });
  }),

  // Workflow endpoints
  http.get("/api/workflows", () => {
    return HttpResponse.json({
      workflows: [],
      total: 0,
    });
  }),

  // Container endpoints
  http.get("/api/containers", () => {
    return HttpResponse.json({
      containers: [],
      total: 0,
    });
  }),

  // GraphQL endpoint
  http.post("/graphql", async ({ request }) => {
    const body = (await request.json()) as {
      query: string;
      variables?: Record<string, unknown>;
    };
    const { query } = body;

    // Basic GraphQL query handling
    if (query.includes("agents")) {
      return HttpResponse.json({
        data: {
          agents: {
            nodes: [],
            totalCount: 0,
          },
        },
      });
    }

    if (query.includes("workflows")) {
      return HttpResponse.json({
        data: {
          workflows: {
            nodes: [],
            totalCount: 0,
          },
        },
      });
    }

    // Default response
    return HttpResponse.json({
      data: null,
      errors: [{ message: "Unknown query" }],
    });
  }),
];

// Create the MSW server
export const mswServer = setupServer(...defaultHandlers);

// ============================================================================
// Server Control Functions
// ============================================================================

/**
 * Start the MSW server
 * Call in beforeAll hook
 */
export function startMockServer() {
  mswServer.listen({
    onUnhandledRequest: "warn",
  });
}

/**
 * Stop the MSW server
 * Call in afterAll hook
 */
export function stopMockServer() {
  mswServer.close();
}

/**
 * Reset handlers to defaults
 * Call in afterEach hook
 */
export function resetMockHandlers() {
  mswServer.resetHandlers();
}

// ============================================================================
// Handler Factories
// ============================================================================

/**
 * Create a mock REST endpoint
 */
export function mockRestEndpoint<T>(
  method: "get" | "post" | "put" | "patch" | "delete",
  path: string,
  response: T,
  options: MockResponseOptions = {}
) {
  const { status = 200, delay: delayMs = 0, headers = {} } = options;

  const handler = http[method](path, async () => {
    if (delayMs > 0) {
      await delay(delayMs);
    }
    return HttpResponse.json(response, {
      status,
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
    });
  });

  mswServer.use(handler);
  return handler;
}

/**
 * Create a mock GraphQL response
 */
export function mockGraphQLQuery<T>(
  operationName: string,
  response: MockGraphQLResponse<T>,
  options: MockResponseOptions = {}
) {
  const { status = 200, delay: delayMs = 0 } = options;

  const handler = http.post("/graphql", async ({ request }) => {
    const body = (await request.json()) as { operationName?: string };

    if (body.operationName === operationName) {
      if (delayMs > 0) {
        await delay(delayMs);
      }
      return HttpResponse.json(response, { status });
    }

    // Pass through to default handlers
    return undefined;
  });

  mswServer.use(handler);
  return handler;
}

/**
 * Create a mock error response
 */
export function mockErrorResponse(
  method: "get" | "post" | "put" | "patch" | "delete",
  path: string,
  errorCode: number,
  message: string
) {
  const handler = http[method](path, () => {
    return HttpResponse.json(
      { error: message, code: errorCode },
      { status: errorCode }
    );
  });

  mswServer.use(handler);
  return handler;
}

/**
 * Create a network error mock
 */
export function mockNetworkError(
  method: "get" | "post" | "put" | "patch" | "delete",
  path: string
) {
  const handler = http[method](path, () => {
    return HttpResponse.error();
  });

  mswServer.use(handler);
  return handler;
}

// ============================================================================
// Fetch Mock (Alternative to MSW)
// ============================================================================

export interface FetchMock {
  mockResponse: (response: unknown, options?: MockResponseOptions) => void;
  mockError: (error: Error) => void;
  mockOnce: (response: unknown, options?: MockResponseOptions) => void;
  reset: () => void;
  calls: Array<{ url: string; options?: RequestInit }>;
}

/**
 * Create a simple fetch mock for cases where MSW is overkill
 */
export function createFetchMock(): FetchMock {
  const calls: Array<{ url: string; options?: RequestInit }> = [];
  let mockResponseValue: unknown = null;
  let mockOptions: MockResponseOptions = {};
  let mockError: Error | null = null;
  let isOnce = false;

  const mockFetch = vi.fn(async (url: string, options?: RequestInit) => {
    calls.push({ url, options });

    if (mockError) {
      const error = mockError;
      if (isOnce) {
        mockError = null;
        isOnce = false;
      }
      throw error;
    }

    const response = mockResponseValue;
    if (isOnce) {
      mockResponseValue = null;
      isOnce = false;
    }

    return new Response(JSON.stringify(response), {
      status: mockOptions.status ?? 200,
      headers: {
        "Content-Type": "application/json",
        ...mockOptions.headers,
      },
    });
  });

  // Replace global fetch
  const originalFetch = globalThis.fetch;
  globalThis.fetch = mockFetch;

  return {
    mockResponse(response: unknown, options: MockResponseOptions = {}) {
      mockResponseValue = response;
      mockOptions = options;
      mockError = null;
      isOnce = false;
    },
    mockError(error: Error) {
      mockError = error;
      mockResponseValue = null;
      isOnce = false;
    },
    mockOnce(response: unknown, options: MockResponseOptions = {}) {
      mockResponseValue = response;
      mockOptions = options;
      mockError = null;
      isOnce = true;
    },
    reset() {
      mockResponseValue = null;
      mockOptions = {};
      mockError = null;
      isOnce = false;
      calls.length = 0;
      mockFetch.mockClear();
      globalThis.fetch = originalFetch;
    },
    calls,
  };
}

// ============================================================================
// WebSocket Mock
// ============================================================================

export interface WebSocketMock {
  instance: WebSocket | null;
  messages: Array<{ type: "send" | "receive"; data: unknown }>;
  simulateMessage: (data: unknown) => void;
  simulateOpen: () => void;
  simulateClose: (code?: number, reason?: string) => void;
  simulateError: (error: Event) => void;
  reset: () => void;
}

/**
 * Create a WebSocket mock for testing real-time features
 */
export function createWebSocketMock(): WebSocketMock {
  const messages: Array<{ type: "send" | "receive"; data: unknown }> = [];
  let currentInstance: WebSocket | null = null;

  // Mock WebSocket class
  class MockWebSocket {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;

    readyState = MockWebSocket.CONNECTING;
    url: string;
    protocol = "";
    binaryType: BinaryType = "blob";
    bufferedAmount = 0;
    extensions = "";

    onopen: ((event: Event) => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;

    constructor(url: string) {
      this.url = url;
      currentInstance = this as unknown as WebSocket;
    }

    send(data: unknown) {
      messages.push({ type: "send", data });
    }

    close(_code?: number, _reason?: string) {
      this.readyState = MockWebSocket.CLOSED;
    }

    addEventListener = vi.fn();
    removeEventListener = vi.fn();
    dispatchEvent = vi.fn();
  }

  // Store original and replace
  const OriginalWebSocket = globalThis.WebSocket;
  globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;

  return {
    get instance() {
      return currentInstance;
    },
    messages,
    simulateMessage(data: unknown) {
      if (
        currentInstance &&
        (currentInstance as unknown as MockWebSocket).onmessage
      ) {
        messages.push({ type: "receive", data });
        (currentInstance as unknown as MockWebSocket).onmessage!(
          new MessageEvent("message", { data: JSON.stringify(data) })
        );
      }
    },
    simulateOpen() {
      if (currentInstance) {
        (currentInstance as unknown as MockWebSocket).readyState =
          MockWebSocket.OPEN;
        if ((currentInstance as unknown as MockWebSocket).onopen) {
          (currentInstance as unknown as MockWebSocket).onopen!(
            new Event("open")
          );
        }
      }
    },
    simulateClose(code = 1000, reason = "") {
      if (currentInstance) {
        (currentInstance as unknown as MockWebSocket).readyState =
          MockWebSocket.CLOSED;
        if ((currentInstance as unknown as MockWebSocket).onclose) {
          (currentInstance as unknown as MockWebSocket).onclose!(
            new CloseEvent("close", { code, reason })
          );
        }
      }
    },
    simulateError(error: Event) {
      if (
        currentInstance &&
        (currentInstance as unknown as MockWebSocket).onerror
      ) {
        (currentInstance as unknown as MockWebSocket).onerror!(error);
      }
    },
    reset() {
      messages.length = 0;
      currentInstance = null;
      globalThis.WebSocket = OriginalWebSocket;
    },
  };
}
