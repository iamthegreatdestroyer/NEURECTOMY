/**
 * @neurectomy/test-config - Base Test Setup
 *
 * This file runs before all tests in Node.js environment packages.
 * Configure global mocks, test utilities, and environment here.
 */

import { beforeAll, afterAll, afterEach, vi } from "vitest";

// ============================================================================
// Console Spy Setup
// ============================================================================

const originalConsole = {
  log: console.log,
  warn: console.warn,
  error: console.error,
  debug: console.debug,
};

// Track console calls for assertion in tests
export const consoleMocks = {
  log: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  debug: vi.fn(),
};

// ============================================================================
// Environment Variables
// ============================================================================

const testEnv = {
  NODE_ENV: "test",
  LOG_LEVEL: "silent",
  NEURECTOMY_TEST: "true",
};

// ============================================================================
// Global Setup
// ============================================================================

beforeAll(() => {
  // Set test environment variables
  Object.assign(process.env, testEnv);

  // Setup console spies (suppress output in tests but track calls)
  if (process.env.VITEST_SILENT !== "false") {
    console.log = consoleMocks.log;
    console.warn = consoleMocks.warn;
    console.error = consoleMocks.error;
    console.debug = consoleMocks.debug;
  }
});

afterEach(() => {
  // Clear all mocks between tests
  vi.clearAllMocks();

  // Reset console mocks
  consoleMocks.log.mockClear();
  consoleMocks.warn.mockClear();
  consoleMocks.error.mockClear();
  consoleMocks.debug.mockClear();
});

afterAll(() => {
  // Restore original console
  console.log = originalConsole.log;
  console.warn = originalConsole.warn;
  console.error = originalConsole.error;
  console.debug = originalConsole.debug;

  // Reset all mocks
  vi.restoreAllMocks();
});

// ============================================================================
// Global Test Utilities
// ============================================================================

/**
 * Wait for a specified duration
 * Useful for testing async operations with timeouts
 */
export const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Wait for the next tick of the event loop
 */
export const nextTick = (): Promise<void> =>
  new Promise((resolve) => setImmediate(resolve));

/**
 * Create a deferred promise for testing async flows
 */
export function createDeferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Retry a function until it succeeds or max attempts reached
 */
export async function retry<T>(
  fn: () => T | Promise<T>,
  maxAttempts = 3,
  delay = 100
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await wait(delay * attempt);
      }
    }
  }
  throw lastError;
}

// ============================================================================
// Type Declarations
// ============================================================================

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Vi {
    interface Assertion {
      toMatchSnapshot(): void;
    }
  }
}
