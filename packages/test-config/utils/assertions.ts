/**
 * @neurectomy/test-config - Custom Assertions
 *
 * Custom assertion utilities for more expressive tests.
 */

import { expect } from "vitest";

// ============================================================================
// Type Assertions
// ============================================================================

/**
 * Assert that a value is defined (not undefined or null)
 */
export function assertDefined<T>(
  value: T | null | undefined,
  message?: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new Error(
      message ?? `Expected value to be defined, but got ${value}`
    );
  }
}

/**
 * Assert that a value is a string
 */
export function assertString(
  value: unknown,
  message?: string
): asserts value is string {
  if (typeof value !== "string") {
    throw new Error(message ?? `Expected string, but got ${typeof value}`);
  }
}

/**
 * Assert that a value is a number
 */
export function assertNumber(
  value: unknown,
  message?: string
): asserts value is number {
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new Error(message ?? `Expected number, but got ${typeof value}`);
  }
}

/**
 * Assert that a value is an array
 */
export function assertArray<T>(
  value: unknown,
  message?: string
): asserts value is T[] {
  if (!Array.isArray(value)) {
    throw new Error(message ?? `Expected array, but got ${typeof value}`);
  }
}

/**
 * Assert that a value is an object (not null, not array)
 */
export function assertObject(
  value: unknown,
  message?: string
): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(
      message ??
        `Expected object, but got ${Array.isArray(value) ? "array" : typeof value}`
    );
  }
}

// ============================================================================
// Comparison Assertions
// ============================================================================

/**
 * Assert that two arrays have the same elements (order-independent)
 */
export function assertSameElements<T>(actual: T[], expected: T[]): void {
  expect(actual).toHaveLength(expected.length);
  expect(actual.sort()).toEqual(expected.sort());
}

/**
 * Assert that an array contains all specified elements
 */
export function assertContainsAll<T>(actual: T[], expected: T[]): void {
  for (const item of expected) {
    expect(actual).toContain(item);
  }
}

/**
 * Assert that an object has specific keys
 */
export function assertHasKeys(
  obj: Record<string, unknown>,
  keys: string[]
): void {
  for (const key of keys) {
    expect(obj).toHaveProperty(key);
  }
}

/**
 * Assert that an object matches a partial shape
 */
export function assertShape<T extends Record<string, unknown>>(
  actual: unknown,
  expected: Partial<T>
): void {
  assertObject(actual);
  expect(actual).toMatchObject(expected);
}

// ============================================================================
// Async Assertions
// ============================================================================

/**
 * Assert that a promise resolves successfully
 */
export async function assertResolves<T>(
  promise: Promise<T>,
  expectedValue?: T
): Promise<T> {
  const result = await promise;
  if (expectedValue !== undefined) {
    expect(result).toEqual(expectedValue);
  }
  return result;
}

/**
 * Assert that a promise rejects with an error
 */
export async function assertRejects(
  promise: Promise<unknown>,
  errorMessage?: string | RegExp
): Promise<Error> {
  try {
    await promise;
    throw new Error("Expected promise to reject, but it resolved");
  } catch (error) {
    expect(error).toBeInstanceOf(Error);
    if (errorMessage) {
      if (typeof errorMessage === "string") {
        expect((error as Error).message).toContain(errorMessage);
      } else {
        expect((error as Error).message).toMatch(errorMessage);
      }
    }
    return error as Error;
  }
}

/**
 * Assert that a function eventually returns true
 */
export async function assertEventually(
  fn: () => boolean | Promise<boolean>,
  options: { timeout?: number; interval?: number; message?: string } = {}
): Promise<void> {
  const {
    timeout = 5000,
    interval = 100,
    message = "Condition never became true",
  } = options;
  const start = Date.now();

  while (Date.now() - start < timeout) {
    const result = await fn();
    if (result) return;
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(message);
}

// ============================================================================
// Error Assertions
// ============================================================================

/**
 * Assert that a function throws a specific error type
 */
export function assertThrowsType<E extends Error>(
  fn: () => void,
  errorType: new (...args: unknown[]) => E,
  message?: string | RegExp
): E {
  try {
    fn();
    throw new Error(
      `Expected function to throw ${errorType.name}, but it did not throw`
    );
  } catch (error) {
    expect(error).toBeInstanceOf(errorType);
    if (message) {
      if (typeof message === "string") {
        expect((error as Error).message).toContain(message);
      } else {
        expect((error as Error).message).toMatch(message);
      }
    }
    return error as E;
  }
}

/**
 * Assert that an async function throws a specific error type
 */
export async function assertThrowsTypeAsync<E extends Error>(
  fn: () => Promise<void>,
  errorType: new (...args: unknown[]) => E,
  message?: string | RegExp
): Promise<E> {
  try {
    await fn();
    throw new Error(
      `Expected function to throw ${errorType.name}, but it did not throw`
    );
  } catch (error) {
    expect(error).toBeInstanceOf(errorType);
    if (message) {
      if (typeof message === "string") {
        expect((error as Error).message).toContain(message);
      } else {
        expect((error as Error).message).toMatch(message);
      }
    }
    return error as E;
  }
}

// ============================================================================
// Snapshot Assertions
// ============================================================================

/**
 * Assert that a value matches a snapshot, excluding specified keys
 */
export function assertSnapshotExcluding<T extends Record<string, unknown>>(
  value: T,
  excludeKeys: (keyof T)[]
): void {
  const filtered = { ...value };
  for (const key of excludeKeys) {
    delete filtered[key];
  }
  expect(filtered).toMatchSnapshot();
}

/**
 * Assert that a value matches a snapshot with dynamic values replaced
 */
export function assertSnapshotWithDynamicValues<
  T extends Record<string, unknown>,
>(value: T, dynamicKeys: (keyof T)[]): void {
  const replaced = { ...value };
  for (const key of dynamicKeys) {
    if (key in replaced) {
      replaced[key] = `[${String(key)}]` as T[keyof T];
    }
  }
  expect(replaced).toMatchSnapshot();
}

// ============================================================================
// Mock Assertions
// ============================================================================

/**
 * Assert that a mock was called with specific arguments at a specific call index
 */
export function assertCalledWithAt<T extends (...args: unknown[]) => unknown>(
  mockFn: ReturnType<typeof import("vitest").vi.fn<T>>,
  callIndex: number,
  expectedArgs: Parameters<T>
): void {
  expect(mockFn.mock.calls[callIndex]).toEqual(expectedArgs);
}

/**
 * Assert that a mock was called in a specific order with specific arguments
 */
export function assertCallOrder<T extends (...args: unknown[]) => unknown>(
  mockFn: ReturnType<typeof import("vitest").vi.fn<T>>,
  expectedCalls: Parameters<T>[]
): void {
  expect(mockFn.mock.calls).toHaveLength(expectedCalls.length);
  expectedCalls.forEach((args, index) => {
    expect(mockFn.mock.calls[index]).toEqual(args);
  });
}

/**
 * Assert that multiple mocks were called in a specific order
 */
export function assertMockOrder(
  ...mocks: Array<ReturnType<typeof import("vitest").vi.fn>>
): void {
  const orders = mocks.map(
    (mock) => mock.mock.invocationCallOrder[0] ?? Infinity
  );

  for (let i = 1; i < orders.length; i++) {
    expect(orders[i]).toBeGreaterThan(orders[i - 1]);
  }
}

// ============================================================================
// DOM Assertions (for React components)
// ============================================================================

/**
 * Assert that an element is visible
 */
export function assertVisible(element: Element): void {
  expect(element).toBeVisible();
}

/**
 * Assert that an element is accessible
 */
export function assertAccessible(
  element: Element,
  options: {
    role?: string;
    name?: string;
    description?: string;
  } = {}
): void {
  if (options.role) {
    expect(element).toHaveAttribute("role", options.role);
  }
  if (options.name) {
    expect(element).toHaveAccessibleName(options.name);
  }
  if (options.description) {
    expect(element).toHaveAccessibleDescription(options.description);
  }
}

/**
 * Assert that a form field has an error
 */
export function assertFieldError(
  element: Element,
  errorMessage?: string
): void {
  expect(element).toBeInvalid();
  if (errorMessage) {
    expect(element).toHaveAccessibleErrorMessage(errorMessage);
  }
}
