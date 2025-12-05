/**
 * @neurectomy/test-config - Async Test Helpers
 *
 * Utilities for testing asynchronous code.
 */

import { vi } from "vitest";

// ============================================================================
// Timing Utilities
// ============================================================================

/**
 * Wait for a specified duration
 */
export function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait for the next tick of the event loop
 */
export function nextTick(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/**
 * Wait for multiple ticks
 */
export async function waitTicks(count: number): Promise<void> {
  for (let i = 0; i < count; i++) {
    await nextTick();
  }
}

/**
 * Flush all pending promises
 */
export function flushPromises(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/**
 * Advance fake timers and flush promises
 */
export async function advanceTimersAndFlush(ms: number): Promise<void> {
  vi.advanceTimersByTime(ms);
  await flushPromises();
}

// ============================================================================
// Deferred Promise
// ============================================================================

export interface Deferred<T = void> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
  isResolved: () => boolean;
  isRejected: () => boolean;
  isPending: () => boolean;
}

/**
 * Create a deferred promise for controlling async flow in tests
 */
export function createDeferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  let resolved = false;
  let rejected = false;

  const promise = new Promise<T>((res, rej) => {
    resolve = (value: T) => {
      resolved = true;
      res(value);
    };
    reject = (reason?: unknown) => {
      rejected = true;
      rej(reason);
    };
  });

  return {
    promise,
    resolve,
    reject,
    isResolved: () => resolved,
    isRejected: () => rejected,
    isPending: () => !resolved && !rejected,
  };
}

// ============================================================================
// Retry Utilities
// ============================================================================

export interface RetryOptions {
  maxAttempts?: number;
  delay?: number;
  backoff?: "linear" | "exponential";
  maxDelay?: number;
  onRetry?: (attempt: number, error: unknown) => void;
}

/**
 * Retry a function until it succeeds or max attempts reached
 */
export async function retry<T>(
  fn: () => T | Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delay = 100,
    backoff = "linear",
    maxDelay = 5000,
    onRetry,
  } = options;

  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < maxAttempts) {
        onRetry?.(attempt, error);

        let waitTime = delay;
        if (backoff === "exponential") {
          waitTime = Math.min(delay * Math.pow(2, attempt - 1), maxDelay);
        } else {
          waitTime = Math.min(delay * attempt, maxDelay);
        }

        await wait(waitTime);
      }
    }
  }

  throw lastError;
}

/**
 * Retry until a condition is met
 */
export async function retryUntil<T>(
  fn: () => T | Promise<T>,
  condition: (result: T) => boolean,
  options: RetryOptions = {}
): Promise<T> {
  const { maxAttempts = 10, delay = 100 } = options;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = await fn();
    if (condition(result)) {
      return result;
    }

    if (attempt < maxAttempts) {
      await wait(delay);
    }
  }

  throw new Error(`Condition not met after ${maxAttempts} attempts`);
}

// ============================================================================
// Polling Utilities
// ============================================================================

export interface PollOptions {
  interval?: number;
  timeout?: number;
  onPoll?: (attempt: number) => void;
}

/**
 * Poll until a condition is met
 */
export async function pollUntil<T>(
  fn: () => T | Promise<T>,
  condition: (result: T) => boolean,
  options: PollOptions = {}
): Promise<T> {
  const { interval = 100, timeout = 5000, onPoll } = options;
  const start = Date.now();
  let attempt = 0;

  while (Date.now() - start < timeout) {
    attempt++;
    onPoll?.(attempt);

    const result = await fn();
    if (condition(result)) {
      return result;
    }

    await wait(interval);
  }

  throw new Error(`Polling timed out after ${timeout}ms`);
}

/**
 * Wait for a value to become truthy
 */
export async function waitFor<T>(
  fn: () => T | Promise<T>,
  options: PollOptions = {}
): Promise<NonNullable<T>> {
  const result = await pollUntil(fn, (value) => Boolean(value), options);
  return result as NonNullable<T>;
}

/**
 * Wait for a value to equal expected
 */
export async function waitForValue<T>(
  fn: () => T | Promise<T>,
  expected: T,
  options: PollOptions = {}
): Promise<T> {
  return pollUntil(fn, (value) => value === expected, options);
}

// ============================================================================
// Timeout Utilities
// ============================================================================

/**
 * Wrap a promise with a timeout
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message = "Operation timed out"
): Promise<T> {
  let timeoutId: NodeJS.Timeout;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(message));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId!);
  }
}

/**
 * Race multiple promises with a fallback
 */
export async function raceWithFallback<T>(
  promises: Promise<T>[],
  fallback: T,
  timeout: number
): Promise<T> {
  try {
    return await withTimeout(Promise.race(promises), timeout);
  } catch {
    return fallback;
  }
}

// ============================================================================
// Concurrency Control
// ============================================================================

/**
 * Run promises in sequence
 */
export async function sequence<T>(fns: Array<() => Promise<T>>): Promise<T[]> {
  const results: T[] = [];
  for (const fn of fns) {
    results.push(await fn());
  }
  return results;
}

/**
 * Run promises with concurrency limit
 */
export async function parallel<T>(
  fns: Array<() => Promise<T>>,
  concurrency: number
): Promise<T[]> {
  const results: T[] = [];
  const running: Promise<void>[] = [];

  for (let i = 0; i < fns.length; i++) {
    const fn = fns[i];
    const promise = fn().then((result) => {
      results[i] = result;
    });

    running.push(promise);

    if (running.length >= concurrency) {
      await Promise.race(running);
      // Remove completed promises
      const completed = running.filter((p) =>
        Promise.race([p, Promise.resolve("pending")]).then(
          (r) => r !== "pending"
        )
      );
      for (const p of completed) {
        const index = running.indexOf(p);
        if (index > -1) running.splice(index, 1);
      }
    }
  }

  await Promise.all(running);
  return results;
}

// ============================================================================
// Event Waiting
// ============================================================================

/**
 * Wait for an event to be emitted
 */
export function waitForEvent<T = unknown>(
  emitter: {
    on: (event: string, handler: (data: T) => void) => void;
    off: (event: string, handler: (data: T) => void) => void;
  },
  event: string,
  options: { timeout?: number } = {}
): Promise<T> {
  const { timeout = 5000 } = options;

  return new Promise((resolve, reject) => {
    let timeoutId: NodeJS.Timeout;
    let handler: (data: T) => void;

    const cleanup = () => {
      clearTimeout(timeoutId);
      emitter.off(event, handler);
    };

    handler = (data: T) => {
      cleanup();
      resolve(data);
    };

    timeoutId = setTimeout(() => {
      cleanup();
      reject(new Error(`Timeout waiting for event: ${event}`));
    }, timeout);

    emitter.on(event, handler);
  });
}

/**
 * Wait for multiple events
 */
export async function waitForEvents<T = unknown>(
  emitter: {
    on: (event: string, handler: (data: T) => void) => void;
    off: (event: string, handler: (data: T) => void) => void;
  },
  events: string[],
  options: { timeout?: number } = {}
): Promise<Map<string, T>> {
  const results = new Map<string, T>();
  const promises = events.map(async (event) => {
    const data = await waitForEvent<T>(emitter, event, options);
    results.set(event, data);
  });

  await Promise.all(promises);
  return results;
}

// ============================================================================
// State Machines
// ============================================================================

/**
 * Wait for a state machine to reach a specific state
 */
export async function waitForState<S>(
  getState: () => S | Promise<S>,
  targetState: S,
  options: PollOptions = {}
): Promise<void> {
  await pollUntil(getState, (state) => state === targetState, options);
}

/**
 * Track state transitions
 */
export function createStateTracker<S>(): {
  transitions: S[];
  record: (state: S) => void;
  waitFor: (state: S, options?: PollOptions) => Promise<void>;
  reset: () => void;
} {
  const transitions: S[] = [];

  return {
    transitions,
    record(state: S) {
      transitions.push(state);
    },
    async waitFor(state: S, options: PollOptions = {}) {
      await pollUntil(
        () => transitions[transitions.length - 1],
        (current) => current === state,
        options
      );
    },
    reset() {
      transitions.length = 0;
    },
  };
}
