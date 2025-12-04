import { v4 as uuidv4 } from "uuid";

/**
 * Generate a unique identifier using UUID v4.
 */
export function generateId(): string {
  return uuidv4();
}

/**
 * Generate a prefixed identifier for specific entity types.
 */
export function generatePrefixedId(prefix: string): string {
  return `${prefix}_${uuidv4().replace(/-/g, "")}`;
}

/**
 * Create an ISO timestamp for the current moment.
 */
export function createTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Create a Unix timestamp in milliseconds.
 */
export function createUnixTimestamp(): number {
  return Date.now();
}

/**
 * Create a monotonic timestamp using performance.now() if available.
 */
export function createMonotonicTimestamp(): number {
  if (typeof performance !== "undefined" && performance.now) {
    return performance.now();
  }
  return Date.now();
}
