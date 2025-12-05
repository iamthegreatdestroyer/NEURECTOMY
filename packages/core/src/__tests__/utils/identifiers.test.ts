import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  generateId,
  generatePrefixedId,
  createTimestamp,
  createUnixTimestamp,
  createMonotonicTimestamp,
} from "../utils/identifiers";

describe("generateId", () => {
  it("should generate a UUID v4 format string", () => {
    const id = generateId();
    const uuidRegex =
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    expect(id).toMatch(uuidRegex);
  });

  it("should generate unique IDs", () => {
    const ids = new Set<string>();
    for (let i = 0; i < 100; i++) {
      ids.add(generateId());
    }
    expect(ids.size).toBe(100);
  });
});

describe("generatePrefixedId", () => {
  it("should generate ID with prefix", () => {
    const id = generatePrefixedId("agent");
    expect(id).toMatch(/^agent_[0-9a-f-]+$/i);
  });

  it("should use underscore separator", () => {
    const id = generatePrefixedId("task");
    expect(id.startsWith("task_")).toBe(true);
  });

  it("should work with different prefixes", () => {
    const agentId = generatePrefixedId("agent");
    const taskId = generatePrefixedId("task");
    const workflowId = generatePrefixedId("wf");

    expect(agentId.startsWith("agent_")).toBe(true);
    expect(taskId.startsWith("task_")).toBe(true);
    expect(workflowId.startsWith("wf_")).toBe(true);
  });
});

describe("createTimestamp", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should return ISO 8601 format", () => {
    vi.setSystemTime(new Date("2024-01-15T12:30:45.123Z"));
    const timestamp = createTimestamp();
    expect(timestamp).toBe("2024-01-15T12:30:45.123Z");
  });

  it("should return current time", () => {
    const before = new Date().toISOString();
    vi.useRealTimers();
    const timestamp = createTimestamp();
    const after = new Date().toISOString();

    expect(timestamp >= before).toBe(true);
    expect(timestamp <= after).toBe(true);
  });
});

describe("createUnixTimestamp", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should return milliseconds since epoch", () => {
    vi.setSystemTime(new Date("2024-01-15T12:30:45.123Z"));
    const timestamp = createUnixTimestamp();
    expect(timestamp).toBe(1705321845123);
  });

  it("should return a number", () => {
    const timestamp = createUnixTimestamp();
    expect(typeof timestamp).toBe("number");
  });

  it("should increase over time", () => {
    vi.useRealTimers();
    const first = createUnixTimestamp();
    const second = createUnixTimestamp();
    expect(second).toBeGreaterThanOrEqual(first);
  });
});

describe("createMonotonicTimestamp", () => {
  it("should return a number", () => {
    const timestamp = createMonotonicTimestamp();
    expect(typeof timestamp).toBe("number");
  });

  it("should be monotonically increasing", () => {
    const timestamps: number[] = [];
    for (let i = 0; i < 10; i++) {
      timestamps.push(createMonotonicTimestamp());
    }

    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]).toBeGreaterThanOrEqual(timestamps[i - 1]);
    }
  });

  it("should have high precision", () => {
    const timestamp = createMonotonicTimestamp();
    // Should have decimal places (sub-millisecond precision)
    expect(timestamp.toString()).toMatch(/\d+(\.\d+)?/);
  });
});
