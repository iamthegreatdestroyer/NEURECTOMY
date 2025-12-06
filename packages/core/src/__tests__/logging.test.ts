// ============================================================================
// NEURECTOMY Logging Module Tests
// ============================================================================

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  Logger,
  createLogger,
  createLoggerFromEnv,
  createContextualLogger,
  runWithLogContext,
  getLogContext,
  createHttpRequestLog,
  type LogEntry,
  type LogLevel,
  type LogContext,
} from "../logging";

describe("Logger", () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;
  let capturedLogs: LogEntry[];

  beforeEach(() => {
    capturedLogs = [];
    consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  const createTestLogger = (
    overrides: Partial<Parameters<typeof createLogger>[0]> = {}
  ) => {
    return createLogger({
      service: "test-service",
      level: "trace",
      prettyPrint: false,
      transport: (entry) => capturedLogs.push(entry),
      ...overrides,
    });
  };

  describe("basic logging", () => {
    it("should log info messages", () => {
      const logger = createTestLogger();
      logger.info("Test message");

      expect(capturedLogs).toHaveLength(1);
      expect(capturedLogs[0].level).toBe("info");
      expect(capturedLogs[0].message).toBe("Test message");
    });

    it("should log all levels", () => {
      const logger = createTestLogger();
      const levels: LogLevel[] = [
        "trace",
        "debug",
        "info",
        "warn",
        "error",
        "fatal",
      ];

      levels.forEach((level) => {
        logger[level](`${level} message`);
      });

      expect(capturedLogs).toHaveLength(6);
      levels.forEach((level, index) => {
        expect(capturedLogs[index].level).toBe(level);
      });
    });

    it("should include timestamp in ISO format", () => {
      const logger = createTestLogger();
      logger.info("Test");

      expect(capturedLogs[0].timestamp).toMatch(
        /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/
      );
    });

    it("should include service context", () => {
      const logger = createTestLogger({ service: "my-service" });
      logger.info("Test");

      expect(capturedLogs[0].context.service).toBe("my-service");
    });

    it("should include additional data", () => {
      const logger = createTestLogger();
      logger.info("Test", { userId: "123", action: "login" });

      expect(capturedLogs[0].data).toEqual({ userId: "123", action: "login" });
    });
  });

  describe("log level filtering", () => {
    it("should filter logs below configured level", () => {
      const logger = createTestLogger({ level: "warn" });

      logger.trace("trace");
      logger.debug("debug");
      logger.info("info");
      logger.warn("warn");
      logger.error("error");

      expect(capturedLogs).toHaveLength(2);
      expect(capturedLogs[0].level).toBe("warn");
      expect(capturedLogs[1].level).toBe("error");
    });

    it("should log all levels when set to trace", () => {
      const logger = createTestLogger({ level: "trace" });

      logger.trace("trace");
      logger.debug("debug");
      logger.info("info");
      logger.warn("warn");
      logger.error("error");
      logger.fatal("fatal");

      expect(capturedLogs).toHaveLength(6);
    });
  });

  describe("error logging", () => {
    it("should serialize Error objects", () => {
      const logger = createTestLogger();
      const error = new Error("Something went wrong");

      logger.error("Operation failed", error);

      expect(capturedLogs[0].error).toBeDefined();
      expect(capturedLogs[0].error?.name).toBe("Error");
      expect(capturedLogs[0].error?.message).toBe("Something went wrong");
      expect(capturedLogs[0].error?.stack).toBeDefined();
    });

    it("should handle error with code", () => {
      const logger = createTestLogger();
      const error = Object.assign(new Error("Not found"), { code: "ENOENT" });

      logger.error("File error", error);

      expect(capturedLogs[0].error?.code).toBe("ENOENT");
    });

    it("should handle fatal errors", () => {
      const logger = createTestLogger();
      const error = new Error("Critical failure");

      logger.fatal("System crash", error);

      expect(capturedLogs[0].level).toBe("fatal");
      expect(capturedLogs[0].error).toBeDefined();
    });

    it("should handle error with additional data", () => {
      const logger = createTestLogger();
      const error = new Error("Database error");

      logger.error("Query failed", error, { query: "SELECT * FROM users" });

      expect(capturedLogs[0].error).toBeDefined();
      expect(capturedLogs[0].data).toEqual({ query: "SELECT * FROM users" });
    });
  });

  describe("child loggers", () => {
    it("should create child with additional context", () => {
      const logger = createTestLogger();
      const childLogger = logger.child({ requestId: "req-123" });

      childLogger.info("Child message");

      expect(capturedLogs[0].context.requestId).toBe("req-123");
      expect(capturedLogs[0].context.service).toBe("test-service");
    });

    it("should inherit parent context", () => {
      const logger = createTestLogger();
      const child1 = logger.child({ userId: "user-1" });
      const child2 = child1.child({ sessionId: "session-1" });

      child2.info("Nested child");

      expect(capturedLogs[0].context.userId).toBe("user-1");
      expect(capturedLogs[0].context.sessionId).toBe("session-1");
    });

    it("should create child with correlation ID", () => {
      const logger = createTestLogger();
      const child = logger.withCorrelationId("corr-123");

      child.info("Correlated message");

      expect(capturedLogs[0].context.correlationId).toBe("corr-123");
    });

    it("should generate correlation ID if not provided", () => {
      const logger = createTestLogger();
      const child = logger.withCorrelationId();

      child.info("Auto correlated");

      expect(capturedLogs[0].context.correlationId).toBeDefined();
      expect(capturedLogs[0].context.correlationId).toHaveLength(36); // UUID format
    });

    it("should create child with trace context", () => {
      const logger = createTestLogger();
      const child = logger.withTraceContext("trace-123", "span-456");

      child.info("Traced message");

      expect(capturedLogs[0].context.traceId).toBe("trace-123");
      expect(capturedLogs[0].context.spanId).toBe("span-456");
    });

    it("should create request-scoped logger", () => {
      const logger = createTestLogger();
      const reqLogger = logger.forRequest("req-123", "user-456");

      reqLogger.info("Request log");

      expect(capturedLogs[0].context.requestId).toBe("req-123");
      expect(capturedLogs[0].context.userId).toBe("user-456");
      expect(capturedLogs[0].context.correlationId).toBeDefined();
    });
  });

  describe("sensitive data redaction", () => {
    it("should redact password fields", () => {
      const logger = createTestLogger();
      logger.info("Login attempt", { username: "john", password: "secret123" });

      expect(capturedLogs[0].data?.username).toBe("john");
      expect(capturedLogs[0].data?.password).toBe("[REDACTED]");
    });

    it("should redact nested sensitive fields", () => {
      const logger = createTestLogger();
      logger.info("Config", {
        database: {
          host: "localhost",
          password: "dbpass",
        },
      });

      const data = capturedLogs[0].data as Record<
        string,
        Record<string, string>
      >;
      expect(data.database.host).toBe("localhost");
      expect(data.database.password).toBe("[REDACTED]");
    });

    it("should redact common sensitive field names", () => {
      const logger = createTestLogger();
      logger.info("Secrets", {
        apiKey: "key123",
        api_key: "key456",
        token: "tok123",
        secret: "sec123",
        authorization: "Bearer xxx",
      });

      const data = capturedLogs[0].data;
      expect(data?.apiKey).toBe("[REDACTED]");
      expect(data?.api_key).toBe("[REDACTED]");
      expect(data?.token).toBe("[REDACTED]");
      expect(data?.secret).toBe("[REDACTED]");
      expect(data?.authorization).toBe("[REDACTED]");
    });

    it("should support custom redact fields", () => {
      const logger = createTestLogger({
        redactFields: ["customSecret", "myData"],
      });
      logger.info("Custom", {
        customSecret: "value",
        myData: "data",
        safe: "ok",
      });

      expect(capturedLogs[0].data?.customSecret).toBe("[REDACTED]");
      expect(capturedLogs[0].data?.myData).toBe("[REDACTED]");
      expect(capturedLogs[0].data?.safe).toBe("ok");
    });
  });

  describe("operation timing", () => {
    it("should time sync operations", () => {
      const logger = createTestLogger();
      const complete = logger.operationStart("database.query", {
        table: "users",
      });

      // Simulate work
      complete();

      expect(capturedLogs).toHaveLength(2);
      expect(capturedLogs[0].message).toBe("database.query.start");
      expect(capturedLogs[1].message).toBe("database.query.complete");
      expect(capturedLogs[1].data?.durationMs).toBeDefined();
    });

    it("should time async operations", async () => {
      const logger = createTestLogger();

      const result = await logger.time("async.operation", async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return "done";
      });

      expect(result).toBe("done");
      expect(capturedLogs).toHaveLength(2);
      expect(capturedLogs[1].data?.durationMs).toBeGreaterThanOrEqual(10);
    });

    it("should log errors in timed operations", async () => {
      const logger = createTestLogger();

      await expect(
        logger.time("failing.operation", async () => {
          throw new Error("Failed");
        })
      ).rejects.toThrow("Failed");

      expect(
        capturedLogs.some((log) => log.message === "failing.operation.error")
      ).toBe(true);
    });
  });

  describe("metrics logging", () => {
    it("should log metrics", () => {
      const logger = createTestLogger();
      logger.metric("request.duration", 150, { endpoint: "/api/users" });

      expect(capturedLogs[0].message).toBe("metric.request.duration");
      expect(capturedLogs[0].data?.metric).toEqual({
        name: "request.duration",
        value: 150,
        tags: { endpoint: "/api/users" },
      });
    });
  });
});

describe("Factory Functions", () => {
  it("createLoggerFromEnv should use environment variables", () => {
    const originalEnv = process.env.LOG_LEVEL;
    process.env.LOG_LEVEL = "debug";

    const logger = createLoggerFromEnv("env-test-service");

    // Just verify it creates without error
    expect(logger).toBeInstanceOf(Logger);

    process.env.LOG_LEVEL = originalEnv;
  });
});

describe("Async Local Storage Context", () => {
  it("should store and retrieve context", () => {
    const context: LogContext = { correlationId: "ctx-123", userId: "user-1" };

    runWithLogContext(context, () => {
      const retrieved = getLogContext();
      expect(retrieved).toEqual(context);
    });
  });

  it("should return undefined outside of context", () => {
    const context = getLogContext();
    expect(context).toBeUndefined();
  });

  it("should support nested contexts", () => {
    runWithLogContext({ correlationId: "outer" }, () => {
      expect(getLogContext()?.correlationId).toBe("outer");

      runWithLogContext({ correlationId: "inner" }, () => {
        expect(getLogContext()?.correlationId).toBe("inner");
      });

      expect(getLogContext()?.correlationId).toBe("outer");
    });
  });
});

describe("HTTP Request Logging", () => {
  it("should create HTTP request log entry", () => {
    const log = createHttpRequestLog("GET", "/api/users");

    expect(log.method).toBe("GET");
    expect(log.url).toBe("/api/users");
    expect(log.requestId).toBeDefined();
    expect(log.correlationId).toBeDefined();
  });

  it("should use provided IDs", () => {
    const log = createHttpRequestLog("POST", "/api/orders", {
      requestId: "req-custom",
      correlationId: "corr-custom",
    });

    expect(log.requestId).toBe("req-custom");
    expect(log.correlationId).toBe("corr-custom");
  });

  it("should include optional fields", () => {
    const log = createHttpRequestLog("GET", "/api/users", {
      statusCode: 200,
      durationMs: 45,
      userAgent: "Test/1.0",
      ip: "192.168.1.1",
    });

    expect(log.statusCode).toBe(200);
    expect(log.durationMs).toBe(45);
    expect(log.userAgent).toBe("Test/1.0");
    expect(log.ip).toBe("192.168.1.1");
  });
});

describe("Pretty Printing", () => {
  it("should pretty print in development mode", () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    const logger = createLogger({
      service: "test",
      level: "info",
      prettyPrint: true,
    });

    logger.info("Pretty message", { key: "value" });

    expect(consoleSpy).toHaveBeenCalled();
    // Pretty print includes ANSI colors
    const output = consoleSpy.mock.calls[0][0];
    expect(output).toContain("Pretty message");

    consoleSpy.mockRestore();
  });
});
