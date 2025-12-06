// ============================================================================
// NEURECTOMY Structured Logging Module
// Provides production-grade logging with correlation IDs, log levels,
// and structured output compatible with Loki/ELK
// ============================================================================

import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types & Interfaces
// ============================================================================

export type LogLevel = "trace" | "debug" | "info" | "warn" | "error" | "fatal";

export interface LogContext {
  /** Correlation ID for request tracing */
  correlationId?: string;
  /** Trace ID from distributed tracing (OpenTelemetry) */
  traceId?: string;
  /** Span ID from distributed tracing */
  spanId?: string;
  /** Service name */
  service?: string;
  /** Environment (development, staging, production) */
  environment?: string;
  /** User ID if authenticated */
  userId?: string;
  /** Request ID */
  requestId?: string;
  /** Additional arbitrary context */
  [key: string]: unknown;
}

export interface LogEntry {
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Log level */
  level: LogLevel;
  /** Log message */
  message: string;
  /** Structured context */
  context: LogContext;
  /** Error object if present */
  error?: {
    name: string;
    message: string;
    stack?: string;
    code?: string;
  };
  /** Additional data payload */
  data?: Record<string, unknown>;
}

export interface LoggerConfig {
  /** Service name for identification */
  service: string;
  /** Minimum log level to output */
  level: LogLevel;
  /** Environment name */
  environment?: string;
  /** Whether to pretty print (development) or use JSON (production) */
  prettyPrint?: boolean;
  /** Custom transport function */
  transport?: (entry: LogEntry) => void;
  /** Redact sensitive fields */
  redactFields?: string[];
}

// ============================================================================
// Log Level Utilities
// ============================================================================

const LOG_LEVEL_VALUES: Record<LogLevel, number> = {
  trace: 10,
  debug: 20,
  info: 30,
  warn: 40,
  error: 50,
  fatal: 60,
};

const LOG_LEVEL_COLORS: Record<LogLevel, string> = {
  trace: "\x1b[90m", // Gray
  debug: "\x1b[36m", // Cyan
  info: "\x1b[32m", // Green
  warn: "\x1b[33m", // Yellow
  error: "\x1b[31m", // Red
  fatal: "\x1b[35m", // Magenta
};

const RESET_COLOR = "\x1b[0m";

function shouldLog(configuredLevel: LogLevel, messageLevel: LogLevel): boolean {
  return LOG_LEVEL_VALUES[messageLevel] >= LOG_LEVEL_VALUES[configuredLevel];
}

// ============================================================================
// Sensitive Data Redaction
// ============================================================================

const DEFAULT_REDACT_FIELDS = [
  "password",
  "secret",
  "token",
  "apiKey",
  "api_key",
  "authorization",
  "cookie",
  "creditCard",
  "credit_card",
  "ssn",
  "privateKey",
  "private_key",
];

function redactSensitiveData(
  obj: Record<string, unknown>,
  redactFields: string[]
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    const lowerKey = key.toLowerCase();
    const shouldRedact = redactFields.some((field) =>
      lowerKey.includes(field.toLowerCase())
    );

    if (shouldRedact) {
      result[key] = "[REDACTED]";
    } else if (value && typeof value === "object" && !Array.isArray(value)) {
      result[key] = redactSensitiveData(
        value as Record<string, unknown>,
        redactFields
      );
    } else {
      result[key] = value;
    }
  }

  return result;
}

// ============================================================================
// Error Serialization
// ============================================================================

function serializeError(error: Error): LogEntry["error"] {
  return {
    name: error.name,
    message: error.message,
    stack: error.stack,
    code: (error as Error & { code?: string }).code,
  };
}

// ============================================================================
// Logger Class
// ============================================================================

export class Logger {
  private config: Required<LoggerConfig>;
  private baseContext: LogContext;

  constructor(config: LoggerConfig) {
    this.config = {
      service: config.service,
      level: config.level,
      environment: config.environment ?? process.env.NODE_ENV ?? "development",
      prettyPrint: config.prettyPrint ?? process.env.NODE_ENV !== "production",
      transport: config.transport ?? this.defaultTransport.bind(this),
      redactFields: config.redactFields ?? DEFAULT_REDACT_FIELDS,
    };

    this.baseContext = {
      service: this.config.service,
      environment: this.config.environment,
    };
  }

  /**
   * Create a child logger with additional context
   */
  child(context: LogContext): Logger {
    const childLogger = new Logger(this.config);
    childLogger.baseContext = { ...this.baseContext, ...context };
    return childLogger;
  }

  /**
   * Create a child logger with a new correlation ID
   */
  withCorrelationId(correlationId?: string): Logger {
    return this.child({ correlationId: correlationId ?? uuidv4() });
  }

  /**
   * Create a child logger with trace context
   */
  withTraceContext(traceId: string, spanId?: string): Logger {
    return this.child({ traceId, spanId });
  }

  /**
   * Create a child logger for a specific request
   */
  forRequest(requestId?: string, userId?: string): Logger {
    return this.child({
      requestId: requestId ?? uuidv4(),
      userId,
      correlationId: uuidv4(),
    });
  }

  // Log level methods
  trace(message: string, data?: Record<string, unknown>): void {
    this.log("trace", message, data);
  }

  debug(message: string, data?: Record<string, unknown>): void {
    this.log("debug", message, data);
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.log("info", message, data);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.log("warn", message, data);
  }

  error(
    message: string,
    error?: Error | Record<string, unknown>,
    data?: Record<string, unknown>
  ): void {
    if (error instanceof Error) {
      this.logWithError("error", message, error, data);
    } else {
      this.log("error", message, error);
    }
  }

  fatal(
    message: string,
    error?: Error | Record<string, unknown>,
    data?: Record<string, unknown>
  ): void {
    if (error instanceof Error) {
      this.logWithError("fatal", message, error, data);
    } else {
      this.log("fatal", message, error);
    }
  }

  /**
   * Log a metric observation
   */
  metric(name: string, value: number, tags?: Record<string, string>): void {
    this.info(`metric.${name}`, {
      metric: { name, value, tags },
      type: "metric",
    });
  }

  /**
   * Log the start of an operation
   */
  operationStart(
    operation: string,
    data?: Record<string, unknown>
  ): () => void {
    const startTime = Date.now();
    const operationId = uuidv4();

    this.debug(`${operation}.start`, { operationId, ...data });

    return () => {
      const duration = Date.now() - startTime;
      this.info(`${operation}.complete`, {
        operationId,
        durationMs: duration,
        ...data,
      });
    };
  }

  /**
   * Time an async operation
   */
  async time<T>(
    operation: string,
    fn: () => Promise<T>,
    data?: Record<string, unknown>
  ): Promise<T> {
    const complete = this.operationStart(operation, data);
    try {
      const result = await fn();
      complete();
      return result;
    } catch (error) {
      this.error(`${operation}.error`, error as Error, data);
      throw error;
    }
  }

  // Private methods
  private log(
    level: LogLevel,
    message: string,
    data?: Record<string, unknown>
  ): void {
    if (!shouldLog(this.config.level, level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context: this.baseContext,
      data: data
        ? redactSensitiveData(data, this.config.redactFields)
        : undefined,
    };

    this.config.transport(entry);
  }

  private logWithError(
    level: LogLevel,
    message: string,
    error: Error,
    data?: Record<string, unknown>
  ): void {
    if (!shouldLog(this.config.level, level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context: this.baseContext,
      error: serializeError(error),
      data: data
        ? redactSensitiveData(data, this.config.redactFields)
        : undefined,
    };

    this.config.transport(entry);
  }

  private defaultTransport(entry: LogEntry): void {
    if (this.config.prettyPrint) {
      this.prettyPrintEntry(entry);
    } else {
      console.log(JSON.stringify(entry));
    }
  }

  private prettyPrintEntry(entry: LogEntry): void {
    const color = LOG_LEVEL_COLORS[entry.level];
    const levelStr = entry.level.toUpperCase().padEnd(5);
    const contextStr = entry.context.correlationId
      ? ` [${entry.context.correlationId.slice(0, 8)}]`
      : "";

    const prefix = `${color}${entry.timestamp} ${levelStr}${RESET_COLOR}${contextStr}`;

    console.log(`${prefix} ${entry.message}`);

    if (entry.data) {
      console.log(
        `  ${JSON.stringify(entry.data, null, 2).replace(/\n/g, "\n  ")}`
      );
    }

    if (entry.error) {
      console.log(
        `  ${color}Error: ${entry.error.name}: ${entry.error.message}${RESET_COLOR}`
      );
      if (entry.error.stack) {
        console.log(`  ${entry.error.stack.split("\n").slice(1).join("\n  ")}`);
      }
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new logger instance
 */
export function createLogger(config: LoggerConfig): Logger {
  return new Logger(config);
}

/**
 * Create a logger from environment variables
 */
export function createLoggerFromEnv(service: string): Logger {
  return new Logger({
    service,
    level: (process.env.LOG_LEVEL as LogLevel) ?? "info",
    environment: process.env.NODE_ENV ?? "development",
    prettyPrint:
      process.env.LOG_PRETTY === "true" ||
      process.env.NODE_ENV !== "production",
  });
}

// ============================================================================
// Async Local Storage for Request Context
// ============================================================================

import { AsyncLocalStorage } from "node:async_hooks";

const asyncLocalStorage = new AsyncLocalStorage<LogContext>();

/**
 * Run a function with log context available via async local storage
 */
export function runWithLogContext<T>(context: LogContext, fn: () => T): T {
  return asyncLocalStorage.run(context, fn);
}

/**
 * Get the current log context from async local storage
 */
export function getLogContext(): LogContext | undefined {
  return asyncLocalStorage.getStore();
}

/**
 * Create a logger that automatically includes async local storage context
 */
export function createContextualLogger(config: LoggerConfig): Logger {
  const baseLogger = new Logger(config);

  return new Proxy(baseLogger, {
    get(target, prop) {
      const value = target[prop as keyof Logger];

      if (
        typeof value === "function" &&
        ["trace", "debug", "info", "warn", "error", "fatal"].includes(
          prop as string
        )
      ) {
        return (...args: unknown[]) => {
          const asyncContext = getLogContext();
          if (asyncContext) {
            const contextualLogger = target.child(asyncContext);
            return (contextualLogger[prop as keyof Logger] as Function)(
              ...args
            );
          }
          return (value as Function).apply(target, args);
        };
      }

      return value;
    },
  });
}

// ============================================================================
// HTTP Request Logging Middleware Types
// ============================================================================

export interface HttpRequestLog {
  method: string;
  url: string;
  statusCode?: number;
  durationMs?: number;
  userAgent?: string;
  ip?: string;
  requestId: string;
  correlationId: string;
}

/**
 * Create HTTP request log entry
 */
export function createHttpRequestLog(
  method: string,
  url: string,
  options?: Partial<HttpRequestLog>
): HttpRequestLog {
  return {
    method,
    url,
    requestId: options?.requestId ?? uuidv4(),
    correlationId: options?.correlationId ?? uuidv4(),
    ...options,
  };
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  Logger,
  createLogger,
  createLoggerFromEnv,
  createContextualLogger,
  runWithLogContext,
  getLogContext,
  createHttpRequestLog,
};
