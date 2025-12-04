/**
 * Base error class for Neurectomy errors.
 */
export class NeurectomyError extends Error {
  public readonly code: string;
  public readonly timestamp: Date;
  public readonly context?: Record<string, unknown>;

  constructor(
    message: string,
    code: string,
    context?: Record<string, unknown>
  ) {
    super(message);
    this.name = "NeurectomyError";
    this.code = code;
    this.timestamp = new Date();
    this.context = context;

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, NeurectomyError);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      timestamp: this.timestamp.toISOString(),
      context: this.context,
      stack: this.stack,
    };
  }
}

/**
 * Validation error for schema/input validation failures.
 */
export class ValidationError extends NeurectomyError {
  public readonly errors: Array<{ path: string; message: string }>;

  constructor(
    message: string,
    errors: Array<{ path: string; message: string }>,
    context?: Record<string, unknown>
  ) {
    super(message, "VALIDATION_ERROR", context);
    this.name = "ValidationError";
    this.errors = errors;
  }
}

/**
 * Network error for HTTP/connection failures.
 */
export class NetworkError extends NeurectomyError {
  public readonly statusCode?: number;
  public readonly url?: string;

  constructor(
    message: string,
    statusCode?: number,
    url?: string,
    context?: Record<string, unknown>
  ) {
    super(message, "NETWORK_ERROR", context);
    this.name = "NetworkError";
    this.statusCode = statusCode;
    this.url = url;
  }
}

/**
 * Timeout error for operations that exceeded time limits.
 */
export class TimeoutError extends NeurectomyError {
  public readonly timeoutMs: number;
  public readonly operation: string;

  constructor(
    operation: string,
    timeoutMs: number,
    context?: Record<string, unknown>
  ) {
    super(
      `Operation "${operation}" timed out after ${timeoutMs}ms`,
      "TIMEOUT_ERROR",
      context
    );
    this.name = "TimeoutError";
    this.timeoutMs = timeoutMs;
    this.operation = operation;
  }
}

/**
 * Agent error for agent-specific failures.
 */
export class AgentError extends NeurectomyError {
  public readonly agentId: string;

  constructor(
    message: string,
    agentId: string,
    code: string = "AGENT_ERROR",
    context?: Record<string, unknown>
  ) {
    super(message, code, context);
    this.name = "AgentError";
    this.agentId = agentId;
  }
}

/**
 * Workflow error for workflow execution failures.
 */
export class WorkflowError extends NeurectomyError {
  public readonly workflowId: string;
  public readonly taskId?: string;

  constructor(
    message: string,
    workflowId: string,
    taskId?: string,
    context?: Record<string, unknown>
  ) {
    super(message, "WORKFLOW_ERROR", context);
    this.name = "WorkflowError";
    this.workflowId = workflowId;
    this.taskId = taskId;
  }
}
