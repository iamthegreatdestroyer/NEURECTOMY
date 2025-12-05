import { describe, it, expect } from "vitest";
import {
  NeurectomyError,
  ValidationError,
  NetworkError,
  TimeoutError,
  AgentError,
  WorkflowError,
} from "../errors";

describe("NeurectomyError", () => {
  it("should create error with message and code", () => {
    const error = new NeurectomyError("Test error", "TEST_ERROR");
    expect(error.message).toBe("Test error");
    expect(error.code).toBe("TEST_ERROR");
    expect(error.name).toBe("NeurectomyError");
  });

  it("should include timestamp", () => {
    const before = Date.now();
    const error = new NeurectomyError("Test", "TEST");
    const after = Date.now();
    expect(error.timestamp).toBeGreaterThanOrEqual(before);
    expect(error.timestamp).toBeLessThanOrEqual(after);
  });

  it("should accept optional context", () => {
    const context = { userId: "123", action: "create" };
    const error = new NeurectomyError("Test", "TEST", context);
    expect(error.context).toEqual(context);
  });

  it("should serialize to JSON correctly", () => {
    const error = new NeurectomyError("Test", "TEST", { key: "value" });
    const json = error.toJSON();
    expect(json).toMatchObject({
      name: "NeurectomyError",
      message: "Test",
      code: "TEST",
      context: { key: "value" },
    });
    expect(json.timestamp).toBeDefined();
  });

  it("should be instanceof Error", () => {
    const error = new NeurectomyError("Test", "TEST");
    expect(error).toBeInstanceOf(Error);
  });
});

describe("ValidationError", () => {
  it("should extend NeurectomyError", () => {
    const error = new ValidationError("Invalid input", []);
    expect(error).toBeInstanceOf(NeurectomyError);
    expect(error.name).toBe("ValidationError");
  });

  it("should store validation errors array", () => {
    const errors = [
      { field: "email", message: "Invalid email format" },
      { field: "name", message: "Name is required" },
    ];
    const error = new ValidationError("Validation failed", errors);
    expect(error.errors).toEqual(errors);
    expect(error.code).toBe("VALIDATION_ERROR");
  });

  it("should include errors in JSON output", () => {
    const errors = [{ field: "test", message: "error" }];
    const error = new ValidationError("Failed", errors);
    const json = error.toJSON();
    expect(json.errors).toEqual(errors);
  });
});

describe("NetworkError", () => {
  it("should extend NeurectomyError", () => {
    const error = new NetworkError("Network failed", 500, "https://api.test");
    expect(error).toBeInstanceOf(NeurectomyError);
    expect(error.name).toBe("NetworkError");
  });

  it("should store status code and URL", () => {
    const error = new NetworkError("Not found", 404, "https://api.test/users");
    expect(error.statusCode).toBe(404);
    expect(error.url).toBe("https://api.test/users");
    expect(error.code).toBe("NETWORK_ERROR");
  });

  it("should include network details in JSON", () => {
    const error = new NetworkError("Error", 503, "https://api.test");
    const json = error.toJSON();
    expect(json.statusCode).toBe(503);
    expect(json.url).toBe("https://api.test");
  });
});

describe("TimeoutError", () => {
  it("should extend NeurectomyError", () => {
    const error = new TimeoutError("Timeout", 5000, "fetchData");
    expect(error).toBeInstanceOf(NeurectomyError);
    expect(error.name).toBe("TimeoutError");
  });

  it("should store timeout duration and operation", () => {
    const error = new TimeoutError("Request timed out", 30000, "apiCall");
    expect(error.timeoutMs).toBe(30000);
    expect(error.operation).toBe("apiCall");
    expect(error.code).toBe("TIMEOUT_ERROR");
  });

  it("should include timeout details in JSON", () => {
    const error = new TimeoutError("Timeout", 10000, "query");
    const json = error.toJSON();
    expect(json.timeoutMs).toBe(10000);
    expect(json.operation).toBe("query");
  });
});

describe("AgentError", () => {
  it("should extend NeurectomyError", () => {
    const error = new AgentError("Agent failed", "agent-123");
    expect(error).toBeInstanceOf(NeurectomyError);
    expect(error.name).toBe("AgentError");
  });

  it("should store agent ID", () => {
    const error = new AgentError("Agent crashed", "agent-456");
    expect(error.agentId).toBe("agent-456");
    expect(error.code).toBe("AGENT_ERROR");
  });

  it("should include agent ID in JSON", () => {
    const error = new AgentError("Error", "agent-789");
    const json = error.toJSON();
    expect(json.agentId).toBe("agent-789");
  });
});

describe("WorkflowError", () => {
  it("should extend NeurectomyError", () => {
    const error = new WorkflowError("Workflow failed", "wf-123");
    expect(error).toBeInstanceOf(NeurectomyError);
    expect(error.name).toBe("WorkflowError");
  });

  it("should store workflow ID and optional task ID", () => {
    const error = new WorkflowError("Task failed", "wf-123", "task-456");
    expect(error.workflowId).toBe("wf-123");
    expect(error.taskId).toBe("task-456");
    expect(error.code).toBe("WORKFLOW_ERROR");
  });

  it("should work without task ID", () => {
    const error = new WorkflowError("Workflow error", "wf-789");
    expect(error.workflowId).toBe("wf-789");
    expect(error.taskId).toBeUndefined();
  });

  it("should include workflow details in JSON", () => {
    const error = new WorkflowError("Error", "wf-123", "task-456");
    const json = error.toJSON();
    expect(json.workflowId).toBe("wf-123");
    expect(json.taskId).toBe("task-456");
  });
});
