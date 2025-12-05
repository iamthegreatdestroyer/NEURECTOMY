import { describe, it, expect } from "vitest";
import {
  agentConfigSchema,
  taskDefinitionSchema,
  workflowSchema,
} from "../schemas/agent";

describe("agentConfigSchema", () => {
  it("should validate a complete agent config", () => {
    const config = {
      id: "agent-123",
      name: "TestAgent",
      type: "worker",
      version: "1.0.0",
      capabilities: ["process", "analyze"],
      metadata: { team: "core" },
    };
    const result = agentConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
  });

  it("should require id, name, type, and version", () => {
    const result = agentConfigSchema.safeParse({});
    expect(result.success).toBe(false);
    if (!result.success) {
      const fields = result.error.issues.map((i) => i.path[0]);
      expect(fields).toContain("id");
      expect(fields).toContain("name");
      expect(fields).toContain("type");
      expect(fields).toContain("version");
    }
  });

  it("should default capabilities to empty array", () => {
    const config = {
      id: "agent-1",
      name: "Test",
      type: "worker",
      version: "1.0.0",
    };
    const result = agentConfigSchema.parse(config);
    expect(result.capabilities).toEqual([]);
  });

  it("should reject invalid types", () => {
    const config = {
      id: 123, // should be string
      name: "Test",
      type: "worker",
      version: "1.0.0",
    };
    const result = agentConfigSchema.safeParse(config);
    expect(result.success).toBe(false);
  });
});

describe("taskDefinitionSchema", () => {
  it("should validate a complete task definition", () => {
    const task = {
      id: "task-123",
      agentId: "agent-456",
      type: "process",
      input: { data: "test" },
      dependencies: ["task-100"],
      priority: 5,
      retryPolicy: { maxRetries: 3, delay: 1000 },
    };
    const result = taskDefinitionSchema.safeParse(task);
    expect(result.success).toBe(true);
  });

  it("should require id, agentId, and type", () => {
    const result = taskDefinitionSchema.safeParse({});
    expect(result.success).toBe(false);
    if (!result.success) {
      const fields = result.error.issues.map((i) => i.path[0]);
      expect(fields).toContain("id");
      expect(fields).toContain("agentId");
      expect(fields).toContain("type");
    }
  });

  it("should default dependencies to empty array", () => {
    const task = {
      id: "task-1",
      agentId: "agent-1",
      type: "run",
    };
    const result = taskDefinitionSchema.parse(task);
    expect(result.dependencies).toEqual([]);
  });

  it("should default priority to 0", () => {
    const task = {
      id: "task-1",
      agentId: "agent-1",
      type: "run",
    };
    const result = taskDefinitionSchema.parse(task);
    expect(result.priority).toBe(0);
  });
});

describe("workflowSchema", () => {
  it("should validate a complete workflow", () => {
    const workflow = {
      id: "wf-123",
      name: "TestWorkflow",
      tasks: [
        { id: "task-1", agentId: "agent-1", type: "start" },
        {
          id: "task-2",
          agentId: "agent-2",
          type: "process",
          dependencies: ["task-1"],
        },
      ],
      triggers: [{ type: "schedule", cron: "0 * * * *" }],
      variables: { env: "production" },
    };
    const result = workflowSchema.safeParse(workflow);
    expect(result.success).toBe(true);
  });

  it("should require id, name, and tasks", () => {
    const result = workflowSchema.safeParse({});
    expect(result.success).toBe(false);
    if (!result.success) {
      const fields = result.error.issues.map((i) => i.path[0]);
      expect(fields).toContain("id");
      expect(fields).toContain("name");
      expect(fields).toContain("tasks");
    }
  });

  it("should require at least one task", () => {
    const workflow = {
      id: "wf-1",
      name: "Empty",
      tasks: [],
    };
    const result = workflowSchema.safeParse(workflow);
    expect(result.success).toBe(false);
  });

  it("should default triggers and variables", () => {
    const workflow = {
      id: "wf-1",
      name: "Basic",
      tasks: [{ id: "task-1", agentId: "agent-1", type: "run" }],
    };
    const result = workflowSchema.parse(workflow);
    expect(result.triggers).toEqual([]);
    expect(result.variables).toEqual({});
  });
});
