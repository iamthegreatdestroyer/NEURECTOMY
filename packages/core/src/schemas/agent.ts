import { z } from "zod";

/**
 * Schema for agent configuration validation.
 */
export const agentConfigSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1).max(256),
  type: z.enum(["ai", "tool", "composite", "workflow"]),
  version: z
    .string()
    .regex(/^\d+\.\d+\.\d+$/)
    .optional(),
  description: z.string().max(4096).optional(),
  capabilities: z.array(z.string()).optional(),
  parameters: z.record(z.unknown()).optional(),
  metadata: z.record(z.string()).optional(),
  enabled: z.boolean().default(true),
  timeout: z.number().positive().optional(),
  maxRetries: z.number().nonnegative().int().default(3),
  rateLimit: z
    .object({
      requests: z.number().positive().int(),
      windowMs: z.number().positive().int(),
    })
    .optional(),
});

/**
 * Schema for task definition validation.
 */
export const taskDefinitionSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1).max(256),
  description: z.string().max(4096).optional(),
  agentId: z.string(),
  input: z.record(z.unknown()).optional(),
  output: z.record(z.unknown()).optional(),
  dependencies: z.array(z.string()).optional(),
  priority: z.number().int().min(0).max(100).default(50),
  timeout: z.number().positive().optional(),
  retryPolicy: z
    .object({
      maxRetries: z.number().nonnegative().int(),
      backoffMs: z.number().positive().int(),
      exponential: z.boolean().default(true),
    })
    .optional(),
});

/**
 * Schema for workflow definition validation.
 */
export const workflowSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1).max(256),
  description: z.string().max(4096).optional(),
  version: z
    .string()
    .regex(/^\d+\.\d+\.\d+$/)
    .optional(),
  tasks: z.array(taskDefinitionSchema),
  triggers: z
    .array(
      z.object({
        type: z.enum(["manual", "schedule", "webhook", "event"]),
        config: z.record(z.unknown()),
      })
    )
    .optional(),
  variables: z.record(z.unknown()).optional(),
  metadata: z.record(z.string()).optional(),
});

// Type inference
export type AgentConfigInput = z.input<typeof agentConfigSchema>;
export type AgentConfigOutput = z.output<typeof agentConfigSchema>;
export type TaskDefinitionInput = z.input<typeof taskDefinitionSchema>;
export type TaskDefinitionOutput = z.output<typeof taskDefinitionSchema>;
export type WorkflowInput = z.input<typeof workflowSchema>;
export type WorkflowOutput = z.output<typeof workflowSchema>;
