/**
 * @neurectomy/core - Core Utilities and Business Logic
 *
 * Provides essential utilities, validation schemas, and shared business logic
 * for the Neurectomy AI Agent Platform.
 */

// Export types from @neurectomy/types
export type {
  AgentConfig,
  AgentState,
  TaskDefinition,
  WorkflowDefinition,
} from "@neurectomy/types";

// Export utility functions
export { generateId, createTimestamp } from "./utils/identifiers";
export { deepMerge, pick, omit } from "./utils/objects";
export { debounce, throttle } from "./utils/functions";
export { formatDuration, formatBytes, formatNumber } from "./utils/formatters";

// Export validation schemas
export {
  agentConfigSchema,
  taskDefinitionSchema,
  workflowSchema,
} from "./schemas/agent";

// Export error handling
export {
  NeurectomyError,
  ValidationError,
  NetworkError,
  TimeoutError,
} from "./errors";

// Export constants
export { DEFAULT_TIMEOUT, MAX_RETRIES, API_VERSION } from "./constants";
