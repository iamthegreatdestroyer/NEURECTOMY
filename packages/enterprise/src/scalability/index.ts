/**
 * @neurectomy/enterprise - Scalability Module
 *
 * Provides enterprise-grade scalability infrastructure including:
 * - Database sharding with consistent hashing
 * - Automatic failover with health monitoring
 * - Horizontal scaling and load balancing
 *
 * @packageDocumentation
 */

// Re-export all types
export * from "./types.js";

// Re-export database sharding components
export {
  DatabaseShardingManager,
  createDatabaseShardingManager,
  createDefaultShardingConfig,
} from "./database-sharding.js";

// Re-export failover automation components
export {
  FailoverAutomationManager,
  createFailoverAutomationManager,
  createDefaultFailoverConfig,
} from "./failover-automation.js";
