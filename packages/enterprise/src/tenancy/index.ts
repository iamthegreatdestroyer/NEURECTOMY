/**
 * @fileoverview Multi-Tenancy Module
 * @module @neurectomy/enterprise/tenancy
 *
 * @description
 * Comprehensive multi-tenancy support with:
 * - Tenant lifecycle management
 * - Data isolation (schema, row-level, encryption)
 * - Resource quotas and limits
 * - Rate limiting and throttling
 *
 * @AEGIS Enterprise-grade tenant isolation
 */

// Tenant Manager
export {
  TenantManager,
  createTenantManager,
  type TenantLifecycleHooks,
  type TenantMetrics,
} from "./tenant-manager.js";

// Tenant Isolation
export {
  TenantIsolationEngine,
  createTenantIsolationEngine,
  type IsolationBoundary,
  type AccessDecision,
  type DataClassification,
  type IsolationMetrics,
} from "./tenant-isolation.js";

// Resource Limits
export {
  ResourceLimiter,
  createResourceLimiter,
  type Quota,
  type ResourceType,
  type UsageRecord,
  type LimitCheckResult,
  type ThrottleConfig,
  type UsageSummary,
  type ResourceUsageSummary,
  type UsageTrend,
  type ExhaustionPrediction,
} from "./resource-limits.js";

// Re-export types
export type {
  Tenant,
  TenantConfig,
  TenantStatus,
  TenantTier,
  ResourceLimits,
  ResourceUsage,
  TenantIsolationConfig,
  IsolationStrategy,
  IsolationViolation,
  DataAccessContext,
} from "../types.js";

/**
 * Create full tenancy system
 */
export interface TenancySystemConfig {
  defaultLimits: import("../types.js").ResourceLimits;
  isolationConfig: import("../types.js").TenantIsolationConfig;
  enableMetrics?: boolean;
}

export interface TenancySystem {
  manager: import("./tenant-manager.js").TenantManager;
  isolation: import("./tenant-isolation.js").TenantIsolationEngine;
  limiter: import("./resource-limits.js").ResourceLimiter;
}

export async function createTenancySystem(
  config: TenancySystemConfig
): Promise<TenancySystem> {
  const { TenantManager } = await import("./tenant-manager.js");
  const { TenantIsolationEngine } = await import("./tenant-isolation.js");
  const { ResourceLimiter } = await import("./resource-limits.js");

  const manager = new TenantManager({
    defaultLimits: config.defaultLimits,
    enableMetrics: config.enableMetrics,
  });

  const isolation = new TenantIsolationEngine(config.isolationConfig);
  const limiter = new ResourceLimiter(config.defaultLimits);

  // Wire up events
  manager.on("tenant:created", async ({ tenant }) => {
    await isolation.createBoundary(tenant, "row", {});
    await limiter.setQuotas(tenant, config.defaultLimits);
  });

  manager.on("tenant:deleted", async ({ tenantId }) => {
    await isolation.cleanupTenant(tenantId);
    await limiter.removeTenant(tenantId);
  });

  // Initialize
  await manager.initialize();
  await isolation.initialize();
  await limiter.initialize();

  return { manager, isolation, limiter };
}
