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

// Tenant Manager - export only what exists
export {
  TenantManager,
  DEFAULT_TENANCY_CONFIG,
  TIER_LIMITS,
  DEFAULT_SECURITY_SETTINGS,
  type TenantEventType,
  type TenantEvent,
  type CreateTenantInput,
  type UpdateTenantInput,
  type TenantFilter,
} from "./tenant-manager.js";

// Tenant Isolation - export only what exists
export {
  TenantIsolationEngine,
  createTenantIsolationEngine,
  type IsolationBoundary,
  type AccessDecision,
  type DataClassification,
  type IsolationMetrics,
} from "./tenant-isolation.js";

// Resource Limits - export only what exists
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

// Re-export types that actually exist in types.ts
export type {
  Tenant,
  TenantStatus,
  TenantIsolationLevel,
  SubscriptionTier,
  ResourceLimits,
  ResourceLimit,
  LimitEnforcement,
  TenantSettings,
  TenantFeatures,
  TenantSubscription,
  TenantMetadata,
  BrandingSettings,
  SecuritySettings,
  NotificationSettings,
  IntegrationSettings,
  LocaleSettings,
  RateLimitConfig,
  PasswordPolicy,
  TenancyModuleConfig,
} from "../types.js";

/**
 * Create full tenancy system
 */
export interface TenancySystemConfig {
  defaultResourceLimits: import("../types.js").ResourceLimits;
  isolationLevel?: import("../types.js").TenantIsolationLevel;
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
    defaultResourceLimits: config.defaultResourceLimits,
  });

  const isolation = new TenantIsolationEngine({
    defaultLevel: config.isolationLevel ?? "shared",
    encryptionEnabled: true,
    auditEnabled: config.enableMetrics ?? false,
  });

  const limiter = new ResourceLimiter(config.defaultResourceLimits);

  // Initialize isolation and limiter
  await isolation.initialize();
  await limiter.initialize();

  return { manager, isolation, limiter };
}
