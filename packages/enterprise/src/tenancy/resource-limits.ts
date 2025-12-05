/**
 * @fileoverview Tenant Resource Limits & Quota Management
 * @module @neurectomy/enterprise/tenancy/resource-limits
 *
 * @description
 * Comprehensive resource limiting and quota enforcement:
 * - CPU/Memory/Storage quotas
 * - API rate limiting
 * - Concurrent connection limits
 * - Bandwidth throttling
 * - Cost-based limits
 *
 * @VELOCITY Performance-aware quota enforcement
 */

import { EventEmitter } from "events";
import type { ResourceLimits, ResourceUsage, Tenant } from "../types.js";

/**
 * Quota definition
 */
export interface Quota {
  /** Quota identifier */
  id: string;
  /** Tenant ID */
  tenantId: string;
  /** Resource type */
  resourceType: ResourceType;
  /** Maximum allowed */
  limit: number;
  /** Current usage */
  current: number;
  /** Unit of measurement */
  unit: string;
  /** Reset period */
  resetPeriod: "hourly" | "daily" | "weekly" | "monthly" | "never";
  /** Last reset timestamp */
  lastReset: Date;
  /** Whether quota is enforced */
  enforced: boolean;
}

/**
 * Resource types
 */
export type ResourceType =
  | "cpu"
  | "memory"
  | "storage"
  | "bandwidth"
  | "api_calls"
  | "connections"
  | "agents"
  | "experiments"
  | "models"
  | "cost";

/**
 * Usage record
 */
export interface UsageRecord {
  /** Record ID */
  id: string;
  /** Tenant ID */
  tenantId: string;
  /** Resource type */
  resourceType: ResourceType;
  /** Amount used */
  amount: number;
  /** Timestamp */
  timestamp: Date;
  /** Operation that used resource */
  operation: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Limit check result
 */
export interface LimitCheckResult {
  /** Whether operation is allowed */
  allowed: boolean;
  /** Quota that blocked (if any) */
  blockedBy?: Quota;
  /** Current usage percentage */
  usagePercent: number;
  /** Remaining quota */
  remaining: number;
  /** Warning if approaching limit */
  warning?: string;
}

/**
 * Throttle configuration
 */
export interface ThrottleConfig {
  /** Requests per window */
  requestsPerWindow: number;
  /** Window size in seconds */
  windowSizeSeconds: number;
  /** Burst allowance */
  burstSize: number;
  /** Retry-after header value */
  retryAfterSeconds: number;
}

/**
 * Resource Limiter
 *
 * Enforces resource quotas and limits for tenants
 * with sliding window rate limiting and burst handling.
 */
export class ResourceLimiter extends EventEmitter {
  private quotas: Map<string, Quota[]> = new Map();
  private usageHistory: UsageRecord[] = [];
  private rateLimitWindows: Map<string, number[]> = new Map();
  private initialized: boolean = false;

  constructor(private defaultLimits: ResourceLimits) {
    super();
  }

  /**
   * Initialize resource limiter
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    this.emit("initializing");

    // Start periodic reset checker
    this.startResetChecker();

    // Start usage aggregator
    this.startUsageAggregator();

    this.initialized = true;
    this.emit("initialized");
  }

  /**
   * Set quotas for tenant
   */
  async setQuotas(tenant: Tenant, limits: ResourceLimits): Promise<Quota[]> {
    const quotas: Quota[] = [];

    // Create quota for each resource type
    const resourceConfigs: Array<{
      type: ResourceType;
      limit: number;
      unit: string;
      resetPeriod: Quota["resetPeriod"];
    }> = [
      {
        type: "cpu",
        limit: limits.maxCPU || 100,
        unit: "percent",
        resetPeriod: "never",
      },
      {
        type: "memory",
        limit: limits.maxMemory || 4096,
        unit: "MB",
        resetPeriod: "never",
      },
      {
        type: "storage",
        limit: limits.maxStorage || 10240,
        unit: "MB",
        resetPeriod: "monthly",
      },
      {
        type: "api_calls",
        limit: limits.maxApiCalls || 10000,
        unit: "calls",
        resetPeriod: "daily",
      },
      {
        type: "connections",
        limit: limits.maxConnections || 100,
        unit: "connections",
        resetPeriod: "never",
      },
      {
        type: "agents",
        limit: limits.maxAgents || 10,
        unit: "agents",
        resetPeriod: "never",
      },
      {
        type: "experiments",
        limit: limits.maxExperiments || 50,
        unit: "experiments",
        resetPeriod: "monthly",
      },
    ];

    for (const config of resourceConfigs) {
      const quota: Quota = {
        id: this.generateQuotaId(),
        tenantId: tenant.id,
        resourceType: config.type,
        limit: config.limit,
        current: 0,
        unit: config.unit,
        resetPeriod: config.resetPeriod,
        lastReset: new Date(),
        enforced: true,
      };
      quotas.push(quota);
    }

    this.quotas.set(tenant.id, quotas);
    this.emit("quotas:set", { tenant, quotas });

    return quotas;
  }

  /**
   * Check if operation is within limits
   */
  async checkLimit(
    tenantId: string,
    resourceType: ResourceType,
    requestedAmount: number = 1
  ): Promise<LimitCheckResult> {
    const quotas = this.quotas.get(tenantId) || [];
    const quota = quotas.find((q) => q.resourceType === resourceType);

    if (!quota) {
      // No quota defined - use default limits
      return {
        allowed: true,
        usagePercent: 0,
        remaining: this.getDefaultLimit(resourceType),
      };
    }

    if (!quota.enforced) {
      return {
        allowed: true,
        usagePercent: (quota.current / quota.limit) * 100,
        remaining: quota.limit - quota.current,
      };
    }

    const newUsage = quota.current + requestedAmount;
    const usagePercent = (newUsage / quota.limit) * 100;
    const remaining = Math.max(0, quota.limit - quota.current);

    // Check if would exceed
    if (newUsage > quota.limit) {
      this.emit("limit:exceeded", {
        tenantId,
        quota,
        requested: requestedAmount,
      });

      return {
        allowed: false,
        blockedBy: quota,
        usagePercent: Math.min(100, usagePercent),
        remaining: 0,
      };
    }

    // Warning at 80% usage
    const warning =
      usagePercent >= 80
        ? `Approaching ${resourceType} limit: ${usagePercent.toFixed(1)}% used`
        : undefined;

    if (warning) {
      this.emit("limit:warning", { tenantId, quota, usagePercent });
    }

    return {
      allowed: true,
      usagePercent,
      remaining: remaining - requestedAmount,
      warning,
    };
  }

  /**
   * Record resource usage
   */
  async recordUsage(
    tenantId: string,
    resourceType: ResourceType,
    amount: number,
    operation: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    // Update quota
    const quotas = this.quotas.get(tenantId) || [];
    const quota = quotas.find((q) => q.resourceType === resourceType);

    if (quota) {
      quota.current += amount;
    }

    // Record usage
    const record: UsageRecord = {
      id: this.generateRecordId(),
      tenantId,
      resourceType,
      amount,
      timestamp: new Date(),
      operation,
      metadata,
    };

    this.usageHistory.push(record);
    this.emit("usage:recorded", record);

    // Keep history bounded
    if (this.usageHistory.length > 100000) {
      this.usageHistory = this.usageHistory.slice(-50000);
    }
  }

  /**
   * Check rate limit (sliding window)
   */
  checkRateLimit(
    tenantId: string,
    endpoint: string,
    config: ThrottleConfig
  ): { allowed: boolean; retryAfter?: number } {
    const key = `${tenantId}:${endpoint}`;
    const now = Date.now();
    const windowStart = now - config.windowSizeSeconds * 1000;

    // Get request timestamps in window
    let timestamps = this.rateLimitWindows.get(key) || [];

    // Remove old timestamps
    timestamps = timestamps.filter((t) => t > windowStart);

    // Check against limit
    if (timestamps.length >= config.requestsPerWindow) {
      const oldestInWindow = timestamps[0];
      const retryAfter = Math.ceil(
        (oldestInWindow + config.windowSizeSeconds * 1000 - now) / 1000
      );

      this.emit("ratelimit:exceeded", { tenantId, endpoint, retryAfter });

      return {
        allowed: false,
        retryAfter: Math.max(retryAfter, config.retryAfterSeconds),
      };
    }

    // Allow burst
    if (timestamps.length < config.burstSize) {
      timestamps.push(now);
      this.rateLimitWindows.set(key, timestamps);
      return { allowed: true };
    }

    // Check sliding window average
    const windowRequests = timestamps.length;
    const expectedRate = config.requestsPerWindow / config.windowSizeSeconds;
    const currentRate = windowRequests / config.windowSizeSeconds;

    if (currentRate > expectedRate * 1.5) {
      this.emit("ratelimit:throttled", { tenantId, endpoint });
      return {
        allowed: false,
        retryAfter: config.retryAfterSeconds,
      };
    }

    timestamps.push(now);
    this.rateLimitWindows.set(key, timestamps);
    return { allowed: true };
  }

  /**
   * Get usage summary for tenant
   */
  getUsageSummary(tenantId: string): UsageSummary {
    const quotas = this.quotas.get(tenantId) || [];
    const tenantUsage = this.usageHistory.filter(
      (r) => r.tenantId === tenantId
    );

    const byResource: Record<ResourceType, ResourceUsageSummary> = {} as any;

    for (const quota of quotas) {
      const resourceUsage = tenantUsage.filter(
        (r) => r.resourceType === quota.resourceType
      );

      byResource[quota.resourceType] = {
        current: quota.current,
        limit: quota.limit,
        unit: quota.unit,
        usagePercent: (quota.current / quota.limit) * 100,
        remaining: Math.max(0, quota.limit - quota.current),
        resetPeriod: quota.resetPeriod,
        lastReset: quota.lastReset,
        recordCount: resourceUsage.length,
      };
    }

    return {
      tenantId,
      quotas: quotas.length,
      byResource,
      totalRecords: tenantUsage.length,
      generatedAt: new Date(),
    };
  }

  /**
   * Get usage trends for forecasting
   */
  getUsageTrends(
    tenantId: string,
    resourceType: ResourceType,
    periodDays: number = 30
  ): UsageTrend[] {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - periodDays);

    const records = this.usageHistory.filter(
      (r) =>
        r.tenantId === tenantId &&
        r.resourceType === resourceType &&
        r.timestamp > cutoff
    );

    // Group by day
    const byDay = new Map<string, number>();

    for (const record of records) {
      const day = record.timestamp.toISOString().split("T")[0];
      byDay.set(day, (byDay.get(day) || 0) + record.amount);
    }

    const trends: UsageTrend[] = [];
    for (const [date, total] of byDay) {
      trends.push({ date, total });
    }

    return trends.sort((a, b) => a.date.localeCompare(b.date));
  }

  /**
   * Predict quota exhaustion
   */
  predictExhaustion(
    tenantId: string,
    resourceType: ResourceType
  ): ExhaustionPrediction | null {
    const quotas = this.quotas.get(tenantId) || [];
    const quota = quotas.find((q) => q.resourceType === resourceType);

    if (!quota) return null;

    const trends = this.getUsageTrends(tenantId, resourceType, 7);
    if (trends.length < 3) return null;

    // Calculate average daily usage
    const totalUsage = trends.reduce((sum, t) => sum + t.total, 0);
    const avgDaily = totalUsage / trends.length;

    if (avgDaily <= 0) return null;

    const remaining = quota.limit - quota.current;
    const daysUntilExhaustion = remaining / avgDaily;

    const exhaustionDate = new Date();
    exhaustionDate.setDate(exhaustionDate.getDate() + daysUntilExhaustion);

    return {
      resourceType,
      currentUsage: quota.current,
      limit: quota.limit,
      avgDailyUsage: avgDaily,
      daysUntilExhaustion: Math.max(0, daysUntilExhaustion),
      predictedExhaustionDate: exhaustionDate,
      confidence: trends.length >= 7 ? "high" : "medium",
    };
  }

  /**
   * Reset quota for tenant
   */
  async resetQuota(
    tenantId: string,
    resourceType?: ResourceType
  ): Promise<void> {
    const quotas = this.quotas.get(tenantId) || [];

    for (const quota of quotas) {
      if (resourceType && quota.resourceType !== resourceType) continue;

      quota.current = 0;
      quota.lastReset = new Date();
    }

    this.emit("quota:reset", { tenantId, resourceType });
  }

  /**
   * Remove tenant quotas
   */
  async removeTenant(tenantId: string): Promise<void> {
    this.quotas.delete(tenantId);

    // Clean up rate limit windows
    for (const key of this.rateLimitWindows.keys()) {
      if (key.startsWith(`${tenantId}:`)) {
        this.rateLimitWindows.delete(key);
      }
    }

    this.emit("tenant:removed", { tenantId });
  }

  /**
   * Shutdown limiter
   */
  async shutdown(): Promise<void> {
    this.emit("shutting-down");
    this.initialized = false;
    this.emit("shutdown");
  }

  // Private methods

  private startResetChecker(): void {
    // Check for quota resets every hour
    setInterval(
      () => {
        this.checkAndResetQuotas();
      },
      60 * 60 * 1000
    );
  }

  private startUsageAggregator(): void {
    // Aggregate usage stats every 5 minutes
    setInterval(
      () => {
        this.aggregateUsage();
      },
      5 * 60 * 1000
    );
  }

  private checkAndResetQuotas(): void {
    const now = new Date();

    for (const [tenantId, quotas] of this.quotas) {
      for (const quota of quotas) {
        if (quota.resetPeriod === "never") continue;

        const shouldReset = this.shouldResetQuota(quota, now);
        if (shouldReset) {
          quota.current = 0;
          quota.lastReset = now;
          this.emit("quota:auto-reset", { tenantId, quota });
        }
      }
    }
  }

  private shouldResetQuota(quota: Quota, now: Date): boolean {
    const lastReset = quota.lastReset;
    const elapsed = now.getTime() - lastReset.getTime();
    const hour = 60 * 60 * 1000;
    const day = 24 * hour;
    const week = 7 * day;

    switch (quota.resetPeriod) {
      case "hourly":
        return elapsed >= hour;
      case "daily":
        return elapsed >= day;
      case "weekly":
        return elapsed >= week;
      case "monthly":
        return now.getMonth() !== lastReset.getMonth();
      default:
        return false;
    }
  }

  private aggregateUsage(): void {
    // Aggregate and emit metrics
    for (const [tenantId] of this.quotas) {
      const summary = this.getUsageSummary(tenantId);
      this.emit("usage:aggregated", summary);
    }
  }

  private getDefaultLimit(resourceType: ResourceType): number {
    const defaults: Record<ResourceType, number> = {
      cpu: 100,
      memory: 4096,
      storage: 10240,
      bandwidth: 1024 * 1024,
      api_calls: 10000,
      connections: 100,
      agents: 10,
      experiments: 50,
      models: 20,
      cost: 1000,
    };
    return defaults[resourceType] || 1000;
  }

  private generateQuotaId(): string {
    return `quota_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateRecordId(): string {
    return `usage_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }
}

/**
 * Usage summary types
 */
export interface UsageSummary {
  tenantId: string;
  quotas: number;
  byResource: Record<ResourceType, ResourceUsageSummary>;
  totalRecords: number;
  generatedAt: Date;
}

export interface ResourceUsageSummary {
  current: number;
  limit: number;
  unit: string;
  usagePercent: number;
  remaining: number;
  resetPeriod: Quota["resetPeriod"];
  lastReset: Date;
  recordCount: number;
}

export interface UsageTrend {
  date: string;
  total: number;
}

export interface ExhaustionPrediction {
  resourceType: ResourceType;
  currentUsage: number;
  limit: number;
  avgDailyUsage: number;
  daysUntilExhaustion: number;
  predictedExhaustionDate: Date;
  confidence: "low" | "medium" | "high";
}

/**
 * Factory function
 */
export function createResourceLimiter(
  defaultLimits: ResourceLimits
): ResourceLimiter {
  return new ResourceLimiter(defaultLimits);
}

export default ResourceLimiter;
