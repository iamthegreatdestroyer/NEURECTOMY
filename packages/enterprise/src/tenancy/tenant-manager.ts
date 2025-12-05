/**
 * @fileoverview Multi-Tenancy Management System
 * @module @neurectomy/enterprise/tenancy/tenant-manager
 *
 * Agent Assignment: @ARCHITECT (Systems) + @FORTRESS (Security)
 *
 * Implements enterprise multi-tenancy architecture:
 * - Tenant lifecycle management
 * - Resource isolation
 * - Feature flags and limits
 * - Subscription management
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";
import { createHash, randomBytes } from "crypto";
import { v4 as uuidv4 } from "uuid";

import type {
  Tenant,
  TenantSettings,
  TenantFeatures,
  TenantSubscription,
  TenantMetadata,
  TenantStatus,
  TenantIsolationLevel,
  SubscriptionTier,
  ResourceLimits,
  ResourceLimit,
  LimitEnforcement,
  BrandingSettings,
  SecuritySettings,
  NotificationSettings,
  IntegrationSettings,
  LocaleSettings,
  RateLimitConfig,
  PasswordPolicy,
  TenancyModuleConfig,
  EnterpriseEventType,
  EnterpriseEvent,
} from "../types.js";

// ============================================================================
// Tenant Events
// ============================================================================

/**
 * Tenant event types
 */
export type TenantEventType =
  | "tenant:created"
  | "tenant:updated"
  | "tenant:suspended"
  | "tenant:activated"
  | "tenant:deleted"
  | "tenant:migrating"
  | "subscription:updated"
  | "limits:warning"
  | "limits:exceeded"
  | "settings:updated"
  | "features:updated";

/**
 * Tenant event
 */
export interface TenantEvent<T = unknown> {
  type: TenantEventType;
  timestamp: Date;
  tenantId: string;
  data?: T;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Default tenancy configuration
 */
export const DEFAULT_TENANCY_CONFIG: TenancyModuleConfig = {
  defaultIsolationLevel: "shared",
  defaultSubscriptionTier: "free",
  defaultResourceLimits: {
    users: {
      limit: 5,
      used: 0,
      unit: "users",
      enforcement: "hard",
      resetPeriod: "never",
    },
    storage: {
      limit: 1073741824, // 1 GB
      used: 0,
      unit: "bytes",
      enforcement: "hard",
      resetPeriod: "never",
    },
    apiRequests: {
      limit: 10000,
      used: 0,
      unit: "requests",
      enforcement: "throttle",
      resetPeriod: "daily",
    },
    computeUnits: {
      limit: 1000,
      used: 0,
      unit: "units",
      enforcement: "soft",
      resetPeriod: "monthly",
    },
    bandwidth: {
      limit: 10737418240, // 10 GB
      used: 0,
      unit: "bytes",
      enforcement: "notify",
      resetPeriod: "monthly",
    },
    projects: {
      limit: 3,
      used: 0,
      unit: "projects",
      enforcement: "hard",
      resetPeriod: "never",
    },
    customLimits: {},
  },
  allowCustomDomains: false,
  domainVerificationRequired: true,
  autoProvisioningEnabled: true,
  trialDurationDays: 14,
};

/**
 * Subscription tier limits
 */
export const TIER_LIMITS: Record<SubscriptionTier, Partial<ResourceLimits>> = {
  free: {
    users: {
      limit: 5,
      used: 0,
      unit: "users",
      enforcement: "hard",
      resetPeriod: "never",
    },
    storage: {
      limit: 1073741824,
      used: 0,
      unit: "bytes",
      enforcement: "hard",
      resetPeriod: "never",
    },
    projects: {
      limit: 3,
      used: 0,
      unit: "projects",
      enforcement: "hard",
      resetPeriod: "never",
    },
    apiRequests: {
      limit: 10000,
      used: 0,
      unit: "requests",
      enforcement: "throttle",
      resetPeriod: "daily",
    },
  },
  starter: {
    users: {
      limit: 25,
      used: 0,
      unit: "users",
      enforcement: "hard",
      resetPeriod: "never",
    },
    storage: {
      limit: 10737418240,
      used: 0,
      unit: "bytes",
      enforcement: "hard",
      resetPeriod: "never",
    },
    projects: {
      limit: 10,
      used: 0,
      unit: "projects",
      enforcement: "hard",
      resetPeriod: "never",
    },
    apiRequests: {
      limit: 100000,
      used: 0,
      unit: "requests",
      enforcement: "throttle",
      resetPeriod: "daily",
    },
  },
  professional: {
    users: {
      limit: 100,
      used: 0,
      unit: "users",
      enforcement: "soft",
      resetPeriod: "never",
    },
    storage: {
      limit: 107374182400,
      used: 0,
      unit: "bytes",
      enforcement: "soft",
      resetPeriod: "never",
    },
    projects: {
      limit: 50,
      used: 0,
      unit: "projects",
      enforcement: "soft",
      resetPeriod: "never",
    },
    apiRequests: {
      limit: 1000000,
      used: 0,
      unit: "requests",
      enforcement: "soft",
      resetPeriod: "daily",
    },
  },
  enterprise: {
    users: {
      limit: -1,
      used: 0,
      unit: "users",
      enforcement: "notify",
      resetPeriod: "never",
    },
    storage: {
      limit: -1,
      used: 0,
      unit: "bytes",
      enforcement: "notify",
      resetPeriod: "never",
    },
    projects: {
      limit: -1,
      used: 0,
      unit: "projects",
      enforcement: "notify",
      resetPeriod: "never",
    },
    apiRequests: {
      limit: -1,
      used: 0,
      unit: "requests",
      enforcement: "notify",
      resetPeriod: "daily",
    },
  },
  custom: {},
};

/**
 * Default security settings
 */
export const DEFAULT_SECURITY_SETTINGS: SecuritySettings = {
  mfaRequired: false,
  mfaMethods: ["totp", "email"],
  passwordPolicy: {
    minLength: 8,
    maxLength: 128,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: false,
    preventReuse: 5,
    expirationDays: 0,
    lockoutAttempts: 5,
    lockoutDuration: 900000,
  },
  sessionTimeout: 3600000,
  maxConcurrentSessions: 5,
  ipWhitelist: [],
  ipBlacklist: [],
  allowedAuthProviders: ["local", "saml", "oauth2", "oidc"],
  apiRateLimits: {
    requestsPerMinute: 100,
    requestsPerHour: 3000,
    requestsPerDay: 50000,
    burstLimit: 50,
    throttleDelay: 1000,
  },
  dataRetentionDays: 365,
};

// ============================================================================
// Tenant Manager (@ARCHITECT @FORTRESS)
// ============================================================================

/**
 * Multi-tenancy management system
 *
 * @example
 * ```typescript
 * const tenantManager = new TenantManager(config);
 *
 * // Create tenant
 * const tenant = await tenantManager.createTenant({
 *   name: 'Acme Corp',
 *   domain: 'acme.com',
 *   subscription: { tier: 'professional' },
 * });
 *
 * // Check resource limits
 * const canCreate = await tenantManager.checkLimit(tenant.id, 'users', 1);
 *
 * // Update usage
 * await tenantManager.updateUsage(tenant.id, 'users', 1);
 * ```
 */
export class TenantManager extends EventEmitter {
  private config: TenancyModuleConfig;
  private tenants: Map<string, Tenant>;
  private tenantsByDomain: Map<string, string>;
  private tenantsBySlug: Map<string, string>;

  constructor(config: Partial<TenancyModuleConfig> = {}) {
    super();
    this.config = { ...DEFAULT_TENANCY_CONFIG, ...config };
    this.tenants = new Map();
    this.tenantsByDomain = new Map();
    this.tenantsBySlug = new Map();

    // Start periodic limit reset
    this.startLimitResetScheduler();
  }

  // ============================================================================
  // Tenant Lifecycle
  // ============================================================================

  /**
   * Create a new tenant
   */
  async createTenant(input: CreateTenantInput): Promise<Tenant> {
    // Validate name
    if (!input.name || input.name.length < 2) {
      throw new Error("Tenant name must be at least 2 characters");
    }

    // Generate slug
    const slug = input.slug || this.generateSlug(input.name);

    // Check for duplicate slug
    if (this.tenantsBySlug.has(slug)) {
      throw new Error(`Tenant slug '${slug}' already exists`);
    }

    // Check for duplicate domain
    if (input.domain) {
      if (this.tenantsByDomain.has(input.domain)) {
        throw new Error(`Domain '${input.domain}' already in use`);
      }
    }

    const tenantId = input.id || uuidv4();
    const now = new Date();

    // Determine subscription tier
    const tier =
      input.subscription?.tier || this.config.defaultSubscriptionTier;

    // Create tenant
    const tenant: Tenant = {
      id: tenantId,
      name: input.name,
      slug,
      domain: input.domain,
      customDomains: input.customDomains || [],
      status: input.status || "pending",
      isolationLevel: input.isolationLevel || this.config.defaultIsolationLevel,
      subscription: this.createSubscription(tier, input.subscription),
      settings: this.createSettings(input.settings),
      features: this.createFeatures(tier, input.features),
      resourceLimits: this.createResourceLimits(tier, input.resourceLimits),
      metadata: {
        createdAt: now,
        updatedAt: now,
        createdBy: input.createdBy || "system",
        industry: input.industry,
        companySize: input.companySize,
        country: input.country,
        tags: input.tags || [],
        notes: input.notes,
      },
    };

    // Store tenant
    this.tenants.set(tenantId, tenant);
    this.tenantsBySlug.set(slug, tenantId);

    if (tenant.domain) {
      this.tenantsByDomain.set(tenant.domain, tenantId);
    }

    for (const domain of tenant.customDomains) {
      this.tenantsByDomain.set(domain, tenantId);
    }

    this.emit("tenant:event", {
      type: "tenant:created",
      timestamp: now,
      tenantId,
      data: { name: tenant.name, tier },
    } as TenantEvent);

    return tenant;
  }

  /**
   * Update tenant
   */
  async updateTenant(
    tenantId: string,
    updates: UpdateTenantInput
  ): Promise<Tenant> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    const oldSlug = tenant.slug;
    const oldDomain = tenant.domain;
    const oldCustomDomains = [...tenant.customDomains];

    // Apply updates
    if (updates.name !== undefined) tenant.name = updates.name;
    if (updates.slug !== undefined) {
      if (this.tenantsBySlug.has(updates.slug) && updates.slug !== oldSlug) {
        throw new Error(`Slug '${updates.slug}' already exists`);
      }
      tenant.slug = updates.slug;
    }
    if (updates.domain !== undefined) {
      if (
        updates.domain &&
        this.tenantsByDomain.has(updates.domain) &&
        updates.domain !== oldDomain
      ) {
        throw new Error(`Domain '${updates.domain}' already in use`);
      }
      tenant.domain = updates.domain;
    }
    if (updates.customDomains !== undefined) {
      tenant.customDomains = updates.customDomains;
    }
    if (updates.isolationLevel !== undefined) {
      tenant.isolationLevel = updates.isolationLevel;
    }

    // Update settings
    if (updates.settings) {
      tenant.settings = {
        ...tenant.settings,
        ...updates.settings,
        branding: { ...tenant.settings.branding, ...updates.settings.branding },
        security: { ...tenant.settings.security, ...updates.settings.security },
        notifications: {
          ...tenant.settings.notifications,
          ...updates.settings.notifications,
        },
        integrations: {
          ...tenant.settings.integrations,
          ...updates.settings.integrations,
        },
        locale: { ...tenant.settings.locale, ...updates.settings.locale },
        custom: { ...tenant.settings.custom, ...updates.settings.custom },
      };
    }

    // Update features
    if (updates.features) {
      tenant.features = {
        ...tenant.features,
        flags: { ...tenant.features.flags, ...updates.features.flags },
        limits: { ...tenant.features.limits, ...updates.features.limits },
        experiments:
          updates.features.experiments || tenant.features.experiments,
        customFeatures: {
          ...tenant.features.customFeatures,
          ...updates.features.customFeatures,
        },
      };
    }

    // Update metadata
    tenant.metadata.updatedAt = new Date();
    if (updates.tags !== undefined) tenant.metadata.tags = updates.tags;
    if (updates.notes !== undefined) tenant.metadata.notes = updates.notes;

    // Update indexes
    if (tenant.slug !== oldSlug) {
      this.tenantsBySlug.delete(oldSlug);
      this.tenantsBySlug.set(tenant.slug, tenantId);
    }

    if (tenant.domain !== oldDomain) {
      if (oldDomain) this.tenantsByDomain.delete(oldDomain);
      if (tenant.domain) this.tenantsByDomain.set(tenant.domain, tenantId);
    }

    // Update custom domain indexes
    for (const domain of oldCustomDomains) {
      this.tenantsByDomain.delete(domain);
    }
    for (const domain of tenant.customDomains) {
      this.tenantsByDomain.set(domain, tenantId);
    }

    this.emit("tenant:event", {
      type: "tenant:updated",
      timestamp: new Date(),
      tenantId,
      data: updates,
    } as TenantEvent);

    return tenant;
  }

  /**
   * Suspend tenant
   */
  async suspendTenant(tenantId: string, reason?: string): Promise<void> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    tenant.status = "suspended";
    tenant.metadata.updatedAt = new Date();
    tenant.metadata.notes = reason
      ? `${tenant.metadata.notes || ""}\n[Suspended: ${reason}]`
      : tenant.metadata.notes;

    this.emit("tenant:event", {
      type: "tenant:suspended",
      timestamp: new Date(),
      tenantId,
      data: { reason },
    } as TenantEvent);
  }

  /**
   * Activate tenant
   */
  async activateTenant(tenantId: string): Promise<void> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    tenant.status = "active";
    tenant.metadata.updatedAt = new Date();

    this.emit("tenant:event", {
      type: "tenant:activated",
      timestamp: new Date(),
      tenantId,
    } as TenantEvent);
  }

  /**
   * Delete tenant
   */
  async deleteTenant(tenantId: string, hardDelete = false): Promise<void> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    if (hardDelete) {
      // Remove from all indexes
      this.tenants.delete(tenantId);
      this.tenantsBySlug.delete(tenant.slug);
      if (tenant.domain) {
        this.tenantsByDomain.delete(tenant.domain);
      }
      for (const domain of tenant.customDomains) {
        this.tenantsByDomain.delete(domain);
      }
    } else {
      // Soft delete
      tenant.status = "deactivated";
      tenant.metadata.updatedAt = new Date();
    }

    this.emit("tenant:event", {
      type: "tenant:deleted",
      timestamp: new Date(),
      tenantId,
      data: { hardDelete },
    } as TenantEvent);
  }

  // ============================================================================
  // Tenant Lookup
  // ============================================================================

  /**
   * Get tenant by ID
   */
  getTenant(tenantId: string): Tenant | undefined {
    return this.tenants.get(tenantId);
  }

  /**
   * Get tenant by slug
   */
  getTenantBySlug(slug: string): Tenant | undefined {
    const tenantId = this.tenantsBySlug.get(slug);
    return tenantId ? this.tenants.get(tenantId) : undefined;
  }

  /**
   * Get tenant by domain
   */
  getTenantByDomain(domain: string): Tenant | undefined {
    const tenantId = this.tenantsByDomain.get(domain);
    return tenantId ? this.tenants.get(tenantId) : undefined;
  }

  /**
   * List all tenants
   */
  listTenants(filter?: TenantFilter): Tenant[] {
    let tenants = Array.from(this.tenants.values());

    if (filter) {
      if (filter.status) {
        tenants = tenants.filter((t) => t.status === filter.status);
      }
      if (filter.tier) {
        tenants = tenants.filter((t) => t.subscription.tier === filter.tier);
      }
      if (filter.isolationLevel) {
        tenants = tenants.filter(
          (t) => t.isolationLevel === filter.isolationLevel
        );
      }
      if (filter.tags && filter.tags.length > 0) {
        tenants = tenants.filter((t) =>
          filter.tags!.some((tag) => t.metadata.tags.includes(tag))
        );
      }
    }

    return tenants;
  }

  // ============================================================================
  // Subscription Management
  // ============================================================================

  /**
   * Update subscription
   */
  async updateSubscription(
    tenantId: string,
    updates: Partial<TenantSubscription>
  ): Promise<TenantSubscription> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    const oldTier = tenant.subscription.tier;

    tenant.subscription = {
      ...tenant.subscription,
      ...updates,
    };

    // Update resource limits if tier changed
    if (updates.tier && updates.tier !== oldTier) {
      const tierLimits = TIER_LIMITS[updates.tier];
      if (tierLimits) {
        tenant.resourceLimits = this.createResourceLimits(
          updates.tier,
          tenant.resourceLimits
        );
      }
    }

    tenant.metadata.updatedAt = new Date();

    this.emit("tenant:event", {
      type: "subscription:updated",
      timestamp: new Date(),
      tenantId,
      data: { oldTier, newTier: tenant.subscription.tier },
    } as TenantEvent);

    return tenant.subscription;
  }

  // ============================================================================
  // Resource Limits
  // ============================================================================

  /**
   * Check if resource limit allows operation
   */
  checkLimit(
    tenantId: string,
    resource: string,
    amount = 1
  ): { allowed: boolean; current: number; limit: number; remaining: number } {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    const limit = tenant.resourceLimits[
      resource as keyof ResourceLimits
    ] as ResourceLimit;
    if (!limit) {
      // No limit defined, allow
      return { allowed: true, current: 0, limit: -1, remaining: -1 };
    }

    // Unlimited (-1)
    if (limit.limit === -1) {
      return { allowed: true, current: limit.used, limit: -1, remaining: -1 };
    }

    const remaining = limit.limit - limit.used;
    const allowed = limit.enforcement !== "hard" || remaining >= amount;

    return {
      allowed,
      current: limit.used,
      limit: limit.limit,
      remaining: Math.max(0, remaining),
    };
  }

  /**
   * Update resource usage
   */
  async updateUsage(
    tenantId: string,
    resource: string,
    delta: number
  ): Promise<ResourceLimit> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    let limit = tenant.resourceLimits[
      resource as keyof ResourceLimits
    ] as ResourceLimit;
    if (!limit) {
      // Create default limit
      limit = {
        limit: -1,
        used: 0,
        unit: resource,
        enforcement: "notify",
        resetPeriod: "never",
      };
      (tenant.resourceLimits as Record<string, ResourceLimit>)[resource] =
        limit;
    }

    limit.used = Math.max(0, limit.used + delta);

    // Check for warnings/exceeded
    if (limit.limit > 0) {
      const usagePercent = (limit.used / limit.limit) * 100;

      if (usagePercent >= 100) {
        this.emit("tenant:event", {
          type: "limits:exceeded",
          timestamp: new Date(),
          tenantId,
          data: { resource, used: limit.used, limit: limit.limit },
        } as TenantEvent);
      } else if (usagePercent >= 80) {
        this.emit("tenant:event", {
          type: "limits:warning",
          timestamp: new Date(),
          tenantId,
          data: {
            resource,
            used: limit.used,
            limit: limit.limit,
            percent: usagePercent,
          },
        } as TenantEvent);
      }
    }

    return limit;
  }

  /**
   * Reset usage for a resource
   */
  async resetUsage(tenantId: string, resource: string): Promise<void> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    const limit = tenant.resourceLimits[
      resource as keyof ResourceLimits
    ] as ResourceLimit;
    if (limit) {
      limit.used = 0;
    }
  }

  // ============================================================================
  // Feature Flags
  // ============================================================================

  /**
   * Check if feature is enabled
   */
  isFeatureEnabled(tenantId: string, feature: string): boolean {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      return false;
    }

    return tenant.features.flags[feature] === true;
  }

  /**
   * Set feature flag
   */
  async setFeatureFlag(
    tenantId: string,
    feature: string,
    enabled: boolean
  ): Promise<void> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    tenant.features.flags[feature] = enabled;
    tenant.metadata.updatedAt = new Date();

    this.emit("tenant:event", {
      type: "features:updated",
      timestamp: new Date(),
      tenantId,
      data: { feature, enabled },
    } as TenantEvent);
  }

  /**
   * Get feature limit
   */
  getFeatureLimit(tenantId: string, feature: string): number {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      return 0;
    }

    return tenant.features.limits[feature] ?? 0;
  }

  // ============================================================================
  // Settings Management
  // ============================================================================

  /**
   * Update tenant settings
   */
  async updateSettings(
    tenantId: string,
    settings: Partial<TenantSettings>
  ): Promise<TenantSettings> {
    const tenant = this.tenants.get(tenantId);
    if (!tenant) {
      throw new Error(`Tenant ${tenantId} not found`);
    }

    tenant.settings = {
      ...tenant.settings,
      ...settings,
      branding: { ...tenant.settings.branding, ...settings.branding },
      security: { ...tenant.settings.security, ...settings.security },
      notifications: {
        ...tenant.settings.notifications,
        ...settings.notifications,
      },
      integrations: {
        ...tenant.settings.integrations,
        ...settings.integrations,
      },
      locale: { ...tenant.settings.locale, ...settings.locale },
      custom: { ...tenant.settings.custom, ...settings.custom },
    };

    tenant.metadata.updatedAt = new Date();

    this.emit("tenant:event", {
      type: "settings:updated",
      timestamp: new Date(),
      tenantId,
      data: settings,
    } as TenantEvent);

    return tenant.settings;
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  /**
   * Generate URL-safe slug from name
   */
  private generateSlug(name: string): string {
    const base = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");

    // Check for uniqueness
    let slug = base;
    let counter = 1;
    while (this.tenantsBySlug.has(slug)) {
      slug = `${base}-${counter}`;
      counter++;
    }

    return slug;
  }

  /**
   * Create subscription with defaults
   */
  private createSubscription(
    tier: SubscriptionTier,
    input?: Partial<TenantSubscription>
  ): TenantSubscription {
    const now = new Date();
    const trialEndsAt = new Date(
      now.getTime() + this.config.trialDurationDays * 24 * 60 * 60 * 1000
    );

    return {
      tier,
      startDate: now,
      trialEndsAt: tier === "free" ? trialEndsAt : undefined,
      billingCycle: "monthly",
      status: tier === "free" ? "trialing" : "active",
      features: [],
      addons: [],
      ...input,
    };
  }

  /**
   * Create settings with defaults
   */
  private createSettings(input?: Partial<TenantSettings>): TenantSettings {
    return {
      branding: input?.branding || {},
      security: { ...DEFAULT_SECURITY_SETTINGS, ...input?.security },
      notifications: {
        emailEnabled: true,
        slackEnabled: false,
        webhookEnabled: false,
        alertEmails: [],
        digestFrequency: "daily",
        ...input?.notifications,
      },
      integrations: {
        ssoProviders: [],
        apiKeys: [],
        webhooks: [],
        connectors: [],
        ...input?.integrations,
      },
      locale: {
        language: "en",
        timezone: "UTC",
        dateFormat: "YYYY-MM-DD",
        timeFormat: "HH:mm:ss",
        currency: "USD",
        numberFormat: "en-US",
        ...input?.locale,
      },
      custom: input?.custom || {},
    };
  }

  /**
   * Create features with tier defaults
   */
  private createFeatures(
    tier: SubscriptionTier,
    input?: Partial<TenantFeatures>
  ): TenantFeatures {
    const tierFeatures: Record<string, boolean> = {
      free: false,
      starter: false,
      professional: true,
      enterprise: true,
      custom: true,
    };

    return {
      flags: {
        advancedAnalytics: tierFeatures[tier] || tier === "starter",
        customBranding: tier !== "free",
        ssoEnabled: tier === "professional" || tier === "enterprise",
        apiAccess: tier !== "free",
        auditLogs: tier === "professional" || tier === "enterprise",
        customIntegrations: tier === "enterprise",
        ...input?.flags,
      },
      limits: {
        ...input?.limits,
      },
      experiments: input?.experiments || [],
      customFeatures: input?.customFeatures || {},
    };
  }

  /**
   * Create resource limits with tier defaults
   */
  private createResourceLimits(
    tier: SubscriptionTier,
    input?: Partial<ResourceLimits>
  ): ResourceLimits {
    const tierDefaults = TIER_LIMITS[tier] || {};

    return {
      users: tierDefaults.users || this.config.defaultResourceLimits.users,
      storage:
        tierDefaults.storage || this.config.defaultResourceLimits.storage,
      apiRequests:
        tierDefaults.apiRequests ||
        this.config.defaultResourceLimits.apiRequests,
      computeUnits:
        tierDefaults.computeUnits ||
        this.config.defaultResourceLimits.computeUnits,
      bandwidth:
        tierDefaults.bandwidth || this.config.defaultResourceLimits.bandwidth,
      projects:
        tierDefaults.projects || this.config.defaultResourceLimits.projects,
      customLimits: input?.customLimits || {},
    };
  }

  /**
   * Start periodic limit reset scheduler
   */
  private startLimitResetScheduler(): void {
    // Check every hour
    setInterval(() => {
      const now = new Date();

      for (const tenant of this.tenants.values()) {
        for (const [key, limit] of Object.entries(tenant.resourceLimits)) {
          if (key === "customLimits") continue;

          const resourceLimit = limit as ResourceLimit;
          if (this.shouldResetLimit(resourceLimit, now)) {
            resourceLimit.used = 0;
          }
        }
      }
    }, 3600000); // 1 hour
  }

  /**
   * Check if limit should be reset
   */
  private shouldResetLimit(limit: ResourceLimit, now: Date): boolean {
    if (limit.resetPeriod === "never") return false;

    switch (limit.resetPeriod) {
      case "hourly":
        return now.getMinutes() === 0;
      case "daily":
        return now.getHours() === 0 && now.getMinutes() === 0;
      case "monthly":
        return (
          now.getDate() === 1 && now.getHours() === 0 && now.getMinutes() === 0
        );
      default:
        return false;
    }
  }

  /**
   * Get tenant statistics
   */
  getStats(): {
    totalTenants: number;
    byStatus: Record<TenantStatus, number>;
    byTier: Record<SubscriptionTier, number>;
    byIsolation: Record<TenantIsolationLevel, number>;
  } {
    const byStatus: Record<string, number> = {};
    const byTier: Record<string, number> = {};
    const byIsolation: Record<string, number> = {};

    for (const tenant of this.tenants.values()) {
      byStatus[tenant.status] = (byStatus[tenant.status] || 0) + 1;
      byTier[tenant.subscription.tier] =
        (byTier[tenant.subscription.tier] || 0) + 1;
      byIsolation[tenant.isolationLevel] =
        (byIsolation[tenant.isolationLevel] || 0) + 1;
    }

    return {
      totalTenants: this.tenants.size,
      byStatus: byStatus as Record<TenantStatus, number>,
      byTier: byTier as Record<SubscriptionTier, number>,
      byIsolation: byIsolation as Record<TenantIsolationLevel, number>,
    };
  }
}

// ============================================================================
// Input Types
// ============================================================================

/**
 * Create tenant input
 */
export interface CreateTenantInput {
  id?: string;
  name: string;
  slug?: string;
  domain?: string;
  customDomains?: string[];
  status?: TenantStatus;
  isolationLevel?: TenantIsolationLevel;
  subscription?: Partial<TenantSubscription>;
  settings?: Partial<TenantSettings>;
  features?: Partial<TenantFeatures>;
  resourceLimits?: Partial<ResourceLimits>;
  createdBy?: string;
  industry?: string;
  companySize?: string;
  country?: string;
  tags?: string[];
  notes?: string;
}

/**
 * Update tenant input
 */
export interface UpdateTenantInput {
  name?: string;
  slug?: string;
  domain?: string;
  customDomains?: string[];
  isolationLevel?: TenantIsolationLevel;
  settings?: Partial<TenantSettings>;
  features?: Partial<TenantFeatures>;
  tags?: string[];
  notes?: string;
}

/**
 * Tenant filter
 */
export interface TenantFilter {
  status?: TenantStatus;
  tier?: SubscriptionTier;
  isolationLevel?: TenantIsolationLevel;
  tags?: string[];
}
