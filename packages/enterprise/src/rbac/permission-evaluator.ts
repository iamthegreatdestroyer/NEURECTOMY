/**
 * @fileoverview Permission Evaluator Engine
 * @module @neurectomy/enterprise/rbac/permission-evaluator
 *
 * @description
 * High-performance permission evaluation with:
 * - Policy-based access control (PBAC)
 * - Attribute-based access control (ABAC)
 * - Contextual evaluation
 * - Caching and optimization
 *
 * @VELOCITY Optimized permission checking
 */

import { EventEmitter } from "events";
import type {
  Permission,
  Role,
  PolicyRule,
  PolicyCondition,
} from "../types.js";

/**
 * Evaluation context
 */
export interface EvaluationContext {
  /** Principal making the request */
  principal: {
    id: string;
    type: "user" | "group" | "service";
    roles: string[];
    attributes: Record<string, unknown>;
  };
  /** Resource being accessed */
  resource: {
    type: string;
    id: string;
    owner?: string;
    attributes: Record<string, unknown>;
  };
  /** Action being performed */
  action: string;
  /** Environment context */
  environment: {
    timestamp: Date;
    ipAddress?: string;
    userAgent?: string;
    tenantId: string;
    attributes: Record<string, unknown>;
  };
}

/**
 * Evaluation result
 */
export interface EvaluationResult {
  /** Whether access is allowed */
  allowed: boolean;
  /** Reason for decision */
  reason: string;
  /** Matching policy (if any) */
  matchedPolicy?: PolicyRule;
  /** Evaluated policies */
  evaluatedPolicies: string[];
  /** Evaluation time (ms) */
  evaluationTimeMs: number;
  /** Cached result */
  cached: boolean;
}

/**
 * Permission pattern
 */
export interface PermissionPattern {
  /** Resource type pattern */
  resourceType: string;
  /** Action pattern */
  action: string;
  /** Scope (optional) */
  scope?: string;
}

/**
 * Permission Evaluator
 *
 * Evaluates access requests against policies with
 * support for PBAC and ABAC patterns.
 */
export class PermissionEvaluator extends EventEmitter {
  private policies: Map<string, PolicyRule> = new Map();
  private cache: Map<string, EvaluationResult> = new Map();
  private cacheMaxSize: number = 10000;
  private cacheTTL: number = 60000; // 1 minute
  private initialized: boolean = false;

  constructor(private config: PermissionEvaluatorConfig) {
    super();
    this.cacheMaxSize = config.cacheMaxSize || 10000;
    this.cacheTTL = config.cacheTTL || 60000;
  }

  /**
   * Initialize evaluator
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    this.emit("initializing");

    // Load default policies
    await this.loadDefaultPolicies();

    // Start cache cleanup
    this.startCacheCleanup();

    this.initialized = true;
    this.emit("initialized");
  }

  /**
   * Evaluate access request
   */
  async evaluate(context: EvaluationContext): Promise<EvaluationResult> {
    const startTime = performance.now();
    const cacheKey = this.getCacheKey(context);

    // Check cache
    const cached = this.cache.get(cacheKey);
    if (cached && !this.isCacheExpired(cached)) {
      this.emit("evaluation:cached", { context, result: cached });
      return { ...cached, cached: true };
    }

    const evaluatedPolicies: string[] = [];
    let result: EvaluationResult;

    try {
      // Get applicable policies
      const applicablePolicies = this.getApplicablePolicies(context);

      // Evaluate policies in order
      for (const policy of applicablePolicies) {
        evaluatedPolicies.push(policy.id);

        const match = await this.evaluatePolicy(policy, context);

        if (match.matched) {
          result = {
            allowed: policy.effect === "allow",
            reason: match.reason,
            matchedPolicy: policy,
            evaluatedPolicies,
            evaluationTimeMs: performance.now() - startTime,
            cached: false,
          };

          // Cache result
          this.cacheResult(cacheKey, result);

          this.emit("evaluation:complete", { context, result });
          return result;
        }
      }

      // No matching policy - default deny
      result = {
        allowed: false,
        reason: "No matching policy found - default deny",
        evaluatedPolicies,
        evaluationTimeMs: performance.now() - startTime,
        cached: false,
      };
    } catch (error) {
      result = {
        allowed: false,
        reason: `Evaluation error: ${error instanceof Error ? error.message : "Unknown error"}`,
        evaluatedPolicies,
        evaluationTimeMs: performance.now() - startTime,
        cached: false,
      };
    }

    // Cache result
    this.cacheResult(cacheKey, result);

    this.emit("evaluation:complete", { context, result });
    return result;
  }

  /**
   * Add policy
   */
  addPolicy(policy: PolicyRule): void {
    this.policies.set(policy.id, policy);
    this.invalidateCache();
    this.emit("policy:added", policy);
  }

  /**
   * Remove policy
   */
  removePolicy(policyId: string): void {
    this.policies.delete(policyId);
    this.invalidateCache();
    this.emit("policy:removed", { policyId });
  }

  /**
   * Update policy
   */
  updatePolicy(policyId: string, updates: Partial<PolicyRule>): void {
    const policy = this.policies.get(policyId);
    if (!policy) {
      throw new Error(`Policy not found: ${policyId}`);
    }

    Object.assign(policy, updates);
    this.invalidateCache();
    this.emit("policy:updated", policy);
  }

  /**
   * Get policy by ID
   */
  getPolicy(policyId: string): PolicyRule | undefined {
    return this.policies.get(policyId);
  }

  /**
   * Get all policies
   */
  getAllPolicies(): PolicyRule[] {
    return Array.from(this.policies.values());
  }

  /**
   * Check simple permission
   */
  checkPermission(permissions: Set<string>, required: string): boolean {
    // Exact match
    if (permissions.has(required)) return true;

    // Wildcard matching
    const parts = required.split(":");

    // Check progressively broader wildcards
    for (let i = parts.length - 1; i >= 0; i--) {
      const pattern = [...parts.slice(0, i), "*"].join(":");
      if (permissions.has(pattern)) return true;
    }

    // Global wildcard
    return permissions.has("*");
  }

  /**
   * Parse permission string
   */
  parsePermission(permission: string): PermissionPattern {
    const parts = permission.split(":");

    if (parts.length === 1) {
      return {
        resourceType: parts[0],
        action: "*",
      };
    }

    if (parts.length === 2) {
      return {
        resourceType: parts[0],
        action: parts[1],
      };
    }

    return {
      resourceType: parts[0],
      action: parts[1],
      scope: parts.slice(2).join(":"),
    };
  }

  /**
   * Build permission string
   */
  buildPermission(pattern: PermissionPattern): string {
    let permission = pattern.resourceType;

    if (pattern.action) {
      permission += `:${pattern.action}`;
    }

    if (pattern.scope) {
      permission += `:${pattern.scope}`;
    }

    return permission;
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): CacheStats {
    return {
      size: this.cache.size,
      maxSize: this.cacheMaxSize,
      hitRate: this.calculateHitRate(),
    };
  }

  /**
   * Invalidate cache
   */
  invalidateCache(): void {
    this.cache.clear();
    this.emit("cache:invalidated");
  }

  /**
   * Shutdown evaluator
   */
  async shutdown(): Promise<void> {
    this.emit("shutting-down");
    this.cache.clear();
    this.initialized = false;
    this.emit("shutdown");
  }

  // Private methods

  private async loadDefaultPolicies(): Promise<void> {
    const defaultPolicies: PolicyRule[] = [
      // Super admin - allow all
      {
        id: "super_admin_allow_all",
        name: "Super Admin Full Access",
        effect: "allow",
        principals: ["role:super_admin"],
        resources: ["*"],
        actions: ["*"],
        priority: 1000,
      },
      // Tenant admin - tenant scope
      {
        id: "tenant_admin_tenant_access",
        name: "Tenant Admin Access",
        effect: "allow",
        principals: ["role:tenant_admin"],
        resources: ["tenant:${context.environment.tenantId}/*"],
        actions: ["*"],
        priority: 900,
        conditions: [
          {
            type: "attribute",
            attribute: "environment.tenantId",
            operator: "equals",
            value: "${principal.attributes.tenantId}",
          },
        ],
      },
      // Resource owner access
      {
        id: "owner_access",
        name: "Resource Owner Access",
        effect: "allow",
        principals: ["*"],
        resources: ["*"],
        actions: ["read", "update", "delete"],
        priority: 800,
        conditions: [
          {
            type: "attribute",
            attribute: "resource.owner",
            operator: "equals",
            value: "${principal.id}",
          },
        ],
      },
      // Default deny
      {
        id: "default_deny",
        name: "Default Deny",
        effect: "deny",
        principals: ["*"],
        resources: ["*"],
        actions: ["*"],
        priority: 0,
      },
    ];

    for (const policy of defaultPolicies) {
      this.policies.set(policy.id, policy);
    }
  }

  private getApplicablePolicies(context: EvaluationContext): PolicyRule[] {
    const applicable: PolicyRule[] = [];

    for (const policy of this.policies.values()) {
      if (this.isPolicyApplicable(policy, context)) {
        applicable.push(policy);
      }
    }

    // Sort by priority (higher first)
    return applicable.sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }

  private isPolicyApplicable(
    policy: PolicyRule,
    context: EvaluationContext
  ): boolean {
    // Check principal match
    if (!this.matchesPrincipal(policy.principals, context)) {
      return false;
    }

    // Check resource match
    if (!this.matchesResource(policy.resources, context)) {
      return false;
    }

    // Check action match
    if (!this.matchesAction(policy.actions, context)) {
      return false;
    }

    return true;
  }

  private matchesPrincipal(
    principals: string[],
    context: EvaluationContext
  ): boolean {
    for (const principal of principals) {
      if (principal === "*") return true;

      if (principal.startsWith("role:")) {
        const role = principal.substring(5);
        if (context.principal.roles.includes(role)) return true;
      }

      if (principal.startsWith("user:")) {
        const userId = principal.substring(5);
        if (context.principal.id === userId) return true;
      }

      if (principal === context.principal.id) return true;
    }

    return false;
  }

  private matchesResource(
    resources: string[],
    context: EvaluationContext
  ): boolean {
    const resourcePath = `${context.resource.type}:${context.resource.id}`;

    for (const resource of resources) {
      if (resource === "*") return true;

      // Template substitution
      const pattern = this.substituteTemplate(resource, context);

      if (this.matchesPattern(resourcePath, pattern)) return true;
    }

    return false;
  }

  private matchesAction(
    actions: string[],
    context: EvaluationContext
  ): boolean {
    for (const action of actions) {
      if (action === "*") return true;
      if (action === context.action) return true;
    }

    return false;
  }

  private async evaluatePolicy(
    policy: PolicyRule,
    context: EvaluationContext
  ): Promise<{ matched: boolean; reason: string }> {
    // Evaluate conditions if any
    if (policy.conditions && policy.conditions.length > 0) {
      for (const condition of policy.conditions) {
        const conditionMet = this.evaluateCondition(condition, context);
        if (!conditionMet) {
          return {
            matched: false,
            reason: `Condition not met: ${condition.attribute}`,
          };
        }
      }
    }

    return {
      matched: true,
      reason: `Matched policy: ${policy.name}`,
    };
  }

  private evaluateCondition(
    condition: PolicyCondition,
    context: EvaluationContext
  ): boolean {
    // Get attribute value
    const actualValue = this.getAttributeValue(condition.attribute, context);

    // Get expected value (with template substitution)
    const expectedValue = this.substituteTemplate(
      String(condition.value),
      context
    );

    // Evaluate operator
    switch (condition.operator) {
      case "equals":
        return String(actualValue) === expectedValue;
      case "not_equals":
        return String(actualValue) !== expectedValue;
      case "contains":
        return String(actualValue).includes(expectedValue);
      case "starts_with":
        return String(actualValue).startsWith(expectedValue);
      case "ends_with":
        return String(actualValue).endsWith(expectedValue);
      case "in":
        const values = expectedValue.split(",").map((v) => v.trim());
        return values.includes(String(actualValue));
      case "not_in":
        const notValues = expectedValue.split(",").map((v) => v.trim());
        return !notValues.includes(String(actualValue));
      case "greater_than":
        return Number(actualValue) > Number(expectedValue);
      case "less_than":
        return Number(actualValue) < Number(expectedValue);
      case "exists":
        return actualValue !== undefined && actualValue !== null;
      case "not_exists":
        return actualValue === undefined || actualValue === null;
      default:
        return false;
    }
  }

  private getAttributeValue(path: string, context: EvaluationContext): unknown {
    const parts = path.split(".");
    let value: unknown = context;

    for (const part of parts) {
      if (value && typeof value === "object") {
        value = (value as Record<string, unknown>)[part];
      } else {
        return undefined;
      }
    }

    return value;
  }

  private substituteTemplate(
    template: string,
    context: EvaluationContext
  ): string {
    return template.replace(/\$\{([^}]+)\}/g, (_, path) => {
      const value = this.getAttributeValue(path, context);
      return value !== undefined ? String(value) : "";
    });
  }

  private matchesPattern(value: string, pattern: string): boolean {
    // Convert pattern to regex
    const regexPattern = pattern.replace(/\*/g, ".*").replace(/\?/g, ".");

    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(value);
  }

  private getCacheKey(context: EvaluationContext): string {
    return JSON.stringify({
      principal: context.principal.id,
      roles: context.principal.roles.sort(),
      resource: `${context.resource.type}:${context.resource.id}`,
      action: context.action,
      tenant: context.environment.tenantId,
    });
  }

  private cacheResult(key: string, result: EvaluationResult): void {
    // Evict if at capacity
    if (this.cache.size >= this.cacheMaxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }

    // Store with timestamp
    (result as any)._cachedAt = Date.now();
    this.cache.set(key, result);
  }

  private isCacheExpired(result: EvaluationResult): boolean {
    const cachedAt = (result as any)._cachedAt || 0;
    return Date.now() - cachedAt > this.cacheTTL;
  }

  private startCacheCleanup(): void {
    // Cleanup expired cache entries every minute
    setInterval(() => {
      for (const [key, result] of this.cache) {
        if (this.isCacheExpired(result)) {
          this.cache.delete(key);
        }
      }
    }, 60000);
  }

  private cacheHits: number = 0;
  private cacheMisses: number = 0;

  private calculateHitRate(): number {
    const total = this.cacheHits + this.cacheMisses;
    return total > 0 ? this.cacheHits / total : 0;
  }
}

/**
 * Configuration
 */
export interface PermissionEvaluatorConfig {
  cacheMaxSize?: number;
  cacheTTL?: number;
  defaultDeny?: boolean;
}

/**
 * Cache statistics
 */
export interface CacheStats {
  size: number;
  maxSize: number;
  hitRate: number;
}

/**
 * Factory function
 */
export function createPermissionEvaluator(
  config: PermissionEvaluatorConfig
): PermissionEvaluator {
  return new PermissionEvaluator(config);
}

export default PermissionEvaluator;
