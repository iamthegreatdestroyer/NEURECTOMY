/**
 * Feature Flags Module
 *
 * Provides a feature flag system for gradual rollouts and A/B testing.
 * Supports multiple targeting strategies and percentage-based rollouts.
 */

import { generateId } from "./utils/identifiers";

// =============================================================================
// Types
// =============================================================================

/**
 * Feature flag definition.
 */
export interface FeatureFlag {
  /** Unique flag key (e.g., "new-dashboard", "experimental-agent-v2") */
  key: string;
  /** Human-readable name */
  name: string;
  /** Description of what the flag controls */
  description?: string;
  /** Whether the flag is enabled globally */
  enabled: boolean;
  /** Targeting rules (evaluated in order) */
  rules?: TargetingRule[];
  /** Default value when no rules match */
  defaultValue: FlagValue;
  /** Tags for organization */
  tags?: string[];
  /** Creation timestamp */
  createdAt: string;
  /** Last updated timestamp */
  updatedAt: string;
}

/**
 * Possible flag values.
 */
export type FlagValue = boolean | string | number | Record<string, unknown>;

/**
 * Targeting rule for conditional flag evaluation.
 */
export interface TargetingRule {
  /** Rule identifier */
  id: string;
  /** Rule name */
  name: string;
  /** Conditions that must all be true */
  conditions: RuleCondition[];
  /** Value to return if conditions match */
  value: FlagValue;
  /** Percentage of matching users to target (0-100) */
  percentage?: number;
}

/**
 * Condition for targeting rules.
 */
export interface RuleCondition {
  /** Attribute to check (e.g., "userId", "email", "plan") */
  attribute: string;
  /** Comparison operator */
  operator: ConditionOperator;
  /** Value(s) to compare against */
  value: string | number | boolean | string[];
}

/**
 * Supported condition operators.
 */
export type ConditionOperator =
  | "equals"
  | "notEquals"
  | "contains"
  | "notContains"
  | "startsWith"
  | "endsWith"
  | "in"
  | "notIn"
  | "greaterThan"
  | "lessThan"
  | "greaterThanOrEqual"
  | "lessThanOrEqual"
  | "matches"; // Regex

/**
 * User context for flag evaluation.
 */
export interface EvaluationContext {
  /** User identifier */
  userId?: string;
  /** Additional attributes for targeting */
  attributes?: Record<string, string | number | boolean>;
}

/**
 * Result of flag evaluation.
 */
export interface EvaluationResult<T extends FlagValue = FlagValue> {
  /** The resolved value */
  value: T;
  /** Which rule matched (if any) */
  matchedRule?: string;
  /** Reason for the result */
  reason: EvaluationReason;
}

/**
 * Reason codes for evaluation results.
 */
export type EvaluationReason =
  | "FLAG_DISABLED"
  | "DEFAULT_VALUE"
  | "RULE_MATCH"
  | "PERCENTAGE_EXCLUDED"
  | "FLAG_NOT_FOUND";

// =============================================================================
// Feature Flag Client
// =============================================================================

/**
 * Configuration for the feature flag client.
 */
export interface FeatureFlagClientConfig {
  /** Initial flags (for offline/static usage) */
  flags?: FeatureFlag[];
  /** Callback when flags are updated */
  onFlagUpdate?: (flag: FeatureFlag) => void;
  /** Default value when flag not found */
  defaultValue?: FlagValue;
}

/**
 * Feature flag client for evaluating flags.
 */
export class FeatureFlagClient {
  private flags: Map<string, FeatureFlag> = new Map();
  private onFlagUpdate?: (flag: FeatureFlag) => void;
  private defaultValue: FlagValue;

  constructor(config: FeatureFlagClientConfig = {}) {
    this.onFlagUpdate = config.onFlagUpdate;
    this.defaultValue = config.defaultValue ?? false;

    if (config.flags) {
      for (const flag of config.flags) {
        this.flags.set(flag.key, flag);
      }
    }
  }

  /**
   * Check if a boolean feature flag is enabled.
   */
  isEnabled(key: string, context?: EvaluationContext): boolean {
    const result = this.evaluate<boolean>(key, context);
    return result.value === true;
  }

  /**
   * Get the value of a feature flag.
   */
  getValue<T extends FlagValue>(key: string, context?: EvaluationContext): T {
    const result = this.evaluate<T>(key, context);
    return result.value;
  }

  /**
   * Evaluate a feature flag with full result.
   */
  evaluate<T extends FlagValue>(
    key: string,
    context?: EvaluationContext
  ): EvaluationResult<T> {
    const flag = this.flags.get(key);

    if (!flag) {
      return {
        value: this.defaultValue as T,
        reason: "FLAG_NOT_FOUND",
      };
    }

    if (!flag.enabled) {
      return {
        value: flag.defaultValue as T,
        reason: "FLAG_DISABLED",
      };
    }

    // Evaluate rules in order
    if (flag.rules) {
      for (const rule of flag.rules) {
        if (this.evaluateRule(rule, context)) {
          // Check percentage rollout
          if (rule.percentage !== undefined && rule.percentage < 100) {
            if (!this.isInPercentage(key, context?.userId, rule.percentage)) {
              continue; // Skip to next rule
            }
          }

          return {
            value: rule.value as T,
            matchedRule: rule.id,
            reason: "RULE_MATCH",
          };
        }
      }
    }

    return {
      value: flag.defaultValue as T,
      reason: "DEFAULT_VALUE",
    };
  }

  /**
   * Set or update a flag.
   */
  setFlag(flag: FeatureFlag): void {
    this.flags.set(flag.key, flag);
    this.onFlagUpdate?.(flag);
  }

  /**
   * Remove a flag.
   */
  removeFlag(key: string): boolean {
    return this.flags.delete(key);
  }

  /**
   * Get all flags.
   */
  getAllFlags(): FeatureFlag[] {
    return Array.from(this.flags.values());
  }

  /**
   * Load flags from an array.
   */
  loadFlags(flags: FeatureFlag[]): void {
    for (const flag of flags) {
      this.setFlag(flag);
    }
  }

  /**
   * Clear all flags.
   */
  clear(): void {
    this.flags.clear();
  }

  // =========================================================================
  // Private Methods
  // =========================================================================

  private evaluateRule(
    rule: TargetingRule,
    context?: EvaluationContext
  ): boolean {
    if (!rule.conditions.length) {
      return true; // No conditions = always match
    }

    return rule.conditions.every((condition) =>
      this.evaluateCondition(condition, context)
    );
  }

  private evaluateCondition(
    condition: RuleCondition,
    context?: EvaluationContext
  ): boolean {
    const attributeValue = this.getAttributeValue(condition.attribute, context);

    if (attributeValue === undefined) {
      return false;
    }

    const { operator, value } = condition;

    switch (operator) {
      case "equals":
        return attributeValue === value;

      case "notEquals":
        return attributeValue !== value;

      case "contains":
        return String(attributeValue).includes(String(value));

      case "notContains":
        return !String(attributeValue).includes(String(value));

      case "startsWith":
        return String(attributeValue).startsWith(String(value));

      case "endsWith":
        return String(attributeValue).endsWith(String(value));

      case "in":
        return Array.isArray(value) && value.includes(attributeValue as string);

      case "notIn":
        return (
          Array.isArray(value) && !value.includes(attributeValue as string)
        );

      case "greaterThan":
        return Number(attributeValue) > Number(value);

      case "lessThan":
        return Number(attributeValue) < Number(value);

      case "greaterThanOrEqual":
        return Number(attributeValue) >= Number(value);

      case "lessThanOrEqual":
        return Number(attributeValue) <= Number(value);

      case "matches":
        try {
          const regex = new RegExp(String(value));
          return regex.test(String(attributeValue));
        } catch {
          return false;
        }

      default:
        return false;
    }
  }

  private getAttributeValue(
    attribute: string,
    context?: EvaluationContext
  ): string | number | boolean | undefined {
    if (!context) {
      return undefined;
    }

    // Special handling for userId
    if (attribute === "userId") {
      return context.userId;
    }

    return context.attributes?.[attribute];
  }

  /**
   * Determine if a user is in the percentage bucket.
   * Uses consistent hashing for stable results.
   */
  private isInPercentage(
    flagKey: string,
    userId: string | undefined,
    percentage: number
  ): boolean {
    if (!userId) {
      // No user ID = random bucket
      return Math.random() * 100 < percentage;
    }

    // Simple hash for consistent bucketing
    const hashInput = `${flagKey}:${userId}`;
    let hash = 0;
    for (let i = 0; i < hashInput.length; i++) {
      const char = hashInput.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Normalize to 0-100
    const bucket = Math.abs(hash % 100);
    return bucket < percentage;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new feature flag.
 */
export function createFlag(
  key: string,
  options: Partial<Omit<FeatureFlag, "key" | "createdAt" | "updatedAt">> = {}
): FeatureFlag {
  const now = new Date().toISOString();

  return {
    key,
    name: options.name || key,
    description: options.description,
    enabled: options.enabled ?? true,
    rules: options.rules,
    defaultValue: options.defaultValue ?? false,
    tags: options.tags,
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Create a targeting rule.
 */
export function createRule(
  name: string,
  conditions: RuleCondition[],
  value: FlagValue,
  percentage?: number
): TargetingRule {
  return {
    id: generateId(),
    name,
    conditions,
    value,
    percentage,
  };
}

/**
 * Create a condition.
 */
export function createCondition(
  attribute: string,
  operator: ConditionOperator,
  value: RuleCondition["value"]
): RuleCondition {
  return { attribute, operator, value };
}

// =============================================================================
// Singleton Instance
// =============================================================================

let defaultClient: FeatureFlagClient | null = null;

/**
 * Get the default feature flag client.
 */
export function getFeatureFlagClient(): FeatureFlagClient {
  if (!defaultClient) {
    defaultClient = new FeatureFlagClient();
  }
  return defaultClient;
}

/**
 * Initialize the default feature flag client with config.
 */
export function initFeatureFlags(
  config: FeatureFlagClientConfig
): FeatureFlagClient {
  defaultClient = new FeatureFlagClient(config);
  return defaultClient;
}
