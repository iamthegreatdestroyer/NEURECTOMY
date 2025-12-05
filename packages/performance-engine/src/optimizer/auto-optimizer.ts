/**
 * @neurectomy/performance-engine - Auto-Optimizer
 *
 * @elite-agent-collective @VELOCITY @NEURAL
 *
 * Self-learning optimization engine that automatically detects
 * performance issues and applies optimizations in real-time.
 */

import { EventEmitter } from "events";
import * as crypto from "crypto";
import type {
  AutoOptimizationRule,
  OptimizationOpportunity,
  MetricDataPoint,
} from "../types.js";

// ============================================================================
// TYPES
// ============================================================================

interface AutoOptimizerConfig {
  enabled: boolean;
  evaluationInterval: number; // ms
  learningRate: number;
  cooldownMultiplier: number;
  maxConcurrentActions: number;
  safeMode: boolean; // Only log, don't apply
}

interface OptimizationAction {
  id: string;
  ruleId: string;
  type: string;
  parameters: Record<string, unknown>;
  timestamp: number;
  status: "pending" | "applied" | "rolled_back" | "failed";
  result?: {
    success: boolean;
    metricsBeforeAction: MetricDataPoint[];
    metricsAfterAction: MetricDataPoint[];
    improvement: number;
    error?: string;
  };
}

interface OptimizationHistory {
  actions: OptimizationAction[];
  successRate: number;
  totalImprovement: number;
  lastEvaluation: number;
}

interface LearningModel {
  ruleEffectiveness: Map<string, number>;
  contextPatterns: Map<string, ContextPattern>;
  failurePatterns: Map<string, FailurePattern>;
}

interface ContextPattern {
  metricName: string;
  timeOfDay: number[];
  dayOfWeek: number[];
  effectiveActions: string[];
  avgImprovement: number;
}

interface FailurePattern {
  actionType: string;
  conditions: Record<string, unknown>;
  failureCount: number;
  lastFailure: number;
}

// ============================================================================
// AUTO-OPTIMIZER IMPLEMENTATION
// ============================================================================

/**
 * Self-learning auto-optimization engine
 *
 * @elite-agent-collective @VELOCITY - Real-time optimization
 * @elite-agent-collective @NEURAL - Learning and adaptation
 */
export class AutoOptimizer extends EventEmitter {
  private config: AutoOptimizerConfig;
  private rules: Map<string, AutoOptimizationRule> = new Map();
  private history: OptimizationHistory;
  private learningModel: LearningModel;
  private metricBuffer: Map<string, MetricDataPoint[]> = new Map();
  private activeActions: Map<string, OptimizationAction> = new Map();
  private evaluationTimer: NodeJS.Timer | null = null;

  private readonly defaultConfig: AutoOptimizerConfig = {
    enabled: true,
    evaluationInterval: 10000, // 10 seconds
    learningRate: 0.1,
    cooldownMultiplier: 2.0,
    maxConcurrentActions: 3,
    safeMode: false,
  };

  constructor(config?: Partial<AutoOptimizerConfig>) {
    super();
    this.config = { ...this.defaultConfig, ...config };

    this.history = {
      actions: [],
      successRate: 0,
      totalImprovement: 0,
      lastEvaluation: 0,
    };

    this.learningModel = {
      ruleEffectiveness: new Map(),
      contextPatterns: new Map(),
      failurePatterns: new Map(),
    };

    this.initializeDefaultRules();
  }

  /**
   * Start the auto-optimizer
   */
  start(): void {
    if (!this.config.enabled) {
      this.emit("status", { status: "disabled" });
      return;
    }

    if (this.evaluationTimer) {
      return; // Already running
    }

    this.evaluationTimer = setInterval(
      () => this.evaluate(),
      this.config.evaluationInterval
    );

    this.emit("started", { interval: this.config.evaluationInterval });
  }

  /**
   * Stop the auto-optimizer
   */
  stop(): void {
    if (this.evaluationTimer) {
      clearInterval(this.evaluationTimer);
      this.evaluationTimer = null;
    }

    this.emit("stopped");
  }

  /**
   * Record a metric value for evaluation
   */
  recordMetric(
    name: string,
    value: number,
    labels?: Record<string, string>
  ): void {
    const dataPoint: MetricDataPoint = {
      timestamp: Date.now(),
      value,
      labels,
    };

    let buffer = this.metricBuffer.get(name);
    if (!buffer) {
      buffer = [];
      this.metricBuffer.set(name, buffer);
    }

    buffer.push(dataPoint);

    // Keep last 1000 data points per metric
    if (buffer.length > 1000) {
      buffer.shift();
    }
  }

  /**
   * Add an optimization rule
   */
  addRule(rule: Omit<AutoOptimizationRule, "id">): AutoOptimizationRule {
    const id = crypto.randomUUID();
    const fullRule: AutoOptimizationRule = { ...rule, id };

    this.rules.set(id, fullRule);
    this.learningModel.ruleEffectiveness.set(id, 1.0); // Initial effectiveness

    this.emit("rule:added", { ruleId: id, rule: fullRule });
    return fullRule;
  }

  /**
   * Remove an optimization rule
   */
  removeRule(ruleId: string): boolean {
    const removed = this.rules.delete(ruleId);
    if (removed) {
      this.learningModel.ruleEffectiveness.delete(ruleId);
      this.emit("rule:removed", { ruleId });
    }
    return removed;
  }

  /**
   * Get all rules
   */
  getRules(): AutoOptimizationRule[] {
    return Array.from(this.rules.values());
  }

  /**
   * Get optimization history
   */
  getHistory(limit?: number): OptimizationAction[] {
    const actions = [...this.history.actions].reverse();
    return limit ? actions.slice(0, limit) : actions;
  }

  /**
   * Get learning insights
   */
  getLearningInsights(): {
    ruleEffectiveness: Record<string, number>;
    topPerformingRules: string[];
    failurePatterns: Array<{ pattern: string; count: number }>;
    recommendations: string[];
  } {
    const effectiveness: Record<string, number> = {};
    for (const [ruleId, value] of this.learningModel.ruleEffectiveness) {
      const rule = this.rules.get(ruleId);
      effectiveness[rule?.name || ruleId] = value;
    }

    const topPerformingRules = Array.from(
      this.learningModel.ruleEffectiveness.entries()
    )
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([ruleId]) => this.rules.get(ruleId)?.name || ruleId);

    const failurePatterns = Array.from(
      this.learningModel.failurePatterns.entries()
    )
      .map(([pattern, data]) => ({ pattern, count: data.failureCount }))
      .sort((a, b) => b.count - a.count);

    const recommendations = this.generateRecommendations();

    return {
      ruleEffectiveness: effectiveness,
      topPerformingRules,
      failurePatterns,
      recommendations,
    };
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Initialize default optimization rules
   */
  private initializeDefaultRules(): void {
    // Cache size adjustment rule
    this.addRule({
      name: "Cache Size Auto-Adjust",
      description: "Automatically adjust cache size based on hit rate",
      trigger: {
        metricName: "cache_hit_rate",
        condition: "lt",
        threshold: 0.7,
        duration: 60000,
      },
      action: {
        type: "adjust_cache_size",
        parameters: { multiplier: 1.5 },
      },
      cooldown: 300000, // 5 minutes
      enabled: true,
    });

    // Memory pressure relief rule
    this.addRule({
      name: "Memory Pressure Relief",
      description: "Clear caches when memory pressure is high",
      trigger: {
        metricName: "memory_pressure",
        condition: "gte",
        threshold: 0.85,
        duration: 30000,
      },
      action: {
        type: "clear_cache",
        parameters: { retention: 0.5 },
      },
      cooldown: 120000, // 2 minutes
      enabled: true,
    });

    // Worker scaling rule
    this.addRule({
      name: "Worker Auto-Scale",
      description: "Scale workers based on queue depth",
      trigger: {
        metricName: "queue_depth",
        condition: "gt",
        threshold: 100,
        duration: 30000,
      },
      action: {
        type: "scale_workers",
        parameters: { increment: 2 },
      },
      cooldown: 60000, // 1 minute
      enabled: true,
    });

    // GC trigger rule
    this.addRule({
      name: "Proactive GC",
      description: "Trigger GC during low activity periods",
      trigger: {
        metricName: "request_rate",
        condition: "lt",
        threshold: 10,
        duration: 60000,
      },
      action: {
        type: "trigger_gc",
        parameters: {},
      },
      cooldown: 300000, // 5 minutes
      enabled: true,
    });

    // Compression enablement rule
    this.addRule({
      name: "Auto-Enable Compression",
      description: "Enable compression when response sizes are large",
      trigger: {
        metricName: "avg_response_size",
        condition: "gt",
        threshold: 102400, // 100KB
        duration: 60000,
      },
      action: {
        type: "enable_compression",
        parameters: { algorithm: "gzip", level: 6 },
      },
      cooldown: 600000, // 10 minutes
      enabled: true,
    });
  }

  /**
   * Evaluate all rules and trigger actions
   */
  private async evaluate(): Promise<void> {
    this.history.lastEvaluation = Date.now();

    for (const [ruleId, rule] of this.rules) {
      if (!rule.enabled) continue;

      // Check cooldown
      if (rule.lastTriggered) {
        const cooldown = this.getAdaptiveCooldown(ruleId, rule.cooldown);
        if (Date.now() - rule.lastTriggered < cooldown) continue;
      }

      // Check if rule should trigger
      const shouldTrigger = this.evaluateRule(rule);

      if (shouldTrigger) {
        await this.executeAction(rule);
      }
    }

    // Cleanup old history
    this.cleanupHistory();
  }

  /**
   * Evaluate a single rule
   */
  private evaluateRule(rule: AutoOptimizationRule): boolean {
    const { trigger } = rule;
    const metrics = this.metricBuffer.get(trigger.metricName);

    if (!metrics || metrics.length === 0) return false;

    // Get metrics within duration window
    const windowStart = Date.now() - (trigger.duration || 60000);
    const windowMetrics = metrics.filter((m) => m.timestamp >= windowStart);

    if (windowMetrics.length === 0) return false;

    // Calculate average value in window
    const avgValue =
      windowMetrics.reduce((sum, m) => sum + m.value, 0) / windowMetrics.length;

    // Evaluate condition
    switch (trigger.condition) {
      case "gt":
        return avgValue > trigger.threshold;
      case "gte":
        return avgValue >= trigger.threshold;
      case "lt":
        return avgValue < trigger.threshold;
      case "lte":
        return avgValue <= trigger.threshold;
      case "eq":
        return avgValue === trigger.threshold;
      case "neq":
        return avgValue !== trigger.threshold;
      default:
        return false;
    }
  }

  /**
   * Execute an optimization action
   */
  private async executeAction(rule: AutoOptimizationRule): Promise<void> {
    // Check concurrent action limit
    if (this.activeActions.size >= this.config.maxConcurrentActions) {
      return;
    }

    const actionId = crypto.randomUUID();
    const action: OptimizationAction = {
      id: actionId,
      ruleId: rule.id,
      type: rule.action.type,
      parameters: rule.action.parameters,
      timestamp: Date.now(),
      status: "pending",
    };

    this.activeActions.set(actionId, action);
    this.emit("action:started", { action, rule });

    // Capture metrics before action
    const metricsBefore = this.captureCurrentMetrics(rule.trigger.metricName);

    try {
      if (this.config.safeMode) {
        // In safe mode, just log
        this.emit("action:safe_mode", { action, rule });
      } else {
        // Execute the action
        await this.applyAction(action);
      }

      action.status = "applied";
      rule.lastTriggered = Date.now();

      // Wait for metrics to stabilize
      await this.delay(5000);

      // Capture metrics after action
      const metricsAfter = this.captureCurrentMetrics(rule.trigger.metricName);

      // Calculate improvement
      const improvement = this.calculateImprovement(
        metricsBefore,
        metricsAfter,
        rule
      );

      action.result = {
        success: true,
        metricsBeforeAction: metricsBefore,
        metricsAfterAction: metricsAfter,
        improvement,
      };

      // Update learning model
      this.updateLearningModel(rule, action, improvement);

      this.emit("action:completed", { action, rule, improvement });
    } catch (error) {
      action.status = "failed";
      action.result = {
        success: false,
        metricsBeforeAction: metricsBefore,
        metricsAfterAction: [],
        improvement: 0,
        error: error instanceof Error ? error.message : String(error),
      };

      // Record failure pattern
      this.recordFailure(rule, action);

      this.emit("action:failed", { action, rule, error });
    } finally {
      this.activeActions.delete(actionId);
      this.history.actions.push(action);
    }
  }

  /**
   * Apply the actual optimization action
   */
  private async applyAction(action: OptimizationAction): Promise<void> {
    switch (action.type) {
      case "adjust_cache_size":
        await this.applyCacheSizeAdjustment(action.parameters);
        break;
      case "adjust_pool_size":
        await this.applyPoolSizeAdjustment(action.parameters);
        break;
      case "enable_compression":
        await this.applyCompressionSettings(action.parameters);
        break;
      case "enable_batching":
        await this.applyBatchingSettings(action.parameters);
        break;
      case "scale_workers":
        await this.applyWorkerScaling(action.parameters);
        break;
      case "trigger_gc":
        await this.applyGCTrigger();
        break;
      case "clear_cache":
        await this.applyCacheClear(action.parameters);
        break;
      default:
        throw new Error(`Unknown action type: ${action.type}`);
    }
  }

  // Action implementations (stubs - would integrate with actual systems)
  private async applyCacheSizeAdjustment(
    params: Record<string, unknown>
  ): Promise<void> {
    // Implementation would adjust cache configuration
    this.emit("cache:size_adjusted", params);
  }

  private async applyPoolSizeAdjustment(
    params: Record<string, unknown>
  ): Promise<void> {
    this.emit("pool:size_adjusted", params);
  }

  private async applyCompressionSettings(
    params: Record<string, unknown>
  ): Promise<void> {
    this.emit("compression:enabled", params);
  }

  private async applyBatchingSettings(
    params: Record<string, unknown>
  ): Promise<void> {
    this.emit("batching:enabled", params);
  }

  private async applyWorkerScaling(
    params: Record<string, unknown>
  ): Promise<void> {
    this.emit("workers:scaled", params);
  }

  private async applyGCTrigger(): Promise<void> {
    if (global.gc) {
      global.gc();
    }
    this.emit("gc:triggered");
  }

  private async applyCacheClear(
    params: Record<string, unknown>
  ): Promise<void> {
    this.emit("cache:cleared", params);
  }

  /**
   * Capture current metrics
   */
  private captureCurrentMetrics(metricName: string): MetricDataPoint[] {
    const metrics = this.metricBuffer.get(metricName);
    if (!metrics) return [];
    return metrics.slice(-10); // Last 10 data points
  }

  /**
   * Calculate improvement from action
   */
  private calculateImprovement(
    before: MetricDataPoint[],
    after: MetricDataPoint[],
    rule: AutoOptimizationRule
  ): number {
    if (before.length === 0 || after.length === 0) return 0;

    const avgBefore =
      before.reduce((sum, m) => sum + m.value, 0) / before.length;
    const avgAfter = after.reduce((sum, m) => sum + m.value, 0) / after.length;

    // Calculate improvement based on trigger condition
    const { condition, threshold } = rule.trigger;

    switch (condition) {
      case "gt":
      case "gte":
        // For "greater than" triggers, improvement is reduction
        return (avgBefore - avgAfter) / avgBefore;
      case "lt":
      case "lte":
        // For "less than" triggers, improvement is increase
        return (avgAfter - avgBefore) / Math.max(avgBefore, threshold);
      default:
        return 0;
    }
  }

  /**
   * Update learning model with action results
   */
  private updateLearningModel(
    rule: AutoOptimizationRule,
    action: OptimizationAction,
    improvement: number
  ): void {
    // Update rule effectiveness
    const currentEffectiveness =
      this.learningModel.ruleEffectiveness.get(rule.id) || 1.0;
    const newEffectiveness =
      currentEffectiveness +
      this.config.learningRate * (improvement - currentEffectiveness);
    this.learningModel.ruleEffectiveness.set(rule.id, newEffectiveness);

    // Update context patterns
    const hour = new Date().getHours();
    const dayOfWeek = new Date().getDay();
    const contextKey = `${rule.trigger.metricName}:${rule.action.type}`;

    let pattern = this.learningModel.contextPatterns.get(contextKey);
    if (!pattern) {
      pattern = {
        metricName: rule.trigger.metricName,
        timeOfDay: [],
        dayOfWeek: [],
        effectiveActions: [],
        avgImprovement: 0,
      };
      this.learningModel.contextPatterns.set(contextKey, pattern);
    }

    if (improvement > 0) {
      pattern.timeOfDay.push(hour);
      pattern.dayOfWeek.push(dayOfWeek);
      pattern.effectiveActions.push(action.type);
      pattern.avgImprovement = (pattern.avgImprovement + improvement) / 2;
    }

    // Update history stats
    this.history.totalImprovement += improvement;
    const successCount = this.history.actions.filter(
      (a) => a.result?.success
    ).length;
    this.history.successRate =
      successCount / Math.max(1, this.history.actions.length);
  }

  /**
   * Record failure pattern
   */
  private recordFailure(
    rule: AutoOptimizationRule,
    action: OptimizationAction
  ): void {
    const patternKey = `${action.type}:${JSON.stringify(action.parameters)}`;

    let pattern = this.learningModel.failurePatterns.get(patternKey);
    if (!pattern) {
      pattern = {
        actionType: action.type,
        conditions: action.parameters,
        failureCount: 0,
        lastFailure: 0,
      };
      this.learningModel.failurePatterns.set(patternKey, pattern);
    }

    pattern.failureCount++;
    pattern.lastFailure = Date.now();

    // Decrease rule effectiveness on failure
    const currentEffectiveness =
      this.learningModel.ruleEffectiveness.get(rule.id) || 1.0;
    this.learningModel.ruleEffectiveness.set(
      rule.id,
      currentEffectiveness * (1 - this.config.learningRate)
    );
  }

  /**
   * Get adaptive cooldown based on rule effectiveness
   */
  private getAdaptiveCooldown(ruleId: string, baseCooldown: number): number {
    const effectiveness =
      this.learningModel.ruleEffectiveness.get(ruleId) || 1.0;

    // More effective rules get shorter cooldowns
    // Less effective rules get longer cooldowns
    return (
      baseCooldown * (1 + (1 - effectiveness) * this.config.cooldownMultiplier)
    );
  }

  /**
   * Generate recommendations based on learning
   */
  private generateRecommendations(): string[] {
    const recommendations: string[] = [];

    // Check for ineffective rules
    for (const [ruleId, effectiveness] of this.learningModel
      .ruleEffectiveness) {
      const rule = this.rules.get(ruleId);
      if (effectiveness < 0.3 && rule) {
        recommendations.push(
          `Consider disabling rule "${rule.name}" - effectiveness is only ${(effectiveness * 100).toFixed(1)}%`
        );
      }
    }

    // Check for frequent failures
    for (const [, pattern] of this.learningModel.failurePatterns) {
      if (pattern.failureCount > 5) {
        recommendations.push(
          `Action "${pattern.actionType}" has failed ${pattern.failureCount} times - review configuration`
        );
      }
    }

    // Check overall success rate
    if (this.history.successRate < 0.5 && this.history.actions.length > 10) {
      recommendations.push(
        `Overall optimization success rate is low (${(this.history.successRate * 100).toFixed(1)}%) - review trigger thresholds`
      );
    }

    return recommendations;
  }

  /**
   * Cleanup old history entries
   */
  private cleanupHistory(): void {
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    const cutoff = Date.now() - maxAge;

    this.history.actions = this.history.actions.filter(
      (a) => a.timestamp > cutoff
    );
  }

  /**
   * Delay utility
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { AutoOptimizer as default };
export type {
  AutoOptimizerConfig,
  OptimizationAction,
  OptimizationHistory,
  LearningModel,
  ContextPattern,
  FailurePattern,
};
