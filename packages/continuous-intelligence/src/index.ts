/**
 * NEURECTOMY Continuous Intelligence
 *
 * Self-improving AI with predictive maintenance and ML-based optimization
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import {
 *   ContinuousIntelligence,
 *   createContinuousIntelligence,
 * } from '@neurectomy/continuous-intelligence';
 *
 * // Create unified intelligence engine
 * const intelligence = createContinuousIntelligence({
 *   selfImprovement: { enabled: true, learningRate: 0.01 },
 *   predictive: { enabled: true, predictionWindow: 24 },
 *   optimization: { enabled: true, algorithm: 'bayesian' },
 * });
 *
 * // Start monitoring
 * await intelligence.start();
 *
 * // Record observations
 * await intelligence.recordObservation({
 *   component: 'api',
 *   action: 'query',
 *   context: { endpoint: '/users' },
 *   result: { success: true, duration: 150 },
 *   metrics: { latency: 150 }
 * });
 *
 * // Get health snapshot
 * const health = await intelligence.getHealthSnapshot();
 *
 * // Run optimization
 * const result = await intelligence.optimize([
 *   { name: 'latency', objective: 'minimize-latency', targetValue: 100 }
 * ]);
 * ```
 */

import { EventEmitter } from "events";
import {
  SelfImprovementEngine,
  createSelfImprovementEngine,
} from "./self-improvement/index.js";
import {
  PredictiveMaintenance,
  createPredictiveMaintenance,
} from "./predictive/index.js";
import { AutoOptimizer, createAutoOptimizer } from "./optimization/index.js";
import type {
  SelfImprovementConfig,
  PredictiveConfig,
  OptimizationConfig,
  LearningObservation,
  UserFeedback,
  LearningModel,
  ImprovementStrategy,
  SystemHealthSnapshot,
  FailurePrediction,
  OptimizationTarget,
  OptimizationResult,
  TunableParameter,
  QueryOptimization,
  HealthAlert,
  PreventiveAction,
} from "./types.js";

// ==============================================================================
// Unified Configuration
// ==============================================================================

/**
 * Configuration for continuous intelligence
 */
export interface ContinuousIntelligenceConfig {
  /** Self-improvement settings */
  selfImprovement: Partial<SelfImprovementConfig>;
  /** Predictive maintenance settings */
  predictive: Partial<PredictiveConfig>;
  /** Auto-optimization settings */
  optimization: Partial<OptimizationConfig>;
}

const DEFAULT_CONFIG: ContinuousIntelligenceConfig = {
  selfImprovement: {
    enabled: true,
    learningRate: 0.01,
    retentionDays: 30,
    minObservations: 100,
    autoApply: false,
    maxAutoApplyRisk: "low",
  },
  predictive: {
    enabled: true,
    predictionWindow: 24,
    alertThresholds: { warning: 0.7, critical: 0.9 },
    autoRemediate: false,
    maxAutoRemediateSeverity: "warning",
  },
  optimization: {
    enabled: true,
    algorithm: "bayesian",
    maxIterations: 100,
    convergenceThreshold: 0.001,
    explorationRatio: 0.2,
    autoApply: false,
  },
};

// ==============================================================================
// Unified Intelligence Engine
// ==============================================================================

/**
 * Unified continuous intelligence engine
 *
 * Combines self-improvement, predictive maintenance, and auto-optimization
 * into a single cohesive system.
 */
export class ContinuousIntelligence extends EventEmitter {
  private config: ContinuousIntelligenceConfig;
  private selfImprovement: SelfImprovementEngine;
  private predictive: PredictiveMaintenance;
  private optimizer: AutoOptimizer;
  private isStarted = false;
  private syncInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<ContinuousIntelligenceConfig> = {}) {
    super();

    this.config = {
      selfImprovement: {
        ...DEFAULT_CONFIG.selfImprovement,
        ...config.selfImprovement,
      },
      predictive: { ...DEFAULT_CONFIG.predictive, ...config.predictive },
      optimization: { ...DEFAULT_CONFIG.optimization, ...config.optimization },
    };

    // Create sub-engines
    this.selfImprovement = createSelfImprovementEngine(
      this.config.selfImprovement
    );
    this.predictive = createPredictiveMaintenance(this.config.predictive);
    this.optimizer = createAutoOptimizer(this.config.optimization);

    // Wire up event forwarding
    this.setupEventForwarding();
  }

  /**
   * Start the intelligence engine
   */
  async start(): Promise<void> {
    if (this.isStarted) return;

    this.isStarted = true;

    // Start periodic sync between components
    this.syncInterval = setInterval(() => this.syncComponents(), 60000);

    this.emit("started");
  }

  /**
   * Stop the intelligence engine
   */
  async stop(): Promise<void> {
    if (!this.isStarted) return;

    this.isStarted = false;

    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }

    this.emit("stopped");
  }

  // ==============================================================================
  // Self-Improvement API
  // ==============================================================================

  /**
   * Record a learning observation
   */
  async recordObservation(
    input: Omit<LearningObservation, "id" | "timestamp">
  ): Promise<LearningObservation> {
    const observation = await this.selfImprovement.recordObservation(input);

    // Also record metrics to predictive engine
    if (input.metrics) {
      await this.predictive.recordMetric(
        input.component,
        "latency_p99",
        input.metrics.latency
      );
      if (input.metrics.memoryUsage) {
        await this.predictive.recordMetric(
          input.component,
          "memory_usage",
          input.metrics.memoryUsage
        );
      }
      if (input.metrics.cpuUsage) {
        await this.predictive.recordMetric(
          input.component,
          "cpu_usage",
          input.metrics.cpuUsage
        );
      }
    }

    return observation;
  }

  /**
   * Record user feedback for an observation
   */
  async recordFeedback(
    observationId: string,
    feedback: UserFeedback
  ): Promise<void> {
    await this.selfImprovement.recordFeedback(observationId, feedback);
  }

  /**
   * Trigger learning cycle
   */
  async learn(): Promise<LearningModel[]> {
    return this.selfImprovement.learn();
  }

  /**
   * Generate improvement strategies
   */
  async generateStrategies(): Promise<ImprovementStrategy[]> {
    return this.selfImprovement.generateStrategies();
  }

  /**
   * Apply an improvement strategy
   */
  async applyStrategy(strategy: ImprovementStrategy): Promise<boolean> {
    return this.selfImprovement.applyStrategy(strategy);
  }

  /**
   * Get learning models
   */
  getModels(): LearningModel[] {
    return this.selfImprovement.getModels();
  }

  /**
   * Get learning statistics
   */
  getLearningStats(): {
    totalObservations: number;
    successRate: number;
    avgLatency: number;
    componentBreakdown: Record<string, number>;
  } {
    return this.selfImprovement.getStats();
  }

  // ==============================================================================
  // Predictive Maintenance API
  // ==============================================================================

  /**
   * Record a metric value
   */
  async recordMetric(
    component: string,
    metric: string,
    value: number
  ): Promise<void> {
    await this.predictive.recordMetric(component, metric, value);
  }

  /**
   * Get current health snapshot
   */
  async getHealthSnapshot(): Promise<SystemHealthSnapshot> {
    return this.predictive.getHealthSnapshot();
  }

  /**
   * Predict failures for a component
   */
  async predictFailures(component: string): Promise<FailurePrediction[]> {
    return this.predictive.predictFailures(component);
  }

  /**
   * Execute preventive action
   */
  async executePreventiveAction(
    action: PreventiveAction
  ): Promise<{ success: boolean; message: string }> {
    return this.predictive.executePreventiveAction(action);
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): HealthAlert[] {
    return this.predictive.getActiveAlerts();
  }

  /**
   * Acknowledge an alert
   */
  async acknowledgeAlert(alertId: string): Promise<void> {
    await this.predictive.acknowledgeAlert(alertId);
  }

  /**
   * Resolve an alert
   */
  async resolveAlert(alertId: string): Promise<void> {
    await this.predictive.resolveAlert(alertId);
  }

  // ==============================================================================
  // Auto-Optimization API
  // ==============================================================================

  /**
   * Register a tunable parameter
   */
  registerParameter(parameter: TunableParameter): void {
    this.optimizer.registerParameter(parameter);
  }

  /**
   * Unregister a parameter
   */
  unregisterParameter(parameterId: string): void {
    this.optimizer.unregisterParameter(parameterId);
  }

  /**
   * Get all registered parameters
   */
  getParameters(): TunableParameter[] {
    return this.optimizer.getParameters();
  }

  /**
   * Run optimization for given targets
   */
  async optimize(targets: OptimizationTarget[]): Promise<OptimizationResult> {
    return this.optimizer.optimize(targets);
  }

  /**
   * Analyze a query for optimization opportunities
   */
  analyzeQuery(query: string): QueryOptimization[] {
    return this.optimizer.analyzeQuery(query);
  }

  /**
   * Apply optimization result
   */
  async applyOptimization(
    result: OptimizationResult
  ): Promise<{ parameter: string; oldValue: number; newValue: number }[]> {
    return this.optimizer.applyOptimization(result);
  }

  /**
   * Get optimization history
   */
  getOptimizationHistory(): OptimizationResult[] {
    return this.optimizer.getHistory();
  }

  // ==============================================================================
  // Unified API
  // ==============================================================================

  /**
   * Get comprehensive system status
   */
  async getSystemStatus(): Promise<{
    health: SystemHealthSnapshot;
    learning: {
      totalObservations: number;
      successRate: number;
      avgLatency: number;
    };
    predictions: FailurePrediction[];
    alerts: HealthAlert[];
    isRunning: boolean;
  }> {
    const health = await this.getHealthSnapshot();
    const learning = this.getLearningStats();
    const alerts = this.getActiveAlerts();

    // Get predictions for all components
    const predictions: FailurePrediction[] = [];
    for (const component of health.components) {
      const componentPredictions = await this.predictFailures(component.name);
      predictions.push(...componentPredictions);
    }

    return {
      health,
      learning: {
        totalObservations: learning.totalObservations,
        successRate: learning.successRate,
        avgLatency: learning.avgLatency,
      },
      predictions,
      alerts,
      isRunning: this.isStarted,
    };
  }

  /**
   * Run full analysis cycle
   */
  async runAnalysis(): Promise<{
    strategies: ImprovementStrategy[];
    predictions: FailurePrediction[];
    health: SystemHealthSnapshot;
  }> {
    // Learn from observations
    await this.learn();

    // Generate improvement strategies
    const strategies = await this.generateStrategies();

    // Get health snapshot
    const health = await this.getHealthSnapshot();

    // Get predictions for all components
    const predictions: FailurePrediction[] = [];
    for (const component of health.components) {
      const componentPredictions = await this.predictFailures(component.name);
      predictions.push(...componentPredictions);
    }

    return { strategies, predictions, health };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ContinuousIntelligenceConfig>): void {
    if (config.selfImprovement) {
      this.config.selfImprovement = {
        ...this.config.selfImprovement,
        ...config.selfImprovement,
      };
      this.selfImprovement.updateConfig(this.config.selfImprovement);
    }

    if (config.predictive) {
      this.config.predictive = {
        ...this.config.predictive,
        ...config.predictive,
      };
      this.predictive.updateConfig(this.config.predictive);
    }

    if (config.optimization) {
      this.config.optimization = {
        ...this.config.optimization,
        ...config.optimization,
      };
      this.optimizer.updateConfig(this.config.optimization);
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): ContinuousIntelligenceConfig {
    return { ...this.config };
  }

  // Private methods

  private setupEventForwarding(): void {
    // Forward self-improvement events
    this.selfImprovement.on("learning:observation", (data) =>
      this.emit("learning:observation", data)
    );
    this.selfImprovement.on("learning:model-updated", (data) =>
      this.emit("learning:model-updated", data)
    );
    this.selfImprovement.on("learning:improvement-suggested", (data) =>
      this.emit("learning:improvement-suggested", data)
    );

    // Forward predictive events
    this.predictive.on("health:snapshot", (data) =>
      this.emit("health:snapshot", data)
    );
    this.predictive.on("health:failure-predicted", (data) =>
      this.emit("health:failure-predicted", data)
    );
    this.predictive.on("health:alert", (data) =>
      this.emit("health:alert", data)
    );

    // Forward optimization events
    this.optimizer.on("optimization:started", (data) =>
      this.emit("optimization:started", data)
    );
    this.optimizer.on("optimization:step", (data) =>
      this.emit("optimization:step", data)
    );
    this.optimizer.on("optimization:completed", (data) =>
      this.emit("optimization:completed", data)
    );

    // Forward errors
    this.selfImprovement.on("error", (data) => this.emit("error", data));
  }

  private async syncComponents(): Promise<void> {
    // Sync insights between components
    // e.g., use improvement strategies to inform optimization targets
    const strategies = await this.generateStrategies();

    for (const strategy of strategies) {
      // Convert high-confidence strategies to optimization parameters
      if (strategy.confidence >= 0.8 && strategy.type === "parameter-tuning") {
        for (const change of strategy.proposedChanges) {
          if (
            typeof change.currentValue === "number" &&
            typeof change.proposedValue === "number"
          ) {
            this.registerParameter({
              id: `auto-${change.target}`,
              name: change.target,
              component: strategy.targetComponent,
              currentValue: change.currentValue,
              valueType: "continuous",
              range: {
                min: Math.min(change.currentValue, change.proposedValue) * 0.5,
                max: Math.max(change.currentValue, change.proposedValue) * 1.5,
              },
              sensitivity: strategy.confidence,
              impact: { "minimize-latency": 0.5 },
            });
          }
        }
      }
    }
  }
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create a continuous intelligence engine instance
 */
export function createContinuousIntelligence(
  config?: Partial<ContinuousIntelligenceConfig>
): ContinuousIntelligence {
  return new ContinuousIntelligence(config);
}

// ==============================================================================
// Re-exports
// ==============================================================================

// Types
export * from "./types.js";

// Self-Improvement
export {
  SelfImprovementEngine,
  createSelfImprovementEngine,
  type UsagePattern,
  type PatternType,
} from "./self-improvement/index.js";

// Predictive Maintenance
export {
  PredictiveMaintenance,
  createPredictiveMaintenance,
  type TimeSeriesPoint,
  type TimeSeriesAnalysis,
  type AnomalyPoint,
  type MetricHistory,
} from "./predictive/index.js";

// Auto-Optimization
export {
  AutoOptimizer,
  createAutoOptimizer,
  BayesianOptimizer,
  GeneticOptimizer,
  SimulatedAnnealingOptimizer,
  type Optimizer,
} from "./optimization/index.js";
