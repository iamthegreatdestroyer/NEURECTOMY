/**
 * NEURECTOMY Self-Improvement Engine
 *
 * Meta-learning algorithms for continuous system improvement
 *
 * @packageDocumentation
 */

import { EventEmitter } from "events";
import type {
  LearningObservation,
  ActionResult,
  PerformanceMetrics,
  UserFeedback,
  LearningModel,
  ModelType,
  TrainingMetrics,
  ImprovementStrategy,
  ImprovementType,
  ProposedChange,
  ImpactAssessment,
  RiskLevel,
  SelfImprovementConfig,
  ContinuousIntelligenceEvents,
} from "../types.js";

// ==============================================================================
// Configuration Defaults
// ==============================================================================

const DEFAULT_CONFIG: SelfImprovementConfig = {
  enabled: true,
  learningRate: 0.01,
  retentionDays: 30,
  minObservations: 100,
  autoApply: false,
  maxAutoApplyRisk: "low",
};

// ==============================================================================
// Pattern Detection
// ==============================================================================

/**
 * Detected usage pattern
 */
interface UsagePattern {
  id: string;
  type: PatternType;
  component: string;
  frequency: number;
  confidence: number;
  timeOfDay?: { start: number; end: number };
  dayOfWeek?: number[];
  correlations: PatternCorrelation[];
  firstSeen: Date;
  lastSeen: Date;
}

type PatternType =
  | "temporal"
  | "sequential"
  | "concurrent"
  | "resource-spike"
  | "error-cluster"
  | "performance-degradation";

interface PatternCorrelation {
  pattern: string;
  strength: number;
  direction: "positive" | "negative";
}

/**
 * Pattern detector for usage analysis
 */
class PatternDetector {
  private patterns: Map<string, UsagePattern> = new Map();
  private sequenceBuffer: LearningObservation[] = [];
  private readonly maxBufferSize = 1000;

  /**
   * Analyze observation for patterns
   */
  analyze(observation: LearningObservation): UsagePattern[] {
    this.sequenceBuffer.push(observation);
    if (this.sequenceBuffer.length > this.maxBufferSize) {
      this.sequenceBuffer.shift();
    }

    const detected: UsagePattern[] = [];

    // Temporal pattern detection
    const temporal = this.detectTemporalPattern(observation);
    if (temporal) detected.push(temporal);

    // Sequential pattern detection
    const sequential = this.detectSequentialPattern();
    if (sequential) detected.push(sequential);

    // Resource spike detection
    const spike = this.detectResourceSpike(observation);
    if (spike) detected.push(spike);

    // Error cluster detection
    const errorCluster = this.detectErrorCluster();
    if (errorCluster) detected.push(errorCluster);

    return detected;
  }

  private detectTemporalPattern(
    observation: LearningObservation
  ): UsagePattern | null {
    const hour = observation.timestamp.getHours();
    const dayOfWeek = observation.timestamp.getDay();
    const key = `temporal:${observation.component}:${observation.action}`;

    const existing = this.patterns.get(key);
    if (existing) {
      existing.frequency++;
      existing.lastSeen = observation.timestamp;
      existing.confidence = Math.min(0.99, existing.confidence + 0.01);
      return existing;
    }

    // Check if this time slot is becoming a pattern
    const sameTimeSlot = this.sequenceBuffer.filter((obs) => {
      const obsHour = obs.timestamp.getHours();
      return (
        obs.component === observation.component &&
        obs.action === observation.action &&
        Math.abs(obsHour - hour) <= 1
      );
    });

    if (sameTimeSlot.length >= 5) {
      const pattern: UsagePattern = {
        id: `pattern-${Date.now()}`,
        type: "temporal",
        component: observation.component,
        frequency: sameTimeSlot.length,
        confidence: 0.5 + sameTimeSlot.length * 0.05,
        timeOfDay: { start: hour - 1, end: hour + 1 },
        dayOfWeek: [dayOfWeek],
        correlations: [],
        firstSeen: sameTimeSlot[0].timestamp,
        lastSeen: observation.timestamp,
      };
      this.patterns.set(key, pattern);
      return pattern;
    }

    return null;
  }

  private detectSequentialPattern(): UsagePattern | null {
    if (this.sequenceBuffer.length < 10) return null;

    const recent = this.sequenceBuffer.slice(-10);
    const actionSequence = recent.map((o) => `${o.component}:${o.action}`);

    // Simple n-gram detection
    const ngrams = new Map<string, number>();
    for (let i = 0; i < actionSequence.length - 2; i++) {
      const ngram = actionSequence.slice(i, i + 3).join(" -> ");
      ngrams.set(ngram, (ngrams.get(ngram) || 0) + 1);
    }

    // Find repeated sequences
    for (const [ngram, count] of ngrams) {
      if (count >= 3) {
        const key = `sequential:${ngram}`;
        const existing = this.patterns.get(key);
        if (existing) {
          existing.frequency++;
          existing.confidence = Math.min(0.95, existing.confidence + 0.02);
          return existing;
        }

        const pattern: UsagePattern = {
          id: `pattern-${Date.now()}`,
          type: "sequential",
          component: ngram.split(" -> ")[0].split(":")[0],
          frequency: count,
          confidence: 0.6,
          correlations: [],
          firstSeen: recent[0].timestamp,
          lastSeen: recent[recent.length - 1].timestamp,
        };
        this.patterns.set(key, pattern);
        return pattern;
      }
    }

    return null;
  }

  private detectResourceSpike(
    observation: LearningObservation
  ): UsagePattern | null {
    const { memoryUsage, cpuUsage } = observation.metrics;

    if (!memoryUsage && !cpuUsage) return null;

    // Calculate average from buffer
    const averages = this.calculateResourceAverages();

    const memorySpike = memoryUsage && memoryUsage > averages.memory * 1.5;
    const cpuSpike = cpuUsage && cpuUsage > averages.cpu * 1.5;

    if (memorySpike || cpuSpike) {
      const key = `spike:${observation.component}`;
      const existing = this.patterns.get(key);

      if (existing) {
        existing.frequency++;
        existing.lastSeen = observation.timestamp;
        return existing;
      }

      const pattern: UsagePattern = {
        id: `pattern-${Date.now()}`,
        type: "resource-spike",
        component: observation.component,
        frequency: 1,
        confidence: 0.7,
        correlations: [],
        firstSeen: observation.timestamp,
        lastSeen: observation.timestamp,
      };
      this.patterns.set(key, pattern);
      return pattern;
    }

    return null;
  }

  private detectErrorCluster(): UsagePattern | null {
    const recent = this.sequenceBuffer.slice(-50);
    const errors = recent.filter((o) => !o.result.success);

    if (errors.length < 5) return null;

    // Group errors by component
    const byComponent = new Map<string, LearningObservation[]>();
    for (const error of errors) {
      const existing = byComponent.get(error.component) || [];
      existing.push(error);
      byComponent.set(error.component, existing);
    }

    for (const [component, componentErrors] of byComponent) {
      if (componentErrors.length >= 3) {
        const key = `error-cluster:${component}`;
        const existing = this.patterns.get(key);

        if (existing) {
          existing.frequency = componentErrors.length;
          existing.lastSeen =
            componentErrors[componentErrors.length - 1].timestamp;
          existing.confidence = Math.min(
            0.95,
            0.5 + componentErrors.length * 0.1
          );
          return existing;
        }

        const pattern: UsagePattern = {
          id: `pattern-${Date.now()}`,
          type: "error-cluster",
          component,
          frequency: componentErrors.length,
          confidence: 0.5 + componentErrors.length * 0.1,
          correlations: [],
          firstSeen: componentErrors[0].timestamp,
          lastSeen: componentErrors[componentErrors.length - 1].timestamp,
        };
        this.patterns.set(key, pattern);
        return pattern;
      }
    }

    return null;
  }

  private calculateResourceAverages(): { memory: number; cpu: number } {
    let totalMemory = 0;
    let totalCpu = 0;
    let memoryCount = 0;
    let cpuCount = 0;

    for (const obs of this.sequenceBuffer) {
      if (obs.metrics.memoryUsage) {
        totalMemory += obs.metrics.memoryUsage;
        memoryCount++;
      }
      if (obs.metrics.cpuUsage) {
        totalCpu += obs.metrics.cpuUsage;
        cpuCount++;
      }
    }

    return {
      memory: memoryCount > 0 ? totalMemory / memoryCount : 0,
      cpu: cpuCount > 0 ? totalCpu / cpuCount : 0,
    };
  }

  /**
   * Get all detected patterns
   */
  getPatterns(): UsagePattern[] {
    return Array.from(this.patterns.values());
  }

  /**
   * Get patterns for a specific component
   */
  getPatternsForComponent(component: string): UsagePattern[] {
    return Array.from(this.patterns.values()).filter(
      (p) => p.component === component
    );
  }
}

// ==============================================================================
// Learning Model Manager
// ==============================================================================

/**
 * Manages learning models for self-improvement
 */
class ModelManager {
  private models: Map<string, LearningModel> = new Map();

  /**
   * Get or create model for component
   */
  getOrCreateModel(component: string, type: ModelType): LearningModel {
    const key = `${component}:${type}`;
    const existing = this.models.get(key);
    if (existing) return existing;

    const model: LearningModel = {
      id: `model-${Date.now()}`,
      type,
      version: 1,
      parameters: this.initializeParameters(type),
      trainingMetrics: {
        observations: 0,
        iterations: 0,
        accuracy: 0,
        loss: 1,
        trainingDuration: 0,
      },
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.models.set(key, model);
    return model;
  }

  /**
   * Update model with new training data
   */
  updateModel(
    model: LearningModel,
    observations: LearningObservation[],
    learningRate: number
  ): LearningModel {
    const startTime = Date.now();

    // Simple gradient-based parameter update
    for (const obs of observations) {
      const feedback = obs.feedback;
      const performance = obs.result.success ? 1 : -1;
      const weight = feedback?.rating ? (feedback.rating - 3) / 2 : performance;

      for (const param of Object.keys(model.parameters)) {
        const gradient = this.computeGradient(model, obs, param);
        model.parameters[param] += learningRate * weight * gradient;
      }

      model.trainingMetrics.observations++;
      model.trainingMetrics.iterations++;
    }

    // Update metrics
    model.trainingMetrics.trainingDuration += Date.now() - startTime;
    model.trainingMetrics.accuracy = this.calculateAccuracy(
      model,
      observations
    );
    model.trainingMetrics.loss = 1 - model.trainingMetrics.accuracy;
    model.version++;
    model.updatedAt = new Date();

    return model;
  }

  private initializeParameters(type: ModelType): Record<string, number> {
    switch (type) {
      case "decision-tree":
        return { depth: 5, minSamples: 10, confidence: 0.8 };
      case "linear-regression":
        return { intercept: 0, slope: 1, regularization: 0.1 };
      case "neural-network":
        return { hiddenSize: 64, dropout: 0.2, activation: 1 };
      case "reinforcement":
        return { gamma: 0.99, epsilon: 0.1, alpha: 0.01 };
      case "bayesian":
        return { priorMean: 0, priorVariance: 1, sampleSize: 100 };
      case "ensemble":
        return { numModels: 5, votingThreshold: 0.6 };
      default:
        return {};
    }
  }

  private computeGradient(
    model: LearningModel,
    obs: LearningObservation,
    param: string
  ): number {
    // Simplified gradient computation based on observation
    const baseGradient = obs.result.success ? 0.1 : -0.1;
    const latencyFactor = 1 / (1 + obs.metrics.latency / 1000);
    return baseGradient * latencyFactor * (Math.random() * 0.5 + 0.5);
  }

  private calculateAccuracy(
    model: LearningModel,
    observations: LearningObservation[]
  ): number {
    if (observations.length === 0) return 0;

    const successRate =
      observations.filter((o) => o.result.success).length / observations.length;
    const avgLatency =
      observations.reduce((sum, o) => sum + o.metrics.latency, 0) /
      observations.length;

    // Accuracy combines success rate and latency performance
    const latencyScore = Math.max(0, 1 - avgLatency / 5000);
    return (successRate * 0.7 + latencyScore * 0.3) * 0.95;
  }

  /**
   * Get all models
   */
  getAllModels(): LearningModel[] {
    return Array.from(this.models.values());
  }
}

// ==============================================================================
// Strategy Generator
// ==============================================================================

/**
 * Generates improvement strategies from patterns and models
 */
class StrategyGenerator {
  /**
   * Generate improvement strategy from pattern
   */
  fromPattern(
    pattern: UsagePattern,
    observations: LearningObservation[]
  ): ImprovementStrategy | null {
    switch (pattern.type) {
      case "temporal":
        return this.temporalStrategy(pattern, observations);
      case "resource-spike":
        return this.resourceStrategy(pattern, observations);
      case "error-cluster":
        return this.errorStrategy(pattern, observations);
      case "performance-degradation":
        return this.performanceStrategy(pattern, observations);
      default:
        return null;
    }
  }

  private temporalStrategy(
    pattern: UsagePattern,
    observations: LearningObservation[]
  ): ImprovementStrategy {
    return {
      id: `strategy-${Date.now()}`,
      targetComponent: pattern.component,
      type: "caching-strategy",
      currentState: {
        cacheEnabled: false,
        preloadEnabled: false,
      },
      proposedChanges: [
        {
          target: "cacheEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Detected temporal pattern at ${pattern.timeOfDay?.start}-${pattern.timeOfDay?.end}h. Enable caching to improve response times during peak usage.`,
        },
        {
          target: "preloadEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Pre-load frequently accessed data before peak hours based on temporal pattern.`,
        },
      ],
      expectedImpact: {
        performanceGain: 25,
        resourceSavings: 10,
        reliabilityGain: 5,
        implementationEffort: 4,
      },
      confidence: pattern.confidence,
      riskLevel: "low",
    };
  }

  private resourceStrategy(
    pattern: UsagePattern,
    observations: LearningObservation[]
  ): ImprovementStrategy {
    const recentObs = observations.slice(-100);
    const avgMemory =
      recentObs.reduce((sum, o) => sum + (o.metrics.memoryUsage || 0), 0) /
      recentObs.length;

    return {
      id: `strategy-${Date.now()}`,
      targetComponent: pattern.component,
      type: "resource-allocation",
      currentState: {
        memoryLimit: avgMemory,
        cpuLimit: 1,
        autoScale: false,
      },
      proposedChanges: [
        {
          target: "memoryLimit",
          currentValue: avgMemory,
          proposedValue: avgMemory * 1.5,
          rationale: `Resource spikes detected ${pattern.frequency} times. Increase memory allocation to prevent OOM issues.`,
        },
        {
          target: "autoScale",
          currentValue: false,
          proposedValue: true,
          rationale: `Enable auto-scaling to handle resource spikes dynamically.`,
        },
      ],
      expectedImpact: {
        performanceGain: 15,
        resourceSavings: -20,
        reliabilityGain: 30,
        implementationEffort: 2,
      },
      confidence: pattern.confidence * 0.9,
      riskLevel: "medium",
    };
  }

  private errorStrategy(
    pattern: UsagePattern,
    observations: LearningObservation[]
  ): ImprovementStrategy {
    const errorObs = observations
      .filter((o) => !o.result.success && o.component === pattern.component)
      .slice(-20);

    const commonErrors = this.findCommonErrors(errorObs);

    return {
      id: `strategy-${Date.now()}`,
      targetComponent: pattern.component,
      type: "algorithm-switch",
      currentState: {
        retryEnabled: false,
        circuitBreakerEnabled: false,
        fallbackEnabled: false,
      },
      proposedChanges: [
        {
          target: "retryEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Error cluster detected with ${pattern.frequency} errors. Enable retry logic with exponential backoff.`,
        },
        {
          target: "circuitBreakerEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Implement circuit breaker to prevent cascading failures. Common errors: ${commonErrors.join(", ")}`,
        },
        {
          target: "fallbackEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Enable fallback mechanisms for graceful degradation.`,
        },
      ],
      expectedImpact: {
        performanceGain: 5,
        resourceSavings: 5,
        reliabilityGain: 40,
        implementationEffort: 8,
      },
      confidence: pattern.confidence,
      riskLevel: "medium",
    };
  }

  private performanceStrategy(
    pattern: UsagePattern,
    observations: LearningObservation[]
  ): ImprovementStrategy {
    const slowObs = observations
      .filter(
        (o) => o.component === pattern.component && o.metrics.latency > 1000
      )
      .slice(-50);

    const avgLatency =
      slowObs.reduce((sum, o) => sum + o.metrics.latency, 0) / slowObs.length;

    return {
      id: `strategy-${Date.now()}`,
      targetComponent: pattern.component,
      type: "query-optimization",
      currentState: {
        indexesOptimized: false,
        queryCache: false,
        batchingEnabled: false,
      },
      proposedChanges: [
        {
          target: "indexesOptimized",
          currentValue: false,
          proposedValue: true,
          rationale: `Performance degradation detected. Average latency: ${avgLatency.toFixed(0)}ms. Optimize database indexes.`,
        },
        {
          target: "queryCache",
          currentValue: false,
          proposedValue: true,
          rationale: `Enable query result caching for frequently executed queries.`,
        },
        {
          target: "batchingEnabled",
          currentValue: false,
          proposedValue: true,
          rationale: `Enable request batching to reduce round trips.`,
        },
      ],
      expectedImpact: {
        performanceGain: 40,
        resourceSavings: 15,
        reliabilityGain: 10,
        implementationEffort: 12,
      },
      confidence: pattern.confidence * 0.85,
      riskLevel: "low",
    };
  }

  private findCommonErrors(observations: LearningObservation[]): string[] {
    const errorCounts = new Map<string, number>();

    for (const obs of observations) {
      const error = obs.result.error || "Unknown error";
      errorCounts.set(error, (errorCounts.get(error) || 0) + 1);
    }

    return Array.from(errorCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([error]) => error);
  }
}

// ==============================================================================
// Self-Improvement Engine
// ==============================================================================

/**
 * Main self-improvement engine
 *
 * @example
 * ```typescript
 * const engine = new SelfImprovementEngine();
 *
 * // Record observations
 * await engine.recordObservation({
 *   component: 'api',
 *   action: 'query',
 *   context: { endpoint: '/users' },
 *   result: { success: true, duration: 150 },
 *   metrics: { latency: 150 }
 * });
 *
 * // Get improvement suggestions
 * const strategies = await engine.generateStrategies();
 *
 * // Apply approved strategy
 * await engine.applyStrategy(strategies[0]);
 * ```
 */
export class SelfImprovementEngine extends EventEmitter {
  private config: SelfImprovementConfig;
  private observations: LearningObservation[] = [];
  private patternDetector: PatternDetector;
  private modelManager: ModelManager;
  private strategyGenerator: StrategyGenerator;
  private appliedStrategies: ImprovementStrategy[] = [];
  private isLearning = false;

  constructor(config: Partial<SelfImprovementConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.patternDetector = new PatternDetector();
    this.modelManager = new ModelManager();
    this.strategyGenerator = new StrategyGenerator();
  }

  /**
   * Record a learning observation
   */
  async recordObservation(
    input: Omit<LearningObservation, "id" | "timestamp">
  ): Promise<LearningObservation> {
    const observation: LearningObservation = {
      ...input,
      id: `obs-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
      timestamp: new Date(),
    };

    this.observations.push(observation);
    this.emit("learning:observation", { observation });

    // Cleanup old observations
    this.cleanupObservations();

    // Detect patterns
    const patterns = this.patternDetector.analyze(observation);
    for (const pattern of patterns) {
      // Potentially generate strategies for significant patterns
      if (pattern.confidence >= 0.7 && pattern.frequency >= 5) {
        const strategy = this.strategyGenerator.fromPattern(
          pattern,
          this.observations
        );
        if (strategy) {
          this.emit("learning:improvement-suggested", { strategy });
        }
      }
    }

    // Trigger learning if enough observations
    if (
      this.config.enabled &&
      this.observations.length >= this.config.minObservations &&
      !this.isLearning
    ) {
      await this.learn();
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
    const observation = this.observations.find((o) => o.id === observationId);
    if (observation) {
      observation.feedback = feedback;

      // Immediate learning from feedback
      if (feedback.type === "correction" || feedback.rating) {
        const model = this.modelManager.getOrCreateModel(
          observation.component,
          "reinforcement"
        );
        this.modelManager.updateModel(
          model,
          [observation],
          this.config.learningRate * 2
        );
      }
    }
  }

  /**
   * Run learning cycle
   */
  async learn(): Promise<LearningModel[]> {
    if (!this.config.enabled || this.isLearning) {
      return [];
    }

    this.isLearning = true;
    const updatedModels: LearningModel[] = [];

    try {
      // Group observations by component
      const byComponent = new Map<string, LearningObservation[]>();
      for (const obs of this.observations) {
        const existing = byComponent.get(obs.component) || [];
        existing.push(obs);
        byComponent.set(obs.component, existing);
      }

      // Update models for each component
      for (const [component, componentObs] of byComponent) {
        if (componentObs.length >= this.config.minObservations / 2) {
          const model = this.modelManager.getOrCreateModel(
            component,
            "reinforcement"
          );
          const updated = this.modelManager.updateModel(
            model,
            componentObs.slice(-500),
            this.config.learningRate
          );
          updatedModels.push(updated);
          this.emit("learning:model-updated", { model: updated });
        }
      }
    } finally {
      this.isLearning = false;
    }

    return updatedModels;
  }

  /**
   * Generate improvement strategies
   */
  async generateStrategies(): Promise<ImprovementStrategy[]> {
    const strategies: ImprovementStrategy[] = [];
    const patterns = this.patternDetector.getPatterns();

    for (const pattern of patterns) {
      if (pattern.confidence >= 0.6) {
        const strategy = this.strategyGenerator.fromPattern(
          pattern,
          this.observations
        );
        if (strategy) {
          strategies.push(strategy);
        }
      }
    }

    // Sort by expected impact and confidence
    strategies.sort((a, b) => {
      const aScore =
        (a.expectedImpact.performanceGain + a.expectedImpact.reliabilityGain) *
        a.confidence;
      const bScore =
        (b.expectedImpact.performanceGain + b.expectedImpact.reliabilityGain) *
        b.confidence;
      return bScore - aScore;
    });

    return strategies;
  }

  /**
   * Apply an improvement strategy
   */
  async applyStrategy(strategy: ImprovementStrategy): Promise<boolean> {
    // Check risk level against auto-apply settings
    if (this.config.autoApply) {
      const riskOrder: RiskLevel[] = ["low", "medium", "high", "critical"];
      const strategyRiskIndex = riskOrder.indexOf(strategy.riskLevel);
      const maxRiskIndex = riskOrder.indexOf(this.config.maxAutoApplyRisk);

      if (strategyRiskIndex > maxRiskIndex) {
        this.emit("error", {
          error: new Error(
            `Strategy risk level ${strategy.riskLevel} exceeds max auto-apply level ${this.config.maxAutoApplyRisk}`
          ),
          context: "apply-strategy",
        });
        return false;
      }
    }

    // Record the applied strategy
    this.appliedStrategies.push(strategy);

    // Emit event for external handlers to apply changes
    this.emit("learning:improvement-suggested", { strategy });

    return true;
  }

  /**
   * Get current learning models
   */
  getModels(): LearningModel[] {
    return this.modelManager.getAllModels();
  }

  /**
   * Get detected patterns
   */
  getPatterns(): UsagePattern[] {
    return this.patternDetector.getPatterns();
  }

  /**
   * Get observation statistics
   */
  getStats(): {
    totalObservations: number;
    successRate: number;
    avgLatency: number;
    componentBreakdown: Record<string, number>;
  } {
    const total = this.observations.length;
    const successful = this.observations.filter((o) => o.result.success).length;
    const totalLatency = this.observations.reduce(
      (sum, o) => sum + o.metrics.latency,
      0
    );

    const componentBreakdown: Record<string, number> = {};
    for (const obs of this.observations) {
      componentBreakdown[obs.component] =
        (componentBreakdown[obs.component] || 0) + 1;
    }

    return {
      totalObservations: total,
      successRate: total > 0 ? successful / total : 0,
      avgLatency: total > 0 ? totalLatency / total : 0,
      componentBreakdown,
    };
  }

  /**
   * Get applied strategies history
   */
  getAppliedStrategies(): ImprovementStrategy[] {
    return [...this.appliedStrategies];
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SelfImprovementConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): SelfImprovementConfig {
    return { ...this.config };
  }

  private cleanupObservations(): void {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - this.config.retentionDays);

    this.observations = this.observations.filter((o) => o.timestamp >= cutoff);
  }
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create a self-improvement engine instance
 */
export function createSelfImprovementEngine(
  config?: Partial<SelfImprovementConfig>
): SelfImprovementEngine {
  return new SelfImprovementEngine(config);
}

export type { UsagePattern, PatternType };
