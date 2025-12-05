/**
 * NEURECTOMY A/B Testing Engine
 * @module @neurectomy/experimentation-engine/ab-testing
 * @agent @PRISM @FLUX
 *
 * Core A/B testing engine with experiment management,
 * variant assignment, and results tracking.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types
// ============================================================================

export interface ABExperimentConfig {
  id?: string;
  name: string;
  description?: string;
  hypothesis?: string;
  variants: VariantConfig[];
  metrics: MetricConfig[];
  targeting?: TargetingConfig;
  schedule?: ScheduleConfig;
  settings?: ExperimentSettings;
  tags?: string[];
}

export interface VariantConfig {
  id: string;
  name: string;
  description?: string;
  weight: number;
  isControl?: boolean;
  payload?: Record<string, unknown>;
}

export interface MetricConfig {
  id: string;
  name: string;
  type: "conversion" | "count" | "revenue" | "duration" | "custom";
  isPrimary?: boolean;
  goal: "increase" | "decrease";
  minimumDetectableEffect?: number;
}

export interface TargetingConfig {
  rules: TargetingRule[];
  operator: "and" | "or";
  percentage?: number;
}

export interface TargetingRule {
  attribute: string;
  operator:
    | "eq"
    | "neq"
    | "gt"
    | "gte"
    | "lt"
    | "lte"
    | "in"
    | "nin"
    | "contains"
    | "regex";
  value: unknown;
}

export interface ScheduleConfig {
  startDate?: Date;
  endDate?: Date;
  minSampleSize?: number;
  maxDuration?: number;
  autoStop?: boolean;
}

export interface ExperimentSettings {
  confidenceLevel?: number;
  minRuntime?: number;
  maxRuntime?: number;
  enableSequentialTesting?: boolean;
  enablePeeking?: boolean;
}

export interface ABExperiment {
  id: string;
  config: ABExperimentConfig;
  status: ExperimentStatus;
  results: ExperimentResults;
  createdAt: Date;
  startedAt?: Date;
  stoppedAt?: Date;
  updatedAt: Date;
}

export type ExperimentStatus =
  | "draft"
  | "running"
  | "paused"
  | "stopped"
  | "completed";

export interface ExperimentResults {
  totalAssignments: number;
  variantResults: Map<string, VariantResults>;
  winner?: string;
  significance?: number;
  confidenceInterval?: [number, number];
  lastUpdated: Date;
}

export interface VariantResults {
  variantId: string;
  assignments: number;
  conversions: number;
  conversionRate: number;
  metrics: Map<string, MetricResults>;
}

export interface MetricResults {
  metricId: string;
  count: number;
  sum: number;
  mean: number;
  variance: number;
  min: number;
  max: number;
}

export interface Assignment {
  experimentId: string;
  subjectId: string;
  variantId: string;
  timestamp: Date;
  attributes?: Record<string, unknown>;
}

export interface Exposure {
  experimentId: string;
  subjectId: string;
  variantId: string;
  timestamp: Date;
  context?: Record<string, unknown>;
}

export interface Conversion {
  experimentId: string;
  subjectId: string;
  variantId: string;
  metricId: string;
  value: number;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

export interface ABEngineConfig {
  persistAssignments?: boolean;
  enableAnalytics?: boolean;
  batchSize?: number;
  flushInterval?: number;
}

export interface ABEngineEvents {
  experimentCreated: (experiment: ABExperiment) => void;
  experimentStarted: (experiment: ABExperiment) => void;
  experimentStopped: (experiment: ABExperiment, reason: string) => void;
  experimentCompleted: (experiment: ABExperiment) => void;
  assignment: (assignment: Assignment) => void;
  exposure: (exposure: Exposure) => void;
  conversion: (conversion: Conversion) => void;
  winnerDeclared: (experiment: ABExperiment, variantId: string) => void;
}

// ============================================================================
// ABTestingEngine Class
// ============================================================================

export class ABTestingEngine extends EventEmitter<ABEngineEvents> {
  private experiments = new Map<string, ABExperiment>();
  private assignments = new Map<string, Map<string, string>>(); // experimentId -> subjectId -> variantId
  private conversionBuffer: Conversion[] = [];
  private config: Required<ABEngineConfig>;
  private flushTimer?: ReturnType<typeof setInterval>;

  constructor(config: ABEngineConfig = {}) {
    super();
    this.config = {
      persistAssignments: config.persistAssignments ?? true,
      enableAnalytics: config.enableAnalytics ?? true,
      batchSize: config.batchSize ?? 100,
      flushInterval: config.flushInterval ?? 5000,
    };

    if (this.config.flushInterval > 0) {
      this.startFlushTimer();
    }
  }

  // --------------------------------------------------------------------------
  // Experiment Management
  // --------------------------------------------------------------------------

  /**
   * Create a new A/B experiment
   */
  createExperiment(config: ABExperimentConfig): ABExperiment {
    this.validateExperimentConfig(config);

    const experiment: ABExperiment = {
      id: config.id ?? uuidv4(),
      config: {
        ...config,
        settings: {
          confidenceLevel: config.settings?.confidenceLevel ?? 0.95,
          minRuntime: config.settings?.minRuntime ?? 24 * 60 * 60 * 1000, // 24 hours
          maxRuntime: config.settings?.maxRuntime ?? 30 * 24 * 60 * 60 * 1000, // 30 days
          enableSequentialTesting:
            config.settings?.enableSequentialTesting ?? false,
          enablePeeking: config.settings?.enablePeeking ?? true,
        },
      },
      status: "draft",
      results: this.initializeResults(config.variants),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.experiments.set(experiment.id, experiment);
    this.assignments.set(experiment.id, new Map());
    this.emit("experimentCreated", experiment);

    return experiment;
  }

  /**
   * Start an experiment
   */
  startExperiment(experimentId: string): ABExperiment {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${experimentId}`);
    }

    if (experiment.status !== "draft" && experiment.status !== "paused") {
      throw new Error(
        `Cannot start experiment in status: ${experiment.status}`
      );
    }

    experiment.status = "running";
    experiment.startedAt = experiment.startedAt ?? new Date();
    experiment.updatedAt = new Date();

    this.emit("experimentStarted", experiment);

    return experiment;
  }

  /**
   * Pause an experiment
   */
  pauseExperiment(experimentId: string): ABExperiment {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${experimentId}`);
    }

    if (experiment.status !== "running") {
      throw new Error(
        `Cannot pause experiment in status: ${experiment.status}`
      );
    }

    experiment.status = "paused";
    experiment.updatedAt = new Date();

    return experiment;
  }

  /**
   * Stop an experiment
   */
  stopExperiment(
    experimentId: string,
    reason: string = "manual"
  ): ABExperiment {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${experimentId}`);
    }

    experiment.status = "stopped";
    experiment.stoppedAt = new Date();
    experiment.updatedAt = new Date();

    this.emit("experimentStopped", experiment, reason);

    return experiment;
  }

  /**
   * Get experiment by ID
   */
  getExperiment(experimentId: string): ABExperiment | undefined {
    return this.experiments.get(experimentId);
  }

  /**
   * List all experiments
   */
  listExperiments(filter?: {
    status?: ExperimentStatus;
    tags?: string[];
  }): ABExperiment[] {
    let results = Array.from(this.experiments.values());

    if (filter?.status) {
      results = results.filter((e) => e.status === filter.status);
    }

    if (filter?.tags?.length) {
      results = results.filter((e) =>
        filter.tags!.some((tag) => e.config.tags?.includes(tag))
      );
    }

    return results;
  }

  /**
   * Delete an experiment
   */
  deleteExperiment(experimentId: string): boolean {
    this.assignments.delete(experimentId);
    return this.experiments.delete(experimentId);
  }

  // --------------------------------------------------------------------------
  // Assignment
  // --------------------------------------------------------------------------

  /**
   * Get variant assignment for a subject
   */
  getAssignment(
    experimentId: string,
    subjectId: string,
    attributes?: Record<string, unknown>
  ): Assignment | null {
    const experiment = this.experiments.get(experimentId);
    if (!experiment || experiment.status !== "running") {
      return null;
    }

    // Check targeting rules
    if (!this.matchesTargeting(experiment.config.targeting, attributes)) {
      return null;
    }

    // Check for existing assignment
    const experimentAssignments = this.assignments.get(experimentId);
    let variantId = experimentAssignments?.get(subjectId);

    if (!variantId) {
      // Assign to variant
      variantId = this.assignVariant(experiment, subjectId);
      experimentAssignments?.set(subjectId, variantId);

      // Update results
      const variantResults = experiment.results.variantResults.get(variantId);
      if (variantResults) {
        variantResults.assignments++;
        experiment.results.totalAssignments++;
      }
    }

    const assignment: Assignment = {
      experimentId,
      subjectId,
      variantId,
      timestamp: new Date(),
      attributes,
    };

    this.emit("assignment", assignment);

    return assignment;
  }

  /**
   * Record an exposure event
   */
  recordExposure(
    experimentId: string,
    subjectId: string,
    context?: Record<string, unknown>
  ): Exposure | null {
    const assignment = this.getAssignmentRecord(experimentId, subjectId);
    if (!assignment) {
      return null;
    }

    const exposure: Exposure = {
      experimentId,
      subjectId,
      variantId: assignment,
      timestamp: new Date(),
      context,
    };

    this.emit("exposure", exposure);

    return exposure;
  }

  private getAssignmentRecord(
    experimentId: string,
    subjectId: string
  ): string | undefined {
    return this.assignments.get(experimentId)?.get(subjectId);
  }

  private assignVariant(experiment: ABExperiment, subjectId: string): string {
    const variants = experiment.config.variants;
    const totalWeight = variants.reduce((sum, v) => sum + v.weight, 0);

    // Use deterministic hash for consistent assignment
    const hash = this.hashSubject(experiment.id, subjectId);
    const bucket = (hash % 10000) / 10000; // 0-1 range

    let cumulative = 0;
    for (const variant of variants) {
      cumulative += variant.weight / totalWeight;
      if (bucket < cumulative) {
        return variant.id;
      }
    }

    return variants[variants.length - 1].id;
  }

  private hashSubject(experimentId: string, subjectId: string): number {
    const str = `${experimentId}:${subjectId}`;
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  // --------------------------------------------------------------------------
  // Conversions
  // --------------------------------------------------------------------------

  /**
   * Record a conversion event
   */
  recordConversion(
    experimentId: string,
    subjectId: string,
    metricId: string,
    value: number = 1,
    metadata?: Record<string, unknown>
  ): Conversion | null {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      return null;
    }

    const variantId = this.getAssignmentRecord(experimentId, subjectId);
    if (!variantId) {
      return null;
    }

    const conversion: Conversion = {
      experimentId,
      subjectId,
      variantId,
      metricId,
      value,
      timestamp: new Date(),
      metadata,
    };

    this.conversionBuffer.push(conversion);

    // Update results immediately
    this.updateResults(experiment, conversion);

    this.emit("conversion", conversion);

    if (this.conversionBuffer.length >= this.config.batchSize) {
      this.flushConversions();
    }

    return conversion;
  }

  private updateResults(
    experiment: ABExperiment,
    conversion: Conversion
  ): void {
    const variantResults = experiment.results.variantResults.get(
      conversion.variantId
    );
    if (!variantResults) {
      return;
    }

    // Update conversion count for primary metric
    const metric = experiment.config.metrics.find(
      (m) => m.id === conversion.metricId
    );
    if (metric?.type === "conversion") {
      variantResults.conversions++;
      variantResults.conversionRate =
        variantResults.assignments > 0
          ? variantResults.conversions / variantResults.assignments
          : 0;
    }

    // Update metric results
    let metricResults = variantResults.metrics.get(conversion.metricId);
    if (!metricResults) {
      metricResults = {
        metricId: conversion.metricId,
        count: 0,
        sum: 0,
        mean: 0,
        variance: 0,
        min: Infinity,
        max: -Infinity,
      };
      variantResults.metrics.set(conversion.metricId, metricResults);
    }

    // Welford's online algorithm for mean and variance
    const n = metricResults.count + 1;
    const delta = conversion.value - metricResults.mean;
    const newMean = metricResults.mean + delta / n;
    const delta2 = conversion.value - newMean;
    const newVariance =
      (metricResults.variance * (metricResults.count - 1 || 1) +
        delta * delta2) /
      n;

    metricResults.count = n;
    metricResults.sum += conversion.value;
    metricResults.mean = newMean;
    metricResults.variance = newVariance;
    metricResults.min = Math.min(metricResults.min, conversion.value);
    metricResults.max = Math.max(metricResults.max, conversion.value);

    experiment.results.lastUpdated = new Date();
  }

  private flushConversions(): void {
    // In a real implementation, this would persist to storage
    this.conversionBuffer = [];
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flushConversions();
    }, this.config.flushInterval);
  }

  // --------------------------------------------------------------------------
  // Targeting
  // --------------------------------------------------------------------------

  private matchesTargeting(
    targeting: TargetingConfig | undefined,
    attributes: Record<string, unknown> | undefined
  ): boolean {
    if (!targeting || targeting.rules.length === 0) {
      return true;
    }

    if (!attributes) {
      return false;
    }

    // Check percentage
    if (targeting.percentage !== undefined && targeting.percentage < 100) {
      if (Math.random() * 100 > targeting.percentage) {
        return false;
      }
    }

    const results = targeting.rules.map((rule) =>
      this.evaluateRule(rule, attributes)
    );

    return targeting.operator === "and"
      ? results.every((r) => r)
      : results.some((r) => r);
  }

  private evaluateRule(
    rule: TargetingRule,
    attributes: Record<string, unknown>
  ): boolean {
    const value = attributes[rule.attribute];

    switch (rule.operator) {
      case "eq":
        return value === rule.value;
      case "neq":
        return value !== rule.value;
      case "gt":
        return typeof value === "number" && value > (rule.value as number);
      case "gte":
        return typeof value === "number" && value >= (rule.value as number);
      case "lt":
        return typeof value === "number" && value < (rule.value as number);
      case "lte":
        return typeof value === "number" && value <= (rule.value as number);
      case "in":
        return Array.isArray(rule.value) && rule.value.includes(value);
      case "nin":
        return Array.isArray(rule.value) && !rule.value.includes(value);
      case "contains":
        return (
          typeof value === "string" && value.includes(rule.value as string)
        );
      case "regex":
        return (
          typeof value === "string" &&
          new RegExp(rule.value as string).test(value)
        );
      default:
        return false;
    }
  }

  // --------------------------------------------------------------------------
  // Results & Analysis
  // --------------------------------------------------------------------------

  /**
   * Get experiment results
   */
  getResults(experimentId: string): ExperimentResults | undefined {
    return this.experiments.get(experimentId)?.results;
  }

  /**
   * Get variant results
   */
  getVariantResults(
    experimentId: string,
    variantId: string
  ): VariantResults | undefined {
    return this.experiments
      .get(experimentId)
      ?.results.variantResults.get(variantId);
  }

  /**
   * Declare a winner
   */
  declareWinner(experimentId: string, variantId: string): ABExperiment {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${experimentId}`);
    }

    experiment.results.winner = variantId;
    experiment.status = "completed";
    experiment.stoppedAt = new Date();
    experiment.updatedAt = new Date();

    this.emit("winnerDeclared", experiment, variantId);
    this.emit("experimentCompleted", experiment);

    return experiment;
  }

  // --------------------------------------------------------------------------
  // Validation
  // --------------------------------------------------------------------------

  private validateExperimentConfig(config: ABExperimentConfig): void {
    if (!config.name) {
      throw new Error("Experiment name is required");
    }

    if (!config.variants || config.variants.length < 2) {
      throw new Error("At least 2 variants are required");
    }

    const controlCount = config.variants.filter((v) => v.isControl).length;
    if (controlCount > 1) {
      throw new Error("Only one control variant is allowed");
    }

    if (!config.metrics || config.metrics.length === 0) {
      throw new Error("At least one metric is required");
    }

    const totalWeight = config.variants.reduce((sum, v) => sum + v.weight, 0);
    if (totalWeight <= 0) {
      throw new Error("Total variant weight must be positive");
    }
  }

  private initializeResults(variants: VariantConfig[]): ExperimentResults {
    const variantResults = new Map<string, VariantResults>();

    for (const variant of variants) {
      variantResults.set(variant.id, {
        variantId: variant.id,
        assignments: 0,
        conversions: 0,
        conversionRate: 0,
        metrics: new Map(),
      });
    }

    return {
      totalAssignments: 0,
      variantResults,
      lastUpdated: new Date(),
    };
  }

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------

  /**
   * Dispose engine resources
   */
  dispose(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flushConversions();
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new ABTestingEngine instance
 */
export function createABEngine(config?: ABEngineConfig): ABTestingEngine {
  return new ABTestingEngine(config);
}

/**
 * Create a simple A/B experiment configuration
 */
export function defineABExperiment(
  name: string,
  controlName: string,
  treatmentName: string,
  primaryMetric: string,
  options?: Partial<ABExperimentConfig>
): ABExperimentConfig {
  return {
    name,
    variants: [
      { id: "control", name: controlName, weight: 50, isControl: true },
      { id: "treatment", name: treatmentName, weight: 50 },
    ],
    metrics: [
      {
        id: primaryMetric,
        name: primaryMetric,
        type: "conversion",
        isPrimary: true,
        goal: "increase",
      },
    ],
    ...options,
  };
}
