/**
 * NEURECTOMY Experimentation Engine Types
 *
 * @module @neurectomy/experimentation-engine/types
 *
 * Comprehensive type definitions for A/B testing, multi-armed bandits,
 * ML experiments, and statistical analysis.
 */

import { z } from "zod";

// ============================================================================
// Core Types
// ============================================================================

/**
 * Experiment status
 */
export type ExperimentStatus =
  | "draft"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "archived";

/**
 * Experiment type
 */
export type ExperimentType =
  | "ab_test"
  | "multivariate"
  | "multi_armed_bandit"
  | "bayesian_optimization"
  | "factorial"
  | "contextual_bandit"
  | "evolutionary";

/**
 * Statistical significance method
 */
export type SignificanceMethod = "frequentist" | "bayesian" | "bootstrap";

/**
 * Assignment strategy for variants
 */
export type AssignmentStrategy =
  | "random"
  | "deterministic"
  | "sticky"
  | "weighted"
  | "thompson_sampling"
  | "ucb"
  | "epsilon_greedy";

/**
 * Metric type
 */
export type MetricType =
  | "count"
  | "sum"
  | "mean"
  | "ratio"
  | "percentile"
  | "conversion"
  | "revenue"
  | "duration"
  | "custom";

/**
 * Metric goal direction
 */
export type MetricGoal = "increase" | "decrease" | "maintain";

// ============================================================================
// Zod Validation Schemas
// ============================================================================

export const MetricDefinitionSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  type: z.enum([
    "count",
    "sum",
    "mean",
    "ratio",
    "percentile",
    "conversion",
    "revenue",
    "duration",
    "custom",
  ]),
  goal: z.enum(["increase", "decrease", "maintain"]),
  unit: z.string().optional(),
  minimumDetectableEffect: z.number().positive().optional(),
  baseline: z.number().optional(),
  isPrimary: z.boolean().default(false),
  aggregation: z
    .object({
      method: z.enum([
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "percentile",
        "count",
      ]),
      percentile: z.number().min(0).max(100).optional(),
    })
    .optional(),
});

export const VariantSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  isControl: z.boolean().default(false),
  weight: z.number().min(0).max(100).default(50),
  config: z.record(z.unknown()).optional(),
  metadata: z.record(z.unknown()).optional(),
});

export const TargetingRuleSchema = z.object({
  id: z.string(),
  attribute: z.string(),
  operator: z.enum([
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "starts_with",
    "ends_with",
    "regex",
    "in",
    "not_in",
    "greater_than",
    "less_than",
    "between",
    "exists",
    "not_exists",
  ]),
  value: z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]),
});

export const ExperimentConfigSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  type: z.enum([
    "ab_test",
    "multivariate",
    "multi_armed_bandit",
    "bayesian_optimization",
    "factorial",
    "contextual_bandit",
    "evolutionary",
  ]),
  status: z
    .enum(["draft", "running", "paused", "completed", "failed", "archived"])
    .default("draft"),
  variants: z.array(VariantSchema).min(2),
  metrics: z.array(MetricDefinitionSchema).min(1),
  targeting: z
    .object({
      rules: z.array(TargetingRuleSchema),
      operator: z.enum(["and", "or"]).default("and"),
      percentage: z.number().min(0).max(100).default(100),
    })
    .optional(),
  assignment: z
    .object({
      strategy: z
        .enum([
          "random",
          "deterministic",
          "sticky",
          "weighted",
          "thompson_sampling",
          "ucb",
          "epsilon_greedy",
        ])
        .default("random"),
      stickyProperty: z.string().optional(),
      seed: z.string().optional(),
      epsilon: z.number().min(0).max(1).optional(),
      ucbAlpha: z.number().positive().optional(),
    })
    .optional(),
  analysis: z
    .object({
      method: z
        .enum(["frequentist", "bayesian", "bootstrap"])
        .default("frequentist"),
      confidenceLevel: z.number().min(0.5).max(0.99).default(0.95),
      minimumSampleSize: z.number().int().positive().optional(),
      maximumDuration: z.number().positive().optional(),
      sequentialTesting: z.boolean().default(false),
      peekingCorrection: z.boolean().default(true),
    })
    .optional(),
  schedule: z
    .object({
      startTime: z.date().optional(),
      endTime: z.date().optional(),
      autoStop: z.boolean().default(false),
      autoStopCondition: z
        .enum([
          "significance_reached",
          "sample_size_reached",
          "duration_reached",
        ])
        .optional(),
    })
    .optional(),
  tags: z.array(z.string()).optional(),
  owner: z.string().optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export const ExperimentEventSchema = z.object({
  experimentId: z.string(),
  variantId: z.string(),
  userId: z.string().optional(),
  sessionId: z.string().optional(),
  eventType: z.enum(["exposure", "conversion", "metric"]),
  metricId: z.string().optional(),
  value: z.number().optional(),
  timestamp: z.date(),
  properties: z.record(z.unknown()).optional(),
  context: z
    .object({
      userAgent: z.string().optional(),
      ip: z.string().optional(),
      country: z.string().optional(),
      device: z.string().optional(),
      platform: z.string().optional(),
    })
    .optional(),
});

export const VariantResultSchema = z.object({
  variantId: z.string(),
  sampleSize: z.number().int().nonnegative(),
  conversions: z.number().int().nonnegative().optional(),
  metrics: z.record(
    z.object({
      value: z.number(),
      standardError: z.number().optional(),
      confidenceInterval: z.tuple([z.number(), z.number()]).optional(),
      samples: z.number().int().nonnegative(),
    })
  ),
});

export const ExperimentResultSchema = z.object({
  experimentId: z.string(),
  status: z.enum(["preliminary", "conclusive", "inconclusive"]),
  startTime: z.date(),
  endTime: z.date().optional(),
  totalSampleSize: z.number().int().nonnegative(),
  variants: z.array(VariantResultSchema),
  winner: z.string().optional(),
  primaryMetric: z
    .object({
      metricId: z.string(),
      controlValue: z.number(),
      treatmentValue: z.number(),
      absoluteEffect: z.number(),
      relativeEffect: z.number(),
      pValue: z.number().optional(),
      posteriorProbability: z.number().optional(),
      confidenceInterval: z.tuple([z.number(), z.number()]).optional(),
      credibleInterval: z.tuple([z.number(), z.number()]).optional(),
      isSignificant: z.boolean(),
      powerAnalysis: z
        .object({
          observedPower: z.number(),
          requiredSampleSize: z.number().int().positive(),
        })
        .optional(),
    })
    .optional(),
  segmentAnalysis: z
    .array(
      z.object({
        segment: z.string(),
        results: z.record(z.unknown()),
      })
    )
    .optional(),
  recommendations: z.array(z.string()).optional(),
});

export const HyperparameterSchema = z.object({
  name: z.string(),
  type: z.enum(["continuous", "discrete", "categorical"]),
  range: z.union([
    z.object({
      min: z.number(),
      max: z.number(),
      scale: z.enum(["linear", "log"]).optional(),
    }),
    z.object({
      values: z.array(z.union([z.string(), z.number(), z.boolean()])),
    }),
  ]),
  default: z.union([z.string(), z.number(), z.boolean()]).optional(),
});

export const MLExperimentConfigSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  modelType: z.string(),
  hyperparameters: z.array(HyperparameterSchema),
  objective: z.object({
    metric: z.string(),
    goal: z.enum(["minimize", "maximize"]),
  }),
  searchStrategy: z
    .enum(["grid", "random", "bayesian", "hyperband", "evolutionary"])
    .default("bayesian"),
  maxTrials: z.number().int().positive(),
  maxParallelTrials: z.number().int().positive().default(1),
  earlyStoppingRounds: z.number().int().positive().optional(),
  crossValidation: z
    .object({
      enabled: z.boolean().default(true),
      folds: z.number().int().min(2).default(5),
      stratified: z.boolean().default(true),
    })
    .optional(),
  status: z
    .enum(["draft", "running", "paused", "completed", "failed"])
    .default("draft"),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export const MLTrialSchema = z.object({
  id: z.string(),
  experimentId: z.string(),
  hyperparameters: z.record(z.union([z.string(), z.number(), z.boolean()])),
  metrics: z.record(z.number()),
  status: z.enum(["pending", "running", "completed", "failed", "pruned"]),
  startTime: z.date().optional(),
  endTime: z.date().optional(),
  error: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
});

// ============================================================================
// Type Exports
// ============================================================================

export type MetricDefinition = z.infer<typeof MetricDefinitionSchema>;
export type Variant = z.infer<typeof VariantSchema>;
export type TargetingRule = z.infer<typeof TargetingRuleSchema>;
export type ExperimentConfig = z.infer<typeof ExperimentConfigSchema>;
export type ExperimentEvent = z.infer<typeof ExperimentEventSchema>;
export type VariantResult = z.infer<typeof VariantResultSchema>;
export type ExperimentResult = z.infer<typeof ExperimentResultSchema>;
export type Hyperparameter = z.infer<typeof HyperparameterSchema>;
export type MLExperimentConfig = z.infer<typeof MLExperimentConfigSchema>;
export type MLTrial = z.infer<typeof MLTrialSchema>;

// ============================================================================
// Statistical Types
// ============================================================================

export interface StatisticalTestResult {
  testName: string;
  statistic: number;
  pValue: number;
  degreesOfFreedom?: number;
  confidenceInterval?: [number, number];
  effectSize?: number;
  isSignificant: boolean;
}

export interface BayesianResult {
  posteriorMean: number;
  posteriorStd: number;
  credibleInterval: [number, number];
  probabilityOfBeingBest: number;
  expectedLoss: number;
}

export interface PowerAnalysis {
  effect: number;
  sampleSize: number;
  alpha: number;
  power: number;
  minimumDetectableEffect: number;
}

export interface SampleSizeEstimate {
  perVariant: number;
  total: number;
  duration: number;
  assumptions: {
    baselineRate: number;
    minimumEffect: number;
    alpha: number;
    power: number;
    dailyTraffic: number;
  };
}

// ============================================================================
// Bandit Types
// ============================================================================

export interface BanditState {
  arms: Array<{
    id: string;
    pulls: number;
    rewards: number;
    mean: number;
    ucbScore?: number;
    thompsonSample?: number;
  }>;
  totalPulls: number;
  bestArm?: string;
  regret: number;
}

export interface BanditConfig {
  algorithm: "thompson_sampling" | "ucb" | "epsilon_greedy" | "exp3";
  epsilon?: number;
  ucbAlpha?: number;
  decayRate?: number;
  priorAlpha?: number;
  priorBeta?: number;
}

// ============================================================================
// Optimization Types
// ============================================================================

export interface OptimizationState {
  bestTrial: MLTrial | null;
  bestValue: number;
  trials: MLTrial[];
  searchSpace: Hyperparameter[];
  completedTrials: number;
  runningTrials: number;
  pendingTrials: number;
}

export interface AcquisitionFunction {
  type: "ei" | "pi" | "ucb" | "lcb";
  xi?: number;
  kappa?: number;
}

export interface SurrogateModel {
  type: "gaussian_process" | "random_forest" | "tpe";
  kernel?: string;
  lengthScale?: number;
}
