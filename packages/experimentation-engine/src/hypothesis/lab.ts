/**
 * NEURECTOMY Hypothesis Lab
 * @module @neurectomy/experimentation-engine/hypothesis
 * @agent @PRISM @TENSOR
 *
 * Core ML experiment management with hypothesis tracking,
 * parameter optimization, and result analysis.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import type { ExperimentStatus, MetricGoal } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface HypothesisConfig {
  id?: string;
  name: string;
  description?: string;
  hypothesis: string;
  nullHypothesis?: string;
  expectedOutcome?: string;
  confidenceLevel?: number;
  powerLevel?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface Parameter {
  name: string;
  type: "continuous" | "discrete" | "categorical" | "boolean";
  min?: number;
  max?: number;
  values?: (string | number | boolean)[];
  default?: string | number | boolean;
  description?: string;
}

export interface ParameterSpace {
  parameters: Parameter[];
  constraints?: ParameterConstraint[];
}

export interface ParameterConstraint {
  type: "dependency" | "exclusion" | "sum" | "custom";
  parameters: string[];
  condition?: string;
  value?: number;
}

export interface TrialConfig {
  id?: string;
  hypothesisId: string;
  parameters: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface TrialResult {
  trialId: string;
  metrics: Record<string, number>;
  artifacts?: Record<string, string>;
  duration: number;
  status: "success" | "failed" | "timeout";
  error?: string;
  timestamp: Date;
}

export interface Hypothesis {
  id: string;
  config: HypothesisConfig;
  parameterSpace: ParameterSpace;
  trials: Trial[];
  bestTrial?: Trial;
  status: ExperimentStatus;
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
}

export interface Trial {
  id: string;
  hypothesisId: string;
  parameters: Record<string, unknown>;
  result?: TrialResult;
  status: "pending" | "running" | "completed" | "failed";
  startedAt?: Date;
  completedAt?: Date;
  metadata?: Record<string, unknown>;
}

export interface LabConfig {
  maxConcurrentTrials?: number;
  defaultTimeout?: number;
  autoSave?: boolean;
  saveInterval?: number;
  storageBackend?: StorageBackend;
}

export interface StorageBackend {
  save(key: string, data: unknown): Promise<void>;
  load(key: string): Promise<unknown | null>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
}

export interface LabEvents {
  hypothesisCreated: (hypothesis: Hypothesis) => void;
  hypothesisUpdated: (hypothesis: Hypothesis) => void;
  hypothesisCompleted: (hypothesis: Hypothesis) => void;
  trialStarted: (trial: Trial) => void;
  trialCompleted: (trial: Trial, result: TrialResult) => void;
  trialFailed: (trial: Trial, error: Error) => void;
  newBestTrial: (hypothesis: Hypothesis, trial: Trial) => void;
  error: (error: Error) => void;
}

// ============================================================================
// In-Memory Storage Backend
// ============================================================================

class InMemoryStorage implements StorageBackend {
  private store = new Map<string, unknown>();

  async save(key: string, data: unknown): Promise<void> {
    this.store.set(key, JSON.parse(JSON.stringify(data)));
  }

  async load(key: string): Promise<unknown | null> {
    const data = this.store.get(key);
    return data ? JSON.parse(JSON.stringify(data)) : null;
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async list(prefix: string): Promise<string[]> {
    return Array.from(this.store.keys()).filter((k) => k.startsWith(prefix));
  }
}

// ============================================================================
// HypothesisLab Class
// ============================================================================

export class HypothesisLab extends EventEmitter<LabEvents> {
  private hypotheses = new Map<string, Hypothesis>();
  private runningTrials = new Map<string, Trial>();
  private config: Required<LabConfig>;
  private storage: StorageBackend;
  private saveTimer?: ReturnType<typeof setInterval>;

  constructor(config: LabConfig = {}) {
    super();
    this.config = {
      maxConcurrentTrials: config.maxConcurrentTrials ?? 5,
      defaultTimeout: config.defaultTimeout ?? 3600000, // 1 hour
      autoSave: config.autoSave ?? true,
      saveInterval: config.saveInterval ?? 60000, // 1 minute
      storageBackend: config.storageBackend ?? new InMemoryStorage(),
    };
    this.storage = this.config.storageBackend;

    if (this.config.autoSave) {
      this.startAutoSave();
    }
  }

  // --------------------------------------------------------------------------
  // Hypothesis Management
  // --------------------------------------------------------------------------

  /**
   * Create a new hypothesis
   */
  createHypothesis(
    config: HypothesisConfig,
    parameterSpace: ParameterSpace
  ): Hypothesis {
    const hypothesis: Hypothesis = {
      id: config.id ?? uuidv4(),
      config: {
        ...config,
        confidenceLevel: config.confidenceLevel ?? 0.95,
        powerLevel: config.powerLevel ?? 0.8,
      },
      parameterSpace,
      trials: [],
      status: "draft",
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.validateParameterSpace(parameterSpace);
    this.hypotheses.set(hypothesis.id, hypothesis);
    this.emit("hypothesisCreated", hypothesis);

    return hypothesis;
  }

  /**
   * Get hypothesis by ID
   */
  getHypothesis(id: string): Hypothesis | undefined {
    return this.hypotheses.get(id);
  }

  /**
   * List all hypotheses
   */
  listHypotheses(filter?: {
    status?: ExperimentStatus;
    tags?: string[];
  }): Hypothesis[] {
    let results = Array.from(this.hypotheses.values());

    if (filter?.status) {
      results = results.filter((h) => h.status === filter.status);
    }

    if (filter?.tags?.length) {
      results = results.filter((h) =>
        filter.tags!.some((tag) => h.config.tags?.includes(tag))
      );
    }

    return results;
  }

  /**
   * Update hypothesis status
   */
  updateHypothesisStatus(id: string, status: ExperimentStatus): void {
    const hypothesis = this.hypotheses.get(id);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${id}`);
    }

    hypothesis.status = status;
    hypothesis.updatedAt = new Date();

    if (status === "completed") {
      hypothesis.completedAt = new Date();
      this.emit("hypothesisCompleted", hypothesis);
    }

    this.emit("hypothesisUpdated", hypothesis);
  }

  /**
   * Delete hypothesis
   */
  deleteHypothesis(id: string): boolean {
    return this.hypotheses.delete(id);
  }

  // --------------------------------------------------------------------------
  // Trial Management
  // --------------------------------------------------------------------------

  /**
   * Create and start a new trial
   */
  async createTrial(config: TrialConfig): Promise<Trial> {
    const hypothesis = this.hypotheses.get(config.hypothesisId);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${config.hypothesisId}`);
    }

    if (this.runningTrials.size >= this.config.maxConcurrentTrials) {
      throw new Error(
        `Max concurrent trials reached: ${this.config.maxConcurrentTrials}`
      );
    }

    this.validateParameters(config.parameters, hypothesis.parameterSpace);

    const trial: Trial = {
      id: config.id ?? uuidv4(),
      hypothesisId: config.hypothesisId,
      parameters: config.parameters,
      status: "pending",
      metadata: config.metadata,
    };

    hypothesis.trials.push(trial);
    hypothesis.updatedAt = new Date();

    return trial;
  }

  /**
   * Start a trial
   */
  startTrial(trialId: string, hypothesisId: string): Trial {
    const hypothesis = this.hypotheses.get(hypothesisId);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${hypothesisId}`);
    }

    const trial = hypothesis.trials.find((t) => t.id === trialId);
    if (!trial) {
      throw new Error(`Trial not found: ${trialId}`);
    }

    trial.status = "running";
    trial.startedAt = new Date();
    this.runningTrials.set(trialId, trial);
    this.emit("trialStarted", trial);

    return trial;
  }

  /**
   * Complete a trial with results
   */
  completeTrial(
    trialId: string,
    hypothesisId: string,
    result: Omit<TrialResult, "trialId" | "timestamp">
  ): Trial {
    const hypothesis = this.hypotheses.get(hypothesisId);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${hypothesisId}`);
    }

    const trial = hypothesis.trials.find((t) => t.id === trialId);
    if (!trial) {
      throw new Error(`Trial not found: ${trialId}`);
    }

    const fullResult: TrialResult = {
      ...result,
      trialId,
      timestamp: new Date(),
    };

    trial.result = fullResult;
    trial.status = result.status === "success" ? "completed" : "failed";
    trial.completedAt = new Date();
    this.runningTrials.delete(trialId);

    // Check if this is the best trial
    if (result.status === "success") {
      this.updateBestTrial(hypothesis, trial);
    }

    hypothesis.updatedAt = new Date();
    this.emit("trialCompleted", trial, fullResult);

    return trial;
  }

  /**
   * Fail a trial
   */
  failTrial(trialId: string, hypothesisId: string, error: Error): Trial {
    const hypothesis = this.hypotheses.get(hypothesisId);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${hypothesisId}`);
    }

    const trial = hypothesis.trials.find((t) => t.id === trialId);
    if (!trial) {
      throw new Error(`Trial not found: ${trialId}`);
    }

    trial.status = "failed";
    trial.completedAt = new Date();
    trial.result = {
      trialId,
      metrics: {},
      duration: trial.startedAt ? Date.now() - trial.startedAt.getTime() : 0,
      status: "failed",
      error: error.message,
      timestamp: new Date(),
    };

    this.runningTrials.delete(trialId);
    hypothesis.updatedAt = new Date();
    this.emit("trialFailed", trial, error);

    return trial;
  }

  /**
   * Get running trials
   */
  getRunningTrials(): Trial[] {
    return Array.from(this.runningTrials.values());
  }

  // --------------------------------------------------------------------------
  // Parameter Suggestions
  // --------------------------------------------------------------------------

  /**
   * Suggest next parameters based on previous trials
   */
  suggestNextParameters(
    hypothesisId: string,
    strategy: "random" | "grid" | "bayesian" = "random"
  ): Record<string, unknown> {
    const hypothesis = this.hypotheses.get(hypothesisId);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${hypothesisId}`);
    }

    switch (strategy) {
      case "random":
        return this.randomSample(hypothesis.parameterSpace);
      case "grid":
        return this.gridSample(hypothesis);
      case "bayesian":
        return this.bayesianSample(hypothesis);
      default:
        return this.randomSample(hypothesis.parameterSpace);
    }
  }

  private randomSample(space: ParameterSpace): Record<string, unknown> {
    const params: Record<string, unknown> = {};

    for (const param of space.parameters) {
      params[param.name] = this.sampleParameter(param);
    }

    return params;
  }

  private sampleParameter(param: Parameter): unknown {
    switch (param.type) {
      case "continuous":
        return (
          (param.min ?? 0) +
          Math.random() * ((param.max ?? 1) - (param.min ?? 0))
        );
      case "discrete":
        const min = Math.ceil(param.min ?? 0);
        const max = Math.floor(param.max ?? 10);
        return Math.floor(Math.random() * (max - min + 1)) + min;
      case "categorical":
        const values = param.values ?? [];
        return values[Math.floor(Math.random() * values.length)];
      case "boolean":
        return Math.random() > 0.5;
      default:
        return param.default;
    }
  }

  private gridSample(hypothesis: Hypothesis): Record<string, unknown> {
    // Simple grid sampling - find unexplored combinations
    const triedParams = hypothesis.trials.map((t) =>
      JSON.stringify(t.parameters)
    );
    const space = hypothesis.parameterSpace;

    // Generate grid points
    const gridPoints = this.generateGridPoints(space, 5);

    // Find first unexplored point
    for (const point of gridPoints) {
      if (!triedParams.includes(JSON.stringify(point))) {
        return point;
      }
    }

    // Fallback to random if all grid points explored
    return this.randomSample(space);
  }

  private generateGridPoints(
    space: ParameterSpace,
    resolution: number
  ): Record<string, unknown>[] {
    const points: Record<string, unknown>[] = [{}];

    for (const param of space.parameters) {
      const newPoints: Record<string, unknown>[] = [];
      const paramValues = this.getGridValues(param, resolution);

      for (const point of points) {
        for (const value of paramValues) {
          newPoints.push({ ...point, [param.name]: value });
        }
      }

      points.length = 0;
      points.push(...newPoints);
    }

    return points;
  }

  private getGridValues(param: Parameter, resolution: number): unknown[] {
    switch (param.type) {
      case "continuous":
        const step = ((param.max ?? 1) - (param.min ?? 0)) / (resolution - 1);
        return Array.from(
          { length: resolution },
          (_, i) => (param.min ?? 0) + i * step
        );
      case "discrete":
        const min = param.min ?? 0;
        const max = param.max ?? 10;
        const range = max - min;
        const discreteStep = Math.max(1, Math.floor(range / (resolution - 1)));
        return Array.from(
          { length: Math.min(resolution, range + 1) },
          (_, i) => min + i * discreteStep
        );
      case "categorical":
        return param.values ?? [];
      case "boolean":
        return [true, false];
      default:
        return [param.default];
    }
  }

  private bayesianSample(hypothesis: Hypothesis): Record<string, unknown> {
    // Simplified Bayesian optimization using acquisition function
    const completedTrials = hypothesis.trials.filter(
      (t) => t.status === "completed" && t.result
    );

    if (completedTrials.length < 3) {
      // Not enough data, use random
      return this.randomSample(hypothesis.parameterSpace);
    }

    // Use Upper Confidence Bound (UCB) acquisition
    const candidates = Array.from({ length: 100 }, () =>
      this.randomSample(hypothesis.parameterSpace)
    );

    let bestCandidate = candidates[0];
    let bestAcquisition = -Infinity;

    for (const candidate of candidates) {
      const acquisition = this.calculateUCB(candidate, completedTrials);
      if (acquisition > bestAcquisition) {
        bestAcquisition = acquisition;
        bestCandidate = candidate;
      }
    }

    return bestCandidate;
  }

  private calculateUCB(
    params: Record<string, unknown>,
    trials: Trial[]
  ): number {
    // Simplified UCB calculation
    const distances = trials.map((t) =>
      this.parameterDistance(params, t.parameters)
    );
    const nearestIdx = distances.indexOf(Math.min(...distances));
    const nearestTrial = trials[nearestIdx];

    if (!nearestTrial.result) return 0;

    const primaryMetric = Object.values(nearestTrial.result.metrics)[0] ?? 0;
    const exploration =
      2 * Math.sqrt(Math.log(trials.length + 1) / (nearestIdx + 1));

    return primaryMetric + exploration;
  }

  private parameterDistance(
    a: Record<string, unknown>,
    b: Record<string, unknown>
  ): number {
    let distance = 0;
    for (const key of Object.keys(a)) {
      const va = a[key];
      const vb = b[key];
      if (typeof va === "number" && typeof vb === "number") {
        distance += Math.pow(va - vb, 2);
      } else if (va !== vb) {
        distance += 1;
      }
    }
    return Math.sqrt(distance);
  }

  // --------------------------------------------------------------------------
  // Validation
  // --------------------------------------------------------------------------

  private validateParameterSpace(space: ParameterSpace): void {
    for (const param of space.parameters) {
      if (!param.name) {
        throw new Error("Parameter must have a name");
      }

      if (param.type === "continuous" || param.type === "discrete") {
        if (param.min !== undefined && param.max !== undefined) {
          if (param.min > param.max) {
            throw new Error(
              `Invalid range for parameter ${param.name}: min > max`
            );
          }
        }
      }

      if (param.type === "categorical" && !param.values?.length) {
        throw new Error(`Categorical parameter ${param.name} must have values`);
      }
    }
  }

  private validateParameters(
    params: Record<string, unknown>,
    space: ParameterSpace
  ): void {
    for (const param of space.parameters) {
      const value = params[param.name];

      if (value === undefined && param.default === undefined) {
        throw new Error(`Missing required parameter: ${param.name}`);
      }

      if (value !== undefined) {
        switch (param.type) {
          case "continuous":
          case "discrete":
            if (typeof value !== "number") {
              throw new Error(`Parameter ${param.name} must be a number`);
            }
            if (param.min !== undefined && value < param.min) {
              throw new Error(
                `Parameter ${param.name} below minimum: ${value} < ${param.min}`
              );
            }
            if (param.max !== undefined && value > param.max) {
              throw new Error(
                `Parameter ${param.name} above maximum: ${value} > ${param.max}`
              );
            }
            break;
          case "categorical":
            if (!param.values?.includes(value as string | number | boolean)) {
              throw new Error(
                `Invalid value for parameter ${param.name}: ${value}`
              );
            }
            break;
          case "boolean":
            if (typeof value !== "boolean") {
              throw new Error(`Parameter ${param.name} must be boolean`);
            }
            break;
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // Best Trial Tracking
  // --------------------------------------------------------------------------

  private updateBestTrial(hypothesis: Hypothesis, trial: Trial): void {
    if (!trial.result) return;

    const primaryMetricKey = Object.keys(trial.result.metrics)[0];
    if (!primaryMetricKey) return;

    const currentValue = trial.result.metrics[primaryMetricKey];

    if (!hypothesis.bestTrial?.result) {
      hypothesis.bestTrial = trial;
      this.emit("newBestTrial", hypothesis, trial);
      return;
    }

    const bestValue = hypothesis.bestTrial.result.metrics[primaryMetricKey];

    // Assume higher is better by default
    if (currentValue > bestValue) {
      hypothesis.bestTrial = trial;
      this.emit("newBestTrial", hypothesis, trial);
    }
  }

  // --------------------------------------------------------------------------
  // Persistence
  // --------------------------------------------------------------------------

  private startAutoSave(): void {
    this.saveTimer = setInterval(async () => {
      await this.saveAll();
    }, this.config.saveInterval);
  }

  /**
   * Save all hypotheses to storage
   */
  async saveAll(): Promise<void> {
    const hypotheses = Array.from(this.hypotheses.entries());
    for (const [id, hypothesis] of hypotheses) {
      await this.storage.save(`hypothesis:${id}`, hypothesis);
    }
  }

  /**
   * Load hypotheses from storage
   */
  async loadAll(): Promise<void> {
    const keys = await this.storage.list("hypothesis:");
    for (const key of keys) {
      const data = await this.storage.load(key);
      if (data) {
        const hypothesis = data as Hypothesis;
        this.hypotheses.set(hypothesis.id, hypothesis);
      }
    }
  }

  /**
   * Export hypothesis to JSON
   */
  exportHypothesis(id: string): string {
    const hypothesis = this.hypotheses.get(id);
    if (!hypothesis) {
      throw new Error(`Hypothesis not found: ${id}`);
    }
    return JSON.stringify(hypothesis, null, 2);
  }

  /**
   * Import hypothesis from JSON
   */
  importHypothesis(json: string): Hypothesis {
    const data = JSON.parse(json) as Hypothesis;
    data.createdAt = new Date(data.createdAt);
    data.updatedAt = new Date(data.updatedAt);
    if (data.completedAt) {
      data.completedAt = new Date(data.completedAt);
    }
    this.hypotheses.set(data.id, data);
    return data;
  }

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------

  /**
   * Dispose of the lab and clean up resources
   */
  dispose(): void {
    if (this.saveTimer) {
      clearInterval(this.saveTimer);
    }
    this.runningTrials.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new HypothesisLab instance
 */
export function createHypothesisLab(config?: LabConfig): HypothesisLab {
  return new HypothesisLab(config);
}

/**
 * Create a simple hypothesis configuration
 */
export function defineHypothesis(
  name: string,
  hypothesis: string,
  options?: Partial<HypothesisConfig>
): HypothesisConfig {
  return {
    name,
    hypothesis,
    ...options,
  };
}

/**
 * Create a parameter space definition
 */
export function defineParameterSpace(
  parameters: Parameter[],
  constraints?: ParameterConstraint[]
): ParameterSpace {
  return { parameters, constraints };
}
