/**
 * NEURECTOMY Experiment Tracker
 * @module @neurectomy/experimentation-engine/hypothesis
 * @agent @PRISM @TENSOR
 *
 * Tracks experiment runs, metrics, and artifacts with
 * comprehensive logging and comparison capabilities.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types
// ============================================================================

export interface RunConfig {
  id?: string;
  experimentId: string;
  name?: string;
  parameters: Record<string, unknown>;
  tags?: string[];
  description?: string;
  parentRunId?: string;
}

export interface Run {
  id: string;
  experimentId: string;
  name: string;
  parameters: Record<string, unknown>;
  metrics: MetricLog[];
  artifacts: Artifact[];
  tags: string[];
  description?: string;
  parentRunId?: string;
  childRunIds: string[];
  status: RunStatus;
  startTime: Date;
  endTime?: Date;
  duration?: number;
}

export type RunStatus = "running" | "completed" | "failed" | "killed";

export interface MetricLog {
  key: string;
  value: number;
  timestamp: Date;
  step?: number;
}

export interface Artifact {
  id: string;
  name: string;
  type: ArtifactType;
  path: string;
  size?: number;
  metadata?: Record<string, unknown>;
  createdAt: Date;
}

export type ArtifactType =
  | "model"
  | "dataset"
  | "plot"
  | "log"
  | "config"
  | "checkpoint"
  | "other";

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  tags: string[];
  runs: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface TrackerConfig {
  maxMetricsPerRun?: number;
  maxArtifactsPerRun?: number;
  enableAutoLogging?: boolean;
  flushInterval?: number;
}

export interface TrackerEvents {
  experimentCreated: (experiment: Experiment) => void;
  runStarted: (run: Run) => void;
  runCompleted: (run: Run) => void;
  runFailed: (run: Run, error: Error) => void;
  metricLogged: (runId: string, metric: MetricLog) => void;
  artifactLogged: (runId: string, artifact: Artifact) => void;
}

export interface ComparisonResult {
  runs: Run[];
  metrics: MetricComparison[];
  parameters: ParameterComparison[];
  bestRun?: Run;
  ranking: RunRanking[];
}

export interface MetricComparison {
  key: string;
  values: { runId: string; value: number; delta?: number }[];
  min: number;
  max: number;
  mean: number;
  stdDev: number;
}

export interface ParameterComparison {
  key: string;
  values: { runId: string; value: unknown }[];
  uniqueValues: unknown[];
}

export interface RunRanking {
  runId: string;
  rank: number;
  score: number;
  metrics: Record<string, number>;
}

// ============================================================================
// ExperimentTracker Class
// ============================================================================

export class ExperimentTracker extends EventEmitter<TrackerEvents> {
  private experiments = new Map<string, Experiment>();
  private runs = new Map<string, Run>();
  private activeRun: Run | null = null;
  private config: Required<TrackerConfig>;
  private metricBuffer: MetricLog[] = [];
  private flushTimer?: ReturnType<typeof setInterval>;

  constructor(config: TrackerConfig = {}) {
    super();
    this.config = {
      maxMetricsPerRun: config.maxMetricsPerRun ?? 10000,
      maxArtifactsPerRun: config.maxArtifactsPerRun ?? 100,
      enableAutoLogging: config.enableAutoLogging ?? false,
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
   * Create or get an experiment
   */
  createExperiment(
    name: string,
    options?: { description?: string; tags?: string[] }
  ): Experiment {
    // Check if experiment exists
    const existing = Array.from(this.experiments.values()).find(
      (e) => e.name === name
    );
    if (existing) {
      return existing;
    }

    const experiment: Experiment = {
      id: uuidv4(),
      name,
      description: options?.description,
      tags: options?.tags ?? [],
      runs: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.experiments.set(experiment.id, experiment);
    this.emit("experimentCreated", experiment);

    return experiment;
  }

  /**
   * Get experiment by ID or name
   */
  getExperiment(idOrName: string): Experiment | undefined {
    return (
      this.experiments.get(idOrName) ??
      Array.from(this.experiments.values()).find((e) => e.name === idOrName)
    );
  }

  /**
   * List all experiments
   */
  listExperiments(): Experiment[] {
    return Array.from(this.experiments.values());
  }

  // --------------------------------------------------------------------------
  // Run Management
  // --------------------------------------------------------------------------

  /**
   * Start a new run
   */
  startRun(config: RunConfig): Run {
    const experiment = this.experiments.get(config.experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${config.experimentId}`);
    }

    const run: Run = {
      id: config.id ?? uuidv4(),
      experimentId: config.experimentId,
      name: config.name ?? `run-${Date.now()}`,
      parameters: config.parameters,
      metrics: [],
      artifacts: [],
      tags: config.tags ?? [],
      description: config.description,
      parentRunId: config.parentRunId,
      childRunIds: [],
      status: "running",
      startTime: new Date(),
    };

    this.runs.set(run.id, run);
    experiment.runs.push(run.id);
    experiment.updatedAt = new Date();

    // Link to parent if nested run
    if (config.parentRunId) {
      const parent = this.runs.get(config.parentRunId);
      if (parent) {
        parent.childRunIds.push(run.id);
      }
    }

    this.activeRun = run;
    this.emit("runStarted", run);

    return run;
  }

  /**
   * End the current run
   */
  endRun(status: "completed" | "failed" = "completed"): Run {
    if (!this.activeRun) {
      throw new Error("No active run to end");
    }

    this.flushMetrics();

    this.activeRun.status = status;
    this.activeRun.endTime = new Date();
    this.activeRun.duration =
      this.activeRun.endTime.getTime() - this.activeRun.startTime.getTime();

    const completedRun = this.activeRun;
    this.activeRun = null;

    if (status === "completed") {
      this.emit("runCompleted", completedRun);
    } else {
      this.emit("runFailed", completedRun, new Error("Run failed"));
    }

    return completedRun;
  }

  /**
   * Get the active run
   */
  getActiveRun(): Run | null {
    return this.activeRun;
  }

  /**
   * Get run by ID
   */
  getRun(id: string): Run | undefined {
    return this.runs.get(id);
  }

  /**
   * List runs for an experiment
   */
  listRuns(
    experimentId: string,
    filter?: { status?: RunStatus; tags?: string[] }
  ): Run[] {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      return [];
    }

    let results = experiment.runs
      .map((id) => this.runs.get(id))
      .filter((r): r is Run => r !== undefined);

    if (filter?.status) {
      results = results.filter((r) => r.status === filter.status);
    }

    if (filter?.tags?.length) {
      results = results.filter((r) =>
        filter.tags!.some((tag) => r.tags.includes(tag))
      );
    }

    return results;
  }

  // --------------------------------------------------------------------------
  // Metric Logging
  // --------------------------------------------------------------------------

  /**
   * Log a single metric
   */
  logMetric(key: string, value: number, step?: number): void {
    if (!this.activeRun) {
      throw new Error("No active run. Call startRun first.");
    }

    if (this.activeRun.metrics.length >= this.config.maxMetricsPerRun) {
      throw new Error(
        `Max metrics per run exceeded: ${this.config.maxMetricsPerRun}`
      );
    }

    const metric: MetricLog = {
      key,
      value,
      timestamp: new Date(),
      step,
    };

    this.metricBuffer.push(metric);
    this.emit("metricLogged", this.activeRun.id, metric);
  }

  /**
   * Log multiple metrics at once
   */
  logMetrics(metrics: Record<string, number>, step?: number): void {
    for (const [key, value] of Object.entries(metrics)) {
      this.logMetric(key, value, step);
    }
  }

  /**
   * Get metrics for a run
   */
  getMetrics(runId: string, key?: string): MetricLog[] {
    const run = this.runs.get(runId);
    if (!run) {
      return [];
    }

    if (key) {
      return run.metrics.filter((m) => m.key === key);
    }

    return run.metrics;
  }

  /**
   * Get the latest value of a metric
   */
  getLatestMetric(runId: string, key: string): number | undefined {
    const metrics = this.getMetrics(runId, key);
    if (metrics.length === 0) {
      return undefined;
    }
    return metrics[metrics.length - 1].value;
  }

  private flushMetrics(): void {
    if (!this.activeRun || this.metricBuffer.length === 0) {
      return;
    }

    this.activeRun.metrics.push(...this.metricBuffer);
    this.metricBuffer = [];
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flushMetrics();
    }, this.config.flushInterval);
  }

  // --------------------------------------------------------------------------
  // Artifact Logging
  // --------------------------------------------------------------------------

  /**
   * Log an artifact
   */
  logArtifact(
    name: string,
    path: string,
    type: ArtifactType = "other",
    metadata?: Record<string, unknown>
  ): Artifact {
    if (!this.activeRun) {
      throw new Error("No active run. Call startRun first.");
    }

    if (this.activeRun.artifacts.length >= this.config.maxArtifactsPerRun) {
      throw new Error(
        `Max artifacts per run exceeded: ${this.config.maxArtifactsPerRun}`
      );
    }

    const artifact: Artifact = {
      id: uuidv4(),
      name,
      type,
      path,
      metadata,
      createdAt: new Date(),
    };

    this.activeRun.artifacts.push(artifact);
    this.emit("artifactLogged", this.activeRun.id, artifact);

    return artifact;
  }

  /**
   * Get artifacts for a run
   */
  getArtifacts(runId: string, type?: ArtifactType): Artifact[] {
    const run = this.runs.get(runId);
    if (!run) {
      return [];
    }

    if (type) {
      return run.artifacts.filter((a) => a.type === type);
    }

    return run.artifacts;
  }

  // --------------------------------------------------------------------------
  // Run Comparison
  // --------------------------------------------------------------------------

  /**
   * Compare multiple runs
   */
  compareRuns(
    runIds: string[],
    options?: { primaryMetric?: string; higherIsBetter?: boolean }
  ): ComparisonResult {
    const runs = runIds
      .map((id) => this.runs.get(id))
      .filter((r): r is Run => r !== undefined);

    if (runs.length === 0) {
      return {
        runs: [],
        metrics: [],
        parameters: [],
        ranking: [],
      };
    }

    // Collect all metric keys
    const metricKeys = new Set<string>();
    for (const run of runs) {
      for (const metric of run.metrics) {
        metricKeys.add(metric.key);
      }
    }

    // Compare metrics
    const metrics: MetricComparison[] = [];
    for (const key of metricKeys) {
      const values = runs.map((run) => ({
        runId: run.id,
        value: this.getLatestMetric(run.id, key) ?? 0,
      }));

      const numValues = values.map((v) => v.value);
      const mean = numValues.reduce((a, b) => a + b, 0) / numValues.length;
      const variance =
        numValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) /
        numValues.length;

      metrics.push({
        key,
        values,
        min: Math.min(...numValues),
        max: Math.max(...numValues),
        mean,
        stdDev: Math.sqrt(variance),
      });
    }

    // Compare parameters
    const paramKeys = new Set<string>();
    for (const run of runs) {
      for (const key of Object.keys(run.parameters)) {
        paramKeys.add(key);
      }
    }

    const parameters: ParameterComparison[] = [];
    for (const key of paramKeys) {
      const values = runs.map((run) => ({
        runId: run.id,
        value: run.parameters[key],
      }));

      const uniqueValues = [
        ...new Set(values.map((v) => JSON.stringify(v.value))),
      ].map((v) => JSON.parse(v));

      parameters.push({
        key,
        values,
        uniqueValues,
      });
    }

    // Rank runs
    const primaryMetric = options?.primaryMetric ?? metrics[0]?.key;
    const higherIsBetter = options?.higherIsBetter ?? true;

    const ranking: RunRanking[] = runs
      .map((run) => {
        const runMetrics: Record<string, number> = {};
        for (const key of metricKeys) {
          runMetrics[key] = this.getLatestMetric(run.id, key) ?? 0;
        }

        return {
          runId: run.id,
          rank: 0,
          score: primaryMetric ? (runMetrics[primaryMetric] ?? 0) : 0,
          metrics: runMetrics,
        };
      })
      .sort((a, b) => (higherIsBetter ? b.score - a.score : a.score - b.score))
      .map((r, i) => ({ ...r, rank: i + 1 }));

    const bestRun = ranking[0] ? this.runs.get(ranking[0].runId) : undefined;

    return {
      runs,
      metrics,
      parameters,
      bestRun,
      ranking,
    };
  }

  // --------------------------------------------------------------------------
  // Search & Filter
  // --------------------------------------------------------------------------

  /**
   * Search runs by parameters
   */
  searchRuns(query: {
    experimentId?: string;
    parameters?: Record<string, unknown>;
    metricFilters?: { key: string; min?: number; max?: number }[];
    tags?: string[];
    status?: RunStatus;
    startAfter?: Date;
    startBefore?: Date;
  }): Run[] {
    let results = Array.from(this.runs.values());

    if (query.experimentId) {
      results = results.filter((r) => r.experimentId === query.experimentId);
    }

    if (query.parameters) {
      results = results.filter((r) => {
        for (const [key, value] of Object.entries(query.parameters!)) {
          if (JSON.stringify(r.parameters[key]) !== JSON.stringify(value)) {
            return false;
          }
        }
        return true;
      });
    }

    if (query.metricFilters) {
      results = results.filter((r) => {
        for (const filter of query.metricFilters!) {
          const value = this.getLatestMetric(r.id, filter.key);
          if (value === undefined) return false;
          if (filter.min !== undefined && value < filter.min) return false;
          if (filter.max !== undefined && value > filter.max) return false;
        }
        return true;
      });
    }

    if (query.tags?.length) {
      results = results.filter((r) =>
        query.tags!.some((tag) => r.tags.includes(tag))
      );
    }

    if (query.status) {
      results = results.filter((r) => r.status === query.status);
    }

    if (query.startAfter) {
      results = results.filter((r) => r.startTime >= query.startAfter!);
    }

    if (query.startBefore) {
      results = results.filter((r) => r.startTime <= query.startBefore!);
    }

    return results;
  }

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------

  /**
   * Delete a run
   */
  deleteRun(id: string): boolean {
    const run = this.runs.get(id);
    if (!run) {
      return false;
    }

    // Remove from experiment
    const experiment = this.experiments.get(run.experimentId);
    if (experiment) {
      experiment.runs = experiment.runs.filter((r) => r !== id);
    }

    // Remove from parent
    if (run.parentRunId) {
      const parent = this.runs.get(run.parentRunId);
      if (parent) {
        parent.childRunIds = parent.childRunIds.filter((c) => c !== id);
      }
    }

    return this.runs.delete(id);
  }

  /**
   * Dispose tracker resources
   */
  dispose(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flushMetrics();
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new ExperimentTracker instance
 */
export function createTracker(config?: TrackerConfig): ExperimentTracker {
  return new ExperimentTracker(config);
}

/**
 * Context manager for runs
 */
export async function withRun<T>(
  tracker: ExperimentTracker,
  config: RunConfig,
  fn: (run: Run) => Promise<T>
): Promise<T> {
  const run = tracker.startRun(config);
  try {
    const result = await fn(run);
    tracker.endRun("completed");
    return result;
  } catch (error) {
    tracker.endRun("failed");
    throw error;
  }
}
