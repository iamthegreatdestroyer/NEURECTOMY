/**
 * NEURECTOMY MLflow Bridge
 * @module @neurectomy/experimentation-engine/hypothesis
 * @agent @TENSOR @SYNAPSE
 *
 * Bridges TypeScript experimentation-engine with Python MLflow service.
 * Enables bidirectional sync of experiments, runs, metrics, and artifacts
 * between the TypeScript frontend and Python ML backend.
 */

import { EventEmitter } from "eventemitter3";
import type { Run } from "./tracker";

// ============================================================================
// Types
// ============================================================================

export interface MLflowConfig {
  /** MLflow tracking server URL (e.g., http://localhost:5000) */
  trackingUri: string;
  /** Default experiment name */
  defaultExperiment?: string;
  /** HTTP timeout in milliseconds */
  timeout?: number;
  /** Retry configuration */
  retry?: {
    maxRetries: number;
    backoffMs: number;
  };
  /** Authentication token (optional) */
  authToken?: string;
}

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: "active" | "deleted";
  tags: Record<string, string>;
  creation_time: number;
  last_update_time: number;
}

export interface MLflowRun {
  info: {
    run_id: string;
    run_uuid: string;
    experiment_id: string;
    run_name: string;
    user_id: string;
    status: "RUNNING" | "SCHEDULED" | "FINISHED" | "FAILED" | "KILLED";
    start_time: number;
    end_time?: number;
    artifact_uri: string;
    lifecycle_stage: "active" | "deleted";
  };
  data: {
    metrics: Array<{
      key: string;
      value: number;
      timestamp: number;
      step: number;
    }>;
    params: Array<{
      key: string;
      value: string;
    }>;
    tags: Array<{
      key: string;
      value: string;
    }>;
  };
}

export interface MLflowMetric {
  key: string;
  value: number;
  timestamp: number;
  step: number;
}

export interface MLflowParam {
  key: string;
  value: string;
}

export interface MLflowArtifact {
  path: string;
  is_dir: boolean;
  file_size?: number;
}

export interface MLflowBridgeEvents {
  connected: () => void;
  disconnected: () => void;
  error: (error: Error) => void;
  experimentSynced: (experimentId: string) => void;
  runSynced: (runId: string) => void;
  metricsSynced: (runId: string, count: number) => void;
}

export interface SyncResult {
  success: boolean;
  synced: number;
  failed: number;
  errors: string[];
}

// ============================================================================
// MLflow Bridge Implementation
// ============================================================================

export class MLflowBridge extends EventEmitter<MLflowBridgeEvents> {
  private config: Required<MLflowConfig>;
  private connected = false;
  private headers: Record<string, string>;

  constructor(config: MLflowConfig) {
    super();
    this.config = {
      trackingUri: config.trackingUri.replace(/\/$/, ""), // Remove trailing slash
      defaultExperiment: config.defaultExperiment ?? "neurectomy-experiments",
      timeout: config.timeout ?? 30000,
      retry: config.retry ?? { maxRetries: 3, backoffMs: 1000 },
      authToken: config.authToken ?? "",
    };

    this.headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };

    if (this.config.authToken) {
      this.headers["Authorization"] = `Bearer ${this.config.authToken}`;
    }
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Test connection to MLflow server.
   */
  async connect(): Promise<boolean> {
    try {
      const response = await this.request(
        "GET",
        "/api/2.0/mlflow/experiments/search"
      );
      this.connected = response.ok;

      if (this.connected) {
        this.emit("connected");
      }

      return this.connected;
    } catch (error) {
      this.emit("error", error as Error);
      return false;
    }
  }

  /**
   * Check if connected to MLflow.
   */
  isConnected(): boolean {
    return this.connected;
  }

  // ==========================================================================
  // Experiment Operations
  // ==========================================================================

  /**
   * Get or create an MLflow experiment.
   */
  async getOrCreateExperiment(name: string): Promise<string> {
    // Try to get existing experiment
    const searchResponse = await this.request(
      "POST",
      "/api/2.0/mlflow/experiments/search",
      { filter: `name = '${name}'` }
    );

    if (searchResponse.ok) {
      const data = await searchResponse.json();
      if (data.experiments?.length > 0) {
        return data.experiments[0].experiment_id;
      }
    }

    // Create new experiment
    const createResponse = await this.request(
      "POST",
      "/api/2.0/mlflow/experiments/create",
      { name, tags: [{ key: "source", value: "neurectomy-ts" }] }
    );

    if (!createResponse.ok) {
      throw new Error(
        `Failed to create experiment: ${await createResponse.text()}`
      );
    }

    const result = await createResponse.json();
    return result.experiment_id;
  }

  /**
   * List all experiments.
   */
  async listExperiments(): Promise<MLflowExperiment[]> {
    const response = await this.request(
      "POST",
      "/api/2.0/mlflow/experiments/search",
      { max_results: 1000 }
    );

    if (!response.ok) {
      throw new Error(`Failed to list experiments: ${await response.text()}`);
    }

    const data = await response.json();
    return data.experiments || [];
  }

  /**
   * Get experiment by ID.
   */
  async getExperiment(experimentId: string): Promise<MLflowExperiment | null> {
    const response = await this.request(
      "GET",
      `/api/2.0/mlflow/experiments/get?experiment_id=${experimentId}`
    );

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.experiment;
  }

  // ==========================================================================
  // Run Operations
  // ==========================================================================

  /**
   * Create a new MLflow run.
   */
  async createRun(
    experimentId: string,
    runName?: string,
    tags?: Record<string, string>
  ): Promise<string> {
    const tagsList = tags
      ? Object.entries(tags).map(([key, value]) => ({ key, value }))
      : [];

    tagsList.push({ key: "mlflow.source.type", value: "NEURECTOMY_TS" });

    const response = await this.request("POST", "/api/2.0/mlflow/runs/create", {
      experiment_id: experimentId,
      run_name: runName,
      tags: tagsList,
      start_time: Date.now(),
    });

    if (!response.ok) {
      throw new Error(`Failed to create run: ${await response.text()}`);
    }

    const data = await response.json();
    return data.run.info.run_id;
  }

  /**
   * End an MLflow run.
   */
  async endRun(
    runId: string,
    status: "FINISHED" | "FAILED" | "KILLED" = "FINISHED"
  ): Promise<void> {
    const response = await this.request("POST", "/api/2.0/mlflow/runs/update", {
      run_id: runId,
      status,
      end_time: Date.now(),
    });

    if (!response.ok) {
      throw new Error(`Failed to end run: ${await response.text()}`);
    }
  }

  /**
   * Get a run by ID.
   */
  async getRun(runId: string): Promise<MLflowRun | null> {
    const response = await this.request(
      "GET",
      `/api/2.0/mlflow/runs/get?run_id=${runId}`
    );

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.run;
  }

  /**
   * Search for runs with filters.
   */
  async searchRuns(
    experimentIds: string[],
    filter?: string,
    maxResults = 1000
  ): Promise<MLflowRun[]> {
    const response = await this.request("POST", "/api/2.0/mlflow/runs/search", {
      experiment_ids: experimentIds,
      filter: filter,
      max_results: maxResults,
      order_by: ["start_time DESC"],
    });

    if (!response.ok) {
      throw new Error(`Failed to search runs: ${await response.text()}`);
    }

    const data = await response.json();
    return data.runs || [];
  }

  // ==========================================================================
  // Metrics & Parameters
  // ==========================================================================

  /**
   * Log a single metric.
   */
  async logMetric(
    runId: string,
    key: string,
    value: number,
    step?: number,
    timestamp?: number
  ): Promise<void> {
    const response = await this.request(
      "POST",
      "/api/2.0/mlflow/runs/log-metric",
      {
        run_id: runId,
        key,
        value,
        timestamp: timestamp ?? Date.now(),
        step: step ?? 0,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to log metric: ${await response.text()}`);
    }
  }

  /**
   * Log multiple metrics in batch.
   */
  async logMetricsBatch(
    runId: string,
    metrics: Array<{ key: string; value: number; step?: number }>
  ): Promise<void> {
    const timestamp = Date.now();
    const metricsPayload = metrics.map((m) => ({
      key: m.key,
      value: m.value,
      timestamp,
      step: m.step ?? 0,
    }));

    const response = await this.request(
      "POST",
      "/api/2.0/mlflow/runs/log-batch",
      {
        run_id: runId,
        metrics: metricsPayload,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to log metrics batch: ${await response.text()}`);
    }
  }

  /**
   * Log a parameter.
   */
  async logParam(runId: string, key: string, value: string): Promise<void> {
    const response = await this.request(
      "POST",
      "/api/2.0/mlflow/runs/log-param",
      {
        run_id: runId,
        key,
        value,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to log parameter: ${await response.text()}`);
    }
  }

  /**
   * Log multiple parameters in batch.
   */
  async logParamsBatch(
    runId: string,
    params: Record<string, unknown>
  ): Promise<void> {
    const paramsPayload = Object.entries(params).map(([key, value]) => ({
      key,
      value: String(value),
    }));

    const response = await this.request(
      "POST",
      "/api/2.0/mlflow/runs/log-batch",
      {
        run_id: runId,
        params: paramsPayload,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to log params batch: ${await response.text()}`);
    }
  }

  /**
   * Get metric history for a run.
   */
  async getMetricHistory(
    runId: string,
    metricKey: string
  ): Promise<MLflowMetric[]> {
    const response = await this.request(
      "GET",
      `/api/2.0/mlflow/metrics/get-history?run_id=${runId}&metric_key=${metricKey}`
    );

    if (!response.ok) {
      throw new Error(`Failed to get metric history: ${await response.text()}`);
    }

    const data = await response.json();
    return data.metrics || [];
  }

  // ==========================================================================
  // Artifact Operations
  // ==========================================================================

  /**
   * List artifacts for a run.
   */
  async listArtifacts(runId: string, path?: string): Promise<MLflowArtifact[]> {
    let url = `/api/2.0/mlflow/artifacts/list?run_id=${runId}`;
    if (path) {
      url += `&path=${encodeURIComponent(path)}`;
    }

    const response = await this.request("GET", url);

    if (!response.ok) {
      throw new Error(`Failed to list artifacts: ${await response.text()}`);
    }

    const data = await response.json();
    return data.files || [];
  }

  // ==========================================================================
  // TypeScript Tracker Sync
  // ==========================================================================

  /**
   * Sync a TypeScript Run to MLflow.
   * Creates corresponding MLflow run and logs all metrics/params.
   */
  async syncRunToMLflow(run: Run): Promise<string> {
    // Get or create experiment
    const experimentId = await this.getOrCreateExperiment(run.experimentId);

    // Create MLflow run
    const mlflowRunId = await this.createRun(
      experimentId,
      run.name,
      Object.fromEntries(run.tags.map((t) => [t, "true"]))
    );

    // Log parameters
    if (Object.keys(run.parameters).length > 0) {
      await this.logParamsBatch(mlflowRunId, run.parameters);
    }

    // Log metrics
    if (run.metrics.length > 0) {
      const metricsForBatch = run.metrics.map((m) => ({
        key: m.key,
        value: m.value,
        step: m.step,
      }));

      // MLflow batch has limits, chunk if needed
      const chunkSize = 1000;
      for (let i = 0; i < metricsForBatch.length; i += chunkSize) {
        await this.logMetricsBatch(
          mlflowRunId,
          metricsForBatch.slice(i, i + chunkSize)
        );
      }
    }

    // End run with appropriate status
    const status = this.mapRunStatus(run.status);
    await this.endRun(mlflowRunId, status);

    this.emit("runSynced", mlflowRunId);
    return mlflowRunId;
  }

  /**
   * Sync an MLflow run to TypeScript Run format.
   */
  async syncRunFromMLflow(mlflowRunId: string): Promise<Run | null> {
    const mlflowRun = await this.getRun(mlflowRunId);
    if (!mlflowRun) return null;

    // Convert MLflow run to TypeScript Run
    const run: Run = {
      id: mlflowRun.info.run_id,
      experimentId: mlflowRun.info.experiment_id,
      name: mlflowRun.info.run_name || mlflowRun.info.run_id,
      parameters: Object.fromEntries(
        mlflowRun.data.params.map((p) => [p.key, p.value])
      ),
      metrics: mlflowRun.data.metrics.map((m) => ({
        key: m.key,
        value: m.value,
        timestamp: new Date(m.timestamp),
        step: m.step,
      })),
      artifacts: [], // Would need separate artifact list call
      tags: mlflowRun.data.tags
        .filter((t) => !t.key.startsWith("mlflow."))
        .map((t) => t.key),
      status: this.mapMLflowStatus(mlflowRun.info.status),
      startTime: new Date(mlflowRun.info.start_time),
      endTime: mlflowRun.info.end_time
        ? new Date(mlflowRun.info.end_time)
        : undefined,
      childRunIds: [],
    };

    if (run.endTime && run.startTime) {
      run.duration = run.endTime.getTime() - run.startTime.getTime();
    }

    return run;
  }

  /**
   * Sync all runs from an experiment.
   */
  async syncExperimentRuns(experimentName: string): Promise<SyncResult> {
    const result: SyncResult = {
      success: true,
      synced: 0,
      failed: 0,
      errors: [],
    };

    try {
      const experimentId = await this.getOrCreateExperiment(experimentName);
      const mlflowRuns = await this.searchRuns([experimentId]);

      for (const mlflowRun of mlflowRuns) {
        try {
          await this.syncRunFromMLflow(mlflowRun.info.run_id);
          result.synced++;
        } catch (error) {
          result.failed++;
          result.errors.push(
            `Failed to sync run ${mlflowRun.info.run_id}: ${error}`
          );
        }
      }

      this.emit("experimentSynced", experimentId);
    } catch (error) {
      result.success = false;
      result.errors.push(`Failed to sync experiment: ${error}`);
    }

    return result;
  }

  // ==========================================================================
  // Model Registry Operations
  // ==========================================================================

  /**
   * Get latest model version from registry.
   */
  async getLatestModelVersion(
    modelName: string,
    stage?: "Production" | "Staging" | "Archived" | "None"
  ): Promise<string | null> {
    let url = `/api/2.0/mlflow/registered-models/get-latest-versions?name=${encodeURIComponent(modelName)}`;
    if (stage) {
      url += `&stages=${stage}`;
    }

    const response = await this.request("GET", url);

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    if (data.model_versions?.length > 0) {
      return data.model_versions[0].version;
    }

    return null;
  }

  /**
   * List all registered models.
   */
  async listRegisteredModels(): Promise<
    Array<{
      name: string;
      latest_versions: Array<{ version: string; stage: string }>;
    }>
  > {
    const response = await this.request(
      "GET",
      "/api/2.0/mlflow/registered-models/search?max_results=1000"
    );

    if (!response.ok) {
      throw new Error(`Failed to list models: ${await response.text()}`);
    }

    const data = await response.json();
    return data.registered_models || [];
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private async request(
    method: "GET" | "POST" | "PUT" | "DELETE",
    path: string,
    body?: unknown
  ): Promise<Response> {
    const url = `${this.config.trackingUri}${path}`;
    const options: RequestInit = {
      method,
      headers: this.headers,
    };

    if (body && method !== "GET") {
      options.body = JSON.stringify(body);
    }

    // Implement retry logic
    let lastError: Error | null = null;
    for (let i = 0; i <= this.config.retry.maxRetries; i++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(
          () => controller.abort(),
          this.config.timeout
        );

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        return response;
      } catch (error) {
        lastError = error as Error;
        if (i < this.config.retry.maxRetries) {
          await this.sleep(this.config.retry.backoffMs * Math.pow(2, i));
        }
      }
    }

    throw lastError || new Error("Request failed");
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private mapRunStatus(
    status: Run["status"]
  ): "FINISHED" | "FAILED" | "KILLED" {
    switch (status) {
      case "completed":
        return "FINISHED";
      case "failed":
        return "FAILED";
      case "killed":
        return "KILLED";
      default:
        return "FINISHED";
    }
  }

  private mapMLflowStatus(status: MLflowRun["info"]["status"]): Run["status"] {
    switch (status) {
      case "FINISHED":
        return "completed";
      case "FAILED":
        return "failed";
      case "KILLED":
        return "killed";
      case "RUNNING":
      case "SCHEDULED":
        return "running";
      default:
        return "completed";
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create an MLflow bridge instance.
 */
export function createMLflowBridge(config: MLflowConfig): MLflowBridge {
  return new MLflowBridge(config);
}

/**
 * Create a bridge connected to the default NEURECTOMY MLflow server.
 */
export function createDefaultBridge(): MLflowBridge {
  const trackingUri =
    (typeof process !== "undefined" && process.env?.MLFLOW_TRACKING_URI) ||
    "http://localhost:5000";

  return new MLflowBridge({
    trackingUri,
    defaultExperiment: "neurectomy-experiments",
  });
}

// ============================================================================
// High-Level Sync Utilities
// ============================================================================

/**
 * Context manager for syncing a run to MLflow.
 * Automatically creates and closes the MLflow run.
 */
export async function withMLflowRun<T>(
  bridge: MLflowBridge,
  experimentName: string,
  runName: string,
  fn: (runId: string) => Promise<T>
): Promise<T> {
  const experimentId = await bridge.getOrCreateExperiment(experimentName);
  const runId = await bridge.createRun(experimentId, runName);

  try {
    const result = await fn(runId);
    await bridge.endRun(runId, "FINISHED");
    return result;
  } catch (error) {
    await bridge.endRun(runId, "FAILED");
    throw error;
  }
}

/**
 * Decorator-style function to log execution metrics to MLflow.
 */
export function mlflowTracked(bridge: MLflowBridge, experimentName: string) {
  return function <T extends (...args: unknown[]) => Promise<unknown>>(
    fn: T,
    runName?: string
  ): T {
    return (async (...args: unknown[]) => {
      const experimentId = await bridge.getOrCreateExperiment(experimentName);
      const runId = await bridge.createRun(
        experimentId,
        runName || fn.name || "tracked-function"
      );

      const startTime = Date.now();

      try {
        const result = await fn(...args);

        await bridge.logMetric(runId, "duration_ms", Date.now() - startTime);
        await bridge.logMetric(runId, "success", 1);
        await bridge.endRun(runId, "FINISHED");

        return result;
      } catch (error) {
        await bridge.logMetric(runId, "duration_ms", Date.now() - startTime);
        await bridge.logMetric(runId, "success", 0);
        await bridge.endRun(runId, "FAILED");
        throw error;
      }
    }) as T;
  };
}
