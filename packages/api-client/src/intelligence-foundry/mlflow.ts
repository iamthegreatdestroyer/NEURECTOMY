/**
 * MLflow API Client
 *
 * Provides TypeScript client for MLflow Tracking Server API.
 * Handles experiment tracking, run management, metrics logging, and model registry.
 *
 * @module intelligence-foundry/mlflow
 */

import type { AxiosInstance } from "axios";

// ============================================================================
// Type Definitions
// ============================================================================

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: "active" | "deleted";
  tags?: Record<string, string>;
  creation_time?: number;
  last_update_time?: number;
}

export interface MLflowRun {
  run_id: string;
  run_uuid?: string;
  experiment_id: string;
  user_id?: string;
  status: "RUNNING" | "SCHEDULED" | "FINISHED" | "FAILED" | "KILLED";
  start_time: number;
  end_time?: number;
  artifact_uri?: string;
  lifecycle_stage?: "active" | "deleted";
}

export interface MLflowMetric {
  key: string;
  value: number;
  timestamp: number;
  step?: number;
}

export interface MLflowParam {
  key: string;
  value: string;
}

export interface MLflowTag {
  key: string;
  value: string;
}

export interface MLflowRunData {
  metrics: MLflowMetric[];
  params: MLflowParam[];
  tags: MLflowTag[];
}

export interface MLflowRunInfo extends MLflowRun {
  data: MLflowRunData;
}

export interface MLflowModel {
  name: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  user_id?: string;
  description?: string;
  latest_versions?: MLflowModelVersion[];
  tags?: Record<string, string>;
}

export interface MLflowModelVersion {
  name: string;
  version: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  user_id?: string;
  current_stage: "None" | "Staging" | "Production" | "Archived";
  description?: string;
  source: string;
  run_id: string;
  status: "PENDING_REGISTRATION" | "FAILED_REGISTRATION" | "READY";
  status_message?: string;
  tags?: Record<string, string>;
}

export interface CreateExperimentRequest {
  name: string;
  artifact_location?: string;
  tags?: Record<string, string>;
}

export interface StartRunRequest {
  experiment_id: string;
  user_id?: string;
  start_time?: number;
  tags?: Record<string, string>;
  run_name?: string;
}

export interface LogMetricsRequest {
  run_id: string;
  metrics: Array<{
    key: string;
    value: number;
    timestamp: number;
    step?: number;
  }>;
}

export interface LogParamsRequest {
  run_id: string;
  params: Array<{
    key: string;
    value: string;
  }>;
}

export interface ListExperimentsRequest {
  view_type?: "ACTIVE_ONLY" | "DELETED_ONLY" | "ALL";
  max_results?: number;
  page_token?: string;
  filter?: string;
  order_by?: string[];
}

export interface GetRunMetricsRequest {
  run_id: string;
  metric_key?: string;
}

export interface RegisterModelRequest {
  run_id: string;
  model_name: string;
  description?: string;
  tags?: Record<string, string>;
}

export interface MLflowApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}

export interface MLflowErrorResponse {
  error_code: string;
  message: string;
}

// ============================================================================
// MLflow Client Class
// ============================================================================

export class MLflowClient {
  private axios: AxiosInstance;
  private baseUrl: string;

  constructor(axios: AxiosInstance, baseUrl = "/api/mlflow") {
    this.axios = axios;
    this.baseUrl = baseUrl;
  }

  /**
   * Create a new MLflow experiment
   *
   * @param request - Experiment creation parameters
   * @returns Created experiment with ID
   *
   * @example
   * ```typescript
   * const experiment = await mlflowClient.createExperiment({
   *   name: 'BERT Fine-tuning v1',
   *   tags: { 'team': 'nlp', 'project': 'chatbot' }
   * });
   * console.log(`Created experiment: ${experiment.experiment_id}`);
   * ```
   */
  async createExperiment(
    request: CreateExperimentRequest
  ): Promise<MLflowExperiment> {
    try {
      const response = await this.axios.post<MLflowExperiment>(
        `${this.baseUrl}/experiments/create`,
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to create experiment");
    }
  }

  /**
   * Start a new training run within an experiment
   *
   * @param request - Run start parameters
   * @returns Started run with ID and metadata
   *
   * @example
   * ```typescript
   * const run = await mlflowClient.startRun({
   *   experiment_id: 'exp123',
   *   run_name: 'training-run-1',
   *   tags: { 'model_type': 'transformer' }
   * });
   * console.log(`Started run: ${run.run_id}`);
   * ```
   */
  async startRun(request: StartRunRequest): Promise<MLflowRun> {
    try {
      const response = await this.axios.post<MLflowRun>(
        `${this.baseUrl}/runs/start`,
        {
          ...request,
          start_time: request.start_time || Date.now(),
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to start run");
    }
  }

  /**
   * Log metrics for a training run (supports batch logging)
   *
   * @param request - Metrics to log with run ID
   * @returns Success confirmation
   *
   * @example
   * ```typescript
   * await mlflowClient.logMetrics({
   *   run_id: 'run123',
   *   metrics: [
   *     { key: 'train_loss', value: 0.42, timestamp: Date.now(), step: 100 },
   *     { key: 'val_accuracy', value: 0.87, timestamp: Date.now(), step: 100 }
   *   ]
   * });
   * ```
   */
  async logMetrics(request: LogMetricsRequest): Promise<void> {
    try {
      await this.axios.post(`${this.baseUrl}/runs/log-metrics`, request);
    } catch (error) {
      throw this.handleError(error, "Failed to log metrics");
    }
  }

  /**
   * Log parameters for a training run
   *
   * @param request - Parameters to log with run ID
   * @returns Success confirmation
   *
   * @example
   * ```typescript
   * await mlflowClient.logParams({
   *   run_id: 'run123',
   *   params: [
   *     { key: 'learning_rate', value: '0.001' },
   *     { key: 'batch_size', value: '32' }
   *   ]
   * });
   * ```
   */
  async logParams(request: LogParamsRequest): Promise<void> {
    try {
      await this.axios.post(`${this.baseUrl}/runs/log-params`, request);
    } catch (error) {
      throw this.handleError(error, "Failed to log parameters");
    }
  }

  /**
   * End a training run and mark its final status
   *
   * @param runId - Run ID to end
   * @param status - Final status (FINISHED, FAILED, KILLED)
   * @returns Updated run info
   *
   * @example
   * ```typescript
   * await mlflowClient.endRun('run123', 'FINISHED');
   * ```
   */
  async endRun(
    runId: string,
    status: "FINISHED" | "FAILED" | "KILLED" = "FINISHED"
  ): Promise<MLflowRun> {
    try {
      const response = await this.axios.post<MLflowRun>(
        `${this.baseUrl}/runs/end`,
        {
          run_id: runId,
          status,
          end_time: Date.now(),
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to end run");
    }
  }

  /**
   * List all experiments with optional filtering and pagination
   *
   * @param request - Filter and pagination options
   * @returns Array of experiments matching criteria
   *
   * @example
   * ```typescript
   * const experiments = await mlflowClient.listExperiments({
   *   view_type: 'ACTIVE_ONLY',
   *   max_results: 50,
   *   filter: 'tags.team = "nlp"',
   *   order_by: ['creation_time DESC']
   * });
   * ```
   */
  async listExperiments(
    request?: ListExperimentsRequest
  ): Promise<MLflowExperiment[]> {
    try {
      const response = await this.axios.get<{
        experiments: MLflowExperiment[];
      }>(`${this.baseUrl}/experiments/list`, { params: request });
      return response.data.experiments || [];
    } catch (error) {
      throw this.handleError(error, "Failed to list experiments");
    }
  }

  /**
   * Get detailed information about a specific run
   *
   * @param runId - Run ID to retrieve
   * @returns Complete run information with metrics, params, and tags
   *
   * @example
   * ```typescript
   * const runInfo = await mlflowClient.getRun('run123');
   * console.log(`Train loss: ${runInfo.data.metrics.find(m => m.key === 'train_loss')?.value}`);
   * ```
   */
  async getRun(runId: string): Promise<MLflowRunInfo> {
    try {
      const response = await this.axios.get<MLflowRunInfo>(
        `${this.baseUrl}/runs/${runId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to get run");
    }
  }

  /**
   * Get metrics history for a specific run
   *
   * @param request - Run ID and optional metric key filter
   * @returns Array of metric values over time
   *
   * @example
   * ```typescript
   * const metrics = await mlflowClient.getRunMetrics({
   *   run_id: 'run123',
   *   metric_key: 'train_loss'
   * });
   * ```
   */
  async getRunMetrics(request: GetRunMetricsRequest): Promise<MLflowMetric[]> {
    try {
      const response = await this.axios.get<{ metrics: MLflowMetric[] }>(
        `${this.baseUrl}/runs/${request.run_id}/metrics`,
        { params: { metric_key: request.metric_key } }
      );
      return response.data.metrics || [];
    } catch (error) {
      throw this.handleError(error, "Failed to get run metrics");
    }
  }

  /**
   * Register a trained model to the MLflow Model Registry
   *
   * @param request - Model registration parameters
   * @returns Registered model version details
   *
   * @example
   * ```typescript
   * const modelVersion = await mlflowClient.registerModel({
   *   run_id: 'run123',
   *   model_name: 'bert-chatbot-v1',
   *   description: 'BERT fine-tuned on customer support conversations',
   *   tags: { 'stage': 'staging' }
   * });
   * console.log(`Registered model version: ${modelVersion.version}`);
   * ```
   */
  async registerModel(
    request: RegisterModelRequest
  ): Promise<MLflowModelVersion> {
    try {
      const response = await this.axios.post<MLflowModelVersion>(
        `${this.baseUrl}/models/register`,
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to register model");
    }
  }

  /**
   * Get registered model by name
   *
   * @param modelName - Name of the registered model
   * @returns Model details with all versions
   *
   * @example
   * ```typescript
   * const model = await mlflowClient.getModel('bert-chatbot-v1');
   * const productionVersion = model.latest_versions?.find(v => v.current_stage === 'Production');
   * ```
   */
  async getModel(modelName: string): Promise<MLflowModel> {
    try {
      const response = await this.axios.get<MLflowModel>(
        `${this.baseUrl}/models/${encodeURIComponent(modelName)}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to get model");
    }
  }

  /**
   * Search for runs across experiments with advanced filtering
   *
   * @param experimentIds - Array of experiment IDs to search within
   * @param filter - SQL-like filter string
   * @param orderBy - Array of order by clauses
   * @param maxResults - Maximum number of results
   * @returns Array of runs matching search criteria
   *
   * @example
   * ```typescript
   * const runs = await mlflowClient.searchRuns(
   *   ['exp1', 'exp2'],
   *   'metrics.accuracy > 0.9 AND params.optimizer = "adam"',
   *   ['metrics.accuracy DESC'],
   *   10
   * );
   * ```
   */
  async searchRuns(
    experimentIds: string[],
    filter?: string,
    orderBy?: string[],
    maxResults = 100
  ): Promise<MLflowRunInfo[]> {
    try {
      const response = await this.axios.post<{ runs: MLflowRunInfo[] }>(
        `${this.baseUrl}/runs/search`,
        {
          experiment_ids: experimentIds,
          filter,
          order_by: orderBy,
          max_results: maxResults,
        }
      );
      return response.data.runs || [];
    } catch (error) {
      throw this.handleError(error, "Failed to search runs");
    }
  }

  /**
   * Delete an experiment (soft delete, can be restored)
   *
   * @param experimentId - Experiment ID to delete
   * @returns Success confirmation
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    try {
      await this.axios.post(`${this.baseUrl}/experiments/delete`, {
        experiment_id: experimentId,
      });
    } catch (error) {
      throw this.handleError(error, "Failed to delete experiment");
    }
  }

  /**
   * Delete a run (soft delete, can be restored)
   *
   * @param runId - Run ID to delete
   * @returns Success confirmation
   */
  async deleteRun(runId: string): Promise<void> {
    try {
      await this.axios.post(`${this.baseUrl}/runs/delete`, {
        run_id: runId,
      });
    } catch (error) {
      throw this.handleError(error, "Failed to delete run");
    }
  }

  /**
   * Error handler with detailed error messages
   */
  private handleError(error: unknown, context: string): Error {
    if (error instanceof Error) {
      if ("response" in error) {
        const axiosError = error as any;
        const mlflowError = axiosError.response?.data as MLflowErrorResponse;
        if (mlflowError?.message) {
          return new Error(
            `${context}: ${mlflowError.message} (${mlflowError.error_code})`
          );
        }
        return new Error(
          `${context}: ${axiosError.response?.statusText || axiosError.message}`
        );
      }
      return new Error(`${context}: ${error.message}`);
    }
    return new Error(`${context}: Unknown error`);
  }
}

// ============================================================================
// Singleton Instance Factory
// ============================================================================

let mlflowClientInstance: MLflowClient | null = null;

/**
 * Get or create MLflow client singleton instance
 *
 * @param axios - Axios instance to use for HTTP requests
 * @param baseUrl - Base URL for MLflow API (default: /api/mlflow)
 * @returns MLflow client instance
 */
export function getMLflowClient(
  axios: AxiosInstance,
  baseUrl?: string
): MLflowClient {
  if (!mlflowClientInstance) {
    mlflowClientInstance = new MLflowClient(axios, baseUrl);
  }
  return mlflowClientInstance;
}

/**
 * Reset the singleton instance (useful for testing)
 */
export function resetMLflowClient(): void {
  mlflowClientInstance = null;
}
