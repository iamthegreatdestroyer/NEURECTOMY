/**
 * Optuna API Client
 *
 * Provides TypeScript client for Optuna hyperparameter optimization service.
 * Handles study creation, trial management, and optimization tracking.
 *
 * @module intelligence-foundry/optuna
 */

import type { AxiosInstance } from "axios";

// ============================================================================
// Type Definitions
// ============================================================================

export type OptunaSampler = "tpe" | "random" | "grid" | "cmaes";
export type OptunaPruner = "median" | "percentile" | "hyperband" | "none";
export type StudyDirection = "minimize" | "maximize";
export type TrialState =
  | "running"
  | "complete"
  | "pruned"
  | "failed"
  | "waiting";

export interface OptunaParameter {
  name: string;
  type: "float" | "int" | "categorical";
  min?: number;
  max?: number;
  choices?: (string | number)[];
  log?: boolean;
}

export interface OptunaTrial {
  number: number;
  value: number | null;
  params: Record<string, any>;
  state: TrialState;
  datetime_start: string;
  datetime_complete: string | null;
  duration: number | null;
  user_attrs?: Record<string, any>;
  system_attrs?: Record<string, any>;
  intermediate_values?: Array<{ step: number; value: number }>;
}

export interface OptunaStudy {
  study_id: string;
  study_name: string;
  direction: StudyDirection;
  user_attrs?: Record<string, any>;
  system_attrs?: Record<string, any>;
  n_trials: number;
  datetime_start: string;
  best_trial?: OptunaTrial;
  trials: OptunaTrial[];
}

export interface CreateStudyRequest {
  study_name: string;
  direction: StudyDirection;
  sampler?: OptunaSampler;
  pruner?: OptunaPruner;
  load_if_exists?: boolean;
  storage?: string;
}

export interface CreateStudyResponse {
  study_id: string;
  study_name: string;
  direction: StudyDirection;
}

export interface SubmitTrialRequest {
  study_id: string;
  params: Record<string, any>;
  value?: number;
  state?: TrialState;
  user_attrs?: Record<string, any>;
}

export interface SubmitTrialResponse {
  trial_number: number;
  trial_id: string;
  state: TrialState;
}

export interface GetBestTrialRequest {
  study_id: string;
}

export interface ListTrialsRequest {
  study_id: string;
  states?: TrialState[];
  limit?: number;
  offset?: number;
}

export interface StopStudyRequest {
  study_id: string;
}

export interface SuggestRequest {
  study_id: string;
  trial_id: string;
  parameters: OptunaParameter[];
}

export interface SuggestResponse {
  suggestions: Record<string, any>;
}

export interface ReportIntermediateValueRequest {
  study_id: string;
  trial_id: string;
  step: number;
  value: number;
}

export interface OptunaErrorResponse {
  error: string;
  detail?: string;
}

// ============================================================================
// Optuna Client Class
// ============================================================================

export class OptunaClient {
  private axios: AxiosInstance;
  private baseUrl: string;

  constructor(axios: AxiosInstance, baseUrl = "/api/optuna") {
    this.axios = axios;
    this.baseUrl = baseUrl;
  }

  /**
   * Create a new Optuna study for hyperparameter optimization
   *
   * @param request - Study configuration parameters
   * @returns Created study with ID
   *
   * @example
   * ```typescript
   * const study = await optunaClient.createStudy({
   *   study_name: 'bert-hyperparameter-search',
   *   direction: 'maximize',
   *   sampler: 'tpe',
   *   pruner: 'median'
   * });
   * console.log(`Created study: ${study.study_id}`);
   * ```
   */
  async createStudy(request: CreateStudyRequest): Promise<CreateStudyResponse> {
    try {
      const response = await this.axios.post<CreateStudyResponse>(
        `${this.baseUrl}/studies/create`,
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to create study");
    }
  }

  /**
   * Get study details including all trials
   *
   * @param studyId - Study ID to retrieve
   * @returns Complete study information with trials
   *
   * @example
   * ```typescript
   * const study = await optunaClient.getStudy('study123');
   * console.log(`Best value: ${study.best_trial?.value}`);
   * ```
   */
  async getStudy(studyId: string): Promise<OptunaStudy> {
    try {
      const response = await this.axios.get<OptunaStudy>(
        `${this.baseUrl}/studies/${studyId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to get study");
    }
  }

  /**
   * Submit a completed trial with results
   *
   * @param request - Trial parameters and result value
   * @returns Submitted trial information
   *
   * @example
   * ```typescript
   * const trial = await optunaClient.submitTrial({
   *   study_id: 'study123',
   *   params: {
   *     learning_rate: 0.001,
   *     batch_size: 32,
   *     num_layers: 6
   *   },
   *   value: 0.92,
   *   state: 'complete'
   * });
   * ```
   */
  async submitTrial(request: SubmitTrialRequest): Promise<SubmitTrialResponse> {
    try {
      const response = await this.axios.post<SubmitTrialResponse>(
        `${this.baseUrl}/studies/${request.study_id}/trials`,
        {
          params: request.params,
          value: request.value,
          state: request.state || "complete",
          user_attrs: request.user_attrs,
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to submit trial");
    }
  }

  /**
   * Get the best trial from a study based on optimization direction
   *
   * @param request - Study ID to get best trial from
   * @returns Best trial with parameters and value
   *
   * @example
   * ```typescript
   * const bestTrial = await optunaClient.getBestTrial({ study_id: 'study123' });
   * console.log(`Best params: ${JSON.stringify(bestTrial.params)}`);
   * console.log(`Best value: ${bestTrial.value}`);
   * ```
   */
  async getBestTrial(request: GetBestTrialRequest): Promise<OptunaTrial> {
    try {
      const response = await this.axios.get<OptunaTrial>(
        `${this.baseUrl}/studies/${request.study_id}/best-trial`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to get best trial");
    }
  }

  /**
   * List all trials in a study with optional filtering
   *
   * @param request - Study ID and filter options
   * @returns Array of trials matching criteria
   *
   * @example
   * ```typescript
   * const trials = await optunaClient.listTrials({
   *   study_id: 'study123',
   *   states: ['complete', 'running'],
   *   limit: 50
   * });
   * ```
   */
  async listTrials(request: ListTrialsRequest): Promise<OptunaTrial[]> {
    try {
      const response = await this.axios.get<{ trials: OptunaTrial[] }>(
        `${this.baseUrl}/studies/${request.study_id}/trials`,
        {
          params: {
            states: request.states?.join(","),
            limit: request.limit,
            offset: request.offset,
          },
        }
      );
      return response.data.trials || [];
    } catch (error) {
      throw this.handleError(error, "Failed to list trials");
    }
  }

  /**
   * Stop an ongoing optimization study
   *
   * @param request - Study ID to stop
   * @returns Success confirmation
   *
   * @example
   * ```typescript
   * await optunaClient.stopStudy({ study_id: 'study123' });
   * ```
   */
  async stopStudy(request: StopStudyRequest): Promise<void> {
    try {
      await this.axios.post(`${this.baseUrl}/studies/${request.study_id}/stop`);
    } catch (error) {
      throw this.handleError(error, "Failed to stop study");
    }
  }

  /**
   * Get parameter suggestions for a new trial
   *
   * @param request - Study, trial IDs and parameter definitions
   * @returns Suggested parameter values
   *
   * @example
   * ```typescript
   * const suggestions = await optunaClient.suggest({
   *   study_id: 'study123',
   *   trial_id: 'trial456',
   *   parameters: [
   *     { name: 'learning_rate', type: 'float', min: 1e-5, max: 1e-1, log: true },
   *     { name: 'batch_size', type: 'categorical', choices: [16, 32, 64, 128] }
   *   ]
   * });
   * // suggestions.suggestions = { learning_rate: 0.00123, batch_size: 32 }
   * ```
   */
  async suggest(request: SuggestRequest): Promise<SuggestResponse> {
    try {
      const response = await this.axios.post<SuggestResponse>(
        `${this.baseUrl}/studies/${request.study_id}/suggest`,
        {
          trial_id: request.trial_id,
          parameters: request.parameters,
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error, "Failed to get suggestions");
    }
  }

  /**
   * Report intermediate value during trial execution (for pruning)
   *
   * @param request - Study, trial IDs, step, and intermediate value
   * @returns Success confirmation
   *
   * @example
   * ```typescript
   * // Report validation accuracy at step 100
   * await optunaClient.reportIntermediateValue({
   *   study_id: 'study123',
   *   trial_id: 'trial456',
   *   step: 100,
   *   value: 0.85
   * });
   * ```
   */
  async reportIntermediateValue(
    request: ReportIntermediateValueRequest
  ): Promise<void> {
    try {
      await this.axios.post(
        `${this.baseUrl}/studies/${request.study_id}/trials/${request.trial_id}/intermediate`,
        {
          step: request.step,
          value: request.value,
        }
      );
    } catch (error) {
      throw this.handleError(error, "Failed to report intermediate value");
    }
  }

  /**
   * Check if a trial should be pruned based on intermediate values
   *
   * @param studyId - Study ID
   * @param trialId - Trial ID to check
   * @returns Whether the trial should be pruned
   *
   * @example
   * ```typescript
   * const shouldPrune = await optunaClient.shouldPrune('study123', 'trial456');
   * if (shouldPrune) {
   *   console.log('Trial pruned - stopping training early');
   * }
   * ```
   */
  async shouldPrune(studyId: string, trialId: string): Promise<boolean> {
    try {
      const response = await this.axios.get<{ should_prune: boolean }>(
        `${this.baseUrl}/studies/${studyId}/trials/${trialId}/should-prune`
      );
      return response.data.should_prune;
    } catch (error) {
      throw this.handleError(error, "Failed to check pruning status");
    }
  }

  /**
   * Delete a study and all its trials
   *
   * @param studyId - Study ID to delete
   * @returns Success confirmation
   */
  async deleteStudy(studyId: string): Promise<void> {
    try {
      await this.axios.delete(`${this.baseUrl}/studies/${studyId}`);
    } catch (error) {
      throw this.handleError(error, "Failed to delete study");
    }
  }

  /**
   * Get optimization history (convergence plot data)
   *
   * @param studyId - Study ID
   * @returns Array of best values over trial progression
   *
   * @example
   * ```typescript
   * const history = await optunaClient.getOptimizationHistory('study123');
   * // Plot best value progression: history.map((h, i) => ({ trial: i, best: h }))
   * ```
   */
  async getOptimizationHistory(
    studyId: string
  ): Promise<Array<{ trial_number: number; value: number }>> {
    try {
      const response = await this.axios.get<{
        history: Array<{ trial_number: number; value: number }>;
      }>(`${this.baseUrl}/studies/${studyId}/history`);
      return response.data.history || [];
    } catch (error) {
      throw this.handleError(error, "Failed to get optimization history");
    }
  }

  /**
   * Get parameter importance scores
   *
   * @param studyId - Study ID
   * @returns Parameter names with importance scores
   *
   * @example
   * ```typescript
   * const importance = await optunaClient.getParamImportance('study123');
   * // importance = { learning_rate: 0.45, batch_size: 0.30, num_layers: 0.25 }
   * ```
   */
  async getParamImportance(studyId: string): Promise<Record<string, number>> {
    try {
      const response = await this.axios.get<{
        importance: Record<string, number>;
      }>(`${this.baseUrl}/studies/${studyId}/importance`);
      return response.data.importance || {};
    } catch (error) {
      throw this.handleError(error, "Failed to get parameter importance");
    }
  }

  /**
   * Error handler with detailed error messages
   */
  private handleError(error: unknown, context: string): Error {
    if (error instanceof Error) {
      if ("response" in error) {
        const axiosError = error as any;
        const optunaError = axiosError.response?.data as OptunaErrorResponse;
        if (optunaError?.error) {
          return new Error(
            `${context}: ${optunaError.error}${optunaError.detail ? ` - ${optunaError.detail}` : ""}`
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

let optunaClientInstance: OptunaClient | null = null;

/**
 * Get or create Optuna client singleton instance
 *
 * @param axios - Axios instance to use for HTTP requests
 * @param baseUrl - Base URL for Optuna API (default: /api/optuna)
 * @returns Optuna client instance
 */
export function getOptunaClient(
  axios: AxiosInstance,
  baseUrl?: string
): OptunaClient {
  if (!optunaClientInstance) {
    optunaClientInstance = new OptunaClient(axios, baseUrl);
  }
  return optunaClientInstance;
}

/**
 * Reset the singleton instance (useful for testing)
 */
export function resetOptunaClient(): void {
  optunaClientInstance = null;
}
