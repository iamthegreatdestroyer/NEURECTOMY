/**
 * Base Deployment Strategy
 * Abstract class providing common functionality for all deployment strategies
 */

import { EventEmitter } from "eventemitter3";
import type {
  DeploymentConfig,
  DeploymentState,
  DeploymentStatus,
  DeploymentPhase,
  HealthStatus,
  PodStatus,
  DeploymentEventRecord,
  DeploymentEvents,
  AnalysisResult,
  ApprovalStatus,
} from "../types";
import type { KubernetesClient } from "../kubernetes/client";

// =============================================================================
// Types
// =============================================================================

export interface DeploymentStrategyConfig {
  /** Kubernetes client */
  k8sClient: KubernetesClient;
  /** Health check interval in ms */
  healthCheckInterval?: number;
  /** Max retries for operations */
  maxRetries?: number;
  /** Retry delay in ms */
  retryDelay?: number;
}

export interface DeploymentProgress {
  currentStep: number;
  totalSteps: number;
  message: string;
  percentage: number;
}

// =============================================================================
// Base Strategy
// =============================================================================

export abstract class BaseDeploymentStrategy extends EventEmitter<DeploymentEvents> {
  protected k8sClient: KubernetesClient;
  protected healthCheckInterval: number;
  protected maxRetries: number;
  protected retryDelay: number;
  protected aborted: boolean = false;

  constructor(config: DeploymentStrategyConfig) {
    super();
    this.k8sClient = config.k8sClient;
    this.healthCheckInterval = config.healthCheckInterval || 5000;
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  /**
   * Execute the deployment
   */
  abstract execute(config: DeploymentConfig): Promise<DeploymentState>;

  /**
   * Pause the deployment
   */
  abstract pause(deploymentId: string): Promise<void>;

  /**
   * Resume a paused deployment
   */
  abstract resume(deploymentId: string): Promise<void>;

  /**
   * Cancel the deployment
   */
  abstract cancel(deploymentId: string): Promise<void>;

  /**
   * Get the strategy name
   */
  abstract getStrategyName(): string;

  // ===========================================================================
  // Protected Helper Methods
  // ===========================================================================

  /**
   * Create initial deployment state
   */
  protected createInitialState(config: DeploymentConfig): DeploymentState {
    return {
      config,
      status: "pending",
      phase: "pre-deployment",
      currentStep: 0,
      totalSteps: this.calculateTotalSteps(config),
      health: "unknown",
      pods: [],
      events: [],
    };
  }

  /**
   * Calculate total steps for deployment
   */
  protected abstract calculateTotalSteps(config: DeploymentConfig): number;

  /**
   * Update deployment state
   */
  protected updateState(
    state: DeploymentState,
    updates: Partial<DeploymentState>
  ): DeploymentState {
    return { ...state, ...updates };
  }

  /**
   * Add event to deployment state
   */
  protected addEvent(
    state: DeploymentState,
    type: string,
    message: string,
    metadata?: Record<string, unknown>
  ): DeploymentState {
    const event: DeploymentEventRecord = {
      type,
      message,
      timestamp: new Date(),
      metadata,
    };

    return this.updateState(state, {
      events: [...state.events, event],
    });
  }

  /**
   * Check health of deployment pods
   */
  protected async checkHealth(
    name: string,
    namespace: string
  ): Promise<{ status: HealthStatus; pods: PodStatus[] }> {
    try {
      const pods = await this.k8sClient.getPodStatuses(name, namespace);

      if (pods.length === 0) {
        return { status: "unknown", pods };
      }

      const allReady = pods.every((p) => p.ready);
      const someReady = pods.some((p) => p.ready);

      let status: HealthStatus;
      if (allReady) {
        status = "healthy";
      } else if (someReady) {
        status = "degraded";
      } else {
        status = "unhealthy";
      }

      return { status, pods };
    } catch (error) {
      return { status: "unknown", pods: [] };
    }
  }

  /**
   * Wait for deployment to be healthy
   */
  protected async waitForHealthy(
    name: string,
    namespace: string,
    timeoutMs: number = 300000
  ): Promise<boolean> {
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      if (this.aborted) {
        return false;
      }

      const { status } = await this.checkHealth(name, namespace);
      if (status === "healthy") {
        return true;
      }

      await this.sleep(this.healthCheckInterval);
    }

    return false;
  }

  /**
   * Run pre-deployment checks
   */
  protected async runPreDeploymentChecks(
    config: DeploymentConfig
  ): Promise<{ success: boolean; errors: string[] }> {
    const errors: string[] = [];

    // Check if namespace exists
    try {
      await this.k8sClient.ensureNamespace(config.namespace);
    } catch (error) {
      errors.push(`Failed to ensure namespace: ${error}`);
    }

    // Validate image format
    if (!config.image.includes(":")) {
      errors.push("Image should include a tag (e.g., app:v1.0.0)");
    }

    // Check resource limits
    if (config.resources) {
      if (!config.resources.limits?.memory) {
        errors.push("Memory limits should be specified");
      }
      if (!config.resources.limits?.cpu) {
        errors.push("CPU limits should be specified");
      }
    }

    return { success: errors.length === 0, errors };
  }

  /**
   * Emit progress event
   */
  protected emitProgress(state: DeploymentState): void {
    this.emit("deployment:progress", state);
  }

  /**
   * Handle deployment error
   */
  protected handleError(state: DeploymentState, error: Error): DeploymentState {
    const errorState = this.updateState(state, {
      status: "failed",
      error: error.message,
      completedAt: new Date(),
      duration: state.startedAt
        ? Date.now() - state.startedAt.getTime()
        : undefined,
    });

    this.emit("deployment:failed", { ...errorState, error: error.message });
    return errorState;
  }

  /**
   * Retry an operation with exponential backoff
   */
  protected async retry<T>(
    operation: () => Promise<T>,
    maxRetries: number = this.maxRetries,
    delay: number = this.retryDelay
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        if (attempt < maxRetries - 1) {
          await this.sleep(delay * Math.pow(2, attempt));
        }
      }
    }

    throw lastError;
  }

  /**
   * Sleep for specified milliseconds
   */
  protected sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Abort deployment
   */
  protected abort(): void {
    this.aborted = true;
  }

  /**
   * Reset abort flag
   */
  protected resetAbort(): void {
    this.aborted = false;
  }

  /**
   * Parse duration string to milliseconds
   */
  protected parseDuration(duration: string): number {
    const match = duration.match(/^(\d+)(ms|s|m|h|d)$/);
    if (!match) {
      return parseInt(duration, 10);
    }

    const value = parseInt(match[1], 10);
    const unit = match[2];

    switch (unit) {
      case "ms":
        return value;
      case "s":
        return value * 1000;
      case "m":
        return value * 60 * 1000;
      case "h":
        return value * 60 * 60 * 1000;
      case "d":
        return value * 24 * 60 * 60 * 1000;
      default:
        return value;
    }
  }
}
