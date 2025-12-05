/**
 * Canary Deployment Strategy
 * Progressive traffic shifting with automated analysis
 */

import { BaseDeploymentStrategy, type DeploymentStrategyConfig } from "./base";
import type {
  DeploymentConfig,
  DeploymentState,
  CanaryConfig,
  CanaryStep,
  AnalysisResult,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface CanaryDeploymentConfig extends DeploymentConfig {
  canary?: CanaryConfig;
}

interface CanaryEnvironment {
  stableDeploymentName: string;
  canaryDeploymentName: string;
  stableServiceName: string;
  canaryServiceName: string;
  currentWeight: number;
}

// Default configuration
const DEFAULT_CANARY_CONFIG: CanaryConfig = {
  steps: [
    { weight: 10, pause: "5m" },
    { weight: 25, pause: "5m" },
    { weight: 50, pause: "5m" },
    { weight: 75, pause: "5m" },
    { weight: 100 },
  ],
  analysis: {
    interval: "30s",
    threshold: 95,
    metrics: ["error_rate", "latency_p99"],
  },
  maxWeight: 100,
  stepWeight: 10,
};

// =============================================================================
// Canary Strategy Implementation
// =============================================================================

export class CanaryDeploymentStrategy extends BaseDeploymentStrategy {
  private activeDeployments = new Map<string, DeploymentState>();
  private canaryEnvironments = new Map<string, CanaryEnvironment>();
  private analysisIntervals = new Map<string, NodeJS.Timeout>();

  constructor(config: DeploymentStrategyConfig) {
    super(config);
  }

  getStrategyName(): string {
    return "canary";
  }

  /**
   * Calculate total steps for canary deployment
   */
  protected calculateTotalSteps(config: DeploymentConfig): number {
    const canaryConfig = {
      ...DEFAULT_CANARY_CONFIG,
      ...(config as CanaryDeploymentConfig).canary,
    };
    // Steps: pre-deploy, deploy-canary, (canary steps), promote, cleanup
    return 3 + (canaryConfig.steps?.length || 5);
  }

  /**
   * Execute canary deployment
   */
  async execute(config: CanaryDeploymentConfig): Promise<DeploymentState> {
    this.resetAbort();
    let state = this.createInitialState(config);

    const deploymentId = `${config.namespace}/${config.name}`;
    this.activeDeployments.set(deploymentId, state);

    const canaryEnv: CanaryEnvironment = {
      stableDeploymentName: `${config.name}-stable`,
      canaryDeploymentName: `${config.name}-canary`,
      stableServiceName: `${config.name}-stable`,
      canaryServiceName: `${config.name}-canary`,
      currentWeight: 0,
    };
    this.canaryEnvironments.set(deploymentId, canaryEnv);

    try {
      // Phase 1: Pre-deployment checks
      state = await this.executePreDeploymentPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 2: Deploy canary
      state = await this.executeCanaryDeploymentPhase(state, config, canaryEnv);
      if (state.status === "failed") return state;

      // Phase 3: Progressive traffic shifting
      state = await this.executeProgressiveRolloutPhase(
        state,
        config,
        canaryEnv
      );
      if (state.status === "failed") return state;

      // Phase 4: Promote canary to stable
      state = await this.executePromotionPhase(state, config, canaryEnv);
      if (state.status === "failed") return state;

      // Phase 5: Cleanup
      state = await this.executeCleanupPhase(state, config, canaryEnv);

      return state;
    } catch (error) {
      return this.handleError(state, error as Error);
    } finally {
      this.stopAnalysis(deploymentId);
      this.activeDeployments.delete(deploymentId);
      this.canaryEnvironments.delete(deploymentId);
    }
  }

  /**
   * Pause canary deployment
   */
  async pause(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    this.stopAnalysis(deploymentId);

    const updatedState = this.updateState(state, {
      status: "paused",
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:paused", updatedState);
  }

  /**
   * Resume canary deployment
   */
  async resume(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const updatedState = this.updateState(state, {
      status: "in_progress",
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:resumed", updatedState);
  }

  /**
   * Cancel canary deployment with rollback
   */
  async cancel(deploymentId: string): Promise<void> {
    this.abort();
    this.stopAnalysis(deploymentId);

    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const canaryEnv = this.canaryEnvironments.get(deploymentId);
    if (canaryEnv) {
      const config = state.config as CanaryDeploymentConfig;
      // Rollback: route all traffic to stable
      await this.setTrafficWeight(config, canaryEnv, 0);
      // Delete canary deployment
      await this.k8sClient.deleteDeployment(
        canaryEnv.canaryDeploymentName,
        config.namespace
      );
    }

    const updatedState = this.updateState(state, {
      status: "cancelled",
      completedAt: new Date(),
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:cancelled", updatedState);
  }

  /**
   * Manually advance to next canary step
   */
  async advance(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const canaryEnv = this.canaryEnvironments.get(deploymentId);
    if (!canaryEnv) {
      throw new Error(`Canary environment not found for ${deploymentId}`);
    }

    const config = state.config as CanaryDeploymentConfig;
    const canaryConfig = { ...DEFAULT_CANARY_CONFIG, ...config.canary };
    const steps = canaryConfig.steps || [];

    // Find current step
    const currentStepIndex = steps.findIndex(
      (s) => s.weight > canaryEnv.currentWeight
    );

    if (currentStepIndex >= 0 && currentStepIndex < steps.length) {
      const nextStep = steps[currentStepIndex];
      await this.setTrafficWeight(config, canaryEnv, nextStep.weight);
    }
  }

  // ===========================================================================
  // Deployment Phases
  // ===========================================================================

  /**
   * Phase 1: Pre-deployment checks
   */
  private async executePreDeploymentPhase(
    state: DeploymentState,
    config: CanaryDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      status: "in_progress",
      phase: "pre-deployment",
      currentStep: 1,
      startedAt: new Date(),
    });
    state = this.addEvent(state, "info", "Starting canary deployment");
    this.emitProgress(state);
    this.emit("deployment:started", state);

    const { success, errors } = await this.runPreDeploymentChecks(config);
    if (!success) {
      return this.handleError(
        state,
        new Error(`Pre-deployment checks failed: ${errors.join(", ")}`)
      );
    }

    state = this.addEvent(state, "info", "Pre-deployment checks passed");
    return state;
  }

  /**
   * Phase 2: Deploy canary version
   */
  private async executeCanaryDeploymentPhase(
    state: DeploymentState,
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "deploying",
      currentStep: 2,
    });
    state = this.addEvent(state, "info", "Deploying canary version");
    this.emitProgress(state);

    // Ensure stable deployment exists
    await this.ensureStableDeployment(config, canaryEnv);

    // Create canary deployment with 0% initial traffic
    await this.createCanaryDeployment(config, canaryEnv);

    // Wait for canary to be ready
    const healthy = await this.waitForHealthy(
      canaryEnv.canaryDeploymentName,
      config.namespace,
      300000
    );

    if (!healthy) {
      return this.handleError(
        state,
        new Error("Canary deployment failed to become healthy")
      );
    }

    state = this.addEvent(
      state,
      "info",
      `Canary deployment ${canaryEnv.canaryDeploymentName} is ready`
    );
    return state;
  }

  /**
   * Phase 3: Progressive traffic shifting
   */
  private async executeProgressiveRolloutPhase(
    state: DeploymentState,
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<DeploymentState> {
    const canaryConfig = { ...DEFAULT_CANARY_CONFIG, ...config.canary };
    const steps = canaryConfig.steps || [];

    let stepIndex = 0;
    for (const step of steps) {
      if (this.aborted) {
        return this.handleError(state, new Error("Deployment aborted"));
      }

      state = this.updateState(state, {
        phase: "canary",
        currentStep: 3 + stepIndex,
      });
      state = this.addEvent(
        state,
        "info",
        `Canary step ${stepIndex + 1}/${steps.length}: shifting ${step.weight}% traffic`,
        { weight: step.weight }
      );
      this.emitProgress(state);

      // Set traffic weight
      await this.setTrafficWeight(config, canaryEnv, step.weight);

      // Run analysis if configured
      if (canaryConfig.analysis) {
        const analysisResult = await this.runAnalysis(
          config,
          canaryEnv,
          canaryConfig
        );

        state = this.addEvent(
          state,
          analysisResult.success ? "info" : "warning",
          `Analysis result: ${analysisResult.success ? "passed" : "failed"}`,
          analysisResult.metrics
        );

        if (!analysisResult.success) {
          // Rollback on analysis failure
          state = this.addEvent(
            state,
            "error",
            "Canary analysis failed, initiating rollback"
          );
          await this.setTrafficWeight(config, canaryEnv, 0);
          await this.k8sClient.deleteDeployment(
            canaryEnv.canaryDeploymentName,
            config.namespace
          );
          return this.handleError(
            state,
            new Error(`Canary analysis failed: ${analysisResult.message}`)
          );
        }
      }

      // Pause between steps
      if (step.pause && stepIndex < steps.length - 1) {
        const pauseMs = this.parseDuration(step.pause);
        state = this.addEvent(
          state,
          "info",
          `Pausing for ${step.pause} before next step`
        );
        await this.sleep(pauseMs);
      }

      stepIndex++;
    }

    state = this.addEvent(
      state,
      "info",
      "Progressive rollout completed, all traffic shifted to canary"
    );
    return state;
  }

  /**
   * Phase 4: Promote canary to stable
   */
  private async executePromotionPhase(
    state: DeploymentState,
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "promoting",
      currentStep: state.totalSteps - 1,
    });
    state = this.addEvent(state, "info", "Promoting canary to stable");
    this.emitProgress(state);

    // Update stable deployment with canary image
    await this.k8sClient.patchDeployment(
      canaryEnv.stableDeploymentName,
      config.namespace,
      {
        spec: {
          template: {
            spec: {
              containers: [
                {
                  name: config.name,
                  image: config.image,
                },
              ],
            },
          },
        },
      }
    );

    // Wait for stable to be healthy
    await this.waitForHealthy(
      canaryEnv.stableDeploymentName,
      config.namespace,
      300000
    );

    state = this.addEvent(
      state,
      "info",
      "Canary promoted to stable deployment"
    );
    return state;
  }

  /**
   * Phase 5: Cleanup
   */
  private async executeCleanupPhase(
    state: DeploymentState,
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "post-deployment",
      currentStep: state.totalSteps,
    });
    state = this.addEvent(state, "info", "Cleaning up canary resources");
    this.emitProgress(state);

    // Route all traffic to stable
    await this.setTrafficWeight(config, canaryEnv, 0);

    // Delete canary deployment
    await this.k8sClient.deleteDeployment(
      canaryEnv.canaryDeploymentName,
      config.namespace
    );

    // Delete canary service
    try {
      await this.k8sClient.deleteService(
        canaryEnv.canaryServiceName,
        config.namespace
      );
    } catch {
      // Service may not exist
    }

    // Update final health status
    const { status, pods } = await this.checkHealth(
      canaryEnv.stableDeploymentName,
      config.namespace
    );

    state = this.updateState(state, {
      status: "success",
      health: status,
      pods,
      completedAt: new Date(),
      duration: state.startedAt
        ? Date.now() - state.startedAt.getTime()
        : undefined,
    });

    state = this.addEvent(
      state,
      "info",
      "Canary deployment completed successfully",
      { duration: state.duration }
    );

    this.emit("deployment:completed", state);
    return state;
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  /**
   * Ensure stable deployment exists
   */
  private async ensureStableDeployment(
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<void> {
    const stable = await this.k8sClient.getDeployment(
      canaryEnv.stableDeploymentName,
      config.namespace
    );

    if (!stable) {
      // Check if there's an existing deployment with base name
      const existing = await this.k8sClient.getDeployment(
        config.name,
        config.namespace
      );

      if (existing) {
        // Rename existing to stable (by creating stable and deleting original)
        await this.k8sClient.createDeployment(
          canaryEnv.stableDeploymentName,
          config.namespace,
          {
            replicas: existing.spec?.replicas || config.replicas,
            selector: {
              matchLabels: {
                app: config.name,
                track: "stable",
              },
            },
            template: {
              metadata: {
                labels: {
                  app: config.name,
                  track: "stable",
                },
              },
              spec: existing.spec?.template?.spec,
            },
          }
        );
        await this.k8sClient.deleteDeployment(config.name, config.namespace);
      } else {
        throw new Error(
          `No existing deployment found. Please deploy ${config.name} first before running canary.`
        );
      }
    }

    // Ensure stable service exists
    const stableService = await this.k8sClient.getService(
      canaryEnv.stableServiceName,
      config.namespace
    );

    if (!stableService) {
      await this.k8sClient.createService(
        canaryEnv.stableServiceName,
        config.namespace,
        {
          selector: {
            app: config.name,
            track: "stable",
          },
          ports: config.ports?.map((p) => ({
            port: p.servicePort || p.containerPort,
            targetPort: p.containerPort,
            protocol: p.protocol,
          })),
          type: "ClusterIP",
        }
      );
    }
  }

  /**
   * Create canary deployment
   */
  private async createCanaryDeployment(
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment
  ): Promise<void> {
    const canaryLabels = {
      app: config.name,
      track: "canary",
    };

    // Delete existing canary if present
    try {
      await this.k8sClient.deleteDeployment(
        canaryEnv.canaryDeploymentName,
        config.namespace
      );
      await this.sleep(2000); // Wait for deletion
    } catch {
      // Ignore if doesn't exist
    }

    // Create canary deployment with new image
    await this.k8sClient.createDeployment(
      canaryEnv.canaryDeploymentName,
      config.namespace,
      {
        replicas: Math.max(1, Math.floor(config.replicas * 0.1)), // Start with 10% replicas
        selector: { matchLabels: canaryLabels },
        template: {
          metadata: { labels: canaryLabels },
          spec: {
            containers: [
              {
                name: config.name,
                image: config.image,
                ports: config.ports?.map((p) => ({
                  containerPort: p.containerPort,
                  protocol: p.protocol,
                })),
                env: config.env?.map((e) => ({
                  name: e.name,
                  value: e.value,
                })),
                resources: config.resources,
              },
            ],
          },
        },
      }
    );

    // Create canary service
    try {
      await this.k8sClient.deleteService(
        canaryEnv.canaryServiceName,
        config.namespace
      );
    } catch {
      // Ignore
    }

    await this.k8sClient.createService(
      canaryEnv.canaryServiceName,
      config.namespace,
      {
        selector: canaryLabels,
        ports: config.ports?.map((p) => ({
          port: p.servicePort || p.containerPort,
          targetPort: p.containerPort,
          protocol: p.protocol,
        })),
        type: "ClusterIP",
      }
    );
  }

  /**
   * Set traffic weight for canary
   * Note: This uses a simplified approach. In production, use:
   * - Istio VirtualService
   * - Linkerd TrafficSplit
   * - NGINX Ingress annotations
   * - AWS ALB weighted target groups
   */
  private async setTrafficWeight(
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment,
    weight: number
  ): Promise<void> {
    canaryEnv.currentWeight = weight;

    // Calculate replica ratio based on weight
    const totalReplicas = config.replicas;
    const canaryReplicas = Math.max(
      0,
      Math.round((totalReplicas * weight) / 100)
    );
    const stableReplicas = totalReplicas - canaryReplicas;

    // Scale deployments
    await this.k8sClient.scaleDeployment(
      canaryEnv.stableDeploymentName,
      config.namespace,
      stableReplicas
    );

    await this.k8sClient.scaleDeployment(
      canaryEnv.canaryDeploymentName,
      config.namespace,
      canaryReplicas
    );

    // Update main service to include both deployments via shared label
    const mainService = await this.k8sClient.getService(
      config.name,
      config.namespace
    );

    if (mainService) {
      await this.k8sClient.patchService(config.name, config.namespace, {
        spec: {
          selector: {
            app: config.name,
            // Remove 'track' selector to load-balance between both
          },
        },
      });
    } else {
      await this.k8sClient.createService(config.name, config.namespace, {
        selector: { app: config.name },
        ports: config.ports?.map((p) => ({
          port: p.servicePort || p.containerPort,
          targetPort: p.containerPort,
          protocol: p.protocol,
        })),
        type: "LoadBalancer",
      });
    }

    this.emit("traffic:shifted", {
      stable: stableReplicas,
      canary: canaryReplicas,
      weight,
    });
  }

  /**
   * Run canary analysis
   */
  private async runAnalysis(
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment,
    canaryConfig: CanaryConfig
  ): Promise<AnalysisResult> {
    const analysis = canaryConfig.analysis;
    if (!analysis) {
      return { success: true, message: "No analysis configured" };
    }

    const metrics: Record<string, number> = {};
    const intervalMs = this.parseDuration(analysis.interval || "30s");

    // Collect metrics for the analysis interval
    await this.sleep(intervalMs);

    // Simulate metric collection (in production, query Prometheus/Datadog)
    // This would be replaced with actual metric queries
    for (const metric of analysis.metrics || []) {
      switch (metric) {
        case "error_rate":
          // Query actual error rate from metrics backend
          metrics[metric] = await this.getMetric(config, canaryEnv, metric);
          break;
        case "latency_p99":
          metrics[metric] = await this.getMetric(config, canaryEnv, metric);
          break;
        case "latency_p95":
          metrics[metric] = await this.getMetric(config, canaryEnv, metric);
          break;
        case "success_rate":
          metrics[metric] = await this.getMetric(config, canaryEnv, metric);
          break;
        default:
          metrics[metric] = await this.getMetric(config, canaryEnv, metric);
      }
    }

    // Check against threshold
    const threshold = analysis.threshold || 95;
    const successRate =
      metrics["success_rate"] || metrics["error_rate"]
        ? 100 - (metrics["error_rate"] || 0)
        : 100;

    const success = successRate >= threshold;

    return {
      success,
      message: success
        ? "Canary analysis passed"
        : `Success rate ${successRate}% below threshold ${threshold}%`,
      metrics,
    };
  }

  /**
   * Get metric value (stub - implement with your metrics backend)
   */
  private async getMetric(
    config: CanaryDeploymentConfig,
    canaryEnv: CanaryEnvironment,
    metric: string
  ): Promise<number> {
    // In production, this would query Prometheus, Datadog, etc.
    // Example Prometheus query:
    // rate(http_requests_total{app="myapp",track="canary",status=~"5.."}[5m]) /
    // rate(http_requests_total{app="myapp",track="canary"}[5m]) * 100

    // Return mock successful metrics for now
    switch (metric) {
      case "error_rate":
        return Math.random() * 2; // 0-2% error rate
      case "latency_p99":
        return 100 + Math.random() * 50; // 100-150ms
      case "latency_p95":
        return 50 + Math.random() * 30; // 50-80ms
      case "success_rate":
        return 98 + Math.random() * 2; // 98-100%
      default:
        return 0;
    }
  }

  /**
   * Stop analysis interval
   */
  private stopAnalysis(deploymentId: string): void {
    const interval = this.analysisIntervals.get(deploymentId);
    if (interval) {
      clearInterval(interval);
      this.analysisIntervals.delete(deploymentId);
    }
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createCanaryStrategy(
  config: DeploymentStrategyConfig
): CanaryDeploymentStrategy {
  return new CanaryDeploymentStrategy(config);
}
