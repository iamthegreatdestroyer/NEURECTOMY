/**
 * Rolling Update Deployment Strategy
 * Gradually replaces old pods with new ones
 */

import type { V1Deployment } from "@kubernetes/client-node";
import { BaseDeploymentStrategy, type DeploymentStrategyConfig } from "./base";
import type {
  DeploymentConfig,
  DeploymentState,
  RollingConfig,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface RollingDeploymentConfig extends DeploymentConfig {
  rolling?: RollingConfig;
}

// Default configuration
const DEFAULT_ROLLING_CONFIG: RollingConfig = {
  maxSurge: "25%",
  maxUnavailable: "25%",
  minReadySeconds: 10,
  progressDeadlineSeconds: 600,
};

// =============================================================================
// Rolling Strategy Implementation
// =============================================================================

export class RollingDeploymentStrategy extends BaseDeploymentStrategy {
  private activeDeployments = new Map<string, DeploymentState>();

  constructor(config: DeploymentStrategyConfig) {
    super(config);
  }

  getStrategyName(): string {
    return "rolling";
  }

  /**
   * Calculate total steps for rolling deployment
   */
  protected calculateTotalSteps(config: DeploymentConfig): number {
    // Steps: pre-deploy checks, create/update deployment, wait for rollout, verify health, post-deploy
    return 5;
  }

  /**
   * Execute rolling deployment
   */
  async execute(config: RollingDeploymentConfig): Promise<DeploymentState> {
    this.resetAbort();
    let state = this.createInitialState(config);

    const deploymentId = `${config.namespace}/${config.name}`;
    this.activeDeployments.set(deploymentId, state);

    try {
      // Phase 1: Pre-deployment checks
      state = await this.executePreDeploymentPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 2: Create or update deployment
      state = await this.executeDeploymentPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 3: Wait for rollout
      state = await this.executeRolloutPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 4: Verify health
      state = await this.executeHealthVerificationPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 5: Post-deployment
      state = await this.executePostDeploymentPhase(state, config);

      return state;
    } catch (error) {
      return this.handleError(state, error as Error);
    } finally {
      this.activeDeployments.delete(deploymentId);
    }
  }

  /**
   * Pause rolling deployment
   */
  async pause(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const [namespace, name] = deploymentId.split("/");

    // Pause the Kubernetes deployment
    await this.k8sClient.patchDeployment(name, namespace, {
      spec: {
        paused: true,
      },
    });

    const updatedState = this.updateState(state, {
      status: "paused",
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:paused", updatedState);
  }

  /**
   * Resume rolling deployment
   */
  async resume(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const [namespace, name] = deploymentId.split("/");

    // Resume the Kubernetes deployment
    await this.k8sClient.patchDeployment(name, namespace, {
      spec: {
        paused: false,
      },
    });

    const updatedState = this.updateState(state, {
      status: "in_progress",
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:resumed", updatedState);
  }

  /**
   * Cancel rolling deployment
   */
  async cancel(deploymentId: string): Promise<void> {
    this.abort();

    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const [namespace, name] = deploymentId.split("/");

    // Rollback to previous revision
    await this.rollbackDeployment(name, namespace);

    const updatedState = this.updateState(state, {
      status: "cancelled",
      completedAt: new Date(),
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:cancelled", updatedState);
  }

  // ===========================================================================
  // Deployment Phases
  // ===========================================================================

  /**
   * Phase 1: Pre-deployment checks
   */
  private async executePreDeploymentPhase(
    state: DeploymentState,
    config: RollingDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      status: "in_progress",
      phase: "pre-deployment",
      currentStep: 1,
      startedAt: new Date(),
    });
    state = this.addEvent(state, "info", "Starting pre-deployment checks");
    this.emitProgress(state);
    this.emit("deployment:started", state);

    const { success, errors } = await this.runPreDeploymentChecks(config);
    if (!success) {
      return this.handleError(
        state,
        new Error(`Pre-deployment checks failed: ${errors.join(", ")}`)
      );
    }

    state = this.addEvent(
      state,
      "info",
      "Pre-deployment checks passed successfully"
    );
    return state;
  }

  /**
   * Phase 2: Create or update deployment
   */
  private async executeDeploymentPhase(
    state: DeploymentState,
    config: RollingDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "deploying",
      currentStep: 2,
    });
    state = this.addEvent(
      state,
      "info",
      "Creating/updating Kubernetes deployment"
    );
    this.emitProgress(state);

    const rollingConfig = { ...DEFAULT_ROLLING_CONFIG, ...config.rolling };
    const deploymentSpec = this.buildDeploymentSpec(config, rollingConfig);

    // Check if deployment exists
    const existingDeployment = await this.k8sClient.getDeployment(
      config.name,
      config.namespace
    );

    if (existingDeployment) {
      // Update existing deployment
      state = this.addEvent(state, "info", "Updating existing deployment");
      await this.retry(() =>
        this.k8sClient.patchDeployment(
          config.name,
          config.namespace,
          deploymentSpec
        )
      );
    } else {
      // Create new deployment
      state = this.addEvent(state, "info", "Creating new deployment");
      await this.retry(() =>
        this.k8sClient.createDeployment(config.name, config.namespace, {
          replicas: config.replicas,
          selector: { matchLabels: { app: config.name } },
          template: {
            metadata: { labels: { app: config.name } },
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
          strategy: {
            type: "RollingUpdate",
            rollingUpdate: {
              maxSurge: rollingConfig.maxSurge,
              maxUnavailable: rollingConfig.maxUnavailable,
            },
          },
          minReadySeconds: rollingConfig.minReadySeconds,
          progressDeadlineSeconds: rollingConfig.progressDeadlineSeconds,
        })
      );
    }

    state = this.addEvent(
      state,
      "info",
      `Deployment ${existingDeployment ? "updated" : "created"}`
    );
    return state;
  }

  /**
   * Phase 3: Wait for rollout to complete
   */
  private async executeRolloutPhase(
    state: DeploymentState,
    config: RollingDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "waiting",
      currentStep: 3,
    });
    state = this.addEvent(state, "info", "Waiting for rollout to complete");
    this.emitProgress(state);

    const rollingConfig = { ...DEFAULT_ROLLING_CONFIG, ...config.rolling };
    const timeoutMs = (rollingConfig.progressDeadlineSeconds || 600) * 1000;

    const rolloutComplete = await this.waitForRollout(
      config.name,
      config.namespace,
      timeoutMs
    );

    if (!rolloutComplete) {
      return this.handleError(state, new Error("Rollout timed out or failed"));
    }

    state = this.addEvent(state, "info", "Rollout completed successfully");
    return state;
  }

  /**
   * Phase 4: Verify health
   */
  private async executeHealthVerificationPhase(
    state: DeploymentState,
    config: RollingDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "verifying",
      currentStep: 4,
    });
    state = this.addEvent(state, "info", "Verifying deployment health");
    this.emitProgress(state);

    const { status, pods } = await this.checkHealth(
      config.name,
      config.namespace
    );

    state = this.updateState(state, {
      health: status,
      pods,
    });

    if (status !== "healthy") {
      state = this.addEvent(
        state,
        "warning",
        `Health check returned: ${status}`,
        { podCount: pods.length }
      );
    } else {
      state = this.addEvent(
        state,
        "info",
        `All ${pods.length} pods are healthy`
      );
    }

    return state;
  }

  /**
   * Phase 5: Post-deployment
   */
  private async executePostDeploymentPhase(
    state: DeploymentState,
    config: RollingDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "post-deployment",
      currentStep: 5,
    });
    state = this.addEvent(state, "info", "Executing post-deployment tasks");
    this.emitProgress(state);

    // Record completion
    state = this.updateState(state, {
      status: state.health === "healthy" ? "success" : "partial",
      completedAt: new Date(),
      duration: state.startedAt
        ? Date.now() - state.startedAt.getTime()
        : undefined,
    });

    state = this.addEvent(
      state,
      "info",
      `Deployment completed with status: ${state.status}`,
      { duration: state.duration }
    );

    this.emit("deployment:completed", state);
    return state;
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  /**
   * Build deployment spec for patch operation
   */
  private buildDeploymentSpec(
    config: RollingDeploymentConfig,
    rollingConfig: RollingConfig
  ): Partial<V1Deployment> {
    return {
      spec: {
        replicas: config.replicas,
        template: {
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
        strategy: {
          type: "RollingUpdate",
          rollingUpdate: {
            maxSurge: rollingConfig.maxSurge as any,
            maxUnavailable: rollingConfig.maxUnavailable as any,
          },
        },
        minReadySeconds: rollingConfig.minReadySeconds,
        progressDeadlineSeconds: rollingConfig.progressDeadlineSeconds,
      },
    };
  }

  /**
   * Wait for rollout to complete
   */
  private async waitForRollout(
    name: string,
    namespace: string,
    timeoutMs: number
  ): Promise<boolean> {
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      if (this.aborted) {
        return false;
      }

      const deployment = await this.k8sClient.getDeployment(name, namespace);
      if (!deployment) {
        await this.sleep(this.healthCheckInterval);
        continue;
      }

      const status = deployment.status;
      if (!status) {
        await this.sleep(this.healthCheckInterval);
        continue;
      }

      // Check if rollout is complete
      const desiredReplicas = deployment.spec?.replicas || 0;
      const updatedReplicas = status.updatedReplicas || 0;
      const availableReplicas = status.availableReplicas || 0;
      const readyReplicas = status.readyReplicas || 0;

      if (
        updatedReplicas === desiredReplicas &&
        availableReplicas === desiredReplicas &&
        readyReplicas === desiredReplicas
      ) {
        return true;
      }

      // Check for failure conditions
      const conditions = status.conditions || [];
      const progressingCondition = conditions.find(
        (c) => c.type === "Progressing"
      );
      if (
        progressingCondition?.status === "False" &&
        progressingCondition?.reason === "ProgressDeadlineExceeded"
      ) {
        return false;
      }

      await this.sleep(this.healthCheckInterval);
    }

    return false;
  }

  /**
   * Rollback deployment to previous revision
   */
  private async rollbackDeployment(
    name: string,
    namespace: string
  ): Promise<void> {
    // Get current deployment
    const deployment = await this.k8sClient.getDeployment(name, namespace);
    if (!deployment) {
      throw new Error(`Deployment ${name} not found`);
    }

    // Rollback by patching with rollback annotation
    await this.k8sClient.patchDeployment(name, namespace, {
      metadata: {
        annotations: {
          "kubectl.kubernetes.io/rollback-to": "",
        },
      },
    });
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createRollingStrategy(
  config: DeploymentStrategyConfig
): RollingDeploymentStrategy {
  return new RollingDeploymentStrategy(config);
}
