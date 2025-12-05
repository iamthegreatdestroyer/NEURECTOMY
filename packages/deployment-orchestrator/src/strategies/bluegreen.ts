/**
 * Blue-Green Deployment Strategy
 * Runs two identical environments, switches traffic atomically
 */

import { BaseDeploymentStrategy, type DeploymentStrategyConfig } from "./base";
import type {
  DeploymentConfig,
  DeploymentState,
  BlueGreenConfig,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface BlueGreenDeploymentConfig extends DeploymentConfig {
  blueGreen?: BlueGreenConfig;
}

interface BlueGreenEnvironment {
  name: string;
  deploymentName: string;
  serviceName: string;
  isActive: boolean;
}

// Default configuration
const DEFAULT_BLUEGREEN_CONFIG: BlueGreenConfig = {
  activeLabel: "active",
  previewLabel: "preview",
  verificationTimeout: 300,
  autoPromote: false,
  scaleDownDelay: 300,
};

// =============================================================================
// Blue-Green Strategy Implementation
// =============================================================================

export class BlueGreenDeploymentStrategy extends BaseDeploymentStrategy {
  private activeDeployments = new Map<string, DeploymentState>();
  private environments = new Map<
    string,
    { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  >();

  constructor(config: DeploymentStrategyConfig) {
    super(config);
  }

  getStrategyName(): string {
    return "blue-green";
  }

  /**
   * Calculate total steps for blue-green deployment
   */
  protected calculateTotalSteps(config: DeploymentConfig): number {
    // Steps: pre-deploy, deploy-preview, verify-preview, switch-traffic, verify-active, scale-down-old
    return 6;
  }

  /**
   * Execute blue-green deployment
   */
  async execute(config: BlueGreenDeploymentConfig): Promise<DeploymentState> {
    this.resetAbort();
    let state = this.createInitialState(config);

    const deploymentId = `${config.namespace}/${config.name}`;
    this.activeDeployments.set(deploymentId, state);

    try {
      // Determine current environment (blue or green)
      const envState = await this.determineEnvironmentState(config);
      this.environments.set(deploymentId, envState);

      // Phase 1: Pre-deployment checks
      state = await this.executePreDeploymentPhase(state, config);
      if (state.status === "failed") return state;

      // Phase 2: Deploy to preview environment
      state = await this.executePreviewDeploymentPhase(state, config, envState);
      if (state.status === "failed") return state;

      // Phase 3: Verify preview environment
      state = await this.executePreviewVerificationPhase(
        state,
        config,
        envState
      );
      if (state.status === "failed") return state;

      // Phase 4: Switch traffic
      state = await this.executeTrafficSwitchPhase(state, config, envState);
      if (state.status === "failed") return state;

      // Phase 5: Verify active environment
      state = await this.executeActiveVerificationPhase(
        state,
        config,
        envState
      );
      if (state.status === "failed") return state;

      // Phase 6: Scale down old environment
      state = await this.executeScaleDownPhase(state, config, envState);

      return state;
    } catch (error) {
      return this.handleError(state, error as Error);
    } finally {
      this.activeDeployments.delete(deploymentId);
    }
  }

  /**
   * Pause blue-green deployment
   */
  async pause(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const updatedState = this.updateState(state, {
      status: "paused",
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:paused", updatedState);
  }

  /**
   * Resume blue-green deployment
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
   * Cancel blue-green deployment - instant rollback
   */
  async cancel(deploymentId: string): Promise<void> {
    this.abort();

    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const envState = this.environments.get(deploymentId);
    if (envState) {
      const config = state.config as BlueGreenDeploymentConfig;
      // Rollback by switching traffic back to original active environment
      await this.rollback(config, envState);
    }

    const updatedState = this.updateState(state, {
      status: "cancelled",
      completedAt: new Date(),
    });
    this.activeDeployments.set(deploymentId, updatedState);
    this.emit("deployment:cancelled", updatedState);
  }

  /**
   * Promote preview to active (manual promotion)
   */
  async promote(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state || state.status !== "paused") {
      throw new Error(`Deployment ${deploymentId} not ready for promotion`);
    }

    const envState = this.environments.get(deploymentId);
    if (!envState) {
      throw new Error(`Environment state not found for ${deploymentId}`);
    }

    const config = state.config as BlueGreenDeploymentConfig;
    await this.switchTraffic(config, envState);

    const updatedState = this.updateState(state, {
      status: "in_progress",
    });
    this.activeDeployments.set(deploymentId, updatedState);
  }

  // ===========================================================================
  // Deployment Phases
  // ===========================================================================

  /**
   * Phase 1: Pre-deployment checks
   */
  private async executePreDeploymentPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      status: "in_progress",
      phase: "pre-deployment",
      currentStep: 1,
      startedAt: new Date(),
    });
    state = this.addEvent(state, "info", "Starting blue-green deployment");
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
   * Phase 2: Deploy to preview environment
   */
  private async executePreviewDeploymentPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "deploying",
      currentStep: 2,
    });

    const preview = envState.blue.isActive ? envState.green : envState.blue;
    state = this.addEvent(
      state,
      "info",
      `Deploying to preview environment: ${preview.name}`,
      { environment: preview.name }
    );
    this.emitProgress(state);

    // Create or update preview deployment
    await this.deployToEnvironment(config, preview);

    state = this.addEvent(
      state,
      "info",
      `Preview deployment created: ${preview.deploymentName}`
    );
    return state;
  }

  /**
   * Phase 3: Verify preview environment
   */
  private async executePreviewVerificationPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<DeploymentState> {
    state = this.updateState(state, {
      phase: "verifying",
      currentStep: 3,
    });

    const preview = envState.blue.isActive ? envState.green : envState.blue;
    const blueGreenConfig = {
      ...DEFAULT_BLUEGREEN_CONFIG,
      ...config.blueGreen,
    };

    state = this.addEvent(
      state,
      "info",
      `Verifying preview environment: ${preview.name}`
    );
    this.emitProgress(state);

    // Wait for preview to be healthy
    const timeoutMs = (blueGreenConfig.verificationTimeout || 300) * 1000;
    const healthy = await this.waitForHealthy(
      preview.deploymentName,
      config.namespace,
      timeoutMs
    );

    if (!healthy) {
      return this.handleError(
        state,
        new Error(`Preview environment verification failed`)
      );
    }

    const { status, pods } = await this.checkHealth(
      preview.deploymentName,
      config.namespace
    );
    state = this.updateState(state, { health: status, pods });

    // If auto-promote is disabled, pause for manual approval
    if (!blueGreenConfig.autoPromote) {
      state = this.updateState(state, { status: "paused" });
      state = this.addEvent(
        state,
        "info",
        "Preview verified. Waiting for manual promotion.",
        { previewUrl: this.getPreviewUrl(config, preview) }
      );
      this.emit("deployment:awaiting_promotion", state);
    } else {
      state = this.addEvent(state, "info", "Preview verified, auto-promoting");
    }

    return state;
  }

  /**
   * Phase 4: Switch traffic to new environment
   */
  private async executeTrafficSwitchPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<DeploymentState> {
    // Wait if paused
    if (state.status === "paused") {
      return state;
    }

    state = this.updateState(state, {
      phase: "switching",
      currentStep: 4,
    });

    state = this.addEvent(
      state,
      "info",
      "Switching traffic to new environment"
    );
    this.emitProgress(state);

    await this.switchTraffic(config, envState);

    const preview = envState.blue.isActive ? envState.green : envState.blue;
    state = this.addEvent(
      state,
      "info",
      `Traffic switched to ${preview.name}`,
      {
        previousActive: envState.blue.isActive ? "blue" : "green",
        newActive: preview.name,
      }
    );

    return state;
  }

  /**
   * Phase 5: Verify active environment after switch
   */
  private async executeActiveVerificationPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<DeploymentState> {
    if (state.status === "paused") {
      return state;
    }

    state = this.updateState(state, {
      phase: "verifying",
      currentStep: 5,
    });

    const newActive = envState.blue.isActive ? envState.green : envState.blue;
    state = this.addEvent(
      state,
      "info",
      `Verifying active environment: ${newActive.name}`
    );
    this.emitProgress(state);

    const { status, pods } = await this.checkHealth(
      newActive.deploymentName,
      config.namespace
    );
    state = this.updateState(state, { health: status, pods });

    if (status !== "healthy") {
      // Auto-rollback if unhealthy
      state = this.addEvent(
        state,
        "warning",
        "Active environment unhealthy, initiating rollback"
      );
      await this.rollback(config, envState);
      return this.handleError(
        state,
        new Error("Active environment verification failed, rolled back")
      );
    }

    state = this.addEvent(state, "info", "Active environment verified healthy");
    return state;
  }

  /**
   * Phase 6: Scale down old environment
   */
  private async executeScaleDownPhase(
    state: DeploymentState,
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<DeploymentState> {
    if (state.status === "paused") {
      return state;
    }

    state = this.updateState(state, {
      phase: "post-deployment",
      currentStep: 6,
    });

    const blueGreenConfig = {
      ...DEFAULT_BLUEGREEN_CONFIG,
      ...config.blueGreen,
    };
    const oldActive = envState.blue.isActive ? envState.blue : envState.green;

    state = this.addEvent(
      state,
      "info",
      `Scaling down old environment: ${oldActive.name}`
    );
    this.emitProgress(state);

    // Wait before scaling down (allow rollback window)
    const scaleDownDelay = (blueGreenConfig.scaleDownDelay || 300) * 1000;
    await this.sleep(scaleDownDelay);

    // Scale down old environment to 0
    await this.k8sClient.scaleDeployment(
      oldActive.deploymentName,
      config.namespace,
      0
    );

    // Complete deployment
    state = this.updateState(state, {
      status: "success",
      completedAt: new Date(),
      duration: state.startedAt
        ? Date.now() - state.startedAt.getTime()
        : undefined,
    });

    state = this.addEvent(
      state,
      "info",
      "Blue-green deployment completed successfully",
      { duration: state.duration }
    );

    this.emit("deployment:completed", state);
    return state;
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  /**
   * Determine current blue/green environment state
   */
  private async determineEnvironmentState(
    config: BlueGreenDeploymentConfig
  ): Promise<{ blue: BlueGreenEnvironment; green: BlueGreenEnvironment }> {
    const blueGreenConfig = {
      ...DEFAULT_BLUEGREEN_CONFIG,
      ...config.blueGreen,
    };

    const blue: BlueGreenEnvironment = {
      name: "blue",
      deploymentName: `${config.name}-blue`,
      serviceName: `${config.name}-blue`,
      isActive: false,
    };

    const green: BlueGreenEnvironment = {
      name: "green",
      deploymentName: `${config.name}-green`,
      serviceName: `${config.name}-green`,
      isActive: false,
    };

    // Check active service selector
    const mainService = await this.k8sClient.getService(
      config.name,
      config.namespace
    );

    if (mainService?.spec?.selector) {
      const versionLabel = mainService.spec.selector["version"];
      if (versionLabel === "blue") {
        blue.isActive = true;
      } else if (versionLabel === "green") {
        green.isActive = true;
      }
    }

    // If no active environment, default to blue being active
    if (!blue.isActive && !green.isActive) {
      blue.isActive = true;
    }

    return { blue, green };
  }

  /**
   * Deploy to specific environment
   */
  private async deployToEnvironment(
    config: BlueGreenDeploymentConfig,
    env: BlueGreenEnvironment
  ): Promise<void> {
    const labels = {
      app: config.name,
      version: env.name,
    };

    // Check if deployment exists
    const existing = await this.k8sClient.getDeployment(
      env.deploymentName,
      config.namespace
    );

    if (existing) {
      // Update existing deployment
      await this.k8sClient.patchDeployment(
        env.deploymentName,
        config.namespace,
        {
          spec: {
            replicas: config.replicas,
            template: {
              metadata: { labels },
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
          },
        }
      );
    } else {
      // Create new deployment
      await this.k8sClient.createDeployment(
        env.deploymentName,
        config.namespace,
        {
          replicas: config.replicas,
          selector: { matchLabels: labels },
          template: {
            metadata: { labels },
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
    }

    // Wait for deployment to be ready
    await this.waitForHealthy(env.deploymentName, config.namespace);

    // Create or update environment-specific service
    const envService = await this.k8sClient.getService(
      env.serviceName,
      config.namespace
    );

    if (!envService) {
      await this.k8sClient.createService(env.serviceName, config.namespace, {
        selector: labels,
        ports: config.ports?.map((p) => ({
          port: p.servicePort || p.containerPort,
          targetPort: p.containerPort,
          protocol: p.protocol,
        })),
        type: "ClusterIP",
      });
    }
  }

  /**
   * Switch traffic to preview environment
   */
  private async switchTraffic(
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<void> {
    const newActive = envState.blue.isActive ? envState.green : envState.blue;

    // Update main service selector
    const mainService = await this.k8sClient.getService(
      config.name,
      config.namespace
    );

    if (mainService) {
      await this.k8sClient.patchService(config.name, config.namespace, {
        spec: {
          selector: {
            app: config.name,
            version: newActive.name,
          },
        },
      });
    } else {
      // Create main service pointing to new active
      await this.k8sClient.createService(config.name, config.namespace, {
        selector: {
          app: config.name,
          version: newActive.name,
        },
        ports: config.ports?.map((p) => ({
          port: p.servicePort || p.containerPort,
          targetPort: p.containerPort,
          protocol: p.protocol,
        })),
        type: "LoadBalancer",
      });
    }

    // Update environment state
    envState.blue.isActive = !envState.blue.isActive;
    envState.green.isActive = !envState.green.isActive;
  }

  /**
   * Rollback to previous active environment
   */
  private async rollback(
    config: BlueGreenDeploymentConfig,
    envState: { blue: BlueGreenEnvironment; green: BlueGreenEnvironment }
  ): Promise<void> {
    const previousActive = envState.blue.isActive
      ? envState.green
      : envState.blue;

    // Switch back to previous active
    await this.k8sClient.patchService(config.name, config.namespace, {
      spec: {
        selector: {
          app: config.name,
          version: previousActive.name,
        },
      },
    });
  }

  /**
   * Get preview URL for testing
   */
  private getPreviewUrl(
    config: BlueGreenDeploymentConfig,
    preview: BlueGreenEnvironment
  ): string {
    return `http://${preview.serviceName}.${config.namespace}.svc.cluster.local`;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createBlueGreenStrategy(
  config: DeploymentStrategyConfig
): BlueGreenDeploymentStrategy {
  return new BlueGreenDeploymentStrategy(config);
}
