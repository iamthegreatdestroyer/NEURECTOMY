/**
 * Deployment Orchestrator
 * Main orchestrator that coordinates all deployment strategies,
 * approval workflows, rollbacks, and GitOps integrations
 */

import { EventEmitter } from "eventemitter3";
import type { KubernetesClient } from "./kubernetes/client";
import type {
  DeploymentConfig,
  DeploymentState,
  DeploymentStrategy,
  DeploymentResult,
  DeploymentApproval,
} from "./types";
import {
  RollingUpdateStrategy,
  BlueGreenStrategy,
  CanaryStrategy,
} from "./strategies";
import { ApprovalWorkflow } from "./approval/workflow";
import { RollbackManager } from "./rollback/manager";
import { ArgoCDClient, type ArgoCDConfig } from "./gitops/argocd";
import { FluxClient, type FluxClientConfig } from "./gitops/flux";

// =============================================================================
// Types
// =============================================================================

export interface OrchestratorConfig {
  /** Kubernetes client instance */
  k8sClient: KubernetesClient;
  /** Enable ArgoCD integration */
  argocd?: ArgoCDConfig;
  /** Enable Flux integration */
  flux?: FluxClientConfig;
  /** Default namespace for deployments */
  defaultNamespace?: string;
  /** Enable telemetry/metrics */
  enableMetrics?: boolean;
  /** State persistence adapter */
  statePersistence?: StatePersistenceAdapter;
}

export interface StatePersistenceAdapter {
  save(deploymentId: string, state: DeploymentState): Promise<void>;
  load(deploymentId: string): Promise<DeploymentState | null>;
  list(filter?: {
    namespace?: string;
    status?: string;
    strategy?: string;
  }): Promise<DeploymentState[]>;
  delete(deploymentId: string): Promise<void>;
}

export interface DeploymentPlan {
  id: string;
  config: DeploymentConfig;
  strategy: DeploymentStrategy;
  steps: DeploymentStep[];
  estimatedDuration: number;
  risks: DeploymentRisk[];
  requiredApprovals: string[];
}

export interface DeploymentStep {
  id: string;
  name: string;
  description: string;
  type:
    | "preparation"
    | "validation"
    | "approval"
    | "execution"
    | "health-check"
    | "traffic-shift"
    | "cleanup"
    | "rollback";
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface DeploymentRisk {
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  mitigation: string;
}

export interface OrchestratorEvents {
  "deployment:planned": (plan: DeploymentPlan) => void;
  "deployment:started": (
    deploymentId: string,
    config: DeploymentConfig
  ) => void;
  "deployment:step:started": (
    deploymentId: string,
    step: DeploymentStep
  ) => void;
  "deployment:step:completed": (
    deploymentId: string,
    step: DeploymentStep
  ) => void;
  "deployment:step:failed": (
    deploymentId: string,
    step: DeploymentStep
  ) => void;
  "deployment:approval:required": (
    deploymentId: string,
    approval: DeploymentApproval
  ) => void;
  "deployment:approval:granted": (
    deploymentId: string,
    approver: string
  ) => void;
  "deployment:approval:rejected": (
    deploymentId: string,
    approver: string,
    reason: string
  ) => void;
  "deployment:traffic:shifted": (
    deploymentId: string,
    percentage: number
  ) => void;
  "deployment:completed": (
    deploymentId: string,
    result: DeploymentResult
  ) => void;
  "deployment:failed": (deploymentId: string, error: string) => void;
  "deployment:rollback:started": (deploymentId: string) => void;
  "deployment:rollback:completed": (deploymentId: string) => void;
  "deployment:rollback:failed": (deploymentId: string, error: string) => void;
}

// =============================================================================
// Deployment Orchestrator Implementation
// =============================================================================

export class DeploymentOrchestrator extends EventEmitter<OrchestratorEvents> {
  private k8sClient: KubernetesClient;
  private argocdClient?: ArgoCDClient;
  private fluxClient?: FluxClient;
  private defaultNamespace: string;
  private statePersistence?: StatePersistenceAdapter;

  private strategies: Map<DeploymentStrategy, any>;
  private approvalWorkflow: ApprovalWorkflow;
  private rollbackManager: RollbackManager;

  private activeDeployments: Map<string, DeploymentState>;

  constructor(config: OrchestratorConfig) {
    super();
    this.k8sClient = config.k8sClient;
    this.defaultNamespace = config.defaultNamespace || "default";
    this.statePersistence = config.statePersistence;

    // Initialize GitOps clients if configured
    if (config.argocd) {
      this.argocdClient = new ArgoCDClient(config.argocd);
    }
    if (config.flux) {
      this.fluxClient = new FluxClient(config.flux);
    }

    // Initialize strategies
    this.strategies = new Map([
      ["rolling", new RollingUpdateStrategy(this.k8sClient)],
      ["blue-green", new BlueGreenStrategy(this.k8sClient)],
      ["canary", new CanaryStrategy(this.k8sClient)],
    ]);

    // Initialize workflows
    this.approvalWorkflow = new ApprovalWorkflow();
    this.rollbackManager = new RollbackManager(this.k8sClient);

    // Track active deployments
    this.activeDeployments = new Map();

    // Wire up internal event handlers
    this.setupEventForwarding();
  }

  // ===========================================================================
  // Deployment Planning
  // ===========================================================================

  /**
   * Create a deployment plan without executing it
   */
  async plan(config: DeploymentConfig): Promise<DeploymentPlan> {
    const deploymentId = this.generateDeploymentId(config);
    const strategy = config.strategy;

    // Build deployment steps based on strategy
    const steps = this.buildDeploymentSteps(config);

    // Analyze risks
    const risks = await this.analyzeRisks(config);

    // Determine required approvals
    const requiredApprovals = this.determineRequiredApprovals(config, risks);

    // Estimate duration
    const estimatedDuration = this.estimateDuration(config, steps);

    const plan: DeploymentPlan = {
      id: deploymentId,
      config,
      strategy,
      steps,
      estimatedDuration,
      risks,
      requiredApprovals,
    };

    this.emit("deployment:planned", plan);
    return plan;
  }

  // ===========================================================================
  // Deployment Execution
  // ===========================================================================

  /**
   * Execute a deployment
   */
  async deploy(config: DeploymentConfig): Promise<DeploymentResult> {
    const deploymentId = this.generateDeploymentId(config);
    const namespace = config.namespace || this.defaultNamespace;

    // Initialize deployment state
    const state: DeploymentState = {
      id: deploymentId,
      name: config.name,
      namespace,
      environment: config.environment,
      strategy: config.strategy,
      status: "pending",
      currentVersion: "",
      targetVersion: config.image,
      trafficPercentage: 0,
      replicas: {
        desired: config.replicas,
        ready: 0,
        available: 0,
      },
      healthChecks: [],
      events: [],
      startedAt: new Date(),
    };

    this.activeDeployments.set(deploymentId, state);
    await this.persistState(state);

    this.emit("deployment:started", deploymentId, config);

    try {
      // Phase 1: Pre-deployment validation
      await this.executeStep(deploymentId, {
        id: `${deploymentId}-validate`,
        name: "Pre-deployment Validation",
        description: "Validate deployment configuration and prerequisites",
        type: "validation",
        status: "pending",
      });

      // Phase 2: Approval workflow (if required)
      if (config.approvalRequired || this.requiresApproval(config)) {
        await this.waitForApproval(deploymentId, config);
      }

      // Phase 3: Execute deployment strategy
      const strategyExecutor = this.strategies.get(config.strategy);
      if (!strategyExecutor) {
        throw new Error(`Unsupported deployment strategy: ${config.strategy}`);
      }

      const result = await strategyExecutor.deploy(config);

      // Phase 4: Health verification
      await this.executeStep(deploymentId, {
        id: `${deploymentId}-health`,
        name: "Health Verification",
        description: "Verify deployment health and readiness",
        type: "health-check",
        status: "pending",
      });

      // Phase 5: Cleanup
      await this.executeStep(deploymentId, {
        id: `${deploymentId}-cleanup`,
        name: "Cleanup",
        description: "Clean up temporary resources",
        type: "cleanup",
        status: "pending",
      });

      // Update final state
      state.status = "completed";
      state.completedAt = new Date();
      state.trafficPercentage = 100;
      state.currentVersion = config.image;
      await this.persistState(state);

      this.emit("deployment:completed", deploymentId, result);
      return result;
    } catch (error) {
      state.status = "failed";
      state.completedAt = new Date();
      await this.persistState(state);

      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.emit("deployment:failed", deploymentId, errorMessage);

      // Attempt automatic rollback if configured
      if (config.rollbackOnFailure !== false) {
        await this.rollback(deploymentId);
      }

      throw error;
    }
  }

  /**
   * Execute a canary deployment with progressive traffic shifting
   */
  async deployCanary(
    config: DeploymentConfig,
    options?: {
      initialWeight?: number;
      stepWeight?: number;
      stepInterval?: number;
      analysisInterval?: number;
    }
  ): Promise<DeploymentResult> {
    const canaryConfig = {
      ...config,
      strategy: "canary" as const,
      canaryConfig: {
        initialWeight: options?.initialWeight || 10,
        stepWeight: options?.stepWeight || 10,
        maxWeight: 100,
        stepInterval: options?.stepInterval || 60,
        analysisInterval: options?.analysisInterval || 30,
        ...config.canaryConfig,
      },
    };

    return this.deploy(canaryConfig);
  }

  /**
   * Execute a blue-green deployment
   */
  async deployBlueGreen(config: DeploymentConfig): Promise<DeploymentResult> {
    const blueGreenConfig = {
      ...config,
      strategy: "blue-green" as const,
    };

    return this.deploy(blueGreenConfig);
  }

  // ===========================================================================
  // Traffic Management
  // ===========================================================================

  /**
   * Shift traffic percentage (for canary/blue-green)
   */
  async shiftTraffic(deploymentId: string, percentage: number): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    const strategy = this.strategies.get(state.strategy);
    if (strategy?.shiftTraffic) {
      await strategy.shiftTraffic(state.namespace, state.name, percentage);
    }

    state.trafficPercentage = percentage;
    await this.persistState(state);

    this.emit("deployment:traffic:shifted", deploymentId, percentage);
  }

  /**
   * Promote canary to full production
   */
  async promote(deploymentId: string): Promise<void> {
    await this.shiftTraffic(deploymentId, 100);
  }

  // ===========================================================================
  // Rollback Operations
  // ===========================================================================

  /**
   * Rollback a deployment
   */
  async rollback(deploymentId: string, targetRevision?: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    this.emit("deployment:rollback:started", deploymentId);

    try {
      await this.rollbackManager.rollback(
        state.namespace,
        state.name,
        targetRevision
      );

      state.status = "rolled-back";
      state.completedAt = new Date();
      await this.persistState(state);

      this.emit("deployment:rollback:completed", deploymentId);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.emit("deployment:rollback:failed", deploymentId, errorMessage);
      throw error;
    }
  }

  /**
   * Get rollback history for a deployment
   */
  async getRollbackHistory(
    namespace: string,
    deploymentName: string
  ): Promise<any[]> {
    return this.rollbackManager.getHistory(namespace, deploymentName);
  }

  // ===========================================================================
  // Approval Workflow
  // ===========================================================================

  /**
   * Approve a pending deployment
   */
  async approve(
    deploymentId: string,
    approver: string,
    comment?: string
  ): Promise<void> {
    await this.approvalWorkflow.approve({
      deploymentId,
      approvers: [approver],
      status: "approved",
      comments: comment ? [comment] : undefined,
    });

    this.emit("deployment:approval:granted", deploymentId, approver);
  }

  /**
   * Reject a pending deployment
   */
  async reject(
    deploymentId: string,
    approver: string,
    reason: string
  ): Promise<void> {
    await this.approvalWorkflow.reject({
      deploymentId,
      approvers: [approver],
      status: "rejected",
      comments: [reason],
    });

    this.emit("deployment:approval:rejected", deploymentId, approver, reason);
  }

  // ===========================================================================
  // GitOps Integration
  // ===========================================================================

  /**
   * Sync with ArgoCD application
   */
  async syncArgoCD(
    appName: string,
    options?: { revision?: string }
  ): Promise<void> {
    if (!this.argocdClient) {
      throw new Error("ArgoCD integration not configured");
    }

    await this.argocdClient.syncApplication(appName, {
      revision: options?.revision,
    });
    await this.argocdClient.waitForSync(appName);
  }

  /**
   * Sync with Flux
   */
  async syncFlux(
    kind: "GitRepository" | "Kustomization" | "HelmRelease",
    name: string,
    namespace?: string
  ): Promise<void> {
    if (!this.fluxClient) {
      throw new Error("Flux integration not configured");
    }

    await this.fluxClient.reconcile(kind, name, namespace);
    await this.fluxClient.waitForReconciliation(kind, name, namespace);
  }

  // ===========================================================================
  // State Management
  // ===========================================================================

  /**
   * Get deployment state
   */
  async getDeployment(deploymentId: string): Promise<DeploymentState | null> {
    let state = this.activeDeployments.get(deploymentId);
    if (!state && this.statePersistence) {
      state = (await this.statePersistence.load(deploymentId)) || undefined;
    }
    return state || null;
  }

  /**
   * List deployments
   */
  async listDeployments(filter?: {
    namespace?: string;
    status?: string;
    strategy?: string;
  }): Promise<DeploymentState[]> {
    if (this.statePersistence) {
      return this.statePersistence.list(filter);
    }

    let deployments = Array.from(this.activeDeployments.values());

    if (filter) {
      if (filter.namespace) {
        deployments = deployments.filter(
          (d) => d.namespace === filter.namespace
        );
      }
      if (filter.status) {
        deployments = deployments.filter((d) => d.status === filter.status);
      }
      if (filter.strategy) {
        deployments = deployments.filter((d) => d.strategy === filter.strategy);
      }
    }

    return deployments;
  }

  /**
   * Cancel a running deployment
   */
  async cancel(deploymentId: string): Promise<void> {
    const state = this.activeDeployments.get(deploymentId);
    if (!state) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    if (state.status !== "running" && state.status !== "pending") {
      throw new Error(`Cannot cancel deployment in status: ${state.status}`);
    }

    // Stop the deployment strategy
    const strategy = this.strategies.get(state.strategy);
    if (strategy?.cancel) {
      await strategy.cancel(deploymentId);
    }

    state.status = "cancelled";
    state.completedAt = new Date();
    await this.persistState(state);
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private generateDeploymentId(config: DeploymentConfig): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `${config.name}-${config.strategy}-${timestamp}-${random}`;
  }

  private buildDeploymentSteps(config: DeploymentConfig): DeploymentStep[] {
    const steps: DeploymentStep[] = [
      {
        id: "validate",
        name: "Validate Configuration",
        description: "Validate deployment configuration and prerequisites",
        type: "validation",
        status: "pending",
      },
    ];

    if (config.approvalRequired || this.requiresApproval(config)) {
      steps.push({
        id: "approval",
        name: "Wait for Approval",
        description: "Wait for required approvals",
        type: "approval",
        status: "pending",
      });
    }

    // Strategy-specific steps
    switch (config.strategy) {
      case "rolling":
        steps.push({
          id: "rolling-update",
          name: "Rolling Update",
          description: "Gradually replace pods with new version",
          type: "execution",
          status: "pending",
        });
        break;

      case "blue-green":
        steps.push(
          {
            id: "deploy-green",
            name: "Deploy Green Environment",
            description: "Deploy new version to green environment",
            type: "execution",
            status: "pending",
          },
          {
            id: "verify-green",
            name: "Verify Green Environment",
            description: "Verify green environment health",
            type: "health-check",
            status: "pending",
          },
          {
            id: "switch-traffic",
            name: "Switch Traffic",
            description: "Switch traffic from blue to green",
            type: "traffic-shift",
            status: "pending",
          }
        );
        break;

      case "canary":
        steps.push(
          {
            id: "deploy-canary",
            name: "Deploy Canary",
            description: "Deploy canary with initial traffic percentage",
            type: "execution",
            status: "pending",
          },
          {
            id: "canary-analysis",
            name: "Canary Analysis",
            description: "Monitor and analyze canary metrics",
            type: "health-check",
            status: "pending",
          },
          {
            id: "progressive-rollout",
            name: "Progressive Rollout",
            description: "Gradually increase canary traffic",
            type: "traffic-shift",
            status: "pending",
          }
        );
        break;
    }

    steps.push(
      {
        id: "health-check",
        name: "Final Health Check",
        description: "Verify deployment health and readiness",
        type: "health-check",
        status: "pending",
      },
      {
        id: "cleanup",
        name: "Cleanup",
        description: "Clean up temporary resources",
        type: "cleanup",
        status: "pending",
      }
    );

    return steps;
  }

  private async analyzeRisks(
    config: DeploymentConfig
  ): Promise<DeploymentRisk[]> {
    const risks: DeploymentRisk[] = [];

    // Environment risk
    if (config.environment === "production") {
      risks.push({
        severity: "high",
        description: "Production deployment",
        mitigation: "Ensure proper approval and monitoring",
      });
    }

    // Strategy risk
    if (config.strategy === "blue-green") {
      risks.push({
        severity: "medium",
        description: "Blue-green requires 2x resources during deployment",
        mitigation: "Ensure cluster has sufficient capacity",
      });
    }

    // No rollback config
    if (config.rollbackOnFailure === false) {
      risks.push({
        severity: "high",
        description: "Automatic rollback disabled",
        mitigation: "Prepare manual rollback procedure",
      });
    }

    return risks;
  }

  private determineRequiredApprovals(
    config: DeploymentConfig,
    risks: DeploymentRisk[]
  ): string[] {
    const approvals: string[] = [];

    if (config.environment === "production") {
      approvals.push("production-approver");
    }

    if (risks.some((r) => r.severity === "critical" || r.severity === "high")) {
      approvals.push("security-reviewer");
    }

    return approvals;
  }

  private estimateDuration(
    config: DeploymentConfig,
    steps: DeploymentStep[]
  ): number {
    let estimate = 60; // Base: 1 minute

    // Per step estimate
    estimate += steps.length * 30;

    // Strategy-specific
    switch (config.strategy) {
      case "rolling":
        estimate += config.replicas * 30;
        break;
      case "blue-green":
        estimate += 120; // 2 minutes for traffic switch
        break;
      case "canary":
        estimate += (config.canaryConfig?.stepInterval || 60) * 10;
        break;
    }

    return estimate;
  }

  private requiresApproval(config: DeploymentConfig): boolean {
    return (
      config.environment === "production" || config.environment === "staging"
    );
  }

  private async waitForApproval(
    deploymentId: string,
    config: DeploymentConfig
  ): Promise<void> {
    const approval: DeploymentApproval = {
      deploymentId,
      required: true,
      approvers: [],
      status: "pending",
      timeout: 3600000, // 1 hour
    };

    this.emit("deployment:approval:required", deploymentId, approval);

    await this.approvalWorkflow.waitForApproval(approval);
  }

  private async executeStep(
    deploymentId: string,
    step: DeploymentStep
  ): Promise<void> {
    step.status = "running";
    step.startedAt = new Date();
    this.emit("deployment:step:started", deploymentId, step);

    try {
      // Execute step logic (simulated)
      await this.sleep(1000);

      step.status = "completed";
      step.completedAt = new Date();
      this.emit("deployment:step:completed", deploymentId, step);
    } catch (error) {
      step.status = "failed";
      step.completedAt = new Date();
      step.error = error instanceof Error ? error.message : String(error);
      this.emit("deployment:step:failed", deploymentId, step);
      throw error;
    }
  }

  private async persistState(state: DeploymentState): Promise<void> {
    this.activeDeployments.set(state.id, state);
    if (this.statePersistence) {
      await this.statePersistence.save(state.id, state);
    }
  }

  private setupEventForwarding(): void {
    // Forward strategy events
    for (const strategy of this.strategies.values()) {
      if (strategy.on) {
        strategy.on("progress", (data: any) => {
          // Handle progress events
        });
      }
    }

    // Forward ArgoCD events
    if (this.argocdClient) {
      this.argocdClient.on("argocd:sync:completed", (app, status) => {
        // Handle ArgoCD sync completion
      });
    }

    // Forward Flux events
    if (this.fluxClient) {
      this.fluxClient.on("flux:reconcile:completed", (resource, kind) => {
        // Handle Flux reconciliation completion
      });
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createDeploymentOrchestrator(
  config: OrchestratorConfig
): DeploymentOrchestrator {
  return new DeploymentOrchestrator(config);
}
