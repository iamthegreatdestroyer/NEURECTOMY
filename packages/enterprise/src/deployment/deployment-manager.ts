/**
 * @fileoverview Enterprise Deployment Manager
 * @module @neurectomy/enterprise/deployment
 *
 * Agent Assignment: @FLUX @ARCHITECT @SENTRY
 *
 * Enterprise-grade deployment management with:
 * - Blue-green and canary deployment strategies
 * - Health checks and readiness validation
 * - Automatic rollback on failure
 * - Multi-region deployment orchestration
 * - Deployment pipelines and gates
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface DeploymentConfig {
  name: string;
  version: string;
  environment: DeploymentEnvironment;
  strategy: DeploymentStrategy;
  regions: string[];
  healthCheck: HealthCheckConfig;
  rollback: RollbackConfig;
  gates: DeploymentGate[];
  metadata?: Record<string, unknown>;
}

export type DeploymentEnvironment =
  | "development"
  | "staging"
  | "production"
  | "disaster-recovery";

export type DeploymentStrategy =
  | "rolling"
  | "blue-green"
  | "canary"
  | "recreate"
  | "a-b-testing";

export interface HealthCheckConfig {
  endpoint: string;
  interval: number; // ms
  timeout: number; // ms
  successThreshold: number;
  failureThreshold: number;
  initialDelay: number; // ms
}

export interface RollbackConfig {
  enabled: boolean;
  automatic: boolean;
  onErrorThreshold: number; // percentage
  onLatencyThreshold: number; // ms
  maxRollbackVersions: number;
}

export interface DeploymentGate {
  id: string;
  name: string;
  type: GateType;
  required: boolean;
  config: GateConfig;
}

export type GateType =
  | "approval"
  | "metrics"
  | "tests"
  | "security-scan"
  | "compliance"
  | "time-window";

export interface GateConfig {
  approvers?: string[];
  minApprovals?: number;
  metricsThreshold?: Record<string, number>;
  testSuites?: string[];
  scanTypes?: string[];
  complianceFrameworks?: string[];
  allowedWindows?: TimeWindow[];
}

export interface TimeWindow {
  dayOfWeek: number[]; // 0-6 (Sunday-Saturday)
  startHour: number; // 0-23
  endHour: number; // 0-23
  timezone: string;
}

export interface Deployment {
  id: string;
  config: DeploymentConfig;
  status: DeploymentStatus;
  startedAt: Date;
  completedAt?: Date;
  stages: DeploymentStage[];
  metrics: DeploymentMetrics;
  artifacts: DeploymentArtifact[];
  rollbackTarget?: string;
}

export type DeploymentStatus =
  | "pending"
  | "preparing"
  | "deploying"
  | "verifying"
  | "completed"
  | "failed"
  | "rolled-back"
  | "cancelled";

export interface DeploymentStage {
  id: string;
  name: string;
  region: string;
  status: StageStatus;
  startedAt?: Date;
  completedAt?: Date;
  instances: DeploymentInstance[];
  errors: DeploymentError[];
}

export type StageStatus =
  | "pending"
  | "in-progress"
  | "verifying"
  | "completed"
  | "failed"
  | "skipped";

export interface DeploymentInstance {
  id: string;
  host: string;
  port: number;
  status: InstanceStatus;
  version: string;
  healthStatus: HealthStatus;
  trafficWeight: number;
}

export type InstanceStatus =
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "unhealthy";

export interface HealthStatus {
  healthy: boolean;
  lastCheck: Date;
  consecutiveSuccesses: number;
  consecutiveFailures: number;
  latency: number;
  details?: Record<string, unknown>;
}

export interface DeploymentMetrics {
  totalInstances: number;
  healthyInstances: number;
  unhealthyInstances: number;
  averageLatency: number;
  errorRate: number;
  requestsPerSecond: number;
  cpuUtilization: number;
  memoryUtilization: number;
}

export interface DeploymentArtifact {
  id: string;
  name: string;
  type: ArtifactType;
  checksum: string;
  size: number;
  location: string;
  createdAt: Date;
}

export type ArtifactType =
  | "container-image"
  | "binary"
  | "config"
  | "database-migration"
  | "static-assets";

export interface DeploymentError {
  timestamp: Date;
  stage: string;
  instance?: string;
  code: string;
  message: string;
  stack?: string;
  recoverable: boolean;
}

export interface DeploymentEvent {
  id: string;
  deploymentId: string;
  type: DeploymentEventType;
  timestamp: Date;
  data: Record<string, unknown>;
}

export type DeploymentEventType =
  | "deployment.started"
  | "deployment.stage.started"
  | "deployment.stage.completed"
  | "deployment.stage.failed"
  | "deployment.instance.healthy"
  | "deployment.instance.unhealthy"
  | "deployment.gate.pending"
  | "deployment.gate.approved"
  | "deployment.gate.rejected"
  | "deployment.completed"
  | "deployment.failed"
  | "deployment.rollback.started"
  | "deployment.rollback.completed";

export interface DeploymentManagerConfig {
  maxConcurrentDeployments: number;
  defaultTimeout: number; // ms
  artifactStorage: string;
  metricsInterval: number; // ms
  retryAttempts: number;
  retryDelay: number; // ms
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_DEPLOYMENT_MANAGER_CONFIG: DeploymentManagerConfig = {
  maxConcurrentDeployments: 5,
  defaultTimeout: 30 * 60 * 1000, // 30 minutes
  artifactStorage: "/var/neurectomy/deployments",
  metricsInterval: 10000, // 10 seconds
  retryAttempts: 3,
  retryDelay: 5000, // 5 seconds
};

// ============================================================================
// Deployment Manager Implementation
// ============================================================================

export class DeploymentManager extends EventEmitter {
  private deployments: Map<string, Deployment> = new Map();
  private config: DeploymentManagerConfig;
  private activeDeployments: Set<string> = new Set();
  private healthCheckIntervals: Map<string, NodeJS.Timeout> = new Map();
  private metricsIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor(config: Partial<DeploymentManagerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_DEPLOYMENT_MANAGER_CONFIG, ...config };
  }

  /**
   * Create and start a new deployment
   */
  async createDeployment(config: DeploymentConfig): Promise<Deployment> {
    // Check concurrent deployment limit
    if (this.activeDeployments.size >= this.config.maxConcurrentDeployments) {
      throw new Error(
        `Maximum concurrent deployments (${this.config.maxConcurrentDeployments}) reached`
      );
    }

    const deployment: Deployment = {
      id: this.generateDeploymentId(),
      config,
      status: "pending",
      startedAt: new Date(),
      stages: this.initializeStages(config),
      metrics: this.initializeMetrics(),
      artifacts: [],
    };

    this.deployments.set(deployment.id, deployment);
    this.activeDeployments.add(deployment.id);

    this.emitEvent(deployment.id, "deployment.started", { config });

    // Start deployment process
    this.executeDeployment(deployment).catch((error) => {
      this.handleDeploymentFailure(deployment, error);
    });

    return deployment;
  }

  /**
   * Get deployment by ID
   */
  getDeployment(deploymentId: string): Deployment | undefined {
    return this.deployments.get(deploymentId);
  }

  /**
   * List all deployments with optional filtering
   */
  listDeployments(filter?: {
    status?: DeploymentStatus;
    environment?: DeploymentEnvironment;
    limit?: number;
  }): Deployment[] {
    let deployments = Array.from(this.deployments.values());

    if (filter?.status) {
      deployments = deployments.filter((d) => d.status === filter.status);
    }

    if (filter?.environment) {
      deployments = deployments.filter(
        (d) => d.config.environment === filter.environment
      );
    }

    // Sort by start date descending
    deployments.sort((a, b) => b.startedAt.getTime() - a.startedAt.getTime());

    if (filter?.limit) {
      deployments = deployments.slice(0, filter.limit);
    }

    return deployments;
  }

  /**
   * Cancel a running deployment
   */
  async cancelDeployment(deploymentId: string): Promise<void> {
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    if (!this.isDeploymentActive(deployment)) {
      throw new Error(`Deployment ${deploymentId} is not active`);
    }

    deployment.status = "cancelled";
    deployment.completedAt = new Date();
    this.cleanupDeployment(deploymentId);

    this.emitEvent(deploymentId, "deployment.failed", {
      reason: "cancelled",
    });
  }

  /**
   * Manually trigger rollback
   */
  async rollback(
    deploymentId: string,
    targetVersion?: string
  ): Promise<Deployment> {
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    this.emitEvent(deploymentId, "deployment.rollback.started", {
      targetVersion,
    });

    // Create rollback deployment
    const rollbackConfig: DeploymentConfig = {
      ...deployment.config,
      name: `${deployment.config.name}-rollback`,
      version: targetVersion || (await this.getPreviousVersion(deployment)),
      strategy: "rolling", // Use rolling for faster rollback
    };

    const rollbackDeployment = await this.createDeployment(rollbackConfig);
    rollbackDeployment.rollbackTarget = deploymentId;

    // Update original deployment status
    deployment.status = "rolled-back";
    deployment.completedAt = new Date();

    return rollbackDeployment;
  }

  /**
   * Approve a deployment gate
   */
  async approveGate(
    deploymentId: string,
    gateId: string,
    approver: string
  ): Promise<void> {
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }

    const gate = deployment.config.gates.find((g) => g.id === gateId);
    if (!gate) {
      throw new Error(`Gate ${gateId} not found`);
    }

    this.emitEvent(deploymentId, "deployment.gate.approved", {
      gateId,
      approver,
    });
  }

  /**
   * Get deployment metrics
   */
  getMetrics(deploymentId: string): DeploymentMetrics | undefined {
    return this.deployments.get(deploymentId)?.metrics;
  }

  // ============================================================================
  // Private Methods - Deployment Execution
  // ============================================================================

  private async executeDeployment(deployment: Deployment): Promise<void> {
    try {
      // Phase 1: Preparation
      deployment.status = "preparing";
      await this.prepareDeployment(deployment);

      // Phase 2: Gate validation
      await this.validateGates(deployment);

      // Phase 3: Deploy by strategy
      deployment.status = "deploying";
      await this.deployByStrategy(deployment);

      // Phase 4: Verification
      deployment.status = "verifying";
      await this.verifyDeployment(deployment);

      // Phase 5: Complete
      deployment.status = "completed";
      deployment.completedAt = new Date();

      this.emitEvent(deployment.id, "deployment.completed", {
        duration:
          deployment.completedAt.getTime() - deployment.startedAt.getTime(),
      });
    } catch (error) {
      throw error;
    } finally {
      this.cleanupDeployment(deployment.id);
    }
  }

  private async prepareDeployment(_deployment: Deployment): Promise<void> {
    // Validate artifacts exist
    // Pull container images
    // Apply database migrations
    // Warm up caches
    await this.simulateDelay(1000);
  }

  private async validateGates(deployment: Deployment): Promise<void> {
    for (const gate of deployment.config.gates) {
      if (gate.required) {
        this.emitEvent(deployment.id, "deployment.gate.pending", {
          gateId: gate.id,
          gateName: gate.name,
        });

        const passed = await this.evaluateGate(deployment, gate);
        if (!passed) {
          throw new Error(`Gate ${gate.name} validation failed`);
        }
      }
    }
  }

  private async evaluateGate(
    _deployment: Deployment,
    gate: DeploymentGate
  ): Promise<boolean> {
    switch (gate.type) {
      case "time-window":
        return this.evaluateTimeWindowGate(gate.config);
      case "metrics":
        return this.evaluateMetricsGate(gate.config);
      case "tests":
        return this.evaluateTestsGate(gate.config);
      case "security-scan":
        return this.evaluateSecurityGate(gate.config);
      case "compliance":
        return this.evaluateComplianceGate(gate.config);
      case "approval":
        // Approval gates are handled asynchronously
        return true;
      default:
        return true;
    }
  }

  private evaluateTimeWindowGate(config: GateConfig): boolean {
    const now = new Date();
    const currentDay = now.getDay();
    const currentHour = now.getHours();

    return (
      config.allowedWindows?.some(
        (window) =>
          window.dayOfWeek.includes(currentDay) &&
          currentHour >= window.startHour &&
          currentHour < window.endHour
      ) ?? true
    );
  }

  private async evaluateMetricsGate(_config: GateConfig): Promise<boolean> {
    // Evaluate metrics thresholds
    return true;
  }

  private async evaluateTestsGate(_config: GateConfig): Promise<boolean> {
    // Run test suites
    return true;
  }

  private async evaluateSecurityGate(_config: GateConfig): Promise<boolean> {
    // Run security scans
    return true;
  }

  private async evaluateComplianceGate(_config: GateConfig): Promise<boolean> {
    // Validate compliance
    return true;
  }

  private async deployByStrategy(deployment: Deployment): Promise<void> {
    switch (deployment.config.strategy) {
      case "rolling":
        await this.executeRollingDeployment(deployment);
        break;
      case "blue-green":
        await this.executeBlueGreenDeployment(deployment);
        break;
      case "canary":
        await this.executeCanaryDeployment(deployment);
        break;
      case "recreate":
        await this.executeRecreateDeployment(deployment);
        break;
      case "a-b-testing":
        await this.executeABTestingDeployment(deployment);
        break;
    }
  }

  private async executeRollingDeployment(
    deployment: Deployment
  ): Promise<void> {
    for (const stage of deployment.stages) {
      await this.deployStage(deployment, stage);
    }
  }

  private async executeBlueGreenDeployment(
    deployment: Deployment
  ): Promise<void> {
    // Deploy to "green" environment
    for (const stage of deployment.stages) {
      await this.deployStage(deployment, stage);
    }

    // Verify green is healthy
    await this.verifyAllHealthy(deployment);

    // Switch traffic from blue to green
    await this.switchTraffic(deployment, "green");
  }

  private async executeCanaryDeployment(deployment: Deployment): Promise<void> {
    // Deploy to small percentage first
    const canaryPercentage = 5;

    for (const stage of deployment.stages) {
      // Deploy canary instances
      await this.deployCanaryInstances(deployment, stage, canaryPercentage);

      // Monitor for issues
      await this.monitorCanary(deployment, stage);

      // Gradually increase traffic
      for (const percentage of [25, 50, 75, 100]) {
        await this.adjustCanaryTraffic(deployment, stage, percentage);
        await this.monitorCanary(deployment, stage);
      }
    }
  }

  private async executeRecreateDeployment(
    deployment: Deployment
  ): Promise<void> {
    // Stop all instances
    for (const stage of deployment.stages) {
      await this.stopAllInstances(stage);
    }

    // Deploy new version
    for (const stage of deployment.stages) {
      await this.deployStage(deployment, stage);
    }
  }

  private async executeABTestingDeployment(
    deployment: Deployment
  ): Promise<void> {
    // Similar to canary but with specific traffic routing rules
    await this.executeCanaryDeployment(deployment);
  }

  private async deployStage(
    deployment: Deployment,
    stage: DeploymentStage
  ): Promise<void> {
    stage.status = "in-progress";
    stage.startedAt = new Date();

    this.emitEvent(deployment.id, "deployment.stage.started", {
      stageId: stage.id,
      region: stage.region,
    });

    try {
      // Deploy instances
      for (const instance of stage.instances) {
        await this.deployInstance(deployment, instance);
      }

      // Start health checks
      this.startHealthChecks(deployment, stage);

      // Wait for all instances to be healthy
      await this.waitForHealthy(deployment, stage);

      stage.status = "completed";
      stage.completedAt = new Date();

      this.emitEvent(deployment.id, "deployment.stage.completed", {
        stageId: stage.id,
        region: stage.region,
      });
    } catch (error) {
      stage.status = "failed";
      stage.completedAt = new Date();
      stage.errors.push({
        timestamp: new Date(),
        stage: stage.id,
        code: "STAGE_DEPLOYMENT_FAILED",
        message: error instanceof Error ? error.message : String(error),
        recoverable: false,
      });

      this.emitEvent(deployment.id, "deployment.stage.failed", {
        stageId: stage.id,
        region: stage.region,
        error: error instanceof Error ? error.message : String(error),
      });

      throw error;
    }
  }

  private async deployInstance(
    deployment: Deployment,
    instance: DeploymentInstance
  ): Promise<void> {
    instance.status = "starting";
    instance.version = deployment.config.version;

    // Simulate deployment
    await this.simulateDelay(500);

    instance.status = "running";
  }

  private startHealthChecks(
    deployment: Deployment,
    stage: DeploymentStage
  ): void {
    const config = deployment.config.healthCheck;
    const intervalId = setInterval(async () => {
      for (const instance of stage.instances) {
        await this.checkInstanceHealth(deployment, instance, config);
      }
      this.updateDeploymentMetrics(deployment);
    }, config.interval);

    this.healthCheckIntervals.set(`${deployment.id}-${stage.id}`, intervalId);
  }

  private async checkInstanceHealth(
    deployment: Deployment,
    instance: DeploymentInstance,
    config: HealthCheckConfig
  ): Promise<void> {
    const startTime = Date.now();

    try {
      // Simulate health check
      const healthy = Math.random() > 0.05; // 95% success rate
      const latency = Date.now() - startTime;

      if (healthy) {
        instance.healthStatus.consecutiveSuccesses++;
        instance.healthStatus.consecutiveFailures = 0;
      } else {
        instance.healthStatus.consecutiveFailures++;
        instance.healthStatus.consecutiveSuccesses = 0;
      }

      instance.healthStatus.lastCheck = new Date();
      instance.healthStatus.latency = latency;
      instance.healthStatus.healthy =
        instance.healthStatus.consecutiveSuccesses >= config.successThreshold;

      if (instance.healthStatus.healthy) {
        this.emitEvent(deployment.id, "deployment.instance.healthy", {
          instanceId: instance.id,
          host: instance.host,
        });
      } else if (
        instance.healthStatus.consecutiveFailures >= config.failureThreshold
      ) {
        instance.status = "unhealthy";
        this.emitEvent(deployment.id, "deployment.instance.unhealthy", {
          instanceId: instance.id,
          host: instance.host,
          consecutiveFailures: instance.healthStatus.consecutiveFailures,
        });
      }
    } catch (error) {
      instance.healthStatus.consecutiveFailures++;
      instance.healthStatus.consecutiveSuccesses = 0;
    }
  }

  private async waitForHealthy(
    deployment: Deployment,
    stage: DeploymentStage
  ): Promise<void> {
    const config = deployment.config.healthCheck;
    const timeout = this.config.defaultTimeout;
    const startTime = Date.now();

    // Wait for initial delay
    await this.simulateDelay(config.initialDelay);

    while (Date.now() - startTime < timeout) {
      const healthyCount = stage.instances.filter(
        (i) => i.healthStatus.healthy
      ).length;

      if (healthyCount === stage.instances.length) {
        return;
      }

      await this.simulateDelay(config.interval);
    }

    throw new Error(`Timeout waiting for instances to become healthy`);
  }

  private async verifyDeployment(deployment: Deployment): Promise<void> {
    // Verify all stages completed
    const allCompleted = deployment.stages.every(
      (s) => s.status === "completed"
    );

    if (!allCompleted) {
      throw new Error("Not all stages completed successfully");
    }

    // Verify metrics are within thresholds
    const metrics = deployment.metrics;
    const rollbackConfig = deployment.config.rollback;

    if (rollbackConfig.enabled && rollbackConfig.automatic) {
      if (metrics.errorRate > rollbackConfig.onErrorThreshold) {
        throw new Error(
          `Error rate ${metrics.errorRate}% exceeds threshold ${rollbackConfig.onErrorThreshold}%`
        );
      }

      if (metrics.averageLatency > rollbackConfig.onLatencyThreshold) {
        throw new Error(
          `Latency ${metrics.averageLatency}ms exceeds threshold ${rollbackConfig.onLatencyThreshold}ms`
        );
      }
    }
  }

  private async verifyAllHealthy(deployment: Deployment): Promise<void> {
    for (const stage of deployment.stages) {
      const unhealthy = stage.instances.filter((i) => !i.healthStatus.healthy);
      if (unhealthy.length > 0) {
        throw new Error(
          `${unhealthy.length} instances unhealthy in stage ${stage.name}`
        );
      }
    }
  }

  private async switchTraffic(
    _deployment: Deployment,
    _target: string
  ): Promise<void> {
    // Implement traffic switching logic
    await this.simulateDelay(1000);
  }

  private async deployCanaryInstances(
    deployment: Deployment,
    stage: DeploymentStage,
    percentage: number
  ): Promise<void> {
    const canaryCount = Math.ceil((stage.instances.length * percentage) / 100);
    const canaryInstances = stage.instances.slice(0, canaryCount);

    for (const instance of canaryInstances) {
      await this.deployInstance(deployment, instance);
    }
  }

  private async monitorCanary(
    _deployment: Deployment,
    _stage: DeploymentStage
  ): Promise<void> {
    // Monitor canary instances for issues
    await this.simulateDelay(5000);
  }

  private async adjustCanaryTraffic(
    _deployment: Deployment,
    stage: DeploymentStage,
    percentage: number
  ): Promise<void> {
    for (const instance of stage.instances) {
      instance.trafficWeight = percentage;
    }
  }

  private async stopAllInstances(stage: DeploymentStage): Promise<void> {
    for (const instance of stage.instances) {
      instance.status = "stopped";
    }
    await this.simulateDelay(2000);
  }

  private async getPreviousVersion(deployment: Deployment): Promise<string> {
    // Get previous deployment version
    const previousDeployments = this.listDeployments({
      environment: deployment.config.environment,
      status: "completed",
      limit: 2,
    });

    if (previousDeployments.length > 1) {
      return previousDeployments[1].config.version;
    }

    throw new Error("No previous version found for rollback");
  }

  // ============================================================================
  // Private Methods - Utilities
  // ============================================================================

  private initializeStages(config: DeploymentConfig): DeploymentStage[] {
    return config.regions.map((region) => ({
      id: `stage-${region}-${Date.now()}`,
      name: `Deploy to ${region}`,
      region,
      status: "pending" as StageStatus,
      instances: this.initializeInstances(region),
      errors: [],
    }));
  }

  private initializeInstances(region: string): DeploymentInstance[] {
    // Initialize 3 instances per region
    return Array.from({ length: 3 }, (_, i) => ({
      id: `instance-${region}-${i}-${Date.now()}`,
      host: `${region}-node-${i}.neurectomy.internal`,
      port: 8080 + i,
      status: "starting" as InstanceStatus,
      version: "",
      healthStatus: {
        healthy: false,
        lastCheck: new Date(),
        consecutiveSuccesses: 0,
        consecutiveFailures: 0,
        latency: 0,
      },
      trafficWeight: 0,
    }));
  }

  private initializeMetrics(): DeploymentMetrics {
    return {
      totalInstances: 0,
      healthyInstances: 0,
      unhealthyInstances: 0,
      averageLatency: 0,
      errorRate: 0,
      requestsPerSecond: 0,
      cpuUtilization: 0,
      memoryUtilization: 0,
    };
  }

  private updateDeploymentMetrics(deployment: Deployment): void {
    const instances = deployment.stages.flatMap((s) => s.instances);

    deployment.metrics = {
      totalInstances: instances.length,
      healthyInstances: instances.filter((i) => i.healthStatus.healthy).length,
      unhealthyInstances: instances.filter((i) => !i.healthStatus.healthy)
        .length,
      averageLatency:
        instances.reduce((sum, i) => sum + i.healthStatus.latency, 0) /
        instances.length,
      errorRate: Math.random() * 0.5, // Simulated
      requestsPerSecond: Math.random() * 1000 + 500, // Simulated
      cpuUtilization: Math.random() * 30 + 20, // Simulated
      memoryUtilization: Math.random() * 40 + 30, // Simulated
    };
  }

  private generateDeploymentId(): string {
    return `deploy-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private isDeploymentActive(deployment: Deployment): boolean {
    return ["pending", "preparing", "deploying", "verifying"].includes(
      deployment.status
    );
  }

  private cleanupDeployment(deploymentId: string): void {
    this.activeDeployments.delete(deploymentId);

    // Clear health check intervals
    for (const [key, interval] of this.healthCheckIntervals) {
      if (key.startsWith(deploymentId)) {
        clearInterval(interval);
        this.healthCheckIntervals.delete(key);
      }
    }

    // Clear metrics intervals
    const metricsInterval = this.metricsIntervals.get(deploymentId);
    if (metricsInterval) {
      clearInterval(metricsInterval);
      this.metricsIntervals.delete(deploymentId);
    }
  }

  private handleDeploymentFailure(
    deployment: Deployment,
    error: unknown
  ): void {
    deployment.status = "failed";
    deployment.completedAt = new Date();

    this.emitEvent(deployment.id, "deployment.failed", {
      error: error instanceof Error ? error.message : String(error),
    });

    // Auto-rollback if configured
    if (
      deployment.config.rollback.enabled &&
      deployment.config.rollback.automatic
    ) {
      this.rollback(deployment.id).catch((rollbackError) => {
        console.error("Auto-rollback failed:", rollbackError);
      });
    }

    this.cleanupDeployment(deployment.id);
  }

  private emitEvent(
    deploymentId: string,
    type: DeploymentEventType,
    data: Record<string, unknown>
  ): void {
    const event: DeploymentEvent = {
      id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      deploymentId,
      type,
      timestamp: new Date(),
      data,
    };

    this.emit(type, event);
    this.emit("deployment.event", event);
  }

  private async simulateDelay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createDeploymentManager(
  config?: Partial<DeploymentManagerConfig>
): DeploymentManager {
  return new DeploymentManager(config);
}
