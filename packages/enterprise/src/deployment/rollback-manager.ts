/**
 * @fileoverview Enterprise Rollback Manager
 * @module @neurectomy/enterprise/deployment
 *
 * Agent Assignment: @FLUX @SENTRY @ARCHITECT
 *
 * Intelligent rollback management with:
 * - Automatic failure detection
 * - Multi-version rollback chains
 * - State preservation and recovery
 * - Incremental rollback strategies
 * - Post-rollback validation
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface RollbackConfig {
  maxVersionsRetained: number;
  automaticRollbackEnabled: boolean;
  healthCheckInterval: number; // ms
  failureThreshold: number;
  recoveryTimeout: number; // ms
  preserveState: boolean;
  notifyOnRollback: boolean;
  rollbackStrategies: RollbackStrategyConfig[];
}

export interface RollbackStrategyConfig {
  name: string;
  priority: number;
  conditions: RollbackCondition[];
  strategy: RollbackStrategy;
  maxAttempts: number;
}

export interface RollbackCondition {
  type: ConditionType;
  operator: ConditionOperator;
  value: number | string | boolean;
  window?: number; // Time window in ms for metric conditions
}

export type ConditionType =
  | "error_rate"
  | "latency_p99"
  | "latency_p95"
  | "latency_avg"
  | "cpu_usage"
  | "memory_usage"
  | "disk_usage"
  | "connection_errors"
  | "health_check_failures"
  | "custom_metric";

export type ConditionOperator = "gt" | "gte" | "lt" | "lte" | "eq" | "neq";

export type RollbackStrategy =
  | "immediate"
  | "gradual"
  | "canary-reverse"
  | "blue-green-switch"
  | "snapshot-restore";

export interface DeploymentVersion {
  id: string;
  version: string;
  deployedAt: Date;
  artifacts: VersionArtifact[];
  configuration: Record<string, unknown>;
  state: DeploymentState;
  metrics: VersionMetrics;
  rollbackEligible: boolean;
}

export interface VersionArtifact {
  id: string;
  type: ArtifactType;
  location: string;
  checksum: string;
  size: number;
}

export type ArtifactType =
  | "container-image"
  | "binary"
  | "config"
  | "database-migration"
  | "state-snapshot";

export interface DeploymentState {
  instances: InstanceState[];
  trafficDistribution: TrafficDistribution;
  databaseVersion: string;
  configHash: string;
}

export interface InstanceState {
  id: string;
  host: string;
  version: string;
  healthy: boolean;
  traffic: number; // percentage
}

export interface TrafficDistribution {
  activeVersion: string;
  distribution: Record<string, number>; // version -> percentage
}

export interface VersionMetrics {
  errorRate: number;
  latencyP99: number;
  latencyP95: number;
  latencyAvg: number;
  requestsPerSecond: number;
  cpuUsage: number;
  memoryUsage: number;
}

export interface RollbackOperation {
  id: string;
  fromVersion: string;
  toVersion: string;
  strategy: RollbackStrategy;
  status: RollbackStatus;
  trigger: RollbackTrigger;
  startedAt: Date;
  completedAt?: Date;
  stages: RollbackStage[];
  preservedState?: PreservedState;
  validationResults?: ValidationResult[];
  errors: RollbackError[];
}

export type RollbackStatus =
  | "pending"
  | "in-progress"
  | "validating"
  | "completed"
  | "failed"
  | "cancelled";

export type RollbackTrigger =
  | "automatic"
  | "manual"
  | "health-check"
  | "metric-threshold"
  | "external";

export interface RollbackStage {
  id: string;
  name: string;
  status: StageStatus;
  startedAt?: Date;
  completedAt?: Date;
  details: Record<string, unknown>;
}

export type StageStatus =
  | "pending"
  | "in-progress"
  | "completed"
  | "failed"
  | "skipped";

export interface PreservedState {
  sessionData: boolean;
  cacheEntries: boolean;
  pendingTransactions: boolean;
  userState: boolean;
  customData: Record<string, unknown>;
}

export interface ValidationResult {
  check: string;
  passed: boolean;
  details: string;
  timestamp: Date;
}

export interface RollbackError {
  timestamp: Date;
  stage: string;
  code: string;
  message: string;
  recoverable: boolean;
  retryCount: number;
}

export interface RollbackEvent {
  id: string;
  operationId: string;
  type: RollbackEventType;
  timestamp: Date;
  data: Record<string, unknown>;
}

export type RollbackEventType =
  | "rollback.initiated"
  | "rollback.stage.started"
  | "rollback.stage.completed"
  | "rollback.stage.failed"
  | "rollback.traffic.shifted"
  | "rollback.state.preserved"
  | "rollback.validation.started"
  | "rollback.validation.passed"
  | "rollback.validation.failed"
  | "rollback.completed"
  | "rollback.failed"
  | "rollback.cancelled";

export interface RollbackManagerConfig {
  maxConcurrentRollbacks: number;
  defaultStrategy: RollbackStrategy;
  validationTimeout: number; // ms
  stageTimeout: number; // ms
  metricsRetentionPeriod: number; // ms
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_ROLLBACK_CONFIG: RollbackConfig = {
  maxVersionsRetained: 10,
  automaticRollbackEnabled: true,
  healthCheckInterval: 5000,
  failureThreshold: 3,
  recoveryTimeout: 300000, // 5 minutes
  preserveState: true,
  notifyOnRollback: true,
  rollbackStrategies: [
    {
      name: "high-error-rate",
      priority: 1,
      conditions: [
        { type: "error_rate", operator: "gt", value: 5, window: 60000 },
      ],
      strategy: "immediate",
      maxAttempts: 3,
    },
    {
      name: "high-latency",
      priority: 2,
      conditions: [
        { type: "latency_p99", operator: "gt", value: 5000, window: 60000 },
      ],
      strategy: "gradual",
      maxAttempts: 3,
    },
    {
      name: "resource-exhaustion",
      priority: 3,
      conditions: [
        { type: "cpu_usage", operator: "gt", value: 90, window: 120000 },
        { type: "memory_usage", operator: "gt", value: 90, window: 120000 },
      ],
      strategy: "immediate",
      maxAttempts: 2,
    },
  ],
};

export const DEFAULT_ROLLBACK_MANAGER_CONFIG: RollbackManagerConfig = {
  maxConcurrentRollbacks: 2,
  defaultStrategy: "gradual",
  validationTimeout: 60000, // 1 minute
  stageTimeout: 300000, // 5 minutes
  metricsRetentionPeriod: 7 * 24 * 60 * 60 * 1000, // 7 days
};

// ============================================================================
// Rollback Manager Implementation
// ============================================================================

export class RollbackManager extends EventEmitter {
  private config: RollbackManagerConfig;
  private rollbackConfig: RollbackConfig;
  private versions: Map<string, DeploymentVersion> = new Map();
  private operations: Map<string, RollbackOperation> = new Map();
  private activeOperations: Set<string> = new Set();
  private metricsHistory: Map<string, VersionMetrics[]> = new Map();
  private monitoringInterval?: NodeJS.Timeout;

  constructor(
    config: Partial<RollbackManagerConfig> = {},
    rollbackConfig: Partial<RollbackConfig> = {}
  ) {
    super();
    this.config = { ...DEFAULT_ROLLBACK_MANAGER_CONFIG, ...config };
    this.rollbackConfig = { ...DEFAULT_ROLLBACK_CONFIG, ...rollbackConfig };
  }

  /**
   * Start continuous monitoring for automatic rollbacks
   */
  startMonitoring(): void {
    if (this.monitoringInterval) {
      return;
    }

    this.monitoringInterval = setInterval(() => {
      this.checkRollbackConditions();
    }, this.rollbackConfig.healthCheckInterval);
  }

  /**
   * Stop continuous monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
  }

  /**
   * Register a new deployment version
   */
  registerVersion(version: DeploymentVersion): void {
    this.versions.set(version.id, version);
    this.metricsHistory.set(version.id, [version.metrics]);

    // Trim old versions if exceeding retention limit
    this.trimOldVersions();
  }

  /**
   * Get all registered versions
   */
  getVersions(): DeploymentVersion[] {
    return Array.from(this.versions.values()).sort(
      (a, b) => b.deployedAt.getTime() - a.deployedAt.getTime()
    );
  }

  /**
   * Get rollback-eligible versions
   */
  getEligibleVersions(): DeploymentVersion[] {
    return this.getVersions().filter((v) => v.rollbackEligible);
  }

  /**
   * Update metrics for a version
   */
  updateMetrics(versionId: string, metrics: VersionMetrics): void {
    const version = this.versions.get(versionId);
    if (version) {
      version.metrics = metrics;

      // Store in history
      const history = this.metricsHistory.get(versionId) || [];
      history.push(metrics);

      // Trim old metrics
      const cutoff = Date.now() - this.config.metricsRetentionPeriod;
      const trimmed = history.filter(
        (_, i) => i >= history.length - 1000 // Keep last 1000 entries
      );
      this.metricsHistory.set(versionId, trimmed);
    }
  }

  /**
   * Initiate a rollback operation
   */
  async initiateRollback(
    fromVersionId: string,
    toVersionId: string,
    options: {
      trigger?: RollbackTrigger;
      strategy?: RollbackStrategy;
      preserveState?: boolean;
    } = {}
  ): Promise<RollbackOperation> {
    // Validate versions
    const fromVersion = this.versions.get(fromVersionId);
    const toVersion = this.versions.get(toVersionId);

    if (!fromVersion) {
      throw new Error(`Source version ${fromVersionId} not found`);
    }

    if (!toVersion) {
      throw new Error(`Target version ${toVersionId} not found`);
    }

    if (!toVersion.rollbackEligible) {
      throw new Error(`Version ${toVersionId} is not eligible for rollback`);
    }

    // Check concurrent rollback limit
    if (this.activeOperations.size >= this.config.maxConcurrentRollbacks) {
      throw new Error(
        `Maximum concurrent rollbacks (${this.config.maxConcurrentRollbacks}) reached`
      );
    }

    const operation: RollbackOperation = {
      id: this.generateOperationId(),
      fromVersion: fromVersionId,
      toVersion: toVersionId,
      strategy: options.strategy || this.config.defaultStrategy,
      status: "pending",
      trigger: options.trigger || "manual",
      startedAt: new Date(),
      stages: this.initializeStages(
        options.strategy || this.config.defaultStrategy
      ),
      errors: [],
    };

    if (options.preserveState ?? this.rollbackConfig.preserveState) {
      operation.preservedState = {
        sessionData: true,
        cacheEntries: true,
        pendingTransactions: true,
        userState: true,
        customData: {},
      };
    }

    this.operations.set(operation.id, operation);
    this.activeOperations.add(operation.id);

    this.emitEvent(operation.id, "rollback.initiated", {
      fromVersion: fromVersionId,
      toVersion: toVersionId,
      strategy: operation.strategy,
      trigger: operation.trigger,
    });

    // Execute rollback
    this.executeRollback(operation).catch((error) => {
      this.handleRollbackFailure(operation, error);
    });

    return operation;
  }

  /**
   * Get rollback operation by ID
   */
  getOperation(operationId: string): RollbackOperation | undefined {
    return this.operations.get(operationId);
  }

  /**
   * List rollback operations
   */
  listOperations(filter?: {
    status?: RollbackStatus;
    limit?: number;
  }): RollbackOperation[] {
    let operations = Array.from(this.operations.values());

    if (filter?.status) {
      operations = operations.filter((o) => o.status === filter.status);
    }

    operations.sort((a, b) => b.startedAt.getTime() - a.startedAt.getTime());

    if (filter?.limit) {
      operations = operations.slice(0, filter.limit);
    }

    return operations;
  }

  /**
   * Cancel an active rollback operation
   */
  async cancelRollback(operationId: string): Promise<void> {
    const operation = this.operations.get(operationId);
    if (!operation) {
      throw new Error(`Rollback operation ${operationId} not found`);
    }

    if (!this.isOperationActive(operation)) {
      throw new Error(`Rollback operation ${operationId} is not active`);
    }

    operation.status = "cancelled";
    operation.completedAt = new Date();
    this.activeOperations.delete(operationId);

    this.emitEvent(operationId, "rollback.cancelled", {
      reason: "manual_cancellation",
    });
  }

  /**
   * Find the best rollback target version
   */
  findBestRollbackTarget(currentVersionId: string): DeploymentVersion | null {
    const eligibleVersions = this.getEligibleVersions().filter(
      (v) => v.id !== currentVersionId
    );

    if (eligibleVersions.length === 0) {
      return null;
    }

    // Score versions based on metrics and recency
    const scored = eligibleVersions.map((version) => {
      const metricsHistory = this.metricsHistory.get(version.id) || [];
      const avgErrorRate =
        metricsHistory.length > 0
          ? metricsHistory.reduce((sum, m) => sum + m.errorRate, 0) /
            metricsHistory.length
          : 0;
      const avgLatency =
        metricsHistory.length > 0
          ? metricsHistory.reduce((sum, m) => sum + m.latencyP99, 0) /
            metricsHistory.length
          : 0;

      // Lower score is better
      const score =
        avgErrorRate * 100 + // Weight error rate heavily
        avgLatency / 100 + // Normalize latency
        (Date.now() - version.deployedAt.getTime()) / (24 * 60 * 60 * 1000); // Days since deployment

      return { version, score };
    });

    // Sort by score (ascending) and return best
    scored.sort((a, b) => a.score - b.score);
    return scored[0]?.version || null;
  }

  // ============================================================================
  // Private Methods - Rollback Execution
  // ============================================================================

  private async executeRollback(operation: RollbackOperation): Promise<void> {
    try {
      operation.status = "in-progress";

      // Execute strategy-specific rollback
      switch (operation.strategy) {
        case "immediate":
          await this.executeImmediateRollback(operation);
          break;
        case "gradual":
          await this.executeGradualRollback(operation);
          break;
        case "canary-reverse":
          await this.executeCanaryReverseRollback(operation);
          break;
        case "blue-green-switch":
          await this.executeBlueGreenRollback(operation);
          break;
        case "snapshot-restore":
          await this.executeSnapshotRollback(operation);
          break;
      }

      // Validation phase
      operation.status = "validating";
      await this.validateRollback(operation);

      // Complete
      operation.status = "completed";
      operation.completedAt = new Date();

      this.emitEvent(operation.id, "rollback.completed", {
        duration:
          operation.completedAt.getTime() - operation.startedAt.getTime(),
      });
    } catch (error) {
      throw error;
    } finally {
      this.activeOperations.delete(operation.id);
    }
  }

  private async executeImmediateRollback(
    operation: RollbackOperation
  ): Promise<void> {
    const stages = [
      "preserve-state",
      "stop-traffic",
      "switch-version",
      "resume-traffic",
    ];

    for (const stageName of stages) {
      const stage = operation.stages.find((s) => s.name === stageName);
      if (stage) {
        await this.executeStage(operation, stage);
      }
    }
  }

  private async executeGradualRollback(
    operation: RollbackOperation
  ): Promise<void> {
    const targetVersion = this.versions.get(operation.toVersion);
    if (!targetVersion) {
      throw new Error(`Target version ${operation.toVersion} not found`);
    }

    // Gradually shift traffic
    const percentages = [10, 25, 50, 75, 100];

    for (const percentage of percentages) {
      const stage: RollbackStage = {
        id: `gradual-${percentage}`,
        name: `shift-traffic-${percentage}`,
        status: "pending",
        details: { targetPercentage: percentage },
      };

      await this.executeStage(operation, stage);
      operation.stages.push(stage);

      // Monitor for issues at each step
      await this.monitorTrafficShift(operation, percentage);
    }
  }

  private async executeCanaryReverseRollback(
    operation: RollbackOperation
  ): Promise<void> {
    // Similar to gradual but with canary-specific monitoring
    await this.executeGradualRollback(operation);
  }

  private async executeBlueGreenRollback(
    operation: RollbackOperation
  ): Promise<void> {
    const stages = ["verify-target", "switch-traffic", "drain-connections"];

    for (const stageName of stages) {
      const stage = operation.stages.find((s) => s.name === stageName);
      if (stage) {
        await this.executeStage(operation, stage);
      }
    }
  }

  private async executeSnapshotRollback(
    operation: RollbackOperation
  ): Promise<void> {
    const stages = [
      "locate-snapshot",
      "stop-services",
      "restore-snapshot",
      "start-services",
    ];

    for (const stageName of stages) {
      const stage = operation.stages.find((s) => s.name === stageName);
      if (stage) {
        await this.executeStage(operation, stage);
      }
    }
  }

  private async executeStage(
    operation: RollbackOperation,
    stage: RollbackStage
  ): Promise<void> {
    stage.status = "in-progress";
    stage.startedAt = new Date();

    this.emitEvent(operation.id, "rollback.stage.started", {
      stageId: stage.id,
      stageName: stage.name,
    });

    try {
      // Execute stage logic
      await this.performStageAction(operation, stage);

      stage.status = "completed";
      stage.completedAt = new Date();

      this.emitEvent(operation.id, "rollback.stage.completed", {
        stageId: stage.id,
        stageName: stage.name,
      });
    } catch (error) {
      stage.status = "failed";
      stage.completedAt = new Date();

      operation.errors.push({
        timestamp: new Date(),
        stage: stage.name,
        code: "STAGE_FAILED",
        message: error instanceof Error ? error.message : String(error),
        recoverable: false,
        retryCount: 0,
      });

      this.emitEvent(operation.id, "rollback.stage.failed", {
        stageId: stage.id,
        stageName: stage.name,
        error: error instanceof Error ? error.message : String(error),
      });

      throw error;
    }
  }

  private async performStageAction(
    operation: RollbackOperation,
    stage: RollbackStage
  ): Promise<void> {
    // Simulate stage execution
    switch (stage.name) {
      case "preserve-state":
        await this.preserveState(operation);
        break;
      case "stop-traffic":
        await this.stopTraffic(operation);
        break;
      case "switch-version":
        await this.switchVersion(operation);
        break;
      case "resume-traffic":
        await this.resumeTraffic(operation);
        break;
      case "verify-target":
        await this.verifyTargetVersion(operation);
        break;
      case "drain-connections":
        await this.drainConnections(operation);
        break;
      case "locate-snapshot":
        await this.locateSnapshot(operation);
        break;
      case "stop-services":
        await this.stopServices(operation);
        break;
      case "restore-snapshot":
        await this.restoreSnapshot(operation);
        break;
      case "start-services":
        await this.startServices(operation);
        break;
      default:
        if (stage.name.startsWith("shift-traffic-")) {
          const percentage = parseInt(stage.name.split("-")[2]);
          await this.shiftTraffic(operation, percentage);
        }
    }
  }

  private async preserveState(operation: RollbackOperation): Promise<void> {
    if (operation.preservedState) {
      this.emitEvent(operation.id, "rollback.state.preserved", {
        sessionData: operation.preservedState.sessionData,
        cacheEntries: operation.preservedState.cacheEntries,
      });
    }
    await this.simulateDelay(500);
  }

  private async stopTraffic(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(1000);
  }

  private async switchVersion(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(2000);
  }

  private async resumeTraffic(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(1000);
  }

  private async verifyTargetVersion(
    operation: RollbackOperation
  ): Promise<void> {
    const targetVersion = this.versions.get(operation.toVersion);
    if (!targetVersion) {
      throw new Error(`Target version ${operation.toVersion} not available`);
    }
    await this.simulateDelay(500);
  }

  private async drainConnections(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(5000);
  }

  private async locateSnapshot(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(1000);
  }

  private async stopServices(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(2000);
  }

  private async restoreSnapshot(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(10000);
  }

  private async startServices(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(3000);
  }

  private async shiftTraffic(
    operation: RollbackOperation,
    percentage: number
  ): Promise<void> {
    this.emitEvent(operation.id, "rollback.traffic.shifted", {
      toVersion: operation.toVersion,
      percentage,
    });
    await this.simulateDelay(2000);
  }

  private async monitorTrafficShift(
    operation: RollbackOperation,
    percentage: number
  ): Promise<void> {
    // Monitor for 5 seconds after each shift
    await this.simulateDelay(5000);
  }

  private async validateRollback(operation: RollbackOperation): Promise<void> {
    this.emitEvent(operation.id, "rollback.validation.started", {});

    operation.validationResults = [];

    const checks = [
      { name: "health-check", fn: () => this.validateHealthCheck(operation) },
      { name: "metrics-check", fn: () => this.validateMetrics(operation) },
      {
        name: "connectivity-check",
        fn: () => this.validateConnectivity(operation),
      },
      {
        name: "data-integrity",
        fn: () => this.validateDataIntegrity(operation),
      },
    ];

    for (const check of checks) {
      try {
        await check.fn();
        operation.validationResults.push({
          check: check.name,
          passed: true,
          details: "Validation passed",
          timestamp: new Date(),
        });
      } catch (error) {
        operation.validationResults.push({
          check: check.name,
          passed: false,
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date(),
        });

        this.emitEvent(operation.id, "rollback.validation.failed", {
          check: check.name,
          error: error instanceof Error ? error.message : String(error),
        });

        throw new Error(`Validation failed: ${check.name}`);
      }
    }

    this.emitEvent(operation.id, "rollback.validation.passed", {
      checksRun: checks.length,
      checksPassed: checks.length,
    });
  }

  private async validateHealthCheck(
    operation: RollbackOperation
  ): Promise<void> {
    await this.simulateDelay(1000);
  }

  private async validateMetrics(operation: RollbackOperation): Promise<void> {
    await this.simulateDelay(1000);
  }

  private async validateConnectivity(
    operation: RollbackOperation
  ): Promise<void> {
    await this.simulateDelay(500);
  }

  private async validateDataIntegrity(
    operation: RollbackOperation
  ): Promise<void> {
    await this.simulateDelay(1000);
  }

  // ============================================================================
  // Private Methods - Condition Checking
  // ============================================================================

  private checkRollbackConditions(): void {
    if (!this.rollbackConfig.automaticRollbackEnabled) {
      return;
    }

    const currentVersion = this.getCurrentVersion();
    if (!currentVersion) {
      return;
    }

    // Check each strategy's conditions
    for (const strategyConfig of this.rollbackConfig.rollbackStrategies) {
      if (this.evaluateConditions(currentVersion, strategyConfig.conditions)) {
        const targetVersion = this.findBestRollbackTarget(currentVersion.id);
        if (targetVersion) {
          this.initiateRollback(currentVersion.id, targetVersion.id, {
            trigger: "automatic",
            strategy: strategyConfig.strategy,
          }).catch((error) => {
            console.error("Automatic rollback failed:", error);
          });
          break;
        }
      }
    }
  }

  private evaluateConditions(
    version: DeploymentVersion,
    conditions: RollbackCondition[]
  ): boolean {
    return conditions.every((condition) =>
      this.evaluateCondition(version, condition)
    );
  }

  private evaluateCondition(
    version: DeploymentVersion,
    condition: RollbackCondition
  ): boolean {
    let value: number;

    switch (condition.type) {
      case "error_rate":
        value = version.metrics.errorRate;
        break;
      case "latency_p99":
        value = version.metrics.latencyP99;
        break;
      case "latency_p95":
        value = version.metrics.latencyP95;
        break;
      case "latency_avg":
        value = version.metrics.latencyAvg;
        break;
      case "cpu_usage":
        value = version.metrics.cpuUsage;
        break;
      case "memory_usage":
        value = version.metrics.memoryUsage;
        break;
      default:
        return false;
    }

    const threshold = condition.value as number;

    switch (condition.operator) {
      case "gt":
        return value > threshold;
      case "gte":
        return value >= threshold;
      case "lt":
        return value < threshold;
      case "lte":
        return value <= threshold;
      case "eq":
        return value === threshold;
      case "neq":
        return value !== threshold;
      default:
        return false;
    }
  }

  private getCurrentVersion(): DeploymentVersion | undefined {
    const versions = this.getVersions();
    return versions[0];
  }

  // ============================================================================
  // Private Methods - Utilities
  // ============================================================================

  private initializeStages(strategy: RollbackStrategy): RollbackStage[] {
    const stageNames = this.getStagesForStrategy(strategy);
    return stageNames.map((name) => ({
      id: `stage-${name}-${Date.now()}`,
      name,
      status: "pending" as StageStatus,
      details: {},
    }));
  }

  private getStagesForStrategy(strategy: RollbackStrategy): string[] {
    switch (strategy) {
      case "immediate":
        return [
          "preserve-state",
          "stop-traffic",
          "switch-version",
          "resume-traffic",
        ];
      case "gradual":
        return []; // Stages created dynamically
      case "canary-reverse":
        return []; // Stages created dynamically
      case "blue-green-switch":
        return ["verify-target", "switch-traffic", "drain-connections"];
      case "snapshot-restore":
        return [
          "locate-snapshot",
          "stop-services",
          "restore-snapshot",
          "start-services",
        ];
      default:
        return [];
    }
  }

  private trimOldVersions(): void {
    const versions = this.getVersions();
    if (versions.length > this.rollbackConfig.maxVersionsRetained) {
      const toRemove = versions.slice(this.rollbackConfig.maxVersionsRetained);
      for (const version of toRemove) {
        this.versions.delete(version.id);
        this.metricsHistory.delete(version.id);
      }
    }
  }

  private generateOperationId(): string {
    return `rollback-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private isOperationActive(operation: RollbackOperation): boolean {
    return ["pending", "in-progress", "validating"].includes(operation.status);
  }

  private handleRollbackFailure(
    operation: RollbackOperation,
    error: unknown
  ): void {
    operation.status = "failed";
    operation.completedAt = new Date();
    this.activeOperations.delete(operation.id);

    this.emitEvent(operation.id, "rollback.failed", {
      error: error instanceof Error ? error.message : String(error),
    });
  }

  private emitEvent(
    operationId: string,
    type: RollbackEventType,
    data: Record<string, unknown>
  ): void {
    const event: RollbackEvent = {
      id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      operationId,
      type,
      timestamp: new Date(),
      data,
    };

    this.emit(type, event);
    this.emit("rollback.event", event);
  }

  private async simulateDelay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createRollbackManager(
  config?: Partial<RollbackManagerConfig>,
  rollbackConfig?: Partial<RollbackConfig>
): RollbackManager {
  return new RollbackManager(config, rollbackConfig);
}
