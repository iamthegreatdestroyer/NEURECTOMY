/**
 * NEURECTOMY Chaos Simulator
 * @module @neurectomy/experimentation-engine/chaos
 * @agent @ECLIPSE @FORTRESS
 *
 * Chaos engineering framework with controlled failure injection,
 * blast radius control, and recovery validation.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

// ============================================================================
// Schemas
// ============================================================================

export const FaultTypeSchema = z.enum([
  "latency",
  "error",
  "timeout",
  "cpu_stress",
  "memory_stress",
  "disk_stress",
  "network_partition",
  "packet_loss",
  "bandwidth_limit",
  "dns_failure",
  "process_kill",
  "container_stop",
  "node_drain",
  "custom",
]);

export const FaultSeveritySchema = z.enum([
  "low",
  "medium",
  "high",
  "critical",
]);

export const ExperimentStateSchema = z.enum([
  "draft",
  "scheduled",
  "running",
  "paused",
  "completed",
  "failed",
  "aborted",
  "rolled_back",
]);

export const FaultConfigSchema = z.object({
  id: z.string().optional(),
  type: FaultTypeSchema,
  name: z.string(),
  description: z.string().optional(),
  severity: FaultSeveritySchema.default("medium"),
  parameters: z.record(z.unknown()).optional(),
  duration: z.number().positive().optional(),
  probability: z.number().min(0).max(1).default(1),
});

export const TargetSelectorSchema = z.object({
  type: z.enum(["service", "container", "pod", "node", "process", "network"]),
  selector: z.record(z.string()),
  percentage: z.number().min(0).max(100).default(100),
  count: z.number().int().positive().optional(),
  exclude: z.array(z.string()).optional(),
});

export const BlastRadiusConfigSchema = z.object({
  maxAffectedTargets: z.number().int().positive().optional(),
  maxAffectedPercentage: z.number().min(0).max(100).default(50),
  excludeNamespaces: z.array(z.string()).optional(),
  excludeServices: z.array(z.string()).optional(),
  excludeLabels: z.record(z.string()).optional(),
  requireApproval: z.boolean().default(false),
  approvers: z.array(z.string()).optional(),
});

export const SafetyConfigSchema = z.object({
  enableKillSwitch: z.boolean().default(true),
  maxDuration: z.number().positive().default(3600000), // 1 hour default
  healthCheckInterval: z.number().positive().default(10000), // 10s
  healthCheckThreshold: z.number().min(0).max(1).default(0.8),
  rollbackOnFailure: z.boolean().default(true),
  notifyOnStart: z.boolean().default(true),
  notifyOnEnd: z.boolean().default(true),
  pauseOnAlert: z.boolean().default(true),
  alertThresholds: z
    .object({
      errorRate: z.number().min(0).max(1).optional(),
      latencyP99: z.number().positive().optional(),
      availabilityDrop: z.number().min(0).max(1).optional(),
    })
    .optional(),
});

export const HealthCheckConfigSchema = z.object({
  enabled: z.boolean().default(true),
  endpoints: z.array(
    z.object({
      url: z.string().url(),
      method: z.enum(["GET", "POST", "HEAD"]).default("GET"),
      expectedStatus: z.number().int().default(200),
      timeout: z.number().positive().default(5000),
      interval: z.number().positive().default(10000),
    })
  ),
  metrics: z
    .array(
      z.object({
        query: z.string(),
        threshold: z.number(),
        comparison: z.enum(["lt", "lte", "gt", "gte", "eq"]),
      })
    )
    .optional(),
});

export const ChaosExperimentConfigSchema = z.object({
  id: z.string().optional(),
  name: z.string(),
  description: z.string().optional(),
  hypothesis: z.string(),
  faults: z.array(FaultConfigSchema).min(1),
  targets: z.array(TargetSelectorSchema).min(1),
  blastRadius: BlastRadiusConfigSchema.optional(),
  safety: SafetyConfigSchema.optional(),
  healthChecks: HealthCheckConfigSchema.optional(),
  schedule: z
    .object({
      startTime: z.date().optional(),
      duration: z.number().positive(),
      warmupPeriod: z.number().nonnegative().default(0),
      cooldownPeriod: z.number().nonnegative().default(0),
    })
    .optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.unknown()).optional(),
});

// ============================================================================
// Types
// ============================================================================

export type FaultType = z.infer<typeof FaultTypeSchema>;
export type FaultSeverity = z.infer<typeof FaultSeveritySchema>;
export type ExperimentState = z.infer<typeof ExperimentStateSchema>;
export type FaultConfig = z.infer<typeof FaultConfigSchema>;
export type TargetSelector = z.infer<typeof TargetSelectorSchema>;
export type BlastRadiusConfig = z.infer<typeof BlastRadiusConfigSchema>;
export type SafetyConfig = z.infer<typeof SafetyConfigSchema>;
export type HealthCheckConfig = z.infer<typeof HealthCheckConfigSchema>;
export type ChaosExperimentConfig = z.infer<typeof ChaosExperimentConfigSchema>;

export interface ChaosExperiment {
  id: string;
  config: ChaosExperimentConfig;
  state: ExperimentState;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  affectedTargets: AffectedTarget[];
  activeFaults: ActiveFault[];
  healthStatus: HealthStatus;
  results?: ExperimentResults;
  approvals: Approval[];
}

export interface AffectedTarget {
  id: string;
  type: string;
  selector: Record<string, string>;
  faults: string[];
  status: "pending" | "affected" | "recovered" | "failed";
  affectedAt?: Date;
  recoveredAt?: Date;
}

export interface ActiveFault {
  id: string;
  faultConfigId: string;
  targetId: string;
  status: "pending" | "injecting" | "active" | "rolling_back" | "rolled_back";
  injectedAt?: Date;
  rolledBackAt?: Date;
  error?: string;
}

export interface HealthStatus {
  healthy: boolean;
  score: number;
  checks: HealthCheckResult[];
  lastChecked: Date;
}

export interface HealthCheckResult {
  endpoint?: string;
  metric?: string;
  healthy: boolean;
  value?: number;
  threshold?: number;
  error?: string;
  checkedAt: Date;
}

export interface ExperimentResults {
  hypothesis: string;
  hypothesisValidated: boolean;
  summary: string;
  metrics: MetricSnapshot[];
  timeline: TimelineEvent[];
  affectedTargetsCount: number;
  totalDuration: number;
  recoveryTime?: number;
  findings: Finding[];
}

export interface MetricSnapshot {
  name: string;
  baseline: number;
  duringExperiment: number;
  afterRecovery: number;
  degradation: number;
  recovered: boolean;
}

export interface TimelineEvent {
  timestamp: Date;
  type:
    | "start"
    | "fault_inject"
    | "health_check"
    | "alert"
    | "rollback"
    | "recovery"
    | "complete"
    | "abort";
  description: string;
  data?: Record<string, unknown>;
}

export interface Finding {
  severity: FaultSeverity;
  title: string;
  description: string;
  recommendation: string;
  affectedComponents: string[];
}

export interface Approval {
  approver: string;
  approved: boolean;
  timestamp: Date;
  comment?: string;
}

// ============================================================================
// Events
// ============================================================================

export interface ChaosSimulatorEvents {
  "experiment:created": { experiment: ChaosExperiment };
  "experiment:scheduled": { experimentId: string; startTime: Date };
  "experiment:started": { experimentId: string };
  "experiment:paused": { experimentId: string; reason: string };
  "experiment:resumed": { experimentId: string };
  "experiment:completed": { experimentId: string; results: ExperimentResults };
  "experiment:failed": { experimentId: string; error: string };
  "experiment:aborted": { experimentId: string; reason: string };
  "fault:injecting": {
    experimentId: string;
    faultId: string;
    targetId: string;
  };
  "fault:injected": { experimentId: string; faultId: string; targetId: string };
  "fault:failed": { experimentId: string; faultId: string; error: string };
  "fault:rolling_back": { experimentId: string; faultId: string };
  "fault:rolled_back": { experimentId: string; faultId: string };
  "health:checked": { experimentId: string; status: HealthStatus };
  "health:degraded": { experimentId: string; score: number };
  "health:critical": { experimentId: string; score: number };
  "safety:triggered": { experimentId: string; reason: string };
  "approval:requested": { experimentId: string; approvers: string[] };
  "approval:received": { experimentId: string; approval: Approval };
  "target:affected": { experimentId: string; target: AffectedTarget };
  "target:recovered": { experimentId: string; targetId: string };
}

// ============================================================================
// Fault Injector Interface
// ============================================================================

export interface FaultInjector {
  type: FaultType;
  inject(target: AffectedTarget, config: FaultConfig): Promise<string>;
  rollback(faultId: string): Promise<void>;
  verify(faultId: string): Promise<boolean>;
}

// ============================================================================
// Chaos Simulator
// ============================================================================

export class ChaosSimulator extends EventEmitter<ChaosSimulatorEvents> {
  private experiments: Map<string, ChaosExperiment> = new Map();
  private injectors: Map<FaultType, FaultInjector> = new Map();
  private healthCheckIntervals: Map<string, NodeJS.Timeout> = new Map();
  private killSwitchActive = false;

  constructor(
    private readonly options: {
      storage?: ChaosStorage;
      notifier?: ChaosNotifier;
      metricsProvider?: MetricsProvider;
      defaultSafety?: Partial<SafetyConfig>;
    } = {}
  ) {
    super();
    this.registerDefaultInjectors();
  }

  // --------------------------------------------------------------------------
  // Experiment Management
  // --------------------------------------------------------------------------

  /**
   * Create a new chaos experiment
   */
  async createExperiment(
    config: ChaosExperimentConfig
  ): Promise<ChaosExperiment> {
    const validated = ChaosExperimentConfigSchema.parse(config);

    const experiment: ChaosExperiment = {
      id: validated.id || uuidv4(),
      config: {
        ...validated,
        safety: {
          ...this.options.defaultSafety,
          ...validated.safety,
        },
      },
      state: "draft",
      createdAt: new Date(),
      affectedTargets: [],
      activeFaults: [],
      healthStatus: {
        healthy: true,
        score: 1,
        checks: [],
        lastChecked: new Date(),
      },
      approvals: [],
    };

    this.experiments.set(experiment.id, experiment);

    if (this.options.storage) {
      await this.options.storage.saveExperiment(experiment);
    }

    this.emit("experiment:created", { experiment });
    return experiment;
  }

  /**
   * Start a chaos experiment
   */
  async startExperiment(experimentId: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    if (experiment.state !== "draft" && experiment.state !== "scheduled") {
      throw new Error(`Cannot start experiment in state: ${experiment.state}`);
    }

    // Check blast radius approval if required
    if (experiment.config.blastRadius?.requireApproval) {
      const approvalCount = experiment.approvals.filter(
        (a) => a.approved
      ).length;
      const requiredApprovers =
        experiment.config.blastRadius.approvers?.length || 1;

      if (approvalCount < requiredApprovers) {
        throw new Error(
          `Experiment requires ${requiredApprovers} approvals, has ${approvalCount}`
        );
      }
    }

    // Validate targets
    const targets = await this.resolveTargets(experiment.config.targets);
    const blastRadius = experiment.config.blastRadius;

    if (
      blastRadius?.maxAffectedTargets &&
      targets.length > blastRadius.maxAffectedTargets
    ) {
      throw new Error(
        `Target count ${targets.length} exceeds max blast radius ${blastRadius.maxAffectedTargets}`
      );
    }

    experiment.state = "running";
    experiment.startedAt = new Date();
    experiment.affectedTargets = targets.map((t) => ({
      id: uuidv4(),
      type: t.type,
      selector: t.selector,
      faults: [],
      status: "pending" as const,
    }));

    this.emit("experiment:started", { experimentId });

    if (this.options.notifier && experiment.config.safety?.notifyOnStart) {
      await this.options.notifier.notify({
        type: "experiment_started",
        experimentId,
        experimentName: experiment.config.name,
      });
    }

    // Start health checks
    this.startHealthChecks(experimentId);

    // Inject faults with warmup period
    if (experiment.config.schedule?.warmupPeriod) {
      await this.delay(experiment.config.schedule.warmupPeriod);
    }

    try {
      await this.injectFaults(experimentId);

      // Run for specified duration
      const duration = experiment.config.schedule?.duration || 60000;
      await this.runExperimentDuration(experimentId, duration);

      // Complete experiment
      await this.completeExperiment(experimentId);
    } catch (error) {
      await this.handleExperimentError(experimentId, error);
    }
  }

  /**
   * Pause a running experiment
   */
  async pauseExperiment(experimentId: string, reason: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    if (experiment.state !== "running") {
      throw new Error(`Cannot pause experiment in state: ${experiment.state}`);
    }

    experiment.state = "paused";
    this.emit("experiment:paused", { experimentId, reason });
  }

  /**
   * Resume a paused experiment
   */
  async resumeExperiment(experimentId: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    if (experiment.state !== "paused") {
      throw new Error(`Cannot resume experiment in state: ${experiment.state}`);
    }

    experiment.state = "running";
    this.emit("experiment:resumed", { experimentId });
  }

  /**
   * Abort an experiment immediately
   */
  async abortExperiment(experimentId: string, reason: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    experiment.state = "aborted";
    this.stopHealthChecks(experimentId);

    // Rollback all active faults
    await this.rollbackAllFaults(experimentId);

    this.emit("experiment:aborted", { experimentId, reason });

    if (this.options.notifier) {
      await this.options.notifier.notify({
        type: "experiment_aborted",
        experimentId,
        experimentName: experiment.config.name,
        reason,
      });
    }
  }

  /**
   * Activate global kill switch - stops all experiments
   */
  async activateKillSwitch(reason: string): Promise<void> {
    this.killSwitchActive = true;

    for (const [experimentId, experiment] of this.experiments) {
      if (experiment.state === "running" || experiment.state === "paused") {
        await this.abortExperiment(
          experimentId,
          `Kill switch activated: ${reason}`
        );
      }
    }
  }

  /**
   * Deactivate kill switch
   */
  deactivateKillSwitch(): void {
    this.killSwitchActive = false;
  }

  /**
   * Add approval to experiment
   */
  async addApproval(
    experimentId: string,
    approval: Omit<Approval, "timestamp">
  ): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    experiment.approvals.push({
      ...approval,
      timestamp: new Date(),
    });

    this.emit("approval:received", {
      experimentId,
      approval: experiment.approvals[experiment.approvals.length - 1],
    });
  }

  // --------------------------------------------------------------------------
  // Fault Injection
  // --------------------------------------------------------------------------

  /**
   * Register a fault injector
   */
  registerInjector(injector: FaultInjector): void {
    this.injectors.set(injector.type, injector);
  }

  private async injectFaults(experimentId: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    for (const target of experiment.affectedTargets) {
      for (const faultConfig of experiment.config.faults) {
        if (this.killSwitchActive) {
          throw new Error("Kill switch is active");
        }

        // Check probability
        if (Math.random() > faultConfig.probability) {
          continue;
        }

        const injector = this.injectors.get(faultConfig.type);
        if (!injector) {
          console.warn(
            `No injector registered for fault type: ${faultConfig.type}`
          );
          continue;
        }

        const activeFault: ActiveFault = {
          id: uuidv4(),
          faultConfigId: faultConfig.id || faultConfig.type,
          targetId: target.id,
          status: "injecting",
        };

        experiment.activeFaults.push(activeFault);
        this.emit("fault:injecting", {
          experimentId,
          faultId: activeFault.id,
          targetId: target.id,
        });

        try {
          await injector.inject(target, faultConfig);
          activeFault.status = "active";
          activeFault.injectedAt = new Date();
          target.status = "affected";
          target.affectedAt = new Date();
          target.faults.push(activeFault.id);

          this.emit("fault:injected", {
            experimentId,
            faultId: activeFault.id,
            targetId: target.id,
          });
          this.emit("target:affected", { experimentId, target });
        } catch (error) {
          activeFault.status = "failed";
          activeFault.error =
            error instanceof Error ? error.message : String(error);
          this.emit("fault:failed", {
            experimentId,
            faultId: activeFault.id,
            error: activeFault.error,
          });
        }
      }
    }
  }

  private async rollbackAllFaults(experimentId: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    for (const fault of experiment.activeFaults) {
      if (fault.status === "active") {
        fault.status = "rolling_back";
        this.emit("fault:rolling_back", { experimentId, faultId: fault.id });

        const faultConfig = experiment.config.faults.find(
          (f) => (f.id || f.type) === fault.faultConfigId
        );

        if (faultConfig) {
          const injector = this.injectors.get(faultConfig.type);
          if (injector) {
            try {
              await injector.rollback(fault.id);
              fault.status = "rolled_back";
              fault.rolledBackAt = new Date();
              this.emit("fault:rolled_back", {
                experimentId,
                faultId: fault.id,
              });
            } catch (error) {
              console.error(`Failed to rollback fault ${fault.id}:`, error);
            }
          }
        }
      }
    }

    // Update target states
    for (const target of experiment.affectedTargets) {
      if (target.status === "affected") {
        target.status = "recovered";
        target.recoveredAt = new Date();
        this.emit("target:recovered", { experimentId, targetId: target.id });
      }
    }
  }

  // --------------------------------------------------------------------------
  // Health Checks
  // --------------------------------------------------------------------------

  private startHealthChecks(experimentId: string): void {
    const experiment = this.getExperiment(experimentId);
    const config = experiment.config.healthChecks;

    if (!config?.enabled) return;

    const interval = experiment.config.safety?.healthCheckInterval || 10000;

    const checkInterval = setInterval(async () => {
      if (experiment.state !== "running") {
        clearInterval(checkInterval);
        return;
      }

      const status = await this.performHealthChecks(experiment);
      experiment.healthStatus = status;

      this.emit("health:checked", { experimentId, status });

      // Check safety thresholds
      const threshold = experiment.config.safety?.healthCheckThreshold || 0.8;

      if (status.score < threshold) {
        this.emit("health:degraded", { experimentId, score: status.score });

        if (status.score < 0.5) {
          this.emit("health:critical", { experimentId, score: status.score });

          if (experiment.config.safety?.rollbackOnFailure) {
            this.emit("safety:triggered", {
              experimentId,
              reason: `Health score critical: ${status.score}`,
            });
            await this.abortExperiment(
              experimentId,
              `Health check failed: score ${status.score}`
            );
          }
        }
      }
    }, interval);

    this.healthCheckIntervals.set(experimentId, checkInterval);
  }

  private stopHealthChecks(experimentId: string): void {
    const interval = this.healthCheckIntervals.get(experimentId);
    if (interval) {
      clearInterval(interval);
      this.healthCheckIntervals.delete(experimentId);
    }
  }

  private async performHealthChecks(
    experiment: ChaosExperiment
  ): Promise<HealthStatus> {
    const config = experiment.config.healthChecks;
    const checks: HealthCheckResult[] = [];

    // Endpoint checks
    if (config?.endpoints) {
      for (const endpoint of config.endpoints) {
        try {
          const start = Date.now();
          const response = await fetch(endpoint.url, {
            method: endpoint.method,
            signal: AbortSignal.timeout(endpoint.timeout),
          });
          const duration = Date.now() - start;

          checks.push({
            endpoint: endpoint.url,
            healthy: response.status === endpoint.expectedStatus,
            value: duration,
            checkedAt: new Date(),
          });
        } catch (error) {
          checks.push({
            endpoint: endpoint.url,
            healthy: false,
            error: error instanceof Error ? error.message : String(error),
            checkedAt: new Date(),
          });
        }
      }
    }

    // Metric checks
    if (config?.metrics && this.options.metricsProvider) {
      for (const metric of config.metrics) {
        try {
          const value = await this.options.metricsProvider.query(metric.query);
          let healthy = false;

          switch (metric.comparison) {
            case "lt":
              healthy = value < metric.threshold;
              break;
            case "lte":
              healthy = value <= metric.threshold;
              break;
            case "gt":
              healthy = value > metric.threshold;
              break;
            case "gte":
              healthy = value >= metric.threshold;
              break;
            case "eq":
              healthy = value === metric.threshold;
              break;
          }

          checks.push({
            metric: metric.query,
            healthy,
            value,
            threshold: metric.threshold,
            checkedAt: new Date(),
          });
        } catch (error) {
          checks.push({
            metric: metric.query,
            healthy: false,
            error: error instanceof Error ? error.message : String(error),
            checkedAt: new Date(),
          });
        }
      }
    }

    const healthyCount = checks.filter((c) => c.healthy).length;
    const score = checks.length > 0 ? healthyCount / checks.length : 1;

    return {
      healthy: score >= (experiment.config.safety?.healthCheckThreshold || 0.8),
      score,
      checks,
      lastChecked: new Date(),
    };
  }

  // --------------------------------------------------------------------------
  // Experiment Execution
  // --------------------------------------------------------------------------

  private async runExperimentDuration(
    experimentId: string,
    duration: number
  ): Promise<void> {
    const startTime = Date.now();
    const checkInterval = 1000;

    while (Date.now() - startTime < duration) {
      const experiment = this.experiments.get(experimentId);

      if (!experiment || experiment.state !== "running") {
        break;
      }

      if (this.killSwitchActive) {
        throw new Error("Kill switch activated during experiment");
      }

      await this.delay(checkInterval);
    }
  }

  private async completeExperiment(experimentId: string): Promise<void> {
    const experiment = this.getExperiment(experimentId);

    // Stop health checks
    this.stopHealthChecks(experimentId);

    // Rollback faults
    await this.rollbackAllFaults(experimentId);

    // Cooldown period
    if (experiment.config.schedule?.cooldownPeriod) {
      await this.delay(experiment.config.schedule.cooldownPeriod);
    }

    // Generate results
    const results = await this.generateResults(experiment);

    experiment.state = "completed";
    experiment.completedAt = new Date();
    experiment.results = results;

    if (this.options.storage) {
      await this.options.storage.saveExperiment(experiment);
    }

    this.emit("experiment:completed", { experimentId, results });

    if (this.options.notifier && experiment.config.safety?.notifyOnEnd) {
      await this.options.notifier.notify({
        type: "experiment_completed",
        experimentId,
        experimentName: experiment.config.name,
        results,
      });
    }
  }

  private async handleExperimentError(
    experimentId: string,
    error: unknown
  ): Promise<void> {
    const experiment = this.getExperiment(experimentId);
    const errorMessage = error instanceof Error ? error.message : String(error);

    this.stopHealthChecks(experimentId);

    if (experiment.config.safety?.rollbackOnFailure) {
      await this.rollbackAllFaults(experimentId);
    }

    experiment.state = "failed";
    experiment.completedAt = new Date();

    this.emit("experiment:failed", { experimentId, error: errorMessage });

    if (this.options.notifier) {
      await this.options.notifier.notify({
        type: "experiment_failed",
        experimentId,
        experimentName: experiment.config.name,
        error: errorMessage,
      });
    }
  }

  private async generateResults(
    experiment: ChaosExperiment
  ): Promise<ExperimentResults> {
    const timeline: TimelineEvent[] = [];

    timeline.push({
      timestamp: experiment.startedAt || experiment.createdAt,
      type: "start",
      description: "Experiment started",
    });

    for (const fault of experiment.activeFaults) {
      if (fault.injectedAt) {
        timeline.push({
          timestamp: fault.injectedAt,
          type: "fault_inject",
          description: `Fault ${fault.faultConfigId} injected`,
          data: { faultId: fault.id, targetId: fault.targetId },
        });
      }

      if (fault.rolledBackAt) {
        timeline.push({
          timestamp: fault.rolledBackAt,
          type: "rollback",
          description: `Fault ${fault.faultConfigId} rolled back`,
          data: { faultId: fault.id },
        });
      }
    }

    timeline.push({
      timestamp: new Date(),
      type: "complete",
      description: "Experiment completed",
    });

    timeline.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const totalDuration = experiment.startedAt
      ? Date.now() - experiment.startedAt.getTime()
      : 0;

    const recoveryTimes = experiment.affectedTargets
      .filter((t) => t.affectedAt && t.recoveredAt)
      .map((t) => t.recoveredAt!.getTime() - t.affectedAt!.getTime());

    const avgRecoveryTime =
      recoveryTimes.length > 0
        ? recoveryTimes.reduce((a, b) => a + b, 0) / recoveryTimes.length
        : undefined;

    return {
      hypothesis: experiment.config.hypothesis,
      hypothesisValidated: experiment.healthStatus.healthy,
      summary: this.generateSummary(experiment),
      metrics: [],
      timeline,
      affectedTargetsCount: experiment.affectedTargets.filter(
        (t) => t.status === "affected" || t.status === "recovered"
      ).length,
      totalDuration,
      recoveryTime: avgRecoveryTime,
      findings: this.generateFindings(experiment),
    };
  }

  private generateSummary(experiment: ChaosExperiment): string {
    const faultCount = experiment.activeFaults.length;
    const targetCount = experiment.affectedTargets.length;
    const successRate = experiment.healthStatus.score;

    return (
      `Chaos experiment "${experiment.config.name}" completed. ` +
      `Injected ${faultCount} faults across ${targetCount} targets. ` +
      `Final health score: ${(successRate * 100).toFixed(1)}%. ` +
      `Hypothesis ${experiment.healthStatus.healthy ? "validated" : "not validated"}.`
    );
  }

  private generateFindings(experiment: ChaosExperiment): Finding[] {
    const findings: Finding[] = [];

    // Check for slow recovery
    const slowTargets = experiment.affectedTargets.filter((t) => {
      if (!t.affectedAt || !t.recoveredAt) return false;
      const recoveryTime = t.recoveredAt.getTime() - t.affectedAt.getTime();
      return recoveryTime > 60000; // > 60 seconds
    });

    if (slowTargets.length > 0) {
      findings.push({
        severity: "medium",
        title: "Slow Recovery Detected",
        description: `${slowTargets.length} targets took longer than 60 seconds to recover`,
        recommendation:
          "Review auto-healing mechanisms and health check configurations",
        affectedComponents: slowTargets.map((t) => JSON.stringify(t.selector)),
      });
    }

    // Check for failed faults
    const failedFaults = experiment.activeFaults.filter(
      (f) => f.status === "failed"
    );

    if (failedFaults.length > 0) {
      findings.push({
        severity: "low",
        title: "Fault Injection Failures",
        description: `${failedFaults.length} faults failed to inject properly`,
        recommendation:
          "Check fault injector configurations and target accessibility",
        affectedComponents: failedFaults.map((f) => f.faultConfigId),
      });
    }

    // Check for health degradation
    if (!experiment.healthStatus.healthy) {
      findings.push({
        severity: "high",
        title: "System Health Degradation",
        description: `System health dropped to ${(experiment.healthStatus.score * 100).toFixed(1)}%`,
        recommendation:
          "Improve resilience patterns and implement circuit breakers",
        affectedComponents: experiment.healthStatus.checks
          .filter((c) => !c.healthy)
          .map((c) => c.endpoint || c.metric || "unknown"),
      });
    }

    return findings;
  }

  // --------------------------------------------------------------------------
  // Target Resolution
  // --------------------------------------------------------------------------

  private async resolveTargets(
    selectors: TargetSelector[]
  ): Promise<TargetSelector[]> {
    // In a real implementation, this would query the container runtime
    // or Kubernetes API to resolve selectors to actual targets
    return selectors;
  }

  // --------------------------------------------------------------------------
  // Utility Methods
  // --------------------------------------------------------------------------

  private getExperiment(experimentId: string): ChaosExperiment {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment not found: ${experimentId}`);
    }
    return experiment;
  }

  getExperimentById(experimentId: string): ChaosExperiment | undefined {
    return this.experiments.get(experimentId);
  }

  listExperiments(): ChaosExperiment[] {
    return Array.from(this.experiments.values());
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private registerDefaultInjectors(): void {
    // Register default fault injectors
    this.registerInjector(new LatencyInjector());
    this.registerInjector(new ErrorInjector());
    this.registerInjector(new TimeoutInjector());
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    for (const interval of this.healthCheckIntervals.values()) {
      clearInterval(interval);
    }
    this.healthCheckIntervals.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Storage Interface
// ============================================================================

export interface ChaosStorage {
  saveExperiment(experiment: ChaosExperiment): Promise<void>;
  loadExperiment(experimentId: string): Promise<ChaosExperiment | null>;
  listExperiments(
    filter?: Partial<ChaosExperiment>
  ): Promise<ChaosExperiment[]>;
  deleteExperiment(experimentId: string): Promise<void>;
}

// ============================================================================
// Notifier Interface
// ============================================================================

export interface ChaosNotifier {
  notify(notification: ChaosNotification): Promise<void>;
}

export interface ChaosNotification {
  type:
    | "experiment_started"
    | "experiment_completed"
    | "experiment_aborted"
    | "experiment_failed";
  experimentId: string;
  experimentName: string;
  reason?: string;
  error?: string;
  results?: ExperimentResults;
}

// ============================================================================
// Metrics Provider Interface
// ============================================================================

export interface MetricsProvider {
  query(query: string): Promise<number>;
  queryRange(
    query: string,
    start: Date,
    end: Date
  ): Promise<Array<{ timestamp: Date; value: number }>>;
}

// ============================================================================
// Default Fault Injectors
// ============================================================================

class LatencyInjector implements FaultInjector {
  type: FaultType = "latency";
  private activeFaults: Map<string, { cleanup: () => void }> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const latencyMs = (config.parameters?.latencyMs as number) || 1000;

    // In a real implementation, this would inject latency via proxy or iptables
    console.log(`Injecting ${latencyMs}ms latency to target ${target.id}`);

    this.activeFaults.set(faultId, {
      cleanup: () => {
        console.log(`Cleaning up latency fault ${faultId}`);
      },
    });

    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const fault = this.activeFaults.get(faultId);
    if (fault) {
      fault.cleanup();
      this.activeFaults.delete(faultId);
    }
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

class ErrorInjector implements FaultInjector {
  type: FaultType = "error";
  private activeFaults: Map<string, { cleanup: () => void }> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const errorRate = (config.parameters?.errorRate as number) || 0.5;
    const statusCode = (config.parameters?.statusCode as number) || 500;

    console.log(
      `Injecting ${errorRate * 100}% error rate (${statusCode}) to target ${target.id}`
    );

    this.activeFaults.set(faultId, {
      cleanup: () => {
        console.log(`Cleaning up error fault ${faultId}`);
      },
    });

    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const fault = this.activeFaults.get(faultId);
    if (fault) {
      fault.cleanup();
      this.activeFaults.delete(faultId);
    }
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

class TimeoutInjector implements FaultInjector {
  type: FaultType = "timeout";
  private activeFaults: Map<string, { cleanup: () => void }> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const timeoutMs = (config.parameters?.timeoutMs as number) || 30000;

    console.log(`Injecting ${timeoutMs}ms timeout to target ${target.id}`);

    this.activeFaults.set(faultId, {
      cleanup: () => {
        console.log(`Cleaning up timeout fault ${faultId}`);
      },
    });

    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const fault = this.activeFaults.get(faultId);
    if (fault) {
      fault.cleanup();
      this.activeFaults.delete(faultId);
    }
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}
