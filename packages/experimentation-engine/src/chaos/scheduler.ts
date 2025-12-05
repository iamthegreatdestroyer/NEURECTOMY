/**
 * NEURECTOMY Chaos Experiment Scheduler
 * @module @neurectomy/experimentation-engine/chaos
 * @agent @ECLIPSE @FLUX
 *
 * Scheduling system for automated chaos experiments with
 * cron-like execution, dependencies, and coordination.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type { ChaosExperimentConfig, ChaosExperiment } from "./simulator";
import { ChaosSimulator } from "./simulator";

// ============================================================================
// Schedule Configuration Schemas
// ============================================================================

export const CronExpressionSchema = z.string().refine(
  (val) => {
    // Basic cron validation: 5 or 6 fields
    const fields = val.trim().split(/\s+/);
    return fields.length >= 5 && fields.length <= 6;
  },
  { message: "Invalid cron expression" }
);

export const ScheduleWindowSchema = z.object({
  startTime: z.string(), // HH:MM format
  endTime: z.string(), // HH:MM format
  timezone: z.string().default("UTC"),
  daysOfWeek: z.array(z.number().min(0).max(6)).default([1, 2, 3, 4, 5]), // Mon-Fri
});

export const ExperimentScheduleSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  description: z.string().optional(),
  experimentConfig: z.custom<ChaosExperimentConfig>(),
  schedule: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("cron"),
      expression: CronExpressionSchema,
      timezone: z.string().default("UTC"),
    }),
    z.object({
      type: z.literal("interval"),
      intervalMs: z.number().positive(),
      startDelay: z.number().nonnegative().default(0),
    }),
    z.object({
      type: z.literal("once"),
      executeAt: z.date(),
    }),
    z.object({
      type: z.literal("gameday"),
      gamedayId: z.string(),
      order: z.number().int().nonnegative(),
    }),
  ]),
  constraints: z.object({
    window: ScheduleWindowSchema.optional(),
    maxConcurrent: z.number().int().positive().default(1),
    cooldownMs: z.number().nonnegative().default(0),
    requireApproval: z.boolean().default(false),
    skipOnFailure: z.boolean().default(true),
    dependencies: z.array(z.string().uuid()).default([]),
  }),
  enabled: z.boolean().default(true),
  metadata: z.record(z.string()).default({}),
});

export const GamedayConfigSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  description: z.string().optional(),
  scheduledDate: z.date(),
  duration: z.number().positive(), // minutes
  experiments: z.array(z.string().uuid()),
  participants: z.array(
    z.object({
      name: z.string(),
      role: z.enum(["facilitator", "observer", "engineer", "stakeholder"]),
      email: z.string().email().optional(),
    })
  ),
  runbook: z.string().optional(), // Link to runbook
  communicationChannel: z.string().optional(), // Slack channel, Teams, etc.
  status: z.enum(["scheduled", "in_progress", "completed", "cancelled"]),
});

// ============================================================================
// Types
// ============================================================================

export type ExperimentSchedule = z.infer<typeof ExperimentScheduleSchema>;
export type GamedayConfig = z.infer<typeof GamedayConfigSchema>;
export type ScheduleWindow = z.infer<typeof ScheduleWindowSchema>;

export interface SchedulerEvents {
  scheduleCreated: (schedule: ExperimentSchedule) => void;
  scheduleUpdated: (schedule: ExperimentSchedule) => void;
  scheduleDeleted: (scheduleId: string) => void;
  experimentTriggered: (
    schedule: ExperimentSchedule,
    experiment: ChaosExperiment
  ) => void;
  experimentSkipped: (scheduleId: string, reason: string) => void;
  gamedayStarted: (gameday: GamedayConfig) => void;
  gamedayCompleted: (gameday: GamedayConfig, results: GamedayResults) => void;
  error: (error: Error, context: Record<string, unknown>) => void;
}

export interface SchedulerConfig {
  checkIntervalMs: number;
  maxConcurrentExperiments: number;
  defaultCooldownMs: number;
  autoStartOnInit: boolean;
}

export interface ScheduledExecution {
  scheduleId: string;
  experimentId?: string;
  scheduledAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  error?: string;
}

export interface GamedayResults {
  gamedayId: string;
  startedAt: Date;
  completedAt: Date;
  experimentsRun: number;
  experimentsSucceeded: number;
  experimentsFailed: number;
  findings: string[];
  recommendations: string[];
}

// ============================================================================
// Chaos Scheduler Implementation
// ============================================================================

/**
 * ChaosScheduler - Automated chaos experiment scheduling
 *
 * Features:
 * - Cron-based scheduling
 * - Interval-based scheduling
 * - One-time execution
 * - Gameday orchestration
 * - Execution windows
 * - Dependency management
 * - Cooldown periods
 */
export class ChaosScheduler extends EventEmitter<SchedulerEvents> {
  private config: SchedulerConfig;
  private simulator: ChaosSimulator;
  private schedules: Map<string, ExperimentSchedule> = new Map();
  private gamedays: Map<string, GamedayConfig> = new Map();
  private executions: Map<string, ScheduledExecution[]> = new Map();
  private runningExperiments: Map<string, ChaosExperiment> = new Map();
  private timers: Map<string, NodeJS.Timeout> = new Map();
  private checkInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;

  constructor(
    simulator: ChaosSimulator,
    config: Partial<SchedulerConfig> = {}
  ) {
    super();
    this.simulator = simulator;
    this.config = {
      checkIntervalMs: config.checkIntervalMs ?? 60000, // 1 minute
      maxConcurrentExperiments: config.maxConcurrentExperiments ?? 3,
      defaultCooldownMs: config.defaultCooldownMs ?? 300000, // 5 minutes
      autoStartOnInit: config.autoStartOnInit ?? false,
    };

    if (this.config.autoStartOnInit) {
      this.start();
    }
  }

  // ============================================================================
  // Lifecycle Management
  // ============================================================================

  /**
   * Start the scheduler
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.checkInterval = setInterval(
      () => this.checkSchedules(),
      this.config.checkIntervalMs
    );

    // Immediate check
    this.checkSchedules();

    console.log("ChaosScheduler started");
  }

  /**
   * Stop the scheduler
   */
  stop(): void {
    if (!this.isRunning) return;

    this.isRunning = false;

    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = undefined;
    }

    // Clear all timers
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    this.timers.clear();

    console.log("ChaosScheduler stopped");
  }

  /**
   * Shutdown scheduler and abort all running experiments
   */
  async shutdown(): Promise<void> {
    this.stop();

    // Abort all running experiments
    for (const [scheduleId, experiment] of this.runningExperiments) {
      console.log(`Aborting experiment for schedule ${scheduleId}`);
      await this.simulator.abortExperiment(experiment.id);
    }

    this.runningExperiments.clear();
  }

  // ============================================================================
  // Schedule Management
  // ============================================================================

  /**
   * Create a new experiment schedule
   */
  createSchedule(config: Omit<ExperimentSchedule, "id">): ExperimentSchedule {
    const schedule: ExperimentSchedule = {
      ...ExperimentScheduleSchema.parse({
        ...config,
        id: uuidv4(),
      }),
    };

    this.schedules.set(schedule.id, schedule);
    this.executions.set(schedule.id, []);

    // Set up schedule-specific timers
    this.setupScheduleTimer(schedule);

    this.emit("scheduleCreated", schedule);
    return schedule;
  }

  /**
   * Update an existing schedule
   */
  updateSchedule(
    scheduleId: string,
    updates: Partial<Omit<ExperimentSchedule, "id">>
  ): ExperimentSchedule | undefined {
    const existing = this.schedules.get(scheduleId);
    if (!existing) return undefined;

    const updated = ExperimentScheduleSchema.parse({
      ...existing,
      ...updates,
      id: scheduleId,
    });

    this.schedules.set(scheduleId, updated);

    // Reset timer
    const existingTimer = this.timers.get(scheduleId);
    if (existingTimer) {
      clearTimeout(existingTimer);
      this.timers.delete(scheduleId);
    }
    this.setupScheduleTimer(updated);

    this.emit("scheduleUpdated", updated);
    return updated;
  }

  /**
   * Delete a schedule
   */
  deleteSchedule(scheduleId: string): boolean {
    const schedule = this.schedules.get(scheduleId);
    if (!schedule) return false;

    // Clear timer
    const timer = this.timers.get(scheduleId);
    if (timer) {
      clearTimeout(timer);
      this.timers.delete(scheduleId);
    }

    this.schedules.delete(scheduleId);
    this.executions.delete(scheduleId);

    this.emit("scheduleDeleted", scheduleId);
    return true;
  }

  /**
   * Get a schedule by ID
   */
  getSchedule(scheduleId: string): ExperimentSchedule | undefined {
    return this.schedules.get(scheduleId);
  }

  /**
   * List all schedules
   */
  listSchedules(): ExperimentSchedule[] {
    return Array.from(this.schedules.values());
  }

  /**
   * Enable/disable a schedule
   */
  setScheduleEnabled(scheduleId: string, enabled: boolean): boolean {
    const schedule = this.schedules.get(scheduleId);
    if (!schedule) return false;

    schedule.enabled = enabled;
    this.emit("scheduleUpdated", schedule);
    return true;
  }

  // ============================================================================
  // Gameday Management
  // ============================================================================

  /**
   * Create a gameday
   */
  createGameday(config: Omit<GamedayConfig, "id">): GamedayConfig {
    const gameday = GamedayConfigSchema.parse({
      ...config,
      id: uuidv4(),
    });

    this.gamedays.set(gameday.id, gameday);
    return gameday;
  }

  /**
   * Start a gameday
   */
  async startGameday(gamedayId: string): Promise<GamedayResults | undefined> {
    const gameday = this.gamedays.get(gamedayId);
    if (!gameday) return undefined;

    gameday.status = "in_progress";
    this.emit("gamedayStarted", gameday);

    const results: GamedayResults = {
      gamedayId,
      startedAt: new Date(),
      completedAt: new Date(),
      experimentsRun: 0,
      experimentsSucceeded: 0,
      experimentsFailed: 0,
      findings: [],
      recommendations: [],
    };

    // Run experiments in order
    for (const scheduleId of gameday.experiments) {
      const schedule = this.schedules.get(scheduleId);
      if (!schedule) continue;

      try {
        results.experimentsRun++;
        const experiment = await this.triggerExperiment(schedule);

        if (experiment) {
          // Wait for experiment to complete
          await this.waitForExperiment(experiment.id);

          const finalExperiment = this.runningExperiments.get(schedule.id);
          if (finalExperiment?.results) {
            if (finalExperiment.results.success) {
              results.experimentsSucceeded++;
            } else {
              results.experimentsFailed++;
            }
            results.findings.push(...finalExperiment.results.findings);
          }
        }
      } catch (error) {
        results.experimentsFailed++;
        results.findings.push(
          `Experiment ${schedule.name} failed: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    results.completedAt = new Date();
    gameday.status = "completed";

    // Generate recommendations based on findings
    results.recommendations = this.generateRecommendations(results.findings);

    this.emit("gamedayCompleted", gameday, results);
    return results;
  }

  /**
   * Get a gameday by ID
   */
  getGameday(gamedayId: string): GamedayConfig | undefined {
    return this.gamedays.get(gamedayId);
  }

  /**
   * List all gamedays
   */
  listGamedays(): GamedayConfig[] {
    return Array.from(this.gamedays.values());
  }

  // ============================================================================
  // Execution Logic
  // ============================================================================

  /**
   * Check all schedules and trigger if due
   */
  private checkSchedules(): void {
    const now = new Date();

    for (const schedule of this.schedules.values()) {
      if (!schedule.enabled) continue;

      if (this.shouldTrigger(schedule, now)) {
        this.triggerExperiment(schedule).catch((error) => {
          this.emit("error", error as Error, { scheduleId: schedule.id });
        });
      }
    }
  }

  /**
   * Determine if a schedule should trigger
   */
  private shouldTrigger(schedule: ExperimentSchedule, now: Date): boolean {
    // Check execution window
    if (schedule.constraints.window) {
      if (!this.isWithinWindow(schedule.constraints.window, now)) {
        return false;
      }
    }

    // Check concurrent limit
    if (this.runningExperiments.size >= schedule.constraints.maxConcurrent) {
      return false;
    }

    // Check cooldown
    const executions = this.executions.get(schedule.id) || [];
    const lastExecution = executions[executions.length - 1];
    if (lastExecution?.completedAt) {
      const cooldown =
        schedule.constraints.cooldownMs || this.config.defaultCooldownMs;
      if (now.getTime() - lastExecution.completedAt.getTime() < cooldown) {
        return false;
      }
    }

    // Check dependencies
    for (const depId of schedule.constraints.dependencies) {
      const depExecutions = this.executions.get(depId) || [];
      const lastDepExecution = depExecutions[depExecutions.length - 1];
      if (!lastDepExecution || lastDepExecution.status !== "completed") {
        return false;
      }
    }

    // Check schedule type
    switch (schedule.schedule.type) {
      case "cron":
        return this.matchesCron(schedule.schedule.expression, now);

      case "interval":
        return this.matchesInterval(schedule, now);

      case "once":
        return (
          now >= schedule.schedule.executeAt &&
          !executions.some((e) => e.status === "completed")
        );

      case "gameday":
        // Gamedays are triggered manually
        return false;
    }

    return false;
  }

  /**
   * Trigger an experiment
   */
  private async triggerExperiment(
    schedule: ExperimentSchedule
  ): Promise<ChaosExperiment | undefined> {
    // Check approval requirement
    if (schedule.constraints.requireApproval) {
      // In a real system, this would wait for approval
      console.log(`Schedule ${schedule.id} requires approval`);
      this.emit("experimentSkipped", schedule.id, "Approval required");
      return undefined;
    }

    // Create execution record
    const execution: ScheduledExecution = {
      scheduleId: schedule.id,
      scheduledAt: new Date(),
      status: "pending",
    };

    const executions = this.executions.get(schedule.id) || [];
    executions.push(execution);
    this.executions.set(schedule.id, executions);

    try {
      // Create and start experiment
      const experiment = this.simulator.createExperiment(
        schedule.experimentConfig
      );
      execution.experimentId = experiment.id;
      execution.startedAt = new Date();
      execution.status = "running";

      this.runningExperiments.set(schedule.id, experiment);

      await this.simulator.startExperiment(experiment.id);
      this.emit("experimentTriggered", schedule, experiment);

      return experiment;
    } catch (error) {
      execution.status = "failed";
      execution.error = error instanceof Error ? error.message : String(error);
      this.emit("error", error as Error, { scheduleId: schedule.id });
      return undefined;
    }
  }

  /**
   * Wait for an experiment to complete
   */
  private async waitForExperiment(experimentId: string): Promise<void> {
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        let found = false;
        for (const [, exp] of this.runningExperiments) {
          if (
            exp.id === experimentId &&
            (exp.state === "completed" || exp.state === "aborted")
          ) {
            clearInterval(checkInterval);
            resolve();
            found = true;
            break;
          }
        }
        if (!found) {
          // Experiment might have been removed
          clearInterval(checkInterval);
          resolve();
        }
      }, 1000);
    });
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  /**
   * Check if current time is within schedule window
   */
  private isWithinWindow(window: ScheduleWindow, now: Date): boolean {
    const dayOfWeek = now.getDay();
    if (!window.daysOfWeek.includes(dayOfWeek)) {
      return false;
    }

    const [startHour, startMin] = window.startTime.split(":").map(Number);
    const [endHour, endMin] = window.endTime.split(":").map(Number);

    const currentMinutes = now.getHours() * 60 + now.getMinutes();
    const startMinutes = startHour * 60 + startMin;
    const endMinutes = endHour * 60 + endMin;

    return currentMinutes >= startMinutes && currentMinutes <= endMinutes;
  }

  /**
   * Check if cron expression matches current time
   */
  private matchesCron(expression: string, now: Date): boolean {
    // Simple cron matching (minute hour dayOfMonth month dayOfWeek)
    const parts = expression.trim().split(/\s+/);
    if (parts.length < 5) return false;

    const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;

    return (
      this.matchCronField(minute, now.getMinutes()) &&
      this.matchCronField(hour, now.getHours()) &&
      this.matchCronField(dayOfMonth, now.getDate()) &&
      this.matchCronField(month, now.getMonth() + 1) &&
      this.matchCronField(dayOfWeek, now.getDay())
    );
  }

  /**
   * Match a single cron field
   */
  private matchCronField(field: string, value: number): boolean {
    if (field === "*") return true;

    // Handle ranges (e.g., 1-5)
    if (field.includes("-")) {
      const [start, end] = field.split("-").map(Number);
      return value >= start && value <= end;
    }

    // Handle lists (e.g., 1,3,5)
    if (field.includes(",")) {
      const values = field.split(",").map(Number);
      return values.includes(value);
    }

    // Handle steps (e.g., */5)
    if (field.includes("/")) {
      const [, step] = field.split("/");
      return value % parseInt(step, 10) === 0;
    }

    return parseInt(field, 10) === value;
  }

  /**
   * Check if interval schedule should trigger
   */
  private matchesInterval(schedule: ExperimentSchedule, now: Date): boolean {
    if (schedule.schedule.type !== "interval") return false;

    const executions = this.executions.get(schedule.id) || [];
    if (executions.length === 0) {
      // First execution after start delay
      const scheduleCreatedAt = executions[0]?.scheduledAt || now;
      return (
        now.getTime() - scheduleCreatedAt.getTime() >=
        schedule.schedule.startDelay
      );
    }

    const lastExecution = executions[executions.length - 1];
    const lastTime =
      lastExecution.completedAt ||
      lastExecution.startedAt ||
      lastExecution.scheduledAt;

    return now.getTime() - lastTime.getTime() >= schedule.schedule.intervalMs;
  }

  /**
   * Set up schedule-specific timer
   */
  private setupScheduleTimer(schedule: ExperimentSchedule): void {
    if (schedule.schedule.type === "once") {
      const delay = schedule.schedule.executeAt.getTime() - Date.now();
      if (delay > 0) {
        const timer = setTimeout(() => {
          this.triggerExperiment(schedule).catch((error) => {
            this.emit("error", error as Error, { scheduleId: schedule.id });
          });
        }, delay);
        this.timers.set(schedule.id, timer);
      }
    }
  }

  /**
   * Generate recommendations from findings
   */
  private generateRecommendations(findings: string[]): string[] {
    const recommendations: string[] = [];

    for (const finding of findings) {
      const lower = finding.toLowerCase();

      if (lower.includes("timeout") || lower.includes("latency")) {
        recommendations.push(
          "Consider implementing circuit breakers to handle slow dependencies"
        );
        recommendations.push(
          "Review timeout configurations and add fallback mechanisms"
        );
      }

      if (lower.includes("crash") || lower.includes("restart")) {
        recommendations.push(
          "Ensure graceful shutdown handlers are implemented"
        );
        recommendations.push(
          "Review pod/container restart policies and health checks"
        );
      }

      if (lower.includes("memory") || lower.includes("oom")) {
        recommendations.push(
          "Review memory limits and implement memory pressure handling"
        );
        recommendations.push("Add memory usage monitoring and alerting");
      }

      if (lower.includes("network") || lower.includes("partition")) {
        recommendations.push(
          "Implement service mesh for improved network resilience"
        );
        recommendations.push("Add retry logic with exponential backoff");
      }
    }

    // Deduplicate
    return [...new Set(recommendations)];
  }

  // ============================================================================
  // Statistics & Reporting
  // ============================================================================

  /**
   * Get execution history for a schedule
   */
  getExecutionHistory(scheduleId: string): ScheduledExecution[] {
    return this.executions.get(scheduleId) || [];
  }

  /**
   * Get scheduler statistics
   */
  getStatistics(): SchedulerStatistics {
    let totalExecutions = 0;
    let successfulExecutions = 0;
    let failedExecutions = 0;
    let skippedExecutions = 0;

    for (const executions of this.executions.values()) {
      for (const exec of executions) {
        totalExecutions++;
        if (exec.status === "completed") successfulExecutions++;
        if (exec.status === "failed") failedExecutions++;
        if (exec.status === "skipped") skippedExecutions++;
      }
    }

    return {
      totalSchedules: this.schedules.size,
      enabledSchedules: Array.from(this.schedules.values()).filter(
        (s) => s.enabled
      ).length,
      totalGamedays: this.gamedays.size,
      runningExperiments: this.runningExperiments.size,
      totalExecutions,
      successfulExecutions,
      failedExecutions,
      skippedExecutions,
      successRate:
        totalExecutions > 0
          ? (successfulExecutions / totalExecutions) * 100
          : 0,
    };
  }
}

export interface SchedulerStatistics {
  totalSchedules: number;
  enabledSchedules: number;
  totalGamedays: number;
  runningExperiments: number;
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  skippedExecutions: number;
  successRate: number;
}
