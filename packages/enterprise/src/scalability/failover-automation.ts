/**
 * @fileoverview Failover Automation Manager
 * Automatic failover detection and execution with health monitoring
 * @module @neurectomy/enterprise/scalability
 */

import { EventEmitter } from "events";
import type {
  FailoverConfig,
  FailoverEvent,
  FailoverEventType,
  FailoverMetrics,
  FailoverPolicy,
  FailoverStrategy,
  HealthCheckConfig,
  HealthCheckType,
  NodeConfig,
  NodeStatus,
} from "./types.js";

// =============================================================================
// HEALTH CHECKER
// =============================================================================

/**
 * Health check result
 */
interface HealthCheckResult {
  nodeId: string;
  healthy: boolean;
  latency: number;
  timestamp: Date;
  details?: Record<string, unknown>;
  error?: string;
}

/**
 * Performs health checks on nodes
 */
class HealthChecker {
  private config: HealthCheckConfig;
  private consecutiveResults: Map<string, boolean[]> = new Map();

  constructor(config: HealthCheckConfig) {
    this.config = config;
  }

  async check(node: NodeConfig): Promise<HealthCheckResult> {
    const start = Date.now();
    let healthy = false;
    let error: string | undefined;

    try {
      switch (this.config.type) {
        case "http":
          healthy = await this.httpCheck(node);
          break;
        case "tcp":
          healthy = await this.tcpCheck(node);
          break;
        case "database":
          healthy = await this.databaseCheck(node);
          break;
        case "custom":
          if (this.config.customCheck) {
            healthy = await this.config.customCheck();
          }
          break;
        case "composite":
          healthy = await this.compositeCheck(node);
          break;
      }
    } catch (e) {
      healthy = false;
      error = e instanceof Error ? e.message : String(e);
    }

    const latency = Date.now() - start;

    // Track consecutive results
    const results = this.consecutiveResults.get(node.id) || [];
    results.push(healthy);
    if (
      results.length >
      Math.max(this.config.healthyThreshold, this.config.unhealthyThreshold)
    ) {
      results.shift();
    }
    this.consecutiveResults.set(node.id, results);

    return {
      nodeId: node.id,
      healthy,
      latency,
      timestamp: new Date(),
      error,
    };
  }

  isDefinitelyHealthy(nodeId: string): boolean {
    const results = this.consecutiveResults.get(nodeId) || [];
    if (results.length < this.config.healthyThreshold) return false;

    const recentResults = results.slice(-this.config.healthyThreshold);
    return recentResults.every((r) => r === true);
  }

  isDefinitelyUnhealthy(nodeId: string): boolean {
    const results = this.consecutiveResults.get(nodeId) || [];
    if (results.length < this.config.unhealthyThreshold) return false;

    const recentResults = results.slice(-this.config.unhealthyThreshold);
    return recentResults.every((r) => r === false);
  }

  resetResults(nodeId: string): void {
    this.consecutiveResults.delete(nodeId);
  }

  private async httpCheck(node: NodeConfig): Promise<boolean> {
    // Simulate HTTP health check
    const endpoint = this.config.endpoint || `/health`;
    const url = `http://${node.host}:${node.port}${endpoint}`;

    try {
      // In real implementation, use fetch with timeout
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("Timeout")),
          this.config.timeout
        );
        setTimeout(() => {
          clearTimeout(timeout);
          resolve(Math.random() > 0.1); // 90% success rate simulation
        }, Math.random() * 100);
      });

      return true;
    } catch {
      return false;
    }
  }

  private async tcpCheck(node: NodeConfig): Promise<boolean> {
    // Simulate TCP connection check
    try {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("Timeout")),
          this.config.timeout
        );
        setTimeout(() => {
          clearTimeout(timeout);
          resolve(Math.random() > 0.05); // 95% success rate simulation
        }, Math.random() * 50);
      });

      return true;
    } catch {
      return false;
    }
  }

  private async databaseCheck(node: NodeConfig): Promise<boolean> {
    // Simulate database connectivity check
    try {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("Timeout")),
          this.config.timeout
        );
        setTimeout(() => {
          clearTimeout(timeout);
          resolve(Math.random() > 0.08); // 92% success rate simulation
        }, Math.random() * 100);
      });

      return true;
    } catch {
      return false;
    }
  }

  private async compositeCheck(node: NodeConfig): Promise<boolean> {
    // Run multiple checks and require all to pass
    const results = await Promise.all([
      this.httpCheck(node),
      this.tcpCheck(node),
    ]);

    return results.every((r) => r === true);
  }
}

// =============================================================================
// FAILOVER EXECUTOR
// =============================================================================

/**
 * Executes failover operations
 */
class FailoverExecutor {
  private events: FailoverEvent[] = [];

  async execute(
    sourceNode: NodeConfig,
    targetNode: NodeConfig,
    policy: FailoverPolicy,
    onProgress: (event: FailoverEvent) => void
  ): Promise<FailoverEvent> {
    const eventId = `failover-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // Create detection event
    const detectionEvent = this.createEvent(
      eventId,
      "detection",
      sourceNode.id,
      targetNode.id,
      policy
    );
    this.events.push(detectionEvent);
    onProgress(detectionEvent);

    // Create initiated event
    const initiatedEvent = this.createEvent(
      eventId,
      "initiated",
      sourceNode.id,
      targetNode.id,
      policy
    );
    this.events.push(initiatedEvent);
    onProgress(initiatedEvent);

    try {
      // Drain connections if configured
      if (policy.actions.drainTimeout > 0) {
        await this.drainConnections(sourceNode, policy.actions.drainTimeout);
      }

      // Create in-progress event
      const inProgressEvent = this.createEvent(
        eventId,
        "in-progress",
        sourceNode.id,
        targetNode.id,
        policy
      );
      this.events.push(inProgressEvent);
      onProgress(inProgressEvent);

      // Execute failover based on strategy
      await this.executeStrategy(policy.strategy, sourceNode, targetNode);

      // Create completed event
      const completedEvent = this.createEvent(
        eventId,
        "completed",
        sourceNode.id,
        targetNode.id,
        policy,
        true
      );
      completedEvent.endTime = new Date();
      completedEvent.duration =
        completedEvent.endTime.getTime() - completedEvent.startTime.getTime();

      this.events.push(completedEvent);
      onProgress(completedEvent);

      return completedEvent;
    } catch (error) {
      // Create failed event
      const failedEvent = this.createEvent(
        eventId,
        "failed",
        sourceNode.id,
        targetNode.id,
        policy,
        false
      );
      failedEvent.endTime = new Date();
      failedEvent.duration =
        failedEvent.endTime.getTime() - failedEvent.startTime.getTime();
      failedEvent.error =
        error instanceof Error ? error.message : String(error);

      this.events.push(failedEvent);
      onProgress(failedEvent);

      // Attempt rollback if configured
      if (policy.actions.autoRollback) {
        await this.rollback(sourceNode, targetNode, policy, onProgress);
      }

      return failedEvent;
    }
  }

  async rollback(
    sourceNode: NodeConfig,
    targetNode: NodeConfig,
    policy: FailoverPolicy,
    onProgress: (event: FailoverEvent) => void
  ): Promise<FailoverEvent> {
    const eventId = `rollback-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const rollbackEvent = this.createEvent(
      eventId,
      "rollback",
      targetNode.id, // Swap source and target for rollback
      sourceNode.id,
      policy
    );

    try {
      await this.executeStrategy(policy.strategy, targetNode, sourceNode);
      rollbackEvent.success = true;
    } catch (error) {
      rollbackEvent.success = false;
      rollbackEvent.error =
        error instanceof Error ? error.message : String(error);
    }

    rollbackEvent.endTime = new Date();
    rollbackEvent.duration =
      rollbackEvent.endTime.getTime() - rollbackEvent.startTime.getTime();

    this.events.push(rollbackEvent);
    onProgress(rollbackEvent);

    return rollbackEvent;
  }

  getEvents(): FailoverEvent[] {
    return [...this.events];
  }

  getRecentEvents(count: number): FailoverEvent[] {
    return this.events.slice(-count);
  }

  private createEvent(
    id: string,
    type: FailoverEventType,
    sourceNode: string,
    targetNode: string,
    policy: FailoverPolicy,
    success = true
  ): FailoverEvent {
    return {
      id,
      type,
      sourceNode,
      targetNode,
      policy,
      startTime: new Date(),
      success,
      metrics: {
        responseTime: 0,
        errorRate: 0,
        consecutiveFailures: 0,
        uptime: 0,
        requestsInFlight: 0,
      },
    };
  }

  private async drainConnections(
    node: NodeConfig,
    timeout: number
  ): Promise<void> {
    // Simulate connection draining
    await new Promise((resolve) =>
      setTimeout(resolve, Math.min(timeout, 1000))
    );
  }

  private async executeStrategy(
    strategy: FailoverStrategy,
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    switch (strategy) {
      case "active-passive":
        await this.executeActivePassive(source, target);
        break;
      case "active-active":
        await this.executeActiveActive(source, target);
        break;
      case "hot-standby":
        await this.executeHotStandby(source, target);
        break;
      case "warm-standby":
        await this.executeWarmStandby(source, target);
        break;
      case "cold-standby":
        await this.executeColdStandby(source, target);
        break;
      case "multi-region":
        await this.executeMultiRegion(source, target);
        break;
    }
  }

  private async executeActivePassive(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // Deactivate source, activate target
    await new Promise((resolve) => setTimeout(resolve, 200));
  }

  private async executeActiveActive(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // Route traffic away from source
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  private async executeHotStandby(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // Promote standby to primary (fast, already synchronized)
    await new Promise((resolve) => setTimeout(resolve, 150));
  }

  private async executeWarmStandby(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // Start target, sync recent changes, promote
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  private async executeColdStandby(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // Start target from scratch, restore backup, promote
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  private async executeMultiRegion(
    source: NodeConfig,
    target: NodeConfig
  ): Promise<void> {
    // DNS failover to different region
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}

// =============================================================================
// METRICS COLLECTOR
// =============================================================================

interface NodeMetricsHistory {
  nodeId: string;
  samples: FailoverMetrics[];
  maxSamples: number;
}

/**
 * Collects and analyzes node metrics
 */
class MetricsCollector {
  private history: Map<string, NodeMetricsHistory> = new Map();
  private maxSamples: number;

  constructor(maxSamples = 100) {
    this.maxSamples = maxSamples;
  }

  record(nodeId: string, metrics: FailoverMetrics): void {
    let history = this.history.get(nodeId);
    if (!history) {
      history = {
        nodeId,
        samples: [],
        maxSamples: this.maxSamples,
      };
      this.history.set(nodeId, history);
    }

    history.samples.push(metrics);
    if (history.samples.length > this.maxSamples) {
      history.samples.shift();
    }
  }

  getLatest(nodeId: string): FailoverMetrics | null {
    const history = this.history.get(nodeId);
    if (!history || history.samples.length === 0) return null;
    return history.samples[history.samples.length - 1];
  }

  getAverages(nodeId: string, windowSize?: number): FailoverMetrics | null {
    const history = this.history.get(nodeId);
    if (!history || history.samples.length === 0) return null;

    const samples = windowSize
      ? history.samples.slice(-windowSize)
      : history.samples;

    return {
      responseTime:
        samples.reduce((sum, s) => sum + s.responseTime, 0) / samples.length,
      errorRate:
        samples.reduce((sum, s) => sum + s.errorRate, 0) / samples.length,
      consecutiveFailures: Math.max(
        ...samples.map((s) => s.consecutiveFailures)
      ),
      uptime: samples[samples.length - 1].uptime,
      lastSuccessfulRequest: samples
        .filter((s) => s.lastSuccessfulRequest)
        .sort(
          (a, b) =>
            (b.lastSuccessfulRequest?.getTime() || 0) -
            (a.lastSuccessfulRequest?.getTime() || 0)
        )[0]?.lastSuccessfulRequest,
      requestsInFlight: samples[samples.length - 1].requestsInFlight,
    };
  }

  checkThresholds(
    nodeId: string,
    policy: FailoverPolicy
  ): { shouldFailover: boolean; reason?: string } {
    const latest = this.getLatest(nodeId);
    if (!latest) {
      return { shouldFailover: false };
    }

    const { triggerConditions } = policy;

    if (latest.consecutiveFailures >= triggerConditions.consecutiveFailures) {
      return {
        shouldFailover: true,
        reason: `Consecutive failures: ${latest.consecutiveFailures} >= ${triggerConditions.consecutiveFailures}`,
      };
    }

    if (latest.responseTime >= triggerConditions.responseTimeThreshold) {
      return {
        shouldFailover: true,
        reason: `Response time: ${latest.responseTime}ms >= ${triggerConditions.responseTimeThreshold}ms`,
      };
    }

    if (latest.errorRate >= triggerConditions.errorRateThreshold) {
      return {
        shouldFailover: true,
        reason: `Error rate: ${(latest.errorRate * 100).toFixed(2)}% >= ${(triggerConditions.errorRateThreshold * 100).toFixed(2)}%`,
      };
    }

    if (triggerConditions.customCondition) {
      if (triggerConditions.customCondition(latest)) {
        return {
          shouldFailover: true,
          reason: "Custom condition triggered",
        };
      }
    }

    return { shouldFailover: false };
  }

  clear(nodeId?: string): void {
    if (nodeId) {
      this.history.delete(nodeId);
    } else {
      this.history.clear();
    }
  }
}

// =============================================================================
// FAILOVER AUTOMATION MANAGER
// =============================================================================

/**
 * Failover Automation Manager
 * Manages automatic failover with health monitoring and policy execution
 */
export class FailoverAutomationManager extends EventEmitter {
  private config: FailoverConfig;
  private nodes: Map<string, NodeConfig> = new Map();
  private healthChecker: HealthChecker;
  private executor: FailoverExecutor;
  private metricsCollector: MetricsCollector;
  private healthCheckInterval?: ReturnType<typeof setInterval>;
  private lastFailoverTime: Map<string, Date> = new Map();
  private failoversInHour: number = 0;
  private failoverHourReset?: ReturnType<typeof setInterval>;
  private isStarted = false;

  constructor(config: FailoverConfig) {
    super();
    this.config = config;
    this.executor = new FailoverExecutor();
    this.metricsCollector = new MetricsCollector();

    // Initialize health checker with first config (or default)
    this.healthChecker = new HealthChecker(
      config.healthChecks[0] || {
        id: "default",
        type: "http" as HealthCheckType,
        interval: 30000,
        timeout: 5000,
        healthyThreshold: 3,
        unhealthyThreshold: 3,
      }
    );

    // Initialize nodes
    for (const node of config.nodes) {
      this.nodes.set(node.id, node);
    }
  }

  /**
   * Start the failover manager
   */
  async start(): Promise<void> {
    if (this.isStarted) return;

    this.emit("starting");

    // Start health check loop
    this.startHealthChecks();

    // Start hourly failover counter reset
    this.failoverHourReset = setInterval(() => {
      this.failoversInHour = 0;
    }, 3600000);

    this.isStarted = true;
    this.emit("started");
  }

  /**
   * Stop the failover manager
   */
  async stop(): Promise<void> {
    if (!this.isStarted) return;

    this.emit("stopping");

    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    if (this.failoverHourReset) {
      clearInterval(this.failoverHourReset);
    }

    this.isStarted = false;
    this.emit("stopped");
  }

  /**
   * Trigger manual failover
   */
  async triggerFailover(
    sourceNodeId: string,
    targetNodeId?: string,
    policyId?: string
  ): Promise<FailoverEvent> {
    const sourceNode = this.nodes.get(sourceNodeId);
    if (!sourceNode) {
      throw new Error(`Source node ${sourceNodeId} not found`);
    }

    // Find target node
    let targetNode: NodeConfig | undefined;
    if (targetNodeId) {
      targetNode = this.nodes.get(targetNodeId);
    } else {
      targetNode = this.findBestTarget(sourceNode);
    }

    if (!targetNode) {
      throw new Error("No suitable target node found for failover");
    }

    // Find policy
    const policy = policyId
      ? this.config.policies.find((p) => p.id === policyId)
      : this.config.policies.find((p) => p.enabled);

    if (!policy) {
      throw new Error("No suitable failover policy found");
    }

    // Check cooldown
    const lastFailover = this.lastFailoverTime.get(sourceNodeId);
    if (lastFailover) {
      const elapsed = Date.now() - lastFailover.getTime();
      if (elapsed < policy.actions.cooldownPeriod) {
        throw new Error(
          `Failover on cooldown. Wait ${Math.ceil((policy.actions.cooldownPeriod - elapsed) / 1000)}s`
        );
      }
    }

    // Check hourly limit
    if (this.failoversInHour >= this.config.global.maxFailoversPerHour) {
      throw new Error(
        `Max failovers per hour (${this.config.global.maxFailoversPerHour}) exceeded`
      );
    }

    // Execute failover
    this.emit("failover:triggered", {
      sourceNodeId,
      targetNodeId: targetNode.id,
    });

    const event = await this.executor.execute(
      sourceNode,
      targetNode,
      policy,
      (evt) => {
        this.emit(`failover:${evt.type}`, evt);
      }
    );

    // Update tracking
    this.lastFailoverTime.set(sourceNodeId, new Date());
    this.failoversInHour++;

    // Update node states
    if (event.success) {
      sourceNode.status = "unhealthy";
      targetNode.role = "primary";
      sourceNode.role = "secondary";
    }

    return event;
  }

  /**
   * Add a node
   */
  addNode(node: NodeConfig): void {
    this.nodes.set(node.id, node);
    this.emit("node:added", node);
  }

  /**
   * Remove a node
   */
  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.healthChecker.resetResults(nodeId);
    this.metricsCollector.clear(nodeId);
    this.emit("node:removed", { nodeId });
  }

  /**
   * Update node configuration
   */
  updateNode(nodeId: string, updates: Partial<NodeConfig>): void {
    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }
    Object.assign(node, updates);
    this.emit("node:updated", node);
  }

  /**
   * Add a failover policy
   */
  addPolicy(policy: FailoverPolicy): void {
    this.config.policies.push(policy);
    this.emit("policy:added", policy);
  }

  /**
   * Remove a failover policy
   */
  removePolicy(policyId: string): void {
    this.config.policies = this.config.policies.filter(
      (p) => p.id !== policyId
    );
    this.emit("policy:removed", { policyId });
  }

  /**
   * Get all nodes
   */
  getNodes(): NodeConfig[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get node by ID
   */
  getNode(nodeId: string): NodeConfig | undefined {
    return this.nodes.get(nodeId);
  }

  /**
   * Get primary node
   */
  getPrimaryNode(): NodeConfig | undefined {
    return Array.from(this.nodes.values()).find((n) => n.role === "primary");
  }

  /**
   * Get secondary nodes
   */
  getSecondaryNodes(): NodeConfig[] {
    return Array.from(this.nodes.values()).filter(
      (n) => n.role === "secondary"
    );
  }

  /**
   * Get healthy nodes
   */
  getHealthyNodes(): NodeConfig[] {
    return Array.from(this.nodes.values()).filter(
      (n) => n.status === "healthy"
    );
  }

  /**
   * Get node metrics
   */
  getNodeMetrics(nodeId: string): FailoverMetrics | null {
    return this.metricsCollector.getLatest(nodeId);
  }

  /**
   * Get failover events
   */
  getFailoverEvents(): FailoverEvent[] {
    return this.executor.getEvents();
  }

  /**
   * Get recent failover events
   */
  getRecentFailoverEvents(count: number): FailoverEvent[] {
    return this.executor.getRecentEvents(count);
  }

  /**
   * Get failover statistics
   */
  getStats(): {
    totalNodes: number;
    healthyNodes: number;
    primaryNode: string | null;
    failoversInLastHour: number;
    lastFailover: Date | null;
    averageFailoverDuration: number;
  } {
    const events = this.executor.getEvents();
    const completedEvents = events.filter(
      (e) => e.type === "completed" && e.duration !== undefined
    );
    const avgDuration =
      completedEvents.length > 0
        ? completedEvents.reduce((sum, e) => sum + (e.duration || 0), 0) /
          completedEvents.length
        : 0;

    const lastFailover =
      events
        .filter((e) => e.type === "completed")
        .sort(
          (a, b) => (b.endTime?.getTime() || 0) - (a.endTime?.getTime() || 0)
        )[0]?.endTime || null;

    return {
      totalNodes: this.nodes.size,
      healthyNodes: this.getHealthyNodes().length,
      primaryNode: this.getPrimaryNode()?.id || null,
      failoversInLastHour: this.failoversInHour,
      lastFailover,
      averageFailoverDuration: avgDuration,
    };
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  private startHealthChecks(): void {
    const interval = this.config.healthChecks[0]?.interval || 30000;

    this.healthCheckInterval = setInterval(async () => {
      const nodes = Array.from(this.nodes.values());
      for (let i = 0; i < nodes.length; i++) {
        await this.checkNodeHealth(nodes[i]);
      }
    }, interval);

    // Run initial check
    this.nodes.forEach((node) => this.checkNodeHealth(node));
  }

  private async checkNodeHealth(node: NodeConfig): Promise<void> {
    const result = await this.healthChecker.check(node);

    // Update node status based on definitive results
    if (this.healthChecker.isDefinitelyHealthy(node.id)) {
      if (node.status !== "healthy") {
        node.status = "healthy";
        node.lastHealthCheck = new Date();
        this.emit("node:healthy", { nodeId: node.id });
      }
    } else if (this.healthChecker.isDefinitelyUnhealthy(node.id)) {
      if (node.status !== "unhealthy") {
        node.status = "unhealthy";
        node.lastHealthCheck = new Date();
        this.emit("node:unhealthy", { nodeId: node.id });

        // Check if automatic failover is needed
        if (node.role === "primary") {
          await this.handlePrimaryUnhealthy(node);
        }
      }
    }

    // Record metrics
    this.metricsCollector.record(node.id, {
      responseTime: result.latency,
      errorRate: result.healthy ? 0 : 1,
      consecutiveFailures: this.healthChecker.isDefinitelyUnhealthy(node.id)
        ? this.config.healthChecks[0]?.unhealthyThreshold || 3
        : 0,
      uptime: node.status === "healthy" ? 1 : 0,
      lastSuccessfulRequest: result.healthy ? new Date() : undefined,
      requestsInFlight: 0,
    });

    // Check policy thresholds
    for (const policy of this.config.policies) {
      if (!policy.enabled) continue;

      const check = this.metricsCollector.checkThresholds(node.id, policy);
      if (check.shouldFailover && node.role === "primary") {
        this.emit("policy:triggered", {
          nodeId: node.id,
          policy: policy.id,
          reason: check.reason,
        });

        if (this.config.enabled) {
          try {
            await this.triggerFailover(node.id, undefined, policy.id);
          } catch (error) {
            this.emit("failover:error", { nodeId: node.id, error });
          }
        }
      }
    }
  }

  private async handlePrimaryUnhealthy(node: NodeConfig): Promise<void> {
    if (!this.config.enabled) {
      this.emit("warning", {
        code: "PRIMARY_UNHEALTHY",
        message: `Primary node ${node.id} is unhealthy but automatic failover is disabled`,
      });
      return;
    }

    const policy = this.config.policies.find((p) => p.enabled);
    if (!policy) {
      this.emit("warning", {
        code: "NO_POLICY",
        message: "No enabled failover policy found",
      });
      return;
    }

    // Notify before failover if configured
    if (policy.actions.notifyBefore) {
      this.emit("failover:pending", { nodeId: node.id });
    }

    try {
      await this.triggerFailover(node.id, undefined, policy.id);

      // Notify after failover if configured
      if (policy.actions.notifyAfter) {
        this.emit("failover:notification", {
          nodeId: node.id,
          message: "Failover completed successfully",
        });
      }
    } catch (error) {
      this.emit("error", error);
    }
  }

  private findBestTarget(sourceNode: NodeConfig): NodeConfig | undefined {
    const candidates = Array.from(this.nodes.values())
      .filter(
        (n) =>
          n.id !== sourceNode.id &&
          n.status === "healthy" &&
          n.role !== "arbiter"
      )
      .sort((a, b) => {
        // Prefer same region
        if (a.region === sourceNode.region && b.region !== sourceNode.region) {
          return -1;
        }
        if (b.region === sourceNode.region && a.region !== sourceNode.region) {
          return 1;
        }
        // Then by priority
        return a.priority - b.priority;
      });

    return candidates[0];
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a failover automation manager
 */
export function createFailoverAutomationManager(
  config: FailoverConfig
): FailoverAutomationManager {
  return new FailoverAutomationManager(config);
}

/**
 * Create default failover configuration
 */
export function createDefaultFailoverConfig(
  nodes: Array<{
    id: string;
    name: string;
    host: string;
    port: number;
    region: string;
    role: "primary" | "secondary";
  }>
): FailoverConfig {
  return {
    enabled: true,
    strategy: "active-passive" as FailoverStrategy,
    nodes: nodes.map((n, i) => ({
      id: n.id,
      name: n.name,
      host: n.host,
      port: n.port,
      region: n.region,
      role: n.role,
      priority: i + 1,
      status: "healthy" as NodeStatus,
      metadata: {},
    })),
    healthChecks: [
      {
        id: "default-http",
        type: "http" as HealthCheckType,
        endpoint: "/health",
        interval: 30000,
        timeout: 5000,
        healthyThreshold: 3,
        unhealthyThreshold: 3,
        successCodes: [200, 204],
      },
    ],
    policies: [
      {
        id: "default-policy",
        name: "Default Failover Policy",
        strategy: "active-passive" as FailoverStrategy,
        triggerConditions: {
          consecutiveFailures: 3,
          responseTimeThreshold: 5000,
          errorRateThreshold: 0.5,
        },
        actions: {
          notifyBefore: true,
          notifyAfter: true,
          drainTimeout: 30000,
          cooldownPeriod: 300000,
          autoRollback: true,
          rollbackTimeout: 60000,
        },
        enabled: true,
      },
    ],
    global: {
      maxFailoversPerHour: 3,
      notificationChannels: ["email", "slack"],
      metricsRetention: 30,
    },
  };
}
