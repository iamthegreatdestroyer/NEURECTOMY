/**
 * Rollback Manager
 * Automated rollback for failed deployments
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";
import type { KubernetesClient } from "../kubernetes/client";

// =============================================================================
// Types
// =============================================================================

export const RollbackTargetSchema = z.object({
  deploymentId: z.string(),
  name: z.string(),
  namespace: z.string(),
  revision: z.number().optional(),
  previousImage: z.string().optional(),
  previousReplicas: z.number().optional(),
  timestamp: z.date(),
});

export type RollbackTarget = z.infer<typeof RollbackTargetSchema>;

export const RollbackHistoryEntrySchema = z.object({
  id: z.string(),
  deploymentId: z.string(),
  name: z.string(),
  namespace: z.string(),
  triggeredBy: z.enum(["manual", "automatic", "health_check", "analysis"]),
  triggeredAt: z.date(),
  completedAt: z.date().optional(),
  status: z.enum(["in_progress", "success", "failed"]),
  fromRevision: z.number().optional(),
  toRevision: z.number().optional(),
  fromImage: z.string().optional(),
  toImage: z.string().optional(),
  reason: z.string(),
  error: z.string().optional(),
  duration: z.number().optional(),
});

export type RollbackHistoryEntry = z.infer<typeof RollbackHistoryEntrySchema>;

export const RollbackPolicySchema = z.object({
  id: z.string(),
  name: z.string(),
  enabled: z.boolean().default(true),
  autoRollback: z.boolean().default(true),
  maxRollbackAttempts: z.number().default(3),
  cooldownPeriod: z.string().default("5m"),
  preserveHistory: z.number().default(10),
  conditions: z.object({
    onHealthCheckFailure: z.boolean().default(true),
    onAnalysisFailure: z.boolean().default(true),
    onDeploymentTimeout: z.boolean().default(true),
    onCrashLoopBackOff: z.boolean().default(true),
  }),
  excludeNamespaces: z.array(z.string()).optional(),
  includeNamespaces: z.array(z.string()).optional(),
});

export type RollbackPolicy = z.infer<typeof RollbackPolicySchema>;

export interface RollbackEvents {
  "rollback:started": (entry: RollbackHistoryEntry) => void;
  "rollback:completed": (entry: RollbackHistoryEntry) => void;
  "rollback:failed": (entry: RollbackHistoryEntry) => void;
  "rollback:skipped": (reason: string, deploymentId: string) => void;
}

export interface RollbackManagerConfig {
  k8sClient: KubernetesClient;
  policy?: RollbackPolicy;
}

// =============================================================================
// Rollback State Storage
// =============================================================================

interface RollbackState {
  targets: Map<string, RollbackTarget>;
  history: RollbackHistoryEntry[];
  cooldowns: Map<string, Date>;
  attempts: Map<string, number>;
}

// =============================================================================
// Rollback Manager Implementation
// =============================================================================

export class RollbackManager extends EventEmitter<RollbackEvents> {
  private k8sClient: KubernetesClient;
  private policy: RollbackPolicy;
  private state: RollbackState;

  constructor(config: RollbackManagerConfig) {
    super();
    this.k8sClient = config.k8sClient;
    this.policy = config.policy || this.getDefaultPolicy();
    this.state = {
      targets: new Map(),
      history: [],
      cooldowns: new Map(),
      attempts: new Map(),
    };
  }

  /**
   * Record pre-deployment state for potential rollback
   */
  async recordPreDeploymentState(
    name: string,
    namespace: string
  ): Promise<RollbackTarget | null> {
    try {
      const deployment = await this.k8sClient.getDeployment(name, namespace);

      if (!deployment) {
        return null;
      }

      const target: RollbackTarget = {
        deploymentId: `${namespace}/${name}`,
        name,
        namespace,
        revision: parseInt(
          deployment.metadata?.annotations?.[
            "deployment.kubernetes.io/revision"
          ] || "0",
          10
        ),
        previousImage: deployment.spec?.template?.spec?.containers?.[0]?.image,
        previousReplicas: deployment.spec?.replicas || 1,
        timestamp: new Date(),
      };

      this.state.targets.set(target.deploymentId, target);
      return target;
    } catch (error) {
      console.error(
        `Failed to record pre-deployment state for ${namespace}/${name}:`,
        error
      );
      return null;
    }
  }

  /**
   * Clear recorded state (on successful deployment)
   */
  clearState(deploymentId: string): void {
    this.state.targets.delete(deploymentId);
    this.state.attempts.delete(deploymentId);
    this.state.cooldowns.delete(deploymentId);
  }

  /**
   * Execute rollback to previous state
   */
  async rollback(
    name: string,
    namespace: string,
    options: {
      triggeredBy: RollbackHistoryEntry["triggeredBy"];
      reason: string;
      toRevision?: number;
      toImage?: string;
    }
  ): Promise<RollbackHistoryEntry> {
    const deploymentId = `${namespace}/${name}`;

    // Check if rollback is enabled
    if (!this.policy.enabled || !this.policy.autoRollback) {
      const skipReason = "Rollback is disabled by policy";
      this.emit("rollback:skipped", skipReason, deploymentId);
      throw new Error(skipReason);
    }

    // Check namespace filter
    if (!this.isNamespaceAllowed(namespace)) {
      const skipReason = `Namespace ${namespace} is excluded from rollback`;
      this.emit("rollback:skipped", skipReason, deploymentId);
      throw new Error(skipReason);
    }

    // Check cooldown
    if (this.isInCooldown(deploymentId)) {
      const skipReason = "Deployment is in rollback cooldown period";
      this.emit("rollback:skipped", skipReason, deploymentId);
      throw new Error(skipReason);
    }

    // Check max attempts
    const attempts = this.state.attempts.get(deploymentId) || 0;
    if (attempts >= this.policy.maxRollbackAttempts) {
      const skipReason = `Max rollback attempts (${this.policy.maxRollbackAttempts}) exceeded`;
      this.emit("rollback:skipped", skipReason, deploymentId);
      throw new Error(skipReason);
    }

    // Get target state
    const target = this.state.targets.get(deploymentId);

    // Get current deployment state
    const currentDeployment = await this.k8sClient.getDeployment(
      name,
      namespace
    );
    const currentRevision = parseInt(
      currentDeployment?.metadata?.annotations?.[
        "deployment.kubernetes.io/revision"
      ] || "0",
      10
    );
    const currentImage =
      currentDeployment?.spec?.template?.spec?.containers?.[0]?.image;

    // Create history entry
    const entry: RollbackHistoryEntry = {
      id: this.generateId(),
      deploymentId,
      name,
      namespace,
      triggeredBy: options.triggeredBy,
      triggeredAt: new Date(),
      status: "in_progress",
      fromRevision: currentRevision,
      toRevision: options.toRevision || target?.revision,
      fromImage: currentImage,
      toImage: options.toImage || target?.previousImage,
      reason: options.reason,
    };

    this.emit("rollback:started", entry);

    try {
      // Execute rollback
      if (options.toRevision !== undefined) {
        // Rollback to specific revision
        await this.rollbackToRevision(name, namespace, options.toRevision);
      } else if (options.toImage) {
        // Rollback to specific image
        await this.rollbackToImage(name, namespace, options.toImage);
      } else if (target) {
        // Rollback using recorded state
        await this.rollbackToTarget(name, namespace, target);
      } else {
        // Default: use Kubernetes rollback
        await this.rollbackToRevision(name, namespace, currentRevision - 1);
      }

      // Wait for rollback to complete
      await this.waitForRollback(name, namespace);

      // Update entry
      entry.status = "success";
      entry.completedAt = new Date();
      entry.duration =
        entry.completedAt.getTime() - entry.triggeredAt.getTime();

      // Update state
      this.state.attempts.set(deploymentId, attempts + 1);
      this.setCooldown(deploymentId);
      this.addToHistory(entry);

      this.emit("rollback:completed", entry);
      return entry;
    } catch (error) {
      // Update entry
      entry.status = "failed";
      entry.completedAt = new Date();
      entry.duration =
        entry.completedAt.getTime() - entry.triggeredAt.getTime();
      entry.error = error instanceof Error ? error.message : String(error);

      // Update state
      this.state.attempts.set(deploymentId, attempts + 1);
      this.addToHistory(entry);

      this.emit("rollback:failed", entry);
      throw error;
    }
  }

  /**
   * Rollback to specific revision
   */
  async rollbackToRevision(
    name: string,
    namespace: string,
    revision: number
  ): Promise<void> {
    // Get the deployment
    const deployment = await this.k8sClient.getDeployment(name, namespace);
    if (!deployment) {
      throw new Error(`Deployment ${namespace}/${name} not found`);
    }

    // Use rollback annotation (kubectl rollout undo approach)
    await this.k8sClient.patchDeployment(name, namespace, {
      metadata: {
        annotations: {
          "kubectl.kubernetes.io/rollback-to-revision": String(revision),
        },
      },
    });
  }

  /**
   * Rollback to specific image
   */
  async rollbackToImage(
    name: string,
    namespace: string,
    image: string
  ): Promise<void> {
    const deployment = await this.k8sClient.getDeployment(name, namespace);
    if (!deployment) {
      throw new Error(`Deployment ${namespace}/${name} not found`);
    }

    // Update image
    await this.k8sClient.patchDeployment(name, namespace, {
      spec: {
        template: {
          spec: {
            containers: [
              {
                name,
                image,
              },
            ],
          },
        },
      },
    });
  }

  /**
   * Rollback using recorded target state
   */
  private async rollbackToTarget(
    name: string,
    namespace: string,
    target: RollbackTarget
  ): Promise<void> {
    const updates: any = {};

    if (target.previousImage) {
      updates.spec = {
        ...updates.spec,
        template: {
          spec: {
            containers: [
              {
                name,
                image: target.previousImage,
              },
            ],
          },
        },
      };
    }

    if (target.previousReplicas !== undefined) {
      updates.spec = {
        ...updates.spec,
        replicas: target.previousReplicas,
      };
    }

    if (Object.keys(updates).length > 0) {
      await this.k8sClient.patchDeployment(name, namespace, updates);
    } else if (target.revision) {
      await this.rollbackToRevision(name, namespace, target.revision);
    }
  }

  /**
   * Wait for rollback to complete
   */
  private async waitForRollback(
    name: string,
    namespace: string,
    timeoutMs: number = 300000
  ): Promise<void> {
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      const deployment = await this.k8sClient.getDeployment(name, namespace);
      if (!deployment?.status) {
        await this.sleep(5000);
        continue;
      }

      const desiredReplicas = deployment.spec?.replicas || 0;
      const updatedReplicas = deployment.status.updatedReplicas || 0;
      const availableReplicas = deployment.status.availableReplicas || 0;
      const readyReplicas = deployment.status.readyReplicas || 0;

      if (
        updatedReplicas === desiredReplicas &&
        availableReplicas === desiredReplicas &&
        readyReplicas === desiredReplicas
      ) {
        return;
      }

      await this.sleep(5000);
    }

    throw new Error(`Rollback timeout for ${namespace}/${name}`);
  }

  /**
   * Get rollback history
   */
  getHistory(filter?: {
    deploymentId?: string;
    status?: RollbackHistoryEntry["status"];
    triggeredBy?: RollbackHistoryEntry["triggeredBy"];
    limit?: number;
  }): RollbackHistoryEntry[] {
    let history = [...this.state.history];

    if (filter?.deploymentId) {
      history = history.filter((h) => h.deploymentId === filter.deploymentId);
    }
    if (filter?.status) {
      history = history.filter((h) => h.status === filter.status);
    }
    if (filter?.triggeredBy) {
      history = history.filter((h) => h.triggeredBy === filter.triggeredBy);
    }

    history.sort((a, b) => b.triggeredAt.getTime() - a.triggeredAt.getTime());

    if (filter?.limit) {
      history = history.slice(0, filter.limit);
    }

    return history;
  }

  /**
   * Get last successful rollback for deployment
   */
  getLastSuccessfulRollback(
    deploymentId: string
  ): RollbackHistoryEntry | undefined {
    return this.state.history
      .filter((h) => h.deploymentId === deploymentId && h.status === "success")
      .sort((a, b) => b.triggeredAt.getTime() - a.triggeredAt.getTime())[0];
  }

  /**
   * Check if deployment can be rolled back
   */
  canRollback(deploymentId: string): {
    canRollback: boolean;
    reason?: string;
  } {
    if (!this.policy.enabled || !this.policy.autoRollback) {
      return { canRollback: false, reason: "Rollback is disabled" };
    }

    if (this.isInCooldown(deploymentId)) {
      return { canRollback: false, reason: "In cooldown period" };
    }

    const attempts = this.state.attempts.get(deploymentId) || 0;
    if (attempts >= this.policy.maxRollbackAttempts) {
      return { canRollback: false, reason: "Max attempts exceeded" };
    }

    return { canRollback: true };
  }

  /**
   * Reset rollback attempts for deployment
   */
  resetAttempts(deploymentId: string): void {
    this.state.attempts.delete(deploymentId);
    this.state.cooldowns.delete(deploymentId);
  }

  /**
   * Update rollback policy
   */
  updatePolicy(policy: Partial<RollbackPolicy>): void {
    this.policy = { ...this.policy, ...policy };
  }

  /**
   * Get current policy
   */
  getPolicy(): RollbackPolicy {
    return { ...this.policy };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private getDefaultPolicy(): RollbackPolicy {
    return {
      id: "default",
      name: "Default Rollback Policy",
      enabled: true,
      autoRollback: true,
      maxRollbackAttempts: 3,
      cooldownPeriod: "5m",
      preserveHistory: 10,
      conditions: {
        onHealthCheckFailure: true,
        onAnalysisFailure: true,
        onDeploymentTimeout: true,
        onCrashLoopBackOff: true,
      },
    };
  }

  private isNamespaceAllowed(namespace: string): boolean {
    if (this.policy.excludeNamespaces?.includes(namespace)) {
      return false;
    }

    if (
      this.policy.includeNamespaces &&
      this.policy.includeNamespaces.length > 0
    ) {
      return this.policy.includeNamespaces.includes(namespace);
    }

    return true;
  }

  private isInCooldown(deploymentId: string): boolean {
    const cooldownEnd = this.state.cooldowns.get(deploymentId);
    if (!cooldownEnd) {
      return false;
    }
    return new Date() < cooldownEnd;
  }

  private setCooldown(deploymentId: string): void {
    const cooldownMs = this.parseDuration(this.policy.cooldownPeriod);
    const cooldownEnd = new Date(Date.now() + cooldownMs);
    this.state.cooldowns.set(deploymentId, cooldownEnd);
  }

  private addToHistory(entry: RollbackHistoryEntry): void {
    this.state.history.push(entry);

    // Prune old entries
    if (this.state.history.length > this.policy.preserveHistory * 10) {
      this.state.history = this.state.history
        .sort((a, b) => b.triggeredAt.getTime() - a.triggeredAt.getTime())
        .slice(0, this.policy.preserveHistory * 5);
    }
  }

  private generateId(): string {
    return `rb-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  }

  private parseDuration(duration: string): number {
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

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createRollbackManager(
  config: RollbackManagerConfig
): RollbackManager {
  return new RollbackManager(config);
}
