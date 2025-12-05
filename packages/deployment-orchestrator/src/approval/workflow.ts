/**
 * Deployment Approval Workflow
 * Manual approval gates for controlled deployments
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";

// =============================================================================
// Types
// =============================================================================

export const ApprovalRequestSchema = z.object({
  id: z.string(),
  deploymentId: z.string(),
  deploymentName: z.string(),
  namespace: z.string(),
  environment: z.enum(["development", "staging", "production", "canary"]),
  requestedBy: z.string(),
  requestedAt: z.date(),
  expiresAt: z.date().optional(),
  status: z.enum(["pending", "approved", "rejected", "expired", "cancelled"]),
  approvers: z.array(z.string()),
  requiredApprovals: z.number().min(1),
  currentApprovals: z.array(
    z.object({
      approver: z.string(),
      decision: z.enum(["approved", "rejected"]),
      comment: z.string().optional(),
      timestamp: z.date(),
    })
  ),
  metadata: z.record(z.unknown()).optional(),
});

export type ApprovalRequest = z.infer<typeof ApprovalRequestSchema>;

export const ApprovalPolicySchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  enabled: z.boolean().default(true),
  environments: z.array(
    z.enum(["development", "staging", "production", "canary"])
  ),
  requiredApprovals: z.number().min(1).default(1),
  approvers: z.array(z.string()),
  autoApprove: z
    .object({
      enabled: z.boolean(),
      conditions: z
        .object({
          maxChangedFiles: z.number().optional(),
          allowedBranches: z.array(z.string()).optional(),
          requiredLabels: z.array(z.string()).optional(),
        })
        .optional(),
    })
    .optional(),
  timeout: z.string().default("24h"),
  notificationChannels: z.array(z.string()).optional(),
});

export type ApprovalPolicy = z.infer<typeof ApprovalPolicySchema>;

export interface ApprovalWorkflowEvents {
  "approval:requested": (request: ApprovalRequest) => void;
  "approval:approved": (request: ApprovalRequest) => void;
  "approval:rejected": (request: ApprovalRequest) => void;
  "approval:expired": (request: ApprovalRequest) => void;
  "approval:cancelled": (request: ApprovalRequest) => void;
  "approval:decision": (
    request: ApprovalRequest,
    decision: ApprovalDecision
  ) => void;
}

export interface ApprovalDecision {
  approver: string;
  decision: "approved" | "rejected";
  comment?: string;
  timestamp: Date;
}

export interface ApprovalWorkflowConfig {
  /** Storage backend for persisting approval requests */
  storage?: ApprovalStorage;
  /** Notification service for sending approval notifications */
  notifier?: ApprovalNotifier;
  /** Check interval for expired approvals in ms */
  expirationCheckInterval?: number;
}

export interface ApprovalStorage {
  save(request: ApprovalRequest): Promise<void>;
  get(id: string): Promise<ApprovalRequest | null>;
  list(filter?: ApprovalFilter): Promise<ApprovalRequest[]>;
  update(id: string, updates: Partial<ApprovalRequest>): Promise<void>;
  delete(id: string): Promise<void>;
}

export interface ApprovalNotifier {
  notify(
    request: ApprovalRequest,
    event: "requested" | "approved" | "rejected" | "expired"
  ): Promise<void>;
}

export interface ApprovalFilter {
  deploymentId?: string;
  environment?: string;
  status?: ApprovalRequest["status"];
  requestedBy?: string;
}

// =============================================================================
// In-Memory Storage (Default)
// =============================================================================

class InMemoryApprovalStorage implements ApprovalStorage {
  private requests = new Map<string, ApprovalRequest>();

  async save(request: ApprovalRequest): Promise<void> {
    this.requests.set(request.id, { ...request });
  }

  async get(id: string): Promise<ApprovalRequest | null> {
    return this.requests.get(id) || null;
  }

  async list(filter?: ApprovalFilter): Promise<ApprovalRequest[]> {
    let results = Array.from(this.requests.values());

    if (filter) {
      if (filter.deploymentId) {
        results = results.filter((r) => r.deploymentId === filter.deploymentId);
      }
      if (filter.environment) {
        results = results.filter((r) => r.environment === filter.environment);
      }
      if (filter.status) {
        results = results.filter((r) => r.status === filter.status);
      }
      if (filter.requestedBy) {
        results = results.filter((r) => r.requestedBy === filter.requestedBy);
      }
    }

    return results;
  }

  async update(id: string, updates: Partial<ApprovalRequest>): Promise<void> {
    const existing = this.requests.get(id);
    if (existing) {
      this.requests.set(id, { ...existing, ...updates });
    }
  }

  async delete(id: string): Promise<void> {
    this.requests.delete(id);
  }
}

// =============================================================================
// Approval Workflow Manager
// =============================================================================

export class ApprovalWorkflowManager extends EventEmitter<ApprovalWorkflowEvents> {
  private storage: ApprovalStorage;
  private notifier?: ApprovalNotifier;
  private policies = new Map<string, ApprovalPolicy>();
  private expirationChecker?: NodeJS.Timeout;
  private expirationCheckInterval: number;

  constructor(config: ApprovalWorkflowConfig = {}) {
    super();
    this.storage = config.storage || new InMemoryApprovalStorage();
    this.notifier = config.notifier;
    this.expirationCheckInterval = config.expirationCheckInterval || 60000;

    this.startExpirationChecker();
  }

  /**
   * Register an approval policy
   */
  registerPolicy(policy: ApprovalPolicy): void {
    const validated = ApprovalPolicySchema.parse(policy);
    this.policies.set(validated.id, validated);
  }

  /**
   * Get policy by ID
   */
  getPolicy(id: string): ApprovalPolicy | undefined {
    return this.policies.get(id);
  }

  /**
   * List all policies
   */
  listPolicies(): ApprovalPolicy[] {
    return Array.from(this.policies.values());
  }

  /**
   * Delete a policy
   */
  deletePolicy(id: string): boolean {
    return this.policies.delete(id);
  }

  /**
   * Request approval for a deployment
   */
  async requestApproval(params: {
    deploymentId: string;
    deploymentName: string;
    namespace: string;
    environment: ApprovalRequest["environment"];
    requestedBy: string;
    metadata?: Record<string, unknown>;
  }): Promise<ApprovalRequest> {
    // Find applicable policy
    const policy = this.findApplicablePolicy(params.environment);

    if (!policy) {
      // No policy means auto-approve
      const request: ApprovalRequest = {
        id: this.generateId(),
        ...params,
        requestedAt: new Date(),
        status: "approved",
        approvers: [],
        requiredApprovals: 0,
        currentApprovals: [],
      };
      await this.storage.save(request);
      return request;
    }

    // Check for auto-approval conditions
    if (policy.autoApprove?.enabled) {
      const canAutoApprove = await this.checkAutoApprovalConditions(
        params.metadata,
        policy.autoApprove.conditions
      );
      if (canAutoApprove) {
        const request: ApprovalRequest = {
          id: this.generateId(),
          ...params,
          requestedAt: new Date(),
          status: "approved",
          approvers: policy.approvers,
          requiredApprovals: policy.requiredApprovals,
          currentApprovals: [
            {
              approver: "system",
              decision: "approved",
              comment: "Auto-approved based on policy conditions",
              timestamp: new Date(),
            },
          ],
        };
        await this.storage.save(request);
        this.emit("approval:approved", request);
        return request;
      }
    }

    // Calculate expiration
    const expiresAt = new Date(Date.now() + this.parseDuration(policy.timeout));

    // Create approval request
    const request: ApprovalRequest = {
      id: this.generateId(),
      ...params,
      requestedAt: new Date(),
      expiresAt,
      status: "pending",
      approvers: policy.approvers,
      requiredApprovals: policy.requiredApprovals,
      currentApprovals: [],
    };

    await this.storage.save(request);
    this.emit("approval:requested", request);

    // Send notifications
    await this.notifier?.notify(request, "requested");

    return request;
  }

  /**
   * Process an approval decision
   */
  async processDecision(
    requestId: string,
    decision: ApprovalDecision
  ): Promise<ApprovalRequest> {
    const request = await this.storage.get(requestId);

    if (!request) {
      throw new Error(`Approval request ${requestId} not found`);
    }

    if (request.status !== "pending") {
      throw new Error(
        `Approval request ${requestId} is ${request.status}, cannot process decision`
      );
    }

    // Verify approver is authorized
    if (!request.approvers.includes(decision.approver)) {
      throw new Error(
        `User ${decision.approver} is not authorized to approve this request`
      );
    }

    // Check if already decided by this approver
    const existingDecision = request.currentApprovals.find(
      (a) => a.approver === decision.approver
    );
    if (existingDecision) {
      throw new Error(
        `User ${decision.approver} has already provided a decision`
      );
    }

    // Add decision
    request.currentApprovals.push(decision);
    this.emit("approval:decision", request, decision);

    // Check if rejected
    if (decision.decision === "rejected") {
      request.status = "rejected";
      await this.storage.update(requestId, request);
      this.emit("approval:rejected", request);
      await this.notifier?.notify(request, "rejected");
      return request;
    }

    // Check if enough approvals
    const approvalCount = request.currentApprovals.filter(
      (a) => a.decision === "approved"
    ).length;

    if (approvalCount >= request.requiredApprovals) {
      request.status = "approved";
      await this.storage.update(requestId, request);
      this.emit("approval:approved", request);
      await this.notifier?.notify(request, "approved");
      return request;
    }

    // Still pending
    await this.storage.update(requestId, request);
    return request;
  }

  /**
   * Cancel an approval request
   */
  async cancelRequest(requestId: string, cancelledBy: string): Promise<void> {
    const request = await this.storage.get(requestId);

    if (!request) {
      throw new Error(`Approval request ${requestId} not found`);
    }

    if (request.status !== "pending") {
      throw new Error(`Cannot cancel request with status ${request.status}`);
    }

    request.status = "cancelled";
    await this.storage.update(requestId, request);
    this.emit("approval:cancelled", request);
  }

  /**
   * Get approval request by ID
   */
  async getRequest(id: string): Promise<ApprovalRequest | null> {
    return this.storage.get(id);
  }

  /**
   * List approval requests
   */
  async listRequests(filter?: ApprovalFilter): Promise<ApprovalRequest[]> {
    return this.storage.list(filter);
  }

  /**
   * Get pending approvals for an approver
   */
  async getPendingForApprover(approver: string): Promise<ApprovalRequest[]> {
    const pending = await this.storage.list({ status: "pending" });
    return pending.filter(
      (r) =>
        r.approvers.includes(approver) &&
        !r.currentApprovals.some((a) => a.approver === approver)
    );
  }

  /**
   * Check if deployment needs approval
   */
  async needsApproval(
    environment: ApprovalRequest["environment"]
  ): Promise<boolean> {
    const policy = this.findApplicablePolicy(environment);
    return policy !== undefined && policy.enabled;
  }

  /**
   * Wait for approval with timeout
   */
  async waitForApproval(
    requestId: string,
    timeoutMs: number = 3600000
  ): Promise<ApprovalRequest> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const request = await this.storage.get(requestId);

      if (!request) {
        throw new Error(`Approval request ${requestId} not found`);
      }

      if (request.status === "approved") {
        return request;
      }

      if (request.status === "rejected") {
        throw new Error(`Approval request ${requestId} was rejected`);
      }

      if (request.status === "expired") {
        throw new Error(`Approval request ${requestId} expired`);
      }

      if (request.status === "cancelled") {
        throw new Error(`Approval request ${requestId} was cancelled`);
      }

      // Wait before checking again
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    throw new Error(`Timeout waiting for approval request ${requestId}`);
  }

  /**
   * Stop the workflow manager
   */
  stop(): void {
    if (this.expirationChecker) {
      clearInterval(this.expirationChecker);
    }
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private findApplicablePolicy(
    environment: ApprovalRequest["environment"]
  ): ApprovalPolicy | undefined {
    for (const policy of this.policies.values()) {
      if (policy.enabled && policy.environments.includes(environment)) {
        return policy;
      }
    }
    return undefined;
  }

  private async checkAutoApprovalConditions(
    metadata?: Record<string, unknown>,
    conditions?: ApprovalPolicy["autoApprove"]["conditions"]
  ): Promise<boolean> {
    if (!conditions) {
      return true;
    }

    if (conditions.maxChangedFiles !== undefined) {
      const changedFiles = (metadata?.changedFiles as number) || 0;
      if (changedFiles > conditions.maxChangedFiles) {
        return false;
      }
    }

    if (conditions.allowedBranches?.length) {
      const branch = (metadata?.branch as string) || "";
      if (!conditions.allowedBranches.some((b) => branch.match(b))) {
        return false;
      }
    }

    if (conditions.requiredLabels?.length) {
      const labels = (metadata?.labels as string[]) || [];
      if (!conditions.requiredLabels.every((l) => labels.includes(l))) {
        return false;
      }
    }

    return true;
  }

  private startExpirationChecker(): void {
    this.expirationChecker = setInterval(async () => {
      try {
        await this.checkExpiredRequests();
      } catch (error) {
        console.error("Error checking expired approval requests:", error);
      }
    }, this.expirationCheckInterval);
  }

  private async checkExpiredRequests(): Promise<void> {
    const pending = await this.storage.list({ status: "pending" });
    const now = new Date();

    for (const request of pending) {
      if (request.expiresAt && request.expiresAt < now) {
        request.status = "expired";
        await this.storage.update(request.id, request);
        this.emit("approval:expired", request);
        await this.notifier?.notify(request, "expired");
      }
    }
  }

  private generateId(): string {
    return `apr-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
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
}

// =============================================================================
// Factory Function
// =============================================================================

export function createApprovalWorkflow(
  config?: ApprovalWorkflowConfig
): ApprovalWorkflowManager {
  return new ApprovalWorkflowManager(config);
}
