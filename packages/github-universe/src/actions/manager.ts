/**
 * @fileoverview GitHub Actions Manager
 * @module @neurectomy/github-universe/actions
 *
 * Comprehensive GitHub Actions management: workflows, runs, artifacts.
 *
 * @agents @SYNAPSE @FLUX
 */

import { EventEmitter } from "eventemitter3";
import { GitHubClient } from "../client";
import { type RepoIdentifier } from "../types";

// =============================================================================
// EVENTS
// =============================================================================

export interface ActionsManagerEvents {
  "workflow:triggered": (
    identifier: RepoIdentifier,
    workflowId: number,
    runId: number
  ) => void;
  "workflow:completed": (
    identifier: RepoIdentifier,
    runId: number,
    conclusion: string
  ) => void;
  "workflow:cancelled": (identifier: RepoIdentifier, runId: number) => void;
  "artifact:downloaded": (
    identifier: RepoIdentifier,
    artifactId: number
  ) => void;
  error: (error: Error, context: string) => void;
}

// =============================================================================
// TYPES
// =============================================================================

export interface Workflow {
  id: number;
  nodeId: string;
  name: string;
  path: string;
  state:
    | "active"
    | "disabled_manually"
    | "disabled_inactivity"
    | "deleted"
    | "unknown";
  createdAt: string;
  updatedAt: string;
  url: string;
  htmlUrl: string;
  badgeUrl: string;
}

export interface WorkflowRun {
  id: number;
  nodeId: string;
  name: string;
  displayTitle: string;
  headBranch: string;
  headSha: string;
  path: string;
  runNumber: number;
  runAttempt: number;
  event: string;
  status:
    | "queued"
    | "in_progress"
    | "completed"
    | "waiting"
    | "requested"
    | "pending";
  conclusion: string | null;
  workflowId: number;
  htmlUrl: string;
  createdAt: string;
  updatedAt: string;
  runStartedAt: string;
  actor: { login: string; avatarUrl: string } | null;
  triggeringActor: { login: string; avatarUrl: string } | null;
  jobs?: WorkflowJob[];
}

export interface WorkflowJob {
  id: number;
  runId: number;
  nodeId: string;
  name: string;
  status: "queued" | "in_progress" | "completed" | "waiting";
  conclusion: string | null;
  startedAt: string | null;
  completedAt: string | null;
  steps: WorkflowStep[];
  runnerName: string | null;
  runnerGroupName: string | null;
  labels: string[];
}

export interface WorkflowStep {
  name: string;
  status: "queued" | "in_progress" | "completed";
  conclusion: string | null;
  number: number;
  startedAt: string | null;
  completedAt: string | null;
}

export interface Artifact {
  id: number;
  nodeId: string;
  name: string;
  sizeInBytes: number;
  archiveDownloadUrl: string;
  expired: boolean;
  expiresAt: string;
  createdAt: string;
  updatedAt: string;
}

export interface WorkflowDispatchInputs {
  [key: string]: string | boolean | number;
}

export interface WorkflowListOptions {
  perPage?: number;
  page?: number;
}

export interface WorkflowRunListOptions {
  actor?: string;
  branch?: string;
  event?: string;
  status?:
    | "queued"
    | "in_progress"
    | "completed"
    | "waiting"
    | "requested"
    | "pending";
  created?: string;
  headSha?: string;
  perPage?: number;
  page?: number;
  excludePullRequests?: boolean;
}

export interface WorkflowUsage {
  billable: {
    UBUNTU?: { totalMs: number; jobs: number };
    MACOS?: { totalMs: number; jobs: number };
    WINDOWS?: { totalMs: number; jobs: number };
  };
}

// =============================================================================
// ACTIONS MANAGER
// =============================================================================

/**
 * GitHub Actions Manager
 *
 * Manages GitHub Actions workflows, runs, jobs, and artifacts.
 *
 * @example
 * ```typescript
 * const actionsManager = new ActionsManager(client);
 *
 * // List workflows
 * const workflows = await actionsManager.listWorkflows(
 *   { owner: 'neurectomy', repo: 'core' }
 * );
 *
 * // Trigger a workflow
 * const runId = await actionsManager.triggerWorkflow(
 *   { owner: 'neurectomy', repo: 'core' },
 *   'ci.yml',
 *   'main',
 *   { environment: 'staging' }
 * );
 *
 * // Get run status
 * const run = await actionsManager.getRun(
 *   { owner: 'neurectomy', repo: 'core' },
 *   runId
 * );
 * ```
 */
export class ActionsManager extends EventEmitter<ActionsManagerEvents> {
  private client: GitHubClient;

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ===========================================================================
  // WORKFLOW OPERATIONS
  // ===========================================================================

  /**
   * List repository workflows
   */
  async listWorkflows(
    identifier: RepoIdentifier,
    options?: WorkflowListOptions
  ): Promise<Workflow[]> {
    const response = await this.client.rest<{
      total_count: number;
      workflows: Record<string, unknown>[];
    }>("GET /repos/{owner}/{repo}/actions/workflows", {
      owner: identifier.owner,
      repo: identifier.repo,
      per_page: options?.perPage ?? 100,
      page: options?.page ?? 1,
    });

    return response.workflows.map((w) => this.transformWorkflow(w));
  }

  /**
   * Get a workflow by ID or filename
   */
  async getWorkflow(
    identifier: RepoIdentifier,
    workflowIdOrFile: number | string
  ): Promise<Workflow> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/actions/workflows/{workflow_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        workflow_id: workflowIdOrFile,
      }
    );

    return this.transformWorkflow(response);
  }

  /**
   * Enable a workflow
   */
  async enableWorkflow(
    identifier: RepoIdentifier,
    workflowIdOrFile: number | string
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/actions/workflows/{workflow_id}/enable",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        workflow_id: workflowIdOrFile,
      }
    );
  }

  /**
   * Disable a workflow
   */
  async disableWorkflow(
    identifier: RepoIdentifier,
    workflowIdOrFile: number | string
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/actions/workflows/{workflow_id}/disable",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        workflow_id: workflowIdOrFile,
      }
    );
  }

  /**
   * Trigger a workflow_dispatch event
   */
  async triggerWorkflow(
    identifier: RepoIdentifier,
    workflowIdOrFile: number | string,
    ref: string,
    inputs?: WorkflowDispatchInputs
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        workflow_id: workflowIdOrFile,
        ref,
        inputs,
      }
    );

    // Note: GitHub doesn't return the run ID directly, need to poll
    this.emit(
      "workflow:triggered",
      identifier,
      typeof workflowIdOrFile === "number" ? workflowIdOrFile : 0,
      0
    );
  }

  /**
   * Get workflow usage
   */
  async getWorkflowUsage(
    identifier: RepoIdentifier,
    workflowIdOrFile: number | string
  ): Promise<WorkflowUsage> {
    const response = await this.client.rest<{
      billable: Record<string, { total_ms: number; jobs: number }>;
    }>("GET /repos/{owner}/{repo}/actions/workflows/{workflow_id}/timing", {
      owner: identifier.owner,
      repo: identifier.repo,
      workflow_id: workflowIdOrFile,
    });

    return {
      billable: {
        UBUNTU: response.billable.UBUNTU
          ? {
              totalMs: response.billable.UBUNTU.total_ms,
              jobs: response.billable.UBUNTU.jobs,
            }
          : undefined,
        MACOS: response.billable.MACOS
          ? {
              totalMs: response.billable.MACOS.total_ms,
              jobs: response.billable.MACOS.jobs,
            }
          : undefined,
        WINDOWS: response.billable.WINDOWS
          ? {
              totalMs: response.billable.WINDOWS.total_ms,
              jobs: response.billable.WINDOWS.jobs,
            }
          : undefined,
      },
    };
  }

  // ===========================================================================
  // WORKFLOW RUN OPERATIONS
  // ===========================================================================

  /**
   * List workflow runs
   */
  async listRuns(
    identifier: RepoIdentifier,
    workflowIdOrFile?: number | string,
    options?: WorkflowRunListOptions
  ): Promise<WorkflowRun[]> {
    let endpoint = "GET /repos/{owner}/{repo}/actions/runs";
    const params: Record<string, unknown> = {
      owner: identifier.owner,
      repo: identifier.repo,
      actor: options?.actor,
      branch: options?.branch,
      event: options?.event,
      status: options?.status,
      created: options?.created,
      head_sha: options?.headSha,
      per_page: options?.perPage ?? 100,
      page: options?.page ?? 1,
      exclude_pull_requests: options?.excludePullRequests,
    };

    if (workflowIdOrFile) {
      endpoint =
        "GET /repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs";
      params.workflow_id = workflowIdOrFile;
    }

    const response = await this.client.rest<{
      total_count: number;
      workflow_runs: Record<string, unknown>[];
    }>(endpoint, params);

    return response.workflow_runs.map((r) => this.transformRun(r));
  }

  /**
   * Get a workflow run
   */
  async getRun(
    identifier: RepoIdentifier,
    runId: number
  ): Promise<WorkflowRun> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/actions/runs/{run_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
      }
    );

    return this.transformRun(response);
  }

  /**
   * Re-run a workflow run
   */
  async rerunRun(
    identifier: RepoIdentifier,
    runId: number,
    options?: { enableDebugLogging?: boolean }
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
        enable_debug_logging: options?.enableDebugLogging,
      }
    );
  }

  /**
   * Re-run failed jobs in a workflow run
   */
  async rerunFailedJobs(
    identifier: RepoIdentifier,
    runId: number,
    options?: { enableDebugLogging?: boolean }
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
        enable_debug_logging: options?.enableDebugLogging,
      }
    );
  }

  /**
   * Cancel a workflow run
   */
  async cancelRun(identifier: RepoIdentifier, runId: number): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/runs/{run_id}/cancel",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
      }
    );

    this.emit("workflow:cancelled", identifier, runId);
  }

  /**
   * Delete a workflow run
   */
  async deleteRun(identifier: RepoIdentifier, runId: number): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/runs/{run_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
      }
    );
  }

  /**
   * Get workflow run logs
   */
  async getRunLogs(
    identifier: RepoIdentifier,
    runId: number
  ): Promise<ArrayBuffer> {
    const response = await this.client.rest<ArrayBuffer>(
      "GET /repos/{owner}/{repo}/actions/runs/{run_id}/logs",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
        headers: { Accept: "application/vnd.github+json" },
      }
    );

    return response;
  }

  /**
   * Delete workflow run logs
   */
  async deleteRunLogs(
    identifier: RepoIdentifier,
    runId: number
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/runs/{run_id}/logs",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
      }
    );
  }

  /**
   * Approve a workflow run for a fork pull request
   */
  async approveRun(identifier: RepoIdentifier, runId: number): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/runs/{run_id}/approve",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        run_id: runId,
      }
    );
  }

  /**
   * Wait for a run to complete
   */
  async waitForRun(
    identifier: RepoIdentifier,
    runId: number,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<WorkflowRun> {
    const pollInterval = options?.pollIntervalMs ?? 5000;
    const timeout = options?.timeoutMs ?? 600000; // 10 minutes default
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const run = await this.getRun(identifier, runId);

      if (run.status === "completed") {
        this.emit(
          "workflow:completed",
          identifier,
          runId,
          run.conclusion ?? "unknown"
        );
        return run;
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new Error(
      `Workflow run ${runId} did not complete within ${timeout}ms`
    );
  }

  // ===========================================================================
  // JOB OPERATIONS
  // ===========================================================================

  /**
   * List jobs for a workflow run
   */
  async listJobs(
    identifier: RepoIdentifier,
    runId: number,
    options?: { filter?: "latest" | "all"; perPage?: number }
  ): Promise<WorkflowJob[]> {
    const response = await this.client.rest<{
      total_count: number;
      jobs: Record<string, unknown>[];
    }>("GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs", {
      owner: identifier.owner,
      repo: identifier.repo,
      run_id: runId,
      filter: options?.filter ?? "latest",
      per_page: options?.perPage ?? 100,
    });

    return response.jobs.map((j) => this.transformJob(j));
  }

  /**
   * Get a job
   */
  async getJob(
    identifier: RepoIdentifier,
    jobId: number
  ): Promise<WorkflowJob> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/actions/jobs/{job_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        job_id: jobId,
      }
    );

    return this.transformJob(response);
  }

  /**
   * Get job logs
   */
  async getJobLogs(identifier: RepoIdentifier, jobId: number): Promise<string> {
    const response = await this.client.rest<string>(
      "GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        job_id: jobId,
      }
    );

    return response;
  }

  /**
   * Re-run a job
   */
  async rerunJob(
    identifier: RepoIdentifier,
    jobId: number,
    options?: { enableDebugLogging?: boolean }
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/actions/jobs/{job_id}/rerun",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        job_id: jobId,
        enable_debug_logging: options?.enableDebugLogging,
      }
    );
  }

  // ===========================================================================
  // ARTIFACT OPERATIONS
  // ===========================================================================

  /**
   * List artifacts for a repository
   */
  async listArtifacts(
    identifier: RepoIdentifier,
    options?: { perPage?: number; page?: number; name?: string }
  ): Promise<Artifact[]> {
    const response = await this.client.rest<{
      total_count: number;
      artifacts: Record<string, unknown>[];
    }>("GET /repos/{owner}/{repo}/actions/artifacts", {
      owner: identifier.owner,
      repo: identifier.repo,
      per_page: options?.perPage ?? 100,
      page: options?.page ?? 1,
      name: options?.name,
    });

    return response.artifacts.map((a) => this.transformArtifact(a));
  }

  /**
   * List artifacts for a workflow run
   */
  async listRunArtifacts(
    identifier: RepoIdentifier,
    runId: number,
    options?: { perPage?: number; page?: number; name?: string }
  ): Promise<Artifact[]> {
    const response = await this.client.rest<{
      total_count: number;
      artifacts: Record<string, unknown>[];
    }>("GET /repos/{owner}/{repo}/actions/runs/{run_id}/artifacts", {
      owner: identifier.owner,
      repo: identifier.repo,
      run_id: runId,
      per_page: options?.perPage ?? 100,
      page: options?.page ?? 1,
      name: options?.name,
    });

    return response.artifacts.map((a) => this.transformArtifact(a));
  }

  /**
   * Get an artifact
   */
  async getArtifact(
    identifier: RepoIdentifier,
    artifactId: number
  ): Promise<Artifact> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/actions/artifacts/{artifact_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        artifact_id: artifactId,
      }
    );

    return this.transformArtifact(response);
  }

  /**
   * Download an artifact
   */
  async downloadArtifact(
    identifier: RepoIdentifier,
    artifactId: number,
    archiveFormat: "zip" = "zip"
  ): Promise<ArrayBuffer> {
    const response = await this.client.rest<ArrayBuffer>(
      "GET /repos/{owner}/{repo}/actions/artifacts/{artifact_id}/{archive_format}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        artifact_id: artifactId,
        archive_format: archiveFormat,
      }
    );

    this.emit("artifact:downloaded", identifier, artifactId);
    return response;
  }

  /**
   * Delete an artifact
   */
  async deleteArtifact(
    identifier: RepoIdentifier,
    artifactId: number
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/artifacts/{artifact_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        artifact_id: artifactId,
      }
    );
  }

  // ===========================================================================
  // SECRETS MANAGEMENT
  // ===========================================================================

  /**
   * List repository secrets
   */
  async listSecrets(
    identifier: RepoIdentifier
  ): Promise<Array<{ name: string; createdAt: string; updatedAt: string }>> {
    const response = await this.client.rest<{
      total_count: number;
      secrets: Array<{ name: string; created_at: string; updated_at: string }>;
    }>("GET /repos/{owner}/{repo}/actions/secrets", {
      owner: identifier.owner,
      repo: identifier.repo,
    });

    return response.secrets.map((s) => ({
      name: s.name,
      createdAt: s.created_at,
      updatedAt: s.updated_at,
    }));
  }

  /**
   * Get repository public key for encrypting secrets
   */
  async getPublicKey(
    identifier: RepoIdentifier
  ): Promise<{ keyId: string; key: string }> {
    const response = await this.client.rest<{ key_id: string; key: string }>(
      "GET /repos/{owner}/{repo}/actions/secrets/public-key",
      {
        owner: identifier.owner,
        repo: identifier.repo,
      }
    );

    return {
      keyId: response.key_id,
      key: response.key,
    };
  }

  /**
   * Create or update a secret
   * Note: Value must be encrypted with the repository's public key
   */
  async setSecret(
    identifier: RepoIdentifier,
    secretName: string,
    encryptedValue: string,
    keyId: string
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/actions/secrets/{secret_name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        secret_name: secretName,
        encrypted_value: encryptedValue,
        key_id: keyId,
      }
    );
  }

  /**
   * Delete a secret
   */
  async deleteSecret(
    identifier: RepoIdentifier,
    secretName: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/secrets/{secret_name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        secret_name: secretName,
      }
    );
  }

  // ===========================================================================
  // VARIABLES MANAGEMENT
  // ===========================================================================

  /**
   * List repository variables
   */
  async listVariables(
    identifier: RepoIdentifier
  ): Promise<
    Array<{ name: string; value: string; createdAt: string; updatedAt: string }>
  > {
    const response = await this.client.rest<{
      total_count: number;
      variables: Array<{
        name: string;
        value: string;
        created_at: string;
        updated_at: string;
      }>;
    }>("GET /repos/{owner}/{repo}/actions/variables", {
      owner: identifier.owner,
      repo: identifier.repo,
    });

    return response.variables.map((v) => ({
      name: v.name,
      value: v.value,
      createdAt: v.created_at,
      updatedAt: v.updated_at,
    }));
  }

  /**
   * Create a variable
   */
  async createVariable(
    identifier: RepoIdentifier,
    name: string,
    value: string
  ): Promise<void> {
    await this.client.rest("POST /repos/{owner}/{repo}/actions/variables", {
      owner: identifier.owner,
      repo: identifier.repo,
      name,
      value,
    });
  }

  /**
   * Update a variable
   */
  async updateVariable(
    identifier: RepoIdentifier,
    name: string,
    value: string
  ): Promise<void> {
    await this.client.rest(
      "PATCH /repos/{owner}/{repo}/actions/variables/{name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        name,
        value,
      }
    );
  }

  /**
   * Delete a variable
   */
  async deleteVariable(
    identifier: RepoIdentifier,
    name: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/variables/{name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        name,
      }
    );
  }

  // ===========================================================================
  // RUNNERS
  // ===========================================================================

  /**
   * List self-hosted runners
   */
  async listRunners(identifier: RepoIdentifier): Promise<
    Array<{
      id: number;
      name: string;
      os: string;
      status: string;
      busy: boolean;
      labels: Array<{ name: string }>;
    }>
  > {
    const response = await this.client.rest<{
      total_count: number;
      runners: Array<{
        id: number;
        name: string;
        os: string;
        status: string;
        busy: boolean;
        labels: Array<{ name: string }>;
      }>;
    }>("GET /repos/{owner}/{repo}/actions/runners", {
      owner: identifier.owner,
      repo: identifier.repo,
    });

    return response.runners;
  }

  /**
   * Get registration token for a self-hosted runner
   */
  async getRunnerRegistrationToken(
    identifier: RepoIdentifier
  ): Promise<{ token: string; expiresAt: string }> {
    const response = await this.client.rest<{
      token: string;
      expires_at: string;
    }>("POST /repos/{owner}/{repo}/actions/runners/registration-token", {
      owner: identifier.owner,
      repo: identifier.repo,
    });

    return {
      token: response.token,
      expiresAt: response.expires_at,
    };
  }

  /**
   * Delete a self-hosted runner
   */
  async deleteRunner(
    identifier: RepoIdentifier,
    runnerId: number
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/runners/{runner_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        runner_id: runnerId,
      }
    );
  }

  // ===========================================================================
  // CACHES
  // ===========================================================================

  /**
   * List repository caches
   */
  async listCaches(
    identifier: RepoIdentifier,
    options?: {
      key?: string;
      ref?: string;
      sort?: "created_at" | "last_accessed_at" | "size_in_bytes";
    }
  ): Promise<
    Array<{
      id: number;
      key: string;
      ref: string;
      version: string;
      sizeInBytes: number;
      createdAt: string;
      lastAccessedAt: string;
    }>
  > {
    const response = await this.client.rest<{
      total_count: number;
      actions_caches: Array<{
        id: number;
        key: string;
        ref: string;
        version: string;
        size_in_bytes: number;
        created_at: string;
        last_accessed_at: string;
      }>;
    }>("GET /repos/{owner}/{repo}/actions/caches", {
      owner: identifier.owner,
      repo: identifier.repo,
      key: options?.key,
      ref: options?.ref,
      sort: options?.sort,
    });

    return response.actions_caches.map((c) => ({
      id: c.id,
      key: c.key,
      ref: c.ref,
      version: c.version,
      sizeInBytes: c.size_in_bytes,
      createdAt: c.created_at,
      lastAccessedAt: c.last_accessed_at,
    }));
  }

  /**
   * Delete a cache
   */
  async deleteCache(
    identifier: RepoIdentifier,
    cacheId: number
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/actions/caches/{cache_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        cache_id: cacheId,
      }
    );
  }

  /**
   * Delete caches by key
   */
  async deleteCacheByKey(
    identifier: RepoIdentifier,
    key: string,
    ref?: string
  ): Promise<void> {
    await this.client.rest("DELETE /repos/{owner}/{repo}/actions/caches", {
      owner: identifier.owner,
      repo: identifier.repo,
      key,
      ref,
    });
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private transformWorkflow(data: Record<string, unknown>): Workflow {
    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      name: data.name as string,
      path: data.path as string,
      state: data.state as Workflow["state"],
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
      url: data.url as string,
      htmlUrl: data.html_url as string,
      badgeUrl: data.badge_url as string,
    };
  }

  private transformRun(data: Record<string, unknown>): WorkflowRun {
    const actor = data.actor as { login: string; avatar_url: string } | null;
    const triggeringActor = data.triggering_actor as {
      login: string;
      avatar_url: string;
    } | null;

    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      name: data.name as string,
      displayTitle: data.display_title as string,
      headBranch: data.head_branch as string,
      headSha: data.head_sha as string,
      path: data.path as string,
      runNumber: data.run_number as number,
      runAttempt: data.run_attempt as number,
      event: data.event as string,
      status: data.status as WorkflowRun["status"],
      conclusion: data.conclusion as string | null,
      workflowId: data.workflow_id as number,
      htmlUrl: data.html_url as string,
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
      runStartedAt: data.run_started_at as string,
      actor: actor ? { login: actor.login, avatarUrl: actor.avatar_url } : null,
      triggeringActor: triggeringActor
        ? {
            login: triggeringActor.login,
            avatarUrl: triggeringActor.avatar_url,
          }
        : null,
    };
  }

  private transformJob(data: Record<string, unknown>): WorkflowJob {
    const steps = data.steps as
      | Array<{
          name: string;
          status: string;
          conclusion: string | null;
          number: number;
          started_at: string | null;
          completed_at: string | null;
        }>
      | undefined;

    return {
      id: data.id as number,
      runId: data.run_id as number,
      nodeId: data.node_id as string,
      name: data.name as string,
      status: data.status as WorkflowJob["status"],
      conclusion: data.conclusion as string | null,
      startedAt: data.started_at as string | null,
      completedAt: data.completed_at as string | null,
      steps:
        steps?.map((s) => ({
          name: s.name,
          status: s.status as WorkflowStep["status"],
          conclusion: s.conclusion,
          number: s.number,
          startedAt: s.started_at,
          completedAt: s.completed_at,
        })) ?? [],
      runnerName: data.runner_name as string | null,
      runnerGroupName: data.runner_group_name as string | null,
      labels: (data.labels as string[]) ?? [],
    };
  }

  private transformArtifact(data: Record<string, unknown>): Artifact {
    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      name: data.name as string,
      sizeInBytes: data.size_in_bytes as number,
      archiveDownloadUrl: data.archive_download_url as string,
      expired: data.expired as boolean,
      expiresAt: data.expires_at as string,
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
    };
  }
}
