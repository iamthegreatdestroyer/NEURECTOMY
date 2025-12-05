/**
 * ArgoCD GitOps Integration
 * Sync deployments with ArgoCD applications
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";

// =============================================================================
// Types
// =============================================================================

export const ArgoApplicationSchema = z.object({
  name: z.string(),
  namespace: z.string().default("argocd"),
  project: z.string().default("default"),
  source: z.object({
    repoURL: z.string(),
    path: z.string(),
    targetRevision: z.string().default("HEAD"),
    helm: z
      .object({
        valueFiles: z.array(z.string()).optional(),
        values: z.string().optional(),
        parameters: z
          .array(
            z.object({
              name: z.string(),
              value: z.string(),
            })
          )
          .optional(),
      })
      .optional(),
    kustomize: z
      .object({
        images: z.array(z.string()).optional(),
        namePrefix: z.string().optional(),
        nameSuffix: z.string().optional(),
      })
      .optional(),
  }),
  destination: z.object({
    server: z.string().default("https://kubernetes.default.svc"),
    namespace: z.string(),
  }),
  syncPolicy: z
    .object({
      automated: z
        .object({
          prune: z.boolean().default(false),
          selfHeal: z.boolean().default(false),
          allowEmpty: z.boolean().default(false),
        })
        .optional(),
      syncOptions: z.array(z.string()).optional(),
      retry: z
        .object({
          limit: z.number(),
          backoff: z.object({
            duration: z.string(),
            factor: z.number(),
            maxDuration: z.string(),
          }),
        })
        .optional(),
    })
    .optional(),
});

export type ArgoApplication = z.infer<typeof ArgoApplicationSchema>;

export const ArgoSyncStatusSchema = z.enum(["Synced", "OutOfSync", "Unknown"]);

export const ArgoHealthStatusSchema = z.enum([
  "Healthy",
  "Progressing",
  "Degraded",
  "Suspended",
  "Missing",
  "Unknown",
]);

export const ArgoApplicationStatusSchema = z.object({
  sync: z.object({
    status: ArgoSyncStatusSchema,
    revision: z.string().optional(),
    comparedTo: z
      .object({
        source: z.object({
          repoURL: z.string(),
          path: z.string(),
          targetRevision: z.string(),
        }),
        destination: z.object({
          server: z.string(),
          namespace: z.string(),
        }),
      })
      .optional(),
  }),
  health: z.object({
    status: ArgoHealthStatusSchema,
    message: z.string().optional(),
  }),
  operationState: z
    .object({
      phase: z.enum(["Running", "Succeeded", "Failed", "Error", "Terminating"]),
      message: z.string().optional(),
      startedAt: z.string().optional(),
      finishedAt: z.string().optional(),
    })
    .optional(),
  resources: z
    .array(
      z.object({
        group: z.string().optional(),
        version: z.string(),
        kind: z.string(),
        namespace: z.string().optional(),
        name: z.string(),
        status: ArgoSyncStatusSchema,
        health: z
          .object({
            status: ArgoHealthStatusSchema,
            message: z.string().optional(),
          })
          .optional(),
      })
    )
    .optional(),
});

export type ArgoApplicationStatus = z.infer<typeof ArgoApplicationStatusSchema>;

export interface ArgoCDEvents {
  "argocd:sync:started": (app: string, revision: string) => void;
  "argocd:sync:completed": (app: string, status: ArgoApplicationStatus) => void;
  "argocd:sync:failed": (app: string, error: string) => void;
  "argocd:health:changed": (app: string, health: string) => void;
  "argocd:app:created": (app: ArgoApplication) => void;
  "argocd:app:updated": (app: ArgoApplication) => void;
  "argocd:app:deleted": (appName: string) => void;
}

export interface ArgoCDConfig {
  /** ArgoCD server URL */
  serverUrl: string;
  /** Authentication token */
  token?: string;
  /** Username/password auth */
  auth?: {
    username: string;
    password: string;
  };
  /** Skip TLS verification */
  insecure?: boolean;
  /** HTTP client timeout in ms */
  timeout?: number;
}

// =============================================================================
// ArgoCD Client Implementation
// =============================================================================

export class ArgoCDClient extends EventEmitter<ArgoCDEvents> {
  private serverUrl: string;
  private token?: string;
  private auth?: { username: string; password: string };
  private insecure: boolean;
  private timeout: number;

  constructor(config: ArgoCDConfig) {
    super();
    this.serverUrl = config.serverUrl.replace(/\/$/, "");
    this.token = config.token;
    this.auth = config.auth;
    this.insecure = config.insecure || false;
    this.timeout = config.timeout || 30000;
  }

  /**
   * Create an ArgoCD application
   */
  async createApplication(app: ArgoApplication): Promise<ArgoApplication> {
    const validated = ArgoApplicationSchema.parse(app);

    const response = await this.request("/api/v1/applications", {
      method: "POST",
      body: JSON.stringify({
        apiVersion: "argoproj.io/v1alpha1",
        kind: "Application",
        metadata: {
          name: validated.name,
          namespace: validated.namespace,
        },
        spec: {
          project: validated.project,
          source: validated.source,
          destination: validated.destination,
          syncPolicy: validated.syncPolicy,
        },
      }),
    });

    this.emit("argocd:app:created", validated);
    return validated;
  }

  /**
   * Get application by name
   */
  async getApplication(
    name: string
  ): Promise<(ArgoApplication & { status?: ArgoApplicationStatus }) | null> {
    try {
      const response = await this.request(`/api/v1/applications/${name}`);
      return this.parseApplicationResponse(response);
    } catch (error) {
      if (this.is404Error(error)) {
        return null;
      }
      throw error;
    }
  }

  /**
   * List all applications
   */
  async listApplications(options?: {
    project?: string;
    selector?: string;
  }): Promise<ArgoApplication[]> {
    const params = new URLSearchParams();
    if (options?.project) {
      params.set("project", options.project);
    }
    if (options?.selector) {
      params.set("selector", options.selector);
    }

    const queryString = params.toString();
    const url = `/api/v1/applications${queryString ? `?${queryString}` : ""}`;

    const response = await this.request(url);
    const items = response.items || [];

    return items.map((item: any) => this.parseApplicationResponse(item));
  }

  /**
   * Update application
   */
  async updateApplication(app: ArgoApplication): Promise<ArgoApplication> {
    const validated = ArgoApplicationSchema.parse(app);

    await this.request(`/api/v1/applications/${validated.name}`, {
      method: "PUT",
      body: JSON.stringify({
        apiVersion: "argoproj.io/v1alpha1",
        kind: "Application",
        metadata: {
          name: validated.name,
          namespace: validated.namespace,
        },
        spec: {
          project: validated.project,
          source: validated.source,
          destination: validated.destination,
          syncPolicy: validated.syncPolicy,
        },
      }),
    });

    this.emit("argocd:app:updated", validated);
    return validated;
  }

  /**
   * Delete application
   */
  async deleteApplication(
    name: string,
    cascade: boolean = true
  ): Promise<void> {
    await this.request(`/api/v1/applications/${name}?cascade=${cascade}`, {
      method: "DELETE",
    });
    this.emit("argocd:app:deleted", name);
  }

  /**
   * Sync application
   */
  async syncApplication(
    name: string,
    options?: {
      revision?: string;
      prune?: boolean;
      dryRun?: boolean;
      resources?: Array<{
        group?: string;
        kind: string;
        name: string;
        namespace?: string;
      }>;
    }
  ): Promise<void> {
    const revision = options?.revision || "HEAD";
    this.emit("argocd:sync:started", name, revision);

    try {
      await this.request(`/api/v1/applications/${name}/sync`, {
        method: "POST",
        body: JSON.stringify({
          revision: options?.revision,
          prune: options?.prune,
          dryRun: options?.dryRun,
          resources: options?.resources,
        }),
      });
    } catch (error) {
      this.emit(
        "argocd:sync:failed",
        name,
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  /**
   * Wait for sync to complete
   */
  async waitForSync(
    name: string,
    timeoutMs: number = 300000
  ): Promise<ArgoApplicationStatus> {
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      const app = await this.getApplication(name);

      if (!app?.status) {
        await this.sleep(5000);
        continue;
      }

      const operationPhase = app.status.operationState?.phase;

      if (operationPhase === "Succeeded") {
        this.emit("argocd:sync:completed", name, app.status);
        return app.status;
      }

      if (operationPhase === "Failed" || operationPhase === "Error") {
        const error = app.status.operationState?.message || "Sync failed";
        this.emit("argocd:sync:failed", name, error);
        throw new Error(`ArgoCD sync failed: ${error}`);
      }

      // Check if app is synced and healthy (no ongoing operation)
      if (
        !operationPhase &&
        app.status.sync.status === "Synced" &&
        app.status.health.status === "Healthy"
      ) {
        this.emit("argocd:sync:completed", name, app.status);
        return app.status;
      }

      await this.sleep(5000);
    }

    throw new Error(`Timeout waiting for ArgoCD sync: ${name}`);
  }

  /**
   * Get application status
   */
  async getStatus(name: string): Promise<ArgoApplicationStatus | null> {
    const app = await this.getApplication(name);
    return app?.status || null;
  }

  /**
   * Refresh application (fetch latest from git)
   */
  async refreshApplication(name: string, hard: boolean = false): Promise<void> {
    await this.request(
      `/api/v1/applications/${name}?refresh=${hard ? "hard" : "normal"}`,
      { method: "GET" }
    );
  }

  /**
   * Get application resource tree
   */
  async getResourceTree(name: string): Promise<any> {
    return this.request(`/api/v1/applications/${name}/resource-tree`);
  }

  /**
   * Get application manifests
   */
  async getManifests(name: string, revision?: string): Promise<any> {
    const params = revision ? `?revision=${revision}` : "";
    return this.request(`/api/v1/applications/${name}/manifests${params}`);
  }

  /**
   * Rollback application to previous revision
   */
  async rollback(name: string, revision: number): Promise<void> {
    await this.request(`/api/v1/applications/${name}/rollback`, {
      method: "POST",
      body: JSON.stringify({ id: revision }),
    });
  }

  /**
   * Get sync history
   */
  async getHistory(name: string): Promise<any[]> {
    const response = await this.request(`/api/v1/applications/${name}/history`);
    return response.history || [];
  }

  /**
   * Terminate running operation
   */
  async terminateOperation(name: string): Promise<void> {
    await this.request(`/api/v1/applications/${name}/operation`, {
      method: "DELETE",
    });
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private async request(path: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.serverUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`ArgoCD API error (${response.status}): ${errorBody}`);
      }

      const contentType = response.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        return response.json();
      }
      return response.text();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private parseApplicationResponse(response: any): ArgoApplication & {
    status?: ArgoApplicationStatus;
  } {
    return {
      name: response.metadata?.name,
      namespace: response.metadata?.namespace || "argocd",
      project: response.spec?.project || "default",
      source: response.spec?.source,
      destination: response.spec?.destination,
      syncPolicy: response.spec?.syncPolicy,
      status: response.status
        ? ArgoApplicationStatusSchema.parse(response.status)
        : undefined,
    };
  }

  private is404Error(error: unknown): boolean {
    return (
      error instanceof Error &&
      (error.message.includes("404") || error.message.includes("not found"))
    );
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createArgoCDClient(config: ArgoCDConfig): ArgoCDClient {
  return new ArgoCDClient(config);
}
