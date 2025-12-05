/**
 * Flux GitOps Integration
 * Sync deployments with FluxCD
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";
import type { KubernetesClient } from "../kubernetes/client";

// =============================================================================
// Helper Types for Kubernetes Resources
// =============================================================================

/** Generic K8s resource structure for type safety when reading custom resources */
interface K8sResource {
  apiVersion?: string;
  kind?: string;
  metadata: {
    name: string;
    namespace?: string;
    [key: string]: unknown;
  };
  spec: Record<string, unknown>;
  status?: Record<string, unknown>;
}

// =============================================================================
// Types
// =============================================================================

export const FluxGitRepositorySchema = z.object({
  name: z.string(),
  namespace: z.string().default("flux-system"),
  url: z.string(),
  branch: z.string().default("main"),
  interval: z.string().default("1m"),
  secretRef: z.string().optional(),
  ignore: z.string().optional(),
  suspend: z.boolean().optional(),
});

export type FluxGitRepository = z.infer<typeof FluxGitRepositorySchema>;

export const FluxKustomizationSchema = z.object({
  name: z.string(),
  namespace: z.string().default("flux-system"),
  sourceRef: z.object({
    kind: z.enum(["GitRepository", "OCIRepository", "Bucket"]),
    name: z.string(),
    namespace: z.string().optional(),
  }),
  path: z.string().default("./"),
  interval: z.string().default("10m"),
  prune: z.boolean().default(true),
  targetNamespace: z.string().optional(),
  healthChecks: z
    .array(
      z.object({
        apiVersion: z.string(),
        kind: z.string(),
        name: z.string(),
        namespace: z.string(),
      })
    )
    .optional(),
  timeout: z.string().optional(),
  retryInterval: z.string().optional(),
  dependsOn: z
    .array(
      z.object({
        name: z.string(),
        namespace: z.string().optional(),
      })
    )
    .optional(),
  suspend: z.boolean().optional(),
});

export type FluxKustomization = z.infer<typeof FluxKustomizationSchema>;

export const FluxHelmReleaseSchema = z.object({
  name: z.string(),
  namespace: z.string().default("flux-system"),
  chart: z.object({
    spec: z.object({
      chart: z.string(),
      version: z.string().optional(),
      sourceRef: z.object({
        kind: z.enum(["HelmRepository", "GitRepository", "Bucket"]),
        name: z.string(),
        namespace: z.string().optional(),
      }),
      interval: z.string().optional(),
      valuesFiles: z.array(z.string()).optional(),
    }),
  }),
  interval: z.string().default("1m"),
  releaseName: z.string().optional(),
  targetNamespace: z.string().optional(),
  values: z.record(z.unknown()).optional(),
  valuesFrom: z
    .array(
      z.object({
        kind: z.enum(["ConfigMap", "Secret"]),
        name: z.string(),
        valuesKey: z.string().optional(),
        targetPath: z.string().optional(),
        optional: z.boolean().optional(),
      })
    )
    .optional(),
  suspend: z.boolean().optional(),
});

export type FluxHelmRelease = z.infer<typeof FluxHelmReleaseSchema>;

export const FluxResourceStatusSchema = z.object({
  conditions: z.array(
    z.object({
      type: z.string(),
      status: z.enum(["True", "False", "Unknown"]),
      reason: z.string().optional(),
      message: z.string().optional(),
      lastTransitionTime: z.string().optional(),
    })
  ),
  lastAppliedRevision: z.string().optional(),
  lastAttemptedRevision: z.string().optional(),
  observedGeneration: z.number().optional(),
});

export type FluxResourceStatus = z.infer<typeof FluxResourceStatusSchema>;

export interface FluxEvents {
  "flux:reconcile:started": (resource: string, kind: string) => void;
  "flux:reconcile:completed": (resource: string, kind: string) => void;
  "flux:reconcile:failed": (
    resource: string,
    kind: string,
    error: string
  ) => void;
  "flux:resource:created": (resource: any, kind: string) => void;
  "flux:resource:updated": (resource: any, kind: string) => void;
  "flux:resource:deleted": (name: string, kind: string) => void;
}

export interface FluxClientConfig {
  k8sClient: KubernetesClient;
  defaultNamespace?: string;
}

// =============================================================================
// Flux Client Implementation
// =============================================================================

export class FluxClient extends EventEmitter<FluxEvents> {
  private k8sClient: KubernetesClient;
  private defaultNamespace: string;

  constructor(config: FluxClientConfig) {
    super();
    this.k8sClient = config.k8sClient;
    this.defaultNamespace = config.defaultNamespace || "flux-system";
  }

  // ===========================================================================
  // GitRepository Operations
  // ===========================================================================

  /**
   * Create a GitRepository source
   */
  async createGitRepository(
    repo: FluxGitRepository
  ): Promise<FluxGitRepository> {
    const validated = FluxGitRepositorySchema.parse(repo);

    const resource = {
      apiVersion: "source.toolkit.fluxcd.io/v1",
      kind: "GitRepository",
      metadata: {
        name: validated.name,
        namespace: validated.namespace,
      },
      spec: {
        url: validated.url,
        ref: {
          branch: validated.branch,
        },
        interval: validated.interval,
        secretRef: validated.secretRef
          ? { name: validated.secretRef }
          : undefined,
        ignore: validated.ignore,
        suspend: validated.suspend,
      },
    };

    await this.k8sClient.applyResource(resource);
    this.emit("flux:resource:created", validated, "GitRepository");
    return validated;
  }

  /**
   * Get GitRepository
   */
  async getGitRepository(
    name: string,
    namespace?: string
  ): Promise<(FluxGitRepository & { status?: FluxResourceStatus }) | null> {
    const ns = namespace || this.defaultNamespace;

    const resource = (await this.k8sClient.getCustomResource(
      "source.toolkit.fluxcd.io/v1",
      "gitrepositories",
      ns,
      name
    )) as K8sResource | null;

    if (!resource) {
      return null;
    }

    const spec = resource.spec as Record<string, unknown>;
    const ref = spec.ref as { branch?: string } | undefined;
    const secretRef = spec.secretRef as { name?: string } | undefined;

    return {
      name: resource.metadata.name,
      namespace: resource.metadata.namespace,
      url: spec.url as string,
      branch: ref?.branch || "main",
      interval: spec.interval as string,
      secretRef: secretRef?.name,
      ignore: spec.ignore as string | undefined,
      suspend: spec.suspend as boolean | undefined,
      status: resource.status as FluxResourceStatus | undefined,
    };
  }

  /**
   * Update GitRepository
   */
  async updateGitRepository(
    repo: FluxGitRepository
  ): Promise<FluxGitRepository> {
    const validated = FluxGitRepositorySchema.parse(repo);

    const resource = {
      apiVersion: "source.toolkit.fluxcd.io/v1",
      kind: "GitRepository",
      metadata: {
        name: validated.name,
        namespace: validated.namespace,
      },
      spec: {
        url: validated.url,
        ref: {
          branch: validated.branch,
        },
        interval: validated.interval,
        secretRef: validated.secretRef
          ? { name: validated.secretRef }
          : undefined,
        ignore: validated.ignore,
        suspend: validated.suspend,
      },
    };

    await this.k8sClient.applyResource(resource);
    this.emit("flux:resource:updated", validated, "GitRepository");
    return validated;
  }

  /**
   * Delete GitRepository
   */
  async deleteGitRepository(name: string, namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.k8sClient.deleteCustomResource(
      "source.toolkit.fluxcd.io/v1",
      "gitrepositories",
      ns,
      name
    );
    this.emit("flux:resource:deleted", name, "GitRepository");
  }

  // ===========================================================================
  // Kustomization Operations
  // ===========================================================================

  /**
   * Create a Kustomization
   */
  async createKustomization(
    kustomization: FluxKustomization
  ): Promise<FluxKustomization> {
    const validated = FluxKustomizationSchema.parse(kustomization);

    const resource = {
      apiVersion: "kustomize.toolkit.fluxcd.io/v1",
      kind: "Kustomization",
      metadata: {
        name: validated.name,
        namespace: validated.namespace,
      },
      spec: {
        sourceRef: validated.sourceRef,
        path: validated.path,
        interval: validated.interval,
        prune: validated.prune,
        targetNamespace: validated.targetNamespace,
        healthChecks: validated.healthChecks,
        timeout: validated.timeout,
        retryInterval: validated.retryInterval,
        dependsOn: validated.dependsOn,
        suspend: validated.suspend,
      },
    };

    await this.k8sClient.applyResource(resource);
    this.emit("flux:resource:created", validated, "Kustomization");
    return validated;
  }

  /**
   * Get Kustomization
   */
  async getKustomization(
    name: string,
    namespace?: string
  ): Promise<(FluxKustomization & { status?: FluxResourceStatus }) | null> {
    const ns = namespace || this.defaultNamespace;

    const resource = (await this.k8sClient.getCustomResource(
      "kustomize.toolkit.fluxcd.io/v1",
      "kustomizations",
      ns,
      name
    )) as K8sResource | null;

    if (!resource) {
      return null;
    }

    const spec = resource.spec;

    return {
      name: resource.metadata.name,
      namespace: resource.metadata.namespace,
      sourceRef: spec.sourceRef as FluxKustomization["sourceRef"],
      path: (spec.path as string) || "./",
      interval: spec.interval as string,
      prune: spec.prune as boolean,
      targetNamespace: spec.targetNamespace as string | undefined,
      healthChecks: spec.healthChecks as FluxKustomization["healthChecks"],
      timeout: spec.timeout as string | undefined,
      retryInterval: spec.retryInterval as string | undefined,
      dependsOn: spec.dependsOn as FluxKustomization["dependsOn"],
      suspend: spec.suspend as boolean | undefined,
      status: resource.status as FluxResourceStatus | undefined,
    };
  }

  /**
   * Update Kustomization
   */
  async updateKustomization(
    kustomization: FluxKustomization
  ): Promise<FluxKustomization> {
    const validated = FluxKustomizationSchema.parse(kustomization);

    const resource = {
      apiVersion: "kustomize.toolkit.fluxcd.io/v1",
      kind: "Kustomization",
      metadata: {
        name: validated.name,
        namespace: validated.namespace,
      },
      spec: {
        sourceRef: validated.sourceRef,
        path: validated.path,
        interval: validated.interval,
        prune: validated.prune,
        targetNamespace: validated.targetNamespace,
        healthChecks: validated.healthChecks,
        timeout: validated.timeout,
        retryInterval: validated.retryInterval,
        dependsOn: validated.dependsOn,
        suspend: validated.suspend,
      },
    };

    await this.k8sClient.applyResource(resource);
    this.emit("flux:resource:updated", validated, "Kustomization");
    return validated;
  }

  /**
   * Delete Kustomization
   */
  async deleteKustomization(name: string, namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.k8sClient.deleteCustomResource(
      "kustomize.toolkit.fluxcd.io/v1",
      "kustomizations",
      ns,
      name
    );
    this.emit("flux:resource:deleted", name, "Kustomization");
  }

  // ===========================================================================
  // HelmRelease Operations
  // ===========================================================================

  /**
   * Create a HelmRelease
   */
  async createHelmRelease(release: FluxHelmRelease): Promise<FluxHelmRelease> {
    const validated = FluxHelmReleaseSchema.parse(release);

    const resource = {
      apiVersion: "helm.toolkit.fluxcd.io/v2beta1",
      kind: "HelmRelease",
      metadata: {
        name: validated.name,
        namespace: validated.namespace,
      },
      spec: {
        chart: validated.chart,
        interval: validated.interval,
        releaseName: validated.releaseName,
        targetNamespace: validated.targetNamespace,
        values: validated.values,
        valuesFrom: validated.valuesFrom,
        suspend: validated.suspend,
      },
    };

    await this.k8sClient.applyResource(resource);
    this.emit("flux:resource:created", validated, "HelmRelease");
    return validated;
  }

  /**
   * Get HelmRelease
   */
  async getHelmRelease(
    name: string,
    namespace?: string
  ): Promise<(FluxHelmRelease & { status?: FluxResourceStatus }) | null> {
    const ns = namespace || this.defaultNamespace;

    const resource = (await this.k8sClient.getCustomResource(
      "helm.toolkit.fluxcd.io/v2beta1",
      "helmreleases",
      ns,
      name
    )) as K8sResource | null;

    if (!resource) {
      return null;
    }

    const spec = resource.spec;

    return {
      name: resource.metadata.name,
      namespace: resource.metadata.namespace,
      chart: spec.chart as FluxHelmRelease["chart"],
      interval: spec.interval as string,
      releaseName: spec.releaseName as string | undefined,
      targetNamespace: spec.targetNamespace as string | undefined,
      values: spec.values as Record<string, unknown> | undefined,
      valuesFrom: spec.valuesFrom as FluxHelmRelease["valuesFrom"],
      suspend: spec.suspend as boolean | undefined,
      status: resource.status as FluxResourceStatus | undefined,
    };
  }

  // ===========================================================================
  // Reconciliation Operations
  // ===========================================================================

  /**
   * Trigger reconciliation of a Flux resource
   */
  async reconcile(
    kind: "GitRepository" | "Kustomization" | "HelmRelease",
    name: string,
    namespace?: string
  ): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    this.emit("flux:reconcile:started", name, kind);

    try {
      // Trigger reconciliation by updating annotation
      const annotation = {
        "reconcile.fluxcd.io/requestedAt": new Date().toISOString(),
      };

      let apiVersion: string;
      let plural: string;

      switch (kind) {
        case "GitRepository":
          apiVersion = "source.toolkit.fluxcd.io/v1";
          plural = "gitrepositories";
          break;
        case "Kustomization":
          apiVersion = "kustomize.toolkit.fluxcd.io/v1";
          plural = "kustomizations";
          break;
        case "HelmRelease":
          apiVersion = "helm.toolkit.fluxcd.io/v2beta1";
          plural = "helmreleases";
          break;
      }

      await this.k8sClient.patchCustomResource(apiVersion, plural, ns, name, {
        metadata: {
          annotations: annotation,
        },
      });

      this.emit("flux:reconcile:completed", name, kind);
    } catch (error) {
      this.emit(
        "flux:reconcile:failed",
        name,
        kind,
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  /**
   * Wait for reconciliation to complete
   */
  async waitForReconciliation(
    kind: "GitRepository" | "Kustomization" | "HelmRelease",
    name: string,
    namespace?: string,
    timeoutMs: number = 300000
  ): Promise<FluxResourceStatus> {
    const ns = namespace || this.defaultNamespace;
    const start = Date.now();

    let apiVersion: string;
    let plural: string;

    switch (kind) {
      case "GitRepository":
        apiVersion = "source.toolkit.fluxcd.io/v1";
        plural = "gitrepositories";
        break;
      case "Kustomization":
        apiVersion = "kustomize.toolkit.fluxcd.io/v1";
        plural = "kustomizations";
        break;
      case "HelmRelease":
        apiVersion = "helm.toolkit.fluxcd.io/v2beta1";
        plural = "helmreleases";
        break;
    }

    while (Date.now() - start < timeoutMs) {
      const resource = (await this.k8sClient.getCustomResource(
        apiVersion,
        plural,
        ns,
        name
      )) as K8sResource | null;

      if (!resource?.status) {
        await this.sleep(5000);
        continue;
      }

      const status = resource.status as {
        conditions?: Array<{ type: string; status: string; message?: string }>;
      };
      const readyCondition = status.conditions?.find((c) => c.type === "Ready");

      if (readyCondition?.status === "True") {
        return resource.status as FluxResourceStatus;
      }

      if (readyCondition?.status === "False") {
        throw new Error(
          `Flux reconciliation failed: ${readyCondition.message || "Unknown error"}`
        );
      }

      await this.sleep(5000);
    }

    throw new Error(`Timeout waiting for Flux reconciliation: ${name}`);
  }

  /**
   * Suspend a Flux resource
   */
  async suspend(
    kind: "GitRepository" | "Kustomization" | "HelmRelease",
    name: string,
    namespace?: string
  ): Promise<void> {
    const ns = namespace || this.defaultNamespace;

    let apiVersion: string;
    let plural: string;

    switch (kind) {
      case "GitRepository":
        apiVersion = "source.toolkit.fluxcd.io/v1";
        plural = "gitrepositories";
        break;
      case "Kustomization":
        apiVersion = "kustomize.toolkit.fluxcd.io/v1";
        plural = "kustomizations";
        break;
      case "HelmRelease":
        apiVersion = "helm.toolkit.fluxcd.io/v2beta1";
        plural = "helmreleases";
        break;
    }

    await this.k8sClient.patchCustomResource(apiVersion, plural, ns, name, {
      spec: { suspend: true },
    });
  }

  /**
   * Resume a suspended Flux resource
   */
  async resume(
    kind: "GitRepository" | "Kustomization" | "HelmRelease",
    name: string,
    namespace?: string
  ): Promise<void> {
    const ns = namespace || this.defaultNamespace;

    let apiVersion: string;
    let plural: string;

    switch (kind) {
      case "GitRepository":
        apiVersion = "source.toolkit.fluxcd.io/v1";
        plural = "gitrepositories";
        break;
      case "Kustomization":
        apiVersion = "kustomize.toolkit.fluxcd.io/v1";
        plural = "kustomizations";
        break;
      case "HelmRelease":
        apiVersion = "helm.toolkit.fluxcd.io/v2beta1";
        plural = "helmreleases";
        break;
    }

    await this.k8sClient.patchCustomResource(apiVersion, plural, ns, name, {
      spec: { suspend: false },
    });
  }

  /**
   * Get status of all Flux resources in a namespace
   */
  async getFluxStatus(namespace?: string): Promise<{
    gitRepositories: Array<{ name: string; ready: boolean; message?: string }>;
    kustomizations: Array<{ name: string; ready: boolean; message?: string }>;
    helmReleases: Array<{ name: string; ready: boolean; message?: string }>;
  }> {
    const ns = namespace || this.defaultNamespace;

    const [gitRepos, kustomizations, helmReleases] = await Promise.all([
      this.k8sClient.listCustomResources(
        "source.toolkit.fluxcd.io/v1",
        "gitrepositories",
        ns
      ),
      this.k8sClient.listCustomResources(
        "kustomize.toolkit.fluxcd.io/v1",
        "kustomizations",
        ns
      ),
      this.k8sClient.listCustomResources(
        "helm.toolkit.fluxcd.io/v2beta1",
        "helmreleases",
        ns
      ),
    ]);

    const extractStatus = (items: any[]) =>
      items.map((item) => {
        const readyCondition = item.status?.conditions?.find(
          (c: any) => c.type === "Ready"
        );
        return {
          name: item.metadata.name,
          ready: readyCondition?.status === "True",
          message: readyCondition?.message,
        };
      });

    return {
      gitRepositories: extractStatus(gitRepos),
      kustomizations: extractStatus(kustomizations),
      helmReleases: extractStatus(helmReleases),
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createFluxClient(config: FluxClientConfig): FluxClient {
  return new FluxClient(config);
}
