/**
 * Kubernetes Client Wrapper
 * Provides high-level abstractions for Kubernetes API operations
 */

import * as k8s from "@kubernetes/client-node";
import { EventEmitter } from "eventemitter3";
import type {
  DeploymentResource,
  ServiceResource,
  KubernetesResource,
  ContainerSpec,
  PodStatus,
  ResourceConfig,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface KubernetesClientConfig {
  /** Kubeconfig path or use in-cluster config */
  kubeconfig?: string;
  /** Context to use */
  context?: string;
  /** Namespace default */
  defaultNamespace?: string;
}

export interface KubernetesClientEvents {
  "pod:added": k8s.V1Pod;
  "pod:modified": k8s.V1Pod;
  "pod:deleted": k8s.V1Pod;
  "deployment:updated": k8s.V1Deployment;
  error: Error;
}

export interface ScaleOptions {
  replicas: number;
  namespace: string;
  name: string;
}

export interface PatchOptions {
  namespace: string;
  name: string;
  patch: Record<string, unknown>;
}

// =============================================================================
// Kubernetes Client
// =============================================================================

export class KubernetesClient extends EventEmitter<KubernetesClientEvents> {
  private kc: k8s.KubeConfig;
  private coreApi: k8s.CoreV1Api;
  private appsApi: k8s.AppsV1Api;
  private networkingApi: k8s.NetworkingV1Api;
  private customApi: k8s.CustomObjectsApi;
  private defaultNamespace: string;

  constructor(config: KubernetesClientConfig = {}) {
    super();
    this.kc = new k8s.KubeConfig();

    if (config.kubeconfig) {
      this.kc.loadFromFile(config.kubeconfig);
    } else {
      try {
        this.kc.loadFromCluster();
      } catch {
        this.kc.loadFromDefault();
      }
    }

    if (config.context) {
      this.kc.setCurrentContext(config.context);
    }

    this.coreApi = this.kc.makeApiClient(k8s.CoreV1Api);
    this.appsApi = this.kc.makeApiClient(k8s.AppsV1Api);
    this.networkingApi = this.kc.makeApiClient(k8s.NetworkingV1Api);
    this.customApi = this.kc.makeApiClient(k8s.CustomObjectsApi);
    this.defaultNamespace = config.defaultNamespace || "default";
  }

  // ===========================================================================
  // Deployment Operations
  // ===========================================================================

  /**
   * Get a deployment by name
   */
  async getDeployment(
    name: string,
    namespace?: string
  ): Promise<k8s.V1Deployment | null> {
    try {
      const response = await this.appsApi.readNamespacedDeployment(
        name,
        namespace || this.defaultNamespace
      );
      return response.body;
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Create a new deployment
   */
  async createDeployment(
    deployment: DeploymentResource,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns =
      namespace || deployment.metadata.namespace || this.defaultNamespace;
    const response = await this.appsApi.createNamespacedDeployment(
      ns,
      deployment as unknown as k8s.V1Deployment
    );
    return response.body;
  }

  /**
   * Update an existing deployment
   */
  async updateDeployment(
    name: string,
    deployment: Partial<k8s.V1Deployment>,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.appsApi.patchNamespacedDeployment(
      name,
      ns,
      deployment,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      { headers: { "Content-Type": "application/strategic-merge-patch+json" } }
    );
    return response.body;
  }

  /**
   * Alias for updateDeployment - patches an existing deployment
   */
  async patchDeployment(
    name: string,
    namespace: string,
    patch: Partial<k8s.V1Deployment>
  ): Promise<k8s.V1Deployment> {
    return this.updateDeployment(name, patch, namespace);
  }

  /**
   * Delete a deployment
   */
  async deleteDeployment(name: string, namespace?: string): Promise<void> {
    await this.appsApi.deleteNamespacedDeployment(
      name,
      namespace || this.defaultNamespace
    );
  }

  /**
   * Scale a deployment
   */
  async scaleDeployment(options: ScaleOptions): Promise<k8s.V1Scale> {
    const response = await this.appsApi.patchNamespacedDeploymentScale(
      options.name,
      options.namespace || this.defaultNamespace,
      { spec: { replicas: options.replicas } },
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      { headers: { "Content-Type": "application/strategic-merge-patch+json" } }
    );
    return response.body;
  }

  /**
   * Update deployment image
   */
  async updateDeploymentImage(
    name: string,
    containerName: string,
    image: string,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const patch = {
      spec: {
        template: {
          spec: {
            containers: [{ name: containerName, image }],
          },
        },
      },
    };

    return this.updateDeployment(name, patch, namespace);
  }

  /**
   * Get deployment status
   */
  async getDeploymentStatus(
    name: string,
    namespace?: string
  ): Promise<{
    available: boolean;
    ready: number;
    desired: number;
    updated: number;
    conditions: k8s.V1DeploymentCondition[];
  }> {
    const deployment = await this.getDeployment(name, namespace);
    if (!deployment || !deployment.status) {
      throw new Error(`Deployment ${name} not found`);
    }

    const status = deployment.status;
    return {
      available: status.availableReplicas === status.replicas,
      ready: status.readyReplicas || 0,
      desired: status.replicas || 0,
      updated: status.updatedReplicas || 0,
      conditions: status.conditions || [],
    };
  }

  /**
   * Wait for deployment to be ready
   */
  async waitForDeploymentReady(
    name: string,
    namespace?: string,
    timeoutMs: number = 300000
  ): Promise<boolean> {
    const start = Date.now();
    const ns = namespace || this.defaultNamespace;

    while (Date.now() - start < timeoutMs) {
      const status = await this.getDeploymentStatus(name, ns);

      if (
        status.available &&
        status.ready === status.desired &&
        status.updated === status.desired
      ) {
        return true;
      }

      await this.sleep(2000);
    }

    return false;
  }

  // ===========================================================================
  // Pod Operations
  // ===========================================================================

  /**
   * List pods by label selector
   */
  async listPods(
    namespace?: string,
    labelSelector?: string
  ): Promise<k8s.V1Pod[]> {
    const response = await this.coreApi.listNamespacedPod(
      namespace || this.defaultNamespace,
      undefined,
      undefined,
      undefined,
      undefined,
      labelSelector
    );
    return response.body.items;
  }

  /**
   * Get pods for a deployment
   */
  async getDeploymentPods(
    deploymentName: string,
    namespace?: string
  ): Promise<k8s.V1Pod[]> {
    const deployment = await this.getDeployment(deploymentName, namespace);
    if (!deployment || !deployment.spec?.selector?.matchLabels) {
      return [];
    }

    const labelSelector = Object.entries(deployment.spec.selector.matchLabels)
      .map(([k, v]) => `${k}=${v}`)
      .join(",");

    return this.listPods(namespace, labelSelector);
  }

  /**
   * Get pod status summary
   */
  async getPodStatuses(
    deploymentName: string,
    namespace?: string
  ): Promise<PodStatus[]> {
    const pods = await this.getDeploymentPods(deploymentName, namespace);

    return pods.map((pod) => {
      const containerStatuses = pod.status?.containerStatuses || [];
      const restarts = containerStatuses.reduce(
        (sum, c) => sum + (c.restartCount || 0),
        0
      );
      const ready = containerStatuses.every((c) => c.ready);
      const creationTime = pod.metadata?.creationTimestamp;
      const age = creationTime
        ? this.formatDuration(Date.now() - new Date(creationTime).getTime())
        : "unknown";

      // Get version from pod labels
      const version =
        pod.metadata?.labels?.["app.kubernetes.io/version"] ||
        pod.metadata?.labels?.version ||
        "unknown";

      return {
        name: pod.metadata?.name || "unknown",
        phase: pod.status?.phase || "Unknown",
        ready,
        restarts,
        age,
        version,
        node: pod.spec?.nodeName,
      };
    });
  }

  /**
   * Delete a pod (useful for forcing restart)
   */
  async deletePod(name: string, namespace?: string): Promise<void> {
    await this.coreApi.deleteNamespacedPod(
      name,
      namespace || this.defaultNamespace
    );
  }

  // ===========================================================================
  // Service Operations
  // ===========================================================================

  /**
   * Get a service
   */
  async getService(
    name: string,
    namespace?: string
  ): Promise<k8s.V1Service | null> {
    try {
      const response = await this.coreApi.readNamespacedService(
        name,
        namespace || this.defaultNamespace
      );
      return response.body;
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Create a service
   */
  async createService(
    service: ServiceResource,
    namespace?: string
  ): Promise<k8s.V1Service> {
    const ns = namespace || service.metadata.namespace || this.defaultNamespace;
    const response = await this.coreApi.createNamespacedService(
      ns,
      service as unknown as k8s.V1Service
    );
    return response.body;
  }

  /**
   * Update a service
   */
  async updateService(
    name: string,
    patch: Partial<k8s.V1Service>,
    namespace?: string
  ): Promise<k8s.V1Service> {
    const response = await this.coreApi.patchNamespacedService(
      name,
      namespace || this.defaultNamespace,
      patch,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      { headers: { "Content-Type": "application/strategic-merge-patch+json" } }
    );
    return response.body;
  }

  /**
   * Alias for updateService - patches an existing service
   */
  async patchService(
    name: string,
    namespace: string,
    patch: Partial<k8s.V1Service>
  ): Promise<k8s.V1Service> {
    return this.updateService(name, patch, namespace);
  }

  /**
   * Update service selector to point to different deployment
   */
  async switchServiceSelector(
    serviceName: string,
    selector: Record<string, string>,
    namespace?: string
  ): Promise<k8s.V1Service> {
    return this.updateService(serviceName, { spec: { selector } }, namespace);
  }

  /**
   * Delete a service
   */
  async deleteService(name: string, namespace?: string): Promise<void> {
    await this.coreApi.deleteNamespacedService(
      name,
      namespace || this.defaultNamespace
    );
  }

  // ===========================================================================
  // Ingress Operations
  // ===========================================================================

  /**
   * Get an ingress
   */
  async getIngress(
    name: string,
    namespace?: string
  ): Promise<k8s.V1Ingress | null> {
    try {
      const response = await this.networkingApi.readNamespacedIngress(
        name,
        namespace || this.defaultNamespace
      );
      return response.body;
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Update ingress annotations (useful for traffic splitting)
   */
  async updateIngressAnnotations(
    name: string,
    annotations: Record<string, string>,
    namespace?: string
  ): Promise<k8s.V1Ingress> {
    const response = await this.networkingApi.patchNamespacedIngress(
      name,
      namespace || this.defaultNamespace,
      { metadata: { annotations } },
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      { headers: { "Content-Type": "application/strategic-merge-patch+json" } }
    );
    return response.body;
  }

  // ===========================================================================
  // ConfigMap & Secret Operations
  // ===========================================================================

  /**
   * Get a ConfigMap
   */
  async getConfigMap(
    name: string,
    namespace?: string
  ): Promise<k8s.V1ConfigMap | null> {
    try {
      const response = await this.coreApi.readNamespacedConfigMap(
        name,
        namespace || this.defaultNamespace
      );
      return response.body;
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Create or update a ConfigMap
   */
  async upsertConfigMap(
    name: string,
    data: Record<string, string>,
    namespace?: string
  ): Promise<k8s.V1ConfigMap> {
    const ns = namespace || this.defaultNamespace;
    const existing = await this.getConfigMap(name, ns);

    if (existing) {
      const response = await this.coreApi.patchNamespacedConfigMap(
        name,
        ns,
        { data },
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {
          headers: { "Content-Type": "application/strategic-merge-patch+json" },
        }
      );
      return response.body;
    }

    const response = await this.coreApi.createNamespacedConfigMap(ns, {
      apiVersion: "v1",
      kind: "ConfigMap",
      metadata: { name },
      data,
    });
    return response.body;
  }

  // ===========================================================================
  // ReplicaSet Operations
  // ===========================================================================

  /**
   * Get ReplicaSets for a deployment
   */
  async getDeploymentReplicaSets(
    deploymentName: string,
    namespace?: string
  ): Promise<k8s.V1ReplicaSet[]> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.appsApi.listNamespacedReplicaSet(ns);

    return response.body.items.filter((rs) => {
      const ownerRefs = rs.metadata?.ownerReferences || [];
      return ownerRefs.some(
        (ref) => ref.kind === "Deployment" && ref.name === deploymentName
      );
    });
  }

  /**
   * Get deployment revision history
   */
  async getDeploymentRevisions(
    deploymentName: string,
    namespace?: string
  ): Promise<
    { revision: number; replicaSet: k8s.V1ReplicaSet; createdAt: Date }[]
  > {
    const replicaSets = await this.getDeploymentReplicaSets(
      deploymentName,
      namespace
    );

    return replicaSets
      .map((rs) => ({
        revision: parseInt(
          rs.metadata?.annotations?.["deployment.kubernetes.io/revision"] ||
            "0",
          10
        ),
        replicaSet: rs,
        createdAt: new Date(rs.metadata?.creationTimestamp || 0),
      }))
      .sort((a, b) => b.revision - a.revision);
  }

  // ===========================================================================
  // Custom Resource Operations
  // ===========================================================================

  /**
   * Get a custom resource
   * @overload (group, version, namespace, plural, name) - 5 args
   * @overload (apiVersion, plural, namespace, name) - 4 args (apiVersion like "source.toolkit.fluxcd.io/v1")
   */
  async getCustomResource(
    groupOrApiVersion: string,
    versionOrPlural: string,
    namespace: string,
    pluralOrName: string,
    name?: string
  ): Promise<Record<string, unknown> | null> {
    let group: string;
    let version: string;
    let plural: string;
    let resourceName: string;

    if (name !== undefined) {
      // 5 argument form: (group, version, namespace, plural, name)
      group = groupOrApiVersion;
      version = versionOrPlural;
      plural = pluralOrName;
      resourceName = name;
    } else {
      // 4 argument form: (apiVersion, plural, namespace, name)
      const apiVersion = groupOrApiVersion;
      plural = versionOrPlural;
      resourceName = pluralOrName;

      // Parse apiVersion into group and version
      const parts = apiVersion.split("/");
      if (parts.length === 2) {
        group = parts[0];
        version = parts[1];
      } else {
        group = "";
        version = parts[0];
      }
    }

    try {
      const response = await this.customApi.getNamespacedCustomObject(
        group,
        version,
        namespace,
        plural,
        resourceName
      );
      return response.body as Record<string, unknown>;
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Create or update a custom resource
   */
  async applyCustomResource(
    group: string,
    version: string,
    namespace: string,
    plural: string,
    resource: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const name = (resource.metadata as { name?: string })?.name;
    if (!name) {
      throw new Error("Resource must have metadata.name");
    }

    const existing = await this.getCustomResource(
      group,
      version,
      namespace,
      plural,
      name
    );

    if (existing) {
      const response = await this.customApi.patchNamespacedCustomObject(
        group,
        version,
        namespace,
        plural,
        name,
        resource,
        undefined,
        undefined,
        undefined,
        { headers: { "Content-Type": "application/merge-patch+json" } }
      );
      return response.body as Record<string, unknown>;
    }

    const response = await this.customApi.createNamespacedCustomObject(
      group,
      version,
      namespace,
      plural,
      resource
    );
    return response.body as Record<string, unknown>;
  }

  /**
   * Apply a resource from its full specification (extracts group/version/kind from apiVersion)
   */
  async applyResource(
    resource: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiVersion = resource.apiVersion as string;
    const kind = resource.kind as string;
    const metadata = resource.metadata as { name?: string; namespace?: string };

    if (!apiVersion || !kind || !metadata?.name) {
      throw new Error("Resource must have apiVersion, kind, and metadata.name");
    }

    // Parse apiVersion into group and version
    const [groupOrVersion, version] = apiVersion.includes("/")
      ? apiVersion.split("/")
      : ["", apiVersion];

    const group = version ? groupOrVersion : "";
    const actualVersion = version || groupOrVersion;
    const namespace = metadata.namespace || this.defaultNamespace;
    const plural = this.kindToPlural(kind);

    return this.applyCustomResource(
      group,
      actualVersion,
      namespace,
      plural,
      resource
    );
  }

  /**
   * Delete a custom resource
   */
  /**
   * Delete a custom resource
   * @overload (group, version, namespace, plural, name) - 5 args
   * @overload (apiVersion, plural, namespace, name) - 4 args
   */
  async deleteCustomResource(
    groupOrApiVersion: string,
    versionOrPlural: string,
    namespace: string,
    pluralOrName: string,
    name?: string
  ): Promise<void> {
    let group: string;
    let version: string;
    let plural: string;
    let resourceName: string;

    if (name !== undefined) {
      // 5 argument form
      group = groupOrApiVersion;
      version = versionOrPlural;
      plural = pluralOrName;
      resourceName = name;
    } else {
      // 4 argument form
      const apiVersion = groupOrApiVersion;
      plural = versionOrPlural;
      resourceName = pluralOrName;

      const parts = apiVersion.split("/");
      if (parts.length === 2) {
        group = parts[0];
        version = parts[1];
      } else {
        group = "";
        version = parts[0];
      }
    }

    try {
      await this.customApi.deleteNamespacedCustomObject(
        group,
        version,
        namespace,
        plural,
        resourceName
      );
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        // Already deleted, ignore
        return;
      }
      throw error;
    }
  }

  /**
   * Patch a custom resource
   * @overload (apiVersion, plural, namespace, name, patch) - 5 args
   */
  async patchCustomResource(
    apiVersion: string,
    plural: string,
    namespace: string,
    name: string,
    patch: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const parts = apiVersion.split("/");
    const group = parts.length === 2 ? parts[0] : "";
    const version = parts.length === 2 ? parts[1] : parts[0];

    const response = await this.customApi.patchNamespacedCustomObject(
      group,
      version,
      namespace,
      plural,
      name,
      patch,
      undefined,
      undefined,
      undefined,
      { headers: { "Content-Type": "application/merge-patch+json" } }
    );
    return response.body as Record<string, unknown>;
  }

  /**
   * Convert kind to plural (simple pluralization)
   */
  private kindToPlural(kind: string): string {
    const lowerKind = kind.toLowerCase();
    // Common irregular plurals
    const irregulars: Record<string, string> = {
      ingress: "ingresses",
      networkpolicy: "networkpolicies",
    };
    if (irregulars[lowerKind]) {
      return irregulars[lowerKind];
    }
    // Standard pluralization
    if (lowerKind.endsWith("y")) {
      return lowerKind.slice(0, -1) + "ies";
    }
    if (
      lowerKind.endsWith("s") ||
      lowerKind.endsWith("x") ||
      lowerKind.endsWith("ch") ||
      lowerKind.endsWith("sh")
    ) {
      return lowerKind + "es";
    }
    return lowerKind + "s";
  }

  // ===========================================================================
  // Namespace Operations
  // ===========================================================================

  /**
   * Ensure namespace exists
   */
  async ensureNamespace(name: string): Promise<void> {
    try {
      await this.coreApi.readNamespace(name);
    } catch (error: unknown) {
      if (
        (error as { response?: { statusCode?: number } }).response
          ?.statusCode === 404
      ) {
        await this.coreApi.createNamespace({
          apiVersion: "v1",
          kind: "Namespace",
          metadata: { name },
        });
      } else {
        throw error;
      }
    }
  }

  // ===========================================================================
  // Watch Operations
  // ===========================================================================

  /**
   * Watch pods in a namespace
   */
  watchPods(namespace: string, labelSelector?: string): { stop: () => void } {
    const watch = new k8s.Watch(this.kc);
    let stopped = false;

    const path = `/api/v1/namespaces/${namespace}/pods`;
    const queryParams: Record<string, string> = {};
    if (labelSelector) {
      queryParams.labelSelector = labelSelector;
    }

    watch.watch(
      path,
      queryParams,
      (type, obj) => {
        if (stopped) return;
        const pod = obj as k8s.V1Pod;
        switch (type) {
          case "ADDED":
            this.emit("pod:added", pod);
            break;
          case "MODIFIED":
            this.emit("pod:modified", pod);
            break;
          case "DELETED":
            this.emit("pod:deleted", pod);
            break;
        }
      },
      (err) => {
        if (!stopped && err) {
          this.emit("error", err);
        }
      }
    );

    return {
      stop: () => {
        stopped = true;
      },
    };
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  /**
   * Build deployment resource from config
   */
  buildDeploymentResource(config: {
    name: string;
    namespace: string;
    image: string;
    replicas: number;
    containerName?: string;
    port?: number;
    resources?: ResourceConfig;
    labels?: Record<string, string>;
    annotations?: Record<string, string>;
    env?: { name: string; value: string }[];
  }): DeploymentResource {
    const labels = {
      "app.kubernetes.io/name": config.name,
      ...config.labels,
    };

    const containers: ContainerSpec[] = [
      {
        name: config.containerName || config.name,
        image: config.image,
        ports: config.port ? [{ containerPort: config.port }] : undefined,
        resources: config.resources,
        env: config.env,
      },
    ];

    return {
      apiVersion: "apps/v1",
      kind: "Deployment",
      metadata: {
        name: config.name,
        namespace: config.namespace,
        labels,
        annotations: config.annotations,
      },
      spec: {
        replicas: config.replicas,
        selector: {
          matchLabels: labels,
        },
        template: {
          metadata: {
            labels,
          },
          spec: {
            containers,
          },
        },
      },
    };
  }

  /**
   * Build service resource
   */
  buildServiceResource(config: {
    name: string;
    namespace: string;
    port: number;
    targetPort: number;
    selector: Record<string, string>;
    type?: "ClusterIP" | "NodePort" | "LoadBalancer";
  }): ServiceResource {
    return {
      apiVersion: "v1",
      kind: "Service",
      metadata: {
        name: config.name,
        namespace: config.namespace,
      },
      spec: {
        type: config.type || "ClusterIP",
        selector: config.selector,
        ports: [
          {
            port: config.port,
            targetPort: config.targetPort,
          },
        ],
      },
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d`;
    if (hours > 0) return `${hours}h`;
    if (minutes > 0) return `${minutes}m`;
    return `${seconds}s`;
  }
}

export function createKubernetesClient(
  config?: KubernetesClientConfig
): KubernetesClient {
  return new KubernetesClient(config);
}
