/**
 * NEURECTOMY Kubernetes Client
 *
 * @FLUX @APEX - DevOps + Production Code
 *
 * Production-grade Kubernetes API client with watch support,
 * automatic reconnection, and comprehensive resource management.
 */

import * as k8s from "@kubernetes/client-node";
import { EventEmitter } from "eventemitter3";
import pino from "pino";
import { v4 as uuidv4 } from "uuid";

import type {
  K8sConfig,
  K8sPodConfig,
  K8sDeploymentConfig,
  K8sServiceConfig,
  K8sHPAConfig,
  K8sContainerSpec,
  K8sResourceRequirements,
  K8sProbe,
  K8sVolumeMount,
  K8sVolume,
  K8sEnvVar,
  K8sSecurityContext,
  K8sPodSecurityContext,
  K8sLabelSelector,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface K8sClientEvents {
  "pod:added": (pod: k8s.V1Pod) => void;
  "pod:modified": (pod: k8s.V1Pod) => void;
  "pod:deleted": (pod: k8s.V1Pod) => void;
  "deployment:added": (deployment: k8s.V1Deployment) => void;
  "deployment:modified": (deployment: k8s.V1Deployment) => void;
  "deployment:deleted": (deployment: k8s.V1Deployment) => void;
  "service:added": (service: k8s.V1Service) => void;
  "service:modified": (service: k8s.V1Service) => void;
  "service:deleted": (service: k8s.V1Service) => void;
  "node:added": (node: k8s.V1Node) => void;
  "node:modified": (node: k8s.V1Node) => void;
  "node:deleted": (node: k8s.V1Node) => void;
  event: (event: k8s.CoreV1Event) => void;
  error: (error: Error) => void;
  connected: () => void;
  disconnected: () => void;
}

export interface WatchOptions {
  namespace?: string;
  labelSelector?: string;
  fieldSelector?: string;
  resourceVersion?: string;
}

export interface PodMetrics {
  name: string;
  namespace: string;
  containers: ContainerMetrics[];
  timestamp: Date;
}

export interface ContainerMetrics {
  name: string;
  cpuUsage: string;
  memoryUsage: string;
}

export interface NodeMetrics {
  name: string;
  cpuUsage: string;
  memoryUsage: string;
  timestamp: Date;
}

// =============================================================================
// Kubernetes Client Implementation
// =============================================================================

export class KubernetesClient extends EventEmitter<K8sClientEvents> {
  private kc: k8s.KubeConfig;
  private coreApi: k8s.CoreV1Api;
  private appsApi: k8s.AppsV1Api;
  private autoscalingApi: k8s.AutoscalingV2Api;
  private batchApi: k8s.BatchV1Api;
  private networkingApi: k8s.NetworkingV1Api;
  private rbacApi: k8s.RbacAuthorizationV1Api;
  private customObjectsApi: k8s.CustomObjectsApi;
  private watch: k8s.Watch;
  private metricsClient: k8s.Metrics;
  private logger: pino.Logger;
  private defaultNamespace: string;
  private watchers: Map<string, AbortController> = new Map();
  private isConnected: boolean = false;

  constructor(config?: K8sConfig) {
    super();

    this.logger = pino({
      name: "kubernetes-client",
      level: process.env.LOG_LEVEL || "info",
    });

    this.kc = new k8s.KubeConfig();

    if (config?.inCluster) {
      this.kc.loadFromCluster();
    } else if (config?.kubeconfig) {
      this.kc.loadFromFile(config.kubeconfig);
    } else {
      this.kc.loadFromDefault();
    }

    if (config?.context) {
      this.kc.setCurrentContext(config.context);
    }

    this.defaultNamespace = config?.namespace || "default";

    // Initialize API clients
    this.coreApi = this.kc.makeApiClient(k8s.CoreV1Api);
    this.appsApi = this.kc.makeApiClient(k8s.AppsV1Api);
    this.autoscalingApi = this.kc.makeApiClient(k8s.AutoscalingV2Api);
    this.batchApi = this.kc.makeApiClient(k8s.BatchV1Api);
    this.networkingApi = this.kc.makeApiClient(k8s.NetworkingV1Api);
    this.rbacApi = this.kc.makeApiClient(k8s.RbacAuthorizationV1Api);
    this.customObjectsApi = this.kc.makeApiClient(k8s.CustomObjectsApi);
    this.watch = new k8s.Watch(this.kc);
    this.metricsClient = new k8s.Metrics(this.kc);
  }

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  async connect(): Promise<void> {
    try {
      // Use listNamespace to verify connectivity
      await this.coreApi.listNamespace();
      this.isConnected = true;
      this.logger.info("Connected to Kubernetes cluster");
      this.emit("connected");
    } catch (error) {
      this.logger.error({ error }, "Failed to connect to Kubernetes cluster");
      throw new K8sConnectionError(
        "Failed to connect to Kubernetes cluster",
        error as Error
      );
    }
  }

  async disconnect(): Promise<void> {
    // Stop all watchers
    for (const [key, controller] of this.watchers) {
      controller.abort();
      this.watchers.delete(key);
    }

    this.isConnected = false;
    this.emit("disconnected");
    this.logger.info("Disconnected from Kubernetes cluster");
  }

  async getClusterInfo(): Promise<{
    serverVersion: string;
    platform: string;
    nodes: number;
  }> {
    const versionApi = this.kc.makeApiClient(k8s.VersionApi);
    const [versionInfo, nodeList] = await Promise.all([
      versionApi.getCode(),
      this.coreApi.listNode(),
    ]);

    return {
      serverVersion: versionInfo.body.gitVersion,
      platform: versionInfo.body.platform,
      nodes: nodeList.body.items.length,
    };
  }

  // ===========================================================================
  // Namespace Operations
  // ===========================================================================

  async listNamespaces(): Promise<k8s.V1Namespace[]> {
    const response = await this.coreApi.listNamespace();
    return response.body.items;
  }

  async createNamespace(
    name: string,
    labels?: Record<string, string>
  ): Promise<k8s.V1Namespace> {
    const namespace: k8s.V1Namespace = {
      apiVersion: "v1",
      kind: "Namespace",
      metadata: {
        name,
        labels: {
          ...labels,
          "neurectomy.managed": "true",
        },
      },
    };

    const response = await this.coreApi.createNamespace(namespace);
    this.logger.info({ name }, "Created namespace");
    return response.body;
  }

  async deleteNamespace(name: string): Promise<void> {
    await this.coreApi.deleteNamespace(name);
    this.logger.info({ name }, "Deleted namespace");
  }

  // ===========================================================================
  // Pod Operations
  // ===========================================================================

  async createPod(config: K8sPodConfig): Promise<k8s.V1Pod> {
    const namespace = config.namespace || this.defaultNamespace;

    this.logger.info({ name: config.name, namespace }, "Creating pod");

    const pod = this.buildPodSpec(config);
    const response = await this.coreApi.createNamespacedPod(namespace, pod);
    return response.body;
  }

  async getPod(name: string, namespace?: string): Promise<k8s.V1Pod> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.coreApi.readNamespacedPod(name, ns);
    return response.body;
  }

  async listPods(
    namespace?: string,
    labelSelector?: string
  ): Promise<k8s.V1Pod[]> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.coreApi.listNamespacedPod(
      ns,
      undefined,
      undefined,
      undefined,
      undefined,
      labelSelector
    );
    return response.body.items;
  }

  async deletePod(
    name: string,
    namespace?: string,
    gracePeriodSeconds?: number
  ): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.coreApi.deleteNamespacedPod(
      name,
      ns,
      undefined,
      undefined,
      gracePeriodSeconds
    );
    this.logger.info({ name, namespace: ns }, "Deleted pod");
  }

  async getPodLogs(
    name: string,
    namespace?: string,
    options: {
      container?: string;
      follow?: boolean;
      tailLines?: number;
      sinceSeconds?: number;
      timestamps?: boolean;
    } = {}
  ): Promise<string> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.coreApi.readNamespacedPodLog(
      name,
      ns,
      options.container,
      options.follow,
      undefined,
      undefined,
      undefined,
      undefined,
      options.sinceSeconds,
      options.tailLines,
      options.timestamps
    );
    return response.body;
  }

  async execInPod(
    name: string,
    command: string[],
    namespace?: string,
    container?: string
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const ns = namespace || this.defaultNamespace;

    return new Promise((resolve, reject) => {
      const exec = new k8s.Exec(this.kc);
      let stdout = "";
      let stderr = "";

      exec
        .exec(
          ns,
          name,
          container || "",
          command,
          process.stdout,
          process.stderr,
          process.stdin,
          false,
          (status) => {
            resolve({
              stdout,
              stderr,
              exitCode: status.status === "Success" ? 0 : 1,
            });
          }
        )
        .catch(reject);
    });
  }

  // ===========================================================================
  // Deployment Operations
  // ===========================================================================

  async createDeployment(
    config: K8sDeploymentConfig
  ): Promise<k8s.V1Deployment> {
    const namespace = config.namespace || this.defaultNamespace;

    this.logger.info({ name: config.name, namespace }, "Creating deployment");

    const deployment = this.buildDeploymentSpec(config);
    const response = await this.appsApi.createNamespacedDeployment(
      namespace,
      deployment
    );
    return response.body;
  }

  async getDeployment(
    name: string,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.appsApi.readNamespacedDeployment(name, ns);
    return response.body;
  }

  async listDeployments(
    namespace?: string,
    labelSelector?: string
  ): Promise<k8s.V1Deployment[]> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.appsApi.listNamespacedDeployment(
      ns,
      undefined,
      undefined,
      undefined,
      undefined,
      labelSelector
    );
    return response.body.items;
  }

  async updateDeployment(
    name: string,
    update: Partial<K8sDeploymentConfig>,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns = namespace || this.defaultNamespace;

    this.logger.info({ name, namespace: ns }, "Updating deployment");

    const current = await this.getDeployment(name, ns);

    // Apply updates
    if (update.replicas !== undefined) {
      current.spec!.replicas = update.replicas;
    }

    if (update.template) {
      const podSpec = this.buildPodSpec(update.template);
      current.spec!.template = podSpec;
    }

    const response = await this.appsApi.replaceNamespacedDeployment(
      name,
      ns,
      current
    );
    return response.body;
  }

  async scaleDeployment(
    name: string,
    replicas: number,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns = namespace || this.defaultNamespace;

    this.logger.info({ name, namespace: ns, replicas }, "Scaling deployment");

    const scale: k8s.V1Scale = {
      apiVersion: "autoscaling/v1",
      kind: "Scale",
      metadata: { name, namespace: ns },
      spec: { replicas },
    };

    await this.appsApi.replaceNamespacedDeploymentScale(name, ns, scale);
    return this.getDeployment(name, ns);
  }

  async deleteDeployment(name: string, namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.appsApi.deleteNamespacedDeployment(name, ns);
    this.logger.info({ name, namespace: ns }, "Deleted deployment");
  }

  async rolloutRestart(
    name: string,
    namespace?: string
  ): Promise<k8s.V1Deployment> {
    const ns = namespace || this.defaultNamespace;

    this.logger.info({ name, namespace: ns }, "Restarting deployment");

    const patch = {
      spec: {
        template: {
          metadata: {
            annotations: {
              "kubectl.kubernetes.io/restartedAt": new Date().toISOString(),
            },
          },
        },
      },
    };

    const options = {
      headers: {
        "Content-type": k8s.PatchUtils.PATCH_FORMAT_STRATEGIC_MERGE_PATCH,
      },
    };

    const response = await this.appsApi.patchNamespacedDeployment(
      name,
      ns,
      patch,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      options
    );

    return response.body;
  }

  async getDeploymentStatus(
    name: string,
    namespace?: string
  ): Promise<{
    replicas: number;
    readyReplicas: number;
    updatedReplicas: number;
    availableReplicas: number;
    conditions: k8s.V1DeploymentCondition[];
  }> {
    const deployment = await this.getDeployment(name, namespace);
    const status = deployment.status!;

    return {
      replicas: status.replicas || 0,
      readyReplicas: status.readyReplicas || 0,
      updatedReplicas: status.updatedReplicas || 0,
      availableReplicas: status.availableReplicas || 0,
      conditions: status.conditions || [],
    };
  }

  // ===========================================================================
  // Service Operations
  // ===========================================================================

  async createService(config: K8sServiceConfig): Promise<k8s.V1Service> {
    const namespace = config.namespace || this.defaultNamespace;

    this.logger.info({ name: config.name, namespace }, "Creating service");

    const service = this.buildServiceSpec(config);
    const response = await this.coreApi.createNamespacedService(
      namespace,
      service
    );
    return response.body;
  }

  async getService(name: string, namespace?: string): Promise<k8s.V1Service> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.coreApi.readNamespacedService(name, ns);
    return response.body;
  }

  async listServices(
    namespace?: string,
    labelSelector?: string
  ): Promise<k8s.V1Service[]> {
    const ns = namespace || this.defaultNamespace;
    const response = await this.coreApi.listNamespacedService(
      ns,
      undefined,
      undefined,
      undefined,
      undefined,
      labelSelector
    );
    return response.body.items;
  }

  async deleteService(name: string, namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.coreApi.deleteNamespacedService(name, ns);
    this.logger.info({ name, namespace: ns }, "Deleted service");
  }

  // ===========================================================================
  // HPA Operations
  // ===========================================================================

  async createHPA(
    config: K8sHPAConfig
  ): Promise<k8s.V2HorizontalPodAutoscaler> {
    const namespace = config.namespace || this.defaultNamespace;

    this.logger.info({ name: config.name, namespace }, "Creating HPA");

    const hpa = this.buildHPASpec(config);
    const response =
      await this.autoscalingApi.createNamespacedHorizontalPodAutoscaler(
        namespace,
        hpa
      );
    return response.body;
  }

  async getHPA(
    name: string,
    namespace?: string
  ): Promise<k8s.V2HorizontalPodAutoscaler> {
    const ns = namespace || this.defaultNamespace;
    const response =
      await this.autoscalingApi.readNamespacedHorizontalPodAutoscaler(name, ns);
    return response.body;
  }

  async deleteHPA(name: string, namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    await this.autoscalingApi.deleteNamespacedHorizontalPodAutoscaler(name, ns);
    this.logger.info({ name, namespace: ns }, "Deleted HPA");
  }

  // ===========================================================================
  // ConfigMap & Secret Operations
  // ===========================================================================

  async createConfigMap(
    name: string,
    data: Record<string, string>,
    namespace?: string,
    labels?: Record<string, string>
  ): Promise<k8s.V1ConfigMap> {
    const ns = namespace || this.defaultNamespace;

    const configMap: k8s.V1ConfigMap = {
      apiVersion: "v1",
      kind: "ConfigMap",
      metadata: {
        name,
        namespace: ns,
        labels: {
          ...labels,
          "neurectomy.managed": "true",
        },
      },
      data,
    };

    const response = await this.coreApi.createNamespacedConfigMap(
      ns,
      configMap
    );
    this.logger.info({ name, namespace: ns }, "Created ConfigMap");
    return response.body;
  }

  async createSecret(
    name: string,
    data: Record<string, string>,
    namespace?: string,
    type: string = "Opaque",
    labels?: Record<string, string>
  ): Promise<k8s.V1Secret> {
    const ns = namespace || this.defaultNamespace;

    // Base64 encode values
    const encodedData: Record<string, string> = {};
    for (const [key, value] of Object.entries(data)) {
      encodedData[key] = Buffer.from(value).toString("base64");
    }

    const secret: k8s.V1Secret = {
      apiVersion: "v1",
      kind: "Secret",
      metadata: {
        name,
        namespace: ns,
        labels: {
          ...labels,
          "neurectomy.managed": "true",
        },
      },
      type,
      data: encodedData,
    };

    const response = await this.coreApi.createNamespacedSecret(ns, secret);
    this.logger.info({ name, namespace: ns }, "Created Secret");
    return response.body;
  }

  // ===========================================================================
  // Watch Operations
  // ===========================================================================

  async watchPods(options: WatchOptions = {}): Promise<void> {
    const ns = options.namespace || this.defaultNamespace;
    const path = `/api/v1/namespaces/${ns}/pods`;

    await this.startWatch(`pods-${ns}`, path, options, (type, obj) => {
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
    });
  }

  async watchDeployments(options: WatchOptions = {}): Promise<void> {
    const ns = options.namespace || this.defaultNamespace;
    const path = `/apis/apps/v1/namespaces/${ns}/deployments`;

    await this.startWatch(`deployments-${ns}`, path, options, (type, obj) => {
      const deployment = obj as k8s.V1Deployment;
      switch (type) {
        case "ADDED":
          this.emit("deployment:added", deployment);
          break;
        case "MODIFIED":
          this.emit("deployment:modified", deployment);
          break;
        case "DELETED":
          this.emit("deployment:deleted", deployment);
          break;
      }
    });
  }

  async watchServices(options: WatchOptions = {}): Promise<void> {
    const ns = options.namespace || this.defaultNamespace;
    const path = `/api/v1/namespaces/${ns}/services`;

    await this.startWatch(`services-${ns}`, path, options, (type, obj) => {
      const service = obj as k8s.V1Service;
      switch (type) {
        case "ADDED":
          this.emit("service:added", service);
          break;
        case "MODIFIED":
          this.emit("service:modified", service);
          break;
        case "DELETED":
          this.emit("service:deleted", service);
          break;
      }
    });
  }

  async watchNodes(): Promise<void> {
    const path = "/api/v1/nodes";

    await this.startWatch("nodes", path, {}, (type, obj) => {
      const node = obj as k8s.V1Node;
      switch (type) {
        case "ADDED":
          this.emit("node:added", node);
          break;
        case "MODIFIED":
          this.emit("node:modified", node);
          break;
        case "DELETED":
          this.emit("node:deleted", node);
          break;
      }
    });
  }

  async watchEvents(namespace?: string): Promise<void> {
    const ns = namespace || this.defaultNamespace;
    const path = `/api/v1/namespaces/${ns}/events`;

    await this.startWatch(`events-${ns}`, path, {}, (type, obj) => {
      if (type === "ADDED" || type === "MODIFIED") {
        this.emit("event", obj as k8s.CoreV1Event);
      }
    });
  }

  private async startWatch(
    key: string,
    path: string,
    options: WatchOptions,
    callback: (type: string, obj: object) => void
  ): Promise<void> {
    // Stop existing watch if any
    this.stopWatch(key);

    const controller = new AbortController();
    this.watchers.set(key, controller);

    const queryParams: Record<string, string> = {};
    if (options.labelSelector) {
      queryParams.labelSelector = options.labelSelector;
    }
    if (options.fieldSelector) {
      queryParams.fieldSelector = options.fieldSelector;
    }
    if (options.resourceVersion) {
      queryParams.resourceVersion = options.resourceVersion;
    }

    try {
      await this.watch.watch(path, queryParams, callback, (err) => {
        if (err) {
          this.logger.error({ err, path }, "Watch error");
          this.emit("error", err as Error);
        }
        // Attempt to reconnect
        if (!controller.signal.aborted) {
          this.logger.info({ path }, "Reconnecting watch");
          setTimeout(() => this.startWatch(key, path, options, callback), 1000);
        }
      });
    } catch (error) {
      this.logger.error({ error, path }, "Failed to start watch");
      throw error;
    }
  }

  stopWatch(key: string): void {
    const controller = this.watchers.get(key);
    if (controller) {
      controller.abort();
      this.watchers.delete(key);
    }
  }

  stopAllWatches(): void {
    for (const key of this.watchers.keys()) {
      this.stopWatch(key);
    }
  }

  // ===========================================================================
  // Metrics Operations
  // ===========================================================================

  async getPodMetrics(namespace?: string): Promise<PodMetrics[]> {
    const ns = namespace || this.defaultNamespace;

    try {
      const response = await this.metricsClient.getPodMetrics(ns);

      return response.items.map((item) => ({
        name: item.metadata.name,
        namespace: item.metadata.namespace,
        containers: item.containers.map((c) => ({
          name: c.name,
          cpuUsage: c.usage.cpu,
          memoryUsage: c.usage.memory,
        })),
        timestamp: new Date(item.timestamp),
      }));
    } catch (error) {
      this.logger.warn(
        { error },
        "Failed to get pod metrics (metrics-server may not be installed)"
      );
      return [];
    }
  }

  async getNodeMetrics(): Promise<NodeMetrics[]> {
    try {
      const response = await this.metricsClient.getNodeMetrics();

      return response.items.map((item) => ({
        name: item.metadata.name,
        cpuUsage: item.usage.cpu,
        memoryUsage: item.usage.memory,
        timestamp: new Date(item.timestamp),
      }));
    } catch (error) {
      this.logger.warn(
        { error },
        "Failed to get node metrics (metrics-server may not be installed)"
      );
      return [];
    }
  }

  // ===========================================================================
  // Custom Resource Definition Operations
  // ===========================================================================

  async createCustomResource(
    group: string,
    version: string,
    namespace: string,
    plural: string,
    body: object
  ): Promise<object> {
    const response = await this.customObjectsApi.createNamespacedCustomObject(
      group,
      version,
      namespace,
      plural,
      body
    );
    return response.body;
  }

  async getCustomResource(
    group: string,
    version: string,
    namespace: string,
    plural: string,
    name: string
  ): Promise<object> {
    const response = await this.customObjectsApi.getNamespacedCustomObject(
      group,
      version,
      namespace,
      plural,
      name
    );
    return response.body;
  }

  async listCustomResources(
    group: string,
    version: string,
    namespace: string,
    plural: string
  ): Promise<object[]> {
    const response = await this.customObjectsApi.listNamespacedCustomObject(
      group,
      version,
      namespace,
      plural
    );
    return (response.body as { items: object[] }).items;
  }

  async deleteCustomResource(
    group: string,
    version: string,
    namespace: string,
    plural: string,
    name: string
  ): Promise<void> {
    await this.customObjectsApi.deleteNamespacedCustomObject(
      group,
      version,
      namespace,
      plural,
      name
    );
  }

  // ===========================================================================
  // Builder Methods
  // ===========================================================================

  private buildPodSpec(config: K8sPodConfig): k8s.V1Pod {
    const pod: k8s.V1Pod = {
      apiVersion: "v1",
      kind: "Pod",
      metadata: {
        name: config.name,
        namespace: config.namespace || this.defaultNamespace,
        labels: {
          ...config.labels,
          "neurectomy.managed": "true",
        },
        annotations: config.annotations,
      },
      spec: {
        containers: config.containers.map((c) => this.buildContainerSpec(c)),
        initContainers: config.initContainers?.map((c) =>
          this.buildContainerSpec(c)
        ),
        volumes: config.volumes?.map((v) => this.buildVolumeSpec(v)),
        serviceAccountName: config.serviceAccountName,
        securityContext: config.securityContext
          ? this.buildPodSecurityContext(config.securityContext)
          : undefined,
        nodeSelector: config.nodeSelector,
        tolerations: config.tolerations?.map((t) => ({
          key: t.key,
          operator: t.operator,
          value: t.value,
          effect: t.effect,
          tolerationSeconds: t.tolerationSeconds,
        })),
        restartPolicy: config.restartPolicy,
        terminationGracePeriodSeconds: config.terminationGracePeriodSeconds,
      },
    };

    return pod;
  }

  private buildContainerSpec(config: K8sContainerSpec): k8s.V1Container {
    return {
      name: config.name,
      image: config.image,
      command: config.command,
      args: config.args,
      env: config.env?.map((e) => this.buildEnvVar(e)),
      envFrom: config.envFrom?.map((ef) => ({
        configMapRef: ef.configMapRef
          ? { name: ef.configMapRef.name, optional: ef.configMapRef.optional }
          : undefined,
        secretRef: ef.secretRef
          ? { name: ef.secretRef.name, optional: ef.secretRef.optional }
          : undefined,
        prefix: ef.prefix,
      })),
      ports: config.ports?.map((p) => ({
        name: p.name,
        containerPort: p.containerPort,
        hostPort: p.hostPort,
        protocol: p.protocol,
      })),
      resources: config.resources
        ? this.buildResourceRequirements(config.resources)
        : undefined,
      volumeMounts: config.volumeMounts?.map((vm) => ({
        name: vm.name,
        mountPath: vm.mountPath,
        subPath: vm.subPath,
        readOnly: vm.readOnly,
      })),
      livenessProbe: config.livenessProbe
        ? this.buildProbe(config.livenessProbe)
        : undefined,
      readinessProbe: config.readinessProbe
        ? this.buildProbe(config.readinessProbe)
        : undefined,
      startupProbe: config.startupProbe
        ? this.buildProbe(config.startupProbe)
        : undefined,
      securityContext: config.securityContext
        ? this.buildSecurityContext(config.securityContext)
        : undefined,
      imagePullPolicy: config.imagePullPolicy,
    };
  }

  private buildEnvVar(env: K8sEnvVar): k8s.V1EnvVar {
    const envVar: k8s.V1EnvVar = { name: env.name };

    if (env.value !== undefined) {
      envVar.value = env.value;
    } else if (env.valueFrom) {
      envVar.valueFrom = {};
      if (env.valueFrom.configMapKeyRef) {
        envVar.valueFrom.configMapKeyRef = {
          name: env.valueFrom.configMapKeyRef.name,
          key: env.valueFrom.configMapKeyRef.key,
        };
      }
      if (env.valueFrom.secretKeyRef) {
        envVar.valueFrom.secretKeyRef = {
          name: env.valueFrom.secretKeyRef.name,
          key: env.valueFrom.secretKeyRef.key,
        };
      }
      if (env.valueFrom.fieldRef) {
        envVar.valueFrom.fieldRef = {
          fieldPath: env.valueFrom.fieldRef.fieldPath,
        };
      }
    }

    return envVar;
  }

  private buildResourceRequirements(
    resources: K8sResourceRequirements
  ): k8s.V1ResourceRequirements {
    return {
      limits: resources.limits,
      requests: resources.requests,
    };
  }

  private buildProbe(probe: K8sProbe): k8s.V1Probe {
    const k8sProbe: k8s.V1Probe = {
      initialDelaySeconds: probe.initialDelaySeconds,
      periodSeconds: probe.periodSeconds,
      timeoutSeconds: probe.timeoutSeconds,
      successThreshold: probe.successThreshold,
      failureThreshold: probe.failureThreshold,
    };

    if (probe.httpGet) {
      k8sProbe.httpGet = {
        path: probe.httpGet.path,
        port: probe.httpGet.port,
        scheme: probe.httpGet.scheme,
      };
    }

    if (probe.tcpSocket) {
      k8sProbe.tcpSocket = {
        port: probe.tcpSocket.port,
      };
    }

    if (probe.exec) {
      k8sProbe.exec = {
        command: probe.exec.command,
      };
    }

    return k8sProbe;
  }

  private buildVolumeSpec(volume: K8sVolume): k8s.V1Volume {
    const v: k8s.V1Volume = { name: volume.name };

    if (volume.emptyDir) {
      v.emptyDir = {
        medium: volume.emptyDir.medium,
        sizeLimit: volume.emptyDir.sizeLimit,
      };
    }

    if (volume.hostPath) {
      v.hostPath = {
        path: volume.hostPath.path,
        type: volume.hostPath.type,
      };
    }

    if (volume.configMap) {
      v.configMap = {
        name: volume.configMap.name,
        items: volume.configMap.items?.map((i) => ({
          key: i.key,
          path: i.path,
        })),
      };
    }

    if (volume.secret) {
      v.secret = {
        secretName: volume.secret.secretName,
        items: volume.secret.items?.map((i) => ({
          key: i.key,
          path: i.path,
        })),
      };
    }

    if (volume.persistentVolumeClaim) {
      v.persistentVolumeClaim = {
        claimName: volume.persistentVolumeClaim.claimName,
        readOnly: volume.persistentVolumeClaim.readOnly,
      };
    }

    return v;
  }

  private buildSecurityContext(
    context: K8sSecurityContext
  ): k8s.V1SecurityContext {
    return {
      runAsUser: context.runAsUser,
      runAsGroup: context.runAsGroup,
      runAsNonRoot: context.runAsNonRoot,
      readOnlyRootFilesystem: context.readOnlyRootFilesystem,
      allowPrivilegeEscalation: context.allowPrivilegeEscalation,
      privileged: context.privileged,
      capabilities: context.capabilities
        ? {
            add: context.capabilities.add,
            drop: context.capabilities.drop,
          }
        : undefined,
    };
  }

  private buildPodSecurityContext(
    context: K8sPodSecurityContext
  ): k8s.V1PodSecurityContext {
    return {
      runAsUser: context.runAsUser,
      runAsGroup: context.runAsGroup,
      fsGroup: context.fsGroup,
      runAsNonRoot: context.runAsNonRoot,
      seccompProfile: context.seccompProfile
        ? {
            type: context.seccompProfile.type,
            localhostProfile: context.seccompProfile.localhostProfile,
          }
        : undefined,
    };
  }

  private buildDeploymentSpec(config: K8sDeploymentConfig): k8s.V1Deployment {
    return {
      apiVersion: "apps/v1",
      kind: "Deployment",
      metadata: {
        name: config.name,
        namespace: config.namespace || this.defaultNamespace,
        labels: {
          ...config.labels,
          "neurectomy.managed": "true",
        },
        annotations: config.annotations,
      },
      spec: {
        replicas: config.replicas ?? 1,
        selector: this.buildLabelSelector(config.selector),
        template: this.buildPodSpec(config.template),
        strategy: config.strategy
          ? {
              type: config.strategy.type,
              rollingUpdate: config.strategy.rollingUpdate
                ? {
                    maxUnavailable:
                      config.strategy.rollingUpdate.maxUnavailable,
                    maxSurge: config.strategy.rollingUpdate.maxSurge,
                  }
                : undefined,
            }
          : undefined,
        minReadySeconds: config.minReadySeconds,
        revisionHistoryLimit: config.revisionHistoryLimit,
        progressDeadlineSeconds: config.progressDeadlineSeconds,
      },
    };
  }

  private buildServiceSpec(config: K8sServiceConfig): k8s.V1Service {
    return {
      apiVersion: "v1",
      kind: "Service",
      metadata: {
        name: config.name,
        namespace: config.namespace || this.defaultNamespace,
        labels: {
          ...config.labels,
          "neurectomy.managed": "true",
        },
        annotations: config.annotations,
      },
      spec: {
        type: config.type,
        selector: config.selector,
        ports: config.ports.map((p) => ({
          name: p.name,
          protocol: p.protocol,
          port: p.port,
          targetPort: p.targetPort,
          nodePort: p.nodePort,
        })),
        clusterIP: config.clusterIP,
        externalIPs: config.externalIPs,
        loadBalancerIP: config.loadBalancerIP,
        sessionAffinity: config.sessionAffinity,
      },
    };
  }

  private buildHPASpec(config: K8sHPAConfig): k8s.V2HorizontalPodAutoscaler {
    return {
      apiVersion: "autoscaling/v2",
      kind: "HorizontalPodAutoscaler",
      metadata: {
        name: config.name,
        namespace: config.namespace || this.defaultNamespace,
        labels: {
          "neurectomy.managed": "true",
        },
      },
      spec: {
        scaleTargetRef: {
          apiVersion: config.scaleTargetRef.apiVersion,
          kind: config.scaleTargetRef.kind,
          name: config.scaleTargetRef.name,
        },
        minReplicas: config.minReplicas,
        maxReplicas: config.maxReplicas,
        metrics: config.metrics?.map((m) => ({
          type: m.type,
          resource: m.resource
            ? {
                name: m.resource.name,
                target: {
                  type: m.resource.target.type,
                  averageUtilization: m.resource.target.averageUtilization,
                  averageValue: m.resource.target.averageValue,
                },
              }
            : undefined,
        })),
        behavior: config.behavior
          ? {
              scaleDown: config.behavior.scaleDown
                ? {
                    stabilizationWindowSeconds:
                      config.behavior.scaleDown.stabilizationWindowSeconds,
                    selectPolicy: config.behavior.scaleDown.selectPolicy,
                    policies: config.behavior.scaleDown.policies?.map((p) => ({
                      type: p.type,
                      value: p.value,
                      periodSeconds: p.periodSeconds,
                    })),
                  }
                : undefined,
              scaleUp: config.behavior.scaleUp
                ? {
                    stabilizationWindowSeconds:
                      config.behavior.scaleUp.stabilizationWindowSeconds,
                    selectPolicy: config.behavior.scaleUp.selectPolicy,
                    policies: config.behavior.scaleUp.policies?.map((p) => ({
                      type: p.type,
                      value: p.value,
                      periodSeconds: p.periodSeconds,
                    })),
                  }
                : undefined,
            }
          : undefined,
      },
    };
  }

  private buildLabelSelector(selector: K8sLabelSelector): k8s.V1LabelSelector {
    return {
      matchLabels: selector.matchLabels,
      matchExpressions: selector.matchExpressions?.map((e) => ({
        key: e.key,
        operator: e.operator,
        values: e.values,
      })),
    };
  }
}

// =============================================================================
// Custom Errors
// =============================================================================

export class K8sConnectionError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = "K8sConnectionError";
  }
}

export class K8sResourceError extends Error {
  constructor(
    message: string,
    public readonly resourceKind: string,
    public readonly resourceName: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = "K8sResourceError";
  }
}
