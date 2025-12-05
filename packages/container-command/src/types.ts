/**
 * NEURECTOMY Container Command Types
 *
 * @FLUX @ARCHITECT - DevOps + System Design
 *
 * Comprehensive type definitions for container orchestration,
 * supporting Docker, Kubernetes, Firecracker MicroVMs, and WebAssembly runtimes.
 */

import { z } from "zod";

// =============================================================================
// Base Container Types
// =============================================================================

export type ContainerRuntime =
  | "docker"
  | "kubernetes"
  | "firecracker"
  | "wasmtime";

export type ContainerStatus =
  | "created"
  | "running"
  | "paused"
  | "restarting"
  | "removing"
  | "exited"
  | "dead";

export type HealthStatus = "starting" | "healthy" | "unhealthy" | "none";

export interface ContainerInfo {
  id: string;
  name: string;
  image: string;
  imageId: string;
  runtime: ContainerRuntime;
  status: ContainerStatus;
  health: HealthStatus;
  created: Date;
  started?: Date;
  finished?: Date;
  exitCode?: number;
  labels: Record<string, string>;
  ports: PortMapping[];
  mounts: Mount[];
  networks: NetworkInfo[];
  resources: ResourceUsage;
}

export interface PortMapping {
  hostPort: number;
  containerPort: number;
  protocol: "tcp" | "udp";
  hostIp?: string;
}

export interface Mount {
  type: "bind" | "volume" | "tmpfs";
  source: string;
  target: string;
  readOnly: boolean;
}

export interface NetworkInfo {
  name: string;
  networkId: string;
  ipAddress: string;
  gateway: string;
  macAddress: string;
}

export interface ResourceUsage {
  cpuPercent: number;
  memoryUsage: number;
  memoryLimit: number;
  memoryPercent: number;
  networkRx: number;
  networkTx: number;
  blockRead: number;
  blockWrite: number;
  pids: number;
}

// =============================================================================
// Docker Specific Types
// =============================================================================

export interface DockerConfig {
  socketPath?: string;
  host?: string;
  port?: number;
  ca?: string;
  cert?: string;
  key?: string;
  version?: string;
  timeout?: number;
}

export interface DockerContainerConfig {
  name: string;
  image: string;
  tag?: string;
  cmd?: string[];
  entrypoint?: string[];
  env?: Record<string, string>;
  labels?: Record<string, string>;
  workingDir?: string;
  user?: string;
  exposedPorts?: number[];
  portBindings?: PortMapping[];
  mounts?: Mount[];
  networkMode?: string;
  networks?: string[];
  hostname?: string;
  domainname?: string;
  privileged?: boolean;
  readonlyRootfs?: boolean;
  resources?: ResourceLimits;
  healthcheck?: HealthcheckConfig;
  restart?: RestartPolicy;
  autoRemove?: boolean;
  tty?: boolean;
  stdin?: boolean;
  attachStdout?: boolean;
  attachStderr?: boolean;
}

export interface ResourceLimits {
  cpuShares?: number;
  cpuPeriod?: number;
  cpuQuota?: number;
  cpusetCpus?: string;
  memory?: number;
  memorySwap?: number;
  memoryReservation?: number;
  pidsLimit?: number;
  ulimits?: Ulimit[];
}

export interface Ulimit {
  name: string;
  soft: number;
  hard: number;
}

export interface HealthcheckConfig {
  test: string[];
  interval?: number;
  timeout?: number;
  retries?: number;
  startPeriod?: number;
}

export type RestartPolicy = "no" | "always" | "unless-stopped" | "on-failure";

export interface DockerImageConfig {
  repository: string;
  tag?: string;
  dockerfile?: string;
  context?: string;
  buildArgs?: Record<string, string>;
  labels?: Record<string, string>;
  target?: string;
  platform?: string;
  cache?: boolean;
  pull?: boolean;
  squash?: boolean;
}

export interface DockerNetworkConfig {
  name: string;
  driver?: "bridge" | "host" | "overlay" | "macvlan" | "none";
  internal?: boolean;
  attachable?: boolean;
  ingress?: boolean;
  ipam?: IpamConfig;
  options?: Record<string, string>;
  labels?: Record<string, string>;
}

export interface IpamConfig {
  driver?: string;
  config?: {
    subnet?: string;
    ipRange?: string;
    gateway?: string;
    auxAddress?: Record<string, string>;
  }[];
}

export interface DockerVolumeConfig {
  name: string;
  driver?: string;
  driverOpts?: Record<string, string>;
  labels?: Record<string, string>;
}

// =============================================================================
// Kubernetes Specific Types
// =============================================================================

export interface K8sConfig {
  kubeconfig?: string;
  context?: string;
  namespace?: string;
  inCluster?: boolean;
  server?: string;
  token?: string;
  ca?: string;
}

export interface K8sPodConfig {
  name: string;
  namespace?: string;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
  containers: K8sContainerSpec[];
  initContainers?: K8sContainerSpec[];
  volumes?: K8sVolume[];
  serviceAccountName?: string;
  securityContext?: K8sPodSecurityContext;
  nodeSelector?: Record<string, string>;
  tolerations?: K8sToleration[];
  affinity?: K8sAffinity;
  restartPolicy?: "Always" | "OnFailure" | "Never";
  terminationGracePeriodSeconds?: number;
}

export interface K8sContainerSpec {
  name: string;
  image: string;
  command?: string[];
  args?: string[];
  env?: K8sEnvVar[];
  envFrom?: K8sEnvFromSource[];
  ports?: K8sContainerPort[];
  resources?: K8sResourceRequirements;
  volumeMounts?: K8sVolumeMount[];
  livenessProbe?: K8sProbe;
  readinessProbe?: K8sProbe;
  startupProbe?: K8sProbe;
  securityContext?: K8sSecurityContext;
  imagePullPolicy?: "Always" | "IfNotPresent" | "Never";
}

export interface K8sEnvVar {
  name: string;
  value?: string;
  valueFrom?: {
    configMapKeyRef?: { name: string; key: string };
    secretKeyRef?: { name: string; key: string };
    fieldRef?: { fieldPath: string };
  };
}

export interface K8sEnvFromSource {
  configMapRef?: { name: string; optional?: boolean };
  secretRef?: { name: string; optional?: boolean };
  prefix?: string;
}

export interface K8sContainerPort {
  name?: string;
  containerPort: number;
  hostPort?: number;
  protocol?: "TCP" | "UDP" | "SCTP";
}

export interface K8sResourceRequirements {
  limits?: {
    cpu?: string;
    memory?: string;
    "nvidia.com/gpu"?: string;
  };
  requests?: {
    cpu?: string;
    memory?: string;
  };
}

export interface K8sVolumeMount {
  name: string;
  mountPath: string;
  subPath?: string;
  readOnly?: boolean;
}

export interface K8sVolume {
  name: string;
  emptyDir?: { medium?: string; sizeLimit?: string };
  hostPath?: { path: string; type?: string };
  configMap?: { name: string; items?: { key: string; path: string }[] };
  secret?: { secretName: string; items?: { key: string; path: string }[] };
  persistentVolumeClaim?: { claimName: string; readOnly?: boolean };
}

export interface K8sProbe {
  httpGet?: { path: string; port: number | string; scheme?: string };
  tcpSocket?: { port: number | string };
  exec?: { command: string[] };
  initialDelaySeconds?: number;
  periodSeconds?: number;
  timeoutSeconds?: number;
  successThreshold?: number;
  failureThreshold?: number;
}

export interface K8sSecurityContext {
  runAsUser?: number;
  runAsGroup?: number;
  runAsNonRoot?: boolean;
  readOnlyRootFilesystem?: boolean;
  allowPrivilegeEscalation?: boolean;
  privileged?: boolean;
  capabilities?: {
    add?: string[];
    drop?: string[];
  };
}

export interface K8sPodSecurityContext {
  runAsUser?: number;
  runAsGroup?: number;
  fsGroup?: number;
  runAsNonRoot?: boolean;
  seccompProfile?: { type: string; localhostProfile?: string };
}

export interface K8sToleration {
  key?: string;
  operator?: "Exists" | "Equal";
  value?: string;
  effect?: "NoSchedule" | "PreferNoSchedule" | "NoExecute";
  tolerationSeconds?: number;
}

export interface K8sAffinity {
  nodeAffinity?: K8sNodeAffinity;
  podAffinity?: K8sPodAffinity;
  podAntiAffinity?: K8sPodAntiAffinity;
}

export interface K8sNodeAffinity {
  requiredDuringSchedulingIgnoredDuringExecution?: {
    nodeSelectorTerms: K8sNodeSelectorTerm[];
  };
  preferredDuringSchedulingIgnoredDuringExecution?: {
    weight: number;
    preference: K8sNodeSelectorTerm;
  }[];
}

export interface K8sNodeSelectorTerm {
  matchExpressions?: {
    key: string;
    operator: "In" | "NotIn" | "Exists" | "DoesNotExist" | "Gt" | "Lt";
    values?: string[];
  }[];
}

export interface K8sPodAffinity {
  requiredDuringSchedulingIgnoredDuringExecution?: K8sPodAffinityTerm[];
  preferredDuringSchedulingIgnoredDuringExecution?: {
    weight: number;
    podAffinityTerm: K8sPodAffinityTerm;
  }[];
}

export interface K8sPodAntiAffinity extends K8sPodAffinity {}

export interface K8sPodAffinityTerm {
  labelSelector?: K8sLabelSelector;
  topologyKey: string;
  namespaces?: string[];
}

export interface K8sLabelSelector {
  matchLabels?: Record<string, string>;
  matchExpressions?: {
    key: string;
    operator: "In" | "NotIn" | "Exists" | "DoesNotExist";
    values?: string[];
  }[];
}

// Deployment types
export interface K8sDeploymentConfig {
  name: string;
  namespace?: string;
  replicas?: number;
  selector: K8sLabelSelector;
  template: K8sPodConfig;
  strategy?: K8sDeploymentStrategy;
  minReadySeconds?: number;
  revisionHistoryLimit?: number;
  progressDeadlineSeconds?: number;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
}

export interface K8sDeploymentStrategy {
  type: "RollingUpdate" | "Recreate";
  rollingUpdate?: {
    maxUnavailable?: number | string;
    maxSurge?: number | string;
  };
}

// Service types
export interface K8sServiceConfig {
  name: string;
  namespace?: string;
  type?: "ClusterIP" | "NodePort" | "LoadBalancer" | "ExternalName";
  selector?: Record<string, string>;
  ports: K8sServicePort[];
  clusterIP?: string;
  externalIPs?: string[];
  loadBalancerIP?: string;
  sessionAffinity?: "None" | "ClientIP";
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
}

export interface K8sServicePort {
  name?: string;
  protocol?: "TCP" | "UDP" | "SCTP";
  port: number;
  targetPort?: number | string;
  nodePort?: number;
}

// HPA types
export interface K8sHPAConfig {
  name: string;
  namespace?: string;
  scaleTargetRef: {
    apiVersion: string;
    kind: string;
    name: string;
  };
  minReplicas?: number;
  maxReplicas: number;
  metrics?: K8sHPAMetric[];
  behavior?: K8sHPABehavior;
}

export interface K8sHPAMetric {
  type: "Resource" | "Pods" | "Object" | "External";
  resource?: {
    name: string;
    target: {
      type: string;
      averageUtilization?: number;
      averageValue?: string;
    };
  };
}

export interface K8sHPABehavior {
  scaleDown?: K8sHPAScalingRules;
  scaleUp?: K8sHPAScalingRules;
}

export interface K8sHPAScalingRules {
  stabilizationWindowSeconds?: number;
  selectPolicy?: "Max" | "Min" | "Disabled";
  policies?: {
    type: "Pods" | "Percent";
    value: number;
    periodSeconds: number;
  }[];
}

// =============================================================================
// Firecracker MicroVM Types
// =============================================================================

export interface FirecrackerConfig {
  socketPath: string;
  kernel: string;
  rootfs: string;
  bootArgs?: string;
  cpuCount?: number;
  memSizeMib?: number;
}

export interface MicroVMConfig {
  vmId: string;
  kernel: KernelConfig;
  rootfs: RootfsConfig;
  machine: MachineConfig;
  network?: MicroVMNetworkConfig;
  drives?: DriveConfig[];
  vsock?: VsockConfig;
  balloon?: BalloonConfig;
  metricsPath?: string;
  logPath?: string;
}

export interface KernelConfig {
  imagePath: string;
  bootArgs?: string;
}

export interface RootfsConfig {
  imagePath: string;
  readOnly?: boolean;
}

export interface MachineConfig {
  vcpuCount: number;
  memSizeMib: number;
  htEnabled?: boolean;
  trackDirtyPages?: boolean;
}

export interface MicroVMNetworkConfig {
  ifaceId: string;
  hostDevName: string;
  guestMac?: string;
  rxRateLimiter?: RateLimiter;
  txRateLimiter?: RateLimiter;
}

export interface DriveConfig {
  driveId: string;
  pathOnHost: string;
  isRootDevice: boolean;
  isReadOnly?: boolean;
  partuuid?: string;
  rateLimit?: RateLimiter;
}

export interface VsockConfig {
  guestCid: number;
  udsPath: string;
}

export interface BalloonConfig {
  amountMib: number;
  deflateOnOom?: boolean;
  statsPollingIntervalS?: number;
}

export interface RateLimiter {
  bandwidth?: {
    size: number;
    refillTime: number;
    oneTimeBurst?: number;
  };
  ops?: {
    size: number;
    refillTime: number;
    oneTimeBurst?: number;
  };
}

export interface MicroVMSnapshot {
  snapshotPath: string;
  memFilePath: string;
  version: string;
  createdAt: Date;
}

// =============================================================================
// WebAssembly Runtime Types
// =============================================================================

export interface WasmConfig {
  modulePath?: string;
  moduleBytes?: Uint8Array;
  wasi?: WasiConfig;
  fuel?: number;
  epochInterrupts?: boolean;
}

export interface WasiConfig {
  args?: string[];
  env?: Record<string, string>;
  inheritStdio?: boolean;
  preopenDirs?: PreopenDir[];
  stdin?: "inherit" | "null" | string;
  stdout?: "inherit" | "null" | string;
  stderr?: "inherit" | "null" | string;
}

export interface PreopenDir {
  hostPath: string;
  guestPath: string;
  readOnly?: boolean;
}

export interface WasmModule {
  id: string;
  name: string;
  path: string;
  size: number;
  hash: string;
  exports: WasmExport[];
  imports: WasmImport[];
  createdAt: Date;
  metadata?: Record<string, string>;
}

export interface WasmExport {
  name: string;
  kind: "function" | "table" | "memory" | "global";
}

export interface WasmImport {
  module: string;
  name: string;
  kind: "function" | "table" | "memory" | "global";
}

export interface WasmInstance {
  id: string;
  moduleId: string;
  status: "created" | "running" | "suspended" | "terminated";
  fuelConsumed?: number;
  createdAt: Date;
  startedAt?: Date;
  terminatedAt?: Date;
}

// =============================================================================
// Container Events
// =============================================================================

export type ContainerEventType =
  | "create"
  | "start"
  | "stop"
  | "restart"
  | "pause"
  | "unpause"
  | "kill"
  | "die"
  | "destroy"
  | "health_status"
  | "oom"
  | "exec_create"
  | "exec_start"
  | "exec_die";

export interface ContainerEvent {
  type: ContainerEventType;
  containerId: string;
  containerName: string;
  runtime: ContainerRuntime;
  timestamp: Date;
  attributes?: Record<string, string>;
}

// =============================================================================
// Helm Chart Types
// =============================================================================

export interface HelmChart {
  apiVersion: string;
  name: string;
  version: string;
  appVersion?: string;
  description?: string;
  type?: "application" | "library";
  keywords?: string[];
  home?: string;
  sources?: string[];
  dependencies?: HelmDependency[];
  maintainers?: HelmMaintainer[];
  icon?: string;
  deprecated?: boolean;
}

export interface HelmDependency {
  name: string;
  version: string;
  repository: string;
  condition?: string;
  tags?: string[];
  alias?: string;
}

export interface HelmMaintainer {
  name: string;
  email?: string;
  url?: string;
}

export interface HelmValues {
  [key: string]: unknown;
}

export interface HelmRelease {
  name: string;
  namespace: string;
  chart: string;
  version: string;
  values: HelmValues;
  status: "deployed" | "failed" | "pending" | "superseded" | "uninstalled";
  revision: number;
  updatedAt: Date;
}

// =============================================================================
// Zod Schemas for Validation
// =============================================================================

export const DockerContainerConfigSchema = z.object({
  name: z.string().min(1),
  image: z.string().min(1),
  tag: z.string().optional(),
  cmd: z.array(z.string()).optional(),
  entrypoint: z.array(z.string()).optional(),
  env: z.record(z.string()).optional(),
  labels: z.record(z.string()).optional(),
  workingDir: z.string().optional(),
  user: z.string().optional(),
  exposedPorts: z.array(z.number()).optional(),
  networkMode: z.string().optional(),
  hostname: z.string().optional(),
  privileged: z.boolean().optional(),
  readonlyRootfs: z.boolean().optional(),
  autoRemove: z.boolean().optional(),
  tty: z.boolean().optional(),
  stdin: z.boolean().optional(),
});

export const K8sPodConfigSchema = z.object({
  name: z.string().min(1),
  namespace: z.string().optional(),
  labels: z.record(z.string()).optional(),
  annotations: z.record(z.string()).optional(),
  containers: z.array(
    z.object({
      name: z.string().min(1),
      image: z.string().min(1),
      command: z.array(z.string()).optional(),
      args: z.array(z.string()).optional(),
    })
  ),
  restartPolicy: z.enum(["Always", "OnFailure", "Never"]).optional(),
});

export const MicroVMConfigSchema = z.object({
  vmId: z.string().min(1),
  kernel: z.object({
    imagePath: z.string().min(1),
    bootArgs: z.string().optional(),
  }),
  rootfs: z.object({
    imagePath: z.string().min(1),
    readOnly: z.boolean().optional(),
  }),
  machine: z.object({
    vcpuCount: z.number().int().positive(),
    memSizeMib: z.number().int().positive(),
    htEnabled: z.boolean().optional(),
  }),
});

export const WasmConfigSchema = z
  .object({
    modulePath: z.string().optional(),
    moduleBytes: z.instanceof(Uint8Array).optional(),
    fuel: z.number().int().positive().optional(),
    epochInterrupts: z.boolean().optional(),
  })
  .refine((data) => data.modulePath || data.moduleBytes, {
    message: "Either modulePath or moduleBytes must be provided",
  });
