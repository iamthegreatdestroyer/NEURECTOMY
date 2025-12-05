/**
 * @neurectomy/deployment-orchestrator
 * Enterprise-grade deployment orchestration types
 */

import { z } from "zod";

// =============================================================================
// Core Deployment Types
// =============================================================================

export type DeploymentStrategy =
  | "rolling"
  | "blue-green"
  | "canary"
  | "recreate"
  | "a-b-testing";

export type DeploymentStatus =
  | "pending"
  | "initializing"
  | "in-progress"
  | "in_progress"
  | "paused"
  | "waiting-approval"
  | "promoting"
  | "rolling-back"
  | "rolled-back"
  | "succeeded"
  | "success"
  | "completed"
  | "failed"
  | "cancelled"
  | "running"
  | "partial";

export type DeploymentPhase =
  | "pre-deployment"
  | "deployment"
  | "deploying"
  | "validation"
  | "verifying"
  | "waiting"
  | "canary"
  | "promoting"
  | "switching"
  | "traffic-shift"
  | "cleanup"
  | "post-deployment";

export type HealthStatus = "healthy" | "degraded" | "unhealthy" | "unknown";

export type TrafficWeight = number; // 0-100

// =============================================================================
// Deployment Configuration
// =============================================================================

export interface DeploymentConfig {
  /** Unique deployment identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** Target environment */
  environment: string;
  /** Deployment strategy */
  strategy: DeploymentStrategy;
  /** Target namespace */
  namespace: string;
  /** Application/service name */
  application: string;
  /** Target version/tag */
  version: string;
  /** Previous version for rollback */
  previousVersion?: string;
  /** Container image */
  image: string;
  /** Replica count */
  replicas: number;
  /** Resource configuration */
  resources?: ResourceConfig;
  /** Strategy-specific configuration */
  strategyConfig: StrategyConfig;
  /** Health check configuration */
  healthCheck: HealthCheckConfig;
  /** Approval workflow configuration */
  approval?: ApprovalConfig;
  /** Whether approval is required */
  approvalRequired?: boolean;
  /** Whether to rollback on failure */
  rollbackOnFailure?: boolean;
  /** Canary-specific config (shorthand) */
  canaryConfig?: CanaryConfig;
  /** Notification configuration */
  notifications?: NotificationConfig;
  /** Custom labels */
  labels?: Record<string, string>;
  /** Custom annotations */
  annotations?: Record<string, string>;
  /** Timeout in milliseconds */
  timeout?: number;
  /** Created timestamp */
  createdAt: Date;
  /** Created by user */
  createdBy: string;
}

export interface ResourceConfig {
  requests: {
    cpu: string;
    memory: string;
  };
  limits: {
    cpu: string;
    memory: string;
  };
}

export interface StrategyConfig {
  rolling?: RollingConfig;
  blueGreen?: BlueGreenConfig;
  canary?: CanaryConfig;
  abTesting?: ABTestingConfig;
}

// =============================================================================
// Rolling Update Configuration
// =============================================================================

export interface RollingConfig {
  /** Maximum pods that can be unavailable during update */
  maxUnavailable: number | string;
  /** Maximum pods that can be created over desired replica count */
  maxSurge: number | string;
  /** Minimum seconds pod is ready before considered available */
  minReadySeconds: number;
  /** Deadline for progress before marking failed */
  progressDeadlineSeconds: number;
}

// =============================================================================
// Blue-Green Deployment Configuration
// =============================================================================

export interface BlueGreenConfig {
  /** Service name for blue environment */
  blueService: string;
  /** Service name for green environment */
  greenService: string;
  /** Current active environment */
  activeEnvironment: "blue" | "green";
  /** Active label for identifying active deployment */
  activeLabel?: string;
  /** Auto-promote after validation */
  autoPromote: boolean;
  /** Promotion delay in milliseconds */
  promotionDelay?: number;
  /** Verification timeout in seconds */
  verificationTimeout?: number;
  /** Delay before scaling down old environment */
  scaleDownDelay?: number;
  /** Keep previous environment after promotion */
  scaleDownDelaySeconds: number;
  /** Pre-promotion analysis */
  prePromotionAnalysis?: AnalysisConfig;
  /** Post-promotion analysis */
  postPromotionAnalysis?: AnalysisConfig;
}

// =============================================================================
// Canary Release Configuration
// =============================================================================

export interface CanaryConfig {
  /** Canary traffic weight steps */
  steps: CanaryStep[];
  /** Traffic routing method */
  trafficRouting: TrafficRoutingConfig;
  /** Analysis configuration */
  analysis?: AnalysisConfig;
  /** Abort on failure */
  abortOnFailure: boolean;
  /** Maximum failed analysis runs before abort */
  maxFailedAnalysisRuns: number;
}

export interface CanaryStep {
  /** Traffic weight for canary (0-100) */
  weight: TrafficWeight;
  /** Duration to hold at this weight (alternative to pauseDuration) */
  pause?: string;
  /** Duration to hold at this weight */
  pauseDuration?: string;
  /** Run analysis at this step */
  analysis?: boolean;
  /** Manual approval required */
  requiresApproval?: boolean;
}

export interface TrafficRoutingConfig {
  /** Traffic routing type */
  type: "nginx" | "istio" | "linkerd" | "traefik" | "aws-alb" | "gcp-neg";
  /** Stable service name */
  stableService: string;
  /** Canary service name */
  canaryService: string;
  /** Traffic split annotation key */
  annotationKey?: string;
  /** Additional routing configuration */
  config?: Record<string, unknown>;
}

// =============================================================================
// A/B Testing Configuration
// =============================================================================

export interface ABTestingConfig {
  /** Experiment name */
  experimentName: string;
  /** Traffic split between variants */
  variants: ABVariant[];
  /** Routing rules */
  routingRules: ABRoutingRule[];
  /** Success metrics */
  successMetrics: string[];
  /** Experiment duration */
  duration: string;
}

export interface ABVariant {
  name: string;
  weight: TrafficWeight;
  version: string;
}

export interface ABRoutingRule {
  /** Header-based routing */
  header?: {
    name: string;
    value: string;
  };
  /** Cookie-based routing */
  cookie?: {
    name: string;
    value: string;
  };
  /** Target variant */
  variant: string;
}

// =============================================================================
// Analysis & Metrics Configuration
// =============================================================================

export interface AnalysisConfig {
  /** Analysis interval */
  interval: string;
  /** Number of analysis runs */
  count: number;
  /** Success threshold percentage */
  successThreshold: number;
  /** Threshold for metric analysis (alternative name) */
  threshold?: number;
  /** Metrics to analyze */
  metrics: MetricConfig[];
  /** Webhook for custom analysis */
  webhooks?: WebhookAnalysisConfig[];
}

export interface MetricConfig {
  /** Metric name */
  name: string;
  /** Prometheus/metrics query */
  query: string;
  /** Expected condition */
  condition: MetricCondition;
  /** Threshold value */
  threshold: number;
  /** Failure limit */
  failureLimit?: number;
}

/** Simple metric type for string-based metric names */
export type SimpleMetric =
  | "error_rate"
  | "latency_p99"
  | "latency_p95"
  | "success_rate";

export type MetricCondition =
  | "less-than"
  | "less-than-or-equal"
  | "greater-than"
  | "greater-than-or-equal"
  | "equal"
  | "not-equal"
  | "in-range";

export interface WebhookAnalysisConfig {
  name: string;
  url: string;
  method: "GET" | "POST";
  headers?: Record<string, string>;
  body?: string;
  successCondition: string;
  timeoutSeconds: number;
}

// =============================================================================
// Health Check Configuration
// =============================================================================

export interface HealthCheckConfig {
  /** Enable liveness probe */
  liveness: ProbeConfig;
  /** Enable readiness probe */
  readiness: ProbeConfig;
  /** Enable startup probe */
  startup?: ProbeConfig;
  /** Custom health endpoint */
  customEndpoint?: string;
  /** Health check interval during deployment */
  deploymentCheckInterval: number;
  /** Maximum unhealthy pods allowed */
  maxUnhealthyPods: number;
}

export interface ProbeConfig {
  /** Probe type */
  type: "http" | "tcp" | "exec";
  /** HTTP path (for http probe) */
  path?: string;
  /** Port number */
  port: number;
  /** Command (for exec probe) */
  command?: string[];
  /** Initial delay in seconds */
  initialDelaySeconds: number;
  /** Period in seconds */
  periodSeconds: number;
  /** Timeout in seconds */
  timeoutSeconds: number;
  /** Success threshold */
  successThreshold: number;
  /** Failure threshold */
  failureThreshold: number;
}

// =============================================================================
// Approval Workflow Configuration
// =============================================================================

export interface ApprovalConfig {
  /** Approval required */
  required: boolean;
  /** Minimum approvers */
  minApprovers: number;
  /** Allowed approvers (user IDs or groups) */
  approvers: string[];
  /** Approval timeout */
  timeout: string;
  /** Auto-approve conditions */
  autoApproveConditions?: AutoApproveCondition[];
  /** Notifications */
  notifyOnRequest: boolean;
  /** Integration with external systems */
  externalApproval?: ExternalApprovalConfig;
}

export interface AutoApproveCondition {
  /** Condition type */
  type: "environment" | "time-window" | "previous-success" | "custom";
  /** Environment names for auto-approve */
  environments?: string[];
  /** Time window for auto-approve (cron format) */
  timeWindow?: string;
  /** Number of previous successful deployments required */
  previousSuccessCount?: number;
  /** Custom condition expression */
  expression?: string;
}

export interface ExternalApprovalConfig {
  /** External system type */
  type: "github" | "jira" | "servicenow" | "pagerduty" | "slack" | "custom";
  /** Endpoint URL */
  url: string;
  /** API credentials secret name */
  secretName: string;
  /** Approval callback URL */
  callbackUrl?: string;
}

// =============================================================================
// Notification Configuration
// =============================================================================

export interface NotificationConfig {
  /** Notification channels */
  channels: NotificationChannel[];
  /** Events to notify */
  events: DeploymentEvent[];
  /** Include deployment details */
  includeDetails: boolean;
}

export interface NotificationChannel {
  type: "slack" | "teams" | "email" | "webhook" | "pagerduty";
  url?: string;
  secretName?: string;
  recipients?: string[];
}

export type DeploymentEvent =
  | "started"
  | "in-progress"
  | "paused"
  | "promoted"
  | "succeeded"
  | "failed"
  | "rolled-back"
  | "approval-required"
  | "approval-received";

// =============================================================================
// Deployment State & History
// =============================================================================

export interface DeploymentState {
  /** Unique ID for this deployment instance */
  id?: string;
  /** Deployment configuration */
  config: DeploymentConfig;
  /** Current status */
  status: DeploymentStatus;
  /** Current phase */
  phase: DeploymentPhase;
  /** Deployment name */
  name?: string;
  /** Target namespace */
  namespace?: string;
  /** Deployment strategy type */
  strategy?: DeploymentStrategy;
  /** Current canary weight (for canary deployments) */
  canaryWeight?: TrafficWeight;
  /** Traffic percentage */
  trafficPercentage?: number;
  /** Current version being deployed */
  currentVersion?: string;
  /** Current step index */
  currentStep: number;
  /** Total steps */
  totalSteps: number;
  /** Health status */
  health: HealthStatus;
  /** Pod status */
  pods: PodStatus[];
  /** Analysis results */
  analysisResults?: AnalysisResult[];
  /** Approval status */
  approvalStatus?: ApprovalStatus;
  /** Error message if failed */
  error?: string;
  /** Start time */
  startedAt?: Date;
  /** End time */
  completedAt?: Date;
  /** Duration in milliseconds */
  duration?: number;
  /** Events timeline */
  events: DeploymentEventRecord[];
}

export interface PodStatus {
  name: string;
  phase: string;
  ready: boolean;
  restarts: number;
  age: string;
  version: string;
  node?: string;
}

export interface AnalysisResult {
  metric: string;
  value: number;
  threshold: number;
  condition: MetricCondition;
  passed: boolean;
  /** Whether the analysis was successful */
  success?: boolean;
  /** Human-readable message */
  message?: string;
  /** Metrics map for multiple metric results */
  metrics?: Record<string, number>;
  timestamp: Date;
}

export interface ApprovalStatus {
  required: boolean;
  approvals: Approval[];
  approved: boolean;
  requestedAt?: Date;
  approvedAt?: Date;
  expiresAt?: Date;
}

export interface Approval {
  approver: string;
  approved: boolean;
  comment?: string;
  timestamp: Date;
}

export interface DeploymentEventRecord {
  type: string;
  message: string;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// Rollback Configuration
// =============================================================================

export interface RollbackConfig {
  /** Enable automatic rollback on failure */
  autoRollback: boolean;
  /** Conditions that trigger automatic rollback */
  triggers: RollbackTrigger[];
  /** Rollback strategy */
  strategy: "immediate" | "gradual";
  /** Number of revisions to keep */
  revisionHistoryLimit: number;
  /** Notification on rollback */
  notifyOnRollback: boolean;
}

export interface RollbackTrigger {
  type:
    | "health-check-failure"
    | "metric-threshold"
    | "error-rate"
    | "timeout"
    | "manual";
  threshold?: number;
  duration?: string;
}

export interface RollbackRequest {
  /** Deployment ID */
  deploymentId: string;
  /** Target revision (optional, defaults to previous) */
  targetRevision?: number;
  /** Reason for rollback */
  reason: string;
  /** Initiated by */
  initiatedBy: string;
  /** Force rollback even if validation fails */
  force?: boolean;
}

export interface RollbackState {
  /** Original deployment ID */
  originalDeploymentId: string;
  /** Rollback deployment ID */
  rollbackDeploymentId: string;
  /** Target version */
  targetVersion: string;
  /** Source version */
  sourceVersion: string;
  /** Status */
  status: "in-progress" | "succeeded" | "failed";
  /** Start time */
  startedAt: Date;
  /** End time */
  completedAt?: Date;
  /** Initiated by */
  initiatedBy: string;
  /** Reason */
  reason: string;
}

// =============================================================================
// GitOps Configuration
// =============================================================================

export interface GitOpsConfig {
  /** GitOps tool */
  tool: "argocd" | "flux" | "custom";
  /** Git repository URL */
  repository: string;
  /** Branch */
  branch: string;
  /** Path in repository */
  path: string;
  /** Sync policy */
  syncPolicy: GitOpsSyncPolicy;
  /** Credentials secret name */
  credentialsSecret?: string;
  /** Auto-sync on change */
  autoSync: boolean;
  /** Prune resources not in git */
  prune: boolean;
  /** Self-heal (reconcile drift) */
  selfHeal: boolean;
}

export interface GitOpsSyncPolicy {
  /** Automated sync */
  automated: boolean;
  /** Allow empty commits */
  allowEmpty: boolean;
  /** Prune orphaned resources */
  prune: boolean;
  /** Self-heal on drift */
  selfHeal: boolean;
  /** Sync options */
  syncOptions?: string[];
  /** Retry configuration */
  retry?: {
    limit: number;
    backoff: {
      duration: string;
      factor: number;
      maxDuration: string;
    };
  };
}

export interface GitOpsApplication {
  name: string;
  namespace: string;
  project: string;
  source: {
    repoURL: string;
    path: string;
    targetRevision: string;
  };
  destination: {
    server: string;
    namespace: string;
  };
  syncPolicy?: GitOpsSyncPolicy;
  status?: GitOpsApplicationStatus;
}

export interface GitOpsApplicationStatus {
  sync: {
    status: "Synced" | "OutOfSync" | "Unknown";
    revision: string;
  };
  health: {
    status:
      | "Healthy"
      | "Degraded"
      | "Progressing"
      | "Suspended"
      | "Missing"
      | "Unknown";
  };
  conditions?: GitOpsCondition[];
}

export interface GitOpsCondition {
  type: string;
  message: string;
  lastTransitionTime: Date;
}

// =============================================================================
// Kubernetes Resource Types
// =============================================================================

export interface KubernetesResource {
  apiVersion: string;
  kind: string;
  metadata: {
    name: string;
    namespace?: string;
    labels?: Record<string, string>;
    annotations?: Record<string, string>;
  };
  spec?: Record<string, unknown>;
}

export interface DeploymentResource extends KubernetesResource {
  kind: "Deployment";
  spec: {
    replicas: number;
    selector: {
      matchLabels: Record<string, string>;
    };
    template: {
      metadata: {
        labels: Record<string, string>;
      };
      spec: {
        containers: ContainerSpec[];
      };
    };
    strategy?: {
      type: "RollingUpdate" | "Recreate";
      rollingUpdate?: {
        maxSurge: number | string;
        maxUnavailable: number | string;
      };
    };
  };
}

export interface ContainerSpec {
  name: string;
  image: string;
  ports?: { containerPort: number; protocol?: string }[];
  env?: { name: string; value?: string; valueFrom?: Record<string, unknown> }[];
  resources?: ResourceConfig;
  livenessProbe?: KubernetesProbe;
  readinessProbe?: KubernetesProbe;
  startupProbe?: KubernetesProbe;
}

export interface KubernetesProbe {
  httpGet?: { path: string; port: number };
  tcpSocket?: { port: number };
  exec?: { command: string[] };
  initialDelaySeconds?: number;
  periodSeconds?: number;
  timeoutSeconds?: number;
  successThreshold?: number;
  failureThreshold?: number;
}

export interface ServiceResource extends KubernetesResource {
  kind: "Service";
  spec: {
    type?: "ClusterIP" | "NodePort" | "LoadBalancer";
    selector: Record<string, string>;
    ports: {
      port: number;
      targetPort: number;
      protocol?: string;
      name?: string;
    }[];
  };
}

// =============================================================================
// Zod Schemas for Validation
// =============================================================================

export const DeploymentConfigSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(100),
  environment: z.string().min(1),
  strategy: z.enum([
    "rolling",
    "blue-green",
    "canary",
    "recreate",
    "a-b-testing",
  ]),
  namespace: z.string().min(1),
  application: z.string().min(1),
  version: z.string().min(1),
  previousVersion: z.string().optional(),
  image: z.string().min(1),
  replicas: z.number().int().min(0).max(100),
  timeout: z.number().optional(),
  createdAt: z.date(),
  createdBy: z.string().min(1),
});

export const CanaryStepSchema = z.object({
  weight: z.number().int().min(0).max(100),
  pauseDuration: z.string().optional(),
  pause: z.string().optional(),
  analysis: z.boolean().optional(),
  requiresApproval: z.boolean().optional(),
});

export const RollbackRequestSchema = z.object({
  deploymentId: z.string().uuid(),
  targetRevision: z.number().int().optional(),
  reason: z.string().min(1),
  initiatedBy: z.string().min(1),
  force: z.boolean().optional(),
});

// =============================================================================
// Additional Types for Orchestrator
// =============================================================================

/** Result of a deployment operation */
export interface DeploymentResult {
  success: boolean;
  deploymentId: string;
  version: string;
  message?: string;
  error?: string;
  duration?: number;
  metrics?: Record<string, number>;
}

/** Deployment environment configuration */
export interface DeploymentEnvironment {
  name: string;
  namespace: string;
  replicas: number;
  image: string;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
}

/** Deployment approval record */
export interface DeploymentApproval {
  deploymentId: string;
  approver: string;
  approved: boolean;
  comment?: string;
  timestamp: Date;
  expiresAt?: Date;
}

// =============================================================================
// Event Types
// =============================================================================

export interface DeploymentEvents {
  "deployment:created": DeploymentConfig;
  "deployment:started": DeploymentState;
  "deployment:progress": DeploymentState;
  "deployment:paused": DeploymentState;
  "deployment:resumed": DeploymentState;
  "deployment:promoted": DeploymentState;
  "deployment:completed": DeploymentState;
  "deployment:awaiting_promotion": DeploymentState;
  "deployment:failed": DeploymentState & { error: string };
  "deployment:cancelled": DeploymentState;
  "rollback:started": RollbackState;
  "rollback:completed": RollbackState;
  "rollback:failed": RollbackState & { error: string };
  "approval:requested": ApprovalStatus;
  "approval:received": Approval;
  "approval:expired": ApprovalStatus;
  "health:changed": { deployment: string; status: HealthStatus };
  "analysis:completed": { deployment: string; results: AnalysisResult[] };
  "traffic:shifted": { deployment: string; weight: TrafficWeight };
}
