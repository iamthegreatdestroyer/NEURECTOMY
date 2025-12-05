/**
 * NEURECTOMY Deployment Orchestrator
 *
 * Comprehensive deployment orchestration system supporting:
 * - Rolling updates with configurable surge/unavailability
 * - Blue-green deployments with instant traffic switching
 * - Canary releases with progressive traffic shifting
 * - Approval workflows with timeout and escalation
 * - Automatic rollback with state preservation
 * - GitOps integration (ArgoCD & Flux)
 *
 * @packageDocumentation
 */

// Types
export * from "./types";

// Kubernetes Client
export {
  KubernetesClient,
  createKubernetesClient,
  type KubernetesClientConfig,
} from "./kubernetes/client";

// Deployment Strategies
export {
  RollingUpdateStrategy,
  BlueGreenStrategy,
  CanaryStrategy,
} from "./strategies";

// Approval Workflow
export {
  ApprovalWorkflowManager,
  createApprovalWorkflow,
  type ApprovalRequest,
  type ApprovalWorkflowEvents,
  type ApprovalPolicy,
  type ApprovalDecision,
  type ApprovalWorkflowConfig,
} from "./approval/workflow";

// Rollback Manager
export {
  RollbackManager,
  createRollbackManager,
  type RollbackTarget,
  type RollbackHistoryEntry,
  type RollbackPolicy,
  type RollbackEvents,
  type RollbackManagerConfig,
} from "./rollback/manager";

// GitOps Integrations
export {
  ArgoCDClient,
  createArgoCDClient,
  type ArgoCDConfig,
  type ArgoApplication,
  type ArgoApplicationStatus,
  type ArgoCDEvents,
} from "./gitops/argocd";

export {
  FluxClient,
  createFluxClient,
  type FluxClientConfig,
  type FluxGitRepository,
  type FluxKustomization,
  type FluxHelmRelease,
  type FluxResourceStatus,
  type FluxEvents,
} from "./gitops/flux";

// Main Orchestrator
export {
  DeploymentOrchestrator,
  createDeploymentOrchestrator,
  type OrchestratorConfig,
  type StatePersistenceAdapter,
  type DeploymentPlan,
  type DeploymentStep,
  type DeploymentRisk,
  type OrchestratorEvents,
} from "./orchestrator";

// Re-export key types for convenience
export type {
  DeploymentConfig,
  DeploymentState,
  DeploymentResult,
  DeploymentStrategy,
  DeploymentEnvironment,
  DeploymentApproval,
  HealthCheckConfig,
  RollingConfig,
  BlueGreenConfig,
  CanaryConfig,
  TrafficWeight,
} from "./types";
