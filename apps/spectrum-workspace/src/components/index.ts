/**
 * @fileoverview Main component barrel export
 * @description Centralized exports for all UI components
 */

// Agent Execution Graph - Dust.tt-style workflow visualization
export {
  AgentExecutionGraph,
  mockExecutionNodes,
} from "./agent-graph/AgentExecutionGraph";
export type {
  AgentNode,
  AgentStatus,
  AgentExecutionGraphProps,
} from "./agent-graph/AgentExecutionGraph";

// GitOps Pipeline Overlay - Flux-style deployment visualization
export {
  GitOpsOverlay,
  mockEnvironments,
  mockPipeline,
} from "./gitops/GitOpsOverlay";
export type {
  GitOpsOverlayProps,
  PipelineStage,
  PipelineRun,
  PipelineStageStatus,
  DeploymentEnvironment,
  EnvironmentStatus,
} from "./gitops/GitOpsOverlay";

// Experiment Sidebar - MLflow-style experiment tracking
export {
  ExperimentSidebar,
  mockExperiments,
} from "./experiments/ExperimentSidebar";
export type {
  ExperimentRun,
  Experiment,
  ExperimentSidebarProps,
  RunStatus,
} from "./experiments/ExperimentSidebar";

// Enhanced Status Bar - Multi-segment status display
export {
  EnhancedStatusBar,
  mockMetrics,
  mockAgentActivity,
  mockDeployment,
} from "./status-bar";
export type {
  StatusBarMetrics,
  AgentActivity,
  DeploymentStatus,
  StatusBarProps,
} from "./status-bar";

// Editors
export * from "./editors";

// Re-export motion utilities for convenience
export { MotionDiv, MotionSpan } from "../lib/motion";
