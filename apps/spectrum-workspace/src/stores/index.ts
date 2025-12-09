/**
 * Store Index
 *
 * Central export point for all Zustand stores in NEURECTOMY.
 */

export { useWorkspaceStore } from "./workspace-store";
export type {
  WorkspaceState,
  WorkspaceLayout,
  TabItem,
  PanelVisibility,
} from "./workspace-store";

export { useAgentStore } from "./agent-store";
export type {
  AgentState,
  AgentNode,
  AgentConnection,
  AgentWorkflow,
  AgentStatus,
  AgentType,
} from "./agent-store";

export { useContainerStore } from "./container-store";
export type {
  ContainerState,
  DockerContainer,
  KubernetesPod,
  KubernetesNode,
  KubernetesService,
  KubernetesCluster,
  ContainerStatus,
  PodStatus,
} from "./container-store";

export { useStores, resetAllStores } from "./utils";
