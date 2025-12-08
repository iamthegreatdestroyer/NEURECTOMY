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

// Utility hook for accessing all stores
export function useStores() {
  const workspace = useWorkspaceStore();
  const agent = useAgentStore();
  const container = useContainerStore();

  return {
    workspace,
    agent,
    container,
  };
}

// Utility function to reset all stores (useful for testing or logout)
export function resetAllStores() {
  useWorkspaceStore.getState().reset();
  useAgentStore.getState().reset();
  useContainerStore.getState().reset();
}
