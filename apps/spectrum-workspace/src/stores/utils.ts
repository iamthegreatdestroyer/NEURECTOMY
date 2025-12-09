/**
 * Store Utilities
 *
 * Utility functions for working with stores that avoid circular dependencies.
 */

import { useWorkspaceStore } from "./workspace-store";
import { useAgentStore } from "./agent-store";
import { useContainerStore } from "./container-store";

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
