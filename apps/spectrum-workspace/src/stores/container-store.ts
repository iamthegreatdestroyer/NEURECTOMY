/**
 * Container Store
 *
 * State management for Docker containers and Kubernetes clusters.
 * Handles container lifecycle, K8s topology, and resource monitoring.
 */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { devtools } from "zustand/middleware";

export type ContainerStatus =
  | "running"
  | "stopped"
  | "paused"
  | "exited"
  | "dead";
export type PodStatus =
  | "Running"
  | "Pending"
  | "Succeeded"
  | "Failed"
  | "Unknown";

export interface DockerContainer {
  id: string;
  name: string;
  image: string;
  status: ContainerStatus;
  ports: Array<{ host: number; container: number }>;
  networks: string[];
  cpu: number; // Percentage
  memory: number; // MB
  createdAt: string;
  labels?: Record<string, string>;
}

export interface KubernetesPod {
  id: string;
  name: string;
  namespace: string;
  status: PodStatus;
  nodeName: string;
  ip: string;
  containers: Array<{
    name: string;
    image: string;
    status: ContainerStatus;
  }>;
  cpu: number;
  memory: number;
  restarts: number;
  age: string;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
}

export interface KubernetesNode {
  id: string;
  name: string;
  status: "Ready" | "NotReady" | "Unknown";
  role: "master" | "worker";
  version: string;
  os: string;
  architecture: string;
  cpu: {
    total: number;
    used: number;
  };
  memory: {
    total: number; // GB
    used: number; // GB
  };
  pods: {
    total: number;
    used: number;
  };
  position?: { x: number; y: number; z: number }; // For 3D visualization
}

export interface KubernetesService {
  id: string;
  name: string;
  namespace: string;
  type: "ClusterIP" | "NodePort" | "LoadBalancer" | "ExternalName";
  clusterIP: string;
  externalIP?: string;
  ports: Array<{ port: number; targetPort: number; protocol: string }>;
  selector: Record<string, string>;
}

export interface KubernetesCluster {
  id: string;
  name: string;
  context: string;
  server: string;
  version: string;
  nodes: KubernetesNode[];
  pods: KubernetesPod[];
  services: KubernetesService[];
  namespaces: string[];
}

export interface ContainerState {
  // Docker
  containers: DockerContainer[];
  selectedContainerId: string | null;

  // Kubernetes
  clusters: KubernetesCluster[];
  activeClusterId: string | null;
  selectedPodId: string | null;
  selectedNodeId: string | null;

  // Filters
  namespaceFilter: string;
  statusFilter: string[];

  // 3D View settings
  viewMode: "2d" | "3d";
  showMetrics: boolean;
  showConnections: boolean;

  // Actions - Docker
  addContainer: (container: DockerContainer) => void;
  updateContainer: (
    containerId: string,
    updates: Partial<DockerContainer>
  ) => void;
  removeContainer: (containerId: string) => void;
  selectContainer: (containerId: string | null) => void;

  // Actions - Kubernetes
  addCluster: (cluster: KubernetesCluster) => void;
  updateCluster: (
    clusterId: string,
    updates: Partial<KubernetesCluster>
  ) => void;
  removeCluster: (clusterId: string) => void;
  setActiveCluster: (clusterId: string | null) => void;

  updatePod: (
    clusterId: string,
    podId: string,
    updates: Partial<KubernetesPod>
  ) => void;
  selectPod: (podId: string | null) => void;

  updateNode: (
    clusterId: string,
    nodeId: string,
    updates: Partial<KubernetesNode>
  ) => void;
  selectNode: (nodeId: string | null) => void;

  // Actions - Filters
  setNamespaceFilter: (namespace: string) => void;
  setStatusFilter: (statuses: string[]) => void;

  // Actions - View
  setViewMode: (mode: "2d" | "3d") => void;
  toggleMetrics: () => void;
  toggleConnections: () => void;

  reset: () => void;
}

export const useContainerStore = create<ContainerState>()(
  devtools(
    immer((set) => ({
      // Initial state
      containers: [],
      selectedContainerId: null,
      clusters: [],
      activeClusterId: null,
      selectedPodId: null,
      selectedNodeId: null,
      namespaceFilter: "all",
      statusFilter: [],
      viewMode: "3d",
      showMetrics: true,
      showConnections: true,

      // Docker actions
      addContainer: (container) =>
        set((state) => {
          state.containers.push(container);
        }),

      updateContainer: (containerId, updates) =>
        set((state) => {
          const container = state.containers.find((c) => c.id === containerId);
          if (container) {
            Object.assign(container, updates);
          }
        }),

      removeContainer: (containerId) =>
        set((state) => {
          const index = state.containers.findIndex((c) => c.id === containerId);
          if (index !== -1) {
            state.containers.splice(index, 1);
            if (state.selectedContainerId === containerId) {
              state.selectedContainerId = null;
            }
          }
        }),

      selectContainer: (containerId) =>
        set((state) => {
          state.selectedContainerId = containerId;
        }),

      // Kubernetes actions
      addCluster: (cluster) =>
        set((state) => {
          state.clusters.push(cluster);
          if (!state.activeClusterId) {
            state.activeClusterId = cluster.id;
          }
        }),

      updateCluster: (clusterId, updates) =>
        set((state) => {
          const cluster = state.clusters.find((c) => c.id === clusterId);
          if (cluster) {
            Object.assign(cluster, updates);
          }
        }),

      removeCluster: (clusterId) =>
        set((state) => {
          const index = state.clusters.findIndex((c) => c.id === clusterId);
          if (index !== -1) {
            state.clusters.splice(index, 1);
            if (state.activeClusterId === clusterId) {
              state.activeClusterId = state.clusters[0]?.id || null;
            }
          }
        }),

      setActiveCluster: (clusterId) =>
        set((state) => {
          state.activeClusterId = clusterId;
        }),

      updatePod: (clusterId, podId, updates) =>
        set((state) => {
          const cluster = state.clusters.find((c) => c.id === clusterId);
          if (cluster) {
            const pod = cluster.pods.find((p) => p.id === podId);
            if (pod) {
              Object.assign(pod, updates);
            }
          }
        }),

      selectPod: (podId) =>
        set((state) => {
          state.selectedPodId = podId;
        }),

      updateNode: (clusterId, nodeId, updates) =>
        set((state) => {
          const cluster = state.clusters.find((c) => c.id === clusterId);
          if (cluster) {
            const node = cluster.nodes.find((n) => n.id === nodeId);
            if (node) {
              Object.assign(node, updates);
            }
          }
        }),

      selectNode: (nodeId) =>
        set((state) => {
          state.selectedNodeId = nodeId;
        }),

      // Filter actions
      setNamespaceFilter: (namespace) =>
        set((state) => {
          state.namespaceFilter = namespace;
        }),

      setStatusFilter: (statuses) =>
        set((state) => {
          state.statusFilter = statuses;
        }),

      // View actions
      setViewMode: (mode) =>
        set((state) => {
          state.viewMode = mode;
        }),

      toggleMetrics: () =>
        set((state) => {
          state.showMetrics = !state.showMetrics;
        }),

      toggleConnections: () =>
        set((state) => {
          state.showConnections = !state.showConnections;
        }),

      reset: () =>
        set({
          containers: [],
          selectedContainerId: null,
          clusters: [],
          activeClusterId: null,
          selectedPodId: null,
          selectedNodeId: null,
          namespaceFilter: "all",
          statusFilter: [],
          viewMode: "3d",
          showMetrics: true,
          showConnections: true,
        }),
    })),
    { name: "ContainerStore" }
  )
);
