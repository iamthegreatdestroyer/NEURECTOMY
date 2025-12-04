/**
 * @file Graph Store
 * @description Zustand store for graph state management
 * @module @neurectomy/3d-engine/visualization/graph
 * @agents @APEX @CANVAS
 */

import { create, type StoreApi, type UseBoundStore } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type {
  GraphNode,
  GraphEdge,
  GraphConfig,
  GraphPhysicsConfig,
  GraphVisualConfig,
  GraphInteractionConfig,
  GraphLayoutConfig,
  DEFAULT_GRAPH_CONFIG,
} from "./types";

// ============================================================================
// Store Types
// ============================================================================

/**
 * Selection state
 */
export interface SelectionState {
  /** Selected node IDs */
  nodeIds: Set<string>;
  /** Selected edge IDs */
  edgeIds: Set<string>;
  /** Primary selected node (for multi-selection) */
  primaryNodeId: string | null;
  /** Primary selected edge */
  primaryEdgeId: string | null;
}

/**
 * Hover state
 */
export interface HoverState {
  /** Hovered node ID */
  nodeId: string | null;
  /** Hovered edge ID */
  edgeId: string | null;
}

/**
 * Drag state
 */
export interface DragState {
  /** Is dragging active */
  active: boolean;
  /** Dragged node ID */
  nodeId: string | null;
  /** Start position */
  startPosition: { x: number; y: number; z: number } | null;
  /** Current position */
  currentPosition: { x: number; y: number; z: number } | null;
}

/**
 * Simulation state
 */
export interface SimulationState {
  /** Is simulation running */
  running: boolean;
  /** Alpha value (simulation energy) */
  alpha: number;
  /** Simulation tick count */
  tickCount: number;
  /** Last tick timestamp */
  lastTick: number;
  /** Average ms per tick */
  avgTickTime: number;
}

/**
 * Camera state
 */
export interface CameraState {
  /** Camera position */
  position: { x: number; y: number; z: number };
  /** Camera target/lookAt */
  target: { x: number; y: number; z: number };
  /** Camera zoom level */
  zoom: number;
  /** Field of view */
  fov: number;
}

/**
 * Graph store state
 */
export interface GraphState {
  // Data
  nodes: Map<string, GraphNode>;
  edges: Map<string, GraphEdge>;

  // Computed indices (for performance)
  nodesByType: Map<string, Set<string>>;
  edgesBySource: Map<string, Set<string>>;
  edgesByTarget: Map<string, Set<string>>;

  // Configuration
  config: GraphConfig;

  // UI State
  selection: SelectionState;
  hover: HoverState;
  drag: DragState;
  simulation: SimulationState;
  camera: CameraState;

  // View state
  viewMode: "normal" | "blueprint" | "performance" | "debug";
  showLabels: boolean;
  showEdges: boolean;
  highlightedNodeIds: Set<string>;
  dimmedNodeIds: Set<string>;
}

/**
 * Graph store actions
 */
export interface GraphActions {
  // Node operations
  addNode: (node: GraphNode) => void;
  addNodes: (nodes: GraphNode[]) => void;
  updateNode: (id: string, updates: Partial<GraphNode>) => void;
  updateNodePosition: (
    id: string,
    position: { x: number; y: number; z: number }
  ) => void;
  removeNode: (id: string) => void;
  removeNodes: (ids: string[]) => void;

  // Edge operations
  addEdge: (edge: GraphEdge) => void;
  addEdges: (edges: GraphEdge[]) => void;
  updateEdge: (id: string, updates: Partial<GraphEdge>) => void;
  removeEdge: (id: string) => void;
  removeEdges: (ids: string[]) => void;

  // Bulk operations
  setGraph: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  clearGraph: () => void;

  // Selection operations
  selectNode: (id: string, addToSelection?: boolean) => void;
  selectNodes: (ids: string[], replace?: boolean) => void;
  deselectNode: (id: string) => void;
  selectEdge: (id: string, addToSelection?: boolean) => void;
  selectEdges: (ids: string[], replace?: boolean) => void;
  deselectEdge: (id: string) => void;
  clearSelection: () => void;
  selectAll: () => void;

  // Hover operations
  setHoveredNode: (id: string | null) => void;
  setHoveredEdge: (id: string | null) => void;

  // Drag operations
  startDrag: (
    nodeId: string,
    position: { x: number; y: number; z: number }
  ) => void;
  updateDrag: (position: { x: number; y: number; z: number }) => void;
  endDrag: () => void;

  // Simulation control
  startSimulation: () => void;
  stopSimulation: () => void;
  toggleSimulation: () => void;
  tickSimulation: (deltaTime: number) => void;
  resetSimulation: () => void;

  // Camera control
  setCameraPosition: (position: { x: number; y: number; z: number }) => void;
  setCameraTarget: (target: { x: number; y: number; z: number }) => void;
  setCameraZoom: (zoom: number) => void;
  resetCamera: () => void;
  focusOnNode: (id: string) => void;
  focusOnSelection: () => void;
  fitToView: () => void;

  // Configuration
  updateConfig: (config: Partial<GraphConfig>) => void;
  updatePhysicsConfig: (config: Partial<GraphPhysicsConfig>) => void;
  updateVisualConfig: (config: Partial<GraphVisualConfig>) => void;
  updateInteractionConfig: (config: Partial<GraphInteractionConfig>) => void;
  updateLayoutConfig: (config: Partial<GraphLayoutConfig>) => void;

  // View mode
  setViewMode: (mode: GraphState["viewMode"]) => void;
  toggleLabels: () => void;
  toggleEdges: () => void;

  // Highlighting
  highlightNodes: (ids: string[]) => void;
  dimNodes: (ids: string[]) => void;
  clearHighlights: () => void;

  // Pin operations
  pinNode: (id: string) => void;
  unpinNode: (id: string) => void;
  pinSelected: () => void;
  unpinAll: () => void;

  // Query helpers
  getNode: (id: string) => GraphNode | undefined;
  getEdge: (id: string) => GraphEdge | undefined;
  getNodeEdges: (nodeId: string) => GraphEdge[];
  getConnectedNodes: (nodeId: string) => GraphNode[];
  getSelectedNodes: () => GraphNode[];
  getSelectedEdges: () => GraphEdge[];
}

/**
 * Combined state and actions type
 */
export type GraphStore = GraphState & GraphActions;

// Create a temporary store to infer the type
const _tempStore = create<GraphStore>()(
  subscribeWithSelector(immer(() => ({}) as GraphStore))
);

/**
 * Zustand store hook type - inferred from createGraphStore
 */
export type GraphStoreHook = typeof _tempStore;

// ============================================================================
// Initial State
// ============================================================================

const initialState: GraphState = {
  nodes: new Map(),
  edges: new Map(),
  nodesByType: new Map(),
  edgesBySource: new Map(),
  edgesByTarget: new Map(),
  config: {
    physics: {
      enabled: true,
      gravity: -0.1,
      repulsion: 100,
      attraction: 0.01,
      damping: 0.9,
      maxVelocity: 10,
      timeStep: 1 / 60,
      iterations: 1,
      alphaDecay: 0.01,
      velocityDecay: 0.4,
      collisionRadius: 1.5,
    },
    visual: {
      node: {
        defaultRadius: 0.5,
        minRadius: 0.2,
        maxRadius: 2.0,
        defaultColor: "#4a90d9",
        selectedColor: "#ffd700",
        hoveredColor: "#7fb3e8",
        opacity: 1.0,
        metalness: 0.3,
        roughness: 0.7,
        showLabels: true,
        labelFontSize: 12,
        labelOffset: 0.8,
      },
      edge: {
        defaultWidth: 0.02,
        minWidth: 0.01,
        maxWidth: 0.1,
        defaultColor: "#888888",
        selectedColor: "#ffd700",
        hoveredColor: "#aaaaaa",
        opacity: 0.8,
        showArrows: true,
        arrowSize: 0.1,
        curveFactor: 0.2,
        animated: false,
        animationSpeed: 1.0,
      },
      backgroundColor: "#1a1a2e",
      bloom: true,
      bloomIntensity: 0.5,
      shadows: true,
      antialias: true,
      ambientLight: 0.4,
    },
    interaction: {
      enableDrag: true,
      enableSelection: true,
      enableMultiSelection: true,
      enableZoom: true,
      enablePan: true,
      enableRotation: true,
      doubleClickBehavior: "focus",
      rightClickBehavior: "menu",
      hoverDelay: 200,
      dragThreshold: 5,
    },
    layout: {
      algorithm: "force-directed",
      bounds: {
        minX: -50,
        maxX: 50,
        minY: -50,
        maxY: 50,
        minZ: -50,
        maxZ: 50,
      },
      centerOnInit: true,
      fitOnInit: true,
      padding: 10,
    },
  },
  selection: {
    nodeIds: new Set(),
    edgeIds: new Set(),
    primaryNodeId: null,
    primaryEdgeId: null,
  },
  hover: {
    nodeId: null,
    edgeId: null,
  },
  drag: {
    active: false,
    nodeId: null,
    startPosition: null,
    currentPosition: null,
  },
  simulation: {
    running: true,
    alpha: 1,
    tickCount: 0,
    lastTick: 0,
    avgTickTime: 0,
  },
  camera: {
    position: { x: 10, y: 10, z: 10 },
    target: { x: 0, y: 0, z: 0 },
    zoom: 1,
    fov: 45,
  },
  viewMode: "normal",
  showLabels: true,
  showEdges: true,
  highlightedNodeIds: new Set(),
  dimmedNodeIds: new Set(),
};

// ============================================================================
// Store Creation
// ============================================================================

/**
 * Create a new graph store instance
 * @returns Zustand store hook for graph state management
 */
export const createGraphStore = (): GraphStoreHook =>
  create<GraphStore>()(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,

        // ====================================================================
        // Node Operations
        // ====================================================================

        addNode: (node: GraphNode) =>
          set((state: GraphState) => {
            state.nodes.set(node.id, node);

            // Update type index
            if (!state.nodesByType.has(node.type)) {
              state.nodesByType.set(node.type, new Set());
            }
            state.nodesByType.get(node.type)!.add(node.id);
          }),

        addNodes: (nodes: GraphNode[]) =>
          set((state: GraphState) => {
            for (const node of nodes) {
              state.nodes.set(node.id, node);

              if (!state.nodesByType.has(node.type)) {
                state.nodesByType.set(node.type, new Set());
              }
              state.nodesByType.get(node.type)!.add(node.id);
            }
          }),

        updateNode: (id: string, updates: Partial<GraphNode>) =>
          set((state: GraphState) => {
            const node = state.nodes.get(id);
            if (node) {
              Object.assign(node, updates);
            }
          }),

        updateNodePosition: (
          id: string,
          position: { x: number; y: number; z: number }
        ) =>
          set((state: GraphState) => {
            const node = state.nodes.get(id);
            if (node) {
              node.position = position;
              node.velocity = { x: 0, y: 0, z: 0 };
            }
          }),

        removeNode: (id: string) =>
          set((state: GraphState) => {
            const node = state.nodes.get(id);
            if (node) {
              // Remove from type index
              state.nodesByType.get(node.type)?.delete(id);

              // Remove connected edges
              const edgesToRemove: string[] = [];
              state.edges.forEach((edge: GraphEdge, edgeId: string) => {
                if (edge.sourceId === id || edge.targetId === id) {
                  edgesToRemove.push(edgeId);
                }
              });

              for (const edgeId of edgesToRemove) {
                const edge = state.edges.get(edgeId);
                if (edge) {
                  state.edgesBySource.get(edge.sourceId)?.delete(edgeId);
                  state.edgesByTarget.get(edge.targetId)?.delete(edgeId);
                  state.edges.delete(edgeId);
                }
              }

              // Remove node
              state.nodes.delete(id);

              // Clean selection
              state.selection.nodeIds.delete(id);
              if (state.selection.primaryNodeId === id) {
                state.selection.primaryNodeId = null;
              }
            }
          }),

        removeNodes: (ids: string[]) =>
          set((state: GraphState) => {
            for (const id of ids) {
              const node = state.nodes.get(id);
              if (node) {
                state.nodesByType.get(node.type)?.delete(id);
                state.nodes.delete(id);
                state.selection.nodeIds.delete(id);
              }
            }

            // Remove orphaned edges
            const nodesToRemove = new Set(ids);
            state.edges.forEach((edge: GraphEdge, edgeId: string) => {
              if (
                nodesToRemove.has(edge.sourceId) ||
                nodesToRemove.has(edge.targetId)
              ) {
                state.edgesBySource.get(edge.sourceId)?.delete(edgeId);
                state.edgesByTarget.get(edge.targetId)?.delete(edgeId);
                state.edges.delete(edgeId);
              }
            });
          }),

        // ====================================================================
        // Edge Operations
        // ====================================================================

        addEdge: (edge: GraphEdge) =>
          set((state: GraphState) => {
            state.edges.set(edge.id, edge);

            // Update source index
            if (!state.edgesBySource.has(edge.sourceId)) {
              state.edgesBySource.set(edge.sourceId, new Set());
            }
            state.edgesBySource.get(edge.sourceId)!.add(edge.id);

            // Update target index
            if (!state.edgesByTarget.has(edge.targetId)) {
              state.edgesByTarget.set(edge.targetId, new Set());
            }
            state.edgesByTarget.get(edge.targetId)!.add(edge.id);
          }),

        addEdges: (edges: GraphEdge[]) =>
          set((state: GraphState) => {
            for (const edge of edges) {
              state.edges.set(edge.id, edge);

              if (!state.edgesBySource.has(edge.sourceId)) {
                state.edgesBySource.set(edge.sourceId, new Set());
              }
              state.edgesBySource.get(edge.sourceId)!.add(edge.id);

              if (!state.edgesByTarget.has(edge.targetId)) {
                state.edgesByTarget.set(edge.targetId, new Set());
              }
              state.edgesByTarget.get(edge.targetId)!.add(edge.id);
            }
          }),

        updateEdge: (id: string, updates: Partial<GraphEdge>) =>
          set((state: GraphState) => {
            const edge = state.edges.get(id);
            if (edge) {
              Object.assign(edge, updates);
            }
          }),

        removeEdge: (id: string) =>
          set((state: GraphState) => {
            const edge = state.edges.get(id);
            if (edge) {
              state.edgesBySource.get(edge.sourceId)?.delete(id);
              state.edgesByTarget.get(edge.targetId)?.delete(id);
              state.edges.delete(id);
              state.selection.edgeIds.delete(id);
            }
          }),

        removeEdges: (ids: string[]) =>
          set((state: GraphState) => {
            for (const id of ids) {
              const edge = state.edges.get(id);
              if (edge) {
                state.edgesBySource.get(edge.sourceId)?.delete(id);
                state.edgesByTarget.get(edge.targetId)?.delete(id);
                state.edges.delete(id);
                state.selection.edgeIds.delete(id);
              }
            }
          }),

        // ====================================================================
        // Bulk Operations
        // ====================================================================

        setGraph: (nodes: GraphNode[], edges: GraphEdge[]) =>
          set((state: GraphState) => {
            // Clear existing
            state.nodes.clear();
            state.edges.clear();
            state.nodesByType.clear();
            state.edgesBySource.clear();
            state.edgesByTarget.clear();
            state.selection.nodeIds.clear();
            state.selection.edgeIds.clear();

            // Add nodes
            for (const node of nodes) {
              state.nodes.set(node.id, node);

              if (!state.nodesByType.has(node.type)) {
                state.nodesByType.set(node.type, new Set());
              }
              state.nodesByType.get(node.type)!.add(node.id);
            }

            // Add edges
            for (const edge of edges) {
              state.edges.set(edge.id, edge);

              if (!state.edgesBySource.has(edge.sourceId)) {
                state.edgesBySource.set(edge.sourceId, new Set());
              }
              state.edgesBySource.get(edge.sourceId)!.add(edge.id);

              if (!state.edgesByTarget.has(edge.targetId)) {
                state.edgesByTarget.set(edge.targetId, new Set());
              }
              state.edgesByTarget.get(edge.targetId)!.add(edge.id);
            }

            // Reset simulation
            state.simulation.alpha = 1;
            state.simulation.tickCount = 0;
          }),

        clearGraph: () =>
          set((state: GraphState) => {
            state.nodes.clear();
            state.edges.clear();
            state.nodesByType.clear();
            state.edgesBySource.clear();
            state.edgesByTarget.clear();
            state.selection.nodeIds.clear();
            state.selection.edgeIds.clear();
            state.selection.primaryNodeId = null;
            state.selection.primaryEdgeId = null;
          }),

        // ====================================================================
        // Selection Operations
        // ====================================================================

        selectNode: (id: string, addToSelection = false) =>
          set((state: GraphState) => {
            if (!addToSelection) {
              state.selection.nodeIds.clear();
              state.selection.edgeIds.clear();
            }
            state.selection.nodeIds.add(id);
            state.selection.primaryNodeId = id;

            // Update node state
            const node = state.nodes.get(id);
            if (node) {
              node.state.selected = true;
            }
          }),

        selectNodes: (ids: string[], replace = true) =>
          set((state: GraphState) => {
            if (replace) {
              // Deselect current
              state.selection.nodeIds.forEach((nodeId: string) => {
                const node = state.nodes.get(nodeId);
                if (node) node.state.selected = false;
              });
              state.selection.nodeIds.clear();
            }

            for (const id of ids) {
              state.selection.nodeIds.add(id);
              const node = state.nodes.get(id);
              if (node) node.state.selected = true;
            }

            state.selection.primaryNodeId = ids[0] ?? null;
          }),

        deselectNode: (id: string) =>
          set((state: GraphState) => {
            state.selection.nodeIds.delete(id);
            const node = state.nodes.get(id);
            if (node) node.state.selected = false;

            if (state.selection.primaryNodeId === id) {
              state.selection.primaryNodeId =
                state.selection.nodeIds.values().next().value ?? null;
            }
          }),

        selectEdge: (id: string, addToSelection = false) =>
          set((state: GraphState) => {
            if (!addToSelection) {
              state.selection.edgeIds.clear();
            }
            state.selection.edgeIds.add(id);
            state.selection.primaryEdgeId = id;

            const edge = state.edges.get(id);
            if (edge) edge.state.selected = true;
          }),

        selectEdges: (ids: string[], replace = true) =>
          set((state: GraphState) => {
            if (replace) {
              state.selection.edgeIds.forEach((edgeId: string) => {
                const edge = state.edges.get(edgeId);
                if (edge) edge.state.selected = false;
              });
              state.selection.edgeIds.clear();
            }

            for (const id of ids) {
              state.selection.edgeIds.add(id);
              const edge = state.edges.get(id);
              if (edge) edge.state.selected = true;
            }

            state.selection.primaryEdgeId = ids[0] ?? null;
          }),

        deselectEdge: (id: string) =>
          set((state: GraphState) => {
            state.selection.edgeIds.delete(id);
            const edge = state.edges.get(id);
            if (edge) edge.state.selected = false;

            if (state.selection.primaryEdgeId === id) {
              state.selection.primaryEdgeId =
                state.selection.edgeIds.values().next().value ?? null;
            }
          }),

        clearSelection: () =>
          set((state: GraphState) => {
            state.selection.nodeIds.forEach((nodeId: string) => {
              const node = state.nodes.get(nodeId);
              if (node) node.state.selected = false;
            });
            state.selection.edgeIds.forEach((edgeId: string) => {
              const edge = state.edges.get(edgeId);
              if (edge) edge.state.selected = false;
            });

            state.selection.nodeIds.clear();
            state.selection.edgeIds.clear();
            state.selection.primaryNodeId = null;
            state.selection.primaryEdgeId = null;
          }),

        selectAll: () =>
          set((state: GraphState) => {
            state.nodes.forEach((node: GraphNode, id: string) => {
              state.selection.nodeIds.add(id);
              node.state.selected = true;
            });
            state.edges.forEach((edge: GraphEdge, id: string) => {
              state.selection.edgeIds.add(id);
              edge.state.selected = true;
            });
          }),

        // ====================================================================
        // Hover Operations
        // ====================================================================

        setHoveredNode: (id: string | null) =>
          set((state: GraphState) => {
            // Clear previous hover
            if (state.hover.nodeId) {
              const prevNode = state.nodes.get(state.hover.nodeId);
              if (prevNode) prevNode.state.hovered = false;
            }

            state.hover.nodeId = id;

            if (id) {
              const node = state.nodes.get(id);
              if (node) node.state.hovered = true;
            }
          }),

        setHoveredEdge: (id: string | null) =>
          set((state: GraphState) => {
            if (state.hover.edgeId) {
              const prevEdge = state.edges.get(state.hover.edgeId);
              if (prevEdge) prevEdge.state.hovered = false;
            }

            state.hover.edgeId = id;

            if (id) {
              const edge = state.edges.get(id);
              if (edge) edge.state.hovered = true;
            }
          }),

        // ====================================================================
        // Drag Operations
        // ====================================================================

        startDrag: (
          nodeId: string,
          position: { x: number; y: number; z: number }
        ) =>
          set((state: GraphState) => {
            state.drag.active = true;
            state.drag.nodeId = nodeId;
            state.drag.startPosition = position;
            state.drag.currentPosition = position;

            const node = state.nodes.get(nodeId);
            if (node) {
              node.state.dragging = true;
              node.pinned = true;
            }
          }),

        updateDrag: (position: { x: number; y: number; z: number }) =>
          set((state: GraphState) => {
            if (state.drag.active && state.drag.nodeId) {
              state.drag.currentPosition = position;

              const node = state.nodes.get(state.drag.nodeId);
              if (node) {
                node.position = position;
                node.velocity = { x: 0, y: 0, z: 0 };
              }
            }
          }),

        endDrag: () =>
          set((state: GraphState) => {
            if (state.drag.nodeId) {
              const node = state.nodes.get(state.drag.nodeId);
              if (node) {
                node.state.dragging = false;
              }
            }

            state.drag.active = false;
            state.drag.nodeId = null;
            state.drag.startPosition = null;
            state.drag.currentPosition = null;
          }),

        // ====================================================================
        // Simulation Control
        // ====================================================================

        startSimulation: () =>
          set((state: GraphState) => {
            state.simulation.running = true;
          }),

        stopSimulation: () =>
          set((state: GraphState) => {
            state.simulation.running = false;
          }),

        toggleSimulation: () =>
          set((state: GraphState) => {
            state.simulation.running = !state.simulation.running;
          }),

        tickSimulation: (deltaTime: number) =>
          set((state: GraphState) => {
            if (!state.simulation.running || !state.config.physics.enabled) {
              return;
            }

            const config = state.config.physics;
            const startTime = performance.now();

            // Apply forces and update positions
            const nodes = Array.from(state.nodes.values());
            const edges = Array.from(state.edges.values());

            for (let iter = 0; iter < config.iterations; iter++) {
              // Apply repulsion forces
              for (let i = 0; i < nodes.length; i++) {
                const nodeA = nodes[i]!;
                if (nodeA.pinned) continue;

                for (let j = i + 1; j < nodes.length; j++) {
                  const nodeB = nodes[j]!;

                  const dx = nodeB.position.x - nodeA.position.x;
                  const dy = nodeB.position.y - nodeA.position.y;
                  const dz = nodeB.position.z - nodeA.position.z;

                  const distSq = dx * dx + dy * dy + dz * dz + 0.01;
                  const dist = Math.sqrt(distSq);

                  const force =
                    (config.repulsion * state.simulation.alpha) / distSq;

                  const fx = (dx / dist) * force;
                  const fy = (dy / dist) * force;
                  const fz = (dz / dist) * force;

                  if (!nodeA.pinned) {
                    nodeA.velocity.x -= fx / nodeA.mass;
                    nodeA.velocity.y -= fy / nodeA.mass;
                    nodeA.velocity.z -= fz / nodeA.mass;
                  }

                  if (!nodeB.pinned) {
                    nodeB.velocity.x += fx / nodeB.mass;
                    nodeB.velocity.y += fy / nodeB.mass;
                    nodeB.velocity.z += fz / nodeB.mass;
                  }
                }
              }

              // Apply attraction forces from edges
              for (const edge of edges) {
                const source = state.nodes.get(edge.sourceId);
                const target = state.nodes.get(edge.targetId);
                if (!source || !target) continue;

                const dx = target.position.x - source.position.x;
                const dy = target.position.y - source.position.y;
                const dz = target.position.z - source.position.z;

                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;
                const stretch = dist - edge.length;

                const force =
                  stretch *
                  config.attraction *
                  edge.weight *
                  state.simulation.alpha;

                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                const fz = (dz / dist) * force;

                if (!source.pinned) {
                  source.velocity.x += fx / source.mass;
                  source.velocity.y += fy / source.mass;
                  source.velocity.z += fz / source.mass;
                }

                if (!target.pinned) {
                  target.velocity.x -= fx / target.mass;
                  target.velocity.y -= fy / target.mass;
                  target.velocity.z -= fz / target.mass;
                }
              }

              // Apply gravity
              for (const node of nodes) {
                if (!node.pinned) {
                  node.velocity.y += config.gravity * state.simulation.alpha;
                }
              }

              // Update positions with velocity
              for (const node of nodes) {
                if (node.pinned) continue;

                // Apply damping
                node.velocity.x *= config.damping;
                node.velocity.y *= config.damping;
                node.velocity.z *= config.damping;

                // Clamp velocity
                const speed = Math.sqrt(
                  node.velocity.x ** 2 +
                    node.velocity.y ** 2 +
                    node.velocity.z ** 2
                );

                if (speed > config.maxVelocity) {
                  const scale = config.maxVelocity / speed;
                  node.velocity.x *= scale;
                  node.velocity.y *= scale;
                  node.velocity.z *= scale;
                }

                // Update position
                node.position.x += node.velocity.x * config.timeStep;
                node.position.y += node.velocity.y * config.timeStep;
                node.position.z += node.velocity.z * config.timeStep;
              }
            }

            // Decay alpha
            state.simulation.alpha *= 1 - config.alphaDecay;

            // Update stats
            const endTime = performance.now();
            state.simulation.tickCount++;
            state.simulation.lastTick = endTime;
            state.simulation.avgTickTime =
              state.simulation.avgTickTime * 0.9 + (endTime - startTime) * 0.1;

            // Stop if converged
            if (state.simulation.alpha < 0.001) {
              state.simulation.running = false;
            }
          }),

        resetSimulation: () =>
          set((state: GraphState) => {
            state.simulation.alpha = 1;
            state.simulation.tickCount = 0;
            state.simulation.running = true;

            // Reset velocities
            state.nodes.forEach((node: GraphNode) => {
              node.velocity = { x: 0, y: 0, z: 0 };
            });
          }),

        // ====================================================================
        // Camera Control
        // ====================================================================

        setCameraPosition: (position: { x: number; y: number; z: number }) =>
          set((state: GraphState) => {
            state.camera.position = position;
          }),

        setCameraTarget: (target: { x: number; y: number; z: number }) =>
          set((state: GraphState) => {
            state.camera.target = target;
          }),

        setCameraZoom: (zoom: number) =>
          set((state: GraphState) => {
            state.camera.zoom = Math.max(0.1, Math.min(10, zoom));
          }),

        resetCamera: () =>
          set((state: GraphState) => {
            state.camera = {
              position: { x: 10, y: 10, z: 10 },
              target: { x: 0, y: 0, z: 0 },
              zoom: 1,
              fov: 45,
            };
          }),

        focusOnNode: (id: string) => {
          const state = get();
          const node = state.nodes.get(id);
          if (node) {
            set((s: GraphState) => {
              s.camera.target = { ...node.position };
            });
          }
        },

        focusOnSelection: () => {
          const state = get();
          if (state.selection.nodeIds.size === 0) return;

          const selectedNodes = Array.from(state.selection.nodeIds)
            .map((id: string) => state.nodes.get(id))
            .filter(Boolean) as GraphNode[];

          // Calculate centroid
          const centroid = { x: 0, y: 0, z: 0 };
          for (const node of selectedNodes) {
            centroid.x += node.position.x;
            centroid.y += node.position.y;
            centroid.z += node.position.z;
          }
          centroid.x /= selectedNodes.length;
          centroid.y /= selectedNodes.length;
          centroid.z /= selectedNodes.length;

          set((s: GraphState) => {
            s.camera.target = centroid;
          });
        },

        fitToView: () => {
          const state = get();
          if (state.nodes.size === 0) return;

          const nodes = Array.from(state.nodes.values());

          // Calculate bounding box
          let minX = Infinity,
            maxX = -Infinity;
          let minY = Infinity,
            maxY = -Infinity;
          let minZ = Infinity,
            maxZ = -Infinity;

          for (const node of nodes) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
          }

          const center = {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2,
          };

          const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
          const distance = size * 2;

          set((s: GraphState) => {
            s.camera.target = center;
            s.camera.position = {
              x: center.x + distance,
              y: center.y + distance * 0.5,
              z: center.z + distance,
            };
          });
        },

        // ====================================================================
        // Configuration
        // ====================================================================

        updateConfig: (config: Partial<GraphConfig>) =>
          set((state: GraphState) => {
            Object.assign(state.config, config);
          }),

        updatePhysicsConfig: (config: Partial<GraphPhysicsConfig>) =>
          set((state: GraphState) => {
            Object.assign(state.config.physics, config);
          }),

        updateVisualConfig: (config: Partial<GraphVisualConfig>) =>
          set((state: GraphState) => {
            Object.assign(state.config.visual, config);
          }),

        updateInteractionConfig: (config: Partial<GraphInteractionConfig>) =>
          set((state: GraphState) => {
            Object.assign(state.config.interaction, config);
          }),

        updateLayoutConfig: (config: Partial<GraphLayoutConfig>) =>
          set((state: GraphState) => {
            Object.assign(state.config.layout, config);
          }),

        // ====================================================================
        // View Mode
        // ====================================================================

        setViewMode: (mode: GraphState["viewMode"]) =>
          set((state: GraphState) => {
            state.viewMode = mode;
          }),

        toggleLabels: () =>
          set((state: GraphState) => {
            state.showLabels = !state.showLabels;
          }),

        toggleEdges: () =>
          set((state: GraphState) => {
            state.showEdges = !state.showEdges;
          }),

        // ====================================================================
        // Highlighting
        // ====================================================================

        highlightNodes: (ids: string[]) =>
          set((state: GraphState) => {
            // Clear previous
            state.highlightedNodeIds.forEach((nodeId: string) => {
              const node = state.nodes.get(nodeId);
              if (node) node.state.highlighted = false;
            });
            state.highlightedNodeIds.clear();

            // Set new
            for (const id of ids) {
              state.highlightedNodeIds.add(id);
              const node = state.nodes.get(id);
              if (node) node.state.highlighted = true;
            }
          }),

        dimNodes: (ids: string[]) =>
          set((state: GraphState) => {
            state.dimmedNodeIds.clear();

            for (const id of ids) {
              state.dimmedNodeIds.add(id);
              const node = state.nodes.get(id);
              if (node) node.state.dimmed = true;
            }
          }),

        clearHighlights: () =>
          set((state: GraphState) => {
            state.highlightedNodeIds.forEach((nodeId: string) => {
              const node = state.nodes.get(nodeId);
              if (node) node.state.highlighted = false;
            });
            state.dimmedNodeIds.forEach((nodeId: string) => {
              const node = state.nodes.get(nodeId);
              if (node) node.state.dimmed = false;
            });
            state.highlightedNodeIds.clear();
            state.dimmedNodeIds.clear();
          }),

        // ====================================================================
        // Pin Operations
        // ====================================================================

        pinNode: (id: string) =>
          set((state: GraphState) => {
            const node = state.nodes.get(id);
            if (node) {
              node.pinned = true;
              node.velocity = { x: 0, y: 0, z: 0 };
            }
          }),

        unpinNode: (id: string) =>
          set((state: GraphState) => {
            const node = state.nodes.get(id);
            if (node) {
              node.pinned = false;
            }
          }),

        pinSelected: () =>
          set((state: GraphState) => {
            state.selection.nodeIds.forEach((nodeId: string) => {
              const node = state.nodes.get(nodeId);
              if (node) {
                node.pinned = true;
                node.velocity = { x: 0, y: 0, z: 0 };
              }
            });
          }),

        unpinAll: () =>
          set((state: GraphState) => {
            state.nodes.forEach((node: GraphNode) => {
              node.pinned = false;
            });
          }),

        // ====================================================================
        // Query Helpers
        // ====================================================================

        getNode: (id: string) => get().nodes.get(id),

        getEdge: (id: string) => get().edges.get(id),

        getNodeEdges: (nodeId: string) => {
          const state = get();
          const edges: GraphEdge[] = [];

          state.edgesBySource.get(nodeId)?.forEach((edgeId: string) => {
            const edge = state.edges.get(edgeId);
            if (edge) edges.push(edge);
          });

          state.edgesByTarget.get(nodeId)?.forEach((edgeId: string) => {
            const edge = state.edges.get(edgeId);
            if (edge) edges.push(edge);
          });

          return edges;
        },

        getConnectedNodes: (nodeId: string) => {
          const state = get();
          const connectedIds = new Set<string>();

          state.edgesBySource.get(nodeId)?.forEach((edgeId: string) => {
            const edge = state.edges.get(edgeId);
            if (edge) connectedIds.add(edge.targetId);
          });

          state.edgesByTarget.get(nodeId)?.forEach((edgeId: string) => {
            const edge = state.edges.get(edgeId);
            if (edge) connectedIds.add(edge.sourceId);
          });

          return Array.from(connectedIds)
            .map((id: string) => state.nodes.get(id))
            .filter(Boolean) as GraphNode[];
        },

        getSelectedNodes: () => {
          const state = get();
          return Array.from(state.selection.nodeIds)
            .map((id: string) => state.nodes.get(id))
            .filter(Boolean) as GraphNode[];
        },

        getSelectedEdges: () => {
          const state = get();
          return Array.from(state.selection.edgeIds)
            .map((id: string) => state.edges.get(id))
            .filter(Boolean) as GraphEdge[];
        },
      }))
    )
  );

/**
 * Default graph store instance
 */
export const useGraphStore = createGraphStore();
