/**
 * @file Graph3D Types
 * @description Type definitions for force-directed graph visualization
 * @module @neurectomy/3d-engine/visualization/graph
 * @agents @APEX @CANVAS
 */

import type * as THREE from "three";
import type { Vector3, Quaternion } from "../../physics/types";

// ============================================================================
// Node Types
// ============================================================================

/**
 * Graph node data structure
 */
export interface GraphNode {
  /** Unique node identifier */
  id: string;
  /** Node display label */
  label: string;
  /** Node type for visual differentiation */
  type: NodeType;
  /** Node position in 3D space */
  position: Vector3;
  /** Node velocity for physics simulation */
  velocity: Vector3;
  /** Node mass for physics calculations */
  mass: number;
  /** Visual radius */
  radius: number;
  /** Node color (hex) */
  color: string;
  /** Is node pinned (fixed position) */
  pinned: boolean;
  /** Node metadata */
  metadata: NodeMetadata;
  /** Node state */
  state: NodeState;
}

/**
 * Types of graph nodes
 */
export type NodeType =
  | "agent"
  | "llm"
  | "tool"
  | "memory"
  | "database"
  | "api"
  | "user"
  | "system"
  | "process"
  | "service"
  | "custom";

/**
 * Node metadata for rich information display
 */
export interface NodeMetadata {
  /** Description */
  description?: string;
  /** Category for grouping */
  category?: string;
  /** Tags for filtering */
  tags: string[];
  /** Timestamp of creation */
  createdAt?: number;
  /** Timestamp of last update */
  updatedAt?: number;
  /** Custom properties */
  properties: Record<string, unknown>;
}

/**
 * Node operational state
 */
export interface NodeState {
  /** Is node selected */
  selected: boolean;
  /** Is node hovered */
  hovered: boolean;
  /** Is node being dragged */
  dragging: boolean;
  /** Is node highlighted */
  highlighted: boolean;
  /** Is node dimmed */
  dimmed: boolean;
  /** Is node visible */
  visible: boolean;
  /** Activity level (0-1) */
  activity: number;
}

// ============================================================================
// Edge Types
// ============================================================================

/**
 * Graph edge data structure
 */
export interface GraphEdge {
  /** Unique edge identifier */
  id: string;
  /** Source node ID */
  sourceId: string;
  /** Target node ID */
  targetId: string;
  /** Edge type for visual differentiation */
  type: EdgeType;
  /** Edge weight (affects physics) */
  weight: number;
  /** Preferred edge length */
  length: number;
  /** Edge direction */
  direction: EdgeDirection;
  /** Edge color (hex) */
  color: string;
  /** Edge width */
  width: number;
  /** Edge metadata */
  metadata: EdgeMetadata;
  /** Edge state */
  state: EdgeState;
}

/**
 * Types of graph edges
 */
export type EdgeType =
  | "connection"
  | "dependency"
  | "dataflow"
  | "control"
  | "hierarchy"
  | "reference"
  | "association"
  | "custom";

/**
 * Edge direction type
 */
export type EdgeDirection =
  | "none" // Undirected
  | "forward" // Source -> Target
  | "backward" // Target -> Source
  | "bidirectional"; // Both directions

/**
 * Edge metadata for rich information display
 */
export interface EdgeMetadata {
  /** Edge label */
  label?: string;
  /** Description */
  description?: string;
  /** Bandwidth/throughput value */
  bandwidth?: number;
  /** Latency in ms */
  latency?: number;
  /** Custom properties */
  properties: Record<string, unknown>;
}

/**
 * Edge operational state
 */
export interface EdgeState {
  /** Is edge selected */
  selected: boolean;
  /** Is edge hovered */
  hovered: boolean;
  /** Is edge highlighted */
  highlighted: boolean;
  /** Is edge dimmed */
  dimmed: boolean;
  /** Is edge visible */
  visible: boolean;
  /** Animation progress (0-1) */
  animationProgress: number;
}

// ============================================================================
// Graph Configuration
// ============================================================================

/**
 * Graph configuration options
 */
export interface GraphConfig {
  /** Physics configuration */
  physics: GraphPhysicsConfig;
  /** Visual configuration */
  visual: GraphVisualConfig;
  /** Interaction configuration */
  interaction: GraphInteractionConfig;
  /** Layout configuration */
  layout: GraphLayoutConfig;
}

/**
 * Physics configuration for force simulation
 */
export interface GraphPhysicsConfig {
  /** Enable physics simulation */
  enabled: boolean;
  /** Gravity strength */
  gravity: number;
  /** Repulsion strength between nodes */
  repulsion: number;
  /** Attraction strength for edges */
  attraction: number;
  /** Damping factor (0-1) */
  damping: number;
  /** Maximum velocity */
  maxVelocity: number;
  /** Time step for simulation */
  timeStep: number;
  /** Iterations per frame */
  iterations: number;
  /** Alpha decay rate */
  alphaDecay: number;
  /** Velocity decay rate */
  velocityDecay: number;
  /** Collision radius multiplier */
  collisionRadius: number;
}

/**
 * Visual configuration for graph rendering
 */
export interface GraphVisualConfig {
  /** Node visual settings */
  node: NodeVisualConfig;
  /** Edge visual settings */
  edge: EdgeVisualConfig;
  /** Background color */
  backgroundColor: string;
  /** Enable bloom effect */
  bloom: boolean;
  /** Bloom intensity */
  bloomIntensity: number;
  /** Enable shadows */
  shadows: boolean;
  /** Enable antialiasing */
  antialias: boolean;
  /** Ambient light intensity */
  ambientLight: number;
}

/**
 * Node visual configuration
 */
export interface NodeVisualConfig {
  /** Default node radius */
  defaultRadius: number;
  /** Minimum node radius */
  minRadius: number;
  /** Maximum node radius */
  maxRadius: number;
  /** Default node color */
  defaultColor: string;
  /** Selected node color */
  selectedColor: string;
  /** Hovered node color */
  hoveredColor: string;
  /** Node opacity */
  opacity: number;
  /** Node metalness */
  metalness: number;
  /** Node roughness */
  roughness: number;
  /** Show labels */
  showLabels: boolean;
  /** Label font size */
  labelFontSize: number;
  /** Label offset from node */
  labelOffset: number;
}

/**
 * Edge visual configuration
 */
export interface EdgeVisualConfig {
  /** Default edge width */
  defaultWidth: number;
  /** Minimum edge width */
  minWidth: number;
  /** Maximum edge width */
  maxWidth: number;
  /** Default edge color */
  defaultColor: string;
  /** Selected edge color */
  selectedColor: string;
  /** Hovered edge color */
  hoveredColor: string;
  /** Edge opacity */
  opacity: number;
  /** Show arrows */
  showArrows: boolean;
  /** Arrow size */
  arrowSize: number;
  /** Edge curve factor */
  curveFactor: number;
  /** Animate edges */
  animated: boolean;
  /** Animation speed */
  animationSpeed: number;
}

/**
 * Interaction configuration
 */
export interface GraphInteractionConfig {
  /** Enable node dragging */
  enableDrag: boolean;
  /** Enable node selection */
  enableSelection: boolean;
  /** Enable multi-selection */
  enableMultiSelection: boolean;
  /** Enable zoom */
  enableZoom: boolean;
  /** Enable pan */
  enablePan: boolean;
  /** Enable rotation */
  enableRotation: boolean;
  /** Double-click behavior */
  doubleClickBehavior: "focus" | "expand" | "edit" | "none";
  /** Right-click behavior */
  rightClickBehavior: "menu" | "none";
  /** Hover delay in ms */
  hoverDelay: number;
  /** Drag threshold in pixels */
  dragThreshold: number;
}

/**
 * Layout configuration
 */
export interface GraphLayoutConfig {
  /** Layout algorithm */
  algorithm: LayoutAlgorithm;
  /** Initial layout bounds */
  bounds: LayoutBounds;
  /** Center graph in view */
  centerOnInit: boolean;
  /** Fit graph in view */
  fitOnInit: boolean;
  /** Padding around graph */
  padding: number;
}

/**
 * Available layout algorithms
 */
export type LayoutAlgorithm =
  | "force-directed"
  | "hierarchical"
  | "radial"
  | "grid"
  | "circular"
  | "custom";

/**
 * Layout bounds in 3D space
 */
export interface LayoutBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

// ============================================================================
// Graph Events
// ============================================================================

/**
 * Graph event types
 */
export type GraphEventType =
  | "node:click"
  | "node:doubleclick"
  | "node:rightclick"
  | "node:hover"
  | "node:drag:start"
  | "node:drag"
  | "node:drag:end"
  | "node:select"
  | "node:deselect"
  | "edge:click"
  | "edge:doubleclick"
  | "edge:hover"
  | "edge:select"
  | "edge:deselect"
  | "canvas:click"
  | "canvas:rightclick"
  | "simulation:tick"
  | "simulation:end";

/**
 * Base graph event
 */
export interface GraphEvent<T extends GraphEventType = GraphEventType> {
  /** Event type */
  type: T;
  /** Timestamp */
  timestamp: number;
  /** Original DOM event (if applicable) */
  originalEvent?: MouseEvent | PointerEvent;
}

/**
 * Node event
 */
export interface NodeEvent extends GraphEvent {
  /** Node ID */
  nodeId: string;
  /** Node data */
  node: GraphNode;
  /** 3D position of interaction */
  position: Vector3;
  /** Screen position of interaction */
  screenPosition: { x: number; y: number };
}

/**
 * Edge event
 */
export interface EdgeEvent extends GraphEvent {
  /** Edge ID */
  edgeId: string;
  /** Edge data */
  edge: GraphEdge;
  /** Source node */
  sourceNode: GraphNode;
  /** Target node */
  targetNode: GraphNode;
}

/**
 * Selection event
 */
export interface SelectionEvent extends GraphEvent {
  /** Selected node IDs */
  selectedNodeIds: string[];
  /** Selected edge IDs */
  selectedEdgeIds: string[];
  /** Previous selection */
  previousSelection: {
    nodeIds: string[];
    edgeIds: string[];
  };
}

/**
 * Drag event for nodes
 */
export interface DragNodeEvent extends NodeEvent {
  /** Delta movement */
  delta: Vector3;
  /** Total movement from start */
  totalDelta: Vector3;
}

// ============================================================================
// Default Configurations
// ============================================================================

/**
 * Default physics configuration
 */
export const DEFAULT_PHYSICS_CONFIG: GraphPhysicsConfig = {
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
};

/**
 * Default node visual configuration
 */
export const DEFAULT_NODE_VISUAL: NodeVisualConfig = {
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
};

/**
 * Default edge visual configuration
 */
export const DEFAULT_EDGE_VISUAL: EdgeVisualConfig = {
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
};

/**
 * Default visual configuration
 */
export const DEFAULT_VISUAL_CONFIG: GraphVisualConfig = {
  node: DEFAULT_NODE_VISUAL,
  edge: DEFAULT_EDGE_VISUAL,
  backgroundColor: "#1a1a2e",
  bloom: true,
  bloomIntensity: 0.5,
  shadows: true,
  antialias: true,
  ambientLight: 0.4,
};

/**
 * Default interaction configuration
 */
export const DEFAULT_INTERACTION_CONFIG: GraphInteractionConfig = {
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
};

/**
 * Default layout configuration
 */
export const DEFAULT_LAYOUT_CONFIG: GraphLayoutConfig = {
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
};

/**
 * Default graph configuration
 */
export const DEFAULT_GRAPH_CONFIG: GraphConfig = {
  physics: DEFAULT_PHYSICS_CONFIG,
  visual: DEFAULT_VISUAL_CONFIG,
  interaction: DEFAULT_INTERACTION_CONFIG,
  layout: DEFAULT_LAYOUT_CONFIG,
};
