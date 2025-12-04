/**
 * @file Graph Visualization Module
 * @description Complete 3D force-directed graph visualization system
 * @module @neurectomy/3d-engine/visualization/graph
 */

// ============================================================================
// TYPES
// ============================================================================

export type {
  // Node types
  GraphNode,
  NodeType,
  NodeMetadata,
  NodeState,

  // Edge types
  GraphEdge,
  EdgeType,
  EdgeDirection,
  EdgeMetadata,
  EdgeState,

  // Configuration types
  GraphConfig,
  GraphPhysicsConfig,
  GraphVisualConfig,
  GraphInteractionConfig,
  GraphLayoutConfig,

  // Event types
  NodeEvent,
  EdgeEvent,
  SelectionEvent,
  DragNodeEvent,
  GraphEvent,
  GraphEventType,
} from "./types";

// Export type aliases with graph prefix for external use
export type { NodeType as GraphNodeType } from "./types";
export type { EdgeType as GraphEdgeType } from "./types";
export type { EdgeDirection as GraphEdgeDirection } from "./types";

// Export default configurations
export {
  DEFAULT_GRAPH_CONFIG,
  DEFAULT_PHYSICS_CONFIG,
  DEFAULT_VISUAL_CONFIG,
  DEFAULT_INTERACTION_CONFIG,
  DEFAULT_LAYOUT_CONFIG,
} from "./types";

// ============================================================================
// STORE
// ============================================================================

export {
  useGraphStore,
  type GraphState,
  type GraphActions,
  type GraphStore,
  type SelectionState,
  type HoverState,
  type DragState,
  type SimulationState,
  type CameraState,
} from "./store";

// ============================================================================
// COMPONENTS
// ============================================================================

// Node rendering
export { NodeMesh, InstancedNodes } from "./NodeMesh";
export type { NodeMeshProps, InstancedNodesProps } from "./NodeMesh";

// Edge rendering
export { EdgeLine, InstancedEdges } from "./EdgeLine";
export type { EdgeLineProps, InstancedEdgesProps } from "./EdgeLine";

// Main Graph3D component
export { Graph3D } from "./Graph3D";
export type { Graph3DProps, Graph3DRef } from "./Graph3D";

// ============================================================================
// LAYOUTS
// ============================================================================

export {
  ForceDirectedLayout,
  BarnesHutTree,
  HierarchicalLayout,
  RadialLayout,
  LayoutManager,
} from "./layouts";

// ============================================================================
// ADAPTERS
// ============================================================================

export {
  Neo4jAdapter,
  CypherQueryBuilder,
  type Neo4jPath,
  type CypherQueryResult,
  type NodeStyleMapping,
  type EdgeStyleMapping,
  DEFAULT_NODE_STYLES,
  DEFAULT_EDGE_STYLES,
} from "./adapters";

// ============================================================================
// NEO4J VISUALIZATION
// ============================================================================

export {
  // Core adapter
  Neo4jGraphAdapter,
  type Neo4jConfig,
  type Neo4jQueryResult,

  // Transformer
  Neo4jDataTransformer,
  type Neo4jNode,
  type Neo4jRelationship,
  type TransformOptions,

  // Query visualizer
  CypherQueryVisualizer,
  type CypherQuery,
  type QueryVisualization,

  // React hooks and components
  Neo4jGraphRenderer,
  type Neo4jRenderOptions,
  useNeo4jGraph,
} from "./neo4j";

// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================

// Re-export Graph3D as default
export { Graph3D as default } from "./Graph3D";
