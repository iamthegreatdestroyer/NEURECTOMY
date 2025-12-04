/**
 * @neurectomy/3d-engine
 *
 * High-performance 3D/4D visualization engine for the Dimensional Forge.
 * Implements WebGPU-first rendering with WebGL2 fallback.
 *
 * @module @neurectomy/3d-engine
 * @author NEURECTOMY Team
 * @license MIT
 */

// Import React Three Fiber type augmentations for JSX
import "./r3f-types";

// Core types - the primary types for the engine
export * from "./core/types";

// WebGPU exports - re-export as namespace to avoid conflicts
export * as webgpu from "./webgpu";

// Physics exports - re-export as namespace
export * as physics from "./physics";

// Graph3D Visualization - our main completed component
export {
  Graph3D,
  useGraphStore,
  NodeMesh,
  EdgeLine,
  InstancedNodes,
  InstancedEdges,
  DEFAULT_GRAPH_CONFIG,
  DEFAULT_PHYSICS_CONFIG,
  DEFAULT_VISUAL_CONFIG,
  DEFAULT_INTERACTION_CONFIG,
  DEFAULT_LAYOUT_CONFIG,
} from "./visualization/graph";

export type {
  Graph3DProps,
  Graph3DRef,
  GraphNode,
  GraphEdge,
  NodeMetadata,
  EdgeMetadata,
  NodeState,
  EdgeState,
  GraphConfig,
  GraphPhysicsConfig,
  GraphVisualConfig,
  GraphInteractionConfig,
  GraphLayoutConfig,
  GraphState,
  GraphActions,
  GraphStore,
  NodeMeshProps,
  InstancedNodesProps,
  EdgeLineProps,
  InstancedEdgesProps,
  GraphNodeType,
  GraphEdgeType,
  GraphEdgeDirection,
  GraphEvent,
  GraphEventType,
} from "./visualization/graph";

// Temporal/4D Components
export * as temporal from "./temporal";
export {
  Timeline,
  TimelineScrubber,
  PlaybackControls,
  KeyframeMarkers,
  TimeRuler,
  SpeedControl,
  SnapshotThumbnail,
} from "./temporal/components";

export type {
  TimelineProps,
  TimelineScrubberProps,
  PlaybackControlsProps,
  KeyframeMarkersProps,
  TimeRulerProps,
  SpeedControlProps,
  SnapshotThumbnailProps,
  TimelineTheme,
  TimelineViewport,
  TimelineSelection,
} from "./temporal/components";

// Optimization utilities
export * as optimization from "./optimization";
export {
  LODManager,
  QUALITY_PRESETS,
  BarnesHutTree,
  createBodiesFromNodes,
  applyPositionsToNodes,
} from "./optimization";

export type {
  LODLevel,
  LODEntry,
  LODManagerConfig,
  LODStatistics,
  QualityPreset,
  BarnesHutNode,
  BarnesHutConfig,
  Body,
  ForceResult,
} from "./optimization";
