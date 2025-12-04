/**
 * @file Visualization Module Exports
 * @description Barrel export for all visualization components
 * @module @neurectomy/3d-engine/visualization
 */

// Types - Core
export type {
  AgentComponent,
  AgentComponentType,
  AgentComponentMetadata,
  ComponentMetrics,
  ComponentStatus,
  ComponentStyle,
  ComponentGeometry,
  ComponentConnection,
  ConnectionType,
  ConnectionDirection,
  ConnectionStyle,
  ConnectionMetadata,
} from "./types";

// Types - Scene & Layers
export type {
  AgentScene,
  SceneLayer,
  CameraConfig,
  LightingConfig,
  DirectionalLightConfig,
  PointLightConfig,
  EnvironmentConfig,
  FogConfig,
  GridConfig,
} from "./types";

// Types - Blueprint
export type {
  BlueprintConfig,
  BlueprintViewMode,
  BlueprintColorScheme,
} from "./types";

// Types - Measurements & Annotations
export type {
  Measurement,
  MeasurementType,
  Annotation,
  AnnotationStyle,
} from "./types";

// Types - R3F Compatible
export type {
  Vector3D,
  Color4,
  BoundingBox,
  AgentStatus,
  AgentPort,
  VisualizationMode,
  ViewportConfig,
  SelectionState,
  RenderStats,
  AgentConnection,
} from "./types";

// Constants
export {
  DEFAULT_COMPONENT_STYLES,
  DEFAULT_CONNECTION_STYLE,
  DEFAULT_BLUEPRINT_SCHEME,
} from "./types";

// Agent Rendering
export {
  AgentMesh,
  AgentLabel,
  AgentStatus as AgentStatusIndicator,
  AgentGroup,
  type AgentMeshProps,
  type AgentGroupProps,
} from "./agent-renderer";

// Connection Rendering
export {
  ConnectionLine,
  ConnectionArrow,
  DataFlowParticles,
  ConnectionGroup,
  type ConnectionLineProps,
  type ConnectionGroupProps,
} from "./connection-renderer";

// Blueprint Mode
export {
  BlueprintMode,
  BlueprintGrid,
  BlueprintComponent,
  BlueprintConnection,
  MeasurementTool,
  type BlueprintModeProps,
} from "./blueprint-mode";

// Interaction System
export {
  InteractionManager,
  BoxSelection,
  type InteractionConfig,
  type InteractionState,
  type InteractionMode,
  type InteractionEvent,
  type SelectionChangeEvent,
  type DragEvent,
  type ConnectionEvent,
  type DragTarget,
  type ConnectionDraft,
} from "./interaction-system";

// Scene Composer
export { SceneComposer, type SceneComposerProps } from "./scene-composer";

// Graph 3D Visualization
export {
  // Main component
  Graph3D,
  // Store
  useGraphStore,
  // Node components
  NodeMesh,
  InstancedNodes,
  // Edge components
  EdgeLine,
  InstancedEdges,
  // Default configs
  DEFAULT_GRAPH_CONFIG,
  DEFAULT_PHYSICS_CONFIG,
  DEFAULT_VISUAL_CONFIG,
  DEFAULT_INTERACTION_CONFIG,
  DEFAULT_LAYOUT_CONFIG,
} from "./graph";

export type {
  Graph3DProps,
  Graph3DRef,
  GraphState,
  GraphActions,
  GraphStore,
  NodeMeshProps,
  InstancedNodesProps,
  EdgeLineProps,
  InstancedEdgesProps,
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
  NodeEvent,
  EdgeEvent,
  SelectionEvent,
  DragNodeEvent,
  GraphEvent,
  GraphEventType,
  GraphNodeType,
  GraphEdgeType,
  GraphEdgeDirection,
} from "./graph";

// Default export
export { SceneComposer as default } from "./scene-composer";
