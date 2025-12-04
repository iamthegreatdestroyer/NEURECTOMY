/**
 * @file CAD Visualization Types
 * @description Type definitions for agent architecture visualization
 * @module @neurectomy/3d-engine/visualization
 * @agents @ARCHITECT @CANVAS
 */

import * as THREE from 'three';

// ============================================================================
// Agent Component Types
// ============================================================================

/**
 * Base agent component in the 3D scene
 */
export interface AgentComponent {
  /** Unique identifier */
  id: string;
  /** Component type */
  type: AgentComponentType;
  /** Display name */
  name: string;
  /** 3D position */
  position: THREE.Vector3;
  /** 3D rotation */
  rotation: THREE.Euler;
  /** 3D scale */
  scale: THREE.Vector3;
  /** Component metadata */
  metadata: AgentComponentMetadata;
  /** Visual style */
  style: ComponentStyle;
  /** Parent component ID (for hierarchy) */
  parentId?: string;
  /** Child component IDs */
  childIds: string[];
  /** Connection IDs */
  connectionIds: string[];
}

/**
 * Types of agent components
 */
export type AgentComponentType =
  | 'agent'           // Root agent entity
  | 'llm'             // Language model
  | 'tool'            // Tool/function
  | 'memory'          // Memory store
  | 'embedding'       // Embedding generator
  | 'retriever'       // RAG retriever
  | 'router'          // Routing logic
  | 'executor'        // Task executor
  | 'planner'         // Planning system
  | 'evaluator'       // Output evaluator
  | 'guardrail'       // Safety guardrail
  | 'connector'       // External connector
  | 'custom';         // Custom component

/**
 * Component metadata for rich visualization
 */
export interface AgentComponentMetadata {
  /** Component version */
  version?: string;
  /** Description */
  description?: string;
  /** Configuration parameters */
  config?: Record<string, unknown>;
  /** Performance metrics */
  metrics?: ComponentMetrics;
  /** Status */
  status: ComponentStatus;
  /** Tags for filtering */
  tags: string[];
  /** Custom properties */
  properties?: Record<string, unknown>;
}

/**
 * Component performance metrics
 */
export interface ComponentMetrics {
  /** Average latency in ms */
  avgLatencyMs?: number;
  /** Throughput (calls/second) */
  throughput?: number;
  /** Error rate (0-1) */
  errorRate?: number;
  /** Token usage */
  tokenUsage?: number;
  /** Memory usage in bytes */
  memoryUsage?: number;
}

/**
 * Component operational status
 */
export type ComponentStatus =
  | 'active'      // Running normally
  | 'idle'        // Ready but not processing
  | 'processing'  // Currently processing
  | 'error'       // In error state
  | 'disabled'    // Manually disabled
  | 'warning';    // Warning state

// ============================================================================
// Connection Types
// ============================================================================

/**
 * Connection between agent components
 */
export interface ComponentConnection {
  /** Unique connection ID */
  id: string;
  /** Source component ID */
  sourceId: string;
  /** Target component ID */
  targetId: string;
  /** Connection type */
  type: ConnectionType;
  /** Data flow direction */
  direction: ConnectionDirection;
  /** Connection style */
  style: ConnectionStyle;
  /** Animation state */
  animated: boolean;
  /** Connection metadata */
  metadata: ConnectionMetadata;
}

/**
 * Types of connections between components
 */
export type ConnectionType =
  | 'data'        // Data flow
  | 'control'     // Control flow
  | 'memory'      // Memory access
  | 'event'       // Event subscription
  | 'feedback'    // Feedback loop
  | 'dependency'; // Dependency relationship

/**
 * Direction of data flow
 */
export type ConnectionDirection =
  | 'forward'     // Source to target
  | 'backward'    // Target to source
  | 'bidirectional'; // Both directions

/**
 * Visual style for connections
 */
export interface ConnectionStyle {
  /** Line color */
  color: string;
  /** Line width */
  width: number;
  /** Line pattern */
  pattern: 'solid' | 'dashed' | 'dotted';
  /** Glow effect */
  glow: boolean;
  /** Opacity */
  opacity: number;
  /** Curve tension (0-1) */
  curveTension: number;
}

/**
 * Connection metadata
 */
export interface ConnectionMetadata {
  /** Label for the connection */
  label?: string;
  /** Data type being transferred */
  dataType?: string;
  /** Current throughput */
  throughput?: number;
  /** Connection latency */
  latencyMs?: number;
  /** Is currently active */
  active: boolean;
}

// ============================================================================
// Visual Style Types
// ============================================================================

/**
 * Component visual style configuration
 */
export interface ComponentStyle {
  /** Primary color */
  primaryColor: string;
  /** Secondary color */
  secondaryColor: string;
  /** Emissive color for glow */
  emissiveColor?: string;
  /** Emissive intensity */
  emissiveIntensity: number;
  /** Opacity */
  opacity: number;
  /** Wireframe mode */
  wireframe: boolean;
  /** Geometry type */
  geometry: ComponentGeometry;
  /** Icon identifier */
  icon?: string;
  /** Custom shader */
  customShader?: string;
}

/**
 * Geometry shapes for components
 */
export type ComponentGeometry =
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'cone'
  | 'torus'
  | 'octahedron'
  | 'icosahedron'
  | 'custom';

// ============================================================================
// Scene & Layer Types
// ============================================================================

/**
 * Agent architecture scene configuration
 */
export interface AgentScene {
  /** Scene ID */
  id: string;
  /** Scene name */
  name: string;
  /** All components in the scene */
  components: Map<string, AgentComponent>;
  /** All connections in the scene */
  connections: Map<string, ComponentConnection>;
  /** Scene layers */
  layers: SceneLayer[];
  /** Camera configuration */
  camera: CameraConfig;
  /** Lighting configuration */
  lighting: LightingConfig;
  /** Environment settings */
  environment: EnvironmentConfig;
}

/**
 * Scene layer for organization
 */
export interface SceneLayer {
  /** Layer ID */
  id: string;
  /** Layer name */
  name: string;
  /** Layer visibility */
  visible: boolean;
  /** Layer opacity */
  opacity: number;
  /** Component IDs in this layer */
  componentIds: string[];
  /** Z-index for ordering */
  zIndex: number;
  /** Layer color hint */
  color?: string;
}

/**
 * Camera configuration
 */
export interface CameraConfig {
  /** Camera type */
  type: 'perspective' | 'orthographic';
  /** Camera position */
  position: THREE.Vector3;
  /** Look-at target */
  target: THREE.Vector3;
  /** Field of view (perspective) */
  fov?: number;
  /** Zoom level (orthographic) */
  zoom?: number;
  /** Near clipping plane */
  near: number;
  /** Far clipping plane */
  far: number;
}

/**
 * Lighting configuration
 */
export interface LightingConfig {
  /** Ambient light color */
  ambientColor: string;
  /** Ambient light intensity */
  ambientIntensity: number;
  /** Directional lights */
  directionalLights: DirectionalLightConfig[];
  /** Point lights */
  pointLights: PointLightConfig[];
  /** Environment map */
  environmentMap?: string;
}

/**
 * Directional light configuration
 */
export interface DirectionalLightConfig {
  /** Light color */
  color: string;
  /** Light intensity */
  intensity: number;
  /** Light direction */
  direction: THREE.Vector3;
  /** Cast shadows */
  castShadow: boolean;
}

/**
 * Point light configuration
 */
export interface PointLightConfig {
  /** Light color */
  color: string;
  /** Light intensity */
  intensity: number;
  /** Light position */
  position: THREE.Vector3;
  /** Light range */
  distance: number;
  /** Decay factor */
  decay: number;
}

/**
 * Environment configuration
 */
export interface EnvironmentConfig {
  /** Background color or gradient */
  background: string | [string, string];
  /** Fog settings */
  fog?: FogConfig;
  /** Grid settings */
  grid?: GridConfig;
}

/**
 * Fog configuration
 */
export interface FogConfig {
  /** Fog type */
  type: 'linear' | 'exponential';
  /** Fog color */
  color: string;
  /** Near distance (linear) */
  near?: number;
  /** Far distance (linear) */
  far?: number;
  /** Density (exponential) */
  density?: number;
}

/**
 * Grid configuration
 */
export interface GridConfig {
  /** Show grid */
  visible: boolean;
  /** Grid size */
  size: number;
  /** Grid divisions */
  divisions: number;
  /** Primary color */
  primaryColor: string;
  /** Secondary color */
  secondaryColor: string;
}

// ============================================================================
// Selection & Interaction Types
// ============================================================================

/**
 * Selection state
 */
export interface SelectionState {
  /** Currently selected component IDs */
  selectedIds: Set<string>;
  /** Currently hovered component ID */
  hoveredId: string | null;
  /** Selection box bounds (for multi-select) */
  selectionBox: SelectionBox | null;
  /** Selection mode */
  mode: SelectionMode;
}

/**
 * Selection box for multi-select
 */
export interface SelectionBox {
  /** Start corner */
  start: THREE.Vector2;
  /** End corner */
  end: THREE.Vector2;
}

/**
 * Selection mode
 */
export type SelectionMode =
  | 'single'    // Single selection
  | 'multi'     // Multi-selection (shift-click)
  | 'box'       // Box selection
  | 'additive'; // Additive selection (ctrl-click)

/**
 * Transform gizmo mode
 */
export type TransformMode =
  | 'translate'
  | 'rotate'
  | 'scale'
  | 'none';

/**
 * Transform space
 */
export type TransformSpace =
  | 'world'
  | 'local';

// ============================================================================
// Blueprint View Types
// ============================================================================

/**
 * Blueprint view configuration
 */
export interface BlueprintConfig {
  /** Enable blueprint mode */
  enabled: boolean;
  /** Show grid */
  gridEnabled: boolean;
  /** Grid size */
  gridSize: number;
  /** Grid divisions */
  gridDivisions: number;
  /** Grid line color */
  gridColor: string;
  /** Grid center line color */
  gridCenterLineColor: string;
  /** Show dimension annotations */
  showDimensions: boolean;
  /** Show component labels */
  showLabels: boolean;
  /** Show connector/port labels */
  showConnectorLabels: boolean;
  /** Main line color */
  lineColor: string;
  /** Text color */
  textColor: string;
  /** Background color */
  backgroundColor: string;
  /** View scale */
  scale: number;
  /** View mode */
  mode?: BlueprintViewMode;
  /** Show annotations */
  showAnnotations?: boolean;
  /** Show measurements */
  showMeasurements?: boolean;
  /** Show data flow */
  showDataFlow?: boolean;
  /** Line style */
  lineStyle?: 'technical' | 'organic';
  /** Legacy color scheme support */
  colorScheme?: BlueprintColorScheme;
}

/**
 * Blueprint view modes
 */
export type BlueprintViewMode =
  | 'top'       // Top-down view
  | 'front'     // Front view
  | 'side'      // Side view
  | 'isometric' // Isometric view
  | '3d';       // Full 3D view

/**
 * Blueprint color scheme
 */
export interface BlueprintColorScheme {
  /** Background color */
  background: string;
  /** Line color */
  line: string;
  /** Accent color */
  accent: string;
  /** Text color */
  text: string;
  /** Grid color */
  grid: string;
}

// ============================================================================
// Measurement & Annotation Types
// ============================================================================

/**
 * Measurement annotation
 */
export interface Measurement {
  /** Measurement ID */
  id: string;
  /** Measurement type */
  type: MeasurementType;
  /** Start point */
  start: THREE.Vector3;
  /** End point */
  end: THREE.Vector3;
  /** Measured value */
  value: number;
  /** Unit */
  unit: string;
  /** Label position */
  labelPosition: THREE.Vector3;
  /** Visibility */
  visible: boolean;
}

/**
 * Types of measurements
 */
export type MeasurementType =
  | 'distance'
  | 'angle'
  | 'radius'
  | 'area'
  | 'volume';

/**
 * Text annotation
 */
export interface Annotation {
  /** Annotation ID */
  id: string;
  /** Annotation text */
  text: string;
  /** Position in 3D space */
  position: THREE.Vector3;
  /** Anchor component ID */
  anchorComponentId?: string;
  /** Style */
  style: AnnotationStyle;
  /** Visibility */
  visible: boolean;
}

/**
 * Annotation style
 */
export interface AnnotationStyle {
  /** Background color */
  backgroundColor: string;
  /** Text color */
  textColor: string;
  /** Border color */
  borderColor: string;
  /** Font size */
  fontSize: number;
  /** Max width */
  maxWidth: number;
}

// ============================================================================
// Default Configurations
// ============================================================================

/**
 * Default component styles by type
 */
export const DEFAULT_COMPONENT_STYLES: Record<AgentComponentType, Partial<ComponentStyle>> = {
  agent: {
    primaryColor: '#4f46e5',
    secondaryColor: '#818cf8',
    emissiveColor: '#4f46e5',
    emissiveIntensity: 0.2,
    geometry: 'octahedron',
  },
  llm: {
    primaryColor: '#10b981',
    secondaryColor: '#34d399',
    emissiveColor: '#10b981',
    emissiveIntensity: 0.3,
    geometry: 'icosahedron',
  },
  tool: {
    primaryColor: '#f59e0b',
    secondaryColor: '#fbbf24',
    emissiveColor: '#f59e0b',
    emissiveIntensity: 0.15,
    geometry: 'box',
  },
  memory: {
    primaryColor: '#8b5cf6',
    secondaryColor: '#a78bfa',
    emissiveColor: '#8b5cf6',
    emissiveIntensity: 0.2,
    geometry: 'cylinder',
  },
  embedding: {
    primaryColor: '#06b6d4',
    secondaryColor: '#22d3ee',
    emissiveColor: '#06b6d4',
    emissiveIntensity: 0.25,
    geometry: 'sphere',
  },
  retriever: {
    primaryColor: '#ec4899',
    secondaryColor: '#f472b6',
    emissiveColor: '#ec4899',
    emissiveIntensity: 0.2,
    geometry: 'cone',
  },
  router: {
    primaryColor: '#14b8a6',
    secondaryColor: '#2dd4bf',
    emissiveColor: '#14b8a6',
    emissiveIntensity: 0.15,
    geometry: 'torus',
  },
  executor: {
    primaryColor: '#ef4444',
    secondaryColor: '#f87171',
    emissiveColor: '#ef4444',
    emissiveIntensity: 0.2,
    geometry: 'box',
  },
  planner: {
    primaryColor: '#3b82f6',
    secondaryColor: '#60a5fa',
    emissiveColor: '#3b82f6',
    emissiveIntensity: 0.2,
    geometry: 'octahedron',
  },
  evaluator: {
    primaryColor: '#84cc16',
    secondaryColor: '#a3e635',
    emissiveColor: '#84cc16',
    emissiveIntensity: 0.15,
    geometry: 'sphere',
  },
  guardrail: {
    primaryColor: '#dc2626',
    secondaryColor: '#f87171',
    emissiveColor: '#dc2626',
    emissiveIntensity: 0.3,
    geometry: 'box',
  },
  connector: {
    primaryColor: '#6366f1',
    secondaryColor: '#818cf8',
    emissiveColor: '#6366f1',
    emissiveIntensity: 0.15,
    geometry: 'cylinder',
  },
  custom: {
    primaryColor: '#71717a',
    secondaryColor: '#a1a1aa',
    emissiveColor: '#71717a',
    emissiveIntensity: 0.1,
    geometry: 'box',
  },
};

/**
 * Default connection style
 */
export const DEFAULT_CONNECTION_STYLE: ConnectionStyle = {
  color: '#64748b',
  width: 2,
  pattern: 'solid',
  glow: true,
  opacity: 0.8,
  curveTension: 0.3,
};

/**
 * Default blueprint color scheme
 */
export const DEFAULT_BLUEPRINT_SCHEME: BlueprintColorScheme = {
  background: '#0f172a',
  line: '#38bdf8',
  accent: '#f472b6',
  text: '#e2e8f0',
  grid: '#1e293b',
};

// ============================================================================
// R3F-Compatible Types (Simplified for React components)
// ============================================================================

/**
 * Simplified Vector3D for serialization
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Color with alpha
 */
export interface Color4 {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * Bounding box
 */
export interface BoundingBox {
  min: Vector3D;
  max: Vector3D;
}

/**
 * Agent status enumeration
 */
export type AgentStatus = 
  | 'active'
  | 'idle'
  | 'processing'
  | 'error'
  | 'disabled'
  | 'warning';

/**
 * Port/connector on an agent component
 */
export interface AgentPort {
  id: string;
  name: string;
  type: 'input' | 'output';
  dataType?: string;
  connected?: boolean;
}

/**
 * Connection type for R3F components
 */
export type ConnectionType = 
  | 'data'
  | 'control'
  | 'event'
  | 'memory'
  | 'feedback';

/**
 * Visualization mode for scene rendering
 */
export type VisualizationMode =
  | 'default'
  | 'dark'
  | 'light'
  | 'blueprint'
  | 'wireframe';

/**
 * Viewport configuration
 */
export interface ViewportConfig {
  fov: number;
  near: number;
  far: number;
  initialPosition: Vector3D;
  initialTarget: Vector3D;
  enableOrbitControls: boolean;
  enablePan: boolean;
  enableZoom: boolean;
  enableRotate: boolean;
  minDistance: number;
  maxDistance: number;
  maxPolarAngle: number;
}

/**
 * Selection state for R3F components
 */
export interface SelectionState {
  componentIds: string[];
  connectionIds: string[];
}

/**
 * Render statistics
 */
export interface RenderStats {
  fps: number;
  frameTime: number;
  triangles: number;
  drawCalls: number;
  geometries: number;
  textures: number;
  programs: number;
}

/**
 * Agent connection for R3F (simplified)
 */
export interface AgentConnection {
  id: string;
  sourceId: string;
  targetId: string;
  type: ConnectionType;
  animated?: boolean;
  color?: string;
  metadata?: {
    label?: string;
    dataType?: string;
    throughput?: number;
  };
}

