/**
 * Cross-Domain Unified Type System
 *
 * Isomorphic type abstractions that unify concepts across:
 * - Dimensional Forge (3D visualization)
 * - Digital Twin (state simulation)
 * - Intelligence Foundry (ML training)
 *
 * ISOMORPHISM MAPPING:
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │ DIMENSIONAL FORGE    │ DIGITAL TWIN         │ INTELLIGENCE FOUNDRY  │
 * ├──────────────────────┼──────────────────────┼───────────────────────┤
 * │ AgentComponent       │ TwinState            │ ModelLayer            │
 * │ ComponentGraph       │ DependencyGraph      │ NeuralArchitecture    │
 * │ TimelinePoint        │ StateSnapshot        │ Checkpoint            │
 * │ Playback             │ Simulation           │ Inference             │
 * │ Blueprint            │ Configuration        │ Hyperparameters       │
 * │ Renderer             │ Synchronizer         │ Executor              │
 * │ InteractionEvent     │ SyncEvent            │ TrainingEvent         │
 * └──────────────────────┴──────────────────────┴───────────────────────┘
 *
 * @module @neurectomy/3d-engine/cross-domain/types
 * @agents @NEXUS @AXIOM @ARCHITECT
 */

// ============================================================================
// UNIVERSAL IDENTIFIERS
// ============================================================================

/** Universal unique identifier across all domains */
export type UniversalId = string;

/** Timestamp in milliseconds since epoch */
export type Timestamp = number;

/** Duration in milliseconds */
export type Duration = number;

/** Semantic version string */
export type Version = string;

/** Hash for integrity checks */
export type Hash = string;

// ============================================================================
// UNIFIED ENTITY (Isomorphic Base)
// ============================================================================

/**
 * UnifiedEntity - The isomorphic base for AgentComponent, TwinState, and ModelLayer
 *
 * This is the fundamental abstraction that exists across all three domains.
 * Each domain sees this entity through its own lens:
 * - Forge: A 3D renderable component
 * - Twin: A stateful simulation entity
 * - Foundry: A neural network layer/module
 */
export interface UnifiedEntity<TConfig = unknown, TState = unknown> {
  /** Unique identifier (universal) */
  id: UniversalId;

  /** Human-readable name */
  name: string;

  /** Entity type identifier */
  type: string;

  /** Entity version */
  version: Version;

  /** Configuration (Blueprint/Configuration/Hyperparameters) */
  config: TConfig;

  /** Current state (ComponentState/TwinState/LayerState) */
  state: TState;

  /** Input ports */
  inputs: PortDefinition[];

  /** Output ports */
  outputs: PortDefinition[];

  /** Metadata bag */
  metadata: EntityMetadata;

  /** Parent entity reference */
  parentId?: UniversalId;

  /** Child entity references */
  childIds: UniversalId[];

  /** Creation timestamp */
  createdAt: Timestamp;

  /** Last modified timestamp */
  modifiedAt: Timestamp;
}

/**
 * Port definition for entity I/O
 */
export interface PortDefinition {
  id: UniversalId;
  name: string;
  dataType: string;
  required: boolean;
  defaultValue?: unknown;
  description?: string;
}

/**
 * Universal metadata container
 */
export interface EntityMetadata {
  description?: string;
  tags: string[];
  annotations: Record<string, unknown>;

  /** Domain-specific extensions */
  forge?: ForgeMetadata;
  twin?: TwinMetadata;
  foundry?: FoundryMetadata;
}

// ============================================================================
// DOMAIN-SPECIFIC METADATA
// ============================================================================

/**
 * Dimensional Forge specific metadata
 */
export interface ForgeMetadata {
  /** 3D position */
  position: Vector3;
  /** 3D rotation */
  rotation: EulerAngles;
  /** 3D scale */
  scale: Vector3;
  /** Visual style */
  style: VisualStyle;
  /** Visibility state */
  visible: boolean;
  /** Selected state */
  selected: boolean;
  /** LOD level */
  lodLevel: number;
}

/**
 * Digital Twin specific metadata
 */
export interface TwinMetadata {
  /** Twin mode */
  mode: TwinMode;
  /** Sync state */
  syncState: SyncState;
  /** Fidelity level */
  fidelity: FidelityLevel;
  /** Divergence from source */
  divergenceScore: number;
  /** Source agent reference */
  sourceAgentId?: UniversalId;
  /** Last sync timestamp */
  lastSyncAt: Timestamp;
}

/**
 * Intelligence Foundry specific metadata
 */
export interface FoundryMetadata {
  /** Layer type */
  layerType: LayerType;
  /** Parameter count */
  parameterCount: number;
  /** FLOPs estimate */
  flops: number;
  /** Memory footprint bytes */
  memoryBytes: number;
  /** Training state */
  trainingState: TrainingState;
  /** Gradient stats */
  gradientStats?: GradientStats;
}

// ============================================================================
// UNIFIED GRAPH (ComponentGraph/DependencyGraph/NeuralArchitecture)
// ============================================================================

/**
 * UnifiedGraph - Isomorphic graph structure
 *
 * Represents:
 * - Forge: Agent component hierarchy and connections
 * - Twin: State dependency graph
 * - Foundry: Neural network architecture
 */
export interface UnifiedGraph<TNode extends UnifiedEntity = UnifiedEntity> {
  /** Graph identifier */
  id: UniversalId;

  /** Graph name */
  name: string;

  /** All nodes in the graph */
  nodes: Map<UniversalId, TNode>;

  /** All edges in the graph */
  edges: Map<UniversalId, UnifiedEdge>;

  /** Root node ID */
  rootId?: UniversalId;

  /** Graph-level metadata */
  metadata: GraphMetadata;

  /** Topological sort order (cached) */
  topologicalOrder?: UniversalId[];

  /** Graph version */
  version: Version;
}

/**
 * Edge in the unified graph
 */
export interface UnifiedEdge {
  id: UniversalId;
  sourceId: UniversalId;
  sourcePort?: string;
  targetId: UniversalId;
  targetPort?: string;

  /** Edge type (data/control/dependency/gradient) */
  type: EdgeType;

  /** Edge weight/strength */
  weight: number;

  /** Bidirectional flag */
  bidirectional: boolean;

  /** Edge metadata */
  metadata: EdgeMetadata;
}

/**
 * Edge types across domains
 */
export type EdgeType =
  // Forge types
  | "data-flow"
  | "control-flow"
  | "memory-access"
  | "event"
  // Twin types
  | "dependency"
  | "sync"
  | "causation"
  // Foundry types
  | "forward"
  | "gradient"
  | "skip-connection"
  | "attention";

/**
 * Edge metadata
 */
export interface EdgeMetadata {
  label?: string;
  description?: string;

  /** Data flow rate (for visualization) */
  flowRate?: number;

  /** Latency estimate ms */
  latencyMs?: number;

  /** Is this edge animated */
  animated?: boolean;

  /** Visual style overrides */
  style?: EdgeStyle;
}

/**
 * Graph-level metadata
 */
export interface GraphMetadata {
  description?: string;
  tags: string[];

  /** Total node count */
  nodeCount: number;

  /** Total edge count */
  edgeCount: number;

  /** Maximum depth */
  maxDepth: number;

  /** Is graph acyclic */
  isAcyclic: boolean;

  /** Domain hints */
  domains: ("forge" | "twin" | "foundry")[];
}

// ============================================================================
// UNIFIED TEMPORAL POINT (TimelinePoint/StateSnapshot/Checkpoint)
// ============================================================================

/**
 * UnifiedTemporalPoint - Isomorphic temporal snapshot
 *
 * Represents:
 * - Forge: A keyframe in the timeline
 * - Twin: A state snapshot
 * - Foundry: A training checkpoint
 */
export interface UnifiedTemporalPoint<TState = unknown> {
  /** Snapshot identifier */
  id: UniversalId;

  /** Timestamp */
  timestamp: Timestamp;

  /** The captured state */
  state: TState;

  /** State hash for quick comparison */
  hash: Hash;

  /** Parent point (for delta encoding) */
  parentId?: UniversalId;

  /** Delta from parent */
  delta?: StateDelta;

  /** Is this a keyframe (full state) */
  isKeyframe: boolean;

  /** Temporal metadata */
  metadata: TemporalMetadata;

  /** Metrics at this point */
  metrics?: UnifiedMetrics;
}

/**
 * Temporal point metadata
 */
export interface TemporalMetadata {
  label?: string;
  description?: string;
  tags: string[];

  /** Source of the snapshot */
  source: "auto" | "manual" | "checkpoint" | "prediction";

  /** Size in bytes */
  sizeBytes: number;

  /** Compression used */
  compression?: "none" | "lz4" | "zstd" | "brotli";

  /** Branch name (for divergent timelines) */
  branch?: string;

  /** Domain-specific data */
  forgeData?: ForgeTemporalData;
  twinData?: TwinTemporalData;
  foundryData?: FoundryTemporalData;
}

/**
 * Forge-specific temporal data
 */
export interface ForgeTemporalData {
  cameraPosition: Vector3;
  cameraTarget: Vector3;
  selectedIds: UniversalId[];
  visibleLayers: string[];
}

/**
 * Twin-specific temporal data
 */
export interface TwinTemporalData {
  syncState: SyncState;
  divergenceScore: number;
  scenarioId?: string;
}

/**
 * Foundry-specific temporal data
 */
export interface FoundryTemporalData {
  epoch: number;
  step: number;
  learningRate: number;
  loss: number;
  metrics: Record<string, number>;
}

/**
 * State delta for incremental storage
 */
export interface StateDelta {
  operations: DeltaOperation[];
  sizeBytes: number;
}

/**
 * Delta operation types
 */
export interface DeltaOperation {
  type: "set" | "delete" | "insert" | "move" | "patch";
  path: string[];
  value?: unknown;
  previousValue?: unknown;
  fromIndex?: number;
  toIndex?: number;
}

// ============================================================================
// UNIFIED TIMELINE
// ============================================================================

/**
 * UnifiedTimeline - Time-series container for temporal points
 */
export interface UnifiedTimeline<TState = unknown> {
  /** Timeline identifier */
  id: UniversalId;

  /** Timeline name */
  name: string;

  /** All temporal points */
  points: UnifiedTemporalPoint<TState>[];

  /** Keyframe indices for fast seeking */
  keyframeIndices: number[];

  /** Time bounds */
  bounds: TimeRange;

  /** Current playhead position */
  currentTime: Timestamp;

  /** Playback configuration */
  playback: PlaybackConfig;

  /** Timeline branches */
  branches: TimelineBranch[];

  /** Active branch */
  activeBranch: string;
}

/**
 * Time range
 */
export interface TimeRange {
  start: Timestamp;
  end: Timestamp;
}

/**
 * Playback configuration (Playback/Simulation/Inference mode)
 */
export interface PlaybackConfig {
  /** Current state */
  state: PlaybackState;

  /** Playback speed multiplier */
  speed: number;

  /** Playback direction */
  direction: "forward" | "backward";

  /** Loop mode */
  loop: boolean;

  /** Loop range */
  loopRange?: TimeRange;

  /** Step mode (frame by frame) */
  stepping: boolean;
}

export type PlaybackState = "playing" | "paused" | "seeking" | "scrubbing";

/**
 * Timeline branch for divergent exploration
 */
export interface TimelineBranch {
  name: string;
  branchPointId: UniversalId;
  description?: string;
  createdAt: Timestamp;
}

// ============================================================================
// UNIFIED METRICS
// ============================================================================

/**
 * UnifiedMetrics - Cross-domain metrics container
 */
export interface UnifiedMetrics {
  timestamp: Timestamp;

  /** Performance metrics */
  performance: PerformanceMetrics;

  /** Resource metrics */
  resources: ResourceMetrics;

  /** Domain-specific metrics */
  forge?: ForgeMetrics;
  twin?: TwinMetricsData;
  foundry?: FoundryMetrics;

  /** Custom metrics */
  custom: Record<string, number>;
}

export interface PerformanceMetrics {
  latencyMs: MetricSummary;
  throughput: MetricSummary;
  errorRate: number;
}

export interface ResourceMetrics {
  cpuPercent: number;
  memoryMB: number;
  gpuPercent?: number;
  gpuMemoryMB?: number;
  networkBytesIn: number;
  networkBytesOut: number;
}

export interface MetricSummary {
  min: number;
  max: number;
  mean: number;
  median: number;
  p95: number;
  p99: number;
  stdDev: number;
}

export interface ForgeMetrics {
  fps: number;
  drawCalls: number;
  triangles: number;
  visibleNodes: number;
}

export interface TwinMetricsData {
  syncLatencyMs: number;
  divergence: number;
  predictionAccuracy?: number;
}

export interface FoundryMetrics {
  loss: number;
  accuracy?: number;
  gradientNorm: number;
  learningRate: number;
  epoch: number;
  step: number;
}

// ============================================================================
// UNIFIED EVENT SYSTEM
// ============================================================================

/**
 * UnifiedEvent - Cross-domain event (InteractionEvent/SyncEvent/TrainingEvent)
 */
export interface UnifiedEvent<TPayload = unknown> {
  /** Event identifier */
  id: UniversalId;

  /** Event type */
  type: EventType;

  /** Source domain */
  sourceDomain: Domain;

  /** Target domains (for propagation) */
  targetDomains: Domain[];

  /** Event timestamp */
  timestamp: Timestamp;

  /** Source entity ID */
  sourceEntityId?: UniversalId;

  /** Event payload */
  payload: TPayload;

  /** Is event propagatable to other domains */
  propagatable: boolean;

  /** Event priority */
  priority: EventPriority;

  /** Correlation ID for tracing */
  correlationId?: UniversalId;
}

export type Domain = "forge" | "twin" | "foundry";

export type EventType =
  // Forge events
  | "component:created"
  | "component:updated"
  | "component:deleted"
  | "component:selected"
  | "component:moved"
  | "connection:created"
  | "connection:deleted"
  | "timeline:seek"
  | "timeline:play"
  | "timeline:pause"
  // Twin events
  | "state:changed"
  | "state:synced"
  | "state:diverged"
  | "prediction:started"
  | "prediction:completed"
  | "scenario:created"
  | "scenario:evaluated"
  // Foundry events
  | "training:started"
  | "training:step"
  | "training:epoch"
  | "training:completed"
  | "training:failed"
  | "checkpoint:saved"
  | "checkpoint:loaded"
  | "architecture:changed"
  // Cross-domain events
  | "entity:created"
  | "entity:updated"
  | "entity:deleted"
  | "graph:modified"
  | "metrics:updated";

export type EventPriority = "low" | "normal" | "high" | "critical";

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface EulerAngles {
  x: number;
  y: number;
  z: number;
  order?: "XYZ" | "YXZ" | "ZXY" | "ZYX" | "YZX" | "XZY";
}

export interface VisualStyle {
  color: string;
  opacity: number;
  emissive?: string;
  emissiveIntensity?: number;
  wireframe?: boolean;
  geometry?: GeometryType;
}

export type GeometryType =
  | "sphere"
  | "box"
  | "cylinder"
  | "cone"
  | "torus"
  | "octahedron"
  | "icosahedron"
  | "custom";

export interface EdgeStyle {
  color: string;
  width: number;
  dashed: boolean;
  dashScale?: number;
  opacity: number;
}

export type TwinMode = "mirror" | "snapshot" | "sandbox" | "predictive";
export type SyncState = "synced" | "syncing" | "diverged" | "disconnected";
export type FidelityLevel = "full" | "reduced" | "minimal";

export type LayerType =
  | "input"
  | "dense"
  | "conv2d"
  | "conv1d"
  | "lstm"
  | "gru"
  | "attention"
  | "transformer"
  | "embedding"
  | "normalization"
  | "dropout"
  | "pooling"
  | "flatten"
  | "output"
  | "custom";

export type TrainingState =
  | "idle"
  | "training"
  | "validating"
  | "converged"
  | "diverged"
  | "failed";

export interface GradientStats {
  norm: number;
  maxAbs: number;
  minAbs: number;
  mean: number;
  variance: number;
  clipped: boolean;
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export function isForgeEntity(
  entity: UnifiedEntity
): entity is UnifiedEntity & { metadata: { forge: ForgeMetadata } } {
  return entity.metadata.forge !== undefined;
}

export function isTwinEntity(
  entity: UnifiedEntity
): entity is UnifiedEntity & { metadata: { twin: TwinMetadata } } {
  return entity.metadata.twin !== undefined;
}

export function isFoundryEntity(
  entity: UnifiedEntity
): entity is UnifiedEntity & { metadata: { foundry: FoundryMetadata } } {
  return entity.metadata.foundry !== undefined;
}

export function isCrossDomainEntity(entity: UnifiedEntity): boolean {
  const domains = [
    entity.metadata.forge !== undefined,
    entity.metadata.twin !== undefined,
    entity.metadata.foundry !== undefined,
  ].filter(Boolean).length;
  return domains >= 2;
}
