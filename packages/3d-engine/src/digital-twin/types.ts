/**
 * Digital Twin System Types
 *
 * Type definitions for the agent digital twin architecture.
 * Digital twins are virtual replicas of agents that enable
 * parallel experimentation, prediction, and comparison.
 *
 * @module @neurectomy/3d-engine/digital-twin/types
 * @agents @ARCHITECT @NEURAL
 * @phase Phase 3 - Dimensional Forge
 */

// ============================================================================
// Core Twin Types
// ============================================================================

/** Unique identifier for a digital twin */
export type TwinId = string;

/** Unique identifier for a source agent */
export type AgentId = string;

/** Twin synchronization state */
export type SyncState = "synced" | "syncing" | "diverged" | "disconnected";

/** Twin operation mode */
export type TwinMode =
  | "mirror" // Real-time sync with source agent
  | "snapshot" // Frozen state at a point in time
  | "sandbox" // Independent experimentation mode
  | "predictive"; // Running simulations ahead

/** Twin fidelity level */
export type TwinFidelity = "full" | "reduced" | "minimal";

// ============================================================================
// Twin State Interfaces
// ============================================================================

/**
 * Complete state of a digital twin
 */
export interface TwinState {
  /** Unique twin identifier */
  id: TwinId;
  /** Source agent identifier */
  agentId: AgentId;
  /** Human-readable twin name */
  name: string;
  /** Twin operation mode */
  mode: TwinMode;
  /** Current sync state */
  syncState: SyncState;
  /** Fidelity level */
  fidelity: TwinFidelity;
  /** Twin creation timestamp */
  createdAt: number;
  /** Last sync timestamp */
  lastSyncAt: number;
  /** Last modification timestamp */
  modifiedAt: number;
  /** Agent state snapshot */
  agentState: AgentStateSnapshot;
  /** Twin metadata */
  metadata: TwinMetadata;
  /** Divergence from source (0-1) */
  divergenceScore: number;
}

/**
 * Snapshot of agent state for twin replication
 */
export interface AgentStateSnapshot {
  /** Agent configuration */
  config: Record<string, unknown>;
  /** Agent parameters/weights */
  parameters: Record<string, unknown>;
  /** Internal state variables */
  internalState: Record<string, unknown>;
  /** Input/output history */
  ioHistory: IOHistoryEntry[];
  /** Performance metrics */
  metrics: AgentMetrics;
  /** Component graph */
  componentGraph: ComponentGraphSnapshot;
  /** Agent capabilities */
  capabilities?: string[];
  /** Agent connections to other agents/systems */
  connections?: string[];
}

/**
 * Input/output history entry
 */
export interface IOHistoryEntry {
  timestamp: number;
  type: "input" | "output";
  data: unknown;
  context?: Record<string, unknown>;
}

/**
 * Agent performance metrics
 */
export interface AgentMetrics {
  /** Response time statistics */
  responseTime: MetricSummary;
  /** Throughput statistics */
  throughput: MetricSummary;
  /** Error rate */
  errorRate: number;
  /** Resource utilization */
  resourceUtilization: ResourceMetrics;
  /** Custom metrics */
  custom: Record<string, number>;
}

/**
 * Statistical summary of a metric
 */
export interface MetricSummary {
  min: number;
  max: number;
  mean: number;
  median: number;
  p95: number;
  p99: number;
  stdDev: number;
}

/**
 * Resource utilization metrics
 */
export interface ResourceMetrics {
  cpuPercent: number;
  memoryMB: number;
  gpuPercent?: number;
  gpuMemoryMB?: number;
  networkBytesIn: number;
  networkBytesOut: number;
}

/**
 * Snapshot of agent component graph
 */
export interface ComponentGraphSnapshot {
  nodes: ComponentNode[];
  edges: ComponentEdge[];
  rootId: string;
}

/**
 * Component node in the graph
 */
export interface ComponentNode {
  id: string;
  type: string;
  name: string;
  state: "active" | "idle" | "error" | "disabled";
  config: Record<string, unknown>;
  position?: { x: number; y: number; z: number };
}

/**
 * Component edge in the graph
 */
export interface ComponentEdge {
  id: string;
  sourceId: string;
  targetId: string;
  type: "data" | "control" | "dependency";
  weight?: number;
}

/**
 * Twin metadata
 */
export interface TwinMetadata {
  /** Description */
  description?: string;
  /** Tags for organization */
  tags: string[];
  /** Creator identifier */
  createdBy?: string;
  /** Version identifier */
  version: string;
  /** Parent twin ID (for derived twins) */
  parentTwinId?: TwinId;
  /** Branch name (for experimentation) */
  branch?: string;
  /** Custom properties */
  properties: Record<string, unknown>;
}

// ============================================================================
// Synchronization Types
// ============================================================================

/**
 * Sync configuration for a digital twin
 */
export interface SyncConfig {
  /** Sync mode */
  mode: "realtime" | "periodic" | "manual";
  /** Sync interval in ms (for periodic mode) */
  intervalMs?: number;
  /** Fields to sync (empty = all) */
  syncFields?: string[];
  /** Fields to exclude from sync */
  excludeFields?: string[];
  /** Conflict resolution strategy */
  conflictResolution: "source-wins" | "twin-wins" | "merge" | "manual";
  /** Enable compression for sync data */
  compression?: boolean;
  /** Batch sync operations */
  batchSize?: number;
}

/**
 * Sync operation result
 */
export interface SyncResult {
  success: boolean;
  timestamp: number;
  changesApplied: number;
  conflicts: SyncConflict[];
  bytesTransferred: number;
  durationMs: number;
  error?: string;
}

/**
 * Sync conflict details
 */
export interface SyncConflict {
  /** Path to the conflicting field */
  path?: string;
  field?: string;
  /** Source/local value */
  sourceValue?: unknown;
  localValue?: unknown;
  /** Twin/remote value */
  twinValue?: unknown;
  remoteValue?: unknown;
  /** How the conflict was resolved */
  resolution?: ConflictResolution | "source" | "twin" | "merged";
  /** The value after merging */
  mergedValue?: unknown;
  resolvedValue?: unknown;
}

/**
 * Sync event for real-time updates
 */
export interface SyncEvent {
  type: "state-change" | "config-change" | "metric-update" | "error";
  timestamp: number;
  agentId: AgentId;
  twinId: TwinId;
  path: string;
  previousValue?: unknown;
  newValue: unknown;
}

// ============================================================================
// Prediction Types
// ============================================================================

/**
 * Prediction configuration
 */
export interface PredictionConfig {
  /** Simulation horizon (how far to predict) */
  horizonMs: number;
  /** Simulation step size */
  stepMs: number;
  /** Number of parallel scenarios */
  scenarioCount: number;
  /** Input scenarios to simulate */
  inputScenarios: InputScenario[];
  /** Enable uncertainty quantification */
  quantifyUncertainty?: boolean;
  /** Confidence level for predictions */
  confidenceLevel?: number;
}

/**
 * Input scenario for prediction
 */
export interface InputScenario {
  id: string;
  name: string;
  description?: string;
  inputs: PredictedInput[];
  probability?: number;
}

/**
 * Predicted input event
 */
export interface PredictedInput {
  timestamp: number;
  data: unknown;
  probability?: number;
}

/**
 * Prediction result
 */
export interface PredictionResult {
  /** Unique identifier for this prediction result */
  id?: string;
  /** Associated twin ID */
  twinId: TwinId;
  /** Configuration used for prediction */
  config?: PredictionConfig;
  /** Start time of prediction (legacy) */
  startTime?: number;
  /** End time of prediction (legacy) */
  endTime?: number;
  /** Timestamp when prediction was made */
  timestamp?: number;
  /** Prediction horizon in milliseconds */
  horizonMs?: number;
  /** Prediction timeline with points and branches */
  timeline?: PredictionTimeline;
  /** Scenario results (detailed) */
  scenarioResults?: ScenarioResult[];
  /** Scenarios (simplified) */
  scenarios?: Scenario[] | ScenarioResult[];
  /** Prediction metrics */
  metrics?: PredictionMetrics | AgentMetrics;
  /** Summary of prediction */
  summary?: PredictionSummary;
}

/**
 * Result of a single scenario simulation
 */
export interface ScenarioResult {
  scenarioId: string;
  trajectory: StateTrajectory;
  outcomes: PredictedOutcome[];
  metrics: AgentMetrics;
  confidence: number;
}

/**
 * State trajectory over time
 */
export interface StateTrajectory {
  timestamps: number[];
  states: AgentStateSnapshot[];
  interpolated: boolean;
}

/**
 * Predicted outcome
 */
export interface PredictedOutcome {
  timestamp: number;
  type: string;
  value: unknown;
  probability: number;
  confidence: number;
}

/**
 * Summary of all predictions
 */
export interface PredictionSummary {
  mostLikelyOutcome: string;
  outcomeDistribution: Record<string, number>;
  riskScore: number;
  opportunityScore: number;
  recommendedActions: string[];
}

// ============================================================================
// Comparison Types
// ============================================================================

/**
 * Twin comparison request
 */
export interface ComparisonRequest {
  /** First twin or agent ID */
  sourceId: TwinId | AgentId;
  /** Second twin or agent ID */
  targetId: TwinId | AgentId;
  /** Fields to compare (empty = all) */
  fields?: string[];
  /** Comparison options */
  options?: ComparisonOptions;
}

/**
 * Comparison options
 */
export interface ComparisonOptions {
  /** Include deep comparison of nested objects */
  deep?: boolean;
  /** Tolerance for numeric comparisons */
  numericTolerance?: number;
  /** Ignore order in arrays */
  ignoreArrayOrder?: boolean;
  /** Include metrics comparison */
  includeMetrics?: boolean;
  /** Include component graph diff */
  includeGraphDiff?: boolean;
}

/**
 * Twin comparison result
 */
export interface ComparisonResult {
  sourceId: string;
  targetId: string;
  timestamp: number;
  identical: boolean;
  similarityScore: number;
  differences: FieldDifference[];
  graphDiff?: GraphDiff;
  metricsDiff?: MetricsDiff;
}

/**
 * Difference in a specific field
 */
export interface FieldDifference {
  path: string;
  type: "added" | "removed" | "changed" | "type-mismatch";
  sourceValue?: unknown;
  targetValue?: unknown;
  significance: "low" | "medium" | "high" | "critical";
}

/**
 * Difference in component graphs
 */
export interface GraphDiff {
  addedNodes: ComponentNode[];
  removedNodes: ComponentNode[];
  changedNodes: { id: string; changes: FieldDifference[] }[];
  addedEdges: ComponentEdge[];
  removedEdges: ComponentEdge[];
  changedEdges: { id: string; changes: FieldDifference[] }[];
}

/**
 * Difference in metrics
 */
export interface MetricsDiff {
  changes: {
    metric: string;
    sourceValue: number;
    targetValue: number;
    percentChange: number;
    significance: "low" | "medium" | "high";
  }[];
  overallDrift: number;
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Twin lifecycle events
 */
export type TwinEvent =
  | { type: "twin:created"; twin: TwinState }
  | { type: "twin:updated"; twinId: TwinId; changes: Partial<TwinState> }
  | { type: "twin:deleted"; twinId: TwinId }
  | { type: "twin:synced"; twinId: TwinId; result: SyncResult }
  | { type: "twin:diverged"; twinId: TwinId; divergenceScore: number }
  | {
      type: "twin:prediction-complete";
      twinId: TwinId;
      result: PredictionResult;
    }
  | {
      type: "twin:mode-changed";
      twinId: TwinId;
      oldMode: TwinMode;
      newMode: TwinMode;
    }
  | { type: "twin:error"; twinId: TwinId; error: string };

/**
 * Twin event listener
 */
export type TwinEventListener = (event: TwinEvent) => void;

// ============================================================================
// Manager Types
// ============================================================================

/**
 * Twin manager configuration
 */
export interface TwinManagerConfig {
  /** Maximum number of twins */
  maxTwins?: number;
  /** Default sync configuration */
  defaultSyncConfig?: Partial<SyncConfig>;
  /** Default prediction configuration */
  defaultPredictionConfig?: Partial<PredictionConfig>;
  /** Enable auto-cleanup of stale twins */
  autoCleanup?: boolean;
  /** Stale twin threshold (ms) */
  staleThresholdMs?: number;
  /** Enable persistence */
  persistence?: boolean;
  /** Storage backend */
  storageBackend?: "memory" | "indexeddb" | "file";
}

/**
 * Twin query options
 */
export interface TwinQuery {
  /** Filter by agent ID */
  agentId?: AgentId;
  /** Filter by mode */
  mode?: TwinMode;
  /** Filter by sync state */
  syncState?: SyncState;
  /** Filter by tags */
  tags?: string[];
  /** Filter by creation date range */
  createdAfter?: number;
  createdBefore?: number;
  /** Sort field */
  sortBy?: "createdAt" | "modifiedAt" | "name" | "divergenceScore";
  /** Sort direction */
  sortDirection?: "asc" | "desc";
  /** Pagination */
  limit?: number;
  offset?: number;
}

/**
 * Twin statistics
 */
export interface TwinStatistics {
  totalTwins: number;
  twinsByMode: Record<TwinMode, number>;
  twinsBySyncState: Record<SyncState, number>;
  averageDivergence: number;
  totalSyncs: number;
  totalPredictions: number;
  storageUsedBytes: number;
}

// ============================================================================
// Additional Type Aliases (for export consistency)
// ============================================================================

/**
 * Time series metric for tracking values over time
 */
export interface TimeSeriesMetric {
  name: string;
  timestamps: number[];
  values: number[];
  unit?: string;
  aggregationType?: "sum" | "average" | "max" | "min" | "last";
}

/**
 * Resource utilization snapshot
 */
export interface ResourceUtilization {
  cpu: number;
  memory: number;
  gpu?: number;
  disk?: number;
  network?: number;
  timestamp: number;
}

/**
 * Synchronization mode type
 */
export type SyncMode = "realtime" | "periodic" | "manual" | "on-demand";

/**
 * Conflict resolution strategy type
 */
export type ConflictResolution =
  | "source-wins"
  | "twin-wins"
  | "merge"
  | "manual"
  | "last-write-wins"
  | "latest-wins";

/**
 * Scenario input for prediction simulations
 */
export interface ScenarioInput {
  id?: string;
  name?: string;
  description?: string;
  inputs?: Array<{
    timestamp: number;
    data: unknown;
    probability?: number;
  }>;
  weight?: number;
  tags?: string[];
  /** Configuration overrides for this scenario */
  configOverrides?: Record<string, unknown>;
  /** Parameter changes to apply */
  parameterChanges?: Record<string, number>;
  /** Load multiplier (e.g., 2x traffic) */
  loadMultiplier?: number;
  /** Inject failures into the scenario */
  failureInjection?: boolean;
  /** Probability of this scenario input (0-1) */
  probability?: number;
}

/**
 * Twin sync result (alias for SyncResult for API consistency)
 */
export type TwinSyncResult = SyncResult;

/**
 * Prediction timeline for visualization
 */
export interface PredictionTimeline {
  startTime: number;
  endTime: number;
  points: TimelinePoint[];
  branches: TimelineBranch[];
  /** Step size in milliseconds between points */
  stepMs?: number;
}

/**
 * Single point in a prediction timeline
 */
export interface TimelinePoint {
  timestamp: number;
  state: Partial<AgentStateSnapshot>;
  confidence: number;
  scenarioId?: string;
  metadata?: Record<string, unknown>;
  /** Confidence bounds - can be simple numeric or full state snapshots */
  bounds?: {
    lower: number | AgentStateSnapshot;
    upper: number | AgentStateSnapshot;
  };
}

/**
 * Branch in a prediction timeline (for divergent scenarios)
 */
export interface TimelineBranch {
  id: string;
  parentPointIndex: number;
  points: TimelinePoint[];
  probability: number;
}

/**
 * Scenario definition for prediction
 */
export interface Scenario {
  id: string;
  name: string;
  description?: string;
  /** Array of inputs for batch scenarios */
  inputs?: ScenarioInput[];
  /** Single input for simple scenarios */
  input?: Partial<ScenarioInput>;
  constraints?: ScenarioConstraint[];
  weight?: number;
  tags?: string[];
  /** Probability of this scenario occurring (0-1) */
  probability?: number;
  /** Predicted state after applying this scenario */
  predictedState?: AgentStateSnapshot;
  /** Key metrics extracted from the predicted state */
  keyMetrics?: Record<string, number>;
}

/**
 * Constraint for a scenario
 */
export interface ScenarioConstraint {
  type: "min" | "max" | "range" | "exact";
  field: string;
  value: number | [number, number];
}

/**
 * Prediction performance metrics
 */
export interface PredictionMetrics {
  accuracy: number;
  confidence?: number;
  computeTimeMs?: number;
  scenariosEvaluated?: number;
  convergenceRate?: number;
  memoryUsedMB?: number;
  /** Precision metric (TP / (TP + FP)) */
  precision?: number;
  /** Recall metric (TP / (TP + FN)) */
  recall?: number;
  /** F1 score (harmonic mean of precision and recall) */
  f1Score?: number;
  /** Percentage of scenarios covered */
  coveragePercent?: number;
  /** Number of data points used in prediction */
  dataPointsUsed?: number;
}

// ============================================================================
// Diff Types
// ============================================================================

/**
 * Delta representing changes between two twin states
 */
export interface TwinDelta {
  twinId: TwinId;
  fromTimestamp: number;
  toTimestamp: number;
  changes: TwinDiffEntry[];
  summary: DeltaSummary;
}

/**
 * Summary of a delta
 */
export interface DeltaSummary {
  totalChanges: number;
  addedFields: number;
  removedFields: number;
  modifiedFields: number;
  significanceScore: number;
}

/**
 * Full diff between two twins or versions
 */
export interface TwinDiff {
  /** Twin ID (for single-twin version diffs) */
  twinId?: TwinId;
  /** Source twin ID (for cross-twin diffs) */
  sourceTwinId?: TwinId;
  /** Target twin ID (for cross-twin diffs) */
  targetTwinId?: TwinId;
  /** Starting version number */
  fromVersion?: number;
  /** Ending version number */
  toVersion?: number;
  timestamp: number;
  entries: TwinDiffEntry[];
  similarity?: number;
  divergenceScore?: number;
}

/**
 * Single entry in a twin diff
 */
export interface TwinDiffEntry {
  path: string;
  type: "add" | "remove" | "modify" | "added" | "removed" | "changed" | "moved";
  oldValue?: unknown;
  newValue?: unknown;
  sourceValue?: unknown;
  targetValue?: unknown;
  conflict?: boolean;
  significance?: "low" | "medium" | "high" | "critical";
  category?: string;
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * Error codes for twin operations
 */
export type TwinErrorCode =
  | "TWIN_NOT_FOUND"
  | "TWIN_ALREADY_EXISTS"
  | "SYNC_FAILED"
  | "SYNC_CONFLICT"
  | "PREDICTION_FAILED"
  | "INVALID_STATE"
  | "INVALID_CONFIG"
  | "STORAGE_ERROR"
  | "NETWORK_ERROR"
  | "TIMEOUT"
  | "PERMISSION_DENIED"
  | "QUOTA_EXCEEDED"
  | "INVALID_OPERATION";

/**
 * Twin-specific error
 */
export interface TwinError {
  code: TwinErrorCode;
  message: string;
  twinId?: TwinId;
  operation?: string;
  timestamp: number;
  details?: Record<string, unknown>;
  cause?: Error;
}
