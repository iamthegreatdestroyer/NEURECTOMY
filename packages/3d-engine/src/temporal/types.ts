/**
 * @file Temporal Types - 4D Engine Type Definitions
 * @description Type definitions for timeline, state snapshots, and temporal navigation
 * @module @neurectomy/3d-engine/temporal
 * @agents @AXIOM @VELOCITY
 * @phase Phase 3 - Dimensional Forge
 * @step Step 4 - 4D Temporal Engine
 */

// ============================================================================
// Core Temporal Types
// ============================================================================

/**
 * Unique identifier for temporal entities
 */
export type TemporalId = string;

/**
 * Timestamp in milliseconds
 */
export type Timestamp = number;

/**
 * Duration in milliseconds
 */
export type Duration = number;

/**
 * Time range with start and end
 */
export interface TimeRange {
  start: Timestamp;
  end: Timestamp;
}

/**
 * Temporal resolution for playback
 */
export type TemporalResolution =
  | "millisecond" // 1ms precision
  | "frame" // ~16.67ms (60fps)
  | "centisecond" // 10ms
  | "decisecond" // 100ms
  | "second"; // 1000ms

// ============================================================================
// State Snapshot Types
// ============================================================================

/**
 * Complete state snapshot at a point in time
 */
export interface StateSnapshot<T = unknown> {
  /** Snapshot identifier */
  id: TemporalId;
  /** Timestamp when snapshot was taken */
  timestamp: Timestamp;
  /** The captured state */
  state: T;
  /** Snapshot metadata */
  metadata: SnapshotMetadata;
  /** Hash for quick comparison */
  hash: string;
  /** Parent snapshot ID (for delta encoding) */
  parentId?: TemporalId;
  /** Delta from parent (if using delta encoding) */
  delta?: StateDelta;
}

/**
 * Metadata about a snapshot
 */
export interface SnapshotMetadata {
  /** Human-readable label */
  label?: string;
  /** Description of state */
  description?: string;
  /** Tags for categorization */
  tags: string[];
  /** Is this a keyframe (full state) */
  isKeyframe: boolean;
  /** Size in bytes */
  sizeBytes: number;
  /** Compression type used */
  compression?: "none" | "lz4" | "zstd";
  /** Created automatically or manually */
  source: "auto" | "manual" | "checkpoint";
}

/**
 * Delta between two states
 */
export interface StateDelta {
  /** Operations to transform parent to this state */
  operations: DeltaOperation[];
  /** Size of delta in bytes */
  sizeBytes: number;
}

/**
 * Single delta operation
 */
export interface DeltaOperation {
  /** Operation type */
  type: "set" | "delete" | "insert" | "move" | "patch";
  /** Path to the affected property */
  path: string[];
  /** Value (for set, insert, patch) */
  value?: unknown;
  /** Previous value (for undo) */
  previousValue?: unknown;
  /** Source index (for move) */
  fromIndex?: number;
  /** Target index (for move, insert) */
  toIndex?: number;
}

// ============================================================================
// Timeline Types
// ============================================================================

/**
 * Timeline containing multiple snapshots
 */
export interface Timeline<T = unknown> {
  /** Timeline identifier */
  id: TemporalId;
  /** Timeline name */
  name: string;
  /** All snapshots in chronological order */
  snapshots: StateSnapshot<T>[];
  /** Keyframe indices for quick seeking */
  keyframeIndices: number[];
  /** Timeline bounds */
  bounds: TimeRange;
  /** Current playhead position */
  currentTime: Timestamp;
  /** Playback state */
  playback: PlaybackState;
  /** Configuration */
  config: TimelineConfig;
}

/**
 * Playback state
 */
export interface PlaybackState {
  /** Is timeline playing */
  isPlaying: boolean;
  /** Playback direction */
  direction: "forward" | "backward";
  /** Playback speed multiplier */
  speed: number;
  /** Loop mode */
  loopMode: LoopMode;
  /** Loop region (if looping) */
  loopRegion?: TimeRange;
}

/**
 * Loop modes for playback
 */
export type LoopMode =
  | "none" // No looping
  | "loop" // Loop entire timeline
  | "pingpong" // Bounce back and forth
  | "region"; // Loop within region

/**
 * Playback direction
 */
export type PlaybackDirection = "forward" | "backward";

/**
 * Keyframe index entry for efficient timeline navigation
 */
export interface KeyframeIndexEntry {
  /** Timestamp of the keyframe */
  timestamp: Timestamp;
  /** Associated snapshot ID */
  snapshotId: string;
  /** Type of keyframe */
  type: "auto" | "manual";
  /** Human-readable label */
  label: string;
}

/**
 * Keyframe index for efficient timeline navigation
 */
export interface KeyframeIndex {
  /** List of keyframes in order */
  keyframes: KeyframeIndexEntry[];
  /** Start time of indexed range */
  startTime: Timestamp;
  /** End time of indexed range */
  endTime: Timestamp;
  /** Total number of keyframes */
  totalKeyframes: number;
  /** Average interval between keyframes */
  averageInterval: number;
}

/**
 * Timeline configuration
 */
export interface TimelineConfig {
  /** Maximum snapshots to keep */
  maxSnapshots: number;
  /** Keyframe interval in ms */
  keyframeInterval: Duration;
  /** Auto-snapshot interval (0 = disabled) */
  autoSnapshotInterval: Duration;
  /** Use delta encoding */
  useDeltaEncoding: boolean;
  /** Compression for snapshots */
  compression: "none" | "lz4" | "zstd";
  /** Resolution for playback */
  resolution: TemporalResolution;
}

// ============================================================================
// Keyframe Types
// ============================================================================

/**
 * Keyframe marker on timeline
 */
export interface Keyframe {
  /** Keyframe identifier */
  id: TemporalId;
  /** Timestamp of keyframe */
  timestamp: Timestamp;
  /** Keyframe type */
  type: KeyframeType;
  /** Label */
  label?: string;
  /** Color for visual representation */
  color?: string;
  /** Associated snapshot ID */
  snapshotId: TemporalId;
  /** Easing function for interpolation to this keyframe */
  easing?: EasingFunction;
}

/**
 * Types of keyframes
 */
export type KeyframeType =
  | "auto" // Automatically created
  | "manual" // User created
  | "checkpoint" // Explicit checkpoint
  | "branch" // Timeline branch point
  | "merge" // Timeline merge point
  | "error" // Error state marker
  | "milestone"; // Important milestone

/**
 * Easing functions for interpolation
 */
export type EasingFunction =
  | "linear"
  | "easeIn"
  | "easeOut"
  | "easeInOut"
  | "easeInQuad"
  | "easeOutQuad"
  | "easeInOutQuad"
  | "easeInCubic"
  | "easeOutCubic"
  | "easeInOutCubic"
  | "easeInExpo"
  | "easeOutExpo"
  | "easeInOutExpo"
  | "spring";

// ============================================================================
// Animation Track Types
// ============================================================================

/**
 * Animation track for a specific property
 */
export interface AnimationTrack<T = number> {
  /** Track identifier */
  id: TemporalId;
  /** Target entity ID */
  targetId: string;
  /** Property path being animated */
  propertyPath: string[];
  /** Keyframes on this track */
  keyframes: TrackKeyframe<T>[];
  /** Track enabled */
  enabled: boolean;
  /** Track locked (prevents editing) */
  locked: boolean;
  /** Interpolation mode */
  interpolation: InterpolationMode;
}

/**
 * Keyframe on an animation track
 */
export interface TrackKeyframe<T = number> {
  /** Timestamp */
  time: Timestamp;
  /** Value at this keyframe */
  value: T;
  /** Easing to next keyframe */
  easing: EasingFunction;
  /** Tangent handles (for bezier curves) */
  tangents?: {
    in: { x: number; y: number };
    out: { x: number; y: number };
  };
}

/**
 * Interpolation mode for tracks
 */
export type InterpolationMode =
  | "step" // No interpolation, jump between values
  | "linear" // Linear interpolation
  | "bezier" // Bezier curve interpolation
  | "catmullrom"; // Catmull-Rom spline

// ============================================================================
// Temporal Events
// ============================================================================

/**
 * Event that occurred at a point in time
 */
export interface TemporalEvent {
  /** Event identifier */
  id: TemporalId;
  /** Event timestamp */
  timestamp: Timestamp;
  /** Event type */
  type: string;
  /** Event payload */
  payload: unknown;
  /** Source entity ID */
  sourceId?: string;
  /** Target entity ID */
  targetId?: string;
  /** Duration of event (if applicable) */
  duration?: Duration;
  /** Event metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Event marker on timeline
 */
export interface EventMarker {
  /** Event ID */
  eventId: TemporalId;
  /** Display label */
  label: string;
  /** Icon identifier */
  icon?: string;
  /** Color */
  color: string;
  /** Priority for display */
  priority: number;
}

/**
 * Timeline marker for various marker types
 */
export interface TimelineMarker {
  /** Marker ID */
  id: string;
  /** Marker timestamp */
  time: Timestamp;
  /** Marker type */
  type: "keyframe" | "event" | "annotation" | "checkpoint" | "error";
  /** Display label */
  label: string;
  /** Marker color */
  color: string;
  /** Optional icon */
  icon?: string;
  /** Additional data */
  data?: Record<string, unknown>;
}

/**
 * Keyframe marker extending TimelineMarker
 */
export interface KeyframeMarker extends TimelineMarker {
  /** Type is always keyframe */
  type: "keyframe";
  /** Associated snapshot ID */
  snapshotId: string;
  /** Whether automatically generated */
  isAutomatic: boolean;
}

// ============================================================================
// Branching & Time Travel Types
// ============================================================================

/**
 * Timeline branch for alternative histories
 */
export interface TimelineBranch {
  /** Branch identifier */
  id: TemporalId;
  /** Branch name */
  name: string;
  /** Parent branch ID */
  parentId?: TemporalId;
  /** Branch point timestamp */
  branchPoint: Timestamp;
  /** Branch creation time */
  createdAt: Timestamp;
  /** Is this the active branch */
  isActive: boolean;
  /** Branch metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Time travel debug state
 */
export interface TimeTravelState {
  /** Current branch */
  currentBranch: TemporalId;
  /** All branches */
  branches: TimelineBranch[];
  /** History stack for undo */
  undoStack: TemporalId[];
  /** History stack for redo */
  redoStack: TemporalId[];
  /** Bookmarked snapshots */
  bookmarks: TemporalId[];
  /** Comparison mode */
  comparisonMode?: {
    snapshotA: TemporalId;
    snapshotB: TemporalId;
  };
}

// ============================================================================
// Renderer Types
// ============================================================================

/**
 * Timeline renderer configuration
 */
export interface TimelineRendererConfig {
  /** Height of timeline in pixels */
  height: number;
  /** Pixels per second */
  pixelsPerSecond: number;
  /** Min zoom level */
  minZoom: number;
  /** Max zoom level */
  maxZoom: number;
  /** Current zoom level */
  zoom: number;
  /** Show ruler */
  showRuler: boolean;
  /** Show keyframe markers */
  showKeyframes: boolean;
  /** Show event markers */
  showEvents: boolean;
  /** Show waveform (if applicable) */
  showWaveform: boolean;
  /** Snap to grid */
  snapToGrid: boolean;
  /** Grid interval in ms */
  gridInterval: Duration;
  /** Color scheme */
  colors: TimelineColors;
}

/**
 * Timeline color scheme
 */
export interface TimelineColors {
  background: string;
  ruler: string;
  rulerText: string;
  playhead: string;
  keyframe: string;
  keyframeSelected: string;
  selectionRange: string;
  loopRegion: string;
  grid: string;
  waveform: string;
}

// ============================================================================
// Scrubbing Types
// ============================================================================

/**
 * Scrubbing state
 */
export interface ScrubbingState {
  /** Is currently scrubbing */
  isScrubbing: boolean;
  /** Scrub start position */
  scrubStart?: Timestamp;
  /** Current scrub position */
  scrubPosition?: Timestamp;
  /** Scrub velocity (for momentum) */
  velocity: number;
  /** Preview mode during scrub */
  previewMode: "immediate" | "debounced" | "keyframes-only";
}

/**
 * Seek request
 */
export interface SeekRequest {
  /** Target timestamp */
  timestamp: Timestamp;
  /** Seek mode */
  mode: SeekMode;
  /** Callback when seek completes */
  onComplete?: () => void;
}

/**
 * Seek modes
 */
export type SeekMode =
  | "exact" // Seek to exact timestamp
  | "nearest" // Seek to nearest snapshot
  | "keyframe" // Seek to nearest keyframe
  | "prev" // Seek to previous keyframe
  | "next"; // Seek to next keyframe

// ============================================================================
// Performance Types
// ============================================================================

/**
 * Temporal performance metrics
 */
export interface TemporalMetrics {
  /** Total snapshots in memory */
  snapshotCount: number;
  /** Total memory usage */
  memoryUsageBytes: number;
  /** Average snapshot size */
  avgSnapshotSize: number;
  /** Keyframe count */
  keyframeCount: number;
  /** Last seek time */
  lastSeekTimeMs: number;
  /** Average seek time */
  avgSeekTimeMs: number;
  /** Compression ratio (if using compression) */
  compressionRatio?: number;
  /** Cache hit rate */
  cacheHitRate: number;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Result of a temporal operation
 */
export type TemporalResult<T> =
  | { success: true; data: T }
  | { success: false; error: TemporalError };

/**
 * Temporal error
 */
export interface TemporalError {
  code: TemporalErrorCode;
  message: string;
  timestamp?: Timestamp;
  snapshotId?: TemporalId;
}

/**
 * Error codes for temporal operations
 */
export type TemporalErrorCode =
  | "SNAPSHOT_NOT_FOUND"
  | "INVALID_TIMESTAMP"
  | "OUT_OF_BOUNDS"
  | "DELTA_DECODE_FAILED"
  | "COMPRESSION_FAILED"
  | "MEMORY_LIMIT_EXCEEDED"
  | "BRANCH_NOT_FOUND"
  | "MERGE_CONFLICT"
  | "INVALID_OPERATION";

export default {
  // Export type guards and utilities
};
