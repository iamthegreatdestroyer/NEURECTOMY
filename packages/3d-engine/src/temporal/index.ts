/**
 * Temporal/4D Engine Module
 *
 * Provides 4D temporal navigation and visualization capabilities.
 * Includes timeline navigation, state snapshots, and UI components.
 *
 * @module @neurectomy/3d-engine/temporal
 * @agents @NEXUS @ARCHITECT
 * @phase Phase 3 - Dimensional Forge
 */

// Core types - export with renamed conflicting types
export type {
  TemporalId,
  Timestamp,
  Duration,
  TimeRange,
  TemporalResolution,
  StateSnapshot,
  SnapshotMetadata,
  StateDelta,
  DeltaOperation,
  // Rename conflicting types to avoid collision with components
  Timeline as TimelineData,
  PlaybackState as PlaybackStateData,
  LoopMode,
  TimelineConfig as TimelineDataConfig,
  Keyframe as KeyframeData,
  KeyframeType,
  EasingFunction,
  AnimationTrack,
  TrackKeyframe,
  InterpolationMode,
  TemporalEvent,
  EventMarker,
  TimelineBranch,
  TimeTravelState,
  TimelineRendererConfig,
  TimelineColors,
  ScrubbingState,
  SeekRequest,
  SeekMode,
  TemporalMetrics,
  TemporalResult,
  TemporalError,
  TemporalErrorCode,
} from "./types";

// Timeline navigator - exclude PlaybackState (conflicts with types.ts)
export {
  TimelineNavigator,
  createTimelineNavigator,
  DEFAULT_TIMELINE_CONFIG,
} from "./timeline-navigator";

export type {
  TimelineNavigatorConfig,
  TimelineEvents,
  TimelineMarker,
  KeyframeMarker,
  // Re-export navigator's PlaybackState with a different name
  PlaybackState as PlaybackMode,
} from "./timeline-navigator";

// State snapshot system
export * from "./state-snapshot";

// UI Components (includes Timeline component and TimelineProps)
export * from "./components";
