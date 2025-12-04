/**
 * 4D Timeline UI Components
 *
 * React components for temporal navigation through agent evolution history.
 * Provides intuitive controls for timeline scrubbing, keyframe navigation,
 * and playback control.
 *
 * @module @neurectomy/3d-engine/temporal/components
 * @agents @CANVAS @APEX
 * @phase Phase 3 - Dimensional Forge
 * @step Step 7 - 4D Timeline Interface
 */

export { Timeline } from "./Timeline";
export { TimelineScrubber } from "./TimelineScrubber";
export { PlaybackControls } from "./PlaybackControls";
export { KeyframeMarkers } from "./KeyframeMarkers";
export { TimeRuler } from "./TimeRuler";
export { SpeedControl } from "./SpeedControl";
export { SnapshotThumbnail } from "./SnapshotThumbnail";

// Types
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
} from "./types";

// Re-export default theme
export { DEFAULT_TIMELINE_THEME } from "./types";
