/**
 * 4D Timeline Component Types
 *
 * @module @neurectomy/3d-engine/temporal/components/types
 */

import type { TimelineMarker, TimeRange, Timestamp } from "../types";
import type { PlaybackStatus } from "../timeline-navigator";

// ============================================================================
// Theme & Styling
// ============================================================================

export interface TimelineTheme {
  /** Background color */
  background: string;
  /** Track color */
  track: string;
  /** Progress/filled track color */
  progress: string;
  /** Playhead color */
  playhead: string;
  /** Keyframe marker color */
  keyframe: string;
  /** Event marker color */
  event: string;
  /** Selection color */
  selection: string;
  /** Text color */
  text: string;
  /** Muted text color */
  textMuted: string;
  /** Border color */
  border: string;
  /** Active/hover state color */
  hover: string;
}

export const DEFAULT_TIMELINE_THEME: TimelineTheme = {
  background: "#1a1a2e",
  track: "#2d2d44",
  progress: "#4a90d9",
  playhead: "#ff6b6b",
  keyframe: "#ffd700",
  event: "#7fb3e8",
  selection: "rgba(74, 144, 217, 0.3)",
  text: "#ffffff",
  textMuted: "#888888",
  border: "#404060",
  hover: "#4a90d9",
};

// ============================================================================
// Timeline Component Props
// ============================================================================

export interface TimelineProps {
  /** Current time in milliseconds */
  currentTime: Timestamp;
  /** Time range to display */
  timeRange: TimeRange;
  /** Playback status */
  playbackStatus: PlaybackStatus;
  /** Timeline markers (keyframes, events, etc.) */
  markers?: TimelineMarker[];
  /** Whether the timeline is currently scrubbing */
  isScrubbing?: boolean;

  // Callbacks
  /** Called when user seeks to a time */
  onSeek?: (time: Timestamp) => void;
  /** Called when playback state changes */
  onPlaybackChange?: (status: Partial<PlaybackStatus>) => void;
  /** Called when user starts scrubbing */
  onScrubStart?: () => void;
  /** Called while scrubbing */
  onScrub?: (time: Timestamp) => void;
  /** Called when scrubbing ends */
  onScrubEnd?: () => void;
  /** Called when a marker is clicked */
  onMarkerClick?: (marker: TimelineMarker) => void;
  /** Called when time range changes (zoom/pan) */
  onRangeChange?: (range: TimeRange) => void;

  // Display options
  /** Show time ruler */
  showRuler?: boolean;
  /** Show keyframe markers */
  showKeyframes?: boolean;
  /** Show event markers */
  showEvents?: boolean;
  /** Show thumbnail preview on hover */
  showThumbnails?: boolean;
  /** Show minimap */
  showMinimap?: boolean;
  /** Height of the timeline in pixels */
  height?: number;
  /** Custom theme */
  theme?: Partial<TimelineTheme>;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// TimelineScrubber Props
// ============================================================================

export interface TimelineScrubberProps {
  /** Current time position (0-1 normalized) */
  position: number;
  /** Buffer/loaded range (0-1) */
  buffered?: number;
  /** Whether scrubbing is active */
  isScrubbing?: boolean;
  /** Whether to show tooltip with time */
  showTooltip?: boolean;
  /** Format function for tooltip time */
  formatTime?: (time: Timestamp) => string;
  /** Current time for tooltip */
  currentTime?: Timestamp;

  // Callbacks
  onScrubStart?: () => void;
  onScrub?: (position: number) => void;
  onScrubEnd?: () => void;
  onClick?: (position: number) => void;

  // Styling
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// PlaybackControls Props
// ============================================================================

export interface PlaybackControlsProps {
  /** Current playback status */
  status: PlaybackStatus;
  /** Whether to show extended controls */
  extended?: boolean;
  /** Whether to show speed control inline */
  showSpeedControl?: boolean;
  /** Whether to show loop button */
  showLoopControl?: boolean;
  /** Size variant */
  size?: "sm" | "md" | "lg";

  // Callbacks
  onPlay?: () => void;
  onPause?: () => void;
  onStop?: () => void;
  onStepForward?: () => void;
  onStepBackward?: () => void;
  onNextKeyframe?: () => void;
  onPrevKeyframe?: () => void;
  onSpeedChange?: (speed: number) => void;
  onDirectionChange?: (direction: "forward" | "backward") => void;
  onLoopToggle?: () => void;

  // Styling
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// KeyframeMarkers Props
// ============================================================================

export interface KeyframeMarkersProps {
  /** Array of keyframe markers */
  keyframes: TimelineMarker[];
  /** Current time */
  currentTime: Timestamp;
  /** Visible time range */
  timeRange: TimeRange;
  /** Selected keyframe ID */
  selectedId?: string;
  /** Whether keyframes are interactive */
  interactive?: boolean;

  // Callbacks
  onClick?: (keyframe: TimelineMarker) => void;
  onDoubleClick?: (keyframe: TimelineMarker) => void;
  onHover?: (keyframe: TimelineMarker | null) => void;
  onContextMenu?: (keyframe: TimelineMarker, event: React.MouseEvent) => void;

  // Styling
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// TimeRuler Props
// ============================================================================

export interface TimeRulerProps {
  /** Visible time range */
  timeRange: TimeRange;
  /** Current time */
  currentTime?: Timestamp;
  /** Tick density ('sparse' | 'normal' | 'dense') */
  density?: "sparse" | "normal" | "dense";
  /** Time format ('ms' | 'seconds' | 'hms' | 'relative') */
  format?: "ms" | "seconds" | "hms" | "relative";
  /** Show minor ticks */
  showMinorTicks?: boolean;

  // Callbacks
  onClick?: (time: Timestamp) => void;

  // Styling
  height?: number;
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// SpeedControl Props
// ============================================================================

export interface SpeedControlProps {
  /** Current speed multiplier */
  speed: number;
  /** Preset speed options */
  presets?: number[];
  /** Min speed */
  min?: number;
  /** Max speed */
  max?: number;
  /** Step increment for slider */
  step?: number;
  /** Display as slider or buttons */
  variant?: "slider" | "buttons" | "dropdown";

  // Callbacks
  onChange?: (speed: number) => void;

  // Styling
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// SnapshotThumbnail Props
// ============================================================================

export interface SnapshotThumbnailProps {
  /** Snapshot ID */
  snapshotId: string;
  /** Time of snapshot */
  time: Timestamp;
  /** Thumbnail image URL or data */
  thumbnail?: string;
  /** Whether this is the current snapshot */
  isCurrent?: boolean;
  /** Width in pixels */
  width?: number;
  /** Height in pixels */
  height?: number;

  // Callbacks
  onClick?: (snapshotId: string) => void;
  onLoad?: () => void;

  // Styling
  theme?: Partial<TimelineTheme>;
  className?: string;
}

// ============================================================================
// Utility Types
// ============================================================================

export interface TimelineViewport {
  /** Start time of visible range */
  start: Timestamp;
  /** End time of visible range */
  end: Timestamp;
  /** Zoom level (1 = 100%) */
  zoom: number;
  /** Pan offset in pixels */
  panOffset: number;
}

export interface TimelineSelection {
  /** Selection start time */
  start: Timestamp;
  /** Selection end time */
  end: Timestamp;
  /** Whether selection is active */
  active: boolean;
}
