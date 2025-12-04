/**
 * @file Timeline Navigator - 4D Temporal Navigation System
 * @description Core navigation system for traversing agent evolution timeline
 * @module @neurectomy/3d-engine/temporal
 * @agents @STREAM @CANVAS
 * @phase Phase 3 - Dimensional Forge
 * @step Step 4 - 4D Temporal Engine
 */

import type {
  Timestamp,
  TimeRange,
  StateSnapshot,
  TemporalResolution,
  KeyframeIndex,
  InterpolationMode,
  PlaybackDirection,
  TimelineMarker,
  KeyframeMarker,
} from "./types";

// Re-export marker types for backwards compatibility
export type { TimelineMarker, KeyframeMarker } from "./types";

// ============================================================================
// Browser-compatible Event Emitter
// ============================================================================

type EventCallback = (...args: unknown[]) => void;

/**
 * Simple browser-compatible event emitter
 */
class BrowserEventEmitter {
  private events: Map<string, Set<EventCallback>> = new Map();

  on(event: string, callback: EventCallback): this {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }
    this.events.get(event)!.add(callback);
    return this;
  }

  off(event: string, callback: EventCallback): this {
    this.events.get(event)?.delete(callback);
    return this;
  }

  emit(event: string, ...args: unknown[]): boolean {
    const callbacks = this.events.get(event);
    if (!callbacks || callbacks.size === 0) return false;
    callbacks.forEach((cb) => cb(...args));
    return true;
  }

  removeAllListeners(event?: string): this {
    if (event) {
      this.events.delete(event);
    } else {
      this.events.clear();
    }
    return this;
  }
}

// ============================================================================
// Timeline Navigator Configuration
// ============================================================================

/**
 * Configuration for timeline navigator
 */
export interface TimelineNavigatorConfig {
  /** Initial time range */
  initialRange: TimeRange;
  /** Default playback speed */
  defaultPlaybackSpeed: number;
  /** Maximum snapshots to keep in memory */
  maxCacheSize: number;
  /** Resolution for navigation */
  resolution: TemporalResolution;
  /** Enable interpolation between states */
  enableInterpolation: boolean;
  /** Interpolation mode */
  interpolationMode: InterpolationMode;
  /** Auto-keyframe interval (ms) */
  keyframeInterval: number;
  /** Enable scrubbing optimization */
  enableScrubOptimization: boolean;
}

/**
 * Default configuration
 */
export const DEFAULT_TIMELINE_CONFIG: TimelineNavigatorConfig = {
  initialRange: { start: 0, end: 60000 }, // 1 minute
  defaultPlaybackSpeed: 1.0,
  maxCacheSize: 1000,
  resolution: "frame",
  enableInterpolation: true,
  interpolationMode: "linear",
  keyframeInterval: 1000,
  enableScrubOptimization: true,
};

// ============================================================================
// Timeline Events
// ============================================================================

export interface TimelineEvents {
  timeChange: (time: Timestamp) => void;
  rangeChange: (range: TimeRange) => void;
  playbackStateChange: (state: PlaybackState) => void;
  snapshotLoaded: (snapshot: StateSnapshot) => void;
  keyframeReached: (keyframe: KeyframeMarker) => void;
  markerAdded: (marker: TimelineMarker) => void;
  scrubStart: () => void;
  scrubEnd: () => void;
}

// ============================================================================
// Playback State
// ============================================================================

export type PlaybackState = "playing" | "paused" | "scrubbing" | "seeking";

export interface PlaybackStatus {
  state: PlaybackState;
  currentTime: Timestamp;
  speed: number;
  direction: PlaybackDirection;
  loop: boolean;
  loopRange?: TimeRange;
}

// ============================================================================
// Timeline Navigator Class
// ============================================================================

/**
 * Timeline Navigator for 4D temporal navigation
 * Manages time-based navigation through agent evolution history
 */
export class TimelineNavigator extends BrowserEventEmitter {
  private config: TimelineNavigatorConfig;
  private currentTime: Timestamp;
  private timeRange: TimeRange;
  private playbackStatus: PlaybackStatus;
  private markers: Map<string, TimelineMarker>;
  private keyframeIndex: KeyframeIndex;
  private snapshotCache: Map<Timestamp, StateSnapshot>;
  private animationFrameId: number | null = null;
  private lastFrameTime: number = 0;
  private isScrubbing: boolean = false;

  constructor(config: Partial<TimelineNavigatorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_TIMELINE_CONFIG, ...config };
    this.currentTime = this.config.initialRange.start;
    this.timeRange = { ...this.config.initialRange };
    this.markers = new Map();
    this.snapshotCache = new Map();

    this.keyframeIndex = {
      keyframes: [],
      startTime: this.timeRange.start,
      endTime: this.timeRange.end,
      totalKeyframes: 0,
      averageInterval: 0,
    };

    this.playbackStatus = {
      state: "paused",
      currentTime: this.currentTime,
      speed: this.config.defaultPlaybackSpeed,
      direction: "forward",
      loop: false,
    };
  }

  // ============================================================================
  // Time Navigation
  // ============================================================================

  /**
   * Seek to a specific time
   */
  seek(time: Timestamp): void {
    const clampedTime = this.clampTime(time);
    const previousState = this.playbackStatus.state;

    this.playbackStatus.state = "seeking";
    this.emit("playbackStateChange", this.playbackStatus);

    this.currentTime = clampedTime;
    this.playbackStatus.currentTime = clampedTime;

    this.emit("timeChange", clampedTime);

    // Check for keyframe
    const nearestKeyframe = this.findNearestKeyframe(clampedTime);
    if (
      nearestKeyframe &&
      Math.abs(nearestKeyframe.time - clampedTime) < this.getResolutionMs()
    ) {
      this.emit("keyframeReached", nearestKeyframe);
    }

    // Restore previous state (unless was scrubbing)
    if (previousState !== "scrubbing") {
      this.playbackStatus.state =
        previousState === "seeking" ? "paused" : previousState;
      this.emit("playbackStateChange", this.playbackStatus);
    }
  }

  /**
   * Jump to a specific keyframe
   */
  seekToKeyframe(keyframeId: string): void {
    const marker = this.markers.get(keyframeId);
    if (marker && marker.type === "keyframe") {
      this.seek(marker.time);
    }
  }

  /**
   * Jump to next keyframe
   */
  nextKeyframe(): void {
    const nextKf = this.findNextKeyframe(this.currentTime);
    if (nextKf) {
      this.seek(nextKf.time);
    }
  }

  /**
   * Jump to previous keyframe
   */
  previousKeyframe(): void {
    const prevKf = this.findPreviousKeyframe(this.currentTime);
    if (prevKf) {
      this.seek(prevKf.time);
    }
  }

  /**
   * Step forward by one frame/resolution unit
   */
  stepForward(): void {
    this.seek(this.currentTime + this.getResolutionMs());
  }

  /**
   * Step backward by one frame/resolution unit
   */
  stepBackward(): void {
    this.seek(this.currentTime - this.getResolutionMs());
  }

  // ============================================================================
  // Playback Control
  // ============================================================================

  /**
   * Start playback
   */
  play(): void {
    if (this.playbackStatus.state === "playing") return;

    this.playbackStatus.state = "playing";
    this.emit("playbackStateChange", this.playbackStatus);

    this.lastFrameTime = performance.now();
    this.startAnimationLoop();
  }

  /**
   * Pause playback
   */
  pause(): void {
    if (this.playbackStatus.state === "paused") return;

    this.stopAnimationLoop();
    this.playbackStatus.state = "paused";
    this.emit("playbackStateChange", this.playbackStatus);
  }

  /**
   * Toggle play/pause
   */
  togglePlayback(): void {
    if (this.playbackStatus.state === "playing") {
      this.pause();
    } else {
      this.play();
    }
  }

  /**
   * Set playback speed
   */
  setSpeed(speed: number): void {
    this.playbackStatus.speed = Math.max(0.1, Math.min(16, speed));
    this.emit("playbackStateChange", this.playbackStatus);
  }

  /**
   * Set playback direction
   */
  setDirection(direction: PlaybackDirection): void {
    this.playbackStatus.direction = direction;
    this.emit("playbackStateChange", this.playbackStatus);
  }

  /**
   * Toggle loop mode
   */
  toggleLoop(range?: TimeRange): void {
    this.playbackStatus.loop = !this.playbackStatus.loop;
    this.playbackStatus.loopRange = range;
    this.emit("playbackStateChange", this.playbackStatus);
  }

  // ============================================================================
  // Scrubbing
  // ============================================================================

  /**
   * Start scrubbing mode
   */
  startScrub(): void {
    this.isScrubbing = true;
    this.stopAnimationLoop();
    this.playbackStatus.state = "scrubbing";
    this.emit("scrubStart");
    this.emit("playbackStateChange", this.playbackStatus);
  }

  /**
   * Update scrub position
   */
  scrubTo(time: Timestamp): void {
    if (!this.isScrubbing) return;

    const clampedTime = this.clampTime(time);
    this.currentTime = clampedTime;
    this.playbackStatus.currentTime = clampedTime;

    // Emit at reduced rate for performance
    if (this.config.enableScrubOptimization) {
      this.throttledEmitTimeChange(clampedTime);
    } else {
      this.emit("timeChange", clampedTime);
    }
  }

  /**
   * End scrubbing mode
   */
  endScrub(): void {
    this.isScrubbing = false;
    this.playbackStatus.state = "paused";
    this.emit("scrubEnd");
    this.emit("playbackStateChange", this.playbackStatus);
  }

  // ============================================================================
  // Time Range Management
  // ============================================================================

  /**
   * Set visible time range
   */
  setTimeRange(range: TimeRange): void {
    this.timeRange = { ...range };
    this.keyframeIndex.startTime = range.start;
    this.keyframeIndex.endTime = range.end;
    this.emit("rangeChange", this.timeRange);
  }

  /**
   * Zoom in on timeline
   */
  zoomIn(factor: number = 0.5, centerTime?: Timestamp): void {
    const center = centerTime ?? this.currentTime;
    const duration = this.timeRange.end - this.timeRange.start;
    const newDuration = duration * factor;

    const newStart = center - newDuration / 2;
    const newEnd = center + newDuration / 2;

    this.setTimeRange({
      start: Math.max(0, newStart),
      end: newEnd,
    });
  }

  /**
   * Zoom out on timeline
   */
  zoomOut(factor: number = 2, centerTime?: Timestamp): void {
    this.zoomIn(1 / factor, centerTime);
  }

  /**
   * Pan timeline by offset
   */
  pan(offset: number): void {
    this.setTimeRange({
      start: this.timeRange.start + offset,
      end: this.timeRange.end + offset,
    });
  }

  /**
   * Fit visible range to all content
   */
  fitToContent(): void {
    const allTimes = Array.from(this.markers.values()).map((m) => m.time);
    if (allTimes.length === 0) return;

    const minTime = Math.min(...allTimes);
    const maxTime = Math.max(...allTimes);
    const padding = (maxTime - minTime) * 0.1;

    this.setTimeRange({
      start: minTime - padding,
      end: maxTime + padding,
    });
  }

  // ============================================================================
  // Marker Management
  // ============================================================================

  /**
   * Add a timeline marker
   */
  addMarker(marker: TimelineMarker): void {
    this.markers.set(marker.id, marker);
    this.emit("markerAdded", marker);

    if (marker.type === "keyframe") {
      this.updateKeyframeIndex();
    }
  }

  /**
   * Remove a marker
   */
  removeMarker(markerId: string): void {
    const marker = this.markers.get(markerId);
    if (marker) {
      this.markers.delete(markerId);
      if (marker.type === "keyframe") {
        this.updateKeyframeIndex();
      }
    }
  }

  /**
   * Get all markers in range
   */
  getMarkersInRange(range: TimeRange): TimelineMarker[] {
    return Array.from(this.markers.values()).filter(
      (m) => m.time >= range.start && m.time <= range.end
    );
  }

  /**
   * Get all keyframes
   */
  getKeyframes(): KeyframeMarker[] {
    return Array.from(this.markers.values()).filter(
      (m): m is KeyframeMarker => m.type === "keyframe"
    );
  }

  // ============================================================================
  // Snapshot Cache
  // ============================================================================

  /**
   * Cache a snapshot
   */
  cacheSnapshot(snapshot: StateSnapshot): void {
    // Evict oldest if at capacity
    if (this.snapshotCache.size >= this.config.maxCacheSize) {
      const oldestKey = this.snapshotCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.snapshotCache.delete(oldestKey);
      }
    }

    this.snapshotCache.set(snapshot.timestamp, snapshot);
    this.emit("snapshotLoaded", snapshot);
  }

  /**
   * Get cached snapshot at time
   */
  getCachedSnapshot(time: Timestamp): StateSnapshot | undefined {
    return this.snapshotCache.get(time);
  }

  /**
   * Get nearest cached snapshot
   */
  getNearestCachedSnapshot(time: Timestamp): StateSnapshot | undefined {
    let nearest: StateSnapshot | undefined;
    let minDiff = Infinity;

    for (const [t, snapshot] of this.snapshotCache) {
      const diff = Math.abs(t - time);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = snapshot;
      }
    }

    return nearest;
  }

  // ============================================================================
  // Getters
  // ============================================================================

  getCurrentTime(): Timestamp {
    return this.currentTime;
  }

  getTimeRange(): TimeRange {
    return { ...this.timeRange };
  }

  getPlaybackStatus(): PlaybackStatus {
    return { ...this.playbackStatus };
  }

  getKeyframeIndex(): KeyframeIndex {
    return { ...this.keyframeIndex };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private startAnimationLoop(): void {
    const animate = (currentFrameTime: number) => {
      if (this.playbackStatus.state !== "playing") return;

      const deltaTime = currentFrameTime - this.lastFrameTime;
      this.lastFrameTime = currentFrameTime;

      // Calculate time advancement
      const timeAdvance =
        deltaTime *
        this.playbackStatus.speed *
        (this.playbackStatus.direction === "backward" ? -1 : 1);

      let newTime = this.currentTime + timeAdvance;

      // Handle loop
      if (this.playbackStatus.loop && this.playbackStatus.loopRange) {
        const { start, end } = this.playbackStatus.loopRange;
        if (newTime > end) newTime = start;
        if (newTime < start) newTime = end;
      } else {
        // Handle end of timeline
        if (newTime >= this.timeRange.end) {
          newTime = this.timeRange.end;
          this.pause();
        }
        if (newTime <= this.timeRange.start) {
          newTime = this.timeRange.start;
          this.pause();
        }
      }

      this.currentTime = newTime;
      this.playbackStatus.currentTime = newTime;
      this.emit("timeChange", newTime);

      // Check for keyframes
      const nearestKeyframe = this.findNearestKeyframe(newTime);
      if (
        nearestKeyframe &&
        Math.abs(nearestKeyframe.time - newTime) < deltaTime
      ) {
        this.emit("keyframeReached", nearestKeyframe);
      }

      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  private stopAnimationLoop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private clampTime(time: Timestamp): Timestamp {
    return Math.max(this.timeRange.start, Math.min(this.timeRange.end, time));
  }

  private getResolutionMs(): number {
    switch (this.config.resolution) {
      case "millisecond":
        return 1;
      case "frame":
        return 16.67;
      case "centisecond":
        return 10;
      case "decisecond":
        return 100;
      case "second":
        return 1000;
      default:
        return 16.67;
    }
  }

  private findNearestKeyframe(time: Timestamp): KeyframeMarker | undefined {
    const keyframes = this.getKeyframes();
    if (keyframes.length === 0) return undefined;

    return keyframes.reduce(
      (nearest, kf) => {
        if (!nearest) return kf;
        return Math.abs(kf.time - time) < Math.abs(nearest.time - time)
          ? kf
          : nearest;
      },
      undefined as KeyframeMarker | undefined
    );
  }

  private findNextKeyframe(time: Timestamp): KeyframeMarker | undefined {
    return this.getKeyframes()
      .filter((kf) => kf.time > time)
      .sort((a, b) => a.time - b.time)[0];
  }

  private findPreviousKeyframe(time: Timestamp): KeyframeMarker | undefined {
    return this.getKeyframes()
      .filter((kf) => kf.time < time)
      .sort((a, b) => b.time - a.time)[0];
  }

  private updateKeyframeIndex(): void {
    const keyframes = this.getKeyframes().sort((a, b) => a.time - b.time);

    this.keyframeIndex = {
      keyframes: keyframes.map((kf) => ({
        timestamp: kf.time,
        snapshotId: kf.snapshotId,
        type: kf.isAutomatic ? "auto" : "manual",
        label: kf.label,
      })),
      startTime: this.timeRange.start,
      endTime: this.timeRange.end,
      totalKeyframes: keyframes.length,
      averageInterval:
        keyframes.length > 1
          ? ((keyframes[keyframes.length - 1]?.time ?? 0) -
              (keyframes[0]?.time ?? 0)) /
            (keyframes.length - 1)
          : 0,
    };
  }

  private lastThrottledEmit: number = 0;
  private throttledEmitTimeChange(time: Timestamp): void {
    const now = performance.now();
    if (now - this.lastThrottledEmit > 16.67) {
      // ~60fps
      this.emit("timeChange", time);
      this.lastThrottledEmit = now;
    }
  }

  /**
   * Dispose the navigator
   */
  dispose(): void {
    this.stopAnimationLoop();
    this.removeAllListeners();
    this.markers.clear();
    this.snapshotCache.clear();
  }
}

// ============================================================================
// Export Factory
// ============================================================================

export function createTimelineNavigator(
  config?: Partial<TimelineNavigatorConfig>
): TimelineNavigator {
  return new TimelineNavigator(config);
}
