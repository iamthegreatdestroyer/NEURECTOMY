/**
 * Temporal Twin Replay Theater
 *
 * Breakthrough Innovation: Play back digital twin evolution in 3D timeline.
 * Integrates TimelineNavigator with TwinManager for visual state playback.
 *
 * This innovation combines:
 * - Timeline's temporal navigation and playback
 * - Twin's state management and history
 * - Forge's 3D visualization capabilities
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations/replay-theater
 * @agents @NEXUS @STREAM @CANVAS
 * @innovation Temporal Twin Replay Theater (Forge × Twin)
 */

import type {
  UnifiedEntity,
  UnifiedTimeline,
  UnifiedTemporalPoint,
  UnifiedEvent,
  UniversalId,
  Timestamp,
} from "../types";

import { CrossDomainEventBridge } from "../event-bridge";
import { CrossDomainOrchestrator } from "../orchestrator";

// ============================================================================
// Replay Theater Types
// ============================================================================

/**
 * Twin snapshot with visualization data
 */
export interface TwinVisualSnapshot {
  twinId: string;
  timestamp: Timestamp;
  state: unknown;
  visualization: {
    position: { x: number; y: number; z: number };
    color: { r: number; g: number; b: number; a: number };
    scale: number;
    connections: Array<{
      targetId: string;
      strength: number;
      type: string;
    }>;
    annotations: Array<{
      text: string;
      position: { x: number; y: number; z: number };
    }>;
  };
  metrics: {
    divergence: number;
    activity: number;
    health: number;
  };
}

/**
 * Replay session configuration
 */
export interface ReplaySessionConfig {
  /** Session identifier */
  sessionId: string;

  /** Twin IDs to include in replay */
  twinIds: string[];

  /** Time range for replay */
  timeRange: {
    start: Timestamp;
    end: Timestamp;
  };

  /** Playback configuration */
  playback: {
    speed: number;
    loop: boolean;
    direction: "forward" | "backward";
  };

  /** Visualization settings */
  visualization: {
    showConnections: boolean;
    showAnnotations: boolean;
    showMetrics: boolean;
    colorByMetric: "divergence" | "activity" | "health" | "none";
    interpolationMode: "linear" | "smooth" | "step";
  };

  /** Filtering options */
  filters: {
    minDivergence?: number;
    maxDivergence?: number;
    tags?: string[];
  };
}

/**
 * Replay state
 */
export interface ReplayState {
  sessionId: string;
  status: "idle" | "playing" | "paused" | "scrubbing" | "ended";
  currentTime: Timestamp;
  progress: number; // 0-1
  snapshots: Map<string, TwinVisualSnapshot>;
  keyframes: ReplayKeyframe[];
}

/**
 * Keyframe in replay
 */
export interface ReplayKeyframe {
  time: Timestamp;
  label: string;
  type: "auto" | "manual" | "event" | "anomaly";
  importance: number;
  twinStates: Map<string, unknown>;
}

/**
 * Theater event types
 */
export type TheaterEventType =
  | "theater:session:created"
  | "theater:session:started"
  | "theater:session:paused"
  | "theater:session:stopped"
  | "theater:session:ended"
  | "theater:time:changed"
  | "theater:keyframe:reached"
  | "theater:snapshot:generated"
  | "theater:interaction:occurred";

export interface TheaterEvent {
  type: TheaterEventType;
  sessionId: string;
  timestamp: Timestamp;
  data: unknown;
}

// ============================================================================
// Replay Theater Class
// ============================================================================

/**
 * Temporal Twin Replay Theater
 *
 * Enables visual playback of digital twin evolution through time.
 */
export class TemporalTwinReplayTheater {
  private static instance: TemporalTwinReplayTheater | null = null;

  private sessions: Map<string, ReplayState> = new Map();
  private configs: Map<string, ReplaySessionConfig> = new Map();
  private animationFrames: Map<string, number> = new Map();
  private eventBridge: CrossDomainEventBridge;
  private orchestrator: CrossDomainOrchestrator;

  // History storage per twin
  private twinHistory: Map<string, TwinVisualSnapshot[]> = new Map();

  // Listeners
  private listeners: Map<string, Set<(event: TheaterEvent) => void>> =
    new Map();

  private constructor() {
    this.eventBridge = CrossDomainEventBridge.getInstance();
    this.orchestrator = CrossDomainOrchestrator.getInstance();

    this.setupEventListeners();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): TemporalTwinReplayTheater {
    if (!TemporalTwinReplayTheater.instance) {
      TemporalTwinReplayTheater.instance = new TemporalTwinReplayTheater();
    }
    return TemporalTwinReplayTheater.instance;
  }

  /**
   * Reset instance (for testing)
   */
  static resetInstance(): void {
    if (TemporalTwinReplayTheater.instance) {
      TemporalTwinReplayTheater.instance.dispose();
      TemporalTwinReplayTheater.instance = null;
    }
  }

  /**
   * Dispose theater
   */
  dispose(): void {
    // Stop all sessions
    for (const sessionId of this.sessions.keys()) {
      this.stopSession(sessionId);
    }

    this.sessions.clear();
    this.configs.clear();
    this.twinHistory.clear();
    this.listeners.clear();
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  private setupEventListeners(): void {
    // Listen for twin state changes
    this.eventBridge.subscribe<{ twinId: string; state: unknown }>(
      "state:changed",
      (event) => this.handleTwinStateChange(event)
    );

    // Listen for predictions
    this.eventBridge.subscribe<{ twinId: string; predictions: unknown[] }>(
      "prediction:completed",
      (event) => this.handlePredictionCompleted(event)
    );
  }

  private handleTwinStateChange(
    event: UnifiedEvent<{ twinId: string; state: unknown }>
  ): void {
    const { twinId, state } = event.payload ?? {};
    if (!twinId || !state) return;

    // Record snapshot in history
    const snapshot = this.createVisualSnapshot(twinId, state);

    if (!this.twinHistory.has(twinId)) {
      this.twinHistory.set(twinId, []);
    }
    this.twinHistory.get(twinId)!.push(snapshot);

    // Update active sessions
    for (const [sessionId, config] of this.configs) {
      if (config.twinIds.includes(twinId)) {
        const replayState = this.sessions.get(sessionId);
        if (replayState && replayState.status === "playing") {
          replayState.snapshots.set(twinId, snapshot);
        }
      }
    }
  }

  private handlePredictionCompleted(
    event: UnifiedEvent<{ twinId: string; predictions: unknown[] }>
  ): void {
    const { twinId, predictions } = event.payload ?? {};
    if (!twinId || !predictions) return;

    // Create keyframe for significant predictions
    for (const [sessionId, config] of this.configs) {
      if (config.twinIds.includes(twinId)) {
        const replayState = this.sessions.get(sessionId);
        if (replayState) {
          const keyframe: ReplayKeyframe = {
            time: event.timestamp,
            label: `Prediction for ${twinId}`,
            type: "event",
            importance: 0.8,
            twinStates: new Map([[twinId, predictions[0]]]),
          };
          replayState.keyframes.push(keyframe);
        }
      }
    }
  }

  // ==========================================================================
  // Session Management
  // ==========================================================================

  /**
   * Create a new replay session
   */
  createSession(config: ReplaySessionConfig): ReplayState {
    const { sessionId, twinIds, timeRange } = config;

    if (this.sessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} already exists`);
    }

    // Initialize snapshots from history
    const snapshots = new Map<string, TwinVisualSnapshot>();
    for (const twinId of twinIds) {
      const history = this.twinHistory.get(twinId);
      if (history && history.length > 0) {
        // Find snapshot closest to start time
        const snapshot = this.findSnapshotAtTime(history, timeRange.start);
        if (snapshot) {
          snapshots.set(twinId, snapshot);
        }
      }
    }

    // Generate keyframes from history
    const keyframes = this.generateKeyframes(twinIds, timeRange);

    const state: ReplayState = {
      sessionId,
      status: "idle",
      currentTime: timeRange.start,
      progress: 0,
      snapshots,
      keyframes,
    };

    this.sessions.set(sessionId, state);
    this.configs.set(sessionId, config);

    this.emitEvent({
      type: "theater:session:created",
      sessionId,
      timestamp: Date.now(),
      data: { config },
    });

    return state;
  }

  /**
   * Start playback of a session
   */
  startSession(sessionId: string): void {
    const state = this.sessions.get(sessionId);
    const config = this.configs.get(sessionId);

    if (!state || !config) {
      throw new Error(`Session ${sessionId} not found`);
    }

    if (state.status === "playing") return;

    state.status = "playing";

    // Start animation loop
    this.startAnimationLoop(sessionId);

    this.emitEvent({
      type: "theater:session:started",
      sessionId,
      timestamp: Date.now(),
      data: { currentTime: state.currentTime },
    });
  }

  /**
   * Pause playback
   */
  pauseSession(sessionId: string): void {
    const state = this.sessions.get(sessionId);
    if (!state) return;

    state.status = "paused";
    this.stopAnimationLoop(sessionId);

    this.emitEvent({
      type: "theater:session:paused",
      sessionId,
      timestamp: Date.now(),
      data: { currentTime: state.currentTime },
    });
  }

  /**
   * Stop and reset session
   */
  stopSession(sessionId: string): void {
    const state = this.sessions.get(sessionId);
    const config = this.configs.get(sessionId);

    if (!state || !config) return;

    state.status = "idle";
    state.currentTime = config.timeRange.start;
    state.progress = 0;

    this.stopAnimationLoop(sessionId);

    this.emitEvent({
      type: "theater:session:stopped",
      sessionId,
      timestamp: Date.now(),
      data: {},
    });
  }

  /**
   * Seek to specific time
   */
  seekTo(sessionId: string, time: Timestamp): void {
    const state = this.sessions.get(sessionId);
    const config = this.configs.get(sessionId);

    if (!state || !config) return;

    // Clamp time to range
    const clampedTime = Math.max(
      config.timeRange.start,
      Math.min(config.timeRange.end, time)
    );

    state.currentTime = clampedTime;
    state.progress =
      (clampedTime - config.timeRange.start) /
      (config.timeRange.end - config.timeRange.start);

    // Update snapshots
    this.updateSnapshotsForTime(sessionId, clampedTime);

    this.emitEvent({
      type: "theater:time:changed",
      sessionId,
      timestamp: Date.now(),
      data: { currentTime: clampedTime, progress: state.progress },
    });
  }

  /**
   * Seek to keyframe by index
   */
  seekToKeyframe(sessionId: string, keyframeIndex: number): void {
    const state = this.sessions.get(sessionId);
    if (
      !state ||
      keyframeIndex < 0 ||
      keyframeIndex >= state.keyframes.length
    ) {
      return;
    }

    const keyframe = state.keyframes[keyframeIndex];
    this.seekTo(sessionId, keyframe.time);

    this.emitEvent({
      type: "theater:keyframe:reached",
      sessionId,
      timestamp: Date.now(),
      data: { keyframe, index: keyframeIndex },
    });
  }

  /**
   * Get current session state
   */
  getSessionState(sessionId: string): ReplayState | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Get session configuration
   */
  getSessionConfig(sessionId: string): ReplaySessionConfig | undefined {
    return this.configs.get(sessionId);
  }

  // ==========================================================================
  // Animation Loop
  // ==========================================================================

  private startAnimationLoop(sessionId: string): void {
    this.stopAnimationLoop(sessionId); // Clear any existing

    const config = this.configs.get(sessionId)!;
    const state = this.sessions.get(sessionId)!;

    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const delta = currentTime - lastTime;
      lastTime = currentTime;

      if (state.status !== "playing") return;

      // Calculate time step
      const timeStep = delta * config.playback.speed;
      const direction = config.playback.direction === "forward" ? 1 : -1;

      let newTime = state.currentTime + timeStep * direction;

      // Handle boundaries
      if (newTime >= config.timeRange.end) {
        if (config.playback.loop) {
          newTime = config.timeRange.start;
        } else {
          newTime = config.timeRange.end;
          state.status = "ended";
          this.emitEvent({
            type: "theater:session:ended",
            sessionId,
            timestamp: Date.now(),
            data: {},
          });
          return;
        }
      } else if (newTime <= config.timeRange.start) {
        if (config.playback.loop) {
          newTime = config.timeRange.end;
        } else {
          newTime = config.timeRange.start;
          state.status = "ended";
          this.emitEvent({
            type: "theater:session:ended",
            sessionId,
            timestamp: Date.now(),
            data: {},
          });
          return;
        }
      }

      // Update state
      state.currentTime = newTime;
      state.progress =
        (newTime - config.timeRange.start) /
        (config.timeRange.end - config.timeRange.start);

      // Update snapshots
      this.updateSnapshotsForTime(sessionId, newTime);

      // Check for keyframe crossings
      this.checkKeyframeCrossings(
        sessionId,
        state.currentTime - timeStep * direction,
        newTime
      );

      // Continue animation
      const frameId = requestAnimationFrame(animate);
      this.animationFrames.set(sessionId, frameId);
    };

    const frameId = requestAnimationFrame(animate);
    this.animationFrames.set(sessionId, frameId);
  }

  private stopAnimationLoop(sessionId: string): void {
    const frameId = this.animationFrames.get(sessionId);
    if (frameId !== undefined) {
      cancelAnimationFrame(frameId);
      this.animationFrames.delete(sessionId);
    }
  }

  // ==========================================================================
  // Snapshot Management
  // ==========================================================================

  private createVisualSnapshot(
    twinId: string,
    state: unknown
  ): TwinVisualSnapshot {
    // Extract metrics from state
    const metrics = this.extractMetrics(state);

    // Generate visualization properties
    const visualization = this.generateVisualization(twinId, state, metrics);

    return {
      twinId,
      timestamp: Date.now(),
      state,
      visualization,
      metrics,
    };
  }

  private extractMetrics(state: unknown): TwinVisualSnapshot["metrics"] {
    // Default metrics if state doesn't have them
    const stateObj = state as Record<string, unknown> | null;

    return {
      divergence: (stateObj?.divergenceScore as number) ?? 0,
      activity: (stateObj?.activityLevel as number) ?? 0.5,
      health: (stateObj?.healthScore as number) ?? 1.0,
    };
  }

  private generateVisualization(
    twinId: string,
    state: unknown,
    metrics: TwinVisualSnapshot["metrics"]
  ): TwinVisualSnapshot["visualization"] {
    // Generate position based on twin ID hash
    const hash = this.hashString(twinId);
    const angle = (hash % 360) * (Math.PI / 180);
    const radius = 5 + (hash % 10);

    return {
      position: {
        x: Math.cos(angle) * radius,
        y: metrics.activity * 2,
        z: Math.sin(angle) * radius,
      },
      color: this.metricToColor(metrics.health),
      scale: 0.5 + metrics.activity * 0.5,
      connections: [],
      annotations: [],
    };
  }

  private metricToColor(value: number): {
    r: number;
    g: number;
    b: number;
    a: number;
  } {
    // Health gradient: red (0) -> yellow (0.5) -> green (1)
    if (value < 0.5) {
      return {
        r: 1,
        g: value * 2,
        b: 0,
        a: 1,
      };
    } else {
      return {
        r: 1 - (value - 0.5) * 2,
        g: 1,
        b: 0,
        a: 1,
      };
    }
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  private findSnapshotAtTime(
    history: TwinVisualSnapshot[],
    time: Timestamp
  ): TwinVisualSnapshot | null {
    if (history.length === 0) return null;

    // Binary search for nearest snapshot
    let left = 0;
    let right = history.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (history[mid].timestamp < time) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    // Return closest
    if (left > 0) {
      const prev = history[left - 1];
      const curr = history[left];
      if (Math.abs(prev.timestamp - time) < Math.abs(curr.timestamp - time)) {
        return prev;
      }
    }

    return history[left];
  }

  private updateSnapshotsForTime(sessionId: string, time: Timestamp): void {
    const state = this.sessions.get(sessionId);
    const config = this.configs.get(sessionId);

    if (!state || !config) return;

    for (const twinId of config.twinIds) {
      const history = this.twinHistory.get(twinId);
      if (history) {
        const snapshot = this.findSnapshotAtTime(history, time);
        if (snapshot) {
          // Interpolate if needed
          const interpolated =
            config.visualization.interpolationMode !== "step"
              ? this.interpolateSnapshot(
                  history,
                  time,
                  config.visualization.interpolationMode
                )
              : snapshot;

          state.snapshots.set(twinId, interpolated ?? snapshot);
        }
      }
    }

    this.emitEvent({
      type: "theater:snapshot:generated",
      sessionId,
      timestamp: Date.now(),
      data: { time, snapshots: Array.from(state.snapshots.entries()) },
    });
  }

  private interpolateSnapshot(
    history: TwinVisualSnapshot[],
    time: Timestamp,
    mode: "linear" | "smooth"
  ): TwinVisualSnapshot | null {
    if (history.length < 2) return null;

    // Find surrounding snapshots
    let beforeIndex = -1;
    for (let i = 0; i < history.length; i++) {
      if (history[i].timestamp <= time) {
        beforeIndex = i;
      } else {
        break;
      }
    }

    if (beforeIndex === -1 || beforeIndex >= history.length - 1) {
      return null;
    }

    const before = history[beforeIndex];
    const after = history[beforeIndex + 1];

    // Calculate interpolation factor
    const t = (time - before.timestamp) / (after.timestamp - before.timestamp);
    const factor = mode === "smooth" ? this.smoothstep(t) : t;

    // Interpolate visualization
    return {
      ...before,
      timestamp: time,
      visualization: {
        position: {
          x: this.lerp(
            before.visualization.position.x,
            after.visualization.position.x,
            factor
          ),
          y: this.lerp(
            before.visualization.position.y,
            after.visualization.position.y,
            factor
          ),
          z: this.lerp(
            before.visualization.position.z,
            after.visualization.position.z,
            factor
          ),
        },
        color: {
          r: this.lerp(
            before.visualization.color.r,
            after.visualization.color.r,
            factor
          ),
          g: this.lerp(
            before.visualization.color.g,
            after.visualization.color.g,
            factor
          ),
          b: this.lerp(
            before.visualization.color.b,
            after.visualization.color.b,
            factor
          ),
          a: this.lerp(
            before.visualization.color.a,
            after.visualization.color.a,
            factor
          ),
        },
        scale: this.lerp(
          before.visualization.scale,
          after.visualization.scale,
          factor
        ),
        connections: before.visualization.connections,
        annotations: before.visualization.annotations,
      },
      metrics: {
        divergence: this.lerp(
          before.metrics.divergence,
          after.metrics.divergence,
          factor
        ),
        activity: this.lerp(
          before.metrics.activity,
          after.metrics.activity,
          factor
        ),
        health: this.lerp(before.metrics.health, after.metrics.health, factor),
      },
    };
  }

  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  private smoothstep(t: number): number {
    return t * t * (3 - 2 * t);
  }

  // ==========================================================================
  // Keyframe Management
  // ==========================================================================

  private generateKeyframes(
    twinIds: string[],
    timeRange: { start: Timestamp; end: Timestamp }
  ): ReplayKeyframe[] {
    const keyframes: ReplayKeyframe[] = [];
    const allSnapshots: TwinVisualSnapshot[] = [];

    // Collect all snapshots in range
    for (const twinId of twinIds) {
      const history = this.twinHistory.get(twinId);
      if (history) {
        for (const snapshot of history) {
          if (
            snapshot.timestamp >= timeRange.start &&
            snapshot.timestamp <= timeRange.end
          ) {
            allSnapshots.push(snapshot);
          }
        }
      }
    }

    // Sort by timestamp
    allSnapshots.sort((a, b) => a.timestamp - b.timestamp);

    // Generate keyframes at significant points
    let lastTime = timeRange.start;
    const minInterval = (timeRange.end - timeRange.start) / 100; // Max 100 keyframes

    for (const snapshot of allSnapshots) {
      if (snapshot.timestamp - lastTime >= minInterval) {
        // Check for significant changes
        const importance = this.calculateImportance(snapshot);

        if (importance > 0.3) {
          keyframes.push({
            time: snapshot.timestamp,
            label: `State change`,
            type: "auto",
            importance,
            twinStates: new Map([[snapshot.twinId, snapshot.state]]),
          });
          lastTime = snapshot.timestamp;
        }
      }
    }

    return keyframes;
  }

  private calculateImportance(snapshot: TwinVisualSnapshot): number {
    // Higher importance for higher divergence or low health
    const divergenceImpact = snapshot.metrics.divergence;
    const healthImpact = 1 - snapshot.metrics.health;

    return Math.min(1, divergenceImpact * 0.6 + healthImpact * 0.4);
  }

  private checkKeyframeCrossings(
    sessionId: string,
    previousTime: Timestamp,
    currentTime: Timestamp
  ): void {
    const state = this.sessions.get(sessionId);
    if (!state) return;

    const minTime = Math.min(previousTime, currentTime);
    const maxTime = Math.max(previousTime, currentTime);

    for (let i = 0; i < state.keyframes.length; i++) {
      const keyframe = state.keyframes[i];
      if (keyframe.time > minTime && keyframe.time <= maxTime) {
        this.emitEvent({
          type: "theater:keyframe:reached",
          sessionId,
          timestamp: Date.now(),
          data: { keyframe, index: i },
        });
      }
    }
  }

  /**
   * Add manual keyframe
   */
  addKeyframe(
    sessionId: string,
    label: string,
    time?: Timestamp
  ): ReplayKeyframe | null {
    const state = this.sessions.get(sessionId);
    const config = this.configs.get(sessionId);

    if (!state || !config) return null;

    const keyframeTime = time ?? state.currentTime;

    // Collect current states
    const twinStates = new Map<string, unknown>();
    for (const twinId of config.twinIds) {
      const snapshot = state.snapshots.get(twinId);
      if (snapshot) {
        twinStates.set(twinId, snapshot.state);
      }
    }

    const keyframe: ReplayKeyframe = {
      time: keyframeTime,
      label,
      type: "manual",
      importance: 1.0,
      twinStates,
    };

    // Insert in sorted order
    let insertIndex = state.keyframes.length;
    for (let i = 0; i < state.keyframes.length; i++) {
      if (state.keyframes[i].time > keyframeTime) {
        insertIndex = i;
        break;
      }
    }

    state.keyframes.splice(insertIndex, 0, keyframe);

    return keyframe;
  }

  // ==========================================================================
  // Event Emission
  // ==========================================================================

  private emitEvent(event: TheaterEvent): void {
    const sessionListeners = this.listeners.get(event.sessionId);
    if (sessionListeners) {
      for (const listener of sessionListeners) {
        try {
          listener(event);
        } catch (error) {
          console.error("Theater event listener error:", error);
        }
      }
    }

    // Also emit to orchestrator
    this.eventBridge.publish({
      id: `theater-${event.type}-${Date.now()}`,
      type: "metrics:updated",
      payload: event,
      timestamp: event.timestamp,
      sourceDomain: "twin",
      targetDomains: ["forge"],
    });
  }

  /**
   * Subscribe to theater events
   */
  subscribe(
    sessionId: string,
    listener: (event: TheaterEvent) => void
  ): () => void {
    if (!this.listeners.has(sessionId)) {
      this.listeners.set(sessionId, new Set());
    }
    this.listeners.get(sessionId)!.add(listener);

    return () => {
      this.listeners.get(sessionId)?.delete(listener);
    };
  }

  // ==========================================================================
  // History Management
  // ==========================================================================

  /**
   * Record twin state for history
   */
  recordTwinState(twinId: string, state: unknown): void {
    const snapshot = this.createVisualSnapshot(twinId, state);

    if (!this.twinHistory.has(twinId)) {
      this.twinHistory.set(twinId, []);
    }

    this.twinHistory.get(twinId)!.push(snapshot);
  }

  /**
   * Get twin history
   */
  getTwinHistory(twinId: string): TwinVisualSnapshot[] {
    return this.twinHistory.get(twinId) ?? [];
  }

  /**
   * Clear twin history
   */
  clearTwinHistory(twinId: string): void {
    this.twinHistory.delete(twinId);
  }

  /**
   * Get all tracked twin IDs
   */
  getTrackedTwins(): string[] {
    return Array.from(this.twinHistory.keys());
  }
}

// ============================================================================
// React Hook for Theater
// ============================================================================

/**
 * React hook return type
 */
export interface UseReplayTheaterResult {
  state: ReplayState | null;
  config: ReplaySessionConfig | null;
  snapshots: TwinVisualSnapshot[];
  currentKeyframe: ReplayKeyframe | null;
  play: () => void;
  pause: () => void;
  stop: () => void;
  seekTo: (time: Timestamp) => void;
  seekToKeyframe: (index: number) => void;
  addKeyframe: (label: string) => void;
}

/**
 * Create React hook for replay theater
 *
 * Usage:
 * ```tsx
 * const theater = useReplayTheater(sessionId);
 * // theater.play(), theater.pause(), etc.
 * ```
 */
export function createUseReplayTheater() {
  // This would be imported from React
  // For now, return factory function
  return function useReplayTheater(sessionId: string): UseReplayTheaterResult {
    const theater = TemporalTwinReplayTheater.getInstance();

    const state = theater.getSessionState(sessionId) ?? null;
    const config = theater.getSessionConfig(sessionId) ?? null;

    const snapshots = state ? Array.from(state.snapshots.values()) : [];

    const currentKeyframe =
      state?.keyframes.find(
        (kf) => Math.abs(kf.time - (state?.currentTime ?? 0)) < 100
      ) ?? null;

    return {
      state,
      config,
      snapshots,
      currentKeyframe,
      play: () => theater.startSession(sessionId),
      pause: () => theater.pauseSession(sessionId),
      stop: () => theater.stopSession(sessionId),
      seekTo: (time) => theater.seekTo(sessionId, time),
      seekToKeyframe: (index) => theater.seekToKeyframe(sessionId, index),
      addKeyframe: (label) => theater.addKeyframe(sessionId, label),
    };
  };
}

// ============================================================================
// Exports
// ============================================================================

export const ReplayTheater = TemporalTwinReplayTheater;
export default TemporalTwinReplayTheater;
