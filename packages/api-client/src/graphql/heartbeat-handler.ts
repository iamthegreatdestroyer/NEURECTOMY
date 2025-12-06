/**
 * @fileoverview WebSocket Heartbeat Handler for NEURECTOMY
 *
 * @STREAM + @SENTRY Implementation
 * Production-grade heartbeat/keepalive mechanism for GraphQL WebSocket connections
 * with health monitoring, adaptive intervals, and dead connection detection.
 *
 * Features:
 * - Configurable heartbeat intervals
 * - Adaptive heartbeat based on connection quality
 * - Dead connection detection with configurable timeouts
 * - Latency tracking and reporting
 * - Server acknowledgment validation
 * - Integration with ConnectionStateManager
 *
 * @module @neurectomy/api-client/graphql/heartbeat-handler
 */

import { EventEmitter } from "events";
import type {
  ConnectionStateManager,
  ConnectionQuality,
} from "./connection-state";

// =============================================================================
// TYPES & INTERFACES
// =============================================================================

/**
 * Heartbeat handler configuration.
 */
export interface HeartbeatConfig {
  /** Base interval between heartbeats in milliseconds */
  interval: number;

  /** Timeout waiting for pong response in milliseconds */
  timeout: number;

  /** Number of missed heartbeats before declaring connection dead */
  maxMissedHeartbeats: number;

  /** Enable adaptive heartbeat intervals */
  adaptive: boolean;

  /** Minimum interval when adaptive (ms) */
  minInterval: number;

  /** Maximum interval when adaptive (ms) */
  maxInterval: number;

  /** Enable latency tracking */
  trackLatency: boolean;

  /** Number of latency samples to keep */
  latencySampleSize: number;

  /** Enable debug logging */
  debug: boolean;
}

/**
 * Heartbeat event types.
 */
export type HeartbeatEventType =
  | "PING_SENT"
  | "PONG_RECEIVED"
  | "TIMEOUT"
  | "CONNECTION_DEAD"
  | "INTERVAL_ADJUSTED"
  | "LATENCY_SPIKE";

/**
 * Heartbeat event data.
 */
export interface HeartbeatEvent {
  type: HeartbeatEventType;
  timestamp: Date;
  sequenceNumber: number;
  latency?: number;
  missedCount?: number;
  newInterval?: number;
  data?: Record<string, unknown>;
}

/**
 * Heartbeat statistics.
 */
export interface HeartbeatStats {
  /** Total pings sent */
  pingsSent: number;

  /** Total pongs received */
  pongsReceived: number;

  /** Current missed heartbeat count */
  missedCount: number;

  /** Total timeouts */
  totalTimeouts: number;

  /** Last successful heartbeat timestamp */
  lastSuccessfulHeartbeat: Date | null;

  /** Last ping timestamp */
  lastPingTime: Date | null;

  /** Current heartbeat interval */
  currentInterval: number;

  /** Average latency in ms */
  averageLatency: number;

  /** Min latency in ms */
  minLatency: number;

  /** Max latency in ms */
  maxLatency: number;

  /** Last N latency samples */
  latencySamples: number[];

  /** Current sequence number */
  sequenceNumber: number;

  /** Is heartbeat running */
  isRunning: boolean;
}

/**
 * Ping message format for GraphQL WebSocket protocol.
 */
export interface PingMessage {
  type: "ping";
  payload?: {
    timestamp: number;
    sequence: number;
  };
}

/**
 * Pong message format.
 */
export interface PongMessage {
  type: "pong";
  payload?: {
    timestamp: number;
    sequence: number;
    serverTime?: number;
  };
}

/**
 * Function to send ping message.
 */
export type SendPingFn = (message: PingMessage) => Promise<void> | void;

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

const DEFAULT_CONFIG: HeartbeatConfig = {
  interval: 30000, // 30 seconds
  timeout: 5000, // 5 seconds
  maxMissedHeartbeats: 3,
  adaptive: true,
  minInterval: 10000, // 10 seconds
  maxInterval: 60000, // 60 seconds
  trackLatency: true,
  latencySampleSize: 20,
  debug: false,
};

// =============================================================================
// ADAPTIVE INTERVALS BY QUALITY
// =============================================================================

const QUALITY_INTERVALS: Record<ConnectionQuality, number> = {
  EXCELLENT: 45000, // 45 seconds - stable connection, less frequent
  GOOD: 30000, // 30 seconds - standard interval
  FAIR: 20000, // 20 seconds - more frequent checks
  POOR: 15000, // 15 seconds - frequent monitoring
  CRITICAL: 10000, // 10 seconds - maximum monitoring
};

// =============================================================================
// HEARTBEAT HANDLER
// =============================================================================

/**
 * Manages WebSocket keepalive with adaptive heartbeat intervals.
 *
 * @example
 * ```typescript
 * const heartbeat = new HeartbeatHandler({
 *   interval: 30000,
 *   timeout: 5000,
 *   adaptive: true,
 * });
 *
 * heartbeat.on('connectionDead', () => {
 *   console.log('Connection lost, initiating reconnect');
 *   reconnect();
 * });
 *
 * heartbeat.start(async (msg) => {
 *   wsClient.send(JSON.stringify(msg));
 * });
 *
 * // When receiving pong from server
 * wsClient.on('message', (data) => {
 *   const msg = JSON.parse(data);
 *   if (msg.type === 'pong') {
 *     heartbeat.receivePong(msg);
 *   }
 * });
 * ```
 */
export class HeartbeatHandler extends EventEmitter {
  private config: HeartbeatConfig;
  private sendPing: SendPingFn | null = null;
  private stateManager: ConnectionStateManager | null = null;

  private intervalTimer: NodeJS.Timeout | null = null;
  private timeoutTimer: NodeJS.Timeout | null = null;

  private sequenceNumber = 0;
  private pendingPingSequence: number | null = null;
  private pendingPingTimestamp: number | null = null;

  private stats: HeartbeatStats;
  private latencySamples: number[] = [];

  private _isRunning = false;

  constructor(config: Partial<HeartbeatConfig> = {}) {
    super();

    this.config = { ...DEFAULT_CONFIG, ...config };
    this.stats = this.createInitialStats();
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  /**
   * Start the heartbeat mechanism.
   *
   * @param sendPing - Function to send ping message to server
   * @param stateManager - Optional connection state manager for integration
   */
  start(sendPing: SendPingFn, stateManager?: ConnectionStateManager): void {
    if (this._isRunning) {
      this.log("warn", "Heartbeat already running");
      return;
    }

    this.sendPing = sendPing;
    this.stateManager = stateManager ?? null;
    this._isRunning = true;
    this.stats.isRunning = true;

    // Subscribe to quality changes if state manager provided
    if (this.stateManager && this.config.adaptive) {
      this.stateManager.on(
        "metricsUpdate",
        this.handleQualityChange.bind(this)
      );
    }

    this.log(
      "info",
      `Starting heartbeat with interval: ${this.stats.currentInterval}ms`
    );
    this.scheduleNextPing();
  }

  /**
   * Stop the heartbeat mechanism.
   */
  stop(): void {
    if (!this._isRunning) return;

    this._isRunning = false;
    this.stats.isRunning = false;

    if (this.intervalTimer) {
      clearTimeout(this.intervalTimer);
      this.intervalTimer = null;
    }

    if (this.timeoutTimer) {
      clearTimeout(this.timeoutTimer);
      this.timeoutTimer = null;
    }

    this.pendingPingSequence = null;
    this.pendingPingTimestamp = null;

    if (this.stateManager) {
      this.stateManager.removeListener(
        "metricsUpdate",
        this.handleQualityChange.bind(this)
      );
    }

    this.log("info", "Heartbeat stopped");
  }

  /**
   * Reset heartbeat state (e.g., after reconnection).
   */
  reset(): void {
    this.stop();
    this.stats = this.createInitialStats();
    this.latencySamples = [];
    this.sequenceNumber = 0;
  }

  /**
   * Dispose of the heartbeat handler.
   */
  dispose(): void {
    this.stop();
    this.removeAllListeners();
    this.sendPing = null;
    this.stateManager = null;
  }

  // ===========================================================================
  // PING/PONG HANDLING
  // ===========================================================================

  /**
   * Send a ping message.
   */
  private async sendPingMessage(): Promise<void> {
    if (!this.sendPing || !this._isRunning) return;

    // Clear any existing timeout
    if (this.timeoutTimer) {
      clearTimeout(this.timeoutTimer);
      this.timeoutTimer = null;
    }

    this.sequenceNumber++;
    const timestamp = Date.now();

    const pingMessage: PingMessage = {
      type: "ping",
      payload: {
        timestamp,
        sequence: this.sequenceNumber,
      },
    };

    this.pendingPingSequence = this.sequenceNumber;
    this.pendingPingTimestamp = timestamp;

    try {
      await this.sendPing(pingMessage);

      this.stats.pingsSent++;
      this.stats.lastPingTime = new Date();

      this.emitEvent({
        type: "PING_SENT",
        timestamp: new Date(),
        sequenceNumber: this.sequenceNumber,
      });

      // Start timeout timer
      this.timeoutTimer = setTimeout(() => {
        this.handleTimeout();
      }, this.config.timeout);

      this.log("debug", `Ping sent: seq=${this.sequenceNumber}`);
    } catch (error) {
      this.log("error", "Failed to send ping", { error: String(error) });
      this.handleTimeout();
    }
  }

  /**
   * Handle received pong message.
   *
   * @param pong - Pong message from server
   */
  receivePong(pong: PongMessage): void {
    if (!this._isRunning) return;

    const receiveTime = Date.now();

    // Clear timeout
    if (this.timeoutTimer) {
      clearTimeout(this.timeoutTimer);
      this.timeoutTimer = null;
    }

    // Validate sequence number if present
    const sequence = pong.payload?.sequence;
    if (sequence !== undefined && sequence !== this.pendingPingSequence) {
      this.log(
        "warn",
        `Sequence mismatch: expected ${this.pendingPingSequence}, got ${sequence}`
      );
      // Still process it but log the warning
    }

    // Calculate latency
    let latency: number | undefined;
    if (this.pendingPingTimestamp) {
      latency = receiveTime - this.pendingPingTimestamp;
      this.recordLatency(latency);
    }

    // Reset missed count
    this.stats.missedCount = 0;
    this.stats.pongsReceived++;
    this.stats.lastSuccessfulHeartbeat = new Date();

    // Report to state manager
    if (this.stateManager && latency !== undefined) {
      this.stateManager.recordLatency(latency);
    }

    this.emitEvent({
      type: "PONG_RECEIVED",
      timestamp: new Date(),
      sequenceNumber: sequence ?? this.sequenceNumber,
      latency,
    });

    this.pendingPingSequence = null;
    this.pendingPingTimestamp = null;

    this.log("debug", `Pong received: seq=${sequence}, latency=${latency}ms`);

    // Schedule next ping
    this.scheduleNextPing();
  }

  /**
   * Handle ping timeout.
   */
  private handleTimeout(): void {
    this.stats.missedCount++;
    this.stats.totalTimeouts++;

    this.emitEvent({
      type: "TIMEOUT",
      timestamp: new Date(),
      sequenceNumber: this.pendingPingSequence ?? 0,
      missedCount: this.stats.missedCount,
    });

    this.log(
      "warn",
      `Heartbeat timeout: missed=${this.stats.missedCount}/${this.config.maxMissedHeartbeats}`
    );

    // Report failure to state manager
    if (this.stateManager) {
      this.stateManager.recordFailure(new Error("Heartbeat timeout"));
    }

    if (this.stats.missedCount >= this.config.maxMissedHeartbeats) {
      this.handleConnectionDead();
    } else {
      // Try again with shorter interval
      this.scheduleNextPing(Math.min(this.stats.currentInterval / 2, 5000));
    }
  }

  /**
   * Handle connection declared dead.
   */
  private handleConnectionDead(): void {
    this.emitEvent({
      type: "CONNECTION_DEAD",
      timestamp: new Date(),
      sequenceNumber: this.sequenceNumber,
      missedCount: this.stats.missedCount,
    });

    this.log("error", "Connection declared dead");

    this.stop();
    this.emit("connectionDead", {
      missedCount: this.stats.missedCount,
      lastSuccessfulHeartbeat: this.stats.lastSuccessfulHeartbeat,
    });
  }

  // ===========================================================================
  // ADAPTIVE HEARTBEAT
  // ===========================================================================

  /**
   * Handle connection quality change for adaptive intervals.
   */
  private handleQualityChange(event: { newQuality: ConnectionQuality }): void {
    if (!this.config.adaptive || !this._isRunning) return;

    const newInterval = this.calculateIntervalForQuality(event.newQuality);

    if (newInterval !== this.stats.currentInterval) {
      const oldInterval = this.stats.currentInterval;
      this.stats.currentInterval = newInterval;

      this.emitEvent({
        type: "INTERVAL_ADJUSTED",
        timestamp: new Date(),
        sequenceNumber: this.sequenceNumber,
        newInterval,
        data: { oldInterval, quality: event.newQuality },
      });

      this.log(
        "info",
        `Interval adjusted: ${oldInterval}ms -> ${newInterval}ms (quality: ${event.newQuality})`
      );

      // Reschedule with new interval if waiting
      if (this.intervalTimer && !this.pendingPingSequence) {
        clearTimeout(this.intervalTimer);
        this.scheduleNextPing();
      }
    }
  }

  /**
   * Calculate heartbeat interval based on connection quality.
   */
  private calculateIntervalForQuality(quality: ConnectionQuality): number {
    const baseInterval = QUALITY_INTERVALS[quality];

    return Math.max(
      this.config.minInterval,
      Math.min(this.config.maxInterval, baseInterval)
    );
  }

  /**
   * Manually adjust heartbeat interval.
   */
  setInterval(interval: number): void {
    const newInterval = Math.max(
      this.config.minInterval,
      Math.min(this.config.maxInterval, interval)
    );

    if (newInterval !== this.stats.currentInterval) {
      this.stats.currentInterval = newInterval;

      this.emitEvent({
        type: "INTERVAL_ADJUSTED",
        timestamp: new Date(),
        sequenceNumber: this.sequenceNumber,
        newInterval,
      });

      this.log("info", `Interval manually set: ${newInterval}ms`);
    }
  }

  // ===========================================================================
  // LATENCY TRACKING
  // ===========================================================================

  /**
   * Record a latency sample.
   */
  private recordLatency(latency: number): void {
    if (!this.config.trackLatency) return;

    this.latencySamples.push(latency);

    // Maintain sample window
    if (this.latencySamples.length > this.config.latencySampleSize) {
      this.latencySamples.shift();
    }

    // Update stats
    this.stats.latencySamples = [...this.latencySamples];
    this.stats.averageLatency = this.calculateAverageLatency();
    this.stats.minLatency = Math.min(...this.latencySamples);
    this.stats.maxLatency = Math.max(...this.latencySamples);

    // Check for latency spike
    if (this.latencySamples.length >= 3) {
      const recentAvg =
        this.latencySamples.slice(-3).reduce((a, b) => a + b, 0) / 3;

      if (latency > recentAvg * 2 && latency > 100) {
        this.emitEvent({
          type: "LATENCY_SPIKE",
          timestamp: new Date(),
          sequenceNumber: this.sequenceNumber,
          latency,
          data: { averageLatency: this.stats.averageLatency },
        });

        this.log(
          "warn",
          `Latency spike detected: ${latency}ms (avg: ${this.stats.averageLatency}ms)`
        );
      }
    }
  }

  /**
   * Calculate average latency from samples.
   */
  private calculateAverageLatency(): number {
    if (this.latencySamples.length === 0) return 0;
    const sum = this.latencySamples.reduce((a, b) => a + b, 0);
    return Math.round(sum / this.latencySamples.length);
  }

  // ===========================================================================
  // SCHEDULING
  // ===========================================================================

  /**
   * Schedule the next ping.
   */
  private scheduleNextPing(customInterval?: number): void {
    if (!this._isRunning) return;

    if (this.intervalTimer) {
      clearTimeout(this.intervalTimer);
    }

    const interval = customInterval ?? this.stats.currentInterval;

    this.intervalTimer = setTimeout(() => {
      this.sendPingMessage();
    }, interval);
  }

  /**
   * Force an immediate ping (useful after reconnection).
   */
  async forcePing(): Promise<void> {
    if (!this._isRunning) {
      this.log("warn", "Cannot force ping: heartbeat not running");
      return;
    }

    // Clear scheduled ping
    if (this.intervalTimer) {
      clearTimeout(this.intervalTimer);
      this.intervalTimer = null;
    }

    await this.sendPingMessage();
  }

  // ===========================================================================
  // STATISTICS & DIAGNOSTICS
  // ===========================================================================

  /**
   * Get current heartbeat statistics.
   */
  getStats(): Readonly<HeartbeatStats> {
    return { ...this.stats };
  }

  /**
   * Check if heartbeat is running.
   */
  get isRunning(): boolean {
    return this._isRunning;
  }

  /**
   * Get current interval.
   */
  get currentInterval(): number {
    return this.stats.currentInterval;
  }

  /**
   * Check if there's a pending ping.
   */
  get hasPendingPing(): boolean {
    return this.pendingPingSequence !== null;
  }

  // ===========================================================================
  // UTILITIES
  // ===========================================================================

  private createInitialStats(): HeartbeatStats {
    return {
      pingsSent: 0,
      pongsReceived: 0,
      missedCount: 0,
      totalTimeouts: 0,
      lastSuccessfulHeartbeat: null,
      lastPingTime: null,
      currentInterval: this.config.interval,
      averageLatency: 0,
      minLatency: 0,
      maxLatency: 0,
      latencySamples: [],
      sequenceNumber: 0,
      isRunning: false,
    };
  }

  private emitEvent(event: HeartbeatEvent): void {
    this.emit("heartbeat", event);
    this.emit(event.type.toLowerCase(), event);
  }

  private log(
    level: "debug" | "info" | "warn" | "error",
    message: string,
    data?: Record<string, unknown>
  ): void {
    if (!this.config.debug && (level === "debug" || level === "info")) return;

    const logMessage = `[Heartbeat] ${message}`;
    const logData = data ? ` ${JSON.stringify(data)}` : "";

    switch (level) {
      case "debug":
        console.debug(logMessage + logData);
        break;
      case "info":
        console.info(logMessage + logData);
        break;
      case "warn":
        console.warn(logMessage + logData);
        break;
      case "error":
        console.error(logMessage + logData);
        break;
    }
  }
}

// =============================================================================
// FACTORY FUNCTION
// =============================================================================

/**
 * Create a new heartbeat handler.
 *
 * @param config - Configuration options
 * @returns Configured HeartbeatHandler instance
 *
 * @example
 * ```typescript
 * const heartbeat = createHeartbeatHandler({
 *   interval: 30000,
 *   timeout: 5000,
 *   maxMissedHeartbeats: 3,
 *   adaptive: true,
 * });
 *
 * heartbeat.on('connectionDead', handleConnectionLost);
 * heartbeat.on('latency_spike', handleLatencySpike);
 *
 * heartbeat.start((msg) => wsClient.send(JSON.stringify(msg)));
 * ```
 */
export function createHeartbeatHandler(
  config: Partial<HeartbeatConfig> = {}
): HeartbeatHandler {
  return new HeartbeatHandler(config);
}

// =============================================================================
// INTEGRATED KEEPALIVE
// =============================================================================

/**
 * Options for integrated keepalive.
 */
export interface IntegratedKeepaliveOptions {
  /** Heartbeat configuration */
  heartbeat: Partial<HeartbeatConfig>;

  /** Connection state manager */
  stateManager: ConnectionStateManager;

  /** Function to send messages */
  sendMessage: (message: unknown) => Promise<void> | void;

  /** Optional callback when connection is dead */
  onConnectionDead?: () => void;

  /** Optional callback for latency spikes */
  onLatencySpike?: (latency: number) => void;
}

/**
 * Create an integrated keepalive system with connection state management.
 *
 * @example
 * ```typescript
 * const keepalive = createIntegratedKeepalive({
 *   heartbeat: { interval: 30000 },
 *   stateManager,
 *   sendMessage: (msg) => ws.send(JSON.stringify(msg)),
 *   onConnectionDead: () => reconnect(),
 * });
 *
 * keepalive.start();
 *
 * // Handle incoming pongs
 * ws.on('message', (data) => {
 *   const msg = JSON.parse(data);
 *   if (msg.type === 'pong') {
 *     keepalive.handlePong(msg);
 *   }
 * });
 * ```
 */
export function createIntegratedKeepalive(
  options: IntegratedKeepaliveOptions
): {
  start: () => void;
  stop: () => void;
  handlePong: (pong: PongMessage) => void;
  getStats: () => Readonly<HeartbeatStats>;
  forcePing: () => Promise<void>;
} {
  const heartbeat = new HeartbeatHandler(options.heartbeat);

  // Wire up events
  if (options.onConnectionDead) {
    heartbeat.on("connectionDead", options.onConnectionDead);
  }

  if (options.onLatencySpike) {
    heartbeat.on("latency_spike", (event: HeartbeatEvent) => {
      if (event.latency !== undefined) {
        options.onLatencySpike!(event.latency);
      }
    });
  }

  return {
    start: () => {
      heartbeat.start(
        async (msg) => options.sendMessage(msg),
        options.stateManager
      );
    },
    stop: () => heartbeat.stop(),
    handlePong: (pong) => heartbeat.receivePong(pong),
    getStats: () => heartbeat.getStats(),
    forcePing: () => heartbeat.forcePing(),
  };
}

export default HeartbeatHandler;
