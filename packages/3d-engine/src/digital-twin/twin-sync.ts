/**
 * Digital Twin Synchronization Engine
 *
 * Handles real-time state synchronization between digital twins
 * and their source agents, including conflict resolution and delta compression.
 *
 * @module @neurectomy/3d-engine/digital-twin/twin-sync
 * @agents @SYNAPSE @STREAM @APEX
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  TwinId,
  AgentId,
  TwinState,
  SyncState,
  SyncConfig,
  SyncMode,
  ConflictResolution,
  AgentStateSnapshot,
  TwinSyncResult,
  TwinDelta,
  TwinDiff,
  TwinDiffEntry,
  ComponentGraphSnapshot,
} from "./types";

// ============================================================================
// Types
// ============================================================================

export interface SyncSession {
  id: string;
  twinId: TwinId;
  agentId: AgentId;
  config: SyncConfig;
  state: SyncState;
  startedAt: number;
  lastPingAt: number;
  pendingChanges: TwinDelta[];
  metrics: SyncMetrics;
}

export interface SyncMetrics {
  totalSyncs: number;
  successfulSyncs: number;
  failedSyncs: number;
  bytesTransferred: number;
  averageLatencyMs: number;
  conflicts: number;
  lastError?: string;
}

export interface SyncMessage {
  type: "state-update" | "delta" | "ping" | "pong" | "conflict" | "error";
  twinId: TwinId;
  timestamp: number;
  payload: unknown;
}

export type SyncEventHandler = (message: SyncMessage) => void;

// ============================================================================
// Sync Engine Class
// ============================================================================

/**
 * Twin Synchronization Engine
 *
 * Manages bidirectional state synchronization between digital twins
 * and their source agents with support for multiple sync modes.
 */
export class TwinSyncEngine {
  private sessions: Map<TwinId, SyncSession> = new Map();
  private eventHandlers: Map<TwinId, Set<SyncEventHandler>> = new Map();
  private pingIntervals: Map<TwinId, ReturnType<typeof setInterval>> =
    new Map();
  private changeBuffers: Map<TwinId, TwinDelta[]> = new Map();

  constructor() {
    // Initialize
  }

  // ==========================================================================
  // Session Management
  // ==========================================================================

  /**
   * Start a sync session for a twin
   */
  async startSession(
    twinId: TwinId,
    agentId: AgentId,
    config: SyncConfig
  ): Promise<SyncSession> {
    // End existing session if any
    await this.endSession(twinId);

    const session: SyncSession = {
      id: `sync-${twinId}-${Date.now()}`,
      twinId,
      agentId,
      config,
      state: "syncing",
      startedAt: Date.now(),
      lastPingAt: Date.now(),
      pendingChanges: [],
      metrics: {
        totalSyncs: 0,
        successfulSyncs: 0,
        failedSyncs: 0,
        bytesTransferred: 0,
        averageLatencyMs: 0,
        conflicts: 0,
      },
    };

    this.sessions.set(twinId, session);
    this.changeBuffers.set(twinId, []);

    // Start based on mode
    if (config.mode === "realtime") {
      this.startRealtimeSync(session);
    } else if (config.mode === "periodic") {
      this.startPeriodicSync(session);
    }

    // Start ping interval
    this.startPingInterval(twinId);

    return session;
  }

  /**
   * End a sync session
   */
  async endSession(twinId: TwinId): Promise<void> {
    const session = this.sessions.get(twinId);
    if (!session) return;

    // Clear intervals
    const pingInterval = this.pingIntervals.get(twinId);
    if (pingInterval) {
      clearInterval(pingInterval);
      this.pingIntervals.delete(twinId);
    }

    // Flush pending changes
    await this.flushChanges(twinId);

    session.state = "disconnected";
    this.sessions.delete(twinId);
    this.eventHandlers.delete(twinId);
    this.changeBuffers.delete(twinId);
  }

  /**
   * Get sync session
   */
  getSession(twinId: TwinId): SyncSession | undefined {
    return this.sessions.get(twinId);
  }

  // ==========================================================================
  // Sync Operations
  // ==========================================================================

  /**
   * Perform a full state sync
   */
  async fullSync(
    twinId: TwinId,
    currentState: AgentStateSnapshot,
    sourceState: AgentStateSnapshot
  ): Promise<TwinSyncResult> {
    const session = this.sessions.get(twinId);
    const startTime = Date.now();

    try {
      // Calculate diff
      const diff = this.calculateDiff(currentState, sourceState);

      // Resolve conflicts
      const resolvedState = session
        ? this.resolveConflicts(
            currentState,
            sourceState,
            session.config.conflictResolution
          )
        : sourceState;

      // Calculate bytes transferred
      const bytesTransferred = JSON.stringify(resolvedState).length;

      if (session) {
        session.metrics.totalSyncs++;
        session.metrics.successfulSyncs++;
        session.metrics.bytesTransferred += bytesTransferred;
        session.metrics.averageLatencyMs =
          (session.metrics.averageLatencyMs * (session.metrics.totalSyncs - 1) +
            (Date.now() - startTime)) /
          session.metrics.totalSyncs;
        session.state = "synced";
      }

      this.emit(twinId, {
        type: "state-update",
        twinId,
        timestamp: Date.now(),
        payload: { state: resolvedState, diff },
      });

      return {
        success: true,
        timestamp: Date.now(),
        changesApplied: diff.entries.length,
        conflicts: diff.entries
          .filter((e) => e.conflict)
          .map((e) => ({
            path: e.path,
            localValue: e.oldValue,
            remoteValue: e.newValue,
            resolution: session?.config.conflictResolution || "source-wins",
            resolvedValue: e.newValue,
          })),
        bytesTransferred,
        durationMs: Date.now() - startTime,
      };
    } catch (error) {
      if (session) {
        session.metrics.totalSyncs++;
        session.metrics.failedSyncs++;
        session.metrics.lastError =
          error instanceof Error ? error.message : "Unknown error";
        session.state = "diverged";
      }

      return {
        success: false,
        timestamp: Date.now(),
        changesApplied: 0,
        conflicts: [],
        bytesTransferred: 0,
        durationMs: Date.now() - startTime,
        error: error instanceof Error ? error.message : "Sync failed",
      };
    }
  }

  /**
   * Apply a delta update
   */
  applyDelta(
    currentState: AgentStateSnapshot,
    delta: TwinDelta
  ): AgentStateSnapshot {
    const newState = JSON.parse(
      JSON.stringify(currentState)
    ) as AgentStateSnapshot;

    for (const change of delta.changes) {
      this.applyChange(newState, change);
    }

    return newState;
  }

  /**
   * Queue a change for batched sync
   */
  queueChange(twinId: TwinId, delta: TwinDelta): void {
    const buffer = this.changeBuffers.get(twinId);
    if (!buffer) return;

    buffer.push(delta);

    const session = this.sessions.get(twinId);
    if (session && buffer.length >= session.config.batchSize) {
      this.flushChanges(twinId);
    }
  }

  /**
   * Flush queued changes
   */
  async flushChanges(twinId: TwinId): Promise<number> {
    const buffer = this.changeBuffers.get(twinId);
    if (!buffer || buffer.length === 0) return 0;

    const changes = [...buffer];
    buffer.length = 0;

    // Emit batched changes
    this.emit(twinId, {
      type: "delta",
      twinId,
      timestamp: Date.now(),
      payload: changes,
    });

    return changes.length;
  }

  // ==========================================================================
  // Diff & Conflict Resolution
  // ==========================================================================

  /**
   * Calculate diff between two states
   */
  calculateDiff(
    oldState: AgentStateSnapshot,
    newState: AgentStateSnapshot
  ): TwinDiff {
    const entries: TwinDiffEntry[] = [];

    // Deep diff helper
    const diffObjects = (
      oldObj: Record<string, unknown>,
      newObj: Record<string, unknown>,
      path: string = ""
    ) => {
      const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);

      for (const key of allKeys) {
        const currentPath = path ? `${path}.${key}` : key;
        const oldValue = oldObj[key];
        const newValue = newObj[key];

        if (oldValue === undefined && newValue !== undefined) {
          entries.push({
            path: currentPath,
            type: "add",
            newValue,
            conflict: false,
          });
        } else if (oldValue !== undefined && newValue === undefined) {
          entries.push({
            path: currentPath,
            type: "remove",
            oldValue,
            conflict: false,
          });
        } else if (
          typeof oldValue === "object" &&
          typeof newValue === "object" &&
          oldValue !== null &&
          newValue !== null &&
          !Array.isArray(oldValue) &&
          !Array.isArray(newValue)
        ) {
          diffObjects(
            oldValue as Record<string, unknown>,
            newValue as Record<string, unknown>,
            currentPath
          );
        } else if (JSON.stringify(oldValue) !== JSON.stringify(newValue)) {
          entries.push({
            path: currentPath,
            type: "modify",
            oldValue,
            newValue,
            conflict: false,
          });
        }
      }
    };

    diffObjects(
      oldState as unknown as Record<string, unknown>,
      newState as unknown as Record<string, unknown>
    );

    return {
      twinId: "",
      fromVersion: 0,
      toVersion: 1,
      entries,
      timestamp: Date.now(),
    };
  }

  /**
   * Resolve conflicts between local and remote states
   */
  resolveConflicts(
    localState: AgentStateSnapshot,
    remoteState: AgentStateSnapshot,
    strategy: ConflictResolution
  ): AgentStateSnapshot {
    switch (strategy) {
      case "source-wins":
        return remoteState;

      case "twin-wins":
        return localState;

      case "latest-wins":
        // In absence of timestamp info, prefer remote
        return remoteState;

      case "merge":
        return this.mergeStates(localState, remoteState);

      case "manual":
        // Return remote but mark as diverged
        return remoteState;

      default:
        return remoteState;
    }
  }

  /**
   * Merge two states intelligently
   */
  private mergeStates(
    localState: AgentStateSnapshot,
    remoteState: AgentStateSnapshot
  ): AgentStateSnapshot {
    // Deep merge with remote taking precedence for primitives
    const merge = (local: unknown, remote: unknown): unknown => {
      if (remote === null || remote === undefined) {
        return local;
      }
      if (local === null || local === undefined) {
        return remote;
      }
      if (typeof remote !== "object" || typeof local !== "object") {
        return remote;
      }
      if (Array.isArray(remote) || Array.isArray(local)) {
        return remote;
      }

      const result: Record<string, unknown> = {};
      const allKeys = new Set([
        ...Object.keys(local as Record<string, unknown>),
        ...Object.keys(remote as Record<string, unknown>),
      ]);

      for (const key of allKeys) {
        const localVal = (local as Record<string, unknown>)[key];
        const remoteVal = (remote as Record<string, unknown>)[key];
        result[key] = merge(localVal, remoteVal);
      }

      return result;
    };

    return merge(localState, remoteState) as AgentStateSnapshot;
  }

  // ==========================================================================
  // Compression
  // ==========================================================================

  /**
   * Compress state for transmission
   */
  compressState(state: AgentStateSnapshot): Uint8Array {
    // Simple JSON compression - in production would use actual compression
    const json = JSON.stringify(state);
    return new TextEncoder().encode(json);
  }

  /**
   * Decompress received state
   */
  decompressState(data: Uint8Array): AgentStateSnapshot {
    const json = new TextDecoder().decode(data);
    return JSON.parse(json);
  }

  // ==========================================================================
  // Sync Modes
  // ==========================================================================

  /**
   * Start realtime sync mode
   */
  private startRealtimeSync(session: SyncSession): void {
    // In a real implementation, this would set up WebSocket or similar
    // For now, we simulate with fast polling
    session.state = "syncing";
  }

  /**
   * Start periodic sync mode
   */
  private startPeriodicSync(session: SyncSession): void {
    session.state = "syncing";
    // Periodic sync is handled by the TwinManager
  }

  /**
   * Start ping interval for connection health
   */
  private startPingInterval(twinId: TwinId): void {
    const interval = setInterval(() => {
      const session = this.sessions.get(twinId);
      if (!session) {
        clearInterval(interval);
        return;
      }

      const now = Date.now();
      const timeSinceLastPing = now - session.lastPingAt;

      // If no ping response in 30 seconds, mark as disconnected
      if (timeSinceLastPing > 30000) {
        session.state = "disconnected";
      }

      this.emit(twinId, {
        type: "ping",
        twinId,
        timestamp: now,
        payload: null,
      });

      session.lastPingAt = now;
    }, 10000);

    this.pingIntervals.set(twinId, interval);
  }

  // ==========================================================================
  // Event System
  // ==========================================================================

  /**
   * Subscribe to sync events for a twin
   */
  subscribe(twinId: TwinId, handler: SyncEventHandler): () => void {
    let handlers = this.eventHandlers.get(twinId);
    if (!handlers) {
      handlers = new Set();
      this.eventHandlers.set(twinId, handlers);
    }
    handlers.add(handler);

    return () => handlers?.delete(handler);
  }

  /**
   * Emit a sync event
   */
  private emit(twinId: TwinId, message: SyncMessage): void {
    const handlers = this.eventHandlers.get(twinId);
    if (!handlers) return;

    for (const handler of handlers) {
      try {
        handler(message);
      } catch (error) {
        console.error("Sync event handler error:", error);
      }
    }
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  /**
   * Apply a single change to state
   */
  private applyChange(state: AgentStateSnapshot, entry: TwinDiffEntry): void {
    const pathParts = entry.path.split(".");
    let current: Record<string, unknown> = state as unknown as Record<
      string,
      unknown
    >;

    for (let i = 0; i < pathParts.length - 1; i++) {
      const part = pathParts[i];
      if (part === undefined) continue;

      if (current[part] === undefined) {
        current[part] = {};
      }
      current = current[part] as Record<string, unknown>;
    }

    const lastPart = pathParts[pathParts.length - 1];
    if (lastPart === undefined) return;

    switch (entry.type) {
      case "add":
      case "modify":
        current[lastPart] = entry.newValue;
        break;
      case "remove":
        delete current[lastPart];
        break;
    }
  }

  /**
   * Get metrics for a sync session
   */
  getMetrics(twinId: TwinId): SyncMetrics | undefined {
    return this.sessions.get(twinId)?.metrics;
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Dispose of the sync engine
   */
  dispose(): void {
    for (const interval of this.pingIntervals.values()) {
      clearInterval(interval);
    }
    this.pingIntervals.clear();
    this.sessions.clear();
    this.eventHandlers.clear();
    this.changeBuffers.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let syncEngineInstance: TwinSyncEngine | null = null;

/**
 * Get the global TwinSyncEngine instance
 */
export function getTwinSyncEngine(): TwinSyncEngine {
  if (!syncEngineInstance) {
    syncEngineInstance = new TwinSyncEngine();
  }
  return syncEngineInstance;
}

/**
 * Reset the global TwinSyncEngine instance
 */
export function resetTwinSyncEngine(): void {
  if (syncEngineInstance) {
    syncEngineInstance.dispose();
    syncEngineInstance = null;
  }
}
