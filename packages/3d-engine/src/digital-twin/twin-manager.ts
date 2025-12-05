/**
 * Digital Twin Manager
 *
 * Core manager for digital twin lifecycle, state management,
 * and coordination between twins and source agents.
 *
 * @module @neurectomy/3d-engine/digital-twin/twin-manager
 * @agents @ARCHITECT @NEURAL @APEX
 * @phase Phase 3 - Dimensional Forge
 */

// Simple UUID v4 generator (crypto-based)
function generateUUID(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

import type {
  TwinId,
  AgentId,
  TwinState,
  TwinMode,
  SyncState,
  TwinFidelity,
  TwinMetadata,
  AgentStateSnapshot,
  TwinManagerConfig,
  TwinQuery,
  TwinStatistics,
  TwinEvent,
  TwinEventListener,
  SyncConfig,
  ComponentGraphSnapshot,
  AgentMetrics,
} from "./types";

// ============================================================================
// Default Values
// ============================================================================

const DEFAULT_CONFIG: Required<TwinManagerConfig> = {
  maxTwins: 100,
  defaultSyncConfig: {
    mode: "periodic",
    intervalMs: 5000,
    conflictResolution: "source-wins",
    compression: true,
    batchSize: 100,
  },
  defaultPredictionConfig: {
    horizonMs: 60000,
    stepMs: 1000,
    scenarioCount: 3,
    inputScenarios: [],
    quantifyUncertainty: true,
    confidenceLevel: 0.95,
  },
  autoCleanup: true,
  staleThresholdMs: 24 * 60 * 60 * 1000, // 24 hours
  persistence: true,
  storageBackend: "memory",
};

const createDefaultAgentState = (): AgentStateSnapshot => ({
  config: {},
  parameters: {},
  internalState: {},
  ioHistory: [],
  metrics: createDefaultMetrics(),
  componentGraph: createDefaultGraph(),
});

const createDefaultMetrics = (): AgentMetrics => ({
  responseTime: {
    min: 0,
    max: 0,
    mean: 0,
    median: 0,
    p95: 0,
    p99: 0,
    stdDev: 0,
  },
  throughput: { min: 0, max: 0, mean: 0, median: 0, p95: 0, p99: 0, stdDev: 0 },
  errorRate: 0,
  resourceUtilization: {
    cpuPercent: 0,
    memoryMB: 0,
    networkBytesIn: 0,
    networkBytesOut: 0,
  },
  custom: {},
});

const createDefaultGraph = (): ComponentGraphSnapshot => ({
  nodes: [],
  edges: [],
  rootId: "",
});

// ============================================================================
// Twin Manager Class
// ============================================================================

/**
 * Digital Twin Manager
 *
 * Manages the complete lifecycle of digital twins including creation,
 * synchronization, prediction, and comparison operations.
 */
export class TwinManager {
  private config: Required<TwinManagerConfig>;
  private twins: Map<TwinId, TwinState> = new Map();
  private listeners: Set<TwinEventListener> = new Set();
  private syncIntervals: Map<TwinId, ReturnType<typeof setInterval>> =
    new Map();
  private statistics: TwinStatistics;

  constructor(config: TwinManagerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.statistics = this.initStatistics();

    if (this.config.autoCleanup) {
      this.startAutoCleanup();
    }
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  /**
   * Create a new digital twin from an agent
   */
  async createTwin(
    agentId: AgentId,
    options: {
      name?: string;
      mode?: TwinMode;
      fidelity?: TwinFidelity;
      metadata?: Partial<TwinMetadata>;
      initialState?: Partial<AgentStateSnapshot>;
      syncConfig?: Partial<SyncConfig>;
    } = {}
  ): Promise<TwinState> {
    // Check twin limit
    if (this.twins.size >= this.config.maxTwins) {
      throw new Error(`Maximum twin limit (${this.config.maxTwins}) reached`);
    }

    const now = Date.now();
    const twinId = generateUUID();

    const twin: TwinState = {
      id: twinId,
      agentId,
      name: options.name || `Twin-${agentId.slice(0, 8)}`,
      mode: options.mode || "mirror",
      syncState: "disconnected",
      fidelity: options.fidelity || "full",
      createdAt: now,
      lastSyncAt: 0,
      modifiedAt: now,
      agentState: {
        ...createDefaultAgentState(),
        ...options.initialState,
      },
      metadata: {
        tags: [],
        version: "1.0.0",
        properties: {},
        ...options.metadata,
      },
      divergenceScore: 0,
    };

    this.twins.set(twinId, twin);
    this.updateStatistics();

    // Start sync if in mirror mode
    if (twin.mode === "mirror") {
      await this.startSync(twinId, {
        ...this.config.defaultSyncConfig,
        ...options.syncConfig,
      });
    }

    this.emit({ type: "twin:created", twin });

    return twin;
  }

  /**
   * Get a twin by ID
   */
  getTwin(twinId: TwinId): TwinState | undefined {
    return this.twins.get(twinId);
  }

  /**
   * Get all twins
   */
  getAllTwins(): TwinState[] {
    return Array.from(this.twins.values());
  }

  /**
   * Query twins with filters
   */
  queryTwins(query: TwinQuery): TwinState[] {
    let results = Array.from(this.twins.values());

    // Apply filters
    if (query.agentId) {
      results = results.filter((t) => t.agentId === query.agentId);
    }
    if (query.mode) {
      results = results.filter((t) => t.mode === query.mode);
    }
    if (query.syncState) {
      results = results.filter((t) => t.syncState === query.syncState);
    }
    if (query.tags && query.tags.length > 0) {
      results = results.filter((t) =>
        query.tags!.some((tag) => t.metadata.tags.includes(tag))
      );
    }
    if (query.createdAfter) {
      results = results.filter((t) => t.createdAt >= query.createdAfter!);
    }
    if (query.createdBefore) {
      results = results.filter((t) => t.createdAt <= query.createdBefore!);
    }

    // Apply sorting
    if (query.sortBy) {
      const direction = query.sortDirection === "desc" ? -1 : 1;
      results.sort((a, b) => {
        const aVal = a[query.sortBy!];
        const bVal = b[query.sortBy!];
        if (typeof aVal === "string" && typeof bVal === "string") {
          return direction * aVal.localeCompare(bVal);
        }
        return direction * ((aVal as number) - (bVal as number));
      });
    }

    // Apply pagination
    if (query.offset) {
      results = results.slice(query.offset);
    }
    if (query.limit) {
      results = results.slice(0, query.limit);
    }

    return results;
  }

  /**
   * Update a twin's properties
   */
  async updateTwin(
    twinId: TwinId,
    updates: Partial<Pick<TwinState, "name" | "metadata" | "fidelity">>
  ): Promise<TwinState> {
    const twin = this.twins.get(twinId);
    if (!twin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    const updatedTwin: TwinState = {
      ...twin,
      ...updates,
      modifiedAt: Date.now(),
      metadata: {
        ...twin.metadata,
        ...updates.metadata,
      },
    };

    this.twins.set(twinId, updatedTwin);
    this.emit({ type: "twin:updated", twinId, changes: updates });

    return updatedTwin;
  }

  /**
   * Delete a twin
   */
  async deleteTwin(twinId: TwinId): Promise<boolean> {
    const twin = this.twins.get(twinId);
    if (!twin) {
      return false;
    }

    // Stop sync if running
    this.stopSync(twinId);

    this.twins.delete(twinId);
    this.updateStatistics();
    this.emit({ type: "twin:deleted", twinId });

    return true;
  }

  /**
   * Change twin mode
   */
  async setTwinMode(twinId: TwinId, mode: TwinMode): Promise<TwinState> {
    const twin = this.twins.get(twinId);
    if (!twin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    const oldMode = twin.mode;
    if (oldMode === mode) {
      return twin;
    }

    // Handle mode transitions
    if (oldMode === "mirror") {
      this.stopSync(twinId);
    }

    const updatedTwin: TwinState = {
      ...twin,
      mode,
      modifiedAt: Date.now(),
      syncState: mode === "mirror" ? "syncing" : "disconnected",
    };

    this.twins.set(twinId, updatedTwin);

    if (mode === "mirror") {
      await this.startSync(twinId, this.config.defaultSyncConfig);
    }

    this.emit({ type: "twin:mode-changed", twinId, oldMode, newMode: mode });

    return updatedTwin;
  }

  // ==========================================================================
  // State Management
  // ==========================================================================

  /**
   * Update twin's agent state
   */
  updateTwinState(
    twinId: TwinId,
    stateUpdate: Partial<AgentStateSnapshot>
  ): TwinState {
    const twin = this.twins.get(twinId);
    if (!twin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    const updatedTwin: TwinState = {
      ...twin,
      agentState: {
        ...twin.agentState,
        ...stateUpdate,
        // Ensure required nested objects are preserved
        componentGraph:
          stateUpdate.componentGraph ?? twin.agentState.componentGraph,
        metrics: stateUpdate.metrics ?? twin.agentState.metrics,
        config: stateUpdate.config ?? twin.agentState.config,
        parameters: stateUpdate.parameters ?? twin.agentState.parameters,
        capabilities: stateUpdate.capabilities ?? twin.agentState.capabilities,
        connections: stateUpdate.connections ?? twin.agentState.connections,
      } as AgentStateSnapshot,
      modifiedAt: Date.now(),
    };

    this.twins.set(twinId, updatedTwin);
    this.emit({
      type: "twin:updated",
      twinId,
      changes: { agentState: stateUpdate },
    });

    return updatedTwin;
  }

  /**
   * Take a snapshot of current twin state
   */
  createSnapshot(twinId: TwinId): AgentStateSnapshot {
    const twin = this.twins.get(twinId);
    if (!twin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    // Deep clone the state
    return JSON.parse(JSON.stringify(twin.agentState));
  }

  /**
   * Restore twin state from a snapshot
   */
  restoreSnapshot(twinId: TwinId, snapshot: AgentStateSnapshot): TwinState {
    const twin = this.twins.get(twinId);
    if (!twin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    const updatedTwin: TwinState = {
      ...twin,
      agentState: snapshot,
      modifiedAt: Date.now(),
      divergenceScore:
        twin.mode === "mirror" ? this.calculateDivergence(twin, snapshot) : 0,
    };

    this.twins.set(twinId, updatedTwin);

    return updatedTwin;
  }

  /**
   * Fork a twin to create a new experimental branch
   */
  async forkTwin(
    twinId: TwinId,
    options: {
      name?: string;
      branch?: string;
    } = {}
  ): Promise<TwinState> {
    const sourceTwin = this.twins.get(twinId);
    if (!sourceTwin) {
      throw new Error(`Twin ${twinId} not found`);
    }

    return this.createTwin(sourceTwin.agentId, {
      name: options.name || `${sourceTwin.name}-fork`,
      mode: "sandbox",
      fidelity: sourceTwin.fidelity,
      initialState: JSON.parse(JSON.stringify(sourceTwin.agentState)),
      metadata: {
        ...sourceTwin.metadata,
        parentTwinId: twinId,
        branch: options.branch || "experiment",
        version: `${sourceTwin.metadata.version}-fork`,
      },
    });
  }

  // ==========================================================================
  // Synchronization
  // ==========================================================================

  /**
   * Start synchronization for a twin
   */
  private async startSync(
    twinId: TwinId,
    config: Partial<SyncConfig>
  ): Promise<void> {
    const fullConfig: SyncConfig = {
      ...this.config.defaultSyncConfig,
      ...config,
    } as SyncConfig;

    // Clear existing interval
    this.stopSync(twinId);

    const twin = this.twins.get(twinId);
    if (!twin) return;

    // Update sync state
    this.updateSyncState(twinId, "syncing");

    if (fullConfig.mode === "periodic" && fullConfig.intervalMs) {
      const interval = setInterval(async () => {
        await this.performSync(twinId);
      }, fullConfig.intervalMs);

      this.syncIntervals.set(twinId, interval);
    }

    // Perform initial sync
    await this.performSync(twinId);
  }

  /**
   * Stop synchronization for a twin
   */
  private stopSync(twinId: TwinId): void {
    const interval = this.syncIntervals.get(twinId);
    if (interval) {
      clearInterval(interval);
      this.syncIntervals.delete(twinId);
    }
    this.updateSyncState(twinId, "disconnected");
  }

  /**
   * Perform a sync operation
   */
  private async performSync(twinId: TwinId): Promise<void> {
    const twin = this.twins.get(twinId);
    if (!twin || twin.mode !== "mirror") return;

    try {
      // In a real implementation, this would fetch from the actual agent
      // For now, we simulate a successful sync
      const now = Date.now();

      const updatedTwin: TwinState = {
        ...twin,
        lastSyncAt: now,
        syncState: "synced",
        divergenceScore: 0,
      };

      this.twins.set(twinId, updatedTwin);
      this.statistics.totalSyncs++;

      this.emit({
        type: "twin:synced",
        twinId,
        result: {
          success: true,
          timestamp: now,
          changesApplied: 0,
          conflicts: [],
          bytesTransferred: 0,
          durationMs: 0,
        },
      });
    } catch (error) {
      this.updateSyncState(twinId, "diverged");
      this.emit({
        type: "twin:error",
        twinId,
        error: error instanceof Error ? error.message : "Sync failed",
      });
    }
  }

  /**
   * Update sync state for a twin
   */
  private updateSyncState(twinId: TwinId, state: SyncState): void {
    const twin = this.twins.get(twinId);
    if (twin) {
      this.twins.set(twinId, { ...twin, syncState: state });
    }
  }

  /**
   * Calculate divergence between twin and a state
   */
  private calculateDivergence(
    twin: TwinState,
    newState: AgentStateSnapshot
  ): number {
    // Simple divergence calculation based on JSON diff
    const originalJson = JSON.stringify(twin.agentState);
    const newJson = JSON.stringify(newState);

    if (originalJson === newJson) return 0;

    // Calculate Levenshtein-like distance normalized to 0-1
    const maxLen = Math.max(originalJson.length, newJson.length);
    let differences = 0;

    for (let i = 0; i < maxLen; i++) {
      if (originalJson[i] !== newJson[i]) {
        differences++;
      }
    }

    return Math.min(1, differences / maxLen);
  }

  // ==========================================================================
  // Statistics & Cleanup
  // ==========================================================================

  /**
   * Get twin statistics
   */
  getStatistics(): TwinStatistics {
    this.updateStatistics();
    return { ...this.statistics };
  }

  /**
   * Initialize statistics object
   */
  private initStatistics(): TwinStatistics {
    return {
      totalTwins: 0,
      twinsByMode: { mirror: 0, snapshot: 0, sandbox: 0, predictive: 0 },
      twinsBySyncState: { synced: 0, syncing: 0, diverged: 0, disconnected: 0 },
      averageDivergence: 0,
      totalSyncs: 0,
      totalPredictions: 0,
      storageUsedBytes: 0,
    };
  }

  /**
   * Update statistics
   */
  private updateStatistics(): void {
    const twins = Array.from(this.twins.values());

    this.statistics.totalTwins = twins.length;
    this.statistics.twinsByMode = {
      mirror: 0,
      snapshot: 0,
      sandbox: 0,
      predictive: 0,
    };
    this.statistics.twinsBySyncState = {
      synced: 0,
      syncing: 0,
      diverged: 0,
      disconnected: 0,
    };

    let totalDivergence = 0;

    for (const twin of twins) {
      this.statistics.twinsByMode[twin.mode]++;
      this.statistics.twinsBySyncState[twin.syncState]++;
      totalDivergence += twin.divergenceScore;
    }

    this.statistics.averageDivergence =
      twins.length > 0 ? totalDivergence / twins.length : 0;

    // Estimate storage (rough calculation)
    this.statistics.storageUsedBytes = JSON.stringify(
      Array.from(this.twins.values())
    ).length;
  }

  /**
   * Start auto-cleanup of stale twins
   */
  private startAutoCleanup(): void {
    setInterval(() => {
      const now = Date.now();
      const staleThreshold = now - this.config.staleThresholdMs;

      for (const [twinId, twin] of this.twins) {
        if (twin.mode === "snapshot" && twin.modifiedAt < staleThreshold) {
          this.deleteTwin(twinId);
        }
      }
    }, this.config.staleThresholdMs / 10);
  }

  // ==========================================================================
  // Event System
  // ==========================================================================

  /**
   * Subscribe to twin events
   */
  subscribe(listener: TwinEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Emit an event to all listeners
   */
  private emit(event: TwinEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error("Twin event listener error:", error);
      }
    }
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Dispose of the manager and cleanup resources
   */
  dispose(): void {
    // Stop all sync intervals
    for (const interval of this.syncIntervals.values()) {
      clearInterval(interval);
    }
    this.syncIntervals.clear();

    // Clear all twins
    this.twins.clear();

    // Clear listeners
    this.listeners.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let managerInstance: TwinManager | null = null;

/**
 * Get the global TwinManager instance
 */
export function getTwinManager(config?: TwinManagerConfig): TwinManager {
  if (!managerInstance) {
    managerInstance = new TwinManager(config);
  }
  return managerInstance;
}

/**
 * Reset the global TwinManager instance
 */
export function resetTwinManager(): void {
  if (managerInstance) {
    managerInstance.dispose();
    managerInstance = null;
  }
}
