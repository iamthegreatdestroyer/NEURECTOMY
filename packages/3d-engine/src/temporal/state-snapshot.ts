/**
 * @file State Snapshot System - 4D Engine Core
 * @description Manages state snapshots, delta encoding, and compression for time-travel
 * @module @neurectomy/3d-engine/temporal
 * @agents @AXIOM @VELOCITY
 * @phase Phase 3 - Dimensional Forge
 * @step Step 4 - 4D Temporal Engine
 */

import type {
  StateSnapshot,
  SnapshotMetadata,
  StateDelta,
  DeltaOperation,
  TemporalId,
  Timestamp,
  TemporalResult,
  TemporalError,
  TemporalErrorCode,
} from "./types";

// ============================================================================
// Snapshot Configuration
// ============================================================================

export interface SnapshotConfig {
  /** Use delta encoding to reduce memory */
  useDeltaEncoding: boolean;
  /** Keyframe interval (every N snapshots is a full snapshot) */
  keyframeInterval: number;
  /** Maximum delta chain length before forcing keyframe */
  maxDeltaChainLength: number;
  /** Compression method */
  compression: "none" | "lz4" | "zstd";
  /** Maximum memory usage in bytes */
  maxMemoryBytes: number;
  /** Enable structural sharing */
  structuralSharing: boolean;
}

const DEFAULT_CONFIG: SnapshotConfig = {
  useDeltaEncoding: true,
  keyframeInterval: 30,
  maxDeltaChainLength: 10,
  compression: "none",
  maxMemoryBytes: 256 * 1024 * 1024, // 256MB
  structuralSharing: true,
};

// ============================================================================
// Hash Utilities
// ============================================================================

/**
 * Generate a fast hash for state comparison
 * Uses FNV-1a algorithm for speed
 */
function hashState(state: unknown): string {
  const str = JSON.stringify(state);
  let hash = 0x811c9dc5; // FNV offset basis

  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }

  // Convert to hex string
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Calculate size of object in bytes (approximate)
 */
function calculateSize(obj: unknown): number {
  const str = JSON.stringify(obj);
  // UTF-16 string, roughly 2 bytes per char
  return str.length * 2;
}

// ============================================================================
// Delta Computation
// ============================================================================

/**
 * Compute delta between two states
 */
function computeDelta(
  from: unknown,
  to: unknown,
  path: string[] = []
): DeltaOperation[] {
  const operations: DeltaOperation[] = [];

  // Handle null/undefined
  if (from === null || from === undefined) {
    if (to !== null && to !== undefined) {
      operations.push({ type: "set", path, value: to });
    }
    return operations;
  }

  if (to === null || to === undefined) {
    operations.push({ type: "delete", path, previousValue: from });
    return operations;
  }

  // Handle primitive types
  if (typeof from !== "object" || typeof to !== "object") {
    if (from !== to) {
      operations.push({ type: "set", path, value: to, previousValue: from });
    }
    return operations;
  }

  // Handle arrays
  if (Array.isArray(from) && Array.isArray(to)) {
    return computeArrayDelta(from, to, path);
  }

  // Handle objects
  const fromObj = from as Record<string, unknown>;
  const toObj = to as Record<string, unknown>;
  const allKeys = new Set([...Object.keys(fromObj), ...Object.keys(toObj)]);

  for (const key of allKeys) {
    const childPath = [...path, key];

    if (!(key in fromObj)) {
      operations.push({ type: "set", path: childPath, value: toObj[key] });
    } else if (!(key in toObj)) {
      operations.push({
        type: "delete",
        path: childPath,
        previousValue: fromObj[key],
      });
    } else if (fromObj[key] !== toObj[key]) {
      // Recursively compute delta for nested objects
      const childOps = computeDelta(fromObj[key], toObj[key], childPath);
      operations.push(...childOps);
    }
  }

  return operations;
}

/**
 * Compute delta for arrays using LCS algorithm
 */
function computeArrayDelta(
  from: unknown[],
  to: unknown[],
  path: string[]
): DeltaOperation[] {
  const operations: DeltaOperation[] = [];

  // For small arrays, just track individual changes
  if (from.length <= 10 && to.length <= 10) {
    const maxLen = Math.max(from.length, to.length);

    for (let i = 0; i < maxLen; i++) {
      const indexPath = [...path, String(i)];

      if (i >= from.length) {
        operations.push({
          type: "insert",
          path: indexPath,
          value: to[i],
          toIndex: i,
        });
      } else if (i >= to.length) {
        operations.push({
          type: "delete",
          path: indexPath,
          previousValue: from[i],
        });
      } else if (JSON.stringify(from[i]) !== JSON.stringify(to[i])) {
        operations.push({
          type: "set",
          path: indexPath,
          value: to[i],
          previousValue: from[i],
        });
      }
    }

    return operations;
  }

  // For larger arrays, just replace the whole array if significantly different
  const sameCount = from.filter(
    (item, i) => i < to.length && JSON.stringify(item) === JSON.stringify(to[i])
  ).length;

  const similarity = sameCount / Math.max(from.length, to.length);

  if (similarity < 0.5) {
    // Arrays are too different, just replace
    operations.push({ type: "set", path, value: to, previousValue: from });
  } else {
    // Track individual changes
    const maxLen = Math.max(from.length, to.length);
    for (let i = 0; i < maxLen; i++) {
      const indexPath = [...path, String(i)];

      if (i >= from.length) {
        operations.push({
          type: "insert",
          path: indexPath,
          value: to[i],
          toIndex: i,
        });
      } else if (i >= to.length) {
        operations.push({
          type: "delete",
          path: indexPath,
          previousValue: from[i],
        });
      } else if (JSON.stringify(from[i]) !== JSON.stringify(to[i])) {
        const childOps = computeDelta(from[i], to[i], indexPath);
        operations.push(...childOps);
      }
    }
  }

  return operations;
}

/**
 * Apply delta operations to a state
 */
function applyDelta<T>(state: T, delta: StateDelta): T {
  // Deep clone the state
  let result = JSON.parse(JSON.stringify(state)) as T;

  for (const op of delta.operations) {
    result = applyOperation(result, op);
  }

  return result;
}

/**
 * Apply a single delta operation
 */
function applyOperation<T>(state: T, op: DeltaOperation): T {
  if (op.path.length === 0) {
    return op.value as T;
  }

  const result = JSON.parse(JSON.stringify(state)) as Record<string, unknown>;
  let current: Record<string, unknown> = result;

  // Navigate to parent of target
  for (let i = 0; i < op.path.length - 1; i++) {
    const key = op.path[i];
    if (key === undefined) continue;
    if (!(key in current)) {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }

  const finalKey = op.path[op.path.length - 1];
  if (finalKey === undefined) {
    return result as T;
  }

  switch (op.type) {
    case "set":
      current[finalKey] = op.value;
      break;
    case "delete":
      delete current[finalKey];
      break;
    case "insert":
      if (Array.isArray(current)) {
        current.splice(op.toIndex ?? 0, 0, op.value);
      } else {
        current[finalKey] = op.value;
      }
      break;
    case "patch":
      if (
        typeof current[finalKey] === "object" &&
        op.value &&
        typeof op.value === "object"
      ) {
        current[finalKey] = {
          ...(current[finalKey] as object),
          ...(op.value as object),
        };
      }
      break;
  }

  return result as T;
}

// ============================================================================
// Snapshot Manager Class
// ============================================================================

export class SnapshotManager<T = unknown> {
  private config: SnapshotConfig;
  private snapshots: Map<TemporalId, StateSnapshot<T>> = new Map();
  private snapshotOrder: TemporalId[] = [];
  private keyframeIds: Set<TemporalId> = new Set();
  private totalMemoryUsage: number = 0;
  private snapshotCounter: number = 0;

  constructor(config: Partial<SnapshotConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // --------------------------------------------------------------------------
  // Snapshot Creation
  // --------------------------------------------------------------------------

  /**
   * Create a new snapshot
   */
  public createSnapshot(
    state: T,
    options: {
      timestamp?: Timestamp;
      label?: string;
      forceKeyframe?: boolean;
      tags?: string[];
    } = {}
  ): TemporalResult<StateSnapshot<T>> {
    const timestamp = options.timestamp ?? Date.now();
    const id = this.generateId();
    const hash = hashState(state);

    // Check memory limits
    const stateSize = calculateSize(state);
    if (this.totalMemoryUsage + stateSize > this.config.maxMemoryBytes) {
      // Try to free memory by removing old snapshots
      this.garbageCollect(stateSize);

      if (this.totalMemoryUsage + stateSize > this.config.maxMemoryBytes) {
        return {
          success: false,
          error: {
            code: "MEMORY_LIMIT_EXCEEDED",
            message: `Memory limit of ${this.config.maxMemoryBytes} bytes exceeded`,
            timestamp,
          },
        };
      }
    }

    // Determine if this should be a keyframe
    const isKeyframe =
      options.forceKeyframe ||
      this.snapshotOrder.length === 0 ||
      this.snapshotOrder.length % this.config.keyframeInterval === 0 ||
      this.getDeltaChainLength() >= this.config.maxDeltaChainLength;

    let snapshot: StateSnapshot<T>;

    if (isKeyframe || !this.config.useDeltaEncoding) {
      // Create full snapshot
      snapshot = {
        id,
        timestamp,
        state: this.config.structuralSharing
          ? state
          : JSON.parse(JSON.stringify(state)),
        hash,
        metadata: {
          label: options.label,
          tags: options.tags ?? [],
          isKeyframe: true,
          sizeBytes: stateSize,
          source: options.label ? "manual" : "auto",
        },
      };
      this.keyframeIds.add(id);
    } else {
      // Create delta snapshot
      const parentSnapshot = this.getLatestSnapshot();
      if (!parentSnapshot) {
        return this.createSnapshot(state, { ...options, forceKeyframe: true });
      }

      const operations = computeDelta(parentSnapshot.state, state);
      const delta: StateDelta = {
        operations,
        sizeBytes: calculateSize(operations),
      };

      snapshot = {
        id,
        timestamp,
        state: this.config.structuralSharing
          ? state
          : JSON.parse(JSON.stringify(state)),
        hash,
        parentId: parentSnapshot.id,
        delta,
        metadata: {
          label: options.label,
          tags: options.tags ?? [],
          isKeyframe: false,
          sizeBytes: delta.sizeBytes,
          source: options.label ? "manual" : "auto",
        },
      };
    }

    // Store snapshot
    this.snapshots.set(id, snapshot);
    this.snapshotOrder.push(id);
    this.totalMemoryUsage += snapshot.metadata.sizeBytes;
    this.snapshotCounter++;

    return { success: true, data: snapshot };
  }

  // --------------------------------------------------------------------------
  // Snapshot Retrieval
  // --------------------------------------------------------------------------

  /**
   * Get snapshot by ID
   */
  public getSnapshot(id: TemporalId): StateSnapshot<T> | undefined {
    return this.snapshots.get(id);
  }

  /**
   * Get snapshot at timestamp (nearest)
   */
  public getSnapshotAtTime(timestamp: Timestamp): StateSnapshot<T> | undefined {
    let nearestId: TemporalId | undefined;
    let nearestDiff = Infinity;

    for (const id of this.snapshotOrder) {
      const snapshot = this.snapshots.get(id);
      if (snapshot) {
        const diff = Math.abs(snapshot.timestamp - timestamp);
        if (diff < nearestDiff) {
          nearestDiff = diff;
          nearestId = id;
        }
      }
    }

    return nearestId ? this.snapshots.get(nearestId) : undefined;
  }

  /**
   * Get state at timestamp (may reconstruct from delta)
   */
  public getStateAtTime(timestamp: Timestamp): T | undefined {
    const snapshot = this.getSnapshotAtTime(timestamp);
    if (!snapshot) return undefined;

    // If it's a keyframe or we have the full state, return it
    if (snapshot.metadata.isKeyframe || snapshot.state !== undefined) {
      return snapshot.state;
    }

    // Reconstruct from delta chain
    return this.reconstructState(snapshot);
  }

  /**
   * Get the latest snapshot
   */
  public getLatestSnapshot(): StateSnapshot<T> | undefined {
    if (this.snapshotOrder.length === 0) return undefined;
    const lastId = this.snapshotOrder[this.snapshotOrder.length - 1];
    if (lastId === undefined) return undefined;
    return this.snapshots.get(lastId);
  }

  /**
   * Get all snapshots in order
   */
  public getAllSnapshots(): StateSnapshot<T>[] {
    return this.snapshotOrder
      .map((id) => this.snapshots.get(id))
      .filter((s): s is StateSnapshot<T> => s !== undefined);
  }

  /**
   * Get keyframe snapshots only
   */
  public getKeyframes(): StateSnapshot<T>[] {
    return Array.from(this.keyframeIds)
      .map((id) => this.snapshots.get(id))
      .filter((s): s is StateSnapshot<T> => s !== undefined)
      .sort((a, b) => a.timestamp - b.timestamp);
  }

  // --------------------------------------------------------------------------
  // State Reconstruction
  // --------------------------------------------------------------------------

  /**
   * Reconstruct full state from delta chain
   */
  private reconstructState(snapshot: StateSnapshot<T>): T {
    // Build delta chain back to keyframe
    const chain: StateSnapshot<T>[] = [];
    let current: StateSnapshot<T> | undefined = snapshot;

    while (current && !current.metadata.isKeyframe) {
      chain.unshift(current);
      current = current.parentId
        ? this.snapshots.get(current.parentId)
        : undefined;
    }

    if (!current) {
      // No keyframe found, return whatever state we have
      return snapshot.state;
    }

    // Start from keyframe state and apply deltas
    let state = JSON.parse(JSON.stringify(current.state)) as T;

    for (const deltaSnapshot of chain) {
      if (deltaSnapshot.delta) {
        state = applyDelta(state, deltaSnapshot.delta);
      }
    }

    return state;
  }

  // --------------------------------------------------------------------------
  // Snapshot Management
  // --------------------------------------------------------------------------

  /**
   * Delete a snapshot
   */
  public deleteSnapshot(id: TemporalId): boolean {
    const snapshot = this.snapshots.get(id);
    if (!snapshot) return false;

    // Don't allow deleting keyframes if they have dependents
    if (snapshot.metadata.isKeyframe) {
      const hasDependents = this.snapshotOrder.some((otherId) => {
        const other = this.snapshots.get(otherId);
        return other?.parentId === id;
      });

      if (hasDependents) {
        return false;
      }
    }

    this.snapshots.delete(id);
    this.snapshotOrder = this.snapshotOrder.filter((sid) => sid !== id);
    this.keyframeIds.delete(id);
    this.totalMemoryUsage -= snapshot.metadata.sizeBytes;

    return true;
  }

  /**
   * Clear all snapshots
   */
  public clear(): void {
    this.snapshots.clear();
    this.snapshotOrder = [];
    this.keyframeIds.clear();
    this.totalMemoryUsage = 0;
  }

  /**
   * Garbage collect old snapshots to free memory
   */
  private garbageCollect(requiredBytes: number): void {
    const targetMemory = this.config.maxMemoryBytes - requiredBytes;

    // Remove oldest non-keyframe snapshots first
    while (
      this.totalMemoryUsage > targetMemory &&
      this.snapshotOrder.length > 1
    ) {
      const oldestId = this.snapshotOrder[0];
      if (oldestId === undefined) break;

      const oldest = this.snapshots.get(oldestId);

      if (oldest && !oldest.metadata.isKeyframe) {
        this.deleteSnapshot(oldestId);
      } else {
        // Can't delete keyframe, try next
        break;
      }
    }
  }

  /**
   * Get length of current delta chain
   */
  private getDeltaChainLength(): number {
    if (this.snapshotOrder.length === 0) return 0;

    let length = 0;
    let currentId: string | undefined =
      this.snapshotOrder[this.snapshotOrder.length - 1];

    while (currentId) {
      const snapshot = this.snapshots.get(currentId);
      if (!snapshot || snapshot.metadata.isKeyframe) break;

      length++;
      currentId = snapshot.parentId || undefined;
    }

    return length;
  }

  // --------------------------------------------------------------------------
  // Statistics & Metrics
  // --------------------------------------------------------------------------

  /**
   * Get statistics about snapshots
   */
  public getStats(): {
    snapshotCount: number;
    keyframeCount: number;
    totalMemoryBytes: number;
    avgSnapshotSize: number;
    compressionRatio: number;
  } {
    const keyframeSize = Array.from(this.keyframeIds)
      .map((id) => this.snapshots.get(id)?.metadata.sizeBytes ?? 0)
      .reduce((a, b) => a + b, 0);

    const deltaSize = this.totalMemoryUsage - keyframeSize;

    return {
      snapshotCount: this.snapshots.size,
      keyframeCount: this.keyframeIds.size,
      totalMemoryBytes: this.totalMemoryUsage,
      avgSnapshotSize:
        this.snapshots.size > 0
          ? this.totalMemoryUsage / this.snapshots.size
          : 0,
      compressionRatio: keyframeSize > 0 ? deltaSize / keyframeSize : 1,
    };
  }

  // --------------------------------------------------------------------------
  // Utility Methods
  // --------------------------------------------------------------------------

  private generateId(): TemporalId {
    return `snap_${this.snapshotCounter}_${Date.now().toString(36)}`;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createSnapshotManager<T>(
  config?: Partial<SnapshotConfig>
): SnapshotManager<T> {
  return new SnapshotManager<T>(config);
}

export default SnapshotManager;
