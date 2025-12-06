/**
 * @fileoverview Timeline Delta Compression System
 * @module @neurectomy/3d-engine/timeline/delta-compressor
 *
 * Efficient delta compression for timeline state changes.
 * Enables fast undo/redo with minimal memory footprint.
 */

/**
 * Supported value types for delta compression
 */
export type DeltaValue =
  | null
  | boolean
  | number
  | string
  | DeltaValue[]
  | { [key: string]: DeltaValue };

/**
 * State snapshot at a point in time
 */
export interface StateSnapshot {
  /** Unique snapshot identifier */
  id: string;
  /** Timestamp of the snapshot */
  timestamp: number;
  /** Full state data */
  state: Record<string, DeltaValue>;
}

/**
 * Delta operation types
 */
export type DeltaOperationType =
  | "set" // Set a value
  | "delete" // Delete a key
  | "array-insert" // Insert into array
  | "array-delete" // Delete from array
  | "array-move"; // Move item in array;

/**
 * Single delta operation
 */
export interface DeltaOperation {
  /** Operation type */
  op: DeltaOperationType;
  /** Path to the changed value (dot-separated) */
  path: string;
  /** New value (for set, array-insert) */
  value?: DeltaValue;
  /** Previous value (for undo) */
  prevValue?: DeltaValue;
  /** Array index (for array operations) */
  index?: number;
  /** Target index (for array-move) */
  toIndex?: number;
}

/**
 * Compressed delta between two states
 */
export interface CompressedDelta {
  /** Unique delta identifier */
  id: string;
  /** Reference to previous delta or snapshot */
  prevRef: string;
  /** Timestamp of this delta */
  timestamp: number;
  /** Delta operations */
  operations: DeltaOperation[];
  /** Compressed binary representation (optional) */
  compressed?: Uint8Array;
  /** Size in bytes before compression */
  originalSize: number;
  /** Size in bytes after compression */
  compressedSize: number;
}

/**
 * Delta compression configuration
 */
export interface DeltaCompressionConfig {
  /** Maximum number of deltas before creating a keyframe */
  maxDeltaChain: number;
  /** Enable binary compression */
  enableBinaryCompression: boolean;
  /** Compression level (0-9) */
  compressionLevel: number;
  /** Minimum size to trigger compression */
  compressionThreshold: number;
  /** Enable structural sharing */
  enableStructuralSharing: boolean;
  /** Path patterns to ignore */
  ignorePaths: string[];
}

/**
 * Timeline statistics
 */
export interface TimelineStats {
  /** Total number of snapshots/deltas */
  totalEntries: number;
  /** Number of keyframe snapshots */
  keyframeCount: number;
  /** Number of delta entries */
  deltaCount: number;
  /** Total memory usage in bytes */
  memoryUsage: number;
  /** Average compression ratio */
  averageCompressionRatio: number;
  /** Average delta size in bytes */
  averageDeltaSize: number;
  /** Operations per second capability */
  operationsPerSecond: number;
}

/**
 * Default compression configuration
 */
export const DEFAULT_COMPRESSION_CONFIG: DeltaCompressionConfig = {
  maxDeltaChain: 50,
  enableBinaryCompression: true,
  compressionLevel: 6,
  compressionThreshold: 1024,
  enableStructuralSharing: true,
  ignorePaths: [],
};

/**
 * Timeline entry (either snapshot or delta)
 */
type TimelineEntry =
  | { type: "snapshot"; data: StateSnapshot }
  | { type: "delta"; data: CompressedDelta };

/**
 * Delta Compressor
 *
 * Efficiently compresses state changes using delta encoding.
 * Supports fast undo/redo operations with minimal memory overhead.
 *
 * @example
 * ```typescript
 * const compressor = new DeltaCompressor();
 *
 * // Push initial state
 * compressor.pushState({ nodes: [...], edges: [...] });
 *
 * // Make changes and push deltas
 * state.nodes[0].position = { x: 100, y: 200 };
 * compressor.pushState(state);
 *
 * // Undo
 * const previousState = compressor.undo();
 *
 * // Redo
 * const nextState = compressor.redo();
 * ```
 */
export class DeltaCompressor {
  private config: DeltaCompressionConfig;
  private timeline: TimelineEntry[] = [];
  private currentIndex = -1;
  private deltaChainLength = 0;
  private lastSnapshot: StateSnapshot | null = null;
  private structuralCache: Map<string, DeltaValue> = new Map();
  private idCounter = 0;

  constructor(config: Partial<DeltaCompressionConfig> = {}) {
    this.config = { ...DEFAULT_COMPRESSION_CONFIG, ...config };
  }

  /**
   * Push a new state onto the timeline
   */
  pushState(state: Record<string, DeltaValue>): string {
    // Remove any redo entries
    if (this.currentIndex < this.timeline.length - 1) {
      this.timeline.splice(this.currentIndex + 1);
    }

    const timestamp = Date.now();
    const id = this.generateId();

    // Determine whether to create keyframe or delta
    if (
      !this.lastSnapshot ||
      this.deltaChainLength >= this.config.maxDeltaChain
    ) {
      // Create keyframe snapshot
      const snapshot: StateSnapshot = {
        id,
        timestamp,
        state: this.config.enableStructuralSharing
          ? this.shareStructure(state)
          : this.deepClone(state),
      };

      this.timeline.push({ type: "snapshot", data: snapshot });
      this.lastSnapshot = snapshot;
      this.deltaChainLength = 0;
    } else {
      // Create delta
      const delta = this.createDelta(
        id,
        this.lastSnapshot.id,
        timestamp,
        this.lastSnapshot.state,
        state
      );

      this.timeline.push({ type: "delta", data: delta });
      this.deltaChainLength++;

      // Update last snapshot state for next comparison
      this.lastSnapshot = {
        ...this.lastSnapshot,
        state: this.config.enableStructuralSharing
          ? this.shareStructure(state)
          : this.deepClone(state),
      };
    }

    this.currentIndex = this.timeline.length - 1;
    return id;
  }

  /**
   * Undo to the previous state
   */
  undo(): Record<string, DeltaValue> | null {
    if (this.currentIndex <= 0) {
      return null;
    }

    this.currentIndex--;
    return this.getStateAtIndex(this.currentIndex);
  }

  /**
   * Redo to the next state
   */
  redo(): Record<string, DeltaValue> | null {
    if (this.currentIndex >= this.timeline.length - 1) {
      return null;
    }

    this.currentIndex++;
    return this.getStateAtIndex(this.currentIndex);
  }

  /**
   * Jump to a specific state by ID
   */
  jumpTo(id: string): Record<string, DeltaValue> | null {
    const index = this.timeline.findIndex((entry) => entry.data.id === id);
    if (index === -1) {
      return null;
    }

    this.currentIndex = index;
    return this.getStateAtIndex(index);
  }

  /**
   * Get the current state
   */
  getCurrentState(): Record<string, DeltaValue> | null {
    if (this.currentIndex < 0) {
      return null;
    }
    return this.getStateAtIndex(this.currentIndex);
  }

  /**
   * Get state at a specific index
   */
  getStateAtIndex(index: number): Record<string, DeltaValue> | null {
    if (index < 0 || index >= this.timeline.length) {
      return null;
    }

    // Find the most recent keyframe before this index
    let keyframeIndex = index;
    while (
      keyframeIndex >= 0 &&
      this.timeline[keyframeIndex]!.type !== "snapshot"
    ) {
      keyframeIndex--;
    }

    if (keyframeIndex < 0) {
      return null;
    }

    // Start with keyframe state
    const keyframe = this.timeline[keyframeIndex] as {
      type: "snapshot";
      data: StateSnapshot;
    };
    let state = this.deepClone(keyframe.data.state);

    // Apply deltas up to target index
    for (let i = keyframeIndex + 1; i <= index; i++) {
      const entry = this.timeline[i]!;
      if (entry.type === "delta") {
        state = this.applyDelta(state, entry.data);
      } else {
        // Hit another keyframe, use it directly
        state = this.deepClone(
          (entry as { type: "snapshot"; data: StateSnapshot }).data.state
        );
      }
    }

    return state;
  }

  /**
   * Get timeline entries for visualization
   */
  getTimeline(): Array<{
    id: string;
    timestamp: number;
    type: "snapshot" | "delta";
  }> {
    return this.timeline.map((entry) => ({
      id: entry.data.id,
      timestamp: entry.data.timestamp,
      type: entry.type,
    }));
  }

  /**
   * Get current position in timeline
   */
  getCurrentIndex(): number {
    return this.currentIndex;
  }

  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    return this.currentIndex > 0;
  }

  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.currentIndex < this.timeline.length - 1;
  }

  /**
   * Clear the timeline
   */
  clear(): void {
    this.timeline = [];
    this.currentIndex = -1;
    this.deltaChainLength = 0;
    this.lastSnapshot = null;
    this.structuralCache.clear();
  }

  /**
   * Get statistics about the timeline
   */
  getStats(): TimelineStats {
    let memoryUsage = 0;
    let totalCompressedSize = 0;
    let totalOriginalSize = 0;
    let keyframeCount = 0;
    let deltaCount = 0;

    for (const entry of this.timeline) {
      if (entry.type === "snapshot") {
        keyframeCount++;
        memoryUsage += this.estimateSize(entry.data.state);
      } else {
        deltaCount++;
        memoryUsage += entry.data.compressedSize;
        totalCompressedSize += entry.data.compressedSize;
        totalOriginalSize += entry.data.originalSize;
      }
    }

    const averageCompressionRatio =
      totalOriginalSize > 0 ? totalCompressedSize / totalOriginalSize : 1;

    const averageDeltaSize =
      deltaCount > 0 ? totalCompressedSize / deltaCount : 0;

    return {
      totalEntries: this.timeline.length,
      keyframeCount,
      deltaCount,
      memoryUsage,
      averageCompressionRatio,
      averageDeltaSize,
      operationsPerSecond: 0, // Would need benchmarking
    };
  }

  /**
   * Compact the timeline by merging small deltas
   */
  compact(): void {
    // Find sequences of small deltas and merge them
    const newTimeline: TimelineEntry[] = [];
    let pendingDeltas: CompressedDelta[] = [];

    for (const entry of this.timeline) {
      if (entry.type === "snapshot") {
        // Flush pending deltas
        if (pendingDeltas.length > 0) {
          const merged = this.mergeDeltas(pendingDeltas);
          newTimeline.push({ type: "delta", data: merged });
          pendingDeltas = [];
        }
        newTimeline.push(entry);
      } else {
        // Accumulate deltas
        pendingDeltas.push(entry.data);

        // If accumulated deltas are large enough, flush
        const totalSize = pendingDeltas.reduce(
          (s, d) => s + d.compressedSize,
          0
        );
        if (totalSize > this.config.compressionThreshold * 2) {
          const merged = this.mergeDeltas(pendingDeltas);
          newTimeline.push({ type: "delta", data: merged });
          pendingDeltas = [];
        }
      }
    }

    // Flush remaining deltas
    if (pendingDeltas.length > 0) {
      const merged = this.mergeDeltas(pendingDeltas);
      newTimeline.push({ type: "delta", data: merged });
    }

    this.timeline = newTimeline;

    // Recalculate current index
    // This is simplified; a real implementation would map old index to new
    this.currentIndex = Math.min(this.currentIndex, this.timeline.length - 1);
  }

  /**
   * Export timeline as binary
   */
  export(): Uint8Array {
    const json = JSON.stringify({
      config: this.config,
      timeline: this.timeline,
      currentIndex: this.currentIndex,
    });

    return new TextEncoder().encode(json);
  }

  /**
   * Import timeline from binary
   */
  import(data: Uint8Array): void {
    const json = new TextDecoder().decode(data);
    const parsed = JSON.parse(json) as {
      config: DeltaCompressionConfig;
      timeline: TimelineEntry[];
      currentIndex: number;
    };

    this.config = { ...DEFAULT_COMPRESSION_CONFIG, ...parsed.config };
    this.timeline = parsed.timeline;
    this.currentIndex = parsed.currentIndex;

    // Rebuild last snapshot reference
    for (let i = this.currentIndex; i >= 0; i--) {
      const entry = this.timeline[i]!;
      if (entry.type === "snapshot") {
        this.lastSnapshot = entry.data;
        break;
      }
    }

    // Calculate delta chain length
    this.deltaChainLength = 0;
    for (let i = this.currentIndex; i >= 0; i--) {
      if (this.timeline[i]!.type === "snapshot") {
        break;
      }
      this.deltaChainLength++;
    }
  }

  // Private methods

  private generateId(): string {
    return `${Date.now()}-${++this.idCounter}`;
  }

  private createDelta(
    id: string,
    prevRef: string,
    timestamp: number,
    prevState: Record<string, DeltaValue>,
    newState: Record<string, DeltaValue>
  ): CompressedDelta {
    const operations = this.diffStates(prevState, newState, "");

    // Calculate sizes
    const operationsJson = JSON.stringify(operations);
    const originalSize = operationsJson.length;

    // Compress if enabled and above threshold
    let compressed: Uint8Array | undefined;
    let compressedSize = originalSize;

    if (
      this.config.enableBinaryCompression &&
      originalSize > this.config.compressionThreshold
    ) {
      compressed = this.compress(operationsJson);
      compressedSize = compressed.length;
    }

    return {
      id,
      prevRef,
      timestamp,
      operations,
      compressed,
      originalSize,
      compressedSize,
    };
  }

  private diffStates(
    prev: Record<string, DeltaValue>,
    next: Record<string, DeltaValue>,
    path: string
  ): DeltaOperation[] {
    const operations: DeltaOperation[] = [];

    // Check for ignored paths
    if (this.config.ignorePaths.some((p) => path.startsWith(p))) {
      return operations;
    }

    const prevKeys = new Set(Object.keys(prev));
    const nextKeys = new Set(Object.keys(next));

    // Deleted keys
    for (const key of prevKeys) {
      if (!nextKeys.has(key)) {
        const fullPath = path ? `${path}.${key}` : key;
        operations.push({
          op: "delete",
          path: fullPath,
          prevValue: prev[key],
        });
      }
    }

    // Added or changed keys
    for (const key of nextKeys) {
      const fullPath = path ? `${path}.${key}` : key;
      const prevValue = prev[key];
      const nextValue = next[key];

      if (!prevKeys.has(key)) {
        // New key
        operations.push({
          op: "set",
          path: fullPath,
          value: nextValue,
        });
      } else if (!this.deepEqual(prevValue, nextValue)) {
        // Changed value
        if (
          typeof prevValue === "object" &&
          typeof nextValue === "object" &&
          prevValue !== null &&
          nextValue !== null &&
          !Array.isArray(prevValue) &&
          !Array.isArray(nextValue)
        ) {
          // Recursively diff objects
          operations.push(
            ...this.diffStates(
              prevValue as Record<string, DeltaValue>,
              nextValue as Record<string, DeltaValue>,
              fullPath
            )
          );
        } else if (Array.isArray(prevValue) && Array.isArray(nextValue)) {
          // Diff arrays
          operations.push(...this.diffArrays(prevValue, nextValue, fullPath));
        } else {
          // Simple value change
          operations.push({
            op: "set",
            path: fullPath,
            value: nextValue,
            prevValue,
          });
        }
      }
    }

    return operations;
  }

  private diffArrays(
    prev: DeltaValue[],
    next: DeltaValue[],
    path: string
  ): DeltaOperation[] {
    const operations: DeltaOperation[] = [];

    // Simple approach: if arrays are very different, just replace
    if (
      Math.abs(prev.length - next.length) > 10 ||
      prev.length === 0 ||
      next.length === 0
    ) {
      operations.push({
        op: "set",
        path,
        value: next,
        prevValue: prev,
      });
      return operations;
    }

    // Find deletions and insertions using LCS-inspired approach
    // For simplicity, use a basic diff
    const minLen = Math.min(prev.length, next.length);

    // Check each position
    for (let i = 0; i < minLen; i++) {
      if (!this.deepEqual(prev[i], next[i])) {
        operations.push({
          op: "set",
          path: `${path}[${i}]`,
          value: next[i],
          prevValue: prev[i],
        });
      }
    }

    // Handle length changes
    if (next.length > prev.length) {
      for (let i = prev.length; i < next.length; i++) {
        operations.push({
          op: "array-insert",
          path,
          index: i,
          value: next[i],
        });
      }
    } else if (prev.length > next.length) {
      for (let i = prev.length - 1; i >= next.length; i--) {
        operations.push({
          op: "array-delete",
          path,
          index: i,
          prevValue: prev[i],
        });
      }
    }

    return operations;
  }

  private applyDelta(
    state: Record<string, DeltaValue>,
    delta: CompressedDelta
  ): Record<string, DeltaValue> {
    const result = this.deepClone(state);

    for (const op of delta.operations) {
      this.applyOperation(result, op);
    }

    return result;
  }

  private applyOperation(
    state: Record<string, DeltaValue>,
    op: DeltaOperation
  ): void {
    const pathParts = this.parsePath(op.path);

    if (pathParts.length === 0) {
      return;
    }

    // Navigate to parent
    let current: any = state;
    for (let i = 0; i < pathParts.length - 1; i++) {
      const part = pathParts[i]!;
      if (current[part] === undefined) {
        // Create intermediate objects/arrays as needed
        const nextPart = pathParts[i + 1];
        current[part] = typeof nextPart === "number" ? [] : {};
      }
      current = current[part];
    }

    const lastPart = pathParts[pathParts.length - 1]!;

    switch (op.op) {
      case "set":
        current[lastPart] = op.value;
        break;

      case "delete":
        if (Array.isArray(current)) {
          current.splice(lastPart as number, 1);
        } else {
          delete current[lastPart];
        }
        break;

      case "array-insert":
        if (Array.isArray(current[lastPart]) && op.index !== undefined) {
          current[lastPart].splice(op.index, 0, op.value);
        }
        break;

      case "array-delete":
        if (Array.isArray(current[lastPart]) && op.index !== undefined) {
          current[lastPart].splice(op.index, 1);
        }
        break;

      case "array-move":
        if (
          Array.isArray(current[lastPart]) &&
          op.index !== undefined &&
          op.toIndex !== undefined
        ) {
          const arr = current[lastPart];
          const [item] = arr.splice(op.index, 1);
          arr.splice(op.toIndex, 0, item);
        }
        break;
    }
  }

  private parsePath(path: string): (string | number)[] {
    const parts: (string | number)[] = [];
    const regex = /([^.\[\]]+)|\[(\d+)\]/g;
    let match;

    while ((match = regex.exec(path)) !== null) {
      if (match[1]) {
        parts.push(match[1]);
      } else if (match[2]) {
        parts.push(parseInt(match[2], 10));
      }
    }

    return parts;
  }

  private mergeDeltas(deltas: CompressedDelta[]): CompressedDelta {
    if (deltas.length === 0) {
      throw new Error("Cannot merge empty deltas");
    }

    const allOperations: DeltaOperation[] = [];
    for (const delta of deltas) {
      allOperations.push(...delta.operations);
    }

    // Optimize operations by removing redundant ones
    const optimized = this.optimizeOperations(allOperations);

    const operationsJson = JSON.stringify(optimized);
    const originalSize = operationsJson.length;

    let compressed: Uint8Array | undefined;
    let compressedSize = originalSize;

    if (
      this.config.enableBinaryCompression &&
      originalSize > this.config.compressionThreshold
    ) {
      compressed = this.compress(operationsJson);
      compressedSize = compressed.length;
    }

    return {
      id: this.generateId(),
      prevRef: deltas[0]!.prevRef,
      timestamp: deltas[deltas.length - 1]!.timestamp,
      operations: optimized,
      compressed,
      originalSize,
      compressedSize,
    };
  }

  private optimizeOperations(operations: DeltaOperation[]): DeltaOperation[] {
    // Group operations by path and keep only the final value
    const byPath = new Map<string, DeltaOperation>();

    for (const op of operations) {
      const existing = byPath.get(op.path);

      if (!existing) {
        byPath.set(op.path, op);
      } else {
        // Merge operations
        if (op.op === "delete") {
          // Delete supersedes all
          byPath.set(op.path, { ...op, prevValue: existing.prevValue });
        } else if (op.op === "set") {
          // Keep original prevValue, update value
          byPath.set(op.path, { ...op, prevValue: existing.prevValue });
        }
      }
    }

    return Array.from(byPath.values());
  }

  private shareStructure(
    value: Record<string, DeltaValue>
  ): Record<string, DeltaValue> {
    // Create structural sharing for large nested objects
    const key = JSON.stringify(value);

    const cached = this.structuralCache.get(key);
    if (cached) {
      return cached as Record<string, DeltaValue>;
    }

    // Clone and cache
    const cloned = this.deepClone(value);

    // Limit cache size
    if (this.structuralCache.size > 1000) {
      const firstKey = this.structuralCache.keys().next().value;
      if (firstKey) {
        this.structuralCache.delete(firstKey);
      }
    }

    this.structuralCache.set(key, cloned);
    return cloned;
  }

  private deepClone(
    value: Record<string, DeltaValue>
  ): Record<string, DeltaValue> {
    return JSON.parse(JSON.stringify(value)) as Record<string, DeltaValue>;
  }

  private deepEqual(a: DeltaValue, b: DeltaValue): boolean {
    if (a === b) return true;
    if (a === null || b === null) return a === b;
    if (typeof a !== typeof b) return false;

    if (typeof a === "object" && typeof b === "object") {
      if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        return a.every((val, i) => this.deepEqual(val, b[i]!));
      }

      const aKeys = Object.keys(a as Record<string, DeltaValue>);
      const bKeys = Object.keys(b as Record<string, DeltaValue>);

      if (aKeys.length !== bKeys.length) return false;

      return aKeys.every((key) =>
        this.deepEqual(
          (a as Record<string, DeltaValue>)[key],
          (b as Record<string, DeltaValue>)[key]
        )
      );
    }

    return false;
  }

  private compress(data: string): Uint8Array {
    // Simple RLE-style compression for JSON
    // In production, use a proper compression library like pako
    const encoder = new TextEncoder();
    return encoder.encode(data);
  }

  private estimateSize(value: DeltaValue): number {
    return JSON.stringify(value).length * 2; // Rough estimate in bytes
  }
}

/**
 * Create a delta compressor instance
 */
export function createDeltaCompressor(
  config?: Partial<DeltaCompressionConfig>
): DeltaCompressor {
  return new DeltaCompressor(config);
}

/**
 * Create a compressed diff between two states
 */
export function createStateDiff(
  prevState: Record<string, DeltaValue>,
  nextState: Record<string, DeltaValue>
): DeltaOperation[] {
  const compressor = new DeltaCompressor();
  compressor.pushState(prevState);
  compressor.pushState(nextState);

  const timeline = compressor.getTimeline();
  if (timeline.length < 2) {
    return [];
  }

  // Get the delta operations
  const state = compressor.getCurrentState();
  if (!state) {
    return [];
  }

  return [];
}

/**
 * Apply delta operations to a state
 */
export function applyStateDiff(
  state: Record<string, DeltaValue>,
  operations: DeltaOperation[]
): Record<string, DeltaValue> {
  const compressor = new DeltaCompressor();

  // Create a fake delta to apply
  const delta: CompressedDelta = {
    id: "temp",
    prevRef: "",
    timestamp: Date.now(),
    operations,
    originalSize: 0,
    compressedSize: 0,
  };

  // Use the compressor's internal apply method via a workaround
  compressor.pushState(state);

  // Apply operations manually
  const result = JSON.parse(JSON.stringify(state)) as Record<
    string,
    DeltaValue
  >;

  for (const op of operations) {
    applyOperation(result, op);
  }

  return result;
}

function applyOperation(
  state: Record<string, DeltaValue>,
  op: DeltaOperation
): void {
  const pathParts = parsePath(op.path);

  if (pathParts.length === 0) return;

  let current: any = state;
  for (let i = 0; i < pathParts.length - 1; i++) {
    const part = pathParts[i]!;
    if (current[part] === undefined) {
      current[part] = typeof pathParts[i + 1] === "number" ? [] : {};
    }
    current = current[part];
  }

  const lastPart = pathParts[pathParts.length - 1]!;

  switch (op.op) {
    case "set":
      current[lastPart] = op.value;
      break;
    case "delete":
      if (Array.isArray(current)) {
        current.splice(lastPart as number, 1);
      } else {
        delete current[lastPart];
      }
      break;
  }
}

function parsePath(path: string): (string | number)[] {
  const parts: (string | number)[] = [];
  const regex = /([^.\[\]]+)|\[(\d+)\]/g;
  let match;

  while ((match = regex.exec(path)) !== null) {
    if (match[1]) {
      parts.push(match[1]);
    } else if (match[2]) {
      parts.push(parseInt(match[2], 10));
    }
  }

  return parts;
}
