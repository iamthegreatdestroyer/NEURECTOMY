/**
 * @fileoverview Memory Pool Allocator
 * @module @neurectomy/performance-engine/caching/memory-pool
 *
 * Agent Assignment: @CORE (Low-Level Systems) + @VELOCITY (Performance)
 *
 * Implements efficient memory pooling to reduce allocation overhead:
 * - Object pooling for frequently created objects
 * - Buffer pooling for I/O operations
 * - Slab allocator for fixed-size allocations
 * - Arena allocator for batch allocations
 *
 * @author NEURECTOMY Phase 5 - Performance Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";

// ============================================================================
// Memory Pool Types (@CORE)
// ============================================================================

/**
 * Pool statistics
 */
export interface PoolStats {
  totalAllocations: number;
  totalDeallocations: number;
  currentSize: number;
  maxSize: number;
  hitRate: number;
  fragmentationRatio: number;
}

/**
 * Pool configuration
 */
export interface PoolConfig {
  initialSize: number;
  maxSize: number;
  growthFactor: number;
  shrinkThreshold: number;
  resetOnRelease: boolean;
}

/**
 * Pooled object interface
 */
export interface Poolable {
  reset(): void;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_POOL_CONFIG: PoolConfig = {
  initialSize: 10,
  maxSize: 1000,
  growthFactor: 2,
  shrinkThreshold: 0.25,
  resetOnRelease: true,
};

// ============================================================================
// Object Pool (@CORE @VELOCITY)
// ============================================================================

/**
 * Generic object pool for reducing allocation overhead
 */
export class ObjectPool<T extends Poolable> extends EventEmitter {
  private config: PoolConfig;
  private factory: () => T;
  private pool: T[];
  private inUse: Set<T>;
  private stats: PoolStats;

  constructor(factory: () => T, config: Partial<PoolConfig> = {}) {
    super();
    this.config = { ...DEFAULT_POOL_CONFIG, ...config };
    this.factory = factory;
    this.pool = [];
    this.inUse = new Set();
    this.stats = {
      totalAllocations: 0,
      totalDeallocations: 0,
      currentSize: 0,
      maxSize: this.config.maxSize,
      hitRate: 0,
      fragmentationRatio: 0,
    };

    // Pre-allocate initial pool
    this.grow(this.config.initialSize);
  }

  /**
   * Acquire object from pool
   */
  acquire(): T {
    this.stats.totalAllocations++;
    let obj: T;

    if (this.pool.length > 0) {
      obj = this.pool.pop()!;
      this.emit("pool-hit", { poolSize: this.pool.length });
    } else {
      // Pool is empty, create new object
      if (this.inUse.size >= this.config.maxSize) {
        throw new Error("Pool exhausted: maximum size reached");
      }
      obj = this.factory();
      this.emit("pool-miss", { creating: true });
    }

    this.inUse.add(obj);
    this.updateStats();
    return obj;
  }

  /**
   * Release object back to pool
   */
  release(obj: T): void {
    if (!this.inUse.has(obj)) {
      throw new Error("Object was not acquired from this pool");
    }

    this.inUse.delete(obj);
    this.stats.totalDeallocations++;

    // Reset object if configured
    if (this.config.resetOnRelease) {
      obj.reset();
    }

    // Only return to pool if under max size
    if (this.pool.length < this.config.maxSize) {
      this.pool.push(obj);
    }

    this.updateStats();
    this.emit("release", { poolSize: this.pool.length });

    // Check if we should shrink
    this.checkShrink();
  }

  /**
   * Grow the pool by specified amount
   */
  grow(count: number): void {
    const toCreate = Math.min(
      count,
      this.config.maxSize - this.pool.length - this.inUse.size
    );

    for (let i = 0; i < toCreate; i++) {
      this.pool.push(this.factory());
    }

    this.updateStats();
    this.emit("grow", { added: toCreate, poolSize: this.pool.length });
  }

  /**
   * Shrink pool if utilization is low
   */
  private checkShrink(): void {
    const utilization = this.inUse.size / (this.pool.length + this.inUse.size);

    if (
      utilization < this.config.shrinkThreshold &&
      this.pool.length > this.config.initialSize
    ) {
      const toRemove = Math.floor(
        (this.pool.length - this.config.initialSize) / 2
      );
      this.pool.splice(0, toRemove);
      this.updateStats();
      this.emit("shrink", { removed: toRemove, poolSize: this.pool.length });
    }
  }

  /**
   * Update statistics
   */
  private updateStats(): void {
    const total = this.pool.length + this.inUse.size;
    this.stats.currentSize = total;
    this.stats.hitRate =
      this.stats.totalAllocations > 0
        ? 1 - (total - this.pool.length) / this.stats.totalAllocations
        : 0;
    this.stats.fragmentationRatio = total > 0 ? this.pool.length / total : 0;
  }

  /**
   * Get pool statistics
   */
  getStats(): PoolStats {
    return { ...this.stats };
  }

  /**
   * Clear the pool
   */
  clear(): void {
    this.pool = [];
    this.inUse.clear();
    this.updateStats();
    this.emit("clear");
  }

  /**
   * Get current pool size
   */
  get size(): number {
    return this.pool.length;
  }

  /**
   * Get number of objects in use
   */
  get activeCount(): number {
    return this.inUse.size;
  }
}

// ============================================================================
// Buffer Pool (@CORE @VELOCITY)
// ============================================================================

/**
 * Buffer pool configuration
 */
export interface BufferPoolConfig {
  smallBufferSize: number;
  mediumBufferSize: number;
  largeBufferSize: number;
  poolSizes: {
    small: number;
    medium: number;
    large: number;
  };
}

const DEFAULT_BUFFER_CONFIG: BufferPoolConfig = {
  smallBufferSize: 1024, // 1KB
  mediumBufferSize: 16384, // 16KB
  largeBufferSize: 65536, // 64KB
  poolSizes: {
    small: 100,
    medium: 50,
    large: 20,
  },
};

/**
 * Buffer pool for efficient I/O operations
 */
export class BufferPool extends EventEmitter {
  private config: BufferPoolConfig;
  private smallPool: Uint8Array[];
  private mediumPool: Uint8Array[];
  private largePool: Uint8Array[];
  private inUse: Map<Uint8Array, "small" | "medium" | "large">;
  private stats: {
    allocations: { small: number; medium: number; large: number };
    hits: { small: number; medium: number; large: number };
  };

  constructor(config: Partial<BufferPoolConfig> = {}) {
    super();
    this.config = { ...DEFAULT_BUFFER_CONFIG, ...config };
    this.smallPool = [];
    this.mediumPool = [];
    this.largePool = [];
    this.inUse = new Map();
    this.stats = {
      allocations: { small: 0, medium: 0, large: 0 },
      hits: { small: 0, medium: 0, large: 0 },
    };

    // Pre-allocate pools
    this.preallocate();
  }

  /**
   * Pre-allocate buffers
   */
  private preallocate(): void {
    for (let i = 0; i < this.config.poolSizes.small; i++) {
      this.smallPool.push(new Uint8Array(this.config.smallBufferSize));
    }
    for (let i = 0; i < this.config.poolSizes.medium; i++) {
      this.mediumPool.push(new Uint8Array(this.config.mediumBufferSize));
    }
    for (let i = 0; i < this.config.poolSizes.large; i++) {
      this.largePool.push(new Uint8Array(this.config.largeBufferSize));
    }
  }

  /**
   * Acquire buffer of appropriate size
   */
  acquire(minSize: number): Uint8Array {
    let buffer: Uint8Array | undefined;
    let poolType: "small" | "medium" | "large";

    if (minSize <= this.config.smallBufferSize) {
      poolType = "small";
      buffer = this.smallPool.pop();
      if (!buffer) {
        buffer = new Uint8Array(this.config.smallBufferSize);
      } else {
        this.stats.hits.small++;
      }
    } else if (minSize <= this.config.mediumBufferSize) {
      poolType = "medium";
      buffer = this.mediumPool.pop();
      if (!buffer) {
        buffer = new Uint8Array(this.config.mediumBufferSize);
      } else {
        this.stats.hits.medium++;
      }
    } else if (minSize <= this.config.largeBufferSize) {
      poolType = "large";
      buffer = this.largePool.pop();
      if (!buffer) {
        buffer = new Uint8Array(this.config.largeBufferSize);
      } else {
        this.stats.hits.large++;
      }
    } else {
      // Too large for pool, allocate directly
      return new Uint8Array(minSize);
    }

    this.stats.allocations[poolType]++;
    this.inUse.set(buffer, poolType);
    this.emit("acquire", { size: buffer.length, poolType });

    return buffer;
  }

  /**
   * Release buffer back to pool
   */
  release(buffer: Uint8Array): void {
    const poolType = this.inUse.get(buffer);

    if (!poolType) {
      // Not from our pool, ignore
      return;
    }

    this.inUse.delete(buffer);

    // Clear buffer content
    buffer.fill(0);

    // Return to appropriate pool
    switch (poolType) {
      case "small":
        if (this.smallPool.length < this.config.poolSizes.small * 2) {
          this.smallPool.push(buffer);
        }
        break;
      case "medium":
        if (this.mediumPool.length < this.config.poolSizes.medium * 2) {
          this.mediumPool.push(buffer);
        }
        break;
      case "large":
        if (this.largePool.length < this.config.poolSizes.large * 2) {
          this.largePool.push(buffer);
        }
        break;
    }

    this.emit("release", { poolType });
  }

  /**
   * Get pool statistics
   */
  getStats(): {
    poolSizes: { small: number; medium: number; large: number };
    inUse: number;
    allocations: { small: number; medium: number; large: number };
    hitRates: { small: number; medium: number; large: number };
  } {
    return {
      poolSizes: {
        small: this.smallPool.length,
        medium: this.mediumPool.length,
        large: this.largePool.length,
      },
      inUse: this.inUse.size,
      allocations: { ...this.stats.allocations },
      hitRates: {
        small:
          this.stats.allocations.small > 0
            ? this.stats.hits.small / this.stats.allocations.small
            : 0,
        medium:
          this.stats.allocations.medium > 0
            ? this.stats.hits.medium / this.stats.allocations.medium
            : 0,
        large:
          this.stats.allocations.large > 0
            ? this.stats.hits.large / this.stats.allocations.large
            : 0,
      },
    };
  }

  /**
   * Clear all pools
   */
  clear(): void {
    this.smallPool = [];
    this.mediumPool = [];
    this.largePool = [];
    this.inUse.clear();
    this.emit("clear");
  }
}

// ============================================================================
// Slab Allocator (@CORE)
// ============================================================================

/**
 * Slab allocator for fixed-size objects
 */
export class SlabAllocator<T> extends EventEmitter {
  private slabSize: number;
  private objectSize: number;
  private slabs: Array<{
    data: T[];
    free: number[];
    used: Set<number>;
  }>;
  private objectToSlab: Map<T, number>;

  constructor(objectSize: number, slabSize: number = 64) {
    super();
    this.objectSize = objectSize;
    this.slabSize = slabSize;
    this.slabs = [];
    this.objectToSlab = new Map();

    // Create initial slab
    this.createSlab();
  }

  /**
   * Create a new slab
   */
  private createSlab(): number {
    const slabIndex = this.slabs.length;
    const slab = {
      data: new Array(this.slabSize),
      free: Array.from({ length: this.slabSize }, (_, i) => i),
      used: new Set<number>(),
    };
    this.slabs.push(slab);
    this.emit("slab-created", { slabIndex, slabSize: this.slabSize });
    return slabIndex;
  }

  /**
   * Allocate object from slab
   */
  allocate(factory: () => T): T {
    // Find slab with free slot
    let slabIndex = this.slabs.findIndex((s) => s.free.length > 0);

    if (slabIndex === -1) {
      slabIndex = this.createSlab();
    }

    const slab = this.slabs[slabIndex];
    const slot = slab.free.pop()!;

    // Create object if not exists
    if (!slab.data[slot]) {
      slab.data[slot] = factory();
    }

    slab.used.add(slot);
    this.objectToSlab.set(slab.data[slot], slabIndex);
    this.emit("allocate", { slabIndex, slot });

    return slab.data[slot];
  }

  /**
   * Free object back to slab
   */
  free(obj: T): void {
    const slabIndex = this.objectToSlab.get(obj);

    if (slabIndex === undefined) {
      throw new Error("Object not from this allocator");
    }

    const slab = this.slabs[slabIndex];
    const slot = slab.data.indexOf(obj);

    if (slot !== -1 && slab.used.has(slot)) {
      slab.used.delete(slot);
      slab.free.push(slot);
      this.objectToSlab.delete(obj);
      this.emit("free", { slabIndex, slot });
    }
  }

  /**
   * Get allocator statistics
   */
  getStats(): {
    slabCount: number;
    totalSlots: number;
    usedSlots: number;
    fragmentation: number;
  } {
    let usedSlots = 0;
    const totalSlots = this.slabs.length * this.slabSize;

    for (const slab of this.slabs) {
      usedSlots += slab.used.size;
    }

    return {
      slabCount: this.slabs.length,
      totalSlots,
      usedSlots,
      fragmentation: totalSlots > 0 ? 1 - usedSlots / totalSlots : 0,
    };
  }
}

// ============================================================================
// Arena Allocator (@CORE)
// ============================================================================

/**
 * Arena allocator for batch allocations with fast cleanup
 */
export class ArenaAllocator<T> extends EventEmitter {
  private arenas: T[][];
  private currentArena: T[];
  private arenaSize: number;
  private position: number;

  constructor(arenaSize: number = 1000) {
    super();
    this.arenaSize = arenaSize;
    this.arenas = [];
    this.currentArena = [];
    this.position = 0;

    this.createArena();
  }

  /**
   * Create new arena
   */
  private createArena(): void {
    this.currentArena = new Array(this.arenaSize);
    this.arenas.push(this.currentArena);
    this.position = 0;
    this.emit("arena-created", { arenaCount: this.arenas.length });
  }

  /**
   * Allocate in current arena
   */
  allocate(factory: () => T): T {
    if (this.position >= this.arenaSize) {
      this.createArena();
    }

    const obj = factory();
    this.currentArena[this.position] = obj;
    this.position++;

    return obj;
  }

  /**
   * Reset all arenas (fast bulk deallocation)
   */
  reset(): void {
    // Keep only one arena
    if (this.arenas.length > 1) {
      this.arenas = [this.arenas[0]];
    }
    this.currentArena = this.arenas[0];
    this.position = 0;
    this.emit("reset");
  }

  /**
   * Get allocator statistics
   */
  getStats(): {
    arenaCount: number;
    currentPosition: number;
    totalCapacity: number;
    utilization: number;
  } {
    const totalCapacity = this.arenas.length * this.arenaSize;
    const used = (this.arenas.length - 1) * this.arenaSize + this.position;

    return {
      arenaCount: this.arenas.length,
      currentPosition: this.position,
      totalCapacity,
      utilization: totalCapacity > 0 ? used / totalCapacity : 0,
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create an object pool
 */
export function createObjectPool<T extends Poolable>(
  factory: () => T,
  config?: Partial<PoolConfig>
): ObjectPool<T> {
  return new ObjectPool(factory, config);
}

/**
 * Create a buffer pool
 */
export function createBufferPool(
  config?: Partial<BufferPoolConfig>
): BufferPool {
  return new BufferPool(config);
}

/**
 * Create a slab allocator
 */
export function createSlabAllocator<T>(
  objectSize: number,
  slabSize?: number
): SlabAllocator<T> {
  return new SlabAllocator(objectSize, slabSize);
}

/**
 * Create an arena allocator
 */
export function createArenaAllocator<T>(arenaSize?: number): ArenaAllocator<T> {
  return new ArenaAllocator(arenaSize);
}

// ============================================================================
// Exports
// ============================================================================

export { DEFAULT_POOL_CONFIG, DEFAULT_BUFFER_CONFIG };
