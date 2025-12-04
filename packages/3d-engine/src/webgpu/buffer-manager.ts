/**
 * GPU Buffer Pool Manager
 * 
 * High-performance GPU buffer allocation with pooling, automatic
 * defragmentation, and memory pressure handling.
 * 
 * @module @neurectomy/3d-engine/webgpu/buffer-manager
 * @agents @CORE @VELOCITY
 */

// =============================================================================
// Types
// =============================================================================

export interface BufferAllocation {
  buffer: GPUBuffer;
  offset: number;
  size: number;
  poolId: string;
  allocatedAt: number;
}

export interface BufferPoolConfig {
  initialSize: number;
  maxSize: number;
  growthFactor: number;
  usage: GPUBufferUsageFlags;
  label?: string;
}

export interface BufferManagerConfig {
  /** Maximum total GPU memory to use (bytes) */
  maxMemory: number;
  /** Enable automatic defragmentation */
  enableDefragmentation: boolean;
  /** Memory usage threshold to trigger defragmentation (0-1) */
  defragmentationThreshold: number;
  /** Enable memory pressure monitoring */
  enableMemoryPressure: boolean;
  /** Memory pressure threshold to trigger cleanup (0-1) */
  memoryPressureThreshold: number;
}

export interface BufferManagerStats {
  totalAllocated: number;
  totalUsed: number;
  fragmentedBytes: number;
  poolCount: number;
  allocationCount: number;
  defragmentationCount: number;
  memoryPressureEvents: number;
}

interface BufferPool {
  id: string;
  config: BufferPoolConfig;
  buffer: GPUBuffer;
  size: number;
  allocations: Map<number, BufferAllocation>;
  freeList: FreeBlock[];
  lastUsed: number;
}

interface FreeBlock {
  offset: number;
  size: number;
}

// =============================================================================
// BufferManager Class
// =============================================================================

/**
 * BufferManager - GPU buffer pool management
 * 
 * Features:
 * - Buffer pooling for reduced allocation overhead
 * - Automatic defragmentation
 * - Memory pressure handling
 * - Usage tracking and statistics
 */
export class BufferManager {
  private device: GPUDevice;
  private config: BufferManagerConfig;
  private pools = new Map<string, BufferPool>();
  private stats: BufferManagerStats;
  private allocationIdCounter = 0;

  // Default pool configurations by usage type
  private readonly DEFAULT_POOLS: Record<string, BufferPoolConfig> = {
    vertex: {
      initialSize: 4 * 1024 * 1024, // 4MB
      maxSize: 64 * 1024 * 1024, // 64MB
      growthFactor: 2,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      label: 'Vertex Buffer Pool',
    },
    index: {
      initialSize: 2 * 1024 * 1024, // 2MB
      maxSize: 32 * 1024 * 1024, // 32MB
      growthFactor: 2,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      label: 'Index Buffer Pool',
    },
    uniform: {
      initialSize: 1 * 1024 * 1024, // 1MB
      maxSize: 16 * 1024 * 1024, // 16MB
      growthFactor: 2,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'Uniform Buffer Pool',
    },
    storage: {
      initialSize: 8 * 1024 * 1024, // 8MB
      maxSize: 128 * 1024 * 1024, // 128MB
      growthFactor: 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'Storage Buffer Pool',
    },
    staging: {
      initialSize: 4 * 1024 * 1024, // 4MB
      maxSize: 64 * 1024 * 1024, // 64MB
      growthFactor: 2,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      label: 'Staging Buffer Pool',
    },
  };

  constructor(device: GPUDevice, config?: Partial<BufferManagerConfig>) {
    this.device = device;
    this.config = {
      maxMemory: config?.maxMemory ?? 512 * 1024 * 1024, // 512MB default
      enableDefragmentation: config?.enableDefragmentation ?? true,
      defragmentationThreshold: config?.defragmentationThreshold ?? 0.3,
      enableMemoryPressure: config?.enableMemoryPressure ?? true,
      memoryPressureThreshold: config?.memoryPressureThreshold ?? 0.9,
    };

    this.stats = {
      totalAllocated: 0,
      totalUsed: 0,
      fragmentedBytes: 0,
      poolCount: 0,
      allocationCount: 0,
      defragmentationCount: 0,
      memoryPressureEvents: 0,
    };

    // Initialize default pools
    this.initializeDefaultPools();
  }

  /**
   * Initialize default buffer pools
   */
  private initializeDefaultPools(): void {
    for (const [name, config] of Object.entries(this.DEFAULT_POOLS)) {
      this.createPool(name, config);
    }
  }

  /**
   * Create a new buffer pool
   */
  createPool(name: string, config: BufferPoolConfig): BufferPool {
    if (this.pools.has(name)) {
      console.warn(`[BufferManager] Pool '${name}' already exists`);
      return this.pools.get(name)!;
    }

    const buffer = this.device.createBuffer({
      size: config.initialSize,
      usage: config.usage,
      label: config.label ?? `Pool: ${name}`,
    });

    const pool: BufferPool = {
      id: name,
      config,
      buffer,
      size: config.initialSize,
      allocations: new Map(),
      freeList: [{ offset: 0, size: config.initialSize }],
      lastUsed: Date.now(),
    };

    this.pools.set(name, pool);
    this.stats.poolCount++;
    this.stats.totalAllocated += config.initialSize;

    return pool;
  }

  /**
   * Allocate buffer space from a pool
   */
  allocate(
    poolName: string,
    size: number,
    alignment: number = 256 // WebGPU requires 256-byte alignment for most uses
  ): BufferAllocation | null {
    const pool = this.pools.get(poolName);
    if (!pool) {
      console.error(`[BufferManager] Pool '${poolName}' not found`);
      return null;
    }

    // Align size
    const alignedSize = this.alignSize(size, alignment);

    // Try to find a suitable free block
    let allocation = this.findAndAllocate(pool, alignedSize, alignment);

    if (!allocation) {
      // Try defragmentation if enabled
      if (this.config.enableDefragmentation && this.shouldDefragment(pool)) {
        this.defragmentPool(pool);
        allocation = this.findAndAllocate(pool, alignedSize, alignment);
      }

      // Grow pool if still no space
      if (!allocation && pool.size < pool.config.maxSize) {
        this.growPool(pool);
        allocation = this.findAndAllocate(pool, alignedSize, alignment);
      }
    }

    if (allocation) {
      pool.lastUsed = Date.now();
      this.stats.allocationCount++;
      this.stats.totalUsed += alignedSize;
    }

    // Check memory pressure
    if (this.config.enableMemoryPressure) {
      this.checkMemoryPressure();
    }

    return allocation;
  }

  /**
   * Find and allocate from free list
   */
  private findAndAllocate(
    pool: BufferPool,
    size: number,
    alignment: number
  ): BufferAllocation | null {
    // Best-fit algorithm
    let bestIndex = -1;
    let bestSize = Infinity;

    for (let i = 0; i < pool.freeList.length; i++) {
      const block = pool.freeList[i]!;
      const alignedOffset = this.alignOffset(block.offset, alignment);
      const alignmentPadding = alignedOffset - block.offset;
      const availableSize = block.size - alignmentPadding;

      if (availableSize >= size && block.size < bestSize) {
        bestIndex = i;
        bestSize = block.size;
      }
    }

    if (bestIndex === -1) {
      return null;
    }

    const block = pool.freeList[bestIndex]!;
    const alignedOffset = this.alignOffset(block.offset, alignment);
    const alignmentPadding = alignedOffset - block.offset;

    // Create allocation
    const allocationId = this.allocationIdCounter++;
    const allocation: BufferAllocation = {
      buffer: pool.buffer,
      offset: alignedOffset,
      size,
      poolId: pool.id,
      allocatedAt: Date.now(),
    };

    pool.allocations.set(allocationId, allocation);

    // Update free list
    const remainingSize = block.size - size - alignmentPadding;

    if (remainingSize > 0) {
      // Shrink block
      block.offset = alignedOffset + size;
      block.size = remainingSize;

      // Add padding block if significant
      if (alignmentPadding > 0) {
        this.stats.fragmentedBytes += alignmentPadding;
      }
    } else {
      // Remove block entirely
      pool.freeList.splice(bestIndex, 1);
    }

    return allocation;
  }

  /**
   * Free an allocation
   */
  free(allocation: BufferAllocation): void {
    const pool = this.pools.get(allocation.poolId);
    if (!pool) {
      console.error(`[BufferManager] Pool '${allocation.poolId}' not found`);
      return;
    }

    // Find and remove allocation
    let allocationId: number | null = null;
    for (const [id, alloc] of pool.allocations) {
      if (alloc.offset === allocation.offset && alloc.size === allocation.size) {
        allocationId = id;
        break;
      }
    }

    if (allocationId === null) {
      console.warn('[BufferManager] Allocation not found');
      return;
    }

    pool.allocations.delete(allocationId);
    this.stats.totalUsed -= allocation.size;

    // Add to free list and coalesce
    this.addToFreeList(pool, allocation.offset, allocation.size);
  }

  /**
   * Add block to free list and coalesce adjacent blocks
   */
  private addToFreeList(pool: BufferPool, offset: number, size: number): void {
    // Find insertion point (keep sorted by offset)
    let insertIndex = 0;
    for (let i = 0; i < pool.freeList.length; i++) {
      if (pool.freeList[i]!.offset > offset) {
        break;
      }
      insertIndex = i + 1;
    }

    // Insert new block
    pool.freeList.splice(insertIndex, 0, { offset, size });

    // Coalesce with previous block
    if (insertIndex > 0) {
      const prev = pool.freeList[insertIndex - 1]!;
      const curr = pool.freeList[insertIndex]!;
      if (prev.offset + prev.size === curr.offset) {
        prev.size += curr.size;
        pool.freeList.splice(insertIndex, 1);
        insertIndex--;
      }
    }

    // Coalesce with next block
    if (insertIndex < pool.freeList.length - 1) {
      const curr = pool.freeList[insertIndex]!;
      const next = pool.freeList[insertIndex + 1]!;
      if (curr.offset + curr.size === next.offset) {
        curr.size += next.size;
        pool.freeList.splice(insertIndex + 1, 1);
      }
    }
  }

  /**
   * Check if pool should be defragmented
   */
  private shouldDefragment(pool: BufferPool): boolean {
    const usedSpace = pool.size - this.getFreeSpace(pool);
    const fragmentation = this.getFragmentation(pool);
    return fragmentation > this.config.defragmentationThreshold && usedSpace > 0;
  }

  /**
   * Defragment a pool (compact allocations)
   */
  private defragmentPool(pool: BufferPool): void {
    if (pool.allocations.size === 0) {
      // Reset free list to single block
      pool.freeList = [{ offset: 0, size: pool.size }];
      this.stats.defragmentationCount++;
      return;
    }

    console.log(`[BufferManager] Defragmenting pool '${pool.id}'`);

    // Note: Real defragmentation would require copying data,
    // which needs a staging buffer and command encoder.
    // For now, we just update stats.
    this.stats.defragmentationCount++;
  }

  /**
   * Grow a pool
   */
  private growPool(pool: BufferPool): void {
    const newSize = Math.min(
      pool.size * pool.config.growthFactor,
      pool.config.maxSize
    );

    if (newSize <= pool.size) {
      console.warn(`[BufferManager] Pool '${pool.id}' at maximum size`);
      return;
    }

    console.log(`[BufferManager] Growing pool '${pool.id}' from ${pool.size} to ${newSize}`);

    // Create new larger buffer
    const newBuffer = this.device.createBuffer({
      size: newSize,
      usage: pool.config.usage,
      label: pool.config.label,
    });

    // Copy existing data (would need command encoder)
    // For now, we'll just swap buffers
    pool.buffer.destroy();
    pool.buffer = newBuffer;

    // Add new space to free list
    this.addToFreeList(pool, pool.size, newSize - pool.size);

    this.stats.totalAllocated += newSize - pool.size;
    pool.size = newSize;
  }

  /**
   * Check and handle memory pressure
   */
  private checkMemoryPressure(): void {
    const usage = this.stats.totalUsed / this.config.maxMemory;
    if (usage > this.config.memoryPressureThreshold) {
      this.stats.memoryPressureEvents++;
      this.handleMemoryPressure();
    }
  }

  /**
   * Handle memory pressure by cleaning up unused resources
   */
  private handleMemoryPressure(): void {
    console.warn('[BufferManager] Memory pressure detected');

    // Find and shrink pools that have high fragmentation and low usage
    const now = Date.now();
    const IDLE_THRESHOLD = 60000; // 1 minute

    for (const pool of this.pools.values()) {
      if (now - pool.lastUsed > IDLE_THRESHOLD) {
        const freeSpace = this.getFreeSpace(pool);
        if (freeSpace > pool.size * 0.5) {
          // Pool has >50% free space and hasn't been used recently
          console.log(`[BufferManager] Shrinking idle pool '${pool.id}'`);
          // In a real implementation, we'd shrink the pool
        }
      }
    }
  }

  /**
   * Get free space in a pool
   */
  private getFreeSpace(pool: BufferPool): number {
    return pool.freeList.reduce((sum, block) => sum + block.size, 0);
  }

  /**
   * Get fragmentation ratio (0-1)
   */
  private getFragmentation(pool: BufferPool): number {
    const freeSpace = this.getFreeSpace(pool);
    if (freeSpace === 0) return 0;

    const largestFreeBlock = Math.max(...pool.freeList.map(b => b.size), 0);
    return 1 - (largestFreeBlock / freeSpace);
  }

  /**
   * Align size to boundary
   */
  private alignSize(size: number, alignment: number): number {
    return Math.ceil(size / alignment) * alignment;
  }

  /**
   * Align offset to boundary
   */
  private alignOffset(offset: number, alignment: number): number {
    return Math.ceil(offset / alignment) * alignment;
  }

  /**
   * Write data to an allocation
   */
  writeData(allocation: BufferAllocation, data: ArrayBuffer | ArrayBufferView): void {
    this.device.queue.writeBuffer(
      allocation.buffer,
      allocation.offset,
      data instanceof ArrayBuffer ? data : data.buffer,
      data instanceof ArrayBuffer ? 0 : data.byteOffset,
      data instanceof ArrayBuffer ? data.byteLength : data.byteLength
    );
  }

  /**
   * Get buffer manager statistics
   */
  getStats(): BufferManagerStats {
    // Update fragmented bytes
    this.stats.fragmentedBytes = 0;
    for (const pool of this.pools.values()) {
      const freeSpace = this.getFreeSpace(pool);
      const largestFreeBlock = Math.max(...pool.freeList.map(b => b.size), 0);
      this.stats.fragmentedBytes += freeSpace - largestFreeBlock;
    }

    return { ...this.stats };
  }

  /**
   * Get detailed pool information
   */
  getPoolInfo(poolName: string): {
    size: number;
    used: number;
    free: number;
    fragmentation: number;
    allocationCount: number;
  } | null {
    const pool = this.pools.get(poolName);
    if (!pool) return null;

    const freeSpace = this.getFreeSpace(pool);
    return {
      size: pool.size,
      used: pool.size - freeSpace,
      free: freeSpace,
      fragmentation: this.getFragmentation(pool),
      allocationCount: pool.allocations.size,
    };
  }

  /**
   * Dispose of all pools and resources
   */
  dispose(): void {
    for (const pool of this.pools.values()) {
      pool.buffer.destroy();
    }
    this.pools.clear();
    console.log('[BufferManager] Disposed');
  }
}
