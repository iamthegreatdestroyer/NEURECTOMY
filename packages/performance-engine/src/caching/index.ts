/**
 * @fileoverview Caching Module Exports
 * @module @neurectomy/performance-engine/caching
 *
 * Agent Assignment: @VELOCITY @CORE @NEURAL
 *
 * Comprehensive caching systems:
 * - Multi-tier intelligent caching with LRU/LFU/ARC
 * - Bloom filter negative caching
 * - Object pooling for allocation reduction
 * - Buffer pooling for I/O operations
 * - Slab and arena allocators
 *
 * @author NEURECTOMY Phase 5 - Performance Excellence
 * @version 1.0.0
 */

// Cache Manager (@VELOCITY @NEURAL)
export {
  CacheManager,
  BloomFilter,
  type EvictionPolicy,
  type CacheEntry,
  type CacheStats,
  type CacheTierConfig,
  type CacheManagerConfig,
  type CacheEventType,
  type CacheEvent,
  DEFAULT_CACHE_CONFIG,
} from "./cache-manager.js";

// Memory Pool (@CORE @VELOCITY)
export {
  ObjectPool,
  BufferPool,
  SlabAllocator,
  ArenaAllocator,
  type PoolStats,
  type PoolConfig,
  type Poolable,
  type BufferPoolConfig,
  type SlabConfig,
  type SlabStats,
  type ArenaConfig,
  type ArenaStats,
  DEFAULT_POOL_CONFIG,
  DEFAULT_BUFFER_POOL_CONFIG,
  DEFAULT_SLAB_CONFIG,
  DEFAULT_ARENA_CONFIG,
} from "./memory-pool.js";
