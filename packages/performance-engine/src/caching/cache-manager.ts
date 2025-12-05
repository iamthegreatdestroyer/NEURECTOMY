/**
 * @fileoverview Intelligent Caching System
 * @module @neurectomy/performance-engine/caching/cache-manager
 *
 * Agent Assignment: @VELOCITY (Performance) + @NEURAL (Adaptive Learning)
 *
 * Implements multi-tier caching with intelligent eviction policies:
 * - LRU (Least Recently Used)
 * - LFU (Least Frequently Used)
 * - ARC (Adaptive Replacement Cache)
 * - TTL-based expiration
 * - Bloom filter for negative caching
 *
 * @author NEURECTOMY Phase 5 - Performance Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";
import CryptoJS from "crypto-js";

// ============================================================================
// Cache Types (@VELOCITY)
// ============================================================================

/**
 * Cache eviction policy
 */
export type EvictionPolicy = "lru" | "lfu" | "arc" | "ttl" | "adaptive";

/**
 * Cache entry with metadata
 */
export interface CacheEntry<T> {
  key: string;
  value: T;
  size: number;
  createdAt: number;
  lastAccessedAt: number;
  accessCount: number;
  ttl?: number;
  expiresAt?: number;
  priority: number;
  tags: string[];
}

/**
 * Cache statistics
 */
export interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  entries: number;
  size: number;
  maxSize: number;
  evictions: number;
  expirations: number;
}

/**
 * Cache tier configuration
 */
export interface CacheTierConfig {
  name: string;
  maxSize: number;
  maxEntries: number;
  defaultTTL: number;
  policy: EvictionPolicy;
  warmupEnabled: boolean;
}

/**
 * Cache manager configuration
 */
export interface CacheManagerConfig {
  tiers: CacheTierConfig[];
  enableBloomFilter: boolean;
  bloomFilterSize: number;
  bloomFilterHashes: number;
  enableCompression: boolean;
  compressionThreshold: number;
  enablePersistence: boolean;
  persistencePath?: string;
  cleanupInterval: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CACHE_TIER: CacheTierConfig = {
  name: "default",
  maxSize: 100 * 1024 * 1024, // 100MB
  maxEntries: 10000,
  defaultTTL: 3600000, // 1 hour
  policy: "lru",
  warmupEnabled: false,
};

const DEFAULT_CACHE_CONFIG: CacheManagerConfig = {
  tiers: [
    {
      ...DEFAULT_CACHE_TIER,
      name: "hot",
      maxSize: 10 * 1024 * 1024,
      policy: "lfu",
    },
    {
      ...DEFAULT_CACHE_TIER,
      name: "warm",
      maxSize: 50 * 1024 * 1024,
      policy: "lru",
    },
    {
      ...DEFAULT_CACHE_TIER,
      name: "cold",
      maxSize: 200 * 1024 * 1024,
      policy: "ttl",
    },
  ],
  enableBloomFilter: true,
  bloomFilterSize: 100000,
  bloomFilterHashes: 7,
  enableCompression: true,
  compressionThreshold: 1024, // 1KB
  enablePersistence: false,
  cleanupInterval: 60000, // 1 minute
};

// ============================================================================
// Bloom Filter (@VELOCITY)
// ============================================================================

/**
 * Bloom filter for negative caching
 */
export class BloomFilter {
  private bitArray: Uint8Array;
  private size: number;
  private hashCount: number;

  constructor(size: number = 100000, hashCount: number = 7) {
    this.size = size;
    this.hashCount = hashCount;
    this.bitArray = new Uint8Array(Math.ceil(size / 8));
  }

  /**
   * Add item to bloom filter
   */
  add(item: string): void {
    const hashes = this.getHashes(item);
    for (const hash of hashes) {
      const index = hash % this.size;
      const byteIndex = Math.floor(index / 8);
      const bitIndex = index % 8;
      this.bitArray[byteIndex] |= 1 << bitIndex;
    }
  }

  /**
   * Check if item might be in filter
   */
  mightContain(item: string): boolean {
    const hashes = this.getHashes(item);
    for (const hash of hashes) {
      const index = hash % this.size;
      const byteIndex = Math.floor(index / 8);
      const bitIndex = index % 8;
      if ((this.bitArray[byteIndex] & (1 << bitIndex)) === 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * Clear the bloom filter
   */
  clear(): void {
    this.bitArray.fill(0);
  }

  /**
   * Get multiple hashes for an item
   */
  private getHashes(item: string): number[] {
    const hashes: number[] = [];
    for (let i = 0; i < this.hashCount; i++) {
      const hash = CryptoJS.SHA256(`${item}:${i}`).toString();
      hashes.push(parseInt(hash.slice(0, 8), 16));
    }
    return hashes;
  }
}

// ============================================================================
// Cache Tier (@VELOCITY)
// ============================================================================

/**
 * Individual cache tier with configurable eviction policy
 */
export class CacheTier<T> extends EventEmitter {
  private config: CacheTierConfig;
  private entries: Map<string, CacheEntry<T>>;
  private currentSize: number;
  private stats: CacheStats;

  // LRU/LFU tracking
  private accessOrder: string[];
  private frequencyMap: Map<string, number>;

  constructor(config: Partial<CacheTierConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CACHE_TIER, ...config };
    this.entries = new Map();
    this.currentSize = 0;
    this.accessOrder = [];
    this.frequencyMap = new Map();
    this.stats = {
      hits: 0,
      misses: 0,
      hitRate: 0,
      entries: 0,
      size: 0,
      maxSize: this.config.maxSize,
      evictions: 0,
      expirations: 0,
    };
  }

  /**
   * Get value from cache
   */
  get(key: string): T | undefined {
    const entry = this.entries.get(key);

    if (!entry) {
      this.stats.misses++;
      this.updateHitRate();
      return undefined;
    }

    // Check expiration
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.delete(key);
      this.stats.expirations++;
      this.stats.misses++;
      this.updateHitRate();
      return undefined;
    }

    // Update access metadata
    entry.lastAccessedAt = Date.now();
    entry.accessCount++;
    this.updateAccessOrder(key);
    this.frequencyMap.set(key, (this.frequencyMap.get(key) || 0) + 1);

    this.stats.hits++;
    this.updateHitRate();
    this.emit("hit", { key, tier: this.config.name });

    return entry.value;
  }

  /**
   * Set value in cache
   */
  set(
    key: string,
    value: T,
    options: {
      ttl?: number;
      priority?: number;
      tags?: string[];
      size?: number;
    } = {}
  ): boolean {
    const size = options.size || this.estimateSize(value);
    const now = Date.now();

    // Check if we need to evict
    while (
      this.currentSize + size > this.config.maxSize ||
      this.entries.size >= this.config.maxEntries
    ) {
      if (!this.evict()) {
        return false; // Cannot make room
      }
    }

    const ttl = options.ttl ?? this.config.defaultTTL;
    const entry: CacheEntry<T> = {
      key,
      value,
      size,
      createdAt: now,
      lastAccessedAt: now,
      accessCount: 1,
      ttl,
      expiresAt: ttl > 0 ? now + ttl : undefined,
      priority: options.priority ?? 0,
      tags: options.tags ?? [],
    };

    // Remove existing entry if present
    if (this.entries.has(key)) {
      const existing = this.entries.get(key)!;
      this.currentSize -= existing.size;
    }

    this.entries.set(key, entry);
    this.currentSize += size;
    this.updateAccessOrder(key);
    this.frequencyMap.set(key, 1);

    this.updateStats();
    this.emit("set", { key, tier: this.config.name, size });

    return true;
  }

  /**
   * Delete entry from cache
   */
  delete(key: string): boolean {
    const entry = this.entries.get(key);
    if (!entry) return false;

    this.entries.delete(key);
    this.currentSize -= entry.size;
    this.removeFromAccessOrder(key);
    this.frequencyMap.delete(key);

    this.updateStats();
    this.emit("delete", { key, tier: this.config.name });

    return true;
  }

  /**
   * Check if key exists (without updating access)
   */
  has(key: string): boolean {
    const entry = this.entries.get(key);
    if (!entry) return false;

    // Check expiration
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.delete(key);
      this.stats.expirations++;
      return false;
    }

    return true;
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.entries.clear();
    this.currentSize = 0;
    this.accessOrder = [];
    this.frequencyMap.clear();
    this.updateStats();
    this.emit("clear", { tier: this.config.name });
  }

  /**
   * Evict one entry based on policy
   */
  private evict(): boolean {
    let keyToEvict: string | undefined;

    switch (this.config.policy) {
      case "lru":
        keyToEvict = this.accessOrder[0];
        break;

      case "lfu":
        keyToEvict = this.findLFUKey();
        break;

      case "arc":
        keyToEvict = this.findARCKey();
        break;

      case "ttl":
        keyToEvict = this.findExpiredOrOldestKey();
        break;

      case "adaptive":
        keyToEvict = this.findAdaptiveKey();
        break;
    }

    if (keyToEvict) {
      this.delete(keyToEvict);
      this.stats.evictions++;
      this.emit("evict", { key: keyToEvict, policy: this.config.policy });
      return true;
    }

    return false;
  }

  /**
   * Find LFU (least frequently used) key
   */
  private findLFUKey(): string | undefined {
    let minFreq = Infinity;
    let lfuKey: string | undefined;

    for (const [key, freq] of this.frequencyMap) {
      if (freq < minFreq) {
        minFreq = freq;
        lfuKey = key;
      }
    }

    return lfuKey;
  }

  /**
   * Find key using ARC policy
   */
  private findARCKey(): string | undefined {
    // Simplified ARC: prefer LRU for recent entries, LFU for frequent
    const midpoint = Math.floor(this.accessOrder.length / 2);

    // Check LRU portion first
    for (let i = 0; i < midpoint && i < this.accessOrder.length; i++) {
      const key = this.accessOrder[i];
      const freq = this.frequencyMap.get(key) || 0;
      if (freq <= 2) {
        return key;
      }
    }

    // Fall back to pure LRU
    return this.accessOrder[0];
  }

  /**
   * Find expired or oldest key
   */
  private findExpiredOrOldestKey(): string | undefined {
    const now = Date.now();
    let oldestKey: string | undefined;
    let oldestTime = Infinity;

    for (const [key, entry] of this.entries) {
      // Prefer expired entries
      if (entry.expiresAt && entry.expiresAt <= now) {
        return key;
      }

      if (entry.createdAt < oldestTime) {
        oldestTime = entry.createdAt;
        oldestKey = key;
      }
    }

    return oldestKey;
  }

  /**
   * Find key using adaptive policy
   */
  private findAdaptiveKey(): string | undefined {
    // Combine multiple factors
    let bestKey: string | undefined;
    let bestScore = Infinity;
    const now = Date.now();

    for (const [key, entry] of this.entries) {
      // Score = (time since access) / (access count * priority)
      const timeFactor = now - entry.lastAccessedAt;
      const freqFactor = entry.accessCount || 1;
      const priorityFactor = entry.priority + 1;

      const score = timeFactor / (freqFactor * priorityFactor);

      if (score > bestScore) {
        bestScore = score;
        bestKey = key;
      }
    }

    return bestKey;
  }

  /**
   * Update access order for LRU
   */
  private updateAccessOrder(key: string): void {
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  /**
   * Remove key from access order
   */
  private removeFromAccessOrder(key: string): void {
    const index = this.accessOrder.indexOf(key);
    if (index !== -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  /**
   * Estimate size of value
   */
  private estimateSize(value: T): number {
    if (typeof value === "string") {
      return value.length * 2; // UTF-16
    }
    if (typeof value === "number") {
      return 8;
    }
    if (typeof value === "boolean") {
      return 4;
    }
    if (value === null || value === undefined) {
      return 0;
    }
    // Rough estimate for objects
    return JSON.stringify(value).length * 2;
  }

  /**
   * Update hit rate
   */
  private updateHitRate(): void {
    const total = this.stats.hits + this.stats.misses;
    this.stats.hitRate = total > 0 ? this.stats.hits / total : 0;
  }

  /**
   * Update stats
   */
  private updateStats(): void {
    this.stats.entries = this.entries.size;
    this.stats.size = this.currentSize;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    return { ...this.stats };
  }

  /**
   * Get all keys
   */
  keys(): string[] {
    return Array.from(this.entries.keys());
  }

  /**
   * Get entries by tag
   */
  getByTag(tag: string): Array<{ key: string; value: T }> {
    const results: Array<{ key: string; value: T }> = [];

    for (const [key, entry] of this.entries) {
      if (entry.tags.includes(tag)) {
        results.push({ key, value: entry.value });
      }
    }

    return results;
  }

  /**
   * Delete entries by tag
   */
  deleteByTag(tag: string): number {
    let count = 0;

    for (const [key, entry] of this.entries) {
      if (entry.tags.includes(tag)) {
        this.delete(key);
        count++;
      }
    }

    return count;
  }
}

// ============================================================================
// Multi-Tier Cache Manager (@VELOCITY @NEURAL)
// ============================================================================

/**
 * Multi-tier cache manager with intelligent promotion/demotion
 */
export class CacheManager<T> extends EventEmitter {
  private config: CacheManagerConfig;
  private tiers: Map<string, CacheTier<T>>;
  private bloomFilter: BloomFilter;
  private negativeCache: Set<string>;
  private cleanupTimer?: ReturnType<typeof setInterval>;

  constructor(config: Partial<CacheManagerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CACHE_CONFIG, ...config };
    this.tiers = new Map();
    this.negativeCache = new Set();

    // Initialize tiers
    for (const tierConfig of this.config.tiers) {
      this.tiers.set(tierConfig.name, new CacheTier<T>(tierConfig));
    }

    // Initialize bloom filter
    this.bloomFilter = new BloomFilter(
      this.config.bloomFilterSize,
      this.config.bloomFilterHashes
    );

    // Start cleanup timer
    this.startCleanup();
  }

  /**
   * Get value from cache (checking all tiers)
   */
  async get(key: string): Promise<T | undefined> {
    // Check bloom filter for negative caching
    if (this.config.enableBloomFilter && !this.bloomFilter.mightContain(key)) {
      this.emit("bloom-filter-miss", { key });
      return undefined;
    }

    // Check negative cache
    if (this.negativeCache.has(key)) {
      return undefined;
    }

    // Try each tier in order
    const tierNames = Array.from(this.tiers.keys());

    for (let i = 0; i < tierNames.length; i++) {
      const tier = this.tiers.get(tierNames[i])!;
      const value = tier.get(key);

      if (value !== undefined) {
        // Promote to hotter tier if not already hottest
        if (i > 0) {
          await this.promote(key, value, i);
        }
        return value;
      }
    }

    return undefined;
  }

  /**
   * Set value in cache
   */
  async set(
    key: string,
    value: T,
    options: {
      tier?: string;
      ttl?: number;
      priority?: number;
      tags?: string[];
    } = {}
  ): Promise<boolean> {
    const tierName = options.tier || this.config.tiers[0].name;
    const tier = this.tiers.get(tierName);

    if (!tier) {
      throw new Error(`Unknown cache tier: ${tierName}`);
    }

    // Add to bloom filter
    if (this.config.enableBloomFilter) {
      this.bloomFilter.add(key);
    }

    // Remove from negative cache
    this.negativeCache.delete(key);

    // Optionally compress large values
    let finalValue = value;
    if (
      this.config.enableCompression &&
      this.estimateSize(value) > this.config.compressionThreshold
    ) {
      finalValue = await this.compress(value);
    }

    return tier.set(key, finalValue, options);
  }

  /**
   * Delete from all tiers
   */
  delete(key: string): boolean {
    let deleted = false;

    for (const tier of this.tiers.values()) {
      if (tier.delete(key)) {
        deleted = true;
      }
    }

    // Add to negative cache
    this.negativeCache.add(key);

    return deleted;
  }

  /**
   * Check if key exists in any tier
   */
  has(key: string): boolean {
    for (const tier of this.tiers.values()) {
      if (tier.has(key)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Clear all tiers
   */
  clear(): void {
    for (const tier of this.tiers.values()) {
      tier.clear();
    }
    this.bloomFilter.clear();
    this.negativeCache.clear();
  }

  /**
   * Promote entry to hotter tier
   */
  private async promote(
    key: string,
    value: T,
    fromTierIndex: number
  ): Promise<void> {
    const tierNames = Array.from(this.tiers.keys());

    if (fromTierIndex <= 0) return;

    const hotterTierName = tierNames[fromTierIndex - 1];
    const hotterTier = this.tiers.get(hotterTierName)!;
    const currentTierName = tierNames[fromTierIndex];
    const currentTier = this.tiers.get(currentTierName)!;

    // Move to hotter tier
    if (hotterTier.set(key, value)) {
      currentTier.delete(key);
      this.emit("promote", {
        key,
        from: currentTierName,
        to: hotterTierName,
      });
    }
  }

  /**
   * Demote entry to colder tier
   */
  async demote(key: string): Promise<boolean> {
    const tierNames = Array.from(this.tiers.keys());

    for (let i = 0; i < tierNames.length - 1; i++) {
      const tier = this.tiers.get(tierNames[i])!;
      const value = tier.get(key);

      if (value !== undefined) {
        const colderTier = this.tiers.get(tierNames[i + 1])!;

        if (colderTier.set(key, value)) {
          tier.delete(key);
          this.emit("demote", {
            key,
            from: tierNames[i],
            to: tierNames[i + 1],
          });
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Get statistics for all tiers
   */
  getStats(): Record<string, CacheStats> {
    const stats: Record<string, CacheStats> = {};

    for (const [name, tier] of this.tiers) {
      stats[name] = tier.getStats();
    }

    return stats;
  }

  /**
   * Get aggregate statistics
   */
  getAggregateStats(): CacheStats {
    const tierStats = this.getStats();
    const aggregate: CacheStats = {
      hits: 0,
      misses: 0,
      hitRate: 0,
      entries: 0,
      size: 0,
      maxSize: 0,
      evictions: 0,
      expirations: 0,
    };

    for (const stats of Object.values(tierStats)) {
      aggregate.hits += stats.hits;
      aggregate.misses += stats.misses;
      aggregate.entries += stats.entries;
      aggregate.size += stats.size;
      aggregate.maxSize += stats.maxSize;
      aggregate.evictions += stats.evictions;
      aggregate.expirations += stats.expirations;
    }

    const total = aggregate.hits + aggregate.misses;
    aggregate.hitRate = total > 0 ? aggregate.hits / total : 0;

    return aggregate;
  }

  /**
   * Estimate size of value
   */
  private estimateSize(value: T): number {
    if (typeof value === "string") {
      return value.length * 2;
    }
    return JSON.stringify(value).length * 2;
  }

  /**
   * Compress value (placeholder for actual compression)
   */
  private async compress(value: T): Promise<T> {
    // In production, implement actual compression
    return value;
  }

  /**
   * Start cleanup timer
   */
  private startCleanup(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }

    this.cleanupTimer = setInterval(() => {
      this.cleanup();
    }, this.config.cleanupInterval);
  }

  /**
   * Cleanup expired entries
   */
  private cleanup(): void {
    for (const tier of this.tiers.values()) {
      // Trigger get on all keys to check expiration
      for (const key of tier.keys()) {
        tier.has(key); // has() checks expiration
      }
    }

    // Limit negative cache size
    if (this.negativeCache.size > 10000) {
      this.negativeCache.clear();
    }

    this.emit("cleanup");
  }

  /**
   * Stop cache manager
   */
  stop(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = undefined;
    }
  }

  /**
   * Warmup cache from external data
   */
  async warmup(
    data: Array<{ key: string; value: T; tier?: string }>
  ): Promise<number> {
    let loaded = 0;

    for (const item of data) {
      if (await this.set(item.key, item.value, { tier: item.tier })) {
        loaded++;
      }
    }

    this.emit("warmup-complete", { loaded, total: data.length });
    return loaded;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a cache manager with default configuration
 */
export function createCacheManager<T>(
  config?: Partial<CacheManagerConfig>
): CacheManager<T> {
  return new CacheManager<T>(config);
}

/**
 * Create a single-tier cache
 */
export function createCache<T>(
  config?: Partial<CacheTierConfig>
): CacheTier<T> {
  return new CacheTier<T>(config);
}

/**
 * Create a bloom filter
 */
export function createBloomFilter(
  size?: number,
  hashCount?: number
): BloomFilter {
  return new BloomFilter(size, hashCount);
}

// ============================================================================
// Exports
// ============================================================================

export { DEFAULT_CACHE_TIER, DEFAULT_CACHE_CONFIG };
