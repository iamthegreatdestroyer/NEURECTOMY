/**
 * @fileoverview Database Sharding Manager
 * Intelligent database sharding with automatic rebalancing
 * @module @neurectomy/enterprise/scalability
 */

import { EventEmitter } from "events";
import type {
  ShardConfig,
  ShardingConfig,
  ShardingRule,
  ShardingStrategy,
  ShardAssignment,
  ShardStatus,
} from "./types.js";

// =============================================================================
// HASH FUNCTIONS
// =============================================================================

/**
 * MurmurHash3 implementation for consistent hashing
 */
function murmur3(key: string, seed = 0): number {
  let h1 = seed;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;

  for (let i = 0; i < key.length; i++) {
    let k1 = key.charCodeAt(i);
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);

    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = Math.imul(h1, 5) + 0xe6546b64;
  }

  h1 ^= key.length;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;

  return h1 >>> 0;
}

/**
 * XXHash-like fast hashing
 */
function xxhash(key: string, seed = 0): number {
  const PRIME1 = 0x9e3779b1;
  const PRIME2 = 0x85ebca77;
  const PRIME3 = 0xc2b2ae3d;
  const PRIME5 = 0x165667b1;

  let hash = seed + PRIME5;

  for (let i = 0; i < key.length; i++) {
    hash += key.charCodeAt(i) * PRIME3;
    hash = Math.imul((hash << 17) | (hash >>> 15), PRIME2);
  }

  hash ^= hash >>> 15;
  hash = Math.imul(hash, PRIME1);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, PRIME2);
  hash ^= hash >>> 16;

  return hash >>> 0;
}

/**
 * Simple MD5-like hash (not cryptographic, just for distribution)
 */
function md5Like(key: string): number {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    const char = key.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

// =============================================================================
// CONSISTENT HASH RING
// =============================================================================

interface VirtualNode {
  hash: number;
  shardId: string;
}

/**
 * Consistent hash ring for shard distribution
 */
class ConsistentHashRing {
  private ring: VirtualNode[] = [];
  private virtualNodes: number;

  constructor(virtualNodes = 150) {
    this.virtualNodes = virtualNodes;
  }

  addShard(shardId: string): void {
    for (let i = 0; i < this.virtualNodes; i++) {
      const hash = murmur3(`${shardId}:${i}`);
      this.ring.push({ hash, shardId });
    }
    this.ring.sort((a, b) => a.hash - b.hash);
  }

  removeShard(shardId: string): void {
    this.ring = this.ring.filter((node) => node.shardId !== shardId);
  }

  getShard(key: string): string | null {
    if (this.ring.length === 0) return null;

    const hash = murmur3(key);

    // Binary search for the first node with hash >= key hash
    let left = 0;
    let right = this.ring.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.ring[mid].hash < hash) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    // Wrap around to first node if past the end
    const index = left >= this.ring.length ? 0 : left;
    return this.ring[index].shardId;
  }

  getShardCount(): number {
    return new Set(this.ring.map((n) => n.shardId)).size;
  }
}

// =============================================================================
// SHARD ROUTER
// =============================================================================

/**
 * Routes requests to appropriate shards based on rules
 */
class ShardRouter {
  private rules: ShardingRule[] = [];
  private hashRing: ConsistentHashRing;
  private cache: Map<string, { shardId: string; expiry: number }> = new Map();
  private cacheTTL: number;

  constructor(cacheTTL = 60000) {
    this.hashRing = new ConsistentHashRing();
    this.cacheTTL = cacheTTL;
  }

  addRule(rule: ShardingRule): void {
    this.rules.push(rule);
    this.rules.sort((a, b) => b.priority - a.priority);
  }

  removeRule(ruleId: string): void {
    this.rules = this.rules.filter((r) => r.id !== ruleId);
  }

  addShard(shardId: string): void {
    this.hashRing.addShard(shardId);
  }

  removeShard(shardId: string): void {
    this.hashRing.removeShard(shardId);
    // Invalidate cache entries for this shard
    const keysToDelete: string[] = [];
    this.cache.forEach((value, key) => {
      if (value.shardId === shardId) {
        keysToDelete.push(key);
      }
    });
    keysToDelete.forEach((key) => this.cache.delete(key));
  }

  route(
    key: string,
    context?: Record<string, unknown>
  ): ShardAssignment | null {
    // Check cache first
    const cached = this.cache.get(key);
    if (cached && cached.expiry > Date.now()) {
      const rule = this.rules.find((r) => this.matchRule(r, key, context));
      return {
        shardId: cached.shardId,
        rule: rule!,
        key,
        timestamp: new Date(),
        confidence: 1.0,
      };
    }

    // Find matching rule
    for (const rule of this.rules) {
      if (!rule.enabled) continue;

      if (this.matchRule(rule, key, context)) {
        const shardId = this.routeByStrategy(rule, key, context);
        if (shardId) {
          // Cache the result
          this.cache.set(key, {
            shardId,
            expiry: Date.now() + this.cacheTTL,
          });

          return {
            shardId,
            rule,
            key,
            timestamp: new Date(),
            confidence: 0.95,
          };
        }
      }
    }

    // Fallback to consistent hashing
    const shardId = this.hashRing.getShard(key);
    if (shardId) {
      return {
        shardId,
        rule: {
          id: "fallback",
          name: "Fallback Hash",
          strategy: "hash",
          keyField: "id",
          enabled: true,
          priority: 0,
        },
        key,
        timestamp: new Date(),
        confidence: 0.8,
      };
    }

    return null;
  }

  private matchRule(
    rule: ShardingRule,
    key: string,
    context?: Record<string, unknown>
  ): boolean {
    switch (rule.strategy) {
      case "range":
        return this.matchRangeRule(rule, key);
      case "geographic":
        return context?.region !== undefined;
      case "tenant":
        return context?.tenantId !== undefined;
      default:
        return true;
    }
  }

  private matchRangeRule(rule: ShardingRule, key: string): boolean {
    if (rule.rangeStart === undefined || rule.rangeEnd === undefined) {
      return false;
    }

    const numericKey = typeof key === "string" ? parseInt(key, 10) : key;
    if (isNaN(numericKey as number)) {
      return key >= String(rule.rangeStart) && key <= String(rule.rangeEnd);
    }

    return (
      numericKey >= Number(rule.rangeStart) &&
      numericKey <= Number(rule.rangeEnd)
    );
  }

  private routeByStrategy(
    rule: ShardingRule,
    key: string,
    context?: Record<string, unknown>
  ): string | null {
    switch (rule.strategy) {
      case "hash":
        return this.routeByHash(rule, key);
      case "range":
        return this.hashRing.getShard(key);
      case "geographic":
        return this.routeByGeography(rule, context);
      case "tenant":
        return this.routeByTenant(rule, context);
      case "directory":
        // Directory-based requires external lookup
        return this.hashRing.getShard(key);
      case "composite":
        // Composite uses multiple strategies
        return this.routeByComposite(rule, key, context);
      default:
        return this.hashRing.getShard(key);
    }
  }

  private routeByHash(rule: ShardingRule, key: string): string | null {
    const hashFn = this.getHashFunction(rule.hashFunction);
    const hash = hashFn(key);

    // Use consistent hashing with the computed hash
    return this.hashRing.getShard(String(hash));
  }

  private routeByGeography(
    rule: ShardingRule,
    context?: Record<string, unknown>
  ): string | null {
    const region = context?.region as string;
    if (!region || !rule.geographicMapping) return null;

    return rule.geographicMapping[region] || null;
  }

  private routeByTenant(
    rule: ShardingRule,
    context?: Record<string, unknown>
  ): string | null {
    const tenantId = context?.tenantId as string;
    if (!tenantId || !rule.tenantMapping) return null;

    return rule.tenantMapping[tenantId] || this.hashRing.getShard(tenantId);
  }

  private routeByComposite(
    rule: ShardingRule,
    key: string,
    context?: Record<string, unknown>
  ): string | null {
    // Try geographic first, then tenant, then hash
    if (context?.region && rule.geographicMapping) {
      const shardId = this.routeByGeography(rule, context);
      if (shardId) return shardId;
    }

    if (context?.tenantId && rule.tenantMapping) {
      const shardId = this.routeByTenant(rule, context);
      if (shardId) return shardId;
    }

    return this.routeByHash(rule, key);
  }

  private getHashFunction(name?: string): (key: string) => number {
    switch (name) {
      case "xxhash":
        return xxhash;
      case "md5":
        return md5Like;
      case "sha256":
        return (key) => murmur3(key); // Simplified
      case "murmur3":
      default:
        return murmur3;
    }
  }

  clearCache(): void {
    this.cache.clear();
  }
}

// =============================================================================
// SHARD REBALANCER
// =============================================================================

interface RebalanceTask {
  id: string;
  sourceShardId: string;
  targetShardId: string;
  keyRange: { start: string; end: string };
  status: "pending" | "in-progress" | "completed" | "failed";
  progress: number;
  startTime?: Date;
  endTime?: Date;
  error?: string;
}

/**
 * Manages shard rebalancing operations
 */
class ShardRebalancer {
  private tasks: Map<string, RebalanceTask> = new Map();
  private isRunning = false;
  private maxConcurrent: number;

  constructor(maxConcurrent = 2) {
    this.maxConcurrent = maxConcurrent;
  }

  async analyzeImbalance(shards: ShardConfig[]): Promise<{
    imbalanced: boolean;
    maxImbalance: number;
    recommendations: Array<{
      sourceShardId: string;
      targetShardId: string;
      reason: string;
    }>;
  }> {
    if (shards.length < 2) {
      return { imbalanced: false, maxImbalance: 0, recommendations: [] };
    }

    const usages = shards.map((s) => ({
      id: s.id,
      usage: s.capacity.currentUsage,
    }));

    const avgUsage =
      usages.reduce((sum, u) => sum + u.usage, 0) / usages.length;
    const maxDiff = Math.max(
      ...usages.map((u) => Math.abs(u.usage - avgUsage))
    );

    const recommendations: Array<{
      sourceShardId: string;
      targetShardId: string;
      reason: string;
    }> = [];

    // Sort by usage to find overloaded and underutilized shards
    const sorted = [...usages].sort((a, b) => b.usage - a.usage);

    for (let i = 0; i < Math.floor(sorted.length / 2); i++) {
      const overloaded = sorted[i];
      const underutilized = sorted[sorted.length - 1 - i];

      if (overloaded.usage - underutilized.usage > 20) {
        recommendations.push({
          sourceShardId: overloaded.id,
          targetShardId: underutilized.id,
          reason: `Usage imbalance: ${overloaded.usage}% vs ${underutilized.usage}%`,
        });
      }
    }

    return {
      imbalanced: maxDiff > 15, // 15% threshold
      maxImbalance: maxDiff,
      recommendations,
    };
  }

  createTask(
    sourceShardId: string,
    targetShardId: string,
    keyRange: { start: string; end: string }
  ): RebalanceTask {
    const task: RebalanceTask = {
      id: `rebalance-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      sourceShardId,
      targetShardId,
      keyRange,
      status: "pending",
      progress: 0,
    };

    this.tasks.set(task.id, task);
    return task;
  }

  async executeTask(
    task: RebalanceTask,
    onProgress: (progress: number) => void
  ): Promise<void> {
    if (task.status !== "pending") {
      throw new Error(`Task ${task.id} is not in pending state`);
    }

    task.status = "in-progress";
    task.startTime = new Date();

    try {
      // Simulate rebalancing with progress updates
      for (let i = 0; i <= 100; i += 10) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        task.progress = i;
        onProgress(i);
      }

      task.status = "completed";
      task.progress = 100;
      task.endTime = new Date();
    } catch (error) {
      task.status = "failed";
      task.error = error instanceof Error ? error.message : String(error);
      task.endTime = new Date();
      throw error;
    }
  }

  getTask(taskId: string): RebalanceTask | undefined {
    return this.tasks.get(taskId);
  }

  getActiveTasks(): RebalanceTask[] {
    return Array.from(this.tasks.values()).filter(
      (t) => t.status === "pending" || t.status === "in-progress"
    );
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (task && task.status === "pending") {
      this.tasks.delete(taskId);
      return true;
    }
    return false;
  }
}

// =============================================================================
// DATABASE SHARDING MANAGER
// =============================================================================

interface ShardConnection {
  shardId: string;
  isConnected: boolean;
  lastPing?: Date;
  latency?: number;
}

/**
 * Database Sharding Manager
 * Manages database sharding with automatic routing and rebalancing
 */
export class DatabaseShardingManager extends EventEmitter {
  private config: ShardingConfig;
  private shards: Map<string, ShardConfig> = new Map();
  private router: ShardRouter;
  private rebalancer: ShardRebalancer;
  private connections: Map<string, ShardConnection> = new Map();
  private healthCheckInterval?: ReturnType<typeof setInterval>;
  private rebalanceInterval?: ReturnType<typeof setInterval>;
  private isStarted = false;

  constructor(config: ShardingConfig) {
    super();
    this.config = config;
    this.router = new ShardRouter(config.routing.cacheTTL);
    this.rebalancer = new ShardRebalancer(config.rebalancing.maxConcurrent);

    // Initialize shards
    for (const shard of config.shards) {
      this.shards.set(shard.id, shard);
      this.router.addShard(shard.id);
    }

    // Initialize rules
    for (const rule of config.rules) {
      this.router.addRule(rule);
    }
  }

  /**
   * Start the sharding manager
   */
  async start(): Promise<void> {
    if (this.isStarted) return;

    this.emit("starting");

    // Connect to all shards
    await this.connectToShards();

    // Start health checks
    this.startHealthChecks();

    // Start rebalancing if enabled
    if (this.config.rebalancing.enabled) {
      this.startRebalancing();
    }

    this.isStarted = true;
    this.emit("started");
  }

  /**
   * Stop the sharding manager
   */
  async stop(): Promise<void> {
    if (!this.isStarted) return;

    this.emit("stopping");

    // Stop intervals
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    if (this.rebalanceInterval) {
      clearInterval(this.rebalanceInterval);
    }

    // Disconnect from shards
    await this.disconnectFromShards();

    this.isStarted = false;
    this.emit("stopped");
  }

  /**
   * Get the shard assignment for a key
   */
  getShard(
    key: string,
    context?: Record<string, unknown>
  ): ShardAssignment | null {
    if (!this.config.enabled) {
      // Return first active shard if sharding is disabled
      const activeShard = Array.from(this.shards.values()).find(
        (s) => s.status === "active"
      );
      if (activeShard) {
        return {
          shardId: activeShard.id,
          rule: {
            id: "disabled",
            name: "Sharding Disabled",
            strategy: "hash",
            keyField: "id",
            enabled: false,
            priority: 0,
          },
          key,
          timestamp: new Date(),
          confidence: 1.0,
        };
      }
      return null;
    }

    const assignment = this.router.route(key, context);

    if (assignment) {
      // Verify shard is available
      const shard = this.shards.get(assignment.shardId);
      if (!shard || shard.status !== "active") {
        // Try fallback shard
        if (this.config.routing.fallbackShardId) {
          const fallback = this.shards.get(this.config.routing.fallbackShardId);
          if (fallback && fallback.status === "active") {
            assignment.shardId = fallback.id;
            assignment.confidence *= 0.8;
          }
        }
      }

      this.emit("shard:assigned", assignment);
    }

    return assignment;
  }

  /**
   * Add a new shard
   */
  async addShard(shard: ShardConfig): Promise<void> {
    if (this.shards.has(shard.id)) {
      throw new Error(`Shard ${shard.id} already exists`);
    }

    this.shards.set(shard.id, shard);
    this.router.addShard(shard.id);

    // Connect to the new shard
    await this.connectToShard(shard);

    this.emit("shard:added", shard);
  }

  /**
   * Remove a shard
   */
  async removeShard(shardId: string, migrateData = true): Promise<void> {
    const shard = this.shards.get(shardId);
    if (!shard) {
      throw new Error(`Shard ${shardId} not found`);
    }

    // Mark as draining
    shard.status = "readonly";
    this.emit("shard:draining", { shardId });

    if (migrateData) {
      // Find target shards and migrate data
      const otherShards = Array.from(this.shards.values()).filter(
        (s) => s.id !== shardId && s.status === "active"
      );

      if (otherShards.length > 0) {
        // Create migration tasks
        for (const target of otherShards) {
          const task = this.rebalancer.createTask(shardId, target.id, {
            start: "",
            end: "",
          });
          await this.rebalancer.executeTask(task, (progress) => {
            this.emit("shard:migration-progress", {
              shardId,
              targetId: target.id,
              progress,
            });
          });
        }
      }
    }

    // Remove from router and disconnect
    this.router.removeShard(shardId);
    await this.disconnectFromShard(shardId);
    this.shards.delete(shardId);

    this.emit("shard:removed", { shardId });
  }

  /**
   * Update shard configuration
   */
  updateShard(shardId: string, updates: Partial<ShardConfig>): void {
    const shard = this.shards.get(shardId);
    if (!shard) {
      throw new Error(`Shard ${shardId} not found`);
    }

    Object.assign(shard, updates);
    this.emit("shard:updated", shard);
  }

  /**
   * Add a sharding rule
   */
  addRule(rule: ShardingRule): void {
    this.config.rules.push(rule);
    this.router.addRule(rule);
    this.emit("rule:added", rule);
  }

  /**
   * Remove a sharding rule
   */
  removeRule(ruleId: string): void {
    this.config.rules = this.config.rules.filter((r) => r.id !== ruleId);
    this.router.removeRule(ruleId);
    this.emit("rule:removed", { ruleId });
  }

  /**
   * Trigger manual rebalancing
   */
  async rebalance(): Promise<void> {
    const shards = Array.from(this.shards.values());
    const analysis = await this.rebalancer.analyzeImbalance(shards);

    if (!analysis.imbalanced) {
      this.emit("rebalance:skipped", { reason: "Shards are balanced" });
      return;
    }

    this.emit("rebalance:started", analysis);

    for (const rec of analysis.recommendations) {
      const task = this.rebalancer.createTask(
        rec.sourceShardId,
        rec.targetShardId,
        { start: "", end: "" }
      );

      try {
        await this.rebalancer.executeTask(task, (progress) => {
          this.emit("shard:rebalancing", {
            shardId: rec.sourceShardId,
            progress,
          });
        });
      } catch (error) {
        this.emit("rebalance:error", { task, error });
      }
    }

    this.emit("rebalance:completed");
  }

  /**
   * Get all shards
   */
  getShards(): ShardConfig[] {
    return Array.from(this.shards.values());
  }

  /**
   * Get shard by ID
   */
  getShardById(shardId: string): ShardConfig | undefined {
    return this.shards.get(shardId);
  }

  /**
   * Get shard statistics
   */
  getStats(): {
    totalShards: number;
    activeShards: number;
    totalCapacity: number;
    usedCapacity: number;
    avgUsage: number;
  } {
    const shards = Array.from(this.shards.values());
    const activeShards = shards.filter((s) => s.status === "active");
    const totalCapacity = shards.reduce(
      (sum, s) => sum + s.capacity.maxStorage,
      0
    );
    const usedCapacity = shards.reduce(
      (sum, s) => sum + (s.capacity.maxStorage * s.capacity.currentUsage) / 100,
      0
    );
    const avgUsage =
      shards.length > 0
        ? shards.reduce((sum, s) => sum + s.capacity.currentUsage, 0) /
          shards.length
        : 0;

    return {
      totalShards: shards.length,
      activeShards: activeShards.length,
      totalCapacity,
      usedCapacity,
      avgUsage,
    };
  }

  /**
   * Clear routing cache
   */
  clearCache(): void {
    this.router.clearCache();
    this.emit("cache:cleared");
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  private async connectToShards(): Promise<void> {
    const promises = Array.from(this.shards.values()).map((shard) =>
      this.connectToShard(shard)
    );
    await Promise.all(promises);
  }

  private async connectToShard(shard: ShardConfig): Promise<void> {
    try {
      // Simulate connection
      await new Promise((resolve) => setTimeout(resolve, 100));

      this.connections.set(shard.id, {
        shardId: shard.id,
        isConnected: true,
        lastPing: new Date(),
        latency: Math.random() * 50 + 10,
      });

      shard.status = "active";
      this.emit("shard:connected", { shardId: shard.id });
    } catch (error) {
      shard.status = "offline";
      this.emit("shard:connection-failed", { shardId: shard.id, error });
      throw error;
    }
  }

  private async disconnectFromShards(): Promise<void> {
    const promises = Array.from(this.shards.keys()).map((shardId) =>
      this.disconnectFromShard(shardId)
    );
    await Promise.all(promises);
  }

  private async disconnectFromShard(shardId: string): Promise<void> {
    try {
      await new Promise((resolve) => setTimeout(resolve, 50));
      this.connections.delete(shardId);
      this.emit("shard:disconnected", { shardId });
    } catch (error) {
      this.emit("shard:disconnect-failed", { shardId, error });
    }
  }

  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(async () => {
      const entries = Array.from(this.connections.entries());
      for (let i = 0; i < entries.length; i++) {
        const [shardId, connection] = entries[i];
        try {
          const start = Date.now();
          await new Promise((resolve) => setTimeout(resolve, 10)); // Simulate ping
          const latency = Date.now() - start;

          connection.lastPing = new Date();
          connection.latency = latency;
          connection.isConnected = true;

          const shard = this.shards.get(shardId);
          if (shard && shard.status === "offline") {
            shard.status = "active";
            this.emit("shard:recovered", { shardId });
          }
        } catch {
          connection.isConnected = false;
          const shard = this.shards.get(shardId);
          if (shard && shard.status === "active") {
            shard.status = "degraded";
            this.emit("shard:unhealthy", { shardId });
          }
        }
      }
    }, 30000); // Every 30 seconds
  }

  private startRebalancing(): void {
    const interval = this.config.rebalancing.scheduleExpression
      ? 3600000 // Default 1 hour if schedule is set
      : 300000; // 5 minutes otherwise

    this.rebalanceInterval = setInterval(async () => {
      const shards = Array.from(this.shards.values());
      const analysis = await this.rebalancer.analyzeImbalance(shards);

      if (analysis.maxImbalance > this.config.rebalancing.threshold) {
        await this.rebalance();
      }
    }, interval);
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a database sharding manager
 */
export function createDatabaseShardingManager(
  config: ShardingConfig
): DatabaseShardingManager {
  return new DatabaseShardingManager(config);
}

/**
 * Create default sharding configuration
 */
export function createDefaultShardingConfig(
  shards: Array<{ id: string; name: string; connectionString: string }>
): ShardingConfig {
  return {
    enabled: true,
    strategy: "hash",
    shards: shards.map((s, i) => ({
      id: s.id,
      name: s.name,
      connectionString: s.connectionString,
      priority: i + 1,
      status: "active" as ShardStatus,
      capacity: {
        maxConnections: 100,
        maxStorage: 10 * 1024 * 1024 * 1024, // 10GB
        currentUsage: 0,
      },
      metadata: {},
    })),
    rules: [
      {
        id: "default-hash",
        name: "Default Hash Routing",
        strategy: "hash" as ShardingStrategy,
        keyField: "id",
        hashFunction: "murmur3",
        enabled: true,
        priority: 1,
      },
    ],
    rebalancing: {
      enabled: true,
      threshold: 20,
      maxConcurrent: 2,
    },
    routing: {
      cacheEnabled: true,
      cacheTTL: 60000,
    },
  };
}
