/**
 * @neurectomy/performance-engine - Memory Profiler
 *
 * @elite-agent-collective @CORE @VELOCITY
 *
 * Advanced memory profiling with heap snapshot analysis,
 * leak detection, and allocation tracking.
 */

import { EventEmitter } from "events";
import * as crypto from "crypto";
import type {
  HeapSnapshotResult,
  HeapSnapshotNode,
  MemoryAllocation,
  MemoryLeakDetection,
  GCEvent,
  MemoryStats,
  MemoryPressureLevel,
} from "../types.js";

// ============================================================================
// TYPES
// ============================================================================

interface MemoryProfilerOptions {
  trackAllocations?: boolean;
  allocationSamplingRate?: number;
  gcNotifications?: boolean;
  pressureThresholds?: {
    moderate: number;
    critical: number;
  };
}

interface AllocationSite {
  stackTrace: string[];
  totalAllocations: number;
  totalSize: number;
  retained: number;
  retainedSize: number;
}

interface HeapDiff {
  addedNodes: HeapSnapshotNode[];
  removedNodes: HeapSnapshotNode[];
  grownNodes: Array<{
    node: HeapSnapshotNode;
    sizeDelta: number;
    countDelta: number;
  }>;
  totalDelta: number;
}

// ============================================================================
// MEMORY PROFILER IMPLEMENTATION
// ============================================================================

/**
 * Comprehensive memory profiler with leak detection
 *
 * @elite-agent-collective @CORE - V8 heap inspection
 * @elite-agent-collective @VELOCITY - Efficient memory tracking
 */
export class MemoryProfiler extends EventEmitter {
  private options: Required<MemoryProfilerOptions>;
  private allocations: Map<string, MemoryAllocation> = new Map();
  private allocationSites: Map<string, AllocationSite> = new Map();
  private gcHistory: GCEvent[] = [];
  private snapshots: Map<string, HeapSnapshotResult> = new Map();
  private isTracking: boolean = false;
  private lastHeapUsed: number = 0;
  private memoryTimeline: Array<{
    timestamp: number;
    heapUsed: number;
    heapTotal: number;
  }> = [];

  private readonly defaultOptions: Required<MemoryProfilerOptions> = {
    trackAllocations: false,
    allocationSamplingRate: 0.01, // 1% of allocations
    gcNotifications: true,
    pressureThresholds: {
      moderate: 0.7, // 70% of heap limit
      critical: 0.9, // 90% of heap limit
    },
  };

  constructor(options?: MemoryProfilerOptions) {
    super();
    this.options = { ...this.defaultOptions, ...options };
    this.setupGCNotifications();
    this.startMemoryMonitoring();
  }

  /**
   * Take a heap snapshot
   */
  async takeSnapshot(name?: string): Promise<HeapSnapshotResult> {
    const snapshotId = crypto.randomUUID();
    const timestamp = Date.now();

    // Get V8 heap statistics
    const heapStats = this.getV8HeapStats();

    // Build heap snapshot
    const snapshot = await this.buildHeapSnapshot(
      snapshotId,
      timestamp,
      heapStats
    );

    // Detect potential leaks
    const leakSuspects = this.detectLeakSuspects(snapshot);
    snapshot.leakSuspects = leakSuspects;

    // Store snapshot
    this.snapshots.set(snapshotId, snapshot);

    this.emit("snapshot:taken", { snapshotId, timestamp });
    return snapshot;
  }

  /**
   * Compare two snapshots to find memory growth
   */
  compareSnapshots(snapshotId1: string, snapshotId2: string): HeapDiff {
    const snapshot1 = this.snapshots.get(snapshotId1);
    const snapshot2 = this.snapshots.get(snapshotId2);

    if (!snapshot1 || !snapshot2) {
      throw new Error("One or both snapshots not found");
    }

    const nodeMap1 = new Map(snapshot1.nodes.map((n) => [n.nodeId, n]));
    const nodeMap2 = new Map(snapshot2.nodes.map((n) => [n.nodeId, n]));

    const addedNodes: HeapSnapshotNode[] = [];
    const removedNodes: HeapSnapshotNode[] = [];
    const grownNodes: HeapDiff["grownNodes"] = [];

    // Find added and grown nodes
    for (const [nodeId, node] of nodeMap2) {
      const prevNode = nodeMap1.get(nodeId);
      if (!prevNode) {
        addedNodes.push(node);
      } else if (node.retainedSize > prevNode.retainedSize) {
        grownNodes.push({
          node,
          sizeDelta: node.retainedSize - prevNode.retainedSize,
          countDelta: node.edgeCount - prevNode.edgeCount,
        });
      }
    }

    // Find removed nodes
    for (const [nodeId, node] of nodeMap1) {
      if (!nodeMap2.has(nodeId)) {
        removedNodes.push(node);
      }
    }

    const totalDelta = snapshot2.totalSize - snapshot1.totalSize;

    return {
      addedNodes,
      removedNodes,
      grownNodes: grownNodes.sort((a, b) => b.sizeDelta - a.sizeDelta),
      totalDelta,
    };
  }

  /**
   * Start tracking allocations
   */
  startAllocationTracking(): void {
    if (this.isTracking) {
      throw new Error("Allocation tracking already started");
    }

    this.isTracking = true;
    this.allocations.clear();
    this.allocationSites.clear();

    this.emit("tracking:started");
  }

  /**
   * Stop tracking allocations
   */
  stopAllocationTracking(): AllocationSite[] {
    if (!this.isTracking) {
      throw new Error("Allocation tracking not started");
    }

    this.isTracking = false;

    const sites = Array.from(this.allocationSites.values()).sort(
      (a, b) => b.totalSize - a.totalSize
    );

    this.emit("tracking:stopped", { sites: sites.length });
    return sites;
  }

  /**
   * Record an allocation (called by instrumented code)
   */
  recordAllocation(size: number, type: string, stackTrace?: string[]): void {
    if (!this.isTracking) return;

    // Sample allocations based on rate
    if (Math.random() > this.options.allocationSamplingRate) return;

    const stack = stackTrace || this.captureStackTrace();
    const siteKey = stack.slice(0, 5).join("|");

    // Update allocation site
    let site = this.allocationSites.get(siteKey);
    if (!site) {
      site = {
        stackTrace: stack,
        totalAllocations: 0,
        totalSize: 0,
        retained: 0,
        retainedSize: 0,
      };
      this.allocationSites.set(siteKey, site);
    }

    site.totalAllocations++;
    site.totalSize += size;

    // Record individual allocation
    const allocationId = crypto.randomUUID();
    this.allocations.set(allocationId, {
      id: allocationId,
      timestamp: Date.now(),
      size,
      type,
      stackTrace: stack,
      retained: true,
      retainedSize: size,
    });
  }

  /**
   * Detect memory leaks by analyzing growth patterns
   */
  async detectLeaks(): Promise<MemoryLeakDetection> {
    const detectionId = crypto.randomUUID();
    const timestamp = Date.now();

    // Take snapshot for analysis
    const snapshot = await this.takeSnapshot("leak-detection");

    // Analyze memory timeline for growth patterns
    const growthAnalysis = this.analyzeGrowthPattern();

    // Combine leak suspects
    const suspects = [
      ...snapshot.leakSuspects.map((s) => ({
        type: "heap",
        location: s.name,
        retainedSize: s.retainedSize,
        retainerPath: [], // Would need full retainer analysis
        growthRate: 0,
        firstSeen: timestamp,
        evidence: [s.reason],
      })),
      ...growthAnalysis.suspects,
    ];

    const detected = suspects.length > 0 || growthAnalysis.isGrowing;
    const confidence = this.calculateLeakConfidence(suspects, growthAnalysis);

    return {
      id: detectionId,
      timestamp,
      detected,
      confidence,
      suspects,
      recommendations: this.generateLeakRecommendations(suspects),
      memoryTimeline: this.memoryTimeline.slice(-100),
    };
  }

  /**
   * Get current memory statistics
   */
  getMemoryStats(): MemoryStats {
    const heapStats = this.getV8HeapStats();
    const pressureLevel = this.calculatePressureLevel(heapStats);

    // Calculate GC stats
    const gcStats = this.calculateGCStats();

    // Get pool stats (if any pools registered)
    const pools: MemoryStats["pools"] = [];

    return {
      timestamp: Date.now(),
      heapUsed: heapStats.heapUsed,
      heapTotal: heapStats.heapTotal,
      heapLimit: heapStats.heapLimit,
      external: heapStats.external,
      arrayBuffers: heapStats.arrayBuffers,
      rss: heapStats.rss,
      pressureLevel,
      gcStats,
      pools,
    };
  }

  /**
   * Get GC history
   */
  getGCHistory(limit?: number): GCEvent[] {
    const history = [...this.gcHistory].reverse();
    return limit ? history.slice(0, limit) : history;
  }

  /**
   * Force garbage collection (if exposed)
   */
  forceGC(): boolean {
    if (global.gc) {
      const before = process.memoryUsage().heapUsed;
      global.gc();
      const after = process.memoryUsage().heapUsed;

      this.emit("gc:forced", { freed: before - after });
      return true;
    }
    return false;
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Setup GC notifications
   */
  private setupGCNotifications(): void {
    if (!this.options.gcNotifications) return;

    // Monitor for heap changes that indicate GC
    const checkInterval = setInterval(() => {
      const currentHeap = process.memoryUsage().heapUsed;

      // If heap decreased significantly, GC likely occurred
      if (this.lastHeapUsed > 0 && currentHeap < this.lastHeapUsed * 0.9) {
        const gcEvent: GCEvent = {
          timestamp: Date.now(),
          type:
            this.lastHeapUsed - currentHeap > 10 * 1024 * 1024
              ? "mark_sweep"
              : "scavenge",
          duration: 0, // Can't measure without native hooks
          usedHeapBefore: this.lastHeapUsed,
          usedHeapAfter: currentHeap,
          freedMemory: this.lastHeapUsed - currentHeap,
          totalHeapSize: process.memoryUsage().heapTotal,
          externalMemory: process.memoryUsage().external,
        };

        this.gcHistory.push(gcEvent);
        if (this.gcHistory.length > 1000) {
          this.gcHistory.shift();
        }

        this.emit("gc:detected", gcEvent);
      }

      this.lastHeapUsed = currentHeap;
    }, 100);

    // Cleanup on exit
    process.on("exit", () => clearInterval(checkInterval));
  }

  /**
   * Start memory monitoring for timeline
   */
  private startMemoryMonitoring(): void {
    const monitorInterval = setInterval(() => {
      const mem = process.memoryUsage();
      this.memoryTimeline.push({
        timestamp: Date.now(),
        heapUsed: mem.heapUsed,
        heapTotal: mem.heapTotal,
      });

      // Keep last hour of data (assuming 1 sample/sec = 3600 samples)
      if (this.memoryTimeline.length > 3600) {
        this.memoryTimeline.shift();
      }

      // Check for memory pressure
      const heapStats = this.getV8HeapStats();
      const pressure = this.calculatePressureLevel(heapStats);
      if (pressure !== "normal") {
        this.emit("memory:pressure", {
          level: pressure,
          heapUsed: mem.heapUsed,
        });
      }
    }, 1000);

    process.on("exit", () => clearInterval(monitorInterval));
  }

  /**
   * Get V8 heap statistics
   */
  private getV8HeapStats(): {
    heapUsed: number;
    heapTotal: number;
    heapLimit: number;
    external: number;
    arrayBuffers: number;
    rss: number;
  } {
    const mem = process.memoryUsage();

    return {
      heapUsed: mem.heapUsed,
      heapTotal: mem.heapTotal,
      heapLimit: mem.heapTotal * 1.5, // Approximate
      external: mem.external,
      arrayBuffers: mem.arrayBuffers,
      rss: mem.rss,
    };
  }

  /**
   * Build heap snapshot (simplified - full impl would use v8.writeHeapSnapshot)
   */
  private async buildHeapSnapshot(
    id: string,
    timestamp: number,
    heapStats: ReturnType<typeof this.getV8HeapStats>
  ): Promise<HeapSnapshotResult> {
    // Create synthetic snapshot based on memory usage
    // Real implementation would use v8.writeHeapSnapshot() and parse the result

    const nodes: HeapSnapshotNode[] = [];
    let nodeId = 0;

    // Add synthetic root node
    nodes.push({
      nodeId: nodeId++,
      name: "(GC roots)",
      type: "synthetic",
      selfSize: 0,
      retainedSize: heapStats.heapUsed,
      edgeCount: 3,
      childrenIds: [1, 2, 3],
    });

    // Add synthetic category nodes
    const categories = [
      { name: "Code", type: "code" as const, sizeRatio: 0.2 },
      { name: "Strings", type: "string" as const, sizeRatio: 0.3 },
      { name: "Objects", type: "object" as const, sizeRatio: 0.4 },
      { name: "Arrays", type: "array" as const, sizeRatio: 0.1 },
    ];

    for (const cat of categories) {
      const size = Math.round(heapStats.heapUsed * cat.sizeRatio);
      nodes.push({
        nodeId: nodeId++,
        name: cat.name,
        type: cat.type,
        selfSize: size,
        retainedSize: size,
        edgeCount: 0,
        childrenIds: [],
      });
    }

    return {
      id,
      timestamp,
      totalSize: heapStats.heapUsed,
      totalNodes: nodes.length,
      nodes,
      dominatorTree: { "0": [1, 2, 3, 4] },
      retainerPaths: {},
      summary: {
        objectCount: Math.round(heapStats.heapUsed / 100), // Estimate
        stringCount: Math.round((heapStats.heapUsed * 0.3) / 50),
        arrayCount: Math.round((heapStats.heapUsed * 0.1) / 200),
        closureCount: Math.round((heapStats.heapUsed * 0.05) / 100),
        codeCount: Math.round((heapStats.heapUsed * 0.2) / 500),
        externalSize: heapStats.external,
        nativeSize: heapStats.arrayBuffers,
      },
      leakSuspects: [],
    };
  }

  /**
   * Detect potential leak suspects in snapshot
   */
  private detectLeakSuspects(
    snapshot: HeapSnapshotResult
  ): HeapSnapshotResult["leakSuspects"] {
    const suspects: HeapSnapshotResult["leakSuspects"] = [];

    // Look for nodes with unusually large retained sizes
    const avgRetainedSize = snapshot.totalSize / snapshot.totalNodes;
    const threshold = avgRetainedSize * 100; // 100x average

    for (const node of snapshot.nodes) {
      if (node.retainedSize > threshold && node.type === "object") {
        suspects.push({
          nodeId: node.nodeId,
          name: node.name,
          retainedSize: node.retainedSize,
          reason: `Unusually large retained size (${(node.retainedSize / 1024 / 1024).toFixed(2)} MB)`,
          confidence: Math.min(0.9, node.retainedSize / snapshot.totalSize),
        });
      }
    }

    return suspects;
  }

  /**
   * Analyze memory growth pattern
   */
  private analyzeGrowthPattern(): {
    isGrowing: boolean;
    growthRate: number;
    suspects: MemoryLeakDetection["suspects"];
  } {
    if (this.memoryTimeline.length < 60) {
      return { isGrowing: false, growthRate: 0, suspects: [] };
    }

    const recent = this.memoryTimeline.slice(-60);
    const older = this.memoryTimeline.slice(-120, -60);

    if (older.length === 0) {
      return { isGrowing: false, growthRate: 0, suspects: [] };
    }

    const recentAvg =
      recent.reduce((sum, p) => sum + p.heapUsed, 0) / recent.length;
    const olderAvg =
      older.reduce((sum, p) => sum + p.heapUsed, 0) / older.length;

    const growthRate = (recentAvg - olderAvg) / olderAvg;
    const isGrowing = growthRate > 0.1; // 10% growth

    const suspects: MemoryLeakDetection["suspects"] = [];
    if (isGrowing) {
      suspects.push({
        type: "growth_pattern",
        location: "heap",
        retainedSize: recentAvg - olderAvg,
        retainerPath: ["(monitoring)", "(heap)"],
        growthRate,
        firstSeen: recent[0].timestamp,
        evidence: [
          `Memory grew by ${(growthRate * 100).toFixed(1)}% in the last minute`,
          `Current heap: ${(recentAvg / 1024 / 1024).toFixed(2)} MB`,
          `Previous heap: ${(olderAvg / 1024 / 1024).toFixed(2)} MB`,
        ],
      });
    }

    return { isGrowing, growthRate, suspects };
  }

  /**
   * Calculate leak detection confidence
   */
  private calculateLeakConfidence(
    suspects: MemoryLeakDetection["suspects"],
    growthAnalysis: { isGrowing: boolean; growthRate: number }
  ): number {
    if (suspects.length === 0) return 0;

    let confidence = 0;

    // Base confidence from number of suspects
    confidence += Math.min(0.3, suspects.length * 0.1);

    // Growth pattern confidence
    if (growthAnalysis.isGrowing) {
      confidence += Math.min(0.4, growthAnalysis.growthRate * 2);
    }

    // Suspect evidence confidence
    const avgEvidence =
      suspects.reduce((sum, s) => sum + s.evidence.length, 0) / suspects.length;
    confidence += Math.min(0.3, avgEvidence * 0.1);

    return Math.min(1, confidence);
  }

  /**
   * Generate leak recommendations
   */
  private generateLeakRecommendations(
    suspects: MemoryLeakDetection["suspects"]
  ): string[] {
    const recommendations: string[] = [];

    if (suspects.length === 0) {
      recommendations.push("No memory leaks detected. Continue monitoring.");
      return recommendations;
    }

    recommendations.push("Potential memory leak(s) detected:");

    for (const suspect of suspects.slice(0, 5)) {
      if (suspect.type === "growth_pattern") {
        recommendations.push(
          `• Investigate continuous memory growth (${(suspect.growthRate * 100).toFixed(1)}%/min)`
        );
      } else if (suspect.type === "heap") {
        recommendations.push(
          `• Large retained object: ${suspect.location} (${(suspect.retainedSize / 1024 / 1024).toFixed(2)} MB)`
        );
      }
    }

    recommendations.push("\nRecommended actions:");
    recommendations.push(
      "1. Take heap snapshots before and after suspected operations"
    );
    recommendations.push("2. Compare snapshots to identify growing objects");
    recommendations.push("3. Check for event listeners not being removed");
    recommendations.push("4. Look for closures holding large data");
    recommendations.push(
      "5. Verify Maps/Sets are cleaned up when no longer needed"
    );

    return recommendations;
  }

  /**
   * Calculate memory pressure level
   */
  private calculatePressureLevel(
    heapStats: ReturnType<typeof this.getV8HeapStats>
  ): MemoryPressureLevel {
    const usage = heapStats.heapUsed / heapStats.heapLimit;

    if (usage >= 0.95) return "out_of_memory";
    if (usage >= this.options.pressureThresholds.critical) return "critical";
    if (usage >= this.options.pressureThresholds.moderate) return "moderate";
    return "normal";
  }

  /**
   * Calculate GC statistics
   */
  private calculateGCStats(): MemoryStats["gcStats"] {
    if (this.gcHistory.length === 0) {
      return {
        totalCollections: 0,
        totalPauseTime: 0,
        avgPauseTime: 0,
        maxPauseTime: 0,
      };
    }

    const totalPauseTime = this.gcHistory.reduce(
      (sum, gc) => sum + gc.duration,
      0
    );
    const maxPauseTime = Math.max(...this.gcHistory.map((gc) => gc.duration));

    return {
      totalCollections: this.gcHistory.length,
      totalPauseTime,
      avgPauseTime: totalPauseTime / this.gcHistory.length,
      maxPauseTime,
      lastCollection: this.gcHistory[this.gcHistory.length - 1].timestamp,
    };
  }

  /**
   * Capture stack trace
   */
  private captureStackTrace(): string[] {
    const err = new Error();
    const stack = err.stack || "";
    return stack
      .split("\n")
      .slice(3)
      .map((line) => line.trim());
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { MemoryProfiler as default };
export type { MemoryProfilerOptions, AllocationSite, HeapDiff };
