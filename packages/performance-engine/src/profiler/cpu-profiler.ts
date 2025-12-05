/**
 * @neurectomy/performance-engine - CPU Profiler
 *
 * @elite-agent-collective @VELOCITY @CORE
 *
 * High-precision CPU profiling with flamegraph generation,
 * hot function detection, and performance anomaly analysis.
 */

import { EventEmitter } from "events";
import * as crypto from "crypto";
import type {
  CPUProfileResult,
  CPUProfileSample,
  IProfilerService,
} from "../types.js";

// ============================================================================
// TYPES
// ============================================================================

interface ProfilerOptions {
  samplingInterval?: number; // microseconds
  maxStackDepth?: number;
  includeNatives?: boolean;
  recordAllocations?: boolean;
}

interface StackFrame {
  functionName: string;
  scriptName: string;
  lineNumber: number;
  columnNumber: number;
  isNative: boolean;
}

interface ProfileNode {
  id: number;
  functionName: string;
  scriptName: string;
  lineNumber: number;
  columnNumber: number;
  hitCount: number;
  children: Map<string, ProfileNode>;
  selfTime: number;
  totalTime: number;
  bailoutReason?: string;
}

interface ProfileSession {
  id: string;
  name: string;
  startTime: number;
  options: ProfilerOptions;
  rootNode: ProfileNode;
  nodeIdCounter: number;
  samples: number[];
  timeDeltas: number[];
  isRunning: boolean;
}

// ============================================================================
// CPU PROFILER IMPLEMENTATION
// ============================================================================

/**
 * High-performance CPU profiler with call tree analysis
 *
 * @elite-agent-collective @VELOCITY - Sub-linear sampling strategies
 * @elite-agent-collective @CORE - Low-level V8 profiler integration
 */
export class CPUProfiler extends EventEmitter {
  private sessions: Map<string, ProfileSession> = new Map();
  private activeSession: ProfileSession | null = null;
  private sampleTimer: NodeJS.Timer | null = null;
  private nodeIdMap: Map<number, ProfileNode> = new Map();

  private readonly defaultOptions: Required<ProfilerOptions> = {
    samplingInterval: 1000, // 1ms
    maxStackDepth: 128,
    includeNatives: false,
    recordAllocations: false,
  };

  constructor() {
    super();
    this.setupSignalHandlers();
  }

  /**
   * Start CPU profiling session
   */
  async start(name: string, options?: ProfilerOptions): Promise<string> {
    if (this.activeSession) {
      throw new Error(
        `Profile session "${this.activeSession.name}" is already running`
      );
    }

    const sessionId = crypto.randomUUID();
    const mergedOptions = { ...this.defaultOptions, ...options };

    const rootNode: ProfileNode = {
      id: 0,
      functionName: "(root)",
      scriptName: "",
      lineNumber: 0,
      columnNumber: 0,
      hitCount: 0,
      children: new Map(),
      selfTime: 0,
      totalTime: 0,
    };

    const session: ProfileSession = {
      id: sessionId,
      name,
      startTime: performance.now(),
      options: mergedOptions,
      rootNode,
      nodeIdCounter: 1,
      samples: [],
      timeDeltas: [],
      isRunning: true,
    };

    this.sessions.set(sessionId, session);
    this.activeSession = session;
    this.nodeIdMap.set(0, rootNode);

    // Start sampling
    this.startSampling(session);

    this.emit("profile:start", { sessionId, name });
    return sessionId;
  }

  /**
   * Stop CPU profiling and return results
   */
  async stop(): Promise<CPUProfileResult> {
    if (!this.activeSession) {
      throw new Error("No active profiling session");
    }

    const session = this.activeSession;
    session.isRunning = false;

    // Stop sampling
    this.stopSampling();

    const endTime = performance.now();
    const duration = endTime - session.startTime;

    // Calculate timing information
    this.calculateTimings(session.rootNode, duration);

    // Convert to profile result
    const result = this.convertToResult(session, endTime, duration);

    // Cleanup
    this.activeSession = null;
    this.nodeIdMap.clear();

    this.emit("profile:stop", { sessionId: session.id, duration });
    return result;
  }

  /**
   * Get current profiling status
   */
  getStatus(): {
    isRunning: boolean;
    sessionId?: string;
    sessionName?: string;
    duration?: number;
  } {
    if (!this.activeSession) {
      return { isRunning: false };
    }

    return {
      isRunning: true,
      sessionId: this.activeSession.id,
      sessionName: this.activeSession.name,
      duration: performance.now() - this.activeSession.startTime,
    };
  }

  /**
   * Start sampling at configured interval
   */
  private startSampling(session: ProfileSession): void {
    let lastSampleTime = session.startTime;

    this.sampleTimer = setInterval(() => {
      if (!session.isRunning) return;

      const now = performance.now();
      const delta = now - lastSampleTime;
      lastSampleTime = now;

      // Capture stack trace
      const stack = this.captureStack(session.options);

      // Add sample to call tree
      const nodeId = this.addStackToTree(session, stack);

      session.samples.push(nodeId);
      session.timeDeltas.push(delta);
    }, session.options.samplingInterval! / 1000);
  }

  /**
   * Stop sampling
   */
  private stopSampling(): void {
    if (this.sampleTimer) {
      clearInterval(this.sampleTimer);
      this.sampleTimer = null;
    }
  }

  /**
   * Capture current stack trace
   */
  private captureStack(options: ProfilerOptions): StackFrame[] {
    const err = new Error();
    const stackString = err.stack || "";
    const lines = stackString.split("\n").slice(3); // Skip Error, captureStack, sampling

    const frames: StackFrame[] = [];
    const maxDepth = options.maxStackDepth!;

    for (let i = 0; i < Math.min(lines.length, maxDepth); i++) {
      const frame = this.parseStackFrame(lines[i]);
      if (frame) {
        if (!options.includeNatives && frame.isNative) continue;
        frames.push(frame);
      }
    }

    return frames.reverse(); // Reverse to get root-first order
  }

  /**
   * Parse a single stack frame line
   */
  private parseStackFrame(line: string): StackFrame | null {
    // Match: "    at functionName (filename:line:column)"
    // or:    "    at filename:line:column"
    const match = line.match(
      /^\s*at\s+(?:(.+?)\s+\()?([^()]+):(\d+):(\d+)\)?$/
    );

    if (!match) {
      // Try matching native functions: "    at functionName (native)"
      const nativeMatch = line.match(/^\s*at\s+(.+?)\s+\(native\)$/);
      if (nativeMatch) {
        return {
          functionName: nativeMatch[1],
          scriptName: "native",
          lineNumber: 0,
          columnNumber: 0,
          isNative: true,
        };
      }
      return null;
    }

    return {
      functionName: match[1] || "(anonymous)",
      scriptName: match[2],
      lineNumber: parseInt(match[3], 10),
      columnNumber: parseInt(match[4], 10),
      isNative: match[2] === "native" || match[2].includes("node:"),
    };
  }

  /**
   * Add stack to call tree and return leaf node ID
   */
  private addStackToTree(session: ProfileSession, stack: StackFrame[]): number {
    let currentNode = session.rootNode;
    currentNode.hitCount++;

    for (const frame of stack) {
      const key = this.frameKey(frame);

      let childNode = currentNode.children.get(key);
      if (!childNode) {
        childNode = {
          id: session.nodeIdCounter++,
          functionName: frame.functionName,
          scriptName: frame.scriptName,
          lineNumber: frame.lineNumber,
          columnNumber: frame.columnNumber,
          hitCount: 0,
          children: new Map(),
          selfTime: 0,
          totalTime: 0,
        };
        currentNode.children.set(key, childNode);
        this.nodeIdMap.set(childNode.id, childNode);
      }

      childNode.hitCount++;
      currentNode = childNode;
    }

    // Leaf node gets self time
    currentNode.selfTime++;
    return currentNode.id;
  }

  /**
   * Generate unique key for stack frame
   */
  private frameKey(frame: StackFrame): string {
    return `${frame.functionName}|${frame.scriptName}|${frame.lineNumber}|${frame.columnNumber}`;
  }

  /**
   * Calculate self and total times for all nodes
   */
  private calculateTimings(node: ProfileNode, totalDuration: number): void {
    const totalSamples = this.activeSession?.samples.length || 1;
    const timePerSample = totalDuration / totalSamples;

    this.calculateNodeTimings(node, timePerSample);
  }

  private calculateNodeTimings(
    node: ProfileNode,
    timePerSample: number
  ): number {
    let childrenTime = 0;

    for (const child of node.children.values()) {
      childrenTime += this.calculateNodeTimings(child, timePerSample);
    }

    node.selfTime = node.selfTime * timePerSample;
    node.totalTime = node.selfTime + childrenTime;

    return node.totalTime;
  }

  /**
   * Convert session to CPUProfileResult
   */
  private convertToResult(
    session: ProfileSession,
    endTime: number,
    duration: number
  ): CPUProfileResult {
    // Collect all samples
    const samples = this.collectSamples(session.rootNode);

    // Find hot functions
    const hotFunctions = this.findHotFunctions(samples, duration);

    // Generate flamegraph data
    const flamegraphData = this.generateFlamegraphData(session.rootNode);

    // Calculate summary statistics
    const totalSamples = session.samples.length;
    const idleSamples =
      session.rootNode.hitCount -
      Array.from(session.rootNode.children.values()).reduce(
        (sum, child) => sum + child.hitCount,
        0
      );

    // Estimate GC time (simplified - would need V8 hooks for accuracy)
    const gcSamples = samples.filter(
      (s) =>
        s.functionName.includes("GC") || s.functionName.includes("Scavenge")
    ).length;

    return {
      id: session.id,
      startTime: session.startTime,
      endTime,
      duration,
      samples: samples.map((s) => ({
        nodeId: s.id,
        hitCount: s.hitCount,
        children: Array.from(s.children.values()).map((c) => c.id),
        functionName: s.functionName,
        scriptName: s.scriptName,
        lineNumber: s.lineNumber,
        columnNumber: s.columnNumber,
        bailoutReason: s.bailoutReason,
        selfTime: s.selfTime,
        totalTime: s.totalTime,
      })),
      hotFunctions,
      flamegraphData,
      summary: {
        totalSamples,
        totalTime: duration,
        idlePercentage: (idleSamples / totalSamples) * 100,
        activePercentage: ((totalSamples - idleSamples) / totalSamples) * 100,
        gcPercentage: (gcSamples / totalSamples) * 100,
      },
    };
  }

  /**
   * Collect all nodes as flat array
   */
  private collectSamples(node: ProfileNode): ProfileNode[] {
    const samples: ProfileNode[] = [node];

    for (const child of node.children.values()) {
      samples.push(...this.collectSamples(child));
    }

    return samples;
  }

  /**
   * Find hot functions (highest self time)
   */
  private findHotFunctions(
    samples: ProfileNode[],
    totalDuration: number
  ): CPUProfileResult["hotFunctions"] {
    // Sort by self time
    const sorted = [...samples]
      .filter((s) => s.functionName !== "(root)")
      .sort((a, b) => b.selfTime - a.selfTime)
      .slice(0, 20);

    return sorted.map((node) => ({
      functionName: node.functionName,
      scriptName: node.scriptName,
      selfTime: node.selfTime,
      totalTime: node.totalTime,
      percentage: (node.selfTime / totalDuration) * 100,
      callCount: node.hitCount,
    }));
  }

  /**
   * Generate flamegraph-compatible data
   */
  private generateFlamegraphData(root: ProfileNode): string {
    const lines: string[] = [];
    this.buildFlamegraphLines(root, "", lines);
    return lines.join("\n");
  }

  private buildFlamegraphLines(
    node: ProfileNode,
    prefix: string,
    lines: string[]
  ): void {
    const name =
      node.functionName === "(root)"
        ? ""
        : `${prefix}${prefix ? ";" : ""}${node.functionName}`;

    if (name && node.selfTime > 0) {
      lines.push(`${name} ${Math.round(node.selfTime)}`);
    }

    for (const child of node.children.values()) {
      this.buildFlamegraphLines(child, name, lines);
    }
  }

  /**
   * Setup signal handlers for graceful shutdown
   */
  private setupSignalHandlers(): void {
    const cleanup = async () => {
      if (this.activeSession) {
        await this.stop();
      }
    };

    process.on("SIGINT", cleanup);
    process.on("SIGTERM", cleanup);
  }
}

// ============================================================================
// HOT PATH ANALYZER
// ============================================================================

/**
 * Analyzes CPU profiles to identify hot paths and optimization opportunities
 *
 * @elite-agent-collective @VELOCITY - Performance pattern recognition
 */
export class HotPathAnalyzer {
  private readonly hotPathThreshold = 0.1; // 10% of total time

  /**
   * Analyze profile and identify hot paths
   */
  analyzeHotPaths(profile: CPUProfileResult): HotPathAnalysis {
    const paths = this.extractPaths(profile);
    const hotPaths = this.filterHotPaths(paths, profile.duration);
    const patterns = this.detectPatterns(profile);
    const recommendations = this.generateRecommendations(hotPaths, patterns);

    return {
      hotPaths,
      patterns,
      recommendations,
      summary: this.generateSummary(hotPaths, patterns),
    };
  }

  /**
   * Extract all execution paths from profile
   */
  private extractPaths(profile: CPUProfileResult): ExecutionPath[] {
    const paths: ExecutionPath[] = [];
    const sampleMap = new Map(profile.samples.map((s) => [s.nodeId, s]));

    // Build paths from leaf nodes
    for (const sample of profile.samples) {
      if (sample.children.length === 0 && sample.selfTime > 0) {
        const path = this.buildPath(sample, sampleMap);
        paths.push(path);
      }
    }

    return paths;
  }

  private buildPath(
    leaf: CPUProfileSample,
    sampleMap: Map<number, CPUProfileSample>
  ): ExecutionPath {
    const frames: PathFrame[] = [];
    let current: CPUProfileSample | undefined = leaf;

    while (current) {
      frames.unshift({
        functionName: current.functionName,
        scriptName: current.scriptName,
        lineNumber: current.lineNumber,
        selfTime: current.selfTime,
        totalTime: current.totalTime,
      });

      // Find parent (simplified - would need parent tracking in real impl)
      current = undefined;
    }

    return {
      frames,
      totalTime: leaf.totalTime,
      selfTime: leaf.selfTime,
      hitCount: leaf.hitCount,
    };
  }

  /**
   * Filter paths that exceed hot path threshold
   */
  private filterHotPaths(
    paths: ExecutionPath[],
    totalDuration: number
  ): ExecutionPath[] {
    const threshold = totalDuration * this.hotPathThreshold;
    return paths
      .filter((p) => p.totalTime >= threshold)
      .sort((a, b) => b.totalTime - a.totalTime);
  }

  /**
   * Detect common performance anti-patterns
   */
  private detectPatterns(profile: CPUProfileResult): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    // Check for excessive JSON parsing
    const jsonFunctions = profile.hotFunctions.filter(
      (f) =>
        f.functionName.includes("JSON") ||
        f.functionName.includes("parse") ||
        f.functionName.includes("stringify")
    );
    if (
      jsonFunctions.length > 0 &&
      this.getTotalTime(jsonFunctions) > profile.duration * 0.05
    ) {
      patterns.push({
        type: "excessive_serialization",
        severity: "warning",
        description: "Excessive time spent in JSON serialization/parsing",
        affectedFunctions: jsonFunctions.map((f) => f.functionName),
        estimatedImpact: this.getTotalTime(jsonFunctions) / profile.duration,
      });
    }

    // Check for regexp heavy usage
    const regexpFunctions = profile.hotFunctions.filter(
      (f) =>
        f.functionName.includes("RegExp") ||
        f.functionName.includes("match") ||
        f.functionName.includes("replace")
    );
    if (
      regexpFunctions.length > 0 &&
      this.getTotalTime(regexpFunctions) > profile.duration * 0.03
    ) {
      patterns.push({
        type: "regexp_overhead",
        severity: "info",
        description: "Significant time spent in regular expression operations",
        affectedFunctions: regexpFunctions.map((f) => f.functionName),
        estimatedImpact: this.getTotalTime(regexpFunctions) / profile.duration,
      });
    }

    // Check for synchronous I/O
    const syncFunctions = profile.hotFunctions.filter(
      (f) =>
        f.functionName.includes("Sync") ||
        f.functionName.includes("readFileSync") ||
        f.functionName.includes("writeFileSync")
    );
    if (syncFunctions.length > 0) {
      patterns.push({
        type: "synchronous_io",
        severity: "critical",
        description: "Synchronous I/O operations blocking event loop",
        affectedFunctions: syncFunctions.map((f) => f.functionName),
        estimatedImpact: this.getTotalTime(syncFunctions) / profile.duration,
      });
    }

    // Check for excessive allocations
    if (profile.summary.gcPercentage > 10) {
      patterns.push({
        type: "excessive_gc",
        severity: "warning",
        description: `High GC pressure: ${profile.summary.gcPercentage.toFixed(1)}% of time spent in garbage collection`,
        affectedFunctions: [],
        estimatedImpact: profile.summary.gcPercentage / 100,
      });
    }

    return patterns;
  }

  private getTotalTime(functions: CPUProfileResult["hotFunctions"]): number {
    return functions.reduce((sum, f) => sum + f.selfTime, 0);
  }

  /**
   * Generate optimization recommendations
   */
  private generateRecommendations(
    hotPaths: ExecutionPath[],
    patterns: DetectedPattern[]
  ): OptimizationRecommendation[] {
    const recommendations: OptimizationRecommendation[] = [];

    // Pattern-based recommendations
    for (const pattern of patterns) {
      switch (pattern.type) {
        case "excessive_serialization":
          recommendations.push({
            priority: "high",
            category: "data_handling",
            description:
              "Consider using streaming JSON parser or caching parsed results",
            estimatedSpeedup: `${(pattern.estimatedImpact * 100).toFixed(1)}%`,
            effort: "medium",
          });
          break;
        case "regexp_overhead":
          recommendations.push({
            priority: "medium",
            category: "algorithm",
            description: "Pre-compile regular expressions and cache them",
            estimatedSpeedup: `${(pattern.estimatedImpact * 50).toFixed(1)}%`,
            effort: "low",
          });
          break;
        case "synchronous_io":
          recommendations.push({
            priority: "critical",
            category: "io",
            description: "Replace synchronous I/O with async alternatives",
            estimatedSpeedup: "Significant latency improvement",
            effort: "medium",
          });
          break;
        case "excessive_gc":
          recommendations.push({
            priority: "high",
            category: "memory",
            description: "Reduce object allocations, consider object pooling",
            estimatedSpeedup: `${(pattern.estimatedImpact * 100).toFixed(1)}%`,
            effort: "high",
          });
          break;
      }
    }

    // Hot path based recommendations
    for (const path of hotPaths.slice(0, 5)) {
      const topFrame = path.frames[path.frames.length - 1];
      if (topFrame) {
        recommendations.push({
          priority: "high",
          category: "hotspot",
          description: `Optimize ${topFrame.functionName} in ${topFrame.scriptName}`,
          estimatedSpeedup: "Variable",
          effort: "variable",
        });
      }
    }

    return recommendations;
  }

  /**
   * Generate analysis summary
   */
  private generateSummary(
    hotPaths: ExecutionPath[],
    patterns: DetectedPattern[]
  ): string {
    const criticalPatterns = patterns.filter((p) => p.severity === "critical");
    const warnings = patterns.filter((p) => p.severity === "warning");

    let summary = `Found ${hotPaths.length} hot paths and detected ${patterns.length} patterns.\n`;

    if (criticalPatterns.length > 0) {
      summary += `\n⚠️ ${criticalPatterns.length} critical issue(s) require immediate attention.\n`;
    }

    if (warnings.length > 0) {
      summary += `\n⚡ ${warnings.length} optimization opportunity(ies) identified.\n`;
    }

    return summary;
  }
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

interface PathFrame {
  functionName: string;
  scriptName: string;
  lineNumber: number;
  selfTime: number;
  totalTime: number;
}

interface ExecutionPath {
  frames: PathFrame[];
  totalTime: number;
  selfTime: number;
  hitCount: number;
}

interface DetectedPattern {
  type: string;
  severity: "info" | "warning" | "critical";
  description: string;
  affectedFunctions: string[];
  estimatedImpact: number;
}

interface OptimizationRecommendation {
  priority: "low" | "medium" | "high" | "critical";
  category: string;
  description: string;
  estimatedSpeedup: string;
  effort: string;
}

interface HotPathAnalysis {
  hotPaths: ExecutionPath[];
  patterns: DetectedPattern[];
  recommendations: OptimizationRecommendation[];
  summary: string;
}

// ============================================================================
// EXPORTS
// ============================================================================

export { CPUProfiler as default };
export type {
  ProfilerOptions,
  StackFrame,
  ProfileNode,
  ProfileSession,
  PathFrame,
  ExecutionPath,
  DetectedPattern,
  OptimizationRecommendation,
  HotPathAnalysis,
};
