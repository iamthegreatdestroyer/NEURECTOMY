/**
 * Hybrid Force Layout - GPU + CPU Force-Directed Graph Layout
 *
 * Automatically selects optimal compute backend:
 * - WebGPU: 100K+ nodes (if available)
 * - CPU Barnes-Hut: 10K-100K nodes
 * - CPU Naive: <10K nodes
 *
 * @module @neurectomy/3d-engine/visualization/graph/layouts
 * @agents @VELOCITY @APEX @ARCHITECT
 */

import type { GraphNode, GraphEdge } from "../types";
import {
  ForceDirectedLayout,
  ForceLayoutConfig,
  SimulationState,
  DEFAULT_FORCE_CONFIG,
} from "./force-directed";
import {
  createGPUForceLayout,
  GPUForceLayout,
  GPUForceLayoutConfig,
  GPUForceLayoutStats,
} from "../../../webgpu/compute";

// ============================================================================
// Types
// ============================================================================

export type ComputeBackend = "gpu" | "cpu-barnes-hut" | "cpu-naive" | "auto";

export interface HybridForceLayoutConfig extends ForceLayoutConfig {
  /** Preferred compute backend */
  backend: ComputeBackend;
  /** Node count threshold for GPU usage */
  gpuThreshold: number;
  /** Node count threshold for Barnes-Hut vs naive */
  barnesHutThreshold: number;
  /** Enable automatic backend switching based on performance */
  adaptiveBackend: boolean;
}

export interface HybridSimulationState extends SimulationState {
  /** Current active backend */
  backend: ComputeBackend;
  /** GPU statistics (if using GPU) */
  gpuStats?: GPUForceLayoutStats;
  /** Iterations per second */
  iterationsPerSecond: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_HYBRID_CONFIG: HybridForceLayoutConfig = {
  ...DEFAULT_FORCE_CONFIG,
  backend: "auto",
  gpuThreshold: 5000, // Use GPU for 5K+ nodes
  barnesHutThreshold: 1000, // Use Barnes-Hut for 1K+ nodes
  adaptiveBackend: true,
};

// ============================================================================
// Hybrid Force Layout Engine
// ============================================================================

/**
 * Hybrid force-directed layout engine with automatic backend selection
 */
export class HybridForceLayout {
  private config: HybridForceLayoutConfig;
  private state: HybridSimulationState;

  // Backend instances
  private cpuLayout: ForceDirectedLayout | null = null;
  private gpuLayout: GPUForceLayout | null = null;

  // Data
  private nodes: GraphNode[] = [];
  private edges: GraphEdge[] = [];

  // State
  private activeBackend: ComputeBackend = "cpu-naive";
  private isInitialized = false;
  private animationFrameId: number | null = null;

  // Performance tracking
  private iterationTimes: number[] = [];
  private lastSyncTime = 0;

  // Callbacks
  private onTick?: (state: HybridSimulationState) => void;
  private onEnd?: () => void;
  private onBackendChange?: (backend: ComputeBackend) => void;

  constructor(config: Partial<HybridForceLayoutConfig> = {}) {
    this.config = { ...DEFAULT_HYBRID_CONFIG, ...config };
    this.state = {
      alpha: this.config.alphaTarget,
      running: false,
      tickCount: 0,
      energy: 0,
      backend: "auto",
      iterationsPerSecond: 0,
    };
  }

  // ========================================================================
  // Initialization
  // ========================================================================

  async initialize(nodes: GraphNode[], edges: GraphEdge[]): Promise<void> {
    this.nodes = nodes;
    this.edges = edges;

    // Select optimal backend
    this.activeBackend = await this.selectBackend(nodes.length);
    this.state.backend = this.activeBackend;

    // Initialize selected backend
    if (this.activeBackend === "gpu") {
      await this.initializeGPU();
    } else {
      this.initializeCPU();
    }

    this.isInitialized = true;
    console.log(
      `HybridForceLayout initialized with ${this.activeBackend} backend for ${nodes.length} nodes`
    );
  }

  private async selectBackend(nodeCount: number): Promise<ComputeBackend> {
    if (this.config.backend !== "auto") {
      // User explicitly requested a backend
      if (this.config.backend === "gpu") {
        // Verify GPU is available
        const gpuAvailable = await this.checkGPUAvailability();
        if (!gpuAvailable) {
          console.warn(
            "GPU backend requested but not available, falling back to CPU"
          );
          return nodeCount >= this.config.barnesHutThreshold
            ? "cpu-barnes-hut"
            : "cpu-naive";
        }
        return "gpu";
      }
      return this.config.backend;
    }

    // Auto-select based on node count and GPU availability
    if (nodeCount >= this.config.gpuThreshold) {
      const gpuAvailable = await this.checkGPUAvailability();
      if (gpuAvailable) {
        return "gpu";
      }
    }

    return nodeCount >= this.config.barnesHutThreshold
      ? "cpu-barnes-hut"
      : "cpu-naive";
  }

  private async checkGPUAvailability(): Promise<boolean> {
    if (!navigator.gpu) {
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  private async initializeGPU(): Promise<void> {
    // Convert config to GPU format
    const gpuConfig: Partial<GPUForceLayoutConfig> = {
      alpha: this.config.alphaTarget,
      alphaDecay: this.config.alphaDecay,
      alphaMin: this.config.alphaMin,
      velocityDecay: this.config.velocityDecay,
      chargeStrength: this.config.forces.charge,
      chargeDistanceMin: this.config.forces.chargeDistanceMin,
      chargeDistanceMax: this.config.forces.chargeDistanceMax,
      centerStrength: this.config.forces.center,
      linkStrength: this.config.forces.link,
      linkDistance: this.config.forces.linkDistance,
      collisionRadiusMult: this.config.forces.collision,
      theta: this.config.theta,
      is3D: this.config.is3D,
    };

    this.gpuLayout = await createGPUForceLayout(gpuConfig);

    if (!this.gpuLayout) {
      console.warn("Failed to create GPU layout, falling back to CPU");
      this.activeBackend =
        this.nodes.length >= this.config.barnesHutThreshold
          ? "cpu-barnes-hut"
          : "cpu-naive";
      this.state.backend = this.activeBackend;
      this.initializeCPU();
      return;
    }

    // Convert nodes to GPU format
    this.gpuLayout.setData(this.nodes, this.edges);
  }

  private initializeCPU(): void {
    // Configure CPU layout
    const cpuConfig: Partial<ForceLayoutConfig> = {
      ...this.config,
      useBarnesHut: this.activeBackend === "cpu-barnes-hut",
    };

    this.cpuLayout = new ForceDirectedLayout(cpuConfig);
    this.cpuLayout.initialize(this.nodes, this.edges);
  }

  // ========================================================================
  // Simulation Control
  // ========================================================================

  start(
    onTick?: (state: HybridSimulationState) => void,
    onEnd?: () => void,
    onBackendChange?: (backend: ComputeBackend) => void
  ): void {
    if (!this.isInitialized) {
      console.error("HybridForceLayout not initialized");
      return;
    }

    this.onTick = onTick;
    this.onEnd = onEnd;
    this.onBackendChange = onBackendChange;

    this.state.running = true;
    this.state.alpha = this.config.alphaTarget;
    this.state.tickCount = 0;
    this.iterationTimes = [];

    if (this.activeBackend === "gpu" && this.gpuLayout) {
      this.gpuLayout.start();
    }

    this.runSimulation();
  }

  stop(): void {
    this.state.running = false;

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.gpuLayout) {
      this.gpuLayout.stop();
    }

    if (this.cpuLayout) {
      this.cpuLayout.stop();
    }
  }

  private async runSimulation(): Promise<void> {
    const step = async () => {
      if (!this.state.running) return;

      const startTime = performance.now();

      try {
        // Run simulation tick
        const shouldContinue = await this.tick();

        // Track timing
        const iterationTime = performance.now() - startTime;
        this.iterationTimes.push(iterationTime);
        if (this.iterationTimes.length > 60) {
          this.iterationTimes.shift();
        }

        // Calculate IPS
        const avgTime =
          this.iterationTimes.reduce((a, b) => a + b, 0) /
          this.iterationTimes.length;
        this.state.iterationsPerSecond = avgTime > 0 ? 1000 / avgTime : 0;

        if (!shouldContinue) {
          this.state.running = false;

          // Final sync
          if (this.activeBackend === "gpu") {
            await this.syncFromGPU();
          }

          this.onEnd?.();
          return;
        }

        // Update GPU stats
        if (this.activeBackend === "gpu" && this.gpuLayout) {
          this.state.gpuStats = this.gpuLayout.getStats();
        }

        this.onTick?.(this.state);
        this.animationFrameId = requestAnimationFrame(() => step());
      } catch (error) {
        console.error("Simulation error:", error);
        this.state.running = false;
        this.onEnd?.();
      }
    };

    this.animationFrameId = requestAnimationFrame(() => step());
  }

  private async tick(): Promise<boolean> {
    if (this.activeBackend === "gpu" && this.gpuLayout) {
      const shouldContinue = this.gpuLayout.tick();
      this.state.alpha = this.gpuLayout.getAlpha();
      this.state.tickCount++;

      // Sync positions periodically for visualization
      const now = performance.now();
      if (now - this.lastSyncTime > 16) {
        // ~60fps sync
        await this.syncFromGPU();
        this.lastSyncTime = now;
      }

      return shouldContinue;
    } else if (this.cpuLayout) {
      const shouldContinue = this.cpuLayout.tick();
      this.state.alpha = this.cpuLayout.getState().alpha;
      this.state.tickCount = this.cpuLayout.getState().tickCount;
      this.state.energy = this.cpuLayout.getState().energy;
      return shouldContinue;
    }

    return false;
  }

  private async syncFromGPU(): Promise<void> {
    if (!this.gpuLayout) return;

    try {
      const positions = await this.gpuLayout.readPositions();

      // Update node positions
      for (let i = 0; i < this.nodes.length; i++) {
        this.nodes[i]!.position.x = positions[i * 3]!;
        this.nodes[i]!.position.y = positions[i * 3 + 1]!;
        this.nodes[i]!.position.z = positions[i * 3 + 2]!;
      }
    } catch (error) {
      console.error("Failed to sync from GPU:", error);
    }
  }

  // ========================================================================
  // Backend Switching
  // ========================================================================

  async switchBackend(backend: ComputeBackend): Promise<void> {
    if (backend === this.activeBackend) return;

    const wasRunning = this.state.running;
    this.stop();

    // Sync current positions
    if (this.activeBackend === "gpu" && this.gpuLayout) {
      await this.syncFromGPU();
    }

    // Clean up old backend
    this.cleanupBackend();

    // Initialize new backend
    this.activeBackend = backend;
    this.state.backend = backend;

    if (backend === "gpu") {
      await this.initializeGPU();
    } else {
      this.initializeCPU();
    }

    this.onBackendChange?.(backend);

    // Resume if was running
    if (wasRunning) {
      this.start(this.onTick, this.onEnd, this.onBackendChange);
    }
  }

  private cleanupBackend(): void {
    if (this.gpuLayout) {
      this.gpuLayout.destroy();
      this.gpuLayout = null;
    }
    this.cpuLayout = null;
  }

  // ========================================================================
  // Public API
  // ========================================================================

  getState(): HybridSimulationState {
    return { ...this.state };
  }

  getNodes(): GraphNode[] {
    return this.nodes;
  }

  getEdges(): GraphEdge[] {
    return this.edges;
  }

  getActiveBackend(): ComputeBackend {
    return this.activeBackend;
  }

  isGPUAccelerated(): boolean {
    return this.activeBackend === "gpu";
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    backend: ComputeBackend;
    nodeCount: number;
    edgeCount: number;
    iterationsPerSecond: number;
    gpuMemoryUsage?: number;
    cpuBackend?: "barnes-hut" | "naive";
  } {
    const metrics: ReturnType<typeof this.getPerformanceMetrics> = {
      backend: this.activeBackend,
      nodeCount: this.nodes.length,
      edgeCount: this.edges.length,
      iterationsPerSecond: this.state.iterationsPerSecond,
    };

    if (this.activeBackend === "gpu" && this.gpuLayout) {
      const stats = this.gpuLayout.getStats();
      metrics.gpuMemoryUsage = stats.gpuMemoryUsage;
    } else {
      metrics.cpuBackend =
        this.activeBackend === "cpu-barnes-hut" ? "barnes-hut" : "naive";
    }

    return metrics;
  }

  /**
   * Pin/unpin a node
   */
  pinNode(nodeId: string, pinned: boolean): void {
    const node = this.nodes.find((n) => n.id === nodeId);
    if (node) {
      node.pinned = pinned;

      // Re-upload data if using GPU
      if (this.activeBackend === "gpu" && this.gpuLayout) {
        this.gpuLayout.setData(this.nodes, this.edges);
      }
    }
  }

  /**
   * Update node position (e.g., from drag)
   */
  setNodePosition(nodeId: string, x: number, y: number, z: number): void {
    const node = this.nodes.find((n) => n.id === nodeId);
    if (node) {
      node.position.x = x;
      node.position.y = y;
      node.position.z = z;

      // Re-upload data if using GPU
      if (this.activeBackend === "gpu" && this.gpuLayout) {
        this.gpuLayout.setData(this.nodes, this.edges);
      }
    }
  }

  /**
   * Reheat simulation
   */
  reheat(alpha?: number): void {
    const newAlpha = alpha ?? this.config.alphaTarget;
    this.state.alpha = newAlpha;

    if (this.activeBackend === "gpu" && this.gpuLayout) {
      this.gpuLayout.setAlpha(newAlpha);
    }

    if (!this.state.running) {
      this.start(this.onTick, this.onEnd, this.onBackendChange);
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<HybridForceLayoutConfig>): void {
    this.config = { ...this.config, ...config };

    if (this.gpuLayout) {
      this.gpuLayout.updateConfig({
        alphaDecay: this.config.alphaDecay,
        velocityDecay: this.config.velocityDecay,
        chargeStrength: this.config.forces.charge,
        chargeDistanceMin: this.config.forces.chargeDistanceMin,
        chargeDistanceMax: this.config.forces.chargeDistanceMax,
        centerStrength: this.config.forces.center,
        linkStrength: this.config.forces.link,
        linkDistance: this.config.forces.linkDistance,
        collisionRadiusMult: this.config.forces.collision,
        theta: this.config.theta,
        is3D: this.config.is3D,
      });
    }

    if (this.cpuLayout) {
      this.cpuLayout.updateConfig(config);
    }
  }

  // ========================================================================
  // Cleanup
  // ========================================================================

  destroy(): void {
    this.stop();
    this.cleanupBackend();
    this.nodes = [];
    this.edges = [];
    this.isInitialized = false;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export async function createHybridForceLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  config?: Partial<HybridForceLayoutConfig>
): Promise<HybridForceLayout> {
  const layout = new HybridForceLayout(config);
  await layout.initialize(nodes, edges);
  return layout;
}
