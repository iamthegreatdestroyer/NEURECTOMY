/**
 * @fileoverview Physics Engine Controller - Main thread interface for physics worker
 * @module @neurectomy/3d-engine/workers/physics-controller
 *
 * Manages communication with the physics Web Worker, handling data transfer
 * using transferable ArrayBuffers for zero-copy performance.
 */

import type { GraphNode, GraphEdge } from "../visualization/graph/types";
import type { Vector3 } from "three";

/**
 * Configuration for the physics simulation
 */
export interface PhysicsConfig {
  /** Repulsion force strength between nodes */
  repulsion: number;
  /** Attraction force strength for connected nodes */
  attraction: number;
  /** Velocity damping factor (0-1) */
  damping: number;
  /** Barnes-Hut approximation threshold */
  theta: number;
  /** Maximum node velocity */
  maxVelocity: number;
  /** Gravity toward center strength */
  gravityCenterStrength: number;
  /** Collision radius for nodes */
  collisionRadius: number;
  /** Simulation timestep */
  timestep: number;
}

/**
 * Default physics configuration
 */
export const DEFAULT_PHYSICS_CONFIG: PhysicsConfig = {
  repulsion: 10000,
  attraction: 0.001,
  damping: 0.9,
  theta: 0.5,
  maxVelocity: 50,
  gravityCenterStrength: 0.0001,
  collisionRadius: 10,
  timestep: 1,
};

/**
 * Result from a physics simulation step
 */
export interface PhysicsStepResult {
  /** Updated node positions */
  positions: Map<string, Vector3>;
  /** Total system energy */
  energy: number;
  /** Whether simulation has converged */
  converged: boolean;
  /** Number of iterations actually run */
  iterationsRun: number;
}

/**
 * Physics simulation state
 */
export type PhysicsState = "idle" | "running" | "paused" | "stopped";

/**
 * Event types emitted by the physics controller
 */
export interface PhysicsControllerEvents {
  step: PhysicsStepResult;
  converged: void;
  error: Error;
  "state-change": PhysicsState;
}

/**
 * Physics Engine Controller
 *
 * Manages a Web Worker for physics calculations, providing a high-level
 * API for force-directed graph layout simulation.
 *
 * @example
 * ```typescript
 * const controller = new PhysicsController();
 * await controller.initialize(nodes, edges);
 *
 * controller.on('step', (result) => {
 *   updateNodePositions(result.positions);
 * });
 *
 * controller.start();
 * ```
 */
export class PhysicsController {
  private worker: Worker | null = null;
  private state: PhysicsState = "idle";
  private nodes: GraphNode[] = [];
  private nodeIdToIndex: Map<string, number> = new Map();
  private edges: GraphEdge[] = [];
  private config: PhysicsConfig;
  private animationFrameId: number | null = null;
  private iterationsPerFrame = 3;
  private lastStepTime = 0;
  private targetFps = 60;

  // Event listeners
  private stepListeners: Set<(result: PhysicsStepResult) => void> = new Set();
  private convergedListeners: Set<() => void> = new Set();
  private errorListeners: Set<(error: Error) => void> = new Set();
  private stateChangeListeners: Set<(state: PhysicsState) => void> = new Set();

  constructor(config: Partial<PhysicsConfig> = {}) {
    this.config = { ...DEFAULT_PHYSICS_CONFIG, ...config };
  }

  /**
   * Initialize the physics simulation with nodes and edges
   */
  async initialize(nodes: GraphNode[], edges: GraphEdge[]): Promise<void> {
    // Terminate existing worker
    if (this.worker) {
      this.worker.terminate();
    }

    this.nodes = nodes;
    this.edges = edges;

    // Build node ID to index mapping
    this.nodeIdToIndex.clear();
    nodes.forEach((node, index) => {
      this.nodeIdToIndex.set(node.id, index);
    });

    // Create worker
    this.worker = new Worker(new URL("./physics-worker.ts", import.meta.url), {
      type: "module",
    });

    // Set up message handler
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
    this.worker.onerror = (event) => {
      this.emitError(new Error(`Worker error: ${event.message}`));
    };

    // Initialize worker
    await this.sendInit();

    // Send initial data
    await this.sendNodeData();
    await this.sendEdgeData();

    this.setState("idle");
  }

  /**
   * Start the physics simulation
   */
  start(): void {
    if (this.state === "running") return;
    if (!this.worker) {
      this.emitError(new Error("Physics not initialized"));
      return;
    }

    this.setState("running");
    this.scheduleStep();
  }

  /**
   * Pause the physics simulation
   */
  pause(): void {
    if (this.state !== "running") return;

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    this.worker?.postMessage({ type: "pause" });
    this.setState("paused");
  }

  /**
   * Resume the physics simulation
   */
  resume(): void {
    if (this.state !== "paused") return;

    this.worker?.postMessage({ type: "resume" });
    this.setState("running");
    this.scheduleStep();
  }

  /**
   * Stop and clean up the physics simulation
   */
  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.worker) {
      this.worker.postMessage({ type: "stop" });
      this.worker.terminate();
      this.worker = null;
    }

    this.setState("stopped");
  }

  /**
   * Update physics configuration
   */
  updateConfig(config: Partial<PhysicsConfig>): void {
    this.config = { ...this.config, ...config };
    this.worker?.postMessage({ type: "config", config });
  }

  /**
   * Update nodes in the simulation
   */
  async updateNodes(nodes: GraphNode[]): Promise<void> {
    this.nodes = nodes;

    // Rebuild index mapping
    this.nodeIdToIndex.clear();
    nodes.forEach((node, index) => {
      this.nodeIdToIndex.set(node.id, index);
    });

    // Re-initialize if worker exists
    if (this.worker) {
      await this.sendInit();
      await this.sendNodeData();
      await this.sendEdgeData();
    }
  }

  /**
   * Update edges in the simulation
   */
  async updateEdges(edges: GraphEdge[]): Promise<void> {
    this.edges = edges;

    if (this.worker) {
      await this.sendEdgeData();
    }
  }

  /**
   * Fix a node in place (won't be moved by physics)
   */
  fixNode(nodeId: string, fixed = true): void {
    const node = this.nodes.find((n) => n.id === nodeId);
    if (node) {
      (node as any).fixed = fixed;
      this.sendNodeData();
    }
  }

  /**
   * Set node position directly
   */
  setNodePosition(
    nodeId: string,
    position: { x: number; y: number; z: number }
  ): void {
    const node = this.nodes.find((n) => n.id === nodeId);
    if (node) {
      node.position.x = position.x;
      node.position.y = position.y;
      node.position.z = position.z;
      this.sendNodeData();
    }
  }

  /**
   * Run a single step of the simulation synchronously
   */
  async stepOnce(iterations = 1): Promise<PhysicsStepResult | null> {
    if (!this.worker) return null;

    return new Promise((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === "step-result") {
          this.worker!.removeEventListener("message", handler);
          const result = this.processStepResult(event.data);
          resolve(result);
        }
      };

      this.worker!.addEventListener("message", handler);
      this.worker!.postMessage({
        type: "step",
        iterations,
        returnVelocities: false,
      });
    });
  }

  /**
   * Get current physics state
   */
  getState(): PhysicsState {
    return this.state;
  }

  /**
   * Get current configuration
   */
  getConfig(): PhysicsConfig {
    return { ...this.config };
  }

  // Event subscription methods
  on<K extends keyof PhysicsControllerEvents>(
    event: K,
    listener: (data: PhysicsControllerEvents[K]) => void
  ): () => void {
    switch (event) {
      case "step":
        this.stepListeners.add(listener as (result: PhysicsStepResult) => void);
        return () =>
          this.stepListeners.delete(
            listener as (result: PhysicsStepResult) => void
          );
      case "converged":
        this.convergedListeners.add(listener as () => void);
        return () => this.convergedListeners.delete(listener as () => void);
      case "error":
        this.errorListeners.add(listener as (error: Error) => void);
        return () =>
          this.errorListeners.delete(listener as (error: Error) => void);
      case "state-change":
        this.stateChangeListeners.add(
          listener as (state: PhysicsState) => void
        );
        return () =>
          this.stateChangeListeners.delete(
            listener as (state: PhysicsState) => void
          );
      default:
        return () => {};
    }
  }

  // Private methods

  private setState(state: PhysicsState): void {
    if (this.state !== state) {
      this.state = state;
      this.stateChangeListeners.forEach((listener) => listener(state));
    }
  }

  private emitError(error: Error): void {
    this.errorListeners.forEach((listener) => listener(error));
  }

  private async sendInit(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error("Worker not created"));
        return;
      }

      const handler = (event: MessageEvent) => {
        if (event.data.type === "ready") {
          this.worker!.removeEventListener("message", handler);
          resolve();
        } else if (event.data.type === "error") {
          this.worker!.removeEventListener("message", handler);
          reject(new Error(event.data.message));
        }
      };

      this.worker.addEventListener("message", handler);
      this.worker.postMessage({
        type: "init",
        nodeCount: this.nodes.length,
        edgeCount: this.edges.length,
        config: this.config,
      });
    });
  }

  private async sendNodeData(): Promise<void> {
    if (!this.worker) return;

    // Pack node data: [x, y, z, vx, vy, vz, mass, fixed] per node
    const buffer = new ArrayBuffer(this.nodes.length * 8 * 4); // 8 floats per node
    const data = new Float32Array(buffer);

    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      const idx = i * 8;

      data[idx] = node.position.x;
      data[idx + 1] = node.position.y;
      data[idx + 2] = node.position.z;
      data[idx + 3] = node.velocity?.x ?? 0;
      data[idx + 4] = node.velocity?.y ?? 0;
      data[idx + 5] = node.velocity?.z ?? 0;
      data[idx + 6] = (node as any).mass ?? 1;
      data[idx + 7] = (node as any).fixed ? 1 : 0;
    }

    this.worker.postMessage(
      {
        type: "update-nodes",
        positions: buffer,
        nodeCount: this.nodes.length,
      },
      [buffer]
    );
  }

  private async sendEdgeData(): Promise<void> {
    if (!this.worker || this.edges.length === 0) return;

    // Pack edge data:
    // First part: [sourceIndex, targetIndex] as uint32 pairs
    // Second part: [strength, idealLength] as float32 pairs
    const indexBytes = this.edges.length * 2 * 4;
    const floatBytes = this.edges.length * 2 * 4;
    const buffer = new ArrayBuffer(indexBytes + floatBytes);
    const indexView = new Uint32Array(buffer, 0, this.edges.length * 2);
    const floatView = new Float32Array(
      buffer,
      indexBytes,
      this.edges.length * 2
    );

    for (let i = 0; i < this.edges.length; i++) {
      const edge = this.edges[i];
      const sourceIndex = this.nodeIdToIndex.get(edge.sourceId);
      const targetIndex = this.nodeIdToIndex.get(edge.targetId);

      if (sourceIndex === undefined || targetIndex === undefined) {
        continue;
      }

      indexView[i * 2] = sourceIndex;
      indexView[i * 2 + 1] = targetIndex;

      floatView[i * 2] = (edge as any).strength ?? 1;
      floatView[i * 2 + 1] = (edge as any).idealLength ?? 100;
    }

    this.worker.postMessage(
      {
        type: "update-edges",
        edges: buffer,
        edgeCount: this.edges.length,
      },
      [buffer]
    );
  }

  private handleWorkerMessage(event: MessageEvent): void {
    const data = event.data;

    switch (data.type) {
      case "step-result":
        if (this.state === "running") {
          const result = this.processStepResult(data);
          this.stepListeners.forEach((listener) => listener(result));

          if (result.converged) {
            this.convergedListeners.forEach((listener) => listener());
            this.pause();
          } else {
            this.scheduleStep();
          }
        }
        break;

      case "error":
        this.emitError(new Error(data.message));
        break;
    }
  }

  private processStepResult(data: {
    positions: ArrayBuffer;
    velocities?: ArrayBuffer;
    energy: number;
    converged: boolean;
    iterationsRun: number;
  }): PhysicsStepResult {
    const positionData = new Float32Array(data.positions);
    const positions = new Map<string, Vector3>();

    // Update local nodes and build result map
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      const idx = i * 3;

      node.position.x = positionData[idx];
      node.position.y = positionData[idx + 1];
      node.position.z = positionData[idx + 2];

      positions.set(node.id, { ...node.position } as Vector3);
    }

    return {
      positions,
      energy: data.energy,
      converged: data.converged,
      iterationsRun: data.iterationsRun,
    };
  }

  private scheduleStep(): void {
    if (this.state !== "running") return;

    const now = performance.now();
    const elapsed = now - this.lastStepTime;
    const targetFrameTime = 1000 / this.targetFps;

    if (elapsed >= targetFrameTime) {
      this.lastStepTime = now;
      this.worker?.postMessage({
        type: "step",
        iterations: this.iterationsPerFrame,
        returnVelocities: false,
      });
    } else {
      this.animationFrameId = requestAnimationFrame(() => this.scheduleStep());
    }
  }
}

/**
 * Create a physics controller instance
 */
export function createPhysicsController(
  config?: Partial<PhysicsConfig>
): PhysicsController {
  return new PhysicsController(config);
}
