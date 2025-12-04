/**
 * @file Layout Manager
 * @description Unified layout manager for switching between layout algorithms
 * @module @neurectomy/3d-engine/visualization/graph/layouts/manager
 * @agents @VERTEX @ARCHITECT
 */

import type { GraphNode, GraphEdge } from "../types";
import { ForceDirectedLayout, type ForceLayoutConfig } from "./force-directed";
import { HierarchicalLayout, type HierarchicalConfig } from "./hierarchical";
import { RadialLayout, type RadialConfig } from "./radial";

// ============================================================================
// TYPES
// ============================================================================

export type LayoutType =
  | "force-directed"
  | "hierarchical"
  | "radial"
  | "grid"
  | "custom";

export interface LayoutManagerConfig {
  /** Default layout type */
  defaultLayout: LayoutType;
  /** Animation duration for layout transitions (ms) */
  transitionDuration: number;
  /** Enable layout animation */
  animate: boolean;
  /** Force-directed specific config */
  forceConfig?: Partial<ForceLayoutConfig>;
  /** Hierarchical specific config */
  hierarchicalConfig?: Partial<HierarchicalConfig>;
  /** Radial specific config */
  radialConfig?: Partial<RadialConfig>;
}

export interface LayoutResult {
  positions: Map<string, { x: number; y: number; z: number }>;
  bounds: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  metadata: Record<string, unknown>;
}

export interface LayoutTransition {
  from: Map<string, { x: number; y: number; z: number }>;
  to: Map<string, { x: number; y: number; z: number }>;
  progress: number;
  startTime: number;
  duration: number;
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

export const DEFAULT_LAYOUT_MANAGER_CONFIG: LayoutManagerConfig = {
  defaultLayout: "force-directed",
  transitionDuration: 500,
  animate: true,
};

// ============================================================================
// LAYOUT MANAGER
// ============================================================================

/**
 * Unified layout manager for graph visualization
 * Supports multiple layout algorithms with smooth transitions
 */
export class LayoutManager {
  private config: LayoutManagerConfig;
  private currentLayout: LayoutType;
  private forceLayout: ForceDirectedLayout;
  private hierarchicalLayout: HierarchicalLayout;
  private radialLayout: RadialLayout;
  private currentPositions: Map<string, { x: number; y: number; z: number }> =
    new Map();
  private transition: LayoutTransition | null = null;
  private animationFrame: number | null = null;

  constructor(config: Partial<LayoutManagerConfig> = {}) {
    this.config = { ...DEFAULT_LAYOUT_MANAGER_CONFIG, ...config };
    this.currentLayout = this.config.defaultLayout;

    // Initialize layout engines
    this.forceLayout = new ForceDirectedLayout(this.config.forceConfig);
    this.hierarchicalLayout = new HierarchicalLayout(
      this.config.hierarchicalConfig
    );
    this.radialLayout = new RadialLayout(this.config.radialConfig);
  }

  // ==========================================================================
  // PUBLIC API
  // ==========================================================================

  /**
   * Apply layout to graph
   */
  applyLayout(
    nodes: GraphNode[],
    edges: GraphEdge[],
    layoutType?: LayoutType
  ): LayoutResult {
    const type = layoutType ?? this.currentLayout;
    let result: LayoutResult;

    switch (type) {
      case "force-directed":
        result = this.applyForceDirected(nodes, edges);
        break;
      case "hierarchical":
        result = this.applyHierarchical(nodes, edges);
        break;
      case "radial":
        result = this.applyRadial(nodes, edges);
        break;
      case "grid":
        result = this.applyGrid(nodes);
        break;
      default:
        result = this.applyForceDirected(nodes, edges);
    }

    if (this.config.animate && this.currentPositions.size > 0) {
      this.startTransition(result.positions);
    } else {
      this.currentPositions = result.positions;
    }

    this.currentLayout = type;
    return result;
  }

  /**
   * Switch to a different layout with animation
   */
  switchLayout(
    nodes: GraphNode[],
    edges: GraphEdge[],
    newLayout: LayoutType
  ): void {
    if (newLayout === this.currentLayout) return;

    const result = this.applyLayout(nodes, edges, newLayout);
    if (this.config.animate) {
      this.startTransition(result.positions);
    }
  }

  /**
   * Get current positions
   */
  getPositions(): Map<string, { x: number; y: number; z: number }> {
    if (this.transition) {
      return this.interpolatePositions();
    }
    return new Map(this.currentPositions);
  }

  /**
   * Step simulation (for force-directed)
   */
  step(nodes: GraphNode[], edges: GraphEdge[]): boolean {
    if (this.currentLayout !== "force-directed") {
      return true; // Other layouts are static
    }

    this.forceLayout.initialize(nodes, edges);
    return this.forceLayout.tick();
  }

  /**
   * Get current layout type
   */
  getCurrentLayout(): LayoutType {
    return this.currentLayout;
  }

  /**
   * Check if transitioning
   */
  isTransitioning(): boolean {
    return this.transition !== null;
  }

  /**
   * Cancel ongoing transition
   */
  cancelTransition(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.transition) {
      this.currentPositions = this.transition.to;
      this.transition = null;
    }
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.cancelTransition();
    this.currentPositions.clear();
  }

  // ==========================================================================
  // LAYOUT IMPLEMENTATIONS
  // ==========================================================================

  private applyForceDirected(
    nodes: GraphNode[],
    edges: GraphEdge[]
  ): LayoutResult {
    this.forceLayout.initialize(nodes, edges);

    // Run simulation to stability
    let stable = false;
    let iterations = 0;
    const maxIterations = 300;

    while (!stable && iterations < maxIterations) {
      stable = this.forceLayout.tick();
      iterations++;
    }

    const positions = new Map<string, { x: number; y: number; z: number }>();
    const currentNodes = this.forceLayout.getNodes();

    for (const node of currentNodes) {
      positions.set(node.id, { ...node.position });
    }

    return {
      positions,
      bounds: this.calculateBounds(positions),
      metadata: {
        iterations,
        stable,
      },
    };
  }

  private applyHierarchical(
    nodes: GraphNode[],
    edges: GraphEdge[]
  ): LayoutResult {
    return this.hierarchicalLayout.layout(nodes, edges);
  }

  private applyRadial(nodes: GraphNode[], edges: GraphEdge[]): LayoutResult {
    return this.radialLayout.layout(nodes, edges);
  }

  private applyGrid(nodes: GraphNode[]): LayoutResult {
    const positions = new Map<string, { x: number; y: number; z: number }>();
    const spacing = 100;
    const cols = Math.ceil(Math.sqrt(nodes.length));

    nodes.forEach((node, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      positions.set(node.id, {
        x: col * spacing - (cols * spacing) / 2,
        y: 0,
        z: row * spacing - (Math.ceil(nodes.length / cols) * spacing) / 2,
      });
    });

    return {
      positions,
      bounds: this.calculateBounds(positions),
      metadata: {
        type: "grid",
        columns: cols,
        rows: Math.ceil(nodes.length / cols),
      },
    };
  }

  // ==========================================================================
  // ANIMATION
  // ==========================================================================

  private startTransition(
    targetPositions: Map<string, { x: number; y: number; z: number }>
  ): void {
    this.cancelTransition();

    this.transition = {
      from: new Map(this.currentPositions),
      to: targetPositions,
      progress: 0,
      startTime: performance.now(),
      duration: this.config.transitionDuration,
    };

    this.animateTransition();
  }

  private animateTransition(): void {
    if (!this.transition) return;

    const elapsed = performance.now() - this.transition.startTime;
    this.transition.progress = Math.min(1, elapsed / this.transition.duration);

    if (this.transition.progress >= 1) {
      this.currentPositions = this.transition.to;
      this.transition = null;
      return;
    }

    this.animationFrame = requestAnimationFrame(() => this.animateTransition());
  }

  private interpolatePositions(): Map<
    string,
    { x: number; y: number; z: number }
  > {
    if (!this.transition) return new Map(this.currentPositions);

    const result = new Map<string, { x: number; y: number; z: number }>();
    const t = this.easeInOutCubic(this.transition.progress);

    for (const [id, toPos] of this.transition.to) {
      const fromPos = this.transition.from.get(id) ?? toPos;
      result.set(id, {
        x: fromPos.x + (toPos.x - fromPos.x) * t,
        y: fromPos.y + (toPos.y - fromPos.y) * t,
        z: fromPos.z + (toPos.z - fromPos.z) * t,
      });
    }

    return result;
  }

  private easeInOutCubic(t: number): number {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // ==========================================================================
  // UTILITIES
  // ==========================================================================

  private calculateBounds(
    positions: Map<string, { x: number; y: number; z: number }>
  ): LayoutResult["bounds"] {
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (const pos of positions.values()) {
      minX = Math.min(minX, pos.x);
      minY = Math.min(minY, pos.y);
      minZ = Math.min(minZ, pos.z);
      maxX = Math.max(maxX, pos.x);
      maxY = Math.max(maxY, pos.y);
      maxZ = Math.max(maxZ, pos.z);
    }

    return {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
    };
  }
}
