/**
 * @file Force-Directed Layout Engine
 * @description High-performance force-directed graph layout
 * @module @neurectomy/3d-engine/visualization/graph/layouts
 * @agents @VERTEX @AXIOM @VELOCITY
 */

/* eslint-disable @typescript-eslint/no-non-null-assertion */
// This file uses typed arrays with careful bounds checking.
// The non-null assertions below are safe because:
// 1. Arrays are pre-allocated to exact size needed
// 2. All index access is within bounds (0 to n-1)
// 3. Performance-critical code benefits from avoiding runtime checks

import type { GraphNode, GraphEdge } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface ForceLayoutConfig {
  /** Simulation iterations per tick */
  iterations: number;
  /** Cooling factor (alpha decay) */
  alphaDecay: number;
  /** Minimum alpha before stopping */
  alphaMin: number;
  /** Initial alpha */
  alphaTarget: number;
  /** Velocity decay (friction) */
  velocityDecay: number;
  /** Forces configuration */
  forces: {
    /** Center force strength */
    center: number;
    /** Charge (repulsion) force strength */
    charge: number;
    /** Maximum charge distance */
    chargeDistanceMax: number;
    /** Minimum charge distance */
    chargeDistanceMin: number;
    /** Link (spring) force strength */
    link: number;
    /** Ideal link distance */
    linkDistance: number;
    /** Collision force radius multiplier */
    collision: number;
    /** Y-axis force for layering */
    forceY: number;
    /** Target Y value */
    targetY: number;
  };
  /** Use Barnes-Hut optimization for large graphs */
  useBarnesHut: boolean;
  /** Barnes-Hut theta parameter (0-1, higher = faster but less accurate) */
  theta: number;
  /** Enable 3D layout */
  is3D: boolean;
}

export interface SimulationState {
  /** Current alpha (temperature) */
  alpha: number;
  /** Is simulation running */
  running: boolean;
  /** Tick count */
  tickCount: number;
  /** Total energy in system */
  energy: number;
}

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_FORCE_CONFIG: ForceLayoutConfig = {
  iterations: 300,
  alphaDecay: 0.0228, // ~300 iterations to cool
  alphaMin: 0.001,
  alphaTarget: 0.3,
  velocityDecay: 0.4,
  forces: {
    center: 0.1,
    charge: -30,
    chargeDistanceMax: 100,
    chargeDistanceMin: 1,
    link: 1,
    linkDistance: 3,
    collision: 1.2,
    forceY: 0,
    targetY: 0,
  },
  useBarnesHut: true,
  theta: 0.9,
  is3D: true,
};

// ============================================================================
// Force-Directed Layout Engine
// ============================================================================

/**
 * High-performance force-directed graph layout engine
 */
export class ForceDirectedLayout {
  private config: ForceLayoutConfig;
  private state: SimulationState;
  private nodes: GraphNode[] = [];
  private edges: GraphEdge[] = [];
  private nodeIndexMap: Map<string, number> = new Map();
  private animationFrameId: number | null = null;

  // Cached data structures
  private positions: Float32Array = new Float32Array(0);
  private velocities: Float32Array = new Float32Array(0);
  private forces: Float32Array = new Float32Array(0);
  private masses: Float32Array = new Float32Array(0);

  // Callbacks
  private onTick?: (state: SimulationState) => void;
  private onEnd?: () => void;

  constructor(config: Partial<ForceLayoutConfig> = {}) {
    this.config = { ...DEFAULT_FORCE_CONFIG, ...config };
    this.state = {
      alpha: this.config.alphaTarget,
      running: false,
      tickCount: 0,
      energy: 0,
    };
  }

  // Type-safe array accessors that guarantee non-undefined values
  // These are safe because we always allocate arrays with correct size
  private getPos(idx: number): number {
    return this.positions[idx]! ?? 0;
  }

  private setPos(idx: number, val: number): void {
    this.positions[idx]! = val;
  }

  private getVel(idx: number): number {
    return this.velocities[idx]! ?? 0;
  }

  private setVel(idx: number, val: number): void {
    this.velocities[idx]! = val;
  }

  private getForce(idx: number): number {
    return this.forces[idx]! ?? 0;
  }

  private setForce(idx: number, val: number): void {
    this.forces[idx]! = val;
  }

  private addForce(idx: number, val: number): void {
    this.forces[idx]! = this.getForce(idx) + val;
  }

  private getMass(idx: number): number {
    return this.masses[idx]! ?? 1;
  }

  private getNode(idx: number): GraphNode | undefined {
    return this.nodes[idx];
  }

  /**
   * Initialize layout with nodes and edges
   */
  initialize(nodes: GraphNode[], edges: GraphEdge[]): void {
    this.nodes = nodes;
    this.edges = edges;
    this.nodeIndexMap.clear();

    // Build node index map
    nodes.forEach((node, index) => {
      this.nodeIndexMap.set(node.id, index);
    });

    // Initialize typed arrays for performance
    const n = nodes.length;
    const dim = this.config.is3D ? 3 : 2;

    this.positions = new Float32Array(n * dim);
    this.velocities = new Float32Array(n * dim);
    this.forces = new Float32Array(n * dim);
    this.masses = new Float32Array(n);

    // Copy initial positions and masses
    nodes.forEach((node, i) => {
      const baseIdx = i * dim;
      this.positions[baseIdx]! = node.position.x;
      this.positions[baseIdx + 1]! = node.position.y;
      if (this.config.is3D) {
        this.positions[baseIdx + 2]! = node.position.z;
      }
      this.masses[i]! = node.mass;
    });

    // Reset state
    this.state = {
      alpha: this.config.alphaTarget,
      running: false,
      tickCount: 0,
      energy: 0,
    };
  }

  /**
   * Start the simulation
   */
  start(onTick?: (state: SimulationState) => void, onEnd?: () => void): void {
    this.onTick = onTick;
    this.onEnd = onEnd;
    this.state.running = true;
    this.state.alpha = this.config.alphaTarget;
    this.state.tickCount = 0;
    this.runSimulation();
  }

  /**
   * Stop the simulation
   */
  stop(): void {
    this.state.running = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Run one simulation tick
   */
  tick(): boolean {
    if (this.state.alpha < this.config.alphaMin) {
      return false;
    }

    // Clear forces
    this.forces.fill(0);

    // Apply forces
    this.applyCenterForce();
    this.applyChargeForce();
    this.applyLinkForce();
    this.applyCollisionForce();

    if (this.config.forces.forceY !== 0) {
      this.applyYForce();
    }

    // Update velocities and positions
    this.updateVelocitiesAndPositions();

    // Calculate energy
    this.state.energy = this.calculateEnergy();

    // Update alpha (cooling)
    this.state.alpha += (0 - this.state.alpha) * this.config.alphaDecay;
    this.state.tickCount++;

    // Sync positions back to nodes
    this.syncPositionsToNodes();

    return this.state.alpha >= this.config.alphaMin;
  }

  /**
   * Run full simulation loop
   */
  private runSimulation(): void {
    const step = () => {
      if (!this.state.running) return;

      // Run multiple iterations per frame for faster convergence
      const iterationsPerFrame = 3;
      for (let i = 0; i < iterationsPerFrame; i++) {
        const shouldContinue = this.tick();
        if (!shouldContinue) {
          this.state.running = false;
          this.onEnd?.();
          return;
        }
      }

      this.onTick?.(this.state);
      this.animationFrameId = requestAnimationFrame(step);
    };

    this.animationFrameId = requestAnimationFrame(step);
  }

  /**
   * Apply center force (gravity towards center)
   */
  private applyCenterForce(): void {
    const strength = this.config.forces.center * this.state.alpha;
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    // Calculate center of mass
    let cx = 0,
      cy = 0,
      cz = 0;
    let totalMass = 0;

    for (let i = 0; i < n; i++) {
      const mass = this.getMass(i);
      const baseIdx = i * dim;
      cx += this.getPos(baseIdx) * mass;
      cy += this.getPos(baseIdx + 1) * mass;
      if (this.config.is3D) {
        cz += this.getPos(baseIdx + 2) * mass;
      }
      totalMass += mass;
    }

    cx /= totalMass;
    cy /= totalMass;
    cz /= totalMass;

    // Apply force towards center
    for (let i = 0; i < n; i++) {
      const node = this.getNode(i);
      if (!node || node.pinned) continue;

      const baseIdx = i * dim;
      this.addForce(baseIdx, -(this.getPos(baseIdx) - 0) * strength);
      this.addForce(baseIdx + 1, -(this.getPos(baseIdx + 1) - 0) * strength);
      if (this.config.is3D) {
        this.addForce(baseIdx + 2, -(this.getPos(baseIdx + 2) - 0) * strength);
      }
    }
  }

  /**
   * Apply charge (repulsion) force
   */
  private applyChargeForce(): void {
    const n = this.nodes.length;

    if (this.config.useBarnesHut && n > 100) {
      this.applyChargeForceBarnesHut();
    } else {
      this.applyChargeForceNaive();
    }
  }

  /**
   * Naive O(n²) charge force calculation
   */
  private applyChargeForceNaive(): void {
    const strength = this.config.forces.charge;
    const distMin = this.config.forces.chargeDistanceMin;
    const distMax = this.config.forces.chargeDistanceMax;
    const distMinSq = distMin * distMin;
    const distMaxSq = distMax * distMax;
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    for (let i = 0; i < n; i++) {
      if (this.nodes[i]!.pinned) continue;

      const ix = i * dim;
      const px = this.positions[ix]!;
      const py = this.positions[ix + 1]!;
      const pz = this.config.is3D ? this.positions[ix + 2]! : 0;

      for (let j = i + 1; j < n; j++) {
        const jx = j * dim;
        const dx = this.positions[jx]! - px;
        const dy = this.positions[jx + 1]! - py;
        const dz = this.config.is3D ? this.positions[jx + 2]! - pz : 0;

        let distSq = dx * dx + dy * dy + dz * dz;

        if (distSq > distMaxSq) continue;
        if (distSq < distMinSq) distSq = distMinSq;

        const dist = Math.sqrt(distSq);
        const force = (strength * this.state.alpha) / distSq;

        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        const fz = this.config.is3D ? (dz / dist) * force : 0;

        this.forces[ix]! += fx;
        this.forces[ix + 1]! += fy;
        if (this.config.is3D) this.forces[ix + 2]! += fz;

        if (!this.nodes[j]!.pinned) {
          this.forces[jx]! -= fx;
          this.forces[jx + 1]! -= fy;
          if (this.config.is3D) this.forces[jx + 2]! -= fz;
        }
      }
    }
  }

  /**
   * Barnes-Hut O(n log n) charge force calculation
   */
  private applyChargeForceBarnesHut(): void {
    // Build octree/quadtree
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    // Find bounds
    let minX = Infinity,
      maxX = -Infinity;
    let minY = Infinity,
      maxY = -Infinity;
    let minZ = Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < n; i++) {
      const baseIdx = i * dim;
      const x = this.positions[baseIdx]!;
      const y = this.positions[baseIdx + 1]!;
      const z = this.config.is3D ? this.positions[baseIdx + 2]! : 0;

      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }

    // Add padding
    const padding = 1;
    minX -= padding;
    maxX += padding;
    minY -= padding;
    maxY += padding;
    minZ -= padding;
    maxZ += padding;

    // Build tree
    const tree = new OctreeNode(minX, maxX, minY, maxY, minZ, maxZ);

    for (let i = 0; i < n; i++) {
      const baseIdx = i * dim;
      tree.insert({
        x: this.positions[baseIdx]!,
        y: this.positions[baseIdx + 1]!,
        z: this.config.is3D ? this.positions[baseIdx + 2]! : 0,
        mass: this.masses[i]!,
        index: i,
      });
    }

    // Calculate center of mass for all nodes
    tree.calculateCenterOfMass();

    // Apply forces using tree
    const strength = this.config.forces.charge;
    const theta = this.config.theta;
    const distMin = this.config.forces.chargeDistanceMin;
    const distMinSq = distMin * distMin;

    for (let i = 0; i < n; i++) {
      if (this.nodes[i]!.pinned) continue;

      const baseIdx = i * dim;
      const px = this.positions[baseIdx]!;
      const py = this.positions[baseIdx + 1]!;
      const pz = this.config.is3D ? this.positions[baseIdx + 2]! : 0;

      const force = { x: 0, y: 0, z: 0 };
      tree.calculateForce(
        px,
        py,
        pz,
        i,
        theta,
        strength * this.state.alpha,
        distMinSq,
        force
      );

      this.forces[baseIdx]! += force.x;
      this.forces[baseIdx + 1]! += force.y;
      if (this.config.is3D) {
        this.forces[baseIdx + 2]! += force.z;
      }
    }
  }

  /**
   * Apply link (spring) force
   */
  private applyLinkForce(): void {
    const strength = this.config.forces.link * this.state.alpha;
    const idealDistance = this.config.forces.linkDistance;
    const dim = this.config.is3D ? 3 : 2;

    for (const edge of this.edges) {
      const sourceIdx = this.nodeIndexMap.get(edge.sourceId);
      const targetIdx = this.nodeIndexMap.get(edge.targetId);

      if (sourceIdx === undefined || targetIdx === undefined) continue;

      const si = sourceIdx * dim;
      const ti = targetIdx * dim;

      const dx = this.positions[ti]! - this.positions[si]!;
      const dy = this.positions[ti + 1]! - this.positions[si + 1]!;
      const dz = this.config.is3D
        ? this.positions[ti + 2]! - this.positions[si + 2]!
        : 0;

      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const displacement = dist - idealDistance;
      const force = displacement * strength * edge.weight;

      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      const fz = this.config.is3D ? (dz / dist) * force : 0;

      if (!this.nodes[sourceIdx]!.pinned) {
        this.forces[si]! += fx;
        this.forces[si + 1]! += fy;
        if (this.config.is3D) this.forces[si + 2]! += fz;
      }

      if (!this.nodes[targetIdx]!.pinned) {
        this.forces[ti]! -= fx;
        this.forces[ti + 1]! -= fy;
        if (this.config.is3D) this.forces[ti + 2]! -= fz;
      }
    }
  }

  /**
   * Apply collision force
   */
  private applyCollisionForce(): void {
    const radiusMultiplier = this.config.forces.collision;
    if (radiusMultiplier <= 0) return;

    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    for (let i = 0; i < n; i++) {
      if (this.nodes[i]!.pinned) continue;

      const ri = this.nodes[i]!.radius * radiusMultiplier;
      const ix = i * dim;
      const px = this.positions[ix]!;
      const py = this.positions[ix + 1]!;
      const pz = this.config.is3D ? this.positions[ix + 2]! : 0;

      for (let j = i + 1; j < n; j++) {
        const rj = this.nodes[j]!.radius * radiusMultiplier;
        const jx = j * dim;

        const dx = this.positions[jx]! - px;
        const dy = this.positions[jx + 1]! - py;
        const dz = this.config.is3D ? this.positions[jx + 2]! - pz : 0;

        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const minDist = ri + rj;

        if (dist < minDist && dist > 0) {
          const overlap = minDist - dist;
          const force = overlap * 0.5 * this.state.alpha;

          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          const fz = this.config.is3D ? (dz / dist) * force : 0;

          this.forces[ix]! -= fx;
          this.forces[ix + 1]! -= fy;
          if (this.config.is3D) this.forces[ix + 2]! -= fz;

          if (!this.nodes[j]!.pinned) {
            this.forces[jx]! += fx;
            this.forces[jx + 1]! += fy;
            if (this.config.is3D) this.forces[jx + 2]! += fz;
          }
        }
      }
    }
  }

  /**
   * Apply Y-axis force for layered layouts
   */
  private applyYForce(): void {
    const strength = this.config.forces.forceY * this.state.alpha;
    const targetY = this.config.forces.targetY;
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    for (let i = 0; i < n; i++) {
      if (this.nodes[i]!.pinned) continue;

      const baseIdx = i * dim;
      const dy = targetY - this.positions[baseIdx + 1]!;
      this.forces[baseIdx + 1]! += dy * strength;
    }
  }

  /**
   * Update velocities and positions based on forces
   */
  private updateVelocitiesAndPositions(): void {
    const velocityDecay = this.config.velocityDecay;
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    for (let i = 0; i < n; i++) {
      if (this.nodes[i]!.pinned) continue;

      const baseIdx = i * dim;
      const mass = this.masses[i]!;

      // Update velocity: v = v * decay + force / mass
      this.velocities[baseIdx]! =
        this.velocities[baseIdx]! * velocityDecay +
        this.forces[baseIdx]! / mass;
      this.velocities[baseIdx + 1]! =
        this.velocities[baseIdx + 1]! * velocityDecay +
        this.forces[baseIdx + 1]! / mass;

      // Update position: p = p + v
      this.positions[baseIdx]! += this.velocities[baseIdx]!;
      this.positions[baseIdx + 1]! += this.velocities[baseIdx + 1]!;

      if (this.config.is3D) {
        this.velocities[baseIdx + 2]! =
          this.velocities[baseIdx + 2]! * velocityDecay +
          this.forces[baseIdx + 2]! / mass;
        this.positions[baseIdx + 2]! += this.velocities[baseIdx + 2]!;
      }
    }
  }

  /**
   * Calculate total kinetic energy in the system
   */
  private calculateEnergy(): number {
    let energy = 0;
    const dim = this.config.is3D ? 3 : 2;
    const n = this.nodes.length;

    for (let i = 0; i < n; i++) {
      const baseIdx = i * dim;
      const vx = this.velocities[baseIdx]!;
      const vy = this.velocities[baseIdx + 1]!;
      const vz = this.config.is3D ? this.velocities[baseIdx + 2]! : 0;
      const mass = this.masses[i]!;

      energy += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    }

    return energy;
  }

  /**
   * Sync positions from typed arrays back to node objects
   */
  private syncPositionsToNodes(): void {
    const dim = this.config.is3D ? 3 : 2;

    for (let i = 0; i < this.nodes.length; i++) {
      const baseIdx = i * dim;
      this.nodes[i]!.position.x = this.positions[baseIdx]!;
      this.nodes[i]!.position.y = this.positions[baseIdx + 1]!;
      if (this.config.is3D) {
        this.nodes[i]!.position.z = this.positions[baseIdx + 2]!;
      }
      this.nodes[i]!.velocity.x = this.velocities[baseIdx]!;
      this.nodes[i]!.velocity.y = this.velocities[baseIdx + 1]!;
      if (this.config.is3D) {
        this.nodes[i]!.velocity.z = this.velocities[baseIdx + 2]!;
      }
    }
  }

  /**
   * Get current simulation state
   */
  getState(): SimulationState {
    return { ...this.state };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ForceLayoutConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get nodes
   */
  getNodes(): GraphNode[] {
    return this.nodes;
  }

  /**
   * Reheat simulation (restart with higher alpha)
   */
  reheat(alpha: number = this.config.alphaTarget): void {
    this.state.alpha = alpha;
    if (!this.state.running) {
      this.start(this.onTick, this.onEnd);
    }
  }
}

// ============================================================================
// Octree Node for Barnes-Hut
// ============================================================================

interface OctreeBody {
  x: number;
  y: number;
  z: number;
  mass: number;
  index: number;
}

class OctreeNode {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;

  body: OctreeBody | null = null;
  children: OctreeNode[] | null = null;

  // Center of mass
  cx = 0;
  cy = 0;
  cz = 0;
  totalMass = 0;

  constructor(
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
    minZ: number,
    maxZ: number
  ) {
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.minZ = minZ;
    this.maxZ = maxZ;
  }

  get size(): number {
    return Math.max(
      this.maxX - this.minX,
      this.maxY - this.minY,
      this.maxZ - this.minZ
    );
  }

  insert(body: OctreeBody): void {
    // If empty leaf, store body
    if (this.body === null && this.children === null) {
      this.body = body;
      return;
    }

    // If internal node or need to subdivide
    if (this.children === null) {
      this.subdivide();
      // Re-insert existing body
      if (this.body !== null) {
        this.insertIntoChild(this.body);
        this.body = null;
      }
    }

    // Insert new body
    this.insertIntoChild(body);
  }

  private subdivide(): void {
    const midX = (this.minX + this.maxX) / 2;
    const midY = (this.minY + this.maxY) / 2;
    const midZ = (this.minZ + this.maxZ) / 2;

    this.children = [
      new OctreeNode(this.minX, midX, this.minY, midY, this.minZ, midZ),
      new OctreeNode(midX, this.maxX, this.minY, midY, this.minZ, midZ),
      new OctreeNode(this.minX, midX, midY, this.maxY, this.minZ, midZ),
      new OctreeNode(midX, this.maxX, midY, this.maxY, this.minZ, midZ),
      new OctreeNode(this.minX, midX, this.minY, midY, midZ, this.maxZ),
      new OctreeNode(midX, this.maxX, this.minY, midY, midZ, this.maxZ),
      new OctreeNode(this.minX, midX, midY, this.maxY, midZ, this.maxZ),
      new OctreeNode(midX, this.maxX, midY, this.maxY, midZ, this.maxZ),
    ];
  }

  private insertIntoChild(body: OctreeBody): void {
    const midX = (this.minX + this.maxX) / 2;
    const midY = (this.minY + this.maxY) / 2;
    const midZ = (this.minZ + this.maxZ) / 2;

    let childIndex = 0;
    if (body.x >= midX) childIndex += 1;
    if (body.y >= midY) childIndex += 2;
    if (body.z >= midZ) childIndex += 4;

    this.children![childIndex]!.insert(body);
  }

  calculateCenterOfMass(): void {
    if (this.body !== null) {
      this.cx = this.body.x;
      this.cy = this.body.y;
      this.cz = this.body.z;
      this.totalMass = this.body.mass;
      return;
    }

    if (this.children === null) {
      return;
    }

    let cx = 0,
      cy = 0,
      cz = 0,
      totalMass = 0;

    for (const child of this.children) {
      child.calculateCenterOfMass();
      if (child.totalMass > 0) {
        cx += child.cx * child.totalMass;
        cy += child.cy * child.totalMass;
        cz += child.cz * child.totalMass;
        totalMass += child.totalMass;
      }
    }

    if (totalMass > 0) {
      this.cx = cx / totalMass;
      this.cy = cy / totalMass;
      this.cz = cz / totalMass;
      this.totalMass = totalMass;
    }
  }

  calculateForce(
    x: number,
    y: number,
    z: number,
    bodyIndex: number,
    theta: number,
    strength: number,
    distMinSq: number,
    force: { x: number; y: number; z: number }
  ): void {
    if (this.totalMass === 0) return;

    const dx = this.cx - x;
    const dy = this.cy - y;
    const dz = this.cz - z;
    let distSq = dx * dx + dy * dy + dz * dz;

    // If this is a leaf with the same body, skip
    if (this.body !== null && this.body.index === bodyIndex) {
      return;
    }

    const size = this.size;

    // If node is far enough away or is a leaf, treat as single body
    if (this.children === null || (size * size) / distSq < theta * theta) {
      if (distSq < distMinSq) distSq = distMinSq;
      const dist = Math.sqrt(distSq);
      const f = (strength * this.totalMass) / distSq;

      force.x += (dx / dist) * f;
      force.y += (dy / dist) * f;
      force.z += (dz / dist) * f;
    } else {
      // Recurse into children
      for (const child of this.children) {
        child.calculateForce(
          x,
          y,
          z,
          bodyIndex,
          theta,
          strength,
          distMinSq,
          force
        );
      }
    }
  }
}
