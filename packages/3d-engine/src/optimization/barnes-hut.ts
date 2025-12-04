/**
 * Barnes-Hut Spatial Tree Implementation
 *
 * Optimized spatial data structure for force-directed layouts.
 * Reduces O(n²) force calculations to O(n log n) for large graphs.
 *
 * @module @neurectomy/3d-engine/optimization/barnes-hut
 * @agents @VELOCITY @AXIOM @VERTEX
 * @phase Phase 3 - Dimensional Forge
 * @step Force-Directed Graph Layout Optimization
 */

import * as THREE from "three";

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface BarnesHutNode {
  /** Center of mass position */
  centerOfMass: THREE.Vector3;
  /** Total mass (sum of all bodies in this node) */
  totalMass: number;
  /** Number of bodies in this node */
  bodyCount: number;
  /** Bounding box for this node */
  bounds: THREE.Box3;
  /** Size of the bounding region (for theta comparison) */
  size: number;
  /** Children (octants) - null if leaf or empty */
  children: (BarnesHutNode | null)[];
  /** Body index if this is a leaf node with exactly one body */
  bodyIndex: number | null;
}

export interface Body {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  mass: number;
  charge: number; // For repulsion force
  fixed: boolean;
  userData?: any;
}

export interface BarnesHutConfig {
  /** Theta threshold for approximation (0.5-1.0 typical) */
  theta?: number;
  /** Maximum tree depth */
  maxDepth?: number;
  /** Gravitational constant (negative for repulsion) */
  gravity?: number;
  /** Minimum distance to prevent singularity */
  minDistance?: number;
  /** Softening parameter for force calculations */
  softening?: number;
}

export interface ForceResult {
  force: THREE.Vector3;
  bodyIndex: number;
}

// ============================================================================
// Barnes-Hut Tree
// ============================================================================

/**
 * BarnesHutTree - Spatial tree for O(n log n) force calculations
 *
 * Algorithm:
 * 1. Build octree from body positions
 * 2. Calculate center of mass for each node
 * 3. For force calculation, if node.size / distance < theta,
 *    treat entire node as single body (approximation)
 * 4. Otherwise, recurse into children
 */
export class BarnesHutTree {
  private root: BarnesHutNode | null = null;
  private config: Required<BarnesHutConfig>;
  private bodies: Body[] = [];

  // Pre-allocated vectors for force calculations
  private tempDir = new THREE.Vector3();
  private tempForce = new THREE.Vector3();
  private tempCenter = new THREE.Vector3();

  // Statistics
  private nodeCount = 0;
  private forceCalculations = 0;
  private approximations = 0;

  constructor(config: BarnesHutConfig = {}) {
    this.config = {
      theta: config.theta ?? 0.7,
      maxDepth: config.maxDepth ?? 20,
      gravity: config.gravity ?? -10, // Negative for repulsion
      minDistance: config.minDistance ?? 0.001,
      softening: config.softening ?? 0.1,
    };
  }

  // ============================================================================
  // Tree Construction
  // ============================================================================

  /**
   * Build tree from array of bodies
   */
  public build(bodies: Body[]): void {
    this.bodies = bodies;
    this.nodeCount = 0;

    if (bodies.length === 0) {
      this.root = null;
      return;
    }

    // Calculate bounding box
    const bounds = new THREE.Box3();
    for (const body of bodies) {
      bounds.expandByPoint(body.position);
    }

    // Ensure cube bounds (required for octree)
    const size = bounds.getSize(new THREE.Vector3());
    const maxSize = Math.max(size.x, size.y, size.z) * 1.1; // Add padding
    const center = bounds.getCenter(new THREE.Vector3());

    bounds.setFromCenterAndSize(
      center,
      new THREE.Vector3(maxSize, maxSize, maxSize)
    );

    // Build tree
    this.root = this.createNode(bounds);

    for (let i = 0; i < bodies.length; i++) {
      const body = bodies[i]!;
      if (!body.fixed) {
        // Only add non-fixed bodies to tree
        this.insertBody(this.root!, i, 0);
      }
    }

    // Calculate centers of mass
    this.calculateCenterOfMass(this.root!);
  }

  /**
   * Create a new tree node
   */
  private createNode(bounds: THREE.Box3): BarnesHutNode {
    this.nodeCount++;
    const size = bounds.getSize(new THREE.Vector3()).length();

    return {
      centerOfMass: new THREE.Vector3(),
      totalMass: 0,
      bodyCount: 0,
      bounds: bounds.clone(),
      size,
      children: new Array(8).fill(null),
      bodyIndex: null,
    };
  }

  /**
   * Insert a body into the tree
   */
  private insertBody(
    node: BarnesHutNode,
    bodyIndex: number,
    depth: number
  ): void {
    const body = this.bodies[bodyIndex]!;

    // If node is empty, store body here
    if (node.bodyCount === 0) {
      node.bodyIndex = bodyIndex;
      node.bodyCount = 1;
      return;
    }

    // If we've reached max depth, just accumulate
    if (depth >= this.config.maxDepth) {
      node.bodyCount++;
      return;
    }

    // If node has one body, we need to subdivide and move it
    if (node.bodyIndex !== null) {
      const existingBodyIndex = node.bodyIndex;
      node.bodyIndex = null;

      // Insert existing body into appropriate child
      const existingOctant = this.getOctant(
        node,
        this.bodies[existingBodyIndex]!.position
      );
      if (node.children[existingOctant] === null) {
        node.children[existingOctant] = this.createNode(
          this.getOctantBounds(node.bounds, existingOctant)
        );
      }
      this.insertBody(
        node.children[existingOctant]!,
        existingBodyIndex,
        depth + 1
      );
    }

    // Insert new body
    const octant = this.getOctant(node, body.position);
    if (node.children[octant] === null) {
      node.children[octant] = this.createNode(
        this.getOctantBounds(node.bounds, octant)
      );
    }
    this.insertBody(node.children[octant]!, bodyIndex, depth + 1);

    node.bodyCount++;
  }

  /**
   * Determine which octant a position falls into
   */
  private getOctant(node: BarnesHutNode, position: THREE.Vector3): number {
    const center = node.bounds.getCenter(this.tempCenter);

    let octant = 0;
    if (position.x >= center.x) octant |= 1;
    if (position.y >= center.y) octant |= 2;
    if (position.z >= center.z) octant |= 4;

    return octant;
  }

  /**
   * Get bounds for a specific octant
   */
  private getOctantBounds(
    parentBounds: THREE.Box3,
    octant: number
  ): THREE.Box3 {
    const min = parentBounds.min.clone();
    const max = parentBounds.max.clone();
    const center = parentBounds.getCenter(this.tempCenter);

    // X axis
    if (octant & 1) {
      min.x = center.x;
    } else {
      max.x = center.x;
    }

    // Y axis
    if (octant & 2) {
      min.y = center.y;
    } else {
      max.y = center.y;
    }

    // Z axis
    if (octant & 4) {
      min.z = center.z;
    } else {
      max.z = center.z;
    }

    return new THREE.Box3(min, max);
  }

  /**
   * Calculate center of mass for all nodes (bottom-up)
   */
  private calculateCenterOfMass(node: BarnesHutNode): void {
    // Leaf node with single body
    if (node.bodyIndex !== null) {
      const body = this.bodies[node.bodyIndex]!;
      node.centerOfMass.copy(body.position);
      node.totalMass = body.mass;
      return;
    }

    // Internal node - aggregate children
    node.centerOfMass.set(0, 0, 0);
    node.totalMass = 0;

    for (const child of node.children) {
      if (child !== null) {
        this.calculateCenterOfMass(child);

        // Weighted position by mass
        this.tempCenter
          .copy(child.centerOfMass)
          .multiplyScalar(child.totalMass);
        node.centerOfMass.add(this.tempCenter);
        node.totalMass += child.totalMass;
      }
    }

    if (node.totalMass > 0) {
      node.centerOfMass.divideScalar(node.totalMass);
    }
  }

  // ============================================================================
  // Force Calculation
  // ============================================================================

  /**
   * Calculate force on a body from all other bodies
   */
  public calculateForce(bodyIndex: number): THREE.Vector3 {
    this.forceCalculations = 0;
    this.approximations = 0;

    const force = new THREE.Vector3();

    if (this.root === null) return force;

    const body = this.bodies[bodyIndex]!;
    this.calculateForceRecursive(body, bodyIndex, this.root, force);

    return force;
  }

  /**
   * Recursive force calculation with Barnes-Hut approximation
   */
  private calculateForceRecursive(
    body: Body,
    bodyIndex: number,
    node: BarnesHutNode,
    force: THREE.Vector3
  ): void {
    if (node.bodyCount === 0 || node.totalMass === 0) return;

    // Single body in node
    if (node.bodyIndex !== null) {
      if (node.bodyIndex !== bodyIndex) {
        this.addPairwiseForce(body, this.bodies[node.bodyIndex]!, force);
        this.forceCalculations++;
      }
      return;
    }

    // Calculate distance to center of mass
    this.tempDir.subVectors(node.centerOfMass, body.position);
    const distance = this.tempDir.length();

    // Barnes-Hut criterion: s/d < theta
    const ratio = node.size / Math.max(distance, this.config.minDistance);

    if (ratio < this.config.theta) {
      // Use approximation - treat node as single body
      this.addForceFromMass(body, node.centerOfMass, node.totalMass, force);
      this.approximations++;
      this.forceCalculations++;
      return;
    }

    // Recurse into children
    for (const child of node.children) {
      if (child !== null) {
        this.calculateForceRecursive(body, bodyIndex, child, force);
      }
    }
  }

  /**
   * Calculate and add pairwise force between two bodies
   */
  private addPairwiseForce(
    body1: Body,
    body2: Body,
    force: THREE.Vector3
  ): void {
    this.tempDir.subVectors(body2.position, body1.position);
    const distSq = Math.max(
      this.tempDir.lengthSq(),
      this.config.minDistance * this.config.minDistance
    );

    // Add softening to prevent extreme forces
    const softenedDistSq =
      distSq + this.config.softening * this.config.softening;
    const dist = Math.sqrt(softenedDistSq);

    // Force magnitude: G * m1 * m2 / r²
    // For repulsion (negative gravity), force is outward
    const magnitude =
      (this.config.gravity * body1.charge * body2.charge) / softenedDistSq;

    // Normalize direction and apply magnitude
    this.tempForce
      .copy(this.tempDir)
      .divideScalar(dist)
      .multiplyScalar(magnitude);
    force.add(this.tempForce);
  }

  /**
   * Calculate and add force from an aggregated mass
   */
  private addForceFromMass(
    body: Body,
    position: THREE.Vector3,
    mass: number,
    force: THREE.Vector3
  ): void {
    this.tempDir.subVectors(position, body.position);
    const distSq = Math.max(
      this.tempDir.lengthSq(),
      this.config.minDistance * this.config.minDistance
    );

    const softenedDistSq =
      distSq + this.config.softening * this.config.softening;
    const dist = Math.sqrt(softenedDistSq);

    // Use aggregated mass (assuming average charge)
    const firstBodyMass = this.bodies[0]?.mass ?? 1;
    const effectiveCharge = mass / firstBodyMass;
    const magnitude =
      (this.config.gravity * body.charge * effectiveCharge) / softenedDistSq;

    this.tempForce
      .copy(this.tempDir)
      .divideScalar(dist)
      .multiplyScalar(magnitude);
    force.add(this.tempForce);
  }

  /**
   * Calculate forces for all bodies
   */
  public calculateAllForces(): ForceResult[] {
    const results: ForceResult[] = [];

    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i]!;
      if (!body.fixed) {
        results.push({
          force: this.calculateForce(i),
          bodyIndex: i,
        });
      }
    }

    return results;
  }

  // ============================================================================
  // Simulation Step
  // ============================================================================

  /**
   * Perform one simulation step with spring forces
   */
  public step(
    edges: Array<{ source: number; target: number; weight?: number }>,
    damping: number = 0.9,
    springConstant: number = 0.1,
    restLength: number = 50,
    deltaTime: number = 0.016
  ): void {
    // Rebuild tree (positions may have changed)
    this.build(this.bodies);

    // Calculate repulsion forces (Barnes-Hut)
    const forces: THREE.Vector3[] = [];
    for (let i = 0; i < this.bodies.length; i++) {
      const currentBody = this.bodies[i]!;
      forces[i] = currentBody.fixed
        ? new THREE.Vector3()
        : this.calculateForce(i);
    }

    // Add spring forces from edges
    for (const edge of edges) {
      if (
        edge.source >= this.bodies.length ||
        edge.target >= this.bodies.length
      )
        continue;

      const body1 = this.bodies[edge.source]!;
      const body2 = this.bodies[edge.target]!;

      this.tempDir.subVectors(body2.position, body1.position);
      const distance = this.tempDir.length();
      const displacement = distance - restLength * (edge.weight ?? 1);

      // Spring force: F = k * x
      this.tempDir.normalize().multiplyScalar(springConstant * displacement);

      if (!body1.fixed) forces[edge.source]!.add(this.tempDir);
      if (!body2.fixed) forces[edge.target]!.sub(this.tempDir);
    }

    // Apply forces (Verlet integration)
    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i]!;
      if (body.fixed) continue;

      // Update velocity
      body.velocity.addScaledVector(forces[i]!, deltaTime / body.mass);
      body.velocity.multiplyScalar(damping);

      // Clamp velocity to prevent explosions
      const maxVelocity = 100;
      if (body.velocity.length() > maxVelocity) {
        body.velocity.normalize().multiplyScalar(maxVelocity);
      }

      // Update position
      body.position.addScaledVector(body.velocity, deltaTime);
    }
  }

  // ============================================================================
  // Accessors
  // ============================================================================

  /**
   * Get tree statistics
   */
  public getStatistics() {
    return {
      nodeCount: this.nodeCount,
      bodyCount: this.bodies.length,
      lastForceCalculations: this.forceCalculations,
      lastApproximations: this.approximations,
      approximationRatio:
        this.forceCalculations > 0
          ? this.approximations / this.forceCalculations
          : 0,
    };
  }

  /**
   * Get bodies array
   */
  public getBodies(): Body[] {
    return this.bodies;
  }

  /**
   * Get theta parameter
   */
  public getTheta(): number {
    return this.config.theta;
  }

  /**
   * Set theta parameter (affects accuracy vs performance)
   */
  public setTheta(theta: number): void {
    this.config.theta = Math.max(0.1, Math.min(2.0, theta));
  }

  /**
   * Clear the tree
   */
  public clear(): void {
    this.root = null;
    this.bodies = [];
    this.nodeCount = 0;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create bodies from graph nodes
 */
export function createBodiesFromNodes<
  T extends { x: number; y: number; z: number; fixed?: boolean },
>(nodes: T[], defaultMass: number = 1, defaultCharge: number = 1): Body[] {
  return nodes.map((node) => ({
    position: new THREE.Vector3(node.x, node.y, node.z),
    velocity: new THREE.Vector3(),
    mass: defaultMass,
    charge: defaultCharge,
    fixed: node.fixed ?? false,
    userData: node,
  }));
}

/**
 * Apply body positions back to nodes
 */
export function applyPositionsToNodes<
  T extends { x: number; y: number; z: number },
>(bodies: Body[], nodes: T[]): void {
  for (let i = 0; i < Math.min(bodies.length, nodes.length); i++) {
    const node = nodes[i]!;
    const body = bodies[i]!;
    node.x = body.position.x;
    node.y = body.position.y;
    node.z = body.position.z;
  }
}

export default BarnesHutTree;
