/**
 * @file Barnes-Hut Tree Implementation
 * @description Octree-based spatial partitioning for O(n log n) force calculations
 * @module @neurectomy/3d-engine/visualization/graph/layouts/barnes-hut
 * @agents @VELOCITY @AXIOM
 */

import type { GraphNode } from "../types";

// ============================================================================
// TYPES
// ============================================================================

export interface BarnesHutConfig {
  /** Barnes-Hut approximation threshold (0.5-1.0 typical) */
  theta: number;
  /** Minimum cell size before stopping subdivision */
  minCellSize: number;
  /** Maximum tree depth */
  maxDepth: number;
  /** Gravitational constant for attraction */
  gravity: number;
  /** Softening parameter to prevent singularities */
  softening: number;
}

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface BoundingBox {
  min: Vector3D;
  max: Vector3D;
  center: Vector3D;
  size: number;
}

export interface OctreeNode {
  bounds: BoundingBox;
  centerOfMass: Vector3D;
  totalMass: number;
  children: (OctreeNode | null)[];
  body: BodyData | null;
  isLeaf: boolean;
  depth: number;
}

export interface BodyData {
  id: string;
  position: Vector3D;
  mass: number;
  node: GraphNode;
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

export const DEFAULT_BARNES_HUT_CONFIG: BarnesHutConfig = {
  theta: 0.7,
  minCellSize: 1,
  maxDepth: 20,
  gravity: 0.1,
  softening: 0.01,
};

// ============================================================================
// VECTOR UTILITIES
// ============================================================================

const vecAdd = (a: Vector3D, b: Vector3D): Vector3D => ({
  x: a.x + b.x,
  y: a.y + b.y,
  z: a.z + b.z,
});

const vecSub = (a: Vector3D, b: Vector3D): Vector3D => ({
  x: a.x - b.x,
  y: a.y - b.y,
  z: a.z - b.z,
});

const vecScale = (v: Vector3D, s: number): Vector3D => ({
  x: v.x * s,
  y: v.y * s,
  z: v.z * s,
});

const vecLength = (v: Vector3D): number =>
  Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

const vecNormalize = (v: Vector3D): Vector3D => {
  const len = vecLength(v);
  return len > 0 ? vecScale(v, 1 / len) : { x: 0, y: 0, z: 0 };
};

// ============================================================================
// BARNES-HUT TREE
// ============================================================================

/**
 * Barnes-Hut Octree for efficient N-body force calculation
 * Provides O(n log n) complexity instead of O(n²)
 */
export class BarnesHutTree {
  private root: OctreeNode | null = null;
  private config: BarnesHutConfig;
  private bodies: Map<string, BodyData> = new Map();

  constructor(config: Partial<BarnesHutConfig> = {}) {
    this.config = { ...DEFAULT_BARNES_HUT_CONFIG, ...config };
  }

  /**
   * Build octree from nodes
   */
  build(nodes: GraphNode[]): void {
    if (nodes.length === 0) {
      this.root = null;
      return;
    }

    // Convert nodes to bodies
    this.bodies.clear();
    const bodiesArray: BodyData[] = nodes.map((node) => {
      const body: BodyData = {
        id: node.id,
        position: { ...node.position },
        mass: node.mass ?? 1,
        node,
      };
      this.bodies.set(node.id, body);
      return body;
    });

    // Calculate bounding box
    const bounds = this.calculateBounds(bodiesArray);

    // Create root and insert all bodies
    this.root = this.createNode(bounds, 0);
    for (const body of bodiesArray) {
      this.insert(this.root, body);
    }

    // Calculate centers of mass
    this.calculateCenterOfMass(this.root);
  }

  /**
   * Calculate force on a body from all other bodies
   */
  calculateForce(bodyId: string): Vector3D {
    const body = this.bodies.get(bodyId);
    if (!body || !this.root) {
      return { x: 0, y: 0, z: 0 };
    }

    return this.calculateForceOnBody(this.root, body);
  }

  /**
   * Calculate forces for all bodies
   */
  calculateAllForces(): Map<string, Vector3D> {
    const forces = new Map<string, Vector3D>();

    for (const [id, body] of this.bodies) {
      forces.set(id, this.calculateForce(id));
    }

    return forces;
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  private calculateBounds(bodies: BodyData[]): BoundingBox {
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (const body of bodies) {
      minX = Math.min(minX, body.position.x);
      minY = Math.min(minY, body.position.y);
      minZ = Math.min(minZ, body.position.z);
      maxX = Math.max(maxX, body.position.x);
      maxY = Math.max(maxY, body.position.y);
      maxZ = Math.max(maxZ, body.position.z);
    }

    // Add padding
    const padding = 10;
    minX -= padding;
    minY -= padding;
    minZ -= padding;
    maxX += padding;
    maxY += padding;
    maxZ += padding;

    // Make cubic (equal size in all dimensions)
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

    const center: Vector3D = {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      z: (minZ + maxZ) / 2,
    };

    return {
      min: {
        x: center.x - size / 2,
        y: center.y - size / 2,
        z: center.z - size / 2,
      },
      max: {
        x: center.x + size / 2,
        y: center.y + size / 2,
        z: center.z + size / 2,
      },
      center,
      size,
    };
  }

  private createNode(bounds: BoundingBox, depth: number): OctreeNode {
    return {
      bounds,
      centerOfMass: { x: 0, y: 0, z: 0 },
      totalMass: 0,
      children: new Array(8).fill(null),
      body: null,
      isLeaf: true,
      depth,
    };
  }

  private getOctant(bounds: BoundingBox, position: Vector3D): number {
    const center = bounds.center;
    let octant = 0;

    if (position.x >= center.x) octant |= 1;
    if (position.y >= center.y) octant |= 2;
    if (position.z >= center.z) octant |= 4;

    return octant;
  }

  private getChildBounds(
    parentBounds: BoundingBox,
    octant: number
  ): BoundingBox {
    const halfSize = parentBounds.size / 2;
    const quarterSize = halfSize / 2;
    const center = parentBounds.center;

    const childCenter: Vector3D = {
      x: center.x + (octant & 1 ? quarterSize : -quarterSize),
      y: center.y + (octant & 2 ? quarterSize : -quarterSize),
      z: center.z + (octant & 4 ? quarterSize : -quarterSize),
    };

    return {
      min: {
        x: childCenter.x - quarterSize,
        y: childCenter.y - quarterSize,
        z: childCenter.z - quarterSize,
      },
      max: {
        x: childCenter.x + quarterSize,
        y: childCenter.y + quarterSize,
        z: childCenter.z + quarterSize,
      },
      center: childCenter,
      size: halfSize,
    };
  }

  private insert(node: OctreeNode, body: BodyData): void {
    // Check max depth or min cell size
    if (
      node.depth >= this.config.maxDepth ||
      node.bounds.size < this.config.minCellSize
    ) {
      // Force leaf - accumulate mass
      node.totalMass += body.mass;
      if (!node.body) {
        node.body = body;
      }
      return;
    }

    if (node.isLeaf) {
      if (node.body === null) {
        // Empty leaf - place body here
        node.body = body;
        node.totalMass = body.mass;
        return;
      }

      // Leaf with existing body - subdivide
      const existingBody = node.body;
      node.body = null;
      node.isLeaf = false;

      // Re-insert existing body
      const existingOctant = this.getOctant(node.bounds, existingBody.position);
      if (node.children[existingOctant] === null) {
        node.children[existingOctant] = this.createNode(
          this.getChildBounds(node.bounds, existingOctant),
          node.depth + 1
        );
      }
      this.insert(node.children[existingOctant]!, existingBody);
    }

    // Insert new body
    const octant = this.getOctant(node.bounds, body.position);
    if (node.children[octant] === null) {
      node.children[octant] = this.createNode(
        this.getChildBounds(node.bounds, octant),
        node.depth + 1
      );
    }
    this.insert(node.children[octant]!, body);
  }

  private calculateCenterOfMass(node: OctreeNode): void {
    if (node.isLeaf) {
      if (node.body) {
        node.centerOfMass = { ...node.body.position };
        node.totalMass = node.body.mass;
      }
      return;
    }

    let totalMass = 0;
    const weightedSum: Vector3D = { x: 0, y: 0, z: 0 };

    for (const child of node.children) {
      if (child !== null) {
        this.calculateCenterOfMass(child);
        totalMass += child.totalMass;
        weightedSum.x += child.centerOfMass.x * child.totalMass;
        weightedSum.y += child.centerOfMass.y * child.totalMass;
        weightedSum.z += child.centerOfMass.z * child.totalMass;
      }
    }

    node.totalMass = totalMass;
    if (totalMass > 0) {
      node.centerOfMass = vecScale(weightedSum, 1 / totalMass);
    }
  }

  private calculateForceOnBody(node: OctreeNode, body: BodyData): Vector3D {
    if (node.totalMass === 0) {
      return { x: 0, y: 0, z: 0 };
    }

    const direction = vecSub(node.centerOfMass, body.position);
    const distance = vecLength(direction) + this.config.softening;

    // Barnes-Hut criterion: s/d < theta
    const ratio = node.bounds.size / distance;

    if (node.isLeaf || ratio < this.config.theta) {
      // Treat as single body
      if (node.body && node.body.id === body.id) {
        return { x: 0, y: 0, z: 0 }; // Don't calculate force on self
      }

      // Gravitational force: F = G * m1 * m2 / r²
      const forceMagnitude =
        (this.config.gravity * body.mass * node.totalMass) /
        (distance * distance);

      return vecScale(vecNormalize(direction), forceMagnitude);
    }

    // Recurse into children
    let totalForce: Vector3D = { x: 0, y: 0, z: 0 };
    for (const child of node.children) {
      if (child !== null) {
        const childForce = this.calculateForceOnBody(child, body);
        totalForce = vecAdd(totalForce, childForce);
      }
    }

    return totalForce;
  }

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  getStatistics(): {
    nodeCount: number;
    bodyCount: number;
    maxDepth: number;
    avgDepth: number;
  } {
    if (!this.root) {
      return { nodeCount: 0, bodyCount: 0, maxDepth: 0, avgDepth: 0 };
    }

    let nodeCount = 0;
    let maxDepth = 0;
    let totalDepth = 0;
    let leafCount = 0;

    const traverse = (node: OctreeNode): void => {
      nodeCount++;
      if (node.isLeaf) {
        leafCount++;
        totalDepth += node.depth;
        maxDepth = Math.max(maxDepth, node.depth);
      } else {
        for (const child of node.children) {
          if (child) traverse(child);
        }
      }
    };

    traverse(this.root);

    return {
      nodeCount,
      bodyCount: this.bodies.size,
      maxDepth,
      avgDepth: leafCount > 0 ? totalDepth / leafCount : 0,
    };
  }
}
