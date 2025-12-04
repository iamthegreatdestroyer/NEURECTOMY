/**
 * @file Radial Graph Layout
 * @description Circular/radial layout for tree-like structures
 * @module @neurectomy/3d-engine/visualization/graph/layouts
 * @agents @VERTEX @CANVAS
 */

import type { GraphNode, GraphEdge } from "../types";
import type { LayoutResult } from "./manager";

// ============================================================================
// Types
// ============================================================================

export interface RadialConfig {
  /** Radius of the first level */
  startRadius: number;
  /** Radius increment per level */
  radiusIncrement: number;
  /** Starting angle in radians */
  startAngle: number;
  /** Total angle sweep in radians (2π for full circle) */
  sweepAngle: number;
  /** Minimum angle between nodes */
  minAngle: number;
  /** Enable 3D cone shape */
  enable3D: boolean;
  /** Z increment per level for 3D mode */
  zIncrement: number;
  /** Sort children by size */
  sortBySize: boolean;
}

interface RadialNode {
  id: string;
  graphNode: GraphNode;
  parent: RadialNode | null;
  children: RadialNode[];
  level: number;
  angle: number;
  radius: number;
  angleSpan: number;
  subtreeSize: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_RADIAL_CONFIG: RadialConfig = {
  startRadius: 3,
  radiusIncrement: 4,
  startAngle: 0,
  sweepAngle: Math.PI * 2,
  minAngle: 0.05,
  enable3D: true,
  zIncrement: 2,
  sortBySize: true,
};

// ============================================================================
// Radial Layout Engine
// ============================================================================

/**
 * Radial/circular layout algorithm for tree structures
 */
export class RadialLayout {
  private config: RadialConfig;

  constructor(config: Partial<RadialConfig> = {}) {
    this.config = { ...DEFAULT_RADIAL_CONFIG, ...config };
  }

  /**
   * Apply radial layout to nodes and edges
   */
  layout(nodes: GraphNode[], edges: GraphEdge[]): LayoutResult {
    const positions = new Map<string, { x: number; y: number; z: number }>();

    if (nodes.length === 0) {
      return {
        positions,
        bounds: {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 0, y: 0, z: 0 },
        },
        metadata: { empty: true },
      };
    }

    // Build adjacency lists
    const adjacency = this.buildAdjacency(nodes, edges);

    // Find root (node with most connections or first node)
    const root = this.findRoot(nodes, adjacency);

    // Build radial tree
    const tree = this.buildTree(nodes, root, adjacency);

    // Calculate subtree sizes
    this.calculateSubtreeSizes(tree);

    // Calculate angles
    this.calculateAngles(tree, this.config.startAngle, this.config.sweepAngle);

    // Apply positions
    this.applyPositions(tree);

    // Collect positions and calculate bounds
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (const node of nodes) {
      positions.set(node.id, { ...node.position });
      minX = Math.min(minX, node.position.x);
      minY = Math.min(minY, node.position.y);
      minZ = Math.min(minZ, node.position.z);
      maxX = Math.max(maxX, node.position.x);
      maxY = Math.max(maxY, node.position.y);
      maxZ = Math.max(maxZ, node.position.z);
    }

    return {
      positions,
      bounds: {
        min: { x: minX, y: minY, z: minZ },
        max: { x: maxX, y: maxY, z: maxZ },
      },
      metadata: {
        rootId: root.id,
        levels: this.getMaxLevel(tree),
      },
    };
  }

  /**
   * Get maximum level in tree
   */
  private getMaxLevel(root: RadialNode): number {
    let maxLevel = root.level;
    const traverse = (node: RadialNode): void => {
      maxLevel = Math.max(maxLevel, node.level);
      for (const child of node.children) {
        traverse(child);
      }
    };
    traverse(root);
    return maxLevel;
  }

  /**
   * Build adjacency lists from edges (undirected)
   */
  private buildAdjacency(
    nodes: GraphNode[],
    edges: GraphEdge[]
  ): Map<string, Set<string>> {
    const adjacency = new Map<string, Set<string>>();

    for (const node of nodes) {
      adjacency.set(node.id, new Set());
    }

    for (const edge of edges) {
      adjacency.get(edge.sourceId)?.add(edge.targetId);
      adjacency.get(edge.targetId)?.add(edge.sourceId);
    }

    return adjacency;
  }

  /**
   * Find root node (most connected)
   */
  private findRoot(
    nodes: GraphNode[],
    adjacency: Map<string, Set<string>>
  ): GraphNode {
    if (nodes.length === 0) {
      throw new Error("Cannot find root node: nodes array is empty");
    }

    let maxDegree = -1;
    let root: GraphNode = nodes[0]!;

    for (const node of nodes) {
      const degree = adjacency.get(node.id)?.size ?? 0;
      if (degree > maxDegree) {
        maxDegree = degree;
        root = node;
      }
    }

    return root;
  }

  /**
   * Build radial tree from graph using BFS
   */
  private buildTree(
    nodes: GraphNode[],
    root: GraphNode,
    adjacency: Map<string, Set<string>>
  ): RadialNode {
    const nodeMap = new Map<string, GraphNode>();
    for (const node of nodes) {
      nodeMap.set(node.id, node);
    }

    const visited = new Set<string>();

    const buildNode = (
      graphNode: GraphNode,
      parent: RadialNode | null,
      level: number
    ): RadialNode => {
      visited.add(graphNode.id);

      const radialNode: RadialNode = {
        id: graphNode.id,
        graphNode,
        parent,
        children: [],
        level,
        angle: 0,
        radius: this.config.startRadius + level * this.config.radiusIncrement,
        angleSpan: 0,
        subtreeSize: 1,
      };

      const neighbors = adjacency.get(graphNode.id) ?? new Set();
      const childIds = [...neighbors].filter((id) => !visited.has(id));

      for (const childId of childIds) {
        const childGraphNode = nodeMap.get(childId);
        if (childGraphNode) {
          const childRadial = buildNode(childGraphNode, radialNode, level + 1);
          radialNode.children.push(childRadial);
        }
      }

      // Sort children if configured
      if (this.config.sortBySize && radialNode.children.length > 1) {
        radialNode.children.sort((a, b) => b.subtreeSize - a.subtreeSize);
      }

      return radialNode;
    };

    const tree = buildNode(root, null, 0);

    // Handle disconnected nodes - add them as separate trees around the main tree
    const disconnected: RadialNode[] = [];
    for (const node of nodes) {
      if (!visited.has(node.id)) {
        const subtree = buildNode(node, null, 1);
        disconnected.push(subtree);
      }
    }

    // Attach disconnected as children of a virtual root
    if (disconnected.length > 0) {
      tree.children.push(...disconnected);
    }

    return tree;
  }

  /**
   * Calculate subtree sizes bottom-up
   */
  private calculateSubtreeSizes(node: RadialNode): number {
    if (node.children.length === 0) {
      node.subtreeSize = 1;
      return 1;
    }

    let size = 1;
    for (const child of node.children) {
      size += this.calculateSubtreeSizes(child);
    }
    node.subtreeSize = size;
    return size;
  }

  /**
   * Calculate angles for all nodes
   */
  private calculateAngles(
    node: RadialNode,
    startAngle: number,
    availableAngle: number
  ): void {
    node.angle = startAngle + availableAngle / 2;
    node.angleSpan = availableAngle;

    if (node.children.length === 0) return;

    // Distribute angle among children proportional to subtree size
    const totalChildSize = node.children.reduce(
      (sum, child) => sum + child.subtreeSize,
      0
    );

    let currentAngle = startAngle;

    for (const child of node.children) {
      const proportion = child.subtreeSize / totalChildSize;
      let childAngle = availableAngle * proportion;

      // Apply minimum angle constraint
      const minRequiredAngle = this.config.minAngle * child.subtreeSize;
      childAngle = Math.max(childAngle, minRequiredAngle);

      this.calculateAngles(child, currentAngle, childAngle);
      currentAngle += childAngle;
    }
  }

  /**
   * Apply calculated positions to graph nodes
   */
  private applyPositions(node: RadialNode): void {
    // Root at center
    if (node.level === 0) {
      node.graphNode.position.x = 0;
      node.graphNode.position.y = 0;
      node.graphNode.position.z = 0;
    } else {
      // Convert polar to cartesian
      const x = node.radius * Math.cos(node.angle);
      const y = node.radius * Math.sin(node.angle);
      const z = this.config.enable3D ? node.level * this.config.zIncrement : 0;

      node.graphNode.position.x = x;
      node.graphNode.position.y = y;
      node.graphNode.position.z = z;
    }

    // Process children
    for (const child of node.children) {
      this.applyPositions(child);
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<RadialConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
