/**
 * @file Hierarchical Graph Layout
 * @description Tree-based hierarchical layout for directed acyclic graphs
 * @module @neurectomy/3d-engine/visualization/graph/layouts
 * @agents @VERTEX @ARCHITECT
 */

import type { GraphNode, GraphEdge } from "../types";
import type { LayoutResult } from "./manager";

// ============================================================================
// Types
// ============================================================================

export interface HierarchicalConfig {
  /** Direction of the hierarchy */
  direction: "TB" | "BT" | "LR" | "RL";
  /** Level separation distance */
  levelSeparation: number;
  /** Node separation within a level */
  nodeSeparation: number;
  /** Sibling separation */
  siblingSeparation: number;
  /** Subtree separation */
  subtreeSeparation: number;
  /** Sorting method for children */
  sortMethod: "original" | "hubsize" | "directed";
  /** Enable 3D depth separation */
  enable3D: boolean;
  /** Depth separation in 3D mode */
  depthSeparation: number;
}

interface TreeNode {
  id: string;
  graphNode: GraphNode;
  parent: TreeNode | null;
  children: TreeNode[];
  level: number;
  x: number;
  y: number;
  z: number;
  mod: number;
  thread: TreeNode | null;
  ancestor: TreeNode;
  change: number;
  shift: number;
  number: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_HIERARCHICAL_CONFIG: HierarchicalConfig = {
  direction: "TB",
  levelSeparation: 5,
  nodeSeparation: 2,
  siblingSeparation: 1,
  subtreeSeparation: 2,
  sortMethod: "hubsize",
  enable3D: true,
  depthSeparation: 3,
};

// ============================================================================
// Hierarchical Layout Engine
// ============================================================================

/**
 * Tree-based hierarchical layout algorithm
 * Uses Buchheim's algorithm for efficient tree layout
 */
export class HierarchicalLayout {
  private config: HierarchicalConfig;

  constructor(config: Partial<HierarchicalConfig> = {}) {
    this.config = { ...DEFAULT_HIERARCHICAL_CONFIG, ...config };
  }

  /**
   * Apply hierarchical layout to nodes and edges
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

    // Find roots (nodes with no incoming edges)
    const roots = this.findRoots(nodes, edges);

    if (roots.length === 0 && nodes.length > 0) {
      // If no clear roots, pick node with most outgoing edges
      const sorted = [...nodes].sort((a, b) => {
        const aOut = adjacency.get(a.id)?.length ?? 0;
        const bOut = adjacency.get(b.id)?.length ?? 0;
        return bOut - aOut;
      });
      if (sorted[0]) {
        roots.push(sorted[0]);
      }
    }

    // Build trees from roots
    const forest = this.buildForest(nodes, edges, roots, adjacency);

    // Calculate layout for each tree
    let xOffset = 0;
    for (const root of forest) {
      this.firstWalk(root);
      this.secondWalk(root, -root.x, 0);

      // Apply offset and get tree width
      const { width } = this.applyPositions(root, xOffset);
      xOffset += width + this.config.subtreeSeparation;
    }

    // Center the entire forest
    const totalWidth = xOffset - this.config.subtreeSeparation;
    const centerOffset = -totalWidth / 2;

    for (const root of forest) {
      this.offsetTree(root, centerOffset, 0, 0);
    }

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
        direction: this.config.direction,
        treeCount: forest.length,
      },
    };
  }

  /**
   * Build adjacency lists from edges
   */
  private buildAdjacency(
    nodes: GraphNode[],
    edges: GraphEdge[]
  ): Map<string, string[]> {
    const adjacency = new Map<string, string[]>();

    for (const node of nodes) {
      adjacency.set(node.id, []);
    }

    for (const edge of edges) {
      const children = adjacency.get(edge.sourceId);
      if (children) {
        children.push(edge.targetId);
      }
    }

    return adjacency;
  }

  /**
   * Find root nodes (no incoming edges)
   */
  private findRoots(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
    const hasIncoming = new Set<string>();

    for (const edge of edges) {
      hasIncoming.add(edge.targetId);
    }

    return nodes.filter((node) => !hasIncoming.has(node.id));
  }

  /**
   * Build forest of trees from graph
   */
  private buildForest(
    nodes: GraphNode[],
    edges: GraphEdge[],
    roots: GraphNode[],
    adjacency: Map<string, string[]>
  ): TreeNode[] {
    const nodeMap = new Map<string, GraphNode>();
    for (const node of nodes) {
      nodeMap.set(node.id, node);
    }

    const visited = new Set<string>();
    const forest: TreeNode[] = [];

    const buildTree = (
      graphNode: GraphNode,
      parent: TreeNode | null,
      level: number
    ): TreeNode => {
      const treeNode: TreeNode = {
        id: graphNode.id,
        graphNode,
        parent,
        children: [],
        level,
        x: 0,
        y: 0,
        z: 0,
        mod: 0,
        thread: null,
        ancestor: null as unknown as TreeNode,
        change: 0,
        shift: 0,
        number: 0,
      };
      treeNode.ancestor = treeNode;

      visited.add(graphNode.id);

      const childIds = adjacency.get(graphNode.id) ?? [];

      // Sort children based on config
      const sortedChildIds = this.sortChildren(childIds, nodeMap, adjacency);

      for (const childId of sortedChildIds) {
        if (visited.has(childId)) continue;

        const childNode = nodeMap.get(childId);
        if (childNode) {
          const childTree = buildTree(childNode, treeNode, level + 1);
          treeNode.children.push(childTree);
        }
      }

      // Assign sibling numbers
      treeNode.children.forEach((child, i) => {
        child.number = i + 1;
      });

      return treeNode;
    };

    for (const root of roots) {
      if (!visited.has(root.id)) {
        forest.push(buildTree(root, null, 0));
      }
    }

    // Handle disconnected nodes
    for (const node of nodes) {
      if (!visited.has(node.id)) {
        forest.push(buildTree(node, null, 0));
      }
    }

    return forest;
  }

  /**
   * Sort children based on configuration
   */
  private sortChildren(
    childIds: string[],
    nodeMap: Map<string, GraphNode>,
    adjacency: Map<string, string[]>
  ): string[] {
    if (this.config.sortMethod === "original") {
      return childIds;
    }

    return [...childIds].sort((a, b) => {
      const aChildren = adjacency.get(a)?.length ?? 0;
      const bChildren = adjacency.get(b)?.length ?? 0;

      if (this.config.sortMethod === "hubsize") {
        return bChildren - aChildren;
      }

      // 'directed' - use node label or alphabetical
      const aNode = nodeMap.get(a);
      const bNode = nodeMap.get(b);
      const aLabel = aNode?.label ?? a;
      const bLabel = bNode?.label ?? b;
      return String(aLabel).localeCompare(String(bLabel));
    });
  }

  /**
   * First walk of Buchheim's algorithm
   * Computes preliminary x-coordinates
   */
  private firstWalk(v: TreeNode): void {
    if (v.children.length === 0) {
      // Leaf node
      if (this.leftSibling(v)) {
        v.x = this.leftSibling(v)!.x + this.config.siblingSeparation;
      } else {
        v.x = 0;
      }
    } else {
      // Internal node
      let defaultAncestor = v.children[0]!;

      for (const w of v.children) {
        this.firstWalk(w);
        defaultAncestor = this.apportion(w, defaultAncestor);
      }

      this.executeShifts(v);

      const midpoint =
        (v.children[0]!.x + v.children[v.children.length - 1]!.x) / 2;

      const w = this.leftSibling(v);
      if (w) {
        v.x = w.x + this.config.siblingSeparation;
        v.mod = v.x - midpoint;
      } else {
        v.x = midpoint;
      }
    }
  }

  /**
   * Second walk of Buchheim's algorithm
   * Adds modifiers to get final x-coordinates and computes y-coordinates
   */
  private secondWalk(v: TreeNode, m: number, depth: number): void {
    v.x += m;

    // Calculate y based on level
    const levelSign = this.getLevelSign();
    v.y = v.level * this.config.levelSeparation * levelSign;

    // Calculate z for 3D
    if (this.config.enable3D) {
      v.z = depth * this.config.depthSeparation;
    }

    for (let i = 0; i < v.children.length; i++) {
      // Vary depth for visual interest
      const childDepth = depth + (i % 2 === 0 ? 0.5 : -0.5);
      this.secondWalk(v.children[i]!, m + v.mod, childDepth);
    }
  }

  /**
   * Get level direction multiplier
   */
  private getLevelSign(): number {
    switch (this.config.direction) {
      case "TB":
        return 1;
      case "BT":
        return -1;
      case "LR":
        return 1;
      case "RL":
        return -1;
    }
  }

  /**
   * Apportion subtrees
   */
  private apportion(v: TreeNode, defaultAncestor: TreeNode): TreeNode {
    const w = this.leftSibling(v);
    if (w) {
      let vip: TreeNode | null = v;
      let vop: TreeNode | null = v;
      let vim: TreeNode | null = w;
      let vom: TreeNode | null = this.leftmostSibling(vip);

      if (!vom) return defaultAncestor;

      let sip = vip.mod;
      let sop = vop.mod;
      let sim = vim.mod;
      let som = vom.mod;

      while (vim && vip && this.nextRight(vim) && this.nextLeft(vip)) {
        vim = this.nextRight(vim);
        vip = this.nextLeft(vip);
        vom = vom ? this.nextLeft(vom) : null;
        vop = vop ? this.nextRight(vop) : null;

        if (!vim || !vip || !vom || !vop) break;

        vop.ancestor = v;

        const shift =
          vim.x + sim - (vip.x + sip) + this.config.subtreeSeparation;
        if (shift > 0) {
          const a = this.ancestor(vim, v, defaultAncestor);
          this.moveSubtree(a, v, shift);
          sip += shift;
          sop += shift;
        }

        sim += vim.mod;
        sip += vip.mod;
        som += vom.mod;
        sop += vop.mod;
      }

      if (vim && vop && this.nextRight(vim) && !this.nextRight(vop)) {
        vop.thread = this.nextRight(vim);
        vop.mod += sim - sop;
      }

      if (vip && vom && this.nextLeft(vip) && !this.nextLeft(vom)) {
        vom.thread = this.nextLeft(vip);
        vom.mod += sip - som;
        defaultAncestor = v;
      }
    }

    return defaultAncestor;
  }

  /**
   * Execute accumulated shifts
   */
  private executeShifts(v: TreeNode): void {
    let shift = 0;
    let change = 0;

    for (let i = v.children.length - 1; i >= 0; i--) {
      const w = v.children[i]!;
      w.x += shift;
      w.mod += shift;
      change += w.change;
      shift += w.shift + change;
    }
  }

  /**
   * Move subtree
   */
  private moveSubtree(wm: TreeNode, wp: TreeNode, shift: number): void {
    const subtrees = wp.number - wm.number;
    wp.change -= shift / subtrees;
    wp.shift += shift;
    wm.change += shift / subtrees;
    wp.x += shift;
    wp.mod += shift;
  }

  /**
   * Get ancestor for apportion
   */
  private ancestor(
    vim: TreeNode,
    v: TreeNode,
    defaultAncestor: TreeNode
  ): TreeNode {
    if (v.parent && v.parent.children.includes(vim.ancestor)) {
      return vim.ancestor;
    }
    return defaultAncestor;
  }

  /**
   * Navigation helpers
   */
  private leftSibling(v: TreeNode): TreeNode | null {
    if (!v.parent) return null;
    const siblings = v.parent.children;
    const idx = siblings.indexOf(v);
    return idx > 0 ? (siblings[idx - 1] ?? null) : null;
  }

  private leftmostSibling(v: TreeNode): TreeNode | null {
    if (!v.parent) return null;
    return v.parent.children[0] ?? null;
  }

  private nextLeft(v: TreeNode): TreeNode | null {
    if (v.children.length > 0) {
      return v.children[0] ?? null;
    }
    return v.thread;
  }

  private nextRight(v: TreeNode): TreeNode | null {
    if (v.children.length > 0) {
      return v.children[v.children.length - 1] ?? null;
    }
    return v.thread;
  }

  /**
   * Apply calculated positions to graph nodes
   */
  private applyPositions(root: TreeNode, xOffset: number): { width: number } {
    let minX = Infinity;
    let maxX = -Infinity;

    const apply = (node: TreeNode): void => {
      // Swap coordinates based on direction
      let finalX = node.x;
      let finalY = node.y;

      if (this.config.direction === "LR" || this.config.direction === "RL") {
        finalX = node.y;
        finalY = node.x;
      }

      node.graphNode.position.x = finalX + xOffset;
      node.graphNode.position.y = finalY;
      node.graphNode.position.z = this.config.enable3D ? node.z : 0;

      minX = Math.min(minX, node.graphNode.position.x);
      maxX = Math.max(maxX, node.graphNode.position.x);

      for (const child of node.children) {
        apply(child);
      }
    };

    apply(root);

    return { width: maxX - minX + this.config.nodeSeparation };
  }

  /**
   * Offset entire tree
   */
  private offsetTree(node: TreeNode, dx: number, dy: number, dz: number): void {
    node.graphNode.position.x += dx;
    node.graphNode.position.y += dy;
    node.graphNode.position.z += dz;

    for (const child of node.children) {
      this.offsetTree(child, dx, dy, dz);
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<HierarchicalConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
