/**
 * @file Neo4j Data Transformer
 * @description Transform Neo4j query results into visualization-ready data
 * @module @neurectomy/3d-engine/visualization/graph/neo4j
 * @agents @VERTEX @PRISM
 */

import type { GraphNode, GraphEdge, NodeType, EdgeType } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface Neo4jNode {
  identity: string | number;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface Neo4jRelationship {
  identity: string | number;
  type: string;
  startNodeId: string | number;
  endNodeId: string | number;
  properties: Record<string, unknown>;
}

export interface Neo4jPath {
  nodes: Neo4jNode[];
  relationships: Neo4jRelationship[];
}

export interface TransformOptions {
  /** Layout algorithm for initial positioning */
  layout?: "spiral" | "grid" | "random" | "cluster";
  /** Spacing between nodes */
  spacing?: number;
  /** Custom label to node type mapping */
  labelMappings?: Record<string, NodeType>;
  /** Custom relationship type to edge type mapping */
  relationshipMappings?: Record<string, EdgeType>;
  /** Property to use as node label */
  labelProperty?: string;
  /** Property to use for node size */
  sizeProperty?: string;
  /** Property to use for node color */
  colorProperty?: string;
  /** Default node size */
  defaultNodeSize?: number;
  /** Enable hierarchical layout based on relationships */
  hierarchical?: boolean;
  /** Group nodes by label */
  groupByLabel?: boolean;
}

export interface TransformedGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: {
    nodeCount: number;
    edgeCount: number;
    labels: string[];
    relationshipTypes: string[];
    clusters: Map<string, string[]>;
  };
}

// ============================================================================
// Default Mappings
// ============================================================================

const DEFAULT_LABEL_MAPPINGS: Record<string, NodeType> = {
  Agent: "agent",
  LLM: "llm",
  Tool: "tool",
  Memory: "memory",
  Database: "database",
  API: "api",
  User: "user",
  System: "system",
  Process: "process",
  Service: "service",
};

const DEFAULT_RELATIONSHIP_MAPPINGS: Record<string, EdgeType> = {
  CONNECTS_TO: "connection",
  DEPENDS_ON: "dependency",
  EXTENDS: "hierarchy",
  OWNS: "association",
  USES: "association",
  SENDS_TO: "dataflow",
  RECEIVES_FROM: "dataflow",
  TRIGGERS: "control",
  MONITORS: "reference",
  SYNCS_WITH: "connection",
};

const NODE_TYPE_COLORS: Record<NodeType, string> = {
  agent: "#4a90d9",
  llm: "#9b59b6",
  tool: "#27ae60",
  memory: "#f39c12",
  database: "#e74c3c",
  api: "#1abc9c",
  user: "#3498db",
  system: "#95a5a6",
  process: "#e67e22",
  service: "#2ecc71",
  custom: "#7f8c8d",
};

const EDGE_TYPE_COLORS: Record<EdgeType, string> = {
  connection: "#64748b",
  dependency: "#ef4444",
  hierarchy: "#8b5cf6",
  reference: "#a855f7",
  association: "#22c55e",
  dataflow: "#06b6d4",
  control: "#f97316",
  custom: "#6b7280",
};

// ============================================================================
// Layout Algorithms
// ============================================================================

interface PositionGenerator {
  next(): { x: number; y: number; z: number };
}

function createSpiralLayout(spacing: number): PositionGenerator {
  let index = 0;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  return {
    next() {
      const angle = index * goldenAngle;
      const radius = Math.sqrt(index) * spacing;
      const y = (index % 10) * spacing * 0.2 - spacing;
      index++;

      return {
        x: Math.cos(angle) * radius,
        y,
        z: Math.sin(angle) * radius,
      };
    },
  };
}

function createGridLayout(spacing: number): PositionGenerator {
  let index = 0;
  const gridSize = Math.ceil(Math.sqrt(1000)); // Assume max 1000 nodes

  return {
    next() {
      const x = (index % gridSize) * spacing - (gridSize * spacing) / 2;
      const z =
        Math.floor(index / gridSize) * spacing - (gridSize * spacing) / 2;
      const y = 0;
      index++;

      return { x, y, z };
    },
  };
}

function createRandomLayout(spacing: number): PositionGenerator {
  const range = spacing * 10;

  return {
    next() {
      return {
        x: (Math.random() - 0.5) * range,
        y: (Math.random() - 0.5) * range * 0.5,
        z: (Math.random() - 0.5) * range,
      };
    },
  };
}

function createClusterLayout(spacing: number): PositionGenerator {
  const clusterMap = new Map<string, { x: number; z: number }>();
  let clusterCount = 0;
  let currentCluster = "";
  let indexInCluster = 0;

  return {
    next() {
      // This is a simplified version - actual implementation would track clusters
      const clusterOffset = clusterMap.get(currentCluster) || { x: 0, z: 0 };
      const angle = indexInCluster * 0.5;
      const radius = Math.sqrt(indexInCluster) * spacing * 0.5;

      indexInCluster++;

      return {
        x: clusterOffset.x + Math.cos(angle) * radius,
        y: 0,
        z: clusterOffset.z + Math.sin(angle) * radius,
      };
    },
  };
}

function getLayoutGenerator(
  layout: string,
  spacing: number
): PositionGenerator {
  switch (layout) {
    case "grid":
      return createGridLayout(spacing);
    case "random":
      return createRandomLayout(spacing);
    case "cluster":
      return createClusterLayout(spacing);
    case "spiral":
    default:
      return createSpiralLayout(spacing);
  }
}

// ============================================================================
// Neo4j Data Transformer
// ============================================================================

/**
 * Transform Neo4j data into visualization-ready format
 */
export class Neo4jDataTransformer {
  private options: Required<TransformOptions>;
  private positionGenerator: PositionGenerator;
  private nodeIdMap: Map<string | number, string> = new Map();
  private labelSet: Set<string> = new Set();
  private relationshipTypeSet: Set<string> = new Set();
  private clusterMap: Map<string, string[]> = new Map();

  constructor(options: TransformOptions = {}) {
    this.options = {
      layout: options.layout ?? "spiral",
      spacing: options.spacing ?? 2,
      labelMappings: { ...DEFAULT_LABEL_MAPPINGS, ...options.labelMappings },
      relationshipMappings: {
        ...DEFAULT_RELATIONSHIP_MAPPINGS,
        ...options.relationshipMappings,
      },
      labelProperty: options.labelProperty ?? "name",
      sizeProperty: options.sizeProperty ?? "size",
      colorProperty: options.colorProperty ?? "color",
      defaultNodeSize: options.defaultNodeSize ?? 0.3,
      hierarchical: options.hierarchical ?? false,
      groupByLabel: options.groupByLabel ?? true,
    };

    this.positionGenerator = getLayoutGenerator(
      this.options.layout,
      this.options.spacing
    );
  }

  /**
   * Transform Neo4j nodes and relationships to graph visualization
   */
  transform(
    nodes: Neo4jNode[],
    relationships: Neo4jRelationship[]
  ): TransformedGraph {
    // Reset state
    this.nodeIdMap.clear();
    this.labelSet.clear();
    this.relationshipTypeSet.clear();
    this.clusterMap.clear();
    this.positionGenerator = getLayoutGenerator(
      this.options.layout,
      this.options.spacing
    );

    // Process nodes first
    const graphNodes = nodes.map((neo4jNode, index) =>
      this.transformNode(neo4jNode, index)
    );

    // Apply hierarchical layout if enabled
    if (this.options.hierarchical) {
      this.applyHierarchicalLayout(graphNodes, relationships);
    }

    // Process relationships
    const graphEdges = relationships
      .filter((rel) => {
        // Only include edges where both nodes exist
        const sourceId = this.nodeIdMap.get(rel.startNodeId);
        const targetId = this.nodeIdMap.get(rel.endNodeId);
        return sourceId && targetId;
      })
      .map((rel) => this.transformRelationship(rel));

    return {
      nodes: graphNodes,
      edges: graphEdges,
      metadata: {
        nodeCount: graphNodes.length,
        edgeCount: graphEdges.length,
        labels: Array.from(this.labelSet),
        relationshipTypes: Array.from(this.relationshipTypeSet),
        clusters: this.clusterMap,
      },
    };
  }

  /**
   * Transform a single Neo4j node to GraphNode
   */
  private transformNode(neo4jNode: Neo4jNode, index: number): GraphNode {
    const id = `node-${neo4jNode.identity}`;
    this.nodeIdMap.set(neo4jNode.identity, id);

    // Track labels
    neo4jNode.labels.forEach((label) => this.labelSet.add(label));

    // Track clusters
    if (this.options.groupByLabel && neo4jNode.labels.length > 0) {
      const primaryLabel = neo4jNode.labels[0]!;
      if (!this.clusterMap.has(primaryLabel)) {
        this.clusterMap.set(primaryLabel, []);
      }
      this.clusterMap.get(primaryLabel)!.push(id);
    }

    // Determine node type
    let nodeType: NodeType = "custom";
    for (const label of neo4jNode.labels) {
      if (this.options.labelMappings[label]) {
        nodeType = this.options.labelMappings[label];
        break;
      }
    }

    // Get label from properties
    const label = this.getPropertyString(
      neo4jNode.properties,
      this.options.labelProperty,
      `Node ${neo4jNode.identity}`
    );

    // Get size from properties or use default
    const radius = this.getPropertyNumber(
      neo4jNode.properties,
      this.options.sizeProperty,
      this.options.defaultNodeSize
    );

    // Get color from properties or use type-based color
    const color = this.getPropertyString(
      neo4jNode.properties,
      this.options.colorProperty,
      NODE_TYPE_COLORS[nodeType]
    );

    // Get position from layout generator
    const position = this.positionGenerator.next();

    return {
      id,
      label,
      type: nodeType,
      position,
      velocity: { x: 0, y: 0, z: 0 },
      mass: Math.max(1, radius * 3),
      radius,
      color,
      pinned: false,
      metadata: {
        description: this.getPropertyString(
          neo4jNode.properties,
          "description"
        ),
        category: neo4jNode.labels[0],
        tags: neo4jNode.labels,
        properties: neo4jNode.properties,
      },
      state: {
        selected: false,
        hovered: false,
        dragging: false,
        highlighted: false,
        dimmed: false,
        visible: true,
        activity: 0,
      },
    };
  }

  /**
   * Transform a Neo4j relationship to GraphEdge
   */
  private transformRelationship(rel: Neo4jRelationship): GraphEdge {
    const id = `edge-${rel.identity}`;
    const sourceId = this.nodeIdMap.get(rel.startNodeId)!;
    const targetId = this.nodeIdMap.get(rel.endNodeId)!;

    // Track relationship types
    this.relationshipTypeSet.add(rel.type);

    // Determine edge type
    const edgeType = this.options.relationshipMappings[rel.type] || "custom";
    const color = EDGE_TYPE_COLORS[edgeType];

    // Get weight from properties
    const weight = this.getPropertyNumber(rel.properties, "weight", 1);
    const width = Math.max(0.5, Math.min(3, weight));

    return {
      id,
      sourceId,
      targetId,
      type: edgeType,
      direction: "forward",
      weight,
      length: weight * 100,
      color,
      width,
      metadata: {
        label: rel.type,
        properties: rel.properties,
      },
      state: {
        selected: false,
        hovered: false,
        highlighted: false,
        dimmed: false,
        visible: true,
        animationProgress: 0,
      },
    };
  }

  /**
   * Apply hierarchical layout based on relationship direction
   */
  private applyHierarchicalLayout(
    nodes: GraphNode[],
    relationships: Neo4jRelationship[]
  ): void {
    // Build adjacency list
    const outgoing = new Map<string, Set<string>>();
    const incoming = new Map<string, Set<string>>();

    for (const rel of relationships) {
      const sourceId = this.nodeIdMap.get(rel.startNodeId);
      const targetId = this.nodeIdMap.get(rel.endNodeId);

      if (sourceId && targetId) {
        if (!outgoing.has(sourceId)) outgoing.set(sourceId, new Set());
        if (!incoming.has(targetId)) incoming.set(targetId, new Set());

        outgoing.get(sourceId)!.add(targetId);
        incoming.get(targetId)!.add(sourceId);
      }
    }

    // Find root nodes (no incoming edges)
    const roots = nodes.filter(
      (n) => !incoming.has(n.id) || incoming.get(n.id)!.size === 0
    );

    // Calculate levels using BFS
    const levels = new Map<string, number>();
    const queue: Array<{ id: string; level: number }> = roots.map((n) => ({
      id: n.id,
      level: 0,
    }));

    while (queue.length > 0) {
      const { id, level } = queue.shift()!;

      if (levels.has(id)) continue;
      levels.set(id, level);

      const children = outgoing.get(id);
      if (children) {
        for (const childId of children) {
          if (!levels.has(childId)) {
            queue.push({ id: childId, level: level + 1 });
          }
        }
      }
    }

    // Position nodes by level
    const levelNodes = new Map<number, string[]>();
    for (const [id, level] of levels) {
      if (!levelNodes.has(level)) levelNodes.set(level, []);
      levelNodes.get(level)!.push(id);
    }

    // Apply positions
    const levelSpacing = this.options.spacing * 2;
    const nodeSpacing = this.options.spacing;

    for (const [level, nodeIds] of levelNodes) {
      const width = (nodeIds.length - 1) * nodeSpacing;
      nodeIds.forEach((nodeId, i) => {
        const node = nodes.find((n) => n.id === nodeId);
        if (node) {
          node.position = {
            x: i * nodeSpacing - width / 2,
            y: -level * levelSpacing,
            z: 0,
          };
        }
      });
    }
  }

  /**
   * Get string property from Neo4j properties
   */
  private getPropertyString(
    properties: Record<string, unknown>,
    key: string,
    defaultValue: string = ""
  ): string {
    const value = properties[key];
    if (typeof value === "string") return value;
    if (value !== null && value !== undefined) return String(value);
    return defaultValue;
  }

  /**
   * Get number property from Neo4j properties
   */
  private getPropertyNumber(
    properties: Record<string, unknown>,
    key: string,
    defaultValue: number = 0
  ): number {
    const value = properties[key];
    if (typeof value === "number") return value;
    if (typeof value === "string") {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) return parsed;
    }
    return defaultValue;
  }

  /**
   * Transform paths (for path queries)
   */
  transformPaths(paths: Neo4jPath[]): TransformedGraph {
    const allNodes: Neo4jNode[] = [];
    const allRelationships: Neo4jRelationship[] = [];
    const seenNodeIds = new Set<string | number>();
    const seenRelIds = new Set<string | number>();

    for (const path of paths) {
      for (const node of path.nodes) {
        if (!seenNodeIds.has(node.identity)) {
          seenNodeIds.add(node.identity);
          allNodes.push(node);
        }
      }
      for (const rel of path.relationships) {
        if (!seenRelIds.has(rel.identity)) {
          seenRelIds.add(rel.identity);
          allRelationships.push(rel);
        }
      }
    }

    return this.transform(allNodes, allRelationships);
  }

  /**
   * Update options
   */
  updateOptions(options: Partial<TransformOptions>): void {
    this.options = { ...this.options, ...options };
    this.positionGenerator = getLayoutGenerator(
      this.options.layout,
      this.options.spacing
    );
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createNeo4jTransformer(
  options?: TransformOptions
): Neo4jDataTransformer {
  return new Neo4jDataTransformer(options);
}
