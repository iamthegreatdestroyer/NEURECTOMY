/**
 * @file Neo4j Graph Data Adapter
 * @description Connects to Neo4j and transforms data for 3D visualization
 * @module @neurectomy/3d-engine/visualization/graph/adapters
 * @agents @VERTEX @SYNAPSE
 */

import type {
  GraphNode,
  GraphEdge,
  NodeMetadata,
  EdgeMetadata,
} from "../types";

// ============================================================================
// Types
// ============================================================================

export interface Neo4jConfig {
  /** Neo4j bolt URI */
  uri: string;
  /** Database name */
  database?: string;
  /** Authentication credentials */
  auth?: {
    username: string;
    password: string;
  };
  /** Connection timeout in ms */
  connectionTimeout?: number;
  /** Max connection pool size */
  maxConnectionPoolSize?: number;
}

export interface Neo4jNode {
  identity: number | string;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface Neo4jRelationship {
  identity: number | string;
  type: string;
  startNodeId: number | string;
  endNodeId: number | string;
  properties: Record<string, unknown>;
}

export interface Neo4jPath {
  start: Neo4jNode;
  end: Neo4jNode;
  segments: Array<{
    start: Neo4jNode;
    relationship: Neo4jRelationship;
    end: Neo4jNode;
  }>;
}

export interface CypherQueryResult {
  nodes: Neo4jNode[];
  relationships: Neo4jRelationship[];
  paths: Neo4jPath[];
}

export interface NodeStyleMapping {
  /** Map label to color */
  labelColors: Map<string, string>;
  /** Map label to icon */
  labelIcons: Map<string, string>;
  /** Map label to size */
  labelSizes: Map<string, number>;
}

export interface EdgeStyleMapping {
  /** Map relationship type to color */
  typeColors: Map<string, string>;
  /** Map relationship type to width */
  typeWidths: Map<string, number>;
  /** Map relationship type to dash pattern */
  typeDashes: Map<string, boolean>;
}

// ============================================================================
// Default Style Mappings
// ============================================================================

export const DEFAULT_NODE_STYLES: NodeStyleMapping = {
  labelColors: new Map([
    ["Person", "#4CAF50"],
    ["Organization", "#2196F3"],
    ["Location", "#FF9800"],
    ["Event", "#9C27B0"],
    ["Document", "#607D8B"],
    ["Concept", "#00BCD4"],
    ["default", "#757575"],
  ]),
  labelIcons: new Map([
    ["Person", "👤"],
    ["Organization", "🏢"],
    ["Location", "📍"],
    ["Event", "📅"],
    ["Document", "📄"],
    ["Concept", "💡"],
    ["default", "⚪"],
  ]),
  labelSizes: new Map([
    ["Person", 1.0],
    ["Organization", 1.2],
    ["Location", 1.0],
    ["Event", 1.1],
    ["Document", 0.9],
    ["Concept", 0.8],
    ["default", 1.0],
  ]),
};

export const DEFAULT_EDGE_STYLES: EdgeStyleMapping = {
  typeColors: new Map([
    ["KNOWS", "#4CAF50"],
    ["WORKS_AT", "#2196F3"],
    ["LOCATED_IN", "#FF9800"],
    ["ATTENDED", "#9C27B0"],
    ["REFERENCES", "#607D8B"],
    ["RELATED_TO", "#00BCD4"],
    ["default", "#9E9E9E"],
  ]),
  typeWidths: new Map([
    ["KNOWS", 2],
    ["WORKS_AT", 2],
    ["LOCATED_IN", 1.5],
    ["ATTENDED", 1.5],
    ["REFERENCES", 1],
    ["RELATED_TO", 1],
    ["default", 1],
  ]),
  typeDashes: new Map([
    ["REFERENCES", true],
    ["RELATED_TO", true],
    ["default", false],
  ]),
};

// ============================================================================
// Neo4j Adapter
// ============================================================================

/**
 * Adapter for converting Neo4j data to 3D graph visualization format
 *
 * @example
 * ```typescript
 * const adapter = new Neo4jAdapter({
 *   uri: 'bolt://localhost:7687',
 *   auth: { username: 'neo4j', password: 'password' }
 * });
 *
 * const { nodes, edges } = await adapter.query('MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100');
 * ```
 */
export class Neo4jAdapter {
  private config: Neo4jConfig;
  private nodeStyles: NodeStyleMapping;
  private edgeStyles: EdgeStyleMapping;
  private seenNodes: Map<string, GraphNode>;
  private seenEdges: Map<string, GraphEdge>;

  constructor(
    config: Neo4jConfig,
    nodeStyles: Partial<NodeStyleMapping> = {},
    edgeStyles: Partial<EdgeStyleMapping> = {}
  ) {
    this.config = config;
    this.nodeStyles = {
      labelColors: new Map([
        ...DEFAULT_NODE_STYLES.labelColors,
        ...(nodeStyles.labelColors ?? []),
      ]),
      labelIcons: new Map([
        ...DEFAULT_NODE_STYLES.labelIcons,
        ...(nodeStyles.labelIcons ?? []),
      ]),
      labelSizes: new Map([
        ...DEFAULT_NODE_STYLES.labelSizes,
        ...(nodeStyles.labelSizes ?? []),
      ]),
    };
    this.edgeStyles = {
      typeColors: new Map([
        ...DEFAULT_EDGE_STYLES.typeColors,
        ...(edgeStyles.typeColors ?? []),
      ]),
      typeWidths: new Map([
        ...DEFAULT_EDGE_STYLES.typeWidths,
        ...(edgeStyles.typeWidths ?? []),
      ]),
      typeDashes: new Map([
        ...DEFAULT_EDGE_STYLES.typeDashes,
        ...(edgeStyles.typeDashes ?? []),
      ]),
    };
    this.seenNodes = new Map();
    this.seenEdges = new Map();
  }

  /**
   * Execute a Cypher query and return visualization data
   * Note: This is a mock implementation. In production, use neo4j-driver.
   */
  async query(
    cypher: string,
    params: Record<string, unknown> = {}
  ): Promise<{
    nodes: GraphNode[];
    edges: GraphEdge[];
    metadata: {
      query: string;
      params: Record<string, unknown>;
      nodeCount: number;
      edgeCount: number;
      labels: string[];
      relationshipTypes: string[];
    };
  }> {
    // In production, this would use neo4j-driver:
    // const driver = neo4j.driver(this.config.uri, neo4j.auth.basic(...));
    // const session = driver.session({ database: this.config.database });
    // const result = await session.run(cypher, params);

    console.log(`Neo4j Query: ${cypher}`, params);

    // For now, return empty result
    // The actual implementation would process result.records
    return {
      nodes: Array.from(this.seenNodes.values()),
      edges: Array.from(this.seenEdges.values()),
      metadata: {
        query: cypher,
        params,
        nodeCount: this.seenNodes.size,
        edgeCount: this.seenEdges.size,
        labels: this.getUniqueLabels(),
        relationshipTypes: this.getUniqueRelationshipTypes(),
      },
    };
  }

  /**
   * Convert raw Neo4j query result to visualization format
   */
  convertQueryResult(result: CypherQueryResult): {
    nodes: GraphNode[];
    edges: GraphEdge[];
  } {
    // Process nodes
    for (const neo4jNode of result.nodes) {
      this.processNeo4jNode(neo4jNode);
    }

    // Process relationships
    for (const rel of result.relationships) {
      this.processNeo4jRelationship(rel);
    }

    // Process paths
    for (const path of result.paths) {
      this.processNeo4jPath(path);
    }

    return {
      nodes: Array.from(this.seenNodes.values()),
      edges: Array.from(this.seenEdges.values()),
    };
  }

  /**
   * Convert a Neo4j node to GraphNode
   */
  private processNeo4jNode(neo4jNode: Neo4jNode): GraphNode {
    const id = String(neo4jNode.identity);

    if (this.seenNodes.has(id)) {
      return this.seenNodes.get(id)!;
    }

    const primaryLabel = neo4jNode.labels[0] ?? "default";
    const color =
      this.nodeStyles.labelColors.get(primaryLabel) ??
      this.nodeStyles.labelColors.get("default") ??
      "#757575";
    const size =
      this.nodeStyles.labelSizes.get(primaryLabel) ??
      this.nodeStyles.labelSizes.get("default") ??
      1.0;
    const icon =
      this.nodeStyles.labelIcons.get(primaryLabel) ??
      this.nodeStyles.labelIcons.get("default") ??
      "⚪";

    const label = this.extractLabel(neo4jNode);

    const node: GraphNode = {
      id,
      label,
      type: this.mapLabelToNodeType(primaryLabel),
      position: {
        x: Math.random() * 20 - 10,
        y: Math.random() * 20 - 10,
        z: Math.random() * 10 - 5,
      },
      velocity: { x: 0, y: 0, z: 0 },
      mass: size * 10,
      radius: size,
      color,
      pinned: false,
      metadata: {
        description: `Neo4j node: ${primaryLabel}`,
        category: primaryLabel,
        tags: neo4jNode.labels,
        properties: {
          neo4jId: neo4jNode.identity,
          icon,
          ...neo4jNode.properties,
        },
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

    this.seenNodes.set(id, node);
    return node;
  }

  /**
   * Map Neo4j label to NodeType
   */
  private mapLabelToNodeType(label: string): GraphNode["type"] {
    const labelLower = label.toLowerCase();
    const typeMap: Record<string, GraphNode["type"]> = {
      agent: "agent",
      llm: "llm",
      tool: "tool",
      memory: "memory",
      database: "database",
      api: "api",
      user: "user",
      system: "system",
      process: "process",
      service: "service",
    };
    return typeMap[labelLower] ?? "custom";
  }

  /**
   * Extract display label from Neo4j node
   */
  private extractLabel(node: Neo4jNode): string {
    // Try common property names for labels
    const labelProps = ["name", "title", "label", "id", "displayName"];

    for (const prop of labelProps) {
      const value = node.properties[prop];
      if (typeof value === "string" && value.length > 0) {
        return value.length > 30 ? value.substring(0, 27) + "..." : value;
      }
    }

    // Fall back to first label + id
    return `${node.labels[0] ?? "Node"}:${node.identity}`;
  }

  /**
   * Convert a Neo4j relationship to GraphEdge
   */
  private processNeo4jRelationship(rel: Neo4jRelationship): GraphEdge {
    const id = `rel-${rel.identity}`;

    if (this.seenEdges.has(id)) {
      return this.seenEdges.get(id)!;
    }

    const color =
      this.edgeStyles.typeColors.get(rel.type) ??
      this.edgeStyles.typeColors.get("default") ??
      "#9E9E9E";
    const width =
      this.edgeStyles.typeWidths.get(rel.type) ??
      this.edgeStyles.typeWidths.get("default") ??
      1;
    const _dashed =
      this.edgeStyles.typeDashes.get(rel.type) ??
      this.edgeStyles.typeDashes.get("default") ??
      false;

    // Map Neo4j relationship type to EdgeType
    const edgeType = this.mapRelTypeToEdgeType(rel.type);

    const edge: GraphEdge = {
      id,
      sourceId: String(rel.startNodeId),
      targetId: String(rel.endNodeId),
      type: edgeType,
      weight: 1,
      length: 100,
      direction: "forward",
      color,
      width,
      metadata: {
        label: rel.type,
        description: undefined,
        properties: {
          neo4jId: rel.identity,
          ...rel.properties,
        },
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

    this.seenEdges.set(id, edge);
    return edge;
  }

  /**
   * Map Neo4j relationship type to EdgeType union
   */
  private mapRelTypeToEdgeType(relType: string): import("../types").EdgeType {
    const typeLower = relType.toLowerCase();
    const typeMap: Record<string, import("../types").EdgeType> = {
      // Connection types
      connects: "connection",
      connected_to: "connection",
      link: "connection",
      // Dependency types
      depends_on: "dependency",
      requires: "dependency",
      uses: "dependency",
      // Dataflow types
      sends_to: "dataflow",
      receives_from: "dataflow",
      data_flow: "dataflow",
      // Control types
      controls: "control",
      manages: "control",
      owns: "control",
      // Hierarchy types
      parent_of: "hierarchy",
      child_of: "hierarchy",
      contains: "hierarchy",
      // Reference types
      references: "reference",
      refers_to: "reference",
      // Association types
      associated_with: "association",
      related_to: "association",
    };
    return typeMap[typeLower] ?? "custom";
  }

  /**
   * Process a Neo4j path
   */
  private processNeo4jPath(path: Neo4jPath): void {
    this.processNeo4jNode(path.start);
    this.processNeo4jNode(path.end);

    for (const segment of path.segments) {
      this.processNeo4jNode(segment.start);
      this.processNeo4jNode(segment.end);
      this.processNeo4jRelationship(segment.relationship);
    }
  }

  /**
   * Get unique labels from processed nodes
   */
  private getUniqueLabels(): string[] {
    const labels = new Set<string>();

    for (const node of this.seenNodes.values()) {
      // Get labels from metadata.properties.neo4jLabels
      const nodeLabels =
        (node.metadata?.properties?.neo4jLabels as string[]) ?? [];
      for (const label of nodeLabels) {
        labels.add(label);
      }
      // Also add the node type as a label
      labels.add(node.type);
    }

    return Array.from(labels);
  }

  /**
   * Get unique relationship types from processed edges
   */
  private getUniqueRelationshipTypes(): string[] {
    const types = new Set<string>();

    for (const edge of this.seenEdges.values()) {
      // Get label from metadata or use edge type
      const relType = edge.metadata?.label ?? edge.type;
      if (relType) {
        types.add(relType);
      }
    }

    return Array.from(types);
  }

  /**
   * Clear cached nodes and edges
   */
  clear(): void {
    this.seenNodes.clear();
    this.seenEdges.clear();
  }

  /**
   * Add custom node style mapping
   */
  setNodeStyle(
    label: string,
    style: { color?: string; icon?: string; size?: number }
  ): void {
    if (style.color) this.nodeStyles.labelColors.set(label, style.color);
    if (style.icon) this.nodeStyles.labelIcons.set(label, style.icon);
    if (style.size) this.nodeStyles.labelSizes.set(label, style.size);
  }

  /**
   * Add custom edge style mapping
   */
  setEdgeStyle(
    type: string,
    style: { color?: string; width?: number; dashed?: boolean }
  ): void {
    if (style.color) this.edgeStyles.typeColors.set(type, style.color);
    if (style.width !== undefined)
      this.edgeStyles.typeWidths.set(type, style.width);
    if (style.dashed !== undefined)
      this.edgeStyles.typeDashes.set(type, style.dashed);
  }
}

// ============================================================================
// Cypher Query Builder
// ============================================================================

/**
 * Helper class for building common Cypher queries
 */
export class CypherQueryBuilder {
  /**
   * Build a query to fetch all nodes with optional label filter
   */
  static allNodes(label?: string, limit = 100): string {
    const labelClause = label ? `:${label}` : "";
    return `MATCH (n${labelClause}) RETURN n LIMIT ${limit}`;
  }

  /**
   * Build a query to fetch all relationships with optional type filter
   */
  static allRelationships(type?: string, limit = 100): string {
    const typeClause = type ? `:${type}` : "";
    return `MATCH (n)-[r${typeClause}]->(m) RETURN n, r, m LIMIT ${limit}`;
  }

  /**
   * Build a query to fetch a subgraph starting from a node
   */
  static subgraph(
    startNodeId: number | string,
    depth = 2,
    limit = 100
  ): string {
    return `
      MATCH (start) WHERE id(start) = ${startNodeId}
      CALL apoc.path.subgraphAll(start, {maxLevel: ${depth}})
      YIELD nodes, relationships
      RETURN nodes, relationships
      LIMIT ${limit}
    `;
  }

  /**
   * Build a shortest path query
   */
  static shortestPath(fromId: number | string, toId: number | string): string {
    return `
      MATCH (from) WHERE id(from) = ${fromId}
      MATCH (to) WHERE id(to) = ${toId}
      MATCH p = shortestPath((from)-[*]-(to))
      RETURN p
    `;
  }

  /**
   * Build a neighborhood query
   */
  static neighborhood(nodeId: number | string, hops = 1): string {
    return `
      MATCH (n) WHERE id(n) = ${nodeId}
      CALL apoc.neighbors.athop(n, '', ${hops})
      YIELD node
      RETURN n, node
    `;
  }

  /**
   * Build a search query by property value
   */
  static searchByProperty(
    property: string,
    value: string,
    label?: string
  ): string {
    const labelClause = label ? `:${label}` : "";
    return `
      MATCH (n${labelClause})
      WHERE n.${property} CONTAINS '${value}'
      RETURN n
      LIMIT 50
    `;
  }

  /**
   * Build a full-text search query (requires index)
   */
  static fullTextSearch(
    indexName: string,
    searchTerm: string,
    limit = 50
  ): string {
    return `
      CALL db.index.fulltext.queryNodes('${indexName}', '${searchTerm}')
      YIELD node, score
      RETURN node, score
      ORDER BY score DESC
      LIMIT ${limit}
    `;
  }
}
