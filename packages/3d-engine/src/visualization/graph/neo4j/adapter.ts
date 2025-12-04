/**
 * @file Neo4j Adapter
 * @description Connection and query adapter for Neo4j Aura
 * @module @neurectomy/3d-engine/visualization/graph/neo4j
 * @agents @VERTEX @APEX
 */

import type { GraphNode, GraphEdge, NodeType, EdgeType } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface Neo4jConfig {
  /** Neo4j connection URI */
  uri: string;
  /** Database user */
  user: string;
  /** Database password */
  password: string;
  /** Database name (default: neo4j) */
  database?: string;
  /** Connection pool size */
  maxConnectionPoolSize?: number;
  /** Connection timeout in ms */
  connectionTimeout?: number;
  /** Encryption enabled */
  encrypted?: boolean;
}

export interface Neo4jQueryResult<T = Record<string, unknown>> {
  /** Query records */
  records: T[];
  /** Query summary */
  summary: {
    /** Query execution time in ms */
    queryTime: number;
    /** Number of nodes created */
    nodesCreated: number;
    /** Number of relationships created */
    relationshipsCreated: number;
    /** Database hit count */
    dbHits: number;
  };
  /** Any errors */
  error?: Error;
}

export interface Neo4jNodeRecord {
  identity: string | number;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface Neo4jRelationshipRecord {
  identity: string | number;
  type: string;
  startNodeId: string | number;
  endNodeId: string | number;
  properties: Record<string, unknown>;
}

export interface Neo4jGraphResult {
  nodes: Neo4jNodeRecord[];
  relationships: Neo4jRelationshipRecord[];
}

// ============================================================================
// Neo4j Driver Interface (for mock/real implementation)
// ============================================================================

interface Neo4jSession {
  run<T = Record<string, unknown>>(
    query: string,
    parameters?: Record<string, unknown>
  ): Promise<{
    records: Array<{ get: (key: string) => T; toObject: () => T }>;
    summary: {
      resultConsumedAfter: { toNumber: () => number };
      counters: {
        nodesCreated: () => number;
        relationshipsCreated: () => number;
      };
    };
  }>;
  close(): Promise<void>;
}

interface Neo4jDriver {
  session(config?: { database?: string }): Neo4jSession;
  close(): Promise<void>;
  verifyConnectivity(): Promise<void>;
}

// ============================================================================
// Neo4j Graph Adapter
// ============================================================================

/**
 * Adapter for connecting to Neo4j and transforming graph data
 */
export class Neo4jGraphAdapter {
  private config: Neo4jConfig;
  private driver: Neo4jDriver | null = null;
  private isConnected: boolean = false;
  private connectionPromise: Promise<void> | null = null;

  // Node type mapping from Neo4j labels
  private labelToNodeType: Map<string, NodeType> = new Map([
    ["Agent", "agent"],
    ["LLM", "llm"],
    ["Tool", "tool"],
    ["Memory", "memory"],
    ["Database", "database"],
    ["API", "api"],
    ["User", "user"],
    ["System", "system"],
    ["Process", "process"],
    ["Service", "service"],
  ]);

  // Relationship type mapping
  private relTypeToEdgeType: Map<string, EdgeType> = new Map([
    ["CONNECTS_TO", "connection"],
    ["DEPENDS_ON", "dependency"],
    ["EXTENDS", "hierarchy"],
    ["OWNS", "reference"],
    ["USES", "association"],
    ["SENDS_TO", "dataflow"],
    ["RECEIVES_FROM", "dataflow"],
    ["TRIGGERS", "control"],
    ["MONITORS", "reference"],
    ["SYNCS_WITH", "connection"],
  ]);

  constructor(config: Neo4jConfig) {
    this.config = {
      database: "neo4j",
      maxConnectionPoolSize: 50,
      connectionTimeout: 30000,
      encrypted: true,
      ...config,
    };
  }

  /**
   * Connect to Neo4j database
   */
  async connect(): Promise<void> {
    if (this.isConnected) return;
    if (this.connectionPromise) return this.connectionPromise;

    this.connectionPromise = this.initializeConnection();
    await this.connectionPromise;
  }

  private async initializeConnection(): Promise<void> {
    try {
      // Dynamic import for neo4j-driver (browser/node compatible)
      // In a real implementation, you'd use the actual neo4j-driver
      // For now, we'll create a mock driver for the structure
      this.driver = await this.createDriver();
      await this.driver.verifyConnectivity();
      this.isConnected = true;
      console.log("[Neo4jGraphAdapter] Connected to Neo4j");
    } catch (error) {
      this.isConnected = false;
      console.error("[Neo4jGraphAdapter] Connection failed:", error);
      throw error;
    } finally {
      this.connectionPromise = null;
    }
  }

  /**
   * Create Neo4j driver (mock implementation for type safety)
   */
  private async createDriver(): Promise<Neo4jDriver> {
    // In production, you would use:
    // const neo4j = await import('neo4j-driver');
    // return neo4j.driver(this.config.uri, neo4j.auth.basic(this.config.user, this.config.password));

    // Mock driver for development/browser environments
    return {
      session: (config?: { database?: string }) => ({
        run: async <T>(query: string, params?: Record<string, unknown>) => {
          console.log("[Neo4j Mock] Running query:", query, params);
          return {
            records: [] as Array<{
              get: (key: string) => T;
              toObject: () => T;
            }>,
            summary: {
              resultConsumedAfter: { toNumber: () => 0 },
              counters: {
                nodesCreated: () => 0,
                relationshipsCreated: () => 0,
              },
            },
          };
        },
        close: async () => {},
      }),
      close: async () => {},
      verifyConnectivity: async () => {},
    };
  }

  /**
   * Disconnect from Neo4j
   */
  async disconnect(): Promise<void> {
    if (this.driver) {
      await this.driver.close();
      this.driver = null;
      this.isConnected = false;
      console.log("[Neo4jGraphAdapter] Disconnected");
    }
  }

  /**
   * Execute a Cypher query
   */
  async query<T = Record<string, unknown>>(
    cypher: string,
    parameters?: Record<string, unknown>
  ): Promise<Neo4jQueryResult<T>> {
    if (!this.driver) {
      throw new Error("Not connected to Neo4j. Call connect() first.");
    }

    const session = this.driver.session({ database: this.config.database });
    const startTime = performance.now();

    try {
      const result = await session.run<T>(cypher, parameters);
      const queryTime = performance.now() - startTime;

      return {
        records: result.records.map((r) => r.toObject()),
        summary: {
          queryTime,
          nodesCreated: result.summary.counters.nodesCreated(),
          relationshipsCreated: result.summary.counters.relationshipsCreated(),
          dbHits: 0, // Would come from profiling
        },
      };
    } catch (error) {
      return {
        records: [],
        summary: {
          queryTime: performance.now() - startTime,
          nodesCreated: 0,
          relationshipsCreated: 0,
          dbHits: 0,
        },
        error: error instanceof Error ? error : new Error(String(error)),
      };
    } finally {
      await session.close();
    }
  }

  /**
   * Fetch entire graph (with optional limit)
   */
  async fetchGraph(limit: number = 1000): Promise<Neo4jGraphResult> {
    const result = await this.query<{
      n: Neo4jNodeRecord;
      r: Neo4jRelationshipRecord;
      m: Neo4jNodeRecord;
    }>(
      `
      MATCH (n)-[r]->(m)
      RETURN n, r, m
      LIMIT $limit
    `,
      { limit }
    );

    if (result.error) {
      throw result.error;
    }

    const nodesMap = new Map<string, Neo4jNodeRecord>();
    const relationships: Neo4jRelationshipRecord[] = [];

    for (const record of result.records) {
      // Add source node
      const nId = String(record.n.identity);
      if (!nodesMap.has(nId)) {
        nodesMap.set(nId, record.n);
      }
      // Add target node
      const mId = String(record.m.identity);
      if (!nodesMap.has(mId)) {
        nodesMap.set(mId, record.m);
      }
      // Add relationship
      relationships.push(record.r);
    }

    return {
      nodes: Array.from(nodesMap.values()),
      relationships,
    };
  }

  /**
   * Fetch nodes by label
   */
  async fetchNodesByLabel(
    label: string,
    limit: number = 500
  ): Promise<Neo4jNodeRecord[]> {
    const result = await this.query<{ n: Neo4jNodeRecord }>(
      `
      MATCH (n:${label})
      RETURN n
      LIMIT $limit
    `,
      { limit }
    );

    if (result.error) {
      throw result.error;
    }

    return result.records.map((r) => r.n);
  }

  /**
   * Fetch subgraph around a specific node
   */
  async fetchSubgraph(
    nodeId: string | number,
    depth: number = 2
  ): Promise<Neo4jGraphResult> {
    const result = await this.query<{
      path: {
        nodes: Neo4jNodeRecord[];
        relationships: Neo4jRelationshipRecord[];
      };
    }>(
      `
      MATCH path = (start)-[*1..${depth}]-(end)
      WHERE id(start) = $nodeId
      RETURN path
    `,
      { nodeId }
    );

    if (result.error) {
      throw result.error;
    }

    const nodesMap = new Map<string, Neo4jNodeRecord>();
    const relationshipsMap = new Map<string, Neo4jRelationshipRecord>();

    for (const record of result.records) {
      for (const node of record.path.nodes) {
        nodesMap.set(String(node.identity), node);
      }
      for (const rel of record.path.relationships) {
        relationshipsMap.set(String(rel.identity), rel);
      }
    }

    return {
      nodes: Array.from(nodesMap.values()),
      relationships: Array.from(relationshipsMap.values()),
    };
  }

  /**
   * Search nodes by property
   */
  async searchNodes(
    property: string,
    value: string,
    fuzzy: boolean = false
  ): Promise<Neo4jNodeRecord[]> {
    const query = fuzzy
      ? `MATCH (n) WHERE n.${property} =~ $pattern RETURN n`
      : `MATCH (n) WHERE n.${property} = $value RETURN n`;

    const params = fuzzy ? { pattern: `(?i).*${value}.*` } : { value };

    const result = await this.query<{ n: Neo4jNodeRecord }>(query, params);

    if (result.error) {
      throw result.error;
    }

    return result.records.map((r) => r.n);
  }

  /**
   * Convert Neo4j node to GraphNode
   */
  neo4jNodeToGraphNode(
    neo4jNode: Neo4jNodeRecord,
    index: number = 0
  ): GraphNode {
    const id = String(neo4jNode.identity);
    const label =
      (neo4jNode.properties.name as string) ||
      (neo4jNode.properties.label as string) ||
      `Node ${id}`;

    // Determine node type from labels
    let nodeType: NodeType = "custom";
    for (const neoLabel of neo4jNode.labels) {
      const mappedType = this.labelToNodeType.get(neoLabel);
      if (mappedType) {
        nodeType = mappedType;
        break;
      }
    }

    // Determine color based on type
    const typeColors: Record<NodeType, string> = {
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

    // Spiral layout for initial positioning
    const angle = index * 2.4; // Golden angle
    const radius = Math.sqrt(index) * 2;

    return {
      id,
      label,
      type: nodeType,
      position: {
        x: Math.cos(angle) * radius,
        y: (index % 5) * 0.5 - 1,
        z: Math.sin(angle) * radius,
      },
      velocity: { x: 0, y: 0, z: 0 },
      mass: 1,
      radius: 0.3,
      color: typeColors[nodeType],
      pinned: false,
      metadata: {
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
   * Convert Neo4j relationship to GraphEdge
   */
  neo4jRelationshipToGraphEdge(neo4jRel: Neo4jRelationshipRecord): GraphEdge {
    const id = String(neo4jRel.identity);
    const sourceId = String(neo4jRel.startNodeId);
    const targetId = String(neo4jRel.endNodeId);

    // Determine edge type from relationship type
    let edgeType: EdgeType = "custom";
    const mappedType = this.relTypeToEdgeType.get(neo4jRel.type);
    if (mappedType) {
      edgeType = mappedType;
    }

    // Determine color based on type
    const typeColors: Record<EdgeType, string> = {
      connection: "#64748b",
      dependency: "#ef4444",
      hierarchy: "#8b5cf6",
      reference: "#3b82f6",
      association: "#22c55e",
      dataflow: "#06b6d4",
      control: "#f97316",
      custom: "#6b7280",
    };

    return {
      id,
      sourceId,
      targetId,
      type: edgeType,
      direction: "forward",
      weight: 1,
      length: 100,
      color: typeColors[edgeType],
      width: 1,
      metadata: {
        label: neo4jRel.type,
        properties: neo4jRel.properties,
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
   * Transform Neo4j graph result to visualization format
   */
  transformToVisualization(neo4jResult: Neo4jGraphResult): {
    nodes: GraphNode[];
    edges: GraphEdge[];
  } {
    const nodes = neo4jResult.nodes.map((n, i) =>
      this.neo4jNodeToGraphNode(n, i)
    );

    const edges = neo4jResult.relationships.map((r) =>
      this.neo4jRelationshipToGraphEdge(r)
    );

    return { nodes, edges };
  }

  /**
   * Register custom label to node type mapping
   */
  registerLabelMapping(label: string, nodeType: NodeType): void {
    this.labelToNodeType.set(label, nodeType);
  }

  /**
   * Register custom relationship type to edge type mapping
   */
  registerRelationshipMapping(relType: string, edgeType: EdgeType): void {
    this.relTypeToEdgeType.set(relType, edgeType);
  }

  /**
   * Check if connected
   */
  get connected(): boolean {
    return this.isConnected;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Neo4j graph adapter
 */
export function createNeo4jAdapter(config: Neo4jConfig): Neo4jGraphAdapter {
  return new Neo4jGraphAdapter(config);
}
