/**
 * @file Cypher Query Visualizer
 * @description Visualize Cypher queries and their execution patterns
 * @module @neurectomy/3d-engine/visualization/graph/neo4j
 * @agents @VERTEX @PRISM @CANVAS
 */

import type { GraphNode, GraphEdge } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface CypherQuery {
  /** Query string */
  query: string;
  /** Query parameters */
  parameters?: Record<string, unknown>;
  /** Query name/description */
  name?: string;
}

export interface QueryStep {
  /** Step type */
  type:
    | "match"
    | "where"
    | "return"
    | "create"
    | "merge"
    | "delete"
    | "set"
    | "with"
    | "optional_match"
    | "call"
    | "union";
  /** Step pattern (parsed) */
  pattern?: string;
  /** Variables bound in this step */
  variables: string[];
  /** Conditions (for WHERE) */
  conditions?: string[];
  /** Return expressions (for RETURN) */
  returns?: string[];
}

export interface ParsedQuery {
  /** Original query */
  original: string;
  /** Parsed steps */
  steps: QueryStep[];
  /** All variables */
  variables: string[];
  /** Node patterns */
  nodePatterns: NodePattern[];
  /** Relationship patterns */
  relationshipPatterns: RelationshipPattern[];
  /** Errors during parsing */
  errors: string[];
}

export interface NodePattern {
  /** Variable name */
  variable?: string;
  /** Node labels */
  labels: string[];
  /** Property filters */
  properties?: Record<string, unknown>;
}

export interface RelationshipPattern {
  /** Variable name */
  variable?: string;
  /** Relationship type */
  type?: string;
  /** Direction */
  direction: "outgoing" | "incoming" | "undirected";
  /** Source variable */
  sourceVar?: string;
  /** Target variable */
  targetVar?: string;
  /** Variable length */
  variableLength?: { min?: number; max?: number };
}

export interface QueryVisualization {
  /** Query plan nodes */
  nodes: GraphNode[];
  /** Query plan edges */
  edges: GraphEdge[];
  /** Parsed query info */
  parsed: ParsedQuery;
  /** Execution stats (if available) */
  stats?: {
    dbHits: number;
    rows: number;
    time: number;
  };
}

// ============================================================================
// Cypher Parser (Simplified)
// ============================================================================

const CYPHER_KEYWORDS = [
  "MATCH",
  "OPTIONAL MATCH",
  "WHERE",
  "RETURN",
  "CREATE",
  "MERGE",
  "DELETE",
  "DETACH DELETE",
  "SET",
  "REMOVE",
  "WITH",
  "UNWIND",
  "ORDER BY",
  "SKIP",
  "LIMIT",
  "UNION",
  "CALL",
  "YIELD",
];

/**
 * Parse a Cypher query into structured components
 */
function parseCypherQuery(query: string): ParsedQuery {
  const errors: string[] = [];
  const steps: QueryStep[] = [];
  const variables: string[] = [];
  const nodePatterns: NodePattern[] = [];
  const relationshipPatterns: RelationshipPattern[] = [];

  // Normalize query
  const normalized = query.trim().replace(/\s+/g, " ");

  try {
    // Split by keywords
    const keywordRegex = new RegExp(
      `(${CYPHER_KEYWORDS.map((k) => k.replace(" ", "\\s+")).join("|")})`,
      "gi"
    );

    const parts = normalized.split(keywordRegex).filter((p) => p.trim());

    let currentKeyword = "";
    for (const part of parts) {
      const upperPart = part.toUpperCase().trim();

      if (
        CYPHER_KEYWORDS.some(
          (k) => k === upperPart || k === upperPart.replace(/\s+/g, " ")
        )
      ) {
        currentKeyword = upperPart;
      } else if (currentKeyword) {
        const step = parseStep(currentKeyword, part.trim());
        if (step) {
          steps.push(step);
          variables.push(
            ...step.variables.filter((v) => !variables.includes(v))
          );
        }

        // Extract patterns
        if (
          currentKeyword === "MATCH" ||
          currentKeyword === "OPTIONAL MATCH" ||
          currentKeyword === "CREATE" ||
          currentKeyword === "MERGE"
        ) {
          const patterns = extractPatterns(part.trim());
          nodePatterns.push(...patterns.nodes);
          relationshipPatterns.push(...patterns.relationships);
        }
      }
    }
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "Parse error");
  }

  return {
    original: query,
    steps,
    variables,
    nodePatterns,
    relationshipPatterns,
    errors,
  };
}

/**
 * Parse a single query step
 */
function parseStep(keyword: string, content: string): QueryStep | null {
  const variables = extractVariables(content);

  switch (keyword) {
    case "MATCH":
    case "OPTIONAL MATCH":
      return {
        type: keyword === "OPTIONAL MATCH" ? "optional_match" : "match",
        pattern: content,
        variables,
      };
    case "WHERE":
      return {
        type: "where",
        variables,
        conditions: content.split(/\s+AND\s+/i),
      };
    case "RETURN":
      return {
        type: "return",
        variables,
        returns: content.split(",").map((s) => s.trim()),
      };
    case "CREATE":
      return { type: "create", pattern: content, variables };
    case "MERGE":
      return { type: "merge", pattern: content, variables };
    case "DELETE":
    case "DETACH DELETE":
      return { type: "delete", variables };
    case "SET":
      return { type: "set", variables };
    case "WITH":
      return { type: "with", variables };
    case "CALL":
      return { type: "call", variables };
    case "UNION":
      return { type: "union", variables: [] };
    default:
      return null;
  }
}

/**
 * Extract variable names from content
 */
function extractVariables(content: string): string[] {
  const variables: string[] = [];

  // Match variable patterns like (n), [r], n.property
  const varRegex = /[(\[]?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::[^)\]]*)?[)\]]?/g;
  let match;

  while ((match = varRegex.exec(content)) !== null) {
    const varName = match[1];
    if (
      varName &&
      !CYPHER_KEYWORDS.includes(varName.toUpperCase()) &&
      !variables.includes(varName)
    ) {
      variables.push(varName);
    }
  }

  return variables;
}

/**
 * Extract node and relationship patterns
 */
function extractPatterns(content: string): {
  nodes: NodePattern[];
  relationships: RelationshipPattern[];
} {
  const nodes: NodePattern[] = [];
  const relationships: RelationshipPattern[] = [];

  // Extract node patterns: (variable:Label {props})
  const nodeRegex =
    /\(\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*(?::([^)\s{]+))?\s*(?:\{([^}]*)\})?\s*\)/g;
  let match;

  while ((match = nodeRegex.exec(content)) !== null) {
    nodes.push({
      variable: match[1] || undefined,
      labels: match[2] ? match[2].split(":").filter(Boolean) : [],
      properties: match[3] ? parseProperties(match[3]) : undefined,
    });
  }

  // Extract relationship patterns: -[r:TYPE]->
  const relRegex =
    /(<)?-\[?\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*(?::([^*\]\s]+))?\s*(?:\*(\d+)?(?:\.\.(\d+))?)?\s*\]?-(>)?/g;

  while ((match = relRegex.exec(content)) !== null) {
    const hasLeft = !!match[1];
    const hasRight = !!match[6];

    let direction: "outgoing" | "incoming" | "undirected" = "undirected";
    if (hasRight && !hasLeft) direction = "outgoing";
    else if (hasLeft && !hasRight) direction = "incoming";

    relationships.push({
      variable: match[2] || undefined,
      type: match[3] || undefined,
      direction,
      variableLength:
        match[4] || match[5]
          ? {
              min: match[4] ? parseInt(match[4]) : undefined,
              max: match[5] ? parseInt(match[5]) : undefined,
            }
          : undefined,
    });
  }

  return { nodes, relationships };
}

/**
 * Parse property string into object
 */
function parseProperties(propString: string): Record<string, unknown> {
  const props: Record<string, unknown> = {};

  // Simple key: value parsing
  const propRegex = /([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(['"]?)([^'",}]+)\2/g;
  let match;

  while ((match = propRegex.exec(propString)) !== null) {
    const key = match[1];
    if (!key) continue; // Skip if key is undefined

    let value: unknown = match[3];

    // Try to parse as number
    const num = parseFloat(value as string);
    if (!isNaN(num)) value = num;
    // Check for boolean
    else if (value === "true") value = true;
    else if (value === "false") value = false;

    props[key] = value;
  }

  return props;
}

// ============================================================================
// Cypher Query Visualizer
// ============================================================================

/**
 * Visualize Cypher queries as 3D flow diagrams
 */
export class CypherQueryVisualizer {
  private queryHistory: ParsedQuery[] = [];

  /**
   * Parse and visualize a Cypher query
   */
  visualize(query: CypherQuery): QueryVisualization {
    const parsed = parseCypherQuery(query.query);
    this.queryHistory.push(parsed);

    const { nodes, edges } = this.createQueryVisualization(parsed);

    return {
      nodes,
      edges,
      parsed,
    };
  }

  /**
   * Create visualization graph from parsed query
   */
  private createQueryVisualization(parsed: ParsedQuery): {
    nodes: GraphNode[];
    edges: GraphEdge[];
  } {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];

    // Create nodes for each step
    parsed.steps.forEach((step, index) => {
      const node = this.createStepNode(step, index);
      nodes.push(node);

      // Connect to previous step
      if (index > 0) {
        edges.push({
          id: `edge-step-${index - 1}-${index}`,
          sourceId: `step-${index - 1}`,
          targetId: `step-${index}`,
          type: "dataflow",
          direction: "forward",
          weight: 1,
          length: 100,
          color: "#4a90d9",
          width: 2,
          metadata: { label: "flow", properties: {} },
          state: {
            selected: false,
            hovered: false,
            highlighted: false,
            dimmed: false,
            visible: true,
            animationProgress: 0,
          },
        });
      }
    });

    // Create nodes for pattern elements
    let patternOffset = parsed.steps.length;
    const varNodeMap = new Map<string, string>();

    parsed.nodePatterns.forEach((pattern, index) => {
      const nodeId = `pattern-node-${index}`;
      const varName = pattern.variable || `_node${index}`;
      varNodeMap.set(varName, nodeId);

      nodes.push({
        id: nodeId,
        label: pattern.variable
          ? `${pattern.variable}${pattern.labels.length ? ":" + pattern.labels.join(":") : ""}`
          : pattern.labels.join(":") || "()",
        type: "database",
        position: {
          x: (index - parsed.nodePatterns.length / 2) * 2,
          y: -2,
          z: 2,
        },
        velocity: { x: 0, y: 0, z: 0 },
        mass: 1,
        radius: 0.4,
        color: "#e74c3c",
        pinned: false,
        metadata: {
          tags: pattern.labels,
          properties: pattern.properties || {},
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
      });
    });

    // Create edges for relationships
    parsed.relationshipPatterns.forEach((rel, index) => {
      if (rel.sourceVar && rel.targetVar) {
        const sourceId = varNodeMap.get(rel.sourceVar);
        const targetId = varNodeMap.get(rel.targetVar);

        if (sourceId && targetId) {
          edges.push({
            id: `pattern-rel-${index}`,
            sourceId,
            targetId,
            type: "connection",
            direction: rel.direction === "incoming" ? "backward" : "forward",
            weight: 1,
            length: 100,
            color: "#3498db",
            width: 2,
            metadata: {
              label: rel.type || "REL",
              properties: {},
            },
            state: {
              selected: false,
              hovered: false,
              highlighted: false,
              dimmed: false,
              visible: true,
              animationProgress: 0,
            },
          });
        }
      }
    });

    return { nodes, edges };
  }

  /**
   * Create a node for a query step
   */
  private createStepNode(step: QueryStep, index: number): GraphNode {
    const typeColors: Record<string, string> = {
      match: "#27ae60",
      optional_match: "#2ecc71",
      where: "#f39c12",
      return: "#9b59b6",
      create: "#3498db",
      merge: "#1abc9c",
      delete: "#e74c3c",
      set: "#e67e22",
      with: "#95a5a6",
      call: "#34495e",
      union: "#7f8c8d",
    };

    const label = step.type.toUpperCase().replace("_", " ");
    const sublabel =
      step.variables.length > 0 ? ` (${step.variables.join(", ")})` : "";

    return {
      id: `step-${index}`,
      label: label + sublabel,
      type: "process",
      position: {
        x: 0,
        y: -index * 1.5,
        z: 0,
      },
      velocity: { x: 0, y: 0, z: 0 },
      mass: 1,
      radius: 0.5,
      color: typeColors[step.type] || "#7f8c8d",
      pinned: true,
      metadata: {
        description:
          step.pattern ||
          step.conditions?.join(" AND ") ||
          step.returns?.join(", "),
        category: "query-step",
        tags: [step.type, ...step.variables],
        properties: {
          pattern: step.pattern,
          conditions: step.conditions,
          returns: step.returns,
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
  }

  /**
   * Visualize query execution plan (if profile data available)
   */
  visualizeExecutionPlan(
    query: CypherQuery,
    planData: {
      operators: Array<{
        id: string;
        operatorType: string;
        dbHits: number;
        rows: number;
        children?: string[];
      }>;
    }
  ): QueryVisualization {
    const parsed = parseCypherQuery(query.query);
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];

    // Create nodes for each operator
    planData.operators.forEach((op, index) => {
      const intensity = Math.min(1, op.dbHits / 1000); // Normalize to 0-1

      nodes.push({
        id: op.id,
        label: `${op.operatorType}\n${op.rows} rows, ${op.dbHits} dbHits`,
        type: "process",
        position: {
          x: (index % 5) * 2 - 4,
          y: -Math.floor(index / 5) * 2,
          z: 0,
        },
        velocity: { x: 0, y: 0, z: 0 },
        mass: 1 + intensity,
        radius: 0.3 + intensity * 0.3,
        color: this.getHeatmapColor(intensity),
        pinned: false,
        metadata: {
          category: "execution-plan",
          tags: [op.operatorType],
          properties: { dbHits: op.dbHits, rows: op.rows, intensity },
        },
        state: {
          selected: false,
          hovered: false,
          dragging: false,
          highlighted: false,
          dimmed: false,
          visible: true,
          activity: intensity,
        },
      });

      // Create edges to children
      if (op.children) {
        op.children.forEach((childId) => {
          edges.push({
            id: `edge-${op.id}-${childId}`,
            sourceId: op.id,
            targetId: childId,
            type: "dataflow",
            direction: "forward",
            weight: 1,
            length: 100,
            color: "#64748b",
            width: 1,
            metadata: { label: "", properties: {} },
            state: {
              selected: false,
              hovered: false,
              highlighted: false,
              dimmed: false,
              visible: true,
              animationProgress: 0,
            },
          });
        });
      }
    });

    const totalDbHits = planData.operators.reduce(
      (sum, op) => sum + op.dbHits,
      0
    );
    const totalRows = Math.max(...planData.operators.map((op) => op.rows));

    return {
      nodes,
      edges,
      parsed,
      stats: {
        dbHits: totalDbHits,
        rows: totalRows,
        time: 0, // Would come from actual execution
      },
    };
  }

  /**
   * Get heatmap color based on intensity (0-1)
   */
  private getHeatmapColor(intensity: number): string {
    // Green (low) -> Yellow (medium) -> Red (high)
    const r = Math.round(255 * Math.min(1, intensity * 2));
    const g = Math.round(255 * Math.min(1, (1 - intensity) * 2));
    const b = 0;

    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }

  /**
   * Get query history
   */
  getHistory(): ParsedQuery[] {
    return [...this.queryHistory];
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.queryHistory = [];
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createCypherVisualizer(): CypherQueryVisualizer {
  return new CypherQueryVisualizer();
}
