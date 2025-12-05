/**
 * NEURECTOMY 3D Container Topology Visualizer
 *
 * @module @neurectomy/container-command/topology
 * @agent @PIXEL @FLUX
 *
 * Provides 3D visualization of container relationships and resource topology:
 * - Real-time container state visualization
 * - Service mesh traffic flow rendering
 * - Resource dependency graphs
 * - Kubernetes namespace/pod hierarchy
 * - Network policy visualization
 * - Health status heatmaps
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                    3D Topology Engine                                │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
 * │  │  Graph Builder  │  │  Layout Engine  │  │  Render Pipeline    │  │
 * │  │  - Nodes        │  │  - Force-Direct │  │  - Meshes           │  │
 * │  │  - Edges        │  │  - Hierarchical │  │  - Materials        │  │
 * │  │  - Clusters     │  │  - Radial       │  │  - Animations       │  │
 * │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
 * │  │ Data Collector  │  │ State Manager   │  │  Event Handler      │  │
 * │  │  - K8s API      │  │  - Diffs        │  │  - Selection        │  │
 * │  │  - Docker       │  │  - Animations   │  │  - Navigation       │  │
 * │  │  - Mesh         │  │  - Transitions  │  │  - Interaction      │  │
 * │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
 * └─────────────────────────────────────────────────────────────────────┘
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";
import pino from "pino";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * 3D Vector type
 */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/**
 * Node types in the topology
 */
export type TopologyNodeType =
  | "namespace"
  | "deployment"
  | "replicaset"
  | "pod"
  | "container"
  | "service"
  | "ingress"
  | "configmap"
  | "secret"
  | "pvc"
  | "node"
  | "microvm"
  | "wasm_instance";

/**
 * Edge types between nodes
 */
export type TopologyEdgeType =
  | "owns"
  | "manages"
  | "selects"
  | "mounts"
  | "exposes"
  | "traffic"
  | "network_policy"
  | "depends_on";

/**
 * Node health status
 */
export type HealthStatus = "healthy" | "warning" | "critical" | "unknown";

/**
 * Layout algorithm types
 */
export type LayoutAlgorithm =
  | "force_directed"
  | "hierarchical"
  | "radial"
  | "tree"
  | "cluster"
  | "grid";

// ============================================================================
// Zod Validation Schemas
// ============================================================================

export const TopologyNodeSchema = z.object({
  id: z.string(),
  type: z.enum([
    "namespace",
    "deployment",
    "replicaset",
    "pod",
    "container",
    "service",
    "ingress",
    "configmap",
    "secret",
    "pvc",
    "node",
    "microvm",
    "wasm_instance",
  ]),
  name: z.string(),
  namespace: z.string().optional(),
  labels: z.record(z.string()).optional(),
  annotations: z.record(z.string()).optional(),
  status: z
    .enum(["healthy", "warning", "critical", "unknown"])
    .default("unknown"),
  metrics: z
    .object({
      cpu: z.number().min(0).max(100).optional(),
      memory: z.number().min(0).max(100).optional(),
      network: z.number().optional(),
      requests: z.number().optional(),
      errors: z.number().optional(),
    })
    .optional(),
  position: z
    .object({
      x: z.number(),
      y: z.number(),
      z: z.number(),
    })
    .optional(),
  metadata: z.record(z.unknown()).optional(),
});

export const TopologyEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: z.enum([
    "owns",
    "manages",
    "selects",
    "mounts",
    "exposes",
    "traffic",
    "network_policy",
    "depends_on",
  ]),
  weight: z.number().min(0).default(1),
  bidirectional: z.boolean().default(false),
  metadata: z.record(z.unknown()).optional(),
  trafficMetrics: z
    .object({
      requestsPerSecond: z.number().optional(),
      latencyP50: z.number().optional(),
      latencyP99: z.number().optional(),
      successRate: z.number().optional(),
    })
    .optional(),
});

export const TopologyGraphSchema = z.object({
  id: z.string(),
  name: z.string(),
  nodes: z.array(TopologyNodeSchema),
  edges: z.array(TopologyEdgeSchema),
  clusters: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        nodeIds: z.array(z.string()),
        color: z.string().optional(),
      })
    )
    .optional(),
  timestamp: z.date(),
});

export const LayoutConfigSchema = z.object({
  algorithm: z
    .enum([
      "force_directed",
      "hierarchical",
      "radial",
      "tree",
      "cluster",
      "grid",
    ])
    .default("force_directed"),
  dimensions: z.number().int().min(2).max(3).default(3),
  spacing: z.number().positive().default(50),
  iterations: z.number().int().positive().default(100),
  gravity: z.number().default(0.1),
  repulsion: z.number().default(1000),
  attraction: z.number().default(0.01),
  damping: z.number().min(0).max(1).default(0.9),
  centerForce: z.number().default(0.05),
});

export const RenderConfigSchema = z.object({
  nodeSize: z
    .object({
      min: z.number().positive().default(5),
      max: z.number().positive().default(30),
      scale: z.enum(["linear", "log", "sqrt"]).default("sqrt"),
    })
    .optional(),
  edgeWidth: z
    .object({
      min: z.number().positive().default(1),
      max: z.number().positive().default(10),
    })
    .optional(),
  colors: z
    .object({
      healthy: z.string().default("#00ff00"),
      warning: z.string().default("#ffff00"),
      critical: z.string().default("#ff0000"),
      unknown: z.string().default("#888888"),
      selected: z.string().default("#00ffff"),
      highlighted: z.string().default("#ff00ff"),
    })
    .optional(),
  animations: z
    .object({
      enabled: z.boolean().default(true),
      duration: z.number().positive().default(500),
      easing: z
        .enum(["linear", "ease_in", "ease_out", "ease_in_out"])
        .default("ease_out"),
    })
    .optional(),
  labels: z
    .object({
      show: z.boolean().default(true),
      fontSize: z.number().positive().default(12),
      maxLength: z.number().int().positive().default(20),
    })
    .optional(),
});

// ============================================================================
// Type Exports
// ============================================================================

export type TopologyNode = z.infer<typeof TopologyNodeSchema>;
export type TopologyEdge = z.infer<typeof TopologyEdgeSchema>;
export type TopologyGraph = z.infer<typeof TopologyGraphSchema>;
export type LayoutConfig = z.infer<typeof LayoutConfigSchema>;
export type RenderConfig = z.infer<typeof RenderConfigSchema>;

// ============================================================================
// Events
// ============================================================================

export interface TopologyEvents {
  "graph:updated": { graph: TopologyGraph; diff: GraphDiff };
  "node:added": { node: TopologyNode };
  "node:removed": { nodeId: string };
  "node:updated": { node: TopologyNode; changes: Partial<TopologyNode> };
  "edge:added": { edge: TopologyEdge };
  "edge:removed": { edgeId: string };
  "layout:computed": { duration: number };
  "render:frame": { fps: number };
  "selection:changed": { nodeIds: string[]; edgeIds: string[] };
  "hover:node": { nodeId: string | null };
  "hover:edge": { edgeId: string | null };
  "cluster:expanded": { clusterId: string };
  "cluster:collapsed": { clusterId: string };
  error: { operation: string; error: Error };
}

export interface GraphDiff {
  nodesAdded: string[];
  nodesRemoved: string[];
  nodesUpdated: string[];
  edgesAdded: string[];
  edgesRemoved: string[];
  edgesUpdated: string[];
}

// ============================================================================
// Render Output Types
// ============================================================================

export interface NodeMesh {
  nodeId: string;
  geometry: "sphere" | "cube" | "cylinder" | "octahedron" | "icosahedron";
  position: Vector3;
  scale: Vector3;
  color: string;
  opacity: number;
  wireframe: boolean;
  glowIntensity: number;
}

export interface EdgeMesh {
  edgeId: string;
  geometry: "line" | "tube" | "arrow" | "curve";
  points: Vector3[];
  color: string;
  width: number;
  opacity: number;
  animated: boolean;
  particleFlow?: {
    speed: number;
    density: number;
    color: string;
  };
}

export interface LabelMesh {
  id: string;
  text: string;
  position: Vector3;
  fontSize: number;
  color: string;
  backgroundColor?: string;
  visible: boolean;
}

export interface RenderOutput {
  nodes: NodeMesh[];
  edges: EdgeMesh[];
  labels: LabelMesh[];
  clusters: Array<{
    id: string;
    boundingBox: { min: Vector3; max: Vector3 };
    color: string;
    opacity: number;
  }>;
  camera?: {
    position: Vector3;
    target: Vector3;
    fov: number;
  };
}

// ============================================================================
// Layout Engine
// ============================================================================

/**
 * Force-directed layout implementation
 */
class ForceDirectedLayout {
  private config: LayoutConfig;
  private logger: pino.Logger;

  constructor(config: LayoutConfig, logger: pino.Logger) {
    this.config = config;
    this.logger = logger.child({ layout: "force_directed" });
  }

  /**
   * Compute layout positions for all nodes
   */
  compute(nodes: TopologyNode[], edges: TopologyEdge[]): Map<string, Vector3> {
    const positions = new Map<string, Vector3>();
    const velocities = new Map<string, Vector3>();

    // Initialize random positions
    for (const node of nodes) {
      positions.set(node.id, node.position || this.randomPosition());
      velocities.set(node.id, { x: 0, y: 0, z: 0 });
    }

    // Build adjacency map
    const adjacency = this.buildAdjacency(edges);

    // Run iterations
    for (let i = 0; i < this.config.iterations; i++) {
      // Apply forces
      for (const node of nodes) {
        const pos = positions.get(node.id)!;
        const vel = velocities.get(node.id)!;

        // Repulsion from all other nodes
        const repulsion = this.computeRepulsion(node.id, positions, nodes);

        // Attraction to connected nodes
        const attraction = this.computeAttraction(
          node.id,
          positions,
          adjacency.get(node.id) || []
        );

        // Center gravity
        const gravity = this.computeGravity(pos);

        // Update velocity
        vel.x =
          (vel.x + repulsion.x + attraction.x + gravity.x) *
          this.config.damping;
        vel.y =
          (vel.y + repulsion.y + attraction.y + gravity.y) *
          this.config.damping;
        vel.z =
          (vel.z + repulsion.z + attraction.z + gravity.z) *
          this.config.damping;

        // Update position
        pos.x += vel.x;
        pos.y += vel.y;
        pos.z += vel.z;
      }
    }

    this.logger.debug(
      { nodeCount: nodes.length, iterations: this.config.iterations },
      "Layout computed"
    );

    return positions;
  }

  private randomPosition(): Vector3 {
    const range = this.config.spacing * 10;
    return {
      x: (Math.random() - 0.5) * range,
      y: (Math.random() - 0.5) * range,
      z: this.config.dimensions === 3 ? (Math.random() - 0.5) * range : 0,
    };
  }

  private buildAdjacency(edges: TopologyEdge[]): Map<string, string[]> {
    const adj = new Map<string, string[]>();

    for (const edge of edges) {
      if (!adj.has(edge.source)) adj.set(edge.source, []);
      if (!adj.has(edge.target)) adj.set(edge.target, []);

      adj.get(edge.source)!.push(edge.target);
      if (edge.bidirectional) {
        adj.get(edge.target)!.push(edge.source);
      }
    }

    return adj;
  }

  private computeRepulsion(
    nodeId: string,
    positions: Map<string, Vector3>,
    nodes: TopologyNode[]
  ): Vector3 {
    const force: Vector3 = { x: 0, y: 0, z: 0 };
    const pos = positions.get(nodeId)!;

    for (const other of nodes) {
      if (other.id === nodeId) continue;

      const otherPos = positions.get(other.id)!;
      const dx = pos.x - otherPos.x;
      const dy = pos.y - otherPos.y;
      const dz = pos.z - otherPos.z;

      const distSq = dx * dx + dy * dy + dz * dz + 0.01; // Avoid division by zero
      const dist = Math.sqrt(distSq);

      const strength = this.config.repulsion / distSq;

      force.x += (dx / dist) * strength;
      force.y += (dy / dist) * strength;
      force.z += (dz / dist) * strength;
    }

    return force;
  }

  private computeAttraction(
    nodeId: string,
    positions: Map<string, Vector3>,
    neighbors: string[]
  ): Vector3 {
    const force: Vector3 = { x: 0, y: 0, z: 0 };
    const pos = positions.get(nodeId)!;

    for (const neighborId of neighbors) {
      const neighborPos = positions.get(neighborId);
      if (!neighborPos) continue;

      const dx = neighborPos.x - pos.x;
      const dy = neighborPos.y - pos.y;
      const dz = neighborPos.z - pos.z;

      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;

      // Spring force (Hooke's law)
      const strength = (dist - this.config.spacing) * this.config.attraction;

      force.x += (dx / dist) * strength;
      force.y += (dy / dist) * strength;
      force.z += (dz / dist) * strength;
    }

    return force;
  }

  private computeGravity(pos: Vector3): Vector3 {
    return {
      x: -pos.x * this.config.centerForce,
      y: -pos.y * this.config.centerForce,
      z: -pos.z * this.config.centerForce,
    };
  }
}

/**
 * Hierarchical layout implementation
 */
class HierarchicalLayout {
  private config: LayoutConfig;
  private logger: pino.Logger;

  constructor(config: LayoutConfig, logger: pino.Logger) {
    this.config = config;
    this.logger = logger.child({ layout: "hierarchical" });
  }

  compute(nodes: TopologyNode[], edges: TopologyEdge[]): Map<string, Vector3> {
    const positions = new Map<string, Vector3>();

    // Build hierarchy based on node types
    const levels = this.assignLevels(nodes, edges);
    const nodesByLevel = this.groupByLevel(nodes, levels);

    // Position nodes by level
    let currentY = 0;

    for (const [level, levelNodes] of nodesByLevel) {
      const spacing = this.config.spacing;
      const width = (levelNodes.length - 1) * spacing;
      let currentX = -width / 2;

      for (const node of levelNodes) {
        positions.set(node.id, {
          x: currentX,
          y: currentY,
          z: 0,
        });
        currentX += spacing;
      }

      currentY -= spacing * 1.5;
    }

    this.logger.debug(
      { nodeCount: nodes.length, levels: nodesByLevel.size },
      "Hierarchical layout computed"
    );

    return positions;
  }

  private assignLevels(
    nodes: TopologyNode[],
    edges: TopologyEdge[]
  ): Map<string, number> {
    const levels = new Map<string, number>();
    const typeOrder: Record<TopologyNodeType, number> = {
      namespace: 0,
      node: 1,
      deployment: 2,
      replicaset: 3,
      pod: 4,
      container: 5,
      microvm: 4,
      wasm_instance: 5,
      service: 2,
      ingress: 1,
      configmap: 6,
      secret: 6,
      pvc: 6,
    };

    for (const node of nodes) {
      levels.set(node.id, typeOrder[node.type] ?? 5);
    }

    return levels;
  }

  private groupByLevel(
    nodes: TopologyNode[],
    levels: Map<string, number>
  ): Map<number, TopologyNode[]> {
    const groups = new Map<number, TopologyNode[]>();

    for (const node of nodes) {
      const level = levels.get(node.id) ?? 0;
      if (!groups.has(level)) groups.set(level, []);
      groups.get(level)!.push(node);
    }

    return new Map([...groups].sort((a, b) => a[0] - b[0]));
  }
}

/**
 * Radial layout implementation
 */
class RadialLayout {
  private config: LayoutConfig;
  private logger: pino.Logger;

  constructor(config: LayoutConfig, logger: pino.Logger) {
    this.config = config;
    this.logger = logger.child({ layout: "radial" });
  }

  compute(
    nodes: TopologyNode[],
    edges: TopologyEdge[],
    centerId?: string
  ): Map<string, Vector3> {
    const positions = new Map<string, Vector3>();

    // Find center node (or use first node)
    const center = centerId
      ? nodes.find((n) => n.id === centerId)
      : nodes.find((n) => n.type === "namespace") || nodes[0];

    if (!center) return positions;

    // Place center
    positions.set(center.id, { x: 0, y: 0, z: 0 });

    // Build distance map from center
    const distances = this.computeDistances(center.id, nodes, edges);

    // Group nodes by distance
    const rings = new Map<number, TopologyNode[]>();
    for (const node of nodes) {
      if (node.id === center.id) continue;
      const dist = distances.get(node.id) ?? Infinity;
      if (!rings.has(dist)) rings.set(dist, []);
      rings.get(dist)!.push(node);
    }

    // Position nodes in concentric rings
    for (const [ring, ringNodes] of rings) {
      const radius = ring * this.config.spacing;
      const angleStep = (2 * Math.PI) / ringNodes.length;

      ringNodes.forEach((node, i) => {
        const angle = i * angleStep;
        positions.set(node.id, {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: this.config.dimensions === 3 ? (ring - 1) * 10 : 0,
        });
      });
    }

    this.logger.debug(
      { nodeCount: nodes.length, rings: rings.size },
      "Radial layout computed"
    );

    return positions;
  }

  private computeDistances(
    centerId: string,
    nodes: TopologyNode[],
    edges: TopologyEdge[]
  ): Map<string, number> {
    const distances = new Map<string, number>();
    distances.set(centerId, 0);

    // Build adjacency
    const adj = new Map<string, string[]>();
    for (const edge of edges) {
      if (!adj.has(edge.source)) adj.set(edge.source, []);
      if (!adj.has(edge.target)) adj.set(edge.target, []);
      adj.get(edge.source)!.push(edge.target);
      adj.get(edge.target)!.push(edge.source);
    }

    // BFS from center
    const queue = [centerId];
    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentDist = distances.get(current)!;

      for (const neighbor of adj.get(current) || []) {
        if (!distances.has(neighbor)) {
          distances.set(neighbor, currentDist + 1);
          queue.push(neighbor);
        }
      }
    }

    // Assign max distance to disconnected nodes
    for (const node of nodes) {
      if (!distances.has(node.id)) {
        distances.set(node.id, 100);
      }
    }

    return distances;
  }
}

// ============================================================================
// Render Engine
// ============================================================================

class TopologyRenderer {
  private config: RenderConfig;
  private logger: pino.Logger;
  private nodeTypeGeometries: Record<TopologyNodeType, NodeMesh["geometry"]>;

  constructor(config: RenderConfig, logger: pino.Logger) {
    this.config = RenderConfigSchema.parse(config);
    this.logger = logger.child({ component: "renderer" });

    this.nodeTypeGeometries = {
      namespace: "icosahedron",
      deployment: "cube",
      replicaset: "cube",
      pod: "sphere",
      container: "cylinder",
      service: "octahedron",
      ingress: "octahedron",
      configmap: "cube",
      secret: "cube",
      pvc: "cylinder",
      node: "icosahedron",
      microvm: "sphere",
      wasm_instance: "sphere",
    };
  }

  /**
   * Render the topology graph to mesh data
   */
  render(
    graph: TopologyGraph,
    positions: Map<string, Vector3>,
    selection: { nodeIds: Set<string>; edgeIds: Set<string> }
  ): RenderOutput {
    const nodeMeshes = this.renderNodes(
      graph.nodes,
      positions,
      selection.nodeIds
    );
    const edgeMeshes = this.renderEdges(
      graph.edges,
      positions,
      selection.edgeIds
    );
    const labels = this.renderLabels(graph.nodes, positions);
    const clusterMeshes = this.renderClusters(graph.clusters || [], positions);

    return {
      nodes: nodeMeshes,
      edges: edgeMeshes,
      labels,
      clusters: clusterMeshes,
    };
  }

  private renderNodes(
    nodes: TopologyNode[],
    positions: Map<string, Vector3>,
    selectedIds: Set<string>
  ): NodeMesh[] {
    return nodes.map((node) => {
      const pos = positions.get(node.id) || { x: 0, y: 0, z: 0 };
      const isSelected = selectedIds.has(node.id);

      return {
        nodeId: node.id,
        geometry: this.nodeTypeGeometries[node.type] || "sphere",
        position: pos,
        scale: this.computeNodeScale(node),
        color: isSelected
          ? this.config.colors?.selected || "#00ffff"
          : this.getStatusColor(node.status),
        opacity: 1.0,
        wireframe: false,
        glowIntensity: isSelected ? 0.5 : node.status === "critical" ? 0.3 : 0,
      };
    });
  }

  private renderEdges(
    edges: TopologyEdge[],
    positions: Map<string, Vector3>,
    selectedIds: Set<string>
  ): EdgeMesh[] {
    const result: EdgeMesh[] = [];

    for (const edge of edges) {
      const sourcePos = positions.get(edge.source);
      const targetPos = positions.get(edge.target);

      if (!sourcePos || !targetPos) {
        continue;
      }

      const isSelected = selectedIds.has(edge.id);
      const isTraffic = edge.type === "traffic";

      const edgeMesh: EdgeMesh = {
        edgeId: edge.id,
        geometry: isTraffic ? "tube" : "line",
        points: [sourcePos, targetPos],
        color: isSelected
          ? this.config.colors?.selected || "#00ffff"
          : this.getEdgeColor(edge),
        width: this.computeEdgeWidth(edge),
        opacity: isSelected ? 1.0 : 0.6,
        animated: isTraffic,
      };

      if (isTraffic && edge.trafficMetrics?.requestsPerSecond) {
        edgeMesh.particleFlow = {
          speed: Math.min(edge.trafficMetrics.requestsPerSecond / 100, 5),
          density: Math.min(edge.trafficMetrics.requestsPerSecond / 50, 20),
          color: this.getTrafficColor(edge.trafficMetrics.successRate),
        };
      }

      result.push(edgeMesh);
    }

    return result;
  }

  private renderLabels(
    nodes: TopologyNode[],
    positions: Map<string, Vector3>
  ): LabelMesh[] {
    if (!this.config.labels?.show) return [];

    return nodes.map((node) => {
      const pos = positions.get(node.id) || { x: 0, y: 0, z: 0 };
      const labelText =
        node.name.length > (this.config.labels?.maxLength || 20)
          ? node.name.slice(0, (this.config.labels?.maxLength || 20) - 3) +
            "..."
          : node.name;

      return {
        id: `label-${node.id}`,
        text: labelText,
        position: {
          x: pos.x,
          y: pos.y + 10,
          z: pos.z,
        },
        fontSize: this.config.labels?.fontSize || 12,
        color: "#ffffff",
        visible: true,
      };
    });
  }

  private renderClusters(
    clusters: NonNullable<TopologyGraph["clusters"]>,
    positions: Map<string, Vector3>
  ): RenderOutput["clusters"] {
    return clusters
      .map((cluster) => {
        const nodePositions = cluster.nodeIds
          .map((id) => positions.get(id))
          .filter((p): p is Vector3 => p !== undefined);

        if (nodePositions.length === 0) {
          return null;
        }

        const min: Vector3 = {
          x: Math.min(...nodePositions.map((p) => p.x)) - 20,
          y: Math.min(...nodePositions.map((p) => p.y)) - 20,
          z: Math.min(...nodePositions.map((p) => p.z)) - 20,
        };

        const max: Vector3 = {
          x: Math.max(...nodePositions.map((p) => p.x)) + 20,
          y: Math.max(...nodePositions.map((p) => p.y)) + 20,
          z: Math.max(...nodePositions.map((p) => p.z)) + 20,
        };

        return {
          id: cluster.id,
          boundingBox: { min, max },
          color: cluster.color || "#444444",
          opacity: 0.1,
        };
      })
      .filter((c): c is NonNullable<typeof c> => c !== null);
  }

  private computeNodeScale(node: TopologyNode): Vector3 {
    const sizeConfig = this.config.nodeSize || {
      min: 5,
      max: 30,
      scale: "sqrt",
    };
    let size = sizeConfig.min;

    // Scale based on type importance
    const typeScale: Record<TopologyNodeType, number> = {
      namespace: 1.0,
      node: 0.9,
      deployment: 0.7,
      replicaset: 0.5,
      service: 0.7,
      ingress: 0.7,
      pod: 0.4,
      container: 0.3,
      microvm: 0.4,
      wasm_instance: 0.3,
      configmap: 0.2,
      secret: 0.2,
      pvc: 0.3,
    };

    const scale = typeScale[node.type] ?? 0.5;
    size = sizeConfig.min + (sizeConfig.max - sizeConfig.min) * scale;

    return { x: size, y: size, z: size };
  }

  private computeEdgeWidth(edge: TopologyEdge): number {
    const widthConfig = this.config.edgeWidth || { min: 1, max: 10 };

    if (edge.trafficMetrics?.requestsPerSecond) {
      const rps = edge.trafficMetrics.requestsPerSecond;
      const normalized = Math.min(Math.log10(rps + 1) / 3, 1);
      return widthConfig.min + (widthConfig.max - widthConfig.min) * normalized;
    }

    return widthConfig.min + edge.weight * (widthConfig.max - widthConfig.min);
  }

  private getStatusColor(status: HealthStatus): string {
    const colors = this.config.colors || {
      healthy: "#00ff00",
      warning: "#ffff00",
      critical: "#ff0000",
      unknown: "#888888",
    };

    return colors[status] || colors.unknown || "#888888";
  }

  private getEdgeColor(edge: TopologyEdge): string {
    switch (edge.type) {
      case "traffic":
        return this.getTrafficColor(edge.trafficMetrics?.successRate);
      case "network_policy":
        return "#ff6600";
      case "owns":
      case "manages":
        return "#4488ff";
      case "selects":
        return "#88ff44";
      case "mounts":
        return "#ff44ff";
      default:
        return "#888888";
    }
  }

  private getTrafficColor(successRate?: number): string {
    if (successRate === undefined) return "#888888";
    if (successRate >= 99) return "#00ff00";
    if (successRate >= 95) return "#88ff00";
    if (successRate >= 90) return "#ffff00";
    if (successRate >= 80) return "#ff8800";
    return "#ff0000";
  }
}

// ============================================================================
// Data Collectors
// ============================================================================

interface DataCollector {
  collect(): Promise<{ nodes: TopologyNode[]; edges: TopologyEdge[] }>;
}

class KubernetesCollector implements DataCollector {
  private logger: pino.Logger;
  private kubeClient: any;
  private namespace?: string;

  constructor(kubeClient: any, namespace?: string, logger?: pino.Logger) {
    this.kubeClient = kubeClient;
    this.namespace = namespace;
    this.logger = (logger || pino()).child({ collector: "kubernetes" });
  }

  async collect(): Promise<{ nodes: TopologyNode[]; edges: TopologyEdge[] }> {
    const nodes: TopologyNode[] = [];
    const edges: TopologyEdge[] = [];

    // In production, would query Kubernetes API
    // This is a placeholder showing structure

    this.logger.debug("Collecting Kubernetes resources");

    return { nodes, edges };
  }
}

class DockerCollector implements DataCollector {
  private logger: pino.Logger;
  private dockerClient: any;

  constructor(dockerClient: any, logger?: pino.Logger) {
    this.dockerClient = dockerClient;
    this.logger = (logger || pino()).child({ collector: "docker" });
  }

  async collect(): Promise<{ nodes: TopologyNode[]; edges: TopologyEdge[] }> {
    const nodes: TopologyNode[] = [];
    const edges: TopologyEdge[] = [];

    this.logger.debug("Collecting Docker resources");

    return { nodes, edges };
  }
}

// ============================================================================
// Topology Manager
// ============================================================================

export interface TopologyManagerOptions {
  layoutConfig?: Partial<LayoutConfig>;
  renderConfig?: Partial<RenderConfig>;
  collectors?: DataCollector[];
  pollInterval?: number;
  logger?: pino.Logger;
}

/**
 * 3D Container Topology Manager
 *
 * @agent @PIXEL @FLUX
 *
 * Manages 3D visualization of container relationships and resource topology.
 * Provides real-time updates, interactive navigation, and multiple layout algorithms.
 *
 * @example
 * ```typescript
 * const topology = new TopologyManager({
 *   layoutConfig: { algorithm: "force_directed" },
 *   renderConfig: {
 *     colors: { healthy: "#00ff00", critical: "#ff0000" }
 *   },
 *   pollInterval: 5000
 * });
 *
 * await topology.initialize();
 *
 * // Subscribe to updates
 * topology.on("graph:updated", ({ graph }) => {
 *   // Re-render the scene with new data
 *   const output = topology.render();
 *   scene.update(output);
 * });
 *
 * // Add nodes manually
 * topology.addNode({
 *   id: "pod-1",
 *   type: "pod",
 *   name: "agent-pod-1",
 *   namespace: "neurectomy",
 *   status: "healthy"
 * });
 *
 * // Start automatic data collection
 * await topology.startPolling();
 * ```
 */
export class TopologyManager extends EventEmitter<TopologyEvents> {
  private logger: pino.Logger;
  private layoutConfig: LayoutConfig;
  private renderConfig: RenderConfig;
  private collectors: DataCollector[];
  private pollInterval: number;

  private graph: TopologyGraph;
  private positions: Map<string, Vector3> = new Map();
  private selection: { nodeIds: Set<string>; edgeIds: Set<string> };
  private hoveredNode: string | null = null;
  private hoveredEdge: string | null = null;

  private layoutEngine: ForceDirectedLayout | HierarchicalLayout | RadialLayout;
  private renderer: TopologyRenderer;

  private pollTimer?: NodeJS.Timeout;
  private animationFrame: number = 0;
  private lastRenderTime: number = 0;

  constructor(options: TopologyManagerOptions = {}) {
    super();

    this.logger = (options.logger || pino()).child({ module: "topology-3d" });
    this.layoutConfig = LayoutConfigSchema.parse(options.layoutConfig || {});
    this.renderConfig = RenderConfigSchema.parse(options.renderConfig || {});
    this.collectors = options.collectors || [];
    this.pollInterval = options.pollInterval || 5000;

    // Initialize empty graph
    this.graph = {
      id: "main",
      name: "Container Topology",
      nodes: [],
      edges: [],
      timestamp: new Date(),
    };

    this.selection = {
      nodeIds: new Set(),
      edgeIds: new Set(),
    };

    // Initialize layout engine
    this.layoutEngine = this.createLayoutEngine();
    this.renderer = new TopologyRenderer(this.renderConfig, this.logger);
  }

  private createLayoutEngine():
    | ForceDirectedLayout
    | HierarchicalLayout
    | RadialLayout {
    switch (this.layoutConfig.algorithm) {
      case "hierarchical":
        return new HierarchicalLayout(this.layoutConfig, this.logger);
      case "radial":
        return new RadialLayout(this.layoutConfig, this.logger);
      case "force_directed":
      default:
        return new ForceDirectedLayout(this.layoutConfig, this.logger);
    }
  }

  /**
   * Initialize the topology manager
   */
  async initialize(): Promise<void> {
    this.logger.info("Initializing topology manager");

    // Perform initial data collection
    await this.collectData();

    this.logger.info(
      { nodeCount: this.graph.nodes.length },
      "Topology initialized"
    );
  }

  /**
   * Start automatic data polling
   */
  startPolling(): void {
    if (this.pollTimer) return;

    this.logger.info({ interval: this.pollInterval }, "Starting data polling");

    this.pollTimer = setInterval(async () => {
      try {
        await this.collectData();
      } catch (error) {
        this.logger.error({ error }, "Data collection failed");
        this.emit("error", {
          operation: "collectData",
          error: error as Error,
        });
      }
    }, this.pollInterval);
  }

  /**
   * Stop automatic data polling
   */
  stopPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = undefined;
      this.logger.info("Stopped data polling");
    }
  }

  /**
   * Collect data from all collectors
   */
  async collectData(): Promise<void> {
    const oldGraph = this.graph;
    const allNodes: TopologyNode[] = [];
    const allEdges: TopologyEdge[] = [];

    for (const collector of this.collectors) {
      try {
        const { nodes, edges } = await collector.collect();
        allNodes.push(...nodes);
        allEdges.push(...edges);
      } catch (error) {
        this.logger.warn({ error }, "Collector failed");
      }
    }

    // Merge with existing manual nodes/edges
    const manualNodes = this.graph.nodes.filter((n) => n.metadata?.manual);
    const manualEdges = this.graph.edges.filter((e) => e.metadata?.manual);

    this.graph = {
      ...this.graph,
      nodes: [...allNodes, ...manualNodes],
      edges: [...allEdges, ...manualEdges],
      timestamp: new Date(),
    };

    // Compute diff
    const diff = this.computeDiff(oldGraph, this.graph);

    // Recompute layout if topology changed
    if (diff.nodesAdded.length > 0 || diff.nodesRemoved.length > 0) {
      this.computeLayout();
    }

    this.emit("graph:updated", { graph: this.graph, diff });
  }

  /**
   * Add a node manually
   */
  addNode(
    node: Omit<TopologyNode, "metadata"> & {
      metadata?: Record<string, unknown>;
    }
  ): void {
    const validatedNode = TopologyNodeSchema.parse({
      ...node,
      metadata: { ...node.metadata, manual: true },
    });

    this.graph.nodes.push(validatedNode);
    this.computeLayout();

    this.emit("node:added", { node: validatedNode });
    this.emit("graph:updated", {
      graph: this.graph,
      diff: {
        nodesAdded: [validatedNode.id],
        nodesRemoved: [],
        nodesUpdated: [],
        edgesAdded: [],
        edgesRemoved: [],
        edgesUpdated: [],
      },
    });
  }

  /**
   * Remove a node
   */
  removeNode(nodeId: string): void {
    const index = this.graph.nodes.findIndex((n) => n.id === nodeId);
    if (index === -1) return;

    this.graph.nodes.splice(index, 1);

    // Remove connected edges
    const removedEdges: string[] = [];
    this.graph.edges = this.graph.edges.filter((e) => {
      if (e.source === nodeId || e.target === nodeId) {
        removedEdges.push(e.id);
        return false;
      }
      return true;
    });

    this.positions.delete(nodeId);
    this.selection.nodeIds.delete(nodeId);

    this.emit("node:removed", { nodeId });
    this.emit("graph:updated", {
      graph: this.graph,
      diff: {
        nodesAdded: [],
        nodesRemoved: [nodeId],
        nodesUpdated: [],
        edgesAdded: [],
        edgesRemoved: removedEdges,
        edgesUpdated: [],
      },
    });
  }

  /**
   * Update a node
   */
  updateNode(nodeId: string, changes: Partial<TopologyNode>): void {
    const node = this.graph.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    Object.assign(node, changes);

    this.emit("node:updated", { node, changes });
  }

  /**
   * Add an edge
   */
  addEdge(edge: TopologyEdge): void {
    const validatedEdge = TopologyEdgeSchema.parse({
      ...edge,
      metadata: { ...edge.metadata, manual: true },
    });

    this.graph.edges.push(validatedEdge);

    this.emit("edge:added", { edge: validatedEdge });
  }

  /**
   * Remove an edge
   */
  removeEdge(edgeId: string): void {
    const index = this.graph.edges.findIndex((e) => e.id === edgeId);
    if (index === -1) return;

    this.graph.edges.splice(index, 1);
    this.selection.edgeIds.delete(edgeId);

    this.emit("edge:removed", { edgeId });
  }

  /**
   * Compute layout positions
   */
  computeLayout(): void {
    const startTime = Date.now();

    if (this.layoutEngine instanceof RadialLayout) {
      this.positions = this.layoutEngine.compute(
        this.graph.nodes,
        this.graph.edges
      );
    } else {
      this.positions = this.layoutEngine.compute(
        this.graph.nodes,
        this.graph.edges
      );
    }

    const duration = Date.now() - startTime;
    this.emit("layout:computed", { duration });
  }

  /**
   * Change layout algorithm
   */
  setLayoutAlgorithm(algorithm: LayoutAlgorithm): void {
    this.layoutConfig.algorithm = algorithm;
    this.layoutEngine = this.createLayoutEngine();
    this.computeLayout();
  }

  /**
   * Render the current state
   */
  render(): RenderOutput {
    const now = Date.now();
    const fps =
      this.lastRenderTime > 0 ? 1000 / (now - this.lastRenderTime) : 60;
    this.lastRenderTime = now;

    this.emit("render:frame", { fps });

    return this.renderer.render(this.graph, this.positions, this.selection);
  }

  /**
   * Select nodes/edges
   */
  select(nodeIds: string[], edgeIds: string[] = []): void {
    this.selection.nodeIds = new Set(nodeIds);
    this.selection.edgeIds = new Set(edgeIds);

    this.emit("selection:changed", { nodeIds, edgeIds });
  }

  /**
   * Clear selection
   */
  clearSelection(): void {
    this.selection.nodeIds.clear();
    this.selection.edgeIds.clear();

    this.emit("selection:changed", { nodeIds: [], edgeIds: [] });
  }

  /**
   * Set hovered node
   */
  hoverNode(nodeId: string | null): void {
    if (this.hoveredNode !== nodeId) {
      this.hoveredNode = nodeId;
      this.emit("hover:node", { nodeId });
    }
  }

  /**
   * Set hovered edge
   */
  hoverEdge(edgeId: string | null): void {
    if (this.hoveredEdge !== edgeId) {
      this.hoveredEdge = edgeId;
      this.emit("hover:edge", { edgeId });
    }
  }

  /**
   * Get current graph
   */
  getGraph(): TopologyGraph {
    return { ...this.graph };
  }

  /**
   * Get node position
   */
  getNodePosition(nodeId: string): Vector3 | undefined {
    return this.positions.get(nodeId);
  }

  /**
   * Get all positions
   */
  getAllPositions(): Map<string, Vector3> {
    return new Map(this.positions);
  }

  /**
   * Find node by coordinates (for picking)
   */
  findNodeAtPosition(
    point: Vector3,
    threshold: number = 20
  ): TopologyNode | null {
    for (const node of this.graph.nodes) {
      const pos = this.positions.get(node.id);
      if (!pos) continue;

      const dx = pos.x - point.x;
      const dy = pos.y - point.y;
      const dz = pos.z - point.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < threshold) {
        return node;
      }
    }

    return null;
  }

  /**
   * Expand cluster to show individual nodes
   */
  expandCluster(clusterId: string): void {
    // Implementation would break apart cluster into individual nodes
    this.emit("cluster:expanded", { clusterId });
  }

  /**
   * Collapse nodes into a cluster
   */
  collapseToCluster(nodeIds: string[], clusterId: string, name: string): void {
    if (!this.graph.clusters) {
      this.graph.clusters = [];
    }

    this.graph.clusters.push({
      id: clusterId,
      name,
      nodeIds,
    });

    this.emit("cluster:collapsed", { clusterId });
  }

  /**
   * Cleanup
   */
  cleanup(): void {
    this.stopPolling();
    this.removeAllListeners();
    this.logger.info("Topology manager cleaned up");
  }

  // Private methods

  private computeDiff(
    oldGraph: TopologyGraph,
    newGraph: TopologyGraph
  ): GraphDiff {
    const oldNodeIds = new Set(oldGraph.nodes.map((n) => n.id));
    const newNodeIds = new Set(newGraph.nodes.map((n) => n.id));
    const oldEdgeIds = new Set(oldGraph.edges.map((e) => e.id));
    const newEdgeIds = new Set(newGraph.edges.map((e) => e.id));

    return {
      nodesAdded: [...newNodeIds].filter((id) => !oldNodeIds.has(id)),
      nodesRemoved: [...oldNodeIds].filter((id) => !newNodeIds.has(id)),
      nodesUpdated: [], // Would need deep comparison
      edgesAdded: [...newEdgeIds].filter((id) => !oldEdgeIds.has(id)),
      edgesRemoved: [...oldEdgeIds].filter((id) => !newEdgeIds.has(id)),
      edgesUpdated: [],
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create topology manager with Kubernetes collector
 */
export function createKubernetesTopology(
  kubeClient: any,
  options?: TopologyManagerOptions & { namespace?: string }
): TopologyManager {
  const collector = new KubernetesCollector(
    kubeClient,
    options?.namespace,
    options?.logger
  );

  return new TopologyManager({
    ...options,
    collectors: [collector, ...(options?.collectors || [])],
  });
}

/**
 * Create topology manager with Docker collector
 */
export function createDockerTopology(
  dockerClient: any,
  options?: TopologyManagerOptions
): TopologyManager {
  const collector = new DockerCollector(dockerClient, options?.logger);

  return new TopologyManager({
    ...options,
    collectors: [collector, ...(options?.collectors || [])],
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate sample topology for testing
 */
export function generateSampleTopology(): TopologyGraph {
  const nodes: TopologyNode[] = [
    { id: "ns-1", type: "namespace", name: "neurectomy", status: "healthy" },
    {
      id: "deploy-1",
      type: "deployment",
      name: "agent-controller",
      namespace: "neurectomy",
      status: "healthy",
    },
    {
      id: "rs-1",
      type: "replicaset",
      name: "agent-controller-abc123",
      namespace: "neurectomy",
      status: "healthy",
    },
    {
      id: "pod-1",
      type: "pod",
      name: "agent-controller-abc123-xyz",
      namespace: "neurectomy",
      status: "healthy",
    },
    {
      id: "pod-2",
      type: "pod",
      name: "agent-controller-abc123-uvw",
      namespace: "neurectomy",
      status: "healthy",
    },
    {
      id: "svc-1",
      type: "service",
      name: "agent-controller",
      namespace: "neurectomy",
      status: "healthy",
    },
    {
      id: "ing-1",
      type: "ingress",
      name: "agent-ingress",
      namespace: "neurectomy",
      status: "healthy",
    },
  ];

  const edges: TopologyEdge[] = [
    {
      id: "e-1",
      source: "ns-1",
      target: "deploy-1",
      type: "owns",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-2",
      source: "deploy-1",
      target: "rs-1",
      type: "manages",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-3",
      source: "rs-1",
      target: "pod-1",
      type: "manages",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-4",
      source: "rs-1",
      target: "pod-2",
      type: "manages",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-5",
      source: "svc-1",
      target: "pod-1",
      type: "selects",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-6",
      source: "svc-1",
      target: "pod-2",
      type: "selects",
      weight: 1,
      bidirectional: false,
    },
    {
      id: "e-7",
      source: "ing-1",
      target: "svc-1",
      type: "exposes",
      weight: 1,
      bidirectional: false,
    },
  ];

  return {
    id: "sample",
    name: "Sample Topology",
    nodes,
    edges,
    timestamp: new Date(),
  };
}

/**
 * Export topology to JSON
 */
export function exportTopologyToJSON(graph: TopologyGraph): string {
  return JSON.stringify(graph, null, 2);
}

/**
 * Import topology from JSON
 */
export function importTopologyFromJSON(json: string): TopologyGraph {
  const parsed = JSON.parse(json);
  return TopologyGraphSchema.parse({
    ...parsed,
    timestamp: new Date(parsed.timestamp),
  });
}
