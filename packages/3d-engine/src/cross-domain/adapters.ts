/**
 * Cross-Domain Type Adapters
 *
 * Bidirectional transformers that convert between domain-specific types
 * and the unified type system. These enable seamless interoperability
 * between Forge, Twin, and Foundry modules.
 *
 * @module @neurectomy/3d-engine/cross-domain/adapters
 * @agents @SYNAPSE @MORPH @APEX
 */

import type {
  UnifiedEntity,
  UnifiedGraph,
  UnifiedEdge,
  UnifiedTemporalPoint,
  UnifiedTimeline,
  UnifiedMetrics,
  UnifiedEvent,
  EntityMetadata,
  ForgeMetadata,
  TwinMetadata,
  FoundryMetadata,
  Vector3,
  EulerAngles,
  VisualStyle,
  EdgeType,
  Domain,
  UniversalId,
  Timestamp,
  PlaybackConfig,
  TimeRange,
} from "./types";

// Import domain-specific types (these would come from the actual modules)
import type {
  AgentComponent,
  AgentComponentType,
  ComponentConnection,
  ComponentStatus,
} from "../visualization/types";

import type {
  TwinState,
  AgentStateSnapshot,
  ComponentGraphSnapshot,
  ComponentNode,
  ComponentEdge,
  TwinMode,
  SyncState as TwinSyncState,
  TwinFidelity,
} from "../digital-twin/types";

import type { StateSnapshot, Timeline, PlaybackState } from "../temporal/types";

// ============================================================================
// Forge Adapter
// ============================================================================

/**
 * Adapter for converting Forge types to/from unified types
 */
export class ForgeAdapter {
  /**
   * Convert AgentComponent to UnifiedEntity
   */
  static toUnifiedEntity(component: AgentComponent): UnifiedEntity {
    return {
      id: component.id,
      name: component.name,
      type: component.type,
      version: component.metadata.version ?? "1.0.0",
      config: component.metadata.config ?? {},
      state: {
        status: component.metadata.status,
        metrics: component.metadata.metrics,
      },
      inputs: [], // Would be derived from connection analysis
      outputs: [],
      metadata: {
        description: component.metadata.description,
        tags: component.metadata.tags,
        annotations: component.metadata.properties ?? {},
        forge: {
          position: {
            x: component.position.x,
            y: component.position.y,
            z: component.position.z,
          },
          rotation: {
            x: component.rotation.x,
            y: component.rotation.y,
            z: component.rotation.z,
            order: component.rotation.order as "XYZ",
          },
          scale: {
            x: component.scale.x,
            y: component.scale.y,
            z: component.scale.z,
          },
          style: {
            color: component.style.color,
            opacity: component.style.opacity,
            emissive: component.style.emissive,
            emissiveIntensity: component.style.emissiveIntensity,
            wireframe: component.style.wireframe,
            geometry: component.style.geometry,
          },
          visible: true, // Default
          selected: false, // Default
          lodLevel: 0, // Default
        },
      },
      parentId: component.parentId,
      childIds: component.childIds,
      createdAt: Date.now(),
      modifiedAt: Date.now(),
    };
  }

  /**
   * Convert UnifiedEntity to AgentComponent
   */
  static fromUnifiedEntity(entity: UnifiedEntity): AgentComponent {
    const forge = entity.metadata.forge;
    const state = entity.state as {
      status?: ComponentStatus;
      metrics?: unknown;
    };

    return {
      id: entity.id,
      type: entity.type as AgentComponentType,
      name: entity.name,
      position: forge?.position
        ? ({
            x: forge.position.x,
            y: forge.position.y,
            z: forge.position.z,
          } as any)
        : ({ x: 0, y: 0, z: 0 } as any),
      rotation: forge?.rotation
        ? ({
            x: forge.rotation.x,
            y: forge.rotation.y,
            z: forge.rotation.z,
            order: forge.rotation.order ?? "XYZ",
          } as any)
        : ({ x: 0, y: 0, z: 0, order: "XYZ" } as any),
      scale: forge?.scale
        ? ({ x: forge.scale.x, y: forge.scale.y, z: forge.scale.z } as any)
        : ({ x: 1, y: 1, z: 1 } as any),
      metadata: {
        version: entity.version,
        description: entity.metadata.description,
        config: entity.config as Record<string, unknown>,
        metrics: state.metrics as any,
        status: state.status ?? "idle",
        tags: entity.metadata.tags,
        properties: entity.metadata.annotations,
      },
      style: forge?.style ?? {
        color: "#3b82f6",
        opacity: 1,
        geometry: "box",
      },
      parentId: entity.parentId,
      childIds: entity.childIds,
      connectionIds: [],
    };
  }

  /**
   * Convert ComponentConnection to UnifiedEdge
   */
  static connectionToEdge(connection: ComponentConnection): UnifiedEdge {
    return {
      id: connection.id,
      sourceId: connection.sourceId,
      targetId: connection.targetId,
      type: this.mapConnectionTypeToEdgeType(connection.type),
      weight: 1.0,
      bidirectional: connection.direction === "bidirectional",
      metadata: {
        label: connection.metadata.label,
        animated: connection.animated,
        style: {
          color: connection.style.color,
          width: connection.style.width,
          dashed: connection.style.dashed,
          opacity: connection.style.opacity,
        },
      },
    };
  }

  private static mapConnectionTypeToEdgeType(type: string): EdgeType {
    const mapping: Record<string, EdgeType> = {
      data: "data-flow",
      control: "control-flow",
      memory: "memory-access",
      event: "event",
      feedback: "data-flow",
      dependency: "dependency",
    };
    return mapping[type] ?? "data-flow";
  }
}

// ============================================================================
// Twin Adapter
// ============================================================================

/**
 * Adapter for converting Twin types to/from unified types
 */
export class TwinAdapter {
  /**
   * Convert TwinState to UnifiedEntity
   */
  static toUnifiedEntity(twin: TwinState): UnifiedEntity {
    return {
      id: twin.id,
      name: twin.name,
      type: "twin",
      version: twin.metadata.version,
      config: twin.agentState.config,
      state: {
        parameters: twin.agentState.parameters,
        internalState: twin.agentState.internalState,
        ioHistory: twin.agentState.ioHistory,
      },
      inputs: [],
      outputs: [],
      metadata: {
        description: twin.metadata.description,
        tags: twin.metadata.tags,
        annotations: twin.metadata.properties,
        twin: {
          mode: twin.mode,
          syncState: twin.syncState,
          fidelity: twin.fidelity as "full" | "reduced" | "minimal",
          divergenceScore: twin.divergenceScore,
          sourceAgentId: twin.agentId,
          lastSyncAt: twin.lastSyncAt,
        },
      },
      parentId: twin.metadata.parentTwinId,
      childIds: [],
      createdAt: twin.createdAt,
      modifiedAt: twin.modifiedAt,
    };
  }

  /**
   * Convert UnifiedEntity to TwinState
   */
  static fromUnifiedEntity(entity: UnifiedEntity): TwinState {
    const twin = entity.metadata.twin;
    const state = entity.state as {
      parameters?: Record<string, unknown>;
      internalState?: Record<string, unknown>;
      ioHistory?: Array<{ timestamp: number; type: string; data: unknown }>;
    };

    return {
      id: entity.id,
      agentId: twin?.sourceAgentId ?? entity.id,
      name: entity.name,
      mode: twin?.mode ?? "snapshot",
      syncState: twin?.syncState ?? "disconnected",
      fidelity: twin?.fidelity ?? "full",
      createdAt: entity.createdAt,
      lastSyncAt: twin?.lastSyncAt ?? entity.modifiedAt,
      modifiedAt: entity.modifiedAt,
      agentState: {
        config: entity.config as Record<string, unknown>,
        parameters: state.parameters ?? {},
        internalState: state.internalState ?? {},
        ioHistory: (state.ioHistory ?? []) as any,
        metrics: {
          responseTime: {
            min: 0,
            max: 0,
            mean: 0,
            median: 0,
            p95: 0,
            p99: 0,
            stdDev: 0,
          },
          throughput: {
            min: 0,
            max: 0,
            mean: 0,
            median: 0,
            p95: 0,
            p99: 0,
            stdDev: 0,
          },
          errorRate: 0,
          resourceUtilization: {
            cpuPercent: 0,
            memoryMB: 0,
            networkBytesIn: 0,
            networkBytesOut: 0,
          },
          custom: {},
        },
        componentGraph: {
          nodes: [],
          edges: [],
          rootId: entity.id,
        },
      },
      metadata: {
        description: entity.metadata.description,
        tags: entity.metadata.tags,
        createdBy: undefined,
        version: entity.version,
        parentTwinId: entity.parentId,
        branch: twin?.sourceAgentId,
        properties: entity.metadata.annotations,
      },
      divergenceScore: twin?.divergenceScore ?? 0,
    };
  }

  /**
   * Convert ComponentGraphSnapshot to UnifiedGraph
   */
  static graphToUnified(
    graph: ComponentGraphSnapshot,
    name: string = "Twin Graph"
  ): UnifiedGraph {
    const nodes = new Map<UniversalId, UnifiedEntity>();
    const edges = new Map<UniversalId, UnifiedEdge>();

    // Convert nodes
    for (const node of graph.nodes) {
      nodes.set(node.id, {
        id: node.id,
        name: node.name,
        type: node.type,
        version: "1.0.0",
        config: node.config,
        state: { status: node.state },
        inputs: [],
        outputs: [],
        metadata: {
          tags: [],
          annotations: {},
          forge: node.position
            ? {
                position: node.position,
                rotation: { x: 0, y: 0, z: 0 },
                scale: { x: 1, y: 1, z: 1 },
                style: { color: "#3b82f6", opacity: 1 },
                visible: true,
                selected: false,
                lodLevel: 0,
              }
            : undefined,
        },
        childIds: [],
        createdAt: Date.now(),
        modifiedAt: Date.now(),
      });
    }

    // Convert edges
    for (const edge of graph.edges) {
      edges.set(edge.id, {
        id: edge.id,
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        type: edge.type as EdgeType,
        weight: edge.weight ?? 1.0,
        bidirectional: false,
        metadata: {},
      });
    }

    return {
      id: `graph-${Date.now()}`,
      name,
      nodes,
      edges,
      rootId: graph.rootId,
      metadata: {
        tags: [],
        nodeCount: nodes.size,
        edgeCount: edges.size,
        maxDepth: this.calculateMaxDepth(graph),
        isAcyclic: true, // Assume for now
        domains: ["twin"],
      },
      version: "1.0.0",
    };
  }

  private static calculateMaxDepth(graph: ComponentGraphSnapshot): number {
    // Simple BFS to find max depth
    if (!graph.rootId || graph.nodes.length === 0) return 0;

    const children = new Map<string, string[]>();
    for (const edge of graph.edges) {
      if (!children.has(edge.sourceId)) {
        children.set(edge.sourceId, []);
      }
      children.get(edge.sourceId)!.push(edge.targetId);
    }

    let maxDepth = 0;
    const queue: Array<{ id: string; depth: number }> = [
      { id: graph.rootId, depth: 0 },
    ];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      maxDepth = Math.max(maxDepth, depth);

      for (const childId of children.get(id) ?? []) {
        queue.push({ id: childId, depth: depth + 1 });
      }
    }

    return maxDepth;
  }
}

// ============================================================================
// Foundry Adapter
// ============================================================================

/**
 * Neural network layer representation for Foundry
 */
export interface ModelLayer {
  id: string;
  name: string;
  type: string;
  inputShape: number[];
  outputShape: number[];
  parameters: Record<string, unknown>;
  trainable: boolean;
  dtype: string;
}

/**
 * Neural architecture representation
 */
export interface NeuralArchitecture {
  id: string;
  name: string;
  layers: ModelLayer[];
  connections: Array<{ from: string; to: string; type: string }>;
  totalParams: number;
  trainableParams: number;
}

/**
 * Training checkpoint
 */
export interface TrainingCheckpoint {
  id: string;
  epoch: number;
  step: number;
  timestamp: Timestamp;
  metrics: Record<string, number>;
  weights?: ArrayBuffer;
  optimizerState?: Record<string, unknown>;
}

/**
 * Adapter for converting Foundry types to/from unified types
 */
export class FoundryAdapter {
  /**
   * Convert ModelLayer to UnifiedEntity
   */
  static toUnifiedEntity(layer: ModelLayer): UnifiedEntity {
    return {
      id: layer.id,
      name: layer.name,
      type: layer.type,
      version: "1.0.0",
      config: {
        inputShape: layer.inputShape,
        outputShape: layer.outputShape,
        trainable: layer.trainable,
        dtype: layer.dtype,
        ...layer.parameters,
      },
      state: {
        initialized: true,
        trainingState: "idle",
      },
      inputs: [
        {
          id: `${layer.id}-input`,
          name: "input",
          dataType: `tensor<${layer.dtype}>`,
          required: true,
        },
      ],
      outputs: [
        {
          id: `${layer.id}-output`,
          name: "output",
          dataType: `tensor<${layer.dtype}>`,
          required: true,
        },
      ],
      metadata: {
        tags: ["neural-layer", layer.type],
        annotations: {},
        foundry: {
          layerType: layer.type as any,
          parameterCount: this.estimateParams(layer),
          flops: this.estimateFLOPs(layer),
          memoryBytes: this.estimateMemory(layer),
          trainingState: "idle",
        },
      },
      childIds: [],
      createdAt: Date.now(),
      modifiedAt: Date.now(),
    };
  }

  /**
   * Convert UnifiedEntity to ModelLayer
   */
  static fromUnifiedEntity(entity: UnifiedEntity): ModelLayer {
    const config = entity.config as Record<string, unknown>;
    const foundry = entity.metadata.foundry;

    return {
      id: entity.id,
      name: entity.name,
      type: foundry?.layerType ?? entity.type,
      inputShape: (config.inputShape as number[]) ?? [],
      outputShape: (config.outputShape as number[]) ?? [],
      parameters: config,
      trainable: (config.trainable as boolean) ?? true,
      dtype: (config.dtype as string) ?? "float32",
    };
  }

  /**
   * Convert NeuralArchitecture to UnifiedGraph
   */
  static architectureToUnified(arch: NeuralArchitecture): UnifiedGraph {
    const nodes = new Map<UniversalId, UnifiedEntity>();
    const edges = new Map<UniversalId, UnifiedEdge>();

    // Convert layers to nodes
    for (let i = 0; i < arch.layers.length; i++) {
      const layer = arch.layers[i];
      const entity = this.toUnifiedEntity(layer);

      // Add spatial positioning for visualization
      entity.metadata.forge = {
        position: { x: 0, y: i * 2, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
        style: {
          color: this.getLayerColor(layer.type),
          opacity: 1,
          geometry: this.getLayerGeometry(layer.type),
        },
        visible: true,
        selected: false,
        lodLevel: 0,
      };

      nodes.set(layer.id, entity);
    }

    // Convert connections to edges
    for (const conn of arch.connections) {
      const edgeId = `${conn.from}-${conn.to}`;
      edges.set(edgeId, {
        id: edgeId,
        sourceId: conn.from,
        targetId: conn.to,
        type: conn.type === "skip" ? "skip-connection" : "forward",
        weight: 1.0,
        bidirectional: false,
        metadata: {},
      });
    }

    // Find root (input layer)
    const targetIds = new Set(arch.connections.map((c) => c.to));
    const rootId = arch.layers.find((l) => !targetIds.has(l.id))?.id;

    return {
      id: arch.id,
      name: arch.name,
      nodes,
      edges,
      rootId,
      metadata: {
        description: `Neural architecture with ${arch.totalParams} parameters`,
        tags: ["neural-architecture"],
        nodeCount: nodes.size,
        edgeCount: edges.size,
        maxDepth: arch.layers.length,
        isAcyclic: true, // Standard feed-forward assumption
        domains: ["foundry"],
      },
      version: "1.0.0",
    };
  }

  /**
   * Convert TrainingCheckpoint to UnifiedTemporalPoint
   */
  static checkpointToTemporal(
    checkpoint: TrainingCheckpoint
  ): UnifiedTemporalPoint {
    return {
      id: checkpoint.id,
      timestamp: checkpoint.timestamp,
      state: {
        epoch: checkpoint.epoch,
        step: checkpoint.step,
        metrics: checkpoint.metrics,
        hasWeights: !!checkpoint.weights,
      },
      hash: `ckpt-${checkpoint.epoch}-${checkpoint.step}`,
      isKeyframe: true, // Checkpoints are always keyframes
      metadata: {
        label: `Epoch ${checkpoint.epoch}, Step ${checkpoint.step}`,
        tags: ["checkpoint", "training"],
        source: "checkpoint",
        sizeBytes: checkpoint.weights?.byteLength ?? 0,
        foundryData: {
          epoch: checkpoint.epoch,
          step: checkpoint.step,
          learningRate: checkpoint.metrics.learningRate ?? 0,
          loss: checkpoint.metrics.loss ?? 0,
          metrics: checkpoint.metrics,
        },
      },
    };
  }

  private static estimateParams(layer: ModelLayer): number {
    // Simplified parameter estimation
    const inSize = layer.inputShape.reduce((a, b) => a * b, 1);
    const outSize = layer.outputShape.reduce((a, b) => a * b, 1);

    switch (layer.type) {
      case "dense":
        return inSize * outSize + outSize; // weights + bias
      case "conv2d":
        const kernelSize = (layer.parameters.kernelSize as number) ?? 3;
        const filters = layer.outputShape[layer.outputShape.length - 1];
        const channels = layer.inputShape[layer.inputShape.length - 1];
        return kernelSize * kernelSize * channels * filters + filters;
      default:
        return inSize * outSize;
    }
  }

  private static estimateFLOPs(layer: ModelLayer): number {
    const params = this.estimateParams(layer);
    return params * 2; // Simplified: 1 mult + 1 add per parameter
  }

  private static estimateMemory(layer: ModelLayer): number {
    const params = this.estimateParams(layer);
    return params * 4; // float32 = 4 bytes
  }

  private static getLayerColor(type: string): string {
    const colors: Record<string, string> = {
      input: "#22c55e",
      dense: "#3b82f6",
      conv2d: "#8b5cf6",
      conv1d: "#a78bfa",
      lstm: "#ec4899",
      gru: "#f43f5e",
      attention: "#f59e0b",
      transformer: "#eab308",
      embedding: "#06b6d4",
      normalization: "#64748b",
      dropout: "#94a3b8",
      pooling: "#6366f1",
      output: "#ef4444",
    };
    return colors[type] ?? "#64748b";
  }

  private static getLayerGeometry(type: string): string {
    const geometries: Record<string, string> = {
      input: "cylinder",
      dense: "box",
      conv2d: "box",
      attention: "octahedron",
      transformer: "icosahedron",
      output: "sphere",
    };
    return geometries[type] ?? "box";
  }
}

// ============================================================================
// Temporal Adapter
// ============================================================================

/**
 * Adapter for converting Temporal types to/from unified types
 */
export class TemporalAdapter {
  /**
   * Convert StateSnapshot to UnifiedTemporalPoint
   */
  static toUnifiedPoint<T>(
    snapshot: StateSnapshot<T>
  ): UnifiedTemporalPoint<T> {
    return {
      id: snapshot.id,
      timestamp: snapshot.timestamp,
      state: snapshot.state,
      hash: snapshot.hash,
      parentId: snapshot.parentId,
      delta: snapshot.delta,
      isKeyframe: snapshot.metadata.isKeyframe,
      metadata: {
        label: snapshot.metadata.label,
        description: snapshot.metadata.description,
        tags: snapshot.metadata.tags,
        source: snapshot.metadata.source,
        sizeBytes: snapshot.metadata.sizeBytes,
        compression: snapshot.metadata.compression,
      },
    };
  }

  /**
   * Convert Timeline to UnifiedTimeline
   */
  static toUnifiedTimeline<T>(timeline: Timeline<T>): UnifiedTimeline<T> {
    return {
      id: timeline.id,
      name: timeline.name,
      points: timeline.snapshots.map((s) => this.toUnifiedPoint(s)),
      keyframeIndices: timeline.keyframeIndices,
      bounds: timeline.bounds,
      currentTime: timeline.currentTime,
      playback: {
        state: timeline.playback.isPlaying ? "playing" : "paused",
        speed: timeline.playback.speed ?? 1.0,
        direction: timeline.playback.direction ?? "forward",
        loop: timeline.playback.loop ?? false,
        loopRange: timeline.playback.loopRange,
        stepping: false,
      },
      branches: [],
      activeBranch: "main",
    };
  }
}

// ============================================================================
// Universal Adapter Registry
// ============================================================================

/**
 * Registry of all adapters for easy access
 */
export const Adapters = {
  Forge: ForgeAdapter,
  Twin: TwinAdapter,
  Foundry: FoundryAdapter,
  Temporal: TemporalAdapter,
};

/**
 * Convert any domain entity to unified entity
 */
export function toUnifiedEntity(
  entity: unknown,
  domain: Domain
): UnifiedEntity {
  switch (domain) {
    case "forge":
      return ForgeAdapter.toUnifiedEntity(entity as AgentComponent);
    case "twin":
      return TwinAdapter.toUnifiedEntity(entity as TwinState);
    case "foundry":
      return FoundryAdapter.toUnifiedEntity(entity as ModelLayer);
    default:
      throw new Error(`Unknown domain: ${domain}`);
  }
}

/**
 * Convert unified entity to domain-specific type
 */
export function fromUnifiedEntity<T>(entity: UnifiedEntity, domain: Domain): T {
  switch (domain) {
    case "forge":
      return ForgeAdapter.fromUnifiedEntity(entity) as T;
    case "twin":
      return TwinAdapter.fromUnifiedEntity(entity) as T;
    case "foundry":
      return FoundryAdapter.fromUnifiedEntity(entity) as T;
    default:
      throw new Error(`Unknown domain: ${domain}`);
  }
}
