/**
 * Cross-Domain Isomorphisms
 *
 * Mathematical mappings between equivalent concepts across domains.
 * These isomorphisms enable the recognition of shared patterns and
 * enable features to work seamlessly across Forge, Twin, and Foundry.
 *
 * ISOMORPHISM THEORY:
 * An isomorphism f: A → B is a bijective mapping where both f and f⁻¹
 * preserve structure. In our context, this means operations in domain A
 * have equivalent operations in domain B that produce equivalent results.
 *
 * @module @neurectomy/3d-engine/cross-domain/isomorphisms
 * @agents @AXIOM @NEXUS @NEURAL
 */

import type {
  UnifiedEntity,
  UnifiedGraph,
  UnifiedEdge,
  UnifiedTemporalPoint,
  UnifiedTimeline,
  UnifiedEvent,
  Domain,
  EventType,
  UniversalId,
  Timestamp,
} from "./types";

import { Adapters, toUnifiedEntity, fromUnifiedEntity } from "./adapters";

// ============================================================================
// Isomorphism Interface
// ============================================================================

/**
 * Generic isomorphism between two types
 */
export interface Isomorphism<A, B> {
  /** Forward mapping: A → B */
  forward(a: A): B;

  /** Inverse mapping: B → A */
  inverse(b: B): A;

  /** Check if mapping preserves structure */
  preservesStructure(a: A, b: B): boolean;

  /** Get the source domain */
  sourceDomain: Domain;

  /** Get the target domain */
  targetDomain: Domain;
}

/**
 * Bidirectional isomorphism with composition support
 */
export abstract class BidirectionalIsomorphism<A, B> implements Isomorphism<
  A,
  B
> {
  abstract sourceDomain: Domain;
  abstract targetDomain: Domain;

  abstract forward(a: A): B;
  abstract inverse(b: B): A;

  /**
   * Default structure preservation check
   */
  preservesStructure(a: A, b: B): boolean {
    const roundTrip = this.inverse(this.forward(a));
    return this.isEquivalent(a, roundTrip);
  }

  /**
   * Check equivalence (override for custom equality)
   */
  protected isEquivalent(a1: A, a2: A): boolean {
    return JSON.stringify(a1) === JSON.stringify(a2);
  }

  /**
   * Compose with another isomorphism: (A → B) ∘ (B → C) = (A → C)
   */
  compose<C>(other: Isomorphism<B, C>): ComposedIsomorphism<A, B, C> {
    return new ComposedIsomorphism(this, other);
  }

  /**
   * Get the inverse isomorphism
   */
  invert(): InverseIsomorphism<A, B> {
    return new InverseIsomorphism(this);
  }
}

/**
 * Composed isomorphism: f ∘ g
 */
class ComposedIsomorphism<A, B, C> extends BidirectionalIsomorphism<A, C> {
  sourceDomain: Domain;
  targetDomain: Domain;

  constructor(
    private first: Isomorphism<A, B>,
    private second: Isomorphism<B, C>
  ) {
    super();
    this.sourceDomain = first.sourceDomain;
    this.targetDomain = second.targetDomain;
  }

  forward(a: A): C {
    return this.second.forward(this.first.forward(a));
  }

  inverse(c: C): A {
    return this.first.inverse(this.second.inverse(c));
  }
}

/**
 * Inverse isomorphism: f⁻¹
 */
class InverseIsomorphism<A, B> extends BidirectionalIsomorphism<B, A> {
  sourceDomain: Domain;
  targetDomain: Domain;

  constructor(private original: Isomorphism<A, B>) {
    super();
    this.sourceDomain = original.targetDomain;
    this.targetDomain = original.sourceDomain;
  }

  forward(b: B): A {
    return this.original.inverse(b);
  }

  inverse(a: A): B {
    return this.original.forward(a);
  }
}

// ============================================================================
// Entity Isomorphisms
// ============================================================================

/**
 * AgentComponent ↔ TwinState isomorphism
 */
export class ComponentTwinIsomorphism extends BidirectionalIsomorphism<
  unknown,
  unknown
> {
  sourceDomain: Domain = "forge";
  targetDomain: Domain = "twin";

  forward(component: unknown): unknown {
    const unified = toUnifiedEntity(component, "forge");
    return fromUnifiedEntity(unified, "twin");
  }

  inverse(twin: unknown): unknown {
    const unified = toUnifiedEntity(twin, "twin");
    return fromUnifiedEntity(unified, "forge");
  }
}

/**
 * TwinState ↔ ModelLayer isomorphism
 */
export class TwinFoundryIsomorphism extends BidirectionalIsomorphism<
  unknown,
  unknown
> {
  sourceDomain: Domain = "twin";
  targetDomain: Domain = "foundry";

  forward(twin: unknown): unknown {
    const unified = toUnifiedEntity(twin, "twin");
    return fromUnifiedEntity(unified, "foundry");
  }

  inverse(layer: unknown): unknown {
    const unified = toUnifiedEntity(layer, "foundry");
    return fromUnifiedEntity(unified, "twin");
  }
}

/**
 * AgentComponent ↔ ModelLayer isomorphism (composed)
 */
export class ComponentFoundryIsomorphism extends BidirectionalIsomorphism<
  unknown,
  unknown
> {
  sourceDomain: Domain = "forge";
  targetDomain: Domain = "foundry";

  private componentTwin = new ComponentTwinIsomorphism();
  private twinFoundry = new TwinFoundryIsomorphism();

  forward(component: unknown): unknown {
    return this.twinFoundry.forward(this.componentTwin.forward(component));
  }

  inverse(layer: unknown): unknown {
    return this.componentTwin.inverse(this.twinFoundry.inverse(layer));
  }
}

// ============================================================================
// Graph Isomorphisms
// ============================================================================

/**
 * Check if two graphs are isomorphic
 */
export function areGraphsIsomorphic(
  g1: UnifiedGraph,
  g2: UnifiedGraph
): boolean {
  // Quick checks
  if (g1.nodes.size !== g2.nodes.size) return false;
  if (g1.edges.size !== g2.edges.size) return false;

  // Check structural equivalence via adjacency
  const adj1 = buildAdjacencySignature(g1);
  const adj2 = buildAdjacencySignature(g2);

  return adj1 === adj2;
}

/**
 * Build adjacency signature for isomorphism checking
 */
function buildAdjacencySignature(graph: UnifiedGraph): string {
  const degrees: number[] = [];

  for (const nodeId of graph.nodes.keys()) {
    let inDegree = 0;
    let outDegree = 0;

    for (const edge of graph.edges.values()) {
      if (edge.targetId === nodeId) inDegree++;
      if (edge.sourceId === nodeId) outDegree++;
    }

    degrees.push(inDegree * 1000 + outDegree);
  }

  return degrees.sort((a, b) => a - b).join(",");
}

/**
 * Find node mappings between isomorphic graphs
 */
export function findGraphMapping(
  source: UnifiedGraph,
  target: UnifiedGraph
): Map<UniversalId, UniversalId> | null {
  if (!areGraphsIsomorphic(source, target)) return null;

  const mapping = new Map<UniversalId, UniversalId>();

  // Build degree-based buckets
  const sourceBuckets = buildDegreeBuckets(source);
  const targetBuckets = buildDegreeBuckets(target);

  // Match nodes with same degree signature
  for (const [signature, sourceNodes] of sourceBuckets) {
    const targetNodes = targetBuckets.get(signature);
    if (!targetNodes || targetNodes.length !== sourceNodes.length) {
      return null; // Not isomorphic after all
    }

    // For now, simple 1:1 mapping (in practice, need backtracking)
    for (let i = 0; i < sourceNodes.length; i++) {
      mapping.set(sourceNodes[i], targetNodes[i]);
    }
  }

  return mapping;
}

function buildDegreeBuckets(graph: UnifiedGraph): Map<string, UniversalId[]> {
  const buckets = new Map<string, UniversalId[]>();

  for (const nodeId of graph.nodes.keys()) {
    let inDegree = 0;
    let outDegree = 0;

    for (const edge of graph.edges.values()) {
      if (edge.targetId === nodeId) inDegree++;
      if (edge.sourceId === nodeId) outDegree++;
    }

    const signature = `${inDegree}:${outDegree}`;
    if (!buckets.has(signature)) {
      buckets.set(signature, []);
    }
    buckets.get(signature)!.push(nodeId);
  }

  return buckets;
}

// ============================================================================
// Temporal Isomorphisms
// ============================================================================

/**
 * TimelinePoint ↔ StateSnapshot ↔ Checkpoint isomorphism
 */
export class TemporalIsomorphism extends BidirectionalIsomorphism<
  UnifiedTemporalPoint,
  UnifiedTemporalPoint
> {
  sourceDomain: Domain;
  targetDomain: Domain;

  constructor(source: Domain, target: Domain) {
    super();
    this.sourceDomain = source;
    this.targetDomain = target;
  }

  forward(point: UnifiedTemporalPoint): UnifiedTemporalPoint {
    // Temporal points are already unified, just update domain metadata
    return {
      ...point,
      metadata: {
        ...point.metadata,
        tags: [
          ...point.metadata.tags,
          `from:${this.sourceDomain}`,
          `to:${this.targetDomain}`,
        ],
      },
    };
  }

  inverse(point: UnifiedTemporalPoint): UnifiedTemporalPoint {
    return {
      ...point,
      metadata: {
        ...point.metadata,
        tags: point.metadata.tags.filter(
          (t) => !t.startsWith("from:") && !t.startsWith("to:")
        ),
      },
    };
  }
}

/**
 * Map timeline from one domain to another
 */
export function mapTimeline(
  timeline: UnifiedTimeline,
  targetDomain: Domain
): UnifiedTimeline {
  const sourceDomain = inferTimelineDomain(timeline);
  const iso = new TemporalIsomorphism(sourceDomain, targetDomain);

  return {
    ...timeline,
    points: timeline.points.map((p) => iso.forward(p)),
  };
}

function inferTimelineDomain(timeline: UnifiedTimeline): Domain {
  const point = timeline.points[0];
  if (!point) return "twin";

  if (point.metadata.forgeData) return "forge";
  if (point.metadata.twinData) return "twin";
  if (point.metadata.foundryData) return "foundry";

  return "twin";
}

// ============================================================================
// Event Isomorphisms
// ============================================================================

/**
 * Event type mappings between domains
 */
const EVENT_TYPE_ISOMORPHISMS: Record<
  EventType,
  Partial<Record<Domain, EventType>>
> = {
  // Forge → Twin/Foundry
  "component:created": {
    twin: "state:changed",
    foundry: "architecture:changed",
  },
  "component:updated": {
    twin: "state:changed",
    foundry: "architecture:changed",
  },
  "component:deleted": {
    twin: "state:changed",
    foundry: "architecture:changed",
  },
  "component:selected": { twin: "state:changed" },
  "component:moved": { twin: "state:changed" },
  "connection:created": {
    twin: "state:changed",
    foundry: "architecture:changed",
  },
  "connection:deleted": {
    twin: "state:changed",
    foundry: "architecture:changed",
  },
  "timeline:seek": { twin: "state:changed" },
  "timeline:play": { twin: "prediction:started" },
  "timeline:pause": { twin: "prediction:completed" },

  // Twin → Forge/Foundry
  "state:changed": {
    forge: "component:updated",
    foundry: "architecture:changed",
  },
  "state:synced": { forge: "component:updated" },
  "state:diverged": { forge: "component:updated", foundry: "training:step" },
  "prediction:started": { forge: "timeline:play" },
  "prediction:completed": {
    forge: "timeline:pause",
    foundry: "training:completed",
  },
  "scenario:created": { forge: "component:created" },
  "scenario:evaluated": {
    forge: "component:updated",
    foundry: "training:step",
  },

  // Foundry → Forge/Twin
  "training:started": { forge: "timeline:play", twin: "prediction:started" },
  "training:step": { forge: "component:updated", twin: "state:changed" },
  "training:epoch": { forge: "component:updated", twin: "state:changed" },
  "training:completed": {
    forge: "timeline:pause",
    twin: "prediction:completed",
  },
  "training:failed": { forge: "component:updated", twin: "state:diverged" },
  "checkpoint:saved": { forge: "component:created", twin: "state:synced" },
  "checkpoint:loaded": { forge: "component:updated", twin: "state:synced" },
  "architecture:changed": { forge: "component:updated", twin: "state:changed" },

  // Universal (identity mappings)
  "entity:created": {},
  "entity:updated": {},
  "entity:deleted": {},
  "graph:modified": {},
  "metrics:updated": {},
};

/**
 * Map event type to equivalent in target domain
 */
export function mapEventType(
  sourceType: EventType,
  targetDomain: Domain
): EventType | null {
  const mappings = EVENT_TYPE_ISOMORPHISMS[sourceType];
  return mappings?.[targetDomain] ?? null;
}

/**
 * Transform event for target domain
 */
export function transformEvent<T>(
  event: UnifiedEvent<T>,
  targetDomain: Domain
): UnifiedEvent<T> | null {
  if (event.sourceDomain === targetDomain) {
    return event; // No transformation needed
  }

  const targetType = mapEventType(event.type, targetDomain);
  if (!targetType) {
    return null; // No valid mapping
  }

  return {
    ...event,
    id: `${event.id}-${targetDomain}`,
    type: targetType,
    targetDomains: [targetDomain],
    correlationId: event.correlationId ?? event.id,
  };
}

// ============================================================================
// Pattern Recognition
// ============================================================================

/**
 * Recognized cross-domain pattern
 */
export interface CrossDomainPattern {
  id: string;
  name: string;
  description: string;
  forgePattern?: string;
  twinPattern?: string;
  foundryPattern?: string;
  applications: string[];
}

/**
 * Known cross-domain patterns
 */
export const CROSS_DOMAIN_PATTERNS: CrossDomainPattern[] = [
  {
    id: "sequential-pipeline",
    name: "Sequential Pipeline",
    description: "Linear sequence of processing stages",
    forgePattern: "Linear component chain",
    twinPattern: "Sequential state transitions",
    foundryPattern: "Feed-forward network",
    applications: ["Data processing", "Inference pipeline", "ETL"],
  },
  {
    id: "parallel-branches",
    name: "Parallel Branches",
    description: "Multiple parallel processing paths",
    forgePattern: "Branching component graph",
    twinPattern: "Scenario branches",
    foundryPattern: "Multi-head attention / Ensemble",
    applications: ["A/B testing", "Model ensemble", "Parallel computation"],
  },
  {
    id: "feedback-loop",
    name: "Feedback Loop",
    description: "Output feeds back to input",
    forgePattern: "Cyclic connection",
    twinPattern: "Iterative refinement",
    foundryPattern: "Recurrent connection",
    applications: ["Iterative optimization", "RNN", "Control systems"],
  },
  {
    id: "aggregation",
    name: "Aggregation",
    description: "Multiple inputs combine to single output",
    forgePattern: "Merge component",
    twinPattern: "State aggregation",
    foundryPattern: "Pooling / Reduce layer",
    applications: ["Feature aggregation", "Consensus", "Attention"],
  },
  {
    id: "broadcast",
    name: "Broadcast",
    description: "Single input fans out to multiple outputs",
    forgePattern: "Split component",
    twinPattern: "State broadcast",
    foundryPattern: "Skip connections / Residual",
    applications: ["Data distribution", "Skip connections", "Multicast"],
  },
  {
    id: "hierarchy",
    name: "Hierarchical Structure",
    description: "Tree-like parent-child relationships",
    forgePattern: "Nested components",
    twinPattern: "Hierarchical state",
    foundryPattern: "Encoder-decoder / U-Net",
    applications: ["Nested agents", "Feature hierarchy", "Compression"],
  },
];

/**
 * Detect patterns in a unified graph
 */
export function detectPatterns(graph: UnifiedGraph): CrossDomainPattern[] {
  const detected: CrossDomainPattern[] = [];

  // Check for sequential pipeline
  if (isSequential(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[0]);
  }

  // Check for parallel branches
  if (hasParallelBranches(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[1]);
  }

  // Check for feedback loops
  if (hasCycles(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[2]);
  }

  // Check for aggregation
  if (hasAggregation(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[3]);
  }

  // Check for broadcast
  if (hasBroadcast(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[4]);
  }

  // Check for hierarchy
  if (isHierarchical(graph)) {
    detected.push(CROSS_DOMAIN_PATTERNS[5]);
  }

  return detected;
}

function isSequential(graph: UnifiedGraph): boolean {
  // Check if graph is a single chain
  let maxInDegree = 0;
  let maxOutDegree = 0;

  for (const nodeId of graph.nodes.keys()) {
    let inDegree = 0;
    let outDegree = 0;

    for (const edge of graph.edges.values()) {
      if (edge.targetId === nodeId) inDegree++;
      if (edge.sourceId === nodeId) outDegree++;
    }

    maxInDegree = Math.max(maxInDegree, inDegree);
    maxOutDegree = Math.max(maxOutDegree, outDegree);
  }

  return maxInDegree <= 1 && maxOutDegree <= 1;
}

function hasParallelBranches(graph: UnifiedGraph): boolean {
  for (const nodeId of graph.nodes.keys()) {
    let outDegree = 0;
    for (const edge of graph.edges.values()) {
      if (edge.sourceId === nodeId) outDegree++;
    }
    if (outDegree > 1) return true;
  }
  return false;
}

function hasCycles(graph: UnifiedGraph): boolean {
  const visited = new Set<UniversalId>();
  const stack = new Set<UniversalId>();

  const adjList = new Map<UniversalId, UniversalId[]>();
  for (const edge of graph.edges.values()) {
    if (!adjList.has(edge.sourceId)) {
      adjList.set(edge.sourceId, []);
    }
    adjList.get(edge.sourceId)!.push(edge.targetId);
  }

  function dfs(nodeId: UniversalId): boolean {
    visited.add(nodeId);
    stack.add(nodeId);

    for (const neighbor of adjList.get(nodeId) ?? []) {
      if (!visited.has(neighbor)) {
        if (dfs(neighbor)) return true;
      } else if (stack.has(neighbor)) {
        return true; // Back edge found
      }
    }

    stack.delete(nodeId);
    return false;
  }

  for (const nodeId of graph.nodes.keys()) {
    if (!visited.has(nodeId)) {
      if (dfs(nodeId)) return true;
    }
  }

  return false;
}

function hasAggregation(graph: UnifiedGraph): boolean {
  for (const nodeId of graph.nodes.keys()) {
    let inDegree = 0;
    for (const edge of graph.edges.values()) {
      if (edge.targetId === nodeId) inDegree++;
    }
    if (inDegree > 1) return true;
  }
  return false;
}

function hasBroadcast(graph: UnifiedGraph): boolean {
  return hasParallelBranches(graph); // Same condition
}

function isHierarchical(graph: UnifiedGraph): boolean {
  // Check if any nodes have parentId set
  for (const node of graph.nodes.values()) {
    if (node.parentId) return true;
  }
  return false;
}

// ============================================================================
// Isomorphism Registry
// ============================================================================

/**
 * Registry of all isomorphisms
 */
export const IsomorphismRegistry = {
  // Entity isomorphisms
  componentToTwin: new ComponentTwinIsomorphism(),
  twinToFoundry: new TwinFoundryIsomorphism(),
  componentToFoundry: new ComponentFoundryIsomorphism(),

  // Temporal isomorphisms
  forgeToTwinTemporal: new TemporalIsomorphism("forge", "twin"),
  twinToFoundryTemporal: new TemporalIsomorphism("twin", "foundry"),
  forgeToFoundryTemporal: new TemporalIsomorphism("forge", "foundry"),

  // Utility functions
  areGraphsIsomorphic,
  findGraphMapping,
  mapTimeline,
  mapEventType,
  transformEvent,
  detectPatterns,
};

export default IsomorphismRegistry;
