/**
 * Causal Training Debugger - P0 Breakthrough Innovation
 *
 * Trace training decisions through twin state history.
 * Visualize cause-effect chains in 3D - understand WHY
 * a model behaves the way it does by traversing the
 * causal graph of training decisions.
 *
 * "Every behavior has a cause. Find it."
 *
 * @module CausalTrainingDebugger
 * @category CrossDomain/P0-Breakthrough
 */

import { EventEmitter } from "events";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * 3D Position
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Causal Node - represents an event/state in the training process
 */
export interface CausalNode {
  id: string;
  nodeType: CausalNodeType;

  // Temporal
  timestamp: number;
  epoch: number;
  batch: number;

  // Content
  label: string;
  description: string;
  data: Record<string, unknown>;

  // Causal relationships
  causes: string[]; // Node IDs that caused this
  effects: string[]; // Node IDs caused by this

  // Attribution
  causalStrength: Map<string, number>; // How much each cause contributed
  counterfactuals: Counterfactual[];

  // 3D visualization
  position: Vector3D;
  color: string;
  size: number;
  importance: number;
}

/**
 * Causal node types
 */
export type CausalNodeType =
  | "input" // Training input
  | "forward_pass" // Forward propagation
  | "loss_computation" // Loss calculated
  | "backward_pass" // Backward propagation
  | "gradient" // Gradient computed
  | "weight_update" // Weights changed
  | "hyperparameter" // Hyperparameter set/changed
  | "regularization" // Regularization applied
  | "batch_norm" // Batch normalization
  | "dropout" // Dropout applied
  | "learning_rate" // Learning rate (schedule)
  | "optimizer_state" // Optimizer state (momentum, etc.)
  | "checkpoint" // Model checkpoint
  | "evaluation" // Evaluation/validation
  | "decision" // Training decision (early stop, etc.)
  | "twin_state" // Digital twin state
  | "prediction"; // Model prediction

/**
 * Counterfactual - what would have happened if...
 */
export interface Counterfactual {
  id: string;
  nodeId: string;

  // What was changed
  intervention: Intervention;

  // What would have been different
  predictedOutcome: Record<string, unknown>;
  confidenceScore: number;

  // Comparison
  actualVsPredicted: {
    metric: string;
    actual: number;
    predicted: number;
    difference: number;
  }[];
}

/**
 * Intervention - change to test
 */
export interface Intervention {
  targetProperty: string;
  originalValue: unknown;
  interventionValue: unknown;
  description: string;
}

/**
 * Causal Edge - connection between nodes
 */
export interface CausalEdge {
  id: string;
  sourceId: string;
  targetId: string;

  // Causal properties
  edgeType: CausalEdgeType;
  strength: number; // 0-1, how strong the causal link is
  timeDelay: number; // Time between cause and effect

  // Attribution
  mechanism: string; // How cause leads to effect
  confidence: number;

  // Visualization
  color: string;
  width: number;
  animated: boolean;
}

/**
 * Edge types
 */
export type CausalEdgeType =
  | "direct" // Direct causation (A causes B)
  | "indirect" // Indirect (A -> ... -> B)
  | "confounder" // Common cause
  | "mediator" // Mediating variable
  | "inhibitor" // A prevents B
  | "amplifier"; // A amplifies B

/**
 * Causal Path - chain of cause-effect relationships
 */
export interface CausalPath {
  id: string;
  nodes: string[]; // Ordered node IDs
  edges: string[]; // Ordered edge IDs

  // Path properties
  totalStrength: number; // Product of edge strengths
  totalDelay: number; // Sum of time delays
  pathType: CausalPathType;

  // Analysis
  description: string;
  keyInsights: string[];
}

/**
 * Path types
 */
export type CausalPathType =
  | "forward" // Causes to effects
  | "backward" // Effects to causes
  | "bidirectional"; // Feedback loop

/**
 * Training Session - container for causal graph
 */
export interface TrainingSession {
  id: string;
  name: string;
  modelName: string;

  // Graph structure
  nodes: Map<string, CausalNode>;
  edges: Map<string, CausalEdge>;
  paths: Map<string, CausalPath>;

  // Twin integration
  twinStates: Map<number, TwinState>; // Timestamp -> state

  // Timeline
  startTime: number;
  endTime: number | null;
  currentTime: number;

  // Metrics
  metrics: TrainingMetrics;

  // Investigation state
  selectedNode: string | null;
  investigatedPaths: string[];
  bookmarks: Bookmark[];
}

/**
 * Twin state snapshot
 */
export interface TwinState {
  timestamp: number;
  epoch: number;

  // Model state
  weights: number[];
  gradients: number[];
  activations: number[];

  // Performance
  loss: number;
  accuracy: number;

  // Environment
  inputDistribution: number[];
  outputDistribution: number[];
}

/**
 * Training metrics
 */
export interface TrainingMetrics {
  totalEpochs: number;
  totalBatches: number;
  totalNodes: number;
  totalEdges: number;
  averageCausalStrength: number;
  longestPath: number;
  rootCauses: string[];
  terminalEffects: string[];
}

/**
 * Bookmark for interesting findings
 */
export interface Bookmark {
  id: string;
  nodeId: string;
  label: string;
  notes: string;
  timestamp: number;
}

/**
 * Investigation query
 */
export interface InvestigationQuery {
  // What to find
  queryType: QueryType;

  // Parameters
  targetNode?: string;
  targetMetric?: string;
  targetTimeRange?: [number, number];

  // Filters
  minCausalStrength?: number;
  maxPathLength?: number;
  nodeTypes?: CausalNodeType[];

  // Options
  includeCounterfactuals?: boolean;
  includeIndirect?: boolean;
}

/**
 * Query types
 */
export type QueryType =
  | "why_did_this_happen" // Find causes of a node
  | "what_did_this_cause" // Find effects of a node
  | "critical_path" // Find most important path
  | "root_cause" // Find original cause
  | "common_cause" // Find common ancestors
  | "counterfactual" // What if analysis
  | "attribution" // Attribute outcome to causes
  | "divergence"; // When did training diverge

/**
 * Investigation result
 */
export interface InvestigationResult {
  queryId: string;
  queryType: QueryType;

  // Found items
  nodes: CausalNode[];
  edges: CausalEdge[];
  paths: CausalPath[];

  // Analysis
  summary: string;
  insights: string[];
  recommendations: string[];

  // Confidence
  confidence: number;
  limitations: string[];
}

/**
 * Debugger configuration
 */
export interface DebuggerConfig {
  // Graph layout
  layoutAlgorithm: "force" | "hierarchical" | "temporal" | "radial";
  nodeSpacing: number;
  timeScale: number;

  // Causal analysis
  minCausalThreshold: number;
  maxPathDepth: number;
  counterfactualSamples: number;

  // Visualization
  showWeakLinks: boolean;
  animateFlow: boolean;
  highlightCriticalPath: boolean;

  // Performance
  maxNodesDisplayed: number;
  aggregateSmallNodes: boolean;
}

/**
 * Debugger events
 */
export interface DebuggerEvents {
  "node:added": { node: CausalNode };
  "node:selected": { nodeId: string };
  "node:investigated": { nodeId: string; result: InvestigationResult };

  "edge:added": { edge: CausalEdge };
  "edge:highlighted": { edgeId: string };

  "path:discovered": { path: CausalPath };
  "path:highlighted": { pathId: string };

  "investigation:started": { query: InvestigationQuery };
  "investigation:completed": { result: InvestigationResult };

  "bookmark:added": { bookmark: Bookmark };

  "timeline:moved": { newTime: number };

  "insight:generated": { insight: string };
}

// ============================================================================
// Causal Training Debugger Implementation
// ============================================================================

/**
 * Causal Training Debugger
 *
 * Trace through training history to understand cause-effect
 * relationships. Find out WHY your model behaves the way it does.
 */
export class CausalTrainingDebugger extends EventEmitter {
  private config: DebuggerConfig;
  private sessions: Map<string, TrainingSession>;
  private activeSession: string | null;

  // Visualization state
  private cameraPosition: Vector3D;
  private cameraTarget: Vector3D;
  private highlightedNodes: Set<string>;
  private highlightedEdges: Set<string>;

  constructor(config: Partial<DebuggerConfig> = {}) {
    super();

    this.config = this.mergeConfig(config);
    this.sessions = new Map();
    this.activeSession = null;

    this.cameraPosition = { x: 0, y: 0, z: 100 };
    this.cameraTarget = { x: 0, y: 0, z: 0 };
    this.highlightedNodes = new Set();
    this.highlightedEdges = new Set();
  }

  /**
   * Merge config with defaults
   */
  private mergeConfig(config: Partial<DebuggerConfig>): DebuggerConfig {
    return {
      layoutAlgorithm: config.layoutAlgorithm ?? "temporal",
      nodeSpacing: config.nodeSpacing ?? 20,
      timeScale: config.timeScale ?? 1,
      minCausalThreshold: config.minCausalThreshold ?? 0.1,
      maxPathDepth: config.maxPathDepth ?? 20,
      counterfactualSamples: config.counterfactualSamples ?? 100,
      showWeakLinks: config.showWeakLinks ?? false,
      animateFlow: config.animateFlow ?? true,
      highlightCriticalPath: config.highlightCriticalPath ?? true,
      maxNodesDisplayed: config.maxNodesDisplayed ?? 500,
      aggregateSmallNodes: config.aggregateSmallNodes ?? true,
    };
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  /**
   * Create a new debugging session
   */
  createSession(name: string, modelName: string): TrainingSession {
    const id = this.generateId("session");

    const session: TrainingSession = {
      id,
      name,
      modelName,
      nodes: new Map(),
      edges: new Map(),
      paths: new Map(),
      twinStates: new Map(),
      startTime: Date.now(),
      endTime: null,
      currentTime: Date.now(),
      metrics: {
        totalEpochs: 0,
        totalBatches: 0,
        totalNodes: 0,
        totalEdges: 0,
        averageCausalStrength: 0,
        longestPath: 0,
        rootCauses: [],
        terminalEffects: [],
      },
      selectedNode: null,
      investigatedPaths: [],
      bookmarks: [],
    };

    this.sessions.set(id, session);
    this.activeSession = id;

    return session;
  }

  /**
   * Get active session
   */
  getActiveSession(): TrainingSession | null {
    if (!this.activeSession) return null;
    return this.sessions.get(this.activeSession) ?? null;
  }

  /**
   * Set active session
   */
  setActiveSession(sessionId: string): void {
    if (this.sessions.has(sessionId)) {
      this.activeSession = sessionId;
    }
  }

  /**
   * End session
   */
  endSession(sessionId?: string): void {
    const id = sessionId ?? this.activeSession;
    if (!id) return;

    const session = this.sessions.get(id);
    if (session) {
      session.endTime = Date.now();
      this.updateMetrics(session);
    }
  }

  // ============================================================================
  // Node Management
  // ============================================================================

  /**
   * Record a causal node
   */
  recordNode(
    nodeType: CausalNodeType,
    label: string,
    data: Record<string, unknown>,
    causes: string[] = [],
    epoch?: number,
    batch?: number
  ): CausalNode {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const id = this.generateId("node");

    const node: CausalNode = {
      id,
      nodeType,
      timestamp: Date.now(),
      epoch: epoch ?? session.metrics.totalEpochs,
      batch: batch ?? session.metrics.totalBatches,
      label,
      description: this.generateDescription(nodeType, label, data),
      data,
      causes,
      effects: [],
      causalStrength: new Map(),
      counterfactuals: [],
      position: this.calculateNodePosition(session, nodeType, epoch),
      color: this.getNodeColor(nodeType),
      size: this.getNodeSize(nodeType),
      importance: 0.5,
    };

    session.nodes.set(id, node);

    // Create edges from causes
    for (const causeId of causes) {
      const cause = session.nodes.get(causeId);
      if (cause) {
        cause.effects.push(id);
        this.createEdge(session, causeId, id);
      }
    }

    // Update metrics
    session.metrics.totalNodes++;

    this.emit("node:added", { node });

    return node;
  }

  /**
   * Generate description for node
   */
  private generateDescription(
    nodeType: CausalNodeType,
    label: string,
    data: Record<string, unknown>
  ): string {
    const templates: Record<
      CausalNodeType,
      (l: string, d: Record<string, unknown>) => string
    > = {
      input: (l) => `Training input batch: ${l}`,
      forward_pass: (l) => `Forward propagation through ${l}`,
      loss_computation: (l, d) => `Loss computed: ${d["loss"] ?? "unknown"}`,
      backward_pass: (l) => `Backward propagation through ${l}`,
      gradient: (l, d) => `Gradient for ${l}: norm=${d["norm"] ?? "unknown"}`,
      weight_update: (l, d) => `Weight update: Δ=${d["delta"] ?? "unknown"}`,
      hyperparameter: (l, d) =>
        `Hyperparameter ${l} set to ${d["value"] ?? "unknown"}`,
      regularization: (l, d) =>
        `Regularization ${l}: strength=${d["strength"] ?? "unknown"}`,
      batch_norm: (l) => `Batch normalization: ${l}`,
      dropout: (l, d) => `Dropout applied: rate=${d["rate"] ?? 0.5}`,
      learning_rate: (l, d) => `Learning rate: ${d["value"] ?? "unknown"}`,
      optimizer_state: (l) => `Optimizer state updated: ${l}`,
      checkpoint: (l) => `Model checkpoint saved: ${l}`,
      evaluation: (l, d) =>
        `Evaluation: ${l}, acc=${d["accuracy"] ?? "unknown"}`,
      decision: (l) => `Training decision: ${l}`,
      twin_state: (l, d) => `Digital twin state: ${d["summary"] ?? l}`,
      prediction: (l, d) => `Model prediction: ${d["output"] ?? "unknown"}`,
    };

    return templates[nodeType](label, data);
  }

  /**
   * Calculate position for node in 3D space
   */
  private calculateNodePosition(
    session: TrainingSession,
    nodeType: CausalNodeType,
    epoch?: number
  ): Vector3D {
    const typeOffsets: Record<CausalNodeType, number> = {
      input: 0,
      forward_pass: 1,
      loss_computation: 2,
      backward_pass: 3,
      gradient: 4,
      weight_update: 5,
      hyperparameter: -1,
      regularization: 2.5,
      batch_norm: 1.5,
      dropout: 1.5,
      learning_rate: -1,
      optimizer_state: 5.5,
      checkpoint: 6,
      evaluation: 7,
      decision: 8,
      twin_state: -2,
      prediction: 7,
    };

    switch (this.config.layoutAlgorithm) {
      case "temporal": {
        const time = (epoch ?? 0) * this.config.timeScale;
        const y = typeOffsets[nodeType] * this.config.nodeSpacing;
        const z = (Math.random() - 0.5) * 10;
        return { x: time, y, z };
      }

      case "hierarchical": {
        const layer = typeOffsets[nodeType];
        const count = Array.from(session.nodes.values()).filter(
          (n) => n.nodeType === nodeType
        ).length;
        return {
          x: (count % 10) * this.config.nodeSpacing,
          y: layer * this.config.nodeSpacing,
          z: Math.floor(count / 10) * this.config.nodeSpacing,
        };
      }

      case "radial": {
        const angle = (session.nodes.size * 137.5 * Math.PI) / 180; // Golden angle
        const radius = 10 + Math.sqrt(session.nodes.size) * 5;
        const height = typeOffsets[nodeType] * 10;
        return {
          x: Math.cos(angle) * radius,
          y: height,
          z: Math.sin(angle) * radius,
        };
      }

      default: // force
        return {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          z: (Math.random() - 0.5) * 100,
        };
    }
  }

  /**
   * Get color for node type
   */
  private getNodeColor(nodeType: CausalNodeType): string {
    const colors: Record<CausalNodeType, string> = {
      input: "#4caf50",
      forward_pass: "#2196f3",
      loss_computation: "#f44336",
      backward_pass: "#9c27b0",
      gradient: "#ff9800",
      weight_update: "#00bcd4",
      hyperparameter: "#607d8b",
      regularization: "#795548",
      batch_norm: "#9e9e9e",
      dropout: "#ffeb3b",
      learning_rate: "#e91e63",
      optimizer_state: "#3f51b5",
      checkpoint: "#8bc34a",
      evaluation: "#ffc107",
      decision: "#ff5722",
      twin_state: "#00bcd4",
      prediction: "#673ab7",
    };

    return colors[nodeType] || "#ffffff";
  }

  /**
   * Get size for node type
   */
  private getNodeSize(nodeType: CausalNodeType): number {
    const sizes: Record<CausalNodeType, number> = {
      input: 3,
      forward_pass: 4,
      loss_computation: 5,
      backward_pass: 4,
      gradient: 3,
      weight_update: 4,
      hyperparameter: 4,
      regularization: 3,
      batch_norm: 2,
      dropout: 2,
      learning_rate: 4,
      optimizer_state: 3,
      checkpoint: 5,
      evaluation: 5,
      decision: 6,
      twin_state: 5,
      prediction: 4,
    };

    return sizes[nodeType] || 3;
  }

  /**
   * Create edge between nodes
   */
  private createEdge(
    session: TrainingSession,
    sourceId: string,
    targetId: string,
    edgeType: CausalEdgeType = "direct"
  ): CausalEdge {
    const id = this.generateId("edge");

    const source = session.nodes.get(sourceId);
    const target = session.nodes.get(targetId);

    const timeDelay =
      source && target ? target.timestamp - source.timestamp : 0;

    const edge: CausalEdge = {
      id,
      sourceId,
      targetId,
      edgeType,
      strength: this.estimateCausalStrength(source, target),
      timeDelay,
      mechanism: this.inferMechanism(source, target),
      confidence: 0.8,
      color: this.getEdgeColor(edgeType),
      width: 1,
      animated: this.config.animateFlow,
    };

    session.edges.set(id, edge);
    session.metrics.totalEdges++;

    // Update node causal strength
    if (target) {
      target.causalStrength.set(sourceId, edge.strength);
    }

    this.emit("edge:added", { edge });

    return edge;
  }

  /**
   * Estimate causal strength between nodes
   */
  private estimateCausalStrength(
    source: CausalNode | undefined,
    target: CausalNode | undefined
  ): number {
    if (!source || !target) return 0.5;

    // Direct parent-child relationships are strong
    if (source.effects.includes(target.id)) return 0.9;

    // Same type nodes have moderate strength
    if (source.nodeType === target.nodeType) return 0.5;

    // Known strong relationships
    const strongPairs: [CausalNodeType, CausalNodeType][] = [
      ["forward_pass", "loss_computation"],
      ["loss_computation", "backward_pass"],
      ["backward_pass", "gradient"],
      ["gradient", "weight_update"],
      ["learning_rate", "weight_update"],
      ["hyperparameter", "decision"],
    ];

    for (const [s, t] of strongPairs) {
      if (source.nodeType === s && target.nodeType === t) return 0.85;
    }

    return 0.5;
  }

  /**
   * Infer mechanism of causation
   */
  private inferMechanism(
    source: CausalNode | undefined,
    target: CausalNode | undefined
  ): string {
    if (!source || !target) return "unknown";

    const key = `${source.nodeType}->${target.nodeType}`;

    const mechanisms: Record<string, string> = {
      "forward_pass->loss_computation": "prediction error calculation",
      "loss_computation->backward_pass": "gradient flow initialization",
      "backward_pass->gradient": "chain rule differentiation",
      "gradient->weight_update": "optimizer step",
      "learning_rate->weight_update": "step size modulation",
      "dropout->forward_pass": "regularization noise injection",
      "batch_norm->forward_pass": "activation normalization",
      "hyperparameter->decision": "training control",
      "twin_state->decision": "feedback-driven adaptation",
      "evaluation->decision": "performance-based decision",
    };

    return mechanisms[key] || "causal influence";
  }

  /**
   * Get edge color
   */
  private getEdgeColor(edgeType: CausalEdgeType): string {
    const colors: Record<CausalEdgeType, string> = {
      direct: "#ffffff",
      indirect: "#aaaaaa",
      confounder: "#ffaa00",
      mediator: "#00aaff",
      inhibitor: "#ff0000",
      amplifier: "#00ff00",
    };

    return colors[edgeType] || "#ffffff";
  }

  // ============================================================================
  // Investigation
  // ============================================================================

  /**
   * Investigate: why did this happen?
   */
  investigateWhy(nodeId: string, depth: number = 5): InvestigationResult {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const query: InvestigationQuery = {
      queryType: "why_did_this_happen",
      targetNode: nodeId,
      maxPathLength: depth,
    };

    this.emit("investigation:started", { query });

    const targetNode = session.nodes.get(nodeId);
    if (!targetNode) {
      return this.createEmptyResult(query, "Node not found");
    }

    // Find all causes
    const causes = this.findAllCauses(session, nodeId, depth);
    const edges = this.getConnectingEdges(
      session,
      causes.map((n) => n.id)
    );
    const paths = this.findCausalPaths(session, nodeId, "backward", depth);

    // Rank by importance
    const rankedCauses = this.rankByImportance(causes, targetNode);

    // Generate insights
    const insights = this.generateWhyInsights(targetNode, rankedCauses);

    const result: InvestigationResult = {
      queryId: this.generateId("query"),
      queryType: "why_did_this_happen",
      nodes: rankedCauses,
      edges,
      paths,
      summary: `Found ${causes.length} causes for "${targetNode.label}"`,
      insights,
      recommendations: this.generateRecommendations(targetNode, rankedCauses),
      confidence: this.calculateConfidence(rankedCauses),
      limitations: ["Indirect causes may be incomplete"],
    };

    this.emit("investigation:completed", { result });

    // Highlight in visualization
    this.highlightInvestigation(rankedCauses, edges);

    return result;
  }

  /**
   * Investigate: what did this cause?
   */
  investigateEffects(nodeId: string, depth: number = 5): InvestigationResult {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const query: InvestigationQuery = {
      queryType: "what_did_this_cause",
      targetNode: nodeId,
      maxPathLength: depth,
    };

    this.emit("investigation:started", { query });

    const sourceNode = session.nodes.get(nodeId);
    if (!sourceNode) {
      return this.createEmptyResult(query, "Node not found");
    }

    // Find all effects
    const effects = this.findAllEffects(session, nodeId, depth);
    const edges = this.getConnectingEdges(
      session,
      effects.map((n) => n.id)
    );
    const paths = this.findCausalPaths(session, nodeId, "forward", depth);

    // Generate insights
    const insights = this.generateEffectInsights(sourceNode, effects);

    const result: InvestigationResult = {
      queryId: this.generateId("query"),
      queryType: "what_did_this_cause",
      nodes: effects,
      edges,
      paths,
      summary: `"${sourceNode.label}" caused ${effects.length} downstream effects`,
      insights,
      recommendations: [],
      confidence: this.calculateConfidence(effects),
      limitations: ["Some indirect effects may be missed"],
    };

    this.emit("investigation:completed", { result });

    return result;
  }

  /**
   * Find root cause
   */
  findRootCause(nodeId: string): InvestigationResult {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const query: InvestigationQuery = {
      queryType: "root_cause",
      targetNode: nodeId,
    };

    this.emit("investigation:started", { query });

    const targetNode = session.nodes.get(nodeId);
    if (!targetNode) {
      return this.createEmptyResult(query, "Node not found");
    }

    // Trace back to root causes (nodes with no causes)
    const visited = new Set<string>();
    const rootCauses: CausalNode[] = [];

    const traverse = (currentId: string): void => {
      if (visited.has(currentId)) return;
      visited.add(currentId);

      const node = session.nodes.get(currentId);
      if (!node) return;

      if (node.causes.length === 0) {
        rootCauses.push(node);
      } else {
        for (const causeId of node.causes) {
          traverse(causeId);
        }
      }
    };

    traverse(nodeId);

    // Build paths from root causes to target
    const paths: CausalPath[] = [];
    for (const root of rootCauses) {
      const path = this.buildPath(session, root.id, nodeId);
      if (path) paths.push(path);
    }

    // Sort paths by strength
    paths.sort((a, b) => b.totalStrength - a.totalStrength);

    const result: InvestigationResult = {
      queryId: this.generateId("query"),
      queryType: "root_cause",
      nodes: rootCauses,
      edges: this.getEdgesInPaths(session, paths),
      paths,
      summary: `Found ${rootCauses.length} root causes`,
      insights: rootCauses.map(
        (r) => `Root cause: ${r.label} (${r.nodeType}) at epoch ${r.epoch}`
      ),
      recommendations: this.generateRootCauseRecommendations(rootCauses),
      confidence: paths.length > 0 ? paths[0].totalStrength : 0,
      limitations: [],
    };

    this.emit("investigation:completed", { result });

    return result;
  }

  /**
   * Counterfactual analysis
   */
  analyzeCounterfactual(
    nodeId: string,
    intervention: Intervention
  ): Counterfactual {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const node = session.nodes.get(nodeId);
    if (!node) throw new Error("Node not found");

    // Simulate intervention (simplified)
    const predictedOutcome = this.simulateIntervention(
      session,
      node,
      intervention
    );

    // Compare with actual
    const comparisons: Counterfactual["actualVsPredicted"] = [];

    for (const [key, predicted] of Object.entries(predictedOutcome)) {
      const actual = node.data[key];
      if (typeof actual === "number" && typeof predicted === "number") {
        comparisons.push({
          metric: key,
          actual,
          predicted,
          difference: predicted - actual,
        });
      }
    }

    const counterfactual: Counterfactual = {
      id: this.generateId("cf"),
      nodeId,
      intervention,
      predictedOutcome,
      confidenceScore: 0.7,
      actualVsPredicted: comparisons,
    };

    node.counterfactuals.push(counterfactual);

    return counterfactual;
  }

  /**
   * Simulate intervention
   */
  private simulateIntervention(
    session: TrainingSession,
    node: CausalNode,
    intervention: Intervention
  ): Record<string, unknown> {
    // Simplified simulation - in reality would run causal model
    const result = { ...node.data };

    // Basic linear causal effect
    const strength =
      Array.from(node.causalStrength.values()).reduce((sum, s) => sum + s, 0) /
      Math.max(1, node.causalStrength.size);

    // Estimate effect based on intervention
    if (
      typeof intervention.originalValue === "number" &&
      typeof intervention.interventionValue === "number"
    ) {
      const delta = intervention.interventionValue - intervention.originalValue;

      // Propagate effect to downstream metrics
      for (const effectId of node.effects) {
        const effect = session.nodes.get(effectId);
        if (effect) {
          // Simple linear model
          for (const [key, value] of Object.entries(effect.data)) {
            if (typeof value === "number") {
              result[key] = value + delta * strength;
            }
          }
        }
      }
    }

    return result;
  }

  /**
   * Find critical path
   */
  findCriticalPath(): CausalPath | null {
    const session = this.getActiveSession();
    if (!session) return null;

    const paths = Array.from(session.paths.values());
    if (paths.length === 0) {
      // Build paths from roots to leaves
      this.buildAllPaths(session);
    }

    // Find path with highest total strength
    let criticalPath: CausalPath | null = null;
    let maxStrength = 0;

    for (const path of session.paths.values()) {
      if (path.totalStrength > maxStrength) {
        maxStrength = path.totalStrength;
        criticalPath = path;
      }
    }

    if (criticalPath) {
      this.emit("path:highlighted", { pathId: criticalPath.id });
    }

    return criticalPath;
  }

  /**
   * Find all causes of a node
   */
  private findAllCauses(
    session: TrainingSession,
    nodeId: string,
    depth: number
  ): CausalNode[] {
    const causes: CausalNode[] = [];
    const visited = new Set<string>();

    const traverse = (currentId: string, currentDepth: number): void => {
      if (currentDepth > depth || visited.has(currentId)) return;
      visited.add(currentId);

      const node = session.nodes.get(currentId);
      if (!node) return;

      for (const causeId of node.causes) {
        const cause = session.nodes.get(causeId);
        if (cause) {
          causes.push(cause);
          traverse(causeId, currentDepth + 1);
        }
      }
    };

    traverse(nodeId, 0);

    return causes;
  }

  /**
   * Find all effects of a node
   */
  private findAllEffects(
    session: TrainingSession,
    nodeId: string,
    depth: number
  ): CausalNode[] {
    const effects: CausalNode[] = [];
    const visited = new Set<string>();

    const traverse = (currentId: string, currentDepth: number): void => {
      if (currentDepth > depth || visited.has(currentId)) return;
      visited.add(currentId);

      const node = session.nodes.get(currentId);
      if (!node) return;

      for (const effectId of node.effects) {
        const effect = session.nodes.get(effectId);
        if (effect) {
          effects.push(effect);
          traverse(effectId, currentDepth + 1);
        }
      }
    };

    traverse(nodeId, 0);

    return effects;
  }

  /**
   * Get connecting edges between nodes
   */
  private getConnectingEdges(
    session: TrainingSession,
    nodeIds: string[]
  ): CausalEdge[] {
    const nodeSet = new Set(nodeIds);
    return Array.from(session.edges.values()).filter(
      (e) => nodeSet.has(e.sourceId) || nodeSet.has(e.targetId)
    );
  }

  /**
   * Find causal paths
   */
  private findCausalPaths(
    session: TrainingSession,
    startId: string,
    direction: "forward" | "backward",
    maxDepth: number
  ): CausalPath[] {
    const paths: CausalPath[] = [];

    const findPath = (
      currentId: string,
      currentPath: string[],
      currentEdges: string[],
      depth: number
    ): void => {
      if (depth > maxDepth) return;

      const node = session.nodes.get(currentId);
      if (!node) return;

      const nextNodes = direction === "forward" ? node.effects : node.causes;

      if (nextNodes.length === 0) {
        // End of path
        if (currentPath.length > 1) {
          const path = this.createPath(
            session,
            currentPath,
            currentEdges,
            direction
          );
          paths.push(path);
        }
        return;
      }

      for (const nextId of nextNodes) {
        // Find edge
        const edge = Array.from(session.edges.values()).find((e) =>
          direction === "forward"
            ? e.sourceId === currentId && e.targetId === nextId
            : e.targetId === currentId && e.sourceId === nextId
        );

        findPath(
          nextId,
          [...currentPath, nextId],
          edge ? [...currentEdges, edge.id] : currentEdges,
          depth + 1
        );
      }
    };

    findPath(startId, [startId], [], 0);

    return paths;
  }

  /**
   * Create path object
   */
  private createPath(
    session: TrainingSession,
    nodeIds: string[],
    edgeIds: string[],
    direction: "forward" | "backward"
  ): CausalPath {
    const id = this.generateId("path");

    // Calculate total strength
    let totalStrength = 1;
    let totalDelay = 0;

    for (const edgeId of edgeIds) {
      const edge = session.edges.get(edgeId);
      if (edge) {
        totalStrength *= edge.strength;
        totalDelay += edge.timeDelay;
      }
    }

    // Generate description
    const startNode = session.nodes.get(nodeIds[0]);
    const endNode = session.nodes.get(nodeIds[nodeIds.length - 1]);

    const path: CausalPath = {
      id,
      nodes: nodeIds,
      edges: edgeIds,
      totalStrength,
      totalDelay,
      pathType: direction === "forward" ? "forward" : "backward",
      description: `${startNode?.label ?? "?"} → ... → ${endNode?.label ?? "?"}`,
      keyInsights: [],
    };

    session.paths.set(id, path);
    this.emit("path:discovered", { path });

    return path;
  }

  /**
   * Build path between two nodes
   */
  private buildPath(
    session: TrainingSession,
    fromId: string,
    toId: string
  ): CausalPath | null {
    // BFS to find shortest path
    const queue: Array<{ nodeId: string; path: string[]; edges: string[] }> = [
      { nodeId: fromId, path: [fromId], edges: [] },
    ];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const current = queue.shift()!;

      if (current.nodeId === toId) {
        return this.createPath(session, current.path, current.edges, "forward");
      }

      if (visited.has(current.nodeId)) continue;
      visited.add(current.nodeId);

      const node = session.nodes.get(current.nodeId);
      if (!node) continue;

      for (const effectId of node.effects) {
        const edge = Array.from(session.edges.values()).find(
          (e) => e.sourceId === current.nodeId && e.targetId === effectId
        );

        queue.push({
          nodeId: effectId,
          path: [...current.path, effectId],
          edges: edge ? [...current.edges, edge.id] : current.edges,
        });
      }
    }

    return null;
  }

  /**
   * Build all paths in session
   */
  private buildAllPaths(session: TrainingSession): void {
    // Find roots (no causes)
    const roots = Array.from(session.nodes.values()).filter(
      (n) => n.causes.length === 0
    );

    // Find leaves (no effects)
    const leaves = Array.from(session.nodes.values()).filter(
      (n) => n.effects.length === 0
    );

    // Build paths from each root to each leaf
    for (const root of roots) {
      for (const leaf of leaves) {
        const path = this.buildPath(session, root.id, leaf.id);
        if (path) {
          session.paths.set(path.id, path);
        }
      }
    }
  }

  /**
   * Get edges in paths
   */
  private getEdgesInPaths(
    session: TrainingSession,
    paths: CausalPath[]
  ): CausalEdge[] {
    const edgeIds = new Set<string>();

    for (const path of paths) {
      for (const edgeId of path.edges) {
        edgeIds.add(edgeId);
      }
    }

    return Array.from(edgeIds)
      .map((id) => session.edges.get(id))
      .filter((e): e is CausalEdge => e !== undefined);
  }

  /**
   * Rank nodes by importance
   */
  private rankByImportance(
    nodes: CausalNode[],
    target: CausalNode
  ): CausalNode[] {
    // Rank by causal strength to target
    return nodes.sort((a, b) => {
      const strengthA = target.causalStrength.get(a.id) ?? 0;
      const strengthB = target.causalStrength.get(b.id) ?? 0;
      return strengthB - strengthA;
    });
  }

  /**
   * Generate "why" insights
   */
  private generateWhyInsights(
    target: CausalNode,
    causes: CausalNode[]
  ): string[] {
    const insights: string[] = [];

    // Top cause
    if (causes.length > 0) {
      const topCause = causes[0];
      const strength = target.causalStrength.get(topCause.id) ?? 0;
      insights.push(
        `Primary cause: "${topCause.label}" (strength: ${(strength * 100).toFixed(1)}%)`
      );
    }

    // Categorize by type
    const typeGroups = new Map<CausalNodeType, number>();
    for (const cause of causes) {
      typeGroups.set(cause.nodeType, (typeGroups.get(cause.nodeType) ?? 0) + 1);
    }

    const sortedTypes = Array.from(typeGroups.entries()).sort(
      (a, b) => b[1] - a[1]
    );

    if (sortedTypes.length > 0) {
      insights.push(
        `Most common cause type: ${sortedTypes[0][0]} (${sortedTypes[0][1]} occurrences)`
      );
    }

    // Temporal insight
    const times = causes.map((c) => c.timestamp);
    if (times.length > 0) {
      const earliest = Math.min(...times);
      const latest = Math.max(...times);
      const span = latest - earliest;
      insights.push(
        `Causal chain spans ${(span / 1000).toFixed(1)}s of training time`
      );
    }

    return insights;
  }

  /**
   * Generate effect insights
   */
  private generateEffectInsights(
    source: CausalNode,
    effects: CausalNode[]
  ): string[] {
    const insights: string[] = [];

    insights.push(
      `"${source.label}" influenced ${effects.length} downstream events`
    );

    // Direct vs indirect
    const direct = effects.filter((e) => source.effects.includes(e.id));
    insights.push(
      `${direct.length} direct effects, ${effects.length - direct.length} indirect effects`
    );

    // Impact spread
    const epochs = new Set(effects.map((e) => e.epoch));
    if (epochs.size > 1) {
      insights.push(`Effects spread across ${epochs.size} epochs`);
    }

    return insights;
  }

  /**
   * Generate recommendations
   */
  private generateRecommendations(
    target: CausalNode,
    causes: CausalNode[]
  ): string[] {
    const recommendations: string[] = [];

    // Based on target type
    if (target.nodeType === "loss_computation") {
      const lrCauses = causes.filter((c) => c.nodeType === "learning_rate");
      if (lrCauses.length > 0) {
        recommendations.push(
          "Consider adjusting learning rate - it strongly influences loss computation"
        );
      }
    }

    // Based on cause types
    const hyperparamsCount = causes.filter(
      (c) => c.nodeType === "hyperparameter"
    ).length;
    if (hyperparamsCount > 3) {
      recommendations.push(
        "Multiple hyperparameters influence this outcome - consider systematic tuning"
      );
    }

    return recommendations;
  }

  /**
   * Generate root cause recommendations
   */
  private generateRootCauseRecommendations(rootCauses: CausalNode[]): string[] {
    const recommendations: string[] = [];

    for (const root of rootCauses.slice(0, 3)) {
      recommendations.push(
        `Investigate "${root.label}" - it's a root cause in the causal chain`
      );
    }

    return recommendations;
  }

  /**
   * Calculate confidence
   */
  private calculateConfidence(nodes: CausalNode[]): number {
    if (nodes.length === 0) return 0;

    // Average of causal strengths
    let totalStrength = 0;
    let count = 0;

    for (const node of nodes) {
      for (const strength of node.causalStrength.values()) {
        totalStrength += strength;
        count++;
      }
    }

    return count > 0 ? totalStrength / count : 0.5;
  }

  /**
   * Create empty result
   */
  private createEmptyResult(
    query: InvestigationQuery,
    reason: string
  ): InvestigationResult {
    return {
      queryId: this.generateId("query"),
      queryType: query.queryType,
      nodes: [],
      edges: [],
      paths: [],
      summary: `Investigation failed: ${reason}`,
      insights: [],
      recommendations: [],
      confidence: 0,
      limitations: [reason],
    };
  }

  /**
   * Highlight investigation results
   */
  private highlightInvestigation(
    nodes: CausalNode[],
    edges: CausalEdge[]
  ): void {
    this.highlightedNodes = new Set(nodes.map((n) => n.id));
    this.highlightedEdges = new Set(edges.map((e) => e.id));
  }

  /**
   * Update session metrics
   */
  private updateMetrics(session: TrainingSession): void {
    // Find roots and leaves
    const roots: string[] = [];
    const leaves: string[] = [];

    for (const node of session.nodes.values()) {
      if (node.causes.length === 0) roots.push(node.id);
      if (node.effects.length === 0) leaves.push(node.id);
    }

    session.metrics.rootCauses = roots;
    session.metrics.terminalEffects = leaves;

    // Average causal strength
    let totalStrength = 0;
    let strengthCount = 0;

    for (const edge of session.edges.values()) {
      totalStrength += edge.strength;
      strengthCount++;
    }

    session.metrics.averageCausalStrength =
      strengthCount > 0 ? totalStrength / strengthCount : 0;

    // Longest path
    this.buildAllPaths(session);
    let maxLength = 0;

    for (const path of session.paths.values()) {
      maxLength = Math.max(maxLength, path.nodes.length);
    }

    session.metrics.longestPath = maxLength;
  }

  // ============================================================================
  // Utility
  // ============================================================================

  /**
   * Generate unique ID
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Add bookmark
   */
  addBookmark(nodeId: string, label: string, notes: string): Bookmark {
    const session = this.getActiveSession();
    if (!session) throw new Error("No active session");

    const bookmark: Bookmark = {
      id: this.generateId("bookmark"),
      nodeId,
      label,
      notes,
      timestamp: Date.now(),
    };

    session.bookmarks.push(bookmark);
    this.emit("bookmark:added", { bookmark });

    return bookmark;
  }

  /**
   * Move timeline
   */
  moveTimeline(time: number): void {
    const session = this.getActiveSession();
    if (!session) return;

    session.currentTime = time;
    this.emit("timeline:moved", { newTime: time });
  }

  /**
   * Get all sessions
   */
  getAllSessions(): TrainingSession[] {
    return Array.from(this.sessions.values());
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.sessions.clear();
    this.highlightedNodes.clear();
    this.highlightedEdges.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create Causal Training Debugger
 */
export function createCausalTrainingDebugger(
  config?: Partial<DebuggerConfig>
): CausalTrainingDebugger {
  return new CausalTrainingDebugger(config);
}
