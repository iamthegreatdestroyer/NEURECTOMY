/**
 * Predictive Cascades POC
 *
 * Predicts failure propagation and cascading effects across distributed systems
 * using dependency graphs, confidence decay, and probabilistic propagation.
 *
 * Key Innovations:
 * - Dependency-aware failure propagation modeling
 * - Confidence decay with temporal and spatial distance
 * - Cascade detection with early warning
 * - Impact amplification and mitigation analysis
 * - Multi-hop dependency traversal
 *
 * Research Foundations:
 * - Watts (2002): A simple model of global cascades on random networks
 * - Newman (2003): The structure and function of complex networks
 * - Barabási & Albert (1999): Emergence of scaling in random networks
 * - Dobson et al. (2007): Complex systems analysis of cascading failures
 *
 * @elite-agents @ORACLE @VELOCITY @PRISM
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type NodeId = string;
type CascadeId = string;
type Timestamp = number;

enum NodeStatus {
  HEALTHY = "healthy",
  DEGRADED = "degraded",
  FAILED = "failed",
  RECOVERING = "recovering",
}

enum DependencyType {
  SYNCHRONOUS = "sync",
  ASYNCHRONOUS = "async",
  DATA = "data",
  CONTROL = "control",
}

interface SystemNode {
  id: NodeId;
  name: string;
  status: NodeStatus;
  health: number; // 0-1
  capacity: number;
  load: number;
  resilience: number; // Ability to handle failures
  criticalityScore: number;
}

interface Dependency {
  from: NodeId;
  to: NodeId;
  type: DependencyType;
  strength: number; // 0-1, probability of propagation
  latency: number; // milliseconds
  required: boolean; // If true, failure propagates immediately
}

interface FailureEvent {
  nodeId: NodeId;
  timestamp: Timestamp;
  severity: number; // 0-1
  cause: string;
  predicted: boolean;
}

interface CascadePrediction {
  cascadeId: CascadeId;
  originNode: NodeId;
  timestamp: Timestamp;
  affectedNodes: Map<
    NodeId,
    {
      probability: number;
      expectedTime: Timestamp;
      impactSeverity: number;
      propagationPath: NodeId[];
    }
  >;
  totalImpact: number;
  confidence: number;
}

interface PropagationStep {
  fromNode: NodeId;
  toNode: NodeId;
  timestamp: Timestamp;
  probability: number;
  cumulativeProbability: number;
  depth: number;
}

// ============================================================================
// Dependency Graph
// ============================================================================

class DependencyGraph {
  private nodes: Map<NodeId, SystemNode>;
  private dependencies: Map<NodeId, Dependency[]>; // outgoing dependencies
  private reverseDeps: Map<NodeId, Dependency[]>; // incoming dependencies

  constructor() {
    this.nodes = new Map();
    this.dependencies = new Map();
    this.reverseDeps = new Map();
  }

  addNode(node: SystemNode): void {
    this.nodes.set(node.id, node);
    if (!this.dependencies.has(node.id)) {
      this.dependencies.set(node.id, []);
    }
    if (!this.reverseDeps.has(node.id)) {
      this.reverseDeps.set(node.id, []);
    }
  }

  addDependency(dep: Dependency): void {
    // Outgoing
    const outgoing = this.dependencies.get(dep.from) ?? [];
    outgoing.push(dep);
    this.dependencies.set(dep.from, outgoing);

    // Incoming
    const incoming = this.reverseDeps.get(dep.to) ?? [];
    incoming.push(dep);
    this.reverseDeps.set(dep.to, incoming);
  }

  getNode(nodeId: NodeId): SystemNode | undefined {
    return this.nodes.get(nodeId);
  }

  getDependencies(nodeId: NodeId): Dependency[] {
    return this.dependencies.get(nodeId) ?? [];
  }

  getDependents(nodeId: NodeId): Dependency[] {
    return this.reverseDeps.get(nodeId) ?? [];
  }

  getAllNodes(): SystemNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Find critical nodes (high impact if failed)
   */
  findCriticalNodes(): SystemNode[] {
    const criticality = new Map<NodeId, number>();

    for (const node of this.nodes.values()) {
      let score = 0;

      // Base criticality
      score += node.criticalityScore;

      // Number of dependents
      const dependents = this.reverseDeps.get(node.id) ?? [];
      score += dependents.length * 0.1;

      // Required dependencies
      const required = dependents.filter((d) => d.required).length;
      score += required * 0.3;

      criticality.set(node.id, score);
    }

    return Array.from(this.nodes.values()).sort(
      (a, b) => (criticality.get(b.id) ?? 0) - (criticality.get(a.id) ?? 0)
    );
  }

  /**
   * Compute shortest path between nodes
   */
  shortestPath(from: NodeId, to: NodeId): NodeId[] | null {
    const queue: Array<{ node: NodeId; path: NodeId[] }> = [
      { node: from, path: [from] },
    ];
    const visited = new Set<NodeId>();

    while (queue.length > 0) {
      const current = queue.shift()!;

      if (current.node === to) {
        return current.path;
      }

      if (visited.has(current.node)) continue;
      visited.add(current.node);

      const deps = this.dependencies.get(current.node) ?? [];
      for (const dep of deps) {
        if (!visited.has(dep.to)) {
          queue.push({
            node: dep.to,
            path: [...current.path, dep.to],
          });
        }
      }
    }

    return null;
  }
}

// ============================================================================
// Cascade Predictor
// ============================================================================

class CascadePredictor {
  private graph: DependencyGraph;
  private confidenceDecayRate: number;
  private propagationThreshold: number;

  constructor(
    graph: DependencyGraph,
    confidenceDecayRate: number = 0.1,
    propagationThreshold: number = 0.3
  ) {
    this.graph = graph;
    this.confidenceDecayRate = confidenceDecayRate;
    this.propagationThreshold = propagationThreshold;
  }

  /**
   * Predict cascade from initial failure
   */
  predictCascade(
    originNode: NodeId,
    initialSeverity: number,
    maxDepth: number = 5
  ): CascadePrediction {
    const cascadeId = `cascade_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const affectedNodes = new Map<
      NodeId,
      {
        probability: number;
        expectedTime: Timestamp;
        impactSeverity: number;
        propagationPath: NodeId[];
      }
    >();

    const startTime = Date.now();

    // BFS with probability tracking
    const queue: PropagationStep[] = [
      {
        fromNode: originNode,
        toNode: originNode,
        timestamp: startTime,
        probability: 1.0,
        cumulativeProbability: 1.0,
        depth: 0,
      },
    ];

    const visited = new Map<NodeId, number>(); // node -> best probability seen

    while (queue.length > 0) {
      const step = queue.shift()!;

      if (step.depth > maxDepth) continue;

      // Check if we've seen this node with better probability
      const bestProb = visited.get(step.toNode) ?? 0;
      if (step.cumulativeProbability <= bestProb) continue;

      visited.set(step.toNode, step.cumulativeProbability);

      // Skip if probability too low
      if (step.cumulativeProbability < this.propagationThreshold) continue;

      // Record affected node
      const node = this.graph.getNode(step.toNode);
      if (node && step.toNode !== originNode) {
        const existingData = affectedNodes.get(step.toNode);

        if (
          !existingData ||
          step.cumulativeProbability > existingData.probability
        ) {
          affectedNodes.set(step.toNode, {
            probability: step.cumulativeProbability,
            expectedTime: step.timestamp,
            impactSeverity:
              initialSeverity *
              step.cumulativeProbability *
              (1 - node.resilience),
            propagationPath: this.graph.shortestPath(
              originNode,
              step.toNode
            ) ?? [originNode, step.toNode],
          });
        }
      }

      // Propagate to dependencies
      const deps = this.graph.getDependencies(step.toNode);
      for (const dep of deps) {
        const targetNode = this.graph.getNode(dep.to);
        if (!targetNode) continue;

        // Compute propagation probability
        let propagationProb = this.computePropagationProbability(
          node!,
          targetNode,
          dep,
          step.cumulativeProbability,
          step.depth
        );

        // Required dependencies have higher propagation
        if (dep.required) {
          propagationProb = Math.max(propagationProb, 0.8);
        }

        const newCumulativeProb = step.cumulativeProbability * propagationProb;

        queue.push({
          fromNode: step.toNode,
          toNode: dep.to,
          timestamp: step.timestamp + dep.latency,
          probability: propagationProb,
          cumulativeProbability: newCumulativeProb,
          depth: step.depth + 1,
        });
      }
    }

    // Compute total impact
    let totalImpact = 0;
    for (const data of affectedNodes.values()) {
      totalImpact += data.probability * data.impactSeverity;
    }

    // Compute prediction confidence
    const confidence = this.computeConfidence(affectedNodes.size, maxDepth);

    return {
      cascadeId,
      originNode,
      timestamp: startTime,
      affectedNodes,
      totalImpact,
      confidence,
    };
  }

  /**
   * Compute probability of failure propagation
   */
  private computePropagationProbability(
    sourceNode: SystemNode,
    targetNode: SystemNode,
    dependency: Dependency,
    cumulativeProb: number,
    depth: number
  ): number {
    // Base propagation from dependency strength
    let prob = dependency.strength;

    // Reduce by target resilience
    prob *= 1 - targetNode.resilience;

    // Increase if target is overloaded
    const loadFactor = targetNode.load / targetNode.capacity;
    if (loadFactor > 0.8) {
      prob *= 1.0 + (loadFactor - 0.8) * 2;
    }

    // Confidence decay with depth
    prob *= Math.exp(-depth * this.confidenceDecayRate);

    // Synchronous dependencies propagate faster
    if (dependency.type === DependencyType.SYNCHRONOUS) {
      prob *= 1.2;
    }

    return Math.min(1.0, prob);
  }

  /**
   * Compute prediction confidence
   */
  private computeConfidence(affectedCount: number, maxDepth: number): number {
    // Confidence decreases with cascade size and depth
    const sizeConfidence = 1.0 / (1.0 + Math.log(affectedCount + 1));
    const depthConfidence = Math.exp(-maxDepth * 0.1);
    return (sizeConfidence + depthConfidence) / 2;
  }

  /**
   * Detect ongoing cascades
   */
  detectCascade(recentFailures: FailureEvent[]): boolean {
    if (recentFailures.length < 2) return false;

    // Sort by timestamp
    const sorted = [...recentFailures].sort(
      (a, b) => a.timestamp - b.timestamp
    );

    // Check temporal clustering (failures within short time window)
    const timeWindow = 60000; // 1 minute
    let clusteredFailures = 0;

    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].timestamp - sorted[i - 1].timestamp < timeWindow) {
        clusteredFailures++;
      }
    }

    // Check dependency relationships
    let connectedFailures = 0;
    for (let i = 0; i < sorted.length; i++) {
      for (let j = i + 1; j < sorted.length; j++) {
        const path = this.graph.shortestPath(
          sorted[i].nodeId,
          sorted[j].nodeId
        );
        if (path && path.length <= 3) {
          connectedFailures++;
        }
      }
    }

    // Cascade detected if we have clustered and connected failures
    return clusteredFailures >= 2 || connectedFailures >= 2;
  }
}

// ============================================================================
// Propagation Engine
// ============================================================================

class PropagationEngine {
  private graph: DependencyGraph;
  private predictor: CascadePredictor;
  private failureHistory: FailureEvent[];

  constructor(graph: DependencyGraph) {
    this.graph = graph;
    this.predictor = new CascadePredictor(graph);
    this.failureHistory = [];
  }

  /**
   * Simulate failure and track propagation
   */
  simulateFailure(
    nodeId: NodeId,
    severity: number
  ): {
    prediction: CascadePrediction;
    actualPropagation: Map<NodeId, { failed: boolean; time: Timestamp }>;
  } {
    // Predict cascade
    const prediction = this.predictor.predictCascade(nodeId, severity);

    // Simulate actual propagation
    const actualPropagation = new Map<
      NodeId,
      { failed: boolean; time: Timestamp }
    >();

    for (const [nodeId, data] of prediction.affectedNodes) {
      // Probabilistic failure
      const failed = Math.random() < data.probability;
      actualPropagation.set(nodeId, {
        failed,
        time: data.expectedTime,
      });

      if (failed) {
        this.recordFailure({
          nodeId,
          timestamp: data.expectedTime,
          severity: data.impactSeverity,
          cause: `Cascade from ${prediction.originNode}`,
          predicted: true,
        });
      }
    }

    return { prediction, actualPropagation };
  }

  /**
   * Record failure event
   */
  recordFailure(event: FailureEvent): void {
    this.failureHistory.push(event);

    // Keep only recent history
    const cutoff = Date.now() - 3600000; // 1 hour
    this.failureHistory = this.failureHistory.filter(
      (e) => e.timestamp > cutoff
    );
  }

  /**
   * Get cascade detection status
   */
  isCascadeOngoing(): boolean {
    return this.predictor.detectCascade(this.failureHistory);
  }

  /**
   * Analyze mitigation strategies
   */
  analyzeMitigation(prediction: CascadePrediction): Array<{
    nodeId: NodeId;
    action: string;
    impactReduction: number;
    priority: number;
  }> {
    const strategies: Array<{
      nodeId: NodeId;
      action: string;
      impactReduction: number;
      priority: number;
    }> = [];

    for (const [nodeId, data] of prediction.affectedNodes) {
      const node = this.graph.getNode(nodeId);
      if (!node) continue;

      // Strategy 1: Increase capacity
      if (node.load / node.capacity > 0.7) {
        strategies.push({
          nodeId,
          action: "scale_up",
          impactReduction: data.impactSeverity * 0.4,
          priority: data.probability * data.impactSeverity,
        });
      }

      // Strategy 2: Circuit breaker
      const deps = this.graph.getDependents(nodeId);
      if (deps.length > 2) {
        strategies.push({
          nodeId,
          action: "enable_circuit_breaker",
          impactReduction: data.impactSeverity * 0.6,
          priority: data.probability * data.impactSeverity * 1.2,
        });
      }

      // Strategy 3: Graceful degradation
      strategies.push({
        nodeId,
        action: "enable_degraded_mode",
        impactReduction: data.impactSeverity * 0.3,
        priority: data.probability * data.impactSeverity * 0.8,
      });
    }

    // Sort by priority
    return strategies.sort((a, b) => b.priority - a.priority);
  }

  getRecentFailures(count: number = 10): FailureEvent[] {
    return this.failureHistory.slice(-count);
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstratePredictiveCascades(): Promise<void> {
  console.log("=".repeat(80));
  console.log("PREDICTIVE CASCADES DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Build Dependency Graph
  console.log("\n🕸️ Demo 1: Microservices Dependency Graph");
  console.log("-".repeat(80));

  const graph = new DependencyGraph();

  // Add nodes
  const nodes: SystemNode[] = [
    {
      id: "api-gateway",
      name: "API Gateway",
      status: NodeStatus.HEALTHY,
      health: 0.95,
      capacity: 1000,
      load: 600,
      resilience: 0.7,
      criticalityScore: 0.9,
    },
    {
      id: "auth-service",
      name: "Auth Service",
      status: NodeStatus.HEALTHY,
      health: 0.9,
      capacity: 500,
      load: 300,
      resilience: 0.8,
      criticalityScore: 0.8,
    },
    {
      id: "user-service",
      name: "User Service",
      status: NodeStatus.HEALTHY,
      health: 0.92,
      capacity: 800,
      load: 500,
      resilience: 0.6,
      criticalityScore: 0.7,
    },
    {
      id: "order-service",
      name: "Order Service",
      status: NodeStatus.HEALTHY,
      health: 0.88,
      capacity: 600,
      load: 480,
      resilience: 0.5,
      criticalityScore: 0.75,
    },
    {
      id: "payment-service",
      name: "Payment Service",
      status: NodeStatus.HEALTHY,
      health: 0.93,
      capacity: 400,
      load: 350,
      resilience: 0.9,
      criticalityScore: 0.95,
    },
    {
      id: "inventory-service",
      name: "Inventory Service",
      status: NodeStatus.HEALTHY,
      health: 0.85,
      capacity: 500,
      load: 450,
      resilience: 0.4,
      criticalityScore: 0.6,
    },
    {
      id: "database",
      name: "Database",
      status: NodeStatus.HEALTHY,
      health: 0.98,
      capacity: 2000,
      load: 1200,
      resilience: 0.85,
      criticalityScore: 1.0,
    },
  ];

  for (const node of nodes) {
    graph.addNode(node);
    console.log(
      `  ${node.name}: load=${node.load}/${node.capacity}, resilience=${node.resilience.toFixed(2)}`
    );
  }

  // Add dependencies
  const dependencies: Dependency[] = [
    {
      from: "api-gateway",
      to: "auth-service",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.9,
      latency: 50,
      required: true,
    },
    {
      from: "api-gateway",
      to: "user-service",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.8,
      latency: 60,
      required: false,
    },
    {
      from: "api-gateway",
      to: "order-service",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.85,
      latency: 70,
      required: false,
    },
    {
      from: "order-service",
      to: "payment-service",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.95,
      latency: 100,
      required: true,
    },
    {
      from: "order-service",
      to: "inventory-service",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.8,
      latency: 80,
      required: true,
    },
    {
      from: "auth-service",
      to: "database",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.9,
      latency: 20,
      required: true,
    },
    {
      from: "user-service",
      to: "database",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.85,
      latency: 25,
      required: true,
    },
    {
      from: "order-service",
      to: "database",
      type: DependencyType.SYNCHRONOUS,
      strength: 0.9,
      latency: 30,
      required: true,
    },
  ];

  for (const dep of dependencies) {
    graph.addDependency(dep);
  }

  console.log(
    `\n✓ Created graph with ${nodes.length} nodes and ${dependencies.length} dependencies`
  );

  // Demo 2: Critical Node Analysis
  console.log("\n🎯 Demo 2: Critical Node Identification");
  console.log("-".repeat(80));

  const criticalNodes = graph.findCriticalNodes();
  console.log("Most critical nodes (by impact if failed):");

  for (let i = 0; i < Math.min(3, criticalNodes.length); i++) {
    const node = criticalNodes[i];
    console.log(
      `  ${i + 1}. ${node.name} (criticality: ${node.criticalityScore.toFixed(2)})`
    );
  }

  // Demo 3: Cascade Prediction
  console.log("\n⚠️ Demo 3: Cascade Prediction from Database Failure");
  console.log("-".repeat(80));

  const predictor = new CascadePredictor(graph);
  const prediction = predictor.predictCascade("database", 0.9, 5);

  console.log(`Cascade ID: ${prediction.cascadeId}`);
  console.log(`Origin: ${prediction.originNode}`);
  console.log(`Predicted affected nodes: ${prediction.affectedNodes.size}`);
  console.log(`Total impact score: ${prediction.totalImpact.toFixed(2)}`);
  console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);

  console.log("\nAffected nodes (by probability):");
  const sortedAffected = Array.from(prediction.affectedNodes.entries()).sort(
    (a, b) => b[1].probability - a[1].probability
  );

  for (const [nodeId, data] of sortedAffected.slice(0, 5)) {
    const node = graph.getNode(nodeId);
    console.log(`  ${node?.name}:`);
    console.log(
      `    Failure probability: ${(data.probability * 100).toFixed(1)}%`
    );
    console.log(
      `    Expected time: +${data.expectedTime - prediction.timestamp}ms`
    );
    console.log(`    Impact severity: ${data.impactSeverity.toFixed(2)}`);
    console.log(`    Path: ${data.propagationPath.join(" → ")}`);
  }

  // Demo 4: Simulation
  console.log("\n🎲 Demo 4: Failure Simulation");
  console.log("-".repeat(80));

  const engine = new PropagationEngine(graph);
  const simulation = engine.simulateFailure("auth-service", 0.8);

  console.log(`Simulating failure of: auth-service (severity: 0.8)`);
  console.log(
    `\nPredicted: ${simulation.prediction.affectedNodes.size} nodes at risk`
  );

  const actuallyFailed = Array.from(
    simulation.actualPropagation.values()
  ).filter((p) => p.failed).length;

  console.log(`Actual: ${actuallyFailed} nodes failed`);

  console.log("\nActual propagation:");
  for (const [nodeId, result] of simulation.actualPropagation) {
    const node = graph.getNode(nodeId);
    const predicted = simulation.prediction.affectedNodes.get(nodeId);
    console.log(
      `  ${node?.name}: ${result.failed ? "❌ FAILED" : "✅ OK"} ` +
        `(predicted ${(predicted!.probability * 100).toFixed(0)}%)`
    );
  }

  // Demo 5: Cascade Detection
  console.log("\n🚨 Demo 5: Cascade Detection");
  console.log("-".repeat(80));

  console.log("Injecting sequence of failures...\n");

  const failureSequence = [
    { nodeId: "database", severity: 0.7, delay: 0 },
    { nodeId: "auth-service", severity: 0.6, delay: 100 },
    { nodeId: "user-service", severity: 0.5, delay: 200 },
  ];

  for (const failure of failureSequence) {
    engine.recordFailure({
      nodeId: failure.nodeId,
      timestamp: Date.now() + failure.delay,
      severity: failure.severity,
      cause: "Cascading failure",
      predicted: false,
    });

    console.log(`Failure recorded: ${graph.getNode(failure.nodeId)?.name}`);
  }

  const cascadeDetected = engine.isCascadeOngoing();
  console.log(`\nCascade detected: ${cascadeDetected ? "⚠️ YES" : "✅ NO"}`);

  // Demo 6: Mitigation Analysis
  console.log("\n🛡️ Demo 6: Mitigation Strategy Analysis");
  console.log("-".repeat(80));

  const mitigation = engine.analyzeMitigation(prediction);

  console.log(`Generated ${mitigation.length} mitigation strategies\n`);
  console.log("Top 5 recommended actions:");

  for (let i = 0; i < Math.min(5, mitigation.length); i++) {
    const strategy = mitigation[i];
    const node = graph.getNode(strategy.nodeId);
    console.log(`  ${i + 1}. ${node?.name}:`);
    console.log(`     Action: ${strategy.action}`);
    console.log(
      `     Impact reduction: ${(strategy.impactReduction * 100).toFixed(1)}%`
    );
    console.log(`     Priority: ${strategy.priority.toFixed(2)}`);
  }

  console.log("\n✅ Predictive Cascades demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  DependencyGraph,
  CascadePredictor,
  PropagationEngine,
  NodeStatus,
  DependencyType,
  type SystemNode,
  type Dependency,
  type FailureEvent,
  type CascadePrediction,
  type PropagationStep,
};
