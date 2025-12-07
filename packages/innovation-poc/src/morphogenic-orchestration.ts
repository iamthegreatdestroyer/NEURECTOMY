/**
 * Self-Evolving Morphogenic Orchestration - Proof of Concept
 *
 * Implements topology-aware genetic algorithms for evolving
 * agent orchestration graphs that optimize for emergent behaviors.
 *
 * @module morphogenic-orchestration
 * @agents @GENESIS @ARCHITECT @APEX
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export type NodeId = string;
export type EdgeId = string;

/**
 * Orchestration node (agent or component)
 */
export interface OrchestrationNode {
  id: NodeId;
  type: "agent" | "service" | "data" | "decision";
  properties: Map<string, any>;

  /** Computational cost */
  cost: number;

  /** Performance metrics */
  latency: number;
  throughput: number;
}

/**
 * Orchestration edge (communication/dependency)
 */
export interface OrchestrationEdge {
  id: EdgeId;
  source: NodeId;
  target: NodeId;
  type: "data" | "control" | "feedback";

  /** Edge weight (importance/bandwidth) */
  weight: number;

  /** Communication cost */
  latency: number;
}

/**
 * Morphogenic graph (evolving topology)
 */
export interface MorphogenicGraph {
  nodes: Map<NodeId, OrchestrationNode>;
  edges: Map<EdgeId, OrchestrationEdge>;

  /** Generation number */
  generation: number;

  /** Fitness score */
  fitness: number;

  /** Metadata for evolution */
  metadata: {
    mutations: number;
    crossovers: number;
    age: number;
  };
}

/**
 * Topology mutation
 */
export interface TopologyMutation {
  type:
    | "add_node"
    | "remove_node"
    | "add_edge"
    | "remove_edge"
    | "modify_node"
    | "modify_edge";
  target?: NodeId | EdgeId;
  data?: any;
}

/**
 * Fitness metrics
 */
export interface FitnessMetrics {
  /** End-to-end latency */
  latency: number;

  /** Total throughput */
  throughput: number;

  /** Resource efficiency */
  cost: number;

  /** Robustness to failures */
  resilience: number;

  /** Emergent behavior score */
  emergence: number;

  /** Overall fitness */
  overall: number;
}

/**
 * Evolution parameters
 */
export interface EvolutionConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  elitismRate: number;

  /** Fitness weights */
  fitnessWeights: {
    latency: number;
    throughput: number;
    cost: number;
    resilience: number;
    emergence: number;
  };
}

// ============================================================================
// MORPHOGENIC GRAPH BUILDER
// ============================================================================

/**
 * Build and manipulate morphogenic graphs
 */
export class MorphogenicGraphBuilder {
  private graph: MorphogenicGraph;
  private nodeIdCounter = 0;
  private edgeIdCounter = 0;

  constructor() {
    this.graph = {
      nodes: new Map(),
      edges: new Map(),
      generation: 0,
      fitness: 0,
      metadata: {
        mutations: 0,
        crossovers: 0,
        age: 0,
      },
    };
  }

  /**
   * Add node to graph
   */
  addNode(
    type: OrchestrationNode["type"],
    properties: Map<string, any> = new Map()
  ): NodeId {
    const id = `node_${this.nodeIdCounter++}`;

    const node: OrchestrationNode = {
      id,
      type,
      properties,
      cost: Math.random() * 10,
      latency: Math.random() * 100,
      throughput: Math.random() * 1000,
    };

    this.graph.nodes.set(id, node);
    return id;
  }

  /**
   * Remove node (and connected edges)
   */
  removeNode(nodeId: NodeId): void {
    this.graph.nodes.delete(nodeId);

    // Remove connected edges
    for (const [edgeId, edge] of this.graph.edges) {
      if (edge.source === nodeId || edge.target === nodeId) {
        this.graph.edges.delete(edgeId);
      }
    }
  }

  /**
   * Add edge between nodes
   */
  addEdge(
    source: NodeId,
    target: NodeId,
    type: OrchestrationEdge["type"]
  ): EdgeId {
    if (!this.graph.nodes.has(source) || !this.graph.nodes.has(target)) {
      throw new Error("Source or target node does not exist");
    }

    const id = `edge_${this.edgeIdCounter++}`;

    const edge: OrchestrationEdge = {
      id,
      source,
      target,
      type,
      weight: Math.random(),
      latency: Math.random() * 50,
    };

    this.graph.edges.set(id, edge);
    return id;
  }

  /**
   * Remove edge
   */
  removeEdge(edgeId: EdgeId): void {
    this.graph.edges.delete(edgeId);
  }

  /**
   * Clone graph (for evolution)
   */
  clone(): MorphogenicGraphBuilder {
    const cloned = new MorphogenicGraphBuilder();

    // Deep copy nodes
    for (const [id, node] of this.graph.nodes) {
      cloned.graph.nodes.set(id, {
        ...node,
        properties: new Map(node.properties),
      });
    }

    // Deep copy edges
    for (const [id, edge] of this.graph.edges) {
      cloned.graph.edges.set(id, { ...edge });
    }

    cloned.graph.generation = this.graph.generation;
    cloned.graph.fitness = this.graph.fitness;
    cloned.graph.metadata = { ...this.graph.metadata };
    cloned.nodeIdCounter = this.nodeIdCounter;
    cloned.edgeIdCounter = this.edgeIdCounter;

    return cloned;
  }

  /**
   * Build immutable graph
   */
  build(): MorphogenicGraph {
    return this.graph;
  }

  /**
   * Get graph for mutation/crossover
   */
  getMutableGraph(): MorphogenicGraph {
    return this.graph;
  }
}

// ============================================================================
// FITNESS EVALUATOR
// ============================================================================

/**
 * Evaluate fitness of morphogenic graph
 */
export class FitnessEvaluator {
  private config: EvolutionConfig;

  constructor(config: EvolutionConfig) {
    this.config = config;
  }

  /**
   * Evaluate graph fitness
   */
  evaluate(graph: MorphogenicGraph): FitnessMetrics {
    // Compute individual metrics
    const latency = this.evaluateLatency(graph);
    const throughput = this.evaluateThroughput(graph);
    const cost = this.evaluateCost(graph);
    const resilience = this.evaluateResilience(graph);
    const emergence = this.evaluateEmergence(graph);

    // Weighted sum for overall fitness
    const weights = this.config.fitnessWeights;
    const overall =
      weights.latency * (1 / (1 + latency)) +
      weights.throughput * throughput +
      weights.cost * (1 / (1 + cost)) +
      weights.resilience * resilience +
      weights.emergence * emergence;

    return {
      latency,
      throughput,
      cost,
      resilience,
      emergence,
      overall,
    };
  }

  /**
   * Evaluate end-to-end latency (critical path)
   */
  private evaluateLatency(graph: MorphogenicGraph): number {
    // Find critical path using topological sort + dynamic programming
    const sorted = this.topologicalSort(graph);
    const distances = new Map<NodeId, number>();

    // Initialize
    for (const nodeId of sorted) {
      distances.set(nodeId, 0);
    }

    // Compute longest path (critical path)
    for (const nodeId of sorted) {
      const node = graph.nodes.get(nodeId)!;
      const currentDist = distances.get(nodeId)!;

      // Update successors
      for (const edge of graph.edges.values()) {
        if (edge.source === nodeId) {
          const targetDist = distances.get(edge.target)!;
          const newDist = currentDist + node.latency + edge.latency;
          distances.set(edge.target, Math.max(targetDist, newDist));
        }
      }
    }

    // Return maximum distance
    return Math.max(...distances.values());
  }

  /**
   * Evaluate aggregate throughput
   */
  private evaluateThroughput(graph: MorphogenicGraph): number {
    let totalThroughput = 0;

    for (const node of graph.nodes.values()) {
      totalThroughput += node.throughput;
    }

    return totalThroughput / graph.nodes.size; // Average throughput
  }

  /**
   * Evaluate total cost
   */
  private evaluateCost(graph: MorphogenicGraph): number {
    let totalCost = 0;

    for (const node of graph.nodes.values()) {
      totalCost += node.cost;
    }

    return totalCost;
  }

  /**
   * Evaluate resilience (redundancy + connectivity)
   */
  private evaluateResilience(graph: MorphogenicGraph): number {
    // Simple metric: average node degree
    const degrees = new Map<NodeId, number>();

    for (const nodeId of graph.nodes.keys()) {
      degrees.set(nodeId, 0);
    }

    for (const edge of graph.edges.values()) {
      degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
      degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
    }

    const avgDegree =
      [...degrees.values()].reduce((a, b) => a + b, 0) / degrees.size;

    // Normalize to 0-1
    return Math.min(avgDegree / 10, 1);
  }

  /**
   * Evaluate emergence (complexity metrics)
   */
  private evaluateEmergence(graph: MorphogenicGraph): number {
    // Metrics for emergent behavior:
    // 1. Cyclomatic complexity
    // 2. Modularity
    // 3. Feedback loops

    const numNodes = graph.nodes.size;
    const numEdges = graph.edges.size;

    // Cyclomatic complexity: E - N + 2
    const complexity = numEdges - numNodes + 2;

    // Count feedback loops (cycles)
    const feedbackLoops = this.countCycles(graph);

    // Emergence score (higher complexity + feedback = more emergence)
    return (complexity + feedbackLoops * 2) / (numNodes + 1);
  }

  /**
   * Count cycles in graph
   */
  private countCycles(graph: MorphogenicGraph): number {
    // Simplified: count back edges in DFS
    const visited = new Set<NodeId>();
    const recStack = new Set<NodeId>();
    let cycles = 0;

    const dfs = (nodeId: NodeId) => {
      visited.add(nodeId);
      recStack.add(nodeId);

      // Visit neighbors
      for (const edge of graph.edges.values()) {
        if (edge.source === nodeId) {
          if (!visited.has(edge.target)) {
            dfs(edge.target);
          } else if (recStack.has(edge.target)) {
            cycles++; // Back edge = cycle
          }
        }
      }

      recStack.delete(nodeId);
    };

    for (const nodeId of graph.nodes.keys()) {
      if (!visited.has(nodeId)) {
        dfs(nodeId);
      }
    }

    return cycles;
  }

  /**
   * Topological sort (for DAG analysis)
   */
  private topologicalSort(graph: MorphogenicGraph): NodeId[] {
    const sorted: NodeId[] = [];
    const visited = new Set<NodeId>();

    const visit = (nodeId: NodeId) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);

      // Visit successors
      for (const edge of graph.edges.values()) {
        if (edge.source === nodeId) {
          visit(edge.target);
        }
      }

      sorted.unshift(nodeId);
    };

    for (const nodeId of graph.nodes.keys()) {
      visit(nodeId);
    }

    return sorted;
  }
}

// ============================================================================
// EVOLUTION ENGINE
// ============================================================================

/**
 * Evolve morphogenic graphs using genetic algorithms
 */
export class EvolutionEngine {
  private config: EvolutionConfig;
  private evaluator: FitnessEvaluator;
  private population: MorphogenicGraphBuilder[] = [];
  private bestEver: { graph: MorphogenicGraph; fitness: number } | null = null;

  constructor(config: EvolutionConfig) {
    this.config = config;
    this.evaluator = new FitnessEvaluator(config);
  }

  /**
   * Initialize random population
   */
  initializePopulation(): void {
    this.population = [];

    for (let i = 0; i < this.config.populationSize; i++) {
      const builder = this.createRandomGraph();
      this.population.push(builder);
    }
  }

  /**
   * Create random graph
   */
  private createRandomGraph(): MorphogenicGraphBuilder {
    const builder = new MorphogenicGraphBuilder();

    // Add random nodes
    const numNodes = 5 + Math.floor(Math.random() * 10);
    const nodeIds: NodeId[] = [];

    for (let i = 0; i < numNodes; i++) {
      const types: OrchestrationNode["type"][] = [
        "agent",
        "service",
        "data",
        "decision",
      ];
      const type = types[Math.floor(Math.random() * types.length)];
      nodeIds.push(builder.addNode(type));
    }

    // Add random edges
    const numEdges = Math.floor(numNodes * 1.5);
    for (let i = 0; i < numEdges; i++) {
      const source = nodeIds[Math.floor(Math.random() * nodeIds.length)];
      const target = nodeIds[Math.floor(Math.random() * nodeIds.length)];

      if (source !== target) {
        try {
          const edgeTypes: OrchestrationEdge["type"][] = [
            "data",
            "control",
            "feedback",
          ];
          const type = edgeTypes[Math.floor(Math.random() * edgeTypes.length)];
          builder.addEdge(source, target, type);
        } catch (e) {
          // Ignore errors (e.g., duplicate edges)
        }
      }
    }

    return builder;
  }

  /**
   * Run evolution for specified generations
   */
  async evolve(): Promise<MorphogenicGraph> {
    console.log(
      `Starting evolution: ${this.config.generations} generations, population ${this.config.populationSize}`
    );

    this.initializePopulation();

    for (let gen = 0; gen < this.config.generations; gen++) {
      // Evaluate fitness
      const fitnesses = this.population.map((builder) => {
        const graph = builder.build();
        const metrics = this.evaluator.evaluate(graph);
        graph.fitness = metrics.overall;
        return { builder, fitness: metrics.overall, metrics };
      });

      // Sort by fitness (descending)
      fitnesses.sort((a, b) => b.fitness - a.fitness);

      // Track best ever
      const best = fitnesses[0];
      if (!this.bestEver || best.fitness > this.bestEver.fitness) {
        this.bestEver = {
          graph: best.builder.build(),
          fitness: best.fitness,
        };
      }

      // Log progress
      if (gen % 10 === 0) {
        console.log(
          `Generation ${gen}: Best fitness = ${best.fitness.toFixed(4)}, Avg = ${(fitnesses.reduce((sum, f) => sum + f.fitness, 0) / fitnesses.length).toFixed(4)}`
        );
      }

      // Selection & reproduction
      const newPopulation: MorphogenicGraphBuilder[] = [];

      // Elitism: keep top performers
      const eliteCount = Math.floor(
        this.config.populationSize * this.config.elitismRate
      );
      for (let i = 0; i < eliteCount; i++) {
        newPopulation.push(fitnesses[i].builder.clone());
      }

      // Crossover & mutation
      while (newPopulation.length < this.config.populationSize) {
        // Tournament selection
        const parent1 = this.tournamentSelect(fitnesses);
        const parent2 = this.tournamentSelect(fitnesses);

        let offspring: MorphogenicGraphBuilder;

        if (Math.random() < this.config.crossoverRate) {
          offspring = this.crossover(parent1, parent2);
        } else {
          offspring = parent1.clone();
        }

        if (Math.random() < this.config.mutationRate) {
          this.mutate(offspring);
        }

        const graph = offspring.getMutableGraph();
        graph.generation = gen + 1;

        newPopulation.push(offspring);
      }

      this.population = newPopulation;
    }

    console.log(
      `Evolution complete. Best fitness: ${this.bestEver!.fitness.toFixed(4)}`
    );
    return this.bestEver!.graph;
  }

  /**
   * Tournament selection
   */
  private tournamentSelect(
    fitnesses: Array<{ builder: MorphogenicGraphBuilder; fitness: number }>
  ): MorphogenicGraphBuilder {
    const tournamentSize = 3;
    let best = fitnesses[Math.floor(Math.random() * fitnesses.length)];

    for (let i = 1; i < tournamentSize; i++) {
      const candidate = fitnesses[Math.floor(Math.random() * fitnesses.length)];
      if (candidate.fitness > best.fitness) {
        best = candidate;
      }
    }

    return best.builder;
  }

  /**
   * Crossover (graph recombination)
   */
  private crossover(
    parent1: MorphogenicGraphBuilder,
    parent2: MorphogenicGraphBuilder
  ): MorphogenicGraphBuilder {
    // Simple crossover: take nodes from parent1, edges from parent2
    const offspring = parent1.clone();
    const graph1 = parent1.build();
    const graph2 = parent2.build();

    // Replace 50% of edges with edges from parent2
    const edgeIds = [...graph1.edges.keys()];
    const numReplace = Math.floor(edgeIds.length / 2);

    for (let i = 0; i < numReplace; i++) {
      const edgeId = edgeIds[Math.floor(Math.random() * edgeIds.length)];
      offspring.removeEdge(edgeId);
    }

    // Add random edges from parent2
    const parent2Edges = [...graph2.edges.values()];
    for (let i = 0; i < numReplace && i < parent2Edges.length; i++) {
      const edge =
        parent2Edges[Math.floor(Math.random() * parent2Edges.length)];
      try {
        offspring.addEdge(edge.source, edge.target, edge.type);
      } catch (e) {
        // Ignore errors
      }
    }

    const offspringGraph = offspring.getMutableGraph();
    offspringGraph.metadata.crossovers++;

    return offspring;
  }

  /**
   * Mutate graph
   */
  private mutate(builder: MorphogenicGraphBuilder): void {
    const graph = builder.getMutableGraph();
    const mutationType = Math.random();

    if (mutationType < 0.3 && graph.nodes.size < 20) {
      // Add node
      const types: OrchestrationNode["type"][] = [
        "agent",
        "service",
        "data",
        "decision",
      ];
      const type = types[Math.floor(Math.random() * types.length)];
      builder.addNode(type);
      graph.metadata.mutations++;
    } else if (mutationType < 0.5 && graph.nodes.size > 3) {
      // Remove node
      const nodeIds = [...graph.nodes.keys()];
      const nodeId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
      builder.removeNode(nodeId);
      graph.metadata.mutations++;
    } else if (mutationType < 0.7) {
      // Add edge
      const nodeIds = [...graph.nodes.keys()];
      if (nodeIds.length >= 2) {
        const source = nodeIds[Math.floor(Math.random() * nodeIds.length)];
        const target = nodeIds[Math.floor(Math.random() * nodeIds.length)];
        if (source !== target) {
          try {
            const edgeTypes: OrchestrationEdge["type"][] = [
              "data",
              "control",
              "feedback",
            ];
            const type =
              edgeTypes[Math.floor(Math.random() * edgeTypes.length)];
            builder.addEdge(source, target, type);
            graph.metadata.mutations++;
          } catch (e) {
            // Ignore errors
          }
        }
      }
    } else if (graph.edges.size > 0) {
      // Remove edge
      const edgeIds = [...graph.edges.keys()];
      const edgeId = edgeIds[Math.floor(Math.random() * edgeIds.length)];
      builder.removeEdge(edgeId);
      graph.metadata.mutations++;
    }
  }
}

// ============================================================================
// DEMO & TESTING
// ============================================================================

/**
 * Demonstration of morphogenic orchestration
 */
export async function demonstrateMorphogenicOrchestration(): Promise<void> {
  console.log("=".repeat(70));
  console.log("SELF-EVOLVING MORPHOGENIC ORCHESTRATION - PROOF OF CONCEPT");
  console.log("=".repeat(70));
  console.log();

  // Configuration
  const config: EvolutionConfig = {
    populationSize: 20,
    generations: 50,
    mutationRate: 0.3,
    crossoverRate: 0.7,
    elitismRate: 0.1,
    fitnessWeights: {
      latency: 0.3,
      throughput: 0.2,
      cost: 0.2,
      resilience: 0.15,
      emergence: 0.15,
    },
  };

  console.log("Evolution Configuration:");
  console.log(`  Population: ${config.populationSize}`);
  console.log(`  Generations: ${config.generations}`);
  console.log(`  Mutation Rate: ${config.mutationRate}`);
  console.log(`  Crossover Rate: ${config.crossoverRate}`);
  console.log();

  // Run evolution
  const engine = new EvolutionEngine(config);
  const bestGraph = await engine.evolve();

  console.log();
  console.log("Best Graph:");
  console.log(`  Nodes: ${bestGraph.nodes.size}`);
  console.log(`  Edges: ${bestGraph.edges.size}`);
  console.log(`  Generation: ${bestGraph.generation}`);
  console.log(`  Fitness: ${bestGraph.fitness.toFixed(4)}`);
  console.log(`  Mutations: ${bestGraph.metadata.mutations}`);
  console.log(`  Crossovers: ${bestGraph.metadata.crossovers}`);
  console.log();

  // Evaluate final metrics
  const evaluator = new FitnessEvaluator(config);
  const metrics = evaluator.evaluate(bestGraph);

  console.log("Final Fitness Metrics:");
  console.log(`  Latency: ${metrics.latency.toFixed(2)} ms`);
  console.log(`  Throughput: ${metrics.throughput.toFixed(2)} ops/s`);
  console.log(`  Cost: ${metrics.cost.toFixed(2)} units`);
  console.log(`  Resilience: ${metrics.resilience.toFixed(4)}`);
  console.log(`  Emergence: ${metrics.emergence.toFixed(4)}`);
  console.log(`  Overall: ${metrics.overall.toFixed(4)}`);
  console.log();

  console.log("=".repeat(70));
  console.log("MORPHOGENIC ORCHESTRATION POC COMPLETE");
  console.log("=".repeat(70));
}

// Export all components
export default {
  MorphogenicGraphBuilder,
  FitnessEvaluator,
  EvolutionEngine,
  demonstrateMorphogenicOrchestration,
};
