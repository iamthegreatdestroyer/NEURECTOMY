/**
 * Causal Reasoning Engine - Proof of Concept
 *
 * Implements causal graphs, structural causal models, do-calculus,
 * and counterfactual reasoning.
 *
 * @module causal-reasoning
 * @agents @AXIOM @PRISM @APEX
 */

import { create, all, MathJsInstance } from "mathjs";

const math: MathJsInstance = create(all);

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export type VariableName = string;
export type Value = number | string | boolean;

/**
 * Causal graph (Directed Acyclic Graph)
 */
export interface CausalGraph {
  /** Set of variables */
  variables: Set<VariableName>;

  /** Directed edges: parent → child */
  edges: Map<VariableName, Set<VariableName>>;

  /** Reverse index: child → parents */
  parents: Map<VariableName, Set<VariableName>>;
}

/**
 * Structural equation
 * X := f(PA_X, U_X)
 */
export interface StructuralEquation {
  variable: VariableName;

  /** Parent variables */
  parents: VariableName[];

  /** Function: parents → value */
  mechanism: (...parentValues: Value[]) => Value;

  /** Exogenous noise variable */
  noise?: () => Value;
}

/**
 * Structural Causal Model (SCM)
 */
export interface StructuralCausalModel {
  graph: CausalGraph;
  equations: Map<VariableName, StructuralEquation>;

  /** Observational distribution P(V) */
  observational: Map<VariableName, Value>;

  /** Interventional distributions P(V | do(X=x)) */
  interventional: Map<string, Map<VariableName, Value>>;
}

/**
 * Conditional independence test
 */
export interface ConditionalIndependence {
  X: VariableName;
  Y: VariableName;
  Z: Set<VariableName>; // Conditioning set
  isIndependent: boolean;
}

/**
 * Counterfactual query
 * "What if X had been x, would Y have been y?"
 */
export interface CounterfactualQuery {
  /** Intervention: do(X=x) */
  intervention: { variable: VariableName; value: Value };

  /** Query: Y=y? */
  query: { variable: VariableName; value?: Value };

  /** Evidence: observed values before intervention */
  evidence: Map<VariableName, Value>;
}

/**
 * Counterfactual result
 */
export interface CounterfactualResult {
  /** Counterfactual value of query variable */
  value: Value;

  /** Probability (for stochastic SCMs) */
  probability: number;

  /** Full counterfactual world */
  counterfactualWorld: Map<VariableName, Value>;
}

// ============================================================================
// CAUSAL GRAPH BUILDER
// ============================================================================

/**
 * Build and manipulate causal graphs
 */
export class CausalGraphBuilder {
  private graph: CausalGraph;

  constructor() {
    this.graph = {
      variables: new Set(),
      edges: new Map(),
      parents: new Map(),
    };
  }

  /**
   * Add variable to graph
   */
  addVariable(name: VariableName): this {
    this.graph.variables.add(name);
    if (!this.graph.edges.has(name)) {
      this.graph.edges.set(name, new Set());
    }
    if (!this.graph.parents.has(name)) {
      this.graph.parents.set(name, new Set());
    }
    return this;
  }

  /**
   * Add causal edge: parent → child
   */
  addEdge(parent: VariableName, child: VariableName): this {
    // Ensure variables exist
    this.addVariable(parent);
    this.addVariable(child);

    // Check for cycles
    if (this.wouldCreateCycle(parent, child)) {
      throw new Error(`Adding edge ${parent}→${child} would create a cycle`);
    }

    // Add edge
    this.graph.edges.get(parent)!.add(child);
    this.graph.parents.get(child)!.add(parent);

    return this;
  }

  /**
   * Check if adding edge would create cycle
   */
  private wouldCreateCycle(from: VariableName, to: VariableName): boolean {
    // DFS from 'to' to see if we can reach 'from'
    const visited = new Set<VariableName>();
    const stack = [to];

    while (stack.length > 0) {
      const current = stack.pop()!;
      if (current === from) return true;

      if (visited.has(current)) continue;
      visited.add(current);

      const children = this.graph.edges.get(current);
      if (children) {
        stack.push(...children);
      }
    }

    return false;
  }

  /**
   * Get parents of variable
   */
  getParents(variable: VariableName): Set<VariableName> {
    return this.graph.parents.get(variable) || new Set();
  }

  /**
   * Get children of variable
   */
  getChildren(variable: VariableName): Set<VariableName> {
    return this.graph.edges.get(variable) || new Set();
  }

  /**
   * Get ancestors (transitive closure of parents)
   */
  getAncestors(variable: VariableName): Set<VariableName> {
    const ancestors = new Set<VariableName>();
    const stack = [variable];

    while (stack.length > 0) {
      const current = stack.pop()!;
      const parents = this.getParents(current);

      for (const parent of parents) {
        if (!ancestors.has(parent)) {
          ancestors.add(parent);
          stack.push(parent);
        }
      }
    }

    return ancestors;
  }

  /**
   * Get descendants (transitive closure of children)
   */
  getDescendants(variable: VariableName): Set<VariableName> {
    const descendants = new Set<VariableName>();
    const stack = [variable];

    while (stack.length > 0) {
      const current = stack.pop()!;
      const children = this.getChildren(current);

      for (const child of children) {
        if (!descendants.has(child)) {
          descendants.add(child);
          stack.push(child);
        }
      }
    }

    return descendants;
  }

  /**
   * Check d-separation: X ⊥ Y | Z
   *
   * Two variables X and Y are d-separated by Z if
   * all paths between X and Y are blocked by Z
   */
  dSeparated(X: VariableName, Y: VariableName, Z: Set<VariableName>): boolean {
    // Simplified d-separation check (full implementation is complex)
    // Rule: If Z contains all variables on all paths from X to Y, then d-separated

    // Find all paths from X to Y
    const paths = this.findAllPaths(X, Y);

    // Check if all paths are blocked by Z
    for (const path of paths) {
      const isBlocked = path.some((node) => Z.has(node));
      if (!isBlocked) {
        return false; // Found unblocked path
      }
    }

    return true; // All paths blocked
  }

  /**
   * Find all paths between two variables (simple BFS)
   */
  private findAllPaths(
    start: VariableName,
    end: VariableName
  ): VariableName[][] {
    const paths: VariableName[][] = [];
    const queue: { node: VariableName; path: VariableName[] }[] = [
      { node: start, path: [start] },
    ];

    while (queue.length > 0) {
      const { node, path } = queue.shift()!;

      if (node === end) {
        paths.push(path);
        continue;
      }

      // Explore children
      const children = this.getChildren(node);
      for (const child of children) {
        if (!path.includes(child)) {
          queue.push({ node: child, path: [...path, child] });
        }
      }
    }

    return paths;
  }

  /**
   * Build graph
   */
  build(): CausalGraph {
    return this.graph;
  }

  /**
   * Topological sort (for causal ordering)
   */
  topologicalSort(): VariableName[] {
    const sorted: VariableName[] = [];
    const visited = new Set<VariableName>();
    const temp = new Set<VariableName>();

    const visit = (node: VariableName) => {
      if (temp.has(node)) {
        throw new Error("Graph has cycle");
      }
      if (visited.has(node)) return;

      temp.add(node);

      const children = this.getChildren(node);
      for (const child of children) {
        visit(child);
      }

      temp.delete(node);
      visited.add(node);
      sorted.unshift(node); // Add to beginning
    };

    for (const variable of this.graph.variables) {
      if (!visited.has(variable)) {
        visit(variable);
      }
    }

    return sorted;
  }
}

// ============================================================================
// STRUCTURAL CAUSAL MODEL
// ============================================================================

/**
 * Structural Causal Model implementation
 */
export class SCM {
  private model: StructuralCausalModel;
  private graphBuilder: CausalGraphBuilder;

  constructor() {
    this.graphBuilder = new CausalGraphBuilder();
    this.model = {
      graph: this.graphBuilder.build(),
      equations: new Map(),
      observational: new Map(),
      interventional: new Map(),
    };
  }

  /**
   * Add structural equation
   */
  addEquation(equation: StructuralEquation): this {
    // Add variable and edges to graph
    this.graphBuilder.addVariable(equation.variable);

    for (const parent of equation.parents) {
      this.graphBuilder.addEdge(parent, equation.variable);
    }

    this.model.graph = this.graphBuilder.build();
    this.model.equations.set(equation.variable, equation);

    return this;
  }

  /**
   * Observe (sample from observational distribution)
   */
  observe(): Map<VariableName, Value> {
    const values = new Map<VariableName, Value>();

    // Evaluate equations in topological order
    const order = this.graphBuilder.topologicalSort();

    for (const variable of order) {
      const equation = this.model.equations.get(variable);
      if (!equation) continue;

      // Get parent values
      const parentValues = equation.parents.map((p) => values.get(p)!);

      // Apply mechanism
      let value = equation.mechanism(...parentValues);

      // Add noise if present
      if (equation.noise) {
        const noiseValue = equation.noise();
        if (typeof value === "number" && typeof noiseValue === "number") {
          value = value + noiseValue;
        }
      }

      values.set(variable, value);
    }

    this.model.observational = values;
    return values;
  }

  /**
   * Intervene: do(X=x)
   *
   * Intervention mutilates the graph by removing incoming edges to X
   * and replacing X's mechanism with constant function f_X(·) = x
   */
  intervene(variable: VariableName, value: Value): Map<VariableName, Value> {
    const values = new Map<VariableName, Value>();

    // Set intervention target
    values.set(variable, value);

    // Evaluate equations in topological order, skipping intervened variable
    const order = this.graphBuilder.topologicalSort();

    for (const v of order) {
      if (v === variable) continue; // Skip intervened variable

      const equation = this.model.equations.get(v);
      if (!equation) continue;

      // Get parent values (including intervened value if parent)
      const parentValues = equation.parents.map((p) => values.get(p)!);

      // Apply mechanism
      let val = equation.mechanism(...parentValues);

      // Add noise if present
      if (equation.noise) {
        const noiseValue = equation.noise();
        if (typeof val === "number" && typeof noiseValue === "number") {
          val = val + noiseValue;
        }
      }

      values.set(v, val);
    }

    // Cache interventional distribution
    const interventionKey = `do(${variable}=${value})`;
    this.model.interventional.set(interventionKey, values);

    return values;
  }

  /**
   * Compute Average Treatment Effect (ATE)
   * ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
   */
  averageTreatmentEffect(
    treatment: VariableName,
    outcome: VariableName,
    numSamples: number = 1000
  ): number {
    let sum0 = 0;
    let sum1 = 0;

    for (let i = 0; i < numSamples; i++) {
      // Intervene with treatment = 0
      const values0 = this.intervene(treatment, 0);
      sum0 += values0.get(outcome) as number;

      // Intervene with treatment = 1
      const values1 = this.intervene(treatment, 1);
      sum1 += values1.get(outcome) as number;
    }

    const mean0 = sum0 / numSamples;
    const mean1 = sum1 / numSamples;

    return mean1 - mean0;
  }

  /**
   * Get graph for visualization
   */
  getGraph(): CausalGraph {
    return this.model.graph;
  }
}

// ============================================================================
// COUNTERFACTUAL ENGINE
// ============================================================================

/**
 * Counterfactual reasoning using three-step process:
 * 1. Abduction: Infer exogenous variables from evidence
 * 2. Action: Perform intervention
 * 3. Prediction: Compute outcome in counterfactual world
 */
export class CounterfactualEngine {
  private scm: SCM;

  constructor(scm: SCM) {
    this.scm = scm;
  }

  /**
   * Answer counterfactual query
   *
   * Query: "Given evidence E, what would Y be if we had done X=x?"
   */
  query(query: CounterfactualQuery): CounterfactualResult {
    // Step 1: Abduction - infer latent variables from evidence
    // (Simplified: assume evidence directly observed)

    // Step 2: Action - perform intervention
    const counterfactualWorld = this.scm.intervene(
      query.intervention.variable,
      query.intervention.value
    );

    // Step 3: Prediction - read off query variable
    const value = counterfactualWorld.get(query.query.variable)!;

    return {
      value,
      probability: 1.0, // Deterministic for POC
      counterfactualWorld,
    };
  }

  /**
   * Compute Probability of Necessity (PN)
   * PN = P(Y_0 = 0 | X = 1, Y = 1)
   * "Was X necessary for Y?"
   */
  probabilityOfNecessity(
    cause: VariableName,
    effect: VariableName,
    numSamples: number = 1000
  ): number {
    let necessaryCount = 0;
    let relevantCount = 0;

    for (let i = 0; i < numSamples; i++) {
      // Observe factual world where X=1, Y=1
      const factual = this.scm.observe();

      if (factual.get(cause) === 1 && factual.get(effect) === 1) {
        relevantCount++;

        // Counterfactual: what if X had been 0?
        const counterfactual = this.scm.intervene(cause, 0);

        // Was Y=0 in counterfactual world?
        if (counterfactual.get(effect) === 0) {
          necessaryCount++;
        }
      }
    }

    return relevantCount > 0 ? necessaryCount / relevantCount : 0;
  }

  /**
   * Compute Probability of Sufficiency (PS)
   * PS = P(Y_1 = 1 | X = 0, Y = 0)
   * "Was X sufficient for Y?"
   */
  probabilityOfSufficiency(
    cause: VariableName,
    effect: VariableName,
    numSamples: number = 1000
  ): number {
    let sufficientCount = 0;
    let relevantCount = 0;

    for (let i = 0; i < numSamples; i++) {
      // Observe factual world where X=0, Y=0
      const factual = this.scm.observe();

      if (factual.get(cause) === 0 && factual.get(effect) === 0) {
        relevantCount++;

        // Counterfactual: what if X had been 1?
        const counterfactual = this.scm.intervene(cause, 1);

        // Was Y=1 in counterfactual world?
        if (counterfactual.get(effect) === 1) {
          sufficientCount++;
        }
      }
    }

    return relevantCount > 0 ? sufficientCount / relevantCount : 0;
  }
}

// ============================================================================
// DO-CALCULUS
// ============================================================================

/**
 * do-Calculus rules (Pearl 1995)
 *
 * Rules for transforming interventional distributions:
 * 1. Insertion/deletion of observations
 * 2. Action/observation exchange
 * 3. Insertion/deletion of actions
 */
export class DoCalculus {
  private graphBuilder: CausalGraphBuilder;

  constructor(graph: CausalGraph) {
    this.graphBuilder = new CausalGraphBuilder();
    // Copy graph
    for (const v of graph.variables) {
      this.graphBuilder.addVariable(v);
    }
    for (const [parent, children] of graph.edges) {
      for (const child of children) {
        this.graphBuilder.addEdge(parent, child);
      }
    }
  }

  /**
   * Rule 1: Insertion/deletion of observations
   * P(Y | do(X), Z, W) = P(Y | do(X), W) if (Y ⊥ Z | X, W) in G_X̄
   */
  rule1(
    Y: VariableName,
    X: VariableName,
    Z: Set<VariableName>,
    W: Set<VariableName>
  ): boolean {
    // Check d-separation in G with X's incoming edges removed
    const conditioning = new Set([X, ...W]);
    return this.graphBuilder.dSeparated(Y, Array.from(Z)[0], conditioning);
  }

  /**
   * Rule 2: Action/observation exchange
   * P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if (Y ⊥ Z | X, W) in G_X̄Z
   */
  rule2(
    Y: VariableName,
    X: VariableName,
    Z: Set<VariableName>,
    W: Set<VariableName>
  ): boolean {
    const conditioning = new Set([X, ...W]);
    return this.graphBuilder.dSeparated(Y, Array.from(Z)[0], conditioning);
  }

  /**
   * Rule 3: Insertion/deletion of actions
   * P(Y | do(X), do(Z), W) = P(Y | do(X), W) if (Y ⊥ Z | X, W) in G_X̄,Z(W)
   */
  rule3(
    Y: VariableName,
    X: VariableName,
    Z: Set<VariableName>,
    W: Set<VariableName>
  ): boolean {
    const conditioning = new Set([X, ...W]);
    return this.graphBuilder.dSeparated(Y, Array.from(Z)[0], conditioning);
  }

  /**
   * Check if causal effect P(Y | do(X)) is identifiable
   * (can be computed from observational data)
   */
  isIdentifiable(Y: VariableName, X: VariableName): boolean {
    // Simplified: effect is identifiable if no confounders
    const ancestors = this.graphBuilder.getAncestors(Y);
    const xAncestors = this.graphBuilder.getAncestors(X);

    // Check for backdoor path
    const backdoorPath = [...xAncestors].some(
      (a) => ancestors.has(a) && a !== X
    );

    return !backdoorPath; // Identifiable if no backdoor path
  }
}

// ============================================================================
// DEMO & TESTING
// ============================================================================

/**
 * Demonstration of causal reasoning
 */
export async function demonstrateCausalReasoning(): Promise<void> {
  console.log("=".repeat(70));
  console.log("CAUSAL REASONING ENGINE - PROOF OF CONCEPT");
  console.log("=".repeat(70));
  console.log();

  // Demo 1: Simple causal model (X → Y)
  console.log("Demo 1: Simple Causal Chain");
  console.log("-".repeat(70));

  const scm = new SCM();

  // X → Y → Z
  scm.addEquation({
    variable: "X",
    parents: [],
    mechanism: () => (Math.random() > 0.5 ? 1 : 0),
  });

  scm.addEquation({
    variable: "Y",
    parents: ["X"],
    mechanism: (x: Value) => (x as number) * 2 + 1,
  });

  scm.addEquation({
    variable: "Z",
    parents: ["Y"],
    mechanism: (y: Value) => (y as number) + Math.random(),
  });

  // Observational data
  console.log("Observational distribution P(X, Y, Z):");
  for (let i = 0; i < 3; i++) {
    const obs = scm.observe();
    console.log(
      `  Sample ${i + 1}: X=${obs.get("X")}, Y=${obs.get("Y")}, Z=${obs.get("Z")?.toFixed(2)}`
    );
  }
  console.log();

  // Interventional data
  console.log("Interventional distribution P(Y, Z | do(X=1)):");
  for (let i = 0; i < 3; i++) {
    const intervention = scm.intervene("X", 1);
    console.log(
      `  Sample ${i + 1}: Y=${intervention.get("Y")}, Z=${intervention.get("Z")?.toFixed(2)}`
    );
  }
  console.log();

  // Demo 2: Confounding
  console.log("Demo 2: Confounding (Simpson's Paradox)");
  console.log("-".repeat(70));

  const confoundedSCM = new SCM();

  // Confounder U → X and U → Y
  confoundedSCM.addEquation({
    variable: "U",
    parents: [],
    mechanism: () => (Math.random() > 0.5 ? 1 : 0),
  });

  confoundedSCM.addEquation({
    variable: "X",
    parents: ["U"],
    mechanism: (u: Value) => u as number,
  });

  confoundedSCM.addEquation({
    variable: "Y",
    parents: ["U", "X"],
    mechanism: (u: Value, x: Value) => (u as number) * 5 - (x as number) * 2,
  });

  // Average Treatment Effect
  const ate = confoundedSCM.averageTreatmentEffect("X", "Y", 1000);
  console.log(`Average Treatment Effect (ATE): ${ate.toFixed(3)}`);
  console.log("Interpretation: X has negative causal effect on Y");
  console.log("(even though they may be positively correlated!)");
  console.log();

  // Demo 3: Counterfactuals
  console.log("Demo 3: Counterfactual Reasoning");
  console.log("-".repeat(70));

  const cfEngine = new CounterfactualEngine(scm);

  const cfQuery: CounterfactualQuery = {
    intervention: { variable: "X", value: 0 },
    query: { variable: "Z" },
    evidence: new Map([
      ["X", 1],
      ["Y", 3],
    ]),
  };

  const cfResult = cfEngine.query(cfQuery);
  console.log('Counterfactual query: "What would Z be if X had been 0?"');
  console.log(`Evidence: X=1, Y=3`);
  console.log(`Counterfactual Z: ${cfResult.value}`);
  console.log();

  // Demo 4: Causal necessity/sufficiency
  console.log("Demo 4: Causal Necessity and Sufficiency");
  console.log("-".repeat(70));

  const pn = cfEngine.probabilityOfNecessity("X", "Y", 1000);
  const ps = cfEngine.probabilityOfSufficiency("X", "Y", 1000);

  console.log(`Probability of Necessity: ${pn.toFixed(3)}`);
  console.log("Interpretation: P(X was necessary for Y)");
  console.log();
  console.log(`Probability of Sufficiency: ${ps.toFixed(3)}`);
  console.log("Interpretation: P(X was sufficient for Y)");
  console.log();

  console.log("=".repeat(70));
  console.log("CAUSAL REASONING POC COMPLETE");
  console.log("=".repeat(70));
}

// Export all components
export default {
  CausalGraphBuilder,
  SCM,
  CounterfactualEngine,
  DoCalculus,
  demonstrateCausalReasoning,
};
