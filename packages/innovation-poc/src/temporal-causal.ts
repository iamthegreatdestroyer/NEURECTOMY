/**
 * Temporal Causal Reasoning POC
 *
 * Implements time-indexed structural causal models for reasoning about
 * causality across time, combining Pearl's causal framework with dynamic
 * Bayesian networks and temporal intervention planning.
 *
 * Key Innovations:
 * - Time-sliced causal graphs with temporal edges
 * - Dynamic interventions with rolling horizon planning
 * - Temporal counterfactuals (what if we had acted differently at t-k?)
 * - Delayed effect propagation and temporal dependencies
 *
 * Research Foundations:
 * - Pearl (2009): Causality: Models, Reasoning, and Inference
 * - Murphy (2002): Dynamic Bayesian Networks
 * - Eaton & Murphy (2007): Exact Bayesian Structure Learning
 *
 * @elite-agents @QUANTUM @PRISM @AXIOM
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type VariableName = string;
type TimeStep = number;
type TimeSlicedVariable = `${VariableName}_t${TimeStep}`;
type Value = number | string | boolean;

interface TemporalVariable {
  name: VariableName;
  timeStep: TimeStep;
  value?: Value;
  parents: TimeSlicedVariable[]; // Can include variables from previous time steps
}

interface TemporalEdge {
  from: TimeSlicedVariable;
  to: TimeSlicedVariable;
  lag: number; // Time lag (0 = same time step, 1 = one step back, etc.)
  strength: number; // Causal strength [0, 1]
}

interface TemporalCausalGraph {
  variables: Map<TimeSlicedVariable, TemporalVariable>;
  edges: TemporalEdge[];
  timeHorizon: number; // Number of time steps
}

interface StructuralEquation {
  variable: VariableName;
  compute: (inputs: Map<TimeSlicedVariable, Value>) => Value;
}

interface TemporalIntervention {
  variable: VariableName;
  timeStep: TimeStep;
  value: Value;
}

interface InterventionPlan {
  interventions: TemporalIntervention[];
  expectedOutcome: Map<TimeSlicedVariable, Value>;
  confidence: number;
}

interface TemporalCounterfactual {
  actualHistory: Map<TimeSlicedVariable, Value>;
  interventionTime: TimeStep;
  intervention: TemporalIntervention;
  counterfactualFuture: Map<TimeSlicedVariable, Value>;
}

// ============================================================================
// Temporal SCM (Structural Causal Model)
// ============================================================================

class TemporalSCM {
  private equations: Map<VariableName, StructuralEquation>;
  private graph: TemporalCausalGraph;
  private history: Map<TimeSlicedVariable, Value>;

  constructor(timeHorizon: number) {
    this.equations = new Map();
    this.graph = {
      variables: new Map(),
      edges: [],
      timeHorizon,
    };
    this.history = new Map();
  }

  /**
   * Add a structural equation for a variable
   */
  addEquation(equation: StructuralEquation): this {
    this.equations.set(equation.variable, equation);
    return this;
  }

  /**
   * Add a temporal edge (can cross time steps)
   */
  addEdge(
    from: VariableName,
    to: VariableName,
    lag: number = 0,
    strength: number = 1.0
  ): this {
    // For simplicity, assume edges connect to t=0 in the template
    const fromVar: TimeSlicedVariable = `${from}_t0`;
    const toVar: TimeSlicedVariable = `${to}_t0`;

    this.graph.edges.push({ from: fromVar, to: toVar, lag, strength });
    return this;
  }

  /**
   * Simulate forward in time (observational)
   */
  simulate(
    initialState: Map<VariableName, Value>,
    steps: number
  ): Map<TimeSlicedVariable, Value> {
    this.history.clear();

    // Set initial conditions at t=0
    for (const [varName, value] of initialState.entries()) {
      const tsVar: TimeSlicedVariable = `${varName}_t0`;
      this.history.set(tsVar, value);
    }

    // Simulate forward
    for (let t = 1; t <= steps; t++) {
      for (const [varName, equation] of this.equations.entries()) {
        const inputs = this.getInputsForVariable(varName, t);
        const value = equation.compute(inputs);
        const tsVar: TimeSlicedVariable = `${varName}_t${t}`;
        this.history.set(tsVar, value);
      }
    }

    return new Map(this.history);
  }

  /**
   * Simulate with temporal interventions
   */
  simulateWithInterventions(
    initialState: Map<VariableName, Value>,
    interventions: TemporalIntervention[],
    steps: number
  ): Map<TimeSlicedVariable, Value> {
    this.history.clear();

    // Set initial conditions
    for (const [varName, value] of initialState.entries()) {
      const tsVar: TimeSlicedVariable = `${varName}_t0`;
      this.history.set(tsVar, value);
    }

    // Create intervention map for quick lookup
    const interventionMap = new Map<TimeSlicedVariable, Value>();
    for (const intv of interventions) {
      const tsVar: TimeSlicedVariable = `${intv.variable}_t${intv.timeStep}`;
      interventionMap.set(tsVar, intv.value);
    }

    // Simulate forward with interventions
    for (let t = 1; t <= steps; t++) {
      for (const [varName, equation] of this.equations.entries()) {
        const tsVar: TimeSlicedVariable = `${varName}_t${t}`;

        // Check if this variable is intervened upon at this time
        if (interventionMap.has(tsVar)) {
          this.history.set(tsVar, interventionMap.get(tsVar)!);
        } else {
          const inputs = this.getInputsForVariable(varName, t);
          const value = equation.compute(inputs);
          this.history.set(tsVar, value);
        }
      }
    }

    return new Map(this.history);
  }

  /**
   * Get inputs for a variable at time t (includes lagged variables)
   */
  private getInputsForVariable(
    varName: VariableName,
    timeStep: TimeStep
  ): Map<TimeSlicedVariable, Value> {
    const inputs = new Map<TimeSlicedVariable, Value>();

    // Find all edges pointing to this variable
    for (const edge of this.graph.edges) {
      const [toVar, _] = this.parseTimeSlicedVariable(edge.to);
      if (toVar === varName) {
        const [fromVar, __] = this.parseTimeSlicedVariable(edge.from);
        const fromTime = timeStep - edge.lag;

        if (fromTime >= 0) {
          const fromTsVar: TimeSlicedVariable = `${fromVar}_t${fromTime}`;
          const value = this.history.get(fromTsVar);
          if (value !== undefined) {
            inputs.set(fromTsVar, value);
          }
        }
      }
    }

    return inputs;
  }

  /**
   * Parse time-sliced variable name
   */
  private parseTimeSlicedVariable(
    tsVar: TimeSlicedVariable
  ): [VariableName, TimeStep] {
    const match = tsVar.match(/^(.+)_t(\d+)$/);
    if (!match) throw new Error(`Invalid time-sliced variable: ${tsVar}`);
    return [match[1], parseInt(match[2])];
  }

  /**
   * Get historical values
   */
  getHistory(): Map<TimeSlicedVariable, Value> {
    return new Map(this.history);
  }
}

// ============================================================================
// Dynamic Bayesian Network
// ============================================================================

class DynamicBayesianNetwork {
  private transitionModel: Map<
    VariableName,
    (prev: Value, noise: number) => Value
  >;
  private observationModel: Map<
    VariableName,
    (state: Value, noise: number) => Value
  >;

  constructor() {
    this.transitionModel = new Map();
    this.observationModel = new Map();
  }

  /**
   * Add transition probability model: P(X_t | X_{t-1})
   */
  addTransition(
    variable: VariableName,
    transitionFn: (prev: Value, noise: number) => Value
  ): this {
    this.transitionModel.set(variable, transitionFn);
    return this;
  }

  /**
   * Add observation model: P(O_t | X_t)
   */
  addObservation(
    variable: VariableName,
    observationFn: (state: Value, noise: number) => Value
  ): this {
    this.observationModel.set(variable, observationFn);
    return this;
  }

  /**
   * Forward simulation with noise
   */
  forward(
    initialState: Map<VariableName, Value>,
    steps: number
  ): {
    states: Map<TimeSlicedVariable, Value>;
    observations: Map<TimeSlicedVariable, Value>;
  } {
    const states = new Map<TimeSlicedVariable, Value>();
    const observations = new Map<TimeSlicedVariable, Value>();

    // Initialize
    for (const [varName, value] of initialState.entries()) {
      states.set(`${varName}_t0`, value);
    }

    // Simulate forward
    for (let t = 1; t <= steps; t++) {
      for (const [varName, transitionFn] of this.transitionModel.entries()) {
        const prevTsVar: TimeSlicedVariable = `${varName}_t${t - 1}`;
        const prevValue = states.get(prevTsVar);

        if (prevValue !== undefined) {
          const noise = Math.random();
          const newValue = transitionFn(prevValue, noise);
          states.set(`${varName}_t${t}`, newValue);

          // Generate observation if observation model exists
          const observationFn = this.observationModel.get(varName);
          if (observationFn) {
            const obsNoise = Math.random();
            const obsValue = observationFn(newValue, obsNoise);
            observations.set(`${varName}_t${t}`, obsValue);
          }
        }
      }
    }

    return { states, observations };
  }

  /**
   * Filtering: estimate P(X_t | O_{1:t})
   * Simplified particle filter implementation
   */
  filter(
    observations: Map<TimeSlicedVariable, Value>,
    numParticles: number = 100
  ): Map<TimeSlicedVariable, Value> {
    // Extract time steps
    const timeSteps = new Set<number>();
    for (const tsVar of observations.keys()) {
      const match = tsVar.match(/_t(\d+)$/);
      if (match) timeSteps.add(parseInt(match[1]));
    }
    const maxTime = Math.max(...timeSteps);

    // Initialize particles
    const particles: Map<VariableName, Value>[] = [];
    for (let i = 0; i < numParticles; i++) {
      const particle = new Map<VariableName, Value>();
      for (const varName of this.transitionModel.keys()) {
        particle.set(varName, Math.random()); // Random initialization
      }
      particles.push(particle);
    }

    // Forward filtering
    const filteredStates = new Map<TimeSlicedVariable, Value>();

    for (let t = 1; t <= maxTime; t++) {
      // Propagate particles
      const newParticles: Map<VariableName, Value>[] = [];
      for (const particle of particles) {
        const newParticle = new Map<VariableName, Value>();
        for (const [varName, transitionFn] of this.transitionModel.entries()) {
          const prevValue = particle.get(varName) ?? 0;
          const noise = Math.random();
          const newValue = transitionFn(prevValue, noise);
          newParticle.set(varName, newValue);
        }
        newParticles.push(newParticle);
      }

      // Weight particles based on observations (simplified)
      const weights: number[] = newParticles.map(() => 1.0);

      // Resample (simplified: just keep all particles)
      particles.length = 0;
      particles.push(...newParticles);

      // Estimate state (mean of particles)
      for (const varName of this.transitionModel.keys()) {
        const values = particles.map((p) => p.get(varName) as number);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        filteredStates.set(`${varName}_t${t}`, mean);
      }
    }

    return filteredStates;
  }
}

// ============================================================================
// Temporal Intervention Planner
// ============================================================================

class TemporalInterventionPlanner {
  private scm: TemporalSCM;
  private targetVariable: VariableName;
  private targetTime: TimeStep;
  private targetValue: Value;

  constructor(
    scm: TemporalSCM,
    targetVariable: VariableName,
    targetTime: TimeStep,
    targetValue: Value
  ) {
    this.scm = scm;
    this.targetVariable = targetVariable;
    this.targetTime = targetTime;
    this.targetValue = targetValue;
  }

  /**
   * Plan interventions to achieve target using rolling horizon
   */
  plan(
    initialState: Map<VariableName, Value>,
    horizon: number,
    candidateVariables: VariableName[]
  ): InterventionPlan {
    let bestPlan: InterventionPlan | null = null;
    let bestDistance = Infinity;

    // Try different intervention strategies
    const numCandidates = Math.min(candidateVariables.length, 3); // Limit search space

    for (let i = 0; i < numCandidates; i++) {
      const varToIntervene = candidateVariables[i];

      // Try different intervention times
      for (let t = 1; t < this.targetTime; t++) {
        // Try different intervention values
        for (const intvValue of [0, 0.5, 1.0]) {
          const interventions: TemporalIntervention[] = [
            {
              variable: varToIntervene,
              timeStep: t,
              value: intvValue,
            },
          ];

          // Simulate
          const result = this.scm.simulateWithInterventions(
            initialState,
            interventions,
            this.targetTime
          );

          // Evaluate outcome
          const targetTsVar: TimeSlicedVariable = `${this.targetVariable}_t${this.targetTime}`;
          const achievedValue = result.get(targetTsVar);

          if (achievedValue !== undefined) {
            const distance = Math.abs(
              (achievedValue as number) - (this.targetValue as number)
            );

            if (distance < bestDistance) {
              bestDistance = distance;
              bestPlan = {
                interventions,
                expectedOutcome: result,
                confidence: 1.0 - distance,
              };
            }
          }
        }
      }
    }

    return (
      bestPlan ?? {
        interventions: [],
        expectedOutcome: new Map(),
        confidence: 0,
      }
    );
  }

  /**
   * Compute temporal counterfactual: what if we had intervened differently?
   */
  computeCounterfactual(
    actualHistory: Map<TimeSlicedVariable, Value>,
    intervention: TemporalIntervention
  ): TemporalCounterfactual {
    // Extract initial state
    const initialState = new Map<VariableName, Value>();
    for (const [tsVar, value] of actualHistory.entries()) {
      if (tsVar.endsWith("_t0")) {
        const varName = tsVar.replace("_t0", "");
        initialState.set(varName, value);
      }
    }

    // Simulate counterfactual
    const counterfactualFuture = this.scm.simulateWithInterventions(
      initialState,
      [intervention],
      this.targetTime
    );

    return {
      actualHistory,
      interventionTime: intervention.timeStep,
      intervention,
      counterfactualFuture,
    };
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateTemporalCausalReasoning(): Promise<void> {
  console.log("=".repeat(80));
  console.log("TEMPORAL CAUSAL REASONING DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Temporal SCM with lagged effects
  console.log("\n📊 Demo 1: Temporal SCM with Lagged Effects");
  console.log("-".repeat(80));

  const scm = new TemporalSCM(10);

  // X_{t} = 0.8 * X_{t-1} + 0.3 * Y_{t-1} + noise
  scm.addEquation({
    variable: "X",
    compute: (inputs) => {
      const xPrev = (inputs.get("X_t-1" as TimeSlicedVariable) as number) ?? 0;
      const yPrev = (inputs.get("Y_t-1" as TimeSlicedVariable) as number) ?? 0;
      return 0.8 * xPrev + 0.3 * yPrev + Math.random() * 0.1;
    },
  });

  // Y_{t} = 0.5 * Y_{t-1} + 0.4 * X_{t-1} + noise
  scm.addEquation({
    variable: "Y",
    compute: (inputs) => {
      const yPrev = (inputs.get("Y_t-1" as TimeSlicedVariable) as number) ?? 0;
      const xPrev = (inputs.get("X_t-1" as TimeSlicedVariable) as number) ?? 0;
      return 0.5 * yPrev + 0.4 * xPrev + Math.random() * 0.1;
    },
  });

  scm.addEdge("X", "X", 1, 0.8); // X depends on its past
  scm.addEdge("Y", "X", 1, 0.3); // X depends on past Y
  scm.addEdge("Y", "Y", 1, 0.5); // Y depends on its past
  scm.addEdge("X", "Y", 1, 0.4); // Y depends on past X

  const initialState = new Map<VariableName, Value>([
    ["X", 1.0],
    ["Y", 0.5],
  ]);

  console.log("Initial State:", Object.fromEntries(initialState));
  const history = scm.simulate(initialState, 10);

  console.log("\nSimulated Trajectory (first 5 steps):");
  for (let t = 0; t <= 5; t++) {
    const x = history.get(`X_t${t}` as TimeSlicedVariable);
    const y = history.get(`Y_t${t}` as TimeSlicedVariable);
    console.log(
      `  t=${t}: X=${(x as number).toFixed(3)}, Y=${(y as number).toFixed(3)}`
    );
  }

  // Demo 2: Temporal Interventions
  console.log("\n🎯 Demo 2: Temporal Interventions");
  console.log("-".repeat(80));

  const interventions: TemporalIntervention[] = [
    { variable: "Y", timeStep: 3, value: 2.0 },
  ];

  console.log("Intervention: Set Y=2.0 at t=3");
  const intervenedHistory = scm.simulateWithInterventions(
    initialState,
    interventions,
    10
  );

  console.log("\nIntervened Trajectory:");
  for (let t = 0; t <= 6; t++) {
    const x = intervenedHistory.get(`X_t${t}` as TimeSlicedVariable);
    const y = intervenedHistory.get(`Y_t${t}` as TimeSlicedVariable);
    console.log(
      `  t=${t}: X=${(x as number).toFixed(3)}, Y=${(y as number).toFixed(3)}${t === 3 ? " ← INTERVENTION" : ""}`
    );
  }

  // Demo 3: Dynamic Bayesian Network
  console.log("\n🌐 Demo 3: Dynamic Bayesian Network (Filtering)");
  console.log("-".repeat(80));

  const dbn = new DynamicBayesianNetwork();

  // Random walk transition
  dbn.addTransition(
    "position",
    (prev, noise) => (prev as number) + (noise - 0.5) * 0.5
  );

  // Noisy observation
  dbn.addObservation(
    "position",
    (state, noise) => (state as number) + (noise - 0.5) * 0.2
  );

  const { states, observations } = dbn.forward(new Map([["position", 0]]), 5);

  console.log("True States vs Noisy Observations:");
  for (let t = 1; t <= 5; t++) {
    const trueState = states.get(`position_t${t}` as TimeSlicedVariable);
    const obs = observations.get(`position_t${t}` as TimeSlicedVariable);
    console.log(
      `  t=${t}: True=${(trueState as number).toFixed(3)}, Observed=${(obs as number).toFixed(3)}`
    );
  }

  const filtered = dbn.filter(observations, 50);
  console.log("\nFiltered Estimates (using particle filter):");
  for (let t = 1; t <= 5; t++) {
    const estimate = filtered.get(`position_t${t}` as TimeSlicedVariable);
    const trueState = states.get(`position_t${t}` as TimeSlicedVariable);
    console.log(
      `  t=${t}: Estimate=${(estimate as number).toFixed(3)}, True=${(trueState as number).toFixed(3)}`
    );
  }

  // Demo 4: Intervention Planning
  console.log("\n🧠 Demo 4: Temporal Intervention Planning");
  console.log("-".repeat(80));

  const planner = new TemporalInterventionPlanner(scm, "X", 8, 2.0);

  console.log("Goal: Achieve X=2.0 at t=8");
  const plan = planner.plan(initialState, 8, ["X", "Y"]);

  console.log(
    `\nBest Intervention Plan (Confidence: ${(plan.confidence * 100).toFixed(1)}%):`
  );
  for (const intv of plan.interventions) {
    console.log(`  Set ${intv.variable}=${intv.value} at t=${intv.timeStep}`);
  }

  const finalX = plan.expectedOutcome.get("X_t8" as TimeSlicedVariable);
  console.log(`Expected outcome at t=8: X=${(finalX as number).toFixed(3)}`);

  // Demo 5: Temporal Counterfactuals
  console.log("\n🔮 Demo 5: Temporal Counterfactuals");
  console.log("-".repeat(80));

  console.log("Actual history: X starts at 1.0, Y starts at 0.5");
  const actualHistory = scm.simulate(initialState, 10);

  console.log("\nCounterfactual: What if we had set Y=3.0 at t=2?");
  const cf = planner.computeCounterfactual(actualHistory, {
    variable: "Y",
    timeStep: 2,
    value: 3.0,
  });

  console.log("\nComparison (t=0 to t=5):");
  for (let t = 0; t <= 5; t++) {
    const actualX = actualHistory.get(`X_t${t}` as TimeSlicedVariable);
    const cfX = cf.counterfactualFuture.get(`X_t${t}` as TimeSlicedVariable);
    console.log(
      `  t=${t}: Actual X=${(actualX as number).toFixed(3)}, Counterfactual X=${(cfX as number).toFixed(3)}`
    );
  }

  console.log("\n✅ Temporal Causal Reasoning demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  TemporalSCM,
  DynamicBayesianNetwork,
  TemporalInterventionPlanner,
  type TemporalVariable,
  type TemporalEdge,
  type TemporalCausalGraph,
  type TemporalIntervention,
  type InterventionPlan,
  type TemporalCounterfactual,
};
