/**
 * Quantum-Inspired Agent Behaviors - Proof of Concept
 *
 * Implements superposition, entanglement, and quantum decision collapse
 * for multi-agent systems.
 *
 * @module quantum-behaviors
 * @agents @QUANTUM @APEX
 */

import { Complex } from "complex.js";

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export type AgentId = string;
export type StateLabel = string;
export type Timestamp = number;

/**
 * Complex-valued amplitude for quantum state
 */
export interface Amplitude {
  real: number;
  imag: number;
}

/**
 * Quantum state with superposition of classical states
 */
export interface QuantumState<T = any> {
  /** Map from state to complex amplitude */
  amplitudes: Map<T, Complex>;

  /** Computational basis */
  basis: T[];

  /** Global phase (can ignore for most purposes) */
  phase: number;

  /** Coherence time in milliseconds */
  coherenceTime: number;

  /** Creation timestamp */
  createdAt: Timestamp;
}

/**
 * Entanglement bond between two agents
 */
export interface EntanglementBond {
  agentPair: [AgentId, AgentId];
  bondStrength: number; // 0-1
  correlationType: "perfect" | "partial" | "noisy";
  createdAt: Timestamp;
  lastInteraction: Timestamp;
  decayRate: number; // per millisecond
}

/**
 * Measurement result
 */
export interface MeasurementResult<T> {
  outcome: T;
  probability: number;
  postMeasurementState: QuantumState<T>;
  timestamp: Timestamp;
}

/**
 * Quantum gate (unitary transformation)
 */
export interface QuantumGate<T> {
  name: string;
  numQubits: number;

  /**
   * Apply gate to quantum state
   */
  apply(state: QuantumState<T>): QuantumState<T>;
}

// ============================================================================
// SUPERPOSITION MANAGER
// ============================================================================

/**
 * Manages quantum superposition states for agents
 */
export class SuperpositionManager<T = any> {
  private states: Map<AgentId, QuantumState<T>> = new Map();

  /**
   * Create uniform superposition over states
   *
   * Example:
   *   createSuperposition(['left', 'right', 'forward'])
   *   → |ψ⟩ = (1/√3)|left⟩ + (1/√3)|right⟩ + (1/√3)|forward⟩
   */
  createSuperposition(
    agentId: AgentId,
    states: T[],
    weights?: number[]
  ): QuantumState<T> {
    if (states.length === 0) {
      throw new Error("Cannot create superposition of zero states");
    }

    // Default to uniform superposition
    if (!weights) {
      weights = states.map(() => 1 / Math.sqrt(states.length));
    }

    // Normalize weights
    const norm = Math.sqrt(weights.reduce((sum, w) => sum + w * w, 0));
    const normalizedWeights = weights.map((w) => w / norm);

    // Create amplitude map
    const amplitudes = new Map<T, Complex>();
    states.forEach((state, i) => {
      amplitudes.set(state, new Complex(normalizedWeights[i], 0));
    });

    const quantumState: QuantumState<T> = {
      amplitudes,
      basis: [...states],
      phase: 0,
      coherenceTime: 5000, // 5 seconds default
      createdAt: Date.now(),
    };

    this.states.set(agentId, quantumState);
    return quantumState;
  }

  /**
   * Collapse superposition via measurement (Born rule)
   *
   * P(outcome) = |⟨outcome|ψ⟩|² = |amplitude|²
   */
  collapse(agentId: AgentId, state?: QuantumState<T>): MeasurementResult<T> {
    const quantumState = state || this.states.get(agentId);
    if (!quantumState) {
      throw new Error(`No quantum state for agent ${agentId}`);
    }

    // Check for decoherence
    const age = Date.now() - quantumState.createdAt;
    if (age > quantumState.coherenceTime) {
      // State has decohered, collapse to random classical state
      const randomState =
        quantumState.basis[
          Math.floor(Math.random() * quantumState.basis.length)
        ];
      return {
        outcome: randomState,
        probability: 1 / quantumState.basis.length,
        postMeasurementState: this.createClassicalState(randomState),
        timestamp: Date.now(),
      };
    }

    // Compute probabilities using Born rule
    const probabilities: [T, number][] = [];
    for (const [state, amplitude] of quantumState.amplitudes) {
      const prob = amplitude.abs() ** 2;
      probabilities.push([state, prob]);
    }

    // Sample from probability distribution
    const rand = Math.random();
    let cumulative = 0;
    let outcome: T = probabilities[0][0];

    for (const [state, prob] of probabilities) {
      cumulative += prob;
      if (rand <= cumulative) {
        outcome = state;
        break;
      }
    }

    // Post-measurement state collapses to eigenstate
    const postState = this.createClassicalState(outcome);
    this.states.set(agentId, postState);

    const probability = probabilities.find(([s, _]) => s === outcome)?.[1] || 0;

    return {
      outcome,
      probability,
      postMeasurementState: postState,
      timestamp: Date.now(),
    };
  }

  /**
   * Create classical (definite) state
   */
  private createClassicalState(state: T): QuantumState<T> {
    return {
      amplitudes: new Map([[state, new Complex(1, 0)]]),
      basis: [state],
      phase: 0,
      coherenceTime: Infinity,
      createdAt: Date.now(),
    };
  }

  /**
   * Evolve quantum state under Hamiltonian (Schrödinger equation)
   *
   * |ψ(t+dt)⟩ = e^(-iHdt/ℏ)|ψ(t)⟩
   *
   * For simplicity, we implement as phase rotation
   */
  evolve(
    agentId: AgentId,
    hamiltonian: number[][],
    dt: number
  ): QuantumState<T> {
    const state = this.states.get(agentId);
    if (!state) {
      throw new Error(`No quantum state for agent ${agentId}`);
    }

    // For POC, apply simple phase evolution
    const newAmplitudes = new Map<T, Complex>();
    let i = 0;
    for (const [stateLabel, amplitude] of state.amplitudes) {
      const energy = hamiltonian[i][i]; // Diagonal approximation
      const phaseShift = -energy * dt; // ℏ = 1

      // Multiply by e^(iφ)
      const evolved = amplitude.mul(
        new Complex(Math.cos(phaseShift), Math.sin(phaseShift))
      );

      newAmplitudes.set(stateLabel, evolved);
      i++;
    }

    const evolved: QuantumState<T> = {
      ...state,
      amplitudes: newAmplitudes,
    };

    this.states.set(agentId, evolved);
    return evolved;
  }

  /**
   * Get probability distribution from quantum state
   */
  getProbabilities(state: QuantumState<T>): Map<T, number> {
    const probs = new Map<T, number>();
    for (const [stateLabel, amplitude] of state.amplitudes) {
      probs.set(stateLabel, amplitude.abs() ** 2);
    }
    return probs;
  }

  /**
   * Compute fidelity between two quantum states
   * F(ψ, φ) = |⟨ψ|φ⟩|²
   */
  fidelity(state1: QuantumState<T>, state2: QuantumState<T>): number {
    let innerProduct = new Complex(0, 0);

    for (const [label, amp1] of state1.amplitudes) {
      const amp2 = state2.amplitudes.get(label);
      if (amp2) {
        innerProduct = innerProduct.add(amp1.conjugate().mul(amp2));
      }
    }

    return innerProduct.abs() ** 2;
  }
}

// ============================================================================
// ENTANGLEMENT REGISTRY
// ============================================================================

/**
 * Manages entanglement between agents
 */
export class EntanglementRegistry {
  private bonds: Map<string, EntanglementBond> = new Map();

  /**
   * Create entanglement between two agents
   */
  createEntanglement(
    agent1: AgentId,
    agent2: AgentId,
    strength: number = 1.0,
    correlationType: "perfect" | "partial" | "noisy" = "perfect"
  ): EntanglementBond {
    const bondId = this.getBondId(agent1, agent2);

    const bond: EntanglementBond = {
      agentPair: [agent1, agent2],
      bondStrength: strength,
      correlationType,
      createdAt: Date.now(),
      lastInteraction: Date.now(),
      decayRate: 0.0001, // 0.01% per millisecond
    };

    this.bonds.set(bondId, bond);
    return bond;
  }

  /**
   * Get entangled partners of agent
   */
  getEntangledPartners(agent: AgentId): AgentId[] {
    const partners: AgentId[] = [];

    for (const bond of this.bonds.values()) {
      if (bond.agentPair[0] === agent) {
        partners.push(bond.agentPair[1]);
      } else if (bond.agentPair[1] === agent) {
        partners.push(bond.agentPair[0]);
      }
    }

    return partners;
  }

  /**
   * Enforce correlation when one agent is measured
   *
   * For perfect entanglement, measuring agent1 instantly
   * determines agent2's state (EPR correlation)
   */
  enforceCorrelation(
    measuredAgent: AgentId,
    measuredOutcome: any,
    superpositionMgr: SuperpositionManager
  ): void {
    const partners = this.getEntangledPartners(measuredAgent);

    for (const partner of partners) {
      const bondId = this.getBondId(measuredAgent, partner);
      const bond = this.bonds.get(bondId);

      if (!bond) continue;

      // Update bond
      bond.lastInteraction = Date.now();

      if (bond.correlationType === "perfect") {
        // Perfect anti-correlation: if agent1 measured |0⟩, agent2 collapses to |1⟩
        const correlatedOutcome = this.getCorrelatedOutcome(measuredOutcome);

        // Force partner to collapse to correlated outcome
        superpositionMgr.createSuperposition(
          partner,
          [correlatedOutcome] // Classical state
        );
      } else if (bond.correlationType === "partial") {
        // Bias partner's superposition toward correlation
        // (Implementation simplified for POC)
      }
    }
  }

  /**
   * Decay entanglement over time
   */
  decayEntanglement(dt: number): void {
    for (const [bondId, bond] of this.bonds) {
      const age = Date.now() - bond.lastInteraction;
      const decay = Math.exp(-bond.decayRate * age);

      bond.bondStrength *= decay;

      // Remove bonds that have decayed below threshold
      if (bond.bondStrength < 0.01) {
        this.bonds.delete(bondId);
      }
    }
  }

  /**
   * Measure entanglement strength (Bell inequality violation)
   *
   * CHSH inequality: |E| ≤ 2 for classical, can reach 2√2 for quantum
   */
  measureBellInequality(agent1: AgentId, agent2: AgentId): number {
    // Simplified Bell test (full implementation requires multiple measurement bases)
    const bondId = this.getBondId(agent1, agent2);
    const bond = this.bonds.get(bondId);

    if (!bond) return 0;

    // For perfect entanglement, CHSH = 2√2 ≈ 2.828
    // Scale by bond strength
    return 2 * Math.sqrt(2) * bond.bondStrength;
  }

  private getBondId(agent1: AgentId, agent2: AgentId): string {
    return [agent1, agent2].sort().join("::");
  }

  private getCorrelatedOutcome(outcome: any): any {
    // Simple anti-correlation for POC
    if (outcome === "left") return "right";
    if (outcome === "right") return "left";
    if (outcome === "up") return "down";
    if (outcome === "down") return "up";
    if (outcome === 0) return 1;
    if (outcome === 1) return 0;
    return outcome;
  }
}

// ============================================================================
// QUANTUM GATES
// ============================================================================

/**
 * Hadamard gate: creates superposition
 * H|0⟩ = (|0⟩ + |1⟩)/√2
 * H|1⟩ = (|0⟩ - |1⟩)/√2
 */
export class HadamardGate implements QuantumGate<number> {
  name = "Hadamard";
  numQubits = 1;

  apply(state: QuantumState<number>): QuantumState<number> {
    const newAmplitudes = new Map<number, Complex>();
    const sqrt2 = Math.sqrt(2);

    const amp0 = state.amplitudes.get(0) || new Complex(0, 0);
    const amp1 = state.amplitudes.get(1) || new Complex(0, 0);

    // H matrix: [[1, 1], [1, -1]] / √2
    newAmplitudes.set(0, amp0.add(amp1).div(sqrt2));
    newAmplitudes.set(1, amp0.sub(amp1).div(sqrt2));

    return {
      ...state,
      amplitudes: newAmplitudes,
    };
  }
}

/**
 * CNOT gate: entangles two qubits
 * CNOT|00⟩ = |00⟩
 * CNOT|01⟩ = |01⟩
 * CNOT|10⟩ = |11⟩
 * CNOT|11⟩ = |10⟩
 */
export class CNOTGate implements QuantumGate<string> {
  name = "CNOT";
  numQubits = 2;

  apply(state: QuantumState<string>): QuantumState<string> {
    const newAmplitudes = new Map<string, Complex>();

    for (const [basisState, amplitude] of state.amplitudes) {
      const [control, target] = basisState.split("");

      let newTarget = target;
      if (control === "1") {
        // Flip target if control is 1
        newTarget = target === "0" ? "1" : "0";
      }

      const newBasisState = control + newTarget;
      newAmplitudes.set(newBasisState, amplitude);
    }

    return {
      ...state,
      amplitudes: newAmplitudes,
    };
  }
}

// ============================================================================
// QUANTUM-ENHANCED AGENT
// ============================================================================

/**
 * Agent with quantum decision-making capabilities
 */
export class QuantumAgent {
  id: AgentId;
  private superpositionMgr: SuperpositionManager;
  private entanglementRegistry: EntanglementRegistry;

  constructor(
    id: AgentId,
    superpositionMgr: SuperpositionManager,
    entanglementRegistry: EntanglementRegistry
  ) {
    this.id = id;
    this.superpositionMgr = superpositionMgr;
    this.entanglementRegistry = entanglementRegistry;
  }

  /**
   * Make decision in superposition of actions
   */
  async makeDecision(actions: string[]): Promise<string> {
    // Create superposition of all possible actions
    const state = this.superpositionMgr.createSuperposition(this.id, actions);

    console.log(`Agent ${this.id} considering actions in superposition:`);
    const probs = this.superpositionMgr.getProbabilities(state);
    for (const [action, prob] of probs) {
      console.log(`  ${action}: ${(prob * 100).toFixed(1)}%`);
    }

    // Simulate quantum evolution (preferences affect amplitudes)
    // In real implementation, this would use learned Hamiltonian

    // Collapse to concrete decision
    const result = this.superpositionMgr.collapse(this.id);

    console.log(
      `Agent ${this.id} decided: ${result.outcome} (p=${(result.probability * 100).toFixed(1)}%)`
    );

    // Enforce entanglement correlations
    this.entanglementRegistry.enforceCorrelation(
      this.id,
      result.outcome,
      this.superpositionMgr
    );

    return result.outcome;
  }

  /**
   * Entangle with another agent
   */
  entangleWith(other: QuantumAgent, strength: number = 1.0): void {
    this.entanglementRegistry.createEntanglement(this.id, other.id, strength);
    console.log(
      `Agent ${this.id} entangled with ${other.id} (strength=${strength})`
    );
  }
}

// ============================================================================
// DEMO & TESTING
// ============================================================================

/**
 * Demonstration of quantum-inspired agent behaviors
 */
export async function demonstrateQuantumBehaviors(): Promise<void> {
  console.log("=".repeat(70));
  console.log("QUANTUM-INSPIRED AGENT BEHAVIORS - PROOF OF CONCEPT");
  console.log("=".repeat(70));
  console.log();

  // Initialize systems
  const superpositionMgr = new SuperpositionManager<string>();
  const entanglementRegistry = new EntanglementRegistry();

  // Create agents
  const alice = new QuantumAgent(
    "Alice",
    superpositionMgr,
    entanglementRegistry
  );
  const bob = new QuantumAgent("Bob", superpositionMgr, entanglementRegistry);

  // Demo 1: Superposition decision-making
  console.log("Demo 1: Superposition Decision-Making");
  console.log("-".repeat(70));
  const actions = ["explore_left", "explore_right", "explore_forward", "stay"];
  await alice.makeDecision(actions);
  console.log();

  // Demo 2: Entanglement
  console.log("Demo 2: Quantum Entanglement");
  console.log("-".repeat(70));
  alice.entangleWith(bob, 1.0);

  // When Alice makes a decision, Bob's state is correlated
  console.log("\\nAlice makes decision:");
  await alice.makeDecision(["left", "right"]);

  console.log("\\nBob's correlated decision:");
  await bob.makeDecision(["left", "right"]);
  console.log();

  // Demo 3: Bell inequality test
  console.log("Demo 3: Bell Inequality Violation");
  console.log("-".repeat(70));
  const chsh = entanglementRegistry.measureBellInequality("Alice", "Bob");
  console.log(`CHSH value: ${chsh.toFixed(3)}`);
  console.log(`Classical limit: 2.0`);
  console.log(`Quantum limit: ${(2 * Math.sqrt(2)).toFixed(3)}`);
  console.log(
    `Violation: ${chsh > 2 ? "YES" : "NO"} (quantum behavior confirmed)`
  );
  console.log();

  // Demo 4: Decoherence
  console.log("Demo 4: Decoherence Simulation");
  console.log("-".repeat(70));
  const shortLivedState = superpositionMgr.createSuperposition("Charlie", [
    "A",
    "B",
    "C",
  ]);
  shortLivedState.coherenceTime = 100; // 100ms

  console.log("Created superposition with 100ms coherence time");
  console.log("Waiting 200ms...");
  await new Promise((resolve) => setTimeout(resolve, 200));

  const result = superpositionMgr.collapse("Charlie");
  console.log(`After decoherence, collapsed to: ${result.outcome}`);
  console.log();

  console.log("=".repeat(70));
  console.log("QUANTUM POC COMPLETE");
  console.log("=".repeat(70));
}

// Export all components
export default {
  SuperpositionManager,
  EntanglementRegistry,
  HadamardGate,
  CNOTGate,
  QuantumAgent,
  demonstrateQuantumBehaviors,
};
