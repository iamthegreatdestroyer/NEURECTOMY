/**
 * Morphogenic Model Evolution - P0 Breakthrough Innovation
 *
 * Models that evolve their structure based on digital twin feedback.
 * Inspired by biological morphogenesis, neural architectures grow,
 * differentiate, and adapt based on environmental signals from the
 * digital twin simulation.
 *
 * This creates truly adaptive AI that reshapes itself in response
 * to changing conditions in the digital twin world.
 *
 * @module MorphogenicEvolution
 * @category CrossDomain/P0-Breakthrough
 */

import { EventEmitter } from "events";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * 3D Vector
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Morphogen - signaling molecule that guides growth
 */
export interface Morphogen {
  id: string;
  name: string;
  type: MorphogenType;

  // Spatial distribution
  sourcePosition: Vector3D;
  concentration: Map<string, number>; // Cell/neuron ID -> concentration
  diffusionRate: number;
  decayRate: number;

  // Effects
  growthEffect: number; // Positive = growth, negative = inhibition
  differentiationBias: string[]; // Which cell types it promotes

  // Dynamics
  productionRate: number;
  isActive: boolean;
}

/**
 * Morphogen types
 */
export type MorphogenType =
  | "growth_factor" // Promotes cell division
  | "inhibitor" // Prevents growth
  | "differentiation" // Promotes specialization
  | "guidance" // Directs axon growth
  | "survival" // Prevents cell death
  | "plasticity"; // Promotes learning

/**
 * Neural cell (stem cell that becomes neurons)
 */
export interface NeuralCell {
  id: string;
  position: Vector3D;

  // Cell state
  cellType: CellType;
  maturity: number; // 0-1, how mature the cell is
  potency: Potency; // Differentiation potential

  // Morphogenic state
  morphogenReceptors: Map<string, number>; // Morphogen ID -> sensitivity
  receivedSignals: Map<string, number>; // Recent morphogen concentrations

  // Growth state
  divisionPotential: number;
  growthDirection: Vector3D;
  migrationTarget: Vector3D | null;

  // Neural properties (if differentiated)
  neuralProperties?: {
    activation: number;
    weights: Map<string, number>;
    dendrites: string[];
    axon: string[];
  };

  // Visual
  color: string;
  size: number;
}

/**
 * Cell types
 */
export type CellType =
  | "stem" // Undifferentiated
  | "progenitor" // Partially committed
  | "neuron" // Fully differentiated neuron
  | "interneuron" // Inhibitory neuron
  | "glial" // Supporting cell
  | "apoptotic"; // Dying cell

/**
 * Cell potency levels
 */
export type Potency =
  | "pluripotent" // Can become any cell type
  | "multipotent" // Can become several types
  | "unipotent" // Can only become one type
  | "terminal"; // Fully differentiated

/**
 * Growth pattern
 */
export interface GrowthPattern {
  id: string;
  name: string;

  // Spatial pattern
  shape: PatternShape;
  center: Vector3D;
  scale: Vector3D;

  // Timing
  startTime: number;
  duration: number;
  phase: number;

  // Effects
  morphogens: string[];
  cellTypeBias: Map<CellType, number>;
}

/**
 * Pattern shapes
 */
export type PatternShape =
  | "radial" // Expanding from center
  | "gradient" // Linear gradient
  | "periodic" // Repeating pattern
  | "fractal" // Self-similar
  | "reaction_diffusion"; // Turing patterns

/**
 * Twin feedback signal
 */
export interface TwinFeedback {
  id: string;
  timestamp: number;

  // Performance metrics
  taskPerformance: Map<string, number>;
  errorPatterns: ErrorPattern[];

  // Environmental signals
  inputDistribution: number[];
  outputRequirements: number[];

  // Adaptation signals
  growthSignals: GrowthSignal[];
  pruningSignals: PruningSignal[];
}

/**
 * Error pattern detected
 */
export interface ErrorPattern {
  region: string; // Which part of network
  errorType: string; // Type of error
  frequency: number; // How often it occurs
  severity: number; // How bad it is
}

/**
 * Growth signal from twin
 */
export interface GrowthSignal {
  targetRegion: Vector3D;
  radius: number;
  intensity: number;
  cellTypeSuggestion?: CellType;
}

/**
 * Pruning signal from twin
 */
export interface PruningSignal {
  targetRegion: Vector3D;
  radius: number;
  intensity: number;
  reason: string;
}

/**
 * Morphogenic network - the evolving neural network
 */
export interface MorphogenicNetwork {
  id: string;
  name: string;

  // Structure
  cells: Map<string, NeuralCell>;
  morphogens: Map<string, Morphogen>;
  connections: Map<string, CellConnection>;
  patterns: Map<string, GrowthPattern>;

  // Twin integration
  twinId: string;
  twinFeedbackHistory: TwinFeedback[];

  // Evolution state
  generation: number;
  age: number;
  evolutionPhase: EvolutionPhase;

  // Metrics
  fitness: number;
  complexity: number;
  adaptationRate: number;

  // Spatial bounds
  bounds: {
    min: Vector3D;
    max: Vector3D;
  };
}

/**
 * Cell connection
 */
export interface CellConnection {
  id: string;
  sourceId: string;
  targetId: string;

  weight: number;
  strength: number;
  age: number;

  // Morphogenic properties
  guidedBy: string[]; // Morphogens that influenced this
  plasticity: number;
}

/**
 * Evolution phases
 */
export type EvolutionPhase =
  | "genesis" // Initial creation
  | "proliferation" // Rapid growth
  | "migration" // Cells finding positions
  | "differentiation" // Cells specializing
  | "synaptogenesis" // Connection formation
  | "refinement" // Pruning and optimization
  | "maturation" // Final stabilization
  | "adaptation"; // Ongoing adaptation

/**
 * Evolution configuration
 */
export interface EvolutionConfig {
  // Growth parameters
  maxCells: number;
  cellDivisionRate: number;
  cellDeathRate: number;

  // Morphogen parameters
  morphogenDiffusionRate: number;
  morphogenDecayRate: number;

  // Differentiation
  differentiationThreshold: number;
  maturationRate: number;

  // Connection parameters
  connectionFormationRate: number;
  pruningThreshold: number;

  // Twin integration
  twinFeedbackWeight: number;
  adaptationSpeed: number;
}

/**
 * Evolution events
 */
export interface EvolutionEvents {
  "cell:divided": { parentId: string; childId: string };
  "cell:differentiated": { cellId: string; newType: CellType };
  "cell:died": { cellId: string; cause: string };
  "cell:migrated": { cellId: string; from: Vector3D; to: Vector3D };

  "morphogen:released": { morphogenId: string; position: Vector3D };
  "morphogen:gradient_formed": { morphogenId: string };

  "connection:formed": { connectionId: string };
  "connection:pruned": { connectionId: string; reason: string };
  "connection:strengthened": { connectionId: string; newWeight: number };

  "pattern:activated": { patternId: string };
  "pattern:completed": { patternId: string };

  "phase:changed": { from: EvolutionPhase; to: EvolutionPhase };
  "generation:advanced": { generation: number };

  "twin:feedback_received": { feedback: TwinFeedback };
  "adaptation:triggered": { signal: string };
}

// ============================================================================
// Morphogenic Model Evolution Implementation
// ============================================================================

/**
 * Morphogenic Model Evolution Engine
 *
 * Creates neural networks that grow and adapt like biological organisms,
 * guided by morphogen gradients and twin feedback signals.
 */
export class MorphogenicEvolution extends EventEmitter {
  private config: EvolutionConfig;
  private networks: Map<string, MorphogenicNetwork>;

  // Simulation state
  private isRunning: boolean;
  private simulationTime: number;
  private lastUpdateTime: number;
  private animationFrame: number | null;

  constructor(config: Partial<EvolutionConfig> = {}) {
    super();

    this.config = this.mergeConfig(config);
    this.networks = new Map();

    this.isRunning = false;
    this.simulationTime = 0;
    this.lastUpdateTime = 0;
    this.animationFrame = null;
  }

  /**
   * Merge config with defaults
   */
  private mergeConfig(config: Partial<EvolutionConfig>): EvolutionConfig {
    return {
      maxCells: config.maxCells ?? 1000,
      cellDivisionRate: config.cellDivisionRate ?? 0.1,
      cellDeathRate: config.cellDeathRate ?? 0.01,
      morphogenDiffusionRate: config.morphogenDiffusionRate ?? 0.5,
      morphogenDecayRate: config.morphogenDecayRate ?? 0.1,
      differentiationThreshold: config.differentiationThreshold ?? 0.7,
      maturationRate: config.maturationRate ?? 0.05,
      connectionFormationRate: config.connectionFormationRate ?? 0.2,
      pruningThreshold: config.pruningThreshold ?? 0.1,
      twinFeedbackWeight: config.twinFeedbackWeight ?? 0.5,
      adaptationSpeed: config.adaptationSpeed ?? 0.1,
    };
  }

  // ============================================================================
  // Network Creation
  // ============================================================================

  /**
   * Create a new morphogenic network
   */
  createNetwork(name: string, seedConfig?: SeedConfig): MorphogenicNetwork {
    const id = this.generateId("network");

    const network: MorphogenicNetwork = {
      id,
      name,
      cells: new Map(),
      morphogens: new Map(),
      connections: new Map(),
      patterns: new Map(),
      twinId: this.generateId("twin"),
      twinFeedbackHistory: [],
      generation: 0,
      age: 0,
      evolutionPhase: "genesis",
      fitness: 0,
      complexity: 0,
      adaptationRate: 0,
      bounds: {
        min: { x: -100, y: -100, z: -100 },
        max: { x: 100, y: 100, z: 100 },
      },
    };

    // Initialize with seed cells
    if (seedConfig) {
      this.seedNetwork(network, seedConfig);
    } else {
      this.seedNetworkDefault(network);
    }

    // Create default morphogens
    this.createDefaultMorphogens(network);

    this.networks.set(id, network);

    return network;
  }

  /**
   * Seed network with initial cells
   */
  private seedNetwork(network: MorphogenicNetwork, config: SeedConfig): void {
    for (let i = 0; i < config.numStemCells; i++) {
      const position: Vector3D = {
        x: (Math.random() - 0.5) * config.spreadRadius * 2,
        y: (Math.random() - 0.5) * config.spreadRadius * 2,
        z: (Math.random() - 0.5) * config.spreadRadius * 2,
      };

      this.createStemCell(network, position);
    }
  }

  /**
   * Default seeding
   */
  private seedNetworkDefault(network: MorphogenicNetwork): void {
    this.seedNetwork(network, {
      numStemCells: 10,
      spreadRadius: 20,
    });
  }

  /**
   * Create a stem cell
   */
  private createStemCell(
    network: MorphogenicNetwork,
    position: Vector3D
  ): NeuralCell {
    const id = this.generateId("cell");

    const cell: NeuralCell = {
      id,
      position,
      cellType: "stem",
      maturity: 0,
      potency: "pluripotent",
      morphogenReceptors: this.createDefaultReceptors(),
      receivedSignals: new Map(),
      divisionPotential: 1.0,
      growthDirection: { x: 0, y: 1, z: 0 },
      migrationTarget: null,
      color: "#ffffff",
      size: 3,
    };

    network.cells.set(id, cell);

    return cell;
  }

  /**
   * Create default morphogen receptors
   */
  private createDefaultReceptors(): Map<string, number> {
    return new Map([
      ["growth_factor", 1.0],
      ["differentiation", 1.0],
      ["guidance", 1.0],
      ["survival", 1.0],
      ["plasticity", 1.0],
    ]);
  }

  /**
   * Create default morphogens
   */
  private createDefaultMorphogens(network: MorphogenicNetwork): void {
    // Growth factor at center
    this.createMorphogen(network, {
      name: "Central Growth Factor",
      type: "growth_factor",
      sourcePosition: { x: 0, y: 0, z: 0 },
      diffusionRate: 0.5,
      decayRate: 0.1,
      growthEffect: 1.0,
      productionRate: 0.5,
    });

    // Differentiation gradient
    this.createMorphogen(network, {
      name: "Differentiation Signal",
      type: "differentiation",
      sourcePosition: { x: 0, y: 50, z: 0 },
      diffusionRate: 0.3,
      decayRate: 0.15,
      growthEffect: 0,
      differentiationBias: ["neuron"],
      productionRate: 0.3,
    });

    // Guidance cue for output
    this.createMorphogen(network, {
      name: "Output Guidance",
      type: "guidance",
      sourcePosition: { x: 0, y: 0, z: 100 },
      diffusionRate: 0.4,
      decayRate: 0.2,
      growthEffect: 0.5,
      productionRate: 0.4,
    });
  }

  /**
   * Create a morphogen
   */
  private createMorphogen(
    network: MorphogenicNetwork,
    config: MorphogenConfig
  ): Morphogen {
    const id = this.generateId("morphogen");

    const morphogen: Morphogen = {
      id,
      name: config.name,
      type: config.type,
      sourcePosition: config.sourcePosition,
      concentration: new Map(),
      diffusionRate: config.diffusionRate,
      decayRate: config.decayRate,
      growthEffect: config.growthEffect,
      differentiationBias: config.differentiationBias || [],
      productionRate: config.productionRate,
      isActive: true,
    };

    network.morphogens.set(id, morphogen);

    return morphogen;
  }

  // ============================================================================
  // Simulation Loop
  // ============================================================================

  /**
   * Start evolution simulation
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.lastUpdateTime = performance.now();
    this.startAnimationLoop();
  }

  /**
   * Stop simulation
   */
  stop(): void {
    this.isRunning = false;
    this.stopAnimationLoop();
  }

  /**
   * Start animation loop
   */
  private startAnimationLoop(): void {
    if (this.animationFrame !== null) return;

    const animate = (currentTime: number) => {
      if (!this.isRunning) return;

      const deltaTime = Math.min(
        (currentTime - this.lastUpdateTime) / 1000,
        0.1
      );
      this.lastUpdateTime = currentTime;

      this.update(deltaTime);

      this.animationFrame = requestAnimationFrame(animate);
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Stop animation loop
   */
  private stopAnimationLoop(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Main update loop
   */
  private update(deltaTime: number): void {
    this.simulationTime += deltaTime;

    for (const network of this.networks.values()) {
      this.updateNetwork(network, deltaTime);
    }

    this.emit("frame:updated", {
      simulationTime: this.simulationTime,
      deltaTime,
    });
  }

  /**
   * Update a network
   */
  private updateNetwork(network: MorphogenicNetwork, deltaTime: number): void {
    network.age += deltaTime;

    // Update morphogens (diffusion, decay)
    this.updateMorphogens(network, deltaTime);

    // Update cells
    this.updateCells(network, deltaTime);

    // Update connections
    this.updateConnections(network, deltaTime);

    // Check phase transitions
    this.checkPhaseTransition(network);

    // Update metrics
    this.updateMetrics(network);
  }

  /**
   * Update morphogens
   */
  private updateMorphogens(
    network: MorphogenicNetwork,
    deltaTime: number
  ): void {
    for (const morphogen of network.morphogens.values()) {
      if (!morphogen.isActive) continue;

      // Update concentration at source
      // (Production at source)

      // Diffuse to cells
      for (const cell of network.cells.values()) {
        const distance = this.distance3D(
          morphogen.sourcePosition,
          cell.position
        );

        // Gaussian diffusion
        const concentration =
          morphogen.productionRate *
          Math.exp(
            (-distance * distance) /
              (2 * Math.pow(morphogen.diffusionRate * 100, 2))
          );

        // Apply decay
        const currentConc = morphogen.concentration.get(cell.id) || 0;
        const newConc =
          currentConc * (1 - morphogen.decayRate * deltaTime) +
          concentration * deltaTime;

        morphogen.concentration.set(cell.id, Math.max(0, newConc));

        // Cell receives signal
        const receptor = cell.morphogenReceptors.get(morphogen.type) || 0;
        cell.receivedSignals.set(morphogen.id, newConc * receptor);
      }
    }
  }

  /**
   * Update cells
   */
  private updateCells(network: MorphogenicNetwork, deltaTime: number): void {
    const cellsToAdd: NeuralCell[] = [];
    const cellsToRemove: string[] = [];

    for (const cell of network.cells.values()) {
      // Update maturity
      cell.maturity = Math.min(
        1.0,
        cell.maturity + this.config.maturationRate * deltaTime
      );

      // Process morphogen signals
      this.processCellSignals(network, cell, deltaTime);

      // Cell division
      if (this.shouldDivide(network, cell)) {
        const child = this.divideCell(network, cell);
        if (child) cellsToAdd.push(child);
      }

      // Cell differentiation
      if (this.shouldDifferentiate(cell)) {
        this.differentiateCell(network, cell);
      }

      // Cell migration
      this.migrateCell(cell, deltaTime);

      // Cell death
      if (this.shouldDie(cell)) {
        cellsToRemove.push(cell.id);
        this.emit("cell:died", { cellId: cell.id, cause: "apoptosis" });
      }

      // Update visual
      this.updateCellVisual(cell);
    }

    // Apply changes
    for (const cell of cellsToAdd) {
      network.cells.set(cell.id, cell);
    }

    for (const cellId of cellsToRemove) {
      this.removeCell(network, cellId);
    }
  }

  /**
   * Process morphogen signals for a cell
   */
  private processCellSignals(
    network: MorphogenicNetwork,
    cell: NeuralCell,
    deltaTime: number
  ): void {
    let growthSignal = 0;
    let differentiationSignal = 0;
    let guidanceX = 0;
    let guidanceY = 0;
    let guidanceZ = 0;

    for (const morphogen of network.morphogens.values()) {
      const signal = cell.receivedSignals.get(morphogen.id) || 0;

      switch (morphogen.type) {
        case "growth_factor":
          growthSignal += signal * morphogen.growthEffect;
          break;
        case "differentiation":
          differentiationSignal += signal;
          break;
        case "guidance":
          // Pull toward morphogen source
          const dir = this.normalize3D(
            this.subtract3D(morphogen.sourcePosition, cell.position)
          );
          guidanceX += dir.x * signal;
          guidanceY += dir.y * signal;
          guidanceZ += dir.z * signal;
          break;
        case "inhibitor":
          growthSignal -= signal;
          break;
        case "survival":
          // Reduce death probability
          cell.maturity += signal * 0.01 * deltaTime;
          break;
      }
    }

    // Update cell state based on signals
    cell.divisionPotential = Math.max(
      0,
      Math.min(1, cell.divisionPotential + growthSignal * 0.1 * deltaTime)
    );

    // Update growth direction
    if (
      Math.abs(guidanceX) + Math.abs(guidanceY) + Math.abs(guidanceZ) >
      0.01
    ) {
      cell.growthDirection = this.normalize3D({
        x: guidanceX,
        y: guidanceY,
        z: guidanceZ,
      });
    }
  }

  /**
   * Check if cell should divide
   */
  private shouldDivide(network: MorphogenicNetwork, cell: NeuralCell): boolean {
    if (cell.potency === "terminal") return false;
    if (network.cells.size >= this.config.maxCells) return false;

    const divisionChance =
      cell.divisionPotential * this.config.cellDivisionRate;
    return Math.random() < divisionChance;
  }

  /**
   * Divide a cell
   */
  private divideCell(
    network: MorphogenicNetwork,
    parent: NeuralCell
  ): NeuralCell | null {
    // Create child at offset from parent
    const offset = {
      x: (Math.random() - 0.5) * 5,
      y: (Math.random() - 0.5) * 5,
      z: (Math.random() - 0.5) * 5,
    };

    const childPosition = this.add3D(parent.position, offset);

    // Create child cell
    const child: NeuralCell = {
      id: this.generateId("cell"),
      position: childPosition,
      cellType: parent.cellType,
      maturity: 0,
      potency: this.degradePotency(parent.potency),
      morphogenReceptors: new Map(parent.morphogenReceptors),
      receivedSignals: new Map(),
      divisionPotential: 0.5,
      growthDirection: parent.growthDirection,
      migrationTarget: null,
      color: parent.color,
      size: parent.size * 0.8,
    };

    // Reduce parent's division potential
    parent.divisionPotential *= 0.5;

    this.emit("cell:divided", { parentId: parent.id, childId: child.id });

    return child;
  }

  /**
   * Degrade potency after division
   */
  private degradePotency(potency: Potency): Potency {
    const degradation: Record<Potency, Potency> = {
      pluripotent: "multipotent",
      multipotent: "unipotent",
      unipotent: "unipotent",
      terminal: "terminal",
    };
    return degradation[potency];
  }

  /**
   * Check if cell should differentiate
   */
  private shouldDifferentiate(cell: NeuralCell): boolean {
    if (cell.cellType !== "stem" && cell.cellType !== "progenitor")
      return false;
    if (cell.potency === "terminal") return false;

    // Check differentiation signal strength
    let differentiationSignal = 0;
    for (const [, value] of cell.receivedSignals) {
      differentiationSignal += value;
    }

    return (
      cell.maturity > 0.5 &&
      differentiationSignal > this.config.differentiationThreshold
    );
  }

  /**
   * Differentiate a cell
   */
  private differentiateCell(
    network: MorphogenicNetwork,
    cell: NeuralCell
  ): void {
    // Determine new cell type based on signals
    let maxSignal = 0;
    let newType: CellType = "neuron";

    for (const morphogen of network.morphogens.values()) {
      if (morphogen.type !== "differentiation") continue;

      const signal = cell.receivedSignals.get(morphogen.id) || 0;
      if (signal > maxSignal && morphogen.differentiationBias.length > 0) {
        maxSignal = signal;
        newType = morphogen.differentiationBias[0] as CellType;
      }
    }

    const oldType = cell.cellType;
    cell.cellType = newType;
    cell.potency = "terminal";

    // Initialize neural properties if becoming neuron
    if (newType === "neuron" || newType === "interneuron") {
      cell.neuralProperties = {
        activation: 0,
        weights: new Map(),
        dendrites: [],
        axon: [],
      };
    }

    // Update visual
    this.updateCellVisual(cell);

    this.emit("cell:differentiated", { cellId: cell.id, newType });
  }

  /**
   * Migrate cell toward target
   */
  private migrateCell(cell: NeuralCell, deltaTime: number): void {
    if (!cell.migrationTarget) {
      // Random walk influenced by growth direction
      const speed = 2 * deltaTime;
      cell.position = this.add3D(cell.position, {
        x: cell.growthDirection.x * speed + (Math.random() - 0.5) * speed * 0.5,
        y: cell.growthDirection.y * speed + (Math.random() - 0.5) * speed * 0.5,
        z: cell.growthDirection.z * speed + (Math.random() - 0.5) * speed * 0.5,
      });
    } else {
      // Move toward target
      const direction = this.normalize3D(
        this.subtract3D(cell.migrationTarget, cell.position)
      );
      const speed = 5 * deltaTime;

      cell.position = this.add3D(cell.position, {
        x: direction.x * speed,
        y: direction.y * speed,
        z: direction.z * speed,
      });

      // Check if reached target
      if (this.distance3D(cell.position, cell.migrationTarget) < 1) {
        const from = cell.migrationTarget;
        cell.migrationTarget = null;
        this.emit("cell:migrated", {
          cellId: cell.id,
          from,
          to: cell.position,
        });
      }
    }
  }

  /**
   * Check if cell should die
   */
  private shouldDie(cell: NeuralCell): boolean {
    if (cell.cellType === "apoptotic") return true;

    // Low survival signal
    let survivalSignal = 0;
    for (const [, value] of cell.receivedSignals) {
      survivalSignal += value;
    }

    // Die if very low signal and mature
    return (
      survivalSignal < 0.1 &&
      cell.maturity > 0.8 &&
      Math.random() < this.config.cellDeathRate
    );
  }

  /**
   * Remove a cell
   */
  private removeCell(network: MorphogenicNetwork, cellId: string): void {
    const cell = network.cells.get(cellId);
    if (!cell) return;

    // Remove all connections
    for (const [connId, conn] of network.connections) {
      if (conn.sourceId === cellId || conn.targetId === cellId) {
        network.connections.delete(connId);
      }
    }

    network.cells.delete(cellId);
  }

  /**
   * Update cell visual
   */
  private updateCellVisual(cell: NeuralCell): void {
    const colors: Record<CellType, string> = {
      stem: "#ffffff",
      progenitor: "#aaddff",
      neuron: "#4fc3f7",
      interneuron: "#ff7043",
      glial: "#81c784",
      apoptotic: "#888888",
    };

    cell.color = colors[cell.cellType] || "#ffffff";
    cell.size = 2 + cell.maturity * 2;

    // Add glow for active neurons
    if (
      cell.neuralProperties &&
      Math.abs(cell.neuralProperties.activation) > 0.5
    ) {
      cell.size += 1;
    }
  }

  /**
   * Update connections
   */
  private updateConnections(
    network: MorphogenicNetwork,
    deltaTime: number
  ): void {
    // Form new connections
    if (
      network.evolutionPhase === "synaptogenesis" ||
      network.evolutionPhase === "adaptation"
    ) {
      this.formConnections(network);
    }

    // Prune weak connections
    this.pruneConnections(network, deltaTime);

    // Update existing connections
    for (const connection of network.connections.values()) {
      connection.age += deltaTime;
    }
  }

  /**
   * Form new connections between neurons
   */
  private formConnections(network: MorphogenicNetwork): void {
    const neurons = Array.from(network.cells.values()).filter(
      (c) => c.cellType === "neuron" || c.cellType === "interneuron"
    );

    if (neurons.length < 2) return;

    // Try to form connections
    for (
      let i = 0;
      i < neurons.length && Math.random() < this.config.connectionFormationRate;
      i++
    ) {
      const source = neurons[Math.floor(Math.random() * neurons.length)];
      const target = neurons[Math.floor(Math.random() * neurons.length)];

      if (source.id === target.id) continue;

      // Check if connection already exists
      const existingConn = Array.from(network.connections.values()).find(
        (c) => c.sourceId === source.id && c.targetId === target.id
      );

      if (existingConn) continue;

      // Distance-based probability
      const distance = this.distance3D(source.position, target.position);
      const probability = Math.exp(-distance / 50);

      if (Math.random() < probability) {
        this.createConnection(network, source.id, target.id);
      }
    }
  }

  /**
   * Create a connection
   */
  private createConnection(
    network: MorphogenicNetwork,
    sourceId: string,
    targetId: string
  ): CellConnection {
    const id = this.generateId("conn");

    const connection: CellConnection = {
      id,
      sourceId,
      targetId,
      weight: (Math.random() - 0.5) * 0.1,
      strength: 1,
      age: 0,
      guidedBy: [],
      plasticity: 0.1,
    };

    network.connections.set(id, connection);

    // Update cell neural properties
    const source = network.cells.get(sourceId);
    const target = network.cells.get(targetId);

    if (source?.neuralProperties) {
      source.neuralProperties.axon.push(id);
      source.neuralProperties.weights.set(targetId, connection.weight);
    }

    if (target?.neuralProperties) {
      target.neuralProperties.dendrites.push(id);
    }

    this.emit("connection:formed", { connectionId: id });

    return connection;
  }

  /**
   * Prune weak connections
   */
  private pruneConnections(
    network: MorphogenicNetwork,
    deltaTime: number
  ): void {
    const toPrune: string[] = [];

    for (const [id, conn] of network.connections) {
      // Pruning based on weight strength and age
      if (
        Math.abs(conn.weight) < this.config.pruningThreshold &&
        conn.age > 10
      ) {
        toPrune.push(id);
      }
    }

    for (const id of toPrune) {
      this.removeConnection(network, id, "weak_weight");
    }
  }

  /**
   * Remove a connection
   */
  private removeConnection(
    network: MorphogenicNetwork,
    connectionId: string,
    reason: string
  ): void {
    const conn = network.connections.get(connectionId);
    if (!conn) return;

    // Update cell neural properties
    const source = network.cells.get(conn.sourceId);
    const target = network.cells.get(conn.targetId);

    if (source?.neuralProperties) {
      source.neuralProperties.axon = source.neuralProperties.axon.filter(
        (id) => id !== connectionId
      );
      source.neuralProperties.weights.delete(conn.targetId);
    }

    if (target?.neuralProperties) {
      target.neuralProperties.dendrites =
        target.neuralProperties.dendrites.filter((id) => id !== connectionId);
    }

    network.connections.delete(connectionId);

    this.emit("connection:pruned", { connectionId, reason });
  }

  /**
   * Check and handle phase transitions
   */
  private checkPhaseTransition(network: MorphogenicNetwork): void {
    const cellCount = network.cells.size;
    const neuronCount = Array.from(network.cells.values()).filter(
      (c) => c.cellType === "neuron" || c.cellType === "interneuron"
    ).length;
    const connectionCount = network.connections.size;
    const avgMaturity =
      Array.from(network.cells.values()).reduce(
        (sum, c) => sum + c.maturity,
        0
      ) / Math.max(1, cellCount);

    let newPhase: EvolutionPhase = network.evolutionPhase;

    switch (network.evolutionPhase) {
      case "genesis":
        if (cellCount > 5) newPhase = "proliferation";
        break;
      case "proliferation":
        if (cellCount > this.config.maxCells * 0.5) newPhase = "migration";
        break;
      case "migration":
        if (avgMaturity > 0.3) newPhase = "differentiation";
        break;
      case "differentiation":
        if (neuronCount > cellCount * 0.5) newPhase = "synaptogenesis";
        break;
      case "synaptogenesis":
        if (connectionCount > neuronCount * 2) newPhase = "refinement";
        break;
      case "refinement":
        if (avgMaturity > 0.8) newPhase = "maturation";
        break;
      case "maturation":
        // Can transition to adaptation based on twin feedback
        break;
    }

    if (newPhase !== network.evolutionPhase) {
      const oldPhase = network.evolutionPhase;
      network.evolutionPhase = newPhase;
      this.emit("phase:changed", { from: oldPhase, to: newPhase });
    }
  }

  /**
   * Update network metrics
   */
  private updateMetrics(network: MorphogenicNetwork): void {
    const cellCount = network.cells.size;
    const connectionCount = network.connections.size;

    // Complexity metric
    network.complexity = cellCount + connectionCount * 0.5;

    // Adaptation rate (based on recent changes)
    // Would track changes over time in real implementation
    network.adaptationRate = 0.1;
  }

  // ============================================================================
  // Twin Feedback Integration
  // ============================================================================

  /**
   * Process feedback from digital twin
   */
  processTwinFeedback(networkId: string, feedback: TwinFeedback): void {
    const network = this.networks.get(networkId);
    if (!network) return;

    network.twinFeedbackHistory.push(feedback);

    // Limit history size
    if (network.twinFeedbackHistory.length > 100) {
      network.twinFeedbackHistory.shift();
    }

    this.emit("twin:feedback_received", { feedback });

    // Process growth signals
    for (const signal of feedback.growthSignals) {
      this.applyGrowthSignal(network, signal);
    }

    // Process pruning signals
    for (const signal of feedback.pruningSignals) {
      this.applyPruningSignal(network, signal);
    }

    // Transition to adaptation phase if needed
    if (network.evolutionPhase === "maturation") {
      network.evolutionPhase = "adaptation";
      this.emit("phase:changed", { from: "maturation", to: "adaptation" });
    }

    this.emit("adaptation:triggered", { signal: "twin_feedback" });
  }

  /**
   * Apply growth signal from twin
   */
  private applyGrowthSignal(
    network: MorphogenicNetwork,
    signal: GrowthSignal
  ): void {
    // Create temporary morphogen at signal location
    const morphogen = this.createMorphogen(network, {
      name: "Twin Growth Signal",
      type: "growth_factor",
      sourcePosition: signal.targetRegion,
      diffusionRate: 0.3,
      decayRate: 0.5, // Temporary, decays quickly
      growthEffect: signal.intensity,
      productionRate: signal.intensity * 0.5,
    });

    // Mark for removal after effect
    setTimeout(() => {
      morphogen.isActive = false;
      network.morphogens.delete(morphogen.id);
    }, 10000);

    // If specific cell type suggested, add differentiation signal
    if (signal.cellTypeSuggestion) {
      const diffMorphogen = this.createMorphogen(network, {
        name: "Twin Differentiation Signal",
        type: "differentiation",
        sourcePosition: signal.targetRegion,
        diffusionRate: 0.2,
        decayRate: 0.3,
        growthEffect: 0,
        differentiationBias: [signal.cellTypeSuggestion],
        productionRate: signal.intensity * 0.3,
      });

      setTimeout(() => {
        diffMorphogen.isActive = false;
        network.morphogens.delete(diffMorphogen.id);
      }, 10000);
    }
  }

  /**
   * Apply pruning signal from twin
   */
  private applyPruningSignal(
    network: MorphogenicNetwork,
    signal: PruningSignal
  ): void {
    // Find cells in region
    const affectedCells = Array.from(network.cells.values()).filter(
      (c) => this.distance3D(c.position, signal.targetRegion) < signal.radius
    );

    // Create inhibitor morphogen
    const inhibitor = this.createMorphogen(network, {
      name: "Twin Pruning Signal",
      type: "inhibitor",
      sourcePosition: signal.targetRegion,
      diffusionRate: 0.2,
      decayRate: 0.3,
      growthEffect: -signal.intensity,
      productionRate: signal.intensity * 0.5,
    });

    // Mark cells for death based on intensity
    for (const cell of affectedCells) {
      if (Math.random() < signal.intensity * 0.5) {
        cell.cellType = "apoptotic";
      }
    }

    // Remove inhibitor after effect
    setTimeout(() => {
      inhibitor.isActive = false;
      network.morphogens.delete(inhibitor.id);
    }, 5000);
  }

  /**
   * Simulate error-driven adaptation
   */
  adaptToErrors(networkId: string, errors: ErrorPattern[]): void {
    const network = this.networks.get(networkId);
    if (!network) return;

    for (const error of errors) {
      // Find region associated with error
      // (Simplified - in reality would map error to network region)
      const regionCenter = { x: 0, y: 0, z: 0 };

      if (error.severity > 0.5) {
        // High severity - suggest restructuring
        this.processTwinFeedback(networkId, {
          id: this.generateId("feedback"),
          timestamp: Date.now(),
          taskPerformance: new Map(),
          errorPatterns: [error],
          inputDistribution: [],
          outputRequirements: [],
          growthSignals: [
            {
              targetRegion: regionCenter,
              radius: 20,
              intensity: error.severity,
            },
          ],
          pruningSignals: [
            {
              targetRegion: regionCenter,
              radius: 10,
              intensity: error.severity * 0.5,
              reason: error.errorType,
            },
          ],
        });
      }
    }
  }

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Generate unique ID
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Calculate 3D distance
   */
  private distance3D(a: Vector3D, b: Vector3D): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Add two vectors
   */
  private add3D(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
  }

  /**
   * Subtract vectors
   */
  private subtract3D(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  }

  /**
   * Normalize vector
   */
  private normalize3D(v: Vector3D): Vector3D {
    const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len === 0) return { x: 0, y: 0, z: 0 };
    return { x: v.x / len, y: v.y / len, z: v.z / len };
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Get network
   */
  getNetwork(id: string): MorphogenicNetwork | undefined {
    return this.networks.get(id);
  }

  /**
   * Get all networks
   */
  getAllNetworks(): MorphogenicNetwork[] {
    return Array.from(this.networks.values());
  }

  /**
   * Remove network
   */
  removeNetwork(id: string): void {
    this.networks.delete(id);
  }

  /**
   * Get configuration
   */
  getConfig(): EvolutionConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<EvolutionConfig>): void {
    Object.assign(this.config, config);
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.stop();
    this.networks.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Supporting Types
// ============================================================================

/**
 * Seed configuration
 */
export interface SeedConfig {
  numStemCells: number;
  spreadRadius: number;
}

/**
 * Morphogen configuration
 */
export interface MorphogenConfig {
  name: string;
  type: MorphogenType;
  sourcePosition: Vector3D;
  diffusionRate: number;
  decayRate: number;
  growthEffect: number;
  differentiationBias?: string[];
  productionRate: number;
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create Morphogenic Evolution engine
 */
export function createMorphogenicEvolution(
  config?: Partial<EvolutionConfig>
): MorphogenicEvolution {
  return new MorphogenicEvolution(config);
}
