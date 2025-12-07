/**
 * Living Architecture Laboratory - P0 Breakthrough Innovation
 *
 * Neural network twins rendered in 3D that breathe, pulse, and evolve.
 * Combines TwinManager (digital twin state), 3D renderer (visualization),
 * and training loop (ML) into a unified living system where architectures
 * exist as living entities that grow, adapt, and respond to their environment.
 *
 * This is the ultimate synthesis of all three domains: Dimensional Forge +
 * Intelligence Foundry + Digital Twins.
 *
 * @module LivingArchitectureLab
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
 * Living neuron in the architecture
 */
export interface LivingNeuron {
  id: string;
  position: Vector3D;

  // Biological metaphor
  vitality: number; // 0-1, health of neuron
  energy: number; // Current activation energy
  age: number; // Time since creation
  generation: number; // Evolution generation

  // Neural properties
  activation: number;
  bias: number;
  weights: Map<string, number>;

  // Living properties
  pulsePhase: number; // Current pulse animation phase
  pulseFrequency: number; // How fast it pulses
  growthRate: number; // Rate of weight change

  // Connections
  dendrites: string[]; // Incoming connections
  axons: string[]; // Outgoing connections

  // Visual
  color: string;
  size: number;
  glow: number;
}

/**
 * Living synapse (connection)
 */
export interface LivingSynapse {
  id: string;
  sourceNeuronId: string;
  targetNeuronId: string;

  // Synapse properties
  weight: number;
  strength: number; // Visual thickness

  // Living properties
  activity: number; // Current signal flow
  plasticity: number; // How easily it changes
  age: number;

  // LTP/LTD simulation
  recentActivity: number[]; // Recent activation history
  potentiation: number; // Long-term potentiation level

  // Visual
  color: string;
  particleFlow: number; // Signal particles flowing
}

/**
 * Living layer
 */
export interface LivingLayer {
  id: string;
  name: string;
  type: LivingLayerType;
  position: Vector3D;

  // Layer structure
  neurons: string[]; // Neuron IDs in this layer
  shape: number[]; // Layer shape

  // Living properties
  averageVitality: number;
  averageActivity: number;
  layerAge: number;

  // Boundaries
  bounds: {
    min: Vector3D;
    max: Vector3D;
  };

  // Visual
  color: string;
  opacity: number;
}

/**
 * Layer types
 */
export type LivingLayerType =
  | "input"
  | "dense"
  | "conv2d"
  | "recurrent"
  | "attention"
  | "output"
  | "residual"
  | "normalization";

/**
 * Living architecture - the neural network organism
 */
export interface LivingArchitecture {
  id: string;
  name: string;

  // Structure
  layers: Map<string, LivingLayer>;
  neurons: Map<string, LivingNeuron>;
  synapses: Map<string, LivingSynapse>;

  // Twin state
  twinId: string;
  twinState: ArchitectureTwinState;

  // Vital signs
  vitalSigns: VitalSigns;

  // Evolution
  generation: number;
  mutations: Mutation[];
  fitnessHistory: number[];

  // Training state
  trainingState: LiveTrainingState;

  // Lifecycle
  birthTime: number;
  lastUpdateTime: number;
  lifeStage: LifeStage;
}

/**
 * Digital twin state for architecture
 */
export interface ArchitectureTwinState {
  // Current state
  currentLoss: number;
  currentAccuracy: number;
  epoch: number;
  step: number;

  // Historical states for time travel
  stateHistory: ArchitectureStateSnapshot[];
  maxHistorySize: number;

  // Predicted future
  predictedTrajectory: PredictedState[];
}

/**
 * Architecture state snapshot
 */
export interface ArchitectureStateSnapshot {
  timestamp: number;
  loss: number;
  accuracy: number;
  weights: Map<string, Float32Array>;
  neuronStates: Map<string, NeuronStateSnapshot>;
}

/**
 * Neuron state snapshot
 */
export interface NeuronStateSnapshot {
  activation: number;
  vitality: number;
  energy: number;
}

/**
 * Predicted future state
 */
export interface PredictedState {
  timestep: number;
  predictedLoss: number;
  predictedAccuracy: number;
  confidence: number;
}

/**
 * Vital signs of the architecture
 */
export interface VitalSigns {
  // Health indicators
  overallHealth: number; // 0-1
  learningRate: number; // Effective learning rate
  gradientFlow: number; // How well gradients flow

  // Activity
  heartbeat: number; // Training step rate
  respiration: number; // Data throughput
  temperature: number; // Computational intensity

  // Diagnostics
  vanishingGradients: boolean;
  explodingGradients: boolean;
  deadNeurons: number;
  saturatedNeurons: number;
}

/**
 * Mutation in architecture
 */
export interface Mutation {
  id: string;
  timestamp: number;
  type: MutationType;
  target: string; // Layer/neuron ID affected
  description: string;
  impact: number; // How much it changed fitness
}

/**
 * Mutation types
 */
export type MutationType =
  | "add_neuron"
  | "remove_neuron"
  | "add_layer"
  | "remove_layer"
  | "change_activation"
  | "adjust_width"
  | "add_skip_connection"
  | "prune_synapse"
  | "weight_perturbation";

/**
 * Life stages
 */
export type LifeStage =
  | "embryonic" // Initial random weights
  | "infant" // Early training, rapid learning
  | "adolescent" // Finding structure
  | "adult" // Stable, refined
  | "mature" // Fully converged
  | "senescent"; // Overfitting, declining

/**
 * Live training state
 */
export interface LiveTrainingState {
  isTraining: boolean;
  isPaused: boolean;
  currentEpoch: number;
  currentBatch: number;
  totalEpochs: number;
  batchSize: number;

  // Metrics
  lossHistory: number[];
  accuracyHistory: number[];
  learningRateSchedule: number[];

  // Real-time
  currentLoss: number;
  currentAccuracy: number;
  samplesProcessed: number;
  trainingSpeed: number; // Samples per second
}

/**
 * Lab configuration
 */
export interface LabConfig {
  // Simulation
  simulationSpeed: number;
  physicsEnabled: boolean;

  // Visualization
  showConnections: boolean;
  showActivations: boolean;
  showGradients: boolean;
  pulseEnabled: boolean;

  // Evolution
  autoEvolve: boolean;
  evolutionRate: number;
  fitnessThreshold: number;

  // Training
  autoTrain: boolean;
  learningRate: number;
  optimizer: "sgd" | "adam" | "rmsprop";
}

/**
 * Lab event types
 */
export interface LabEvents {
  "architecture:created": { architecture: LivingArchitecture };
  "architecture:updated": { architectureId: string; changes: string[] };
  "architecture:evolved": { architectureId: string; mutation: Mutation };
  "architecture:died": { architectureId: string; cause: string };

  "neuron:born": { neuronId: string; layerId: string };
  "neuron:died": { neuronId: string; cause: string };
  "neuron:activated": { neuronId: string; activation: number };

  "synapse:formed": { synapseId: string; source: string; target: string };
  "synapse:pruned": { synapseId: string; reason: string };
  "synapse:strengthened": { synapseId: string; newWeight: number };

  "training:started": { architectureId: string };
  "training:step": { architectureId: string; loss: number; accuracy: number };
  "training:epoch": { architectureId: string; epoch: number };
  "training:completed": {
    architectureId: string;
    finalMetrics: Record<string, number>;
  };

  "vital:warning": { architectureId: string; warning: string };
  "vital:critical": { architectureId: string; issue: string };

  "twin:synced": { architectureId: string; twinId: string };
  "twin:diverged": { architectureId: string; divergence: number };
}

/**
 * Observation point for consciousness integration
 */
export interface ObservationPoint {
  position: Vector3D;
  direction: Vector3D;
  focus: string | null; // ID of focused element
  attentionRadius: number;
}

/**
 * Camera configuration
 */
export interface LabCamera {
  position: Vector3D;
  target: Vector3D;
  fov: number;
  near: number;
  far: number;
}

// ============================================================================
// Living Architecture Laboratory Implementation
// ============================================================================

/**
 * Living Architecture Laboratory
 *
 * The ultimate cross-domain synthesis: neural networks as living organisms
 * that breathe, grow, evolve, and respond to observation.
 */
export class LivingArchitectureLab extends EventEmitter {
  private config: LabConfig;
  private architectures: Map<string, LivingArchitecture>;
  private camera: LabCamera;
  private observation: ObservationPoint;

  // Simulation state
  private isRunning: boolean;
  private simulationTime: number;
  private lastFrameTime: number;
  private animationFrame: number | null;

  // Metrics
  private frameRate: number;
  private updateCount: number;

  constructor(config: Partial<LabConfig> = {}) {
    super();

    this.config = this.mergeConfig(config);
    this.architectures = new Map();
    this.camera = this.createDefaultCamera();
    this.observation = this.createDefaultObservation();

    this.isRunning = false;
    this.simulationTime = 0;
    this.lastFrameTime = 0;
    this.animationFrame = null;

    this.frameRate = 0;
    this.updateCount = 0;
  }

  /**
   * Merge user config with defaults
   */
  private mergeConfig(config: Partial<LabConfig>): LabConfig {
    return {
      simulationSpeed: config.simulationSpeed ?? 1.0,
      physicsEnabled: config.physicsEnabled ?? true,
      showConnections: config.showConnections ?? true,
      showActivations: config.showActivations ?? true,
      showGradients: config.showGradients ?? false,
      pulseEnabled: config.pulseEnabled ?? true,
      autoEvolve: config.autoEvolve ?? false,
      evolutionRate: config.evolutionRate ?? 0.01,
      fitnessThreshold: config.fitnessThreshold ?? 0.9,
      autoTrain: config.autoTrain ?? false,
      learningRate: config.learningRate ?? 0.001,
      optimizer: config.optimizer ?? "adam",
    };
  }

  /**
   * Create default camera
   */
  private createDefaultCamera(): LabCamera {
    return {
      position: { x: 0, y: 50, z: 100 },
      target: { x: 0, y: 0, z: 0 },
      fov: 60,
      near: 0.1,
      far: 2000,
    };
  }

  /**
   * Create default observation point
   */
  private createDefaultObservation(): ObservationPoint {
    return {
      position: { x: 0, y: 50, z: 100 },
      direction: { x: 0, y: 0, z: -1 },
      focus: null,
      attentionRadius: 50,
    };
  }

  // ============================================================================
  // Architecture Creation
  // ============================================================================

  /**
   * Create a living architecture from specification
   */
  createArchitecture(name: string, spec: ArchitectureSpec): LivingArchitecture {
    const id = this.generateId("arch");

    const architecture: LivingArchitecture = {
      id,
      name,
      layers: new Map(),
      neurons: new Map(),
      synapses: new Map(),
      twinId: this.generateId("twin"),
      twinState: this.createInitialTwinState(),
      vitalSigns: this.createInitialVitalSigns(),
      generation: 0,
      mutations: [],
      fitnessHistory: [],
      trainingState: this.createInitialTrainingState(),
      birthTime: Date.now(),
      lastUpdateTime: Date.now(),
      lifeStage: "embryonic",
    };

    // Build layers and neurons from spec
    this.buildArchitectureFromSpec(architecture, spec);

    this.architectures.set(id, architecture);

    this.emit("architecture:created", { architecture });

    return architecture;
  }

  /**
   * Build architecture from specification
   */
  private buildArchitectureFromSpec(
    architecture: LivingArchitecture,
    spec: ArchitectureSpec
  ): void {
    let zPosition = 0;
    let previousLayerNeurons: string[] = [];

    for (const layerSpec of spec.layers) {
      const layer = this.createLayer(architecture, layerSpec, zPosition);
      architecture.layers.set(layer.id, layer);

      // Create neurons for this layer
      const neuronIds = this.createNeuronsForLayer(
        architecture,
        layer,
        layerSpec
      );
      layer.neurons = neuronIds;

      // Connect to previous layer
      if (previousLayerNeurons.length > 0) {
        this.connectLayers(
          architecture,
          previousLayerNeurons,
          neuronIds,
          layerSpec.connectionType
        );
      }

      previousLayerNeurons = neuronIds;
      zPosition += 30; // Layer spacing
    }

    // Update layer bounds
    for (const layer of architecture.layers.values()) {
      this.updateLayerBounds(architecture, layer);
    }
  }

  /**
   * Create a layer
   */
  private createLayer(
    architecture: LivingArchitecture,
    spec: LayerSpec,
    zPosition: number
  ): LivingLayer {
    const id = this.generateId("layer");

    return {
      id,
      name: spec.name || `Layer ${architecture.layers.size + 1}`,
      type: spec.type,
      position: { x: 0, y: 0, z: zPosition },
      neurons: [],
      shape: spec.shape,
      averageVitality: 1.0,
      averageActivity: 0,
      layerAge: 0,
      bounds: {
        min: { x: -10, y: -10, z: zPosition - 5 },
        max: { x: 10, y: 10, z: zPosition + 5 },
      },
      color: this.getLayerColor(spec.type),
      opacity: 0.3,
    };
  }

  /**
   * Create neurons for a layer
   */
  private createNeuronsForLayer(
    architecture: LivingArchitecture,
    layer: LivingLayer,
    spec: LayerSpec
  ): string[] {
    const neuronIds: string[] = [];
    const numNeurons = spec.shape.reduce((a, b) => a * b, 1);

    // Arrange neurons in a grid pattern
    const gridSize = Math.ceil(Math.sqrt(numNeurons));
    const spacing = 5;

    for (let i = 0; i < numNeurons; i++) {
      const gridX = i % gridSize;
      const gridY = Math.floor(i / gridSize);

      const neuron = this.createNeuron(layer, {
        x: (gridX - gridSize / 2) * spacing,
        y: (gridY - gridSize / 2) * spacing,
        z: layer.position.z,
      });

      architecture.neurons.set(neuron.id, neuron);
      neuronIds.push(neuron.id);

      this.emit("neuron:born", { neuronId: neuron.id, layerId: layer.id });
    }

    return neuronIds;
  }

  /**
   * Create a single neuron
   */
  private createNeuron(layer: LivingLayer, position: Vector3D): LivingNeuron {
    const id = this.generateId("neuron");

    return {
      id,
      position,
      vitality: 1.0,
      energy: Math.random() * 0.5,
      age: 0,
      generation: 0,
      activation: 0,
      bias: (Math.random() - 0.5) * 0.1,
      weights: new Map(),
      pulsePhase: Math.random() * Math.PI * 2,
      pulseFrequency: 0.5 + Math.random() * 0.5,
      growthRate: 0.01,
      dendrites: [],
      axons: [],
      color: layer.color,
      size: 2,
      glow: 0.3,
    };
  }

  /**
   * Connect two layers
   */
  private connectLayers(
    architecture: LivingArchitecture,
    sourceNeurons: string[],
    targetNeurons: string[],
    connectionType: ConnectionType = "full"
  ): void {
    if (connectionType === "full") {
      // Fully connected
      for (const sourceId of sourceNeurons) {
        for (const targetId of targetNeurons) {
          this.createSynapse(architecture, sourceId, targetId);
        }
      }
    } else if (connectionType === "sparse") {
      // Sparse random connections
      const connectionProb = 0.3;
      for (const sourceId of sourceNeurons) {
        for (const targetId of targetNeurons) {
          if (Math.random() < connectionProb) {
            this.createSynapse(architecture, sourceId, targetId);
          }
        }
      }
    } else if (connectionType === "local") {
      // Local connections (for conv-like)
      const receptiveField = 3;
      for (let i = 0; i < targetNeurons.length; i++) {
        const start = Math.max(0, i - receptiveField);
        const end = Math.min(sourceNeurons.length, i + receptiveField);
        for (let j = start; j < end; j++) {
          this.createSynapse(architecture, sourceNeurons[j], targetNeurons[i]);
        }
      }
    }
  }

  /**
   * Create a synapse
   */
  private createSynapse(
    architecture: LivingArchitecture,
    sourceId: string,
    targetId: string
  ): LivingSynapse {
    const id = this.generateId("syn");

    const synapse: LivingSynapse = {
      id,
      sourceNeuronId: sourceId,
      targetNeuronId: targetId,
      weight: (Math.random() - 0.5) * 0.1,
      strength: 1,
      activity: 0,
      plasticity: 0.1,
      age: 0,
      recentActivity: [],
      potentiation: 0,
      color: "#88aaff",
      particleFlow: 0,
    };

    architecture.synapses.set(id, synapse);

    // Update neuron connections
    const source = architecture.neurons.get(sourceId);
    const target = architecture.neurons.get(targetId);

    if (source) {
      source.axons.push(id);
      source.weights.set(targetId, synapse.weight);
    }

    if (target) {
      target.dendrites.push(id);
    }

    this.emit("synapse:formed", {
      synapseId: id,
      source: sourceId,
      target: targetId,
    });

    return synapse;
  }

  /**
   * Update layer bounds
   */
  private updateLayerBounds(
    architecture: LivingArchitecture,
    layer: LivingLayer
  ): void {
    const neurons = layer.neurons
      .map((id) => architecture.neurons.get(id)!)
      .filter(Boolean);

    if (neurons.length === 0) return;

    layer.bounds.min = {
      x: Math.min(...neurons.map((n) => n.position.x)) - 5,
      y: Math.min(...neurons.map((n) => n.position.y)) - 5,
      z: layer.position.z - 5,
    };

    layer.bounds.max = {
      x: Math.max(...neurons.map((n) => n.position.x)) + 5,
      y: Math.max(...neurons.map((n) => n.position.y)) + 5,
      z: layer.position.z + 5,
    };
  }

  /**
   * Get layer color by type
   */
  private getLayerColor(type: LivingLayerType): string {
    const colors: Record<LivingLayerType, string> = {
      input: "#4fc3f7",
      dense: "#7986cb",
      conv2d: "#81c784",
      recurrent: "#ffb74d",
      attention: "#ba68c8",
      output: "#4db6ac",
      residual: "#ff8a65",
      normalization: "#90a4ae",
    };
    return colors[type] || "#ffffff";
  }

  /**
   * Create initial twin state
   */
  private createInitialTwinState(): ArchitectureTwinState {
    return {
      currentLoss: Infinity,
      currentAccuracy: 0,
      epoch: 0,
      step: 0,
      stateHistory: [],
      maxHistorySize: 100,
      predictedTrajectory: [],
    };
  }

  /**
   * Create initial vital signs
   */
  private createInitialVitalSigns(): VitalSigns {
    return {
      overallHealth: 1.0,
      learningRate: 0.001,
      gradientFlow: 1.0,
      heartbeat: 0,
      respiration: 0,
      temperature: 0.5,
      vanishingGradients: false,
      explodingGradients: false,
      deadNeurons: 0,
      saturatedNeurons: 0,
    };
  }

  /**
   * Create initial training state
   */
  private createInitialTrainingState(): LiveTrainingState {
    return {
      isTraining: false,
      isPaused: false,
      currentEpoch: 0,
      currentBatch: 0,
      totalEpochs: 100,
      batchSize: 32,
      lossHistory: [],
      accuracyHistory: [],
      learningRateSchedule: [],
      currentLoss: 0,
      currentAccuracy: 0,
      samplesProcessed: 0,
      trainingSpeed: 0,
    };
  }

  // ============================================================================
  // Simulation Loop
  // ============================================================================

  /**
   * Start the lab simulation
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.startAnimationLoop();
  }

  /**
   * Stop the simulation
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

      const deltaTime = (currentTime - this.lastFrameTime) / 1000;
      this.lastFrameTime = currentTime;

      this.update(deltaTime * this.config.simulationSpeed);

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
    this.updateCount++;

    // Update frame rate every second
    if (this.updateCount % 60 === 0) {
      this.frameRate = 60 / deltaTime;
    }

    for (const architecture of this.architectures.values()) {
      this.updateArchitecture(architecture, deltaTime);
    }

    this.emit("frame:updated", {
      simulationTime: this.simulationTime,
      deltaTime,
      frameRate: this.frameRate,
    });
  }

  /**
   * Update a single architecture
   */
  private updateArchitecture(
    architecture: LivingArchitecture,
    deltaTime: number
  ): void {
    architecture.lastUpdateTime = Date.now();

    // Update neurons
    for (const neuron of architecture.neurons.values()) {
      this.updateNeuron(architecture, neuron, deltaTime);
    }

    // Update synapses
    for (const synapse of architecture.synapses.values()) {
      this.updateSynapse(architecture, synapse, deltaTime);
    }

    // Update layers
    for (const layer of architecture.layers.values()) {
      this.updateLayer(architecture, layer, deltaTime);
    }

    // Update vital signs
    this.updateVitalSigns(architecture);

    // Update life stage
    this.updateLifeStage(architecture);

    // Auto-evolution if enabled
    if (this.config.autoEvolve) {
      this.checkEvolution(architecture);
    }

    // Sync with twin
    this.syncTwinState(architecture);
  }

  /**
   * Update a neuron
   */
  private updateNeuron(
    architecture: LivingArchitecture,
    neuron: LivingNeuron,
    deltaTime: number
  ): void {
    // Age the neuron
    neuron.age += deltaTime;

    // Update pulse animation
    if (this.config.pulseEnabled) {
      neuron.pulsePhase += deltaTime * neuron.pulseFrequency * Math.PI * 2;
      neuron.pulsePhase %= Math.PI * 2;
    }

    // Energy decay
    neuron.energy *= Math.exp(-0.1 * deltaTime);

    // Activation decay
    neuron.activation *= Math.exp(-0.5 * deltaTime);

    // Update glow based on activity
    neuron.glow = 0.3 + neuron.activation * 0.7;

    // Update size based on energy
    neuron.size = 2 + neuron.energy * 2;

    // Vitality decay (very slow)
    if (Math.abs(neuron.activation) < 0.01 && neuron.age > 100) {
      neuron.vitality -= 0.0001 * deltaTime;
    }

    // Check for dead neuron
    if (neuron.vitality <= 0) {
      this.emit("neuron:died", {
        neuronId: neuron.id,
        cause: "vitality_depleted",
      });
    }
  }

  /**
   * Update a synapse
   */
  private updateSynapse(
    architecture: LivingArchitecture,
    synapse: LivingSynapse,
    deltaTime: number
  ): void {
    // Age the synapse
    synapse.age += deltaTime;

    // Activity decay
    synapse.activity *= Math.exp(-1.0 * deltaTime);

    // Update particle flow visualization
    synapse.particleFlow = synapse.activity * 10;

    // Update color based on weight
    const weightMagnitude = Math.abs(synapse.weight);
    const isExcitatory = synapse.weight > 0;

    if (isExcitatory) {
      const intensity = Math.min(255, Math.floor(weightMagnitude * 500));
      synapse.color = `rgb(${intensity}, 100, 100)`;
    } else {
      const intensity = Math.min(255, Math.floor(weightMagnitude * 500));
      synapse.color = `rgb(100, 100, ${intensity})`;
    }

    // Update strength based on recent activity (Hebbian-like)
    if (synapse.recentActivity.length > 0) {
      const avgActivity =
        synapse.recentActivity.reduce((a, b) => a + b, 0) /
        synapse.recentActivity.length;
      synapse.potentiation +=
        (avgActivity - 0.5) * synapse.plasticity * deltaTime;
      synapse.potentiation = Math.max(-1, Math.min(1, synapse.potentiation));
    }

    // Trim activity history
    if (synapse.recentActivity.length > 10) {
      synapse.recentActivity.shift();
    }
  }

  /**
   * Update a layer
   */
  private updateLayer(
    architecture: LivingArchitecture,
    layer: LivingLayer,
    deltaTime: number
  ): void {
    layer.layerAge += deltaTime;

    // Compute average vitality and activity
    const neurons = layer.neurons
      .map((id) => architecture.neurons.get(id))
      .filter((n): n is LivingNeuron => n !== undefined);

    if (neurons.length > 0) {
      layer.averageVitality =
        neurons.reduce((sum, n) => sum + n.vitality, 0) / neurons.length;
      layer.averageActivity =
        neurons.reduce((sum, n) => sum + Math.abs(n.activation), 0) /
        neurons.length;
    }

    // Update opacity based on activity
    layer.opacity = 0.2 + layer.averageActivity * 0.3;
  }

  /**
   * Update vital signs
   */
  private updateVitalSigns(architecture: LivingArchitecture): void {
    const neurons = Array.from(architecture.neurons.values());
    const synapses = Array.from(architecture.synapses.values());

    // Count dead and saturated neurons
    let deadCount = 0;
    let saturatedCount = 0;

    for (const neuron of neurons) {
      if (neuron.vitality <= 0) deadCount++;
      if (Math.abs(neuron.activation) > 0.99) saturatedCount++;
    }

    architecture.vitalSigns.deadNeurons = deadCount;
    architecture.vitalSigns.saturatedNeurons = saturatedCount;

    // Overall health
    const healthFactor =
      1 - (deadCount + saturatedCount * 0.5) / Math.max(1, neurons.length);
    architecture.vitalSigns.overallHealth = Math.max(
      0,
      Math.min(1, healthFactor)
    );

    // Gradient flow (simplified - based on synapse activity)
    const avgSynapseActivity =
      synapses.length > 0
        ? synapses.reduce((sum, s) => sum + s.activity, 0) / synapses.length
        : 0;
    architecture.vitalSigns.gradientFlow = Math.min(1, avgSynapseActivity * 10);

    // Temperature (computational intensity)
    const avgActivity =
      neurons.length > 0
        ? neurons.reduce((sum, n) => sum + Math.abs(n.activation), 0) /
          neurons.length
        : 0;
    architecture.vitalSigns.temperature = avgActivity;

    // Emit warnings
    if (architecture.vitalSigns.overallHealth < 0.5) {
      this.emit("vital:warning", {
        architectureId: architecture.id,
        warning: "Low overall health",
      });
    }

    if (deadCount > neurons.length * 0.1) {
      this.emit("vital:critical", {
        architectureId: architecture.id,
        issue: `${deadCount} dead neurons detected`,
      });
    }
  }

  /**
   * Update life stage
   */
  private updateLifeStage(architecture: LivingArchitecture): void {
    const age = (Date.now() - architecture.birthTime) / 1000;
    const fitness =
      architecture.fitnessHistory.length > 0
        ? architecture.fitnessHistory[architecture.fitnessHistory.length - 1]
        : 0;

    // Simple life stage transitions
    if (age < 10) {
      architecture.lifeStage = "embryonic";
    } else if (age < 60 && fitness < 0.5) {
      architecture.lifeStage = "infant";
    } else if (fitness < 0.7) {
      architecture.lifeStage = "adolescent";
    } else if (fitness < 0.9) {
      architecture.lifeStage = "adult";
    } else if (architecture.vitalSigns.overallHealth > 0.8) {
      architecture.lifeStage = "mature";
    } else {
      architecture.lifeStage = "senescent";
    }
  }

  /**
   * Check for evolution
   */
  private checkEvolution(architecture: LivingArchitecture): void {
    if (Math.random() > this.config.evolutionRate) return;

    const fitness =
      architecture.fitnessHistory.length > 0
        ? architecture.fitnessHistory[architecture.fitnessHistory.length - 1]
        : 0;

    if (fitness < this.config.fitnessThreshold) {
      // Evolve
      this.evolveArchitecture(architecture);
    }
  }

  /**
   * Evolve architecture with a random mutation
   */
  evolveArchitecture(architecture: LivingArchitecture): Mutation {
    const mutations: MutationType[] = [
      "add_neuron",
      "prune_synapse",
      "weight_perturbation",
      "adjust_width",
    ];

    const mutationType =
      mutations[Math.floor(Math.random() * mutations.length)];

    const mutation = this.applyMutation(architecture, mutationType);

    architecture.generation++;
    architecture.mutations.push(mutation);

    this.emit("architecture:evolved", {
      architectureId: architecture.id,
      mutation,
    });

    return mutation;
  }

  /**
   * Apply a mutation
   */
  private applyMutation(
    architecture: LivingArchitecture,
    type: MutationType
  ): Mutation {
    const mutation: Mutation = {
      id: this.generateId("mut"),
      timestamp: Date.now(),
      type,
      target: "",
      description: "",
      impact: 0,
    };

    switch (type) {
      case "add_neuron": {
        // Add neuron to a random layer
        const layers = Array.from(architecture.layers.values());
        if (layers.length > 0) {
          const layer = layers[Math.floor(Math.random() * layers.length)];
          const position: Vector3D = {
            x: layer.position.x + (Math.random() - 0.5) * 20,
            y: layer.position.y + (Math.random() - 0.5) * 20,
            z: layer.position.z,
          };

          const neuron = this.createNeuron(layer, position);
          architecture.neurons.set(neuron.id, neuron);
          layer.neurons.push(neuron.id);

          mutation.target = neuron.id;
          mutation.description = `Added neuron to layer ${layer.name}`;
        }
        break;
      }

      case "prune_synapse": {
        // Remove weakest synapse
        const synapses = Array.from(architecture.synapses.values());
        if (synapses.length > 0) {
          const weakest = synapses.reduce((min, s) =>
            Math.abs(s.weight) < Math.abs(min.weight) ? s : min
          );

          this.removeSynapse(architecture, weakest.id);

          mutation.target = weakest.id;
          mutation.description = `Pruned weak synapse (weight: ${weakest.weight.toFixed(4)})`;
        }
        break;
      }

      case "weight_perturbation": {
        // Perturb all weights slightly
        for (const synapse of architecture.synapses.values()) {
          synapse.weight += (Math.random() - 0.5) * 0.01;
        }

        mutation.description = "Perturbed all weights";
        break;
      }

      case "adjust_width": {
        // Randomly add or remove neurons
        const layers = Array.from(architecture.layers.values());
        if (layers.length > 0) {
          const layer = layers[Math.floor(Math.random() * layers.length)];

          if (Math.random() > 0.5 && layer.neurons.length > 1) {
            // Remove random neuron
            const neuronId =
              layer.neurons[Math.floor(Math.random() * layer.neurons.length)];
            this.removeNeuron(architecture, neuronId);
            mutation.description = `Removed neuron from layer ${layer.name}`;
          } else {
            // Add neuron
            const position: Vector3D = {
              x: layer.position.x + (Math.random() - 0.5) * 20,
              y: layer.position.y + (Math.random() - 0.5) * 20,
              z: layer.position.z,
            };

            const neuron = this.createNeuron(layer, position);
            architecture.neurons.set(neuron.id, neuron);
            layer.neurons.push(neuron.id);

            mutation.description = `Added neuron to layer ${layer.name}`;
          }

          mutation.target = layer.id;
        }
        break;
      }
    }

    return mutation;
  }

  /**
   * Remove a synapse
   */
  private removeSynapse(
    architecture: LivingArchitecture,
    synapseId: string
  ): void {
    const synapse = architecture.synapses.get(synapseId);
    if (!synapse) return;

    // Remove from neurons
    const source = architecture.neurons.get(synapse.sourceNeuronId);
    const target = architecture.neurons.get(synapse.targetNeuronId);

    if (source) {
      source.axons = source.axons.filter((id) => id !== synapseId);
      source.weights.delete(synapse.targetNeuronId);
    }

    if (target) {
      target.dendrites = target.dendrites.filter((id) => id !== synapseId);
    }

    architecture.synapses.delete(synapseId);

    this.emit("synapse:pruned", { synapseId, reason: "mutation" });
  }

  /**
   * Remove a neuron
   */
  private removeNeuron(
    architecture: LivingArchitecture,
    neuronId: string
  ): void {
    const neuron = architecture.neurons.get(neuronId);
    if (!neuron) return;

    // Remove all connected synapses
    for (const synapseId of [...neuron.dendrites, ...neuron.axons]) {
      this.removeSynapse(architecture, synapseId);
    }

    // Remove from layer
    for (const layer of architecture.layers.values()) {
      layer.neurons = layer.neurons.filter((id) => id !== neuronId);
    }

    architecture.neurons.delete(neuronId);

    this.emit("neuron:died", { neuronId, cause: "mutation" });
  }

  /**
   * Sync twin state
   */
  private syncTwinState(architecture: LivingArchitecture): void {
    // Create state snapshot
    const snapshot: ArchitectureStateSnapshot = {
      timestamp: Date.now(),
      loss: architecture.twinState.currentLoss,
      accuracy: architecture.twinState.currentAccuracy,
      weights: new Map(),
      neuronStates: new Map(),
    };

    // Capture neuron states
    for (const [id, neuron] of architecture.neurons) {
      snapshot.neuronStates.set(id, {
        activation: neuron.activation,
        vitality: neuron.vitality,
        energy: neuron.energy,
      });
    }

    // Add to history
    architecture.twinState.stateHistory.push(snapshot);

    // Limit history size
    while (
      architecture.twinState.stateHistory.length >
      architecture.twinState.maxHistorySize
    ) {
      architecture.twinState.stateHistory.shift();
    }

    this.emit("twin:synced", {
      architectureId: architecture.id,
      twinId: architecture.twinId,
    });
  }

  // ============================================================================
  // Forward Pass Visualization
  // ============================================================================

  /**
   * Run a forward pass with visualization
   */
  async forwardPass(
    architectureId: string,
    input: number[]
  ): Promise<number[]> {
    const architecture = this.architectures.get(architectureId);
    if (!architecture)
      throw new Error(`Architecture ${architectureId} not found`);

    // Get layers in order (by z position)
    const orderedLayers = Array.from(architecture.layers.values()).sort(
      (a, b) => a.position.z - b.position.z
    );

    // Set input layer activations
    const inputLayer = orderedLayers[0];
    const inputNeurons = inputLayer.neurons.map(
      (id) => architecture.neurons.get(id)!
    );

    for (let i = 0; i < Math.min(input.length, inputNeurons.length); i++) {
      inputNeurons[i].activation = input[i];
      inputNeurons[i].energy = Math.abs(input[i]);

      this.emit("neuron:activated", {
        neuronId: inputNeurons[i].id,
        activation: input[i],
      });
    }

    // Propagate through layers
    for (let l = 1; l < orderedLayers.length; l++) {
      const layer = orderedLayers[l];

      for (const neuronId of layer.neurons) {
        const neuron = architecture.neurons.get(neuronId)!;

        // Compute weighted sum
        let sum = neuron.bias;

        for (const synapseId of neuron.dendrites) {
          const synapse = architecture.synapses.get(synapseId)!;
          const sourceNeuron = architecture.neurons.get(
            synapse.sourceNeuronId
          )!;

          const contribution = sourceNeuron.activation * synapse.weight;
          sum += contribution;

          // Visualize signal flow
          synapse.activity = Math.abs(contribution);
          synapse.recentActivity.push(synapse.activity);
        }

        // Apply activation (ReLU for simplicity)
        neuron.activation = Math.max(0, sum);
        neuron.energy = Math.abs(neuron.activation);

        this.emit("neuron:activated", {
          neuronId,
          activation: neuron.activation,
        });
      }

      // Small delay for visualization
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    // Get output
    const outputLayer = orderedLayers[orderedLayers.length - 1];
    const outputs = outputLayer.neurons.map(
      (id) => architecture.neurons.get(id)!.activation
    );

    return outputs;
  }

  // ============================================================================
  // Training
  // ============================================================================

  /**
   * Start training
   */
  startTraining(architectureId: string, config: TrainingConfig): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    architecture.trainingState.isTraining = true;
    architecture.trainingState.isPaused = false;
    architecture.trainingState.totalEpochs = config.epochs;
    architecture.trainingState.batchSize = config.batchSize;

    this.config.learningRate = config.learningRate;

    this.emit("training:started", { architectureId });

    // Training would be implemented with actual ML framework integration
  }

  /**
   * Pause training
   */
  pauseTraining(architectureId: string): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    architecture.trainingState.isPaused = true;
  }

  /**
   * Resume training
   */
  resumeTraining(architectureId: string): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    architecture.trainingState.isPaused = false;
  }

  /**
   * Stop training
   */
  stopTraining(architectureId: string): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    architecture.trainingState.isTraining = false;

    this.emit("training:completed", {
      architectureId,
      finalMetrics: {
        loss: architecture.trainingState.currentLoss,
        accuracy: architecture.trainingState.currentAccuracy,
      },
    });
  }

  /**
   * Record training step
   */
  recordTrainingStep(
    architectureId: string,
    loss: number,
    accuracy: number
  ): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    architecture.trainingState.currentLoss = loss;
    architecture.trainingState.currentAccuracy = accuracy;
    architecture.trainingState.lossHistory.push(loss);
    architecture.trainingState.accuracyHistory.push(accuracy);
    architecture.trainingState.currentBatch++;

    architecture.twinState.currentLoss = loss;
    architecture.twinState.currentAccuracy = accuracy;
    architecture.twinState.step++;

    // Update fitness history
    architecture.fitnessHistory.push(accuracy);

    // Update vital signs heartbeat
    architecture.vitalSigns.heartbeat = 1;

    this.emit("training:step", { architectureId, loss, accuracy });
  }

  // ============================================================================
  // Time Travel (Twin State History)
  // ============================================================================

  /**
   * Travel to a specific point in the architecture's history
   */
  timeTravel(architectureId: string, timestamp: number): boolean {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return false;

    // Find closest snapshot
    const history = architecture.twinState.stateHistory;
    if (history.length === 0) return false;

    let closest = history[0];
    let minDiff = Math.abs(history[0].timestamp - timestamp);

    for (const snapshot of history) {
      const diff = Math.abs(snapshot.timestamp - timestamp);
      if (diff < minDiff) {
        minDiff = diff;
        closest = snapshot;
      }
    }

    // Restore state
    this.restoreSnapshot(architecture, closest);

    return true;
  }

  /**
   * Restore architecture to a snapshot
   */
  private restoreSnapshot(
    architecture: LivingArchitecture,
    snapshot: ArchitectureStateSnapshot
  ): void {
    // Restore neuron states
    for (const [id, state] of snapshot.neuronStates) {
      const neuron = architecture.neurons.get(id);
      if (neuron) {
        neuron.activation = state.activation;
        neuron.vitality = state.vitality;
        neuron.energy = state.energy;
      }
    }

    // Restore metrics
    architecture.twinState.currentLoss = snapshot.loss;
    architecture.twinState.currentAccuracy = snapshot.accuracy;

    this.emit("architecture:updated", {
      architectureId: architecture.id,
      changes: ["state_restored"],
    });
  }

  // ============================================================================
  // Observation (Consciousness Integration)
  // ============================================================================

  /**
   * Update observation point
   */
  setObservation(observation: Partial<ObservationPoint>): void {
    Object.assign(this.observation, observation);

    // Observation affects the system (consciousness-aware)
    if (this.observation.focus) {
      this.applyObservationEffect(this.observation.focus);
    }
  }

  /**
   * Apply observation effect (consciousness collapses possibilities)
   */
  private applyObservationEffect(focusId: string): void {
    // When observed, neurons stabilize their activation
    for (const architecture of this.architectures.values()) {
      const neuron = architecture.neurons.get(focusId);
      if (neuron) {
        // Observation reduces uncertainty
        neuron.pulseFrequency *= 0.9; // Slower pulse when observed
        neuron.energy *= 1.1; // Slight energy boost
      }

      const synapse = architecture.synapses.get(focusId);
      if (synapse) {
        // Observation strengthens the synapse slightly
        synapse.potentiation += 0.01;
      }
    }
  }

  // ============================================================================
  // Camera Control
  // ============================================================================

  /**
   * Set camera
   */
  setCamera(camera: Partial<LabCamera>): void {
    Object.assign(this.camera, camera);
    this.emit("camera:changed", this.camera);
  }

  /**
   * Get camera
   */
  getCamera(): LabCamera {
    return { ...this.camera };
  }

  /**
   * Focus camera on architecture
   */
  focusOnArchitecture(architectureId: string): void {
    const architecture = this.architectures.get(architectureId);
    if (!architecture) return;

    // Find center of architecture
    const neurons = Array.from(architecture.neurons.values());
    if (neurons.length === 0) return;

    const center = {
      x: neurons.reduce((s, n) => s + n.position.x, 0) / neurons.length,
      y: neurons.reduce((s, n) => s + n.position.y, 0) / neurons.length,
      z: neurons.reduce((s, n) => s + n.position.z, 0) / neurons.length,
    };

    this.camera.target = center;
    this.camera.position = {
      x: center.x + 50,
      y: center.y + 50,
      z: center.z + 100,
    };

    this.emit("camera:changed", this.camera);
  }

  // ============================================================================
  // Presets
  // ============================================================================

  /**
   * Create MLP preset
   */
  createMLPPreset(
    name: string,
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number
  ): LivingArchitecture {
    const layers: LayerSpec[] = [
      {
        name: "Input",
        type: "input",
        shape: [inputSize],
        connectionType: "full",
      },
    ];

    for (let i = 0; i < hiddenSizes.length; i++) {
      layers.push({
        name: `Hidden ${i + 1}`,
        type: "dense",
        shape: [hiddenSizes[i]],
        connectionType: "full",
      });
    }

    layers.push({
      name: "Output",
      type: "output",
      shape: [outputSize],
      connectionType: "full",
    });

    return this.createArchitecture(name, { layers });
  }

  /**
   * Create autoencoder preset
   */
  createAutoencoderPreset(
    name: string,
    inputSize: number,
    latentSize: number
  ): LivingArchitecture {
    const encoderSizes = [
      Math.floor(inputSize * 0.75),
      Math.floor(inputSize * 0.5),
    ];
    const decoderSizes = encoderSizes.slice().reverse();

    const layers: LayerSpec[] = [
      {
        name: "Input",
        type: "input",
        shape: [inputSize],
        connectionType: "full",
      },
    ];

    for (let i = 0; i < encoderSizes.length; i++) {
      layers.push({
        name: `Encoder ${i + 1}`,
        type: "dense",
        shape: [encoderSizes[i]],
        connectionType: "full",
      });
    }

    layers.push({
      name: "Latent",
      type: "dense",
      shape: [latentSize],
      connectionType: "full",
    });

    for (let i = 0; i < decoderSizes.length; i++) {
      layers.push({
        name: `Decoder ${i + 1}`,
        type: "dense",
        shape: [decoderSizes[i]],
        connectionType: "full",
      });
    }

    layers.push({
      name: "Output",
      type: "output",
      shape: [inputSize],
      connectionType: "full",
    });

    return this.createArchitecture(name, { layers });
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Generate unique ID
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get architecture
   */
  getArchitecture(id: string): LivingArchitecture | undefined {
    return this.architectures.get(id);
  }

  /**
   * Get all architectures
   */
  getAllArchitectures(): LivingArchitecture[] {
    return Array.from(this.architectures.values());
  }

  /**
   * Remove architecture
   */
  removeArchitecture(id: string): void {
    const architecture = this.architectures.get(id);
    if (!architecture) return;

    this.architectures.delete(id);

    this.emit("architecture:died", { architectureId: id, cause: "removed" });
  }

  /**
   * Get configuration
   */
  getConfig(): LabConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<LabConfig>): void {
    Object.assign(this.config, config);
    this.emit("config:changed", this.config);
  }

  /**
   * Get simulation state
   */
  getSimulationState(): {
    isRunning: boolean;
    simulationTime: number;
    frameRate: number;
  } {
    return {
      isRunning: this.isRunning,
      simulationTime: this.simulationTime,
      frameRate: this.frameRate,
    };
  }

  /**
   * Export architecture
   */
  exportArchitecture(id: string): string {
    const architecture = this.architectures.get(id);
    if (!architecture) throw new Error(`Architecture ${id} not found`);

    return JSON.stringify(
      {
        id: architecture.id,
        name: architecture.name,
        layers: Array.from(architecture.layers.values()),
        neurons: Array.from(architecture.neurons.entries()).map(([id, n]) => ({
          id,
          position: n.position,
          bias: n.bias,
          weights: Array.from(n.weights.entries()),
        })),
        synapses: Array.from(architecture.synapses.values()).map((s) => ({
          id: s.id,
          source: s.sourceNeuronId,
          target: s.targetNeuronId,
          weight: s.weight,
        })),
        generation: architecture.generation,
        lifeStage: architecture.lifeStage,
      },
      null,
      2
    );
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.stop();
    this.architectures.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Supporting Types
// ============================================================================

/**
 * Architecture specification
 */
export interface ArchitectureSpec {
  layers: LayerSpec[];
}

/**
 * Layer specification
 */
export interface LayerSpec {
  name?: string;
  type: LivingLayerType;
  shape: number[];
  connectionType?: ConnectionType;
}

/**
 * Connection types
 */
export type ConnectionType = "full" | "sparse" | "local" | "residual";

/**
 * Training configuration
 */
export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: "sgd" | "adam" | "rmsprop";
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create Living Architecture Laboratory
 */
export function createLivingArchitectureLab(
  config?: Partial<LabConfig>
): LivingArchitectureLab {
  return new LivingArchitectureLab(config);
}
