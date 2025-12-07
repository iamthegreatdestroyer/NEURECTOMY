/**
 * 3D Neural Playground
 *
 * Interactive 3D neural network visualization with real-time training.
 * Users can explore, manipulate, and train neural networks in an immersive
 * 3D environment powered by the Dimensional Forge.
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations/neural-playground
 * @agents @NEXUS @TENSOR @CANVAS
 * @innovation Forge×Foundry Synergy #1
 *
 * ## Concept
 *
 * Traditional neural network visualization is 2D and static. The 3D Neural
 * Playground provides:
 * 1. Spatial network architecture visualization
 * 2. Real-time weight/gradient visualization as flowing particles
 * 3. Interactive node manipulation (add/remove/connect neurons)
 * 4. Live training with 3D loss landscape visualization
 * 5. VR-ready exploration of network internals
 *
 * ## Architecture
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                       Neural3DPlayground                            │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
 * │  │   Network   │────▶│  Renderer   │────▶│  3D Scene   │          │
 * │  │   Model     │     │   Engine    │     │   Output    │          │
 * │  └──────┬──────┘     └──────┬──────┘     └─────────────┘          │
 * │         │                   │                                      │
 * │         ▼                   ▼                                      │
 * │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
 * │  │  Training   │────▶│ Interaction │◀────│   Camera    │          │
 * │  │   Loop      │     │   Handler   │     │  Controls   │          │
 * │  └─────────────┘     └─────────────┘     └─────────────┘          │
 * └─────────────────────────────────────────────────────────────────────┘
 * ```
 */

import { CrossDomainEventBridge, type CrossDomainEvent } from "../event-bridge";
import type { UniversalId } from "../types";

// ============================================================================
// TYPES
// ============================================================================

/**
 * Configuration for the 3D Neural Playground
 */
export interface NeuralPlaygroundConfig {
  /** Unique playground ID */
  id?: string;
  /** Canvas element ID or element */
  canvas: string | HTMLCanvasElement;
  /** Initial network architecture */
  architecture?: NetworkArchitecture;
  /** Visualization settings */
  visualization: VisualizationConfig;
  /** Training settings */
  training: TrainingConfig;
  /** Interaction settings */
  interaction: InteractionConfig;
  /** Camera settings */
  camera: CameraConfig;
  /** VR/AR settings */
  xr?: XRConfig;
}

/**
 * Network architecture definition
 */
export interface NetworkArchitecture {
  /** Architecture name */
  name: string;
  /** Layer definitions */
  layers: LayerDefinition[];
  /** Connection style */
  connectionStyle: "full" | "sparse" | "skip" | "custom";
  /** Custom connections (for connectionStyle: "custom") */
  customConnections?: Connection[];
}

/**
 * Layer definition
 */
export interface LayerDefinition {
  /** Layer ID */
  id: string;
  /** Layer type */
  type:
    | "input"
    | "dense"
    | "conv2d"
    | "conv3d"
    | "lstm"
    | "attention"
    | "embedding"
    | "output";
  /** Number of neurons/units */
  units: number;
  /** Activation function */
  activation?: string;
  /** Layer-specific parameters */
  params?: Record<string, unknown>;
  /** 3D position hint */
  position?: Vector3;
  /** Layer color */
  color?: string;
}

/**
 * Connection between layers
 */
export interface Connection {
  /** Source layer ID */
  fromLayerId: string;
  /** Target layer ID */
  toLayerId: string;
  /** Connection type */
  type: "forward" | "skip" | "residual" | "attention";
  /** Connection weight (for visualization) */
  weight?: number;
}

/**
 * 3D Vector
 */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/**
 * Visualization configuration
 */
export interface VisualizationConfig {
  /** Neuron representation style */
  neuronStyle: "sphere" | "cube" | "point" | "glow";
  /** Neuron base size */
  neuronSize: number;
  /** Show activation values as size/color */
  showActivations: boolean;
  /** Activation color gradient */
  activationGradient: [string, string, string]; // [negative, zero, positive]
  /** Connection visualization */
  connectionStyle: "line" | "particle" | "tube" | "ribbon";
  /** Show weight flow as particles */
  showWeightFlow: boolean;
  /** Particle speed multiplier */
  particleSpeed: number;
  /** Show gradient magnitude */
  showGradients: boolean;
  /** Gradient visualization mode */
  gradientMode: "color" | "size" | "pulse";
  /** Background style */
  background: "dark" | "light" | "space" | "grid" | "custom";
  /** Custom background color */
  backgroundColor?: string;
  /** Enable bloom effect */
  enableBloom: boolean;
  /** Bloom intensity */
  bloomIntensity: number;
  /** Enable depth of field */
  enableDOF: boolean;
  /** Show layer boundaries */
  showLayerBounds: boolean;
  /** Show metrics overlay */
  showMetrics: boolean;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  /** Enable live training */
  enableLiveTraining: boolean;
  /** Training batch size */
  batchSize: number;
  /** Learning rate */
  learningRate: number;
  /** Optimizer */
  optimizer: "sgd" | "adam" | "rmsprop" | "adamw";
  /** Loss function */
  lossFunction: "mse" | "cross_entropy" | "binary_cross_entropy" | "custom";
  /** Update visualization every N steps */
  visualizationUpdateInterval: number;
  /** Show loss landscape */
  showLossLandscape: boolean;
  /** Loss landscape resolution */
  lossLandscapeResolution: number;
}

/**
 * Interaction configuration
 */
export interface InteractionConfig {
  /** Enable node selection */
  enableSelection: boolean;
  /** Enable drag to move nodes */
  enableDrag: boolean;
  /** Enable adding new neurons */
  enableAddNeuron: boolean;
  /** Enable removing neurons */
  enableRemoveNeuron: boolean;
  /** Enable creating connections */
  enableCreateConnection: boolean;
  /** Enable weight editing */
  enableWeightEdit: boolean;
  /** Enable architecture editing */
  enableArchitectureEdit: boolean;
  /** Interaction mode */
  mode: "explore" | "edit" | "train";
  /** Highlight on hover */
  highlightOnHover: boolean;
  /** Show tooltips */
  showTooltips: boolean;
}

/**
 * Camera configuration
 */
export interface CameraConfig {
  /** Camera type */
  type: "perspective" | "orthographic";
  /** Initial position */
  position: Vector3;
  /** Look-at target */
  target: Vector3;
  /** Field of view (degrees) */
  fov: number;
  /** Enable orbit controls */
  enableOrbit: boolean;
  /** Enable zoom */
  enableZoom: boolean;
  /** Enable pan */
  enablePan: boolean;
  /** Auto-rotate */
  autoRotate: boolean;
  /** Auto-rotate speed */
  autoRotateSpeed: number;
}

/**
 * XR (VR/AR) configuration
 */
export interface XRConfig {
  /** Enable VR mode */
  enableVR: boolean;
  /** Enable AR mode */
  enableAR: boolean;
  /** Hand tracking */
  handTracking: boolean;
  /** Controller interaction */
  controllerInteraction: boolean;
  /** Teleportation movement */
  teleportation: boolean;
  /** Scale in XR */
  scale: number;
}

/**
 * Neuron state for visualization
 */
export interface NeuronState {
  /** Neuron ID */
  id: UniversalId;
  /** Layer ID */
  layerId: string;
  /** Position in layer */
  index: number;
  /** 3D position */
  position: Vector3;
  /** Current activation value */
  activation: number;
  /** Current gradient */
  gradient: number;
  /** Is selected */
  selected: boolean;
  /** Is highlighted */
  highlighted: boolean;
  /** Bias value */
  bias: number;
}

/**
 * Weight state for visualization
 */
export interface WeightState {
  /** Weight ID */
  id: UniversalId;
  /** Source neuron ID */
  fromNeuronId: UniversalId;
  /** Target neuron ID */
  toNeuronId: UniversalId;
  /** Weight value */
  value: number;
  /** Gradient for this weight */
  gradient: number;
  /** Particle positions (for flow visualization) */
  particles: Vector3[];
}

/**
 * Playground state
 */
export interface PlaygroundState {
  /** Playground ID */
  id: UniversalId;
  /** Current architecture */
  architecture: NetworkArchitecture;
  /** Neuron states */
  neurons: Map<UniversalId, NeuronState>;
  /** Weight states */
  weights: Map<UniversalId, WeightState>;
  /** Training state */
  training: {
    isTraining: boolean;
    currentEpoch: number;
    currentBatch: number;
    currentLoss: number;
    lossHistory: number[];
    accuracy: number;
  };
  /** Camera state */
  camera: {
    position: Vector3;
    target: Vector3;
    zoom: number;
  };
  /** Selection state */
  selection: {
    selectedNeurons: UniversalId[];
    selectedWeights: UniversalId[];
    hoveredNeuron: UniversalId | null;
    hoveredWeight: UniversalId | null;
  };
  /** Interaction mode */
  mode: "explore" | "edit" | "train";
  /** Is rendering */
  isRendering: boolean;
}

/**
 * Training step result
 */
export interface TrainingStepResult {
  /** Step number */
  step: number;
  /** Epoch number */
  epoch: number;
  /** Loss value */
  loss: number;
  /** Accuracy */
  accuracy: number;
  /** Per-layer activations */
  activations: Map<string, number[]>;
  /** Per-layer gradients */
  gradients: Map<string, number[]>;
  /** Weight updates */
  weightUpdates: Map<UniversalId, number>;
}

/**
 * Playground event
 */
export interface PlaygroundEvent {
  type: PlaygroundEventType;
  payload: unknown;
  timestamp: number;
}

export type PlaygroundEventType =
  | "neuron:selected"
  | "neuron:added"
  | "neuron:removed"
  | "neuron:moved"
  | "weight:selected"
  | "weight:edited"
  | "connection:created"
  | "connection:removed"
  | "layer:added"
  | "layer:removed"
  | "training:started"
  | "training:step"
  | "training:paused"
  | "training:completed"
  | "camera:moved"
  | "mode:changed"
  | "xr:entered"
  | "xr:exited";

// ============================================================================
// IMPLEMENTATION
// ============================================================================

/**
 * 3D Neural Playground
 *
 * Interactive 3D environment for exploring and training neural networks.
 *
 * @example
 * ```typescript
 * const playground = new Neural3DPlayground({
 *   canvas: 'neural-canvas',
 *   architecture: {
 *     name: 'Simple MLP',
 *     layers: [
 *       { id: 'input', type: 'input', units: 784 },
 *       { id: 'hidden1', type: 'dense', units: 256, activation: 'relu' },
 *       { id: 'hidden2', type: 'dense', units: 128, activation: 'relu' },
 *       { id: 'output', type: 'output', units: 10, activation: 'softmax' },
 *     ],
 *     connectionStyle: 'full',
 *   },
 *   visualization: {
 *     neuronStyle: 'glow',
 *     showActivations: true,
 *     showWeightFlow: true,
 *     enableBloom: true,
 *   },
 *   training: {
 *     enableLiveTraining: true,
 *     learningRate: 0.001,
 *     optimizer: 'adam',
 *   },
 *   interaction: {
 *     enableSelection: true,
 *     enableArchitectureEdit: true,
 *     mode: 'explore',
 *   },
 *   camera: {
 *     type: 'perspective',
 *     position: { x: 0, y: 0, z: 50 },
 *     target: { x: 0, y: 0, z: 0 },
 *     enableOrbit: true,
 *   },
 * });
 *
 * // Initialize and render
 * await playground.initialize();
 * playground.startRendering();
 *
 * // Load training data and train
 * await playground.loadTrainingData(mnist);
 * playground.startTraining();
 *
 * // Subscribe to events
 * playground.on('neuron:selected', (neuron) => {
 *   console.log('Selected neuron:', neuron);
 * });
 * ```
 */
export class Neural3DPlayground {
  private config: NeuralPlaygroundConfig;
  private eventBridge: CrossDomainEventBridge;

  private state: PlaygroundState;
  private canvas: HTMLCanvasElement | null = null;
  private animationFrameId: number | null = null;

  private listeners: Map<PlaygroundEventType, Set<(payload: unknown) => void>> =
    new Map();
  private trainingData: unknown[] = [];
  private validationData: unknown[] = [];

  constructor(config: NeuralPlaygroundConfig) {
    this.config = {
      visualization: {
        neuronStyle: "sphere",
        neuronSize: 1,
        showActivations: true,
        activationGradient: ["#ff0000", "#ffffff", "#00ff00"],
        connectionStyle: "particle",
        showWeightFlow: true,
        particleSpeed: 1,
        showGradients: true,
        gradientMode: "color",
        background: "dark",
        enableBloom: true,
        bloomIntensity: 1.5,
        enableDOF: false,
        showLayerBounds: true,
        showMetrics: true,
        ...config.visualization,
      },
      training: {
        enableLiveTraining: true,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: "adam",
        lossFunction: "cross_entropy",
        visualizationUpdateInterval: 10,
        showLossLandscape: false,
        lossLandscapeResolution: 50,
        ...config.training,
      },
      interaction: {
        enableSelection: true,
        enableDrag: true,
        enableAddNeuron: true,
        enableRemoveNeuron: true,
        enableCreateConnection: true,
        enableWeightEdit: true,
        enableArchitectureEdit: true,
        mode: "explore",
        highlightOnHover: true,
        showTooltips: true,
        ...config.interaction,
      },
      camera: {
        type: "perspective",
        position: { x: 0, y: 0, z: 50 },
        target: { x: 0, y: 0, z: 0 },
        fov: 60,
        enableOrbit: true,
        enableZoom: true,
        enablePan: true,
        autoRotate: false,
        autoRotateSpeed: 0.5,
        ...config.camera,
      },
      ...config,
    };

    this.eventBridge = CrossDomainEventBridge.getInstance();

    // Initialize state
    this.state = {
      id: config.id || `playground-${Date.now()}`,
      architecture: config.architecture || {
        name: "Empty Network",
        layers: [],
        connectionStyle: "full",
      },
      neurons: new Map(),
      weights: new Map(),
      training: {
        isTraining: false,
        currentEpoch: 0,
        currentBatch: 0,
        currentLoss: 0,
        lossHistory: [],
        accuracy: 0,
      },
      camera: {
        position: this.config.camera.position,
        target: this.config.camera.target,
        zoom: 1,
      },
      selection: {
        selectedNeurons: [],
        selectedWeights: [],
        hoveredNeuron: null,
        hoveredWeight: null,
      },
      mode: this.config.interaction.mode,
      isRendering: false,
    };
  }

  /**
   * Initialize the playground
   */
  public async initialize(): Promise<void> {
    // Get canvas
    if (typeof this.config.canvas === "string") {
      this.canvas = document.getElementById(
        this.config.canvas
      ) as HTMLCanvasElement;
    } else {
      this.canvas = this.config.canvas;
    }

    if (!this.canvas) {
      throw new Error("Canvas element not found");
    }

    // Initialize 3D renderer (would use Three.js or similar in real implementation)
    await this.initializeRenderer();

    // Build network visualization from architecture
    if (this.config.architecture) {
      await this.buildNetworkVisualization(this.config.architecture);
    }

    // Setup event handlers
    this.setupEventHandlers();

    // Emit initialization event
    this.eventBridge.emit({
      id: this.generateId(),
      type: "forge:playground:initialized",
      source: "forge",
      timestamp: Date.now(),
      payload: { playgroundId: this.state.id },
    });
  }

  /**
   * Start rendering loop
   */
  public startRendering(): void {
    if (this.state.isRendering) return;

    this.state.isRendering = true;
    this.renderLoop();
  }

  /**
   * Stop rendering loop
   */
  public stopRendering(): void {
    this.state.isRendering = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Load training data
   */
  public async loadTrainingData(
    data: unknown[],
    validationSplit = 0.2
  ): Promise<void> {
    const splitIndex = Math.floor(data.length * (1 - validationSplit));
    this.trainingData = data.slice(0, splitIndex);
    this.validationData = data.slice(splitIndex);
  }

  /**
   * Start training
   */
  public startTraining(): void {
    if (this.state.training.isTraining) return;

    this.state.training.isTraining = true;
    this.state.mode = "train";

    this.emit("training:started", {
      config: this.config.training,
      dataSize: this.trainingData.length,
    });

    this.trainingLoop();
  }

  /**
   * Pause training
   */
  public pauseTraining(): void {
    this.state.training.isTraining = false;
    this.emit("training:paused", {
      epoch: this.state.training.currentEpoch,
      loss: this.state.training.currentLoss,
    });
  }

  /**
   * Resume training
   */
  public resumeTraining(): void {
    if (this.state.training.isTraining) return;

    this.state.training.isTraining = true;
    this.trainingLoop();
  }

  /**
   * Set interaction mode
   */
  public setMode(mode: "explore" | "edit" | "train"): void {
    this.state.mode = mode;
    this.emit("mode:changed", { mode });
  }

  /**
   * Get current state
   */
  public getState(): PlaygroundState {
    return { ...this.state };
  }

  /**
   * Set network architecture
   */
  public async setArchitecture(
    architecture: NetworkArchitecture
  ): Promise<void> {
    this.state.architecture = architecture;
    await this.buildNetworkVisualization(architecture);
  }

  /**
   * Add a layer
   */
  public addLayer(layer: LayerDefinition): void {
    if (!this.config.interaction.enableArchitectureEdit) return;

    this.state.architecture.layers.push(layer);
    this.rebuildLayer(layer);

    this.emit("layer:added", { layer });
  }

  /**
   * Remove a layer
   */
  public removeLayer(layerId: string): void {
    if (!this.config.interaction.enableArchitectureEdit) return;

    const index = this.state.architecture.layers.findIndex(
      (l) => l.id === layerId
    );
    if (index === -1) return;

    this.state.architecture.layers.splice(index, 1);

    // Remove neurons in this layer
    const neuronsToRemove: UniversalId[] = [];
    this.state.neurons.forEach((neuron, id) => {
      if (neuron.layerId === layerId) {
        neuronsToRemove.push(id);
      }
    });
    neuronsToRemove.forEach((id) => this.state.neurons.delete(id));

    // Remove weights connected to this layer
    const weightsToRemove: UniversalId[] = [];
    this.state.weights.forEach((weight, id) => {
      const fromNeuron = this.state.neurons.get(weight.fromNeuronId);
      const toNeuron = this.state.neurons.get(weight.toNeuronId);
      if (!fromNeuron || !toNeuron) {
        weightsToRemove.push(id);
      }
    });
    weightsToRemove.forEach((id) => this.state.weights.delete(id));

    this.emit("layer:removed", { layerId });
  }

  /**
   * Add a neuron to a layer
   */
  public addNeuron(layerId: string): NeuronState | null {
    if (!this.config.interaction.enableAddNeuron) return null;

    const layer = this.state.architecture.layers.find((l) => l.id === layerId);
    if (!layer) return null;

    const neuronId = this.generateId();
    const existingNeurons = Array.from(this.state.neurons.values()).filter(
      (n) => n.layerId === layerId
    );
    const index = existingNeurons.length;

    const neuron: NeuronState = {
      id: neuronId,
      layerId,
      index,
      position: this.calculateNeuronPosition(layerId, index),
      activation: 0,
      gradient: 0,
      selected: false,
      highlighted: false,
      bias: Math.random() * 0.2 - 0.1,
    };

    this.state.neurons.set(neuronId, neuron);
    layer.units++;

    // Connect to previous and next layers
    this.createConnectionsForNeuron(neuronId, layerId);

    this.emit("neuron:added", { neuron });

    return neuron;
  }

  /**
   * Remove a neuron
   */
  public removeNeuron(neuronId: UniversalId): void {
    if (!this.config.interaction.enableRemoveNeuron) return;

    const neuron = this.state.neurons.get(neuronId);
    if (!neuron) return;

    // Remove associated weights
    const weightsToRemove: UniversalId[] = [];
    this.state.weights.forEach((weight, id) => {
      if (weight.fromNeuronId === neuronId || weight.toNeuronId === neuronId) {
        weightsToRemove.push(id);
      }
    });
    weightsToRemove.forEach((id) => this.state.weights.delete(id));

    // Update layer units count
    const layer = this.state.architecture.layers.find(
      (l) => l.id === neuron.layerId
    );
    if (layer) {
      layer.units--;
    }

    this.state.neurons.delete(neuronId);
    this.emit("neuron:removed", { neuronId });
  }

  /**
   * Select a neuron
   */
  public selectNeuron(neuronId: UniversalId): void {
    if (!this.config.interaction.enableSelection) return;

    const neuron = this.state.neurons.get(neuronId);
    if (!neuron) return;

    neuron.selected = true;
    this.state.selection.selectedNeurons.push(neuronId);

    this.emit("neuron:selected", { neuron });
  }

  /**
   * Clear selection
   */
  public clearSelection(): void {
    this.state.selection.selectedNeurons.forEach((id) => {
      const neuron = this.state.neurons.get(id);
      if (neuron) neuron.selected = false;
    });
    this.state.selection.selectedNeurons = [];
    this.state.selection.selectedWeights = [];
  }

  /**
   * Move camera to focus on layer
   */
  public focusOnLayer(layerId: string): void {
    const neurons = Array.from(this.state.neurons.values()).filter(
      (n) => n.layerId === layerId
    );
    if (neurons.length === 0) return;

    const center = this.calculateCentroid(neurons.map((n) => n.position));
    this.state.camera.target = center;

    // Emit camera move
    this.emit("camera:moved", { target: center });
  }

  /**
   * Subscribe to events
   */
  public on(
    eventType: PlaygroundEventType,
    callback: (payload: unknown) => void
  ): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);

    return () => {
      this.listeners.get(eventType)?.delete(callback);
    };
  }

  /**
   * Enter VR mode
   */
  public async enterVR(): Promise<void> {
    if (!this.config.xr?.enableVR) return;

    // Would use WebXR API in real implementation
    this.emit("xr:entered", { mode: "vr" });
  }

  /**
   * Exit XR mode
   */
  public exitXR(): void {
    this.emit("xr:exited", {});
  }

  /**
   * Dispose and cleanup
   */
  public dispose(): void {
    this.stopRendering();
    this.pauseTraining();
    this.listeners.clear();
    this.state.neurons.clear();
    this.state.weights.clear();
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  private async initializeRenderer(): Promise<void> {
    // In a real implementation, this would initialize Three.js:
    // - Create scene, camera, renderer
    // - Set up lights
    // - Initialize post-processing (bloom, DOF)
    // - Set up controls (orbit, XR controllers)
  }

  private async buildNetworkVisualization(
    architecture: NetworkArchitecture
  ): Promise<void> {
    this.state.neurons.clear();
    this.state.weights.clear();

    // Create neurons for each layer
    architecture.layers.forEach((layer, layerIndex) => {
      for (let i = 0; i < layer.units; i++) {
        const neuronId = this.generateId();
        const position = this.calculateNeuronPosition(layer.id, i, {
          layerIndex,
          totalLayers: architecture.layers.length,
          layerSize: layer.units,
        });

        const neuron: NeuronState = {
          id: neuronId,
          layerId: layer.id,
          index: i,
          position,
          activation: 0,
          gradient: 0,
          selected: false,
          highlighted: false,
          bias: Math.random() * 0.2 - 0.1,
        };

        this.state.neurons.set(neuronId, neuron);
      }
    });

    // Create connections based on connection style
    if (architecture.connectionStyle === "full") {
      this.createFullConnections(architecture.layers);
    } else if (
      architecture.connectionStyle === "custom" &&
      architecture.customConnections
    ) {
      this.createCustomConnections(architecture.customConnections);
    }
  }

  private calculateNeuronPosition(
    layerId: string,
    index: number,
    context?: {
      layerIndex: number;
      totalLayers: number;
      layerSize: number;
    }
  ): Vector3 {
    // Calculate 3D position for neuron
    // Arrange layers along Z axis, neurons in circular pattern per layer
    const layerIndex =
      context?.layerIndex ||
      this.state.architecture.layers.findIndex((l) => l.id === layerId);
    const totalLayers =
      context?.totalLayers || this.state.architecture.layers.length;
    const layer = this.state.architecture.layers[layerIndex];
    const layerSize = context?.layerSize || layer?.units || 1;

    const layerSpacing = 15;
    const z = (layerIndex - (totalLayers - 1) / 2) * layerSpacing;

    // Arrange neurons in a circle for the layer
    const radius = Math.sqrt(layerSize) * 2;
    const angle = (index / layerSize) * Math.PI * 2;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;

    return { x, y, z };
  }

  private createFullConnections(layers: LayerDefinition[]): void {
    for (let i = 0; i < layers.length - 1; i++) {
      const fromLayer = layers[i];
      const toLayer = layers[i + 1];

      const fromNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === fromLayer.id
      );
      const toNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === toLayer.id
      );

      for (const fromNeuron of fromNeurons) {
        for (const toNeuron of toNeurons) {
          const weightId = this.generateId();
          const weight: WeightState = {
            id: weightId,
            fromNeuronId: fromNeuron.id,
            toNeuronId: toNeuron.id,
            value: Math.random() * 0.2 - 0.1, // Xavier-like init
            gradient: 0,
            particles: [],
          };
          this.state.weights.set(weightId, weight);
        }
      }
    }
  }

  private createCustomConnections(connections: Connection[]): void {
    for (const conn of connections) {
      const fromNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === conn.fromLayerId
      );
      const toNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === conn.toLayerId
      );

      for (const fromNeuron of fromNeurons) {
        for (const toNeuron of toNeurons) {
          const weightId = this.generateId();
          const weight: WeightState = {
            id: weightId,
            fromNeuronId: fromNeuron.id,
            toNeuronId: toNeuron.id,
            value: conn.weight ?? Math.random() * 0.2 - 0.1,
            gradient: 0,
            particles: [],
          };
          this.state.weights.set(weightId, weight);
        }
      }
    }
  }

  private createConnectionsForNeuron(
    neuronId: UniversalId,
    layerId: string
  ): void {
    const layerIndex = this.state.architecture.layers.findIndex(
      (l) => l.id === layerId
    );

    // Connect from previous layer
    if (layerIndex > 0) {
      const prevLayer = this.state.architecture.layers[layerIndex - 1];
      const prevNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === prevLayer.id
      );

      for (const prevNeuron of prevNeurons) {
        const weightId = this.generateId();
        const weight: WeightState = {
          id: weightId,
          fromNeuronId: prevNeuron.id,
          toNeuronId: neuronId,
          value: Math.random() * 0.2 - 0.1,
          gradient: 0,
          particles: [],
        };
        this.state.weights.set(weightId, weight);
      }
    }

    // Connect to next layer
    if (layerIndex < this.state.architecture.layers.length - 1) {
      const nextLayer = this.state.architecture.layers[layerIndex + 1];
      const nextNeurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === nextLayer.id
      );

      for (const nextNeuron of nextNeurons) {
        const weightId = this.generateId();
        const weight: WeightState = {
          id: weightId,
          fromNeuronId: neuronId,
          toNeuronId: nextNeuron.id,
          value: Math.random() * 0.2 - 0.1,
          gradient: 0,
          particles: [],
        };
        this.state.weights.set(weightId, weight);
      }
    }
  }

  private rebuildLayer(layer: LayerDefinition): void {
    // Add neurons for the new layer
    const layerIndex = this.state.architecture.layers.findIndex(
      (l) => l.id === layer.id
    );

    for (let i = 0; i < layer.units; i++) {
      const neuronId = this.generateId();
      const position = this.calculateNeuronPosition(layer.id, i, {
        layerIndex,
        totalLayers: this.state.architecture.layers.length,
        layerSize: layer.units,
      });

      const neuron: NeuronState = {
        id: neuronId,
        layerId: layer.id,
        index: i,
        position,
        activation: 0,
        gradient: 0,
        selected: false,
        highlighted: false,
        bias: Math.random() * 0.2 - 0.1,
      };

      this.state.neurons.set(neuronId, neuron);
      this.createConnectionsForNeuron(neuronId, layer.id);
    }
  }

  private setupEventHandlers(): void {
    if (!this.canvas) return;

    // Mouse events for interaction
    this.canvas.addEventListener("click", this.handleClick.bind(this));
    this.canvas.addEventListener("mousemove", this.handleMouseMove.bind(this));

    // Listen for foundry training events
    this.eventBridge.subscribe(
      "foundry:training:step",
      (event: CrossDomainEvent) => {
        if (this.state.training.isTraining) {
          this.handleTrainingStep(event.payload as TrainingStepResult);
        }
      }
    );
  }

  private handleClick(event: MouseEvent): void {
    // In real implementation, would raycast to find clicked neuron
    // For now, emit a placeholder event
    this.emit("neuron:selected", { x: event.clientX, y: event.clientY });
  }

  private handleMouseMove(event: MouseEvent): void {
    if (!this.config.interaction.highlightOnHover) return;
    // Would raycast and highlight hovered neuron
  }

  private renderLoop(): void {
    if (!this.state.isRendering) return;

    this.render();

    this.animationFrameId = requestAnimationFrame(() => this.renderLoop());
  }

  private render(): void {
    // In real implementation, would render using Three.js:
    // 1. Update neuron meshes (position, color based on activation)
    // 2. Update connection lines/particles
    // 3. Update particles for weight flow
    // 4. Render scene
    // 5. Apply post-processing
  }

  private async trainingLoop(): Promise<void> {
    if (!this.state.training.isTraining) return;

    const { batchSize } = this.config.training;
    const totalBatches = Math.ceil(this.trainingData.length / batchSize);

    for (let batch = 0; batch < totalBatches; batch++) {
      if (!this.state.training.isTraining) break;

      const batchStart = batch * batchSize;
      const batchData = this.trainingData.slice(
        batchStart,
        batchStart + batchSize
      );

      // Simulate training step
      const result = await this.runTrainingStep(batchData, batch);

      this.state.training.currentBatch = batch;
      this.state.training.currentLoss = result.loss;
      this.state.training.lossHistory.push(result.loss);
      this.state.training.accuracy = result.accuracy;

      // Update visualization
      if (batch % this.config.training.visualizationUpdateInterval === 0) {
        this.updateVisualizationFromTraining(result);
      }

      this.emit("training:step", result);

      // Yield to allow UI updates
      await new Promise((resolve) => setTimeout(resolve, 10));
    }

    this.state.training.currentEpoch++;

    if (this.state.training.isTraining) {
      this.trainingLoop(); // Continue with next epoch
    }
  }

  private async runTrainingStep(
    batchData: unknown[],
    batchIndex: number
  ): Promise<TrainingStepResult> {
    // Simulate forward pass and backward pass
    // In real implementation, would call actual model training

    const activations = new Map<string, number[]>();
    const gradients = new Map<string, number[]>();

    // Generate simulated activations
    this.state.architecture.layers.forEach((layer) => {
      const layerActivations: number[] = [];
      const layerGradients: number[] = [];

      for (let i = 0; i < layer.units; i++) {
        layerActivations.push(Math.random() * 2 - 1);
        layerGradients.push((Math.random() - 0.5) * 0.1);
      }

      activations.set(layer.id, layerActivations);
      gradients.set(layer.id, layerGradients);
    });

    const weightUpdates = new Map<UniversalId, number>();
    this.state.weights.forEach((weight, id) => {
      weightUpdates.set(id, (Math.random() - 0.5) * 0.01);
    });

    return {
      step: batchIndex,
      epoch: this.state.training.currentEpoch,
      loss:
        0.5 * Math.exp(-this.state.training.currentEpoch * 0.1) +
        Math.random() * 0.05,
      accuracy: Math.min(
        0.99,
        0.5 + this.state.training.currentEpoch * 0.05 + Math.random() * 0.02
      ),
      activations,
      gradients,
      weightUpdates,
    };
  }

  private handleTrainingStep(result: TrainingStepResult): void {
    // Handle external training step event
    this.updateVisualizationFromTraining(result);
  }

  private updateVisualizationFromTraining(result: TrainingStepResult): void {
    // Update neuron activations
    result.activations.forEach((activations, layerId) => {
      const neurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === layerId
      );
      neurons.forEach((neuron, index) => {
        if (index < activations.length) {
          neuron.activation = activations[index];
        }
      });
    });

    // Update neuron gradients
    result.gradients.forEach((grads, layerId) => {
      const neurons = Array.from(this.state.neurons.values()).filter(
        (n) => n.layerId === layerId
      );
      neurons.forEach((neuron, index) => {
        if (index < grads.length) {
          neuron.gradient = grads[index];
        }
      });
    });

    // Update weights
    result.weightUpdates.forEach((update, weightId) => {
      const weight = this.state.weights.get(weightId);
      if (weight) {
        weight.value += update;
        weight.gradient = update;
      }
    });
  }

  private calculateCentroid(positions: Vector3[]): Vector3 {
    const sum = positions.reduce(
      (acc, pos) => ({
        x: acc.x + pos.x,
        y: acc.y + pos.y,
        z: acc.z + pos.z,
      }),
      { x: 0, y: 0, z: 0 }
    );

    return {
      x: sum.x / positions.length,
      y: sum.y / positions.length,
      z: sum.z / positions.length,
    };
  }

  private emit(eventType: PlaygroundEventType, payload: unknown): void {
    const event: PlaygroundEvent = {
      type: eventType,
      payload,
      timestamp: Date.now(),
    };

    // Notify local listeners
    this.listeners.get(eventType)?.forEach((callback) => callback(payload));

    // Emit to event bridge
    this.eventBridge.emit({
      id: this.generateId(),
      type: `forge:playground:${eventType}`,
      source: "forge",
      timestamp: event.timestamp,
      payload: { playgroundId: this.state.id, ...payload },
    });
  }

  private generateId(): UniversalId {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

export default Neural3DPlayground;
