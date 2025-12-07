/**
 * 3D Neural Playground - Forge×Foundry Innovation #1
 *
 * Interactive 3D environment for designing and training neural architectures.
 * Users can visualize network topology in real-time as they modify layers,
 * connections, and hyperparameters - with immediate visual feedback on
 * training dynamics.
 *
 * @module NeuralPlayground
 * @category CrossDomain/Innovation
 */

import { EventEmitter } from "events";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * Neural network layer configuration
 */
export interface LayerConfig {
  id: string;
  type: LayerType;
  units: number;
  activation: ActivationType;
  position: Vector3D;
  connections: ConnectionConfig[];
  metadata: LayerMetadata;
}

/**
 * Supported layer types
 */
export type LayerType =
  | "input"
  | "dense"
  | "conv2d"
  | "conv3d"
  | "pooling"
  | "dropout"
  | "batchnorm"
  | "attention"
  | "transformer"
  | "lstm"
  | "gru"
  | "embedding"
  | "output";

/**
 * Activation function types
 */
export type ActivationType =
  | "relu"
  | "leaky_relu"
  | "sigmoid"
  | "tanh"
  | "softmax"
  | "gelu"
  | "swish"
  | "none";

/**
 * 3D position vector
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Connection between layers
 */
export interface ConnectionConfig {
  sourceLayerId: string;
  targetLayerId: string;
  weights?: Float32Array;
  gradients?: Float32Array;
  style: ConnectionStyle;
}

/**
 * Visual style for connections
 */
export interface ConnectionStyle {
  color: string;
  thickness: number;
  animated: boolean;
  flowDirection: "forward" | "backward" | "bidirectional";
}

/**
 * Layer metadata for visualization
 */
export interface LayerMetadata {
  parameterCount: number;
  computeTime: number;
  memoryUsage: number;
  gradientMagnitude: number;
  activationStats: ActivationStats;
}

/**
 * Activation statistics
 */
export interface ActivationStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  sparsity: number;
  histogram: number[];
}

/**
 * Network architecture definition
 */
export interface NetworkArchitecture {
  id: string;
  name: string;
  layers: LayerConfig[];
  connections: ConnectionConfig[];
  hyperparameters: HyperparameterConfig;
  trainingConfig: TrainingConfig;
}

/**
 * Hyperparameter configuration
 */
export interface HyperparameterConfig {
  learningRate: number;
  batchSize: number;
  optimizer: OptimizerType;
  regularization: RegularizationConfig;
  scheduler: LRSchedulerConfig;
}

/**
 * Optimizer types
 */
export type OptimizerType = "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad";

/**
 * Regularization configuration
 */
export interface RegularizationConfig {
  l1: number;
  l2: number;
  dropout: number;
}

/**
 * Learning rate scheduler configuration
 */
export interface LRSchedulerConfig {
  type: "constant" | "step" | "cosine" | "exponential" | "warmup";
  params: Record<string, number>;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  epochs: number;
  earlyStopping: EarlyStoppingConfig;
  checkpointing: CheckpointConfig;
}

/**
 * Early stopping configuration
 */
export interface EarlyStoppingConfig {
  enabled: boolean;
  patience: number;
  metric: string;
  mode: "min" | "max";
}

/**
 * Checkpoint configuration
 */
export interface CheckpointConfig {
  enabled: boolean;
  interval: number;
  maxToKeep: number;
}

/**
 * Training state
 */
export interface TrainingState {
  status: TrainingStatus;
  currentEpoch: number;
  currentBatch: number;
  metrics: TrainingMetrics;
  layerActivations: Map<string, Float32Array>;
  gradientFlow: Map<string, GradientInfo>;
}

/**
 * Training status
 */
export type TrainingStatus =
  | "idle"
  | "initializing"
  | "training"
  | "validating"
  | "paused"
  | "completed"
  | "error";

/**
 * Training metrics
 */
export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  learningRate: number;
  gradientNorm: number;
  throughput: number;
  customMetrics: Map<string, number>;
}

/**
 * Gradient information
 */
export interface GradientInfo {
  magnitude: number;
  direction: Vector3D;
  vanishing: boolean;
  exploding: boolean;
}

/**
 * Playground scene configuration
 */
export interface PlaygroundSceneConfig {
  cameraPosition: Vector3D;
  cameraTarget: Vector3D;
  lighting: LightingConfig;
  background: BackgroundConfig;
  gridEnabled: boolean;
}

/**
 * Lighting configuration
 */
export interface LightingConfig {
  ambient: number;
  directional: DirectionalLight[];
  bloom: boolean;
  shadows: boolean;
}

/**
 * Directional light
 */
export interface DirectionalLight {
  position: Vector3D;
  intensity: number;
  color: string;
}

/**
 * Background configuration
 */
export interface BackgroundConfig {
  type: "solid" | "gradient" | "cubemap";
  colors: string[];
}

/**
 * Interaction event
 */
export interface InteractionEvent {
  type: InteractionType;
  target: string;
  position: Vector3D;
  data: unknown;
}

/**
 * Interaction types
 */
export type InteractionType =
  | "layer_select"
  | "layer_move"
  | "layer_resize"
  | "connection_create"
  | "connection_remove"
  | "hyperparameter_change"
  | "view_rotate"
  | "view_zoom";

/**
 * Visualization mode
 */
export type VisualizationMode =
  | "topology"
  | "activations"
  | "gradients"
  | "weights"
  | "training";

// ============================================================================
// Neural Playground Implementation
// ============================================================================

/**
 * 3D Neural Playground for interactive network design and training visualization
 */
export class NeuralPlayground extends EventEmitter {
  private architecture: NetworkArchitecture;
  private trainingState: TrainingState;
  private sceneConfig: PlaygroundSceneConfig;
  private visualizationMode: VisualizationMode;
  private layerRenderers: Map<string, LayerRenderer>;
  private connectionRenderers: Map<string, ConnectionRenderer>;
  private animationFrame: number | null;
  private isActive: boolean;
  private history: NetworkArchitecture[];
  private historyIndex: number;

  constructor() {
    super();

    this.architecture = this.createDefaultArchitecture();
    this.trainingState = this.createInitialTrainingState();
    this.sceneConfig = this.createDefaultSceneConfig();
    this.visualizationMode = "topology";
    this.layerRenderers = new Map();
    this.connectionRenderers = new Map();
    this.animationFrame = null;
    this.isActive = false;
    this.history = [];
    this.historyIndex = -1;

    this.initializeRenderers();
  }

  /**
   * Create default network architecture
   */
  private createDefaultArchitecture(): NetworkArchitecture {
    return {
      id: this.generateId(),
      name: "New Network",
      layers: [],
      connections: [],
      hyperparameters: {
        learningRate: 0.001,
        batchSize: 32,
        optimizer: "adam",
        regularization: { l1: 0, l2: 0.0001, dropout: 0.2 },
        scheduler: { type: "cosine", params: { minLr: 0.0001 } },
      },
      trainingConfig: {
        epochs: 100,
        earlyStopping: {
          enabled: true,
          patience: 10,
          metric: "val_loss",
          mode: "min",
        },
        checkpointing: { enabled: true, interval: 5, maxToKeep: 3 },
      },
    };
  }

  /**
   * Create initial training state
   */
  private createInitialTrainingState(): TrainingState {
    return {
      status: "idle",
      currentEpoch: 0,
      currentBatch: 0,
      metrics: {
        loss: 0,
        accuracy: 0,
        learningRate: 0.001,
        gradientNorm: 0,
        throughput: 0,
        customMetrics: new Map(),
      },
      layerActivations: new Map(),
      gradientFlow: new Map(),
    };
  }

  /**
   * Create default scene configuration
   */
  private createDefaultSceneConfig(): PlaygroundSceneConfig {
    return {
      cameraPosition: { x: 0, y: 5, z: 15 },
      cameraTarget: { x: 0, y: 0, z: 0 },
      lighting: {
        ambient: 0.4,
        directional: [
          {
            position: { x: 10, y: 10, z: 10 },
            intensity: 0.8,
            color: "#ffffff",
          },
          {
            position: { x: -5, y: 5, z: -5 },
            intensity: 0.4,
            color: "#88aaff",
          },
        ],
        bloom: true,
        shadows: true,
      },
      background: {
        type: "gradient",
        colors: ["#1a1a2e", "#16213e"],
      },
      gridEnabled: true,
    };
  }

  /**
   * Initialize layer and connection renderers
   */
  private initializeRenderers(): void {
    // Will be initialized when layers are added
  }

  // ============================================================================
  // Layer Management
  // ============================================================================

  /**
   * Add a new layer to the network
   */
  addLayer(config: Partial<LayerConfig>): string {
    const layer = this.createLayer(config);

    this.saveHistory();
    this.architecture.layers.push(layer);

    this.createLayerRenderer(layer);
    this.autoLayoutLayers();

    this.emit("layer:added", layer);
    this.emit("architecture:changed", this.architecture);

    return layer.id;
  }

  /**
   * Create a layer with defaults
   */
  private createLayer(config: Partial<LayerConfig>): LayerConfig {
    const layerIndex = this.architecture.layers.length;

    return {
      id: config.id || this.generateId(),
      type: config.type || "dense",
      units: config.units || 64,
      activation: config.activation || "relu",
      position: config.position || this.calculateDefaultPosition(layerIndex),
      connections: config.connections || [],
      metadata: {
        parameterCount: 0,
        computeTime: 0,
        memoryUsage: 0,
        gradientMagnitude: 0,
        activationStats: {
          mean: 0,
          std: 0,
          min: 0,
          max: 0,
          sparsity: 0,
          histogram: [],
        },
      },
    };
  }

  /**
   * Calculate default position for new layer
   */
  private calculateDefaultPosition(index: number): Vector3D {
    const spacing = 3;
    const totalLayers = this.architecture.layers.length;
    const offset = (totalLayers * spacing) / 2;

    return {
      x: index * spacing - offset,
      y: 0,
      z: 0,
    };
  }

  /**
   * Remove a layer
   */
  removeLayer(layerId: string): void {
    this.saveHistory();

    const index = this.architecture.layers.findIndex((l) => l.id === layerId);
    if (index === -1) return;

    // Remove associated connections
    this.architecture.connections = this.architecture.connections.filter(
      (c) => c.sourceLayerId !== layerId && c.targetLayerId !== layerId
    );

    this.architecture.layers.splice(index, 1);
    this.destroyLayerRenderer(layerId);

    this.emit("layer:removed", layerId);
    this.emit("architecture:changed", this.architecture);
  }

  /**
   * Update layer configuration
   */
  updateLayer(layerId: string, updates: Partial<LayerConfig>): void {
    this.saveHistory();

    const layer = this.architecture.layers.find((l) => l.id === layerId);
    if (!layer) return;

    Object.assign(layer, updates);
    this.updateLayerRenderer(layer);

    this.emit("layer:updated", layer);
    this.emit("architecture:changed", this.architecture);
  }

  /**
   * Get layer by ID
   */
  getLayer(layerId: string): LayerConfig | undefined {
    return this.architecture.layers.find((l) => l.id === layerId);
  }

  /**
   * Get all layers
   */
  getLayers(): LayerConfig[] {
    return [...this.architecture.layers];
  }

  // ============================================================================
  // Connection Management
  // ============================================================================

  /**
   * Create connection between layers
   */
  createConnection(sourceId: string, targetId: string): void {
    // Validate connection
    if (!this.validateConnection(sourceId, targetId)) {
      this.emit("error", {
        type: "invalid_connection",
        message: "Invalid connection",
      });
      return;
    }

    this.saveHistory();

    const connection: ConnectionConfig = {
      sourceLayerId: sourceId,
      targetLayerId: targetId,
      style: {
        color: "#4fc3f7",
        thickness: 2,
        animated: true,
        flowDirection: "forward",
      },
    };

    this.architecture.connections.push(connection);
    this.createConnectionRenderer(connection);

    this.emit("connection:created", connection);
    this.emit("architecture:changed", this.architecture);
  }

  /**
   * Validate connection between layers
   */
  private validateConnection(sourceId: string, targetId: string): boolean {
    // Can't connect to self
    if (sourceId === targetId) return false;

    // Check if layers exist
    const source = this.getLayer(sourceId);
    const target = this.getLayer(targetId);
    if (!source || !target) return false;

    // Check for existing connection
    const exists = this.architecture.connections.some(
      (c) => c.sourceLayerId === sourceId && c.targetLayerId === targetId
    );
    if (exists) return false;

    // Check for cycles (simple check - could be more sophisticated)
    return !this.wouldCreateCycle(sourceId, targetId);
  }

  /**
   * Check if connection would create a cycle
   */
  private wouldCreateCycle(sourceId: string, targetId: string): boolean {
    const visited = new Set<string>();
    const stack: string[] = [sourceId];

    while (stack.length > 0) {
      const current = stack.pop()!;

      if (current === targetId) return false; // Target is "upstream", no cycle
      if (visited.has(current)) continue;

      visited.add(current);

      // Find connections from current
      const outgoing = this.architecture.connections
        .filter((c) => c.sourceLayerId === current)
        .map((c) => c.targetLayerId);

      // Check if any lead back to source
      if (outgoing.includes(sourceId)) return true;

      stack.push(...outgoing);
    }

    return false;
  }

  /**
   * Remove connection
   */
  removeConnection(sourceId: string, targetId: string): void {
    this.saveHistory();

    const index = this.architecture.connections.findIndex(
      (c) => c.sourceLayerId === sourceId && c.targetLayerId === targetId
    );

    if (index === -1) return;

    const connection = this.architecture.connections[index];
    this.architecture.connections.splice(index, 1);
    this.destroyConnectionRenderer(sourceId, targetId);

    this.emit("connection:removed", connection);
    this.emit("architecture:changed", this.architecture);
  }

  // ============================================================================
  // Visualization
  // ============================================================================

  /**
   * Set visualization mode
   */
  setVisualizationMode(mode: VisualizationMode): void {
    this.visualizationMode = mode;
    this.updateAllRenderers();
    this.emit("visualization:changed", mode);
  }

  /**
   * Get current visualization mode
   */
  getVisualizationMode(): VisualizationMode {
    return this.visualizationMode;
  }

  /**
   * Update all renderers based on current mode
   */
  private updateAllRenderers(): void {
    for (const [id, renderer] of this.layerRenderers) {
      renderer.setMode(this.visualizationMode);
    }

    for (const [id, renderer] of this.connectionRenderers) {
      renderer.setMode(this.visualizationMode);
    }
  }

  /**
   * Create renderer for layer
   */
  private createLayerRenderer(layer: LayerConfig): void {
    const renderer = new LayerRenderer(layer);
    renderer.setMode(this.visualizationMode);
    this.layerRenderers.set(layer.id, renderer);
  }

  /**
   * Update layer renderer
   */
  private updateLayerRenderer(layer: LayerConfig): void {
    const renderer = this.layerRenderers.get(layer.id);
    if (renderer) {
      renderer.update(layer);
    }
  }

  /**
   * Destroy layer renderer
   */
  private destroyLayerRenderer(layerId: string): void {
    const renderer = this.layerRenderers.get(layerId);
    if (renderer) {
      renderer.dispose();
      this.layerRenderers.delete(layerId);
    }
  }

  /**
   * Create connection renderer
   */
  private createConnectionRenderer(connection: ConnectionConfig): void {
    const key = `${connection.sourceLayerId}->${connection.targetLayerId}`;
    const renderer = new ConnectionRenderer(connection);
    renderer.setMode(this.visualizationMode);
    this.connectionRenderers.set(key, renderer);
  }

  /**
   * Destroy connection renderer
   */
  private destroyConnectionRenderer(sourceId: string, targetId: string): void {
    const key = `${sourceId}->${targetId}`;
    const renderer = this.connectionRenderers.get(key);
    if (renderer) {
      renderer.dispose();
      this.connectionRenderers.delete(key);
    }
  }

  // ============================================================================
  // Training Visualization
  // ============================================================================

  /**
   * Start training visualization
   */
  startTraining(): void {
    if (this.trainingState.status === "training") return;

    this.trainingState.status = "training";
    this.setVisualizationMode("training");
    this.startAnimationLoop();

    this.emit("training:started");
  }

  /**
   * Pause training
   */
  pauseTraining(): void {
    if (this.trainingState.status !== "training") return;

    this.trainingState.status = "paused";
    this.emit("training:paused");
  }

  /**
   * Resume training
   */
  resumeTraining(): void {
    if (this.trainingState.status !== "paused") return;

    this.trainingState.status = "training";
    this.emit("training:resumed");
  }

  /**
   * Stop training
   */
  stopTraining(): void {
    this.trainingState.status = "idle";
    this.stopAnimationLoop();

    this.emit("training:stopped");
  }

  /**
   * Update training metrics
   */
  updateTrainingMetrics(metrics: Partial<TrainingMetrics>): void {
    Object.assign(this.trainingState.metrics, metrics);
    this.emit("training:metrics", this.trainingState.metrics);
  }

  /**
   * Update layer activations during training
   */
  updateActivations(layerId: string, activations: Float32Array): void {
    this.trainingState.layerActivations.set(layerId, activations);

    // Update activation stats
    const layer = this.getLayer(layerId);
    if (layer) {
      layer.metadata.activationStats = this.computeActivationStats(activations);
    }

    this.emit("training:activations", { layerId, activations });
  }

  /**
   * Compute activation statistics
   */
  private computeActivationStats(activations: Float32Array): ActivationStats {
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    let zeroCount = 0;

    for (const val of activations) {
      sum += val;
      sumSq += val * val;
      min = Math.min(min, val);
      max = Math.max(max, val);
      if (val === 0) zeroCount++;
    }

    const n = activations.length;
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const std = Math.sqrt(Math.max(0, variance));
    const sparsity = zeroCount / n;

    // Create histogram
    const numBins = 50;
    const histogram = new Array(numBins).fill(0);
    const binWidth = (max - min) / numBins || 1;

    for (const val of activations) {
      const bin = Math.min(numBins - 1, Math.floor((val - min) / binWidth));
      histogram[bin]++;
    }

    return { mean, std, min, max, sparsity, histogram };
  }

  /**
   * Update gradient flow visualization
   */
  updateGradients(layerId: string, gradients: Float32Array): void {
    const magnitude = this.computeGradientMagnitude(gradients);
    const vanishing = magnitude < 1e-7;
    const exploding = magnitude > 1e3;

    const gradientInfo: GradientInfo = {
      magnitude,
      direction: { x: 0, y: magnitude > 0 ? 1 : -1, z: 0 },
      vanishing,
      exploding,
    };

    this.trainingState.gradientFlow.set(layerId, gradientInfo);

    // Update layer metadata
    const layer = this.getLayer(layerId);
    if (layer) {
      layer.metadata.gradientMagnitude = magnitude;
    }

    this.emit("training:gradients", { layerId, gradientInfo });
  }

  /**
   * Compute gradient magnitude
   */
  private computeGradientMagnitude(gradients: Float32Array): number {
    let sumSq = 0;
    for (const val of gradients) {
      sumSq += val * val;
    }
    return Math.sqrt(sumSq);
  }

  // ============================================================================
  // Animation Loop
  // ============================================================================

  /**
   * Start animation loop
   */
  private startAnimationLoop(): void {
    if (this.animationFrame !== null) return;

    this.isActive = true;
    const animate = () => {
      if (!this.isActive) return;

      this.updateAnimation();
      this.animationFrame = requestAnimationFrame(animate);
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Stop animation loop
   */
  private stopAnimationLoop(): void {
    this.isActive = false;
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Update animation frame
   */
  private updateAnimation(): void {
    // Animate layer renderers
    for (const renderer of this.layerRenderers.values()) {
      renderer.animate(this.trainingState);
    }

    // Animate connection renderers
    for (const renderer of this.connectionRenderers.values()) {
      renderer.animate(this.trainingState);
    }

    this.emit("frame:updated");
  }

  // ============================================================================
  // Layout
  // ============================================================================

  /**
   * Auto-layout layers
   */
  autoLayoutLayers(): void {
    const layout = this.computeLayerLayout();

    for (const [layerId, position] of Object.entries(layout)) {
      const layer = this.getLayer(layerId);
      if (layer) {
        layer.position = position;
        this.updateLayerRenderer(layer);
      }
    }

    this.emit("layout:updated", layout);
  }

  /**
   * Compute optimal layer layout
   */
  private computeLayerLayout(): Record<string, Vector3D> {
    const layout: Record<string, Vector3D> = {};
    const layers = this.architecture.layers;

    // Group layers by depth (topological sort)
    const depths = this.computeLayerDepths();
    const maxDepth = Math.max(0, ...depths.values());

    // Group layers by depth
    const layersByDepth: Map<number, LayerConfig[]> = new Map();
    for (const layer of layers) {
      const depth = depths.get(layer.id) || 0;
      if (!layersByDepth.has(depth)) {
        layersByDepth.set(depth, []);
      }
      layersByDepth.get(depth)!.push(layer);
    }

    // Position layers
    const xSpacing = 4;
    const ySpacing = 3;

    for (let depth = 0; depth <= maxDepth; depth++) {
      const layersAtDepth = layersByDepth.get(depth) || [];
      const yOffset = ((layersAtDepth.length - 1) * ySpacing) / 2;

      layersAtDepth.forEach((layer, index) => {
        layout[layer.id] = {
          x: depth * xSpacing - (maxDepth * xSpacing) / 2,
          y: index * ySpacing - yOffset,
          z: 0,
        };
      });
    }

    return layout;
  }

  /**
   * Compute layer depths using topological sort
   */
  private computeLayerDepths(): Map<string, number> {
    const depths = new Map<string, number>();
    const visited = new Set<string>();

    // Find input layers (no incoming connections)
    const inputLayers = this.architecture.layers.filter((layer) => {
      return !this.architecture.connections.some(
        (c) => c.targetLayerId === layer.id
      );
    });

    // BFS from input layers
    const queue: Array<{ id: string; depth: number }> = inputLayers.map(
      (l) => ({
        id: l.id,
        depth: 0,
      })
    );

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;

      if (visited.has(id)) {
        depths.set(id, Math.max(depths.get(id) || 0, depth));
        continue;
      }

      visited.add(id);
      depths.set(id, depth);

      // Find outgoing connections
      const outgoing = this.architecture.connections
        .filter((c) => c.sourceLayerId === id)
        .map((c) => c.targetLayerId);

      for (const targetId of outgoing) {
        queue.push({ id: targetId, depth: depth + 1 });
      }
    }

    return depths;
  }

  // ============================================================================
  // Templates
  // ============================================================================

  /**
   * Load network template
   */
  loadTemplate(template: NetworkTemplate): void {
    this.saveHistory();

    this.architecture = this.templateToArchitecture(template);
    this.rebuildRenderers();
    this.autoLayoutLayers();

    this.emit("template:loaded", template.name);
    this.emit("architecture:changed", this.architecture);
  }

  /**
   * Convert template to architecture
   */
  private templateToArchitecture(
    template: NetworkTemplate
  ): NetworkArchitecture {
    const architecture: NetworkArchitecture = {
      id: this.generateId(),
      name: template.name,
      layers: [],
      connections: [],
      hyperparameters: template.defaultHyperparameters,
      trainingConfig: {
        epochs: 100,
        earlyStopping: {
          enabled: true,
          patience: 10,
          metric: "val_loss",
          mode: "min",
        },
        checkpointing: { enabled: true, interval: 5, maxToKeep: 3 },
      },
    };

    // Create layers
    for (const layerDef of template.layers) {
      const layer = this.createLayer({
        type: layerDef.type,
        units: layerDef.units,
        activation: layerDef.activation,
      });
      architecture.layers.push(layer);
    }

    // Create sequential connections
    for (let i = 0; i < architecture.layers.length - 1; i++) {
      architecture.connections.push({
        sourceLayerId: architecture.layers[i].id,
        targetLayerId: architecture.layers[i + 1].id,
        style: {
          color: "#4fc3f7",
          thickness: 2,
          animated: true,
          flowDirection: "forward",
        },
      });
    }

    return architecture;
  }

  /**
   * Rebuild all renderers
   */
  private rebuildRenderers(): void {
    // Clear existing
    for (const renderer of this.layerRenderers.values()) {
      renderer.dispose();
    }
    for (const renderer of this.connectionRenderers.values()) {
      renderer.dispose();
    }

    this.layerRenderers.clear();
    this.connectionRenderers.clear();

    // Create new
    for (const layer of this.architecture.layers) {
      this.createLayerRenderer(layer);
    }

    for (const connection of this.architecture.connections) {
      this.createConnectionRenderer(connection);
    }
  }

  // ============================================================================
  // History (Undo/Redo)
  // ============================================================================

  /**
   * Save current state to history
   */
  private saveHistory(): void {
    // Remove any future history if we're not at the end
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }

    // Deep copy architecture
    const snapshot = JSON.parse(JSON.stringify(this.architecture));
    this.history.push(snapshot);
    this.historyIndex = this.history.length - 1;

    // Limit history size
    if (this.history.length > 50) {
      this.history.shift();
      this.historyIndex--;
    }
  }

  /**
   * Undo last change
   */
  undo(): boolean {
    if (this.historyIndex <= 0) return false;

    this.historyIndex--;
    this.architecture = JSON.parse(
      JSON.stringify(this.history[this.historyIndex])
    );
    this.rebuildRenderers();
    this.autoLayoutLayers();

    this.emit("history:undo");
    this.emit("architecture:changed", this.architecture);

    return true;
  }

  /**
   * Redo undone change
   */
  redo(): boolean {
    if (this.historyIndex >= this.history.length - 1) return false;

    this.historyIndex++;
    this.architecture = JSON.parse(
      JSON.stringify(this.history[this.historyIndex])
    );
    this.rebuildRenderers();
    this.autoLayoutLayers();

    this.emit("history:redo");
    this.emit("architecture:changed", this.architecture);

    return true;
  }

  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    return this.historyIndex > 0;
  }

  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.historyIndex < this.history.length - 1;
  }

  // ============================================================================
  // Serialization
  // ============================================================================

  /**
   * Export architecture to JSON
   */
  exportArchitecture(): string {
    return JSON.stringify(this.architecture, null, 2);
  }

  /**
   * Import architecture from JSON
   */
  importArchitecture(json: string): void {
    try {
      const architecture = JSON.parse(json) as NetworkArchitecture;

      this.saveHistory();
      this.architecture = architecture;
      this.rebuildRenderers();
      this.autoLayoutLayers();

      this.emit("architecture:imported");
      this.emit("architecture:changed", this.architecture);
    } catch (error) {
      this.emit("error", {
        type: "import_error",
        message: "Invalid architecture JSON",
      });
    }
  }

  /**
   * Export to framework code
   */
  exportToCode(framework: "pytorch" | "tensorflow" | "jax"): string {
    const exporter = new CodeExporter(this.architecture);
    return exporter.export(framework);
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Get current architecture
   */
  getArchitecture(): NetworkArchitecture {
    return JSON.parse(JSON.stringify(this.architecture));
  }

  /**
   * Get training state
   */
  getTrainingState(): TrainingState {
    return { ...this.trainingState };
  }

  /**
   * Dispose playground
   */
  dispose(): void {
    this.stopAnimationLoop();

    for (const renderer of this.layerRenderers.values()) {
      renderer.dispose();
    }
    for (const renderer of this.connectionRenderers.values()) {
      renderer.dispose();
    }

    this.layerRenderers.clear();
    this.connectionRenderers.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Supporting Classes
// ============================================================================

/**
 * Network template definition
 */
export interface NetworkTemplate {
  name: string;
  description: string;
  layers: Array<{
    type: LayerType;
    units: number;
    activation: ActivationType;
  }>;
  defaultHyperparameters: HyperparameterConfig;
}

/**
 * Layer renderer (stub - would integrate with 3D engine)
 */
class LayerRenderer {
  private layer: LayerConfig;
  private mode: VisualizationMode;

  constructor(layer: LayerConfig) {
    this.layer = layer;
    this.mode = "topology";
  }

  setMode(mode: VisualizationMode): void {
    this.mode = mode;
  }

  update(layer: LayerConfig): void {
    this.layer = layer;
  }

  animate(state: TrainingState): void {
    // Animation logic based on mode and training state
  }

  dispose(): void {
    // Cleanup
  }
}

/**
 * Connection renderer (stub - would integrate with 3D engine)
 */
class ConnectionRenderer {
  private connection: ConnectionConfig;
  private mode: VisualizationMode;

  constructor(connection: ConnectionConfig) {
    this.connection = connection;
    this.mode = "topology";
  }

  setMode(mode: VisualizationMode): void {
    this.mode = mode;
  }

  animate(state: TrainingState): void {
    // Animation logic - show gradient flow, activation flow, etc.
  }

  dispose(): void {
    // Cleanup
  }
}

/**
 * Code exporter for different frameworks
 */
class CodeExporter {
  private architecture: NetworkArchitecture;

  constructor(architecture: NetworkArchitecture) {
    this.architecture = architecture;
  }

  export(framework: "pytorch" | "tensorflow" | "jax"): string {
    switch (framework) {
      case "pytorch":
        return this.exportPyTorch();
      case "tensorflow":
        return this.exportTensorFlow();
      case "jax":
        return this.exportJAX();
    }
  }

  private exportPyTorch(): string {
    const lines = [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      `class ${this.toPascalCase(this.architecture.name)}(nn.Module):`,
      "    def __init__(self):",
      "        super().__init__()",
    ];

    // Define layers
    for (const layer of this.architecture.layers) {
      const layerDef = this.getPyTorchLayerDef(layer);
      if (layerDef) {
        lines.push(`        self.${layer.id} = ${layerDef}`);
      }
    }

    lines.push("");
    lines.push("    def forward(self, x):");

    // Forward pass
    for (let i = 0; i < this.architecture.layers.length; i++) {
      const layer = this.architecture.layers[i];
      const activation = this.getPyTorchActivation(layer.activation);

      if (layer.type !== "input") {
        if (activation) {
          lines.push(`        x = ${activation}(self.${layer.id}(x))`);
        } else {
          lines.push(`        x = self.${layer.id}(x)`);
        }
      }
    }

    lines.push("        return x");

    return lines.join("\n");
  }

  private exportTensorFlow(): string {
    const lines = [
      "import tensorflow as tf",
      "from tensorflow import keras",
      "from tensorflow.keras import layers",
      "",
      "def create_model():",
      "    model = keras.Sequential([",
    ];

    for (const layer of this.architecture.layers) {
      const layerDef = this.getTensorFlowLayerDef(layer);
      if (layerDef) {
        lines.push(`        ${layerDef},`);
      }
    }

    lines.push("    ])");
    lines.push("    return model");

    return lines.join("\n");
  }

  private exportJAX(): string {
    return "# JAX/Flax export not yet implemented";
  }

  private getPyTorchLayerDef(layer: LayerConfig): string | null {
    switch (layer.type) {
      case "dense":
        return `nn.Linear(in_features, ${layer.units})`;
      case "conv2d":
        return `nn.Conv2d(in_channels, ${layer.units}, kernel_size=3, padding=1)`;
      case "dropout":
        return `nn.Dropout(p=0.2)`;
      case "batchnorm":
        return `nn.BatchNorm1d(${layer.units})`;
      default:
        return null;
    }
  }

  private getPyTorchActivation(activation: ActivationType): string | null {
    switch (activation) {
      case "relu":
        return "F.relu";
      case "sigmoid":
        return "torch.sigmoid";
      case "tanh":
        return "torch.tanh";
      case "softmax":
        return "F.softmax";
      case "gelu":
        return "F.gelu";
      default:
        return null;
    }
  }

  private getTensorFlowLayerDef(layer: LayerConfig): string | null {
    switch (layer.type) {
      case "input":
        return `layers.InputLayer(input_shape=(None,))`;
      case "dense":
        return `layers.Dense(${layer.units}, activation='${layer.activation}')`;
      case "dropout":
        return `layers.Dropout(0.2)`;
      default:
        return null;
    }
  }

  private toPascalCase(str: string): string {
    return str
      .replace(/[^a-zA-Z0-9]+/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join("");
  }
}

// ============================================================================
// Predefined Templates
// ============================================================================

/**
 * Predefined network templates
 */
export const NetworkTemplates: Record<string, NetworkTemplate> = {
  mlp: {
    name: "MLP",
    description: "Multi-Layer Perceptron",
    layers: [
      { type: "input", units: 784, activation: "none" },
      { type: "dense", units: 256, activation: "relu" },
      { type: "dense", units: 128, activation: "relu" },
      { type: "dense", units: 10, activation: "softmax" },
    ],
    defaultHyperparameters: {
      learningRate: 0.001,
      batchSize: 32,
      optimizer: "adam",
      regularization: { l1: 0, l2: 0.0001, dropout: 0.2 },
      scheduler: { type: "cosine", params: { minLr: 0.0001 } },
    },
  },

  cnn: {
    name: "CNN",
    description: "Convolutional Neural Network",
    layers: [
      { type: "input", units: 0, activation: "none" },
      { type: "conv2d", units: 32, activation: "relu" },
      { type: "pooling", units: 0, activation: "none" },
      { type: "conv2d", units: 64, activation: "relu" },
      { type: "pooling", units: 0, activation: "none" },
      { type: "dense", units: 128, activation: "relu" },
      { type: "dense", units: 10, activation: "softmax" },
    ],
    defaultHyperparameters: {
      learningRate: 0.001,
      batchSize: 64,
      optimizer: "adam",
      regularization: { l1: 0, l2: 0.0001, dropout: 0.25 },
      scheduler: { type: "step", params: { stepSize: 10, gamma: 0.1 } },
    },
  },

  transformer: {
    name: "Transformer",
    description: "Transformer Encoder",
    layers: [
      { type: "input", units: 512, activation: "none" },
      { type: "embedding", units: 512, activation: "none" },
      { type: "attention", units: 512, activation: "none" },
      { type: "dense", units: 2048, activation: "gelu" },
      { type: "dense", units: 512, activation: "none" },
      { type: "output", units: 10, activation: "softmax" },
    ],
    defaultHyperparameters: {
      learningRate: 0.0001,
      batchSize: 16,
      optimizer: "adamw",
      regularization: { l1: 0, l2: 0.01, dropout: 0.1 },
      scheduler: { type: "warmup", params: { warmupSteps: 1000 } },
    },
  },
};

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a new Neural Playground instance
 */
export function createNeuralPlayground(): NeuralPlayground {
  return new NeuralPlayground();
}
