/**
 * @file 3D Neural Playground - Forge×Foundry Innovation
 * @description Interactive 3D environment for experimenting with neural architectures.
 * Enables visual construction, modification, and real-time training of neural networks.
 *
 * Innovation: Combines 3D visualization with ML model building for intuitive
 * neural architecture design and experimentation.
 *
 * @module cross-domain/innovations/forge-foundry
 */

import { EventEmitter } from "events";
import * as tf from "@tensorflow/tfjs";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * Neural layer types supported in the playground
 */
export type NeuralLayerType =
  | "dense"
  | "conv2d"
  | "maxPool2d"
  | "avgPool2d"
  | "flatten"
  | "dropout"
  | "batchNorm"
  | "lstm"
  | "gru"
  | "attention"
  | "embedding"
  | "reshape"
  | "concatenate"
  | "add"
  | "multiply";

/**
 * Activation functions
 */
export type ActivationType =
  | "relu"
  | "leakyRelu"
  | "elu"
  | "selu"
  | "sigmoid"
  | "tanh"
  | "softmax"
  | "swish"
  | "gelu"
  | "linear";

/**
 * 3D position in playground space
 */
export interface Position3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Visual representation of a neural layer in 3D space
 */
export interface NeuralBlock3D {
  id: string;
  type: NeuralLayerType;
  position: Position3D;
  rotation: { x: number; y: number; z: number };
  scale: { width: number; height: number; depth: number };
  config: LayerConfig;
  connections: {
    inputs: string[];
    outputs: string[];
  };
  visualStyle: BlockVisualStyle;
  metadata: {
    paramCount: number;
    outputShape: number[];
    computeTime?: number;
    memoryUsage?: number;
  };
}

/**
 * Layer configuration options
 */
export interface LayerConfig {
  // Dense layer
  units?: number;
  activation?: ActivationType;
  useBias?: boolean;

  // Conv layer
  filters?: number;
  kernelSize?: number | [number, number];
  strides?: number | [number, number];
  padding?: "valid" | "same";

  // Pooling
  poolSize?: number | [number, number];

  // Dropout
  rate?: number;

  // Recurrent
  returnSequences?: boolean;

  // Attention
  numHeads?: number;
  keyDim?: number;

  // Embedding
  inputDim?: number;
  outputDim?: number;

  // General
  name?: string;
  trainable?: boolean;
}

/**
 * Visual styling for blocks
 */
export interface BlockVisualStyle {
  color: string;
  opacity: number;
  texture?: string;
  glow?: {
    enabled: boolean;
    color: string;
    intensity: number;
  };
  animation?: {
    type: "pulse" | "rotate" | "flow";
    speed: number;
  };
}

/**
 * Connection between neural blocks
 */
export interface NeuralConnection3D {
  id: string;
  sourceBlock: string;
  targetBlock: string;
  sourcePort: "output";
  targetPort: "input";
  pathPoints: Position3D[];
  style: {
    color: string;
    width: number;
    animated: boolean;
    flowDirection: "forward" | "backward" | "bidirectional";
  };
  dataFlow?: {
    tensorShape: number[];
    dataType: string;
    gradientFlow: boolean;
  };
}

/**
 * Playground scene containing neural architecture
 */
export interface PlaygroundScene {
  id: string;
  name: string;
  blocks: Map<string, NeuralBlock3D>;
  connections: Map<string, NeuralConnection3D>;
  camera: CameraState;
  environment: EnvironmentSettings;
  metadata: {
    created: Date;
    modified: Date;
    author?: string;
    description?: string;
    tags?: string[];
  };
}

/**
 * Camera state for 3D view
 */
export interface CameraState {
  position: Position3D;
  target: Position3D;
  fov: number;
  near: number;
  far: number;
  mode: "orbit" | "fly" | "follow";
}

/**
 * Environment visual settings
 */
export interface EnvironmentSettings {
  backgroundColor: string;
  gridEnabled: boolean;
  gridSize: number;
  lighting: {
    ambient: { color: string; intensity: number };
    directional: { color: string; intensity: number; position: Position3D };
  };
  fog?: {
    enabled: boolean;
    color: string;
    near: number;
    far: number;
  };
}

/**
 * Training visualization state
 */
export interface TrainingVisualization {
  active: boolean;
  currentEpoch: number;
  totalEpochs: number;
  batchProgress: number;
  metrics: {
    loss: number[];
    accuracy?: number[];
    valLoss?: number[];
    valAccuracy?: number[];
    custom: Record<string, number[]>;
  };
  gradientFlow: Map<string, GradientVisualization>;
  activations: Map<string, ActivationVisualization>;
}

/**
 * Gradient flow visualization
 */
export interface GradientVisualization {
  layerId: string;
  magnitude: number;
  direction: number[];
  vanishing: boolean;
  exploding: boolean;
  histogram: number[];
}

/**
 * Activation visualization
 */
export interface ActivationVisualization {
  layerId: string;
  shape: number[];
  values: Float32Array;
  statistics: {
    min: number;
    max: number;
    mean: number;
    std: number;
    sparsity: number;
  };
  displayMode: "2d" | "3d" | "histogram";
}

/**
 * Playground interaction event
 */
export interface PlaygroundEvent {
  type:
    | "block-selected"
    | "block-moved"
    | "block-added"
    | "block-removed"
    | "connection-created"
    | "connection-removed"
    | "config-changed"
    | "training-started"
    | "training-stopped"
    | "epoch-complete"
    | "error";
  payload: unknown;
  timestamp: Date;
}

// ============================================================================
// Block Templates
// ============================================================================

/**
 * Predefined block templates for common architectures
 */
export const BLOCK_TEMPLATES: Record<
  string,
  { layers: Array<{ type: NeuralLayerType; config: LayerConfig }> }
> = {
  mlp: {
    layers: [
      { type: "dense", config: { units: 128, activation: "relu" } },
      { type: "dropout", config: { rate: 0.3 } },
      { type: "dense", config: { units: 64, activation: "relu" } },
      { type: "dense", config: { units: 10, activation: "softmax" } },
    ],
  },
  cnn: {
    layers: [
      {
        type: "conv2d",
        config: { filters: 32, kernelSize: 3, activation: "relu" },
      },
      { type: "maxPool2d", config: { poolSize: 2 } },
      {
        type: "conv2d",
        config: { filters: 64, kernelSize: 3, activation: "relu" },
      },
      { type: "maxPool2d", config: { poolSize: 2 } },
      { type: "flatten", config: {} },
      { type: "dense", config: { units: 128, activation: "relu" } },
      { type: "dense", config: { units: 10, activation: "softmax" } },
    ],
  },
  resnet_block: {
    layers: [
      {
        type: "conv2d",
        config: { filters: 64, kernelSize: 3, padding: "same" },
      },
      { type: "batchNorm", config: {} },
      {
        type: "conv2d",
        config: { filters: 64, kernelSize: 3, padding: "same" },
      },
      { type: "batchNorm", config: {} },
      { type: "add", config: {} }, // Skip connection
    ],
  },
  lstm_encoder: {
    layers: [
      { type: "embedding", config: { inputDim: 10000, outputDim: 128 } },
      { type: "lstm", config: { units: 64, returnSequences: true } },
      { type: "lstm", config: { units: 64, returnSequences: false } },
      { type: "dense", config: { units: 32, activation: "relu" } },
    ],
  },
  attention_block: {
    layers: [
      { type: "attention", config: { numHeads: 8, keyDim: 64 } },
      { type: "dense", config: { units: 256, activation: "relu" } },
      { type: "dense", config: { units: 64, activation: "linear" } },
    ],
  },
};

/**
 * Visual style presets for different layer types
 */
const LAYER_VISUAL_STYLES: Record<NeuralLayerType, BlockVisualStyle> = {
  dense: {
    color: "#4a90d9",
    opacity: 0.9,
    glow: { enabled: true, color: "#6ab0ff", intensity: 0.3 },
  },
  conv2d: {
    color: "#d94a4a",
    opacity: 0.9,
    glow: { enabled: true, color: "#ff6a6a", intensity: 0.3 },
  },
  maxPool2d: {
    color: "#d98c4a",
    opacity: 0.8,
  },
  avgPool2d: {
    color: "#d9a84a",
    opacity: 0.8,
  },
  flatten: {
    color: "#8c8c8c",
    opacity: 0.7,
  },
  dropout: {
    color: "#9b59b6",
    opacity: 0.6,
    animation: { type: "pulse", speed: 1 },
  },
  batchNorm: {
    color: "#27ae60",
    opacity: 0.8,
  },
  lstm: {
    color: "#e74c3c",
    opacity: 0.9,
    glow: { enabled: true, color: "#ff7675", intensity: 0.4 },
  },
  gru: {
    color: "#c0392b",
    opacity: 0.9,
    glow: { enabled: true, color: "#e74c3c", intensity: 0.4 },
  },
  attention: {
    color: "#f39c12",
    opacity: 0.9,
    glow: { enabled: true, color: "#f1c40f", intensity: 0.5 },
    animation: { type: "flow", speed: 2 },
  },
  embedding: {
    color: "#1abc9c",
    opacity: 0.9,
  },
  reshape: {
    color: "#95a5a6",
    opacity: 0.7,
  },
  concatenate: {
    color: "#3498db",
    opacity: 0.8,
  },
  add: {
    color: "#2ecc71",
    opacity: 0.8,
  },
  multiply: {
    color: "#e67e22",
    opacity: 0.8,
  },
};

// ============================================================================
// Neural3DPlayground Class
// ============================================================================

/**
 * 3D Neural Playground - Interactive neural architecture builder
 *
 * @example
 * ```typescript
 * const playground = new Neural3DPlayground();
 *
 * // Create a scene
 * const scene = await playground.createScene('My Neural Network');
 *
 * // Add layers
 * const input = await playground.addBlock(scene.id, 'dense', {
 *   units: 784,
 *   activation: 'linear',
 *   name: 'input'
 * }, { x: 0, y: 0, z: 0 });
 *
 * const hidden = await playground.addBlock(scene.id, 'dense', {
 *   units: 128,
 *   activation: 'relu',
 *   name: 'hidden'
 * }, { x: 5, y: 0, z: 0 });
 *
 * // Connect layers
 * await playground.connect(scene.id, input.id, hidden.id);
 *
 * // Compile to TensorFlow model
 * const model = await playground.compileToModel(scene.id);
 * ```
 */
export class Neural3DPlayground extends EventEmitter {
  private scenes: Map<string, PlaygroundScene> = new Map();
  private activeScene: string | null = null;
  private trainingState: TrainingVisualization | null = null;
  private compiledModels: Map<string, tf.LayersModel> = new Map();
  private undoStack: Map<string, PlaygroundEvent[]> = new Map();
  private redoStack: Map<string, PlaygroundEvent[]> = new Map();

  constructor() {
    super();
  }

  // ==========================================================================
  // Scene Management
  // ==========================================================================

  /**
   * Create a new playground scene
   */
  async createScene(
    name: string,
    options?: Partial<EnvironmentSettings>
  ): Promise<PlaygroundScene> {
    const id = `scene-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const scene: PlaygroundScene = {
      id,
      name,
      blocks: new Map(),
      connections: new Map(),
      camera: {
        position: { x: 0, y: 10, z: 20 },
        target: { x: 0, y: 0, z: 0 },
        fov: 60,
        near: 0.1,
        far: 1000,
        mode: "orbit",
      },
      environment: {
        backgroundColor: "#1a1a2e",
        gridEnabled: true,
        gridSize: 50,
        lighting: {
          ambient: { color: "#ffffff", intensity: 0.4 },
          directional: {
            color: "#ffffff",
            intensity: 0.8,
            position: { x: 10, y: 20, z: 10 },
          },
        },
        ...options,
      },
      metadata: {
        created: new Date(),
        modified: new Date(),
      },
    };

    this.scenes.set(id, scene);
    this.undoStack.set(id, []);
    this.redoStack.set(id, []);
    this.activeScene = id;

    this.emit("scene-created", { scene });
    return scene;
  }

  /**
   * Get a scene by ID
   */
  getScene(sceneId: string): PlaygroundScene | undefined {
    return this.scenes.get(sceneId);
  }

  /**
   * Set active scene
   */
  setActiveScene(sceneId: string): void {
    if (this.scenes.has(sceneId)) {
      this.activeScene = sceneId;
      this.emit("scene-activated", { sceneId });
    }
  }

  // ==========================================================================
  // Block Operations
  // ==========================================================================

  /**
   * Add a neural block to the scene
   */
  async addBlock(
    sceneId: string,
    type: NeuralLayerType,
    config: LayerConfig,
    position: Position3D
  ): Promise<NeuralBlock3D> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const id = `block-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Calculate scale based on layer type and config
    const scale = this.calculateBlockScale(type, config);

    const block: NeuralBlock3D = {
      id,
      type,
      position,
      rotation: { x: 0, y: 0, z: 0 },
      scale,
      config,
      connections: {
        inputs: [],
        outputs: [],
      },
      visualStyle: { ...LAYER_VISUAL_STYLES[type] },
      metadata: {
        paramCount: this.estimateParamCount(type, config),
        outputShape: [], // Will be calculated on compilation
      },
    };

    scene.blocks.set(id, block);
    scene.metadata.modified = new Date();

    this.recordAction(sceneId, {
      type: "block-added",
      payload: { block },
      timestamp: new Date(),
    });

    this.emit("block-added", { sceneId, block });
    return block;
  }

  /**
   * Remove a block and its connections
   */
  async removeBlock(sceneId: string, blockId: string): Promise<void> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const block = scene.blocks.get(blockId);
    if (!block) return;

    // Remove all connections to/from this block
    const connectionsToRemove: string[] = [];
    for (const [connId, conn] of scene.connections) {
      if (conn.sourceBlock === blockId || conn.targetBlock === blockId) {
        connectionsToRemove.push(connId);
      }
    }

    for (const connId of connectionsToRemove) {
      await this.disconnect(sceneId, connId);
    }

    scene.blocks.delete(blockId);
    scene.metadata.modified = new Date();

    this.recordAction(sceneId, {
      type: "block-removed",
      payload: { block },
      timestamp: new Date(),
    });

    this.emit("block-removed", { sceneId, blockId });
  }

  /**
   * Move a block to a new position
   */
  async moveBlock(
    sceneId: string,
    blockId: string,
    newPosition: Position3D
  ): Promise<void> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const block = scene.blocks.get(blockId);
    if (!block) throw new Error(`Block ${blockId} not found`);

    const oldPosition = { ...block.position };
    block.position = newPosition;
    scene.metadata.modified = new Date();

    // Update connection paths
    await this.updateConnectionPaths(scene, blockId);

    this.recordAction(sceneId, {
      type: "block-moved",
      payload: { blockId, oldPosition, newPosition },
      timestamp: new Date(),
    });

    this.emit("block-moved", { sceneId, blockId, position: newPosition });
  }

  /**
   * Update block configuration
   */
  async updateBlockConfig(
    sceneId: string,
    blockId: string,
    config: Partial<LayerConfig>
  ): Promise<void> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const block = scene.blocks.get(blockId);
    if (!block) throw new Error(`Block ${blockId} not found`);

    const oldConfig = { ...block.config };
    block.config = { ...block.config, ...config };
    block.scale = this.calculateBlockScale(block.type, block.config);
    block.metadata.paramCount = this.estimateParamCount(
      block.type,
      block.config
    );
    scene.metadata.modified = new Date();

    // Invalidate compiled model
    this.compiledModels.delete(sceneId);

    this.recordAction(sceneId, {
      type: "config-changed",
      payload: { blockId, oldConfig, newConfig: block.config },
      timestamp: new Date(),
    });

    this.emit("config-changed", { sceneId, blockId, config: block.config });
  }

  // ==========================================================================
  // Connection Operations
  // ==========================================================================

  /**
   * Connect two blocks
   */
  async connect(
    sceneId: string,
    sourceBlockId: string,
    targetBlockId: string
  ): Promise<NeuralConnection3D> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const sourceBlock = scene.blocks.get(sourceBlockId);
    const targetBlock = scene.blocks.get(targetBlockId);

    if (!sourceBlock)
      throw new Error(`Source block ${sourceBlockId} not found`);
    if (!targetBlock)
      throw new Error(`Target block ${targetBlockId} not found`);

    // Check for existing connection
    for (const conn of scene.connections.values()) {
      if (
        conn.sourceBlock === sourceBlockId &&
        conn.targetBlock === targetBlockId
      ) {
        throw new Error("Connection already exists");
      }
    }

    // Check for circular dependencies
    if (this.wouldCreateCycle(scene, sourceBlockId, targetBlockId)) {
      throw new Error("Connection would create a cycle");
    }

    const id = `conn-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const connection: NeuralConnection3D = {
      id,
      sourceBlock: sourceBlockId,
      targetBlock: targetBlockId,
      sourcePort: "output",
      targetPort: "input",
      pathPoints: this.calculateConnectionPath(
        sourceBlock.position,
        targetBlock.position
      ),
      style: {
        color: "#4a90d9",
        width: 2,
        animated: true,
        flowDirection: "forward",
      },
    };

    scene.connections.set(id, connection);
    sourceBlock.connections.outputs.push(targetBlockId);
    targetBlock.connections.inputs.push(sourceBlockId);
    scene.metadata.modified = new Date();

    // Invalidate compiled model
    this.compiledModels.delete(sceneId);

    this.recordAction(sceneId, {
      type: "connection-created",
      payload: { connection },
      timestamp: new Date(),
    });

    this.emit("connection-created", { sceneId, connection });
    return connection;
  }

  /**
   * Remove a connection
   */
  async disconnect(sceneId: string, connectionId: string): Promise<void> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const connection = scene.connections.get(connectionId);
    if (!connection) return;

    const sourceBlock = scene.blocks.get(connection.sourceBlock);
    const targetBlock = scene.blocks.get(connection.targetBlock);

    if (sourceBlock) {
      sourceBlock.connections.outputs = sourceBlock.connections.outputs.filter(
        (id) => id !== connection.targetBlock
      );
    }

    if (targetBlock) {
      targetBlock.connections.inputs = targetBlock.connections.inputs.filter(
        (id) => id !== connection.sourceBlock
      );
    }

    scene.connections.delete(connectionId);
    scene.metadata.modified = new Date();

    // Invalidate compiled model
    this.compiledModels.delete(sceneId);

    this.recordAction(sceneId, {
      type: "connection-removed",
      payload: { connection },
      timestamp: new Date(),
    });

    this.emit("connection-removed", { sceneId, connectionId });
  }

  // ==========================================================================
  // Model Compilation
  // ==========================================================================

  /**
   * Compile the visual architecture to a TensorFlow.js model
   */
  async compileToModel(
    sceneId: string,
    inputShape: number[],
    options?: {
      optimizer?: string;
      loss?: string;
      metrics?: string[];
    }
  ): Promise<tf.LayersModel> {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    // Check for cached model
    if (this.compiledModels.has(sceneId)) {
      return this.compiledModels.get(sceneId)!;
    }

    // Get topologically sorted blocks
    const sortedBlocks = this.topologicalSort(scene);

    if (sortedBlocks.length === 0) {
      throw new Error("No blocks in scene");
    }

    // Build model using functional API
    const blockOutputs = new Map<string, tf.SymbolicTensor>();

    // Create input layer
    const input = tf.input({ shape: inputShape });
    let currentTensor: tf.SymbolicTensor = input;

    // Find input blocks (blocks with no inputs)
    const inputBlocks = sortedBlocks.filter(
      (b) => b.connections.inputs.length === 0
    );

    if (inputBlocks.length === 0) {
      throw new Error("No input block found");
    }

    // Process first input block
    currentTensor = this.applyLayer(inputBlocks[0], currentTensor);
    blockOutputs.set(inputBlocks[0].id, currentTensor);

    // Process remaining blocks
    for (const block of sortedBlocks.slice(1)) {
      if (block.connections.inputs.length === 0) continue;

      // Get input tensors from connected blocks
      const inputTensors = block.connections.inputs
        .map((inputId) => blockOutputs.get(inputId))
        .filter((t): t is tf.SymbolicTensor => t !== undefined);

      if (inputTensors.length === 0) {
        throw new Error(`Block ${block.id} has no valid inputs`);
      }

      // Handle multiple inputs (concatenate, add, multiply)
      if (inputTensors.length > 1) {
        if (block.type === "concatenate") {
          currentTensor = tf.layers
            .concatenate()
            .apply(inputTensors) as tf.SymbolicTensor;
        } else if (block.type === "add") {
          currentTensor = tf.layers
            .add()
            .apply(inputTensors) as tf.SymbolicTensor;
        } else if (block.type === "multiply") {
          currentTensor = tf.layers
            .multiply()
            .apply(inputTensors) as tf.SymbolicTensor;
        } else {
          // Default: use first input
          currentTensor = inputTensors[0];
        }
      } else {
        currentTensor = inputTensors[0];
      }

      // Apply layer
      currentTensor = this.applyLayer(block, currentTensor);
      blockOutputs.set(block.id, currentTensor);
    }

    // Create model
    const model = tf.model({
      inputs: input,
      outputs: currentTensor,
    });

    // Compile model
    model.compile({
      optimizer: options?.optimizer || "adam",
      loss: options?.loss || "categoricalCrossentropy",
      metrics: options?.metrics || ["accuracy"],
    });

    // Cache compiled model
    this.compiledModels.set(sceneId, model);

    // Update block metadata with output shapes
    this.updateOutputShapes(scene, model);

    this.emit("model-compiled", { sceneId, model });
    return model;
  }

  /**
   * Apply a block as a TensorFlow layer
   */
  private applyLayer(
    block: NeuralBlock3D,
    input: tf.SymbolicTensor
  ): tf.SymbolicTensor {
    const config = block.config;

    switch (block.type) {
      case "dense":
        return tf.layers
          .dense({
            units: config.units || 64,
            activation: config.activation || "linear",
            useBias: config.useBias ?? true,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "conv2d":
        return tf.layers
          .conv2d({
            filters: config.filters || 32,
            kernelSize: config.kernelSize || 3,
            strides: config.strides || 1,
            padding: config.padding || "valid",
            activation: config.activation || "linear",
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "maxPool2d":
        return tf.layers
          .maxPooling2d({
            poolSize: config.poolSize || 2,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "avgPool2d":
        return tf.layers
          .averagePooling2d({
            poolSize: config.poolSize || 2,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "flatten":
        return tf.layers
          .flatten({ name: config.name })
          .apply(input) as tf.SymbolicTensor;

      case "dropout":
        return tf.layers
          .dropout({
            rate: config.rate || 0.5,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "batchNorm":
        return tf.layers
          .batchNormalization({ name: config.name })
          .apply(input) as tf.SymbolicTensor;

      case "lstm":
        return tf.layers
          .lstm({
            units: config.units || 64,
            returnSequences: config.returnSequences ?? false,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "gru":
        return tf.layers
          .gru({
            units: config.units || 64,
            returnSequences: config.returnSequences ?? false,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "embedding":
        return tf.layers
          .embedding({
            inputDim: config.inputDim || 10000,
            outputDim: config.outputDim || 128,
            name: config.name,
          })
          .apply(input) as tf.SymbolicTensor;

      case "reshape":
        // Reshape requires target shape in config
        return tf.layers
          .reshape({ targetShape: [1] as number[], name: config.name })
          .apply(input) as tf.SymbolicTensor;

      default:
        // Pass through for unsupported layers
        return input;
    }
  }

  // ==========================================================================
  // Training Visualization
  // ==========================================================================

  /**
   * Start training with visualization
   */
  async startTraining(
    sceneId: string,
    data: {
      xs: tf.Tensor;
      ys: tf.Tensor;
      validationXs?: tf.Tensor;
      validationYs?: tf.Tensor;
    },
    config: {
      epochs: number;
      batchSize: number;
      callbacks?: tf.CustomCallbackArgs;
    }
  ): Promise<tf.History> {
    const model = this.compiledModels.get(sceneId);
    if (!model) {
      throw new Error("Model not compiled. Call compileToModel first.");
    }

    this.trainingState = {
      active: true,
      currentEpoch: 0,
      totalEpochs: config.epochs,
      batchProgress: 0,
      metrics: {
        loss: [],
        custom: {},
      },
      gradientFlow: new Map(),
      activations: new Map(),
    };

    this.emit("training-started", { sceneId });

    const callbacks: tf.CustomCallbackArgs = {
      onEpochBegin: async (epoch: number) => {
        if (this.trainingState) {
          this.trainingState.currentEpoch = epoch;
        }
        this.emit("epoch-begin", { sceneId, epoch });
      },
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        if (this.trainingState && logs) {
          this.trainingState.metrics.loss.push(logs.loss || 0);
          if (logs.acc !== undefined) {
            this.trainingState.metrics.accuracy =
              this.trainingState.metrics.accuracy || [];
            this.trainingState.metrics.accuracy.push(logs.acc);
          }
          if (logs.val_loss !== undefined) {
            this.trainingState.metrics.valLoss =
              this.trainingState.metrics.valLoss || [];
            this.trainingState.metrics.valLoss.push(logs.val_loss);
          }
          if (logs.val_acc !== undefined) {
            this.trainingState.metrics.valAccuracy =
              this.trainingState.metrics.valAccuracy || [];
            this.trainingState.metrics.valAccuracy.push(logs.val_acc);
          }
        }
        this.emit("epoch-complete", { sceneId, epoch, logs });
      },
      onBatchEnd: async (batch: number, logs?: tf.Logs) => {
        if (this.trainingState) {
          this.trainingState.batchProgress = batch;
        }
        this.emit("batch-complete", { sceneId, batch, logs });
      },
      ...config.callbacks,
    };

    try {
      const history = await model.fit(data.xs, data.ys, {
        epochs: config.epochs,
        batchSize: config.batchSize,
        validationData:
          data.validationXs && data.validationYs
            ? [data.validationXs, data.validationYs]
            : undefined,
        callbacks,
      });

      return history;
    } finally {
      if (this.trainingState) {
        this.trainingState.active = false;
      }
      this.emit("training-complete", { sceneId });
    }
  }

  /**
   * Stop current training
   */
  stopTraining(): void {
    if (this.trainingState) {
      this.trainingState.active = false;
      this.emit("training-stopped", {});
    }
  }

  /**
   * Get current training state
   */
  getTrainingState(): TrainingVisualization | null {
    return this.trainingState;
  }

  // ==========================================================================
  // Templates & Presets
  // ==========================================================================

  /**
   * Load a predefined architecture template
   */
  async loadTemplate(
    sceneId: string,
    templateName: keyof typeof BLOCK_TEMPLATES
  ): Promise<void> {
    const template = BLOCK_TEMPLATES[templateName];
    if (!template) {
      throw new Error(`Template ${templateName} not found`);
    }

    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    // Clear existing blocks
    for (const blockId of scene.blocks.keys()) {
      await this.removeBlock(sceneId, blockId);
    }

    // Add blocks from template
    const blockIds: string[] = [];
    let xPosition = 0;

    for (const layerDef of template.layers) {
      const block = await this.addBlock(
        sceneId,
        layerDef.type,
        layerDef.config,
        {
          x: xPosition,
          y: 0,
          z: 0,
        }
      );
      blockIds.push(block.id);
      xPosition += 5;
    }

    // Connect blocks sequentially
    for (let i = 0; i < blockIds.length - 1; i++) {
      await this.connect(sceneId, blockIds[i], blockIds[i + 1]);
    }

    this.emit("template-loaded", { sceneId, templateName });
  }

  /**
   * Export scene as JSON
   */
  exportScene(sceneId: string): string {
    const scene = this.scenes.get(sceneId);
    if (!scene) throw new Error(`Scene ${sceneId} not found`);

    const exportData = {
      ...scene,
      blocks: Array.from(scene.blocks.values()),
      connections: Array.from(scene.connections.values()),
    };

    return JSON.stringify(exportData, null, 2);
  }

  /**
   * Import scene from JSON
   */
  async importScene(json: string): Promise<PlaygroundScene> {
    const data = JSON.parse(json);

    const scene = await this.createScene(data.name, data.environment);

    // Import blocks
    for (const blockData of data.blocks) {
      await this.addBlock(
        scene.id,
        blockData.type,
        blockData.config,
        blockData.position
      );
    }

    // Import connections
    for (const connData of data.connections) {
      await this.connect(scene.id, connData.sourceBlock, connData.targetBlock);
    }

    return scene;
  }

  // ==========================================================================
  // Undo/Redo
  // ==========================================================================

  /**
   * Undo last action
   */
  async undo(sceneId: string): Promise<void> {
    const actions = this.undoStack.get(sceneId);
    if (!actions || actions.length === 0) return;

    const action = actions.pop()!;
    this.redoStack.get(sceneId)?.push(action);

    // Reverse the action
    await this.reverseAction(sceneId, action);

    this.emit("undo", { sceneId, action });
  }

  /**
   * Redo last undone action
   */
  async redo(sceneId: string): Promise<void> {
    const actions = this.redoStack.get(sceneId);
    if (!actions || actions.length === 0) return;

    const action = actions.pop()!;
    this.undoStack.get(sceneId)?.push(action);

    // Reapply the action
    await this.reapplyAction(sceneId, action);

    this.emit("redo", { sceneId, action });
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  private calculateBlockScale(
    type: NeuralLayerType,
    config: LayerConfig
  ): { width: number; height: number; depth: number } {
    // Scale based on layer complexity
    const baseScale = { width: 2, height: 2, depth: 2 };

    switch (type) {
      case "dense":
        baseScale.width = Math.min(
          5,
          Math.log10((config.units || 64) + 1) * 1.5
        );
        break;
      case "conv2d":
        baseScale.width = Math.min(
          4,
          Math.log10((config.filters || 32) + 1) * 1.5
        );
        baseScale.depth = Math.min(
          3,
          ((config.kernelSize as number) || 3) * 0.5
        );
        break;
      case "lstm":
      case "gru":
        baseScale.width = 3;
        baseScale.height = 3;
        break;
      case "attention":
        baseScale.width = 4;
        baseScale.height = 2;
        baseScale.depth = 2;
        break;
      case "flatten":
        baseScale.width = 1;
        baseScale.height = 0.5;
        baseScale.depth = 4;
        break;
    }

    return baseScale;
  }

  private estimateParamCount(
    type: NeuralLayerType,
    config: LayerConfig
  ): number {
    // Rough parameter estimation
    switch (type) {
      case "dense":
        return (config.units || 64) * ((config.units || 64) + 1);
      case "conv2d": {
        const kernelSize = (config.kernelSize as number) || 3;
        return (config.filters || 32) * kernelSize * kernelSize;
      }
      case "lstm":
      case "gru":
        return (config.units || 64) * 4 * ((config.units || 64) + 1);
      case "embedding":
        return (config.inputDim || 10000) * (config.outputDim || 128);
      default:
        return 0;
    }
  }

  private calculateConnectionPath(
    source: Position3D,
    target: Position3D
  ): Position3D[] {
    // Simple bezier-like path
    const midX = (source.x + target.x) / 2;
    return [
      source,
      { x: midX, y: source.y + 2, z: source.z },
      { x: midX, y: target.y + 2, z: target.z },
      target,
    ];
  }

  private wouldCreateCycle(
    scene: PlaygroundScene,
    sourceId: string,
    targetId: string
  ): boolean {
    // DFS to check for cycles
    const visited = new Set<string>();

    const dfs = (blockId: string): boolean => {
      if (blockId === sourceId) return true;
      if (visited.has(blockId)) return false;

      visited.add(blockId);

      const block = scene.blocks.get(blockId);
      if (!block) return false;

      for (const outputId of block.connections.outputs) {
        if (dfs(outputId)) return true;
      }

      return false;
    };

    return dfs(targetId);
  }

  private topologicalSort(scene: PlaygroundScene): NeuralBlock3D[] {
    const sorted: NeuralBlock3D[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (blockId: string) => {
      if (visited.has(blockId)) return;
      if (visiting.has(blockId)) throw new Error("Cycle detected");

      visiting.add(blockId);

      const block = scene.blocks.get(blockId);
      if (block) {
        for (const inputId of block.connections.inputs) {
          visit(inputId);
        }
        visiting.delete(blockId);
        visited.add(blockId);
        sorted.push(block);
      }
    };

    for (const blockId of scene.blocks.keys()) {
      visit(blockId);
    }

    return sorted;
  }

  private async updateConnectionPaths(
    scene: PlaygroundScene,
    blockId: string
  ): Promise<void> {
    for (const conn of scene.connections.values()) {
      if (conn.sourceBlock === blockId || conn.targetBlock === blockId) {
        const source = scene.blocks.get(conn.sourceBlock);
        const target = scene.blocks.get(conn.targetBlock);
        if (source && target) {
          conn.pathPoints = this.calculateConnectionPath(
            source.position,
            target.position
          );
        }
      }
    }
  }

  private updateOutputShapes(
    scene: PlaygroundScene,
    _model: tf.LayersModel
  ): void {
    // Get output shapes from model layers
    // This is a simplified version - full implementation would trace through the model
    for (const block of scene.blocks.values()) {
      // Mark as placeholder - real implementation would extract from model
      block.metadata.outputShape = [];
    }
  }

  private recordAction(sceneId: string, action: PlaygroundEvent): void {
    const actions = this.undoStack.get(sceneId);
    if (actions) {
      actions.push(action);
      // Limit undo stack size
      if (actions.length > 50) {
        actions.shift();
      }
    }
    // Clear redo stack on new action
    this.redoStack.set(sceneId, []);
  }

  private async reverseAction(
    sceneId: string,
    action: PlaygroundEvent
  ): Promise<void> {
    switch (action.type) {
      case "block-added": {
        const { block } = action.payload as { block: NeuralBlock3D };
        await this.removeBlock(sceneId, block.id);
        break;
      }
      case "block-removed": {
        const { block } = action.payload as { block: NeuralBlock3D };
        await this.addBlock(sceneId, block.type, block.config, block.position);
        break;
      }
      case "block-moved": {
        const { blockId, oldPosition } = action.payload as {
          blockId: string;
          oldPosition: Position3D;
        };
        await this.moveBlock(sceneId, blockId, oldPosition);
        break;
      }
      case "config-changed": {
        const { blockId, oldConfig } = action.payload as {
          blockId: string;
          oldConfig: LayerConfig;
        };
        await this.updateBlockConfig(sceneId, blockId, oldConfig);
        break;
      }
      case "connection-created": {
        const { connection } = action.payload as {
          connection: NeuralConnection3D;
        };
        await this.disconnect(sceneId, connection.id);
        break;
      }
      case "connection-removed": {
        const { connection } = action.payload as {
          connection: NeuralConnection3D;
        };
        await this.connect(
          sceneId,
          connection.sourceBlock,
          connection.targetBlock
        );
        break;
      }
    }
  }

  private async reapplyAction(
    sceneId: string,
    action: PlaygroundEvent
  ): Promise<void> {
    switch (action.type) {
      case "block-added": {
        const { block } = action.payload as { block: NeuralBlock3D };
        await this.addBlock(sceneId, block.type, block.config, block.position);
        break;
      }
      case "block-removed": {
        const { block } = action.payload as { block: NeuralBlock3D };
        await this.removeBlock(sceneId, block.id);
        break;
      }
      case "block-moved": {
        const { blockId, newPosition } = action.payload as {
          blockId: string;
          newPosition: Position3D;
        };
        await this.moveBlock(sceneId, blockId, newPosition);
        break;
      }
      case "config-changed": {
        const { blockId, newConfig } = action.payload as {
          blockId: string;
          newConfig: LayerConfig;
        };
        await this.updateBlockConfig(sceneId, blockId, newConfig);
        break;
      }
      case "connection-created": {
        const { connection } = action.payload as {
          connection: NeuralConnection3D;
        };
        await this.connect(
          sceneId,
          connection.sourceBlock,
          connection.targetBlock
        );
        break;
      }
      case "connection-removed": {
        const { connection } = action.payload as {
          connection: NeuralConnection3D;
        };
        await this.disconnect(sceneId, connection.id);
        break;
      }
    }
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    for (const model of this.compiledModels.values()) {
      model.dispose();
    }
    this.compiledModels.clear();
    this.scenes.clear();
    this.undoStack.clear();
    this.redoStack.clear();
    this.trainingState = null;
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new Neural 3D Playground instance
 */
export function createNeural3DPlayground(): Neural3DPlayground {
  return new Neural3DPlayground();
}

/**
 * Quick create a playground with a scene and template loaded
 */
export async function createPlaygroundWithTemplate(
  sceneName: string,
  template: keyof typeof BLOCK_TEMPLATES
): Promise<{ playground: Neural3DPlayground; scene: PlaygroundScene }> {
  const playground = createNeural3DPlayground();
  const scene = await playground.createScene(sceneName);
  await playground.loadTemplate(scene.id, template);

  return { playground, scene };
}

export default Neural3DPlayground;
