/**
 * Training Progress 4D Journey - Forge×Foundry Innovation #2
 *
 * A 4D visualization of the training evolution - navigable timeline
 * that shows loss landscapes, weight distributions, activation patterns,
 * and gradient flows across training epochs. Users can "time travel"
 * through the training process to understand how the model evolved.
 *
 * @module Training4DJourney
 * @category CrossDomain/Innovation
 */

import { EventEmitter } from "events";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * Training snapshot at a specific point in time
 */
export interface TrainingSnapshot {
  id: string;
  epoch: number;
  batch: number;
  timestamp: number;

  // Core metrics
  metrics: TrainingMetricsSnapshot;

  // Model state
  weights: WeightsSnapshot;
  gradients: GradientsSnapshot;
  activations: ActivationsSnapshot;

  // Optimizer state
  optimizerState: OptimizerStateSnapshot;

  // Loss landscape sample
  lossLandscape: LossLandscapeSample;

  // Memory efficient delta from previous
  deltaFromPrevious?: SnapshotDelta;
}

/**
 * Training metrics at snapshot
 */
export interface TrainingMetricsSnapshot {
  loss: number;
  accuracy: number;
  learningRate: number;
  gradientNorm: number;

  // Per-layer metrics
  layerMetrics: Map<string, LayerMetricsSnapshot>;

  // Validation metrics (if available)
  validationLoss?: number;
  validationAccuracy?: number;

  // Custom metrics
  custom: Map<string, number>;
}

/**
 * Per-layer metrics
 */
export interface LayerMetricsSnapshot {
  layerId: string;
  parameterNorm: number;
  gradientNorm: number;
  activationMean: number;
  activationStd: number;
  sparsity: number;
}

/**
 * Weights snapshot (compressed)
 */
export interface WeightsSnapshot {
  // Statistical summary (always stored)
  statistics: Map<string, WeightStatistics>;

  // Full weights (optional, for key snapshots)
  fullWeights?: Map<string, Float32Array>;

  // Compressed representation
  compressed?: CompressedWeights;
}

/**
 * Weight statistics for a layer
 */
export interface WeightStatistics {
  layerId: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  l1Norm: number;
  l2Norm: number;
  sparsity: number;
  histogram: number[];
}

/**
 * Compressed weight representation
 */
export interface CompressedWeights {
  method: "svd" | "quantized" | "sparse" | "pca";
  data: ArrayBuffer;
  metadata: Record<string, unknown>;
}

/**
 * Gradients snapshot
 */
export interface GradientsSnapshot {
  statistics: Map<string, GradientStatistics>;
  flowVisualization: GradientFlowVisualization;
}

/**
 * Gradient statistics for a layer
 */
export interface GradientStatistics {
  layerId: string;
  mean: number;
  std: number;
  norm: number;
  vanishing: boolean;
  exploding: boolean;
  signAgreement: number; // How consistent gradient signs are
}

/**
 * Gradient flow visualization data
 */
export interface GradientFlowVisualization {
  // Per-layer flow magnitudes for visualization
  layerFlows: Array<{
    layerId: string;
    magnitude: number;
    direction: Vector3D;
    color: string;
  }>;

  // Global flow health
  overallHealth: "healthy" | "vanishing" | "exploding" | "unstable";
}

/**
 * Activations snapshot
 */
export interface ActivationsSnapshot {
  statistics: Map<string, ActivationStatistics>;
  patterns: ActivationPatterns;
}

/**
 * Activation statistics
 */
export interface ActivationStatistics {
  layerId: string;
  mean: number;
  std: number;
  percentActive: number; // For ReLU-like activations
  distribution: number[]; // Histogram
}

/**
 * Activation patterns for visualization
 */
export interface ActivationPatterns {
  // Representative activation patterns (t-SNE reduced)
  reducedPatterns: Array<{
    layerId: string;
    pattern: Float32Array;
    position: Vector3D; // In reduced space
  }>;
}

/**
 * Optimizer state snapshot
 */
export interface OptimizerStateSnapshot {
  type: string;
  momentum?: Map<string, Float32Array>;
  velocity?: Map<string, Float32Array>;
  adaptiveRates?: Map<string, Float32Array>;
  stepCount: number;
}

/**
 * Loss landscape sample
 */
export interface LossLandscapeSample {
  // Current position in weight space
  currentPosition: Vector3D;

  // Local curvature estimate
  curvature: LandscapeCurvature;

  // Sampled neighborhood
  neighborhood: LossNeighborhood;

  // Trajectory path
  trajectoryPoint: Vector3D;
}

/**
 * Landscape curvature at a point
 */
export interface LandscapeCurvature {
  // Principal curvatures (eigenvalues of Hessian approximation)
  principalCurvatures: number[];

  // Condition number (indicates landscape difficulty)
  conditionNumber: number;

  // Local sharpness metric
  sharpness: number;
}

/**
 * Loss values in neighborhood
 */
export interface LossNeighborhood {
  // Grid of loss values in 2D projected space
  grid: Float32Array;
  gridSize: number;

  // Range in weight space
  range: number;

  // Projection directions
  direction1: Float32Array;
  direction2: Float32Array;
}

/**
 * Delta from previous snapshot (for efficiency)
 */
export interface SnapshotDelta {
  previousSnapshotId: string;
  weightDeltas: Map<string, Float32Array>;
  changedLayers: string[];
}

/**
 * 3D vector
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * 4D journey configuration
 */
export interface Journey4DConfig {
  // Snapshot frequency
  snapshotEveryNBatches: number;
  keySnapshotEveryNEpochs: number;

  // What to capture
  captureWeights: boolean;
  captureGradients: boolean;
  captureActivations: boolean;
  captureLossLandscape: boolean;

  // Compression settings
  compressionLevel: "none" | "light" | "heavy";

  // Memory limits
  maxSnapshotsInMemory: number;
  enableDiskPersistence: boolean;

  // Visualization settings
  landscapeSampleResolution: number;
  trajectorySmoothing: number;
}

/**
 * Time navigation state
 */
export interface TimeNavigationState {
  currentSnapshotIndex: number;
  playbackSpeed: number;
  isPlaying: boolean;
  playDirection: "forward" | "backward";
  loopMode: "none" | "loop" | "pingpong";
}

/**
 * Visualization mode for the journey
 */
export type JourneyVisualizationMode =
  | "loss_landscape"
  | "weight_evolution"
  | "gradient_flow"
  | "activation_patterns"
  | "trajectory"
  | "comprehensive";

/**
 * Camera configuration for 4D view
 */
export interface Journey4DCamera {
  position: Vector3D;
  target: Vector3D;
  upVector: Vector3D;
  fieldOfView: number;
  near: number;
  far: number;
}

/**
 * Analysis results
 */
export interface JourneyAnalysis {
  // Critical points in training
  criticalPoints: CriticalPoint[];

  // Phase transitions
  phases: TrainingPhase[];

  // Anomalies detected
  anomalies: TrainingAnomaly[];

  // Overall trajectory analysis
  trajectoryAnalysis: TrajectoryAnalysis;
}

/**
 * Critical point in training
 */
export interface CriticalPoint {
  snapshotId: string;
  type:
    | "plateau"
    | "breakthrough"
    | "degradation"
    | "convergence"
    | "divergence";
  epoch: number;
  description: string;
  importance: number;
}

/**
 * Training phase
 */
export interface TrainingPhase {
  name: string;
  startEpoch: number;
  endEpoch: number;
  characteristics: string[];
  dominantDynamics: string;
}

/**
 * Training anomaly
 */
export interface TrainingAnomaly {
  snapshotId: string;
  type:
    | "gradient_explosion"
    | "gradient_vanishing"
    | "loss_spike"
    | "nan_detected"
    | "numerical_instability";
  severity: "warning" | "error" | "critical";
  description: string;
  suggestedAction: string;
}

/**
 * Trajectory analysis
 */
export interface TrajectoryAnalysis {
  totalDistance: number;
  averageSpeed: number;
  convergenceRate: number;
  oscillationScore: number;
  directionalConsistency: number;
}

// ============================================================================
// Training 4D Journey Implementation
// ============================================================================

/**
 * Training Progress 4D Journey
 *
 * Captures and visualizes the complete training evolution in 4D
 * (3D space + time), enabling temporal navigation through the
 * model's learning journey.
 */
export class Training4DJourney extends EventEmitter {
  private config: Journey4DConfig;
  private snapshots: TrainingSnapshot[];
  private snapshotIndex: Map<string, number>;
  private timeNavigation: TimeNavigationState;
  private visualizationMode: JourneyVisualizationMode;
  private camera: Journey4DCamera;
  private analysis: JourneyAnalysis | null;
  private isCapturing: boolean;
  private animationFrame: number | null;

  // Trajectory data
  private trajectoryPath: Vector3D[];
  private lossLandscapeMesh: LossLandscapeMesh | null;

  constructor(config: Partial<Journey4DConfig> = {}) {
    super();

    this.config = this.mergeConfig(config);
    this.snapshots = [];
    this.snapshotIndex = new Map();
    this.timeNavigation = this.createInitialNavigationState();
    this.visualizationMode = "comprehensive";
    this.camera = this.createDefaultCamera();
    this.analysis = null;
    this.isCapturing = false;
    this.animationFrame = null;
    this.trajectoryPath = [];
    this.lossLandscapeMesh = null;
  }

  /**
   * Merge user config with defaults
   */
  private mergeConfig(config: Partial<Journey4DConfig>): Journey4DConfig {
    return {
      snapshotEveryNBatches: config.snapshotEveryNBatches ?? 100,
      keySnapshotEveryNEpochs: config.keySnapshotEveryNEpochs ?? 5,
      captureWeights: config.captureWeights ?? true,
      captureGradients: config.captureGradients ?? true,
      captureActivations: config.captureActivations ?? true,
      captureLossLandscape: config.captureLossLandscape ?? true,
      compressionLevel: config.compressionLevel ?? "light",
      maxSnapshotsInMemory: config.maxSnapshotsInMemory ?? 1000,
      enableDiskPersistence: config.enableDiskPersistence ?? false,
      landscapeSampleResolution: config.landscapeSampleResolution ?? 25,
      trajectorySmoothing: config.trajectorySmoothing ?? 0.5,
    };
  }

  /**
   * Create initial navigation state
   */
  private createInitialNavigationState(): TimeNavigationState {
    return {
      currentSnapshotIndex: 0,
      playbackSpeed: 1.0,
      isPlaying: false,
      playDirection: "forward",
      loopMode: "none",
    };
  }

  /**
   * Create default camera
   */
  private createDefaultCamera(): Journey4DCamera {
    return {
      position: { x: 10, y: 10, z: 10 },
      target: { x: 0, y: 0, z: 0 },
      upVector: { x: 0, y: 1, z: 0 },
      fieldOfView: 60,
      near: 0.1,
      far: 1000,
    };
  }

  // ============================================================================
  // Snapshot Capture
  // ============================================================================

  /**
   * Start capturing training snapshots
   */
  startCapture(): void {
    if (this.isCapturing) return;

    this.isCapturing = true;
    this.emit("capture:started");
  }

  /**
   * Stop capturing
   */
  stopCapture(): void {
    this.isCapturing = false;
    this.emit("capture:stopped");
  }

  /**
   * Capture a training snapshot
   */
  captureSnapshot(
    epoch: number,
    batch: number,
    modelState: ModelStateForCapture
  ): string {
    if (!this.isCapturing) {
      throw new Error("Capture not started");
    }

    const snapshot = this.createSnapshot(epoch, batch, modelState);
    this.addSnapshot(snapshot);

    // Update trajectory
    this.updateTrajectory(snapshot);

    // Check for anomalies
    this.checkForAnomalies(snapshot);

    this.emit("snapshot:captured", snapshot.id);

    return snapshot.id;
  }

  /**
   * Create snapshot from model state
   */
  private createSnapshot(
    epoch: number,
    batch: number,
    modelState: ModelStateForCapture
  ): TrainingSnapshot {
    const id = this.generateSnapshotId(epoch, batch);
    const isKeySnapshot =
      epoch % this.config.keySnapshotEveryNEpochs === 0 && batch === 0;

    // Compute loss landscape sample
    const lossLandscape = this.config.captureLossLandscape
      ? this.sampleLossLandscape(modelState)
      : this.createEmptyLossLandscape();

    // Compute delta from previous if not key snapshot
    let deltaFromPrevious: SnapshotDelta | undefined;
    if (!isKeySnapshot && this.snapshots.length > 0) {
      deltaFromPrevious = this.computeDelta(
        this.snapshots[this.snapshots.length - 1],
        modelState
      );
    }

    return {
      id,
      epoch,
      batch,
      timestamp: Date.now(),
      metrics: this.captureMetrics(modelState),
      weights: this.captureWeights(modelState, isKeySnapshot),
      gradients: this.captureGradients(modelState),
      activations: this.captureActivations(modelState),
      optimizerState: this.captureOptimizerState(modelState),
      lossLandscape,
      deltaFromPrevious,
    };
  }

  /**
   * Generate snapshot ID
   */
  private generateSnapshotId(epoch: number, batch: number): string {
    return `snapshot_e${epoch}_b${batch}_${Date.now()}`;
  }

  /**
   * Capture training metrics
   */
  private captureMetrics(state: ModelStateForCapture): TrainingMetricsSnapshot {
    const layerMetrics = new Map<string, LayerMetricsSnapshot>();

    for (const [layerId, layerState] of state.layers) {
      layerMetrics.set(layerId, {
        layerId,
        parameterNorm: this.computeNorm(layerState.weights),
        gradientNorm: this.computeNorm(layerState.gradients),
        activationMean: this.computeMean(layerState.activations),
        activationStd: this.computeStd(layerState.activations),
        sparsity: this.computeSparsity(layerState.activations),
      });
    }

    return {
      loss: state.loss,
      accuracy: state.accuracy,
      learningRate: state.learningRate,
      gradientNorm: state.globalGradientNorm,
      layerMetrics,
      validationLoss: state.validationLoss,
      validationAccuracy: state.validationAccuracy,
      custom: new Map(Object.entries(state.customMetrics || {})),
    };
  }

  /**
   * Capture weights snapshot
   */
  private captureWeights(
    state: ModelStateForCapture,
    captureFullWeights: boolean
  ): WeightsSnapshot {
    const statistics = new Map<string, WeightStatistics>();
    const fullWeights = captureFullWeights
      ? new Map<string, Float32Array>()
      : undefined;

    for (const [layerId, layerState] of state.layers) {
      statistics.set(
        layerId,
        this.computeWeightStatistics(layerId, layerState.weights)
      );

      if (captureFullWeights) {
        fullWeights!.set(layerId, new Float32Array(layerState.weights));
      }
    }

    // Apply compression if configured
    let compressed: CompressedWeights | undefined;
    if (this.config.compressionLevel !== "none" && !captureFullWeights) {
      compressed = this.compressWeights(state, this.config.compressionLevel);
    }

    return { statistics, fullWeights, compressed };
  }

  /**
   * Compute weight statistics
   */
  private computeWeightStatistics(
    layerId: string,
    weights: Float32Array
  ): WeightStatistics {
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    let l1Norm = 0;
    let zeroCount = 0;

    for (const w of weights) {
      sum += w;
      sumSq += w * w;
      min = Math.min(min, w);
      max = Math.max(max, w);
      l1Norm += Math.abs(w);
      if (Math.abs(w) < 1e-7) zeroCount++;
    }

    const n = weights.length;
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const std = Math.sqrt(Math.max(0, variance));
    const l2Norm = Math.sqrt(sumSq);
    const sparsity = zeroCount / n;

    // Compute histogram
    const histogram = this.computeHistogram(weights, 50);

    return {
      layerId,
      mean,
      std,
      min,
      max,
      l1Norm,
      l2Norm,
      sparsity,
      histogram,
    };
  }

  /**
   * Capture gradients snapshot
   */
  private captureGradients(state: ModelStateForCapture): GradientsSnapshot {
    const statistics = new Map<string, GradientStatistics>();
    const layerFlows: GradientFlowVisualization["layerFlows"] = [];

    let hasVanishing = false;
    let hasExploding = false;

    for (const [layerId, layerState] of state.layers) {
      const stats = this.computeGradientStatistics(
        layerId,
        layerState.gradients
      );
      statistics.set(layerId, stats);

      if (stats.vanishing) hasVanishing = true;
      if (stats.exploding) hasExploding = true;

      // Create flow visualization data
      layerFlows.push({
        layerId,
        magnitude: stats.norm,
        direction: { x: 0, y: stats.norm > 0 ? 1 : 0, z: 0 },
        color: this.getGradientFlowColor(stats),
      });
    }

    // Determine overall health
    let overallHealth: GradientFlowVisualization["overallHealth"] = "healthy";
    if (hasExploding) overallHealth = "exploding";
    else if (hasVanishing) overallHealth = "vanishing";
    else if (hasVanishing && hasExploding) overallHealth = "unstable";

    return {
      statistics,
      flowVisualization: { layerFlows, overallHealth },
    };
  }

  /**
   * Compute gradient statistics
   */
  private computeGradientStatistics(
    layerId: string,
    gradients: Float32Array
  ): GradientStatistics {
    let sum = 0;
    let sumSq = 0;
    let positiveCount = 0;

    for (const g of gradients) {
      sum += g;
      sumSq += g * g;
      if (g > 0) positiveCount++;
    }

    const n = gradients.length;
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const std = Math.sqrt(Math.max(0, variance));
    const norm = Math.sqrt(sumSq);
    const signAgreement = Math.abs(positiveCount / n - 0.5) * 2; // 0 = balanced, 1 = all same sign

    const vanishing = norm < 1e-7;
    const exploding = norm > 1e3 || !isFinite(norm);

    return {
      layerId,
      mean,
      std,
      norm,
      vanishing,
      exploding,
      signAgreement,
    };
  }

  /**
   * Get gradient flow color based on health
   */
  private getGradientFlowColor(stats: GradientStatistics): string {
    if (stats.exploding) return "#ff4444";
    if (stats.vanishing) return "#4444ff";

    // Healthy - color by magnitude
    const normalizedMag = Math.min(1, stats.norm / 10);
    const r = Math.floor(normalizedMag * 255);
    const g = Math.floor((1 - normalizedMag) * 255);
    return `rgb(${r}, ${g}, 100)`;
  }

  /**
   * Capture activations snapshot
   */
  private captureActivations(state: ModelStateForCapture): ActivationsSnapshot {
    const statistics = new Map<string, ActivationStatistics>();
    const reducedPatterns: ActivationPatterns["reducedPatterns"] = [];

    for (const [layerId, layerState] of state.layers) {
      const acts = layerState.activations;

      statistics.set(layerId, {
        layerId,
        mean: this.computeMean(acts),
        std: this.computeStd(acts),
        percentActive: this.computePercentActive(acts),
        distribution: this.computeHistogram(acts, 50),
      });

      // Reduce to 3D for visualization (simple PCA projection)
      const position = this.reduceActivationTo3D(acts);
      reducedPatterns.push({
        layerId,
        pattern: acts.slice(0, Math.min(100, acts.length)), // Keep first 100 for detail
        position,
      });
    }

    return {
      statistics,
      patterns: { reducedPatterns },
    };
  }

  /**
   * Reduce activation pattern to 3D position
   */
  private reduceActivationTo3D(activations: Float32Array): Vector3D {
    // Simple projection using first 3 principal components approximation
    let x = 0,
      y = 0,
      z = 0;
    const n = activations.length;

    for (let i = 0; i < n; i++) {
      const val = activations[i];
      x += val * Math.cos(i * 0.1);
      y += val * Math.sin(i * 0.1);
      z += val * Math.cos(i * 0.05);
    }

    return {
      x: (x / n) * 10,
      y: (y / n) * 10,
      z: (z / n) * 10,
    };
  }

  /**
   * Capture optimizer state
   */
  private captureOptimizerState(
    state: ModelStateForCapture
  ): OptimizerStateSnapshot {
    return {
      type: state.optimizerType,
      stepCount: state.stepCount,
      // Momentum/velocity capture would be implemented based on optimizer type
    };
  }

  /**
   * Sample loss landscape around current point
   */
  private sampleLossLandscape(
    state: ModelStateForCapture
  ): LossLandscapeSample {
    const resolution = this.config.landscapeSampleResolution;
    const range = 1.0; // Range in weight space

    // Get random orthogonal directions for 2D projection
    const direction1 = this.getRandomDirection(state.totalParameters);
    const direction2 = this.getOrthogonalDirection(direction1);

    // Sample grid of loss values
    const grid = new Float32Array(resolution * resolution);

    // In practice, this would evaluate the loss at each grid point
    // For now, we create a synthetic landscape based on current loss
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = (i / resolution - 0.5) * 2 * range;
        const y = (j / resolution - 0.5) * 2 * range;

        // Synthetic bowl-shaped landscape centered on current loss
        const distance = Math.sqrt(x * x + y * y);
        grid[i * resolution + j] = state.loss + distance * state.loss * 0.5;
      }
    }

    // Estimate curvature
    const curvature = this.estimateCurvature(grid, resolution);

    return {
      currentPosition: { x: 0, y: state.loss, z: 0 },
      curvature,
      neighborhood: {
        grid,
        gridSize: resolution,
        range,
        direction1,
        direction2,
      },
      trajectoryPoint: this.computeTrajectoryPoint(state),
    };
  }

  /**
   * Get random unit direction
   */
  private getRandomDirection(dimensions: number): Float32Array {
    const dir = new Float32Array(Math.min(dimensions, 100));
    let norm = 0;

    for (let i = 0; i < dir.length; i++) {
      dir[i] = Math.random() * 2 - 1;
      norm += dir[i] * dir[i];
    }

    norm = Math.sqrt(norm);
    for (let i = 0; i < dir.length; i++) {
      dir[i] /= norm;
    }

    return dir;
  }

  /**
   * Get orthogonal direction via Gram-Schmidt
   */
  private getOrthogonalDirection(dir1: Float32Array): Float32Array {
    const dir2 = new Float32Array(dir1.length);

    // Start with random
    let dot = 0;
    for (let i = 0; i < dir2.length; i++) {
      dir2[i] = Math.random() * 2 - 1;
      dot += dir2[i] * dir1[i];
    }

    // Subtract projection onto dir1
    for (let i = 0; i < dir2.length; i++) {
      dir2[i] -= dot * dir1[i];
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < dir2.length; i++) {
      norm += dir2[i] * dir2[i];
    }
    norm = Math.sqrt(norm);

    for (let i = 0; i < dir2.length; i++) {
      dir2[i] /= norm;
    }

    return dir2;
  }

  /**
   * Estimate curvature from loss grid
   */
  private estimateCurvature(
    grid: Float32Array,
    resolution: number
  ): LandscapeCurvature {
    const center = Math.floor(resolution / 2);
    const centerLoss = grid[center * resolution + center];

    // Compute second derivatives
    const h = 1.0 / resolution;

    const d2x =
      (grid[center * resolution + center + 1] +
        grid[center * resolution + center - 1] -
        2 * centerLoss) /
      (h * h);

    const d2y =
      (grid[(center + 1) * resolution + center] +
        grid[(center - 1) * resolution + center] -
        2 * centerLoss) /
      (h * h);

    const dxy =
      (grid[(center + 1) * resolution + center + 1] +
        grid[(center - 1) * resolution + center - 1] -
        grid[(center + 1) * resolution + center - 1] -
        grid[(center - 1) * resolution + center + 1]) /
      (4 * h * h);

    // Eigenvalues of Hessian
    const trace = d2x + d2y;
    const det = d2x * d2y - dxy * dxy;
    const discriminant = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));

    const lambda1 = trace / 2 + discriminant;
    const lambda2 = trace / 2 - discriminant;

    const conditionNumber = Math.abs(lambda1) / (Math.abs(lambda2) + 1e-10);
    const sharpness = Math.max(Math.abs(lambda1), Math.abs(lambda2));

    return {
      principalCurvatures: [lambda1, lambda2],
      conditionNumber,
      sharpness,
    };
  }

  /**
   * Compute trajectory point in reduced space
   */
  private computeTrajectoryPoint(state: ModelStateForCapture): Vector3D {
    // Use loss and gradient norm to position in 3D
    return {
      x: this.trajectoryPath.length * 0.1,
      y: state.loss,
      z: Math.log(state.globalGradientNorm + 1),
    };
  }

  /**
   * Create empty loss landscape
   */
  private createEmptyLossLandscape(): LossLandscapeSample {
    return {
      currentPosition: { x: 0, y: 0, z: 0 },
      curvature: {
        principalCurvatures: [0, 0],
        conditionNumber: 1,
        sharpness: 0,
      },
      neighborhood: {
        grid: new Float32Array(0),
        gridSize: 0,
        range: 0,
        direction1: new Float32Array(0),
        direction2: new Float32Array(0),
      },
      trajectoryPoint: { x: 0, y: 0, z: 0 },
    };
  }

  /**
   * Compute delta from previous snapshot
   */
  private computeDelta(
    previous: TrainingSnapshot,
    current: ModelStateForCapture
  ): SnapshotDelta {
    const weightDeltas = new Map<string, Float32Array>();
    const changedLayers: string[] = [];

    for (const [layerId, layerState] of current.layers) {
      const prevStats = previous.weights.statistics.get(layerId);
      if (!prevStats) continue;

      // Check if layer changed significantly
      const currentStats = this.computeWeightStatistics(
        layerId,
        layerState.weights
      );
      const change =
        Math.abs(currentStats.mean - prevStats.mean) +
        Math.abs(currentStats.std - prevStats.std);

      if (change > 1e-6) {
        changedLayers.push(layerId);
      }
    }

    return {
      previousSnapshotId: previous.id,
      weightDeltas,
      changedLayers,
    };
  }

  /**
   * Compress weights
   */
  private compressWeights(
    state: ModelStateForCapture,
    level: "light" | "heavy"
  ): CompressedWeights {
    // Implement actual compression (SVD, quantization, etc.)
    return {
      method: level === "light" ? "quantized" : "svd",
      data: new ArrayBuffer(0),
      metadata: { compressionRatio: level === "light" ? 2 : 10 },
    };
  }

  /**
   * Add snapshot to collection
   */
  private addSnapshot(snapshot: TrainingSnapshot): void {
    this.snapshots.push(snapshot);
    this.snapshotIndex.set(snapshot.id, this.snapshots.length - 1);

    // Manage memory
    if (this.snapshots.length > this.config.maxSnapshotsInMemory) {
      this.evictOldSnapshots();
    }
  }

  /**
   * Evict old snapshots to manage memory
   */
  private evictOldSnapshots(): void {
    // Keep key snapshots, evict others
    const toRemove = Math.floor(this.snapshots.length * 0.2);
    let removed = 0;

    for (let i = 0; i < this.snapshots.length && removed < toRemove; i++) {
      const snapshot = this.snapshots[i];
      // Don't remove key snapshots or recent ones
      if (snapshot.weights.fullWeights || i > this.snapshots.length - 100) {
        continue;
      }

      // Remove full data, keep statistics
      snapshot.activations.patterns.reducedPatterns = [];
      removed++;
    }

    this.emit("snapshots:evicted", removed);
  }

  /**
   * Update trajectory path
   */
  private updateTrajectory(snapshot: TrainingSnapshot): void {
    this.trajectoryPath.push(snapshot.lossLandscape.trajectoryPoint);

    // Apply smoothing
    if (
      this.trajectoryPath.length >= 3 &&
      this.config.trajectorySmoothing > 0
    ) {
      const i = this.trajectoryPath.length - 2;
      const alpha = this.config.trajectorySmoothing;

      this.trajectoryPath[i] = {
        x:
          (alpha *
            (this.trajectoryPath[i - 1].x + this.trajectoryPath[i + 1].x)) /
            2 +
          (1 - alpha) * this.trajectoryPath[i].x,
        y:
          (alpha *
            (this.trajectoryPath[i - 1].y + this.trajectoryPath[i + 1].y)) /
            2 +
          (1 - alpha) * this.trajectoryPath[i].y,
        z:
          (alpha *
            (this.trajectoryPath[i - 1].z + this.trajectoryPath[i + 1].z)) /
            2 +
          (1 - alpha) * this.trajectoryPath[i].z,
      };
    }
  }

  /**
   * Check for anomalies
   */
  private checkForAnomalies(snapshot: TrainingSnapshot): void {
    const anomalies: TrainingAnomaly[] = [];

    // Check for NaN
    if (isNaN(snapshot.metrics.loss)) {
      anomalies.push({
        snapshotId: snapshot.id,
        type: "nan_detected",
        severity: "critical",
        description: "NaN loss detected",
        suggestedAction: "Reduce learning rate or check data",
      });
    }

    // Check gradient health
    if (snapshot.gradients.flowVisualization.overallHealth === "exploding") {
      anomalies.push({
        snapshotId: snapshot.id,
        type: "gradient_explosion",
        severity: "error",
        description: "Gradient explosion detected",
        suggestedAction: "Apply gradient clipping or reduce learning rate",
      });
    }

    if (snapshot.gradients.flowVisualization.overallHealth === "vanishing") {
      anomalies.push({
        snapshotId: snapshot.id,
        type: "gradient_vanishing",
        severity: "warning",
        description: "Vanishing gradients detected",
        suggestedAction: "Use residual connections or different activation",
      });
    }

    // Check for loss spike
    if (this.snapshots.length > 1) {
      const prevLoss = this.snapshots[this.snapshots.length - 2].metrics.loss;
      if (snapshot.metrics.loss > prevLoss * 2) {
        anomalies.push({
          snapshotId: snapshot.id,
          type: "loss_spike",
          severity: "warning",
          description: `Loss spiked from ${prevLoss.toFixed(4)} to ${snapshot.metrics.loss.toFixed(4)}`,
          suggestedAction: "Check for bad batch or reduce learning rate",
        });
      }
    }

    for (const anomaly of anomalies) {
      this.emit("anomaly:detected", anomaly);
    }
  }

  // ============================================================================
  // Time Navigation
  // ============================================================================

  /**
   * Navigate to specific snapshot
   */
  navigateToSnapshot(snapshotId: string): void {
    const index = this.snapshotIndex.get(snapshotId);
    if (index === undefined) return;

    this.timeNavigation.currentSnapshotIndex = index;
    this.emit("navigation:changed", this.getCurrentSnapshot());
  }

  /**
   * Navigate to epoch
   */
  navigateToEpoch(epoch: number): void {
    const snapshot = this.snapshots.find(
      (s) => s.epoch === epoch && s.batch === 0
    );
    if (snapshot) {
      this.navigateToSnapshot(snapshot.id);
    }
  }

  /**
   * Navigate by offset
   */
  navigateByOffset(offset: number): void {
    const newIndex = Math.max(
      0,
      Math.min(
        this.snapshots.length - 1,
        this.timeNavigation.currentSnapshotIndex + offset
      )
    );

    this.timeNavigation.currentSnapshotIndex = newIndex;
    this.emit("navigation:changed", this.getCurrentSnapshot());
  }

  /**
   * Start playback
   */
  startPlayback(direction: "forward" | "backward" = "forward"): void {
    this.timeNavigation.isPlaying = true;
    this.timeNavigation.playDirection = direction;
    this.startPlaybackLoop();
    this.emit("playback:started", direction);
  }

  /**
   * Stop playback
   */
  stopPlayback(): void {
    this.timeNavigation.isPlaying = false;
    this.stopPlaybackLoop();
    this.emit("playback:stopped");
  }

  /**
   * Set playback speed
   */
  setPlaybackSpeed(speed: number): void {
    this.timeNavigation.playbackSpeed = Math.max(0.1, Math.min(10, speed));
    this.emit("playback:speedChanged", this.timeNavigation.playbackSpeed);
  }

  /**
   * Set loop mode
   */
  setLoopMode(mode: "none" | "loop" | "pingpong"): void {
    this.timeNavigation.loopMode = mode;
  }

  /**
   * Start playback animation loop
   */
  private startPlaybackLoop(): void {
    if (this.animationFrame !== null) return;

    const frameInterval = 1000 / (30 * this.timeNavigation.playbackSpeed);
    let lastFrameTime = Date.now();

    const animate = () => {
      if (!this.timeNavigation.isPlaying) return;

      const now = Date.now();
      if (now - lastFrameTime >= frameInterval) {
        this.advancePlayback();
        lastFrameTime = now;
      }

      this.animationFrame = requestAnimationFrame(animate);
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Stop playback loop
   */
  private stopPlaybackLoop(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Advance playback by one frame
   */
  private advancePlayback(): void {
    const direction = this.timeNavigation.playDirection === "forward" ? 1 : -1;
    const newIndex = this.timeNavigation.currentSnapshotIndex + direction;

    // Handle bounds
    if (newIndex < 0) {
      if (this.timeNavigation.loopMode === "loop") {
        this.timeNavigation.currentSnapshotIndex = this.snapshots.length - 1;
      } else if (this.timeNavigation.loopMode === "pingpong") {
        this.timeNavigation.playDirection = "forward";
        this.timeNavigation.currentSnapshotIndex = 0;
      } else {
        this.stopPlayback();
        return;
      }
    } else if (newIndex >= this.snapshots.length) {
      if (this.timeNavigation.loopMode === "loop") {
        this.timeNavigation.currentSnapshotIndex = 0;
      } else if (this.timeNavigation.loopMode === "pingpong") {
        this.timeNavigation.playDirection = "backward";
        this.timeNavigation.currentSnapshotIndex = this.snapshots.length - 1;
      } else {
        this.stopPlayback();
        return;
      }
    } else {
      this.timeNavigation.currentSnapshotIndex = newIndex;
    }

    this.emit("navigation:changed", this.getCurrentSnapshot());
  }

  /**
   * Get current snapshot
   */
  getCurrentSnapshot(): TrainingSnapshot | null {
    if (this.snapshots.length === 0) return null;
    return this.snapshots[this.timeNavigation.currentSnapshotIndex];
  }

  /**
   * Get snapshot at index
   */
  getSnapshotAt(index: number): TrainingSnapshot | null {
    return this.snapshots[index] || null;
  }

  /**
   * Get snapshot by ID
   */
  getSnapshotById(id: string): TrainingSnapshot | null {
    const index = this.snapshotIndex.get(id);
    return index !== undefined ? this.snapshots[index] : null;
  }

  /**
   * Get all snapshots
   */
  getAllSnapshots(): TrainingSnapshot[] {
    return [...this.snapshots];
  }

  /**
   * Get trajectory path
   */
  getTrajectoryPath(): Vector3D[] {
    return [...this.trajectoryPath];
  }

  // ============================================================================
  // Visualization
  // ============================================================================

  /**
   * Set visualization mode
   */
  setVisualizationMode(mode: JourneyVisualizationMode): void {
    this.visualizationMode = mode;
    this.emit("visualization:changed", mode);
  }

  /**
   * Get visualization mode
   */
  getVisualizationMode(): JourneyVisualizationMode {
    return this.visualizationMode;
  }

  /**
   * Set camera
   */
  setCamera(camera: Partial<Journey4DCamera>): void {
    Object.assign(this.camera, camera);
    this.emit("camera:changed", this.camera);
  }

  /**
   * Get camera
   */
  getCamera(): Journey4DCamera {
    return { ...this.camera };
  }

  /**
   * Build loss landscape mesh for 3D rendering
   */
  buildLossLandscapeMesh(): LossLandscapeMesh {
    if (this.snapshots.length === 0) {
      return { vertices: [], indices: [], colors: [] };
    }

    const snapshot = this.getCurrentSnapshot()!;
    const { grid, gridSize, range } = snapshot.lossLandscape.neighborhood;

    const vertices: number[] = [];
    const indices: number[] = [];
    const colors: number[] = [];

    // Generate mesh vertices
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = (i / gridSize - 0.5) * 2 * range * 10;
        const z = (j / gridSize - 0.5) * 2 * range * 10;
        const y = grid[i * gridSize + j] * 2;

        vertices.push(x, y, z);

        // Color by height (loss)
        const normalizedLoss = Math.min(1, grid[i * gridSize + j] / 10);
        colors.push(normalizedLoss, 0.2, 1 - normalizedLoss);
      }
    }

    // Generate indices
    for (let i = 0; i < gridSize - 1; i++) {
      for (let j = 0; j < gridSize - 1; j++) {
        const tl = i * gridSize + j;
        const tr = tl + 1;
        const bl = tl + gridSize;
        const br = bl + 1;

        indices.push(tl, bl, tr);
        indices.push(tr, bl, br);
      }
    }

    this.lossLandscapeMesh = { vertices, indices, colors };
    return this.lossLandscapeMesh;
  }

  // ============================================================================
  // Analysis
  // ============================================================================

  /**
   * Analyze the training journey
   */
  analyzeJourney(): JourneyAnalysis {
    const criticalPoints = this.findCriticalPoints();
    const phases = this.identifyPhases();
    const anomalies = this.collectAnomalies();
    const trajectoryAnalysis = this.analyzeTrajectory();

    this.analysis = {
      criticalPoints,
      phases,
      anomalies,
      trajectoryAnalysis,
    };

    this.emit("analysis:complete", this.analysis);
    return this.analysis;
  }

  /**
   * Find critical points in training
   */
  private findCriticalPoints(): CriticalPoint[] {
    const criticalPoints: CriticalPoint[] = [];

    if (this.snapshots.length < 10) return criticalPoints;

    const windowSize = 5;

    for (let i = windowSize; i < this.snapshots.length - windowSize; i++) {
      const before = this.snapshots.slice(i - windowSize, i);
      const after = this.snapshots.slice(i, i + windowSize);
      const current = this.snapshots[i];

      const avgLossBefore =
        before.reduce((s, p) => s + p.metrics.loss, 0) / before.length;
      const avgLossAfter =
        after.reduce((s, p) => s + p.metrics.loss, 0) / after.length;
      const lossChange = (avgLossAfter - avgLossBefore) / avgLossBefore;

      // Plateau detection
      const variance = this.computeVariance(
        before.concat(after).map((s) => s.metrics.loss)
      );
      if (variance < 0.001 && Math.abs(lossChange) < 0.01) {
        criticalPoints.push({
          snapshotId: current.id,
          type: "plateau",
          epoch: current.epoch,
          description: `Training plateau at epoch ${current.epoch}`,
          importance: 0.6,
        });
      }

      // Breakthrough detection
      if (lossChange < -0.2) {
        criticalPoints.push({
          snapshotId: current.id,
          type: "breakthrough",
          epoch: current.epoch,
          description: `Breakthrough at epoch ${current.epoch} - ${Math.abs(lossChange * 100).toFixed(1)}% loss reduction`,
          importance: 0.9,
        });
      }

      // Degradation detection
      if (lossChange > 0.1) {
        criticalPoints.push({
          snapshotId: current.id,
          type: "degradation",
          epoch: current.epoch,
          description: `Training degradation at epoch ${current.epoch}`,
          importance: 0.7,
        });
      }
    }

    // Check convergence at end
    if (this.snapshots.length >= 10) {
      const lastSnapshots = this.snapshots.slice(-10);
      const variance = this.computeVariance(
        lastSnapshots.map((s) => s.metrics.loss)
      );

      if (variance < 0.0001) {
        const last = lastSnapshots[lastSnapshots.length - 1];
        criticalPoints.push({
          snapshotId: last.id,
          type: "convergence",
          epoch: last.epoch,
          description: "Training converged",
          importance: 1.0,
        });
      }
    }

    return criticalPoints.sort((a, b) => b.importance - a.importance);
  }

  /**
   * Identify training phases
   */
  private identifyPhases(): TrainingPhase[] {
    const phases: TrainingPhase[] = [];

    if (this.snapshots.length < 20) {
      return [
        {
          name: "Initial Training",
          startEpoch: 0,
          endEpoch:
            this.snapshots.length > 0
              ? this.snapshots[this.snapshots.length - 1].epoch
              : 0,
          characteristics: ["Learning in progress"],
          dominantDynamics: "gradient descent",
        },
      ];
    }

    // Simple phase detection based on loss derivative changes
    let phaseStart = 0;
    let currentPhaseType = "warmup";

    for (let i = 1; i < this.snapshots.length; i++) {
      const prevLoss = this.snapshots[i - 1].metrics.loss;
      const currLoss = this.snapshots[i].metrics.loss;
      const lossRatio = currLoss / prevLoss;

      let newPhaseType = currentPhaseType;

      if (lossRatio > 1.1) {
        newPhaseType = "instability";
      } else if (lossRatio > 0.99 && lossRatio < 1.01) {
        newPhaseType = "plateau";
      } else if (lossRatio < 0.95) {
        newPhaseType = "rapid_learning";
      } else {
        newPhaseType = "fine_tuning";
      }

      if (
        newPhaseType !== currentPhaseType ||
        i === this.snapshots.length - 1
      ) {
        phases.push({
          name: this.getPhaseDisplayName(currentPhaseType),
          startEpoch: this.snapshots[phaseStart].epoch,
          endEpoch: this.snapshots[i - 1].epoch,
          characteristics: this.getPhaseCharacteristics(currentPhaseType),
          dominantDynamics: currentPhaseType,
        });

        phaseStart = i;
        currentPhaseType = newPhaseType;
      }
    }

    return phases;
  }

  /**
   * Get phase display name
   */
  private getPhaseDisplayName(type: string): string {
    const names: Record<string, string> = {
      warmup: "Warmup Phase",
      rapid_learning: "Rapid Learning",
      plateau: "Learning Plateau",
      fine_tuning: "Fine-Tuning",
      instability: "Training Instability",
    };
    return names[type] || type;
  }

  /**
   * Get phase characteristics
   */
  private getPhaseCharacteristics(type: string): string[] {
    const chars: Record<string, string[]> = {
      warmup: ["Initial weight adjustments", "High learning rate sensitivity"],
      rapid_learning: [
        "Fast loss decrease",
        "Feature learning",
        "High gradient flow",
      ],
      plateau: [
        "Stable loss",
        "Potential local minimum",
        "Consider LR adjustment",
      ],
      fine_tuning: ["Gradual improvements", "Weight refinement"],
      instability: [
        "Loss fluctuations",
        "Potential overfitting",
        "Check learning rate",
      ],
    };
    return chars[type] || [];
  }

  /**
   * Collect all anomalies
   */
  private collectAnomalies(): TrainingAnomaly[] {
    // Would collect from anomaly events
    return [];
  }

  /**
   * Analyze trajectory
   */
  private analyzeTrajectory(): TrajectoryAnalysis {
    if (this.trajectoryPath.length < 2) {
      return {
        totalDistance: 0,
        averageSpeed: 0,
        convergenceRate: 0,
        oscillationScore: 0,
        directionalConsistency: 0,
      };
    }

    let totalDistance = 0;
    const directions: Vector3D[] = [];

    for (let i = 1; i < this.trajectoryPath.length; i++) {
      const prev = this.trajectoryPath[i - 1];
      const curr = this.trajectoryPath[i];

      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      const dz = curr.z - prev.z;

      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      totalDistance += distance;

      if (distance > 0) {
        directions.push({
          x: dx / distance,
          y: dy / distance,
          z: dz / distance,
        });
      }
    }

    const averageSpeed = totalDistance / this.trajectoryPath.length;

    // Compute convergence rate (how fast loss decreases)
    const startLoss = this.trajectoryPath[0].y;
    const endLoss = this.trajectoryPath[this.trajectoryPath.length - 1].y;
    const convergenceRate = (startLoss - endLoss) / this.trajectoryPath.length;

    // Compute oscillation (direction changes)
    let directionChanges = 0;
    for (let i = 1; i < directions.length; i++) {
      const dot =
        directions[i].x * directions[i - 1].x +
        directions[i].y * directions[i - 1].y +
        directions[i].z * directions[i - 1].z;
      if (dot < 0) directionChanges++;
    }
    const oscillationScore = directionChanges / Math.max(1, directions.length);

    // Directional consistency
    let avgDir = { x: 0, y: 0, z: 0 };
    for (const d of directions) {
      avgDir.x += d.x;
      avgDir.y += d.y;
      avgDir.z += d.z;
    }
    const n = directions.length;
    const avgMag = Math.sqrt(
      (avgDir.x / n) ** 2 + (avgDir.y / n) ** 2 + (avgDir.z / n) ** 2
    );
    const directionalConsistency = avgMag;

    return {
      totalDistance,
      averageSpeed,
      convergenceRate,
      oscillationScore,
      directionalConsistency,
    };
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Compute norm of array
   */
  private computeNorm(arr: Float32Array): number {
    let sumSq = 0;
    for (const v of arr) sumSq += v * v;
    return Math.sqrt(sumSq);
  }

  /**
   * Compute mean
   */
  private computeMean(arr: Float32Array): number {
    let sum = 0;
    for (const v of arr) sum += v;
    return sum / arr.length;
  }

  /**
   * Compute standard deviation
   */
  private computeStd(arr: Float32Array): number {
    const mean = this.computeMean(arr);
    let sumSq = 0;
    for (const v of arr) sumSq += (v - mean) ** 2;
    return Math.sqrt(sumSq / arr.length);
  }

  /**
   * Compute sparsity
   */
  private computeSparsity(arr: Float32Array): number {
    let zeroCount = 0;
    for (const v of arr) {
      if (Math.abs(v) < 1e-7) zeroCount++;
    }
    return zeroCount / arr.length;
  }

  /**
   * Compute percent active (for ReLU)
   */
  private computePercentActive(arr: Float32Array): number {
    let activeCount = 0;
    for (const v of arr) {
      if (v > 0) activeCount++;
    }
    return activeCount / arr.length;
  }

  /**
   * Compute histogram
   */
  private computeHistogram(arr: Float32Array, bins: number): number[] {
    if (arr.length === 0) return new Array(bins).fill(0);

    let min = Infinity,
      max = -Infinity;
    for (const v of arr) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }

    const histogram = new Array(bins).fill(0);
    const binWidth = (max - min) / bins || 1;

    for (const v of arr) {
      const bin = Math.min(bins - 1, Math.floor((v - min) / binWidth));
      histogram[bin]++;
    }

    return histogram;
  }

  /**
   * Compute variance
   */
  private computeVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  }

  /**
   * Get snapshot count
   */
  getSnapshotCount(): number {
    return this.snapshots.length;
  }

  /**
   * Get time navigation state
   */
  getNavigationState(): TimeNavigationState {
    return { ...this.timeNavigation };
  }

  /**
   * Get analysis
   */
  getAnalysis(): JourneyAnalysis | null {
    return this.analysis;
  }

  /**
   * Export journey data
   */
  exportJourney(): string {
    return JSON.stringify(
      {
        config: this.config,
        snapshots: this.snapshots.map((s) => ({
          id: s.id,
          epoch: s.epoch,
          batch: s.batch,
          metrics: {
            loss: s.metrics.loss,
            accuracy: s.metrics.accuracy,
            learningRate: s.metrics.learningRate,
          },
        })),
        trajectoryPath: this.trajectoryPath,
        analysis: this.analysis,
      },
      null,
      2
    );
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.stopPlayback();
    this.snapshots = [];
    this.snapshotIndex.clear();
    this.trajectoryPath = [];
    this.removeAllListeners();
  }
}

// ============================================================================
// Supporting Types
// ============================================================================

/**
 * Model state for capture (would be provided by ML framework integration)
 */
export interface ModelStateForCapture {
  loss: number;
  accuracy: number;
  learningRate: number;
  globalGradientNorm: number;
  validationLoss?: number;
  validationAccuracy?: number;
  optimizerType: string;
  stepCount: number;
  totalParameters: number;
  customMetrics?: Record<string, number>;
  layers: Map<string, LayerStateForCapture>;
}

/**
 * Layer state for capture
 */
export interface LayerStateForCapture {
  weights: Float32Array;
  gradients: Float32Array;
  activations: Float32Array;
}

/**
 * Loss landscape mesh
 */
export interface LossLandscapeMesh {
  vertices: number[];
  indices: number[];
  colors: number[];
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create Training 4D Journey
 */
export function createTraining4DJourney(
  config?: Partial<Journey4DConfig>
): Training4DJourney {
  return new Training4DJourney(config);
}
