/**
 * Predictive Visualization Cascade
 *
 * Breakthrough Innovation: Show predicted states visually before they occur.
 * Creates predictive overlays in 3D view showing future agent states.
 *
 * This innovation combines:
 * - Twin's prediction engine capabilities
 * - Forge's 3D visualization rendering
 * - Timeline's temporal awareness
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations/predictive-cascade
 * @agents @NEXUS @ORACLE @CANVAS
 * @innovation Predictive Visualization Cascade (Forge × Twin)
 */

import type {
  UnifiedEntity,
  UnifiedEvent,
  UniversalId,
  Timestamp,
  Domain,
} from "../types";

import { CrossDomainEventBridge } from "../event-bridge";
import { CrossDomainOrchestrator } from "../orchestrator";

// ============================================================================
// Predictive Cascade Types
// ============================================================================

/**
 * Prediction with visualization data
 */
export interface PredictiveState {
  /** Unique prediction ID */
  id: string;

  /** Entity being predicted */
  entityId: UniversalId;

  /** Predicted future time */
  predictedTime: Timestamp;

  /** When prediction was made */
  generatedAt: Timestamp;

  /** Confidence level (0-1) */
  confidence: number;

  /** Predicted state data */
  state: unknown;

  /** Visualization properties */
  visualization: PredictiveVisualization;

  /** Prediction metadata */
  metadata: PredictionMetadata;
}

/**
 * Visualization properties for prediction overlay
 */
export interface PredictiveVisualization {
  /** Position in 3D space */
  position: { x: number; y: number; z: number };

  /** Rotation */
  rotation: { x: number; y: number; z: number };

  /** Scale */
  scale: { x: number; y: number; z: number };

  /** Opacity based on confidence */
  opacity: number;

  /** Color with prediction styling */
  color: {
    primary: { r: number; g: number; b: number };
    glow: { r: number; g: number; b: number };
    outline: { r: number; g: number; b: number };
  };

  /** Animation properties */
  animation: {
    /** Pulsing frequency */
    pulseFrequency: number;
    /** Trail effect length */
    trailLength: number;
    /** Particle density */
    particleDensity: number;
  };

  /** Connection lines to current state */
  connectionToPresent: {
    show: boolean;
    style: "solid" | "dashed" | "dotted" | "gradient";
    width: number;
    color: { r: number; g: number; b: number; a: number };
  };

  /** Uncertainty visualization */
  uncertaintyCloud: {
    show: boolean;
    radius: number;
    density: number;
    color: { r: number; g: number; b: number; a: number };
  };
}

/**
 * Prediction metadata
 */
export interface PredictionMetadata {
  /** Model used for prediction */
  modelId: string;

  /** Prediction horizon */
  horizonMs: number;

  /** Input features used */
  inputFeatures: string[];

  /** Uncertainty bounds */
  uncertainty: {
    lower: number;
    upper: number;
    stdDev: number;
  };

  /** Alternative predictions */
  alternatives: Array<{
    confidence: number;
    state: unknown;
  }>;

  /** Tags for filtering */
  tags: string[];
}

/**
 * Cascade configuration
 */
export interface PredictiveCascadeConfig {
  /** Entity IDs to generate predictions for */
  entityIds: UniversalId[];

  /** Prediction horizons in ms */
  horizons: number[];

  /** Update interval in ms */
  updateInterval: number;

  /** Minimum confidence to display */
  minConfidence: number;

  /** Maximum predictions to show per entity */
  maxPredictionsPerEntity: number;

  /** Visualization mode */
  visualizationMode: "ghost" | "trail" | "cloud" | "timeline";

  /** Show uncertainty */
  showUncertainty: boolean;

  /** Show connections to present */
  showConnections: boolean;

  /** Auto-refresh predictions */
  autoRefresh: boolean;
}

/**
 * Cascade state
 */
export interface CascadeState {
  id: string;
  config: PredictiveCascadeConfig;
  status: "active" | "paused" | "stopped";
  predictions: Map<UniversalId, PredictiveState[]>;
  lastUpdate: Timestamp;
  statistics: CascadeStatistics;
}

/**
 * Cascade statistics
 */
export interface CascadeStatistics {
  totalPredictions: number;
  averageConfidence: number;
  accuracyScore: number;
  predictionLatency: number;
  entityCoverage: number;
}

// ============================================================================
// Event Types
// ============================================================================

export type CascadeEventType =
  | "cascade:created"
  | "cascade:started"
  | "cascade:paused"
  | "cascade:stopped"
  | "cascade:prediction:generated"
  | "cascade:prediction:expired"
  | "cascade:prediction:verified"
  | "cascade:accuracy:updated";

export interface CascadeEvent {
  type: CascadeEventType;
  cascadeId: string;
  timestamp: Timestamp;
  data: unknown;
}

// ============================================================================
// Predictive Visualization Cascade Class
// ============================================================================

/**
 * Predictive Visualization Cascade
 *
 * Generates and visualizes predictions for entity future states.
 */
export class PredictiveVisualizationCascade {
  private static instance: PredictiveVisualizationCascade | null = null;

  private cascades: Map<string, CascadeState> = new Map();
  private updateTimers: Map<string, ReturnType<typeof setInterval>> = new Map();
  private eventBridge: CrossDomainEventBridge;
  private orchestrator: CrossDomainOrchestrator;

  // Prediction history for accuracy tracking
  private predictionHistory: Map<
    string,
    Array<{
      predicted: PredictiveState;
      actual: unknown;
      timestamp: Timestamp;
      accurate: boolean;
    }>
  > = new Map();

  // Listeners
  private listeners: Map<string, Set<(event: CascadeEvent) => void>> =
    new Map();

  private constructor() {
    this.eventBridge = CrossDomainEventBridge.getInstance();
    this.orchestrator = CrossDomainOrchestrator.getInstance();

    this.setupEventListeners();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): PredictiveVisualizationCascade {
    if (!PredictiveVisualizationCascade.instance) {
      PredictiveVisualizationCascade.instance =
        new PredictiveVisualizationCascade();
    }
    return PredictiveVisualizationCascade.instance;
  }

  /**
   * Reset instance (for testing)
   */
  static resetInstance(): void {
    if (PredictiveVisualizationCascade.instance) {
      PredictiveVisualizationCascade.instance.dispose();
      PredictiveVisualizationCascade.instance = null;
    }
  }

  /**
   * Dispose cascade
   */
  dispose(): void {
    for (const cascadeId of this.cascades.keys()) {
      this.stopCascade(cascadeId);
    }

    this.cascades.clear();
    this.predictionHistory.clear();
    this.listeners.clear();
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  private setupEventListeners(): void {
    // Listen for entity state changes to verify predictions
    this.eventBridge.subscribe<UnifiedEntity>("entity:updated", (event) =>
      this.handleEntityUpdated(event)
    );

    // Listen for prediction completion from twin
    this.eventBridge.subscribe<{ entityId: string; predictions: unknown[] }>(
      "prediction:completed",
      (event) => this.handlePredictionCompleted(event)
    );
  }

  private handleEntityUpdated(event: UnifiedEvent<UnifiedEntity>): void {
    const entity = event.payload;
    if (!entity) return;

    // Check if any predictions need verification
    this.verifyPredictions(entity.id, entity.state);
  }

  private handlePredictionCompleted(
    event: UnifiedEvent<{ entityId: string; predictions: unknown[] }>
  ): void {
    const { entityId, predictions } = event.payload ?? {};
    if (!entityId || !predictions) return;

    // Process predictions for active cascades
    for (const [cascadeId, cascade] of this.cascades) {
      if (
        cascade.status === "active" &&
        cascade.config.entityIds.includes(entityId as UniversalId)
      ) {
        this.processPredictions(
          cascadeId,
          entityId as UniversalId,
          predictions
        );
      }
    }
  }

  // ==========================================================================
  // Cascade Management
  // ==========================================================================

  /**
   * Create a new prediction cascade
   */
  createCascade(config: PredictiveCascadeConfig): CascadeState {
    const cascadeId = `cascade-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    const state: CascadeState = {
      id: cascadeId,
      config,
      status: "paused",
      predictions: new Map(),
      lastUpdate: 0,
      statistics: {
        totalPredictions: 0,
        averageConfidence: 0,
        accuracyScore: 0,
        predictionLatency: 0,
        entityCoverage: 0,
      },
    };

    this.cascades.set(cascadeId, state);

    this.emitEvent({
      type: "cascade:created",
      cascadeId,
      timestamp: Date.now(),
      data: { config },
    });

    return state;
  }

  /**
   * Start a cascade
   */
  startCascade(cascadeId: string): void {
    const cascade = this.cascades.get(cascadeId);
    if (!cascade) return;

    cascade.status = "active";

    // Generate initial predictions
    this.refreshPredictions(cascadeId);

    // Start auto-refresh if enabled
    if (cascade.config.autoRefresh) {
      const timer = setInterval(
        () => this.refreshPredictions(cascadeId),
        cascade.config.updateInterval
      );
      this.updateTimers.set(cascadeId, timer);
    }

    this.emitEvent({
      type: "cascade:started",
      cascadeId,
      timestamp: Date.now(),
      data: {},
    });
  }

  /**
   * Pause a cascade
   */
  pauseCascade(cascadeId: string): void {
    const cascade = this.cascades.get(cascadeId);
    if (!cascade) return;

    cascade.status = "paused";

    // Stop auto-refresh
    const timer = this.updateTimers.get(cascadeId);
    if (timer) {
      clearInterval(timer);
      this.updateTimers.delete(cascadeId);
    }

    this.emitEvent({
      type: "cascade:paused",
      cascadeId,
      timestamp: Date.now(),
      data: {},
    });
  }

  /**
   * Stop and remove a cascade
   */
  stopCascade(cascadeId: string): void {
    const cascade = this.cascades.get(cascadeId);
    if (!cascade) return;

    cascade.status = "stopped";
    cascade.predictions.clear();

    // Stop auto-refresh
    const timer = this.updateTimers.get(cascadeId);
    if (timer) {
      clearInterval(timer);
      this.updateTimers.delete(cascadeId);
    }

    this.emitEvent({
      type: "cascade:stopped",
      cascadeId,
      timestamp: Date.now(),
      data: {},
    });

    this.cascades.delete(cascadeId);
  }

  /**
   * Get cascade state
   */
  getCascadeState(cascadeId: string): CascadeState | undefined {
    return this.cascades.get(cascadeId);
  }

  /**
   * Get all predictions for an entity
   */
  getPredictions(cascadeId: string, entityId: UniversalId): PredictiveState[] {
    const cascade = this.cascades.get(cascadeId);
    if (!cascade) return [];

    return cascade.predictions.get(entityId) ?? [];
  }

  /**
   * Get all active predictions across cascades
   */
  getAllActivePredictions(): Map<UniversalId, PredictiveState[]> {
    const allPredictions = new Map<UniversalId, PredictiveState[]>();

    for (const cascade of this.cascades.values()) {
      if (cascade.status !== "active") continue;

      for (const [entityId, predictions] of cascade.predictions) {
        if (!allPredictions.has(entityId)) {
          allPredictions.set(entityId, []);
        }
        allPredictions.get(entityId)!.push(...predictions);
      }
    }

    return allPredictions;
  }

  // ==========================================================================
  // Prediction Generation
  // ==========================================================================

  /**
   * Refresh predictions for a cascade
   */
  private async refreshPredictions(cascadeId: string): Promise<void> {
    const cascade = this.cascades.get(cascadeId);
    if (!cascade || cascade.status !== "active") return;

    const startTime = performance.now();

    for (const entityId of cascade.config.entityIds) {
      const entity = this.orchestrator.getEntity(entityId);
      if (!entity) continue;

      // Generate predictions for each horizon
      const predictions: PredictiveState[] = [];

      for (const horizon of cascade.config.horizons) {
        const prediction = await this.generatePrediction(
          entity,
          horizon,
          cascade.config
        );
        if (
          prediction &&
          prediction.confidence >= cascade.config.minConfidence
        ) {
          predictions.push(prediction);
        }
      }

      // Limit predictions per entity
      const limitedPredictions = predictions
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, cascade.config.maxPredictionsPerEntity);

      cascade.predictions.set(entityId, limitedPredictions);

      // Emit event for each prediction
      for (const prediction of limitedPredictions) {
        this.emitEvent({
          type: "cascade:prediction:generated",
          cascadeId,
          timestamp: Date.now(),
          data: { entityId, prediction },
        });
      }
    }

    // Update statistics
    cascade.lastUpdate = Date.now();
    cascade.statistics = this.calculateStatistics(cascade);
    cascade.statistics.predictionLatency = performance.now() - startTime;

    // Publish to event bridge for visualization
    this.eventBridge.publish({
      id: `cascade-update-${cascadeId}-${Date.now()}`,
      type: "metrics:updated",
      payload: {
        cascadeId,
        predictions: Object.fromEntries(cascade.predictions),
        statistics: cascade.statistics,
      },
      timestamp: Date.now(),
      sourceDomain: "twin",
      targetDomains: ["forge"],
    });
  }

  /**
   * Generate a single prediction
   */
  private async generatePrediction(
    entity: UnifiedEntity,
    horizonMs: number,
    config: PredictiveCascadeConfig
  ): Promise<PredictiveState | null> {
    const now = Date.now();
    const predictedTime = now + horizonMs;

    // Simulate prediction (in real implementation, would call prediction engine)
    const predictedState = this.simulateFutureState(entity, horizonMs);
    const confidence = this.calculateConfidence(entity, horizonMs);

    // Generate visualization
    const visualization = this.generateVisualization(
      entity,
      predictedState,
      confidence,
      horizonMs,
      config
    );

    return {
      id: `pred-${entity.id}-${predictedTime}`,
      entityId: entity.id,
      predictedTime,
      generatedAt: now,
      confidence,
      state: predictedState,
      visualization,
      metadata: {
        modelId: "default-predictor",
        horizonMs,
        inputFeatures: ["position", "state", "connections"],
        uncertainty: {
          lower: confidence * 0.8,
          upper: Math.min(1, confidence * 1.2),
          stdDev: (1 - confidence) * 0.5,
        },
        alternatives: [],
        tags: [],
      },
    };
  }

  /**
   * Simulate future state (placeholder for real prediction engine)
   */
  private simulateFutureState(
    entity: UnifiedEntity,
    horizonMs: number
  ): unknown {
    // Simple linear extrapolation of position
    const decay = Math.exp(-horizonMs / 60000); // Decay confidence over time

    return {
      ...entity.state,
      predicted: true,
      horizonMs,
      decay,
    };
  }

  /**
   * Calculate prediction confidence
   */
  private calculateConfidence(
    entity: UnifiedEntity,
    horizonMs: number
  ): number {
    // Confidence decreases with prediction horizon
    const baseConfidence = 0.9;
    const decay = Math.exp(-horizonMs / 30000);

    return baseConfidence * decay;
  }

  /**
   * Generate visualization properties
   */
  private generateVisualization(
    entity: UnifiedEntity,
    predictedState: unknown,
    confidence: number,
    horizonMs: number,
    config: PredictiveCascadeConfig
  ): PredictiveVisualization {
    const currentVis = entity.visualization;

    // Offset position based on horizon
    const timeOffset = horizonMs / 10000; // 10 seconds = 1 unit offset

    // Ghost color (blue-ish for future)
    const ghostColor = {
      r: 0.3,
      g: 0.6,
      b: 1.0,
    };

    return {
      position: {
        x: currentVis.position.x + timeOffset * 0.5,
        y: currentVis.position.y + timeOffset * 0.2,
        z: currentVis.position.z,
      },
      rotation: { x: 0, y: 0, z: 0 },
      scale: {
        x: currentVis.scale * (0.8 + confidence * 0.2),
        y: currentVis.scale * (0.8 + confidence * 0.2),
        z: currentVis.scale * (0.8 + confidence * 0.2),
      },
      opacity: confidence * 0.7,
      color: {
        primary: ghostColor,
        glow: {
          r: ghostColor.r * 0.5,
          g: ghostColor.g * 0.5,
          b: ghostColor.b * 0.5,
        },
        outline: {
          r: 1.0,
          g: 1.0,
          b: 1.0,
        },
      },
      animation: {
        pulseFrequency: 1 + (1 - confidence) * 2,
        trailLength:
          config.visualizationMode === "trail" ? horizonMs / 1000 : 0,
        particleDensity:
          config.visualizationMode === "cloud" ? 50 * confidence : 0,
      },
      connectionToPresent: {
        show: config.showConnections,
        style: "gradient",
        width: 2 * confidence,
        color: { ...ghostColor, a: confidence * 0.5 },
      },
      uncertaintyCloud: {
        show: config.showUncertainty,
        radius: (1 - confidence) * 2,
        density: 20,
        color: { r: 1, g: 0.8, b: 0, a: 0.3 },
      },
    };
  }

  // ==========================================================================
  // Prediction Verification
  // ==========================================================================

  /**
   * Verify predictions against actual state
   */
  private verifyPredictions(entityId: UniversalId, actualState: unknown): void {
    const now = Date.now();

    for (const [cascadeId, cascade] of this.cascades) {
      const predictions = cascade.predictions.get(entityId);
      if (!predictions) continue;

      // Check predictions that have reached their predicted time
      const toVerify = predictions.filter((p) => p.predictedTime <= now);
      const remaining = predictions.filter((p) => p.predictedTime > now);

      for (const prediction of toVerify) {
        const accurate = this.isPredictionAccurate(
          prediction.state,
          actualState
        );

        // Record for accuracy tracking
        if (!this.predictionHistory.has(cascadeId)) {
          this.predictionHistory.set(cascadeId, []);
        }
        this.predictionHistory.get(cascadeId)!.push({
          predicted: prediction,
          actual: actualState,
          timestamp: now,
          accurate,
        });

        this.emitEvent({
          type: "cascade:prediction:verified",
          cascadeId,
          timestamp: now,
          data: { prediction, actualState, accurate },
        });
      }

      // Update predictions list
      cascade.predictions.set(entityId, remaining);

      // Emit expired events
      for (const prediction of toVerify) {
        this.emitEvent({
          type: "cascade:prediction:expired",
          cascadeId,
          timestamp: now,
          data: { predictionId: prediction.id },
        });
      }

      // Update accuracy statistics
      cascade.statistics = this.calculateStatistics(cascade);

      this.emitEvent({
        type: "cascade:accuracy:updated",
        cascadeId,
        timestamp: now,
        data: { statistics: cascade.statistics },
      });
    }
  }

  /**
   * Check if prediction was accurate
   */
  private isPredictionAccurate(predicted: unknown, actual: unknown): boolean {
    // Simple comparison - in real implementation, would use domain-specific metrics
    try {
      const predStr = JSON.stringify(predicted);
      const actualStr = JSON.stringify(actual);

      // Calculate similarity
      const similarity = this.stringSimilarity(predStr, actualStr);
      return similarity > 0.8;
    } catch {
      return false;
    }
  }

  /**
   * Calculate string similarity (Levenshtein-based)
   */
  private stringSimilarity(a: string, b: string): number {
    const maxLen = Math.max(a.length, b.length);
    if (maxLen === 0) return 1;

    const distance = this.levenshteinDistance(a, b);
    return 1 - distance / maxLen;
  }

  private levenshteinDistance(a: string, b: string): number {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix: number[][] = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[b.length][a.length];
  }

  // ==========================================================================
  // Statistics
  // ==========================================================================

  /**
   * Calculate cascade statistics
   */
  private calculateStatistics(cascade: CascadeState): CascadeStatistics {
    let totalPredictions = 0;
    let totalConfidence = 0;

    for (const predictions of cascade.predictions.values()) {
      totalPredictions += predictions.length;
      for (const p of predictions) {
        totalConfidence += p.confidence;
      }
    }

    const averageConfidence =
      totalPredictions > 0 ? totalConfidence / totalPredictions : 0;

    // Calculate accuracy from history
    const history = this.predictionHistory.get(cascade.id) ?? [];
    const accurateCount = history.filter((h) => h.accurate).length;
    const accuracyScore =
      history.length > 0 ? accurateCount / history.length : 0;

    // Entity coverage
    const coveredEntities = cascade.predictions.size;
    const entityCoverage =
      cascade.config.entityIds.length > 0
        ? coveredEntities / cascade.config.entityIds.length
        : 0;

    return {
      totalPredictions,
      averageConfidence,
      accuracyScore,
      predictionLatency: cascade.statistics.predictionLatency,
      entityCoverage,
    };
  }

  // ==========================================================================
  // Event Emission
  // ==========================================================================

  private emitEvent(event: CascadeEvent): void {
    const listeners = this.listeners.get(event.cascadeId);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(event);
        } catch (error) {
          console.error("Cascade event listener error:", error);
        }
      }
    }
  }

  /**
   * Subscribe to cascade events
   */
  subscribe(
    cascadeId: string,
    listener: (event: CascadeEvent) => void
  ): () => void {
    if (!this.listeners.has(cascadeId)) {
      this.listeners.set(cascadeId, new Set());
    }
    this.listeners.get(cascadeId)!.add(listener);

    return () => {
      this.listeners.get(cascadeId)?.delete(listener);
    };
  }
}

// ============================================================================
// React Integration Types
// ============================================================================

/**
 * Prediction visualization props
 */
export interface PredictionVisualizationProps {
  prediction: PredictiveState;
  mode: "ghost" | "trail" | "cloud" | "timeline";
  showConnection: boolean;
  showUncertainty: boolean;
}

/**
 * Cascade view props
 */
export interface CascadeViewProps {
  cascadeId: string;
  entityIds: UniversalId[];
  mode: "ghost" | "trail" | "cloud" | "timeline";
}

// ============================================================================
// Exports
// ============================================================================

export const PredictiveCascade = PredictiveVisualizationCascade;
export default PredictiveVisualizationCascade;
