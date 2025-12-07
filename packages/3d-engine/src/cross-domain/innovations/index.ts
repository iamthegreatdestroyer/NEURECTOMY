/**
 * Cross-Domain Innovations Index
 *
 * Breakthrough innovations that emerge from cross-domain synthesis.
 * These innovations create capabilities that no single module could achieve alone.
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations
 * @agents @NEXUS @GENESIS
 */

// =============================================================================
// Forge × Twin Innovations
// =============================================================================

export {
  TemporalTwinReplayTheater,
  ReplayTheater,
  createUseReplayTheater,
  type TwinVisualSnapshot,
  type ReplaySessionConfig,
  type ReplayState,
  type ReplayKeyframe,
  type TheaterEvent,
  type TheaterEventType,
  type UseReplayTheaterResult,
} from "./replay-theater";

export {
  PredictiveVisualizationCascade,
  PredictiveCascade,
  type PredictiveState,
  type PredictiveVisualization,
  type PredictionMetadata,
  type PredictiveCascadeConfig,
  type CascadeState,
  type CascadeStatistics,
  type CascadeEvent,
  type CascadeEventType,
  type PredictionVisualizationProps,
  type CascadeViewProps,
} from "./predictive-cascade";

// =============================================================================
// Forge × Foundry × Twin Innovations
// =============================================================================

export {
  ConsciousnessHeatmapGenerator,
  ConsciousnessHeatmaps,
  type AttentionPattern,
  type AttentionWeights,
  type AttentionInsights,
  type ReasoningPattern,
  type AttentionAnomaly,
  type HeatmapVoxel,
  type ConsciousnessHeatmap,
  type FlowLine,
  type DecisionPoint,
  type HeatmapMetrics,
  type HeatmapConfig,
  type HeatmapEvent,
  type HeatmapEventType,
} from "./consciousness-heatmaps";

// =============================================================================
// Twin × Foundry Innovations
// =============================================================================

export {
  TwinGuidedArchitectureSearch,
  type ArchitectureSearchConfig,
  type TrainingObservation,
  type ActivationStatistics,
  type TrainingAnomaly,
  type BehaviorPattern,
  type ArchitectureRecommendation,
  type SimulationResult,
  type SearchSession,
} from "./architecture-search";

export {
  ModelInLoopSync,
  type ModelSyncConfig,
  type TwinState,
  type Prediction,
  type Outcome,
  type PredictionResult,
  type SyncMetrics,
  type DriftAlert,
  type SyncSession,
} from "./model-sync";

export {
  CascadeAwareTraining,
  type CascadeTrainingConfig,
  type ModelAction,
  type CascadeEvent as TrainingCascadeEvent,
  type CascadeTree,
  type CascadeLoss,
  type CascadeAnalysis,
  type TrainingProgress,
  type CascadeTrainingSession,
  type CascadeTrainingMetrics,
} from "./cascade-training";

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create and configure replay theater for a set of twins
 */
export function createReplayTheater(
  twinIds: string[],
  options?: Partial<import("./replay-theater").ReplaySessionConfig>
): {
  theater: import("./replay-theater").TemporalTwinReplayTheater;
  sessionId: string;
} {
  const { TemporalTwinReplayTheater } = require("./replay-theater");

  const theater = TemporalTwinReplayTheater.getInstance();
  const sessionId = `replay-${Date.now()}`;

  theater.createSession({
    sessionId,
    twinIds,
    timeRange: options?.timeRange ?? {
      start: Date.now() - 300000,
      end: Date.now(),
    },
    playback: options?.playback ?? {
      speed: 1,
      loop: false,
      direction: "forward",
    },
    visualization: options?.visualization ?? {
      showConnections: true,
      showAnnotations: true,
      showMetrics: true,
      colorByMetric: "health",
      interpolationMode: "smooth",
    },
    filters: options?.filters ?? {},
    ...options,
  });

  return { theater, sessionId };
}

/**
 * Create predictive cascade for entities
 */
export function createPredictiveCascade(
  entityIds: string[],
  options?: Partial<import("./predictive-cascade").PredictiveCascadeConfig>
): {
  cascade: import("./predictive-cascade").PredictiveVisualizationCascade;
  cascadeId: string;
} {
  const { PredictiveVisualizationCascade } = require("./predictive-cascade");

  const cascade = PredictiveVisualizationCascade.getInstance();

  const state = cascade.createCascade({
    entityIds,
    horizons: options?.horizons ?? [5000, 15000, 30000],
    updateInterval: options?.updateInterval ?? 1000,
    minConfidence: options?.minConfidence ?? 0.5,
    maxPredictionsPerEntity: options?.maxPredictionsPerEntity ?? 3,
    visualizationMode: options?.visualizationMode ?? "ghost",
    showUncertainty: options?.showUncertainty ?? true,
    showConnections: options?.showConnections ?? true,
    autoRefresh: options?.autoRefresh ?? true,
    ...options,
  });

  return { cascade, cascadeId: state.id };
}

/**
 * Create consciousness heatmap for entity monitoring
 */
export function createConsciousnessHeatmap(
  entityId: string,
  options?: Partial<import("./consciousness-heatmaps").HeatmapConfig>
): import("./consciousness-heatmaps").ConsciousnessHeatmapGenerator {
  const { ConsciousnessHeatmapGenerator } = require("./consciousness-heatmaps");

  const generator = ConsciousnessHeatmapGenerator.getInstance();
  generator.startMonitoring(entityId, options);

  return generator;
}

/**
 * Create architecture search instance
 */
export function createArchitectureSearch(
  options?: Partial<import("./architecture-search").ArchitectureSearchConfig>
): import("./architecture-search").TwinGuidedArchitectureSearch {
  const { TwinGuidedArchitectureSearch } = require("./architecture-search");
  return new TwinGuidedArchitectureSearch(options);
}

/**
 * Create model-in-loop sync instance
 */
export function createModelSync(
  options?: Partial<import("./model-sync").ModelSyncConfig>
): import("./model-sync").ModelInLoopSync {
  const { ModelInLoopSync } = require("./model-sync");
  return new ModelInLoopSync(options);
}

/**
 * Create cascade-aware training instance
 */
export function createCascadeTraining(
  options?: Partial<import("./cascade-training").CascadeTrainingConfig>
): import("./cascade-training").CascadeAwareTraining {
  const { CascadeAwareTraining } = require("./cascade-training");
  return new CascadeAwareTraining(options);
}

// =============================================================================
// P0 Breakthrough Innovations
// =============================================================================

export {
  LivingArchitectureLaboratory,
  createLivingArchitectureLab,
  type LivingNeuron,
  type LivingConnection,
  type NeuralOrganism,
  type OrganismMetrics,
  type EnvironmentalConditions,
  type LifecycleEvent,
  type LabExperiment,
  type ExperimentResult,
} from "./breakthroughs/living-architecture-lab";

export {
  MorphogenicModelEvolution,
  createMorphogenicEvolution,
  type MorphogenicGene,
  type EvolutionaryPressure,
  type MorphogenicRule,
  type StructuralMutation,
  type FitnessLandscape,
  type EvolutionaryLineage,
  type GenerationSnapshot,
} from "./breakthroughs/morphogenic-evolution";

export {
  CausalTrainingDebugger,
  createCausalDebugger,
  type CausalNode,
  type CausalEdge,
  type CausalGraph,
  type TrainingEvent,
  type CausalQuery,
  type CounterfactualScenario,
  type DebuggerInsight,
} from "./breakthroughs/causal-training-debugger";

export {
  QuantumArchitectureSearch,
  createQuantumArchitectureSearch,
  quickQuantumSearch,
  type ArchitectureBasisState,
  type ArchitectureSuperposition,
  type MeasurementResult,
  type QuantumGate,
  type SearchStrategy,
  type SearchConstraints,
  type LayerSpec,
  type LayerType,
  type SuperpositionVisualization,
} from "./breakthroughs/quantum-architecture-search";

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Combined innovation result for full-stack integration
 */
export interface InnovationStackResult {
  replayTheater?: ReturnType<typeof createReplayTheater>;
  predictiveCascade?: ReturnType<typeof createPredictiveCascade>;
  consciousnessHeatmap?: ReturnType<typeof createConsciousnessHeatmap>;
}

/**
 * Create full innovation stack for an entity
 */
export function createInnovationStack(
  entityId: string,
  options?: {
    enableReplay?: boolean;
    enablePredictions?: boolean;
    enableHeatmaps?: boolean;
  }
): InnovationStackResult {
  const result: InnovationStackResult = {};

  if (options?.enableReplay !== false) {
    result.replayTheater = createReplayTheater([entityId]);
  }

  if (options?.enablePredictions !== false) {
    result.predictiveCascade = createPredictiveCascade([entityId]);
  }

  if (options?.enableHeatmaps !== false) {
    result.consciousnessHeatmap = createConsciousnessHeatmap(entityId);
  }

  return result;
}
