/**
 * @fileoverview Neurectomy Experimentation Engine
 * @module @neurectomy/experimentation-engine
 *
 * Comprehensive experimentation framework for AI agent development.
 * Provides hypothesis testing, A/B testing, chaos engineering, and
 * swarm intelligence capabilities.
 *
 * @packageDocumentation
 */

// =============================================================================
// HYPOTHESIS LAB MODULE
// =============================================================================

export {
  // Lab
  HypothesisLab,
  createHypothesisLab,
  defineHypothesis,
  defineParameterSpace,
  type HypothesisConfig,
  type Parameter,
  type ParameterSpace,
  type ParameterConstraint,
  type TrialConfig,
  type TrialResult,
  type Hypothesis,
  type Trial,
  type LabConfig,
  type StorageBackend,
  type LabEvents,
  // Tracker
  ExperimentTracker,
  createTracker,
  withRun,
  type RunConfig,
  type Run,
  type RunStatus,
  type MetricLog,
  type Artifact,
  type ArtifactType,
  type Experiment,
  type TrackerConfig,
  type TrackerEvents,
  type ComparisonResult,
  type MetricComparison,
  type ParameterComparison,
  type RunRanking,
  // Versioning
  ModelRegistry,
  createRegistry,
  defineSignature,
  defineArtifact,
  type ModelVersion,
  type ModelStage,
  type ModelArtifact,
  type ModelSignature,
  type SignatureField,
  type ModelLineage,
  type RegisteredModel,
  type VersioningConfig,
  type VersioningEvents,
  type VersionComparison,
  type MetricDiff,
  type ParameterDiff,
  type StructuralChange,
} from "./hypothesis";

// =============================================================================
// A/B TESTING MODULE
// =============================================================================

export {
  // Engine
  ABTestingEngine,
  type ABExperimentConfig,
  type VariantConfig,
  type MetricConfig,
  type TargetingConfig,
  type TargetingRule,
  type ScheduleConfig,
  type ExperimentSettings,
  type ABExperiment,
  type VariantState,
  type MetricState,
  type ExperimentResult as ABExperimentResult,
  type ABTestingEvents,
  // Statistics
  proportionZTest,
  twoSampleTTest,
  chiSquaredTest,
  mannWhitneyUTest,
  bayesianABTest,
  calculateSampleSize as abCalculateSampleSize,
  calculatePower,
  sequentialTest,
  calculateEffectSize as abCalculateEffectSize,
  calculateConfidenceInterval as abCalculateConfidenceInterval,
  multipleTestingCorrection,
  type StatisticalTestResult,
  type BayesianResult,
  type SampleSizeResult,
  type SequentialTestResult,
  type MultipleTesting,
  // Assignment
  AssignmentManager,
  RandomAssignment,
  DeterministicAssignment,
  WeightedAssignment,
  EpsilonGreedyAssignment,
  ThompsonSamplingAssignment,
  UCB1Assignment,
  ContextualAssignment,
  type AssignmentStrategy,
  type AssignmentStrategyType,
  type VariantWeight,
  type AssignmentContext,
  type BanditState,
  type AssignmentManagerEvents,
  // Backwards compatibility
  ABTestingEngine as ABTestManager,
  type ABExperimentConfig as ABTestConfig,
} from "./ab-testing";

// =============================================================================
// CHAOS ENGINEERING MODULE
// =============================================================================

export {
  // Core simulator
  ChaosSimulator,
  type FaultInjector,
  type ChaosExperiment,
  type ChaosExperimentConfig,
  type AffectedTarget,
  type FaultConfig,
  type FaultType,
  type FaultSeverity,
  type ExperimentState,
  type SafetyConfig,
  type ChaosStorage,
  type ChaosNotifier,
  type MetricsProvider,
  type ExperimentResults,
  type ActiveFault,
  type HealthStatus,
  type HealthCheckResult,
  type MetricSnapshot,
  type TimelineEvent,
  type Finding,
  type Approval,
  type ChaosSimulatorEvents,
  type ChaosNotification,
  FaultTypeSchema,
  FaultSeveritySchema,
  ExperimentStateSchema,
  FaultConfigSchema,
  TargetSelectorSchema,
  BlastRadiusConfigSchema,
  SafetyConfigSchema,
  HealthCheckConfigSchema,
  ChaosExperimentConfigSchema,
  // Extended faults
  NetworkPartitionInjector,
  PacketLossInjector,
  BandwidthLimitInjector,
  DNSFailureInjector,
  CPUStressInjector,
  MemoryStressInjector,
  DiskStressInjector,
  ProcessKillInjector,
  ContainerActionInjector,
  NodeDrainInjector,
  FaultRegistry,
  ChaosScenarios,
  NetworkPartitionConfigSchema,
  PacketLossConfigSchema,
  BandwidthLimitConfigSchema,
  DNSFailureConfigSchema,
  CPUStressConfigSchema,
  MemoryStressConfigSchema,
  DiskStressConfigSchema,
  IOStressConfigSchema,
  ProcessKillConfigSchema,
  ContainerActionConfigSchema,
  NodeDrainConfigSchema,
  type NetworkPartitionConfig,
  type PacketLossConfig,
  type BandwidthLimitConfig,
  type DNSFailureConfig,
  type CPUStressConfig,
  type MemoryStressConfig,
  type DiskStressConfig,
  type IOStressConfig,
  type ProcessKillConfig,
  type ContainerActionConfig,
  type NodeDrainConfig,
  // Scheduler
  ChaosScheduler,
  CronExpressionSchema,
  ScheduleWindowSchema,
  ExperimentScheduleSchema,
  GamedayConfigSchema,
  type ExperimentSchedule,
  type GamedayConfig,
  type ScheduleWindow,
  type SchedulerEvents,
  type SchedulerConfig,
  type ScheduledExecution,
  type GamedayResults,
  type SchedulerStatistics,
} from "./chaos";

// =============================================================================
// SWARM INTELLIGENCE MODULE
// =============================================================================

export {
  // Arena
  SwarmArena,
  type SwarmArenaConfig,
  ArenaConfigSchema,
  ArenaConfigSchema as SwarmArenaConfigSchema,
  type AgentState,
  AgentTypeSchema,
  AgentConfigSchema,
  type ArenaCell,
  type Position,
  type ArenaState,
  type BehaviorEngine,
  RandomBehaviorEngine,
  RuleBasedBehaviorEngine,
  ScriptedBehaviorEngine,
  type EmergentBehavior,
  // Agents
  AgentSpawner,
  type AgentTemplate,
  AgentTemplateSchema,
  type SpawnerConfig,
  SpawnerConfigSchema,
  type SpawnPattern,
  type SpawnResult,
  PredefinedTemplates,
  // Tournament
  Tournament,
  type TournamentConfig,
  TournamentConfigSchema,
  type Genome,
  type Match,
  type TournamentResults,
  type SelectionMethod,
  type CrossoverMethod,
  type MutationMethod,
  type TournamentFormat,
  // Metadata
  SWARM_VERSION,
  SWARM_CAPABILITIES,
} from "./swarm";

// =============================================================================
// SHARED TYPES
// =============================================================================

export * from "./types";

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

import { HypothesisLab, type LabConfig } from "./hypothesis";
import { ABTestingEngine } from "./ab-testing";
import type { ABEngineConfig } from "./ab-testing/engine";
import {
  ChaosSimulator,
  ChaosScheduler,
  type ChaosStorage,
  type ChaosNotifier,
  type MetricsProvider,
  type SafetyConfig,
} from "./chaos";
import {
  SwarmArena,
  AgentSpawner,
  Tournament,
  type SwarmArenaConfig,
  type TournamentConfig,
  type SpawnerConfig,
} from "./swarm";

/**
 * Configuration for the complete experimentation engine
 */
export interface ExperimentationEngineConfig {
  hypothesis?: Partial<LabConfig>;
  abTesting?: Partial<ABEngineConfig>;
  chaos?: {
    storage?: ChaosStorage;
    notifier?: ChaosNotifier;
    metricsProvider?: MetricsProvider;
    defaultSafety?: Partial<SafetyConfig>;
  };
  swarm?: Partial<SwarmArenaConfig>;
  tournament?: Partial<TournamentConfig>;
  spawner?: Partial<Omit<SpawnerConfig, "arena">>;
}

/**
 * Complete experimentation engine instance
 */
export interface ExperimentationEngine {
  hypothesis: HypothesisLab;
  abTesting: ABTestingEngine;
  chaos: ChaosSimulator;
  scheduler: ChaosScheduler;
  arena: SwarmArena;
  spawner: AgentSpawner;
  tournament: Tournament;

  // Lifecycle methods
  start(): Promise<void>;
  stop(): Promise<void>;

  // Status
  getStatus(): EngineStatus;
}

/**
 * Engine status information
 */
export interface EngineStatus {
  hypothesis: {
    activeExperiments: number;
    completedExperiments: number;
  };
  abTesting: {
    activeTests: number;
    totalAllocations: number;
  };
  chaos: {
    activeExperiments: number;
    scheduledExperiments: number;
    totalFaultsInjected: number;
  };
  swarm: {
    activeAgents: number;
    currentGeneration: number;
    patternsDetected: number;
  };
}

/**
 * Create a complete experimentation engine with all components
 */
export function createExperimentationEngine(
  config: ExperimentationEngineConfig = {}
): ExperimentationEngine {
  // Initialize all components
  const hypothesis = new HypothesisLab(config.hypothesis as LabConfig);
  const abTesting = new ABTestingEngine(config.abTesting as ABEngineConfig);
  const chaos = new ChaosSimulator(config.chaos);
  const scheduler = new ChaosScheduler(chaos);
  const arena = new SwarmArena({
    gridSize: { width: 100, height: 100 },
    maxAgents: 500,
    tickRate: 60,
    enableCollisions: true,
    enableEmergentDetection: true,
    wrapAround: true,
    ...config.swarm,
  } as SwarmArenaConfig);
  const spawner = new AgentSpawner({
    arena,
    templates: [],
    defaultMutationRate: 0.1,
    attributeNoise: 0.1,
    uniqueNames: true,
    ...config.spawner,
  });
  const tournament = new Tournament({
    populationSize: 100,
    generations: 50,
    selectionMethod: "tournament",
    crossoverMethod: "uniform",
    mutationMethod: "gaussian",
    mutationRate: 0.05,
    crossoverRate: 0.9,
    elitismRate: 0.05,
    ...config.tournament,
  } as TournamentConfig);

  return {
    hypothesis,
    abTesting,
    chaos,
    scheduler,
    arena,
    spawner,
    tournament,

    async start(): Promise<void> {
      // Start all engines that have start methods
      arena.start();
    },

    async stop(): Promise<void> {
      // Stop all running components
      arena.stop();
    },

    getStatus(): EngineStatus {
      return {
        hypothesis: {
          activeExperiments: 0,
          completedExperiments: 0,
        },
        abTesting: {
          activeTests: 0,
          totalAllocations: 0,
        },
        chaos: {
          activeExperiments: 0,
          scheduledExperiments: 0,
          totalFaultsInjected: 0,
        },
        swarm: {
          activeAgents: arena.getAliveAgents?.()?.length ?? 0,
          currentGeneration: tournament.getCurrentGeneration?.() ?? 0,
          patternsDetected: 0,
        },
      };
    },
  };
}

/**
 * Create a lightweight engine with only essential components
 */
export function createLightweightEngine(): Pick<
  ExperimentationEngine,
  "hypothesis" | "abTesting" | "chaos"
> {
  return {
    hypothesis: new HypothesisLab(),
    abTesting: new ABTestingEngine(),
    chaos: new ChaosSimulator(),
  };
}

/**
 * Create an engine focused on swarm experimentation
 */
export function createSwarmEngine(config?: {
  arenaConfig?: Partial<SwarmArenaConfig>;
  tournamentConfig?: Partial<TournamentConfig>;
}): Pick<ExperimentationEngine, "arena" | "spawner" | "tournament"> {
  const arena = new SwarmArena({
    gridSize: { width: 100, height: 100 },
    maxAgents: 500,
    tickRate: 60,
    enableCollisions: true,
    enableEmergentDetection: true,
    wrapAround: true,
    ...config?.arenaConfig,
  } as SwarmArenaConfig);

  const spawner = new AgentSpawner({
    arena,
    templates: [],
    defaultMutationRate: 0.1,
    attributeNoise: 0.1,
    uniqueNames: true,
  });

  const tournament = new Tournament({
    populationSize: 100,
    generations: 50,
    selectionMethod: "tournament",
    crossoverMethod: "uniform",
    mutationMethod: "gaussian",
    mutationRate: 0.05,
    crossoverRate: 0.9,
    elitismRate: 0.05,
    ...config?.tournamentConfig,
  } as TournamentConfig);

  return { arena, spawner, tournament };
}

/**
 * Create an engine focused on chaos engineering
 */
export function createChaosEngine(config?: {
  storage?: ChaosStorage;
  notifier?: ChaosNotifier;
  metricsProvider?: MetricsProvider;
  defaultSafety?: Partial<SafetyConfig>;
}): Pick<ExperimentationEngine, "chaos" | "scheduler"> {
  const chaos = new ChaosSimulator(config);
  const scheduler = new ChaosScheduler(chaos);

  return { chaos, scheduler };
}

// =============================================================================
// MODULE METADATA
// =============================================================================

/**
 * Experimentation engine version
 */
export const VERSION = "1.0.0";

/**
 * Module capabilities and limits
 */
export const CAPABILITIES = {
  hypothesis: {
    maxConcurrentExperiments: 100,
    supportedTests: [
      "t-test",
      "chi-square",
      "anova",
      "mann-whitney",
      "bayesian",
    ],
  },
  abTesting: {
    maxConcurrentTests: 50,
    maxVariantsPerTest: 10,
    allocationStrategies: ["random", "weighted", "sticky", "bandit"],
  },
  chaos: {
    maxConcurrentFaults: 20,
    supportedFaultTypes: [
      "latency",
      "error",
      "timeout",
      "network-partition",
      "packet-loss",
      "bandwidth-limit",
      "dns-failure",
      "cpu-stress",
      "memory-stress",
      "disk-stress",
      "process-kill",
      "container-action",
      "node-drain",
    ],
    blastRadii: ["single", "small", "medium", "large", "full"],
  },
  swarm: {
    maxGridSize: 1000,
    maxAgents: 100000,
    behaviorEngines: ["random", "rule-based", "scripted", "neural"],
    patternDetection: [
      "flocking",
      "clustering",
      "swarming",
      "spiral",
      "lane",
      "oscillation",
    ],
    selectionMethods: [
      "top_n",
      "tournament",
      "roulette",
      "rank",
      "elitist",
      "random",
    ],
    crossoverMethods: ["uniform", "single_point", "arithmetic", "blend"],
    mutationMethods: ["gaussian", "uniform", "polynomial"],
  },
} as const;

/**
 * Default export for convenient importing
 */
export default {
  createExperimentationEngine,
  createLightweightEngine,
  createSwarmEngine,
  createChaosEngine,
  VERSION,
  CAPABILITIES,
};
