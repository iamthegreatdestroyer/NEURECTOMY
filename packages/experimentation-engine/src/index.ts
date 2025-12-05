/**
 * @fileoverview Neurectomy Experimentation Engine
 * @module @neurectomy/experimentation-engine
 *
 * Comprehensive experimentation framework for AI agent development.
 * Provides hypothesis testing, A/B testing, chaos engineering, and
 * swarm intelligence capabilities.
 *
 * @example
 * ```typescript
 * import {
 *   // Hypothesis Testing
 *   HypothesisLab,
 *   ExperimentRunner,
 *   StatisticalAnalyzer,
 *
 *   // A/B Testing
 *   ABTestManager,
 *   VariantAllocator,
 *   MetricsCollector,
 *
 *   // Chaos Engineering
 *   ChaosSimulator,
 *   ChaosScheduler,
 *   FaultRegistry,
 *
 *   // Swarm Intelligence
 *   SwarmArena,
 *   AgentSpawner,
 *   Tournament,
 * } from '@neurectomy/experimentation-engine';
 *
 * // Create a complete experiment pipeline
 * const lab = new HypothesisLab();
 * const chaos = new ChaosSimulator();
 * const arena = new SwarmArena();
 *
 * // Run experiments with chaos injection
 * await lab.runExperiment(myHypothesis, {
 *   chaosEnabled: true,
 *   chaosSimulator: chaos,
 * });
 * ```
 *
 * @packageDocumentation
 */

// =============================================================================
// HYPOTHESIS LAB MODULE
// =============================================================================

export {
  // Core classes
  HypothesisLab,
  ExperimentRunner,
  StatisticalAnalyzer,

  // Types and schemas
  type Hypothesis,
  HypothesisSchema,
  type Experiment,
  ExperimentSchema,
  type ExperimentResult,
  ExperimentResultSchema,
  type StatisticalTest,
  type TestResult,
  TestResultSchema,
  type HypothesisLabConfig,
  HypothesisLabConfigSchema,

  // Utilities
  calculatePValue,
  calculateConfidenceInterval,
  calculateEffectSize,
  calculateSampleSize,
  tTest,
  chiSquareTest,
  anovaTest,
  mannWhitneyTest,
} from "./hypothesis";

// =============================================================================
// A/B TESTING MODULE
// =============================================================================

export {
  // Core classes
  ABTestManager,
  VariantAllocator,
  MetricsCollector,
  TrafficRouter,

  // Types and schemas
  type ABTest,
  ABTestSchema,
  type Variant,
  VariantSchema,
  type Allocation,
  AllocationSchema,
  type ABTestResult,
  ABTestResultSchema,
  type Metric,
  MetricSchema,
  type MetricValue,
  MetricValueSchema,
  type ABTestConfig,
  ABTestConfigSchema,

  // Allocation strategies
  type AllocationStrategy,
  RandomAllocator,
  WeightedAllocator,
  StickyAllocator,
  MultiArmedBanditAllocator,

  // Statistical utilities
  calculateSignificance,
  calculateLift,
  calculateMDE,
  isStatisticallySignificant,
} from "./ab-testing";

// =============================================================================
// CHAOS ENGINEERING MODULE
// =============================================================================

export {
  // Core classes
  ChaosSimulator,
  ChaosScheduler,

  // Fault injectors
  type FaultInjector,
  LatencyInjector,
  ErrorInjector,
  TimeoutInjector,
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

  // Registry and scenarios
  FaultRegistry,
  ChaosScenarios,

  // Types and schemas
  type ChaosConfig,
  ChaosConfigSchema,
  type FaultConfig,
  FaultConfigSchema,
  type ChaosExperiment,
  ChaosExperimentSchema,
  type ChaosResult,
  ChaosResultSchema,
  type ScheduledExperiment,
  ScheduledExperimentSchema,
  type GamedayConfig,
  GamedayConfigSchema,
  type ExecutionWindow,
  ExecutionWindowSchema,

  // Targeting
  type TargetSelector,
  TargetSelectorSchema,
  type TargetType,

  // Blast radius
  type BlastRadius,
  BlastRadiusSchema,
} from "./chaos";

// =============================================================================
// SWARM INTELLIGENCE MODULE
// =============================================================================

export {
  // Core classes
  SwarmArena,
  AgentSpawner,
  Tournament,

  // Behavior engines
  type BehaviorEngine,
  RandomBehaviorEngine,
  RuleBasedBehaviorEngine,
  ScriptedBehaviorEngine,

  // Templates
  PredefinedTemplates,

  // Arena types
  type SwarmArenaConfig,
  SwarmArenaConfigSchema,
  type AgentState,
  AgentStateSchema,
  type ArenaCell,
  ArenaCellSchema,
  type Position,
  PositionSchema,
  type Velocity,
  VelocitySchema,
  type ArenaStats,
  type EmergentPattern,
  EmergentPatternSchema,
  type PatternType,

  // Spawner types
  type AgentTemplate,
  AgentTemplateSchema,
  type SpawnConfig,
  SpawnConfigSchema,
  type SpawnPattern,
  type SpawnResult,
  SpawnResultSchema,
  type PopulationSnapshot,
  PopulationSnapshotSchema,

  // Tournament types
  type TournamentConfig,
  TournamentConfigSchema,
  type Genome,
  GenomeSchema,
  type Individual,
  IndividualSchema,
  type Generation,
  GenerationSchema,
  type TournamentResult,
  TournamentResultSchema,
  type FitnessFunction,
  type SelectionMethod,
  type CrossoverMethod,
  type MutationMethod,
  type EvolutionStats,
  EvolutionStatsSchema,

  // Evolution operators
  topNSelection,
  tournamentSelection,
  rouletteSelection,
  rankSelection,
  elitistSelection,
  uniformCrossover,
  singlePointCrossover,
  arithmeticCrossover,
  blendCrossover,
  gaussianMutation,
  uniformMutation,
  polynomialMutation,

  // Factory functions
  createQuickArena,
  createArenaWithSpawner,
  createQuickTournament,
  runSwarmExperiment,

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

import { HypothesisLab, type HypothesisLabConfig } from "./hypothesis";
import { ABTestManager, type ABTestConfig } from "./ab-testing";
import { ChaosSimulator, ChaosScheduler, type ChaosConfig } from "./chaos";
import {
  SwarmArena,
  AgentSpawner,
  Tournament,
  type SwarmArenaConfig,
  type TournamentConfig,
} from "./swarm";

/**
 * Configuration for the complete experimentation engine
 */
export interface ExperimentationEngineConfig {
  hypothesis?: Partial<HypothesisLabConfig>;
  abTesting?: Partial<ABTestConfig>;
  chaos?: Partial<ChaosConfig>;
  swarm?: Partial<SwarmArenaConfig>;
  tournament?: Partial<TournamentConfig>;
}

/**
 * Complete experimentation engine instance
 */
export interface ExperimentationEngine {
  hypothesis: HypothesisLab;
  abTesting: ABTestManager;
  chaos: ChaosSimulator;
  scheduler: ChaosScheduler;
  arena: SwarmArena;
  spawner: AgentSpawner;
  tournament: Tournament;

  // Lifecycle methods
  start(): Promise<void>;
  stop(): Promise<void>;
  reset(): void;

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
 *
 * @example
 * ```typescript
 * const engine = createExperimentationEngine({
 *   chaos: { defaultBlastRadius: 'small' },
 *   swarm: { maxAgents: 1000 },
 * });
 *
 * await engine.start();
 *
 * // Use individual components
 * await engine.hypothesis.runExperiment(myHypothesis);
 * await engine.chaos.runExperiment(myFaults);
 * engine.arena.start();
 *
 * // Get overall status
 * console.log(engine.getStatus());
 *
 * await engine.stop();
 * ```
 */
export function createExperimentationEngine(
  config: ExperimentationEngineConfig = {}
): ExperimentationEngine {
  // Initialize all components
  const hypothesis = new HypothesisLab(
    config.hypothesis as HypothesisLabConfig
  );
  const abTesting = new ABTestManager(config.abTesting as ABTestConfig);
  const chaos = new ChaosSimulator(config.chaos as ChaosConfig);
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
  const spawner = new AgentSpawner(arena);
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
      scheduler.start?.();
    },

    async stop(): Promise<void> {
      // Stop all running components
      arena.stop();
      scheduler.stop?.();
      await chaos.stopAll?.();
    },

    reset(): void {
      // Reset all components to initial state
      hypothesis.reset?.();
      abTesting.reset?.();
      chaos.reset?.();
      arena.reset?.();
      tournament.reset?.();
    },

    getStatus(): EngineStatus {
      return {
        hypothesis: {
          activeExperiments: hypothesis.getActiveCount?.() ?? 0,
          completedExperiments: hypothesis.getCompletedCount?.() ?? 0,
        },
        abTesting: {
          activeTests: abTesting.getActiveCount?.() ?? 0,
          totalAllocations: abTesting.getTotalAllocations?.() ?? 0,
        },
        chaos: {
          activeExperiments: chaos.getActiveCount?.() ?? 0,
          scheduledExperiments: scheduler.getScheduledCount?.() ?? 0,
          totalFaultsInjected: chaos.getTotalFaults?.() ?? 0,
        },
        swarm: {
          activeAgents: arena.getAgentCount?.() ?? 0,
          currentGeneration: tournament.getCurrentGeneration?.() ?? 0,
          patternsDetected: arena.getDetectedPatterns?.()?.length ?? 0,
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
    abTesting: new ABTestManager(),
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

  const spawner = new AgentSpawner(arena);

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
export function createChaosEngine(
  config?: Partial<ChaosConfig>
): Pick<ExperimentationEngine, "chaos" | "scheduler"> {
  const chaos = new ChaosSimulator(config as ChaosConfig);
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
