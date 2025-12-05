/**
 * @fileoverview Swarm Intelligence Module
 * @module @neurectomy/experimentation-engine/swarm
 *
 * Multi-agent swarm simulation and evolutionary tournament system.
 * Provides emergent behavior detection, genetic algorithm evolution,
 * and agent population management.
 *
 * @example
 * ```typescript
 * import {
 *   SwarmArena,
 *   AgentSpawner,
 *   Tournament,
 *   PredefinedTemplates,
 * } from '@neurectomy/experimentation-engine/swarm';
 *
 * // Create arena with behavior detection
 * const arena = new SwarmArena({
 *   gridSize: { width: 100, height: 100 },
 *   maxAgents: 500,
 *   tickRate: 60,
 *   enableCollisions: true,
 *   enableEmergentDetection: true,
 * });
 *
 * // Use spawner for population management
 * const spawner = new AgentSpawner(arena);
 * spawner.spawnFromTemplate(PredefinedTemplates.scout, 50);
 * spawner.spawnFromTemplate(PredefinedTemplates.harvester, 30);
 *
 * // Run evolution tournament
 * const tournament = new Tournament({
 *   populationSize: 100,
 *   generations: 50,
 *   selectionMethod: 'tournament',
 *   crossoverMethod: 'uniform',
 *   mutationMethod: 'gaussian',
 * });
 *
 * await tournament.evolve();
 * const champion = tournament.getChampion();
 * ```
 *
 * @packageDocumentation
 */

// =============================================================================
// ARENA - Multi-Agent Simulation Environment
// =============================================================================

export {
  SwarmArena,
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
  type BehaviorEngine,
  RandomBehaviorEngine,
  RuleBasedBehaviorEngine,
  ScriptedBehaviorEngine,
  type EmergentPattern,
  EmergentPatternSchema,
  type PatternType,
} from "./arena";

// =============================================================================
// AGENTS - Spawning and Population Management
// =============================================================================

export {
  AgentSpawner,
  type AgentTemplate,
  AgentTemplateSchema,
  type SpawnConfig,
  SpawnConfigSchema,
  type SpawnPattern,
  type SpawnResult,
  SpawnResultSchema,
  type PopulationSnapshot,
  PopulationSnapshotSchema,
  PredefinedTemplates,
} from "./agents";

// =============================================================================
// TOURNAMENT - Evolution and Genetic Algorithms
// =============================================================================

export {
  Tournament,
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
  // Selection strategies
  topNSelection,
  tournamentSelection,
  rouletteSelection,
  rankSelection,
  elitistSelection,
  // Crossover operators
  uniformCrossover,
  singlePointCrossover,
  arithmeticCrossover,
  blendCrossover,
  // Mutation operators
  gaussianMutation,
  uniformMutation,
  polynomialMutation,
} from "./tournament";

// =============================================================================
// TYPE UTILITIES
// =============================================================================

/**
 * Re-export common types for convenience
 */
export type {
  // Arena types
  SwarmArenaConfig as ArenaConfig,
  AgentState as Agent,
  ArenaCell as Cell,
  EmergentPattern as Pattern,

  // Spawner types
  AgentTemplate as Template,
  SpawnConfig as Spawn,
  SpawnResult as SpawnInfo,
  PopulationSnapshot as Population,

  // Tournament types
  TournamentConfig as EvolutionConfig,
  Genome as DNA,
  Individual as Organism,
  Generation as Gen,
  TournamentResult as EvolutionResult,
} from "./arena";

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

import { SwarmArena, type SwarmArenaConfig } from "./arena";
import { AgentSpawner } from "./agents";
import { Tournament, type TournamentConfig } from "./tournament";

/**
 * Create a pre-configured arena for quick experimentation
 */
export function createQuickArena(
  preset: "small" | "medium" | "large" | "massive" = "medium"
): SwarmArena {
  const presets: Record<string, Partial<SwarmArenaConfig>> = {
    small: {
      gridSize: { width: 50, height: 50 },
      maxAgents: 100,
      tickRate: 30,
    },
    medium: {
      gridSize: { width: 100, height: 100 },
      maxAgents: 500,
      tickRate: 60,
    },
    large: {
      gridSize: { width: 200, height: 200 },
      maxAgents: 2000,
      tickRate: 60,
    },
    massive: {
      gridSize: { width: 500, height: 500 },
      maxAgents: 10000,
      tickRate: 30,
    },
  };

  return new SwarmArena({
    ...presets[preset],
    enableCollisions: true,
    enableEmergentDetection: true,
    wrapAround: true,
  } as SwarmArenaConfig);
}

/**
 * Create an arena with spawner attached
 */
export function createArenaWithSpawner(
  config: Partial<SwarmArenaConfig> = {}
): { arena: SwarmArena; spawner: AgentSpawner } {
  const arena = new SwarmArena({
    gridSize: { width: 100, height: 100 },
    maxAgents: 500,
    tickRate: 60,
    enableCollisions: true,
    enableEmergentDetection: true,
    wrapAround: true,
    ...config,
  } as SwarmArenaConfig);

  const spawner = new AgentSpawner(arena);

  return { arena, spawner };
}

/**
 * Create a tournament with default evolution settings
 */
export function createQuickTournament(
  preset: "fast" | "balanced" | "thorough" = "balanced"
): Tournament {
  const presets: Record<string, Partial<TournamentConfig>> = {
    fast: {
      populationSize: 50,
      generations: 20,
      selectionMethod: "top_n",
      crossoverMethod: "uniform",
      mutationMethod: "gaussian",
      mutationRate: 0.1,
      crossoverRate: 0.8,
      elitismRate: 0.1,
    },
    balanced: {
      populationSize: 100,
      generations: 50,
      selectionMethod: "tournament",
      crossoverMethod: "uniform",
      mutationMethod: "gaussian",
      mutationRate: 0.05,
      crossoverRate: 0.9,
      elitismRate: 0.05,
    },
    thorough: {
      populationSize: 200,
      generations: 100,
      selectionMethod: "tournament",
      crossoverMethod: "blend",
      mutationMethod: "polynomial",
      mutationRate: 0.02,
      crossoverRate: 0.95,
      elitismRate: 0.02,
    },
  };

  return new Tournament(presets[preset] as TournamentConfig);
}

/**
 * Run a complete swarm evolution experiment
 */
export async function runSwarmExperiment(options: {
  arenaPreset?: "small" | "medium" | "large" | "massive";
  tournamentPreset?: "fast" | "balanced" | "thorough";
  initialPopulation?: number;
  simulationTicks?: number;
  onTick?: (tick: number, stats: Record<string, unknown>) => void;
  onGeneration?: (gen: number, best: number) => void;
}): Promise<{
  arenaStats: Record<string, unknown>;
  evolutionResult: unknown;
  patterns: unknown[];
}> {
  const {
    arenaPreset = "medium",
    tournamentPreset = "balanced",
    initialPopulation = 100,
    simulationTicks = 1000,
    onTick,
    onGeneration,
  } = options;

  // Setup
  const { arena, spawner } = createArenaWithSpawner();
  const tournament = createQuickTournament(tournamentPreset);

  // Spawn initial population
  spawner.spawnRandom(initialPopulation);

  // Run simulation
  arena.start();

  for (let tick = 0; tick < simulationTicks; tick++) {
    await new Promise((resolve) => setTimeout(resolve, 16)); // ~60fps

    if (onTick) {
      onTick(tick, arena.getStats());
    }
  }

  arena.stop();

  // Get detected patterns
  const patterns = arena.getDetectedPatterns?.() ?? [];

  // Run evolution based on simulation results
  if (onGeneration) {
    tournament.on("generation", (gen, best) => onGeneration(gen, best));
  }

  await tournament.evolve();
  const evolutionResult = tournament.getResult();

  return {
    arenaStats: arena.getStats(),
    evolutionResult,
    patterns,
  };
}

// =============================================================================
// MODULE METADATA
// =============================================================================

export const SWARM_VERSION = "1.0.0";

export const SWARM_CAPABILITIES = {
  arena: {
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
  },
  spawner: {
    patterns: ["random", "cluster", "grid", "ring", "line", "corner"],
    templates: [
      "scout",
      "harvester",
      "guardian",
      "hunter",
      "messenger",
      "generalist",
    ],
  },
  tournament: {
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
