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
 * const spawner = new AgentSpawner({ arena, templates: [], defaultMutationRate: 0.1, attributeNoise: 0.1, uniqueNames: true });
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
  // Schemas
  ArenaConfigSchema,
  AgentConfigSchema,
  AgentTypeSchema,
  ArenaTopologySchema,
  ResourceTypeSchema,
  InteractionModeSchema,
  // Types
  type ArenaConfig,
  type AgentConfig,
  type AgentType,
  type ArenaTopology,
  type ResourceType,
  type InteractionMode,
  type Position,
  type SwarmAgent,
  type AgentState,
  type AgentAction,
  type ArenaCell,
  type ArenaState,
  type Message,
  type ArenaEvent,
  type ArenaResults,
  type AgentRanking,
  type TeamRanking,
  type TimelineEntry,
  type EmergentBehavior,
  type SwarmArenaEvents,
  // Behavior engines
  type BehaviorEngine,
  RandomBehaviorEngine,
  RuleBasedBehaviorEngine,
  ScriptedBehaviorEngine,
} from "./arena";

// Backwards compatibility alias
export { type ArenaConfig as SwarmArenaConfig } from "./arena";

// =============================================================================
// AGENTS - Spawning and Population Management
// =============================================================================

export {
  AgentSpawner,
  // Schemas
  AgentTemplateSchema,
  SpawnPatternSchema,
  PopulationConfigSchema,
  SpawnerConfigSchema,
  // Types
  type AgentTemplate,
  type SpawnPattern,
  type PopulationConfig,
  type SpawnerConfig,
  type SpawnResult,
  type GenerationStats,
  // Predefined templates
  PredefinedTemplates,
} from "./agents";

// =============================================================================
// TOURNAMENT - Evolution and Genetic Algorithms
// =============================================================================

export {
  Tournament,
  // Schemas
  TournamentConfigSchema,
  TournamentFormatSchema,
  SelectionMethodSchema,
  CrossoverMethodSchema,
  MutationMethodSchema,
  // Types
  type TournamentConfig,
  type TournamentFormat,
  type SelectionMethod,
  type CrossoverMethod,
  type MutationMethod,
  type Genome,
  type Match,
  type TournamentResults,
  type GenerationResult,
  type EvolutionMetrics,
  type TournamentEvents,
} from "./tournament";

// =============================================================================
// MODULE METADATA
// =============================================================================

export const SWARM_VERSION = "1.0.0";

export const SWARM_CAPABILITIES = {
  arena: {
    maxGridSize: 1000,
    maxAgents: 100000,
    behaviorEngines: ["random", "rule-based", "scripted", "neural"] as const,
    patternDetection: [
      "flocking",
      "clustering",
      "swarming",
      "spiral",
      "lane",
      "oscillation",
    ] as const,
  },
  spawner: {
    patterns: ["random", "cluster", "grid", "ring", "line", "corner"] as const,
    templates: [
      "scout",
      "harvester",
      "guardian",
      "hunter",
      "messenger",
      "generalist",
    ] as const,
  },
  tournament: {
    formats: [
      "round_robin",
      "single_elimination",
      "double_elimination",
      "swiss",
      "league",
      "battle_royale",
      "team_battle",
    ] as const,
  },
} as const;
