/**
 * NEURECTOMY Swarm Tournament System
 * @module @neurectomy/experimentation-engine/swarm
 * @agent @OMNISCIENT @GENESIS
 *
 * Evolution tournament system for agent competition
 * and selection-based improvement.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  AgentConfig,
  ArenaConfig,
  ArenaResults,
  AgentRanking,
} from "./arena";
import { SwarmArena } from "./arena";
import { AgentSpawner, PredefinedTemplates } from "./agents";
import type { AgentTemplate, PopulationConfig } from "./agents";

// ============================================================================
// Tournament Configuration Schemas
// ============================================================================

export const TournamentFormatSchema = z.enum([
  "round_robin", // All vs all
  "single_elimination", // Knockout
  "double_elimination", // Knockout with losers bracket
  "swiss", // Swiss system pairing
  "league", // Season-style points
  "battle_royale", // All agents one arena
  "team_battle", // Team vs team
]);

export const SelectionMethodSchema = z.enum([
  "top_n", // Take top N performers
  "tournament", // Tournament selection
  "roulette", // Proportional to fitness
  "rank", // Rank-based selection
  "elitist", // Best always survive
  "random", // Random selection
]);

export const CrossoverMethodSchema = z.enum([
  "uniform", // Random attribute swap
  "single_point", // Split at single point
  "two_point", // Split at two points
  "arithmetic", // Weighted average
  "blend", // BLX-alpha crossover
]);

export const MutationMethodSchema = z.enum([
  "gaussian", // Add Gaussian noise
  "uniform", // Uniform random change
  "swap", // Swap attributes
  "scramble", // Randomize subset
  "polynomial", // Polynomial mutation
]);

export const TournamentConfigSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  description: z.string().optional(),
  format: TournamentFormatSchema,
  arenaConfig: z.custom<Omit<ArenaConfig, "id">>(),
  populationSize: z.number().int().positive(),
  generations: z.number().int().positive(),
  matchesPerGeneration: z.number().int().positive().optional(),
  selectionConfig: z.object({
    method: SelectionMethodSchema,
    survivalRate: z.number().min(0).max(1).default(0.5),
    eliteCount: z.number().int().nonnegative().default(2),
  }),
  evolutionConfig: z.object({
    crossoverMethod: CrossoverMethodSchema,
    crossoverRate: z.number().min(0).max(1).default(0.7),
    mutationMethod: MutationMethodSchema,
    mutationRate: z.number().min(0).max(1).default(0.1),
    mutationStrength: z.number().min(0).max(1).default(0.2),
  }),
  fitnessWeights: z.object({
    score: z.number().default(1.0),
    survival: z.number().default(0.3),
    resources: z.number().default(0.2),
    interactions: z.number().default(0.1),
  }),
  teams: z
    .array(
      z.object({
        name: z.string(),
        templateId: z.string(),
        color: z.string().optional(),
      })
    )
    .optional(),
  seedAgents: z.array(z.custom<Partial<AgentConfig>>()).optional(),
});

// ============================================================================
// Types
// ============================================================================

export type TournamentFormat = z.infer<typeof TournamentFormatSchema>;
export type SelectionMethod = z.infer<typeof SelectionMethodSchema>;
export type CrossoverMethod = z.infer<typeof CrossoverMethodSchema>;
export type MutationMethod = z.infer<typeof MutationMethodSchema>;
export type TournamentConfig = z.infer<typeof TournamentConfigSchema>;

export interface Genome {
  id: string;
  parentIds: string[];
  generation: number;
  attributes: AgentConfig["attributes"];
  behaviorType: string;
  fitness: number;
  wins: number;
  losses: number;
  draws: number;
  totalScore: number;
  metadata: Record<string, unknown>;
}

export interface Match {
  id: string;
  generation: number;
  round: number;
  participants: string[];
  winner?: string;
  results: Map<string, AgentRanking>;
  arenaResults?: ArenaResults;
  startTime: Date;
  endTime?: Date;
}

export interface TournamentResults {
  tournamentId: string;
  config: TournamentConfig;
  generations: GenerationResult[];
  champion: Genome;
  allTimeTop: Genome[];
  totalMatches: number;
  totalDuration: number;
  evolutionMetrics: EvolutionMetrics;
}

export interface GenerationResult {
  generation: number;
  population: Genome[];
  matches: Match[];
  bestFitness: number;
  avgFitness: number;
  diversity: number;
  elites: Genome[];
}

export interface EvolutionMetrics {
  fitnessOverTime: number[];
  diversityOverTime: number[];
  convergenceGeneration?: number;
  bestAttributes: Record<string, number>;
  attributeEvolution: Record<string, number[]>;
}

export interface TournamentEvents {
  initialized: (config: TournamentConfig) => void;
  generationStarted: (generation: number) => void;
  matchStarted: (match: Match) => void;
  matchCompleted: (match: Match) => void;
  generationCompleted: (result: GenerationResult) => void;
  evolutionStep: (generation: number, population: Genome[]) => void;
  newChampion: (genome: Genome) => void;
  completed: (results: TournamentResults) => void;
  error: (error: Error) => void;
}

// ============================================================================
// Tournament Implementation
// ============================================================================

/**
 * Tournament - Evolution-based agent competition
 *
 * Features:
 * - Multiple tournament formats
 * - Genetic algorithm evolution
 * - Fitness-based selection
 * - Crossover and mutation
 * - Diversity preservation
 */
export class Tournament extends EventEmitter<TournamentEvents> {
  private config: TournamentConfig;
  private population: Map<string, Genome> = new Map();
  private matches: Match[] = [];
  private generationResults: GenerationResult[] = [];
  private currentGeneration: number = 0;
  private champion: Genome | null = null;
  private templates: Map<string, AgentTemplate> = new Map();
  private startTime?: Date;
  private running: boolean = false;

  constructor(config: Omit<TournamentConfig, "id">) {
    super();

    this.config = TournamentConfigSchema.parse({
      ...config,
      id: uuidv4(),
    });

    // Register templates
    for (const template of PredefinedTemplates) {
      this.templates.set(template.id, template);
    }

    this.emit("initialized", this.config);
  }

  // ============================================================================
  // Tournament Control
  // ============================================================================

  /**
   * Initialize the population
   */
  async initialize(): Promise<void> {
    this.population.clear();
    this.matches = [];
    this.generationResults = [];
    this.currentGeneration = 0;
    this.champion = null;

    // Create initial population
    if (this.config.seedAgents && this.config.seedAgents.length > 0) {
      // Use seed agents
      for (const seed of this.config.seedAgents) {
        const genome = this.createGenomeFromConfig(seed);
        this.population.set(genome.id, genome);
      }
    }

    // Fill remaining population
    while (this.population.size < this.config.populationSize) {
      const genome = this.createRandomGenome();
      this.population.set(genome.id, genome);
    }
  }

  /**
   * Run the tournament
   */
  async run(): Promise<TournamentResults> {
    this.running = true;
    this.startTime = new Date();

    await this.initialize();

    for (let gen = 0; gen < this.config.generations && this.running; gen++) {
      this.currentGeneration = gen;
      this.emit("generationStarted", gen);

      // Run matches for this generation
      const matches = await this.runGeneration();

      // Calculate fitness
      this.calculateFitness();

      // Record generation results
      const genResult = this.recordGenerationResult(matches);
      this.generationResults.push(genResult);
      this.emit("generationCompleted", genResult);

      // Update champion
      const best = this.getBestGenome();
      if (best && (!this.champion || best.fitness > this.champion.fitness)) {
        this.champion = best;
        this.emit("newChampion", best);
      }

      // Evolution step (except last generation)
      if (gen < this.config.generations - 1) {
        await this.evolve();
        this.emit(
          "evolutionStep",
          gen + 1,
          Array.from(this.population.values())
        );
      }
    }

    const results = this.generateResults();
    this.running = false;
    this.emit("completed", results);

    return results;
  }

  /**
   * Stop the tournament
   */
  stop(): void {
    this.running = false;
  }

  // ============================================================================
  // Generation Execution
  // ============================================================================

  private async runGeneration(): Promise<Match[]> {
    const matches: Match[] = [];
    const pairs = this.generateMatchPairs();

    for (const pair of pairs) {
      const match = await this.runMatch(pair);
      matches.push(match);
      this.matches.push(match);
    }

    return matches;
  }

  private generateMatchPairs(): string[][] {
    const genomes = Array.from(this.population.keys());

    switch (this.config.format) {
      case "round_robin":
        return this.generateRoundRobinPairs(genomes);

      case "battle_royale":
        return [genomes]; // All in one match

      case "single_elimination":
        return this.generateSingleEliminationPairs(genomes);

      case "swiss":
        return this.generateSwissPairs(genomes);

      default:
        return this.generateRoundRobinPairs(genomes);
    }
  }

  private generateRoundRobinPairs(genomes: string[]): string[][] {
    const pairs: string[][] = [];
    const limit = this.config.matchesPerGeneration || genomes.length * 2;

    for (let i = 0; i < genomes.length && pairs.length < limit; i++) {
      for (let j = i + 1; j < genomes.length && pairs.length < limit; j++) {
        pairs.push([genomes[i]!, genomes[j]!]);
      }
    }

    return pairs;
  }

  private generateSingleEliminationPairs(genomes: string[]): string[][] {
    // Shuffle and pair up
    const shuffled = [...genomes].sort(() => Math.random() - 0.5);
    const pairs: string[][] = [];

    for (let i = 0; i < shuffled.length - 1; i += 2) {
      pairs.push([shuffled[i]!, shuffled[i + 1]!]);
    }

    return pairs;
  }

  private generateSwissPairs(genomes: string[]): string[][] {
    // Sort by fitness/wins and pair adjacent
    const sorted = genomes.sort((a, b) => {
      const genomeA = this.population.get(a)!;
      const genomeB = this.population.get(b)!;
      return genomeB.fitness - genomeA.fitness;
    });

    const pairs: string[][] = [];
    for (let i = 0; i < sorted.length - 1; i += 2) {
      pairs.push([sorted[i]!, sorted[i + 1]!]);
    }

    return pairs;
  }

  private async runMatch(participants: string[]): Promise<Match> {
    const match: Match = {
      id: uuidv4(),
      generation: this.currentGeneration,
      round: this.matches.filter((m) => m.generation === this.currentGeneration)
        .length,
      participants,
      results: new Map(),
      startTime: new Date(),
    };

    this.emit("matchStarted", match);

    // Create arena
    const arena = new SwarmArena({
      ...this.config.arenaConfig,
      name: `Tournament_G${this.currentGeneration}_M${match.round}`,
    });

    // Create spawner
    const spawner = new AgentSpawner({
      arena,
      templates: Array.from(this.templates.values()),
      defaultMutationRate: 0,
      attributeNoise: 0,
      uniqueNames: true,
    });

    // Spawn agents from genomes
    const arenaConfig = arena.getConfig();
    const spawnRadius =
      Math.min(arenaConfig.dimensions.width, arenaConfig.dimensions.height) *
      0.4;

    for (let i = 0; i < participants.length; i++) {
      const genomeId = participants[i]!;
      const genome = this.population.get(genomeId)!;

      // Spawn position based on index
      const angle = (i / participants.length) * Math.PI * 2;
      const centerX = arenaConfig.dimensions.width / 2;
      const centerY = arenaConfig.dimensions.height / 2;

      const populationConfig: PopulationConfig = {
        templateId: "generalist",
        team: genomeId,
        spawnPattern: {
          type: "cluster",
          count: 3,
          center: {
            x: centerX + Math.cos(angle) * spawnRadius,
            y: centerY + Math.sin(angle) * spawnRadius,
          },
          radius: 3,
        },
      };

      // Override with genome attributes
      const result = spawner.spawnPopulation(populationConfig);
      for (const agent of result.agents) {
        Object.assign(agent.config.attributes, genome.attributes);
      }
    }

    // Run the match
    arena.start();

    // Wait for completion (with timeout)
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        arena.stop();
        resolve();
      }, 30000); // 30 second timeout

      arena.on("completed", () => {
        clearTimeout(timeout);
        resolve();
      });
    });

    const arenaResults = arena.stop();

    // Process results
    match.arenaResults = arenaResults;
    match.endTime = new Date();

    // Determine winner and update genomes
    for (const ranking of arenaResults.rankings) {
      if (ranking.team) {
        match.results.set(ranking.team, ranking);

        const genome = this.population.get(ranking.team);
        if (genome) {
          genome.totalScore += ranking.score;
        }
      }
    }

    // Determine winner (team with highest total score)
    let bestTeam: string | undefined;
    let bestScore = -Infinity;
    for (const [team, ranking] of match.results) {
      if (ranking.score > bestScore) {
        bestScore = ranking.score;
        bestTeam = team;
      }
    }

    if (bestTeam) {
      match.winner = bestTeam;

      for (const participant of participants) {
        const genome = this.population.get(participant);
        if (genome) {
          if (participant === bestTeam) {
            genome.wins++;
          } else if (participants.length === 2) {
            genome.losses++;
          } else {
            genome.draws++;
          }
        }
      }
    }

    this.emit("matchCompleted", match);
    return match;
  }

  // ============================================================================
  // Fitness Calculation
  // ============================================================================

  private calculateFitness(): void {
    const weights = this.config.fitnessWeights;

    for (const genome of this.population.values()) {
      let fitness = 0;

      // Score component
      fitness += genome.totalScore * weights.score;

      // Win rate component
      const totalMatches = genome.wins + genome.losses + genome.draws;
      if (totalMatches > 0) {
        const winRate = genome.wins / totalMatches;
        fitness += winRate * 100 * weights.survival;
      }

      // Additional components from match results
      // (resources and interactions would come from aggregated match data)

      genome.fitness = fitness;
    }
  }

  // ============================================================================
  // Evolution
  // ============================================================================

  private async evolve(): Promise<void> {
    const sortedGenomes = Array.from(this.population.values()).sort(
      (a, b) => b.fitness - a.fitness
    );

    // Selection
    const survivors = this.select(sortedGenomes);

    // Preserve elites
    const elites = sortedGenomes.slice(
      0,
      this.config.selectionConfig.eliteCount
    );

    // Create new population
    const newPopulation = new Map<string, Genome>();

    // Add elites unchanged
    for (const elite of elites) {
      newPopulation.set(elite.id, {
        ...elite,
        generation: this.currentGeneration + 1,
      });
    }

    // Fill rest with offspring
    while (newPopulation.size < this.config.populationSize) {
      if (
        survivors.length >= 2 &&
        Math.random() < this.config.evolutionConfig.crossoverRate
      ) {
        // Crossover
        const parent1 =
          survivors[Math.floor(Math.random() * survivors.length)]!;
        const parent2 =
          survivors[Math.floor(Math.random() * survivors.length)]!;
        const offspring = this.crossover(parent1, parent2);

        // Mutation
        if (Math.random() < this.config.evolutionConfig.mutationRate) {
          this.mutate(offspring);
        }

        newPopulation.set(offspring.id, offspring);
      } else {
        // Clone and mutate
        const parent = survivors[Math.floor(Math.random() * survivors.length)]!;
        const clone = this.clone(parent);
        this.mutate(clone);
        newPopulation.set(clone.id, clone);
      }
    }

    this.population = newPopulation;
  }

  private select(sortedGenomes: Genome[]): Genome[] {
    const count = Math.floor(
      sortedGenomes.length * this.config.selectionConfig.survivalRate
    );

    switch (this.config.selectionConfig.method) {
      case "top_n":
        return sortedGenomes.slice(0, count);

      case "tournament": {
        const selected: Genome[] = [];
        while (selected.length < count) {
          // Tournament of 3
          const contestants = [];
          for (let i = 0; i < 3; i++) {
            contestants.push(
              sortedGenomes[Math.floor(Math.random() * sortedGenomes.length)]!
            );
          }
          contestants.sort((a, b) => b.fitness - a.fitness);
          selected.push(contestants[0]!);
        }
        return selected;
      }

      case "roulette": {
        const totalFitness = sortedGenomes.reduce(
          (s, g) => s + Math.max(0, g.fitness),
          0
        );
        const selected: Genome[] = [];

        while (selected.length < count && totalFitness > 0) {
          let random = Math.random() * totalFitness;
          for (const genome of sortedGenomes) {
            random -= Math.max(0, genome.fitness);
            if (random <= 0) {
              selected.push(genome);
              break;
            }
          }
        }
        return selected;
      }

      case "rank": {
        const selected: Genome[] = [];
        const ranks = sortedGenomes.map((_, i) => sortedGenomes.length - i);
        const totalRank = ranks.reduce((a, b) => a + b, 0);

        while (selected.length < count) {
          let random = Math.random() * totalRank;
          for (let i = 0; i < sortedGenomes.length; i++) {
            random -= ranks[i]!;
            if (random <= 0) {
              selected.push(sortedGenomes[i]!);
              break;
            }
          }
        }
        return selected;
      }

      case "elitist":
        return sortedGenomes.slice(0, count);

      case "random":
        return [...sortedGenomes]
          .sort(() => Math.random() - 0.5)
          .slice(0, count);

      default:
        return sortedGenomes.slice(0, count);
    }
  }

  private crossover(parent1: Genome, parent2: Genome): Genome {
    const offspring: Genome = {
      id: uuidv4(),
      parentIds: [parent1.id, parent2.id],
      generation: this.currentGeneration + 1,
      attributes: { ...parent1.attributes },
      behaviorType: parent1.behaviorType,
      fitness: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      totalScore: 0,
      metadata: {},
    };

    switch (this.config.evolutionConfig.crossoverMethod) {
      case "uniform":
        for (const key of Object.keys(
          offspring.attributes
        ) as (keyof typeof offspring.attributes)[]) {
          if (Math.random() < 0.5) {
            offspring.attributes[key] = parent2.attributes[key];
          }
        }
        break;

      case "single_point": {
        const keys = Object.keys(
          offspring.attributes
        ) as (keyof typeof offspring.attributes)[];
        const point = Math.floor(Math.random() * keys.length);
        for (let i = point; i < keys.length; i++) {
          offspring.attributes[keys[i]!] = parent2.attributes[keys[i]!];
        }
        break;
      }

      case "arithmetic": {
        const alpha = Math.random();
        for (const key of Object.keys(
          offspring.attributes
        ) as (keyof typeof offspring.attributes)[]) {
          offspring.attributes[key] =
            alpha * parent1.attributes[key] +
            (1 - alpha) * parent2.attributes[key];
        }
        break;
      }

      case "blend": {
        const alpha = 0.5;
        for (const key of Object.keys(
          offspring.attributes
        ) as (keyof typeof offspring.attributes)[]) {
          const min = Math.min(
            parent1.attributes[key],
            parent2.attributes[key]
          );
          const max = Math.max(
            parent1.attributes[key],
            parent2.attributes[key]
          );
          const range = max - min;
          offspring.attributes[key] =
            min - alpha * range + Math.random() * (1 + 2 * alpha) * range;
        }
        break;
      }
    }

    return offspring;
  }

  private mutate(genome: Genome): void {
    const strength = this.config.evolutionConfig.mutationStrength;

    for (const key of Object.keys(
      genome.attributes
    ) as (keyof typeof genome.attributes)[]) {
      if (Math.random() < this.config.evolutionConfig.mutationRate) {
        switch (this.config.evolutionConfig.mutationMethod) {
          case "gaussian": {
            const gaussian =
              this.gaussianRandom() * strength * genome.attributes[key];
            genome.attributes[key] = Math.max(
              0,
              genome.attributes[key] + gaussian
            );
            break;
          }

          case "uniform": {
            const uniform =
              (Math.random() - 0.5) * 2 * strength * genome.attributes[key];
            genome.attributes[key] = Math.max(
              0,
              genome.attributes[key] + uniform
            );
            break;
          }

          case "polynomial": {
            const eta = 20;
            const u = Math.random();
            let delta;
            if (u < 0.5) {
              delta = Math.pow(2 * u, 1 / (eta + 1)) - 1;
            } else {
              delta = 1 - Math.pow(2 * (1 - u), 1 / (eta + 1));
            }
            genome.attributes[key] = Math.max(
              0,
              genome.attributes[key] * (1 + delta * strength)
            );
            break;
          }
        }
      }
    }
  }

  private clone(genome: Genome): Genome {
    return {
      id: uuidv4(),
      parentIds: [genome.id],
      generation: this.currentGeneration + 1,
      attributes: { ...genome.attributes },
      behaviorType: genome.behaviorType,
      fitness: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      totalScore: 0,
      metadata: {},
    };
  }

  private gaussianRandom(): number {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // ============================================================================
  // Genome Creation
  // ============================================================================

  private createRandomGenome(): Genome {
    return {
      id: uuidv4(),
      parentIds: [],
      generation: 0,
      attributes: {
        energy: 80 + Math.random() * 40,
        speed: 1 + Math.random() * 2,
        visionRange: 4 + Math.random() * 8,
        strength: 1 + Math.random() * 4,
        intelligence: 1 + Math.random() * 3,
        communication: 1 + Math.random() * 3,
      },
      behaviorType: "rule_based",
      fitness: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      totalScore: 0,
      metadata: {},
    };
  }

  private createGenomeFromConfig(config: Partial<AgentConfig>): Genome {
    const genome = this.createRandomGenome();

    if (config.attributes) {
      Object.assign(genome.attributes, config.attributes);
    }

    if (config.behavior) {
      genome.behaviorType = config.behavior.type;
    }

    return genome;
  }

  // ============================================================================
  // Results
  // ============================================================================

  private recordGenerationResult(matches: Match[]): GenerationResult {
    const population = Array.from(this.population.values());
    const fitnesses = population.map((g) => g.fitness);

    return {
      generation: this.currentGeneration,
      population: population.map((g) => ({ ...g })),
      matches,
      bestFitness: Math.max(...fitnesses),
      avgFitness: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      diversity: this.calculateDiversity(population),
      elites: population
        .sort((a, b) => b.fitness - a.fitness)
        .slice(0, this.config.selectionConfig.eliteCount),
    };
  }

  private calculateDiversity(population: Genome[]): number {
    if (population.length < 2) return 0;

    let totalDistance = 0;
    let pairs = 0;

    for (let i = 0; i < population.length; i++) {
      for (let j = i + 1; j < population.length; j++) {
        totalDistance += this.genomeDistance(population[i]!, population[j]!);
        pairs++;
      }
    }

    return pairs > 0 ? totalDistance / pairs : 0;
  }

  private genomeDistance(a: Genome, b: Genome): number {
    let sumSquares = 0;
    for (const key of Object.keys(
      a.attributes
    ) as (keyof typeof a.attributes)[]) {
      const diff = a.attributes[key] - b.attributes[key];
      sumSquares += diff * diff;
    }
    return Math.sqrt(sumSquares);
  }

  private generateResults(): TournamentResults {
    const champion = this.getBestGenome() || this.createRandomGenome();

    const fitnessOverTime = this.generationResults.map((g) => g.bestFitness);
    const diversityOverTime = this.generationResults.map((g) => g.diversity);

    // Find convergence
    let convergenceGeneration: number | undefined;
    for (let i = 10; i < fitnessOverTime.length; i++) {
      const recent = fitnessOverTime.slice(i - 10, i);
      const variance =
        recent.reduce((sum, f) => sum + Math.pow(f - recent[0]!, 2), 0) /
        recent.length;
      if (variance < 1) {
        convergenceGeneration = i - 10;
        break;
      }
    }

    // Best attributes
    const bestAttributes: Record<string, number> = {};
    for (const key of Object.keys(
      champion.attributes
    ) as (keyof typeof champion.attributes)[]) {
      bestAttributes[key] = champion.attributes[key];
    }

    // Attribute evolution
    const attributeEvolution: Record<string, number[]> = {};
    for (const key of Object.keys(champion.attributes)) {
      attributeEvolution[key] = this.generationResults.map(
        (g) =>
          g.elites.reduce(
            (sum, e) => sum + (e.attributes as Record<string, number>)[key],
            0
          ) / Math.max(g.elites.length, 1)
      );
    }

    return {
      tournamentId: this.config.id,
      config: this.config,
      generations: this.generationResults,
      champion,
      allTimeTop: Array.from(this.population.values())
        .sort((a, b) => b.fitness - a.fitness)
        .slice(0, 10),
      totalMatches: this.matches.length,
      totalDuration: this.startTime ? Date.now() - this.startTime.getTime() : 0,
      evolutionMetrics: {
        fitnessOverTime,
        diversityOverTime,
        convergenceGeneration,
        bestAttributes,
        attributeEvolution,
      },
    };
  }

  private getBestGenome(): Genome | null {
    let best: Genome | null = null;
    for (const genome of this.population.values()) {
      if (!best || genome.fitness > best.fitness) {
        best = genome;
      }
    }
    return best;
  }

  // ============================================================================
  // Accessors
  // ============================================================================

  getConfig(): TournamentConfig {
    return this.config;
  }

  getPopulation(): Genome[] {
    return Array.from(this.population.values());
  }

  getMatches(): Match[] {
    return this.matches;
  }

  getCurrentGeneration(): number {
    return this.currentGeneration;
  }

  getChampion(): Genome | null {
    return this.champion;
  }

  isRunning(): boolean {
    return this.running;
  }
}
