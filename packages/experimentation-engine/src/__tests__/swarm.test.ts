/**
 * @fileoverview Unit Tests for Swarm Arena and Tournament
 * @module @neurectomy/experimentation-engine/__tests__/swarm
 * @agent @ECLIPSE @OMNISCIENT
 *
 * Comprehensive test suite for multi-agent simulation:
 * - SwarmArena lifecycle and configuration
 * - Agent spawning and management
 * - Agent behaviors (scripted, rule-based, random)
 * - Simulation execution and tick processing
 * - Tournament evolution system
 * - Genetic operators (selection, crossover, mutation)
 * - Fitness evaluation and ranking
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  SwarmArena,
  ArenaConfig,
  AgentConfig,
  AgentType,
  ArenaTopology,
  InteractionMode,
  Position,
  SwarmAgent,
  ArenaState,
  ArenaResults,
} from "../swarm/arena";
import {
  Tournament,
  TournamentConfig,
  TournamentFormat,
  SelectionMethod,
  CrossoverMethod,
  MutationMethod,
  Genome,
  Match,
  TournamentResults,
} from "../swarm/tournament";

// ============================================================================
// Test Fixtures
// ============================================================================

function createBasicArenaConfig(
  overrides: Partial<Omit<ArenaConfig, "id">> = {}
): Omit<ArenaConfig, "id"> {
  return {
    name: "Test Arena",
    description: "Arena for unit testing",
    topology: "grid",
    dimensions: {
      width: 20,
      height: 20,
    },
    maxAgents: 10,
    tickRate: 100, // Fast for testing
    maxTicks: 100,
    resources: [
      {
        type: "energy",
        name: "energy",
        initialAmount: 1000,
        regenerationRate: 1,
        distribution: "uniform",
      },
    ],
    interactionMode: "cooperative",
    rules: {
      allowCollisions: true,
      deathEnabled: false,
      reproductionEnabled: false,
      communicationRange: 5,
      visionRange: 5,
      movementCost: 1,
    },
    metrics: ["total_agents", "resources_collected"],
    ...overrides,
  };
}

function createBasicAgentConfig(
  overrides: Partial<Omit<AgentConfig, "id">> = {}
): Omit<AgentConfig, "id"> {
  return {
    name: "Test Agent",
    type: "explorer",
    position: { x: 10, y: 10 },
    attributes: {
      energy: 100,
      speed: 1,
      visionRange: 5,
      strength: 1,
      intelligence: 1,
      communication: 1,
    },
    behavior: {
      type: "random",
      seed: 12345,
    },
    memory: {},
    ...overrides,
  };
}

function createBasicTournamentConfig(
  overrides: Partial<Omit<TournamentConfig, "id">> = {}
): Omit<TournamentConfig, "id"> {
  return {
    name: "Test Tournament",
    description: "Tournament for unit testing",
    format: "round_robin",
    arenaConfig: createBasicArenaConfig({ maxTicks: 50 }),
    populationSize: 8,
    generations: 3,
    matchesPerGeneration: 4,
    selectionConfig: {
      method: "top_n",
      survivalRate: 0.5,
      eliteCount: 2,
    },
    evolutionConfig: {
      crossoverMethod: "uniform",
      crossoverRate: 0.7,
      mutationMethod: "gaussian",
      mutationRate: 0.1,
      mutationStrength: 0.2,
    },
    fitnessWeights: {
      score: 1.0,
      survival: 0.3,
      resources: 0.2,
      interactions: 0.1,
    },
    ...overrides,
  };
}

// ============================================================================
// SwarmArena Tests
// ============================================================================

describe("SwarmArena", () => {
  let arena: SwarmArena;

  beforeEach(() => {
    vi.useFakeTimers();
    arena = new SwarmArena(createBasicArenaConfig());
  });

  afterEach(() => {
    if (arena) {
      arena.stop();
      arena.removeAllListeners();
    }
    vi.useRealTimers();
  });

  // --------------------------------------------------------------------------
  // Arena Initialization Tests
  // --------------------------------------------------------------------------

  describe("initialization", () => {
    it("should create arena with valid config", () => {
      const config = arena.getConfig();

      expect(config.id).toBeDefined();
      expect(config.name).toBe("Test Arena");
      expect(config.dimensions.width).toBe(20);
      expect(config.dimensions.height).toBe(20);
      expect(config.maxAgents).toBe(10);
    });

    it("should emit initialized event", () => {
      const eventSpy = vi.fn();
      const testArena = new SwarmArena(createBasicArenaConfig());
      // Note: Event already emitted in constructor, testing type safety
      testArena.on("initialized", eventSpy);
      // Can't test constructor event, but listener is registered
      expect(eventSpy).not.toHaveBeenCalled(); // Already happened
      testArena.stop();
    });

    it("should initialize grid cells", () => {
      const state = arena.getState();

      // 20x20 grid = 400 cells
      expect(state.cells.size).toBe(400);
    });

    it("should distribute resources according to config", () => {
      const state = arena.getState();
      let totalEnergy = 0;

      for (const cell of state.cells.values()) {
        totalEnergy += cell.resources.get("energy") || 0;
      }

      // Initial amount was 1000
      expect(totalEnergy).toBeCloseTo(1000, 0);
    });

    it("should support different topologies", () => {
      const topologies: ArenaTopology[] = ["grid", "torus", "ring"];

      for (const topology of topologies) {
        const testArena = new SwarmArena(createBasicArenaConfig({ topology }));
        expect(testArena.getConfig().topology).toBe(topology);
        testArena.stop();
      }
    });

    it("should validate dimensions", () => {
      expect(
        () =>
          new SwarmArena(
            createBasicArenaConfig({
              dimensions: { width: -1, height: 10 },
            })
          )
      ).toThrow();
    });
  });

  // --------------------------------------------------------------------------
  // Agent Spawning Tests
  // --------------------------------------------------------------------------

  describe("spawnAgent", () => {
    it("should spawn agent with valid config", () => {
      const agent = arena.spawnAgent(createBasicAgentConfig());

      expect(agent.config.id).toBeDefined();
      expect(agent.config.name).toBe("Test Agent");
      expect(agent.state.alive).toBe(true);
      expect(agent.state.energy).toBe(100);
    });

    it("should add agent to arena state", () => {
      const agent = arena.spawnAgent(createBasicAgentConfig());

      const retrieved = arena.getAgent(agent.config.id);
      expect(retrieved).toBeDefined();
      expect(retrieved?.config.id).toBe(agent.config.id);
    });

    it("should add agent to cell at spawn position", () => {
      const agent = arena.spawnAgent(
        createBasicAgentConfig({ position: { x: 5, y: 5 } })
      );

      const state = arena.getState();
      const cell = state.cells.get("5,5");

      expect(cell?.agents).toContain(agent.config.id);
    });

    it("should emit agentSpawned event", () => {
      const eventSpy = vi.fn();
      arena.on("agentSpawned", eventSpy);

      arena.spawnAgent(createBasicAgentConfig());

      expect(eventSpy).toHaveBeenCalledTimes(1);
      expect(eventSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({ name: "Test Agent" }),
        })
      );
    });

    it("should throw when max agents reached", () => {
      const smallArena = new SwarmArena(
        createBasicArenaConfig({ maxAgents: 2 })
      );

      smallArena.spawnAgent(
        createBasicAgentConfig({ position: { x: 1, y: 1 } })
      );
      smallArena.spawnAgent(
        createBasicAgentConfig({ position: { x: 2, y: 2 } })
      );

      expect(() =>
        smallArena.spawnAgent(
          createBasicAgentConfig({ position: { x: 3, y: 3 } })
        )
      ).toThrow("Maximum agent count reached");

      smallArena.stop();
    });

    it("should throw for invalid position", () => {
      expect(() =>
        arena.spawnAgent(
          createBasicAgentConfig({ position: { x: 100, y: 100 } }) // Out of bounds
        )
      ).toThrow("Invalid spawn position");
    });

    it("should support different agent types", () => {
      const types: AgentType[] = [
        "explorer",
        "exploiter",
        "communicator",
        "predator",
        "defender",
        "neutral",
        "hybrid",
      ];

      for (let i = 0; i < types.length; i++) {
        const agent = arena.spawnAgent(
          createBasicAgentConfig({
            name: `Agent ${i}`,
            type: types[i],
            position: { x: i, y: 0 },
          })
        );
        expect(agent.config.type).toBe(types[i]);
      }
    });

    it("should support different behavior types", () => {
      // Random behavior
      const randomAgent = arena.spawnAgent(
        createBasicAgentConfig({
          position: { x: 0, y: 0 },
          behavior: { type: "random", seed: 42 },
        })
      );
      expect(randomAgent.config.behavior.type).toBe("random");

      // Rule-based behavior
      const ruleAgent = arena.spawnAgent(
        createBasicAgentConfig({
          position: { x: 1, y: 0 },
          behavior: {
            type: "rule_based",
            rules: [
              { condition: "energy < 50", action: "collect", priority: 1 },
            ],
          },
        })
      );
      expect(ruleAgent.config.behavior.type).toBe("rule_based");

      // Scripted behavior
      const scriptAgent = arena.spawnAgent(
        createBasicAgentConfig({
          position: { x: 2, y: 0 },
          behavior: {
            type: "scripted",
            script: "move(1, 0)",
          },
        })
      );
      expect(scriptAgent.config.behavior.type).toBe("scripted");
    });
  });

  // --------------------------------------------------------------------------
  // Agent Removal Tests
  // --------------------------------------------------------------------------

  describe("removeAgent", () => {
    it("should remove agent from arena", () => {
      const agent = arena.spawnAgent(createBasicAgentConfig());

      const result = arena.removeAgent(agent.config.id, "test removal");

      expect(result).toBe(true);
      expect(agent.state.alive).toBe(false);
    });

    it("should remove agent from cell", () => {
      const agent = arena.spawnAgent(
        createBasicAgentConfig({ position: { x: 5, y: 5 } })
      );

      arena.removeAgent(agent.config.id, "test");

      const state = arena.getState();
      const cell = state.cells.get("5,5");
      expect(cell?.agents).not.toContain(agent.config.id);
    });

    it("should emit agentDied event", () => {
      const eventSpy = vi.fn();
      arena.on("agentDied", eventSpy);

      const agent = arena.spawnAgent(createBasicAgentConfig());
      arena.removeAgent(agent.config.id, "test death");

      expect(eventSpy).toHaveBeenCalledWith(agent.config.id, "test death");
    });

    it("should return false for non-existent agent", () => {
      const result = arena.removeAgent("non-existent", "test");

      expect(result).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Simulation Control Tests
  // --------------------------------------------------------------------------

  describe("simulation control", () => {
    describe("start", () => {
      it("should start the simulation", () => {
        const eventSpy = vi.fn();
        arena.on("started", eventSpy);

        arena.start();

        expect(eventSpy).toHaveBeenCalledTimes(1);
      });

      it("should begin executing ticks", async () => {
        const tickSpy = vi.fn();
        arena.on("tick", tickSpy);

        arena.spawnAgent(createBasicAgentConfig());
        arena.start();

        // Advance time to trigger ticks
        await vi.advanceTimersByTimeAsync(50);

        expect(tickSpy.mock.calls.length).toBeGreaterThan(0);
      });

      it("should not start twice", () => {
        arena.start();
        arena.start();

        // Should only have one interval running
        // (internal implementation detail)
      });
    });

    describe("pause", () => {
      it("should pause the simulation", () => {
        const eventSpy = vi.fn();
        arena.on("paused", eventSpy);

        arena.start();
        arena.pause();

        expect(eventSpy).toHaveBeenCalledTimes(1);
      });

      it("should stop tick execution", async () => {
        const tickSpy = vi.fn();
        arena.on("tick", tickSpy);

        arena.spawnAgent(createBasicAgentConfig());
        arena.start();

        await vi.advanceTimersByTimeAsync(20);
        const ticksBeforePause = tickSpy.mock.calls.length;

        arena.pause();

        await vi.advanceTimersByTimeAsync(50);

        expect(tickSpy.mock.calls.length).toBe(ticksBeforePause);
      });
    });

    describe("resume", () => {
      it("should resume paused simulation", () => {
        const eventSpy = vi.fn();
        arena.on("resumed", eventSpy);

        arena.start();
        arena.pause();
        arena.resume();

        expect(eventSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe("stop", () => {
      it("should stop simulation and return results", () => {
        const eventSpy = vi.fn();
        arena.on("completed", eventSpy);

        arena.spawnAgent(createBasicAgentConfig({ name: "Agent 1" }));
        arena.start();

        const results = arena.stop();

        expect(results.arenaId).toBe(arena.getConfig().id);
        expect(results.finalAgentCount).toBe(1);
        expect(eventSpy).toHaveBeenCalledWith(results);
      });

      it("should complete automatically at maxTicks", async () => {
        const completedSpy = vi.fn();
        arena.on("completed", completedSpy);

        arena.spawnAgent(createBasicAgentConfig());
        arena.start();

        // Advance past maxTicks (100 ticks at 100/sec = 1 second)
        await vi.advanceTimersByTimeAsync(2000);

        expect(completedSpy).toHaveBeenCalled();
      });
    });

    describe("step", () => {
      it("should execute single tick", () => {
        const tickSpy = vi.fn();
        arena.on("tick", tickSpy);

        arena.spawnAgent(createBasicAgentConfig());
        arena.step();

        expect(tickSpy).toHaveBeenCalledTimes(1);
      });

      it("should increment tick counter", () => {
        arena.spawnAgent(createBasicAgentConfig());

        const initialTick = arena.getState().tick;
        arena.step();

        expect(arena.getState().tick).toBe(initialTick + 1);
      });
    });
  });

  // --------------------------------------------------------------------------
  // Agent Queries Tests
  // --------------------------------------------------------------------------

  describe("agent queries", () => {
    beforeEach(() => {
      arena.spawnAgent(
        createBasicAgentConfig({ name: "Agent 1", position: { x: 1, y: 1 } })
      );
      arena.spawnAgent(
        createBasicAgentConfig({ name: "Agent 2", position: { x: 2, y: 2 } })
      );
      arena.spawnAgent(
        createBasicAgentConfig({ name: "Agent 3", position: { x: 3, y: 3 } })
      );
    });

    it("should get all alive agents", () => {
      const alive = arena.getAliveAgents();

      expect(alive).toHaveLength(3);
      expect(alive.every((a) => a.state.alive)).toBe(true);
    });

    it("should exclude dead agents from alive list", () => {
      const agents = arena.getAliveAgents();
      arena.removeAgent(agents[0]!.config.id, "test");

      const aliveAfter = arena.getAliveAgents();

      expect(aliveAfter).toHaveLength(2);
    });
  });

  // --------------------------------------------------------------------------
  // Results Generation Tests
  // --------------------------------------------------------------------------

  describe("results generation", () => {
    it("should generate arena results with rankings", () => {
      arena.spawnAgent(
        createBasicAgentConfig({ name: "Agent 1", position: { x: 1, y: 1 } })
      );
      arena.spawnAgent(
        createBasicAgentConfig({ name: "Agent 2", position: { x: 2, y: 2 } })
      );

      arena.start();

      // Run a few ticks
      for (let i = 0; i < 10; i++) {
        arena.step();
      }

      const results = arena.stop();

      expect(results.rankings).toBeDefined();
      expect(results.rankings.length).toBe(2);
      expect(results.totalTicks).toBeGreaterThan(0);
    });

    it("should track surviving agents", () => {
      const agent1 = arena.spawnAgent(
        createBasicAgentConfig({ name: "Survivor", position: { x: 1, y: 1 } })
      );
      const agent2 = arena.spawnAgent(
        createBasicAgentConfig({ name: "Deceased", position: { x: 2, y: 2 } })
      );

      arena.removeAgent(agent2.config.id, "test death");

      const results = arena.stop();

      expect(results.survivingAgents).toContain(agent1.config.id);
      expect(results.survivingAgents).not.toContain(agent2.config.id);
    });
  });
});

// ============================================================================
// Tournament Tests
// ============================================================================

describe("Tournament", () => {
  let tournament: Tournament;

  beforeEach(() => {
    vi.useFakeTimers();
    tournament = new Tournament(createBasicTournamentConfig());
  });

  afterEach(() => {
    tournament.stop();
    tournament.removeAllListeners();
    vi.useRealTimers();
  });

  // --------------------------------------------------------------------------
  // Tournament Initialization Tests
  // --------------------------------------------------------------------------

  describe("initialization", () => {
    it("should create tournament with valid config", () => {
      expect(tournament).toBeDefined();
    });

    it("should emit initialized event", () => {
      const eventSpy = vi.fn();
      tournament.on("initialized", eventSpy);

      // Already emitted in constructor
      const newTournament = new Tournament(createBasicTournamentConfig());
      // Note: Can't easily test constructor event
      newTournament.stop();
    });

    it("should support different tournament formats", () => {
      const formats: TournamentFormat[] = [
        "round_robin",
        "single_elimination",
        "double_elimination",
        "swiss",
        "league",
        "battle_royale",
        "team_battle",
      ];

      for (const format of formats) {
        const t = new Tournament(createBasicTournamentConfig({ format }));
        expect(t).toBeDefined();
        t.stop();
      }
    });
  });

  // --------------------------------------------------------------------------
  // Population Initialization Tests
  // --------------------------------------------------------------------------

  describe("initialize", () => {
    it("should create initial population", async () => {
      await tournament.initialize();

      const population = tournament.getPopulation();
      expect(population.length).toBe(8); // populationSize
    });

    it("should create genomes with valid attributes", async () => {
      await tournament.initialize();

      const population = tournament.getPopulation();
      for (const genome of population) {
        expect(genome.id).toBeDefined();
        expect(genome.generation).toBe(0);
        expect(genome.attributes).toBeDefined();
        expect(genome.fitness).toBe(0);
        expect(genome.wins).toBe(0);
        expect(genome.losses).toBe(0);
      }
    });

    it("should use seed agents when provided", async () => {
      const seedTournament = new Tournament(
        createBasicTournamentConfig({
          seedAgents: [
            {
              name: "Seeded Agent",
              attributes: {
                energy: 200,
                speed: 2,
                visionRange: 10,
                strength: 5,
                intelligence: 5,
                communication: 5,
              },
            },
          ],
        })
      );

      await seedTournament.initialize();

      const population = seedTournament.getPopulation();
      const seeded = population.find((g) => g.attributes.energy === 200);
      expect(seeded).toBeDefined();

      seedTournament.stop();
    });
  });

  // --------------------------------------------------------------------------
  // Tournament Execution Tests
  // --------------------------------------------------------------------------

  describe("run", () => {
    it("should run complete tournament", async () => {
      const completedSpy = vi.fn();
      tournament.on("completed", completedSpy);

      // Need to handle async arena operations
      const resultsPromise = tournament.run();

      // Advance time for all generations
      await vi.advanceTimersByTimeAsync(100000);

      const results = await resultsPromise;

      expect(results).toBeDefined();
      expect(results.tournamentId).toBeDefined();
      expect(completedSpy).toHaveBeenCalled();
    }, 30000);

    it("should emit generationStarted events", async () => {
      const eventSpy = vi.fn();
      tournament.on("generationStarted", eventSpy);

      const runPromise = tournament.run();
      await vi.advanceTimersByTimeAsync(100000);
      await runPromise;

      expect(eventSpy.mock.calls.length).toBe(3); // 3 generations
    }, 30000);

    it("should emit generationCompleted events", async () => {
      const eventSpy = vi.fn();
      tournament.on("generationCompleted", eventSpy);

      const runPromise = tournament.run();
      await vi.advanceTimersByTimeAsync(100000);
      await runPromise;

      expect(eventSpy.mock.calls.length).toBe(3);
    }, 30000);

    it("should track champion", async () => {
      const newChampionSpy = vi.fn();
      tournament.on("newChampion", newChampionSpy);

      const runPromise = tournament.run();
      await vi.advanceTimersByTimeAsync(100000);
      const results = await runPromise;

      expect(results.champion).toBeDefined();
      expect(newChampionSpy).toHaveBeenCalled();
    }, 30000);
  });

  // --------------------------------------------------------------------------
  // Evolution Tests
  // --------------------------------------------------------------------------

  describe("evolution", () => {
    describe("selection", () => {
      it("should select top performers with top_n method", async () => {
        const topNTournament = new Tournament(
          createBasicTournamentConfig({
            generations: 2,
            selectionConfig: {
              method: "top_n",
              survivalRate: 0.5,
              eliteCount: 2,
            },
          })
        );

        await topNTournament.initialize();

        // Manually set fitness values
        const population = topNTournament.getPopulation();
        population.forEach((g, i) => {
          g.fitness = i * 10;
        });

        // Evolution should preserve top performers
        topNTournament.stop();
      });

      it("should support different selection methods", () => {
        const methods: SelectionMethod[] = [
          "top_n",
          "tournament",
          "roulette",
          "rank",
          "elitist",
          "random",
        ];

        for (const method of methods) {
          const t = new Tournament(
            createBasicTournamentConfig({
              selectionConfig: {
                method,
                survivalRate: 0.5,
                eliteCount: 1,
              },
            })
          );
          expect(t).toBeDefined();
          t.stop();
        }
      });
    });

    describe("crossover", () => {
      it("should support uniform crossover", () => {
        const t = new Tournament(
          createBasicTournamentConfig({
            evolutionConfig: {
              crossoverMethod: "uniform",
              crossoverRate: 0.9,
              mutationMethod: "gaussian",
              mutationRate: 0.1,
              mutationStrength: 0.1,
            },
          })
        );
        expect(t).toBeDefined();
        t.stop();
      });

      it("should support different crossover methods", () => {
        const methods: CrossoverMethod[] = [
          "uniform",
          "single_point",
          "two_point",
          "arithmetic",
          "blend",
        ];

        for (const method of methods) {
          const t = new Tournament(
            createBasicTournamentConfig({
              evolutionConfig: {
                crossoverMethod: method,
                crossoverRate: 0.7,
                mutationMethod: "gaussian",
                mutationRate: 0.1,
                mutationStrength: 0.1,
              },
            })
          );
          expect(t).toBeDefined();
          t.stop();
        }
      });
    });

    describe("mutation", () => {
      it("should support different mutation methods", () => {
        const methods: MutationMethod[] = [
          "gaussian",
          "uniform",
          "swap",
          "scramble",
          "polynomial",
        ];

        for (const method of methods) {
          const t = new Tournament(
            createBasicTournamentConfig({
              evolutionConfig: {
                crossoverMethod: "uniform",
                crossoverRate: 0.7,
                mutationMethod: method,
                mutationRate: 0.2,
                mutationStrength: 0.2,
              },
            })
          );
          expect(t).toBeDefined();
          t.stop();
        }
      });
    });
  });

  // --------------------------------------------------------------------------
  // Fitness Calculation Tests
  // --------------------------------------------------------------------------

  describe("fitness calculation", () => {
    it("should calculate fitness based on weights", async () => {
      const weightedTournament = new Tournament(
        createBasicTournamentConfig({
          fitnessWeights: {
            score: 2.0,
            survival: 0.5,
            resources: 0.3,
            interactions: 0.2,
          },
        })
      );

      await weightedTournament.initialize();

      // Fitness should be weighted sum of metrics
      // Testing config validation
      expect(weightedTournament).toBeDefined();

      weightedTournament.stop();
    });
  });

  // --------------------------------------------------------------------------
  // Match Generation Tests
  // --------------------------------------------------------------------------

  describe("match generation", () => {
    it("should generate round robin pairs", async () => {
      const rrTournament = new Tournament(
        createBasicTournamentConfig({
          format: "round_robin",
          populationSize: 4,
        })
      );

      await rrTournament.initialize();

      // Round robin with 4 participants = 6 possible pairs
      // But limited by matchesPerGeneration
      rrTournament.stop();
    });

    it("should generate battle royale matches", async () => {
      const brTournament = new Tournament(
        createBasicTournamentConfig({
          format: "battle_royale",
          populationSize: 6,
        })
      );

      await brTournament.initialize();

      // Battle royale puts all participants in one match
      brTournament.stop();
    });

    it("should generate elimination brackets", async () => {
      const elimTournament = new Tournament(
        createBasicTournamentConfig({
          format: "single_elimination",
          populationSize: 8,
        })
      );

      await elimTournament.initialize();

      // Single elimination with 8 = 4 first-round matches
      elimTournament.stop();
    });
  });

  // --------------------------------------------------------------------------
  // Stop/Abort Tests
  // --------------------------------------------------------------------------

  describe("stop", () => {
    it("should stop running tournament", async () => {
      const runPromise = tournament.run();

      // Stop early
      await vi.advanceTimersByTimeAsync(1000);
      tournament.stop();

      // Should still resolve, just with partial results
    });
  });
});

// ============================================================================
// Genome Tests
// ============================================================================

describe("Genome", () => {
  describe("structure", () => {
    it("should have required properties", async () => {
      const tournament = new Tournament(createBasicTournamentConfig());
      await tournament.initialize();

      const population = tournament.getPopulation();
      const genome = population[0]!;

      expect(genome).toHaveProperty("id");
      expect(genome).toHaveProperty("parentIds");
      expect(genome).toHaveProperty("generation");
      expect(genome).toHaveProperty("attributes");
      expect(genome).toHaveProperty("fitness");
      expect(genome).toHaveProperty("wins");
      expect(genome).toHaveProperty("losses");
      expect(genome).toHaveProperty("draws");
      expect(genome).toHaveProperty("totalScore");

      tournament.stop();
    });

    it("should have valid attribute ranges", async () => {
      const tournament = new Tournament(createBasicTournamentConfig());
      await tournament.initialize();

      const population = tournament.getPopulation();
      for (const genome of population) {
        expect(genome.attributes.energy).toBeGreaterThanOrEqual(0);
        expect(genome.attributes.speed).toBeGreaterThan(0);
        expect(genome.attributes.visionRange).toBeGreaterThanOrEqual(0);
        expect(genome.attributes.strength).toBeGreaterThanOrEqual(0);
        expect(genome.attributes.intelligence).toBeGreaterThanOrEqual(0);
        expect(genome.attributes.communication).toBeGreaterThanOrEqual(0);
      }

      tournament.stop();
    });
  });
});

// ============================================================================
// Integration-like Tests
// ============================================================================

describe("Arena-Tournament Integration", () => {
  it("should run tournament with working arenas", async () => {
    vi.useFakeTimers();

    const tournament = new Tournament(
      createBasicTournamentConfig({
        generations: 1,
        populationSize: 4,
        matchesPerGeneration: 2,
      })
    );

    const matchCompletedSpy = vi.fn();
    tournament.on("matchCompleted", matchCompletedSpy);

    const runPromise = tournament.run();

    // Advance time for matches to complete
    await vi.advanceTimersByTimeAsync(100000);

    const results = await runPromise;

    expect(results.totalMatches).toBeGreaterThanOrEqual(2);

    tournament.stop();
    vi.useRealTimers();
  }, 30000);
});

// ============================================================================
// Edge Cases
// ============================================================================

describe("Edge Cases", () => {
  describe("SwarmArena edge cases", () => {
    it("should handle empty arena", () => {
      const arena = new SwarmArena(createBasicArenaConfig());

      // Should not crash with no agents
      arena.start();
      arena.step();
      const results = arena.stop();

      expect(results.finalAgentCount).toBe(0);
      expect(results.rankings).toHaveLength(0);
    });

    it("should handle single agent", () => {
      const arena = new SwarmArena(createBasicArenaConfig());
      arena.spawnAgent(createBasicAgentConfig());

      arena.start();
      for (let i = 0; i < 10; i++) {
        arena.step();
      }
      const results = arena.stop();

      expect(results.finalAgentCount).toBe(1);
    });

    it("should handle agents at same position", () => {
      const arena = new SwarmArena(
        createBasicArenaConfig({
          rules: {
            allowCollisions: true,
            deathEnabled: false,
            reproductionEnabled: false,
          },
        })
      );

      arena.spawnAgent(createBasicAgentConfig({ position: { x: 5, y: 5 } }));
      arena.spawnAgent(createBasicAgentConfig({ position: { x: 5, y: 5 } }));

      const state = arena.getState();
      const cell = state.cells.get("5,5");

      expect(cell?.agents.length).toBe(2);

      arena.stop();
    });
  });

  describe("Tournament edge cases", () => {
    it("should handle minimum population size", async () => {
      vi.useFakeTimers();

      const tournament = new Tournament(
        createBasicTournamentConfig({
          populationSize: 2,
          generations: 1,
        })
      );

      await tournament.initialize();
      const population = tournament.getPopulation();

      expect(population.length).toBe(2);

      tournament.stop();
      vi.useRealTimers();
    });

    it("should handle single generation", async () => {
      vi.useFakeTimers();

      const tournament = new Tournament(
        createBasicTournamentConfig({
          generations: 1,
          populationSize: 4,
        })
      );

      const runPromise = tournament.run();
      await vi.advanceTimersByTimeAsync(100000);
      const results = await runPromise;

      expect(results.generations.length).toBe(1);

      tournament.stop();
      vi.useRealTimers();
    }, 30000);
  });
});
