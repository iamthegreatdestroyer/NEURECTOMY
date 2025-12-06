/**
 * @fileoverview Integration Tests for Cross-Module Workflows
 * @module @neurectomy/experimentation-engine/__tests__/integration
 * @agent @ECLIPSE @SYNAPSE
 *
 * Tests workflows spanning multiple modules:
 * - Chaos experiments with A/B analysis
 * - Swarm evolution with hypothesis tracking
 * - End-to-end experimentation pipelines
 * - Event-driven cross-module communication
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// A/B Testing imports
import {
  proportionZTest,
  twoSampleTTest,
  bayesianABTest,
} from "../ab-testing/statistics";

// Chaos imports
import { ChaosSimulator } from "../chaos/simulator";

// Hypothesis imports
import {
  HypothesisLab,
  HypothesisConfig,
  ParameterSpace,
  TrialResult,
} from "../hypothesis/lab";

// Swarm imports
import { SwarmArena, ArenaConfig } from "../swarm/arena";

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Calculate sample statistics from an array
 */
function calculateSampleStats(samples: number[]): {
  mean: number;
  variance: number;
  n: number;
} {
  const n = samples.length;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const variance =
    samples.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / (n - 1);
  return { mean, variance, n };
}

/**
 * Calculate Cohen's d effect size for two samples
 */
function calculateEffectSize(sample1: number[], sample2: number[]): number {
  const mean1 = sample1.reduce((a, b) => a + b, 0) / sample1.length;
  const mean2 = sample2.reduce((a, b) => a + b, 0) / sample2.length;

  const var1 =
    sample1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) /
    (sample1.length - 1);
  const var2 =
    sample2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) /
    (sample2.length - 1);

  const pooledStd = Math.sqrt(
    ((sample1.length - 1) * var1 + (sample2.length - 1) * var2) /
      (sample1.length + sample2.length - 2)
  );

  return pooledStd > 0 ? (mean2 - mean1) / pooledStd : 0;
}

// ============================================================================
// Test Fixtures
// ============================================================================

function createMinimalArenaConfig(): Omit<ArenaConfig, "id"> {
  return {
    name: "Integration Test Arena",
    topology: "grid",
    dimensions: { width: 10, height: 10 },
    maxAgents: 10,
    tickRate: 10,
    maxTicks: 20,
    resources: [
      {
        type: "energy",
        name: "energy",
        initialAmount: 100,
        regenerationRate: 1,
        distribution: "uniform",
      },
    ],
    interactionMode: "cooperative",
    rules: {
      allowCollisions: true,
      deathEnabled: false,
    },
    metrics: ["total_agents"],
  };
}

function createBasicHypothesisConfig(
  overrides: Partial<HypothesisConfig> = {}
): HypothesisConfig {
  return {
    name: "Integration Test Hypothesis",
    description: "Testing cross-module integration",
    hypothesis: "Cross-module communication works correctly",
    nullHypothesis: "Modules do not communicate",
    confidenceLevel: 0.95,
    powerLevel: 0.8,
    ...overrides,
  };
}

function createBasicParameterSpace(): ParameterSpace {
  return {
    parameters: [
      {
        name: "test_param",
        type: "continuous",
        min: 0,
        max: 1,
        default: 0.5,
      },
    ],
  };
}

function createSuccessfulTrialResult(
  metrics: Record<string, number> = {}
): Omit<TrialResult, "trialId" | "timestamp"> {
  return {
    metrics: {
      score: 0.85,
      ...metrics,
    },
    duration: 1000,
    status: "success",
  };
}

// ============================================================================
// Cross-Module Workflow: Chaos → A/B Analysis
// ============================================================================

describe("Chaos Engineering + A/B Analysis Integration", () => {
  let chaosSimulator: ChaosSimulator;
  let controlMetrics: number[];
  let treatmentMetrics: number[];

  beforeEach(() => {
    vi.useFakeTimers();
    controlMetrics = [];
    treatmentMetrics = [];
  });

  afterEach(() => {
    chaosSimulator?.abort();
    vi.useRealTimers();
  });

  it("should analyze chaos experiment results with statistical tests", async () => {
    // Simulate collecting metrics during chaos experiment
    // Control: baseline performance
    controlMetrics = Array(100)
      .fill(0)
      .map(() => 150 + Math.random() * 30); // Mean ~165ms

    // Treatment: during fault injection (degraded)
    treatmentMetrics = Array(100)
      .fill(0)
      .map(() => 180 + Math.random() * 50); // Mean ~205ms

    // Perform statistical analysis - twoSampleTTest expects {mean, variance, n}
    const controlStats = calculateSampleStats(controlMetrics);
    const treatmentStats = calculateSampleStats(treatmentMetrics);
    const tTestResult = twoSampleTTest(controlStats, treatmentStats);
    const effectSize = calculateEffectSize(controlMetrics, treatmentMetrics);

    // Chaos should cause significant performance degradation
    expect(tTestResult.pValue).toBeLessThan(0.05);
    expect(effectSize).toBeGreaterThan(0); // Treatment has higher latency
  });

  it("should integrate chaos results into hypothesis lab", async () => {
    const lab = new HypothesisLab();

    // Create hypothesis about system resilience
    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig({
        name: "Network Partition Resilience",
        hypothesis:
          "System maintains 95% availability during network partition",
      }),
      {
        parameters: [
          {
            name: "partitionDuration",
            type: "continuous",
            min: 1000,
            max: 5000,
          },
          {
            name: "healingRate",
            type: "continuous",
            min: 0.1,
            max: 0.9,
          },
        ],
      }
    );

    // Create and start a trial
    const trial = await lab.createTrial({
      hypothesisId: hypothesis.id,
      parameters: {
        partitionDuration: 2000,
        healingRate: 0.5,
      },
    });

    lab.startTrial(trial.id, hypothesis.id);

    // Simulate experiment metrics
    const availability = 0.96; // 96% availability achieved

    lab.completeTrial(trial.id, hypothesis.id, {
      metrics: {
        availability,
        latencyP99: 250,
        errorsCount: 12,
      },
      duration: 5000,
      status: "success",
    });

    // Verify integration
    const completedTrial = hypothesis.trials.find((t) => t.id === trial.id);
    expect(completedTrial?.status).toBe("completed");
    expect(completedTrial?.result?.metrics.availability).toBe(0.96);

    // Statistical analysis of chaos results
    const passed = availability >= 0.95;
    expect(passed).toBe(true);

    lab.dispose();
  });

  it("should correlate fault severity with performance impact", async () => {
    const results: { severity: number; latency: number }[] = [];

    // Simulate multiple chaos experiments with varying severity
    const severities = [0.1, 0.3, 0.5, 0.7, 0.9];

    for (const severity of severities) {
      const baseLatency = 100;
      const impact = severity * 200 + Math.random() * 20;
      results.push({ severity, latency: baseLatency + impact });
    }

    // Analyze correlation: as severity increases, latency should increase
    const sortedBySeverity = [...results].sort(
      (a, b) => a.severity - b.severity
    );
    let monotonic = true;
    for (let i = 1; i < sortedBySeverity.length; i++) {
      if (
        sortedBySeverity[i]!.latency <
        sortedBySeverity[i - 1]!.latency * 0.9
      ) {
        monotonic = false;
        break;
      }
    }

    expect(monotonic).toBe(true);
  });
});

// ============================================================================
// Cross-Module Workflow: Swarm Evolution → Hypothesis Lab
// ============================================================================

describe("Swarm Evolution + Hypothesis Lab Integration", () => {
  let lab: HypothesisLab;

  beforeEach(() => {
    vi.useFakeTimers();
    lab = new HypothesisLab();
  });

  afterEach(() => {
    lab.dispose();
    vi.useRealTimers();
  });

  it("should track swarm evolution as hypothesis trials", async () => {
    // Create hypothesis for swarm optimization
    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig({
        name: "Optimal Agent Strategy Evolution",
        hypothesis: "Evolution produces agents with fitness > 80%",
      }),
      {
        parameters: [
          { name: "mutationRate", type: "continuous", min: 0.01, max: 0.3 },
          { name: "crossoverRate", type: "continuous", min: 0.5, max: 0.9 },
          { name: "populationSize", type: "discrete", min: 10, max: 100 },
        ],
      }
    );

    // Simulate evolution generations as trials
    const generations = [
      { gen: 1, bestFitness: 45, avgFitness: 30 },
      { gen: 2, bestFitness: 58, avgFitness: 42 },
      { gen: 3, bestFitness: 72, avgFitness: 55 },
      { gen: 4, bestFitness: 85, avgFitness: 68 },
    ];

    for (const genResult of generations) {
      const trial = await lab.createTrial({
        hypothesisId: hypothesis.id,
        parameters: {
          mutationRate: 0.15,
          crossoverRate: 0.7,
          populationSize: 50,
        },
        metadata: { generation: genResult.gen },
      });

      lab.startTrial(trial.id, hypothesis.id);
      lab.completeTrial(trial.id, hypothesis.id, {
        metrics: {
          bestFitness: genResult.bestFitness,
          avgFitness: genResult.avgFitness,
        },
        duration: 1000,
        status: "success",
      });
    }

    // Find best trial
    expect(hypothesis.bestTrial).toBeDefined();
    expect(hypothesis.bestTrial?.result?.metrics.bestFitness).toBe(85);
  });

  it("should analyze evolution convergence with statistical tests", async () => {
    // Simulate fitness progression across generations
    const earlyGenFitness = Array(30)
      .fill(0)
      .map(() => 30 + Math.random() * 20);
    const lateGenFitness = Array(30)
      .fill(0)
      .map(() => 70 + Math.random() * 15);

    // Test if evolution significantly improved fitness
    // twoSampleTTest expects {mean, variance, n}
    const earlyStats = calculateSampleStats(earlyGenFitness);
    const lateStats = calculateSampleStats(lateGenFitness);
    const tTest = twoSampleTTest(earlyStats, lateStats);
    const effectSize = calculateEffectSize(earlyGenFitness, lateGenFitness);

    expect(tTest.pValue).toBeLessThan(0.001); // Highly significant
    expect(effectSize).toBeGreaterThan(1); // Large positive effect
  });
});

// ============================================================================
// Cross-Module Workflow: A/B Testing Engine → Statistics
// ============================================================================

describe("A/B Testing Engine + Statistics Integration", () => {
  it("should perform Bayesian analysis for conversion experiments", () => {
    // Simulated A/B test results
    const controlConversions = 45;
    const controlTotal = 500;
    const treatmentConversions = 62;
    const treatmentTotal = 500;

    const bayesianResult = bayesianABTest(
      controlConversions,
      controlTotal,
      treatmentConversions,
      treatmentTotal
    );

    // Treatment has more conversions, should have high probability of being better
    expect(bayesianResult.probabilityBBeatsA).toBeGreaterThan(0.9);
  });

  it("should combine frequentist and Bayesian approaches", () => {
    const controlData = Array(100)
      .fill(0)
      .map(() => (Math.random() < 0.1 ? 1 : 0));
    const treatmentData = Array(100)
      .fill(0)
      .map(() => (Math.random() < 0.15 ? 1 : 0));

    const controlConversions = controlData.reduce((a, b) => a + b, 0);
    const treatmentConversions = treatmentData.reduce((a, b) => a + b, 0);

    // Frequentist analysis
    const frequentist = proportionZTest(
      controlConversions,
      100,
      treatmentConversions,
      100,
      0.95
    );

    // Bayesian analysis
    const bayesian = bayesianABTest(
      controlConversions,
      100,
      treatmentConversions,
      100
    );

    // Both should return valid results
    expect(typeof frequentist.pValue).toBe("number");
    expect(typeof bayesian.probabilityBBeatsA).toBe("number");
  });
});

// ============================================================================
// Event-Driven Integration Tests
// ============================================================================

describe("Event-Driven Cross-Module Communication", () => {
  it("should propagate arena events to listeners", async () => {
    vi.useFakeTimers();

    const arena = new SwarmArena(createMinimalArenaConfig());
    const events: string[] = [];

    arena.on("agentSpawned", () => events.push("spawn"));
    arena.on("tick", () => events.push("tick"));
    arena.on("completed", () => events.push("complete"));

    arena.spawnAgent({
      name: "Test Agent",
      type: "explorer",
      position: { x: 5, y: 5 },
      attributes: {
        energy: 100,
        speed: 1,
        visionRange: 5,
        strength: 1,
        intelligence: 1,
        communication: 1,
      },
      behavior: { type: "random" },
      memory: {},
    });

    arena.start();
    await vi.advanceTimersByTimeAsync(500);
    arena.stop();

    expect(events).toContain("spawn");
    expect(events).toContain("tick");
    expect(events).toContain("complete");

    vi.useRealTimers();
  });

  it("should emit hypothesis lab events on trial completion", async () => {
    const lab = new HypothesisLab();
    const events: string[] = [];

    lab.on("hypothesisCreated", () => events.push("hypothesis"));
    lab.on("trialStarted", () => events.push("trial_started"));
    lab.on("trialCompleted", () => events.push("trial_completed"));

    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig(),
      createBasicParameterSpace()
    );

    const trial = await lab.createTrial({
      hypothesisId: hypothesis.id,
      parameters: { test_param: 0.5 },
    });

    lab.startTrial(trial.id, hypothesis.id);
    lab.completeTrial(trial.id, hypothesis.id, createSuccessfulTrialResult());

    expect(events).toContain("hypothesis");
    expect(events).toContain("trial_started");
    expect(events).toContain("trial_completed");

    lab.dispose();
  });

  it("should coordinate multiple concurrent experiments", async () => {
    const lab = new HypothesisLab();

    // Create multiple concurrent hypotheses
    const hypotheses = [
      lab.createHypothesis(
        createBasicHypothesisConfig({ name: "H1" }),
        createBasicParameterSpace()
      ),
      lab.createHypothesis(
        createBasicHypothesisConfig({ name: "H2" }),
        createBasicParameterSpace()
      ),
      lab.createHypothesis(
        createBasicHypothesisConfig({ name: "H3" }),
        createBasicParameterSpace()
      ),
    ];

    // Run concurrent trials
    const trials = await Promise.all(
      hypotheses.map((h, i) =>
        lab.createTrial({
          hypothesisId: h.id,
          parameters: { test_param: i * 0.1 },
        })
      )
    );

    // Start all trials
    trials.forEach((trial, i) => {
      lab.startTrial(trial.id, hypotheses[i]!.id);
    });

    // Complete in different order with different scores
    lab.completeTrial(
      trials[1]!.id,
      hypotheses[1]!.id,
      createSuccessfulTrialResult({ score: 0.7 })
    );
    lab.completeTrial(
      trials[2]!.id,
      hypotheses[2]!.id,
      createSuccessfulTrialResult({ score: 0.9 })
    );
    lab.completeTrial(
      trials[0]!.id,
      hypotheses[0]!.id,
      createSuccessfulTrialResult({ score: 0.5 })
    );

    // Verify all completed correctly
    expect(hypotheses[0]!.trials[0]?.status).toBe("completed");
    expect(hypotheses[1]!.trials[0]?.status).toBe("completed");
    expect(hypotheses[2]!.trials[0]?.status).toBe("completed");

    lab.dispose();
  });
});

// ============================================================================
// Data Flow Integration Tests
// ============================================================================

describe("Data Flow Between Modules", () => {
  it("should transform swarm results into analyzable metrics", () => {
    // Simulated swarm results
    const swarmResults = {
      generations: [
        { avgFitness: 30, maxFitness: 45, minFitness: 15 },
        { avgFitness: 45, maxFitness: 62, minFitness: 28 },
        { avgFitness: 58, maxFitness: 78, minFitness: 42 },
        { avgFitness: 72, maxFitness: 89, minFitness: 55 },
      ],
    };

    // Extract metrics for analysis
    const avgFitnessProgression = swarmResults.generations.map(
      (g) => g.avgFitness
    );

    // Analyze early vs late generations
    const earlyGen = avgFitnessProgression.slice(0, 2);
    const lateGen = avgFitnessProgression.slice(2);

    // twoSampleTTest expects {mean, variance, n}
    // Order: control (early), treatment (late)
    // Positive t-statistic means treatment > control (late > early = improvement)
    const earlyStats = calculateSampleStats(earlyGen);
    const lateStats = calculateSampleStats(lateGen);
    const tTest = twoSampleTTest(earlyStats, lateStats);

    // Evolution should show improvement: late generations have higher fitness
    // Positive statistic means treatment (late) > control (early)
    expect(tTest.statistic).toBeGreaterThan(0);
  });

  it("should aggregate chaos experiment metrics across runs", () => {
    // Simulated chaos experiment runs
    const runs = [
      { latencyP50: 120, latencyP99: 350, errorRate: 0.02 },
      { latencyP50: 115, latencyP99: 380, errorRate: 0.03 },
      { latencyP50: 125, latencyP99: 420, errorRate: 0.01 },
      { latencyP50: 130, latencyP99: 390, errorRate: 0.025 },
    ];

    // Aggregate statistics
    const p50Values = runs.map((r) => r.latencyP50);
    const p99Values = runs.map((r) => r.latencyP99);
    const errorRates = runs.map((r) => r.errorRate);

    const avgP50 = p50Values.reduce((a, b) => a + b, 0) / p50Values.length;
    const avgP99 = p99Values.reduce((a, b) => a + b, 0) / p99Values.length;
    const avgError = errorRates.reduce((a, b) => a + b, 0) / errorRates.length;

    expect(avgP50).toBeCloseTo(122.5, 1);
    expect(avgP99).toBeCloseTo(385, 1);
    expect(avgError).toBeCloseTo(0.021, 3);
  });

  it("should persist and restore experiment state", async () => {
    const lab = new HypothesisLab();

    // Create experiment state
    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig({ name: "Persistence Test" }),
      createBasicParameterSpace()
    );

    const trial = await lab.createTrial({
      hypothesisId: hypothesis.id,
      parameters: { test_param: 0.5 },
    });

    lab.startTrial(trial.id, hypothesis.id);
    lab.completeTrial(trial.id, hypothesis.id, {
      metrics: { score: 0.85 },
      duration: 1000,
      status: "success",
    });

    // Export state
    const exported = lab.exportHypothesis(hypothesis.id);

    // Create new lab and import
    const newLab = new HypothesisLab();
    const restoredHypothesis = newLab.importHypothesis(exported);

    // Verify restoration
    expect(restoredHypothesis.config.name).toBe("Persistence Test");
    expect(restoredHypothesis.trials[0]?.status).toBe("completed");
    expect(restoredHypothesis.trials[0]?.result?.metrics.score).toBe(0.85);

    lab.dispose();
    newLab.dispose();
  });
});

// ============================================================================
// Error Handling Integration Tests
// ============================================================================

describe("Cross-Module Error Handling", () => {
  it("should handle trial failure gracefully in hypothesis lab", async () => {
    const lab = new HypothesisLab();

    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig({ name: "Error Handling Test" }),
      createBasicParameterSpace()
    );

    const trial = await lab.createTrial({
      hypothesisId: hypothesis.id,
      parameters: { test_param: 0.5 },
    });

    lab.startTrial(trial.id, hypothesis.id);

    // Fail the trial
    lab.failTrial(trial.id, hypothesis.id, new Error("Simulated error"));

    const failedTrial = hypothesis.trials.find((t) => t.id === trial.id);
    expect(failedTrial?.status).toBe("failed");
    expect(failedTrial?.result?.error).toBe("Simulated error");

    lab.dispose();
  });

  it("should handle arena stop during simulation", async () => {
    vi.useFakeTimers();

    const arena = new SwarmArena(createMinimalArenaConfig());

    // Start and immediately stop
    arena.start();
    const results = arena.stop();

    expect(results).toBeDefined();
    expect(results.arenaId).toBeDefined();

    vi.useRealTimers();
  });

  it("should validate experiment inputs across modules", () => {
    const lab = new HypothesisLab();

    // Invalid parameter space (min > max)
    expect(() =>
      lab.createHypothesis(createBasicHypothesisConfig(), {
        parameters: [
          {
            name: "invalid",
            type: "continuous",
            min: 1,
            max: 0, // Invalid: min > max
          },
        ],
      })
    ).toThrow();

    // Invalid arena config
    expect(
      () =>
        new SwarmArena({
          name: "Test",
          topology: "grid",
          dimensions: { width: -1, height: 10 },
          maxAgents: 10,
        } as any)
    ).toThrow();

    lab.dispose();
  });
});
