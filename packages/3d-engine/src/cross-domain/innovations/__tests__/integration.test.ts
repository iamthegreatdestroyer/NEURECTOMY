/**
 * @fileoverview Integration Tests for Cross-Domain Innovations
 * @module @neurectomy/3d-engine/cross-domain/innovations/__tests__/integration
 * @agent @ECLIPSE @NEXUS @GENESIS
 *
 * Comprehensive tests for all 15 cross-domain innovations:
 * - Foundation (Tasks 1-2): Type system & Event bridge
 * - Forge×Twin (Tasks 3-5): Replay Theater, Predictive Cascade, Consciousness Heatmaps
 * - Twin×Foundry (Tasks 6-8): Architecture Search, Model Sync, Cascade Training
 * - Forge×Foundry (Tasks 9-11): Neural Playground, Training Journey, Model Router
 * - P0 Breakthroughs (Tasks 12-15): Living Architecture, Morphogenic Evolution, Causal Debugger, Quantum Search
 *
 * Tests verify:
 * 1. Event bridge communication between all modules
 * 2. Orchestrator coordination of cross-domain operations
 * 3. Integration between innovations
 * 4. End-to-end workflows
 * 5. Performance and reliability
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Mock TensorFlow.js to avoid import errors in test environment
vi.mock("@tensorflow/tfjs", () => ({
  default: {},
  tensor: vi.fn(),
  sequential: vi.fn(() => ({
    add: vi.fn(),
    compile: vi.fn(),
    fit: vi.fn(),
    predict: vi.fn(),
  })),
  layers: {
    dense: vi.fn(),
    dropout: vi.fn(),
    conv2d: vi.fn(),
  },
}));

// Foundation imports
import { CrossDomainEventBridge } from "../../event-bridge";
import { CrossDomainOrchestrator } from "../../orchestrator";
import { ForgeAdapter, TwinAdapter, FoundryAdapter } from "../../adapters";

// Forge×Twin innovations
import {
  TemporalTwinReplayTheater,
  createReplayTheater,
} from "../replay-theater";
import {
  PredictiveVisualizationCascade,
  createPredictiveCascade,
} from "../predictive-cascade";
import {
  ConsciousnessHeatmapGenerator,
  createConsciousnessHeatmap,
} from "../consciousness-heatmaps";

// Twin×Foundry innovations
import {
  TwinGuidedArchitectureSearch,
  createArchitectureSearch,
} from "../architecture-search";
import { ModelInLoopSync, createModelSync } from "../model-sync";
import {
  CascadeAwareTraining,
  createCascadeTraining,
} from "../cascade-training";

// Forge×Foundry innovations
import {
  Neural3DPlayground,
  createNeural3DPlayground,
} from "../forge-foundry/playground-neural-3d";
import {
  Training4DJourney,
  createTraining4DJourney,
} from "../forge-foundry/training-4d-journey";
import {
  ModelRouterCosmos,
  createModelRouterCosmos,
} from "../forge-foundry/model-router-cosmos";

// P0 Breakthrough innovations
import {
  LivingArchitectureLaboratory,
  createLivingArchitectureLab,
} from "../breakthroughs/living-architecture-lab";
import {
  MorphogenicModelEvolution,
  createMorphogenicEvolution,
} from "../breakthroughs/morphogenic-evolution";
import {
  CausalTrainingDebugger,
  createCausalDebugger,
} from "../breakthroughs/causal-training-debugger";
import {
  QuantumArchitectureSearch,
  createQuantumArchitectureSearch,
  quickQuantumSearch,
} from "../breakthroughs/quantum-architecture-search";

// ============================================================================
// Test Fixtures & Utilities
// ============================================================================

/**
 * Create mock TwinManager for testing
 */
function createMockTwinManager() {
  return {
    getTwinState: vi.fn().mockResolvedValue({
      id: "test-twin",
      state: { position: [0, 0, 0], velocity: [1, 0, 0] },
      timestamp: Date.now(),
    }),
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
  };
}

/**
 * Create mock TimelineNavigator for testing
 */
function createMockTimelineNavigator() {
  return {
    getCurrentTime: vi.fn().mockReturnValue(Date.now()),
    getScene: vi.fn().mockReturnValue({ add: vi.fn(), remove: vi.fn() }),
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
  };
}

/**
 * Create mock training service for testing
 */
function createMockTrainingService() {
  return {
    startTraining: vi.fn().mockResolvedValue({ success: true }),
    getTrainingMetrics: vi.fn().mockResolvedValue({
      epoch: 10,
      loss: 0.5,
      accuracy: 0.85,
    }),
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
  };
}

/**
 * Wait for async operations
 */
async function waitFor(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait for event to be emitted
 */
async function waitForEvent(
  emitter: any,
  eventName: string,
  timeout = 1000
): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout waiting for event: ${eventName}`));
    }, timeout);

    emitter.once(eventName, (data: any) => {
      clearTimeout(timer);
      resolve(data);
    });
  });
}

// ============================================================================
// Foundation Tests (Tasks 1-2)
// ============================================================================

describe("Foundation: Cross-Domain Infrastructure", () => {
  let eventBridge: CrossDomainEventBridge;
  let orchestrator: CrossDomainOrchestrator;

  beforeEach(() => {
    eventBridge = CrossDomainEventBridge.getInstance();
    orchestrator = CrossDomainOrchestrator.getInstance();
  });

  afterEach(() => {
    eventBridge.removeAllListeners();
  });

  describe("Event Bridge", () => {
    it("should create singleton instance", () => {
      const instance1 = CrossDomainEventBridge.getInstance();
      const instance2 = CrossDomainEventBridge.getInstance();
      expect(instance1).toBe(instance2);
    });

    it("should publish and subscribe to events", async () => {
      const handler = vi.fn();
      const unsubscribe = eventBridge.subscribe("twin:state:updated", handler);

      eventBridge.publish("twin:state:updated", {
        twinId: "test",
        state: { x: 1 },
      });

      await waitFor(10);
      expect(handler).toHaveBeenCalledWith({
        twinId: "test",
        state: { x: 1 },
      });

      unsubscribe();
    });

    it("should support wildcard subscriptions", async () => {
      const handler = vi.fn();
      const unsubscribe = eventBridge.subscribe("twin:*", handler);

      eventBridge.publish("twin:state:updated", { data: 1 });
      eventBridge.publish("twin:prediction:generated", { data: 2 });

      await waitFor(10);
      expect(handler).toHaveBeenCalledTimes(2);

      unsubscribe();
    });

    it("should handle multiple subscribers", async () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      eventBridge.subscribe("test:event", handler1);
      eventBridge.subscribe("test:event", handler2);

      eventBridge.publish("test:event", { value: 123 });

      await waitFor(10);
      expect(handler1).toHaveBeenCalled();
      expect(handler2).toHaveBeenCalled();
    });
  });

  describe("Orchestrator", () => {
    it("should create singleton instance", () => {
      const instance1 = CrossDomainOrchestrator.getInstance();
      const instance2 = CrossDomainOrchestrator.getInstance();
      expect(instance1).toBe(instance2);
    });

    it("should coordinate cross-domain operations", async () => {
      const operation = orchestrator.coordinate({
        id: "test-op",
        type: "twin-prediction",
        domains: ["twin", "forge"],
        payload: { twinId: "test" },
      });

      expect(operation.id).toBe("test-op");
      expect(operation.status).toBe("pending");
    });
  });
});

// ============================================================================
// Forge×Twin Tests (Tasks 3-5)
// ============================================================================

describe("Forge×Twin Innovations", () => {
  let mockTwinManager: any;
  let mockTimelineNavigator: any;

  beforeEach(() => {
    mockTwinManager = createMockTwinManager();
    mockTimelineNavigator = createMockTimelineNavigator();
  });

  describe("Temporal Twin Replay Theater (Task 3)", () => {
    it("should create replay theater instance", () => {
      const theater = createReplayTheater(["twin-1", "twin-2"]);
      expect(theater).toBeDefined();
    });

    it("should capture twin snapshots", async () => {
      const theater = new TemporalTwinReplayTheater(
        mockTwinManager,
        mockTimelineNavigator
      );

      const snapshot = await theater.captureSnapshot("twin-1");
      expect(snapshot).toHaveProperty("twinId", "twin-1");
      expect(snapshot).toHaveProperty("timestamp");
      expect(snapshot).toHaveProperty("state");
    });

    it("should start and stop replay sessions", async () => {
      const theater = new TemporalTwinReplayTheater(
        mockTwinManager,
        mockTimelineNavigator
      );

      const sessionId = await theater.startReplay("twin-1", {
        startTime: Date.now() - 1000,
        endTime: Date.now(),
        playbackSpeed: 1.0,
      });

      expect(sessionId).toBeDefined();

      await theater.stopReplay(sessionId);
    });

    it("should emit replay events", async () => {
      const theater = new TemporalTwinReplayTheater(
        mockTwinManager,
        mockTimelineNavigator
      );

      const eventPromise = waitForEvent(theater, "replay-started");

      const sessionId = await theater.startReplay("twin-1", {
        startTime: Date.now() - 1000,
        endTime: Date.now(),
        playbackSpeed: 1.0,
      });

      const event = await eventPromise;
      expect(event.sessionId).toBe(sessionId);
    });
  });

  describe("Predictive Visualization Cascade (Task 4)", () => {
    it("should create predictive cascade instance", () => {
      const cascade = createPredictiveCascade(["twin-1"]);
      expect(cascade).toBeDefined();
    });

    it("should generate predictions", async () => {
      const cascade = new PredictiveVisualizationCascade(
        mockTwinManager,
        mockTimelineNavigator
      );

      const prediction = await cascade.generatePrediction("twin-1", {
        horizonMs: 1000,
        confidence: 0.8,
      });

      expect(prediction).toHaveProperty("twinId", "twin-1");
      expect(prediction).toHaveProperty("predictions");
    });

    it("should create branching timelines", async () => {
      const cascade = new PredictiveVisualizationCascade(
        mockTwinManager,
        mockTimelineNavigator
      );

      await cascade.generatePrediction("twin-1", { horizonMs: 1000 });
      const visualization = cascade.getVisualization("twin-1");

      expect(visualization).toBeDefined();
      expect(visualization.branches.length).toBeGreaterThan(0);
    });
  });

  describe("Consciousness-Aware Heatmaps (Task 5)", () => {
    it("should create heatmap generator instance", () => {
      const generator = createConsciousnessHeatmap("entity-1");
      expect(generator).toBeDefined();
    });

    it("should generate attention patterns", async () => {
      const generator = ConsciousnessHeatmapGenerator.getInstance();
      generator.startMonitoring("entity-1");

      await waitFor(100);

      const heatmap = generator.getHeatmap("entity-1");
      expect(heatmap).toBeDefined();
    });

    it("should track consciousness states", async () => {
      const generator = ConsciousnessHeatmapGenerator.getInstance();
      generator.startMonitoring("entity-1");

      await waitFor(100);

      const state = generator.getConsciousnessState("entity-1");
      expect(state).toHaveProperty("awarenessLevel");
      expect(state).toHaveProperty("focusPoints");
    });
  });
});

// ============================================================================
// Twin×Foundry Tests (Tasks 6-8)
// ============================================================================

describe("Twin×Foundry Innovations", () => {
  let mockTwinManager: any;
  let mockTrainingService: any;

  beforeEach(() => {
    mockTwinManager = createMockTwinManager();
    mockTrainingService = createMockTrainingService();
  });

  describe("Twin-Guided Architecture Search (Task 6)", () => {
    it("should create architecture search instance", () => {
      const search = createArchitectureSearch();
      expect(search).toBeDefined();
    });

    it("should propose architecture changes", async () => {
      const search = new TwinGuidedArchitectureSearch(
        mockTwinManager,
        mockTrainingService
      );

      search.startSearch("model-1", {
        maxLayers: 10,
        targetAccuracy: 0.9,
      });

      await waitFor(100);

      const proposal = await search.proposeArchitectureChange("model-1");
      expect(proposal).toBeDefined();
      expect(proposal).toHaveProperty("architectureId");
    });
  });

  describe("Model-in-Loop Sync (Task 7)", () => {
    it("should create model sync instance", () => {
      const sync = createModelSync();
      expect(sync).toBeDefined();
    });

    it("should sync model with twin state", async () => {
      const sync = new ModelInLoopSync(mockTwinManager, mockTrainingService);

      await sync.startSync("twin-1", "model-1");
      await waitFor(100);

      const validation = await sync.validatePrediction("twin-1", "model-1");
      expect(validation).toHaveProperty("isValid");
    });
  });

  describe("Cascade-Aware Training (Task 8)", () => {
    it("should create cascade training instance", () => {
      const training = createCascadeTraining();
      expect(training).toBeDefined();
    });

    it("should simulate cascade effects", async () => {
      const training = new CascadeAwareTraining(
        mockTwinManager,
        mockTrainingService
      );

      const simulation = await training.simulateCascade("model-1", {
        initialState: { x: 1, y: 2 },
        steps: 10,
      });

      expect(simulation).toBeDefined();
      expect(simulation.cascadeSteps).toHaveLength(10);
    });
  });
});

// ============================================================================
// Forge×Foundry Tests (Tasks 9-11)
// ============================================================================

describe("Forge×Foundry Innovations", () => {
  let mockTimelineNavigator: any;
  let mockTrainingService: any;

  beforeEach(() => {
    mockTimelineNavigator = createMockTimelineNavigator();
    mockTrainingService = createMockTrainingService();
  });

  describe("3D Neural Playground (Task 9)", () => {
    it("should create playground instance", () => {
      const playground = createNeural3DPlayground();
      expect(playground).toBeDefined();
    });

    it("should create neural network", async () => {
      const playground = new Neural3DPlayground(mockTimelineNavigator);

      const network = await playground.createNetwork({
        layers: [
          { type: "dense", units: 128 },
          { type: "dense", units: 64 },
          { type: "dense", units: 10 },
        ],
      });

      expect(network).toBeDefined();
      expect(network.layers).toHaveLength(3);
    });

    it("should train network interactively", async () => {
      const playground = new Neural3DPlayground(mockTimelineNavigator);

      const network = await playground.createNetwork({
        layers: [{ type: "dense", units: 10 }],
      });

      const result = await playground.trainStep(network.id, {
        data: [[1, 2, 3]],
        labels: [[1]],
      });

      expect(result).toHaveProperty("loss");
    });
  });

  describe("Training Progress 4D Journey (Task 10)", () => {
    it("should create journey instance", () => {
      const journey = createTraining4DJourney();
      expect(journey).toBeDefined();
    });

    it("should capture training snapshots", async () => {
      const journey = new Training4DJourney(
        mockTimelineNavigator,
        mockTrainingService
      );

      await journey.captureSnapshot("model-1");

      const snapshots = journey.getSnapshots("model-1");
      expect(snapshots.length).toBeGreaterThan(0);
    });

    it("should navigate to specific epoch", async () => {
      const journey = new Training4DJourney(
        mockTimelineNavigator,
        mockTrainingService
      );

      await journey.captureSnapshot("model-1");
      await journey.navigateToEpoch("model-1", 5);

      const currentEpoch = journey.getCurrentEpoch("model-1");
      expect(currentEpoch).toBe(5);
    });
  });

  describe("Model Router Cosmos (Task 11)", () => {
    it("should create router cosmos instance", () => {
      const cosmos = createModelRouterCosmos();
      expect(cosmos).toBeDefined();
    });

    it("should add model nodes", () => {
      const cosmos = new ModelRouterCosmos(mockTimelineNavigator);

      cosmos.addNode({
        id: "model-1",
        type: "expert",
        position: [0, 0, 0],
      });

      const node = cosmos.getNode("model-1");
      expect(node).toBeDefined();
    });

    it("should route requests through ensemble", async () => {
      const cosmos = new ModelRouterCosmos(mockTimelineNavigator);

      cosmos.addNode({ id: "model-1", type: "expert", position: [0, 0, 0] });
      cosmos.addNode({ id: "model-2", type: "expert", position: [1, 0, 0] });
      cosmos.addRoute({ source: "model-1", target: "model-2", weight: 0.5 });

      const result = await cosmos.submitRequest("model-1", {
        input: [1, 2, 3],
      });
      expect(result).toBeDefined();
    });
  });
});

// ============================================================================
// P0 Breakthrough Tests (Tasks 12-15)
// ============================================================================

describe("P0 Breakthrough Innovations", () => {
  let mockTwinManager: any;
  let mockTimelineNavigator: any;
  let mockTrainingService: any;

  beforeEach(() => {
    mockTwinManager = createMockTwinManager();
    mockTimelineNavigator = createMockTimelineNavigator();
    mockTrainingService = createMockTrainingService();
  });

  describe("Living Architecture Laboratory (Task 12)", () => {
    it("should create living architecture lab", () => {
      const lab = createLivingArchitectureLab();
      expect(lab).toBeDefined();
    });

    it("should create living neural network", async () => {
      const lab = new LivingArchitectureLaboratory(
        mockTwinManager,
        mockTimelineNavigator,
        mockTrainingService
      );

      const organism = await lab.createOrganism({
        architecture: {
          layers: [
            { type: "dense", units: 64 },
            { type: "dense", units: 32 },
          ],
        },
        environmentId: "env-1",
      });

      expect(organism).toBeDefined();
      expect(organism.vitality).toBeGreaterThan(0);
    });

    it("should simulate organism lifecycle", async () => {
      const lab = new LivingArchitectureLaboratory(
        mockTwinManager,
        mockTimelineNavigator,
        mockTrainingService
      );

      const organism = await lab.createOrganism({
        architecture: { layers: [{ type: "dense", units: 32 }] },
        environmentId: "env-1",
      });

      await waitFor(100);

      const metrics = lab.getOrganismMetrics(organism.id);
      expect(metrics).toHaveProperty("vitality");
      expect(metrics).toHaveProperty("energy");
      expect(metrics).toHaveProperty("age");
    });
  });

  describe("Morphogenic Model Evolution (Task 13)", () => {
    it("should create morphogenic evolution instance", () => {
      const evolution = createMorphogenicEvolution();
      expect(evolution).toBeDefined();
    });

    it("should evolve model structure", async () => {
      const evolution = new MorphogenicModelEvolution(
        mockTwinManager,
        mockTrainingService
      );

      const initialModel = {
        id: "model-1",
        architecture: { layers: [{ type: "dense", units: 64 }] },
      };

      const evolved = await evolution.evolveStructure(initialModel, {
        generations: 5,
        mutationRate: 0.1,
      });

      expect(evolved).toBeDefined();
      expect(evolved.generation).toBe(5);
    });

    it("should track evolutionary lineage", async () => {
      const evolution = new MorphogenicModelEvolution(
        mockTwinManager,
        mockTrainingService
      );

      const initialModel = {
        id: "model-1",
        architecture: { layers: [{ type: "dense", units: 64 }] },
      };

      await evolution.evolveStructure(initialModel, { generations: 3 });

      const lineage = evolution.getLineage("model-1");
      expect(lineage).toBeDefined();
      expect(lineage.ancestors.length).toBeGreaterThan(0);
    });
  });

  describe("Causal Training Debugger (Task 14)", () => {
    it("should create causal debugger instance", () => {
      const causalDebugger = createCausalDebugger();
      expect(causalDebugger).toBeDefined();
    });

    it("should build causal graph", async () => {
      const causalDebugger = new CausalTrainingDebugger(
        mockTwinManager,
        mockTrainingService
      );

      await causalDebugger.startDebugging("model-1");
      await waitFor(100);

      const graph = causalDebugger.getCausalGraph("model-1");
      expect(graph).toBeDefined();
      expect(graph.nodes.length).toBeGreaterThan(0);
    });

    it("should run counterfactual analysis", async () => {
      const causalDebugger = new CausalTrainingDebugger(
        mockTwinManager,
        mockTrainingService
      );

      await causalDebugger.startDebugging("model-1");

      const counterfactual = await causalDebugger.analyzeCounterfactual(
        "model-1",
        {
          intervention: { learningRate: 0.01 },
          targetMetric: "accuracy",
        }
      );

      expect(counterfactual).toBeDefined();
      expect(counterfactual).toHaveProperty("predictedOutcome");
    });
  });

  describe("Quantum Architecture Search (Task 15)", () => {
    it("should create quantum search instance", () => {
      const search = createQuantumArchitectureSearch();
      expect(search).toBeDefined();
    });

    it("should create architecture superposition", () => {
      const search = new QuantumArchitectureSearch();

      const superposition = search["generateRandomArchitectures"](10, {
        maxLayers: 5,
        maxParameters: 10000,
        allowedLayerTypes: new Set(["dense", "conv2d"]),
        customConstraints: new Map(),
      });

      expect(superposition).toHaveLength(10);
    });

    it("should run quantum-inspired search", async () => {
      const search = new QuantumArchitectureSearch();

      const evaluationFn = async (arch: any) => ({
        accuracy: Math.random(),
        loss: Math.random(),
        latency: 100,
        parameterCount: 1000,
      });

      const searchPromise = search.startSearch(
        "test-search",
        "quantum-random",
        {
          maxLayers: 5,
          maxParameters: 10000,
          allowedLayerTypes: new Set(["dense"]),
          customConstraints: new Map(),
        },
        evaluationFn
      );

      // Wait for at least one iteration
      await waitFor(200);

      const searchState = search.getSearchState("test-search");
      expect(searchState).toBeDefined();
      expect(searchState?.iterations).toBeGreaterThan(0);
    }, 10000); // Extended timeout for search

    it("should collapse superposition on measurement", async () => {
      const result = await quickQuantumSearch(
        {
          maxLayers: 3,
          maxParameters: 5000,
          allowedLayerTypes: new Set(["dense"]),
        },
        async (arch) => ({
          accuracy: 0.85,
          loss: 0.15,
          latency: 50,
          parameterCount: arch.layers.length * 100,
        })
      );

      expect(result).toBeDefined();
      expect(result.collapsedArchitecture).toBeDefined();
      expect(result.actualPerformance).toBeDefined();
    }, 10000);
  });
});

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

describe("End-to-End Cross-Domain Workflows", () => {
  let eventBridge: CrossDomainEventBridge;
  let orchestrator: CrossDomainOrchestrator;

  beforeEach(() => {
    eventBridge = CrossDomainEventBridge.getInstance();
    orchestrator = CrossDomainOrchestrator.getInstance();
  });

  afterEach(() => {
    eventBridge.removeAllListeners();
  });

  it("should coordinate Forge×Twin×Foundry workflow", async () => {
    const mockTwinManager = createMockTwinManager();
    const mockTimelineNavigator = createMockTimelineNavigator();
    const mockTrainingService = createMockTrainingService();

    // 1. Start with Living Architecture Lab (all three domains)
    const lab = new LivingArchitectureLaboratory(
      mockTwinManager,
      mockTimelineNavigator,
      mockTrainingService
    );

    const organism = await lab.createOrganism({
      architecture: { layers: [{ type: "dense", units: 64 }] },
      environmentId: "env-1",
    });

    expect(organism).toBeDefined();

    // 2. Use Replay Theater to capture history (Forge×Twin)
    const theater = new TemporalTwinReplayTheater(
      mockTwinManager,
      mockTimelineNavigator
    );

    const snapshot = await theater.captureSnapshot(organism.id);
    expect(snapshot).toBeDefined();

    // 3. Generate predictions (Forge×Twin)
    const cascade = new PredictiveVisualizationCascade(
      mockTwinManager,
      mockTimelineNavigator
    );

    const prediction = await cascade.generatePrediction(organism.id, {
      horizonMs: 1000,
    });
    expect(prediction).toBeDefined();

    // 4. Evolve architecture based on predictions (Twin×Foundry)
    const evolution = new MorphogenicModelEvolution(
      mockTwinManager,
      mockTrainingService
    );

    const evolved = await evolution.evolveStructure(
      {
        id: organism.id,
        architecture: organism.architecture,
      },
      { generations: 2 }
    );

    expect(evolved).toBeDefined();
  }, 15000);

  it("should handle event propagation across all innovations", async () => {
    const events: any[] = [];

    // Subscribe to all cross-domain events
    eventBridge.subscribe("*", (data) => {
      events.push(data);
    });

    // Trigger events from different domains
    eventBridge.publish("twin:state:updated", { twinId: "test" });
    eventBridge.publish("forge:timeline:changed", { time: 1000 });
    eventBridge.publish("foundry:training:started", { modelId: "model-1" });

    await waitFor(50);

    expect(events.length).toBeGreaterThanOrEqual(3);
  });

  it("should maintain performance under load", async () => {
    const startTime = Date.now();
    const iterations = 100;

    for (let i = 0; i < iterations; i++) {
      eventBridge.publish("test:event", { iteration: i });
    }

    const endTime = Date.now();
    const duration = endTime - startTime;

    // Should handle 100 events in under 100ms
    expect(duration).toBeLessThan(100);
  });
});

// ============================================================================
// Performance & Reliability Tests
// ============================================================================

describe("Performance & Reliability", () => {
  it("should handle concurrent operations", async () => {
    const eventBridge = CrossDomainEventBridge.getInstance();
    const handlers = Array(10)
      .fill(0)
      .map(() => vi.fn());

    handlers.forEach((handler) => {
      eventBridge.subscribe("test:concurrent", handler);
    });

    const operations = Array(50)
      .fill(0)
      .map((_, i) => eventBridge.publish("test:concurrent", { index: i }));

    await Promise.all(operations);
    await waitFor(50);

    handlers.forEach((handler) => {
      expect(handler).toHaveBeenCalledTimes(50);
    });
  });

  it("should recover from errors gracefully", async () => {
    const eventBridge = CrossDomainEventBridge.getInstance();
    const errorHandler = vi.fn(() => {
      throw new Error("Test error");
    });
    const successHandler = vi.fn();

    eventBridge.subscribe("test:error", errorHandler);
    eventBridge.subscribe("test:error", successHandler);

    expect(() => {
      eventBridge.publish("test:error", { data: "test" });
    }).not.toThrow();

    await waitFor(10);
    expect(successHandler).toHaveBeenCalled();
  });

  it("should cleanup resources properly", () => {
    const eventBridge = CrossDomainEventBridge.getInstance();
    const handler = vi.fn();

    const unsubscribe = eventBridge.subscribe("test:cleanup", handler);

    eventBridge.publish("test:cleanup", { data: 1 });
    expect(handler).toHaveBeenCalledTimes(1);

    unsubscribe();

    eventBridge.publish("test:cleanup", { data: 2 });
    expect(handler).toHaveBeenCalledTimes(1); // Still 1, not called again
  });
});
