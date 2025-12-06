/**
 * @fileoverview Unit Tests for Hypothesis Lab
 * @module @neurectomy/experimentation-engine/__tests__/hypothesis-lab
 * @agent @ECLIPSE @PRISM
 *
 * Comprehensive test suite for ML experiment management:
 * - HypothesisLab lifecycle and configuration
 * - Hypothesis creation and management
 * - Trial management and execution
 * - Parameter space validation
 * - Optimization strategies (random, grid, Bayesian)
 * - Best trial tracking
 * - Persistence and import/export
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  HypothesisLab,
  createHypothesisLab,
  defineHypothesis,
  defineParameterSpace,
  HypothesisConfig,
  ParameterSpace,
  Parameter,
  Hypothesis,
  Trial,
  TrialConfig,
  TrialResult,
  LabConfig,
  StorageBackend,
} from "../hypothesis/lab";

// ============================================================================
// Mock Implementations
// ============================================================================

/**
 * Mock storage backend for testing
 */
class MockStorage implements StorageBackend {
  private store = new Map<string, unknown>();
  saveCalls: Array<{ key: string; data: unknown }> = [];
  loadCalls: string[] = [];

  async save(key: string, data: unknown): Promise<void> {
    this.saveCalls.push({ key, data });
    this.store.set(key, JSON.parse(JSON.stringify(data)));
  }

  async load(key: string): Promise<unknown | null> {
    this.loadCalls.push(key);
    const data = this.store.get(key);
    return data ? JSON.parse(JSON.stringify(data)) : null;
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async list(prefix: string): Promise<string[]> {
    return Array.from(this.store.keys()).filter((k) => k.startsWith(prefix));
  }

  reset(): void {
    this.store.clear();
    this.saveCalls = [];
    this.loadCalls = [];
  }
}

// ============================================================================
// Test Fixtures
// ============================================================================

function createBasicHypothesisConfig(
  overrides: Partial<HypothesisConfig> = {}
): HypothesisConfig {
  return {
    name: "Test ML Hypothesis",
    description: "Testing learning rate impact on accuracy",
    hypothesis: "Increasing learning rate will improve convergence speed",
    nullHypothesis: "Learning rate has no effect on convergence",
    expectedOutcome: "Optimal learning rate around 0.001",
    confidenceLevel: 0.95,
    powerLevel: 0.8,
    tags: ["ml", "optimization"],
    ...overrides,
  };
}

function createBasicParameterSpace(
  overrides: Partial<ParameterSpace> = {}
): ParameterSpace {
  return {
    parameters: [
      {
        name: "learning_rate",
        type: "continuous",
        min: 0.0001,
        max: 0.1,
        default: 0.001,
        description: "Model learning rate",
      },
      {
        name: "batch_size",
        type: "discrete",
        min: 16,
        max: 256,
        default: 32,
        description: "Training batch size",
      },
      {
        name: "optimizer",
        type: "categorical",
        values: ["adam", "sgd", "rmsprop"],
        default: "adam",
        description: "Optimization algorithm",
      },
      {
        name: "use_dropout",
        type: "boolean",
        default: true,
        description: "Whether to use dropout",
      },
    ],
    ...overrides,
  };
}

function createTrialConfig(
  hypothesisId: string,
  params: Record<string, unknown> = {}
): TrialConfig {
  return {
    hypothesisId,
    parameters: {
      learning_rate: 0.001,
      batch_size: 32,
      optimizer: "adam",
      use_dropout: true,
      ...params,
    },
  };
}

function createSuccessfulTrialResult(
  metrics: Record<string, number> = {}
): Omit<TrialResult, "trialId" | "timestamp"> {
  return {
    metrics: {
      accuracy: 0.85,
      loss: 0.15,
      ...metrics,
    },
    duration: 5000,
    status: "success",
  };
}

// ============================================================================
// HypothesisLab Tests
// ============================================================================

describe("HypothesisLab", () => {
  let lab: HypothesisLab;
  let mockStorage: MockStorage;

  beforeEach(() => {
    mockStorage = new MockStorage();
    lab = new HypothesisLab({
      storageBackend: mockStorage,
      autoSave: false,
      maxConcurrentTrials: 5,
      defaultTimeout: 60000,
    });
  });

  afterEach(() => {
    lab.dispose();
    mockStorage.reset();
  });

  // --------------------------------------------------------------------------
  // Hypothesis Creation Tests
  // --------------------------------------------------------------------------

  describe("createHypothesis", () => {
    it("should create hypothesis with valid config", () => {
      const config = createBasicHypothesisConfig();
      const space = createBasicParameterSpace();

      const hypothesis = lab.createHypothesis(config, space);

      expect(hypothesis.id).toBeDefined();
      expect(hypothesis.config.name).toBe("Test ML Hypothesis");
      expect(hypothesis.status).toBe("draft");
      expect(hypothesis.trials).toEqual([]);
      expect(hypothesis.createdAt).toBeInstanceOf(Date);
      expect(hypothesis.updatedAt).toBeInstanceOf(Date);
    });

    it("should use provided ID when given", () => {
      const config = createBasicHypothesisConfig({ id: "custom-id-123" });
      const space = createBasicParameterSpace();

      const hypothesis = lab.createHypothesis(config, space);

      expect(hypothesis.id).toBe("custom-id-123");
    });

    it("should set default confidence and power levels", () => {
      const config = createBasicHypothesisConfig({
        confidenceLevel: undefined,
        powerLevel: undefined,
      });
      const space = createBasicParameterSpace();

      const hypothesis = lab.createHypothesis(config, space);

      expect(hypothesis.config.confidenceLevel).toBe(0.95);
      expect(hypothesis.config.powerLevel).toBe(0.8);
    });

    it("should emit hypothesisCreated event", () => {
      const eventSpy = vi.fn();
      lab.on("hypothesisCreated", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      expect(eventSpy).toHaveBeenCalledTimes(1);
      expect(eventSpy).toHaveBeenCalledWith(hypothesis);
    });

    it("should validate parameter space on creation", () => {
      const config = createBasicHypothesisConfig();
      const invalidSpace: ParameterSpace = {
        parameters: [
          {
            name: "learning_rate",
            type: "continuous",
            min: 1.0,
            max: 0.1, // Invalid: min > max
          },
        ],
      };

      expect(() => lab.createHypothesis(config, invalidSpace)).toThrow(
        "Invalid range"
      );
    });

    it("should require values for categorical parameters", () => {
      const config = createBasicHypothesisConfig();
      const invalidSpace: ParameterSpace = {
        parameters: [
          {
            name: "optimizer",
            type: "categorical",
            values: [], // Empty values
          },
        ],
      };

      expect(() => lab.createHypothesis(config, invalidSpace)).toThrow(
        "must have values"
      );
    });

    it("should require name for parameters", () => {
      const config = createBasicHypothesisConfig();
      const invalidSpace: ParameterSpace = {
        parameters: [
          {
            name: "", // Empty name
            type: "continuous",
          },
        ],
      };

      expect(() => lab.createHypothesis(config, invalidSpace)).toThrow(
        "must have a name"
      );
    });
  });

  // --------------------------------------------------------------------------
  // Hypothesis Retrieval Tests
  // --------------------------------------------------------------------------

  describe("getHypothesis", () => {
    it("should return hypothesis by ID", () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      const retrieved = lab.getHypothesis(hypothesis.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(hypothesis.id);
    });

    it("should return undefined for non-existent ID", () => {
      const retrieved = lab.getHypothesis("non-existent-id");

      expect(retrieved).toBeUndefined();
    });
  });

  describe("listHypotheses", () => {
    beforeEach(() => {
      // Create multiple hypotheses with different statuses
      const h1 = lab.createHypothesis(
        createBasicHypothesisConfig({ tags: ["ml", "production"] }),
        createBasicParameterSpace()
      );
      h1.status = "running";

      const h2 = lab.createHypothesis(
        createBasicHypothesisConfig({ tags: ["ml", "experiment"] }),
        createBasicParameterSpace()
      );
      h2.status = "completed";

      lab.createHypothesis(
        createBasicHypothesisConfig({ tags: ["research"] }),
        createBasicParameterSpace()
      );
    });

    it("should list all hypotheses", () => {
      const all = lab.listHypotheses();

      expect(all).toHaveLength(3);
    });

    it("should filter by status", () => {
      const running = lab.listHypotheses({ status: "running" });

      expect(running).toHaveLength(1);
      expect(running[0].status).toBe("running");
    });

    it("should filter by tags", () => {
      const mlHypotheses = lab.listHypotheses({ tags: ["ml"] });

      expect(mlHypotheses).toHaveLength(2);
    });

    it("should combine status and tag filters", () => {
      const runningMl = lab.listHypotheses({ status: "running", tags: ["ml"] });

      expect(runningMl).toHaveLength(1);
    });
  });

  describe("updateHypothesisStatus", () => {
    it("should update hypothesis status", () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      lab.updateHypothesisStatus(hypothesis.id, "running");

      expect(hypothesis.status).toBe("running");
    });

    it("should set completedAt when status is completed", () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      expect(hypothesis.completedAt).toBeUndefined();

      lab.updateHypothesisStatus(hypothesis.id, "completed");

      expect(hypothesis.completedAt).toBeInstanceOf(Date);
    });

    it("should emit hypothesisCompleted event when completed", () => {
      const eventSpy = vi.fn();
      lab.on("hypothesisCompleted", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      lab.updateHypothesisStatus(hypothesis.id, "completed");

      expect(eventSpy).toHaveBeenCalledWith(hypothesis);
    });

    it("should throw for non-existent hypothesis", () => {
      expect(() =>
        lab.updateHypothesisStatus("non-existent", "running")
      ).toThrow("Hypothesis not found");
    });
  });

  describe("deleteHypothesis", () => {
    it("should delete hypothesis", () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      const result = lab.deleteHypothesis(hypothesis.id);

      expect(result).toBe(true);
      expect(lab.getHypothesis(hypothesis.id)).toBeUndefined();
    });

    it("should return false for non-existent hypothesis", () => {
      const result = lab.deleteHypothesis("non-existent");

      expect(result).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Trial Management Tests
  // --------------------------------------------------------------------------

  describe("createTrial", () => {
    let hypothesis: Hypothesis;

    beforeEach(() => {
      hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
    });

    it("should create trial with valid parameters", async () => {
      const trialConfig = createTrialConfig(hypothesis.id);

      const trial = await lab.createTrial(trialConfig);

      expect(trial.id).toBeDefined();
      expect(trial.hypothesisId).toBe(hypothesis.id);
      expect(trial.status).toBe("pending");
      expect(trial.parameters.learning_rate).toBe(0.001);
    });

    it("should add trial to hypothesis", async () => {
      const trialConfig = createTrialConfig(hypothesis.id);

      const trial = await lab.createTrial(trialConfig);

      expect(hypothesis.trials).toContain(trial);
    });

    it("should throw for non-existent hypothesis", async () => {
      const trialConfig = createTrialConfig("non-existent");

      await expect(lab.createTrial(trialConfig)).rejects.toThrow(
        "Hypothesis not found"
      );
    });

    it("should validate parameters against space", async () => {
      const trialConfig = createTrialConfig(hypothesis.id, {
        learning_rate: 999, // Above max
      });

      await expect(lab.createTrial(trialConfig)).rejects.toThrow(
        "above maximum"
      );
    });

    it("should validate parameter types", async () => {
      const trialConfig = createTrialConfig(hypothesis.id, {
        use_dropout: "yes", // Should be boolean
      });

      await expect(lab.createTrial(trialConfig)).rejects.toThrow(
        "must be boolean"
      );
    });

    it("should validate categorical values", async () => {
      const trialConfig = createTrialConfig(hypothesis.id, {
        optimizer: "invalid-optimizer",
      });

      await expect(lab.createTrial(trialConfig)).rejects.toThrow(
        "Invalid value"
      );
    });

    it("should respect max concurrent trials", async () => {
      // Create lab with max 2 concurrent trials
      const restrictedLab = new HypothesisLab({
        maxConcurrentTrials: 2,
        autoSave: false,
      });

      const h = restrictedLab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      // Create and start 2 trials
      const trial1 = await restrictedLab.createTrial(createTrialConfig(h.id));
      const trial2 = await restrictedLab.createTrial(createTrialConfig(h.id));
      restrictedLab.startTrial(trial1.id, h.id);
      restrictedLab.startTrial(trial2.id, h.id);

      // Third trial should fail
      await expect(
        restrictedLab.createTrial(createTrialConfig(h.id))
      ).rejects.toThrow("Max concurrent trials");

      restrictedLab.dispose();
    });
  });

  describe("startTrial", () => {
    it("should transition trial to running state", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));

      lab.startTrial(trial.id, hypothesis.id);

      expect(trial.status).toBe("running");
      expect(trial.startedAt).toBeInstanceOf(Date);
    });

    it("should emit trialStarted event", async () => {
      const eventSpy = vi.fn();
      lab.on("trialStarted", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);

      expect(eventSpy).toHaveBeenCalledWith(trial);
    });

    it("should throw for non-existent trial", () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      expect(() => lab.startTrial("non-existent", hypothesis.id)).toThrow(
        "Trial not found"
      );
    });
  });

  describe("completeTrial", () => {
    it("should complete trial with results", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);

      const result = createSuccessfulTrialResult();
      lab.completeTrial(trial.id, hypothesis.id, result);

      expect(trial.status).toBe("completed");
      expect(trial.result).toBeDefined();
      expect(trial.result?.metrics.accuracy).toBe(0.85);
      expect(trial.completedAt).toBeInstanceOf(Date);
    });

    it("should emit trialCompleted event", async () => {
      const eventSpy = vi.fn();
      lab.on("trialCompleted", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);
      lab.completeTrial(trial.id, hypothesis.id, createSuccessfulTrialResult());

      expect(eventSpy).toHaveBeenCalled();
    });

    it("should update best trial when metrics improve", async () => {
      const eventSpy = vi.fn();
      lab.on("newBestTrial", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      // First trial
      const trial1 = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial1.id, hypothesis.id);
      lab.completeTrial(
        trial1.id,
        hypothesis.id,
        createSuccessfulTrialResult({ accuracy: 0.8 })
      );

      expect(hypothesis.bestTrial?.id).toBe(trial1.id);
      expect(eventSpy).toHaveBeenCalledTimes(1);

      // Second trial with better metrics
      const trial2 = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial2.id, hypothesis.id);
      lab.completeTrial(
        trial2.id,
        hypothesis.id,
        createSuccessfulTrialResult({ accuracy: 0.9 })
      );

      expect(hypothesis.bestTrial?.id).toBe(trial2.id);
      expect(eventSpy).toHaveBeenCalledTimes(2);
    });

    it("should not update best trial when metrics are worse", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      // First trial with good metrics
      const trial1 = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial1.id, hypothesis.id);
      lab.completeTrial(
        trial1.id,
        hypothesis.id,
        createSuccessfulTrialResult({ accuracy: 0.9 })
      );

      // Second trial with worse metrics
      const trial2 = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial2.id, hypothesis.id);
      lab.completeTrial(
        trial2.id,
        hypothesis.id,
        createSuccessfulTrialResult({ accuracy: 0.8 })
      );

      expect(hypothesis.bestTrial?.id).toBe(trial1.id);
    });
  });

  describe("failTrial", () => {
    it("should mark trial as failed", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);

      const error = new Error("Training failed: OOM");
      lab.failTrial(trial.id, hypothesis.id, error);

      expect(trial.status).toBe("failed");
      expect(trial.result?.status).toBe("failed");
      expect(trial.result?.error).toBe("Training failed: OOM");
    });

    it("should emit trialFailed event", async () => {
      const eventSpy = vi.fn();
      lab.on("trialFailed", eventSpy);

      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);
      lab.failTrial(trial.id, hypothesis.id, new Error("Test error"));

      expect(eventSpy).toHaveBeenCalled();
    });
  });

  describe("getRunningTrials", () => {
    it("should return all running trials", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      const trial1 = await lab.createTrial(createTrialConfig(hypothesis.id));
      const trial2 = await lab.createTrial(createTrialConfig(hypothesis.id));

      lab.startTrial(trial1.id, hypothesis.id);
      lab.startTrial(trial2.id, hypothesis.id);

      const running = lab.getRunningTrials();

      expect(running).toHaveLength(2);
      expect(running.every((t) => t.status === "running")).toBe(true);
    });

    it("should not include completed trials", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);
      lab.completeTrial(trial.id, hypothesis.id, createSuccessfulTrialResult());

      const running = lab.getRunningTrials();

      expect(running).toHaveLength(0);
    });
  });

  // --------------------------------------------------------------------------
  // Parameter Suggestion Tests
  // --------------------------------------------------------------------------

  describe("suggestNextParameters", () => {
    let hypothesis: Hypothesis;

    beforeEach(() => {
      hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
    });

    describe("random strategy", () => {
      it("should generate parameters within bounds", () => {
        const params = lab.suggestNextParameters(hypothesis.id, "random");

        expect(params.learning_rate).toBeGreaterThanOrEqual(0.0001);
        expect(params.learning_rate).toBeLessThanOrEqual(0.1);
        expect(params.batch_size).toBeGreaterThanOrEqual(16);
        expect(params.batch_size).toBeLessThanOrEqual(256);
        expect(["adam", "sgd", "rmsprop"]).toContain(params.optimizer);
        expect(typeof params.use_dropout).toBe("boolean");
      });

      it("should generate different parameters on multiple calls", () => {
        const params1 = lab.suggestNextParameters(hypothesis.id, "random");
        const params2 = lab.suggestNextParameters(hypothesis.id, "random");
        const params3 = lab.suggestNextParameters(hypothesis.id, "random");

        // With enough parameters, at least one should differ
        const allSame =
          JSON.stringify(params1) === JSON.stringify(params2) &&
          JSON.stringify(params2) === JSON.stringify(params3);

        // Very unlikely all three are identical with random sampling
        expect(allSame).toBe(false);
      });
    });

    describe("grid strategy", () => {
      it("should explore grid systematically", async () => {
        // Create a simpler space for grid testing
        const simpleSpace: ParameterSpace = {
          parameters: [
            { name: "x", type: "discrete", min: 1, max: 3 },
            { name: "y", type: "categorical", values: ["a", "b"] },
          ],
        };

        const simpleHypothesis = lab.createHypothesis(
          createBasicHypothesisConfig(),
          simpleSpace
        );

        // Get suggestions and mark as tried
        const suggestions: Array<Record<string, unknown>> = [];
        for (let i = 0; i < 6; i++) {
          const params = lab.suggestNextParameters(simpleHypothesis.id, "grid");
          suggestions.push(params);

          // Mark as tried by creating a trial
          const trial = await lab.createTrial({
            hypothesisId: simpleHypothesis.id,
            parameters: params,
          });
        }

        // Should have explored different combinations
        const uniqueSuggestions = new Set(suggestions.map(JSON.stringify));
        expect(uniqueSuggestions.size).toBeGreaterThan(1);
      });
    });

    describe("bayesian strategy", () => {
      it("should fall back to random with few trials", () => {
        // With no trials, should use random
        const params = lab.suggestNextParameters(hypothesis.id, "bayesian");

        // Just verify we get valid parameters
        expect(params.learning_rate).toBeDefined();
        expect(params.batch_size).toBeDefined();
      });

      it("should use acquisition function with enough trials", async () => {
        // Create several completed trials
        for (let i = 0; i < 5; i++) {
          const trial = await lab.createTrial({
            hypothesisId: hypothesis.id,
            parameters: {
              learning_rate: 0.001 + i * 0.001,
              batch_size: 32 + i * 16,
              optimizer: "adam",
              use_dropout: true,
            },
          });
          lab.startTrial(trial.id, hypothesis.id);
          lab.completeTrial(trial.id, hypothesis.id, {
            metrics: { accuracy: 0.7 + i * 0.05 },
            duration: 1000,
            status: "success",
          });
        }

        // Now Bayesian should be active
        const params = lab.suggestNextParameters(hypothesis.id, "bayesian");

        expect(params.learning_rate).toBeDefined();
      });
    });

    it("should throw for non-existent hypothesis", () => {
      expect(() => lab.suggestNextParameters("non-existent")).toThrow(
        "Hypothesis not found"
      );
    });
  });

  // --------------------------------------------------------------------------
  // Persistence Tests
  // --------------------------------------------------------------------------

  describe("persistence", () => {
    describe("saveAll", () => {
      it("should save all hypotheses to storage", async () => {
        lab.createHypothesis(
          createBasicHypothesisConfig(),
          createBasicParameterSpace()
        );
        lab.createHypothesis(
          createBasicHypothesisConfig({ name: "Second Hypothesis" }),
          createBasicParameterSpace()
        );

        await lab.saveAll();

        expect(mockStorage.saveCalls).toHaveLength(2);
        expect(mockStorage.saveCalls[0].key).toMatch(/^hypothesis:/);
      });
    });

    describe("loadAll", () => {
      it("should load hypotheses from storage", async () => {
        // Save some hypotheses
        const h1 = lab.createHypothesis(
          createBasicHypothesisConfig(),
          createBasicParameterSpace()
        );
        await lab.saveAll();

        // Create new lab and load
        const newLab = new HypothesisLab({
          storageBackend: mockStorage,
          autoSave: false,
        });

        await newLab.loadAll();

        const loaded = newLab.getHypothesis(h1.id);
        expect(loaded).toBeDefined();
        expect(loaded?.config.name).toBe(h1.config.name);

        newLab.dispose();
      });
    });

    describe("exportHypothesis", () => {
      it("should export hypothesis to JSON", () => {
        const hypothesis = lab.createHypothesis(
          createBasicHypothesisConfig(),
          createBasicParameterSpace()
        );

        const json = lab.exportHypothesis(hypothesis.id);
        const parsed = JSON.parse(json);

        expect(parsed.id).toBe(hypothesis.id);
        expect(parsed.config.name).toBe("Test ML Hypothesis");
        expect(parsed.parameterSpace.parameters).toHaveLength(4);
      });

      it("should throw for non-existent hypothesis", () => {
        expect(() => lab.exportHypothesis("non-existent")).toThrow(
          "Hypothesis not found"
        );
      });
    });

    describe("importHypothesis", () => {
      it("should import hypothesis from JSON", () => {
        const hypothesis = lab.createHypothesis(
          createBasicHypothesisConfig(),
          createBasicParameterSpace()
        );
        const json = lab.exportHypothesis(hypothesis.id);

        // Create new lab and import
        const newLab = new HypothesisLab({ autoSave: false });
        const imported = newLab.importHypothesis(json);

        expect(imported.id).toBe(hypothesis.id);
        expect(imported.config.name).toBe(hypothesis.config.name);
        expect(imported.createdAt).toBeInstanceOf(Date);

        newLab.dispose();
      });
    });
  });

  // --------------------------------------------------------------------------
  // Auto-Save Tests
  // --------------------------------------------------------------------------

  describe("auto-save", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("should auto-save at configured interval", async () => {
      const autoSaveLab = new HypothesisLab({
        storageBackend: mockStorage,
        autoSave: true,
        saveInterval: 1000,
      });

      autoSaveLab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );

      expect(mockStorage.saveCalls).toHaveLength(0);

      // Advance time past save interval
      await vi.advanceTimersByTimeAsync(1100);

      expect(mockStorage.saveCalls.length).toBeGreaterThan(0);

      autoSaveLab.dispose();
    });
  });

  // --------------------------------------------------------------------------
  // Cleanup Tests
  // --------------------------------------------------------------------------

  describe("dispose", () => {
    it("should clear running trials on dispose", async () => {
      const hypothesis = lab.createHypothesis(
        createBasicHypothesisConfig(),
        createBasicParameterSpace()
      );
      const trial = await lab.createTrial(createTrialConfig(hypothesis.id));
      lab.startTrial(trial.id, hypothesis.id);

      expect(lab.getRunningTrials()).toHaveLength(1);

      lab.dispose();

      expect(lab.getRunningTrials()).toHaveLength(0);
    });

    it("should remove all event listeners", () => {
      const eventSpy = vi.fn();
      lab.on("hypothesisCreated", eventSpy);

      lab.dispose();

      // After dispose, creating hypothesis shouldn't trigger events
      // (but we can't easily test this without accessing internals)
      expect(lab.listenerCount("hypothesisCreated")).toBe(0);
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe("Factory Functions", () => {
  describe("createHypothesisLab", () => {
    it("should create lab with default config", () => {
      const lab = createHypothesisLab();

      expect(lab).toBeInstanceOf(HypothesisLab);

      lab.dispose();
    });

    it("should create lab with custom config", () => {
      const lab = createHypothesisLab({
        maxConcurrentTrials: 10,
        defaultTimeout: 120000,
      });

      expect(lab).toBeInstanceOf(HypothesisLab);

      lab.dispose();
    });
  });

  describe("defineHypothesis", () => {
    it("should create hypothesis config", () => {
      const config = defineHypothesis(
        "Learning Rate Study",
        "Higher learning rate leads to faster convergence",
        {
          confidenceLevel: 0.99,
          tags: ["ml", "hyperparameter"],
        }
      );

      expect(config.name).toBe("Learning Rate Study");
      expect(config.hypothesis).toBe(
        "Higher learning rate leads to faster convergence"
      );
      expect(config.confidenceLevel).toBe(0.99);
      expect(config.tags).toContain("ml");
    });
  });

  describe("defineParameterSpace", () => {
    it("should create parameter space", () => {
      const space = defineParameterSpace(
        [
          { name: "lr", type: "continuous", min: 0.001, max: 0.1 },
          { name: "epochs", type: "discrete", min: 10, max: 100 },
        ],
        [{ type: "sum", parameters: ["lr", "epochs"], value: 1 }]
      );

      expect(space.parameters).toHaveLength(2);
      expect(space.constraints).toHaveLength(1);
    });
  });
});

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

describe("Edge Cases", () => {
  let lab: HypothesisLab;

  beforeEach(() => {
    lab = new HypothesisLab({ autoSave: false });
  });

  afterEach(() => {
    lab.dispose();
  });

  it("should handle empty parameter space", () => {
    const config = createBasicHypothesisConfig();
    const emptySpace: ParameterSpace = { parameters: [] };

    const hypothesis = lab.createHypothesis(config, emptySpace);

    expect(hypothesis).toBeDefined();
    expect(hypothesis.parameterSpace.parameters).toHaveLength(0);
  });

  it("should handle very large parameter values", async () => {
    const config = createBasicHypothesisConfig();
    const largeSpace: ParameterSpace = {
      parameters: [
        { name: "big_num", type: "continuous", min: 1e-10, max: 1e10 },
      ],
    };

    const hypothesis = lab.createHypothesis(config, largeSpace);
    const params = lab.suggestNextParameters(hypothesis.id, "random");

    expect(params.big_num).toBeGreaterThanOrEqual(1e-10);
    expect(params.big_num).toBeLessThanOrEqual(1e10);
  });

  it("should handle unicode in hypothesis names", () => {
    const config = createBasicHypothesisConfig({
      name: "测试假设 🧪 Тест",
      hypothesis: "Unicode hypothesis: αβγδ",
    });

    const hypothesis = lab.createHypothesis(
      config,
      createBasicParameterSpace()
    );

    expect(hypothesis.config.name).toBe("测试假设 🧪 Тест");
    expect(hypothesis.config.hypothesis).toBe("Unicode hypothesis: αβγδ");
  });

  it("should handle concurrent trial operations", async () => {
    const hypothesis = lab.createHypothesis(
      createBasicHypothesisConfig(),
      createBasicParameterSpace()
    );

    // Create multiple trials concurrently
    const trials = await Promise.all([
      lab.createTrial(createTrialConfig(hypothesis.id)),
      lab.createTrial(createTrialConfig(hypothesis.id)),
      lab.createTrial(createTrialConfig(hypothesis.id)),
    ]);

    expect(trials).toHaveLength(3);
    expect(new Set(trials.map((t) => t.id)).size).toBe(3); // All unique IDs
  });
});
