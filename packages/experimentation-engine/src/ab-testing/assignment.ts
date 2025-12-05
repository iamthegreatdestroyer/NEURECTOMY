/**
 * NEURECTOMY A/B Testing User Assignment
 * @module @neurectomy/experimentation-engine/ab-testing
 * @agent @PRISM @FLUX
 *
 * User assignment strategies including deterministic hashing,
 * multi-armed bandits, and contextual assignment.
 */

import { EventEmitter } from "eventemitter3";

// ============================================================================
// Types
// ============================================================================

export interface AssignmentStrategy {
  type: AssignmentStrategyType;
  assign(
    subjectId: string,
    variants: VariantWeight[],
    context?: AssignmentContext
  ): string;
  update?(variantId: string, reward: number): void;
}

export type AssignmentStrategyType =
  | "random"
  | "deterministic"
  | "weighted"
  | "epsilon_greedy"
  | "thompson_sampling"
  | "ucb1"
  | "contextual";

export interface VariantWeight {
  id: string;
  weight: number;
  isControl?: boolean;
}

export interface AssignmentContext {
  attributes?: Record<string, unknown>;
  previousAssignments?: string[];
  experimentId?: string;
}

export interface BanditState {
  variantId: string;
  trials: number;
  successes: number;
  failures: number;
  totalReward: number;
  alpha: number; // Beta distribution parameter
  beta: number; // Beta distribution parameter
}

export interface AssignmentManagerConfig {
  defaultStrategy?: AssignmentStrategyType;
  stickyAssignments?: boolean;
  salt?: string;
}

export interface AssignmentManagerEvents {
  assignment: (
    subjectId: string,
    variantId: string,
    strategyType: string
  ) => void;
  banditUpdate: (variantId: string, reward: number) => void;
}

// ============================================================================
// Hash Functions
// ============================================================================

/**
 * MurmurHash3 - Fast, non-cryptographic hash
 */
function murmurhash3(key: string, seed: number = 0): number {
  let h1 = seed;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;

  for (let i = 0; i < key.length; i++) {
    let k1 = key.charCodeAt(i);
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);

    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = Math.imul(h1, 5) + 0xe6546b64;
  }

  h1 ^= key.length;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;

  return h1 >>> 0; // Convert to unsigned
}

/**
 * Convert hash to bucket (0-9999)
 */
function hashToBucket(hash: number, buckets: number = 10000): number {
  return hash % buckets;
}

/**
 * Create deterministic assignment key
 */
function createAssignmentKey(
  experimentId: string,
  subjectId: string,
  salt: string = ""
): string {
  return `${salt}:${experimentId}:${subjectId}`;
}

// ============================================================================
// Assignment Strategies
// ============================================================================

/**
 * Random assignment strategy
 */
export class RandomAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "random";

  assign(_subjectId: string, variants: VariantWeight[]): string {
    const totalWeight = variants.reduce((sum, v) => sum + v.weight, 0);
    const random = Math.random() * totalWeight;

    let cumulative = 0;
    for (const variant of variants) {
      cumulative += variant.weight;
      if (random < cumulative) {
        return variant.id;
      }
    }

    return variants[variants.length - 1].id;
  }
}

/**
 * Deterministic hash-based assignment (sticky)
 */
export class DeterministicAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "deterministic";
  private salt: string;
  private experimentId: string;

  constructor(experimentId: string, salt: string = "") {
    this.experimentId = experimentId;
    this.salt = salt;
  }

  assign(subjectId: string, variants: VariantWeight[]): string {
    const key = createAssignmentKey(this.experimentId, subjectId, this.salt);
    const hash = murmurhash3(key);
    const bucket = hashToBucket(hash);

    const totalWeight = variants.reduce((sum, v) => sum + v.weight, 0);
    const normalizedBucket = (bucket / 10000) * totalWeight;

    let cumulative = 0;
    for (const variant of variants) {
      cumulative += variant.weight;
      if (normalizedBucket < cumulative) {
        return variant.id;
      }
    }

    return variants[variants.length - 1].id;
  }
}

/**
 * Weighted random assignment
 */
export class WeightedAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "weighted";

  assign(_subjectId: string, variants: VariantWeight[]): string {
    const totalWeight = variants.reduce((sum, v) => sum + v.weight, 0);
    const random = Math.random() * totalWeight;

    let cumulative = 0;
    for (const variant of variants) {
      cumulative += variant.weight;
      if (random < cumulative) {
        return variant.id;
      }
    }

    return variants[variants.length - 1].id;
  }
}

/**
 * Epsilon-Greedy multi-armed bandit
 */
export class EpsilonGreedyAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "epsilon_greedy";
  private epsilon: number;
  private state = new Map<string, BanditState>();
  private decayRate: number;
  private minEpsilon: number;
  private totalTrials: number = 0;

  constructor(
    variants: string[],
    epsilon: number = 0.1,
    decayRate: number = 0.999,
    minEpsilon: number = 0.01
  ) {
    this.epsilon = epsilon;
    this.decayRate = decayRate;
    this.minEpsilon = minEpsilon;

    for (const variantId of variants) {
      this.state.set(variantId, {
        variantId,
        trials: 0,
        successes: 0,
        failures: 0,
        totalReward: 0,
        alpha: 1,
        beta: 1,
      });
    }
  }

  assign(_subjectId: string, variants: VariantWeight[]): string {
    // Exploration: random choice
    if (Math.random() < this.getCurrentEpsilon()) {
      const idx = Math.floor(Math.random() * variants.length);
      return variants[idx].id;
    }

    // Exploitation: best performing variant
    let bestVariant = variants[0].id;
    let bestRate = -1;

    for (const variant of variants) {
      const state = this.state.get(variant.id);
      if (state) {
        const rate = state.trials > 0 ? state.totalReward / state.trials : 0;
        if (rate > bestRate) {
          bestRate = rate;
          bestVariant = variant.id;
        }
      }
    }

    return bestVariant;
  }

  update(variantId: string, reward: number): void {
    const state = this.state.get(variantId);
    if (state) {
      state.trials++;
      state.totalReward += reward;
      if (reward > 0) {
        state.successes++;
        state.alpha++;
      } else {
        state.failures++;
        state.beta++;
      }
    }
    this.totalTrials++;
  }

  private getCurrentEpsilon(): number {
    return Math.max(
      this.minEpsilon,
      this.epsilon * Math.pow(this.decayRate, this.totalTrials)
    );
  }

  getState(): Map<string, BanditState> {
    return new Map(this.state);
  }
}

/**
 * Thompson Sampling multi-armed bandit
 */
export class ThompsonSamplingAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "thompson_sampling";
  private state = new Map<string, BanditState>();

  constructor(
    variants: string[],
    priorAlpha: number = 1,
    priorBeta: number = 1
  ) {
    for (const variantId of variants) {
      this.state.set(variantId, {
        variantId,
        trials: 0,
        successes: 0,
        failures: 0,
        totalReward: 0,
        alpha: priorAlpha,
        beta: priorBeta,
      });
    }
  }

  assign(_subjectId: string, variants: VariantWeight[]): string {
    let bestVariant = variants[0].id;
    let bestSample = -1;

    for (const variant of variants) {
      const state = this.state.get(variant.id);
      if (state) {
        // Sample from Beta distribution
        const sample = this.betaSample(state.alpha, state.beta);
        if (sample > bestSample) {
          bestSample = sample;
          bestVariant = variant.id;
        }
      }
    }

    return bestVariant;
  }

  update(variantId: string, reward: number): void {
    const state = this.state.get(variantId);
    if (state) {
      state.trials++;
      state.totalReward += reward;
      if (reward > 0) {
        state.successes++;
        state.alpha++;
      } else {
        state.failures++;
        state.beta++;
      }
    }
  }

  private betaSample(alpha: number, beta: number): number {
    const x = this.gammaSample(alpha);
    const y = this.gammaSample(beta);
    return x / (x + y);
  }

  private gammaSample(shape: number): number {
    if (shape < 1) {
      return this.gammaSample(1 + shape) * Math.pow(Math.random(), 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x: number;
      let v: number;
      do {
        x = this.normalSample();
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = Math.random();
      const x2 = x * x;

      if (u < 1 - 0.0331 * x2 * x2) {
        return d * v;
      }

      if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
        return d * v;
      }
    }
  }

  private normalSample(): number {
    let u: number, v: number, s: number;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);

    return u * Math.sqrt((-2 * Math.log(s)) / s);
  }

  getState(): Map<string, BanditState> {
    return new Map(this.state);
  }

  getProbabilities(): Map<string, number> {
    const probs = new Map<string, number>();
    const samples = 10000;

    const sampleResults = new Map<string, number>();
    for (const [id] of this.state) {
      sampleResults.set(id, 0);
    }

    for (let i = 0; i < samples; i++) {
      let bestId = "";
      let bestSample = -1;
      for (const [id, state] of this.state) {
        const sample = this.betaSample(state.alpha, state.beta);
        if (sample > bestSample) {
          bestSample = sample;
          bestId = id;
        }
      }
      sampleResults.set(bestId, (sampleResults.get(bestId) || 0) + 1);
    }

    for (const [id, count] of sampleResults) {
      probs.set(id, count / samples);
    }

    return probs;
  }
}

/**
 * UCB1 (Upper Confidence Bound) multi-armed bandit
 */
export class UCB1Assignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "ucb1";
  private state = new Map<string, BanditState>();
  private totalTrials: number = 0;
  private explorationFactor: number;

  constructor(variants: string[], explorationFactor: number = 2) {
    this.explorationFactor = explorationFactor;

    for (const variantId of variants) {
      this.state.set(variantId, {
        variantId,
        trials: 0,
        successes: 0,
        failures: 0,
        totalReward: 0,
        alpha: 1,
        beta: 1,
      });
    }
  }

  assign(_subjectId: string, variants: VariantWeight[]): string {
    // First, try each variant once
    for (const variant of variants) {
      const state = this.state.get(variant.id);
      if (state && state.trials === 0) {
        return variant.id;
      }
    }

    // Calculate UCB1 score for each variant
    let bestVariant = variants[0].id;
    let bestScore = -Infinity;

    for (const variant of variants) {
      const state = this.state.get(variant.id);
      if (state && state.trials > 0) {
        const exploitation = state.totalReward / state.trials;
        const exploration = Math.sqrt(
          (this.explorationFactor * Math.log(this.totalTrials)) / state.trials
        );
        const score = exploitation + exploration;

        if (score > bestScore) {
          bestScore = score;
          bestVariant = variant.id;
        }
      }
    }

    return bestVariant;
  }

  update(variantId: string, reward: number): void {
    const state = this.state.get(variantId);
    if (state) {
      state.trials++;
      state.totalReward += reward;
      if (reward > 0) {
        state.successes++;
      } else {
        state.failures++;
      }
    }
    this.totalTrials++;
  }

  getState(): Map<string, BanditState> {
    return new Map(this.state);
  }
}

/**
 * Contextual bandit with linear model
 */
export class ContextualAssignment implements AssignmentStrategy {
  type: AssignmentStrategyType = "contextual";
  private featureNames: string[];
  private weights = new Map<string, number[]>();
  private learningRate: number;

  constructor(
    variants: string[],
    featureNames: string[],
    learningRate: number = 0.01
  ) {
    this.featureNames = featureNames;
    this.learningRate = learningRate;

    // Initialize weights for each variant
    for (const variantId of variants) {
      this.weights.set(
        variantId,
        new Array(featureNames.length).fill(0).map(() => Math.random() * 0.1)
      );
    }
  }

  assign(
    _subjectId: string,
    variants: VariantWeight[],
    context?: AssignmentContext
  ): string {
    if (!context?.attributes) {
      // Fall back to random if no context
      const idx = Math.floor(Math.random() * variants.length);
      return variants[idx].id;
    }

    const features = this.extractFeatures(context.attributes);
    let bestVariant = variants[0].id;
    let bestScore = -Infinity;

    for (const variant of variants) {
      const variantWeights = this.weights.get(variant.id);
      if (variantWeights) {
        const score = this.predict(features, variantWeights);
        if (score > bestScore) {
          bestScore = score;
          bestVariant = variant.id;
        }
      }
    }

    return bestVariant;
  }

  update(variantId: string, reward: number, context?: AssignmentContext): void {
    if (!context?.attributes) {
      return;
    }

    const features = this.extractFeatures(context.attributes);
    const variantWeights = this.weights.get(variantId);

    if (variantWeights) {
      const prediction = this.predict(features, variantWeights);
      const error = reward - prediction;

      // Gradient descent update
      for (let i = 0; i < variantWeights.length; i++) {
        variantWeights[i] += this.learningRate * error * features[i];
      }
    }
  }

  private extractFeatures(attributes: Record<string, unknown>): number[] {
    return this.featureNames.map((name) => {
      const value = attributes[name];
      if (typeof value === "number") {
        return value;
      }
      if (typeof value === "boolean") {
        return value ? 1 : 0;
      }
      if (typeof value === "string") {
        // Simple hash to number
        let hash = 0;
        for (let i = 0; i < value.length; i++) {
          hash = (hash << 5) - hash + value.charCodeAt(i);
        }
        return (hash % 1000) / 1000;
      }
      return 0;
    });
  }

  private predict(features: number[], weights: number[]): number {
    let sum = 0;
    for (let i = 0; i < features.length; i++) {
      sum += features[i] * weights[i];
    }
    return sum;
  }

  getWeights(): Map<string, number[]> {
    return new Map(this.weights);
  }
}

// ============================================================================
// AssignmentManager Class
// ============================================================================

export class AssignmentManager extends EventEmitter<AssignmentManagerEvents> {
  private strategies = new Map<string, AssignmentStrategy>();
  private stickyAssignments = new Map<string, Map<string, string>>(); // experimentId -> subjectId -> variantId
  private config: Required<AssignmentManagerConfig>;

  constructor(config: AssignmentManagerConfig = {}) {
    super();
    this.config = {
      defaultStrategy: config.defaultStrategy ?? "deterministic",
      stickyAssignments: config.stickyAssignments ?? true,
      salt: config.salt ?? "",
    };
  }

  /**
   * Register a strategy for an experiment
   */
  registerStrategy(experimentId: string, strategy: AssignmentStrategy): void {
    this.strategies.set(experimentId, strategy);
  }

  /**
   * Get assignment for a subject
   */
  getAssignment(
    experimentId: string,
    subjectId: string,
    variants: VariantWeight[],
    context?: AssignmentContext
  ): string {
    // Check sticky assignment
    if (this.config.stickyAssignments) {
      const experimentAssignments = this.stickyAssignments.get(experimentId);
      const existingAssignment = experimentAssignments?.get(subjectId);
      if (existingAssignment) {
        return existingAssignment;
      }
    }

    // Get strategy or create default
    let strategy = this.strategies.get(experimentId);
    if (!strategy) {
      strategy = this.createDefaultStrategy(experimentId, variants);
      this.strategies.set(experimentId, strategy);
    }

    // Make assignment
    const variantId = strategy.assign(subjectId, variants, context);

    // Store sticky assignment
    if (this.config.stickyAssignments) {
      if (!this.stickyAssignments.has(experimentId)) {
        this.stickyAssignments.set(experimentId, new Map());
      }
      this.stickyAssignments.get(experimentId)!.set(subjectId, variantId);
    }

    this.emit("assignment", subjectId, variantId, strategy.type);

    return variantId;
  }

  /**
   * Record reward for bandit update
   */
  recordReward(
    experimentId: string,
    variantId: string,
    reward: number,
    context?: AssignmentContext
  ): void {
    const strategy = this.strategies.get(experimentId);
    if (strategy?.update) {
      if (strategy.type === "contextual") {
        (strategy as ContextualAssignment).update(variantId, reward, context);
      } else {
        strategy.update(variantId, reward);
      }
      this.emit("banditUpdate", variantId, reward);
    }
  }

  /**
   * Clear sticky assignments for an experiment
   */
  clearAssignments(experimentId: string): void {
    this.stickyAssignments.delete(experimentId);
  }

  /**
   * Get bandit state for an experiment
   */
  getBanditState(experimentId: string): Map<string, BanditState> | undefined {
    const strategy = this.strategies.get(experimentId);
    if (
      strategy &&
      "getState" in strategy &&
      typeof (strategy as any).getState === "function"
    ) {
      return (strategy as any).getState();
    }
    return undefined;
  }

  private createDefaultStrategy(
    experimentId: string,
    variants: VariantWeight[]
  ): AssignmentStrategy {
    switch (this.config.defaultStrategy) {
      case "random":
        return new RandomAssignment();
      case "deterministic":
        return new DeterministicAssignment(experimentId, this.config.salt);
      case "weighted":
        return new WeightedAssignment();
      case "epsilon_greedy":
        return new EpsilonGreedyAssignment(variants.map((v) => v.id));
      case "thompson_sampling":
        return new ThompsonSamplingAssignment(variants.map((v) => v.id));
      case "ucb1":
        return new UCB1Assignment(variants.map((v) => v.id));
      default:
        return new DeterministicAssignment(experimentId, this.config.salt);
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create an AssignmentManager instance
 */
export function createAssignmentManager(
  config?: AssignmentManagerConfig
): AssignmentManager {
  return new AssignmentManager(config);
}

/**
 * Create a specific assignment strategy
 */
export function createStrategy(
  type: AssignmentStrategyType,
  variants: string[],
  options?: {
    experimentId?: string;
    salt?: string;
    epsilon?: number;
    explorationFactor?: number;
    featureNames?: string[];
    learningRate?: number;
  }
): AssignmentStrategy {
  switch (type) {
    case "random":
      return new RandomAssignment();
    case "deterministic":
      return new DeterministicAssignment(
        options?.experimentId ?? "default",
        options?.salt ?? ""
      );
    case "weighted":
      return new WeightedAssignment();
    case "epsilon_greedy":
      return new EpsilonGreedyAssignment(variants, options?.epsilon);
    case "thompson_sampling":
      return new ThompsonSamplingAssignment(variants);
    case "ucb1":
      return new UCB1Assignment(variants, options?.explorationFactor);
    case "contextual":
      return new ContextualAssignment(
        variants,
        options?.featureNames ?? [],
        options?.learningRate
      );
    default:
      return new RandomAssignment();
  }
}
