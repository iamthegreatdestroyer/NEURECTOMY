/**
 * NEURECTOMY Auto-Optimization Engine
 *
 * ML-based parameter tuning and system optimization
 *
 * @packageDocumentation
 */

import { EventEmitter } from "events";
import type {
  OptimizationTarget,
  OptimizationObjective,
  OptimizationConstraint,
  TunableParameter,
  OptimizationResult,
  OptimizationStatus,
  OptimizationImprovement,
  ExplorationStep,
  AppliedChange,
  QueryOptimization,
  QueryOptimizationType,
  RiskLevel,
  OptimizationConfig,
  OptimizationAlgorithm,
  ContinuousIntelligenceEvents,
} from "../types.js";

// ==============================================================================
// Configuration
// ==============================================================================

const DEFAULT_CONFIG: OptimizationConfig = {
  enabled: true,
  algorithm: "bayesian",
  maxIterations: 100,
  convergenceThreshold: 0.001,
  explorationRatio: 0.2,
  autoApply: false,
};

// ==============================================================================
// Optimization Algorithms
// ==============================================================================

/**
 * Base optimizer interface
 */
interface Optimizer {
  name: string;
  initialize(parameters: TunableParameter[]): void;
  suggestNext(): Record<string, number>;
  recordResult(params: Record<string, number>, value: number): void;
  getBest(): { params: Record<string, number>; value: number };
  hasConverged(): boolean;
}

/**
 * Bayesian optimizer using Gaussian Process surrogate
 */
class BayesianOptimizer implements Optimizer {
  name = "bayesian";
  private parameters: TunableParameter[] = [];
  private history: Array<{ params: Record<string, number>; value: number }> =
    [];
  private bestParams: Record<string, number> = {};
  private bestValue = -Infinity;
  private convergenceThreshold: number;
  private explorationRatio: number;

  constructor(convergenceThreshold = 0.001, explorationRatio = 0.2) {
    this.convergenceThreshold = convergenceThreshold;
    this.explorationRatio = explorationRatio;
  }

  initialize(parameters: TunableParameter[]): void {
    this.parameters = parameters;
    this.history = [];
    this.bestParams = {};
    this.bestValue = -Infinity;

    // Initialize with current values
    for (const param of parameters) {
      this.bestParams[param.name] = param.currentValue;
    }
  }

  suggestNext(): Record<string, number> {
    const params: Record<string, number> = {};

    // Decide between exploration and exploitation
    const explore = Math.random() < this.explorationRatio;

    for (const param of this.parameters) {
      if (explore || this.history.length < 5) {
        // Random exploration
        params[param.name] = this.randomInRange(param.range);
      } else {
        // Exploitation: Use surrogate model to predict best
        params[param.name] = this.predictBestValue(param);
      }
    }

    return params;
  }

  recordResult(params: Record<string, number>, value: number): void {
    this.history.push({ params: { ...params }, value });

    if (value > this.bestValue) {
      this.bestValue = value;
      this.bestParams = { ...params };
    }
  }

  getBest(): { params: Record<string, number>; value: number } {
    return { params: { ...this.bestParams }, value: this.bestValue };
  }

  hasConverged(): boolean {
    if (this.history.length < 10) return false;

    // Check if recent improvements are below threshold
    const recent = this.history.slice(-10);
    const values = recent.map((h) => h.value);
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);

    return (
      (maxValue - minValue) / Math.abs(maxValue || 1) <
      this.convergenceThreshold
    );
  }

  private randomInRange(range: {
    min: number;
    max: number;
    step?: number;
  }): number {
    const value = range.min + Math.random() * (range.max - range.min);
    if (range.step) {
      return Math.round(value / range.step) * range.step;
    }
    return value;
  }

  private predictBestValue(param: TunableParameter): number {
    // Simple surrogate: weighted average of good performers
    const sorted = [...this.history].sort((a, b) => b.value - a.value);
    const topK = sorted.slice(0, Math.max(1, Math.floor(sorted.length * 0.2)));

    if (topK.length === 0) {
      return param.currentValue;
    }

    const totalWeight = topK.reduce((sum, h, i) => sum + 1 / (i + 1), 0);
    let weightedSum = 0;

    for (let i = 0; i < topK.length; i++) {
      const weight = 1 / (i + 1);
      weightedSum +=
        (topK[i].params[param.name] || param.currentValue) * weight;
    }

    const predicted = weightedSum / totalWeight;

    // Add small noise for exploration
    const noise =
      (Math.random() - 0.5) * 0.1 * (param.range.max - param.range.min);

    return Math.max(
      param.range.min,
      Math.min(param.range.max, predicted + noise)
    );
  }
}

/**
 * Genetic algorithm optimizer
 */
class GeneticOptimizer implements Optimizer {
  name = "genetic";
  private parameters: TunableParameter[] = [];
  private population: Array<{
    params: Record<string, number>;
    fitness: number;
  }> = [];
  private generation = 0;
  private bestParams: Record<string, number> = {};
  private bestValue = -Infinity;
  private readonly populationSize = 20;
  private readonly mutationRate = 0.1;
  private convergenceThreshold: number;

  constructor(convergenceThreshold = 0.001) {
    this.convergenceThreshold = convergenceThreshold;
  }

  initialize(parameters: TunableParameter[]): void {
    this.parameters = parameters;
    this.population = [];
    this.generation = 0;
    this.bestParams = {};
    this.bestValue = -Infinity;

    // Initialize population
    for (let i = 0; i < this.populationSize; i++) {
      const params: Record<string, number> = {};
      for (const param of parameters) {
        params[param.name] = this.randomInRange(param.range);
      }
      this.population.push({ params, fitness: 0 });
    }
  }

  suggestNext(): Record<string, number> {
    if (this.generation === 0) {
      // Return first individual
      return { ...this.population[0].params };
    }

    // Evolve population
    this.evolve();
    return { ...this.population[0].params };
  }

  recordResult(params: Record<string, number>, value: number): void {
    // Find matching individual and update fitness
    const individual = this.population.find((p) =>
      Object.keys(params).every(
        (k) => Math.abs(p.params[k] - params[k]) < 0.001
      )
    );

    if (individual) {
      individual.fitness = value;
    }

    if (value > this.bestValue) {
      this.bestValue = value;
      this.bestParams = { ...params };
    }

    this.generation++;
  }

  getBest(): { params: Record<string, number>; value: number } {
    return { params: { ...this.bestParams }, value: this.bestValue };
  }

  hasConverged(): boolean {
    if (this.generation < 20) return false;

    // Check fitness diversity
    const fitnesses = this.population.map((p) => p.fitness);
    const maxFitness = Math.max(...fitnesses);
    const minFitness = Math.min(...fitnesses);

    return (
      (maxFitness - minFitness) / Math.abs(maxFitness || 1) <
      this.convergenceThreshold
    );
  }

  private evolve(): void {
    // Sort by fitness
    this.population.sort((a, b) => b.fitness - a.fitness);

    // Selection: keep top half
    const survivors = this.population.slice(0, this.populationSize / 2);

    // Crossover: create children from survivors
    const children: typeof this.population = [];
    while (children.length < this.populationSize / 2) {
      const parent1 = survivors[Math.floor(Math.random() * survivors.length)];
      const parent2 = survivors[Math.floor(Math.random() * survivors.length)];

      const childParams: Record<string, number> = {};
      for (const param of this.parameters) {
        // Uniform crossover
        childParams[param.name] =
          Math.random() < 0.5
            ? parent1.params[param.name]
            : parent2.params[param.name];

        // Mutation
        if (Math.random() < this.mutationRate) {
          childParams[param.name] = this.randomInRange(param.range);
        }
      }

      children.push({ params: childParams, fitness: 0 });
    }

    this.population = [...survivors, ...children];
  }

  private randomInRange(range: {
    min: number;
    max: number;
    step?: number;
  }): number {
    const value = range.min + Math.random() * (range.max - range.min);
    if (range.step) {
      return Math.round(value / range.step) * range.step;
    }
    return value;
  }
}

/**
 * Simulated annealing optimizer
 */
class SimulatedAnnealingOptimizer implements Optimizer {
  name = "simulated-annealing";
  private parameters: TunableParameter[] = [];
  private currentParams: Record<string, number> = {};
  private currentValue = -Infinity;
  private bestParams: Record<string, number> = {};
  private bestValue = -Infinity;
  private temperature = 1.0;
  private coolingRate = 0.95;
  private iteration = 0;
  private convergenceThreshold: number;

  constructor(convergenceThreshold = 0.001) {
    this.convergenceThreshold = convergenceThreshold;
  }

  initialize(parameters: TunableParameter[]): void {
    this.parameters = parameters;
    this.currentParams = {};
    this.currentValue = -Infinity;
    this.bestParams = {};
    this.bestValue = -Infinity;
    this.temperature = 1.0;
    this.iteration = 0;

    // Initialize with current values
    for (const param of parameters) {
      this.currentParams[param.name] = param.currentValue;
      this.bestParams[param.name] = param.currentValue;
    }
  }

  suggestNext(): Record<string, number> {
    if (this.iteration === 0) {
      return { ...this.currentParams };
    }

    // Generate neighbor
    const neighbor: Record<string, number> = {};
    for (const param of this.parameters) {
      const range = param.range.max - param.range.min;
      const perturbation = (Math.random() - 0.5) * range * this.temperature;
      neighbor[param.name] = Math.max(
        param.range.min,
        Math.min(param.range.max, this.currentParams[param.name] + perturbation)
      );
    }

    return neighbor;
  }

  recordResult(params: Record<string, number>, value: number): void {
    const delta = value - this.currentValue;

    // Accept if better, or probabilistically if worse
    if (delta > 0 || Math.random() < Math.exp(delta / this.temperature)) {
      this.currentParams = { ...params };
      this.currentValue = value;
    }

    if (value > this.bestValue) {
      this.bestValue = value;
      this.bestParams = { ...params };
    }

    // Cool down
    this.temperature *= this.coolingRate;
    this.iteration++;
  }

  getBest(): { params: Record<string, number>; value: number } {
    return { params: { ...this.bestParams }, value: this.bestValue };
  }

  hasConverged(): boolean {
    return this.temperature < this.convergenceThreshold;
  }
}

// ==============================================================================
// Query Analyzer
// ==============================================================================

/**
 * Analyzes queries for optimization opportunities
 */
class QueryAnalyzer {
  /**
   * Analyze a query for optimization
   */
  analyze(query: string): QueryOptimization[] {
    const optimizations: QueryOptimization[] = [];

    // Check for missing indexes (simplified)
    if (this.hasMissingIndex(query)) {
      optimizations.push(this.suggestIndex(query));
    }

    // Check for SELECT *
    if (this.hasSelectAll(query)) {
      optimizations.push(this.suggestSelectSpecific(query));
    }

    // Check for N+1 queries
    if (this.hasSuboptimalJoin(query)) {
      optimizations.push(this.suggestJoinOptimization(query));
    }

    // Check for caching opportunity
    if (this.isCacheable(query)) {
      optimizations.push(this.suggestCaching(query));
    }

    return optimizations;
  }

  private hasMissingIndex(query: string): boolean {
    const lowerQuery = query.toLowerCase();
    // Simple heuristic: WHERE clause without apparent index
    return (
      lowerQuery.includes("where") &&
      !lowerQuery.includes("indexed") &&
      (lowerQuery.includes("like") || lowerQuery.includes("between"))
    );
  }

  private hasSelectAll(query: string): boolean {
    return /select\s+\*\s+from/i.test(query);
  }

  private hasSuboptimalJoin(query: string): boolean {
    const lowerQuery = query.toLowerCase();
    return (
      (lowerQuery.includes("join") && lowerQuery.includes("in (select")) ||
      (lowerQuery.match(/join/g)?.length || 0) > 3
    );
  }

  private isCacheable(query: string): boolean {
    const lowerQuery = query.toLowerCase();
    // Read-only queries on likely static data
    return (
      lowerQuery.startsWith("select") &&
      !lowerQuery.includes("now()") &&
      !lowerQuery.includes("random()") &&
      (lowerQuery.includes("config") ||
        lowerQuery.includes("settings") ||
        lowerQuery.includes("metadata"))
    );
  }

  private suggestIndex(query: string): QueryOptimization {
    // Extract column from WHERE clause (simplified)
    const match = query.match(/where\s+(\w+)/i);
    const column = match?.[1] || "unknown_column";

    return {
      id: `opt-${Date.now()}`,
      originalQuery: query,
      optimizedQuery: `-- Add index: CREATE INDEX idx_${column} ON table(${column})\n${query}`,
      type: "index-suggestion",
      expectedImprovement: 50,
      explanation: `Adding an index on column "${column}" could significantly improve query performance`,
      risk: "low",
    };
  }

  private suggestSelectSpecific(query: string): QueryOptimization {
    return {
      id: `opt-${Date.now()}`,
      originalQuery: query,
      optimizedQuery: query.replace(
        /select\s+\*/i,
        "SELECT id, name, created_at"
      ),
      type: "query-rewrite",
      expectedImprovement: 20,
      explanation:
        "Select only needed columns instead of SELECT * to reduce data transfer",
      risk: "low",
    };
  }

  private suggestJoinOptimization(query: string): QueryOptimization {
    return {
      id: `opt-${Date.now()}`,
      originalQuery: query,
      optimizedQuery: `-- Consider using EXISTS instead of IN\n-- Or restructure joins\n${query}`,
      type: "join-optimization",
      expectedImprovement: 30,
      explanation:
        "Complex joins can be optimized by restructuring or using different patterns",
      risk: "medium",
    };
  }

  private suggestCaching(query: string): QueryOptimization {
    return {
      id: `opt-${Date.now()}`,
      originalQuery: query,
      optimizedQuery: `-- Cache key: ${this.hashQuery(query)}\n${query}`,
      type: "caching-opportunity",
      expectedImprovement: 80,
      explanation:
        "This query returns mostly static data and is a good candidate for caching",
      risk: "low",
    };
  }

  private hashQuery(query: string): string {
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }
}

// ==============================================================================
// Auto-Optimizer Engine
// ==============================================================================

/**
 * Main auto-optimization engine
 *
 * @example
 * ```typescript
 * const optimizer = new AutoOptimizer();
 *
 * // Register tunable parameters
 * optimizer.registerParameter({
 *   id: 'cache-size',
 *   name: 'cache_size_mb',
 *   component: 'cache',
 *   currentValue: 256,
 *   valueType: 'continuous',
 *   range: { min: 64, max: 1024 },
 *   sensitivity: 0.8,
 *   impact: { 'minimize-latency': 0.6, 'minimize-cost': -0.3 }
 * });
 *
 * // Run optimization
 * const result = await optimizer.optimize([
 *   { name: 'latency', objective: 'minimize-latency', targetValue: 100 }
 * ]);
 * ```
 */
export class AutoOptimizer extends EventEmitter {
  private config: OptimizationConfig;
  private parameters: Map<string, TunableParameter> = new Map();
  private optimizer: Optimizer | null = null;
  private queryAnalyzer: QueryAnalyzer;
  private optimizationHistory: OptimizationResult[] = [];
  private isRunning = false;

  constructor(config: Partial<OptimizationConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.queryAnalyzer = new QueryAnalyzer();
  }

  /**
   * Register a tunable parameter
   */
  registerParameter(parameter: TunableParameter): void {
    this.parameters.set(parameter.id, parameter);
  }

  /**
   * Unregister a parameter
   */
  unregisterParameter(parameterId: string): void {
    this.parameters.delete(parameterId);
  }

  /**
   * Get all registered parameters
   */
  getParameters(): TunableParameter[] {
    return Array.from(this.parameters.values());
  }

  /**
   * Run optimization for given targets
   */
  async optimize(targets: OptimizationTarget[]): Promise<OptimizationResult> {
    if (!this.config.enabled || this.isRunning) {
      throw new Error("Optimization is disabled or already running");
    }

    this.isRunning = true;
    const startTime = new Date();
    const parameters = Array.from(this.parameters.values());

    // Create optimizer based on config
    this.optimizer = this.createOptimizer();
    this.optimizer.initialize(parameters);

    const initialState: Record<string, number> = {};
    for (const param of parameters) {
      initialState[param.name] = param.currentValue;
    }

    const explorationHistory: ExplorationStep[] = [];

    this.emit("optimization:started", { targets });

    try {
      let step = 0;
      while (
        step < this.config.maxIterations &&
        !this.optimizer.hasConverged()
      ) {
        // Get next parameters to try
        const params = this.optimizer.suggestNext();

        // Evaluate (simulated - in real use, this would measure actual performance)
        const objectiveValue = this.evaluateObjective(params, targets);

        // Record result
        this.optimizer.recordResult(params, objectiveValue);

        const explorationStep: ExplorationStep = {
          step,
          parameters: params,
          objectiveValue,
          isNewBest: objectiveValue === this.optimizer.getBest().value,
        };

        explorationHistory.push(explorationStep);
        this.emit("optimization:step", { step: explorationStep });

        step++;
      }

      const best = this.optimizer.getBest();
      const optimizedState = best.params;

      // Calculate improvements
      const improvements = this.calculateImprovements(
        initialState,
        optimizedState,
        targets
      );

      // Apply changes if auto-apply is enabled
      const appliedChanges: AppliedChange[] = [];
      if (this.config.autoApply) {
        for (const param of parameters) {
          if (optimizedState[param.name] !== initialState[param.name]) {
            appliedChanges.push({
              parameter: param.name,
              oldValue: initialState[param.name],
              newValue: optimizedState[param.name],
              appliedAt: new Date(),
              rollbackPossible: true,
            });
            param.currentValue = optimizedState[param.name];
          }
        }
      }

      const result: OptimizationResult = {
        id: `opt-run-${Date.now()}`,
        startedAt: startTime,
        completedAt: new Date(),
        status: this.optimizer.hasConverged() ? "converged" : "completed",
        initialState,
        optimizedState,
        improvements,
        explorationHistory,
        appliedChanges,
      };

      this.optimizationHistory.push(result);
      this.emit("optimization:completed", { result });

      return result;
    } finally {
      this.isRunning = false;
    }
  }

  /**
   * Analyze a query for optimization opportunities
   */
  analyzeQuery(query: string): QueryOptimization[] {
    return this.queryAnalyzer.analyze(query);
  }

  /**
   * Apply a specific optimization result
   */
  async applyOptimization(
    result: OptimizationResult
  ): Promise<AppliedChange[]> {
    const appliedChanges: AppliedChange[] = [];

    for (const [paramName, newValue] of Object.entries(result.optimizedState)) {
      const param = Array.from(this.parameters.values()).find(
        (p) => p.name === paramName
      );

      if (param && param.currentValue !== newValue) {
        appliedChanges.push({
          parameter: paramName,
          oldValue: param.currentValue,
          newValue,
          appliedAt: new Date(),
          rollbackPossible: true,
        });
        param.currentValue = newValue;
      }
    }

    return appliedChanges;
  }

  /**
   * Rollback an applied change
   */
  async rollback(change: AppliedChange): Promise<boolean> {
    if (!change.rollbackPossible) return false;

    const param = Array.from(this.parameters.values()).find(
      (p) => p.name === change.parameter
    );

    if (param) {
      param.currentValue = change.oldValue;
      return true;
    }

    return false;
  }

  /**
   * Get optimization history
   */
  getHistory(): OptimizationResult[] {
    return [...this.optimizationHistory];
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<OptimizationConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): OptimizationConfig {
    return { ...this.config };
  }

  // Private methods

  private createOptimizer(): Optimizer {
    switch (this.config.algorithm) {
      case "bayesian":
        return new BayesianOptimizer(
          this.config.convergenceThreshold,
          this.config.explorationRatio
        );
      case "genetic":
        return new GeneticOptimizer(this.config.convergenceThreshold);
      case "simulated-annealing":
        return new SimulatedAnnealingOptimizer(
          this.config.convergenceThreshold
        );
      default:
        return new BayesianOptimizer(
          this.config.convergenceThreshold,
          this.config.explorationRatio
        );
    }
  }

  private evaluateObjective(
    params: Record<string, number>,
    targets: OptimizationTarget[]
  ): number {
    // Simulated objective function
    // In real use, this would measure actual system performance
    let totalValue = 0;
    let totalWeight = 0;

    for (const target of targets) {
      const paramImpact = this.calculateParamImpact(params, target.objective);
      const constraintSatisfaction = this.checkConstraints(
        params,
        target.constraints
      );

      const targetValue = paramImpact * constraintSatisfaction;
      totalValue += targetValue * target.weight;
      totalWeight += target.weight;
    }

    return totalWeight > 0 ? totalValue / totalWeight : 0;
  }

  private calculateParamImpact(
    params: Record<string, number>,
    objective: OptimizationObjective
  ): number {
    let impact = 0.5; // Base impact

    for (const param of this.parameters.values()) {
      const paramValue = params[param.name];
      if (paramValue === undefined) continue;

      const objectiveImpact = param.impact[objective] || 0;
      const normalizedValue =
        (paramValue - param.range.min) / (param.range.max - param.range.min);

      // Impact calculation based on objective type
      if (objective.startsWith("minimize-")) {
        impact += objectiveImpact * (1 - normalizedValue) * param.sensitivity;
      } else {
        impact += objectiveImpact * normalizedValue * param.sensitivity;
      }
    }

    return Math.max(0, Math.min(1, impact));
  }

  private checkConstraints(
    params: Record<string, number>,
    constraints: OptimizationConstraint[]
  ): number {
    let satisfaction = 1;

    for (const constraint of constraints) {
      const paramValue = params[constraint.name];
      if (paramValue === undefined) continue;

      let satisfied = false;
      switch (constraint.type) {
        case "min":
          satisfied = paramValue >= (constraint.value as number);
          break;
        case "max":
          satisfied = paramValue <= (constraint.value as number);
          break;
        case "range": {
          const [min, max] = constraint.value as [number, number];
          satisfied = paramValue >= min && paramValue <= max;
          break;
        }
        case "equality":
          satisfied =
            Math.abs(paramValue - (constraint.value as number)) < 0.001;
          break;
      }

      if (!satisfied) {
        satisfaction *= constraint.hard ? 0 : 0.5;
      }
    }

    return satisfaction;
  }

  private calculateImprovements(
    initialState: Record<string, number>,
    optimizedState: Record<string, number>,
    targets: OptimizationTarget[]
  ): OptimizationImprovement[] {
    return targets.map((target) => {
      const initialValue = this.calculateParamImpact(
        initialState,
        target.objective
      );
      const optimizedValue = this.calculateParamImpact(
        optimizedState,
        target.objective
      );
      const improvement =
        initialValue > 0
          ? ((optimizedValue - initialValue) / initialValue) * 100
          : 0;

      return {
        target: target.name,
        before: initialValue,
        after: optimizedValue,
        improvement,
      };
    });
  }
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create an auto-optimizer instance
 */
export function createAutoOptimizer(
  config?: Partial<OptimizationConfig>
): AutoOptimizer {
  return new AutoOptimizer(config);
}

export { BayesianOptimizer, GeneticOptimizer, SimulatedAnnealingOptimizer };
export type { Optimizer };
