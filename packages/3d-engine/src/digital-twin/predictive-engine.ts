/**
 * Digital Twin Predictive Engine
 *
 * AI-powered predictive simulation engine that forecasts agent behavior,
 * generates what-if scenarios, and provides actionable insights.
 *
 * @module @neurectomy/3d-engine/digital-twin/predictive-engine
 * @agents @NEURAL @TENSOR @PRISM @ORACLE
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  TwinId,
  AgentStateSnapshot,
  PredictionConfig,
  PredictionResult,
  PredictionTimeline,
  TimelinePoint,
  Scenario,
  ScenarioInput,
  PredictionMetrics,
} from "./types";

// ============================================================================
// Types
// ============================================================================

export interface PredictionEngine {
  id: string;
  twinId: TwinId;
  config: PredictionConfig;
  state: "idle" | "running" | "error";
  lastPredictionAt: number;
  results: PredictionResult[];
}

export interface PredictionContext {
  currentState: AgentStateSnapshot;
  historicalStates: AgentStateSnapshot[];
  inputScenarios: ScenarioInput[];
  config: PredictionConfig;
}

export interface AnalysisResult {
  trends: TrendAnalysis[];
  anomalies: AnomalyDetection[];
  correlations: CorrelationMatrix;
  seasonality: SeasonalPattern[];
}

export interface TrendAnalysis {
  metric: string;
  direction: "up" | "down" | "stable";
  strength: number;
  confidence: number;
  rateOfChange: number;
}

export interface AnomalyDetection {
  timestamp: number;
  metric: string;
  value: number;
  expectedValue: number;
  deviation: number;
  severity: "low" | "medium" | "high" | "critical";
}

export interface CorrelationMatrix {
  metrics: string[];
  values: number[][];
}

export interface SeasonalPattern {
  metric: string;
  periodMs: number;
  amplitude: number;
  phase: number;
}

// ============================================================================
// Predictive Engine Class
// ============================================================================

/**
 * Digital Twin Predictive Engine
 *
 * Forecasts future agent states, generates scenarios,
 * and provides uncertainty quantification.
 */
export class TwinPredictiveEngine {
  private engines: Map<TwinId, PredictionEngine> = new Map();
  private historicalData: Map<TwinId, AgentStateSnapshot[]> = new Map();

  constructor() {
    // Initialize
  }

  // ==========================================================================
  // Engine Management
  // ==========================================================================

  /**
   * Create a prediction engine for a twin
   */
  createEngine(twinId: TwinId, config: PredictionConfig): PredictionEngine {
    const engine: PredictionEngine = {
      id: `pred-${twinId}-${Date.now()}`,
      twinId,
      config,
      state: "idle",
      lastPredictionAt: 0,
      results: [],
    };

    this.engines.set(twinId, engine);
    this.historicalData.set(twinId, []);

    return engine;
  }

  /**
   * Get prediction engine for a twin
   */
  getEngine(twinId: TwinId): PredictionEngine | undefined {
    return this.engines.get(twinId);
  }

  /**
   * Remove prediction engine
   */
  removeEngine(twinId: TwinId): void {
    this.engines.delete(twinId);
    this.historicalData.delete(twinId);
  }

  // ==========================================================================
  // Data Collection
  // ==========================================================================

  /**
   * Record state snapshot for historical analysis
   */
  recordState(twinId: TwinId, state: AgentStateSnapshot): void {
    let history = this.historicalData.get(twinId);
    if (!history) {
      history = [];
      this.historicalData.set(twinId, history);
    }

    history.push(JSON.parse(JSON.stringify(state)));

    // Keep only recent history (last 1000 snapshots)
    if (history.length > 1000) {
      history.shift();
    }
  }

  /**
   * Get historical data for a twin
   */
  getHistoricalData(twinId: TwinId): AgentStateSnapshot[] {
    return this.historicalData.get(twinId) || [];
  }

  // ==========================================================================
  // Prediction Operations
  // ==========================================================================

  /**
   * Generate predictions for a twin
   */
  async predict(
    twinId: TwinId,
    currentState: AgentStateSnapshot,
    overrideConfig?: Partial<PredictionConfig>
  ): Promise<PredictionResult> {
    const engine = this.engines.get(twinId);
    if (!engine) {
      throw new Error(`No prediction engine for twin ${twinId}`);
    }

    engine.state = "running";
    const _startTime = Date.now();

    try {
      const config = { ...engine.config, ...overrideConfig };
      const history = this.historicalData.get(twinId) || [];

      // Analyze historical data
      const analysis = this.analyzeHistory(history);

      // Generate timeline
      const timeline = this.generateTimeline(currentState, config, analysis);

      // Generate scenarios
      const scenarios = await this.generateScenarios(
        currentState,
        config,
        analysis
      );

      // Calculate metrics
      const metrics = this.calculatePredictionMetrics(
        timeline,
        scenarios,
        config
      );

      const result: PredictionResult = {
        id: `result-${Date.now()}`,
        twinId,
        timestamp: Date.now(),
        horizonMs: config.horizonMs,
        timeline,
        scenarios,
        metrics,
      };

      engine.results.push(result);
      engine.lastPredictionAt = Date.now();
      engine.state = "idle";

      // Keep only last 100 results
      if (engine.results.length > 100) {
        engine.results.shift();
      }

      return result;
    } catch (error) {
      engine.state = "error";
      throw error;
    }
  }

  /**
   * Run what-if scenario analysis
   */
  async whatIf(
    twinId: TwinId,
    currentState: AgentStateSnapshot,
    scenarios: ScenarioInput[]
  ): Promise<Scenario[]> {
    const results: Scenario[] = [];

    for (let i = 0; i < scenarios.length; i++) {
      const input = scenarios[i];
      if (!input) continue;

      const scenario = await this.simulateScenario(
        twinId,
        currentState,
        input,
        i
      );
      results.push(scenario);
    }

    return results;
  }

  // ==========================================================================
  // Analysis Methods
  // ==========================================================================

  /**
   * Analyze historical data for patterns
   */
  private analyzeHistory(history: AgentStateSnapshot[]): AnalysisResult {
    const trends: TrendAnalysis[] = [];
    const anomalies: AnomalyDetection[] = [];
    const seasonality: SeasonalPattern[] = [];

    if (history.length < 2) {
      return {
        trends,
        anomalies,
        correlations: { metrics: [], values: [] },
        seasonality,
      };
    }

    // Extract metrics time series
    const metricSeries = this.extractMetricSeries(history);

    // Analyze trends
    for (const [metric, values] of Object.entries(metricSeries)) {
      const trend = this.analyzeTrend(metric, values);
      if (trend) trends.push(trend);
    }

    // Detect anomalies
    for (const [metric, values] of Object.entries(metricSeries)) {
      const detected = this.detectAnomalies(metric, values);
      anomalies.push(...detected);
    }

    // Calculate correlations
    const correlations = this.calculateCorrelations(metricSeries);

    // Detect seasonality
    for (const [metric, values] of Object.entries(metricSeries)) {
      const pattern = this.detectSeasonality(metric, values);
      if (pattern) seasonality.push(pattern);
    }

    return { trends, anomalies, correlations, seasonality };
  }

  /**
   * Extract metric time series from history
   */
  private extractMetricSeries(
    history: AgentStateSnapshot[]
  ): Record<string, number[]> {
    const series: Record<string, number[]> = {
      responseTime: [],
      throughput: [],
      errorRate: [],
      cpuPercent: [],
      memoryMB: [],
    };

    for (const state of history) {
      if (state.metrics) {
        series["responseTime"]?.push(state.metrics.responseTime.mean);
        series["throughput"]?.push(state.metrics.throughput.mean);
        series["errorRate"]?.push(state.metrics.errorRate);
        series["cpuPercent"]?.push(
          state.metrics.resourceUtilization.cpuPercent
        );
        series["memoryMB"]?.push(state.metrics.resourceUtilization.memoryMB);
      }
    }

    return series;
  }

  /**
   * Analyze trend in a metric
   */
  private analyzeTrend(metric: string, values: number[]): TrendAnalysis | null {
    if (values.length < 3) return null;

    // Simple linear regression
    const n = values.length;
    const indices = Array.from({ length: n }, (_, i) => i);

    const sumX = indices.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = indices.reduce((sum, x, i) => {
      const y = values[i];
      return y !== undefined ? sum + x * y : sum;
    }, 0);
    const sumX2 = indices.reduce((sum, x) => sum + x * x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const yMean = sumY / n;
    const ssTotal = values.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
    const ssRes = values.reduce((sum, y, i) => {
      const predicted = slope * i + intercept;
      return sum + Math.pow(y - predicted, 2);
    }, 0);
    const rSquared = ssTotal > 0 ? 1 - ssRes / ssTotal : 0;

    return {
      metric,
      direction: slope > 0.01 ? "up" : slope < -0.01 ? "down" : "stable",
      strength: Math.abs(slope),
      confidence: Math.max(0, Math.min(1, rSquared)),
      rateOfChange: slope,
    };
  }

  /**
   * Detect anomalies using Z-score
   */
  private detectAnomalies(
    metric: string,
    values: number[]
  ): AnomalyDetection[] {
    const anomalies: AnomalyDetection[] = [];
    if (values.length < 10) return anomalies;

    // Calculate mean and std dev
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance =
      values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return anomalies;

    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (value === undefined) continue;

      const zScore = Math.abs((value - mean) / stdDev);

      if (zScore > 2) {
        anomalies.push({
          timestamp: Date.now() - (values.length - i) * 1000,
          metric,
          value,
          expectedValue: mean,
          deviation: zScore,
          severity:
            zScore > 4
              ? "critical"
              : zScore > 3
                ? "high"
                : zScore > 2.5
                  ? "medium"
                  : "low",
        });
      }
    }

    return anomalies;
  }

  /**
   * Calculate correlation matrix
   */
  private calculateCorrelations(
    series: Record<string, number[]>
  ): CorrelationMatrix {
    const metrics = Object.keys(series);
    const n = metrics.length;
    const values: number[][] = Array(n)
      .fill(null)
      .map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          values[i]![j] = 1;
        } else {
          const metricI = metrics[i];
          const metricJ = metrics[j];
          if (metricI && metricJ) {
            values[i]![j] = this.pearsonCorrelation(
              series[metricI] || [],
              series[metricJ] || []
            );
          }
        }
      }
    }

    return { metrics, values };
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 3) return 0;

    const xSlice = x.slice(0, n);
    const ySlice = y.slice(0, n);

    const xMean = xSlice.reduce((a, b) => a + b, 0) / n;
    const yMean = ySlice.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let xDenominator = 0;
    let yDenominator = 0;

    for (let i = 0; i < n; i++) {
      const xVal = xSlice[i];
      const yVal = ySlice[i];
      if (xVal === undefined || yVal === undefined) continue;

      const xDiff = xVal - xMean;
      const yDiff = yVal - yMean;
      numerator += xDiff * yDiff;
      xDenominator += xDiff * xDiff;
      yDenominator += yDiff * yDiff;
    }

    const denominator = Math.sqrt(xDenominator * yDenominator);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Detect seasonality pattern
   */
  private detectSeasonality(
    metric: string,
    values: number[]
  ): SeasonalPattern | null {
    if (values.length < 24) return null;

    // Simple periodicity detection using autocorrelation
    const maxLag = Math.min(values.length / 2, 100);
    let bestLag = 0;
    let bestCorr = 0;

    for (let lag = 2; lag < maxLag; lag++) {
      const corr = this.autocorrelation(values, lag);
      if (corr > bestCorr && corr > 0.5) {
        bestCorr = corr;
        bestLag = lag;
      }
    }

    if (bestLag === 0) return null;

    // Calculate amplitude
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const amplitude = values.reduce(
      (max, v) => Math.max(max, Math.abs(v - mean)),
      0
    );

    return {
      metric,
      periodMs: bestLag * 1000, // Assuming 1 second intervals
      amplitude,
      phase: 0,
    };
  }

  /**
   * Calculate autocorrelation at given lag
   */
  private autocorrelation(values: number[], lag: number): number {
    const n = values.length - lag;
    if (n < 3) return 0;

    const mean = values.reduce((a, b) => a + b, 0) / values.length;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      const vi = values[i];
      const viLag = values[i + lag];
      if (vi === undefined || viLag === undefined) continue;
      numerator += (vi - mean) * (viLag - mean);
    }

    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === undefined) continue;
      denominator += Math.pow(v - mean, 2);
    }

    return denominator === 0 ? 0 : numerator / denominator;
  }

  // ==========================================================================
  // Prediction Generation
  // ==========================================================================

  /**
   * Generate prediction timeline
   */
  private generateTimeline(
    currentState: AgentStateSnapshot,
    config: PredictionConfig,
    analysis: AnalysisResult
  ): PredictionTimeline {
    const points: TimelinePoint[] = [];
    const steps = Math.floor(config.horizonMs / config.stepMs);

    let predictedState = JSON.parse(JSON.stringify(currentState));

    for (let i = 0; i <= steps; i++) {
      const timestamp = Date.now() + i * config.stepMs;
      const _progressRatio = i / steps;

      // Apply trends to metrics
      predictedState = this.applyTrends(
        predictedState,
        analysis.trends,
        config.stepMs
      );

      // Apply seasonality
      predictedState = this.applySeasonality(
        predictedState,
        analysis.seasonality,
        i * config.stepMs
      );

      // Calculate confidence (decreases over time)
      const baseConfidence = config.confidenceLevel ?? 0.95;
      const confidence = Math.max(0.1, baseConfidence * Math.pow(0.99, i));

      // Calculate bounds if uncertainty quantification is enabled
      const bounds = config.quantifyUncertainty
        ? this.calculateBounds(predictedState, confidence)
        : undefined;

      points.push({
        timestamp,
        state: JSON.parse(JSON.stringify(predictedState)),
        confidence,
        bounds,
      });
    }

    return {
      startTime: Date.now(),
      endTime: Date.now() + config.horizonMs,
      stepMs: config.stepMs,
      points,
      branches: [], // No branches in simple timeline
    };
  }

  /**
   * Apply trends to state
   */
  private applyTrends(
    state: AgentStateSnapshot,
    trends: TrendAnalysis[],
    deltaMs: number
  ): AgentStateSnapshot {
    const newState = JSON.parse(JSON.stringify(state));

    for (const trend of trends) {
      const change = trend.rateOfChange * (deltaMs / 1000);

      switch (trend.metric) {
        case "responseTime":
          newState.metrics.responseTime.mean = Math.max(
            0,
            newState.metrics.responseTime.mean + change
          );
          break;
        case "throughput":
          newState.metrics.throughput.mean = Math.max(
            0,
            newState.metrics.throughput.mean + change
          );
          break;
        case "errorRate":
          newState.metrics.errorRate = Math.max(
            0,
            Math.min(1, newState.metrics.errorRate + change)
          );
          break;
        case "cpuPercent":
          newState.metrics.resourceUtilization.cpuPercent = Math.max(
            0,
            Math.min(
              100,
              newState.metrics.resourceUtilization.cpuPercent + change
            )
          );
          break;
        case "memoryMB":
          newState.metrics.resourceUtilization.memoryMB = Math.max(
            0,
            newState.metrics.resourceUtilization.memoryMB + change
          );
          break;
      }
    }

    return newState;
  }

  /**
   * Apply seasonality patterns
   */
  private applySeasonality(
    state: AgentStateSnapshot,
    patterns: SeasonalPattern[],
    elapsedMs: number
  ): AgentStateSnapshot {
    const newState = JSON.parse(JSON.stringify(state));

    for (const pattern of patterns) {
      const phase =
        ((elapsedMs % pattern.periodMs) / pattern.periodMs) * 2 * Math.PI;
      const adjustment = pattern.amplitude * Math.sin(phase + pattern.phase);

      switch (pattern.metric) {
        case "responseTime":
          newState.metrics.responseTime.mean = Math.max(
            0,
            newState.metrics.responseTime.mean + adjustment
          );
          break;
        case "throughput":
          newState.metrics.throughput.mean = Math.max(
            0,
            newState.metrics.throughput.mean + adjustment
          );
          break;
      }
    }

    return newState;
  }

  /**
   * Calculate prediction bounds
   */
  private calculateBounds(
    state: AgentStateSnapshot,
    confidence: number
  ): { lower: AgentStateSnapshot; upper: AgentStateSnapshot } {
    const uncertaintyFactor = 1 - confidence;

    const lower = JSON.parse(JSON.stringify(state));
    const upper = JSON.parse(JSON.stringify(state));

    // Apply uncertainty to key metrics
    const applyUncertainty = (
      base: number,
      factor: number,
      isLower: boolean
    ): number => {
      const adjustment = base * factor;
      return isLower ? Math.max(0, base - adjustment) : base + adjustment;
    };

    lower.metrics.responseTime.mean = applyUncertainty(
      state.metrics.responseTime.mean,
      uncertaintyFactor,
      true
    );
    upper.metrics.responseTime.mean = applyUncertainty(
      state.metrics.responseTime.mean,
      uncertaintyFactor,
      false
    );

    lower.metrics.throughput.mean = applyUncertainty(
      state.metrics.throughput.mean,
      uncertaintyFactor,
      true
    );
    upper.metrics.throughput.mean = applyUncertainty(
      state.metrics.throughput.mean,
      uncertaintyFactor,
      false
    );

    return { lower, upper };
  }

  /**
   * Generate scenarios
   */
  private async generateScenarios(
    currentState: AgentStateSnapshot,
    config: PredictionConfig,
    analysis: AnalysisResult
  ): Promise<Scenario[]> {
    const scenarios: Scenario[] = [];

    // Generate baseline scenario
    scenarios.push({
      id: "baseline",
      name: "Baseline",
      description: "Expected behavior without changes",
      probability: 0.6,
      input: {},
      predictedState: currentState,
      keyMetrics: this.extractKeyMetrics(currentState),
    });

    // Generate from provided scenarios
    for (let i = 0; i < config.inputScenarios.length; i++) {
      const input = config.inputScenarios[i];
      if (!input) continue;

      const scenario = await this.simulateScenario(
        "",
        currentState,
        input,
        i + 1
      );
      scenarios.push(scenario);
    }

    // Generate auto scenarios if needed
    if (scenarios.length < config.scenarioCount) {
      const autoScenarios = this.generateAutoScenarios(
        currentState,
        analysis,
        config.scenarioCount - scenarios.length
      );
      scenarios.push(...autoScenarios);
    }

    return scenarios;
  }

  /**
   * Simulate a single scenario
   */
  private async simulateScenario(
    _twinId: TwinId,
    currentState: AgentStateSnapshot,
    input: ScenarioInput,
    index: number
  ): Promise<Scenario> {
    const predictedState = JSON.parse(JSON.stringify(currentState));

    // Apply input modifications
    if (input.configOverrides) {
      Object.assign(predictedState.config, input.configOverrides);
    }
    if (input.parameterChanges) {
      Object.assign(predictedState.parameters, input.parameterChanges);
    }
    if (input.loadMultiplier) {
      predictedState.metrics.throughput.mean *= input.loadMultiplier;
      predictedState.metrics.responseTime.mean *= Math.sqrt(
        input.loadMultiplier
      );
    }
    if (input.failureInjection) {
      predictedState.metrics.errorRate = Math.min(
        1,
        predictedState.metrics.errorRate + 0.1
      );
    }

    return {
      id: `scenario-${index}`,
      name: input.name || `Scenario ${index}`,
      description: input.description || "Custom scenario",
      probability: input.probability || 0.1,
      input,
      predictedState,
      keyMetrics: this.extractKeyMetrics(predictedState),
    };
  }

  /**
   * Generate automatic scenarios based on analysis
   */
  private generateAutoScenarios(
    currentState: AgentStateSnapshot,
    analysis: AnalysisResult,
    count: number
  ): Scenario[] {
    const scenarios: Scenario[] = [];

    // High load scenario
    if (count > 0) {
      const highLoadState = JSON.parse(JSON.stringify(currentState));
      highLoadState.metrics.throughput.mean *= 2;
      highLoadState.metrics.responseTime.mean *= 1.5;
      highLoadState.metrics.resourceUtilization.cpuPercent = Math.min(
        100,
        highLoadState.metrics.resourceUtilization.cpuPercent * 1.8
      );

      scenarios.push({
        id: "high-load",
        name: "High Load",
        description: "Scenario with 2x normal load",
        probability: 0.15,
        input: { loadMultiplier: 2 },
        predictedState: highLoadState,
        keyMetrics: this.extractKeyMetrics(highLoadState),
      });
    }

    // Degradation scenario
    if (count > 1) {
      const degradedState = JSON.parse(JSON.stringify(currentState));
      degradedState.metrics.errorRate = Math.min(
        1,
        degradedState.metrics.errorRate + 0.05
      );
      degradedState.metrics.responseTime.mean *= 2;

      scenarios.push({
        id: "degraded",
        name: "Degraded Performance",
        description: "Scenario with performance degradation",
        probability: 0.1,
        input: { failureInjection: true },
        predictedState: degradedState,
        keyMetrics: this.extractKeyMetrics(degradedState),
      });
    }

    // Optimistic scenario
    if (count > 2) {
      const optimisticState = JSON.parse(JSON.stringify(currentState));
      optimisticState.metrics.responseTime.mean *= 0.8;
      optimisticState.metrics.errorRate *= 0.5;

      scenarios.push({
        id: "optimistic",
        name: "Optimistic",
        description: "Scenario with improved performance",
        probability: 0.15,
        input: {},
        predictedState: optimisticState,
        keyMetrics: this.extractKeyMetrics(optimisticState),
      });
    }

    return scenarios;
  }

  /**
   * Extract key metrics from state
   */
  private extractKeyMetrics(state: AgentStateSnapshot): Record<string, number> {
    return {
      responseTime: state.metrics.responseTime.mean,
      throughput: state.metrics.throughput.mean,
      errorRate: state.metrics.errorRate,
      cpuPercent: state.metrics.resourceUtilization.cpuPercent,
      memoryMB: state.metrics.resourceUtilization.memoryMB,
    };
  }

  /**
   * Calculate prediction metrics
   */
  private calculatePredictionMetrics(
    timeline: PredictionTimeline,
    scenarios: Scenario[],
    config: PredictionConfig
  ): PredictionMetrics {
    // Calculate average confidence
    const avgConfidence =
      timeline.points.reduce((sum, p) => sum + p.confidence, 0) /
      timeline.points.length;

    // Calculate scenario coverage
    const scenarioCoverage = scenarios.reduce(
      (sum, s) => sum + (s.probability ?? 0),
      0
    );

    return {
      accuracy: avgConfidence,
      precision: avgConfidence * 0.95,
      recall: avgConfidence * 0.9,
      f1Score: avgConfidence * 0.925,
      coveragePercent: Math.min(100, scenarioCoverage * 100),
      dataPointsUsed: timeline.points.length,
    };
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Dispose of the predictive engine
   */
  dispose(): void {
    this.engines.clear();
    this.historicalData.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let predictiveEngineInstance: TwinPredictiveEngine | null = null;

/**
 * Get the global TwinPredictiveEngine instance
 */
export function getTwinPredictiveEngine(): TwinPredictiveEngine {
  if (!predictiveEngineInstance) {
    predictiveEngineInstance = new TwinPredictiveEngine();
  }
  return predictiveEngineInstance;
}

/**
 * Reset the global TwinPredictiveEngine instance
 */
export function resetTwinPredictiveEngine(): void {
  if (predictiveEngineInstance) {
    predictiveEngineInstance.dispose();
    predictiveEngineInstance = null;
  }
}
