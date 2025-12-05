/**
 * NEURECTOMY Predictive Maintenance Engine
 *
 * Anomaly detection and failure prediction for proactive system maintenance
 *
 * @packageDocumentation
 */

import { EventEmitter } from "events";
import type {
  SystemHealthSnapshot,
  ComponentHealth,
  HealthStatus,
  TrendDirection,
  HealthFactorDetail,
  ResourceUtilization,
  ResourceMetric,
  HealthAlert,
  AlertSeverity,
  FailurePrediction,
  FailureType,
  FailureFactor,
  PreventiveAction,
  PreventiveActionType,
  PredictiveConfig,
  ContinuousIntelligenceEvents,
} from "../types.js";

// ==============================================================================
// Configuration
// ==============================================================================

const DEFAULT_CONFIG: PredictiveConfig = {
  enabled: true,
  predictionWindow: 24, // hours
  alertThresholds: {
    warning: 0.7,
    critical: 0.9,
  },
  autoRemediate: false,
  maxAutoRemediateSeverity: "warning",
};

// ==============================================================================
// Time Series Analysis
// ==============================================================================

interface TimeSeriesPoint {
  timestamp: Date;
  value: number;
}

interface TimeSeriesAnalysis {
  trend: TrendDirection;
  slope: number;
  volatility: number;
  seasonality: SeasonalityInfo | null;
  forecast: number[];
  anomalies: AnomalyPoint[];
}

interface SeasonalityInfo {
  period: number; // hours
  amplitude: number;
  phase: number;
}

interface AnomalyPoint {
  timestamp: Date;
  value: number;
  expectedValue: number;
  deviation: number;
  severity: AlertSeverity;
}

/**
 * Time series analyzer for metrics
 */
class TimeSeriesAnalyzer {
  private readonly windowSize = 100;
  private readonly forecastHorizon = 24; // hours

  /**
   * Analyze time series data
   */
  analyze(data: TimeSeriesPoint[]): TimeSeriesAnalysis {
    if (data.length < 10) {
      return {
        trend: "unknown",
        slope: 0,
        volatility: 0,
        seasonality: null,
        forecast: [],
        anomalies: [],
      };
    }

    const values = data.map((p) => p.value);
    const trend = this.detectTrend(values);
    const slope = this.calculateSlope(values);
    const volatility = this.calculateVolatility(values);
    const seasonality = this.detectSeasonality(data);
    const forecast = this.generateForecast(data, slope, seasonality);
    const anomalies = this.detectAnomalies(data);

    return { trend, slope, volatility, seasonality, forecast, anomalies };
  }

  private detectTrend(values: number[]): TrendDirection {
    const slope = this.calculateSlope(values);
    const threshold = 0.01;

    if (Math.abs(slope) < threshold) return "stable";
    return slope > 0 ? "degrading" : "improving";
  }

  private calculateSlope(values: number[]): number {
    const n = values.length;
    if (n < 2) return 0;

    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumX2 += i * i;
    }

    const denominator = n * sumX2 - sumX * sumX;
    if (denominator === 0) return 0;

    return (n * sumXY - sumX * sumY) / denominator;
  }

  private calculateVolatility(values: number[]): number {
    if (values.length < 2) return 0;

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
    const variance =
      squaredDiffs.reduce((a, b) => a + b, 0) / (values.length - 1);

    return Math.sqrt(variance);
  }

  private detectSeasonality(data: TimeSeriesPoint[]): SeasonalityInfo | null {
    if (data.length < 48) return null; // Need at least 2 days

    const values = data.map((p) => p.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const detrended = values.map((v) => v - mean);

    // Simple autocorrelation for period detection
    const periods = [1, 6, 12, 24, 168]; // hours, 6h, 12h, 24h, 1 week
    let bestPeriod = 0;
    let bestCorrelation = 0;

    for (const period of periods) {
      if (data.length < period * 2) continue;

      let correlation = 0;
      let count = 0;

      for (let i = period; i < detrended.length; i++) {
        correlation += detrended[i] * detrended[i - period];
        count++;
      }

      correlation = count > 0 ? correlation / count : 0;
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestPeriod = period;
      }
    }

    if (bestCorrelation < 0.3 || bestPeriod === 0) return null;

    // Calculate amplitude and phase
    const amplitude = Math.max(...values) - Math.min(...values);
    const maxIndex = values.indexOf(Math.max(...values));
    const phase = (maxIndex % bestPeriod) / bestPeriod;

    return { period: bestPeriod, amplitude, phase };
  }

  private generateForecast(
    data: TimeSeriesPoint[],
    slope: number,
    seasonality: SeasonalityInfo | null
  ): number[] {
    if (data.length === 0) return [];

    const lastValue = data[data.length - 1].value;
    const forecast: number[] = [];

    for (let h = 1; h <= this.forecastHorizon; h++) {
      let predicted = lastValue + slope * h;

      if (seasonality) {
        const seasonalComponent =
          (seasonality.amplitude / 2) *
          Math.sin(2 * Math.PI * (h / seasonality.period + seasonality.phase));
        predicted += seasonalComponent;
      }

      forecast.push(Math.max(0, predicted));
    }

    return forecast;
  }

  private detectAnomalies(data: TimeSeriesPoint[]): AnomalyPoint[] {
    if (data.length < 20) return [];

    const values = data.map((p) => p.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = this.calculateVolatility(values);

    const anomalies: AnomalyPoint[] = [];

    for (let i = 0; i < data.length; i++) {
      const deviation = Math.abs(data[i].value - mean) / (stdDev || 1);

      if (deviation > 2) {
        anomalies.push({
          timestamp: data[i].timestamp,
          value: data[i].value,
          expectedValue: mean,
          deviation,
          severity:
            deviation > 4 ? "critical" : deviation > 3 ? "error" : "warning",
        });
      }
    }

    return anomalies;
  }
}

// ==============================================================================
// Health Scorer
// ==============================================================================

interface HealthFactorConfig {
  name: string;
  weight: number;
  warningThreshold: number;
  criticalThreshold: number;
  unit: string;
  higherIsBetter: boolean;
}

const DEFAULT_HEALTH_FACTORS: HealthFactorConfig[] = [
  {
    name: "cpu_usage",
    weight: 0.2,
    warningThreshold: 70,
    criticalThreshold: 90,
    unit: "%",
    higherIsBetter: false,
  },
  {
    name: "memory_usage",
    weight: 0.2,
    warningThreshold: 75,
    criticalThreshold: 90,
    unit: "%",
    higherIsBetter: false,
  },
  {
    name: "disk_usage",
    weight: 0.15,
    warningThreshold: 80,
    criticalThreshold: 95,
    unit: "%",
    higherIsBetter: false,
  },
  {
    name: "error_rate",
    weight: 0.25,
    warningThreshold: 1,
    criticalThreshold: 5,
    unit: "%",
    higherIsBetter: false,
  },
  {
    name: "latency_p99",
    weight: 0.2,
    warningThreshold: 500,
    criticalThreshold: 2000,
    unit: "ms",
    higherIsBetter: false,
  },
];

/**
 * Calculate health scores for components
 */
class HealthScorer {
  private factors: HealthFactorConfig[];

  constructor(factors: HealthFactorConfig[] = DEFAULT_HEALTH_FACTORS) {
    this.factors = factors;
  }

  /**
   * Calculate overall health score from factor values
   */
  calculateScore(values: Record<string, number>): {
    score: number;
    status: HealthStatus;
    factors: HealthFactorDetail[];
  } {
    const factorDetails: HealthFactorDetail[] = [];
    let totalWeight = 0;
    let weightedScore = 0;

    for (const factor of this.factors) {
      const value = values[factor.name];
      if (value === undefined) continue;

      const factorScore = this.calculateFactorScore(factor, value);
      weightedScore += factorScore * factor.weight;
      totalWeight += factor.weight;

      factorDetails.push({
        name: factor.name,
        value,
        warningThreshold: factor.warningThreshold,
        criticalThreshold: factor.criticalThreshold,
        unit: factor.unit,
      });
    }

    const score = totalWeight > 0 ? weightedScore / totalWeight : 100;
    const status = this.scoreToStatus(score);

    return { score, status, factors: factorDetails };
  }

  private calculateFactorScore(
    factor: HealthFactorConfig,
    value: number
  ): number {
    const { warningThreshold, criticalThreshold, higherIsBetter } = factor;

    let normalizedValue: number;
    if (higherIsBetter) {
      // Higher value = better health
      if (value >= warningThreshold) return 100;
      if (value <= criticalThreshold) return 0;
      normalizedValue =
        ((value - criticalThreshold) / (warningThreshold - criticalThreshold)) *
        100;
    } else {
      // Lower value = better health
      if (value <= warningThreshold * 0.5) return 100;
      if (value >= criticalThreshold) return 0;
      normalizedValue =
        ((criticalThreshold - value) /
          (criticalThreshold - warningThreshold * 0.5)) *
        100;
    }

    return Math.max(0, Math.min(100, normalizedValue));
  }

  private scoreToStatus(score: number): HealthStatus {
    if (score >= 90) return "healthy";
    if (score >= 70) return "degraded";
    if (score >= 50) return "warning";
    return "critical";
  }
}

// ==============================================================================
// Failure Predictor
// ==============================================================================

interface PredictionContext {
  component: string;
  history: TimeSeriesPoint[];
  currentHealth: ComponentHealth;
  recentAlerts: HealthAlert[];
}

/**
 * Predicts potential failures based on trends and patterns
 */
class FailurePredictor {
  private analyzer: TimeSeriesAnalyzer;

  constructor() {
    this.analyzer = new TimeSeriesAnalyzer();
  }

  /**
   * Predict potential failures for a component
   */
  predict(context: PredictionContext): FailurePrediction[] {
    const predictions: FailurePrediction[] = [];
    const analysis = this.analyzer.analyze(context.history);

    // Check for resource exhaustion
    const resourcePrediction = this.predictResourceExhaustion(
      context,
      analysis
    );
    if (resourcePrediction) predictions.push(resourcePrediction);

    // Check for cascading failure risk
    const cascadePrediction = this.predictCascadingFailure(context, analysis);
    if (cascadePrediction) predictions.push(cascadePrediction);

    // Check for timeout risk
    const timeoutPrediction = this.predictTimeouts(context, analysis);
    if (timeoutPrediction) predictions.push(timeoutPrediction);

    return predictions.filter((p) => p.probability >= 0.3);
  }

  private predictResourceExhaustion(
    context: PredictionContext,
    analysis: TimeSeriesAnalysis
  ): FailurePrediction | null {
    if (analysis.trend !== "degrading" || analysis.slope <= 0) {
      return null;
    }

    const currentValue =
      context.history.length > 0
        ? context.history[context.history.length - 1].value
        : 0;

    // Estimate time to 100% usage
    const remainingCapacity = 100 - currentValue;
    const hoursToExhaustion =
      analysis.slope > 0 ? remainingCapacity / analysis.slope : Infinity;

    if (hoursToExhaustion > 72) return null; // Ignore if > 3 days

    const probability = Math.min(0.95, 0.5 + (48 - hoursToExhaustion) / 96);
    const predictedTime = new Date(
      Date.now() + hoursToExhaustion * 60 * 60 * 1000
    );

    return {
      id: `prediction-${Date.now()}`,
      component: context.component,
      failureType: "out-of-memory",
      probability,
      predictedTime,
      confidenceInterval: {
        lower: new Date(predictedTime.getTime() - 2 * 60 * 60 * 1000),
        upper: new Date(predictedTime.getTime() + 4 * 60 * 60 * 1000),
      },
      factors: [
        {
          name: "resource_trend",
          weight: 0.6,
          currentValue: analysis.slope,
          trend: "degrading",
        },
        {
          name: "volatility",
          weight: 0.2,
          currentValue: analysis.volatility,
          trend: analysis.volatility > 10 ? "degrading" : "stable",
        },
        {
          name: "current_usage",
          weight: 0.2,
          currentValue,
          trend: currentValue > 70 ? "degrading" : "stable",
        },
      ],
      preventiveActions: [
        {
          id: "scale-up-1",
          type: "scale-up",
          description: "Increase resource allocation by 50%",
          effectiveness: 0.85,
          cost: "medium",
          automatable: true,
        },
        {
          id: "gc-1",
          type: "garbage-collect",
          description: "Force garbage collection to reclaim memory",
          effectiveness: 0.4,
          cost: "low",
          automatable: true,
        },
      ],
    };
  }

  private predictCascadingFailure(
    context: PredictionContext,
    analysis: TimeSeriesAnalysis
  ): FailurePrediction | null {
    // Check for error clustering
    const recentErrors = context.recentAlerts.filter(
      (a) => a.severity === "error" || a.severity === "critical"
    );

    if (recentErrors.length < 3) return null;

    const errorRate =
      recentErrors.length / Math.max(context.recentAlerts.length, 1);
    if (errorRate < 0.3) return null;

    const probability = Math.min(0.9, errorRate * 1.2);

    return {
      id: `prediction-${Date.now()}`,
      component: context.component,
      failureType: "cascading-failure",
      probability,
      predictedTime: new Date(Date.now() + 2 * 60 * 60 * 1000), // 2 hours
      confidenceInterval: {
        lower: new Date(Date.now() + 30 * 60 * 1000),
        upper: new Date(Date.now() + 6 * 60 * 60 * 1000),
      },
      factors: [
        {
          name: "error_rate",
          weight: 0.5,
          currentValue: errorRate * 100,
          trend: "degrading",
        },
        {
          name: "error_count",
          weight: 0.3,
          currentValue: recentErrors.length,
          trend: "degrading",
        },
        {
          name: "health_score",
          weight: 0.2,
          currentValue: context.currentHealth.score,
          trend: context.currentHealth.trend,
        },
      ],
      preventiveActions: [
        {
          id: "circuit-breaker-1",
          type: "failover",
          description: "Enable circuit breaker to isolate failing component",
          effectiveness: 0.8,
          cost: "low",
          automatable: true,
        },
        {
          id: "restart-1",
          type: "restart",
          description: "Graceful restart of the component",
          effectiveness: 0.7,
          cost: "medium",
          automatable: true,
        },
      ],
    };
  }

  private predictTimeouts(
    context: PredictionContext,
    analysis: TimeSeriesAnalysis
  ): FailurePrediction | null {
    // Check if latency is trending up
    if (analysis.trend !== "degrading") return null;

    const currentLatency =
      context.history.length > 0
        ? context.history[context.history.length - 1].value
        : 0;

    // Assume timeout threshold is 5000ms
    const timeoutThreshold = 5000;
    if (currentLatency < timeoutThreshold * 0.5) return null;

    const timeToTimeout =
      analysis.slope > 0
        ? (timeoutThreshold - currentLatency) / analysis.slope
        : Infinity;

    if (timeToTimeout > 12) return null; // Ignore if > 12 hours

    const probability = Math.min(
      0.85,
      0.4 + (currentLatency / timeoutThreshold) * 0.5
    );

    return {
      id: `prediction-${Date.now()}`,
      component: context.component,
      failureType: "timeout",
      probability,
      predictedTime: new Date(Date.now() + timeToTimeout * 60 * 60 * 1000),
      confidenceInterval: {
        lower: new Date(Date.now() + (timeToTimeout - 1) * 60 * 60 * 1000),
        upper: new Date(Date.now() + (timeToTimeout + 2) * 60 * 60 * 1000),
      },
      factors: [
        {
          name: "latency_trend",
          weight: 0.5,
          currentValue: analysis.slope,
          trend: "degrading",
        },
        {
          name: "current_latency",
          weight: 0.3,
          currentValue: currentLatency,
          trend: "degrading",
        },
        {
          name: "volatility",
          weight: 0.2,
          currentValue: analysis.volatility,
          trend: analysis.volatility > 100 ? "degrading" : "stable",
        },
      ],
      preventiveActions: [
        {
          id: "cache-1",
          type: "cache-clear",
          description: "Clear caches to reduce response times",
          effectiveness: 0.5,
          cost: "low",
          automatable: true,
        },
        {
          id: "scale-out-1",
          type: "scale-out",
          description: "Add more instances to distribute load",
          effectiveness: 0.75,
          cost: "high",
          automatable: true,
        },
      ],
    };
  }
}

// ==============================================================================
// Predictive Maintenance Engine
// ==============================================================================

interface MetricHistory {
  component: string;
  metric: string;
  data: TimeSeriesPoint[];
}

/**
 * Main predictive maintenance engine
 *
 * @example
 * ```typescript
 * const engine = new PredictiveMaintenance();
 *
 * // Record metrics
 * await engine.recordMetric('api', 'cpu_usage', 65);
 * await engine.recordMetric('api', 'memory_usage', 78);
 *
 * // Get health snapshot
 * const health = await engine.getHealthSnapshot();
 *
 * // Get failure predictions
 * const predictions = await engine.predictFailures('api');
 * ```
 */
export class PredictiveMaintenance extends EventEmitter {
  private config: PredictiveConfig;
  private metricHistory: Map<string, MetricHistory> = new Map();
  private alerts: HealthAlert[] = [];
  private predictions: FailurePrediction[] = [];
  private healthScorer: HealthScorer;
  private failurePredictor: FailurePredictor;
  private componentHealth: Map<string, ComponentHealth> = new Map();

  constructor(config: Partial<PredictiveConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.healthScorer = new HealthScorer();
    this.failurePredictor = new FailurePredictor();
  }

  /**
   * Record a metric value
   */
  async recordMetric(
    component: string,
    metric: string,
    value: number
  ): Promise<void> {
    const key = `${component}:${metric}`;
    let history = this.metricHistory.get(key);

    if (!history) {
      history = { component, metric, data: [] };
      this.metricHistory.set(key, history);
    }

    history.data.push({ timestamp: new Date(), value });

    // Keep last 1000 points
    if (history.data.length > 1000) {
      history.data = history.data.slice(-1000);
    }

    // Update component health
    await this.updateComponentHealth(component);

    // Check for alerts
    await this.checkAlerts(component, metric, value);
  }

  /**
   * Get current health snapshot
   */
  async getHealthSnapshot(): Promise<SystemHealthSnapshot> {
    const components: ComponentHealth[] = [];
    const processedComponents = new Set<string>();

    for (const [key, history] of this.metricHistory) {
      const component = history.component;
      if (processedComponents.has(component)) continue;
      processedComponents.add(component);

      const health =
        this.componentHealth.get(component) ||
        (await this.calculateComponentHealth(component));
      components.push(health);
    }

    const resources = this.calculateResourceUtilization();
    const activeAlerts = this.alerts.filter((a) => !a.resolvedAt);
    const overallScore = this.calculateOverallScore(components);

    const snapshot: SystemHealthSnapshot = {
      id: `snapshot-${Date.now()}`,
      timestamp: new Date(),
      components,
      resources,
      alerts: activeAlerts,
      overallScore,
    };

    this.emit("health:snapshot", { snapshot });
    return snapshot;
  }

  /**
   * Predict failures for a component
   */
  async predictFailures(component: string): Promise<FailurePrediction[]> {
    const health = this.componentHealth.get(component);
    if (!health) return [];

    // Get history for primary metric (memory usage)
    const historyKey = `${component}:memory_usage`;
    const history = this.metricHistory.get(historyKey);

    const context: PredictionContext = {
      component,
      history: history?.data || [],
      currentHealth: health,
      recentAlerts: this.alerts.filter(
        (a) =>
          a.component === component &&
          Date.now() - a.createdAt.getTime() < 24 * 60 * 60 * 1000
      ),
    };

    const predictions = this.failurePredictor.predict(context);

    for (const prediction of predictions) {
      this.predictions.push(prediction);
      this.emit("health:failure-predicted", { prediction });
    }

    return predictions;
  }

  /**
   * Execute preventive action
   */
  async executePreventiveAction(
    action: PreventiveAction
  ): Promise<{ success: boolean; message: string }> {
    // Check auto-remediation settings
    if (!this.config.autoRemediate) {
      return {
        success: false,
        message: "Auto-remediation is disabled",
      };
    }

    // Emit event for external handler
    this.emit("health:remediation", { action });

    // Simulate action execution
    return {
      success: true,
      message: `Initiated ${action.type}: ${action.description}`,
    };
  }

  /**
   * Acknowledge an alert
   */
  async acknowledgeAlert(alertId: string): Promise<void> {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledgedAt = new Date();
    }
  }

  /**
   * Resolve an alert
   */
  async resolveAlert(alertId: string): Promise<void> {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.resolvedAt = new Date();
    }
  }

  /**
   * Get all predictions
   */
  getPredictions(): FailurePrediction[] {
    return [...this.predictions];
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): HealthAlert[] {
    return this.alerts.filter((a) => !a.resolvedAt);
  }

  /**
   * Get component health
   */
  getComponentHealth(component: string): ComponentHealth | undefined {
    return this.componentHealth.get(component);
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<PredictiveConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // Private methods

  private async updateComponentHealth(component: string): Promise<void> {
    const health = await this.calculateComponentHealth(component);
    this.componentHealth.set(component, health);
  }

  private async calculateComponentHealth(
    component: string
  ): Promise<ComponentHealth> {
    const metrics: Record<string, number> = {};

    // Collect all metrics for this component
    for (const [key, history] of this.metricHistory) {
      if (history.component === component && history.data.length > 0) {
        metrics[history.metric] = history.data[history.data.length - 1].value;
      }
    }

    const { score, status, factors } =
      this.healthScorer.calculateScore(metrics);

    // Calculate trend
    const trend = this.calculateTrend(component);

    // Calculate predicted failure time if degrading
    let predictedFailure: Date | undefined;
    if (trend === "degrading" && score < 50) {
      const hoursToFailure = Math.max(1, score / 10);
      predictedFailure = new Date(Date.now() + hoursToFailure * 60 * 60 * 1000);
    }

    return {
      name: component,
      status,
      score,
      trend,
      predictedFailure,
      factors,
    };
  }

  private calculateTrend(component: string): TrendDirection {
    const memoryHistory = this.metricHistory.get(`${component}:memory_usage`);
    if (!memoryHistory || memoryHistory.data.length < 10) return "unknown";

    const recent = memoryHistory.data.slice(-10);
    const values = recent.map((p) => p.value);

    // Simple linear regression
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0;
    const n = values.length;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumX2 += i * i;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    if (Math.abs(slope) < 0.5) return "stable";
    return slope > 0 ? "degrading" : "improving";
  }

  private calculateResourceUtilization(): ResourceUtilization {
    const createMetric = (metricName: string): ResourceMetric => {
      const values: number[] = [];

      for (const [key, history] of this.metricHistory) {
        if (history.metric === metricName && history.data.length > 0) {
          values.push(...history.data.slice(-100).map((p) => p.value));
        }
      }

      if (values.length === 0) {
        return { current: 0, average: 0, peak: 0 };
      }

      const current = values[values.length - 1];
      const average = values.reduce((a, b) => a + b, 0) / values.length;
      const peak = Math.max(...values);

      return { current, average, peak };
    };

    return {
      cpu: createMetric("cpu_usage"),
      memory: createMetric("memory_usage"),
      disk: createMetric("disk_usage"),
      network: createMetric("network_usage"),
    };
  }

  private calculateOverallScore(components: ComponentHealth[]): number {
    if (components.length === 0) return 100;

    const totalScore = components.reduce((sum, c) => sum + c.score, 0);
    return totalScore / components.length;
  }

  private async checkAlerts(
    component: string,
    metric: string,
    value: number
  ): Promise<void> {
    const { warning, critical } = this.config.alertThresholds;

    // Normalize value to 0-1 scale for comparison
    const normalizedValue = value / 100;

    if (normalizedValue >= critical) {
      await this.createAlert(component, metric, value, "critical");
    } else if (normalizedValue >= warning) {
      await this.createAlert(component, metric, value, "warning");
    }
  }

  private async createAlert(
    component: string,
    metric: string,
    value: number,
    severity: AlertSeverity
  ): Promise<void> {
    // Check for existing unresolved alert
    const existing = this.alerts.find(
      (a) =>
        a.component === component && a.message.includes(metric) && !a.resolvedAt
    );

    if (existing) return;

    const alert: HealthAlert = {
      id: `alert-${Date.now()}`,
      severity,
      component,
      message: `${metric} is at ${value.toFixed(1)}% on ${component}`,
      recommendation: this.getRecommendation(metric, severity),
      createdAt: new Date(),
    };

    this.alerts.push(alert);
    this.emit("health:alert", { alert });
  }

  private getRecommendation(metric: string, severity: AlertSeverity): string {
    const recommendations: Record<string, Record<AlertSeverity, string>> = {
      cpu_usage: {
        info: "Monitor CPU usage",
        warning: "Consider scaling horizontally or optimizing code",
        error: "Immediate action required - scale up or investigate process",
        critical: "CRITICAL: Emergency scale-up required",
      },
      memory_usage: {
        info: "Monitor memory consumption",
        warning: "Check for memory leaks, consider increasing limits",
        error: "Force GC, restart service if needed",
        critical: "CRITICAL: Restart immediately, investigate memory leak",
      },
      disk_usage: {
        info: "Monitor disk usage",
        warning: "Clean up old files, consider expanding storage",
        error: "Immediate cleanup required",
        critical: "CRITICAL: Disk nearly full - emergency cleanup",
      },
      error_rate: {
        info: "Monitor error rates",
        warning: "Investigate error sources",
        error: "Enable circuit breaker, investigate root cause",
        critical: "CRITICAL: Service degraded - failover recommended",
      },
      latency_p99: {
        info: "Monitor response times",
        warning: "Check database queries, cache effectiveness",
        error: "Scale out, optimize hot paths",
        critical: "CRITICAL: Service unresponsive - immediate action",
      },
    };

    return (
      recommendations[metric]?.[severity] ||
      `Take action for ${severity} level ${metric}`
    );
  }
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create a predictive maintenance engine instance
 */
export function createPredictiveMaintenance(
  config?: Partial<PredictiveConfig>
): PredictiveMaintenance {
  return new PredictiveMaintenance(config);
}

export type {
  TimeSeriesPoint,
  TimeSeriesAnalysis,
  AnomalyPoint,
  MetricHistory,
};
