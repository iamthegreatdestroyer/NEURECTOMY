/**
 * Consciousness-Aware Heatmaps
 *
 * Breakthrough Innovation: Visualize agent reasoning as dynamic 3D thought patterns.
 * Maps cognitive activity to spatial heatmaps showing attention, decision-making,
 * and information flow within AI agents.
 *
 * This innovation combines:
 * - Agent introspection capabilities
 * - Real-time attention pattern analysis
 * - 3D volumetric visualization
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations/consciousness-heatmaps
 * @agents @NEXUS @NEURAL @CANVAS
 * @innovation Consciousness-Aware Heatmaps (Forge × Foundry × Twin)
 */

import type {
  UnifiedEntity,
  UnifiedEvent,
  UniversalId,
  Timestamp,
} from "../types";

import { CrossDomainEventBridge } from "../event-bridge";

// ============================================================================
// Consciousness Heatmap Types
// ============================================================================

/**
 * Attention pattern representing where the agent is "looking"
 */
export interface AttentionPattern {
  /** Pattern ID */
  id: string;

  /** Agent/entity this pattern belongs to */
  entityId: UniversalId;

  /** Timestamp of capture */
  timestamp: Timestamp;

  /** Attention weights across components */
  weights: AttentionWeights;

  /** Derived insights */
  insights: AttentionInsights;
}

/**
 * Raw attention weights
 */
export interface AttentionWeights {
  /** Layer-wise attention (for transformer-based agents) */
  layerAttention: Map<string, number[][]>;

  /** Component attention (for modular agents) */
  componentAttention: Map<string, number>;

  /** Input feature attention */
  inputAttention: Map<string, number>;

  /** Cross-attention between modules */
  crossAttention: Map<string, Map<string, number>>;

  /** Temporal attention (history lookback) */
  temporalAttention: number[];
}

/**
 * Derived insights from attention
 */
export interface AttentionInsights {
  /** Primary focus area */
  primaryFocus: string;

  /** Focus intensity (0-1) */
  focusIntensity: number;

  /** Attention entropy (high = distributed, low = focused) */
  entropy: number;

  /** Detected reasoning patterns */
  reasoningPatterns: ReasoningPattern[];

  /** Anomalies detected */
  anomalies: AttentionAnomaly[];
}

/**
 * Detected reasoning pattern
 */
export interface ReasoningPattern {
  type: "sequential" | "parallel" | "recursive" | "associative" | "comparative";
  confidence: number;
  involvedComponents: string[];
  duration: number;
}

/**
 * Attention anomaly
 */
export interface AttentionAnomaly {
  type: "spike" | "dead" | "oscillation" | "drift" | "saturation";
  component: string;
  severity: number; // 0-1
  description: string;
}

/**
 * 3D heatmap voxel
 */
export interface HeatmapVoxel {
  /** Position in 3D space */
  position: { x: number; y: number; z: number };

  /** Heat value (0-1) */
  intensity: number;

  /** Color based on type of activity */
  color: { r: number; g: number; b: number; a: number };

  /** Associated component */
  component: string;

  /** Activity type */
  activityType: "attention" | "activation" | "flow" | "decision";
}

/**
 * Complete 3D heatmap
 */
export interface ConsciousnessHeatmap {
  id: string;
  entityId: UniversalId;
  timestamp: Timestamp;

  /** Heatmap resolution */
  resolution: { x: number; y: number; z: number };

  /** Bounding box */
  bounds: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };

  /** Voxel data */
  voxels: HeatmapVoxel[];

  /** Flow lines showing information movement */
  flowLines: FlowLine[];

  /** Decision points */
  decisionPoints: DecisionPoint[];

  /** Overall metrics */
  metrics: HeatmapMetrics;
}

/**
 * Information flow line
 */
export interface FlowLine {
  id: string;
  source: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
  intensity: number;
  type: "feedforward" | "feedback" | "lateral" | "skip";
  color: { r: number; g: number; b: number; a: number };
}

/**
 * Decision point visualization
 */
export interface DecisionPoint {
  id: string;
  position: { x: number; y: number; z: number };
  alternatives: Array<{
    option: string;
    probability: number;
    selected: boolean;
  }>;
  confidence: number;
  timestamp: Timestamp;
}

/**
 * Heatmap metrics
 */
export interface HeatmapMetrics {
  averageIntensity: number;
  maxIntensity: number;
  activeVoxelRatio: number;
  flowDensity: number;
  decisionCount: number;
  entropyScore: number;
}

/**
 * Heatmap configuration
 */
export interface HeatmapConfig {
  /** Entity IDs to monitor */
  entityIds: UniversalId[];

  /** Update frequency in ms */
  updateFrequency: number;

  /** Resolution (voxels per dimension) */
  resolution: number;

  /** Intensity threshold for display */
  intensityThreshold: number;

  /** Color scheme */
  colorScheme: "thermal" | "viridis" | "plasma" | "cool-warm" | "custom";

  /** Custom color stops */
  customColors?: Array<{
    value: number;
    color: { r: number; g: number; b: number };
  }>;

  /** Show flow lines */
  showFlowLines: boolean;

  /** Show decision points */
  showDecisionPoints: boolean;

  /** Temporal smoothing (frames) */
  temporalSmoothing: number;

  /** 3D rendering mode */
  renderMode: "volume" | "particles" | "mesh" | "hybrid";
}

// ============================================================================
// Event Types
// ============================================================================

export type HeatmapEventType =
  | "heatmap:created"
  | "heatmap:updated"
  | "heatmap:disposed"
  | "attention:captured"
  | "pattern:detected"
  | "anomaly:detected"
  | "decision:recorded";

export interface HeatmapEvent {
  type: HeatmapEventType;
  entityId: UniversalId;
  timestamp: Timestamp;
  data: unknown;
}

// ============================================================================
// Consciousness Heatmap Generator
// ============================================================================

/**
 * Consciousness-Aware Heatmap Generator
 *
 * Generates 3D heatmaps visualizing agent cognitive activity.
 */
export class ConsciousnessHeatmapGenerator {
  private static instance: ConsciousnessHeatmapGenerator | null = null;

  private configs: Map<UniversalId, HeatmapConfig> = new Map();
  private heatmaps: Map<UniversalId, ConsciousnessHeatmap> = new Map();
  private attentionHistory: Map<UniversalId, AttentionPattern[]> = new Map();
  private updateTimers: Map<UniversalId, ReturnType<typeof setInterval>> =
    new Map();
  private eventBridge: CrossDomainEventBridge;

  // Listeners
  private listeners: Set<(event: HeatmapEvent) => void> = new Set();

  private constructor() {
    this.eventBridge = CrossDomainEventBridge.getInstance();
    this.setupEventListeners();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): ConsciousnessHeatmapGenerator {
    if (!ConsciousnessHeatmapGenerator.instance) {
      ConsciousnessHeatmapGenerator.instance =
        new ConsciousnessHeatmapGenerator();
    }
    return ConsciousnessHeatmapGenerator.instance;
  }

  /**
   * Reset instance
   */
  static resetInstance(): void {
    if (ConsciousnessHeatmapGenerator.instance) {
      ConsciousnessHeatmapGenerator.instance.dispose();
      ConsciousnessHeatmapGenerator.instance = null;
    }
  }

  /**
   * Dispose generator
   */
  dispose(): void {
    for (const entityId of this.configs.keys()) {
      this.stopMonitoring(entityId);
    }

    this.configs.clear();
    this.heatmaps.clear();
    this.attentionHistory.clear();
    this.listeners.clear();
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  private setupEventListeners(): void {
    // Listen for agent activity events
    this.eventBridge.subscribe<{ entityId: string; activity: unknown }>(
      "metrics:updated",
      (event) => this.handleMetricsUpdate(event)
    );

    // Listen for agent state changes
    this.eventBridge.subscribe<{ entityId: string; state: unknown }>(
      "state:changed",
      (event) => this.handleStateChange(event)
    );
  }

  private handleMetricsUpdate(
    event: UnifiedEvent<{ entityId: string; activity: unknown }>
  ): void {
    const { entityId, activity } = event.payload ?? {};
    if (!entityId || !this.configs.has(entityId as UniversalId)) return;

    // Extract attention data from activity
    const attention = this.extractAttentionFromActivity(activity);
    if (attention) {
      this.captureAttention(entityId as UniversalId, attention);
    }
  }

  private handleStateChange(
    event: UnifiedEvent<{ entityId: string; state: unknown }>
  ): void {
    const { entityId, state } = event.payload ?? {};
    if (!entityId || !this.configs.has(entityId as UniversalId)) return;

    // Look for decision-making in state changes
    this.detectDecisionPoints(entityId as UniversalId, state);
  }

  // ==========================================================================
  // Monitoring Control
  // ==========================================================================

  /**
   * Start monitoring an entity
   */
  startMonitoring(
    entityId: UniversalId,
    config: Partial<HeatmapConfig> = {}
  ): void {
    const fullConfig: HeatmapConfig = {
      entityIds: [entityId],
      updateFrequency: 100,
      resolution: 32,
      intensityThreshold: 0.1,
      colorScheme: "thermal",
      showFlowLines: true,
      showDecisionPoints: true,
      temporalSmoothing: 5,
      renderMode: "hybrid",
      ...config,
    };

    this.configs.set(entityId, fullConfig);
    this.attentionHistory.set(entityId, []);

    // Initialize heatmap
    const heatmap = this.createEmptyHeatmap(entityId, fullConfig);
    this.heatmaps.set(entityId, heatmap);

    // Start update timer
    const timer = setInterval(
      () => this.updateHeatmap(entityId),
      fullConfig.updateFrequency
    );
    this.updateTimers.set(entityId, timer);

    this.emitEvent({
      type: "heatmap:created",
      entityId,
      timestamp: Date.now(),
      data: { config: fullConfig },
    });
  }

  /**
   * Stop monitoring an entity
   */
  stopMonitoring(entityId: UniversalId): void {
    const timer = this.updateTimers.get(entityId);
    if (timer) {
      clearInterval(timer);
      this.updateTimers.delete(entityId);
    }

    this.configs.delete(entityId);
    this.heatmaps.delete(entityId);
    this.attentionHistory.delete(entityId);

    this.emitEvent({
      type: "heatmap:disposed",
      entityId,
      timestamp: Date.now(),
      data: {},
    });
  }

  /**
   * Get current heatmap for entity
   */
  getHeatmap(entityId: UniversalId): ConsciousnessHeatmap | undefined {
    return this.heatmaps.get(entityId);
  }

  /**
   * Get attention history for entity
   */
  getAttentionHistory(entityId: UniversalId): AttentionPattern[] {
    return this.attentionHistory.get(entityId) ?? [];
  }

  // ==========================================================================
  // Attention Capture
  // ==========================================================================

  /**
   * Capture attention pattern
   */
  captureAttention(
    entityId: UniversalId,
    weights: Partial<AttentionWeights>
  ): void {
    const config = this.configs.get(entityId);
    if (!config) return;

    const fullWeights: AttentionWeights = {
      layerAttention: weights.layerAttention ?? new Map(),
      componentAttention: weights.componentAttention ?? new Map(),
      inputAttention: weights.inputAttention ?? new Map(),
      crossAttention: weights.crossAttention ?? new Map(),
      temporalAttention: weights.temporalAttention ?? [],
    };

    const insights = this.analyzeAttention(fullWeights);

    const pattern: AttentionPattern = {
      id: `attention-${entityId}-${Date.now()}`,
      entityId,
      timestamp: Date.now(),
      weights: fullWeights,
      insights,
    };

    // Add to history
    const history = this.attentionHistory.get(entityId)!;
    history.push(pattern);

    // Trim history based on temporal smoothing
    const maxHistory = config.temporalSmoothing * 10;
    while (history.length > maxHistory) {
      history.shift();
    }

    this.emitEvent({
      type: "attention:captured",
      entityId,
      timestamp: Date.now(),
      data: { pattern },
    });

    // Check for patterns and anomalies
    if (insights.reasoningPatterns.length > 0) {
      this.emitEvent({
        type: "pattern:detected",
        entityId,
        timestamp: Date.now(),
        data: { patterns: insights.reasoningPatterns },
      });
    }

    if (insights.anomalies.length > 0) {
      this.emitEvent({
        type: "anomaly:detected",
        entityId,
        timestamp: Date.now(),
        data: { anomalies: insights.anomalies },
      });
    }
  }

  /**
   * Manually record attention (for external systems)
   */
  recordAttention(
    entityId: UniversalId,
    component: string,
    intensity: number
  ): void {
    const config = this.configs.get(entityId);
    if (!config) return;

    const componentAttention = new Map([[component, intensity]]);
    this.captureAttention(entityId, { componentAttention });
  }

  // ==========================================================================
  // Attention Analysis
  // ==========================================================================

  /**
   * Analyze attention weights for insights
   */
  private analyzeAttention(weights: AttentionWeights): AttentionInsights {
    // Find primary focus
    let maxAttention = 0;
    let primaryFocus = "unknown";

    for (const [component, attention] of weights.componentAttention) {
      if (attention > maxAttention) {
        maxAttention = attention;
        primaryFocus = component;
      }
    }

    // Calculate entropy
    const attentionValues = Array.from(weights.componentAttention.values());
    const entropy = this.calculateEntropy(attentionValues);

    // Detect reasoning patterns
    const reasoningPatterns = this.detectReasoningPatterns(weights);

    // Detect anomalies
    const anomalies = this.detectAnomalies(weights);

    return {
      primaryFocus,
      focusIntensity: maxAttention,
      entropy,
      reasoningPatterns,
      anomalies,
    };
  }

  /**
   * Calculate entropy of attention distribution
   */
  private calculateEntropy(values: number[]): number {
    if (values.length === 0) return 0;

    const sum = values.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;

    const normalized = values.map((v) => v / sum);

    let entropy = 0;
    for (const p of normalized) {
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }

    // Normalize to [0, 1]
    const maxEntropy = Math.log2(values.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }

  /**
   * Detect reasoning patterns from attention
   */
  private detectReasoningPatterns(
    weights: AttentionWeights
  ): ReasoningPattern[] {
    const patterns: ReasoningPattern[] = [];

    // Sequential: High attention on consecutive components
    const components = Array.from(weights.componentAttention.keys());
    let sequentialScore = 0;
    for (let i = 0; i < components.length - 1; i++) {
      const current = weights.componentAttention.get(components[i]) ?? 0;
      const next = weights.componentAttention.get(components[i + 1]) ?? 0;
      if (current > 0.3 && next > 0.3) {
        sequentialScore += 0.2;
      }
    }
    if (sequentialScore > 0.5) {
      patterns.push({
        type: "sequential",
        confidence: Math.min(sequentialScore, 1),
        involvedComponents: components.filter(
          (c) => (weights.componentAttention.get(c) ?? 0) > 0.3
        ),
        duration: 0,
      });
    }

    // Parallel: High attention on multiple components simultaneously
    const highAttentionCount = Array.from(
      weights.componentAttention.values()
    ).filter((v) => v > 0.5).length;
    if (highAttentionCount >= 3) {
      patterns.push({
        type: "parallel",
        confidence: Math.min(highAttentionCount / 5, 1),
        involvedComponents: components.filter(
          (c) => (weights.componentAttention.get(c) ?? 0) > 0.5
        ),
        duration: 0,
      });
    }

    // Recursive: High cross-attention from component to itself
    for (const [source, targets] of weights.crossAttention) {
      const selfAttention = targets.get(source) ?? 0;
      if (selfAttention > 0.7) {
        patterns.push({
          type: "recursive",
          confidence: selfAttention,
          involvedComponents: [source],
          duration: 0,
        });
      }
    }

    return patterns;
  }

  /**
   * Detect attention anomalies
   */
  private detectAnomalies(weights: AttentionWeights): AttentionAnomaly[] {
    const anomalies: AttentionAnomaly[] = [];

    for (const [component, attention] of weights.componentAttention) {
      // Spike: Very high attention
      if (attention > 0.95) {
        anomalies.push({
          type: "spike",
          component,
          severity: attention,
          description: `Attention spike detected in ${component}`,
        });
      }

      // Dead: Zero or near-zero attention
      if (attention < 0.01) {
        anomalies.push({
          type: "dead",
          component,
          severity: 1 - attention,
          description: `No attention detected in ${component}`,
        });
      }

      // Saturation: All components at similar high level
      const values = Array.from(weights.componentAttention.values());
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance =
        values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) /
        values.length;
      if (mean > 0.8 && variance < 0.01) {
        anomalies.push({
          type: "saturation",
          component: "all",
          severity: mean,
          description: "Attention saturated across all components",
        });
        break; // Only report once
      }
    }

    return anomalies;
  }

  // ==========================================================================
  // Heatmap Generation
  // ==========================================================================

  /**
   * Create empty heatmap
   */
  private createEmptyHeatmap(
    entityId: UniversalId,
    config: HeatmapConfig
  ): ConsciousnessHeatmap {
    const res = config.resolution;

    return {
      id: `heatmap-${entityId}-${Date.now()}`,
      entityId,
      timestamp: Date.now(),
      resolution: { x: res, y: res, z: res },
      bounds: {
        min: { x: -5, y: -5, z: -5 },
        max: { x: 5, y: 5, z: 5 },
      },
      voxels: [],
      flowLines: [],
      decisionPoints: [],
      metrics: {
        averageIntensity: 0,
        maxIntensity: 0,
        activeVoxelRatio: 0,
        flowDensity: 0,
        decisionCount: 0,
        entropyScore: 0,
      },
    };
  }

  /**
   * Update heatmap from attention history
   */
  private updateHeatmap(entityId: UniversalId): void {
    const config = this.configs.get(entityId);
    const heatmap = this.heatmaps.get(entityId);
    const history = this.attentionHistory.get(entityId);

    if (!config || !heatmap || !history || history.length === 0) return;

    const res = config.resolution;

    // Apply temporal smoothing
    const smoothedAttention = this.smoothAttention(
      history,
      config.temporalSmoothing
    );

    // Generate voxels
    const voxels: HeatmapVoxel[] = [];
    const components = Array.from(smoothedAttention.keys());

    for (let i = 0; i < components.length; i++) {
      const component = components[i];
      const intensity = smoothedAttention.get(component) ?? 0;

      if (intensity < config.intensityThreshold) continue;

      // Map component to 3D position
      const angle = (i / components.length) * Math.PI * 2;
      const radius = 3;
      const position = {
        x: Math.cos(angle) * radius,
        y: intensity * 4 - 2,
        z: Math.sin(angle) * radius,
      };

      // Get color from scheme
      const color = this.getColorForIntensity(intensity, config);

      voxels.push({
        position,
        intensity,
        color,
        component,
        activityType: "attention",
      });
    }

    // Generate flow lines
    const flowLines = config.showFlowLines
      ? this.generateFlowLines(smoothedAttention, history)
      : [];

    // Calculate metrics
    const metrics = this.calculateHeatmapMetrics(
      voxels,
      flowLines,
      heatmap.decisionPoints
    );

    // Update heatmap
    heatmap.timestamp = Date.now();
    heatmap.voxels = voxels;
    heatmap.flowLines = flowLines;
    heatmap.metrics = metrics;

    this.emitEvent({
      type: "heatmap:updated",
      entityId,
      timestamp: Date.now(),
      data: { heatmap },
    });

    // Publish to event bridge
    this.eventBridge.publish({
      id: `heatmap-${entityId}-${Date.now()}`,
      type: "metrics:updated",
      payload: { entityId, heatmap },
      timestamp: Date.now(),
      sourceDomain: "foundry",
      targetDomains: ["forge"],
    });
  }

  /**
   * Smooth attention over temporal window
   */
  private smoothAttention(
    history: AttentionPattern[],
    windowSize: number
  ): Map<string, number> {
    const smoothed = new Map<string, number>();
    const counts = new Map<string, number>();

    const recentHistory = history.slice(-windowSize);

    for (const pattern of recentHistory) {
      for (const [component, attention] of pattern.weights.componentAttention) {
        const current = smoothed.get(component) ?? 0;
        const count = counts.get(component) ?? 0;
        smoothed.set(component, current + attention);
        counts.set(component, count + 1);
      }
    }

    // Average
    for (const [component, total] of smoothed) {
      const count = counts.get(component) ?? 1;
      smoothed.set(component, total / count);
    }

    return smoothed;
  }

  /**
   * Generate flow lines from cross-attention
   */
  private generateFlowLines(
    attention: Map<string, number>,
    history: AttentionPattern[]
  ): FlowLine[] {
    const lines: FlowLine[] = [];
    const components = Array.from(attention.keys());

    if (history.length === 0) return lines;

    const recent = history[history.length - 1];

    for (const [source, targets] of recent.weights.crossAttention) {
      const sourceIndex = components.indexOf(source);
      if (sourceIndex === -1) continue;

      for (const [target, strength] of targets) {
        const targetIndex = components.indexOf(target);
        if (targetIndex === -1 || strength < 0.1) continue;

        const sourceAngle = (sourceIndex / components.length) * Math.PI * 2;
        const targetAngle = (targetIndex / components.length) * Math.PI * 2;
        const radius = 3;

        lines.push({
          id: `flow-${source}-${target}`,
          source: {
            x: Math.cos(sourceAngle) * radius,
            y: (attention.get(source) ?? 0) * 4 - 2,
            z: Math.sin(sourceAngle) * radius,
          },
          target: {
            x: Math.cos(targetAngle) * radius,
            y: (attention.get(target) ?? 0) * 4 - 2,
            z: Math.sin(targetAngle) * radius,
          },
          intensity: strength,
          type: source === target ? "feedback" : "feedforward",
          color: { r: 1, g: 1, b: 1, a: strength },
        });
      }
    }

    return lines;
  }

  /**
   * Get color for intensity value
   */
  private getColorForIntensity(
    intensity: number,
    config: HeatmapConfig
  ): { r: number; g: number; b: number; a: number } {
    switch (config.colorScheme) {
      case "thermal":
        // Black -> Red -> Yellow -> White
        if (intensity < 0.33) {
          return { r: intensity * 3, g: 0, b: 0, a: 1 };
        } else if (intensity < 0.66) {
          return { r: 1, g: (intensity - 0.33) * 3, b: 0, a: 1 };
        } else {
          const t = (intensity - 0.66) * 3;
          return { r: 1, g: 1, b: t, a: 1 };
        }

      case "viridis":
        // Purple -> Blue -> Green -> Yellow
        return {
          r: 0.267 + intensity * 0.993 * 0.733,
          g: 0.004 + intensity * 0.906,
          b: 0.329 + intensity * (0.143 - 0.329),
          a: 1,
        };

      case "plasma":
        // Purple -> Pink -> Orange -> Yellow
        return {
          r: 0.05 + intensity * 0.95,
          g: intensity * 0.9,
          b: 0.53 - intensity * 0.53,
          a: 1,
        };

      case "cool-warm":
        // Blue -> White -> Red
        if (intensity < 0.5) {
          const t = intensity * 2;
          return { r: t, g: t, b: 1, a: 1 };
        } else {
          const t = (intensity - 0.5) * 2;
          return { r: 1, g: 1 - t, b: 1 - t, a: 1 };
        }

      case "custom":
        if (config.customColors && config.customColors.length >= 2) {
          return this.interpolateCustomColors(intensity, config.customColors);
        }
        // Fallback to thermal
        return this.getColorForIntensity(intensity, {
          ...config,
          colorScheme: "thermal",
        });

      default:
        return { r: intensity, g: intensity, b: intensity, a: 1 };
    }
  }

  /**
   * Interpolate custom color stops
   */
  private interpolateCustomColors(
    value: number,
    stops: Array<{ value: number; color: { r: number; g: number; b: number } }>
  ): { r: number; g: number; b: number; a: number } {
    const sortedStops = [...stops].sort((a, b) => a.value - b.value);

    // Find surrounding stops
    let lower = sortedStops[0];
    let upper = sortedStops[sortedStops.length - 1];

    for (let i = 0; i < sortedStops.length - 1; i++) {
      if (value >= sortedStops[i].value && value <= sortedStops[i + 1].value) {
        lower = sortedStops[i];
        upper = sortedStops[i + 1];
        break;
      }
    }

    // Interpolate
    const t =
      upper.value !== lower.value
        ? (value - lower.value) / (upper.value - lower.value)
        : 0;

    return {
      r: lower.color.r + (upper.color.r - lower.color.r) * t,
      g: lower.color.g + (upper.color.g - lower.color.g) * t,
      b: lower.color.b + (upper.color.b - lower.color.b) * t,
      a: 1,
    };
  }

  // ==========================================================================
  // Decision Points
  // ==========================================================================

  /**
   * Detect and record decision points
   */
  private detectDecisionPoints(entityId: UniversalId, state: unknown): void {
    const heatmap = this.heatmaps.get(entityId);
    const config = this.configs.get(entityId);

    if (!heatmap || !config || !config.showDecisionPoints) return;

    // Look for decision markers in state
    const stateObj = state as Record<string, unknown> | null;
    if (!stateObj?.decisions) return;

    const decisions = stateObj.decisions as Array<{
      options: Array<{ name: string; probability: number }>;
      selected: string;
    }>;

    for (const decision of decisions) {
      const position = {
        x: (Math.random() - 0.5) * 8,
        y: 2,
        z: (Math.random() - 0.5) * 8,
      };

      const alternatives = decision.options.map((opt) => ({
        option: opt.name,
        probability: opt.probability,
        selected: opt.name === decision.selected,
      }));

      const selectedProb =
        alternatives.find((a) => a.selected)?.probability ?? 0.5;

      const point: DecisionPoint = {
        id: `decision-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        position,
        alternatives,
        confidence: selectedProb,
        timestamp: Date.now(),
      };

      heatmap.decisionPoints.push(point);

      // Limit decision point history
      while (heatmap.decisionPoints.length > 50) {
        heatmap.decisionPoints.shift();
      }

      this.emitEvent({
        type: "decision:recorded",
        entityId,
        timestamp: Date.now(),
        data: { decision: point },
      });
    }
  }

  // ==========================================================================
  // Metrics Calculation
  // ==========================================================================

  /**
   * Calculate heatmap metrics
   */
  private calculateHeatmapMetrics(
    voxels: HeatmapVoxel[],
    flowLines: FlowLine[],
    decisionPoints: DecisionPoint[]
  ): HeatmapMetrics {
    const intensities = voxels.map((v) => v.intensity);
    const totalVoxels = Math.pow(32, 3); // Assuming 32^3 resolution

    return {
      averageIntensity:
        intensities.length > 0
          ? intensities.reduce((a, b) => a + b, 0) / intensities.length
          : 0,
      maxIntensity: intensities.length > 0 ? Math.max(...intensities) : 0,
      activeVoxelRatio: voxels.length / totalVoxels,
      flowDensity:
        flowLines.reduce((acc, line) => acc + line.intensity, 0) /
        Math.max(flowLines.length, 1),
      decisionCount: decisionPoints.length,
      entropyScore: this.calculateEntropy(intensities),
    };
  }

  /**
   * Extract attention from generic activity object
   */
  private extractAttentionFromActivity(
    activity: unknown
  ): Partial<AttentionWeights> | null {
    const activityObj = activity as Record<string, unknown> | null;
    if (!activityObj) return null;

    const componentAttention = new Map<string, number>();

    // Try to extract attention from various formats
    if (activityObj.attention) {
      const attn = activityObj.attention as Record<string, number>;
      for (const [key, value] of Object.entries(attn)) {
        componentAttention.set(key, value);
      }
    } else if (activityObj.activations) {
      const acts = activityObj.activations as Record<string, number>;
      for (const [key, value] of Object.entries(acts)) {
        componentAttention.set(key, Math.abs(value));
      }
    } else {
      return null;
    }

    return { componentAttention };
  }

  // ==========================================================================
  // Event Emission
  // ==========================================================================

  private emitEvent(event: HeatmapEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error("Heatmap event listener error:", error);
      }
    }
  }

  /**
   * Subscribe to heatmap events
   */
  subscribe(listener: (event: HeatmapEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

// ============================================================================
// Exports
// ============================================================================

export const ConsciousnessHeatmaps = ConsciousnessHeatmapGenerator;
export default ConsciousnessHeatmapGenerator;
