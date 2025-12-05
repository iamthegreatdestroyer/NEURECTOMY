/**
 * NEURECTOMY Continuous Intelligence - Core Types
 *
 * @packageDocumentation
 */

// ==============================================================================
// Meta Learning Types
// ==============================================================================

/**
 * Learning observation for self-improvement
 */
export interface LearningObservation {
  /** Unique observation ID */
  id: string;
  /** Timestamp */
  timestamp: Date;
  /** Component that generated observation */
  component: string;
  /** Action type */
  action: string;
  /** Input context */
  context: Record<string, unknown>;
  /** Action result */
  result: ActionResult;
  /** Performance metrics */
  metrics: PerformanceMetrics;
  /** User feedback (if any) */
  feedback?: UserFeedback;
}

/**
 * Action result from an operation
 */
export interface ActionResult {
  /** Success status */
  success: boolean;
  /** Duration in milliseconds */
  duration: number;
  /** Output data */
  output?: unknown;
  /** Error if failed */
  error?: string;
  /** Side effects */
  sideEffects?: string[];
}

/**
 * Performance metrics for observation
 */
export interface PerformanceMetrics {
  /** Latency in ms */
  latency: number;
  /** Throughput (requests/second) */
  throughput?: number;
  /** Memory usage in bytes */
  memoryUsage?: number;
  /** CPU usage percentage */
  cpuUsage?: number;
  /** Custom metrics */
  custom?: Record<string, number>;
}

/**
 * User feedback for learning
 */
export interface UserFeedback {
  /** Feedback type */
  type: "positive" | "negative" | "correction";
  /** Rating (1-5) */
  rating?: number;
  /** Correction data */
  correction?: unknown;
  /** Comment */
  comment?: string;
}

/**
 * Learning model parameters
 */
export interface LearningModel {
  /** Model identifier */
  id: string;
  /** Model type */
  type: ModelType;
  /** Model version */
  version: number;
  /** Parameters */
  parameters: Record<string, number>;
  /** Training metrics */
  trainingMetrics: TrainingMetrics;
  /** Created at */
  createdAt: Date;
  /** Last updated */
  updatedAt: Date;
}

/**
 * Model type for learning
 */
export type ModelType =
  | "decision-tree"
  | "linear-regression"
  | "neural-network"
  | "reinforcement"
  | "bayesian"
  | "ensemble";

/**
 * Training metrics
 */
export interface TrainingMetrics {
  /** Total observations used */
  observations: number;
  /** Training iterations */
  iterations: number;
  /** Accuracy score */
  accuracy: number;
  /** Loss value */
  loss: number;
  /** Validation accuracy */
  validationAccuracy?: number;
  /** Training duration */
  trainingDuration: number;
}

/**
 * Self-improvement strategy
 */
export interface ImprovementStrategy {
  /** Strategy ID */
  id: string;
  /** Target component */
  targetComponent: string;
  /** Improvement type */
  type: ImprovementType;
  /** Current state */
  currentState: Record<string, unknown>;
  /** Proposed changes */
  proposedChanges: ProposedChange[];
  /** Expected impact */
  expectedImpact: ImpactAssessment;
  /** Confidence score */
  confidence: number;
  /** Risk level */
  riskLevel: RiskLevel;
}

/**
 * Type of improvement
 */
export type ImprovementType =
  | "parameter-tuning"
  | "algorithm-switch"
  | "resource-allocation"
  | "caching-strategy"
  | "query-optimization"
  | "architecture-change";

/**
 * Proposed change
 */
export interface ProposedChange {
  /** Change target */
  target: string;
  /** Current value */
  currentValue: unknown;
  /** Proposed value */
  proposedValue: unknown;
  /** Rationale */
  rationale: string;
}

/**
 * Impact assessment
 */
export interface ImpactAssessment {
  /** Performance improvement % */
  performanceGain: number;
  /** Resource savings % */
  resourceSavings: number;
  /** Reliability improvement */
  reliabilityGain: number;
  /** Implementation effort (hours) */
  implementationEffort: number;
}

/**
 * Risk level
 */
export type RiskLevel = "low" | "medium" | "high" | "critical";

// ==============================================================================
// Predictive Maintenance Types
// ==============================================================================

/**
 * System health snapshot
 */
export interface SystemHealthSnapshot {
  /** Snapshot ID */
  id: string;
  /** Timestamp */
  timestamp: Date;
  /** Component health */
  components: ComponentHealth[];
  /** Resource utilization */
  resources: ResourceUtilization;
  /** Active alerts */
  alerts: HealthAlert[];
  /** Overall health score (0-100) */
  overallScore: number;
}

/**
 * Component health status
 */
export interface ComponentHealth {
  /** Component name */
  name: string;
  /** Status */
  status: HealthStatus;
  /** Health score (0-100) */
  score: number;
  /** Degradation trend */
  trend: TrendDirection;
  /** Predicted failure time (if degrading) */
  predictedFailure?: Date;
  /** Health factors */
  factors: HealthFactorDetail[];
}

/**
 * Health status
 */
export type HealthStatus =
  | "healthy"
  | "degraded"
  | "warning"
  | "critical"
  | "unknown";

/**
 * Trend direction
 */
export type TrendDirection = "improving" | "stable" | "degrading" | "unknown";

/**
 * Health factor detail
 */
export interface HealthFactorDetail {
  /** Factor name */
  name: string;
  /** Current value */
  value: number;
  /** Threshold for warning */
  warningThreshold: number;
  /** Threshold for critical */
  criticalThreshold: number;
  /** Unit */
  unit: string;
}

/**
 * Resource utilization
 */
export interface ResourceUtilization {
  /** CPU utilization % */
  cpu: ResourceMetric;
  /** Memory utilization */
  memory: ResourceMetric;
  /** Disk utilization */
  disk: ResourceMetric;
  /** Network utilization */
  network: ResourceMetric;
  /** Custom resources */
  custom?: Record<string, ResourceMetric>;
}

/**
 * Resource metric
 */
export interface ResourceMetric {
  /** Current value */
  current: number;
  /** Average over window */
  average: number;
  /** Peak value */
  peak: number;
  /** Predicted value in 1 hour */
  predicted1h?: number;
  /** Predicted value in 24 hours */
  predicted24h?: number;
  /** Time to exhaustion */
  timeToExhaustion?: Date;
}

/**
 * Health alert
 */
export interface HealthAlert {
  /** Alert ID */
  id: string;
  /** Severity */
  severity: AlertSeverity;
  /** Component affected */
  component: string;
  /** Alert message */
  message: string;
  /** Recommended action */
  recommendation: string;
  /** Created at */
  createdAt: Date;
  /** Acknowledged at */
  acknowledgedAt?: Date;
  /** Resolved at */
  resolvedAt?: Date;
}

/**
 * Alert severity
 */
export type AlertSeverity = "info" | "warning" | "error" | "critical";

/**
 * Failure prediction
 */
export interface FailurePrediction {
  /** Prediction ID */
  id: string;
  /** Component */
  component: string;
  /** Failure type */
  failureType: FailureType;
  /** Probability (0-1) */
  probability: number;
  /** Predicted time */
  predictedTime: Date;
  /** Confidence interval */
  confidenceInterval: {
    lower: Date;
    upper: Date;
  };
  /** Contributing factors */
  factors: FailureFactor[];
  /** Preventive actions */
  preventiveActions: PreventiveAction[];
}

/**
 * Failure type
 */
export type FailureType =
  | "out-of-memory"
  | "cpu-exhaustion"
  | "disk-full"
  | "connection-pool-exhaustion"
  | "deadlock"
  | "cascading-failure"
  | "timeout"
  | "data-corruption";

/**
 * Failure contributing factor
 */
export interface FailureFactor {
  /** Factor name */
  name: string;
  /** Contribution weight (0-1) */
  weight: number;
  /** Current value */
  currentValue: number;
  /** Trend */
  trend: TrendDirection;
}

/**
 * Preventive action
 */
export interface PreventiveAction {
  /** Action ID */
  id: string;
  /** Action type */
  type: PreventiveActionType;
  /** Description */
  description: string;
  /** Effectiveness (0-1) */
  effectiveness: number;
  /** Implementation cost (low/medium/high) */
  cost: "low" | "medium" | "high";
  /** Automated execution possible */
  automatable: boolean;
}

/**
 * Preventive action type
 */
export type PreventiveActionType =
  | "scale-up"
  | "scale-out"
  | "restart"
  | "garbage-collect"
  | "cache-clear"
  | "connection-reset"
  | "failover"
  | "backup";

// ==============================================================================
// Auto-Optimization Types
// ==============================================================================

/**
 * Optimization target
 */
export interface OptimizationTarget {
  /** Target ID */
  id: string;
  /** Target name */
  name: string;
  /** Optimization objective */
  objective: OptimizationObjective;
  /** Current value */
  currentValue: number;
  /** Target value */
  targetValue: number;
  /** Constraints */
  constraints: OptimizationConstraint[];
  /** Weight in multi-objective */
  weight: number;
}

/**
 * Optimization objective
 */
export type OptimizationObjective =
  | "minimize-latency"
  | "maximize-throughput"
  | "minimize-cost"
  | "maximize-availability"
  | "minimize-error-rate"
  | "balance";

/**
 * Optimization constraint
 */
export interface OptimizationConstraint {
  /** Constraint name */
  name: string;
  /** Constraint type */
  type: "min" | "max" | "range" | "equality";
  /** Constraint value(s) */
  value: number | [number, number];
  /** Hard constraint (must satisfy) vs soft (prefer) */
  hard: boolean;
}

/**
 * Tunable parameter
 */
export interface TunableParameter {
  /** Parameter ID */
  id: string;
  /** Parameter name */
  name: string;
  /** Component */
  component: string;
  /** Current value */
  currentValue: number;
  /** Value type */
  valueType: "continuous" | "discrete" | "categorical";
  /** Valid range */
  range: {
    min: number;
    max: number;
    step?: number;
  };
  /** Sensitivity to optimization */
  sensitivity: number;
  /** Impact on objectives */
  impact: Record<string, number>;
}

/**
 * Optimization run result
 */
export interface OptimizationResult {
  /** Run ID */
  id: string;
  /** Started at */
  startedAt: Date;
  /** Completed at */
  completedAt: Date;
  /** Status */
  status: OptimizationStatus;
  /** Initial state */
  initialState: Record<string, number>;
  /** Optimized state */
  optimizedState: Record<string, number>;
  /** Improvement metrics */
  improvements: OptimizationImprovement[];
  /** Exploration history */
  explorationHistory: ExplorationStep[];
  /** Applied changes */
  appliedChanges: AppliedChange[];
}

/**
 * Optimization status
 */
export type OptimizationStatus =
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "converged";

/**
 * Optimization improvement
 */
export interface OptimizationImprovement {
  /** Target name */
  target: string;
  /** Before value */
  before: number;
  /** After value */
  after: number;
  /** Improvement percentage */
  improvement: number;
}

/**
 * Exploration step in optimization
 */
export interface ExplorationStep {
  /** Step number */
  step: number;
  /** Parameters tried */
  parameters: Record<string, number>;
  /** Objective value */
  objectiveValue: number;
  /** Is new best */
  isNewBest: boolean;
}

/**
 * Applied parameter change
 */
export interface AppliedChange {
  /** Parameter name */
  parameter: string;
  /** Old value */
  oldValue: number;
  /** New value */
  newValue: number;
  /** Applied at */
  appliedAt: Date;
  /** Rollback possible */
  rollbackPossible: boolean;
}

/**
 * Query optimization suggestion
 */
export interface QueryOptimization {
  /** Optimization ID */
  id: string;
  /** Original query */
  originalQuery: string;
  /** Optimized query */
  optimizedQuery: string;
  /** Optimization type */
  type: QueryOptimizationType;
  /** Expected improvement */
  expectedImprovement: number;
  /** Explanation */
  explanation: string;
  /** Risk level */
  risk: RiskLevel;
}

/**
 * Query optimization type
 */
export type QueryOptimizationType =
  | "index-suggestion"
  | "query-rewrite"
  | "join-optimization"
  | "predicate-pushdown"
  | "caching-opportunity"
  | "batch-optimization";

// ==============================================================================
// Configuration Types
// ==============================================================================

/**
 * Self-improvement configuration
 */
export interface SelfImprovementConfig {
  /** Enable auto-learning */
  enabled: boolean;
  /** Learning rate */
  learningRate: number;
  /** Observation retention days */
  retentionDays: number;
  /** Minimum observations before learning */
  minObservations: number;
  /** Auto-apply improvements */
  autoApply: boolean;
  /** Max auto-apply risk level */
  maxAutoApplyRisk: RiskLevel;
}

/**
 * Predictive maintenance configuration
 */
export interface PredictiveConfig {
  /** Enable predictions */
  enabled: boolean;
  /** Prediction window (hours) */
  predictionWindow: number;
  /** Alert thresholds */
  alertThresholds: {
    warning: number;
    critical: number;
  };
  /** Enable auto-remediation */
  autoRemediate: boolean;
  /** Max auto-remediate severity */
  maxAutoRemediateSeverity: AlertSeverity;
}

/**
 * Auto-optimization configuration
 */
export interface OptimizationConfig {
  /** Enable optimization */
  enabled: boolean;
  /** Optimization algorithm */
  algorithm: OptimizationAlgorithm;
  /** Max iterations */
  maxIterations: number;
  /** Convergence threshold */
  convergenceThreshold: number;
  /** Exploration vs exploitation ratio */
  explorationRatio: number;
  /** Auto-apply optimizations */
  autoApply: boolean;
}

/**
 * Optimization algorithm
 */
export type OptimizationAlgorithm =
  | "bayesian"
  | "genetic"
  | "gradient-descent"
  | "grid-search"
  | "random-search"
  | "simulated-annealing";

// ==============================================================================
// Event Types
// ==============================================================================

/**
 * Continuous intelligence events
 */
export interface ContinuousIntelligenceEvents {
  /** Learning observation recorded */
  "learning:observation": { observation: LearningObservation };
  /** Model updated */
  "learning:model-updated": { model: LearningModel };
  /** Improvement suggested */
  "learning:improvement-suggested": { strategy: ImprovementStrategy };
  /** Health snapshot taken */
  "health:snapshot": { snapshot: SystemHealthSnapshot };
  /** Failure predicted */
  "health:failure-predicted": { prediction: FailurePrediction };
  /** Alert triggered */
  "health:alert": { alert: HealthAlert };
  /** Optimization started */
  "optimization:started": { targets: OptimizationTarget[] };
  /** Optimization step */
  "optimization:step": { step: ExplorationStep };
  /** Optimization completed */
  "optimization:completed": { result: OptimizationResult };
  /** Error occurred */
  error: { error: Error; context: string };
}
