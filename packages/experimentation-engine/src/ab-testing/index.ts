/**
 * NEURECTOMY A/B Testing Module
 * @module @neurectomy/experimentation-engine/ab-testing
 * @agent @PRISM @FLUX
 *
 * Complete A/B testing framework with experiment management,
 * statistical analysis, and intelligent assignment strategies.
 */

// Engine exports
export {
  ABTestingEngine,
  type ABExperimentConfig,
  type VariantConfig,
  type MetricConfig,
  type TargetingConfig,
  type TargetingRule,
  type ScheduleConfig,
  type ExperimentSettings,
  type ABExperiment,
  type VariantState,
  type MetricState,
  type ExperimentResult,
  type ABTestingEvents,
} from "./engine";

// Statistics exports
export {
  proportionZTest,
  twoSampleTTest,
  chiSquaredTest,
  mannWhitneyUTest,
  bayesianABTest,
  calculateSampleSize,
  calculatePower,
  sequentialTest,
  calculateEffectSize,
  calculateConfidenceInterval,
  multipleTestingCorrection,
  type StatisticalTestResult,
  type BayesianResult,
  type SampleSizeResult,
  type SequentialTestResult,
  type MultipleTesting,
} from "./statistics";

// Assignment exports
export {
  AssignmentManager,
  RandomAssignment,
  DeterministicAssignment,
  WeightedAssignment,
  EpsilonGreedyAssignment,
  ThompsonSamplingAssignment,
  UCB1Assignment,
  ContextualAssignment,
  type AssignmentStrategy,
  type AssignmentStrategyType,
  type VariantWeight,
  type AssignmentContext,
  type BanditState,
  type AssignmentManagerEvents,
} from "./assignment";

// =============================================================================
// Backwards compatibility aliases
// =============================================================================

/** @deprecated Use ABTestingEngine instead */
export { ABTestingEngine as ABTestManager } from "./engine";

/** @deprecated Use ABExperimentConfig instead */
export type { ABExperimentConfig as ABTestConfig } from "./engine";
