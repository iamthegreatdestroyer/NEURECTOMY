/**
 * NEURECTOMY Hypothesis Lab Exports
 * @module @neurectomy/experimentation-engine/hypothesis
 */

// Lab
export {
  HypothesisLab,
  createHypothesisLab,
  defineHypothesis,
  defineParameterSpace,
  type HypothesisConfig,
  type Parameter,
  type ParameterSpace,
  type ParameterConstraint,
  type TrialConfig,
  type TrialResult,
  type Hypothesis,
  type Trial,
  type LabConfig,
  type StorageBackend,
  type LabEvents,
} from "./lab";

// Tracker
export {
  ExperimentTracker,
  createTracker,
  withRun,
  type RunConfig,
  type Run,
  type RunStatus,
  type MetricLog,
  type Artifact,
  type ArtifactType,
  type Experiment,
  type TrackerConfig,
  type TrackerEvents,
  type ComparisonResult,
  type MetricComparison,
  type ParameterComparison,
  type RunRanking,
} from "./tracker";

// Versioning
export {
  ModelRegistry,
  createRegistry,
  defineSignature,
  defineArtifact,
  type ModelVersion,
  type ModelStage,
  type ModelArtifact,
  type ModelSignature,
  type SignatureField,
  type ModelLineage,
  type RegisteredModel,
  type VersioningConfig,
  type VersioningEvents,
  type VersionComparison,
  type MetricDiff,
  type ParameterDiff,
  type StructuralChange,
} from "./versioning";

// MLflow Bridge - TypeScript to Python MLflow sync
// @TENSOR @SYNAPSE - Enables bidirectional experiment tracking
export {
  MLflowBridge,
  createMLflowBridge,
  createDefaultBridge,
  withMLflowRun,
  mlflowTracked,
  type MLflowConfig,
  type MLflowExperiment,
  type MLflowRun,
  type MLflowMetric,
  type MLflowParam,
  type MLflowArtifact,
  type MLflowBridgeEvents,
  type SyncResult,
} from "./mlflow-bridge";
