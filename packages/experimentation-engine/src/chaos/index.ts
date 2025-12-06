/**
 * NEURECTOMY Chaos Engineering Module Exports
 * @module @neurectomy/experimentation-engine/chaos
 */

// Core simulator
export {
  ChaosSimulator,
  type FaultInjector,
  type ChaosExperiment,
  type ChaosExperimentConfig,
  type AffectedTarget,
  type FaultConfig,
  type FaultType,
  type FaultSeverity,
  type ExperimentState,
  type SafetyConfig,
  type ChaosStorage,
  type ChaosNotifier,
  type MetricsProvider,
  type ExperimentResults,
  type ActiveFault,
  type HealthStatus,
  type HealthCheckResult,
  type MetricSnapshot,
  type TimelineEvent,
  type Finding,
  type Approval,
  type ChaosSimulatorEvents,
  type ChaosNotification,
  // Schemas
  FaultTypeSchema,
  FaultSeveritySchema,
  ExperimentStateSchema,
  FaultConfigSchema,
  TargetSelectorSchema,
  BlastRadiusConfigSchema,
  SafetyConfigSchema,
  HealthCheckConfigSchema,
  ChaosExperimentConfigSchema,
} from "./simulator";

// Extended fault types
export {
  // Injectors
  NetworkPartitionInjector,
  PacketLossInjector,
  BandwidthLimitInjector,
  DNSFailureInjector,
  CPUStressInjector,
  MemoryStressInjector,
  DiskStressInjector,
  ProcessKillInjector,
  ContainerActionInjector,
  NodeDrainInjector,
  // Registry
  FaultRegistry,
  // Scenarios
  ChaosScenarios,
  // Schemas
  NetworkPartitionConfigSchema,
  PacketLossConfigSchema,
  BandwidthLimitConfigSchema,
  DNSFailureConfigSchema,
  CPUStressConfigSchema,
  MemoryStressConfigSchema,
  DiskStressConfigSchema,
  IOStressConfigSchema,
  ProcessKillConfigSchema,
  ContainerActionConfigSchema,
  NodeDrainConfigSchema,
  // Types
  type NetworkPartitionConfig,
  type PacketLossConfig,
  type BandwidthLimitConfig,
  type DNSFailureConfig,
  type CPUStressConfig,
  type MemoryStressConfig,
  type DiskStressConfig,
  type IOStressConfig,
  type ProcessKillConfig,
  type ContainerActionConfig,
  type NodeDrainConfig,
} from "./faults";

// Scheduler
export {
  ChaosScheduler,
  // Schemas
  CronExpressionSchema,
  ScheduleWindowSchema,
  ExperimentScheduleSchema,
  GamedayConfigSchema,
  // Types
  type ExperimentSchedule,
  type GamedayConfig,
  type ScheduleWindow,
  type SchedulerEvents,
  type SchedulerConfig,
  type ScheduledExecution,
  type GamedayResults,
  type SchedulerStatistics,
} from "./scheduler";
