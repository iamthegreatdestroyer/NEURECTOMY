/**
 * Digital Twin System
 *
 * Complete digital twin infrastructure for creating, managing,
 * and predicting agent behavior through virtual replicas.
 *
 * @module @neurectomy/3d-engine/digital-twin
 * @agents @ARCHITECT @NEURAL @SYNAPSE
 * @phase Phase 3 - Dimensional Forge
 */

// ============================================================================
// Type Exports
// ============================================================================

export type {
  // Core Types
  TwinId,
  AgentId,
  TwinMode,
  SyncState,
  TwinFidelity,

  // State Types
  TwinState,
  TwinMetadata,
  AgentStateSnapshot,
  ComponentGraphSnapshot,
  IOHistoryEntry,
  AgentMetrics,
  TimeSeriesMetric,
  ResourceUtilization,

  // Configuration Types
  TwinManagerConfig,
  SyncConfig,
  SyncMode,
  ConflictResolution,
  PredictionConfig,
  ScenarioInput,

  // Result Types
  TwinSyncResult,
  SyncConflict,
  PredictionResult,
  PredictionTimeline,
  TimelinePoint,
  Scenario,
  PredictionMetrics,

  // Diff Types
  TwinDelta,
  TwinDiff,
  TwinDiffEntry,

  // Query Types
  TwinQuery,
  TwinStatistics,

  // Event Types
  TwinEvent,
  TwinEventListener,

  // Error Types
  TwinErrorCode,
  TwinError,
} from "./types";

// ============================================================================
// Manager Exports
// ============================================================================

export { TwinManager, getTwinManager, resetTwinManager } from "./twin-manager";

// ============================================================================
// Sync Engine Exports
// ============================================================================

export {
  TwinSyncEngine,
  getTwinSyncEngine,
  resetTwinSyncEngine,
  type SyncSession,
  type SyncMetrics,
  type SyncMessage,
  type SyncEventHandler,
} from "./twin-sync";

// ============================================================================
// Predictive Engine Exports
// ============================================================================

export {
  TwinPredictiveEngine,
  getTwinPredictiveEngine,
  resetTwinPredictiveEngine,
  type PredictionEngine,
  type PredictionContext,
  type AnalysisResult,
  type TrendAnalysis,
  type AnomalyDetection,
  type CorrelationMatrix,
  type SeasonalPattern,
} from "./predictive-engine";

// ============================================================================
// Convenience Functions
// ============================================================================

import { getTwinManager } from "./twin-manager";
import { getTwinSyncEngine } from "./twin-sync";
import { getTwinPredictiveEngine } from "./predictive-engine";
import type {
  TwinId,
  AgentId,
  TwinState,
  TwinManagerConfig,
  PredictionConfig,
} from "./types";

/**
 * Quick-start helper to create a digital twin with default settings
 */
export async function createDigitalTwin(
  agentId: AgentId,
  name?: string,
  config?: Partial<TwinManagerConfig>
): Promise<TwinState> {
  const manager = getTwinManager(config);
  return manager.createTwin(agentId, { name });
}

/**
 * Quick-start helper to set up prediction for a twin
 */
export function enablePrediction(
  twinId: TwinId,
  config?: Partial<PredictionConfig>
): void {
  const engine = getTwinPredictiveEngine();

  const defaultConfig: PredictionConfig = {
    horizonMs: 60000,
    stepMs: 1000,
    scenarioCount: 3,
    inputScenarios: [],
    quantifyUncertainty: true,
    confidenceLevel: 0.95,
    ...config,
  };

  engine.createEngine(twinId, defaultConfig);
}

/**
 * Quick-start helper to sync a twin
 */
export async function syncTwin(twinId: TwinId): Promise<boolean> {
  const manager = getTwinManager();
  const twin = manager.getTwin(twinId);

  if (!twin) {
    return false;
  }

  const syncEngine = getTwinSyncEngine();
  const session = await syncEngine.startSession(twinId, twin.agentId, {
    mode: "periodic",
    intervalMs: 5000,
    conflictResolution: "source-wins",
    compression: true,
    batchSize: 100,
  });

  return session.state === "syncing";
}

/**
 * Cleanup all digital twin resources
 */
export function disposeDigitalTwinSystem(): void {
  const { resetTwinManager } = require("./twin-manager");
  const { resetTwinSyncEngine } = require("./twin-sync");
  const { resetTwinPredictiveEngine } = require("./predictive-engine");

  resetTwinManager();
  resetTwinSyncEngine();
  resetTwinPredictiveEngine();
}
