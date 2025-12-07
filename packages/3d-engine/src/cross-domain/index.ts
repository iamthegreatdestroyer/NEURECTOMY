/**
 * Cross-Domain Integration Module
 *
 * Unified type system and event bridge for Forge × Twin × Foundry synergies.
 * This module provides the foundation for all breakthrough innovations.
 *
 * @module @neurectomy/3d-engine/cross-domain
 * @agents @NEXUS @ARCHITECT @NEURAL
 * @phase Cross-Domain Innovation Integration
 *
 * ## Architecture
 *
 * ```
 * ┌──────────────────────────────────────────────────────────────────┐
 * │                     CrossDomainOrchestrator                       │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
 * │  │   Forge     │  │    Twin     │  │   Foundry   │              │
 * │  │  (3D/4D)    │  │ (Simulation)│  │    (ML)     │              │
 * │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
 * │         │                │                │                      │
 * │         └────────────────┼────────────────┘                      │
 * │                          │                                       │
 * │  ┌───────────────────────┴───────────────────────┐              │
 * │  │              CrossDomainEventBridge            │              │
 * │  │  (Pub/Sub, Routing, Transforms)               │              │
 * │  └───────────────────────────────────────────────┘              │
 * │                          │                                       │
 * │  ┌───────────────────────┴───────────────────────┐              │
 * │  │                    Adapters                     │              │
 * │  │  ForgeAdapter | TwinAdapter | FoundryAdapter   │              │
 * │  └───────────────────────────────────────────────┘              │
 * │                          │                                       │
 * │  ┌───────────────────────┴───────────────────────┐              │
 * │  │                 Isomorphisms                    │              │
 * │  │  Pattern Recognition | Type Mappings           │              │
 * │  └───────────────────────────────────────────────┘              │
 * └──────────────────────────────────────────────────────────────────┘
 * ```
 *
 * ## Usage Example
 *
 * ```typescript
 * import { crossDomain } from '@neurectomy/3d-engine/cross-domain';
 *
 * // Register an entity from Forge domain
 * const entity = crossDomain.registerEntity(agentComponent, 'forge');
 *
 * // Build a graph from entities
 * const graph = crossDomain.buildGraph([entity.id], 'my-graph');
 *
 * // Create timeline for state tracking
 * const timeline = crossDomain.createTimeline('timeline-1', [entity.id]);
 *
 * // Start training session
 * const session = crossDomain.startTraining({
 *   modelId: 'model-1',
 *   config: { epochs: 100, lr: 0.001 },
 * });
 *
 * // Get statistics
 * console.log(crossDomain.getStatistics());
 * ```
 */

// Types - Unified type definitions for cross-domain operations
export * from "./types";

// Event Bridge - Pub/sub event system with routing
export * from "./event-bridge";

// Adapters - Bidirectional type transformations
export * from "./adapters";

// Isomorphisms - Mathematical mappings between domains
export * from "./isomorphisms";

// Orchestrator - Central coordinator for cross-domain operations
export {
  CrossDomainOrchestrator,
  getOrchestrator,
  crossDomain,
  type OrchestratorConfig,
} from "./orchestrator";

// ============================================================================
// BREAKTHROUGH INNOVATIONS
// ============================================================================

// Forge × Twin Innovations
export {
  TemporalTwinReplayTheater,
  type ReplayTheaterConfig,
  type ReplaySession,
  type KeyframeMarker,
  type PlaybackState,
} from "./innovations/replay-theater";

export {
  PredictiveCascade,
  type PredictiveCascadeConfig,
  type PredictionOverlay,
  type PredictionBranch,
  type CascadeVisualization,
} from "./innovations/predictive-cascade";

export {
  ConsciousnessHeatmaps,
  type ConsciousnessHeatmapConfig,
  type CognitiveActivity,
  type HeatmapLayer,
  type AttentionPattern,
} from "./innovations/consciousness-heatmaps";

// Re-export all innovations from namespace
export * as innovations from "./innovations";

// Re-export key types for convenience
export type {
  UnifiedEntity,
  UnifiedGraph,
  UnifiedTimeline,
  UnifiedEvent,
  Domain,
  UniversalId,
  EventType,
  TrainingSession,
  TrainingMetrics,
} from "./types";

export type { Isomorphism, CrossDomainPattern } from "./isomorphisms";
