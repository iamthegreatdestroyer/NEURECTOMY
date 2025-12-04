/**
 * CAD Visualization System Index
 * 
 * Technical visualization components for agent architecture rendering.
 * Implements blueprint modes, technical drawing overlays, and precision interactions.
 * 
 * @module @neurectomy/3d-engine/cad
 * @agents @CANVAS @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 * @step Step 3 - CAD Visualization System
 */

// Core CAD rendering
export * from './agent-renderer';
export * from './interaction-system';

// Re-export visualization types for convenience
export type {
  AgentComponent,
  AgentConnection,
  BlueprintConfig,
  ViewportConfig,
} from '../visualization/types';

