/**
 * @fileoverview Web Workers module exports
 * @module @neurectomy/3d-engine/workers
 */

export {
  PhysicsController,
  createPhysicsController,
  DEFAULT_PHYSICS_CONFIG,
  type PhysicsConfig,
  type PhysicsStepResult,
  type PhysicsState,
  type PhysicsControllerEvents,
} from "./physics-controller";
