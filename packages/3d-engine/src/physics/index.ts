/**
 * @neurectomy/3d-engine Physics Module
 *
 * High-performance physics simulation powered by Rapier3D WASM.
 * Supports rigid body dynamics, collision detection, and constraint systems.
 */

export { RapierWorld, type RapierWorldConfig } from "./rapier-world";
export { RigidBodyManager, type RigidBodyConfig } from "./rigid-body";
export {
  ColliderManager,
  type ColliderConfig,
  type ColliderShape,
} from "./collider";
export {
  JointManager,
  type JointConfig,
  type RevoluteJointConfig,
  type FixedJointConfig,
  type PrismaticJointConfig,
  type SphericalJointConfig,
} from "./joints";
export {
  RaycastManager,
  type RaycastResult,
  type RaycastOptions,
} from "./raycast";

// Re-export Rapier types for convenience
export type {
  Vector3,
  Quaternion,
  RigidBodyType,
  ColliderDesc,
  RigidBodyDesc,
} from "./types";
