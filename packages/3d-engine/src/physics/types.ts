/**
 * Type definitions for Rapier3D physics integration.
 */

import type RAPIER from "@dimforge/rapier3d-compat";

// Re-export core Rapier types
export type World = RAPIER.World;
export type RigidBody = RAPIER.RigidBody;
export type Collider = RAPIER.Collider;
export type ColliderDesc = RAPIER.ColliderDesc;
export type RigidBodyDesc = RAPIER.RigidBodyDesc;
export type ImpulseJoint = RAPIER.ImpulseJoint;
export type MultibodyJoint = RAPIER.MultibodyJoint;
export type Ray = RAPIER.Ray;
export type QueryFilterFlags = RAPIER.QueryFilterFlags;

/**
 * 3D vector type compatible with Rapier.
 */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/**
 * Quaternion rotation type.
 */
export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

/**
 * Rigid body types matching Rapier.
 */
export enum RigidBodyType {
  /** Does not move, infinite mass */
  Fixed = "fixed",
  /** Fully dynamic, affected by forces */
  Dynamic = "dynamic",
  /** Moves only via velocity, not affected by forces */
  KinematicPositionBased = "kinematicPositionBased",
  /** Moves via direct position updates */
  KinematicVelocityBased = "kinematicVelocityBased",
}

/**
 * Collider shape types.
 */
export enum ColliderShapeType {
  Ball = "ball",
  Capsule = "capsule",
  Cuboid = "cuboid",
  Cylinder = "cylinder",
  Cone = "cone",
  Convex = "convex",
  Trimesh = "trimesh",
  Heightfield = "heightfield",
}

/**
 * Physics event types.
 */
export enum PhysicsEventType {
  CollisionStart = "collisionStart",
  CollisionEnd = "collisionEnd",
  ContactForce = "contactForce",
  Sleep = "sleep",
  Wake = "wake",
}

/**
 * Collision event data.
 */
export interface CollisionEvent {
  type: PhysicsEventType.CollisionStart | PhysicsEventType.CollisionEnd;
  collider1Handle: number;
  collider2Handle: number;
  rigidBody1Handle: number | null;
  rigidBody2Handle: number | null;
}

/**
 * Contact force event data.
 */
export interface ContactForceEvent {
  type: PhysicsEventType.ContactForce;
  collider1Handle: number;
  collider2Handle: number;
  totalForce: Vector3;
  totalForceMagnitude: number;
  maxForceDirection: Vector3;
  maxForceMagnitude: number;
}

/**
 * Physics body handle for external tracking.
 */
export interface PhysicsHandle {
  bodyHandle: number;
  colliderHandles: number[];
  userData?: unknown;
}

/**
 * Joint type enumeration.
 */
export enum JointType {
  Fixed = "fixed",
  Revolute = "revolute",
  Prismatic = "prismatic",
  Spherical = "spherical",
  Rope = "rope",
  Spring = "spring",
  Generic = "generic",
}

/**
 * Physics simulation step configuration.
 */
export interface SimulationStepConfig {
  /** Time step in seconds */
  dt: number;
  /** Velocity iteration count */
  velocityIterations: number;
  /** Stabilization iteration count */
  stabilizationIterations: number;
}

/**
 * Physics world configuration.
 */
export interface PhysicsWorldConfig {
  /** Gravity vector */
  gravity: Vector3;
  /** Integration parameters */
  integrationParameters?: Partial<IntegrationParameters>;
  /** Enable CCD (Continuous Collision Detection) */
  ccdEnabled?: boolean;
  /** Maximum CCD substeps */
  maxCcdSubsteps?: number;
}

/**
 * Integration parameters for physics simulation.
 */
export interface IntegrationParameters {
  /** Time step duration */
  dt: number;
  /** Minimum CCD step size */
  minCcdDt: number;
  /** Error reduction parameter */
  erp: number;
  /** Damping ratio */
  dampingRatio: number;
  /** Joint error reduction parameter */
  jointErp: number;
  /** Joint damping ratio */
  jointDampingRatio: number;
  /** Allowed linear error */
  allowedLinearError: number;
  /** Maximum penetration correction */
  maxPenetrationCorrection: number;
  /** Prediction distance */
  predictionDistance: number;
  /** Number of solver iterations */
  numSolverIterations: number;
  /** Number of additional friction iterations */
  numAdditionalFrictionIterations: number;
  /** Number of internal PGS iterations */
  numInternalPgsIterations: number;
}
