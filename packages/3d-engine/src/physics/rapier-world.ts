/**
 * Rapier3D World Manager
 *
 * Main physics world controller handling initialization, stepping,
 * and event management for the Rapier physics engine.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import { PhysicsEventType } from "./types";
import type {
  Vector3,
  PhysicsWorldConfig,
  CollisionEvent,
  ContactForceEvent,
  SimulationStepConfig,
} from "./types";

export interface RapierWorldConfig extends PhysicsWorldConfig {
  /** Auto-initialize on creation */
  autoInit?: boolean;
  /** Event callback for collisions */
  onCollision?: (event: CollisionEvent) => void;
  /** Event callback for contact forces */
  onContactForce?: (event: ContactForceEvent) => void;
}

/**
 * Physics world wrapper for Rapier3D.
 */
export class RapierWorld {
  private world: RAPIER.World | null = null;
  private eventQueue: RAPIER.EventQueue | null = null;
  private initialized = false;
  private config: RapierWorldConfig;
  private simulationAccumulator = 0;
  private readonly fixedTimeStep = 1 / 60; // 60 Hz physics

  // Event handlers
  private collisionHandlers: Set<(event: CollisionEvent) => void> = new Set();
  private contactForceHandlers: Set<(event: ContactForceEvent) => void> =
    new Set();

  constructor(config: RapierWorldConfig) {
    this.config = {
      autoInit: true,
      ...config,
    };

    if (config.onCollision) {
      this.collisionHandlers.add(config.onCollision);
    }
    if (config.onContactForce) {
      this.contactForceHandlers.add(config.onContactForce);
    }
  }

  /**
   * Initialize the Rapier WASM module.
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    await RAPIER.init();

    const gravity = new RAPIER.Vector3(
      this.config.gravity.x,
      this.config.gravity.y,
      this.config.gravity.z
    );

    this.world = new RAPIER.World(gravity);
    this.eventQueue = new RAPIER.EventQueue(true);

    // Configure integration parameters if provided
    if (this.config.integrationParameters) {
      const params = this.world.integrationParameters;
      Object.assign(params, this.config.integrationParameters);
    }

    this.initialized = true;
  }

  /**
   * Check if world is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get the underlying Rapier world.
   */
  getRawWorld(): RAPIER.World {
    return this.assertInitialized();
  }

  /**
   * Step the physics simulation.
   */
  step(deltaTime?: number): void {
    const world = this.assertInitialized();

    const dt = deltaTime ?? this.fixedTimeStep;

    // Use fixed timestep with accumulator for deterministic physics
    this.simulationAccumulator += dt;

    while (this.simulationAccumulator >= this.fixedTimeStep) {
      world.step(this.eventQueue!);
      this.processEvents();
      this.simulationAccumulator -= this.fixedTimeStep;
    }
  }

  /**
   * Step with custom configuration.
   */
  stepWithConfig(config: SimulationStepConfig): void {
    const world = this.assertInitialized();

    const params = world.integrationParameters;
    const originalDt = params.dt;
    const originalVelIters = params.numSolverIterations;

    params.dt = config.dt;
    params.numSolverIterations = config.velocityIterations;

    world.step(this.eventQueue!);
    this.processEvents();

    params.dt = originalDt;
    params.numSolverIterations = originalVelIters;
  }

  /**
   * Process collision and contact force events.
   */
  private processEvents(): void {
    if (!this.eventQueue) return;

    // Process collision events
    this.eventQueue.drainCollisionEvents((handle1, handle2, started) => {
      const event: CollisionEvent = {
        type: started
          ? PhysicsEventType.CollisionStart
          : PhysicsEventType.CollisionEnd,
        collider1Handle: handle1,
        collider2Handle: handle2,
        rigidBody1Handle: this.getColliderParentHandle(handle1),
        rigidBody2Handle: this.getColliderParentHandle(handle2),
      };

      this.collisionHandlers.forEach((handler) => handler(event));
    });

    // Process contact force events
    this.eventQueue.drainContactForceEvents((event) => {
      const contactEvent: ContactForceEvent = {
        type: PhysicsEventType.ContactForce,
        collider1Handle: event.collider1(),
        collider2Handle: event.collider2(),
        totalForce: {
          x: event.totalForce().x,
          y: event.totalForce().y,
          z: event.totalForce().z,
        },
        totalForceMagnitude: event.totalForceMagnitude(),
        maxForceDirection: {
          x: event.maxForceDirection().x,
          y: event.maxForceDirection().y,
          z: event.maxForceDirection().z,
        },
        maxForceMagnitude: event.maxForceMagnitude(),
      };

      this.contactForceHandlers.forEach((handler) => handler(contactEvent));
    });
  }

  /**
   * Get the parent rigid body handle for a collider.
   */
  private getColliderParentHandle(colliderHandle: number): number | null {
    const collider = this.world!.getCollider(colliderHandle);
    if (!collider) return null;
    const parent = collider.parent();
    return parent?.handle ?? null;
  }

  /**
   * Add collision event handler.
   */
  onCollision(handler: (event: CollisionEvent) => void): () => void {
    this.collisionHandlers.add(handler);
    return () => this.collisionHandlers.delete(handler);
  }

  /**
   * Add contact force event handler.
   */
  onContactForce(handler: (event: ContactForceEvent) => void): () => void {
    this.contactForceHandlers.add(handler);
    return () => this.contactForceHandlers.delete(handler);
  }

  /**
   * Set world gravity.
   */
  setGravity(gravity: Vector3): void {
    const world = this.assertInitialized();
    world.gravity = new RAPIER.Vector3(gravity.x, gravity.y, gravity.z);
  }

  /**
   * Get current gravity.
   */
  getGravity(): Vector3 {
    const world = this.assertInitialized();
    const g = world.gravity;
    return { x: g.x, y: g.y, z: g.z };
  }

  /**
   * Create a rigid body in the world.
   */
  createRigidBody(desc: RAPIER.RigidBodyDesc): RAPIER.RigidBody {
    const world = this.assertInitialized();
    return world.createRigidBody(desc);
  }

  /**
   * Create a collider and attach to a rigid body.
   */
  createCollider(
    desc: RAPIER.ColliderDesc,
    parent?: RAPIER.RigidBody
  ): RAPIER.Collider {
    const world = this.assertInitialized();
    return world.createCollider(desc, parent);
  }

  /**
   * Remove a rigid body from the world.
   */
  removeRigidBody(body: RAPIER.RigidBody): void {
    const world = this.assertInitialized();
    world.removeRigidBody(body);
  }

  /**
   * Remove a collider from the world.
   */
  removeCollider(
    collider: RAPIER.Collider,
    wakeUpParent: boolean = true
  ): void {
    const world = this.assertInitialized();
    world.removeCollider(collider, wakeUpParent);
  }

  /**
   * Create an impulse joint.
   */
  createImpulseJoint(
    params: RAPIER.JointData,
    body1: RAPIER.RigidBody,
    body2: RAPIER.RigidBody,
    wakeUp: boolean = true
  ): RAPIER.ImpulseJoint {
    const world = this.assertInitialized();
    return world.createImpulseJoint(params, body1, body2, wakeUp);
  }

  /**
   * Remove an impulse joint.
   */
  removeImpulseJoint(joint: RAPIER.ImpulseJoint, wakeUp: boolean = true): void {
    const world = this.assertInitialized();
    world.removeImpulseJoint(joint, wakeUp);
  }

  /**
   * Get rigid body by handle.
   */
  getRigidBody(handle: number): RAPIER.RigidBody | null {
    const world = this.assertInitialized();
    return world.getRigidBody(handle) ?? null;
  }

  /**
   * Get collider by handle.
   */
  getCollider(handle: number): RAPIER.Collider | null {
    const world = this.assertInitialized();
    return world.getCollider(handle) ?? null;
  }

  /**
   * Get total number of rigid bodies.
   */
  get numRigidBodies(): number {
    return this.world?.bodies.len() ?? 0;
  }

  /**
   * Get total number of colliders.
   */
  get numColliders(): number {
    return this.world?.colliders.len() ?? 0;
  }

  /**
   * Get all rigid bodies.
   */
  forEachRigidBody(callback: (body: RAPIER.RigidBody) => void): void {
    const world = this.assertInitialized();
    world.bodies.forEach(callback);
  }

  /**
   * Get all colliders.
   */
  forEachCollider(callback: (collider: RAPIER.Collider) => void): void {
    const world = this.assertInitialized();
    world.colliders.forEach(callback);
  }

  /**
   * Reset the physics world.
   */
  reset(): void {
    if (this.world) {
      // Remove all bodies
      const bodiesToRemove: RAPIER.RigidBody[] = [];
      this.world.bodies.forEach((body) => bodiesToRemove.push(body));
      bodiesToRemove.forEach((body) => this.world!.removeRigidBody(body));
    }
    this.simulationAccumulator = 0;
  }

  /**
   * Dispose of the physics world.
   */
  dispose(): void {
    if (this.eventQueue) {
      this.eventQueue.free();
      this.eventQueue = null;
    }
    if (this.world) {
      this.world.free();
      this.world = null;
    }
    this.initialized = false;
    this.collisionHandlers.clear();
    this.contactForceHandlers.clear();
  }

  /**
   * Assert that the world is initialized.
   * @returns The initialized world instance
   */
  private assertInitialized(): RAPIER.World {
    if (!this.initialized || !this.world) {
      throw new Error("Physics world not initialized. Call init() first.");
    }
    return this.world;
  }
}

/**
 * Create a new physics world with default Earth gravity.
 */
export function createDefaultWorld(): RapierWorld {
  return new RapierWorld({
    gravity: { x: 0, y: -9.81, z: 0 },
  });
}

/**
 * Create a zero-gravity physics world.
 */
export function createZeroGravityWorld(): RapierWorld {
  return new RapierWorld({
    gravity: { x: 0, y: 0, z: 0 },
  });
}
