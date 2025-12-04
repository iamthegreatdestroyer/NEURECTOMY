/**
 * Rigid Body Manager
 *
 * Factory and lifecycle management for Rapier rigid bodies.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import type { RapierWorld } from "./rapier-world";
import type {
  Vector3,
  Quaternion,
  RigidBodyType,
  PhysicsHandle,
} from "./types";

export interface RigidBodyConfig {
  /** Body type: fixed, dynamic, or kinematic */
  type: RigidBodyType;
  /** Initial position */
  position?: Vector3;
  /** Initial rotation */
  rotation?: Quaternion;
  /** Linear velocity */
  linearVelocity?: Vector3;
  /** Angular velocity */
  angularVelocity?: Vector3;
  /** Linear damping factor */
  linearDamping?: number;
  /** Angular damping factor */
  angularDamping?: number;
  /** Gravity scale (1.0 = normal, 0 = no gravity) */
  gravityScale?: number;
  /** Enable CCD for fast-moving objects */
  ccdEnabled?: boolean;
  /** User-defined data */
  userData?: unknown;
  /** Can sleep when inactive */
  canSleep?: boolean;
  /** Dominance group for collision resolution */
  dominanceGroup?: number;
  /** Lock translations */
  lockTranslations?: { x: boolean; y: boolean; z: boolean };
  /** Lock rotations */
  lockRotations?: { x: boolean; y: boolean; z: boolean };
}

/**
 * Manages rigid body creation and lifecycle.
 */
export class RigidBodyManager {
  private world: RapierWorld;
  private bodies: Map<number, PhysicsHandle> = new Map();
  private userDataMap: Map<unknown, number> = new Map();

  constructor(world: RapierWorld) {
    this.world = world;
  }

  /**
   * Create a rigid body with the given configuration.
   */
  create(config: RigidBodyConfig): PhysicsHandle {
    const desc = this.createBodyDesc(config);
    const body = this.world.createRigidBody(desc);

    const handle: PhysicsHandle = {
      bodyHandle: body.handle,
      colliderHandles: [],
      userData: config.userData,
    };

    this.bodies.set(body.handle, handle);

    if (config.userData !== undefined) {
      this.userDataMap.set(config.userData, body.handle);
    }

    return handle;
  }

  /**
   * Create a rigid body descriptor.
   */
  private createBodyDesc(config: RigidBodyConfig): RAPIER.RigidBodyDesc {
    let desc: RAPIER.RigidBodyDesc;

    switch (config.type) {
      case "fixed":
        desc = RAPIER.RigidBodyDesc.fixed();
        break;
      case "dynamic":
        desc = RAPIER.RigidBodyDesc.dynamic();
        break;
      case "kinematicPositionBased":
        desc = RAPIER.RigidBodyDesc.kinematicPositionBased();
        break;
      case "kinematicVelocityBased":
        desc = RAPIER.RigidBodyDesc.kinematicVelocityBased();
        break;
      default:
        desc = RAPIER.RigidBodyDesc.dynamic();
    }

    // Position
    if (config.position) {
      desc.setTranslation(
        config.position.x,
        config.position.y,
        config.position.z
      );
    }

    // Rotation
    if (config.rotation) {
      desc.setRotation(config.rotation);
    }

    // Velocities
    if (config.linearVelocity) {
      desc.setLinvel(
        config.linearVelocity.x,
        config.linearVelocity.y,
        config.linearVelocity.z
      );
    }

    if (config.angularVelocity) {
      desc.setAngvel(config.angularVelocity);
    }

    // Damping
    if (config.linearDamping !== undefined) {
      desc.setLinearDamping(config.linearDamping);
    }

    if (config.angularDamping !== undefined) {
      desc.setAngularDamping(config.angularDamping);
    }

    // Gravity scale
    if (config.gravityScale !== undefined) {
      desc.setGravityScale(config.gravityScale);
    }

    // CCD
    if (config.ccdEnabled !== undefined) {
      desc.setCcdEnabled(config.ccdEnabled);
    }

    // Sleep
    if (config.canSleep !== undefined) {
      desc.setCanSleep(config.canSleep);
    }

    // Dominance
    if (config.dominanceGroup !== undefined) {
      desc.setDominanceGroup(config.dominanceGroup);
    }

    // Lock translations
    if (config.lockTranslations) {
      desc.lockTranslations();
      desc.enabledTranslations(
        !config.lockTranslations.x,
        !config.lockTranslations.y,
        !config.lockTranslations.z
      );
    }

    // Lock rotations
    if (config.lockRotations) {
      desc.lockRotations();
      desc.enabledRotations(
        !config.lockRotations.x,
        !config.lockRotations.y,
        !config.lockRotations.z
      );
    }

    return desc;
  }

  /**
   * Get body by handle.
   */
  get(handle: number): RAPIER.RigidBody | null {
    return this.world.getRigidBody(handle);
  }

  /**
   * Get body by user data.
   */
  getByUserData(userData: unknown): RAPIER.RigidBody | null {
    const handle = this.userDataMap.get(userData);
    if (handle === undefined) return null;
    return this.get(handle);
  }

  /**
   * Get handle info.
   */
  getHandle(handle: number): PhysicsHandle | undefined {
    return this.bodies.get(handle);
  }

  /**
   * Set body position.
   */
  setPosition(handle: number, position: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.setTranslation(
        new RAPIER.Vector3(position.x, position.y, position.z),
        true
      );
    }
  }

  /**
   * Set body rotation.
   */
  setRotation(handle: number, rotation: Quaternion): void {
    const body = this.get(handle);
    if (body) {
      body.setRotation(rotation, true);
    }
  }

  /**
   * Get body position.
   */
  getPosition(handle: number): Vector3 | null {
    const body = this.get(handle);
    if (!body) return null;
    const pos = body.translation();
    return { x: pos.x, y: pos.y, z: pos.z };
  }

  /**
   * Get body rotation.
   */
  getRotation(handle: number): Quaternion | null {
    const body = this.get(handle);
    if (!body) return null;
    const rot = body.rotation();
    return { x: rot.x, y: rot.y, z: rot.z, w: rot.w };
  }

  /**
   * Apply force to body.
   */
  applyForce(handle: number, force: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.addForce(new RAPIER.Vector3(force.x, force.y, force.z), true);
    }
  }

  /**
   * Apply impulse to body.
   */
  applyImpulse(handle: number, impulse: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.applyImpulse(
        new RAPIER.Vector3(impulse.x, impulse.y, impulse.z),
        true
      );
    }
  }

  /**
   * Apply torque impulse.
   */
  applyTorqueImpulse(handle: number, torque: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.applyTorqueImpulse(
        new RAPIER.Vector3(torque.x, torque.y, torque.z),
        true
      );
    }
  }

  /**
   * Set linear velocity.
   */
  setLinearVelocity(handle: number, velocity: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.setLinvel(
        new RAPIER.Vector3(velocity.x, velocity.y, velocity.z),
        true
      );
    }
  }

  /**
   * Get linear velocity.
   */
  getLinearVelocity(handle: number): Vector3 | null {
    const body = this.get(handle);
    if (!body) return null;
    const vel = body.linvel();
    return { x: vel.x, y: vel.y, z: vel.z };
  }

  /**
   * Set angular velocity.
   */
  setAngularVelocity(handle: number, velocity: Vector3): void {
    const body = this.get(handle);
    if (body) {
      body.setAngvel(
        new RAPIER.Vector3(velocity.x, velocity.y, velocity.z),
        true
      );
    }
  }

  /**
   * Get angular velocity.
   */
  getAngularVelocity(handle: number): Vector3 | null {
    const body = this.get(handle);
    if (!body) return null;
    const vel = body.angvel();
    return { x: vel.x, y: vel.y, z: vel.z };
  }

  /**
   * Wake up a sleeping body.
   */
  wakeUp(handle: number): void {
    const body = this.get(handle);
    if (body) {
      body.wakeUp();
    }
  }

  /**
   * Put body to sleep.
   */
  sleep(handle: number): void {
    const body = this.get(handle);
    if (body) {
      body.sleep();
    }
  }

  /**
   * Check if body is sleeping.
   */
  isSleeping(handle: number): boolean {
    const body = this.get(handle);
    return body?.isSleeping() ?? false;
  }

  /**
   * Remove a rigid body.
   */
  remove(handle: number): void {
    const body = this.get(handle);
    if (body) {
      const info = this.bodies.get(handle);
      if (info?.userData !== undefined) {
        this.userDataMap.delete(info.userData);
      }
      this.bodies.delete(handle);
      this.world.removeRigidBody(body);
    }
  }

  /**
   * Remove all bodies.
   */
  clear(): void {
    this.bodies.forEach((_, handle) => {
      this.remove(handle);
    });
    this.bodies.clear();
    this.userDataMap.clear();
  }

  /**
   * Get count of managed bodies.
   */
  get count(): number {
    return this.bodies.size;
  }

  /**
   * Iterate over all bodies.
   */
  forEach(
    callback: (handle: PhysicsHandle, body: RAPIER.RigidBody) => void
  ): void {
    this.bodies.forEach((handle, bodyHandle) => {
      const body = this.get(bodyHandle);
      if (body) {
        callback(handle, body);
      }
    });
  }
}
