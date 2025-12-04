/**
 * Joint Manager
 *
 * Factory for creating physics joints/constraints between rigid bodies.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import type { RapierWorld } from "./rapier-world";
import type { Vector3 } from "./types";

/**
 * Joint type discriminator
 */
type JointType = "fixed" | "revolute" | "prismatic" | "spherical";

/**
 * Base joint configuration.
 */
interface BaseJointConfig {
  /** Joint type discriminator */
  type: JointType;
  /** First body handle */
  body1Handle: number;
  /** Second body handle */
  body2Handle: number;
  /** Wake up bodies when joint is created */
  wakeUp?: boolean;
  /** User data for this joint */
  userData?: unknown;
}

/**
 * Fixed joint - locks bodies together.
 */
export interface FixedJointConfig extends BaseJointConfig {
  type: "fixed";
  /** Local frame anchor on body 1 */
  localFrame1?: {
    translation: Vector3;
    rotation?: { x: number; y: number; z: number; w: number };
  };
  /** Local frame anchor on body 2 */
  localFrame2?: {
    translation: Vector3;
    rotation?: { x: number; y: number; z: number; w: number };
  };
}

/**
 * Revolute joint - rotation around single axis.
 */
export interface RevoluteJointConfig extends BaseJointConfig {
  type: "revolute";
  /** Anchor point on body 1 in local space */
  anchor1: Vector3;
  /** Anchor point on body 2 in local space */
  anchor2: Vector3;
  /** Axis of rotation in local space of body 1 */
  axis: Vector3;
  /** Joint limits (min, max) in radians */
  limits?: [number, number];
  /** Motor target velocity */
  motorVelocity?: number;
  /** Motor max torque */
  motorMaxTorque?: number;
}

/**
 * Prismatic joint - translation along single axis.
 */
export interface PrismaticJointConfig extends BaseJointConfig {
  type: "prismatic";
  /** Anchor point on body 1 */
  anchor1: Vector3;
  /** Anchor point on body 2 */
  anchor2: Vector3;
  /** Axis of translation */
  axis: Vector3;
  /** Joint limits (min, max) */
  limits?: [number, number];
  /** Motor target position */
  motorPosition?: number;
  /** Motor target velocity */
  motorVelocity?: number;
  /** Motor max force */
  motorMaxForce?: number;
}

/**
 * Spherical joint - rotation around all axes.
 */
export interface SphericalJointConfig extends BaseJointConfig {
  type: "spherical";
  /** Anchor point on body 1 */
  anchor1: Vector3;
  /** Anchor point on body 2 */
  anchor2: Vector3;
}

export type JointConfig =
  | FixedJointConfig
  | RevoluteJointConfig
  | PrismaticJointConfig
  | SphericalJointConfig;

interface JointHandle {
  handle: number;
  config: JointConfig;
}

/**
 * Manages joint creation and lifecycle.
 */
export class JointManager {
  private world: RapierWorld;
  private joints: Map<number, JointHandle> = new Map();
  private userDataMap: Map<unknown, number> = new Map();
  private handleCounter = 0;

  constructor(world: RapierWorld) {
    this.world = world;
  }

  /**
   * Create a joint between two bodies.
   */
  create(config: JointConfig): number {
    const body1 = this.world.getRigidBody(config.body1Handle);
    const body2 = this.world.getRigidBody(config.body2Handle);

    if (!body1 || !body2) {
      throw new Error("Cannot create joint: one or both bodies not found");
    }

    let jointData: RAPIER.JointData;

    switch (config.type) {
      case "fixed":
        jointData = this.createFixedJoint(config);
        break;
      case "revolute":
        jointData = this.createRevoluteJoint(config);
        break;
      case "prismatic":
        jointData = this.createPrismaticJoint(config);
        break;
      case "spherical":
        jointData = this.createSphericalJoint(config);
        break;
      default:
        throw new Error(
          `Unknown joint type: ${(config as BaseJointConfig).type}`
        );
    }

    const joint = this.world.createImpulseJoint(
      jointData,
      body1,
      body2,
      config.wakeUp ?? true
    );

    const internalHandle = this.handleCounter++;
    this.joints.set(internalHandle, {
      handle: joint.handle,
      config,
    });

    if (config.userData !== undefined) {
      this.userDataMap.set(config.userData, internalHandle);
    }

    return internalHandle;
  }

  /**
   * Create fixed joint data.
   */
  private createFixedJoint(config: FixedJointConfig): RAPIER.JointData {
    const frame1 = config.localFrame1 ?? { translation: { x: 0, y: 0, z: 0 } };
    const frame2 = config.localFrame2 ?? { translation: { x: 0, y: 0, z: 0 } };

    return RAPIER.JointData.fixed(
      new RAPIER.Vector3(
        frame1.translation.x,
        frame1.translation.y,
        frame1.translation.z
      ),
      frame1.rotation ?? { x: 0, y: 0, z: 0, w: 1 },
      new RAPIER.Vector3(
        frame2.translation.x,
        frame2.translation.y,
        frame2.translation.z
      ),
      frame2.rotation ?? { x: 0, y: 0, z: 0, w: 1 }
    );
  }

  /**
   * Create revolute joint data.
   */
  private createRevoluteJoint(config: RevoluteJointConfig): RAPIER.JointData {
    const anchor1 = new RAPIER.Vector3(
      config.anchor1.x,
      config.anchor1.y,
      config.anchor1.z
    );
    const anchor2 = new RAPIER.Vector3(
      config.anchor2.x,
      config.anchor2.y,
      config.anchor2.z
    );
    const axis = new RAPIER.Vector3(
      config.axis.x,
      config.axis.y,
      config.axis.z
    );

    const jointData = RAPIER.JointData.revolute(anchor1, anchor2, axis);

    if (config.limits) {
      jointData.limitsEnabled = true;
      jointData.limits = config.limits;
    }

    return jointData;
  }

  /**
   * Create prismatic joint data.
   */
  private createPrismaticJoint(config: PrismaticJointConfig): RAPIER.JointData {
    const anchor1 = new RAPIER.Vector3(
      config.anchor1.x,
      config.anchor1.y,
      config.anchor1.z
    );
    const anchor2 = new RAPIER.Vector3(
      config.anchor2.x,
      config.anchor2.y,
      config.anchor2.z
    );
    const axis = new RAPIER.Vector3(
      config.axis.x,
      config.axis.y,
      config.axis.z
    );

    const jointData = RAPIER.JointData.prismatic(anchor1, anchor2, axis);

    if (config.limits) {
      jointData.limitsEnabled = true;
      jointData.limits = config.limits;
    }

    return jointData;
  }

  /**
   * Create spherical joint data.
   */
  private createSphericalJoint(config: SphericalJointConfig): RAPIER.JointData {
    const anchor1 = new RAPIER.Vector3(
      config.anchor1.x,
      config.anchor1.y,
      config.anchor1.z
    );
    const anchor2 = new RAPIER.Vector3(
      config.anchor2.x,
      config.anchor2.y,
      config.anchor2.z
    );

    return RAPIER.JointData.spherical(anchor1, anchor2);
  }

  /**
   * Get joint by handle.
   */
  get(handle: number): JointHandle | undefined {
    return this.joints.get(handle);
  }

  /**
   * Get joint by user data.
   */
  getByUserData(userData: unknown): JointHandle | undefined {
    const handle = this.userDataMap.get(userData);
    if (handle === undefined) return undefined;
    return this.get(handle);
  }

  /**
   * Remove a joint.
   */
  remove(handle: number): void {
    const jointHandle = this.joints.get(handle);
    if (jointHandle) {
      const config = jointHandle.config;
      if (config.userData !== undefined) {
        this.userDataMap.delete(config.userData);
      }
      this.joints.delete(handle);
      // Note: Rapier joint removal would require accessing the raw world
    }
  }

  /**
   * Get count of joints.
   */
  get count(): number {
    return this.joints.size;
  }

  /**
   * Clear all joints.
   */
  clear(): void {
    this.joints.forEach((_, handle) => {
      this.remove(handle);
    });
    this.joints.clear();
    this.userDataMap.clear();
  }
}
