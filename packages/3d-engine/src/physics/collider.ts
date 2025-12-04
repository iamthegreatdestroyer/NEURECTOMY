/**
 * Collider Manager
 *
 * Factory and lifecycle management for Rapier colliders.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import type { RapierWorld } from "./rapier-world";
import type { Vector3, Quaternion } from "./types";
import { ColliderShapeType } from "./types";

/**
 * Base collider shape configuration.
 */
interface BaseShapeConfig {
  type: ColliderShapeType;
}

interface BallShape extends BaseShapeConfig {
  type: ColliderShapeType.Ball;
  radius: number;
}

interface CapsuleShape extends BaseShapeConfig {
  type: ColliderShapeType.Capsule;
  halfHeight: number;
  radius: number;
}

interface CuboidShape extends BaseShapeConfig {
  type: ColliderShapeType.Cuboid;
  halfExtents: Vector3;
}

interface CylinderShape extends BaseShapeConfig {
  type: ColliderShapeType.Cylinder;
  halfHeight: number;
  radius: number;
}

interface ConeShape extends BaseShapeConfig {
  type: ColliderShapeType.Cone;
  halfHeight: number;
  radius: number;
}

interface ConvexShape extends BaseShapeConfig {
  type: ColliderShapeType.Convex;
  vertices: Float32Array;
}

interface TrimeshShape extends BaseShapeConfig {
  type: ColliderShapeType.Trimesh;
  vertices: Float32Array;
  indices: Uint32Array;
}

interface HeightfieldShape extends BaseShapeConfig {
  type: ColliderShapeType.Heightfield;
  rows: number;
  cols: number;
  heights: Float32Array;
  scale: Vector3;
}

export type ColliderShape =
  | BallShape
  | CapsuleShape
  | CuboidShape
  | CylinderShape
  | ConeShape
  | ConvexShape
  | TrimeshShape
  | HeightfieldShape;

export interface ColliderConfig {
  /** Shape configuration */
  shape: ColliderShape;
  /** Position offset from parent body */
  position?: Vector3;
  /** Rotation offset from parent body */
  rotation?: Quaternion;
  /** Density for mass computation */
  density?: number;
  /** Friction coefficient */
  friction?: number;
  /** Restitution (bounciness) */
  restitution?: number;
  /** Is this a sensor (triggers events but no collision response) */
  isSensor?: boolean;
  /** Collision groups */
  collisionGroups?: number;
  /** Solver groups */
  solverGroups?: number;
  /** Active collision types */
  activeCollisionTypes?: number;
  /** Active events */
  activeEvents?: number;
  /** User data */
  userData?: unknown;
}

/**
 * Manages collider creation and lifecycle.
 */
export class ColliderManager {
  private world: RapierWorld;
  private colliders: Map<number, ColliderConfig> = new Map();
  private userDataMap: Map<unknown, number> = new Map();

  constructor(world: RapierWorld) {
    this.world = world;
  }

  /**
   * Create a collider descriptor from shape config.
   */
  private createColliderDesc(shape: ColliderShape): RAPIER.ColliderDesc {
    switch (shape.type) {
      case ColliderShapeType.Ball:
        return RAPIER.ColliderDesc.ball(shape.radius);

      case ColliderShapeType.Capsule:
        return RAPIER.ColliderDesc.capsule(shape.halfHeight, shape.radius);

      case ColliderShapeType.Cuboid:
        return RAPIER.ColliderDesc.cuboid(
          shape.halfExtents.x,
          shape.halfExtents.y,
          shape.halfExtents.z
        );

      case ColliderShapeType.Cylinder:
        return RAPIER.ColliderDesc.cylinder(shape.halfHeight, shape.radius);

      case ColliderShapeType.Cone:
        return RAPIER.ColliderDesc.cone(shape.halfHeight, shape.radius);

      case ColliderShapeType.Convex:
        return RAPIER.ColliderDesc.convexHull(shape.vertices)!;

      case ColliderShapeType.Trimesh:
        return RAPIER.ColliderDesc.trimesh(shape.vertices, shape.indices);

      case ColliderShapeType.Heightfield:
        return RAPIER.ColliderDesc.heightfield(
          shape.rows,
          shape.cols,
          shape.heights,
          new RAPIER.Vector3(shape.scale.x, shape.scale.y, shape.scale.z)
        );

      default:
        throw new Error(
          `Unknown collider shape type: ${(shape as BaseShapeConfig).type}`
        );
    }
  }

  /**
   * Create a collider and optionally attach to a body.
   */
  create(config: ColliderConfig, bodyHandle?: number): number {
    const desc = this.createColliderDesc(config.shape);

    // Position offset
    if (config.position) {
      desc.setTranslation(
        config.position.x,
        config.position.y,
        config.position.z
      );
    }

    // Rotation offset
    if (config.rotation) {
      desc.setRotation(config.rotation);
    }

    // Physics properties
    if (config.density !== undefined) {
      desc.setDensity(config.density);
    }

    if (config.friction !== undefined) {
      desc.setFriction(config.friction);
    }

    if (config.restitution !== undefined) {
      desc.setRestitution(config.restitution);
    }

    // Sensor
    if (config.isSensor !== undefined) {
      desc.setSensor(config.isSensor);
    }

    // Collision groups
    if (config.collisionGroups !== undefined) {
      desc.setCollisionGroups(config.collisionGroups);
    }

    // Solver groups
    if (config.solverGroups !== undefined) {
      desc.setSolverGroups(config.solverGroups);
    }

    // Active collision types
    if (config.activeCollisionTypes !== undefined) {
      desc.setActiveCollisionTypes(config.activeCollisionTypes);
    }

    // Active events
    if (config.activeEvents !== undefined) {
      desc.setActiveEvents(config.activeEvents);
    }

    // Create collider
    const parent =
      bodyHandle !== undefined
        ? (this.world.getRigidBody(bodyHandle) ?? undefined)
        : undefined;

    const collider = this.world.createCollider(desc, parent);

    this.colliders.set(collider.handle, config);

    if (config.userData !== undefined) {
      this.userDataMap.set(config.userData, collider.handle);
    }

    return collider.handle;
  }

  /**
   * Get collider by handle.
   */
  get(handle: number): RAPIER.Collider | null {
    return this.world.getCollider(handle);
  }

  /**
   * Get collider by user data.
   */
  getByUserData(userData: unknown): RAPIER.Collider | null {
    const handle = this.userDataMap.get(userData);
    if (handle === undefined) return null;
    return this.get(handle);
  }

  /**
   * Get collider config.
   */
  getConfig(handle: number): ColliderConfig | undefined {
    return this.colliders.get(handle);
  }

  /**
   * Set collider friction.
   */
  setFriction(handle: number, friction: number): void {
    const collider = this.get(handle);
    if (collider) {
      collider.setFriction(friction);
    }
  }

  /**
   * Set collider restitution.
   */
  setRestitution(handle: number, restitution: number): void {
    const collider = this.get(handle);
    if (collider) {
      collider.setRestitution(restitution);
    }
  }

  /**
   * Set collider sensor state.
   */
  setSensor(handle: number, isSensor: boolean): void {
    const collider = this.get(handle);
    if (collider) {
      collider.setSensor(isSensor);
    }
  }

  /**
   * Set collision groups.
   */
  setCollisionGroups(handle: number, groups: number): void {
    const collider = this.get(handle);
    if (collider) {
      collider.setCollisionGroups(groups);
    }
  }

  /**
   * Remove a collider.
   */
  remove(handle: number): void {
    const collider = this.get(handle);
    if (collider) {
      const config = this.colliders.get(handle);
      if (config?.userData !== undefined) {
        this.userDataMap.delete(config.userData);
      }
      this.colliders.delete(handle);
      this.world.removeCollider(collider);
    }
  }

  /**
   * Remove all colliders.
   */
  clear(): void {
    this.colliders.forEach((_, handle) => {
      this.remove(handle);
    });
    this.colliders.clear();
    this.userDataMap.clear();
  }

  /**
   * Get count of managed colliders.
   */
  get count(): number {
    return this.colliders.size;
  }
}

/**
 * Helper to create collision group bitmasks.
 *
 * @param membership - Groups this collider belongs to (bits 16-31)
 * @param filter - Groups this collider can collide with (bits 0-15)
 */
export function createCollisionGroups(
  membership: number,
  filter: number
): number {
  return (membership << 16) | filter;
}

/**
 * Predefined collision groups.
 */
export const CollisionGroup = {
  DEFAULT: 0x0001,
  STATIC: 0x0002,
  DYNAMIC: 0x0004,
  KINEMATIC: 0x0008,
  SENSOR: 0x0010,
  PLAYER: 0x0020,
  ENEMY: 0x0040,
  PROJECTILE: 0x0080,
  TERRAIN: 0x0100,
  VEHICLE: 0x0200,
  ALL: 0xffff,
} as const;
