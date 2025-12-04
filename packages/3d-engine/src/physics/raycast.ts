/**
 * Raycast Manager
 *
 * High-performance raycasting and shape casting for collision detection.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import type { RapierWorld } from "./rapier-world";
import type { Vector3 } from "./types";

export interface RaycastOptions {
  /** Maximum distance for the ray */
  maxToi?: number;
  /** Solid shapes intersect at entry, non-solid at exit */
  solid?: boolean;
  /** Filter by collision groups */
  filterGroups?: number;
  /** Exclude specific collider handles */
  excludeColliders?: number[];
  /** Exclude specific rigid body handles */
  excludeBodies?: number[];
}

export interface RaycastResult {
  /** Did the ray hit something? */
  hit: boolean;
  /** Handle of the hit collider */
  colliderHandle?: number;
  /** Handle of the rigid body owning the collider */
  bodyHandle?: number;
  /** Time of impact (distance along ray direction) */
  toi?: number;
  /** World-space hit point */
  point?: Vector3;
  /** Surface normal at hit point */
  normal?: Vector3;
}

export interface ShapeCastResult extends RaycastResult {
  /** Witness point on the shape */
  witness1?: Vector3;
  /** Witness point on the hit geometry */
  witness2?: Vector3;
}

/**
 * Manages raycasting and shape casting operations.
 */
export class RaycastManager {
  private world: RapierWorld;

  constructor(world: RapierWorld) {
    this.world = world;
  }

  /**
   * Cast a ray and return the first hit.
   */
  castRay(
    origin: Vector3,
    direction: Vector3,
    options: RaycastOptions = {}
  ): RaycastResult {
    const raw = this.world.getRawWorld();

    const ray = new RAPIER.Ray(
      new RAPIER.Vector3(origin.x, origin.y, origin.z),
      new RAPIER.Vector3(direction.x, direction.y, direction.z)
    );

    const maxToi = options.maxToi ?? Infinity;
    const solid = options.solid ?? true;

    // Create filter predicate
    const filterPredicate = this.createFilterPredicate(options);

    // Use castRayAndGetNormal to get both hit info and normal
    const hit = raw.castRayAndGetNormal(
      ray,
      maxToi,
      solid,
      undefined, // QueryFilterFlags
      options.filterGroups,
      undefined, // Collider to exclude
      undefined, // Body to exclude
      filterPredicate
    );

    if (!hit) {
      return { hit: false };
    }

    const collider = raw.getCollider(hit.collider.handle);
    const bodyHandle = collider?.parent()?.handle;

    // Calculate hit point
    const point = ray.pointAt(hit.toi);

    return {
      hit: true,
      colliderHandle: hit.collider.handle,
      bodyHandle: bodyHandle ?? undefined,
      toi: hit.toi,
      point: { x: point.x, y: point.y, z: point.z },
      normal: hit.normal
        ? { x: hit.normal.x, y: hit.normal.y, z: hit.normal.z }
        : undefined,
    };
  }

  /**
   * Cast a ray and return all hits.
   */
  castRayAll(
    origin: Vector3,
    direction: Vector3,
    options: RaycastOptions = {}
  ): RaycastResult[] {
    const raw = this.world.getRawWorld();
    const results: RaycastResult[] = [];

    const ray = new RAPIER.Ray(
      new RAPIER.Vector3(origin.x, origin.y, origin.z),
      new RAPIER.Vector3(direction.x, direction.y, direction.z)
    );

    const maxToi = options.maxToi ?? Infinity;
    const solid = options.solid ?? true;
    const filterPredicate = this.createFilterPredicate(options);

    raw.intersectionsWithRay(
      ray,
      maxToi,
      solid,
      (intersection) => {
        const collider = raw.getCollider(intersection.collider.handle);
        const bodyHandle = collider?.parent()?.handle;
        const point = ray.pointAt(intersection.toi);

        results.push({
          hit: true,
          colliderHandle: intersection.collider.handle,
          bodyHandle: bodyHandle ?? undefined,
          toi: intersection.toi,
          point: { x: point.x, y: point.y, z: point.z },
          normal: intersection.normal
            ? {
                x: intersection.normal.x,
                y: intersection.normal.y,
                z: intersection.normal.z,
              }
            : undefined,
        });

        return true; // Continue iterating
      },
      undefined,
      options.filterGroups,
      undefined,
      undefined,
      filterPredicate
    );

    // Sort by distance
    results.sort((a, b) => (a.toi ?? 0) - (b.toi ?? 0));

    return results;
  }

  /**
   * Check if a point is inside any collider.
   */
  pointInside(point: Vector3, options: RaycastOptions = {}): RaycastResult {
    const raw = this.world.getRawWorld();
    const p = new RAPIER.Vector3(point.x, point.y, point.z);
    const filterPredicate = this.createFilterPredicate(options);

    const collider = raw.intersectionsWithPoint(
      p,
      (intersection) => {
        // Return false to stop at first hit
        return false;
      },
      undefined,
      options.filterGroups,
      undefined,
      undefined,
      filterPredicate
    );

    // The callback approach doesn't return the collider directly,
    // so we need to use projectPoint instead for a single result
    const projected = raw.projectPoint(
      p,
      true,
      undefined,
      options.filterGroups
    );

    if (!projected) {
      return { hit: false };
    }

    const projCollider = raw.getCollider(projected.collider.handle);
    const bodyHandle = projCollider?.parent()?.handle;

    return {
      hit: projected.isInside,
      colliderHandle: projected.collider.handle,
      bodyHandle: bodyHandle ?? undefined,
      point: {
        x: projected.point.x,
        y: projected.point.y,
        z: projected.point.z,
      },
    };
  }

  /**
   * Project a point onto the nearest collider surface.
   */
  projectPoint(
    point: Vector3,
    solid: boolean = true,
    options: RaycastOptions = {}
  ): RaycastResult {
    const raw = this.world.getRawWorld();
    const p = new RAPIER.Vector3(point.x, point.y, point.z);

    const projected = raw.projectPoint(
      p,
      solid,
      undefined,
      options.filterGroups
    );

    if (!projected) {
      return { hit: false };
    }

    const collider = raw.getCollider(projected.collider.handle);
    const bodyHandle = collider?.parent()?.handle;

    return {
      hit: true,
      colliderHandle: projected.collider.handle,
      bodyHandle: bodyHandle ?? undefined,
      point: {
        x: projected.point.x,
        y: projected.point.y,
        z: projected.point.z,
      },
    };
  }

  /**
   * Get all colliders intersecting with a sphere.
   */
  intersectSphere(
    center: Vector3,
    radius: number,
    options: RaycastOptions = {}
  ): number[] {
    const raw = this.world.getRawWorld();
    const results: number[] = [];

    const shape = new RAPIER.Ball(radius);
    const position = new RAPIER.Vector3(center.x, center.y, center.z);
    const rotation = { x: 0, y: 0, z: 0, w: 1 };
    const filterPredicate = this.createFilterPredicate(options);

    raw.intersectionsWithShape(
      position,
      rotation,
      shape,
      (collider) => {
        results.push(collider.handle);
        return true; // Continue
      },
      undefined,
      options.filterGroups,
      undefined,
      undefined,
      filterPredicate
    );

    return results;
  }

  /**
   * Get all colliders intersecting with a box.
   */
  intersectBox(
    center: Vector3,
    halfExtents: Vector3,
    options: RaycastOptions = {}
  ): number[] {
    const raw = this.world.getRawWorld();
    const results: number[] = [];

    const shape = new RAPIER.Cuboid(
      halfExtents.x,
      halfExtents.y,
      halfExtents.z
    );
    const position = new RAPIER.Vector3(center.x, center.y, center.z);
    const rotation = { x: 0, y: 0, z: 0, w: 1 };
    const filterPredicate = this.createFilterPredicate(options);

    raw.intersectionsWithShape(
      position,
      rotation,
      shape,
      (collider) => {
        results.push(collider.handle);
        return true;
      },
      undefined,
      options.filterGroups,
      undefined,
      undefined,
      filterPredicate
    );

    return results;
  }

  /**
   * Create a filter predicate from options.
   */
  private createFilterPredicate(
    options: RaycastOptions
  ): ((collider: RAPIER.Collider) => boolean) | undefined {
    const excludeColliders = options.excludeColliders;
    const excludeBodies = options.excludeBodies;

    if (!excludeColliders?.length && !excludeBodies?.length) {
      return undefined;
    }

    return (collider: RAPIER.Collider) => {
      if (excludeColliders?.includes(collider.handle)) {
        return false;
      }

      const parent = collider.parent();
      if (parent && excludeBodies?.includes(parent.handle)) {
        return false;
      }

      return true;
    };
  }
}
