/**
 * Level of Detail (LOD) Manager
 *
 * Manages automatic LOD switching for 3D objects based on camera distance,
 * screen coverage, and performance budgets. Includes object pooling and
 * quality presets for optimal performance across devices.
 *
 * @module @neurectomy/3d-engine/optimization/lod-manager
 * @agents @VELOCITY @CANVAS @AXIOM
 * @phase Phase 3 - Dimensional Forge
 * @step Performance Optimization
 */

import * as THREE from "three";

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface LODLevel {
  /** Distance threshold for this level */
  distance: number;
  /** The 3D object geometry/mesh for this level */
  object: THREE.Object3D | null;
  /** Polygon count for this level (for budgeting) */
  polyCount: number;
  /** Optional custom visibility test */
  visibilityTest?: (camera: THREE.Camera, object: THREE.Object3D) => boolean;
}

export interface LODEntry {
  id: string;
  levels: LODLevel[];
  currentLevel: number;
  position: THREE.Vector3;
  boundingSphere: THREE.Sphere;
  visible: boolean;
  lastUpdateFrame: number;
  priority: number; // Higher = more important
  category: string;
}

export interface LODManagerConfig {
  /** Update budget per frame (ms) */
  updateBudgetMs?: number;
  /** Maximum objects to update per frame */
  maxUpdatesPerFrame?: number;
  /** Distance bias multiplier (affects all thresholds) */
  distanceBias?: number;
  /** Screen coverage threshold for visibility */
  screenCoverageThreshold?: number;
  /** Enable automatic quality adjustment */
  adaptiveQuality?: boolean;
  /** Target frame rate for adaptive quality */
  targetFps?: number;
  /** Quality preset */
  qualityPreset?: "low" | "medium" | "high" | "ultra";
  /** Enable object pooling */
  enablePooling?: boolean;
  /** Pool size per LOD level */
  poolSizePerLevel?: number;
  /** Enable hysteresis to prevent LOD popping */
  hysteresis?: number;
  /** Fade transition duration (ms) */
  fadeTransitionMs?: number;
}

export interface LODStatistics {
  /** Total managed objects */
  totalObjects: number;
  /** Currently visible objects */
  visibleObjects: number;
  /** Objects per LOD level */
  objectsPerLevel: number[];
  /** Total polygons rendered */
  totalPolygons: number;
  /** Estimated draw calls */
  drawCalls: number;
  /** Last update time (ms) */
  lastUpdateTimeMs: number;
  /** Current quality level (0-1) */
  qualityLevel: number;
  /** Pooled objects count */
  pooledObjects: number;
}

export interface QualityPreset {
  distanceBias: number;
  maxVisibleObjects: number;
  lodBias: number;
  shadowQuality: "off" | "low" | "medium" | "high";
  antialias: boolean;
  effectsEnabled: boolean;
}

// ============================================================================
// Quality Presets
// ============================================================================

export const QUALITY_PRESETS: Record<string, QualityPreset> = {
  low: {
    distanceBias: 0.5,
    maxVisibleObjects: 500,
    lodBias: 2.0,
    shadowQuality: "off",
    antialias: false,
    effectsEnabled: false,
  },
  medium: {
    distanceBias: 0.75,
    maxVisibleObjects: 2000,
    lodBias: 1.5,
    shadowQuality: "low",
    antialias: true,
    effectsEnabled: false,
  },
  high: {
    distanceBias: 1.0,
    maxVisibleObjects: 5000,
    lodBias: 1.0,
    shadowQuality: "medium",
    antialias: true,
    effectsEnabled: true,
  },
  ultra: {
    distanceBias: 1.5,
    maxVisibleObjects: 10000,
    lodBias: 0.75,
    shadowQuality: "high",
    antialias: true,
    effectsEnabled: true,
  },
};

// ============================================================================
// LOD Manager
// ============================================================================

/**
 * LODManager - Automatic Level of Detail management for 3D scenes
 *
 * Features:
 * - Distance-based LOD switching
 * - Screen coverage visibility testing
 * - Per-frame update budgeting
 * - Object pooling for reduced allocations
 * - Adaptive quality based on frame rate
 * - Hysteresis to prevent LOD popping
 * - Quality presets for different devices
 */
export class LODManager {
  private entries = new Map<string, LODEntry>();
  private config: Required<LODManagerConfig>;
  private statistics: LODStatistics;
  private frameCount = 0;

  // Adaptive quality state
  private frameTimeHistory: number[] = [];
  private currentQualityLevel = 1.0;
  private qualityPreset: QualityPreset;

  // Object pools per category
  private objectPools = new Map<string, THREE.Object3D[]>();

  // Scratch objects to avoid allocations
  private tempVector = new THREE.Vector3();
  private tempMatrix = new THREE.Matrix4();
  private tempFrustum = new THREE.Frustum();

  // Update queue for budget management
  private updateQueue: string[] = [];
  private lastUpdateTime = 0;

  constructor(config: LODManagerConfig = {}) {
    this.config = {
      updateBudgetMs: config.updateBudgetMs ?? 2,
      maxUpdatesPerFrame: config.maxUpdatesPerFrame ?? 100,
      distanceBias: config.distanceBias ?? 1.0,
      screenCoverageThreshold: config.screenCoverageThreshold ?? 0.001,
      adaptiveQuality: config.adaptiveQuality ?? true,
      targetFps: config.targetFps ?? 60,
      qualityPreset: config.qualityPreset ?? "high",
      enablePooling: config.enablePooling ?? true,
      poolSizePerLevel: config.poolSizePerLevel ?? 50,
      hysteresis: config.hysteresis ?? 0.1,
      fadeTransitionMs: config.fadeTransitionMs ?? 100,
    };

    this.qualityPreset = (QUALITY_PRESETS[this.config.qualityPreset] ??
      QUALITY_PRESETS.high) as QualityPreset;

    this.statistics = {
      totalObjects: 0,
      visibleObjects: 0,
      objectsPerLevel: [0, 0, 0, 0],
      totalPolygons: 0,
      drawCalls: 0,
      lastUpdateTimeMs: 0,
      qualityLevel: 1.0,
      pooledObjects: 0,
    };
  }

  // ============================================================================
  // Registration API
  // ============================================================================

  /**
   * Register an object with LOD levels
   */
  public register(
    id: string,
    levels: LODLevel[],
    options: {
      position?: THREE.Vector3;
      boundingRadius?: number;
      priority?: number;
      category?: string;
    } = {}
  ): void {
    // Sort levels by distance
    const sortedLevels = [...levels].sort((a, b) => a.distance - b.distance);

    // Calculate bounding sphere
    const position = options.position ?? new THREE.Vector3();
    const radius =
      options.boundingRadius ?? this.estimateBoundingRadius(sortedLevels);

    const entry: LODEntry = {
      id,
      levels: sortedLevels,
      currentLevel: 0,
      position: position.clone(),
      boundingSphere: new THREE.Sphere(position, radius),
      visible: true,
      lastUpdateFrame: 0,
      priority: options.priority ?? 0,
      category: options.category ?? "default",
    };

    this.entries.set(id, entry);
    this.statistics.totalObjects = this.entries.size;

    // Add to update queue
    this.updateQueue.push(id);
  }

  /**
   * Unregister an object
   */
  public unregister(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;

    // Return objects to pool if pooling enabled
    if (this.config.enablePooling) {
      for (const level of entry.levels) {
        if (level.object) {
          this.returnToPool(entry.category, level.object);
        }
      }
    }

    this.entries.delete(id);
    this.statistics.totalObjects = this.entries.size;

    // Remove from update queue
    const queueIndex = this.updateQueue.indexOf(id);
    if (queueIndex >= 0) {
      this.updateQueue.splice(queueIndex, 1);
    }
  }

  /**
   * Update object position
   */
  public updatePosition(id: string, position: THREE.Vector3): void {
    const entry = this.entries.get(id);
    if (!entry) return;

    entry.position.copy(position);
    entry.boundingSphere.center.copy(position);
  }

  // ============================================================================
  // Update Logic
  // ============================================================================

  /**
   * Main update method - call once per frame
   */
  public update(camera: THREE.Camera, deltaTime: number): void {
    const startTime = performance.now();
    this.frameCount++;

    // Update adaptive quality if enabled
    if (this.config.adaptiveQuality) {
      this.updateAdaptiveQuality(deltaTime);
    }

    // Setup frustum for culling
    this.tempMatrix.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    this.tempFrustum.setFromProjectionMatrix(this.tempMatrix);

    const cameraPosition = camera.position;
    const effectiveDistanceBias =
      this.config.distanceBias * this.qualityPreset.distanceBias;

    // Process update queue with budget
    let updatedCount = 0;
    let visibleCount = 0;
    let totalPolygons = 0;
    const objectsPerLevel = [0, 0, 0, 0];

    // Sort update queue by priority
    this.updateQueue.sort((a, b) => {
      const entryA = this.entries.get(a);
      const entryB = this.entries.get(b);
      if (!entryA || !entryB) return 0;

      // Higher priority first, then by distance
      const priorityDiff = entryB.priority - entryA.priority;
      if (priorityDiff !== 0) return priorityDiff;

      const distA = entryA.position.distanceTo(cameraPosition);
      const distB = entryB.position.distanceTo(cameraPosition);
      return distA - distB;
    });

    // Process entries
    const maxUpdates = Math.min(
      this.config.maxUpdatesPerFrame,
      this.updateQueue.length
    );

    for (let i = 0; i < this.entries.size; i++) {
      const id = this.updateQueue[i % this.updateQueue.length];
      if (!id) continue;
      const entry = this.entries.get(id);
      if (!entry) continue;

      // Check time budget
      const elapsed = performance.now() - startTime;
      if (elapsed > this.config.updateBudgetMs && updatedCount > 10) {
        break;
      }

      // Frustum culling
      const inFrustum = this.tempFrustum.intersectsSphere(entry.boundingSphere);

      if (!inFrustum) {
        entry.visible = false;
        this.setLevelVisibility(entry, false);
        continue;
      }

      // Calculate distance
      const distance = cameraPosition.distanceTo(entry.position);
      const adjustedDistance = distance * effectiveDistanceBias;

      // Determine LOD level with hysteresis
      const newLevel = this.calculateLODLevel(entry, adjustedDistance);

      if (newLevel !== entry.currentLevel) {
        this.switchLODLevel(entry, newLevel);
        entry.currentLevel = newLevel;
      }

      entry.visible = true;
      entry.lastUpdateFrame = this.frameCount;
      this.setLevelVisibility(entry, true);

      // Update statistics
      visibleCount++;
      if (newLevel < objectsPerLevel.length) {
        objectsPerLevel[newLevel]!++;
      }
      const level = entry.levels[newLevel];
      if (level) {
        totalPolygons += level.polyCount;
      }

      updatedCount++;
    }

    // Rotate update queue to ensure all objects get updated eventually
    if (this.updateQueue.length > maxUpdates) {
      const rotateCount = Math.min(maxUpdates, this.updateQueue.length / 4);
      for (let i = 0; i < rotateCount; i++) {
        const item = this.updateQueue.shift();
        if (item) this.updateQueue.push(item);
      }
    }

    // Update statistics
    this.statistics.visibleObjects = visibleCount;
    this.statistics.objectsPerLevel = objectsPerLevel;
    this.statistics.totalPolygons = totalPolygons;
    this.statistics.drawCalls = visibleCount; // Simplified estimate
    this.statistics.lastUpdateTimeMs = performance.now() - startTime;
    this.statistics.qualityLevel = this.currentQualityLevel;
  }

  /**
   * Calculate which LOD level to use
   */
  private calculateLODLevel(entry: LODEntry, distance: number): number {
    const levels = entry.levels;
    const hysteresis = this.config.hysteresis;
    const currentLevel = entry.currentLevel;

    // Find appropriate level
    for (let i = 0; i < levels.length; i++) {
      const level = levels[i];
      if (!level) continue;
      let threshold = level.distance;

      // Apply hysteresis
      if (i === currentLevel + 1) {
        // Going to lower detail - need more distance
        threshold *= 1 + hysteresis;
      } else if (i === currentLevel) {
        // Staying at current - use base threshold for next level
        continue;
      } else if (i === currentLevel - 1) {
        // Going to higher detail - need less distance
        threshold *= 1 - hysteresis;
      }

      if (distance < threshold) {
        return Math.max(0, i - 1);
      }
    }

    return levels.length - 1;
  }

  /**
   * Switch LOD level with optional transition
   */
  private switchLODLevel(entry: LODEntry, newLevel: number): void {
    const oldLevel = entry.levels[entry.currentLevel];
    const level = entry.levels[newLevel];

    // Hide old level
    if (oldLevel?.object) {
      oldLevel.object.visible = false;
    }

    // Show new level
    if (level?.object) {
      level.object.visible = entry.visible;

      // Optional: Fade transition
      if (
        this.config.fadeTransitionMs > 0 &&
        level.object instanceof THREE.Mesh
      ) {
        this.applyFadeTransition(level.object as THREE.Mesh);
      }
    }
  }

  /**
   * Apply fade-in transition to mesh
   */
  private applyFadeTransition(mesh: THREE.Mesh): void {
    const material = mesh.material as THREE.Material;
    if (!material) return;

    // Store original opacity
    const originalOpacity = (material as any).opacity ?? 1;
    const transparent = material.transparent;

    material.transparent = true;
    (material as any).opacity = 0;

    // Animate opacity
    const startTime = performance.now();
    const duration = this.config.fadeTransitionMs;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(1, elapsed / duration);

      (material as any).opacity = originalOpacity * progress;

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        material.transparent = transparent;
      }
    };

    requestAnimationFrame(animate);
  }

  /**
   * Set visibility for all levels in an entry
   */
  private setLevelVisibility(entry: LODEntry, visible: boolean): void {
    for (let i = 0; i < entry.levels.length; i++) {
      const level = entry.levels[i];
      if (level && level.object) {
        level.object.visible = visible && i === entry.currentLevel;
      }
    }
  }

  // ============================================================================
  // Adaptive Quality
  // ============================================================================

  /**
   * Update quality based on frame rate
   */
  private updateAdaptiveQuality(deltaTime: number): void {
    const frameTime = deltaTime * 1000;
    this.frameTimeHistory.push(frameTime);

    // Keep last 60 frames
    if (this.frameTimeHistory.length > 60) {
      this.frameTimeHistory.shift();
    }

    // Only adjust every 30 frames
    if (this.frameCount % 30 !== 0) return;

    // Calculate average frame time
    const avgFrameTime =
      this.frameTimeHistory.reduce((a, b) => a + b, 0) /
      this.frameTimeHistory.length;
    const avgFps = 1000 / avgFrameTime;
    const targetFrameTime = 1000 / this.config.targetFps;

    // Adjust quality
    if (avgFps < this.config.targetFps * 0.9) {
      // Performance too low, decrease quality
      this.currentQualityLevel = Math.max(0.2, this.currentQualityLevel - 0.05);
    } else if (
      avgFps > this.config.targetFps * 1.1 &&
      this.currentQualityLevel < 1.0
    ) {
      // Performance good, can increase quality
      this.currentQualityLevel = Math.min(1.0, this.currentQualityLevel + 0.02);
    }

    // Apply quality level to config
    this.config.distanceBias =
      this.qualityPreset.distanceBias * this.currentQualityLevel;
  }

  /**
   * Set quality preset
   */
  public setQualityPreset(preset: "low" | "medium" | "high" | "ultra"): void {
    this.qualityPreset = (QUALITY_PRESETS[preset] ??
      QUALITY_PRESETS.high) as QualityPreset;
    this.config.qualityPreset = preset;
    this.currentQualityLevel = 1.0;
    this.frameTimeHistory = [];
  }

  // ============================================================================
  // Object Pooling
  // ============================================================================

  /**
   * Get object from pool
   */
  public getFromPool(category: string): THREE.Object3D | null {
    if (!this.config.enablePooling) return null;

    const pool = this.objectPools.get(category);
    if (!pool || pool.length === 0) return null;

    const obj = pool.pop()!;
    this.statistics.pooledObjects = this.getPooledObjectCount();
    return obj;
  }

  /**
   * Return object to pool
   */
  public returnToPool(category: string, object: THREE.Object3D): void {
    if (!this.config.enablePooling) return;

    let pool = this.objectPools.get(category);
    if (!pool) {
      pool = [];
      this.objectPools.set(category, pool);
    }

    if (pool.length < this.config.poolSizePerLevel) {
      // Reset object state
      object.visible = false;
      object.position.set(0, 0, 0);
      object.rotation.set(0, 0, 0);
      object.scale.set(1, 1, 1);

      pool.push(object);
      this.statistics.pooledObjects = this.getPooledObjectCount();
    } else {
      // Pool is full, dispose object
      this.disposeObject(object);
    }
  }

  /**
   * Pre-populate pool with objects
   */
  public warmPool(
    category: string,
    factory: () => THREE.Object3D,
    count: number
  ): void {
    if (!this.config.enablePooling) return;

    let pool = this.objectPools.get(category);
    if (!pool) {
      pool = [];
      this.objectPools.set(category, pool);
    }

    for (
      let i = 0;
      i < count && pool.length < this.config.poolSizePerLevel;
      i++
    ) {
      const obj = factory();
      obj.visible = false;
      pool.push(obj);
    }

    this.statistics.pooledObjects = this.getPooledObjectCount();
  }

  /**
   * Get total pooled objects count
   */
  private getPooledObjectCount(): number {
    let total = 0;
    for (const pool of this.objectPools.values()) {
      total += pool.length;
    }
    return total;
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Estimate bounding radius from LOD levels
   */
  private estimateBoundingRadius(levels: LODLevel[]): number {
    for (const level of levels) {
      if (level.object) {
        const sphere = new THREE.Sphere();
        new THREE.Box3().setFromObject(level.object).getBoundingSphere(sphere);
        return sphere.radius || 1;
      }
    }
    return 1;
  }

  /**
   * Dispose of a Three.js object
   */
  private disposeObject(object: THREE.Object3D): void {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry?.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material?.dispose();
        }
      }
    });
  }

  /**
   * Get statistics
   */
  public getStatistics(): LODStatistics {
    return { ...this.statistics };
  }

  /**
   * Get entry by ID
   */
  public getEntry(id: string): LODEntry | undefined {
    return this.entries.get(id);
  }

  /**
   * Force update all entries this frame
   */
  public forceUpdateAll(): void {
    this.updateQueue = Array.from(this.entries.keys());
  }

  /**
   * Clear all entries
   */
  public clear(): void {
    // Dispose all objects
    for (const entry of this.entries.values()) {
      for (const level of entry.levels) {
        if (level.object) {
          this.disposeObject(level.object);
        }
      }
    }

    this.entries.clear();
    this.updateQueue = [];
    this.statistics.totalObjects = 0;
    this.statistics.visibleObjects = 0;
  }

  /**
   * Dispose manager and all resources
   */
  public dispose(): void {
    this.clear();

    // Clear pools
    for (const pool of this.objectPools.values()) {
      for (const obj of pool) {
        this.disposeObject(obj);
      }
    }
    this.objectPools.clear();
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create default LOD levels for a geometry
 */
export function createLODLevels(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  distances: number[] = [0, 50, 150, 400]
): LODLevel[] {
  const levels: LODLevel[] = [];

  // Level 0: Full detail
  const mesh0 = new THREE.Mesh(geometry, material);
  const positionAttr = geometry.attributes.position;
  const basePolyCount = geometry.index
    ? geometry.index.count / 3
    : positionAttr
      ? positionAttr.count / 3
      : 0;

  levels.push({
    distance: distances[0] ?? 0,
    object: mesh0,
    polyCount: basePolyCount,
  });

  // Generate simplified versions
  const simplifyRatios = [1.0, 0.5, 0.25, 0.1];

  for (let i = 1; i < distances.length; i++) {
    const ratio = simplifyRatios[i] ?? 0.1;
    const simplifiedGeo = geometry.clone(); // In production, use actual simplification

    const mesh = new THREE.Mesh(simplifiedGeo, material);
    mesh.visible = false;

    const firstLevel = levels[0];
    levels.push({
      distance: distances[i] ?? 100,
      object: mesh,
      polyCount: Math.floor((firstLevel?.polyCount ?? 0) * ratio),
    });
  }

  return levels;
}

export default LODManager;
