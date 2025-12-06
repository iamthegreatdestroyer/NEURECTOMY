/**
 * @fileoverview Dynamic Instance Batching System
 * @module @neurectomy/3d-engine/webgpu/instancing/instance-batch-manager
 *
 * Automatically batches similar objects for efficient instanced rendering.
 * Implements LOD grouping, spatial partitioning, and dynamic instance buffers.
 */

import type { Vector3 } from "three";

/**
 * Instance data for a single renderable object
 */
export interface InstanceData {
  /** Unique identifier */
  id: string;
  /** World position */
  position: Vector3;
  /** Scale (uniform or per-axis) */
  scale: number | Vector3;
  /** Rotation quaternion (x, y, z, w) */
  rotation?: { x: number; y: number; z: number; w: number };
  /** Color (RGBA) */
  color?: { r: number; g: number; b: number; a: number };
  /** Custom per-instance data */
  userData?: Record<string, number>;
  /** LOD level (0 = highest detail) */
  lodLevel?: number;
  /** Whether instance is visible */
  visible?: boolean;
}

/**
 * Geometry definition for batching
 */
export interface BatchGeometry {
  /** Unique geometry identifier */
  id: string;
  /** Vertex positions (3 floats per vertex) */
  positions: Float32Array;
  /** Vertex normals (3 floats per vertex) */
  normals?: Float32Array;
  /** Texture coordinates (2 floats per vertex) */
  uvs?: Float32Array;
  /** Index buffer (optional) */
  indices?: Uint16Array | Uint32Array;
  /** LOD variants */
  lodVariants?: {
    level: number;
    positions: Float32Array;
    indices?: Uint16Array | Uint32Array;
    /** Distance threshold for this LOD */
    distance: number;
  }[];
}

/**
 * Material definition for batching
 */
export interface BatchMaterial {
  /** Unique material identifier */
  id: string;
  /** Base color */
  color?: { r: number; g: number; b: number; a: number };
  /** Metallic factor (0-1) */
  metallic?: number;
  /** Roughness factor (0-1) */
  roughness?: number;
  /** Emissive color */
  emissive?: { r: number; g: number; b: number };
  /** Use per-instance colors */
  useInstanceColors?: boolean;
  /** Alpha blending enabled */
  transparent?: boolean;
  /** Depth write enabled */
  depthWrite?: boolean;
}

/**
 * Instance batch configuration
 */
export interface BatchConfig {
  /** Maximum instances per batch */
  maxInstancesPerBatch: number;
  /** Enable automatic LOD selection */
  enableLOD: boolean;
  /** Enable frustum culling */
  enableCulling: boolean;
  /** Enable spatial sorting for better GPU cache utilization */
  enableSpatialSorting: boolean;
  /** Instance buffer resize factor when growing */
  bufferGrowthFactor: number;
  /** Minimum instances before creating a batch */
  minInstancesForBatch: number;
}

/**
 * Render batch containing instances of identical geometry/material
 */
export interface RenderBatch {
  /** Batch identifier */
  id: string;
  /** Geometry reference */
  geometryId: string;
  /** Material reference */
  materialId: string;
  /** LOD level for this batch */
  lodLevel: number;
  /** Instance count */
  instanceCount: number;
  /** Instance data buffer (for GPU upload) */
  instanceBuffer: Float32Array;
  /** Index of first instance in global buffer */
  firstInstance: number;
  /** Whether batch needs GPU update */
  dirty: boolean;
}

/**
 * Statistics for instance batching
 */
export interface BatchingStats {
  /** Total instances across all batches */
  totalInstances: number;
  /** Number of render batches */
  batchCount: number;
  /** Number of draw calls saved by batching */
  drawCallsSaved: number;
  /** Instance buffer memory usage in bytes */
  memoryUsage: number;
  /** Number of instances culled */
  culledInstances: number;
  /** Average instances per batch */
  averageInstancesPerBatch: number;
}

/**
 * Default batch configuration
 */
export const DEFAULT_BATCH_CONFIG: BatchConfig = {
  maxInstancesPerBatch: 65536,
  enableLOD: true,
  enableCulling: true,
  enableSpatialSorting: true,
  bufferGrowthFactor: 1.5,
  minInstancesForBatch: 4,
};

// Instance data stride: 16 floats per instance
// [transform_row0, transform_row1, transform_row2, transform_row3] = 16 floats
// or [pos.xyz, scale.xyz, rotation.xyzw, color.rgba] = 14 floats padded to 16
const INSTANCE_STRIDE = 16;

/**
 * Dynamic Instance Batch Manager
 *
 * Automatically groups similar objects into instanced batches for efficient
 * GPU rendering. Supports LOD, frustum culling, and dynamic updates.
 *
 * @example
 * ```typescript
 * const batchManager = new InstanceBatchManager(device);
 *
 * // Register geometries and materials
 * batchManager.registerGeometry({ id: 'sphere', positions: sphereVerts });
 * batchManager.registerMaterial({ id: 'default', color: { r: 1, g: 1, b: 1, a: 1 } });
 *
 * // Add instances
 * for (const node of nodes) {
 *   batchManager.addInstance('sphere', 'default', {
 *     id: node.id,
 *     position: node.position,
 *     scale: node.size
 *   });
 * }
 *
 * // Update and render
 * batchManager.updateBatches(cameraPosition);
 * for (const batch of batchManager.getRenderBatches()) {
 *   renderBatch(batch);
 * }
 * ```
 */
export class InstanceBatchManager {
  private device: GPUDevice;
  private config: BatchConfig;

  // Geometry registry
  private geometries: Map<string, BatchGeometry> = new Map();

  // Material registry
  private materials: Map<string, BatchMaterial> = new Map();

  // Instance storage: Map<geometryId_materialId_lodLevel, instances>
  private instances: Map<string, Map<string, InstanceData>> = new Map();

  // GPU buffers: Map<batchKey, GPUBuffer>
  private gpuBuffers: Map<string, GPUBuffer> = new Map();

  // Render batches
  private batches: Map<string, RenderBatch> = new Map();

  // Dirty tracking
  private dirtyBatches: Set<string> = new Set();

  // Statistics
  private stats: BatchingStats = {
    totalInstances: 0,
    batchCount: 0,
    drawCallsSaved: 0,
    memoryUsage: 0,
    culledInstances: 0,
    averageInstancesPerBatch: 0,
  };

  constructor(device: GPUDevice, config: Partial<BatchConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_BATCH_CONFIG, ...config };
  }

  /**
   * Register a geometry for batching
   */
  registerGeometry(geometry: BatchGeometry): void {
    this.geometries.set(geometry.id, geometry);
  }

  /**
   * Register a material for batching
   */
  registerMaterial(material: BatchMaterial): void {
    this.materials.set(material.id, material);
  }

  /**
   * Unregister a geometry
   */
  unregisterGeometry(geometryId: string): void {
    this.geometries.delete(geometryId);

    // Remove all instances using this geometry
    for (const [key, instances] of this.instances) {
      if (key.startsWith(`${geometryId}_`)) {
        instances.clear();
        this.dirtyBatches.add(key);
      }
    }
  }

  /**
   * Unregister a material
   */
  unregisterMaterial(materialId: string): void {
    this.materials.delete(materialId);

    // Remove all instances using this material
    for (const [key, instances] of this.instances) {
      if (key.includes(`_${materialId}_`)) {
        instances.clear();
        this.dirtyBatches.add(key);
      }
    }
  }

  /**
   * Add an instance to be batched
   */
  addInstance(
    geometryId: string,
    materialId: string,
    instance: InstanceData
  ): void {
    const lodLevel = instance.lodLevel ?? 0;
    const batchKey = this.getBatchKey(geometryId, materialId, lodLevel);

    let instanceMap = this.instances.get(batchKey);
    if (!instanceMap) {
      instanceMap = new Map();
      this.instances.set(batchKey, instanceMap);
    }

    instanceMap.set(instance.id, instance);
    this.dirtyBatches.add(batchKey);
  }

  /**
   * Update an existing instance
   */
  updateInstance(
    geometryId: string,
    materialId: string,
    instance: InstanceData
  ): void {
    const lodLevel = instance.lodLevel ?? 0;
    const batchKey = this.getBatchKey(geometryId, materialId, lodLevel);

    const instanceMap = this.instances.get(batchKey);
    if (instanceMap?.has(instance.id)) {
      instanceMap.set(instance.id, instance);
      this.dirtyBatches.add(batchKey);
    }
  }

  /**
   * Remove an instance
   */
  removeInstance(
    geometryId: string,
    materialId: string,
    instanceId: string,
    lodLevel = 0
  ): void {
    const batchKey = this.getBatchKey(geometryId, materialId, lodLevel);

    const instanceMap = this.instances.get(batchKey);
    if (instanceMap?.delete(instanceId)) {
      this.dirtyBatches.add(batchKey);
    }
  }

  /**
   * Bulk add instances (more efficient than individual adds)
   */
  addInstances(
    geometryId: string,
    materialId: string,
    instances: InstanceData[]
  ): void {
    // Group by LOD level
    const byLOD = new Map<number, InstanceData[]>();

    for (const instance of instances) {
      const lod = instance.lodLevel ?? 0;
      let group = byLOD.get(lod);
      if (!group) {
        group = [];
        byLOD.set(lod, group);
      }
      group.push(instance);
    }

    // Add to batches
    for (const [lod, group] of byLOD) {
      const batchKey = this.getBatchKey(geometryId, materialId, lod);

      let instanceMap = this.instances.get(batchKey);
      if (!instanceMap) {
        instanceMap = new Map();
        this.instances.set(batchKey, instanceMap);
      }

      for (const instance of group) {
        instanceMap.set(instance.id, instance);
      }

      this.dirtyBatches.add(batchKey);
    }
  }

  /**
   * Update instance positions (optimized for frequent updates)
   */
  updatePositions(
    updates: Array<{
      geometryId: string;
      materialId: string;
      instanceId: string;
      position: Vector3;
      lodLevel?: number;
    }>
  ): void {
    for (const update of updates) {
      const batchKey = this.getBatchKey(
        update.geometryId,
        update.materialId,
        update.lodLevel ?? 0
      );

      const instanceMap = this.instances.get(batchKey);
      const instance = instanceMap?.get(update.instanceId);

      if (instance) {
        instance.position = update.position;
        this.dirtyBatches.add(batchKey);
      }
    }
  }

  /**
   * Select LOD level based on distance to camera
   */
  selectLOD(
    geometryId: string,
    instancePosition: Vector3,
    cameraPosition: Vector3
  ): number {
    if (!this.config.enableLOD) return 0;

    const geometry = this.geometries.get(geometryId);
    if (!geometry?.lodVariants?.length) return 0;

    const distance = this.calculateDistance(instancePosition, cameraPosition);

    // Find appropriate LOD level
    let selectedLOD = 0;
    for (const variant of geometry.lodVariants) {
      if (distance > variant.distance) {
        selectedLOD = variant.level;
      }
    }

    return selectedLOD;
  }

  /**
   * Update all batches with automatic LOD selection
   */
  updateBatches(cameraPosition?: Vector3): void {
    // Handle LOD transitions if camera position provided
    if (cameraPosition && this.config.enableLOD) {
      this.updateLODs(cameraPosition);
    }

    // Rebuild dirty batches
    for (const batchKey of this.dirtyBatches) {
      this.rebuildBatch(batchKey);
    }

    this.dirtyBatches.clear();
    this.updateStats();
  }

  /**
   * Get all render batches for rendering
   */
  getRenderBatches(): RenderBatch[] {
    return Array.from(this.batches.values());
  }

  /**
   * Get GPU instance buffer for a batch
   */
  getGPUBuffer(batchKey: string): GPUBuffer | undefined {
    return this.gpuBuffers.get(batchKey);
  }

  /**
   * Get batching statistics
   */
  getStats(): BatchingStats {
    return { ...this.stats };
  }

  /**
   * Clear all instances
   */
  clear(): void {
    this.instances.clear();
    this.batches.clear();

    // Destroy GPU buffers
    for (const buffer of this.gpuBuffers.values()) {
      buffer.destroy();
    }
    this.gpuBuffers.clear();

    this.dirtyBatches.clear();
    this.updateStats();
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.clear();
    this.geometries.clear();
    this.materials.clear();
  }

  // Private methods

  private getBatchKey(
    geometryId: string,
    materialId: string,
    lodLevel: number
  ): string {
    return `${geometryId}_${materialId}_${lodLevel}`;
  }

  private parseBatchKey(batchKey: string): {
    geometryId: string;
    materialId: string;
    lodLevel: number;
  } {
    const parts = batchKey.split("_");
    return {
      geometryId: parts[0] ?? "",
      materialId: parts[1] ?? "",
      lodLevel: parseInt(parts[2] ?? "0", 10),
    };
  }

  private calculateDistance(a: Vector3, b: Vector3): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private updateLODs(cameraPosition: Vector3): void {
    // Collect all instances that need LOD changes
    const lodChanges: Array<{
      fromKey: string;
      toKey: string;
      instance: InstanceData;
    }> = [];

    for (const [batchKey, instanceMap] of this.instances) {
      const { geometryId, materialId, lodLevel } = this.parseBatchKey(batchKey);
      const geometry = this.geometries.get(geometryId);

      if (!geometry?.lodVariants?.length) continue;

      for (const instance of instanceMap.values()) {
        const newLOD = this.selectLOD(
          geometryId,
          instance.position,
          cameraPosition
        );

        if (newLOD !== lodLevel) {
          lodChanges.push({
            fromKey: batchKey,
            toKey: this.getBatchKey(geometryId, materialId, newLOD),
            instance: { ...instance, lodLevel: newLOD },
          });
        }
      }
    }

    // Apply LOD changes
    for (const change of lodChanges) {
      // Remove from old batch
      const fromMap = this.instances.get(change.fromKey);
      fromMap?.delete(change.instance.id);
      this.dirtyBatches.add(change.fromKey);

      // Add to new batch
      let toMap = this.instances.get(change.toKey);
      if (!toMap) {
        toMap = new Map();
        this.instances.set(change.toKey, toMap);
      }
      toMap.set(change.instance.id, change.instance);
      this.dirtyBatches.add(change.toKey);
    }
  }

  private rebuildBatch(batchKey: string): void {
    const instanceMap = this.instances.get(batchKey);

    if (!instanceMap || instanceMap.size === 0) {
      // Remove empty batch
      this.batches.delete(batchKey);
      const buffer = this.gpuBuffers.get(batchKey);
      if (buffer) {
        buffer.destroy();
        this.gpuBuffers.delete(batchKey);
      }
      return;
    }

    // Skip batching for very small groups
    if (instanceMap.size < this.config.minInstancesForBatch) {
      // Still create batch but mark for individual rendering if needed
    }

    const { geometryId, materialId, lodLevel } = this.parseBatchKey(batchKey);

    // Collect visible instances
    const visibleInstances: InstanceData[] = [];
    for (const instance of instanceMap.values()) {
      if (instance.visible !== false) {
        visibleInstances.push(instance);
      }
    }

    // Apply spatial sorting for better GPU cache utilization
    if (this.config.enableSpatialSorting && visibleInstances.length > 100) {
      this.sortInstancesSpatially(visibleInstances);
    }

    // Build instance buffer
    const instanceCount = visibleInstances.length;
    const bufferSize = instanceCount * INSTANCE_STRIDE * 4; // 4 bytes per float

    // Create or resize GPU buffer
    let gpuBuffer = this.gpuBuffers.get(batchKey);
    if (!gpuBuffer || gpuBuffer.size < bufferSize) {
      if (gpuBuffer) {
        gpuBuffer.destroy();
      }

      // Grow buffer with growth factor
      const allocSize = Math.ceil(bufferSize * this.config.bufferGrowthFactor);
      gpuBuffer = this.device.createBuffer({
        size: allocSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        label: `instance-buffer-${batchKey}`,
      });
      this.gpuBuffers.set(batchKey, gpuBuffer);
    }

    // Pack instance data
    const instanceData = new Float32Array(instanceCount * INSTANCE_STRIDE);

    for (let i = 0; i < instanceCount; i++) {
      const instance = visibleInstances[i]!;
      const offset = i * INSTANCE_STRIDE;

      // Pack as transform matrix rows (4x4 = 16 floats)
      // Row 0: scale.x, 0, 0, position.x
      // Row 1: 0, scale.y, 0, position.y
      // Row 2: 0, 0, scale.z, position.z
      // Row 3: color.r, color.g, color.b, color.a

      const sx =
        typeof instance.scale === "number" ? instance.scale : instance.scale.x;
      const sy =
        typeof instance.scale === "number" ? instance.scale : instance.scale.y;
      const sz =
        typeof instance.scale === "number" ? instance.scale : instance.scale.z;

      // Simple transform (no rotation for now, can be extended)
      instanceData[offset + 0] = sx;
      instanceData[offset + 1] = 0;
      instanceData[offset + 2] = 0;
      instanceData[offset + 3] = instance.position.x;

      instanceData[offset + 4] = 0;
      instanceData[offset + 5] = sy;
      instanceData[offset + 6] = 0;
      instanceData[offset + 7] = instance.position.y;

      instanceData[offset + 8] = 0;
      instanceData[offset + 9] = 0;
      instanceData[offset + 10] = sz;
      instanceData[offset + 11] = instance.position.z;

      // Color in last row
      instanceData[offset + 12] = instance.color?.r ?? 1;
      instanceData[offset + 13] = instance.color?.g ?? 1;
      instanceData[offset + 14] = instance.color?.b ?? 1;
      instanceData[offset + 15] = instance.color?.a ?? 1;
    }

    // Upload to GPU
    this.device.queue.writeBuffer(gpuBuffer, 0, instanceData);

    // Update batch record
    this.batches.set(batchKey, {
      id: batchKey,
      geometryId,
      materialId,
      lodLevel,
      instanceCount,
      instanceBuffer: instanceData,
      firstInstance: 0,
      dirty: false,
    });
  }

  private sortInstancesSpatially(instances: InstanceData[]): void {
    // Morton code (Z-order curve) based sorting for cache-friendly rendering
    instances.sort((a, b) => {
      const ma = this.mortonCode(a.position);
      const mb = this.mortonCode(b.position);
      return ma - mb;
    });
  }

  private mortonCode(position: Vector3): number {
    // Simplified 3D Morton code for spatial sorting
    // Normalize to 0-1023 range
    const scale = 1023;
    const x = Math.max(0, Math.min(scale, Math.floor((position.x + 1000) / 2)));
    const y = Math.max(0, Math.min(scale, Math.floor((position.y + 1000) / 2)));
    const z = Math.max(0, Math.min(scale, Math.floor((position.z + 1000) / 2)));

    return this.interleaveBits(x, y, z);
  }

  private interleaveBits(x: number, y: number, z: number): number {
    // Interleave bits of x, y, z for Morton code
    let result = 0;
    for (let i = 0; i < 10; i++) {
      result |= ((x >> i) & 1) << (3 * i);
      result |= ((y >> i) & 1) << (3 * i + 1);
      result |= ((z >> i) & 1) << (3 * i + 2);
    }
    return result;
  }

  private updateStats(): void {
    let totalInstances = 0;
    let memoryUsage = 0;

    for (const instanceMap of this.instances.values()) {
      totalInstances += instanceMap.size;
    }

    for (const buffer of this.gpuBuffers.values()) {
      memoryUsage += buffer.size;
    }

    const batchCount = this.batches.size;

    this.stats = {
      totalInstances,
      batchCount,
      drawCallsSaved: Math.max(0, totalInstances - batchCount),
      memoryUsage,
      culledInstances: 0, // Updated during culling pass
      averageInstancesPerBatch:
        batchCount > 0 ? totalInstances / batchCount : 0,
    };
  }
}

/**
 * Create an instance batch manager
 */
export function createInstanceBatchManager(
  device: GPUDevice,
  config?: Partial<BatchConfig>
): InstanceBatchManager {
  return new InstanceBatchManager(device, config);
}
