/**
 * WebGPU Compute Module - GPU-Accelerated Algorithms
 *
 * Exports compute shader infrastructure for parallel algorithms:
 * - Force-directed graph layout (100K+ nodes at 60 FPS)
 * - Future: Barnes-Hut octree, DBSCAN clustering, k-means
 *
 * @module @neurectomy/3d-engine/webgpu/compute
 * @agents @VELOCITY @CORE
 */

export {
  GPUForceLayout,
  createGPUForceLayout,
  type GPUForceLayoutConfig,
  type GPUForceLayoutStats,
  type GPUNode,
  type GPUEdge,
} from "./gpu-force-layout";
