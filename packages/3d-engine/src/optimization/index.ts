/**
 * Performance Optimization Module
 *
 * Advanced optimization systems for large-scale 3D visualization.
 * Includes LOD management, Barnes-Hut spatial optimization, and GPU memory management.
 *
 * @module @neurectomy/3d-engine/optimization
 * @agents @VELOCITY @AXIOM
 * @phase Phase 3 - Dimensional Forge
 */

// LOD System
export { LODManager, QUALITY_PRESETS } from "./lod-manager";

export type {
  LODLevel,
  LODEntry,
  LODManagerConfig,
  LODStatistics,
  QualityPreset,
} from "./lod-manager";

// Barnes-Hut Spatial Tree
export {
  BarnesHutTree,
  createBodiesFromNodes,
  applyPositionsToNodes,
} from "./barnes-hut";

export type {
  BarnesHutNode,
  BarnesHutConfig,
  Body,
  ForceResult,
} from "./barnes-hut";
