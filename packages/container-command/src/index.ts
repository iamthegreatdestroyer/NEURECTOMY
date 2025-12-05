/**
 * NEURECTOMY Container Command Package
 *
 * @module @neurectomy/container-command
 *
 * Phase 4: Container Command Foundation & Advanced Runtimes
 *
 * This package provides comprehensive container orchestration capabilities:
 *
 * Docker Module (@FLUX @SYNAPSE):
 * - Docker Engine API client with full lifecycle management
 * - Docker Compose orchestration with dependency ordering
 * - Streaming logs and real-time stats
 *
 * Kubernetes Module (@FLUX @APEX, @FLUX @FORGE):
 * - Kubernetes API client with watch support and auto-reconnection
 * - Helm chart generator from agent definitions
 * - Full resource management (Pods, Deployments, Services, HPAs)
 *
 * Image Module (@FLUX @FORGE):
 * - Multi-stage image build pipeline with layer caching
 * - Security scanning integration (Trivy)
 * - Registry management and push operations
 *
 * Firecracker Module (@CORE @VELOCITY):
 * - MicroVM management for sub-second agent isolation
 * - Snapshot/restore for instant VM cloning
 * - MMDS metadata service for agent configuration
 *
 * WebAssembly Module (@CORE @APEX):
 * - Wasmtime runtime for lightweight agent execution
 * - WASI support for system interface access
 * - Component model for module composition
 * - Secure sandboxed execution
 *
 * Types:
 * - Comprehensive type definitions for all container runtimes
 * - Zod validation schemas for configuration
 */

// Types - export all core types from types.ts
export * from "./types";

// Docker
export * from "./docker";

// Kubernetes
export * from "./kubernetes";

// Image Pipeline
export * from "./image";

// Firecracker MicroVMs - exclude FirecrackerConfig (already in types)
export {
  FirecrackerManager,
  FirecrackerApiError,
  type FirecrackerConfig,
} from "./firecracker";

// WebAssembly Runtime - export with type annotations where needed
export {
  WasmtimeManager,
  WasiBuilder,
  type WasmModuleConfig,
  type WasmImport,
  type WasmInstance,
  type WasmModule,
  type WasmImportInfo,
  type WasmInstanceState,
} from "./wasm";

// Service Mesh
export * from "./service-mesh";

// 3D Topology Visualization - export with correct names
export {
  TopologyManager,
  generateSampleTopology,
  type TopologyNode,
  type TopologyEdge,
  type TopologyGraph,
  type Vector3,
  type NodeMesh,
  type EdgeMesh,
  type HealthStatus as TopologyHealthStatus,
} from "./topology";
