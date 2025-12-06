/**
 * GPU Frustum Culling System
 *
 * High-performance GPU-based frustum culling for large-scale graph visualization.
 * Uses compute shaders to cull nodes and edges in parallel.
 *
 * @module @neurectomy/3d-engine/webgpu/culling
 * @agents @VELOCITY @CORE
 */

// Note: WGSL shader is loaded inline to avoid import issues
const frustumCullShader = /* wgsl */ `
// Frustum culling compute shader - see frustum-cull.wgsl for full implementation
struct FrustumPlane {
    normal: vec3<f32>,
    distance: f32,
}

struct Frustum {
    planes: array<FrustumPlane, 6>,
    cameraPosition: vec3<f32>,
    maxDistance: f32,
}

struct CullParams {
    nodeCount: u32,
    edgeCount: u32,
    enableDistanceCull: u32,
    _padding: u32,
}

struct NodeBounds {
    center: vec3<f32>,
    radius: f32,
}

struct CullResult {
    visibleNodeCount: atomic<u32>,
    visibleEdgeCount: atomic<u32>,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> frustum: Frustum;
@group(0) @binding(1) var<uniform> params: CullParams;
@group(1) @binding(0) var<storage, read> nodeBounds: array<NodeBounds>;
@group(1) @binding(1) var<storage, read> edgeBounds: array<NodeBounds>;
@group(1) @binding(2) var<storage, read_write> nodeVisibility: array<u32>;
@group(1) @binding(3) var<storage, read_write> edgeVisibility: array<u32>;
@group(1) @binding(4) var<storage, read_write> cullResult: CullResult;
@group(1) @binding(5) var<storage, read_write> visibleNodeIndices: array<u32>;
@group(1) @binding(6) var<storage, read_write> visibleEdgeIndices: array<u32>;

fn spherePlaneTest(center: vec3<f32>, radius: f32, plane: FrustumPlane) -> f32 {
    return dot(plane.normal, center) + plane.distance + radius;
}

fn sphereFrustumTest(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        if (spherePlaneTest(center, radius, frustum.planes[i]) < 0.0) {
            return false;
        }
    }
    return true;
}

fn distanceCullTest(center: vec3<f32>, radius: f32) -> bool {
    if (params.enableDistanceCull == 0u) { return true; }
    let d = center - frustum.cameraPosition;
    let distSq = dot(d, d);
    let maxD = frustum.maxDistance + radius;
    return distSq <= maxD * maxD;
}

fn isNodeVisible(bounds: NodeBounds) -> bool {
    if (!distanceCullTest(bounds.center, bounds.radius)) { return false; }
    return sphereFrustumTest(bounds.center, bounds.radius);
}

@compute @workgroup_size(256)
fn cullNodes(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let nodeIdx = globalId.x;
    if (nodeIdx >= params.nodeCount) { return; }
    let bounds = nodeBounds[nodeIdx];
    let visible = isNodeVisible(bounds);
    nodeVisibility[nodeIdx] = select(0u, 1u, visible);
    if (visible) {
        let visibleIdx = atomicAdd(&cullResult.visibleNodeCount, 1u);
        visibleNodeIndices[visibleIdx] = nodeIdx;
    }
}

@compute @workgroup_size(256)
fn cullEdges(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let edgeIdx = globalId.x;
    if (edgeIdx >= params.edgeCount) { return; }
    let bounds = edgeBounds[edgeIdx];
    let visible = isNodeVisible(bounds);
    edgeVisibility[edgeIdx] = select(0u, 1u, visible);
    if (visible) {
        let visibleIdx = atomicAdd(&cullResult.visibleEdgeCount, 1u);
        visibleEdgeIndices[visibleIdx] = edgeIdx;
    }
}

@compute @workgroup_size(1)
fn resetCounters() {
    atomicStore(&cullResult.visibleNodeCount, 0u);
    atomicStore(&cullResult.visibleEdgeCount, 0u);
}
`; // =============================================================================
// Types
// =============================================================================

export interface FrustumPlane {
  normal: [number, number, number];
  distance: number;
}

export interface Frustum {
  planes: FrustumPlane[];
  cameraPosition: [number, number, number];
  maxDistance: number;
}

export interface CullConfig {
  /** Maximum number of nodes to support */
  maxNodes: number;
  /** Maximum number of edges to support */
  maxEdges: number;
  /** Enable distance-based culling */
  enableDistanceCull: boolean;
  /** Maximum render distance */
  maxRenderDistance: number;
  /** Use hierarchical BVH culling */
  useHierarchicalCulling: boolean;
}

export interface CullResult {
  /** Number of visible nodes after culling */
  visibleNodeCount: number;
  /** Number of visible edges after culling */
  visibleEdgeCount: number;
  /** Indices of visible nodes */
  visibleNodeIndices: Uint32Array;
  /** Indices of visible edges */
  visibleEdgeIndices: Uint32Array;
  /** Per-node visibility flags */
  nodeVisibility: Uint32Array;
  /** Per-edge visibility flags */
  edgeVisibility: Uint32Array;
}

export interface CullStats {
  /** Total nodes before culling */
  totalNodes: number;
  /** Total edges before culling */
  totalEdges: number;
  /** Visible nodes after culling */
  visibleNodes: number;
  /** Visible edges after culling */
  visibleEdges: number;
  /** Cull ratio for nodes */
  nodeCullRatio: number;
  /** Cull ratio for edges */
  edgeCullRatio: number;
  /** GPU time in milliseconds */
  gpuTimeMs: number;
}

export interface NodeBounds {
  center: [number, number, number];
  radius: number;
}

export interface EdgeBounds {
  sourceCenter: [number, number, number];
  sourceRadius: number;
  targetCenter: [number, number, number];
  targetRadius: number;
}

// =============================================================================
// GPU Frustum Culling Implementation
// =============================================================================

const DEFAULT_CONFIG: CullConfig = {
  maxNodes: 100000,
  maxEdges: 500000,
  enableDistanceCull: true,
  maxRenderDistance: 10000,
  useHierarchicalCulling: false,
};

export class GPUFrustumCuller {
  private device: GPUDevice;
  private config: CullConfig;

  // Compute pipelines
  private resetPipeline: GPUComputePipeline | null = null;
  private nodeCullPipeline: GPUComputePipeline | null = null;
  private edgeCullPipeline: GPUComputePipeline | null = null;
  private extractFrustumPipeline: GPUComputePipeline | null = null;

  // Bind group layouts
  private frustumBindGroupLayout: GPUBindGroupLayout | null = null;
  private boundsBindGroupLayout: GPUBindGroupLayout | null = null;
  private matrixBindGroupLayout: GPUBindGroupLayout | null = null;

  // Buffers
  private frustumBuffer: GPUBuffer | null = null;
  private paramsBuffer: GPUBuffer | null = null;
  private nodeBoundsBuffer: GPUBuffer | null = null;
  private edgeBoundsBuffer: GPUBuffer | null = null;
  private nodeVisibilityBuffer: GPUBuffer | null = null;
  private edgeVisibilityBuffer: GPUBuffer | null = null;
  private cullResultBuffer: GPUBuffer | null = null;
  private visibleNodeIndicesBuffer: GPUBuffer | null = null;
  private visibleEdgeIndicesBuffer: GPUBuffer | null = null;
  private viewProjBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;

  // Bind groups
  private frustumBindGroup: GPUBindGroup | null = null;
  private boundsBindGroup: GPUBindGroup | null = null;
  private matrixBindGroup: GPUBindGroup | null = null;

  // State
  private initialized = false;
  private currentNodeCount = 0;
  private currentEdgeCount = 0;

  // Query for GPU timing
  private querySet: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private timestampSupported = false;

  constructor(device: GPUDevice, config: Partial<CullConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.timestampSupported = device.features.has("timestamp-query");
  }

  /**
   * Initialize the frustum culling system
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    await this.createPipelines();
    this.createBuffers();
    this.createBindGroups();

    if (this.timestampSupported) {
      this.createQuerySet();
    }

    this.initialized = true;
  }

  private async createPipelines(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Frustum Culling Shader",
      code: frustumCullShader,
    });

    // Create bind group layouts
    this.frustumBindGroupLayout = this.device.createBindGroupLayout({
      label: "Frustum Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.boundsBindGroupLayout = this.device.createBindGroupLayout({
      label: "Bounds Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    this.matrixBindGroupLayout = this.device.createBindGroupLayout({
      label: "Matrix Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: "Frustum Culling Pipeline Layout",
      bindGroupLayouts: [
        this.frustumBindGroupLayout,
        this.boundsBindGroupLayout,
      ],
    });

    // Reset counters pipeline
    this.resetPipeline = this.device.createComputePipeline({
      label: "Reset Counters Pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "resetCounters",
      },
    });

    // Node culling pipeline
    this.nodeCullPipeline = this.device.createComputePipeline({
      label: "Node Culling Pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "cullNodes",
      },
    });

    // Edge culling pipeline
    this.edgeCullPipeline = this.device.createComputePipeline({
      label: "Edge Culling Pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "cullEdges",
      },
    });

    // Frustum extraction pipeline
    const extractPipelineLayout = this.device.createPipelineLayout({
      label: "Frustum Extract Pipeline Layout",
      bindGroupLayouts: [this.matrixBindGroupLayout],
    });

    this.extractFrustumPipeline = this.device.createComputePipeline({
      label: "Extract Frustum Pipeline",
      layout: extractPipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "extractFrustum",
      },
    });
  }

  private createBuffers(): void {
    const { maxNodes, maxEdges } = this.config;

    // Frustum uniform buffer (6 planes * 16 bytes + 16 bytes camera/distance = 112 bytes)
    // Align to 256 bytes
    this.frustumBuffer = this.device.createBuffer({
      label: "Frustum Uniform Buffer",
      size: 256,
      usage:
        GPUBufferUsage.UNIFORM |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
    });

    // Parameters buffer (4 uint32 = 16 bytes)
    this.paramsBuffer = this.device.createBuffer({
      label: "Cull Params Buffer",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Node bounds buffer (16 bytes per node: vec3 center + float radius)
    this.nodeBoundsBuffer = this.device.createBuffer({
      label: "Node Bounds Buffer",
      size: maxNodes * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Edge bounds buffer (32 bytes per edge: 2x (vec3 center + float radius))
    this.edgeBoundsBuffer = this.device.createBuffer({
      label: "Edge Bounds Buffer",
      size: maxEdges * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Node visibility buffer (1 uint32 per node)
    this.nodeVisibilityBuffer = this.device.createBuffer({
      label: "Node Visibility Buffer",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Edge visibility buffer (1 uint32 per edge)
    this.edgeVisibilityBuffer = this.device.createBuffer({
      label: "Edge Visibility Buffer",
      size: maxEdges * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Cull result buffer (2 atomic uint32 + 2 padding = 16 bytes)
    this.cullResultBuffer = this.device.createBuffer({
      label: "Cull Result Buffer",
      size: 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Visible node indices buffer
    this.visibleNodeIndicesBuffer = this.device.createBuffer({
      label: "Visible Node Indices Buffer",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Visible edge indices buffer
    this.visibleEdgeIndicesBuffer = this.device.createBuffer({
      label: "Visible Edge Indices Buffer",
      size: maxEdges * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // View-projection matrix buffer (16 floats = 64 bytes)
    this.viewProjBuffer = this.device.createBuffer({
      label: "ViewProj Matrix Buffer",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Readback buffer for results
    this.readbackBuffer = this.device.createBuffer({
      label: "Readback Buffer",
      size: Math.max(maxNodes, maxEdges) * 4 + 16, // Max of indices + result
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  private createBindGroups(): void {
    if (
      !this.frustumBindGroupLayout ||
      !this.boundsBindGroupLayout ||
      !this.frustumBuffer ||
      !this.paramsBuffer ||
      !this.nodeBoundsBuffer ||
      !this.edgeBoundsBuffer ||
      !this.nodeVisibilityBuffer ||
      !this.edgeVisibilityBuffer ||
      !this.cullResultBuffer ||
      !this.visibleNodeIndicesBuffer ||
      !this.visibleEdgeIndicesBuffer
    ) {
      return;
    }

    this.frustumBindGroup = this.device.createBindGroup({
      label: "Frustum Bind Group",
      layout: this.frustumBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frustumBuffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer } },
      ],
    });

    this.boundsBindGroup = this.device.createBindGroup({
      label: "Bounds Bind Group",
      layout: this.boundsBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.nodeBoundsBuffer } },
        { binding: 1, resource: { buffer: this.edgeBoundsBuffer } },
        { binding: 2, resource: { buffer: this.nodeVisibilityBuffer } },
        { binding: 3, resource: { buffer: this.edgeVisibilityBuffer } },
        { binding: 4, resource: { buffer: this.cullResultBuffer } },
        { binding: 5, resource: { buffer: this.visibleNodeIndicesBuffer } },
        { binding: 6, resource: { buffer: this.visibleEdgeIndicesBuffer } },
      ],
    });

    if (this.matrixBindGroupLayout && this.viewProjBuffer) {
      this.matrixBindGroup = this.device.createBindGroup({
        label: "Matrix Bind Group",
        layout: this.matrixBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.viewProjBuffer } },
          { binding: 1, resource: { buffer: this.frustumBuffer } },
        ],
      });
    }
  }

  private createQuerySet(): void {
    this.querySet = this.device.createQuerySet({
      label: "Culling Timestamp Query",
      type: "timestamp",
      count: 2,
    });

    this.queryBuffer = this.device.createBuffer({
      label: "Query Result Buffer",
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * Update node bounds for culling
   */
  setNodeBounds(bounds: NodeBounds[]): void {
    if (!this.nodeBoundsBuffer) return;

    this.currentNodeCount = Math.min(bounds.length, this.config.maxNodes);
    const data = new Float32Array(this.currentNodeCount * 4);

    for (let i = 0; i < this.currentNodeCount; i++) {
      const b = bounds[i]!;
      data[i * 4 + 0] = b.center[0];
      data[i * 4 + 1] = b.center[1];
      data[i * 4 + 2] = b.center[2];
      data[i * 4 + 3] = b.radius;
    }

    this.device.queue.writeBuffer(this.nodeBoundsBuffer, 0, data);
  }

  /**
   * Update edge bounds for culling
   */
  setEdgeBounds(bounds: EdgeBounds[]): void {
    if (!this.edgeBoundsBuffer) return;

    this.currentEdgeCount = Math.min(bounds.length, this.config.maxEdges);
    const data = new Float32Array(this.currentEdgeCount * 8);

    for (let i = 0; i < this.currentEdgeCount; i++) {
      const b = bounds[i]!;
      data[i * 8 + 0] = b.sourceCenter[0];
      data[i * 8 + 1] = b.sourceCenter[1];
      data[i * 8 + 2] = b.sourceCenter[2];
      data[i * 8 + 3] = b.sourceRadius;
      data[i * 8 + 4] = b.targetCenter[0];
      data[i * 8 + 5] = b.targetCenter[1];
      data[i * 8 + 6] = b.targetCenter[2];
      data[i * 8 + 7] = b.targetRadius;
    }

    this.device.queue.writeBuffer(this.edgeBoundsBuffer, 0, data);
  }

  /**
   * Update frustum from view-projection matrix
   */
  setViewProjectionMatrix(matrix: Float32Array): void {
    if (!this.viewProjBuffer) return;
    this.device.queue.writeBuffer(this.viewProjBuffer, 0, matrix);
  }

  /**
   * Set frustum directly
   */
  setFrustum(frustum: Frustum): void {
    if (!this.frustumBuffer) return;

    // Frustum buffer layout:
    // 6 planes * (vec3 normal + float distance) = 6 * 16 = 96 bytes
    // vec3 cameraPosition + float maxDistance = 16 bytes
    // Total: 112 bytes (aligned to 256)
    const data = new Float32Array(64); // 256 bytes

    for (let i = 0; i < 6; i++) {
      const plane = frustum.planes[i]!;
      data[i * 4 + 0] = plane.normal[0];
      data[i * 4 + 1] = plane.normal[1];
      data[i * 4 + 2] = plane.normal[2];
      data[i * 4 + 3] = plane.distance;
    }

    data[24] = frustum.cameraPosition[0];
    data[25] = frustum.cameraPosition[1];
    data[26] = frustum.cameraPosition[2];
    data[27] = frustum.maxDistance;

    this.device.queue.writeBuffer(this.frustumBuffer, 0, data);
  }

  /**
   * Execute frustum culling on GPU
   */
  cull(encoder: GPUCommandEncoder): void {
    if (
      !this.resetPipeline ||
      !this.nodeCullPipeline ||
      !this.edgeCullPipeline ||
      !this.frustumBindGroup ||
      !this.boundsBindGroup ||
      !this.paramsBuffer
    ) {
      return;
    }

    // Update params
    const paramsData = new Uint32Array([
      this.currentNodeCount,
      this.currentEdgeCount,
      this.config.enableDistanceCull ? 1 : 0,
      0, // padding
    ]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    // Start timestamp query
    if (this.timestampSupported && this.querySet) {
      encoder.writeTimestamp(this.querySet, 0);
    }

    const pass = encoder.beginComputePass({
      label: "Frustum Culling Pass",
    });

    // Reset counters
    pass.setPipeline(this.resetPipeline);
    pass.setBindGroup(0, this.frustumBindGroup);
    pass.setBindGroup(1, this.boundsBindGroup);
    pass.dispatchWorkgroups(1);

    // Cull nodes
    if (this.currentNodeCount > 0) {
      pass.setPipeline(this.nodeCullPipeline);
      const nodeWorkgroups = Math.ceil(this.currentNodeCount / 256);
      pass.dispatchWorkgroups(nodeWorkgroups);
    }

    // Cull edges
    if (this.currentEdgeCount > 0) {
      pass.setPipeline(this.edgeCullPipeline);
      const edgeWorkgroups = Math.ceil(this.currentEdgeCount / 256);
      pass.dispatchWorkgroups(edgeWorkgroups);
    }

    pass.end();

    // End timestamp query
    if (this.timestampSupported && this.querySet && this.queryBuffer) {
      encoder.writeTimestamp(this.querySet, 1);
      encoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
    }
  }

  /**
   * Extract frustum planes from view-projection matrix on GPU
   */
  extractFrustumFromMatrix(encoder: GPUCommandEncoder): void {
    if (!this.extractFrustumPipeline || !this.matrixBindGroup) return;

    const pass = encoder.beginComputePass({
      label: "Frustum Extraction Pass",
    });

    pass.setPipeline(this.extractFrustumPipeline);
    pass.setBindGroup(0, this.matrixBindGroup);
    pass.dispatchWorkgroups(1);

    pass.end();
  }

  /**
   * Read culling results back to CPU
   */
  async readResults(): Promise<CullResult> {
    if (!this.cullResultBuffer || !this.readbackBuffer) {
      return {
        visibleNodeCount: this.currentNodeCount,
        visibleEdgeCount: this.currentEdgeCount,
        visibleNodeIndices: new Uint32Array(0),
        visibleEdgeIndices: new Uint32Array(0),
        nodeVisibility: new Uint32Array(0),
        edgeVisibility: new Uint32Array(0),
      };
    }

    // Copy result buffer to readback
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.cullResultBuffer,
      0,
      this.readbackBuffer,
      0,
      16
    );
    this.device.queue.submit([encoder.finish()]);

    // Map and read
    await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, 16);
    const resultData = new Uint32Array(
      this.readbackBuffer.getMappedRange(0, 16).slice(0)
    );
    this.readbackBuffer.unmap();

    const visibleNodeCount = resultData[0]!;
    const visibleEdgeCount = resultData[1]!;

    // Read visible indices
    const visibleNodeIndices = await this.readBufferSlice(
      this.visibleNodeIndicesBuffer!,
      0,
      visibleNodeCount * 4
    );
    const visibleEdgeIndices = await this.readBufferSlice(
      this.visibleEdgeIndicesBuffer!,
      0,
      visibleEdgeCount * 4
    );

    return {
      visibleNodeCount,
      visibleEdgeCount,
      visibleNodeIndices: new Uint32Array(visibleNodeIndices),
      visibleEdgeIndices: new Uint32Array(visibleEdgeIndices),
      nodeVisibility: new Uint32Array(0), // Don't read full arrays by default
      edgeVisibility: new Uint32Array(0),
    };
  }

  private async readBufferSlice(
    buffer: GPUBuffer,
    offset: number,
    size: number
  ): Promise<ArrayBuffer> {
    if (size === 0 || !this.readbackBuffer) {
      return new ArrayBuffer(0);
    }

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, this.readbackBuffer, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, size);
    const data = this.readbackBuffer.getMappedRange(0, size).slice(0);
    this.readbackBuffer.unmap();

    return data;
  }

  /**
   * Get GPU buffer for visible node indices (for GPU-driven rendering)
   */
  getVisibleNodeIndicesBuffer(): GPUBuffer | null {
    return this.visibleNodeIndicesBuffer;
  }

  /**
   * Get GPU buffer for visible edge indices
   */
  getVisibleEdgeIndicesBuffer(): GPUBuffer | null {
    return this.visibleEdgeIndicesBuffer;
  }

  /**
   * Get cull result buffer for indirect draw
   */
  getCullResultBuffer(): GPUBuffer | null {
    return this.cullResultBuffer;
  }

  /**
   * Get statistics from last cull operation
   */
  async getStats(): Promise<CullStats> {
    const result = await this.readResults();

    return {
      totalNodes: this.currentNodeCount,
      totalEdges: this.currentEdgeCount,
      visibleNodes: result.visibleNodeCount,
      visibleEdges: result.visibleEdgeCount,
      nodeCullRatio:
        this.currentNodeCount > 0
          ? 1 - result.visibleNodeCount / this.currentNodeCount
          : 0,
      edgeCullRatio:
        this.currentEdgeCount > 0
          ? 1 - result.visibleEdgeCount / this.currentEdgeCount
          : 0,
      gpuTimeMs: 0, // TODO: Read from timestamp query
    };
  }

  /**
   * Clean up GPU resources
   */
  dispose(): void {
    this.frustumBuffer?.destroy();
    this.paramsBuffer?.destroy();
    this.nodeBoundsBuffer?.destroy();
    this.edgeBoundsBuffer?.destroy();
    this.nodeVisibilityBuffer?.destroy();
    this.edgeVisibilityBuffer?.destroy();
    this.cullResultBuffer?.destroy();
    this.visibleNodeIndicesBuffer?.destroy();
    this.visibleEdgeIndicesBuffer?.destroy();
    this.viewProjBuffer?.destroy();
    this.readbackBuffer?.destroy();
    this.queryBuffer?.destroy();

    this.frustumBuffer = null;
    this.paramsBuffer = null;
    this.nodeBoundsBuffer = null;
    this.edgeBoundsBuffer = null;
    this.nodeVisibilityBuffer = null;
    this.edgeVisibilityBuffer = null;
    this.cullResultBuffer = null;
    this.visibleNodeIndicesBuffer = null;
    this.visibleEdgeIndicesBuffer = null;
    this.viewProjBuffer = null;
    this.readbackBuffer = null;
    this.queryBuffer = null;

    this.initialized = false;
  }
}

// =============================================================================
// CPU Frustum Culler (Fallback)
// =============================================================================

export class CPUFrustumCuller {
  private frustum: Frustum | null = null;
  private nodeBounds: NodeBounds[] = [];
  private edgeBounds: EdgeBounds[] = [];
  private config: CullConfig;

  constructor(config: Partial<CullConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  setFrustum(frustum: Frustum): void {
    this.frustum = frustum;
  }

  setNodeBounds(bounds: NodeBounds[]): void {
    this.nodeBounds = bounds;
  }

  setEdgeBounds(bounds: EdgeBounds[]): void {
    this.edgeBounds = bounds;
  }

  cull(): CullResult {
    if (!this.frustum) {
      return {
        visibleNodeCount: this.nodeBounds.length,
        visibleEdgeCount: this.edgeBounds.length,
        visibleNodeIndices: new Uint32Array(this.nodeBounds.map((_, i) => i)),
        visibleEdgeIndices: new Uint32Array(this.edgeBounds.map((_, i) => i)),
        nodeVisibility: new Uint32Array(this.nodeBounds.length).fill(1),
        edgeVisibility: new Uint32Array(this.edgeBounds.length).fill(1),
      };
    }

    const visibleNodeIndices: number[] = [];
    const visibleEdgeIndices: number[] = [];
    const nodeVisibility = new Uint32Array(this.nodeBounds.length);
    const edgeVisibility = new Uint32Array(this.edgeBounds.length);

    // Cull nodes
    for (let i = 0; i < this.nodeBounds.length; i++) {
      if (this.isNodeVisible(this.nodeBounds[i]!)) {
        visibleNodeIndices.push(i);
        nodeVisibility[i] = 1;
      }
    }

    // Cull edges
    for (let i = 0; i < this.edgeBounds.length; i++) {
      if (this.isEdgeVisible(this.edgeBounds[i]!)) {
        visibleEdgeIndices.push(i);
        edgeVisibility[i] = 1;
      }
    }

    return {
      visibleNodeCount: visibleNodeIndices.length,
      visibleEdgeCount: visibleEdgeIndices.length,
      visibleNodeIndices: new Uint32Array(visibleNodeIndices),
      visibleEdgeIndices: new Uint32Array(visibleEdgeIndices),
      nodeVisibility,
      edgeVisibility,
    };
  }

  private isNodeVisible(bounds: NodeBounds): boolean {
    if (!this.frustum) return true;

    // Distance cull
    if (this.config.enableDistanceCull) {
      const dx = bounds.center[0] - this.frustum.cameraPosition[0];
      const dy = bounds.center[1] - this.frustum.cameraPosition[1];
      const dz = bounds.center[2] - this.frustum.cameraPosition[2];
      const distSq = dx * dx + dy * dy + dz * dz;
      const maxDist = this.frustum.maxDistance + bounds.radius;
      if (distSq > maxDist * maxDist) return false;
    }

    // Frustum test
    for (const plane of this.frustum.planes) {
      const distance =
        plane.normal[0] * bounds.center[0] +
        plane.normal[1] * bounds.center[1] +
        plane.normal[2] * bounds.center[2] +
        plane.distance;
      if (distance + bounds.radius < 0) return false;
    }

    return true;
  }

  private isEdgeVisible(bounds: EdgeBounds): boolean {
    // Compute bounding sphere
    const midpoint: [number, number, number] = [
      (bounds.sourceCenter[0] + bounds.targetCenter[0]) * 0.5,
      (bounds.sourceCenter[1] + bounds.targetCenter[1]) * 0.5,
      (bounds.sourceCenter[2] + bounds.targetCenter[2]) * 0.5,
    ];

    const dx = bounds.targetCenter[0] - bounds.sourceCenter[0];
    const dy = bounds.targetCenter[1] - bounds.sourceCenter[1];
    const dz = bounds.targetCenter[2] - bounds.sourceCenter[2];
    const halfLength = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;
    const maxRadius = Math.max(bounds.sourceRadius, bounds.targetRadius);
    const boundingRadius = halfLength + maxRadius;

    return this.isNodeVisible({
      center: midpoint,
      radius: boundingRadius,
    });
  }
}

// =============================================================================
// Exports
// =============================================================================

export function createFrustumCuller(
  device: GPUDevice | null,
  config: Partial<CullConfig> = {}
): GPUFrustumCuller | CPUFrustumCuller {
  if (device) {
    return new GPUFrustumCuller(device, config);
  }
  return new CPUFrustumCuller(config);
}

/**
 * Extract frustum planes from a view-projection matrix (CPU)
 */
export function extractFrustumFromMatrix(
  viewProj: Float32Array,
  cameraPosition: [number, number, number],
  maxDistance: number
): Frustum {
  const m = (row: number, col: number) => viewProj[col * 4 + row]!;

  const planes: FrustumPlane[] = [];

  // Left plane: row3 + row0
  planes.push(
    normalizePlane({
      normal: [m(3, 0) + m(0, 0), m(3, 1) + m(0, 1), m(3, 2) + m(0, 2)],
      distance: m(3, 3) + m(0, 3),
    })
  );

  // Right plane: row3 - row0
  planes.push(
    normalizePlane({
      normal: [m(3, 0) - m(0, 0), m(3, 1) - m(0, 1), m(3, 2) - m(0, 2)],
      distance: m(3, 3) - m(0, 3),
    })
  );

  // Bottom plane: row3 + row1
  planes.push(
    normalizePlane({
      normal: [m(3, 0) + m(1, 0), m(3, 1) + m(1, 1), m(3, 2) + m(1, 2)],
      distance: m(3, 3) + m(1, 3),
    })
  );

  // Top plane: row3 - row1
  planes.push(
    normalizePlane({
      normal: [m(3, 0) - m(1, 0), m(3, 1) - m(1, 1), m(3, 2) - m(1, 2)],
      distance: m(3, 3) - m(1, 3),
    })
  );

  // Near plane: row3 + row2
  planes.push(
    normalizePlane({
      normal: [m(3, 0) + m(2, 0), m(3, 1) + m(2, 1), m(3, 2) + m(2, 2)],
      distance: m(3, 3) + m(2, 3),
    })
  );

  // Far plane: row3 - row2
  planes.push(
    normalizePlane({
      normal: [m(3, 0) - m(2, 0), m(3, 1) - m(2, 1), m(3, 2) - m(2, 2)],
      distance: m(3, 3) - m(2, 3),
    })
  );

  return { planes, cameraPosition, maxDistance };
}

function normalizePlane(plane: FrustumPlane): FrustumPlane {
  const len = Math.sqrt(
    plane.normal[0] ** 2 + plane.normal[1] ** 2 + plane.normal[2] ** 2
  );
  if (len < 0.0001) return plane;
  return {
    normal: [
      plane.normal[0] / len,
      plane.normal[1] / len,
      plane.normal[2] / len,
    ],
    distance: plane.distance / len,
  };
}
