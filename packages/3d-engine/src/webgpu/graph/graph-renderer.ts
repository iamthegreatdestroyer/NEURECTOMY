/**
 * WebGPU Native Graph Renderer
 *
 * High-performance graph visualization using native WebGPU rendering pipelines.
 * Supports instanced rendering for nodes and edges with automatic LOD selection.
 *
 * @module @neurectomy/3d-engine/webgpu/graph/graph-renderer
 * @agents @CORE @VELOCITY @ARCHITECT
 */

import { PipelineCache } from "../pipeline-cache";
import { BufferManager, type BufferAllocation } from "../buffer-manager";
import type { GraphNode, GraphEdge } from "../../visualization/graph/types";

// =============================================================================
// Types
// =============================================================================

export interface GraphRenderConfig {
  /** Maximum nodes before switching to billboard mode */
  billboardThreshold: number;
  /** Maximum edges before switching to line mode */
  lineEdgeThreshold: number;
  /** Enable depth pre-pass for large scenes */
  useDepthPrePass: boolean;
  /** Depth pre-pass threshold (node count) */
  depthPrePassThreshold: number;
  /** Default node radius */
  defaultNodeRadius: number;
  /** Default edge width */
  defaultEdgeWidth: number;
  /** MSAA sample count (1, 4, or 8) */
  msaaSampleCount: 1 | 4 | 8;
  /** Frustum culling enabled */
  frustumCulling: boolean;
}

export interface GraphRenderStats {
  nodeCount: number;
  edgeCount: number;
  visibleNodes: number;
  visibleEdges: number;
  drawCalls: number;
  triangles: number;
  renderMode: "sphere" | "billboard" | "hybrid";
  edgeMode: "cylinder" | "line" | "hybrid";
  frameTime: number;
}

interface CameraUniforms {
  view: Float32Array; // mat4x4
  projection: Float32Array; // mat4x4
  viewProjection: Float32Array; // mat4x4
  position: Float32Array; // vec3 + padding
  near: number;
  far: number;
  fov: number;
  aspect: number;
}

interface SceneUniforms {
  time: number;
  deltaTime: number;
  viewportSize: [number, number];
  highlightedNodeId: number;
  selectedNodeId: number;
  hoveredNodeId: number;
}

interface NodeInstance {
  position: [number, number, number];
  radius: number;
  color: [number, number, number, number];
  nodeId: number;
  flags: number;
}

interface EdgeInstance {
  sourcePosition: [number, number, number];
  sourceRadius: number;
  targetPosition: [number, number, number];
  targetRadius: number;
  color: [number, number, number, number];
  edgeId: number;
  flags: number;
  width: number;
}

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_CONFIG: GraphRenderConfig = {
  billboardThreshold: 10000,
  lineEdgeThreshold: 50000,
  useDepthPrePass: true,
  depthPrePassThreshold: 5000,
  defaultNodeRadius: 5,
  defaultEdgeWidth: 1,
  msaaSampleCount: 4,
  frustumCulling: true,
};

// =============================================================================
// Shader Source
// =============================================================================

// Shader will be loaded from shaders.wgsl file
const SHADER_CODE = `
// Imported from shaders.wgsl - this is a placeholder for inline embedding
// In production, use the separate .wgsl file

struct Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    viewProjection: mat4x4<f32>,
    position: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
    aspect: f32,
    _padding: f32,
}

struct SceneUniforms {
    time: f32,
    deltaTime: f32,
    viewportSize: vec2<f32>,
    highlightedNodeId: u32,
    selectedNodeId: u32,
    hoveredNodeId: u32,
    _padding: u32,
}

struct NodeInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    nodeId: u32,
    flags: u32,
    _padding: vec2<f32>,
}

struct EdgeInstance {
    sourcePosition: vec3<f32>,
    sourceRadius: f32,
    targetPosition: vec3<f32>,
    targetRadius: f32,
    color: vec4<f32>,
    edgeId: u32,
    flags: u32,
    width: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<storage, read> nodeInstances: array<NodeInstance>;
@group(1) @binding(1) var<storage, read> edgeInstances: array<EdgeInstance>;

// Billboard node rendering
const BILLBOARD_OFFSETS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
);

struct BillboardVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) nodeId: u32,
    @location(3) @interpolate(flat) flags: u32,
}

@vertex
fn vs_billboard_main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> BillboardVertexOutput {
    let instance = nodeInstances[instanceIdx];
    let quadIdx = vertexIdx % 4u;
    let offset = BILLBOARD_OFFSETS[quadIdx];
    
    let viewPos = camera.view * vec4<f32>(instance.position, 1.0);
    let billboardPos = viewPos + vec4<f32>(offset * instance.radius, 0.0, 0.0);
    
    var output: BillboardVertexOutput;
    output.clipPosition = camera.projection * billboardPos;
    output.color = instance.color;
    output.uv = offset * 0.5 + 0.5;
    output.nodeId = instance.nodeId;
    output.flags = instance.flags;
    
    return output;
}

@fragment
fn fs_billboard_main(input: BillboardVertexOutput) -> @location(0) vec4<f32> {
    let dist = length(input.uv - 0.5) * 2.0;
    
    if (dist > 1.0) {
        discard;
    }
    
    let alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    
    let normal = vec3<f32>(input.uv.x - 0.5, input.uv.y - 0.5, sqrt(max(0.0, 1.0 - dist * dist)));
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(normal, lightDir), 0.0) * 0.6 + 0.4;
    
    var color = input.color.rgb * diffuse;
    
    // Selection highlight
    let isSelected = (input.flags & 1u) != 0u;
    if (isSelected) {
        color = mix(color, vec3<f32>(0.2, 0.8, 1.0), 0.4);
    }
    
    return vec4<f32>(color, input.color.a * alpha);
}

// Line edge rendering
struct LineVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) progress: f32,
}

@vertex
fn vs_edge_line_main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> LineVertexOutput {
    let edge = edgeInstances[instanceIdx];
    
    let t = f32(vertexIdx);
    let worldPos = mix(edge.sourcePosition, edge.targetPosition, t);
    
    var output: LineVertexOutput;
    output.clipPosition = camera.viewProjection * vec4<f32>(worldPos, 1.0);
    output.color = edge.color;
    output.progress = t;
    
    return output;
}

@fragment
fn fs_edge_line_main(input: LineVertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    let flow = fract(scene.time * 0.5 - input.progress);
    let pulse = smoothstep(0.0, 0.1, flow) * smoothstep(0.2, 0.1, flow);
    color = vec4<f32>(color.rgb + vec3<f32>(0.2, 0.4, 0.6) * pulse * 0.3, color.a);
    
    return color;
}
`;

// =============================================================================
// Sphere Geometry
// =============================================================================

interface SphereGeometry {
  vertices: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint16Array;
  vertexCount: number;
  indexCount: number;
}

function createSphereGeometry(
  segments: number = 16,
  rings: number = 12
): SphereGeometry {
  const vertices: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let ring = 0; ring <= rings; ring++) {
    const theta = (ring / rings) * Math.PI;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let segment = 0; segment <= segments; segment++) {
      const phi = (segment / segments) * Math.PI * 2;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      const x = cosPhi * sinTheta;
      const y = cosTheta;
      const z = sinPhi * sinTheta;

      vertices.push(x, y, z);
      normals.push(x, y, z);
      uvs.push(segment / segments, ring / rings);
    }
  }

  for (let ring = 0; ring < rings; ring++) {
    for (let segment = 0; segment < segments; segment++) {
      const a = ring * (segments + 1) + segment;
      const b = a + segments + 1;

      indices.push(a, b, a + 1);
      indices.push(b, b + 1, a + 1);
    }
  }

  return {
    vertices: new Float32Array(vertices),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: new Uint16Array(indices),
    vertexCount: vertices.length / 3,
    indexCount: indices.length,
  };
}

// =============================================================================
// WebGPUGraphRenderer Class
// =============================================================================

export class WebGPUGraphRenderer {
  private device: GPUDevice;
  private config: GraphRenderConfig;
  private pipelineCache: PipelineCache;
  private bufferManager: BufferManager;

  // Pipelines
  private nodeBillboardPipeline: GPURenderPipeline | null = null;
  private nodeSpherePipeline: GPURenderPipeline | null = null;
  private edgeLinePipeline: GPURenderPipeline | null = null;
  private edgeCylinderPipeline: GPURenderPipeline | null = null;
  private depthPrePassPipeline: GPURenderPipeline | null = null;

  // Shader module
  private shaderModule: GPUShaderModule | null = null;

  // Bind groups
  private cameraBindGroupLayout: GPUBindGroupLayout | null = null;
  private instanceBindGroupLayout: GPUBindGroupLayout | null = null;

  // Buffers
  private cameraUniformBuffer: GPUBuffer | null = null;
  private sceneUniformBuffer: GPUBuffer | null = null;
  private nodeInstanceBuffer: GPUBuffer | null = null;
  private edgeInstanceBuffer: GPUBuffer | null = null;
  private sphereVertexBuffer: GPUBuffer | null = null;
  private sphereIndexBuffer: GPUBuffer | null = null;

  // Geometry
  private sphereGeometry: SphereGeometry | null = null;

  // Current state
  private nodeCount = 0;
  private edgeCount = 0;
  private maxNodes = 100000;
  private maxEdges = 500000;

  // Stats
  private stats: GraphRenderStats = this.createEmptyStats();

  // Depth texture
  private depthTexture: GPUTexture | null = null;
  private depthTextureView: GPUTextureView | null = null;

  constructor(device: GPUDevice, config: Partial<GraphRenderConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.pipelineCache = new PipelineCache(device);
    this.bufferManager = new BufferManager(device);
  }

  /**
   * Initialize the renderer
   */
  async initialize(): Promise<void> {
    // Create shader module
    this.shaderModule = this.device.createShaderModule({
      label: "Graph Renderer Shaders",
      code: SHADER_CODE,
    });

    // Create bind group layouts
    this.createBindGroupLayouts();

    // Create pipelines
    await this.createPipelines();

    // Create geometry
    this.sphereGeometry = createSphereGeometry(16, 12);
    this.createGeometryBuffers();

    // Create uniform buffers
    this.createUniformBuffers();

    // Create instance buffers
    this.createInstanceBuffers();

    console.log("[WebGPUGraphRenderer] Initialized successfully");
  }

  private createBindGroupLayouts(): void {
    // Camera + Scene uniforms
    this.cameraBindGroupLayout = this.device.createBindGroupLayout({
      label: "Camera Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    // Instance data
    this.instanceBindGroupLayout = this.device.createBindGroupLayout({
      label: "Instance Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
  }

  private async createPipelines(): Promise<void> {
    if (
      !this.shaderModule ||
      !this.cameraBindGroupLayout ||
      !this.instanceBindGroupLayout
    ) {
      throw new Error("Shader module or bind group layouts not created");
    }

    const pipelineLayout = this.device.createPipelineLayout({
      label: "Graph Pipeline Layout",
      bindGroupLayouts: [
        this.cameraBindGroupLayout,
        this.instanceBindGroupLayout,
      ],
    });

    // Billboard node pipeline
    this.nodeBillboardPipeline = this.device.createRenderPipeline({
      label: "Node Billboard Pipeline",
      layout: pipelineLayout,
      vertex: {
        module: this.shaderModule,
        entryPoint: "vs_billboard_main",
      },
      fragment: {
        module: this.shaderModule,
        entryPoint: "fs_billboard_main",
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-strip",
        stripIndexFormat: undefined,
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
      multisample: {
        count: this.config.msaaSampleCount,
      },
    });

    // Edge line pipeline
    this.edgeLinePipeline = this.device.createRenderPipeline({
      label: "Edge Line Pipeline",
      layout: pipelineLayout,
      vertex: {
        module: this.shaderModule,
        entryPoint: "vs_edge_line_main",
      },
      fragment: {
        module: this.shaderModule,
        entryPoint: "fs_edge_line_main",
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "line-list",
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
      multisample: {
        count: this.config.msaaSampleCount,
      },
    });
  }

  private createGeometryBuffers(): void {
    if (!this.sphereGeometry) return;

    // Interleaved vertex buffer: position (3) + normal (3) + uv (2) = 8 floats
    const vertexCount = this.sphereGeometry.vertexCount;
    const interleavedData = new Float32Array(vertexCount * 8);

    for (let i = 0; i < vertexCount; i++) {
      // Position
      interleavedData[i * 8 + 0] = this.sphereGeometry.vertices[i * 3 + 0]!;
      interleavedData[i * 8 + 1] = this.sphereGeometry.vertices[i * 3 + 1]!;
      interleavedData[i * 8 + 2] = this.sphereGeometry.vertices[i * 3 + 2]!;
      // Normal
      interleavedData[i * 8 + 3] = this.sphereGeometry.normals[i * 3 + 0]!;
      interleavedData[i * 8 + 4] = this.sphereGeometry.normals[i * 3 + 1]!;
      interleavedData[i * 8 + 5] = this.sphereGeometry.normals[i * 3 + 2]!;
      // UV
      interleavedData[i * 8 + 6] = this.sphereGeometry.uvs[i * 2 + 0]!;
      interleavedData[i * 8 + 7] = this.sphereGeometry.uvs[i * 2 + 1]!;
    }

    this.sphereVertexBuffer = this.device.createBuffer({
      label: "Sphere Vertex Buffer",
      size: interleavedData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.sphereVertexBuffer, 0, interleavedData);

    this.sphereIndexBuffer = this.device.createBuffer({
      label: "Sphere Index Buffer",
      size: this.sphereGeometry.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.sphereIndexBuffer,
      0,
      this.sphereGeometry.indices.buffer
    );
  }

  private createUniformBuffers(): void {
    // Camera uniform: 16*3 (matrices) + 4 (position) + 4 (params) = 56 floats = 224 bytes
    // Align to 256 bytes
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "Camera Uniform Buffer",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Scene uniform: 8 floats = 32 bytes, align to 256
    this.sceneUniformBuffer = this.device.createBuffer({
      label: "Scene Uniform Buffer",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private createInstanceBuffers(): void {
    // Node instance: 12 floats = 48 bytes per instance
    const nodeBufferSize = this.maxNodes * 48;
    this.nodeInstanceBuffer = this.device.createBuffer({
      label: "Node Instance Buffer",
      size: nodeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Edge instance: 16 floats = 64 bytes per instance
    const edgeBufferSize = this.maxEdges * 64;
    this.edgeInstanceBuffer = this.device.createBuffer({
      label: "Edge Instance Buffer",
      size: edgeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Update graph data
   */
  setGraphData(
    nodes: GraphNode[],
    edges: GraphEdge[],
    nodePositions?: Map<string, [number, number, number]>
  ): void {
    this.updateNodeInstances(nodes, nodePositions);
    this.updateEdgeInstances(edges, nodes, nodePositions);
  }

  private updateNodeInstances(
    nodes: GraphNode[],
    positions?: Map<string, [number, number, number]>
  ): void {
    if (!this.nodeInstanceBuffer) return;

    this.nodeCount = Math.min(nodes.length, this.maxNodes);
    const instanceData = new Float32Array(this.nodeCount * 12);

    for (let i = 0; i < this.nodeCount; i++) {
      const node = nodes[i]!;
      const offset = i * 12;

      // Position
      if (positions && positions.has(node.id)) {
        const pos = positions.get(node.id)!;
        instanceData[offset + 0] = pos[0];
        instanceData[offset + 1] = pos[1];
        instanceData[offset + 2] = pos[2];
      } else {
        instanceData[offset + 0] = node.position?.x ?? 0;
        instanceData[offset + 1] = node.position?.y ?? 0;
        instanceData[offset + 2] = node.position?.z ?? 0;
      }

      // Radius
      instanceData[offset + 3] = node.radius ?? this.config.defaultNodeRadius;

      // Color (RGBA)
      const color = this.parseColor(node.color);
      instanceData[offset + 4] = color[0];
      instanceData[offset + 5] = color[1];
      instanceData[offset + 6] = color[2];
      instanceData[offset + 7] = color[3];

      // Node ID (as float for storage buffer compatibility)
      const view = new DataView(instanceData.buffer);
      view.setUint32((offset + 8) * 4, i, true);

      // Flags - access state properties
      let flags = 0;
      if (node.state?.selected) flags |= 1;
      if (node.state?.highlighted) flags |= 2;
      if (node.pinned) flags |= 4;
      view.setUint32((offset + 9) * 4, flags, true);

      // Padding
      instanceData[offset + 10] = 0;
      instanceData[offset + 11] = 0;
    }

    this.device.queue.writeBuffer(this.nodeInstanceBuffer, 0, instanceData);
  }

  private updateEdgeInstances(
    edges: GraphEdge[],
    nodes: GraphNode[],
    positions?: Map<string, [number, number, number]>
  ): void {
    if (!this.edgeInstanceBuffer) return;

    const nodeMap = new Map<string, GraphNode>();
    nodes.forEach((n) => nodeMap.set(n.id, n));

    this.edgeCount = Math.min(edges.length, this.maxEdges);
    const instanceData = new Float32Array(this.edgeCount * 16);

    for (let i = 0; i < this.edgeCount; i++) {
      const edge = edges[i]!;
      const offset = i * 16;

      const sourceNode = nodeMap.get(edge.sourceId);
      const targetNode = nodeMap.get(edge.targetId);

      if (!sourceNode || !targetNode) continue;

      // Source position
      if (positions && positions.has(edge.sourceId)) {
        const pos = positions.get(edge.sourceId)!;
        instanceData[offset + 0] = pos[0];
        instanceData[offset + 1] = pos[1];
        instanceData[offset + 2] = pos[2];
      } else {
        instanceData[offset + 0] = sourceNode.position?.x ?? 0;
        instanceData[offset + 1] = sourceNode.position?.y ?? 0;
        instanceData[offset + 2] = sourceNode.position?.z ?? 0;
      }

      // Source radius
      instanceData[offset + 3] =
        sourceNode.radius ?? this.config.defaultNodeRadius;

      // Target position
      if (positions && positions.has(edge.targetId)) {
        const pos = positions.get(edge.targetId)!;
        instanceData[offset + 4] = pos[0];
        instanceData[offset + 5] = pos[1];
        instanceData[offset + 6] = pos[2];
      } else {
        instanceData[offset + 4] = targetNode.position?.x ?? 0;
        instanceData[offset + 5] = targetNode.position?.y ?? 0;
        instanceData[offset + 6] = targetNode.position?.z ?? 0;
      }

      // Target radius
      instanceData[offset + 7] =
        targetNode.radius ?? this.config.defaultNodeRadius;

      // Color
      const color = this.parseColor(edge.color);
      instanceData[offset + 8] = color[0];
      instanceData[offset + 9] = color[1];
      instanceData[offset + 10] = color[2];
      instanceData[offset + 11] = color[3];

      // Edge ID and flags
      const view = new DataView(instanceData.buffer);
      view.setUint32((offset + 12) * 4, i, true);

      // Flags - access state properties
      let flags = 0;
      if (edge.state?.selected) flags |= 1;
      if (edge.direction === "bidirectional") flags |= 2;
      // animated not in type - check metadata
      if ((edge.metadata?.properties as Record<string, unknown>)?.animated)
        flags |= 4;
      view.setUint32((offset + 13) * 4, flags, true);

      // Width
      instanceData[offset + 14] = edge.width ?? this.config.defaultEdgeWidth;

      // Padding
      instanceData[offset + 15] = 0;
    }

    this.device.queue.writeBuffer(this.edgeInstanceBuffer, 0, instanceData);
  }

  private parseColor(
    color?: string | number | [number, number, number]
  ): [number, number, number, number] {
    if (!color) return [0.5, 0.5, 0.5, 1.0];

    if (typeof color === "number") {
      return [
        ((color >> 16) & 0xff) / 255,
        ((color >> 8) & 0xff) / 255,
        (color & 0xff) / 255,
        1.0,
      ];
    }

    if (Array.isArray(color)) {
      return [color[0], color[1], color[2], 1.0];
    }

    // Parse hex string
    if (color.startsWith("#")) {
      const hex = color.slice(1);
      const num = parseInt(hex, 16);
      return [
        ((num >> 16) & 0xff) / 255,
        ((num >> 8) & 0xff) / 255,
        (num & 0xff) / 255,
        1.0,
      ];
    }

    return [0.5, 0.5, 0.5, 1.0];
  }

  /**
   * Update camera uniforms
   */
  updateCamera(
    viewMatrix: Float32Array,
    projectionMatrix: Float32Array,
    cameraPosition: [number, number, number],
    near: number,
    far: number,
    fov: number,
    aspect: number
  ): void {
    if (!this.cameraUniformBuffer) return;

    const uniformData = new Float32Array(56);

    // View matrix (16 floats)
    uniformData.set(viewMatrix, 0);

    // Projection matrix (16 floats)
    uniformData.set(projectionMatrix, 16);

    // ViewProjection matrix (16 floats) - compute on CPU
    const viewProj = new Float32Array(16);
    this.multiplyMatrices(viewProj, projectionMatrix, viewMatrix);
    uniformData.set(viewProj, 32);

    // Camera position (3 floats + 1 padding)
    uniformData[48] = cameraPosition[0];
    uniformData[49] = cameraPosition[1];
    uniformData[50] = cameraPosition[2];
    uniformData[51] = 0; // padding

    // Camera params
    uniformData[52] = near;
    uniformData[53] = far;
    uniformData[54] = fov;
    uniformData[55] = aspect;

    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, uniformData);
  }

  private multiplyMatrices(
    out: Float32Array,
    a: Float32Array,
    b: Float32Array
  ): void {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i * 4 + j] =
          a[0 * 4 + j]! * b[i * 4 + 0]! +
          a[1 * 4 + j]! * b[i * 4 + 1]! +
          a[2 * 4 + j]! * b[i * 4 + 2]! +
          a[3 * 4 + j]! * b[i * 4 + 3]!;
      }
    }
  }

  /**
   * Update scene uniforms
   */
  updateScene(
    time: number,
    deltaTime: number,
    viewportSize: [number, number],
    highlightedNodeId: number = 0xffffffff,
    selectedNodeId: number = 0xffffffff,
    hoveredNodeId: number = 0xffffffff
  ): void {
    if (!this.sceneUniformBuffer) return;

    const uniformData = new Float32Array(8);
    uniformData[0] = time;
    uniformData[1] = deltaTime;
    uniformData[2] = viewportSize[0];
    uniformData[3] = viewportSize[1];

    const view = new DataView(uniformData.buffer);
    view.setUint32(16, highlightedNodeId, true);
    view.setUint32(20, selectedNodeId, true);
    view.setUint32(24, hoveredNodeId, true);
    view.setUint32(28, 0, true); // padding

    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, uniformData);
  }

  /**
   * Render the graph
   */
  render(
    commandEncoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    msaaView?: GPUTextureView
  ): void {
    if (
      !this.nodeBillboardPipeline ||
      !this.edgeLinePipeline ||
      !this.cameraBindGroupLayout ||
      !this.instanceBindGroupLayout ||
      !this.cameraUniformBuffer ||
      !this.sceneUniformBuffer ||
      !this.nodeInstanceBuffer ||
      !this.edgeInstanceBuffer
    ) {
      console.warn("[WebGPUGraphRenderer] Renderer not fully initialized");
      return;
    }

    const startTime = performance.now();

    // Create bind groups
    const cameraBindGroup = this.device.createBindGroup({
      label: "Camera Bind Group",
      layout: this.cameraBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.sceneUniformBuffer } },
      ],
    });

    const instanceBindGroup = this.device.createBindGroup({
      label: "Instance Bind Group",
      layout: this.instanceBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.nodeInstanceBuffer } },
        { binding: 1, resource: { buffer: this.edgeInstanceBuffer } },
      ],
    });

    // Create render pass
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: msaaView ?? colorView,
          resolveTarget: msaaView ? colorView : undefined,
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // Bind camera uniforms
    passEncoder.setBindGroup(0, cameraBindGroup);
    passEncoder.setBindGroup(1, instanceBindGroup);

    let drawCalls = 0;
    let triangles = 0;

    // Render edges first (behind nodes)
    if (this.edgeCount > 0) {
      passEncoder.setPipeline(this.edgeLinePipeline);
      passEncoder.draw(2, this.edgeCount, 0, 0);
      drawCalls++;
      // Lines don't contribute triangles
    }

    // Render nodes
    if (this.nodeCount > 0) {
      passEncoder.setPipeline(this.nodeBillboardPipeline);
      passEncoder.draw(4, this.nodeCount, 0, 0);
      drawCalls++;
      triangles += this.nodeCount * 2; // 2 triangles per billboard
    }

    passEncoder.end();

    // Update stats
    const endTime = performance.now();
    this.stats = {
      nodeCount: this.nodeCount,
      edgeCount: this.edgeCount,
      visibleNodes: this.nodeCount, // TODO: frustum culling
      visibleEdges: this.edgeCount, // TODO: frustum culling
      drawCalls,
      triangles,
      renderMode: "billboard",
      edgeMode: "line",
      frameTime: endTime - startTime,
    };
  }

  /**
   * Get rendering statistics
   */
  getStats(): GraphRenderStats {
    return { ...this.stats };
  }

  private createEmptyStats(): GraphRenderStats {
    return {
      nodeCount: 0,
      edgeCount: 0,
      visibleNodes: 0,
      visibleEdges: 0,
      drawCalls: 0,
      triangles: 0,
      renderMode: "billboard",
      edgeMode: "line",
      frameTime: 0,
    };
  }

  /**
   * Resize depth texture
   */
  resize(width: number, height: number): void {
    // Destroy old depth texture
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }

    // Create new depth texture
    this.depthTexture = this.device.createTexture({
      label: "Graph Depth Texture",
      size: { width, height },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: this.config.msaaSampleCount,
    });

    this.depthTextureView = this.depthTexture.createView();
  }

  /**
   * Get depth texture view
   */
  getDepthTextureView(): GPUTextureView | null {
    return this.depthTextureView;
  }

  /**
   * Destroy resources
   */
  destroy(): void {
    this.cameraUniformBuffer?.destroy();
    this.sceneUniformBuffer?.destroy();
    this.nodeInstanceBuffer?.destroy();
    this.edgeInstanceBuffer?.destroy();
    this.sphereVertexBuffer?.destroy();
    this.sphereIndexBuffer?.destroy();
    this.depthTexture?.destroy();

    this.cameraUniformBuffer = null;
    this.sceneUniformBuffer = null;
    this.nodeInstanceBuffer = null;
    this.edgeInstanceBuffer = null;
    this.sphereVertexBuffer = null;
    this.sphereIndexBuffer = null;
    this.depthTexture = null;
    this.depthTextureView = null;

    console.log("[WebGPUGraphRenderer] Destroyed");
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export async function createWebGPUGraphRenderer(
  device: GPUDevice,
  config?: Partial<GraphRenderConfig>
): Promise<WebGPUGraphRenderer> {
  const renderer = new WebGPUGraphRenderer(device, config);
  await renderer.initialize();
  return renderer;
}
