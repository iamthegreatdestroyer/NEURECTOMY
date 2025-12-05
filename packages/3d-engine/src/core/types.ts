/**
 * Core Type Definitions for 3D Engine
 *
 * @module @neurectomy/3d-engine/core/types
 * @agents @CORE @ARCHITECT @AXIOM
 */

import type { Position3D, Transform3D } from "@neurectomy/types";
export type { Quaternion } from "@neurectomy/types";

// =============================================================================
// Renderer Configuration
// =============================================================================

export type RendererBackend = "webgpu" | "webgl2" | "webgl";

export interface RendererCapabilities {
  backend: RendererBackend;
  maxTextureSize: number;
  maxTextureLayers: number;
  maxComputeWorkgroupSize: [number, number, number];
  maxStorageBufferBindingSize: number;
  supportsCompute: boolean;
  supportsTimestampQuery: boolean;
  supportsIndirectDraw: boolean;
  vendor: string;
  architecture: string;
}

export interface RendererConfig {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  preferredBackend: RendererBackend;
  antialias: boolean;
  alpha: boolean;
  preserveDrawingBuffer: boolean;
  powerPreference: "default" | "high-performance" | "low-power";
  failIfMajorPerformanceCaveat: boolean;
  pixelRatio: number;
  maxFPS: number;
  enableProfiling: boolean;
}

export interface RendererStats {
  fps: number;
  frameTime: number;
  gpuTime: number;
  drawCalls: number;
  triangles: number;
  vertices: number;
  textureMemory: number;
  bufferMemory: number;
  shaderSwitches: number;
  stateChanges: number;
}

// =============================================================================
// GPU Resource Types
// =============================================================================

export interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: GPUBufferUsageFlags;
  mappedAtCreation?: boolean;
}

export interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
}

export interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
  sourceMap?: object;
  hints?: Record<string, GPUShaderModuleCompilationHint>;
}

// =============================================================================
// Scene Graph Types
// =============================================================================

export type NodeType =
  | "root"
  | "group"
  | "mesh"
  | "light"
  | "camera"
  | "agent"
  | "connection"
  | "annotation"
  | "particle-system"
  | "volume"
  | "custom";

export interface SceneNode {
  id: string;
  type: NodeType;
  name: string;
  parent: string | null;
  children: string[];
  transform: Transform3D;
  visible: boolean;
  interactive: boolean;
  layer: number;
  userData: Record<string, unknown>;
}

export interface MeshNode extends SceneNode {
  type: "mesh";
  geometry: GeometryHandle;
  material: MaterialHandle;
  castShadow: boolean;
  receiveShadow: boolean;
  frustumCulled: boolean;
  renderOrder: number;
}

export interface AgentNode extends SceneNode {
  type: "agent";
  agentId: string;
  codename: string;
  tier: string;
  status: string;
  color: string;
  connections: string[];
  metadata: Record<string, unknown>;
}

export interface ConnectionNode extends SceneNode {
  type: "connection";
  sourceId: string;
  targetId: string;
  connectionType: "data-flow" | "dependency" | "communication" | "hierarchy";
  animated: boolean;
  thickness: number;
  color: string;
  dashPattern?: number[];
}

export interface LightNode extends SceneNode {
  type: "light";
  lightType: "ambient" | "directional" | "point" | "spot" | "hemisphere";
  color: [number, number, number];
  intensity: number;
  castShadow: boolean;
  shadowMapSize: number;
  shadowBias: number;
}

export interface CameraNode extends SceneNode {
  type: "camera";
  cameraType: "perspective" | "orthographic";
  fov: number;
  aspect: number;
  near: number;
  far: number;
  zoom: number;
}

// =============================================================================
// Geometry & Material Types
// =============================================================================

export type GeometryHandle = { __brand: "GeometryHandle"; id: string };
export type MaterialHandle = { __brand: "MaterialHandle"; id: string };
export type TextureHandle = { __brand: "TextureHandle"; id: string };
export type ShaderHandle = { __brand: "ShaderHandle"; id: string };

export interface GeometryDescriptor {
  id: string;
  type: "box" | "sphere" | "cylinder" | "plane" | "torus" | "custom";
  vertices: Float32Array;
  indices?: Uint32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  colors?: Float32Array;
  tangents?: Float32Array;
  boundingBox: BoundingBox;
  boundingSphere: BoundingSphere;
}

export interface MaterialDescriptor {
  id: string;
  type: "standard" | "physical" | "basic" | "line" | "point" | "custom";
  color: [number, number, number, number];
  emissive?: [number, number, number];
  emissiveIntensity?: number;
  metalness?: number;
  roughness?: number;
  opacity?: number;
  transparent?: boolean;
  side: "front" | "back" | "double";
  wireframe?: boolean;
  depthTest?: boolean;
  depthWrite?: boolean;
  blending?: "normal" | "additive" | "subtractive" | "multiply";
  textures?: {
    map?: TextureHandle;
    normalMap?: TextureHandle;
    roughnessMap?: TextureHandle;
    metalnessMap?: TextureHandle;
    emissiveMap?: TextureHandle;
    aoMap?: TextureHandle;
  };
  uniforms?: Record<string, ShaderUniform>;
  shader?: ShaderHandle;
}

export interface ShaderUniform {
  type:
    | "float"
    | "vec2"
    | "vec3"
    | "vec4"
    | "mat3"
    | "mat4"
    | "int"
    | "sampler2D";
  value: number | number[] | Float32Array | TextureHandle;
}

// =============================================================================
// Spatial Types
// =============================================================================

export interface BoundingBox {
  min: Position3D;
  max: Position3D;
}

export interface BoundingSphere {
  center: Position3D;
  radius: number;
}

export interface Ray {
  origin: Position3D;
  direction: Position3D;
}

export interface Frustum {
  planes: FrustumPlane[];
}

export interface FrustumPlane {
  normal: Position3D;
  distance: number;
}

export interface RaycastHit {
  nodeId: string;
  point: Position3D;
  normal: Position3D;
  distance: number;
  uv?: { u: number; v: number };
  faceIndex?: number;
}

// =============================================================================
// Render Pipeline Types
// =============================================================================

export interface RenderPass {
  id: string;
  name: string;
  enabled: boolean;
  order: number;
  inputs: string[];
  outputs: string[];
  execute: (context: RenderContext) => void;
}

export interface RenderContext {
  device: GPUDevice;
  commandEncoder: GPUCommandEncoder;
  currentPass: GPURenderPassEncoder | null;
  scene: SceneGraph;
  camera: CameraNode;
  lights: LightNode[];
  time: number;
  deltaTime: number;
  frameIndex: number;
}

export interface RenderTarget {
  id: string;
  width: number;
  height: number;
  colorTextures: GPUTexture[];
  depthTexture?: GPUTexture;
  sampleCount: number;
}

export interface SceneGraph {
  nodes: Map<string, SceneNode>;
  rootId: string;
  getNode(id: string): SceneNode | undefined;
  addNode(node: SceneNode): void;
  removeNode(id: string): void;
  updateNode(id: string, updates: Partial<SceneNode>): void;
  getChildren(id: string): SceneNode[];
  getDescendants(id: string): SceneNode[];
  getWorldTransform(id: string): Transform3D;
  raycast(ray: Ray, options?: RaycastOptions): RaycastHit[];
}

export interface RaycastOptions {
  recursive?: boolean;
  layers?: number[];
  onlyVisible?: boolean;
  onlyInteractive?: boolean;
  maxDistance?: number;
}

// =============================================================================
// Animation Types
// =============================================================================

export type EasingFunction =
  | "linear"
  | "easeInQuad"
  | "easeOutQuad"
  | "easeInOutQuad"
  | "easeInCubic"
  | "easeOutCubic"
  | "easeInOutCubic"
  | "easeInQuart"
  | "easeOutQuart"
  | "easeInOutQuart"
  | "easeInExpo"
  | "easeOutExpo"
  | "easeInOutExpo"
  | "easeInElastic"
  | "easeOutElastic"
  | "easeInOutElastic"
  | "easeInBack"
  | "easeOutBack"
  | "easeInOutBack"
  | "easeInBounce"
  | "easeOutBounce"
  | "easeInOutBounce";

export interface AnimationClip {
  id: string;
  name: string;
  duration: number;
  tracks: AnimationTrack[];
  loop: "once" | "repeat" | "pingpong";
  speed: number;
}

export interface AnimationTrack<T = unknown> {
  targetId: string;
  property: string;
  keyframes: AnimationKeyframe<T>[];
  interpolation: "linear" | "step" | "cubic";
}

export interface AnimationKeyframe<T = unknown> {
  time: number;
  value: T;
  easing?: EasingFunction;
  tangentIn?: T;
  tangentOut?: T;
}

export interface AnimationMixer {
  clips: Map<string, AnimationClip>;
  activeClips: Map<string, AnimationState>;
  play(clipId: string, options?: PlayOptions): void;
  stop(clipId: string): void;
  pause(clipId: string): void;
  setTime(clipId: string, time: number): void;
  crossFade(fromClipId: string, toClipId: string, duration: number): void;
  update(deltaTime: number): void;
}

export interface AnimationState {
  clipId: string;
  time: number;
  weight: number;
  playing: boolean;
  paused: boolean;
}

export interface PlayOptions {
  startTime?: number;
  weight?: number;
  fadeIn?: number;
}

// =============================================================================
// Event Types
// =============================================================================

export type EngineEventType =
  | "node:added"
  | "node:removed"
  | "node:updated"
  | "node:selected"
  | "node:deselected"
  | "node:hovered"
  | "node:unhovered"
  | "camera:changed"
  | "render:frame"
  | "render:resize"
  | "interaction:click"
  | "interaction:drag-start"
  | "interaction:drag"
  | "interaction:drag-end"
  | "physics:collision"
  | "animation:complete"
  | "error";

export interface EngineEvent<T = unknown> {
  type: EngineEventType;
  target?: string;
  data: T;
  timestamp: number;
}

export type EngineEventHandler<T = unknown> = (event: EngineEvent<T>) => void;

// =============================================================================
// Selection & Interaction Types
// =============================================================================

export interface SelectionState {
  selectedIds: Set<string>;
  hoveredId: string | null;
  focusedId: string | null;
}

export interface TransformGizmo {
  mode: "translate" | "rotate" | "scale";
  space: "local" | "world";
  size: number;
  visible: boolean;
  enabled: boolean;
  snap: boolean;
  snapTranslate: number;
  snapRotate: number;
  snapScale: number;
}

export interface InteractionState {
  isDragging: boolean;
  dragTarget: string | null;
  dragStartPosition: Position3D | null;
  dragCurrentPosition: Position3D | null;
  modifiers: {
    shift: boolean;
    ctrl: boolean;
    alt: boolean;
    meta: boolean;
  };
}
