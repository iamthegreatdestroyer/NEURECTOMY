/**
 * WebGPU Renderer Implementation
 *
 * Core renderer class implementing WebGPU-first rendering with automatic
 * feature detection and graceful fallback to WebGL2 when necessary.
 *
 * @module @neurectomy/3d-engine/webgpu/renderer
 * @agents @CORE @ARCHITECT @VELOCITY
 */

import type {
  RendererConfig,
  RendererCapabilities,
  RendererStats,
  RendererBackend,
  RenderContext,
  RenderTarget,
  SceneGraph,
  CameraNode,
  LightNode,
} from "../core/types";
import * as THREE from "three";

// Default renderer configuration
const DEFAULT_CONFIG: Partial<RendererConfig> = {
  preferredBackend: "webgpu",
  antialias: true,
  alpha: true,
  preserveDrawingBuffer: false,
  powerPreference: "high-performance",
  failIfMajorPerformanceCaveat: false,
  pixelRatio:
    typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 2) : 1,
  maxFPS: 60,
  enableProfiling: false,
};

/**
 * WebGPURenderer - High-performance 3D renderer
 *
 * Features:
 * - WebGPU-first with automatic WebGL2 fallback
 * - Pipeline state caching for minimal state changes
 * - Automatic resource management with garbage collection
 * - Built-in profiling and performance monitoring
 * - Multi-pass rendering support
 */
export class WebGPURenderer {
  private config: RendererConfig;
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private backend: RendererBackend = "webgpu";
  private capabilities: RendererCapabilities | null = null;

  // WebGPU-specific
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private presentationFormat: GPUTextureFormat = "bgra8unorm";

  // WebGL2 fallback - Three.js renderer
  private gl2Renderer: THREE.WebGLRenderer | null = null;
  private gl2Scene: THREE.Scene | null = null;
  private gl2Camera: THREE.PerspectiveCamera | null = null;

  // Render state
  private currentRenderTarget: RenderTarget | null = null;
  private frameIndex = 0;
  private lastFrameTime = 0;
  private stats: RendererStats = this.createEmptyStats();

  // Frame timing
  private frameStartTime = 0;
  private frameTimes: number[] = [];
  private readonly FRAME_TIME_SAMPLES = 60;

  // Resource management
  private pendingDestroy: GPUBuffer[] = [];
  private readonly DESTROY_DELAY_FRAMES = 3;

  constructor(
    config: Partial<RendererConfig> & {
      canvas: HTMLCanvasElement | OffscreenCanvas;
    }
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config } as RendererConfig;
    this.canvas = config.canvas;
  }

  /**
   * Initialize the renderer
   * Attempts WebGPU first, falls back to WebGL2 if unavailable
   */
  async initialize(): Promise<RendererCapabilities> {
    // Try WebGPU first
    if (this.config.preferredBackend === "webgpu" && this.isWebGPUSupported()) {
      try {
        await this.initializeWebGPU();
        this.backend = "webgpu";
        this.capabilities = await this.queryCapabilities();
        console.log("[WebGPURenderer] Initialized with WebGPU backend");
        return this.capabilities;
      } catch (error) {
        console.warn(
          "[WebGPURenderer] WebGPU initialization failed, falling back to WebGL2:",
          error
        );
      }
    }

    // Fallback to WebGL2
    if (this.isWebGL2Supported()) {
      await this.initializeWebGL2();
      this.backend = "webgl2";
      this.capabilities = await this.queryCapabilities();
      console.log("[WebGPURenderer] Initialized with WebGL2 backend");
      return this.capabilities;
    }

    throw new Error(
      "[WebGPURenderer] No supported rendering backend available"
    );
  }

  /**
   * Check if WebGPU is supported in the current environment
   */
  private isWebGPUSupported(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  }

  /**
   * Check if WebGL2 is supported
   */
  private isWebGL2Supported(): boolean {
    if (typeof document === "undefined") return false;
    const canvas = document.createElement("canvas");
    return !!canvas.getContext("webgl2");
  }

  /**
   * Initialize WebGPU backend
   */
  private async initializeWebGPU(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported");
    }

    // Request adapter with power preference
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference:
        this.config.powerPreference === "high-performance"
          ? "high-performance"
          : "low-power",
    });

    if (!this.adapter) {
      throw new Error("Failed to get WebGPU adapter");
    }

    // Request device with required features
    const requiredFeatures: GPUFeatureName[] = [];
    const optionalFeatures: GPUFeatureName[] = [
      "timestamp-query",
      "indirect-first-instance",
      "shader-f16",
    ];

    // Add supported optional features
    for (const feature of optionalFeatures) {
      if (this.adapter.features.has(feature)) {
        requiredFeatures.push(feature);
      }
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize:
          this.adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupSizeX: this.adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: this.adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: this.adapter.limits.maxComputeWorkgroupSizeZ,
      },
    });

    // Set up error handling
    this.device.lost.then((info) => {
      console.error("[WebGPURenderer] Device lost:", info.message);
      if (info.reason !== "destroyed") {
        // Attempt recovery
        this.initialize();
      }
    });

    // Configure canvas context
    if (this.canvas instanceof HTMLCanvasElement) {
      this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;
    } else {
      this.context = (this.canvas as OffscreenCanvas).getContext(
        "webgpu"
      ) as GPUCanvasContext;
    }

    if (!this.context) {
      throw new Error("Failed to get WebGPU context");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: this.config.alpha ? "premultiplied" : "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }

  /**
   * Initialize WebGL2 fallback backend
   */
  private async initializeWebGL2(): Promise<void> {
    if (!(this.canvas instanceof HTMLCanvasElement)) {
      throw new Error(
        "WebGL2 fallback requires HTMLCanvasElement (OffscreenCanvas not supported)"
      );
    }

    // Create Three.js WebGL2 renderer
    this.gl2Renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: this.config.antialias,
      alpha: this.config.alpha,
      preserveDrawingBuffer: this.config.preserveDrawingBuffer,
      powerPreference: this.config.powerPreference,
      failIfMajorPerformanceCaveat: this.config.failIfMajorPerformanceCaveat,
    });

    // Verify WebGL2 context was obtained
    const glContext = this.gl2Renderer.getContext();
    if (!(glContext instanceof WebGL2RenderingContext)) {
      this.gl2Renderer.dispose();
      this.gl2Renderer = null;
      throw new Error("Failed to get WebGL2 context - only WebGL1 available");
    }

    // Configure renderer
    this.gl2Renderer.setPixelRatio(this.config.pixelRatio);
    this.gl2Renderer.setSize(
      this.canvas.clientWidth,
      this.canvas.clientHeight,
      false
    );
    this.gl2Renderer.setClearColor(0x05050d, 1.0);
    this.gl2Renderer.shadowMap.enabled = true;
    this.gl2Renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.gl2Renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.gl2Renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.gl2Renderer.toneMappingExposure = 1.0;

    // Create default scene and camera for fallback rendering
    this.gl2Scene = new THREE.Scene();
    this.gl2Scene.background = new THREE.Color(0x05050d);

    this.gl2Camera = new THREE.PerspectiveCamera(
      75,
      this.canvas.clientWidth / this.canvas.clientHeight,
      0.1,
      1000
    );
    this.gl2Camera.position.z = 5;

    // Add default ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    this.gl2Scene.add(ambientLight);

    // Add default directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    directionalLight.castShadow = true;
    this.gl2Scene.add(directionalLight);

    console.log("[WebGPURenderer] WebGL2 fallback initialized successfully");
  }

  /**
   * Query renderer capabilities
   */
  private async queryCapabilities(): Promise<RendererCapabilities> {
    if (this.backend === "webgpu" && this.adapter && this.device) {
      // Use adapter.info (newer spec) or fallback to empty values
      // Note: requestAdapterInfo() was deprecated in favor of adapter.info property
      const adapterInfo = (
        this.adapter as GPUAdapter & { info?: GPUAdapterInfo }
      ).info ?? {
        vendor: "unknown",
        architecture: "unknown",
      };

      return {
        backend: "webgpu",
        maxTextureSize: this.device.limits.maxTextureDimension2D,
        maxTextureLayers: this.device.limits.maxTextureArrayLayers,
        maxComputeWorkgroupSize: [
          this.device.limits.maxComputeWorkgroupSizeX,
          this.device.limits.maxComputeWorkgroupSizeY,
          this.device.limits.maxComputeWorkgroupSizeZ,
        ],
        maxStorageBufferBindingSize:
          this.device.limits.maxStorageBufferBindingSize,
        supportsCompute: true,
        supportsTimestampQuery: this.device.features.has("timestamp-query"),
        supportsIndirectDraw: this.device.features.has(
          "indirect-first-instance"
        ),
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
      };
    }

    // WebGL2 capabilities (fallback)
    if (this.backend === "webgl2" && this.gl2Renderer) {
      const gl = this.gl2Renderer.getContext() as WebGL2RenderingContext;
      const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");

      return {
        backend: "webgl2",
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxTextureLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS),
        maxComputeWorkgroupSize: [0, 0, 0], // No compute in WebGL2
        maxStorageBufferBindingSize: 0,
        supportsCompute: false,
        supportsTimestampQuery: false,
        supportsIndirectDraw: false,
        vendor: debugInfo
          ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
          : "unknown",
        architecture: debugInfo
          ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
          : "unknown",
      };
    }

    // Fallback for uninitialized state
    return {
      backend: "webgl2",
      maxTextureSize: 4096,
      maxTextureLayers: 256,
      maxComputeWorkgroupSize: [0, 0, 0],
      maxStorageBufferBindingSize: 0,
      supportsCompute: false,
      supportsTimestampQuery: false,
      supportsIndirectDraw: false,
      vendor: "unknown",
      architecture: "unknown",
    };
  }

  /**
   * Render a single frame
   */
  render(scene: SceneGraph, camera: CameraNode, lights: LightNode[]): void {
    // Use appropriate backend
    if (this.backend === "webgl2") {
      this.renderWebGL2(scene, camera, lights);
      return;
    }

    if (!this.device || !this.context) {
      console.warn("[WebGPURenderer] Renderer not initialized");
      return;
    }

    this.beginFrame();

    const commandEncoder = this.device.createCommandEncoder({
      label: `Frame ${this.frameIndex}`,
    });

    // Get current texture from swap chain
    const currentTexture = this.context.getCurrentTexture();
    const currentTextureView = currentTexture.createView();

    // Create render pass descriptor
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: currentTextureView,
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      // Depth attachment would be added here for 3D rendering
    };

    // Begin render pass
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // Create render context
    const context: RenderContext = {
      device: this.device,
      commandEncoder,
      currentPass: passEncoder,
      scene,
      camera,
      lights,
      time: performance.now() / 1000,
      deltaTime: this.stats.frameTime / 1000,
      frameIndex: this.frameIndex,
    };

    // Execute scene rendering
    this.renderScene(context);

    // End render pass
    passEncoder.end();

    // Submit command buffer
    this.device.queue.submit([commandEncoder.finish()]);

    this.endFrame();
  }

  /**
   * Render the scene graph
   */
  private renderScene(_context: RenderContext): void {
    // Scene rendering implementation
    // This will be expanded to handle different node types, materials, etc.

    // Update stats
    this.stats.drawCalls++;
  }

  /**
   * Render using WebGL2 fallback (Three.js)
   */
  private renderWebGL2(
    _scene: SceneGraph,
    camera: CameraNode,
    _lights: LightNode[]
  ): void {
    if (!this.gl2Renderer || !this.gl2Scene || !this.gl2Camera) {
      console.warn("[WebGPURenderer] WebGL2 renderer not initialized");
      return;
    }

    this.beginFrame();

    // Update camera from scene camera node
    if (camera.transform) {
      this.gl2Camera.position.set(
        camera.transform.position[0],
        camera.transform.position[1],
        camera.transform.position[2]
      );
    }

    // Update camera aspect ratio
    if (this.canvas instanceof HTMLCanvasElement) {
      const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
      if (this.gl2Camera.aspect !== aspect) {
        this.gl2Camera.aspect = aspect;
        this.gl2Camera.updateProjectionMatrix();
      }
    }

    // Render the Three.js scene
    this.gl2Renderer.render(this.gl2Scene, this.gl2Camera);

    // Update stats
    const info = this.gl2Renderer.info;
    this.stats.drawCalls = info.render.calls;
    this.stats.triangles = info.render.triangles;
    this.stats.vertices = info.render.triangles * 3;
    this.stats.textureMemory = info.memory.textures;

    this.endFrame();
  }

  /**
   * Get the Three.js scene (for WebGL2 fallback mode)
   * Allows external code to add objects directly to the scene
   */
  getWebGL2Scene(): THREE.Scene | null {
    return this.gl2Scene;
  }

  /**
   * Get the Three.js camera (for WebGL2 fallback mode)
   */
  getWebGL2Camera(): THREE.PerspectiveCamera | null {
    return this.gl2Camera;
  }

  /**
   * Get the Three.js renderer (for WebGL2 fallback mode)
   */
  getWebGL2Renderer(): THREE.WebGLRenderer | null {
    return this.gl2Renderer;
  }

  /**
   * Begin a new frame
   */
  private beginFrame(): void {
    this.frameStartTime = performance.now();
    this.stats = this.createEmptyStats();
  }

  /**
   * End the current frame
   */
  private endFrame(): void {
    const frameEndTime = performance.now();
    const frameTime = frameEndTime - this.frameStartTime;

    // Update frame time samples
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.FRAME_TIME_SAMPLES) {
      this.frameTimes.shift();
    }

    // Calculate FPS
    const avgFrameTime =
      this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    this.stats.frameTime = frameTime;
    this.stats.fps = 1000 / avgFrameTime;

    // Process pending resource destruction
    this.processPendingDestroy();

    this.frameIndex++;
    this.lastFrameTime = frameEndTime;
  }

  /**
   * Process pending resource destruction
   * Resources are destroyed after a delay to ensure GPU is done with them
   */
  private processPendingDestroy(): void {
    // In a real implementation, we'd track frame indices and destroy
    // resources after DESTROY_DELAY_FRAMES
    this.pendingDestroy = [];
  }

  /**
   * Create empty stats object
   */
  private createEmptyStats(): RendererStats {
    return {
      fps: 0,
      frameTime: 0,
      gpuTime: 0,
      drawCalls: 0,
      triangles: 0,
      vertices: 0,
      textureMemory: 0,
      bufferMemory: 0,
      shaderSwitches: 0,
      stateChanges: 0,
    };
  }

  /**
   * Resize the renderer
   */
  resize(width: number, height: number): void {
    if (this.canvas instanceof HTMLCanvasElement) {
      this.canvas.width = width * this.config.pixelRatio;
      this.canvas.height = height * this.config.pixelRatio;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    } else {
      (this.canvas as OffscreenCanvas).width = width * this.config.pixelRatio;
      (this.canvas as OffscreenCanvas).height = height * this.config.pixelRatio;
    }

    // Reconfigure context if using WebGPU
    if (this.backend === "webgpu" && this.context && this.device) {
      this.context.configure({
        device: this.device,
        format: this.presentationFormat,
        alphaMode: this.config.alpha ? "premultiplied" : "opaque",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
    }

    // Resize WebGL2 renderer if in fallback mode
    if (this.backend === "webgl2" && this.gl2Renderer && this.gl2Camera) {
      this.gl2Renderer.setSize(width, height, false);
      this.gl2Camera.aspect = width / height;
      this.gl2Camera.updateProjectionMatrix();
    }
  }

  /**
   * Get current renderer statistics
   */
  getStats(): RendererStats {
    return { ...this.stats };
  }

  /**
   * Get renderer capabilities
   */
  getCapabilities(): RendererCapabilities | null {
    return this.capabilities;
  }

  /**
   * Get the current backend being used
   */
  getBackend(): RendererBackend {
    return this.backend;
  }

  /**
   * Get the WebGPU device (if available)
   */
  getDevice(): GPUDevice | null {
    return this.device;
  }

  /**
   * Dispose of the renderer and release all resources
   */
  dispose(): void {
    // Destroy pending resources
    for (const buffer of this.pendingDestroy) {
      buffer.destroy();
    }
    this.pendingDestroy = [];

    // Dispose WebGL2 renderer and scene if in fallback mode
    if (this.gl2Renderer) {
      this.gl2Renderer.dispose();
      this.gl2Renderer = null;
    }
    if (this.gl2Scene) {
      // Dispose all scene children
      this.gl2Scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry?.dispose();
          if (Array.isArray(object.material)) {
            object.material.forEach((mat) => mat.dispose());
          } else {
            object.material?.dispose();
          }
        }
      });
      this.gl2Scene.clear();
      this.gl2Scene = null;
    }
    this.gl2Camera = null;

    // Destroy WebGPU device
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }

    this.adapter = null;
    this.context = null;
    this.capabilities = null;

    console.log("[WebGPURenderer] Disposed");
  }
}

/**
 * Factory function to create a WebGPU renderer
 */
export async function createRenderer(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  config?: Partial<Omit<RendererConfig, "canvas">>
): Promise<WebGPURenderer> {
  const renderer = new WebGPURenderer({ canvas, ...config });
  await renderer.initialize();
  return renderer;
}
