/**
 * WebGPU Bridge for Three.js
 * 
 * Integrates Three.js WebGPURenderer with NEURECTOMY's WebGPU core,
 * enabling seamless transition and custom render passes.
 * 
 * @module @neurectomy/3d-engine/three/webgpu-bridge
 * @agents @CORE @APEX
 */

import * as THREE from 'three';
import { WebGPURenderer, type WebGPURendererConfig } from '../webgpu/renderer';
import type { RenderPassConfig } from '../webgpu/render-passes';

// =============================================================================
// Types
// =============================================================================

export interface WebGPUBridgeConfig {
  canvas: HTMLCanvasElement;
  antialias?: boolean;
  powerPreference?: GPUPowerPreference;
  useWebGPURenderer?: boolean; // Use Three.js WebGPU renderer when available
  fallbackToWebGL?: boolean;
  sampleCount?: number;
  debug?: boolean;
}

export interface RenderTarget {
  texture: THREE.Texture;
  depthTexture?: THREE.DepthTexture;
  width: number;
  height: number;
}

export interface PostProcessPass {
  name: string;
  shader: string;
  uniforms: Record<string, THREE.IUniform>;
  enabled: boolean;
}

export type RendererType = 'webgpu' | 'webgl' | 'webgpu-three';

// =============================================================================
// WebGPUBridge Class
// =============================================================================

/**
 * WebGPUBridge - Connects Three.js with WebGPU
 * 
 * Features:
 * - Automatic WebGPU/WebGL fallback
 * - Custom render passes integration
 * - Post-processing pipeline
 * - Multi-sample anti-aliasing
 * - HDR support
 */
export class WebGPUBridge {
  private config: Required<WebGPUBridgeConfig>;
  private canvas: HTMLCanvasElement;
  
  // Renderer state
  private threeRenderer?: THREE.WebGLRenderer;
  private webgpuRenderer?: WebGPURenderer;
  private rendererType: RendererType = 'webgl';
  private isInitialized = false;

  // Render targets
  private renderTargets = new Map<string, THREE.WebGLRenderTarget>();
  private postProcessPasses: PostProcessPass[] = [];
  
  // Post-processing
  private quadGeometry?: THREE.BufferGeometry;
  private quadMesh?: THREE.Mesh;
  private postProcessScene?: THREE.Scene;
  private postProcessCamera?: THREE.OrthographicCamera;

  // Stats
  private frameCount = 0;
  private lastFrameTime = 0;
  private fps = 0;

  constructor(config: WebGPUBridgeConfig) {
    this.config = {
      canvas: config.canvas,
      antialias: config.antialias ?? true,
      powerPreference: config.powerPreference ?? 'high-performance',
      useWebGPURenderer: config.useWebGPURenderer ?? true,
      fallbackToWebGL: config.fallbackToWebGL ?? true,
      sampleCount: config.sampleCount ?? 4,
      debug: config.debug ?? false,
    };

    this.canvas = config.canvas;
  }

  /**
   * Initialize the renderer
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;

    // Try WebGPU first
    if (this.config.useWebGPURenderer && 'gpu' in navigator) {
      try {
        const success = await this.initializeWebGPU();
        if (success) {
          this.rendererType = 'webgpu';
          this.isInitialized = true;
          console.log('[WebGPUBridge] Initialized with WebGPU');
          return true;
        }
      } catch (error) {
        console.warn('[WebGPUBridge] WebGPU initialization failed:', error);
      }
    }

    // Fallback to WebGL
    if (this.config.fallbackToWebGL) {
      try {
        this.initializeWebGL();
        this.rendererType = 'webgl';
        this.isInitialized = true;
        console.log('[WebGPUBridge] Initialized with WebGL fallback');
        return true;
      } catch (error) {
        console.error('[WebGPUBridge] WebGL initialization failed:', error);
      }
    }

    return false;
  }

  /**
   * Initialize WebGPU renderer
   */
  private async initializeWebGPU(): Promise<boolean> {
    const rendererConfig: WebGPURendererConfig = {
      canvas: this.canvas,
      powerPreference: this.config.powerPreference,
      antialias: this.config.antialias,
      sampleCount: this.config.sampleCount,
      debug: this.config.debug,
    };

    this.webgpuRenderer = new WebGPURenderer(rendererConfig);
    const initialized = await this.webgpuRenderer.initialize();

    if (!initialized) {
      this.webgpuRenderer = undefined;
      return false;
    }

    return true;
  }

  /**
   * Initialize WebGL renderer
   */
  private initializeWebGL(): void {
    this.threeRenderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: this.config.antialias,
      powerPreference: this.config.powerPreference,
      alpha: true,
      stencil: false,
    });

    this.threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.threeRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.threeRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.threeRenderer.toneMappingExposure = 1;
    this.threeRenderer.shadowMap.enabled = true;
    this.threeRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Setup post-processing
    this.setupPostProcessing();
  }

  /**
   * Setup post-processing pipeline
   */
  private setupPostProcessing(): void {
    // Create full-screen quad
    this.quadGeometry = new THREE.PlaneGeometry(2, 2);
    
    // Create post-process scene and camera
    this.postProcessScene = new THREE.Scene();
    this.postProcessCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  /**
   * Get renderer type
   */
  getRendererType(): RendererType {
    return this.rendererType;
  }

  /**
   * Get Three.js renderer (for WebGL mode)
   */
  getThreeRenderer(): THREE.WebGLRenderer | undefined {
    return this.threeRenderer;
  }

  /**
   * Get WebGPU renderer (for WebGPU mode)
   */
  getWebGPURenderer(): WebGPURenderer | undefined {
    return this.webgpuRenderer;
  }

  /**
   * Check if using WebGPU
   */
  isWebGPU(): boolean {
    return this.rendererType === 'webgpu' || this.rendererType === 'webgpu-three';
  }

  /**
   * Resize the renderer
   */
  resize(width: number, height: number): void {
    const pixelRatio = Math.min(window.devicePixelRatio, 2);

    if (this.threeRenderer) {
      this.threeRenderer.setSize(width, height);
      this.threeRenderer.setPixelRatio(pixelRatio);
    }

    if (this.webgpuRenderer) {
      this.webgpuRenderer.resize(width, height);
    }

    // Resize render targets
    for (const target of this.renderTargets.values()) {
      target.setSize(width * pixelRatio, height * pixelRatio);
    }
  }

  /**
   * Create a render target
   */
  createRenderTarget(
    name: string,
    width: number,
    height: number,
    options: Partial<THREE.WebGLRenderTargetOptions> = {}
  ): THREE.WebGLRenderTarget {
    const pixelRatio = Math.min(window.devicePixelRatio, 2);

    const target = new THREE.WebGLRenderTarget(
      width * pixelRatio,
      height * pixelRatio,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
        ...options,
      }
    );

    this.renderTargets.set(name, target);
    return target;
  }

  /**
   * Get render target
   */
  getRenderTarget(name: string): THREE.WebGLRenderTarget | undefined {
    return this.renderTargets.get(name);
  }

  /**
   * Add post-process pass
   */
  addPostProcessPass(pass: PostProcessPass): void {
    this.postProcessPasses.push(pass);
  }

  /**
   * Remove post-process pass
   */
  removePostProcessPass(name: string): void {
    const index = this.postProcessPasses.findIndex(p => p.name === name);
    if (index !== -1) {
      this.postProcessPasses.splice(index, 1);
    }
  }

  /**
   * Set post-process pass enabled
   */
  setPostProcessEnabled(name: string, enabled: boolean): void {
    const pass = this.postProcessPasses.find(p => p.name === name);
    if (pass) {
      pass.enabled = enabled;
    }
  }

  /**
   * Render a scene
   */
  render(scene: THREE.Scene, camera: THREE.Camera): void {
    const startTime = performance.now();

    if (this.rendererType === 'webgpu' && this.webgpuRenderer) {
      // WebGPU render (custom implementation)
      this.renderWithWebGPU(scene, camera);
    } else if (this.threeRenderer) {
      // WebGL render
      this.renderWithWebGL(scene, camera);
    }

    // Update FPS
    this.frameCount++;
    const elapsed = startTime - this.lastFrameTime;
    if (elapsed >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / elapsed);
      this.frameCount = 0;
      this.lastFrameTime = startTime;
    }
  }

  /**
   * Render with WebGPU
   */
  private renderWithWebGPU(scene: THREE.Scene, camera: THREE.Camera): void {
    if (!this.webgpuRenderer) return;

    // For now, fall back to WebGL for Three.js scene rendering
    // Full WebGPU integration would require custom scene traversal
    console.warn('[WebGPUBridge] Full WebGPU rendering not yet implemented, using WebGL');
    this.initializeWebGL();
    this.renderWithWebGL(scene, camera);
  }

  /**
   * Render with WebGL
   */
  private renderWithWebGL(scene: THREE.Scene, camera: THREE.Camera): void {
    if (!this.threeRenderer) return;

    // Check if we have enabled post-process passes
    const enabledPasses = this.postProcessPasses.filter(p => p.enabled);

    if (enabledPasses.length === 0) {
      // Direct render
      this.threeRenderer.render(scene, camera);
    } else {
      // Render with post-processing
      this.renderWithPostProcess(scene, camera, enabledPasses);
    }
  }

  /**
   * Render with post-processing
   */
  private renderWithPostProcess(
    scene: THREE.Scene,
    camera: THREE.Camera,
    passes: PostProcessPass[]
  ): void {
    if (!this.threeRenderer || !this.postProcessScene || !this.postProcessCamera) return;

    // Get or create render targets
    let readTarget = this.getRenderTarget('postprocess-read');
    let writeTarget = this.getRenderTarget('postprocess-write');

    if (!readTarget || !writeTarget) {
      const { width, height } = this.threeRenderer.getSize(new THREE.Vector2());
      readTarget = this.createRenderTarget('postprocess-read', width, height);
      writeTarget = this.createRenderTarget('postprocess-write', width, height);
    }

    // Render scene to first target
    this.threeRenderer.setRenderTarget(readTarget);
    this.threeRenderer.render(scene, camera);

    // Apply each pass
    for (let i = 0; i < passes.length; i++) {
      const pass = passes[i]!;
      const isLast = i === passes.length - 1;

      // Create shader material for this pass
      const material = new THREE.ShaderMaterial({
        uniforms: {
          ...pass.uniforms,
          tDiffuse: { value: readTarget.texture },
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position, 1.0);
          }
        `,
        fragmentShader: pass.shader,
      });

      // Create or update quad
      if (!this.quadMesh) {
        this.quadMesh = new THREE.Mesh(this.quadGeometry, material);
        this.postProcessScene.add(this.quadMesh);
      } else {
        this.quadMesh.material = material;
      }

      // Render to write target or screen
      if (isLast) {
        this.threeRenderer.setRenderTarget(null);
      } else {
        this.threeRenderer.setRenderTarget(writeTarget);
      }

      this.threeRenderer.render(this.postProcessScene, this.postProcessCamera);

      // Swap targets
      [readTarget, writeTarget] = [writeTarget, readTarget];

      // Dispose material
      material.dispose();
    }
  }

  /**
   * Get current FPS
   */
  getFPS(): number {
    return this.fps;
  }

  /**
   * Get renderer info
   */
  getInfo(): {
    type: RendererType;
    fps: number;
    memory?: {
      geometries: number;
      textures: number;
    };
    render?: {
      calls: number;
      triangles: number;
      points: number;
      lines: number;
    };
  } {
    const info: ReturnType<typeof this.getInfo> = {
      type: this.rendererType,
      fps: this.fps,
    };

    if (this.threeRenderer) {
      const glInfo = this.threeRenderer.info;
      info.memory = {
        geometries: glInfo.memory.geometries,
        textures: glInfo.memory.textures,
      };
      info.render = {
        calls: glInfo.render.calls,
        triangles: glInfo.render.triangles,
        points: glInfo.render.points,
        lines: glInfo.render.lines,
      };
    }

    return info;
  }

  /**
   * Take a screenshot
   */
  screenshot(
    scene: THREE.Scene,
    camera: THREE.Camera,
    width?: number,
    height?: number
  ): string {
    if (!this.threeRenderer) return '';

    const originalSize = this.threeRenderer.getSize(new THREE.Vector2());
    const targetWidth = width ?? originalSize.x;
    const targetHeight = height ?? originalSize.y;

    // Create screenshot render target
    const renderTarget = new THREE.WebGLRenderTarget(targetWidth, targetHeight, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });

    // Render to target
    this.threeRenderer.setRenderTarget(renderTarget);
    this.threeRenderer.render(scene, camera);

    // Read pixels
    const pixels = new Uint8Array(targetWidth * targetHeight * 4);
    this.threeRenderer.readRenderTargetPixels(
      renderTarget,
      0, 0,
      targetWidth, targetHeight,
      pixels
    );

    // Reset render target
    this.threeRenderer.setRenderTarget(null);

    // Convert to canvas and data URL
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');

    if (ctx) {
      const imageData = ctx.createImageData(targetWidth, targetHeight);
      
      // Flip Y and copy pixels
      for (let y = 0; y < targetHeight; y++) {
        for (let x = 0; x < targetWidth; x++) {
          const srcIdx = ((targetHeight - y - 1) * targetWidth + x) * 4;
          const dstIdx = (y * targetWidth + x) * 4;
          imageData.data[dstIdx] = pixels[srcIdx]!;
          imageData.data[dstIdx + 1] = pixels[srcIdx + 1]!;
          imageData.data[dstIdx + 2] = pixels[srcIdx + 2]!;
          imageData.data[dstIdx + 3] = pixels[srcIdx + 3]!;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    }

    // Cleanup
    renderTarget.dispose();

    return canvas.toDataURL('image/png');
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    // Dispose render targets
    for (const target of this.renderTargets.values()) {
      target.dispose();
    }
    this.renderTargets.clear();

    // Dispose post-process resources
    this.quadGeometry?.dispose();
    if (this.quadMesh?.material instanceof THREE.Material) {
      this.quadMesh.material.dispose();
    }

    // Dispose renderers
    this.threeRenderer?.dispose();
    this.webgpuRenderer?.destroy();

    this.isInitialized = false;
    console.log('[WebGPUBridge] Disposed');
  }
}

// =============================================================================
// Post-Process Shaders
// =============================================================================

export const PostProcessShaders = {
  /**
   * FXAA anti-aliasing
   */
  fxaa: `
    uniform sampler2D tDiffuse;
    uniform vec2 resolution;
    varying vec2 vUv;

    #define FXAA_REDUCE_MIN (1.0 / 128.0)
    #define FXAA_REDUCE_MUL (1.0 / 8.0)
    #define FXAA_SPAN_MAX 8.0

    void main() {
      vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
      
      vec3 rgbNW = texture2D(tDiffuse, vUv + vec2(-1.0, -1.0) * inverseVP).xyz;
      vec3 rgbNE = texture2D(tDiffuse, vUv + vec2(1.0, -1.0) * inverseVP).xyz;
      vec3 rgbSW = texture2D(tDiffuse, vUv + vec2(-1.0, 1.0) * inverseVP).xyz;
      vec3 rgbSE = texture2D(tDiffuse, vUv + vec2(1.0, 1.0) * inverseVP).xyz;
      vec3 rgbM = texture2D(tDiffuse, vUv).xyz;
      
      vec3 luma = vec3(0.299, 0.587, 0.114);
      float lumaNW = dot(rgbNW, luma);
      float lumaNE = dot(rgbNE, luma);
      float lumaSW = dot(rgbSW, luma);
      float lumaSE = dot(rgbSE, luma);
      float lumaM = dot(rgbM, luma);
      
      float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
      float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
      
      vec2 dir;
      dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
      dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
      
      float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
      float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
      
      dir = min(vec2(FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX), dir * rcpDirMin)) * inverseVP;
      
      vec3 rgbA = 0.5 * (
        texture2D(tDiffuse, vUv + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture2D(tDiffuse, vUv + dir * (2.0 / 3.0 - 0.5)).xyz
      );
      
      vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture2D(tDiffuse, vUv + dir * -0.5).xyz +
        texture2D(tDiffuse, vUv + dir * 0.5).xyz
      );
      
      float lumaB = dot(rgbB, luma);
      
      if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
        gl_FragColor = vec4(rgbA, 1.0);
      } else {
        gl_FragColor = vec4(rgbB, 1.0);
      }
    }
  `,

  /**
   * Bloom effect
   */
  bloom: `
    uniform sampler2D tDiffuse;
    uniform float threshold;
    uniform float intensity;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      
      // Extract bright areas
      float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
      vec3 bloom = color.rgb * smoothstep(threshold, threshold + 0.5, brightness);
      
      gl_FragColor = vec4(color.rgb + bloom * intensity, 1.0);
    }
  `,

  /**
   * Vignette effect
   */
  vignette: `
    uniform sampler2D tDiffuse;
    uniform float intensity;
    uniform float smoothness;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      
      vec2 uv = vUv * (1.0 - vUv.yx);
      float vignette = uv.x * uv.y * 15.0;
      vignette = pow(vignette, intensity * smoothness);
      
      gl_FragColor = vec4(color.rgb * vignette, 1.0);
    }
  `,

  /**
   * Chromatic aberration
   */
  chromaticAberration: `
    uniform sampler2D tDiffuse;
    uniform float amount;
    varying vec2 vUv;

    void main() {
      vec2 offset = amount * (vUv - 0.5);
      
      float r = texture2D(tDiffuse, vUv + offset).r;
      float g = texture2D(tDiffuse, vUv).g;
      float b = texture2D(tDiffuse, vUv - offset).b;
      
      gl_FragColor = vec4(r, g, b, 1.0);
    }
  `,

  /**
   * Film grain
   */
  filmGrain: `
    uniform sampler2D tDiffuse;
    uniform float time;
    uniform float intensity;
    varying vec2 vUv;

    float random(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      
      float grain = random(vUv + time) * intensity;
      
      gl_FragColor = vec4(color.rgb + grain - intensity * 0.5, 1.0);
    }
  `,
};
