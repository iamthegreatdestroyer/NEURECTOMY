/**
 * Render Pipeline Cache
 * 
 * Caches GPU render and compute pipelines to minimize state changes
 * and reduce pipeline creation overhead.
 * 
 * @module @neurectomy/3d-engine/webgpu/pipeline-cache
 * @agents @ARCHITECT @VELOCITY
 */

// =============================================================================
// Types
// =============================================================================

export interface PipelineDescriptor {
  vertex: {
    module: GPUShaderModule;
    entryPoint: string;
    buffers?: GPUVertexBufferLayout[];
  };
  fragment?: {
    module: GPUShaderModule;
    entryPoint: string;
    targets: GPUColorTargetState[];
  };
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  layout?: GPUPipelineLayout | 'auto';
  label?: string;
}

export interface ComputePipelineDescriptor {
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
    constants?: Record<string, number>;
  };
  layout?: GPUPipelineLayout | 'auto';
  label?: string;
}

export interface PipelineCacheStats {
  renderPipelineCount: number;
  computePipelineCount: number;
  cacheHits: number;
  cacheMisses: number;
  hitRate: number;
}

interface CachedRenderPipeline {
  pipeline: GPURenderPipeline;
  descriptor: PipelineDescriptor;
  createdAt: number;
  lastUsed: number;
  useCount: number;
}

interface CachedComputePipeline {
  pipeline: GPUComputePipeline;
  descriptor: ComputePipelineDescriptor;
  createdAt: number;
  lastUsed: number;
  useCount: number;
}

// =============================================================================
// Default States
// =============================================================================

export const DEFAULT_PRIMITIVE_STATE: GPUPrimitiveState = {
  topology: 'triangle-list',
  frontFace: 'ccw',
  cullMode: 'back',
};

export const DEFAULT_DEPTH_STENCIL_STATE: GPUDepthStencilState = {
  depthWriteEnabled: true,
  depthCompare: 'less',
  format: 'depth24plus',
};

export const DEFAULT_MULTISAMPLE_STATE: GPUMultisampleState = {
  count: 1,
};

export const DEFAULT_COLOR_TARGET: GPUColorTargetState = {
  format: 'bgra8unorm',
  writeMask: GPUColorWrite.ALL,
};

export const DEFAULT_BLEND_STATE: GPUBlendState = {
  color: {
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
    operation: 'add',
  },
  alpha: {
    srcFactor: 'one',
    dstFactor: 'one-minus-src-alpha',
    operation: 'add',
  },
};

// =============================================================================
// PipelineCache Class
// =============================================================================

/**
 * PipelineCache - Manages GPU pipeline caching
 * 
 * Features:
 * - Automatic pipeline caching based on descriptor hashing
 * - LRU eviction for memory management
 * - Usage statistics and hit rate tracking
 * - Async pipeline creation support
 */
export class PipelineCache {
  private device: GPUDevice;
  private renderPipelines = new Map<string, CachedRenderPipeline>();
  private computePipelines = new Map<string, CachedComputePipeline>();
  private maxCacheSize: number;
  private cacheHits = 0;
  private cacheMisses = 0;

  constructor(device: GPUDevice, maxCacheSize = 100) {
    this.device = device;
    this.maxCacheSize = maxCacheSize;
  }

  /**
   * Get or create a render pipeline
   */
  getRenderPipeline(descriptor: PipelineDescriptor): GPURenderPipeline {
    const key = this.hashRenderPipelineDescriptor(descriptor);
    const cached = this.renderPipelines.get(key);

    if (cached) {
      this.cacheHits++;
      cached.lastUsed = Date.now();
      cached.useCount++;
      return cached.pipeline;
    }

    this.cacheMisses++;

    // Create new pipeline
    const pipeline = this.createRenderPipeline(descriptor);

    // Cache it
    this.renderPipelines.set(key, {
      pipeline,
      descriptor,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      useCount: 1,
    });

    // Evict if necessary
    this.evictIfNeeded();

    return pipeline;
  }

  /**
   * Get or create a render pipeline asynchronously
   */
  async getRenderPipelineAsync(descriptor: PipelineDescriptor): Promise<GPURenderPipeline> {
    const key = this.hashRenderPipelineDescriptor(descriptor);
    const cached = this.renderPipelines.get(key);

    if (cached) {
      this.cacheHits++;
      cached.lastUsed = Date.now();
      cached.useCount++;
      return cached.pipeline;
    }

    this.cacheMisses++;

    // Create new pipeline asynchronously
    const pipeline = await this.createRenderPipelineAsync(descriptor);

    // Cache it
    this.renderPipelines.set(key, {
      pipeline,
      descriptor,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      useCount: 1,
    });

    // Evict if necessary
    this.evictIfNeeded();

    return pipeline;
  }

  /**
   * Get or create a compute pipeline
   */
  getComputePipeline(descriptor: ComputePipelineDescriptor): GPUComputePipeline {
    const key = this.hashComputePipelineDescriptor(descriptor);
    const cached = this.computePipelines.get(key);

    if (cached) {
      this.cacheHits++;
      cached.lastUsed = Date.now();
      cached.useCount++;
      return cached.pipeline;
    }

    this.cacheMisses++;

    // Create new pipeline
    const pipeline = this.device.createComputePipeline({
      layout: descriptor.layout ?? 'auto',
      compute: descriptor.compute,
      label: descriptor.label,
    });

    // Cache it
    this.computePipelines.set(key, {
      pipeline,
      descriptor,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      useCount: 1,
    });

    // Evict if necessary
    this.evictIfNeeded();

    return pipeline;
  }

  /**
   * Create a render pipeline from descriptor
   */
  private createRenderPipeline(descriptor: PipelineDescriptor): GPURenderPipeline {
    return this.device.createRenderPipeline({
      layout: descriptor.layout ?? 'auto',
      vertex: descriptor.vertex,
      fragment: descriptor.fragment,
      primitive: descriptor.primitive ?? DEFAULT_PRIMITIVE_STATE,
      depthStencil: descriptor.depthStencil,
      multisample: descriptor.multisample ?? DEFAULT_MULTISAMPLE_STATE,
      label: descriptor.label,
    });
  }

  /**
   * Create a render pipeline asynchronously
   */
  private async createRenderPipelineAsync(descriptor: PipelineDescriptor): Promise<GPURenderPipeline> {
    return this.device.createRenderPipelineAsync({
      layout: descriptor.layout ?? 'auto',
      vertex: descriptor.vertex,
      fragment: descriptor.fragment,
      primitive: descriptor.primitive ?? DEFAULT_PRIMITIVE_STATE,
      depthStencil: descriptor.depthStencil,
      multisample: descriptor.multisample ?? DEFAULT_MULTISAMPLE_STATE,
      label: descriptor.label,
    });
  }

  /**
   * Hash a render pipeline descriptor for caching
   */
  private hashRenderPipelineDescriptor(descriptor: PipelineDescriptor): string {
    const parts: string[] = [];

    // Hash vertex stage
    parts.push(`v:${descriptor.vertex.entryPoint}`);
    if (descriptor.vertex.buffers) {
      for (const buffer of descriptor.vertex.buffers) {
        parts.push(`vb:${buffer.arrayStride}:${buffer.stepMode ?? 'vertex'}`);
        for (const attr of buffer.attributes as GPUVertexAttribute[]) {
          parts.push(`va:${attr.shaderLocation}:${attr.format}:${attr.offset}`);
        }
      }
    }

    // Hash fragment stage
    if (descriptor.fragment) {
      parts.push(`f:${descriptor.fragment.entryPoint}`);
      for (const target of descriptor.fragment.targets) {
        parts.push(`ft:${target.format}:${target.writeMask ?? GPUColorWrite.ALL}`);
        if (target.blend) {
          parts.push(`fb:${target.blend.color.operation}:${target.blend.alpha.operation}`);
        }
      }
    }

    // Hash primitive state
    const primitive = descriptor.primitive ?? DEFAULT_PRIMITIVE_STATE;
    parts.push(`p:${primitive.topology}:${primitive.cullMode}:${primitive.frontFace}`);

    // Hash depth stencil state
    if (descriptor.depthStencil) {
      parts.push(`d:${descriptor.depthStencil.format}:${descriptor.depthStencil.depthCompare}`);
    }

    // Hash multisample state
    const multisample = descriptor.multisample ?? DEFAULT_MULTISAMPLE_STATE;
    parts.push(`m:${multisample.count}`);

    return parts.join('|');
  }

  /**
   * Hash a compute pipeline descriptor for caching
   */
  private hashComputePipelineDescriptor(descriptor: ComputePipelineDescriptor): string {
    const parts: string[] = [];
    parts.push(`c:${descriptor.compute.entryPoint}`);
    if (descriptor.compute.constants) {
      for (const [key, value] of Object.entries(descriptor.compute.constants)) {
        parts.push(`cc:${key}:${value}`);
      }
    }
    return parts.join('|');
  }

  /**
   * Evict least recently used pipelines if cache is full
   */
  private evictIfNeeded(): void {
    const totalSize = this.renderPipelines.size + this.computePipelines.size;
    if (totalSize <= this.maxCacheSize) return;

    const toEvict = totalSize - this.maxCacheSize;
    const allEntries: Array<{ key: string; lastUsed: number; type: 'render' | 'compute' }> = [];

    for (const [key, cached] of this.renderPipelines) {
      allEntries.push({ key, lastUsed: cached.lastUsed, type: 'render' });
    }
    for (const [key, cached] of this.computePipelines) {
      allEntries.push({ key, lastUsed: cached.lastUsed, type: 'compute' });
    }

    // Sort by last used (oldest first)
    allEntries.sort((a, b) => a.lastUsed - b.lastUsed);

    // Evict oldest entries
    for (let i = 0; i < toEvict && i < allEntries.length; i++) {
      const entry = allEntries[i]!;
      if (entry.type === 'render') {
        this.renderPipelines.delete(entry.key);
      } else {
        this.computePipelines.delete(entry.key);
      }
    }
  }

  /**
   * Invalidate all cached pipelines
   */
  invalidateAll(): void {
    this.renderPipelines.clear();
    this.computePipelines.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }

  /**
   * Get cache statistics
   */
  getStats(): PipelineCacheStats {
    const total = this.cacheHits + this.cacheMisses;
    return {
      renderPipelineCount: this.renderPipelines.size,
      computePipelineCount: this.computePipelines.size,
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      hitRate: total > 0 ? this.cacheHits / total : 0,
    };
  }

  /**
   * Pre-warm the cache with common pipeline configurations
   */
  async warmup(descriptors: PipelineDescriptor[]): Promise<void> {
    const promises = descriptors.map(d => this.getRenderPipelineAsync(d));
    await Promise.all(promises);
    console.log(`[PipelineCache] Warmed up with ${descriptors.length} pipelines`);
  }

  /**
   * Dispose of the cache
   */
  dispose(): void {
    this.renderPipelines.clear();
    this.computePipelines.clear();
    console.log('[PipelineCache] Disposed');
  }
}

// =============================================================================
// Pipeline Builder Helper
// =============================================================================

/**
 * PipelineBuilder - Fluent API for building pipeline descriptors
 */
export class PipelineBuilder {
  private descriptor: Partial<PipelineDescriptor> = {};

  static create(): PipelineBuilder {
    return new PipelineBuilder();
  }

  vertex(module: GPUShaderModule, entryPoint: string = 'vs_main'): this {
    this.descriptor.vertex = { module, entryPoint };
    return this;
  }

  vertexBuffers(buffers: GPUVertexBufferLayout[]): this {
    if (this.descriptor.vertex) {
      this.descriptor.vertex.buffers = buffers;
    }
    return this;
  }

  fragment(module: GPUShaderModule, entryPoint: string = 'fs_main'): this {
    this.descriptor.fragment = {
      module,
      entryPoint,
      targets: [DEFAULT_COLOR_TARGET],
    };
    return this;
  }

  colorTarget(format: GPUTextureFormat, blend?: GPUBlendState): this {
    if (this.descriptor.fragment) {
      this.descriptor.fragment.targets = [{
        format,
        blend,
        writeMask: GPUColorWrite.ALL,
      }];
    }
    return this;
  }

  multipleColorTargets(targets: GPUColorTargetState[]): this {
    if (this.descriptor.fragment) {
      this.descriptor.fragment.targets = targets;
    }
    return this;
  }

  primitive(state: GPUPrimitiveState): this {
    this.descriptor.primitive = state;
    return this;
  }

  topology(topology: GPUPrimitiveTopology): this {
    this.descriptor.primitive = {
      ...this.descriptor.primitive,
      topology,
    };
    return this;
  }

  cullMode(cullMode: GPUCullMode): this {
    this.descriptor.primitive = {
      ...this.descriptor.primitive,
      cullMode,
    };
    return this;
  }

  depthStencil(state: GPUDepthStencilState): this {
    this.descriptor.depthStencil = state;
    return this;
  }

  depthTest(format: GPUTextureFormat = 'depth24plus', compare: GPUCompareFunction = 'less'): this {
    this.descriptor.depthStencil = {
      format,
      depthWriteEnabled: true,
      depthCompare: compare,
    };
    return this;
  }

  multisample(count: number = 4): this {
    this.descriptor.multisample = { count };
    return this;
  }

  layout(layout: GPUPipelineLayout | 'auto'): this {
    this.descriptor.layout = layout;
    return this;
  }

  label(label: string): this {
    this.descriptor.label = label;
    return this;
  }

  build(): PipelineDescriptor {
    if (!this.descriptor.vertex) {
      throw new Error('Vertex stage is required');
    }
    return this.descriptor as PipelineDescriptor;
  }
}
