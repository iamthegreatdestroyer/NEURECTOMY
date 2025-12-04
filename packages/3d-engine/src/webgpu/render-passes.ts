/**
 * Render Passes
 *
 * Pre-configured render pass implementations for common use cases:
 * - Forward rendering
 * - Shadow mapping
 * - Post-processing
 * - Outline/selection highlighting
 *
 * @module @neurectomy/3d-engine/webgpu/render-passes
 * @agents @ARCHITECT @CANVAS
 */

import type { ManagedTexture } from "./texture-manager";

// =============================================================================
// Types
// =============================================================================

export interface RenderPassConfig {
  colorAttachments: GPURenderPassColorAttachment[];
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  label?: string;
}

export interface ForwardPassConfig {
  colorView: GPUTextureView;
  depthView: GPUTextureView;
  clearColor?: GPUColor;
  clearDepth?: number;
  msaaView?: GPUTextureView;
}

export interface ShadowPassConfig {
  depthView: GPUTextureView;
  size: number;
}

export interface PostProcessPassConfig {
  inputTexture: GPUTextureView;
  outputView: GPUTextureView;
  effects: PostProcessEffect[];
}

export interface OutlinePassConfig {
  colorView: GPUTextureView;
  depthView: GPUTextureView;
  selectedMask: GPUTextureView;
  outlineColor?: GPUColor;
  outlineWidth?: number;
}

export interface PostProcessEffect {
  type:
    | "bloom"
    | "tonemap"
    | "fxaa"
    | "ssao"
    | "dof"
    | "chromatic-aberration"
    | "vignette";
  enabled: boolean;
  params?: Record<string, number>;
}

export interface RenderPassStats {
  totalPasses: number;
  colorAttachments: number;
  depthAttachments: number;
  msaaEnabled: boolean;
}

// =============================================================================
// Default Values
// =============================================================================

const DEFAULT_CLEAR_COLOR: GPUColor = { r: 0.05, g: 0.05, b: 0.08, a: 1.0 };
const DEFAULT_CLEAR_DEPTH = 1.0;
const DEFAULT_OUTLINE_COLOR: GPUColor = { r: 0.2, g: 0.6, b: 1.0, a: 1.0 };
const DEFAULT_OUTLINE_WIDTH = 2;

// =============================================================================
// RenderPassBuilder Class
// =============================================================================

/**
 * RenderPassBuilder - Fluent API for building render passes
 */
export class RenderPassBuilder {
  private colorAttachments: GPURenderPassColorAttachment[] = [];
  private depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  private label?: string;

  static create(): RenderPassBuilder {
    return new RenderPassBuilder();
  }

  /**
   * Add a color attachment
   */
  addColorAttachment(
    view: GPUTextureView,
    options: {
      clearValue?: GPUColor;
      loadOp?: GPULoadOp;
      storeOp?: GPUStoreOp;
      resolveTarget?: GPUTextureView;
    } = {}
  ): this {
    this.colorAttachments.push({
      view,
      clearValue: options.clearValue ?? DEFAULT_CLEAR_COLOR,
      loadOp: options.loadOp ?? "clear",
      storeOp: options.storeOp ?? "store",
      resolveTarget: options.resolveTarget,
    });
    return this;
  }

  /**
   * Set depth/stencil attachment
   */
  setDepthStencil(
    view: GPUTextureView,
    options: {
      depthClearValue?: number;
      depthLoadOp?: GPULoadOp;
      depthStoreOp?: GPUStoreOp;
      depthReadOnly?: boolean;
      stencilClearValue?: number;
      stencilLoadOp?: GPULoadOp;
      stencilStoreOp?: GPUStoreOp;
      stencilReadOnly?: boolean;
    } = {}
  ): this {
    this.depthStencilAttachment = {
      view,
      depthClearValue: options.depthClearValue ?? DEFAULT_CLEAR_DEPTH,
      depthLoadOp: options.depthLoadOp ?? "clear",
      depthStoreOp: options.depthStoreOp ?? "store",
      depthReadOnly: options.depthReadOnly ?? false,
      stencilClearValue: options.stencilClearValue ?? 0,
      stencilLoadOp: options.stencilLoadOp ?? "clear",
      stencilStoreOp: options.stencilStoreOp ?? "store",
      stencilReadOnly: options.stencilReadOnly ?? false,
    };
    return this;
  }

  /**
   * Set the pass label
   */
  setLabel(label: string): this {
    this.label = label;
    return this;
  }

  /**
   * Build the render pass descriptor
   */
  build(): GPURenderPassDescriptor {
    return {
      colorAttachments: this.colorAttachments,
      depthStencilAttachment: this.depthStencilAttachment,
      label: this.label,
    };
  }
}

// =============================================================================
// Pre-configured Render Passes
// =============================================================================

/**
 * Create a forward rendering pass
 *
 * Standard opaque geometry rendering with depth testing.
 */
export function createForwardPass(
  config: ForwardPassConfig
): GPURenderPassDescriptor {
  const builder = RenderPassBuilder.create().setLabel("forward-pass");

  // If MSAA is enabled, render to MSAA view and resolve to final view
  if (config.msaaView) {
    builder.addColorAttachment(config.msaaView, {
      clearValue: config.clearColor ?? DEFAULT_CLEAR_COLOR,
      resolveTarget: config.colorView,
    });
  } else {
    builder.addColorAttachment(config.colorView, {
      clearValue: config.clearColor ?? DEFAULT_CLEAR_COLOR,
    });
  }

  builder.setDepthStencil(config.depthView, {
    depthClearValue: config.clearDepth ?? DEFAULT_CLEAR_DEPTH,
  });

  return builder.build();
}

/**
 * Create a transparent geometry pass
 *
 * Renders transparent objects with depth read but no write.
 */
export function createTransparentPass(
  config: ForwardPassConfig
): GPURenderPassDescriptor {
  const builder = RenderPassBuilder.create().setLabel("transparent-pass");

  // Don't clear - preserve previous pass output
  builder.addColorAttachment(config.colorView, {
    loadOp: "load",
  });

  builder.setDepthStencil(config.depthView, {
    depthLoadOp: "load",
    depthStoreOp: "discard", // Transparent pass doesn't write depth
    depthReadOnly: true,
  });

  return builder.build();
}

/**
 * Create a shadow map rendering pass
 *
 * Depth-only pass for shadow map generation.
 */
export function createShadowPass(
  config: ShadowPassConfig
): GPURenderPassDescriptor {
  return {
    colorAttachments: [],
    depthStencilAttachment: {
      view: config.depthView,
      depthClearValue: DEFAULT_CLEAR_DEPTH,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
    label: "shadow-pass",
  };
}

/**
 * Create an outline/selection pass
 *
 * Renders selected object outlines using stencil buffer.
 */
export function createOutlinePass(
  config: OutlinePassConfig
): GPURenderPassDescriptor {
  return RenderPassBuilder.create()
    .setLabel("outline-pass")
    .addColorAttachment(config.colorView, {
      loadOp: "load", // Preserve scene
      storeOp: "store",
    })
    .setDepthStencil(config.depthView, {
      depthLoadOp: "load",
      depthStoreOp: "store",
      depthReadOnly: true,
    })
    .build();
}

/**
 * Create a post-processing pass
 *
 * Full-screen quad rendering for post-process effects.
 */
export function createPostProcessPass(
  outputView: GPUTextureView,
  label?: string
): GPURenderPassDescriptor {
  return RenderPassBuilder.create()
    .setLabel(label ?? "post-process-pass")
    .addColorAttachment(outputView, {
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    })
    .build();
}

/**
 * Create a UI/overlay pass
 *
 * Final pass for rendering UI elements on top of scene.
 */
export function createUIPass(
  colorView: GPUTextureView
): GPURenderPassDescriptor {
  return RenderPassBuilder.create()
    .setLabel("ui-pass")
    .addColorAttachment(colorView, {
      loadOp: "load", // Preserve scene + post-processing
      storeOp: "store",
    })
    .build();
}

/**
 * Create a G-Buffer pass for deferred rendering
 *
 * Multiple render targets: albedo, normal, position, metallic-roughness
 */
export function createGBufferPass(
  albedoView: GPUTextureView,
  normalView: GPUTextureView,
  positionView: GPUTextureView,
  materialView: GPUTextureView,
  depthView: GPUTextureView
): GPURenderPassDescriptor {
  return RenderPassBuilder.create()
    .setLabel("gbuffer-pass")
    .addColorAttachment(albedoView, {
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    })
    .addColorAttachment(normalView, {
      clearValue: { r: 0.5, g: 0.5, b: 1.0, a: 0 },
    })
    .addColorAttachment(positionView, {
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    })
    .addColorAttachment(materialView, {
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    })
    .setDepthStencil(depthView, {
      depthClearValue: DEFAULT_CLEAR_DEPTH,
    })
    .build();
}

/**
 * Create a lighting pass for deferred rendering
 */
export function createLightingPass(
  outputView: GPUTextureView
): GPURenderPassDescriptor {
  return RenderPassBuilder.create()
    .setLabel("lighting-pass")
    .addColorAttachment(outputView, {
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    })
    .build();
}

// =============================================================================
// RenderPassManager Class
// =============================================================================

/**
 * RenderPassManager - Manages render pass lifecycle and statistics
 */
export class RenderPassManager {
  private device: GPUDevice;
  private passes = new Map<string, GPURenderPassDescriptor>();
  private statistics = {
    totalPasses: 0,
    passExecutions: new Map<string, number>(),
  };

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Register a named render pass
   */
  register(name: string, descriptor: GPURenderPassDescriptor): void {
    this.passes.set(name, descriptor);
    this.statistics.passExecutions.set(name, 0);
  }

  /**
   * Get a registered render pass
   */
  get(name: string): GPURenderPassDescriptor | undefined {
    return this.passes.get(name);
  }

  /**
   * Begin a render pass on the encoder
   */
  beginPass(
    encoder: GPUCommandEncoder,
    name: string
  ): GPURenderPassEncoder | undefined {
    const descriptor = this.passes.get(name);
    if (!descriptor) {
      console.warn(`[RenderPassManager] Pass '${name}' not found`);
      return undefined;
    }

    this.statistics.totalPasses++;
    this.statistics.passExecutions.set(
      name,
      (this.statistics.passExecutions.get(name) ?? 0) + 1
    );

    return encoder.beginRenderPass(descriptor);
  }

  /**
   * Create a render pass encoder directly from descriptor
   */
  createPassEncoder(
    encoder: GPUCommandEncoder,
    descriptor: GPURenderPassDescriptor
  ): GPURenderPassEncoder {
    this.statistics.totalPasses++;
    return encoder.beginRenderPass(descriptor);
  }

  /**
   * Get pass statistics
   */
  getStats(): RenderPassStats {
    let colorAttachments = 0;
    let depthAttachments = 0;
    let msaaEnabled = false;

    for (const pass of this.passes.values()) {
      const colorAttachmentArray = Array.from(pass.colorAttachments);
      colorAttachments += colorAttachmentArray.length;
      if (pass.depthStencilAttachment) {
        depthAttachments++;
      }
      for (const attachment of colorAttachmentArray) {
        if (
          attachment &&
          "resolveTarget" in attachment &&
          attachment.resolveTarget
        ) {
          msaaEnabled = true;
        }
      }
    }

    return {
      totalPasses: this.passes.size,
      colorAttachments,
      depthAttachments,
      msaaEnabled,
    };
  }

  /**
   * Get execution statistics
   */
  getExecutionStats(): Map<string, number> {
    return new Map(this.statistics.passExecutions);
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.statistics.totalPasses = 0;
    this.statistics.passExecutions.clear();
    for (const name of this.passes.keys()) {
      this.statistics.passExecutions.set(name, 0);
    }
  }

  /**
   * Dispose all passes
   */
  dispose(): void {
    this.passes.clear();
    this.statistics.passExecutions.clear();
    console.log("[RenderPassManager] Disposed");
  }
}

// =============================================================================
// FrameGraph for Complex Rendering
// =============================================================================

export interface FrameGraphNode {
  name: string;
  pass: GPURenderPassDescriptor;
  inputs: string[];
  outputs: string[];
  execute: (encoder: GPURenderPassEncoder) => void;
}

/**
 * FrameGraph - Declarative render graph for complex multi-pass rendering
 *
 * Automatically handles resource dependencies and pass ordering.
 */
export class FrameGraph {
  private nodes: FrameGraphNode[] = [];
  private executionOrder: FrameGraphNode[] = [];
  private dirty = true;

  /**
   * Add a render pass node to the graph
   */
  addNode(node: FrameGraphNode): this {
    this.nodes.push(node);
    this.dirty = true;
    return this;
  }

  /**
   * Compile the frame graph - determine execution order
   */
  compile(): void {
    if (!this.dirty) return;

    // Topological sort based on dependencies
    const sorted: FrameGraphNode[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (node: FrameGraphNode) => {
      if (visited.has(node.name)) return;
      if (visiting.has(node.name)) {
        throw new Error(`Circular dependency detected at node: ${node.name}`);
      }

      visiting.add(node.name);

      // Find nodes that produce our inputs
      for (const input of node.inputs) {
        const producer = this.nodes.find((n) => n.outputs.includes(input));
        if (producer) {
          visit(producer);
        }
      }

      visiting.delete(node.name);
      visited.add(node.name);
      sorted.push(node);
    };

    for (const node of this.nodes) {
      visit(node);
    }

    this.executionOrder = sorted;
    this.dirty = false;
  }

  /**
   * Execute the frame graph
   */
  execute(device: GPUDevice): GPUCommandBuffer {
    this.compile();

    const encoder = device.createCommandEncoder({ label: "frame-graph" });

    for (const node of this.executionOrder) {
      const passEncoder = encoder.beginRenderPass(node.pass);
      node.execute(passEncoder);
      passEncoder.end();
    }

    return encoder.finish();
  }

  /**
   * Clear the graph
   */
  clear(): void {
    this.nodes = [];
    this.executionOrder = [];
    this.dirty = true;
  }

  /**
   * Get execution order (for debugging)
   */
  getExecutionOrder(): string[] {
    this.compile();
    return this.executionOrder.map((n) => n.name);
  }
}
