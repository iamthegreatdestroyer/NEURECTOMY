/**
 * WGSL Shader Compiler & Manager
 * 
 * Provides shader compilation, caching, hot-reloading, and comprehensive
 * error reporting for WebGPU shader development.
 * 
 * @module @neurectomy/3d-engine/webgpu/shader-compiler
 * @agents @CORE @AXIOM
 */

import type { ShaderHandle } from '../core/types';

// =============================================================================
// Types
// =============================================================================

export interface ShaderSource {
  vertex?: string;
  fragment?: string;
  compute?: string;
}

export interface ShaderCompilationResult {
  success: boolean;
  module: GPUShaderModule | null;
  errors: ShaderError[];
  warnings: ShaderWarning[];
  compilationTime: number;
}

export interface ShaderError {
  type: 'error';
  message: string;
  lineNumber?: number;
  columnNumber?: number;
  lineContent?: string;
  suggestion?: string;
}

export interface ShaderWarning {
  type: 'warning';
  message: string;
  lineNumber?: number;
  columnNumber?: number;
}

export interface ShaderDefines {
  [key: string]: string | number | boolean;
}

export interface CompiledShader {
  handle: ShaderHandle;
  module: GPUShaderModule;
  source: string;
  defines: ShaderDefines;
  vertexEntryPoint?: string;
  fragmentEntryPoint?: string;
  computeEntryPoint?: string;
  compiledAt: number;
  compilationTime: number;
}

export interface ShaderCompilerOptions {
  enableHotReload?: boolean;
  validateOnCompile?: boolean;
  stripComments?: boolean;
  minify?: boolean;
  defines?: ShaderDefines;
}

// =============================================================================
// Shader Templates
// =============================================================================

/**
 * Built-in shader library for common rendering operations
 */
export const SHADER_LIBRARY = {
  // Common structures and functions
  COMMON: `
// Common type definitions
struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) color: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) color: vec4f,
}

struct CameraUniforms {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  viewProjectionMatrix: mat4x4f,
  inverseViewMatrix: mat4x4f,
  cameraPosition: vec3f,
  near: f32,
  far: f32,
  fov: f32,
  aspectRatio: f32,
  time: f32,
}

struct ModelUniforms {
  modelMatrix: mat4x4f,
  normalMatrix: mat3x3f,
}

struct MaterialUniforms {
  color: vec4f,
  emissive: vec3f,
  emissiveIntensity: f32,
  metalness: f32,
  roughness: f32,
  opacity: f32,
  _padding: f32,
}

// Utility functions
fn linearToSRGB(color: vec3f) -> vec3f {
  let cutoff = color < vec3f(0.0031308);
  let higher = vec3f(1.055) * pow(color, vec3f(1.0 / 2.4)) - vec3f(0.055);
  let lower = color * vec3f(12.92);
  return select(higher, lower, cutoff);
}

fn sRGBToLinear(color: vec3f) -> vec3f {
  let cutoff = color < vec3f(0.04045);
  let higher = pow((color + vec3f(0.055)) / vec3f(1.055), vec3f(2.4));
  let lower = color / vec3f(12.92);
  return select(higher, lower, cutoff);
}
`,

  // Standard PBR vertex shader
  STANDARD_VERTEX: `
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  let worldPosition = model.modelMatrix * vec4f(input.position, 1.0);
  output.position = camera.viewProjectionMatrix * worldPosition;
  output.worldPosition = worldPosition.xyz;
  output.normal = model.normalMatrix * input.normal;
  output.uv = input.uv;
  output.color = input.color;
  
  return output;
}
`,

  // Standard PBR fragment shader
  STANDARD_FRAGMENT: `
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function
fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  
  let num = a2;
  var denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;
  
  return num / denom;
}

// Schlick-GGX geometry function
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  
  let num = NdotV;
  let denom = NdotV * (1.0 - k) + k;
  
  return num / denom;
}

// Smith's method for geometry function
fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx2 = geometrySchlickGGX(NdotV, roughness);
  let ggx1 = geometrySchlickGGX(NdotL, roughness);
  
  return ggx1 * ggx2;
}

// Fresnel-Schlick approximation
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let N = normalize(input.normal);
  let V = normalize(camera.cameraPosition - input.worldPosition);
  
  // Material properties
  let albedo = material.color.rgb * input.color.rgb;
  let metalness = material.metalness;
  let roughness = material.roughness;
  
  // Calculate F0 (reflectance at normal incidence)
  let F0 = mix(vec3f(0.04), albedo, metalness);
  
  // Simple directional light for now
  let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
  let lightColor = vec3f(1.0, 0.98, 0.95);
  let lightIntensity = 2.0;
  
  let L = lightDir;
  let H = normalize(V + L);
  
  // Cook-Torrance BRDF
  let NDF = distributionGGX(N, H, roughness);
  let G = geometrySmith(N, V, L, roughness);
  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  
  let numerator = NDF * G * F;
  let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  let specular = numerator / denominator;
  
  let kS = F;
  var kD = vec3f(1.0) - kS;
  kD *= 1.0 - metalness;
  
  let NdotL = max(dot(N, L), 0.0);
  let Lo = (kD * albedo / PI + specular) * lightColor * lightIntensity * NdotL;
  
  // Ambient
  let ambient = vec3f(0.03) * albedo;
  
  // Emissive
  let emissive = material.emissive * material.emissiveIntensity;
  
  var color = ambient + Lo + emissive;
  
  // Tone mapping (Reinhard)
  color = color / (color + vec3f(1.0));
  
  // Gamma correction
  color = linearToSRGB(color);
  
  return vec4f(color, material.opacity * material.color.a);
}
`,

  // Line shader for connections
  LINE_VERTEX: `
struct LineVertexInput {
  @location(0) position: vec3f,
  @location(1) color: vec4f,
}

struct LineVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(input: LineVertexInput) -> LineVertexOutput {
  var output: LineVertexOutput;
  output.position = camera.viewProjectionMatrix * vec4f(input.position, 1.0);
  output.color = input.color;
  return output;
}
`,

  LINE_FRAGMENT: `
@fragment
fn fs_main(input: LineVertexOutput) -> @location(0) vec4f {
  return input.color;
}
`,

  // Grid shader
  GRID_VERTEX: `
struct GridVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) nearPoint: vec3f,
  @location(1) farPoint: vec3f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Fullscreen quad positions
const gridPlane = array<vec3f, 6>(
  vec3f(1, 1, 0), vec3f(-1, -1, 0), vec3f(-1, 1, 0),
  vec3f(-1, -1, 0), vec3f(1, 1, 0), vec3f(1, -1, 0)
);

fn unprojectPoint(x: f32, y: f32, z: f32) -> vec3f {
  let unprojectedPoint = camera.inverseViewMatrix * inverse(camera.projectionMatrix) * vec4f(x, y, z, 1.0);
  return unprojectedPoint.xyz / unprojectedPoint.w;
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> GridVertexOutput {
  var output: GridVertexOutput;
  let p = gridPlane[vertexIndex];
  output.position = vec4f(p, 1.0);
  output.nearPoint = unprojectPoint(p.x, p.y, 0.0);
  output.farPoint = unprojectPoint(p.x, p.y, 1.0);
  return output;
}
`,

  GRID_FRAGMENT: `
struct GridFragmentOutput {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

fn grid(fragPos3D: vec3f, scale: f32) -> vec4f {
  let coord = fragPos3D.xz * scale;
  let derivative = fwidth(coord);
  let grid = abs(fract(coord - 0.5) - 0.5) / derivative;
  let line = min(grid.x, grid.y);
  let minimumz = min(derivative.y, 1.0);
  let minimumx = min(derivative.x, 1.0);
  var color = vec4f(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));
  
  // Z axis (blue)
  if (fragPos3D.x > -0.1 * minimumx && fragPos3D.x < 0.1 * minimumx) {
    color = vec4f(0.0, 0.0, 1.0, color.a);
  }
  // X axis (red)
  if (fragPos3D.z > -0.1 * minimumz && fragPos3D.z < 0.1 * minimumz) {
    color = vec4f(1.0, 0.0, 0.0, color.a);
  }
  
  return color;
}

fn computeDepth(pos: vec3f) -> f32 {
  let clipSpacePos = camera.viewProjectionMatrix * vec4f(pos, 1.0);
  return clipSpacePos.z / clipSpacePos.w;
}

fn computeLinearDepth(pos: vec3f) -> f32 {
  let clipSpacePos = camera.viewProjectionMatrix * vec4f(pos, 1.0);
  let clipSpaceDepth = clipSpacePos.z / clipSpacePos.w * 2.0 - 1.0;
  let linearDepth = (2.0 * camera.near * camera.far) / (camera.far + camera.near - clipSpaceDepth * (camera.far - camera.near));
  return linearDepth / camera.far;
}

@fragment
fn fs_main(input: GridVertexOutput) -> GridFragmentOutput {
  var output: GridFragmentOutput;
  
  let t = -input.nearPoint.y / (input.farPoint.y - input.nearPoint.y);
  let fragPos3D = input.nearPoint + t * (input.farPoint - input.nearPoint);
  
  output.depth = computeDepth(fragPos3D);
  
  let linearDepth = computeLinearDepth(fragPos3D);
  let fading = max(0.0, (0.5 - linearDepth));
  
  // Multiple grid scales
  let grid1 = grid(fragPos3D, 1.0);
  let grid2 = grid(fragPos3D, 0.1);
  
  var gridColor = grid1 * f32(t > 0.0);
  gridColor += grid2 * f32(t > 0.0) * 0.5;
  
  output.color = vec4f(gridColor.rgb, gridColor.a * fading);
  
  return output;
}
`,
} as const;

// =============================================================================
// ShaderCompiler Class
// =============================================================================

/**
 * ShaderCompiler - Manages WGSL shader compilation and caching
 */
export class ShaderCompiler {
  private device: GPUDevice;
  private cache = new Map<string, CompiledShader>();
  private options: Required<ShaderCompilerOptions>;
  private nextHandleId = 0;

  // Hot reload
  private watchedShaders = new Map<string, { path: string; lastModified: number }>();

  constructor(device: GPUDevice, options: ShaderCompilerOptions = {}) {
    this.device = device;
    this.options = {
      enableHotReload: options.enableHotReload ?? false,
      validateOnCompile: options.validateOnCompile ?? true,
      stripComments: options.stripComments ?? false,
      minify: options.minify ?? false,
      defines: options.defines ?? {},
    };
  }

  /**
   * Compile a shader from WGSL source
   */
  async compile(
    source: string,
    options?: {
      label?: string;
      defines?: ShaderDefines;
      vertexEntryPoint?: string;
      fragmentEntryPoint?: string;
      computeEntryPoint?: string;
    }
  ): Promise<ShaderCompilationResult> {
    const startTime = performance.now();

    // Apply defines
    const defines = { ...this.options.defines, ...options?.defines };
    const processedSource = this.preprocessShader(source, defines);

    // Check cache
    const cacheKey = this.generateCacheKey(processedSource, defines);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return {
        success: true,
        module: cached.module,
        errors: [],
        warnings: [],
        compilationTime: 0,
      };
    }

    try {
      // Validate source before compilation
      if (this.options.validateOnCompile) {
        const validationResult = this.validateShader(processedSource);
        if (!validationResult.valid) {
          return {
            success: false,
            module: null,
            errors: validationResult.errors,
            warnings: validationResult.warnings,
            compilationTime: performance.now() - startTime,
          };
        }
      }

      // Compile shader module
      const module = this.device.createShaderModule({
        label: options?.label ?? 'Compiled Shader',
        code: processedSource,
      });

      // Get compilation info (async)
      const compilationInfo = await module.getCompilationInfo();
      const errors: ShaderError[] = [];
      const warnings: ShaderWarning[] = [];

      for (const message of compilationInfo.messages) {
        if (message.type === 'error') {
          errors.push({
            type: 'error',
            message: message.message,
            lineNumber: message.lineNum,
            columnNumber: message.linePos,
          });
        } else if (message.type === 'warning') {
          warnings.push({
            type: 'warning',
            message: message.message,
            lineNumber: message.lineNum,
            columnNumber: message.linePos,
          });
        }
      }

      if (errors.length > 0) {
        return {
          success: false,
          module: null,
          errors,
          warnings,
          compilationTime: performance.now() - startTime,
        };
      }

      // Cache the compiled shader
      const handle = this.createHandle();
      const compiledShader: CompiledShader = {
        handle,
        module,
        source: processedSource,
        defines,
        vertexEntryPoint: options?.vertexEntryPoint,
        fragmentEntryPoint: options?.fragmentEntryPoint,
        computeEntryPoint: options?.computeEntryPoint,
        compiledAt: Date.now(),
        compilationTime: performance.now() - startTime,
      };

      this.cache.set(cacheKey, compiledShader);

      return {
        success: true,
        module,
        errors: [],
        warnings,
        compilationTime: compiledShader.compilationTime,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      // Parse error message for line/column info
      const parsedError = this.parseCompilationError(errorMessage, processedSource);

      return {
        success: false,
        module: null,
        errors: [parsedError],
        warnings: [],
        compilationTime: performance.now() - startTime,
      };
    }
  }

  /**
   * Compile standard shader with common includes
   */
  async compileStandard(
    vertexSource?: string,
    fragmentSource?: string,
    options?: {
      label?: string;
      defines?: ShaderDefines;
    }
  ): Promise<ShaderCompilationResult> {
    const fullSource = [
      SHADER_LIBRARY.COMMON,
      vertexSource ?? SHADER_LIBRARY.STANDARD_VERTEX,
      fragmentSource ?? SHADER_LIBRARY.STANDARD_FRAGMENT,
    ].join('\n');

    return this.compile(fullSource, {
      ...options,
      vertexEntryPoint: 'vs_main',
      fragmentEntryPoint: 'fs_main',
    });
  }

  /**
   * Preprocess shader source with defines
   */
  private preprocessShader(source: string, defines: ShaderDefines): string {
    let processedSource = source;

    // Replace define placeholders
    for (const [key, value] of Object.entries(defines)) {
      const regex = new RegExp(`\\$\\{${key}\\}|#${key}#`, 'g');
      processedSource = processedSource.replace(regex, String(value));
    }

    // Add defines as constants at the top
    const defineConstants = Object.entries(defines)
      .map(([key, value]) => {
        if (typeof value === 'boolean') {
          return `const ${key}: bool = ${value};`;
        } else if (typeof value === 'number') {
          return Number.isInteger(value)
            ? `const ${key}: i32 = ${value};`
            : `const ${key}: f32 = ${value};`;
        }
        return `// ${key} = ${value}`;
      })
      .join('\n');

    if (defineConstants) {
      processedSource = `// Defines\n${defineConstants}\n\n${processedSource}`;
    }

    // Strip comments if requested
    if (this.options.stripComments) {
      processedSource = processedSource
        .replace(/\/\/.*$/gm, '')
        .replace(/\/\*[\s\S]*?\*\//g, '');
    }

    // Minify if requested
    if (this.options.minify) {
      processedSource = processedSource
        .replace(/\s+/g, ' ')
        .replace(/\s*([{};,()=+\-*/<>])\s*/g, '$1');
    }

    return processedSource;
  }

  /**
   * Validate shader source without compiling
   */
  private validateShader(source: string): { valid: boolean; errors: ShaderError[]; warnings: ShaderWarning[] } {
    const errors: ShaderError[] = [];
    const warnings: ShaderWarning[] = [];

    // Basic structural validation
    const lines = source.split('\n');

    // Check for common issues
    let braceCount = 0;
    let parenCount = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const lineNum = i + 1;

      // Track brace/paren balance
      braceCount += (line.match(/{/g) || []).length - (line.match(/}/g) || []).length;
      parenCount += (line.match(/\(/g) || []).length - (line.match(/\)/g) || []).length;

      // Check for common mistakes
      if (line.includes('gl_') && !line.includes('//')) {
        warnings.push({
          type: 'warning',
          message: 'GLSL-style built-in detected. Use WGSL equivalents instead.',
          lineNumber: lineNum,
        });
      }

      if (line.includes('vec3(') || line.includes('vec4(')) {
        warnings.push({
          type: 'warning',
          message: 'Use vec3f/vec4f instead of vec3/vec4 in WGSL.',
          lineNumber: lineNum,
        });
      }
    }

    if (braceCount !== 0) {
      errors.push({
        type: 'error',
        message: `Unbalanced braces: ${braceCount > 0 ? 'missing }' : 'extra }'}`,
        suggestion: 'Check your brace matching',
      });
    }

    if (parenCount !== 0) {
      errors.push({
        type: 'error',
        message: `Unbalanced parentheses: ${parenCount > 0 ? 'missing )' : 'extra )'}`,
        suggestion: 'Check your parenthesis matching',
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Parse compilation error message for better diagnostics
   */
  private parseCompilationError(message: string, source: string): ShaderError {
    // Try to extract line/column from error message
    const lineMatch = message.match(/line (\d+)/i);
    const columnMatch = message.match(/column (\d+)/i);

    const lineNumber = lineMatch ? parseInt(lineMatch[1]!, 10) : undefined;
    const columnNumber = columnMatch ? parseInt(columnMatch[1]!, 10) : undefined;

    let lineContent: string | undefined;
    if (lineNumber !== undefined) {
      const lines = source.split('\n');
      lineContent = lines[lineNumber - 1];
    }

    return {
      type: 'error',
      message,
      lineNumber,
      columnNumber,
      lineContent,
    };
  }

  /**
   * Generate cache key for shader
   */
  private generateCacheKey(source: string, defines: ShaderDefines): string {
    const definesStr = JSON.stringify(defines);
    return `${this.hashString(source)}_${this.hashString(definesStr)}`;
  }

  /**
   * Simple string hash function
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  /**
   * Create a new shader handle
   */
  private createHandle(): ShaderHandle {
    return {
      __brand: 'ShaderHandle',
      id: `shader_${this.nextHandleId++}`,
    };
  }

  /**
   * Clear shader cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; entries: string[] } {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys()),
    };
  }
}
