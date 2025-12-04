/**
 * Materials Library
 * 
 * PBR materials optimized for agent visualization with animated effects,
 * holographic shaders, and status-based color schemes.
 * 
 * @module @neurectomy/3d-engine/three/materials
 * @agents @CANVAS @APEX
 */

import * as THREE from 'three';

// =============================================================================
// Types
// =============================================================================

export type AgentStatus = 
  | 'idle' 
  | 'running' 
  | 'success' 
  | 'error' 
  | 'pending' 
  | 'paused'
  | 'thinking';

export type AgentType = 
  | 'core'
  | 'memory'
  | 'tool'
  | 'prompt'
  | 'output'
  | 'router'
  | 'custom';

export interface AgentMaterialConfig {
  status?: AgentStatus;
  type?: AgentType;
  emissiveIntensity?: number;
  opacity?: number;
  wireframe?: boolean;
  selected?: boolean;
  hovered?: boolean;
}

export interface ConnectionMaterialConfig {
  active?: boolean;
  dataFlowing?: boolean;
  strength?: number;
  color?: THREE.Color | string;
}

export interface GridMaterialConfig {
  primaryColor?: THREE.Color | string;
  secondaryColor?: THREE.Color | string;
  gridSize?: number;
  fadeDistance?: number;
}

export interface HologramMaterialConfig {
  color?: THREE.Color | string;
  scanlineIntensity?: number;
  glitchIntensity?: number;
  noiseScale?: number;
}

// =============================================================================
// Color Palettes
// =============================================================================

export const STATUS_COLORS: Record<AgentStatus, THREE.Color> = {
  idle: new THREE.Color(0x4a5568),      // Gray
  running: new THREE.Color(0x3b82f6),   // Blue
  success: new THREE.Color(0x10b981),   // Green
  error: new THREE.Color(0xef4444),     // Red
  pending: new THREE.Color(0xf59e0b),   // Amber
  paused: new THREE.Color(0x6b7280),    // Gray
  thinking: new THREE.Color(0x8b5cf6),  // Purple
};

export const TYPE_COLORS: Record<AgentType, THREE.Color> = {
  core: new THREE.Color(0x60a5fa),      // Light blue
  memory: new THREE.Color(0xa78bfa),    // Purple
  tool: new THREE.Color(0x34d399),      // Green
  prompt: new THREE.Color(0xfbbf24),    // Yellow
  output: new THREE.Color(0xf87171),    // Red
  router: new THREE.Color(0x2dd4bf),    // Teal
  custom: new THREE.Color(0xf472b6),    // Pink
};

export const NEURECTOMY_PALETTE = {
  background: new THREE.Color(0x0a0a0f),
  grid: new THREE.Color(0x1e293b),
  gridAccent: new THREE.Color(0x334155),
  highlight: new THREE.Color(0x3b82f6),
  selection: new THREE.Color(0xfbbf24),
  connection: new THREE.Color(0x22d3ee),
  text: new THREE.Color(0xe2e8f0),
  success: new THREE.Color(0x10b981),
  warning: new THREE.Color(0xf59e0b),
  error: new THREE.Color(0xef4444),
};

// =============================================================================
// Shader Chunks
// =============================================================================

const COMMON_UNIFORMS = `
  uniform float time;
  uniform float selected;
  uniform float hovered;
`;

const PULSE_EFFECT = `
  float pulse = sin(time * 3.0) * 0.5 + 0.5;
  float selectedPulse = selected * pulse * 0.3;
  float hoveredPulse = hovered * pulse * 0.15;
`;

const FRESNEL_EFFECT = `
  vec3 viewDir = normalize(cameraPosition - worldPosition);
  float fresnel = pow(1.0 - max(dot(viewDir, normalWorld), 0.0), 3.0);
`;

const SCANLINE_EFFECT = `
  float scanline = sin(worldPosition.y * 50.0 + time * 10.0) * 0.5 + 0.5;
  scanline = pow(scanline, 10.0) * 0.1;
`;

// =============================================================================
// MaterialFactory Class
// =============================================================================

/**
 * MaterialFactory - Creates optimized materials for NEURECTOMY
 * 
 * Features:
 * - Agent visualization materials with status colors
 * - Connection line materials with flow animation
 * - Grid and environment materials
 * - Holographic/cyberpunk effects
 * - Selection and hover states
 */
export class MaterialFactory {
  private cache = new Map<string, THREE.Material>();
  private time = { value: 0 };

  constructor() {
    // Start time update
    this.startTimeUpdate();
  }

  private startTimeUpdate(): void {
    const animate = () => {
      this.time.value += 0.016;
      requestAnimationFrame(animate);
    };
    animate();
  }

  /**
   * Create agent node material
   */
  createAgentMaterial(config: AgentMaterialConfig = {}): THREE.MeshStandardMaterial {
    const status = config.status ?? 'idle';
    const type = config.type ?? 'core';
    const cacheKey = `agent-${status}-${type}-${config.selected}-${config.hovered}`;

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!.clone() as THREE.MeshStandardMaterial;
    }

    const baseColor = TYPE_COLORS[type];
    const emissiveColor = STATUS_COLORS[status];

    const material = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: emissiveColor,
      emissiveIntensity: config.emissiveIntensity ?? 0.3,
      metalness: 0.7,
      roughness: 0.3,
      transparent: config.opacity !== undefined && config.opacity < 1,
      opacity: config.opacity ?? 1,
      wireframe: config.wireframe ?? false,
    });

    // Add selection highlight
    if (config.selected) {
      material.emissive = NEURECTOMY_PALETTE.selection;
      material.emissiveIntensity = 0.6;
    } else if (config.hovered) {
      material.emissive = NEURECTOMY_PALETTE.highlight;
      material.emissiveIntensity = 0.4;
    }

    this.cache.set(cacheKey, material);
    return material.clone();
  }

  /**
   * Create animated agent material with shader
   */
  createAnimatedAgentMaterial(config: AgentMaterialConfig = {}): THREE.ShaderMaterial {
    const status = config.status ?? 'idle';
    const type = config.type ?? 'core';
    const baseColor = TYPE_COLORS[type];
    const statusColor = STATUS_COLORS[status];

    return new THREE.ShaderMaterial({
      uniforms: {
        time: this.time,
        baseColor: { value: baseColor },
        statusColor: { value: statusColor },
        selected: { value: config.selected ? 1.0 : 0.0 },
        hovered: { value: config.hovered ? 1.0 : 0.0 },
        emissiveIntensity: { value: config.emissiveIntensity ?? 0.3 },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec2 vUv;

        void main() {
          vNormal = normalize(normalMatrix * normal);
          vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        ${COMMON_UNIFORMS}
        uniform vec3 baseColor;
        uniform vec3 statusColor;
        uniform float emissiveIntensity;

        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec2 vUv;

        void main() {
          vec3 worldPosition = vWorldPosition;
          vec3 normalWorld = vNormal;

          // Fresnel effect
          vec3 viewDir = normalize(cameraPosition - worldPosition);
          float fresnel = pow(1.0 - max(dot(viewDir, normalWorld), 0.0), 3.0);

          // Pulse effect
          ${PULSE_EFFECT}

          // Base color with fresnel rim
          vec3 color = baseColor;
          color += statusColor * emissiveIntensity;
          color += vec3(fresnel) * 0.3;
          color += selectedPulse * vec3(1.0, 0.8, 0.2);
          color += hoveredPulse * vec3(0.2, 0.5, 1.0);

          // Scanline effect for running status
          float scanline = sin(worldPosition.y * 50.0 + time * 10.0) * 0.5 + 0.5;
          scanline = pow(scanline, 10.0) * 0.05;
          color += scanline * statusColor;

          gl_FragColor = vec4(color, 1.0);
        }
      `,
      transparent: true,
    });
  }

  /**
   * Create connection line material
   */
  createConnectionMaterial(config: ConnectionMaterialConfig = {}): THREE.ShaderMaterial {
    const baseColor = config.color 
      ? (config.color instanceof THREE.Color ? config.color : new THREE.Color(config.color))
      : NEURECTOMY_PALETTE.connection;

    return new THREE.ShaderMaterial({
      uniforms: {
        time: this.time,
        color: { value: baseColor },
        active: { value: config.active ? 1.0 : 0.0 },
        dataFlowing: { value: config.dataFlowing ? 1.0 : 0.0 },
        strength: { value: config.strength ?? 1.0 },
      },
      vertexShader: `
        attribute float lineDistance;
        varying float vLineDistance;
        varying vec3 vWorldPosition;

        void main() {
          vLineDistance = lineDistance;
          vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 color;
        uniform float active;
        uniform float dataFlowing;
        uniform float strength;

        varying float vLineDistance;
        varying vec3 vWorldPosition;

        void main() {
          vec3 finalColor = color;
          float alpha = 0.5 * strength;

          // Active glow
          if (active > 0.5) {
            alpha = 0.8;
            finalColor += vec3(0.2);
          }

          // Data flow animation
          if (dataFlowing > 0.5) {
            float flow = fract(vLineDistance * 0.1 - time * 2.0);
            float pulse = smoothstep(0.0, 0.1, flow) * smoothstep(0.3, 0.2, flow);
            finalColor += pulse * 0.5;
            alpha += pulse * 0.3;
          }

          // Dashed pattern for inactive
          if (active < 0.5) {
            float dash = step(0.5, fract(vLineDistance * 0.5));
            alpha *= dash * 0.5 + 0.5;
          }

          gl_FragColor = vec4(finalColor, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }

  /**
   * Create grid material
   */
  createGridMaterial(config: GridMaterialConfig = {}): THREE.ShaderMaterial {
    const primaryColor = config.primaryColor 
      ? (config.primaryColor instanceof THREE.Color ? config.primaryColor : new THREE.Color(config.primaryColor))
      : NEURECTOMY_PALETTE.grid;
    
    const secondaryColor = config.secondaryColor
      ? (config.secondaryColor instanceof THREE.Color ? config.secondaryColor : new THREE.Color(config.secondaryColor))
      : NEURECTOMY_PALETTE.gridAccent;

    return new THREE.ShaderMaterial({
      uniforms: {
        primaryColor: { value: primaryColor },
        secondaryColor: { value: secondaryColor },
        gridSize: { value: config.gridSize ?? 1.0 },
        fadeDistance: { value: config.fadeDistance ?? 100.0 },
        cameraPos: { value: new THREE.Vector3() },
      },
      vertexShader: `
        varying vec3 vWorldPosition;

        void main() {
          vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 primaryColor;
        uniform vec3 secondaryColor;
        uniform float gridSize;
        uniform float fadeDistance;
        uniform vec3 cameraPos;

        varying vec3 vWorldPosition;

        void main() {
          vec2 coord = vWorldPosition.xz / gridSize;
          vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
          float line = min(grid.x, grid.y);

          // Major grid lines
          vec2 majorCoord = vWorldPosition.xz / (gridSize * 10.0);
          vec2 majorGrid = abs(fract(majorCoord - 0.5) - 0.5) / fwidth(majorCoord);
          float majorLine = min(majorGrid.x, majorGrid.y);

          // Distance fade
          float dist = length(vWorldPosition.xz - cameraPos.xz);
          float fade = 1.0 - smoothstep(fadeDistance * 0.5, fadeDistance, dist);

          // Combine
          float minor = 1.0 - min(line, 1.0);
          float major = 1.0 - min(majorLine, 1.0);

          vec3 color = mix(primaryColor, secondaryColor, major);
          float alpha = max(minor * 0.3, major * 0.6) * fade;

          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  /**
   * Create hologram material
   */
  createHologramMaterial(config: HologramMaterialConfig = {}): THREE.ShaderMaterial {
    const color = config.color
      ? (config.color instanceof THREE.Color ? config.color : new THREE.Color(config.color))
      : NEURECTOMY_PALETTE.highlight;

    return new THREE.ShaderMaterial({
      uniforms: {
        time: this.time,
        color: { value: color },
        scanlineIntensity: { value: config.scanlineIntensity ?? 0.3 },
        glitchIntensity: { value: config.glitchIntensity ?? 0.1 },
        noiseScale: { value: config.noiseScale ?? 5.0 },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec2 vUv;

        uniform float time;
        uniform float glitchIntensity;

        void main() {
          vNormal = normalize(normalMatrix * normal);
          vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          vUv = uv;

          // Glitch offset
          vec3 pos = position;
          float glitch = step(0.98, sin(time * 50.0 + position.y * 10.0));
          pos.x += glitch * glitchIntensity * sin(time * 100.0);

          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 color;
        uniform float scanlineIntensity;
        uniform float glitchIntensity;
        uniform float noiseScale;

        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec2 vUv;

        // Simple noise function
        float noise(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {
          vec3 finalColor = color;

          // Fresnel edge glow
          vec3 viewDir = normalize(cameraPosition - vWorldPosition);
          float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.0);

          // Scanlines
          float scanline = sin(vWorldPosition.y * 100.0 + time * 5.0) * 0.5 + 0.5;
          scanline = pow(scanline, 4.0) * scanlineIntensity;

          // Noise
          float n = noise(vUv * noiseScale + time);
          
          // Glitch bands
          float glitchBand = step(0.99, sin(vUv.y * 50.0 + time * 20.0)) * glitchIntensity;

          // Combine
          float alpha = fresnel * 0.8 + 0.2;
          alpha += scanline;
          alpha = clamp(alpha, 0.0, 1.0);

          finalColor += fresnel * 0.5;
          finalColor += scanline * color;
          finalColor += n * 0.05;
          finalColor += glitchBand * vec3(1.0);

          gl_FragColor = vec4(finalColor, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
  }

  /**
   * Create glass/transparent material
   */
  createGlassMaterial(color: THREE.Color | string = '#ffffff', opacity: number = 0.3): THREE.MeshPhysicalMaterial {
    return new THREE.MeshPhysicalMaterial({
      color: typeof color === 'string' ? new THREE.Color(color) : color,
      transmission: 0.9,
      roughness: 0.1,
      metalness: 0,
      ior: 1.5,
      thickness: 0.5,
      transparent: true,
      opacity,
    });
  }

  /**
   * Create selection outline material
   */
  createOutlineMaterial(color: THREE.Color | string = NEURECTOMY_PALETTE.selection): THREE.ShaderMaterial {
    const outlineColor = typeof color === 'string' ? new THREE.Color(color) : color;

    return new THREE.ShaderMaterial({
      uniforms: {
        time: this.time,
        outlineColor: { value: outlineColor },
        outlineWidth: { value: 0.03 },
      },
      vertexShader: `
        uniform float outlineWidth;
        uniform float time;

        void main() {
          vec3 pos = position + normal * outlineWidth;
          // Animate outline
          pos += normal * sin(time * 5.0) * 0.005;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 outlineColor;
        uniform float time;

        void main() {
          float pulse = sin(time * 3.0) * 0.2 + 0.8;
          gl_FragColor = vec4(outlineColor * pulse, 1.0);
        }
      `,
      side: THREE.BackSide,
      transparent: true,
    });
  }

  /**
   * Create point cloud material
   */
  createPointMaterial(
    color: THREE.Color | string = '#ffffff',
    size: number = 0.1
  ): THREE.PointsMaterial {
    return new THREE.PointsMaterial({
      color: typeof color === 'string' ? new THREE.Color(color) : color,
      size,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
  }

  /**
   * Update time uniform (call in render loop)
   */
  updateTime(deltaTime: number): void {
    this.time.value += deltaTime;
  }

  /**
   * Clear material cache
   */
  clearCache(): void {
    for (const material of this.cache.values()) {
      material.dispose();
    }
    this.cache.clear();
  }

  /**
   * Dispose all materials
   */
  dispose(): void {
    this.clearCache();
    console.log('[MaterialFactory] Disposed');
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let materialFactory: MaterialFactory | null = null;

export function getMaterialFactory(): MaterialFactory {
  if (!materialFactory) {
    materialFactory = new MaterialFactory();
  }
  return materialFactory;
}
