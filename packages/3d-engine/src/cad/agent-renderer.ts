/**
 * Agent Renderer - CAD-Quality Agent Visualization
 *
 * High-fidelity rendering system for AI agent nodes with multiple visualization modes.
 * Supports schematic, realistic, blueprint, and x-ray rendering styles.
 *
 * @module @neurectomy/3d-engine/cad/agent-renderer
 * @agents @CANVAS @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 * @step Step 3 - CAD Visualization System
 */

import * as THREE from "three";
import {
  CSS2DRenderer,
  CSS2DObject,
} from "three/examples/jsm/renderers/CSS2DRenderer";

// =============================================================================
// TYPES & INTERFACES
// =============================================================================

/**
 * Agent visualization modes
 */
export enum AgentRenderMode {
  /** Clean schematic view */
  SCHEMATIC = "schematic",
  /** Photorealistic rendering */
  REALISTIC = "realistic",
  /** Technical blueprint style */
  BLUEPRINT = "blueprint",
  /** X-ray internal view */
  XRAY = "xray",
  /** Wireframe mesh */
  WIREFRAME = "wireframe",
  /** Holographic projection */
  HOLOGRAPHIC = "holographic",
  /** Heat map based on metrics */
  HEATMAP = "heatmap",
}

/**
 * Agent node types for specialized rendering
 */
export enum AgentNodeType {
  /** Core processing agent */
  PROCESSOR = "processor",
  /** Memory/storage agent */
  MEMORY = "memory",
  /** Input/output interface */
  INTERFACE = "interface",
  /** Communication hub */
  HUB = "hub",
  /** Decision making agent */
  DECISION = "decision",
  /** Transformation/processing agent */
  TRANSFORM = "transform",
  /** External service connector */
  CONNECTOR = "connector",
  /** Monitoring/observability agent */
  MONITOR = "monitor",
  /** Custom agent type */
  CUSTOM = "custom",
}

/**
 * Agent status for visual indicators
 */
export enum AgentNodeStatus {
  IDLE = "idle",
  ACTIVE = "active",
  PROCESSING = "processing",
  ERROR = "error",
  WARNING = "warning",
  DISABLED = "disabled",
  INITIALIZING = "initializing",
  TERMINATING = "terminating",
}

/**
 * Agent data for rendering
 */
export interface AgentRenderData {
  id: string;
  name: string;
  type: AgentNodeType;
  status: AgentNodeStatus;
  position: THREE.Vector3;
  rotation?: THREE.Euler;
  scale?: THREE.Vector3;

  // Visual customization
  color?: THREE.Color;
  icon?: string;
  label?: string;
  sublabel?: string;

  // Metrics for visualization
  metrics?: {
    cpu?: number;
    memory?: number;
    throughput?: number;
    latency?: number;
    errorRate?: number;
    custom?: Record<string, number>;
  };

  // Internal structure
  ports?: AgentPort[];
  subcomponents?: AgentSubcomponent[];

  // Metadata
  metadata?: Record<string, unknown>;
}

/**
 * Agent connection port
 */
export interface AgentPort {
  id: string;
  name: string;
  type: "input" | "output" | "bidirectional";
  direction: THREE.Vector3;
  connected: boolean;
  dataType?: string;
}

/**
 * Internal subcomponent for x-ray view
 */
export interface AgentSubcomponent {
  id: string;
  name: string;
  type: string;
  position: THREE.Vector3;
  scale: number;
  active: boolean;
}

/**
 * Agent renderer configuration
 */
export interface AgentRendererConfig {
  mode: AgentRenderMode;
  showLabels: boolean;
  showPorts: boolean;
  showMetrics: boolean;
  showConnections: boolean;
  animateActive: boolean;
  qualityLevel: "low" | "medium" | "high" | "ultra";

  // Colors
  colors: {
    idle: THREE.Color;
    active: THREE.Color;
    error: THREE.Color;
    warning: THREE.Color;
    disabled: THREE.Color;
    processing: THREE.Color;
    highlight: THREE.Color;
    selection: THREE.Color;
  };

  // Geometry settings
  geometry: {
    baseSize: number;
    portSize: number;
    labelOffset: number;
    outlineWidth: number;
  };

  // Animation settings
  animation: {
    pulseSpeed: number;
    rotationSpeed: number;
    transitionDuration: number;
  };
}

/**
 * Rendered agent mesh container
 */
export interface RenderedAgent {
  id: string;
  data: AgentRenderData;
  group: THREE.Group;
  body: THREE.Mesh;
  outline?: THREE.Mesh;
  ports: Map<string, THREE.Mesh>;
  label?: CSS2DObject;
  metrics?: CSS2DObject;
  animations: AgentAnimation[];
}

/**
 * Animation controller
 */
export interface AgentAnimation {
  type: "pulse" | "rotate" | "glow" | "shake" | "float";
  active: boolean;
  speed: number;
  phase: number;
}

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

const DEFAULT_CONFIG: AgentRendererConfig = {
  mode: AgentRenderMode.SCHEMATIC,
  showLabels: true,
  showPorts: true,
  showMetrics: false,
  showConnections: true,
  animateActive: true,
  qualityLevel: "high",

  colors: {
    idle: new THREE.Color(0x4a90d9),
    active: new THREE.Color(0x50c878),
    error: new THREE.Color(0xe74c3c),
    warning: new THREE.Color(0xf39c12),
    disabled: new THREE.Color(0x7f8c8d),
    processing: new THREE.Color(0x9b59b6),
    highlight: new THREE.Color(0xffffff),
    selection: new THREE.Color(0x00ffff),
  },

  geometry: {
    baseSize: 1.0,
    portSize: 0.15,
    labelOffset: 1.5,
    outlineWidth: 0.05,
  },

  animation: {
    pulseSpeed: 2.0,
    rotationSpeed: 0.5,
    transitionDuration: 0.3,
  },
};

// =============================================================================
// GEOMETRY GENERATORS
// =============================================================================

/**
 * Generate agent geometry based on type
 */
class AgentGeometryFactory {
  private geometryCache: Map<string, THREE.BufferGeometry> = new Map();
  private qualityLevel: "low" | "medium" | "high" | "ultra";

  constructor(qualityLevel: "low" | "medium" | "high" | "ultra" = "high") {
    this.qualityLevel = qualityLevel;
  }

  private getSegments(): { radial: number; height: number } {
    switch (this.qualityLevel) {
      case "low":
        return { radial: 8, height: 1 };
      case "medium":
        return { radial: 16, height: 2 };
      case "high":
        return { radial: 32, height: 4 };
      case "ultra":
        return { radial: 64, height: 8 };
    }
  }

  /**
   * Get or create geometry for agent type
   */
  getGeometry(type: AgentNodeType, size: number = 1): THREE.BufferGeometry {
    const cacheKey = `${type}-${size}-${this.qualityLevel}`;

    if (this.geometryCache.has(cacheKey)) {
      return this.geometryCache.get(cacheKey)!;
    }

    const geometry = this.createGeometry(type, size);
    this.geometryCache.set(cacheKey, geometry);
    return geometry;
  }

  private createGeometry(
    type: AgentNodeType,
    size: number
  ): THREE.BufferGeometry {
    const seg = this.getSegments();

    switch (type) {
      case AgentNodeType.PROCESSOR:
        return this.createProcessorGeometry(size, seg);

      case AgentNodeType.MEMORY:
        return this.createMemoryGeometry(size, seg);

      case AgentNodeType.INTERFACE:
        return this.createInterfaceGeometry(size, seg);

      case AgentNodeType.HUB:
        return this.createHubGeometry(size, seg);

      case AgentNodeType.DECISION:
        return this.createDecisionGeometry(size, seg);

      case AgentNodeType.TRANSFORM:
        return this.createTransformGeometry(size, seg);

      case AgentNodeType.CONNECTOR:
        return this.createConnectorGeometry(size, seg);

      case AgentNodeType.MONITOR:
        return this.createMonitorGeometry(size, seg);

      default:
        return this.createDefaultGeometry(size, seg);
    }
  }

  private createProcessorGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // CPU-like box with beveled edges
    const shape = new THREE.Shape();
    const bevel = size * 0.1;

    shape.moveTo(-size / 2 + bevel, -size / 2);
    shape.lineTo(size / 2 - bevel, -size / 2);
    shape.quadraticCurveTo(size / 2, -size / 2, size / 2, -size / 2 + bevel);
    shape.lineTo(size / 2, size / 2 - bevel);
    shape.quadraticCurveTo(size / 2, size / 2, size / 2 - bevel, size / 2);
    shape.lineTo(-size / 2 + bevel, size / 2);
    shape.quadraticCurveTo(-size / 2, size / 2, -size / 2, size / 2 - bevel);
    shape.lineTo(-size / 2, -size / 2 + bevel);
    shape.quadraticCurveTo(-size / 2, -size / 2, -size / 2 + bevel, -size / 2);

    return new THREE.ExtrudeGeometry(shape, {
      depth: size * 0.3,
      bevelEnabled: true,
      bevelThickness: size * 0.02,
      bevelSize: size * 0.02,
      bevelSegments: seg.height,
    });
  }

  private createMemoryGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Cylinder with flat ends - database-like
    const geometry = new THREE.CylinderGeometry(
      size * 0.6,
      size * 0.6,
      size * 0.8,
      seg.radial,
      seg.height
    );

    // Rotate to stand upright
    geometry.rotateX(Math.PI / 2);

    return geometry;
  }

  private createInterfaceGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Arrow-like shape pointing forward
    const shape = new THREE.Shape();

    shape.moveTo(0, size * 0.5);
    shape.lineTo(size * 0.3, size * 0.2);
    shape.lineTo(size * 0.15, size * 0.2);
    shape.lineTo(size * 0.15, -size * 0.5);
    shape.lineTo(-size * 0.15, -size * 0.5);
    shape.lineTo(-size * 0.15, size * 0.2);
    shape.lineTo(-size * 0.3, size * 0.2);
    shape.closePath();

    return new THREE.ExtrudeGeometry(shape, {
      depth: size * 0.2,
      bevelEnabled: true,
      bevelThickness: size * 0.02,
      bevelSize: size * 0.02,
      bevelSegments: seg.height,
    });
  }

  private createHubGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Octahedron - communication hub
    return new THREE.OctahedronGeometry(
      size * 0.6,
      Math.min(seg.radial / 8, 2)
    );
  }

  private createDecisionGeometry(
    size: number,
    _seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Diamond shape - decision point
    const geometry = new THREE.OctahedronGeometry(size * 0.5, 0);

    // Scale to make it more diamond-like
    geometry.scale(1, 1.5, 1);

    return geometry;
  }

  private createTransformGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Gear-like torus
    return new THREE.TorusGeometry(
      size * 0.4,
      size * 0.15,
      seg.radial / 2,
      seg.radial
    );
  }

  private createConnectorGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Plug-like shape
    const geometry = new THREE.CapsuleGeometry(
      size * 0.3,
      size * 0.5,
      Math.max(4, seg.radial / 4),
      seg.radial
    );

    geometry.rotateX(Math.PI / 2);

    return geometry;
  }

  private createMonitorGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Eye-like shape - monitoring
    return new THREE.SphereGeometry(size * 0.5, seg.radial, seg.radial / 2);
  }

  private createDefaultGeometry(
    size: number,
    seg: { radial: number; height: number }
  ): THREE.BufferGeometry {
    // Default rounded box
    return new THREE.BoxGeometry(
      size,
      size,
      size,
      seg.height,
      seg.height,
      seg.height
    );
  }

  /**
   * Create port geometry
   */
  createPortGeometry(size: number): THREE.BufferGeometry {
    const seg = this.getSegments();
    return new THREE.SphereGeometry(size, seg.radial / 2, seg.radial / 4);
  }

  /**
   * Create outline geometry from base geometry
   */
  createOutlineGeometry(
    baseGeometry: THREE.BufferGeometry,
    thickness: number
  ): THREE.BufferGeometry {
    // Clone and scale up for outline effect
    const outlineGeometry = baseGeometry.clone();

    // Compute vertex normals if not present
    outlineGeometry.computeVertexNormals();

    const positions = outlineGeometry.getAttribute("position");
    const normals = outlineGeometry.getAttribute("normal");

    if (positions && normals) {
      for (let i = 0; i < positions.count; i++) {
        positions.setXYZ(
          i,
          positions.getX(i) + normals.getX(i) * thickness,
          positions.getY(i) + normals.getY(i) * thickness,
          positions.getZ(i) + normals.getZ(i) * thickness
        );
      }

      positions.needsUpdate = true;
    }

    return outlineGeometry;
  }

  /**
   * Dispose all cached geometries
   */
  dispose(): void {
    for (const geometry of this.geometryCache.values()) {
      geometry.dispose();
    }
    this.geometryCache.clear();
  }
}

// =============================================================================
// MATERIAL GENERATORS
// =============================================================================

/**
 * Generate materials for different render modes
 */
class AgentMaterialFactory {
  private materialCache: Map<string, THREE.Material> = new Map();
  private config: AgentRendererConfig;

  constructor(config: AgentRendererConfig) {
    this.config = config;
  }

  /**
   * Update configuration
   */
  setConfig(config: AgentRendererConfig): void {
    this.config = config;
    // Clear cache when config changes
    this.dispose();
  }

  /**
   * Get material for agent based on status and mode
   */
  getMaterial(status: AgentNodeStatus, mode: AgentRenderMode): THREE.Material {
    const cacheKey = `${status}-${mode}`;

    if (this.materialCache.has(cacheKey)) {
      return this.materialCache.get(cacheKey)!.clone();
    }

    const material = this.createMaterial(status, mode);
    this.materialCache.set(cacheKey, material);
    return material.clone();
  }

  private getStatusColor(status: AgentNodeStatus): THREE.Color {
    switch (status) {
      case AgentNodeStatus.IDLE:
        return this.config.colors.idle;
      case AgentNodeStatus.ACTIVE:
        return this.config.colors.active;
      case AgentNodeStatus.PROCESSING:
        return this.config.colors.processing;
      case AgentNodeStatus.ERROR:
        return this.config.colors.error;
      case AgentNodeStatus.WARNING:
        return this.config.colors.warning;
      case AgentNodeStatus.DISABLED:
        return this.config.colors.disabled;
      default:
        return this.config.colors.idle;
    }
  }

  private createMaterial(
    status: AgentNodeStatus,
    mode: AgentRenderMode
  ): THREE.Material {
    const color = this.getStatusColor(status);

    switch (mode) {
      case AgentRenderMode.SCHEMATIC:
        return this.createSchematicMaterial(color);

      case AgentRenderMode.REALISTIC:
        return this.createRealisticMaterial(color);

      case AgentRenderMode.BLUEPRINT:
        return this.createBlueprintMaterial(color);

      case AgentRenderMode.XRAY:
        return this.createXrayMaterial(color);

      case AgentRenderMode.WIREFRAME:
        return this.createWireframeMaterial(color);

      case AgentRenderMode.HOLOGRAPHIC:
        return this.createHolographicMaterial(color);

      case AgentRenderMode.HEATMAP:
        return this.createHeatmapMaterial(color);

      default:
        return this.createSchematicMaterial(color);
    }
  }

  private createSchematicMaterial(
    color: THREE.Color
  ): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color,
      metalness: 0.3,
      roughness: 0.7,
      flatShading: false,
    });
  }

  private createRealisticMaterial(
    color: THREE.Color
  ): THREE.MeshPhysicalMaterial {
    return new THREE.MeshPhysicalMaterial({
      color,
      metalness: 0.8,
      roughness: 0.2,
      clearcoat: 0.5,
      clearcoatRoughness: 0.3,
      reflectivity: 0.5,
    });
  }

  private createBlueprintMaterial(_color: THREE.Color): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(0x00ffff) },
        uGridSize: { value: 0.1 },
        uLineWidth: { value: 0.02 },
        uTime: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uGridSize;
        uniform float uLineWidth;
        uniform float uTime;
        
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          // Grid pattern
          vec2 grid = abs(fract(vPosition.xy / uGridSize - 0.5) - 0.5) / fwidth(vPosition.xy / uGridSize);
          float gridLine = min(grid.x, grid.y);
          float gridAlpha = 1.0 - min(gridLine, 1.0);
          
          // Edge detection
          float edge = 1.0 - pow(abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 0.5);
          
          // Combine
          float alpha = max(gridAlpha * 0.3, edge * 0.8);
          
          // Scanline effect
          float scanline = sin(vPosition.y * 50.0 + uTime * 2.0) * 0.5 + 0.5;
          alpha *= 0.8 + scanline * 0.2;
          
          gl_FragColor = vec4(uColor, alpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
    });
  }

  private createXrayMaterial(color: THREE.Color): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: color },
        uOpacity: { value: 0.6 },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vViewPosition = -mvPosition.xyz;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uOpacity;
        
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        
        void main() {
          vec3 normal = normalize(vNormal);
          vec3 viewDir = normalize(vViewPosition);
          
          // Fresnel effect for x-ray look
          float fresnel = pow(1.0 - abs(dot(normal, viewDir)), 2.0);
          
          gl_FragColor = vec4(uColor, fresnel * uOpacity);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }

  private createWireframeMaterial(color: THREE.Color): THREE.MeshBasicMaterial {
    return new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
      wireframeLinewidth: 1,
    });
  }

  private createHolographicMaterial(color: THREE.Color): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: color },
        uTime: { value: 0 },
        uScanlineSpeed: { value: 1.0 },
        uGlitchIntensity: { value: 0.1 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        void main() {
          vUv = uv;
          vPosition = position;
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uTime;
        uniform float uScanlineSpeed;
        uniform float uGlitchIntensity;
        
        varying vec2 vUv;
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        float random(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        void main() {
          // Scanlines
          float scanline = sin(vPosition.y * 30.0 - uTime * uScanlineSpeed * 3.0) * 0.5 + 0.5;
          scanline = pow(scanline, 4.0);
          
          // Horizontal glitch lines
          float glitch = step(0.98, random(vec2(floor(vPosition.y * 20.0), uTime * 10.0)));
          
          // Edge glow
          float edge = 1.0 - pow(abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 0.3);
          
          // Combine effects
          vec3 finalColor = uColor;
          finalColor += vec3(scanline * 0.1);
          finalColor += vec3(glitch * uGlitchIntensity);
          
          float alpha = edge * 0.8 + 0.2;
          alpha += scanline * 0.1;
          
          gl_FragColor = vec4(finalColor, alpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }

  private createHeatmapMaterial(_baseColor: THREE.Color): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uValue: { value: 0.5 },
        uMinColor: { value: new THREE.Color(0x0000ff) },
        uMidColor: { value: new THREE.Color(0x00ff00) },
        uMaxColor: { value: new THREE.Color(0xff0000) },
      },
      vertexShader: `
        varying vec3 vNormal;
        
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uValue;
        uniform vec3 uMinColor;
        uniform vec3 uMidColor;
        uniform vec3 uMaxColor;
        
        varying vec3 vNormal;
        
        void main() {
          vec3 color;
          if (uValue < 0.5) {
            color = mix(uMinColor, uMidColor, uValue * 2.0);
          } else {
            color = mix(uMidColor, uMaxColor, (uValue - 0.5) * 2.0);
          }
          
          // Simple lighting
          float light = dot(vNormal, normalize(vec3(1.0, 1.0, 1.0))) * 0.5 + 0.5;
          
          gl_FragColor = vec4(color * light, 1.0);
        }
      `,
    });
  }

  /**
   * Get outline material
   */
  getOutlineMaterial(color: THREE.Color): THREE.MeshBasicMaterial {
    return new THREE.MeshBasicMaterial({
      color,
      side: THREE.BackSide,
    });
  }

  /**
   * Get port material
   */
  getPortMaterial(
    type: "input" | "output" | "bidirectional",
    connected: boolean
  ): THREE.MeshBasicMaterial {
    let color: THREE.Color;

    switch (type) {
      case "input":
        color = new THREE.Color(0x00ff88);
        break;
      case "output":
        color = new THREE.Color(0xff8800);
        break;
      case "bidirectional":
        color = new THREE.Color(0x8800ff);
        break;
    }

    if (!connected) {
      color.multiplyScalar(0.5);
    }

    return new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: connected ? 1.0 : 0.5,
    });
  }

  /**
   * Dispose all cached materials
   */
  dispose(): void {
    for (const material of this.materialCache.values()) {
      material.dispose();
    }
    this.materialCache.clear();
  }
}

// =============================================================================
// MAIN AGENT RENDERER
// =============================================================================

/**
 * High-fidelity agent renderer with CAD-quality visualization
 */
export class AgentRenderer {
  private config: AgentRendererConfig;
  private geometryFactory: AgentGeometryFactory;
  private materialFactory: AgentMaterialFactory;

  private agents: Map<string, RenderedAgent> = new Map();
  private scene: THREE.Scene;
  private labelRenderer: CSS2DRenderer | null = null;

  private time: number = 0;
  private selectedAgentId: string | null = null;
  private highlightedAgentIds: Set<string> = new Set();

  // Event emitter
  private eventTarget: EventTarget = new EventTarget();

  constructor(scene: THREE.Scene, config: Partial<AgentRendererConfig> = {}) {
    this.scene = scene;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.geometryFactory = new AgentGeometryFactory(this.config.qualityLevel);
    this.materialFactory = new AgentMaterialFactory(this.config);
  }

  // ============================================
  // CONFIGURATION
  // ============================================

  /**
   * Update renderer configuration
   */
  setConfig(config: Partial<AgentRendererConfig>): void {
    this.config = { ...this.config, ...config };
    this.materialFactory.setConfig(this.config);

    // Re-render all agents with new config
    for (const agent of this.agents.values()) {
      this.updateAgentVisuals(agent);
    }
  }

  /**
   * Set render mode
   */
  setMode(mode: AgentRenderMode): void {
    if (this.config.mode === mode) return;

    this.config.mode = mode;

    // Update all agent materials
    for (const agent of this.agents.values()) {
      this.updateAgentMaterial(agent);
    }

    this.emitEvent("modeChanged", { mode });
  }

  /**
   * Setup label renderer for 2D overlays
   */
  setupLabelRenderer(container: HTMLElement): CSS2DRenderer {
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(container.clientWidth, container.clientHeight);
    this.labelRenderer.domElement.style.position = "absolute";
    this.labelRenderer.domElement.style.top = "0";
    this.labelRenderer.domElement.style.left = "0";
    this.labelRenderer.domElement.style.pointerEvents = "none";
    container.appendChild(this.labelRenderer.domElement);

    return this.labelRenderer;
  }

  // ============================================
  // AGENT MANAGEMENT
  // ============================================

  /**
   * Add an agent to the scene
   */
  addAgent(data: AgentRenderData): RenderedAgent {
    if (this.agents.has(data.id)) {
      console.warn(`Agent ${data.id} already exists, updating instead`);
      return this.updateAgent(data.id, data);
    }

    // Create agent group
    const group = new THREE.Group();
    group.position.copy(data.position);

    if (data.rotation) {
      group.rotation.copy(data.rotation);
    }

    if (data.scale) {
      group.scale.copy(data.scale);
    }

    group.userData = { agentId: data.id, type: "agent" };

    // Create body mesh
    const geometry = this.geometryFactory.getGeometry(
      data.type,
      this.config.geometry.baseSize
    );
    const material = this.materialFactory.getMaterial(
      data.status,
      this.config.mode
    );
    const body = new THREE.Mesh(geometry, material);
    body.userData = { agentId: data.id, part: "body" };
    group.add(body);

    // Create outline
    let outline: THREE.Mesh | undefined;
    if (
      this.config.mode !== AgentRenderMode.WIREFRAME &&
      this.config.mode !== AgentRenderMode.XRAY
    ) {
      const outlineGeometry = this.geometryFactory.createOutlineGeometry(
        geometry,
        this.config.geometry.outlineWidth
      );
      const outlineMaterial = this.materialFactory.getOutlineMaterial(
        data.color ?? this.config.colors.idle
      );
      outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
      outline.visible = false; // Only show on hover/select
      outline.userData = { agentId: data.id, part: "outline" };
      group.add(outline);
    }

    // Create ports
    const ports = new Map<string, THREE.Mesh>();
    if (this.config.showPorts && data.ports) {
      for (const portData of data.ports) {
        const portMesh = this.createPortMesh(portData);
        portMesh.userData = {
          agentId: data.id,
          portId: portData.id,
          part: "port",
        };
        group.add(portMesh);
        ports.set(portData.id, portMesh);
      }
    }

    // Create label
    let label: CSS2DObject | undefined;
    if (this.config.showLabels && (data.label || data.name)) {
      label = this.createLabel(data);
      group.add(label);
    }

    // Create metrics display
    let metrics: CSS2DObject | undefined;
    if (this.config.showMetrics && data.metrics) {
      metrics = this.createMetricsDisplay(data);
      group.add(metrics);
    }

    // Setup animations
    const animations: AgentAnimation[] = [];
    if (this.config.animateActive && data.status === AgentNodeStatus.ACTIVE) {
      animations.push({
        type: "pulse",
        active: true,
        speed: this.config.animation.pulseSpeed,
        phase: Math.random() * Math.PI * 2,
      });
    }

    if (data.status === AgentNodeStatus.PROCESSING) {
      animations.push({
        type: "rotate",
        active: true,
        speed: this.config.animation.rotationSpeed,
        phase: 0,
      });
    }

    // Add to scene
    this.scene.add(group);

    // Store rendered agent
    const renderedAgent: RenderedAgent = {
      id: data.id,
      data,
      group,
      body,
      outline,
      ports,
      label,
      metrics,
      animations,
    };

    this.agents.set(data.id, renderedAgent);
    this.emitEvent("agentAdded", { agentId: data.id, agent: renderedAgent });

    return renderedAgent;
  }

  /**
   * Update an existing agent
   */
  updateAgent(agentId: string, data: Partial<AgentRenderData>): RenderedAgent {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }

    // Update data
    Object.assign(agent.data, data);

    // Update transform
    if (data.position) {
      agent.group.position.copy(data.position);
    }

    if (data.rotation) {
      agent.group.rotation.copy(data.rotation);
    }

    if (data.scale) {
      agent.group.scale.copy(data.scale);
    }

    // Update material if status changed
    if (data.status !== undefined) {
      this.updateAgentMaterial(agent);
      this.updateAgentAnimations(agent);
    }

    // Update label
    if (data.label !== undefined || data.name !== undefined) {
      if (agent.label) {
        agent.group.remove(agent.label);
      }

      if (this.config.showLabels) {
        agent.label = this.createLabel(agent.data);
        agent.group.add(agent.label);
      }
    }

    // Update metrics
    if (data.metrics !== undefined) {
      if (agent.metrics) {
        agent.group.remove(agent.metrics);
      }

      if (this.config.showMetrics) {
        agent.metrics = this.createMetricsDisplay(agent.data);
        agent.group.add(agent.metrics);
      }
    }

    this.emitEvent("agentUpdated", { agentId, agent });

    return agent;
  }

  /**
   * Remove an agent from the scene
   */
  removeAgent(agentId: string): boolean {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    // Remove from scene
    this.scene.remove(agent.group);

    // Dispose geometry and materials
    agent.body.geometry.dispose();
    (agent.body.material as THREE.Material).dispose();

    if (agent.outline) {
      agent.outline.geometry.dispose();
      (agent.outline.material as THREE.Material).dispose();
    }

    for (const portMesh of agent.ports.values()) {
      portMesh.geometry.dispose();
      (portMesh.material as THREE.Material).dispose();
    }

    // Remove from map
    this.agents.delete(agentId);

    // Clear selection/highlight if needed
    if (this.selectedAgentId === agentId) {
      this.selectedAgentId = null;
    }
    this.highlightedAgentIds.delete(agentId);

    this.emitEvent("agentRemoved", { agentId });

    return true;
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): RenderedAgent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get all agents
   */
  getAllAgents(): RenderedAgent[] {
    return Array.from(this.agents.values());
  }

  // ============================================
  // SELECTION & HIGHLIGHTING
  // ============================================

  /**
   * Select an agent
   */
  selectAgent(agentId: string | null): void {
    // Deselect previous
    if (this.selectedAgentId) {
      const prevAgent = this.agents.get(this.selectedAgentId);
      if (prevAgent?.outline) {
        prevAgent.outline.visible = this.highlightedAgentIds.has(
          this.selectedAgentId
        );
        (prevAgent.outline.material as THREE.MeshBasicMaterial).color.copy(
          this.config.colors.highlight
        );
      }
    }

    this.selectedAgentId = agentId;

    // Select new
    if (agentId) {
      const agent = this.agents.get(agentId);
      if (agent?.outline) {
        agent.outline.visible = true;
        (agent.outline.material as THREE.MeshBasicMaterial).color.copy(
          this.config.colors.selection
        );
      }
    }

    this.emitEvent("selectionChanged", { agentId });
  }

  /**
   * Highlight an agent
   */
  highlightAgent(agentId: string, highlighted: boolean): void {
    const agent = this.agents.get(agentId);
    if (!agent) return;

    if (highlighted) {
      this.highlightedAgentIds.add(agentId);
      if (agent.outline && agentId !== this.selectedAgentId) {
        agent.outline.visible = true;
        (agent.outline.material as THREE.MeshBasicMaterial).color.copy(
          this.config.colors.highlight
        );
      }
    } else {
      this.highlightedAgentIds.delete(agentId);
      if (agent.outline && agentId !== this.selectedAgentId) {
        agent.outline.visible = false;
      }
    }

    this.emitEvent("highlightChanged", { agentId, highlighted });
  }

  /**
   * Get selected agent ID
   */
  getSelectedAgentId(): string | null {
    return this.selectedAgentId;
  }

  // ============================================
  // ANIMATION & UPDATE
  // ============================================

  /**
   * Update animation frame
   */
  update(deltaTime: number): void {
    this.time += deltaTime;

    // Update animated materials
    for (const agent of this.agents.values()) {
      // Update shader uniforms
      const material = agent.body.material;
      if (material instanceof THREE.ShaderMaterial) {
        if (material.uniforms.uTime) {
          material.uniforms.uTime.value = this.time;
        }
      }

      // Run animations
      for (const animation of agent.animations) {
        if (!animation.active) continue;

        switch (animation.type) {
          case "pulse":
            this.animatePulse(agent, animation, deltaTime);
            break;

          case "rotate":
            this.animateRotate(agent, animation, deltaTime);
            break;

          case "glow":
            this.animateGlow(agent, animation, deltaTime);
            break;

          case "shake":
            this.animateShake(agent, animation, deltaTime);
            break;

          case "float":
            this.animateFloat(agent, animation, deltaTime);
            break;
        }

        animation.phase += animation.speed * deltaTime;
      }
    }
  }

  private animatePulse(
    agent: RenderedAgent,
    animation: AgentAnimation,
    _deltaTime: number
  ): void {
    const scale = 1.0 + Math.sin(animation.phase) * 0.05;
    agent.body.scale.setScalar(scale);
  }

  private animateRotate(
    agent: RenderedAgent,
    animation: AgentAnimation,
    deltaTime: number
  ): void {
    agent.body.rotation.y += animation.speed * deltaTime;
  }

  private animateGlow(
    agent: RenderedAgent,
    animation: AgentAnimation,
    _deltaTime: number
  ): void {
    const material = agent.body.material;
    if (material instanceof THREE.MeshStandardMaterial) {
      material.emissiveIntensity = 0.5 + Math.sin(animation.phase) * 0.3;
    }
  }

  private animateShake(
    agent: RenderedAgent,
    animation: AgentAnimation,
    _deltaTime: number
  ): void {
    const intensity = 0.02;
    agent.body.position.x = Math.sin(animation.phase * 10) * intensity;
    agent.body.position.z = Math.cos(animation.phase * 10) * intensity;
  }

  private animateFloat(
    agent: RenderedAgent,
    animation: AgentAnimation,
    _deltaTime: number
  ): void {
    agent.body.position.y = Math.sin(animation.phase) * 0.1;
  }

  /**
   * Render labels (call after main render)
   */
  renderLabels(camera: THREE.Camera): void {
    if (this.labelRenderer) {
      this.labelRenderer.render(this.scene, camera);
    }
  }

  /**
   * Resize label renderer
   */
  resizeLabelRenderer(width: number, height: number): void {
    if (this.labelRenderer) {
      this.labelRenderer.setSize(width, height);
    }
  }

  // ============================================
  // HELPER METHODS
  // ============================================

  private createPortMesh(portData: AgentPort): THREE.Mesh {
    const geometry = this.geometryFactory.createPortGeometry(
      this.config.geometry.portSize
    );
    const material = this.materialFactory.getPortMaterial(
      portData.type,
      portData.connected
    );

    const mesh = new THREE.Mesh(geometry, material);

    // Position port on agent surface
    const offset = this.config.geometry.baseSize * 0.6;
    mesh.position.copy(portData.direction.clone().multiplyScalar(offset));

    return mesh;
  }

  private createLabel(data: AgentRenderData): CSS2DObject {
    const div = document.createElement("div");
    div.className = "agent-label";
    div.style.cssText = `
      color: white;
      font-family: 'Roboto Mono', monospace;
      font-size: 12px;
      text-align: center;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      white-space: nowrap;
      pointer-events: none;
    `;

    div.innerHTML = `
      <div style="font-weight: bold;">${data.label || data.name}</div>
      ${data.sublabel ? `<div style="font-size: 10px; opacity: 0.7;">${data.sublabel}</div>` : ""}
    `;

    const label = new CSS2DObject(div);
    label.position.set(0, this.config.geometry.labelOffset, 0);

    return label;
  }

  private createMetricsDisplay(data: AgentRenderData): CSS2DObject {
    const div = document.createElement("div");
    div.className = "agent-metrics";
    div.style.cssText = `
      color: #00ffff;
      font-family: 'Roboto Mono', monospace;
      font-size: 10px;
      background: rgba(0, 20, 40, 0.9);
      padding: 6px 10px;
      border-radius: 4px;
      border: 1px solid #00ffff40;
      pointer-events: none;
    `;

    const metrics = data.metrics!;
    const rows: string[] = [];

    if (metrics.cpu !== undefined) {
      rows.push(`CPU: ${(metrics.cpu * 100).toFixed(1)}%`);
    }
    if (metrics.memory !== undefined) {
      rows.push(`MEM: ${(metrics.memory * 100).toFixed(1)}%`);
    }
    if (metrics.throughput !== undefined) {
      rows.push(`THR: ${metrics.throughput.toFixed(0)}/s`);
    }
    if (metrics.latency !== undefined) {
      rows.push(`LAT: ${metrics.latency.toFixed(1)}ms`);
    }

    div.innerHTML = rows.join("<br>");

    const display = new CSS2DObject(div);
    display.position.set(1.5, 0, 0);

    return display;
  }

  private updateAgentVisuals(agent: RenderedAgent): void {
    this.updateAgentMaterial(agent);

    // Update label visibility
    if (agent.label) {
      agent.label.visible = this.config.showLabels;
    }

    // Update metrics visibility
    if (agent.metrics) {
      agent.metrics.visible = this.config.showMetrics;
    }

    // Update port visibility
    for (const portMesh of agent.ports.values()) {
      portMesh.visible = this.config.showPorts;
    }
  }

  private updateAgentMaterial(agent: RenderedAgent): void {
    // Dispose old material
    (agent.body.material as THREE.Material).dispose();

    // Create new material
    agent.body.material = this.materialFactory.getMaterial(
      agent.data.status,
      this.config.mode
    );
  }

  private updateAgentAnimations(agent: RenderedAgent): void {
    // Clear existing animations
    agent.animations.length = 0;

    if (!this.config.animateActive) return;

    // Add status-based animations
    switch (agent.data.status) {
      case AgentNodeStatus.ACTIVE:
        agent.animations.push({
          type: "pulse",
          active: true,
          speed: this.config.animation.pulseSpeed,
          phase: Math.random() * Math.PI * 2,
        });
        break;

      case AgentNodeStatus.PROCESSING:
        agent.animations.push({
          type: "rotate",
          active: true,
          speed: this.config.animation.rotationSpeed,
          phase: 0,
        });
        break;

      case AgentNodeStatus.ERROR:
        agent.animations.push({
          type: "shake",
          active: true,
          speed: 10,
          phase: 0,
        });
        break;
    }
  }

  // ============================================
  // EVENT HANDLING
  // ============================================

  private emitEvent(type: string, detail: unknown): void {
    this.eventTarget.dispatchEvent(new CustomEvent(type, { detail }));
  }

  /**
   * Add event listener
   */
  addEventListener(type: string, listener: EventListener): void {
    this.eventTarget.addEventListener(type, listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(type: string, listener: EventListener): void {
    this.eventTarget.removeEventListener(type, listener);
  }

  // ============================================
  // CLEANUP
  // ============================================

  /**
   * Dispose renderer and all resources
   */
  dispose(): void {
    // Remove all agents
    for (const agentId of this.agents.keys()) {
      this.removeAgent(agentId);
    }

    // Dispose factories
    this.geometryFactory.dispose();
    this.materialFactory.dispose();

    // Remove label renderer
    if (this.labelRenderer) {
      this.labelRenderer.domElement.remove();
      this.labelRenderer = null;
    }

    this.emitEvent("disposed", {});
  }
}

export default AgentRenderer;
