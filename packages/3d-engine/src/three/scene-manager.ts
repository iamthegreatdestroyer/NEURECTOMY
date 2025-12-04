/**
 * Scene Graph Engine
 * 
 * High-performance scene management optimized for large agent architectures,
 * with spatial indexing, instancing, and efficient updates.
 * 
 * @module @neurectomy/3d-engine/three/scene-manager
 * @agents @APEX @CANVAS
 */

import * as THREE from 'three';
import { CameraSystem, type CameraConfig } from './camera-system';
import { WebGPUBridge, type WebGPUBridgeConfig } from './webgpu-bridge';
import { MaterialFactory, getMaterialFactory, NEURECTOMY_PALETTE } from './materials';

// =============================================================================
// Types
// =============================================================================

export interface SceneManagerConfig {
  container: HTMLElement;
  cameraConfig?: CameraConfig;
  rendererConfig?: Partial<WebGPUBridgeConfig>;
  backgroundColor?: THREE.Color | string;
  showGrid?: boolean;
  showAxes?: boolean;
  enableShadows?: boolean;
  ambientLightIntensity?: number;
  directionalLightIntensity?: number;
}

export interface SceneLayer {
  name: string;
  visible: boolean;
  objects: Set<THREE.Object3D>;
}

export interface SelectionState {
  selected: Set<THREE.Object3D>;
  hovered: THREE.Object3D | null;
  focused: THREE.Object3D | null;
}

export interface SceneStats {
  fps: number;
  frameTime: number;
  drawCalls: number;
  triangles: number;
  objects: number;
  geometries: number;
  textures: number;
}

export type SceneEventType = 
  | 'object-added'
  | 'object-removed'
  | 'selection-changed'
  | 'hover-changed'
  | 'camera-changed'
  | 'resize';

export interface SceneEvent {
  type: SceneEventType;
  target?: THREE.Object3D | THREE.Object3D[];
  data?: unknown;
}

export type SceneEventCallback = (event: SceneEvent) => void;

// =============================================================================
// SceneManager Class
// =============================================================================

/**
 * SceneManager - Central hub for 3D scene management
 * 
 * Features:
 * - Scene graph with layers and groups
 * - Object selection and hover states
 * - Camera system integration
 * - WebGPU/WebGL rendering bridge
 * - Grid and helpers
 * - Event system
 * - Performance stats
 */
export class SceneManager {
  // Core components
  public scene: THREE.Scene;
  public cameraSystem: CameraSystem;
  public bridge: WebGPUBridge;
  public materialFactory: MaterialFactory;

  // Container and canvas
  private container: HTMLElement;
  private canvas: HTMLCanvasElement;

  // Configuration
  private config: Required<SceneManagerConfig>;

  // Scene organization
  private layers = new Map<string, SceneLayer>();
  private groups = new Map<string, THREE.Group>();

  // Selection
  private selection: SelectionState = {
    selected: new Set(),
    hovered: null,
    focused: null,
  };

  // Raycasting
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();

  // Helpers
  private gridHelper?: THREE.GridHelper;
  private axesHelper?: THREE.AxesHelper;
  private lights: THREE.Light[] = [];

  // Animation
  private clock = new THREE.Clock();
  private animationId: number | null = null;
  private isRunning = false;

  // Events
  private eventListeners = new Map<SceneEventType, Set<SceneEventCallback>>();

  // Stats
  private stats: SceneStats = {
    fps: 0,
    frameTime: 0,
    drawCalls: 0,
    triangles: 0,
    objects: 0,
    geometries: 0,
    textures: 0,
  };
  private frameTimes: number[] = [];

  constructor(config: SceneManagerConfig) {
    this.config = {
      container: config.container,
      cameraConfig: config.cameraConfig ?? {},
      rendererConfig: config.rendererConfig ?? {},
      backgroundColor: config.backgroundColor ?? NEURECTOMY_PALETTE.background,
      showGrid: config.showGrid ?? true,
      showAxes: config.showAxes ?? false,
      enableShadows: config.enableShadows ?? true,
      ambientLightIntensity: config.ambientLightIntensity ?? 0.4,
      directionalLightIntensity: config.directionalLightIntensity ?? 0.8,
    };

    this.container = config.container;
    this.materialFactory = getMaterialFactory();

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.container.appendChild(this.canvas);

    // Create scene
    this.scene = new THREE.Scene();
    const bgColor = typeof this.config.backgroundColor === 'string'
      ? new THREE.Color(this.config.backgroundColor)
      : this.config.backgroundColor;
    this.scene.background = bgColor;

    // Create camera system
    this.cameraSystem = new CameraSystem(this.config.cameraConfig);

    // Create renderer bridge
    this.bridge = new WebGPUBridge({
      canvas: this.canvas,
      ...this.config.rendererConfig,
    });

    // Setup default layers
    this.createLayer('default');
    this.createLayer('agents');
    this.createLayer('connections');
    this.createLayer('annotations');
    this.createLayer('helpers');

    // Setup scene
    this.setupLights();
    this.setupHelpers();
    this.setupEventListeners();
  }

  /**
   * Initialize the scene manager
   */
  async initialize(): Promise<boolean> {
    const success = await this.bridge.initialize();
    if (!success) {
      console.error('[SceneManager] Failed to initialize renderer');
      return false;
    }

    // Initial resize
    this.resize();

    console.log('[SceneManager] Initialized');
    return true;
  }

  /**
   * Setup lighting
   */
  private setupLights(): void {
    // Ambient light
    const ambient = new THREE.AmbientLight(0xffffff, this.config.ambientLightIntensity);
    this.scene.add(ambient);
    this.lights.push(ambient);

    // Main directional light
    const directional = new THREE.DirectionalLight(0xffffff, this.config.directionalLightIntensity);
    directional.position.set(10, 20, 10);
    directional.castShadow = this.config.enableShadows;

    if (this.config.enableShadows) {
      directional.shadow.mapSize.width = 2048;
      directional.shadow.mapSize.height = 2048;
      directional.shadow.camera.near = 0.5;
      directional.shadow.camera.far = 500;
      directional.shadow.camera.left = -50;
      directional.shadow.camera.right = 50;
      directional.shadow.camera.top = 50;
      directional.shadow.camera.bottom = -50;
    }

    this.scene.add(directional);
    this.lights.push(directional);

    // Fill light
    const fill = new THREE.DirectionalLight(0x88ccff, 0.3);
    fill.position.set(-10, 10, -10);
    this.scene.add(fill);
    this.lights.push(fill);

    // Rim light
    const rim = new THREE.DirectionalLight(0xff8866, 0.2);
    rim.position.set(0, -10, -20);
    this.scene.add(rim);
    this.lights.push(rim);
  }

  /**
   * Setup helpers (grid, axes)
   */
  private setupHelpers(): void {
    // Grid
    if (this.config.showGrid) {
      this.gridHelper = new THREE.GridHelper(
        100, 100,
        NEURECTOMY_PALETTE.gridAccent,
        NEURECTOMY_PALETTE.grid
      );
      this.gridHelper.material.opacity = 0.3;
      this.gridHelper.material.transparent = true;
      this.addToLayer(this.gridHelper, 'helpers');
    }

    // Axes
    if (this.config.showAxes) {
      this.axesHelper = new THREE.AxesHelper(10);
      this.addToLayer(this.axesHelper, 'helpers');
    }
  }

  /**
   * Setup event listeners
   */
  private setupEventListeners(): void {
    // Resize observer
    const resizeObserver = new ResizeObserver(() => this.resize());
    resizeObserver.observe(this.container);

    // Pointer events
    this.canvas.addEventListener('pointerdown', this.onPointerDown.bind(this));
    this.canvas.addEventListener('pointermove', this.onPointerMove.bind(this));
    this.canvas.addEventListener('pointerup', this.onPointerUp.bind(this));
    this.canvas.addEventListener('wheel', this.onWheel.bind(this));
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Keyboard events
    window.addEventListener('keydown', this.onKeyDown.bind(this));
  }

  /**
   * Handle pointer down
   */
  private onPointerDown(event: PointerEvent): void {
    this.updatePointer(event);
    this.cameraSystem.onPointerDown(event);

    // Left click - select
    if (event.button === 0) {
      const intersects = this.raycast();
      if (intersects.length > 0) {
        const object = this.findSelectableParent(intersects[0]!.object);
        if (object) {
          this.select(object, event.shiftKey);
        }
      } else if (!event.shiftKey) {
        this.clearSelection();
      }
    }
  }

  /**
   * Handle pointer move
   */
  private onPointerMove(event: PointerEvent): void {
    this.updatePointer(event);
    this.cameraSystem.onPointerMove(event);

    // Update hover state
    const intersects = this.raycast();
    const hovered = intersects.length > 0
      ? this.findSelectableParent(intersects[0]!.object)
      : null;

    if (hovered !== this.selection.hovered) {
      this.selection.hovered = hovered;
      this.emit({ type: 'hover-changed', target: hovered ?? undefined });
    }
  }

  /**
   * Handle pointer up
   */
  private onPointerUp(): void {
    this.cameraSystem.onPointerUp();
  }

  /**
   * Handle wheel
   */
  private onWheel(event: WheelEvent): void {
    this.cameraSystem.onWheel(event);
  }

  /**
   * Handle key down
   */
  private onKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'Escape':
        this.clearSelection();
        break;
      case 'f':
      case 'F':
        if (this.selection.selected.size > 0) {
          this.focusSelected();
        }
        break;
      case 'Delete':
      case 'Backspace':
        if (this.selection.selected.size > 0) {
          this.deleteSelected();
        }
        break;
      case 'a':
      case 'A':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.selectAll();
        }
        break;
    }
  }

  /**
   * Update pointer position
   */
  private updatePointer(event: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  /**
   * Perform raycasting
   */
  private raycast(): THREE.Intersection[] {
    this.raycaster.setFromCamera(this.pointer, this.cameraSystem.camera);
    
    // Only raycast against certain layers
    const objects: THREE.Object3D[] = [];
    for (const [name, layer] of this.layers) {
      if (name !== 'helpers' && layer.visible) {
        objects.push(...layer.objects);
      }
    }

    return this.raycaster.intersectObjects(objects, true);
  }

  /**
   * Find selectable parent of an object
   */
  private findSelectableParent(object: THREE.Object3D): THREE.Object3D | null {
    let current: THREE.Object3D | null = object;
    
    while (current) {
      if (current.userData.selectable !== false && current.parent === this.scene) {
        return current;
      }
      if (current.userData.selectable === true) {
        return current;
      }
      current = current.parent;
    }

    return null;
  }

  /**
   * Create a layer
   */
  createLayer(name: string): void {
    if (!this.layers.has(name)) {
      this.layers.set(name, {
        name,
        visible: true,
        objects: new Set(),
      });
    }
  }

  /**
   * Add object to layer
   */
  addToLayer(object: THREE.Object3D, layerName: string = 'default'): void {
    let layer = this.layers.get(layerName);
    if (!layer) {
      this.createLayer(layerName);
      layer = this.layers.get(layerName)!;
    }

    layer.objects.add(object);
    this.scene.add(object);
    this.emit({ type: 'object-added', target: object });
  }

  /**
   * Remove object from scene
   */
  remove(object: THREE.Object3D): void {
    this.scene.remove(object);
    
    // Remove from layers
    for (const layer of this.layers.values()) {
      layer.objects.delete(object);
    }

    // Remove from selection
    this.selection.selected.delete(object);
    if (this.selection.hovered === object) {
      this.selection.hovered = null;
    }
    if (this.selection.focused === object) {
      this.selection.focused = null;
    }

    this.emit({ type: 'object-removed', target: object });
  }

  /**
   * Set layer visibility
   */
  setLayerVisible(name: string, visible: boolean): void {
    const layer = this.layers.get(name);
    if (layer) {
      layer.visible = visible;
      for (const obj of layer.objects) {
        obj.visible = visible;
      }
    }
  }

  /**
   * Create a group
   */
  createGroup(name: string): THREE.Group {
    const group = new THREE.Group();
    group.name = name;
    this.groups.set(name, group);
    return group;
  }

  /**
   * Get group
   */
  getGroup(name: string): THREE.Group | undefined {
    return this.groups.get(name);
  }

  /**
   * Select an object
   */
  select(object: THREE.Object3D, additive: boolean = false): void {
    if (!additive) {
      this.clearSelection();
    }

    this.selection.selected.add(object);
    object.userData.selected = true;

    this.emit({ type: 'selection-changed', target: Array.from(this.selection.selected) });
  }

  /**
   * Deselect an object
   */
  deselect(object: THREE.Object3D): void {
    this.selection.selected.delete(object);
    object.userData.selected = false;

    this.emit({ type: 'selection-changed', target: Array.from(this.selection.selected) });
  }

  /**
   * Clear selection
   */
  clearSelection(): void {
    for (const obj of this.selection.selected) {
      obj.userData.selected = false;
    }
    this.selection.selected.clear();

    this.emit({ type: 'selection-changed', target: [] });
  }

  /**
   * Select all objects
   */
  selectAll(): void {
    for (const [name, layer] of this.layers) {
      if (name !== 'helpers' && layer.visible) {
        for (const obj of layer.objects) {
          if (obj.userData.selectable !== false) {
            this.selection.selected.add(obj);
            obj.userData.selected = true;
          }
        }
      }
    }

    this.emit({ type: 'selection-changed', target: Array.from(this.selection.selected) });
  }

  /**
   * Focus on selected objects
   */
  focusSelected(): void {
    if (this.selection.selected.size === 0) return;

    if (this.selection.selected.size === 1) {
      const obj = Array.from(this.selection.selected)[0]!;
      this.cameraSystem.focusOn(obj);
      this.selection.focused = obj;
    } else {
      // Focus on bounding box of all selected
      const box = new THREE.Box3();
      for (const obj of this.selection.selected) {
        box.expandByObject(obj);
      }
      
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = (this.cameraSystem.camera as THREE.PerspectiveCamera).fov * (Math.PI / 180);
      const distance = (maxDim * 1.5) / (2 * Math.tan(fov / 2));

      this.cameraSystem.transitionTo({
        position: center.clone().add(new THREE.Vector3(0, distance * 0.5, distance)),
        target: center,
      });
    }
  }

  /**
   * Delete selected objects
   */
  deleteSelected(): void {
    const toDelete = Array.from(this.selection.selected);
    this.clearSelection();

    for (const obj of toDelete) {
      this.remove(obj);
      
      // Dispose geometry and materials
      obj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    }
  }

  /**
   * Get selected objects
   */
  getSelected(): THREE.Object3D[] {
    return Array.from(this.selection.selected);
  }

  /**
   * Get hovered object
   */
  getHovered(): THREE.Object3D | null {
    return this.selection.hovered;
  }

  /**
   * Resize handler
   */
  resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.canvas.width = width * window.devicePixelRatio;
    this.canvas.height = height * window.devicePixelRatio;

    this.cameraSystem.resize(width, height);
    this.bridge.resize(width, height);

    this.emit({ type: 'resize', data: { width, height } });
  }

  /**
   * Start rendering loop
   */
  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.clock.start();
    this.animate();
    console.log('[SceneManager] Started');
  }

  /**
   * Stop rendering loop
   */
  stop(): void {
    if (!this.isRunning) return;
    this.isRunning = false;
    
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    console.log('[SceneManager] Stopped');
  }

  /**
   * Animation loop
   */
  private animate(): void {
    if (!this.isRunning) return;

    this.animationId = requestAnimationFrame(() => this.animate());

    const deltaTime = this.clock.getDelta();
    const frameStart = performance.now();

    // Update systems
    this.cameraSystem.update(deltaTime);
    this.materialFactory.updateTime(deltaTime);

    // Render
    this.bridge.render(this.scene, this.cameraSystem.camera);

    // Update stats
    this.updateStats(performance.now() - frameStart);
  }

  /**
   * Update performance stats
   */
  private updateStats(frameTime: number): void {
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > 60) {
      this.frameTimes.shift();
    }

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    
    const info = this.bridge.getInfo();
    
    this.stats = {
      fps: info.fps,
      frameTime: avgFrameTime,
      drawCalls: info.render?.calls ?? 0,
      triangles: info.render?.triangles ?? 0,
      objects: this.scene.children.length,
      geometries: info.memory?.geometries ?? 0,
      textures: info.memory?.textures ?? 0,
    };
  }

  /**
   * Get current stats
   */
  getStats(): SceneStats {
    return { ...this.stats };
  }

  /**
   * Add event listener
   */
  on(type: SceneEventType, callback: SceneEventCallback): void {
    if (!this.eventListeners.has(type)) {
      this.eventListeners.set(type, new Set());
    }
    this.eventListeners.get(type)!.add(callback);
  }

  /**
   * Remove event listener
   */
  off(type: SceneEventType, callback: SceneEventCallback): void {
    this.eventListeners.get(type)?.delete(callback);
  }

  /**
   * Emit event
   */
  private emit(event: SceneEvent): void {
    const listeners = this.eventListeners.get(event.type);
    if (listeners) {
      for (const callback of listeners) {
        callback(event);
      }
    }
  }

  /**
   * Take a screenshot
   */
  screenshot(width?: number, height?: number): string {
    return this.bridge.screenshot(this.scene, this.cameraSystem.camera, width, height);
  }

  /**
   * Dispose all resources
   */
  dispose(): void {
    this.stop();

    // Remove event listeners
    this.eventListeners.clear();

    // Dispose scene contents
    while (this.scene.children.length > 0) {
      const child = this.scene.children[0]!;
      this.scene.remove(child);
      
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach(m => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    }

    // Clear collections
    this.layers.clear();
    this.groups.clear();
    this.selection.selected.clear();

    // Dispose subsystems
    this.cameraSystem.dispose();
    this.bridge.dispose();
    this.materialFactory.dispose();

    // Remove canvas
    this.container.removeChild(this.canvas);

    console.log('[SceneManager] Disposed');
  }
}
