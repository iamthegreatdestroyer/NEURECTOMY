/**
 * Interaction System - Advanced 3D Scene Interaction
 * 
 * Comprehensive interaction system for CAD-quality manipulation of 3D scenes.
 * Supports selection, multi-select, drag, transform, and gesture-based interactions.
 * 
 * @module @neurectomy/3d-engine/cad/interaction-system
 * @agents @CANVAS @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 * @step Step 3 - CAD Visualization System
 */

import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls';

// =============================================================================
// TYPES & INTERFACES
// =============================================================================

/**
 * Interaction modes
 */
export enum InteractionMode {
  /** Default view/navigation mode */
  VIEW = 'view',
  /** Selection mode */
  SELECT = 'select',
  /** Multi-select with lasso/box */
  MULTI_SELECT = 'multi_select',
  /** Transform objects */
  TRANSFORM = 'transform',
  /** Measure distances/angles */
  MEASURE = 'measure',
  /** Annotate scene */
  ANNOTATE = 'annotate',
  /** Create connections */
  CONNECT = 'connect',
  /** Pan mode */
  PAN = 'pan',
  /** Zoom mode */
  ZOOM = 'zoom'
}

/**
 * Transform modes for manipulating objects
 */
export enum TransformMode {
  TRANSLATE = 'translate',
  ROTATE = 'rotate',
  SCALE = 'scale'
}

/**
 * Transform space
 */
export enum TransformSpace {
  WORLD = 'world',
  LOCAL = 'local'
}

/**
 * Selection types
 */
export enum SelectionType {
  SINGLE = 'single',
  ADDITIVE = 'additive',
  SUBTRACTIVE = 'subtractive',
  BOX = 'box',
  LASSO = 'lasso'
}

/**
 * Interaction event data
 */
export interface InteractionEvent {
  type: string;
  target: THREE.Object3D | null;
  targets: THREE.Object3D[];
  point: THREE.Vector3 | null;
  screenPosition: THREE.Vector2;
  button: number;
  modifiers: {
    shift: boolean;
    ctrl: boolean;
    alt: boolean;
    meta: boolean;
  };
  originalEvent: MouseEvent | TouchEvent | WheelEvent | KeyboardEvent;
}

/**
 * Drag event data
 */
export interface DragEvent extends InteractionEvent {
  startPosition: THREE.Vector3;
  currentPosition: THREE.Vector3;
  delta: THREE.Vector3;
  screenDelta: THREE.Vector2;
}

/**
 * Transform event data
 */
export interface TransformEvent {
  type: 'start' | 'change' | 'end';
  mode: TransformMode;
  target: THREE.Object3D;
  previousTransform: {
    position: THREE.Vector3;
    rotation: THREE.Euler;
    scale: THREE.Vector3;
  };
  currentTransform: {
    position: THREE.Vector3;
    rotation: THREE.Euler;
    scale: THREE.Vector3;
  };
}

/**
 * Selection box data
 */
export interface SelectionBox {
  startScreen: THREE.Vector2;
  endScreen: THREE.Vector2;
  active: boolean;
}

/**
 * Lasso selection data
 */
export interface LassoSelection {
  points: THREE.Vector2[];
  active: boolean;
}

/**
 * Interaction system configuration
 */
export interface InteractionConfig {
  /** Enable/disable interactions */
  enabled: boolean;
  
  /** Allow multi-select */
  multiSelect: boolean;
  
  /** Double-click behavior */
  doubleClickAction: 'focus' | 'edit' | 'none';
  
  /** Hover highlight delay (ms) */
  hoverDelay: number;
  
  /** Drag threshold (pixels) */
  dragThreshold: number;
  
  /** Double-click time (ms) */
  doubleClickTime: number;
  
  /** Snapping */
  snapping: {
    enabled: boolean;
    gridSize: number;
    angleSnap: number;
    snapToObjects: boolean;
  };
  
  /** Transform controls */
  transformControls: {
    size: number;
    showX: boolean;
    showY: boolean;
    showZ: boolean;
  };
  
  /** Selection box color */
  selectionBoxColor: THREE.Color;
  
  /** Layers to interact with */
  interactableLayers: number[];
}

/**
 * Snapping info
 */
export interface SnapInfo {
  snapped: boolean;
  point: THREE.Vector3;
  normal?: THREE.Vector3;
  snapType: 'grid' | 'object' | 'vertex' | 'edge' | 'face' | 'none';
  snapObject?: THREE.Object3D;
}

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

const DEFAULT_CONFIG: InteractionConfig = {
  enabled: true,
  multiSelect: true,
  doubleClickAction: 'focus',
  hoverDelay: 100,
  dragThreshold: 5,
  doubleClickTime: 300,
  
  snapping: {
    enabled: true,
    gridSize: 1.0,
    angleSnap: 15,
    snapToObjects: true
  },
  
  transformControls: {
    size: 1,
    showX: true,
    showY: true,
    showZ: true
  },
  
  selectionBoxColor: new THREE.Color(0x00ffff),
  interactableLayers: [0]
};

// =============================================================================
// HELPER CLASSES
// =============================================================================

/**
 * Selection manager for tracking selected objects
 */
class SelectionManager {
  private selected: Set<THREE.Object3D> = new Set();
  private primary: THREE.Object3D | null = null;
  private eventTarget: EventTarget = new EventTarget();
  
  /**
   * Select an object
   */
  select(object: THREE.Object3D, type: SelectionType = SelectionType.SINGLE): void {
    switch (type) {
      case SelectionType.SINGLE:
        this.clearSelection();
        this.addToSelection(object);
        this.primary = object;
        break;
      
      case SelectionType.ADDITIVE:
        this.addToSelection(object);
        if (!this.primary) {
          this.primary = object;
        }
        break;
      
      case SelectionType.SUBTRACTIVE:
        this.removeFromSelection(object);
        if (this.primary === object) {
          this.primary = this.selected.size > 0 
            ? this.selected.values().next().value 
            : null;
        }
        break;
    }
    
    this.emitChange();
  }
  
  /**
   * Select multiple objects
   */
  selectMultiple(objects: THREE.Object3D[], type: SelectionType = SelectionType.SINGLE): void {
    if (type === SelectionType.SINGLE) {
      this.clearSelection();
    }
    
    for (const object of objects) {
      if (type === SelectionType.SUBTRACTIVE) {
        this.removeFromSelection(object);
      } else {
        this.addToSelection(object);
      }
    }
    
    if (!this.primary && objects.length > 0) {
      this.primary = objects[0];
    }
    
    this.emitChange();
  }
  
  /**
   * Toggle selection of an object
   */
  toggleSelection(object: THREE.Object3D): void {
    if (this.selected.has(object)) {
      this.removeFromSelection(object);
      if (this.primary === object) {
        this.primary = this.selected.size > 0 
          ? this.selected.values().next().value 
          : null;
      }
    } else {
      this.addToSelection(object);
      if (!this.primary) {
        this.primary = object;
      }
    }
    
    this.emitChange();
  }
  
  /**
   * Clear all selections
   */
  clearSelection(): void {
    for (const object of this.selected) {
      object.userData.selected = false;
    }
    this.selected.clear();
    this.primary = null;
    this.emitChange();
  }
  
  /**
   * Get selected objects
   */
  getSelection(): THREE.Object3D[] {
    return Array.from(this.selected);
  }
  
  /**
   * Get primary selection
   */
  getPrimary(): THREE.Object3D | null {
    return this.primary;
  }
  
  /**
   * Check if object is selected
   */
  isSelected(object: THREE.Object3D): boolean {
    return this.selected.has(object);
  }
  
  /**
   * Get selection count
   */
  getCount(): number {
    return this.selected.size;
  }
  
  /**
   * Get bounding box of selection
   */
  getBoundingBox(): THREE.Box3 {
    const box = new THREE.Box3();
    
    for (const object of this.selected) {
      box.expandByObject(object);
    }
    
    return box;
  }
  
  /**
   * Get center of selection
   */
  getCenter(): THREE.Vector3 {
    const center = new THREE.Vector3();
    
    if (this.selected.size === 0) return center;
    
    for (const object of this.selected) {
      center.add(object.position);
    }
    
    center.divideScalar(this.selected.size);
    return center;
  }
  
  private addToSelection(object: THREE.Object3D): void {
    if (!this.selected.has(object)) {
      this.selected.add(object);
      object.userData.selected = true;
    }
  }
  
  private removeFromSelection(object: THREE.Object3D): void {
    if (this.selected.has(object)) {
      this.selected.delete(object);
      object.userData.selected = false;
    }
  }
  
  private emitChange(): void {
    this.eventTarget.dispatchEvent(new CustomEvent('selectionChanged', {
      detail: {
        selection: this.getSelection(),
        primary: this.primary,
        count: this.selected.size
      }
    }));
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
}

/**
 * Hover manager for tracking hovered objects
 */
class HoverManager {
  private hoveredObject: THREE.Object3D | null = null;
  private hoverTimeout: number | null = null;
  private hoverDelay: number;
  private eventTarget: EventTarget = new EventTarget();
  
  constructor(hoverDelay: number = 100) {
    this.hoverDelay = hoverDelay;
  }
  
  /**
   * Set hover delay
   */
  setDelay(delay: number): void {
    this.hoverDelay = delay;
  }
  
  /**
   * Update hover state
   */
  update(object: THREE.Object3D | null): void {
    if (object === this.hoveredObject) return;
    
    // Clear pending hover
    if (this.hoverTimeout !== null) {
      clearTimeout(this.hoverTimeout);
      this.hoverTimeout = null;
    }
    
    // Unhover previous
    if (this.hoveredObject) {
      const prev = this.hoveredObject;
      this.hoveredObject = null;
      prev.userData.hovered = false;
      
      this.eventTarget.dispatchEvent(new CustomEvent('hoverEnd', {
        detail: { object: prev }
      }));
    }
    
    // Hover new with delay
    if (object) {
      this.hoverTimeout = window.setTimeout(() => {
        this.hoveredObject = object;
        object.userData.hovered = true;
        
        this.eventTarget.dispatchEvent(new CustomEvent('hoverStart', {
          detail: { object }
        }));
      }, this.hoverDelay);
    }
  }
  
  /**
   * Force clear hover
   */
  clear(): void {
    if (this.hoverTimeout !== null) {
      clearTimeout(this.hoverTimeout);
      this.hoverTimeout = null;
    }
    
    if (this.hoveredObject) {
      this.hoveredObject.userData.hovered = false;
      this.hoveredObject = null;
    }
  }
  
  /**
   * Get hovered object
   */
  getHovered(): THREE.Object3D | null {
    return this.hoveredObject;
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
}

// =============================================================================
// MAIN INTERACTION SYSTEM
// =============================================================================

/**
 * Comprehensive 3D interaction system
 */
export class InteractionSystem {
  private config: InteractionConfig;
  private scene: THREE.Scene;
  private camera: THREE.Camera;
  private renderer: THREE.WebGLRenderer;
  private domElement: HTMLElement;
  
  // Managers
  private selectionManager: SelectionManager;
  private hoverManager: HoverManager;
  
  // State
  private mode: InteractionMode = InteractionMode.SELECT;
  private transformMode: TransformMode = TransformMode.TRANSLATE;
  private transformSpace: TransformSpace = TransformSpace.WORLD;
  private enabled: boolean = true;
  
  // Transform controls
  private transformControls: TransformControls | null = null;
  
  // Raycasting
  private raycaster: THREE.Raycaster = new THREE.Raycaster();
  private mouse: THREE.Vector2 = new THREE.Vector2();
  private interactableObjects: THREE.Object3D[] = [];
  
  // Drag state
  private isDragging: boolean = false;
  private dragStart: THREE.Vector2 = new THREE.Vector2();
  private dragStartWorld: THREE.Vector3 = new THREE.Vector3();
  private dragPlane: THREE.Plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  
  // Selection box
  private selectionBox: SelectionBox = {
    startScreen: new THREE.Vector2(),
    endScreen: new THREE.Vector2(),
    active: false
  };
  private selectionBoxHelper: THREE.LineSegments | null = null;
  
  // Lasso selection
  private lassoSelection: LassoSelection = {
    points: [],
    active: false
  };
  
  // Click tracking
  private lastClickTime: number = 0;
  private lastClickTarget: THREE.Object3D | null = null;
  private mouseDownPosition: THREE.Vector2 = new THREE.Vector2();
  private mouseDownTime: number = 0;
  
  // Event emitter
  private eventTarget: EventTarget = new EventTarget();
  
  // Bound event handlers
  private boundHandlers: {
    mouseDown: (e: MouseEvent) => void;
    mouseMove: (e: MouseEvent) => void;
    mouseUp: (e: MouseEvent) => void;
    wheel: (e: WheelEvent) => void;
    keyDown: (e: KeyboardEvent) => void;
    keyUp: (e: KeyboardEvent) => void;
    contextMenu: (e: MouseEvent) => void;
  };
  
  constructor(
    scene: THREE.Scene,
    camera: THREE.Camera,
    renderer: THREE.WebGLRenderer,
    config: Partial<InteractionConfig> = {}
  ) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.domElement = renderer.domElement;
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Initialize managers
    this.selectionManager = new SelectionManager();
    this.hoverManager = new HoverManager(this.config.hoverDelay);
    
    // Setup raycaster layers
    for (const layer of this.config.interactableLayers) {
      this.raycaster.layers.enable(layer);
    }
    
    // Bind event handlers
    this.boundHandlers = {
      mouseDown: this.handleMouseDown.bind(this),
      mouseMove: this.handleMouseMove.bind(this),
      mouseUp: this.handleMouseUp.bind(this),
      wheel: this.handleWheel.bind(this),
      keyDown: this.handleKeyDown.bind(this),
      keyUp: this.handleKeyUp.bind(this),
      contextMenu: this.handleContextMenu.bind(this)
    };
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Create selection box helper
    this.createSelectionBoxHelper();
  }
  
  // ============================================
  // SETUP
  // ============================================
  
  private setupEventListeners(): void {
    this.domElement.addEventListener('mousedown', this.boundHandlers.mouseDown);
    this.domElement.addEventListener('mousemove', this.boundHandlers.mouseMove);
    this.domElement.addEventListener('mouseup', this.boundHandlers.mouseUp);
    this.domElement.addEventListener('wheel', this.boundHandlers.wheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this.boundHandlers.contextMenu);
    window.addEventListener('keydown', this.boundHandlers.keyDown);
    window.addEventListener('keyup', this.boundHandlers.keyUp);
  }
  
  private removeEventListeners(): void {
    this.domElement.removeEventListener('mousedown', this.boundHandlers.mouseDown);
    this.domElement.removeEventListener('mousemove', this.boundHandlers.mouseMove);
    this.domElement.removeEventListener('mouseup', this.boundHandlers.mouseUp);
    this.domElement.removeEventListener('wheel', this.boundHandlers.wheel);
    this.domElement.removeEventListener('contextmenu', this.boundHandlers.contextMenu);
    window.removeEventListener('keydown', this.boundHandlers.keyDown);
    window.removeEventListener('keyup', this.boundHandlers.keyUp);
  }
  
  private createSelectionBoxHelper(): void {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(24); // 8 corners * 3 components
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const indices = [
      0, 1, 1, 2, 2, 3, 3, 0, // Front face
      4, 5, 5, 6, 6, 7, 7, 4, // Back face
      0, 4, 1, 5, 2, 6, 3, 7  // Connections
    ];
    geometry.setIndex(indices);
    
    const material = new THREE.LineBasicMaterial({
      color: this.config.selectionBoxColor,
      depthTest: false,
      transparent: true,
      opacity: 0.8
    });
    
    this.selectionBoxHelper = new THREE.LineSegments(geometry, material);
    this.selectionBoxHelper.visible = false;
    this.selectionBoxHelper.renderOrder = 999;
    this.scene.add(this.selectionBoxHelper);
  }
  
  /**
   * Setup transform controls
   */
  setupTransformControls(): TransformControls {
    if (this.transformControls) {
      return this.transformControls;
    }
    
    this.transformControls = new TransformControls(
      this.camera,
      this.domElement
    );
    
    this.transformControls.size = this.config.transformControls.size;
    this.transformControls.showX = this.config.transformControls.showX;
    this.transformControls.showY = this.config.transformControls.showY;
    this.transformControls.showZ = this.config.transformControls.showZ;
    
    // Handle transform events
    this.transformControls.addEventListener('dragging-changed', (event) => {
      // Disable camera controls while transforming
      this.emitEvent('transformDragging', { dragging: event.value });
    });
    
    this.transformControls.addEventListener('objectChange', () => {
      const object = this.transformControls!.object;
      if (object) {
        this.emitEvent('transformChange', {
          object,
          mode: this.transformMode
        });
      }
    });
    
    this.scene.add(this.transformControls);
    
    return this.transformControls;
  }
  
  // ============================================
  // MODE MANAGEMENT
  // ============================================
  
  /**
   * Set interaction mode
   */
  setMode(mode: InteractionMode): void {
    if (this.mode === mode) return;
    
    const prevMode = this.mode;
    this.mode = mode;
    
    // Handle mode-specific setup
    switch (mode) {
      case InteractionMode.TRANSFORM:
        this.setupTransformControls();
        this.attachTransformToSelection();
        break;
      
      default:
        this.detachTransformControls();
        break;
    }
    
    this.emitEvent('modeChanged', { mode, previousMode: prevMode });
  }
  
  /**
   * Get current mode
   */
  getMode(): InteractionMode {
    return this.mode;
  }
  
  /**
   * Set transform mode
   */
  setTransformMode(mode: TransformMode): void {
    this.transformMode = mode;
    
    if (this.transformControls) {
      this.transformControls.mode = mode;
    }
    
    this.emitEvent('transformModeChanged', { mode });
  }
  
  /**
   * Set transform space
   */
  setTransformSpace(space: TransformSpace): void {
    this.transformSpace = space;
    
    if (this.transformControls) {
      this.transformControls.space = space;
    }
    
    this.emitEvent('transformSpaceChanged', { space });
  }
  
  /**
   * Enable/disable interactions
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.config.enabled = enabled;
    
    if (!enabled) {
      this.hoverManager.clear();
      if (this.transformControls) {
        this.transformControls.enabled = false;
      }
    } else if (this.transformControls) {
      this.transformControls.enabled = true;
    }
  }
  
  // ============================================
  // INTERACTABLE OBJECTS
  // ============================================
  
  /**
   * Register objects as interactable
   */
  setInteractableObjects(objects: THREE.Object3D[]): void {
    this.interactableObjects = objects;
  }
  
  /**
   * Add interactable object
   */
  addInteractableObject(object: THREE.Object3D): void {
    if (!this.interactableObjects.includes(object)) {
      this.interactableObjects.push(object);
    }
  }
  
  /**
   * Remove interactable object
   */
  removeInteractableObject(object: THREE.Object3D): void {
    const index = this.interactableObjects.indexOf(object);
    if (index !== -1) {
      this.interactableObjects.splice(index, 1);
    }
  }
  
  // ============================================
  // RAYCASTING
  // ============================================
  
  /**
   * Update mouse position from event
   */
  private updateMouse(event: MouseEvent): void {
    const rect = this.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }
  
  /**
   * Raycast to find objects under cursor
   */
  raycast(recursive: boolean = true): THREE.Intersection[] {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    return this.raycaster.intersectObjects(this.interactableObjects, recursive);
  }
  
  /**
   * Get first intersected object
   */
  getObjectUnderCursor(): THREE.Object3D | null {
    const intersects = this.raycast();
    
    if (intersects.length > 0) {
      // Find the root interactable object
      let target = intersects[0].object;
      
      while (target.parent && !this.interactableObjects.includes(target)) {
        target = target.parent;
      }
      
      if (this.interactableObjects.includes(target)) {
        return target;
      }
    }
    
    return null;
  }
  
  /**
   * Get world position under cursor on plane
   */
  getWorldPositionOnPlane(plane: THREE.Plane = this.dragPlane): THREE.Vector3 | null {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    
    const target = new THREE.Vector3();
    const result = this.raycaster.ray.intersectPlane(plane, target);
    
    return result;
  }
  
  // ============================================
  // EVENT HANDLERS
  // ============================================
  
  private handleMouseDown(event: MouseEvent): void {
    if (!this.enabled) return;
    
    this.updateMouse(event);
    this.mouseDownPosition.set(event.clientX, event.clientY);
    this.mouseDownTime = performance.now();
    
    const intersects = this.raycast();
    const target = this.getObjectUnderCursor();
    
    // Create interaction event
    const interactionEvent = this.createInteractionEvent('mouseDown', event, target, intersects);
    
    // Handle based on mode
    switch (this.mode) {
      case InteractionMode.SELECT:
      case InteractionMode.VIEW:
        if (target) {
          this.handleSelectionClick(target, event);
        }
        break;
      
      case InteractionMode.MULTI_SELECT:
        if (event.button === 0) {
          this.startSelectionBox(event);
        }
        break;
      
      case InteractionMode.TRANSFORM:
        // Transform controls handle their own events
        break;
      
      case InteractionMode.CONNECT:
        if (target) {
          this.startConnection(target, interactionEvent);
        }
        break;
    }
    
    // Prepare for potential drag
    if (target && intersects.length > 0) {
      this.dragStartWorld.copy(intersects[0].point);
      
      // Set drag plane at the intersection point
      const normal = this.camera.getWorldDirection(new THREE.Vector3()).negate();
      this.dragPlane.setFromNormalAndCoplanarPoint(normal, this.dragStartWorld);
    }
    
    this.dragStart.set(event.clientX, event.clientY);
    
    this.emitEvent('mouseDown', interactionEvent);
  }
  
  private handleMouseMove(event: MouseEvent): void {
    if (!this.enabled) return;
    
    this.updateMouse(event);
    
    const target = this.getObjectUnderCursor();
    
    // Update hover
    this.hoverManager.update(target);
    
    // Check for drag
    const dx = event.clientX - this.mouseDownPosition.x;
    const dy = event.clientY - this.mouseDownPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (event.buttons & 1) { // Left mouse button held
      if (distance > this.config.dragThreshold && !this.isDragging) {
        this.isDragging = true;
        this.emitEvent('dragStart', this.createDragEvent(event));
      }
      
      if (this.isDragging) {
        // Update selection box
        if (this.selectionBox.active) {
          this.updateSelectionBox(event);
        }
        
        this.emitEvent('drag', this.createDragEvent(event));
      }
    }
    
    // Create and emit move event
    const interactionEvent = this.createInteractionEvent('mouseMove', event, target, []);
    this.emitEvent('mouseMove', interactionEvent);
  }
  
  private handleMouseUp(event: MouseEvent): void {
    if (!this.enabled) return;
    
    this.updateMouse(event);
    
    const target = this.getObjectUnderCursor();
    const clickDuration = performance.now() - this.mouseDownTime;
    const wasDragging = this.isDragging;
    
    // End drag
    if (this.isDragging) {
      this.isDragging = false;
      this.emitEvent('dragEnd', this.createDragEvent(event));
    }
    
    // End selection box
    if (this.selectionBox.active) {
      this.endSelectionBox(event);
    }
    
    // Handle click (if not dragging)
    if (!wasDragging) {
      const interactionEvent = this.createInteractionEvent('click', event, target, this.raycast());
      this.emitEvent('click', interactionEvent);
      
      // Check for double-click
      const now = performance.now();
      if (
        target === this.lastClickTarget &&
        now - this.lastClickTime < this.config.doubleClickTime
      ) {
        this.handleDoubleClick(target, interactionEvent);
        this.lastClickTime = 0;
      } else {
        this.lastClickTime = now;
        this.lastClickTarget = target;
      }
    }
    
    const interactionEvent = this.createInteractionEvent('mouseUp', event, target, []);
    this.emitEvent('mouseUp', interactionEvent);
  }
  
  private handleWheel(event: WheelEvent): void {
    if (!this.enabled) return;
    
    const interactionEvent = this.createInteractionEvent(
      'wheel',
      event,
      this.getObjectUnderCursor(),
      []
    );
    
    this.emitEvent('wheel', { ...interactionEvent, deltaY: event.deltaY });
  }
  
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.enabled) return;
    
    // Handle keyboard shortcuts
    switch (event.key) {
      case 'Escape':
        this.selectionManager.clearSelection();
        this.cancelCurrentAction();
        break;
      
      case 'Delete':
      case 'Backspace':
        this.deleteSelection();
        break;
      
      case 'a':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.selectAll();
        }
        break;
      
      case 'g':
        if (this.mode === InteractionMode.TRANSFORM) {
          this.setTransformMode(TransformMode.TRANSLATE);
        }
        break;
      
      case 'r':
        if (this.mode === InteractionMode.TRANSFORM) {
          this.setTransformMode(TransformMode.ROTATE);
        }
        break;
      
      case 's':
        if (this.mode === InteractionMode.TRANSFORM && !event.ctrlKey) {
          this.setTransformMode(TransformMode.SCALE);
        }
        break;
      
      case 'x':
        if (this.transformControls) {
          this.transformControls.showX = !this.transformControls.showX;
        }
        break;
      
      case 'y':
        if (this.transformControls) {
          this.transformControls.showY = !this.transformControls.showY;
        }
        break;
      
      case 'z':
        if (this.transformControls) {
          this.transformControls.showZ = !this.transformControls.showZ;
        }
        break;
    }
    
    this.emitEvent('keyDown', { key: event.key, event });
  }
  
  private handleKeyUp(event: KeyboardEvent): void {
    if (!this.enabled) return;
    
    this.emitEvent('keyUp', { key: event.key, event });
  }
  
  private handleContextMenu(event: MouseEvent): void {
    event.preventDefault();
    
    if (!this.enabled) return;
    
    this.updateMouse(event);
    const target = this.getObjectUnderCursor();
    
    const interactionEvent = this.createInteractionEvent('contextMenu', event, target, this.raycast());
    this.emitEvent('contextMenu', interactionEvent);
  }
  
  // ============================================
  // SELECTION HANDLING
  // ============================================
  
  private handleSelectionClick(target: THREE.Object3D, event: MouseEvent): void {
    const selectionType = this.getSelectionType(event);
    
    if (selectionType === SelectionType.ADDITIVE || selectionType === SelectionType.SUBTRACTIVE) {
      this.selectionManager.toggleSelection(target);
    } else {
      this.selectionManager.select(target, selectionType);
    }
    
    // Update transform controls
    if (this.mode === InteractionMode.TRANSFORM) {
      this.attachTransformToSelection();
    }
  }
  
  private handleDoubleClick(target: THREE.Object3D | null, event: InteractionEvent): void {
    this.emitEvent('doubleClick', event);
    
    switch (this.config.doubleClickAction) {
      case 'focus':
        if (target) {
          this.emitEvent('focusRequest', { target });
        }
        break;
      
      case 'edit':
        if (target) {
          this.emitEvent('editRequest', { target });
        }
        break;
    }
  }
  
  private getSelectionType(event: MouseEvent): SelectionType {
    if (event.shiftKey && this.config.multiSelect) {
      return SelectionType.ADDITIVE;
    }
    if (event.ctrlKey && this.config.multiSelect) {
      return SelectionType.SUBTRACTIVE;
    }
    return SelectionType.SINGLE;
  }
  
  // ============================================
  // SELECTION BOX
  // ============================================
  
  private startSelectionBox(event: MouseEvent): void {
    this.selectionBox.active = true;
    this.selectionBox.startScreen.set(event.clientX, event.clientY);
    this.selectionBox.endScreen.set(event.clientX, event.clientY);
    
    if (this.selectionBoxHelper) {
      this.selectionBoxHelper.visible = true;
    }
  }
  
  private updateSelectionBox(event: MouseEvent): void {
    if (!this.selectionBox.active) return;
    
    this.selectionBox.endScreen.set(event.clientX, event.clientY);
    
    // Update helper visualization (2D overlay would be better here)
    // For now, just emit update
    this.emitEvent('selectionBoxUpdate', {
      start: this.selectionBox.startScreen,
      end: this.selectionBox.endScreen
    });
  }
  
  private endSelectionBox(event: MouseEvent): void {
    if (!this.selectionBox.active) return;
    
    this.selectionBox.endScreen.set(event.clientX, event.clientY);
    this.selectionBox.active = false;
    
    if (this.selectionBoxHelper) {
      this.selectionBoxHelper.visible = false;
    }
    
    // Find objects in selection box
    const objectsInBox = this.getObjectsInSelectionBox();
    
    const selectionType = event.shiftKey 
      ? SelectionType.ADDITIVE 
      : event.ctrlKey 
        ? SelectionType.SUBTRACTIVE 
        : SelectionType.SINGLE;
    
    this.selectionManager.selectMultiple(objectsInBox, selectionType);
    
    this.emitEvent('selectionBoxEnd', {
      start: this.selectionBox.startScreen,
      end: this.selectionBox.endScreen,
      objects: objectsInBox
    });
  }
  
  private getObjectsInSelectionBox(): THREE.Object3D[] {
    const result: THREE.Object3D[] = [];
    
    const rect = this.domElement.getBoundingClientRect();
    
    // Normalize selection box coordinates
    const minX = Math.min(this.selectionBox.startScreen.x, this.selectionBox.endScreen.x);
    const maxX = Math.max(this.selectionBox.startScreen.x, this.selectionBox.endScreen.x);
    const minY = Math.min(this.selectionBox.startScreen.y, this.selectionBox.endScreen.y);
    const maxY = Math.max(this.selectionBox.startScreen.y, this.selectionBox.endScreen.y);
    
    for (const object of this.interactableObjects) {
      // Project object center to screen
      const screenPos = new THREE.Vector3();
      object.getWorldPosition(screenPos);
      screenPos.project(this.camera);
      
      const x = (screenPos.x + 1) / 2 * rect.width + rect.left;
      const y = (-screenPos.y + 1) / 2 * rect.height + rect.top;
      
      // Check if inside selection box
      if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
        result.push(object);
      }
    }
    
    return result;
  }
  
  // ============================================
  // TRANSFORM CONTROLS
  // ============================================
  
  private attachTransformToSelection(): void {
    if (!this.transformControls) return;
    
    const primary = this.selectionManager.getPrimary();
    
    if (primary) {
      this.transformControls.attach(primary);
    } else {
      this.transformControls.detach();
    }
  }
  
  private detachTransformControls(): void {
    if (this.transformControls) {
      this.transformControls.detach();
    }
  }
  
  // ============================================
  // CONNECTION HANDLING
  // ============================================
  
  private connectionStart: THREE.Object3D | null = null;
  
  private startConnection(target: THREE.Object3D, event: InteractionEvent): void {
    this.connectionStart = target;
    this.emitEvent('connectionStart', { source: target, event });
  }
  
  /**
   * Complete a connection
   */
  completeConnection(target: THREE.Object3D): void {
    if (this.connectionStart && target !== this.connectionStart) {
      this.emitEvent('connectionComplete', {
        source: this.connectionStart,
        target
      });
    }
    this.connectionStart = null;
  }
  
  /**
   * Cancel current connection
   */
  cancelConnection(): void {
    this.connectionStart = null;
    this.emitEvent('connectionCancelled', {});
  }
  
  // ============================================
  // UTILITY METHODS
  // ============================================
  
  private createInteractionEvent(
    type: string,
    event: MouseEvent | TouchEvent | WheelEvent | KeyboardEvent,
    target: THREE.Object3D | null,
    intersects: THREE.Intersection[]
  ): InteractionEvent {
    const screenPos = event instanceof MouseEvent
      ? new THREE.Vector2(event.clientX, event.clientY)
      : new THREE.Vector2();
    
    let modifiers = { shift: false, ctrl: false, alt: false, meta: false };
    if (event instanceof MouseEvent || event instanceof KeyboardEvent) {
      modifiers = {
        shift: event.shiftKey,
        ctrl: event.ctrlKey,
        alt: event.altKey,
        meta: event.metaKey
      };
    }
    
    return {
      type,
      target,
      targets: this.selectionManager.getSelection(),
      point: intersects.length > 0 ? intersects[0].point : null,
      screenPosition: screenPos,
      button: event instanceof MouseEvent ? event.button : 0,
      modifiers,
      originalEvent: event
    };
  }
  
  private createDragEvent(event: MouseEvent): DragEvent {
    const currentWorld = this.getWorldPositionOnPlane() ?? new THREE.Vector3();
    
    return {
      ...this.createInteractionEvent('drag', event, this.selectionManager.getPrimary(), []),
      startPosition: this.dragStartWorld.clone(),
      currentPosition: currentWorld,
      delta: currentWorld.clone().sub(this.dragStartWorld),
      screenDelta: new THREE.Vector2(
        event.clientX - this.dragStart.x,
        event.clientY - this.dragStart.y
      )
    };
  }
  
  private cancelCurrentAction(): void {
    if (this.selectionBox.active) {
      this.selectionBox.active = false;
      if (this.selectionBoxHelper) {
        this.selectionBoxHelper.visible = false;
      }
    }
    
    if (this.lassoSelection.active) {
      this.lassoSelection.active = false;
      this.lassoSelection.points = [];
    }
    
    if (this.connectionStart) {
      this.cancelConnection();
    }
    
    this.isDragging = false;
  }
  
  private deleteSelection(): void {
    const selection = this.selectionManager.getSelection();
    
    if (selection.length > 0) {
      this.emitEvent('deleteRequest', { objects: selection });
    }
  }
  
  private selectAll(): void {
    this.selectionManager.selectMultiple(this.interactableObjects, SelectionType.SINGLE);
  }
  
  // ============================================
  // PUBLIC API
  // ============================================
  
  /**
   * Get selection manager
   */
  getSelectionManager(): SelectionManager {
    return this.selectionManager;
  }
  
  /**
   * Get hover manager
   */
  getHoverManager(): HoverManager {
    return this.hoverManager;
  }
  
  /**
   * Get transform controls
   */
  getTransformControls(): TransformControls | null {
    return this.transformControls;
  }
  
  /**
   * Apply snapping to a point
   */
  applySnapping(point: THREE.Vector3): SnapInfo {
    if (!this.config.snapping.enabled) {
      return { snapped: false, point, snapType: 'none' };
    }
    
    const snappedPoint = point.clone();
    let snapType: SnapInfo['snapType'] = 'none';
    let snapped = false;
    
    // Grid snapping
    if (this.config.snapping.gridSize > 0) {
      const gridSize = this.config.snapping.gridSize;
      snappedPoint.x = Math.round(snappedPoint.x / gridSize) * gridSize;
      snappedPoint.y = Math.round(snappedPoint.y / gridSize) * gridSize;
      snappedPoint.z = Math.round(snappedPoint.z / gridSize) * gridSize;
      
      if (!snappedPoint.equals(point)) {
        snapped = true;
        snapType = 'grid';
      }
    }
    
    // Object snapping
    if (this.config.snapping.snapToObjects) {
      // TODO: Implement object vertex/edge/face snapping
    }
    
    return { snapped, point: snappedPoint, snapType };
  }
  
  /**
   * Update camera reference
   */
  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
    
    if (this.transformControls) {
      this.transformControls.camera = camera;
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
   * Dispose interaction system
   */
  dispose(): void {
    this.removeEventListeners();
    
    if (this.transformControls) {
      this.scene.remove(this.transformControls);
      this.transformControls.dispose();
      this.transformControls = null;
    }
    
    if (this.selectionBoxHelper) {
      this.scene.remove(this.selectionBoxHelper);
      this.selectionBoxHelper.geometry.dispose();
      (this.selectionBoxHelper.material as THREE.Material).dispose();
      this.selectionBoxHelper = null;
    }
    
    this.hoverManager.clear();
    this.selectionManager.clearSelection();
  }
}

export default InteractionSystem;
