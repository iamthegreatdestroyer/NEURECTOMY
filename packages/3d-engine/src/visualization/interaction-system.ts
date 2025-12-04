/**
 * @file 3D Interaction System for Agent Visualization
 * @description Handles selection, hover, drag-drop, and manipulation in 3D space
 * @module @neurectomy/3d-engine/visualization
 * @agents @CANVAS @APEX
 */

import { Vector3, Raycaster, Camera, Object3D, Plane, Vector2 } from 'three';
import { EventEmitter } from 'events';
import type { AgentComponent, AgentConnection, Vector3D } from './types';

// ============================================================================
// Types & Interfaces
// ============================================================================

export type InteractionMode = 
  | 'select'      // Click to select
  | 'multiselect' // Shift+click for multiple selection
  | 'pan'         // Middle mouse to pan
  | 'rotate'      // Right mouse to rotate view
  | 'drag'        // Drag selected objects
  | 'connect'     // Draw connections between components
  | 'measure'     // Measure distances
  | 'annotate';   // Add annotations

export interface InteractionState {
  mode: InteractionMode;
  selectedComponentIds: Set<string>;
  selectedConnectionIds: Set<string>;
  hoveredComponentId: string | null;
  hoveredConnectionId: string | null;
  dragTarget: DragTarget | null;
  connectionDraft: ConnectionDraft | null;
  measurementStart: Vector3D | null;
}

export interface DragTarget {
  componentId: string;
  startPosition: Vector3D;
  currentPosition: Vector3D;
  offset: Vector3D;
}

export interface ConnectionDraft {
  sourceId: string;
  sourcePosition: Vector3D;
  currentPosition: Vector3D;
}

export interface InteractionEvent {
  type: string;
  componentId?: string;
  connectionId?: string;
  position?: Vector3D;
  delta?: Vector3D;
  modifiers: {
    shift: boolean;
    ctrl: boolean;
    alt: boolean;
    meta: boolean;
  };
}

export interface SelectionChangeEvent {
  componentIds: string[];
  connectionIds: string[];
  source: 'click' | 'box-select' | 'keyboard' | 'api';
}

export interface DragEvent {
  componentId: string;
  startPosition: Vector3D;
  currentPosition: Vector3D;
  delta: Vector3D;
  phase: 'start' | 'move' | 'end';
}

export interface ConnectionEvent {
  sourceId: string;
  targetId: string;
  position?: Vector3D;
}

// ============================================================================
// Interaction Manager
// ============================================================================

export interface InteractionConfig {
  enableSelection?: boolean;
  enableMultiSelect?: boolean;
  enableDrag?: boolean;
  enableHover?: boolean;
  enableConnection?: boolean;
  enableMeasure?: boolean;
  dragPlane?: 'xz' | 'xy' | 'camera';
  snapToGrid?: boolean;
  gridSize?: number;
  selectionOutlineColor?: string;
  hoverOutlineColor?: string;
}

const DEFAULT_CONFIG: Required<InteractionConfig> = {
  enableSelection: true,
  enableMultiSelect: true,
  enableDrag: true,
  enableHover: true,
  enableConnection: true,
  enableMeasure: true,
  dragPlane: 'xz',
  snapToGrid: false,
  gridSize: 0.5,
  selectionOutlineColor: '#3b82f6',
  hoverOutlineColor: '#60a5fa',
};

/**
 * Manages all 3D interactions for agent visualization
 */
export class InteractionManager extends EventEmitter {
  private config: Required<InteractionConfig>;
  private state: InteractionState;
  private raycaster: Raycaster;
  private camera: Camera | null = null;
  private scene: Object3D | null = null;
  private interactableObjects: Map<string, Object3D> = new Map();
  private dragPlane: Plane;
  private mousePosition: Vector2 = new Vector2();
  private isPointerDown: boolean = false;
  private pointerDownTime: number = 0;

  constructor(config: InteractionConfig = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.raycaster = new Raycaster();
    this.dragPlane = new Plane(new Vector3(0, 1, 0), 0);
    
    this.state = {
      mode: 'select',
      selectedComponentIds: new Set(),
      selectedConnectionIds: new Set(),
      hoveredComponentId: null,
      hoveredConnectionId: null,
      dragTarget: null,
      connectionDraft: null,
      measurementStart: null,
    };
  }

  // --------------------------------------------------------------------------
  // Setup & Configuration
  // --------------------------------------------------------------------------

  /**
   * Initialize with camera and scene
   */
  public initialize(camera: Camera, scene: Object3D): void {
    this.camera = camera;
    this.scene = scene;
    this.updateDragPlane();
  }

  /**
   * Register an object as interactable
   */
  public registerInteractable(id: string, object: Object3D): void {
    this.interactableObjects.set(id, object);
    object.userData.interactableId = id;
  }

  /**
   * Unregister an interactable object
   */
  public unregisterInteractable(id: string): void {
    this.interactableObjects.delete(id);
  }

  /**
   * Set interaction mode
   */
  public setMode(mode: InteractionMode): void {
    if (this.state.mode !== mode) {
      this.cancelCurrentAction();
      this.state.mode = mode;
      this.emit('modeChange', mode);
    }
  }

  /**
   * Get current state
   */
  public getState(): Readonly<InteractionState> {
    return { ...this.state };
  }

  // --------------------------------------------------------------------------
  // Event Handlers
  // --------------------------------------------------------------------------

  /**
   * Handle pointer move event
   */
  public onPointerMove(event: PointerEvent, canvas: HTMLCanvasElement): void {
    const rect = canvas.getBoundingClientRect();
    this.mousePosition.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mousePosition.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    if (this.isPointerDown && this.state.dragTarget) {
      this.handleDragMove(event);
    } else if (this.isPointerDown && this.state.connectionDraft) {
      this.handleConnectionDrag(event);
    } else if (this.config.enableHover) {
      this.handleHover();
    }
  }

  /**
   * Handle pointer down event
   */
  public onPointerDown(event: PointerEvent): void {
    this.isPointerDown = true;
    this.pointerDownTime = Date.now();

    const modifiers = {
      shift: event.shiftKey,
      ctrl: event.ctrlKey,
      alt: event.altKey,
      meta: event.metaKey,
    };

    // Left click
    if (event.button === 0) {
      const hit = this.raycast();
      
      if (hit) {
        if (this.state.mode === 'connect' && this.config.enableConnection) {
          this.startConnection(hit.id, hit.point);
        } else if (this.config.enableDrag && this.state.selectedComponentIds.has(hit.id)) {
          this.startDrag(hit.id, hit.point);
        } else if (this.config.enableSelection) {
          this.handleSelect(hit.id, modifiers);
        }
      } else if (!modifiers.shift && this.config.enableSelection) {
        this.clearSelection('click');
      }
    }
  }

  /**
   * Handle pointer up event
   */
  public onPointerUp(event: PointerEvent): void {
    const clickDuration = Date.now() - this.pointerDownTime;
    const wasClick = clickDuration < 200;

    if (this.state.dragTarget) {
      this.endDrag();
    }

    if (this.state.connectionDraft) {
      const hit = this.raycast();
      if (hit && hit.id !== this.state.connectionDraft.sourceId) {
        this.completeConnection(hit.id);
      } else {
        this.cancelConnection();
      }
    }

    this.isPointerDown = false;
    this.pointerDownTime = 0;
  }

  /**
   * Handle keyboard events
   */
  public onKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'Escape':
        this.cancelCurrentAction();
        break;
      case 'Delete':
      case 'Backspace':
        this.deleteSelected();
        break;
      case 'a':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.selectAll();
        }
        break;
      case 'd':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.duplicateSelected();
        }
        break;
    }
  }

  // --------------------------------------------------------------------------
  // Raycasting
  // --------------------------------------------------------------------------

  private raycast(): { id: string; point: Vector3D; object: Object3D } | null {
    if (!this.camera || !this.scene) return null;

    this.raycaster.setFromCamera(this.mousePosition, this.camera);
    const objects = Array.from(this.interactableObjects.values());
    const intersects = this.raycaster.intersectObjects(objects, true);

    if (intersects.length > 0) {
      const hit = intersects[0];
      let targetObject = hit.object;
      
      // Walk up to find interactable parent
      while (targetObject && !targetObject.userData.interactableId) {
        targetObject = targetObject.parent as Object3D;
      }

      if (targetObject?.userData.interactableId) {
        return {
          id: targetObject.userData.interactableId,
          point: {
            x: hit.point.x,
            y: hit.point.y,
            z: hit.point.z,
          },
          object: targetObject,
        };
      }
    }

    return null;
  }

  // --------------------------------------------------------------------------
  // Selection
  // --------------------------------------------------------------------------

  private handleSelect(
    id: string,
    modifiers: { shift: boolean; ctrl: boolean; alt: boolean; meta: boolean }
  ): void {
    if (modifiers.shift && this.config.enableMultiSelect) {
      // Toggle selection
      if (this.state.selectedComponentIds.has(id)) {
        this.state.selectedComponentIds.delete(id);
      } else {
        this.state.selectedComponentIds.add(id);
      }
    } else {
      // Single select
      this.state.selectedComponentIds.clear();
      this.state.selectedComponentIds.add(id);
    }

    this.emitSelectionChange('click');
  }

  /**
   * Select components by IDs
   */
  public select(ids: string[], source: SelectionChangeEvent['source'] = 'api'): void {
    this.state.selectedComponentIds = new Set(ids);
    this.emitSelectionChange(source);
  }

  /**
   * Clear selection
   */
  public clearSelection(source: SelectionChangeEvent['source'] = 'api'): void {
    if (this.state.selectedComponentIds.size > 0 || this.state.selectedConnectionIds.size > 0) {
      this.state.selectedComponentIds.clear();
      this.state.selectedConnectionIds.clear();
      this.emitSelectionChange(source);
    }
  }

  /**
   * Select all interactable objects
   */
  public selectAll(): void {
    this.state.selectedComponentIds = new Set(this.interactableObjects.keys());
    this.emitSelectionChange('keyboard');
  }

  private emitSelectionChange(source: SelectionChangeEvent['source']): void {
    this.emit('selectionChange', {
      componentIds: Array.from(this.state.selectedComponentIds),
      connectionIds: Array.from(this.state.selectedConnectionIds),
      source,
    } as SelectionChangeEvent);
  }

  // --------------------------------------------------------------------------
  // Hover
  // --------------------------------------------------------------------------

  private handleHover(): void {
    const hit = this.raycast();
    const newHoveredId = hit?.id || null;

    if (newHoveredId !== this.state.hoveredComponentId) {
      const previousHovered = this.state.hoveredComponentId;
      this.state.hoveredComponentId = newHoveredId;

      if (previousHovered) {
        this.emit('hoverEnd', previousHovered);
      }
      if (newHoveredId) {
        this.emit('hoverStart', newHoveredId);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Drag & Drop
  // --------------------------------------------------------------------------

  private startDrag(componentId: string, point: Vector3D): void {
    const object = this.interactableObjects.get(componentId);
    if (!object) return;

    const position = {
      x: object.position.x,
      y: object.position.y,
      z: object.position.z,
    };

    this.state.dragTarget = {
      componentId,
      startPosition: { ...position },
      currentPosition: { ...position },
      offset: {
        x: point.x - position.x,
        y: point.y - position.y,
        z: point.z - position.z,
      },
    };

    this.emit('dragStart', {
      componentId,
      startPosition: position,
      currentPosition: position,
      delta: { x: 0, y: 0, z: 0 },
      phase: 'start',
    } as DragEvent);
  }

  private handleDragMove(event: PointerEvent): void {
    if (!this.state.dragTarget || !this.camera) return;

    // Get point on drag plane
    const planePoint = this.getPointOnDragPlane();
    if (!planePoint) return;

    // Apply offset
    let newPosition: Vector3D = {
      x: planePoint.x - this.state.dragTarget.offset.x,
      y: this.state.dragTarget.startPosition.y, // Keep y constant for XZ plane
      z: planePoint.z - this.state.dragTarget.offset.z,
    };

    // Apply grid snapping
    if (this.config.snapToGrid) {
      newPosition = this.snapToGrid(newPosition);
    }

    const delta: Vector3D = {
      x: newPosition.x - this.state.dragTarget.currentPosition.x,
      y: newPosition.y - this.state.dragTarget.currentPosition.y,
      z: newPosition.z - this.state.dragTarget.currentPosition.z,
    };

    this.state.dragTarget.currentPosition = newPosition;

    // Move all selected components
    this.emit('dragMove', {
      componentId: this.state.dragTarget.componentId,
      startPosition: this.state.dragTarget.startPosition,
      currentPosition: newPosition,
      delta,
      phase: 'move',
    } as DragEvent);
  }

  private endDrag(): void {
    if (!this.state.dragTarget) return;

    this.emit('dragEnd', {
      componentId: this.state.dragTarget.componentId,
      startPosition: this.state.dragTarget.startPosition,
      currentPosition: this.state.dragTarget.currentPosition,
      delta: {
        x: this.state.dragTarget.currentPosition.x - this.state.dragTarget.startPosition.x,
        y: this.state.dragTarget.currentPosition.y - this.state.dragTarget.startPosition.y,
        z: this.state.dragTarget.currentPosition.z - this.state.dragTarget.startPosition.z,
      },
      phase: 'end',
    } as DragEvent);

    this.state.dragTarget = null;
  }

  private getPointOnDragPlane(): Vector3D | null {
    if (!this.camera) return null;

    this.raycaster.setFromCamera(this.mousePosition, this.camera);
    const point = new Vector3();
    
    if (this.raycaster.ray.intersectPlane(this.dragPlane, point)) {
      return { x: point.x, y: point.y, z: point.z };
    }
    
    return null;
  }

  private updateDragPlane(): void {
    if (!this.camera) return;

    switch (this.config.dragPlane) {
      case 'xz':
        this.dragPlane.set(new Vector3(0, 1, 0), 0);
        break;
      case 'xy':
        this.dragPlane.set(new Vector3(0, 0, 1), 0);
        break;
      case 'camera':
        // Normal facing camera
        const cameraDir = new Vector3(0, 0, -1);
        cameraDir.applyQuaternion(this.camera.quaternion);
        this.dragPlane.set(cameraDir, 0);
        break;
    }
  }

  private snapToGrid(position: Vector3D): Vector3D {
    const gridSize = this.config.gridSize;
    return {
      x: Math.round(position.x / gridSize) * gridSize,
      y: Math.round(position.y / gridSize) * gridSize,
      z: Math.round(position.z / gridSize) * gridSize,
    };
  }

  // --------------------------------------------------------------------------
  // Connection Drawing
  // --------------------------------------------------------------------------

  private startConnection(sourceId: string, point: Vector3D): void {
    this.state.connectionDraft = {
      sourceId,
      sourcePosition: point,
      currentPosition: point,
    };

    this.emit('connectionStart', { sourceId, position: point });
  }

  private handleConnectionDrag(event: PointerEvent): void {
    if (!this.state.connectionDraft) return;

    const planePoint = this.getPointOnDragPlane();
    if (planePoint) {
      this.state.connectionDraft.currentPosition = planePoint;
      this.emit('connectionDrag', {
        sourceId: this.state.connectionDraft.sourceId,
        currentPosition: planePoint,
      });
    }
  }

  private completeConnection(targetId: string): void {
    if (!this.state.connectionDraft) return;

    const event: ConnectionEvent = {
      sourceId: this.state.connectionDraft.sourceId,
      targetId,
    };

    this.emit('connectionComplete', event);
    this.state.connectionDraft = null;
  }

  private cancelConnection(): void {
    if (this.state.connectionDraft) {
      this.emit('connectionCancel', {
        sourceId: this.state.connectionDraft.sourceId,
      });
      this.state.connectionDraft = null;
    }
  }

  // --------------------------------------------------------------------------
  // Actions
  // --------------------------------------------------------------------------

  private cancelCurrentAction(): void {
    if (this.state.dragTarget) {
      // Reset to start position
      this.emit('dragCancel', this.state.dragTarget.componentId);
      this.state.dragTarget = null;
    }
    if (this.state.connectionDraft) {
      this.cancelConnection();
    }
    if (this.state.measurementStart) {
      this.state.measurementStart = null;
      this.emit('measureCancel');
    }
    this.clearSelection();
  }

  private deleteSelected(): void {
    if (this.state.selectedComponentIds.size > 0) {
      this.emit('deleteComponents', Array.from(this.state.selectedComponentIds));
    }
    if (this.state.selectedConnectionIds.size > 0) {
      this.emit('deleteConnections', Array.from(this.state.selectedConnectionIds));
    }
  }

  private duplicateSelected(): void {
    if (this.state.selectedComponentIds.size > 0) {
      this.emit('duplicateComponents', Array.from(this.state.selectedComponentIds));
    }
  }

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------

  public dispose(): void {
    this.interactableObjects.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Box Selection Helper
// ============================================================================

export interface BoxSelectionConfig {
  color?: string;
  opacity?: number;
}

export class BoxSelection {
  private startPoint: Vector2 | null = null;
  private endPoint: Vector2 | null = null;
  private isActive: boolean = false;

  constructor(private config: BoxSelectionConfig = {}) {}

  public start(x: number, y: number): void {
    this.startPoint = new Vector2(x, y);
    this.endPoint = new Vector2(x, y);
    this.isActive = true;
  }

  public update(x: number, y: number): void {
    if (this.isActive && this.endPoint) {
      this.endPoint.set(x, y);
    }
  }

  public end(): { left: number; top: number; width: number; height: number } | null {
    if (!this.isActive || !this.startPoint || !this.endPoint) {
      return null;
    }

    const rect = {
      left: Math.min(this.startPoint.x, this.endPoint.x),
      top: Math.min(this.startPoint.y, this.endPoint.y),
      width: Math.abs(this.endPoint.x - this.startPoint.x),
      height: Math.abs(this.endPoint.y - this.startPoint.y),
    };

    this.reset();
    return rect;
  }

  public reset(): void {
    this.startPoint = null;
    this.endPoint = null;
    this.isActive = false;
  }

  public getRect(): { left: number; top: number; width: number; height: number } | null {
    if (!this.isActive || !this.startPoint || !this.endPoint) {
      return null;
    }

    return {
      left: Math.min(this.startPoint.x, this.endPoint.x),
      top: Math.min(this.startPoint.y, this.endPoint.y),
      width: Math.abs(this.endPoint.x - this.startPoint.x),
      height: Math.abs(this.endPoint.y - this.startPoint.y),
    };
  }
}

export default InteractionManager;
