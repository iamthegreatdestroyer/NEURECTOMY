/**
 * Keyboard Navigation System
 *
 * Comprehensive keyboard navigation for 3D scenes with
 * spatial navigation, focus management, and roving tabindex.
 *
 * @module @neurectomy/3d-engine/accessibility/keyboard-navigation
 * @agents @CANVAS @APEX
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  AccessibleElementId,
  KeyBinding,
  NavigationMode,
  KeyboardNavigationConfig,
  FocusTrapConfig,
} from './types';

// ============================================================================
// Default Key Bindings
// ============================================================================

/**
 * Default key bindings for 3D navigation
 */
export const DEFAULT_KEY_BINDINGS: KeyBinding[] = [
  // Navigation
  { key: 'ArrowUp', action: 'Navigate up', handler: 'navigateUp', repeatable: true },
  { key: 'ArrowDown', action: 'Navigate down', handler: 'navigateDown', repeatable: true },
  { key: 'ArrowLeft', action: 'Navigate left', handler: 'navigateLeft', repeatable: true },
  { key: 'ArrowRight', action: 'Navigate right', handler: 'navigateRight', repeatable: true },
  { key: 'ArrowUp', modifiers: { shift: true }, action: 'Navigate forward (Z)', handler: 'navigateForward', repeatable: true },
  { key: 'ArrowDown', modifiers: { shift: true }, action: 'Navigate backward (Z)', handler: 'navigateBackward', repeatable: true },
  
  // Selection
  { key: 'Enter', action: 'Select element', handler: 'selectElement' },
  { key: ' ', action: 'Toggle selection', handler: 'toggleSelection' },
  { key: 'Escape', action: 'Clear selection', handler: 'clearSelection' },
  { key: 'a', modifiers: { ctrl: true }, action: 'Select all', handler: 'selectAll' },
  
  // Camera
  { key: 'w', action: 'Pan up', handler: 'panUp', repeatable: true, context: ['viewport'] },
  { key: 's', action: 'Pan down', handler: 'panDown', repeatable: true, context: ['viewport'] },
  { key: 'a', action: 'Pan left', handler: 'panLeft', repeatable: true, context: ['viewport'] },
  { key: 'd', action: 'Pan right', handler: 'panRight', repeatable: true, context: ['viewport'] },
  { key: '+', action: 'Zoom in', handler: 'zoomIn', repeatable: true },
  { key: '=', action: 'Zoom in', handler: 'zoomIn', repeatable: true },
  { key: '-', action: 'Zoom out', handler: 'zoomOut', repeatable: true },
  { key: '0', action: 'Reset view', handler: 'resetView' },
  
  // Focus navigation
  { key: 'Tab', action: 'Next element', handler: 'focusNext' },
  { key: 'Tab', modifiers: { shift: true }, action: 'Previous element', handler: 'focusPrevious' },
  { key: 'Home', action: 'First element', handler: 'focusFirst' },
  { key: 'End', action: 'Last element', handler: 'focusLast' },
  
  // Tree navigation (for graph/hierarchy)
  { key: 'ArrowRight', modifiers: { ctrl: true }, action: 'Expand node', handler: 'expandNode', context: ['graph'] },
  { key: 'ArrowLeft', modifiers: { ctrl: true }, action: 'Collapse node', handler: 'collapseNode', context: ['graph'] },
  
  // Timeline navigation
  { key: 'ArrowLeft', action: 'Previous time point', handler: 'previousTimePoint', context: ['timeline'] },
  { key: 'ArrowRight', action: 'Next time point', handler: 'nextTimePoint', context: ['timeline'] },
  { key: 'Home', action: 'Start of timeline', handler: 'timelineStart', context: ['timeline'] },
  { key: 'End', action: 'End of timeline', handler: 'timelineEnd', context: ['timeline'] },
  { key: ' ', action: 'Play/pause timeline', handler: 'togglePlayback', context: ['timeline'] },
  
  // Quick actions
  { key: 'f', action: 'Focus on selection', handler: 'focusOnSelection' },
  { key: '/', action: 'Open search', handler: 'openSearch' },
  { key: '?', action: 'Show keyboard shortcuts', handler: 'showShortcuts' },
  { key: 'i', action: 'Show element info', handler: 'showInfo' },
];

/**
 * Default navigation configuration
 */
const DEFAULT_CONFIG: KeyboardNavigationConfig = {
  mode: 'spatial',
  bindings: DEFAULT_KEY_BINDINGS,
  rovingTabindex: true,
  skipHidden: true,
  skipDisabled: true,
  announceOnFocus: true,
  focusIndicator: 'outline',
  focusIndicatorColor: '#4F46E5',
  focusIndicatorSize: 3,
};

// ============================================================================
// Types
// ============================================================================

interface FocusableElement {
  id: AccessibleElementId;
  position: { x: number; y: number; z: number };
  bounds?: { width: number; height: number; depth: number };
  tabIndex: number;
  hidden: boolean;
  disabled: boolean;
  element?: HTMLElement;
}

type KeyHandler = (event: KeyboardEvent, context: NavigationContext) => boolean;

interface NavigationContext {
  mode: NavigationMode;
  focusedElementId: AccessibleElementId | null;
  selectedElementIds: Set<AccessibleElementId>;
  elements: Map<AccessibleElementId, FocusableElement>;
  viewport?: { width: number; height: number };
}

// ============================================================================
// Keyboard Navigation Manager
// ============================================================================

/**
 * Keyboard Navigation Manager
 *
 * Handles all keyboard interactions for the 3D scene
 * including spatial navigation, focus management, and shortcuts.
 */
export class KeyboardNavigationManager {
  private config: KeyboardNavigationConfig;
  private elements: Map<AccessibleElementId, FocusableElement> = new Map();
  private focusedElementId: AccessibleElementId | null = null;
  private selectedElementIds: Set<AccessibleElementId> = new Set();
  private handlers: Map<string, KeyHandler> = new Map();
  private activeContexts: Set<string> = new Set(['viewport']);
  private focusTrap: FocusTrapConfig | null = null;
  private listeners: Set<(event: NavigationEvent) => void> = new Set();
  private keydownHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(config: Partial<KeyboardNavigationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.registerDefaultHandlers();
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  /**
   * Initialize keyboard event listeners
   */
  initialize(container: HTMLElement): void {
    this.keydownHandler = (event: KeyboardEvent) => {
      this.handleKeyDown(event);
    };

    container.addEventListener('keydown', this.keydownHandler);
    
    // Make container focusable
    if (!container.hasAttribute('tabindex')) {
      container.setAttribute('tabindex', '0');
    }
  }

  /**
   * Register default key handlers
   */
  private registerDefaultHandlers(): void {
    // Navigation handlers
    this.handlers.set('navigateUp', (event, ctx) => {
      this.navigateSpatial('up', ctx);
      return true;
    });
    
    this.handlers.set('navigateDown', (event, ctx) => {
      this.navigateSpatial('down', ctx);
      return true;
    });
    
    this.handlers.set('navigateLeft', (event, ctx) => {
      this.navigateSpatial('left', ctx);
      return true;
    });
    
    this.handlers.set('navigateRight', (event, ctx) => {
      this.navigateSpatial('right', ctx);
      return true;
    });
    
    this.handlers.set('navigateForward', (event, ctx) => {
      this.navigateSpatial('forward', ctx);
      return true;
    });
    
    this.handlers.set('navigateBackward', (event, ctx) => {
      this.navigateSpatial('backward', ctx);
      return true;
    });

    // Selection handlers
    this.handlers.set('selectElement', () => {
      if (this.focusedElementId) {
        this.selectElement(this.focusedElementId);
      }
      return true;
    });
    
    this.handlers.set('toggleSelection', () => {
      if (this.focusedElementId) {
        this.toggleSelection(this.focusedElementId);
      }
      return true;
    });
    
    this.handlers.set('clearSelection', () => {
      this.clearSelection();
      return true;
    });
    
    this.handlers.set('selectAll', () => {
      this.selectAll();
      return true;
    });

    // Focus handlers
    this.handlers.set('focusNext', () => {
      this.focusNext();
      return true;
    });
    
    this.handlers.set('focusPrevious', () => {
      this.focusPrevious();
      return true;
    });
    
    this.handlers.set('focusFirst', () => {
      this.focusFirst();
      return true;
    });
    
    this.handlers.set('focusLast', () => {
      this.focusLast();
      return true;
    });

    // Placeholder handlers for camera/view actions
    this.handlers.set('panUp', () => { this.emit('camera-pan', { direction: 'up' }); return true; });
    this.handlers.set('panDown', () => { this.emit('camera-pan', { direction: 'down' }); return true; });
    this.handlers.set('panLeft', () => { this.emit('camera-pan', { direction: 'left' }); return true; });
    this.handlers.set('panRight', () => { this.emit('camera-pan', { direction: 'right' }); return true; });
    this.handlers.set('zoomIn', () => { this.emit('camera-zoom', { direction: 'in' }); return true; });
    this.handlers.set('zoomOut', () => { this.emit('camera-zoom', { direction: 'out' }); return true; });
    this.handlers.set('resetView', () => { this.emit('camera-reset', {}); return true; });
    
    // Graph handlers
    this.handlers.set('expandNode', () => { this.emit('node-expand', { id: this.focusedElementId }); return true; });
    this.handlers.set('collapseNode', () => { this.emit('node-collapse', { id: this.focusedElementId }); return true; });
    
    // Timeline handlers
    this.handlers.set('previousTimePoint', () => { this.emit('timeline-step', { direction: 'backward' }); return true; });
    this.handlers.set('nextTimePoint', () => { this.emit('timeline-step', { direction: 'forward' }); return true; });
    this.handlers.set('timelineStart', () => { this.emit('timeline-jump', { position: 'start' }); return true; });
    this.handlers.set('timelineEnd', () => { this.emit('timeline-jump', { position: 'end' }); return true; });
    this.handlers.set('togglePlayback', () => { this.emit('timeline-toggle', {}); return true; });
    
    // Action handlers
    this.handlers.set('focusOnSelection', () => { this.emit('focus-selection', {}); return true; });
    this.handlers.set('openSearch', () => { this.emit('open-search', {}); return true; });
    this.handlers.set('showShortcuts', () => { this.emit('show-shortcuts', {}); return true; });
    this.handlers.set('showInfo', () => { this.emit('show-info', { id: this.focusedElementId }); return true; });
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  /**
   * Handle keydown events
   */
  private handleKeyDown(event: KeyboardEvent): void {
    // Find matching binding
    const binding = this.findMatchingBinding(event);
    if (!binding) return;

    // Check context
    if (binding.context && !binding.context.some((ctx) => this.activeContexts.has(ctx))) {
      return;
    }

    // Get handler
    const handler = this.handlers.get(binding.handler);
    if (!handler) return;

    // Build context
    const context: NavigationContext = {
      mode: this.config.mode,
      focusedElementId: this.focusedElementId,
      selectedElementIds: this.selectedElementIds,
      elements: this.elements,
    };

    // Execute handler
    const handled = handler(event, context);
    
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  }

  /**
   * Find matching key binding
   */
  private findMatchingBinding(event: KeyboardEvent): KeyBinding | null {
    for (const binding of this.config.bindings) {
      if (this.matchesBinding(event, binding)) {
        return binding;
      }
    }
    return null;
  }

  /**
   * Check if event matches binding
   */
  private matchesBinding(event: KeyboardEvent, binding: KeyBinding): boolean {
    if (event.key !== binding.key) return false;

    const modifiers = binding.modifiers || {};
    if (!!modifiers.ctrl !== event.ctrlKey) return false;
    if (!!modifiers.alt !== event.altKey) return false;
    if (!!modifiers.shift !== event.shiftKey) return false;
    if (!!modifiers.meta !== event.metaKey) return false;

    return true;
  }

  // ==========================================================================
  // Element Registration
  // ==========================================================================

  /**
   * Register a focusable element
   */
  registerElement(
    id: AccessibleElementId,
    position: { x: number; y: number; z: number },
    options: {
      bounds?: { width: number; height: number; depth: number };
      tabIndex?: number;
      hidden?: boolean;
      disabled?: boolean;
      element?: HTMLElement;
    } = {}
  ): void {
    const focusable: FocusableElement = {
      id,
      position,
      bounds: options.bounds,
      tabIndex: options.tabIndex ?? 0,
      hidden: options.hidden ?? false,
      disabled: options.disabled ?? false,
      element: options.element,
    };

    this.elements.set(id, focusable);
    this.updateTabIndices();
  }

  /**
   * Unregister an element
   */
  unregisterElement(id: AccessibleElementId): void {
    this.elements.delete(id);
    
    if (this.focusedElementId === id) {
      this.focusedElementId = null;
      this.focusFirst();
    }
    
    this.selectedElementIds.delete(id);
    this.updateTabIndices();
  }

  /**
   * Update element position
   */
  updateElementPosition(
    id: AccessibleElementId,
    position: { x: number; y: number; z: number }
  ): void {
    const element = this.elements.get(id);
    if (element) {
      element.position = position;
    }
  }

  /**
   * Update element state
   */
  updateElementState(
    id: AccessibleElementId,
    state: { hidden?: boolean; disabled?: boolean }
  ): void {
    const element = this.elements.get(id);
    if (element) {
      if (state.hidden !== undefined) element.hidden = state.hidden;
      if (state.disabled !== undefined) element.disabled = state.disabled;
      this.updateTabIndices();
    }
  }

  // ==========================================================================
  // Focus Management
  // ==========================================================================

  /**
   * Get list of focusable elements
   */
  private getFocusableElements(): FocusableElement[] {
    const elements: FocusableElement[] = [];
    
    for (const element of this.elements.values()) {
      if (this.config.skipHidden && element.hidden) continue;
      if (this.config.skipDisabled && element.disabled) continue;
      
      // Check focus trap
      if (this.focusTrap?.enabled) {
        if (this.focusTrap.includedElements && 
            !this.focusTrap.includedElements.includes(element.id)) {
          continue;
        }
        if (this.focusTrap.excludedElements?.includes(element.id)) {
          continue;
        }
      }
      
      elements.push(element);
    }
    
    return elements;
  }

  /**
   * Update tab indices for roving tabindex
   */
  private updateTabIndices(): void {
    if (!this.config.rovingTabindex) return;

    for (const element of this.elements.values()) {
      if (element.element) {
        element.element.tabIndex = element.id === this.focusedElementId ? 0 : -1;
      }
    }
  }

  /**
   * Focus an element
   */
  focusElement(id: AccessibleElementId): void {
    const element = this.elements.get(id);
    if (!element) return;
    
    if (this.config.skipHidden && element.hidden) return;
    if (this.config.skipDisabled && element.disabled) return;

    const previousId = this.focusedElementId;
    this.focusedElementId = id;
    this.updateTabIndices();

    // Focus DOM element if available
    if (element.element) {
      element.element.focus();
    }

    // Emit event
    this.emit('focus-changed', {
      previousId,
      currentId: id,
      element,
    });
  }

  /**
   * Focus next element in tab order
   */
  focusNext(): void {
    const elements = this.getFocusableElements();
    if (elements.length === 0) return;

    const currentIndex = elements.findIndex((e) => e.id === this.focusedElementId);
    let nextIndex = currentIndex + 1;

    // Wrap around
    if (nextIndex >= elements.length) {
      if (this.focusTrap?.wrapFocus !== false) {
        nextIndex = 0;
      } else {
        return;
      }
    }

    const nextElement = elements[nextIndex];
    if (nextElement) {
      this.focusElement(nextElement.id);
    }
  }

  /**
   * Focus previous element in tab order
   */
  focusPrevious(): void {
    const elements = this.getFocusableElements();
    if (elements.length === 0) return;

    const currentIndex = elements.findIndex((e) => e.id === this.focusedElementId);
    let prevIndex = currentIndex - 1;

    // Wrap around
    if (prevIndex < 0) {
      if (this.focusTrap?.wrapFocus !== false) {
        prevIndex = elements.length - 1;
      } else {
        return;
      }
    }

    const prevElement = elements[prevIndex];
    if (prevElement) {
      this.focusElement(prevElement.id);
    }
  }

  /**
   * Focus first element
   */
  focusFirst(): void {
    const elements = this.getFocusableElements();
    const firstElement = elements[0];
    if (firstElement) {
      this.focusElement(firstElement.id);
    }
  }

  /**
   * Focus last element
   */
  focusLast(): void {
    const elements = this.getFocusableElements();
    const lastElement = elements[elements.length - 1];
    if (lastElement) {
      this.focusElement(lastElement.id);
    }
  }

  /**
   * Get currently focused element ID
   */
  getFocusedElementId(): AccessibleElementId | null {
    return this.focusedElementId;
  }

  // ==========================================================================
  // Spatial Navigation
  // ==========================================================================

  /**
   * Navigate spatially in 3D space
   */
  private navigateSpatial(
    direction: 'up' | 'down' | 'left' | 'right' | 'forward' | 'backward',
    context: NavigationContext
  ): void {
    if (!this.focusedElementId) {
      this.focusFirst();
      return;
    }

    const currentElement = this.elements.get(this.focusedElementId);
    if (!currentElement) return;

    const candidates = this.getFocusableElements().filter(
      (e) => e.id !== this.focusedElementId
    );

    const best = this.findBestCandidate(currentElement, candidates, direction);
    if (best) {
      this.focusElement(best.id);
    }
  }

  /**
   * Find best candidate in direction
   */
  private findBestCandidate(
    current: FocusableElement,
    candidates: FocusableElement[],
    direction: 'up' | 'down' | 'left' | 'right' | 'forward' | 'backward'
  ): FocusableElement | null {
    const directionVector = this.getDirectionVector(direction);
    
    let best: FocusableElement | null = null;
    let bestScore = Infinity;

    for (const candidate of candidates) {
      const delta = {
        x: candidate.position.x - current.position.x,
        y: candidate.position.y - current.position.y,
        z: candidate.position.z - current.position.z,
      };

      // Calculate dot product with direction
      const dot =
        delta.x * directionVector.x +
        delta.y * directionVector.y +
        delta.z * directionVector.z;

      // Skip candidates not in direction
      if (dot <= 0) continue;

      // Calculate perpendicular distance
      const distance = Math.sqrt(
        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
      );
      
      // Score: prefer closer elements more aligned with direction
      const alignment = dot / distance;
      const score = distance / (alignment * alignment);

      if (score < bestScore) {
        bestScore = score;
        best = candidate;
      }
    }

    return best;
  }

  /**
   * Get direction vector
   */
  private getDirectionVector(
    direction: 'up' | 'down' | 'left' | 'right' | 'forward' | 'backward'
  ): { x: number; y: number; z: number } {
    switch (direction) {
      case 'up': return { x: 0, y: 1, z: 0 };
      case 'down': return { x: 0, y: -1, z: 0 };
      case 'left': return { x: -1, y: 0, z: 0 };
      case 'right': return { x: 1, y: 0, z: 0 };
      case 'forward': return { x: 0, y: 0, z: 1 };
      case 'backward': return { x: 0, y: 0, z: -1 };
    }
  }

  // ==========================================================================
  // Selection
  // ==========================================================================

  /**
   * Select an element
   */
  selectElement(id: AccessibleElementId): void {
    this.selectedElementIds.clear();
    this.selectedElementIds.add(id);
    this.emit('selection-changed', { selected: [id] });
  }

  /**
   * Toggle selection of an element
   */
  toggleSelection(id: AccessibleElementId): void {
    if (this.selectedElementIds.has(id)) {
      this.selectedElementIds.delete(id);
    } else {
      this.selectedElementIds.add(id);
    }
    this.emit('selection-changed', { selected: Array.from(this.selectedElementIds) });
  }

  /**
   * Clear selection
   */
  clearSelection(): void {
    this.selectedElementIds.clear();
    this.emit('selection-changed', { selected: [] });
  }

  /**
   * Select all elements
   */
  selectAll(): void {
    const elements = this.getFocusableElements();
    this.selectedElementIds.clear();
    for (const element of elements) {
      this.selectedElementIds.add(element.id);
    }
    this.emit('selection-changed', { selected: Array.from(this.selectedElementIds) });
  }

  /**
   * Get selected element IDs
   */
  getSelectedElementIds(): AccessibleElementId[] {
    return Array.from(this.selectedElementIds);
  }

  // ==========================================================================
  // Context Management
  // ==========================================================================

  /**
   * Set active context
   */
  setContext(context: string): void {
    this.activeContexts.add(context);
  }

  /**
   * Remove context
   */
  removeContext(context: string): void {
    this.activeContexts.delete(context);
  }

  /**
   * Set navigation mode
   */
  setNavigationMode(mode: NavigationMode): void {
    this.config.mode = mode;
    this.emit('mode-changed', { mode });
  }

  // ==========================================================================
  // Focus Trap
  // ==========================================================================

  /**
   * Enable focus trap
   */
  enableFocusTrap(config: Partial<FocusTrapConfig> = {}): void {
    this.focusTrap = {
      enabled: true,
      allowEscape: config.allowEscape ?? true,
      wrapFocus: config.wrapFocus ?? true,
      returnFocus: config.returnFocus ?? true,
      ...config,
    };

    if (this.focusTrap.initialFocus) {
      this.focusElement(this.focusTrap.initialFocus);
    }
  }

  /**
   * Disable focus trap
   */
  disableFocusTrap(): void {
    this.focusTrap = null;
  }

  // ==========================================================================
  // Custom Handlers
  // ==========================================================================

  /**
   * Register a custom key handler
   */
  registerHandler(name: string, handler: KeyHandler): void {
    this.handlers.set(name, handler);
  }

  /**
   * Unregister a handler
   */
  unregisterHandler(name: string): void {
    this.handlers.delete(name);
  }

  /**
   * Add custom key binding
   */
  addBinding(binding: KeyBinding): void {
    this.config.bindings.push(binding);
  }

  /**
   * Remove key binding
   */
  removeBinding(handler: string): void {
    this.config.bindings = this.config.bindings.filter((b) => b.handler !== handler);
  }

  // ==========================================================================
  // Events
  // ==========================================================================

  /**
   * Navigation event
   */
  interface NavigationEvent {
    type: string;
    data: Record<string, unknown>;
  }

  /**
   * Subscribe to navigation events
   */
  subscribe(listener: (event: NavigationEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Emit event
   */
  private emit(type: string, data: Record<string, unknown>): void {
    const event = { type, data };
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error('Navigation event listener error:', error);
      }
    }
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Get key bindings
   */
  getBindings(): KeyBinding[] {
    return [...this.config.bindings];
  }

  /**
   * Get focus indicator config
   */
  getFocusIndicatorConfig(): {
    style: string;
    color: string;
    size: number;
  } {
    return {
      style: this.config.focusIndicator,
      color: this.config.focusIndicatorColor || '#4F46E5',
      size: this.config.focusIndicatorSize || 3,
    };
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Dispose of the manager
   */
  dispose(): void {
    this.elements.clear();
    this.handlers.clear();
    this.listeners.clear();
    this.focusedElementId = null;
    this.selectedElementIds.clear();
    this.focusTrap = null;
  }
}

// ============================================================================
// Navigation Event Type (exported separately)
// ============================================================================

export interface NavigationEvent {
  type: string;
  data: Record<string, unknown>;
}

// ============================================================================
// Singleton Instance
// ============================================================================

let navigationInstance: KeyboardNavigationManager | null = null;

/**
 * Get the global KeyboardNavigationManager instance
 */
export function getKeyboardNavigationManager(
  config?: Partial<KeyboardNavigationConfig>
): KeyboardNavigationManager {
  if (!navigationInstance) {
    navigationInstance = new KeyboardNavigationManager(config);
  }
  return navigationInstance;
}

/**
 * Reset the global KeyboardNavigationManager instance
 */
export function resetKeyboardNavigationManager(): void {
  if (navigationInstance) {
    navigationInstance.dispose();
    navigationInstance = null;
  }
}
