/**
 * Panel Interface
 *
 * Defines the contract for dockable panels in the workspace.
 * Based on Zed's Panel trait adapted for TypeScript/React.
 *
 * Panels are persistent UI elements that can be docked to the
 * left, right, or bottom of the workspace.
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import type {
  EntityId,
  DockPosition,
  FocusHandle,
  Action,
  SizeConstraints,
  Disposable,
  Subscription,
} from "./types";

// ============================================================================
// Panel Identity
// ============================================================================

/**
 * Panel ID type for type-safe panel identification
 */
export type PanelId = EntityId<"panel">;

/**
 * Built-in panel identifiers
 */
export const PanelIds = {
  PROJECT_PANEL: "panel:project" as PanelId,
  SEARCH_PANEL: "panel:search" as PanelId,
  SOURCE_CONTROL_PANEL: "panel:source-control" as PanelId,
  EXTENSIONS_PANEL: "panel:extensions" as PanelId,
  AI_PANEL: "panel:ai" as PanelId,
  TERMINAL_PANEL: "panel:terminal" as PanelId,
  PROBLEMS_PANEL: "panel:problems" as PanelId,
  OUTPUT_PANEL: "panel:output" as PanelId,
  DEBUG_CONSOLE_PANEL: "panel:debug-console" as PanelId,
  OUTLINE_PANEL: "panel:outline" as PanelId,
  TIMELINE_PANEL: "panel:timeline" as PanelId,
} as const;

// ============================================================================
// Panel Configuration
// ============================================================================

/**
 * Configuration for panel registration
 */
export interface PanelConfig {
  /** Unique panel identifier */
  id: PanelId;

  /** Human-readable panel name */
  name: string;

  /** Icon to display in activity bar */
  icon: LucideIcon;

  /** Default dock position */
  defaultPosition: DockPosition;

  /** Default visibility state */
  defaultVisible?: boolean;

  /** Default size (width for left/right, height for bottom) */
  defaultSize?: number;

  /** Size constraints */
  sizeConstraints?: SizeConstraints;

  /** Order in activity bar (lower = higher) */
  order?: number;

  /** Activity bar section (top or bottom) */
  activityBarPosition?: "top" | "bottom";

  /** Badge provider for notification counts */
  badge?: () => number | string | undefined;

  /** Toggle action for keyboard shortcuts */
  toggleAction?: Action;

  /** Whether panel can be moved to different docks */
  canMove?: boolean;

  /** Whether panel can be hidden */
  canHide?: boolean;
}

// ============================================================================
// Panel State
// ============================================================================

/**
 * Observable panel state
 */
export interface PanelState {
  /** Current visibility */
  readonly isVisible: boolean;

  /** Current dock position */
  readonly position: DockPosition;

  /** Current size */
  readonly size: number;

  /** Whether panel is focused */
  readonly isFocused: boolean;

  /** Whether panel is collapsed (minimized but visible) */
  readonly isCollapsed: boolean;

  /** Whether panel is maximized */
  readonly isMaximized: boolean;

  /** Current badge value */
  readonly badge?: number | string;
}

/**
 * Panel state updates
 */
export interface PanelStateUpdate {
  isVisible?: boolean;
  position?: DockPosition;
  size?: number;
  isFocused?: boolean;
  isCollapsed?: boolean;
  isMaximized?: boolean;
  badge?: number | string;
}

// ============================================================================
// Panel Interface (Core)
// ============================================================================

/**
 * Core panel interface - the primary contract for all panels.
 *
 * This is analogous to Zed's Panel trait:
 * ```rust
 * pub trait Panel: FocusableView {
 *     fn persistent_name() -> &'static str;
 *     fn position(&self, cx: &WindowContext) -> DockPosition;
 *     fn position_is_valid(&self, position: DockPosition) -> bool;
 *     fn set_position(&mut self, position: DockPosition, cx: &mut ViewContext<Self>);
 *     fn size(&self, cx: &WindowContext) -> Pixels;
 *     fn set_size(&mut self, size: Option<Pixels>, cx: &mut ViewContext<Self>);
 *     fn icon(&self, cx: &WindowContext) -> Option<IconName>;
 *     fn icon_tooltip(&self, cx: &WindowContext) -> Option<&'static str>;
 *     fn toggle_action(&self) -> Box<dyn Action>;
 * }
 * ```
 */
export interface Panel extends Disposable {
  // ─────────────────────────────────────────────────────────────────────────
  // Identity
  // ─────────────────────────────────────────────────────────────────────────

  /** Unique panel identifier - used for persistence */
  readonly id: PanelId;

  /** Human-readable name */
  readonly name: string;

  /** Panel configuration */
  readonly config: PanelConfig;

  // ─────────────────────────────────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────────────────────────────────

  /** Current panel state */
  readonly state: PanelState;

  /** Subscribe to state changes */
  onStateChange(callback: (state: PanelState) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Focus (Zed: FocusableView)
  // ─────────────────────────────────────────────────────────────────────────

  /** Focus handle for keyboard navigation */
  readonly focusHandle: FocusHandle;

  /** Request focus to this panel */
  focus(): void;

  /** Check if panel has focus */
  hasFocus(): boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Position & Size (Zed: position, size, set_position, set_size)
  // ─────────────────────────────────────────────────────────────────────────

  /** Get current dock position */
  getPosition(): DockPosition;

  /** Set dock position */
  setPosition(position: DockPosition): void;

  /** Check if position is valid for this panel */
  isPositionValid(position: DockPosition): boolean;

  /** Get current size */
  getSize(): number;

  /** Set size */
  setSize(size: number): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Visibility
  // ─────────────────────────────────────────────────────────────────────────

  /** Show the panel */
  show(): void;

  /** Hide the panel */
  hide(): void;

  /** Toggle visibility */
  toggle(): void;

  /** Check visibility */
  isVisible(): boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Rendering
  // ─────────────────────────────────────────────────────────────────────────

  /** Render panel header (optional) */
  renderHeader?(): ReactNode;

  /** Render panel content */
  renderContent(): ReactNode;

  /** Render panel toolbar actions */
  renderToolbar?(): ReactNode;

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  /** Called when panel is first activated */
  onActivate?(): void;

  /** Called when panel is deactivated */
  onDeactivate?(): void;

  /** Called when panel visibility changes */
  onVisibilityChange?(visible: boolean): void;

  /** Called when panel position changes */
  onPositionChange?(position: DockPosition): void;

  /** Called when panel is resized */
  onResize?(size: number): void;
}

// ============================================================================
// Panel Factory
// ============================================================================

/**
 * Factory function for creating panel instances
 */
export type PanelFactory = (config: PanelConfig) => Panel;

/**
 * Panel registration entry
 */
export interface PanelRegistration {
  config: PanelConfig;
  factory: PanelFactory;
}

// ============================================================================
// Panel Registry Interface
// ============================================================================

/**
 * Central registry for all panels in the workspace
 */
export interface PanelRegistry {
  /** Register a new panel type */
  register(registration: PanelRegistration): Disposable;

  /** Unregister a panel type */
  unregister(id: PanelId): void;

  /** Get panel registration */
  get(id: PanelId): PanelRegistration | undefined;

  /** Get all registered panels */
  getAll(): readonly PanelRegistration[];

  /** Check if panel is registered */
  has(id: PanelId): boolean;

  /** Create a panel instance */
  create(id: PanelId): Panel | undefined;

  /** Subscribe to registration changes */
  onRegistrationChange(
    callback: (registrations: readonly PanelRegistration[]) => void
  ): Subscription;
}

// ============================================================================
// Panel Manager Interface
// ============================================================================

/**
 * Manages active panel instances in the workspace
 */
export interface PanelManager {
  /** Get panel instance by ID */
  getPanel(id: PanelId): Panel | undefined;

  /** Get all active panels */
  getAllPanels(): readonly Panel[];

  /** Get panels by position */
  getPanelsByPosition(position: DockPosition): readonly Panel[];

  /** Get visible panels */
  getVisiblePanels(): readonly Panel[];

  /** Get focused panel */
  getFocusedPanel(): Panel | undefined;

  /** Activate a panel (create if needed, show, and focus) */
  activatePanel(id: PanelId): Panel | undefined;

  /** Toggle panel visibility */
  togglePanel(id: PanelId): void;

  /** Focus next panel in current dock */
  focusNextPanel(): void;

  /** Focus previous panel in current dock */
  focusPreviousPanel(): void;

  /** Move panel to different dock */
  movePanel(id: PanelId, position: DockPosition): void;

  /** Subscribe to panel changes */
  onPanelChange(callback: (panels: readonly Panel[]) => void): Subscription;

  /** Subscribe to focus changes */
  onFocusChange(callback: (panel: Panel | undefined) => void): Subscription;
}

// ============================================================================
// Base Panel Implementation Helper
// ============================================================================

/**
 * Abstract base class for implementing panels.
 * Provides common functionality and state management.
 */
export abstract class BasePanel implements Panel {
  readonly id: PanelId;
  readonly name: string;
  readonly config: PanelConfig;

  protected _state: PanelState;
  protected stateListeners: Set<(state: PanelState) => void> = new Set();
  protected _focusHandle: FocusHandle | null = null;
  protected disposed = false;

  constructor(config: PanelConfig) {
    this.id = config.id;
    this.name = config.name;
    this.config = config;

    this._state = {
      isVisible: config.defaultVisible ?? false,
      position: config.defaultPosition,
      size: config.defaultSize ?? 300,
      isFocused: false,
      isCollapsed: false,
      isMaximized: false,
    };
  }

  get state(): PanelState {
    return this._state;
  }

  get focusHandle(): FocusHandle {
    if (!this._focusHandle) {
      throw new Error("Focus handle not initialized");
    }
    return this._focusHandle;
  }

  onStateChange(callback: (state: PanelState) => void): Subscription {
    this.stateListeners.add(callback);
    return {
      id: crypto.randomUUID(),
      unsubscribe: () => this.stateListeners.delete(callback),
    };
  }

  protected updateState(update: PanelStateUpdate): void {
    const prev = this._state;
    this._state = { ...this._state, ...update };

    // Notify listeners
    for (const listener of this.stateListeners) {
      listener(this._state);
    }

    // Call lifecycle hooks
    if (update.isVisible !== undefined && update.isVisible !== prev.isVisible) {
      this.onVisibilityChange?.(update.isVisible);
    }
    if (update.position !== undefined && update.position !== prev.position) {
      this.onPositionChange?.(update.position);
    }
    if (update.size !== undefined && update.size !== prev.size) {
      this.onResize?.(update.size);
    }
  }

  // Position & Size
  getPosition(): DockPosition {
    return this._state.position;
  }

  setPosition(position: DockPosition): void {
    if (this.isPositionValid(position)) {
      this.updateState({ position });
    }
  }

  isPositionValid(position: DockPosition): boolean {
    // Override in subclass for position restrictions
    return true;
  }

  getSize(): number {
    return this._state.size;
  }

  setSize(size: number): void {
    const { sizeConstraints } = this.config;
    let constrainedSize = size;

    if (sizeConstraints) {
      const isHorizontal = this._state.position !== "bottom";
      const min = isHorizontal
        ? sizeConstraints.minWidth
        : sizeConstraints.minHeight;
      const max = isHorizontal
        ? sizeConstraints.maxWidth
        : sizeConstraints.maxHeight;

      if (min !== undefined) constrainedSize = Math.max(min, constrainedSize);
      if (max !== undefined) constrainedSize = Math.min(max, constrainedSize);
    }

    this.updateState({ size: constrainedSize });
  }

  // Visibility
  show(): void {
    this.updateState({ isVisible: true });
    this.onActivate?.();
  }

  hide(): void {
    this.updateState({ isVisible: false });
    this.onDeactivate?.();
  }

  toggle(): void {
    if (this._state.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  isVisible(): boolean {
    return this._state.isVisible;
  }

  // Focus
  focus(): void {
    this._focusHandle?.focus();
    this.updateState({ isFocused: true });
  }

  hasFocus(): boolean {
    return this._state.isFocused;
  }

  // Abstract methods to implement
  abstract renderContent(): ReactNode;

  // Optional hooks
  onActivate?(): void;
  onDeactivate?(): void;
  onVisibilityChange?(visible: boolean): void;
  onPositionChange?(position: DockPosition): void;
  onResize?(size: number): void;

  // Cleanup
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.stateListeners.clear();
      this.onDeactivate?.();
    }
  }
}
