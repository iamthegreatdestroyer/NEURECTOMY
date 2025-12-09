/**
 * Workspace Interface
 *
 * Defines the contract for the main workspace container.
 * Based on Zed's Workspace struct adapted for TypeScript/React.
 *
 * The Workspace is the top-level container holding:
 * - Project (file tree, settings)
 * - Panes (editor groups with tabs)
 * - Docks (panel containers)
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

import type { ReactNode } from "react";
import type {
  EntityId,
  DockPosition,
  SplitDirection,
  FocusHandle,
  Disposable,
  Subscription,
  FileEntry,
} from "./types";
import type { Panel, PanelId } from "./Panel";
import type {
  Item,
  ItemId,
  ItemHandle,
  ItemType,
  ItemSerializedState,
} from "./Item";

// ============================================================================
// Workspace Identity
// ============================================================================

/**
 * Workspace ID type
 */
export type WorkspaceId = EntityId<"workspace">;

/**
 * Pane ID type
 */
export type PaneId = EntityId<"pane">;

/**
 * Dock ID type
 */
export type DockId = EntityId<"dock">;

// ============================================================================
// Project Types
// ============================================================================

/**
 * Project configuration
 */
export interface ProjectConfig {
  /** Project root path */
  rootPath: string;

  /** Project name */
  name: string;

  /** Additional workspace folders */
  folders?: string[];

  /** Excluded patterns */
  excludePatterns?: string[];

  /** Whether to watch for file changes */
  watchFiles?: boolean;
}

/**
 * Project interface for workspace file management
 */
export interface Project extends Disposable {
  /** Project configuration */
  readonly config: ProjectConfig;

  /** Get root file entries */
  getRootEntries(): Promise<FileEntry[]>;

  /** Get entries for a directory */
  getEntries(path: string): Promise<FileEntry[]>;

  /** Read file contents */
  readFile(path: string): Promise<string>;

  /** Write file contents */
  writeFile(path: string, content: string): Promise<void>;

  /** Create a file */
  createFile(path: string, content?: string): Promise<void>;

  /** Create a directory */
  createDirectory(path: string): Promise<void>;

  /** Delete a file or directory */
  delete(path: string, recursive?: boolean): Promise<void>;

  /** Rename/move a file or directory */
  rename(oldPath: string, newPath: string): Promise<void>;

  /** Check if path exists */
  exists(path: string): Promise<boolean>;

  /** Get file info */
  getFileInfo(path: string): Promise<FileEntry | null>;

  /** Subscribe to file changes */
  onFileChange(
    callback: (event: { type: string; path: string }) => void
  ): Subscription;
}

// ============================================================================
// Pane Types
// ============================================================================

/**
 * Pane state
 */
export interface PaneState {
  /** Pane ID */
  readonly id: PaneId;

  /** Items in this pane */
  readonly items: readonly ItemHandle[];

  /** Active item ID */
  readonly activeItemId: ItemId | null;

  /** Activation history (most recent first) */
  readonly activationHistory: readonly ItemId[];

  /** Whether pane is focused */
  readonly isFocused: boolean;

  /** Whether pane is zoomed (maximized) */
  readonly isZoomed: boolean;
}

/**
 * A pane is a container for items (tabs).
 *
 * Based on Zed's Pane struct:
 * ```rust
 * pub struct Pane {
 *     items: Vec<Box<dyn ItemHandle>>,
 *     active_item_index: usize,
 *     activation_history: Vec<EntityId<AnyView>>,
 *     focus_handle: FocusHandle,
 *     toolbar: View<Toolbar>,
 *     // ...
 * }
 * ```
 */
export interface Pane extends Disposable {
  /** Pane state */
  readonly state: PaneState;

  /** Focus handle */
  readonly focusHandle: FocusHandle;

  // ─────────────────────────────────────────────────────────────────────────
  // Item Management
  // ─────────────────────────────────────────────────────────────────────────

  /** Add an item to the pane */
  addItem(
    item: Item,
    options?: { index?: number; activate?: boolean }
  ): ItemHandle;

  /** Remove an item from the pane */
  removeItem(itemId: ItemId): Promise<boolean>;

  /** Get item by ID */
  getItem(itemId: ItemId): ItemHandle | undefined;

  /** Get all items */
  getItems(): readonly ItemHandle[];

  /** Get active item */
  getActiveItem(): ItemHandle | undefined;

  /** Set active item */
  setActiveItem(itemId: ItemId): void;

  /** Move item to index */
  moveItem(itemId: ItemId, toIndex: number): void;

  /** Find item by deduplication key */
  findByKey(key: string): ItemHandle | undefined;

  // ─────────────────────────────────────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────────────────────────────────────

  /** Activate next item */
  activateNextItem(): void;

  /** Activate previous item */
  activatePreviousItem(): void;

  /** Activate item from history */
  activatePreviouslyActive(): void;

  /** Go back in activation history */
  goBack(): void;

  /** Go forward in activation history */
  goForward(): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Splitting
  // ─────────────────────────────────────────────────────────────────────────

  /** Split the active item in direction */
  split(direction: SplitDirection): Pane | null;

  /** Whether pane can be split */
  canSplit(): boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Focus
  // ─────────────────────────────────────────────────────────────────────────

  /** Focus this pane */
  focus(): void;

  /** Check if pane has focus */
  hasFocus(): boolean;

  /** Toggle zoom (maximize) */
  toggleZoom(): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────────────────────────────────────

  /** Subscribe to state changes */
  onStateChange(callback: (state: PaneState) => void): Subscription;

  /** Subscribe to item changes */
  onItemChange(callback: (event: PaneItemChangeEvent) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Rendering
  // ─────────────────────────────────────────────────────────────────────────

  /** Render the tab bar */
  renderTabBar(): ReactNode;

  /** Render the active item content */
  renderContent(): ReactNode;

  /** Render the toolbar */
  renderToolbar(): ReactNode;

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /** Serialize pane state */
  serialize(): PaneSerializedState;
}

/**
 * Pane item change event
 */
export interface PaneItemChangeEvent {
  type: "added" | "removed" | "activated" | "moved" | "updated";
  itemId: ItemId;
  index?: number;
}

/**
 * Serialized pane state
 */
export interface PaneSerializedState {
  id: string;
  items: ItemSerializedState[];
  activeItemId: string | null;
}

// ============================================================================
// Pane Group Types
// ============================================================================

/**
 * A node in the pane group tree
 */
export type PaneGroupNode = PaneGroupLeaf | PaneGroupSplit;

/**
 * A leaf node containing a single pane
 */
export interface PaneGroupLeaf {
  type: "leaf";
  paneId: PaneId;
  flexGrow?: number;
}

/**
 * A split node containing multiple children
 */
export interface PaneGroupSplit {
  type: "split";
  direction: SplitDirection;
  children: PaneGroupNode[];
  sizes: number[];
}

/**
 * Pane group manages the recursive split structure.
 *
 * Based on Zed's PaneGroup.
 */
export interface PaneGroup extends Disposable {
  /** Root node of the pane tree */
  readonly root: PaneGroupNode;

  /** All panes in the group */
  readonly panes: Map<PaneId, Pane>;

  /** Active pane ID */
  readonly activePaneId: PaneId | null;

  // ─────────────────────────────────────────────────────────────────────────
  // Pane Management
  // ─────────────────────────────────────────────────────────────────────────

  /** Get pane by ID */
  getPane(paneId: PaneId): Pane | undefined;

  /** Get active pane */
  getActivePane(): Pane | undefined;

  /** Set active pane */
  setActivePane(paneId: PaneId): void;

  /** Create a new pane */
  createPane(): Pane;

  /** Remove a pane */
  removePane(paneId: PaneId): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Splitting
  // ─────────────────────────────────────────────────────────────────────────

  /** Split a pane */
  splitPane(paneId: PaneId, direction: SplitDirection): Pane | null;

  /** Unsplit (close) a pane and merge with sibling */
  unsplitPane(paneId: PaneId): void;

  /** Resize splits */
  resizeSplit(nodeIndex: number, sizes: number[]): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────────────────────────────────────

  /** Focus pane in direction */
  focusPaneInDirection(direction: "up" | "down" | "left" | "right"): void;

  /** Move item to pane in direction */
  moveItemToDirection(direction: "up" | "down" | "left" | "right"): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────────────────────────────────────

  /** Subscribe to pane changes */
  onPaneChange(callback: (event: PaneGroupChangeEvent) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /** Serialize pane group state */
  serialize(): PaneGroupSerializedState;
}

/**
 * Pane group change event
 */
export interface PaneGroupChangeEvent {
  type: "pane-added" | "pane-removed" | "pane-activated" | "layout-changed";
  paneId?: PaneId;
}

/**
 * Serialized pane group state
 */
export interface PaneGroupSerializedState {
  root: PaneGroupNode;
  panes: Record<string, PaneSerializedState>;
  activePaneId: string | null;
}

// ============================================================================
// Dock Types
// ============================================================================

/**
 * Dock state
 */
export interface DockState {
  /** Dock position */
  readonly position: DockPosition;

  /** Panels in this dock */
  readonly panels: readonly Panel[];

  /** Active panel ID */
  readonly activePanelId: PanelId | null;

  /** Whether dock is visible */
  readonly isVisible: boolean;

  /** Dock size (width for left/right, height for bottom) */
  readonly size: number;

  /** Whether dock is collapsed */
  readonly isCollapsed: boolean;
}

/**
 * A dock is a container for panels.
 *
 * Based on Zed's Dock.
 */
export interface Dock extends Disposable {
  /** Dock state */
  readonly state: DockState;

  /** Focus handle */
  readonly focusHandle: FocusHandle;

  // ─────────────────────────────────────────────────────────────────────────
  // Panel Management
  // ─────────────────────────────────────────────────────────────────────────

  /** Add panel to dock */
  addPanel(panel: Panel): void;

  /** Remove panel from dock */
  removePanel(panelId: PanelId): void;

  /** Get panel by ID */
  getPanel(panelId: PanelId): Panel | undefined;

  /** Get all panels */
  getPanels(): readonly Panel[];

  /** Get active panel */
  getActivePanel(): Panel | undefined;

  /** Set active panel */
  setActivePanel(panelId: PanelId): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Visibility
  // ─────────────────────────────────────────────────────────────────────────

  /** Show the dock */
  show(): void;

  /** Hide the dock */
  hide(): void;

  /** Toggle dock visibility */
  toggle(): void;

  /** Set dock size */
  setSize(size: number): void;

  /** Toggle collapse state */
  toggleCollapse(): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────────────────────────────────────

  /** Subscribe to state changes */
  onStateChange(callback: (state: DockState) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /** Serialize dock state */
  serialize(): DockSerializedState;
}

/**
 * Serialized dock state
 */
export interface DockSerializedState {
  position: DockPosition;
  panelIds: string[];
  activePanelId: string | null;
  isVisible: boolean;
  size: number;
  isCollapsed: boolean;
}

// ============================================================================
// Workspace Interface (Core)
// ============================================================================

/**
 * Workspace state
 */
export interface WorkspaceState {
  /** Workspace ID */
  readonly id: WorkspaceId;

  /** Project state */
  readonly project: Project | null;

  /** Pane group */
  readonly paneGroup: PaneGroup;

  /** Docks */
  readonly docks: {
    left: Dock;
    right: Dock;
    bottom: Dock;
  };

  /** Focused element type */
  readonly focusedElement: "pane" | "dock" | "panel" | null;

  /** Focused pane ID */
  readonly focusedPaneId: PaneId | null;

  /** Focused dock position */
  readonly focusedDockPosition: DockPosition | null;
}

/**
 * Core workspace interface - the top-level container.
 *
 * Based on Zed's Workspace:
 * ```rust
 * pub struct Workspace {
 *     project: Model<Project>,
 *     center: PaneGroup,
 *     left_dock: View<Dock>,
 *     right_dock: View<Dock>,
 *     bottom_dock: View<Dock>,
 *     focus_handle: FocusHandle,
 *     // ...
 * }
 * ```
 */
export interface Workspace extends Disposable {
  /** Workspace state */
  readonly state: WorkspaceState;

  /** Focus handle */
  readonly focusHandle: FocusHandle;

  // ─────────────────────────────────────────────────────────────────────────
  // Project Management
  // ─────────────────────────────────────────────────────────────────────────

  /** Open a project */
  openProject(config: ProjectConfig): Promise<void>;

  /** Close the current project */
  closeProject(): Promise<void>;

  /** Get the current project */
  getProject(): Project | null;

  // ─────────────────────────────────────────────────────────────────────────
  // Item Operations
  // ─────────────────────────────────────────────────────────────────────────

  /** Open an item (file, terminal, etc.) */
  openItem(
    type: ItemType,
    options?: {
      path?: string;
      paneId?: PaneId;
      preview?: boolean;
      activate?: boolean;
    }
  ): Promise<ItemHandle | null>;

  /** Open a file in the editor */
  openFile(
    path: string,
    options?: {
      paneId?: PaneId;
      line?: number;
      column?: number;
      preview?: boolean;
    }
  ): Promise<ItemHandle | null>;

  /** Close an item */
  closeItem(itemId: ItemId): Promise<boolean>;

  /** Close all items in a pane */
  closeAllItems(paneId?: PaneId): Promise<boolean>;

  /** Save an item */
  saveItem(itemId: ItemId): Promise<boolean>;

  /** Save all dirty items */
  saveAllItems(): Promise<boolean>;

  // ─────────────────────────────────────────────────────────────────────────
  // Pane Operations
  // ─────────────────────────────────────────────────────────────────────────

  /** Get pane group */
  getPaneGroup(): PaneGroup;

  /** Get active pane */
  getActivePane(): Pane | undefined;

  /** Get pane by ID */
  getPane(paneId: PaneId): Pane | undefined;

  /** Split active pane */
  splitPane(direction: SplitDirection): Pane | null;

  /** Close a pane */
  closePane(paneId: PaneId): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Dock Operations
  // ─────────────────────────────────────────────────────────────────────────

  /** Get dock by position */
  getDock(position: DockPosition): Dock;

  /** Toggle dock visibility */
  toggleDock(position: DockPosition): void;

  /** Toggle panel visibility */
  togglePanel(panelId: PanelId): void;

  /** Activate a panel */
  activatePanel(panelId: PanelId): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Focus Management
  // ─────────────────────────────────────────────────────────────────────────

  /** Focus the workspace */
  focus(): void;

  /** Focus a pane */
  focusPane(paneId: PaneId): void;

  /** Focus a dock */
  focusDock(position: DockPosition): void;

  /** Focus next pane */
  focusNextPane(): void;

  /** Focus previous pane */
  focusPreviousPane(): void;

  /** Toggle zoom on active pane */
  toggleZoom(): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────────────────────────────────────

  /** Subscribe to state changes */
  onStateChange(callback: (state: WorkspaceState) => void): Subscription;

  /** Subscribe to focus changes */
  onFocusChange(
    callback: (focused: { type: string; id?: string }) => void
  ): Subscription;

  /** Subscribe to layout changes */
  onLayoutChange(callback: () => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /** Serialize workspace state */
  serialize(): WorkspaceSerializedState;

  /** Restore workspace from serialized state */
  deserialize(state: WorkspaceSerializedState): Promise<void>;
}

/**
 * Serialized workspace state
 */
export interface WorkspaceSerializedState {
  version: number;
  project: ProjectConfig | null;
  paneGroup: PaneGroupSerializedState;
  docks: {
    left: DockSerializedState;
    right: DockSerializedState;
    bottom: DockSerializedState;
  };
  focusedElement: "pane" | "dock" | "panel" | null;
  focusedPaneId: string | null;
  focusedDockPosition: DockPosition | null;
}

// ============================================================================
// Workspace Factory
// ============================================================================

/**
 * Options for creating a workspace
 */
export interface WorkspaceOptions {
  /** Initial project config */
  project?: ProjectConfig;

  /** Initial state to restore */
  initialState?: WorkspaceSerializedState;
}

/**
 * Factory function for creating workspaces
 */
export type WorkspaceFactory = (options?: WorkspaceOptions) => Workspace;
