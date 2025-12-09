/**
 * Workspace Context Types
 *
 * Defines the context shape for workspace state management.
 * Bridges the entity system with React components.
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 */

import { createContext, useContext } from "react";
import type { Panel, PanelState } from "@/interfaces/Panel";
import type { Item, ItemState } from "@/interfaces/Item";
import type { EntityId, DockPosition, Action } from "@/interfaces/types";
import type { GlobalStore } from "@/state/Store";

// ============================================================================
// Workspace State Types
// ============================================================================

/**
 * Pane represents a single editor container that holds items (tabs).
 * Panes can be split horizontally or vertically.
 */
export interface Pane {
  /** Unique identifier */
  id: EntityId;
  /** Items (tabs) in this pane */
  items: EntityId[];
  /** Currently active item */
  activeItemId: EntityId | null;
  /** Pane's position in the split tree */
  parent: EntityId | null;
  /** Pane group index if in a split */
  flexGrow: number;
}

/**
 * PaneGroup represents a split container holding multiple panes.
 */
export interface PaneGroup {
  /** Unique identifier */
  id: EntityId;
  /** Split direction */
  axis: "horizontal" | "vertical";
  /** Child panes or pane groups */
  children: EntityId[];
  /** Parent pane group (null if root) */
  parent: EntityId | null;
  /** Flex ratios for children */
  flexes: number[];
}

/**
 * Dock represents a dockable panel area (left, right, bottom).
 */
export interface Dock {
  /** Dock position */
  position: DockPosition;
  /** Registered panels in this dock */
  panels: EntityId[];
  /** Currently active panel */
  activePanelId: EntityId | null;
  /** Whether the dock is visible */
  isVisible: boolean;
  /** Dock size in pixels */
  size: number;
  /** Whether dock is collapsed */
  isCollapsed: boolean;
}

/**
 * Complete workspace state snapshot
 */
export interface WorkspaceSnapshot {
  /** All panes */
  panes: Map<EntityId, Pane>;
  /** All pane groups */
  paneGroups: Map<EntityId, PaneGroup>;
  /** Root pane group ID */
  rootPaneGroupId: EntityId | null;
  /** Active pane ID */
  activePaneId: EntityId | null;
  /** Docks by position */
  docks: Map<DockPosition, Dock>;
  /** All registered panels */
  panels: Map<EntityId, Panel>;
  /** All open items */
  items: Map<EntityId, Item>;
  /** Panel states */
  panelStates: Map<EntityId, PanelState>;
  /** Item states */
  itemStates: Map<EntityId, ItemState>;
}

// ============================================================================
// Workspace Actions
// ============================================================================

export interface WorkspaceActions {
  // === Pane Operations ===
  /** Create a new pane */
  createPane: () => EntityId;
  /** Split an existing pane */
  splitPane: (
    paneId: EntityId,
    direction: "horizontal" | "vertical"
  ) => EntityId;
  /** Close a pane */
  closePane: (paneId: EntityId) => void;
  /** Activate a pane */
  activatePane: (paneId: EntityId) => void;

  // === Item Operations ===
  /** Open an item in a pane */
  openItem: (item: Item, paneId?: EntityId, options?: OpenItemOptions) => void;
  /** Close an item */
  closeItem: (itemId: EntityId, paneId: EntityId) => void;
  /** Activate an item */
  activateItem: (itemId: EntityId, paneId: EntityId) => void;
  /** Move an item between panes */
  moveItem: (
    itemId: EntityId,
    fromPaneId: EntityId,
    toPaneId: EntityId,
    index?: number
  ) => void;
  /** Pin/unpin an item */
  toggleItemPin: (itemId: EntityId, paneId: EntityId) => void;
  /** Save an item */
  saveItem: (itemId: EntityId) => Promise<void>;
  /** Save all dirty items */
  saveAllItems: () => Promise<void>;

  // === Panel Operations ===
  /** Register a panel */
  registerPanel: (panel: Panel) => void;
  /** Unregister a panel */
  unregisterPanel: (panelId: EntityId) => void;
  /** Toggle a panel's visibility */
  togglePanel: (panelId: EntityId) => void;
  /** Activate a panel in its dock */
  activatePanel: (panelId: EntityId) => void;
  /** Move a panel to a different dock */
  movePanel: (panelId: EntityId, position: DockPosition) => void;
  /** Focus a panel */
  focusPanel: (panelId: EntityId) => void;

  // === Dock Operations ===
  /** Toggle a dock's visibility */
  toggleDock: (position: DockPosition) => void;
  /** Resize a dock */
  resizeDock: (position: DockPosition, size: number) => void;
  /** Collapse/expand a dock */
  collapseDock: (position: DockPosition, collapsed: boolean) => void;

  // === Layout Operations ===
  /** Reset layout to default */
  resetLayout: () => void;
  /** Save current layout */
  saveLayout: () => Promise<void>;
  /** Load a saved layout */
  loadLayout: (layoutId: string) => Promise<void>;

  // === Focus Operations ===
  /** Cycle focus between panes */
  cyclePaneFocus: (direction: "next" | "previous") => void;
  /** Focus the center (editor) area */
  focusCenter: () => void;
  /** Focus a specific dock */
  focusDock: (position: DockPosition) => void;
}

export interface OpenItemOptions {
  /** Whether to open as preview (single preview tab, replaced on next open) */
  preview?: boolean;
  /** Whether to pin the item */
  pinned?: boolean;
  /** Whether to focus the item after opening */
  focus?: boolean;
  /** Index to insert at (default: end) */
  index?: number;
  /** Whether to split if pane exists */
  split?: "horizontal" | "vertical";
}

// ============================================================================
// Workspace Context
// ============================================================================

export interface WorkspaceContextValue {
  /** Global store instance */
  store: GlobalStore;
  /** Current workspace snapshot */
  snapshot: WorkspaceSnapshot;
  /** Workspace actions */
  actions: WorkspaceActions;
  /** Action dispatcher */
  dispatch: (action: Action) => void;
}

export const WorkspaceContext = createContext<WorkspaceContextValue | null>(
  null
);

/**
 * Hook to access workspace context.
 * Must be used within WorkspaceProvider.
 */
export function useWorkspaceContext(): WorkspaceContextValue {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error(
      "useWorkspaceContext must be used within a WorkspaceProvider"
    );
  }
  return context;
}

// ============================================================================
// Item Context (for items to access their own state)
// ============================================================================

export interface ItemContextValue {
  /** Item ID */
  itemId: EntityId;
  /** Pane ID containing this item */
  paneId: EntityId;
  /** Whether this item is active */
  isActive: boolean;
  /** Whether this item has focus */
  isFocused: boolean;
  /** Item state */
  state: ItemState;
  /** Update item state */
  updateState: (updates: Partial<ItemState>) => void;
  /** Request close */
  requestClose: () => void;
  /** Mark as dirty */
  markDirty: (dirty: boolean) => void;
}

export const ItemContext = createContext<ItemContextValue | null>(null);

export function useItemContext(): ItemContextValue {
  const context = useContext(ItemContext);
  if (!context) {
    throw new Error("useItemContext must be used within an ItemProvider");
  }
  return context;
}

// ============================================================================
// Panel Context (for panels to access their own state)
// ============================================================================

export interface PanelContextValue {
  /** Panel ID */
  panelId: EntityId;
  /** Dock position */
  position: DockPosition;
  /** Whether this panel is active in its dock */
  isActive: boolean;
  /** Whether this panel has focus */
  isFocused: boolean;
  /** Panel state */
  state: PanelState;
  /** Update panel state */
  updateState: (updates: Partial<PanelState>) => void;
  /** Request focus */
  requestFocus: () => void;
}

export const PanelContext = createContext<PanelContextValue | null>(null);

export function usePanelContext(): PanelContextValue {
  const context = useContext(PanelContext);
  if (!context) {
    throw new Error("usePanelContext must be used within a PanelProvider");
  }
  return context;
}

// ============================================================================
// Pane Context (for pane components)
// ============================================================================

export interface PaneContextValue {
  /** Pane ID */
  paneId: EntityId;
  /** Whether this pane is active */
  isActive: boolean;
  /** Items in this pane */
  items: EntityId[];
  /** Active item ID */
  activeItemId: EntityId | null;
  /** Split the pane */
  split: (direction: "horizontal" | "vertical") => void;
  /** Close the pane */
  close: () => void;
  /** Activate an item */
  activateItem: (itemId: EntityId) => void;
  /** Close an item */
  closeItem: (itemId: EntityId) => void;
}

export const PaneContext = createContext<PaneContextValue | null>(null);

export function usePaneContext(): PaneContextValue {
  const context = useContext(PaneContext);
  if (!context) {
    throw new Error("usePaneContext must be used within a PaneProvider");
  }
  return context;
}
