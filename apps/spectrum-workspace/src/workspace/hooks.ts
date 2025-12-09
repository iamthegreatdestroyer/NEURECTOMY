/**
 * Workspace Hooks
 *
 * Custom hooks for accessing workspace functionality.
 * Provides ergonomic APIs for common operations.
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 */

import { useCallback, useMemo } from "react";
import {
  useWorkspaceContext,
  usePaneContext,
  useItemContext,
  usePanelContext,
  Pane,
  PaneGroup,
  Dock,
} from "./WorkspaceContext";
import type { Panel, PanelState } from "@/interfaces/Panel";
import type { Item, ItemState } from "@/interfaces/Item";
import type { EntityId, DockPosition } from "@/interfaces/types";

// ============================================================================
// Workspace Hooks
// ============================================================================

/**
 * Get all workspace actions.
 * Primary hook for workspace manipulation.
 */
export function useWorkspaceActions() {
  const { actions } = useWorkspaceContext();
  return actions;
}

/**
 * Get the full workspace snapshot.
 * Use sparingly - prefer more specific hooks.
 */
export function useWorkspaceSnapshot() {
  const { snapshot } = useWorkspaceContext();
  return snapshot;
}

/**
 * Get the active pane ID.
 */
export function useActivePaneId(): EntityId | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.activePaneId;
}

/**
 * Get the active pane data.
 */
export function useActivePane(): Pane | null {
  const { snapshot } = useWorkspaceContext();
  if (!snapshot.activePaneId) return null;
  return snapshot.panes.get(snapshot.activePaneId) ?? null;
}

/**
 * Get a specific pane by ID.
 */
export function usePane(paneId: EntityId): Pane | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.panes.get(paneId) ?? null;
}

/**
 * Get all panes.
 */
export function usePanes(): Map<EntityId, Pane> {
  const { snapshot } = useWorkspaceContext();
  return snapshot.panes;
}

/**
 * Get the root pane group.
 */
export function useRootPaneGroup(): PaneGroup | null {
  const { snapshot } = useWorkspaceContext();
  if (!snapshot.rootPaneGroupId) return null;
  return snapshot.paneGroups.get(snapshot.rootPaneGroupId) ?? null;
}

// ============================================================================
// Dock Hooks
// ============================================================================

/**
 * Get a dock by position.
 */
export function useDock(position: DockPosition): Dock | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.docks.get(position) ?? null;
}

/**
 * Get all docks.
 */
export function useDocks(): Map<DockPosition, Dock> {
  const { snapshot } = useWorkspaceContext();
  return snapshot.docks;
}

/**
 * Get dock visibility states.
 */
export function useDockVisibility(): Record<DockPosition, boolean> {
  const { snapshot } = useWorkspaceContext();
  return {
    left: snapshot.docks.get("left")?.isVisible ?? false,
    right: snapshot.docks.get("right")?.isVisible ?? false,
    bottom: snapshot.docks.get("bottom")?.isVisible ?? false,
  };
}

/**
 * Get dock sizes.
 */
export function useDockSizes(): Record<DockPosition, number> {
  const { snapshot } = useWorkspaceContext();
  return {
    left: snapshot.docks.get("left")?.size ?? 260,
    right: snapshot.docks.get("right")?.size ?? 300,
    bottom: snapshot.docks.get("bottom")?.size ?? 200,
  };
}

/**
 * Hook to control a specific dock.
 */
export function useDockControls(position: DockPosition) {
  const { actions } = useWorkspaceContext();
  const dock = useDock(position);

  const toggle = useCallback(() => {
    actions.toggleDock(position);
  }, [actions, position]);

  const resize = useCallback(
    (size: number) => {
      actions.resizeDock(position, size);
    },
    [actions, position]
  );

  const collapse = useCallback(
    (collapsed: boolean) => {
      actions.collapseDock(position, collapsed);
    },
    [actions, position]
  );

  return {
    dock,
    isVisible: dock?.isVisible ?? false,
    size: dock?.size ?? 260,
    isCollapsed: dock?.isCollapsed ?? false,
    toggle,
    resize,
    collapse,
  };
}

// ============================================================================
// Panel Hooks
// ============================================================================

/**
 * Get a panel by ID.
 */
export function usePanel(panelId: EntityId): Panel | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.panels.get(panelId) ?? null;
}

/**
 * Get all panels.
 */
export function usePanels(): Map<EntityId, Panel> {
  const { snapshot } = useWorkspaceContext();
  return snapshot.panels;
}

/**
 * Get panels for a specific dock.
 */
export function usePanelsInDock(position: DockPosition): Panel[] {
  const { snapshot } = useWorkspaceContext();
  const dock = snapshot.docks.get(position);
  if (!dock) return [];

  return dock.panels
    .map((id) => snapshot.panels.get(id))
    .filter((p): p is Panel => p !== undefined);
}

/**
 * Get the active panel in a dock.
 */
export function useActivePanelInDock(position: DockPosition): Panel | null {
  const { snapshot } = useWorkspaceContext();
  const dock = snapshot.docks.get(position);
  if (!dock?.activePanelId) return null;
  return snapshot.panels.get(dock.activePanelId) ?? null;
}

/**
 * Get panel state by ID.
 */
export function usePanelState(panelId: EntityId): PanelState | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.panelStates.get(panelId) ?? null;
}

/**
 * Hook to control a specific panel.
 */
export function usePanelControls(panelId: EntityId) {
  const { actions } = useWorkspaceContext();
  const panel = usePanel(panelId);
  const state = usePanelState(panelId);

  const toggle = useCallback(() => {
    actions.togglePanel(panelId);
  }, [actions, panelId]);

  const activate = useCallback(() => {
    actions.activatePanel(panelId);
  }, [actions, panelId]);

  const focus = useCallback(() => {
    actions.focusPanel(panelId);
  }, [actions, panelId]);

  const moveTo = useCallback(
    (position: DockPosition) => {
      actions.movePanel(panelId, position);
    },
    [actions, panelId]
  );

  return {
    panel,
    state,
    isVisible: state?.isVisible ?? false,
    isFocused: state?.isFocused ?? false,
    toggle,
    activate,
    focus,
    moveTo,
  };
}

// ============================================================================
// Item Hooks
// ============================================================================

/**
 * Get an item by ID.
 */
export function useItem(itemId: EntityId): Item | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.items.get(itemId) ?? null;
}

/**
 * Get all items.
 */
export function useItems(): Map<EntityId, Item> {
  const { snapshot } = useWorkspaceContext();
  return snapshot.items;
}

/**
 * Get items in a specific pane.
 */
export function useItemsInPane(paneId: EntityId): Item[] {
  const { snapshot } = useWorkspaceContext();
  const pane = snapshot.panes.get(paneId);
  if (!pane) return [];

  return pane.items
    .map((id) => snapshot.items.get(id))
    .filter((i): i is Item => i !== undefined);
}

/**
 * Get the active item in a pane.
 */
export function useActiveItemInPane(paneId: EntityId): Item | null {
  const { snapshot } = useWorkspaceContext();
  const pane = snapshot.panes.get(paneId);
  if (!pane?.activeItemId) return null;
  return snapshot.items.get(pane.activeItemId) ?? null;
}

/**
 * Get item state by ID.
 */
export function useItemState(itemId: EntityId): ItemState | null {
  const { snapshot } = useWorkspaceContext();
  return snapshot.itemStates.get(itemId) ?? null;
}

/**
 * Hook to control a specific item.
 */
export function useItemControls(itemId: EntityId, paneId: EntityId) {
  const { actions } = useWorkspaceContext();
  const item = useItem(itemId);
  const state = useItemState(itemId);

  const close = useCallback(() => {
    actions.closeItem(itemId, paneId);
  }, [actions, itemId, paneId]);

  const activate = useCallback(() => {
    actions.activateItem(itemId, paneId);
  }, [actions, itemId, paneId]);

  const togglePin = useCallback(() => {
    actions.toggleItemPin(itemId, paneId);
  }, [actions, itemId, paneId]);

  const save = useCallback(async () => {
    await actions.saveItem(itemId);
  }, [actions, itemId]);

  const moveTo = useCallback(
    (toPaneId: EntityId, index?: number) => {
      actions.moveItem(itemId, paneId, toPaneId, index);
    },
    [actions, itemId, paneId]
  );

  return {
    item,
    state,
    isDirty: state?.isDirty ?? false,
    isPreview: state?.isPreview ?? false,
    isPinned: state?.isPinned ?? false,
    close,
    activate,
    togglePin,
    save,
    moveTo,
  };
}

// ============================================================================
// Dirty Items Hooks
// ============================================================================

/**
 * Get all dirty items.
 */
export function useDirtyItems(): Item[] {
  const { snapshot } = useWorkspaceContext();
  const dirtyItems: Item[] = [];

  snapshot.itemStates.forEach((state, id) => {
    if (state.isDirty) {
      const item = snapshot.items.get(id);
      if (item) dirtyItems.push(item);
    }
  });

  return dirtyItems;
}

/**
 * Check if any items are dirty.
 */
export function useHasDirtyItems(): boolean {
  const { snapshot } = useWorkspaceContext();

  for (const [_, state] of snapshot.itemStates) {
    if (state.isDirty) return true;
  }

  return false;
}

// ============================================================================
// Layout Hooks
// ============================================================================

/**
 * Get layout actions.
 */
export function useLayoutActions() {
  const { actions } = useWorkspaceContext();

  return useMemo(
    () => ({
      reset: actions.resetLayout,
      save: actions.saveLayout,
      load: actions.loadLayout,
    }),
    [actions]
  );
}

// ============================================================================
// Focus Hooks
// ============================================================================

/**
 * Get focus actions.
 */
export function useFocusActions() {
  const { actions } = useWorkspaceContext();

  return useMemo(
    () => ({
      cyclePaneFocus: actions.cyclePaneFocus,
      focusCenter: actions.focusCenter,
      focusDock: actions.focusDock,
    }),
    [actions]
  );
}

// ============================================================================
// Context-Based Hooks (for use within providers)
// ============================================================================

/**
 * Get current pane context.
 * Must be used within PaneProvider.
 */
export function useCurrentPane() {
  const context = usePaneContext();
  return context;
}

/**
 * Get current item context.
 * Must be used within ItemProvider.
 */
export function useCurrentItem() {
  const context = useItemContext();
  return context;
}

/**
 * Get current panel context.
 * Must be used within PanelProvider.
 */
export function useCurrentPanel() {
  const context = usePanelContext();
  return context;
}

// ============================================================================
// Compound Hooks
// ============================================================================

/**
 * Get all workspace state needed to render the shell.
 * Optimized to minimize re-renders.
 */
export function useShellState() {
  const { snapshot } = useWorkspaceContext();

  return useMemo(
    () => ({
      // Dock visibility
      leftDockVisible: snapshot.docks.get("left")?.isVisible ?? false,
      rightDockVisible: snapshot.docks.get("right")?.isVisible ?? false,
      bottomDockVisible: snapshot.docks.get("bottom")?.isVisible ?? false,

      // Dock sizes
      leftDockSize: snapshot.docks.get("left")?.size ?? 260,
      rightDockSize: snapshot.docks.get("right")?.size ?? 300,
      bottomDockSize: snapshot.docks.get("bottom")?.size ?? 200,

      // Dock collapsed states
      leftDockCollapsed: snapshot.docks.get("left")?.isCollapsed ?? false,
      rightDockCollapsed: snapshot.docks.get("right")?.isCollapsed ?? false,
      bottomDockCollapsed: snapshot.docks.get("bottom")?.isCollapsed ?? false,

      // Active panels
      leftActivePanel: snapshot.docks.get("left")?.activePanelId ?? null,
      rightActivePanel: snapshot.docks.get("right")?.activePanelId ?? null,
      bottomActivePanel: snapshot.docks.get("bottom")?.activePanelId ?? null,

      // Pane info
      activePaneId: snapshot.activePaneId,
      rootPaneGroupId: snapshot.rootPaneGroupId,
    }),
    [snapshot]
  );
}

/**
 * Get all data needed to render a tab bar.
 */
export function useTabBarState(paneId: EntityId) {
  const { snapshot } = useWorkspaceContext();
  const pane = snapshot.panes.get(paneId);

  return useMemo(() => {
    if (!pane) return { tabs: [], activeTabId: null };

    const tabs = pane.items.map((itemId) => {
      const item = snapshot.items.get(itemId);
      const state = snapshot.itemStates.get(itemId);

      return {
        id: itemId,
        title: item?.tabContent().title ?? "Untitled",
        icon: item?.tabContent().icon,
        tooltip: item?.tabContent().tooltip,
        isDirty: state?.isDirty ?? false,
        isPreview: state?.isPreview ?? false,
        isPinned: state?.isPinned ?? false,
      };
    });

    return {
      tabs,
      activeTabId: pane.activeItemId,
    };
  }, [pane, snapshot.items, snapshot.itemStates]);
}

/**
 * Get all data needed to render a dock.
 */
export function useDockState(position: DockPosition) {
  const { snapshot } = useWorkspaceContext();
  const dock = snapshot.docks.get(position);

  return useMemo(() => {
    if (!dock)
      return {
        isVisible: false,
        size: 260,
        isCollapsed: false,
        panels: [],
        activePanelId: null,
      };

    const panels = dock.panels.map((panelId) => {
      const panel = snapshot.panels.get(panelId);
      const state = snapshot.panelStates.get(panelId);

      return {
        id: panelId,
        name: panel?.persistentName() ?? "Unknown",
        icon: panel?.icon(),
        isActive: panelId === dock.activePanelId,
        isVisible: state?.isVisible ?? true,
        isFocused: state?.isFocused ?? false,
      };
    });

    return {
      isVisible: dock.isVisible,
      size: dock.size,
      isCollapsed: dock.isCollapsed,
      panels,
      activePanelId: dock.activePanelId,
    };
  }, [dock, snapshot.panels, snapshot.panelStates]);
}
