/**
 * Workspace Provider Component
 *
 * Central provider that manages workspace state using the entity system
 * and provides actions to all child components.
 *
 * Integrates:
 * - Entity-based reactive state
 * - Panel registration and lifecycle
 * - Pane management and splitting
 * - Item (tab) management
 * - Focus management
 * - Layout persistence
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 */

import {
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  WorkspaceContext,
  WorkspaceContextValue,
  WorkspaceSnapshot,
  WorkspaceActions,
  Pane,
  PaneGroup,
  Dock,
  OpenItemOptions,
} from "./WorkspaceContext";
import { GlobalStore, globalStore } from "@/state/Store";
import { Entity } from "@/state/Entity";
import type {
  Panel,
  PanelState,
  createDefaultPanelState,
} from "@/interfaces/Panel";
import type { Item, ItemState } from "@/interfaces/Item";
import type { EntityId, DockPosition, Action } from "@/interfaces/types";

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_DOCK_SIZES: Record<DockPosition, number> = {
  left: 260,
  right: 300,
  bottom: 200,
};

const MIN_DOCK_SIZES: Record<DockPosition, number> = {
  left: 170,
  right: 170,
  bottom: 100,
};

const MAX_DOCK_SIZES: Record<DockPosition, number> = {
  left: 600,
  right: 600,
  bottom: 500,
};

// ============================================================================
// ID Generation
// ============================================================================

let idCounter = 0;

function generateId(prefix: string): EntityId {
  return `${prefix}-${Date.now()}-${++idCounter}` as EntityId;
}

// ============================================================================
// Default State Factories
// ============================================================================

function createDefaultPaneState(): PanelState {
  return {
    isVisible: true,
    position: "left",
    size: 260,
    isFocused: false,
    isCollapsed: false,
    isMaximized: false,
    order: 0,
    zIndex: 0,
  };
}

function createDefaultItemState(): ItemState {
  return {
    isDirty: false,
    isPreview: false,
    isPinned: false,
    navigationHistory: [],
    historyIndex: -1,
    lastSavedVersion: 0,
    lastModified: new Date(),
  };
}

function createDefaultPane(id?: EntityId): Pane {
  return {
    id: id ?? generateId("pane"),
    items: [],
    activeItemId: null,
    parent: null,
    flexGrow: 1,
  };
}

function createDefaultPaneGroup(axis: "horizontal" | "vertical"): PaneGroup {
  return {
    id: generateId("pane-group"),
    axis,
    children: [],
    parent: null,
    flexes: [],
  };
}

function createDefaultDock(position: DockPosition): Dock {
  return {
    position,
    panels: [],
    activePanelId: null,
    isVisible: position === "left", // Left dock visible by default
    size: DEFAULT_DOCK_SIZES[position],
    isCollapsed: false,
  };
}

// ============================================================================
// Workspace Provider
// ============================================================================

export interface WorkspaceProviderProps {
  children: ReactNode;
  /** Custom global store instance (for testing) */
  store?: GlobalStore;
  /** Initial layout configuration */
  initialLayout?: {
    leftDockVisible?: boolean;
    rightDockVisible?: boolean;
    bottomDockVisible?: boolean;
    leftDockSize?: number;
    rightDockSize?: number;
    bottomDockSize?: number;
  };
  /** Persist layout to storage */
  persistLayout?: boolean;
  /** Storage key for layout persistence */
  storageKey?: string;
}

export function WorkspaceProvider({
  children,
  store = globalStore,
  initialLayout,
  persistLayout = true,
  storageKey = "neurectomy-workspace-layout",
}: WorkspaceProviderProps) {
  // === Internal State ===
  const [version, setVersion] = useState(0);
  const forceUpdate = useCallback(() => setVersion((v) => v + 1), []);

  // === Entity Maps ===
  const panesRef = useRef<Map<EntityId, Entity<Pane>>>(new Map());
  const paneGroupsRef = useRef<Map<EntityId, Entity<PaneGroup>>>(new Map());
  const docksRef = useRef<Map<DockPosition, Entity<Dock>>>(new Map());
  const panelsRef = useRef<Map<EntityId, Panel>>(new Map());
  const itemsRef = useRef<Map<EntityId, Item>>(new Map());
  const panelStatesRef = useRef<Map<EntityId, Entity<PanelState>>>(new Map());
  const itemStatesRef = useRef<Map<EntityId, Entity<ItemState>>>(new Map());

  // === Layout State ===
  const [rootPaneGroupId, setRootPaneGroupId] = useState<EntityId | null>(null);
  const [activePaneId, setActivePaneId] = useState<EntityId | null>(null);

  // === Initialize Default Layout ===
  useEffect(() => {
    // Initialize docks
    const positions: DockPosition[] = ["left", "right", "bottom"];
    positions.forEach((position) => {
      if (!docksRef.current.has(position)) {
        const dock = createDefaultDock(position);
        if (initialLayout) {
          if (
            position === "left" &&
            initialLayout.leftDockVisible !== undefined
          ) {
            dock.isVisible = initialLayout.leftDockVisible;
          }
          if (
            position === "right" &&
            initialLayout.rightDockVisible !== undefined
          ) {
            dock.isVisible = initialLayout.rightDockVisible;
          }
          if (
            position === "bottom" &&
            initialLayout.bottomDockVisible !== undefined
          ) {
            dock.isVisible = initialLayout.bottomDockVisible;
          }
          if (position === "left" && initialLayout.leftDockSize !== undefined) {
            dock.size = initialLayout.leftDockSize;
          }
          if (
            position === "right" &&
            initialLayout.rightDockSize !== undefined
          ) {
            dock.size = initialLayout.rightDockSize;
          }
          if (
            position === "bottom" &&
            initialLayout.bottomDockSize !== undefined
          ) {
            dock.size = initialLayout.bottomDockSize;
          }
        }
        const entity = new Entity(dock, generateId(`dock-${position}`));
        entity.subscribe(() => forceUpdate());
        docksRef.current.set(position, entity);
      }
    });

    // Initialize root pane group with one pane if not exists
    if (!rootPaneGroupId) {
      const pane = createDefaultPane();
      const paneEntity = new Entity(pane, pane.id);
      paneEntity.subscribe(() => forceUpdate());
      panesRef.current.set(pane.id, paneEntity);

      const group = createDefaultPaneGroup("horizontal");
      group.children = [pane.id];
      group.flexes = [1];
      const groupEntity = new Entity(group, group.id);
      groupEntity.subscribe(() => forceUpdate());
      paneGroupsRef.current.set(group.id, groupEntity);

      setRootPaneGroupId(group.id);
      setActivePaneId(pane.id);
    }

    // Load persisted layout
    if (persistLayout) {
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved) {
          const layout = JSON.parse(saved);
          // Apply saved dock sizes
          positions.forEach((position) => {
            const dock = docksRef.current.get(position);
            if (dock && layout.docks?.[position]) {
              dock.update({
                size: layout.docks[position].size,
                isVisible: layout.docks[position].isVisible,
                isCollapsed: layout.docks[position].isCollapsed,
              });
            }
          });
        }
      } catch (e) {
        console.warn("Failed to load workspace layout:", e);
      }
    }
  }, []);

  // === Build Snapshot ===
  const snapshot = useMemo<WorkspaceSnapshot>(() => {
    const panes = new Map<EntityId, Pane>();
    panesRef.current.forEach((entity, id) => panes.set(id, entity.current));

    const paneGroups = new Map<EntityId, PaneGroup>();
    paneGroupsRef.current.forEach((entity, id) =>
      paneGroups.set(id, entity.current)
    );

    const docks = new Map<DockPosition, Dock>();
    docksRef.current.forEach((entity, position) =>
      docks.set(position, entity.current)
    );

    const panelStates = new Map<EntityId, PanelState>();
    panelStatesRef.current.forEach((entity, id) =>
      panelStates.set(id, entity.current)
    );

    const itemStates = new Map<EntityId, ItemState>();
    itemStatesRef.current.forEach((entity, id) =>
      itemStates.set(id, entity.current)
    );

    return {
      panes,
      paneGroups,
      rootPaneGroupId,
      activePaneId,
      docks,
      panels: new Map(panelsRef.current),
      items: new Map(itemsRef.current),
      panelStates,
      itemStates,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, rootPaneGroupId, activePaneId]);

  // === Actions ===
  const actions = useMemo<WorkspaceActions>(() => {
    // === Pane Operations ===
    const createPane = (): EntityId => {
      const pane = createDefaultPane();
      const entity = new Entity(pane, pane.id);
      entity.subscribe(() => forceUpdate());
      panesRef.current.set(pane.id, entity);
      forceUpdate();
      return pane.id;
    };

    const splitPane = (
      paneId: EntityId,
      direction: "horizontal" | "vertical"
    ): EntityId => {
      const paneEntity = panesRef.current.get(paneId);
      if (!paneEntity) {
        throw new Error(`Pane not found: ${paneId}`);
      }

      // Create new pane
      const newPane = createDefaultPane();
      const newPaneEntity = new Entity(newPane, newPane.id);
      newPaneEntity.subscribe(() => forceUpdate());
      panesRef.current.set(newPane.id, newPaneEntity);

      // Find parent group
      const pane = paneEntity.current;
      let parentGroupId = pane.parent;

      if (parentGroupId) {
        const parentGroup = paneGroupsRef.current.get(parentGroupId);
        if (parentGroup && parentGroup.current.axis === direction) {
          // Same direction - add to existing group
          const idx = parentGroup.current.children.indexOf(paneId);
          const newChildren = [...parentGroup.current.children];
          newChildren.splice(idx + 1, 0, newPane.id);
          const newFlexes = [...parentGroup.current.flexes];
          newFlexes.splice(idx + 1, 0, parentGroup.current.flexes[idx] / 2);
          newFlexes[idx] = newFlexes[idx] / 2;
          parentGroup.update({ children: newChildren, flexes: newFlexes });
          newPaneEntity.update({ parent: parentGroupId });
        } else {
          // Different direction - create new group
          const newGroup = createDefaultPaneGroup(direction);
          newGroup.children = [paneId, newPane.id];
          newGroup.flexes = [0.5, 0.5];
          newGroup.parent = parentGroupId;

          const groupEntity = new Entity(newGroup, newGroup.id);
          groupEntity.subscribe(() => forceUpdate());
          paneGroupsRef.current.set(newGroup.id, groupEntity);

          // Update parent to replace pane with group
          if (parentGroup) {
            const idx = parentGroup.current.children.indexOf(paneId);
            const newChildren = [...parentGroup.current.children];
            newChildren[idx] = newGroup.id;
            parentGroup.update({ children: newChildren });
          }

          paneEntity.update({ parent: newGroup.id });
          newPaneEntity.update({ parent: newGroup.id });
        }
      } else {
        // This is root - wrap in new group
        const newGroup = createDefaultPaneGroup(direction);
        newGroup.children = [paneId, newPane.id];
        newGroup.flexes = [0.5, 0.5];

        const groupEntity = new Entity(newGroup, newGroup.id);
        groupEntity.subscribe(() => forceUpdate());
        paneGroupsRef.current.set(newGroup.id, groupEntity);

        paneEntity.update({ parent: newGroup.id });
        newPaneEntity.update({ parent: newGroup.id });

        // Update root
        const rootGroup = rootPaneGroupId
          ? paneGroupsRef.current.get(rootPaneGroupId)
          : null;
        if (rootGroup) {
          const idx = rootGroup.current.children.indexOf(paneId);
          if (idx >= 0) {
            const newChildren = [...rootGroup.current.children];
            newChildren[idx] = newGroup.id;
            rootGroup.update({ children: newChildren });
            groupEntity.update({ parent: rootPaneGroupId });
          }
        }
      }

      forceUpdate();
      return newPane.id;
    };

    const closePane = (paneId: EntityId): void => {
      const paneEntity = panesRef.current.get(paneId);
      if (!paneEntity) return;

      const pane = paneEntity.current;

      // Close all items in pane
      pane.items.forEach((itemId) => {
        const item = itemsRef.current.get(itemId);
        if (item) {
          item.deactivate?.();
          itemsRef.current.delete(itemId);
          itemStatesRef.current.delete(itemId);
        }
      });

      // Remove from parent group
      if (pane.parent) {
        const parentGroup = paneGroupsRef.current.get(pane.parent);
        if (parentGroup) {
          const idx = parentGroup.current.children.indexOf(paneId);
          if (idx >= 0) {
            const newChildren = [...parentGroup.current.children];
            newChildren.splice(idx, 1);
            const newFlexes = [...parentGroup.current.flexes];
            newFlexes.splice(idx, 1);
            parentGroup.update({ children: newChildren, flexes: newFlexes });

            // If only one child left, flatten
            if (newChildren.length === 1) {
              const childId = newChildren[0];
              const grandparent = parentGroup.current.parent;
              if (grandparent) {
                const grandparentGroup = paneGroupsRef.current.get(grandparent);
                if (grandparentGroup) {
                  const gpIdx = grandparentGroup.current.children.indexOf(
                    parentGroup.current.id
                  );
                  if (gpIdx >= 0) {
                    const gpChildren = [...grandparentGroup.current.children];
                    gpChildren[gpIdx] = childId;
                    grandparentGroup.update({ children: gpChildren });

                    // Update child's parent
                    const childPane = panesRef.current.get(childId as EntityId);
                    const childGroup = paneGroupsRef.current.get(
                      childId as EntityId
                    );
                    if (childPane) childPane.update({ parent: grandparent });
                    if (childGroup) childGroup.update({ parent: grandparent });

                    paneGroupsRef.current.delete(parentGroup.current.id);
                  }
                }
              }
            }
          }
        }
      }

      panesRef.current.delete(paneId);

      // Update active pane if needed
      if (activePaneId === paneId) {
        const firstPane = panesRef.current.keys().next().value;
        setActivePaneId(firstPane ?? null);
      }

      forceUpdate();
    };

    const activatePane = (paneId: EntityId): void => {
      if (panesRef.current.has(paneId)) {
        setActivePaneId(paneId);
        forceUpdate();
      }
    };

    // === Item Operations ===
    const openItem = (
      item: Item,
      paneId?: EntityId,
      options: OpenItemOptions = {}
    ): void => {
      const targetPaneId = paneId ?? activePaneId;
      if (!targetPaneId) {
        console.warn("No pane available to open item");
        return;
      }

      const paneEntity = panesRef.current.get(targetPaneId);
      if (!paneEntity) {
        console.warn(`Pane not found: ${targetPaneId}`);
        return;
      }

      // Check if item already exists
      const existingItemId = Array.from(itemsRef.current.entries()).find(
        ([_, i]) => i.id() === item.id()
      )?.[0];

      if (existingItemId) {
        // Item already open - activate it
        const currentPane = Array.from(panesRef.current.entries()).find(
          ([_, p]) => p.current.items.includes(existingItemId)
        );
        if (currentPane) {
          activateItem(existingItemId, currentPane[0]);
          return;
        }
      }

      // Handle preview mode
      if (options.preview) {
        // Find and replace existing preview tab
        const pane = paneEntity.current;
        const previewItemId = pane.items.find((id) => {
          const state = itemStatesRef.current.get(id);
          return state?.current.isPreview;
        });
        if (previewItemId) {
          closeItem(previewItemId, targetPaneId);
        }
      }

      // Register item
      const itemId = item.id();
      itemsRef.current.set(itemId, item);

      // Create item state
      const itemState = createDefaultItemState();
      itemState.isPreview = options.preview ?? false;
      itemState.isPinned = options.pinned ?? false;
      const stateEntity = new Entity(itemState, itemId);
      stateEntity.subscribe(() => forceUpdate());
      itemStatesRef.current.set(itemId, stateEntity);

      // Add to pane
      const pane = paneEntity.current;
      const newItems = [...pane.items];
      const insertIndex = options.index ?? newItems.length;
      newItems.splice(insertIndex, 0, itemId);
      paneEntity.update({ items: newItems });

      // Activate item
      item.activate?.();
      if (options.focus !== false) {
        paneEntity.update({ activeItemId: itemId });
        setActivePaneId(targetPaneId);
      }

      forceUpdate();
    };

    const closeItem = (itemId: EntityId, paneId: EntityId): void => {
      const paneEntity = panesRef.current.get(paneId);
      if (!paneEntity) return;

      const pane = paneEntity.current;
      const idx = pane.items.indexOf(itemId);
      if (idx < 0) return;

      // Get item and check if dirty
      const item = itemsRef.current.get(itemId);
      const itemState = itemStatesRef.current.get(itemId);

      if (item && itemState?.current.isDirty) {
        // TODO: Show save dialog
        console.warn("Closing dirty item - unsaved changes will be lost");
      }

      // Deactivate and clean up
      item?.deactivate?.();
      itemsRef.current.delete(itemId);
      itemStatesRef.current.delete(itemId);

      // Remove from pane
      const newItems = [...pane.items];
      newItems.splice(idx, 1);

      // Update active item if needed
      let newActiveItemId = pane.activeItemId;
      if (pane.activeItemId === itemId) {
        // Activate adjacent tab
        if (newItems.length > 0) {
          newActiveItemId = newItems[Math.min(idx, newItems.length - 1)];
        } else {
          newActiveItemId = null;
        }
      }

      paneEntity.update({ items: newItems, activeItemId: newActiveItemId });
      forceUpdate();
    };

    const activateItem = (itemId: EntityId, paneId: EntityId): void => {
      const paneEntity = panesRef.current.get(paneId);
      if (!paneEntity) return;

      const item = itemsRef.current.get(itemId);
      if (!item) return;

      // Convert preview to permanent on interaction
      const itemState = itemStatesRef.current.get(itemId);
      if (itemState?.current.isPreview) {
        itemState.update({ isPreview: false });
      }

      item.activate?.();
      paneEntity.update({ activeItemId: itemId });
      setActivePaneId(paneId);
      forceUpdate();
    };

    const moveItem = (
      itemId: EntityId,
      fromPaneId: EntityId,
      toPaneId: EntityId,
      index?: number
    ): void => {
      const fromPane = panesRef.current.get(fromPaneId);
      const toPane = panesRef.current.get(toPaneId);
      if (!fromPane || !toPane) return;

      const fromItems = [...fromPane.current.items];
      const fromIdx = fromItems.indexOf(itemId);
      if (fromIdx < 0) return;

      // Remove from source
      fromItems.splice(fromIdx, 1);
      fromPane.update({ items: fromItems });

      // Add to destination
      const toItems = [...toPane.current.items];
      const insertIdx = index ?? toItems.length;
      toItems.splice(insertIdx, 0, itemId);
      toPane.update({ items: toItems, activeItemId: itemId });

      setActivePaneId(toPaneId);
      forceUpdate();
    };

    const toggleItemPin = (itemId: EntityId, _paneId: EntityId): void => {
      const itemState = itemStatesRef.current.get(itemId);
      if (itemState) {
        itemState.update({ isPinned: !itemState.current.isPinned });
        forceUpdate();
      }
    };

    const saveItem = async (itemId: EntityId): Promise<void> => {
      const item = itemsRef.current.get(itemId);
      if (!item) return;

      try {
        await item.save?.();
        const itemState = itemStatesRef.current.get(itemId);
        if (itemState) {
          itemState.update({ isDirty: false, lastModified: new Date() });
        }
        forceUpdate();
      } catch (e) {
        console.error("Failed to save item:", e);
        throw e;
      }
    };

    const saveAllItems = async (): Promise<void> => {
      const dirtyItems = Array.from(itemStatesRef.current.entries())
        .filter(([_, state]) => state.current.isDirty)
        .map(([id]) => id);

      await Promise.all(dirtyItems.map((id) => saveItem(id)));
    };

    // === Panel Operations ===
    const registerPanel = (panel: Panel): void => {
      const panelId = panel.id();
      panelsRef.current.set(panelId, panel);

      // Create panel state
      const state = createDefaultPaneState();
      state.position = panel.position();
      state.size = panel.size?.() ?? DEFAULT_DOCK_SIZES[panel.position()];
      const stateEntity = new Entity(state, panelId);
      stateEntity.subscribe(() => forceUpdate());
      panelStatesRef.current.set(panelId, stateEntity);

      // Add to dock
      const dock = docksRef.current.get(panel.position());
      if (dock) {
        const panels = [...dock.current.panels, panelId];
        dock.update({ panels });

        // Auto-activate if first panel
        if (!dock.current.activePanelId) {
          dock.update({ activePanelId: panelId });
        }
      }

      forceUpdate();
    };

    const unregisterPanel = (panelId: EntityId): void => {
      const panel = panelsRef.current.get(panelId);
      if (!panel) return;

      // Remove from dock
      const dock = docksRef.current.get(panel.position());
      if (dock) {
        const panels = dock.current.panels.filter((id) => id !== panelId);
        const newActiveId =
          dock.current.activePanelId === panelId
            ? (panels[0] ?? null)
            : dock.current.activePanelId;
        dock.update({ panels, activePanelId: newActiveId });
      }

      panelsRef.current.delete(panelId);
      panelStatesRef.current.delete(panelId);
      forceUpdate();
    };

    const togglePanel = (panelId: EntityId): void => {
      const panelState = panelStatesRef.current.get(panelId);
      const panel = panelsRef.current.get(panelId);
      if (!panelState || !panel) return;

      panelState.update({ isVisible: !panelState.current.isVisible });

      // Also toggle dock visibility
      const dock = docksRef.current.get(panel.position());
      if (dock) {
        if (panelState.current.isVisible) {
          // Was visible, now hidden - check if any panels still visible
          const anyVisible = dock.current.panels.some((id) => {
            const state = panelStatesRef.current.get(id);
            return id !== panelId && state?.current.isVisible;
          });
          if (!anyVisible) {
            dock.update({ isVisible: false });
          }
        } else {
          // Now visible - ensure dock is visible
          dock.update({ isVisible: true, activePanelId: panelId });
        }
      }

      forceUpdate();
    };

    const activatePanel = (panelId: EntityId): void => {
      const panel = panelsRef.current.get(panelId);
      if (!panel) return;

      const dock = docksRef.current.get(panel.position());
      if (dock) {
        dock.update({ activePanelId: panelId, isVisible: true });
      }

      const panelState = panelStatesRef.current.get(panelId);
      if (panelState) {
        panelState.update({ isVisible: true });
      }

      forceUpdate();
    };

    const movePanel = (panelId: EntityId, newPosition: DockPosition): void => {
      const panel = panelsRef.current.get(panelId);
      if (!panel) return;

      const oldDock = docksRef.current.get(panel.position());
      const newDock = docksRef.current.get(newPosition);

      if (oldDock) {
        const panels = oldDock.current.panels.filter((id) => id !== panelId);
        const newActiveId =
          oldDock.current.activePanelId === panelId
            ? (panels[0] ?? null)
            : oldDock.current.activePanelId;
        oldDock.update({ panels, activePanelId: newActiveId });
      }

      if (newDock) {
        const panels = [...newDock.current.panels, panelId];
        newDock.update({ panels, activePanelId: panelId });
      }

      const panelState = panelStatesRef.current.get(panelId);
      if (panelState) {
        panelState.update({ position: newPosition });
      }

      forceUpdate();
    };

    const focusPanel = (panelId: EntityId): void => {
      // Unfocus all other panels
      panelStatesRef.current.forEach((state, id) => {
        if (id !== panelId && state.current.isFocused) {
          state.update({ isFocused: false });
        }
      });

      const panelState = panelStatesRef.current.get(panelId);
      if (panelState) {
        panelState.update({ isFocused: true });
      }

      forceUpdate();
    };

    // === Dock Operations ===
    const toggleDock = (position: DockPosition): void => {
      const dock = docksRef.current.get(position);
      if (dock) {
        dock.update({ isVisible: !dock.current.isVisible });
        forceUpdate();
      }
    };

    const resizeDock = (position: DockPosition, size: number): void => {
      const dock = docksRef.current.get(position);
      if (dock) {
        const clampedSize = Math.max(
          MIN_DOCK_SIZES[position],
          Math.min(MAX_DOCK_SIZES[position], size)
        );
        dock.update({ size: clampedSize });
        forceUpdate();
      }
    };

    const collapseDock = (position: DockPosition, collapsed: boolean): void => {
      const dock = docksRef.current.get(position);
      if (dock) {
        dock.update({ isCollapsed: collapsed });
        forceUpdate();
      }
    };

    // === Layout Operations ===
    const resetLayout = (): void => {
      // Clear everything and reinitialize
      panesRef.current.clear();
      paneGroupsRef.current.clear();
      itemsRef.current.clear();
      itemStatesRef.current.clear();

      // Reset docks to defaults
      docksRef.current.forEach((dock, position) => {
        dock.update({
          ...createDefaultDock(position),
          panels: dock.current.panels,
          activePanelId: dock.current.activePanelId,
        });
      });

      // Create fresh pane
      const pane = createDefaultPane();
      const paneEntity = new Entity(pane, pane.id);
      paneEntity.subscribe(() => forceUpdate());
      panesRef.current.set(pane.id, paneEntity);

      const group = createDefaultPaneGroup("horizontal");
      group.children = [pane.id];
      group.flexes = [1];
      const groupEntity = new Entity(group, group.id);
      groupEntity.subscribe(() => forceUpdate());
      paneGroupsRef.current.set(group.id, groupEntity);

      setRootPaneGroupId(group.id);
      setActivePaneId(pane.id);

      // Clear saved layout
      if (persistLayout) {
        localStorage.removeItem(storageKey);
      }

      forceUpdate();
    };

    const saveLayout = async (): Promise<void> => {
      if (!persistLayout) return;

      const layout: Record<string, unknown> = {
        docks: {} as Record<DockPosition, unknown>,
      };

      docksRef.current.forEach((dock, position) => {
        (layout.docks as Record<DockPosition, unknown>)[position] = {
          size: dock.current.size,
          isVisible: dock.current.isVisible,
          isCollapsed: dock.current.isCollapsed,
        };
      });

      localStorage.setItem(storageKey, JSON.stringify(layout));
    };

    const loadLayout = async (_layoutId: string): Promise<void> => {
      // For now, just load from localStorage
      // Future: Load from backend by layoutId
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const layout = JSON.parse(saved);
        docksRef.current.forEach((dock, position) => {
          if (layout.docks?.[position]) {
            dock.update(layout.docks[position]);
          }
        });
        forceUpdate();
      }
    };

    // === Focus Operations ===
    const cyclePaneFocus = (direction: "next" | "previous"): void => {
      const paneIds = Array.from(panesRef.current.keys());
      if (paneIds.length === 0) return;

      const currentIndex = activePaneId ? paneIds.indexOf(activePaneId) : -1;
      let nextIndex: number;

      if (direction === "next") {
        nextIndex = (currentIndex + 1) % paneIds.length;
      } else {
        nextIndex = (currentIndex - 1 + paneIds.length) % paneIds.length;
      }

      setActivePaneId(paneIds[nextIndex]);
      forceUpdate();
    };

    const focusCenter = (): void => {
      if (activePaneId) {
        const pane = panesRef.current.get(activePaneId);
        if (pane && pane.current.activeItemId) {
          const item = itemsRef.current.get(pane.current.activeItemId);
          item?.focus?.();
        }
      }
    };

    const focusDock = (position: DockPosition): void => {
      const dock = docksRef.current.get(position);
      if (dock && dock.current.activePanelId) {
        focusPanel(dock.current.activePanelId);
      }
    };

    return {
      createPane,
      splitPane,
      closePane,
      activatePane,
      openItem,
      closeItem,
      activateItem,
      moveItem,
      toggleItemPin,
      saveItem,
      saveAllItems,
      registerPanel,
      unregisterPanel,
      togglePanel,
      activatePanel,
      movePanel,
      focusPanel,
      toggleDock,
      resizeDock,
      collapseDock,
      resetLayout,
      saveLayout,
      loadLayout,
      cyclePaneFocus,
      focusCenter,
      focusDock,
    };
  }, [activePaneId, rootPaneGroupId, forceUpdate, persistLayout, storageKey]);

  // === Action Dispatcher ===
  const dispatch = useCallback(
    (action: Action): void => {
      action.run();
      forceUpdate();
    },
    [forceUpdate]
  );

  // === Auto-save layout on changes ===
  useEffect(() => {
    if (persistLayout) {
      const timeout = setTimeout(() => {
        actions.saveLayout();
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [version, persistLayout, actions]);

  // === Context Value ===
  const contextValue = useMemo<WorkspaceContextValue>(
    () => ({
      store,
      snapshot,
      actions,
      dispatch,
    }),
    [store, snapshot, actions, dispatch]
  );

  return (
    <WorkspaceContext.Provider value={contextValue}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export default WorkspaceProvider;
